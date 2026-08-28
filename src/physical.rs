//! Physical outputs represented as explicit compositions of FEEC operators.

use crate::model::DerivedQuantity;
use crate::operator::LinearMap;
use crate::prior::GaussianPrior;
use crate::{FeecGmrfError, Result};
use manifold::{
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
    topology::complex::Complex,
};

/// A named physical pushforward from latent cochains to reported values.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalMap {
    name: String,
    map: LinearMap,
}

/// 3D magnetic pushforwards built from `D1` and Whitney reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticFieldMaps3d {
    pub flux_cochain: LinearMap,
    /// Cell-major Whitney 2-form vector proxy:
    /// `[Bx_cell0, By_cell0, Bz_cell0, Bx_cell1, ...]`.
    /// This differs from the component-major 1-form reconstruction API.
    pub magnetic_field: PhysicalMap,
    pub volume_average_components: [LinearMap; 3],
    pub vector_rms_weights: Vec<f64>,
    pub domain_volume: f64,
}

/// Compactly supported, mesh-independent Wendland C2 sensor kernel in 3D.
///
/// At distance `d` from `center`, the unnormalized weight is
/// `(1-q)^4 (4q+1)` for `q=d/radius < 1`, and zero otherwise.  The magnetic
/// sensor builder normalizes its discrete volume quadrature, so the kernel's
/// absolute scale does not affect the reported average.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WendlandC2Kernel3d {
    center: [f64; 3],
    radius: f64,
}

impl WendlandC2Kernel3d {
    /// Construct a fixed physical kernel.
    pub fn new(center: [f64; 3], radius: f64) -> Result<Self> {
        if center.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "Wendland kernel center coordinates must be finite".to_string(),
            ));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Wendland kernel radius must be finite and positive".to_string(),
            ));
        }
        Ok(Self { center, radius })
    }

    pub fn center(self) -> [f64; 3] {
        self.center
    }

    pub fn radius(self) -> f64 {
        self.radius
    }

    /// Evaluate the unnormalized radial weight at a physical point.
    pub fn evaluate(self, point: [f64; 3]) -> Result<f64> {
        if point.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "Wendland kernel evaluation point must be finite".to_string(),
            ));
        }
        let distance = point
            .iter()
            .zip(self.center)
            .map(|(coordinate, center)| (coordinate - center).powi(2))
            .sum::<f64>()
            .sqrt();
        let q = distance / self.radius;
        Ok(if q < 1.0 {
            (1.0 - q).powi(4) * (4.0 * q + 1.0)
        } else {
            0.0
        })
    }

    /// Analytic integral over R3, useful for quadrature diagnostics when the
    /// support ball lies wholly inside the physical domain.
    pub fn whole_space_integral(self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius.powi(3) / 21.0
    }
}

/// Fixed-footprint magnetic volume sensors using the canonical `A -> D1 -> B`
/// FEEC pushforward.
///
/// Outputs are kernel-major, with `[Bx, By, Bz]` for each kernel.  The stored
/// normalizations are the mesh quadrature approximations to each kernel
/// integral and allow cross-mesh quadrature auditing.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticVolumeSensorBank3d {
    kernels: Vec<WendlandC2Kernel3d>,
    quadrature_normalizations: Vec<f64>,
    magnetic_volume_averages: PhysicalMap,
}

impl MagneticVolumeSensorBank3d {
    /// Assemble fixed physical-volume averages from cell-barycenter
    /// quadrature of the canonical Whitney 2-form magnetic reconstruction.
    pub fn from_feec(
        topology: &Complex,
        coords: &MeshCoords,
        kernels: &[WendlandC2Kernel3d],
    ) -> Result<Self> {
        if topology.dim() != 3 || coords.dim() != 3 {
            return Err(FeecGmrfError::Dimension(
                "3D magnetic volume sensors require a 3D topology and coordinates".to_string(),
            ));
        }
        if kernels.is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "magnetic volume sensor bank requires at least one kernel".to_string(),
            ));
        }

        let magnetic = MagneticFieldMaps3d::from_feec(topology, coords)?;
        let cell_geometry = topology
            .cells()
            .handle_iter()
            .map(|cell| {
                let simplex = cell.coord_simplex(coords);
                let barycenter = simplex.barycenter();
                let volume = simplex.vol();
                ([barycenter[0], barycenter[1], barycenter[2]], volume)
            })
            .collect::<Vec<_>>();
        if cell_geometry
            .iter()
            .any(|(_, volume)| !volume.is_finite() || *volume <= 0.0)
        {
            return Err(FeecGmrfError::Assembly(
                "magnetic volume sensor mesh contains a non-positive or non-finite cell volume"
                    .to_string(),
            ));
        }

        let mut rows = Vec::with_capacity(3 * kernels.len());
        let mut quadrature_normalizations = Vec::with_capacity(kernels.len());
        for (kernel_index, kernel) in kernels.iter().copied().enumerate() {
            let unnormalized = cell_geometry
                .iter()
                .map(|(point, volume)| kernel.evaluate(*point).map(|weight| weight * volume))
                .collect::<Result<Vec<_>>>()?;
            let normalization = unnormalized.iter().sum::<f64>();
            if !normalization.is_finite() || normalization <= 0.0 {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "magnetic volume kernel {kernel_index} has zero support under the mesh quadrature"
                )));
            }
            quadrature_normalizations.push(normalization);
            for component in 0..3 {
                rows.push(
                    unnormalized
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, weight)| *weight != 0.0)
                        .map(|(cell, weight)| (3 * cell + component, weight / normalization))
                        .collect(),
                );
            }
        }
        let averaging =
            LinearMap::weighted_rows(magnetic.magnetic_field.map().output_dimension(), &rows)?;
        let magnetic_volume_averages = PhysicalMap::new(
            "fixed_footprint_magnetic_volume_averages",
            averaging.compose(magnetic.magnetic_field.map())?,
        )?;
        Ok(Self {
            kernels: kernels.to_vec(),
            quadrature_normalizations,
            magnetic_volume_averages,
        })
    }

    pub fn kernels(&self) -> &[WendlandC2Kernel3d] {
        &self.kernels
    }

    pub fn quadrature_normalizations(&self) -> &[f64] {
        &self.quadrature_normalizations
    }

    pub fn map(&self) -> &PhysicalMap {
        &self.magnetic_volume_averages
    }
}

impl MagneticFieldMaps3d {
    /// Assemble the full-space FEEC magnetic chain and its common QoI maps.
    pub fn from_feec(topology: &Complex, coords: &MeshCoords) -> Result<Self> {
        if topology.dim() != 3 || coords.dim() != 3 {
            return Err(FeecGmrfError::Dimension(
                "3D magnetic maps require a 3D topology and coordinates".to_string(),
            ));
        }
        let selected_cells = (0..topology.nsimplices(3)).collect::<Vec<_>>();
        Self::from_feec_with_volume_average_cells(topology, coords, &selected_cells)
    }

    /// Assemble the magnetic chain with volume averages over selected cells.
    ///
    /// The full cellwise magnetic-field map is unchanged. Only the three
    /// volume-average component maps and their normalized RMS weights are
    /// restricted to the supplied physical region.
    pub fn from_feec_with_volume_average_cells(
        topology: &Complex,
        coords: &MeshCoords,
        selected_cells: &[usize],
    ) -> Result<Self> {
        let flux_cochain = linear_map_from_rows(
            feg_infer::physical::build_exterior_derivative_1_operator(topology)
                .map_err(FeecGmrfError::Assembly)?,
        )?;
        let magnetic_field = PhysicalMap::new(
            "magnetic_field",
            linear_map_from_rows(
                feg_infer::physical::build_full_magnetic_flux_density_operator_3d(topology, coords)
                    .map_err(FeecGmrfError::Assembly)?,
            )?,
        )?;
        let averages = feg_infer::physical::build_selected_cell_magnetic_flux_density_averages_3d(
            topology,
            coords,
            selected_cells,
        )
        .map_err(FeecGmrfError::Assembly)?;
        Ok(Self {
            flux_cochain,
            magnetic_field,
            volume_average_components: [
                linear_map_from_rows(averages.x)?,
                linear_map_from_rows(averages.y)?,
                linear_map_from_rows(averages.z)?,
            ],
            vector_rms_weights: averages.vector_rms_weights,
            domain_volume: averages.domain_volume,
        })
    }

    /// Convert cellwise component variances into the volume-weighted
    /// RMS standard deviation of the reconstructed magnetic field.
    pub fn vector_rms_standard_deviation(&self, variances: &[f64]) -> Result<f64> {
        if variances.len() != self.vector_rms_weights.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "magnetic variance count {} does not match weight count {}",
                variances.len(),
                self.vector_rms_weights.len()
            )));
        }
        if variances
            .iter()
            .any(|variance| !variance.is_finite() || *variance < 0.0)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "magnetic variances must be finite and non-negative".to_string(),
            ));
        }
        Ok(variances
            .iter()
            .zip(&self.vector_rms_weights)
            .map(|(variance, weight)| variance * weight)
            .sum::<f64>()
            .sqrt())
    }

    /// Volume-weighted RMS magnitude of a cellwise reconstructed vector field.
    pub fn vector_rms_norm(&self, values: &[f64]) -> Result<f64> {
        if values.len() != self.vector_rms_weights.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "magnetic value count {} does not match weight count {}",
                values.len(),
                self.vector_rms_weights.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "magnetic field values must be finite".to_string(),
            ));
        }
        Ok(values
            .iter()
            .zip(&self.vector_rms_weights)
            .map(|(value, weight)| value * value * weight)
            .sum::<f64>()
            .sqrt())
    }

    /// Relative vector-field RMS gap, normalized by the reference field RMS.
    pub fn relative_vector_rms_error(&self, values: &[f64], reference: &[f64]) -> Result<f64> {
        if values.len() != reference.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "magnetic value count {} does not match reference count {}",
                values.len(),
                reference.len()
            )));
        }
        let reference_norm = self.vector_rms_norm(reference)?;
        if reference_norm == 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "relative magnetic-field error requires a non-zero reference".to_string(),
            ));
        }
        let difference = values
            .iter()
            .zip(reference)
            .map(|(value, reference)| value - reference)
            .collect::<Vec<_>>();
        Ok(self.vector_rms_norm(&difference)? / reference_norm)
    }
}

/// Build an outward flux functional over selected boundary faces.
pub fn outward_boundary_flux_map_3d(
    topology: &Complex,
    coords: &MeshCoords,
    face_indices: &[usize],
) -> Result<LinearMap> {
    linear_map_from_rows(
        feg_infer::physical::build_outward_boundary_flux_operator_3d(
            topology,
            coords,
            face_indices,
        )
        .map_err(FeecGmrfError::Assembly)?,
    )
}

/// Normalized lumped P1 mass weights for the domain-averaged RMS of a scalar
/// field represented by vertex coefficients.
pub fn scalar_field_l2_rms_weights(topology: &Complex, coords: &MeshCoords) -> Result<Vec<f64>> {
    Ok(
        feg_infer::physical::build_scalar_field_rms_weights(topology, coords)
            .map_err(FeecGmrfError::Assembly)?
            .vertex_weights,
    )
}

/// Reconstruct a 1-form cochain as cell-barycenter physical vector components.
/// Outputs are ordered component-major, with all cells for component zero
/// followed by all cells for each remaining component.
pub fn reconstructed_barycenter_1form_map(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<PhysicalMap> {
    let reconstruction =
        feg_infer::prior::matern::one_form::build_reconstructed_barycenter_field_operator(
            topology, coords,
        )
        .map_err(FeecGmrfError::Assembly)?;
    let mut rows =
        Vec::with_capacity(reconstruction.component_count() * reconstruction.cell_count());
    for component in 0..reconstruction.component_count() {
        rows.extend_from_slice(reconstruction.component_rows(component).ok_or_else(|| {
            FeecGmrfError::Assembly(format!(
                "reconstructed 1-form field is missing component {component}"
            ))
        })?);
    }
    PhysicalMap::new(
        "reconstructed_1form_field",
        LinearMap::weighted_rows(topology.edges().len(), &rows)?,
    )
}

impl PhysicalMap {
    /// Construct a physical map from an explicit linear operator.
    pub fn new(name: impl Into<String>, map: LinearMap) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "physical quantity name must not be empty".to_string(),
            ));
        }
        Ok(Self { name, map })
    }

    /// Physical quantity name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Composed operator.
    pub fn map(&self) -> &LinearMap {
        &self.map
    }

    /// Apply the physical map.
    pub fn apply(&self, cochain: &[f64]) -> Result<Vec<f64>> {
        self.map.apply(cochain)
    }

    /// Convert this physical map into a named model-derived quantity.
    pub fn into_derived_quantity(self) -> Result<DerivedQuantity> {
        DerivedQuantity::new(self.name, self.map)
    }
}

/// Construct the FEEC magnetic-field map
/// `A --D1--> flux cochain --reconstruction/Hodge--> physical B`.
pub fn magnetic_field_map(
    exterior_derivative_1: &LinearMap,
    flux_reconstruction: &LinearMap,
) -> Result<PhysicalMap> {
    PhysicalMap::new(
        "magnetic_field",
        flux_reconstruction.compose(exterior_derivative_1)?,
    )
}

/// Result of scaling a prior to a target root-mean-square physical standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalRmsCalibration {
    /// RMS standard deviation before calibration.
    pub uncalibrated_rms: f64,
    /// Requested RMS standard deviation.
    pub target_rms: f64,
    /// Scalar multiplying the original precision.
    pub precision_scale: f64,
}

/// Scale a prior so the mean marginal variance of `physical_map * x` is
/// `target_rms^2`.
///
/// This uses the generic GMRF covariance action and therefore works for any
/// FEEC form degree, geometry, or physical reconstruction. The returned prior
/// has the same mean and form-degree metadata as the input.
pub fn calibrate_prior_to_physical_rms(
    prior: &GaussianPrior,
    physical_map: &LinearMap,
    target_rms: f64,
) -> Result<(GaussianPrior, PhysicalRmsCalibration)> {
    if physical_map.output_dimension() == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "physical RMS calibration requires at least one output".to_string(),
        ));
    }
    let weights =
        vec![1.0 / physical_map.output_dimension() as f64; physical_map.output_dimension()];
    calibrate_prior_to_weighted_physical_rms(prior, physical_map, &weights, target_rms)
}

/// Estimator-aware variant of [`calibrate_prior_to_physical_rms`].
pub fn calibrate_prior_to_physical_rms_with_method(
    prior: &GaussianPrior,
    physical_map: &LinearMap,
    target_rms: f64,
    method: crate::infer::VarianceMethod,
) -> Result<(
    GaussianPrior,
    PhysicalRmsCalibration,
    crate::infer::WeightedVarianceEstimate,
)> {
    if physical_map.output_dimension() == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "physical RMS calibration requires at least one output".to_string(),
        ));
    }
    let weights =
        vec![1.0 / physical_map.output_dimension() as f64; physical_map.output_dimension()];
    calibrate_prior_to_weighted_physical_rms_with_method(
        prior,
        physical_map,
        &weights,
        target_rms,
        method,
    )
}

/// Scale a prior so `sum_i output_weights[i] * Var((Gx)_i) = target_rms^2`.
///
/// The supplied weight scale is preserved.
pub fn calibrate_prior_to_weighted_physical_rms(
    prior: &GaussianPrior,
    physical_map: &LinearMap,
    output_weights: &[f64],
    target_rms: f64,
) -> Result<(GaussianPrior, PhysicalRmsCalibration)> {
    let (prior, calibration, _) = calibrate_prior_to_weighted_physical_rms_with_method(
        prior,
        physical_map,
        output_weights,
        target_rms,
        crate::infer::VarianceMethod::Exact,
    )?;
    Ok((prior, calibration))
}

/// Estimator-aware weighted physical-RMS calibration.
pub fn calibrate_prior_to_weighted_physical_rms_with_method(
    prior: &GaussianPrior,
    physical_map: &LinearMap,
    output_weights: &[f64],
    target_rms: f64,
    method: crate::infer::VarianceMethod,
) -> Result<(
    GaussianPrior,
    PhysicalRmsCalibration,
    crate::infer::WeightedVarianceEstimate,
)> {
    if !target_rms.is_finite() || target_rms <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "target physical RMS must be finite and positive".to_string(),
        ));
    }
    if physical_map.output_dimension() == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "physical RMS calibration requires at least one output".to_string(),
        ));
    }
    if output_weights.len() != physical_map.output_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "physical RMS weight count {} does not match output dimension {}",
            output_weights.len(),
            physical_map.output_dimension()
        )));
    }
    if output_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || output_weights.iter().sum::<f64>() <= 0.0
    {
        return Err(FeecGmrfError::InvalidParameter(
            "physical RMS weights must be finite, non-negative, and have positive sum".to_string(),
        ));
    }

    let estimate =
        prior.pushforward_weighted_variance_estimate(physical_map, output_weights, method)?;
    let weighted_variance = estimate.weighted_trace;
    if !weighted_variance.is_finite() || weighted_variance <= 0.0 {
        return Err(FeecGmrfError::Inference(
            "physical pushforward has zero or non-finite prior variance".to_string(),
        ));
    }
    let uncalibrated_rms = weighted_variance.sqrt();
    let precision_scale = weighted_variance / target_rms.powi(2);
    let calibrated = prior.with_precision_scale(precision_scale)?;
    Ok((
        calibrated,
        PhysicalRmsCalibration {
            uncalibrated_rms,
            target_rms,
            precision_scale,
        },
        estimate,
    ))
}

fn linear_map_from_rows(operator: gmrf_core::SparseRowOperator) -> Result<LinearMap> {
    LinearMap::weighted_rows(operator.ncols, &operator.rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::EssentialBoundaryConditions;
    use crate::operator::SparseMat;
    use manifold::gen::cartesian::CartesianMeshInfo;

    #[test]
    fn selected_cell_magnetic_average_is_exposed_by_root_api() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let maps =
            MagneticFieldMaps3d::from_feec_with_volume_average_cells(&topology, &coords, &[0])
                .unwrap();
        assert!(maps.domain_volume > 0.0 && maps.domain_volume < 1.0);
        assert_eq!(maps.volume_average_components.len(), 3);
        assert!(maps
            .volume_average_components
            .iter()
            .all(|map| map.output_dimension() == 1));
        assert!((maps.vector_rms_weights.iter().sum::<f64>() - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn fixed_footprint_magnetic_sensors_preserve_constant_fields() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 3, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let kernels = [
            WendlandC2Kernel3d::new([0.35, 0.35, 0.35], 0.3).unwrap(),
            WendlandC2Kernel3d::new([0.65, 0.65, 0.65], 0.3).unwrap(),
        ];
        let sensors = MagneticVolumeSensorBank3d::from_feec(&topology, &coords, &kernels).unwrap();
        let field = exterior::field::DiffFormClosure::one_form(
            |point| common::linalg::nalgebra::Vector::from_column_slice(&[0.0, 0.0, point[1]]),
            3,
        );
        let cochain = ddf::cochain::cochain_projection(&field, &topology, &coords, None);
        let values = sensors.map().apply(cochain.coeffs.as_slice()).unwrap();
        assert_eq!(values.len(), 6);
        for kernel in 0..2 {
            assert!((values[3 * kernel] - 1.0).abs() < 1.0e-10);
            assert!(values[3 * kernel + 1].abs() < 1.0e-10);
            assert!(values[3 * kernel + 2].abs() < 1.0e-10);
        }
    }

    #[test]
    fn fixed_footprint_kernel_quadrature_converges_to_whole_space_integral() {
        let kernel = WendlandC2Kernel3d::new([0.5; 3], 0.35).unwrap();
        let exact = kernel.whole_space_integral();
        let approximation = |resolution| {
            let mesh = CartesianMeshInfo::new_unit_scaled(3, resolution, 1.0);
            let (topology, coords) = mesh.compute_coord_complex();
            MagneticVolumeSensorBank3d::from_feec(&topology, &coords, &[kernel])
                .unwrap()
                .quadrature_normalizations()[0]
        };
        let coarse_error = (approximation(2) - exact).abs();
        let fine_error = (approximation(8) - exact).abs();
        assert!(
            fine_error < coarse_error,
            "refined fixed-kernel quadrature error {fine_error:.6e} did not improve over {coarse_error:.6e}"
        );
        assert!(fine_error / exact < 0.1);
    }

    #[test]
    fn fixed_footprint_kernels_validate_parameters_and_mesh_support() {
        assert!(WendlandC2Kernel3d::new([f64::NAN, 0.0, 0.0], 1.0).is_err());
        assert!(WendlandC2Kernel3d::new([0.0; 3], 0.0).is_err());
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let outside = WendlandC2Kernel3d::new([4.0; 3], 0.1).unwrap();
        assert!(MagneticVolumeSensorBank3d::from_feec(&topology, &coords, &[outside]).is_err());
    }

    #[test]
    fn magnetic_map_is_explicit_operator_composition() {
        let derivative = LinearMap::new(SparseMat::diagonal(2, 2.0)).unwrap();
        let reconstruction = LinearMap::new(SparseMat::diagonal(2, 3.0)).unwrap();
        let magnetic = magnetic_field_map(&derivative, &reconstruction).unwrap();
        assert_eq!(magnetic.apply(&[1.0, -1.0]).unwrap(), vec![6.0, -6.0]);
    }

    #[test]
    fn physical_rms_calibration_uses_pushforward_covariance() {
        let prior = GaussianPrior::new(vec![0.0; 2], SparseMat::diagonal(2, 1.0)).unwrap();
        let map = LinearMap::new(SparseMat::diagonal(2, 2.0)).unwrap();
        let (calibrated, report) = calibrate_prior_to_physical_rms(&prior, &map, 0.5).unwrap();
        assert!((report.uncalibrated_rms - 2.0).abs() < 1e-12);
        assert!((report.precision_scale - 16.0).abs() < 1e-12);
        assert_eq!(calibrated.precision(), &SparseMat::diagonal(2, 16.0));
    }

    #[test]
    fn physical_rms_calibration_accepts_full_boundary_constrained_map() {
        let prior = GaussianPrior::new(vec![0.0; 2], SparseMat::diagonal(2, 1.0))
            .unwrap()
            .condition_on_essential_boundary(
                EssentialBoundaryConditions::prescribed(vec![1], vec![3.0]).unwrap(),
            )
            .unwrap();
        let full_map = LinearMap::identity(2);
        let (calibrated, report) = calibrate_prior_to_physical_rms(&prior, &full_map, 1.0).unwrap();
        assert!((report.uncalibrated_rms - 0.5_f64.sqrt()).abs() < 1e-12);
        assert!((report.precision_scale - 0.5).abs() < 1e-12);
        assert_eq!(calibrated.precision(), &SparseMat::diagonal(1, 0.5));
        assert_eq!(calibrated.cochain_mean().unwrap(), [0.0, 3.0]);
    }

    #[test]
    fn weighted_calibration_uses_weights_without_normalizing_them() {
        let prior = GaussianPrior::new(vec![2.0, -1.0], SparseMat::diagonal(2, 1.0)).unwrap();
        let map = LinearMap::identity(2);
        let (calibrated, report) =
            calibrate_prior_to_weighted_physical_rms(&prior, &map, &[1.0, 2.0], 1.0).unwrap();
        assert!((report.uncalibrated_rms - 3.0_f64.sqrt()).abs() < 1e-12);
        assert!((report.precision_scale - 3.0).abs() < 1e-12);
        assert_eq!(calibrated.mean(), prior.mean());
    }

    #[test]
    fn magnetic_rms_diagnostic_uses_canonical_component_weights() {
        let identity = LinearMap::identity(2);
        let maps = MagneticFieldMaps3d {
            flux_cochain: identity.clone(),
            magnetic_field: PhysicalMap::new("b", identity.clone()).unwrap(),
            volume_average_components: [identity.clone(), identity.clone(), identity],
            vector_rms_weights: vec![0.25, 0.75],
            domain_volume: 1.0,
        };
        let rms = maps.vector_rms_standard_deviation(&[4.0, 12.0]).unwrap();
        assert!((rms - 10.0_f64.sqrt()).abs() < 1.0e-12);
        assert!(maps.vector_rms_standard_deviation(&[1.0]).is_err());
        assert!(maps.vector_rms_standard_deviation(&[1.0, -1.0]).is_err());
        assert!((maps.vector_rms_norm(&[2.0, 2.0]).unwrap() - 2.0).abs() < 1.0e-12);
        assert!(
            (maps
                .relative_vector_rms_error(&[3.0, 3.0], &[2.0, 2.0])
                .unwrap()
                - 0.5)
                .abs()
                < 1.0e-12
        );
    }
}
