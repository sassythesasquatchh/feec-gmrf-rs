//! Physical outputs represented as explicit compositions of FEEC operators.

use crate::infer::Posterior;
use crate::model::{
    DerivedQuantity, LinearConstraint, LinearGaussianModelBuilder, LinearObservation,
};
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

/// A positively oriented axis-aligned surface patch for integrating a magnetic
/// flux 2-cochain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisAlignedFluxPatch3d {
    normal_axis: usize,
    coordinate: f64,
    tangential_ranges: [[f64; 2]; 2],
}

impl AxisAlignedFluxPatch3d {
    /// Define a patch with positive normal along `normal_axis`.
    ///
    /// The two ranges follow the cyclic tangential axes: `(y,z)` for an
    /// x-normal patch, `(z,x)` for a y-normal patch, and `(x,y)` for a z-normal
    /// patch. Patch boundaries must conform to mesh faces when assembled.
    pub fn new(
        normal_axis: usize,
        coordinate: f64,
        tangential_ranges: [[f64; 2]; 2],
    ) -> Result<Self> {
        if normal_axis >= 3 {
            return Err(FeecGmrfError::Dimension(
                "axis-aligned flux patch normal axis must be 0, 1, or 2".to_string(),
            ));
        }
        if !coordinate.is_finite()
            || tangential_ranges
                .iter()
                .flatten()
                .any(|bound| !bound.is_finite())
        {
            return Err(FeecGmrfError::InvalidParameter(
                "axis-aligned flux patch coordinates must be finite".to_string(),
            ));
        }
        if tangential_ranges.iter().any(|range| range[0] >= range[1]) {
            return Err(FeecGmrfError::InvalidParameter(
                "axis-aligned flux patch ranges must have positive width".to_string(),
            ));
        }
        Ok(Self {
            normal_axis,
            coordinate,
            tangential_ranges,
        })
    }

    pub fn normal_axis(self) -> usize {
        self.normal_axis
    }

    pub fn coordinate(self) -> f64 {
        self.coordinate
    }

    pub fn tangential_ranges(self) -> [[f64; 2]; 2] {
        self.tangential_ranges
    }

    pub fn area(self) -> f64 {
        self.tangential_ranges
            .iter()
            .map(|range| range[1] - range[0])
            .product()
    }
}

/// Pair full-space FEEC 2-cochains with oriented, mesh-conforming surface
/// patches in three dimensions.
///
/// Every coefficient is `+1` or `-1`: the map sums the already-integrated face
/// degrees of freedom with the orientation induced by the patch normal. It
/// performs no point evaluation, Whitney reconstruction, or barycentre
/// quadrature.
pub fn axis_aligned_flux_patch_cochain_map_3d(
    topology: &Complex,
    coords: &MeshCoords,
    patches: &[AxisAlignedFluxPatch3d],
) -> Result<LinearMap> {
    const COORDINATE_TOLERANCE: f64 = 1.0e-10;
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(FeecGmrfError::Dimension(
            "axis-aligned flux patches require a 3D topology and coordinates".to_string(),
        ));
    }
    if patches.is_empty() {
        return Err(FeecGmrfError::InvalidParameter(
            "at least one axis-aligned flux patch is required".to_string(),
        ));
    }

    let mut rows = Vec::with_capacity(patches.len());
    for (patch_index, patch) in patches.iter().copied().enumerate() {
        let tangential_axes = cyclic_tangential_axes(patch.normal_axis);
        let mut row = Vec::new();
        let mut selected_area = 0.0;
        for (face_index, face) in topology.skeleton(2).handle_iter().enumerate() {
            let points = face
                .vertices
                .iter()
                .map(|&vertex| {
                    let point = coords.coord(vertex);
                    [point[0], point[1], point[2]]
                })
                .collect::<Vec<_>>();
            if points.iter().any(|point| {
                (point[patch.normal_axis] - patch.coordinate).abs() > COORDINATE_TOLERANCE
                    || tangential_axes.iter().enumerate().any(|(range, &axis)| {
                        point[axis] < patch.tangential_ranges[range][0] - COORDINATE_TOLERANCE
                            || point[axis]
                                > patch.tangential_ranges[range][1] + COORDINATE_TOLERANCE
                    })
            }) {
                continue;
            }
            let projected_area =
                signed_projected_triangle_area(points[0], points[1], points[2], tangential_axes);
            if projected_area.abs() <= COORDINATE_TOLERANCE {
                continue;
            }
            row.push((face_index, projected_area.signum()));
            selected_area += projected_area.abs();
        }
        if row.is_empty() {
            return Err(FeecGmrfError::Assembly(format!(
                "axis-aligned flux patch {patch_index} selects no mesh faces"
            )));
        }
        let expected_area = patch.area();
        let area_tolerance = (1.0e-10 * expected_area).max(1.0e-12);
        if (selected_area - expected_area).abs() > area_tolerance {
            return Err(FeecGmrfError::Assembly(format!(
                "axis-aligned flux patch {patch_index} selects area {selected_area:.16e}, expected {expected_area:.16e}; patch boundaries must conform to mesh faces"
            )));
        }
        rows.push(row);
    }
    LinearMap::weighted_rows(topology.nsimplices(2), &rows)
}

fn cyclic_tangential_axes(normal_axis: usize) -> [usize; 2] {
    match normal_axis {
        0 => [1, 2],
        1 => [2, 0],
        2 => [0, 1],
        _ => unreachable!("validated three-dimensional axis"),
    }
}

fn signed_projected_triangle_area(
    first: [f64; 3],
    second: [f64; 3],
    third: [f64; 3],
    axes: [usize; 2],
) -> f64 {
    0.5 * ((second[axes[0]] - first[axes[0]]) * (third[axes[1]] - first[axes[1]])
        - (second[axes[1]] - first[axes[1]]) * (third[axes[0]] - first[axes[0]]))
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

impl PhysicalRmsCalibration {
    /// Scalar multiplying the Matérn amplitude parameter `tau`.
    ///
    /// Matérn precision is proportional to `tau^2`, so this is the positive
    /// square root of [`Self::precision_scale`]. The correlation structure is
    /// unchanged by this calibration.
    pub fn tau_multiplier(self) -> f64 {
        self.precision_scale.sqrt()
    }
}

/// Controls a one-parameter calibration of prior precision against posterior
/// physical RMS after a fixed linear-Gaussian conditioning design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosteriorPhysicalRmsCalibrationOptions {
    /// Required relative error in the achieved posterior RMS.
    pub relative_tolerance: f64,
    /// Maximum log-scale bisection steps after a bracket has been found.
    pub maximum_iterations: usize,
    /// Maximum geometric expansions used to bracket the target.
    pub maximum_bracket_expansions: usize,
    /// Positive factor by which the precision scale is expanded or contracted.
    pub bracket_expansion_factor: f64,
}

impl Default for PosteriorPhysicalRmsCalibrationOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-6,
            maximum_iterations: 32,
            maximum_bracket_expansions: 16,
            bracket_expansion_factor: 4.0,
        }
    }
}

impl PosteriorPhysicalRmsCalibrationOptions {
    pub fn validate(self) -> Result<Self> {
        if !self.relative_tolerance.is_finite() || self.relative_tolerance <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "posterior RMS calibration tolerance must be finite and positive".to_string(),
            ));
        }
        if self.maximum_iterations == 0 || self.maximum_bracket_expansions == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "posterior RMS calibration iteration limits must be positive".to_string(),
            ));
        }
        if !self.bracket_expansion_factor.is_finite() || self.bracket_expansion_factor <= 1.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "posterior RMS calibration bracket factor must be finite and greater than one"
                    .to_string(),
            ));
        }
        Ok(self)
    }
}

/// Result of calibrating one prior-precision scalar to a posterior physical RMS.
///
/// The observation operators and noise model are held fixed. For a Matérn
/// prior, [`Self::tau_multiplier`] is the scalar applied to `tau` before the
/// model is conditioned again. This is distinct from scaling an already
/// conditioned covariance: the returned [`Posterior`] is the coherent
/// posterior of the recalibrated prior and the original likelihood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosteriorPhysicalRmsCalibration {
    pub uncalibrated_rms: f64,
    pub target_rms: f64,
    pub achieved_rms: f64,
    pub precision_scale: f64,
    pub relative_error: f64,
    pub bracket_expansions: usize,
    pub iterations: usize,
}

impl PosteriorPhysicalRmsCalibration {
    /// Scalar multiplying a Matérn `tau` whose precision is proportional to
    /// `tau^2`.
    pub fn tau_multiplier(self) -> f64 {
        self.precision_scale.sqrt()
    }
}

/// Calibrate one global prior-precision scalar so the conditioned model has a
/// requested RMS physical standard deviation.
///
/// Every trial uses the same observations, observation noise, hard
/// constraints, and physical map. The scalar is found by bracketed bisection
/// in log precision, exploiting the monotone decrease of posterior covariance
/// with increasing proper prior precision. The final posterior is rebuilt at
/// the selected scale; no post-hoc covariance multiplication is performed.
pub fn calibrate_prior_to_posterior_physical_rms(
    prior: &GaussianPrior,
    observations: &[LinearObservation],
    constraints: &[LinearConstraint],
    physical_map: &LinearMap,
    target_rms: f64,
    options: PosteriorPhysicalRmsCalibrationOptions,
) -> Result<(GaussianPrior, Posterior, PosteriorPhysicalRmsCalibration)> {
    let options = options.validate()?;
    if !target_rms.is_finite() || target_rms <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "target posterior physical RMS must be finite and positive".to_string(),
        ));
    }
    if physical_map.output_dimension() == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "posterior physical RMS calibration requires at least one output".to_string(),
        ));
    }

    let evaluate = |scale: f64| -> Result<(GaussianPrior, Posterior, f64)> {
        let scaled = prior.with_precision_scale(scale)?;
        let mut builder = LinearGaussianModelBuilder::new(scaled.clone());
        for observation in observations {
            builder = builder.observe(observation.clone())?;
        }
        for constraint in constraints {
            builder = builder.constrain(constraint.clone())?;
        }
        let mut posterior = builder.condition()?;
        let covariance = posterior.pushforward_covariance(physical_map)?;
        let rms = covariance_mean_marginal_rms(&covariance)?;
        Ok((scaled, posterior, rms))
    };

    let (uncalibrated_prior, uncalibrated_posterior, uncalibrated_rms) = evaluate(1.0)?;
    let relative_error = (uncalibrated_rms - target_rms).abs() / target_rms;
    if relative_error <= options.relative_tolerance {
        return Ok((
            uncalibrated_prior,
            uncalibrated_posterior,
            PosteriorPhysicalRmsCalibration {
                uncalibrated_rms,
                target_rms,
                achieved_rms: uncalibrated_rms,
                precision_scale: 1.0,
                relative_error,
                bracket_expansions: 0,
                iterations: 0,
            },
        ));
    }

    let mut lower_scale;
    let mut lower_rms;
    let mut upper_scale;
    let mut upper_rms;
    let mut bracket_expansions = 0;
    if uncalibrated_rms > target_rms {
        lower_scale = 1.0;
        lower_rms = uncalibrated_rms;
        upper_scale = 1.0;
        upper_rms = uncalibrated_rms;
        while upper_rms > target_rms && bracket_expansions < options.maximum_bracket_expansions {
            upper_scale *= options.bracket_expansion_factor;
            if !upper_scale.is_finite() {
                break;
            }
            upper_rms = evaluate(upper_scale)?.2;
            bracket_expansions += 1;
        }
    } else {
        upper_scale = 1.0;
        upper_rms = uncalibrated_rms;
        lower_scale = 1.0;
        lower_rms = uncalibrated_rms;
        while lower_rms < target_rms && bracket_expansions < options.maximum_bracket_expansions {
            lower_scale /= options.bracket_expansion_factor;
            if !lower_scale.is_finite() || lower_scale <= 0.0 {
                break;
            }
            lower_rms = evaluate(lower_scale)?.2;
            bracket_expansions += 1;
        }
    }
    if !(lower_scale < upper_scale && lower_rms >= target_rms && upper_rms <= target_rms) {
        return Err(FeecGmrfError::Inference(format!(
            "could not bracket posterior physical RMS target {target_rms:.16e}: scale/rms lower {lower_scale:.16e}/{lower_rms:.16e}, upper {upper_scale:.16e}/{upper_rms:.16e}"
        )));
    }

    for iteration in 1..=options.maximum_iterations {
        let scale = (lower_scale * upper_scale).sqrt();
        let (scaled, posterior, rms) = evaluate(scale)?;
        let relative_error = (rms - target_rms).abs() / target_rms;
        if relative_error <= options.relative_tolerance {
            return Ok((
                scaled,
                posterior,
                PosteriorPhysicalRmsCalibration {
                    uncalibrated_rms,
                    target_rms,
                    achieved_rms: rms,
                    precision_scale: scale,
                    relative_error,
                    bracket_expansions,
                    iterations: iteration,
                },
            ));
        }
        if rms > target_rms {
            lower_scale = scale;
            lower_rms = rms;
        } else {
            upper_scale = scale;
            upper_rms = rms;
        }
    }
    Err(FeecGmrfError::Inference(format!(
        "posterior physical RMS calibration did not reach relative tolerance {:.6e} after {} iterations; final bracket scale/rms lower {:.16e}/{:.16e}, upper {:.16e}/{:.16e}",
        options.relative_tolerance,
        options.maximum_iterations,
        lower_scale,
        lower_rms,
        upper_scale,
        upper_rms
    )))
}

/// Root-mean-square marginal standard deviation represented by a covariance
/// matrix.
///
/// This is `sqrt(trace(C) / n)`. It is useful when a collection of physical
/// integral quantities, rather than point samples, defines an amplitude
/// calibration target.
pub fn covariance_mean_marginal_rms(covariance: &[Vec<f64>]) -> Result<f64> {
    if covariance.is_empty()
        || covariance
            .iter()
            .any(|row| row.len() != covariance.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(FeecGmrfError::Dimension(
            "physical covariance must be nonempty, square, and finite".to_string(),
        ));
    }
    let mean_variance = covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .sum::<f64>()
        / covariance.len() as f64;
    if !mean_variance.is_finite() || mean_variance <= 0.0 {
        return Err(FeecGmrfError::Inference(
            "physical covariance has non-positive mean marginal variance".to_string(),
        ));
    }
    Ok(mean_variance.sqrt())
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
    fn axis_aligned_flux_patch_pairs_integral_cochains_without_reconstruction() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 4, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let patch = AxisAlignedFluxPatch3d::new(0, 0.5, [[0.0, 0.5], [0.0, 0.5]]).unwrap();
        let patch_map =
            axis_aligned_flux_patch_cochain_map_3d(&topology, &coords, &[patch]).unwrap();
        let magnetic = MagneticFieldMaps3d::from_feec(&topology, &coords).unwrap();
        let vector_potential = exterior::field::DiffFormClosure::one_form(
            |point| common::linalg::nalgebra::Vector::from_column_slice(&[0.0, 0.0, point[1]]),
            3,
        );
        let a_cochain =
            ddf::cochain::cochain_projection(&vector_potential, &topology, &coords, None);
        let b_cochain = magnetic
            .flux_cochain
            .apply(a_cochain.coeffs.as_slice())
            .unwrap();
        let flux = patch_map.apply(&b_cochain).unwrap();

        assert!((flux[0] - patch.area()).abs() < 1.0e-12);
        assert!(AxisAlignedFluxPatch3d::new(3, 0.5, [[0.0, 0.5], [0.0, 0.5]]).is_err());
        let nonconforming = AxisAlignedFluxPatch3d::new(0, 0.3, [[0.0, 0.5], [0.0, 0.5]]).unwrap();
        assert!(
            axis_aligned_flux_patch_cochain_map_3d(&topology, &coords, &[nonconforming]).is_err()
        );
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
    fn fixed_footprint_sensors_converge_for_nonconstant_analytic_curl() {
        let center = [0.42, 0.55, 0.61];
        let kernel = WendlandC2Kernel3d::new(center, 0.3).unwrap();
        let expected = [center[0] * center[2], -center[1] * center[2], 0.0];
        let error = |resolution| {
            let mesh = CartesianMeshInfo::new_unit_scaled(3, resolution, 1.0);
            let (topology, coords) = mesh.compute_coord_complex();
            let sensors =
                MagneticVolumeSensorBank3d::from_feec(&topology, &coords, &[kernel]).unwrap();
            let field = exterior::field::DiffFormClosure::one_form(
                |point| {
                    common::linalg::nalgebra::Vector::from_column_slice(&[
                        0.0,
                        0.0,
                        point[0] * point[1] * point[2],
                    ])
                },
                3,
            );
            let cochain = ddf::cochain::cochain_projection(&field, &topology, &coords, None);
            let values = sensors.map().apply(cochain.coeffs.as_slice()).unwrap();
            values
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let errors = [error(2), error(4), error(8)];
        assert!(
            errors.windows(2).all(|pair| pair[1] < pair[0]),
            "nonconstant analytic-curl sensor errors did not decrease: {errors:?}"
        );
        assert!(
            errors[2] < 2.0e-3,
            "fine sensor error is {0:.6e}",
            errors[2]
        );
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
        assert!((report.tau_multiplier() - 4.0).abs() < 1e-12);
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
    fn posterior_rms_calibration_reconditions_instead_of_scaling_output_covariance() {
        let prior = GaussianPrior::new(vec![0.0], SparseMat::diagonal(1, 1.0)).unwrap();
        let identity = LinearMap::identity(1);
        let observation = LinearObservation::new(
            identity.clone(),
            vec![0.0],
            crate::model::GaussianNoise::variance(1.0).unwrap(),
        )
        .unwrap();
        let (calibrated, mut posterior, report) = calibrate_prior_to_posterior_physical_rms(
            &prior,
            &[observation],
            &[],
            &identity,
            0.25,
            PosteriorPhysicalRmsCalibrationOptions::default(),
        )
        .unwrap();

        // For Q=s and unit observation precision, posterior variance is
        // 1/(s+1). Target variance 1/16 therefore requires s=15.
        assert!((report.precision_scale - 15.0).abs() < 2.0e-5);
        assert!((report.tau_multiplier() - 15.0_f64.sqrt()).abs() < 3.0e-6);
        assert!(report.relative_error <= 1.0e-6);
        let calibrated_action = calibrated.precision().apply_checked(&[1.0]).unwrap();
        assert!((calibrated_action[0] - 15.0).abs() < 2.0e-5);
        let covariance = posterior.pushforward_covariance(&identity).unwrap();
        assert!((covariance[0][0] - 1.0 / 16.0).abs() < 2.0e-7);
    }

    #[test]
    fn covariance_mean_marginal_rms_uses_covariance_trace() {
        let covariance = vec![vec![4.0, 1.0], vec![1.0, 12.0]];
        assert!(
            (covariance_mean_marginal_rms(&covariance).unwrap() - 8.0_f64.sqrt()).abs() < 1.0e-12
        );
        assert!(covariance_mean_marginal_rms(&[]).is_err());
    }

    #[test]
    fn posterior_rms_calibration_rejects_unreachable_noise_limited_target() {
        let prior = GaussianPrior::new(vec![0.0], SparseMat::diagonal(1, 1.0)).unwrap();
        let identity = LinearMap::identity(1);
        let observation = LinearObservation::new(
            identity.clone(),
            vec![0.0],
            crate::model::GaussianNoise::variance(1.0).unwrap(),
        )
        .unwrap();
        let result = calibrate_prior_to_posterior_physical_rms(
            &prior,
            &[observation],
            &[],
            &identity,
            2.0,
            PosteriorPhysicalRmsCalibrationOptions {
                maximum_bracket_expansions: 4,
                ..PosteriorPhysicalRmsCalibrationOptions::default()
            },
        );
        let Err(error) = result else {
            panic!("noise-limited target unexpectedly calibrated");
        };
        assert!(error.to_string().contains("could not bracket"));
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
