use crate::prior::{
    matern::{
        one_form::{
            build_hodge_laplacian_1form, build_matern_mass_inverse_1form_with_coords,
            build_matern_system_matrix_1form, MaternMassInverse as Matern1FormMassInverse,
        },
        MaternAlpha,
    },
    sparse_anchor_hodge::spectrum_matched_potential_precision,
};
use crate::sparse::{
    feec_csr_to_core_triplet, feec_csr_to_gmrf, restrict_columns_and_fold_fixed,
    restrict_square_with_layout, sparse_row_operator_from_feec_csr,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::ManifoldComplexExt;
use feg_core::GaussianPriorSpec;
use formoniq::reduction::DofLayout;
use gmrf_core::SparseRowOperator;
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};

#[derive(Debug, Clone, Copy)]
pub struct ExactTwoFormPotentialPriorConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: Matern1FormMassInverse,
    pub diagonal_shift: f64,
}

impl Default for ExactTwoFormPotentialPriorConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            tau: 1.0,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
            diagonal_shift: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExactTwoFormPotentialPrior {
    pub spec: GaussianPriorSpec,
    pub precision: FeecCsr,
    pub mean: FeecVector,
    pub potential_to_field: FeecCsr,
    pub field_bias: FeecVector,
    pub field_dimension: usize,
    pub kappa: f64,
    pub tau: f64,
}

impl ExactTwoFormPotentialPrior {
    pub fn field_operator(&self) -> Result<SparseRowOperator, String> {
        sparse_row_operator_from_feec_csr(&self.potential_to_field)
    }
}

pub fn build_exact_two_form_potential_prior(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
    mean: Vec<f64>,
    config: ExactTwoFormPotentialPriorConfig,
) -> Result<ExactTwoFormPotentialPrior, String> {
    validate_config(config)?;
    if topology.dim() < 2 {
        return Err("exact B=dA prior requires topological dimension at least 2".to_string());
    }
    if coords.nvertices() != topology.nsimplices(0) {
        return Err(format!(
            "coordinate vertex count {} must match topology vertex count {}",
            coords.nvertices(),
            topology.nsimplices(0)
        ));
    }
    if layout.full_dimension != topology.nsimplices(1) {
        return Err(format!(
            "potential layout full dimension {} must match edge count {}",
            layout.full_dimension,
            topology.nsimplices(1)
        ));
    }
    if mean.len() != layout.reduced_dimension() {
        return Err(format!(
            "potential prior mean length {} must match active edge dimension {}",
            mean.len(),
            layout.reduced_dimension()
        ));
    }

    let metric = coords.to_edge_lengths(topology);
    build_exact_two_form_potential_prior_with_metric(
        topology, coords, &metric, layout, mean, config,
    )
}

pub fn build_exact_two_form_potential_prior_with_metric(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    layout: &DofLayout,
    mean: Vec<f64>,
    config: ExactTwoFormPotentialPriorConfig,
) -> Result<ExactTwoFormPotentialPrior, String> {
    validate_config(config)?;
    let hodge = build_hodge_laplacian_1form(topology, metric);
    let system = build_matern_system_matrix_1form(&hodge, config.kappa);
    let mass_inverse = build_matern_mass_inverse_1form_with_coords(
        topology,
        coords,
        metric,
        &hodge.mass_u,
        config.mass_inverse,
    )?;
    let full_precision = spectrum_matched_potential_precision(
        &system,
        &mass_inverse,
        MaternAlpha::Two,
        config.kappa,
        config.tau,
    )?;
    let mut precision = restrict_square_with_layout(&full_precision, layout)?;
    if config.diagonal_shift > 0.0 {
        precision = add_diagonal_shift(&precision, config.diagonal_shift);
    }
    feec_csr_to_gmrf(&precision)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("exact B=dA potential precision did not factorize: {err}"))?;

    let full_d = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let zero_bias = FeecVector::zeros(full_d.nrows());
    let (potential_to_field, field_bias) =
        restrict_columns_and_fold_fixed(&full_d, &zero_bias, layout)?;
    let mean_vec = FeecVector::from_vec(mean.clone());

    Ok(ExactTwoFormPotentialPrior {
        spec: GaussianPriorSpec {
            mean,
            precision: feec_csr_to_core_triplet(&precision),
        },
        precision,
        mean: mean_vec,
        potential_to_field,
        field_bias,
        field_dimension: full_d.nrows(),
        kappa: config.kappa,
        tau: config.tau,
    })
}

fn validate_config(config: ExactTwoFormPotentialPriorConfig) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("exact B=dA potential prior kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("exact B=dA potential prior tau must be finite and positive".to_string());
    }
    if !config.diagonal_shift.is_finite() || config.diagonal_shift < 0.0 {
        return Err(
            "exact B=dA potential prior diagonal shift must be finite and nonnegative".to_string(),
        );
    }
    Ok(())
}

fn add_diagonal_shift(matrix: &FeecCsr, shift: f64) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    FeecCsr::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::feec_csr_to_dense;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn max_abs(matrix: &FeecCsr) -> f64 {
        matrix
            .triplet_iter()
            .map(|(_, _, value)| value.abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn exact_two_form_potential_prior_rejects_negative_diagonal_shift() {
        let config = ExactTwoFormPotentialPriorConfig {
            diagonal_shift: -1.0,
            ..ExactTwoFormPotentialPriorConfig::default()
        };
        assert!(validate_config(config)
            .unwrap_err()
            .contains("diagonal shift"));
    }

    #[test]
    fn exact_two_form_potential_prior_factorizes_and_maps_active_edges_to_faces() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let layout = DofLayout::identity(topology.nsimplices(1));
        let prior = build_exact_two_form_potential_prior(
            &topology,
            &coords,
            &layout,
            vec![0.0; layout.reduced_dimension()],
            ExactTwoFormPotentialPriorConfig::default(),
        )
        .expect("exact B=dA prior should build");

        assert_eq!(prior.spec.mean.len(), topology.nsimplices(1));
        assert_eq!(prior.potential_to_field.nrows(), topology.nsimplices(2));
        assert_eq!(prior.potential_to_field.ncols(), topology.nsimplices(1));
        assert_eq!(prior.field_bias.len(), topology.nsimplices(2));
        feec_csr_to_gmrf(&prior.precision)
            .cholesky_sqrt_lower()
            .expect("precision should factorize");
    }

    #[test]
    fn exact_two_form_potential_transform_is_closed() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let layout = DofLayout::identity(topology.nsimplices(1));
        let prior = build_exact_two_form_potential_prior(
            &topology,
            &coords,
            &layout,
            vec![0.0; layout.reduced_dimension()],
            ExactTwoFormPotentialPriorConfig::default(),
        )
        .expect("exact B=dA prior should build");

        let d2 = FeecCsr::from(&topology.exterior_derivative_operator(2));
        let closed = &d2 * &prior.potential_to_field;
        assert!(
            max_abs(&closed) <= 1e-12,
            "d2 * d1 should vanish, got max abs {:.3e}",
            max_abs(&closed)
        );
        let dense = feec_csr_to_dense(&prior.potential_to_field);
        assert!(dense.iter().any(|value| value.abs() > 0.0));
    }
}
