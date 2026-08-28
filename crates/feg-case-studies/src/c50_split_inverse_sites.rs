//! C50-EMP-03: isolate the two inverse-metric sites in a Whitney 1-form law.
//!
//! This experiment uses the submitted posthoc ordering: assemble the
//! full-cochain projected inverse first, then take its active principal block
//! on the compatible homogeneous full-relative complex. The lower degree-zero
//! construction is the degree-zero specialization of FEEC's existing
//! full-polynomial projected assembler; no inverse or Matérn equation is
//! reproduced here.

use crate::cube_mass_inverse_variance::smooth_polynomial_one_form_cochain;
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feec_gmrf::prelude::{
    relative_precision_certificate, relative_scalar_error, relative_sparse_frobenius_error,
    relative_vector_l2_error, AdjacentMaternPriorBuilder, EliminatedMassInversePolicy,
    EssentialBoundaryElimination, ExactFormOperators, GaussianPrior, LinearMap,
    MagneticFieldMaps3d, MaternAlpha, MaternParameters, MaternRepresentation, PriorNormalization,
    Result, SparseMat,
};
use formoniq::assemble::assemble_whitney_projected_sparse_inverse_galmat_for_grade;
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{coord::simplex::SimplexHandleExt, metric::mesh::MeshLengths},
    topology::complex::Complex,
};

const DIMENSION: usize = 3;
const FORM_DEGREE: usize = 1;
const KAPPA: f64 = 4.0;
const TAU: f64 = 1.0;
const DEFAULT_MAXIMUM_EXACT_DIMENSION: usize = 1_024;

/// Fixed cheap refinements retained for C50-EMP-03.
pub const C50_SPLIT_INVERSE_SITE_REFINEMENTS: [usize; 3] = [2, 3, 4];

const PHYSICAL_TARGET: [f64; 3] = [0.37, 0.41, 0.43];
const PHYSICAL_COMPONENT: usize = 2;

/// Realization selected at one inverse-metric site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitInverseRealization {
    ExactMaterialized,
    /// Full-space projected inverse followed by active principal restriction.
    ProjectedPosthocRestricted,
}

/// One no-rescaling comparison against the exact/exact law on the same mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitInverseSiteRow {
    pub nboxes_per_dimension: usize,
    pub state_dimension: usize,
    pub lower_dimension: usize,
    pub alpha: MaternAlpha,
    pub lower_adjoint_inverse: SplitInverseRealization,
    pub recurrence_inverse: SplitInverseRealization,
    pub representation: MaternRepresentation,
    pub normalization: PriorNormalization,
    pub lower_projected_inverse_nnz: usize,
    pub recurrence_projected_inverse_nnz: usize,
    pub system_relative_frobenius_error: f64,
    pub precision_relative_frobenius_error: f64,
    pub precision_relative_spectral_error: f64,
    pub minimum_precision_ratio: f64,
    pub maximum_precision_ratio: f64,
    pub minimum_covariance_ratio: f64,
    pub maximum_covariance_ratio: f64,
    pub has_unit_relative_envelope: bool,
    pub smooth_l2_covariance_action_relative_error: f64,
    pub smooth_l2_functional_variance: f64,
    pub smooth_l2_functional_variance_relative_error: f64,
    pub smooth_cochain_covariance_action_relative_error: f64,
    pub smooth_cochain_functional_variance: f64,
    pub smooth_cochain_functional_variance_relative_error: f64,
    pub magnetic_target: [f64; 3],
    pub magnetic_evaluation_point: [f64; 3],
    pub magnetic_component: usize,
    pub magnetic_covariance_action_relative_error: f64,
    pub magnetic_functional_variance: f64,
    pub magnetic_functional_variance_relative_error: f64,
    pub weighted_magnetic_covariance_action_relative_error: f64,
    pub weighted_magnetic_functional_variance: f64,
    pub weighted_magnetic_functional_variance_relative_error: f64,
}

/// Run the cheap split-site experiment for the requested Cartesian refinements.
///
/// The maintained fixed study uses [`C50_SPLIT_INVERSE_SITE_REFINEMENTS`].
/// Larger levels remain explicitly caller-selected because the exact reference
/// materializes dense inverses.
pub fn run_c50_split_inverse_site_study(refinements: &[usize]) -> Result<Vec<SplitInverseSiteRow>> {
    let mut rows = Vec::with_capacity(8 * refinements.len());
    for &refinement in refinements {
        if refinement == 0 {
            return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
                "C50 split-site refinements must be positive".to_string(),
            ));
        }
        rows.extend(run_one_refinement(refinement)?);
    }
    Ok(rows)
}

/// Run the maintained `n = 2, 3, 4` C50-EMP-03 comparison.
pub fn run_default_c50_split_inverse_site_study() -> Result<Vec<SplitInverseSiteRow>> {
    run_c50_split_inverse_site_study(&C50_SPLIT_INVERSE_SITE_REFINEMENTS)
}

fn run_one_refinement(refinement: usize) -> Result<Vec<SplitInverseSiteRow>> {
    let mesh = CartesianMeshInfo::new_unit_scaled(DIMENSION, refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let operators =
        ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, FORM_DEGREE)?;
    let lower = operators
        .incoming()
        .expect("positive-degree exact operators have an incoming branch");
    let lower_projected =
        posthoc_projected_active_inverse(&topology, &metric, 0, lower.lower().boundary_layout())?;
    let recurrence_projected = posthoc_projected_active_inverse(
        &topology,
        &metric,
        FORM_DEGREE,
        operators.state().boundary_layout(),
    )?;

    let alpha_one = MaternParameters::new(MaternAlpha::One, KAPPA, TAU)?;
    let exact_system = build_prior(
        &operators,
        alpha_one,
        SplitInverseRealization::ExactMaterialized,
        SplitInverseRealization::ExactMaterialized,
        &lower_projected,
        &recurrence_projected,
    )?;
    let projected_system = build_prior(
        &operators,
        alpha_one,
        SplitInverseRealization::ProjectedPosthocRestricted,
        SplitInverseRealization::ExactMaterialized,
        &lower_projected,
        &recurrence_projected,
    )?;

    let smooth_full = smooth_polynomial_one_form_cochain(&topology, &coordinates);
    let state_elimination =
        EssentialBoundaryElimination::from_layout(operators.state().boundary_layout().clone())?;
    let smooth_cochain = state_elimination.reduce_state(smooth_full.as_slice())?;
    let smooth_l2_functional =
        LinearMap::new(operators.state().mass().clone())?.apply(&smooth_cochain)?;
    let physical_maps = MagneticFieldMaps3d::from_feec(&topology, &coordinates)?;
    let magnetic_eliminated = state_elimination.eliminate_map(
        physical_maps.magnetic_field.map(),
        &vec![0.0; physical_maps.magnetic_field.map().output_dimension()],
    )?;
    let (magnetic_cell, magnetic_evaluation_point) =
        nearest_cell_barycenter(&topology, &coordinates, PHYSICAL_TARGET)?;
    let magnetic_row = 3 * magnetic_cell + PHYSICAL_COMPONENT;
    let magnetic_functional = magnetic_eliminated
        .reduced()
        .select_outputs(&[magnetic_row])?
        .apply_transpose(&[1.0])?;
    let weighted_magnetic_output_weights = topology
        .cells()
        .handle_iter()
        .flat_map(|cell| {
            let barycenter = cell.coord_simplex(&coordinates).barycenter();
            // Fixed continuum QoI: domain-average of
            // `(x + y z / 2) B_z`, evaluated with the maintained barycenter
            // reconstruction and cell-volume weights.
            let spatial_weight = barycenter[0] + 0.5 * barycenter[1] * barycenter[2];
            let volume_weight = physical_maps.vector_rms_weights[3 * cell.kidx()];
            [0.0, 0.0, volume_weight * spatial_weight]
        })
        .collect::<Vec<_>>();
    let weighted_magnetic_functional = magnetic_eliminated
        .reduced()
        .apply_transpose(&weighted_magnetic_output_weights)?;

    let choices = [
        (
            SplitInverseRealization::ExactMaterialized,
            SplitInverseRealization::ExactMaterialized,
        ),
        (
            SplitInverseRealization::ProjectedPosthocRestricted,
            SplitInverseRealization::ExactMaterialized,
        ),
        (
            SplitInverseRealization::ExactMaterialized,
            SplitInverseRealization::ProjectedPosthocRestricted,
        ),
        (
            SplitInverseRealization::ProjectedPosthocRestricted,
            SplitInverseRealization::ProjectedPosthocRestricted,
        ),
    ];
    let mut rows = Vec::with_capacity(2 * choices.len());
    for alpha in [MaternAlpha::Two, MaternAlpha::Three] {
        let parameters = MaternParameters::new(alpha, KAPPA, TAU)?;
        let mut priors = Vec::with_capacity(choices.len());
        for (lower_choice, recurrence_choice) in choices {
            priors.push((
                lower_choice,
                recurrence_choice,
                build_prior(
                    &operators,
                    parameters,
                    lower_choice,
                    recurrence_choice,
                    &lower_projected,
                    &recurrence_projected,
                )?,
            ));
        }

        let exact_precision = priors[0].2.precision().clone();
        let mut exact_factor = priors[0].2.factor()?;
        let exact_l2_action = exact_factor.covariance_action(&smooth_l2_functional)?;
        let exact_cochain_action = exact_factor.covariance_action(&smooth_cochain)?;
        let exact_magnetic_action = exact_factor.covariance_action(&magnetic_functional)?;
        let exact_weighted_magnetic_action =
            exact_factor.covariance_action(&weighted_magnetic_functional)?;
        let exact_l2_variance = dot(&smooth_l2_functional, &exact_l2_action)?;
        let exact_cochain_variance = dot(&smooth_cochain, &exact_cochain_action)?;
        let exact_magnetic_variance = dot(&magnetic_functional, &exact_magnetic_action)?;
        let exact_weighted_magnetic_variance = dot(
            &weighted_magnetic_functional,
            &exact_weighted_magnetic_action,
        )?;

        for (lower_choice, recurrence_choice, prior) in priors {
            let system = if lower_choice == SplitInverseRealization::ExactMaterialized {
                &exact_system
            } else {
                &projected_system
            };
            let mut factor = prior.factor()?;
            let l2_action = factor.covariance_action(&smooth_l2_functional)?;
            let cochain_action = factor.covariance_action(&smooth_cochain)?;
            let magnetic_action = factor.covariance_action(&magnetic_functional)?;
            let weighted_magnetic_action =
                factor.covariance_action(&weighted_magnetic_functional)?;
            let l2_variance = dot(&smooth_l2_functional, &l2_action)?;
            let cochain_variance = dot(&smooth_cochain, &cochain_action)?;
            let magnetic_variance = dot(&magnetic_functional, &magnetic_action)?;
            let weighted_magnetic_variance =
                dot(&weighted_magnetic_functional, &weighted_magnetic_action)?;
            let metadata = prior
                .matern_metadata()
                .expect("Matérn builder records metadata");
            let certificate = relative_precision_certificate(
                prior.precision(),
                &exact_precision,
                DEFAULT_MAXIMUM_EXACT_DIMENSION,
            )?;
            rows.push(SplitInverseSiteRow {
                nboxes_per_dimension: refinement,
                state_dimension: operators.state().dimension(),
                lower_dimension: lower.lower().dimension(),
                alpha,
                lower_adjoint_inverse: lower_choice,
                recurrence_inverse: recurrence_choice,
                representation: metadata.representation(),
                normalization: metadata.normalization(),
                lower_projected_inverse_nnz: lower_projected.nnz(),
                recurrence_projected_inverse_nnz: recurrence_projected.nnz(),
                system_relative_frobenius_error: relative_sparse_frobenius_error(
                    system.precision(),
                    exact_system.precision(),
                )?,
                precision_relative_frobenius_error: relative_sparse_frobenius_error(
                    prior.precision(),
                    &exact_precision,
                )?,
                precision_relative_spectral_error: certificate.relative_error,
                minimum_precision_ratio: certificate.minimum_precision_ratio,
                maximum_precision_ratio: certificate.maximum_precision_ratio,
                minimum_covariance_ratio: certificate.minimum_covariance_ratio,
                maximum_covariance_ratio: certificate.maximum_covariance_ratio,
                has_unit_relative_envelope: certificate.has_unit_relative_envelope(),
                smooth_l2_covariance_action_relative_error: relative_vector_l2_error(
                    &l2_action,
                    &exact_l2_action,
                )?,
                smooth_l2_functional_variance: l2_variance,
                smooth_l2_functional_variance_relative_error: relative_scalar_error(
                    l2_variance,
                    exact_l2_variance,
                )?,
                smooth_cochain_covariance_action_relative_error: relative_vector_l2_error(
                    &cochain_action,
                    &exact_cochain_action,
                )?,
                smooth_cochain_functional_variance: cochain_variance,
                smooth_cochain_functional_variance_relative_error: relative_scalar_error(
                    cochain_variance,
                    exact_cochain_variance,
                )?,
                magnetic_target: PHYSICAL_TARGET,
                magnetic_evaluation_point,
                magnetic_component: PHYSICAL_COMPONENT,
                magnetic_covariance_action_relative_error: relative_vector_l2_error(
                    &magnetic_action,
                    &exact_magnetic_action,
                )?,
                magnetic_functional_variance: magnetic_variance,
                magnetic_functional_variance_relative_error: relative_scalar_error(
                    magnetic_variance,
                    exact_magnetic_variance,
                )?,
                weighted_magnetic_covariance_action_relative_error: relative_vector_l2_error(
                    &weighted_magnetic_action,
                    &exact_weighted_magnetic_action,
                )?,
                weighted_magnetic_functional_variance: weighted_magnetic_variance,
                weighted_magnetic_functional_variance_relative_error: relative_scalar_error(
                    weighted_magnetic_variance,
                    exact_weighted_magnetic_variance,
                )?,
            });
        }
    }
    Ok(rows)
}

fn build_prior(
    operators: &ExactFormOperators,
    parameters: MaternParameters,
    lower_choice: SplitInverseRealization,
    recurrence_choice: SplitInverseRealization,
    lower_projected: &SparseMat,
    recurrence_projected: &SparseMat,
) -> Result<GaussianPrior> {
    let builder = AdjacentMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .maximum_materialized_dimension(DEFAULT_MAXIMUM_EXACT_DIMENSION)?;
    let builder = match lower_choice {
        SplitInverseRealization::ExactMaterialized => builder,
        SplitInverseRealization::ProjectedPosthocRestricted => builder.lower_adjoint_inverse(
            EliminatedMassInversePolicy::Provided(lower_projected.clone()),
        ),
    };
    let builder = match recurrence_choice {
        SplitInverseRealization::ExactMaterialized => builder,
        SplitInverseRealization::ProjectedPosthocRestricted => builder.recurrence_inverse(
            EliminatedMassInversePolicy::Provided(recurrence_projected.clone()),
        ),
    };
    builder.build()
}

fn posthoc_projected_active_inverse(
    topology: &Complex,
    metric: &MeshLengths,
    degree: usize,
    layout: &feec_gmrf::operator::BoundaryLayout,
) -> Result<SparseMat> {
    let full = FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_for_grade(
        topology, metric, degree,
    ));
    let full = feec_gmrf::operator::sparse_mat_from_feec_csr(&full);
    EssentialBoundaryElimination::from_layout(layout.clone())?.reduce_square(&full)
}

fn dot(lhs: &[f64], rhs: &[f64]) -> Result<f64> {
    if lhs.len() != rhs.len() {
        return Err(feec_gmrf::FeecGmrfError::Dimension(format!(
            "functional and covariance action lengths differ: {} and {}",
            lhs.len(),
            rhs.len()
        )));
    }
    Ok(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
}

fn nearest_cell_barycenter(
    topology: &Complex,
    coordinates: &manifold::geometry::coord::mesh::MeshCoords,
    target: [f64; 3],
) -> Result<(usize, [f64; 3])> {
    topology
        .cells()
        .handle_iter()
        .map(|cell| {
            let barycenter = cell.coord_simplex(coordinates).barycenter();
            let point = [barycenter[0], barycenter[1], barycenter[2]];
            let distance_squared = point
                .iter()
                .zip(target)
                .map(|(coordinate, target)| (coordinate - target).powi(2))
                .sum::<f64>();
            (cell.kidx(), point, distance_squared)
        })
        .min_by(|lhs, rhs| lhs.2.total_cmp(&rhs.2))
        .map(|(index, point, _)| (index, point))
        .ok_or_else(|| {
            feec_gmrf::FeecGmrfError::InvalidParameter(
                "C50 physical functional requires at least one mesh cell".to_string(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_ablation_reports_all_four_inverse_site_combinations_without_rescaling() {
        let rows = run_default_c50_split_inverse_site_study().unwrap();
        assert_eq!(rows.len(), 24);
        let exact = &rows[0];
        assert_eq!(
            (exact.lower_adjoint_inverse, exact.recurrence_inverse),
            (
                SplitInverseRealization::ExactMaterialized,
                SplitInverseRealization::ExactMaterialized
            )
        );
        assert_eq!(exact.system_relative_frobenius_error, 0.0);
        assert_eq!(exact.precision_relative_frobenius_error, 0.0);
        assert!(exact.precision_relative_spectral_error <= 1.0e-12);
        assert_eq!(rows[8].nboxes_per_dimension, 3);
        assert_eq!(rows[16].nboxes_per_dimension, 4);
        for row in rows {
            assert_eq!(row.normalization, PriorNormalization::None);
            assert!(row.system_relative_frobenius_error.is_finite());
            assert!(row.precision_relative_frobenius_error.is_finite());
            assert!(row.precision_relative_spectral_error.is_finite());
            assert!(row.minimum_precision_ratio > 0.0);
            assert!(row.maximum_precision_ratio > 0.0);
            assert!(row.smooth_l2_covariance_action_relative_error.is_finite());
            assert!(row.smooth_l2_functional_variance > 0.0);
            assert!(row
                .smooth_cochain_covariance_action_relative_error
                .is_finite());
            assert!(row.smooth_cochain_functional_variance > 0.0);
            assert_eq!(row.magnetic_target, PHYSICAL_TARGET);
            assert_eq!(row.magnetic_component, PHYSICAL_COMPONENT);
            assert!(row.magnetic_covariance_action_relative_error.is_finite());
            assert!(row.magnetic_functional_variance > 0.0);
            assert!(row
                .weighted_magnetic_covariance_action_relative_error
                .is_finite());
            assert!(row.weighted_magnetic_functional_variance > 0.0);
            if row.lower_adjoint_inverse == SplitInverseRealization::ProjectedPosthocRestricted
                && row.recurrence_inverse == SplitInverseRealization::ExactMaterialized
            {
                // The canonical magnetic map begins with D1, so it annihilates
                // the incoming exact branch affected by the lower-adjoint
                // inverse. This is a structural identity, not a fitted result.
                assert!(row.magnetic_functional_variance_relative_error <= 1.0e-10);
                assert!(row.weighted_magnetic_functional_variance_relative_error <= 1.0e-10);
            }
        }
    }
}
