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
    relative_sparse_frobenius_error, AdjacentMaternPriorBuilder, EliminatedMassInversePolicy,
    EssentialBoundaryElimination, ExactFormOperators, GaussianPrior, LinearMap, MaternAlpha,
    MaternParameters, MaternRepresentation, PriorNormalization, Result, SparseMat,
};
use formoniq::assemble::assemble_whitney_projected_sparse_inverse_galmat_for_grade;
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::metric::mesh::MeshLengths,
    topology::complex::Complex,
};

const DIMENSION: usize = 3;
const FORM_DEGREE: usize = 1;
const KAPPA: f64 = 4.0;
const TAU: f64 = 1.0;
const DEFAULT_MAXIMUM_EXACT_DIMENSION: usize = 1_024;

/// Fixed cheap refinements retained for C50-EMP-03.
pub const C50_SPLIT_INVERSE_SITE_REFINEMENTS: [usize; 2] = [2, 4];

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
    pub lower_adjoint_inverse: SplitInverseRealization,
    pub recurrence_inverse: SplitInverseRealization,
    pub representation: MaternRepresentation,
    pub normalization: PriorNormalization,
    pub lower_projected_inverse_nnz: usize,
    pub recurrence_projected_inverse_nnz: usize,
    pub system_relative_frobenius_error: f64,
    pub precision_relative_frobenius_error: f64,
    pub smooth_l2_covariance_action_relative_error: f64,
    pub smooth_l2_functional_variance: f64,
    pub smooth_l2_functional_variance_relative_error: f64,
    pub smooth_cochain_covariance_action_relative_error: f64,
    pub smooth_cochain_functional_variance: f64,
    pub smooth_cochain_functional_variance_relative_error: f64,
}

/// Run the cheap split-site experiment for the requested Cartesian refinements.
///
/// The maintained fixed study uses [`C50_SPLIT_INVERSE_SITE_REFINEMENTS`].
/// Larger levels remain explicitly caller-selected because the exact reference
/// materializes dense inverses.
pub fn run_c50_split_inverse_site_study(refinements: &[usize]) -> Result<Vec<SplitInverseSiteRow>> {
    let mut rows = Vec::with_capacity(4 * refinements.len());
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

/// Run the maintained `n = 2, 4` C50-EMP-03 comparison.
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

    let alpha_two = MaternParameters::new(MaternAlpha::Two, KAPPA, TAU)?;
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
    let mut priors = Vec::with_capacity(choices.len());
    for (lower_choice, recurrence_choice) in choices {
        priors.push((
            lower_choice,
            recurrence_choice,
            build_prior(
                &operators,
                alpha_two,
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
    let exact_l2_variance = dot(&smooth_l2_functional, &exact_l2_action)?;
    let exact_cochain_variance = dot(&smooth_cochain, &exact_cochain_action)?;

    let mut rows = Vec::with_capacity(priors.len());
    for (lower_choice, recurrence_choice, prior) in priors {
        let system = if lower_choice == SplitInverseRealization::ExactMaterialized {
            &exact_system
        } else {
            &projected_system
        };
        let mut factor = prior.factor()?;
        let l2_action = factor.covariance_action(&smooth_l2_functional)?;
        let cochain_action = factor.covariance_action(&smooth_cochain)?;
        let l2_variance = dot(&smooth_l2_functional, &l2_action)?;
        let cochain_variance = dot(&smooth_cochain, &cochain_action)?;
        let metadata = prior
            .matern_metadata()
            .expect("Matérn builder records metadata");
        rows.push(SplitInverseSiteRow {
            nboxes_per_dimension: refinement,
            state_dimension: operators.state().dimension(),
            lower_dimension: lower.lower().dimension(),
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
            smooth_l2_covariance_action_relative_error: relative_vector_error(
                &l2_action,
                &exact_l2_action,
            )?,
            smooth_l2_functional_variance: l2_variance,
            smooth_l2_functional_variance_relative_error: relative_scalar_error(
                l2_variance,
                exact_l2_variance,
            )?,
            smooth_cochain_covariance_action_relative_error: relative_vector_error(
                &cochain_action,
                &exact_cochain_action,
            )?,
            smooth_cochain_functional_variance: cochain_variance,
            smooth_cochain_functional_variance_relative_error: relative_scalar_error(
                cochain_variance,
                exact_cochain_variance,
            )?,
        });
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

fn relative_vector_error(candidate: &[f64], reference: &[f64]) -> Result<f64> {
    if candidate.len() != reference.len() {
        return Err(feec_gmrf::FeecGmrfError::Dimension(format!(
            "covariance action lengths differ: {} and {}",
            candidate.len(),
            reference.len()
        )));
    }
    let numerator = candidate
        .iter()
        .zip(reference)
        .map(|(candidate, reference)| (candidate - reference).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if denominator == 0.0 {
        return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
            "relative covariance-action error requires a nonzero reference".to_string(),
        ));
    }
    Ok(numerator / denominator)
}

fn relative_scalar_error(candidate: f64, reference: f64) -> Result<f64> {
    if !candidate.is_finite() || !reference.is_finite() || reference == 0.0 {
        return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
            "relative variance error requires finite values and a nonzero reference".to_string(),
        ));
    }
    Ok((candidate - reference).abs() / reference.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_ablation_reports_all_four_inverse_site_combinations_without_rescaling() {
        let rows = run_default_c50_split_inverse_site_study().unwrap();
        assert_eq!(rows.len(), 8);
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
        assert_eq!(rows[4].nboxes_per_dimension, 4);
        for row in rows {
            assert_eq!(row.normalization, PriorNormalization::None);
            assert!(row.system_relative_frobenius_error.is_finite());
            assert!(row.precision_relative_frobenius_error.is_finite());
            assert!(row.smooth_l2_covariance_action_relative_error.is_finite());
            assert!(row.smooth_l2_functional_variance > 0.0);
            assert!(row
                .smooth_cochain_covariance_action_relative_error
                .is_finite());
            assert!(row.smooth_cochain_functional_variance > 0.0);
        }
    }
}
