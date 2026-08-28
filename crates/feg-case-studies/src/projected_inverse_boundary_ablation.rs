//! Cheap ablation of boundary ordering for the projected sparse inverse.
//!
//! This study isolates one question from the submitted implementation: should
//! homogeneous essential coefficients be removed before the auxiliary
//! full-polynomial vertex blocks are inverted, or should the already assembled
//! full-space inverse simply be restricted afterwards?  It uses only the
//! refinements `n = 1, 2, 4` and an exact reduced mixed Schur complement as the
//! same-mesh reference.

use common::linalg::{
    faer::FaerCholesky,
    nalgebra::{CsrMatrix, Matrix, Vector},
};
use formoniq::{
    problems::reduced_linear::{
        assemble_reduced_hodge_operators, MassInverseApproximation, ReducedHodgeOperators,
    },
    reduction::{EssentialBoundarySpec, PrescribedDof},
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    topology::{complex::Complex, handle::KSimplexIdx},
};

const DIMENSION: usize = 3;
const FORM_DEGREE: usize = 2;
const REFINEMENTS: [usize; 3] = [1, 2, 4];
const EXACT_AUXILIARY_LIMIT: usize = 4_096;

/// One same-mesh comparison of the two projected-inverse boundary orderings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedInverseBoundaryAblationRow {
    pub nboxes_per_dimension: usize,
    pub active_auxiliary_dofs: usize,
    pub active_state_dofs: usize,
    pub posthoc_inverse_nnz: usize,
    pub boundary_first_inverse_nnz: usize,
    pub exact_inverse_nnz: usize,
    pub posthoc_inverse_spd: bool,
    pub boundary_first_inverse_spd: bool,
    pub exact_inverse_spd: bool,
    pub posthoc_inverse_vertex_local: bool,
    pub boundary_first_inverse_vertex_local: bool,
    pub exact_inverse_vertex_local: bool,
    pub posthoc_inverse_matrix_relative_error: f64,
    pub boundary_first_inverse_matrix_relative_error: f64,
    pub posthoc_inverse_action_relative_error: f64,
    pub boundary_first_inverse_action_relative_error: f64,
    pub posthoc_schur_matrix_relative_error: f64,
    pub boundary_first_schur_matrix_relative_error: f64,
    pub posthoc_solution_mass_relative_error: f64,
    pub boundary_first_solution_mass_relative_error: f64,
}

/// Run the fixed cheap `n = 1, 2, 4` boundary-ordering ablation.
pub fn run_projected_inverse_boundary_ablation(
) -> Result<Vec<ProjectedInverseBoundaryAblationRow>, String> {
    REFINEMENTS.into_iter().map(run_one_refinement).collect()
}

fn run_one_refinement(
    nboxes_per_dimension: usize,
) -> Result<ProjectedInverseBoundaryAblationRow, String> {
    let mesh = CartesianMeshInfo::new_unit(DIMENSION, nboxes_per_dimension);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = full_relative_boundary(&topology);

    let posthoc = assemble_reduced_hodge_operators(
        &topology,
        &metric,
        Some(&coords),
        FORM_DEGREE,
        &boundary,
        Some(MassInverseApproximation::WhitneyProjected),
        MassInverseApproximation::WhitneyProjected,
    )?;
    let boundary_first = assemble_reduced_hodge_operators(
        &topology,
        &metric,
        Some(&coords),
        FORM_DEGREE,
        &boundary,
        Some(MassInverseApproximation::WhitneyProjectedBoundaryFirst),
        MassInverseApproximation::WhitneyProjected,
    )?;
    let exact = assemble_reduced_hodge_operators(
        &topology,
        &metric,
        Some(&coords),
        FORM_DEGREE,
        &boundary,
        Some(MassInverseApproximation::ExactReducedConsistent {
            maximum_dimension: EXACT_AUXILIARY_LIMIT,
        }),
        MassInverseApproximation::WhitneyProjected,
    )?;

    ensure_same_layouts(&posthoc, &boundary_first, &exact)?;
    let posthoc_inverse = posthoc
        .lower_mass_inverse
        .as_ref()
        .ok_or_else(|| "post-hoc assembly omitted its lower mass inverse".to_string())?;
    let boundary_first_inverse = boundary_first
        .lower_mass_inverse
        .as_ref()
        .ok_or_else(|| "boundary-first assembly omitted its lower mass inverse".to_string())?;
    let exact_inverse = exact
        .lower_mass_inverse
        .as_ref()
        .ok_or_else(|| "exact assembly omitted its lower mass inverse".to_string())?;

    let posthoc_inverse_dense = Matrix::from(posthoc_inverse);
    let boundary_first_inverse_dense = Matrix::from(boundary_first_inverse);
    let exact_inverse_dense = Matrix::from(exact_inverse);
    let posthoc_operator_dense = Matrix::from(&posthoc.laplacian);
    let boundary_first_operator_dense = Matrix::from(&boundary_first.laplacian);
    let exact_operator_dense = Matrix::from(&exact.laplacian);

    let auxiliary_probe = deterministic_probe(exact_inverse.nrows());
    let exact_inverse_action = &exact_inverse_dense * &auxiliary_probe;
    let posthoc_inverse_action = &posthoc_inverse_dense * &auxiliary_probe;
    let boundary_first_inverse_action = &boundary_first_inverse_dense * &auxiliary_probe;

    let rhs = deterministic_probe(exact.laplacian.nrows());
    let exact_solution = FaerCholesky::new(exact.laplacian.clone()).solve(&rhs);
    let posthoc_solution = FaerCholesky::new(posthoc.laplacian.clone()).solve(&rhs);
    let boundary_first_solution = FaerCholesky::new(boundary_first.laplacian.clone()).solve(&rhs);
    let state_mass = Matrix::from(&exact.mass);

    let active_auxiliary = boundary
        .auxiliary
        .iter()
        .map(|entry| entry.index)
        .collect::<std::collections::BTreeSet<_>>();
    let auxiliary_active_dofs = (0..topology.nsimplices(FORM_DEGREE - 1))
        .filter(|index| !active_auxiliary.contains(index))
        .collect::<Vec<_>>();

    Ok(ProjectedInverseBoundaryAblationRow {
        nboxes_per_dimension,
        active_auxiliary_dofs: exact_inverse.nrows(),
        active_state_dofs: exact.laplacian.nrows(),
        posthoc_inverse_nnz: posthoc_inverse.nnz(),
        boundary_first_inverse_nnz: boundary_first_inverse.nnz(),
        exact_inverse_nnz: exact_inverse.nnz(),
        posthoc_inverse_spd: posthoc_inverse_dense.clone().cholesky().is_some(),
        boundary_first_inverse_spd: boundary_first_inverse_dense.clone().cholesky().is_some(),
        exact_inverse_spd: exact_inverse_dense.clone().cholesky().is_some(),
        posthoc_inverse_vertex_local: inverse_is_vertex_local(
            &topology,
            posthoc_inverse,
            &auxiliary_active_dofs,
        ),
        boundary_first_inverse_vertex_local: inverse_is_vertex_local(
            &topology,
            boundary_first_inverse,
            &auxiliary_active_dofs,
        ),
        exact_inverse_vertex_local: inverse_is_vertex_local(
            &topology,
            exact_inverse,
            &auxiliary_active_dofs,
        ),
        posthoc_inverse_matrix_relative_error: relative_frobenius(
            &posthoc_inverse_dense,
            &exact_inverse_dense,
        ),
        boundary_first_inverse_matrix_relative_error: relative_frobenius(
            &boundary_first_inverse_dense,
            &exact_inverse_dense,
        ),
        posthoc_inverse_action_relative_error: relative_vector_error(
            &posthoc_inverse_action,
            &exact_inverse_action,
        ),
        boundary_first_inverse_action_relative_error: relative_vector_error(
            &boundary_first_inverse_action,
            &exact_inverse_action,
        ),
        posthoc_schur_matrix_relative_error: relative_frobenius(
            &posthoc_operator_dense,
            &exact_operator_dense,
        ),
        boundary_first_schur_matrix_relative_error: relative_frobenius(
            &boundary_first_operator_dense,
            &exact_operator_dense,
        ),
        posthoc_solution_mass_relative_error: relative_mass_error(
            &posthoc_solution,
            &exact_solution,
            &state_mass,
        ),
        boundary_first_solution_mass_relative_error: relative_mass_error(
            &boundary_first_solution,
            &exact_solution,
            &state_mass,
        ),
    })
}

fn full_relative_boundary(topology: &Complex) -> EssentialBoundarySpec {
    EssentialBoundarySpec {
        state: homogeneous_boundary_dofs(topology, FORM_DEGREE),
        auxiliary: homogeneous_boundary_dofs(topology, FORM_DEGREE - 1),
    }
}

fn homogeneous_boundary_dofs(topology: &Complex, grade: usize) -> Vec<PrescribedDof> {
    topology
        .boundary_subcomplex_simplices(grade)
        .into_iter()
        .map(|simplex| PrescribedDof {
            index: simplex.kidx,
            value: 0.0,
        })
        .collect()
}

fn ensure_same_layouts(
    posthoc: &ReducedHodgeOperators,
    boundary_first: &ReducedHodgeOperators,
    exact: &ReducedHodgeOperators,
) -> Result<(), String> {
    if posthoc.layout != boundary_first.layout || posthoc.layout != exact.layout {
        return Err("boundary ablation strategies produced different state layouts".to_string());
    }
    Ok(())
}

fn deterministic_probe(dimension: usize) -> Vector {
    Vector::from_iterator(
        dimension,
        (0..dimension).map(|index| {
            let coordinate = (index + 1) as f64;
            (0.618_033_988_749_894_8 * coordinate).sin()
                + (0.414_213_562_373_095_03 * coordinate).cos()
        }),
    )
}

fn relative_frobenius(candidate: &Matrix, reference: &Matrix) -> f64 {
    (candidate - reference).norm() / reference.norm().max(f64::EPSILON)
}

fn relative_vector_error(candidate: &Vector, reference: &Vector) -> f64 {
    (candidate - reference).norm() / reference.norm().max(f64::EPSILON)
}

fn relative_mass_error(candidate: &Vector, reference: &Vector, mass: &Matrix) -> f64 {
    let difference = candidate - reference;
    let numerator = difference.dot(&(mass * &difference)).max(0.0).sqrt();
    let denominator = reference
        .dot(&(mass * reference))
        .max(0.0)
        .sqrt()
        .max(f64::EPSILON);
    numerator / denominator
}

fn inverse_is_vertex_local(
    topology: &Complex,
    inverse: &CsrMatrix,
    active_full_dofs: &[usize],
) -> bool {
    inverse.triplet_iter().all(|(row, col, value)| {
        if *value == 0.0 {
            return true;
        }
        simplices_share_vertex(
            topology,
            FORM_DEGREE - 1,
            active_full_dofs[row],
            active_full_dofs[col],
        )
    })
}

fn simplices_share_vertex(
    topology: &Complex,
    grade: usize,
    lhs: KSimplexIdx,
    rhs: KSimplexIdx,
) -> bool {
    let lhs = topology.skeleton(grade).handle_by_kidx(lhs);
    let rhs = topology.skeleton(grade).handle_by_kidx(rhs);
    lhs.iter().any(|vertex| rhs.contains(vertex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheap_ablation_reports_all_three_fixed_refinements() {
        let rows = run_projected_inverse_boundary_ablation().unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.nboxes_per_dimension)
                .collect::<Vec<_>>(),
            REFINEMENTS
        );
        for row in rows {
            assert!(row.posthoc_inverse_spd);
            assert!(row.boundary_first_inverse_spd);
            assert!(row.exact_inverse_spd);
            assert!(row.posthoc_inverse_vertex_local);
            assert!(row.boundary_first_inverse_vertex_local);
            assert!(row.posthoc_inverse_matrix_relative_error.is_finite());
            assert!(row.boundary_first_inverse_matrix_relative_error.is_finite());
            assert!(row.posthoc_solution_mass_relative_error.is_finite());
            assert!(row.boundary_first_solution_mass_relative_error.is_finite());
        }
    }
}
