//! Linear curl-only magnetostatics with conforming cellwise materials.
//!
//! Uses the existing Whitney Hodge element integrator and boundary-first
//! projected inverse mass. No gauge penalty is added to the physical operator.

use crate::boundary::{EssentialBoundaryConditions, EssentialBoundaryElimination};
use crate::model::GaussianNoise;
use crate::operator::{LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::{CsrMatrix, Matrix};
use ddf::ManifoldComplexExt;
use formoniq::{
    assemble::{assemble_galmat, assemble_galmat_coord_aware},
    operators::{CoordAwareElMatProvider, ElMatProvider, ElMatProviderBase, HodgeMassElmat},
    reduction::{DofLayout, PrescribedDof},
};
use manifold::{
    geometry::metric::{mesh::MeshLengths, simplex::SimplexLengths},
    topology::{complex::Complex, simplex::Simplex},
};
use std::collections::BTreeMap;

struct CellwiseMass<'a> {
    element: HodgeMassElmat<'a>,
    weights: BTreeMap<Vec<usize>, f64>,
}
impl ElMatProviderBase for CellwiseMass<'_> {
    fn row_grade(&self) -> usize {
        2
    }
    fn col_grade(&self) -> usize {
        2
    }
}
impl CoordAwareElMatProvider for CellwiseMass<'_> {
    fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> Matrix {
        self.element.eval(geometry) * self.weights[&cell.vertices]
    }
}

/// Full physical operator and homogeneous correction reduction.
#[derive(Debug, Clone)]
pub struct LinearMagnetostaticSystem {
    pub full_operator: LinearMap,
    pub correction_operator: LinearMap,
    pub free_test_selector: LinearMap,
    pub boundary: EssentialBoundaryElimination,
    pub state_mass: SparseMat,
    /// Projected inverse constructed on the already reduced relative space.
    pub residual_mass_inverse: SparseMat,
}

impl LinearMagnetostaticSystem {
    /// Assemble `D1^T M2(nu) D1`, for strictly positive reluctivities in
    /// `topology.cells()` order. Region labels should come from the mesh;
    /// curved-interface classification at quadrature points is unnecessary.
    pub fn from_cell_reluctivities(
        topology: &Complex,
        metric: &MeshLengths,
        reluctivities: &[f64],
    ) -> Result<Self> {
        if topology.dim() != 3 || reluctivities.len() != topology.nsimplices(3) {
            return Err(FeecGmrfError::Dimension(
                "magnetostatics requires one reluctivity per 3D cell".into(),
            ));
        }
        if reluctivities.iter().any(|nu| !nu.is_finite() || *nu <= 0.0) {
            return Err(FeecGmrfError::InvalidParameter(
                "cell reluctivities must be finite and positive".into(),
            ));
        }
        let weights = topology
            .cells()
            .iter()
            .zip(reluctivities)
            .map(|(c, nu)| (c.vertices.clone(), *nu))
            .collect();
        let m2 = CsrMatrix::from(&assemble_galmat_coord_aware(
            topology,
            metric,
            CellwiseMass {
                element: HodgeMassElmat::new(3, 2),
                weights,
            },
        ));
        let d1 = CsrMatrix::from(&topology.exterior_derivative_operator(1));
        let full_operator = LinearMap::from_feec_csr(&(d1.transpose() * m2 * d1))?;
        let boundary = EssentialBoundaryConditions::homogeneous_on_boundary(topology, 1)?
            .eliminate(topology.nsimplices(1))?;
        let free_test_selector =
            LinearMap::selector(topology.nsimplices(1), boundary.layout().active_dofs())?;
        let p = boundary.active_to_full();
        let correction_operator = free_test_selector.compose(&full_operator)?.compose(p)?;
        let m1 = LinearMap::from_feec_csr(&CsrMatrix::from(&assemble_galmat(
            topology,
            metric,
            HodgeMassElmat::new(3, 1),
        )))?;
        let state_mass = free_test_selector.compose(&m1)?.compose(p)?.into_matrix();
        let layout = DofLayout::from_parts(
            topology.nsimplices(1),
            boundary.layout().active_dofs().to_vec(),
            boundary
                .layout()
                .fixed_dofs()
                .iter()
                .map(|(index, value)| PrescribedDof {
                    index: *index,
                    value: *value,
                })
                .collect(),
        )
        .map_err(FeecGmrfError::Assembly)?;
        let inverse =
            formoniq::assemble::assemble_reduced_whitney_projected_sparse_inverse_galmat_for_grade(
                topology, metric, 1, &layout,
            )
            .map_err(FeecGmrfError::Assembly)?;
        let residual_mass_inverse =
            feg_infer::sparse::feec_csr_to_core_triplet(&CsrMatrix::from(&inverse));
        Ok(Self {
            full_operator,
            correction_operator,
            free_test_selector,
            boundary,
            state_mass,
            residual_mass_inverse,
        })
    }

    /// Weak test rows applied to an explicit total-field lifting `G`.
    pub fn lifting_residual(&self, lifting: &LinearMap) -> Result<LinearMap> {
        self.free_test_selector
            .compose(&self.full_operator)?
            .compose(lifting)
    }

    /// Precision `W = sigma^-2 R_h` for the weak residual, with `R_h` the
    /// boundary-first projected inverse of the reference 1-form mass.
    /// `sigma` refers to this dual weak norm, not independent rowwise noise.
    pub fn residual_noise(&self, sigma: f64) -> Result<GaussianNoise> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "PDE residual scale must be positive".into(),
            ));
        }
        GaussianNoise::precision(
            self.residual_mass_inverse
                .clone()
                .scaled(1.0 / (sigma * sigma)),
        )
    }

    /// Deterministic control solve with a compatible curl-only right hand side.
    ///
    /// CG starts from zero on the semidefinite curl system. It selects a
    /// coefficient representative without adding a physical gauge penalty.
    /// Only `dA` is unique. A failed tolerance check is reported as an error;
    /// the system is never shifted to obtain a result.
    pub fn solve_lifted_deterministic(
        &self,
        lifting: &LinearMap,
        input: &[f64],
        tolerance: f64,
    ) -> Result<Vec<f64>> {
        if !tolerance.is_finite() || tolerance <= 0.0 || input.iter().any(|x| !x.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "deterministic tolerance must be positive and input finite".into(),
            ));
        }
        let forcing = self.lifting_residual(lifting)?.apply(input)?;
        let mut rhs = gmrf_core::Vector::from_iterator(forcing.len(), forcing.iter().map(|x| -x));
        // In the uniform-material control the incident lifting already solves
        // the PDE, up to assembly roundoff. Test the residual in its original
        // units before scaling; do not amplify a roundoff-only RHS.
        if rhs.norm() <= tolerance {
            return lifting.apply(input);
        }
        // The shared CG implementation has an absolute machine-epsilon
        // breakdown guard on inner products. Scale both sides equally so it
        // does not stop above the requested residual tolerance. This changes
        // neither the nullspace nor the solution, and adds no diagonal shift.
        let equation_scale = 1.0e6 / rhs.norm().max(1.0);
        rhs = rhs * equation_scale;
        let matrix = feg_infer::sparse::core_triplet_to_gmrf(
            &self
                .correction_operator
                .matrix()
                .clone()
                .scaled(equation_scale),
        );
        let mut solver = gmrf_core::Solver::new(gmrf_core::SolverConfig {
            algorithm: gmrf_core::SolverAlgorithm::Iterative(
                gmrf_core::IterativeMethod::ConjugateGradient,
            ),
            tolerance,
            max_iterations: 30 * self.correction_operator.input_dimension(),
            preconditioner: gmrf_core::PreconditionerKind::Jacobi,
        });
        let state = solver
            .solve_matrix(&matrix, &rhs)
            .map_err(|e| FeecGmrfError::Inference(e.to_string()))?;
        let mut total = self.boundary.active_to_full().apply(state.as_slice())?;
        let lift = lifting.apply(input)?;
        for (q, g) in total.iter_mut().zip(lift) {
            *q += g;
        }
        Ok(total)
    }
}
