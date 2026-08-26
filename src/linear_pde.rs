//! State-only models for boundary-reduced linear FEEC systems.

use crate::boundary::EssentialBoundaryElimination;
use crate::model::{
    DerivedQuantity, LinearConstraint, LinearGaussianModelBuilder, LinearObservation,
};
use crate::operator::{BoundaryLayout, FormDegree, FormOperators, LinearMap};
use crate::physical::PhysicalMap;
use crate::prior::{GaussianPrior, MassInversePolicy, MaternPriorBuilder};
use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::Vector as FeecVector;
use formoniq::problems::reduced_linear::ReducedLinearPdeAssembly;
use gmrf_core::{Solver, Vector as GmrfVector};

/// Boundary-reduced affine linear PDE residual `K z + bias`.
#[derive(Debug, Clone)]
pub struct LinearPdeSystem {
    assembly: ReducedLinearPdeAssembly,
    form_space: Option<(FormDegree, usize)>,
}

impl LinearPdeSystem {
    /// Wrap and validate a FEEC reduced assembly.
    pub fn from_reduced_assembly(assembly: ReducedLinearPdeAssembly) -> Result<Self> {
        let state_dim = assembly.state_dimension();
        let residual_dim = assembly.residual_dimension();
        if state_dim == 0 || residual_dim == 0 {
            return Err(FeecGmrfError::Dimension(
                "a linear PDE system must have non-empty state and residual spaces".to_string(),
            ));
        }
        if assembly.residual_bias.len() != residual_dim {
            return Err(FeecGmrfError::Dimension(format!(
                "residual bias length {} does not match residual dimension {residual_dim}",
                assembly.residual_bias.len()
            )));
        }
        if assembly.state_mass.nrows() != state_dim || assembly.state_mass.ncols() != state_dim {
            return Err(FeecGmrfError::Dimension(
                "reduced state mass must be square on the state space".to_string(),
            ));
        }
        if assembly.layout.reduced_dimension() != state_dim {
            return Err(FeecGmrfError::Dimension(format!(
                "layout has {} active coefficients but system has {state_dim}",
                assembly.layout.reduced_dimension()
            )));
        }
        if let Some(inverse) = &assembly.state_mass_inverse {
            if inverse.nrows() != residual_dim || inverse.ncols() != residual_dim {
                return Err(FeecGmrfError::Dimension(
                    "state mass inverse must be square on the residual space".to_string(),
                ));
            }
        }
        Ok(Self {
            assembly,
            form_space: None,
        })
    }

    /// Attach form-degree metadata required by Matérn prior construction.
    pub fn with_form_space(mut self, degree: usize, complex_dimension: usize) -> Result<Self> {
        self.form_space = Some((
            FormDegree::new(degree, complex_dimension)?,
            complex_dimension,
        ));
        Ok(self)
    }

    /// Fold a reduced deterministic right-hand side into the affine residual.
    ///
    /// Afterwards the residual is `operator * z + boundary_bias - rhs`.
    pub fn with_right_hand_side(mut self, right_hand_side: &[f64]) -> Result<Self> {
        if right_hand_side.len() != self.assembly.residual_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "PDE right-hand side length {} does not match residual dimension {}",
                right_hand_side.len(),
                self.assembly.residual_dimension()
            )));
        }
        self.assembly.residual_bias -= FeecVector::from_column_slice(right_hand_side);
        Ok(self)
    }

    pub fn assembly(&self) -> &ReducedLinearPdeAssembly {
        &self.assembly
    }

    pub fn state_dimension(&self) -> usize {
        self.assembly.state_dimension()
    }

    pub fn cochain_dimension(&self) -> usize {
        self.assembly.layout.full_dimension
    }

    /// State norm induced by the reduced FEEC mass matrix.
    pub fn state_l2_norm(&self, state: &[f64]) -> Result<f64> {
        if state.len() != self.state_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "state length {} does not match PDE state dimension {}",
                state.len(),
                self.state_dimension()
            )));
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "state coefficients must be finite".to_string(),
            ));
        }
        let mass_state = LinearMap::new(feg_infer::sparse::feec_csr_to_core_triplet(
            &self.assembly.state_mass,
        ))?
        .apply(state)?;
        let squared_norm = state
            .iter()
            .zip(mass_state)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        if !squared_norm.is_finite() || squared_norm < -1.0e-12 {
            return Err(FeecGmrfError::Inference(
                "state mass matrix produced an invalid squared norm".to_string(),
            ));
        }
        Ok(squared_norm.max(0.0).sqrt())
    }

    /// Relative reduced-state L2 gap, normalized by the reference state norm.
    pub fn relative_state_l2_error(&self, state: &[f64], reference: &[f64]) -> Result<f64> {
        if state.len() != reference.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "state length {} does not match reference length {}",
                state.len(),
                reference.len()
            )));
        }
        let reference_norm = self.state_l2_norm(reference)?;
        if reference_norm == 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "relative state error requires a non-zero reference".to_string(),
            ));
        }
        let difference = state
            .iter()
            .zip(reference)
            .map(|(value, reference)| value - reference)
            .collect::<Vec<_>>();
        Ok(self.state_l2_norm(&difference)? / reference_norm)
    }

    /// Solve the deterministic affine system `operator * z + bias = 0`.
    pub fn solve_deterministic(&self) -> Result<DeterministicLinearPdeSolution> {
        if self.assembly.operator.nrows() != self.assembly.operator.ncols() {
            return Err(FeecGmrfError::Unsupported(
                "deterministic state solve requires a square reduced PDE operator".to_string(),
            ));
        }
        let operator = feg_infer::sparse::feec_csr_to_gmrf(&self.assembly.operator);
        let rhs = GmrfVector::from_iterator(
            self.assembly.residual_bias.len(),
            self.assembly.residual_bias.iter().map(|value| -*value),
        );
        let active = Solver::default().solve_matrix(&operator, &rhs)?;
        let active = active.iter().copied().collect::<Vec<_>>();
        let cochain = self.boundary_elimination()?.lift_state(&active)?;
        Ok(DeterministicLinearPdeSolution { active, cochain })
    }

    /// Build a Matérn prior from this system's reduced FEEC operators.
    pub fn matern_prior_builder(&self) -> Result<MaternPriorBuilder<'static>> {
        let (degree, complex_dimension) = self.form_space.ok_or_else(|| {
            FeecGmrfError::InvalidParameter(
                "Matérn construction requires LinearPdeSystem::with_form_space".to_string(),
            )
        })?;
        if self.assembly.operator.nrows() != self.assembly.operator.ncols() {
            return Err(FeecGmrfError::Unsupported(
                "Matérn construction requires a square state operator".to_string(),
            ));
        }
        let mass = feg_infer::sparse::feec_csr_to_core_triplet(&self.assembly.state_mass);
        let laplacian = feg_infer::sparse::feec_csr_to_core_triplet(&self.assembly.operator);
        let operators = FormOperators::new(degree, complex_dimension, mass, laplacian)?
            .with_boundary_layout(self.boundary_layout()?)?;
        let inverse = self.assembly.state_mass_inverse.as_ref().ok_or_else(|| {
            FeecGmrfError::Unsupported(
                "Matérn construction requires the reduced projected state-mass inverse".to_string(),
            )
        })?;
        Ok(
            MaternPriorBuilder::from_operators(operators).mass_inverse(
                MassInversePolicy::Provided(feg_infer::sparse::feec_csr_to_core_triplet(inverse)),
            ),
        )
    }

    /// Construct a zero-mean scalar-diagonal prior on active coefficients.
    pub fn diagonal_prior(&self, coefficient_precision: f64) -> Result<GaussianPrior> {
        if !coefficient_precision.is_finite() || coefficient_precision <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "diagonal coefficient precision must be finite and positive".to_string(),
            ));
        }
        let mut prior = GaussianPrior::new(
            vec![0.0; self.state_dimension()],
            crate::operator::SparseMat::diagonal(self.state_dimension(), coefficient_precision),
        )?;
        if let Some((degree, _)) = self.form_space {
            prior = prior.with_form_degree(degree);
        }
        if self.has_boundary_reduction() {
            prior.with_boundary_elimination(self.boundary_elimination()?)
        } else {
            Ok(prior)
        }
    }

    /// Construct a zero-mean FEEC L2-white prior on active coefficients.
    ///
    /// The precision is `precision_scale * M`, where `M` is the reduced state
    /// mass matrix. Spatial structure can then be supplied by a PDE likelihood
    /// or another model term.
    pub fn l2_white_noise_prior(&self, precision_scale: f64) -> Result<GaussianPrior> {
        if !precision_scale.is_finite() || precision_scale <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "L2-white precision scale must be finite and positive".to_string(),
            ));
        }
        let precision = feg_infer::sparse::feec_csr_to_core_triplet(&self.assembly.state_mass)
            .scaled(precision_scale);
        let mut prior = GaussianPrior::new(vec![0.0; self.state_dimension()], precision)?;
        if let Some((degree, _)) = self.form_space {
            prior = prior.with_form_degree(degree);
        }
        if self.has_boundary_reduction() {
            prior.with_boundary_elimination(self.boundary_elimination()?)
        } else {
            Ok(prior)
        }
    }

    /// Return the affine PDE residual as an explicitly named diagnostic.
    pub fn residual_quantity(&self, name: impl Into<String>) -> Result<DerivedQuantity> {
        DerivedQuantity::with_bias(
            name,
            LinearMap::new(feg_infer::sparse::feec_csr_to_core_triplet(
                &self.assembly.operator,
            ))?,
            self.assembly.residual_bias.iter().copied().collect(),
        )
    }

    fn boundary_layout(&self) -> Result<BoundaryLayout> {
        BoundaryLayout::new(
            self.assembly.layout.full_dimension,
            self.assembly.layout.active_dofs.clone(),
            self.assembly
                .layout
                .prescribed_dofs
                .iter()
                .map(|entry| (entry.index, entry.value))
                .collect(),
        )
    }

    fn boundary_elimination(&self) -> Result<EssentialBoundaryElimination> {
        EssentialBoundaryElimination::from_layout(self.boundary_layout()?)
    }

    fn has_boundary_reduction(&self) -> bool {
        !self.assembly.layout.prescribed_dofs.is_empty()
            || self.assembly.layout.active_dofs.len() != self.assembly.layout.full_dimension
    }
}

/// Deterministic solution in both active and full FEEC orderings.
#[derive(Debug, Clone, PartialEq)]
pub struct DeterministicLinearPdeSolution {
    active: Vec<f64>,
    cochain: Vec<f64>,
}

impl DeterministicLinearPdeSolution {
    pub fn active(&self) -> &[f64] {
        &self.active
    }

    pub fn cochain(&self) -> &[f64] {
        &self.cochain
    }
}

/// Noise model for a weak affine PDE residual observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PdeResidualNoise {
    MassWeightedL2StandardDeviation(f64),
}

impl PdeResidualNoise {
    pub fn mass_weighted_l2_standard_deviation(standard_deviation: f64) -> Result<Self> {
        if !standard_deviation.is_finite() || standard_deviation <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "PDE residual standard deviation must be finite and positive".to_string(),
            ));
        }
        Ok(Self::MassWeightedL2StandardDeviation(standard_deviation))
    }
}

/// Builder for a state-only linear PDE model.
pub struct LinearPdeModelBuilder<'a> {
    system: &'a LinearPdeSystem,
    inner: LinearGaussianModelBuilder,
}

impl<'a> LinearPdeModelBuilder<'a> {
    pub fn new(prior: GaussianPrior, system: &'a LinearPdeSystem) -> Result<Self> {
        if prior.dimension() != system.state_dimension()
            || prior.cochain_dimension() != system.cochain_dimension()
        {
            return Err(FeecGmrfError::Dimension(format!(
                "prior dimensions {}/{} do not match PDE dimensions {}/{}",
                prior.dimension(),
                prior.cochain_dimension(),
                system.state_dimension(),
                system.cochain_dimension()
            )));
        }
        if system.has_boundary_reduction() {
            if prior.boundary_elimination() != Some(&system.boundary_elimination()?) {
                return Err(FeecGmrfError::InvalidParameter(
                    "prior and PDE system must carry the same boundary layout".to_string(),
                ));
            }
        } else if let Some(elimination) = prior.boundary_elimination() {
            if elimination != &system.boundary_elimination()? {
                return Err(FeecGmrfError::InvalidParameter(
                    "prior and PDE system must carry the same boundary layout".to_string(),
                ));
            }
        }
        Ok(Self {
            system,
            inner: LinearGaussianModelBuilder::new(prior),
        })
    }

    pub fn observe(mut self, observation: LinearObservation) -> Result<Self> {
        self.inner = self.inner.observe(observation)?;
        Ok(self)
    }

    /// Add a hard equality constraint on the active or full FEEC state.
    pub fn constrain(mut self, constraint: LinearConstraint) -> Result<Self> {
        self.inner = self.inner.constrain(constraint)?;
        Ok(self)
    }

    pub fn observe_weak_residual(mut self, noise: PdeResidualNoise) -> Result<Self> {
        let precision = match noise {
            PdeResidualNoise::MassWeightedL2StandardDeviation(sigma) => self
                .system
                .assembly
                .state_mass_inverse
                .as_ref()
                .ok_or_else(|| {
                    FeecGmrfError::Unsupported(
                        "mass-weighted PDE residual requires a state-mass inverse".to_string(),
                    )
                })
                .map(|inverse| {
                    feg_infer::sparse::feec_csr_to_core_triplet(inverse)
                        .scaled(1.0 / (sigma * sigma))
                })?,
        };
        let residual = self
            .system
            .residual_quantity("__pde_residual_observation")?;
        let observation = LinearObservation::with_bias(
            residual.operator().clone(),
            vec![0.0; residual.operator().output_dimension()],
            residual.bias().to_vec(),
            crate::model::GaussianNoise::precision(precision)?,
        )?;
        self.inner = self.inner.observe(observation)?;
        Ok(self)
    }

    pub fn derive(mut self, quantity: DerivedQuantity) -> Result<Self> {
        self.inner = self.inner.derive(quantity)?;
        Ok(self)
    }

    pub fn derive_physical(mut self, physical: PhysicalMap) -> Result<Self> {
        self.inner = self.inner.derive_physical(physical)?;
        Ok(self)
    }

    pub fn condition(self) -> Result<crate::infer::Posterior> {
        crate::infer::condition_linear_pde_model(self.inner, &self.system.assembly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Vector};
    use formoniq::reduction::DofLayout;

    fn diagonal_system(with_inverse: bool) -> LinearPdeSystem {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 4.0);
        let matrix = CsrMatrix::from(&coo);
        LinearPdeSystem::from_reduced_assembly(ReducedLinearPdeAssembly {
            operator: matrix.clone(),
            residual_bias: Vector::zeros(2),
            state_mass: matrix.clone(),
            state_mass_inverse: with_inverse.then(|| matrix.clone()),
            layout: DofLayout::identity(2),
            forcing_operator: matrix.clone(),
            neumann_operator: matrix,
        })
        .unwrap()
    }

    #[test]
    fn deterministic_solution_uses_rhs_with_the_declared_sign() {
        let system = diagonal_system(true)
            .with_right_hand_side(&[2.0, 8.0])
            .unwrap();
        let solution = system.solve_deterministic().unwrap();
        assert!((solution.active()[0] - 1.0).abs() < 1.0e-12);
        assert!((solution.active()[1] - 2.0).abs() < 1.0e-12);
        let residual = system.residual_quantity("r").unwrap();
        assert_eq!(residual.operator().apply(&[1.0, 2.0]).unwrap(), [2.0, 8.0]);
        assert_eq!(residual.bias(), [-2.0, -8.0]);
    }

    #[test]
    fn state_l2_metrics_use_the_reduced_mass_matrix() {
        let system = diagonal_system(true);
        assert!((system.state_l2_norm(&[1.0, 2.0]).unwrap() - 18.0_f64.sqrt()).abs() < 1.0e-12);
        assert!(
            (system
                .relative_state_l2_error(&[2.0, 2.0], &[1.0, 2.0])
                .unwrap()
                - (2.0_f64 / 18.0).sqrt())
            .abs()
                < 1.0e-12
        );
        assert!(system.state_l2_norm(&[1.0]).is_err());
        assert!(system
            .relative_state_l2_error(&[0.0, 0.0], &[0.0, 0.0])
            .is_err());
    }

    #[test]
    fn weak_l2_residual_requires_mass_inverse() {
        let system = diagonal_system(false);
        let prior = system.diagonal_prior(1.0).unwrap();
        let result = LinearPdeModelBuilder::new(prior, &system)
            .unwrap()
            .observe_weak_residual(
                PdeResidualNoise::mass_weighted_l2_standard_deviation(0.1).unwrap(),
            );
        assert!(result.is_err());
    }

    #[test]
    fn identity_layout_accepts_matern_prior_without_boundary_metadata() {
        let system = diagonal_system(true).with_form_space(1, 2).unwrap();
        let prior = system.matern_prior_builder().unwrap().build().unwrap();
        assert!(prior.boundary_elimination().is_none());
        assert!(LinearPdeModelBuilder::new(prior, &system).is_ok());
    }

    #[test]
    fn l2_white_prior_scales_state_mass_and_validates_scale() {
        let system = diagonal_system(true).with_form_space(1, 2).unwrap();
        let prior = system.l2_white_noise_prior(3.0).unwrap();
        assert_eq!(prior.mean(), [0.0, 0.0]);
        assert_eq!(
            prior.precision(),
            &crate::operator::SparseMat::from_rows(2, &[vec![(0, 6.0)], vec![(1, 12.0)]],).unwrap()
        );
        assert_eq!(prior.form_degree(), Some(FormDegree::new(1, 2).unwrap()));
        assert!(system.l2_white_noise_prior(0.0).is_err());
        assert!(system.l2_white_noise_prior(-1.0).is_err());
        assert!(system.l2_white_noise_prior(f64::NAN).is_err());
    }

    #[test]
    fn l2_white_prior_preserves_boundary_layout() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 4.0);
        let matrix = CsrMatrix::from(&coo);
        let layout = DofLayout {
            full_dimension: 3,
            active_dofs: vec![0, 2],
            prescribed_dofs: vec![formoniq::reduction::PrescribedDof {
                index: 1,
                value: -1.25,
            }],
        };
        let system = LinearPdeSystem::from_reduced_assembly(ReducedLinearPdeAssembly {
            operator: matrix.clone(),
            residual_bias: Vector::zeros(2),
            state_mass: matrix.clone(),
            state_mass_inverse: Some(matrix.clone()),
            layout,
            forcing_operator: matrix.clone(),
            neumann_operator: matrix,
        })
        .unwrap()
        .with_form_space(1, 2)
        .unwrap();

        let prior = system.l2_white_noise_prior(1.0).unwrap();
        assert_eq!(prior.dimension(), 2);
        assert_eq!(prior.cochain_dimension(), 3);
        assert_eq!(prior.cochain_mean().unwrap(), [0.0, -1.25, 0.0]);
    }
}
