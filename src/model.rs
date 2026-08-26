//! Typed composition of priors, observations, constraints, and derived quantities.

use crate::operator::LinearMap;
use crate::prior::GaussianPrior;
use crate::{FeecGmrfError, Result};

/// Gaussian observation noise represented either by a common variance or by
/// an explicit sparse precision on observation space.
#[derive(Debug, Clone, PartialEq)]
pub enum GaussianNoise {
    /// Independent rows with one common variance.
    ScalarVariance(f64),
    /// Correlated rows represented by an observation-space precision.
    Precision(crate::operator::SparseMat),
}

impl GaussianNoise {
    /// Construct a scalar observation-variance model.
    pub fn variance(variance: f64) -> Result<Self> {
        if !variance.is_finite() || variance <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Gaussian noise variance must be finite and positive".to_string(),
            ));
        }
        Ok(Self::ScalarVariance(variance))
    }

    /// Construct independent Gaussian noise from a common standard deviation.
    pub fn standard_deviation(standard_deviation: f64) -> Result<Self> {
        if !standard_deviation.is_finite() || standard_deviation <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Gaussian noise standard deviation must be finite and positive".to_string(),
            ));
        }
        Self::variance(standard_deviation * standard_deviation)
    }

    /// Construct a correlated noise model from an observation precision.
    pub fn precision(precision: crate::operator::SparseMat) -> Result<Self> {
        if precision.nrows() != precision.ncols() {
            return Err(FeecGmrfError::Dimension(
                "observation precision must be square".to_string(),
            ));
        }
        crate::operator::validate_sparse(&precision)?;
        Ok(Self::Precision(precision))
    }

    /// Construct independent heteroscedastic noise from per-row variances.
    pub fn independent_variances(variances: &[f64]) -> Result<Self> {
        if variances
            .iter()
            .any(|variance| !variance.is_finite() || *variance <= 0.0)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "independent Gaussian variances must be finite and positive".to_string(),
            ));
        }
        let mut precision = crate::operator::SparseMat::new(variances.len(), variances.len());
        for (index, variance) in variances.iter().copied().enumerate() {
            precision.push(index, index, 1.0 / variance);
        }
        Self::precision(precision)
    }

    /// Construct independent heteroscedastic noise from per-row standard deviations.
    pub fn independent_standard_deviations(standard_deviations: &[f64]) -> Result<Self> {
        if standard_deviations
            .iter()
            .any(|standard_deviation| !standard_deviation.is_finite() || *standard_deviation <= 0.0)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "independent Gaussian standard deviations must be finite and positive".to_string(),
            ));
        }
        Self::independent_variances(
            &standard_deviations
                .iter()
                .map(|standard_deviation| standard_deviation * standard_deviation)
                .collect::<Vec<_>>(),
        )
    }
}

/// One sparse affine observation row with its own Gaussian noise variance.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearObservationRow {
    pub entries: Vec<(usize, f64)>,
    pub value: f64,
    pub bias: f64,
    pub noise_variance: f64,
}

impl LinearObservationRow {
    /// Construct a zero-bias observation row.
    pub fn new(entries: Vec<(usize, f64)>, value: f64, noise_variance: f64) -> Result<Self> {
        Self::with_bias(entries, value, 0.0, noise_variance)
    }

    /// Construct an affine observation row.
    pub fn with_bias(
        entries: Vec<(usize, f64)>,
        value: f64,
        bias: f64,
        noise_variance: f64,
    ) -> Result<Self> {
        if !value.is_finite()
            || !bias.is_finite()
            || !noise_variance.is_finite()
            || noise_variance <= 0.0
            || entries.iter().any(|(_, entry)| !entry.is_finite())
        {
            return Err(FeecGmrfError::InvalidParameter(
                "observation-row values, entries, and positive variance must be finite".to_string(),
            ));
        }
        Ok(Self {
            entries,
            value,
            bias,
            noise_variance,
        })
    }
}

/// A linear Gaussian observation `y = Hx + b + epsilon`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearObservation {
    pub(crate) operator: LinearMap,
    pub(crate) values: Vec<f64>,
    pub(crate) bias: Vec<f64>,
    pub(crate) noise: GaussianNoise,
}

impl LinearObservation {
    /// Construct a zero-bias linear observation.
    pub fn new(operator: LinearMap, values: Vec<f64>, noise: GaussianNoise) -> Result<Self> {
        let bias = vec![0.0; values.len()];
        Self::with_bias(operator, values, bias, noise)
    }

    /// Observe selected coefficients by index.
    pub fn at_indices(
        input_dimension: usize,
        indices: &[usize],
        values: Vec<f64>,
        noise: GaussianNoise,
    ) -> Result<Self> {
        Self::new(
            LinearMap::selector(input_dimension, indices)?,
            values,
            noise,
        )
    }

    /// Construct a heteroscedastic observation system from sparse rows.
    pub fn from_rows(input_dimension: usize, rows: Vec<LinearObservationRow>) -> Result<Self> {
        let operator = LinearMap::weighted_rows(
            input_dimension,
            &rows
                .iter()
                .map(|row| row.entries.clone())
                .collect::<Vec<_>>(),
        )?;
        let values = rows.iter().map(|row| row.value).collect();
        let bias = rows.iter().map(|row| row.bias).collect();
        let variances = rows
            .iter()
            .map(|row| row.noise_variance)
            .collect::<Vec<_>>();
        Self::with_bias(
            operator,
            values,
            bias,
            GaussianNoise::independent_variances(&variances)?,
        )
    }

    /// Construct a linear observation with an explicit affine bias.
    pub fn with_bias(
        operator: LinearMap,
        values: Vec<f64>,
        bias: Vec<f64>,
        noise: GaussianNoise,
    ) -> Result<Self> {
        if operator.output_dimension() != values.len() || bias.len() != values.len() {
            return Err(FeecGmrfError::Dimension(
                "observation rows, values, and bias lengths must match".to_string(),
            ));
        }
        if let GaussianNoise::Precision(precision) = &noise {
            if precision.nrows() != values.len() {
                return Err(FeecGmrfError::Dimension(format!(
                    "observation precision dimension {} does not match {} observation rows",
                    precision.nrows(),
                    values.len()
                )));
            }
        }
        if !values.iter().chain(&bias).all(|value| value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "observation values and bias must be finite".to_string(),
            ));
        }
        Ok(Self {
            operator,
            values,
            bias,
            noise,
        })
    }
}

/// A hard linear equality `Cx = d`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearConstraint {
    pub(crate) operator: LinearMap,
    pub(crate) target: Vec<f64>,
}

impl LinearConstraint {
    /// Construct a validated equality constraint.
    pub fn new(operator: LinearMap, target: Vec<f64>) -> Result<Self> {
        if operator.output_dimension() != target.len() {
            return Err(FeecGmrfError::Dimension(
                "constraint rows and target length must match".to_string(),
            ));
        }
        if !target.iter().all(|value| value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "constraint target must be finite".to_string(),
            ));
        }
        Ok(Self { operator, target })
    }
}

/// A named linear physical or diagnostic output.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedQuantity {
    pub(crate) name: String,
    pub(crate) operator: LinearMap,
    pub(crate) bias: Vec<f64>,
}

impl DerivedQuantity {
    /// Construct a named derived quantity.
    pub fn new(name: impl Into<String>, operator: LinearMap) -> Result<Self> {
        let bias = vec![0.0; operator.output_dimension()];
        Self::with_bias(name, operator, bias)
    }

    /// Construct a named affine derived quantity `operator * x + bias`.
    pub fn with_bias(name: impl Into<String>, operator: LinearMap, bias: Vec<f64>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "derived quantity name must not be empty".to_string(),
            ));
        }
        if bias.len() != operator.output_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "derived quantity bias length {} does not match output dimension {}",
                bias.len(),
                operator.output_dimension()
            )));
        }
        if bias.iter().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "derived quantity bias must contain only finite values".to_string(),
            ));
        }
        Ok(Self {
            name,
            operator,
            bias,
        })
    }

    /// Stable output name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Linear pushforward operator.
    pub fn operator(&self) -> &LinearMap {
        &self.operator
    }

    /// Affine output bias.
    pub fn bias(&self) -> &[f64] {
        &self.bias
    }
}

/// Builder for reusable linear FEEC--GMRF models.
pub struct LinearGaussianModelBuilder {
    pub(crate) prior: GaussianPrior,
    pub(crate) observations: Vec<LinearObservation>,
    pub(crate) constraints: Vec<LinearConstraint>,
    pub(crate) derived: Vec<DerivedQuantity>,
}

impl LinearGaussianModelBuilder {
    /// Start a model from a Gaussian prior.
    pub fn new(prior: GaussianPrior) -> Self {
        Self {
            prior,
            observations: Vec::new(),
            constraints: Vec::new(),
            derived: Vec::new(),
        }
    }

    /// Add an observation term.
    pub fn observe(mut self, observation: LinearObservation) -> Result<Self> {
        let eliminated = self
            .prior
            .eliminate_map(&observation.operator, &observation.bias)?;
        let (operator, bias) = eliminated.into_parts();
        self.observations.push(LinearObservation::with_bias(
            operator,
            observation.values,
            bias,
            observation.noise,
        )?);
        Ok(self)
    }

    /// Add a hard equality constraint.
    pub fn constrain(mut self, constraint: LinearConstraint) -> Result<Self> {
        let eliminated = self.prior.eliminate_map(
            &constraint.operator,
            &vec![0.0; constraint.operator.output_dimension()],
        )?;
        let (operator, fixed_offset) = eliminated.into_parts();
        let target = constraint
            .target
            .iter()
            .zip(fixed_offset)
            .map(|(target, offset)| target - offset)
            .collect::<Vec<_>>();
        let mut active_rows = vec![false; operator.output_dimension()];
        for (row, _, value) in operator.matrix().triplet_iter() {
            if value != 0.0 {
                active_rows[row] = true;
            }
        }
        let mut retained = Vec::new();
        for (row, active) in active_rows.into_iter().enumerate() {
            if active {
                retained.push(row);
                continue;
            }
            let tolerance = 1.0e-12 * (1.0 + constraint.target[row].abs());
            if target[row].abs() > tolerance {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "constraint row {row} conflicts with prescribed essential-boundary values"
                )));
            }
        }
        if retained.is_empty() {
            return Ok(self);
        }
        let target = retained.iter().map(|row| target[*row]).collect();
        let operator = operator.select_outputs(&retained)?;
        self.constraints
            .push(LinearConstraint::new(operator, target)?);
        Ok(self)
    }

    /// Add a physical or diagnostic pushforward.
    pub fn derive(mut self, quantity: DerivedQuantity) -> Result<Self> {
        if self
            .derived
            .iter()
            .any(|existing| existing.name == quantity.name)
        {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "duplicate derived quantity `{}`",
                quantity.name
            )));
        }
        let eliminated = self
            .prior
            .eliminate_map(&quantity.operator, &quantity.bias)?;
        let (operator, bias) = eliminated.into_parts();
        self.derived
            .push(DerivedQuantity::with_bias(quantity.name, operator, bias)?);
        Ok(self)
    }

    /// Add a named physical pushforward.
    pub fn derive_physical(self, physical: crate::physical::PhysicalMap) -> Result<Self> {
        self.derive(physical.into_derived_quantity()?)
    }

    /// Factor and condition the model.
    pub fn condition(self) -> Result<crate::infer::Posterior> {
        crate::infer::condition_linear_model(self)
    }

    /// Prepare a fixed prior/operator/noise design for repeated observation values.
    pub fn prepare(self) -> Result<crate::infer::PreparedLinearGaussianModel> {
        crate::infer::prepare_linear_model(self)
    }
}

/// Existing nonlinear model contracts and solvers.
pub mod nonlinear {
    use super::GaussianNoise;
    use crate::boundary::{EliminatedResidualModel, EssentialBoundaryElimination};
    use crate::prior::GaussianPrior;
    use crate::{FeecGmrfError, Result};
    pub use feg_core::{NonlinearResidualEvaluation, NonlinearResidualModel as ResidualModel};
    pub use feg_infer::nonlinear::{
        GaussNewtonConfig, GaussNewtonLinearSolve, GaussNewtonRunDiagnostics,
        GaussNewtonStepRegularization, NonlinearLaplaceResult,
    };
    use rand::Rng;

    /// Nonlinear Laplace result with full-cochain reconstruction.
    pub struct NonlinearPosterior {
        inner: NonlinearLaplaceResult,
        cochain_map: Vec<f64>,
        cochain_variances: Vec<f64>,
        boundary_elimination: Option<EssentialBoundaryElimination>,
    }

    impl NonlinearPosterior {
        /// MAP point in active coordinates.
        pub fn latent_map(&self) -> &[f64] {
            &self.inner.map
        }

        /// MAP point in full cochain ordering, including prescribed values.
        pub fn cochain_map(&self) -> &[f64] {
            &self.cochain_map
        }

        /// Laplace marginal variances in full cochain ordering. Empty when
        /// variance estimation was disabled in the Gauss--Newton configuration.
        pub fn cochain_variances(&self) -> &[f64] {
            &self.cochain_variances
        }

        /// Essential-boundary elimination carried by this posterior, when present.
        pub fn boundary_elimination(&self) -> Option<&EssentialBoundaryElimination> {
            self.boundary_elimination.as_ref()
        }

        /// Access detailed Gauss--Newton/Laplace diagnostics.
        pub fn inner(&self) -> &NonlinearLaplaceResult {
            &self.inner
        }

        /// Return the lower-level inference result.
        pub fn into_inner(self) -> NonlinearLaplaceResult {
            self.inner
        }

        /// Generate a Laplace posterior sample in full cochain ordering.
        pub fn sample_cochain<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>> {
            let active = self
                .inner
                .posterior_gmrf
                .sample(rng)
                .map_err(FeecGmrfError::from)?;
            let active = active.iter().copied().collect::<Vec<_>>();
            match &self.boundary_elimination {
                Some(elimination) => elimination.lift_state(&active),
                None => Ok(active),
            }
        }
    }

    impl std::ops::Deref for NonlinearPosterior {
        type Target = NonlinearLaplaceResult;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl std::ops::DerefMut for NonlinearPosterior {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inner
        }
    }

    /// A validated nonlinear residual likelihood term.
    pub struct NonlinearResidualTerm<'a> {
        name: String,
        model: &'a dyn ResidualModel,
        observations: Vec<f64>,
        noise: GaussianNoise,
    }

    impl<'a> NonlinearResidualTerm<'a> {
        /// Construct a residual term with observed residual values.
        pub fn new(
            name: impl Into<String>,
            model: &'a dyn ResidualModel,
            observations: Vec<f64>,
            noise: GaussianNoise,
        ) -> Result<Self> {
            let name = name.into();
            if name.trim().is_empty() {
                return Err(FeecGmrfError::InvalidParameter(
                    "nonlinear residual term name must not be empty".to_string(),
                ));
            }
            if observations.len() != model.residual_dimension() {
                return Err(FeecGmrfError::Dimension(format!(
                    "nonlinear residual term `{name}` has {} observations for {} residual rows",
                    observations.len(),
                    model.residual_dimension()
                )));
            }
            if !observations.iter().all(|value| value.is_finite()) {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "nonlinear residual term `{name}` observations must be finite"
                )));
            }
            if let GaussianNoise::Precision(precision) = &noise {
                if precision.nrows() != observations.len() {
                    return Err(FeecGmrfError::Dimension(format!(
                        "nonlinear residual precision dimension {} does not match {} rows",
                        precision.nrows(),
                        observations.len()
                    )));
                }
            }
            Ok(Self {
                name,
                model,
                observations,
                noise,
            })
        }

        /// Construct a zero-target residual term.
        pub fn zero(
            name: impl Into<String>,
            model: &'a dyn ResidualModel,
            noise: GaussianNoise,
        ) -> Result<Self> {
            Self::new(name, model, vec![0.0; model.residual_dimension()], noise)
        }
    }

    /// Reusable nonlinear MAP/Laplace model builder.
    pub struct NonlinearLaplaceModelBuilder<'a> {
        prior: GaussianPrior,
        residual_terms: Vec<NonlinearResidualTerm<'a>>,
        config: GaussNewtonConfig,
    }

    impl<'a> NonlinearLaplaceModelBuilder<'a> {
        /// Start a nonlinear model from a Gaussian prior.
        pub fn new(prior: GaussianPrior) -> Self {
            Self {
                prior,
                residual_terms: Vec::new(),
                config: GaussNewtonConfig::default(),
            }
        }

        /// Set Gauss--Newton and Laplace solver policies.
        pub fn config(mut self, config: GaussNewtonConfig) -> Self {
            self.config = config;
            self
        }

        /// Add a validated nonlinear residual likelihood.
        pub fn residual(mut self, term: NonlinearResidualTerm<'a>) -> Result<Self> {
            let dimension = term.model.state_dimension();
            if dimension != self.prior.dimension() && dimension != self.prior.cochain_dimension() {
                return Err(FeecGmrfError::Dimension(format!(
                    "nonlinear residual state dimension {} matches neither active dimension {} nor full cochain dimension {}",
                    dimension,
                    self.prior.dimension(),
                    self.prior.cochain_dimension()
                )));
            }
            if self
                .residual_terms
                .iter()
                .any(|existing| existing.name == term.name)
            {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "duplicate nonlinear residual term `{}`",
                    term.name
                )));
            }
            self.residual_terms.push(term);
            Ok(self)
        }

        /// Solve for the MAP point and construct the Laplace approximation.
        pub fn solve(self) -> Result<NonlinearPosterior> {
            if self.residual_terms.is_empty() {
                return Err(FeecGmrfError::InvalidParameter(
                    "a nonlinear model requires at least one residual term".to_string(),
                ));
            }
            let boundary_elimination = self.prior.boundary_elimination().cloned();
            let eliminated_models = self
                .residual_terms
                .iter()
                .map(|term| {
                    if term.model.state_dimension() == self.prior.dimension() {
                        Ok(None)
                    } else {
                        let elimination = boundary_elimination.clone().ok_or_else(|| {
                            FeecGmrfError::Dimension(
                                "a full-cochain residual requires boundary elimination".to_string(),
                            )
                        })?;
                        EliminatedResidualModel::new(term.model, elimination).map(Some)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let terms = self
                .residual_terms
                .iter()
                .enumerate()
                .map(
                    |(index, term)| feg_infer::nonlinear::NonlinearResidualTerm {
                        name: term.name.clone(),
                        model: eliminated_models[index]
                            .as_ref()
                            .map_or(term.model, |model| model as &dyn ResidualModel),
                        observations: term.observations.clone(),
                        noise: match &term.noise {
                            GaussianNoise::ScalarVariance(variance) => {
                                feg_infer::nonlinear::GaussianNoiseModel::ScalarVariance(*variance)
                            }
                            GaussianNoise::Precision(precision) => {
                                feg_infer::nonlinear::GaussianNoiseModel::Precision(
                                    precision.clone(),
                                )
                            }
                        },
                    },
                )
                .collect();
            let problem = feg_infer::nonlinear::NonlinearLaplaceProblem {
                prior: feg_core::GaussianPriorSpec {
                    mean: self.prior.mean().to_vec(),
                    precision: self.prior.precision().clone(),
                },
                residual_terms: terms,
                linear_measurements: Vec::new(),
                precision_weighted_measurements: Vec::new(),
                derived_quantities: Vec::new(),
            };
            let inner = feg_infer::nonlinear::solve_nonlinear_laplace(&problem, &self.config)
                .map_err(FeecGmrfError::Inference)?;
            let (cochain_map, cochain_variances) = match &boundary_elimination {
                Some(elimination) => (
                    elimination.lift_state(&inner.map)?,
                    if inner.posterior_variance.is_empty() {
                        Vec::new()
                    } else {
                        elimination.lift_variances(&inner.posterior_variance)?
                    },
                ),
                None => (inner.map.clone(), inner.posterior_variance.clone()),
            };
            Ok(NonlinearPosterior {
                inner,
                cochain_map,
                cochain_variances,
                boundary_elimination,
            })
        }
    }
}
