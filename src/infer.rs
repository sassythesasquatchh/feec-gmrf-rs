//! Inference results and adapters to generic GMRF algebra.

use crate::boundary::EssentialBoundaryElimination;
use crate::model::{DerivedQuantity, LinearGaussianModelBuilder};
use crate::operator::{LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
use gmrf_core::observation::{
    condition_linear_gaussian_with_factor, ht_precision_weighted_observations,
    ht_weighted_observations, LinearObservationTerm,
};
use gmrf_core::types::{CooMatrix, DenseMatrix, SparseMatrix, Vector};
use gmrf_core::{
    estimate_constrained_transformed_variances,
    estimate_factored_transformed_variance_weighted_trace, estimate_factored_transformed_variances,
    estimate_hutchinson_weighted_covariance_trace, exact_weighted_covariance_trace,
    ConstrainedPrecisionSolver, Gmrf, ProbeBatchConfig, SparseRowOperator, TransformedVarianceMode,
};
pub use gmrf_core::{ProbeDistribution, VarianceFloor};
use rand::Rng;
use std::collections::BTreeMap;

/// Deterministic Monte Carlo configuration for marginal-variance estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonteCarloVarianceConfig {
    /// Total number of posterior samples.
    pub samples: usize,
    /// Number of deterministic batches used for error diagnostics.
    pub batches: usize,
    /// Base seed from which batch seeds are derived.
    pub seed: u64,
}

impl MonteCarloVarianceConfig {
    /// Construct a validated Monte Carlo variance configuration.
    pub fn new(samples: usize, batches: usize, seed: u64) -> Result<Self> {
        if samples == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Monte Carlo variance estimation requires at least one sample".to_string(),
            ));
        }
        if batches == 0 || batches > samples {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "Monte Carlo batch count must lie in 1..={samples}"
            )));
        }
        Ok(Self {
            samples,
            batches,
            seed,
        })
    }
}

/// Deterministic Hutchinson configuration for marginal-variance estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HutchinsonVarianceConfig {
    /// Total number of stochastic probes.
    pub probes: usize,
    /// Number of deterministic batches used for error diagnostics.
    pub batches: usize,
    /// Base seed from which batch seeds are derived.
    pub seed: u64,
    /// Distribution used to generate probe vectors.
    pub distribution: ProbeDistribution,
    /// Stabilization applied to noisy variance estimates.
    pub floor: VarianceFloor,
}

impl HutchinsonVarianceConfig {
    /// Construct a validated Rademacher-probe configuration.
    pub fn new(probes: usize, batches: usize, seed: u64) -> Result<Self> {
        let config = Self {
            probes,
            batches,
            seed,
            distribution: ProbeDistribution::Rademacher,
            floor: VarianceFloor::Zero,
        };
        config.validate()?;
        Ok(config)
    }

    /// Select the probe distribution.
    pub fn distribution(mut self, distribution: ProbeDistribution) -> Self {
        self.distribution = distribution;
        self
    }

    /// Select the variance-floor policy.
    pub fn floor(mut self, floor: VarianceFloor) -> Result<Self> {
        self.floor = floor;
        self.validate()?;
        Ok(self)
    }

    fn validate(self) -> Result<()> {
        if self.probes == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Hutchinson variance estimation requires at least one probe".to_string(),
            ));
        }
        if self.batches == 0 || self.batches > self.probes {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "Hutchinson batch count must lie in 1..={}",
                self.probes
            )));
        }
        if let VarianceFloor::PositiveMean { scale } = self.floor {
            if !scale.is_finite() || scale < 0.0 {
                return Err(FeecGmrfError::InvalidParameter(
                    "variance-floor scale must be finite and non-negative".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn probe_config(self) -> ProbeBatchConfig {
        ProbeBatchConfig {
            num_probes: self.probes,
            batch_count: self.batches,
            rng_seed: self.seed,
        }
    }
}

/// Marginal-variance computation requested from a posterior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarianceMethod {
    /// Deterministic covariance solves.
    Exact,
    /// Seeded posterior Monte Carlo with batch diagnostics.
    MonteCarlo(MonteCarloVarianceConfig),
    /// Seeded Hutchinson diagonal probing with batch diagnostics.
    Hutchinson(HutchinsonVarianceConfig),
    /// Exact solves up to the supplied latent dimension, then Hutchinson.
    Auto {
        exact_max_dofs: usize,
        hutchinson: HutchinsonVarianceConfig,
    },
}

impl VarianceMethod {
    /// Construct a validated exact/Hutchinson automatic policy.
    pub fn auto(exact_max_dofs: usize, hutchinson: HutchinsonVarianceConfig) -> Result<Self> {
        if exact_max_dofs == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "automatic variance estimation requires a positive exact-DOF limit".to_string(),
            ));
        }
        hutchinson.validate()?;
        Ok(Self::Auto {
            exact_max_dofs,
            hutchinson,
        })
    }

    fn transformed_mode(self) -> Option<TransformedVarianceMode> {
        match self {
            Self::Exact => Some(TransformedVarianceMode::Exact),
            Self::MonteCarlo(_) => None,
            Self::Hutchinson(config) => Some(TransformedVarianceMode::Hutchinson {
                config: config.probe_config(),
                floor: config.floor,
                distribution: config.distribution,
            }),
            Self::Auto {
                exact_max_dofs,
                hutchinson,
            } => Some(TransformedVarianceMode::Auto {
                exact_max_dofs,
                config: hutchinson.probe_config(),
                floor: hutchinson.floor,
                distribution: hutchinson.distribution,
            }),
        }
    }
}

/// Estimator that produced a public variance result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceEstimator {
    /// Exact covariance solves.
    Exact,
    /// Posterior Monte Carlo samples.
    MonteCarlo,
    /// Hutchinson diagonal probing.
    Hutchinson,
}

impl VarianceEstimator {
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Marginal variances with optional stochastic error diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceEstimate {
    pub values: Vec<f64>,
    pub estimator: VarianceEstimator,
    pub sample_count: usize,
    pub batch_sizes: Vec<usize>,
    pub batch_standard_error: Option<Vec<f64>>,
    pub relative_standard_error: Option<Vec<f64>>,
    /// Number of negative entries before any estimator-specific stabilization.
    pub negative_count: usize,
    /// Smallest entry before any estimator-specific stabilization.
    pub minimum_value: f64,
}

/// Marginal variances and their weighted covariance trace.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedVarianceEstimate {
    pub variances: VarianceEstimate,
    pub weighted_trace: f64,
    pub weighted_trace_standard_error: Option<f64>,
    pub weighted_trace_relative_standard_error: Option<f64>,
}

/// Scalar estimate of `trace(W Covariance)` with stochastic diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedCovarianceTraceEstimate {
    pub value: f64,
    pub estimator: VarianceEstimator,
    pub sample_count: usize,
    pub batch_sizes: Vec<usize>,
    pub batch_standard_error: Option<f64>,
    pub relative_standard_error: Option<f64>,
}

/// Sparse factorization size diagnostics suitable for study reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorizationDiagnostics {
    pub dimension: usize,
    /// Nonzeros in the complete symmetric precision representation.
    pub precision_nonzeros: usize,
    /// Nonzeros in the stored sparse Cholesky factor.
    pub factor_nonzeros: usize,
    /// Factor nonzeros divided by precision nonzeros on or below the diagonal.
    pub fill_ratio: f64,
}

/// A Gaussian prior with a cached sparse factor for repeated uncertainty queries.
pub struct FactoredGaussianPrior {
    posterior: Posterior,
}

#[derive(Debug, Clone)]
enum PreparedObservationNoise {
    ScalarVariance(f64),
    Precision(SparseMatrix),
}

#[derive(Debug, Clone)]
struct PreparedObservation {
    matrix: SparseMatrix,
    bias: Vector,
    noise: PreparedObservationNoise,
}

/// Fixed Gaussian design with one reusable posterior factorization.
pub struct PreparedLinearGaussianModel {
    prior_mean: Vector,
    observations: Vec<PreparedObservation>,
    default_values: Vec<Vec<f64>>,
    posterior_precision: SparseMatrix,
    posterior_factor: gmrf_core::types::SparseCholeskyFactor,
    public_precision: SparseMat,
    constraints: Option<(DenseMatrix, Vector)>,
    constraint_solver: Option<ConstrainedPrecisionSolver>,
    derived: BTreeMap<String, (LinearMap, Vec<f64>)>,
    boundary_elimination: Option<EssentialBoundaryElimination>,
}

impl PreparedLinearGaussianModel {
    /// Number of separately supplied observation vectors.
    pub fn observation_term_count(&self) -> usize {
        self.observations.len()
    }

    /// Number of rows expected for each observation vector.
    pub fn observation_row_counts(&self) -> Vec<usize> {
        self.observations
            .iter()
            .map(|observation| observation.matrix.nrows())
            .collect()
    }

    /// Sparse sizes of the reusable posterior precision and factor.
    pub fn factorization_diagnostics(&self) -> FactorizationDiagnostics {
        let precision_nonzeros = self.public_precision.nnz();
        let lower_precision_nonzeros = lower_triangle_nonzeros(&self.public_precision);
        let factor_nonzeros = self.posterior_factor.nnz();
        FactorizationDiagnostics {
            dimension: self.prior_mean.len(),
            precision_nonzeros,
            factor_nonzeros,
            fill_ratio: if lower_precision_nonzeros == 0 {
                0.0
            } else {
                factor_nonzeros as f64 / lower_precision_nonzeros as f64
            },
        }
    }

    /// Condition using the observation values originally supplied to the builder.
    pub fn condition(&self) -> Result<Posterior> {
        self.condition_with_observation_values(&self.default_values)
    }

    /// Reuse the prepared factor and return only the active-coordinate mean.
    ///
    /// This avoids cloning the cached factor into a full [`Posterior`] when a
    /// repeated-data workflow only needs its posterior mean.
    pub fn latent_mean_with_observation_values(&self, values: &[Vec<f64>]) -> Result<Vec<f64>> {
        let (_, mean) = self.mean_vectors_with_observation_values(values)?;
        Ok(mean.iter().copied().collect())
    }

    /// Reuse the prepared factor and return only the full-cochain mean.
    pub fn cochain_mean_with_observation_values(&self, values: &[Vec<f64>]) -> Result<Vec<f64>> {
        let mean = self.latent_mean_with_observation_values(values)?;
        match &self.boundary_elimination {
            Some(elimination) => elimination.lift_state(&mean),
            None => Ok(mean),
        }
    }

    /// Reuse the prepared factor for replacement observation vectors.
    pub fn condition_with_observation_values(&self, values: &[Vec<f64>]) -> Result<Posterior> {
        let (unconstrained_mean, mean) = self.mean_vectors_with_observation_values(values)?;
        let information = self.posterior_precision.mul_vec(&unconstrained_mean);
        let gmrf = Gmrf::from_information_and_precision_with_sqrt(
            information,
            self.posterior_precision.clone(),
            self.posterior_factor.clone(),
        )?;
        let mean = mean.iter().copied().collect::<Vec<_>>();
        let cochain_mean = match &self.boundary_elimination {
            Some(elimination) => elimination.lift_state(&mean)?,
            None => mean.clone(),
        };
        Ok(Posterior {
            gmrf,
            mean,
            cochain_mean,
            precision: self.public_precision.clone(),
            constraints: self.constraints.clone(),
            derived: self.derived.clone(),
            boundary_elimination: self.boundary_elimination.clone(),
        })
    }

    fn mean_vectors_with_observation_values(
        &self,
        values: &[Vec<f64>],
    ) -> Result<(Vector, Vector)> {
        if values.len() != self.observations.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "prepared model expects {} observation vectors, received {}",
                self.observations.len(),
                values.len()
            )));
        }
        let mut delta_information = Vector::zeros(self.prior_mean.len());
        for (term_index, (observation, values)) in self.observations.iter().zip(values).enumerate()
        {
            if values.len() != observation.matrix.nrows() {
                return Err(FeecGmrfError::Dimension(format!(
                    "prepared observation term {term_index} expects {} rows, received {}",
                    observation.matrix.nrows(),
                    values.len()
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "prepared observation term {term_index} contains a non-finite value"
                )));
            }
            let prior_output = observation.matrix.mul_vec(&self.prior_mean);
            let centered = Vector::from_iterator(
                values.len(),
                values
                    .iter()
                    .zip(observation.bias.iter())
                    .zip(prior_output.iter())
                    .map(|((value, bias), prior)| value - bias - prior),
            );
            let update = match &observation.noise {
                PreparedObservationNoise::ScalarVariance(variance) => {
                    ht_weighted_observations(&observation.matrix, &centered, 1.0 / variance)
                }
                PreparedObservationNoise::Precision(precision) => {
                    ht_precision_weighted_observations(&observation.matrix, &centered, precision)
                }
            };
            delta_information += update;
        }

        let delta_mean = self.posterior_factor.solve(&delta_information)?;
        let unconstrained_mean = Vector::from_iterator(
            self.prior_mean.len(),
            self.prior_mean
                .iter()
                .zip(delta_mean.iter())
                .map(|(prior, delta)| prior + delta),
        );
        let mean = match (&self.constraints, &self.constraint_solver) {
            (Some((_, target)), Some(solver)) => {
                let information = self.posterior_precision.mul_vec(&unconstrained_mean);
                solver.solve(&information, target)?
            }
            (None, None) => unconstrained_mean.clone(),
            _ => {
                return Err(FeecGmrfError::Inference(
                    "prepared constraint state is inconsistent".to_string(),
                ))
            }
        };
        Ok((unconstrained_mean, mean))
    }
}

impl FactoredGaussianPrior {
    pub(crate) fn new(prior: crate::prior::GaussianPrior) -> Result<Self> {
        Ok(Self {
            posterior: condition_linear_model(LinearGaussianModelBuilder::new(prior))?,
        })
    }

    /// Prior mean in active coordinates.
    pub fn mean(&self) -> &[f64] {
        self.posterior.mean()
    }

    /// Prior mean in full cochain ordering.
    pub fn cochain_mean(&self) -> &[f64] {
        self.posterior.cochain_mean()
    }

    /// Sparse precision/factor sizes.
    pub fn factorization_diagnostics(&self) -> Result<FactorizationDiagnostics> {
        self.posterior.factorization_diagnostics()
    }

    /// Estimate active-coordinate marginal variances.
    pub fn latent_variance_estimate(&mut self, method: VarianceMethod) -> Result<VarianceEstimate> {
        self.posterior.latent_variance_estimate(method)
    }

    /// Estimate marginal variances of an active- or full-cochain map.
    pub fn pushforward_variance_estimate(
        &mut self,
        map: &LinearMap,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        self.posterior.pushforward_variance_estimate(map, method)
    }

    /// Estimate mapped variances and their weighted trace.
    pub fn pushforward_weighted_variance_estimate(
        &mut self,
        map: &LinearMap,
        output_weights: &[f64],
        method: VarianceMethod,
    ) -> Result<WeightedVarianceEstimate> {
        self.posterior
            .pushforward_weighted_variance_estimate(map, output_weights, method)
    }

    /// Estimate `trace(W Cov(x))` for a sparse latent-space weight matrix.
    pub fn weighted_covariance_trace(
        &self,
        weight: &SparseMat,
        method: VarianceMethod,
    ) -> Result<WeightedCovarianceTraceEstimate> {
        self.posterior.weighted_covariance_trace(weight, method)
    }

    /// Compute an exact mapped covariance matrix.
    pub fn pushforward_covariance(&mut self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        self.posterior.pushforward_covariance(map)
    }

    /// Generate a prior sample in full cochain ordering.
    pub fn sample_cochain<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>> {
        self.posterior.sample_cochain(rng)
    }
}

/// Conditioned Gaussian model with reusable factorization and physical outputs.
pub struct Posterior {
    gmrf: Gmrf,
    mean: Vec<f64>,
    cochain_mean: Vec<f64>,
    precision: SparseMat,
    constraints: Option<(DenseMatrix, Vector)>,
    derived: BTreeMap<String, (LinearMap, Vec<f64>)>,
    boundary_elimination: Option<EssentialBoundaryElimination>,
}

impl Posterior {
    /// Posterior mean in active coordinates, including hard constraints.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Posterior mean in active coordinates.
    pub fn latent_mean(&self) -> &[f64] {
        &self.mean
    }

    /// Posterior mean as a complete FEEC cochain, including prescribed values.
    pub fn cochain_mean(&self) -> &[f64] {
        &self.cochain_mean
    }

    /// Essential-boundary elimination carried by this posterior, when present.
    pub fn boundary_elimination(&self) -> Option<&EssentialBoundaryElimination> {
        self.boundary_elimination.as_ref()
    }

    /// Posterior precision before dense low-rank equality constraints.
    pub fn precision(&self) -> &SparseMat {
        &self.precision
    }

    /// Cached sparse Cholesky factor for the unconstrained posterior precision.
    ///
    /// Hard equality constraints are applied through a separate dense low-rank
    /// correction, leaving this factor available for sparse solves.
    pub fn precision_factor(&self) -> Option<&gmrf_core::types::SparseCholeskyFactor> {
        self.gmrf.precision_factor()
    }

    /// Sparse precision/factor sizes for reproducible performance reporting.
    pub fn factorization_diagnostics(&self) -> Result<FactorizationDiagnostics> {
        let factor = self.precision_factor().ok_or_else(|| {
            FeecGmrfError::Inference(
                "posterior precision does not carry a reusable sparse factor".to_string(),
            )
        })?;
        let precision_nonzeros = self.precision.nnz();
        let lower_precision_nonzeros = lower_triangle_nonzeros(&self.precision);
        let factor_nonzeros = factor.nnz();
        Ok(FactorizationDiagnostics {
            dimension: self.mean.len(),
            precision_nonzeros,
            factor_nonzeros,
            fill_ratio: if lower_precision_nonzeros == 0 {
                0.0
            } else {
                factor_nonzeros as f64 / lower_precision_nonzeros as f64
            },
        })
    }

    /// Compute exact marginal variances of the latent coefficients.
    pub fn latent_variances(&mut self) -> Result<Vec<f64>> {
        let values = match &self.constraints {
            Some((constraints, _)) => {
                self.gmrf
                    .exact_constrained_variance_decomposition(constraints)?
                    .constrained_diag
            }
            None => {
                self.gmrf
                    .exact_constrained_variance_decomposition(&DenseMatrix::zeros(
                        0,
                        self.mean.len(),
                    ))?
                    .unconstrained_diag
            }
        };
        Ok(values.iter().copied().collect())
    }

    /// Estimate latent marginal variances using exact solves or seeded Monte Carlo.
    pub fn latent_variance_estimate(&mut self, method: VarianceMethod) -> Result<VarianceEstimate> {
        match method {
            VarianceMethod::Exact => Ok(exact_estimate(self.latent_variances()?)),
            VarianceMethod::MonteCarlo(config) => {
                let estimate = match &self.constraints {
                    Some((matrix, target)) => {
                        gmrf_core::estimate_monte_carlo_constrained_variances(
                            &mut self.gmrf,
                            matrix,
                            target,
                            config.samples,
                            config.batches,
                            config.seed,
                        )?
                    }
                    None => gmrf_core::estimate_monte_carlo_variances(
                        &mut self.gmrf,
                        config.samples,
                        config.batches,
                        config.seed,
                    )?,
                };
                Ok(public_estimate(estimate))
            }
            VarianceMethod::Hutchinson(_) | VarianceMethod::Auto { .. } => {
                let identity = sparse_row_operator(&LinearMap::identity(self.mean.len()))?;
                self.transformed_variance_estimate(&identity, method)
            }
        }
    }

    /// Compute exact marginal variances in the full cochain ordering.
    /// Prescribed coefficients have exactly zero variance.
    pub fn cochain_variances(&mut self) -> Result<Vec<f64>> {
        let active = self.latent_variances()?;
        match &self.boundary_elimination {
            Some(elimination) => elimination.lift_variances(&active),
            None => Ok(active),
        }
    }

    /// Estimate full-cochain marginal variances, inserting exact zeros at fixed DOFs.
    pub fn cochain_variance_estimate(
        &mut self,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        let estimate = self.latent_variance_estimate(method)?;
        match &self.boundary_elimination {
            Some(elimination) => lift_variance_estimate(estimate, elimination),
            None => Ok(estimate),
        }
    }

    /// Apply a named derived-quantity map to the posterior mean.
    pub fn derived_mean(&self, name: &str) -> Result<Vec<f64>> {
        let (map, bias) = self.derived.get(name).ok_or_else(|| {
            FeecGmrfError::InvalidParameter(format!("unknown derived quantity `{name}`"))
        })?;
        let mut output = map.apply(&self.mean)?;
        for (value, bias) in output.iter_mut().zip(bias) {
            *value += bias;
        }
        Ok(output)
    }

    /// Apply an ad hoc map to the posterior mean.
    pub fn pushforward_mean(&self, map: &LinearMap) -> Result<Vec<f64>> {
        if map.input_dimension() == self.mean.len() {
            return map.apply(&self.mean);
        }
        let elimination = self.boundary_elimination.as_ref().ok_or_else(|| {
            FeecGmrfError::Dimension(format!(
                "pushforward input dimension {} does not match posterior dimension {}",
                map.input_dimension(),
                self.mean.len()
            ))
        })?;
        elimination
            .eliminate_map(map, &vec![0.0; map.output_dimension()])?
            .apply(&self.mean)
    }

    /// Generate a posterior sample, respecting hard equality constraints.
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>> {
        let sample = match &self.constraints {
            Some((matrix, target)) => self.gmrf.sample_constrained(matrix, target, rng)?,
            None => self.gmrf.sample(rng)?,
        };
        Ok(sample.iter().copied().collect())
    }

    /// Generate a posterior sample in full cochain ordering, inserting all
    /// prescribed essential-boundary values exactly.
    pub fn sample_cochain<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>> {
        let active = self.sample(rng)?;
        match &self.boundary_elimination {
            Some(elimination) => elimination.lift_state(&active),
            None => Ok(active),
        }
    }

    /// Compute exact marginal variances for a named derived quantity.
    pub fn derived_variances(&mut self, name: &str) -> Result<Vec<f64>> {
        let (map, _) = self.derived.get(name).ok_or_else(|| {
            FeecGmrfError::InvalidParameter(format!("unknown derived quantity `{name}`"))
        })?;
        let operator = sparse_row_operator(map)?;
        let values = match &self.constraints {
            Some((constraints, _)) => {
                self.gmrf
                    .exact_transformed_variance_decomposition(&operator, constraints)?
                    .constrained_diag
            }
            None => {
                self.gmrf
                    .exact_transformed_variance_decomposition(
                        &operator,
                        &DenseMatrix::zeros(0, self.mean.len()),
                    )?
                    .unconstrained_diag
            }
        };
        Ok(values.iter().copied().collect())
    }

    /// Estimate marginal variances for a named derived quantity.
    pub fn derived_variance_estimate(
        &mut self,
        name: &str,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        let map = self
            .derived
            .get(name)
            .ok_or_else(|| {
                FeecGmrfError::InvalidParameter(format!("unknown derived quantity `{name}`"))
            })?
            .0
            .clone();
        self.active_pushforward_variance_estimate(&map, method)
    }

    /// Estimate marginal variances for an ad hoc full- or active-space map.
    pub fn pushforward_variance_estimate(
        &mut self,
        map: &LinearMap,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        let map = self.active_map(map)?;
        self.active_pushforward_variance_estimate(&map, method)
    }

    /// Estimate variances and their weighted trace for a named derived quantity.
    pub fn derived_weighted_variance_estimate(
        &mut self,
        name: &str,
        output_weights: &[f64],
        method: VarianceMethod,
    ) -> Result<WeightedVarianceEstimate> {
        let map = self
            .derived
            .get(name)
            .ok_or_else(|| {
                FeecGmrfError::InvalidParameter(format!("unknown derived quantity `{name}`"))
            })?
            .0
            .clone();
        self.active_weighted_variance_estimate(&map, output_weights, method)
    }

    /// Estimate variances and their weighted trace for an ad hoc map.
    pub fn pushforward_weighted_variance_estimate(
        &mut self,
        map: &LinearMap,
        output_weights: &[f64],
        method: VarianceMethod,
    ) -> Result<WeightedVarianceEstimate> {
        let map = self.active_map(map)?;
        self.active_weighted_variance_estimate(&map, output_weights, method)
    }

    /// Estimate `trace(W Cov(x))` for a sparse latent-space weight matrix.
    pub fn weighted_covariance_trace(
        &self,
        weight: &SparseMat,
        method: VarianceMethod,
    ) -> Result<WeightedCovarianceTraceEstimate> {
        if weight.nrows() != self.mean.len() || weight.ncols() != self.mean.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "covariance-trace weight shape {}x{} does not match latent dimension {}",
                weight.nrows(),
                weight.ncols(),
                self.mean.len()
            )));
        }
        self.ensure_unconstrained_trace()?;
        let factor = self.required_factor()?;
        let weight = gmrf_sparse(weight);
        let estimate = match method {
            VarianceMethod::Exact => exact_weighted_covariance_trace(factor, &weight)?,
            VarianceMethod::Hutchinson(config) => estimate_hutchinson_weighted_covariance_trace(
                factor,
                &weight,
                config.probe_config(),
                config.distribution,
            )?,
            VarianceMethod::Auto {
                exact_max_dofs,
                hutchinson,
            } => {
                if self.mean.len() <= exact_max_dofs {
                    exact_weighted_covariance_trace(factor, &weight)?
                } else {
                    estimate_hutchinson_weighted_covariance_trace(
                        factor,
                        &weight,
                        hutchinson.probe_config(),
                        hutchinson.distribution,
                    )?
                }
            }
            VarianceMethod::MonteCarlo(_) => return Err(FeecGmrfError::Unsupported(
                "general sparse weighted covariance traces support exact or Hutchinson estimation"
                    .to_string(),
            )),
        };
        Ok(public_weighted_trace(estimate))
    }

    /// Compute the exact covariance matrix for a named derived quantity.
    ///
    /// The returned rows use the output ordering of the quantity's
    /// [`LinearMap`]. Hard equality constraints are included through the
    /// low-rank covariance correction implemented in `gmrf-core`.
    pub fn derived_covariance(&mut self, name: &str) -> Result<Vec<Vec<f64>>> {
        let (map, _) = self.derived.get(name).ok_or_else(|| {
            FeecGmrfError::InvalidParameter(format!("unknown derived quantity `{name}`"))
        })?;
        let operator = sparse_row_operator(map)?;
        let covariance = match &self.constraints {
            Some((constraints, _)) => {
                self.gmrf
                    .exact_transformed_covariance_decomposition(&operator, constraints)?
                    .constrained
            }
            None => self.gmrf.exact_transformed_covariance(&operator)?,
        };
        Ok((0..covariance.nrows())
            .map(|row| {
                (0..covariance.ncols())
                    .map(|column| covariance[(row, column)])
                    .collect()
            })
            .collect())
    }

    /// Compute the exact covariance matrix for an ad hoc full- or active-space map.
    pub fn pushforward_covariance(&mut self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        let map = self.active_map(map)?;
        let operator = sparse_row_operator(&map)?;
        let covariance = match &self.constraints {
            Some((constraints, _)) => {
                self.gmrf
                    .exact_transformed_covariance_decomposition(&operator, constraints)?
                    .constrained
            }
            None => self.gmrf.exact_transformed_covariance(&operator)?,
        };
        Ok((0..covariance.nrows())
            .map(|row| {
                (0..covariance.ncols())
                    .map(|column| covariance[(row, column)])
                    .collect()
            })
            .collect())
    }

    fn active_map(&self, map: &LinearMap) -> Result<LinearMap> {
        if map.input_dimension() == self.mean.len() {
            return Ok(map.clone());
        }
        let elimination = self.boundary_elimination.as_ref().ok_or_else(|| {
            FeecGmrfError::Dimension(format!(
                "pushforward input dimension {} does not match posterior dimension {}",
                map.input_dimension(),
                self.mean.len()
            ))
        })?;
        Ok(elimination
            .eliminate_map(map, &vec![0.0; map.output_dimension()])?
            .reduced()
            .clone())
    }

    fn active_pushforward_variance_estimate(
        &mut self,
        map: &LinearMap,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        let operator = sparse_row_operator(map)?;
        match method {
            VarianceMethod::Exact => {
                let values = match &self.constraints {
                    Some((constraints, _)) => {
                        self.gmrf
                            .exact_transformed_variance_decomposition(&operator, constraints)?
                            .constrained_diag
                    }
                    None => {
                        self.gmrf
                            .exact_transformed_variance_decomposition(
                                &operator,
                                &DenseMatrix::zeros(0, self.mean.len()),
                            )?
                            .unconstrained_diag
                    }
                };
                Ok(exact_estimate(values.iter().copied().collect()))
            }
            VarianceMethod::MonteCarlo(config) => {
                let estimate = match &self.constraints {
                    Some((constraints, target)) => {
                        gmrf_core::estimate_monte_carlo_constrained_transformed_variances(
                            &mut self.gmrf,
                            &operator,
                            constraints,
                            target,
                            config.samples,
                            config.batches,
                            config.seed,
                        )?
                    }
                    None => gmrf_core::estimate_monte_carlo_transformed_variances(
                        &mut self.gmrf,
                        &operator,
                        config.samples,
                        config.batches,
                        config.seed,
                    )?,
                };
                Ok(public_estimate(estimate))
            }
            VarianceMethod::Hutchinson(_) | VarianceMethod::Auto { .. } => {
                self.transformed_variance_estimate(&operator, method)
            }
        }
    }

    fn transformed_variance_estimate(
        &self,
        operator: &SparseRowOperator,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        let mode = method.transformed_mode().ok_or_else(|| {
            FeecGmrfError::InvalidParameter(
                "Monte Carlo variance requests require the sampling path".to_string(),
            )
        })?;
        let estimate = match &self.constraints {
            Some((constraints, _)) => {
                let precision = self.gmrf.precision_matrix().ok_or_else(|| {
                    FeecGmrfError::Unsupported(
                        "constrained Hutchinson estimation requires an explicit precision"
                            .to_string(),
                    )
                })?;
                let solver = ConstrainedPrecisionSolver::new(precision, constraints)?;
                estimate_constrained_transformed_variances(&solver, operator, mode)?
            }
            None => {
                let factor = self.gmrf.precision_factor().ok_or_else(|| {
                    FeecGmrfError::Unsupported(
                        "variance estimation requires a reusable sparse precision factor"
                            .to_string(),
                    )
                })?;
                estimate_factored_transformed_variances(factor, operator, mode)?
            }
        };
        Ok(public_estimate(estimate))
    }

    fn active_weighted_variance_estimate(
        &mut self,
        map: &LinearMap,
        output_weights: &[f64],
        method: VarianceMethod,
    ) -> Result<WeightedVarianceEstimate> {
        validate_output_weights(output_weights, map.output_dimension())?;
        if self.constraints.is_none() {
            if let Some(mode) = method.transformed_mode() {
                let operator = sparse_row_operator(map)?;
                let factor = self.required_factor()?;
                let weights = Vector::from_vec(output_weights.to_vec());
                let estimate = estimate_factored_transformed_variance_weighted_trace(
                    factor, &operator, &weights, mode,
                )?;
                return Ok(WeightedVarianceEstimate {
                    variances: public_estimate(estimate.variances),
                    weighted_trace: estimate.weighted_trace.value,
                    weighted_trace_standard_error: estimate.weighted_trace.batch_standard_error,
                    weighted_trace_relative_standard_error: estimate
                        .weighted_trace
                        .relative_standard_error,
                });
            }
        }
        let variances = self.active_pushforward_variance_estimate(map, method)?;
        let weighted_trace = variances
            .values
            .iter()
            .zip(output_weights)
            .map(|(variance, weight)| variance * weight)
            .sum::<f64>();
        if !weighted_trace.is_finite() {
            return Err(FeecGmrfError::Inference(
                "weighted covariance trace is non-finite".to_string(),
            ));
        }
        Ok(WeightedVarianceEstimate {
            variances,
            weighted_trace,
            weighted_trace_standard_error: None,
            weighted_trace_relative_standard_error: None,
        })
    }

    fn ensure_unconstrained_trace(&self) -> Result<()> {
        if self.constraints.is_some() {
            return Err(FeecGmrfError::Unsupported(
                "general sparse weighted traces with hard equality constraints are not yet exposed"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn required_factor(&self) -> Result<&gmrf_core::types::SparseCholeskyFactor> {
        self.gmrf.precision_factor().ok_or_else(|| {
            FeecGmrfError::Unsupported(
                "weighted covariance traces require a reusable sparse precision factor".to_string(),
            )
        })
    }
}

fn exact_estimate(values: Vec<f64>) -> VarianceEstimate {
    let minimum_value = values.iter().copied().fold(f64::INFINITY, f64::min);
    let negative_count = values.iter().filter(|value| **value < 0.0).count();
    VarianceEstimate {
        values,
        estimator: VarianceEstimator::Exact,
        sample_count: 0,
        batch_sizes: Vec::new(),
        batch_standard_error: None,
        relative_standard_error: None,
        negative_count,
        minimum_value: if minimum_value.is_finite() {
            minimum_value
        } else {
            0.0
        },
    }
}

fn lower_triangle_nonzeros(matrix: &SparseMat) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, column, value)| row >= column && *value != 0.0)
        .count()
}

fn public_estimate(estimate: gmrf_core::VarianceEstimate) -> VarianceEstimate {
    VarianceEstimate {
        values: estimate.values.iter().copied().collect(),
        estimator: match estimate.estimator {
            gmrf_core::VarianceEstimator::MonteCarlo => VarianceEstimator::MonteCarlo,
            gmrf_core::VarianceEstimator::Hutchinson => VarianceEstimator::Hutchinson,
            _ => VarianceEstimator::Exact,
        },
        sample_count: estimate.sample_count,
        batch_sizes: estimate.batch_sizes,
        batch_standard_error: estimate
            .batch_standard_error
            .map(|values| values.iter().copied().collect()),
        relative_standard_error: estimate
            .relative_standard_error
            .map(|values| values.iter().copied().collect()),
        negative_count: estimate.num_negative,
        minimum_value: estimate.min_value,
    }
}

fn public_weighted_trace(
    estimate: gmrf_core::WeightedTraceEstimate,
) -> WeightedCovarianceTraceEstimate {
    WeightedCovarianceTraceEstimate {
        value: estimate.value,
        estimator: match estimate.estimator {
            gmrf_core::VarianceEstimator::Hutchinson => VarianceEstimator::Hutchinson,
            gmrf_core::VarianceEstimator::MonteCarlo => VarianceEstimator::MonteCarlo,
            _ => VarianceEstimator::Exact,
        },
        sample_count: estimate.sample_count,
        batch_sizes: estimate.batch_sizes,
        batch_standard_error: estimate.batch_standard_error,
        relative_standard_error: estimate.relative_standard_error,
    }
}

fn lift_variance_estimate(
    estimate: VarianceEstimate,
    elimination: &EssentialBoundaryElimination,
) -> Result<VarianceEstimate> {
    Ok(VarianceEstimate {
        values: elimination.lift_variances(&estimate.values)?,
        batch_standard_error: estimate
            .batch_standard_error
            .as_deref()
            .map(|values| elimination.lift_variances(values))
            .transpose()?,
        relative_standard_error: estimate
            .relative_standard_error
            .as_deref()
            .map(|values| elimination.lift_variances(values))
            .transpose()?,
        estimator: estimate.estimator,
        sample_count: estimate.sample_count,
        batch_sizes: estimate.batch_sizes,
        negative_count: estimate.negative_count,
        minimum_value: estimate.minimum_value,
    })
}

fn validate_output_weights(weights: &[f64], output_dimension: usize) -> Result<()> {
    if weights.len() != output_dimension {
        return Err(FeecGmrfError::Dimension(format!(
            "output weight count {} does not match output dimension {output_dimension}",
            weights.len()
        )));
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(FeecGmrfError::InvalidParameter(
            "output weights must be finite, non-negative, and have positive sum".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn prepare_linear_model(
    builder: LinearGaussianModelBuilder,
) -> Result<PreparedLinearGaussianModel> {
    let prior_precision = gmrf_sparse(builder.prior.precision());
    let prior_mean = Vector::from_vec(builder.prior.mean().to_vec());
    let boundary_elimination = builder.prior.boundary_elimination().cloned();
    let default_values = builder
        .observations
        .iter()
        .map(|observation| observation.values.clone())
        .collect::<Vec<_>>();
    let observations = builder
        .observations
        .iter()
        .map(|observation| PreparedObservation {
            matrix: gmrf_sparse(observation.operator.matrix()),
            bias: Vector::from_vec(observation.bias.clone()),
            noise: match &observation.noise {
                crate::model::GaussianNoise::ScalarVariance(variance) => {
                    PreparedObservationNoise::ScalarVariance(*variance)
                }
                crate::model::GaussianNoise::Precision(precision) => {
                    PreparedObservationNoise::Precision(gmrf_sparse(precision))
                }
            },
        })
        .collect::<Vec<_>>();
    let zero_values = observations
        .iter()
        .map(|observation| Vector::zeros(observation.matrix.nrows()))
        .collect::<Vec<_>>();
    let terms = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| match &observation.noise {
            PreparedObservationNoise::ScalarVariance(variance) => {
                LinearObservationTerm::scalar_variance(
                    &observation.matrix,
                    &zero_values[index],
                    None,
                    *variance,
                )
            }
            PreparedObservationNoise::Precision(precision) => LinearObservationTerm::precision(
                &observation.matrix,
                &zero_values[index],
                None,
                precision,
            ),
        })
        .collect::<Vec<_>>();
    let factored = condition_linear_gaussian_with_factor(&prior_precision, &terms)?;
    let constraints = stack_constraints(&builder.constraints, builder.prior.dimension())?;
    let constraint_solver = constraints
        .as_ref()
        .map(|(matrix, _)| ConstrainedPrecisionSolver::new(&factored.posterior_precision, matrix))
        .transpose()?;
    let public_precision = sparse_mat(&factored.posterior_precision);
    let derived = builder
        .derived
        .into_iter()
        .map(
            |DerivedQuantity {
                 name,
                 operator,
                 bias,
             }| (name, (operator, bias)),
        )
        .collect();
    Ok(PreparedLinearGaussianModel {
        prior_mean,
        observations,
        default_values,
        posterior_precision: factored.posterior_precision,
        posterior_factor: factored.posterior_factor,
        public_precision,
        constraints,
        constraint_solver,
        derived,
        boundary_elimination,
    })
}

pub(crate) fn condition_linear_model(builder: LinearGaussianModelBuilder) -> Result<Posterior> {
    let prior_precision = gmrf_sparse(builder.prior.precision());
    let prior_mean = builder.prior.mean().to_vec();
    let boundary_elimination = builder.prior.boundary_elimination().cloned();

    let matrices = builder
        .observations
        .iter()
        .map(|term| gmrf_sparse(term.operator.matrix()))
        .collect::<Vec<_>>();
    let observations = builder
        .observations
        .iter()
        .map(|term| {
            let prior_pushforward = term.operator.apply(&prior_mean)?;
            Ok(Vector::from_vec(
                term.values
                    .iter()
                    .zip(&term.bias)
                    .zip(prior_pushforward)
                    .map(|((value, bias), prior)| value - bias - prior)
                    .collect(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let noise_precisions = builder
        .observations
        .iter()
        .map(|term| match &term.noise {
            crate::model::GaussianNoise::ScalarVariance(_) => None,
            crate::model::GaussianNoise::Precision(precision) => Some(gmrf_sparse(precision)),
        })
        .collect::<Vec<_>>();
    let terms = builder
        .observations
        .iter()
        .enumerate()
        .map(|(index, term)| match &term.noise {
            crate::model::GaussianNoise::ScalarVariance(variance) => {
                LinearObservationTerm::scalar_variance(
                    &matrices[index],
                    &observations[index],
                    None,
                    *variance,
                )
            }
            crate::model::GaussianNoise::Precision(_) => LinearObservationTerm::precision(
                &matrices[index],
                &observations[index],
                None,
                noise_precisions[index]
                    .as_ref()
                    .expect("precision noise was converted above"),
            ),
        })
        .collect::<Vec<_>>();
    let factored = condition_linear_gaussian_with_factor(&prior_precision, &terms)?;
    let unconstrained_mean = prior_mean
        .iter()
        .zip(factored.posterior_mean.iter())
        .map(|(prior, delta)| prior + delta)
        .collect::<Vec<_>>();

    let information = factored.posterior_precision.mul_vec(&Vector::from_iterator(
        unconstrained_mean.len(),
        unconstrained_mean.iter().copied(),
    ));
    let mut gmrf = Gmrf::from_information_and_precision_with_sqrt(
        information,
        factored.posterior_precision.clone(),
        factored.posterior_factor,
    )?;

    let constraints = stack_constraints(&builder.constraints, builder.prior.dimension())?;
    let mean = match &constraints {
        Some((matrix, target)) => gmrf
            .constrained_mean(matrix, target)?
            .iter()
            .copied()
            .collect(),
        None => unconstrained_mean,
    };
    let precision = sparse_mat(&factored.posterior_precision);
    let derived = builder
        .derived
        .into_iter()
        .map(
            |DerivedQuantity {
                 name,
                 operator,
                 bias,
             }| (name, (operator, bias)),
        )
        .collect();
    let cochain_mean = match &boundary_elimination {
        Some(elimination) => elimination.lift_state(&mean)?,
        None => mean.clone(),
    };
    Ok(Posterior {
        gmrf,
        mean,
        cochain_mean,
        precision,
        constraints,
        derived,
        boundary_elimination,
    })
}

pub(crate) fn condition_linear_pde_model(
    builder: LinearGaussianModelBuilder,
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
) -> Result<Posterior> {
    let boundary_elimination = builder.prior.boundary_elimination().cloned();
    let prior = feg_core::GaussianPriorSpec {
        mean: builder.prior.mean().to_vec(),
        precision: builder.prior.precision().clone(),
    };
    let observations = builder
        .observations
        .iter()
        .map(
            |term| feg_infer::linear_pde::StateOnlyLinearObservationSpec {
                operator: term.operator.matrix().clone(),
                observations: term.values.clone(),
                bias: term.bias.clone(),
                noise: match &term.noise {
                    crate::model::GaussianNoise::ScalarVariance(variance) => {
                        feg_infer::linear_pde::StateOnlyLinearObservationNoise::ScalarVariance(
                            *variance,
                        )
                    }
                    crate::model::GaussianNoise::Precision(precision) => {
                        feg_infer::linear_pde::StateOnlyLinearObservationNoise::Precision(
                            precision.clone(),
                        )
                    }
                },
            },
        )
        .collect::<Vec<_>>();
    let lower =
        feg_infer::linear_pde::build_state_only_linear_pde_posterior(&prior, system, &observations)
            .map_err(FeecGmrfError::Inference)?;
    let constraints = stack_constraints(&builder.constraints, builder.prior.dimension())?;
    let mut gmrf = lower.posterior;
    let mean = match &constraints {
        Some((matrix, target)) => gmrf
            .constrained_mean(matrix, target)?
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        None => gmrf.mean().iter().copied().collect::<Vec<_>>(),
    };
    let cochain_mean = match &boundary_elimination {
        Some(elimination) => elimination.lift_state(&mean)?,
        None => mean.clone(),
    };
    let derived = builder
        .derived
        .into_iter()
        .map(
            |DerivedQuantity {
                 name,
                 operator,
                 bias,
             }| (name, (operator, bias)),
        )
        .collect();
    Ok(Posterior {
        precision: sparse_mat(&lower.precision),
        gmrf,
        mean,
        cochain_mean,
        constraints,
        derived,
        boundary_elimination,
    })
}

fn stack_constraints(
    constraints: &[crate::model::LinearConstraint],
    dimension: usize,
) -> Result<Option<(DenseMatrix, Vector)>> {
    if constraints.is_empty() {
        return Ok(None);
    }
    let rows = constraints
        .iter()
        .map(|constraint| constraint.operator.output_dimension())
        .sum();
    let mut dense = DenseMatrix::zeros(rows, dimension);
    let mut target = Vec::with_capacity(rows);
    let mut row_offset = 0;
    for constraint in constraints {
        for (row, col, value) in constraint.operator.matrix().triplet_iter() {
            dense[(row_offset + row, col)] += value;
        }
        target.extend_from_slice(&constraint.target);
        row_offset += constraint.operator.output_dimension();
    }
    let target = Vector::from_vec(target);
    Ok(Some((dense, target)))
}

pub(crate) fn gmrf_sparse(matrix: &SparseMat) -> SparseMatrix {
    let mut coo = CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, value);
    }
    SparseMatrix::from(&coo)
}

fn sparse_mat(matrix: &SparseMatrix) -> SparseMat {
    SparseMat::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| feg_core::SparseTriplet {
                row,
                col,
                value: *value,
            }),
    )
}

pub(crate) fn sparse_row_operator(map: &LinearMap) -> Result<SparseRowOperator> {
    SparseRowOperator::from_sparse_matrix(&gmrf_sparse(map.matrix()))
        .map_err(|error| FeecGmrfError::Inference(error.to_string()))
}
