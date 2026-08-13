//! Inference results and adapters to generic GMRF algebra.

use crate::boundary::EssentialBoundaryElimination;
use crate::model::{DerivedQuantity, LinearGaussianModelBuilder};
use crate::operator::{LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
use gmrf_core::observation::{condition_linear_gaussian_with_factor, LinearObservationTerm};
use gmrf_core::types::{CooMatrix, DenseMatrix, SparseMatrix, Vector};
use gmrf_core::{Gmrf, SparseRowOperator};
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

/// Marginal-variance computation requested from a posterior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceMethod {
    /// Deterministic covariance solves.
    Exact,
    /// Seeded posterior Monte Carlo with batch diagnostics.
    MonteCarlo(MonteCarloVarianceConfig),
}

/// Estimator that produced a public variance result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceEstimator {
    /// Exact covariance solves.
    Exact,
    /// Posterior Monte Carlo samples.
    MonteCarlo,
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
        }
    }
}

fn exact_estimate(values: Vec<f64>) -> VarianceEstimate {
    VarianceEstimate {
        values,
        estimator: VarianceEstimator::Exact,
        sample_count: 0,
        batch_sizes: Vec::new(),
        batch_standard_error: None,
        relative_standard_error: None,
    }
}

fn public_estimate(estimate: gmrf_core::VarianceEstimate) -> VarianceEstimate {
    VarianceEstimate {
        values: estimate.values.iter().copied().collect(),
        estimator: match estimate.estimator {
            gmrf_core::VarianceEstimator::MonteCarlo => VarianceEstimator::MonteCarlo,
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
    if !builder.constraints.is_empty() {
        return Err(FeecGmrfError::Unsupported(
            "the state-only linear PDE facade does not currently expose hard constraints"
                .to_string(),
        ));
    }
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
    let mean = lower.posterior.mean().iter().copied().collect::<Vec<_>>();
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
        gmrf: lower.posterior,
        mean,
        cochain_mean,
        constraints: None,
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
