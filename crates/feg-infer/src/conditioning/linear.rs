use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec};
use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use gmrf_core::observation::{
    apply_linear_observation_terms, condition_linear_gaussian_with_factor,
    ht_weighted_observations, LinearObservationTerm,
};
use gmrf_core::types::{
    DenseMatrix as GmrfDenseMatrix, SparseCholeskyFactor, SparseMatrix as GmrfSparseMatrix,
    Vector as GmrfVector,
};
use gmrf_core::{
    estimate_batched_transformed_hutchinson_decomposition,
    estimate_factored_transformed_variance_weighted_trace, estimate_factored_transformed_variances,
    Gmrf, GmrfError, ProbeBatchConfig, SparseRowOperator, TransformedVarianceDecomposition,
    TransformedVarianceMode, VarianceEstimator, VarianceFloor,
};
use std::collections::BTreeMap;
use std::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedVarianceMode {
    Exact,
    Hutchinson,
}

#[derive(Debug, Clone)]
pub struct DerivedOperator {
    pub operator: SparseRowOperator,
    pub variance_mode: DerivedVarianceMode,
}

pub type DerivedOperatorSet = BTreeMap<String, DerivedOperator>;

#[derive(Debug, Clone)]
pub struct HarmonicSubspace {
    pub basis: FeecMatrix,
    pub constraints: GmrfDenseMatrix,
    pub projector: Option<SparseRowOperator>,
}

pub type HutchinsonConfig = ProbeBatchConfig;

#[derive(Debug, Clone)]
pub struct LinearGaussianConditioningProblem {
    pub prior_precision: GmrfSparseMatrix,
    pub observation_operator: GmrfSparseMatrix,
    pub observations: GmrfVector,
    pub noise_variance: f64,
    pub harmonic_subspace: Option<HarmonicSubspace>,
    pub derived_operators: DerivedOperatorSet,
    pub hutchinson: HutchinsonConfig,
}

#[derive(Debug, Clone)]
pub struct LinearGaussianConditioningResult {
    pub observations: GmrfVector,
    pub posterior_precision: GmrfSparseMatrix,
    pub information: GmrfVector,
    pub posterior_mean: GmrfVector,
    pub constrained_posterior_mean: Option<GmrfVector>,
    pub posterior_observations: GmrfVector,
    pub observation_residual: GmrfVector,
    pub prior_latent_variance: TransformedVarianceDecomposition,
    pub posterior_latent_variance: TransformedVarianceDecomposition,
    pub derived_prior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
    pub derived_posterior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeecWeightedObservationRow {
    pub entries: Vec<(usize, f64)>,
    pub observed_value: f64,
    pub noise_variance: f64,
}

#[derive(Debug, Clone)]
pub struct FeecLinearConditioningResult {
    pub posterior_precision: GmrfSparseMatrix,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub information: GmrfVector,
    pub latent_mean: FeecVector,
    pub observed_mean: FeecVector,
}

#[derive(Debug, Clone)]
pub struct FeecDerivedVarianceRequest {
    pub name: String,
    pub operator: SparseRowOperator,
    pub mode: TransformedVarianceMode,
}

#[derive(Debug, Clone)]
pub struct FeecDerivedWeightedTraceRequest {
    pub name: String,
    pub operator: SparseRowOperator,
    pub output_weights: FeecVector,
    pub mode: TransformedVarianceMode,
}

#[derive(Debug, Clone)]
pub struct FeecDerivedWeightedTraceResult {
    pub variances: FeecVector,
    pub weighted_trace: f64,
    pub estimator: VarianceEstimator,
    pub sample_count: usize,
}

pub fn feec_observation_matrix(
    observations: &[FeecWeightedObservationRow],
    ncols: usize,
    scaled: bool,
    variance_floor: f64,
) -> Result<FeecCsr, Box<dyn Error>> {
    if ncols == 0 {
        return Err("observation matrix must have at least one column".into());
    }
    validate_variance_floor(variance_floor)?;

    let mut coo = FeecCoo::new(observations.len(), ncols);
    for (row, observation) in observations.iter().enumerate() {
        let scale = if scaled {
            observation_scale(observation.noise_variance, variance_floor)?
        } else {
            if !observation.noise_variance.is_finite() {
                return Err("observation variance must be finite".into());
            }
            1.0
        };
        for (col, value) in &observation.entries {
            if *col >= ncols {
                return Err(format!("observation column {col} exceeds dimension {ncols}").into());
            }
            if !value.is_finite() {
                return Err("observation row contains non-finite entry".into());
            }
            if *value != 0.0 {
                coo.push(row, *col, scale * *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

pub fn feec_scaled_observation_system(
    observations: &[FeecWeightedObservationRow],
    ncols: usize,
    variance_floor: f64,
) -> Result<(FeecCsr, FeecVector), Box<dyn Error>> {
    let matrix = feec_observation_matrix(observations, ncols, true, variance_floor)?;
    let values = FeecVector::from_iterator(
        observations.len(),
        observations
            .iter()
            .map(|observation| {
                let scale = observation_scale(observation.noise_variance, variance_floor)?;
                if !observation.observed_value.is_finite() {
                    return Err("observation value must be finite".into());
                }
                Ok(scale * observation.observed_value)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
    );
    Ok((matrix, values))
}

pub fn feec_latent_observation_system(
    latent_to_observed: &FeecCsr,
    observed_offset: &FeecVector,
    observations: &[FeecWeightedObservationRow],
    observed_dimension: usize,
    variance_floor: f64,
) -> Result<(FeecCsr, FeecVector), Box<dyn Error>> {
    if latent_to_observed.nrows() != observed_dimension {
        return Err("latent-to-observed row count must match observed dimension".into());
    }
    if observed_offset.len() != observed_dimension {
        return Err("observed offset length must match observed dimension".into());
    }

    let (scaled_matrix, scaled_values) =
        feec_scaled_observation_system(observations, observed_dimension, variance_floor)?;
    let offset_values = &scaled_matrix * observed_offset;
    let adjusted_values = &scaled_values - &offset_values;
    let latent_observation = &scaled_matrix * latent_to_observed;
    Ok((latent_observation, adjusted_values))
}

pub fn condition_feec_linear_observation_model(
    prior_precision: &FeecCsr,
    latent_to_observed: &FeecCsr,
    observed_offset: &FeecVector,
    observations: &[FeecWeightedObservationRow],
    observed_dimension: usize,
    variance_floor: f64,
) -> Result<FeecLinearConditioningResult, Box<dyn Error>> {
    if prior_precision.nrows() != prior_precision.ncols() {
        return Err("prior precision must be square".into());
    }
    if latent_to_observed.ncols() != prior_precision.nrows() {
        return Err("latent-to-observed column count must match prior dimension".into());
    }

    let (latent_observation, adjusted_values) = feec_latent_observation_system(
        latent_to_observed,
        observed_offset,
        observations,
        observed_dimension,
        variance_floor,
    )?;
    let prior_precision = feec_csr_to_gmrf(prior_precision);
    let latent_observation = feec_csr_to_gmrf(&latent_observation);
    let adjusted_values = feec_vec_to_gmrf(&adjusted_values);
    let factored = condition_linear_gaussian_with_factor(
        &prior_precision,
        &[LinearObservationTerm::scalar_variance(
            &latent_observation,
            &adjusted_values,
            None,
            1.0,
        )],
    )?;
    let posterior_precision_nnz = factored.posterior_precision.nnz();
    let posterior_factor_nnz = factored.posterior_factor.nnz();
    let latent_mean = gmrf_vec_to_feec(&factored.posterior_mean);
    let latent_observed_mean = latent_to_observed * &latent_mean;
    let mut observed_mean = observed_offset.clone();
    observed_mean += latent_observed_mean;

    Ok(FeecLinearConditioningResult {
        posterior_precision: factored.posterior_precision,
        posterior_precision_nnz,
        posterior_factor_nnz,
        information: factored.information,
        latent_mean,
        observed_mean,
    })
}

pub fn estimate_factored_feec_derived_variances(
    factor: &SparseCholeskyFactor,
    requests: &[FeecDerivedVarianceRequest],
) -> Result<BTreeMap<String, FeecVector>, Box<dyn Error>> {
    let mut out = BTreeMap::new();
    for request in requests {
        let estimate =
            estimate_factored_transformed_variances(factor, &request.operator, request.mode)?;
        if out
            .insert(request.name.clone(), gmrf_vec_to_feec(&estimate.values))
            .is_some()
        {
            return Err(format!("duplicate derived variance request `{}`", request.name).into());
        }
    }
    Ok(out)
}

pub fn estimate_feec_derived_variances(
    precision: &GmrfSparseMatrix,
    requests: &[FeecDerivedVarianceRequest],
) -> Result<BTreeMap<String, FeecVector>, Box<dyn Error>> {
    let factor = precision.cholesky_sqrt_lower()?;
    estimate_factored_feec_derived_variances(&factor, requests)
}

pub fn estimate_factored_feec_derived_weighted_traces(
    factor: &SparseCholeskyFactor,
    requests: &[FeecDerivedWeightedTraceRequest],
) -> Result<BTreeMap<String, FeecDerivedWeightedTraceResult>, Box<dyn Error>> {
    let mut out = BTreeMap::new();
    for request in requests {
        let weights = feec_vec_to_gmrf(&request.output_weights);
        let estimate = estimate_factored_transformed_variance_weighted_trace(
            factor,
            &request.operator,
            &weights,
            request.mode,
        )?;
        let result = FeecDerivedWeightedTraceResult {
            variances: gmrf_vec_to_feec(&estimate.variances.values),
            weighted_trace: estimate.weighted_trace.value,
            estimator: estimate.weighted_trace.estimator,
            sample_count: estimate.weighted_trace.sample_count,
        };
        if out.insert(request.name.clone(), result).is_some() {
            return Err(format!("duplicate weighted trace request `{}`", request.name).into());
        }
    }
    Ok(out)
}

pub fn estimate_feec_derived_weighted_traces(
    precision: &GmrfSparseMatrix,
    requests: &[FeecDerivedWeightedTraceRequest],
) -> Result<BTreeMap<String, FeecDerivedWeightedTraceResult>, Box<dyn Error>> {
    let factor = precision.cholesky_sqrt_lower()?;
    estimate_factored_feec_derived_weighted_traces(&factor, requests)
}

fn validate_variance_floor(variance_floor: f64) -> Result<(), Box<dyn Error>> {
    if !variance_floor.is_finite() || variance_floor < 0.0 {
        return Err("variance floor must be finite and non-negative".into());
    }
    Ok(())
}

fn validate_observation_variance(variance: f64, variance_floor: f64) -> Result<(), Box<dyn Error>> {
    if !variance.is_finite() {
        return Err("observation variance must be finite".into());
    }
    if variance.max(variance_floor) <= 0.0 {
        return Err("observation variance must be positive after flooring".into());
    }
    Ok(())
}

fn observation_scale(variance: f64, variance_floor: f64) -> Result<f64, Box<dyn Error>> {
    validate_observation_variance(variance, variance_floor)?;
    Ok(variance.max(variance_floor).sqrt().recip())
}

#[derive(Debug, Clone)]
pub struct PreparedLinearGaussianConditioningProblem {
    observation_operator: GmrfSparseMatrix,
    noise_variance: f64,
    harmonic_subspace: Option<HarmonicSubspace>,
    posterior_precision: GmrfSparseMatrix,
    prior_latent_variance: TransformedVarianceDecomposition,
    posterior_latent_variance: TransformedVarianceDecomposition,
    derived_prior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
    derived_posterior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
}

impl LinearGaussianConditioningProblem {
    pub fn solve(&self) -> Result<LinearGaussianConditioningResult, Box<dyn Error>> {
        let prepared = self.prepare()?;
        prepared.solve_with_observations(&self.observations)
    }

    pub fn prepare(&self) -> Result<PreparedLinearGaussianConditioningProblem, Box<dyn Error>> {
        self.validate()?;

        let zero_observations = GmrfVector::zeros(self.observation_operator.nrows());
        let (posterior_precision, _) = apply_linear_observation_terms(
            &self.prior_precision,
            &[LinearObservationTerm::scalar_variance(
                &self.observation_operator,
                &zero_observations,
                None,
                self.noise_variance,
            )],
        );

        let harmonic_constraints = self.harmonic_subspace.as_ref().map_or_else(
            || GmrfDenseMatrix::zeros(0, self.prior_precision.nrows()),
            |subspace| subspace.constraints.clone(),
        );

        let prior_factor = self.prior_precision.cholesky_sqrt_lower()?;
        let mut prior = Gmrf::from_mean_and_precision(
            GmrfVector::zeros(self.prior_precision.nrows()),
            self.prior_precision.clone(),
        )?
        .with_precision_sqrt(prior_factor);
        let prior_latent = prior.exact_constrained_variance_decomposition(&harmonic_constraints)?;
        let prior_latent_variance = TransformedVarianceDecomposition {
            unconstrained_diag: prior_latent.unconstrained_diag,
            constrained_diag: prior_latent.constrained_diag,
            removed_diag: prior_latent.removed_diag,
        };

        let posterior_factor = posterior_precision.cholesky_sqrt_lower()?;
        let mut posterior = Gmrf::from_mean_and_precision(
            GmrfVector::zeros(posterior_precision.nrows()),
            posterior_precision.clone(),
        )?
        .with_precision_sqrt(posterior_factor);
        let posterior_latent =
            posterior.exact_constrained_variance_decomposition(&harmonic_constraints)?;
        let posterior_latent_variance = TransformedVarianceDecomposition {
            unconstrained_diag: posterior_latent.unconstrained_diag,
            constrained_diag: posterior_latent.constrained_diag,
            removed_diag: posterior_latent.removed_diag,
        };

        let mut derived_prior_variances = BTreeMap::new();
        let mut derived_posterior_variances = BTreeMap::new();
        for (name, derived) in &self.derived_operators {
            let prior_decomposition = match derived.variance_mode {
                DerivedVarianceMode::Exact => exact_or_hutchinson_decomposition(
                    &mut prior,
                    &self.prior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.hutchinson,
                )?,
                DerivedVarianceMode::Hutchinson => estimate_hutchinson_decomposition(
                    &self.prior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.hutchinson,
                )?,
            };
            let posterior_decomposition = match derived.variance_mode {
                DerivedVarianceMode::Exact => exact_or_hutchinson_decomposition(
                    &mut posterior,
                    &posterior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.hutchinson,
                )?,
                DerivedVarianceMode::Hutchinson => estimate_hutchinson_decomposition(
                    &posterior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.hutchinson,
                )?,
            };

            derived_prior_variances.insert(name.clone(), prior_decomposition);
            derived_posterior_variances.insert(name.clone(), posterior_decomposition);
        }

        Ok(PreparedLinearGaussianConditioningProblem {
            observation_operator: self.observation_operator.clone(),
            noise_variance: self.noise_variance,
            harmonic_subspace: self.harmonic_subspace.clone(),
            posterior_precision,
            prior_latent_variance,
            posterior_latent_variance,
            derived_prior_variances,
            derived_posterior_variances,
        })
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        let state_dim = self.prior_precision.nrows();
        if self.prior_precision.ncols() != state_dim {
            return Err("prior precision must be square".into());
        }
        if self.observation_operator.ncols() != state_dim {
            return Err("observation operator column count must match latent dimension".into());
        }
        if self.observations.len() != self.observation_operator.nrows() {
            return Err("observations length must match observation operator rows".into());
        }
        if !self.noise_variance.is_finite() || self.noise_variance <= 0.0 {
            return Err("noise_variance must be finite and positive".into());
        }
        if self.hutchinson.num_probes == 0 {
            return Err("hutchinson.num_probes must be >= 1".into());
        }
        if self.hutchinson.batch_count == 0 {
            return Err("hutchinson.batch_count must be >= 1".into());
        }
        if let Some(subspace) = &self.harmonic_subspace {
            if subspace.constraints.ncols() != state_dim {
                return Err("harmonic constraint columns must match latent dimension".into());
            }
            if let Some(projector) = &subspace.projector {
                if projector.ncols != state_dim {
                    return Err(
                        "harmonic projector column count must match latent dimension".into(),
                    );
                }
            }
        }
        for derived in self.derived_operators.values() {
            if derived.operator.ncols != state_dim {
                return Err("derived operator column count must match latent dimension".into());
            }
        }
        Ok(())
    }
}

impl PreparedLinearGaussianConditioningProblem {
    pub fn solve_with_observations(
        &self,
        observations: &GmrfVector,
    ) -> Result<LinearGaussianConditioningResult, Box<dyn Error>> {
        if observations.len() != self.observation_operator.nrows() {
            return Err("observations length must match observation operator rows".into());
        }

        let information = ht_weighted_observations(
            &self.observation_operator,
            observations,
            1.0 / self.noise_variance,
        );
        let posterior_precision = self.posterior_precision.clone();
        let mut posterior =
            Gmrf::from_information_and_precision(information.clone(), posterior_precision.clone())?;
        let posterior_mean = posterior.mean().clone();
        let constrained_posterior_mean = self
            .harmonic_subspace
            .as_ref()
            .map(|subspace| {
                posterior.constrained_mean(
                    &subspace.constraints,
                    &GmrfVector::zeros(subspace.constraints.nrows()),
                )
            })
            .transpose()?;

        let posterior_observations = &self.observation_operator * &posterior_mean;
        let observation_residual = &posterior_observations - observations;

        Ok(LinearGaussianConditioningResult {
            observations: observations.clone(),
            posterior_precision,
            information,
            posterior_mean,
            constrained_posterior_mean,
            posterior_observations,
            observation_residual,
            prior_latent_variance: self.prior_latent_variance.clone(),
            posterior_latent_variance: self.posterior_latent_variance.clone(),
            derived_prior_variances: self.derived_prior_variances.clone(),
            derived_posterior_variances: self.derived_posterior_variances.clone(),
        })
    }
}

fn estimate_hutchinson_decomposition(
    precision: &GmrfSparseMatrix,
    operator: &SparseRowOperator,
    constraints: &GmrfDenseMatrix,
    config: HutchinsonConfig,
) -> Result<TransformedVarianceDecomposition, Box<dyn Error>> {
    let factor = precision.cholesky_sqrt_lower()?;
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())?
            .with_precision_sqrt(factor);
    estimate_batched_transformed_hutchinson_decomposition(
        &mut gmrf,
        operator,
        constraints,
        config,
        VarianceFloor::PositiveMean { scale: 1e-12 },
    )
    .map(|estimate| estimate.decomposition)
    .map_err(|err| err.into())
}

fn exact_or_hutchinson_decomposition(
    gmrf: &mut Gmrf,
    precision: &GmrfSparseMatrix,
    operator: &SparseRowOperator,
    constraints: &GmrfDenseMatrix,
    hutchinson: HutchinsonConfig,
) -> Result<TransformedVarianceDecomposition, Box<dyn Error>> {
    match gmrf.exact_transformed_variance_decomposition(operator, constraints) {
        Ok(exact) => Ok(exact),
        Err(GmrfError::NumericalInstability(
            "removed transformed marginal variance exceeded unconstrained variance",
        )) => estimate_hutchinson_decomposition(precision, operator, constraints, hutchinson),
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::observation::ht_weighted_observations;
    use gmrf_core::types::CooMatrix;

    fn assert_sparse_matrix_eq(left: &GmrfSparseMatrix, right: &GmrfSparseMatrix) {
        assert_eq!(left.nrows(), right.nrows());
        assert_eq!(left.ncols(), right.ncols());

        let mut left_entries = left
            .triplet_iter()
            .map(|(row, col, value)| (row, col, *value))
            .collect::<Vec<_>>();
        let mut right_entries = right
            .triplet_iter()
            .map(|(row, col, value)| (row, col, *value))
            .collect::<Vec<_>>();
        left_entries.sort_by(|a, b| a.partial_cmp(b).expect("finite sparse entries"));
        right_entries.sort_by(|a, b| a.partial_cmp(b).expect("finite sparse entries"));
        assert_eq!(left_entries, right_entries);
    }

    fn assert_decomposition_eq(
        left: &TransformedVarianceDecomposition,
        right: &TransformedVarianceDecomposition,
    ) {
        assert_eq!(left.unconstrained_diag, right.unconstrained_diag);
        assert_eq!(left.constrained_diag, right.constrained_diag);
        assert_eq!(left.removed_diag, right.removed_diag);
    }

    fn test_precision() -> GmrfSparseMatrix {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 5.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 4.0);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 3.0);
        GmrfSparseMatrix::from(&coo)
    }

    fn test_observation_operator() -> GmrfSparseMatrix {
        let mut coo = CooMatrix::new(2, 3);
        coo.push(0, 0, 1.0);
        coo.push(1, 2, 1.0);
        GmrfSparseMatrix::from(&coo)
    }

    fn test_problem() -> LinearGaussianConditioningProblem {
        let mut derived_operators = DerivedOperatorSet::new();
        derived_operators.insert(
            "identity".to_string(),
            DerivedOperator {
                operator: SparseRowOperator::identity(3),
                variance_mode: DerivedVarianceMode::Exact,
            },
        );
        derived_operators.insert(
            "sum".to_string(),
            DerivedOperator {
                operator: SparseRowOperator::new(3, vec![vec![(0, 1.0), (1, -0.5)]])
                    .expect("valid sparse row operator"),
                variance_mode: DerivedVarianceMode::Exact,
            },
        );

        LinearGaussianConditioningProblem {
            prior_precision: test_precision(),
            observation_operator: test_observation_operator(),
            observations: GmrfVector::from_vec(vec![0.75, -0.25]),
            noise_variance: 0.2,
            harmonic_subspace: Some(HarmonicSubspace {
                basis: FeecMatrix::zeros(3, 0),
                constraints: GmrfDenseMatrix::from_fn(1, 3, |_, j| if j == 0 { 1.0 } else { 0.0 }),
                projector: None,
            }),
            derived_operators,
            hutchinson: HutchinsonConfig {
                num_probes: 24,
                batch_count: 4,
                rng_seed: 7,
            },
        }
    }

    #[test]
    fn solve_matches_direct_gaussian_update_and_constraint_projection() {
        let problem = test_problem();
        let result = problem.solve().expect("conditioning should succeed");

        let information = ht_weighted_observations(
            &problem.observation_operator,
            &problem.observations,
            1.0 / problem.noise_variance,
        );
        let (posterior_precision, _) = apply_linear_observation_terms(
            &problem.prior_precision,
            &[LinearObservationTerm::scalar_variance(
                &problem.observation_operator,
                &problem.observations,
                None,
                problem.noise_variance,
            )],
        );
        let mut posterior =
            Gmrf::from_information_and_precision(information.clone(), posterior_precision.clone())
                .expect("posterior should build");

        assert_eq!(result.information, information);
        assert_sparse_matrix_eq(&result.posterior_precision, &posterior_precision);
        assert_eq!(result.posterior_mean, posterior.mean().clone());

        let constrained = posterior
            .constrained_mean(
                &problem
                    .harmonic_subspace
                    .as_ref()
                    .expect("harmonic subspace")
                    .constraints,
                &GmrfVector::zeros(1),
            )
            .expect("constrained mean should succeed");
        assert_eq!(
            result.constrained_posterior_mean,
            Some(constrained),
            "shared conditioning core should reuse gmrf-core constrained_mean",
        );
    }

    #[test]
    fn prepare_reuses_precision_and_variances_across_observation_vectors() {
        let problem = test_problem();
        let prepared = problem.prepare().expect("problem should prepare");

        let first = prepared
            .solve_with_observations(&GmrfVector::from_vec(vec![0.75, -0.25]))
            .expect("first solve should succeed");
        let second = prepared
            .solve_with_observations(&GmrfVector::from_vec(vec![-0.25, 0.5]))
            .expect("second solve should succeed");

        assert_sparse_matrix_eq(&first.posterior_precision, &second.posterior_precision);
        assert_eq!(
            first.prior_latent_variance.unconstrained_diag,
            second.prior_latent_variance.unconstrained_diag
        );
        assert_eq!(
            first.posterior_latent_variance.constrained_diag,
            second.posterior_latent_variance.constrained_diag
        );
        assert_eq!(
            first.derived_prior_variances.len(),
            second.derived_prior_variances.len()
        );
        for (name, first_decomposition) in &first.derived_prior_variances {
            assert_decomposition_eq(first_decomposition, &second.derived_prior_variances[name]);
        }
        assert_eq!(
            first.derived_posterior_variances.len(),
            second.derived_posterior_variances.len()
        );
        for (name, first_decomposition) in &first.derived_posterior_variances {
            assert_decomposition_eq(
                first_decomposition,
                &second.derived_posterior_variances[name],
            );
        }
        assert_ne!(first.posterior_mean, second.posterior_mean);
    }

    #[test]
    fn exact_derived_variances_match_direct_gmrf_computation() {
        let problem = test_problem();
        let result = problem.solve().expect("conditioning should succeed");
        let constraints = &problem
            .harmonic_subspace
            .as_ref()
            .expect("harmonic subspace")
            .constraints;

        let mut prior =
            Gmrf::from_mean_and_precision(GmrfVector::zeros(3), problem.prior_precision.clone())
                .expect("prior should build");
        let expected_prior = prior
            .exact_transformed_variance_decomposition(
                &problem.derived_operators["sum"].operator,
                constraints,
            )
            .expect("prior transformed variances should succeed");

        let mut posterior = Gmrf::from_information_and_precision(
            result.information.clone(),
            result.posterior_precision.clone(),
        )
        .expect("posterior should build");
        let expected_posterior = posterior
            .exact_transformed_variance_decomposition(
                &problem.derived_operators["sum"].operator,
                constraints,
            )
            .expect("posterior transformed variances should succeed");

        assert_decomposition_eq(&result.derived_prior_variances["sum"], &expected_prior);
        assert_decomposition_eq(
            &result.derived_posterior_variances["sum"],
            &expected_posterior,
        );
    }

    #[test]
    fn feec_conditioning_adapter_matches_gmrf_observation_update() {
        let mut prior_coo = FeecCoo::new(2, 2);
        prior_coo.push(0, 0, 3.0);
        prior_coo.push(1, 1, 2.0);
        let prior_precision = FeecCsr::from(&prior_coo);

        let mut latent_to_observed_coo = FeecCoo::new(3, 2);
        latent_to_observed_coo.push(0, 0, 1.0);
        latent_to_observed_coo.push(1, 1, -2.0);
        latent_to_observed_coo.push(2, 0, 0.5);
        latent_to_observed_coo.push(2, 1, 1.0);
        let latent_to_observed = FeecCsr::from(&latent_to_observed_coo);
        let observed_offset = FeecVector::from_vec(vec![0.25, -0.5, 1.0]);
        let observations = vec![
            FeecWeightedObservationRow {
                entries: vec![(0, 1.0), (2, 2.0)],
                observed_value: 1.25,
                noise_variance: 0.5,
            },
            FeecWeightedObservationRow {
                entries: vec![(1, -1.0)],
                observed_value: -0.75,
                noise_variance: 2.0,
            },
        ];

        let result = condition_feec_linear_observation_model(
            &prior_precision,
            &latent_to_observed,
            &observed_offset,
            &observations,
            3,
            1e-14,
        )
        .expect("FEEC adapter should condition");

        let (scaled_matrix, scaled_values) =
            feec_scaled_observation_system(&observations, 3, 1e-14).unwrap();
        let adjusted_values = &scaled_values - &(&scaled_matrix * &observed_offset);
        let latent_observation = &scaled_matrix * &latent_to_observed;
        let expected_observation = feec_csr_to_gmrf(&latent_observation);
        let expected_values = feec_vec_to_gmrf(&adjusted_values);
        let (expected_precision, expected_information) = apply_linear_observation_terms(
            &feec_csr_to_gmrf(&prior_precision),
            &[LinearObservationTerm::scalar_variance(
                &expected_observation,
                &expected_values,
                None,
                1.0,
            )],
        );
        let expected_posterior = Gmrf::from_information_and_precision(
            expected_information.clone(),
            expected_precision.clone(),
        )
        .expect("expected posterior should build");

        assert_sparse_matrix_eq(&result.posterior_precision, &expected_precision);
        assert_eq!(result.information, expected_information);
        assert_eq!(
            result.latent_mean,
            gmrf_vec_to_feec(expected_posterior.mean())
        );
        let mut expected_observed_mean = observed_offset;
        expected_observed_mean += &latent_to_observed * &result.latent_mean;
        assert_eq!(result.observed_mean, expected_observed_mean);
    }

    #[test]
    fn feec_derived_variance_requests_match_manual_factor_solves() {
        let precision = test_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::new(
            3,
            vec![vec![(0, 1.0), (1, -0.5)], vec![(1, 2.0), (2, 0.25)]],
        )
        .unwrap();
        let requests = vec![FeecDerivedVarianceRequest {
            name: "field".to_string(),
            operator: operator.clone(),
            mode: TransformedVarianceMode::Exact,
        }];

        let estimates = estimate_factored_feec_derived_variances(&factor, &requests).unwrap();
        let expected = FeecVector::from_iterator(
            operator.nrows(),
            (0..operator.nrows()).map(|row| {
                let rhs = operator.row_as_vector(row).unwrap();
                let solved = factor.solve(&rhs).unwrap();
                rhs.dot(&solved)
            }),
        );

        assert_eq!(estimates["field"], expected);
    }

    #[test]
    fn feec_weighted_trace_requests_reuse_transformed_variances() {
        let precision = test_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::new(
            3,
            vec![vec![(0, 1.0), (1, -0.5)], vec![(1, 2.0), (2, 0.25)]],
        )
        .unwrap();
        let weights = FeecVector::from_vec(vec![0.75, 1.25]);
        let requests = vec![FeecDerivedWeightedTraceRequest {
            name: "energy".to_string(),
            operator: operator.clone(),
            output_weights: weights.clone(),
            mode: TransformedVarianceMode::Exact,
        }];

        let estimates = estimate_factored_feec_derived_weighted_traces(&factor, &requests).unwrap();
        let estimate = &estimates["energy"];
        let expected_trace = estimate.variances.dot(&weights);

        assert_eq!(estimate.estimator, VarianceEstimator::ExactSolves);
        assert_eq!(estimate.sample_count, 1);
        assert!((estimate.weighted_trace - expected_trace).abs() < 1e-12);
    }

    #[test]
    fn feec_unscaled_observation_matrix_allows_zero_variance_rows() {
        let matrix = feec_observation_matrix(
            &[FeecWeightedObservationRow {
                entries: vec![(1, 2.0)],
                observed_value: 0.0,
                noise_variance: 0.0,
            }],
            3,
            false,
            0.0,
        )
        .expect("unscaled derived rows should not require positive noise");
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), 3);
    }
}
