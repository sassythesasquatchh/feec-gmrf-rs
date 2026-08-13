use crate::sparse::{
    core_triplet_to_feec_csr, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec,
    lift_vector_with_layout, reduced_index_map, restrict_columns_and_fold_fixed,
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_core::{
    GaussianPriorSpec, LinearGaussianMeasurementSpec, LinearUncertainInputSpec,
    RepresentationPreference, SparseTripletMatrix,
};
use formoniq::problems::reduced_linear::ReducedLinearPdeAssembly;
use formoniq::reduction::DofLayout;
use gmrf_core::types::{
    CooMatrix as GmrfCoo, DenseMatrix as GmrfDenseMatrix, SparseCholeskyFactor,
    SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector,
};
use gmrf_core::{
    apply_linear_observation_terms, estimate_hutchinson_transformed_variances,
    estimate_hutchinson_variances, estimate_local_rbmc_transformed_variances,
    estimate_local_rbmc_variances, estimate_monte_carlo_transformed_variances,
    estimate_monte_carlo_variances, exact_solve_diag_with_progress,
    exact_solve_transformed_diag_with_progress, selected_inverse_diag,
    selected_inverse_transformed_diag, BlockId, Gmrf, LatentBlockMode,
    LinearObservationStackBuilder, LinearObservationTerm, Permutation, PermutedIndex,
    ProbeDistribution, SparseRowOperator,
};
use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;
use std::time::Instant;

/// Observation noise for the root state-only PDE builder.
#[derive(Debug, Clone)]
pub enum StateOnlyLinearObservationNoise {
    ScalarVariance(f64),
    Precision(SparseTripletMatrix),
}

/// Active-coordinate affine observation used by the state-only PDE adapter.
#[derive(Debug, Clone)]
pub struct StateOnlyLinearObservationSpec {
    pub operator: SparseTripletMatrix,
    pub observations: Vec<f64>,
    pub bias: Vec<f64>,
    pub noise: StateOnlyLinearObservationNoise,
}

/// Factored state-only posterior in physical state coordinates.
pub struct StateOnlyLinearPdePosterior {
    pub posterior: Gmrf,
    pub precision: GmrfSparseMatrix,
}

/// Condition a proper state prior on affine scalar- or precision-weighted observations.
///
/// The reduced PDE assembly supplies the state/layout contract; weak PDE residuals
/// are represented by callers as ordinary affine observations of its operator.
pub fn build_state_only_linear_pde_posterior(
    state_prior: &GaussianPriorSpec,
    system: &ReducedLinearPdeAssembly,
    observations: &[StateOnlyLinearObservationSpec],
) -> Result<StateOnlyLinearPdePosterior, String> {
    state_prior.validate()?;
    if state_prior.dimension() != system.state_dimension() {
        return Err(format!(
            "state prior dimension {} does not match reduced PDE dimension {}",
            state_prior.dimension(),
            system.state_dimension()
        ));
    }
    if system.layout.reduced_dimension() != system.state_dimension() {
        return Err("reduced PDE layout does not match its state dimension".to_string());
    }

    let prior_precision = feec_csr_to_gmrf(&core_triplet_to_feec_csr(&state_prior.precision));
    let prior_mean = GmrfVector::from_vec(state_prior.mean.clone());
    let matrices = observations
        .iter()
        .map(|observation| {
            if observation.operator.nrows() != observation.observations.len()
                || observation.bias.len() != observation.observations.len()
                || observation.operator.ncols() != state_prior.dimension()
            {
                return Err("state-only observation dimensions do not align".to_string());
            }
            Ok(feec_csr_to_gmrf(&core_triplet_to_feec_csr(
                &observation.operator,
            )))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let centered = observations
        .iter()
        .zip(&matrices)
        .map(|(observation, operator)| {
            let prior_output = operator.mul_vec(&prior_mean);
            GmrfVector::from_iterator(
                observation.observations.len(),
                observation
                    .observations
                    .iter()
                    .zip(&observation.bias)
                    .zip(prior_output.iter())
                    .map(|((value, bias), prior)| value - bias - prior),
            )
        })
        .collect::<Vec<_>>();
    let precision_matrices = observations
        .iter()
        .map(|observation| match &observation.noise {
            StateOnlyLinearObservationNoise::ScalarVariance(_) => None,
            StateOnlyLinearObservationNoise::Precision(precision) => {
                Some(feec_csr_to_gmrf(&core_triplet_to_feec_csr(precision)))
            }
        })
        .collect::<Vec<_>>();
    let terms = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| match observation.noise {
            StateOnlyLinearObservationNoise::ScalarVariance(variance) => {
                LinearObservationTerm::scalar_variance(
                    &matrices[index],
                    &centered[index],
                    None,
                    variance,
                )
            }
            StateOnlyLinearObservationNoise::Precision(_) => LinearObservationTerm::precision(
                &matrices[index],
                &centered[index],
                None,
                precision_matrices[index]
                    .as_ref()
                    .expect("precision observation was converted above"),
            ),
        })
        .collect::<Vec<_>>();
    let factored = gmrf_core::condition_linear_gaussian_with_factor(&prior_precision, &terms)
        .map_err(|error| error.to_string())?;
    let mean = GmrfVector::from_iterator(
        state_prior.dimension(),
        state_prior
            .mean
            .iter()
            .zip(factored.posterior_mean.iter())
            .map(|(prior, delta)| prior + delta),
    );
    let information = factored.posterior_precision.mul_vec(&mean);
    let posterior = Gmrf::from_information_and_precision_with_sqrt(
        information,
        factored.posterior_precision.clone(),
        factored.posterior_factor,
    )
    .map_err(|error| error.to_string())?;
    Ok(StateOnlyLinearPdePosterior {
        posterior,
        precision: factored.posterior_precision,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPdeVarianceMode {
    Exact,
    ExactSolves,
    MonteCarlo,
    Hutchinson,
    LocalRbmc,
    SelectedInverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearPdeVarianceConfig {
    pub mode: LinearPdeVarianceMode,
    pub num_variance_probes: usize,
    pub variance_batch_count: usize,
    pub rng_seed: u64,
    pub local_rb_block_size: usize,
}

impl Default for LinearPdeVarianceConfig {
    fn default() -> Self {
        Self {
            mode: LinearPdeVarianceMode::Exact,
            num_variance_probes: 64,
            variance_batch_count: 4,
            rng_seed: 0,
            local_rb_block_size: 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearPdePrecisionPolicy {
    SymmetrizeFailFast { max_relative_asymmetry: f64 },
    DiagonalEquilibrated { max_relative_asymmetry: f64 },
}

impl LinearPdePrecisionPolicy {
    pub fn max_relative_asymmetry(self) -> f64 {
        match self {
            Self::SymmetrizeFailFast {
                max_relative_asymmetry,
            }
            | Self::DiagonalEquilibrated {
                max_relative_asymmetry,
            } => max_relative_asymmetry,
        }
    }

    pub fn uses_diagonal_equilibration(self) -> bool {
        matches!(self, Self::DiagonalEquilibrated { .. })
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::SymmetrizeFailFast { .. } => "symmetrize_fail_fast",
            Self::DiagonalEquilibrated { .. } => "diagonal_equilibrated",
        }
    }
}

impl Default for LinearPdePrecisionPolicy {
    fn default() -> Self {
        Self::SymmetrizeFailFast {
            max_relative_asymmetry: 1.0e-10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LinearPdeUqSolverConfig {
    pub variance: LinearPdeVarianceConfig,
    pub precision_policy: LinearPdePrecisionPolicy,
    pub log_diagnostics: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedInputRepresentation {
    Collapsed,
    Latent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputRepresentationDebug {
    pub name: String,
    pub representation: ResolvedInputRepresentation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearPdeFactorizationDebug {
    pub dimension: usize,
    pub matrix_nnz: usize,
    pub matrix_lower_triangle_nnz: usize,
    pub factor_nnz: usize,
    pub fill_in_ratio_vs_lower_triangle: f64,
    pub factor_numeric_values_mib: f64,
    pub min_diagonal: f64,
    pub max_abs_diagonal: f64,
    pub diagonal_ratio: f64,
    pub max_abs_asymmetry: f64,
    pub max_relative_asymmetry: f64,
    pub precision_policy: LinearPdePrecisionPolicy,
    pub equilibration_min_scale: f64,
    pub equilibration_max_scale: f64,
}

impl LinearPdeFactorizationDebug {
    pub fn skipped(
        dimension: usize,
        matrix_nnz: usize,
        precision_policy: LinearPdePrecisionPolicy,
    ) -> Self {
        Self {
            dimension,
            matrix_nnz,
            matrix_lower_triangle_nnz: 0,
            factor_nnz: 0,
            fill_in_ratio_vs_lower_triangle: 0.0,
            factor_numeric_values_mib: 0.0,
            min_diagonal: f64::NAN,
            max_abs_diagonal: f64::NAN,
            diagonal_ratio: f64::NAN,
            max_abs_asymmetry: 0.0,
            max_relative_asymmetry: 0.0,
            precision_policy,
            equilibration_min_scale: 1.0,
            equilibration_max_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearPdeUqDebug {
    pub input_representations: Vec<InputRepresentationDebug>,
    pub joint_dimension: usize,
    pub flat_state_prior: bool,
    pub prior_factorization: LinearPdeFactorizationDebug,
    pub posterior_factorization: LinearPdeFactorizationDebug,
}

#[derive(Debug, Clone)]
pub struct LinearPdeUqProblem {
    pub state_prior: GaussianPriorSpec,
    pub system: ReducedLinearPdeAssembly,
    pub uncertain_inputs: Vec<LinearUncertainInputSpec>,
    pub physical_measurements: Vec<LinearGaussianMeasurementSpec>,
    pub joint_measurements: Vec<LinearPdeJointMeasurementSpec>,
    pub derived_quantities: Vec<LinearPdeDerivedQuantitySpec>,
    pub joint_derived_quantities: Vec<LinearPdeJointDerivedQuantitySpec>,
    pub pde_variance: Option<f64>,
    pub pde_precision: Option<SparseTripletMatrix>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearPdeLatentCoordinateScaling {
    pub input_name: String,
    pub scale: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearPdeCoordinateScaling {
    pub state_scale: Vec<f64>,
    pub latent_scales: Vec<LinearPdeLatentCoordinateScaling>,
}

#[derive(Debug, Clone)]
pub struct LinearPdeDerivedQuantitySpec {
    pub name: String,
    pub operator: SparseRowOperator,
}

#[derive(Debug, Clone)]
pub struct LinearPdeLatentMeasurementBlockSpec {
    pub input_name: String,
    pub operator: SparseTripletMatrix,
}

#[derive(Debug, Clone)]
pub struct LinearPdeJointMeasurementSpec {
    pub name: String,
    pub state_operator: Option<SparseTripletMatrix>,
    pub latent_operators: Vec<LinearPdeLatentMeasurementBlockSpec>,
    pub observations: Vec<f64>,
    pub bias: Vec<f64>,
    pub variance: f64,
}

#[derive(Debug, Clone)]
pub struct LinearPdeLatentDerivedBlockSpec {
    pub input_name: String,
    pub operator: SparseRowOperator,
}

#[derive(Debug, Clone)]
pub struct LinearPdeJointDerivedQuantitySpec {
    pub name: String,
    pub state_operator: Option<SparseRowOperator>,
    pub latent_operators: Vec<LinearPdeLatentDerivedBlockSpec>,
}

#[derive(Debug, Clone)]
pub struct LinearPdeDerivedMarginalResult {
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
}

#[derive(Debug, Clone)]
pub struct LinearPdeUqResult {
    pub posterior_mean: FeecVector,
    pub posterior_variance: FeecVector,
    pub prior_variance: FeecVector,
    pub derived_variances: BTreeMap<String, LinearPdeDerivedMarginalResult>,
    pub latent_inputs: Vec<LinearPdeLatentInputPosterior>,
    pub reduced_posterior_mean: FeecVector,
    pub reduced_posterior_variance: FeecVector,
    pub pde_residual_mean: FeecVector,
    pub debug: LinearPdeUqDebug,
}

#[derive(Debug, Clone)]
pub struct LinearPdePushforwardCovarianceResult {
    pub names: Vec<String>,
    pub prior_covariance: GmrfDenseMatrix,
    pub posterior_covariance: GmrfDenseMatrix,
}

#[derive(Debug, Clone)]
pub struct LinearPdePushforwardMeanCovarianceResult {
    pub names: Vec<String>,
    pub posterior_mean: Vec<f64>,
    pub joint_posterior_mean: Vec<f64>,
    pub prior_covariance: GmrfDenseMatrix,
    pub posterior_covariance: GmrfDenseMatrix,
    pub debug: LinearPdeUqDebug,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearPdeLatentInputPosterior {
    pub name: String,
    pub offset: usize,
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
}

pub struct LinearPdeJointPosterior {
    pub posterior: Gmrf,
    pub derived_quantities: BTreeMap<String, SparseRowOperator>,
    pub state_dimension: usize,
    pub joint_dimension: usize,
}

struct PreparedLinearPdeProblem {
    state_dimension: usize,
    joint_dimension: usize,
    state_mean: FeecVector,
    flat_state_prior: bool,
    centered_residual_bias: FeecVector,
    pde_operator: GmrfSparseMatrix,
    prior_precision: GmrfSparseMatrix,
    posterior_precision: GmrfSparseMatrix,
    information: GmrfVector,
    derived_quantities: BTreeMap<String, SparseRowOperator>,
    input_representations: Vec<InputRepresentationDebug>,
    latent_input_slices: Vec<LatentInputSlice>,
}

#[derive(Debug, Clone)]
struct LatentInputSlice {
    name: String,
    offset: usize,
    prior_mean: Vec<f64>,
}

struct FactorizedPrecision {
    precision: GmrfSparseMatrix,
    factor: SparseCholeskyFactor,
    debug: LinearPdeFactorizationDebug,
    scale: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct ValidatedCoordinateScaling {
    state_scale: Vec<f64>,
    latent_scale_by_name: BTreeMap<String, Vec<f64>>,
}

pub fn solve_linear_pde_uq(problem: &LinearPdeUqProblem) -> Result<LinearPdeUqResult, String> {
    solve_linear_pde_uq_with_config(problem, &LinearPdeUqSolverConfig::default())
}

pub fn solve_scaled_linear_pde_uq_with_config(
    problem: &LinearPdeUqProblem,
    scaling: &LinearPdeCoordinateScaling,
    config: &LinearPdeUqSolverConfig,
) -> Result<LinearPdeUqResult, String> {
    let validated = validate_coordinate_scaling(problem, scaling)?;
    let scaled_problem = scale_linear_pde_problem(problem, &validated)?;
    let result = solve_linear_pde_uq_with_config(&scaled_problem, config)?;
    unscale_linear_pde_result(result, &scaled_problem.system.layout, &validated)
}

fn validate_coordinate_scaling(
    problem: &LinearPdeUqProblem,
    scaling: &LinearPdeCoordinateScaling,
) -> Result<ValidatedCoordinateScaling, String> {
    validate_scale_vector(
        "state coordinate",
        &scaling.state_scale,
        problem.system.state_dimension(),
    )?;
    let input_dimensions = problem
        .uncertain_inputs
        .iter()
        .map(|input| {
            (
                input.name.as_str(),
                (input.prior.dimension(), input.preference),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut latent_scale_by_name = BTreeMap::new();
    for latent_scale in &scaling.latent_scales {
        let Some((dimension, preference)) = input_dimensions
            .get(latent_scale.input_name.as_str())
            .copied()
        else {
            return Err(format!(
                "coordinate scaling references unknown latent input `{}`",
                latent_scale.input_name
            ));
        };
        if latent_scale_by_name
            .insert(latent_scale.input_name.clone(), latent_scale.scale.clone())
            .is_some()
        {
            return Err(format!(
                "coordinate scaling contains duplicate latent input `{}`",
                latent_scale.input_name
            ));
        }
        validate_scale_vector(
            &format!("latent `{}` coordinate", latent_scale.input_name),
            &latent_scale.scale,
            dimension,
        )?;
        if !matches!(preference, RepresentationPreference::ForceLatent) {
            return Err(format!(
                "coordinate scaling for latent input `{}` requires ForceLatent representation",
                latent_scale.input_name
            ));
        }
    }
    Ok(ValidatedCoordinateScaling {
        state_scale: scaling.state_scale.clone(),
        latent_scale_by_name,
    })
}

fn validate_scale_vector(label: &str, scale: &[f64], expected_len: usize) -> Result<(), String> {
    if scale.len() != expected_len {
        return Err(format!(
            "{label} scale length {} must match dimension {expected_len}",
            scale.len()
        ));
    }
    for (index, value) in scale.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "{label} scale[{index}] must be finite and positive, got {value:.6e}"
            ));
        }
    }
    Ok(())
}

fn scale_linear_pde_problem(
    problem: &LinearPdeUqProblem,
    scaling: &ValidatedCoordinateScaling,
) -> Result<LinearPdeUqProblem, String> {
    let state_scale = scaling.state_scale.as_slice();
    let state_prior = GaussianPriorSpec {
        mean: divide_by_scale(&problem.state_prior.mean, state_scale, "state prior mean")?,
        precision: scale_square_triplet_by_coordinates(
            &problem.state_prior.precision,
            state_scale,
            "state prior precision",
        )?,
    };
    let mut system = problem.system.clone();
    system.operator = scale_feec_csr_columns(&system.operator, state_scale, "PDE state operator")?;

    let uncertain_inputs = problem
        .uncertain_inputs
        .iter()
        .map(|input| scale_uncertain_input(input, scaling))
        .collect::<Result<Vec<_>, _>>()?;
    let physical_measurements = problem
        .physical_measurements
        .iter()
        .map(|measurement| scale_state_measurement(measurement, &system.layout, state_scale))
        .collect::<Result<Vec<_>, _>>()?;
    let joint_measurements = problem
        .joint_measurements
        .iter()
        .map(|measurement| scale_joint_measurement(measurement, &system.layout, scaling))
        .collect::<Result<Vec<_>, _>>()?;
    let derived_quantities = problem
        .derived_quantities
        .iter()
        .map(|derived| scale_state_derived_quantity(derived, &system.layout, state_scale))
        .collect::<Result<Vec<_>, _>>()?;
    let joint_derived_quantities = problem
        .joint_derived_quantities
        .iter()
        .map(|derived| scale_joint_derived_quantity(derived, &system.layout, scaling))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(LinearPdeUqProblem {
        state_prior,
        system,
        uncertain_inputs,
        physical_measurements,
        joint_measurements,
        derived_quantities,
        joint_derived_quantities,
        pde_variance: problem.pde_variance,
        pde_precision: problem.pde_precision.clone(),
    })
}

fn scale_uncertain_input(
    input: &LinearUncertainInputSpec,
    scaling: &ValidatedCoordinateScaling,
) -> Result<LinearUncertainInputSpec, String> {
    let scale = scaling
        .latent_scale_by_name
        .get(&input.name)
        .map(Vec::as_slice);
    let Some(scale) = scale else {
        return Ok(input.clone());
    };
    Ok(LinearUncertainInputSpec {
        name: input.name.clone(),
        operator: scale_triplet_columns(
            &input.operator,
            scale,
            &format!("latent `{}` residual operator", input.name),
        )?,
        prior: GaussianPriorSpec {
            mean: divide_by_scale(
                &input.prior.mean,
                scale,
                &format!("latent `{}` prior mean", input.name),
            )?,
            precision: scale_square_triplet_by_coordinates(
                &input.prior.precision,
                scale,
                &format!("latent `{}` prior precision", input.name),
            )?,
        },
        preference: input.preference,
        collapsed_precision: input.collapsed_precision.clone(),
    })
}

fn scale_state_measurement(
    measurement: &LinearGaussianMeasurementSpec,
    layout: &DofLayout,
    state_scale: &[f64],
) -> Result<LinearGaussianMeasurementSpec, String> {
    Ok(LinearGaussianMeasurementSpec {
        name: measurement.name.clone(),
        operator: scale_full_state_triplet_active_columns(
            &measurement.operator,
            layout,
            state_scale,
            &format!("measurement `{}` state operator", measurement.name),
        )?,
        observations: measurement.observations.clone(),
        bias: measurement.bias.clone(),
        variance: measurement.variance,
    })
}

fn scale_joint_measurement(
    measurement: &LinearPdeJointMeasurementSpec,
    layout: &DofLayout,
    scaling: &ValidatedCoordinateScaling,
) -> Result<LinearPdeJointMeasurementSpec, String> {
    Ok(LinearPdeJointMeasurementSpec {
        name: measurement.name.clone(),
        state_operator: measurement
            .state_operator
            .as_ref()
            .map(|operator| {
                scale_full_state_triplet_active_columns(
                    operator,
                    layout,
                    &scaling.state_scale,
                    &format!("joint measurement `{}` state operator", measurement.name),
                )
            })
            .transpose()?,
        latent_operators: measurement
            .latent_operators
            .iter()
            .map(|block| {
                let scale = scaling
                    .latent_scale_by_name
                    .get(&block.input_name)
                    .map(Vec::as_slice);
                Ok(LinearPdeLatentMeasurementBlockSpec {
                    input_name: block.input_name.clone(),
                    operator: if let Some(scale) = scale {
                        scale_triplet_columns(
                            &block.operator,
                            scale,
                            &format!(
                                "joint measurement `{}` latent `{}` operator",
                                measurement.name, block.input_name
                            ),
                        )?
                    } else {
                        block.operator.clone()
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        observations: measurement.observations.clone(),
        bias: measurement.bias.clone(),
        variance: measurement.variance,
    })
}

fn scale_state_derived_quantity(
    derived: &LinearPdeDerivedQuantitySpec,
    layout: &DofLayout,
    state_scale: &[f64],
) -> Result<LinearPdeDerivedQuantitySpec, String> {
    Ok(LinearPdeDerivedQuantitySpec {
        name: derived.name.clone(),
        operator: scale_full_state_sparse_row_active_columns(
            &derived.operator,
            layout,
            state_scale,
            &format!("derived `{}` state operator", derived.name),
        )?,
    })
}

fn scale_joint_derived_quantity(
    derived: &LinearPdeJointDerivedQuantitySpec,
    layout: &DofLayout,
    scaling: &ValidatedCoordinateScaling,
) -> Result<LinearPdeJointDerivedQuantitySpec, String> {
    Ok(LinearPdeJointDerivedQuantitySpec {
        name: derived.name.clone(),
        state_operator: derived
            .state_operator
            .as_ref()
            .map(|operator| {
                scale_full_state_sparse_row_active_columns(
                    operator,
                    layout,
                    &scaling.state_scale,
                    &format!("joint derived `{}` state operator", derived.name),
                )
            })
            .transpose()?,
        latent_operators: derived
            .latent_operators
            .iter()
            .map(|block| {
                let scale = scaling
                    .latent_scale_by_name
                    .get(&block.input_name)
                    .map(Vec::as_slice);
                Ok(LinearPdeLatentDerivedBlockSpec {
                    input_name: block.input_name.clone(),
                    operator: if let Some(scale) = scale {
                        scale_sparse_row_columns(
                            &block.operator,
                            scale,
                            &format!(
                                "joint derived `{}` latent `{}` operator",
                                derived.name, block.input_name
                            ),
                        )?
                    } else {
                        block.operator.clone()
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

fn unscale_linear_pde_result(
    mut result: LinearPdeUqResult,
    layout: &DofLayout,
    scaling: &ValidatedCoordinateScaling,
) -> Result<LinearPdeUqResult, String> {
    multiply_reduced_vector_by_scale(
        &mut result.reduced_posterior_mean,
        &scaling.state_scale,
        "reduced posterior mean",
    )?;
    multiply_reduced_variance_by_scale(
        &mut result.reduced_posterior_variance,
        &scaling.state_scale,
        "reduced posterior variance",
    )?;
    multiply_full_vector_active_by_scale(
        &mut result.posterior_mean,
        layout,
        &scaling.state_scale,
        "posterior mean",
    )?;
    multiply_full_variance_active_by_scale(
        &mut result.posterior_variance,
        layout,
        &scaling.state_scale,
        "posterior variance",
    )?;
    multiply_full_variance_active_by_scale(
        &mut result.prior_variance,
        layout,
        &scaling.state_scale,
        "prior variance",
    )?;
    for input in &mut result.latent_inputs {
        if let Some(scale) = scaling.latent_scale_by_name.get(&input.name) {
            if input.mean.len() != scale.len() || input.variance.len() != scale.len() {
                return Err(format!(
                    "latent `{}` posterior dimension does not match coordinate scale",
                    input.name
                ));
            }
            for (value, scale) in input.mean.iter_mut().zip(scale) {
                *value *= *scale;
            }
            for (value, scale) in input.variance.iter_mut().zip(scale) {
                *value *= *scale * *scale;
            }
        }
    }
    Ok(result)
}

fn divide_by_scale(values: &[f64], scale: &[f64], label: &str) -> Result<Vec<f64>, String> {
    if values.len() != scale.len() {
        return Err(format!(
            "{label} length {} must match scale length {}",
            values.len(),
            scale.len()
        ));
    }
    Ok(values
        .iter()
        .zip(scale)
        .map(|(value, scale)| value / scale)
        .collect())
}

fn scale_triplet_columns(
    matrix: &SparseTripletMatrix,
    column_scale: &[f64],
    label: &str,
) -> Result<SparseTripletMatrix, String> {
    if matrix.ncols() != column_scale.len() {
        return Err(format!(
            "{label} column count {} must match scale length {}",
            matrix.ncols(),
            column_scale.len()
        ));
    }
    Ok(SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| feg_core::SparseTriplet {
                row,
                col,
                value: value * column_scale[col],
            }),
    ))
}

fn scale_feec_csr_columns(
    matrix: &FeecCsr,
    column_scale: &[f64],
    label: &str,
) -> Result<FeecCsr, String> {
    if matrix.ncols() != column_scale.len() {
        return Err(format!(
            "{label} column count {} must match scale length {}",
            matrix.ncols(),
            column_scale.len()
        ));
    }
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        let scaled = *value * column_scale[col];
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn scale_square_triplet_by_coordinates(
    matrix: &SparseTripletMatrix,
    coordinate_scale: &[f64],
    label: &str,
) -> Result<SparseTripletMatrix, String> {
    if matrix.nrows() != coordinate_scale.len() || matrix.ncols() != coordinate_scale.len() {
        return Err(format!(
            "{label} dimension {}x{} must match scale length {}",
            matrix.nrows(),
            matrix.ncols(),
            coordinate_scale.len()
        ));
    }
    Ok(SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| feg_core::SparseTriplet {
                row,
                col,
                value: value * coordinate_scale[row] * coordinate_scale[col],
            }),
    ))
}

fn scale_full_state_triplet_active_columns(
    matrix: &SparseTripletMatrix,
    layout: &DofLayout,
    state_scale: &[f64],
    label: &str,
) -> Result<SparseTripletMatrix, String> {
    if matrix.ncols() != layout.full_dimension {
        return Err(format!(
            "{label} column count {} must match full state dimension {}",
            matrix.ncols(),
            layout.full_dimension
        ));
    }
    if state_scale.len() != layout.reduced_dimension() {
        return Err(format!(
            "{label} state scale length {} must match reduced dimension {}",
            state_scale.len(),
            layout.reduced_dimension()
        ));
    }
    let reduced_map = reduced_index_map(layout);
    Ok(SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| feg_core::SparseTriplet {
                row,
                col,
                value: if let Some(reduced_col) = reduced_map[col] {
                    value * state_scale[reduced_col]
                } else {
                    value
                },
            }),
    ))
}

fn scale_sparse_row_columns(
    operator: &SparseRowOperator,
    column_scale: &[f64],
    label: &str,
) -> Result<SparseRowOperator, String> {
    if operator.ncols != column_scale.len() {
        return Err(format!(
            "{label} column count {} must match scale length {}",
            operator.ncols,
            column_scale.len()
        ));
    }
    SparseRowOperator::new(
        operator.ncols,
        operator
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(col, value)| (*col, *value * column_scale[*col]))
                    .collect()
            })
            .collect(),
    )
    .map_err(|err| err.to_string())
}

fn scale_full_state_sparse_row_active_columns(
    operator: &SparseRowOperator,
    layout: &DofLayout,
    state_scale: &[f64],
    label: &str,
) -> Result<SparseRowOperator, String> {
    if operator.ncols != layout.full_dimension {
        return Err(format!(
            "{label} column count {} must match full state dimension {}",
            operator.ncols, layout.full_dimension
        ));
    }
    if state_scale.len() != layout.reduced_dimension() {
        return Err(format!(
            "{label} state scale length {} must match reduced dimension {}",
            state_scale.len(),
            layout.reduced_dimension()
        ));
    }
    let reduced_map = reduced_index_map(layout);
    SparseRowOperator::new(
        operator.ncols,
        operator
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(col, value)| {
                        let value = if let Some(reduced_col) = reduced_map[*col] {
                            *value * state_scale[reduced_col]
                        } else {
                            *value
                        };
                        (*col, value)
                    })
                    .collect()
            })
            .collect(),
    )
    .map_err(|err| err.to_string())
}

fn multiply_reduced_vector_by_scale(
    values: &mut FeecVector,
    scale: &[f64],
    label: &str,
) -> Result<(), String> {
    if values.len() != scale.len() {
        return Err(format!(
            "{label} length {} must match scale length {}",
            values.len(),
            scale.len()
        ));
    }
    for (value, scale) in values.iter_mut().zip(scale) {
        *value *= *scale;
    }
    Ok(())
}

fn multiply_reduced_variance_by_scale(
    values: &mut FeecVector,
    scale: &[f64],
    label: &str,
) -> Result<(), String> {
    if values.len() != scale.len() {
        return Err(format!(
            "{label} length {} must match scale length {}",
            values.len(),
            scale.len()
        ));
    }
    for (value, scale) in values.iter_mut().zip(scale) {
        *value *= *scale * *scale;
    }
    Ok(())
}

fn multiply_full_vector_active_by_scale(
    values: &mut FeecVector,
    layout: &DofLayout,
    scale: &[f64],
    label: &str,
) -> Result<(), String> {
    if values.len() != layout.full_dimension {
        return Err(format!(
            "{label} length {} must match full state dimension {}",
            values.len(),
            layout.full_dimension
        ));
    }
    if scale.len() != layout.reduced_dimension() {
        return Err(format!(
            "{label} scale length {} must match reduced dimension {}",
            scale.len(),
            layout.reduced_dimension()
        ));
    }
    for (reduced_index, full_index) in layout.active_dofs.iter().copied().enumerate() {
        values[full_index] *= scale[reduced_index];
    }
    Ok(())
}

fn multiply_full_variance_active_by_scale(
    values: &mut FeecVector,
    layout: &DofLayout,
    scale: &[f64],
    label: &str,
) -> Result<(), String> {
    if values.len() != layout.full_dimension {
        return Err(format!(
            "{label} length {} must match full state dimension {}",
            values.len(),
            layout.full_dimension
        ));
    }
    if scale.len() != layout.reduced_dimension() {
        return Err(format!(
            "{label} scale length {} must match reduced dimension {}",
            scale.len(),
            layout.reduced_dimension()
        ));
    }
    for (reduced_index, full_index) in layout.active_dofs.iter().copied().enumerate() {
        let scale = scale[reduced_index];
        values[full_index] *= scale * scale;
    }
    Ok(())
}

pub fn solve_linear_pde_uq_with_pushforward_covariance<S: AsRef<str>>(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
    qoi_names: &[S],
) -> Result<LinearPdePushforwardCovarianceResult, String> {
    if qoi_names.is_empty() {
        return Err("at least one pushforward QoI name is required".to_string());
    }
    let prepared = prepare_linear_pde_problem(problem, config)?;
    if prepared.flat_state_prior {
        return Err("exact pushforward covariance requires a proper prior precision".to_string());
    }
    let (names, operator) =
        select_scalar_pushforward_operator(&prepared.derived_quantities, qoi_names)?;

    let prior_factorized = factorize_precision_with_diagnostics(
        "pushforward_prior",
        &prepared.prior_precision,
        config,
    )?;
    let prior_operator =
        scale_sparse_row_operator_columns(&operator, prior_factorized.scale.as_deref())?;
    let mut prior =
        gmrf_from_zero_mean_precision(prior_factorized.precision, prior_factorized.factor)
            .map_err(|err| format!("failed to build prior GMRF: {err}"))?;
    let prior_covariance = prior
        .exact_transformed_covariance(&prior_operator)
        .map_err(|err| err.to_string())?;

    let posterior_factorized = factorize_precision_with_diagnostics(
        "pushforward_posterior",
        &prepared.posterior_precision,
        config,
    )?;
    let posterior_operator =
        scale_sparse_row_operator_columns(&operator, posterior_factorized.scale.as_deref())?;
    let posterior_information =
        scale_gmrf_vector(&prepared.information, posterior_factorized.scale.as_deref())?;
    let mut posterior = Gmrf::from_information_and_precision_with_sqrt(
        posterior_information,
        posterior_factorized.precision,
        posterior_factorized.factor,
    )
    .map_err(|err| err.to_string())?;
    let posterior_covariance = posterior
        .exact_transformed_covariance(&posterior_operator)
        .map_err(|err| err.to_string())?;

    Ok(LinearPdePushforwardCovarianceResult {
        names,
        prior_covariance,
        posterior_covariance,
    })
}

pub fn solve_linear_pde_uq_with_pushforward_mean_covariance<S: AsRef<str>>(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
    qoi_names: &[S],
) -> Result<LinearPdePushforwardMeanCovarianceResult, String> {
    if qoi_names.is_empty() {
        return Err("at least one pushforward QoI name is required".to_string());
    }
    let prepared = prepare_linear_pde_problem(problem, config)?;
    if prepared.flat_state_prior {
        return Err("exact pushforward covariance requires a proper prior precision".to_string());
    }
    let (names, operator) =
        select_scalar_pushforward_operator(&prepared.derived_quantities, qoi_names)?;

    let prior_factorized = factorize_precision_with_diagnostics(
        "pushforward_prior",
        &prepared.prior_precision,
        config,
    )?;
    let prior_operator =
        scale_sparse_row_operator_columns(&operator, prior_factorized.scale.as_deref())?;
    let mut prior =
        gmrf_from_zero_mean_precision(prior_factorized.precision, prior_factorized.factor)
            .map_err(|err| format!("failed to build prior GMRF: {err}"))?;
    let prior_covariance = prior
        .exact_transformed_covariance(&prior_operator)
        .map_err(|err| err.to_string())?;
    drop(prior);

    let posterior_factorized = factorize_precision_with_diagnostics(
        "pushforward_posterior",
        &prepared.posterior_precision,
        config,
    )?;
    let posterior_operator =
        scale_sparse_row_operator_columns(&operator, posterior_factorized.scale.as_deref())?;
    let posterior_information =
        scale_gmrf_vector(&prepared.information, posterior_factorized.scale.as_deref())?;
    let mut posterior = Gmrf::from_information_and_precision_with_sqrt(
        posterior_information,
        posterior_factorized.precision,
        posterior_factorized.factor,
    )
    .map_err(|err| err.to_string())?;
    let posterior_working_mean_vector = posterior.mean_vector().clone();
    let posterior_mean_vector = scale_gmrf_vector(
        &posterior_working_mean_vector,
        posterior_factorized.scale.as_deref(),
    )?;
    let posterior_covariance = posterior
        .exact_transformed_covariance(&posterior_operator)
        .map_err(|err| err.to_string())?;
    let posterior_mean = operator
        .apply(&posterior_mean_vector)
        .map_err(|err| err.to_string())?
        .as_slice()
        .to_vec();

    Ok(LinearPdePushforwardMeanCovarianceResult {
        names,
        posterior_mean,
        joint_posterior_mean: posterior_mean_vector.as_slice().to_vec(),
        prior_covariance,
        posterior_covariance,
        debug: LinearPdeUqDebug {
            input_representations: prepared.input_representations,
            joint_dimension: prepared.joint_dimension,
            flat_state_prior: prepared.flat_state_prior,
            prior_factorization: prior_factorized.debug,
            posterior_factorization: posterior_factorized.debug,
        },
    })
}

pub fn build_linear_pde_joint_posterior_with_config(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<LinearPdeJointPosterior, String> {
    if config.precision_policy.uses_diagonal_equilibration() {
        return Err(
            "build_linear_pde_joint_posterior_with_config exposes raw joint posterior access; use a fail-fast non-equilibrated precision policy or the original-coordinate solve/pushforward APIs"
                .to_string(),
        );
    }
    let prepared = prepare_linear_pde_problem(problem, config)?;
    let posterior_factorized =
        factorize_precision_with_diagnostics("posterior", &prepared.posterior_precision, config)?;
    let posterior = Gmrf::from_information_and_precision_with_sqrt(
        prepared.information,
        posterior_factorized.precision,
        posterior_factorized.factor,
    )
    .map_err(|err| err.to_string())?;

    Ok(LinearPdeJointPosterior {
        posterior,
        derived_quantities: prepared.derived_quantities,
        state_dimension: prepared.state_dimension,
        joint_dimension: prepared.joint_dimension,
    })
}

pub fn solve_linear_pde_uq_with_config(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<LinearPdeUqResult, String> {
    let prepared = prepare_linear_pde_problem(problem, config)?;
    let state_dim = prepared.state_dimension;
    let joint_dimension = prepared.joint_dimension;
    let state_mean = prepared.state_mean;
    let flat_state_prior = prepared.flat_state_prior;
    let centered_residual_bias = prepared.centered_residual_bias;
    let pde_operator = prepared.pde_operator;
    let prior_gmrf_precision = prepared.prior_precision;
    let posterior_precision = prepared.posterior_precision;
    let information = prepared.information;
    let derived_quantities = prepared.derived_quantities;
    let latent_input_slices = prepared.latent_input_slices;
    let (prior_variances, prior_derived_variances, prior_factorization) = if flat_state_prior {
        log_diagnostics(
            config,
            format_args!("prior_flat_state_skipped dimension={joint_dimension}"),
        );
        (
            improper_prior_variances(joint_dimension),
            improper_derived_variances(&derived_quantities),
            flat_prior_factorization(
                joint_dimension,
                prior_gmrf_precision.nnz(),
                config.precision_policy,
            ),
        )
    } else {
        let prior_factorized =
            factorize_precision_with_diagnostics("prior", &prior_gmrf_precision, config)?;
        let prior_derived_quantities = scale_derived_quantity_operators(
            &derived_quantities,
            prior_factorized.scale.as_deref(),
        )?;
        let mut prior =
            gmrf_from_zero_mean_precision(prior_factorized.precision, prior_factorized.factor)
                .map_err(|err| format!("failed to build prior GMRF: {err}"))?;
        log_diagnostics(
            config,
            format_args!(
                "prior_variances_start mode={} dimension={}",
                variance_mode_name(config.variance.mode),
                joint_dimension
            ),
        );
        let prior_working_variances = estimate_variances(&mut prior, config, None)?;
        let prior_variances =
            scale_gmrf_variances(&prior_working_variances, prior_factorized.scale.as_deref())?;
        let prior_derived_variances =
            estimate_derived_variances(&mut prior, &prior_derived_quantities, config, None)?;
        log_diagnostics(
            config,
            format_args!(
                "prior_variances_done mode={} dimension={}",
                variance_mode_name(config.variance.mode),
                joint_dimension
            ),
        );
        drop(prior);
        log_diagnostics(
            config,
            format_args!("prior_factor_released dimension={joint_dimension}"),
        );
        (
            prior_variances,
            prior_derived_variances,
            prior_factorized.debug,
        )
    };

    let posterior_factorized =
        factorize_precision_with_diagnostics("posterior", &posterior_precision, config)?;
    let posterior_information =
        scale_gmrf_vector(&information, posterior_factorized.scale.as_deref())?;
    let posterior_derived_quantities = scale_derived_quantity_operators(
        &derived_quantities,
        posterior_factorized.scale.as_deref(),
    )?;
    let mut posterior = Gmrf::from_information_and_precision_with_sqrt(
        posterior_information,
        posterior_factorized.precision,
        posterior_factorized.factor,
    )
    .map_err(|err| err.to_string())?;
    log_diagnostics(
        config,
        format_args!(
            "posterior_variances_start mode={} dimension={}",
            variance_mode_name(config.variance.mode),
            joint_dimension
        ),
    );
    let posterior_working_variances =
        estimate_variances(&mut posterior, config, Some(&prior_variances))?;
    let posterior_variances = scale_gmrf_variances(
        &posterior_working_variances,
        posterior_factorized.scale.as_deref(),
    )?;
    let posterior_derived_variances = estimate_derived_variances(
        &mut posterior,
        &posterior_derived_quantities,
        config,
        Some(&prior_derived_variances),
    )?;
    log_diagnostics(
        config,
        format_args!(
            "posterior_variances_done mode={} dimension={}",
            variance_mode_name(config.variance.mode),
            joint_dimension
        ),
    );

    let centered_joint_mean =
        scale_gmrf_vector(posterior.mean(), posterior_factorized.scale.as_deref())?;
    let reduced_centered_mean = gmrf_vec_to_feec(&GmrfVector::from_vec(
        centered_joint_mean.as_slice()[0..state_dim].to_vec(),
    ));
    let reduced_posterior_mean = &reduced_centered_mean + &state_mean;
    let reduced_posterior_variance = gmrf_vec_to_feec(&GmrfVector::from_vec(
        posterior_variances.as_slice()[0..state_dim].to_vec(),
    ));
    let reduced_prior_variance = gmrf_vec_to_feec(&GmrfVector::from_vec(
        prior_variances.as_slice()[0..state_dim].to_vec(),
    ));

    let posterior_mean = lift_vector_with_layout(&problem.system.layout, &reduced_posterior_mean)?;
    let posterior_variance =
        lift_variances_with_layout(&problem.system.layout, &reduced_posterior_variance)?;
    let prior_variance =
        lift_variances_with_layout(&problem.system.layout, &reduced_prior_variance)?;
    let derived_variances =
        feec_derived_variances(&prior_derived_variances, &posterior_derived_variances);
    let latent_inputs = latent_input_posteriors(
        &latent_input_slices,
        centered_joint_mean.as_slice(),
        posterior_variances.as_slice(),
    );
    let pde_residual_mean =
        gmrf_vec_to_feec(&pde_operator.mul_vec(&centered_joint_mean)) + centered_residual_bias;

    Ok(LinearPdeUqResult {
        posterior_mean,
        posterior_variance,
        prior_variance,
        derived_variances,
        latent_inputs,
        reduced_posterior_mean,
        reduced_posterior_variance,
        pde_residual_mean,
        debug: LinearPdeUqDebug {
            input_representations: prepared.input_representations,
            joint_dimension,
            flat_state_prior,
            prior_factorization,
            posterior_factorization: posterior_factorized.debug,
        },
    })
}

fn prepare_linear_pde_problem(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<PreparedLinearPdeProblem, String> {
    validate_problem(problem)?;
    validate_solver_config(config)?;

    let state_dimension = problem.system.state_dimension();
    let residual_dim = problem.system.residual_dimension();
    let flat_state_prior = problem.state_prior.precision.nnz() == 0;
    let mut centered_residual_bias = problem.system.residual_bias.clone();
    let state_operator = problem.system.operator.clone();
    let state_mean = FeecVector::from_vec(problem.state_prior.mean.clone());
    centered_residual_bias += &state_operator * &state_mean;

    let resolved = resolve_input_representations(&problem.uncertain_inputs, residual_dim)?;
    let collapsed_count = resolved
        .iter()
        .filter(|(_, representation, _)| *representation == ResolvedInputRepresentation::Collapsed)
        .count();
    if collapsed_count > 1 {
        return Err(
            "at most one uncertain input may be collapsed in v1; keep the remaining inputs latent"
                .to_string(),
        );
    }
    if problem.pde_variance.is_some() && problem.pde_precision.is_some() {
        return Err(
            "pde_variance and pde_precision are mutually exclusive; use one PDE residual noise model"
                .to_string(),
        );
    }
    if collapsed_count > 0 && (problem.pde_variance.is_some() || problem.pde_precision.is_some()) {
        return Err(
            "PDE residual noise cannot be combined with collapsed uncertain inputs in v1; keep the input latent or remove the PDE residual noise model".to_string(),
        );
    }

    let mut block_precisions = vec![problem.state_prior.precision.clone()];
    let mut latent_blocks = Vec::<(usize, FeecCsr)>::new();
    let mut input_representations = Vec::with_capacity(problem.uncertain_inputs.len());
    let mut latent_input_slices = Vec::new();
    let mut joint_dimension = state_dimension;
    let mut collapsed_precision = None;

    for (input, representation, maybe_precision) in resolved {
        let operator = core_triplet_to_feec_csr(&input.operator);
        let mean = FeecVector::from_vec(input.prior.mean.clone());
        centered_residual_bias += &operator * &mean;
        input_representations.push(InputRepresentationDebug {
            name: input.name.clone(),
            representation,
        });

        match representation {
            ResolvedInputRepresentation::Latent => {
                let offset = joint_dimension;
                joint_dimension += input.prior.dimension();
                latent_input_slices.push(LatentInputSlice {
                    name: input.name.clone(),
                    offset,
                    prior_mean: input.prior.mean.clone(),
                });
                latent_blocks.push((offset, operator));
                block_precisions.push(input.prior.precision.clone());
            }
            ResolvedInputRepresentation::Collapsed => {
                let precision = maybe_precision.ok_or_else(|| {
                    format!(
                        "uncertain input `{}` was resolved as collapsed without a residual precision",
                        input.name
                    )
                })?;
                collapsed_precision = Some(core_triplet_to_feec_csr(&precision));
            }
        }
    }

    let mut derived_quantities = restrict_derived_quantities(
        &problem.derived_quantities,
        &problem.system.layout,
        joint_dimension,
    )?;
    restrict_joint_derived_quantities(
        &problem.joint_derived_quantities,
        &problem.system.layout,
        joint_dimension,
        &latent_input_slices,
        &mut derived_quantities,
    )?;

    let prior_precision = block_diag_precision(&block_precisions);
    let pde_operator = build_joint_operator(
        joint_dimension,
        &[(0, state_operator.clone())]
            .into_iter()
            .chain(
                latent_blocks
                    .iter()
                    .map(|(offset, operator)| (*offset, operator.clone())),
            )
            .collect::<Vec<_>>(),
    );
    let pde_bias = feec_vec_to_gmrf(&centered_residual_bias);
    let zero_observations = GmrfVector::zeros(residual_dim);

    let mut builder = LinearObservationStackBuilder::new(joint_dimension);
    if let Some(variance) = problem.pde_variance {
        builder
            .push_block(
                0,
                &pde_operator,
                zero_observations.as_slice(),
                pde_bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())?;
    }

    for measurement in &problem.physical_measurements {
        let centered =
            restrict_measurement_to_reduced(measurement, &problem.system.layout, &state_mean)?;
        builder
            .push_block(
                0,
                &feec_csr_to_gmrf(&centered.operator),
                centered.observations.as_slice(),
                centered.bias.as_slice(),
                centered.variance,
            )
            .map_err(|err| err.to_string())?;
    }
    for measurement in &problem.joint_measurements {
        push_joint_measurement(
            &mut builder,
            measurement,
            &problem.system.layout,
            &state_mean,
            &latent_input_slices,
        )?;
    }

    let stacked = builder.finish();
    log_diagnostics(
        config,
        format_args!(
            "posterior_assembly_ready dimension={} observation_rows={} observation_nnz={}",
            joint_dimension,
            stacked.matrix.nrows(),
            stacked.matrix.nnz()
        ),
    );
    let pde_precision_gmrf = problem
        .pde_precision
        .as_ref()
        .map(|precision| feec_csr_to_gmrf(&core_triplet_to_feec_csr(precision)));
    let collapsed_precision_gmrf = collapsed_precision.as_ref().map(feec_csr_to_gmrf);
    let mut observation_terms = Vec::new();
    if stacked.matrix.nrows() > 0 {
        observation_terms.push(LinearObservationTerm::scalar_variance(
            &stacked.matrix,
            &stacked.observations,
            Some(&stacked.bias),
            stacked.noise_variance,
        ));
    }
    if let Some(precision) = &pde_precision_gmrf {
        observation_terms.push(LinearObservationTerm::precision(
            &pde_operator,
            &zero_observations,
            Some(&pde_bias),
            precision,
        ));
    }
    if let Some(precision) = &collapsed_precision_gmrf {
        observation_terms.push(LinearObservationTerm::precision(
            &pde_operator,
            &zero_observations,
            Some(&pde_bias),
            precision,
        ));
    }
    let (posterior_precision, information) =
        apply_linear_observation_terms(&prior_precision, &observation_terms);

    Ok(PreparedLinearPdeProblem {
        state_dimension,
        joint_dimension,
        state_mean,
        flat_state_prior,
        centered_residual_bias,
        pde_operator,
        prior_precision,
        posterior_precision,
        information,
        derived_quantities,
        input_representations,
        latent_input_slices,
    })
}

fn latent_input_posteriors(
    slices: &[LatentInputSlice],
    joint_mean: &[f64],
    joint_variance: &[f64],
) -> Vec<LinearPdeLatentInputPosterior> {
    slices
        .iter()
        .map(|slice| {
            let dimension = slice.prior_mean.len();
            let mean = (0..dimension)
                .map(|index| joint_mean[slice.offset + index] + slice.prior_mean[index])
                .collect();
            let variance = joint_variance[slice.offset..slice.offset + dimension].to_vec();
            LinearPdeLatentInputPosterior {
                name: slice.name.clone(),
                offset: slice.offset,
                mean,
                variance,
            }
        })
        .collect()
}

fn select_scalar_pushforward_operator<S: AsRef<str>>(
    derived_quantities: &BTreeMap<String, SparseRowOperator>,
    qoi_names: &[S],
) -> Result<(Vec<String>, SparseRowOperator), String> {
    let mut names = Vec::with_capacity(qoi_names.len());
    let mut rows = Vec::with_capacity(qoi_names.len());
    let mut ncols = None;
    for name in qoi_names {
        let name = name.as_ref();
        let operator = derived_quantities
            .get(name)
            .ok_or_else(|| format!("unknown pushforward QoI `{name}`"))?;
        if operator.nrows() != 1 {
            return Err(format!(
                "pushforward QoI `{name}` must be scalar, found {} rows",
                operator.nrows()
            ));
        }
        if let Some(expected) = ncols {
            if operator.ncols != expected {
                return Err(format!(
                    "pushforward QoI `{name}` has {} columns, expected {expected}",
                    operator.ncols
                ));
            }
        } else {
            ncols = Some(operator.ncols);
        }
        names.push(name.to_string());
        rows.push(operator.rows[0].clone());
    }
    let ncols = ncols.ok_or_else(|| "no pushforward QoIs were selected".to_string())?;
    Ok((
        names,
        SparseRowOperator::new(ncols, rows).map_err(|err| err.to_string())?,
    ))
}

struct CenteredMeasurement {
    operator: FeecCsr,
    observations: FeecVector,
    bias: FeecVector,
    variance: f64,
}

fn validate_problem(problem: &LinearPdeUqProblem) -> Result<(), String> {
    problem.state_prior.validate()?;
    if problem.state_prior.dimension() != problem.system.state_dimension() {
        return Err(format!(
            "state prior dimension {} must match reduced state dimension {}",
            problem.state_prior.dimension(),
            problem.system.state_dimension()
        ));
    }
    if problem.state_prior.mean.len() != problem.system.state_dimension() {
        return Err("state prior mean must be defined on the reduced state".to_string());
    }
    if problem.system.residual_bias.len() != problem.system.residual_dimension() {
        return Err(format!(
            "residual bias length {} must match residual dimension {}",
            problem.system.residual_bias.len(),
            problem.system.residual_dimension()
        ));
    }
    if let Some(variance) = problem.pde_variance {
        if !variance.is_finite() || variance <= 0.0 {
            return Err("pde_variance must be finite and positive when provided".to_string());
        }
    }
    if let Some(precision) = &problem.pde_precision {
        if precision.nrows() != problem.system.residual_dimension()
            || precision.ncols() != problem.system.residual_dimension()
        {
            return Err(format!(
                "pde_precision must be {}x{}, got {}x{}",
                problem.system.residual_dimension(),
                problem.system.residual_dimension(),
                precision.nrows(),
                precision.ncols()
            ));
        }
    }
    for input in &problem.uncertain_inputs {
        input.validate(problem.system.residual_dimension())?;
    }
    let mut input_dimensions = BTreeMap::<String, usize>::new();
    for input in &problem.uncertain_inputs {
        if input_dimensions
            .insert(input.name.clone(), input.prior.dimension())
            .is_some()
        {
            return Err(format!(
                "uncertain input names must be unique; duplicate `{}`",
                input.name
            ));
        }
    }
    for measurement in &problem.physical_measurements {
        measurement.validate(problem.system.layout.full_dimension)?;
    }
    for measurement in &problem.joint_measurements {
        validate_joint_measurement(
            measurement,
            problem.system.layout.full_dimension,
            &input_dimensions,
        )?;
    }
    for derived in &problem.derived_quantities {
        if derived.operator.ncols != problem.system.layout.full_dimension {
            return Err(format!(
                "derived operator `{}` column count {} must match full state dimension {}",
                derived.name, derived.operator.ncols, problem.system.layout.full_dimension
            ));
        }
    }
    for derived in &problem.joint_derived_quantities {
        validate_joint_derived_quantity(
            derived,
            problem.system.layout.full_dimension,
            &input_dimensions,
        )?;
    }
    Ok(())
}

fn validate_joint_measurement(
    measurement: &LinearPdeJointMeasurementSpec,
    full_state_dimension: usize,
    input_dimensions: &BTreeMap<String, usize>,
) -> Result<(), String> {
    let row_count = measurement.observations.len();
    if row_count != measurement.bias.len() {
        return Err(format!(
            "joint measurement `{}` observation length {} must match bias length {}",
            measurement.name,
            row_count,
            measurement.bias.len()
        ));
    }
    if !measurement.variance.is_finite() || measurement.variance <= 0.0 {
        return Err(format!(
            "joint measurement `{}` variance must be finite and positive",
            measurement.name
        ));
    }
    if measurement.state_operator.is_none() && measurement.latent_operators.is_empty() {
        return Err(format!(
            "joint measurement `{}` must contain a state operator or at least one latent operator",
            measurement.name
        ));
    }
    if let Some(operator) = &measurement.state_operator {
        if operator.ncols() != full_state_dimension {
            return Err(format!(
                "joint measurement `{}` state operator column count {} must match full state dimension {}",
                measurement.name,
                operator.ncols(),
                full_state_dimension
            ));
        }
        if operator.nrows() != row_count {
            return Err(format!(
                "joint measurement `{}` state operator row count {} must match observation length {}",
                measurement.name,
                operator.nrows(),
                row_count
            ));
        }
    }
    for block in &measurement.latent_operators {
        let Some(input_dimension) = input_dimensions.get(&block.input_name) else {
            return Err(format!(
                "joint measurement `{}` references unknown latent input `{}`",
                measurement.name, block.input_name
            ));
        };
        if block.operator.ncols() != *input_dimension {
            return Err(format!(
                "joint measurement `{}` latent block `{}` column count {} must match input dimension {}",
                measurement.name,
                block.input_name,
                block.operator.ncols(),
                input_dimension
            ));
        }
        if block.operator.nrows() != row_count {
            return Err(format!(
                "joint measurement `{}` latent block `{}` row count {} must match observation length {}",
                measurement.name,
                block.input_name,
                block.operator.nrows(),
                row_count
            ));
        }
    }
    Ok(())
}

fn validate_joint_derived_quantity(
    derived: &LinearPdeJointDerivedQuantitySpec,
    full_state_dimension: usize,
    input_dimensions: &BTreeMap<String, usize>,
) -> Result<(), String> {
    if derived.state_operator.is_none() && derived.latent_operators.is_empty() {
        return Err(format!(
            "joint derived quantity `{}` must contain a state operator or at least one latent operator",
            derived.name
        ));
    }
    let mut row_count = None;
    if let Some(operator) = &derived.state_operator {
        if operator.ncols != full_state_dimension {
            return Err(format!(
                "joint derived quantity `{}` state operator column count {} must match full state dimension {}",
                derived.name, operator.ncols, full_state_dimension
            ));
        }
        row_count = Some(operator.nrows());
    }
    for block in &derived.latent_operators {
        let Some(input_dimension) = input_dimensions.get(&block.input_name) else {
            return Err(format!(
                "joint derived quantity `{}` references unknown latent input `{}`",
                derived.name, block.input_name
            ));
        };
        if block.operator.ncols != *input_dimension {
            return Err(format!(
                "joint derived quantity `{}` latent block `{}` column count {} must match input dimension {}",
                derived.name,
                block.input_name,
                block.operator.ncols,
                input_dimension
            ));
        }
        if let Some(expected_rows) = row_count {
            if block.operator.nrows() != expected_rows {
                return Err(format!(
                    "joint derived quantity `{}` latent block `{}` row count {} must match row count {}",
                    derived.name,
                    block.input_name,
                    block.operator.nrows(),
                    expected_rows
                ));
            }
        } else {
            row_count = Some(block.operator.nrows());
        }
    }
    Ok(())
}

fn validate_solver_config(config: &LinearPdeUqSolverConfig) -> Result<(), String> {
    let max_relative_asymmetry = config.precision_policy.max_relative_asymmetry();
    if !max_relative_asymmetry.is_finite() || max_relative_asymmetry < 0.0 {
        return Err(
            "precision_policy.max_relative_asymmetry must be finite and nonnegative".to_string(),
        );
    }
    match config.variance.mode {
        LinearPdeVarianceMode::Exact
        | LinearPdeVarianceMode::ExactSolves
        | LinearPdeVarianceMode::SelectedInverse => Ok(()),
        LinearPdeVarianceMode::MonteCarlo
        | LinearPdeVarianceMode::Hutchinson
        | LinearPdeVarianceMode::LocalRbmc => {
            if config.variance.num_variance_probes == 0 {
                return Err("variance.num_variance_probes must be >= 1".to_string());
            }
            if config.variance.variance_batch_count == 0 {
                return Err("variance.variance_batch_count must be >= 1".to_string());
            }
            if matches!(config.variance.mode, LinearPdeVarianceMode::LocalRbmc)
                && config.variance.local_rb_block_size < 2
            {
                return Err("variance.local_rb_block_size must be >= 2".to_string());
            }
            Ok(())
        }
    }
}

fn resolve_input_representations(
    inputs: &[LinearUncertainInputSpec],
    residual_dimension: usize,
) -> Result<
    Vec<(
        LinearUncertainInputSpec,
        ResolvedInputRepresentation,
        Option<SparseTripletMatrix>,
    )>,
    String,
> {
    let mut forced_collapsed = Vec::new();
    let mut auto_candidates = Vec::new();
    let mut collapsed_available = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        let available = available_collapsed_precision(input, residual_dimension);
        if matches!(input.preference, RepresentationPreference::ForceCollapsed)
            && available.is_none()
        {
            return Err(format!(
                "uncertain input `{}` cannot be forced collapsed because no sparse residual precision is available",
                input.name
            ));
        }
        if matches!(input.preference, RepresentationPreference::ForceCollapsed) {
            forced_collapsed.push(index);
        } else if matches!(input.preference, RepresentationPreference::Auto) && available.is_some()
        {
            auto_candidates.push(index);
        }
        collapsed_available.push(available);
    }

    if forced_collapsed.len() > 1 {
        return Err("only one uncertain input may be forced collapsed in v1".to_string());
    }

    let selected_auto = if forced_collapsed.is_empty() && auto_candidates.len() == 1 {
        Some(auto_candidates[0])
    } else {
        None
    };

    Ok(inputs
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, input)| {
            let representation =
                if forced_collapsed.contains(&index) || selected_auto == Some(index) {
                    ResolvedInputRepresentation::Collapsed
                } else {
                    ResolvedInputRepresentation::Latent
                };
            (input, representation, collapsed_available[index].clone())
        })
        .collect())
}

fn available_collapsed_precision(
    input: &LinearUncertainInputSpec,
    residual_dimension: usize,
) -> Option<SparseTripletMatrix> {
    if let Some(precision) = &input.collapsed_precision {
        return Some(precision.clone());
    }
    if input.operator.nrows() == residual_dimension
        && input.operator.ncols() == residual_dimension
        && input.prior.dimension() == residual_dimension
        && is_signed_identity(&input.operator)
    {
        return Some(input.prior.precision.clone());
    }
    None
}

fn is_signed_identity(matrix: &feg_core::SparseTripletMatrix) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    let mut seen = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row != col {
            return false;
        }
        if seen[row] != 0.0 {
            return false;
        }
        if value.abs() != 1.0 {
            return false;
        }
        seen[row] = value.abs();
    }
    seen.iter().all(|value| (*value - 1.0).abs() <= 1e-12)
}

fn build_joint_operator(dimension: usize, blocks: &[(usize, FeecCsr)]) -> GmrfSparseMatrix {
    let row_count = blocks.first().map(|(_, block)| block.nrows()).unwrap_or(0);
    let mut coo = GmrfCoo::new(row_count, dimension);
    for (offset, block) in blocks {
        for (row, col, value) in block.triplet_iter() {
            coo.push(row, offset + col, *value);
        }
    }
    GmrfSparseMatrix::from(&coo)
}

fn block_diag_precision(blocks: &[feg_core::SparseTripletMatrix]) -> GmrfSparseMatrix {
    let dimension = blocks.iter().map(|block| block.nrows()).sum();
    let mut coo = GmrfCoo::new(dimension, dimension);
    let mut offset = 0;
    for block in blocks {
        for (row, col, value) in block.triplet_iter() {
            coo.push(offset + row, offset + col, value);
        }
        offset += block.nrows();
    }
    GmrfSparseMatrix::from(&coo)
}

#[derive(Debug, Clone, Copy)]
struct PrecisionDiagnostics {
    min_diagonal: f64,
    max_abs_diagonal: f64,
    diagonal_ratio: f64,
    max_abs_asymmetry: f64,
    max_relative_asymmetry: f64,
}

struct CanonicalPrecision {
    precision: GmrfSparseMatrix,
    diagnostics: PrecisionDiagnostics,
    scale: Option<Vec<f64>>,
    equilibration_min_scale: f64,
    equilibration_max_scale: f64,
}

fn canonicalize_precision(
    precision: &GmrfSparseMatrix,
    policy: LinearPdePrecisionPolicy,
) -> Result<CanonicalPrecision, String> {
    let diagnostics = precision_diagnostics(precision)?;
    if diagnostics.max_relative_asymmetry > policy.max_relative_asymmetry() {
        return Err(format!(
            "precision matrix asymmetry exceeds policy tolerance: {}",
            format_precision_diagnostics(&diagnostics, policy, 1.0, 1.0)
        ));
    }

    let symmetrized = symmetrize_precision(precision);
    if !policy.uses_diagonal_equilibration() {
        return Ok(CanonicalPrecision {
            precision: symmetrized,
            diagnostics,
            scale: None,
            equilibration_min_scale: 1.0,
            equilibration_max_scale: 1.0,
        });
    }

    let diagonal = diagonal_values(&symmetrized)?;
    let mut scale = Vec::with_capacity(diagonal.len());
    let mut min_scale = f64::INFINITY;
    let mut max_scale = 0.0_f64;
    for (index, value) in diagonal.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "diagonal equilibration requires finite positive diagonals; diagonal[{index}]={value:.6e}; {}",
                format_precision_diagnostics(&diagnostics, policy, 1.0, 1.0)
            ));
        }
        let entry_scale = 1.0 / value.sqrt();
        if !entry_scale.is_finite() {
            return Err(format!(
                "diagonal equilibration produced non-finite scale at index {index}; {}",
                format_precision_diagnostics(&diagnostics, policy, 1.0, 1.0)
            ));
        }
        min_scale = min_scale.min(entry_scale);
        max_scale = max_scale.max(entry_scale);
        scale.push(entry_scale);
    }

    let equilibrated = scale_precision_matrix(&symmetrized, &scale)?;
    Ok(CanonicalPrecision {
        precision: equilibrated,
        diagnostics,
        scale: Some(scale),
        equilibration_min_scale: min_scale,
        equilibration_max_scale: max_scale,
    })
}

fn precision_diagnostics(matrix: &GmrfSparseMatrix) -> Result<PrecisionDiagnostics, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "precision matrix must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut entries = BTreeMap::<(usize, usize), f64>::new();
    for (row, col, value) in matrix.triplet_iter() {
        if !value.is_finite() {
            return Err(format!(
                "precision matrix contains non-finite entry at ({row}, {col})"
            ));
        }
        *entries.entry((row, col)).or_insert(0.0) += *value;
    }

    let mut diagonal = vec![0.0; matrix.nrows()];
    let mut max_abs_entry = 0.0_f64;
    for (&(row, col), &value) in &entries {
        max_abs_entry = max_abs_entry.max(value.abs());
        if row == col {
            diagonal[row] += value;
        }
    }

    let min_diagonal = if diagonal.is_empty() {
        0.0
    } else {
        diagonal.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let max_abs_diagonal = diagonal.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let diagonal_ratio = if min_diagonal > 0.0 {
        max_abs_diagonal / min_diagonal
    } else {
        f64::INFINITY
    };

    let mut checked_pairs = BTreeSet::<(usize, usize)>::new();
    let mut max_abs_asymmetry = 0.0_f64;
    for &(row, col) in entries.keys() {
        let key = if row <= col { (row, col) } else { (col, row) };
        if !checked_pairs.insert(key) {
            continue;
        }
        let upper = entries.get(&key).copied().unwrap_or(0.0);
        let lower = entries.get(&(key.1, key.0)).copied().unwrap_or(0.0);
        max_abs_asymmetry = max_abs_asymmetry.max((upper - lower).abs());
    }
    let max_relative_asymmetry = if max_abs_entry > 0.0 {
        max_abs_asymmetry / max_abs_entry
    } else {
        0.0
    };

    Ok(PrecisionDiagnostics {
        min_diagonal,
        max_abs_diagonal,
        diagonal_ratio,
        max_abs_asymmetry,
        max_relative_asymmetry,
    })
}

fn diagonal_values(matrix: &GmrfSparseMatrix) -> Result<Vec<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "precision matrix must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    Ok(diagonal)
}

fn symmetrize_precision(matrix: &GmrfSparseMatrix) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            coo.push(row, col, *value);
        } else {
            coo.push(row, col, 0.5 * *value);
            coo.push(col, row, 0.5 * *value);
        }
    }
    GmrfSparseMatrix::from(&coo)
}

fn scale_precision_matrix(
    matrix: &GmrfSparseMatrix,
    scale: &[f64],
) -> Result<GmrfSparseMatrix, String> {
    if matrix.nrows() != scale.len() || matrix.ncols() != scale.len() {
        return Err(format!(
            "precision scale length {} must match matrix dimension {}x{}",
            scale.len(),
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value * scale[row] * scale[col]);
    }
    Ok(GmrfSparseMatrix::from(&coo))
}

fn format_precision_diagnostics(
    diagnostics: &PrecisionDiagnostics,
    policy: LinearPdePrecisionPolicy,
    equilibration_min_scale: f64,
    equilibration_max_scale: f64,
) -> String {
    format!(
        "policy={} min_diag={:.6e} max_abs_diag={:.6e} diag_ratio={:.6e} max_abs_asym={:.6e} max_rel_asym={:.6e} asym_tolerance={:.6e} equilibration_scale_min={:.6e} equilibration_scale_max={:.6e}",
        policy.label(),
        diagnostics.min_diagonal,
        diagnostics.max_abs_diagonal,
        diagnostics.diagonal_ratio,
        diagnostics.max_abs_asymmetry,
        diagnostics.max_relative_asymmetry,
        policy.max_relative_asymmetry(),
        equilibration_min_scale,
        equilibration_max_scale
    )
}

fn gmrf_from_zero_mean_precision(
    precision: GmrfSparseMatrix,
    factor: SparseCholeskyFactor,
) -> Result<Gmrf, String> {
    Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision)
        .map_err(|err| err.to_string())
        .map(|gmrf| gmrf.with_precision_sqrt(factor))
}

fn scale_gmrf_vector(vector: &GmrfVector, scale: Option<&[f64]>) -> Result<GmrfVector, String> {
    let Some(scale) = scale else {
        return Ok(vector.clone());
    };
    if vector.len() != scale.len() {
        return Err(format!(
            "scale length {} must match vector length {}",
            scale.len(),
            vector.len()
        ));
    }
    Ok(GmrfVector::from_vec(
        vector
            .as_slice()
            .iter()
            .zip(scale.iter())
            .map(|(value, scale)| value * scale)
            .collect(),
    ))
}

fn scale_gmrf_variances(
    variances: &GmrfVector,
    scale: Option<&[f64]>,
) -> Result<GmrfVector, String> {
    let Some(scale) = scale else {
        return Ok(variances.clone());
    };
    if variances.len() != scale.len() {
        return Err(format!(
            "scale length {} must match variance length {}",
            scale.len(),
            variances.len()
        ));
    }
    Ok(GmrfVector::from_vec(
        variances
            .as_slice()
            .iter()
            .zip(scale.iter())
            .map(|(value, scale)| value * scale * scale)
            .collect(),
    ))
}

fn scale_sparse_row_operator_columns(
    operator: &SparseRowOperator,
    scale: Option<&[f64]>,
) -> Result<SparseRowOperator, String> {
    let Some(scale) = scale else {
        return Ok(operator.clone());
    };
    if operator.ncols != scale.len() {
        return Err(format!(
            "operator column count {} must match scale length {}",
            operator.ncols,
            scale.len()
        ));
    }
    SparseRowOperator::new(
        operator.ncols,
        operator
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(column, value)| (*column, value * scale[*column]))
                    .collect()
            })
            .collect(),
    )
    .map_err(|err| err.to_string())
}

fn scale_derived_quantity_operators(
    derived_quantities: &BTreeMap<String, SparseRowOperator>,
    scale: Option<&[f64]>,
) -> Result<BTreeMap<String, SparseRowOperator>, String> {
    derived_quantities
        .iter()
        .map(|(name, operator)| {
            scale_sparse_row_operator_columns(operator, scale).map(|scaled| (name.clone(), scaled))
        })
        .collect()
}

fn factorize_precision_with_diagnostics(
    label: &str,
    precision: &GmrfSparseMatrix,
    config: &LinearPdeUqSolverConfig,
) -> Result<FactorizedPrecision, String> {
    let canonical = canonicalize_precision(precision, config.precision_policy)
        .map_err(|err| format!("{label} precision canonicalization failed: {err}"))?;
    let matrix_nnz = canonical.precision.nnz();
    let matrix_lower_triangle_nnz = lower_triangle_nnz(&canonical.precision);
    log_diagnostics(
        config,
        format_args!(
            "{label}_factorization_start dimension={} matrix_nnz={} lower_triangle_nnz={} {}",
            canonical.precision.nrows(),
            matrix_nnz,
            matrix_lower_triangle_nnz,
            format_precision_diagnostics(
                &canonical.diagnostics,
                config.precision_policy,
                canonical.equilibration_min_scale,
                canonical.equilibration_max_scale
            )
        ),
    );
    let factor = canonical.precision.cholesky_sqrt_lower().map_err(|err| {
        format!(
            "{label} precision Cholesky factorization failed under policy {}: {}; {}",
            config.precision_policy.label(),
            err,
            format_precision_diagnostics(
                &canonical.diagnostics,
                config.precision_policy,
                canonical.equilibration_min_scale,
                canonical.equilibration_max_scale
            )
        )
    })?;
    let factorization = LinearPdeFactorizationDebug {
        dimension: canonical.precision.nrows(),
        matrix_nnz,
        matrix_lower_triangle_nnz,
        factor_nnz: factor.nnz(),
        fill_in_ratio_vs_lower_triangle: factor.nnz() as f64
            / matrix_lower_triangle_nnz.max(1) as f64,
        factor_numeric_values_mib: factor.nnz() as f64 * size_of::<f64>() as f64
            / (1024.0 * 1024.0),
        min_diagonal: canonical.diagnostics.min_diagonal,
        max_abs_diagonal: canonical.diagnostics.max_abs_diagonal,
        diagonal_ratio: canonical.diagnostics.diagonal_ratio,
        max_abs_asymmetry: canonical.diagnostics.max_abs_asymmetry,
        max_relative_asymmetry: canonical.diagnostics.max_relative_asymmetry,
        precision_policy: config.precision_policy,
        equilibration_min_scale: canonical.equilibration_min_scale,
        equilibration_max_scale: canonical.equilibration_max_scale,
    };
    log_diagnostics(
        config,
        format_args!(
            "{label}_factorization_done dimension={} matrix_nnz={} lower_triangle_nnz={} factor_nnz={} fill_in_vs_lower={:.3}x factor_values_mib={:.3} {}",
            factorization.dimension,
            factorization.matrix_nnz,
            factorization.matrix_lower_triangle_nnz,
            factorization.factor_nnz,
            factorization.fill_in_ratio_vs_lower_triangle,
            factorization.factor_numeric_values_mib,
            format_precision_diagnostics(
                &canonical.diagnostics,
                config.precision_policy,
                canonical.equilibration_min_scale,
                canonical.equilibration_max_scale
            )
        ),
    );
    Ok(FactorizedPrecision {
        precision: canonical.precision,
        factor,
        debug: factorization,
        scale: canonical.scale,
    })
}

fn lower_triangle_nnz(matrix: &GmrfSparseMatrix) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, col, _)| row >= col)
        .count()
}

fn log_diagnostics(config: &LinearPdeUqSolverConfig, args: std::fmt::Arguments<'_>) {
    if config.log_diagnostics {
        eprintln!("[linear_pde] {args}");
    }
}

fn variance_mode_name(mode: LinearPdeVarianceMode) -> &'static str {
    match mode {
        LinearPdeVarianceMode::Exact => "exact",
        LinearPdeVarianceMode::ExactSolves => "exact-solves",
        LinearPdeVarianceMode::MonteCarlo => "monte-carlo",
        LinearPdeVarianceMode::Hutchinson => "hutchinson",
        LinearPdeVarianceMode::LocalRbmc => "local-rbmc",
        LinearPdeVarianceMode::SelectedInverse => "selected-inverse",
    }
}

fn improper_prior_variances(dimension: usize) -> GmrfVector {
    GmrfVector::from_vec(vec![f64::INFINITY; dimension])
}

fn improper_derived_variances(
    derived_quantities: &BTreeMap<String, SparseRowOperator>,
) -> BTreeMap<String, GmrfVector> {
    derived_quantities
        .iter()
        .map(|(name, operator)| {
            (
                name.clone(),
                GmrfVector::from_vec(vec![f64::INFINITY; operator.nrows()]),
            )
        })
        .collect()
}

fn flat_prior_factorization(
    dimension: usize,
    matrix_nnz: usize,
    precision_policy: LinearPdePrecisionPolicy,
) -> LinearPdeFactorizationDebug {
    LinearPdeFactorizationDebug::skipped(dimension, matrix_nnz, precision_policy)
}

fn estimate_variances(
    gmrf: &mut Gmrf,
    config: &LinearPdeUqSolverConfig,
    prior_variances: Option<&GmrfVector>,
) -> Result<GmrfVector, String> {
    let raw = match config.variance.mode {
        LinearPdeVarianceMode::Exact | LinearPdeVarianceMode::ExactSolves => {
            exact_solve_variances(gmrf, config)?
        }
        LinearPdeVarianceMode::MonteCarlo => monte_carlo_variances(gmrf, &config.variance)?,
        LinearPdeVarianceMode::Hutchinson => hutchinson_variances(gmrf, &config.variance)?,
        LinearPdeVarianceMode::LocalRbmc => local_rbmc_variances(gmrf, &config.variance)?,
        LinearPdeVarianceMode::SelectedInverse => selected_inverse_variances(gmrf)?,
    };
    let _ = prior_variances;
    Ok(raw)
}

fn estimate_derived_variances(
    gmrf: &mut Gmrf,
    derived_quantities: &BTreeMap<String, SparseRowOperator>,
    config: &LinearPdeUqSolverConfig,
    prior_variances: Option<&BTreeMap<String, GmrfVector>>,
) -> Result<BTreeMap<String, GmrfVector>, String> {
    let mut derived_variances = BTreeMap::new();
    for (name, operator) in derived_quantities {
        log_diagnostics(
            config,
            format_args!(
                "derived_variances_start name={} mode={} rows={} cols={}",
                name,
                variance_mode_name(config.variance.mode),
                operator.nrows(),
                operator.ncols
            ),
        );
        let raw = match config.variance.mode {
            LinearPdeVarianceMode::Exact | LinearPdeVarianceMode::ExactSolves => {
                exact_solve_transformed_variances(gmrf, operator, config, name)?
            }
            LinearPdeVarianceMode::MonteCarlo => {
                monte_carlo_transformed_variances(gmrf, operator, &config.variance)?
            }
            LinearPdeVarianceMode::Hutchinson => {
                hutchinson_transformed_variances(gmrf, operator, &config.variance)?
            }
            LinearPdeVarianceMode::LocalRbmc => {
                local_rbmc_transformed_variances(gmrf, operator, &config.variance)?
            }
            LinearPdeVarianceMode::SelectedInverse => {
                selected_inverse_transformed_variances(gmrf, operator)?
            }
        };
        let _ = prior_variances;
        log_diagnostics(
            config,
            format_args!(
                "derived_variances_done name={} mode={} rows={}",
                name,
                variance_mode_name(config.variance.mode),
                operator.nrows()
            ),
        );
        derived_variances.insert(name.clone(), raw);
    }
    Ok(derived_variances)
}

fn exact_solve_variances(
    gmrf: &mut Gmrf,
    config: &LinearPdeUqSolverConfig,
) -> Result<GmrfVector, String> {
    let factor = gmrf
        .precision_factor()
        .ok_or_else(|| "exact-solve variance requires a precision factor".to_string())?;
    let started = Instant::now();
    let progress_interval = if config.log_diagnostics {
        exact_solve_progress_interval(factor.dimension())
    } else {
        0
    };
    exact_solve_diag_with_progress(factor, progress_interval, |completed, total| {
        log_diagnostics(
            config,
            format_args!(
                "exact_solve_variances_progress solved={}/{} elapsed_s={:.3}",
                completed,
                total,
                started.elapsed().as_secs_f64()
            ),
        );
    })
    .map(|estimate| estimate.values)
    .map_err(|err| err.to_string())
}

fn exact_solve_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    config: &LinearPdeUqSolverConfig,
    name: &str,
) -> Result<GmrfVector, String> {
    let factor = gmrf.precision_factor().ok_or_else(|| {
        "exact-solve transformed variance requires a precision factor".to_string()
    })?;
    let started = Instant::now();
    let progress_interval = if config.log_diagnostics {
        exact_solve_progress_interval(operator.nrows())
    } else {
        0
    };
    exact_solve_transformed_diag_with_progress(
        factor,
        operator,
        4096,
        progress_interval,
        |completed, total| {
            log_diagnostics(
                config,
                format_args!(
                    "exact_solve_transformed_variances_progress name={} solved={}/{} elapsed_s={:.3}",
                    name,
                    completed,
                    total,
                    started.elapsed().as_secs_f64()
                ),
            );
        },
    )
    .map(|estimate| estimate.values)
    .map_err(|err| err.to_string())
}

fn exact_solve_progress_interval(dimension: usize) -> usize {
    if dimension <= 1024 {
        dimension.max(1)
    } else {
        512
    }
}

fn monte_carlo_variances(
    gmrf: &mut Gmrf,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    estimate_monte_carlo_variances(
        gmrf,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed,
    )
    .map(|estimate| estimate.values)
    .map_err(|err| err.to_string())
}

fn hutchinson_variances(
    gmrf: &mut Gmrf,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    estimate_hutchinson_variances(
        gmrf,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed,
        ProbeDistribution::Rademacher,
    )
    .map(|estimate| estimate.values)
    .map_err(|err| err.to_string())
}

fn local_rbmc_variances(
    gmrf: &mut Gmrf,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    let precision = gmrf
        .precision_matrix()
        .ok_or_else(|| "local RB variance requires an explicit precision matrix".to_string())?;
    let factor = gmrf
        .precision_factor()
        .ok_or_else(|| "local RB variance requires a precision factor".to_string())?;
    estimate_local_rbmc_variances(
        precision,
        factor,
        &LatentBlockMode::ContiguousPermuted {
            block_size: config.local_rb_block_size,
        },
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed,
    )
    .map(|estimate| estimate.estimate.values)
    .map_err(|err| err.to_string())
}

fn selected_inverse_variances(gmrf: &Gmrf) -> Result<GmrfVector, String> {
    let factor = gmrf
        .precision_factor()
        .ok_or_else(|| "selected inverse variance requires a precision factor".to_string())?;
    selected_inverse_diag(factor)
        .map(|estimate| estimate.values)
        .map_err(|err| err.to_string())
}

fn monte_carlo_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    estimate_monte_carlo_transformed_variances(
        gmrf,
        operator,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed,
    )
    .map(|estimate| estimate.values)
    .map_err(|err| err.to_string())
}

fn hutchinson_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    estimate_hutchinson_transformed_variances(
        gmrf,
        operator,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed,
        ProbeDistribution::Rademacher,
    )
    .map(|estimate| estimate.values)
    .map_err(|err| err.to_string())
}

fn local_rbmc_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    let precision = gmrf.precision_matrix().ok_or_else(|| {
        "local RB transformed variance requires an explicit precision matrix".to_string()
    })?;
    let factor = gmrf
        .precision_factor()
        .ok_or_else(|| "local RB transformed variance requires a precision factor".to_string())?;
    let block_mode = transformed_support_patch_blocks(
        operator,
        &factor.permutation(),
        config.local_rb_block_size,
    )?;
    estimate_local_rbmc_transformed_variances(
        precision,
        factor,
        operator,
        &block_mode,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed,
    )
    .map(|estimate| estimate.estimate.values)
    .map_err(|err| err.to_string())
}

fn transformed_support_patch_blocks(
    operator: &SparseRowOperator,
    permutation: &Permutation,
    target_block_size: usize,
) -> Result<LatentBlockMode, String> {
    if operator.nrows() == 0 {
        return Ok(LatentBlockMode::Explicit {
            blocks: Vec::new(),
            row_assignments: Some(Vec::new()),
        });
    }
    if operator.ncols == 0 {
        return Err("cannot build local RB patches for an operator with zero columns".to_string());
    }

    let row_supports = operator
        .rows
        .iter()
        .map(|row| unique_row_support(row))
        .collect::<Vec<_>>();
    let max_support_size = row_supports
        .iter()
        .map(|support| support.len())
        .max()
        .unwrap_or(1);
    let target_block_size = target_block_size.max(max_support_size).max(1);
    let adjacency = support_graph_adjacency(operator.ncols, &row_supports);

    let mut blocks = Vec::new();
    let mut assignments = vec![BlockId(usize::MAX); operator.nrows()];
    let mut assigned = vec![false; operator.nrows()];

    while let Some(seed_row) = assigned.iter().position(|is_assigned| !*is_assigned) {
        let mut patch = BTreeSet::new();
        if row_supports[seed_row].is_empty() {
            patch.insert(0);
        } else {
            patch.extend(row_supports[seed_row].iter().copied());
        }
        grow_support_patch(&mut patch, &adjacency, target_block_size);

        let block_id = BlockId(blocks.len());
        let mut assigned_any = false;
        for (row_index, support) in row_supports.iter().enumerate() {
            if assigned[row_index] {
                continue;
            }
            let contained =
                support.is_empty() || support.iter().all(|column| patch.contains(column));
            if contained {
                assigned[row_index] = true;
                assignments[row_index] = block_id;
                assigned_any = true;
            }
        }

        if !assigned_any {
            return Err(
                "failed to assign a transformed local RB row to a support patch".to_string(),
            );
        }

        let mut block = patch
            .into_iter()
            .map(|original| {
                permutation
                    .orig_to_perm
                    .get(original)
                    .copied()
                    .map(PermutedIndex)
                    .ok_or_else(|| {
                        format!(
                            "operator column {original} is outside the Cholesky permutation domain"
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        block.sort_unstable();
        block.dedup();
        blocks.push(block);
    }

    Ok(LatentBlockMode::Explicit {
        blocks,
        row_assignments: Some(assignments),
    })
}

#[cfg(test)]
fn row_support_blocks(
    operator: &SparseRowOperator,
    permutation: &Permutation,
) -> Result<LatentBlockMode, String> {
    if operator.nrows() == 0 {
        return Ok(LatentBlockMode::Explicit {
            blocks: Vec::new(),
            row_assignments: Some(Vec::new()),
        });
    }
    if operator.ncols == 0 {
        return Err(
            "cannot build local RB row-support blocks for an operator with zero columns"
                .to_string(),
        );
    }

    let mut blocks = Vec::with_capacity(operator.nrows());
    let mut assignments = Vec::with_capacity(operator.nrows());
    for (row_index, row) in operator.rows.iter().enumerate() {
        let mut block = unique_row_support(row)
            .into_iter()
            .map(|original| {
                permutation
                    .orig_to_perm
                    .get(original)
                    .copied()
                    .map(PermutedIndex)
                    .ok_or_else(|| {
                        format!(
                            "operator column {original} is outside the Cholesky permutation domain"
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if block.is_empty() {
            block.push(PermutedIndex(0));
        }
        block.sort_unstable();
        block.dedup();
        blocks.push(block);
        assignments.push(BlockId(row_index));
    }

    Ok(LatentBlockMode::Explicit {
        blocks,
        row_assignments: Some(assignments),
    })
}

fn unique_row_support(row: &[(usize, f64)]) -> Vec<usize> {
    let mut support = row.iter().map(|(column, _)| *column).collect::<Vec<_>>();
    support.sort_unstable();
    support.dedup();
    support
}

fn support_graph_adjacency(ncols: usize, row_supports: &[Vec<usize>]) -> Vec<BTreeSet<usize>> {
    let mut adjacency = vec![BTreeSet::new(); ncols];
    for support in row_supports {
        for &column in support {
            adjacency[column].extend(
                support
                    .iter()
                    .copied()
                    .filter(|neighbor| *neighbor != column),
            );
        }
    }
    adjacency
}

fn grow_support_patch(
    patch: &mut BTreeSet<usize>,
    adjacency: &[BTreeSet<usize>],
    target_block_size: usize,
) {
    let mut frontier = patch.iter().copied().collect::<BTreeSet<_>>();
    while patch.len() < target_block_size && !frontier.is_empty() {
        let mut next_frontier = BTreeSet::new();
        for column in &frontier {
            for neighbor in &adjacency[*column] {
                if patch.insert(*neighbor) {
                    next_frontier.insert(*neighbor);
                }
            }
        }
        frontier = next_frontier;
    }
}

fn selected_inverse_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
) -> Result<GmrfVector, String> {
    let factor = gmrf.precision_factor().ok_or_else(|| {
        "selected inverse transformed variance requires a precision factor".to_string()
    })?;
    let selected =
        selected_inverse_transformed_diag(factor, operator).map_err(|err| err.to_string())?;
    if let Some(estimate) = selected.estimate {
        Ok(estimate.values)
    } else {
        Err(format!(
            "selected inverse transformed variance closure too large: requested_pairs={} closure_pairs={} factor_pairs={} closure_over_factor={:.3} closure_limit={} status={:?}; use exact variance mode explicitly for repeated covariance solves",
            selected.diagnostics.requested_pairs,
            selected.diagnostics.closure_pairs,
            selected.diagnostics.factor_pairs,
            selected.diagnostics.closure_over_factor,
            selected.diagnostics.closure_limit,
            selected.diagnostics.status
        ))
    }
}

fn feec_derived_variances(
    prior: &BTreeMap<String, GmrfVector>,
    posterior: &BTreeMap<String, GmrfVector>,
) -> BTreeMap<String, LinearPdeDerivedMarginalResult> {
    let mut derived = BTreeMap::new();
    for (name, prior_variance) in prior {
        let posterior_variance = posterior
            .get(name)
            .expect("posterior derived variances must align with prior names");
        derived.insert(
            name.clone(),
            LinearPdeDerivedMarginalResult {
                prior_variance: gmrf_vec_to_feec(prior_variance),
                posterior_variance: gmrf_vec_to_feec(posterior_variance),
            },
        );
    }
    derived
}

fn restrict_measurement_to_reduced(
    measurement: &LinearGaussianMeasurementSpec,
    layout: &DofLayout,
    reduced_state_mean: &FeecVector,
) -> Result<CenteredMeasurement, String> {
    let operator = core_triplet_to_feec_csr(&measurement.operator);
    let bias = FeecVector::from_vec(measurement.bias.clone());
    let (reduced_operator, folded_bias) =
        restrict_columns_and_fold_fixed(&operator, &bias, layout)?;
    let centered_bias = folded_bias + &reduced_operator * reduced_state_mean;
    Ok(CenteredMeasurement {
        operator: reduced_operator,
        observations: FeecVector::from_vec(measurement.observations.clone()),
        bias: centered_bias,
        variance: measurement.variance,
    })
}

fn push_joint_measurement(
    builder: &mut LinearObservationStackBuilder,
    measurement: &LinearPdeJointMeasurementSpec,
    layout: &DofLayout,
    reduced_state_mean: &FeecVector,
    latent_input_slices: &[LatentInputSlice],
) -> Result<(), String> {
    let row_count = measurement.observations.len();
    let mut centered_bias = FeecVector::from_vec(measurement.bias.clone());
    let mut blocks = Vec::<(usize, GmrfSparseMatrix)>::new();

    if let Some(operator) = &measurement.state_operator {
        let operator = core_triplet_to_feec_csr(operator);
        let (reduced_operator, folded_bias) =
            restrict_columns_and_fold_fixed(&operator, &centered_bias, layout)?;
        centered_bias = folded_bias + &reduced_operator * reduced_state_mean;
        blocks.push((0, feec_csr_to_gmrf(&reduced_operator)));
    }

    for block in &measurement.latent_operators {
        let slice = latent_slice_by_name(latent_input_slices, &block.input_name).ok_or_else(|| {
            format!(
                "joint measurement `{}` references `{}`, but that input is not represented as a latent block",
                measurement.name, block.input_name
            )
        })?;
        let operator = core_triplet_to_feec_csr(&block.operator);
        if operator.nrows() != row_count {
            return Err(format!(
                "joint measurement `{}` latent block `{}` row count {} must match observation length {}",
                measurement.name,
                block.input_name,
                operator.nrows(),
                row_count
            ));
        }
        centered_bias += &operator * &FeecVector::from_vec(slice.prior_mean.clone());
        blocks.push((slice.offset, feec_csr_to_gmrf(&operator)));
    }

    if blocks.is_empty() {
        return Err(format!(
            "joint measurement `{}` contains no active observation blocks",
            measurement.name
        ));
    }
    let block_refs = blocks
        .iter()
        .map(|(offset, matrix)| (*offset, matrix))
        .collect::<Vec<_>>();
    builder
        .push_blocks(
            &block_refs,
            measurement.observations.as_slice(),
            centered_bias.as_slice(),
            measurement.variance,
        )
        .map_err(|err| err.to_string())
}

fn latent_slice_by_name<'a>(
    latent_input_slices: &'a [LatentInputSlice],
    name: &str,
) -> Option<&'a LatentInputSlice> {
    latent_input_slices.iter().find(|slice| slice.name == name)
}

fn restrict_derived_quantities(
    derived_quantities: &[LinearPdeDerivedQuantitySpec],
    layout: &DofLayout,
    joint_dimension: usize,
) -> Result<BTreeMap<String, SparseRowOperator>, String> {
    let mut restricted = BTreeMap::new();
    for derived in derived_quantities {
        if restricted.contains_key(&derived.name) {
            return Err(format!(
                "derived quantity names must be unique; duplicate `{}`",
                derived.name
            ));
        }
        let reduced = restrict_operator_to_reduced(&derived.operator, layout)?;
        restricted.insert(
            derived.name.clone(),
            state_operator_to_joint(reduced, joint_dimension)?,
        );
    }
    Ok(restricted)
}

fn restrict_joint_derived_quantities(
    derived_quantities: &[LinearPdeJointDerivedQuantitySpec],
    layout: &DofLayout,
    joint_dimension: usize,
    latent_input_slices: &[LatentInputSlice],
    restricted: &mut BTreeMap<String, SparseRowOperator>,
) -> Result<(), String> {
    for derived in derived_quantities {
        if restricted.contains_key(&derived.name) {
            return Err(format!(
                "derived quantity names must be unique; duplicate `{}`",
                derived.name
            ));
        }
        restricted.insert(
            derived.name.clone(),
            restrict_joint_derived_operator(derived, layout, joint_dimension, latent_input_slices)?,
        );
    }
    Ok(())
}

fn restrict_joint_derived_operator(
    derived: &LinearPdeJointDerivedQuantitySpec,
    layout: &DofLayout,
    joint_dimension: usize,
    latent_input_slices: &[LatentInputSlice],
) -> Result<SparseRowOperator, String> {
    let mut rows: Option<Vec<BTreeMap<usize, f64>>> = None;
    if let Some(state_operator) = &derived.state_operator {
        let state_reduced = restrict_operator_to_reduced(state_operator, layout)?;
        let mut state_rows = vec![BTreeMap::new(); state_reduced.nrows()];
        for (row_index, row) in state_reduced.rows.iter().enumerate() {
            for (column, value) in row {
                *state_rows[row_index].entry(*column).or_insert(0.0) += *value;
            }
        }
        rows = Some(state_rows);
    }

    for block in &derived.latent_operators {
        let slice = latent_slice_by_name(latent_input_slices, &block.input_name).ok_or_else(|| {
            format!(
                "joint derived quantity `{}` references `{}`, but that input is not represented as a latent block",
                derived.name, block.input_name
            )
        })?;
        let row_maps = rows.get_or_insert_with(|| vec![BTreeMap::new(); block.operator.nrows()]);
        if row_maps.len() != block.operator.nrows() {
            return Err(format!(
                "joint derived quantity `{}` latent block `{}` row count {} must match row count {}",
                derived.name,
                block.input_name,
                block.operator.nrows(),
                row_maps.len()
            ));
        }
        for (row_index, row) in block.operator.rows.iter().enumerate() {
            for (column, value) in row {
                *row_maps[row_index]
                    .entry(slice.offset + *column)
                    .or_insert(0.0) += *value;
            }
        }
    }

    let rows = rows.ok_or_else(|| {
        format!(
            "joint derived quantity `{}` contains no active operator blocks",
            derived.name
        )
    })?;
    let rows = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .filter(|(_, value)| value.abs() > 0.0)
                .collect()
        })
        .collect();
    SparseRowOperator::new(joint_dimension, rows).map_err(|err| err.to_string())
}

fn restrict_operator_to_reduced(
    operator: &SparseRowOperator,
    layout: &DofLayout,
) -> Result<SparseRowOperator, String> {
    if operator.ncols != layout.full_dimension {
        return Err(format!(
            "derived operator column count {} does not match full state dimension {}",
            operator.ncols, layout.full_dimension
        ));
    }

    let reduced_map = reduced_index_map(layout);
    let mut rows = Vec::with_capacity(operator.nrows());
    for row in &operator.rows {
        let mut entries = BTreeMap::<usize, f64>::new();
        for (col, value) in row {
            if let Some(reduced_col) = reduced_map[*col] {
                *entries.entry(reduced_col).or_insert(0.0) += *value;
            }
        }
        rows.push(
            entries
                .into_iter()
                .filter(|(_, value)| value.abs() > 0.0)
                .collect(),
        );
    }
    SparseRowOperator::new(layout.reduced_dimension(), rows).map_err(|err| err.to_string())
}

fn state_operator_to_joint(
    operator: SparseRowOperator,
    joint_dimension: usize,
) -> Result<SparseRowOperator, String> {
    if operator.ncols > joint_dimension {
        return Err(format!(
            "reduced derived operator column count {} exceeds joint dimension {}",
            operator.ncols, joint_dimension
        ));
    }
    SparseRowOperator::new(joint_dimension, operator.rows).map_err(|err| err.to_string())
}

fn lift_variances_with_layout(
    layout: &DofLayout,
    reduced: &FeecVector,
) -> Result<FeecVector, String> {
    if reduced.len() != layout.reduced_dimension() {
        return Err(format!(
            "reduced variance length {} does not match layout reduced dimension {}",
            reduced.len(),
            layout.reduced_dimension()
        ));
    }
    let mut full = FeecVector::zeros(layout.full_dimension);
    for (reduced_index, full_index) in layout.active_dofs.iter().copied().enumerate() {
        full[full_index] = reduced[reduced_index];
    }
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::adapt_boundary_spec;
    use crate::prior::matern::zero_form::{
        build_matern_precision_0form, MaternConfig, MaternMassInverse,
    };
    use crate::sparse::feec_csr_to_core_triplet;
    use common::linalg::nalgebra::CooMatrix as FeecCoo;
    use feg_core::{
        BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, LinearGaussianMeasurementSpec,
        RepresentationPreference, SparseTripletMatrix,
    };
    use formoniq::assemble;
    use formoniq::problems::reduced_linear::{
        build_reduced_hodge_laplace_1form_system, build_reduced_laplace_beltrami_system,
    };
    use formoniq::reduction::EssentialBoundarySpec;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn max_abs_difference(lhs: &FeecVector, rhs: &FeecVector) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max)
    }

    fn to_core_triplets(matrix: &FeecCsr) -> SparseTripletMatrix {
        SparseTripletMatrix::from_triplets(
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

    fn diagonal_precision(dimension: usize, diagonal_value: f64) -> SparseTripletMatrix {
        SparseTripletMatrix::from_triplets(
            dimension,
            dimension,
            (0..dimension).map(|index| feg_core::SparseTriplet {
                row: index,
                col: index,
                value: diagonal_value,
            }),
        )
    }

    fn native_diagonal_precision(dimension: usize, diagonal_value: f64) -> FeecCsr {
        core_triplet_to_feec_csr(&diagonal_precision(dimension, diagonal_value))
    }

    fn add_sparse(lhs: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
        let mut coo = FeecCoo::new(lhs.nrows(), lhs.ncols());
        for (row, col, value) in lhs.triplet_iter() {
            coo.push(row, col, *value);
        }
        for (row, col, value) in rhs.triplet_iter() {
            coo.push(row, col, *value);
        }
        FeecCsr::from(&coo)
    }

    fn tridiagonal_precision(
        dimension: usize,
        diagonal: f64,
        off_diagonal: f64,
    ) -> SparseTripletMatrix {
        let mut matrix = SparseTripletMatrix::new(dimension, dimension);
        for index in 0..dimension {
            matrix.push(index, index, diagonal);
            if index + 1 < dimension {
                matrix.push(index, index + 1, off_diagonal);
                matrix.push(index + 1, index, off_diagonal);
            }
        }
        matrix
    }

    fn gmrf_sparse_matrix(
        dimension: usize,
        triplets: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> GmrfSparseMatrix {
        let mut coo = GmrfCoo::new(dimension, dimension);
        for (row, col, value) in triplets {
            coo.push(row, col, value);
        }
        GmrfSparseMatrix::from(&coo)
    }

    fn mixed_scale_gaussian_problem() -> LinearPdeUqProblem {
        let dimension = 2;
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(dimension, 1.0),
            residual_bias: vec![0.0; dimension].into(),
            state_mass: native_diagonal_precision(dimension, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(dimension, 1.0)),
            layout: DofLayout::identity(dimension),
            forcing_operator: native_diagonal_precision(dimension, -1.0),
            neumann_operator: native_diagonal_precision(dimension, -1.0),
        };
        let measurement_operator = SparseTripletMatrix::from_triplets(
            dimension,
            dimension,
            [
                feg_core::SparseTriplet {
                    row: 0,
                    col: 0,
                    value: 1.0,
                },
                feg_core::SparseTriplet {
                    row: 1,
                    col: 1,
                    value: 1.0,
                },
            ],
        );
        let derived_operator = SparseRowOperator::new(2, vec![vec![(0, 2.0), (1, -0.5)]])
            .expect("derived operator should be valid");
        LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; dimension],
                precision: SparseTripletMatrix::from_triplets(
                    dimension,
                    dimension,
                    [
                        feg_core::SparseTriplet {
                            row: 0,
                            col: 0,
                            value: 1.0e-6,
                        },
                        feg_core::SparseTriplet {
                            row: 1,
                            col: 1,
                            value: 1.0e6,
                        },
                    ],
                ),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: vec![LinearGaussianMeasurementSpec {
                name: "identity".to_string(),
                operator: measurement_operator,
                observations: vec![3.0, -2.0],
                bias: vec![0.0; dimension],
                variance: 0.25,
            }],
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "combo".to_string(),
                operator: derived_operator,
            }],
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: None,
        }
    }

    fn scaled_coordinate_test_problem(preference: RepresentationPreference) -> LinearPdeUqProblem {
        let dimension = 2;
        let mut pde_operator = SparseTripletMatrix::new(2, 2);
        pde_operator.push(0, 0, 1.0);
        pde_operator.push(0, 1, 2.0);
        pde_operator.push(1, 0, -0.25);
        pde_operator.push(1, 1, 1.5);
        let mut source_operator = SparseTripletMatrix::new(2, 1);
        source_operator.push(0, 0, -1.5);
        source_operator.push(1, 0, 0.5);
        let mut measurement_operator = SparseTripletMatrix::new(2, 2);
        measurement_operator.push(0, 0, 1.0);
        measurement_operator.push(1, 1, 1.0);
        let state_combo = SparseRowOperator::new(2, vec![vec![(0, 2.0), (1, -0.5)]])
            .expect("state combo operator should be valid");
        LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![1.0, -2.0],
                precision: tridiagonal_precision(2, 3.0, -0.2),
            },
            system: ReducedLinearPdeAssembly {
                operator: core_triplet_to_feec_csr(&pde_operator),
                residual_bias: vec![0.1, -0.2].into(),
                state_mass: native_diagonal_precision(dimension, 1.0),
                state_mass_inverse: Some(native_diagonal_precision(dimension, 1.0)),
                layout: DofLayout::identity(dimension),
                forcing_operator: native_diagonal_precision(dimension, -1.0),
                neumann_operator: native_diagonal_precision(dimension, -1.0),
            },
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "source".to_string(),
                operator: source_operator,
                prior: GaussianPriorSpec {
                    mean: vec![0.3],
                    precision: diagonal_precision(1, 4.0),
                },
                preference,
                collapsed_precision: None,
            }],
            physical_measurements: vec![LinearGaussianMeasurementSpec {
                name: "identity".to_string(),
                operator: measurement_operator,
                observations: vec![0.6, -1.1],
                bias: vec![0.0; dimension],
                variance: 0.7,
            }],
            joint_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "combo".to_string(),
                operator: state_combo,
            }],
            joint_derived_quantities: vec![LinearPdeJointDerivedQuantitySpec {
                name: "source_plus_x".to_string(),
                state_operator: Some(
                    SparseRowOperator::new(2, vec![vec![(0, 1.0)]])
                        .expect("state row should be valid"),
                ),
                latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                    input_name: "source".to_string(),
                    operator: SparseRowOperator::new(1, vec![vec![(0, 3.0)]])
                        .expect("latent row should be valid"),
                }],
            }],
            pde_variance: Some(0.4),
            pde_precision: None,
        }
    }

    fn scaled_coordinate_test_scaling() -> LinearPdeCoordinateScaling {
        LinearPdeCoordinateScaling {
            state_scale: vec![2.0, 0.1],
            latent_scales: vec![LinearPdeLatentCoordinateScaling {
                input_name: "source".to_string(),
                scale: vec![0.25],
            }],
        }
    }

    #[test]
    fn non_spd_precision_fails_without_diagonal_ridge() {
        let precision = gmrf_sparse_matrix(2, [(0, 0, 1.0), (1, 1, -1.0)]);
        let config = LinearPdeUqSolverConfig::default();
        let err = match factorize_precision_with_diagnostics("test", &precision, &config) {
            Ok(_) => panic!("indefinite precision should not factorize"),
            Err(err) => err,
        };

        assert!(err.contains("Cholesky") || err.contains("positive"));
        assert!(
            !err.contains("shift") && !err.contains("ridge"),
            "failure must not mention an added stabilizing shift: {err}"
        );
    }

    #[test]
    fn small_asymmetric_precision_is_symmetrized_below_tolerance() {
        let precision = gmrf_sparse_matrix(
            2,
            [(0, 0, 2.0), (1, 1, 2.0), (0, 1, 0.1), (1, 0, 0.1 + 1.0e-12)],
        );
        let config = LinearPdeUqSolverConfig::default();
        let factorized = factorize_precision_with_diagnostics("test", &precision, &config)
            .expect("small asymmetry should be canonicalized");

        assert!(factorized.debug.factor_nnz > 0);
        assert!(factorized.debug.max_relative_asymmetry > 0.0);
        assert!(
            factorized.debug.max_relative_asymmetry
                <= config.precision_policy.max_relative_asymmetry()
        );
    }

    #[test]
    fn large_asymmetric_precision_fails_with_diagnostics() {
        let precision = gmrf_sparse_matrix(2, [(0, 0, 2.0), (1, 1, 2.0), (0, 1, 1.0e-4)]);
        let config = LinearPdeUqSolverConfig::default();
        let err = match factorize_precision_with_diagnostics("test", &precision, &config) {
            Ok(_) => panic!("large asymmetry should fail before factorization"),
            Err(err) => err,
        };

        assert!(err.contains("asymmetry"));
        assert!(err.contains("max_rel_asym"));
        assert!(err.contains("asym_tolerance"));
    }

    #[test]
    fn diagonal_equilibration_preserves_mixed_scale_mean_and_variance() {
        let problem = mixed_scale_gaussian_problem();
        let fail_fast = solve_linear_pde_uq_with_config(
            &problem,
            &LinearPdeUqSolverConfig {
                precision_policy: LinearPdePrecisionPolicy::default(),
                ..LinearPdeUqSolverConfig::default()
            },
        )
        .expect("fail-fast solve should succeed");
        let equilibrated = solve_linear_pde_uq_with_config(
            &problem,
            &LinearPdeUqSolverConfig {
                precision_policy: LinearPdePrecisionPolicy::DiagonalEquilibrated {
                    max_relative_asymmetry: 1.0e-10,
                },
                ..LinearPdeUqSolverConfig::default()
            },
        )
        .expect("equilibrated solve should succeed");

        assert!(
            max_abs_difference(&fail_fast.posterior_mean, &equilibrated.posterior_mean) < 1.0e-9
        );
        assert!(
            max_abs_difference(
                &fail_fast.posterior_variance,
                &equilibrated.posterior_variance
            ) < 1.0e-9
        );
        assert_eq!(
            equilibrated
                .debug
                .posterior_factorization
                .precision_policy
                .label(),
            "diagonal_equilibrated"
        );
        assert!(equilibrated
            .debug
            .posterior_factorization
            .equilibration_min_scale
            .is_finite());
        assert!(equilibrated
            .debug
            .posterior_factorization
            .equilibration_max_scale
            .is_finite());
    }

    #[test]
    fn derived_quantity_variance_stays_in_original_units_under_equilibration() {
        let problem = mixed_scale_gaussian_problem();
        let fail_fast =
            solve_linear_pde_uq_with_config(&problem, &LinearPdeUqSolverConfig::default())
                .expect("fail-fast solve should succeed");
        let equilibrated = solve_linear_pde_uq_with_config(
            &problem,
            &LinearPdeUqSolverConfig {
                precision_policy: LinearPdePrecisionPolicy::DiagonalEquilibrated {
                    max_relative_asymmetry: 1.0e-10,
                },
                ..LinearPdeUqSolverConfig::default()
            },
        )
        .expect("equilibrated solve should succeed");
        let fail_fast_combo = fail_fast
            .derived_variances
            .get("combo")
            .expect("combo derived variance should be reported");
        let equilibrated_combo = equilibrated
            .derived_variances
            .get("combo")
            .expect("combo derived variance should be reported");

        assert!(
            max_abs_difference(
                &fail_fast_combo.posterior_variance,
                &equilibrated_combo.posterior_variance
            ) < 1.0e-9
        );
        assert!(
            (equilibrated_combo.posterior_variance[0] - 1.0).abs() < 1.0e-5,
            "derived posterior variance should be near the original-unit analytic value"
        );
    }

    #[test]
    fn scaled_linear_pde_coordinates_preserve_physical_outputs() {
        let problem = scaled_coordinate_test_problem(RepresentationPreference::ForceLatent);
        let scaling = scaled_coordinate_test_scaling();
        let config = LinearPdeUqSolverConfig::default();
        let unscaled = solve_linear_pde_uq_with_config(&problem, &config)
            .expect("unscaled solve should succeed");
        let scaled = solve_scaled_linear_pde_uq_with_config(&problem, &scaling, &config)
            .expect("scaled solve should succeed");

        assert!(max_abs_difference(&unscaled.posterior_mean, &scaled.posterior_mean) < 1.0e-10);
        assert!(
            max_abs_difference(&unscaled.posterior_variance, &scaled.posterior_variance) < 1.0e-10
        );
        assert!(
            max_abs_difference(
                &unscaled.reduced_posterior_mean,
                &scaled.reduced_posterior_mean
            ) < 1.0e-10
        );
        assert!(
            max_abs_difference(
                &unscaled.reduced_posterior_variance,
                &scaled.reduced_posterior_variance
            ) < 1.0e-10
        );
        assert!(
            max_abs_difference(&unscaled.pde_residual_mean, &scaled.pde_residual_mean) < 1.0e-10
        );

        let unscaled_source = unscaled
            .latent_inputs
            .iter()
            .find(|input| input.name == "source")
            .expect("unscaled source posterior should be present");
        let scaled_source = scaled
            .latent_inputs
            .iter()
            .find(|input| input.name == "source")
            .expect("scaled source posterior should be present");
        assert!((unscaled_source.mean[0] - scaled_source.mean[0]).abs() < 1.0e-10);
        assert!((unscaled_source.variance[0] - scaled_source.variance[0]).abs() < 1.0e-10);
    }

    #[test]
    fn scaled_linear_pde_derived_variances_stay_physical() {
        let problem = scaled_coordinate_test_problem(RepresentationPreference::ForceLatent);
        let scaling = scaled_coordinate_test_scaling();
        let config = LinearPdeUqSolverConfig::default();
        let unscaled = solve_linear_pde_uq_with_config(&problem, &config)
            .expect("unscaled solve should succeed");
        let scaled = solve_scaled_linear_pde_uq_with_config(&problem, &scaling, &config)
            .expect("scaled solve should succeed");

        for name in ["combo", "source_plus_x"] {
            let unscaled_derived = unscaled
                .derived_variances
                .get(name)
                .unwrap_or_else(|| panic!("{name} should be reported"));
            let scaled_derived = scaled
                .derived_variances
                .get(name)
                .unwrap_or_else(|| panic!("{name} should be reported"));
            assert!(
                max_abs_difference(
                    &unscaled_derived.prior_variance,
                    &scaled_derived.prior_variance
                ) < 1.0e-10,
                "{name} prior variance changed under coordinate scaling"
            );
            assert!(
                max_abs_difference(
                    &unscaled_derived.posterior_variance,
                    &scaled_derived.posterior_variance
                ) < 1.0e-10,
                "{name} posterior variance changed under coordinate scaling"
            );
        }
    }

    #[test]
    fn scaled_linear_pde_rejects_invalid_coordinate_scales() {
        let problem = scaled_coordinate_test_problem(RepresentationPreference::ForceLatent);
        let err = solve_scaled_linear_pde_uq_with_config(
            &problem,
            &LinearPdeCoordinateScaling {
                state_scale: vec![1.0],
                latent_scales: Vec::new(),
            },
            &LinearPdeUqSolverConfig::default(),
        )
        .expect_err("wrong state scale length should fail");
        assert!(err.contains("state coordinate scale length"));

        let err = solve_scaled_linear_pde_uq_with_config(
            &problem,
            &LinearPdeCoordinateScaling {
                state_scale: vec![1.0, 0.0],
                latent_scales: Vec::new(),
            },
            &LinearPdeUqSolverConfig::default(),
        )
        .expect_err("zero state scale should fail");
        assert!(err.contains("finite and positive"));

        let err = solve_scaled_linear_pde_uq_with_config(
            &problem,
            &LinearPdeCoordinateScaling {
                state_scale: vec![1.0, 1.0],
                latent_scales: vec![
                    LinearPdeLatentCoordinateScaling {
                        input_name: "source".to_string(),
                        scale: vec![1.0],
                    },
                    LinearPdeLatentCoordinateScaling {
                        input_name: "source".to_string(),
                        scale: vec![1.0],
                    },
                ],
            },
            &LinearPdeUqSolverConfig::default(),
        )
        .expect_err("duplicate latent scale should fail");
        assert!(err.contains("duplicate latent input"));

        let err = solve_scaled_linear_pde_uq_with_config(
            &problem,
            &LinearPdeCoordinateScaling {
                state_scale: vec![1.0, 1.0],
                latent_scales: vec![LinearPdeLatentCoordinateScaling {
                    input_name: "source".to_string(),
                    scale: vec![1.0, 2.0],
                }],
            },
            &LinearPdeUqSolverConfig::default(),
        )
        .expect_err("wrong latent scale length should fail");
        assert!(err.contains("latent `source` coordinate scale length"));

        let err = solve_scaled_linear_pde_uq_with_config(
            &problem,
            &LinearPdeCoordinateScaling {
                state_scale: vec![1.0, 1.0],
                latent_scales: vec![LinearPdeLatentCoordinateScaling {
                    input_name: "missing".to_string(),
                    scale: vec![1.0],
                }],
            },
            &LinearPdeUqSolverConfig::default(),
        )
        .expect_err("unknown latent scale should fail");
        assert!(err.contains("unknown latent input"));

        let auto_problem = scaled_coordinate_test_problem(RepresentationPreference::Auto);
        let err = solve_scaled_linear_pde_uq_with_config(
            &auto_problem,
            &scaled_coordinate_test_scaling(),
            &LinearPdeUqSolverConfig::default(),
        )
        .expect_err("latent coordinate scaling should require ForceLatent");
        assert!(err.contains("requires ForceLatent"));
    }

    fn small_variance_mode_problem() -> LinearPdeUqProblem {
        let dimension = 3;
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(dimension, 1.0),
            residual_bias: vec![0.1, -0.2, 0.3].into(),
            state_mass: native_diagonal_precision(dimension, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(dimension, 1.0)),
            layout: DofLayout::identity(dimension),
            forcing_operator: native_diagonal_precision(dimension, -1.0),
            neumann_operator: native_diagonal_precision(dimension, -1.0),
        };
        let edge_operator = SparseRowOperator::new(
            dimension,
            vec![vec![(0, -1.0), (1, 1.0)], vec![(1, -1.0), (2, 1.0)]],
        )
        .expect("edge operator should be valid");
        LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; dimension],
                precision: tridiagonal_precision(dimension, 2.0, -0.25),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "edges".to_string(),
                operator: edge_operator,
            }],
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(0.4),
            pde_precision: None,
        }
    }

    #[test]
    fn variance_estimator_modes_run_through_linear_pde_path() {
        for mode in [
            LinearPdeVarianceMode::Exact,
            LinearPdeVarianceMode::ExactSolves,
            LinearPdeVarianceMode::SelectedInverse,
            LinearPdeVarianceMode::MonteCarlo,
            LinearPdeVarianceMode::Hutchinson,
            LinearPdeVarianceMode::LocalRbmc,
        ] {
            let problem = small_variance_mode_problem();
            let result = solve_linear_pde_uq_with_config(
                &problem,
                &LinearPdeUqSolverConfig {
                    variance: LinearPdeVarianceConfig {
                        mode,
                        num_variance_probes: 16,
                        variance_batch_count: 4,
                        rng_seed: 99,
                        local_rb_block_size: 2,
                    },
                    precision_policy: LinearPdePrecisionPolicy::default(),
                    log_diagnostics: false,
                },
            )
            .unwrap_or_else(|err| panic!("{mode:?} variance solve should succeed: {err}"));

            assert_eq!(result.posterior_variance.len(), 3);
            assert!(result
                .posterior_variance
                .iter()
                .all(|value| value.is_finite()));
            let derived = result
                .derived_variances
                .get("edges")
                .unwrap_or_else(|| panic!("{mode:?} should report derived edge variances"));
            assert_eq!(derived.posterior_variance.len(), 2);
            assert!(derived
                .posterior_variance
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn transformed_local_rb_default_uses_support_patches_not_row_blocks() {
        let operator = SparseRowOperator::new(
            8,
            vec![
                vec![(0, -1.0), (1, 1.0)],
                vec![(1, -1.0), (2, 1.0)],
                vec![(2, -1.0), (3, 1.0)],
                vec![(3, -1.0), (4, 1.0)],
                vec![(4, -1.0), (5, 1.0)],
                vec![(5, -1.0), (6, 1.0)],
                vec![(6, -1.0), (7, 1.0)],
            ],
        )
        .expect("chain operator should be valid");
        let permutation = Permutation::identity(8);

        let patched = transformed_support_patch_blocks(&operator, &permutation, 4)
            .expect("support patches should build");
        let row_support =
            row_support_blocks(&operator, &permutation).expect("row-support baseline should build");

        let (patch_blocks, patch_assignments) = match patched {
            LatentBlockMode::Explicit {
                blocks,
                row_assignments: Some(row_assignments),
            } => (blocks, row_assignments),
            _ => panic!("support patches should use explicit row assignments"),
        };
        let row_block_count = match row_support {
            LatentBlockMode::Explicit { blocks, .. } => blocks.len(),
            _ => panic!("row-support baseline should use explicit blocks"),
        };

        assert!(
            patch_blocks.len() < row_block_count,
            "support patches should coarsen the row-per-block baseline"
        );
        assert_eq!(patch_assignments.len(), operator.nrows());
        for (row_index, row) in operator.rows.iter().enumerate() {
            let assigned_block = patch_assignments[row_index].0;
            let block = &patch_blocks[assigned_block];
            for (column, _) in row {
                assert!(
                    block.contains(&PermutedIndex(*column)),
                    "row {row_index} support must be contained in its assigned patch"
                );
            }
        }
    }

    fn build_reduced_1form_whittle_prior(system: &ReducedLinearPdeAssembly) -> SparseTripletMatrix {
        let operator = system.operator.clone();
        let mass = system.state_mass.clone();
        let mass_inverse = system
            .state_mass_inverse
            .clone()
            .expect("mixed 1-form reduced system should expose a projected inverse");
        let a = add_sparse(&operator, &mass);
        let precision = &a.transpose() * &(&mass_inverse * &a);
        to_core_triplets(&precision)
    }

    #[test]
    fn collapsed_and_latent_forcing_agree_for_identity_residual_map() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system = build_reduced_laplace_beltrami_system(
            &topology,
            &geometry,
            &EssentialBoundarySpec::default(),
        )
        .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: system.operator.clone(),
                mass: system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let forcing_mean = vec![1.0; system.residual_dimension()];
        let forcing_precision = diagonal_precision(system.residual_dimension(), 4.0);
        let residual_variance = 0.5;
        let collapsed_precision = diagonal_precision(system.residual_dimension(), 1.0 / 0.75);

        let build_problem =
            |preference, collapsed_precision: Option<SparseTripletMatrix>, pde_variance| {
                LinearPdeUqProblem {
                    state_prior: GaussianPriorSpec {
                        mean: vec![0.0; system.state_dimension()],
                        precision: to_core_triplets(&precision),
                    },
                    system: system.clone(),
                    uncertain_inputs: vec![LinearUncertainInputSpec {
                        name: "forcing".to_string(),
                        operator: feec_csr_to_core_triplet(&system.forcing_operator),
                        prior: GaussianPriorSpec {
                            mean: forcing_mean.clone(),
                            precision: forcing_precision.clone(),
                        },
                        preference,
                        collapsed_precision,
                    }],
                    joint_measurements: Vec::new(),
                    physical_measurements: Vec::new(),
                    derived_quantities: Vec::new(),
                    joint_derived_quantities: Vec::new(),
                    pde_variance,
                    pde_precision: None,
                }
            };

        let collapsed = solve_linear_pde_uq(&build_problem(
            RepresentationPreference::Auto,
            Some(collapsed_precision),
            None,
        ))
        .expect("collapsed forcing solve should succeed");
        let latent = solve_linear_pde_uq(&build_problem(
            RepresentationPreference::ForceLatent,
            None,
            Some(residual_variance),
        ))
        .expect("latent forcing solve should succeed");

        assert!(
            max_abs_difference(&collapsed.posterior_mean, &latent.posterior_mean) <= 1e-8,
            "collapsed and latent forcing means should agree"
        );
        assert!(
            max_abs_difference(&collapsed.posterior_variance, &latent.posterior_variance) <= 1e-8,
            "collapsed and latent forcing variances should agree"
        );
    }

    #[test]
    fn sparse_pde_precision_conditions_linear_residual() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(2, 1.0),
            residual_bias: vec![-1.0, 2.0].into(),
            state_mass: native_diagonal_precision(2, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(2, 1.0)),
            layout: DofLayout::identity(2),
            forcing_operator: native_diagonal_precision(2, -1.0),
            neumann_operator: native_diagonal_precision(2, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0, 0.0],
                precision: diagonal_precision(2, 1.0),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: Some(diagonal_precision(2, 1.0e6)),
        })
        .expect("sparse PDE precision solve should succeed");

        assert!((result.posterior_mean[0] - 1.0).abs() < 1e-4);
        assert!((result.posterior_mean[1] + 2.0).abs() < 1e-4);
        assert!(result
            .pde_residual_mean
            .iter()
            .all(|value| value.abs() < 1e-3));
    }

    #[test]
    fn latent_input_posterior_reports_actual_mean_and_variance() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(1, 1.0),
            residual_bias: vec![0.0].into(),
            state_mass: native_diagonal_precision(1, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(1, 1.0)),
            layout: DofLayout::identity(1),
            forcing_operator: native_diagonal_precision(1, -1.0),
            neumann_operator: native_diagonal_precision(1, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 1.0),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "calibration".to_string(),
                operator: diagonal_precision(1, -1.0),
                prior: GaussianPriorSpec {
                    mean: vec![2.0],
                    precision: diagonal_precision(1, 4.0),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1.0),
            pde_precision: None,
        })
        .expect("latent input solve should succeed");

        assert_eq!(result.latent_inputs.len(), 1);
        let input = &result.latent_inputs[0];
        assert_eq!(input.name, "calibration");
        assert_eq!(input.offset, 1);
        assert!((input.mean[0] - 16.0 / 9.0).abs() < 1e-12);
        assert!((input.variance[0] - 2.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn joint_measurement_recovers_analytic_scalar_latent_posterior() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(1, 1.0),
            residual_bias: vec![0.0].into(),
            state_mass: native_diagonal_precision(1, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(1, 1.0)),
            layout: DofLayout::identity(1),
            forcing_operator: native_diagonal_precision(1, -1.0),
            neumann_operator: native_diagonal_precision(1, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 1.0),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "beta".to_string(),
                operator: SparseTripletMatrix::new(1, 1),
                prior: GaussianPriorSpec {
                    mean: vec![0.0],
                    precision: diagonal_precision(1, 0.25),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: vec![LinearPdeJointMeasurementSpec {
                name: "beta_observation".to_string(),
                state_operator: None,
                latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
                    input_name: "beta".to_string(),
                    operator: SparseTripletMatrix::from_triplets(
                        1,
                        1,
                        [feg_core::SparseTriplet {
                            row: 0,
                            col: 0,
                            value: 1.0,
                        }],
                    ),
                }],
                observations: vec![3.0],
                bias: vec![0.0],
                variance: 0.25,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: None,
        })
        .expect("joint latent measurement solve should succeed");

        let beta = result
            .latent_inputs
            .iter()
            .find(|input| input.name == "beta")
            .expect("beta posterior should be reported");
        let expected_variance = 1.0 / (0.25 + 4.0);
        let expected_mean = expected_variance * 12.0;
        assert!((beta.mean[0] - expected_mean).abs() < 1e-12);
        assert!((beta.variance[0] - expected_variance).abs() < 1e-12);
    }

    #[test]
    fn flat_state_prior_uses_pde_and_joint_measurement_without_prior_precision() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(1, 1.0),
            residual_bias: vec![0.0].into(),
            state_mass: native_diagonal_precision(1, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(1, 1.0)),
            layout: DofLayout::identity(1),
            forcing_operator: native_diagonal_precision(1, -1.0),
            neumann_operator: native_diagonal_precision(1, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: SparseTripletMatrix::new(1, 1),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "alpha".to_string(),
                operator: SparseTripletMatrix::new(1, 1),
                prior: GaussianPriorSpec {
                    mean: vec![1.0],
                    precision: diagonal_precision(1, 1.0),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: vec![LinearPdeJointMeasurementSpec {
                name: "field_sensor".to_string(),
                state_operator: Some(diagonal_precision(1, 1.0)),
                latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
                    input_name: "alpha".to_string(),
                    operator: diagonal_precision(1, 1.0),
                }],
                observations: vec![1.25],
                bias: vec![0.0],
                variance: 1.0e-4,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1.0e-8),
            pde_precision: None,
        })
        .expect("flat-state joint solve should succeed");

        let alpha = result
            .latent_inputs
            .iter()
            .find(|input| input.name == "alpha")
            .expect("alpha posterior should be reported");
        assert!(result.debug.flat_state_prior);
        assert_eq!(result.debug.prior_factorization.factor_nnz, 0);
        assert!(result.prior_variance[0].is_infinite());
        assert!(result.posterior_variance[0].is_finite());
        assert!(result.posterior_variance[0] >= 0.0);
        assert!(result.posterior_mean[0].abs() < 1e-3);
        assert!((alpha.mean[0] - 1.25).abs() < 1e-3);
        assert!(alpha.variance[0].is_finite() && alpha.variance[0] >= 0.0);
    }

    #[test]
    fn joint_derived_quantity_reports_named_latent_block_variance() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(1, 1.0),
            residual_bias: vec![0.0].into(),
            state_mass: native_diagonal_precision(1, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(1, 1.0)),
            layout: DofLayout::identity(1),
            forcing_operator: native_diagonal_precision(1, -1.0),
            neumann_operator: native_diagonal_precision(1, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 1.0),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "beta".to_string(),
                operator: SparseTripletMatrix::new(1, 1),
                prior: GaussianPriorSpec {
                    mean: vec![0.0],
                    precision: diagonal_precision(1, 0.25),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: vec![LinearPdeJointDerivedQuantitySpec {
                name: "two_beta".to_string(),
                state_operator: None,
                latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                    input_name: "beta".to_string(),
                    operator: SparseRowOperator::new(1, vec![vec![(0, 2.0)]])
                        .expect("single-row latent operator should be valid"),
                }],
            }],
            pde_variance: None,
            pde_precision: None,
        })
        .expect("joint derived variance solve should succeed");

        let derived = result
            .derived_variances
            .get("two_beta")
            .expect("joint derived variance should be reported");
        assert!((derived.prior_variance[0] - 16.0).abs() < 1e-12);
        assert!((derived.posterior_variance[0] - 16.0).abs() < 1e-12);
    }

    #[test]
    fn pushforward_covariance_includes_state_latent_cross_terms() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(1, 1.0),
            residual_bias: vec![0.0].into(),
            state_mass: native_diagonal_precision(1, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(1, 1.0)),
            layout: DofLayout::identity(1),
            forcing_operator: native_diagonal_precision(1, -1.0),
            neumann_operator: native_diagonal_precision(1, -1.0),
        };
        let problem = LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 1.0),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "beta".to_string(),
                operator: SparseTripletMatrix::new(1, 1),
                prior: GaussianPriorSpec {
                    mean: vec![0.0],
                    precision: diagonal_precision(1, 0.25),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: vec![LinearPdeJointMeasurementSpec {
                name: "field_sensor".to_string(),
                state_operator: Some(diagonal_precision(1, 1.0)),
                latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
                    input_name: "beta".to_string(),
                    operator: diagonal_precision(1, 1.0),
                }],
                observations: vec![1.0],
                bias: vec![0.0],
                variance: 0.25,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "x".to_string(),
                operator: SparseRowOperator::new(1, vec![vec![(0, 1.0)]])
                    .expect("state row should be valid"),
            }],
            joint_derived_quantities: vec![LinearPdeJointDerivedQuantitySpec {
                name: "two_beta".to_string(),
                state_operator: None,
                latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                    input_name: "beta".to_string(),
                    operator: SparseRowOperator::new(1, vec![vec![(0, 2.0)]])
                        .expect("latent row should be valid"),
                }],
            }],
            pde_variance: None,
            pde_precision: None,
        };

        let covariance = solve_linear_pde_uq_with_pushforward_covariance(
            &problem,
            &LinearPdeUqSolverConfig::default(),
            &["x", "two_beta"],
        )
        .expect("pushforward covariance should solve");

        assert_eq!(covariance.names, vec!["x", "two_beta"]);
        assert!((covariance.prior_covariance[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((covariance.prior_covariance[(1, 1)] - 16.0).abs() < 1e-12);
        assert!(covariance.prior_covariance[(0, 1)].abs() < 1e-12);

        let det = 5.0 * 4.25 - 16.0;
        let expected_x_var = 4.25 / det;
        let expected_two_beta_var = 20.0 / det;
        let expected_cross = -8.0 / det;
        assert!((covariance.posterior_covariance[(0, 0)] - expected_x_var).abs() < 1e-12);
        assert!((covariance.posterior_covariance[(1, 1)] - expected_two_beta_var).abs() < 1e-12);
        assert!((covariance.posterior_covariance[(0, 1)] - expected_cross).abs() < 1e-12);
        assert!((covariance.posterior_covariance[(1, 0)] - expected_cross).abs() < 1e-12);
    }

    #[test]
    fn prior_only_linear_pde_solve_returns_prior_state() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(2, 1.0),
            residual_bias: vec![-3.0, 4.0].into(),
            state_mass: native_diagonal_precision(2, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(2, 1.0)),
            layout: DofLayout::identity(2),
            forcing_operator: native_diagonal_precision(2, -1.0),
            neumann_operator: native_diagonal_precision(2, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![1.5, -2.0],
                precision: diagonal_precision(2, 4.0),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: None,
        })
        .expect("prior-only solve should succeed");

        assert!((result.posterior_mean[0] - 1.5).abs() < 1e-12);
        assert!((result.posterior_mean[1] + 2.0).abs() < 1e-12);
        assert!((result.posterior_variance[0] - 0.25).abs() < 1e-12);
        assert!((result.posterior_variance[1] - 0.25).abs() < 1e-12);
        assert!(result.latent_inputs.is_empty());
        assert!((result.pde_residual_mean[0] + 1.5).abs() < 1e-12);
        assert!((result.pde_residual_mean[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn prior_only_latent_input_preserves_source_prior() {
        let system = ReducedLinearPdeAssembly {
            operator: native_diagonal_precision(1, 1.0),
            residual_bias: vec![0.0].into(),
            state_mass: native_diagonal_precision(1, 1.0),
            state_mass_inverse: Some(native_diagonal_precision(1, 1.0)),
            layout: DofLayout::identity(1),
            forcing_operator: native_diagonal_precision(1, -1.0),
            neumann_operator: native_diagonal_precision(1, -1.0),
        };
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 2.0),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "source_alpha".to_string(),
                operator: diagonal_precision(1, -1.0),
                prior: GaussianPriorSpec {
                    mean: vec![1.15],
                    precision: diagonal_precision(1, 25.0),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: None,
        })
        .expect("joint prior-only solve should succeed");

        assert_eq!(result.latent_inputs.len(), 1);
        let input = &result.latent_inputs[0];
        assert_eq!(input.name, "source_alpha");
        assert!((input.mean[0] - 1.15).abs() < 1e-12);
        assert!((input.variance[0] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn soft_dirichlet_boundary_approaches_hard_boundary_for_0form() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let boundary_dofs =
            assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, |point| {
                point[1] == 1.0
            });

        let hard_boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
            "hard",
            boundary_dofs.clone(),
            vec![1.0; boundary_dofs.len()],
            BoundaryTreatment::HardEssential,
        ));
        let soft_boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
            "soft",
            boundary_dofs.clone(),
            vec![1.0; boundary_dofs.len()],
            BoundaryTreatment::SoftEssential { variance: 1e-8 },
        ));

        let hard_adapted = adapt_boundary_spec(&hard_boundary, topology.nsimplices(0), 0).unwrap();
        let soft_adapted = adapt_boundary_spec(&soft_boundary, topology.nsimplices(0), 0).unwrap();
        let hard_system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &hard_adapted.essential)
                .expect("hard system should assemble");
        let soft_system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &soft_adapted.essential)
                .expect("soft system should assemble");

        let hard_precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: hard_system.operator.clone(),
                mass: hard_system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let soft_precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: soft_system.operator.clone(),
                mass: soft_system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        let hard = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; hard_system.state_dimension()],
                precision: to_core_triplets(&hard_precision),
            },
            system: hard_system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
            pde_precision: None,
        })
        .expect("hard solve should succeed");
        let soft = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; soft_system.state_dimension()],
                precision: to_core_triplets(&soft_precision),
            },
            system: soft_system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: soft_adapted.soft_state_measurements,
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
            pde_precision: None,
        })
        .expect("soft solve should succeed");

        for dof in boundary_dofs {
            assert!((hard.posterior_mean[dof] - 1.0).abs() < 1e-12);
            assert!((soft.posterior_mean[dof] - 1.0).abs() < 5e-3);
        }
    }

    #[test]
    fn uncertain_neumann_and_measurements_reduce_solution_variance_locally() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 3, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system = build_reduced_laplace_beltrami_system(
            &topology,
            &geometry,
            &EssentialBoundarySpec::default(),
        )
        .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: system.operator.clone(),
                mass: system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let precision = to_core_triplets(&precision);

        let measurement_vertex = 0usize;
        let measurement = LinearGaussianMeasurementSpec {
            name: "sensor".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                system.layout.full_dimension,
                [feg_core::SparseTriplet {
                    row: 0,
                    col: measurement_vertex,
                    value: 1.0,
                }],
            ),
            observations: vec![1.0],
            bias: vec![0.0],
            variance: 1e-6,
        };
        let neumann_precision = diagonal_precision(system.residual_dimension(), 1.0);

        let without_measurement = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: precision.clone(),
            },
            system: system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "neumann".to_string(),
                operator: feec_csr_to_core_triplet(&system.neumann_operator),
                prior: GaussianPriorSpec {
                    mean: vec![0.0; system.residual_dimension()],
                    precision: neumann_precision.clone(),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
            pde_precision: None,
        })
        .expect("solve without measurement should succeed");
        let with_measurement = {
            let measurement_problem = LinearPdeUqProblem {
                state_prior: GaussianPriorSpec {
                    mean: vec![0.0; system.state_dimension()],
                    precision,
                },
                system: system.clone(),
                uncertain_inputs: vec![LinearUncertainInputSpec {
                    name: "neumann".to_string(),
                    operator: feec_csr_to_core_triplet(&system.neumann_operator),
                    prior: GaussianPriorSpec {
                        mean: vec![0.0; system.residual_dimension()],
                        precision: neumann_precision,
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                }],
                joint_measurements: Vec::new(),
                physical_measurements: vec![measurement],
                derived_quantities: Vec::new(),
                joint_derived_quantities: Vec::new(),
                pde_variance: Some(1e-8),
                pde_precision: None,
            };
            solve_linear_pde_uq(&measurement_problem)
        }
        .expect("solve with measurement should succeed");

        let min_variance = with_measurement
            .posterior_variance
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_variance = with_measurement
            .posterior_variance
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            with_measurement.posterior_variance[measurement_vertex]
                < without_measurement.posterior_variance[measurement_vertex]
        );
        assert!(
            max_variance - min_variance > 1e-8,
            "uncertain inputs plus physical measurements should produce a non-uniform posterior variance"
        );
    }

    #[test]
    fn identity_derived_quantity_matches_state_marginal_variances() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system = build_reduced_laplace_beltrami_system(
            &topology,
            &geometry,
            &EssentialBoundarySpec::default(),
        )
        .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: system.operator.clone(),
                mass: system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: to_core_triplets(&precision),
            },
            system: system.clone(),
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "identity".to_string(),
                operator: SparseRowOperator::identity(system.layout.full_dimension),
            }],
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-6),
            pde_precision: None,
        })
        .expect("solve with identity derived quantity should succeed");

        let identity = result
            .derived_variances
            .get("identity")
            .expect("identity derived variance should be present");
        assert!(
            max_abs_difference(&identity.prior_variance, &result.prior_variance) <= 1e-10,
            "identity prior variance should match state prior variance"
        );
        assert!(
            max_abs_difference(&identity.posterior_variance, &result.posterior_variance) <= 1e-10,
            "identity posterior variance should match state posterior variance"
        );
    }

    #[test]
    fn joint_posterior_builder_exposes_restricted_identity_mean() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system = build_reduced_laplace_beltrami_system(
            &topology,
            &geometry,
            &EssentialBoundarySpec::default(),
        )
        .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: system.operator.clone(),
                mass: system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let problem = LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.5; system.state_dimension()],
                precision: to_core_triplets(&precision),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "identity".to_string(),
                operator: SparseRowOperator::identity(topology.nsimplices(0)),
            }],
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-6),
            pde_precision: None,
        };
        let config = LinearPdeUqSolverConfig {
            variance: LinearPdeVarianceConfig::default(),
            precision_policy: LinearPdePrecisionPolicy::default(),
            log_diagnostics: false,
        };
        let result = solve_linear_pde_uq_with_config(&problem, &config)
            .expect("solve with identity derived quantity should succeed");
        let posterior = build_linear_pde_joint_posterior_with_config(&problem, &config)
            .expect("joint posterior build should succeed");
        let identity = posterior
            .derived_quantities
            .get("identity")
            .expect("identity derived operator should be present");
        let applied = identity
            .apply(posterior.posterior.mean_vector())
            .expect("identity derived operator should apply");
        let expected_centered_mean = &result.reduced_posterior_mean
            - &FeecVector::from_vec(problem.state_prior.mean.clone());

        assert_eq!(posterior.state_dimension, problem.system.state_dimension());
        assert_eq!(posterior.joint_dimension, posterior.posterior.dimension());
        assert_eq!(applied.len(), expected_centered_mean.len());
        assert!(
            max_abs_difference(
                &gmrf_vec_to_feec(&applied),
                &expected_centered_mean,
            ) <= 1e-10,
            "joint posterior builder should expose the same centered reduced posterior mean as the solver"
        );
    }

    #[test]
    fn mixed_1form_system_with_latent_loads_produces_finite_solution_outputs() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let state_dofs = topology
            .boundary_subcomplex_simplices(1)
            .into_iter()
            .take(1)
            .map(|simp| simp.kidx)
            .collect::<Vec<_>>();
        let auxiliary_dofs = topology
            .boundary_subcomplex_simplices(0)
            .into_iter()
            .take(1)
            .map(|simp| simp.kidx)
            .collect::<Vec<_>>();
        let boundary = BoundarySpec::default()
            .with_state_region(BoundaryRegionSpec::new(
                "soft-state",
                state_dofs.clone(),
                vec![0.0],
                BoundaryTreatment::SoftEssential { variance: 1e-6 },
            ))
            .with_auxiliary_region(BoundaryRegionSpec::new(
                "hard-aux",
                auxiliary_dofs,
                vec![0.0],
                BoundaryTreatment::HardEssential,
            ));
        let adapted =
            adapt_boundary_spec(&boundary, topology.nsimplices(1), topology.nsimplices(0)).unwrap();
        let system =
            build_reduced_hodge_laplace_1form_system(&topology, &geometry, &adapted.essential)
                .expect("mixed 1-form system should assemble");
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: build_reduced_1form_whittle_prior(&system),
            },
            system: system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "forcing".to_string(),
                operator: feec_csr_to_core_triplet(&system.forcing_operator),
                prior: GaussianPriorSpec {
                    mean: vec![0.0; system.residual_dimension()],
                    precision: diagonal_precision(system.residual_dimension(), 1.0),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: Vec::new(),
            physical_measurements: adapted.soft_state_measurements,
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
            pde_precision: None,
        })
        .expect("mixed 1-form solve should succeed");

        assert!(result.posterior_mean.iter().all(|value| value.is_finite()));
        assert!(result
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn debug_reports_factorization_fill_stats() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system = build_reduced_laplace_beltrami_system(
            &topology,
            &geometry,
            &EssentialBoundarySpec::default(),
        )
        .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::prior::matern::zero_form::LaplaceBeltrami0Form {
                laplacian: system.operator.clone(),
                mass: system.state_mass.clone(),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let result = solve_linear_pde_uq_with_config(
            &LinearPdeUqProblem {
                state_prior: GaussianPriorSpec {
                    mean: vec![0.0; system.state_dimension()],
                    precision: to_core_triplets(&precision),
                },
                system,
                uncertain_inputs: Vec::new(),
                joint_measurements: Vec::new(),
                physical_measurements: Vec::new(),
                derived_quantities: Vec::new(),
                joint_derived_quantities: Vec::new(),
                pde_variance: Some(1e-6),
                pde_precision: None,
            },
            &LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig::default(),
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: false,
            },
        )
        .expect("diagnostic solve should succeed");

        let prior = result.debug.prior_factorization;
        let posterior = result.debug.posterior_factorization;
        assert_eq!(result.debug.joint_dimension, prior.dimension);
        assert_eq!(prior.dimension, posterior.dimension);
        assert!(prior.matrix_nnz >= prior.matrix_lower_triangle_nnz);
        assert!(posterior.matrix_nnz >= posterior.matrix_lower_triangle_nnz);
        assert!(prior.factor_nnz > 0);
        assert!(posterior.factor_nnz > 0);
        assert!(prior.fill_in_ratio_vs_lower_triangle > 0.0);
        assert!(posterior.fill_in_ratio_vs_lower_triangle > 0.0);
        assert!(prior.factor_numeric_values_mib > 0.0);
        assert!(posterior.factor_numeric_values_mib > 0.0);
    }

    #[test]
    fn auto_only_selects_collapsed_when_sparse_precision_is_available() {
        let input = LinearUncertainInputSpec {
            name: "nonlocal".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                2,
                [feg_core::SparseTriplet {
                    row: 0,
                    col: 1,
                    value: 2.0,
                }],
            ),
            prior: GaussianPriorSpec {
                mean: vec![0.0, 0.0],
                precision: SparseTripletMatrix::new(2, 2),
            },
            preference: RepresentationPreference::Auto,
            collapsed_precision: None,
        };
        let resolved = resolve_input_representations(&[input], 1).expect("resolution should work");
        assert_eq!(
            resolved[0].1,
            ResolvedInputRepresentation::Latent,
            "Auto should keep inputs latent when no sparse collapsed precision is available"
        );
    }
}
