use crate::linear_pde::{
    LinearPdeDerivedMarginalResult, LinearPdeDerivedQuantitySpec, LinearPdeVarianceConfig,
    LinearPdeVarianceMode,
};
use crate::sparse::{
    core_triplet_to_gmrf as sparse_from_core, gmrf_sparse_to_core_triplet as sparse_to_core,
};
use crate::sparse::{feec_csr_to_core_triplet, gmrf_vec_to_feec};
use feg_core::{
    GaussianPriorSpec, LinearGaussianMeasurementSpec, NonlinearResidualEvaluation,
    NonlinearResidualModel, PrecisionWeightedGaussianMeasurementSpec, SparseTriplet,
    SparseTripletMatrix,
};
use gmrf_core::observation::LinearObservationTerm;
use gmrf_core::types::{CooMatrix as GmrfCoo, SparseMatrix as GmrfSparseMatrix};
use gmrf_core::{
    apply_linear_observation_terms_with_stats, exact_solve_diag, exact_solve_transformed_diag,
    selected_inverse_diag, selected_inverse_transformed_diag, CholeskyOrdering, Gmrf,
    IterativeMethod, LinearObservationUpdateStats, PreconditionerKind, Solver, SolverAlgorithm,
    SolverConfig, SparseCholeskyFactor, SparseCholeskySymbolic, Vector as GmrfVector,
};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::{hash_map::DefaultHasher, BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum GaussianNoiseModel {
    ScalarVariance(f64),
    Precision(SparseTripletMatrix),
}

pub struct NonlinearResidualTerm<'a> {
    pub name: String,
    pub model: &'a dyn NonlinearResidualModel,
    pub observations: Vec<f64>,
    pub noise: GaussianNoiseModel,
}

impl<'a> NonlinearResidualTerm<'a> {
    pub fn zero(
        name: impl Into<String>,
        model: &'a dyn NonlinearResidualModel,
        noise: GaussianNoiseModel,
    ) -> Self {
        Self {
            name: name.into(),
            model,
            observations: vec![0.0; model.residual_dimension()],
            noise,
        }
    }
}

pub struct NonlinearLaplaceProblem<'a> {
    pub prior: GaussianPriorSpec,
    pub residual_terms: Vec<NonlinearResidualTerm<'a>>,
    pub linear_measurements: Vec<LinearGaussianMeasurementSpec>,
    pub precision_weighted_measurements: Vec<PrecisionWeightedGaussianMeasurementSpec>,
    pub derived_quantities: Vec<LinearPdeDerivedQuantitySpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothAbsLinearResidualModel {
    operator: SparseTripletMatrix,
    bias: Vec<f64>,
    smoothing: f64,
}

impl SmoothAbsLinearResidualModel {
    pub fn new(
        operator: SparseTripletMatrix,
        bias: Vec<f64>,
        smoothing: f64,
    ) -> Result<Self, String> {
        if operator.nrows() != bias.len() {
            return Err(format!(
                "smooth-abs linear residual bias length {} must match operator row count {}",
                bias.len(),
                operator.nrows()
            ));
        }
        if !smoothing.is_finite() || smoothing <= 0.0 {
            return Err("smooth-abs smoothing must be finite and positive".to_string());
        }
        if bias.iter().any(|value| !value.is_finite()) {
            return Err("smooth-abs linear residual bias contains a non-finite value".to_string());
        }
        for (row, col, value) in operator.triplet_iter() {
            if row >= operator.nrows() || col >= operator.ncols() {
                return Err(format!(
                    "smooth-abs linear residual operator entry ({row}, {col}) exceeds dimensions {}x{}",
                    operator.nrows(),
                    operator.ncols()
                ));
            }
            if !value.is_finite() {
                return Err(
                    "smooth-abs linear residual operator contains a non-finite value".to_string(),
                );
            }
        }
        Ok(Self {
            operator,
            bias,
            smoothing,
        })
    }

    pub fn operator(&self) -> &SparseTripletMatrix {
        &self.operator
    }

    pub fn bias(&self) -> &[f64] {
        &self.bias
    }

    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    pub fn smooth_abs_values(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        self.linear_values(state).map(|linear| {
            linear
                .into_iter()
                .map(|value| smooth_abs_value(value, self.smoothing))
                .collect()
        })
    }

    fn linear_values(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        let mut values = core_sparse_mul_vec(&self.operator, state)?;
        for (value, bias) in values.iter_mut().zip(self.bias.iter()) {
            *value += bias;
        }
        Ok(values)
    }
}

impl NonlinearResidualModel for SmoothAbsLinearResidualModel {
    fn state_dimension(&self) -> usize {
        self.operator.ncols()
    }

    fn residual_dimension(&self) -> usize {
        self.operator.nrows()
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        self.smooth_abs_values(state)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let linear_values = self.linear_values(state)?;
        let residual = linear_values
            .iter()
            .copied()
            .map(|value| smooth_abs_value(value, self.smoothing))
            .collect::<Vec<_>>();
        let scales = linear_values
            .iter()
            .zip(residual.iter())
            .map(|(linear, smooth)| linear / smooth)
            .collect::<Vec<_>>();
        let mut jacobian = BTreeMap::<(usize, usize), f64>::new();
        for (row, col, value) in self.operator.triplet_iter() {
            let scaled = scales[row] * value;
            if scaled != 0.0 {
                *jacobian.entry((row, col)).or_insert(0.0) += scaled;
            }
        }
        Ok(NonlinearResidualEvaluation {
            residual,
            jacobian: SparseTripletMatrix::from_triplets(
                self.operator.nrows(),
                self.operator.ncols(),
                jacobian
                    .into_iter()
                    .filter(|(_, value)| *value != 0.0)
                    .map(|((row, col), value)| SparseTriplet { row, col, value }),
            ),
        })
    }
}

fn smooth_abs_value(value: f64, smoothing: f64) -> f64 {
    (value * value + smoothing * smoothing).sqrt()
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothGroupedNormSample {
    pub rows: Vec<usize>,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothGroupedNormObservation {
    pub name: String,
    pub samples: Vec<SmoothGroupedNormSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothGroupedNormLinearResidualModel {
    operator: SparseTripletMatrix,
    bias: Vec<f64>,
    groups: Vec<SmoothGroupedNormObservation>,
    smoothing: f64,
}

impl SmoothGroupedNormLinearResidualModel {
    pub fn new(
        operator: SparseTripletMatrix,
        bias: Vec<f64>,
        groups: Vec<SmoothGroupedNormObservation>,
        smoothing: f64,
    ) -> Result<Self, String> {
        if operator.nrows() != bias.len() {
            return Err(format!(
                "smooth grouped-norm bias length {} must match operator row count {}",
                bias.len(),
                operator.nrows()
            ));
        }
        if groups.is_empty() {
            return Err("at least one smooth grouped-norm observation is required".to_string());
        }
        if !smoothing.is_finite() || smoothing <= 0.0 {
            return Err("smooth grouped-norm smoothing must be finite and positive".to_string());
        }
        if bias.iter().any(|value| !value.is_finite()) {
            return Err("smooth grouped-norm bias contains a non-finite value".to_string());
        }
        for (row, col, value) in operator.triplet_iter() {
            if row >= operator.nrows() || col >= operator.ncols() {
                return Err(format!(
                    "smooth grouped-norm operator entry ({row}, {col}) exceeds dimensions {}x{}",
                    operator.nrows(),
                    operator.ncols()
                ));
            }
            if !value.is_finite() {
                return Err("smooth grouped-norm operator contains a non-finite value".to_string());
            }
        }
        for group in &groups {
            if group.samples.is_empty() {
                return Err(format!(
                    "smooth grouped-norm observation `{}` must contain at least one sample",
                    group.name
                ));
            }
            for sample in &group.samples {
                if sample.rows.is_empty() {
                    return Err(format!(
                        "smooth grouped-norm observation `{}` contains an empty sample",
                        group.name
                    ));
                }
                if !sample.weight.is_finite() {
                    return Err(format!(
                        "smooth grouped-norm observation `{}` contains a non-finite sample weight",
                        group.name
                    ));
                }
                for row in &sample.rows {
                    if *row >= operator.nrows() {
                        return Err(format!(
                            "smooth grouped-norm observation `{}` references row {}, but operator has {} rows",
                            group.name,
                            row,
                            operator.nrows()
                        ));
                    }
                }
            }
        }
        Ok(Self {
            operator,
            bias,
            groups,
            smoothing,
        })
    }

    pub fn operator(&self) -> &SparseTripletMatrix {
        &self.operator
    }

    pub fn bias(&self) -> &[f64] {
        &self.bias
    }

    pub fn groups(&self) -> &[SmoothGroupedNormObservation] {
        &self.groups
    }

    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    pub fn smooth_norm_values(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        let linear_values = self.linear_values(state)?;
        Ok(self.group_values_from_linear_values(&linear_values))
    }

    fn linear_values(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        let mut values = core_sparse_mul_vec(&self.operator, state)?;
        for (value, bias) in values.iter_mut().zip(self.bias.iter()) {
            *value += bias;
        }
        Ok(values)
    }

    fn group_values_from_linear_values(&self, linear_values: &[f64]) -> Vec<f64> {
        self.groups
            .iter()
            .map(|group| {
                group
                    .samples
                    .iter()
                    .map(|sample| {
                        let squared_norm = sample
                            .rows
                            .iter()
                            .map(|row| linear_values[*row] * linear_values[*row])
                            .sum::<f64>()
                            + self.smoothing * self.smoothing;
                        sample.weight * squared_norm.sqrt()
                    })
                    .sum()
            })
            .collect()
    }
}

impl NonlinearResidualModel for SmoothGroupedNormLinearResidualModel {
    fn state_dimension(&self) -> usize {
        self.operator.ncols()
    }

    fn residual_dimension(&self) -> usize {
        self.groups.len()
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        self.smooth_norm_values(state)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let linear_values = self.linear_values(state)?;
        let residual = self.group_values_from_linear_values(&linear_values);
        let mut operator_rows = vec![Vec::<(usize, f64)>::new(); self.operator.nrows()];
        for (row, col, value) in self.operator.triplet_iter() {
            operator_rows[row].push((col, value));
        }

        let mut jacobian = BTreeMap::<(usize, usize), f64>::new();
        for (group_index, group) in self.groups.iter().enumerate() {
            for sample in &group.samples {
                let squared_norm = sample
                    .rows
                    .iter()
                    .map(|row| linear_values[*row] * linear_values[*row])
                    .sum::<f64>()
                    + self.smoothing * self.smoothing;
                let smooth_norm = squared_norm.sqrt();
                for row in &sample.rows {
                    let scale = sample.weight * linear_values[*row] / smooth_norm;
                    if scale == 0.0 {
                        continue;
                    }
                    for (col, value) in &operator_rows[*row] {
                        let scaled = scale * *value;
                        if scaled != 0.0 {
                            *jacobian.entry((group_index, *col)).or_insert(0.0) += scaled;
                        }
                    }
                }
            }
        }

        Ok(NonlinearResidualEvaluation {
            residual,
            jacobian: SparseTripletMatrix::from_triplets(
                self.groups.len(),
                self.operator.ncols(),
                jacobian
                    .into_iter()
                    .filter(|(_, value)| *value != 0.0)
                    .map(|((row, col), value)| SparseTriplet { row, col, value }),
            ),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GaussNewtonConfig {
    pub initial_guess: Option<Vec<f64>>,
    pub max_iterations: usize,
    pub step_tolerance: f64,
    pub gradient_tolerance: f64,
    pub armijo_c1: f64,
    pub max_line_search_steps: usize,
    pub step_regularization: GaussNewtonStepRegularization,
    pub linear_solve: GaussNewtonLinearSolve,
    pub stabilize_precision: bool,
    pub reuse_cholesky_stabilization_shift: bool,
    pub estimate_latent_variance: bool,
    pub variance: LinearPdeVarianceConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaussNewtonLinearSolve {
    IterativeCg {
        tolerance: f64,
        max_iterations: usize,
        warm_start: bool,
    },
    DirectCholesky,
}

impl GaussNewtonLinearSolve {
    pub fn iterative_cg_default() -> Self {
        Self::IterativeCg {
            tolerance: 1e-8,
            max_iterations: 2048,
            warm_start: true,
        }
    }
}

impl Default for GaussNewtonLinearSolve {
    fn default() -> Self {
        Self::iterative_cg_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GaussNewtonStepRegularization {
    #[default]
    None,
    LevenbergMarquardtGrid,
    AdaptiveLevenbergMarquardt,
}

impl GaussNewtonStepRegularization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LevenbergMarquardtGrid => "lm-grid",
            Self::AdaptiveLevenbergMarquardt => "adaptive-lm",
        }
    }
}

impl FromStr for GaussNewtonStepRegularization {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" | "off" => Ok(Self::None),
            "lm-grid" | "levenberg-marquardt-grid" | "grid" => Ok(Self::LevenbergMarquardtGrid),
            "adaptive-lm" | "adaptive-levenberg-marquardt" | "adaptive" => {
                Ok(Self::AdaptiveLevenbergMarquardt)
            }
            other => Err(format!(
                "unknown Gauss-Newton step regularization `{other}`; expected none, lm-grid, or adaptive-lm"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussNewtonLinearSolveMode {
    IterativeCg,
    DirectCholesky,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussNewtonLinearSolveStats {
    pub mode: GaussNewtonLinearSolveMode,
    pub iterations: usize,
    pub final_residual_norm: f64,
    pub converged: bool,
    pub factor_nnz: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GaussNewtonRunDiagnostics {
    pub step_solve_attempts: usize,
    pub accepted_iterations: usize,
    pub line_search_residual_evaluations: usize,
    pub final_factorizations: usize,
    pub metis_cache_hits: usize,
    pub metis_cache_misses: usize,
    pub cholesky_factor_attempts: usize,
    pub cholesky_factor_successes: usize,
    pub cholesky_unshifted_attempts: usize,
    pub cholesky_symmetrized_attempts: usize,
    pub cholesky_cached_shift_attempts: usize,
    pub cholesky_cached_shift_successes: usize,
    pub cholesky_shifted_attempts: usize,
    pub cholesky_shifted_successes: usize,
    pub cholesky_max_shift: f64,
    pub cholesky_factorization_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaplaceFactorizationStats {
    pub nnz: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearAssemblyTermKind {
    LinearMeasurement,
    PrecisionWeightedMeasurement,
    Residual,
}

impl NonlinearAssemblyTermKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinearMeasurement => "linear_measurement",
            Self::PrecisionWeightedMeasurement => "precision_weighted_measurement",
            Self::Residual => "residual",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonlinearAssemblyTermStats {
    pub name: String,
    pub kind: NonlinearAssemblyTermKind,
    pub operator_rows: usize,
    pub operator_cols: usize,
    pub operator_nnz: usize,
    pub precision_update_nnz: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearAssemblyStats {
    pub dimension: usize,
    pub prior_precision_nnz: usize,
    pub prior_precision_lower_triangle_nnz: usize,
    pub terms: Vec<NonlinearAssemblyTermStats>,
    pub posterior_precision_nnz: usize,
    pub posterior_precision_lower_triangle_nnz: usize,
    pub factor_nnz: Option<usize>,
    pub fill_ratio_vs_lower_triangle: Option<f64>,
}

impl NonlinearAssemblyStats {
    pub fn term_operator_nnz(&self, kind: NonlinearAssemblyTermKind) -> usize {
        self.terms
            .iter()
            .filter(|term| term.kind == kind)
            .map(|term| term.operator_nnz)
            .sum()
    }

    pub fn term_precision_update_nnz(&self, kind: NonlinearAssemblyTermKind) -> usize {
        self.terms
            .iter()
            .filter(|term| term.kind == kind)
            .map(|term| term.precision_update_nnz)
            .sum()
    }

    fn set_posterior_precision_stats(&mut self, precision: &GmrfSparseMatrix) {
        self.posterior_precision_nnz = precision.nnz();
        self.posterior_precision_lower_triangle_nnz = lower_triangle_nnz(precision);
        self.refresh_fill_ratio();
    }

    fn set_factor_nnz(&mut self, factor_nnz: usize) {
        self.factor_nnz = Some(factor_nnz);
        self.refresh_fill_ratio();
    }

    fn refresh_fill_ratio(&mut self) {
        self.fill_ratio_vs_lower_triangle = self.factor_nnz.map(|factor_nnz| {
            factor_nnz as f64 / self.posterior_precision_lower_triangle_nnz.max(1) as f64
        });
    }
}

impl Default for GaussNewtonConfig {
    fn default() -> Self {
        Self {
            initial_guess: None,
            max_iterations: 25,
            step_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            armijo_c1: 1e-4,
            max_line_search_steps: 12,
            step_regularization: GaussNewtonStepRegularization::default(),
            linear_solve: GaussNewtonLinearSolve::default(),
            stabilize_precision: true,
            reuse_cholesky_stabilization_shift: false,
            estimate_latent_variance: true,
            variance: LinearPdeVarianceConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SquareNewtonConfig {
    pub initial_guess: Option<Vec<f64>>,
    pub max_iterations: usize,
    pub residual_tolerance: f64,
    pub step_tolerance: f64,
    pub armijo_c1: f64,
    pub max_line_search_steps: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub stabilize_jacobian: bool,
}

impl Default for SquareNewtonConfig {
    fn default() -> Self {
        Self {
            initial_guess: None,
            max_iterations: 25,
            residual_tolerance: 1e-8,
            step_tolerance: 1e-10,
            armijo_c1: 1e-4,
            max_line_search_steps: 40,
            linear_solve: GaussNewtonLinearSolve::iterative_cg_default(),
            stabilize_jacobian: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SquareNewtonIteration {
    pub iteration: usize,
    pub residual_norm: f64,
    pub trial_residual_norm: f64,
    pub step_norm: f64,
    pub alpha: f64,
    pub linear_solve: GaussNewtonLinearSolveStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SquareNewtonResult {
    pub solution: Vec<f64>,
    pub residual: Vec<f64>,
    pub residual_norm: f64,
    pub history: Vec<SquareNewtonIteration>,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussNewtonIteration {
    pub iteration: usize,
    pub objective: f64,
    pub trial_objective: f64,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub alpha: f64,
    pub regularization: GaussNewtonStepRegularization,
    pub regularization_lambda: f64,
    pub residual_norm: f64,
    pub linear_solve: GaussNewtonLinearSolveStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearResidualReport {
    pub name: String,
    pub residual: Vec<f64>,
    pub weighted_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussNewtonObjectiveSample {
    pub alpha: f64,
    pub objective: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussNewtonFirstStepDiagnostics {
    pub objective: f64,
    pub weighted_residual_norm: f64,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub directional_derivative: f64,
    pub accepted_alpha: Option<f64>,
    pub accepted_objective: Option<f64>,
    pub regularization: GaussNewtonStepRegularization,
    pub regularization_lambda: f64,
    pub linear_solve: GaussNewtonLinearSolveStats,
    pub linear_solve_absolute_residual_norm: f64,
    pub linear_solve_relative_residual_norm: f64,
    pub assembly: NonlinearAssemblyStats,
    pub objective_grid: Vec<GaussNewtonObjectiveSample>,
}

pub struct NonlinearLaplaceResult {
    pub map: Vec<f64>,
    pub posterior_precision: SparseTripletMatrix,
    pub posterior_gmrf: Gmrf,
    pub posterior_variance: Vec<f64>,
    pub derived_variances: BTreeMap<String, LinearPdeDerivedMarginalResult>,
    pub final_residuals: Vec<NonlinearResidualReport>,
    pub history: Vec<GaussNewtonIteration>,
    pub assembly: NonlinearAssemblyStats,
    pub final_factorization: LaplaceFactorizationStats,
    pub diagnostics: GaussNewtonRunDiagnostics,
    pub converged: bool,
}

pub struct TransformedResidualModel<'a> {
    ambient_model: &'a dyn NonlinearResidualModel,
    transform: SparseTripletMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceNormalizedPrecision {
    pub precision: SparseTripletMatrix,
    pub normalization_scale: f64,
}

pub struct SelectedResidualModel<'a> {
    model: &'a dyn NonlinearResidualModel,
    rows: Vec<usize>,
}

impl<'a> SelectedResidualModel<'a> {
    pub fn new(model: &'a dyn NonlinearResidualModel, rows: Vec<usize>) -> Result<Self, String> {
        let mut seen = std::collections::BTreeSet::new();
        for row in &rows {
            if *row >= model.residual_dimension() {
                return Err(format!(
                    "selected residual row {row} is outside residual dimension {}",
                    model.residual_dimension()
                ));
            }
            if !seen.insert(*row) {
                return Err(format!(
                    "selected residual row {row} appears more than once"
                ));
            }
        }
        Ok(Self { model, rows })
    }

    pub fn rows(&self) -> &[usize] {
        &self.rows
    }
}

impl NonlinearResidualModel for SelectedResidualModel<'_> {
    fn state_dimension(&self) -> usize {
        self.model.state_dimension()
    }

    fn residual_dimension(&self) -> usize {
        self.rows.len()
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        let residual = self.model.residual(state)?;
        if residual.len() != self.model.residual_dimension() {
            return Err(format!(
                "selected residual source produced {} rows, expected {}",
                residual.len(),
                self.model.residual_dimension()
            ));
        }
        Ok(self.rows.iter().map(|row| residual[*row]).collect())
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let evaluation = self.model.residual_and_jacobian(state)?;
        evaluation.validate(
            self.model.residual_dimension(),
            self.model.state_dimension(),
        )?;
        let mut selected_residual = Vec::with_capacity(self.rows.len());
        let row_map = self
            .rows
            .iter()
            .copied()
            .enumerate()
            .map(|(selected, original)| (original, selected))
            .collect::<BTreeMap<_, _>>();
        for row in &self.rows {
            selected_residual.push(evaluation.residual[*row]);
        }
        let jacobian = SparseTripletMatrix::from_triplets(
            self.rows.len(),
            self.model.state_dimension(),
            evaluation
                .jacobian
                .triplet_iter()
                .filter_map(|(row, col, value)| {
                    row_map.get(&row).map(|selected_row| SparseTriplet {
                        row: *selected_row,
                        col,
                        value,
                    })
                }),
        );
        Ok(NonlinearResidualEvaluation {
            residual: selected_residual,
            jacobian,
        })
    }
}

pub fn trace_normalized_precision(
    base_precision: &SparseTripletMatrix,
    variance: f64,
) -> Result<TraceNormalizedPrecision, String> {
    if base_precision.nrows() == 0 || base_precision.nrows() != base_precision.ncols() {
        return Err(format!(
            "trace-normalized precision requires a nonempty square matrix, got {}x{}",
            base_precision.nrows(),
            base_precision.ncols()
        ));
    }
    if !variance.is_finite() || variance <= 0.0 {
        return Err("trace-normalized precision variance must be finite and positive".to_string());
    }
    let mut trace = 0.0;
    for (row, col, value) in base_precision.triplet_iter() {
        if row == col {
            trace += value;
        }
    }
    if !trace.is_finite() || trace <= 0.0 {
        return Err(format!(
            "trace-normalized precision requires positive finite trace, got {trace:.6e}"
        ));
    }
    let normalization_scale = base_precision.nrows() as f64 / trace;
    let scale = normalization_scale / variance;
    Ok(TraceNormalizedPrecision {
        precision: SparseTripletMatrix::from_triplets(
            base_precision.nrows(),
            base_precision.ncols(),
            base_precision
                .triplet_iter()
                .map(|(row, col, value)| SparseTriplet {
                    row,
                    col,
                    value: scale * value,
                }),
        ),
        normalization_scale,
    })
}

impl<'a> TransformedResidualModel<'a> {
    pub fn new(
        ambient_model: &'a dyn NonlinearResidualModel,
        transform: SparseTripletMatrix,
    ) -> Result<Self, String> {
        if transform.nrows() != ambient_model.state_dimension() {
            return Err(format!(
                "residual transform row count {} must match ambient state dimension {}",
                transform.nrows(),
                ambient_model.state_dimension()
            ));
        }
        Ok(Self {
            ambient_model,
            transform,
        })
    }

    pub fn transform(&self) -> &SparseTripletMatrix {
        &self.transform
    }
}

impl NonlinearResidualModel for TransformedResidualModel<'_> {
    fn state_dimension(&self) -> usize {
        self.transform.ncols()
    }

    fn residual_dimension(&self) -> usize {
        self.ambient_model.residual_dimension()
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        if state.len() != self.state_dimension() {
            return Err(format!(
                "latent state length {} must match transformed residual dimension {}",
                state.len(),
                self.state_dimension()
            ));
        }
        let ambient_state = core_sparse_mul_vec(&self.transform, state)?;
        let residual = self.ambient_model.residual(&ambient_state)?;
        if residual.len() != self.ambient_model.residual_dimension() {
            return Err(format!(
                "transformed residual source produced {} rows, expected {}",
                residual.len(),
                self.ambient_model.residual_dimension()
            ));
        }
        Ok(residual)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        if state.len() != self.state_dimension() {
            return Err(format!(
                "latent state length {} must match transformed residual dimension {}",
                state.len(),
                self.state_dimension()
            ));
        }
        let ambient_state = core_sparse_mul_vec(&self.transform, state)?;
        let ambient = self.ambient_model.residual_and_jacobian(&ambient_state)?;
        ambient.validate(
            self.ambient_model.residual_dimension(),
            self.ambient_model.state_dimension(),
        )?;
        Ok(NonlinearResidualEvaluation {
            residual: ambient.residual,
            jacobian: sparse_matmul(&ambient.jacobian, &self.transform)?,
        })
    }
}

pub fn hodge_1form_transformed_residual_model<'a>(
    prior: &'a crate::prior::hodge::Hodge1FormDecomposedPrior,
    ambient_model: &'a dyn NonlinearResidualModel,
) -> Result<TransformedResidualModel<'a>, String> {
    TransformedResidualModel::new(
        ambient_model,
        feec_csr_to_core_triplet(&prior.latent_to_ambient),
    )
}

pub fn solve_nonlinear_laplace(
    problem: &NonlinearLaplaceProblem<'_>,
    config: &GaussNewtonConfig,
) -> Result<NonlinearLaplaceResult, String> {
    validate_problem(problem)?;
    validate_config(problem, config)?;

    let timing_enabled = nonlinear_timing_enabled();
    let total_start = Instant::now();
    let prior_precision = sparse_from_core(&problem.prior.precision);
    let prior_mean = GmrfVector::from_vec(problem.prior.mean.clone());
    let prior_information = prior_precision.mul_vec(&prior_mean);
    let mut state = GmrfVector::from_vec(
        config
            .initial_guess
            .clone()
            .unwrap_or_else(|| problem.prior.mean.clone()),
    );
    let mut history = Vec::new();
    let mut converged = false;
    let mut objective_gradient_seconds = 0.0;
    let mut local_assembly_seconds = 0.0;
    let mut step_search_seconds = 0.0;
    let initial_eval_start = Instant::now();
    let mut evaluations = evaluate_residual_terms(problem, state.as_slice())?;
    let initial_eval_seconds = initial_eval_start.elapsed().as_secs_f64();
    let metis_stats_start = metis_ordering_cache_stats_snapshot();
    let mut diagnostics = GaussNewtonRunDiagnostics::default();
    let mut adaptive_lm = AdaptiveLmState::default();

    for iteration in 0..config.max_iterations {
        let objective_gradient_start = Instant::now();
        let objective_value =
            objective(problem, &prior_precision, &prior_mean, &state, &evaluations)?;
        let gradient = gradient(problem, &prior_precision, &prior_mean, &state, &evaluations)?;
        let gradient_norm = gradient.norm();
        objective_gradient_seconds += objective_gradient_start.elapsed().as_secs_f64();
        if gradient_norm <= config.gradient_tolerance {
            converged = true;
            break;
        }

        let local_assembly_start = Instant::now();
        let (local_precision, _local_information, _) = assemble_local_system(
            problem,
            &prior_precision,
            &prior_information,
            &state,
            &evaluations,
        )?;
        let local_precision = symmetrize_precision(&local_precision);
        local_assembly_seconds += local_assembly_start.elapsed().as_secs_f64();
        let step_search_start = Instant::now();
        let step_context = GaussNewtonStepContext {
            problem,
            config,
            prior_precision: &prior_precision,
            prior_mean: &prior_mean,
            state: &state,
            local_precision: &local_precision,
            gradient: &gradient,
            gradient_norm,
            objective_value,
        };
        let step_search =
            find_regularized_gauss_newton_step(&step_context, &mut adaptive_lm, &mut diagnostics)?;
        step_search_seconds += step_search_start.elapsed().as_secs_f64();
        let AcceptedGaussNewtonStep {
            trial_state,
            trial_evaluations,
            trial_objective,
            alpha,
            regularization_lambda,
            step_norm,
            linear_solve,
        } = match step_search {
            GaussNewtonStepSearch::Converged => {
                converged = true;
                break;
            }
            GaussNewtonStepSearch::Accepted(step) => step,
            GaussNewtonStepSearch::Failed(reason) => {
                let current_residual_norm = residual_weighted_norm(problem, &evaluations)?;
                if current_residual_norm <= 1e-8 {
                    converged = true;
                    break;
                }
                return Err(format!(
                    "{reason}; objective={objective_value:.6e}, weighted_residual_norm={current_residual_norm:.6e}, gradient_norm={gradient_norm:.6e}"
                ));
            }
        };
        let residual_norm = residual_weighted_norm(problem, &trial_evaluations)?;
        history.push(GaussNewtonIteration {
            iteration,
            objective: objective_value,
            trial_objective,
            gradient_norm,
            step_norm,
            alpha,
            regularization: config.step_regularization,
            regularization_lambda,
            residual_norm,
            linear_solve,
        });
        state = trial_state;
        evaluations = trial_evaluations;
        if step_norm <= config.step_tolerance {
            converged = true;
            break;
        }
    }

    let final_assembly_start = Instant::now();
    let (posterior_precision, _, mut assembly) = assemble_local_system(
        problem,
        &prior_precision,
        &prior_information,
        &state,
        &evaluations,
    )?;
    let posterior_precision = symmetrize_precision(&posterior_precision);
    assembly.set_posterior_precision_stats(&posterior_precision);
    let final_assembly_seconds = final_assembly_start.elapsed().as_secs_f64();
    let factorization_start = Instant::now();
    let (posterior_precision, posterior_factor) = stabilize_precision_and_factor(
        &posterior_precision,
        config.stabilize_precision,
        "final Laplace precision",
        true,
        false,
        Some(&mut diagnostics),
    )?;
    assembly.set_posterior_precision_stats(&posterior_precision);
    let final_factorization = LaplaceFactorizationStats {
        nnz: posterior_factor.nnz(),
        elapsed_seconds: factorization_start.elapsed().as_secs_f64(),
    };
    diagnostics.final_factorizations += 1;
    assembly.set_factor_nnz(final_factorization.nnz);
    let mut posterior_gmrf =
        Gmrf::from_mean_and_precision(state.clone(), posterior_precision.clone())
            .map_err(|err| err.to_string())?
            .with_precision_sqrt(posterior_factor);
    let latent_variance_start = Instant::now();
    let posterior_variance = if config.estimate_latent_variance {
        estimate_latent_variance(&mut posterior_gmrf, &config.variance)?
            .as_slice()
            .to_vec()
    } else {
        Vec::new()
    };
    let latent_variance_seconds = latent_variance_start.elapsed().as_secs_f64();
    let derived_variance_start = Instant::now();
    let derived_variances = estimate_derived_variances(
        problem,
        &prior_precision,
        &mut posterior_gmrf,
        &config.variance,
    )?;
    let derived_variance_seconds = derived_variance_start.elapsed().as_secs_f64();
    let residual_report_start = Instant::now();
    let final_residuals = residual_reports(problem, &evaluations)?;
    let residual_report_seconds = residual_report_start.elapsed().as_secs_f64();
    diagnostics.accepted_iterations = history.len();
    let metis_stats_end = metis_ordering_cache_stats_snapshot();
    diagnostics.metis_cache_hits = metis_stats_end.hits.saturating_sub(metis_stats_start.hits);
    diagnostics.metis_cache_misses = metis_stats_end
        .misses
        .saturating_sub(metis_stats_start.misses);

    if timing_enabled {
        eprintln!(
            "nonlinear laplace timing: total={:.6}s iterations={} initial_eval={:.6}s objective_gradient={:.6}s local_assembly={:.6}s step_search={:.6}s final_assembly={:.6}s final_factorization={:.6}s latent_variance={:.6}s derived_variance={:.6}s residual_reports={:.6}s cholesky_attempts={} cholesky_successes={} cholesky_cached_shift_attempts={} cholesky_shifted_attempts={} cholesky_factorization={:.6}s",
            total_start.elapsed().as_secs_f64(),
            history.len(),
            initial_eval_seconds,
            objective_gradient_seconds,
            local_assembly_seconds,
            step_search_seconds,
            final_assembly_seconds,
            final_factorization.elapsed_seconds,
            latent_variance_seconds,
            derived_variance_seconds,
            residual_report_seconds,
            diagnostics.cholesky_factor_attempts,
            diagnostics.cholesky_factor_successes,
            diagnostics.cholesky_cached_shift_attempts,
            diagnostics.cholesky_shifted_attempts,
            diagnostics.cholesky_factorization_seconds
        );
    }

    Ok(NonlinearLaplaceResult {
        map: state.as_slice().to_vec(),
        posterior_precision: sparse_to_core(&posterior_precision),
        posterior_gmrf,
        posterior_variance,
        derived_variances,
        final_residuals,
        history,
        assembly,
        final_factorization,
        diagnostics,
        converged,
    })
}

pub fn solve_square_nonlinear_system(
    model: &dyn NonlinearResidualModel,
    config: &SquareNewtonConfig,
) -> Result<SquareNewtonResult, String> {
    validate_square_newton_config(model, config)?;

    let mut state = GmrfVector::from_vec(
        config
            .initial_guess
            .clone()
            .unwrap_or_else(|| vec![0.0; model.state_dimension()]),
    );
    let mut evaluation = model.residual_and_jacobian(state.as_slice())?;
    evaluation.validate(model.residual_dimension(), model.state_dimension())?;
    let mut residual = GmrfVector::from_vec(evaluation.residual.clone());
    let mut residual_norm = residual.norm();
    let mut history = Vec::new();
    let mut converged = residual_norm <= config.residual_tolerance;

    for iteration in 0..config.max_iterations {
        if converged {
            break;
        }
        let jacobian = sparse_from_core(&evaluation.jacobian);
        let rhs = -1.0 * &residual;
        let (direction, linear_solve) = solve_square_newton_linear_system(
            &jacobian,
            &rhs,
            &config.linear_solve,
            config.stabilize_jacobian,
        )?;
        let full_step_norm = direction.norm();
        if full_step_norm <= config.step_tolerance {
            converged = true;
            break;
        }

        let objective_value = 0.5 * residual_norm * residual_norm;
        let directional_derivative = -residual_norm * residual_norm;
        let mut accepted = None;
        let mut alpha = 1.0;
        for _ in 0..=config.max_line_search_steps {
            let trial_state = &state + &(alpha * &direction);
            let trial_residual_values = model.residual(trial_state.as_slice())?;
            if trial_residual_values.len() != model.residual_dimension() {
                return Err(format!(
                    "square Newton residual-only evaluation returned {} rows, expected {}",
                    trial_residual_values.len(),
                    model.residual_dimension()
                ));
            }
            let trial_residual = GmrfVector::from_vec(trial_residual_values);
            let trial_residual_norm = trial_residual.norm();
            let trial_objective = 0.5 * trial_residual_norm * trial_residual_norm;
            if trial_objective
                <= objective_value + config.armijo_c1 * alpha * directional_derivative
            {
                let trial_evaluation = model.residual_and_jacobian(trial_state.as_slice())?;
                trial_evaluation.validate(model.residual_dimension(), model.state_dimension())?;
                accepted = Some((
                    trial_state,
                    trial_evaluation,
                    trial_residual,
                    trial_residual_norm,
                    alpha,
                ));
                break;
            }
            alpha *= 0.5;
        }
        let Some((trial_state, trial_evaluation, trial_residual, trial_residual_norm, alpha)) =
            accepted
        else {
            return Err(format!(
                "square Newton line search failed after {} halvings from residual norm {:.6e}",
                config.max_line_search_steps, residual_norm
            ));
        };
        history.push(SquareNewtonIteration {
            iteration,
            residual_norm,
            trial_residual_norm,
            step_norm: alpha * full_step_norm,
            alpha,
            linear_solve,
        });
        state = trial_state;
        evaluation = trial_evaluation;
        residual = trial_residual;
        residual_norm = trial_residual_norm;
        converged = residual_norm <= config.residual_tolerance
            || alpha * full_step_norm <= config.step_tolerance;
    }

    if !converged {
        evaluation = model.residual_and_jacobian(state.as_slice())?;
        evaluation.validate(model.residual_dimension(), model.state_dimension())?;
        residual = GmrfVector::from_vec(evaluation.residual.clone());
        residual_norm = residual.norm();
        converged = residual_norm <= config.residual_tolerance;
    }

    Ok(SquareNewtonResult {
        solution: state.as_slice().to_vec(),
        residual: residual.as_slice().to_vec(),
        residual_norm,
        history,
        converged,
    })
}

pub fn diagnose_gauss_newton_first_step(
    problem: &NonlinearLaplaceProblem<'_>,
    config: &GaussNewtonConfig,
) -> Result<GaussNewtonFirstStepDiagnostics, String> {
    validate_problem(problem)?;
    validate_config(problem, config)?;

    let prior_precision = sparse_from_core(&problem.prior.precision);
    let prior_mean = GmrfVector::from_vec(problem.prior.mean.clone());
    let prior_information = prior_precision.mul_vec(&prior_mean);
    let state = GmrfVector::from_vec(
        config
            .initial_guess
            .clone()
            .unwrap_or_else(|| problem.prior.mean.clone()),
    );
    let evaluations = evaluate_residual_terms(problem, state.as_slice())?;
    let objective_value = objective(problem, &prior_precision, &prior_mean, &state, &evaluations)?;
    let weighted_residual_norm = residual_weighted_norm(problem, &evaluations)?;
    let gradient = gradient(problem, &prior_precision, &prior_mean, &state, &evaluations)?;
    let gradient_norm = gradient.norm();

    let (local_precision, _, mut assembly) = assemble_local_system(
        problem,
        &prior_precision,
        &prior_information,
        &state,
        &evaluations,
    )?;
    let local_precision = symmetrize_precision(&local_precision);
    assembly.set_posterior_precision_stats(&local_precision);
    let step_context = GaussNewtonStepContext {
        problem,
        config,
        prior_precision: &prior_precision,
        prior_mean: &prior_mean,
        state: &state,
        local_precision: &local_precision,
        gradient: &gradient,
        gradient_norm,
        objective_value,
    };
    let candidate = diagnose_regularized_gauss_newton_step(&step_context)?;

    Ok(GaussNewtonFirstStepDiagnostics {
        objective: objective_value,
        weighted_residual_norm,
        gradient_norm,
        step_norm: candidate.step_norm,
        directional_derivative: candidate.directional_derivative,
        accepted_alpha: candidate.accepted_alpha,
        accepted_objective: candidate.accepted_objective,
        regularization: config.step_regularization,
        regularization_lambda: candidate.regularization_lambda,
        linear_solve: candidate.linear_solve,
        linear_solve_absolute_residual_norm: candidate.linear_solve_absolute_residual_norm,
        linear_solve_relative_residual_norm: candidate.linear_solve_relative_residual_norm,
        assembly,
        objective_grid: candidate.objective_grid,
    })
}

const LEVENBERG_MARQUARDT_LAMBDAS: [f64; 7] = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2];
const NO_STEP_REGULARIZATION_LAMBDAS: [f64; 1] = [0.0];

struct AcceptedGaussNewtonStep {
    trial_state: GmrfVector,
    trial_evaluations: Vec<ResidualTermEvaluation>,
    trial_objective: f64,
    alpha: f64,
    regularization_lambda: f64,
    step_norm: f64,
    linear_solve: GaussNewtonLinearSolveStats,
}

#[derive(Debug, Clone, Default)]
struct AdaptiveLmState {
    next_lambda_index: usize,
}

enum GaussNewtonStepSearch {
    Accepted(AcceptedGaussNewtonStep),
    Converged,
    Failed(String),
}

struct DiagnosticGaussNewtonStep {
    regularization_lambda: f64,
    step_norm: f64,
    directional_derivative: f64,
    accepted_alpha: Option<f64>,
    accepted_objective: Option<f64>,
    linear_solve: GaussNewtonLinearSolveStats,
    linear_solve_absolute_residual_norm: f64,
    linear_solve_relative_residual_norm: f64,
    objective_grid: Vec<GaussNewtonObjectiveSample>,
}

struct GaussNewtonStepContext<'a, 'problem> {
    problem: &'a NonlinearLaplaceProblem<'problem>,
    config: &'a GaussNewtonConfig,
    prior_precision: &'a GmrfSparseMatrix,
    prior_mean: &'a GmrfVector,
    state: &'a GmrfVector,
    local_precision: &'a GmrfSparseMatrix,
    gradient: &'a GmrfVector,
    gradient_norm: f64,
    objective_value: f64,
}

fn find_regularized_gauss_newton_step(
    context: &GaussNewtonStepContext<'_, '_>,
    adaptive_lm: &mut AdaptiveLmState,
    diagnostics: &mut GaussNewtonRunDiagnostics,
) -> Result<GaussNewtonStepSearch, String> {
    let problem = context.problem;
    let config = context.config;
    let prior_precision = context.prior_precision;
    let prior_mean = context.prior_mean;
    let state = context.state;
    let local_precision = context.local_precision;
    let gradient = context.gradient;
    let gradient_norm = context.gradient_norm;
    let objective_value = context.objective_value;
    let mut last_failure = None;
    for lambda_index in step_regularization_lambda_indices(
        config.step_regularization,
        adaptive_lm.next_lambda_index,
    ) {
        let lambda = LEVENBERG_MARQUARDT_LAMBDAS[lambda_index];
        let step_precision = regularized_step_precision(local_precision, lambda)?;
        let local_information = step_precision.mul_vec(state) - gradient;
        diagnostics.step_solve_attempts += 1;
        let (linearized_mode, linear_solve) = match solve_gauss_newton_linear_system(
            &step_precision,
            &local_information,
            state,
            &config.linear_solve,
            GaussNewtonLinearSolveOptions {
                stabilize_precision: config.stabilize_precision,
                assume_symmetric: true,
                prefer_cached_stabilization_shift: config.reuse_cholesky_stabilization_shift,
            },
            Some(&mut *diagnostics),
        ) {
            Ok(solve) => solve,
            Err(err) => {
                if config.step_regularization == GaussNewtonStepRegularization::None {
                    return Err(err);
                }
                last_failure = Some(format!(
                    "regularized Gauss-Newton solve failed for lambda={lambda:.3e}: {err}"
                ));
                continue;
            }
        };
        let direction = &linearized_mode - state;
        let step_norm = direction.norm();
        if !direction.iter().all(|value| value.is_finite()) {
            let reason =
                format!("Gauss-Newton step contained non-finite values for lambda={lambda:.3e}");
            if config.step_regularization == GaussNewtonStepRegularization::None {
                return Err(reason);
            }
            last_failure = Some(reason);
            continue;
        }
        if step_norm <= config.step_tolerance {
            return Ok(GaussNewtonStepSearch::Converged);
        }

        let directional_derivative = gradient.dot(&direction);
        let descent_tolerance = descent_tolerance(gradient_norm, step_norm);
        if directional_derivative >= 0.0 && directional_derivative > descent_tolerance {
            let reason = format!(
                "Gauss-Newton direction is not a descent direction for lambda={lambda:.3e}: g^T p = {directional_derivative:.6e}, ||g|| = {gradient_norm:.6e}, ||p|| = {step_norm:.6e}"
            );
            if config.step_regularization == GaussNewtonStepRegularization::None {
                return Err(reason);
            }
            last_failure = Some(reason);
            continue;
        }
        let armijo_directional_derivative = if directional_derivative < 0.0 {
            directional_derivative
        } else {
            -descent_tolerance
        };

        let mut alpha = 1.0;
        for _ in 0..=config.max_line_search_steps {
            let trial_state = state + &(alpha * &direction);
            diagnostics.line_search_residual_evaluations += 1;
            let trial_residuals =
                evaluate_residual_terms_residual_only(problem, trial_state.as_slice())?;
            let trial_objective = objective_from_residuals(
                problem,
                prior_precision,
                prior_mean,
                &trial_state,
                &trial_residuals,
            )?;
            if trial_objective
                <= objective_value + config.armijo_c1 * alpha * armijo_directional_derivative
            {
                let trial_evaluations = evaluate_residual_terms(problem, trial_state.as_slice())?;
                update_adaptive_lm_state(
                    config.step_regularization,
                    adaptive_lm,
                    lambda_index,
                    alpha,
                );
                return Ok(GaussNewtonStepSearch::Accepted(AcceptedGaussNewtonStep {
                    trial_state,
                    trial_evaluations,
                    trial_objective,
                    alpha,
                    regularization_lambda: lambda,
                    step_norm: alpha * step_norm,
                    linear_solve,
                }));
            }
            alpha *= 0.5;
        }
        last_failure = Some(format!(
            "line search failed after {} halvings for lambda={lambda:.3e}",
            config.max_line_search_steps
        ));
    }
    Ok(GaussNewtonStepSearch::Failed(last_failure.unwrap_or_else(
        || "Gauss-Newton step search found no candidate".to_string(),
    )))
}

fn diagnose_regularized_gauss_newton_step(
    context: &GaussNewtonStepContext<'_, '_>,
) -> Result<DiagnosticGaussNewtonStep, String> {
    let problem = context.problem;
    let config = context.config;
    let prior_precision = context.prior_precision;
    let prior_mean = context.prior_mean;
    let state = context.state;
    let local_precision = context.local_precision;
    let gradient = context.gradient;
    let gradient_norm = context.gradient_norm;
    let objective_value = context.objective_value;
    let mut fallback = None;
    for &lambda in step_regularization_lambdas(config.step_regularization) {
        let step_precision = regularized_step_precision(local_precision, lambda)?;
        let local_information = step_precision.mul_vec(state) - gradient;
        let (linearized_mode, linear_solve) = match solve_gauss_newton_linear_system(
            &step_precision,
            &local_information,
            state,
            &config.linear_solve,
            GaussNewtonLinearSolveOptions {
                stabilize_precision: config.stabilize_precision,
                assume_symmetric: true,
                prefer_cached_stabilization_shift: false,
            },
            None,
        ) {
            Ok(solve) => solve,
            Err(err) => {
                if config.step_regularization == GaussNewtonStepRegularization::None {
                    return Err(err);
                }
                continue;
            }
        };
        let direction = &linearized_mode - state;
        let step_norm = direction.norm();
        let directional_derivative = gradient.dot(&direction);
        let linear_residual = &local_information - &step_precision.mul_vec(&linearized_mode);
        let linear_solve_absolute_residual_norm = linear_residual.norm();
        let linear_solve_relative_residual_norm =
            linear_solve_absolute_residual_norm / local_information.norm().max(1.0);

        let descent_tol = descent_tolerance(gradient_norm, step_norm);
        let armijo_directional_derivative = if directional_derivative < 0.0 {
            directional_derivative
        } else {
            -descent_tol
        };
        let mut objective_grid = Vec::with_capacity(21);
        let mut accepted_alpha = None;
        let mut accepted_objective = None;
        for exponent in 0..=20 {
            let alpha = 0.5_f64.powi(exponent as i32);
            let trial_state = state + &(alpha * &direction);
            let trial_residuals =
                evaluate_residual_terms_residual_only(problem, trial_state.as_slice())?;
            let trial_objective = objective_from_residuals(
                problem,
                prior_precision,
                prior_mean,
                &trial_state,
                &trial_residuals,
            )?;
            if accepted_alpha.is_none()
                && exponent <= config.max_line_search_steps
                && trial_objective
                    <= objective_value + config.armijo_c1 * alpha * armijo_directional_derivative
            {
                accepted_alpha = Some(alpha);
                accepted_objective = Some(trial_objective);
            }
            objective_grid.push(GaussNewtonObjectiveSample {
                alpha,
                objective: trial_objective,
            });
        }

        let candidate = DiagnosticGaussNewtonStep {
            regularization_lambda: lambda,
            step_norm,
            directional_derivative,
            accepted_alpha,
            accepted_objective,
            linear_solve,
            linear_solve_absolute_residual_norm,
            linear_solve_relative_residual_norm,
            objective_grid,
        };
        let is_descent = directional_derivative < 0.0 || directional_derivative <= descent_tol;
        if is_descent && candidate.accepted_alpha.is_some() {
            return Ok(candidate);
        }
        fallback = Some(candidate);
        if config.step_regularization == GaussNewtonStepRegularization::None {
            break;
        }
    }
    fallback.ok_or_else(|| "failed to compute Gauss-Newton first-step diagnostics".to_string())
}

#[derive(Debug, Clone, Copy)]
struct GaussNewtonLinearSolveOptions {
    stabilize_precision: bool,
    assume_symmetric: bool,
    prefer_cached_stabilization_shift: bool,
}

fn solve_gauss_newton_linear_system(
    precision: &GmrfSparseMatrix,
    information: &GmrfVector,
    current_state: &GmrfVector,
    linear_solve: &GaussNewtonLinearSolve,
    options: GaussNewtonLinearSolveOptions,
    diagnostics: Option<&mut GaussNewtonRunDiagnostics>,
) -> Result<(GmrfVector, GaussNewtonLinearSolveStats), String> {
    match *linear_solve {
        GaussNewtonLinearSolve::DirectCholesky => {
            let (precision, factor) = stabilize_precision_and_factor(
                precision,
                options.stabilize_precision,
                "Gauss-Newton precision",
                options.assume_symmetric,
                options.prefer_cached_stabilization_shift,
                diagnostics,
            )?;
            let solution = factor
                .solve(information)
                .map_err(|err| format!("failed to solve Gauss-Newton step: {err}"))?;
            let residual_norm = (information - &precision.mul_vec(&solution)).norm();
            Ok((
                solution,
                GaussNewtonLinearSolveStats {
                    mode: GaussNewtonLinearSolveMode::DirectCholesky,
                    iterations: 0,
                    final_residual_norm: residual_norm,
                    converged: true,
                    factor_nnz: Some(factor.nnz()),
                },
            ))
        }
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            warm_start,
        } => {
            if !tolerance.is_finite() || tolerance <= 0.0 {
                return Err("CG tolerance must be finite and positive".to_string());
            }
            if max_iterations == 0 {
                return Err("CG max_iterations must be at least one".to_string());
            }
            let mut solver = Solver::new(SolverConfig {
                algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
                tolerance,
                max_iterations,
                preconditioner: PreconditionerKind::Jacobi,
            });
            let initial_guess = warm_start.then_some(current_state);
            let report = solver
                .solve_matrix_with_initial_guess(precision, information, initial_guess)
                .map_err(|err| format!("failed to solve Gauss-Newton step with CG: {err}"))?;
            Ok((
                report.solution,
                GaussNewtonLinearSolveStats {
                    mode: GaussNewtonLinearSolveMode::IterativeCg,
                    iterations: report.iterations,
                    final_residual_norm: report.final_residual_norm,
                    converged: report.converged,
                    factor_nnz: None,
                },
            ))
        }
    }
}

fn solve_square_newton_linear_system(
    jacobian: &GmrfSparseMatrix,
    rhs: &GmrfVector,
    linear_solve: &GaussNewtonLinearSolve,
    stabilize_jacobian: bool,
) -> Result<(GmrfVector, GaussNewtonLinearSolveStats), String> {
    match *linear_solve {
        GaussNewtonLinearSolve::DirectCholesky => {
            let (matrix, factor) = stabilize_precision_and_factor(
                jacobian,
                stabilize_jacobian,
                "square Newton Jacobian",
                false,
                false,
                None,
            )?;
            let solution = factor
                .solve(rhs)
                .map_err(|err| format!("failed to solve square Newton step: {err}"))?;
            let residual_norm = (rhs - &matrix.mul_vec(&solution)).norm();
            Ok((
                solution,
                GaussNewtonLinearSolveStats {
                    mode: GaussNewtonLinearSolveMode::DirectCholesky,
                    iterations: 0,
                    final_residual_norm: residual_norm,
                    converged: true,
                    factor_nnz: Some(factor.nnz()),
                },
            ))
        }
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            ..
        } => {
            if !tolerance.is_finite() || tolerance <= 0.0 {
                return Err("CG tolerance must be finite and positive".to_string());
            }
            if max_iterations == 0 {
                return Err("CG max_iterations must be at least one".to_string());
            }
            let matrix = if stabilize_jacobian {
                maybe_stabilize_precision(jacobian, true)?
            } else {
                jacobian.clone()
            };
            let mut solver = Solver::new(SolverConfig {
                algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
                tolerance,
                max_iterations,
                preconditioner: PreconditionerKind::Jacobi,
            });
            let report = solver
                .solve_matrix_with_initial_guess(&matrix, rhs, None)
                .map_err(|err| format!("failed to solve square Newton step with CG: {err}"))?;
            Ok((
                report.solution,
                GaussNewtonLinearSolveStats {
                    mode: GaussNewtonLinearSolveMode::IterativeCg,
                    iterations: report.iterations,
                    final_residual_norm: report.final_residual_norm,
                    converged: report.converged,
                    factor_nnz: None,
                },
            ))
        }
    }
}

fn step_regularization_lambdas(regularization: GaussNewtonStepRegularization) -> &'static [f64] {
    match regularization {
        GaussNewtonStepRegularization::None => &NO_STEP_REGULARIZATION_LAMBDAS,
        GaussNewtonStepRegularization::LevenbergMarquardtGrid => &LEVENBERG_MARQUARDT_LAMBDAS,
        GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt => &LEVENBERG_MARQUARDT_LAMBDAS,
    }
}

fn step_regularization_lambda_indices(
    regularization: GaussNewtonStepRegularization,
    adaptive_start: usize,
) -> Vec<usize> {
    match regularization {
        GaussNewtonStepRegularization::None => vec![0],
        GaussNewtonStepRegularization::LevenbergMarquardtGrid => {
            (0..LEVENBERG_MARQUARDT_LAMBDAS.len()).collect()
        }
        GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt => {
            let start = adaptive_start.min(LEVENBERG_MARQUARDT_LAMBDAS.len() - 1);
            (start..LEVENBERG_MARQUARDT_LAMBDAS.len()).collect()
        }
    }
}

fn update_adaptive_lm_state(
    regularization: GaussNewtonStepRegularization,
    adaptive_lm: &mut AdaptiveLmState,
    accepted_lambda_index: usize,
    alpha: f64,
) {
    if regularization != GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt {
        return;
    }
    let last = LEVENBERG_MARQUARDT_LAMBDAS.len() - 1;
    adaptive_lm.next_lambda_index = if alpha >= 0.5 {
        accepted_lambda_index.saturating_sub(1)
    } else if alpha < 0.25 {
        (accepted_lambda_index + 1).min(last)
    } else {
        accepted_lambda_index.min(last)
    };
}

fn regularized_step_precision(
    precision: &GmrfSparseMatrix,
    lambda: f64,
) -> Result<GmrfSparseMatrix, String> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err("Levenberg-Marquardt lambda must be finite and nonnegative".to_string());
    }
    if lambda == 0.0 {
        return Ok(precision.clone());
    }
    add_levenberg_marquardt_diagonal(precision, lambda)
}

fn descent_tolerance(gradient_norm: f64, step_norm: f64) -> f64 {
    (1e-5 * gradient_norm * step_norm)
        .max(f64::EPSILON.sqrt() * gradient_norm * step_norm)
        .max(1e-30)
}

struct ResidualTermEvaluation {
    residual: GmrfVector,
    jacobian: GmrfSparseMatrix,
}

enum OwnedNoise {
    ScalarVariance(f64),
    Precision(GmrfSparseMatrix),
}

struct OwnedObservationTerm {
    name: String,
    kind: NonlinearAssemblyTermKind,
    matrix: GmrfSparseMatrix,
    observations: GmrfVector,
    bias: GmrfVector,
    noise: OwnedNoise,
}

impl OwnedObservationTerm {
    fn as_linear_term(&self) -> LinearObservationTerm<'_> {
        match &self.noise {
            OwnedNoise::ScalarVariance(variance) => LinearObservationTerm::scalar_variance(
                &self.matrix,
                &self.observations,
                Some(&self.bias),
                *variance,
            ),
            OwnedNoise::Precision(precision) => LinearObservationTerm::precision(
                &self.matrix,
                &self.observations,
                Some(&self.bias),
                precision,
            ),
        }
    }
}

fn validate_problem(problem: &NonlinearLaplaceProblem<'_>) -> Result<(), String> {
    problem.prior.validate()?;
    let state_dimension = problem.prior.dimension();
    for term in &problem.residual_terms {
        if term.model.state_dimension() != state_dimension {
            return Err(format!(
                "nonlinear residual `{}` state dimension {} must match prior dimension {}",
                term.name,
                term.model.state_dimension(),
                state_dimension
            ));
        }
        if term.observations.len() != term.model.residual_dimension() {
            return Err(format!(
                "nonlinear residual `{}` observation length {} must match residual dimension {}",
                term.name,
                term.observations.len(),
                term.model.residual_dimension()
            ));
        }
        validate_noise(&term.name, term.model.residual_dimension(), &term.noise)?;
    }
    for measurement in &problem.linear_measurements {
        measurement.validate(state_dimension)?;
    }
    for measurement in &problem.precision_weighted_measurements {
        measurement.validate(state_dimension)?;
    }
    for quantity in &problem.derived_quantities {
        if quantity.operator.ncols != state_dimension {
            return Err(format!(
                "derived quantity `{}` column count {} must match state dimension {}",
                quantity.name, quantity.operator.ncols, state_dimension
            ));
        }
    }
    Ok(())
}

fn validate_config(
    problem: &NonlinearLaplaceProblem<'_>,
    config: &GaussNewtonConfig,
) -> Result<(), String> {
    if let Some(initial_guess) = &config.initial_guess {
        if initial_guess.len() != problem.prior.dimension() {
            return Err(format!(
                "initial guess length {} must match prior dimension {}",
                initial_guess.len(),
                problem.prior.dimension()
            ));
        }
    }
    if config.max_iterations == 0 {
        return Err("max_iterations must be >= 1".to_string());
    }
    if !config.step_tolerance.is_finite() || config.step_tolerance <= 0.0 {
        return Err("step_tolerance must be finite and positive".to_string());
    }
    if !config.gradient_tolerance.is_finite() || config.gradient_tolerance <= 0.0 {
        return Err("gradient_tolerance must be finite and positive".to_string());
    }
    if !config.armijo_c1.is_finite() || config.armijo_c1 <= 0.0 || config.armijo_c1 >= 1.0 {
        return Err("armijo_c1 must be finite and in (0, 1)".to_string());
    }
    match config.linear_solve {
        GaussNewtonLinearSolve::DirectCholesky => {}
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            ..
        } => {
            if !tolerance.is_finite() || tolerance <= 0.0 {
                return Err("linear_solve CG tolerance must be finite and positive".to_string());
            }
            if max_iterations == 0 {
                return Err("linear_solve CG max_iterations must be >= 1".to_string());
            }
        }
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
            Ok(())
        }
    }
}

fn validate_square_newton_config(
    model: &dyn NonlinearResidualModel,
    config: &SquareNewtonConfig,
) -> Result<(), String> {
    if model.state_dimension() == 0 {
        return Err("square Newton model state dimension must be nonzero".to_string());
    }
    if model.state_dimension() != model.residual_dimension() {
        return Err(format!(
            "square Newton requires residual dimension {} to match state dimension {}",
            model.residual_dimension(),
            model.state_dimension()
        ));
    }
    if let Some(initial_guess) = &config.initial_guess {
        if initial_guess.len() != model.state_dimension() {
            return Err(format!(
                "initial guess length {} must match square Newton state dimension {}",
                initial_guess.len(),
                model.state_dimension()
            ));
        }
        if initial_guess.iter().any(|value| !value.is_finite()) {
            return Err("square Newton initial guess must contain only finite values".to_string());
        }
    }
    if config.max_iterations == 0 {
        return Err("square Newton max_iterations must be >= 1".to_string());
    }
    if !config.residual_tolerance.is_finite() || config.residual_tolerance <= 0.0 {
        return Err("square Newton residual_tolerance must be finite and positive".to_string());
    }
    if !config.step_tolerance.is_finite() || config.step_tolerance <= 0.0 {
        return Err("square Newton step_tolerance must be finite and positive".to_string());
    }
    if !config.armijo_c1.is_finite() || config.armijo_c1 <= 0.0 || config.armijo_c1 >= 1.0 {
        return Err("square Newton armijo_c1 must be finite and in (0, 1)".to_string());
    }
    if config.max_line_search_steps == 0 {
        return Err("square Newton max_line_search_steps must be >= 1".to_string());
    }
    match config.linear_solve {
        GaussNewtonLinearSolve::DirectCholesky => {}
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            ..
        } => {
            if !tolerance.is_finite() || tolerance <= 0.0 {
                return Err("square Newton CG tolerance must be finite and positive".to_string());
            }
            if max_iterations == 0 {
                return Err("square Newton CG max_iterations must be >= 1".to_string());
            }
        }
    }
    Ok(())
}

fn validate_noise(name: &str, dimension: usize, noise: &GaussianNoiseModel) -> Result<(), String> {
    match noise {
        GaussianNoiseModel::ScalarVariance(variance) => {
            if !variance.is_finite() || *variance <= 0.0 {
                return Err(format!(
                    "Gaussian noise variance for `{name}` must be finite and positive"
                ));
            }
        }
        GaussianNoiseModel::Precision(precision) => {
            if precision.nrows() != dimension || precision.ncols() != dimension {
                return Err(format!(
                    "Gaussian noise precision for `{name}` must be {dimension}x{dimension}, got {}x{}",
                    precision.nrows(),
                    precision.ncols()
                ));
            }
        }
    }
    Ok(())
}

fn evaluate_residual_terms(
    problem: &NonlinearLaplaceProblem<'_>,
    state: &[f64],
) -> Result<Vec<ResidualTermEvaluation>, String> {
    problem
        .residual_terms
        .iter()
        .map(|term| {
            let evaluation = term.model.residual_and_jacobian(state)?;
            evaluation.validate(
                term.model.residual_dimension(),
                term.model.state_dimension(),
            )?;
            Ok(ResidualTermEvaluation {
                residual: GmrfVector::from_vec(evaluation.residual),
                jacobian: sparse_from_core(&evaluation.jacobian),
            })
        })
        .collect()
}

fn evaluate_residual_terms_residual_only(
    problem: &NonlinearLaplaceProblem<'_>,
    state: &[f64],
) -> Result<Vec<GmrfVector>, String> {
    problem
        .residual_terms
        .iter()
        .map(|term| {
            let residual = term.model.residual(state)?;
            if residual.len() != term.model.residual_dimension() {
                return Err(format!(
                    "residual-only evaluation for `{}` returned {} rows, expected {}",
                    term.name,
                    residual.len(),
                    term.model.residual_dimension()
                ));
            }
            Ok(GmrfVector::from_vec(residual))
        })
        .collect()
}

fn assemble_local_system(
    problem: &NonlinearLaplaceProblem<'_>,
    prior_precision: &GmrfSparseMatrix,
    prior_information: &GmrfVector,
    state: &GmrfVector,
    evaluations: &[ResidualTermEvaluation],
) -> Result<(GmrfSparseMatrix, GmrfVector, NonlinearAssemblyStats), String> {
    let owned_terms = owned_observation_terms(problem, state, evaluations)?;
    let terms = owned_terms
        .iter()
        .map(OwnedObservationTerm::as_linear_term)
        .collect::<Vec<_>>();
    let (posterior_precision, observation_information, conditioning_stats) =
        apply_linear_observation_terms_with_stats(prior_precision, &terms);
    let term_stats = owned_terms
        .iter()
        .zip(conditioning_stats.terms)
        .map(|(term, stats)| nonlinear_term_stats(term, stats))
        .collect::<Vec<_>>();
    let stats = NonlinearAssemblyStats {
        dimension: prior_precision.nrows(),
        prior_precision_nnz: conditioning_stats.prior_precision_nnz,
        prior_precision_lower_triangle_nnz: lower_triangle_nnz(prior_precision),
        terms: term_stats,
        posterior_precision_nnz: conditioning_stats.posterior_precision_nnz,
        posterior_precision_lower_triangle_nnz: lower_triangle_nnz(&posterior_precision),
        factor_nnz: None,
        fill_ratio_vs_lower_triangle: None,
    };
    Ok((
        posterior_precision,
        prior_information + &observation_information,
        stats,
    ))
}

fn owned_observation_terms(
    problem: &NonlinearLaplaceProblem<'_>,
    state: &GmrfVector,
    evaluations: &[ResidualTermEvaluation],
) -> Result<Vec<OwnedObservationTerm>, String> {
    let mut owned_terms = Vec::new();
    for measurement in &problem.linear_measurements {
        owned_terms.push(OwnedObservationTerm {
            name: measurement.name.clone(),
            kind: NonlinearAssemblyTermKind::LinearMeasurement,
            matrix: sparse_from_core(&measurement.operator),
            observations: GmrfVector::from_vec(measurement.observations.clone()),
            bias: GmrfVector::from_vec(measurement.bias.clone()),
            noise: OwnedNoise::ScalarVariance(measurement.variance),
        });
    }
    for measurement in &problem.precision_weighted_measurements {
        owned_terms.push(OwnedObservationTerm {
            name: measurement.name.clone(),
            kind: NonlinearAssemblyTermKind::PrecisionWeightedMeasurement,
            matrix: sparse_from_core(&measurement.operator),
            observations: GmrfVector::from_vec(measurement.observations.clone()),
            bias: GmrfVector::from_vec(measurement.bias.clone()),
            noise: OwnedNoise::Precision(sparse_from_core(&measurement.precision)),
        });
    }
    for (term, evaluation) in problem.residual_terms.iter().zip(evaluations.iter()) {
        let linear_prediction = evaluation.jacobian.mul_vec(state);
        let bias = &evaluation.residual - &linear_prediction;
        owned_terms.push(OwnedObservationTerm {
            name: term.name.clone(),
            kind: NonlinearAssemblyTermKind::Residual,
            matrix: evaluation.jacobian.clone(),
            observations: GmrfVector::from_vec(term.observations.clone()),
            bias,
            noise: match &term.noise {
                GaussianNoiseModel::ScalarVariance(variance) => {
                    OwnedNoise::ScalarVariance(*variance)
                }
                GaussianNoiseModel::Precision(precision) => {
                    OwnedNoise::Precision(sparse_from_core(precision))
                }
            },
        });
    }
    Ok(owned_terms)
}

fn nonlinear_term_stats(
    term: &OwnedObservationTerm,
    stats: LinearObservationUpdateStats,
) -> NonlinearAssemblyTermStats {
    NonlinearAssemblyTermStats {
        name: term.name.clone(),
        kind: term.kind,
        operator_rows: stats.operator_rows,
        operator_cols: stats.operator_cols,
        operator_nnz: stats.operator_nnz,
        precision_update_nnz: stats.precision_update_nnz,
    }
}

fn objective(
    problem: &NonlinearLaplaceProblem<'_>,
    prior_precision: &GmrfSparseMatrix,
    prior_mean: &GmrfVector,
    state: &GmrfVector,
    evaluations: &[ResidualTermEvaluation],
) -> Result<f64, String> {
    let residuals = evaluations
        .iter()
        .map(|evaluation| evaluation.residual.clone())
        .collect::<Vec<_>>();
    objective_from_residuals(problem, prior_precision, prior_mean, state, &residuals)
}

fn objective_from_residuals(
    problem: &NonlinearLaplaceProblem<'_>,
    prior_precision: &GmrfSparseMatrix,
    prior_mean: &GmrfVector,
    state: &GmrfVector,
    residuals: &[GmrfVector],
) -> Result<f64, String> {
    let diff = state - prior_mean;
    let prior_weighted = prior_precision.mul_vec(&diff);
    let mut value = 0.5 * diff.dot(&prior_weighted);

    for measurement in &problem.linear_measurements {
        let operator = sparse_from_core(&measurement.operator);
        let prediction = operator.mul_vec(state) + GmrfVector::from_vec(measurement.bias.clone());
        let residual = prediction - GmrfVector::from_vec(measurement.observations.clone());
        value += 0.5 * residual.dot(&residual) / measurement.variance;
    }
    for measurement in &problem.precision_weighted_measurements {
        let operator = sparse_from_core(&measurement.operator);
        let prediction = operator.mul_vec(state) + GmrfVector::from_vec(measurement.bias.clone());
        let residual = prediction - GmrfVector::from_vec(measurement.observations.clone());
        let precision = sparse_from_core(&measurement.precision);
        value += 0.5 * residual.dot(&precision.mul_vec(&residual));
    }
    for (term, residual) in problem.residual_terms.iter().zip(residuals.iter()) {
        let residual = residual - &GmrfVector::from_vec(term.observations.clone());
        value += noise_quadratic(&residual, &term.noise)?;
    }
    Ok(value)
}

fn gradient(
    problem: &NonlinearLaplaceProblem<'_>,
    prior_precision: &GmrfSparseMatrix,
    prior_mean: &GmrfVector,
    state: &GmrfVector,
    evaluations: &[ResidualTermEvaluation],
) -> Result<GmrfVector, String> {
    let diff = state - prior_mean;
    let mut gradient = prior_precision.mul_vec(&diff);

    for measurement in &problem.linear_measurements {
        let operator = sparse_from_core(&measurement.operator);
        let prediction = operator.mul_vec(state) + GmrfVector::from_vec(measurement.bias.clone());
        let residual = prediction - GmrfVector::from_vec(measurement.observations.clone());
        gradient +=
            gmrf_core::ht_weighted_observations(&operator, &residual, 1.0 / measurement.variance);
    }
    for measurement in &problem.precision_weighted_measurements {
        let operator = sparse_from_core(&measurement.operator);
        let prediction = operator.mul_vec(state) + GmrfVector::from_vec(measurement.bias.clone());
        let residual = prediction - GmrfVector::from_vec(measurement.observations.clone());
        let precision = sparse_from_core(&measurement.precision);
        gradient += gmrf_core::ht_precision_weighted_observations(&operator, &residual, &precision);
    }
    for (term, evaluation) in problem.residual_terms.iter().zip(evaluations.iter()) {
        let residual = &evaluation.residual - &GmrfVector::from_vec(term.observations.clone());
        gradient += match &term.noise {
            GaussianNoiseModel::ScalarVariance(variance) => {
                gmrf_core::ht_weighted_observations(&evaluation.jacobian, &residual, 1.0 / variance)
            }
            GaussianNoiseModel::Precision(precision) => {
                let precision = sparse_from_core(precision);
                gmrf_core::ht_precision_weighted_observations(
                    &evaluation.jacobian,
                    &residual,
                    &precision,
                )
            }
        };
    }
    Ok(gradient)
}

fn residual_weighted_norm(
    problem: &NonlinearLaplaceProblem<'_>,
    evaluations: &[ResidualTermEvaluation],
) -> Result<f64, String> {
    let mut squared = 0.0;
    for (term, evaluation) in problem.residual_terms.iter().zip(evaluations.iter()) {
        let residual = &evaluation.residual - &GmrfVector::from_vec(term.observations.clone());
        squared += 2.0 * noise_quadratic(&residual, &term.noise)?;
    }
    Ok(squared.sqrt())
}

fn residual_reports(
    problem: &NonlinearLaplaceProblem<'_>,
    evaluations: &[ResidualTermEvaluation],
) -> Result<Vec<NonlinearResidualReport>, String> {
    problem
        .residual_terms
        .iter()
        .zip(evaluations.iter())
        .map(|(term, evaluation)| {
            let residual = &evaluation.residual - &GmrfVector::from_vec(term.observations.clone());
            Ok(NonlinearResidualReport {
                name: term.name.clone(),
                weighted_norm: (2.0 * noise_quadratic(&residual, &term.noise)?).sqrt(),
                residual: residual.as_slice().to_vec(),
            })
        })
        .collect()
}

fn noise_quadratic(residual: &GmrfVector, noise: &GaussianNoiseModel) -> Result<f64, String> {
    match noise {
        GaussianNoiseModel::ScalarVariance(variance) => Ok(0.5 * residual.dot(residual) / variance),
        GaussianNoiseModel::Precision(precision) => {
            let precision = sparse_from_core(precision);
            Ok(0.5 * residual.dot(&precision.mul_vec(residual)))
        }
    }
}

fn estimate_latent_variance(
    posterior: &mut Gmrf,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    match config.mode {
        LinearPdeVarianceMode::Exact | LinearPdeVarianceMode::ExactSolves => {
            let factor = posterior
                .precision_factor()
                .ok_or_else(|| "posterior precision factor is missing".to_string())?;
            exact_solve_diag(factor)
                .map(|estimate| estimate.values)
                .map_err(|err| err.to_string())
        }
        LinearPdeVarianceMode::SelectedInverse => {
            let factor = posterior
                .precision_factor()
                .ok_or_else(|| "posterior precision factor is missing".to_string())?;
            selected_inverse_diag(factor)
                .map(|estimate| estimate.values)
                .map_err(|err| err.to_string())
        }
        LinearPdeVarianceMode::MonteCarlo => {
            let mut rng = StdRng::seed_from_u64(config.rng_seed);
            posterior
                .mc_variances(config.num_variance_probes, &mut rng)
                .map_err(|err| err.to_string())
        }
        LinearPdeVarianceMode::Hutchinson | LinearPdeVarianceMode::LocalRbmc => {
            let mut rng = StdRng::seed_from_u64(config.rng_seed);
            posterior
                .hutchinson_variances(config.num_variance_probes, &mut rng)
                .map_err(|err| err.to_string())
        }
    }
}

fn estimate_derived_variances(
    problem: &NonlinearLaplaceProblem<'_>,
    prior_precision: &GmrfSparseMatrix,
    posterior: &mut Gmrf,
    config: &LinearPdeVarianceConfig,
) -> Result<BTreeMap<String, LinearPdeDerivedMarginalResult>, String> {
    if problem.derived_quantities.is_empty() {
        return Ok(BTreeMap::new());
    }
    let prior_factor = prior_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision for derived variances: {err}"))?;
    let posterior_factor = posterior
        .precision_factor()
        .ok_or_else(|| "posterior precision factor is missing".to_string())?;

    let mut out = BTreeMap::new();
    for quantity in &problem.derived_quantities {
        let prior = exact_solve_transformed_diag(&prior_factor, &quantity.operator)
            .map_err(|err| err.to_string())?
            .values;
        let posterior = match config.mode {
            LinearPdeVarianceMode::SelectedInverse => {
                match selected_inverse_transformed_diag(posterior_factor, &quantity.operator) {
                    Ok(selected) => match selected.estimate {
                        Some(estimate) => estimate.values,
                        None => {
                            exact_solve_transformed_diag(posterior_factor, &quantity.operator)
                                .map_err(|err| err.to_string())?
                                .values
                        }
                    },
                    Err(_) => {
                        exact_solve_transformed_diag(posterior_factor, &quantity.operator)
                            .map_err(|err| err.to_string())?
                            .values
                    }
                }
            }
            LinearPdeVarianceMode::Exact
            | LinearPdeVarianceMode::ExactSolves
            | LinearPdeVarianceMode::MonteCarlo
            | LinearPdeVarianceMode::Hutchinson
            | LinearPdeVarianceMode::LocalRbmc => {
                exact_solve_transformed_diag(posterior_factor, &quantity.operator)
                    .map_err(|err| err.to_string())?
                    .values
            }
        };
        out.insert(
            quantity.name.clone(),
            LinearPdeDerivedMarginalResult {
                prior_variance: gmrf_vec_to_feec(&prior),
                posterior_variance: gmrf_vec_to_feec(&posterior),
            },
        );
    }
    Ok(out)
}

fn maybe_stabilize_precision(
    precision: &GmrfSparseMatrix,
    stabilize: bool,
) -> Result<GmrfSparseMatrix, String> {
    if !stabilize {
        return Ok(precision.clone());
    }
    if default_cholesky_sqrt_lower(precision, "precision stabilization").is_ok() {
        return Ok(precision.clone());
    }
    let symmetrized = symmetrize_precision(precision);
    if default_cholesky_sqrt_lower(&symmetrized, "symmetrized precision stabilization").is_ok() {
        return Ok(symmetrized);
    }

    let (min_diag, max_abs_diag) = diagonal_stats(&symmetrized);
    let mut shift = if min_diag.is_finite() && min_diag <= 0.0 {
        (-min_diag) + max_abs_diag * 1e-8
    } else {
        max_abs_diag * 1e-12
    }
    .max(1e-10);
    let mut last_error = "precision matrix is not positive definite".to_string();
    for _ in 0..12 {
        let shifted = add_diagonal_shift(&symmetrized, shift);
        match default_cholesky_sqrt_lower(&shifted, "shifted precision stabilization") {
            Ok(_) => return Ok(shifted),
            Err(err) => {
                last_error = err;
                shift *= 10.0;
            }
        }
    }
    Err(last_error)
}

fn stabilize_precision_and_factor(
    precision: &GmrfSparseMatrix,
    stabilize: bool,
    context: &str,
    assume_symmetric: bool,
    prefer_cached_stabilization_shift: bool,
    diagnostics: Option<&mut GaussNewtonRunDiagnostics>,
) -> Result<(GmrfSparseMatrix, SparseCholeskyFactor), String> {
    let mut diagnostics = diagnostics;
    if !stabilize {
        let factor = diagnostic_cholesky_sqrt_lower(
            precision,
            context,
            CholeskyStabilizationAttemptKind::Unshifted,
            None,
            &mut diagnostics,
        )?;
        return Ok((precision.clone(), factor));
    }
    let symmetrized = if assume_symmetric {
        None
    } else {
        Some(symmetrize_precision(precision))
    };
    let matrix_for_shift = symmetrized.as_ref().unwrap_or(precision);
    if prefer_cached_stabilization_shift {
        if let Some(shift) = cached_stabilization_shift(matrix_for_shift)? {
            let shifted = add_diagonal_shift(matrix_for_shift, shift);
            if let Ok(factor) = diagnostic_cholesky_sqrt_lower(
                &shifted,
                &format!("cached-shifted {context}"),
                CholeskyStabilizationAttemptKind::CachedShift,
                Some(shift),
                &mut diagnostics,
            ) {
                return Ok((shifted, factor));
            }
        }
    }
    match diagnostic_cholesky_sqrt_lower(
        precision,
        context,
        CholeskyStabilizationAttemptKind::Unshifted,
        None,
        &mut diagnostics,
    ) {
        Ok(factor) => Ok((precision.clone(), factor)),
        Err(first_error) => {
            let symmetrized = match symmetrized {
                Some(symmetrized) => symmetrized,
                None => precision.clone(),
            };
            let sym_result = if assume_symmetric {
                Err(first_error.clone())
            } else {
                diagnostic_cholesky_sqrt_lower(
                    &symmetrized,
                    &format!("symmetrized {context}"),
                    CholeskyStabilizationAttemptKind::Symmetrized,
                    None,
                    &mut diagnostics,
                )
            };
            match sym_result {
                Ok(factor) => Ok((symmetrized, factor)),
                Err(sym_error) => {
                    let (min_diag, max_abs_diag) = diagonal_stats(&symmetrized);
                    let mut shift = if min_diag.is_finite() && min_diag <= 0.0 {
                        (-min_diag) + max_abs_diag * 1e-8
                    } else {
                        max_abs_diag * 1e-12
                    }
                    .max(1e-10);
                    let mut last_error = if sym_error.is_empty() {
                        first_error
                    } else {
                        sym_error
                    };
                    for _ in 0..12 {
                        let shifted = add_diagonal_shift(&symmetrized, shift);
                        match diagnostic_cholesky_sqrt_lower(
                            &shifted,
                            &format!("shifted {context}"),
                            CholeskyStabilizationAttemptKind::Shifted,
                            Some(shift),
                            &mut diagnostics,
                        ) {
                            Ok(factor) => {
                                remember_stabilization_shift(&symmetrized, shift)?;
                                return Ok((shifted, factor));
                            }
                            Err(shift_error) => {
                                last_error = shift_error;
                                shift *= 10.0;
                            }
                        }
                    }
                    Err(last_error)
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CholeskyStabilizationAttemptKind {
    Unshifted,
    Symmetrized,
    CachedShift,
    Shifted,
}

fn diagnostic_cholesky_sqrt_lower(
    precision: &GmrfSparseMatrix,
    context: &str,
    kind: CholeskyStabilizationAttemptKind,
    shift: Option<f64>,
    diagnostics: &mut Option<&mut GaussNewtonRunDiagnostics>,
) -> Result<SparseCholeskyFactor, String> {
    let start = Instant::now();
    let result = default_cholesky_sqrt_lower(precision, context);
    let elapsed = start.elapsed().as_secs_f64();
    if let Some(diagnostics) = diagnostics.as_deref_mut() {
        diagnostics.cholesky_factor_attempts += 1;
        diagnostics.cholesky_factorization_seconds += elapsed;
        match kind {
            CholeskyStabilizationAttemptKind::Unshifted => {
                diagnostics.cholesky_unshifted_attempts += 1;
            }
            CholeskyStabilizationAttemptKind::Symmetrized => {
                diagnostics.cholesky_symmetrized_attempts += 1;
            }
            CholeskyStabilizationAttemptKind::CachedShift => {
                diagnostics.cholesky_cached_shift_attempts += 1;
                if let Some(shift) = shift {
                    diagnostics.cholesky_max_shift = diagnostics.cholesky_max_shift.max(shift);
                }
            }
            CholeskyStabilizationAttemptKind::Shifted => {
                diagnostics.cholesky_shifted_attempts += 1;
                if let Some(shift) = shift {
                    diagnostics.cholesky_max_shift = diagnostics.cholesky_max_shift.max(shift);
                }
            }
        }
        if result.is_ok() {
            diagnostics.cholesky_factor_successes += 1;
            match kind {
                CholeskyStabilizationAttemptKind::CachedShift => {
                    diagnostics.cholesky_cached_shift_successes += 1;
                }
                CholeskyStabilizationAttemptKind::Shifted => {
                    diagnostics.cholesky_shifted_successes += 1;
                }
                CholeskyStabilizationAttemptKind::Unshifted
                | CholeskyStabilizationAttemptKind::Symmetrized => {}
            }
        }
    }
    result
}

fn default_cholesky_sqrt_lower(
    precision: &GmrfSparseMatrix,
    context: &str,
) -> Result<SparseCholeskyFactor, String> {
    let ordering = match cached_metis_nested_dissection_ordering(precision) {
        Ok(Some(permutation)) => CholeskyOrdering::Custom(permutation),
        Ok(None) => {
            eprintln!(
                "warning: ndmetis unavailable while factoring {context}; falling back to AMD"
            );
            CholeskyOrdering::Amd
        }
        Err(err) => {
            eprintln!(
                "warning: METIS ordering failed while factoring {context}: {err}; falling back to AMD"
            );
            CholeskyOrdering::Amd
        }
    };
    cached_cholesky_symbolic(precision, ordering)?
        .factor(precision)
        .map_err(|err| format!("failed to factor {context}: {err}"))
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct CholeskyGraphFingerprint {
    dimension: usize,
    offdiag_entries: usize,
    hash: u64,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct CholeskyPatternFingerprint {
    dimension: usize,
    entries: usize,
    hash: u64,
}

static METIS_ORDERING_CACHE: OnceLock<
    Mutex<HashMap<CholeskyGraphFingerprint, Option<gmrf_core::Permutation>>>,
> = OnceLock::new();

static CHOLESKY_SYMBOLIC_CACHE: OnceLock<
    Mutex<HashMap<CholeskyPatternFingerprint, SparseCholeskySymbolic>>,
> = OnceLock::new();

static CHOLESKY_STABILIZATION_SHIFT_CACHE: OnceLock<Mutex<HashMap<CholeskyGraphFingerprint, f64>>> =
    OnceLock::new();

static METIS_ORDERING_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static METIS_ORDERING_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Default)]
struct MetisOrderingCacheStats {
    hits: usize,
    misses: usize,
}

fn metis_ordering_cache_stats_snapshot() -> MetisOrderingCacheStats {
    MetisOrderingCacheStats {
        hits: METIS_ORDERING_CACHE_HITS.load(Ordering::Relaxed),
        misses: METIS_ORDERING_CACHE_MISSES.load(Ordering::Relaxed),
    }
}

fn cached_metis_nested_dissection_ordering(
    precision: &GmrfSparseMatrix,
) -> Result<Option<gmrf_core::Permutation>, String> {
    let timing_enabled = metis_timing_enabled();
    let total_start = Instant::now();
    let fingerprint_start = Instant::now();
    let key = cholesky_graph_fingerprint(precision);
    let fingerprint_seconds = fingerprint_start.elapsed().as_secs_f64();
    let cache = METIS_ORDERING_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache
            .lock()
            .map_err(|_| "METIS ordering cache lock was poisoned".to_string())?;
        if let Some(ordering) = guard.get(&key) {
            METIS_ORDERING_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            if timing_enabled {
                eprintln!(
                    "METIS ordering cache: hit dim={} offdiag_entries={} fingerprint={:.6}s total={:.6}s",
                    key.dimension,
                    key.offdiag_entries,
                    fingerprint_seconds,
                    total_start.elapsed().as_secs_f64()
                );
            }
            return Ok(ordering.clone());
        }
    }

    METIS_ORDERING_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    let metis_start = Instant::now();
    let ordering = crate::metis_ordering::metis_nested_dissection_ordering(precision)?;
    let metis_seconds = metis_start.elapsed().as_secs_f64();
    let mut guard = cache
        .lock()
        .map_err(|_| "METIS ordering cache lock was poisoned".to_string())?;
    guard.insert(key, ordering.clone());
    if timing_enabled {
        eprintln!(
            "METIS ordering cache: miss dim={} offdiag_entries={} fingerprint={:.6}s metis={:.6}s total={:.6}s",
            key.dimension,
            key.offdiag_entries,
            fingerprint_seconds,
            metis_seconds,
            total_start.elapsed().as_secs_f64()
        );
    }
    Ok(ordering)
}

fn cached_cholesky_symbolic(
    precision: &GmrfSparseMatrix,
    ordering: CholeskyOrdering,
) -> Result<SparseCholeskySymbolic, String> {
    let key = cholesky_pattern_fingerprint(precision, &ordering);
    let cache = CHOLESKY_SYMBOLIC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache
            .lock()
            .map_err(|_| "Cholesky symbolic cache lock was poisoned".to_string())?;
        if let Some(symbolic) = guard.get(&key) {
            return Ok(symbolic.clone());
        }
    }
    let symbolic = precision
        .analyze_cholesky_with_ordering(ordering)
        .map_err(|err| format!("failed to analyze sparse Cholesky pattern: {err}"))?;
    let mut guard = cache
        .lock()
        .map_err(|_| "Cholesky symbolic cache lock was poisoned".to_string())?;
    guard.insert(key, symbolic.clone());
    Ok(symbolic)
}

fn cached_stabilization_shift(precision: &GmrfSparseMatrix) -> Result<Option<f64>, String> {
    let key = cholesky_graph_fingerprint(precision);
    let cache = CHOLESKY_STABILIZATION_SHIFT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let guard = cache
        .lock()
        .map_err(|_| "Cholesky stabilization shift cache lock was poisoned".to_string())?;
    Ok(guard.get(&key).copied())
}

fn remember_stabilization_shift(precision: &GmrfSparseMatrix, shift: f64) -> Result<(), String> {
    if !shift.is_finite() || shift <= 0.0 {
        return Ok(());
    }
    let key = cholesky_graph_fingerprint(precision);
    let cache = CHOLESKY_STABILIZATION_SHIFT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| "Cholesky stabilization shift cache lock was poisoned".to_string())?;
    guard
        .entry(key)
        .and_modify(|cached| *cached = (*cached).min(shift))
        .or_insert(shift);
    Ok(())
}

#[cfg(test)]
fn clear_cholesky_stabilization_shift_cache() {
    if let Some(cache) = CHOLESKY_STABILIZATION_SHIFT_CACHE.get() {
        cache.lock().expect("shift cache lock").clear();
    }
}

fn nonlinear_timing_enabled() -> bool {
    env_flag_enabled("FEG_NONLINEAR_TIMING")
}

fn metis_timing_enabled() -> bool {
    env_flag_enabled("FEG_METIS_TIMING")
}

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn cholesky_graph_fingerprint(precision: &GmrfSparseMatrix) -> CholeskyGraphFingerprint {
    let mut hasher = DefaultHasher::new();
    precision.nrows().hash(&mut hasher);
    let mut offdiag_entries = 0usize;
    for (row, col, value) in precision.triplet_iter() {
        if row == col || *value == 0.0 {
            continue;
        }
        let (lo, hi) = if row < col { (row, col) } else { (col, row) };
        lo.hash(&mut hasher);
        hi.hash(&mut hasher);
        offdiag_entries += 1;
    }
    CholeskyGraphFingerprint {
        dimension: precision.nrows(),
        offdiag_entries,
        hash: hasher.finish(),
    }
}

fn cholesky_pattern_fingerprint(
    precision: &GmrfSparseMatrix,
    ordering: &CholeskyOrdering,
) -> CholeskyPatternFingerprint {
    let mut hasher = DefaultHasher::new();
    precision.nrows().hash(&mut hasher);
    match ordering {
        CholeskyOrdering::Amd => "amd".hash(&mut hasher),
        CholeskyOrdering::Identity => "identity".hash(&mut hasher),
        CholeskyOrdering::Custom(permutation) => {
            "custom".hash(&mut hasher);
            permutation.orig_to_perm.hash(&mut hasher);
            permutation.perm_to_orig.hash(&mut hasher);
        }
    }
    let mut entries = 0usize;
    for (row, col, _) in precision.triplet_iter() {
        row.hash(&mut hasher);
        col.hash(&mut hasher);
        entries += 1;
    }
    CholeskyPatternFingerprint {
        dimension: precision.nrows(),
        entries,
        hash: hasher.finish(),
    }
}

fn lower_triangle_nnz(matrix: &GmrfSparseMatrix) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, col, _)| row >= col)
        .count()
}

fn diagonal_stats(matrix: &GmrfSparseMatrix) -> (f64, f64) {
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let min_diag = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_diag = diagonal.iter().copied().map(f64::abs).fold(0.0, f64::max);
    (min_diag, max_abs_diag.max(1.0))
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

fn add_diagonal_shift(matrix: &GmrfSparseMatrix, shift: f64) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    GmrfSparseMatrix::from(&coo)
}

fn add_levenberg_marquardt_diagonal(
    matrix: &GmrfSparseMatrix,
    lambda: f64,
) -> Result<GmrfSparseMatrix, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err("Levenberg-Marquardt regularization requires a square matrix".to_string());
    }
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let max_abs_diag = diagonal
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0);
    let diagonal_floor = max_abs_diag * 1e-12;
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (index, value) in diagonal.into_iter().enumerate() {
        coo.push(index, index, lambda * value.abs().max(diagonal_floor));
    }
    Ok(GmrfSparseMatrix::from(&coo))
}

fn core_sparse_mul_vec(matrix: &SparseTripletMatrix, values: &[f64]) -> Result<Vec<f64>, String> {
    if matrix.ncols() != values.len() {
        return Err(format!(
            "sparse matrix column count {} must match vector length {}",
            matrix.ncols(),
            values.len()
        ));
    }
    let mut out = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        out[row] += value * values[col];
    }
    Ok(out)
}

fn sparse_matmul(
    lhs: &SparseTripletMatrix,
    rhs: &SparseTripletMatrix,
) -> Result<SparseTripletMatrix, String> {
    if lhs.ncols() != rhs.nrows() {
        return Err(format!(
            "cannot multiply sparse matrices with dimensions {}x{} and {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    let mut rhs_rows = vec![Vec::<(usize, f64)>::new(); rhs.nrows()];
    for (row, col, value) in rhs.triplet_iter() {
        rhs_rows[row].push((col, value));
    }
    let mut values = BTreeMap::<(usize, usize), f64>::new();
    for (row, mid, lhs_value) in lhs.triplet_iter() {
        for (col, rhs_value) in &rhs_rows[mid] {
            let value = lhs_value * *rhs_value;
            if value != 0.0 {
                *values.entry((row, *col)).or_insert(0.0) += value;
            }
        }
    }
    Ok(SparseTripletMatrix::from_triplets(
        lhs.nrows(),
        rhs.ncols(),
        values
            .into_iter()
            .filter(|(_, value)| *value != 0.0)
            .map(|((row, col), value)| SparseTriplet { row, col, value }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_pde::LinearPdeVarianceConfig;
    use gmrf_core::{apply_gaussian_observations, Vector as GmrfVector};

    struct LinearResidual {
        matrix: SparseTripletMatrix,
        bias: Vec<f64>,
    }

    impl NonlinearResidualModel for LinearResidual {
        fn state_dimension(&self) -> usize {
            self.matrix.ncols()
        }

        fn residual_dimension(&self) -> usize {
            self.matrix.nrows()
        }

        fn residual_and_jacobian(
            &self,
            state: &[f64],
        ) -> Result<NonlinearResidualEvaluation, String> {
            let mut residual = core_sparse_mul_vec(&self.matrix, state)?;
            for (value, bias) in residual.iter_mut().zip(self.bias.iter()) {
                *value += bias;
            }
            Ok(NonlinearResidualEvaluation {
                residual,
                jacobian: self.matrix.clone(),
            })
        }
    }

    struct SquareResidual;

    impl NonlinearResidualModel for SquareResidual {
        fn state_dimension(&self) -> usize {
            1
        }

        fn residual_dimension(&self) -> usize {
            1
        }

        fn residual_and_jacobian(
            &self,
            state: &[f64],
        ) -> Result<NonlinearResidualEvaluation, String> {
            let x = state[0];
            let mut jacobian = SparseTripletMatrix::new(1, 1);
            jacobian.push(0, 0, 2.0 * x);
            Ok(NonlinearResidualEvaluation {
                residual: vec![x * x - 1.0],
                jacobian,
            })
        }
    }

    fn scalar_precision(value: f64) -> GmrfSparseMatrix {
        let mut coo = GmrfCoo::new(1, 1);
        coo.push(0, 0, value);
        GmrfSparseMatrix::from(&coo)
    }

    #[test]
    fn cholesky_stabilization_diagnostics_count_shifted_attempts() {
        clear_cholesky_stabilization_shift_cache();
        let precision = scalar_precision(0.0);
        let mut diagnostics = GaussNewtonRunDiagnostics::default();

        let (stabilized, factor) = stabilize_precision_and_factor(
            &precision,
            true,
            "singular scalar precision",
            true,
            false,
            Some(&mut diagnostics),
        )
        .expect("singular scalar precision should be stabilized by a diagonal shift");

        assert_eq!(factor.dimension(), 1);
        assert!(stabilized
            .triplet_iter()
            .any(|(row, col, value)| { row == col && *value > 0.0 }));
        assert_eq!(diagnostics.cholesky_unshifted_attempts, 1);
        assert_eq!(diagnostics.cholesky_symmetrized_attempts, 0);
        assert_eq!(diagnostics.cholesky_shifted_attempts, 1);
        assert_eq!(diagnostics.cholesky_shifted_successes, 1);
        assert_eq!(diagnostics.cholesky_factor_attempts, 2);
        assert_eq!(diagnostics.cholesky_factor_successes, 1);
        assert!(diagnostics.cholesky_max_shift > 0.0);
        assert!(diagnostics.cholesky_factorization_seconds >= 0.0);
    }

    #[test]
    fn cholesky_stabilization_can_reuse_cached_shift() {
        clear_cholesky_stabilization_shift_cache();
        let precision = scalar_precision(0.0);
        let mut first = GaussNewtonRunDiagnostics::default();
        stabilize_precision_and_factor(
            &precision,
            true,
            "singular scalar precision",
            true,
            false,
            Some(&mut first),
        )
        .expect("first solve should populate the shift cache");

        let mut second = GaussNewtonRunDiagnostics::default();
        stabilize_precision_and_factor(
            &precision,
            true,
            "singular scalar precision",
            true,
            true,
            Some(&mut second),
        )
        .expect("second solve should use the cached shift");

        assert_eq!(second.cholesky_cached_shift_attempts, 1);
        assert_eq!(second.cholesky_cached_shift_successes, 1);
        assert_eq!(second.cholesky_unshifted_attempts, 0);
        assert_eq!(second.cholesky_shifted_attempts, 0);
        assert_eq!(second.cholesky_factor_attempts, 1);
    }

    #[test]
    fn square_newton_solves_square_residual_with_direct_tangent() {
        let result = solve_square_nonlinear_system(
            &SquareResidual,
            &SquareNewtonConfig {
                initial_guess: Some(vec![0.2]),
                max_iterations: 12,
                residual_tolerance: 1e-10,
                linear_solve: GaussNewtonLinearSolve::DirectCholesky,
                ..SquareNewtonConfig::default()
            },
        )
        .expect("square Newton should solve scalar residual");

        assert!(result.converged);
        assert!(result.residual_norm <= 1e-10);
        assert!((result.solution[0] - 1.0).abs() <= 1e-8);
        assert!(result.history.iter().any(|entry| entry.alpha < 1.0));
        assert!(result.history.iter().all(|entry| {
            entry.linear_solve.mode == GaussNewtonLinearSolveMode::DirectCholesky
                && entry.linear_solve.factor_nnz.is_some()
                && entry.trial_residual_norm <= entry.residual_norm
        }));
    }

    #[test]
    fn square_newton_solves_linear_residual_with_iterative_tangent() {
        let mut matrix = SparseTripletMatrix::new(1, 1);
        matrix.push(0, 0, 2.0);
        let model = LinearResidual {
            matrix,
            bias: vec![1.0],
        };
        let result = solve_square_nonlinear_system(
            &model,
            &SquareNewtonConfig {
                initial_guess: Some(vec![0.0]),
                max_iterations: 4,
                residual_tolerance: 1e-12,
                linear_solve: GaussNewtonLinearSolve::IterativeCg {
                    tolerance: 1e-12,
                    max_iterations: 16,
                    warm_start: false,
                },
                ..SquareNewtonConfig::default()
            },
        )
        .expect("square Newton should solve linear residual with CG");

        assert!(result.converged);
        assert!((result.solution[0] + 0.5).abs() <= 1e-12);
        assert!(result.history.iter().all(|entry| {
            entry.linear_solve.mode == GaussNewtonLinearSolveMode::IterativeCg
                && entry.linear_solve.factor_nnz.is_none()
        }));
    }

    #[test]
    fn square_newton_rejects_rectangular_residual() {
        let mut matrix = SparseTripletMatrix::new(2, 1);
        matrix.push(0, 0, 1.0);
        matrix.push(1, 0, 2.0);
        let model = LinearResidual {
            matrix,
            bias: vec![0.0, 0.0],
        };

        let err = solve_square_nonlinear_system(&model, &SquareNewtonConfig::default())
            .expect_err("rectangular residual should be rejected");
        assert!(err.contains("square Newton requires residual dimension"));
    }

    struct AmbientResidual;

    impl NonlinearResidualModel for AmbientResidual {
        fn state_dimension(&self) -> usize {
            2
        }

        fn residual_dimension(&self) -> usize {
            1
        }

        fn residual_and_jacobian(
            &self,
            state: &[f64],
        ) -> Result<NonlinearResidualEvaluation, String> {
            let mut jacobian = SparseTripletMatrix::new(1, 2);
            jacobian.push(0, 0, 2.0 * state[0]);
            jacobian.push(0, 1, 2.0);
            Ok(NonlinearResidualEvaluation {
                residual: vec![state[0] * state[0] + 2.0 * state[1]],
                jacobian,
            })
        }
    }

    fn diagonal_prior(dimension: usize, precision: f64) -> GaussianPriorSpec {
        let mut matrix = SparseTripletMatrix::new(dimension, dimension);
        for index in 0..dimension {
            matrix.push(index, index, precision);
        }
        GaussianPriorSpec {
            mean: vec![0.0; dimension],
            precision: matrix,
        }
    }

    #[test]
    fn smooth_abs_linear_residual_has_finite_zero_value_and_jacobian() {
        let mut operator = SparseTripletMatrix::new(1, 2);
        operator.push(0, 0, 2.0);
        operator.push(0, 1, -2.0);
        let model = SmoothAbsLinearResidualModel::new(operator, vec![0.0], 1e-3).unwrap();

        let evaluation = model.residual_and_jacobian(&[1.0, 1.0]).unwrap();

        assert_eq!(model.state_dimension(), 2);
        assert_eq!(model.residual_dimension(), 1);
        assert!((evaluation.residual[0] - 1e-3).abs() <= 1e-15);
        assert!(evaluation.residual[0].is_finite());
        assert_eq!(evaluation.jacobian.nnz(), 0);
    }

    #[test]
    fn smooth_abs_linear_residual_jacobian_matches_finite_difference() {
        let mut operator = SparseTripletMatrix::new(2, 2);
        operator.push(0, 0, 1.5);
        operator.push(0, 1, -0.25);
        operator.push(1, 0, -2.0);
        operator.push(1, 1, 0.5);
        let model = SmoothAbsLinearResidualModel::new(operator, vec![0.1, -0.3], 1e-4).unwrap();
        let state = [0.7, -1.2];
        let evaluation = model.residual_and_jacobian(&state).unwrap();

        for output_row in 0..model.residual_dimension() {
            for input_col in 0..model.state_dimension() {
                let analytic = evaluation
                    .jacobian
                    .triplet_iter()
                    .filter(|(row, col, _)| *row == output_row && *col == input_col)
                    .map(|(_, _, value)| value)
                    .sum::<f64>();
                let eps = 1e-6;
                let mut plus_state = state;
                let mut minus_state = state;
                plus_state[input_col] += eps;
                minus_state[input_col] -= eps;
                let plus = model.residual_and_jacobian(&plus_state).unwrap().residual[output_row];
                let minus = model.residual_and_jacobian(&minus_state).unwrap().residual[output_row];
                let finite_difference = (plus - minus) / (2.0 * eps);
                assert!(
                    (analytic - finite_difference).abs() <= 1e-8,
                    "row {output_row} col {input_col}: analytic {analytic} finite_difference {finite_difference}"
                );
            }
        }
    }

    #[test]
    fn grouped_norm_linear_residual_has_finite_zero_value_and_jacobian() {
        let mut operator = SparseTripletMatrix::new(3, 2);
        operator.push(0, 0, 1.0);
        operator.push(1, 1, -1.0);
        operator.push(2, 0, 0.5);
        let model = SmoothGroupedNormLinearResidualModel::new(
            operator,
            vec![0.0; 3],
            vec![SmoothGroupedNormObservation {
                name: "zero_vector".to_string(),
                samples: vec![SmoothGroupedNormSample {
                    rows: vec![0, 1, 2],
                    weight: 1.0,
                }],
            }],
            1e-3,
        )
        .unwrap();

        let evaluation = model.residual_and_jacobian(&[0.0, 0.0]).unwrap();

        assert_eq!(model.state_dimension(), 2);
        assert_eq!(model.residual_dimension(), 1);
        assert!((evaluation.residual[0] - 1e-3).abs() <= 1e-15);
        assert!(evaluation.residual[0].is_finite());
        assert_eq!(evaluation.jacobian.nnz(), 0);
    }

    #[test]
    fn grouped_norm_linear_residual_jacobian_matches_finite_difference() {
        let mut operator = SparseTripletMatrix::new(6, 3);
        operator.push(0, 0, 1.2);
        operator.push(0, 1, -0.4);
        operator.push(1, 1, 0.7);
        operator.push(2, 2, -1.1);
        operator.push(3, 0, -0.3);
        operator.push(3, 2, 1.5);
        operator.push(4, 1, 0.9);
        operator.push(5, 0, 0.25);
        operator.push(5, 2, -0.6);
        let model = SmoothGroupedNormLinearResidualModel::new(
            operator,
            vec![0.1, -0.2, 0.05, 0.3, -0.4, 0.2],
            vec![
                SmoothGroupedNormObservation {
                    name: "surface_average".to_string(),
                    samples: vec![
                        SmoothGroupedNormSample {
                            rows: vec![0, 1, 2],
                            weight: 0.25,
                        },
                        SmoothGroupedNormSample {
                            rows: vec![3, 4, 5],
                            weight: 0.75,
                        },
                    ],
                },
                SmoothGroupedNormObservation {
                    name: "point_norm".to_string(),
                    samples: vec![SmoothGroupedNormSample {
                        rows: vec![3, 4, 5],
                        weight: 1.0,
                    }],
                },
            ],
            1e-5,
        )
        .unwrap();
        let state = [0.7, -1.2, 0.4];
        let evaluation = model.residual_and_jacobian(&state).unwrap();

        for output_row in 0..model.residual_dimension() {
            for input_col in 0..model.state_dimension() {
                let analytic = evaluation
                    .jacobian
                    .triplet_iter()
                    .filter(|(row, col, _)| *row == output_row && *col == input_col)
                    .map(|(_, _, value)| value)
                    .sum::<f64>();
                let eps = 1e-6;
                let mut plus_state = state;
                let mut minus_state = state;
                plus_state[input_col] += eps;
                minus_state[input_col] -= eps;
                let plus = model.residual_and_jacobian(&plus_state).unwrap().residual[output_row];
                let minus = model.residual_and_jacobian(&minus_state).unwrap().residual[output_row];
                let finite_difference = (plus - minus) / (2.0 * eps);
                assert!(
                    (analytic - finite_difference).abs() <= 1e-8,
                    "row {output_row} col {input_col}: analytic {analytic} finite_difference {finite_difference}"
                );
            }
        }
    }

    #[test]
    fn grouped_norm_linear_residual_rejects_invalid_rows() {
        let operator = SparseTripletMatrix::new(2, 1);
        let error = SmoothGroupedNormLinearResidualModel::new(
            operator,
            vec![0.0; 2],
            vec![SmoothGroupedNormObservation {
                name: "bad".to_string(),
                samples: vec![SmoothGroupedNormSample {
                    rows: vec![0, 2],
                    weight: 1.0,
                }],
            }],
            1e-3,
        )
        .unwrap_err();

        assert!(error.contains("references row 2"));
    }

    #[test]
    fn invalid_noise_and_dimension_mismatches_are_rejected() {
        let model = SquareResidual;
        let bad_noise = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 1.0),
            residual_terms: vec![NonlinearResidualTerm::zero(
                "square",
                &model,
                GaussianNoiseModel::ScalarVariance(0.0),
            )],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };
        assert!(solve_nonlinear_laplace(&bad_noise, &GaussNewtonConfig::default()).is_err());

        let bad_obs = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 1.0),
            residual_terms: vec![NonlinearResidualTerm {
                name: "square".to_string(),
                model: &model,
                observations: vec![0.0, 0.0],
                noise: GaussianNoiseModel::ScalarVariance(1.0),
            }],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };
        assert!(solve_nonlinear_laplace(&bad_obs, &GaussNewtonConfig::default()).is_err());
    }

    #[test]
    fn linear_residual_matches_sparse_gaussian_update() {
        let mut matrix = SparseTripletMatrix::new(1, 1);
        matrix.push(0, 0, 2.0);
        let model = LinearResidual {
            matrix: matrix.clone(),
            bias: vec![1.0],
        };
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 1.0),
            residual_terms: vec![NonlinearResidualTerm {
                name: "linear".to_string(),
                model: &model,
                observations: vec![3.0],
                noise: GaussianNoiseModel::ScalarVariance(0.5),
            }],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };

        let cg_result = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.0]),
                ..GaussNewtonConfig::default()
            },
        )
        .expect("linear residual should solve with CG intermediate solve");

        let direct_result = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.0]),
                linear_solve: GaussNewtonLinearSolve::DirectCholesky,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("linear residual should solve with direct intermediate solve");

        let prior = sparse_from_core(&problem.prior.precision);
        let (expected_precision, expected_information) = apply_gaussian_observations(
            &prior,
            &sparse_from_core(&matrix),
            &GmrfVector::from_vec(vec![3.0]),
            Some(&GmrfVector::from_vec(vec![1.0])),
            0.5,
        );
        let expected_mean = expected_precision
            .cholesky_sqrt_lower()
            .unwrap()
            .solve(&expected_information)
            .unwrap();
        assert!((cg_result.map[0] - expected_mean[0]).abs() <= 1e-12);
        assert!((direct_result.map[0] - expected_mean[0]).abs() <= 1e-12);
        assert!((cg_result.map[0] - direct_result.map[0]).abs() <= 1e-12);
        assert!(cg_result
            .history
            .iter()
            .all(|entry| entry.linear_solve.mode == GaussNewtonLinearSolveMode::IterativeCg));
        assert!(direct_result
            .history
            .iter()
            .all(|entry| entry.linear_solve.mode == GaussNewtonLinearSolveMode::DirectCholesky));
        assert!(
            (cg_result
                .posterior_precision
                .triplet_iter()
                .map(|(_, _, v)| v)
                .sum::<f64>()
                - 9.0)
                .abs()
                <= 1e-12
        );
    }

    #[test]
    fn nonlinear_square_residual_converges_and_reports_laplace_precision() {
        let model = SquareResidual;
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 0.1),
            residual_terms: vec![NonlinearResidualTerm::zero(
                "square",
                &model,
                GaussianNoiseModel::ScalarVariance(1.0),
            )],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };
        let result = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![2.0]),
                gradient_tolerance: 1e-10,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("nonlinear residual should solve");

        assert!(result.converged);
        assert!(result.final_factorization.nnz > 0);
        assert!(result.final_factorization.elapsed_seconds.is_finite());
        assert_eq!(result.assembly.dimension, 1);
        assert!(result.assembly.prior_precision_nnz > 0);
        assert!(result.assembly.posterior_precision_nnz > 0);
        assert_eq!(
            result.assembly.factor_nnz,
            Some(result.final_factorization.nnz)
        );
        assert!(result
            .assembly
            .fill_ratio_vs_lower_triangle
            .unwrap()
            .is_finite());
        assert_eq!(
            result
                .assembly
                .term_operator_nnz(NonlinearAssemblyTermKind::Residual),
            1
        );
        assert_eq!(
            result
                .assembly
                .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual),
            1
        );
        assert!(result
            .history
            .iter()
            .all(|entry| entry.linear_solve.mode == GaussNewtonLinearSolveMode::IterativeCg));
        let expected_map = 0.95_f64.sqrt();
        assert!((result.map[0] - expected_map).abs() <= 1e-8);
        let expected_precision = 0.1 + (2.0 * result.map[0]).powi(2);
        let actual_precision = result
            .posterior_precision
            .triplet_iter()
            .filter(|(row, col, _)| row == col)
            .map(|(_, _, value)| value)
            .sum::<f64>();
        assert!((actual_precision - expected_precision).abs() <= 1e-8);
    }

    #[test]
    fn line_search_damps_oversized_step() {
        let model = SquareResidual;
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 0.01),
            residual_terms: vec![NonlinearResidualTerm::zero(
                "square",
                &model,
                GaussianNoiseModel::ScalarVariance(1.0),
            )],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };
        let result = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.1]),
                max_iterations: 8,
                gradient_tolerance: 1e-10,
                variance: LinearPdeVarianceConfig::default(),
                ..GaussNewtonConfig::default()
            },
        )
        .expect("damped nonlinear residual should solve");

        assert!(result.history.iter().any(|entry| entry.alpha < 1.0));
        assert!(result
            .history
            .iter()
            .all(|entry| entry.trial_objective <= entry.objective));
    }

    #[test]
    fn lm_grid_uses_zero_lambda_when_plain_gauss_newton_is_accepted() {
        let mut matrix = SparseTripletMatrix::new(1, 1);
        matrix.push(0, 0, 2.0);
        let model = LinearResidual {
            matrix,
            bias: vec![1.0],
        };
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 1.0),
            residual_terms: vec![NonlinearResidualTerm {
                name: "linear".to_string(),
                model: &model,
                observations: vec![3.0],
                noise: GaussianNoiseModel::ScalarVariance(0.5),
            }],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };

        let result = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.0]),
                step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
                linear_solve: GaussNewtonLinearSolve::DirectCholesky,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("linear residual should solve with LM grid");

        assert!(!result.history.is_empty());
        assert!(result.history.iter().all(|entry| {
            entry.regularization == GaussNewtonStepRegularization::LevenbergMarquardtGrid
                && entry.regularization_lambda == 0.0
        }));
    }

    #[test]
    fn lm_grid_decreases_objective_on_damped_nonlinear_residual() {
        let model = SquareResidual;
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 0.01),
            residual_terms: vec![NonlinearResidualTerm::zero(
                "square",
                &model,
                GaussianNoiseModel::ScalarVariance(1.0),
            )],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };

        let result = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.1]),
                max_iterations: 8,
                gradient_tolerance: 1e-10,
                step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("LM-grid nonlinear residual should solve");

        assert!(result.converged);
        assert!(result.history.iter().all(|entry| {
            entry.regularization == GaussNewtonStepRegularization::LevenbergMarquardtGrid
                && entry.trial_objective <= entry.objective
        }));
    }

    #[test]
    fn adaptive_lm_matches_grid_and_uses_no_more_step_solves() {
        let model = SquareResidual;
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 0.01),
            residual_terms: vec![NonlinearResidualTerm::zero(
                "square",
                &model,
                GaussianNoiseModel::ScalarVariance(1.0),
            )],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };

        let grid = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.1]),
                max_iterations: 8,
                gradient_tolerance: 1e-10,
                step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("LM-grid nonlinear residual should solve");
        let adaptive = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.1]),
                max_iterations: 8,
                gradient_tolerance: 1e-10,
                step_regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("adaptive LM nonlinear residual should solve");

        assert!(adaptive.converged);
        assert!((adaptive.map[0] - grid.map[0]).abs() <= 1e-8);
        assert!(
            adaptive.diagnostics.step_solve_attempts <= grid.diagnostics.step_solve_attempts,
            "adaptive attempts {}, grid attempts {}",
            adaptive.diagnostics.step_solve_attempts,
            grid.diagnostics.step_solve_attempts
        );
        assert_eq!(
            adaptive.diagnostics.accepted_iterations,
            adaptive.history.len()
        );
    }

    #[test]
    fn step_regularization_parser_accepts_expected_modes() {
        assert_eq!(
            "none".parse::<GaussNewtonStepRegularization>().unwrap(),
            GaussNewtonStepRegularization::None
        );
        assert_eq!(
            "lm-grid".parse::<GaussNewtonStepRegularization>().unwrap(),
            GaussNewtonStepRegularization::LevenbergMarquardtGrid
        );
        assert_eq!(
            "adaptive-lm"
                .parse::<GaussNewtonStepRegularization>()
                .unwrap(),
            GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt
        );
        assert!("spectral-lm"
            .parse::<GaussNewtonStepRegularization>()
            .is_err());
    }

    #[test]
    fn lm_grid_does_not_shift_final_laplace_precision() {
        let mut matrix = SparseTripletMatrix::new(1, 1);
        matrix.push(0, 0, 2.0);
        let model = LinearResidual {
            matrix,
            bias: vec![1.0],
        };
        let problem = NonlinearLaplaceProblem {
            prior: diagonal_prior(1, 1.0),
            residual_terms: vec![NonlinearResidualTerm {
                name: "linear".to_string(),
                model: &model,
                observations: vec![3.0],
                noise: GaussianNoiseModel::ScalarVariance(0.5),
            }],
            linear_measurements: Vec::new(),
            precision_weighted_measurements: Vec::new(),
            derived_quantities: Vec::new(),
        };

        let undamped = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.0]),
                linear_solve: GaussNewtonLinearSolve::DirectCholesky,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("undamped linear residual should solve");
        let regularized = solve_nonlinear_laplace(
            &problem,
            &GaussNewtonConfig {
                initial_guess: Some(vec![0.0]),
                linear_solve: GaussNewtonLinearSolve::DirectCholesky,
                step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
                ..GaussNewtonConfig::default()
            },
        )
        .expect("LM-grid linear residual should solve");

        assert!((undamped.map[0] - regularized.map[0]).abs() <= 1e-12);
        let undamped_entries = undamped
            .posterior_precision
            .triplet_iter()
            .collect::<Vec<_>>();
        let regularized_entries = regularized
            .posterior_precision
            .triplet_iter()
            .collect::<Vec<_>>();
        assert_eq!(undamped_entries, regularized_entries);
    }

    #[test]
    fn transformed_residual_applies_sparse_chain_rule() {
        let ambient = AmbientResidual;
        let mut transform = SparseTripletMatrix::new(2, 1);
        transform.push(0, 0, 2.0);
        transform.push(1, 0, -1.0);
        let transformed = TransformedResidualModel::new(&ambient, transform).unwrap();

        let z = vec![0.3];
        let evaluation = transformed.residual_and_jacobian(&z).unwrap();
        let analytic = evaluation
            .jacobian
            .triplet_iter()
            .find(|(row, col, _)| *row == 0 && *col == 0)
            .map(|(_, _, value)| value)
            .unwrap();

        let eps = 1e-6;
        let plus = transformed
            .residual_and_jacobian(&[z[0] + eps])
            .unwrap()
            .residual[0];
        let minus = transformed
            .residual_and_jacobian(&[z[0] - eps])
            .unwrap()
            .residual[0];
        let finite_difference = (plus - minus) / (2.0 * eps);
        assert!((analytic - finite_difference).abs() <= 1e-8);
    }

    #[test]
    fn selected_residual_model_selects_residual_and_jacobian_rows() {
        let mut matrix = SparseTripletMatrix::new(3, 2);
        matrix.push(0, 0, 1.0);
        matrix.push(1, 0, 2.0);
        matrix.push(1, 1, 3.0);
        matrix.push(2, 1, 4.0);
        let model = LinearResidual {
            matrix,
            bias: vec![10.0, 20.0, 30.0],
        };
        let selected = SelectedResidualModel::new(&model, vec![2, 0]).unwrap();
        let evaluation = selected.residual_and_jacobian(&[5.0, 7.0]).unwrap();

        assert_eq!(selected.state_dimension(), 2);
        assert_eq!(selected.residual_dimension(), 2);
        assert_eq!(evaluation.residual, vec![58.0, 15.0]);
        let triplets = evaluation.jacobian.triplet_iter().collect::<Vec<_>>();
        assert!(triplets.contains(&(0, 1, 4.0)));
        assert!(triplets.contains(&(1, 0, 1.0)));
    }

    #[test]
    fn trace_normalized_precision_has_unit_mean_diagonal_before_variance_scale() {
        let mut base = SparseTripletMatrix::new(2, 2);
        base.push(0, 0, 2.0);
        base.push(1, 1, 6.0);
        base.push(0, 1, 1.0);
        base.push(1, 0, 1.0);

        let normalized = trace_normalized_precision(&base, 0.5).unwrap();
        assert!((normalized.normalization_scale - 0.25).abs() <= 1e-12);
        let trace = normalized
            .precision
            .triplet_iter()
            .filter(|(row, col, _)| row == col)
            .map(|(_, _, value)| value)
            .sum::<f64>();
        assert!((trace / 2.0 - 2.0).abs() <= 1e-12);
    }
}
