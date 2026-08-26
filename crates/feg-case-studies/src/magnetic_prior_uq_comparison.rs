use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feec_gmrf::prelude::{
    gaussian_predictive_diagnostics_95, sparse_mat_from_feec_csr, FactoredGaussianPrior,
    GaussianNoise, GaussianPrior, HutchinsonVarianceConfig, LinearGaussianModelBuilder, LinearMap,
    LinearObservation, Posterior, PreparedLinearGaussianModel, ProbeDistribution,
    VarianceEstimator, VarianceMethod, WeightedVarianceEstimate,
};
use feg_infer::{
    physical::build_reduced_magnetic_flux_density_operator_3d,
    prior::{
        matern::{
            one_form::{
                build_hodge_laplacian_1form, build_matern_mass_inverse_1form_with_coords,
                build_matern_precision_1form_with_mass_inverse_for_alpha,
                build_matern_system_matrix_1form, MaternMassInverse,
            },
            MaternAlpha,
        },
        sparse_anchor_hodge::spectrum_matched_potential_precision,
        trace_normalization::trace_normalization_from_target_trace,
    },
    sparse::{
        add_feec_diagonal_shift, restrict_square_with_layout, scale_matrix, symmetrize_feec_csr,
    },
};
use formoniq::problems::nonlinear_magnetostatic::{
    build_reduced_vector_potential_magnetostatic_3d, NonlinearMagnetostaticAssemblyConfig,
    NonlinearReluctivityLaw, ReducedVectorPotentialMagnetostatic3d,
};
use formoniq::reduction::EssentialBoundarySpec;
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
    topology::complex::Complex,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::PI,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
    time::Instant,
};

const ALPHA: MaternAlpha = MaternAlpha::Two;
const TARGET_CUBE_SIDE_LENGTH_M: f64 = 1.0;
const DEFAULT_PRACTICAL_RANGE_M: f64 = 0.75;
const SMOOTH_TRUTH_REPLICATE: usize = 0;
const SMOOTH_TRUTH_SEED: u64 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MagneticPriorModelKind {
    OrdinaryPotential,
    SpectrumMatchedPotential,
}

impl MagneticPriorModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryPotential => "ordinary_potential",
            Self::SpectrumMatchedPotential => "spectrum_matched_potential",
        }
    }
}

impl std::fmt::Display for MagneticPriorModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for MagneticPriorModelKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ordinary" | "ordinary-potential" | "ordinary_potential" => {
                Ok(Self::OrdinaryPotential)
            }
            "spectrum-matched"
            | "spectrum-matched-potential"
            | "spectrum_matched"
            | "spectrum_matched_potential"
            | "corrected"
            | "corrected_potential" => Ok(Self::SpectrumMatchedPotential),
            _ => Err(format!(
                "unknown magnetic prior model `{value}`; expected ordinary_potential or spectrum_matched_potential"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MagneticTruthSource {
    SmoothManufactured,
    SpectrumMatchedPriorSample,
}

impl MagneticTruthSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SmoothManufactured => "smooth_truth",
            Self::SpectrumMatchedPriorSample => "corrected_prior_sample_truth",
        }
    }
}

impl std::fmt::Display for MagneticTruthSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for MagneticTruthSource {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "smooth" | "smooth_truth" | "smooth-manufactured" => Ok(Self::SmoothManufactured),
            "corrected"
            | "corrected_prior_sample_truth"
            | "corrected-prior-sample-truth"
            | "spectrum_matched_prior_sample"
            | "spectrum-matched-prior-sample" => Ok(Self::SpectrumMatchedPriorSample),
            _ => Err(format!(
                "unknown magnetic truth source `{value}`; expected smooth_truth or corrected_prior_sample_truth"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MagneticPriorUqComparisonConfig {
    pub level: usize,
    pub practical_range_m: f64,
    pub target_b_rms_tesla: f64,
    pub truth_b_rms_tesla: f64,
    pub observation_std_tesla: f64,
    pub training_sensor_cells: usize,
    pub validation_cells_per_bin: usize,
    pub test_cells_per_bin: usize,
    pub smooth_noise_replicates: usize,
    pub corrected_sample_truth_replicates: usize,
    pub exact_max_dofs: usize,
    pub hutchinson_probes: usize,
    pub hutchinson_batches: usize,
    pub rng_seed: u64,
}

impl Default for MagneticPriorUqComparisonConfig {
    fn default() -> Self {
        Self {
            level: 5,
            practical_range_m: DEFAULT_PRACTICAL_RANGE_M,
            target_b_rms_tesla: 0.10,
            truth_b_rms_tesla: 0.10,
            observation_std_tesla: 0.005,
            training_sensor_cells: 24,
            validation_cells_per_bin: 24,
            test_cells_per_bin: 24,
            smooth_noise_replicates: 20,
            corrected_sample_truth_replicates: 30,
            exact_max_dofs: 1500,
            hutchinson_probes: 128,
            hutchinson_batches: 4,
            rng_seed: 0xB0B5_1E1D,
        }
    }
}

impl MagneticPriorUqComparisonConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke() -> Self {
        Self {
            level: 2,
            training_sensor_cells: 4,
            validation_cells_per_bin: 2,
            test_cells_per_bin: 2,
            smooth_noise_replicates: 1,
            corrected_sample_truth_replicates: 1,
            exact_max_dofs: 128,
            hutchinson_probes: 8,
            hutchinson_batches: 2,
            ..Self::default()
        }
    }

    /// Immutable configuration used by the submitted thesis.
    pub fn thesis_submitted() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct MagneticPriorUqComparisonReport {
    pub config: MagneticPriorUqComparisonConfig,
    pub fixed_hyperparameters: FixedHyperparametersRow,
    pub prior_rows: Vec<PriorCalibrationRow>,
    pub truth_diagnostic_rows: Vec<TruthDiagnosticRow>,
    pub heldout_prediction_rows: Vec<HeldoutPredictionRow>,
    pub replicate_bin_metric_rows: Vec<ReplicateBinMetricRow>,
    pub aggregate_bin_metric_rows: Vec<AggregateBinMetricRow>,
    pub prior_roughness_rows: Vec<PriorRoughnessRow>,
    pub efficiency_rows: Vec<EfficiencyDiagnosticsRow>,
    pub observation_rows: Vec<ObservationRow>,
    pub sensor_design_rows: Vec<SensorDesignRow>,
}

#[derive(Debug, Clone)]
pub struct FixedHyperparametersRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub target_b_rms_tesla: f64,
    pub observation_std_tesla: f64,
    pub smooth_noise_replicates: usize,
    pub corrected_sample_truth_replicates: usize,
}

#[derive(Debug, Clone)]
pub struct PriorCalibrationRow {
    pub scenario: MagneticTruthSource,
    pub level: usize,
    pub alpha: MaternAlpha,
    pub model: MagneticPriorModelKind,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub dofs: usize,
    pub cells: usize,
    pub domain_volume: f64,
    pub target_b_rms_tesla: f64,
    pub raw_trace: f64,
    pub raw_mean_b2: f64,
    pub raw_hutchinson_trace: Option<f64>,
    pub raw_hutchinson_relative_standard_error: Option<f64>,
    pub tau_normalizer: f64,
    pub precision_scale: f64,
    pub diagonal_shift: f64,
    pub normalized_mean_b2: f64,
    pub exact_or_hutchinson_error: f64,
    pub normalization_source: String,
}

#[derive(Debug, Clone)]
pub struct TruthDiagnosticRow {
    pub scenario: MagneticTruthSource,
    pub truth_replicate: usize,
    pub truth_seed: u64,
    pub realized_b_rms_tesla: f64,
    pub mean_adjacent_b_jump2: f64,
    pub rms_adjacent_b_jump_tesla: f64,
}

#[derive(Debug, Clone)]
pub struct HeldoutPredictionRow {
    pub scenario: MagneticTruthSource,
    pub truth_replicate: usize,
    pub noise_replicate: usize,
    pub truth_seed: u64,
    pub noise_seed: u64,
    pub level: usize,
    pub alpha: MaternAlpha,
    pub model: MagneticPriorModelKind,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub split: String,
    pub bin: String,
    pub cell: usize,
    pub component: usize,
    pub operator_row: usize,
    pub truth_tesla: f64,
    pub observation_tesla: f64,
    pub posterior_mean_tesla: f64,
    pub posterior_sd_tesla: f64,
    pub predictive_sd_tesla: f64,
    pub standardized_residual: f64,
}

#[derive(Debug, Clone)]
pub struct ReplicateBinMetricRow {
    pub scenario: MagneticTruthSource,
    pub truth_replicate: usize,
    pub noise_replicate: usize,
    pub truth_seed: u64,
    pub noise_seed: u64,
    pub level: usize,
    pub alpha: MaternAlpha,
    pub model: MagneticPriorModelKind,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub split: String,
    pub bin: String,
    pub rows: usize,
    pub rmse_tesla: f64,
    pub nlpd: f64,
    pub standardized_rms: f64,
    pub coverage_95: f64,
}

#[derive(Debug, Clone)]
pub struct AggregateBinMetricRow {
    pub scenario: MagneticTruthSource,
    pub level: usize,
    pub alpha: MaternAlpha,
    pub model: MagneticPriorModelKind,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub split: String,
    pub bin: String,
    pub replicate_count: usize,
    pub rows_per_replicate: usize,
    pub rmse_tesla_mean: f64,
    pub rmse_tesla_se: f64,
    pub rmse_tesla_q05: f64,
    pub rmse_tesla_median: f64,
    pub rmse_tesla_q95: f64,
    pub nlpd_mean: f64,
    pub nlpd_se: f64,
    pub nlpd_q05: f64,
    pub nlpd_median: f64,
    pub nlpd_q95: f64,
    pub z_rms_mean: f64,
    pub z_rms_se: f64,
    pub z_rms_q05: f64,
    pub z_rms_median: f64,
    pub z_rms_q95: f64,
    pub coverage_95_mean: f64,
    pub coverage_95_se: f64,
    pub coverage_95_q05: f64,
    pub coverage_95_median: f64,
    pub coverage_95_q95: f64,
}

#[derive(Debug, Clone)]
pub struct PriorRoughnessRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub model: MagneticPriorModelKind,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub adjacent_cell_pairs: usize,
    pub jump_rows: usize,
    pub mean_adjacent_b_jump2: f64,
    pub rms_adjacent_b_jump: f64,
    pub estimator: String,
    pub qoi_rhs: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct EfficiencyDiagnosticsRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub model: MagneticPriorModelKind,
    pub practical_range_m: f64,
    pub kappa: f64,
    pub dofs: usize,
    pub b_rows: usize,
    pub precision_nnz: usize,
    pub precision_density: f64,
    pub trace_estimator: String,
    pub prior_factor_seconds: f64,
    pub prior_trace_seconds: f64,
    pub validation_posterior_factor_seconds: f64,
    pub validation_posterior_variance_seconds: f64,
    pub test_posterior_factor_seconds: f64,
    pub test_posterior_variance_seconds: f64,
    pub roughness_seconds: f64,
    pub prior_qoi_rhs: usize,
    pub validation_posterior_qoi_rhs: usize,
    pub test_posterior_qoi_rhs: usize,
    pub roughness_qoi_rhs: usize,
}

#[derive(Debug, Clone)]
pub struct ObservationRow {
    pub scenario: MagneticTruthSource,
    pub truth_replicate: usize,
    pub noise_replicate: usize,
    pub truth_seed: u64,
    pub noise_seed: u64,
    pub level: usize,
    pub alpha: MaternAlpha,
    pub split: String,
    pub bin: String,
    pub cell: usize,
    pub component: usize,
    pub operator_row: usize,
    pub truth_tesla: f64,
    pub observation_tesla: f64,
    pub noise_tesla: f64,
}

#[derive(Debug, Clone)]
pub struct SensorDesignRow {
    pub level: usize,
    pub split: String,
    pub bin: String,
    pub cell: usize,
    pub component: usize,
    pub operator_row: usize,
}

struct CubeWorkspace {
    topology: Complex,
    coords: MeshCoords,
    model: ReducedVectorPotentialMagnetostatic3d,
    cell_volumes: Vec<f64>,
    domain_volume: f64,
    b_operator: LinearMap,
}

struct PriorBuild {
    prior: GaussianPrior,
    raw_factored: FactoredGaussianPrior,
    raw_trace: PriorTraceEstimate,
    tau_normalizer: f64,
    precision_scale: f64,
    prior_factor_seconds: f64,
    prior_trace_seconds: f64,
    prior_qoi_rhs: usize,
    precision_nnz: usize,
    precision_density: f64,
    diagonal_shift: f64,
}

struct ModelRun {
    model: MagneticPriorModelKind,
    prior: PriorBuild,
    validation_plan: PosteriorPlan,
    test_plan: PosteriorPlan,
}

struct TransformedVarianceSolve {
    values: Vec<f64>,
    qoi_rhs: usize,
    elapsed_seconds: f64,
    estimator: &'static str,
}

struct PosteriorPlan {
    conditioning_rows: Vec<usize>,
    eval_rows: Vec<usize>,
    prepared: PreparedLinearGaussianModel,
    eval_operator: LinearMap,
    variance: Vec<f64>,
    factor_seconds: f64,
    variance_seconds: f64,
    qoi_rhs: usize,
}

struct PosteriorEvaluation {
    eval_rows: Vec<usize>,
    mean: Vec<f64>,
    variance: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct PriorTraceEstimate {
    value: f64,
    estimator: VarianceEstimator,
    relative_standard_error: Option<f64>,
}

impl From<&WeightedVarianceEstimate> for PriorTraceEstimate {
    fn from(estimate: &WeightedVarianceEstimate) -> Self {
        Self {
            value: estimate.weighted_trace,
            estimator: estimate.variances.estimator,
            relative_standard_error: estimate.weighted_trace_relative_standard_error,
        }
    }
}

struct SensorBin {
    label: &'static str,
    cells: Vec<usize>,
    rows: Vec<usize>,
}

struct SensorDesign {
    training_cells: Vec<usize>,
    validation_bins: Vec<SensorBin>,
    test_bins: Vec<SensorBin>,
    training_rows: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct PredictionMetrics {
    rmse_tesla: f64,
    nlpd: f64,
    standardized_rms: f64,
    coverage_95: f64,
}

pub fn compute_magnetic_prior_uq_comparison_report(
    config: MagneticPriorUqComparisonConfig,
) -> Result<MagneticPriorUqComparisonReport, String> {
    validate_config(&config)?;
    let workspace = build_cube_workspace(config.level)?;
    let sensor_design = deterministic_sensor_design(
        &workspace.topology,
        &workspace.coords,
        config.training_sensor_cells,
        config.validation_cells_per_bin,
        config.test_cells_per_bin,
    )?;
    let sensor_design_rows = sensor_design_rows(&config, &sensor_design);
    let practical_range_m = config.practical_range_m;
    let kappa = kappa_from_practical_range(practical_range_m);
    let fixed_hyperparameters = FixedHyperparametersRow {
        level: config.level,
        alpha: ALPHA,
        practical_range_m,
        kappa,
        target_b_rms_tesla: config.target_b_rms_tesla,
        observation_std_tesla: config.observation_std_tesla,
        smooth_noise_replicates: config.smooth_noise_replicates,
        corrected_sample_truth_replicates: config.corrected_sample_truth_replicates,
    };

    let validation_rows = rows_for_bins(&sensor_design.validation_bins);
    let test_conditioning_rows = concat_rows(&sensor_design.training_rows, &validation_rows);
    let test_rows = rows_for_bins(&sensor_design.test_bins);

    let mut prior_rows = Vec::new();
    let mut prior_roughness_rows = Vec::new();
    let mut efficiency_rows = Vec::new();
    let mut model_runs = Vec::new();

    for model in [
        MagneticPriorModelKind::OrdinaryPotential,
        MagneticPriorModelKind::SpectrumMatchedPotential,
    ] {
        let mut prior = build_prior(&config, &workspace, model, practical_range_m, kappa)?;
        let validation_plan = build_posterior_plan(
            &config,
            &workspace.b_operator,
            &prior,
            &sensor_design.training_rows,
            &validation_rows,
            config
                .rng_seed
                .wrapping_add(1_001)
                .wrapping_add(model_seed_offset(model)),
        )?;
        let test_plan = build_posterior_plan(
            &config,
            &workspace.b_operator,
            &prior,
            &test_conditioning_rows,
            &test_rows,
            config
                .rng_seed
                .wrapping_add(1_501)
                .wrapping_add(model_seed_offset(model)),
        )?;
        let roughness = compute_prior_roughness(
            &config,
            &workspace,
            model,
            practical_range_m,
            kappa,
            &mut prior,
        )?;

        for scenario in [
            MagneticTruthSource::SmoothManufactured,
            MagneticTruthSource::SpectrumMatchedPriorSample,
        ] {
            prior_rows.push(prior_calibration_row(
                &config,
                &workspace,
                scenario,
                model,
                practical_range_m,
                kappa,
                &prior,
            ));
        }

        efficiency_rows.push(EfficiencyDiagnosticsRow {
            level: config.level,
            alpha: ALPHA,
            model,
            practical_range_m,
            kappa,
            dofs: prior.prior.dimension(),
            b_rows: workspace.b_operator.output_dimension(),
            precision_nnz: prior.precision_nnz,
            precision_density: prior.precision_density,
            trace_estimator: if prior.raw_trace.estimator.is_exact() {
                "exact".to_string()
            } else {
                "hutchinson".to_string()
            },
            prior_factor_seconds: prior.prior_factor_seconds,
            prior_trace_seconds: prior.prior_trace_seconds,
            validation_posterior_factor_seconds: validation_plan.factor_seconds,
            validation_posterior_variance_seconds: validation_plan.variance_seconds,
            test_posterior_factor_seconds: test_plan.factor_seconds,
            test_posterior_variance_seconds: test_plan.variance_seconds,
            roughness_seconds: roughness.elapsed_seconds,
            prior_qoi_rhs: prior.prior_qoi_rhs,
            validation_posterior_qoi_rhs: validation_plan.qoi_rhs,
            test_posterior_qoi_rhs: test_plan.qoi_rhs,
            roughness_qoi_rhs: roughness.qoi_rhs,
        });
        prior_roughness_rows.push(roughness);
        model_runs.push(ModelRun {
            model,
            prior,
            validation_plan,
            test_plan,
        });
    }

    let corrected_prior_index = model_runs
        .iter()
        .position(|run| run.model == MagneticPriorModelKind::SpectrumMatchedPotential)
        .ok_or_else(|| "spectrum-matched prior was not built".to_string())?;
    let mut truth_diagnostic_rows = Vec::new();
    let mut observation_rows = Vec::new();
    let mut heldout_prediction_rows = Vec::new();
    let mut replicate_bin_metric_rows = Vec::new();

    let smooth_truth = build_smooth_truth(&config, &workspace)?;
    let smooth_truth_b = workspace
        .b_operator
        .apply(&smooth_truth)
        .map_err(|err| err.to_string())?;
    truth_diagnostic_rows.push(truth_diagnostic_row(
        MagneticTruthSource::SmoothManufactured,
        SMOOTH_TRUTH_REPLICATE,
        SMOOTH_TRUTH_SEED,
        &workspace,
        &smooth_truth_b,
    )?);
    for noise_replicate in 0..config.smooth_noise_replicates {
        let noise_seed = noise_seed(
            config.rng_seed,
            MagneticTruthSource::SmoothManufactured,
            SMOOTH_TRUTH_REPLICATE,
            noise_replicate,
        );
        let observations = build_observations(
            &config,
            MagneticTruthSource::SmoothManufactured,
            SMOOTH_TRUTH_REPLICATE,
            noise_replicate,
            SMOOTH_TRUTH_SEED,
            noise_seed,
            &smooth_truth_b,
            &sensor_design,
        )?;
        evaluate_replicate(
            &config,
            MagneticTruthSource::SmoothManufactured,
            SMOOTH_TRUTH_REPLICATE,
            noise_replicate,
            SMOOTH_TRUTH_SEED,
            noise_seed,
            practical_range_m,
            kappa,
            &sensor_design,
            &model_runs,
            &observations,
            &mut heldout_prediction_rows,
            &mut replicate_bin_metric_rows,
        )?;
        observation_rows.extend(observations);
    }

    for truth_replicate in 0..config.corrected_sample_truth_replicates {
        let truth_seed = truth_seed(
            config.rng_seed,
            MagneticTruthSource::SpectrumMatchedPriorSample,
            truth_replicate,
        );
        let truth = sample_normalized_prior_truth(
            &mut model_runs[corrected_prior_index].prior,
            truth_seed,
        )?;
        let truth_b = workspace
            .b_operator
            .apply(&truth)
            .map_err(|err| err.to_string())?;
        truth_diagnostic_rows.push(truth_diagnostic_row(
            MagneticTruthSource::SpectrumMatchedPriorSample,
            truth_replicate,
            truth_seed,
            &workspace,
            &truth_b,
        )?);
        let noise_replicate = 0;
        let noise_seed = noise_seed(
            config.rng_seed,
            MagneticTruthSource::SpectrumMatchedPriorSample,
            truth_replicate,
            noise_replicate,
        );
        let observations = build_observations(
            &config,
            MagneticTruthSource::SpectrumMatchedPriorSample,
            truth_replicate,
            noise_replicate,
            truth_seed,
            noise_seed,
            &truth_b,
            &sensor_design,
        )?;
        evaluate_replicate(
            &config,
            MagneticTruthSource::SpectrumMatchedPriorSample,
            truth_replicate,
            noise_replicate,
            truth_seed,
            noise_seed,
            practical_range_m,
            kappa,
            &sensor_design,
            &model_runs,
            &observations,
            &mut heldout_prediction_rows,
            &mut replicate_bin_metric_rows,
        )?;
        observation_rows.extend(observations);
    }

    let aggregate_bin_metric_rows = aggregate_bin_metrics(&replicate_bin_metric_rows)?;

    Ok(MagneticPriorUqComparisonReport {
        config,
        fixed_hyperparameters,
        prior_rows,
        truth_diagnostic_rows,
        heldout_prediction_rows,
        replicate_bin_metric_rows,
        aggregate_bin_metric_rows,
        prior_roughness_rows,
        efficiency_rows,
        observation_rows,
        sensor_design_rows,
    })
}

pub fn write_magnetic_prior_uq_comparison_outputs(
    report: &MagneticPriorUqComparisonReport,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    remove_legacy_output(out_dir.join("kappa_sweep.csv"))?;
    remove_legacy_output(out_dir.join("selected_hyperparameters.csv"))?;
    remove_legacy_output(out_dir.join("heldout_bin_metrics.csv"))?;
    write_fixed_hyperparameters_csv(report, &out_dir.join("fixed_hyperparameters.csv"))?;
    write_prior_calibration_csv(report, &out_dir.join("prior_calibration.csv"))?;
    write_truth_diagnostics_csv(report, &out_dir.join("truth_diagnostics.csv"))?;
    write_heldout_predictions_csv(report, &out_dir.join("heldout_predictions.csv"))?;
    write_replicate_bin_metrics_csv(report, &out_dir.join("replicate_bin_metrics.csv"))?;
    write_aggregate_bin_metrics_csv(report, &out_dir.join("aggregate_bin_metrics.csv"))?;
    write_sensor_design_csv(report, &out_dir.join("sensor_design.csv"))?;
    write_prior_roughness_csv(report, &out_dir.join("prior_roughness.csv"))?;
    write_efficiency_diagnostics_csv(report, &out_dir.join("efficiency_diagnostics.csv"))?;
    write_observations_csv(report, &out_dir.join("observations.csv"))
}

fn remove_legacy_output(path: impl AsRef<Path>) -> io::Result<()> {
    match fs::remove_file(path.as_ref()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn build_prior(
    config: &MagneticPriorUqComparisonConfig,
    workspace: &CubeWorkspace,
    model: MagneticPriorModelKind,
    practical_range_m: f64,
    kappa: f64,
) -> Result<PriorBuild, String> {
    let metric = workspace.coords.to_edge_lengths(&workspace.topology);
    let hodge = build_hodge_laplacian_1form(&workspace.topology, &metric);
    let mass_inverse = build_matern_mass_inverse_1form_with_coords(
        &workspace.topology,
        &workspace.coords,
        &metric,
        &hodge.mass_u,
        MaternMassInverse::Nc1ProjectedSparseInverse,
    );
    let mass_inverse = mass_inverse?;
    let raw_full_precision = match model {
        MagneticPriorModelKind::OrdinaryPotential => {
            build_matern_precision_1form_with_mass_inverse_for_alpha(
                &hodge,
                &mass_inverse,
                ALPHA,
                kappa,
                1.0,
            )
        }
        MagneticPriorModelKind::SpectrumMatchedPotential => {
            let system = build_matern_system_matrix_1form(&hodge, kappa);
            spectrum_matched_potential_precision(&system, &mass_inverse, ALPHA, kappa, 1.0)?
        }
    };
    let raw_precision = restrict_square_with_layout(&raw_full_precision, workspace.model.layout())?;
    let raw_precision = symmetrize_feec_csr(&raw_precision);
    let (raw_precision, _raw_prior, mut raw_factored, diagonal_shift, prior_factor_seconds) =
        factorize_raw_precision(raw_precision, model)?;

    let weights = cell_major_b_weights(&workspace.cell_volumes);
    let trace_start = Instant::now();
    let method = if raw_precision.nrows() <= config.exact_max_dofs {
        VarianceMethod::Exact
    } else {
        hutchinson_method(
            config,
            prior_seed(config.rng_seed, model, practical_range_m),
        )?
    };
    let raw_variance_trace = raw_factored
        .pushforward_weighted_variance_estimate(&workspace.b_operator, &weights, method)
        .map_err(|err| err.to_string())?;
    let prior_trace_seconds = trace_start.elapsed().as_secs_f64();
    let raw_trace = PriorTraceEstimate::from(&raw_variance_trace);
    let target_trace =
        config.target_b_rms_tesla * config.target_b_rms_tesla * workspace.domain_volume;
    let normalization = trace_normalization_from_target_trace(raw_trace.value, target_trace)?;
    let precision_scale = normalization.precision_scale;
    let prior_precision = scale_matrix(&raw_precision, precision_scale);
    let prior = GaussianPrior::new(
        vec![0.0; prior_precision.nrows()],
        sparse_mat_from_feec_csr(&prior_precision),
    )
    .map_err(|err| err.to_string())?;
    let prior_qoi_rhs = if raw_variance_trace.variances.estimator.is_exact() {
        workspace.b_operator.output_dimension()
    } else {
        raw_variance_trace.variances.sample_count
    };
    let n = prior.dimension();
    let precision_nnz = prior_precision.nnz();
    Ok(PriorBuild {
        prior,
        raw_factored,
        raw_trace,
        tau_normalizer: normalization.tau_multiplier,
        precision_scale,
        prior_factor_seconds,
        prior_trace_seconds,
        prior_qoi_rhs,
        precision_nnz,
        precision_density: precision_nnz as f64 / ((n * n) as f64),
        diagonal_shift,
    })
}

fn build_posterior_plan(
    config: &MagneticPriorUqComparisonConfig,
    b_operator: &LinearMap,
    prior: &PriorBuild,
    conditioning_rows: &[usize],
    eval_rows: &[usize],
    seed: u64,
) -> Result<PosteriorPlan, String> {
    let training_operator = b_operator
        .select_outputs(conditioning_rows)
        .map_err(|err| err.to_string())?;
    let factor_start = Instant::now();
    let prepared = LinearGaussianModelBuilder::new(prior.prior.clone())
        .observe(
            LinearObservation::new(
                training_operator,
                vec![0.0; conditioning_rows.len()],
                GaussianNoise::standard_deviation(config.observation_std_tesla)
                    .map_err(|err| err.to_string())?,
            )
            .map_err(|err| err.to_string())?,
        )
        .map_err(|err| err.to_string())?
        .prepare()
        .map_err(|err| format!("posterior precision factorization failed: {err}"))?;
    let posterior_factor_seconds = factor_start.elapsed().as_secs_f64();
    let eval_operator = b_operator
        .select_outputs(eval_rows)
        .map_err(|err| err.to_string())?;
    let mut zero_posterior = prepared.condition().map_err(|err| err.to_string())?;
    let variance_solve = transformed_variances(&mut zero_posterior, &eval_operator, config, seed)?;
    Ok(PosteriorPlan {
        conditioning_rows: conditioning_rows.to_vec(),
        eval_rows: eval_rows.to_vec(),
        prepared,
        eval_operator,
        variance: variance_solve.values,
        factor_seconds: posterior_factor_seconds,
        variance_seconds: variance_solve.elapsed_seconds,
        qoi_rhs: variance_solve.qoi_rhs,
    })
}

fn evaluate_with_posterior_plan(
    _config: &MagneticPriorUqComparisonConfig,
    plan: &PosteriorPlan,
    observations: &[ObservationRow],
) -> Result<PosteriorEvaluation, String> {
    let observation_lookup = observation_lookup(observations);
    let y = plan
        .conditioning_rows
        .iter()
        .map(|row| {
            observation_lookup
                .get(row)
                .map(|obs| obs.observation_tesla)
                .unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();
    if y.iter().any(|value| !value.is_finite()) {
        return Err("conditioning rows include a row without an observation".to_string());
    }
    let posterior_mean = plan
        .prepared
        .latent_mean_with_observation_values(&[y])
        .map_err(|err| err.to_string())?;
    let mean = plan
        .eval_operator
        .apply(&posterior_mean)
        .map_err(|err| err.to_string())?;
    Ok(PosteriorEvaluation {
        eval_rows: plan.eval_rows.clone(),
        mean,
        variance: plan.variance.clone(),
    })
}

fn factorize_raw_precision(
    precision: FeecCsr,
    model: MagneticPriorModelKind,
) -> Result<(FeecCsr, GaussianPrior, FactoredGaussianPrior, f64, f64), String> {
    let started = Instant::now();
    let prior = GaussianPrior::new(
        vec![0.0; precision.nrows()],
        sparse_mat_from_feec_csr(&precision),
    )
    .map_err(|err| err.to_string())?;
    match prior.factor() {
        Ok(factored) => {
            return Ok((
                precision,
                prior,
                factored,
                0.0,
                started.elapsed().as_secs_f64(),
            ))
        }
        Err(err) if model != MagneticPriorModelKind::SpectrumMatchedPotential => {
            return Err(format!(
                "{model} raw prior precision factorization failed: {err}"
            ));
        }
        Err(_) => {}
    }

    let base_shift = mean_abs_diagonal(&precision).max(1.0) * 1.0e-12;
    let mut shift = base_shift;
    let mut last_error = String::new();
    for _ in 0..8 {
        let shifted = add_feec_diagonal_shift(&precision, shift)?;
        let prior = GaussianPrior::new(
            vec![0.0; shifted.nrows()],
            sparse_mat_from_feec_csr(&shifted),
        )
        .map_err(|err| err.to_string())?;
        match prior.factor() {
            Ok(factored) => {
                return Ok((
                    shifted,
                    prior,
                    factored,
                    shift,
                    started.elapsed().as_secs_f64(),
                ));
            }
            Err(err) => {
                last_error = err.to_string();
                shift *= 100.0;
            }
        }
    }
    Err(format!(
        "{model} raw prior precision factorization failed after diagonal stabilization attempts; last error: {last_error}"
    ))
}

fn mean_abs_diagonal(matrix: &FeecCsr) -> f64 {
    if matrix.nrows() == 0 {
        return 0.0;
    }
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    diagonal.iter().map(|value| value.abs()).sum::<f64>() / diagonal.len() as f64
}

fn compute_prior_roughness(
    config: &MagneticPriorUqComparisonConfig,
    workspace: &CubeWorkspace,
    model: MagneticPriorModelKind,
    practical_range_m: f64,
    kappa: f64,
    prior: &mut PriorBuild,
) -> Result<PriorRoughnessRow, String> {
    let adjacent_pairs = adjacent_cell_pairs(&workspace.topology);
    let jump_operator = adjacent_b_jump_operator(&workspace.b_operator, &adjacent_pairs)?;
    let solve = factored_transformed_variances(
        &mut prior.raw_factored,
        &jump_operator,
        config,
        config
            .rng_seed
            .wrapping_add(2_001)
            .wrapping_add(model_seed_offset(model)),
    )?;
    let scaled = scale_vector(&solve.values, prior.precision_scale.recip());
    let mean_jump2 = scaled.iter().sum::<f64>() / scaled.len() as f64;
    Ok(PriorRoughnessRow {
        level: config.level,
        alpha: ALPHA,
        model,
        practical_range_m,
        kappa,
        adjacent_cell_pairs: adjacent_pairs.len(),
        jump_rows: jump_operator.output_dimension(),
        mean_adjacent_b_jump2: mean_jump2,
        rms_adjacent_b_jump: mean_jump2.max(0.0).sqrt(),
        estimator: solve.estimator.to_string(),
        qoi_rhs: solve.qoi_rhs,
        elapsed_seconds: solve.elapsed_seconds,
    })
}

fn metrics_for_rows(
    row_indices: &[usize],
    posterior: &PosteriorEvaluation,
    observations: &[ObservationRow],
    sigma: f64,
) -> Result<PredictionMetrics, String> {
    let eval_lookup = posterior_eval_lookup(posterior);
    let observation_lookup = observation_lookup(observations);
    let mut truth_sq = 0.0;
    let mut observed = Vec::with_capacity(row_indices.len());
    let mut predicted = Vec::with_capacity(row_indices.len());
    let mut latent_variances = Vec::with_capacity(row_indices.len());
    for row in row_indices {
        let eval = *eval_lookup
            .get(row)
            .ok_or_else(|| format!("posterior is missing evaluated row {row}"))?;
        let obs = observation_lookup
            .get(row)
            .ok_or_else(|| format!("missing observation for row {row}"))?;
        let mean = posterior.mean[eval];
        let variance = posterior.variance[eval].max(0.0);
        let truth_residual = mean - obs.truth_tesla;
        truth_sq += truth_residual * truth_residual;
        observed.push(obs.observation_tesla);
        predicted.push(mean);
        latent_variances.push(variance);
    }
    let count = row_indices.len();
    if count == 0 {
        return Err("prediction metrics require at least one row".to_string());
    }
    let diagnostics = gaussian_predictive_diagnostics_95(
        &observed,
        &predicted,
        &latent_variances,
        &vec![sigma * sigma; count],
    )
    .map_err(|err| err.to_string())?;
    Ok(PredictionMetrics {
        rmse_tesla: (truth_sq / count as f64).sqrt(),
        nlpd: diagnostics.mean_negative_log_predictive_density,
        standardized_rms: (diagnostics
            .standardized_residuals
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / count as f64)
            .sqrt(),
        coverage_95: diagnostics.empirical_coverage,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_replicate(
    config: &MagneticPriorUqComparisonConfig,
    scenario: MagneticTruthSource,
    truth_replicate: usize,
    noise_replicate: usize,
    truth_seed: u64,
    noise_seed: u64,
    practical_range_m: f64,
    kappa: f64,
    design: &SensorDesign,
    model_runs: &[ModelRun],
    observations: &[ObservationRow],
    prediction_rows: &mut Vec<HeldoutPredictionRow>,
    metric_rows: &mut Vec<ReplicateBinMetricRow>,
) -> Result<(), String> {
    for run in model_runs {
        let validation = evaluate_with_posterior_plan(config, &run.validation_plan, observations)?;
        for bin in &design.validation_bins {
            let metrics = metrics_for_rows(
                &bin.rows,
                &validation,
                observations,
                config.observation_std_tesla,
            )?;
            metric_rows.push(ReplicateBinMetricRow {
                scenario,
                truth_replicate,
                noise_replicate,
                truth_seed,
                noise_seed,
                level: config.level,
                alpha: ALPHA,
                model: run.model,
                practical_range_m,
                kappa,
                split: "validation".to_string(),
                bin: bin.label.to_string(),
                rows: bin.rows.len(),
                rmse_tesla: metrics.rmse_tesla,
                nlpd: metrics.nlpd,
                standardized_rms: metrics.standardized_rms,
                coverage_95: metrics.coverage_95,
            });
            prediction_rows.extend(prediction_rows_for_bin(
                config,
                scenario,
                truth_replicate,
                noise_replicate,
                truth_seed,
                noise_seed,
                run.model,
                practical_range_m,
                kappa,
                "validation",
                bin,
                &validation,
                observations,
            )?);
        }

        let test = evaluate_with_posterior_plan(config, &run.test_plan, observations)?;
        for bin in &design.test_bins {
            let metrics =
                metrics_for_rows(&bin.rows, &test, observations, config.observation_std_tesla)?;
            metric_rows.push(ReplicateBinMetricRow {
                scenario,
                truth_replicate,
                noise_replicate,
                truth_seed,
                noise_seed,
                level: config.level,
                alpha: ALPHA,
                model: run.model,
                practical_range_m,
                kappa,
                split: "test".to_string(),
                bin: bin.label.to_string(),
                rows: bin.rows.len(),
                rmse_tesla: metrics.rmse_tesla,
                nlpd: metrics.nlpd,
                standardized_rms: metrics.standardized_rms,
                coverage_95: metrics.coverage_95,
            });
            prediction_rows.extend(prediction_rows_for_bin(
                config,
                scenario,
                truth_replicate,
                noise_replicate,
                truth_seed,
                noise_seed,
                run.model,
                practical_range_m,
                kappa,
                "test",
                bin,
                &test,
                observations,
            )?);
        }
    }
    Ok(())
}

// Each prediction row records the seeds, split, and physical parameters needed
// to reproduce its held-out datum. The magnetic prior-mismatch smoke profile
// covers this artifact path.
#[allow(clippy::too_many_arguments)]
fn prediction_rows_for_bin(
    config: &MagneticPriorUqComparisonConfig,
    scenario: MagneticTruthSource,
    truth_replicate: usize,
    noise_replicate: usize,
    truth_seed: u64,
    noise_seed: u64,
    model: MagneticPriorModelKind,
    practical_range_m: f64,
    kappa: f64,
    split: &str,
    bin: &SensorBin,
    posterior: &PosteriorEvaluation,
    observations: &[ObservationRow],
) -> Result<Vec<HeldoutPredictionRow>, String> {
    let eval_lookup = posterior_eval_lookup(posterior);
    let observation_lookup = observation_lookup(observations);
    let mut rows = Vec::with_capacity(bin.rows.len());
    for row in &bin.rows {
        let eval = *eval_lookup
            .get(row)
            .ok_or_else(|| format!("posterior is missing evaluated row {row}"))?;
        let obs = observation_lookup
            .get(row)
            .ok_or_else(|| format!("missing observation for row {row}"))?;
        let posterior_mean = posterior.mean[eval];
        let posterior_sd = posterior.variance[eval].max(0.0).sqrt();
        let predictive_sd =
            (posterior.variance[eval].max(0.0) + config.observation_std_tesla.powi(2)).sqrt();
        rows.push(HeldoutPredictionRow {
            scenario,
            truth_replicate,
            noise_replicate,
            truth_seed,
            noise_seed,
            level: config.level,
            alpha: ALPHA,
            model,
            practical_range_m,
            kappa,
            split: split.to_string(),
            bin: bin.label.to_string(),
            cell: obs.cell,
            component: obs.component,
            operator_row: obs.operator_row,
            truth_tesla: obs.truth_tesla,
            observation_tesla: obs.observation_tesla,
            posterior_mean_tesla: posterior_mean,
            posterior_sd_tesla: posterior_sd,
            predictive_sd_tesla: predictive_sd,
            standardized_residual: (posterior_mean - obs.observation_tesla) / predictive_sd,
        });
    }
    Ok(rows)
}

fn prior_calibration_row(
    config: &MagneticPriorUqComparisonConfig,
    workspace: &CubeWorkspace,
    scenario: MagneticTruthSource,
    model: MagneticPriorModelKind,
    practical_range_m: f64,
    kappa: f64,
    prior: &PriorBuild,
) -> PriorCalibrationRow {
    let normalized_mean_b2 =
        prior.raw_trace.value / prior.precision_scale / workspace.domain_volume;
    let target_mean_b2 = config.target_b_rms_tesla * config.target_b_rms_tesla;
    let exact_or_hutchinson_error = if prior.raw_trace.estimator.is_exact() {
        ((normalized_mean_b2 - target_mean_b2) / target_mean_b2).abs()
    } else {
        prior.raw_trace.relative_standard_error.unwrap_or(f64::NAN)
    };
    PriorCalibrationRow {
        scenario,
        level: config.level,
        alpha: ALPHA,
        model,
        practical_range_m,
        kappa,
        dofs: prior.prior.dimension(),
        cells: workspace.cell_volumes.len(),
        domain_volume: workspace.domain_volume,
        target_b_rms_tesla: config.target_b_rms_tesla,
        raw_trace: prior.raw_trace.value,
        raw_mean_b2: prior.raw_trace.value / workspace.domain_volume,
        raw_hutchinson_trace: (!prior.raw_trace.estimator.is_exact())
            .then_some(prior.raw_trace.value),
        raw_hutchinson_relative_standard_error: prior.raw_trace.relative_standard_error,
        tau_normalizer: prior.tau_normalizer,
        precision_scale: prior.precision_scale,
        diagonal_shift: prior.diagonal_shift,
        normalized_mean_b2,
        exact_or_hutchinson_error,
        normalization_source: if prior.raw_trace.estimator.is_exact() {
            "exact".to_string()
        } else {
            "hutchinson".to_string()
        },
    }
}

fn build_cube_workspace(level: usize) -> Result<CubeWorkspace, String> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, level, TARGET_CUBE_SIDE_LENGTH_M);
    let (topology, coords) = mesh.compute_coord_complex();
    let material = NonlinearReluctivityLaw::new(1.0, 0.0)?;
    let model = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(material, EssentialBoundarySpec::default()),
    )?;
    let cell_volumes = cell_volumes(&topology, &coords)?;
    let domain_volume = cell_volumes.iter().sum::<f64>();
    let b_operator =
        build_reduced_magnetic_flux_density_operator_3d(&topology, &coords, model.layout())?;
    let b_operator = LinearMap::weighted_rows(b_operator.ncols, &b_operator.rows)
        .map_err(|err| err.to_string())?;
    Ok(CubeWorkspace {
        topology,
        coords,
        model,
        cell_volumes,
        domain_volume,
        b_operator,
    })
}

fn build_smooth_truth(
    config: &MagneticPriorUqComparisonConfig,
    workspace: &CubeWorkspace,
) -> Result<Vec<f64>, String> {
    let mut truth = smooth_edge_truth_3d(
        &workspace.model,
        &workspace.topology,
        &workspace.coords,
        1.0,
    );
    let b_values = workspace
        .b_operator
        .apply(&truth)
        .map_err(|err| err.to_string())?;
    let rms = volume_weighted_b_rms(&b_values, &workspace.cell_volumes)?;
    let scale = config.truth_b_rms_tesla / rms;
    for value in &mut truth {
        *value *= scale;
    }
    Ok(truth)
}

fn sample_normalized_prior_truth(prior: &mut PriorBuild, seed: u64) -> Result<Vec<f64>, String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let raw = prior
        .raw_factored
        .sample_cochain(&mut rng)
        .map_err(|err| err.to_string())?;
    Ok(scale_vector(&raw, prior.precision_scale.sqrt().recip()))
}

fn truth_diagnostic_row(
    scenario: MagneticTruthSource,
    truth_replicate: usize,
    truth_seed: u64,
    workspace: &CubeWorkspace,
    truth_b: &[f64],
) -> Result<TruthDiagnosticRow, String> {
    let realized_b_rms_tesla = volume_weighted_b_rms(truth_b, &workspace.cell_volumes)?;
    let adjacent_pairs = adjacent_cell_pairs(&workspace.topology);
    let mean_adjacent_b_jump2 = realized_adjacent_b_jump2(truth_b, &adjacent_pairs)?;
    Ok(TruthDiagnosticRow {
        scenario,
        truth_replicate,
        truth_seed,
        realized_b_rms_tesla,
        mean_adjacent_b_jump2,
        rms_adjacent_b_jump_tesla: mean_adjacent_b_jump2.max(0.0).sqrt(),
    })
}

fn realized_adjacent_b_jump2(
    b_values: &[f64],
    adjacent_pairs: &[(usize, usize)],
) -> Result<f64, String> {
    if adjacent_pairs.is_empty() {
        return Err(
            "adjacent jump diagnostic requires at least one adjacent cell pair".to_string(),
        );
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for &(lhs, rhs) in adjacent_pairs {
        for component in 0..3 {
            let lhs_row = 3 * lhs + component;
            let rhs_row = 3 * rhs + component;
            if lhs_row >= b_values.len() || rhs_row >= b_values.len() {
                return Err("adjacent jump row exceeds B output dimension".to_string());
            }
            let jump = b_values[lhs_row] - b_values[rhs_row];
            sum += jump * jump;
            count += 1;
        }
    }
    Ok(sum / count as f64)
}

fn deterministic_sensor_design(
    topology: &Complex,
    coords: &MeshCoords,
    training_cells: usize,
    validation_cells_per_bin: usize,
    test_cells_per_bin: usize,
) -> Result<SensorDesign, String> {
    let ranked = ranked_cells_by_center_distance(topology, coords);
    let required = training_cells + 3 * (validation_cells_per_bin + test_cells_per_bin);
    if required > ranked.len() {
        return Err(format!(
            "requested {required} sensor cells but level has only {} cells",
            ranked.len()
        ));
    }
    let mut selected = BTreeSet::new();
    let training = take_nearest_available(&ranked, &mut selected, training_cells);
    let available = ranked
        .iter()
        .filter(|(cell, _)| !selected.contains(cell))
        .copied()
        .collect::<Vec<_>>();
    let third = available.len() / 3;
    let bands = [
        ("near", &available[..third]),
        ("mid", &available[third..2 * third]),
        ("far", &available[2 * third..]),
    ];
    let mut validation_bins = Vec::with_capacity(3);
    let mut test_bins = Vec::with_capacity(3);
    for (label, band) in bands {
        let picked =
            take_evenly_spaced_band_cells(band, validation_cells_per_bin + test_cells_per_bin)?;
        let (mut validation, mut test) =
            split_interleaved_cells(&picked, validation_cells_per_bin, test_cells_per_bin)?;
        validation.sort_unstable();
        test.sort_unstable();
        for cell in validation.iter().chain(test.iter()) {
            if !selected.insert(*cell) {
                return Err(format!("sensor cell {cell} was selected more than once"));
            }
        }
        validation_bins.push(SensorBin {
            label,
            rows: cell_component_rows(&validation),
            cells: validation,
        });
        test_bins.push(SensorBin {
            label,
            rows: cell_component_rows(&test),
            cells: test,
        });
    }
    Ok(SensorDesign {
        training_rows: cell_component_rows(&training),
        validation_bins,
        test_bins,
        training_cells: training,
    })
}

// Observation provenance keeps independent truth/noise seeds and replicate labels;
// the magnetic smoke regression verifies the resulting observation inventory.
#[allow(clippy::too_many_arguments)]
fn build_observations(
    config: &MagneticPriorUqComparisonConfig,
    scenario: MagneticTruthSource,
    truth_replicate: usize,
    noise_replicate: usize,
    truth_seed: u64,
    noise_seed: u64,
    truth_b: &[f64],
    design: &SensorDesign,
) -> Result<Vec<ObservationRow>, String> {
    let capacity = design.training_rows.len()
        + rows_for_bins(&design.validation_bins).len()
        + design
            .test_bins
            .iter()
            .map(|bin| bin.rows.len())
            .sum::<usize>();
    let mut rows = Vec::with_capacity(capacity);
    let mut rng = StdRng::seed_from_u64(noise_seed);
    push_observation_rows(
        &mut rows,
        config,
        scenario,
        truth_replicate,
        noise_replicate,
        truth_seed,
        noise_seed,
        "training",
        "training",
        &design.training_cells,
        &design.training_rows,
        truth_b,
        &mut rng,
    )?;
    for bin in &design.validation_bins {
        push_observation_rows(
            &mut rows,
            config,
            scenario,
            truth_replicate,
            noise_replicate,
            truth_seed,
            noise_seed,
            "validation",
            bin.label,
            &bin.cells,
            &bin.rows,
            truth_b,
            &mut rng,
        )?;
    }
    for bin in &design.test_bins {
        push_observation_rows(
            &mut rows,
            config,
            scenario,
            truth_replicate,
            noise_replicate,
            truth_seed,
            noise_seed,
            "test",
            bin.label,
            &bin.cells,
            &bin.rows,
            truth_b,
            &mut rng,
        )?;
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn push_observation_rows(
    rows: &mut Vec<ObservationRow>,
    config: &MagneticPriorUqComparisonConfig,
    scenario: MagneticTruthSource,
    truth_replicate: usize,
    noise_replicate: usize,
    truth_seed: u64,
    noise_seed: u64,
    split: &str,
    bin: &str,
    cells: &[usize],
    operator_rows: &[usize],
    truth_b: &[f64],
    rng: &mut StdRng,
) -> Result<(), String> {
    for (cell, &row) in cells
        .iter()
        .flat_map(|cell| [*cell, *cell, *cell])
        .zip(operator_rows.iter())
    {
        if row >= truth_b.len() {
            return Err(format!(
                "observation row {row} exceeds B output dimension {}",
                truth_b.len()
            ));
        }
        let noise = config.observation_std_tesla * rng.sample::<f64, _>(StandardNormal);
        rows.push(ObservationRow {
            scenario,
            truth_replicate,
            noise_replicate,
            truth_seed,
            noise_seed,
            level: config.level,
            alpha: ALPHA,
            split: split.to_string(),
            bin: bin.to_string(),
            cell,
            component: row % 3,
            operator_row: row,
            truth_tesla: truth_b[row],
            observation_tesla: truth_b[row] + noise,
            noise_tesla: noise,
        });
    }
    Ok(())
}

fn transformed_variances(
    posterior: &mut Posterior,
    operator: &LinearMap,
    config: &MagneticPriorUqComparisonConfig,
    seed: u64,
) -> Result<TransformedVarianceSolve, String> {
    let start = Instant::now();
    let estimate = posterior
        .pushforward_variance_estimate(operator, variance_method(config, seed)?)
        .map_err(|err| err.to_string())?;
    Ok(TransformedVarianceSolve {
        values: estimate.values,
        qoi_rhs: if estimate.estimator.is_exact() {
            operator.output_dimension()
        } else {
            estimate.sample_count
        },
        elapsed_seconds: start.elapsed().as_secs_f64(),
        estimator: if estimate.estimator.is_exact() {
            "exact"
        } else {
            "hutchinson"
        },
    })
}

fn factored_transformed_variances(
    prior: &mut FactoredGaussianPrior,
    operator: &LinearMap,
    config: &MagneticPriorUqComparisonConfig,
    seed: u64,
) -> Result<TransformedVarianceSolve, String> {
    let start = Instant::now();
    let estimate = prior
        .pushforward_variance_estimate(operator, variance_method(config, seed)?)
        .map_err(|err| err.to_string())?;
    Ok(TransformedVarianceSolve {
        values: estimate.values,
        qoi_rhs: if estimate.estimator.is_exact() {
            operator.output_dimension()
        } else {
            estimate.sample_count
        },
        elapsed_seconds: start.elapsed().as_secs_f64(),
        estimator: if estimate.estimator.is_exact() {
            "exact"
        } else {
            "hutchinson"
        },
    })
}

fn adjacent_cell_pairs(topology: &Complex) -> Vec<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    for face in topology.facets().handle_iter() {
        let cells = face.cocells().map(|cell| cell.kidx()).collect::<Vec<_>>();
        if cells.len() == 2 {
            let a = cells[0].min(cells[1]);
            let b = cells[0].max(cells[1]);
            pairs.insert((a, b));
        }
    }
    pairs.into_iter().collect()
}

fn adjacent_b_jump_operator(
    b_operator: &LinearMap,
    adjacent_pairs: &[(usize, usize)],
) -> Result<LinearMap, String> {
    let mut rows = Vec::with_capacity(3 * adjacent_pairs.len());
    for &(lhs_cell, rhs_cell) in adjacent_pairs {
        for component in 0..3 {
            rows.push(vec![
                (3 * lhs_cell + component, 1.0),
                (3 * rhs_cell + component, -1.0),
            ]);
        }
    }
    LinearMap::weighted_rows(b_operator.output_dimension(), &rows)
        .and_then(|difference| difference.compose(b_operator))
        .map_err(|err| err.to_string())
}

fn observation_lookup(observations: &[ObservationRow]) -> BTreeMap<usize, &ObservationRow> {
    observations
        .iter()
        .map(|observation| (observation.operator_row, observation))
        .collect()
}

fn posterior_eval_lookup(posterior: &PosteriorEvaluation) -> BTreeMap<usize, usize> {
    posterior
        .eval_rows
        .iter()
        .copied()
        .enumerate()
        .map(|(index, row)| (row, index))
        .collect()
}

fn ranked_cells_by_center_distance(topology: &Complex, coords: &MeshCoords) -> Vec<(usize, f64)> {
    let mut ranked = top_cell_barycenters(topology, coords)
        .into_iter()
        .enumerate()
        .map(|(cell, point)| {
            let distance_to_center = point
                .iter()
                .map(|value| {
                    let d = value - 0.5 * TARGET_CUBE_SIDE_LENGTH_M;
                    d * d
                })
                .sum::<f64>();
            (cell, distance_to_center)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|lhs, rhs| {
        lhs.1
            .partial_cmp(&rhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(lhs.0.cmp(&rhs.0))
    });
    ranked
}

fn take_nearest_available(
    ranked: &[(usize, f64)],
    selected: &mut BTreeSet<usize>,
    count: usize,
) -> Vec<usize> {
    let mut cells = Vec::with_capacity(count);
    for &(cell, _) in ranked {
        if selected.insert(cell) {
            cells.push(cell);
            if cells.len() == count {
                break;
            }
        }
    }
    cells
}

fn take_evenly_spaced_band_cells(
    ranked_band: &[(usize, f64)],
    count: usize,
) -> Result<Vec<usize>, String> {
    if count > ranked_band.len() {
        return Err(format!(
            "requested {count} cells from a radial band with only {} available cells",
            ranked_band.len()
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let cells = (0..count)
        .map(|index| {
            let scaled =
                ((index as f64 + 0.5) * ranked_band.len() as f64 / count as f64).floor() as usize;
            ranked_band[scaled.min(ranked_band.len() - 1)].0
        })
        .collect::<Vec<_>>();
    if cells.iter().copied().collect::<BTreeSet<_>>().len() != count {
        return Err("radial band sampling produced duplicate cells".to_string());
    }
    Ok(cells)
}

fn split_interleaved_cells(
    cells: &[usize],
    validation_count: usize,
    test_count: usize,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if cells.len() != validation_count + test_count {
        return Err("interleaved sensor split received an inconsistent cell count".to_string());
    }
    let mut validation = Vec::with_capacity(validation_count);
    let mut test = Vec::with_capacity(test_count);
    for (index, cell) in cells.iter().copied().enumerate() {
        if (index % 2 == 0 && validation.len() < validation_count) || test.len() == test_count {
            validation.push(cell);
        } else if test.len() < test_count {
            test.push(cell);
        } else {
            validation.push(cell);
        }
    }
    Ok((validation, test))
}

fn cell_component_rows(cells: &[usize]) -> Vec<usize> {
    cells
        .iter()
        .flat_map(|cell| [3 * *cell, 3 * *cell + 1, 3 * *cell + 2])
        .collect()
}

fn concat_rows(first: &[usize], second: &[usize]) -> Vec<usize> {
    first
        .iter()
        .chain(second.iter())
        .copied()
        .collect::<Vec<_>>()
}

fn rows_for_bins(bins: &[SensorBin]) -> Vec<usize> {
    bins.iter()
        .flat_map(|bin| bin.rows.iter().copied())
        .collect()
}

fn sensor_design_rows(
    config: &MagneticPriorUqComparisonConfig,
    design: &SensorDesign,
) -> Vec<SensorDesignRow> {
    let mut rows = Vec::new();
    push_sensor_design_rows(
        &mut rows,
        config.level,
        "training",
        "training",
        &design.training_cells,
        &design.training_rows,
    );
    for bin in &design.validation_bins {
        push_sensor_design_rows(
            &mut rows,
            config.level,
            "validation",
            bin.label,
            &bin.cells,
            &bin.rows,
        );
    }
    for bin in &design.test_bins {
        push_sensor_design_rows(
            &mut rows,
            config.level,
            "test",
            bin.label,
            &bin.cells,
            &bin.rows,
        );
    }
    rows
}

fn push_sensor_design_rows(
    rows: &mut Vec<SensorDesignRow>,
    level: usize,
    split: &str,
    bin: &str,
    cells: &[usize],
    operator_rows: &[usize],
) {
    for (cell, &row) in cells
        .iter()
        .flat_map(|cell| [*cell, *cell, *cell])
        .zip(operator_rows.iter())
    {
        rows.push(SensorDesignRow {
            level,
            split: split.to_string(),
            bin: bin.to_string(),
            cell,
            component: row % 3,
            operator_row: row,
        });
    }
}

fn smooth_edge_truth_3d(
    model: &ReducedVectorPotentialMagnetostatic3d,
    topology: &Complex,
    coords: &MeshCoords,
    scale: f64,
) -> Vec<f64> {
    let mut full = vec![0.0; topology.nsimplices(1)];
    for edge in topology.edges().handle_iter() {
        let [v0, v1]: [usize; 2] = edge.vertices.clone().try_into().unwrap();
        let p0 = coords.coord(v0);
        let p1 = coords.coord(v1);
        let midpoint = [
            0.5 * (p0[0] + p1[0]),
            0.5 * (p0[1] + p1[1]),
            0.5 * (p0[2] + p1[2]),
        ];
        let tangent = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let vector_potential = [
            scale * (PI * midpoint[1]).sin() * (PI * midpoint[2]).sin(),
            scale * (PI * midpoint[2]).sin() * (PI * midpoint[0]).sin(),
            scale * (PI * midpoint[0]).sin() * (PI * midpoint[1]).sin(),
        ];
        full[edge.kidx()] = vector_potential[0] * tangent[0]
            + vector_potential[1] * tangent[1]
            + vector_potential[2] * tangent[2];
    }
    model
        .layout()
        .active_dofs
        .iter()
        .map(|&edge| full[edge])
        .collect()
}

fn top_cell_barycenters(topology: &Complex, coords: &MeshCoords) -> Vec<[f64; 3]> {
    topology
        .cells()
        .handle_iter()
        .map(|cell| {
            let barycenter = cell.coord_simplex(coords).barycenter();
            [barycenter[0], barycenter[1], barycenter[2]]
        })
        .collect()
}

fn cell_volumes(topology: &Complex, coords: &MeshCoords) -> Result<Vec<f64>, String> {
    let mut volumes = Vec::with_capacity(topology.cells().len());
    for cell in topology.cells().handle_iter() {
        let volume = cell.coord_simplex(coords).vol();
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "cell {} has non-positive finite volume {volume}",
                cell.kidx()
            ));
        }
        volumes.push(volume);
    }
    Ok(volumes)
}

fn volume_weighted_b_rms(b_values: &[f64], cell_volumes: &[f64]) -> Result<f64, String> {
    if b_values.len() != 3 * cell_volumes.len() {
        return Err(format!(
            "B value length {} does not match 3 * cell count {}",
            b_values.len(),
            cell_volumes.len()
        ));
    }
    let volume = cell_volumes.iter().sum::<f64>();
    if !volume.is_finite() || volume <= 0.0 {
        return Err("domain volume must be finite and positive".to_string());
    }
    let sum = cell_volumes
        .iter()
        .enumerate()
        .map(|(cell, volume)| {
            volume
                * (b_values[3 * cell] * b_values[3 * cell]
                    + b_values[3 * cell + 1] * b_values[3 * cell + 1]
                    + b_values[3 * cell + 2] * b_values[3 * cell + 2])
        })
        .sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("truth B RMS is not finite and positive".to_string());
    }
    Ok((sum / volume).sqrt())
}

fn cell_major_b_weights(cell_volumes: &[f64]) -> Vec<f64> {
    cell_volumes
        .iter()
        .flat_map(|volume| [*volume, *volume, *volume])
        .collect()
}

fn scale_vector(values: &[f64], scale: f64) -> Vec<f64> {
    values.iter().map(|value| *value * scale).collect()
}

fn hutchinson_method(
    config: &MagneticPriorUqComparisonConfig,
    rng_seed: u64,
) -> Result<VarianceMethod, String> {
    Ok(VarianceMethod::Hutchinson(
        HutchinsonVarianceConfig::new(
            config.hutchinson_probes,
            config.hutchinson_batches,
            rng_seed,
        )
        .map_err(|err| err.to_string())?
        .distribution(ProbeDistribution::Rademacher),
    ))
}

fn variance_method(
    config: &MagneticPriorUqComparisonConfig,
    rng_seed: u64,
) -> Result<VarianceMethod, String> {
    VarianceMethod::auto(
        config.exact_max_dofs,
        HutchinsonVarianceConfig::new(
            config.hutchinson_probes,
            config.hutchinson_batches,
            rng_seed,
        )
        .map_err(|err| err.to_string())?
        .distribution(ProbeDistribution::Rademacher),
    )
    .map_err(|err| err.to_string())
}

fn kappa_from_practical_range(range: f64) -> f64 {
    (8.0_f64).sqrt() / range
}

fn model_seed_offset(model: MagneticPriorModelKind) -> u64 {
    match model {
        MagneticPriorModelKind::OrdinaryPotential => 11,
        MagneticPriorModelKind::SpectrumMatchedPotential => 29,
    }
}

fn scenario_seed_offset(scenario: MagneticTruthSource) -> u64 {
    match scenario {
        MagneticTruthSource::SmoothManufactured => 10_000,
        MagneticTruthSource::SpectrumMatchedPriorSample => 20_000,
    }
}

fn truth_seed(base: u64, scenario: MagneticTruthSource, truth_replicate: usize) -> u64 {
    base.wrapping_add(scenario_seed_offset(scenario))
        .wrapping_add((truth_replicate as u64).wrapping_mul(1_009))
}

fn noise_seed(
    base: u64,
    scenario: MagneticTruthSource,
    truth_replicate: usize,
    noise_replicate: usize,
) -> u64 {
    base.wrapping_add(scenario_seed_offset(scenario))
        .wrapping_add(50_000)
        .wrapping_add((truth_replicate as u64).wrapping_mul(1_009))
        .wrapping_add((noise_replicate as u64).wrapping_mul(9_176))
}

fn prior_seed(base: u64, model: MagneticPriorModelKind, range: f64) -> u64 {
    base.wrapping_add(model_seed_offset(model))
        .wrapping_add(range.to_bits().rotate_left(17))
}

fn validate_config(config: &MagneticPriorUqComparisonConfig) -> Result<(), String> {
    if config.level == 0 {
        return Err("cube mesh level must be positive".to_string());
    }
    validate_positive(config.practical_range_m, "practical_range_m")?;
    validate_positive(config.target_b_rms_tesla, "target_b_rms_tesla")?;
    validate_positive(config.truth_b_rms_tesla, "truth_b_rms_tesla")?;
    validate_positive(config.observation_std_tesla, "observation_std_tesla")?;
    if config.training_sensor_cells == 0 {
        return Err("training_sensor_cells must be positive".to_string());
    }
    if config.validation_cells_per_bin == 0 {
        return Err("validation_cells_per_bin must be positive".to_string());
    }
    if config.test_cells_per_bin == 0 {
        return Err("test_cells_per_bin must be positive".to_string());
    }
    if config.smooth_noise_replicates == 0 {
        return Err("smooth_noise_replicates must be positive".to_string());
    }
    if config.corrected_sample_truth_replicates == 0 {
        return Err("corrected_sample_truth_replicates must be positive".to_string());
    }
    if config.hutchinson_probes == 0 {
        return Err("hutchinson_probes must be positive".to_string());
    }
    if config.hutchinson_batches == 0 {
        return Err("hutchinson_batches must be positive".to_string());
    }
    Ok(())
}

fn validate_positive(value: f64, name: &str) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be finite and positive, got {value}"))
    }
}

fn aggregate_bin_metrics(
    replicate_rows: &[ReplicateBinMetricRow],
) -> Result<Vec<AggregateBinMetricRow>, String> {
    let mut grouped = BTreeMap::<
        (MagneticTruthSource, MagneticPriorModelKind, String, String),
        Vec<&ReplicateBinMetricRow>,
    >::new();
    for row in replicate_rows {
        grouped
            .entry((row.scenario, row.model, row.split.clone(), row.bin.clone()))
            .or_default()
            .push(row);
    }
    let mut aggregates = Vec::with_capacity(grouped.len());
    for ((scenario, model, split, bin), rows) in grouped {
        let rmse = rows.iter().map(|row| row.rmse_tesla).collect::<Vec<_>>();
        let nlpd = rows.iter().map(|row| row.nlpd).collect::<Vec<_>>();
        let z_rms = rows
            .iter()
            .map(|row| row.standardized_rms)
            .collect::<Vec<_>>();
        let coverage = rows.iter().map(|row| row.coverage_95).collect::<Vec<_>>();
        let first = rows
            .first()
            .ok_or_else(|| "aggregate group unexpectedly empty".to_string())?;
        aggregates.push(AggregateBinMetricRow {
            scenario,
            level: first.level,
            alpha: first.alpha,
            model,
            practical_range_m: first.practical_range_m,
            kappa: first.kappa,
            split,
            bin,
            replicate_count: rows.len(),
            rows_per_replicate: first.rows,
            rmse_tesla_mean: mean(&rmse)?,
            rmse_tesla_se: standard_error(&rmse)?,
            rmse_tesla_q05: quantile(&rmse, 0.05)?,
            rmse_tesla_median: quantile(&rmse, 0.50)?,
            rmse_tesla_q95: quantile(&rmse, 0.95)?,
            nlpd_mean: mean(&nlpd)?,
            nlpd_se: standard_error(&nlpd)?,
            nlpd_q05: quantile(&nlpd, 0.05)?,
            nlpd_median: quantile(&nlpd, 0.50)?,
            nlpd_q95: quantile(&nlpd, 0.95)?,
            z_rms_mean: mean(&z_rms)?,
            z_rms_se: standard_error(&z_rms)?,
            z_rms_q05: quantile(&z_rms, 0.05)?,
            z_rms_median: quantile(&z_rms, 0.50)?,
            z_rms_q95: quantile(&z_rms, 0.95)?,
            coverage_95_mean: mean(&coverage)?,
            coverage_95_se: standard_error(&coverage)?,
            coverage_95_q05: quantile(&coverage, 0.05)?,
            coverage_95_median: quantile(&coverage, 0.50)?,
            coverage_95_q95: quantile(&coverage, 0.95)?,
        });
    }
    Ok(aggregates)
}

fn mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("mean requires at least one value".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("mean received a non-finite value".to_string());
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn standard_error(values: &[f64]) -> Result<f64, String> {
    if values.len() <= 1 {
        return Ok(0.0);
    }
    let mean = mean(values)?;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Ok((variance / values.len() as f64).sqrt())
}

fn quantile(values: &[f64], probability: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("quantile requires at least one value".to_string());
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(format!("invalid quantile probability {probability}"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("quantile received a non-finite value".to_string());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        Ok(sorted[lower])
    } else {
        let fraction = position - lower as f64;
        Ok(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
    }
}

fn write_fixed_hyperparameters_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,alpha,practical_range_m,kappa,target_b_rms_tesla,observation_std_tesla,smooth_noise_replicates,corrected_sample_truth_replicates"
    )?;
    let row = &report.fixed_hyperparameters;
    writeln!(
        writer,
        "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{}",
        row.level,
        row.alpha.as_u32(),
        row.practical_range_m,
        row.kappa,
        row.target_b_rms_tesla,
        row.observation_std_tesla,
        row.smooth_noise_replicates,
        row.corrected_sample_truth_replicates
    )?;
    writer.flush()
}

fn write_prior_calibration_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,model,level,alpha,practical_range_m,kappa,dofs,cells,domain_volume,target_b_rms_tesla,raw_trace,raw_mean_b2,raw_hutchinson_trace,raw_hutchinson_relative_standard_error,tau_normalizer,precision_scale,diagonal_shift,normalized_mean_b2,exact_or_hutchinson_error,normalization_source"
    )?;
    for row in &report.prior_rows {
        writeln!(
            writer,
            "{},,,,,{},{},{},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.scenario,
            row.model,
            row.level,
            row.alpha.as_u32(),
            row.practical_range_m,
            row.kappa,
            row.dofs,
            row.cells,
            row.domain_volume,
            row.target_b_rms_tesla,
            row.raw_trace,
            row.raw_mean_b2,
            optional_f64(row.raw_hutchinson_trace),
            optional_f64(row.raw_hutchinson_relative_standard_error),
            row.tau_normalizer,
            row.precision_scale,
            row.diagonal_shift,
            row.normalized_mean_b2,
            row.exact_or_hutchinson_error,
            row.normalization_source
        )?;
    }
    writer.flush()
}

fn write_truth_diagnostics_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,model,realized_b_rms_tesla,mean_adjacent_b_jump2,rms_adjacent_b_jump_tesla"
    )?;
    for row in &report.truth_diagnostic_rows {
        writeln!(
            writer,
            "{},{},,{},,,{:.16e},{:.16e},{:.16e}",
            row.scenario,
            row.truth_replicate,
            row.truth_seed,
            row.realized_b_rms_tesla,
            row.mean_adjacent_b_jump2,
            row.rms_adjacent_b_jump_tesla
        )?;
    }
    writer.flush()
}

fn write_heldout_predictions_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,level,alpha,model,practical_range_m,kappa,split,bin,cell,component,operator_row,truth_tesla,observation_tesla,posterior_mean_tesla,posterior_sd_tesla,predictive_sd_tesla,standardized_residual"
    )?;
    for row in &report.heldout_prediction_rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{:.16e},{:.16e},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            row.scenario,
            row.truth_replicate,
            row.noise_replicate,
            row.truth_seed,
            row.noise_seed,
            row.level,
            row.alpha.as_u32(),
            row.model,
            row.practical_range_m,
            row.kappa,
            row.split,
            row.bin,
            row.cell,
            row.component,
            row.operator_row,
            row.truth_tesla,
            row.observation_tesla,
            row.posterior_mean_tesla,
            row.posterior_sd_tesla,
            row.predictive_sd_tesla,
            row.standardized_residual
        )?;
    }
    writer.flush()
}

fn write_replicate_bin_metrics_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,level,alpha,model,practical_range_m,kappa,split,bin,rows,rmse_tesla,nlpd,z_rms,coverage_95"
    )?;
    for row in &report.replicate_bin_metric_rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{:.16e},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e}",
            row.scenario,
            row.truth_replicate,
            row.noise_replicate,
            row.truth_seed,
            row.noise_seed,
            row.level,
            row.alpha.as_u32(),
            row.model,
            row.practical_range_m,
            row.kappa,
            row.split,
            row.bin,
            row.rows,
            row.rmse_tesla,
            row.nlpd,
            row.standardized_rms,
            row.coverage_95
        )?;
    }
    writer.flush()
}

fn write_aggregate_bin_metrics_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,model,level,alpha,practical_range_m,kappa,split,bin,replicate_count,rows_per_replicate,rmse_tesla_mean,rmse_tesla_se,rmse_tesla_q05,rmse_tesla_median,rmse_tesla_q95,nlpd_mean,nlpd_se,nlpd_q05,nlpd_median,nlpd_q95,z_rms_mean,z_rms_se,z_rms_q05,z_rms_median,z_rms_q95,coverage_95_mean,coverage_95_se,coverage_95_q05,coverage_95_median,coverage_95_q95"
    )?;
    for row in &report.aggregate_bin_metric_rows {
        writeln!(
            writer,
            "{},,,,,{},{},{},{:.16e},{:.16e},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            row.scenario,
            row.model,
            row.level,
            row.alpha.as_u32(),
            row.practical_range_m,
            row.kappa,
            row.split,
            row.bin,
            row.replicate_count,
            row.rows_per_replicate,
            row.rmse_tesla_mean,
            row.rmse_tesla_se,
            row.rmse_tesla_q05,
            row.rmse_tesla_median,
            row.rmse_tesla_q95,
            row.nlpd_mean,
            row.nlpd_se,
            row.nlpd_q05,
            row.nlpd_median,
            row.nlpd_q95,
            row.z_rms_mean,
            row.z_rms_se,
            row.z_rms_q05,
            row.z_rms_median,
            row.z_rms_q95,
            row.coverage_95_mean,
            row.coverage_95_se,
            row.coverage_95_q05,
            row.coverage_95_median,
            row.coverage_95_q95
        )?;
    }
    writer.flush()
}

fn write_sensor_design_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,model,level,split,bin,cell,component,operator_row"
    )?;
    for row in &report.sensor_design_rows {
        writeln!(
            writer,
            "all,,,,,,{},{},{},{},{},{}",
            row.level, row.split, row.bin, row.cell, row.component, row.operator_row
        )?;
    }
    writer.flush()
}

fn write_prior_roughness_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,alpha,model,practical_range_m,kappa,adjacent_cell_pairs,jump_rows,mean_adjacent_b_jump2,rms_adjacent_b_jump,estimator,qoi_rhs,elapsed_seconds"
    )?;
    for row in &report.prior_roughness_rows {
        writeln!(
            writer,
            "{},{},{},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{},{},{:.16e}",
            row.level,
            row.alpha.as_u32(),
            row.model,
            row.practical_range_m,
            row.kappa,
            row.adjacent_cell_pairs,
            row.jump_rows,
            row.mean_adjacent_b_jump2,
            row.rms_adjacent_b_jump,
            row.estimator,
            row.qoi_rhs,
            row.elapsed_seconds
        )?;
    }
    writer.flush()
}

fn write_efficiency_diagnostics_csv(
    report: &MagneticPriorUqComparisonReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,model,level,alpha,practical_range_m,kappa,dofs,b_rows,precision_nnz,precision_density,trace_estimator,prior_factor_seconds,prior_trace_seconds,validation_posterior_factor_seconds,validation_posterior_variance_seconds,test_posterior_factor_seconds,test_posterior_variance_seconds,roughness_seconds,prior_qoi_rhs,validation_posterior_qoi_rhs,test_posterior_qoi_rhs,roughness_qoi_rhs"
    )?;
    for row in &report.efficiency_rows {
        writeln!(
            writer,
            "all,,,,,{},{},{},{:.16e},{:.16e},{},{},{},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{}",
            row.model,
            row.level,
            row.alpha.as_u32(),
            row.practical_range_m,
            row.kappa,
            row.dofs,
            row.b_rows,
            row.precision_nnz,
            row.precision_density,
            row.trace_estimator,
            row.prior_factor_seconds,
            row.prior_trace_seconds,
            row.validation_posterior_factor_seconds,
            row.validation_posterior_variance_seconds,
            row.test_posterior_factor_seconds,
            row.test_posterior_variance_seconds,
            row.roughness_seconds,
            row.prior_qoi_rhs,
            row.validation_posterior_qoi_rhs,
            row.test_posterior_qoi_rhs,
            row.roughness_qoi_rhs
        )?;
    }
    writer.flush()
}

fn write_observations_csv(report: &MagneticPriorUqComparisonReport, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,truth_replicate,noise_replicate,truth_seed,noise_seed,level,alpha,split,bin,cell,component,operator_row,truth_tesla,observation_tesla,noise_tesla"
    )?;
    for row in &report.observation_rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e}",
            row.scenario,
            row.truth_replicate,
            row.noise_replicate,
            row.truth_seed,
            row.noise_seed,
            row.level,
            row.alpha.as_u32(),
            row.split,
            row.bin,
            row.cell,
            row.component,
            row.operator_row,
            row.truth_tesla,
            row.observation_tesla,
            row.noise_tesla
        )?;
    }
    writer.flush()
}

fn optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.16e}"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "heavy-tests")]
    fn tiny_config() -> MagneticPriorUqComparisonConfig {
        MagneticPriorUqComparisonConfig {
            level: 2,
            practical_range_m: DEFAULT_PRACTICAL_RANGE_M,
            training_sensor_cells: 1,
            validation_cells_per_bin: 1,
            test_cells_per_bin: 1,
            smooth_noise_replicates: 2,
            corrected_sample_truth_replicates: 2,
            exact_max_dofs: 10_000,
            hutchinson_probes: 8,
            hutchinson_batches: 2,
            ..MagneticPriorUqComparisonConfig::default()
        }
    }

    #[test]
    fn magnetic_prior_uq_comparison_models_parse() {
        assert_eq!(
            "ordinary_potential"
                .parse::<MagneticPriorModelKind>()
                .unwrap(),
            MagneticPriorModelKind::OrdinaryPotential
        );
        assert_eq!(
            "spectrum_matched_potential"
                .parse::<MagneticPriorModelKind>()
                .unwrap(),
            MagneticPriorModelKind::SpectrumMatchedPotential
        );
        assert_eq!(
            "corrected_prior_sample_truth"
                .parse::<MagneticTruthSource>()
                .unwrap(),
            MagneticTruthSource::SpectrumMatchedPriorSample
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_uses_fixed_alpha_and_range() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let first = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("first report should build");
        let second = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("second report should build");
        assert_eq!(first.fixed_hyperparameters.alpha, MaternAlpha::Two);
        assert_eq!(first.fixed_hyperparameters.practical_range_m, 0.75);
        assert_eq!(
            first.fixed_hyperparameters.kappa,
            kappa_from_practical_range(0.75)
        );
        assert_eq!(
            first.fixed_hyperparameters.kappa,
            second.fixed_hyperparameters.kappa
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_normalizes_b_rms_for_both_priors() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let report = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("report should build");
        assert_eq!(report.prior_rows.len(), 4);
        let target = report.config.target_b_rms_tesla * report.config.target_b_rms_tesla;
        for row in &report.prior_rows {
            assert_eq!(row.alpha, MaternAlpha::Two);
            assert!(
                (row.normalized_mean_b2 - target).abs() < 1e-8 * target.max(1.0),
                "model={} normalized mean B2 {} target {}",
                row.model,
                row.normalized_mean_b2,
                target
            );
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_reuses_observations_across_models() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let report = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("report should build");
        let mut by_model = BTreeMap::<
            MagneticPriorModelKind,
            BTreeMap<
                (
                    MagneticTruthSource,
                    usize,
                    usize,
                    String,
                    String,
                    usize,
                    usize,
                ),
                f64,
            >,
        >::new();
        for row in &report.heldout_prediction_rows {
            by_model.entry(row.model).or_default().insert(
                (
                    row.scenario,
                    row.truth_replicate,
                    row.noise_replicate,
                    row.split.clone(),
                    row.bin.clone(),
                    row.cell,
                    row.component,
                ),
                row.observation_tesla,
            );
        }
        let ordinary = by_model
            .get(&MagneticPriorModelKind::OrdinaryPotential)
            .expect("ordinary rows");
        let corrected = by_model
            .get(&MagneticPriorModelKind::SpectrumMatchedPotential)
            .expect("corrected rows");
        assert_eq!(ordinary.len(), corrected.len());
        for (key, value) in ordinary {
            assert_eq!(Some(value), corrected.get(key));
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_sensor_splits_are_disjoint() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let report = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("report should build");
        let mut by_cell = BTreeMap::<usize, BTreeSet<(String, String)>>::new();
        for row in &report.sensor_design_rows {
            by_cell
                .entry(row.cell)
                .or_default()
                .insert((row.split.clone(), row.bin.clone()));
        }
        assert!(by_cell.values().all(|owners| owners.len() == 1));
        for split in ["training", "validation", "test"] {
            assert!(report
                .sensor_design_rows
                .iter()
                .any(|row| row.split == split));
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_emits_configurable_replicates() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let report = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("report should build");
        let smooth = report
            .observation_rows
            .iter()
            .filter(|row| row.scenario == MagneticTruthSource::SmoothManufactured)
            .map(|row| (row.truth_replicate, row.noise_replicate))
            .collect::<BTreeSet<_>>();
        let sampled = report
            .truth_diagnostic_rows
            .iter()
            .filter(|row| row.scenario == MagneticTruthSource::SpectrumMatchedPriorSample)
            .map(|row| row.truth_replicate)
            .collect::<BTreeSet<_>>();
        assert_eq!(smooth.len(), report.config.smooth_noise_replicates);
        assert_eq!(
            sampled.len(),
            report.config.corrected_sample_truth_replicates
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_sampled_truth_is_not_rescaled_per_draw() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let report = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("report should build");
        let target = report.config.target_b_rms_tesla;
        let sampled = report
            .truth_diagnostic_rows
            .iter()
            .filter(|row| row.scenario == MagneticTruthSource::SpectrumMatchedPriorSample)
            .collect::<Vec<_>>();
        assert_eq!(
            sampled.len(),
            report.config.corrected_sample_truth_replicates
        );
        assert!(sampled
            .iter()
            .any(|row| (row.realized_b_rms_tesla - target).abs() > 1.0e-8));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_prior_uq_comparison_metrics_are_finite() {
        let _guard = crate::test_util::lock_feec_harmonic_tests();
        let report = compute_magnetic_prior_uq_comparison_report(tiny_config())
            .expect("report should build");
        assert!(!report.replicate_bin_metric_rows.is_empty());
        for row in &report.replicate_bin_metric_rows {
            assert!(row.rmse_tesla.is_finite());
            assert!(row.nlpd.is_finite());
            assert!(row.standardized_rms.is_finite());
            assert!((0.0..=1.0).contains(&row.coverage_95));
        }
        assert!(!report.aggregate_bin_metric_rows.is_empty());
        for row in &report.aggregate_bin_metric_rows {
            assert!(row.rmse_tesla_mean.is_finite());
            assert!(row.nlpd_mean.is_finite());
            assert!(row.z_rms_mean.is_finite());
            assert!((0.0..=1.0).contains(&row.coverage_95_mean));
        }
        for row in &report.prior_roughness_rows {
            assert!(row.mean_adjacent_b_jump2.is_finite());
            assert!(row.rms_adjacent_b_jump.is_finite());
        }
    }
}
