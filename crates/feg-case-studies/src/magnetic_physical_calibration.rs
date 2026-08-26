use feec_gmrf::prelude::{
    gaussian_predictive_diagnostics_95, sparse_mat_from_feec_csr, write_csv, FactoredGaussianPrior,
    GaussianNoise, GaussianPrior, HutchinsonVarianceConfig, LinearGaussianModelBuilder, LinearMap,
    LinearObservation, Posterior, ProbeDistribution, ReportCell, ReportTable, VarianceEstimator,
    VarianceMethod, WeightedVarianceEstimate,
};
use feg_infer::{
    physical::build_reduced_magnetic_flux_density_operator_3d,
    prior::{
        matern::{
            one_form::{
                build_matern_precision_1form_with_mass_inverse_for_alpha, HodgeLaplacian1Form,
            },
            MaternAlpha,
        },
        trace_normalization::trace_normalization_from_target_trace,
    },
    sparse::{scale_matrix, symmetrize_feec_csr},
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
use std::{collections::BTreeSet, f64::consts::PI, fs, io, path::Path, time::Instant};

const DEFAULT_PRACTICAL_RANGE_M: f64 = 0.25;
const TARGET_CUBE_SIDE_LENGTH_M: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagneticTruthMode {
    SmoothManufactured,
    PriorSample,
}

impl MagneticTruthMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::SmoothManufactured => "smooth_manufactured",
            Self::PriorSample => "prior_sample",
        }
    }
}

impl std::str::FromStr for MagneticTruthMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "smooth" | "smooth-manufactured" | "smooth_manufactured" => {
                Ok(Self::SmoothManufactured)
            }
            "prior-sample" | "prior_sample" => Ok(Self::PriorSample),
            _ => Err(format!(
                "unsupported truth mode `{value}`; expected smooth or prior-sample"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MagneticPhysicalCalibrationConfig {
    pub levels: Vec<usize>,
    pub alphas: Vec<MaternAlpha>,
    pub practical_range_m: f64,
    pub target_b_rms_tesla: f64,
    pub tau_user: f64,
    pub truth_b_rms_tesla: f64,
    pub truth_mode: MagneticTruthMode,
    pub observation_std_tesla: f64,
    pub training_sensor_cells: usize,
    pub heldout_sensor_cells: usize,
    pub exact_max_dofs: usize,
    pub hutchinson_probes: usize,
    pub hutchinson_batches: usize,
    pub rng_seed: u64,
}

impl Default for MagneticPhysicalCalibrationConfig {
    fn default() -> Self {
        Self {
            levels: vec![2, 3, 4],
            alphas: vec![MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three],
            practical_range_m: DEFAULT_PRACTICAL_RANGE_M,
            target_b_rms_tesla: 0.10,
            tau_user: 1.0,
            truth_b_rms_tesla: 0.10,
            truth_mode: MagneticTruthMode::SmoothManufactured,
            observation_std_tesla: 0.005,
            training_sensor_cells: 24,
            heldout_sensor_cells: 24,
            exact_max_dofs: 700,
            hutchinson_probes: 128,
            hutchinson_batches: 4,
            rng_seed: 0xB0B5_1E1D,
        }
    }
}

impl MagneticPhysicalCalibrationConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke() -> Self {
        Self {
            levels: vec![2],
            alphas: vec![MaternAlpha::Two],
            training_sensor_cells: 4,
            heldout_sensor_cells: 4,
            exact_max_dofs: 128,
            hutchinson_probes: 8,
            hutchinson_batches: 2,
            ..Self::default()
        }
    }

    /// Immutable configuration used by the submitted thesis.
    ///
    /// The submitted calibration uses levels 2--8 and 512 probes. [`Default`]
    /// provides the three-level, 128-probe interactive profile.
    pub fn thesis_submitted() -> Self {
        Self {
            levels: (2..=8).collect(),
            hutchinson_probes: 512,
            ..Self::default()
        }
    }

    pub fn kappa(&self) -> f64 {
        (8.0_f64).sqrt() / self.practical_range_m
    }
}

#[derive(Debug, Clone)]
pub struct MagneticPhysicalCalibrationReport {
    pub config: MagneticPhysicalCalibrationConfig,
    pub prior_rows: Vec<MagneticPriorCalibrationRow>,
    pub sensor_rows: Vec<MagneticSensorCalibrationRow>,
    pub variance_rows: Vec<MagneticBVarianceStatsRow>,
    pub efficiency_rows: Vec<MagneticEfficiencyDiagnosticsRow>,
    pub observation_rows: Vec<MagneticObservationRow>,
}

#[derive(Debug, Clone)]
pub struct MagneticPriorCalibrationRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub dofs: usize,
    pub cells: usize,
    pub domain_volume: f64,
    pub kappa: f64,
    pub target_b_rms_tesla: f64,
    pub tau_user: f64,
    pub raw_trace: f64,
    pub raw_mean_b2: f64,
    pub raw_hutchinson_trace: Option<f64>,
    pub raw_hutchinson_relative_standard_error: Option<f64>,
    pub tau_normalizer: f64,
    pub precision_scale: f64,
    pub normalized_mean_b2: f64,
    pub exact_or_hutchinson_error: f64,
    pub normalization_source: String,
}

#[derive(Debug, Clone)]
pub struct MagneticSensorCalibrationRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub sigma_obs_tesla: f64,
    pub training_rows: usize,
    pub heldout_rows: usize,
    pub prior_sensor_sd_stats: SummaryStats,
    pub posterior_sensor_sd_stats: SummaryStats,
    pub training_chi2_per_row: f64,
    pub heldout_rmse_tesla: f64,
    pub heldout_standardized_rms: f64,
    pub heldout_coverage_95: f64,
}

#[derive(Debug, Clone)]
pub struct MagneticBVarianceStatsRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub stage: String,
    pub stats: SummaryStats,
}

#[derive(Debug, Clone)]
pub struct MagneticEfficiencyDiagnosticsRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub dofs: usize,
    pub b_rows: usize,
    pub trace_estimator: String,
    pub prior_factorizations: usize,
    pub posterior_factorizations: usize,
    pub normalized_prior_factorizations: usize,
    pub prior_qoi_rhs: usize,
    pub posterior_qoi_rhs: usize,
    pub trace_reused_from_variance_solve: bool,
    pub prior_factor_seconds: f64,
    pub posterior_factor_seconds: f64,
    pub prior_qoi_seconds: f64,
    pub posterior_qoi_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct MagneticObservationRow {
    pub level: usize,
    pub alpha: MaternAlpha,
    pub split: String,
    pub cell: usize,
    pub component: usize,
    pub operator_row: usize,
    pub truth_tesla: f64,
    pub observation_tesla: f64,
    pub noise_tesla: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SummaryStats {
    pub min: f64,
    pub p05: f64,
    pub median: f64,
    pub mean: f64,
    pub p95: f64,
    pub max: f64,
}

impl SummaryStats {
    fn from_slice(values: &[f64]) -> Result<Self, String> {
        if values.is_empty() {
            return Err("summary stats require at least one value".to_string());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("summary stats contain non-finite values".to_string());
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Self {
            min: sorted[0],
            p05: percentile(&sorted, 0.05),
            median: percentile(&sorted, 0.50),
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
            p95: percentile(&sorted, 0.95),
            max: *sorted.last().expect("nonempty"),
        })
    }
}

struct CubeWorkspace {
    topology: Complex,
    coords: MeshCoords,
    model: ReducedVectorPotentialMagnetostatic3d,
    cell_volumes: Vec<f64>,
    domain_volume: f64,
    b_operator: LinearMap,
}

struct CaseResult {
    prior_row: MagneticPriorCalibrationRow,
    sensor_row: MagneticSensorCalibrationRow,
    variance_rows: Vec<MagneticBVarianceStatsRow>,
    efficiency_row: MagneticEfficiencyDiagnosticsRow,
    observation_rows: Vec<MagneticObservationRow>,
}

struct PrecisionBuild {
    prior: GaussianPrior,
    raw_factored: FactoredGaussianPrior,
    raw_trace: CalibrationTraceEstimate,
    prior_b_variance: Vec<f64>,
    tau_normalizer: f64,
    precision_scale: f64,
    prior_factor_seconds: f64,
    prior_qoi_seconds: f64,
    prior_qoi_rhs: usize,
    trace_reused_from_variance_solve: bool,
}

#[derive(Debug, Clone, Copy)]
struct CalibrationTraceEstimate {
    value: f64,
    estimator: VarianceEstimator,
    relative_standard_error: Option<f64>,
}

impl From<&WeightedVarianceEstimate> for CalibrationTraceEstimate {
    fn from(estimate: &WeightedVarianceEstimate) -> Self {
        Self {
            value: estimate.weighted_trace,
            estimator: estimate.variances.estimator,
            relative_standard_error: estimate.weighted_trace_relative_standard_error,
        }
    }
}

struct TransformedVarianceSolve {
    values: Vec<f64>,
    qoi_rhs: usize,
    elapsed_seconds: f64,
}

struct SensorSplit {
    training_cells: Vec<usize>,
    heldout_cells: Vec<usize>,
    training_rows: Vec<usize>,
    heldout_rows: Vec<usize>,
}

pub fn compute_magnetic_physical_calibration_report(
    config: MagneticPhysicalCalibrationConfig,
) -> Result<MagneticPhysicalCalibrationReport, String> {
    validate_config(&config)?;
    let mut prior_rows = Vec::new();
    let mut sensor_rows = Vec::new();
    let mut variance_rows = Vec::new();
    let mut efficiency_rows = Vec::new();
    let mut observation_rows = Vec::new();

    for &level in &config.levels {
        let workspace = build_cube_workspace(level)?;
        let sensor_split = deterministic_sensor_split(
            &workspace.topology,
            &workspace.coords,
            config.training_sensor_cells,
            config.heldout_sensor_cells,
        )?;
        for &alpha in &config.alphas {
            let case = compute_case(&config, &workspace, &sensor_split, level, alpha)?;
            prior_rows.push(case.prior_row);
            sensor_rows.push(case.sensor_row);
            variance_rows.extend(case.variance_rows);
            efficiency_rows.push(case.efficiency_row);
            observation_rows.extend(case.observation_rows);
        }
    }

    Ok(MagneticPhysicalCalibrationReport {
        config,
        prior_rows,
        sensor_rows,
        variance_rows,
        efficiency_rows,
        observation_rows,
    })
}

pub fn write_magnetic_physical_calibration_outputs(
    report: &MagneticPhysicalCalibrationReport,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_prior_calibration_csv(report, &out_dir.join("prior_calibration.csv"))?;
    write_sensor_calibration_csv(report, &out_dir.join("sensor_calibration.csv"))?;
    write_b_variance_stats_csv(report, &out_dir.join("b_variance_stats.csv"))?;
    write_efficiency_diagnostics_csv(report, &out_dir.join("efficiency_diagnostics.csv"))?;
    write_observations_csv(report, &out_dir.join("observations.csv"))
}

fn compute_case(
    config: &MagneticPhysicalCalibrationConfig,
    workspace: &CubeWorkspace,
    sensor_split: &SensorSplit,
    level: usize,
    alpha: MaternAlpha,
) -> Result<CaseResult, String> {
    let kappa = config.kappa();
    let mut precision = build_prior_precision(config, workspace, level, alpha)?;
    let target_mean_b2 =
        config.target_b_rms_tesla * config.target_b_rms_tesla / (config.tau_user * config.tau_user);
    let normalized_mean_b2 =
        precision.raw_trace.value / precision.precision_scale / workspace.domain_volume;
    let exact_or_hutchinson_error = if precision.raw_trace.estimator.is_exact() {
        ((normalized_mean_b2 - target_mean_b2) / target_mean_b2).abs()
    } else {
        precision
            .raw_trace
            .relative_standard_error
            .unwrap_or(f64::NAN)
    };

    let truth = build_truth(config, workspace, &mut precision, level, alpha)?;
    let truth_b = workspace
        .b_operator
        .apply(&truth)
        .map_err(|err| err.to_string())?;
    let observations = build_observations(config, level, alpha, &truth_b, sensor_split)?;
    let training_observations = observations
        .iter()
        .filter(|row| row.split == "training")
        .map(|row| row.observation_tesla)
        .collect::<Vec<_>>();
    let training_operator = workspace
        .b_operator
        .select_outputs(&sensor_split.training_rows)
        .map_err(|err| err.to_string())?;
    let posterior_factor_start = Instant::now();
    let mut posterior = LinearGaussianModelBuilder::new(precision.prior.clone())
        .observe(
            LinearObservation::new(
                training_operator,
                training_observations,
                GaussianNoise::standard_deviation(config.observation_std_tesla)
                    .map_err(|err| err.to_string())?,
            )
            .map_err(|err| err.to_string())?,
        )
        .map_err(|err| err.to_string())?
        .condition()
        .map_err(|err| format!("posterior conditioning failed: {err}"))?;
    let posterior_factor_seconds = posterior_factor_start.elapsed().as_secs_f64();
    let posterior_mean = posterior.mean().to_vec();
    let posterior_b = workspace
        .b_operator
        .apply(&posterior_mean)
        .map_err(|err| err.to_string())?;

    let use_exact_variance = precision.prior.dimension() <= config.exact_max_dofs;
    let prior_b_variance = precision.prior_b_variance.clone();
    let posterior_b_variance = transformed_variances(
        &mut posterior,
        &workspace.b_operator,
        use_exact_variance,
        config,
        case_seed(config.rng_seed, level, alpha).wrapping_add(311),
    )?;

    let prior_sensor_sd =
        sensor_standard_deviations(&prior_b_variance, &sensor_split.training_rows);
    let posterior_sensor_sd =
        sensor_standard_deviations(&posterior_b_variance.values, &sensor_split.training_rows);
    let training_chi2_per_row = mean_row_chi2(
        &sensor_split.training_rows,
        &posterior_b,
        &observations,
        config.observation_std_tesla,
        "training",
    )?;
    let heldout = heldout_metrics(
        &sensor_split.heldout_rows,
        &truth_b,
        &posterior_b,
        &posterior_b_variance.values,
        &observations,
        config.observation_std_tesla,
    )?;
    let prior_trace = cell_trace_variance(&prior_b_variance)?;
    let posterior_trace = cell_trace_variance(&posterior_b_variance.values)?;

    Ok(CaseResult {
        prior_row: MagneticPriorCalibrationRow {
            level,
            alpha,
            dofs: precision.prior.dimension(),
            cells: workspace.cell_volumes.len(),
            domain_volume: workspace.domain_volume,
            kappa,
            target_b_rms_tesla: config.target_b_rms_tesla,
            tau_user: config.tau_user,
            raw_trace: precision.raw_trace.value,
            raw_mean_b2: precision.raw_trace.value / workspace.domain_volume,
            raw_hutchinson_trace: if precision.raw_trace.estimator.is_exact() {
                None
            } else {
                Some(precision.raw_trace.value)
            },
            raw_hutchinson_relative_standard_error: precision.raw_trace.relative_standard_error,
            tau_normalizer: precision.tau_normalizer,
            precision_scale: precision.precision_scale,
            normalized_mean_b2,
            exact_or_hutchinson_error,
            normalization_source: if precision.raw_trace.estimator.is_exact() {
                "exact".to_string()
            } else {
                "hutchinson".to_string()
            },
        },
        sensor_row: MagneticSensorCalibrationRow {
            level,
            alpha,
            sigma_obs_tesla: config.observation_std_tesla,
            training_rows: sensor_split.training_rows.len(),
            heldout_rows: sensor_split.heldout_rows.len(),
            prior_sensor_sd_stats: SummaryStats::from_slice(&prior_sensor_sd)?,
            posterior_sensor_sd_stats: SummaryStats::from_slice(&posterior_sensor_sd)?,
            training_chi2_per_row,
            heldout_rmse_tesla: heldout.rmse_tesla,
            heldout_standardized_rms: heldout.standardized_rms,
            heldout_coverage_95: heldout.coverage_95,
        },
        variance_rows: vec![
            MagneticBVarianceStatsRow {
                level,
                alpha,
                stage: "prior".to_string(),
                stats: SummaryStats::from_slice(&prior_trace)?,
            },
            MagneticBVarianceStatsRow {
                level,
                alpha,
                stage: "posterior".to_string(),
                stats: SummaryStats::from_slice(&posterior_trace)?,
            },
        ],
        efficiency_row: MagneticEfficiencyDiagnosticsRow {
            level,
            alpha,
            dofs: precision.prior.dimension(),
            b_rows: workspace.b_operator.output_dimension(),
            trace_estimator: if precision.raw_trace.estimator.is_exact() {
                "exact".to_string()
            } else {
                "hutchinson".to_string()
            },
            prior_factorizations: 1,
            posterior_factorizations: 1,
            normalized_prior_factorizations: 0,
            prior_qoi_rhs: precision.prior_qoi_rhs,
            posterior_qoi_rhs: posterior_b_variance.qoi_rhs,
            trace_reused_from_variance_solve: precision.trace_reused_from_variance_solve,
            prior_factor_seconds: precision.prior_factor_seconds,
            posterior_factor_seconds,
            prior_qoi_seconds: precision.prior_qoi_seconds,
            posterior_qoi_seconds: posterior_b_variance.elapsed_seconds,
        },
        observation_rows: observations,
    })
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

fn build_prior_precision(
    config: &MagneticPhysicalCalibrationConfig,
    workspace: &CubeWorkspace,
    level: usize,
    alpha: MaternAlpha,
) -> Result<PrecisionBuild, String> {
    let zero_state = vec![0.0; workspace.model.reduced_dimension()];
    let source_free = workspace
        .model
        .source_free_residual_and_jacobian(&zero_state)?;
    let hodge = HodgeLaplacian1Form {
        mass_u: workspace.model.state_mass().clone(),
        laplacian: source_free.jacobian,
    };
    let mass_inverse = workspace
        .model
        .state_mass_inverse()
        .ok_or_else(|| "magnetostatic cube model is missing state_mass_inverse".to_string())?
        .clone();
    let raw_precision =
        symmetrize_feec_csr(&build_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            alpha,
            config.kappa(),
            1.0,
        ));
    let prior_factor_start = Instant::now();
    let raw_prior = GaussianPrior::new(
        vec![0.0; raw_precision.nrows()],
        sparse_mat_from_feec_csr(&raw_precision),
    )
    .map_err(|err| format!("raw prior construction failed: {err}"))?;
    let mut raw_factored = raw_prior
        .factor()
        .map_err(|err| format!("raw prior precision factorization failed: {err}"))?;
    let prior_factor_seconds = prior_factor_start.elapsed().as_secs_f64();
    let weights = cell_major_b_weights(&workspace.cell_volumes);
    let row_seed = case_seed(config.rng_seed, level, alpha);
    let prior_qoi_start = Instant::now();
    let variance_method = if raw_precision.nrows() <= config.exact_max_dofs {
        VarianceMethod::Exact
    } else {
        hutchinson_method(config, row_seed)?
    };
    let raw_variance_trace = raw_factored
        .pushforward_weighted_variance_estimate(&workspace.b_operator, &weights, variance_method)
        .map_err(|err| err.to_string())?;
    let prior_qoi_seconds = prior_qoi_start.elapsed().as_secs_f64();
    let raw_trace = CalibrationTraceEstimate::from(&raw_variance_trace);
    let target_trace =
        config.target_b_rms_tesla * config.target_b_rms_tesla * workspace.domain_volume;
    let normalization = trace_normalization_from_target_trace(raw_trace.value, target_trace)?;
    let precision_scale = normalization.precision_scale * config.tau_user * config.tau_user;
    let prior_precision = scale_matrix(&raw_precision, precision_scale);
    let prior = GaussianPrior::new(
        vec![0.0; prior_precision.nrows()],
        sparse_mat_from_feec_csr(&prior_precision),
    )
    .map_err(|err| format!("normalized prior construction failed: {err}"))?;
    let prior_b_variance = scale_vector(
        &raw_variance_trace.variances.values,
        precision_scale.recip(),
    );
    let prior_qoi_rhs = if raw_variance_trace.variances.estimator.is_exact() {
        workspace.b_operator.output_dimension()
    } else {
        raw_variance_trace.variances.sample_count
    };
    Ok(PrecisionBuild {
        prior,
        raw_factored,
        raw_trace,
        prior_b_variance,
        tau_normalizer: normalization.tau_multiplier,
        precision_scale,
        prior_factor_seconds,
        prior_qoi_seconds,
        prior_qoi_rhs,
        trace_reused_from_variance_solve: true,
    })
}

fn build_truth(
    config: &MagneticPhysicalCalibrationConfig,
    workspace: &CubeWorkspace,
    precision: &mut PrecisionBuild,
    level: usize,
    alpha: MaternAlpha,
) -> Result<Vec<f64>, String> {
    match config.truth_mode {
        MagneticTruthMode::SmoothManufactured => {
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
        MagneticTruthMode::PriorSample => {
            let mut rng =
                StdRng::seed_from_u64(case_seed(config.rng_seed, level, alpha).wrapping_add(411));
            let raw_sample = precision
                .raw_factored
                .sample_cochain(&mut rng)
                .map_err(|err| err.to_string())?;
            Ok(scale_vector(
                &raw_sample,
                precision.precision_scale.sqrt().recip(),
            ))
        }
    }
}

fn build_observations(
    config: &MagneticPhysicalCalibrationConfig,
    level: usize,
    alpha: MaternAlpha,
    truth_b: &[f64],
    sensor_split: &SensorSplit,
) -> Result<Vec<MagneticObservationRow>, String> {
    let mut rows =
        Vec::with_capacity(sensor_split.training_rows.len() + sensor_split.heldout_rows.len());
    let noise_seed = match config.truth_mode {
        MagneticTruthMode::SmoothManufactured => config
            .rng_seed
            .wrapping_add((level as u64).wrapping_mul(10_007)),
        MagneticTruthMode::PriorSample => case_seed(config.rng_seed, level, alpha),
    };
    let mut rng = StdRng::seed_from_u64(noise_seed.wrapping_add(511));
    for (cell, &row) in sensor_split
        .training_cells
        .iter()
        .flat_map(|cell| [*cell, *cell, *cell])
        .zip(sensor_split.training_rows.iter())
    {
        push_observation_row(
            &mut rows, config, level, alpha, "training", cell, row, truth_b, &mut rng,
        )?;
    }
    for (cell, &row) in sensor_split
        .heldout_cells
        .iter()
        .flat_map(|cell| [*cell, *cell, *cell])
        .zip(sensor_split.heldout_rows.iter())
    {
        push_observation_row(
            &mut rows, config, level, alpha, "heldout", cell, row, truth_b, &mut rng,
        )?;
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn push_observation_row(
    rows: &mut Vec<MagneticObservationRow>,
    config: &MagneticPhysicalCalibrationConfig,
    level: usize,
    alpha: MaternAlpha,
    split: &str,
    cell: usize,
    row: usize,
    truth_b: &[f64],
    rng: &mut StdRng,
) -> Result<(), String> {
    if row >= truth_b.len() {
        return Err(format!(
            "observation row {row} exceeds B output dimension {}",
            truth_b.len()
        ));
    }
    let noise = config.observation_std_tesla * rng.sample::<f64, _>(StandardNormal);
    rows.push(MagneticObservationRow {
        level,
        alpha,
        split: split.to_string(),
        cell,
        component: row % 3,
        operator_row: row,
        truth_tesla: truth_b[row],
        observation_tesla: truth_b[row] + noise,
        noise_tesla: noise,
    });
    Ok(())
}

fn deterministic_sensor_split(
    topology: &Complex,
    coords: &MeshCoords,
    training_cells: usize,
    heldout_cells: usize,
) -> Result<SensorSplit, String> {
    let requested = training_cells + heldout_cells;
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
    if requested > ranked.len() {
        return Err(format!(
            "requested {requested} sensor cells but level has only {} cells",
            ranked.len()
        ));
    }
    let selected = ranked
        .into_iter()
        .take(requested)
        .map(|(cell, _)| cell)
        .collect::<Vec<_>>();
    let training = selected[..training_cells].to_vec();
    let heldout = selected[training_cells..].to_vec();
    let overlap = training
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .intersection(&heldout.iter().copied().collect::<BTreeSet<_>>())
        .count();
    if overlap != 0 {
        return Err("training and heldout sensor cells overlap".to_string());
    }
    Ok(SensorSplit {
        training_rows: cell_component_rows(&training),
        heldout_rows: cell_component_rows(&heldout),
        training_cells: training,
        heldout_cells: heldout,
    })
}

fn cell_component_rows(cells: &[usize]) -> Vec<usize> {
    cells
        .iter()
        .flat_map(|cell| [3 * *cell, 3 * *cell + 1, 3 * *cell + 2])
        .collect()
}

fn transformed_variances(
    posterior: &mut Posterior,
    operator: &LinearMap,
    use_exact: bool,
    config: &MagneticPhysicalCalibrationConfig,
    seed: u64,
) -> Result<TransformedVarianceSolve, String> {
    let start = Instant::now();
    let mode = if use_exact {
        VarianceMethod::Exact
    } else {
        hutchinson_method(config, seed)?
    };
    let estimate = posterior
        .pushforward_variance_estimate(operator, mode)
        .map_err(|err| err.to_string())?;
    Ok(TransformedVarianceSolve {
        values: estimate.values,
        qoi_rhs: if use_exact {
            operator.output_dimension()
        } else {
            estimate.sample_count
        },
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

fn mean_row_chi2(
    row_indices: &[usize],
    posterior_b: &[f64],
    observations: &[MagneticObservationRow],
    sigma: f64,
    split: &str,
) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for row in observations.iter().filter(|row| row.split == split) {
        if !row_indices.contains(&row.operator_row) {
            return Err(format!(
                "observation row {} is not in {split} row set",
                row.operator_row
            ));
        }
        let residual = posterior_b[row.operator_row] - row.observation_tesla;
        sum += residual * residual / (sigma * sigma);
        count += 1;
    }
    if count == 0 {
        return Err(format!("{split} chi2 requires at least one row"));
    }
    Ok(sum / count as f64)
}

struct HeldoutMetrics {
    rmse_tesla: f64,
    standardized_rms: f64,
    coverage_95: f64,
}

fn heldout_metrics(
    heldout_rows: &[usize],
    truth_b: &[f64],
    posterior_b: &[f64],
    posterior_variance: &[f64],
    observations: &[MagneticObservationRow],
    sigma: f64,
) -> Result<HeldoutMetrics, String> {
    let mut truth_sq = 0.0;
    let mut predictive_observations = Vec::new();
    let mut predictive_means = Vec::new();
    let mut predictive_variances = Vec::new();
    for row in observations.iter().filter(|row| row.split == "heldout") {
        if !heldout_rows.contains(&row.operator_row) {
            return Err(format!(
                "observation row {} is not in heldout row set",
                row.operator_row
            ));
        }
        let prediction = posterior_b[row.operator_row];
        let truth_residual = prediction - truth_b[row.operator_row];
        truth_sq += truth_residual * truth_residual;
        predictive_observations.push(row.observation_tesla);
        predictive_means.push(prediction);
        predictive_variances.push(posterior_variance[row.operator_row].max(0.0));
    }
    let count = predictive_observations.len();
    if count == 0 {
        return Err("heldout metrics require at least one row".to_string());
    }
    let diagnostics = gaussian_predictive_diagnostics_95(
        &predictive_observations,
        &predictive_means,
        &predictive_variances,
        &vec![sigma * sigma; count],
    )
    .map_err(|err| err.to_string())?;
    Ok(HeldoutMetrics {
        rmse_tesla: (truth_sq / count as f64).sqrt(),
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

fn sensor_standard_deviations(variances: &[f64], row_indices: &[usize]) -> Vec<f64> {
    row_indices
        .iter()
        .map(|&row| variances[row].max(0.0).sqrt())
        .collect()
}

fn cell_trace_variance(variances: &[f64]) -> Result<Vec<f64>, String> {
    if variances.len() % 3 != 0 {
        return Err(format!(
            "B variance length {} is not cell-major 3-vector output",
            variances.len()
        ));
    }
    Ok((0..variances.len() / 3)
        .map(|cell| {
            variances[3 * cell].max(0.0)
                + variances[3 * cell + 1].max(0.0)
                + variances[3 * cell + 2].max(0.0)
        })
        .collect())
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

fn hutchinson_method(
    config: &MagneticPhysicalCalibrationConfig,
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

fn case_seed(base: u64, level: usize, alpha: MaternAlpha) -> u64 {
    base.wrapping_add((level as u64).wrapping_mul(10_007))
        .wrapping_add((alpha.as_u32() as u64).wrapping_mul(1_009))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = position - lo as f64;
        (1.0 - t) * sorted[lo] + t * sorted[hi]
    }
}

fn validate_config(config: &MagneticPhysicalCalibrationConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one cube mesh level is required".to_string());
    }
    if config.levels.contains(&0) {
        return Err("cube mesh levels must be positive".to_string());
    }
    if config.alphas.is_empty() {
        return Err("at least one Matérn alpha is required".to_string());
    }
    validate_positive(config.practical_range_m, "practical_range_m")?;
    validate_positive(config.target_b_rms_tesla, "target_b_rms_tesla")?;
    validate_positive(config.tau_user, "tau_user")?;
    validate_positive(config.truth_b_rms_tesla, "truth_b_rms_tesla")?;
    validate_positive(config.observation_std_tesla, "observation_std_tesla")?;
    if config.training_sensor_cells == 0 {
        return Err("training_sensor_cells must be positive".to_string());
    }
    if config.heldout_sensor_cells == 0 {
        return Err("heldout_sensor_cells must be positive".to_string());
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

fn write_prior_calibration_csv(
    report: &MagneticPhysicalCalibrationReport,
    path: &Path,
) -> io::Result<()> {
    let mut table = report_table(
        "prior_calibration",
        &[
            "level",
            "alpha",
            "dofs",
            "cells",
            "domain_volume",
            "kappa",
            "target_b_rms_tesla",
            "tau_user",
            "raw_trace",
            "raw_mean_b2",
            "raw_hutchinson_trace",
            "raw_hutchinson_relative_standard_error",
            "tau_normalizer",
            "precision_scale",
            "normalized_mean_b2",
            "exact_or_hutchinson_error",
            "normalization_source",
        ],
    )?;
    for row in &report.prior_rows {
        table
            .push_row(vec![
                integer(row.level)?,
                ReportCell::Integer(i64::from(row.alpha.as_u32())),
                integer(row.dofs)?,
                integer(row.cells)?,
                ReportCell::Float(row.domain_volume),
                ReportCell::Float(row.kappa),
                ReportCell::Float(row.target_b_rms_tesla),
                ReportCell::Float(row.tau_user),
                ReportCell::Float(row.raw_trace),
                ReportCell::Float(row.raw_mean_b2),
                optional_cell(row.raw_hutchinson_trace),
                optional_cell(row.raw_hutchinson_relative_standard_error),
                ReportCell::Float(row.tau_normalizer),
                ReportCell::Float(row.precision_scale),
                ReportCell::Float(row.normalized_mean_b2),
                finite_or_text(row.exact_or_hutchinson_error),
                ReportCell::Text(row.normalization_source.clone()),
            ])
            .map_err(report_io)?;
    }
    write_csv(path, &table).map_err(report_io)
}

fn write_sensor_calibration_csv(
    report: &MagneticPhysicalCalibrationReport,
    path: &Path,
) -> io::Result<()> {
    let mut table = report_table(
        "sensor_calibration",
        &[
            "level",
            "alpha",
            "sigma_obs_tesla",
            "training_rows",
            "heldout_rows",
            "prior_sensor_sd_min",
            "prior_sensor_sd_p05",
            "prior_sensor_sd_median",
            "prior_sensor_sd_mean",
            "prior_sensor_sd_p95",
            "prior_sensor_sd_max",
            "posterior_sensor_sd_min",
            "posterior_sensor_sd_p05",
            "posterior_sensor_sd_median",
            "posterior_sensor_sd_mean",
            "posterior_sensor_sd_p95",
            "posterior_sensor_sd_max",
            "training_chi2_per_row",
            "heldout_rmse_tesla",
            "heldout_standardized_rms",
            "heldout_coverage_95",
        ],
    )?;
    for row in &report.sensor_rows {
        let mut cells = vec![
            integer(row.level)?,
            ReportCell::Integer(i64::from(row.alpha.as_u32())),
            ReportCell::Float(row.sigma_obs_tesla),
            integer(row.training_rows)?,
            integer(row.heldout_rows)?,
        ];
        cells.extend(summary_cells(row.prior_sensor_sd_stats));
        cells.extend(summary_cells(row.posterior_sensor_sd_stats));
        cells.extend([
            ReportCell::Float(row.training_chi2_per_row),
            ReportCell::Float(row.heldout_rmse_tesla),
            ReportCell::Float(row.heldout_standardized_rms),
            ReportCell::Float(row.heldout_coverage_95),
        ]);
        table.push_row(cells).map_err(report_io)?;
    }
    write_csv(path, &table).map_err(report_io)
}

fn write_b_variance_stats_csv(
    report: &MagneticPhysicalCalibrationReport,
    path: &Path,
) -> io::Result<()> {
    let mut table = report_table(
        "b_variance_stats",
        &[
            "level", "alpha", "stage", "min", "p05", "median", "mean", "p95", "max",
        ],
    )?;
    for row in &report.variance_rows {
        let mut cells = vec![
            integer(row.level)?,
            ReportCell::Integer(i64::from(row.alpha.as_u32())),
            ReportCell::Text(row.stage.clone()),
        ];
        cells.extend(summary_cells(row.stats));
        table.push_row(cells).map_err(report_io)?;
    }
    write_csv(path, &table).map_err(report_io)
}

fn write_efficiency_diagnostics_csv(
    report: &MagneticPhysicalCalibrationReport,
    path: &Path,
) -> io::Result<()> {
    let mut table = report_table(
        "efficiency_diagnostics",
        &[
            "level",
            "alpha",
            "dofs",
            "b_rows",
            "trace_estimator",
            "prior_factorizations",
            "posterior_factorizations",
            "normalized_prior_factorizations",
            "prior_qoi_rhs",
            "posterior_qoi_rhs",
            "trace_reused_from_variance_solve",
            "prior_factor_seconds",
            "posterior_factor_seconds",
            "prior_qoi_seconds",
            "posterior_qoi_seconds",
        ],
    )?;
    for row in &report.efficiency_rows {
        table
            .push_row(vec![
                integer(row.level)?,
                ReportCell::Integer(i64::from(row.alpha.as_u32())),
                integer(row.dofs)?,
                integer(row.b_rows)?,
                ReportCell::Text(row.trace_estimator.clone()),
                integer(row.prior_factorizations)?,
                integer(row.posterior_factorizations)?,
                integer(row.normalized_prior_factorizations)?,
                integer(row.prior_qoi_rhs)?,
                integer(row.posterior_qoi_rhs)?,
                ReportCell::Boolean(row.trace_reused_from_variance_solve),
                ReportCell::Float(row.prior_factor_seconds),
                ReportCell::Float(row.posterior_factor_seconds),
                ReportCell::Float(row.prior_qoi_seconds),
                ReportCell::Float(row.posterior_qoi_seconds),
            ])
            .map_err(report_io)?;
    }
    write_csv(path, &table).map_err(report_io)
}

fn write_observations_csv(
    report: &MagneticPhysicalCalibrationReport,
    path: &Path,
) -> io::Result<()> {
    let mut table = report_table(
        "observations",
        &[
            "level",
            "alpha",
            "split",
            "cell",
            "component",
            "operator_row",
            "truth_tesla",
            "observation_tesla",
            "noise_tesla",
        ],
    )?;
    for row in &report.observation_rows {
        table
            .push_row(vec![
                integer(row.level)?,
                ReportCell::Integer(i64::from(row.alpha.as_u32())),
                ReportCell::Text(row.split.clone()),
                integer(row.cell)?,
                integer(row.component)?,
                integer(row.operator_row)?,
                ReportCell::Float(row.truth_tesla),
                ReportCell::Float(row.observation_tesla),
                ReportCell::Float(row.noise_tesla),
            ])
            .map_err(report_io)?;
    }
    write_csv(path, &table).map_err(report_io)
}

fn report_table(id: &str, columns: &[&str]) -> io::Result<ReportTable> {
    ReportTable::new(
        id,
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
    .map_err(report_io)
}

fn summary_cells(stats: SummaryStats) -> [ReportCell; 6] {
    [
        ReportCell::Float(stats.min),
        ReportCell::Float(stats.p05),
        ReportCell::Float(stats.median),
        ReportCell::Float(stats.mean),
        ReportCell::Float(stats.p95),
        ReportCell::Float(stats.max),
    ]
}

fn optional_cell(value: Option<f64>) -> ReportCell {
    value.map(finite_or_text).unwrap_or(ReportCell::Missing)
}

fn finite_or_text(value: f64) -> ReportCell {
    if value.is_finite() {
        ReportCell::Float(value)
    } else {
        ReportCell::Text(value.to_string())
    }
}

fn integer(value: usize) -> io::Result<ReportCell> {
    i64::try_from(value)
        .map(ReportCell::Integer)
        .map_err(|_| io::Error::other("integer exceeds report-table range"))
}

fn report_io(error: feec_gmrf::FeecGmrfError) -> io::Error {
    match error {
        feec_gmrf::FeecGmrfError::Io(error) => error,
        error => io::Error::other(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> MagneticPhysicalCalibrationConfig {
        MagneticPhysicalCalibrationConfig {
            levels: vec![1],
            training_sensor_cells: 2,
            heldout_sensor_cells: 2,
            exact_max_dofs: 10_000,
            hutchinson_probes: 8,
            hutchinson_batches: 2,
            ..MagneticPhysicalCalibrationConfig::default()
        }
    }

    #[test]
    fn magnetic_physical_calibration_weights_are_cell_major_and_volume_normalized() {
        let workspace = build_cube_workspace(1).expect("workspace should build");
        let weights = cell_major_b_weights(&workspace.cell_volumes);
        assert_eq!(weights.len(), 3 * workspace.cell_volumes.len());
        for cell in 0..workspace.cell_volumes.len() {
            assert_eq!(weights[3 * cell], workspace.cell_volumes[cell]);
            assert_eq!(weights[3 * cell + 1], workspace.cell_volumes[cell]);
            assert_eq!(weights[3 * cell + 2], workspace.cell_volumes[cell]);
        }
        assert!((workspace.domain_volume - 1.0).abs() < 1e-12);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_physical_calibration_exact_trace_matches_target_b_rms() {
        let report = compute_magnetic_physical_calibration_report(tiny_config())
            .expect("report should build");
        for row in &report.prior_rows {
            assert_eq!(row.normalization_source, "exact");
            let target =
                row.target_b_rms_tesla * row.target_b_rms_tesla / (row.tau_user * row.tau_user);
            assert!(
                (row.normalized_mean_b2 - target).abs() < 1e-8 * target.max(1.0),
                "alpha={} normalized mean B2 {} target {}",
                row.alpha.as_u32(),
                row.normalized_mean_b2,
                target
            );
        }
        for row in &report.efficiency_rows {
            assert_eq!(row.prior_factorizations, 1);
            assert_eq!(row.posterior_factorizations, 1);
            assert_eq!(row.normalized_prior_factorizations, 0);
            assert!(row.trace_reused_from_variance_solve);
            assert_eq!(row.prior_qoi_rhs, row.b_rows);
            assert_eq!(row.posterior_qoi_rhs, row.b_rows);
        }
    }

    #[test]
    fn magnetic_physical_calibration_sensor_split_is_deterministic_and_disjoint() {
        let workspace = build_cube_workspace(1).expect("workspace should build");
        let split = deterministic_sensor_split(&workspace.topology, &workspace.coords, 2, 3)
            .expect("split should build");
        assert_eq!(split.training_rows.len(), 6);
        assert_eq!(split.heldout_rows.len(), 9);
        let training = split
            .training_cells
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let heldout = split.heldout_cells.iter().copied().collect::<BTreeSet<_>>();
        assert!(training.is_disjoint(&heldout));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_physical_calibration_conditioning_reduces_training_sensor_sd() {
        let report =
            compute_magnetic_physical_calibration_report(MagneticPhysicalCalibrationConfig {
                alphas: vec![MaternAlpha::Two],
                ..tiny_config()
            })
            .expect("report should build");
        let row = report.sensor_rows.first().expect("sensor row");
        assert!(row.posterior_sensor_sd_stats.mean < row.prior_sensor_sd_stats.mean);
        assert!(row.training_chi2_per_row.is_finite());
        assert!(row.heldout_standardized_rms.is_finite());
        let efficiency = report.efficiency_rows.first().expect("efficiency row");
        assert_eq!(efficiency.prior_factorizations, 1);
        assert_eq!(efficiency.posterior_factorizations, 1);
        assert_eq!(efficiency.normalized_prior_factorizations, 0);
    }

    #[test]
    fn magnetic_physical_calibration_rejects_invalid_config() {
        let mut config = tiny_config();
        config.target_b_rms_tesla = 0.0;
        assert!(compute_magnetic_physical_calibration_report(config)
            .unwrap_err()
            .contains("target_b_rms_tesla"));

        let mut config = tiny_config();
        config.observation_std_tesla = -1.0;
        assert!(compute_magnetic_physical_calibration_report(config)
            .unwrap_err()
            .contains("observation_std_tesla"));

        let mut config = tiny_config();
        config.levels.clear();
        assert!(compute_magnetic_physical_calibration_report(config)
            .unwrap_err()
            .contains("level"));

        let mut config = tiny_config();
        config.training_sensor_cells = 10_000;
        assert!(compute_magnetic_physical_calibration_report(config)
            .unwrap_err()
            .contains("sensor cells"));
    }

    #[test]
    fn magnetic_physical_calibration_report_tables_have_canonical_inventory() {
        let report =
            compute_magnetic_physical_calibration_report(MagneticPhysicalCalibrationConfig {
                alphas: vec![MaternAlpha::Two],
                ..tiny_config()
            })
            .expect("small calibration report should build");
        let out_dir = std::env::temp_dir().join(format!(
            "magnetic_calibration_report_tables_{}",
            std::process::id()
        ));
        write_magnetic_physical_calibration_outputs(&report, &out_dir)
            .expect("report tables should write");
        for (name, required_column) in [
            ("prior_calibration.csv", "normalization_source"),
            ("sensor_calibration.csv", "heldout_coverage_95"),
            ("b_variance_stats.csv", "stage"),
            ("efficiency_diagnostics.csv", "trace_estimator"),
            ("observations.csv", "observation_tesla"),
        ] {
            let csv = fs::read_to_string(out_dir.join(name)).expect("expected CSV artifact");
            assert!(csv.lines().next().unwrap().contains(required_column));
        }
        let _ = fs::remove_dir_all(out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn magnetic_physical_calibration_prior_sample_self_check_is_finite() {
        let report =
            compute_magnetic_physical_calibration_report(MagneticPhysicalCalibrationConfig {
                alphas: vec![MaternAlpha::Two],
                truth_mode: MagneticTruthMode::PriorSample,
                training_sensor_cells: 2,
                heldout_sensor_cells: 2,
                exact_max_dofs: 10_000,
                hutchinson_probes: 8,
                hutchinson_batches: 2,
                levels: vec![1],
                ..MagneticPhysicalCalibrationConfig::default()
            })
            .expect("prior-sample report should build");
        let sensor = report.sensor_rows.first().expect("sensor row");
        assert!(sensor.heldout_standardized_rms.is_finite());
        assert!(sensor.heldout_standardized_rms < 4.0);
        assert!((0.0..=1.0).contains(&sensor.heldout_coverage_95));
    }
}
