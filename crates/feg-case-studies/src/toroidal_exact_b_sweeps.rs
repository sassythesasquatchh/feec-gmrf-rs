use crate::toroidal_inductor::{
    run_toroidal_exact_b_recovery_experiment,
    toroidal_exact_b_canonical_source_designed_flux_config,
    toroidal_exact_b_field_recovery_observation_indices, ToroidalExactBObservationIndexOverride,
    ToroidalExactBObservationMode, ToroidalExactBRecoveryConfig, ToroidalExactBRecoveryReport,
    SOURCE_MODE_COUNT,
};
use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::Path,
    time::Instant,
};

const SOURCE_NOISE_AZIMUTH_COUNT: usize = 12;
const SOURCE_NOISE_TRAIN_FRACTION: f64 = 0.33;
const FIELD_COVERAGE_AZIMUTH_COUNT: usize = 16;
const BASELINE_OBSERVATION_NOISE_STD: f64 = 3.0e-10;
const SOURCE_NOISE_BASELINE_STD: f64 = 1.0e-10;
const FIELD_COVERAGE_NOISE_STD: f64 = BASELINE_OBSERVATION_NOISE_STD;
const EPS: f64 = 1.0e-300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBSweepProfile {
    Smoke,
    ThesisSubmitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBSweepKind {
    SourceNoise,
    FieldCoverage,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToroidalExactBSweepReport {
    pub source_noise_cases: usize,
    pub field_coverage_cases: usize,
    pub source_noise_min_coverage: f64,
    pub field_coverage_min_coverage: f64,
}

#[derive(Debug, Clone, Copy)]
struct SourceNoiseCase {
    label: &'static str,
    observation_noise_std: f64,
}

#[derive(Debug, Clone, Copy)]
struct FieldCoverageCase {
    label: &'static str,
    training_count: usize,
}

#[derive(Debug, Clone)]
struct SourceNoiseSummaryRow {
    label: String,
    observation_noise_std: f64,
    surface_flux_azimuth_count: usize,
    observation_train_fraction: f64,
    training_rows: usize,
    heldout_rows: usize,
    train_rmse: f64,
    heldout_latent_rmse: f64,
    heldout_latent_nlpd: f64,
    heldout_latent_covered95: usize,
    heldout_latent_coverage_fraction: f64,
    heldout_mean_posterior_flux_sd: f64,
    heldout_latent_max_abs_z: f64,
    heldout_latent_rms_z: f64,
    heldout_latent_mean_abs_residual: f64,
    heldout_noisy_rmse: f64,
    heldout_noisy_nlpd: f64,
    heldout_noisy_covered95: usize,
    heldout_noisy_coverage_fraction: f64,
    heldout_mean_predictive_sd: f64,
    heldout_noisy_max_abs_z: f64,
    heldout_noisy_rms_z: f64,
    heldout_noisy_mean_abs_residual: f64,
    source_rmse: f64,
    source_l2_error: f64,
    source_max_abs_error: f64,
    source_mean_posterior_sd: f64,
    source_covered95: usize,
    source_coverage_fraction: f64,
    b_relative_error: f64,
    source_response_condition: f64,
    source_response_snr_min: f64,
    source_response_snr_max: f64,
    runtime_seconds: f64,
    status: String,
    error: String,
}

#[derive(Debug, Clone)]
struct SourceModeRow {
    case_label: String,
    observation_noise_std: f64,
    mode_index: usize,
    truth: f64,
    posterior_mean: f64,
    posterior_sd: f64,
    error: f64,
    z_score: f64,
    lower95: f64,
    upper95: f64,
    covered95: bool,
}

#[derive(Debug, Clone)]
struct FieldCoverageSummaryRow {
    label: String,
    observation_noise_std: f64,
    surface_flux_azimuth_count: usize,
    training_rows: usize,
    heldout_rows: usize,
    train_rmse: f64,
    heldout_latent_rmse: f64,
    heldout_latent_nlpd: f64,
    heldout_latent_covered95: usize,
    heldout_latent_coverage_fraction: f64,
    heldout_mean_prior_flux_sd: f64,
    heldout_mean_posterior_flux_sd: f64,
    heldout_latent_max_abs_z: f64,
    heldout_latent_rms_z: f64,
    heldout_latent_mean_abs_residual: f64,
    heldout_noisy_rmse: f64,
    heldout_noisy_nlpd: f64,
    heldout_noisy_covered95: usize,
    heldout_noisy_coverage_fraction: f64,
    heldout_mean_predictive_sd: f64,
    heldout_noisy_max_abs_z: f64,
    heldout_noisy_rms_z: f64,
    heldout_noisy_mean_abs_residual: f64,
    source_rmse: f64,
    source_l2_error: f64,
    source_max_abs_error: f64,
    b_relative_error: f64,
    source_response_condition: f64,
    runtime_seconds: f64,
    status: String,
    error: String,
}

#[derive(Debug, Clone)]
struct FieldPredictionRow {
    case_label: String,
    row_index: usize,
    name: String,
    truth: f64,
    noisy_observation: f64,
    prediction: f64,
    residual: f64,
    noisy_residual: f64,
    prior_flux_sd: f64,
    posterior_flux_sd: f64,
    predictive_sd: f64,
    z_score: f64,
    noisy_z_score: f64,
    lower95: f64,
    upper95: f64,
    covered95: bool,
    noisy_lower95: f64,
    noisy_upper95: f64,
    noisy_covered95: bool,
}

#[derive(Debug, Clone)]
struct DesignRow {
    experiment: &'static str,
    case_label: String,
    role: String,
    row_index: usize,
    name: String,
}

#[derive(Debug, Clone, Copy)]
struct SourceMetrics {
    rmse: f64,
    l2_error: f64,
    max_abs_error: f64,
    mean_posterior_sd: f64,
    covered95: usize,
    coverage_fraction: f64,
}

#[derive(Debug, Clone, Copy)]
struct FluxVarianceMetrics {
    mean_prior_sd: f64,
    mean_posterior_sd: f64,
}

pub fn run_toroidal_exact_b_sweeps(
    output_dir: impl AsRef<Path>,
    profile: ToroidalExactBSweepProfile,
    kind: ToroidalExactBSweepKind,
) -> Result<ToroidalExactBSweepReport, Box<dyn Error>> {
    run_toroidal_exact_b_sweeps_with_case_limits(
        output_dir,
        kind,
        source_cases_len(profile),
        field_cases_len(profile),
    )
}

/// Run a research sweep with explicit prefixes of the registered case grids.
///
/// The immutable registry profiles call [`run_toroidal_exact_b_sweeps`]. This
/// variant is the canonical custom-configuration path and rejects case counts
/// outside the maintained grids.
pub fn run_toroidal_exact_b_sweeps_with_case_limits(
    output_dir: impl AsRef<Path>,
    kind: ToroidalExactBSweepKind,
    source_noise_case_count: usize,
    field_coverage_case_count: usize,
) -> Result<ToroidalExactBSweepReport, Box<dyn Error>> {
    let output_dir = output_dir.as_ref().to_path_buf();
    fs::create_dir_all(&output_dir)?;

    let all_source_cases = [
        SourceNoiseCase {
            label: "noise_std=1e-10",
            observation_noise_std: SOURCE_NOISE_BASELINE_STD,
        },
        SourceNoiseCase {
            label: "noise_std=3e-10",
            observation_noise_std: 3.0e-10,
        },
        SourceNoiseCase {
            label: "noise_std=1e-9",
            observation_noise_std: 1.0e-9,
        },
        SourceNoiseCase {
            label: "noise_std=3e-9",
            observation_noise_std: 3.0e-9,
        },
        SourceNoiseCase {
            label: "noise_std=1e-8",
            observation_noise_std: 1.0e-8,
        },
        SourceNoiseCase {
            label: "noise_std=3e-8",
            observation_noise_std: 3.0e-8,
        },
    ];
    let all_field_cases = [
        FieldCoverageCase {
            label: "training_rows=6",
            training_count: 6,
        },
        FieldCoverageCase {
            label: "training_rows=12",
            training_count: 12,
        },
        FieldCoverageCase {
            label: "training_rows=24",
            training_count: 24,
        },
        FieldCoverageCase {
            label: "training_rows=36",
            training_count: 36,
        },
    ];
    validate_case_limits(
        kind,
        source_noise_case_count,
        field_coverage_case_count,
        all_source_cases.len(),
        all_field_cases.len(),
    )?;
    let source_cases = match kind {
        ToroidalExactBSweepKind::FieldCoverage => Vec::new(),
        _ => all_source_cases[..source_noise_case_count].to_vec(),
    };
    let field_cases = match kind {
        ToroidalExactBSweepKind::SourceNoise => Vec::new(),
        _ => all_field_cases[..field_coverage_case_count].to_vec(),
    };

    let mut source_summary_rows = Vec::new();
    let mut source_mode_rows = Vec::new();
    let mut field_summary_rows = Vec::new();
    let mut field_prediction_rows = Vec::new();
    let mut design_rows = Vec::new();
    let mut source_noise_index_override = None;

    for case in source_cases {
        eprintln!(
            "exact-B source-noise sweep: {} noise={:.1e}",
            case.label, case.observation_noise_std
        );
        let output = run_source_noise_case(case, source_noise_index_override.as_ref());
        print_source_rows(std::slice::from_ref(&output.summary));
        io::stdout().flush()?;
        if source_noise_index_override.is_none() {
            if output.summary.status == "ok" {
                let fixed_indices =
                    source_noise_index_override_from_design_rows(case.label, &output.design_rows)?;
                eprintln!(
                    "exact-B source-noise sweep: fixed training rows {:?}",
                    fixed_indices.training_indices
                );
                source_noise_index_override = Some(fixed_indices);
            } else {
                return Err(format!(
                    "failed to establish fixed source-noise design from {}: {}",
                    case.label, output.summary.error
                )
                .into());
            }
        }
        source_summary_rows.push(output.summary);
        source_mode_rows.extend(output.source_modes);
        design_rows.extend(output.design_rows);
    }

    for case in field_cases {
        eprintln!(
            "exact-B field-coverage sweep: {} noise={:.1e}",
            case.label, FIELD_COVERAGE_NOISE_STD
        );
        let output = run_field_coverage_case(case);
        print_field_rows(std::slice::from_ref(&output.summary));
        io::stdout().flush()?;
        field_summary_rows.push(output.summary);
        field_prediction_rows.extend(output.predictions);
        design_rows.extend(output.design_rows);
    }

    let expected_source_cases = if kind == ToroidalExactBSweepKind::FieldCoverage {
        0
    } else {
        source_noise_case_count
    };
    let expected_field_cases = if kind == ToroidalExactBSweepKind::SourceNoise {
        0
    } else {
        field_coverage_case_count
    };
    if source_summary_rows.len() != expected_source_cases
        || field_summary_rows.len() != expected_field_cases
    {
        return Err("revised exact-B sweep emitted an unexpected number of rows".into());
    }

    fs::write(
        output_dir.join("source_noise_sweep_summary.csv"),
        source_noise_summary_csv(&source_summary_rows),
    )?;
    fs::write(
        output_dir.join("source_noise_modes.csv"),
        source_mode_csv(&source_mode_rows),
    )?;
    fs::write(
        output_dir.join("field_coverage_sweep_summary.csv"),
        field_coverage_summary_csv(&field_summary_rows),
    )?;
    fs::write(
        output_dir.join("field_heldout_predictions.csv"),
        field_prediction_csv(&field_prediction_rows),
    )?;
    fs::write(
        output_dir.join("observation_design.csv"),
        design_csv(&design_rows),
    )?;

    println!("Toroidal exact B=dA revised flux sweeps");
    println!("  source-noise rows={}", source_summary_rows.len());
    print_source_rows(&source_summary_rows);
    println!("  field-coverage rows={}", field_summary_rows.len());
    print_field_rows(&field_summary_rows);
    println!("  outputs: {}", output_dir.display());
    Ok(ToroidalExactBSweepReport {
        source_noise_cases: source_summary_rows.len(),
        field_coverage_cases: field_summary_rows.len(),
        source_noise_min_coverage: minimum_coverage(
            source_summary_rows
                .iter()
                .map(|row| row.heldout_noisy_coverage_fraction),
        ),
        field_coverage_min_coverage: minimum_coverage(
            field_summary_rows
                .iter()
                .map(|row| row.heldout_noisy_coverage_fraction),
        ),
    })
}

fn validate_case_limits(
    kind: ToroidalExactBSweepKind,
    source_count: usize,
    field_count: usize,
    maximum_source_count: usize,
    maximum_field_count: usize,
) -> Result<(), Box<dyn Error>> {
    if source_count > maximum_source_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("source_noise_case_count must not exceed {maximum_source_count}"),
        )
        .into());
    }
    if field_count > maximum_field_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("field_coverage_case_count must not exceed {maximum_field_count}"),
        )
        .into());
    }
    if kind != ToroidalExactBSweepKind::FieldCoverage && source_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source_noise_case_count must be positive for a source-noise sweep",
        )
        .into());
    }
    if kind != ToroidalExactBSweepKind::SourceNoise && field_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "field_coverage_case_count must be positive for a field-coverage sweep",
        )
        .into());
    }
    Ok(())
}

fn minimum_coverage(values: impl Iterator<Item = f64>) -> f64 {
    values.reduce(f64::min).unwrap_or(0.0)
}

fn source_cases_len(profile: ToroidalExactBSweepProfile) -> usize {
    match profile {
        ToroidalExactBSweepProfile::Smoke => 1,
        ToroidalExactBSweepProfile::ThesisSubmitted => 6,
    }
}

fn field_cases_len(profile: ToroidalExactBSweepProfile) -> usize {
    match profile {
        ToroidalExactBSweepProfile::Smoke => 1,
        ToroidalExactBSweepProfile::ThesisSubmitted => 4,
    }
}

struct SourceCaseOutput {
    summary: SourceNoiseSummaryRow,
    source_modes: Vec<SourceModeRow>,
    design_rows: Vec<DesignRow>,
}

struct FieldCaseOutput {
    summary: FieldCoverageSummaryRow,
    predictions: Vec<FieldPredictionRow>,
    design_rows: Vec<DesignRow>,
}

fn run_source_noise_case(
    case: SourceNoiseCase,
    index_override: Option<&ToroidalExactBObservationIndexOverride>,
) -> SourceCaseOutput {
    let started = Instant::now();
    let mut config = base_config();
    config.observation_mode = ToroidalExactBObservationMode::SourceDesignedFluxes;
    config.observation_noise_std = case.observation_noise_std;
    config.surface_flux_azimuth_count = SOURCE_NOISE_AZIMUTH_COUNT;
    config.observation_train_fraction = SOURCE_NOISE_TRAIN_FRACTION;
    config.observation_index_override = index_override.cloned();

    match run_toroidal_exact_b_recovery_experiment(&config) {
        Ok(report) => {
            match source_noise_row_from_report(&case, &report, started.elapsed().as_secs_f64()) {
                Ok(summary) => SourceCaseOutput {
                    source_modes: source_mode_rows(case, &report),
                    design_rows: design_rows("source_noise", case.label, &report),
                    summary,
                },
                Err(error) => SourceCaseOutput {
                    summary: source_noise_error_row(case, started.elapsed().as_secs_f64(), error),
                    source_modes: Vec::new(),
                    design_rows: Vec::new(),
                },
            }
        }
        Err(error) => SourceCaseOutput {
            summary: source_noise_error_row(case, started.elapsed().as_secs_f64(), error),
            source_modes: Vec::new(),
            design_rows: Vec::new(),
        },
    }
}

fn source_noise_index_override_from_design_rows(
    case_label: &str,
    rows: &[DesignRow],
) -> Result<ToroidalExactBObservationIndexOverride, Box<dyn Error>> {
    let mut training_indices = Vec::new();
    let mut heldout_indices = Vec::new();
    for row in rows {
        match row.role.as_str() {
            "train" => training_indices.push(row.row_index),
            "heldout" => heldout_indices.push(row.row_index),
            _ => {}
        }
    }
    if training_indices.is_empty() || heldout_indices.is_empty() {
        return Err(format!(
            "source-noise case {case_label} did not emit a usable train/heldout design"
        )
        .into());
    }
    training_indices.sort_unstable();
    heldout_indices.sort_unstable();
    Ok(ToroidalExactBObservationIndexOverride {
        training_indices,
        heldout_indices,
    })
}

fn run_field_coverage_case(case: FieldCoverageCase) -> FieldCaseOutput {
    let started = Instant::now();
    let mut config = base_config();
    config.observation_mode = ToroidalExactBObservationMode::SurfaceFluxes;
    config.observation_noise_std = FIELD_COVERAGE_NOISE_STD;
    config.surface_flux_azimuth_count = FIELD_COVERAGE_AZIMUTH_COUNT;
    config.observation_train_fraction =
        case.training_count as f64 / (FIELD_COVERAGE_AZIMUTH_COUNT * 3) as f64;
    config.observation_index_override = match toroidal_exact_b_field_recovery_observation_indices(
        FIELD_COVERAGE_AZIMUTH_COUNT,
        case.training_count,
    ) {
        Ok(indices) => Some(indices),
        Err(error) => {
            return FieldCaseOutput {
                summary: field_coverage_error_row(case, started.elapsed().as_secs_f64(), error),
                predictions: Vec::new(),
                design_rows: Vec::new(),
            };
        }
    };

    match run_toroidal_exact_b_recovery_experiment(&config) {
        Ok(report) => {
            match field_coverage_row_from_report(&case, &report, started.elapsed().as_secs_f64()) {
                Ok(summary) => FieldCaseOutput {
                    predictions: field_prediction_rows(case, &report).unwrap_or_default(),
                    design_rows: design_rows("field_coverage", case.label, &report),
                    summary,
                },
                Err(error) => FieldCaseOutput {
                    summary: field_coverage_error_row(case, started.elapsed().as_secs_f64(), error),
                    predictions: Vec::new(),
                    design_rows: Vec::new(),
                },
            }
        }
        Err(error) => FieldCaseOutput {
            summary: field_coverage_error_row(case, started.elapsed().as_secs_f64(), error),
            predictions: Vec::new(),
            design_rows: Vec::new(),
        },
    }
}

fn base_config() -> ToroidalExactBRecoveryConfig {
    let mut config = toroidal_exact_b_canonical_source_designed_flux_config();
    config.observation_noise_std = BASELINE_OBSERVATION_NOISE_STD;
    config.output_dir = None;
    config.write_outputs = false;
    config
}

fn source_noise_row_from_report(
    case: &SourceNoiseCase,
    report: &ToroidalExactBRecoveryReport,
    runtime_seconds: f64,
) -> Result<SourceNoiseSummaryRow, String> {
    let source = source_metrics(report)?;
    let b_relative_error = report
        .summary
        .b_relative_error
        .ok_or_else(|| "source-noise case did not produce a relative B error".to_string())?;
    let row = SourceNoiseSummaryRow {
        label: case.label.to_string(),
        observation_noise_std: case.observation_noise_std,
        surface_flux_azimuth_count: SOURCE_NOISE_AZIMUTH_COUNT,
        observation_train_fraction: SOURCE_NOISE_TRAIN_FRACTION,
        training_rows: report.summary.training_rows,
        heldout_rows: report.summary.heldout_rows,
        train_rmse: report.summary.train_rmse,
        heldout_latent_rmse: report.summary.heldout_rmse,
        heldout_latent_nlpd: report.summary.heldout_nlpd,
        heldout_latent_covered95: report.summary.heldout_covered95,
        heldout_latent_coverage_fraction: report.summary.heldout_coverage_fraction,
        heldout_mean_posterior_flux_sd: report.summary.heldout_mean_posterior_flux_sd,
        heldout_latent_max_abs_z: report.summary.heldout_max_abs_z,
        heldout_latent_rms_z: report.summary.heldout_rms_z,
        heldout_latent_mean_abs_residual: report.summary.heldout_mean_abs_residual,
        heldout_noisy_rmse: report.summary.heldout_noisy_rmse,
        heldout_noisy_nlpd: report.summary.heldout_noisy_nlpd,
        heldout_noisy_covered95: report.summary.heldout_noisy_covered95,
        heldout_noisy_coverage_fraction: report.summary.heldout_noisy_coverage_fraction,
        heldout_mean_predictive_sd: report.summary.heldout_mean_predictive_sd,
        heldout_noisy_max_abs_z: report.summary.heldout_noisy_max_abs_z,
        heldout_noisy_rms_z: report.summary.heldout_noisy_rms_z,
        heldout_noisy_mean_abs_residual: report.summary.heldout_noisy_mean_abs_residual,
        source_rmse: source.rmse,
        source_l2_error: source.l2_error,
        source_max_abs_error: source.max_abs_error,
        source_mean_posterior_sd: source.mean_posterior_sd,
        source_covered95: source.covered95,
        source_coverage_fraction: source.coverage_fraction,
        b_relative_error,
        source_response_condition: report.source_response.condition,
        source_response_snr_min: report.source_response.snr_min,
        source_response_snr_max: report.source_response.snr_max,
        runtime_seconds,
        status: "ok".to_string(),
        error: String::new(),
    };
    if source_noise_metrics_are_finite(&row) {
        Ok(row)
    } else {
        Err("source-noise case produced non-finite summary metrics".to_string())
    }
}

fn field_coverage_row_from_report(
    case: &FieldCoverageCase,
    report: &ToroidalExactBRecoveryReport,
    runtime_seconds: f64,
) -> Result<FieldCoverageSummaryRow, String> {
    let source = source_metrics(report)?;
    let flux_variance = flux_variance_metrics(report)?;
    let b_relative_error = report
        .summary
        .b_relative_error
        .ok_or_else(|| "field-coverage case did not produce a relative B error".to_string())?;
    let row = FieldCoverageSummaryRow {
        label: case.label.to_string(),
        observation_noise_std: FIELD_COVERAGE_NOISE_STD,
        surface_flux_azimuth_count: FIELD_COVERAGE_AZIMUTH_COUNT,
        training_rows: report.summary.training_rows,
        heldout_rows: report.summary.heldout_rows,
        train_rmse: report.summary.train_rmse,
        heldout_latent_rmse: report.summary.heldout_rmse,
        heldout_latent_nlpd: report.summary.heldout_nlpd,
        heldout_latent_covered95: report.summary.heldout_covered95,
        heldout_latent_coverage_fraction: report.summary.heldout_coverage_fraction,
        heldout_mean_prior_flux_sd: flux_variance.mean_prior_sd,
        heldout_mean_posterior_flux_sd: flux_variance.mean_posterior_sd,
        heldout_latent_max_abs_z: report.summary.heldout_max_abs_z,
        heldout_latent_rms_z: report.summary.heldout_rms_z,
        heldout_latent_mean_abs_residual: report.summary.heldout_mean_abs_residual,
        heldout_noisy_rmse: report.summary.heldout_noisy_rmse,
        heldout_noisy_nlpd: report.summary.heldout_noisy_nlpd,
        heldout_noisy_covered95: report.summary.heldout_noisy_covered95,
        heldout_noisy_coverage_fraction: report.summary.heldout_noisy_coverage_fraction,
        heldout_mean_predictive_sd: report.summary.heldout_mean_predictive_sd,
        heldout_noisy_max_abs_z: report.summary.heldout_noisy_max_abs_z,
        heldout_noisy_rms_z: report.summary.heldout_noisy_rms_z,
        heldout_noisy_mean_abs_residual: report.summary.heldout_noisy_mean_abs_residual,
        source_rmse: source.rmse,
        source_l2_error: source.l2_error,
        source_max_abs_error: source.max_abs_error,
        b_relative_error,
        source_response_condition: report.source_response.condition,
        runtime_seconds,
        status: "ok".to_string(),
        error: String::new(),
    };
    if field_coverage_metrics_are_finite(&row) {
        Ok(row)
    } else {
        Err("field-coverage case produced non-finite summary metrics".to_string())
    }
}

fn source_metrics(report: &ToroidalExactBRecoveryReport) -> Result<SourceMetrics, String> {
    if report.source_posterior.len() != SOURCE_MODE_COUNT {
        return Err(format!(
            "source posterior should have {SOURCE_MODE_COUNT} rows"
        ));
    }
    let l2_error = report
        .source_posterior
        .iter()
        .map(|row| row.error * row.error)
        .sum::<f64>()
        .sqrt();
    let max_abs_error = report
        .source_posterior
        .iter()
        .map(|row| row.error.abs())
        .fold(0.0, f64::max);
    let mean_posterior_sd = report
        .source_posterior
        .iter()
        .map(|row| row.posterior_variance.max(0.0).sqrt())
        .sum::<f64>()
        / SOURCE_MODE_COUNT as f64;
    let covered95 = report
        .source_posterior
        .iter()
        .filter(|row| {
            let sd = row.posterior_variance.max(0.0).sqrt();
            let lower = row.posterior_mean - 1.96 * sd;
            let upper = row.posterior_mean + 1.96 * sd;
            row.truth >= lower && row.truth <= upper
        })
        .count();
    Ok(SourceMetrics {
        rmse: l2_error / (SOURCE_MODE_COUNT as f64).sqrt(),
        l2_error,
        max_abs_error,
        mean_posterior_sd,
        covered95,
        coverage_fraction: covered95 as f64 / SOURCE_MODE_COUNT as f64,
    })
}

fn flux_variance_metrics(
    report: &ToroidalExactBRecoveryReport,
) -> Result<FluxVarianceMetrics, String> {
    let rows = report
        .heldout_predictions
        .iter()
        .map(|heldout| flux_variance_for_heldout(report, &heldout.name))
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        return Err("field-coverage case produced no heldout flux variance rows".to_string());
    }
    let mean_prior_sd = rows.iter().map(|(prior, _)| *prior).sum::<f64>() / rows.len() as f64;
    let mean_posterior_sd =
        rows.iter().map(|(_, posterior)| *posterior).sum::<f64>() / rows.len() as f64;
    Ok(FluxVarianceMetrics {
        mean_prior_sd,
        mean_posterior_sd,
    })
}

fn flux_variance_for_heldout(
    report: &ToroidalExactBRecoveryReport,
    name: &str,
) -> Result<(f64, f64), String> {
    let derived_name = format!("exact_b::{name}");
    let variance = report
        .result
        .derived_variances
        .get(&derived_name)
        .ok_or_else(|| format!("missing heldout variance `{derived_name}`"))?;
    if variance.prior_variance.len() != 1 || variance.posterior_variance.len() != 1 {
        return Err(format!("heldout variance `{derived_name}` must be scalar"));
    }
    Ok((
        variance.prior_variance[0].max(0.0).sqrt(),
        variance.posterior_variance[0].max(0.0).sqrt(),
    ))
}

fn source_mode_rows(
    case: SourceNoiseCase,
    report: &ToroidalExactBRecoveryReport,
) -> Vec<SourceModeRow> {
    report
        .source_posterior
        .iter()
        .map(|row| {
            let posterior_sd = row.posterior_variance.max(0.0).sqrt();
            let lower95 = row.posterior_mean - 1.96 * posterior_sd;
            let upper95 = row.posterior_mean + 1.96 * posterior_sd;
            SourceModeRow {
                case_label: case.label.to_string(),
                observation_noise_std: case.observation_noise_std,
                mode_index: row.mode_index,
                truth: row.truth,
                posterior_mean: row.posterior_mean,
                posterior_sd,
                error: row.error,
                z_score: row.error / posterior_sd.max(EPS),
                lower95,
                upper95,
                covered95: row.truth >= lower95 && row.truth <= upper95,
            }
        })
        .collect()
}

fn field_prediction_rows(
    case: FieldCoverageCase,
    report: &ToroidalExactBRecoveryReport,
) -> Result<Vec<FieldPredictionRow>, String> {
    report
        .heldout_predictions
        .iter()
        .map(|row| {
            let (prior_flux_sd, posterior_flux_sd) = flux_variance_for_heldout(report, &row.name)?;
            Ok(FieldPredictionRow {
                case_label: case.label.to_string(),
                row_index: probe_index(report, &row.name)?,
                name: row.name.clone(),
                truth: row.truth,
                noisy_observation: row.noisy_observation,
                prediction: row.prediction,
                residual: row.residual,
                noisy_residual: row.noisy_residual,
                prior_flux_sd,
                posterior_flux_sd,
                predictive_sd: row.predictive_sd,
                z_score: row.standardized_residual,
                noisy_z_score: row.noisy_standardized_residual,
                lower95: row.lower95,
                upper95: row.upper95,
                covered95: row.covered95,
                noisy_lower95: row.noisy_lower95,
                noisy_upper95: row.noisy_upper95,
                noisy_covered95: row.noisy_covered95,
            })
        })
        .collect()
}

fn design_rows(
    experiment: &'static str,
    case_label: &str,
    report: &ToroidalExactBRecoveryReport,
) -> Vec<DesignRow> {
    report
        .probes
        .iter()
        .enumerate()
        .filter(|(_, probe)| probe.role == "train" || probe.role == "heldout")
        .map(|(row_index, probe)| DesignRow {
            experiment,
            case_label: case_label.to_string(),
            role: probe.role.clone(),
            row_index,
            name: probe.name.clone(),
        })
        .collect()
}

fn probe_index(report: &ToroidalExactBRecoveryReport, name: &str) -> Result<usize, String> {
    report
        .probes
        .iter()
        .position(|probe| probe.name == name)
        .ok_or_else(|| format!("missing probe `{name}`"))
}

fn source_noise_metrics_are_finite(row: &SourceNoiseSummaryRow) -> bool {
    [
        row.train_rmse,
        row.heldout_latent_rmse,
        row.heldout_latent_nlpd,
        row.heldout_latent_coverage_fraction,
        row.heldout_mean_posterior_flux_sd,
        row.heldout_latent_max_abs_z,
        row.heldout_latent_rms_z,
        row.heldout_latent_mean_abs_residual,
        row.heldout_noisy_rmse,
        row.heldout_noisy_nlpd,
        row.heldout_noisy_coverage_fraction,
        row.heldout_mean_predictive_sd,
        row.heldout_noisy_max_abs_z,
        row.heldout_noisy_rms_z,
        row.heldout_noisy_mean_abs_residual,
        row.source_rmse,
        row.source_l2_error,
        row.source_max_abs_error,
        row.source_mean_posterior_sd,
        row.source_coverage_fraction,
        row.b_relative_error,
        row.source_response_condition,
        row.source_response_snr_min,
        row.source_response_snr_max,
        row.runtime_seconds,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn field_coverage_metrics_are_finite(row: &FieldCoverageSummaryRow) -> bool {
    [
        row.train_rmse,
        row.heldout_latent_rmse,
        row.heldout_latent_nlpd,
        row.heldout_latent_coverage_fraction,
        row.heldout_mean_prior_flux_sd,
        row.heldout_mean_posterior_flux_sd,
        row.heldout_latent_max_abs_z,
        row.heldout_latent_rms_z,
        row.heldout_latent_mean_abs_residual,
        row.heldout_noisy_rmse,
        row.heldout_noisy_nlpd,
        row.heldout_noisy_coverage_fraction,
        row.heldout_mean_predictive_sd,
        row.heldout_noisy_max_abs_z,
        row.heldout_noisy_rms_z,
        row.heldout_noisy_mean_abs_residual,
        row.source_rmse,
        row.source_l2_error,
        row.source_max_abs_error,
        row.b_relative_error,
        row.source_response_condition,
        row.runtime_seconds,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn source_noise_error_row(
    case: SourceNoiseCase,
    runtime_seconds: f64,
    error: String,
) -> SourceNoiseSummaryRow {
    SourceNoiseSummaryRow {
        label: case.label.to_string(),
        observation_noise_std: case.observation_noise_std,
        surface_flux_azimuth_count: SOURCE_NOISE_AZIMUTH_COUNT,
        observation_train_fraction: SOURCE_NOISE_TRAIN_FRACTION,
        training_rows: 0,
        heldout_rows: 0,
        train_rmse: f64::NAN,
        heldout_latent_rmse: f64::NAN,
        heldout_latent_nlpd: f64::NAN,
        heldout_latent_covered95: 0,
        heldout_latent_coverage_fraction: f64::NAN,
        heldout_mean_posterior_flux_sd: f64::NAN,
        heldout_latent_max_abs_z: f64::NAN,
        heldout_latent_rms_z: f64::NAN,
        heldout_latent_mean_abs_residual: f64::NAN,
        heldout_noisy_rmse: f64::NAN,
        heldout_noisy_nlpd: f64::NAN,
        heldout_noisy_covered95: 0,
        heldout_noisy_coverage_fraction: f64::NAN,
        heldout_mean_predictive_sd: f64::NAN,
        heldout_noisy_max_abs_z: f64::NAN,
        heldout_noisy_rms_z: f64::NAN,
        heldout_noisy_mean_abs_residual: f64::NAN,
        source_rmse: f64::NAN,
        source_l2_error: f64::NAN,
        source_max_abs_error: f64::NAN,
        source_mean_posterior_sd: f64::NAN,
        source_covered95: 0,
        source_coverage_fraction: f64::NAN,
        b_relative_error: f64::NAN,
        source_response_condition: f64::NAN,
        source_response_snr_min: f64::NAN,
        source_response_snr_max: f64::NAN,
        runtime_seconds,
        status: "error".to_string(),
        error,
    }
}

fn field_coverage_error_row(
    case: FieldCoverageCase,
    runtime_seconds: f64,
    error: String,
) -> FieldCoverageSummaryRow {
    FieldCoverageSummaryRow {
        label: case.label.to_string(),
        observation_noise_std: FIELD_COVERAGE_NOISE_STD,
        surface_flux_azimuth_count: FIELD_COVERAGE_AZIMUTH_COUNT,
        training_rows: 0,
        heldout_rows: 0,
        train_rmse: f64::NAN,
        heldout_latent_rmse: f64::NAN,
        heldout_latent_nlpd: f64::NAN,
        heldout_latent_covered95: 0,
        heldout_latent_coverage_fraction: f64::NAN,
        heldout_mean_prior_flux_sd: f64::NAN,
        heldout_mean_posterior_flux_sd: f64::NAN,
        heldout_latent_max_abs_z: f64::NAN,
        heldout_latent_rms_z: f64::NAN,
        heldout_latent_mean_abs_residual: f64::NAN,
        heldout_noisy_rmse: f64::NAN,
        heldout_noisy_nlpd: f64::NAN,
        heldout_noisy_covered95: 0,
        heldout_noisy_coverage_fraction: f64::NAN,
        heldout_mean_predictive_sd: f64::NAN,
        heldout_noisy_max_abs_z: f64::NAN,
        heldout_noisy_rms_z: f64::NAN,
        heldout_noisy_mean_abs_residual: f64::NAN,
        source_rmse: f64::NAN,
        source_l2_error: f64::NAN,
        source_max_abs_error: f64::NAN,
        b_relative_error: f64::NAN,
        source_response_condition: f64::NAN,
        runtime_seconds,
        status: "error".to_string(),
        error,
    }
}

fn source_noise_summary_csv(rows: &[SourceNoiseSummaryRow]) -> String {
    let mut csv = "case,observation_noise_std,surface_flux_azimuth_count,observation_train_fraction,training_rows,heldout_rows,train_rmse,heldout_latent_rmse,heldout_latent_nlpd,heldout_latent_covered95,heldout_latent_coverage_fraction,heldout_mean_posterior_flux_sd,heldout_latent_max_abs_z,heldout_latent_rms_z,heldout_latent_mean_abs_residual,heldout_noisy_rmse,heldout_noisy_nlpd,heldout_noisy_covered95,heldout_noisy_coverage_fraction,heldout_mean_predictive_sd,heldout_noisy_max_abs_z,heldout_noisy_rms_z,heldout_noisy_mean_abs_residual,source_rmse,source_l2_error,source_max_abs_error,source_mean_posterior_sd,source_covered95,source_coverage_fraction,b_relative_error,source_response_condition,source_response_snr_min,source_response_snr_max,runtime_seconds,status,error\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{:.16e},{},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.6},{},{}\n",
            csv_string(&row.label),
            row.observation_noise_std,
            row.surface_flux_azimuth_count,
            row.observation_train_fraction,
            row.training_rows,
            row.heldout_rows,
            row.train_rmse,
            row.heldout_latent_rmse,
            row.heldout_latent_nlpd,
            row.heldout_latent_covered95,
            row.heldout_latent_coverage_fraction,
            row.heldout_mean_posterior_flux_sd,
            row.heldout_latent_max_abs_z,
            row.heldout_latent_rms_z,
            row.heldout_latent_mean_abs_residual,
            row.heldout_noisy_rmse,
            row.heldout_noisy_nlpd,
            row.heldout_noisy_covered95,
            row.heldout_noisy_coverage_fraction,
            row.heldout_mean_predictive_sd,
            row.heldout_noisy_max_abs_z,
            row.heldout_noisy_rms_z,
            row.heldout_noisy_mean_abs_residual,
            row.source_rmse,
            row.source_l2_error,
            row.source_max_abs_error,
            row.source_mean_posterior_sd,
            row.source_covered95,
            row.source_coverage_fraction,
            row.b_relative_error,
            row.source_response_condition,
            row.source_response_snr_min,
            row.source_response_snr_max,
            row.runtime_seconds,
            csv_string(&row.status),
            csv_string(&row.error),
        ));
    }
    csv
}

fn source_mode_csv(rows: &[SourceModeRow]) -> String {
    let mut csv = "case,observation_noise_std,mode_index,truth,posterior_mean,posterior_sd,error,z_score,lower95,upper95,covered95\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
            csv_string(&row.case_label),
            row.observation_noise_std,
            row.mode_index,
            row.truth,
            row.posterior_mean,
            row.posterior_sd,
            row.error,
            row.z_score,
            row.lower95,
            row.upper95,
            row.covered95,
        ));
    }
    csv
}

fn field_coverage_summary_csv(rows: &[FieldCoverageSummaryRow]) -> String {
    let mut csv = "case,observation_noise_std,surface_flux_azimuth_count,training_rows,heldout_rows,train_rmse,heldout_latent_rmse,heldout_latent_nlpd,heldout_latent_covered95,heldout_latent_coverage_fraction,heldout_mean_prior_flux_sd,heldout_mean_posterior_flux_sd,heldout_latent_max_abs_z,heldout_latent_rms_z,heldout_latent_mean_abs_residual,heldout_noisy_rmse,heldout_noisy_nlpd,heldout_noisy_covered95,heldout_noisy_coverage_fraction,heldout_mean_predictive_sd,heldout_noisy_max_abs_z,heldout_noisy_rms_z,heldout_noisy_mean_abs_residual,source_rmse,source_l2_error,source_max_abs_error,b_relative_error,source_response_condition,runtime_seconds,status,error\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.6},{},{}\n",
            csv_string(&row.label),
            row.observation_noise_std,
            row.surface_flux_azimuth_count,
            row.training_rows,
            row.heldout_rows,
            row.train_rmse,
            row.heldout_latent_rmse,
            row.heldout_latent_nlpd,
            row.heldout_latent_covered95,
            row.heldout_latent_coverage_fraction,
            row.heldout_mean_prior_flux_sd,
            row.heldout_mean_posterior_flux_sd,
            row.heldout_latent_max_abs_z,
            row.heldout_latent_rms_z,
            row.heldout_latent_mean_abs_residual,
            row.heldout_noisy_rmse,
            row.heldout_noisy_nlpd,
            row.heldout_noisy_covered95,
            row.heldout_noisy_coverage_fraction,
            row.heldout_mean_predictive_sd,
            row.heldout_noisy_max_abs_z,
            row.heldout_noisy_rms_z,
            row.heldout_noisy_mean_abs_residual,
            row.source_rmse,
            row.source_l2_error,
            row.source_max_abs_error,
            row.b_relative_error,
            row.source_response_condition,
            row.runtime_seconds,
            csv_string(&row.status),
            csv_string(&row.error),
        ));
    }
    csv
}

fn field_prediction_csv(rows: &[FieldPredictionRow]) -> String {
    let mut csv = "case,row_index,name,latent_truth,noisy_observation,prediction,latent_residual,noisy_residual,prior_flux_sd,posterior_flux_sd,predictive_sd,latent_z_score,noisy_z_score,latent_lower95,latent_upper95,latent_covered95,noisy_lower95,noisy_upper95,noisy_covered95\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{}\n",
            csv_string(&row.case_label),
            row.row_index,
            csv_string(&row.name),
            row.truth,
            row.noisy_observation,
            row.prediction,
            row.residual,
            row.noisy_residual,
            row.prior_flux_sd,
            row.posterior_flux_sd,
            row.predictive_sd,
            row.z_score,
            row.noisy_z_score,
            row.lower95,
            row.upper95,
            row.covered95,
            row.noisy_lower95,
            row.noisy_upper95,
            row.noisy_covered95,
        ));
    }
    csv
}

fn design_csv(rows: &[DesignRow]) -> String {
    let mut csv = "experiment,case,role,row_index,name\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            csv_string(row.experiment),
            csv_string(&row.case_label),
            csv_string(&row.role),
            row.row_index,
            csv_string(&row.name),
        ));
    }
    csv
}

fn print_source_rows(rows: &[SourceNoiseSummaryRow]) {
    for row in rows {
        println!(
            "    {} status={} train={} heldout={} source_rmse={:.3e} source_sd={:.3e} latent_rmse={:.3e} noisy_rmse={:.3e} pred_sd={:.3e}",
            row.label,
            row.status,
            row.training_rows,
            row.heldout_rows,
            row.source_rmse,
            row.source_mean_posterior_sd,
            row.heldout_latent_rmse,
            row.heldout_noisy_rmse,
            row.heldout_mean_predictive_sd
        );
        if !row.error.is_empty() {
            println!("      error: {}", row.error);
        }
    }
}

fn print_field_rows(rows: &[FieldCoverageSummaryRow]) {
    for row in rows {
        println!(
            "    {} status={} train={} heldout={} latent_rmse={:.3e} noisy_rmse={:.3e} posterior_flux_sd={:.3e} pred_sd={:.3e} Berr={:.3e}",
            row.label,
            row.status,
            row.training_rows,
            row.heldout_rows,
            row.heldout_latent_rmse,
            row.heldout_noisy_rmse,
            row.heldout_mean_posterior_flux_sd,
            row.heldout_mean_predictive_sd,
            row.b_relative_error
        );
        if !row.error.is_empty() {
            println!("      error: {}", row.error);
        }
    }
}

fn csv_string(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design_row(case_label: &str, role: &str, row_index: usize) -> DesignRow {
        DesignRow {
            experiment: "source_noise",
            case_label: case_label.to_string(),
            role: role.to_string(),
            row_index,
            name: format!("flux_{row_index}"),
        }
    }

    #[test]
    fn source_noise_index_override_preserves_sorted_design_roles() {
        let rows = vec![
            design_row("noise_std=1e-10", "heldout", 5),
            design_row("noise_std=1e-10", "train", 4),
            design_row("noise_std=1e-10", "train", 1),
            design_row("noise_std=1e-10", "heldout", 3),
        ];

        let index_override =
            source_noise_index_override_from_design_rows("noise_std=1e-10", &rows).unwrap();

        assert_eq!(index_override.training_indices, vec![1, 4]);
        assert_eq!(index_override.heldout_indices, vec![3, 5]);
    }

    #[test]
    fn source_noise_index_override_rejects_missing_roles() {
        let rows = vec![design_row("noise_std=1e-10", "train", 1)];

        assert!(source_noise_index_override_from_design_rows("noise_std=1e-10", &rows).is_err());
    }

    #[test]
    fn custom_sweeps_reject_empty_or_oversized_case_grids() {
        assert!(validate_case_limits(ToroidalExactBSweepKind::SourceNoise, 0, 1, 6, 4).is_err());
        assert!(validate_case_limits(ToroidalExactBSweepKind::FieldCoverage, 1, 0, 6, 4).is_err());
        assert!(validate_case_limits(ToroidalExactBSweepKind::Both, 7, 1, 6, 4).is_err());
        assert!(validate_case_limits(ToroidalExactBSweepKind::Both, 1, 5, 6, 4).is_err());
        validate_case_limits(ToroidalExactBSweepKind::Both, 1, 1, 6, 4).unwrap();
    }
}
