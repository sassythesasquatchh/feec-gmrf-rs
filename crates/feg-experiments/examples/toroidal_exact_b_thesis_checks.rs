use feg_case_studies::toroidal_inductor::{
    run_toroidal_exact_b_recovery_experiment,
    toroidal_exact_b_canonical_source_designed_flux_config,
    toroidal_exact_b_prior_tau_for_physical_b_multiplier, ToroidalExactBRecoveryConfig,
    ToroidalExactBRecoveryReport, ToroidalExactBReferenceSolveMode,
    TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT,
};
use std::{error::Error, fs, path::PathBuf, time::Instant};

#[derive(Debug, Clone, Copy)]
struct ThesisCheckCase {
    label: &'static str,
    pde_variance: f64,
    prior_multiplier: f64,
}

#[derive(Debug, Clone, Copy)]
struct SourceMetrics {
    l2_error: f64,
    rmse: f64,
    max_abs_error: f64,
    mean_posterior_sd: f64,
}

#[derive(Debug, Clone, Copy)]
struct HeldoutMetrics {
    mean_posterior_sd: f64,
    max_abs_z: f64,
    rms_z: f64,
    mean_abs_residual: f64,
    mean_predictive_sd: f64,
    noisy_max_abs_z: f64,
    noisy_rms_z: f64,
    noisy_mean_abs_residual: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from("out/examples/toroidal_exact_b_thesis_checks");
    fs::create_dir_all(&out_dir)?;
    let output_path = out_dir.join("thesis_checks_summary.csv");
    let cases = [
        ThesisCheckCase {
            label: "deterministic_pde_pde3e-8_prior10x",
            pde_variance: 3.0e-8,
            prior_multiplier: 10.0,
        },
        ThesisCheckCase {
            label: "deterministic_pde_pde1e-8_prior10x",
            pde_variance: 1.0e-8,
            prior_multiplier: 10.0,
        },
        ThesisCheckCase {
            label: "deterministic_pde_pde3e-8_prior30x",
            pde_variance: 3.0e-8,
            prior_multiplier: 30.0,
        },
        ThesisCheckCase {
            label: "deterministic_pde_pde3e-8_prior100x",
            pde_variance: 3.0e-8,
            prior_multiplier: 100.0,
        },
        ThesisCheckCase {
            label: "deterministic_pde_pde1e-8_prior100x",
            pde_variance: 1.0e-8,
            prior_multiplier: 100.0,
        },
    ];

    let mut csv = thesis_check_header();
    for case in cases {
        println!(
            "running {} (pde_variance={:.3e}, prior={}x)",
            case.label, case.pde_variance, case.prior_multiplier
        );
        let start = Instant::now();
        let config = thesis_check_config(case)?;
        let tau = config.prior_tau;
        match run_toroidal_exact_b_recovery_experiment(&config) {
            Ok(report) => {
                let runtime_seconds = start.elapsed().as_secs_f64();
                print_success(case, &report, runtime_seconds);
                csv.push_str(&thesis_check_success_row(
                    case,
                    &config,
                    &report,
                    runtime_seconds,
                )?);
            }
            Err(error) => {
                let runtime_seconds = start.elapsed().as_secs_f64();
                println!(
                    "  failed after {:.2}s: {}",
                    runtime_seconds,
                    error.replace('\n', " ")
                );
                csv.push_str(&thesis_check_error_row(case, tau, runtime_seconds, &error));
            }
        }
        fs::write(&output_path, &csv)?;
    }

    println!("wrote {}", output_path.display());
    Ok(())
}

fn thesis_check_config(case: ThesisCheckCase) -> Result<ToroidalExactBRecoveryConfig, String> {
    let mut config = toroidal_exact_b_canonical_source_designed_flux_config();
    config.reference_solve_mode = ToroidalExactBReferenceSolveMode::DeterministicPde;
    config.reference_solver_diagonal_shift =
        TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT;
    config.base.pde_variance = case.pde_variance;
    config.prior_tau = toroidal_exact_b_prior_tau_for_physical_b_multiplier(case.prior_multiplier)?;
    config.output_dir = None;
    config.write_outputs = false;
    Ok(config)
}

fn thesis_check_header() -> String {
    "case_label,reference_solve_mode,reference_solver_diagonal_shift,inference_pde_variance,prior_multiplier,prior_tau,prior_calibration_label,train_rows,heldout_rows,source_l2_error,source_rmse,source_max_abs_error,source_mean_posterior_sd,train_rmse,heldout_latent_rmse,heldout_latent_nlpd,heldout_latent_coverage_count,heldout_latent_coverage_fraction,heldout_mean_posterior_sd,heldout_latent_max_abs_z,heldout_latent_rms_z,heldout_latent_mean_abs_residual,heldout_noisy_rmse,heldout_noisy_nlpd,heldout_noisy_coverage_count,heldout_noisy_coverage_fraction,heldout_mean_predictive_sd,heldout_noisy_max_abs_z,heldout_noisy_rms_z,heldout_noisy_mean_abs_residual,b_relative_error,final_residual_norm,source_response_condition,source_response_snr_min,source_response_snr_max,posterior_factor_nnz,runtime_seconds,status,error\n".to_string()
}

fn thesis_check_success_row(
    case: ThesisCheckCase,
    config: &ToroidalExactBRecoveryConfig,
    report: &ToroidalExactBRecoveryReport,
    runtime_seconds: f64,
) -> Result<String, String> {
    let source = source_metrics(report)?;
    let heldout = heldout_metrics(report)?;
    Ok(format!(
        "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.6},ok,\n",
        case.label,
        report.summary.reference_solve_mode.label(),
        report.summary.reference_solver_diagonal_shift,
        config.base.pde_variance,
        case.prior_multiplier,
        report.summary.prior_tau,
        report.summary.prior_calibration_label,
        report.summary.training_rows,
        report.summary.heldout_rows,
        source.l2_error,
        source.rmse,
        source.max_abs_error,
        source.mean_posterior_sd,
        report.summary.train_rmse,
        report.summary.heldout_rmse,
        report.summary.heldout_nlpd,
        report.summary.heldout_covered95,
        report.summary.heldout_coverage_fraction,
        heldout.mean_posterior_sd,
        heldout.max_abs_z,
        heldout.rms_z,
        heldout.mean_abs_residual,
        report.summary.heldout_noisy_rmse,
        report.summary.heldout_noisy_nlpd,
        report.summary.heldout_noisy_covered95,
        report.summary.heldout_noisy_coverage_fraction,
        heldout.mean_predictive_sd,
        heldout.noisy_max_abs_z,
        heldout.noisy_rms_z,
        heldout.noisy_mean_abs_residual,
        csv_option(report.summary.b_relative_error),
        report.summary.final_residual_norm,
        report.source_response.condition,
        report.source_response.snr_min,
        report.source_response.snr_max,
        report.summary.posterior_factor_nnz,
        runtime_seconds
    ))
}

fn thesis_check_error_row(
    case: ThesisCheckCase,
    prior_tau: f64,
    runtime_seconds: f64,
    error: &str,
) -> String {
    format!(
        "{},deterministic_pde,{:.16e},{:.16e},{:.16e},{:.16e},unknown,0,0,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,{:.6},error,{}\n",
        case.label,
        0.0,
        case.pde_variance,
        case.prior_multiplier,
        prior_tau,
        runtime_seconds,
        csv_escape(error)
    )
}

fn print_success(_case: ThesisCheckCase, report: &ToroidalExactBRecoveryReport, elapsed: f64) {
    let source = source_metrics(report).expect("successful report should have source metrics");
    let heldout = heldout_metrics(report).expect("successful report should have heldout metrics");
    println!(
        "  ok {:.2}s: source_l2={:.3e} latent_rmse={:.3e} latent_sd={:.3e} noisy_rmse={:.3e} pred_sd={:.3e} B_rel={:.3e}",
        elapsed,
        source.l2_error,
        report.summary.heldout_rmse,
        heldout.mean_posterior_sd,
        report.summary.heldout_noisy_rmse,
        heldout.mean_predictive_sd,
        report.summary.b_relative_error.unwrap_or(f64::NAN)
    );
}

fn source_metrics(report: &ToroidalExactBRecoveryReport) -> Result<SourceMetrics, String> {
    if report.source_posterior.is_empty() {
        return Err("source posterior is empty".to_string());
    }
    let count = report.source_posterior.len() as f64;
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
        / count;
    Ok(SourceMetrics {
        l2_error,
        rmse: l2_error / count.sqrt(),
        max_abs_error,
        mean_posterior_sd,
    })
}

fn heldout_metrics(report: &ToroidalExactBRecoveryReport) -> Result<HeldoutMetrics, String> {
    if report.heldout_predictions.is_empty() {
        return Err("heldout predictions are empty".to_string());
    }
    let count = report.heldout_predictions.len() as f64;
    let mean_posterior_sd = report
        .heldout_predictions
        .iter()
        .map(|row| row.posterior_sd)
        .sum::<f64>()
        / count;
    let max_abs_z = report
        .heldout_predictions
        .iter()
        .map(|row| row.standardized_residual.abs())
        .fold(0.0, f64::max);
    let rms_z = (report
        .heldout_predictions
        .iter()
        .map(|row| row.standardized_residual * row.standardized_residual)
        .sum::<f64>()
        / count)
        .sqrt();
    let mean_abs_residual = report
        .heldout_predictions
        .iter()
        .map(|row| row.residual.abs())
        .sum::<f64>()
        / count;
    let mean_predictive_sd = report
        .heldout_predictions
        .iter()
        .map(|row| row.predictive_sd)
        .sum::<f64>()
        / count;
    let noisy_max_abs_z = report
        .heldout_predictions
        .iter()
        .map(|row| row.noisy_standardized_residual.abs())
        .fold(0.0, f64::max);
    let noisy_rms_z = (report
        .heldout_predictions
        .iter()
        .map(|row| row.noisy_standardized_residual * row.noisy_standardized_residual)
        .sum::<f64>()
        / count)
        .sqrt();
    let noisy_mean_abs_residual = report
        .heldout_predictions
        .iter()
        .map(|row| row.noisy_residual.abs())
        .sum::<f64>()
        / count;
    Ok(HeldoutMetrics {
        mean_posterior_sd,
        max_abs_z,
        rms_z,
        mean_abs_residual,
        mean_predictive_sd,
        noisy_max_abs_z,
        noisy_rms_z,
        noisy_mean_abs_residual,
    })
}

fn csv_option(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.16e}"))
        .unwrap_or_else(|| "nan".to_string())
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\"").replace('\n', " "))
    } else {
        value.to_string()
    }
}
