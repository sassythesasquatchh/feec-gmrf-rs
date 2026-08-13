use feg_case_studies::toroidal_inductor::{
    run_toroidal_exact_b_recovery_experiment,
    toroidal_exact_b_canonical_source_designed_flux_config, ToroidalExactBRecoveryConfig,
    ToroidalExactBRecoveryReport,
};
use std::error::Error;

const SOURCE_ETA_L2_ACCEPTANCE: f64 = 3e-2;
const B_RELATIVE_ERROR_ACCEPTANCE: f64 = 2e-2;

#[derive(Debug, Clone, Copy)]
struct AcceptanceMetrics {
    eta_l2: f64,
    b_relative_error: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = toroidal_exact_b_canonical_source_designed_flux_config();

    let report = run_toroidal_exact_b_recovery_experiment(&config)?;
    let metrics = acceptance_metrics(&report);
    validate_acceptance(&report, metrics)?;

    print_report(&config, &report, metrics);
    Ok(())
}

fn acceptance_metrics(report: &ToroidalExactBRecoveryReport) -> AcceptanceMetrics {
    let eta_l2 = report
        .source_posterior
        .iter()
        .map(|row| row.error * row.error)
        .sum::<f64>()
        .sqrt();
    let b_relative_error = report.summary.b_relative_error.unwrap_or(f64::NAN);
    AcceptanceMetrics {
        eta_l2,
        b_relative_error,
    }
}

fn validate_acceptance(
    report: &ToroidalExactBRecoveryReport,
    metrics: AcceptanceMetrics,
) -> Result<(), String> {
    let finite_intervals = report.heldout_predictions.iter().all(|row| {
        row.posterior_sd.is_finite()
            && row.posterior_sd >= 0.0
            && row.predictive_sd.is_finite()
            && row.predictive_sd >= 0.0
            && row.lower95.is_finite()
            && row.upper95.is_finite()
            && row.standardized_residual.is_finite()
            && row.noisy_lower95.is_finite()
            && row.noisy_upper95.is_finite()
            && row.noisy_standardized_residual.is_finite()
    });

    if report.source_posterior.len() != 4 {
        return Err("expected four source posterior rows".to_string());
    }
    if report.summary.training_rows != 12 || report.summary.heldout_rows != 24 {
        return Err(format!(
            "expected 12 training fluxes and 24 heldout fluxes, got {} and {}",
            report.summary.training_rows, report.summary.heldout_rows
        ));
    }
    if !finite_intervals {
        return Err("heldout predictive intervals must be finite".to_string());
    }
    if metrics.eta_l2 > SOURCE_ETA_L2_ACCEPTANCE {
        return Err(format!(
            "source eta L2 error {:.3e} exceeds target",
            metrics.eta_l2
        ));
    }
    if !metrics.b_relative_error.is_finite()
        || metrics.b_relative_error > B_RELATIVE_ERROR_ACCEPTANCE
    {
        return Err(format!(
            "relative B error {:.3e} exceeds target",
            metrics.b_relative_error
        ));
    }
    if !report.source_response.condition.is_finite() || report.source_response.condition > 100.0 {
        return Err(format!(
            "source-response condition {:.3e} exceeds target",
            report.source_response.condition
        ));
    }
    Ok(())
}

fn print_report(
    config: &ToroidalExactBRecoveryConfig,
    report: &ToroidalExactBRecoveryReport,
    metrics: AcceptanceMetrics,
) {
    println!("Toroidal exact B=dA source-designed flux workflow");
    println!(
        "  weights: inference_pde_variance={:.3e} reference_pde_variance={:.3e} observation_noise_std={:.3e}",
        config.base.pde_variance,
        config.reference_pde_variance.unwrap_or(config.base.pde_variance),
        config.observation_noise_std
    );
    println!(
        "  reference: mode={} solve={} diagonal_shift={:.3e}",
        report.summary.reference_mode.label(),
        report.summary.reference_solve_mode.label(),
        report.summary.reference_solver_diagonal_shift
    );
    println!(
        "  prior: tau={:.6e} calibration={}",
        report.summary.prior_tau, report.summary.prior_calibration_label
    );
    println!(
        "  rows: active_dofs={} source_modes={} train={} heldout={} residual_rows={}/{}",
        report.summary.active_dofs,
        report.summary.source_modes,
        report.summary.training_rows,
        report.summary.heldout_rows,
        report.summary.residual_rows_used,
        report.summary.residual_rows_total
    );
    println!(
        "  heldout latent: rmse={:.6e} mean_sd={:.6e} max|z|={:.3e} coverage={}/{} ({:.3})",
        report.summary.heldout_rmse,
        report.summary.heldout_mean_posterior_flux_sd,
        report.summary.heldout_max_abs_z,
        report.summary.heldout_covered95,
        report.summary.heldout_rows,
        report.summary.heldout_coverage_fraction
    );
    println!(
        "  heldout noisy: rmse={:.6e} pred_sd={:.6e} max|z|={:.3e} coverage={}/{} ({:.3})",
        report.summary.heldout_noisy_rmse,
        report.summary.heldout_mean_predictive_sd,
        report.summary.heldout_noisy_max_abs_z,
        report.summary.heldout_noisy_covered95,
        report.summary.heldout_rows,
        report.summary.heldout_noisy_coverage_fraction
    );
    println!(
        "  source: eta_l2={:.6e} B_relative_error={:.6e} condition={:.3e} snr_min={:.3e} snr_max={:.3e}",
        metrics.eta_l2,
        metrics.b_relative_error,
        report.source_response.condition,
        report.source_response.snr_min,
        report.source_response.snr_max
    );
    for row in &report.source_posterior {
        println!(
            "  eta{} truth={:+.6e} mean={:+.6e} sd={:.3e} error={:+.6e}",
            row.mode_index,
            row.truth,
            row.posterior_mean,
            row.posterior_variance.max(0.0).sqrt(),
            row.error
        );
    }

    if let Some(out_dir) = &config.output_dir {
        println!("  outputs: {}", out_dir.display());
    }
}
