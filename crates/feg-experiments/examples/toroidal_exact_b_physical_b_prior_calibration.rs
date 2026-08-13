use feg_case_studies::toroidal_inductor::{
    toroidal_exact_b_canonical_source_designed_flux_config,
    toroidal_exact_b_physical_b_prior_calibration,
    write_toroidal_exact_b_physical_b_prior_calibration_csv,
    TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER,
};
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = toroidal_exact_b_canonical_source_designed_flux_config();
    config.prior_tau = 1.0;
    config.output_dir = None;
    config.write_outputs = false;

    let out_dir = PathBuf::from("out/examples/toroidal_exact_b_physical_b_prior_calibration");
    let start = Instant::now();
    let report = toroidal_exact_b_physical_b_prior_calibration(
        &config,
        TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER,
    )
    .map_err(|err| format!("physical B prior calibration failed: {err}"))?;
    write_toroidal_exact_b_physical_b_prior_calibration_csv(
        &report,
        out_dir.join("physical_b_prior_calibration.csv"),
    )?;

    println!("Exact B=dA physical-B prior calibration");
    println!("  output: {}", out_dir.display());
    println!(
        "  reference_solve: {} diagonal_shift={:.3e}",
        report.reference_solve_mode.label(),
        report.reference_solver_diagonal_shift
    );
    println!("  active_dofs: {}", report.active_dofs);
    println!("  cells: {}", report.cells);
    println!("  nominal_B_rms: {:.6e}", report.nominal_b_rms);
    println!("  target_prior_B_rms: {:.6e}", report.target_prior_b_rms);
    println!("  raw_trace: {:.6e}", report.raw_trace);
    println!("  precision_scale: {:.6e}", report.precision_scale);
    println!("  effective_prior_tau: {:.16e}", report.effective_prior_tau);
    println!("  elapsed: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
