use feg_case_studies::toroidal_inductor::{
    run_toroidal_exact_b_diagnostics, ToroidalExactBDiagnosticObservationMode,
    ToroidalExactBDiagnosticsConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = ToroidalExactBDiagnosticsConfig::default();
    let mut quick = false;
    let mut force_source_response = false;
    let mut pde_sweep_only = false;
    let mut custom_pde_variances = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--quick" => quick = true,
            "--source-response" => force_source_response = true,
            "--pde-sweep-only" => pde_sweep_only = true,
            "--mesh" => config.base.base.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?));
            }
            "--observation-mode" => {
                config.observation_modes = vec![parse_observation_mode(&next_arg(
                    &mut args,
                    "--observation-mode",
                )?)?];
            }
            "--observation-modes" => {
                config.observation_modes =
                    parse_observation_modes(&next_arg(&mut args, "--observation-modes")?)?;
            }
            "--pde-variances" => {
                config.pde_variances = parse_f64_list(&next_arg(&mut args, "--pde-variances")?)?;
                custom_pde_variances = true;
            }
            "--surface-flux-azimuth-count" => {
                config.base.surface_flux_azimuth_count =
                    next_arg(&mut args, "--surface-flux-azimuth-count")?.parse()?;
            }
            "--heldout-count" => {
                config.base.heldout_count = next_arg(&mut args, "--heldout-count")?.parse()?;
            }
            "--seed" => {
                config.base.observation_seed = next_arg(&mut args, "--seed")?.parse()?;
            }
            "--skip-output" => config.write_outputs = false,
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }

    if quick {
        config.pde_variances = vec![1e-8, 1e-6];
        config.prior_taus = vec![1e-6, 1e-5];
        config.source_prior_stds = vec![0.25, 2.5];
        config.observation_noise_stds = vec![1e-10];
        config.include_source_response = false;
    }
    if pde_sweep_only {
        if !custom_pde_variances {
            config.pde_variances = vec![1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5];
        }
        config.prior_taus = vec![config.base.prior_tau];
        config.source_prior_stds = vec![config.base.source_prior_std];
        config.observation_noise_stds = vec![config.base.observation_noise_std];
    }
    if force_source_response {
        config.include_source_response = true;
    }

    let report = run_toroidal_exact_b_diagnostics(&config)?;
    println!("Toroidal exact B=dA diagnostics");
    println!("  rows={}", report.rows.len());
    for row in &report.rows {
        println!(
            "  {} {} train={:.3e} heldout={:.3e} Berr={:.3e} eta=[{:.3e},{:.3e},{:.3e},{:.3e}] sd=[{:.3e},{:.3e},{:.3e},{:.3e}]",
            row.observation_mode.label(),
            row.sweep,
            row.train_rmse,
            row.heldout_rmse,
            row.b_relative_error,
            row.eta_posterior_mean[0],
            row.eta_posterior_mean[1],
            row.eta_posterior_mean[2],
            row.eta_posterior_mean[3],
            row.eta_posterior_variance[0].max(0.0).sqrt(),
            row.eta_posterior_variance[1].max(0.0).sqrt(),
            row.eta_posterior_variance[2].max(0.0).sqrt(),
            row.eta_posterior_variance[3].max(0.0).sqrt(),
        );
    }
    if let Some(output_dir) = &config.output_dir {
        if config.write_outputs {
            println!("  outputs: {}", output_dir.display());
        }
    }
    Ok(())
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_observation_modes(
    value: &str,
) -> Result<Vec<ToroidalExactBDiagnosticObservationMode>, Box<dyn Error>> {
    value
        .split(',')
        .map(|part| parse_observation_mode(part.trim()))
        .collect()
}

fn parse_observation_mode(
    value: &str,
) -> Result<ToroidalExactBDiagnosticObservationMode, Box<dyn Error>> {
    match value {
        "cell"
        | "cells"
        | "cell-components"
        | "cell_components"
        | "cell-magnetic-components"
        | "cell_magnetic_components" => {
            Ok(ToroidalExactBDiagnosticObservationMode::CellMagneticComponents)
        }
        "flux" | "surface-flux" | "surface_flux" | "surface-fluxes" | "surface_fluxes" => {
            Ok(ToroidalExactBDiagnosticObservationMode::SurfaceFluxes)
        }
        "source-designed-flux"
        | "source_designed_flux"
        | "source-designed-fluxes"
        | "source_designed_fluxes"
        | "designed-flux"
        | "designed_flux" => Ok(ToroidalExactBDiagnosticObservationMode::SourceDesignedFluxes),
        "oracle" | "full-field" | "full_field" | "full-field-oracle" | "full_field_oracle" => {
            Ok(ToroidalExactBDiagnosticObservationMode::FullFieldOracle)
        }
        "pde" | "pde-only" | "pde_only" => Ok(ToroidalExactBDiagnosticObservationMode::PdeOnly),
        other => Err(format!(
            "unknown observation mode `{other}`; expected cell-components, surface-flux, source-designed-flux, full-field-oracle, or pde-only"
        )
        .into()),
    }
}

fn parse_f64_list(value: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    value
        .split(',')
        .map(|part| {
            let trimmed = part.trim();
            let parsed = trimmed.parse::<f64>()?;
            if !parsed.is_finite() || parsed <= 0.0 {
                Err(format!("list value `{trimmed}` must be finite and positive").into())
            } else {
                Ok(parsed)
            }
        })
        .collect()
}

fn print_help() {
    println!("Usage: toroidal_exact_b_diagnostics [options]");
    println!("  --quick");
    println!("  --source-response");
    println!("  --pde-sweep-only");
    println!("  --mesh <path>");
    println!("  --observation-mode <cell-components|surface-flux|source-designed-flux|full-field-oracle|pde-only>");
    println!("  --observation-modes <comma-separated modes>");
    println!("  --pde-variances <comma-separated positive floats>");
    println!("  --surface-flux-azimuth-count <usize>");
    println!("  --heldout-count <usize>");
    println!("  --seed <u64>");
    println!("  --output-dir <path>");
    println!("  --skip-output");
}
