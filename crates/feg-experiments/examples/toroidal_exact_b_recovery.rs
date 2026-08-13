use feg_case_studies::toroidal_inductor::{
    run_toroidal_exact_b_recovery_experiment, ToroidalExactBObservationMode,
    ToroidalExactBRecoveryConfig, ToroidalExactBReferenceMode,
};
use feg_infer::linear_pde::LinearPdePrecisionPolicy;
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = ToroidalExactBRecoveryConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--mesh" => config.base.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?));
            }
            "--reference-csv" => {
                config.reference_observation_csv_path =
                    Some(PathBuf::from(next_arg(&mut args, "--reference-csv")?));
            }
            "--mode" => {
                config.reference_mode = parse_mode(&next_arg(&mut args, "--mode")?)?;
            }
            "--observation-mode" => {
                config.observation_mode =
                    parse_observation_mode(&next_arg(&mut args, "--observation-mode")?)?;
            }
            "--pde-variance" => {
                config.base.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?;
            }
            "--prior-kappa" => {
                config.prior_kappa = next_arg(&mut args, "--prior-kappa")?.parse()?;
            }
            "--prior-tau" => {
                config.prior_tau = next_arg(&mut args, "--prior-tau")?.parse()?;
            }
            "--source-prior-std" => {
                config.source_prior_std = next_arg(&mut args, "--source-prior-std")?.parse()?;
            }
            "--train-fraction" => {
                config.observation_train_fraction =
                    next_arg(&mut args, "--train-fraction")?.parse()?;
            }
            "--noise-std" => {
                config.observation_noise_std = next_arg(&mut args, "--noise-std")?.parse()?;
            }
            "--heldout-count" => {
                config.heldout_count = next_arg(&mut args, "--heldout-count")?.parse()?;
            }
            "--surface-flux-azimuth-count" => {
                config.surface_flux_azimuth_count =
                    next_arg(&mut args, "--surface-flux-azimuth-count")?.parse()?;
                config.observation_mode = ToroidalExactBObservationMode::SurfaceFluxes;
            }
            "--seed" => {
                config.observation_seed = next_arg(&mut args, "--seed")?.parse()?;
            }
            "--source-deltas" => {
                config.source_deltas =
                    parse_source_deltas(&next_arg(&mut args, "--source-deltas")?)?;
                config.reference_mode = ToroidalExactBReferenceMode::PerturbedSource;
            }
            "--skip-output" => config.write_outputs = false,
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    if matches!(
        config.observation_mode,
        ToroidalExactBObservationMode::SourceDesignedFluxes
    ) {
        config.base.precision_policy = LinearPdePrecisionPolicy::DiagonalEquilibrated {
            max_relative_asymmetry: 1.0e-10,
        };
    }

    let report = run_toroidal_exact_b_recovery_experiment(&config)?;
    println!("Toroidal exact B=dA recovery");
    println!(
        "  mode={} observations={} active_dofs={} train={} heldout={} residual_rows={}/{}",
        report.summary.reference_mode.label(),
        report.summary.observation_mode.label(),
        report.summary.active_dofs,
        report.summary.training_rows,
        report.summary.heldout_rows,
        report.summary.residual_rows_used,
        report.summary.residual_rows_total
    );
    println!(
        "  train_rmse={:.6e} heldout_rmse={:.6e} heldout_nlpd={:.6e} coverage={}/{}",
        report.summary.train_rmse,
        report.summary.heldout_rmse,
        report.summary.heldout_nlpd,
        report.summary.heldout_covered95,
        report.summary.heldout_rows
    );
    for row in &report.source_posterior {
        println!(
            "  eta{} truth={:.6e} posterior={:.6e} +/- {:.3e}",
            row.mode_index,
            row.truth,
            row.posterior_mean,
            row.posterior_variance.max(0.0).sqrt()
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

fn parse_observation_mode(value: &str) -> Result<ToroidalExactBObservationMode, Box<dyn Error>> {
    value
        .parse::<ToroidalExactBObservationMode>()
        .map_err(|err| err.into())
}

fn parse_mode(value: &str) -> Result<ToroidalExactBReferenceMode, Box<dyn Error>> {
    match value {
        "nominal" | "nominal-debug" | "nominal_debug" => {
            Ok(ToroidalExactBReferenceMode::NominalDebug)
        }
        "perturbed" | "perturbed-source" | "perturbed_source" => {
            Ok(ToroidalExactBReferenceMode::PerturbedSource)
        }
        other => Err(
            format!("unknown mode `{other}`; expected nominal-debug or perturbed-source").into(),
        ),
    }
}

fn parse_source_deltas(value: &str) -> Result<[f64; 4], Box<dyn Error>> {
    let parts = value
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 4 {
        return Err("source deltas must contain exactly four comma-separated values".into());
    }
    Ok([parts[0], parts[1], parts[2], parts[3]])
}

fn print_help() {
    println!("Usage: toroidal_exact_b_recovery [options]");
    println!("  --mode <nominal-debug|perturbed-source>");
    println!("  --observation-mode <cell-components|surface-flux|source-designed-flux>");
    println!("  --mesh <path>");
    println!("  --reference-csv <path>");
    println!("  --output-dir <path>");
    println!("  --pde-variance <float>");
    println!("  --prior-kappa <float>");
    println!("  --prior-tau <float>");
    println!("  --source-prior-std <float>");
    println!("  --source-deltas <d0,d1,d2,d3>");
    println!("  --train-fraction <float>");
    println!("  --noise-std <float>");
    println!("  --heldout-count <usize>");
    println!("  --surface-flux-azimuth-count <usize>");
    println!("  --seed <u64>");
    println!("  --skip-output");
}
