use feg_case_studies::cube_zero_form_kernel_validation::{
    compute_cube_zero_form_kernel_validation_report,
    write_cube_zero_form_kernel_validation_outputs, CubeZeroFormKernelValidationConfig,
};
use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
struct CliConfig {
    out_dir: PathBuf,
    experiment: CubeZeroFormKernelValidationConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("out/matern_0form_cube_kernel_validation"),
            experiment: CubeZeroFormKernelValidationConfig::default(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_args()?;
    let _ = std::fs::remove_dir_all(&cli.out_dir);
    let report = compute_cube_zero_form_kernel_validation_report(cli.experiment)
        .map_err(|err| format!("cube 0-form kernel validation failed: {err}"))?;
    write_cube_zero_form_kernel_validation_outputs(&report, &cli.out_dir)?;

    println!("0-form cube Matern kernel validation");
    println!(
        "hyperparameters: dim={} alpha={:.3} nu={:.3} sigma2={:.3} range={:.3} kappa={:.6} tau={:.8e} noise={:.1e} probe_anchor={}",
        report.hyperparameters.dimension,
        report.hyperparameters.alpha,
        report.hyperparameters.nu,
        report.hyperparameters.sigma2,
        report.hyperparameters.practical_range,
        report.hyperparameters.kappa,
        report.hyperparameters.tau,
        report.hyperparameters.noise_variance,
        report.config.probe_anchor_level
    );
    if report.config.include_spectral && !report.spectral_available {
        println!("spectral method skipped: PETSc eigen solver binary was not found");
    }
    for level in &report.levels {
        println!(
            "level={} ndofs={} eval={} observations={} pairs={} h/range={:.3} margin/range={:.3} tau_calibration={:.4}",
            level.level,
            level.ndofs,
            level.eval_indices.len(),
            level.observation_indices.len(),
            level.correlation_pairs.len(),
            level.diagnostics.h_over_range,
            level.diagnostics.margin_over_range,
            level.diagnostics.gmrf_tau_calibration_multiplier
        );
        for metric in &level.metrics {
            if metric.method == "euclidean_reference" {
                continue;
            }
            let k = metric
                .spectral_k
                .map(|value| format!(" k={value}"))
                .unwrap_or_default();
            println!(
                "  {}{} variance_rmse={:.6e} rel_rmse={:.6e} max_abs={:.6e} total={:.3}s",
                metric.method,
                k,
                metric.variance_rmse,
                metric.relative_variance_rmse,
                metric.max_abs_variance_error,
                metric.total_seconds
            );
        }
    }
    if !report.spectral_sweep.is_empty() {
        println!(
            "spectral k sweep rows={} at level={}",
            report.spectral_sweep.len(),
            report.config.spectral_sweep_level
        );
    }
    println!("wrote outputs to {}", cli.out_dir.display());

    Ok(())
}

fn parse_args() -> Result<CliConfig, String> {
    let mut cli = CliConfig::default();
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                cli.out_dir = PathBuf::from(next_value(&mut args, "--out-dir")?);
            }
            "--levels" => {
                cli.experiment.levels = parse_usize_list(&next_value(&mut args, "--levels")?)?;
            }
            "--probe-anchor-level" => {
                cli.experiment.probe_anchor_level = next_value(&mut args, "--probe-anchor-level")?
                    .parse()
                    .map_err(|_| "--probe-anchor-level expects a positive integer".to_string())?;
            }
            "--spectral-k" => {
                cli.experiment.spectral_k = next_value(&mut args, "--spectral-k")?
                    .parse()
                    .map_err(|_| "--spectral-k expects a positive integer".to_string())?;
            }
            "--spectral-sweep-level" => {
                cli.experiment.spectral_sweep_level =
                    next_value(&mut args, "--spectral-sweep-level")?
                        .parse()
                        .map_err(|_| {
                            "--spectral-sweep-level expects a positive integer".to_string()
                        })?;
            }
            "--spectral-sweep-ks" => {
                cli.experiment.spectral_sweep_ks =
                    parse_usize_list(&next_value(&mut args, "--spectral-sweep-ks")?)?;
            }
            "--max-correlation-pairs" => {
                cli.experiment.max_correlation_pairs =
                    next_value(&mut args, "--max-correlation-pairs")?
                        .parse()
                        .map_err(|_| {
                            "--max-correlation-pairs expects a positive integer".to_string()
                        })?;
            }
            "--correlation-bins" => {
                cli.experiment.correlation_bin_count = next_value(&mut args, "--correlation-bins")?
                    .parse()
                    .map_err(|_| "--correlation-bins expects a positive integer".to_string())?;
            }
            "--interior-margin" => {
                cli.experiment.interior_margin = next_value(&mut args, "--interior-margin")?
                    .parse()
                    .map_err(|_| "--interior-margin expects a float".to_string())?;
            }
            "--range" => {
                cli.experiment.practical_range = next_value(&mut args, "--range")?
                    .parse()
                    .map_err(|_| "--range expects a float".to_string())?;
            }
            "--sigma2" => {
                cli.experiment.sigma2 = next_value(&mut args, "--sigma2")?
                    .parse()
                    .map_err(|_| "--sigma2 expects a float".to_string())?;
            }
            "--noise-variance" => {
                cli.experiment.noise_variance = next_value(&mut args, "--noise-variance")?
                    .parse()
                    .map_err(|_| "--noise-variance expects a float".to_string())?;
            }
            "--no-spectral" => {
                cli.experiment.include_spectral = false;
                cli.experiment.require_spectral = false;
            }
            "--require-spectral" => {
                cli.experiment.include_spectral = true;
                cli.experiment.require_spectral = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; pass --help for usage")),
        }
    }
    Ok(cli)
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, String> {
    let parsed = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid positive integer `{}`", part.trim()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() || parsed.contains(&0) {
        return Err("list must contain positive integers".to_string());
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example matern_0form_cube_kernel_validation -- [options]\n\
\n\
Options:\n\
  --out-dir <path>                 Output directory (default: out/matern_0form_cube_kernel_validation)\n\
  --levels <csv>                   Cube cells-per-axis levels (default: 12,24,36,48)\n\
  --probe-anchor-level <n>         Anchor level for fixed probe grid (default: 12)\n\
  --spectral-k <n>                 Main spectral mode count cap (default: 64)\n\
  --spectral-sweep-level <n>       Mesh level for spectral k sweep (default: 12)\n\
  --spectral-sweep-ks <csv>        Spectral k sweep values (default: 64,128,256,512)\n\
  --max-correlation-pairs <n>      Deterministic pair sample cap (default: 12000)\n\
  --correlation-bins <n>           Distance-bin count (default: 32)\n\
  --interior-margin <x>            Interior-only margin from boundary (default: 0.25)\n\
  --range <rho>                    Practical Matern range (default: 0.20)\n\
  --sigma2 <x>                     Target Euclidean marginal variance (default: 1.0)\n\
  --noise-variance <x>             Observation noise variance (default: 1e-4)\n\
  --no-spectral                    Skip spectral validation even if PETSc is available\n\
  --require-spectral               Fail if PETSc eigen solver is unavailable\n"
    );
}
