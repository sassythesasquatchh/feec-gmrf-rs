use feg_case_studies::torus::posterior_residual_weight::{
    default_torus_shell_mesh_path, run_torus_1form_pde_posterior_mean_weight_sweep,
    write_torus_1form_pde_posterior_mean_weight_sweep_outputs, Torus1FormPdeMeshLevel,
    Torus1FormPdePosteriorMeanWeightSweepConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, out_dir) = parse_args()?;
    let total_start = Instant::now();

    let result = run_torus_1form_pde_posterior_mean_weight_sweep(&config)?;
    write_torus_1form_pde_posterior_mean_weight_sweep_outputs(&result, &out_dir)?;

    println!("Torus 1-form PDE posterior mean weight convergence");
    println!(
        "mesh_levels={}",
        config
            .mesh_levels
            .iter()
            .map(|level| format!("{}:{}", level.resolution, level.mesh_path.display()))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "kappa={} tau={} weights={}",
        config.kappa,
        config.tau,
        config
            .weights
            .iter()
            .map(|weight| format!("{weight:.3e}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "| {:>3} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} |",
        "res", "weight", "noise var", "ref rel L2", "cont rel L2", "rel residual"
    );
    println!(
        "| {:-<3} | {:-<12} | {:-<12} | {:-<12} | {:-<12} | {:-<12} |",
        "", "", "", "", "", ""
    );
    for row in &result.rows {
        println!(
            "| {:>3} | {:>12.3e} | {:>12.3e} | {:>12.3e} | {:>12.3e} | {:>12.3e} |",
            row.resolution,
            row.weight,
            row.noise_variance,
            row.posterior_deterministic_relative_l2_error,
            row.posterior_continuum_relative_l2_error,
            row.posterior_relative_residual_norm,
        );
    }
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn parse_args(
) -> Result<(Torus1FormPdePosteriorMeanWeightSweepConfig, PathBuf), Box<dyn std::error::Error>> {
    let mut config = Torus1FormPdePosteriorMeanWeightSweepConfig::default();
    let mut out_dir = PathBuf::from("out/matern_1form_torus_pde_posterior_mean_convergence");
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-path" => {
                config.mesh_levels = vec![Torus1FormPdeMeshLevel {
                    resolution: 0,
                    mesh_path: PathBuf::from(
                        args.next()
                            .ok_or_else(|| invalid_input("missing value for --mesh-path"))?,
                    ),
                }];
            }
            "--mesh-levels" => {
                config.mesh_levels = parse_mesh_levels_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --mesh-levels"))?,
                )?;
            }
            "--kappa" => {
                config.kappa = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --kappa"))?,
                    "--kappa",
                )?;
            }
            "--tau" => {
                config.tau = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --tau"))?,
                    "--tau",
                )?;
            }
            "--weights" => {
                config.weights = parse_weights_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --weights"))?,
                )?;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --out-dir"))?,
                );
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(invalid_input(format!("unrecognized argument `{other}`")).into());
            }
        }
    }

    Ok((config, out_dir))
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example matern_1form_torus_pde_posterior_mean_convergence -- [options]"
    );
    println!("Options:");
    println!("  --mesh-levels <csv>         Built-in torus resolution levels, e.g. 0,1,2,3");
    println!("  --mesh-path <path>          Custom single input torus mesh path");
    println!("  --kappa <f64>               Matérn kappa parameter");
    println!("  --tau <f64>                 Matérn tau parameter");
    println!("  --weights <csv>             Observation precision weights, e.g. 1e2,1e4,1e6");
    println!("  --out-dir <path>            Output directory");
}

fn parse_f64_arg(value: String, flag: &str) -> Result<f64, Box<dyn std::error::Error>> {
    value
        .parse::<f64>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")).into())
}

fn parse_weights_arg(value: String) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if value.trim().is_empty() {
        return Err(invalid_input("--weights must not be empty").into());
    }
    value
        .split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return Err(invalid_input("--weights contains an empty entry").into());
            }
            trimmed
                .parse::<f64>()
                .map_err(|err| invalid_input(format!("invalid value in --weights: {err}")).into())
        })
        .collect()
}

fn parse_mesh_levels_arg(
    value: String,
) -> Result<Vec<Torus1FormPdeMeshLevel>, Box<dyn std::error::Error>> {
    if value.trim().is_empty() {
        return Err(invalid_input("--mesh-levels must not be empty").into());
    }
    value
        .split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return Err(invalid_input("--mesh-levels contains an empty entry").into());
            }
            let resolution = trimmed
                .parse::<usize>()
                .map_err(|err| invalid_input(format!("invalid value in --mesh-levels: {err}")))?;
            Ok(Torus1FormPdeMeshLevel {
                resolution,
                mesh_path: default_torus_shell_mesh_path(resolution),
            })
        })
        .collect()
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
