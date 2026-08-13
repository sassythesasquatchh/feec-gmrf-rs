use feg_case_studies::genus2_nondecomposed_period_variance::{
    compute_genus2_nondecomposed_period_variance, Genus2NondecomposedPeriodVarianceConfig,
};
use std::{env, path::PathBuf};

fn main() -> Result<(), String> {
    let config = parse_args(env::args().skip(1).collect())?;
    let report = compute_genus2_nondecomposed_period_variance(config.clone())?;
    println!("mesh={}", config.mesh_path.display());
    println!(
        "topology: vertices={} edges={} faces={} chi={} b0={} b1={} b2={}",
        report.topology_summary.vertex_count,
        report.topology_summary.edge_count,
        report.topology_summary.face_count,
        report.topology_summary.euler_characteristic,
        report.topology_summary.b0,
        report.topology_summary.b1,
        report.topology_summary.b2
    );
    println!(
        "kappa={:.6e} tau={:.6e} local_noise_std={:.6e} cycle_noise_std={:.6e}",
        report.kappa, report.tau, report.local_noise_std, report.cycle_noise_std
    );
    println!("scenario,local_observations,cycle_observations,cycle_index,prior_variance,posterior_variance,variance_ratio");
    for row in &report.rows {
        println!(
            "{},{},{},{},{:.12e},{:.12e},{:.12e}",
            row.scenario.as_str(),
            row.local_observation_count,
            row.cycle_observation_count,
            row.cycle_index,
            row.prior_variance,
            row.posterior_variance,
            row.variance_ratio
        );
    }
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Genus2NondecomposedPeriodVarianceConfig, String> {
    let mut config = Genus2NondecomposedPeriodVarianceConfig::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--mesh" => {
                index += 1;
                config.mesh_path = PathBuf::from(arg(&args, index, "--mesh")?);
            }
            "--kappa" => {
                index += 1;
                config.kappa = parse_f64(arg(&args, index, "--kappa")?, "--kappa")?;
            }
            "--tau" => {
                index += 1;
                config.tau = parse_f64(arg(&args, index, "--tau")?, "--tau")?;
            }
            "--local-observations" => {
                index += 1;
                config.local_observation_count = parse_usize(
                    arg(&args, index, "--local-observations")?,
                    "--local-observations",
                )?;
            }
            "--local-noise-std" => {
                index += 1;
                config.local_noise_std =
                    parse_f64(arg(&args, index, "--local-noise-std")?, "--local-noise-std")?;
            }
            "--cycle-noise-std" => {
                index += 1;
                config.cycle_noise_std =
                    parse_f64(arg(&args, index, "--cycle-noise-std")?, "--cycle-noise-std")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
        index += 1;
    }
    Ok(config)
}

fn arg<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} expects an unsigned integer"))
}

fn parse_f64(value: &str, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{flag} expects a floating-point value"))
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example genus2_nondecomposed_period_variance -- [options]\n\
         Options:\n\
           --mesh <path>                    Genus-2 .msh path\n\
           --kappa <f64>                    Nondecomposed prior kappa (default: 4.0)\n\
           --tau <f64>                      Nondecomposed prior tau (default: 1.0)\n\
           --local-observations <usize>     Local edge observations (default: 24)\n\
           --local-noise-std <f64>          Local edge noise std (default: 1e-2)\n\
           --cycle-noise-std <f64>          Cycle observation noise std (default: 1e-4)"
    );
}
