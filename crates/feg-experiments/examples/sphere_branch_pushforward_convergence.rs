use feg_case_studies::sphere_branch_pushforward_convergence::{
    run_sphere_branch_pushforward_convergence, SphereBranchPushforwardConvergenceConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let result = run_sphere_branch_pushforward_convergence(&config)?;

    println!("component_rows={}", result.component_variances.len());
    println!("summary_rows={}", result.covariance_summaries.len());
    println!("fit_rows={}", result.fit_summaries.len());
    println!(
        "component_variance={}",
        config.output_dir.join("component_variance.csv").display()
    );
    println!(
        "covariance_summary={}",
        config.output_dir.join("covariance_summary.csv").display()
    );
    println!(
        "fit_summary={}",
        config.output_dir.join("fit_summary.csv").display()
    );
    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<SphereBranchPushforwardConvergenceConfig, String> {
    let mut config = SphereBranchPushforwardConvergenceConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--levels" => {
                let value = next_value(&mut args, "--levels")?;
                config.refinement_levels = parse_levels(&value)?;
            }
            "--out" | "--output-dir" => {
                config.output_dir = PathBuf::from(next_value(&mut args, arg.as_str())?);
            }
            "--kappa" => {
                config.kappa = parse_f64(next_value(&mut args, "--kappa")?, "--kappa")?;
            }
            "--tau" => {
                config.tau = parse_f64(next_value(&mut args, "--tau")?, "--tau")?;
            }
            "--analytic-lmax" | "--lmax" => {
                config.analytic_lmax =
                    parse_usize(next_value(&mut args, arg.as_str())?, arg.as_str())?;
            }
            "--max-cells" => {
                config.max_cells =
                    parse_usize(next_value(&mut args, "--max-cells")?, "--max-cells")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help")),
        }
    }
    Ok(config)
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn parse_levels(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid refinement level `{}`", part.trim()))
        })
        .collect()
}

fn parse_f64(value: String, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("invalid floating-point value `{value}` for {flag}"))
}

fn parse_usize(value: String, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid unsigned integer `{value}` for {flag}"))
}

fn print_usage() {
    println!(
        "Usage: cargo run -p feg-case-studies --release --example sphere_branch_pushforward_convergence -- [options]"
    );
    println!();
    println!("Options:");
    println!("  --levels <n,n,...>       Sphere refinement levels");
    println!("  --out <path>             Output directory");
    println!("  --kappa <value>          Positive Matern kappa");
    println!("  --tau <value>            Positive Matern tau");
    println!("  --analytic-lmax <n>      Analytic spherical harmonic cutoff");
    println!("  --max-cells <n>          Maximum selected barycenter cells per level");
}
