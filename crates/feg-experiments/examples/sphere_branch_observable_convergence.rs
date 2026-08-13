use feg_case_studies::sphere_branch_observable_convergence::{
    run_sphere_branch_observable_convergence, SphereBranchObservableConvergenceConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let result = run_sphere_branch_observable_convergence(&config)?;

    println!("observable_rows={}", result.variance_rows.len());
    println!("summary_rows={}", result.summary_rows.len());
    println!("fit_rows={}", result.fit_summary_rows.len());
    println!("pointwise_rows={}", result.pointwise_rows.len());
    println!(
        "pointwise_summary_rows={}",
        result.pointwise_summary_rows.len()
    );
    println!(
        "pointwise_fit_rows={}",
        result.pointwise_fit_summary_rows.len()
    );
    println!(
        "observable_variance={}",
        config.output_dir.join("observable_variance.csv").display()
    );
    println!(
        "summary={}",
        config.output_dir.join("summary.csv").display()
    );
    println!(
        "fit_summary={}",
        config.output_dir.join("fit_summary.csv").display()
    );
    println!(
        "pointwise_variance={}",
        config.output_dir.join("pointwise_variance.csv").display()
    );
    println!(
        "pointwise_summary={}",
        config.output_dir.join("pointwise_summary.csv").display()
    );
    println!(
        "pointwise_fit_summary={}",
        config
            .output_dir
            .join("pointwise_fit_summary.csv")
            .display()
    );
    println!("readme={}", config.output_dir.join("README.md").display());
    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<SphereBranchObservableConvergenceConfig, String> {
    let mut config = SphereBranchObservableConvergenceConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--levels" => {
                let value = next_value(&mut args, "--levels")?;
                config.levels = parse_levels(&value)?;
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
            "--lmax" => {
                config.lmax = parse_usize(next_value(&mut args, "--lmax")?, "--lmax")?;
            }
            "--pointwise-analytic-lmax" => {
                config.pointwise_analytic_lmax = parse_usize(
                    next_value(&mut args, "--pointwise-analytic-lmax")?,
                    "--pointwise-analytic-lmax",
                )?;
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
        "Usage: cargo run -p feg-case-studies --release --example sphere_branch_observable_convergence -- [options]"
    );
    println!();
    println!("Options:");
    println!("  --levels <n,n,...>       Sphere refinement levels (default: 1,2,3,4,5)");
    println!("  --out <path>             Output directory");
    println!("  --kappa <value>          Positive Matern kappa (default: 1)");
    println!("  --tau <value>            Positive Matern tau (default: 1)");
    println!("  --lmax <n>               Maximum harmonic degree, 1..=3 (default: 3)");
    println!(
        "  --pointwise-analytic-lmax <n>  Analytic kernel truncation for pointwise diagnostics (default: 400)"
    );
}
