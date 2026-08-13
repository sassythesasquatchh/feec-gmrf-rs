use feg_case_studies::nc1_matern_convergence::{
    run_nc1_matern_convergence_experiment, Nc1MaternConvergenceConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let result = run_nc1_matern_convergence_experiment(&config)?;

    println!("summary_rows={}", result.summary_rows.len());
    println!("covariance_rows={}", result.covariance_rows.len());
    println!("fit_rows={}", result.fit_summary_rows.len());
    println!(
        "summary={}",
        config.output_dir.join("summary.csv").display()
    );
    println!(
        "covariance_entries={}",
        config.output_dir.join("covariance_entries.csv").display()
    );
    println!(
        "fit_summary={}",
        config.output_dir.join("fit_summary.csv").display()
    );
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Nc1MaternConvergenceConfig, String> {
    let mut config = Nc1MaternConvergenceConfig::default();
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
                .map_err(|_| format!("invalid mesh level `{}`", part.trim()))
        })
        .collect()
}

fn parse_f64(value: String, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("invalid floating-point value `{value}` for {flag}"))
}

fn print_usage() {
    println!(
        "Usage: cargo run -p feg-case-studies --release --example nc1_matern_convergence -- [options]"
    );
    println!();
    println!("Options:");
    println!("  --levels <n,n,...>       Mesh levels");
    println!("  --out <path>             Output directory");
    println!("  --kappa <value>          Positive Matern kappa");
    println!("  --tau <value>            Positive Matern tau");
}
