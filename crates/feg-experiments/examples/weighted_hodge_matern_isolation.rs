use feg_case_studies::weighted_hodge_matern_isolation::{
    compute_weighted_hodge_matern_isolation_report, write_weighted_hodge_matern_isolation_outputs,
    VarianceNormalization, WeightedHodgeMaternIsolationConfig, WeightedMassInverseKind,
};
use std::{error::Error, io, path::PathBuf};

#[derive(Debug, Clone)]
struct Config {
    diagnostic: WeightedHodgeMaternIsolationConfig,
    output_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            diagnostic: WeightedHodgeMaternIsolationConfig::default(),
            output_dir: PathBuf::from("out/weighted_hodge_matern_isolation"),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let report = compute_weighted_hodge_matern_isolation_report(config.diagnostic.clone())
        .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;
    write_weighted_hodge_matern_isolation_outputs(&report, &config.output_dir)?;
    print_summary(&report);
    println!("wrote outputs to {}", config.output_dir.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut config = Config::default();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--level" => {
                i += 1;
                config.diagnostic.level = parse_next(&args, i, "--level")?;
            }
            "--contrasts" => {
                i += 1;
                config.diagnostic.contrasts = parse_csv_f64(next_value(&args, i, "--contrasts")?)?;
            }
            "--kappa-factors" => {
                i += 1;
                config.diagnostic.kappa_factors =
                    parse_csv_f64(next_value(&args, i, "--kappa-factors")?)?;
            }
            "--max-exact-dofs" => {
                i += 1;
                config.diagnostic.max_exact_dofs = parse_next(&args, i, "--max-exact-dofs")?;
            }
            "--output-dir" => {
                i += 1;
                config.output_dir = PathBuf::from(next_value(&args, i, "--output-dir")?);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            flag => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown flag {flag}"),
                )
                .into());
            }
        }
        i += 1;
    }
    Ok(config)
}

fn parse_next<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    flag: &str,
) -> Result<T, Box<dyn Error>>
where
    T::Err: std::fmt::Display,
{
    let value = next_value(args, index, flag)?;
    value.parse::<T>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for {flag}: {err}"),
        )
        .into()
    })
}

fn next_value(args: &[String], index: usize, flag: &str) -> Result<String, Box<dyn Error>> {
    args.get(index).cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value after {flag}"),
        )
        .into()
    })
}

fn parse_csv_f64(value: String) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = value
        .split(',')
        .map(|part| {
            part.trim().parse::<f64>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid floating-point value `{part}`: {err}"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(values)
}

fn print_summary(
    report: &feg_case_studies::weighted_hodge_matern_isolation::WeightedHodgeMaternIsolationReport,
) {
    println!("Weighted Hodge Matern split-square isolation");
    println!(
        "  level={} vertices={} edges={} active_edges={} triangles={}",
        report.config.level,
        report.vertex_count,
        report.edge_count,
        report.active_edge_count,
        report.triangle_count
    );
    println!(
        "  unit-weight diffs: operator={:.3e} mass={:.3e} projected_inverse={:.3e}",
        report.unit_weight_diagnostics.operator_max_abs_difference,
        report.unit_weight_diagnostics.mass_max_abs_difference,
        report
            .unit_weight_diagnostics
            .projected_inverse_max_abs_difference
    );
    for scenario in report.scenarios.iter().take(3) {
        let exact = scenario.strategies.iter().find(|strategy| {
            strategy.kind == WeightedMassInverseKind::ExactConsistentWeightedMass
                && strategy.normalization == VarianceNormalization::TraceMatchedBaseline
        });
        if let Some(strategy) = exact {
            println!(
                "  contrast={:.3e} kappa_factor={:.3e}: exact trace-matched prior_mean={:.3e} posterior_mean={:.3e} probe_ratio={:.3e}",
                scenario.contrast,
                scenario.kappa_factor,
                strategy.prior_variance_stats.mean,
                strategy.posterior_variance_stats.mean,
                strategy.mean_probe_variance_ratio
            );
        }
    }
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example weighted_hodge_matern_isolation -- [options]"
    );
    println!("  --level <n>");
    println!("  --contrasts <a,b,c>");
    println!("  --kappa-factors <a,b,c>");
    println!("  --max-exact-dofs <n>");
    println!("  --output-dir <path>");
}
