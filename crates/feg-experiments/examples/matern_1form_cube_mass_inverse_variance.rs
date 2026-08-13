use feg_case_studies::cube_mass_inverse_variance::{
    compute_matern_1form_cube_mass_inverse_variance_report,
    write_matern_1form_cube_mass_inverse_variance_outputs, CubeMassInverseVarianceConfig,
};
use std::{io, path::PathBuf};

#[derive(Debug, Clone)]
struct CliConfig {
    experiment: CubeMassInverseVarianceConfig,
    out_dir: PathBuf,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            experiment: CubeMassInverseVarianceConfig::default(),
            out_dir: PathBuf::from("out/matern_1form_cube_mass_inverse_variance"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let report = compute_matern_1form_cube_mass_inverse_variance_report(config.experiment)
        .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;
    write_matern_1form_cube_mass_inverse_variance_outputs(&report, &config.out_dir)?;
    print_summary(&report, &config.out_dir);
    Ok(())
}

fn parse_args() -> Result<CliConfig, Box<dyn std::error::Error>> {
    let mut config = CliConfig::default();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--levels" => {
                index += 1;
                config.experiment.levels =
                    parse_levels(parse_next::<String>(&args, index, "--levels")?)?;
            }
            "--kappa" => {
                index += 1;
                config.experiment.kappa = parse_next(&args, index, "--kappa")?;
            }
            "--tau" => {
                index += 1;
                config.experiment.tau = parse_next(&args, index, "--tau")?;
            }
            "--out-dir" => {
                index += 1;
                config.out_dir = PathBuf::from(parse_next::<String>(&args, index, "--out-dir")?);
            }
            "--drop-tolerance" => {
                index += 1;
                config.experiment.drop_tolerance = parse_next(&args, index, "--drop-tolerance")?;
            }
            "--max-consistent-dofs" => {
                index += 1;
                config.experiment.max_consistent_dofs =
                    parse_next(&args, index, "--max-consistent-dofs")?;
            }
            "--barycentric-stabilization-factors" => {
                index += 1;
                config.experiment.barycentric_stabilization_factors = parse_f64_list(
                    parse_next::<String>(&args, index, "--barycentric-stabilization-factors")?,
                    "--barycentric-stabilization-factors",
                )?;
            }
            "--no-barycentric-oracle" => {
                config.experiment.include_barycentric_oracle = false;
            }
            "--help" | "-h" => {
                print_help();
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
        index += 1;
    }
    Ok(config)
}

fn parse_next<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    let value = args.get(index).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value after {flag}"),
        )
    })?;
    value.parse::<T>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for {flag}: {err}"),
        )
        .into()
    })
}

fn parse_levels(value: String) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut levels = Vec::new();
    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        levels.push(token.parse::<usize>().map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid mesh level `{token}`: {err}"),
            )
        })?);
    }
    if levels.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--levels must contain at least one integer",
        )
        .into());
    }
    Ok(levels)
}

fn parse_f64_list(value: String, flag: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        values.push(token.parse::<f64>().map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid value `{token}` for {flag}: {err}"),
            )
        })?);
    }
    if values.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{flag} must contain at least one number"),
        )
        .into());
    }
    Ok(values)
}

fn print_summary(
    report: &feg_case_studies::cube_mass_inverse_variance::CubeMassInverseVarianceReport,
    out_dir: &PathBuf,
) {
    println!(
        "1-form cube Matern mass-inverse variance experiment: kappa={}, tau={}",
        report.config.kappa, report.config.tau
    );
    for level in &report.levels {
        println!(
            "level {}: vertices={}, edges={}, faces={}, tetrahedra={}",
            level.level,
            level.vertex_count,
            level.edge_count,
            level.face_count,
            level.tetrahedron_count
        );
        for strategy in &level.strategies {
            println!(
                "  {:52} interior_mean_var={:.6e} interior_rms_log_delta={:.6e} e_c={:.6e} fill={:.3}x lambda=[{:.3e},{:.3e}] total={:.3}s",
                strategy.label.as_str(),
                strategy.variance_stats.mean,
                strategy
                    .comparison_to_consistent
                    .rms_log_delta_vs_consistent,
                strategy.consistency_error,
                strategy.fill_in_ratio,
                strategy.mass_inverse_eigen.lambda_min,
                strategy.mass_inverse_eigen.lambda_max,
                strategy.timings.total_seconds,
            );
        }
    }
    println!("wrote outputs to {}", out_dir.display());
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example matern_1form_cube_mass_inverse_variance -- [options]\n\
\n\
Options:\n\
  --levels <n,n,...>            Cube cells-per-axis levels (default: 2,4,6,8)\n\
  --kappa <value>               Matern kappa (default: 4)\n\
  --tau <value>                 Matern tau (default: 1)\n\
  --out-dir <path>              Output directory (default: out/matern_1form_cube_mass_inverse_variance)\n\
  --drop-tolerance <value>      Dense inverse CSR drop tolerance (default: 1e-14)\n\
  --max-consistent-dofs <n>     Refuse dense-consistent runs above this edge count (default: 5000)\n\
  --barycentric-stabilization-factors <a,b,...>\n\
                                 Barycentric dual stabilization sweep (default: 0.25,0.5,1,2,4)\n\
  --no-barycentric-oracle       Do not add the best sweep row as an oracle-calibrated diagnostic\n\
  --help                        Show this help"
    );
}
