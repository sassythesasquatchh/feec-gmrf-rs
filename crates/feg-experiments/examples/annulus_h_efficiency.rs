use feg_case_studies::annulus_h_formulation::{
    run_annulus_h_efficiency_sweep, write_annulus_h_efficiency_outputs, AnnulusEfficiencyConfig,
};
use std::{env, error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let config = parse_args(env::args().skip(1))?;
    let result = run_annulus_h_efficiency_sweep(&config)?;
    write_annulus_h_efficiency_outputs(&result, &config)?;

    println!("annular H speed/sparsity efficiency sweep");
    println!("output_dir={}", config.output_dir.display());
    println!("mesh_levels={}", config.mesh_sizes.len());
    for row in &result.rows {
        println!(
            "mesh={} vertices={} model={} status={} selected_total={:.3}s prior_density={:.3e} posterior_density={:.3e} factor_nnz={} line_rmse={:.3e}",
            row.mesh_size,
            row.vertex_count,
            row.model.as_str(),
            row.status,
            row.selected_total_seconds,
            row.prior_precision_density,
            row.posterior_precision_density,
            row.posterior_factor_nnz,
            row.rmse_line
        );
    }
    for row in &result.speedup_rows {
        println!(
            "speedup mesh={} vertices={} gp/feec={:.3}",
            row.mesh_size, row.vertex_count, row.speedup
        );
    }
    println!(
        "wrote efficiency_summary.csv and efficiency_speedup.csv in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<AnnulusEfficiencyConfig, Box<dyn Error>> {
    let mut config = AnnulusEfficiencyConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-sizes" => {
                config.mesh_sizes = parse_mesh_sizes(&next_value(&mut args, "--mesh-sizes")?)?;
            }
            "--output-dir" | "--out" => {
                config.output_dir = PathBuf::from(next_value(&mut args, arg.as_str())?);
            }
            "--sample-noise" => {
                config.sample_observation_noise = true;
                config.base.sample_observation_noise = true;
            }
            "--dense-gp-vertex-limit" => {
                config.dense_gp_vertex_limit = Some(parse_usize(
                    &next_value(&mut args, "--dense-gp-vertex-limit")?,
                    "--dense-gp-vertex-limit",
                )?);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    Ok(config)
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn parse_mesh_sizes(value: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            let parsed = part
                .trim()
                .parse::<f64>()
                .map_err(|err| format!("invalid mesh size `{}`: {err}", part.trim()))?;
            if !parsed.is_finite() || parsed <= 0.0 {
                return Err(format!(
                    "mesh size must be finite and positive, got {parsed}"
                ));
            }
            Ok(parsed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("mesh-sizes must contain at least one value".into());
    }
    Ok(values)
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, Box<dyn Error>> {
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid value `{value}` for {flag}: {err}").into())
}

fn print_usage() {
    println!("Usage: cargo run --release -p feg-case-studies --example annulus_h_efficiency -- [options]");
    println!();
    println!("Options:");
    println!("  --mesh-sizes <h,h,...>       Mesh sizes to sweep");
    println!("  --output-dir <path>          Output directory");
    println!("  --dense-gp-vertex-limit <n>  Skip dense GP above this vertex count");
    println!("  --sample-noise               Sample synthetic training noise");
}
