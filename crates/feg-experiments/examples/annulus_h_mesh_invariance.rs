use feg_case_studies::{
    annulus_baselines::AnnulusModelKind,
    annulus_h_formulation::{
        run_annulus_h_mesh_invariance, write_annulus_h_mesh_invariance_outputs,
        AnnulusMeshInvarianceConfig,
    },
};
use std::{env, error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let config = parse_args(env::args().skip(1))?;
    let result = run_annulus_h_mesh_invariance(&config)?;
    write_annulus_h_mesh_invariance_outputs(&result, &config)?;

    println!("annular H mesh-invariance QoI experiment");
    println!("output_dir={}", config.output_dir.display());
    println!("profile={}", config.profile.as_str());
    println!("mesh_levels={}", config.mesh_sizes.len());
    println!("models={}", format_models(&config.model_kinds));
    for row in &result.fit_rows {
        if !is_primary_qoi(&row.qoi_family) {
            continue;
        }
        println!(
            "model={} regime={} qoi={} final_two={:.3e} tail_spread={:.3e} slope={:.3e} ratio_to_full={:.3e} threshold={:.3e} trend={} status={}",
            row.model.as_str(),
            row.regime.as_str(),
            row.qoi_family,
            row.relative_change,
            row.tail_relative_spread,
            row.slope_log_variance_vs_log_h,
            row.model_ratio_to_full,
            row.threshold,
            row.trend,
            row.status
        );
    }
    println!(
        "wrote mesh_metadata.csv, qoi_variance_by_mesh.csv, mesh_invariance_summary.csv, mesh_invariance_fit.csv, and mesh_invariance_model_contrast.csv in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<AnnulusMeshInvarianceConfig, Box<dyn Error>> {
    let mut config = AnnulusMeshInvarianceConfig::default();
    let mut mesh_sizes_override = None;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-sizes" => {
                mesh_sizes_override =
                    Some(parse_mesh_sizes(&next_value(&mut args, "--mesh-sizes")?)?);
            }
            "--profile" => {
                config.profile = next_value(&mut args, "--profile")?.parse()?;
                config.mesh_sizes = config.profile.default_mesh_sizes();
            }
            "--output-dir" | "--out" => {
                config.output_dir = PathBuf::from(next_value(&mut args, arg.as_str())?);
            }
            "--heldout-loops" => {
                config.heldout_loop_count = parse_usize(
                    &next_value(&mut args, "--heldout-loops")?,
                    "--heldout-loops",
                )?;
            }
            "--residuals" => {
                config.residual_count =
                    parse_usize(&next_value(&mut args, "--residuals")?, "--residuals")?;
            }
            "--sample-noise" => {
                config.sample_observation_noise = true;
            }
            "--models" => {
                config.model_kinds = parse_models(&next_value(&mut args, "--models")?)?;
            }
            "--tail-count" => {
                config.convergence_tail_count =
                    parse_usize(&next_value(&mut args, "--tail-count")?, "--tail-count")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    if let Some(mesh_sizes) = mesh_sizes_override {
        config.mesh_sizes = mesh_sizes;
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

fn parse_models(value: &str) -> Result<Vec<AnnulusModelKind>, Box<dyn Error>> {
    let models = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<AnnulusModelKind>())
        .collect::<Result<Vec<_>, _>>()?;
    if models.is_empty() {
        return Err("--models must contain at least one model".into());
    }
    Ok(models)
}

fn format_models(models: &[AnnulusModelKind]) -> String {
    models
        .iter()
        .map(|model| model.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

fn is_primary_qoi(qoi_family: &str) -> bool {
    matches!(
        qoi_family,
        "q_period_variance"
            | "circulation_mean_variance"
            | "dense_line_away_median_variance"
            | "field_x_away_median_variance"
            | "field_y_away_median_variance"
            | "field_magnitude_away_median_variance"
    )
}

fn print_usage() {
    println!("Usage: cargo run --release -p feg-case-studies --example annulus_h_mesh_invariance -- [options]");
    println!();
    println!("Options:");
    println!("  --profile <kind>         quick or thesis");
    println!("  --mesh-sizes <h,h,...>   Mesh sizes to sweep");
    println!("  --output-dir <path>      Output directory");
    println!("  --heldout-loops <n>      Number of held-out noncontractible loops");
    println!("  --residuals <n>          Number of residual faces used in B/D");
    println!("  --models <a,b>           feec_gmrf, feec_split_no_spectral_correction");
    println!("  --tail-count <n>         Number of finest levels used for tail diagnostics");
    println!("  --sample-noise           Sample synthetic training noise");
}
