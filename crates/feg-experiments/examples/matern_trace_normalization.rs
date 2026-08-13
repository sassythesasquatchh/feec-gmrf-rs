use feg_case_studies::matern_trace_normalization::{
    compute_matern_trace_normalization_report, write_matern_trace_normalization_outputs,
    MaternTraceNormalizationConfig,
};
use feg_infer::prior::matern::MaternAlpha;
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = MaternTraceNormalizationConfig::default();
    let mut out_dir = PathBuf::from("out/matern_trace_normalization");

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                out_dir = PathBuf::from(args.next().ok_or("--out-dir requires a value")?);
            }
            "--levels" => {
                config.levels = parse_usize_csv(&args.next().ok_or("--levels requires a value")?)?;
            }
            "--alphas" => {
                config.alphas = parse_alpha_csv(&args.next().ok_or("--alphas requires a value")?)?;
            }
            "--kappa" => {
                config.kappa = args.next().ok_or("--kappa requires a value")?.parse()?;
            }
            "--target-mean-trace-variance" => {
                config.target_mean_trace_variance = args
                    .next()
                    .ok_or("--target-mean-trace-variance requires a value")?
                    .parse()?;
            }
            "--exact-max-dofs" => {
                config.exact_max_dofs = args
                    .next()
                    .ok_or("--exact-max-dofs requires a value")?
                    .parse()?;
            }
            "--hutchinson-probes" => {
                config.hutchinson_probes = args
                    .next()
                    .ok_or("--hutchinson-probes requires a value")?
                    .parse()?;
            }
            "--hutchinson-batches" => {
                config.hutchinson_batches = args
                    .next()
                    .ok_or("--hutchinson-batches requires a value")?
                    .parse()?;
            }
            "--rng-seed" => {
                config.rng_seed = args.next().ok_or("--rng-seed requires a value")?.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let start = Instant::now();
    let report = compute_matern_trace_normalization_report(config)
        .map_err(|err| format!("Matérn trace-normalization experiment failed: {err}"))?;
    write_matern_trace_normalization_outputs(&report, &out_dir)?;

    println!("Matérn trace-normalization experiment");
    println!("  rows: {}", report.rows.len());
    println!("  output: {}", out_dir.join("summary.csv").display());
    for row in &report.rows {
        println!(
            "  stage={} level={} alpha={} tau={:.4e} normalized_mean={:.6e}",
            row.stage.label(),
            row.level,
            row.alpha.as_u32(),
            row.tau_multiplier,
            row.normalized_exact_mean_trace_variance
                .unwrap_or(row.normalized_hutchinson_mean_trace_variance)
        );
    }
    println!("  elapsed: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn parse_usize_csv(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| Ok(part.trim().parse::<usize>()?))
        .collect()
}

fn parse_alpha_csv(value: &str) -> Result<Vec<MaternAlpha>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| part.trim().parse::<MaternAlpha>().map_err(|err| err.into()))
        .collect()
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example matern_trace_normalization -- [options]\n\
Options:\n\
  --out-dir <path>                         Output directory (default: out/matern_trace_normalization)\n\
  --levels <n,n,...>                       2D square mesh levels (default: 8,16,32,64)\n\
  --alphas <a,a,...>                       Matérn alpha values 1,2,3 (default: 1,2,3)\n\
  --kappa <value>                          Fixed kappa (default: sqrt(8)/0.2)\n\
  --target-mean-trace-variance <value>     Target weighted mean variance (default: 1)\n\
  --exact-max-dofs <n>                     Exact trace cutoff by latent dofs (default: 400)\n\
  --hutchinson-probes <n>                  Hutchinson probes per trace estimate (default: 128)\n\
  --hutchinson-batches <n>                 Hutchinson batch count (default: 8)\n\
  --rng-seed <n>                           Base deterministic RNG seed"
    );
}
