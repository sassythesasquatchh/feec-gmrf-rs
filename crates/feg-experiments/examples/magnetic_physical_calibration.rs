use feg_case_studies::magnetic_physical_calibration::{
    compute_magnetic_physical_calibration_report, write_magnetic_physical_calibration_outputs,
    MagneticPhysicalCalibrationConfig, MagneticTruthMode,
};
use feg_infer::prior::matern::MaternAlpha;
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = MagneticPhysicalCalibrationConfig::default();
    let mut out_dir = PathBuf::from("out/magnetic_physical_calibration");

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
            "--practical-range-m" => {
                config.practical_range_m = args
                    .next()
                    .ok_or("--practical-range-m requires a value")?
                    .parse()?;
            }
            "--target-b-rms-tesla" => {
                config.target_b_rms_tesla = args
                    .next()
                    .ok_or("--target-b-rms-tesla requires a value")?
                    .parse()?;
            }
            "--tau-user" => {
                config.tau_user = args.next().ok_or("--tau-user requires a value")?.parse()?;
            }
            "--truth-b-rms-tesla" => {
                config.truth_b_rms_tesla = args
                    .next()
                    .ok_or("--truth-b-rms-tesla requires a value")?
                    .parse()?;
            }
            "--truth-mode" => {
                config.truth_mode = args
                    .next()
                    .ok_or("--truth-mode requires a value")?
                    .parse::<MagneticTruthMode>()?;
            }
            "--observation-std-tesla" => {
                config.observation_std_tesla = args
                    .next()
                    .ok_or("--observation-std-tesla requires a value")?
                    .parse()?;
            }
            "--training-sensor-cells" => {
                config.training_sensor_cells = args
                    .next()
                    .ok_or("--training-sensor-cells requires a value")?
                    .parse()?;
            }
            "--heldout-sensor-cells" => {
                config.heldout_sensor_cells = args
                    .next()
                    .ok_or("--heldout-sensor-cells requires a value")?
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
    let report = compute_magnetic_physical_calibration_report(config)
        .map_err(|err| format!("magnetic physical calibration failed: {err}"))?;
    write_magnetic_physical_calibration_outputs(&report, &out_dir)?;

    println!("Synthetic cube magnetic physical-units calibration");
    println!("  rows: {}", report.prior_rows.len());
    println!("  truth_mode: {}", report.config.truth_mode.label());
    println!("  output: {}", out_dir.display());
    for row in &report.prior_rows {
        println!(
            "  level={} alpha={} tau_normalizer={:.4e} mean_B2={:.6e}",
            row.level,
            row.alpha.as_u32(),
            row.tau_normalizer,
            row.normalized_mean_b2
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
        "Usage: cargo run --release -p feg-case-studies --example magnetic_physical_calibration -- [options]\n\
Options:\n\
  --out-dir <path>                    Output directory (default: out/magnetic_physical_calibration)\n\
  --levels <n,n,...>                  3D cube mesh levels (default: 2,3,4)\n\
  --alphas <a,a,...>                  Matérn alpha values 1,2,3 (default: 1,2,3)\n\
  --practical-range-m <value>         Practical range in metres (default: 0.25)\n\
  --target-b-rms-tesla <value>        Prior RMS B target in Tesla (default: 0.10)\n\
  --tau-user <value>                  User-facing precision/tau multiplier (default: 1)\n\
  --truth-b-rms-tesla <value>         Smooth truth RMS B in Tesla (default: 0.10)\n\
  --truth-mode <smooth|prior-sample>  Truth generator (default: smooth)\n\
  --observation-std-tesla <value>     Sensor standard deviation in Tesla (default: 0.005)\n\
  --training-sensor-cells <n>         Training sensor cells (default: 24)\n\
  --heldout-sensor-cells <n>          Heldout sensor cells (default: 24)\n\
  --exact-max-dofs <n>                Exact trace/variance cutoff by latent dofs (default: 700)\n\
  --hutchinson-probes <n>             Hutchinson probes for large cases (default: 128)\n\
  --hutchinson-batches <n>            Hutchinson batch count (default: 4)\n\
  --rng-seed <n>                      Base deterministic RNG seed"
    );
}
