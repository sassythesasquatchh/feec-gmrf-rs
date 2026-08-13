use feg_case_studies::magnetic_prior_uq_comparison::{
    compute_magnetic_prior_uq_comparison_report, write_magnetic_prior_uq_comparison_outputs,
    MagneticPriorUqComparisonConfig,
};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = MagneticPriorUqComparisonConfig::default();
    let mut out_dir = PathBuf::from("out/magnetic_prior_uq_comparison");

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                out_dir = PathBuf::from(next_value(&mut args, "--out-dir")?);
            }
            "--level" => {
                config.level = next_value(&mut args, "--level")?.parse()?;
            }
            "--range" | "--practical-range-m" => {
                config.practical_range_m = next_value(&mut args, "--range")?.parse()?;
            }
            "--target-b-rms-tesla" => {
                config.target_b_rms_tesla =
                    next_value(&mut args, "--target-b-rms-tesla")?.parse()?;
            }
            "--truth-b-rms-tesla" => {
                config.truth_b_rms_tesla = next_value(&mut args, "--truth-b-rms-tesla")?.parse()?;
            }
            "--observation-std-tesla" => {
                config.observation_std_tesla =
                    next_value(&mut args, "--observation-std-tesla")?.parse()?;
            }
            "--training-sensor-cells" => {
                config.training_sensor_cells =
                    next_value(&mut args, "--training-sensor-cells")?.parse()?;
            }
            "--validation-cells-per-bin" => {
                config.validation_cells_per_bin =
                    next_value(&mut args, "--validation-cells-per-bin")?.parse()?;
            }
            "--test-cells-per-bin" => {
                config.test_cells_per_bin =
                    next_value(&mut args, "--test-cells-per-bin")?.parse()?;
            }
            "--smooth-noise-replicates" => {
                config.smooth_noise_replicates =
                    next_value(&mut args, "--smooth-noise-replicates")?.parse()?;
            }
            "--corrected-sample-truth-replicates" => {
                config.corrected_sample_truth_replicates =
                    next_value(&mut args, "--corrected-sample-truth-replicates")?.parse()?;
            }
            "--exact-max-dofs" => {
                config.exact_max_dofs = next_value(&mut args, "--exact-max-dofs")?.parse()?;
            }
            "--hutchinson-probes" => {
                config.hutchinson_probes = next_value(&mut args, "--hutchinson-probes")?.parse()?;
            }
            "--hutchinson-batches" => {
                config.hutchinson_batches =
                    next_value(&mut args, "--hutchinson-batches")?.parse()?;
            }
            "--rng-seed" => {
                config.rng_seed = next_value(&mut args, "--rng-seed")?.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let start = Instant::now();
    let report = compute_magnetic_prior_uq_comparison_report(config)
        .map_err(|err| format!("magnetic prior UQ comparison failed: {err}"))?;
    write_magnetic_prior_uq_comparison_outputs(&report, &out_dir)?;

    println!("3D magnetic prior UQ comparison");
    println!("  level: {}", report.config.level);
    println!("  alpha: {}", report.fixed_hyperparameters.alpha.as_u32());
    println!(
        "  fixed range={:.4} m kappa={:.6e}",
        report.fixed_hyperparameters.practical_range_m, report.fixed_hyperparameters.kappa
    );
    println!("  output: {}", out_dir.display());
    for row in &report.aggregate_bin_metric_rows {
        println!(
            "  scenario={} model={} split={} bin={} nlpd={:.6e} z_rms={:.4} coverage={:.3}",
            row.scenario,
            row.model,
            row.split,
            row.bin,
            row.nlpd_mean,
            row.z_rms_mean,
            row.coverage_95_mean
        );
    }
    println!("  elapsed: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example magnetic_prior_uq_comparison -- [options]\n\
Options:\n\
  --out-dir <path>                  Output directory (default: out/magnetic_prior_uq_comparison)\n\
  --level <n>                       3D cube mesh level (default: 5)\n\
  --range <r>                       Fixed practical range in metres (default: 0.75)\n\
  --target-b-rms-tesla <value>      Prior RMS B target in Tesla (default: 0.10)\n\
  --truth-b-rms-tesla <value>       Smooth truth RMS B in Tesla (default: 0.10)\n\
  --observation-std-tesla <value>   Sensor standard deviation in Tesla (default: 0.005)\n\
  --training-sensor-cells <n>       Central training sensor cells (default: 24)\n\
  --validation-cells-per-bin <n>    Validation cells in each near/mid/far bin (default: 24)\n\
  --test-cells-per-bin <n>          Test cells in each near/mid/far bin (default: 24)\n\
  --smooth-noise-replicates <n>     Smooth-truth observation-noise replicates (default: 20)\n\
  --corrected-sample-truth-replicates <n> Corrected-prior truth draws (default: 30)\n\
  --exact-max-dofs <n>              Exact variance cutoff by latent dofs (default: 1500)\n\
  --hutchinson-probes <n>           Hutchinson probes for large cases (default: 128)\n\
  --hutchinson-batches <n>          Hutchinson batch count (default: 4)\n\
  --rng-seed <n>                    Base deterministic RNG seed"
    );
}
