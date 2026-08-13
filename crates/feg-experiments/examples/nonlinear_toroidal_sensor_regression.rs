use feg_case_studies::toroidal_inductor::{
    run_toroidal_sensor_regression_experiment, NonlinearToroidalConfig,
    ToroidalSensorRegressionConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = ToroidalSensorRegressionConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--mesh" => {
                config.base.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?);
            }
            "--beta-core" => {
                config.base.beta_core = next_arg(&mut args, "--beta-core")?.parse()?;
            }
            "--pde-variance" => {
                config.base.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?;
            }
            "--sensor-variance" => {
                config.sensor_variance = next_arg(&mut args, "--sensor-variance")?.parse()?;
            }
            "--noise-std" => {
                config.synthetic_noise_std = next_arg(&mut args, "--noise-std")?.parse()?;
            }
            "--max-iterations" => {
                config.base.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
            }
            "--azimuth-count" => {
                config.azimuth_count = next_arg(&mut args, "--azimuth-count")?.parse()?;
            }
            "--train-counts" => {
                config.train_counts = parse_usize_list(&next_arg(&mut args, "--train-counts")?)?;
            }
            "--residual-stride" => {
                config.residual_stride = next_arg(&mut args, "--residual-stride")?.parse()?;
            }
            "--seed" => {
                config.seed = next_arg(&mut args, "--seed")?.parse()?;
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    config.base.write_outputs = false;
    config.base.include_cell_b_variance = false;

    let report = run_toroidal_sensor_regression_experiment(&config)?;
    println!(
        "reference,converged={},iterations={},residual={:.6e},sensors={}",
        report.reference.converged,
        report.reference.iterations,
        report.reference.final_residual_norm,
        report.sensors.len()
    );
    println!("prior,variant,train,holdout,train_rmse,holdout_rmse,mean_abs_z,coverage_2sigma,residual,cg_iterations,cg_residual_max,factor_nnz,factor_seconds,factorizes");
    for row in report.rows {
        println!(
            "{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{:.6e},{},{:.6e},{}",
            row.prior_mode.label(),
            row.variant.label(),
            row.train_count,
            row.holdout_count,
            row.train_rmse,
            row.holdout_rmse,
            row.mean_abs_z,
            row.coverage_2sigma,
            row.final_residual_norm,
            row.linear_solve_iteration_sum,
            row.linear_solve_residual_max,
            row.posterior_factor_nnz,
            row.final_factorization_seconds,
            row.posterior_factorizes
        );
    }
    Ok(())
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let parsed = value
        .split(',')
        .map(|item| item.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err("list must contain at least one value".into());
    }
    Ok(parsed)
}

fn print_help() {
    println!("Usage: nonlinear_toroidal_sensor_regression [options]");
    println!("  --mesh <path>");
    println!("  --beta-core <float>");
    println!("  --pde-variance <float>");
    println!("  --sensor-variance <float>");
    println!("  --noise-std <float>");
    println!("  --max-iterations <usize>");
    println!("  --azimuth-count <usize>");
    println!("  --train-counts <comma-separated usize list>");
    println!("  --residual-stride <usize>");
    println!("  --seed <u64>");
    let defaults = NonlinearToroidalConfig::default();
    println!("default mesh: {}", defaults.mesh_path.display());
}
