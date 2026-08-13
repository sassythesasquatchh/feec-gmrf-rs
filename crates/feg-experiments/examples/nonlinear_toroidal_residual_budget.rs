use feg_case_studies::toroidal_inductor::{
    run_toroidal_residual_budget_experiment, NonlinearToroidalConfig, ToroidalResidualBudgetConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = ToroidalResidualBudgetConfig::default();
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
            "--max-iterations" => {
                config.base.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
            }
            "--strides" => {
                config.residual_strides = parse_usize_list(&next_arg(&mut args, "--strides")?)?;
            }
            "--ordered" => {
                config.shuffled = false;
            }
            "--seed" => {
                config.seed = next_arg(&mut args, "--seed")?.parse()?;
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    config.base.write_outputs = false;
    config.base.include_cell_b_variance = false;

    let report = run_toroidal_residual_budget_experiment(&config)?;
    println!(
        "reference,converged={},iterations={},residual={:.6e}",
        report.reference.converged,
        report.reference.iterations,
        report.reference.final_residual_norm
    );
    println!("prior,selection,rows,total,iterations,residual,cg_iterations,cg_residual_max,factor_nnz,factor_seconds,map_rel_error,cell_b_rel_error,sensor_rmse,factorizes");
    for row in report.rows {
        println!(
            "{},{},{},{},{},{:.6e},{},{:.6e},{},{:.6e},{:.6e},{:.6e},{:.6e},{}",
            row.prior_mode.label(),
            row.selection_label,
            row.residual_rows_used,
            row.residual_rows_total,
            row.iterations,
            row.final_residual_norm,
            row.linear_solve_iteration_sum,
            row.linear_solve_residual_max,
            row.posterior_factor_nnz,
            row.final_factorization_seconds,
            row.map_relative_error_to_reference,
            row.cell_b_relative_error_to_reference,
            row.flux_sensor_rmse_to_reference,
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
    println!("Usage: nonlinear_toroidal_residual_budget [options]");
    println!("  --mesh <path>");
    println!("  --beta-core <float>");
    println!("  --pde-variance <float>");
    println!("  --max-iterations <usize>");
    println!("  --strides <comma-separated usize list>");
    println!("  --ordered");
    println!("  --seed <u64>");
    let defaults = NonlinearToroidalConfig::default();
    println!("default mesh: {}", defaults.mesh_path.display());
}
