use feg_case_studies::nonlinear_eddy_current::{
    run_nonlinear_eddy_current_weiland_comparison_experiment, NonlinearEddyCurrentComparisonConfig,
};

fn main() -> Result<(), String> {
    let config = parse_args()?;
    let report = run_nonlinear_eddy_current_weiland_comparison_experiment(config)?;

    println!("nonlinear screened 1-form eddy-current Weiland-style comparison");
    println!("mesh_level={}", report.config.mesh_level);
    println!("nu0={:.6e}", report.config.nu0);
    println!("beta={:.6e}", report.config.beta);
    println!("sigma={:.6e}", report.config.sigma);
    println!("source_amplitude={:.6e}", report.config.source_amplitude);
    println!("active_dofs={}", report.reference.active_dofs);
    println!("cells={}", report.reference.cells);
    println!("boundary_edges={}", report.reference.boundary_edges);
    println!("source_norm={:.16e}", report.reference.source_norm);
    println!(
        "linear_mean_norm={:.16e}",
        report.reference.linear_mean_norm
    );
    println!(
        "calibrated_probe_variance={:.16e}",
        report.reference.calibrated_probe_variance
    );
    println!(
        "ngsolve_reference=order:{} maxh:{:.6e} converged:{} iterations:{} samples:{} cell_b_norm:{:.16e}",
        report.reference.ngsolve_order,
        report.reference.ngsolve_maxh,
        report.reference.ngsolve_converged,
        report.reference.ngsolve_iterations,
        report.reference.ngsolve_sample_count,
        report.reference.ngsolve_cell_b_norm
    );
    println!(
        "weak_reference_residual_norm={:.16e}",
        report.reference.weak_reference_residual_norm
    );
    println!(
        "weak_reference_iterations={}",
        report.reference.weak_reference_iterations
    );
    println!(
        "weak_reference_converged={}",
        report.reference.weak_reference_converged
    );
    println!(
        "weak_reference_cell_b_relative_error={:.16e}",
        report.reference.weak_reference_cell_b_relative_error
    );
    if let Some(error) = report.reference.reference_adequacy_cell_b_relative_error {
        println!("reference_adequacy_cell_b_relative_error={:.16e}", error);
    }
    println!(
        "prior_mode,observation_mode,probe_count,total_cells,residual_rows,success,weak_residual_norm,ngsolve_cell_a_rel_error,ngsolve_cell_b_rel_error,ngsolve_sensor_b_rmse,ngsolve_sensor_b_rel_rmse,probe_weighted_norm,iterations,damping_count,posterior_factorizes,factor_nnz,b_var_min,b_var_max,sensor_std_error_max,sensor_coverage_2sigma,failure"
    );
    for row in &report.rows {
        println!(
            "{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.prior_mode.as_str(),
            row.observation_mode.as_str(),
            row.probe_count,
            row.total_cells,
            row.residual_rows,
            row.success,
            row.final_weak_residual_norm,
            row.ngsolve_cell_a_relative_error,
            row.ngsolve_cell_b_relative_error,
            row.ngsolve_sensor_b_rmse,
            row.ngsolve_sensor_b_relative_rmse,
            row.residual_probe_weighted_norm,
            row.iterations,
            row.damping_count,
            row.posterior_factorizes,
            row.final_factor_nnz,
            row.selected_b_variance_min,
            row.selected_b_variance_max,
            row.sensor_standardized_error_max,
            row.sensor_coverage_2sigma,
            row.failure.as_deref().unwrap_or("")
        );
    }

    Ok(())
}

fn parse_args() -> Result<NonlinearEddyCurrentComparisonConfig, String> {
    let mut config = NonlinearEddyCurrentComparisonConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--level" => {
                config.mesh_level = parse_next(&mut args, "--level")?;
            }
            "--beta" => {
                config.beta = parse_next(&mut args, "--beta")?;
            }
            "--sigma" => {
                config.sigma = parse_next(&mut args, "--sigma")?;
            }
            "--tau" => {
                config.linear_proxy_tau = parse_next(&mut args, "--tau")?;
            }
            "--residual-variance" => {
                config.residual_variance = parse_next(&mut args, "--residual-variance")?;
            }
            "--probe-variance" => {
                config.probe_variance = Some(parse_next(&mut args, "--probe-variance")?);
            }
            "--probe-noise-relative-scale" => {
                config.probe_noise_relative_scale =
                    parse_next(&mut args, "--probe-noise-relative-scale")?;
            }
            "--max-iterations" => {
                config.max_iterations = parse_next(&mut args, "--max-iterations")?;
            }
            "--source-amplitude" => {
                config.source_amplitude = parse_next(&mut args, "--source-amplitude")?;
            }
            "--ngsolve-python" => {
                config.ngsolve_python = parse_next_path(&mut args, "--ngsolve-python")?;
            }
            "--ngsolve-src" => {
                config.ngsolve_src = parse_next_path(&mut args, "--ngsolve-src")?;
            }
            "--ngsolve-output-dir" => {
                config.ngsolve_output_dir = parse_next_path(&mut args, "--ngsolve-output-dir")?;
            }
            "--ngsolve-order" => {
                config.ngsolve_order = parse_next(&mut args, "--ngsolve-order")?;
            }
            "--ngsolve-maxh" => {
                config.ngsolve_maxh = parse_next(&mut args, "--ngsolve-maxh")?;
            }
            "--ngsolve-write-vtu" => {
                config.ngsolve_write_vtu = true;
            }
            "--check-reference-adequacy" => {
                config.check_reference_adequacy = true;
            }
            "--quick" => {
                config.budget_fractions = vec![0.0, 0.25, 1.0];
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument `{other}`; pass --help for usage"));
            }
        }
    }
    Ok(config)
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let value = args
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value `{value}` for {flag}"))
}

fn parse_next_path(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<std::path::PathBuf, String> {
    Ok(std::path::PathBuf::from(
        args.next()
            .ok_or_else(|| format!("{flag} requires a value"))?,
    ))
}

fn print_help() {
    println!("Usage: nonlinear_eddy_current_weiland_comparison [options]");
    println!("  --level N                 Cartesian cube mesh level, default 2");
    println!("  --beta X                 nonlinear reluctivity strength, default 0.75");
    println!("  --sigma X                screening/conductivity coefficient, default 0.5");
    println!("  --source-amplitude X     analytic current source amplitude, default 1.0");
    println!("  --tau X                  alpha-2 linear-proxy prior tau, default 1.0");
    println!("  --residual-variance X    weak residual variance, default 1e-8");
    println!("  --probe-variance X       fixed local probe variance, default calibrated");
    println!("  --probe-noise-relative-scale X");
    println!("  --max-iterations N       Gauss-Newton iteration cap, default 25");
    println!("  --ngsolve-python PATH");
    println!("  --ngsolve-src PATH");
    println!("  --ngsolve-output-dir PATH");
    println!("  --ngsolve-order N");
    println!("  --ngsolve-maxh X");
    println!("  --ngsolve-write-vtu");
    println!("  --check-reference-adequacy");
    println!("  --quick                  run only budgets 0, 1/4, and 1");
}
