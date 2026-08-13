use feg_case_studies::toroidal_inductor::{
    run_toroidal_weiland_comparison_experiment_with_row_callback, NonlinearToroidalConfig,
    ToroidalPdeObservationMode, ToroidalWeilandComparisonConfig,
};
use std::{env, error::Error, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = ToroidalWeilandComparisonConfig::default();
    let mut csv_path: Option<PathBuf> = None;
    let mut stream_rows = true;
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
            "--prior-tau" => {
                config.base.linear_proxy_tau = next_arg(&mut args, "--prior-tau")?.parse()?;
            }
            "--prior-kappa" => {
                config.explicit_kappas = vec![next_arg(&mut args, "--prior-kappa")?.parse()?];
                config.include_default_kappa = false;
            }
            "--observation-mode" => {
                config.base.pde_observation_mode =
                    parse_observation_mode(&next_arg(&mut args, "--observation-mode")?)?;
            }
            "--max-iterations" => {
                config.base.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
            }
            "--fractions" => {
                config.residual_fractions = parse_f64_list(&next_arg(&mut args, "--fractions")?)?;
            }
            "--kappa-scales" => {
                config.kappa_diameter_scales =
                    parse_f64_list(&next_arg(&mut args, "--kappa-scales")?)?;
            }
            "--no-default-kappa" => {
                config.include_default_kappa = false;
            }
            "--repetitions" => {
                config.row_selection_repetitions = next_arg(&mut args, "--repetitions")?.parse()?;
            }
            "--seed" => {
                config.seed = next_arg(&mut args, "--seed")?.parse()?;
            }
            "--sensor-azimuth-count" => {
                config.sensor_azimuth_count =
                    next_arg(&mut args, "--sensor-azimuth-count")?.parse()?;
            }
            "--csv" => {
                csv_path = Some(PathBuf::from(next_arg(&mut args, "--csv")?));
            }
            "--no-stream" => {
                stream_rows = false;
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }

    config.base.write_outputs = false;
    config.base.include_cell_b_variance = false;
    config.base.compute_harmonic_diagnostics = false;
    config.base.linear_proxy_allow_kappa_fallback = false;

    let report = run_toroidal_weiland_comparison_experiment_with_row_callback(&config, |row| {
        if stream_rows {
            eprintln!(
                    "row reference_kappa={:.6e} prior={} rows={}/{} success={} residual={:.6e} cell_b_err={:.6e} sensor_rmse={:.6e}",
                    row.reference_kappa,
                    row.prior_mode.label(),
                    row.residual_rows_used,
                    row.residual_rows_total,
                    row.success,
                    row.final_residual_norm,
                    row.cell_b_relative_error_to_reference,
                    row.flux_sensor_rmse_to_reference
                );
        }
    })?;
    if let Some(path) = csv_path.as_ref() {
        write_rows_csv(path, &report.rows)?;
    }
    println!(
        "mesh={},active_dofs={},residual_rows={},observation_mode={},sensors={},diameter={:.6e}",
        report.mesh_path.display(),
        report.active_dofs,
        report.residual_rows_total,
        config.base.pde_observation_mode.label(),
        report.sensor_count,
        report.bounding_box_diameter
    );
    println!("references");
    println!("requested_kappa,actual_kappa,kappa_times_diameter,success,iterations,residual,factor_nnz,factor_seconds,error");
    for reference in &report.references {
        println!(
            "{:.12e},{:.12e},{:.12e},{},{},{:.6e},{},{:.6e},{}",
            reference.requested_kappa,
            reference.actual_kappa,
            reference.kappa_times_diameter,
            reference.success,
            reference.iterations,
            reference.final_residual_norm,
            reference.posterior_factor_nnz,
            reference.final_factorization_seconds,
            csv_text(reference.failure_reason.as_deref())
        );
    }

    println!("summaries");
    println!("reference_kappa,prior,uses_residual,rows,total,fraction,successes,failures,residual_mean,residual_std,cell_b_err_mean,cell_b_err_std,sensor_rmse_mean,sensor_rmse_std,map_err_mean,map_err_std,mean_abs_z,coverage_2sigma");
    for summary in &report.summaries {
        println!(
            "{:.12e},{},{},{},{},{:.6e},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            summary.reference_kappa,
            summary.prior_mode.label(),
            summary.nonlinear_residual_likelihood,
            summary.residual_rows_requested,
            summary.residual_rows_total,
            summary.residual_fraction,
            summary.success_count,
            summary.failure_count,
            summary.final_residual_mean,
            summary.final_residual_std,
            summary.cell_b_relative_error_mean,
            summary.cell_b_relative_error_std,
            summary.flux_sensor_rmse_mean,
            summary.flux_sensor_rmse_std,
            summary.map_relative_error_mean,
            summary.map_relative_error_std,
            summary.sensor_mean_abs_z_mean,
            summary.sensor_coverage_2sigma_mean
        );
    }

    let failures = report
        .rows
        .iter()
        .filter(|row| !row.success)
        .collect::<Vec<_>>();
    if !failures.is_empty() {
        println!("failures");
        println!("reference_kappa,prior,uses_residual,rows,total,seed,error");
        for row in failures {
            println!(
                "{:.12e},{},{},{},{},{},{}",
                row.reference_kappa,
                row.prior_mode.label(),
                row.nonlinear_residual_likelihood,
                row.residual_rows_requested,
                row.residual_rows_total,
                row.seed,
                csv_text(row.failure_reason.as_deref())
            );
        }
    }
    Ok(())
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_f64_list(value: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let parsed = value
        .split(',')
        .map(|item| item.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err("list must contain at least one value".into());
    }
    Ok(parsed)
}

fn parse_observation_mode(value: &str) -> Result<ToroidalPdeObservationMode, Box<dyn Error>> {
    match value {
        "weak-galerkin-rows" | "weak_galerkin_rows" => {
            Ok(ToroidalPdeObservationMode::WeakGalerkinRows)
        }
        "local-strong-cells" | "local_strong_cells" => {
            Ok(ToroidalPdeObservationMode::LocalMagneticStrongCells)
        }
        other => Err(format!(
            "unknown observation mode `{other}`; expected weak-galerkin-rows or local-strong-cells"
        )
        .into()),
    }
}

fn csv_text(value: Option<&str>) -> String {
    value.unwrap_or("").replace([',', '\n'], ";")
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_rows_csv(
    path: &PathBuf,
    rows: &[feg_case_studies::toroidal_inductor::ToroidalWeilandComparisonRow],
) -> Result<(), Box<dyn Error>> {
    let mut out = String::from("reference_kappa,prior,uses_residual,selection,seed,rows_used,rows_total,success,final_residual,cell_b_error,sensor_rmse,map_error,error\n");
    for row in rows {
        out.push_str(&format!(
            "{:.12e},{},{},{},{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{}\n",
            row.reference_kappa,
            row.prior_mode.label(),
            row.nonlinear_residual_likelihood,
            csv_field(&row.selection_label),
            row.seed,
            row.residual_rows_used,
            row.residual_rows_total,
            row.success,
            row.final_residual_norm,
            row.cell_b_relative_error_to_reference,
            row.flux_sensor_rmse_to_reference,
            row.map_relative_error_to_reference,
            csv_field(row.failure_reason.as_deref().unwrap_or(""))
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn print_help() {
    println!("Usage: nonlinear_toroidal_weiland_comparison [options]");
    println!("  --mesh <path>");
    println!("  --beta-core <float>");
    println!("  --pde-variance <float>");
    println!("  --prior-tau <float>");
    println!("  --prior-kappa <float>");
    println!("  --observation-mode <weak-galerkin-rows|local-strong-cells>");
    println!("  --max-iterations <usize>");
    println!("  --fractions <comma-separated floats>");
    println!("  --kappa-scales <comma-separated floats>");
    println!("  --no-default-kappa");
    println!("  --repetitions <usize>");
    println!("  --seed <u64>");
    println!("  --sensor-azimuth-count <usize>");
    println!("  --csv <path>");
    println!("  --no-stream");
    let defaults = NonlinearToroidalConfig::default();
    println!("default mesh: {}", defaults.mesh_path.display());
}
