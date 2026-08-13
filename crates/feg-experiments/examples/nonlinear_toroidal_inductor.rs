use feg_case_studies::toroidal_inductor::{
    diagnose_toroidal_observation, run_nonlinear_toroidal_inductor, NonlinearToroidalConfig,
    NonlinearToroidalReport, ToroidalPdeObservationMode, ToroidalPriorMode,
};
use feg_infer::nonlinear::{GaussNewtonLinearSolve, GaussNewtonLinearSolveMode};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = NonlinearToroidalConfig {
        beta_core: 1e16,
        include_cell_b_variance: false,
        ..NonlinearToroidalConfig::default()
    };
    let mut run_linear_baseline = true;
    let mut diagnose_only = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--mesh" => {
                config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?);
            }
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?));
            }
            "--beta-core" => {
                config.beta_core = next_arg(&mut args, "--beta-core")?.parse()?;
            }
            "--nu-core0" => {
                config.nu_core0 = next_arg(&mut args, "--nu-core0")?.parse()?;
            }
            "--pde-variance" => {
                config.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?;
            }
            "--prior-precision" => {
                config.prior_precision = next_arg(&mut args, "--prior-precision")?.parse()?;
            }
            "--prior-kappa" => {
                config.linear_proxy_kappa = Some(next_arg(&mut args, "--prior-kappa")?.parse()?);
            }
            "--prior-tau" => {
                config.linear_proxy_tau = next_arg(&mut args, "--prior-tau")?.parse()?;
            }
            "--observation-mode" => {
                config.pde_observation_mode =
                    parse_observation_mode(&next_arg(&mut args, "--observation-mode")?)?;
            }
            "--max-iterations" => {
                config.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
            }
            "--direct-cholesky-steps" => {
                config.linear_solve = GaussNewtonLinearSolve::DirectCholesky;
            }
            "--cg-tolerance" => {
                let tolerance = next_arg(&mut args, "--cg-tolerance")?.parse()?;
                config.linear_solve = match config.linear_solve {
                    GaussNewtonLinearSolve::IterativeCg {
                        max_iterations,
                        warm_start,
                        ..
                    } => GaussNewtonLinearSolve::IterativeCg {
                        tolerance,
                        max_iterations,
                        warm_start,
                    },
                    GaussNewtonLinearSolve::DirectCholesky => GaussNewtonLinearSolve::IterativeCg {
                        tolerance,
                        max_iterations: 2048,
                        warm_start: true,
                    },
                };
            }
            "--cg-max-iterations" => {
                let max_iterations = next_arg(&mut args, "--cg-max-iterations")?.parse()?;
                config.linear_solve = match config.linear_solve {
                    GaussNewtonLinearSolve::IterativeCg {
                        tolerance,
                        warm_start,
                        ..
                    } => GaussNewtonLinearSolve::IterativeCg {
                        tolerance,
                        max_iterations,
                        warm_start,
                    },
                    GaussNewtonLinearSolve::DirectCholesky => GaussNewtonLinearSolve::IterativeCg {
                        tolerance: 1e-8,
                        max_iterations,
                        warm_start: true,
                    },
                };
            }
            "--cg-cold-start" => {
                config.linear_solve = match config.linear_solve {
                    GaussNewtonLinearSolve::IterativeCg {
                        tolerance,
                        max_iterations,
                        ..
                    } => GaussNewtonLinearSolve::IterativeCg {
                        tolerance,
                        max_iterations,
                        warm_start: false,
                    },
                    GaussNewtonLinearSolve::DirectCholesky => GaussNewtonLinearSolve::IterativeCg {
                        tolerance: 1e-8,
                        max_iterations: 2048,
                        warm_start: false,
                    },
                };
            }
            "--skip-output" => {
                config.write_outputs = false;
            }
            "--skip-cell-b-variance" => {
                config.include_cell_b_variance = false;
            }
            "--include-cell-b-variance" => {
                config.include_cell_b_variance = true;
            }
            "--skip-linear-baseline" => {
                run_linear_baseline = false;
            }
            "--diagnose-observation-only" => {
                diagnose_only = true;
            }
            other => {
                return Err(format!("unknown argument `{other}`; use --help").into());
            }
        }
    }

    if diagnose_only {
        let report = diagnose_toroidal_observation(&config)?;
        println!(
            "diagnostic: active_dofs={} cells={} observation_mode={} rows={}/{}",
            report.active_dofs,
            report.cells,
            report.pde_observation_mode.label(),
            report.residual_rows_used,
            report.residual_rows_total
        );
        println!(
            "weak residual: zero={:.6e} linear_mean={:.6e}",
            report.weak_residual_at_zero, report.weak_residual_at_linear_mean
        );
        println!(
            "observation residual: zero={} linear_mean={} weighted_zero={} weighted_linear_mean={} delta_zero_to_linear={} source_norm={} field_response_norm={} best_source_scale={} field_source_cosine={} jacobian_nnz={}",
            format_optional(report.observation_residual_at_zero),
            format_optional(report.observation_residual_at_linear_mean),
            format_optional(report.weighted_observation_residual_at_zero),
            format_optional(report.weighted_observation_residual_at_linear_mean),
            format_optional(report.observation_delta_from_zero_to_linear_mean),
            format_optional(report.observation_source_norm),
            format_optional(report.observation_field_response_norm),
            format_optional(report.observation_best_source_scale),
            format_optional(report.observation_field_source_cosine),
            report
                .observation_jacobian_nnz_at_linear_mean
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
        return Ok(());
    }

    if config.beta_core != 0.0 && run_linear_baseline {
        let mut baseline = config.clone();
        baseline.beta_core = 0.0;
        baseline.prior_mode = ToroidalPriorMode::WeakDiagonal;
        baseline.include_nonlinear_residual = true;
        baseline.write_outputs = false;
        baseline.include_cell_b_variance = false;
        println!("beta=0 linear comparison");
        let report = run_nonlinear_toroidal_inductor(&baseline)?;
        print_report(&report);
        println!();
    }

    let mut weak = config.clone();
    weak.prior_mode = ToroidalPriorMode::WeakDiagonal;
    weak.include_nonlinear_residual = true;
    weak.write_outputs = false;
    weak.include_cell_b_variance = false;
    println!("weak diagonal prior + nonlinear weak residual");
    print_report(&run_nonlinear_toroidal_inductor(&weak)?);
    println!();

    let mut proxy_only = config.clone();
    proxy_only.prior_mode = ToroidalPriorMode::LinearProxyMaternAlpha2;
    proxy_only.include_nonlinear_residual = false;
    proxy_only.write_outputs = false;
    proxy_only.include_cell_b_variance = false;
    println!("linear-proxy alpha-2 prior only");
    print_report(&run_nonlinear_toroidal_inductor(&proxy_only)?);
    println!();

    let mut proxy_corrected = config;
    proxy_corrected.prior_mode = ToroidalPriorMode::LinearProxyMaternAlpha2;
    proxy_corrected.include_nonlinear_residual = true;
    println!("linear-proxy alpha-2 prior + nonlinear weak residual");
    print_report(&run_nonlinear_toroidal_inductor(&proxy_corrected)?);
    Ok(())
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
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

fn print_help() {
    println!("Usage: nonlinear_toroidal_inductor [options]");
    println!("  --mesh <path>");
    println!("  --output-dir <path>");
    println!("  --beta-core <float>");
    println!("  --nu-core0 <float>");
    println!("  --pde-variance <float>");
    println!("  --prior-precision <float>");
    println!("  --prior-kappa <float>");
    println!("  --prior-tau <float>");
    println!("  --observation-mode <weak-galerkin-rows|local-strong-cells>");
    println!("  --max-iterations <usize>");
    println!("  --direct-cholesky-steps");
    println!("  --cg-tolerance <float>");
    println!("  --cg-max-iterations <usize>");
    println!("  --cg-cold-start");
    println!("  --skip-output");
    println!("  --skip-cell-b-variance");
    println!("  --include-cell-b-variance");
    println!("  --skip-linear-baseline");
    println!("  --diagnose-observation-only");
}

fn print_report(report: &NonlinearToroidalReport) {
    println!(
        "mesh: vertices={} edges={} cells={} active_dofs={} boundary_edges={} gauge_edges={}",
        report.vertices,
        report.edges,
        report.cells,
        report.active_dofs,
        report.boundary_edge_dofs,
        report.gauge_edge_dofs
    );
    println!(
        "solve: converged={} iterations={} residual {:.6e} -> {:.6e}",
        report.converged,
        report.iterations,
        report.initial_residual_norm,
        report.final_residual_norm
    );
    if let (Some(initial), Some(final_norm)) = (
        report.observation_initial_residual_norm,
        report.observation_final_residual_norm,
    ) {
        println!(
            "observation residual at solver mean: {:.6e} -> {:.6e}",
            initial, final_norm
        );
    }
    println!(
        "prior: mode={} kappa={:.6e} tau={:.6e} fallback={} residual_variance={:.6e} observation_mode={} residual_weighting={} residual_rows={}/{} residual_norm_scale={} nonlinear_residual={}",
        report.prior_mode.label(),
        report.prior_kappa,
        report.prior_tau,
        report.prior_kappa_fallback_used,
        report.residual_variance,
        report.pde_observation_mode.label(),
        report.residual_weighting.label(),
        report.residual_rows_used,
        report.residual_rows_total,
        format_optional(report.residual_precision_normalization),
        report.nonlinear_residual_likelihood
    );
    println!(
        "MAP distance from beta=0 linear mean: {:.6e}",
        report.map_relative_distance_from_linear_mean
    );
    if let Some(error) = report.direct_linear_relative_error {
        println!("direct beta=0 MAP relative error: {:.6e}", error);
    }
    if let Some(error) = report.mixed_reference_b_relative_error {
        println!("mixed-reference B relative error: {:.6e}", error);
    }
    println!(
        "harmonic: coeff_norm={:.6e} coefficients={}",
        report.harmonic_coefficient_norm,
        format_coefficients(&report.harmonic_coefficients)
    );
    println!(
        "posterior: precision_nnz={} factorizes={} latent_variances={}",
        report.posterior_precision_nnz, report.posterior_factorizes, report.latent_variance_len
    );
    println!(
        "linear solve: mode={} cg_iterations={} max_residual={:.6e}; final Cholesky nnz={} time={:.3}s",
        format_linear_solve_mode(report.linear_solve_mode),
        report.linear_solve_iteration_sum,
        report.linear_solve_residual_max,
        report.posterior_factor_nnz,
        report.final_factorization_seconds
    );
    match (
        report.b_variance_len,
        report.b_variance_min,
        report.b_variance_max,
    ) {
        (Some(len), Some(min), Some(max)) => {
            println!(
                "cell B posterior variance: len={} min={:.6e} max={:.6e}",
                len, min, max
            );
        }
        _ => println!("cell B posterior variance: skipped"),
    }
    if !report.sensor_reports.is_empty() {
        println!("flux sensors:");
        for sensor in &report.sensor_reports {
            println!(
                "  {} nonlinear={:.6e} beta0={} mixed={}",
                sensor.name,
                sensor.nonlinear_value,
                format_optional(sensor.beta_zero_value),
                format_optional(sensor.mixed_reference_value)
            );
        }
    }
    if !report.result.history.is_empty() {
        println!("iterations:");
        for step in &report.result.history {
            println!(
                "  {:>2}: objective {:.6e} -> {:.6e}, residual {:.6e}, alpha {:.3}, step {:.6e}, linear_iters={}, linear_residual={:.3e}",
                step.iteration,
                step.objective,
                step.trial_objective,
                step.residual_norm,
                step.alpha,
                step.step_norm,
                step.linear_solve.iterations,
                step.linear_solve.final_residual_norm
            );
        }
    }
}

fn format_linear_solve_mode(mode: GaussNewtonLinearSolveMode) -> &'static str {
    match mode {
        GaussNewtonLinearSolveMode::IterativeCg => "cg_jacobi",
        GaussNewtonLinearSolveMode::DirectCholesky => "direct_cholesky",
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_coefficients(values: &[f64]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("{value:.6e}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
