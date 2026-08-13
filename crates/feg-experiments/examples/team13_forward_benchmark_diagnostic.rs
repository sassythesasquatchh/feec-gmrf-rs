use feg_case_studies::team13::{
    run_team13_forward_benchmark_diagnostic, Team13DomainMode, Team13NonlinearMaterialKind,
    Team13SyntheticBenchmarkGeometryConfig,
};
use feg_infer::nonlinear::GaussNewtonLinearSolve;
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = Team13SyntheticBenchmarkGeometryConfig {
        source_scale_diagnostic_values: vec![1.0],
        sweep_pde_variances: vec![1.0e4],
        sweep_observation_std_tesla: vec![1.0e-3],
        ..Team13SyntheticBenchmarkGeometryConfig::default()
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--domain" => config.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?,
            "--ampere-turns" => {
                config.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--source-scale-values" => {
                config.source_scale_diagnostic_values =
                    parse_csv_f64(&next_arg(&mut args, "--source-scale-values")?)?;
            }
            "--material" => {
                config.material_kind = parse_material_kind(&next_arg(&mut args, "--material")?)?;
            }
            "--truth-pde-variance" => {
                config.truth_pde_variance = next_arg(&mut args, "--truth-pde-variance")?.parse()?;
            }
            "--truth-prior-precision" => {
                config.truth_prior_precision =
                    next_arg(&mut args, "--truth-prior-precision")?.parse()?;
            }
            "--max-iterations" => {
                config.truth_max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
            }
            "--linear-solve" => {
                config.linear_solve = parse_linear_solve(&next_arg(&mut args, "--linear-solve")?)?;
            }
            "--cg-tolerance" => {
                let tolerance = next_arg(&mut args, "--cg-tolerance")?.parse()?;
                config.linear_solve = with_cg_tolerance(config.linear_solve, tolerance)?;
            }
            "--cg-max-iterations" => {
                let max_iterations = next_arg(&mut args, "--cg-max-iterations")?.parse()?;
                config.linear_solve = with_cg_max_iterations(config.linear_solve, max_iterations)?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    let result = run_team13_forward_benchmark_diagnostic(&config)?;
    println!("TEAM 13 FEEC forward benchmark diagnostic");
    println!("  mesh: {}", config.mesh_path.display());
    println!(
        "  domain={} material={} vertices={} edges={} cells={} active_dofs={} boundary_edge_dofs={}",
        result.domain_mode.as_str(),
        result.material_kind.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs
    );
    println!(
        "  observations={} assimilated={} steel_quadrature={}",
        result.observation_count,
        result.assimilated_observation_count,
        result.steel_observation_quadrature.as_str()
    );
    println!(
        "  linear_solve={}",
        linear_solve_label(&config.linear_solve)
    );
    for run in &result.source_scale_diagnostics {
        println!(
            "  alpha={:.6e} converged={} residual={:.6e}->{:.6e} steel_rmse_g052={:.6e} steel_rmse_g047={:.6e}",
            run.source_scale,
            run.converged,
            run.initial_residual_norm,
            run.final_residual_norm,
            run.steel_rmse_g_052,
            run.steel_rmse_g_047
        );
        if let Some(error) = &run.error {
            println!("    error: {error}");
        }
        for summary in &run.group_summaries {
            println!(
                "    group {}: count={} rmse_g052={:.6e} rmse_g047={:.6e} max_g052={:.6e} max_g047={:.6e}",
                summary.group.as_str(),
                summary.count,
                summary.rmse_g_052,
                summary.rmse_g_047,
                summary.max_abs_residual_g_052,
                summary.max_abs_residual_g_047
            );
        }
    }

    Ok(())
}

fn parse_linear_solve(raw: &str) -> Result<GaussNewtonLinearSolve, Box<dyn Error>> {
    match raw {
        "direct-cholesky" | "direct" | "cholesky" => Ok(GaussNewtonLinearSolve::DirectCholesky),
        "iterative-cg" | "cg" => Ok(GaussNewtonLinearSolve::IterativeCg {
            tolerance: 1.0e-8,
            max_iterations: 2048,
            warm_start: true,
        }),
        other => Err(format!(
            "unknown linear solve `{other}`; expected direct-cholesky or iterative-cg"
        )
        .into()),
    }
}

fn with_cg_tolerance(
    solve: GaussNewtonLinearSolve,
    tolerance: f64,
) -> Result<GaussNewtonLinearSolve, Box<dyn Error>> {
    match solve {
        GaussNewtonLinearSolve::IterativeCg {
            max_iterations,
            warm_start,
            ..
        } => Ok(GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            warm_start,
        }),
        GaussNewtonLinearSolve::DirectCholesky => {
            Err("--cg-tolerance requires --linear-solve iterative-cg".into())
        }
    }
}

fn with_cg_max_iterations(
    solve: GaussNewtonLinearSolve,
    max_iterations: usize,
) -> Result<GaussNewtonLinearSolve, Box<dyn Error>> {
    match solve {
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            warm_start,
            ..
        } => Ok(GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            warm_start,
        }),
        GaussNewtonLinearSolve::DirectCholesky => {
            Err("--cg-max-iterations requires --linear-solve iterative-cg".into())
        }
    }
}

fn linear_solve_label(solve: &GaussNewtonLinearSolve) -> String {
    match solve {
        GaussNewtonLinearSolve::DirectCholesky => "direct-cholesky".to_string(),
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            warm_start,
        } => format!(
            "iterative-cg(tol={tolerance:.1e},max_iterations={max_iterations},warm_start={warm_start})"
        ),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn parse_csv_f64(raw: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    raw.split(',')
        .map(|value| Ok(value.trim().parse::<f64>()?))
        .collect()
}

fn parse_domain(raw: &str) -> Result<Team13DomainMode, Box<dyn Error>> {
    match raw {
        "half" | "half-z" => Ok(Team13DomainMode::HalfZNonnegative),
        "full" => Ok(Team13DomainMode::Full),
        other => Err(format!("unknown domain `{other}`; expected half or full").into()),
    }
}

fn parse_material_kind(raw: &str) -> Result<Team13NonlinearMaterialKind, Box<dyn Error>> {
    match raw {
        "ngsolve-tabulated-linear" | "tabulated" | "ngsolve" => {
            Ok(Team13NonlinearMaterialKind::NgsolveTabulatedLinear)
        }
        "smooth-quadratic" | "smooth" => Ok(Team13NonlinearMaterialKind::SmoothQuadratic),
        other => Err(format!("unknown material `{other}`").into()),
    }
}

fn print_help() {
    println!(
        "Usage: team13_forward_benchmark_diagnostic --mesh <path> [options]\n\
         Options:\n\
           --domain <half|full>\n\
           --ampere-turns <float>\n\
           --source-scale-values <csv>\n\
           --material <ngsolve-tabulated-linear|smooth-quadratic>\n\
           --truth-pde-variance <float>\n\
           --truth-prior-precision <float>\n\
           --max-iterations <usize>\n\
           --linear-solve <direct-cholesky|iterative-cg>\n\
           --cg-tolerance <float>\n\
           --cg-max-iterations <usize>"
    );
}
