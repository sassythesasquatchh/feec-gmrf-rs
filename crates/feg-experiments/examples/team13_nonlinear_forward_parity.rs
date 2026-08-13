use feg_case_studies::team13::{
    run_team13_nonlinear_forward_parity, Team13DomainMode, Team13NonlinearForwardParityConfig,
    Team13NonlinearForwardParityResult, Team13NonlinearMaterialKind,
};
use feg_infer::nonlinear::GaussNewtonLinearSolve;
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = Team13NonlinearForwardParityConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--domain" => config.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?,
            "--ampere-turns" => {
                config.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--material" => {
                config.material_kind = parse_material_kind(&next_arg(&mut args, "--material")?)?;
            }
            "--beta-iron" => config.beta_iron = next_arg(&mut args, "--beta-iron")?.parse()?,
            "--b-scale-tesla" => {
                config.b_scale_tesla = next_arg(&mut args, "--b-scale-tesla")?.parse()?
            }
            "--magnitude-smoothing-tesla" => {
                config.magnitude_smoothing_tesla =
                    next_arg(&mut args, "--magnitude-smoothing-tesla")?.parse()?
            }
            "--max-iterations" => {
                config.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
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
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?));
            }
            "--skip-output" => config.output_dir = None,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    let result = run_team13_nonlinear_forward_parity(&config)?;
    print_result(&result, &config.linear_solve);
    Ok(())
}

fn print_result(result: &Team13NonlinearForwardParityResult, solve: &GaussNewtonLinearSolve) {
    println!("TEAM 13 FEEC nonlinear forward parity diagnostic");
    println!("  mesh: {}", result.mesh_path.display());
    println!(
        "  domain={} material={} ampere_turns={:.6e}",
        result.domain_mode.as_str(),
        result.material_kind.as_str(),
        result.ampere_turns
    );
    println!(
        "  mesh: vertices={} edges={} cells={} active_dofs={} boundary_edge_dofs={}",
        result.vertices, result.edges, result.cells, result.active_dofs, result.boundary_edge_dofs
    );
    println!(
        "  nonlinear: converged={} iterations={} residual={:.6e}->{:.6e} solution_l2={:.6e}->{:.6e}",
        result.converged,
        result.iterations,
        result.initial_residual_l2,
        result.final_residual_l2,
        result.initial_solution_l2,
        result.nonlinear_solution_l2
    );
    println!(
        "  operators: residual_dim={} initial_jacobian_nnz={} final_jacobian_nnz={} rhs_l2={:.6e}",
        result.residual_dimension,
        result.initial_jacobian_nnz,
        result.final_jacobian_nnz,
        result.rhs_l2
    );
    println!("  linear_solve={}", linear_solve_label(solve));
    println!(
        "  steel RMSE G=0.52: initial={:.6e} nonlinear={:.6e}",
        result.initial_steel_rmse_g052, result.nonlinear_steel_rmse_g052
    );
    println!(
        "  steel RMSE G=0.47: initial={:.6e} nonlinear={:.6e}",
        result.initial_steel_rmse_g047, result.nonlinear_steel_rmse_g047
    );
    for summary in &result.nonlinear_steel_group_summaries {
        println!(
            "  steel group {}: count={} rmse_g052={:.6e} rmse_g047={:.6e} max_g052={:.6e} max_g047={:.6e}",
            summary.group.as_str(),
            summary.count,
            summary.rmse_g_052,
            summary.rmse_g_047,
            summary.max_abs_residual_g_052,
            summary.max_abs_residual_g_047
        );
    }
    if let Some(output_dir) = &result.output_dir {
        println!("  outputs: {}", output_dir.display());
    }
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
        "Usage: team13_nonlinear_forward_parity --mesh <path> [options]\n\
         Options:\n\
           --domain <half|full>\n\
           --ampere-turns <float>\n\
           --material <ngsolve-tabulated-linear|smooth-quadratic>\n\
           --beta-iron <float>\n\
           --b-scale-tesla <float>\n\
           --magnitude-smoothing-tesla <float>\n\
           --max-iterations <usize>\n\
           --linear-solve <direct-cholesky|iterative-cg>\n\
           --cg-tolerance <float>\n\
           --cg-max-iterations <usize>\n\
           --output-dir <path>\n\
           --skip-output"
    );
}
