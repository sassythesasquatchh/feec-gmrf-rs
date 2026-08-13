use feg_case_studies::team13::{
    run_team13_synthetic_nonlinear_baseline, Team13DomainMode,
    Team13SyntheticNonlinearBaselineConfig,
};
use feg_infer::nonlinear::NonlinearAssemblyTermKind;
use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    process::Command,
};

struct CliConfig {
    mesh_path: Option<PathBuf>,
    geo_path: PathBuf,
    mesh_scale: usize,
    force_mesh_generation: bool,
    solve: Team13SyntheticNonlinearBaselineConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 18,
            force_mesh_generation: false,
            solve: Team13SyntheticNonlinearBaselineConfig::default(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = parse_args()?;
    let mesh_path = config
        .mesh_path
        .take()
        .unwrap_or_else(|| config.solve.mesh_path.clone());
    let geo_path = workspace_path(&config.geo_path)?;
    if config.force_mesh_generation || !mesh_path.exists() {
        generate_mesh(
            &geo_path,
            &mesh_path,
            config.solve.domain_mode,
            config.mesh_scale,
        )?;
    }
    config.solve.mesh_path = mesh_path.clone();

    let result = run_team13_synthetic_nonlinear_baseline(&config.solve)?;

    println!("TEAM 13 synthetic nonlinear surface-magnitude baseline");
    println!("  mesh: {}", mesh_path.display());
    println!(
        "  domain={} vertices={} edges={} cells={} active_dofs={} boundary_edge_dofs={}",
        result.domain_mode.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs
    );
    println!(
        "  beta_iron={:.6e} b_scale_tesla={:.6e} prior_kappa={:.6e} prior_tau={:.6e} diag_shift={:.6e}",
        result.beta_iron,
        result.b_scale_tesla,
        result.prior_kappa,
        result.prior_tau,
        result.prior_diagonal_shift
    );
    println!(
        "  magnitude_smoothing_tesla={:.6e} synthetic_sensors={}",
        result.magnitude_smoothing_tesla, result.synthetic_sensor_count
    );
    println!(
        "  truth: converged={} initial_residual={:.6e} truth_residual={:.6e}",
        result.truth_converged, result.initial_residual_norm, result.truth_residual_norm
    );
    println!();

    for run in &result.observation_runs {
        println!("observation model: {}", run.model_kind.as_str());
        println!(
            "  converged={} sign_mismatches={} sensors={}",
            run.posterior_converged, run.sign_mismatch_count, run.synthetic_sensor_count
        );
        println!(
            "  A relative error: initial={:.6e} posterior={:.6e} ratio={:.6e}",
            run.initial_relative_error,
            run.posterior_relative_error,
            safe_ratio(run.posterior_relative_error, run.initial_relative_error)
        );
        println!(
            "  sensor RMSE: initial={:.6e} posterior={:.6e} ratio={:.6e}",
            run.initial_sensor_rmse, run.posterior_sensor_rmse, run.sensor_rmse_improvement_ratio
        );
        println!(
            "  PDE residual: initial={:.6e} truth={:.6e} posterior={:.6e}",
            run.initial_residual_norm, run.truth_residual_norm, run.posterior_residual_norm
        );
        println!(
            "  variances: finite={} nonnegative={} sensor_variances={}",
            run.all_finite_variances,
            run.nonnegative_variances,
            run.sensor_variances.len()
        );
        println!(
            "  assembly: prior_nnz={} residual_jacobian_nnz={} residual_update_nnz={} linear_measurement_nnz={} linear_measurement_update_nnz={} posterior_nnz={} factor_nnz={} fill_vs_lower={:.6e}",
            run.assembly.prior_precision_nnz,
            run.assembly
                .term_operator_nnz(NonlinearAssemblyTermKind::Residual),
            run.assembly
                .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual),
            run.assembly
                .term_operator_nnz(NonlinearAssemblyTermKind::LinearMeasurement),
            run.assembly
                .term_precision_update_nnz(NonlinearAssemblyTermKind::LinearMeasurement),
            run.assembly.posterior_precision_nnz,
            run.assembly
                .factor_nnz
                .unwrap_or(run.final_factorization.nnz),
            run.assembly
                .fill_ratio_vs_lower_triangle
                .unwrap_or(f64::NAN)
        );
        println!(
            "  factorization: nnz={} elapsed={:.6e}s",
            run.final_factorization.nnz, run.final_factorization.elapsed_seconds
        );
        if let Some(last) = run.posterior_history.last() {
            println!(
                "  final GN step: iter={} alpha={:.6e} objective={:.6e} trial_objective={:.6e} residual={:.6e} step_norm={:.6e}",
                last.iteration,
                last.alpha,
                last.objective,
                last.trial_objective,
                last.residual_norm,
                last.step_norm
            );
        }
        for sensor in &run.sensor_reports {
            println!(
                "    sensor {} observed={:.6e} initial={:.6e} posterior={:.6e} residual={:.6e}",
                sensor.name,
                sensor.observed,
                sensor.nominal_prediction,
                sensor.posterior_prediction,
                sensor.residual
            );
        }
        println!();
    }

    Ok(())
}

fn parse_args() -> Result<CliConfig, Box<dyn Error>> {
    let mut config = CliConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = Some(PathBuf::from(next_arg(&mut args, "--mesh")?)),
            "--geo" => config.geo_path = PathBuf::from(next_arg(&mut args, "--geo")?),
            "--mesh-scale" => {
                config.mesh_scale = next_arg(&mut args, "--mesh-scale")?.parse()?;
            }
            "--force-mesh" => config.force_mesh_generation = true,
            "--max-iterations" => {
                config.solve.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?;
            }
            "--truth-max-iterations" => {
                config.solve.truth_max_iterations =
                    next_arg(&mut args, "--truth-max-iterations")?.parse()?;
            }
            "--pde-variance" => {
                config.solve.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?;
            }
            "--observation-std-tesla" => {
                config.solve.observation_std_tesla =
                    next_arg(&mut args, "--observation-std-tesla")?.parse()?;
            }
            "--magnitude-smoothing-tesla" => {
                config.solve.magnitude_smoothing_tesla =
                    next_arg(&mut args, "--magnitude-smoothing-tesla")?.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }
    Ok(config)
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn workspace_path(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = env::current_dir()?;
    Ok(cwd.join(path))
}

fn generate_mesh(
    geo_path: &Path,
    mesh_path: &Path,
    domain_mode: Team13DomainMode,
    mesh_scale: usize,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = mesh_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let full_domain = if domain_mode == Team13DomainMode::Full {
        "1"
    } else {
        "0"
    };
    let status = Command::new("gmsh")
        .arg("-3")
        .arg(geo_path)
        .arg("-setnumber")
        .arg("FullDomain")
        .arg(full_domain)
        .arg("-setnumber")
        .arg("MeshScale")
        .arg(mesh_scale.to_string())
        .arg("-o")
        .arg(mesh_path)
        .status()?;
    if !status.success() {
        return Err(format!("gmsh failed while generating `{}`", mesh_path.display()).into());
    }
    Ok(())
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= f64::EPSILON {
        if numerator.abs() <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_synthetic_nonlinear_baseline -- [options]\n\
         Options:\n\
           --mesh <path>                         Mesh path to use or generate\n\
           --geo <path>                          Gmsh geometry path (default: geometries/team13_linear.geo)\n\
           --mesh-scale <usize>                  TEAM 13 gmsh MeshScale value (default: 18)\n\
           --force-mesh                          Regenerate the mesh even if it exists\n\
           --max-iterations <usize>              Posterior Gauss-Newton iterations\n\
           --truth-max-iterations <usize>        Truth solve Gauss-Newton iterations\n\
           --pde-variance <float>                PDE weak residual variance\n\
           --observation-std-tesla <float>       Synthetic surface-magnitude observation std dev\n\
           --magnitude-smoothing-tesla <float>   Smooth absolute-value epsilon\n\
           --help                                Print this help"
    );
}
