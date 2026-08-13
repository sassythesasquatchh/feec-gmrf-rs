use feg_case_studies::team13::{
    run_team13_nonlinear_uq, Team13DomainMode, Team13NonlinearConfig, Team13NonlinearMaterialKind,
};
use feg_infer::{
    linear_pde::{LinearPdeVarianceConfig, LinearPdeVarianceMode},
    nonlinear::{GaussNewtonLinearSolve, GaussNewtonStepRegularization, NonlinearAssemblyTermKind},
};
use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone)]
struct ExampleConfig {
    domain_mode: Team13DomainMode,
    mesh_path: Option<PathBuf>,
    geo_path: PathBuf,
    mesh_scale: f64,
    force_mesh_generation: bool,
    solve: Team13NonlinearConfig,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            domain_mode: Team13DomainMode::HalfZNonnegative,
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 8.0,
            force_mesh_generation: false,
            solve: Team13NonlinearConfig::default(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = parse_args()?;
    let workspace = workspace_root()?;
    let geo_path = absolutize(&workspace, &config.geo_path);
    let mesh_path = config
        .mesh_path
        .clone()
        .map(|path| absolutize(&workspace, &path))
        .unwrap_or_else(|| {
            workspace.join(format!(
                "target/team13_nonlinear/team13_{}_nonlinear.msh",
                config.domain_mode.as_str()
            ))
        });
    if config.force_mesh_generation || !mesh_path.exists() {
        generate_mesh(&geo_path, &mesh_path, config.domain_mode, config.mesh_scale)?;
    }

    config.solve.mesh_path = mesh_path.clone();
    config.solve.domain_mode = config.domain_mode;
    if config.solve.write_outputs && config.solve.output_dir.is_none() {
        config.solve.output_dir = Some(workspace.join(format!(
            "out/examples/team13_nonlinear_uq/{}",
            config.domain_mode.as_str()
        )));
    }

    let result = run_team13_nonlinear_uq(&config.solve)?;
    println!("TEAM 13 nonlinear FEEC/GMRF solve");
    println!("  domain: {}", result.domain_mode.as_str());
    println!("  mesh: {}", mesh_path.display());
    println!(
        "  size: vertices={} edges={} cells={} active_dofs={} boundary_edges={}",
        result.vertices, result.edges, result.cells, result.active_dofs, result.boundary_edge_dofs
    );
    println!(
        "  material: {} beta_iron={:.6e}, B_scale={:.6e} T",
        result.material_kind.as_str(),
        result.beta_iron,
        result.b_scale_tesla
    );
    println!(
        "  prior: {} alpha-2, kappa={:.6e}, tau={:.6e}, scale={:.6e}, fallback={}",
        result.prior_kind.as_str(),
        result.prior_kappa,
        result.prior_tau,
        result.field_prior_precision_scale,
        result.prior_kappa_fallback_used
    );
    println!(
        "  residual: zero={:.6e}, linear_mean={:.6e}, final={:.6e}",
        result.initial_residual_norm, result.linear_mean_residual_norm, result.final_residual_norm
    );
    println!(
        "  map distance from beta-zero linear mean: abs={:.6e}, rel={:.6e}",
        result.map_distance_from_linear_mean, result.map_relative_distance_from_linear_mean
    );
    if let Some(error) = result.beta_zero_relative_error {
        println!("  beta=0 parity relative MAP error: {error:.6e}");
    }
    println!(
        "  sensors: count={} assimilated={} linear_rmse={:.6e} T nonlinear_rmse={:.6e} T ratio={:.6e} selected_variances={}",
        result.sensor_reports.len(),
        result.assimilated_measurements,
        result.linear_sensor_rmse,
        result.sensor_rmse,
        result.sensor_rmse_improvement_ratio,
        result.sensor_variances.len()
    );
    println!(
        "  measurement model: 25 NGSolve-style TEAM 13 surface-averaged signed B-component magnitudes; assimilation={}",
        if result.assimilated_measurements > 0 {
            "enabled"
        } else {
            "disabled/reporting-only"
        }
    );
    println!(
        "  B relative difference vs beta-zero linear mean: {:.6e}",
        result.field_metrics_vs_linear.b_vector_relative_l2_error
    );
    println!(
        "  assembly stats: prior_nnz={} residual_jacobian_nnz={} residual_normal_update_nnz={} measurement_operator_nnz={} measurement_update_nnz={} posterior_precision_nnz={} posterior_lower_nnz={}",
        result.assembly.prior_precision_nnz,
        result
            .assembly
            .term_operator_nnz(NonlinearAssemblyTermKind::Residual),
        result
            .assembly
            .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual),
        result
            .assembly
            .term_operator_nnz(NonlinearAssemblyTermKind::LinearMeasurement),
        result
            .assembly
            .term_precision_update_nnz(NonlinearAssemblyTermKind::LinearMeasurement),
        result.assembly.posterior_precision_nnz,
        result.assembly.posterior_precision_lower_triangle_nnz
    );
    println!(
        "  final Cholesky factor: factor_nnz={} fill_vs_lower={:.3}x time={:.3}s converged={}",
        result.final_factorization.nnz,
        result
            .assembly
            .fill_ratio_vs_lower_triangle
            .unwrap_or(f64::NAN),
        result.final_factorization.elapsed_seconds,
        result.converged
    );
    for step in &result.history {
        println!(
            "    iter={} obj={:.6e}->{:.6e} residual={:.6e} alpha={:.3e} lambda={:.1e} step={:.6e} linear={:?}/{} res={:.3e}",
            step.iteration,
            step.objective,
            step.trial_objective,
            step.residual_norm,
            step.alpha,
            step.regularization_lambda,
            step.step_norm,
            step.linear_solve.mode,
            step.linear_solve.iterations,
            step.linear_solve.final_residual_norm
        );
    }
    if let Some(output_dir) = &config.solve.output_dir {
        if config.solve.write_outputs {
            println!("  outputs: {}", output_dir.display());
        }
    }
    Ok(())
}

fn parse_args() -> Result<ExampleConfig, Box<dyn Error>> {
    let mut config = ExampleConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--mesh" => config.mesh_path = Some(PathBuf::from(next_arg(&mut args, "--mesh")?)),
            "--geo" => config.geo_path = PathBuf::from(next_arg(&mut args, "--geo")?),
            "--mesh-scale" => config.mesh_scale = next_arg(&mut args, "--mesh-scale")?.parse()?,
            "--full-domain" => config.domain_mode = Team13DomainMode::Full,
            "--force-mesh-generation" => config.force_mesh_generation = true,
            "--output-dir" => {
                config.solve.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?))
            }
            "--skip-output" => config.solve.write_outputs = false,
            "--assimilate-measurements" => config.solve.assimilate_measurements = true,
            "--no-assimilate-measurements" => config.solve.assimilate_measurements = false,
            "--beta-iron" => {
                config.solve.beta_iron = next_arg(&mut args, "--beta-iron")?.parse()?
            }
            "--material-kind" => {
                config.solve.material_kind =
                    parse_material_kind(&next_arg(&mut args, "--material-kind")?)?
            }
            "--ngsolve-tabulated-material" => {
                config.solve.material_kind = Team13NonlinearMaterialKind::NgsolveTabulatedLinear
            }
            "--smooth-material" => {
                config.solve.material_kind = Team13NonlinearMaterialKind::SmoothQuadratic
            }
            "--b-scale" => {
                config.solve.b_scale_tesla = next_arg(&mut args, "--b-scale")?.parse()?
            }
            "--ampere-turns" => {
                config.solve.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--pde-variance" => {
                config.solve.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?
            }
            "--field-prior-scale" => {
                config.solve.field_prior_precision_scale =
                    next_arg(&mut args, "--field-prior-scale")?.parse()?
            }
            "--prior-kappa" => {
                config.solve.prior_kappa = Some(next_arg(&mut args, "--prior-kappa")?.parse()?)
            }
            "--prior-tau" => {
                config.solve.prior_tau = next_arg(&mut args, "--prior-tau")?.parse()?
            }
            "--allow-kappa-fallback" => config.solve.prior_allow_kappa_fallback = true,
            "--max-iterations" => {
                config.solve.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?
            }
            "--sensor-variance-count" => {
                config.solve.sensor_variance_count =
                    next_arg(&mut args, "--sensor-variance-count")?.parse()?
            }
            "--direct-cholesky-steps" => {
                config.solve.linear_solve = GaussNewtonLinearSolve::DirectCholesky
            }
            "--undamped-steps" => {
                config.solve.step_regularization = GaussNewtonStepRegularization::None
            }
            "--lm-grid-steps" => {
                config.solve.step_regularization =
                    GaussNewtonStepRegularization::LevenbergMarquardtGrid
            }
            "--cg-tolerance" => {
                let tolerance = next_arg(&mut args, "--cg-tolerance")?.parse()?;
                config.solve.linear_solve = match config.solve.linear_solve {
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
            "--variance-mc-probes" => {
                config.solve.variance = LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::MonteCarlo,
                    num_variance_probes: next_arg(&mut args, "--variance-mc-probes")?.parse()?,
                    ..config.solve.variance
                };
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    Ok(config)
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_material_kind(value: &str) -> Result<Team13NonlinearMaterialKind, Box<dyn Error>> {
    match value {
        "ngsolve" | "ngsolve-table" | "ngsolve-tabulated-linear" => {
            Ok(Team13NonlinearMaterialKind::NgsolveTabulatedLinear)
        }
        "smooth" | "smooth-quadratic" => Ok(Team13NonlinearMaterialKind::SmoothQuadratic),
        other => Err(format!(
            "unknown material kind `{other}`; expected ngsolve-tabulated-linear or smooth-quadratic"
        )
        .into()),
    }
}

fn workspace_root() -> io::Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "workspace root not found"))
}

fn absolutize(workspace: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

fn generate_mesh(
    geo_path: &Path,
    mesh_path: &Path,
    domain_mode: Team13DomainMode,
    mesh_scale: f64,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = mesh_path.parent() {
        fs::create_dir_all(parent)?;
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

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_nonlinear_uq -- [options]\n\
         \n\
         Options:\n\
           --mesh PATH                 Use an existing TEAM13 mesh\n\
           --geo PATH                  Geometry path for mesh generation\n\
           --mesh-scale VALUE          Gmsh MeshScale, default 8\n\
           --full-domain               Generate/use the full z-symmetric domain\n\
           --force-mesh-generation     Regenerate the mesh even if it exists\n\
           --material-kind VALUE       ngsolve-tabulated-linear or smooth-quadratic\n\
           --smooth-material           Use the legacy smooth quadratic iron law\n\
           --beta-iron VALUE           Smooth nonlinear iron beta, default 10\n\
           --b-scale VALUE             B scale in tesla, default 1\n\
           --pde-variance VALUE        PDE residual variance, default 1.2e4\n\
           --field-prior-scale VALUE   Prior precision scale, default 1e-12\n\
           --prior-kappa VALUE         Prior kappa, default 1 for unweighted Hodge prior\n\
           --prior-tau VALUE           Prior tau, default 1\n\
           --allow-kappa-fallback      Accepted for compatibility; direct Hodge prior does not fallback\n\
           --max-iterations N          Gauss-Newton iteration cap\n\
          --sensor-variance-count N   Number of linearized TEAM13 surface sensors with reported variance\n\
          --assimilate-measurements   Use the 25 NGSolve-style TEAM13 surface-averaged signed B-component magnitudes in the nonlinear objective\n\
          --no-assimilate-measurements Keep benchmark surface B rows as reporting-only diagnostics\n\
           --skip-output               Do not write VTU/CSV outputs\n\
           --direct-cholesky-steps     Use direct intermediate Gauss-Newton solves\n\
           --undamped-steps            Disable intermediate LM-grid step regularization\n\
           --lm-grid-steps             Enable intermediate LM-grid step regularization"
    );
}
