use feg_case_studies::team13::{
    run_team13_nonlinear_diagnostics, Team13DomainMode, Team13NonlinearDiagnosticsConfig,
    Team13NonlinearMaterialKind,
};
use feg_infer::nonlinear::{
    GaussNewtonLinearSolve, GaussNewtonStepRegularization, NonlinearAssemblyTermKind,
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
    diagnostics: Team13NonlinearDiagnosticsConfig,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            domain_mode: Team13DomainMode::HalfZNonnegative,
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 8.0,
            force_mesh_generation: false,
            diagnostics: Team13NonlinearDiagnosticsConfig::default(),
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

    config.diagnostics.solve.mesh_path = mesh_path.clone();
    config.diagnostics.solve.domain_mode = config.domain_mode;
    config.diagnostics.solve.write_outputs = false;

    let result = run_team13_nonlinear_diagnostics(&config.diagnostics)?;
    println!("metric,value");
    println!("domain,{}", result.domain_mode.as_str());
    println!("mesh,{}", mesh_path.display());
    println!("vertices,{}", result.vertices);
    println!("edges,{}", result.edges);
    println!("cells,{}", result.cells);
    println!("active_dofs,{}", result.active_dofs);
    println!("boundary_edge_dofs,{}", result.boundary_edge_dofs);
    println!("material_kind,{}", result.material_kind.as_str());
    println!("beta_iron,{:.12e}", result.beta_iron);
    println!(
        "reduced_physical_rhs_norm,{:.12e}",
        result.reduced_physical_rhs_norm
    );
    println!(
        "beta_zero_residual_norm,{:.12e}",
        result.beta_zero_residual_norm
    );
    println!(
        "source_free_affine_solve_residual_norm,{:.12e}",
        result.source_free_affine_solve_residual_norm
    );
    println!(
        "nonlinear_residual_at_linear_mean_norm,{:.12e}",
        result.nonlinear_residual_at_linear_mean_norm
    );
    println!("prior_kind,{}", result.prior_kind.as_str());
    println!(
        "field_prior_precision_scale,{:.12e}",
        result.field_prior_precision_scale
    );
    println!(
        "assimilated_measurements,{}",
        result.assimilated_measurements
    );
    println!();
    println!("pde_variance,step_regularization,lambda,classification,objective,weighted_residual_norm,gradient_norm,step_norm,directional_derivative,accepted_alpha,accepted_objective,linear_abs_residual,linear_rel_residual,prior_precision_nnz,residual_jacobian_nnz,residual_normal_update_nnz,measurement_operator_nnz,measurement_update_nnz,posterior_precision_nnz,posterior_lower_triangle_nnz,failure");
    for row in &result.first_steps {
        if let Some(diagnostic) = &row.diagnostics {
            println!(
                "{:.12e},{:?},{:.12e},{:?},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{},{},{},{},",
                row.pde_variance,
                row.step_regularization,
                diagnostic.regularization_lambda,
                row.classification,
                diagnostic.objective,
                diagnostic.weighted_residual_norm,
                diagnostic.gradient_norm,
                diagnostic.step_norm,
                diagnostic.directional_derivative,
                diagnostic.accepted_alpha.unwrap_or(f64::NAN),
                diagnostic.accepted_objective.unwrap_or(f64::NAN),
                diagnostic.linear_solve_absolute_residual_norm,
                diagnostic.linear_solve_relative_residual_norm,
                diagnostic.assembly.prior_precision_nnz,
                diagnostic
                    .assembly
                    .term_operator_nnz(NonlinearAssemblyTermKind::Residual),
                diagnostic
                    .assembly
                    .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual),
                diagnostic
                    .assembly
                    .term_operator_nnz(NonlinearAssemblyTermKind::LinearMeasurement),
                diagnostic
                    .assembly
                    .term_precision_update_nnz(NonlinearAssemblyTermKind::LinearMeasurement),
                diagnostic.assembly.posterior_precision_nnz,
                diagnostic.assembly.posterior_precision_lower_triangle_nnz
            );
        } else {
            println!(
                "{:.12e},{:?},nan,{:?},nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,0,0,0,0,0,{}",
                row.pde_variance,
                row.step_regularization,
                row.classification,
                row.failure_reason.as_deref().unwrap_or("unknown failure")
            );
        }
    }
    println!();
    println!("pde_variance,step_regularization,lambda,alpha,objective");
    for row in &result.first_steps {
        if let Some(diagnostic) = &row.diagnostics {
            for sample in &diagnostic.objective_grid {
                println!(
                    "{:.12e},{:?},{:.12e},{:.12e},{:.12e}",
                    row.pde_variance,
                    row.step_regularization,
                    diagnostic.regularization_lambda,
                    sample.alpha,
                    sample.objective
                );
            }
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
            "--skip-output" => config.diagnostics.solve.write_outputs = false,
            "--assimilate-measurements" => config.diagnostics.solve.assimilate_measurements = true,
            "--no-assimilate-measurements" => {
                config.diagnostics.solve.assimilate_measurements = false
            }
            "--beta-iron" => {
                config.diagnostics.solve.beta_iron = next_arg(&mut args, "--beta-iron")?.parse()?
            }
            "--material-kind" => {
                config.diagnostics.solve.material_kind =
                    parse_material_kind(&next_arg(&mut args, "--material-kind")?)?
            }
            "--ngsolve-tabulated-material" => {
                config.diagnostics.solve.material_kind =
                    Team13NonlinearMaterialKind::NgsolveTabulatedLinear
            }
            "--smooth-material" => {
                config.diagnostics.solve.material_kind =
                    Team13NonlinearMaterialKind::SmoothQuadratic
            }
            "--b-scale" => {
                config.diagnostics.solve.b_scale_tesla =
                    next_arg(&mut args, "--b-scale")?.parse()?
            }
            "--ampere-turns" => {
                config.diagnostics.solve.ampere_turns =
                    next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--prior-kappa" => {
                config.diagnostics.solve.prior_kappa =
                    Some(next_arg(&mut args, "--prior-kappa")?.parse()?)
            }
            "--prior-tau" => {
                config.diagnostics.solve.prior_tau = next_arg(&mut args, "--prior-tau")?.parse()?
            }
            "--field-prior-scale" => {
                config.diagnostics.solve.field_prior_precision_scale =
                    next_arg(&mut args, "--field-prior-scale")?.parse()?
            }
            "--pde-variances" | "--pde-variance-values" => {
                config.diagnostics.pde_variance_values =
                    parse_csv_f64(&next_arg(&mut args, "--pde-variances")?)?
            }
            "--direct-cholesky-steps" => {
                config.diagnostics.solve.linear_solve = GaussNewtonLinearSolve::DirectCholesky
            }
            "--undamped-steps" => {
                config.diagnostics.solve.step_regularization = GaussNewtonStepRegularization::None
            }
            "--lm-grid-steps" => {
                config.diagnostics.solve.step_regularization =
                    GaussNewtonStepRegularization::LevenbergMarquardtGrid
            }
            "--cg-tolerance" => {
                let tolerance = next_arg(&mut args, "--cg-tolerance")?.parse()?;
                config.diagnostics.solve.linear_solve = GaussNewtonLinearSolve::IterativeCg {
                    tolerance,
                    max_iterations: 2048,
                    warm_start: true,
                };
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    Ok(config)
}

fn parse_csv_f64(values: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    values
        .split(',')
        .map(|value| value.trim().parse::<f64>().map_err(Into::into))
        .collect()
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
        "Usage: cargo run --release -p feg-case-studies --example team13_nonlinear_diagnostics -- [options]\n\
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
           --prior-kappa VALUE         Prior kappa, default 1 for unweighted Hodge prior\n\
           --prior-tau VALUE           Prior tau, default 1\n\
           --field-prior-scale VALUE   Prior precision scale, default 1e-12\n\
           --pde-variances CSV         Variance sweep, default includes 1e-4..10 plus calibrated 1.2e4\n\
           --pde-variance-values CSV   Alias for --pde-variances\n\
          --assimilate-measurements   Include the 25 NGSolve-style TEAM13 surface-averaged signed B-component magnitudes in diagnostics\n\
          --no-assimilate-measurements Keep benchmark surface B rows out of diagnostic objectives\n\
           --skip-output               Accepted for parity; diagnostics write no files\n\
           --direct-cholesky-steps     Use direct intermediate Gauss-Newton diagnostic solve\n\
           --undamped-steps            Set solve config step regularization to None\n\
           --lm-grid-steps             Set solve config step regularization to LM grid"
    );
}
