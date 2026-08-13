use feg_case_studies::team13::{
    run_team13_synthetic_benchmark_geometry, Team13DomainMode, Team13NonlinearMaterialKind,
    Team13PublishedSteelBenchmarkReport, Team13PublishedSteelGap,
    Team13SyntheticBenchmarkGeometryConfig, Team13SyntheticBenchmarkGeometryRunResult,
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
    solve: Team13SyntheticBenchmarkGeometryConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 18,
            force_mesh_generation: false,
            solve: Team13SyntheticBenchmarkGeometryConfig::default(),
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

    let result = run_team13_synthetic_benchmark_geometry(&config.solve)?;

    println!("TEAM 13 synthetic benchmark-geometry nonlinear baseline");
    println!("  mesh: {}", mesh_path.display());
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
        "  observations={} assimilated={} steel={} air={} steel_quadrature={}",
        result.observation_count,
        result.assimilated_observation_count,
        result.steel_observation_count,
        result.air_observation_count,
        result.steel_observation_quadrature.as_str()
    );
    println!(
        "  prior_kappa={:.6e} prior_tau={:.6e} diag_shift={:.6e} smoothing={:.6e}",
        result.prior_kappa,
        result.prior_tau,
        result.prior_diagonal_shift,
        result.magnitude_smoothing_tesla
    );
    println!(
        "  truth: converged={} initial_residual={:.6e} truth_residual={:.6e}",
        result.truth_converged, result.initial_residual_norm, result.truth_residual_norm
    );
    println!();

    print_run("default", &result.default_run);
    if !result.sweep_runs.is_empty() {
        println!("weighting sweep");
        for run in &result.sweep_runs {
            print_run("sweep", run);
        }
    }
    if !result.source_scale_diagnostics.is_empty() {
        println!("source-scale forward diagnostic sweep");
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
    }

    Ok(())
}

fn print_run(label: &str, run: &Team13SyntheticBenchmarkGeometryRunResult) {
    println!(
        "{label}: prior=exact-potential pde_variance={:.6e} observation_std_tesla={:.6e} residual_rows={} observations={}/{} converged={}",
        run.pde_variance,
        run.observation_std_tesla,
        run.total_residual_rows,
        run.assimilated_observation_count,
        run.observation_count,
        run.posterior_converged
    );
    println!(
        "  A relative error: initial={:.6e} posterior={:.6e} ratio={:.6e}",
        run.initial_relative_error,
        run.posterior_relative_error,
        safe_ratio(run.posterior_relative_error, run.initial_relative_error)
    );
    println!(
        "  sensor RMSE: initial={:.6e} posterior={:.6e} relative={:.6e} max_abs={:.6e} ratio={:.6e}",
        run.initial_sensor_rmse,
        run.posterior_sensor_rmse,
        run.posterior_sensor_relative_rmse,
        run.posterior_sensor_max_abs_residual,
        run.sensor_rmse_improvement_ratio
    );
    println!(
        "  full PDE residual: initial={:.6e} truth={:.6e} posterior={:.6e}",
        run.initial_residual_norm, run.truth_residual_norm, run.posterior_residual_norm
    );
    println!(
        "  variances: finite={} nonnegative={} transformed={}",
        run.all_finite_variances,
        run.nonnegative_variances,
        run.observation_variances.len()
    );
    println!(
        "  prior variance: min={:.6e} max={:.6e} ratio={:.6e} finite={} nonnegative={} factor_nnz={}",
        run.prior_variance_diagnostics.min_variance,
        run.prior_variance_diagnostics.max_variance,
        run.prior_variance_diagnostics.max_to_min_variance_ratio,
        run.prior_variance_diagnostics.all_finite,
        run.prior_variance_diagnostics.nonnegative,
        run.prior_variance_diagnostics.factor_nnz
    );
    println!(
        "  assembly: prior_nnz={} residual_jacobian_nnz={} residual_update_nnz={} posterior_nnz={} factor_nnz={} fill_vs_lower={:.6e}",
        run.assembly.prior_precision_nnz,
        run.assembly
            .term_operator_nnz(NonlinearAssemblyTermKind::Residual),
        run.assembly
            .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual),
        run.assembly.posterior_precision_nnz,
        run.assembly.factor_nnz.unwrap_or(run.final_factorization.nnz),
        run.assembly.fill_ratio_vs_lower_triangle.unwrap_or(f64::NAN)
    );
    for summary in &run.group_summaries {
        println!(
            "  group {}: count={} initial_rmse={:.6e} posterior_rmse={:.6e} relative={:.6e} max_abs={:.6e}",
            summary.group.as_str(),
            summary.count,
            summary.initial_rmse,
            summary.posterior_rmse,
            summary.posterior_relative_rmse,
            summary.posterior_max_abs_residual
        );
    }
    if !run.published_steel_benchmark_reports.is_empty() {
        println!(
            "  published steel benchmark RMSE (reporting-only): g052 nominal={:.6e} posterior={:.6e}; g047 nominal={:.6e} posterior={:.6e}",
            benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G052, true),
            benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G052, false),
            benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G047, true),
            benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G047, false)
        );
    }
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
    println!();
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
            "--prior-modes" => {
                validate_exact_prior_modes(&next_arg(&mut args, "--prior-modes")?)?;
            }
            "--pde-variance-values" => {
                config.solve.sweep_pde_variances =
                    parse_csv_f64(&next_arg(&mut args, "--pde-variance-values")?)?;
            }
            "--observation-std-values" => {
                config.solve.sweep_observation_std_tesla =
                    parse_csv_f64(&next_arg(&mut args, "--observation-std-values")?)?;
            }
            "--source-scale-values" => {
                config.solve.source_scale_diagnostic_values =
                    parse_csv_f64(&next_arg(&mut args, "--source-scale-values")?)?;
            }
            "--skip-sweep" => {
                config.solve.sweep_pde_variances.clear();
                config.solve.sweep_observation_std_tesla.clear();
            }
            "--skip-source-scale-sweep" => {
                config.solve.source_scale_diagnostic_values.clear();
            }
            "--magnitude-smoothing-tesla" => {
                config.solve.magnitude_smoothing_tesla =
                    next_arg(&mut args, "--magnitude-smoothing-tesla")?.parse()?;
            }
            "--material" => {
                config.solve.material_kind =
                    parse_material_kind(&next_arg(&mut args, "--material")?)?;
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

fn parse_csv_f64(raw: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    raw.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| Ok(part.trim().parse()?))
        .collect()
}

fn parse_material_kind(raw: &str) -> Result<Team13NonlinearMaterialKind, Box<dyn Error>> {
    match raw {
        "ngsolve-tabulated-linear" | "tabulated" => {
            Ok(Team13NonlinearMaterialKind::NgsolveTabulatedLinear)
        }
        "smooth-quadratic" | "smooth" => Ok(Team13NonlinearMaterialKind::SmoothQuadratic),
        other => Err(format!(
            "unknown material `{other}`; expected ngsolve-tabulated-linear or smooth-quadratic"
        )
        .into()),
    }
}

fn validate_exact_prior_modes(raw: &str) -> Result<(), Box<dyn Error>> {
    let modes = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(str::trim)
        .collect::<Vec<_>>();
    if modes.is_empty() {
        return Err("--prior-modes requires exact-potential".into());
    }
    if modes
        .iter()
        .all(|mode| matches!(*mode, "exact-potential" | "exact"))
    {
        return Ok(());
    }
    Err("TEAM 13 synthetic benchmark geometry now supports only exact-potential prior mode".into())
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

fn benchmark_rmse(
    reports: &[Team13PublishedSteelBenchmarkReport],
    gap: Team13PublishedSteelGap,
    use_nominal: bool,
) -> f64 {
    if reports.is_empty() {
        f64::NAN
    } else {
        (reports
            .iter()
            .map(|report| {
                let observed = match gap {
                    Team13PublishedSteelGap::G052 => report.observed_g_052,
                    Team13PublishedSteelGap::G047 => report.observed_g_047,
                };
                let prediction = if use_nominal {
                    report.nominal_prediction
                } else {
                    report.posterior_prediction
                };
                let residual = prediction - observed;
                residual * residual
            })
            .sum::<f64>()
            / reports.len() as f64)
            .sqrt()
    }
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_synthetic_benchmark_geometry -- [options]\n\
         Options:\n\
           --mesh <path>                         Mesh path to use or generate\n\
           --geo <path>                          Gmsh geometry path (default: geometries/team13_linear.geo)\n\
           --mesh-scale <usize>                  TEAM 13 gmsh MeshScale value (default: 18)\n\
           --force-mesh                          Regenerate the mesh even if it exists\n\
           --material <kind>                     tabulated or smooth (default: tabulated)\n\
           --max-iterations <usize>              Posterior Gauss-Newton iterations\n\
           --truth-max-iterations <usize>        Truth solve Gauss-Newton iterations\n\
           --pde-variance <float>                PDE weak residual variance\n\
           --observation-std-tesla <float>       Synthetic magnitude observation std dev\n\
           --prior-modes <csv>                   Compatibility no-op; only exact-potential is accepted\n\
           --pde-variance-values <csv>           PDE variance sweep values\n\
           --observation-std-values <csv>        Observation std sweep values\n\
           --skip-sweep                          Run only the default weighting\n\
           --source-scale-values <csv>           Source-scale forward diagnostic values\n\
           --skip-source-scale-sweep             Disable source-scale diagnostics\n\
           --magnitude-smoothing-tesla <float>   Smooth magnitude epsilon\n\
           --help                                Print this help"
    );
}
