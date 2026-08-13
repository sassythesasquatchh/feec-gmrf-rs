use feg_case_studies::team13::{
    run_team13_map_parity, Team13DomainMode, Team13MapParityConfig, Team13MapParityPdeResidualKind,
    Team13MapParityPriorKind, Team13MapParityRunResult, Team13NonlinearMaterialKind,
    Team13PublishedSteelBenchmarkReport, Team13PublishedSteelGap,
    Team13SteelObservationQuadratureMode,
};
use feg_infer::nonlinear::{GaussNewtonStepRegularization, NonlinearAssemblyTermKind};
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
    solve: Team13MapParityConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 18,
            force_mesh_generation: false,
            solve: Team13MapParityConfig::default(),
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

    let result = run_team13_map_parity(&config.solve)?;

    println!("TEAM 13 internal-reference nonlinear MAP parity");
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
        "  prior={} pde_residual={} step_regularization={} steel_observation={} kappa={:.6e} tau={:.6e} diag_shift={:.6e} smoothing={:.6e}",
        result.prior_kind.as_str(),
        result.pde_residual_kind.as_str(),
        result.step_regularization.as_str(),
        result.steel_observation_quadrature.as_str(),
        result.prior_kappa,
        result.prior_tau,
        result.prior_diagonal_shift,
        result.magnitude_smoothing_tesla
    );
    println!("  cholesky_ordering=metis-default (AMD fallback if ndmetis is unavailable)");
    println!(
        "  truth: converged={} cache_hit={} residual={:.6e}->{:.6e}",
        result.truth_converged,
        result.truth_cache_hit,
        result.initial_residual_norm,
        result.truth_residual_norm
    );
    print_run("default", &result.default_run);
    if !result.sweep_runs.is_empty() {
        println!("weighting sweep");
        for run in &result.sweep_runs {
            print_run("sweep", run);
        }
    }
    if let Some(output_dir) = &result.output_dir {
        println!("outputs: {}", output_dir.display());
    }

    Ok(())
}

fn print_run(label: &str, run: &Team13MapParityRunResult) {
    println!(
        "{label}: pde_variance={:.6e} observation_std_tesla={:.6e} residual_rows={} steel={} converged={}",
        run.pde_variance,
        run.observation_std_tesla,
        run.total_residual_rows,
        run.steel_observation_count,
        run.posterior_converged
    );
    println!(
        "  A relative error: initial={:.6e} posterior={:.6e} ratio={:.6e}",
        run.initial_relative_error,
        run.posterior_relative_error,
        safe_ratio(run.posterior_relative_error, run.initial_relative_error)
    );
    println!(
        "  internal steel RMSE: initial={:.6e} posterior={:.6e} relative={:.6e} max_abs={:.6e} ratio={:.6e}",
        run.initial_steel_rmse,
        run.posterior_steel_rmse,
        run.posterior_steel_relative_rmse,
        run.posterior_steel_max_abs_residual,
        run.steel_rmse_improvement_ratio
    );
    println!(
        "  likelihood PDE residual: initial={:.6e} truth={:.6e} posterior={:.6e}",
        run.initial_residual_norm, run.truth_residual_norm, run.posterior_residual_norm
    );
    println!(
        "  B variances: finite={} nonnegative={} exact_steel={} (latent_A_count={} finite={} nonnegative={})",
        run.b_quantity_variances_finite,
        run.b_quantity_variances_nonnegative,
        run.internal_steel_variances.len(),
        run.latent_variance_count,
        run.latent_variances_finite,
        run.latent_variances_nonnegative
    );
    println!(
        "  assembly: prior_nnz={} residual_terms_operator_nnz={} residual_terms_update_nnz={} posterior_nnz={} factor_nnz={} fill_vs_lower={:.6e}",
        run.assembly.prior_precision_nnz,
        run.assembly
            .term_operator_nnz(NonlinearAssemblyTermKind::Residual),
        run.assembly
            .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual),
        run.assembly.posterior_precision_nnz,
        run.assembly.factor_nnz.unwrap_or(run.final_factorization.nnz),
        run.assembly.fill_ratio_vs_lower_triangle.unwrap_or(f64::NAN)
    );
    println!(
        "  solve diagnostics: step_solve_attempts={} accepted_iterations={} line_search_evals={} final_factorizations={} metis_hits={} metis_misses={}",
        run.diagnostics.step_solve_attempts,
        run.diagnostics.accepted_iterations,
        run.diagnostics.line_search_residual_evaluations,
        run.diagnostics.final_factorizations,
        run.diagnostics.metis_cache_hits,
        run.diagnostics.metis_cache_misses
    );
    println!(
        "  Cholesky stabilization: attempts={} successes={} unshifted={} sym={} cached_shift={}/{} shifted={}/{} max_shift={:.3e} factor_time={:.3}s",
        run.diagnostics.cholesky_factor_attempts,
        run.diagnostics.cholesky_factor_successes,
        run.diagnostics.cholesky_unshifted_attempts,
        run.diagnostics.cholesky_symmetrized_attempts,
        run.diagnostics.cholesky_cached_shift_successes,
        run.diagnostics.cholesky_cached_shift_attempts,
        run.diagnostics.cholesky_shifted_successes,
        run.diagnostics.cholesky_shifted_attempts,
        run.diagnostics.cholesky_max_shift,
        run.diagnostics.cholesky_factorization_seconds
    );
    println!(
        "  published steel reporting-only: g052 initial={:.6e} posterior={:.6e}; g047 initial={:.6e} posterior={:.6e}",
        benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G052, true),
        benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G052, false),
        benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G047, true),
        benchmark_rmse(&run.published_steel_benchmark_reports, Team13PublishedSteelGap::G047, false)
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
    println!();
}

fn parse_args() -> Result<CliConfig, Box<dyn Error>> {
    let mut config = CliConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = Some(PathBuf::from(next_arg(&mut args, "--mesh")?)),
            "--geo" => config.geo_path = PathBuf::from(next_arg(&mut args, "--geo")?),
            "--mesh-scale" => config.mesh_scale = next_arg(&mut args, "--mesh-scale")?.parse()?,
            "--force-mesh" => config.force_mesh_generation = true,
            "--domain" => {
                config.solve.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?
            }
            "--ampere-turns" => {
                config.solve.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--material" => {
                config.solve.material_kind =
                    parse_material_kind(&next_arg(&mut args, "--material")?)?;
            }
            "--prior-kind" => {
                config.solve.prior_kind =
                    parse_map_parity_prior_kind(&next_arg(&mut args, "--prior-kind")?)?;
            }
            "--pde-residual" => {
                config.solve.pde_residual_kind =
                    parse_map_parity_pde_residual_kind(&next_arg(&mut args, "--pde-residual")?)?;
            }
            "--steel-observation" => {
                config.solve.steel_observation_quadrature =
                    parse_steel_observation_mode(&next_arg(&mut args, "--steel-observation")?)?;
            }
            "--step-regularization" => {
                config.solve.step_regularization =
                    parse_step_regularization(&next_arg(&mut args, "--step-regularization")?)?;
            }
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
            "--pde-variance-values" => {
                config.solve.sweep_pde_variances =
                    parse_csv_f64(&next_arg(&mut args, "--pde-variance-values")?)?;
            }
            "--observation-std-values" => {
                config.solve.sweep_observation_std_tesla =
                    parse_csv_f64(&next_arg(&mut args, "--observation-std-values")?)?;
            }
            "--skip-sweep" => {
                config.solve.sweep_pde_variances.clear();
                config.solve.sweep_observation_std_tesla.clear();
            }
            "--force-truth-solve" => config.solve.force_truth_solve = true,
            "--latent-variance" => config.solve.estimate_latent_variance = true,
            "--skip-latent-variance" => config.solve.estimate_latent_variance = false,
            "--magnitude-smoothing-tesla" => {
                config.solve.magnitude_smoothing_tesla =
                    next_arg(&mut args, "--magnitude-smoothing-tesla")?.parse()?;
            }
            "--output-dir" => {
                config.solve.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?));
            }
            "--skip-output" => config.solve.output_dir = None,
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

fn parse_domain(raw: &str) -> Result<Team13DomainMode, Box<dyn Error>> {
    match raw {
        "half" | "half-z" => Ok(Team13DomainMode::HalfZNonnegative),
        "full" => Ok(Team13DomainMode::Full),
        other => Err(format!("unknown domain `{other}`; expected half or full").into()),
    }
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

fn parse_map_parity_prior_kind(raw: &str) -> Result<Team13MapParityPriorKind, Box<dyn Error>> {
    raw.parse::<Team13MapParityPriorKind>()
        .map_err(|err| err.into())
}

fn parse_map_parity_pde_residual_kind(
    raw: &str,
) -> Result<Team13MapParityPdeResidualKind, Box<dyn Error>> {
    raw.parse::<Team13MapParityPdeResidualKind>()
        .map_err(|err| err.into())
}

fn parse_steel_observation_mode(
    raw: &str,
) -> Result<Team13SteelObservationQuadratureMode, Box<dyn Error>> {
    raw.parse::<Team13SteelObservationQuadratureMode>()
        .map_err(|err| err.into())
}

fn parse_step_regularization(raw: &str) -> Result<GaussNewtonStepRegularization, Box<dyn Error>> {
    raw.parse::<GaussNewtonStepRegularization>()
        .map_err(|err| err.into())
}

fn workspace_path(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(env::current_dir()?.join(path))
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
        return f64::NAN;
    }
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

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_map_parity -- [options]\n\
         Options:\n\
           --mesh <path>                         Mesh path to use or generate\n\
           --geo <path>                          Gmsh geometry path (default: geometries/team13_linear.geo)\n\
           --mesh-scale <usize>                  TEAM 13 gmsh MeshScale value (default: 18)\n\
           --force-mesh                          Regenerate the mesh even if it exists\n\
           --domain <half|full>\n\
           --ampere-turns <float>\n\
           --material <kind>                     tabulated or smooth (default: tabulated)\n\
           --prior-kind <kind>                   exact-potential, ordinary-matern-alpha2, or weak-ridge\n\
          --pde-residual <kind>                 ungauged-curl (default) or gauge-fixed\n\
           --steel-observation <mode>            ngsolve-style or face-cochain\n\
           --step-regularization <mode>          none, lm-grid, or adaptive-lm (default)\n\
           --max-iterations <usize>              Posterior Gauss-Newton iterations\n\
           --truth-max-iterations <usize>        Truth solve Newton iterations\n\
           --pde-variance <float>                PDE weak residual variance\n\
           --observation-std-tesla <float>       Internal steel magnitude observation std dev\n\
           --pde-variance-values <csv>           PDE variance sweep values\n\
           --observation-std-values <csv>        Observation std sweep values\n\
           --skip-sweep                          Run only the default weighting\n\
           --force-truth-solve                   Ignore cached internal truth solution\n\
           --latent-variance                     Also compute latent A marginal variances\n\
           --skip-latent-variance                Skip latent A variances (default)\n\
           --magnitude-smoothing-tesla <float>   Smooth magnitude epsilon\n\
           --output-dir <path>\n\
           --skip-output\n\
           --help"
    );
}
