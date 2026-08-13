use feg_case_studies::team13::{
    run_team13_material_gap_uq, team13_material_log_scale_sigma_points, Team13DomainMode,
    Team13MapParityPdeResidualKind, Team13MapParityPriorKind, Team13MaterialGapMeshCase,
    Team13MaterialGapUqConfig, Team13NonlinearMaterialKind, Team13OperatorUncertaintyTangentKind,
    Team13PdeResidualWeighting, Team13PublishedSteelGap, Team13SteelObservationQuadratureMode,
};
use feg_infer::linear_pde::LinearPdeVarianceMode;
use feg_infer::nonlinear::GaussNewtonLinearSolve;
use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() -> Result<(), Box<dyn Error>> {
    let parsed = parse_args()?;
    if parsed.generate_meshes {
        ensure_gap_meshes(&parsed)?;
    }
    let result = run_team13_material_gap_uq(&parsed.config)?;
    println!("TEAM 13 material/gap uncertainty experiment");
    println!("  cases: {}", result.case_results.len());
    println!(
        "  patch decompositions: {}",
        result.variance_decomposition.len()
    );
    if let Some(output_dir) = &result.output_dir {
        println!("  wrote outputs: {}", output_dir.display());
    }
    Ok(())
}

#[derive(Debug)]
struct ParsedArgs {
    config: Team13MaterialGapUqConfig,
    geo_path: PathBuf,
    mesh_scale: f64,
    generate_meshes: bool,
    force_mesh: bool,
}

fn parse_args() -> Result<ParsedArgs, Box<dyn Error>> {
    let mut config = Team13MaterialGapUqConfig::default();
    let mut geo_path = PathBuf::from("geometries/team13_linear_measurement_planes.geo");
    let mut mesh_dir = PathBuf::from("target/team13_material_gap_uq/meshes");
    let mut mesh_scale = 1.0;
    let mut generate_meshes = true;
    let mut force_mesh = false;
    let mut gaps_mm = vec![0.47, 0.52];

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--geo" => geo_path = PathBuf::from(next_arg(&mut args, "--geo")?),
            "--mesh-dir" => mesh_dir = PathBuf::from(next_arg(&mut args, "--mesh-dir")?),
            "--gaps-mm" => gaps_mm = parse_float_list(&next_arg(&mut args, "--gaps-mm")?)?,
            "--mesh-scale" => mesh_scale = next_arg(&mut args, "--mesh-scale")?.parse()?,
            "--force-mesh" => force_mesh = true,
            "--no-generate-mesh" => generate_meshes = false,
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?))
            }
            "--no-output" => config.output_dir = None,
            "--domain" => {
                config.operator.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?
            }
            "--ampere-turns" => {
                config.operator.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--material" => {
                config.operator.material_kind =
                    parse_material_kind(&next_arg(&mut args, "--material")?)?
            }
            "--material-log-std" => {
                let std = next_arg(&mut args, "--material-log-std")?.parse()?;
                config.material_nodes = team13_material_log_scale_sigma_points(std)?;
            }
            "--beta-iron" => {
                config.operator.beta_iron = next_arg(&mut args, "--beta-iron")?.parse()?
            }
            "--b-scale-tesla" => {
                config.operator.b_scale_tesla = next_arg(&mut args, "--b-scale-tesla")?.parse()?
            }
            "--tangent" => {
                config.operator.tangent_kind = next_arg(&mut args, "--tangent")?
                    .parse::<Team13OperatorUncertaintyTangentKind>()?
            }
            "--prior-kind" => {
                config.operator.prior_kind =
                    next_arg(&mut args, "--prior-kind")?.parse::<Team13MapParityPriorKind>()?
            }
            "--prior-kappa" => {
                config.operator.prior_kappa = next_arg(&mut args, "--prior-kappa")?.parse()?
            }
            "--prior-tau" => {
                config.operator.prior_tau = next_arg(&mut args, "--prior-tau")?.parse()?
            }
            "--prior-diagonal-shift" => {
                config.operator.prior_diagonal_shift =
                    next_arg(&mut args, "--prior-diagonal-shift")?.parse()?
            }
            "--pde-residual" => {
                config.operator.pde_residual_kind = next_arg(&mut args, "--pde-residual")?
                    .parse::<Team13MapParityPdeResidualKind>()?
            }
            "--pde-weighting" => {
                config.operator.pde_residual_weighting =
                    parse_pde_weighting(&next_arg(&mut args, "--pde-weighting")?)?
            }
            "--pde-variance" => {
                config.operator.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?
            }
            "--steel-observation" => {
                let value = next_arg(&mut args, "--steel-observation")?;
                if value == "none" || value == "off" {
                    config.operator.include_steel_observations = false;
                } else {
                    config.operator.include_steel_observations = true;
                    config.operator.steel_observation_quadrature =
                        value.parse::<Team13SteelObservationQuadratureMode>()?;
                }
            }
            "--steel-observation-mode" => {
                config.operator.steel_observation_quadrature =
                    next_arg(&mut args, "--steel-observation-mode")?
                        .parse::<Team13SteelObservationQuadratureMode>()?
            }
            "--include-steel-observations" => config.operator.include_steel_observations = true,
            "--observation-std-tesla" => {
                config.operator.observation_std_tesla =
                    next_arg(&mut args, "--observation-std-tesla")?.parse()?
            }
            "--field-variance-mode" => {
                let value = next_arg(&mut args, "--field-variance-mode")?;
                if value == "none" || value == "off" {
                    config.operator.estimate_field_variance = false;
                } else {
                    config.operator.estimate_field_variance = true;
                    config.operator.field_variance.mode = parse_variance_mode(&value)?;
                }
            }
            "--field-variance-probes" => {
                config.operator.field_variance.num_variance_probes =
                    next_arg(&mut args, "--field-variance-probes")?.parse()?
            }
            "--field-variance-batches" => {
                config.operator.field_variance.variance_batch_count =
                    next_arg(&mut args, "--field-variance-batches")?.parse()?
            }
            "--rng-seed" => {
                config.operator.field_variance.rng_seed =
                    next_arg(&mut args, "--rng-seed")?.parse()?
            }
            "--truth-max-iterations" => {
                config.operator.truth_max_iterations =
                    next_arg(&mut args, "--truth-max-iterations")?.parse()?
            }
            "--linear-solve" => {
                config.operator.linear_solve =
                    parse_linear_solve(&next_arg(&mut args, "--linear-solve")?)?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }

    config.gap_cases = build_gap_cases(&mesh_dir, &gaps_mm)?;
    Ok(ParsedArgs {
        config,
        geo_path,
        mesh_scale,
        generate_meshes,
        force_mesh,
    })
}

fn build_gap_cases(
    mesh_dir: &Path,
    gaps_mm: &[f64],
) -> Result<Vec<Team13MaterialGapMeshCase>, Box<dyn Error>> {
    let mut cases = Vec::with_capacity(gaps_mm.len());
    for &gap_mm in gaps_mm {
        if !gap_mm.is_finite() || gap_mm <= 0.0 {
            return Err(format!("gap must be finite and positive, got {gap_mm}").into());
        }
        let observed_gap = published_gap_for_mm(gap_mm);
        let label = observed_gap
            .map(|gap| gap.token().to_string())
            .unwrap_or_else(|| format!("g{}mm", format!("{gap_mm:.3}").replace('.', "p")));
        cases.push(Team13MaterialGapMeshCase {
            label: label.clone(),
            steel_gap_m: gap_mm * 1.0e-3,
            mesh_path: mesh_dir.join(format!("team13_half_{label}.msh")),
            observed_gap,
            weight: 1.0,
        });
    }
    Ok(cases)
}

fn ensure_gap_meshes(parsed: &ParsedArgs) -> Result<(), Box<dyn Error>> {
    for gap_case in &parsed.config.gap_cases {
        if gap_case.mesh_path.exists() && !parsed.force_mesh {
            continue;
        }
        if let Some(parent) = gap_case.mesh_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let full_domain = match parsed.config.operator.domain_mode {
            Team13DomainMode::HalfZNonnegative => "0",
            Team13DomainMode::Full => "1",
        };
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&parsed.geo_path)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg(full_domain)
            .arg("-setnumber")
            .arg("MeshScale")
            .arg(parsed.mesh_scale.to_string())
            .arg("-setnumber")
            .arg("SteelGap")
            .arg(gap_case.steel_gap_m.to_string())
            .arg("-o")
            .arg(&gap_case.mesh_path)
            .status()?;
        if !status.success() {
            return Err(format!(
                "gmsh failed for gap case `{}` with status {status}",
                gap_case.label
            )
            .into());
        }
    }
    Ok(())
}

fn published_gap_for_mm(gap_mm: f64) -> Option<Team13PublishedSteelGap> {
    if (gap_mm - 0.47).abs() <= 1.0e-9 {
        Some(Team13PublishedSteelGap::G047)
    } else if (gap_mm - 0.52).abs() <= 1.0e-9 {
        Some(Team13PublishedSteelGap::G052)
    } else {
        None
    }
}

fn parse_float_list(raw: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = raw
        .split(',')
        .map(|value| value.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        Err("expected at least one comma-separated value".into())
    } else {
        Ok(values)
    }
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

fn parse_pde_weighting(raw: &str) -> Result<Team13PdeResidualWeighting, Box<dyn Error>> {
    match raw {
        "euclidean" => Ok(Team13PdeResidualWeighting::Euclidean),
        "mass-inverse" | "mass" | "mass-weighted" => Ok(Team13PdeResidualWeighting::MassInverse),
        "mass-inverse-trace-normalized" | "mass-trace-normalized" | "mass-normalized" => {
            Ok(Team13PdeResidualWeighting::MassInverseTraceNormalized)
        }
        other => Err(format!("unknown PDE weighting `{other}`").into()),
    }
}

fn parse_variance_mode(raw: &str) -> Result<LinearPdeVarianceMode, Box<dyn Error>> {
    match raw {
        "exact" => Ok(LinearPdeVarianceMode::Exact),
        "exact-solves" => Ok(LinearPdeVarianceMode::ExactSolves),
        "selected-inverse" => Ok(LinearPdeVarianceMode::SelectedInverse),
        "hutchinson" => Ok(LinearPdeVarianceMode::Hutchinson),
        "local-rbmc" => Ok(LinearPdeVarianceMode::LocalRbmc),
        "monte-carlo" => Ok(LinearPdeVarianceMode::MonteCarlo),
        other => Err(format!("unknown variance mode `{other}`").into()),
    }
}

fn parse_linear_solve(raw: &str) -> Result<GaussNewtonLinearSolve, Box<dyn Error>> {
    match raw {
        "direct-cholesky" | "direct" | "cholesky" => Ok(GaussNewtonLinearSolve::DirectCholesky),
        "iterative-cg" | "cg" => Ok(GaussNewtonLinearSolve::IterativeCg {
            tolerance: 1.0e-8,
            max_iterations: 4096,
            warm_start: false,
        }),
        other => Err(format!("unknown linear solve `{other}`").into()),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_material_gap_uq -- [options]\n\
         Options:\n\
           --geo <path>\n\
           --mesh-dir <path>\n\
           --gaps-mm <comma-list>\n\
           --mesh-scale <float>\n\
           --force-mesh\n\
           --no-generate-mesh\n\
           --output-dir <path>\n\
           --domain <half|full>\n\
           --material <ngsolve-tabulated-linear|smooth-quadratic>\n\
           --material-log-std <float>\n\
           --tangent <nonlinear|linear-beta-zero>\n\
           --prior-kind <weak-ridge|ordinary-matern-alpha2|exact-potential>\n\
           --pde-residual <ungauged-curl|gauge-fixed>\n\
           --pde-weighting <euclidean|mass-inverse|mass-inverse-trace-normalized>\n\
           --pde-variance <float>\n\
           --steel-observation <none|face-cochain|ngsolve-style>\n\
           --field-variance-mode <hutchinson|selected-inverse|exact|none>\n\
           --linear-solve <direct-cholesky|iterative-cg>"
    );
}
