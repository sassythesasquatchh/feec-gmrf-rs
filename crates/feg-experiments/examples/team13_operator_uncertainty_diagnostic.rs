use feg_case_studies::team13::{
    run_team13_operator_uncertainty_diagnostic, Team13DomainMode, Team13MapParityPdeResidualKind,
    Team13MapParityPriorKind, Team13NonlinearMaterialKind, Team13OperatorUncertaintyConfig,
    Team13OperatorUncertaintyTangentKind, Team13PdeResidualWeighting,
    Team13SteelObservationQuadratureMode,
};
use feg_infer::linear_pde::LinearPdeVarianceMode;
use feg_infer::nonlinear::GaussNewtonLinearSolve;
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let result = run_team13_operator_uncertainty_diagnostic(&config)?;
    println!("TEAM 13 operator-induced uncertainty diagnostic");
    println!("  mesh: {}", result.mesh_path.display());
    println!(
        "  tangent={} prior={} residual={} weighting={} pde_variance={:.6e} material_log_iron_nu_scale={:.6e}",
        result.tangent_kind.as_str(),
        result.prior_kind.as_str(),
        result.pde_residual_kind.as_str(),
        result.pde_residual_weighting.as_str(),
        result.pde_variance,
        result.material_log_iron_nu_scale
    );
    println!(
        "  posterior: converged={} precision_nnz={} factor_nnz={} fill_ratio={:.3}",
        result.posterior_converged,
        result.posterior_precision_nnz,
        result.posterior_factor_nnz,
        result.fill_ratio_vs_lower
    );
    println!(
        "  field variance: {} available={}",
        result.field_variance_estimator, result.field_variance_available
    );
    if let Some(summary) = result
        .region_summaries
        .iter()
        .find(|summary| summary.region == "iron_air_interface_band")
    {
        println!(
            "  interface variance ratio: vs iron bulk {:.6e}, vs air bulk {:.6e}",
            summary.variance_ratio_to_iron_bulk, summary.variance_ratio_to_air_bulk
        );
    }
    if let Some(output_dir) = &result.output_dir {
        println!("  wrote outputs: {}", output_dir.display());
    }
    Ok(())
}

fn parse_args() -> Result<Team13OperatorUncertaintyConfig, Box<dyn Error>> {
    let mut config = Team13OperatorUncertaintyConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--domain" => config.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?,
            "--ampere-turns" => {
                config.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--material" => {
                config.material_kind = parse_material_kind(&next_arg(&mut args, "--material")?)?
            }
            "--beta-iron" => config.beta_iron = next_arg(&mut args, "--beta-iron")?.parse()?,
            "--b-scale-tesla" => {
                config.b_scale_tesla = next_arg(&mut args, "--b-scale-tesla")?.parse()?
            }
            "--material-log-iron-nu-scale" => {
                config.material_log_iron_nu_scale =
                    next_arg(&mut args, "--material-log-iron-nu-scale")?.parse()?
            }
            "--tangent" => {
                config.tangent_kind = next_arg(&mut args, "--tangent")?
                    .parse::<Team13OperatorUncertaintyTangentKind>()?
            }
            "--prior-kind" => {
                config.prior_kind =
                    next_arg(&mut args, "--prior-kind")?.parse::<Team13MapParityPriorKind>()?
            }
            "--prior-kappa" => {
                config.prior_kappa = next_arg(&mut args, "--prior-kappa")?.parse()?
            }
            "--prior-tau" => config.prior_tau = next_arg(&mut args, "--prior-tau")?.parse()?,
            "--prior-diagonal-shift" => {
                config.prior_diagonal_shift =
                    next_arg(&mut args, "--prior-diagonal-shift")?.parse()?
            }
            "--pde-residual" => {
                config.pde_residual_kind = next_arg(&mut args, "--pde-residual")?
                    .parse::<Team13MapParityPdeResidualKind>()?
            }
            "--pde-weighting" => {
                config.pde_residual_weighting =
                    parse_pde_weighting(&next_arg(&mut args, "--pde-weighting")?)?
            }
            "--pde-variance" => {
                config.pde_variance = next_arg(&mut args, "--pde-variance")?.parse()?
            }
            "--steel-observation" => {
                let value = next_arg(&mut args, "--steel-observation")?;
                if value == "none" || value == "off" {
                    config.include_steel_observations = false;
                } else {
                    config.include_steel_observations = true;
                    config.steel_observation_quadrature =
                        value.parse::<Team13SteelObservationQuadratureMode>()?;
                }
            }
            "--steel-observation-mode" => {
                config.steel_observation_quadrature =
                    next_arg(&mut args, "--steel-observation-mode")?
                        .parse::<Team13SteelObservationQuadratureMode>()?
            }
            "--include-steel-observations" => config.include_steel_observations = true,
            "--observation-std-tesla" => {
                config.observation_std_tesla =
                    next_arg(&mut args, "--observation-std-tesla")?.parse()?
            }
            "--field-variance-mode" => {
                let value = next_arg(&mut args, "--field-variance-mode")?;
                if value == "none" || value == "off" {
                    config.estimate_field_variance = false;
                } else {
                    config.estimate_field_variance = true;
                    config.field_variance.mode = parse_variance_mode(&value)?;
                }
            }
            "--field-variance-probes" => {
                config.field_variance.num_variance_probes =
                    next_arg(&mut args, "--field-variance-probes")?.parse()?
            }
            "--field-variance-batches" => {
                config.field_variance.variance_batch_count =
                    next_arg(&mut args, "--field-variance-batches")?.parse()?
            }
            "--rng-seed" => {
                config.field_variance.rng_seed = next_arg(&mut args, "--rng-seed")?.parse()?
            }
            "--truth-max-iterations" => {
                config.truth_max_iterations =
                    next_arg(&mut args, "--truth-max-iterations")?.parse()?
            }
            "--linear-solve" => {
                config.linear_solve = parse_linear_solve(&next_arg(&mut args, "--linear-solve")?)?
            }
            "--cg-tolerance" => {
                let tolerance = next_arg(&mut args, "--cg-tolerance")?.parse()?;
                config.linear_solve = GaussNewtonLinearSolve::IterativeCg {
                    tolerance,
                    max_iterations: 4096,
                    warm_start: false,
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
                        tolerance: 1.0e-8,
                        max_iterations,
                        warm_start: false,
                    },
                };
            }
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?))
            }
            "--no-output" => config.output_dir = None,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    Ok(config)
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
        "Usage: cargo run --release -p feg-case-studies --example team13_operator_uncertainty_diagnostic -- [options]\n\
         Options:\n\
           --mesh <path>\n\
           --domain <half|full>\n\
           --material <ngsolve-tabulated-linear|smooth-quadratic>\n\
           --material-log-iron-nu-scale <float>\n\
           --tangent <nonlinear|linear-beta-zero>\n\
           --prior-kind <weak-ridge|ordinary-matern-alpha2|exact-potential>\n\
           --pde-residual <ungauged-curl|gauge-fixed>\n\
           --pde-weighting <euclidean|mass-inverse|mass-inverse-trace-normalized>\n\
           --pde-variance <float>\n\
           --steel-observation <none|face-cochain|ngsolve-style>\n\
           --field-variance-mode <hutchinson|selected-inverse|exact|none>\n\
           --field-variance-probes <usize>\n\
           --field-variance-batches <usize>\n\
           --linear-solve <direct-cholesky|iterative-cg>\n\
           --output-dir <path>"
    );
}
