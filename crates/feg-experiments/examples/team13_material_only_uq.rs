use feg_case_studies::team13::{
    run_team13_material_only_uq, Team13MapParityPdeResidualKind, Team13MapParityPriorKind,
    Team13MaterialOnlyUqConfig, Team13MaterialPriorCalibrationMode,
    Team13MaterialPriorCalibrationTarget, Team13PdeResidualWeighting, Team13PublishedSteelGap,
    Team13SteelObservationQuadratureMode,
};
use feg_infer::nonlinear::{GaussNewtonLinearSolve, GaussNewtonStepRegularization};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let result = run_team13_material_only_uq(&config)?;
    println!("TEAM 13 material-only UQ");
    println!("  mesh: {}", result.mesh_path.display());
    println!("  observed gap: {}", result.observed_steel_gap.as_str());
    println!(
        "  material prior: std={:.6e} calibration={} target={}",
        result.material_prior_std,
        result.material_prior_calibration.mode.as_str(),
        result.material_prior_calibration.target.as_str()
    );
    println!(
        "  posterior: converged={} theta=[{:.6e}, {:.6e}, {:.6e}] objective={:.6e}",
        result.posterior_converged,
        result.theta_map[0],
        result.theta_map[1],
        result.theta_map[2],
        result.objective_components.total
    );
    println!(
        "  sensitivity: rank={} singular=[{:.6e}, {:.6e}, {:.6e}]",
        result.sensitivity.rank,
        result.sensitivity.singular_values[0],
        result.sensitivity.singular_values[1],
        result.sensitivity.singular_values[2]
    );
    if let Some(output_dir) = &result.output_dir {
        println!("  wrote outputs: {}", output_dir.display());
    }
    Ok(())
}

fn parse_args() -> Result<Team13MaterialOnlyUqConfig, Box<dyn Error>> {
    let mut config = Team13MaterialOnlyUqConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.operator.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--observed-gap" => {
                config.observed_steel_gap =
                    parse_published_gap(&next_arg(&mut args, "--observed-gap")?)?
            }
            "--ampere-turns" => {
                config.operator.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--anchors-tesla" => {
                config.material_anchor_b_tesla =
                    parse_three_floats(&next_arg(&mut args, "--anchors-tesla")?)?
            }
            "--material-prior-std" => {
                config.material_prior_std = next_arg(&mut args, "--material-prior-std")?.parse()?
            }
            "--material-prior-calibration" => {
                config.material_prior_calibration =
                    next_arg(&mut args, "--material-prior-calibration")?
                        .parse::<Team13MaterialPriorCalibrationMode>()?
            }
            "--material-prior-target" => {
                config.material_prior_calibration_target =
                    next_arg(&mut args, "--material-prior-target")?
                        .parse::<Team13MaterialPriorCalibrationTarget>()?
            }
            "--material-prior-target-steel-rms-tesla" => {
                config.material_prior_target_steel_rms_tesla =
                    Some(next_arg(&mut args, "--material-prior-target-steel-rms-tesla")?.parse()?)
            }
            "--material-prior-std-floor" => {
                config.material_prior_std_floor =
                    next_arg(&mut args, "--material-prior-std-floor")?.parse()?
            }
            "--material-prior-std-ceiling" => {
                config.material_prior_std_ceiling =
                    next_arg(&mut args, "--material-prior-std-ceiling")?.parse()?
            }
            "--magnitude-smoothing-tesla" => {
                config.magnitude_smoothing_tesla =
                    next_arg(&mut args, "--magnitude-smoothing-tesla")?.parse()?
            }
            "--max-iterations" => {
                config.max_iterations = next_arg(&mut args, "--max-iterations")?.parse()?
            }
            "--max-line-search-steps" => {
                config.max_line_search_steps =
                    next_arg(&mut args, "--max-line-search-steps")?.parse()?
            }
            "--max-theta-step-norm" => {
                config.max_theta_step_norm =
                    next_arg(&mut args, "--max-theta-step-norm")?.parse()?
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
            "--steel-observation-mode" => {
                config.operator.steel_observation_quadrature =
                    next_arg(&mut args, "--steel-observation-mode")?
                        .parse::<Team13SteelObservationQuadratureMode>()?
            }
            "--observation-std-tesla" => {
                config.operator.observation_std_tesla =
                    next_arg(&mut args, "--observation-std-tesla")?.parse()?
            }
            "--truth-max-iterations" => {
                config.operator.truth_max_iterations =
                    next_arg(&mut args, "--truth-max-iterations")?.parse()?
            }
            "--linear-solve" => {
                config.operator.linear_solve =
                    parse_linear_solve(&next_arg(&mut args, "--linear-solve")?)?
            }
            "--step-regularization" => {
                config.step_regularization =
                    parse_step_regularization(&next_arg(&mut args, "--step-regularization")?)?
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

fn parse_published_gap(raw: &str) -> Result<Team13PublishedSteelGap, Box<dyn Error>> {
    match raw {
        "g052" | "0.52" | "0.52mm" => Ok(Team13PublishedSteelGap::G052),
        "g047" | "0.47" | "0.47mm" => Ok(Team13PublishedSteelGap::G047),
        other => Err(format!("unknown published gap `{other}`").into()),
    }
}

fn parse_three_floats(raw: &str) -> Result<[f64; 3], Box<dyn Error>> {
    let values = raw
        .split(',')
        .map(|value| value.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [a, b, c] => Ok([*a, *b, *c]),
        _ => Err("expected exactly three comma-separated floats".into()),
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

fn parse_step_regularization(raw: &str) -> Result<GaussNewtonStepRegularization, Box<dyn Error>> {
    match raw {
        "adaptive-lm" | "adaptive-levenberg-marquardt" => {
            Ok(GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt)
        }
        "lm-grid" | "levenberg-marquardt-grid" => {
            Ok(GaussNewtonStepRegularization::LevenbergMarquardtGrid)
        }
        "none" => Ok(GaussNewtonStepRegularization::None),
        other => Err(format!("unknown step regularization `{other}`").into()),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_material_only_uq -- [options]\n\
         Options:\n\
           --mesh <path>\n\
           --observed-gap <g052|g047>\n\
           --anchors-tesla <b0,b1,b2>\n\
           --material-prior-std <float>\n\
           --material-prior-calibration <fixed|steel-prior-predictive-rms>\n\
           --material-prior-target <observation-std|published-gap-difference|explicit>\n\
           --material-prior-target-steel-rms-tesla <float>\n\
           --material-prior-std-floor <float>\n\
           --material-prior-std-ceiling <float>\n\
           --magnitude-smoothing-tesla <float>\n\
           --max-iterations <usize>\n\
           --max-line-search-steps <usize>\n\
           --max-theta-step-norm <float>\n\
           --steel-observation-mode <face-cochain|ngsolve-style>\n\
           --observation-std-tesla <float>\n\
           --truth-max-iterations <usize>\n\
           --step-regularization <adaptive-lm|lm-grid|none>\n\
           --linear-solve <direct-cholesky|iterative-cg>\n\
           --output-dir <path>\n\
           --no-output"
    );
}
