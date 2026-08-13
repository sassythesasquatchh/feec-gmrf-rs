use feg_case_studies::team13::{
    run_team13_deterministic_benchmark, team13_steel_ngsolve_comparison_rmse,
    Team13DeterministicBenchmarkConfig, Team13DomainMode, Team13NonlinearForwardParityConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = Team13DeterministicBenchmarkConfig::default();
    let mut run_nonlinear = false;
    let mut nonlinear_config = Team13NonlinearForwardParityConfig {
        mesh_path: config.linear.mesh_path.clone(),
        domain_mode: config.linear.domain_mode,
        ampere_turns: config.linear.ampere_turns,
        output_dir: Some(PathBuf::from(
            "target/team13_deterministic_benchmark/feec_nonlinear",
        )),
        ..Team13NonlinearForwardParityConfig::default()
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => {
                let mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?);
                config.linear.mesh_path = mesh_path.clone();
                nonlinear_config.mesh_path = mesh_path;
            }
            "--domain" => {
                let domain = parse_domain(&next_arg(&mut args, "--domain")?)?;
                config.linear.domain_mode = domain;
                nonlinear_config.domain_mode = domain;
            }
            "--ampere-turns" => {
                let ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?;
                config.linear.ampere_turns = ampere_turns;
                nonlinear_config.ampere_turns = ampere_turns;
            }
            "--output-dir" => {
                config.linear.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?))
            }
            "--ngsolve-reference-dir" => {
                config.ngsolve_linear_reference_dir =
                    PathBuf::from(next_arg(&mut args, "--ngsolve-reference-dir")?)
            }
            "--run-nonlinear" => run_nonlinear = true,
            "--nonlinear-output-dir" => {
                nonlinear_config.output_dir = Some(PathBuf::from(next_arg(
                    &mut args,
                    "--nonlinear-output-dir",
                )?))
            }
            "--ngsolve-nonlinear-reference-dir" => {
                config.ngsolve_nonlinear_reference_dir = Some(PathBuf::from(next_arg(
                    &mut args,
                    "--ngsolve-nonlinear-reference-dir",
                )?))
            }
            "--nonlinear-max-iterations" => {
                nonlinear_config.max_iterations =
                    next_arg(&mut args, "--nonlinear-max-iterations")?.parse()?
            }
            "--skip-output" => {
                config.linear.output_dir = None;
                nonlinear_config.output_dir = None;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    if run_nonlinear {
        config.nonlinear = Some(nonlinear_config);
    }

    let result = run_team13_deterministic_benchmark(&config)?;
    let linear_ngsolve_rmse = team13_steel_ngsolve_comparison_rmse(&result.linear_comparison);
    println!("TEAM 13 FEEC deterministic benchmark baseline");
    println!("  linear mesh: {}", result.linear.mesh_path.display());
    println!(
        "  linear RMSE vs published: G=0.52 {:.6e}, G=0.47 {:.6e}",
        result.linear.steel_rmse_g052, result.linear.steel_rmse_g047
    );
    println!(
        "  linear FEEC-vs-NGSolve steel RMSE: {:.6e} (max abs {:.6e})",
        linear_ngsolve_rmse,
        comparison_max_abs(&result.linear_comparison)
    );
    if let Some(output_dir) = &result.linear.output_dir {
        println!("  linear outputs: {}", output_dir.display());
    }
    if let Some(nonlinear) = &result.nonlinear {
        println!(
            "  nonlinear RMSE vs published: G=0.52 {:.6e}, G=0.47 {:.6e}",
            nonlinear.nonlinear_steel_rmse_g052, nonlinear.nonlinear_steel_rmse_g047
        );
        if let Some(comparison) = &result.nonlinear_comparison {
            println!(
                "  nonlinear FEEC-vs-NGSolve steel RMSE: {:.6e} (max abs {:.6e})",
                team13_steel_ngsolve_comparison_rmse(comparison),
                comparison_max_abs(comparison)
            );
        }
        if let Some(output_dir) = &nonlinear.output_dir {
            println!("  nonlinear outputs: {}", output_dir.display());
        }
    }

    Ok(())
}

fn comparison_max_abs(
    comparison: &[feg_case_studies::team13::Team13SteelNgsolveComparisonReport],
) -> f64 {
    comparison
        .iter()
        .map(|report| report.feec_minus_ngsolve.abs())
        .fold(0.0, f64::max)
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

fn print_help() {
    println!(
        "Usage: team13_deterministic_benchmark [options]\n\
         Options:\n\
           --mesh <path>\n\
           --domain <half|full>\n\
           --ampere-turns <float>\n\
           --output-dir <path>\n\
           --ngsolve-reference-dir <path>\n\
           --run-nonlinear\n\
           --nonlinear-output-dir <path>\n\
           --ngsolve-nonlinear-reference-dir <path>\n\
           --nonlinear-max-iterations <usize>\n\
           --skip-output"
    );
}
