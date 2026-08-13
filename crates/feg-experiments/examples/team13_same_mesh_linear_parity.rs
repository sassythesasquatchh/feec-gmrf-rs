use feg_case_studies::team13::{
    run_team13_same_mesh_linear_parity, Team13DomainMode, Team13SameMeshLinearParityConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = Team13SameMeshLinearParityConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--domain" => config.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?,
            "--ampere-turns" => {
                config.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?))
            }
            "--skip-output" => config.output_dir = None,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    let result = run_team13_same_mesh_linear_parity(&config)?;
    println!("TEAM 13 FEEC same-mesh beta-zero linear parity diagnostic");
    println!("  mesh: {}", result.mesh_path.display());
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
        "  operator: dim={} nnz={} rhs_l2={:.6e} solution_l2={:.6e} residual_l2={:.6e}",
        result.operator_dimension,
        result.operator_nnz,
        result.rhs_l2,
        result.solution_l2,
        result.linear_residual_l2
    );
    println!(
        "  energy={:.6e} work={:.6e} steel_rmse_g052={:.6e} steel_rmse_g047={:.6e}",
        result.energy, result.work, result.steel_rmse_g052, result.steel_rmse_g047
    );
    println!(
        "  volumes: total={:.6e} iron={:.6e} coil_total={:.6e} source_free_air={:.6e}",
        result.audit.total_volume,
        audit_volume(&result, "iron"),
        audit_volume(&result, "coil_total"),
        audit_volume(&result, "source_free_air"),
    );
    println!(
        "  overlap flags: iron_and_coil_cells={} multiple_coil_cells={} unclassified_cells={}",
        result.audit.iron_and_coil_cells,
        result.audit.multiple_coil_cells,
        result.audit.unclassified_cells
    );
    for summary in &result.steel_group_summaries_g052 {
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

    Ok(())
}

fn audit_volume(
    result: &feg_case_studies::team13::Team13SameMeshLinearParityResult,
    name: &str,
) -> f64 {
    result
        .audit
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .map_or(0.0, |entry| entry.volume)
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
        "Usage: team13_same_mesh_linear_parity --mesh <path> [options]\n\
         Options:\n\
           --domain <half|full>\n\
           --ampere-turns <float>\n\
           --output-dir <path>\n\
           --skip-output"
    );
}
