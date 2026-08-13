use feg_case_studies::team13::{
    run_team13_same_mesh_linear_parity, Team13DomainMode, Team13PublishedSteelBenchmarkReport,
    Team13SameMeshLinearParityConfig, Team13SteelObservationQuadratureMode,
    Team13SteelSurfaceGroup,
};
use std::{env, error::Error, fs, path::PathBuf};

const DEFAULT_MESH: &str = "target/team13_measurement_planes/team13_half_measurement_planes.msh";
const DEFAULT_OUTPUT_DIR: &str = "target/team13_cochain_measurement_comparison";
const DEFAULT_RMSE_TOLERANCE: f64 = 2.0e-2;
const DEFAULT_MAX_ABS_TOLERANCE: f64 = 5.0e-2;

#[derive(Debug, Clone)]
struct ComparisonConfig {
    mesh_path: PathBuf,
    domain_mode: Team13DomainMode,
    ampere_turns: f64,
    output_dir: PathBuf,
    rmse_tolerance: f64,
    max_abs_tolerance: f64,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(DEFAULT_MESH),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            rmse_tolerance: DEFAULT_RMSE_TOLERANCE,
            max_abs_tolerance: DEFAULT_MAX_ABS_TOLERANCE,
        }
    }
}

#[derive(Debug, Clone)]
struct SteelComparisonRow {
    name: String,
    group: Team13SteelSurfaceGroup,
    quadrature_prediction: f64,
    cochain_prediction: f64,
    observed_g052: f64,
    observed_g047: f64,
}

#[derive(Debug, Clone)]
struct SteelComparisonStats {
    group: String,
    count: usize,
    rmse: f64,
    bias: f64,
    max_abs: f64,
    max_abs_name: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    if !config.mesh_path.exists() {
        return Err(format!(
            "mesh `{}` does not exist; generate it with: gmsh -3 geometries/team13_linear_measurement_planes.geo -setnumber FullDomain 0 -setnumber MeshScale 10 -o {}",
            config.mesh_path.display(),
            config.mesh_path.display()
        )
        .into());
    }

    let quadrature = run_team13_same_mesh_linear_parity(&Team13SameMeshLinearParityConfig {
        mesh_path: config.mesh_path.clone(),
        domain_mode: config.domain_mode,
        ampere_turns: config.ampere_turns,
        steel_observation_quadrature: Team13SteelObservationQuadratureMode::NgsolveStyle,
        output_dir: Some(config.output_dir.join("quadrature")),
    })?;
    let cochain = run_team13_same_mesh_linear_parity(&Team13SameMeshLinearParityConfig {
        mesh_path: config.mesh_path.clone(),
        domain_mode: config.domain_mode,
        ampere_turns: config.ampere_turns,
        steel_observation_quadrature: Team13SteelObservationQuadratureMode::FaceCochain,
        output_dir: Some(config.output_dir.join("face_cochain")),
    })?;

    let rows =
        compare_steel_predictions(&quadrature.steel_predictions, &cochain.steel_predictions)?;
    let stats = comparison_stats_by_group(&rows);
    fs::create_dir_all(&config.output_dir)?;
    fs::write(
        config.output_dir.join("steel_cochain_vs_quadrature.csv"),
        comparison_csv(&rows),
    )?;
    fs::write(
        config
            .output_dir
            .join("steel_cochain_vs_quadrature_summary.csv"),
        comparison_summary_csv(&stats),
    )?;

    let overall = stats
        .iter()
        .find(|stats| stats.group == "all")
        .ok_or("missing overall comparison stats")?;
    println!("TEAM 13 face-cochain measurement comparison");
    println!("  mesh: {}", config.mesh_path.display());
    println!(
        "  quadrature RMSE vs published: G=0.52 {:.6e}, G=0.47 {:.6e}",
        quadrature.steel_rmse_g052, quadrature.steel_rmse_g047
    );
    println!(
        "  face-cochain RMSE vs published: G=0.52 {:.6e}, G=0.47 {:.6e}",
        cochain.steel_rmse_g052, cochain.steel_rmse_g047
    );
    println!(
        "  cochain-vs-quadrature: rmse={:.6e} bias={:.6e} max_abs={:.6e} at {}",
        overall.rmse, overall.bias, overall.max_abs, overall.max_abs_name
    );
    println!("  outputs: {}", config.output_dir.display());

    if overall.rmse > config.rmse_tolerance || overall.max_abs > config.max_abs_tolerance {
        return Err(format!(
            "face-cochain comparison exceeded tolerance: rmse {:.6e} <= {:.6e}, max_abs {:.6e} <= {:.6e}",
            overall.rmse, config.rmse_tolerance, overall.max_abs, config.max_abs_tolerance
        )
        .into());
    }
    Ok(())
}

fn compare_steel_predictions(
    quadrature: &[Team13PublishedSteelBenchmarkReport],
    cochain: &[Team13PublishedSteelBenchmarkReport],
) -> Result<Vec<SteelComparisonRow>, String> {
    if quadrature.len() != cochain.len() {
        return Err(format!(
            "comparison row count mismatch: quadrature={} cochain={}",
            quadrature.len(),
            cochain.len()
        ));
    }
    quadrature
        .iter()
        .zip(cochain)
        .map(|(quadrature, cochain)| {
            if quadrature.name != cochain.name || quadrature.group != cochain.group {
                return Err(format!(
                    "comparison row mismatch: quadrature `{}`/{}, cochain `{}`/{}",
                    quadrature.name,
                    quadrature.group.as_str(),
                    cochain.name,
                    cochain.group.as_str()
                ));
            }
            Ok(SteelComparisonRow {
                name: quadrature.name.clone(),
                group: quadrature.group,
                quadrature_prediction: quadrature.posterior_prediction,
                cochain_prediction: cochain.posterior_prediction,
                observed_g052: quadrature.observed_g_052,
                observed_g047: quadrature.observed_g_047,
            })
        })
        .collect()
}

fn comparison_stats_by_group(rows: &[SteelComparisonRow]) -> Vec<SteelComparisonStats> {
    let mut stats = Vec::new();
    stats.push(comparison_stats("all", rows));
    for group in [
        Team13SteelSurfaceGroup::MidSheet,
        Team13SteelSurfaceGroup::BackRightTop,
        Team13SteelSurfaceGroup::BackRightEdge,
    ] {
        let group_rows = rows
            .iter()
            .filter(|row| row.group == group)
            .cloned()
            .collect::<Vec<_>>();
        stats.push(comparison_stats(group.as_str(), &group_rows));
    }
    stats
}

fn comparison_stats(group: &str, rows: &[SteelComparisonRow]) -> SteelComparisonStats {
    let count = rows.len();
    let mut sum_sq = 0.0;
    let mut sum = 0.0;
    let mut max_abs = 0.0;
    let mut max_abs_name = String::new();
    for row in rows {
        let diff = row.cochain_prediction - row.quadrature_prediction;
        sum_sq += diff * diff;
        sum += diff;
        if diff.abs() >= max_abs {
            max_abs = diff.abs();
            max_abs_name = row.name.clone();
        }
    }
    SteelComparisonStats {
        group: group.to_string(),
        count,
        rmse: if count == 0 {
            f64::NAN
        } else {
            (sum_sq / count as f64).sqrt()
        },
        bias: if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        },
        max_abs,
        max_abs_name,
    }
}

fn comparison_csv(rows: &[SteelComparisonRow]) -> String {
    let mut csv = "name,group,quadrature_prediction,cochain_prediction,cochain_minus_quadrature,observed_g052,observed_g047,quadrature_residual_g052,cochain_residual_g052,quadrature_residual_g047,cochain_residual_g047\n"
        .to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            csv_field(&row.name),
            row.group.as_str(),
            row.quadrature_prediction,
            row.cochain_prediction,
            row.cochain_prediction - row.quadrature_prediction,
            row.observed_g052,
            row.observed_g047,
            row.quadrature_prediction - row.observed_g052,
            row.cochain_prediction - row.observed_g052,
            row.quadrature_prediction - row.observed_g047,
            row.cochain_prediction - row.observed_g047,
        ));
    }
    csv
}

fn comparison_summary_csv(stats: &[SteelComparisonStats]) -> String {
    let mut csv = "group,count,rmse,bias,max_abs,max_abs_name\n".to_string();
    for stats in stats {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{}\n",
            stats.group,
            stats.count,
            stats.rmse,
            stats.bias,
            stats.max_abs,
            csv_field(&stats.max_abs_name)
        ));
    }
    csv
}

fn parse_args() -> Result<ComparisonConfig, Box<dyn Error>> {
    let mut config = ComparisonConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--domain" => config.domain_mode = parse_domain(&next_arg(&mut args, "--domain")?)?,
            "--ampere-turns" => {
                config.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            "--output-dir" => {
                config.output_dir = PathBuf::from(next_arg(&mut args, "--output-dir")?)
            }
            "--rmse-tolerance" => {
                config.rmse_tolerance = next_arg(&mut args, "--rmse-tolerance")?.parse()?
            }
            "--max-abs-tolerance" => {
                config.max_abs_tolerance = next_arg(&mut args, "--max-abs-tolerance")?.parse()?
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
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn parse_domain(raw: &str) -> Result<Team13DomainMode, Box<dyn Error>> {
    match raw {
        "half" | "half-z" => Ok(Team13DomainMode::HalfZNonnegative),
        "full" => Ok(Team13DomainMode::Full),
        other => Err(format!("unknown domain `{other}`; expected half or full").into()),
    }
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn print_help() {
    println!(
        "Usage: team13_cochain_measurement_comparison [options]\n\
         Options:\n\
           --mesh <path>\n\
           --domain <half|full>\n\
           --ampere-turns <float>\n\
           --output-dir <path>\n\
           --rmse-tolerance <float>\n\
           --max-abs-tolerance <float>"
    );
}
