use feg_case_studies::team7::{
    generate_team7_geo, solve_team7_linear_probabilistic, Team7Config, Team7LinearSolveResult,
    Team7MaternStabilizer, Team7PriorConfig, Team7PriorMeanMode, Team7SourceCalibrationReport,
    Team7SourceModel, TEAM7_DEFAULT_SOURCE_GUARD_Z_SCORE, TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
    TEAM7_DEFAULT_SOURCE_MIN_ALPHA, TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
    TEAM7_DEFAULT_SOURCE_SHAPE_MODES, TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
};
use feg_infer::linear_pde::{LinearPdeVarianceConfig, LinearPdeVarianceMode};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExampleRunMode {
    Comparison,
    DeterministicOnly,
    SourceOnly,
}

#[derive(Debug, Clone)]
struct ExampleConfig {
    mesh_path: PathBuf,
    geo_path: PathBuf,
    mesh_size: f64,
    output_dir: Option<PathBuf>,
    skip_output: bool,
    force_mesh_generation: bool,
    run_mode: ExampleRunMode,
    source_log_alpha_std: f64,
    source_phase_std_rad: f64,
    source_shape_relative_std: f64,
    source_shape_modes: usize,
    source_guard_z_score: f64,
    source_min_alpha: f64,
    pde_residual_std: f64,
    b_observation_std_tesla: f64,
    jy_observation_std: f64,
    prior: Team7PriorConfig,
    prior_mean: Team7PriorMeanMode,
    variance_mode: LinearPdeVarianceMode,
    variance_probes: usize,
    diagnostics: bool,
}

struct VariantSpec {
    name: &'static str,
    slug: &'static str,
    source_model: Team7SourceModel,
}

struct VariantRun {
    name: &'static str,
    output_dir: Option<PathBuf>,
    result: Team7LinearSolveResult,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("meshes/team7.msh"),
            geo_path: PathBuf::from("geometries/team7.geo"),
            mesh_size: 0.12,
            output_dir: Some(PathBuf::from("out/examples/team7_linear_probabilistic")),
            skip_output: false,
            force_mesh_generation: false,
            run_mode: ExampleRunMode::Comparison,
            source_log_alpha_std: TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
            source_phase_std_rad: TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
            source_shape_relative_std: TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
            source_shape_modes: TEAM7_DEFAULT_SOURCE_SHAPE_MODES,
            source_guard_z_score: TEAM7_DEFAULT_SOURCE_GUARD_Z_SCORE,
            source_min_alpha: TEAM7_DEFAULT_SOURCE_MIN_ALPHA,
            pde_residual_std: Team7Config::default().pde_residual_std,
            b_observation_std_tesla: Team7Config::default().b_observation_std_tesla,
            jy_observation_std: Team7Config::default().jy_observation_std,
            prior: Team7PriorConfig::default(),
            prior_mean: Team7PriorMeanMode::default(),
            variance_mode: LinearPdeVarianceMode::Hutchinson,
            variance_probes: 16,
            diagnostics: false,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let workspace = workspace_root()?;
    let mesh_path = absolutize(&workspace, &args.mesh_path);
    let geo_path = absolutize(&workspace, &args.geo_path);
    if args.force_mesh_generation || !mesh_path.exists() {
        generate_mesh(&geo_path, &mesh_path, args.mesh_size)?;
    }

    let base_output_dir = if args.skip_output {
        None
    } else {
        args.output_dir
            .clone()
            .map(|path| absolutize(&workspace, &path))
    };
    let mut base_config = Team7Config::default();
    base_config.mesh_path = mesh_path.clone();
    base_config.output_dir = None;
    base_config.prior = args.prior;
    base_config.prior_mean = args.prior_mean;
    base_config.pde_residual_std = args.pde_residual_std;
    base_config.b_observation_std_tesla = args.b_observation_std_tesla;
    base_config.jy_observation_std = args.jy_observation_std;
    base_config.solver.variance = LinearPdeVarianceConfig {
        mode: args.variance_mode,
        num_variance_probes: args.variance_probes,
        variance_batch_count: 1,
        rng_seed: 7,
        local_rb_block_size: 16,
    };

    let comparison_mode = args.run_mode == ExampleRunMode::Comparison;
    let mut runs = Vec::new();
    for variant in variant_specs(&args) {
        let mut config = base_config.clone();
        config.source_model = variant.source_model;
        config.output_dir =
            output_dir_for_variant(base_output_dir.as_deref(), comparison_mode, variant.slug);
        let output_dir = config.output_dir.clone();
        let result = solve_team7_linear_probabilistic(&config)?;
        runs.push(VariantRun {
            name: variant.name,
            output_dir,
            result,
        });
    }

    println!("TEAM 7 linear probabilistic FEEC solve");
    println!("  mesh: {}", mesh_path.display());
    println!(
        "  prior: physical weight {:.6e}, Matérn stabilizer {}",
        args.prior.physical_weight,
        describe_stabilizer(args.prior.matern_stabilizer)
    );
    println!("  prior mean: {}", describe_prior_mean(args.prior_mean));
    print_summary_table(&runs);
    if args.diagnostics {
        print_diagnostics(&runs);
    }
    if let Some(output_dir) = &base_output_dir {
        if comparison_mode {
            write_comparison_csv(output_dir, &runs)?;
            println!("  outputs: {}", output_dir.display());
        } else if let Some(run) = runs.first().and_then(|run| run.output_dir.as_ref()) {
            println!("  outputs: {}", run.display());
        }
    }
    let rejected = rejected_source_runs(&runs);
    if !rejected.is_empty() {
        return Err(io::Error::other(format!(
            "source calibration rejected: {}",
            rejected.join(" | ")
        ))
        .into());
    }
    Ok(())
}

fn variant_specs(args: &ExampleConfig) -> Vec<VariantSpec> {
    let source_model = Team7SourceModel::HarmonicCalibration {
        log_alpha_std: args.source_log_alpha_std,
        phase_std_rad: args.source_phase_std_rad,
        shape_relative_std: args.source_shape_relative_std,
        shape_modes: args.source_shape_modes,
        guard_z_score: args.source_guard_z_score,
        min_alpha: args.source_min_alpha,
    };
    match args.run_mode {
        ExampleRunMode::Comparison => vec![
            VariantSpec {
                name: "deterministic",
                slug: "deterministic",
                source_model: Team7SourceModel::Deterministic,
            },
            VariantSpec {
                name: "source_calibration",
                slug: "source_calibration",
                source_model,
            },
        ],
        ExampleRunMode::DeterministicOnly => vec![VariantSpec {
            name: "deterministic",
            slug: "deterministic",
            source_model: Team7SourceModel::Deterministic,
        }],
        ExampleRunMode::SourceOnly => vec![VariantSpec {
            name: "source_calibration",
            slug: "source_calibration",
            source_model,
        }],
    }
}

fn output_dir_for_variant(
    base_output_dir: Option<&Path>,
    comparison_mode: bool,
    slug: &str,
) -> Option<PathBuf> {
    base_output_dir.map(|base| {
        if comparison_mode {
            base.join(slug)
        } else {
            base.to_path_buf()
        }
    })
}

fn print_summary_table(runs: &[VariantRun]) {
    println!(
        "  {:<20} {:>10} {:>10} {:>13} {:>13} {:>13} {:>12} {:>10} {:>10}",
        "variant",
        "reduced",
        "joint",
        "Bz RMSE",
        "Jy RMSE",
        "alpha",
        "phase(rad)",
        "max-z",
        "status"
    );
    for run in runs {
        let result = &run.result;
        let source = source_summary(result.source_calibration.as_ref());
        println!(
            "  {:<20} {:>10} {:>10} {:>13.6e} {:>13.6e} {:>13} {:>12} {:>10} {:>10}",
            run.name,
            result.posterior.reduced_posterior_mean.len(),
            result.posterior.debug.joint_dimension,
            rmse_value(result, "Bz"),
            rmse_value(result, "Jy"),
            source.alpha,
            source.phase,
            source.max_z_score,
            source.status
        );
    }
}

fn print_diagnostics(runs: &[VariantRun]) {
    println!("  diagnostics:");
    for run in runs {
        println!(
            "    {} posterior state mean L2: {:.6e}",
            run.name,
            run.result
                .posterior
                .reduced_posterior_mean
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        );
        for input in &run.result.posterior.latent_inputs {
            println!(
                "    {} latent {} mean {:?} variance {:?}",
                run.name, input.name, input.mean, input.variance
            );
        }
        if let Some(report) = &run.result.source_calibration {
            if let Some(reason) = &report.rejection_reason {
                println!("    {} source rejection: {reason}", run.name);
            }
        }
    }
}

struct SourceSummary {
    alpha: String,
    phase: String,
    max_z_score: String,
    status: String,
}

fn source_summary(report: Option<&Team7SourceCalibrationReport>) -> SourceSummary {
    match report {
        Some(report) => {
            let max_z = report
                .log_alpha_z_score
                .max(report.phase_z_score)
                .max(report.max_shape_z_score);
            SourceSummary {
                alpha: format!("{:.6e}", report.alpha_mean),
                phase: format!("{:.6e}", report.phase_mean_rad),
                max_z_score: format!("{:.3}", max_z),
                status: if report.accepted {
                    "accepted".to_string()
                } else {
                    "rejected".to_string()
                },
            }
        }
        None => SourceSummary {
            alpha: "n/a".to_string(),
            phase: "n/a".to_string(),
            max_z_score: "n/a".to_string(),
            status: "n/a".to_string(),
        },
    }
}

fn rmse_value(result: &Team7LinearSolveResult, family: &str) -> f64 {
    result
        .rmse
        .iter()
        .find(|report| report.family == family)
        .map(|report| report.rmse)
        .unwrap_or(f64::NAN)
}

fn rejected_source_runs(runs: &[VariantRun]) -> Vec<String> {
    runs.iter()
        .filter_map(|run| {
            let report = run.result.source_calibration.as_ref()?;
            if report.accepted {
                None
            } else {
                Some(format!(
                    "{} ({})",
                    run.name,
                    report
                        .rejection_reason
                        .as_deref()
                        .unwrap_or("no rejection reason")
                ))
            }
        })
        .collect()
}

fn write_comparison_csv(output_dir: &Path, runs: &[VariantRun]) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let path = output_dir.join("team7_source_comparison.csv");
    let mut file = File::create(&path)?;
    writeln!(
        file,
        "variant,reduced_dimension,joint_dimension,bz_rmse,jy_rmse,source_alpha,source_phase_rad,source_log_alpha_z,source_phase_z,source_shape_max_z,source_status,source_rejection_reason"
    )?;
    for run in runs {
        let report = run.result.source_calibration.as_ref();
        let alpha = report
            .map(|report| format!("{:.12e}", report.alpha_mean))
            .unwrap_or_default();
        let phase = report
            .map(|report| format!("{:.12e}", report.phase_mean_rad))
            .unwrap_or_default();
        let log_alpha_z = report
            .map(|report| format!("{:.12e}", report.log_alpha_z_score))
            .unwrap_or_default();
        let phase_z = report
            .map(|report| format!("{:.12e}", report.phase_z_score))
            .unwrap_or_default();
        let shape_z = report
            .map(|report| format!("{:.12e}", report.max_shape_z_score))
            .unwrap_or_default();
        let status = report
            .map(|report| {
                if report.accepted {
                    "accepted"
                } else {
                    "rejected"
                }
            })
            .unwrap_or("");
        let reason = report
            .and_then(|report| report.rejection_reason.as_deref())
            .unwrap_or("");
        writeln!(
            file,
            "{},{},{},{:.12e},{:.12e},{},{},{},{},{},{},{}",
            run.name,
            run.result.posterior.reduced_posterior_mean.len(),
            run.result.posterior.debug.joint_dimension,
            rmse_value(&run.result, "Bz"),
            rmse_value(&run.result, "Jy"),
            alpha,
            phase,
            log_alpha_z,
            phase_z,
            shape_z,
            status,
            csv_escape(reason)
        )?;
    }
    Ok(())
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn generate_mesh(geo_path: &Path, mesh_path: &Path, mesh_size: f64) -> Result<(), Box<dyn Error>> {
    generate_team7_geo(geo_path, mesh_size)?;
    if let Some(parent) = mesh_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let gmsh = if Path::new("/opt/homebrew/bin/gmsh").exists() {
        "/opt/homebrew/bin/gmsh"
    } else {
        "gmsh"
    };
    let output = Command::new(gmsh)
        .arg("-3")
        .arg(geo_path)
        .arg("-format")
        .arg("msh41")
        .arg("-o")
        .arg(mesh_path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "gmsh failed for {}: {}",
            geo_path.display(),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(())
}

fn parse_args() -> Result<ExampleConfig, Box<dyn Error>> {
    let mut config = ExampleConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh" => config.mesh_path = PathBuf::from(next_value(&mut args, "--mesh")?),
            "--geo" => config.geo_path = PathBuf::from(next_value(&mut args, "--geo")?),
            "--mesh-size" => {
                config.mesh_size = parse_f64(next_value(&mut args, "--mesh-size")?, "--mesh-size")?
            }
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_value(&mut args, "--output-dir")?))
            }
            "--skip-output" => config.skip_output = true,
            "--generate-mesh" => config.force_mesh_generation = true,
            "--deterministic-only" | "--no-source-mode" => {
                config.run_mode = ExampleRunMode::DeterministicOnly
            }
            "--source-mode" => config.run_mode = ExampleRunMode::SourceOnly,
            "--source-log-alpha-std" => {
                config.source_log_alpha_std = parse_f64(
                    next_value(&mut args, "--source-log-alpha-std")?,
                    "--source-log-alpha-std",
                )?
            }
            "--source-relative-std" => {
                config.source_log_alpha_std = parse_f64(
                    next_value(&mut args, "--source-relative-std")?,
                    "--source-relative-std",
                )?
            }
            "--source-phase-std" => {
                config.source_phase_std_rad = parse_f64(
                    next_value(&mut args, "--source-phase-std")?,
                    "--source-phase-std",
                )?
            }
            "--source-shape-std" => {
                config.source_shape_relative_std = parse_f64(
                    next_value(&mut args, "--source-shape-std")?,
                    "--source-shape-std",
                )?
            }
            "--source-shape-modes" => {
                config.source_shape_modes = parse_usize(
                    next_value(&mut args, "--source-shape-modes")?,
                    "--source-shape-modes",
                )?
            }
            "--source-guard-z" => {
                config.source_guard_z_score = parse_f64(
                    next_value(&mut args, "--source-guard-z")?,
                    "--source-guard-z",
                )?
            }
            "--source-min-alpha" => {
                config.source_min_alpha = parse_f64(
                    next_value(&mut args, "--source-min-alpha")?,
                    "--source-min-alpha",
                )?
            }
            "--source-min-multiplier" => {
                config.source_min_alpha = parse_f64(
                    next_value(&mut args, "--source-min-multiplier")?,
                    "--source-min-multiplier",
                )?
            }
            "--pde-residual-std" => {
                config.pde_residual_std = parse_f64(
                    next_value(&mut args, "--pde-residual-std")?,
                    "--pde-residual-std",
                )?
            }
            "--b-observation-std" => {
                config.b_observation_std_tesla = parse_f64(
                    next_value(&mut args, "--b-observation-std")?,
                    "--b-observation-std",
                )?
            }
            "--jy-observation-std" => {
                config.jy_observation_std = parse_f64(
                    next_value(&mut args, "--jy-observation-std")?,
                    "--jy-observation-std",
                )?
            }
            "--physical-prior-weight" => {
                config.prior.physical_weight = parse_f64(
                    next_value(&mut args, "--physical-prior-weight")?,
                    "--physical-prior-weight",
                )?
            }
            "--matern-stabilizer-absolute" => {
                config.prior.matern_stabilizer = Team7MaternStabilizer::Absolute(parse_f64(
                    next_value(&mut args, "--matern-stabilizer-absolute")?,
                    "--matern-stabilizer-absolute",
                )?)
            }
            "--matern-stabilizer-relative" => {
                config.prior.matern_stabilizer =
                    Team7MaternStabilizer::RelativeToPhysicalDiagonal(parse_f64(
                        next_value(&mut args, "--matern-stabilizer-relative")?,
                        "--matern-stabilizer-relative",
                    )?)
            }
            "--disable-matern-stabilizer" => {
                config.prior.matern_stabilizer = Team7MaternStabilizer::Disabled
            }
            "--zero-prior-mean" => config.prior_mean = Team7PriorMeanMode::Zero,
            "--nominal-prior-mean" => config.prior_mean = Team7PriorMeanMode::DeterministicNominal,
            "--exact-variance" => config.variance_mode = LinearPdeVarianceMode::Exact,
            "--diagnostics" => config.diagnostics = true,
            "--variance-probes" => {
                config.variance_probes = parse_usize(
                    next_value(&mut args, "--variance-probes")?,
                    "--variance-probes",
                )?
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            unknown => return Err(invalid_input(format!("unknown argument `{unknown}`")).into()),
        }
    }
    if !config.source_log_alpha_std.is_finite() || config.source_log_alpha_std <= 0.0 {
        return Err(invalid_input("--source-log-alpha-std must be finite and positive").into());
    }
    if !config.source_phase_std_rad.is_finite() || config.source_phase_std_rad <= 0.0 {
        return Err(invalid_input("--source-phase-std must be finite and positive").into());
    }
    if !config.source_shape_relative_std.is_finite() || config.source_shape_relative_std <= 0.0 {
        return Err(invalid_input("--source-shape-std must be finite and positive").into());
    }
    if config.source_shape_modes > 2 {
        return Err(invalid_input("--source-shape-modes currently supports at most 2").into());
    }
    if !config.source_guard_z_score.is_finite() || config.source_guard_z_score <= 0.0 {
        return Err(invalid_input("--source-guard-z must be finite and positive").into());
    }
    if !config.source_min_alpha.is_finite() || config.source_min_alpha <= 0.0 {
        return Err(invalid_input("--source-min-alpha must be finite and positive").into());
    }
    if !config.pde_residual_std.is_finite() || config.pde_residual_std <= 0.0 {
        return Err(invalid_input("--pde-residual-std must be finite and positive").into());
    }
    if !config.b_observation_std_tesla.is_finite() || config.b_observation_std_tesla <= 0.0 {
        return Err(invalid_input("--b-observation-std must be finite and positive").into());
    }
    if !config.jy_observation_std.is_finite() || config.jy_observation_std <= 0.0 {
        return Err(invalid_input("--jy-observation-std must be finite and positive").into());
    }
    Ok(config)
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| invalid_input("failed to infer workspace root").into())
}

fn absolutize(workspace: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

fn describe_stabilizer(stabilizer: Team7MaternStabilizer) -> String {
    match stabilizer {
        Team7MaternStabilizer::Disabled => "disabled".to_string(),
        Team7MaternStabilizer::Absolute(value) => format!("absolute {value:.6e}"),
        Team7MaternStabilizer::RelativeToPhysicalDiagonal(value) => {
            format!("relative-to-physical-diagonal {value:.6e}")
        }
    }
}

fn describe_prior_mean(mode: Team7PriorMeanMode) -> &'static str {
    match mode {
        Team7PriorMeanMode::Zero => "zero field",
        Team7PriorMeanMode::DeterministicNominal => "deterministic nominal FEEC solve",
    }
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")).into())
}

fn parse_f64(value: String, flag: &'static str) -> Result<f64, Box<dyn Error>> {
    value
        .parse::<f64>()
        .map_err(|err| invalid_input(format!("invalid {flag} value `{value}`: {err}")).into())
}

fn parse_usize(value: String, flag: &'static str) -> Result<usize, Box<dyn Error>> {
    value
        .parse::<usize>()
        .map_err(|err| invalid_input(format!("invalid {flag} value `{value}`: {err}")).into())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team7_linear_probabilistic -- [options]\n\
         \n\
         Options:\n\
           --mesh <path>          Mesh path (default meshes/team7.msh)\n\
           --geo <path>           Generated .geo path (default geometries/team7.geo)\n\
           --mesh-size <h>        Coarse Gmsh size (default 0.12)\n\
           --generate-mesh        Regenerate the mesh even if it already exists\n\
           --skip-output          Do not write VTU/CSV outputs\n\
           --output-dir <path>    Output directory\n\
           --deterministic-only   Run only the deterministic nominal source variant\n\
           --source-mode          Run only guarded source calibration\n\
           --no-source-mode       Compatibility alias for --deterministic-only\n\
           --source-log-alpha-std <s>\n\
                                  Log-amplitude prior std (default 0.02)\n\
           --source-relative-std <s>\n\
                                  Compatibility alias for --source-log-alpha-std\n\
           --source-phase-std <rad>\n\
                                  Phase prior std in radians (default pi/180)\n\
           --source-shape-std <s>\n\
                                  Orthogonal shape-mode prior std (default 0.02)\n\
           --source-shape-modes <n>\n\
                                  Number of orthogonal shape modes, 0-2 (default 2)\n\
           --source-guard-z <z>   Source calibration z-score guard (default 5)\n\
           --source-min-alpha <a>\n\
                                  Minimum accepted exp(log-alpha) (default 0.5)\n\
           --source-min-multiplier <m>\n\
                                  Compatibility alias for --source-min-alpha\n\
           --pde-residual-std <s>\n\
                                  PDE residual noise std (default 1e-3)\n\
           --b-observation-std <s>\n\
                                  B observation std in tesla (default 1e-4)\n\
           --jy-observation-std <s>\n\
                                  Jy observation std (default 5e-2)\n\
           --physical-prior-weight <w>\n\
                                  Weight for B^T C_xi^-1 B (default 1)\n\
           --matern-stabilizer-relative <r>\n\
                                  Relative Matérn stabilizer scale (default 1e-6)\n\
           --matern-stabilizer-absolute <lambda>\n\
                                  Absolute Matérn stabilizer weight\n\
           --disable-matern-stabilizer\n\
                                  Use no independent Matérn stabilizer\n\
           --nominal-prior-mean  Center the field prior on the deterministic FEEC solve (default)\n\
           --zero-prior-mean     Center the field prior on zero\n\
           --exact-variance       Use exact marginal variances\n\
           --variance-probes <n>  Hutchinson probes for marginal variances\n\
           --diagnostics          Print posterior latent-source diagnostics"
    );
}
