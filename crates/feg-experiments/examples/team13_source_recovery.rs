use feg_case_studies::team13::{
    run_team13_source_recovery_experiment, Team13DiscrepancyPriorKind, Team13DomainMode,
    Team13FieldPriorKind, Team13PdeResidualWeighting, Team13SourceRecoveryConfig,
    DEFAULT_SOURCE_ALPHA_TRUE, DEFAULT_SOURCE_ALPHA_TRUE_MODES, MU_R_IRON, NU_AIR, NU_IRON,
    TEAM13_COIL_MODE_COUNT,
};
use feg_infer::linear_pde::{
    LinearPdePrecisionPolicy, LinearPdeUqSolverConfig, LinearPdeVarianceConfig,
    LinearPdeVarianceMode,
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
    ampere_turns: f64,
    source_alpha_true: f64,
    source_alpha_true_modes: [f64; TEAM13_COIL_MODE_COUNT],
    run_eight_mode_recovery: bool,
    source_prior_std: f64,
    pde_variance: f64,
    pde_residual_weighting: Team13PdeResidualWeighting,
    discrepancy_prior: Team13DiscrepancyPriorKind,
    discrepancy_prior_precision_scale: f64,
    field_prior_precision_scale: f64,
    field_priors: Vec<Team13FieldPriorKind>,
    b_observation_std_tesla: f64,
    nominal_observation_csv_path: Option<PathBuf>,
    perturbed_observation_csv_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    skip_output: bool,
    solver: LinearPdeUqSolverConfig,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            domain_mode: Team13DomainMode::HalfZNonnegative,
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 8.0,
            force_mesh_generation: false,
            ampere_turns: 1000.0,
            source_alpha_true: DEFAULT_SOURCE_ALPHA_TRUE,
            source_alpha_true_modes: DEFAULT_SOURCE_ALPHA_TRUE_MODES,
            run_eight_mode_recovery: false,
            source_prior_std: 0.10,
            pde_variance: 1e-6,
            pde_residual_weighting: Team13PdeResidualWeighting::Euclidean,
            discrepancy_prior: Team13DiscrepancyPriorKind::Flat,
            discrepancy_prior_precision_scale: 1e-12,
            field_prior_precision_scale: 1e-12,
            field_priors: vec![
                Team13FieldPriorKind::UnweightedHodgeMatern,
                Team13FieldPriorKind::SplitGraphHodgeMatern,
            ],
            b_observation_std_tesla: 0.02,
            nominal_observation_csv_path: None,
            perturbed_observation_csv_path: None,
            output_dir: None,
            skip_output: false,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Exact,
                    num_variance_probes: 32,
                    variance_batch_count: 4,
                    rng_seed: 115,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: true,
            },
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let workspace = workspace_root()?;
    let geo_path = absolutize(&workspace, &config.geo_path);
    let mesh_path = config
        .mesh_path
        .clone()
        .map(|path| absolutize(&workspace, &path))
        .unwrap_or_else(|| {
            workspace.join(format!(
                "target/team13_linear/team13_{}_linear.msh",
                config.domain_mode.as_str()
            ))
        });
    if config.force_mesh_generation || !mesh_path.exists() {
        generate_mesh(&geo_path, &mesh_path, config.domain_mode, config.mesh_scale)?;
    }

    let output_dir = if config.skip_output {
        None
    } else {
        Some(
            config
                .output_dir
                .clone()
                .map(|path| absolutize(&workspace, &path))
                .unwrap_or_else(|| {
                    workspace.join(format!(
                        "out/examples/team13_source_recovery/{}",
                        config.domain_mode.as_str()
                    ))
                }),
        )
    };

    let result = run_team13_source_recovery_experiment(&Team13SourceRecoveryConfig {
        mesh_path: mesh_path.clone(),
        domain_mode: config.domain_mode,
        ampere_turns: config.ampere_turns,
        source_alpha_true: config.source_alpha_true,
        source_alpha_true_modes: config.source_alpha_true_modes,
        run_eight_mode_recovery: config.run_eight_mode_recovery,
        source_prior_std: config.source_prior_std,
        pde_variance: config.pde_variance,
        pde_residual_weighting: config.pde_residual_weighting,
        discrepancy_prior: config.discrepancy_prior,
        discrepancy_prior_precision_scale: config.discrepancy_prior_precision_scale,
        field_prior_precision_scale: config.field_prior_precision_scale,
        field_priors: config.field_priors.clone(),
        b_observation_std_tesla: config.b_observation_std_tesla,
        nominal_observation_csv_path: config
            .nominal_observation_csv_path
            .clone()
            .map(|path| absolutize(&workspace, &path)),
        perturbed_observation_csv_path: config
            .perturbed_observation_csv_path
            .clone()
            .map(|path| absolutize(&workspace, &path)),
        output_dir: output_dir.clone(),
        solver: config.solver,
        ..Team13SourceRecoveryConfig::default()
    })?;

    println!("TEAM 13 field UQ and source recovery");
    println!("  domain: {}", config.domain_mode.as_str());
    println!("  mesh: {}", mesh_path.display());
    println!(
        "  linear BH: nu_air = {NU_AIR:.12e}, nu_iron = {NU_IRON:.12e}, mu_r_iron = {MU_R_IRON:.6}"
    );
    println!(
        "  source alpha (direct field primary): truth {:.6}, posterior {:.6} +/- {:.6}",
        result.source_posterior.true_alpha,
        result.source_posterior.posterior_mean,
        result.source_posterior.posterior_variance.max(0.0).sqrt()
    );
    println!(
        "  source alpha (baseline joint): posterior {:.6} +/- {:.6}",
        result.baseline_source_posterior.posterior_mean,
        result
            .baseline_source_posterior
            .posterior_variance
            .max(0.0)
            .sqrt()
    );
    println!(
        "  source alpha (fluctuation): posterior {:.6} +/- {:.6}",
        result.fluctuation_source_posterior.posterior_mean,
        result
            .fluctuation_source_posterior
            .posterior_variance
            .max(0.0)
            .sqrt()
    );
    println!(
        "  PDE weighting: {}, discrepancy prior: {}, field prior scale {:.3e}, field priors: {}",
        config.pde_residual_weighting.as_str(),
        config.discrepancy_prior.as_str(),
        config.field_prior_precision_scale,
        config
            .field_priors
            .iter()
            .map(|prior| prior.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    for comparison in &result.field_prior_comparisons {
        println!(
            "  prior {}: source posterior {:.6} +/- {:.6}, sensor rmse {:.6e}, B vector rel err {:.6e}, finite={} nonnegative={}",
            comparison.prior_kind.as_str(),
            comparison.source_posterior.posterior_mean,
            comparison.source_posterior.posterior_variance.max(0.0).sqrt(),
            comparison.sensor_rmse,
            comparison.b_vector_relative_l2_error,
            comparison.all_finite_variances,
            comparison.nonnegative_variances,
        );
    }
    if let Some(eight_mode) = &result.eight_mode {
        println!(
            "  eight-mode decision: {} (pass={})",
            eight_mode.decision.recommendation, eight_mode.decision.technical_pass
        );
        for mode in &eight_mode.fluctuation_source_modes {
            println!(
                "    fluctuation {}: truth {:.6}, posterior {:.6} +/- {:.6}, z {:.3}",
                mode.mode_name,
                mode.true_alpha,
                mode.posterior_mean,
                mode.posterior_variance.max(0.0).sqrt(),
                mode.z_score,
            );
        }
    }
    for stage in &result.stages {
        println!(
            "  {}: dim {}, residual {:.6e}, sensor rmse {:.6e}, B rel err {:.6e}, B vector rel err {:.6e}, B vector ratio mean {:.6e}",
            stage.summary.name,
            stage.summary.latent_dimension,
            stage.summary.pde_residual_norm,
            stage.summary.sensor_rmse,
            stage.summary.b_cochain_relative_l2_error,
            stage.summary.b_vector_relative_l2_error,
            stage.summary.b_vector_trace_variance_ratio_mean,
        );
    }
    if let Some(output_dir) = output_dir {
        println!("  outputs: {}", output_dir.display());
    }
    Ok(())
}

fn parse_args() -> Result<ExampleConfig, Box<dyn Error>> {
    let mut config = ExampleConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--domain" => config.domain_mode = parse_domain(next_value(&mut args, "--domain")?)?,
            "--mesh" => config.mesh_path = Some(PathBuf::from(next_value(&mut args, "--mesh")?)),
            "--geo" => config.geo_path = PathBuf::from(next_value(&mut args, "--geo")?),
            "--mesh-scale" => {
                config.mesh_scale =
                    parse_f64(next_value(&mut args, "--mesh-scale")?, "--mesh-scale")?;
            }
            "--force-mesh" => config.force_mesh_generation = true,
            "--ampere-turns" => {
                config.ampere_turns =
                    parse_f64(next_value(&mut args, "--ampere-turns")?, "--ampere-turns")?;
            }
            "--source-alpha-true" | "--source-alpha" => {
                config.source_alpha_true = parse_f64(
                    next_value(&mut args, "--source-alpha-true")?,
                    "--source-alpha-true",
                )?;
            }
            "--eight-mode" => config.run_eight_mode_recovery = true,
            "--source-alphas" => {
                config.source_alpha_true_modes =
                    parse_source_alpha_modes(next_value(&mut args, "--source-alphas")?)?;
                config.run_eight_mode_recovery = true;
            }
            "--source-prior-std" => {
                config.source_prior_std = parse_f64(
                    next_value(&mut args, "--source-prior-std")?,
                    "--source-prior-std",
                )?;
            }
            "--pde-variance" => {
                config.pde_variance =
                    parse_f64(next_value(&mut args, "--pde-variance")?, "--pde-variance")?;
            }
            "--pde-weighting" => {
                config.pde_residual_weighting =
                    parse_pde_weighting(next_value(&mut args, "--pde-weighting")?)?;
            }
            "--discrepancy-prior" => {
                config.discrepancy_prior =
                    parse_discrepancy_prior(next_value(&mut args, "--discrepancy-prior")?)?;
            }
            "--discrepancy-prior-scale" => {
                config.discrepancy_prior_precision_scale = parse_f64(
                    next_value(&mut args, "--discrepancy-prior-scale")?,
                    "--discrepancy-prior-scale",
                )?;
            }
            "--field-prior-scale" | "--field-prior-precision-scale" => {
                config.field_prior_precision_scale = parse_f64(
                    next_value(&mut args, "--field-prior-scale")?,
                    "--field-prior-scale",
                )?;
            }
            "--field-priors" => {
                config.field_priors = parse_field_priors(next_value(&mut args, "--field-priors")?)?;
            }
            "--b-observation-std-tesla" => {
                config.b_observation_std_tesla = parse_f64(
                    next_value(&mut args, "--b-observation-std-tesla")?,
                    "--b-observation-std-tesla",
                )?;
            }
            "--nominal-observations-csv" => {
                config.nominal_observation_csv_path = Some(PathBuf::from(next_value(
                    &mut args,
                    "--nominal-observations-csv",
                )?));
            }
            "--perturbed-observations-csv" => {
                config.perturbed_observation_csv_path = Some(PathBuf::from(next_value(
                    &mut args,
                    "--perturbed-observations-csv",
                )?));
            }
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_value(&mut args, "--output-dir")?));
            }
            "--skip-output" => config.skip_output = true,
            "--variance-mode" => {
                config.solver.variance.mode =
                    parse_variance_mode(next_value(&mut args, "--variance-mode")?)?;
            }
            "--variance-probes" => {
                config.solver.variance.num_variance_probes = parse_usize(
                    next_value(&mut args, "--variance-probes")?,
                    "--variance-probes",
                )?;
            }
            "--variance-batches" => {
                config.solver.variance.variance_batch_count = parse_usize(
                    next_value(&mut args, "--variance-batches")?,
                    "--variance-batches",
                )?;
            }
            "--rng-seed" => {
                config.solver.variance.rng_seed =
                    parse_u64(next_value(&mut args, "--rng-seed")?, "--rng-seed")?;
            }
            "--quiet-diagnostics" => config.solver.log_diagnostics = false,
            "--no-stabilize-precision" => {
                config.solver.precision_policy = LinearPdePrecisionPolicy::default();
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument `{arg}`")).into()),
        }
    }
    Ok(config)
}

fn generate_mesh(
    geo_path: &Path,
    mesh_path: &Path,
    mode: Team13DomainMode,
    mesh_scale: f64,
) -> Result<(), Box<dyn Error>> {
    if !geo_path.exists() {
        return Err(invalid_input(format!(
            "geometry file `{}` does not exist",
            geo_path.display()
        ))
        .into());
    }
    if let Some(parent) = mesh_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let full_domain = match mode {
        Team13DomainMode::HalfZNonnegative => "0",
        Team13DomainMode::Full => "1",
    };
    let status = Command::new("gmsh")
        .arg("-3")
        .arg(geo_path)
        .arg("-setnumber")
        .arg("FullDomain")
        .arg(full_domain)
        .arg("-setnumber")
        .arg("MeshScale")
        .arg(format!("{mesh_scale:.12}"))
        .arg("-o")
        .arg(mesh_path)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "gmsh failed while generating `{}`",
            mesh_path.display()
        ))
        .into());
    }
    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| invalid_input("could not resolve workspace root").into())
}

fn absolutize(workspace: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

fn parse_domain(value: String) -> Result<Team13DomainMode, Box<dyn Error>> {
    match value.as_str() {
        "half" | "half-z" | "z>=0" => Ok(Team13DomainMode::HalfZNonnegative),
        "full" => Ok(Team13DomainMode::Full),
        _ => Err(invalid_input("domain must be `half` or `full`").into()),
    }
}

fn parse_variance_mode(value: String) -> Result<LinearPdeVarianceMode, Box<dyn Error>> {
    match value.as_str() {
        "exact" => Ok(LinearPdeVarianceMode::Exact),
        "exact-solves" => Ok(LinearPdeVarianceMode::ExactSolves),
        "hutchinson" => Ok(LinearPdeVarianceMode::Hutchinson),
        "local-rbmc" => Ok(LinearPdeVarianceMode::LocalRbmc),
        "monte-carlo" => Ok(LinearPdeVarianceMode::MonteCarlo),
        "selected-inverse" => Ok(LinearPdeVarianceMode::SelectedInverse),
        _ => Err(invalid_input(
            "variance mode must be exact, exact-solves, hutchinson, local-rbmc, monte-carlo, or selected-inverse",
        )
        .into()),
    }
}

fn parse_pde_weighting(value: String) -> Result<Team13PdeResidualWeighting, Box<dyn Error>> {
    match value.as_str() {
        "euclidean" => Ok(Team13PdeResidualWeighting::Euclidean),
        "mass-inverse" | "mass" | "mass-weighted" => Ok(Team13PdeResidualWeighting::MassInverse),
        "mass-inverse-trace-normalized" | "mass-trace-normalized" | "mass-normalized" => {
            Ok(Team13PdeResidualWeighting::MassInverseTraceNormalized)
        }
        _ => Err(invalid_input(
            "PDE weighting must be euclidean, mass-inverse, or mass-inverse-trace-normalized",
        )
        .into()),
    }
}

fn parse_discrepancy_prior(value: String) -> Result<Team13DiscrepancyPriorKind, Box<dyn Error>> {
    match value.as_str() {
        "flat" | "none" | "improper" => Ok(Team13DiscrepancyPriorKind::Flat),
        "weighted-whittle" | "whittle" | "matern" | "matérn" => {
            Ok(Team13DiscrepancyPriorKind::WeightedWhittle)
        }
        _ => Err(invalid_input("discrepancy prior must be flat or weighted-whittle").into()),
    }
}

fn parse_field_priors(value: String) -> Result<Vec<Team13FieldPriorKind>, Box<dyn Error>> {
    let mut priors = Vec::new();
    for raw in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let prior = match raw {
            "unweighted" | "unweighted-hodge" | "unweighted-hodge-matern" => {
                Team13FieldPriorKind::UnweightedHodgeMatern
            }
            "split" | "split-graph" | "split-graph-hodge-matern" => {
                Team13FieldPriorKind::SplitGraphHodgeMatern
            }
            _ => {
                return Err(invalid_input(format!(
                    "field prior must be unweighted or split-graph, got `{raw}`"
                ))
                .into());
            }
        };
        if priors.contains(&prior) {
            return Err(invalid_input(format!(
                "field prior `{}` was requested more than once",
                prior.as_str()
            ))
            .into());
        }
        priors.push(prior);
    }
    if priors.is_empty() {
        return Err(invalid_input("at least one field prior is required").into());
    }
    Ok(priors)
}

fn parse_source_alpha_modes(
    value: String,
) -> Result<[f64; TEAM13_COIL_MODE_COUNT], Box<dyn Error>> {
    let fields = value
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    if fields.len() != TEAM13_COIL_MODE_COUNT {
        return Err(invalid_input(format!(
            "--source-alphas requires exactly {TEAM13_COIL_MODE_COUNT} comma-separated values"
        ))
        .into());
    }
    let mut values = [0.0; TEAM13_COIL_MODE_COUNT];
    for (index, field) in fields.iter().enumerate() {
        values[index] = field.parse::<f64>().map_err(|_| {
            invalid_input(format!(
                "invalid floating-point value `{field}` for --source-alphas"
            ))
        })?;
        if !values[index].is_finite() || values[index] <= 0.0 {
            return Err(invalid_input("--source-alphas values must be finite and positive").into());
        }
    }
    Ok(values)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| invalid_input(format!("missing value for {flag}")).into())
}

fn parse_f64(value: String, flag: &'static str) -> Result<f64, Box<dyn Error>> {
    value.parse().map_err(|_| {
        invalid_input(format!("invalid floating-point value `{value}` for {flag}")).into()
    })
}

fn parse_usize(value: String, flag: &'static str) -> Result<usize, Box<dyn Error>> {
    value
        .parse()
        .map_err(|_| invalid_input(format!("invalid integer value `{value}` for {flag}")).into())
}

fn parse_u64(value: String, flag: &'static str) -> Result<u64, Box<dyn Error>> {
    value
        .parse()
        .map_err(|_| invalid_input(format!("invalid integer value `{value}` for {flag}")).into())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn print_usage() {
    println!("Usage: team13_source_recovery [options]");
    println!("  --domain <half|full>");
    println!("  --mesh <path>");
    println!("  --geo <path>");
    println!("  --mesh-scale <value>");
    println!("  --force-mesh");
    println!("  --output-dir <path>");
    println!("  --skip-output");
    println!("  --ampere-turns <value>");
    println!("  --source-alpha-true <value>");
    println!("  --eight-mode");
    println!("  --source-alphas <a,b,c,d,e,f,g,h>");
    println!("  --source-prior-std <value>");
    println!("  --pde-variance <value>");
    println!("  --pde-weighting <euclidean|mass-inverse|mass-inverse-trace-normalized>");
    println!("  --discrepancy-prior <flat|weighted-whittle>");
    println!("  --discrepancy-prior-scale <value>");
    println!("  --field-prior-scale <value>");
    println!("  --field-priors <unweighted,split-graph>");
    println!("  --b-observation-std-tesla <value>");
    println!("  --nominal-observations-csv <path>");
    println!("  --perturbed-observations-csv <path>");
    println!("  --variance-mode <exact|hutchinson|local-rbmc|monte-carlo|selected-inverse>");
    println!("  --variance-probes <count>");
    println!("  --variance-batches <count>");
    println!("  --rng-seed <value>");
    println!("  --quiet-diagnostics");
    println!("  --no-stabilize-precision");
}
