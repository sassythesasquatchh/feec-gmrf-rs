use feg_case_studies::team13::{
    solve_team13_linear_nominal, solve_team13_linear_uq, Team13BenchmarkReport, Team13DomainMode,
    Team13LinearConfig, Team13MeasurementMode, MU_R_IRON, NU_AIR, NU_IRON,
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
    ampere_turns: f64,
    coil_relative_std: f64,
    pde_variance: f64,
    b_observation_std_tesla: f64,
    measurement_mode: Team13MeasurementMode,
    legacy_measurement_band: f64,
    observation_csv_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    skip_output: bool,
    nominal_only: bool,
    solver: LinearPdeUqSolverConfig,
    force_mesh_generation: bool,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            domain_mode: Team13DomainMode::HalfZNonnegative,
            mesh_path: None,
            geo_path: PathBuf::from("geometries/team13_linear.geo"),
            mesh_scale: 8.0,
            ampere_turns: 1000.0,
            coil_relative_std: 0.05,
            pde_variance: 1e-8,
            b_observation_std_tesla: 0.02,
            measurement_mode: Team13MeasurementMode::BenchmarkExact,
            legacy_measurement_band: 0.03,
            observation_csv_path: None,
            output_dir: None,
            skip_output: false,
            nominal_only: false,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Exact,
                    num_variance_probes: 32,
                    variance_batch_count: 4,
                    rng_seed: 13,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: true,
            },
            force_mesh_generation: false,
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
        config
            .output_dir
            .clone()
            .map(|path| absolutize(&workspace, &path))
            .or_else(|| {
                Some(workspace.join(format!(
                    "out/examples/team13_linear_uq/{}",
                    config.domain_mode.as_str()
                )))
            })
    };

    let solve_config = Team13LinearConfig {
        mesh_path: mesh_path.clone(),
        domain_mode: config.domain_mode,
        ampere_turns: config.ampere_turns,
        coil_relative_std: config.coil_relative_std,
        pde_variance: config.pde_variance,
        b_observation_std_tesla: config.b_observation_std_tesla,
        measurement_mode: config.measurement_mode,
        legacy_measurement_band: config.legacy_measurement_band,
        observation_csv_path: config
            .observation_csv_path
            .clone()
            .map(|path| absolutize(&workspace, &path)),
        output_dir,
        solver: config.solver,
    };
    if config.nominal_only {
        let result = solve_team13_linear_nominal(&solve_config)?;
        println!("TEAM 13 linear FEEC nominal solve");
        println!("  domain: {}", result.domain_mode.as_str());
        println!("  mesh: {}", mesh_path.display());
        println!(
            "  linear BH: nu_air = {NU_AIR:.12e}, nu_iron = {NU_IRON:.12e}, mu_r_iron = {MU_R_IRON:.6}"
        );
        println!(
            "  model: weighted Coulomb-gauge mixed Hodge-Laplacian with unweighted physical source pairing"
        );
        println!(
            "  reference outputs: {}, rmse(1-25, nominal) = {:.6e} T",
            result.benchmark_reports.len(),
            benchmark_rmse(&result.benchmark_reports, true)
        );
        if let Some(path) = &solve_config.observation_csv_path {
            println!("  observations: {}", path.display());
        }
        if let Some(output_dir) = &solve_config.output_dir {
            println!("  outputs: {}", output_dir.display());
        }
    } else {
        let result = solve_team13_linear_uq(&solve_config)?;

        println!("TEAM 13 linear FEEC/GMRF solve");
        println!("  domain: {}", result.domain_mode.as_str());
        println!("  mesh: {}", mesh_path.display());
        println!(
            "  linear BH: nu_air = {NU_AIR:.12e}, nu_iron = {NU_IRON:.12e}, mu_r_iron = {MU_R_IRON:.6}"
        );
        println!(
            "  model: weighted Coulomb-gauge mixed Hodge-Laplacian with unweighted physical source pairing"
        );
        println!(
            "  sensors: {} (mode: {}), rmse = {:.6e} T",
            result.sensor_reports.len(),
            solve_config.measurement_mode.as_str(),
            sensor_rmse(&result.sensor_reports)
        );
        if let Some(path) = &solve_config.observation_csv_path {
            println!("  observations: {}", path.display());
        }
        println!(
            "  reference outputs: {}, rmse(1-25, nominal) = {:.6e} T",
            result.benchmark_reports.len(),
            benchmark_rmse(&result.benchmark_reports, true)
        );
        for input in &result.posterior.latent_inputs {
            let (min_mean, max_mean) = finite_min_max(input.mean.iter().copied());
            println!(
                "  latent input {} mean: avg {:.6e}, range [{:.6e}, {:.6e}]",
                input.name,
                finite_mean(input.mean.iter().copied()),
                min_mean,
                max_mean
            );
        }
        println!(
            "  A posterior/prior variance ratio mean: {:.6e}",
            finite_mean(result.a_variance_ratio.iter().copied())
        );
        print_variance_stats(
            "A prior variance",
            result.posterior.prior_variance.as_slice(),
        );
        print_masked_variance_stats(
            "A prior variance (active DOFs)",
            result.posterior.prior_variance.as_slice(),
            result.state_active_mask.as_slice(),
        );
        print_variance_stats(
            "A posterior variance",
            result.posterior.posterior_variance.as_slice(),
        );
        print_masked_variance_stats(
            "A posterior variance (active DOFs)",
            result.posterior.posterior_variance.as_slice(),
            result.state_active_mask.as_slice(),
        );
        println!(
            "  B cochain posterior/prior variance ratio mean: {:.6e}",
            finite_mean(result.b_variance_ratio.iter().copied())
        );
        if let Some(b_variance) = result.posterior.derived_variances.get("B_cochain") {
            print_variance_stats(
                "B cochain prior variance",
                b_variance.prior_variance.as_slice(),
            );
            print_variance_stats(
                "B cochain posterior variance",
                b_variance.posterior_variance.as_slice(),
            );
        }
        for (name, pushforward) in &result.vector_pushforwards {
            println!(
                "  {name} vector trace posterior/prior variance ratio mean: {:.6e}",
                finite_mean(pushforward.trace_variance_ratio.iter().copied())
            );
            print_variance_stats(
                &format!("{name} vector prior trace variance"),
                pushforward.prior_trace_variance.as_slice(),
            );
            print_variance_stats(
                &format!("{name} vector posterior trace variance"),
                pushforward.posterior_trace_variance.as_slice(),
            );
        }
        if let Some(output_dir) = &solve_config.output_dir {
            println!("  outputs: {}", output_dir.display());
        }
    }

    Ok(())
}

fn parse_args() -> Result<ExampleConfig, Box<dyn Error>> {
    let mut config = ExampleConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--domain" => {
                config.domain_mode = parse_domain(next_value(&mut args, "--domain")?)?;
            }
            "--mesh" => {
                config.mesh_path = Some(PathBuf::from(next_value(&mut args, "--mesh")?));
            }
            "--geo" => {
                config.geo_path = PathBuf::from(next_value(&mut args, "--geo")?);
            }
            "--mesh-scale" => {
                config.mesh_scale =
                    parse_f64(next_value(&mut args, "--mesh-scale")?, "--mesh-scale")?;
            }
            "--ampere-turns" => {
                config.ampere_turns =
                    parse_f64(next_value(&mut args, "--ampere-turns")?, "--ampere-turns")?;
            }
            "--coil-relative-std" => {
                config.coil_relative_std = parse_f64(
                    next_value(&mut args, "--coil-relative-std")?,
                    "--coil-relative-std",
                )?;
            }
            "--pde-variance" => {
                config.pde_variance =
                    parse_f64(next_value(&mut args, "--pde-variance")?, "--pde-variance")?;
            }
            "--b-observation-std-tesla" => {
                config.b_observation_std_tesla = parse_f64(
                    next_value(&mut args, "--b-observation-std-tesla")?,
                    "--b-observation-std-tesla",
                )?;
            }
            "--measurement-mode" => {
                config.measurement_mode =
                    parse_measurement_mode(next_value(&mut args, "--measurement-mode")?)?;
            }
            "--legacy-measurement-band" | "--measurement-band" => {
                config.legacy_measurement_band = parse_f64(
                    next_value(&mut args, "--legacy-measurement-band")?,
                    "--legacy-measurement-band",
                )?;
                config.measurement_mode = Team13MeasurementMode::LegacyBand;
            }
            "--observations-csv" | "--observation-csv" => {
                config.observation_csv_path =
                    Some(PathBuf::from(next_value(&mut args, "--observations-csv")?));
            }
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_value(&mut args, "--output-dir")?));
            }
            "--skip-output" => {
                config.skip_output = true;
            }
            "--nominal-only" => {
                config.nominal_only = true;
            }
            "--force-mesh" => {
                config.force_mesh_generation = true;
            }
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
            "--quiet-diagnostics" => {
                config.solver.log_diagnostics = false;
            }
            "--no-stabilize-precision" => {
                config.solver.precision_policy = LinearPdePrecisionPolicy::default();
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                return Err(invalid_input(format!("unknown argument `{arg}`")).into());
            }
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
        .ok_or_else(|| {
            invalid_input("could not resolve workspace root from CARGO_MANIFEST_DIR").into()
        })
}

fn absolutize(workspace: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

fn sensor_rmse(reports: &[feg_case_studies::team13::Team13SensorReport]) -> f64 {
    if reports.is_empty() {
        return 0.0;
    }
    let mse = reports
        .iter()
        .map(|report| report.residual * report.residual)
        .sum::<f64>()
        / reports.len() as f64;
    mse.sqrt()
}

fn benchmark_rmse(reports: &[Team13BenchmarkReport], use_nominal: bool) -> f64 {
    let squared = reports
        .iter()
        .filter_map(|report| {
            report.observed.map(|observed| {
                let prediction = if use_nominal {
                    report.nominal_prediction
                } else {
                    report.posterior_prediction
                };
                let residual = prediction - observed;
                residual * residual
            })
        })
        .collect::<Vec<_>>();
    if squared.is_empty() {
        return 0.0;
    }
    (squared.iter().sum::<f64>() / squared.len() as f64).sqrt()
}

fn finite_mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn finite_min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if min.is_infinite() || max.is_infinite() {
        (0.0, 0.0)
    } else {
        (min, max)
    }
}

fn print_variance_stats(label: &str, values: &[f64]) {
    let positive = values.iter().filter(|value| **value > 0.0).count();
    let zeros = values.iter().filter(|value| **value == 0.0).count();
    let (min_value, max_value) = finite_min_max(values.iter().copied());
    println!(
        "  {label}: positive {positive}/{}, zero {}, range [{:.6e}, {:.6e}]",
        values.len(),
        zeros,
        min_value,
        max_value
    );
}

fn print_masked_variance_stats(label: &str, values: &[f64], mask: &[f64]) {
    assert_eq!(
        values.len(),
        mask.len(),
        "variance values and mask must have matching lengths"
    );
    let filtered = values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, mask_value)| {
            if *mask_value > 0.0 {
                Some(*value)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    print_variance_stats(label, &filtered);
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

fn parse_measurement_mode(value: String) -> Result<Team13MeasurementMode, Box<dyn Error>> {
    match value.as_str() {
        "benchmark" | "exact" => Ok(Team13MeasurementMode::BenchmarkExact),
        "legacy" | "legacy-band" => Ok(Team13MeasurementMode::LegacyBand),
        _ => Err(invalid_input(
            "measurement mode must be exact, benchmark, legacy, or legacy-band",
        )
        .into()),
    }
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
    println!("Usage: team13_linear_uq [options]");
    println!("  --domain <half|full>");
    println!("  --mesh <path>");
    println!("  --geo <path>");
    println!("  --mesh-scale <value>");
    println!("  --force-mesh");
    println!("  --output-dir <path>");
    println!("  --skip-output");
    println!("  --nominal-only");
    println!("  --ampere-turns <value>");
    println!("  --coil-relative-std <value>");
    println!("  --pde-variance <value>");
    println!("  --b-observation-std-tesla <value>");
    println!("  --measurement-mode <exact|legacy-band>");
    println!("  --legacy-measurement-band <value>");
    println!("  --observations-csv <path>");
    println!("  --variance-mode <exact|hutchinson|local-rbmc|monte-carlo|selected-inverse>");
    println!("  --variance-probes <count>");
    println!("  --variance-batches <count>");
    println!("  --rng-seed <value>");
    println!("  --quiet-diagnostics");
    println!("  --no-stabilize-precision");
}
