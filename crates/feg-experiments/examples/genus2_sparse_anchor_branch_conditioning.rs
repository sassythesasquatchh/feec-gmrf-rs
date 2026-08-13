use feg_case_studies::genus2_sparse_anchor_branch_conditioning::{
    compute_genus2_sparse_anchor_branch_conditioning, BranchPosteriorDiagnostics,
    Genus2SparseAnchorBranchConditioningConfig,
};
use feg_infer::prior::matern::MaternAlpha;
use std::{env, path::PathBuf};

fn main() -> Result<(), String> {
    let config = parse_args(env::args().skip(1).collect())?;
    let report = compute_genus2_sparse_anchor_branch_conditioning(config.clone())?;
    println!("mesh={}", config.mesh_path.display());
    println!(
        "topology: vertices={} edges={} faces={} chi={} b0={} b1={} b2={}",
        report.topology_summary.vertex_count,
        report.topology_summary.edge_count,
        report.topology_summary.face_count,
        report.topology_summary.euler_characteristic,
        report.topology_summary.b0,
        report.topology_summary.b1,
        report.topology_summary.b2
    );
    println!(
        "observations: total={} local={} cycles={} local_noise_variance={:.3e} cycle_noise_variance={:.3e}",
        report.observation_count,
        report.local_observation_count,
        report.cycle_observation_count,
        report.local_noise_variance,
        report.cycle_noise_variance
    );
    println!(
        "truth_mass_norms: exact={:.6e}, coexact={:.6e}, harmonic={:.6e}, mixed={:.6e}",
        report.truth_exact_mass_norm,
        report.truth_coexact_mass_norm,
        report.truth_harmonic_mass_norm,
        report.truth_mixed_mass_norm
    );
    println!(
        "harmonic: latent_dim={} cycle_harmonic_pairing_rank={} truth_coefficients={:?}",
        report.harmonic_latent_dimension,
        report.cycle_harmonic_pairing_rank,
        report.truth_harmonic_coefficients
    );
    println!(
        "label,obs_residual_rel,mass_norm,closure_rel,coclosed_rel,cycle_period_error_rel,harmonic_dim,harmonic_coeff_error_rel"
    );
    for diagnostics in report.diagnostics() {
        print_diagnostics(diagnostics);
    }
    Ok(())
}

fn print_diagnostics(diagnostics: &BranchPosteriorDiagnostics) {
    println!(
        "{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{}",
        diagnostics.label,
        diagnostics.observation_residual_relative,
        diagnostics.mass_norm,
        diagnostics.closure_residual_relative,
        diagnostics.coclosed_residual_relative,
        diagnostics.cycle_period_error_relative,
        diagnostics
            .harmonic_latent_dimension
            .map(|value| value.to_string())
            .unwrap_or_default(),
        diagnostics
            .harmonic_coefficient_error_relative
            .map(|value| format!("{value:.6e}"))
            .unwrap_or_default()
    );
}

fn parse_args(args: Vec<String>) -> Result<Genus2SparseAnchorBranchConditioningConfig, String> {
    let mut config = Genus2SparseAnchorBranchConditioningConfig::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--mesh" => {
                index += 1;
                config.mesh_path = PathBuf::from(arg(&args, index, "--mesh")?);
            }
            "--local-observations" | "--local-observation-count" => {
                index += 1;
                config.local_observation_count = parse_usize(
                    arg(&args, index, "--local-observations")?,
                    "--local-observations",
                )?;
            }
            "--kappa" => {
                index += 1;
                config.kappa = parse_f64(arg(&args, index, "--kappa")?, "--kappa")?;
            }
            "--tau" => {
                index += 1;
                config.tau = parse_f64(arg(&args, index, "--tau")?, "--tau")?;
            }
            "--alpha" => {
                index += 1;
                config.alpha = parse_alpha(arg(&args, index, "--alpha")?)?;
            }
            "--harmonic-precision" => {
                index += 1;
                config.harmonic_precision = parse_f64(
                    arg(&args, index, "--harmonic-precision")?,
                    "--harmonic-precision",
                )?;
            }
            "--local-noise-variance" => {
                index += 1;
                config.local_noise_variance = parse_f64(
                    arg(&args, index, "--local-noise-variance")?,
                    "--local-noise-variance",
                )?;
            }
            "--cycle-noise-variance" => {
                index += 1;
                config.cycle_noise_variance = parse_f64(
                    arg(&args, index, "--cycle-noise-variance")?,
                    "--cycle-noise-variance",
                )?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
        index += 1;
    }
    Ok(config)
}

fn arg<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_alpha(value: &str) -> Result<MaternAlpha, String> {
    match value {
        "1" => Ok(MaternAlpha::One),
        "2" => Ok(MaternAlpha::Two),
        _ => Err("--alpha currently supports 1 or 2".to_string()),
    }
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} expects an unsigned integer"))
}

fn parse_f64(value: &str, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{flag} expects a floating-point value"))
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example genus2_sparse_anchor_branch_conditioning -- [options]\n\
         Options:\n\
           --mesh <path>                         Genus-2 .msh path\n\
           --local-observations <usize>          Local edge observations (default: 16)\n\
           --kappa <f64>                         Branch kappa (default: 1.0)\n\
           --tau <f64>                           Branch tau (default: 1.0)\n\
           --alpha <1|2>                         Integer Matern alpha (default: 2)\n\
           --harmonic-precision <f64>            Harmonic latent precision (default: 1.0)\n\
           --local-noise-variance <f64>          Local edge noise variance (default: 1e-6)\n\
           --cycle-noise-variance <f64>          Cycle noise variance (default: 1e-8)"
    );
}
