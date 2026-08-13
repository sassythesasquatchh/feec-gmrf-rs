use feg_case_studies::sphere_sparse_anchor_branch_conditioning::{
    compute_sphere_sparse_anchor_branch_conditioning, BranchPosteriorDiagnostics,
    SphereSparseAnchorBranchConditioningConfig,
};
use feg_infer::prior::matern::MaternAlpha;
use std::env;

fn main() -> Result<(), String> {
    let config = parse_args(env::args().skip(1).collect())?;
    let report = compute_sphere_sparse_anchor_branch_conditioning(config)?;
    println!(
        "mesh: level={}, vertices={}, edges={}, faces={}, observations={}, noise_variance={:.3e}",
        report.level,
        report.vertex_count,
        report.edge_count,
        report.face_count,
        report.observation_count,
        report.noise_variance
    );
    println!(
        "truth_mass_norms: exact={:.6e}, coexact={:.6e}, mixed={:.6e}",
        report.truth_exact_mass_norm, report.truth_coexact_mass_norm, report.truth_mixed_mass_norm
    );
    println!("label,obs_residual_rel,mass_norm,closure_rel,coclosed_rel");
    for diagnostics in report.diagnostics() {
        print_diagnostics(diagnostics);
    }
    Ok(())
}

fn print_diagnostics(diagnostics: &BranchPosteriorDiagnostics) {
    println!(
        "{},{:.6e},{:.6e},{:.6e},{:.6e}",
        diagnostics.label,
        diagnostics.observation_residual_relative,
        diagnostics.mass_norm,
        diagnostics.closure_residual_relative,
        diagnostics.coclosed_residual_relative
    );
}

fn parse_args(args: Vec<String>) -> Result<SphereSparseAnchorBranchConditioningConfig, String> {
    let mut config = SphereSparseAnchorBranchConditioningConfig::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--level" => {
                index += 1;
                config.level = parse_usize(arg(&args, index, "--level")?, "--level")?;
            }
            "--observations" | "--observation-count" => {
                index += 1;
                config.observation_count =
                    parse_usize(arg(&args, index, "--observations")?, "--observations")?;
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
            "--noise-variance" => {
                index += 1;
                config.noise_variance =
                    parse_f64(arg(&args, index, "--noise-variance")?, "--noise-variance")?;
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
        "Usage: cargo run --release -p feg-case-studies --example sphere_sparse_anchor_branch_conditioning -- [options]\n\
         Options:\n\
           --level <usize>             Sphere subdivision level (default: 2)\n\
           --observations <usize>      Edge cochain observations (default: 12)\n\
           --kappa <f64>               Branch kappa (default: 1.0)\n\
           --tau <f64>                 Branch tau (default: 1.0)\n\
           --alpha <1|2>               Integer Matern alpha (default: 2)\n\
           --noise-variance <f64>      Observation noise variance (default: 1e-6)"
    );
}
