use feg_case_studies::sphere_sparse_anchor_kernel_validation::{
    compute_sphere_sparse_anchor_kernel_validation, SphereSparseAnchorKernelBranchReport,
    SphereSparseAnchorKernelValidationConfig,
};
use feg_infer::prior::matern::MaternAlpha;
use std::env;

fn main() -> Result<(), String> {
    let config = parse_args(env::args().skip(1).collect())?;
    let report = compute_sphere_sparse_anchor_kernel_validation(config)?;
    println!("Sparse-anchor and spectral Hodge-Matern sphere kernel validation");
    println!(
        "level,cells,selected,method,target,branch,rel_frob,diag_rel_l2,scaled_rel_frob,best_scale,model_trace,analytic_trace"
    );
    for level in report.levels {
        for branch in level.branch_reports() {
            print_branch(
                level.refinement_level,
                level.cell_count,
                level.selected_cell_count,
                branch,
            );
        }
    }
    Ok(())
}

fn print_branch(
    level: usize,
    cells: usize,
    selected: usize,
    branch: &SphereSparseAnchorKernelBranchReport,
) {
    let branch_label = branch.branch.map_or("joint", |branch| branch.as_str());
    println!(
        "{level},{cells},{selected},{},{},{branch_label},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
        branch.method.as_str(),
        branch.target.as_str(),
        branch.relative_frobenius_error,
        branch.diagonal_relative_l2_error,
        branch.best_scalar_rescaled_relative_frobenius_error,
        branch.best_scalar,
        branch.model_trace,
        branch.analytic_trace
    );
}

fn parse_args(args: Vec<String>) -> Result<SphereSparseAnchorKernelValidationConfig, String> {
    let mut config = SphereSparseAnchorKernelValidationConfig::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--levels" => {
                index += 1;
                config.refinement_levels = parse_levels(arg(&args, index, "--levels")?)?;
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
            "--lmax" => {
                index += 1;
                config.analytic_lmax = parse_usize(arg(&args, index, "--lmax")?, "--lmax")?;
            }
            "--analytic-lmax" => {
                index += 1;
                config.analytic_lmax =
                    parse_usize(arg(&args, index, "--analytic-lmax")?, "--analytic-lmax")?;
            }
            "--spectral-lmax" => {
                index += 1;
                config.spectral_lmax =
                    parse_usize(arg(&args, index, "--spectral-lmax")?, "--spectral-lmax")?;
            }
            "--max-cells" => {
                index += 1;
                config.max_cells = parse_usize(arg(&args, index, "--max-cells")?, "--max-cells")?;
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

fn parse_levels(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|part| parse_usize(part, "--levels"))
        .collect()
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
        "Usage: cargo run --release -p feg-case-studies --example sphere_sparse_anchor_kernel_validation -- [options]\n\
         Options:\n\
           --levels <list>      Comma-separated sphere refinement levels (default: 1,2)\n\
           --kappa <f64>        SPDE kappa in (kappa^2 + lambda)^-alpha (default: 1.0)\n\
           --tau <f64>          SPDE tau (default: 1.0)\n\
           --alpha <1|2>        Integer Matern alpha (default: 2)\n\
           --lmax <usize>       Alias for --analytic-lmax\n\
           --analytic-lmax <usize>  Full analytic spherical harmonic cutoff (default: 35)\n\
           --spectral-lmax <usize>  Matched spectral analytic cutoff (default: 4)\n\
           --max-cells <usize>  Maximum barycenter cells sampled per level (default: 24)"
    );
}
