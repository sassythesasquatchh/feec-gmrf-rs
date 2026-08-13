use feg_case_studies::sphere_nc1_matern_spectral_reference::{
    run_sphere_nc1_matern_spectral_reference, SphereNc1MaternSpectralReferenceConfig,
};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let result = run_sphere_nc1_matern_spectral_reference(&config)?;

    println!("Sphere NC1 1-form Matern spectral-reference validation");
    println!(
        "level,cells,selected,model,state_dofs,precision_nnz,factor_nnz,rel_frob,diag_rel_l2,scaled_rel_frob,best_scale,same_level_gap"
    );
    for row in &result.rows {
        println!(
            "{},{},{},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            row.refinement_level,
            row.cell_count,
            row.selected_cell_count,
            row.model.as_str(),
            row.state_dofs,
            row.precision_nnz,
            row.factor_nnz,
            row.relative_frobenius_error,
            row.diagonal_relative_l2_error,
            row.best_scalar_rescaled_relative_frobenius_error,
            row.best_scalar,
            row.same_level_relative_gap_vs_full_nc1
        );
    }
    println!(
        "summary={}",
        config.output_dir.join("summary.csv").display()
    );
    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<SphereNc1MaternSpectralReferenceConfig, String> {
    let mut config = SphereNc1MaternSpectralReferenceConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--levels" => {
                let value = next_value(&mut args, "--levels")?;
                config.refinement_levels = parse_levels(&value)?;
            }
            "--out" | "--output-dir" => {
                config.output_dir = PathBuf::from(next_value(&mut args, arg.as_str())?);
            }
            "--kappa" => {
                config.kappa = parse_f64(next_value(&mut args, "--kappa")?, "--kappa")?;
            }
            "--tau" => {
                config.tau = parse_f64(next_value(&mut args, "--tau")?, "--tau")?;
            }
            "--lmax" | "--analytic-lmax" => {
                config.analytic_lmax =
                    parse_usize(next_value(&mut args, arg.as_str())?, arg.as_str())?;
            }
            "--max-cells" => {
                config.max_cells =
                    parse_usize(next_value(&mut args, "--max-cells")?, "--max-cells")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help")),
        }
    }
    Ok(config)
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn parse_levels(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_usize(part.trim().to_string(), "--levels"))
        .collect()
}

fn parse_usize(value: String, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid unsigned integer value `{value}` for {flag}"))
}

fn parse_f64(value: String, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("invalid floating-point value `{value}` for {flag}"))
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example sphere_nc1_matern_spectral_reference -- [options]"
    );
    println!();
    println!("Options:");
    println!("  --levels <n,n,...>       Sphere refinement levels (default: 1,2,3,4)");
    println!("  --out <path>             Output directory");
    println!("  --kappa <value>          Positive Matern kappa (default: 1)");
    println!("  --tau <value>            Positive Matern tau (default: 1)");
    println!("  --analytic-lmax <n>      Analytic spherical harmonic cutoff (default: 35)");
    println!("  --lmax <n>               Alias for --analytic-lmax");
    println!(
        "  --max-cells <n>          Maximum selected barycenter cells per level (default: 32)"
    );
}
