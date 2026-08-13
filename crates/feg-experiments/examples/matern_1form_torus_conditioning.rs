use feg_case_studies::torus::one_form_conditioning::{
    run_torus_1form_conditioning, write_torus_1form_conditioning_outputs,
    Torus1FormConditioningConfig,
};
use feg_infer::prior::matern::MaternAlpha;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let (config, out_dir) = parse_args()?;

    let result = run_torus_1form_conditioning(&config)?;
    write_torus_1form_conditioning_outputs(&result, &out_dir)?;

    println!("Torus 1-form Matérn conditioning");
    println!("mesh={}", config.mesh_path.display());
    println!(
        "alpha={} kappa={} tau={} noise_variance={} variance_probes={} variance_batches={} seed={}",
        config.alpha.as_u32(),
        config.kappa,
        config.tau,
        config.noise_variance,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed
    );
    println!(
        "effective_range={} neighbourhood_radius={} far_radius={}",
        result.effective_range, result.neighbourhood_radius, result.far_radius
    );
    println!("observations={}", result.selected_observations.len());
    print_branch_summary(
        &result.harmonic_free_constrained,
        result.neighbourhood_radius,
        result.far_radius,
    );
    print_branch_summary(
        &result.full_unconstrained,
        result.neighbourhood_radius,
        result.far_radius,
    );
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn parse_args() -> Result<(Torus1FormConditioningConfig, PathBuf), Box<dyn std::error::Error>> {
    let mut config = Torus1FormConditioningConfig::default();
    let mut out_dir = PathBuf::from("out/matern_1form_torus_conditioning");
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--alpha" => {
                i += 1;
                config.alpha = parse_next::<MaternAlpha>(&args, i, "--alpha")?;
            }
            "--kappa" => {
                i += 1;
                config.kappa = parse_next::<f64>(&args, i, "--kappa")?;
            }
            "--tau" => {
                i += 1;
                config.tau = parse_next::<f64>(&args, i, "--tau")?;
            }
            "--noise-variance" => {
                i += 1;
                config.noise_variance = parse_next::<f64>(&args, i, "--noise-variance")?;
            }
            "--samples" => {
                i += 1;
                config.num_variance_probes = parse_next::<usize>(&args, i, "--samples")?;
            }
            "--variance-batches" => {
                i += 1;
                config.variance_batch_count = parse_next::<usize>(&args, i, "--variance-batches")?;
            }
            "--seed" => {
                i += 1;
                config.rng_seed = parse_next::<u64>(&args, i, "--seed")?;
            }
            "--out-dir" => {
                i += 1;
                out_dir = PathBuf::from(parse_next::<String>(&args, i, "--out-dir")?);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            flag => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown flag: {flag}"),
                )
                .into());
            }
        }
        i += 1;
    }
    Ok((config, out_dir))
}

fn parse_next<T: std::str::FromStr>(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    <T as std::str::FromStr>::Err: ToString,
{
    args.get(idx)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("missing value for {flag}"),
            )
        })?
        .parse::<T>()
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid value for {flag}: {}", err.to_string()),
            )
            .into()
        })
}

fn print_usage() {
    println!("Usage: matern_1form_torus_conditioning [options]");
    println!("  --alpha <1|2>");
    println!("  --kappa <float>");
    println!("  --tau <float>");
    println!("  --noise-variance <float>");
    println!("  --samples <usize>");
    println!("  --variance-batches <usize>");
    println!("  --seed <u64>");
    println!("  --out-dir <path>");
}

fn print_branch_summary(
    branch: &feg_case_studies::torus::one_form_conditioning::Torus1FormBranchResult,
    neighbourhood_radius: f64,
    far_radius: f64,
) {
    let (near_obs_dev, far_obs_dev) =
        observation_relative_stats(branch, neighbourhood_radius, far_radius);
    println!("branch={}", branch.name);
    println!(
        "  observed max abs error={} variance ratio={}",
        branch.summary.observed.max_abs_error, branch.summary.observed.variance_ratio_mean
    );
    println!(
        "  near observation deviation={} variance ratio={}",
        near_obs_dev, branch.summary.near.variance_ratio_mean
    );
    println!(
        "  far observation deviation={} variance ratio={}",
        far_obs_dev, branch.summary.far.variance_ratio_mean
    );
    println!(
        "  harmonic coeff truth=[{}, {}] posterior=[{}, {}]",
        branch.harmonic_coefficients_truth[0],
        branch.harmonic_coefficients_truth[1],
        branch.harmonic_coefficients_posterior_mean[0],
        branch.harmonic_coefficients_posterior_mean[1],
    );
}

fn observation_relative_stats(
    branch: &feg_case_studies::torus::one_form_conditioning::Torus1FormBranchResult,
    neighbourhood_radius: f64,
    far_radius: f64,
) -> (f64, f64) {
    let mut near_sum = 0.0;
    let mut near_count = 0_usize;
    let mut far_sum = 0.0;
    let mut far_count = 0_usize;

    for edge_index in 0..branch.posterior_mean.len() {
        let distance = branch.nearest_observation_distance[edge_index];
        let deviation = (branch.posterior_mean[edge_index]
            - branch.nearest_observation_value[edge_index])
            .abs();
        if distance <= neighbourhood_radius {
            near_sum += deviation;
            near_count += 1;
        }
        if distance > far_radius {
            far_sum += deviation;
            far_count += 1;
        }
    }

    (
        safe_mean(near_sum, near_count),
        safe_mean(far_sum, far_count),
    )
}

fn safe_mean(sum: f64, count: usize) -> f64 {
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}
