use feg_case_studies::experimental::c51_thesis_normalized_compact::{
    run_c51_thesis_normalized_compact, C51ThesisNormalizedCompactConfig,
};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() != 3 {
        return Err("usage: c51_thesis_normalized_compact <refinement> <observation-count> <output-directory>".into());
    }
    let refinement = arguments[0].parse::<usize>()?;
    let observation_count = arguments[1].parse::<usize>()?;
    let output = PathBuf::from(&arguments[2]);
    let config = C51ThesisNormalizedCompactConfig::development(refinement, observation_count)?;
    let report = run_c51_thesis_normalized_compact(&config)?;
    report.write_directory(&output)?;
    let smallest_noise = report
        .weak_residual_rows
        .last()
        .expect("development config has at least one residual-noise level");
    println!(
        "C51 development result: n={refinement}, m={observation_count}, target B_RMS={:.6e} T, exact B_RMS={:.6e} T, compact B_RMS={:.6e} T, local held-out posterior error={:.3}%, surface held-out posterior error={:.3}%\nweak PDE at sigma={:.1e}: exact/deterministic={:.3e}, compact/deterministic={:.3e}\nfield-wide posterior/prior ratio errors: latent median/p90/max={:.3}%/{:.3}%/{:.3}%, flux median/p90/max={:.3}%/{:.3}%/{:.3}%\nwrote ordinary development CSVs to {}",
        config.target_b_rms_tesla,
        report.exact.normalized_b_rms,
        report.compact.normalized_b_rms,
        100.0 * report.local_posterior_held_out.primary_envelope(),
        100.0 * report.surface_posterior_held_out.primary_envelope(),
        smallest_noise.residual_standard_deviation,
        smallest_noise.exact_augmented_deterministic_relative_l2_error,
        smallest_noise.compact_deterministic_relative_l2_error,
        100.0 * report.latent_field_wide.posterior_prior_ratio_error.median_relative_error,
        100.0 * report
            .latent_field_wide
            .posterior_prior_ratio_error
            .percentile_90_relative_error,
        100.0 * report
            .latent_field_wide
            .posterior_prior_ratio_error
            .maximum_relative_error,
        100.0 * report.flux_field_wide.posterior_prior_ratio_error.median_relative_error,
        100.0 * report
            .flux_field_wide
            .posterior_prior_ratio_error
            .percentile_90_relative_error,
        100.0 * report
            .flux_field_wide
            .posterior_prior_ratio_error
            .maximum_relative_error,
        output.display(),
    );
    Ok(())
}
