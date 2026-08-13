use feg_case_studies::genus2_1form_hodge_conditioning::{
    run_genus2_1form_hodge_conditioning, write_genus2_1form_hodge_conditioning_outputs,
    Genus2Torus1FormHodgeConditioningConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let config = Genus2Torus1FormHodgeConditioningConfig::default();
    let out_dir = PathBuf::from("out/matern_1form_genus2_hodge_conditioning");

    let result = run_genus2_1form_hodge_conditioning(&config)?;
    write_genus2_1form_hodge_conditioning_outputs(&result, &out_dir)?;

    println!("Genus-2 torus 1-form Hodge-sector conditioning");
    println!("mesh={}", config.mesh_path.display());
    println!(
        "kappa={} tau={} noise_variance={} harmonic_dim={} sample_count={} rng_seed={} raw_sample_seed={}",
        config.kappa,
        config.tau,
        config.noise_variance,
        config.harmonic_dim,
        config.sample_count,
        config.rng_seed,
        config.raw_sample_seed
    );
    println!(
        "topology: vertices={} edges={} faces={} chi={} b0={} b1={} b2={}",
        result.topology_summary.vertex_count,
        result.topology_summary.edge_count,
        result.topology_summary.face_count,
        result.topology_summary.euler_characteristic,
        result.topology_summary.b0,
        result.topology_summary.b1,
        result.topology_summary.b2
    );
    println!("cycle observations={}", result.cycle_observations.len());
    print_branch_summary(&result.exact);
    print_branch_summary(&result.coexact);
    print_branch_summary(&result.harmonic);
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn print_branch_summary(branch: &feg_infer::conditioning::hodge_1form::Hodge1FormBranchResult) {
    println!("branch={}", branch.kind.as_str());
    println!(
        "  latent_dimension={} max_abs_observation_error={} mean_abs_observation_error={}",
        branch.latent_dimension,
        branch.max_abs_observation_error,
        branch.mean_abs_observation_error
    );
    println!(
        "  harmonic_residual_norm_truth={} harmonic_residual_norm_posterior_mean={}",
        branch.harmonic_residual_norm_truth, branch.harmonic_residual_norm_posterior_mean
    );
    println!(
        "  sample_count prior={} posterior={}",
        branch.prior_samples.len(),
        branch.posterior_samples.len()
    );
}
