use feg_case_studies::genus2_topological_inverse::{
    run_genus2_topological_inverse_problem, write_genus2_topological_inverse_outputs,
    Genus2TopologicalInverseConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let config = Genus2TopologicalInverseConfig::default();
    let out_dir = PathBuf::from("out/genus2_topological_inverse");

    let result = run_genus2_topological_inverse_problem(&config)?;
    write_genus2_topological_inverse_outputs(&result, &out_dir)?;

    println!("Genus-2 topological inverse problem");
    println!("mesh={}", config.mesh_path.display());
    println!(
        "kappa={} tau={} local_observations={} local_noise_std={} loop_noise_std={} posterior_sample_count={} surface_vector_variance_probe_count={} rng_seed={}",
        config.kappa,
        config.tau,
        config.local_observation_count,
        config.local_noise_std,
        config.loop_noise_std,
        config.posterior_sample_count,
        config.surface_vector_variance_probe_count,
        config.rng_seed
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
    for scenario in &result.scenarios {
        let mean_ratio = mean(
            scenario
                .period_summaries
                .iter()
                .map(|summary| summary.variance_ratio),
        );
        println!(
            "scenario={} observations={} loop_count={} mean_period_variance_ratio={:.6e}",
            scenario.scenario.as_str(),
            scenario.observation_count,
            scenario.scenario.loop_count(),
            mean_ratio
        );
    }
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());
    Ok(())
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}
