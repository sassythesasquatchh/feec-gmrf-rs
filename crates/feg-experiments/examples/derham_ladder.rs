use feg_case_studies::derham_ladder::{run_derham_ladder_experiment, DerhamLadderConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let config = DerhamLadderConfig::default();
    let result = run_derham_ladder_experiment(&config)?;

    println!("De Rham ladder experiment");
    println!("mesh={}", config.mesh_path.display());
    println!("output_dir={}", config.output_dir.display());
    println!(
        "topology: vertices={} edges={} faces={} cells={} betti={:?}",
        result.mesh.vertices,
        result.mesh.edges,
        result.mesh.faces,
        result.mesh.cells,
        result.mesh.betti
    );
    for grade in &result.grades {
        println!(
            "k={} dim={} obs={} tau={:.6} residual={:.3e} hodge_recon={:.3e}",
            grade.grade,
            grade.dimension,
            grade.observation_count,
            grade.tau,
            grade.max_abs_observation_residual,
            grade.hodge_reconstruction_error
        );
    }
    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
