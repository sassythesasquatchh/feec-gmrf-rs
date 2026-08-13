use feg_case_studies::cube_voids_coexact_transform::{
    run_cube_voids_coexact_inference, CubeVoidsCoexactInferenceConfig,
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/cube_voids_coexact_inference");
    fs::create_dir_all(&output_dir)?;
    let config = CubeVoidsCoexactInferenceConfig {
        output_dir: output_dir.clone(),
        mesh_path: output_dir.join("cube_voids_coexact_inference.msh"),
        geo_path: output_dir.join("cube_voids_coexact_inference.geo"),
        force_mesh: true,
        ..CubeVoidsCoexactInferenceConfig::default()
    };
    let rows = run_cube_voids_coexact_inference(&config)?;
    let csv_path = output_dir.join("coexact_inference_summary_3d.csv");
    let mut writer = BufWriter::new(File::create(&csv_path)?);
    writeln!(
        writer,
        "sparse_inverse,truth_source,observation_design,vertices,edges,faces,cells,b0,b1,b2,observation_count,truth_m1_norm,posterior_m1_norm,relative_m1_error,mass_correlation,observation_rms_residual,truth_lumped_delta_leakage,posterior_lumped_delta_leakage,posterior_lumped_delta_over_truth_norm,truth_weak_delta_leakage,posterior_weak_delta_leakage,posterior_weak_delta_over_truth_norm"
    )?;
    for row in &rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.sparse_inverse,
            row.truth_source.as_str(),
            row.observation_design.as_str(),
            row.vertex_count,
            row.edge_count,
            row.face_count,
            row.cell_count,
            row.b0,
            row.b1,
            row.b2,
            row.observation_count,
            row.truth_m1_norm,
            row.posterior_m1_norm,
            row.relative_m1_error,
            row.mass_correlation,
            row.observation_rms_residual,
            row.truth_lumped_delta_leakage,
            row.posterior_lumped_delta_leakage,
            row.posterior_lumped_delta_over_truth_norm,
            row.truth_weak_delta_leakage,
            row.posterior_weak_delta_leakage,
            row.posterior_weak_delta_over_truth_norm,
        )?;
    }
    writer.flush()?;

    println!("Cube voids coexact Bayesian inference");
    if let Some(first) = rows.first() {
        println!(
            "mesh: vertices={} edges={} faces={} cells={} b0={} b1={} b2={}",
            first.vertex_count,
            first.edge_count,
            first.face_count,
            first.cell_count,
            first.b0,
            first.b1,
            first.b2
        );
    }
    for row in &rows {
        println!(
            "inverse={} truth={} obs={} nobs={} E1={:.6e} corr={:.6e} rms_obs={:.6e} posterior_lumped_delta={:.6e} posterior_weak_delta={:.6e}",
            row.sparse_inverse,
            row.truth_source.as_str(),
            row.observation_design.as_str(),
            row.observation_count,
            row.relative_m1_error,
            row.mass_correlation,
            row.observation_rms_residual,
            row.posterior_lumped_delta_leakage,
            row.posterior_weak_delta_leakage,
        );
    }
    println!("csv={}", csv_path.display());
    println!("runtime={:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
