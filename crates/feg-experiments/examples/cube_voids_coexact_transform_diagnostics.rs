use feg_case_studies::cube_voids_coexact_transform::{
    run_cube_voids_coexact_transform_diagnostics, CubeVoidsCoexactTransformConfig,
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/cube_voids_coexact_transform_diagnostics");
    fs::create_dir_all(&output_dir)?;
    let config = CubeVoidsCoexactTransformConfig {
        output_dir: output_dir.clone(),
        mesh_path: output_dir.join("cube_voids_coexact_transform_diagnostics.msh"),
        geo_path: output_dir.join("cube_voids_coexact_transform_diagnostics.geo"),
        force_mesh: true,
        ..CubeVoidsCoexactTransformConfig::default()
    };
    let diagnostics = run_cube_voids_coexact_transform_diagnostics(&config)?;
    let csv_path = output_dir.join("coexact_transform_diagnostics_3d.csv");
    let mut writer = BufWriter::new(File::create(&csv_path)?);
    writeln!(
        writer,
        "sparse_inverse,vertices,edges,faces,cells,b0,b1,b2,sparse_coexact_m1_operator_norm,exact_mass_coexact_m1_operator_norm,sparse_coexact_codifferential_leakage,exact_mass_coexact_codifferential_leakage,sparse_exact_branch_mass_orthogonality,exact_mass_exact_branch_mass_orthogonality,sparse_vs_exact_mass_transform_relative_m1_error,sparse_coexact_rank,exact_mass_coexact_rank,principal_cosine_min,principal_cosine_mean,principal_angle_max_degrees"
    )?;
    for row in &diagnostics {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{:.12},{:.12},{:.12}",
            row.sparse_inverse,
            row.vertex_count,
            row.edge_count,
            row.face_count,
            row.cell_count,
            row.b0,
            row.b1,
            row.b2,
            row.sparse_coexact_m1_operator_norm,
            row.exact_mass_coexact_m1_operator_norm,
            row.sparse_coexact_codifferential_leakage,
            row.exact_mass_coexact_codifferential_leakage,
            row.sparse_exact_branch_mass_orthogonality,
            row.exact_mass_exact_branch_mass_orthogonality,
            row.sparse_vs_exact_mass_transform_relative_m1_error,
            row.sparse_coexact_rank,
            row.exact_mass_coexact_rank,
            row.principal_cosine_min,
            row.principal_cosine_mean,
            row.principal_angle_max_degrees,
        )?;
    }
    writer.flush()?;

    println!("Cube voids coexact transform diagnostics");
    if let Some(first) = diagnostics.first() {
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
    for row in &diagnostics {
        println!(
            "inverse={} leakage={:.6e} orthogonality={:.6e} rel_m1_error={:.6e} min_cos={:.6e} mean_cos={:.6e} max_angle_deg={:.6e}",
            row.sparse_inverse,
            row.sparse_coexact_codifferential_leakage,
            row.sparse_exact_branch_mass_orthogonality,
            row.sparse_vs_exact_mass_transform_relative_m1_error,
            row.principal_cosine_min,
            row.principal_cosine_mean,
            row.principal_angle_max_degrees
        );
    }
    if let Some(first) = diagnostics.first() {
        println!(
            "exact-mass reference: leakage={:.6e} orthogonality={:.6e}",
            first.exact_mass_coexact_codifferential_leakage,
            first.exact_mass_exact_branch_mass_orthogonality
        );
    }
    println!("csv={}", csv_path.display());
    Ok(())
}
