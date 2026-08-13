//! Torus-specific Gmsh input generation for maintained studies.

use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

pub fn generate_surface_mesh(
    mesh_path: &Path,
    major_radius: f64,
    minor_radius: f64,
    mesh_size: f64,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if mesh_size <= 0.0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "mesh_size must be positive").into(),
        );
    }

    let geo_path = mesh_path.with_extension("geo");
    let geo = format!(
        "SetFactory(\"OpenCASCADE\");\n\
         R = {major_radius};\n\
         r = {minor_radius};\n\
         h = {mesh_size};\n\
         Mesh.ElementOrder = 1;\n\
         Mesh.RecombineAll = 0;\n\
         Mesh.MshFileVersion = 4.1;\n\
         Mesh.MeshSizeMin = h;\n\
         Mesh.MeshSizeMax = h;\n\
         Torus(1) = {{0, 0, 0, R, r, 2*Pi}};\n\
         Mesh 2;\n\
         Save \"{mesh_file}\";\n",
        mesh_file = mesh_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "invalid mesh file name")
            })?,
    );
    fs::create_dir_all(
        geo_path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid mesh path"))?,
    )?;
    fs::write(&geo_path, geo)?;

    let output = Command::new("gmsh")
        .arg("-2")
        .arg(&geo_path)
        .arg("-format")
        .arg("msh41")
        .arg("-o")
        .arg(mesh_path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "gmsh failed for {}: {}",
            geo_path.display(),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }

    Ok(mesh_path.to_path_buf())
}

pub fn mesh_size_tag(mesh_size: f64) -> String {
    format!("{mesh_size:.5}").replace('.', "p")
}
