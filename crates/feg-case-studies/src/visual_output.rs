use ddf::cochain::Cochain;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};
use std::io;
use std::path::{Path, PathBuf};

pub use feec_gmrf::report::{CochainVtuBuilder, TopCellVtuBuilder, VectorLayout3};

/// Return the canonical VTU path for a visual artifact.
///
/// Callers may still pass legacy `.vtk` stems while thesis visual outputs
/// migrate to XML VTU.
pub fn vtu_path(path: impl AsRef<Path>) -> PathBuf {
    let mut path = path.as_ref().to_path_buf();
    path.set_extension("vtu");
    path
}

pub fn write_cochain(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    cochain: &Cochain,
    name: &str,
) -> io::Result<()> {
    let mut builder = CochainVtuBuilder::new(cochain.dim());
    builder
        .add_cochain(name, cochain.clone())
        .map_err(report_io)?;
    builder
        .write(vtu_path(path), coords, topology)
        .map_err(report_io)
}

pub fn write_0cochain_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    write_cochain_fields(path, coords, topology, 0, fields)
}

pub fn write_1cochain_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    write_cochain_fields(path, coords, topology, 1, fields)
}

pub fn write_1form_vector_field(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    cochain: &Cochain,
    name: &str,
) -> io::Result<()> {
    formoniq::io::write_1form_vector_field_vtu(vtu_path(path), coords, topology, cochain, name)
}

pub fn write_1form_vector_proxy_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vector_cochain: &Cochain,
    scalar_fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    feg_infer::vtk::write_1form_vector_proxy_vtu_fields(
        vtu_path(path),
        coords,
        topology,
        vector_name,
        vector_cochain,
        scalar_fields,
    )
}

pub fn write_2form_vector_field(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    cochain: &Cochain,
    name: &str,
) -> io::Result<()> {
    formoniq::io::write_2form_vector_field_vtu(vtu_path(path), coords, topology, cochain, name)
}

pub fn write_top_cell_scalar_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let mut builder = TopCellVtuBuilder::new();
    for (name, values) in fields {
        builder
            .add_scalar(*name, values.to_vec())
            .map_err(report_io)?;
    }
    builder
        .write(vtu_path(path), coords, topology)
        .map_err(report_io)
}

pub fn write_top_cell_vector_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vectors: &[[f64; 3]],
    scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let mut builder = TopCellVtuBuilder::new();
    builder
        .add_vector(vector_name, vectors.to_vec())
        .map_err(report_io)?;
    for (name, values) in scalar_fields {
        builder
            .add_scalar(*name, values.to_vec())
            .map_err(report_io)?;
    }
    builder
        .write(vtu_path(path), coords, topology)
        .map_err(report_io)
}

pub fn write_top_cell_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_fields: &[(&str, &[[f64; 3]])],
    scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let mut builder = TopCellVtuBuilder::new();
    for (name, values) in vector_fields {
        builder
            .add_vector(*name, values.to_vec())
            .map_err(report_io)?;
    }
    for (name, values) in scalar_fields {
        builder
            .add_scalar(*name, values.to_vec())
            .map_err(report_io)?;
    }
    builder
        .write(vtu_path(path), coords, topology)
        .map_err(report_io)
}

fn write_cochain_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    degree: usize,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    let mut builder = CochainVtuBuilder::new(degree);
    for (name, cochain) in fields {
        builder
            .add_cochain(*name, (*cochain).clone())
            .map_err(report_io)?;
    }
    builder
        .write(vtu_path(path), coords, topology)
        .map_err(report_io)
}

fn report_io(error: feec_gmrf::FeecGmrfError) -> io::Error {
    match error {
        feec_gmrf::FeecGmrfError::Io(error) => error,
        error => io::Error::other(error),
    }
}

pub fn write_polyline_fields(
    path: impl AsRef<Path>,
    title: &str,
    coords: &MeshCoords,
    paths: &[&[usize]],
    cell_scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    formoniq::io::write_polyline_vtu_fields(
        vtu_path(path),
        title,
        coords,
        paths,
        cell_scalar_fields,
    )
}
