use ddf::cochain::Cochain;
use manifold::geometry::coord::mesh::MeshCoords;
use manifold::topology::complex::Complex;
use std::io;
use std::path::Path;

pub use formoniq::io::{write_polyline_vtk_fields, write_polyline_vtu_fields};

/// Write multiple 0-cochain scalar fields into a single VTK file.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_0cochain_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    formoniq::io::write_cochain_vtk_fields(path, coords, topology, 0, fields)
}

/// Write multiple 0-cochain scalar fields into a single VTU file.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_0cochain_vtu_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    formoniq::io::write_cochain_vtu_fields(path, coords, topology, 0, fields)
}

/// Write multiple 1-cochain scalar fields into a single VTK file.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_1cochain_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    formoniq::io::write_cochain_vtk_fields(path, coords, topology, 1, fields)
}

/// Write multiple 1-cochain scalar fields into a single VTU file.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_1cochain_vtu_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    formoniq::io::write_cochain_vtu_fields(path, coords, topology, 1, fields)
}

/// Write a 1-form vector proxy plus additional 1-cochain scalar fields.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_1form_vector_proxy_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vector_cochain: &Cochain,
    scalar_fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    formoniq::io::write_1form_vector_proxy_vtk_fields(
        path,
        coords,
        topology,
        vector_name,
        vector_cochain,
        scalar_fields,
    )
}

/// Write a 1-form vector proxy plus additional 1-cochain scalar fields to VTU.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_1form_vector_proxy_vtu_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vector_cochain: &Cochain,
    scalar_fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    formoniq::io::write_1form_vector_proxy_vtu_fields(
        path,
        coords,
        topology,
        vector_name,
        vector_cochain,
        scalar_fields,
    )
}

/// Write multiple scalar fields defined on top-dimensional cells.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_top_cell_scalar_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    formoniq::io::write_top_cell_vtk_fields(path, coords, topology, &[], fields)
}

/// Write multiple scalar fields defined on top-dimensional cells to VTU.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_top_cell_scalar_vtu_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    formoniq::io::write_top_cell_vtu_fields(path, coords, topology, &[], fields)
}

/// Write a vector field plus scalar fields defined on top-dimensional cells.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_top_cell_vector_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vectors: &[[f64; 3]],
    scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    formoniq::io::write_top_cell_vtk_fields(
        path,
        coords,
        topology,
        &[(vector_name, vectors)],
        scalar_fields,
    )
}

/// Write a vector field plus scalar fields defined on top-dimensional cells to VTU.
///
/// This compatibility wrapper delegates to the canonical FEEC writer.
pub fn write_top_cell_vector_vtu_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vectors: &[[f64; 3]],
    scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    formoniq::io::write_top_cell_vtu_fields(
        path,
        coords,
        topology,
        &[(vector_name, vectors)],
        scalar_fields,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::Vector as FeecVector;
    use manifold::gen::cartesian::CartesianMeshInfo;
    use std::fs;

    #[test]
    fn write_0cochain_vtk_fields_writes_multiple_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let vertex_count = coords.nvertices();

        let a = Cochain::new(0, FeecVector::from_element(vertex_count, 1.0));
        let b = Cochain::new(0, FeecVector::from_element(vertex_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_0cochain_fields_{stamp}.vtk"));

        write_0cochain_vtk_fields(&path, &coords, &topology, &[("a", &a), ("b", &b)])
            .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("POINT_DATA"));
        assert!(content.contains("SCALARS a double 1"));
        assert!(content.contains("SCALARS b double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1cochain_vtk_fields_writes_multiple_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();

        let a = Cochain::new(1, FeecVector::from_element(edge_count, 1.0));
        let b = Cochain::new(1, FeecVector::from_element(edge_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1cochain_fields_{stamp}.vtk"));

        write_1cochain_vtk_fields(&path, &coords, &topology, &[("a", &a), ("b", &b)])
            .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("CELL_DATA"));
        assert!(content.contains("SCALARS a double 1"));
        assert!(content.contains("SCALARS b double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1cochain_vtk_fields_preserves_tiny_values_in_scientific_notation() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();
        let tiny = Cochain::new(1, FeecVector::from_element(edge_count, 1.0e-15));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1cochain_tiny_fields_{stamp}.vtk"));

        write_1cochain_vtk_fields(&path, &coords, &topology, &[("tiny", &tiny)])
            .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("1.000000000000e-15"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1form_vector_proxy_vtk_fields_writes_vectors_and_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();

        let vec = Cochain::new(1, FeecVector::from_element(edge_count, 1.0));
        let var = Cochain::new(1, FeecVector::from_element(edge_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1form_proxy_fields_{stamp}.vtk"));

        write_1form_vector_proxy_vtk_fields(
            &path,
            &coords,
            &topology,
            "proxy",
            &vec,
            &[("var", &var)],
        )
        .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("VECTORS proxy double"));
        assert!(content.contains("SCALARS var double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_top_cell_vector_vtk_fields_writes_vectors_and_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let cell_count = topology.cells().len();
        let vectors = vec![[1.0, 0.0, 0.5]; cell_count];
        let a = vec![0.25; cell_count];
        let b = vec![0.75; cell_count];

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_top_cell_vector_fields_{stamp}.vtk"));

        write_top_cell_vector_vtk_fields(
            &path,
            &coords,
            &topology,
            "cell_vectors",
            &vectors,
            &[("a", &a), ("b", &b)],
        )
        .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("VECTORS cell_vectors double"));
        assert!(content.contains("SCALARS a double 1"));
        assert!(content.contains("SCALARS b double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_0cochain_vtu_fields_writes_multiple_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let vertex_count = coords.nvertices();

        let a = Cochain::new(0, FeecVector::from_element(vertex_count, 1.0));
        let b = Cochain::new(0, FeecVector::from_element(vertex_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_0cochain_fields_{stamp}.vtu"));

        write_0cochain_vtu_fields(&path, &coords, &topology, &[("a", &a), ("b", &b)])
            .expect("vtu write should succeed");

        let content = fs::read_to_string(&path).expect("vtu should be readable");
        assert!(content.contains("<VTKFile type=\"UnstructuredGrid\""));
        assert!(content.contains("<PointData Scalars=\"a\">"));
        assert!(content.contains("Name=\"a\" NumberOfComponents=\"1\""));
        assert!(content.contains("Name=\"b\" NumberOfComponents=\"1\""));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1cochain_vtu_fields_preserves_tiny_values() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();
        let tiny = Cochain::new(1, FeecVector::from_element(edge_count, 1.0e-15));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1cochain_tiny_fields_{stamp}.vtu"));

        write_1cochain_vtu_fields(&path, &coords, &topology, &[("tiny", &tiny)])
            .expect("vtu write should succeed");

        let content = fs::read_to_string(&path).expect("vtu should be readable");
        assert!(content.contains("<CellData Scalars=\"tiny\">"));
        assert!(content.contains("1.000000000000e-15"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1form_vector_proxy_vtu_fields_writes_vectors_and_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();

        let vec = Cochain::new(1, FeecVector::from_element(edge_count, 1.0));
        let var = Cochain::new(1, FeecVector::from_element(edge_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1form_proxy_fields_{stamp}.vtu"));

        write_1form_vector_proxy_vtu_fields(
            &path,
            &coords,
            &topology,
            "proxy",
            &vec,
            &[("var", &var)],
        )
        .expect("vtu write should succeed");

        let content = fs::read_to_string(&path).expect("vtu should be readable");
        assert!(content.contains("<CellData Scalars=\"var\" Vectors=\"proxy\">"));
        assert!(content.contains("Name=\"proxy\" NumberOfComponents=\"3\""));
        assert!(content.contains("Name=\"var\" NumberOfComponents=\"1\""));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_top_cell_vector_vtu_fields_writes_vectors_and_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let cell_count = topology.cells().len();
        let vectors = vec![[1.0, 0.0, 0.5]; cell_count];
        let a = vec![0.25; cell_count];
        let b = vec![0.75; cell_count];

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_top_cell_vector_fields_{stamp}.vtu"));

        write_top_cell_vector_vtu_fields(
            &path,
            &coords,
            &topology,
            "cell_vectors",
            &vectors,
            &[("a", &a), ("b", &b)],
        )
        .expect("vtu write should succeed");

        let content = fs::read_to_string(&path).expect("vtu should be readable");
        assert!(content.contains("<CellData Scalars=\"a\" Vectors=\"cell_vectors\">"));
        assert!(content.contains("Name=\"cell_vectors\" NumberOfComponents=\"3\""));
        assert!(content.contains("Name=\"a\" NumberOfComponents=\"1\""));
        assert!(content.contains("Name=\"b\" NumberOfComponents=\"1\""));

        let _ = fs::remove_file(path);
    }
}
