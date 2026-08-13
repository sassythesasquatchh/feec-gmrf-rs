//! FEEC physical-field pushforward adapters.
//!
//! These helpers keep physical vector proxies explicit: first assemble FEEC
//! cochain maps such as exterior derivatives, then apply a Hodge/reconstruction
//! adapter to produce physical reporting quantities.

use crate::{
    linear_pde::LinearPdeDerivedQuantitySpec,
    sparse::{
        restrict_sparse_row_operator_columns, restrict_sparse_row_operator_columns_and_fold_fixed,
        sparse_row_operator_from_feec_csr,
    },
};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use ddf::whitney::lsf::WhitneyLsf;
use ddf::ManifoldComplexExt;
use exterior::field::ExteriorField;
use formoniq::assemble::boundary_orientation_sign;
use formoniq::reduction::DofLayout;
use gmrf_core::SparseRowOperator;
use manifold::{
    geometry::coord::{
        mesh::MeshCoords,
        simplex::{barycenter_local, SimplexHandleExt},
    },
    topology::complex::Complex,
};

const PHYSICAL_PUSHFORWARD_EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct AffineSparseRowOperator {
    pub operator: SparseRowOperator,
    pub bias: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MagneticFluxDensityComponentOperators3d {
    pub x: SparseRowOperator,
    pub y: SparseRowOperator,
    pub z: SparseRowOperator,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MagneticFluxDensityComponentOperators2d {
    pub x: SparseRowOperator,
    pub y: SparseRowOperator,
}

/// Cellwise vector-RMS weights and volume-averaged component maps for a 3D mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticFluxDensityAverages3d {
    pub x: SparseRowOperator,
    pub y: SparseRowOperator,
    pub z: SparseRowOperator,
    /// Cell-major weights aligned with `[Bx0, By0, Bz0, Bx1, ...]`.
    pub vector_rms_weights: Vec<f64>,
    pub domain_volume: f64,
}

/// Normalized lumped P1 mass weights for a scalar field's domain RMS.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarFieldRmsWeights {
    pub vertex_weights: Vec<f64>,
    pub domain_volume: f64,
}

/// FEEC exterior derivative `D0: C^0 -> C^1` as a sparse row operator.
pub fn build_exterior_derivative_0_operator(
    topology: &Complex,
) -> Result<SparseRowOperator, String> {
    let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
    sparse_row_operator_from_feec_csr(&d0)
}

/// FEEC exterior derivative `D1: C^1 -> C^2` as a sparse row operator.
pub fn build_exterior_derivative_1_operator(
    topology: &Complex,
) -> Result<SparseRowOperator, String> {
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    sparse_row_operator_from_feec_csr(&d1)
}

/// Barycenter vector proxy for Whitney 2-form cochains on 3D tetrahedral cells.
///
/// The output is cell-major: `[Bx_cell0, By_cell0, Bz_cell0, Bx_cell1, ...]`.
/// This is the Euclidean Hodge/vector proxy applied after reconstructing the
/// Whitney 2-form at each cell barycenter.
pub fn build_barycenter_2form_vector_proxy_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<SparseRowOperator, String> {
    if topology.dim() != 3 {
        return Err(format!(
            "2-form magnetic vector proxy requires topology dimension 3, got {}",
            topology.dim()
        ));
    }
    if coords.dim() != 3 {
        return Err(format!(
            "2-form magnetic vector proxy requires coordinate dimension 3, got {}",
            coords.dim()
        ));
    }

    let bary_local = barycenter_local(3);
    let mut rows = Vec::with_capacity(3 * topology.cells().len());
    for cell in topology.cells().handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let mut cell_rows = [Vec::new(), Vec::new(), Vec::new()];
        for face in cell.mesh_subsimps(2) {
            let local_face = face.relative_to(&cell);
            let lsf = WhitneyLsf::standard(3, local_face);
            let ambient_value = cell_coords.lift_form(&lsf.at_point(&bary_local));
            let coeffs = ambient_value.coeffs();
            if coeffs.len() != 3 {
                return Err(format!(
                    "expected 3 coefficients for reconstructed 2-form, found {}",
                    coeffs.len()
                ));
            }

            // Euclidean Hodge-star identification of 2-form coefficients with
            // the ambient magnetic flux-density vector components.
            let vector_coefficients = [coeffs[2], -coeffs[1], coeffs[0]];
            for component in 0..3 {
                let coefficient = vector_coefficients[component];
                if coefficient.abs() > PHYSICAL_PUSHFORWARD_EPS {
                    cell_rows[component].push((face.kidx(), coefficient));
                }
            }
        }
        rows.push(std::mem::take(&mut cell_rows[0]));
        rows.push(std::mem::take(&mut cell_rows[1]));
        rows.push(std::mem::take(&mut cell_rows[2]));
    }

    SparseRowOperator::new(topology.nsimplices(2), rows).map_err(|err| err.to_string())
}

pub fn build_barycenter_2form_vector_proxy_component_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    if component_index >= 3 {
        return Err(format!(
            "3D magnetic flux-density component index {component_index} is out of range"
        ));
    }
    let proxy = build_barycenter_2form_vector_proxy_operator_3d(topology, coords)?;
    select_interleaved_component_rows(&proxy, topology.nsimplices(3), 3, component_index)
}

pub fn build_barycenter_2form_vector_proxy_component_operators_3d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<MagneticFluxDensityComponentOperators3d, String> {
    Ok(MagneticFluxDensityComponentOperators3d {
        x: build_barycenter_2form_vector_proxy_component_operator_3d(topology, coords, 0)?,
        y: build_barycenter_2form_vector_proxy_component_operator_3d(topology, coords, 1)?,
        z: build_barycenter_2form_vector_proxy_component_operator_3d(topology, coords, 2)?,
    })
}

/// FEEC magnetic flux-density pushforward `R_B D1` from full 1-form edge
/// cochains to cellwise B-vector values.
pub fn build_full_magnetic_flux_density_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<SparseRowOperator, String> {
    let d1 = build_exterior_derivative_1_operator(topology)?;
    let proxy = build_barycenter_2form_vector_proxy_operator_3d(topology, coords)?;
    SparseRowOperator::compose(&proxy, &d1).map_err(|err| err.to_string())
}

pub fn build_full_magnetic_flux_density_component_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    let d1 = build_exterior_derivative_1_operator(topology)?;
    let proxy = build_barycenter_2form_vector_proxy_component_operator_3d(
        topology,
        coords,
        component_index,
    )?;
    SparseRowOperator::compose(&proxy, &d1).map_err(|err| err.to_string())
}

pub fn build_full_magnetic_flux_density_component_operators_3d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<MagneticFluxDensityComponentOperators3d, String> {
    Ok(MagneticFluxDensityComponentOperators3d {
        x: build_full_magnetic_flux_density_component_operator_3d(topology, coords, 0)?,
        y: build_full_magnetic_flux_density_component_operator_3d(topology, coords, 1)?,
        z: build_full_magnetic_flux_density_component_operator_3d(topology, coords, 2)?,
    })
}

/// Outward magnetic flux through selected boundary faces, composed with `D1`.
///
/// `face_indices` use the global 2-simplex ordering. The result has one row,
/// with each selected face oriented outward from its unique adjacent cell.
pub fn build_outward_boundary_flux_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    face_indices: &[usize],
) -> Result<SparseRowOperator, String> {
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err("outward magnetic flux requires a 3D topology and coordinates".to_string());
    }
    let mut selected = face_indices.to_vec();
    selected.sort_unstable();
    if selected.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("outward magnetic flux face indices must be unique".to_string());
    }
    let mut face_row = Vec::with_capacity(selected.len());
    for face_index in selected {
        if face_index >= topology.nsimplices(2) {
            return Err(format!(
                "boundary face index {face_index} exceeds face count {}",
                topology.nsimplices(2)
            ));
        }
        let face = topology.facets().handle_by_kidx(face_index);
        if face.cocells().count() != 1 {
            return Err(format!("face {face_index} is not a boundary face"));
        }
        face_row.push((face_index, boundary_orientation_sign(face, coords)));
    }
    let face_functional = SparseRowOperator::new(topology.nsimplices(2), vec![face_row])
        .map_err(|error| error.to_string())?;
    SparseRowOperator::compose(
        &face_functional,
        &build_exterior_derivative_1_operator(topology)?,
    )
    .map_err(|error| error.to_string())
}

/// Assemble normalized scalar-field RMS weights from the row sums of the P1
/// mass matrix, equivalently distributing each cell volume equally among its
/// vertices.
pub fn build_scalar_field_rms_weights(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<ScalarFieldRmsWeights, String> {
    if topology.dim() == 0 {
        return Err("scalar-field RMS weights require positive mesh dimension".to_string());
    }
    let mut vertex_weights = vec![0.0; topology.nsimplices(0)];
    let mut domain_volume = 0.0;
    for cell in topology.cells().handle_iter() {
        let volume = cell.coord_simplex(coords).vol();
        if !volume.is_finite() || volume <= 0.0 {
            return Err("mesh contains a non-positive or non-finite cell volume".to_string());
        }
        domain_volume += volume;
        let contribution = volume / (topology.dim() + 1) as f64;
        for vertex in cell.mesh_subsimps(0) {
            vertex_weights[vertex.kidx()] += contribution;
        }
    }
    if !domain_volume.is_finite() || domain_volume <= 0.0 {
        return Err("mesh has non-positive or non-finite volume".to_string());
    }
    for weight in &mut vertex_weights {
        *weight /= domain_volume;
    }
    Ok(ScalarFieldRmsWeights {
        vertex_weights,
        domain_volume,
    })
}

/// Volume-averaged B-component maps and vector-magnitude RMS weights.
pub fn build_magnetic_flux_density_averages_3d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<MagneticFluxDensityAverages3d, String> {
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err("magnetic volume averages require a 3D topology and coordinates".to_string());
    }
    let cell_volumes = topology
        .cells()
        .handle_iter()
        .map(|cell| cell.coord_simplex(coords).vol())
        .collect::<Vec<_>>();
    let domain_volume = cell_volumes.iter().sum::<f64>();
    if !domain_volume.is_finite() || domain_volume <= 0.0 {
        return Err("mesh has non-positive or non-finite volume".to_string());
    }
    let normalized = cell_volumes
        .iter()
        .map(|volume| volume / domain_volume)
        .collect::<Vec<_>>();
    let component_functional = SparseRowOperator::new(
        normalized.len(),
        vec![normalized.iter().copied().enumerate().collect()],
    )
    .map_err(|error| error.to_string())?;
    let components = build_full_magnetic_flux_density_component_operators_3d(topology, coords)?;
    let average = |component: &SparseRowOperator| {
        SparseRowOperator::compose(&component_functional, component)
            .map_err(|error| error.to_string())
    };
    let mut vector_rms_weights = Vec::with_capacity(3 * normalized.len());
    for weight in normalized {
        vector_rms_weights.extend([weight, weight, weight]);
    }
    Ok(MagneticFluxDensityAverages3d {
        x: average(&components.x)?,
        y: average(&components.y)?,
        z: average(&components.z)?,
        vector_rms_weights,
        domain_volume,
    })
}

/// Backward-compatible alias for the full magnetic flux-density pushforward.
pub fn build_magnetic_flux_density_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<SparseRowOperator, String> {
    build_full_magnetic_flux_density_operator_3d(topology, coords)
}

/// Reduced FEEC magnetic flux-density pushforward over active 1-form dofs.
///
/// This operator is intended for covariance and zero-boundary workflows. It
/// rejects nonzero fixed dofs because those require an affine bias term in
/// addition to the reduced linear operator.
pub fn build_reduced_magnetic_flux_density_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
) -> Result<SparseRowOperator, String> {
    if layout
        .prescribed_dofs
        .iter()
        .any(|fixed| fixed.value.abs() > PHYSICAL_PUSHFORWARD_EPS)
    {
        return Err(
            "reduced magnetic flux-density operator currently requires zero fixed dofs".to_string(),
        );
    }
    let full = build_full_magnetic_flux_density_operator_3d(topology, coords)?;
    restrict_sparse_row_operator_columns(&full, layout)
}

pub fn build_reduced_magnetic_flux_density_affine_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
) -> Result<AffineSparseRowOperator, String> {
    let full = build_full_magnetic_flux_density_operator_3d(topology, coords)?;
    let bias = vec![0.0; full.nrows()];
    let (operator, bias) =
        restrict_sparse_row_operator_columns_and_fold_fixed(&full, &bias, layout)?;
    Ok(AffineSparseRowOperator { operator, bias })
}

pub fn build_reduced_magnetic_flux_density_component_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    if layout
        .prescribed_dofs
        .iter()
        .any(|fixed| fixed.value.abs() > PHYSICAL_PUSHFORWARD_EPS)
    {
        return Err(
            "reduced magnetic flux-density component operator currently requires zero fixed dofs"
                .to_string(),
        );
    }
    let full =
        build_full_magnetic_flux_density_component_operator_3d(topology, coords, component_index)?;
    restrict_sparse_row_operator_columns(&full, layout)
}

pub fn build_reduced_magnetic_flux_density_component_operators_3d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
) -> Result<MagneticFluxDensityComponentOperators3d, String> {
    Ok(MagneticFluxDensityComponentOperators3d {
        x: build_reduced_magnetic_flux_density_component_operator_3d(topology, coords, layout, 0)?,
        y: build_reduced_magnetic_flux_density_component_operator_3d(topology, coords, layout, 1)?,
        z: build_reduced_magnetic_flux_density_component_operator_3d(topology, coords, layout, 2)?,
    })
}

pub fn build_reduced_magnetic_flux_density_component_affine_operator_3d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
    component_index: usize,
) -> Result<AffineSparseRowOperator, String> {
    let full =
        build_full_magnetic_flux_density_component_operator_3d(topology, coords, component_index)?;
    let bias = vec![0.0; full.nrows()];
    let (operator, bias) =
        restrict_sparse_row_operator_columns_and_fold_fixed(&full, &bias, layout)?;
    Ok(AffineSparseRowOperator { operator, bias })
}

pub fn build_magnetic_flux_density_derived_quantities_3d(
    topology: &Complex,
    coords: &MeshCoords,
    flux_cochain_name: impl Into<String>,
    component_names: [&str; 3],
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    let components = build_full_magnetic_flux_density_component_operators_3d(topology, coords)?;
    Ok(vec![
        LinearPdeDerivedQuantitySpec {
            name: flux_cochain_name.into(),
            operator: build_exterior_derivative_1_operator(topology)?,
        },
        LinearPdeDerivedQuantitySpec {
            name: component_names[0].to_string(),
            operator: components.x,
        },
        LinearPdeDerivedQuantitySpec {
            name: component_names[1].to_string(),
            operator: components.y,
        },
        LinearPdeDerivedQuantitySpec {
            name: component_names[2].to_string(),
            operator: components.z,
        },
    ])
}

/// Barycenter rotation proxy for Whitney 1-form cochains on 2D triangular cells.
///
/// The output is cell-major: `[Bx_cell0, By_cell0, Bx_cell1, ...]`, with
/// `B = (dA_z/dy, -dA_z/dx)` after the `D0` exterior derivative has produced
/// a Whitney 1-form cochain.
pub fn build_barycenter_1form_rotation_proxy_operator_2d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<SparseRowOperator, String> {
    if topology.dim() != 2 {
        return Err(format!(
            "1-form rotation proxy requires topology dimension 2, got {}",
            topology.dim()
        ));
    }
    if coords.dim() != 2 {
        return Err(format!(
            "1-form rotation proxy requires coordinate dimension 2, got {}",
            coords.dim()
        ));
    }

    let bary_local = barycenter_local(2);
    let mut rows = Vec::with_capacity(2 * topology.cells().len());
    for cell in topology.cells().handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let mut cell_rows = [Vec::new(), Vec::new()];
        for edge in cell.mesh_subsimps(1) {
            let local_edge = edge.relative_to(&cell);
            let lsf = WhitneyLsf::standard(2, local_edge);
            let ambient_value = cell_coords.lift_form(&lsf.at_point(&bary_local));
            let coeffs = ambient_value.coeffs();
            if coeffs.len() != 2 {
                return Err(format!(
                    "expected 2 coefficients for reconstructed 1-form, found {}",
                    coeffs.len()
                ));
            }

            let vector_coefficients = [coeffs[1], -coeffs[0]];
            for component in 0..2 {
                let coefficient = vector_coefficients[component];
                if coefficient.abs() > PHYSICAL_PUSHFORWARD_EPS {
                    cell_rows[component].push((edge.kidx(), coefficient));
                }
            }
        }
        rows.push(std::mem::take(&mut cell_rows[0]));
        rows.push(std::mem::take(&mut cell_rows[1]));
    }

    SparseRowOperator::new(topology.nsimplices(1), rows).map_err(|err| err.to_string())
}

pub fn build_barycenter_1form_rotation_component_operator_2d(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    if component_index >= 2 {
        return Err(format!(
            "2D magnetic flux-density component index {component_index} is out of range"
        ));
    }
    let proxy = build_barycenter_1form_rotation_proxy_operator_2d(topology, coords)?;
    select_interleaved_component_rows(&proxy, topology.nsimplices(2), 2, component_index)
}

pub fn build_scalar_az_magnetic_flux_density_operator_2d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<SparseRowOperator, String> {
    let d0 = build_exterior_derivative_0_operator(topology)?;
    let proxy = build_barycenter_1form_rotation_proxy_operator_2d(topology, coords)?;
    SparseRowOperator::compose(&proxy, &d0).map_err(|err| err.to_string())
}

pub fn build_scalar_az_magnetic_flux_density_component_operator_2d(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    let d0 = build_exterior_derivative_0_operator(topology)?;
    let proxy =
        build_barycenter_1form_rotation_component_operator_2d(topology, coords, component_index)?;
    SparseRowOperator::compose(&proxy, &d0).map_err(|err| err.to_string())
}

pub fn build_scalar_az_magnetic_flux_density_component_operators_2d(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<MagneticFluxDensityComponentOperators2d, String> {
    Ok(MagneticFluxDensityComponentOperators2d {
        x: build_scalar_az_magnetic_flux_density_component_operator_2d(topology, coords, 0)?,
        y: build_scalar_az_magnetic_flux_density_component_operator_2d(topology, coords, 1)?,
    })
}

pub fn build_reduced_scalar_az_magnetic_flux_density_operator_2d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
) -> Result<SparseRowOperator, String> {
    if layout
        .prescribed_dofs
        .iter()
        .any(|fixed| fixed.value.abs() > PHYSICAL_PUSHFORWARD_EPS)
    {
        return Err(
            "reduced scalar-Az magnetic flux-density operator currently requires zero fixed dofs"
                .to_string(),
        );
    }
    let full = build_scalar_az_magnetic_flux_density_operator_2d(topology, coords)?;
    restrict_sparse_row_operator_columns(&full, layout)
}

pub fn build_reduced_scalar_az_magnetic_flux_density_affine_operator_2d(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
) -> Result<AffineSparseRowOperator, String> {
    let full = build_scalar_az_magnetic_flux_density_operator_2d(topology, coords)?;
    let bias = vec![0.0; full.nrows()];
    let (operator, bias) =
        restrict_sparse_row_operator_columns_and_fold_fixed(&full, &bias, layout)?;
    Ok(AffineSparseRowOperator { operator, bias })
}

fn select_interleaved_component_rows(
    operator: &SparseRowOperator,
    item_count: usize,
    component_count: usize,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    if component_index >= component_count {
        return Err(format!(
            "component index {component_index} is out of range for {component_count} components"
        ));
    }
    if operator.nrows() != item_count * component_count {
        return Err(format!(
            "operator row count {} does not match {item_count} items with {component_count} components",
            operator.nrows()
        ));
    }
    let rows = (0..item_count)
        .map(|item| item * component_count + component_index)
        .collect::<Vec<_>>();
    operator.select_rows(&rows).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use formoniq::reduction::PrescribedDof;
    use gmrf_core::types::Vector as GmrfVector;
    use manifold::gen::cartesian::CartesianMeshInfo;

    #[test]
    fn magnetic_flux_density_operator_uses_feec_d1_then_vector_proxy() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let d1 = build_exterior_derivative_1_operator(&topology).unwrap();
        let proxy = build_barycenter_2form_vector_proxy_operator_3d(&topology, &coords).unwrap();
        let composed = build_magnetic_flux_density_operator_3d(&topology, &coords).unwrap();
        let manual = SparseRowOperator::compose(&proxy, &d1).unwrap();

        assert_eq!(d1.ncols, topology.nsimplices(1));
        assert_eq!(d1.nrows(), topology.nsimplices(2));
        assert_eq!(proxy.ncols, topology.nsimplices(2));
        assert_eq!(proxy.nrows(), 3 * topology.nsimplices(3));
        assert_eq!(composed, manual);
    }

    #[test]
    fn boundary_flux_orientation_and_volume_averages_match_constant_field() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let field = exterior::field::DiffFormClosure::one_form(
            |p| common::linalg::nalgebra::Vector::from_column_slice(&[0.0, 0.0, p[1]]),
            3,
        );
        let a = ddf::cochain::cochain_projection(&field, &topology, &coords, None);
        let x1_faces =
            formoniq::assemble::boundary_simplices_where_barycenter(&topology, &coords, 2, |p| {
                (p[0] - 1.0).abs() < 1.0e-12
            });
        let flux = build_outward_boundary_flux_operator_3d(&topology, &coords, &x1_faces).unwrap();
        let value = flux
            .apply(&GmrfVector::from_iterator(
                a.coeffs.len(),
                a.coeffs.iter().copied(),
            ))
            .unwrap();
        assert!((value[0] - 1.0).abs() < 1.0e-10);

        let averages = build_magnetic_flux_density_averages_3d(&topology, &coords).unwrap();
        assert!((averages.vector_rms_weights.iter().sum::<f64>() - 3.0).abs() < 1e-12);
        assert!(
            (averages
                .x
                .apply(&GmrfVector::from_iterator(
                    a.coeffs.len(),
                    a.coeffs.iter().copied()
                ))
                .unwrap()[0]
                - 1.0)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn scalar_rms_weights_integrate_constant_and_linear_fields() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let weights = build_scalar_field_rms_weights(&topology, &coords).unwrap();
        assert!((weights.domain_volume - 1.0).abs() < 1.0e-12);
        assert!((weights.vertex_weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        let mean_x = weights
            .vertex_weights
            .iter()
            .enumerate()
            .map(|(vertex, weight)| weight * coords.coord(vertex)[0])
            .sum::<f64>();
        assert!((mean_x - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn manufactured_em_fluxes_and_volume_averages_have_physical_orientation() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 4, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let c = 0.50 * (45.0_f64 / 29.0).sqrt();
        let field = exterior::field::DiffFormClosure::one_form(
            move |p| {
                common::linalg::nalgebra::Vector::from_column_slice(&[
                    0.0,
                    0.0,
                    c * p[0] * p[1] * p[1],
                ])
            },
            3,
        );
        let a = ddf::cochain::cochain_projection(&field, &topology, &coords, None);
        let a = GmrfVector::from_iterator(a.coeffs.len(), a.coeffs.iter().copied());
        let x1_faces =
            formoniq::assemble::boundary_simplices_where_barycenter(&topology, &coords, 2, |p| {
                (p[0] - 1.0).abs() < 1.0e-12
            });
        let y1_faces =
            formoniq::assemble::boundary_simplices_where_barycenter(&topology, &coords, 2, |p| {
                (p[1] - 1.0).abs() < 1.0e-12
            });
        let flux_x = build_outward_boundary_flux_operator_3d(&topology, &coords, &x1_faces)
            .unwrap()
            .apply(&a)
            .unwrap()[0];
        let flux_y = build_outward_boundary_flux_operator_3d(&topology, &coords, &y1_faces)
            .unwrap()
            .apply(&a)
            .unwrap()[0];
        assert!((flux_x - c).abs() < 1.0e-10);
        assert!((flux_y + c).abs() < 1.0e-10);

        let averages = build_magnetic_flux_density_averages_3d(&topology, &coords).unwrap();
        let bx = averages.x.apply(&a).unwrap()[0];
        let by = averages.y.apply(&a).unwrap()[0];
        let bz = averages.z.apply(&a).unwrap()[0];
        assert!((bx - c / 2.0).abs() < 1.0e-2);
        assert!((by + c / 3.0).abs() < 1.0e-2);
        assert!(bz.abs() < 1.0e-2, "projected mean Bz was {bz}");
    }

    #[test]
    fn reduced_magnetic_flux_density_operator_restricts_to_active_dofs() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let full = build_magnetic_flux_density_operator_3d(&topology, &coords).unwrap();
        let layout = DofLayout::identity(topology.nsimplices(1));
        let reduced =
            build_reduced_magnetic_flux_density_operator_3d(&topology, &coords, &layout).unwrap();
        assert_eq!(reduced, full);
    }

    #[test]
    fn component_operators_select_full_3d_b_components() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let full = build_full_magnetic_flux_density_operator_3d(&topology, &coords).unwrap();
        let components =
            build_full_magnetic_flux_density_component_operators_3d(&topology, &coords).unwrap();

        assert_eq!(
            components.x,
            select_interleaved_component_rows(&full, topology.nsimplices(3), 3, 0).unwrap()
        );
        assert_eq!(
            components.y,
            select_interleaved_component_rows(&full, topology.nsimplices(3), 3, 1).unwrap()
        );
        assert_eq!(
            components.z,
            select_interleaved_component_rows(&full, topology.nsimplices(3), 3, 2).unwrap()
        );
    }

    #[test]
    fn affine_reduced_3d_b_operator_folds_nonzero_fixed_edges() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let full = build_full_magnetic_flux_density_operator_3d(&topology, &coords).unwrap();
        let fixed = PrescribedDof {
            index: 0,
            value: 1.25,
        };
        let active = (0..full.ncols)
            .filter(|index| *index != fixed.index)
            .collect::<Vec<_>>();
        let layout = DofLayout::new(full.ncols, active.clone(), vec![fixed]);
        let affine =
            build_reduced_magnetic_flux_density_affine_operator_3d(&topology, &coords, &layout)
                .unwrap();
        let reduced_state = GmrfVector::from_iterator(
            active.len(),
            active.iter().map(|index| 0.1 + *index as f64 * 0.25),
        );
        let mut full_state = GmrfVector::zeros(full.ncols);
        for (reduced, full_index) in active.iter().copied().enumerate() {
            full_state[full_index] = reduced_state[reduced];
        }
        full_state[fixed.index] = fixed.value;

        let full_value = full.apply(&full_state).unwrap();
        let mut affine_value = affine.operator.apply(&reduced_state).unwrap();
        for row in 0..affine_value.len() {
            affine_value[row] += affine.bias[row];
        }

        assert!((&full_value - &affine_value).norm() < 1e-10);
    }

    #[test]
    fn scalar_az_2d_b_operator_matches_gradient_rotation_formula() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let operator =
            build_scalar_az_magnetic_flux_density_operator_2d(&topology, &coords).unwrap();
        let az = GmrfVector::from_iterator(
            topology.nsimplices(0),
            (0..topology.nsimplices(0)).map(|vertex| {
                let coord = coords.coord(vertex);
                2.0 * coord[0] - 3.0 * coord[1]
            }),
        );

        let b = operator.apply(&az).unwrap();
        for cell in 0..topology.nsimplices(2) {
            assert!((b[2 * cell] + 3.0).abs() < 1e-10);
            assert!((b[2 * cell + 1] + 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn scalar_az_2d_component_operators_select_full_components() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let full = build_scalar_az_magnetic_flux_density_operator_2d(&topology, &coords).unwrap();
        let components =
            build_scalar_az_magnetic_flux_density_component_operators_2d(&topology, &coords)
                .unwrap();

        assert_eq!(
            components.x,
            select_interleaved_component_rows(&full, topology.nsimplices(2), 2, 0).unwrap()
        );
        assert_eq!(
            components.y,
            select_interleaved_component_rows(&full, topology.nsimplices(2), 2, 1).unwrap()
        );
    }
}
