//! Gmsh import retaining physical volume labels alongside FEEC topology.

use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::{Matrix, Vector};
use manifold::{
    geometry::coord::mesh::MeshCoords,
    topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};
use std::{collections::BTreeMap, str::FromStr};

/// Affine tetrahedral mesh with one physical volume tag per topological cell.
pub struct TaggedTetrahedralMesh {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub cell_tags: Vec<i32>,
}

impl TaggedTetrahedralMesh {
    /// Read an ASCII, nonparametric Gmsh 4.1 mesh of linear tetrahedra.
    /// Every tetrahedral volume must have exactly one physical tag. Unlike the
    /// older geometry-only loader, this keeps material interfaces explicit.
    pub fn from_gmsh_bytes(bytes: &[u8]) -> Result<Self> {
        let fail = |message: &str| FeecGmrfError::Assembly(message.into());
        let text = std::str::from_utf8(bytes).map_err(|_| fail("expected ASCII Gmsh 4.1 mesh"))?;
        if text
            .split_once("$MeshFormat")
            .and_then(|(_, s)| s.lines().nth(1))
            .map(str::trim)
            != Some("4.1 0 8")
        {
            return Err(fail("expected ASCII Gmsh 4.1 mesh (4.1 0 8)"));
        }
        let msh = mshio::parse_msh_bytes(bytes).map_err(|e| fail(&format!("Gmsh parse: {e:?}")))?;
        let volumes = msh
            .data
            .entities
            .ok_or_else(|| fail("missing Gmsh entities"))?
            .volumes;
        let volume_tags: BTreeMap<_, _> = volumes
            .iter()
            .map(|v| (v.tag, v.physical_tags.clone()))
            .collect();
        // mshio elides contiguous node tags. Read the explicit ASCII node tags
        // to support shuffled blocks and sparse tags without ordering guesses.
        let section = text
            .split_once("$Nodes")
            .and_then(|(_, s)| s.split_once("$EndNodes"))
            .ok_or_else(|| fail("missing Gmsh nodes"))?
            .0;
        let mut tokens = section.split_whitespace();
        let blocks: usize = next(&mut tokens)?;
        let count: usize = next(&mut tokens)?;
        let _: u64 = next(&mut tokens)?;
        let _: u64 = next(&mut tokens)?;
        let mut nodes = BTreeMap::new();
        for _ in 0..blocks {
            let _: usize = next(&mut tokens)?;
            let _: i32 = next(&mut tokens)?;
            let parametric: usize = next(&mut tokens)?;
            let n: usize = next(&mut tokens)?;
            if parametric != 0 {
                return Err(fail("parametric nodes are unsupported"));
            }
            let tags = (0..n)
                .map(|_| next::<u64>(&mut tokens))
                .collect::<Result<Vec<_>>>()?;
            for tag in tags {
                let point = Vector::from_vec(
                    (0..3)
                        .map(|_| next::<f64>(&mut tokens))
                        .collect::<Result<Vec<_>>>()?,
                );
                if point.iter().any(|x| !x.is_finite()) || nodes.insert(tag, point).is_some() {
                    return Err(fail("invalid or repeated Gmsh node"));
                }
            }
        }
        if nodes.len() != count {
            return Err(fail("Gmsh node count mismatch"));
        }
        let indices: BTreeMap<_, _> = nodes.keys().enumerate().map(|(i, t)| (*t, i)).collect();
        let coords = MeshCoords::new(Matrix::from_columns(
            &nodes.into_values().collect::<Vec<_>>(),
        ));
        let mut cells = Vec::new();
        let mut tags = BTreeMap::new();
        for block in msh
            .data
            .elements
            .ok_or_else(|| fail("missing Gmsh elements"))?
            .element_blocks
        {
            if block.entity_dim != 3 {
                continue;
            }
            if block.element_type != mshio::ElementType::Tet4 {
                return Err(fail(
                    "only linear tetrahedral volume elements are supported",
                ));
            }
            let physical = volume_tags
                .get(&block.entity_tag)
                .ok_or_else(|| fail("element volume is missing"))?;
            if physical.len() != 1 {
                return Err(fail("each volume needs exactly one physical tag"));
            }
            for element in block.elements {
                let mut vertices = element
                    .nodes
                    .iter()
                    .map(|n| {
                        indices
                            .get(n)
                            .copied()
                            .ok_or_else(|| fail("element references missing node"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                vertices.sort_unstable();
                if vertices.windows(2).any(|w| w[0] == w[1])
                    || tags.insert(vertices.clone(), physical[0]).is_some()
                {
                    return Err(fail("degenerate or repeated tetrahedron"));
                }
                let x0 = coords.coord(vertices[0]);
                let jacobian = Matrix::from_columns(
                    &vertices[1..]
                        .iter()
                        .map(|v| coords.coord(*v) - x0)
                        .collect::<Vec<_>>(),
                );
                let determinant = jacobian.determinant();
                if !determinant.is_finite() || determinant == 0.0 {
                    return Err(fail("tetrahedron has zero or nonfinite volume"));
                }
                cells.push(Simplex::from(vertices));
            }
        }
        if cells.is_empty() {
            return Err(fail("mesh contains no tetrahedra"));
        }
        let topology = Complex::from_cells(Skeleton::new(cells));
        let cell_tags = topology.cells().iter().map(|c| tags[&c.vertices]).collect();
        Ok(Self {
            topology,
            coords,
            cell_tags,
        })
    }
}

fn next<T: FromStr>(tokens: &mut std::str::SplitWhitespace<'_>) -> Result<T> {
    tokens
        .next()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| FeecGmrfError::Assembly("invalid Gmsh node section".into()))
}
