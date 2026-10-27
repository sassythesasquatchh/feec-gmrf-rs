//! Pointwise magnetic sensors for lowest-order Whitney vector potentials.

use super::MagneticFieldMaps3d;
use crate::operator::LinearMap;
use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::{Matrix, Vector};
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};

/// Position and positively oriented orthonormal measurement axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagneticPointSensor3d {
    position: [f64; 3],
    axes: [[f64; 3]; 3],
}

impl MagneticPointSensor3d {
    pub fn new(position: [f64; 3], axes: [[f64; 3]; 3]) -> Result<Self> {
        if position
            .iter()
            .chain(axes.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Err(FeecGmrfError::InvalidParameter(
                "sensor position and axes must be finite".into(),
            ));
        }
        let frame = Matrix::from_fn(3, 3, |i, j| axes[i][j]);
        if (&frame * frame.transpose() - Matrix::identity(3, 3)).amax() > 1e-10 {
            return Err(FeecGmrfError::InvalidParameter(
                "sensor axes must be orthonormal".into(),
            ));
        }
        if frame.determinant() < 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "sensor tripod must have positive orientation".into(),
            ));
        }
        Ok(Self { position, axes })
    }

    pub fn cartesian(position: [f64; 3]) -> Result<Self> {
        Self::new(
            position,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
    }
    pub fn position(&self) -> [f64; 3] {
        self.position
    }
    pub fn axes(&self) -> [[f64; 3]; 3] {
        self.axes
    }
}

/// Assemble signed tripod components of `B=dA` at fixed physical points.
///
/// On each affine tetrahedron, `d` of a Whitney 1-form is a constant 2-form.
/// This selects the containing cell's canonical magnetic reconstruction and
/// rotates it into the sensor tripod. Points on element faces use the first
/// containing cell; place sensors strictly inside air cells for unambiguous
/// values of a discontinuous discrete field. This is not nodal interpolation.
pub fn magnetic_point_measurement_map_3d(
    topology: &Complex,
    coords: &MeshCoords,
    sensors: &[MagneticPointSensor3d],
) -> Result<LinearMap> {
    if sensors.is_empty() {
        return Err(FeecGmrfError::InvalidParameter(
            "at least one point sensor is required".into(),
        ));
    }
    let magnetic = MagneticFieldMaps3d::from_feec(topology, coords)?;
    let mut rows = Vec::with_capacity(sensors.len() * 3);
    for sensor in sensors {
        let point = Vector::from_column_slice(&sensor.position);
        let cell = coords
            .find_cell_containing(topology, point.as_view())
            .ok_or_else(|| {
                FeecGmrfError::InvalidParameter(format!(
                    "sensor {:?} lies outside the mesh",
                    sensor.position
                ))
            })?;
        for axis in sensor.axes {
            rows.push((0..3).map(|k| (3 * cell.kidx() + k, axis[k])).collect());
        }
    }
    LinearMap::weighted_rows(3 * topology.nsimplices(3), &rows)?
        .compose(magnetic.magnetic_field.map())
}
