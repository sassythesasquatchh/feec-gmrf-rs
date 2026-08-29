//! Fixed-footprint magnetic observation designs for the C51 studies.
//!
//! This module owns only study design: fixed physical centers, nested ordering,
//! and scalar-component prefixes.  Physical assembly remains in
//! [`MagneticVolumeSensorBank3d`], which uses the canonical FEEC
//! `A -> D1 -> B` pushforward.

use feec_gmrf::prelude::{
    FeecGmrfError, LinearMap, MagneticVolumeSensorBank3d, Result, WendlandC2Kernel3d,
};
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};

/// Fixed radius used by the rank-discovery and JUQ performance observations.
pub const C51_FIXED_OBSERVATION_RADIUS: f64 = 0.16;
/// Immutable identifier for the 12^3 JUQ performance sensor geometry.
pub const C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID: &str = "c51-magnetic-fixed-footprint-maximin-12-v1";
/// Number of grid points per axis in the existing primary rank-discovery design.
pub const C51_PRIMARY_GRID_POINTS_PER_AXIS: usize = 6;
/// Number of vector centers in the existing primary rank-discovery design.
pub const C51_PRIMARY_VECTOR_CENTER_COUNT: usize = C51_PRIMARY_GRID_POINTS_PER_AXIS
    * C51_PRIMARY_GRID_POINTS_PER_AXIS
    * C51_PRIMARY_GRID_POINTS_PER_AXIS;
/// Number of scalar component rows in the existing primary rank-discovery design.
pub const C51_PRIMARY_SCALAR_OBSERVATION_COUNT: usize = 3 * C51_PRIMARY_VECTOR_CENTER_COUNT;

/// Frozen points per axis in the first JUQ performance observation design.
pub const C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS: usize = 12;
/// Frozen vector-center capacity in the first JUQ performance design.
pub const C51_JUQ_PERFORMANCE_VECTOR_CENTER_COUNT: usize = C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS
    * C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS
    * C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS;
/// Frozen scalar-component capacity in the first JUQ performance design.
pub const C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT: usize =
    3 * C51_JUQ_PERFORMANCE_VECTOR_CENTER_COUNT;
/// Immutable direct flux-cochain design used by the C51 computational study.
pub const C51_FLUX_COCHAIN_DESIGN_ID: &str = "c51-flux-cochain-hashed-split-v1";
/// Frozen seed for the direct flux-cochain row ordering.
pub const C51_FLUX_COCHAIN_DESIGN_SEED: u64 = 0xC51F_1A5C_0A17;

const SELECTION_CENTERS: [[f64; 3]; 3] =
    [[0.25, 0.25, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]];
const HELD_OUT_CENTERS: [[f64; 3]; 3] =
    [[0.75, 0.75, 0.25], [0.25, 0.25, 0.75], [0.75, 0.25, 0.25]];

/// Scalar-prefix audit information independent of any latent state dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C51ScalarObservationPrefixMetadata {
    /// Requested scalar observations, not vector sensor locations.
    pub scalar_observation_count: usize,
    /// Minimal number of vector kernels needed to provide those rows.
    pub vector_center_count: usize,
    /// Selected rows for `(Bx, By, Bz)` respectively.
    pub component_counts: [usize; 3],
    /// Exact kernel-major scalar rows selected from the minimally assembled bank.
    pub selected_scalar_rows: Vec<usize>,
}

/// A minimally assembled fixed-footprint observation prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct C51MagneticObservationPrefix {
    kernels: Vec<WendlandC2Kernel3d>,
    quadrature_normalizations: Vec<f64>,
    scalar_map: LinearMap,
    metadata: C51ScalarObservationPrefixMetadata,
}

/// Disjoint direct observations and validation outputs of the magnetic flux
/// cochain `D1 A`.
///
/// Each returned row is an exact integral FEEC degree of freedom. The frozen
/// hash only chooses and orders rows; it performs no pointwise or barycentre
/// reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct C51FluxCochainSplit {
    observation: LinearMap,
    selection: LinearMap,
    held_out: LinearMap,
    observation_indices: Vec<usize>,
    selection_indices: Vec<usize>,
    held_out_indices: Vec<usize>,
}

impl C51FluxCochainSplit {
    pub fn observation(&self) -> &LinearMap {
        &self.observation
    }

    pub fn selection(&self) -> &LinearMap {
        &self.selection
    }

    pub fn held_out(&self) -> &LinearMap {
        &self.held_out
    }

    pub fn observation_indices(&self) -> &[usize] {
        &self.observation_indices
    }

    pub fn selection_indices(&self) -> &[usize] {
        &self.selection_indices
    }

    pub fn held_out_indices(&self) -> &[usize] {
        &self.held_out_indices
    }
}

impl C51MagneticObservationPrefix {
    pub fn kernels(&self) -> &[WendlandC2Kernel3d] {
        &self.kernels
    }

    pub fn quadrature_normalizations(&self) -> &[f64] {
        &self.quadrature_normalizations
    }

    /// Exact `m` scalar component rows requested by the caller.
    pub fn scalar_map(&self) -> &LinearMap {
        &self.scalar_map
    }

    pub fn metadata(&self) -> &C51ScalarObservationPrefixMetadata {
        &self.metadata
    }
}

/// The three fixed physical outputs used to select a spectral rank.
pub fn selection_kernels(radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    kernels_from_centers(&SELECTION_CENTERS, radius)
}

/// Independent fixed physical outputs used to validate a selected rank.
pub fn held_out_kernels(radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    kernels_from_centers(&HELD_OUT_CENTERS, radius)
}

/// Existing 6^3 rank-discovery bank in its original exact order.
pub fn primary_observation_kernels(radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    grid_kernel_prefix(
        C51_PRIMARY_GRID_POINTS_PER_AXIS,
        radius,
        C51_PRIMARY_VECTOR_CENTER_COUNT,
    )
}

/// Frozen JUQ-performance kernel prefix for exactly `m` scalar observations.
///
/// Only `ceil(m/3)` kernels are returned, so downstream FEEC assembly never
/// constructs the unrequested suffix.  The count is deliberately unrelated
/// to the latent dimension: noisy observations remain valid when `m > N`.
pub fn juq_performance_observation_kernel_prefix(
    scalar_observation_count: usize,
) -> Result<(Vec<WendlandC2Kernel3d>, C51ScalarObservationPrefixMetadata)> {
    let metadata = scalar_prefix_metadata(scalar_observation_count)?;
    let kernels = grid_kernel_prefix(
        C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS,
        C51_FIXED_OBSERVATION_RADIUS,
        metadata.vector_center_count,
    )?;
    Ok((kernels, metadata))
}

/// Assemble exactly the requested JUQ-performance scalar prefix through the
/// canonical fixed-volume magnetic sensor API.
pub fn assemble_juq_performance_observation_prefix(
    topology: &Complex,
    coordinates: &MeshCoords,
    scalar_observation_count: usize,
) -> Result<C51MagneticObservationPrefix> {
    let (kernels, metadata) = juq_performance_observation_kernel_prefix(scalar_observation_count)?;
    let bank = MagneticVolumeSensorBank3d::from_feec(topology, coordinates, &kernels)?;
    let scalar_map = bank
        .map()
        .map()
        .select_outputs(&metadata.selected_scalar_rows)?;
    if scalar_map.output_dimension() != scalar_observation_count {
        return Err(FeecGmrfError::Dimension(format!(
            "C51 scalar observation prefix requested {scalar_observation_count} rows but assembled {}",
            scalar_map.output_dimension()
        )));
    }
    Ok(C51MagneticObservationPrefix {
        kernels: bank.kernels().to_vec(),
        quadrature_normalizations: bank.quadrature_normalizations().to_vec(),
        scalar_map,
        metadata,
    })
}

/// Select disjoint exact magnetic flux-cochain rows from a canonical `D1` map.
pub fn assemble_c51_flux_cochain_split(
    flux_cochain: &LinearMap,
    observation_count: usize,
    selection_count: usize,
    held_out_count: usize,
) -> Result<C51FluxCochainSplit> {
    if observation_count == 0 || selection_count == 0 || held_out_count == 0 {
        return Err(invalid(
            "C51 flux-cochain split counts must all be strictly positive",
        ));
    }
    let required = observation_count
        .checked_add(selection_count)
        .and_then(|count| count.checked_add(held_out_count))
        .ok_or_else(|| invalid("C51 flux-cochain split count overflow"))?;
    if required > flux_cochain.output_dimension() {
        return Err(invalid(format!(
            "C51 flux-cochain split requires {required} rows but D1 supplies only {}",
            flux_cochain.output_dimension()
        )));
    }

    let mut ordered = (0..flux_cochain.output_dimension()).collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|&index| {
        (
            splitmix64((index as u64) ^ C51_FLUX_COCHAIN_DESIGN_SEED),
            index,
        )
    });
    let observation_indices = ordered[..observation_count].to_vec();
    let selection_end = observation_count + selection_count;
    let selection_indices = ordered[observation_count..selection_end].to_vec();
    let held_out_indices = ordered[selection_end..required].to_vec();
    Ok(C51FluxCochainSplit {
        observation: flux_cochain.select_outputs(&observation_indices)?,
        selection: flux_cochain.select_outputs(&selection_indices)?,
        held_out: flux_cochain.select_outputs(&held_out_indices)?,
        observation_indices,
        selection_indices,
        held_out_indices,
    })
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn scalar_prefix_metadata(
    scalar_observation_count: usize,
) -> Result<C51ScalarObservationPrefixMetadata> {
    if scalar_observation_count == 0 {
        return Err(invalid(
            "C51 scalar observation count must be strictly positive",
        ));
    }
    if scalar_observation_count > C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT {
        return Err(invalid(format!(
            "C51 scalar observation count {scalar_observation_count} exceeds the frozen JUQ-performance capacity {C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT}"
        )));
    }
    let vector_center_count = scalar_observation_count.div_ceil(3);
    let selected_scalar_rows = (0..scalar_observation_count).collect::<Vec<_>>();
    let complete_centers = scalar_observation_count / 3;
    let remainder = scalar_observation_count % 3;
    let component_counts = [
        complete_centers + usize::from(remainder >= 1),
        complete_centers + usize::from(remainder >= 2),
        complete_centers,
    ];
    Ok(C51ScalarObservationPrefixMetadata {
        scalar_observation_count,
        vector_center_count,
        component_counts,
        selected_scalar_rows,
    })
}

fn kernels_from_centers(centers: &[[f64; 3]], radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    centers
        .iter()
        .map(|center| WendlandC2Kernel3d::new(*center, radius))
        .collect()
}

fn grid_kernel_prefix(
    points_per_axis: usize,
    radius: f64,
    vector_center_count: usize,
) -> Result<Vec<WendlandC2Kernel3d>> {
    let centers = nested_maximin_grid_center_prefix(points_per_axis, radius, vector_center_count)?;
    kernels_from_centers(&centers, radius)
}

/// Deterministic farthest-point ordering with cached nearest distances.
///
/// The cache reduces a prefix of length `k` on a `p`-point grid from the old
/// repeated-nearest-search cost `O(k^2 p)` to `O(k p)`.  Candidate storage and
/// tie-breaking intentionally match the original 6^3 implementation.
fn nested_maximin_grid_center_prefix(
    points_per_axis: usize,
    radius: f64,
    prefix_count: usize,
) -> Result<Vec<[f64; 3]>> {
    if points_per_axis < 2 {
        return Err(invalid(
            "C51 observation grid requires at least two points per axis",
        ));
    }
    if !radius.is_finite() || radius <= 0.0 || radius >= 0.5 {
        return Err(invalid(
            "C51 observation grid radius must be finite and lie in (0, 0.5)",
        ));
    }
    let capacity = points_per_axis
        .checked_pow(3)
        .ok_or_else(|| invalid("C51 observation-grid capacity overflow"))?;
    if prefix_count == 0 || prefix_count > capacity {
        return Err(invalid(format!(
            "C51 observation-grid prefix {prefix_count} must lie in 1..={capacity}"
        )));
    }

    let spacing = (1.0 - 2.0 * radius) / (points_per_axis - 1) as f64;
    let mut remaining = (0..points_per_axis)
        .flat_map(|i| {
            (0..points_per_axis).flat_map(move |j| {
                (0..points_per_axis).map(move |k| {
                    [
                        radius + i as f64 * spacing,
                        radius + j as f64 * spacing,
                        radius + k as f64 * spacing,
                    ]
                })
            })
        })
        .collect::<Vec<_>>();
    let first = remaining
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            distance_squared(**left, [0.5; 3]).total_cmp(&distance_squared(**right, [0.5; 3]))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| invalid("C51 observation candidate grid is empty"))?;
    let first = remaining.swap_remove(first);
    let mut ordered = Vec::with_capacity(prefix_count);
    ordered.push(first);
    if prefix_count == 1 {
        return Ok(ordered);
    }

    let mut nearest_distances = remaining
        .iter()
        .map(|point| distance_squared(*point, first))
        .collect::<Vec<_>>();
    while ordered.len() < prefix_count {
        let next = nearest_distances
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("a validated non-full prefix has a remaining candidate");
        let selected = remaining.swap_remove(next);
        nearest_distances.swap_remove(next);
        ordered.push(selected);
        for (point, nearest) in remaining.iter().zip(&mut nearest_distances) {
            *nearest = nearest.min(distance_squared(*point, selected));
        }
    }
    Ok(ordered)
}

fn distance_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

fn invalid(message: impl Into<String>) -> FeecGmrfError {
    FeecGmrfError::InvalidParameter(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::gen::cartesian::CartesianMeshInfo;

    const REQUIRED_SCALAR_PREFIXES: [usize; 7] = [32, 128, 512, 1_024, 2_048, 4_096, 5_184];

    #[test]
    fn optimized_primary_order_is_exactly_legacy_compatible() {
        let optimized = primary_observation_kernels(C51_FIXED_OBSERVATION_RADIUS).unwrap();
        let legacy = legacy_observation_kernels(
            C51_PRIMARY_GRID_POINTS_PER_AXIS,
            C51_FIXED_OBSERVATION_RADIUS,
        )
        .unwrap();
        assert_eq!(optimized, legacy);
        assert_eq!(optimized.len(), C51_PRIMARY_VECTOR_CENTER_COUNT);
    }

    #[test]
    fn publication_scalar_prefixes_are_exact_nested_and_component_balanced() {
        let (full, full_metadata) =
            juq_performance_observation_kernel_prefix(C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT)
                .unwrap();
        assert_eq!(full.len(), C51_JUQ_PERFORMANCE_VECTOR_CENTER_COUNT);
        assert_eq!(full_metadata.component_counts, [1_728; 3]);

        for scalar_count in REQUIRED_SCALAR_PREFIXES {
            let (prefix, metadata) =
                juq_performance_observation_kernel_prefix(scalar_count).unwrap();
            assert_eq!(metadata.scalar_observation_count, scalar_count);
            assert_eq!(metadata.vector_center_count, scalar_count.div_ceil(3));
            assert_eq!(
                metadata.selected_scalar_rows,
                (0..scalar_count).collect::<Vec<_>>()
            );
            assert_eq!(
                metadata.component_counts.iter().sum::<usize>(),
                scalar_count
            );
            let minimum_component_count = metadata.component_counts.iter().min().unwrap();
            let maximum_component_count = metadata.component_counts.iter().max().unwrap();
            assert!(maximum_component_count - minimum_component_count <= 1);
            assert_eq!(prefix, full[..metadata.vector_center_count]);
        }
    }

    #[test]
    fn publication_grid_has_fixed_interior_support_and_is_deterministic() {
        let (first, _) =
            juq_performance_observation_kernel_prefix(C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT)
                .unwrap();
        let (second, _) =
            juq_performance_observation_kernel_prefix(C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT)
                .unwrap();
        assert_eq!(first, second);
        for kernel in first {
            assert_eq!(kernel.radius(), C51_FIXED_OBSERVATION_RADIUS);
            assert!(kernel.center().iter().all(|coordinate| {
                *coordinate >= C51_FIXED_OBSERVATION_RADIUS
                    && *coordinate <= 1.0 - C51_FIXED_OBSERVATION_RADIUS + f64::EPSILON
            }));
        }
    }

    #[test]
    fn publication_prefix_rejects_invalid_scalar_counts() {
        assert!(juq_performance_observation_kernel_prefix(0).is_err());
        assert!(juq_performance_observation_kernel_prefix(
            C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT + 1
        )
        .is_err());
    }

    #[test]
    fn physical_assembly_builds_only_the_requested_scalar_prefix() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 4, 1.0);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let observations =
            assemble_juq_performance_observation_prefix(&topology, &coordinates, 32).unwrap();
        assert_eq!(observations.kernels().len(), 11);
        assert_eq!(observations.quadrature_normalizations().len(), 11);
        assert_eq!(observations.scalar_map().output_dimension(), 32);
        assert_eq!(observations.metadata().component_counts, [11, 11, 10]);
    }

    #[test]
    fn flux_cochain_split_is_exact_deterministic_and_disjoint() {
        let flux = LinearMap::new(feec_gmrf::prelude::SparseMat::diagonal(16, 2.0)).unwrap();
        let first = assemble_c51_flux_cochain_split(&flux, 8, 4, 3).unwrap();
        let second = assemble_c51_flux_cochain_split(&flux, 8, 4, 3).unwrap();
        assert_eq!(first, second);

        let all_indices = first
            .observation_indices()
            .iter()
            .chain(first.selection_indices())
            .chain(first.held_out_indices())
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(all_indices.len(), 15);

        let input = (0..16).map(|index| index as f64).collect::<Vec<_>>();
        let expected = first
            .observation_indices()
            .iter()
            .map(|&index| 2.0 * input[index])
            .collect::<Vec<_>>();
        assert_eq!(first.observation().apply(&input).unwrap(), expected);
        assert!(assemble_c51_flux_cochain_split(&flux, 14, 2, 1).is_err());
    }

    fn legacy_observation_kernels(count: usize, radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
        let spacing = (1.0 - 2.0 * radius) / (count - 1) as f64;
        let mut remaining = (0..count)
            .flat_map(|i| {
                (0..count).flat_map(move |j| {
                    (0..count).map(move |k| {
                        [
                            radius + i as f64 * spacing,
                            radius + j as f64 * spacing,
                            radius + k as f64 * spacing,
                        ]
                    })
                })
            })
            .collect::<Vec<_>>();
        let mut ordered = Vec::<[f64; 3]>::with_capacity(remaining.len());
        let first = remaining
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                distance_squared(**left, [0.5; 3]).total_cmp(&distance_squared(**right, [0.5; 3]))
            })
            .map(|(index, _)| index)
            .ok_or_else(|| invalid("legacy observation candidate grid is empty"))?;
        ordered.push(remaining.swap_remove(first));
        while !remaining.is_empty() {
            let next = remaining
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    nearest_distance_squared(**left, &ordered)
                        .total_cmp(&nearest_distance_squared(**right, &ordered))
                })
                .map(|(index, _)| index)
                .expect("nonempty legacy candidate set has a farthest point");
            ordered.push(remaining.swap_remove(next));
        }
        kernels_from_centers(&ordered, radius)
    }

    fn nearest_distance_squared(point: [f64; 3], selected: &[[f64; 3]]) -> f64 {
        selected
            .iter()
            .map(|selected| distance_squared(point, *selected))
            .fold(f64::INFINITY, f64::min)
    }
}
