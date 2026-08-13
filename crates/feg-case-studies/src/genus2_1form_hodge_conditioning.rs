#[cfg(test)]
use crate::test_util::lock_feec_harmonic_tests;
use crate::visual_output;
use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use ddf::cochain::{cochain_projection, Cochain};
use exterior::field::EmbeddedDiffFormClosure;
use feg_infer::conditioning::hodge_1form::{
    compute_harmonic_basis_1form, harmonic_coefficients_1form,
    mass_orthonormalize_harmonic_basis_1form, run_hodge_1form_conditioning, Hodge1FormBranchResult,
    Hodge1FormConditioningConfig,
};
use feg_infer::prior::matern::one_form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, MaternConfig as Matern1FormConfig,
    MaternMassInverse as Matern1FormMassInverse,
};
use feg_infer::sparse::{feec_csr_to_gmrf, gmrf_vec_to_feec};
use formoniq::io::sample_1form_cell_vectors;
use gmrf_core::types::Vector as GmrfVector;
use gmrf_core::Gmrf;
use manifold::{
    geometry::coord::mesh::MeshCoords, io::gmsh::gmsh2coord_complex, topology::complex::Complex,
};
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const EPS: f64 = 1e-12;
const HARMONIC_COEFFICIENTS: [f64; 4] = [0.80, -0.55, 0.35, -0.25];

#[derive(Debug, Clone)]
pub struct Genus2Torus1FormHodgeConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub harmonic_dim: usize,
    pub sample_count: usize,
    pub rng_seed: u64,
    pub raw_sample_seed: u64,
}

impl Default for Genus2Torus1FormHodgeConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_genus2_torus_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            noise_variance: 1e-8,
            harmonic_dim: 4,
            sample_count: 2,
            rng_seed: 13,
            raw_sample_seed: 137,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genus2CycleKind {
    Meridian,
    Longitudinal,
}

impl Genus2CycleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Meridian => "meridian",
            Self::Longitudinal => "longitudinal",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Genus2CycleObservation {
    pub name: String,
    pub kind: Genus2CycleKind,
    pub target_vertices: Vec<usize>,
    pub path_vertices: Vec<usize>,
    pub edge_count: usize,
    pub closure_residual_l1: f64,
}

#[derive(Debug, Clone)]
pub struct Genus2TopologySummary {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub euler_characteristic: isize,
    pub b0: usize,
    pub b1: usize,
    pub b2: isize,
    pub boundary_edge_count: usize,
    pub nonmanifold_edge_count: usize,
}

#[derive(Debug, Clone)]
pub struct Genus2Torus1FormHodgeConditioningResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub topology_summary: Genus2TopologySummary,
    pub truth: FeecVector,
    pub raw_prior_sample: FeecVector,
    pub observations: FeecVector,
    pub observation_matrix: FeecCsr,
    pub cycle_observations: Vec<Genus2CycleObservation>,
    pub cycle_harmonic_pairing: FeecMatrix,
    pub harmonic_basis: FeecMatrix,
    pub exact: Hodge1FormBranchResult,
    pub coexact: Hodge1FormBranchResult,
    pub harmonic: Hodge1FormBranchResult,
}

#[derive(Debug, Clone, Copy)]
struct EdgeStep {
    to: usize,
    edge_index: usize,
    sign: f64,
    length: f64,
}

#[derive(Debug, Clone)]
struct CycleSpec {
    name: &'static str,
    kind: Genus2CycleKind,
    points: Vec<[f64; 3]>,
}

#[derive(Debug, Clone)]
struct ShortestPath {
    vertices: Vec<usize>,
    edges: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct QueueState {
    cost: f64,
    vertex: usize,
}

impl Eq for QueueState {}

impl Ord for QueueState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl PartialOrd for QueueState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn default_genus2_torus_mesh_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/genus2_torus_touching.msh")
}

pub fn default_genus2_torus_shell_mesh_path() -> PathBuf {
    default_genus2_torus_mesh_path()
}

pub fn run_genus2_1form_hodge_conditioning(
    config: &Genus2Torus1FormHodgeConditioningConfig,
) -> Result<Genus2Torus1FormHodgeConditioningResult, Box<dyn Error>> {
    validate_config(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = validate_genus2_topology(&topology)?;

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw =
        compute_harmonic_basis_1form(&topology, &metric, config.harmonic_dim, None)
            .map_err(invalid_data)?;
    let harmonic_basis =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;

    let (observation_matrix, cycle_observations) =
        build_cycle_observation_matrix(&topology, &coords)?;
    let cycle_harmonic_pairing = &observation_matrix * &harmonic_basis;
    let pairing_rank = cycle_harmonic_pairing.rank(1e-8);
    if pairing_rank != config.harmonic_dim {
        return Err(invalid_data(format!(
            "cycle-harmonic pairing rank {pairing_rank} does not match harmonic dimension {}",
            config.harmonic_dim
        ))
        .into());
    }

    let truth = build_truth(&topology, &coords, &hodge.mass_u, &harmonic_basis)?;
    let raw_prior_sample =
        sample_full_matern_1form_prior(&topology, &metric, &hodge, config).map_err(invalid_data)?;
    let generic = run_hodge_1form_conditioning(&Hodge1FormConditioningConfig {
        topology: &topology,
        coords: &coords,
        metric: &metric,
        truth: &truth,
        observation_matrix: &observation_matrix,
        kappa: config.kappa,
        tau: config.tau,
        noise_variance: config.noise_variance,
        harmonic_dim: config.harmonic_dim,
        harmonic_basis_override: Some(&harmonic_basis_raw),
        sample_count: config.sample_count,
        rng_seed: config.rng_seed,
    })?;

    Ok(Genus2Torus1FormHodgeConditioningResult {
        topology,
        coords,
        topology_summary,
        truth,
        raw_prior_sample,
        observations: generic.observations,
        observation_matrix,
        cycle_observations,
        cycle_harmonic_pairing,
        harmonic_basis: generic.harmonic_basis,
        exact: generic.exact,
        coexact: generic.coexact,
        harmonic: generic.harmonic,
    })
}

pub fn write_genus2_1form_hodge_conditioning_outputs(
    result: &Genus2Torus1FormHodgeConditioningResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_topology_summary(result, out_dir.join("topology_summary.txt"))?;
    write_cycle_observations_csv(result, out_dir.join("cycle_observations.csv"))?;
    write_cycle_branch_summary_csv(result, out_dir.join("cycle_branch_summary.csv"))?;
    write_cycle_paths_vtu(result, out_dir.join("cycle_paths.vtu"))?;
    write_raw_prior_sample_outputs(result, out_dir)?;
    write_branch_outputs(result, &result.exact, out_dir)?;
    write_branch_outputs(result, &result.coexact, out_dir)?;
    write_branch_outputs(result, &result.harmonic, out_dir)?;
    Ok(())
}

fn validate_config(config: &Genus2Torus1FormHodgeConditioningConfig) -> io::Result<()> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive"));
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive"));
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err(invalid_input("noise_variance must be finite and positive"));
    }
    if config.harmonic_dim != 4 {
        return Err(invalid_input(
            "the genus-2 experiment expects harmonic_dim = 4",
        ));
    }
    Ok(())
}

pub(crate) fn validate_genus2_topology(
    topology: &Complex,
) -> Result<Genus2TopologySummary, io::Error> {
    if topology.dim() != 2 {
        return Err(invalid_data(format!(
            "genus-2 shell must be a surface mesh, got topology dimension {}",
            topology.dim()
        )));
    }

    let vertex_count = topology.vertices().len();
    let edge_count = topology.edges().len();
    let face_count = topology.cells().len();
    let euler_characteristic = vertex_count as isize - edge_count as isize + face_count as isize;
    let boundary_edge_count = topology
        .edges()
        .handle_iter()
        .filter(|edge| edge.cocells().count() == 1)
        .count();
    let nonmanifold_edge_count = topology
        .edges()
        .handle_iter()
        .filter(|edge| edge.cocells().count() != 2)
        .count();
    let b0 = connected_component_count(topology);
    let b1 = topology.homology_dim(1);
    let b2 = euler_characteristic - b0 as isize + b1 as isize;

    let summary = Genus2TopologySummary {
        vertex_count,
        edge_count,
        face_count,
        euler_characteristic,
        b0,
        b1,
        b2,
        boundary_edge_count,
        nonmanifold_edge_count,
    };

    if summary.boundary_edge_count != 0 || summary.nonmanifold_edge_count != 0 {
        return Err(invalid_data(format!(
            "genus-2 shell must be closed and manifold, got boundary_edge_count={} nonmanifold_edge_count={}",
            summary.boundary_edge_count, summary.nonmanifold_edge_count
        )));
    }
    if summary.euler_characteristic != -2 || summary.b0 != 1 || summary.b1 != 4 || summary.b2 != 1 {
        return Err(invalid_data(format!(
            "expected genus-2 topology (chi=-2, b0=1, b1=4, b2=1), got chi={} b0={} b1={} b2={}",
            summary.euler_characteristic, summary.b0, summary.b1, summary.b2
        )));
    }

    Ok(summary)
}

fn connected_component_count(topology: &Complex) -> usize {
    let mut parent = (0..topology.vertices().len()).collect::<Vec<_>>();
    for edge in topology.edges().handle_iter() {
        union_vertices(&mut parent, edge.vertices[0], edge.vertices[1]);
    }
    (0..parent.len())
        .map(|vertex| find_vertex_root(&mut parent, vertex))
        .collect::<std::collections::HashSet<_>>()
        .len()
}

fn find_vertex_root(parent: &mut [usize], vertex: usize) -> usize {
    if parent[vertex] != vertex {
        parent[vertex] = find_vertex_root(parent, parent[vertex]);
    }
    parent[vertex]
}

fn union_vertices(parent: &mut [usize], lhs: usize, rhs: usize) {
    let lhs_root = find_vertex_root(parent, lhs);
    let rhs_root = find_vertex_root(parent, rhs);
    if lhs_root != rhs_root {
        parent[rhs_root] = lhs_root;
    }
}

pub(crate) fn build_cycle_observation_matrix(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<(FeecCsr, Vec<Genus2CycleObservation>), io::Error> {
    let adjacency = build_edge_adjacency(topology, coords);
    let specs = default_cycle_specs();
    let mut rows = Vec::with_capacity(specs.len());
    let mut observations = Vec::with_capacity(specs.len());

    for spec in specs {
        let target_vertices = spec
            .points
            .iter()
            .map(|point| nearest_vertex(coords, *point))
            .collect::<Vec<_>>();
        let (row, path_vertices) =
            build_cycle_row(topology, &adjacency, &target_vertices).map_err(invalid_data)?;
        let closure_residual_l1 = cycle_closure_residual(topology, &row);
        if closure_residual_l1 > 1e-10 {
            return Err(invalid_data(format!(
                "cycle {} is not closed; l1 residual={closure_residual_l1}",
                spec.name
            )));
        }

        observations.push(Genus2CycleObservation {
            name: spec.name.to_string(),
            kind: spec.kind,
            target_vertices,
            path_vertices,
            edge_count: row.len(),
            closure_residual_l1,
        });
        rows.push(row);
    }

    let mut coo = FeecCoo::new(rows.len(), topology.edges().len());
    for (row_index, row) in rows.iter().enumerate() {
        for (edge_index, weight) in row {
            coo.push(row_index, *edge_index, *weight);
        }
    }
    Ok((FeecCsr::from(&coo), observations))
}

fn default_cycle_specs() -> Vec<CycleSpec> {
    let major_radius = 0.80;
    let minor_radius = 0.28;
    let overlap_depth = 0.07;
    let center_offset = major_radius + minor_radius - 0.5 * overlap_depth;
    vec![
        CycleSpec {
            name: "meridian_left_torus",
            kind: Genus2CycleKind::Meridian,
            points: circle_xz_points([-center_offset - major_radius, 0.0, 0.0], minor_radius, 12),
        },
        CycleSpec {
            name: "meridian_right_torus",
            kind: Genus2CycleKind::Meridian,
            points: circle_xz_points([center_offset + major_radius, 0.0, 0.0], minor_radius, 12),
        },
        CycleSpec {
            name: "longitude_left_torus",
            kind: Genus2CycleKind::Longitudinal,
            points: torus_longitude_points(
                [-center_offset, 0.0, 0.0],
                major_radius,
                minor_radius,
                16,
            ),
        },
        CycleSpec {
            name: "longitude_right_torus",
            kind: Genus2CycleKind::Longitudinal,
            points: torus_longitude_points(
                [center_offset, 0.0, 0.0],
                major_radius,
                minor_radius,
                16,
            ),
        },
    ]
}

fn circle_xz_points(center: [f64; 3], radius: f64, count: usize) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / count as f64;
            [
                center[0] + radius * angle.cos(),
                center[1],
                center[2] + radius * angle.sin(),
            ]
        })
        .collect()
}

fn torus_longitude_points(
    center: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
    count: usize,
) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / count as f64;
            [
                center[0] + major_radius * angle.cos(),
                center[1] + major_radius * angle.sin(),
                center[2] + minor_radius,
            ]
        })
        .collect()
}

fn build_edge_adjacency(topology: &Complex, coords: &MeshCoords) -> Vec<Vec<EdgeStep>> {
    let mut adjacency = vec![Vec::new(); topology.vertices().len()];
    for edge in topology.edges().handle_iter() {
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        let length = (coords.coord(a) - coords.coord(b)).norm();
        adjacency[a].push(EdgeStep {
            to: b,
            edge_index: edge.kidx(),
            sign: 1.0,
            length,
        });
        adjacency[b].push(EdgeStep {
            to: a,
            edge_index: edge.kidx(),
            sign: -1.0,
            length,
        });
    }
    adjacency
}

fn nearest_vertex(coords: &MeshCoords, target: [f64; 3]) -> usize {
    (0..coords.nvertices())
        .min_by(|lhs, rhs| {
            distance_to_target(coords, *lhs, target)
                .partial_cmp(&distance_to_target(coords, *rhs, target))
                .unwrap_or(Ordering::Equal)
        })
        .expect("mesh should contain at least one vertex")
}

fn distance_to_target(coords: &MeshCoords, vertex: usize, target: [f64; 3]) -> f64 {
    let coord = coords.coord(vertex);
    let dx = coord[0] - target[0];
    let dy = coord[1] - target[1];
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    let dz = z - target[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn build_cycle_row(
    topology: &Complex,
    adjacency: &[Vec<EdgeStep>],
    target_vertices: &[usize],
) -> Result<(Vec<(usize, f64)>, Vec<usize>), String> {
    let mut vertices = target_vertices.to_vec();
    vertices.dedup();
    if vertices.len() < 2 {
        return Err("cycle requires at least two distinct target vertices".to_string());
    }

    let mut row = BTreeMap::<usize, f64>::new();
    let mut path_vertices = vec![vertices[0]];
    for i in 0..vertices.len() {
        let start = vertices[i];
        let goal = vertices[(i + 1) % vertices.len()];
        let path = shortest_path(adjacency, start, goal)?;
        for (edge_index, sign) in path.edges {
            *row.entry(edge_index).or_insert(0.0) += sign;
        }
        path_vertices.extend(path.vertices.into_iter().skip(1));
    }
    if path_vertices.last().copied() != Some(path_vertices[0]) {
        path_vertices.push(path_vertices[0]);
    }

    let entries = row
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err(format!(
            "cycle through {} vertices produced an empty edge row",
            vertices.len()
        ));
    }
    if entries
        .iter()
        .any(|(edge_index, _)| *edge_index >= topology.edges().len())
    {
        return Err("cycle row contains an invalid edge index".to_string());
    }
    Ok((entries, path_vertices))
}

fn shortest_path(
    adjacency: &[Vec<EdgeStep>],
    start: usize,
    goal: usize,
) -> Result<ShortestPath, String> {
    if start == goal {
        return Ok(ShortestPath {
            vertices: vec![start],
            edges: Vec::new(),
        });
    }

    let mut dist = vec![f64::INFINITY; adjacency.len()];
    let mut prev = vec![None::<(usize, usize, f64)>; adjacency.len()];
    let mut heap = BinaryHeap::new();

    dist[start] = 0.0;
    heap.push(QueueState {
        cost: 0.0,
        vertex: start,
    });

    while let Some(QueueState { cost, vertex }) = heap.pop() {
        if vertex == goal {
            break;
        }
        if cost > dist[vertex] + EPS {
            continue;
        }
        for step in &adjacency[vertex] {
            let next_cost = cost + step.length;
            if next_cost + EPS < dist[step.to] {
                dist[step.to] = next_cost;
                prev[step.to] = Some((vertex, step.edge_index, step.sign));
                heap.push(QueueState {
                    cost: next_cost,
                    vertex: step.to,
                });
            }
        }
    }

    if !dist[goal].is_finite() {
        return Err(format!(
            "no 1-skeleton path found from vertex {start} to {goal}"
        ));
    }

    let mut current = goal;
    let mut reversed_vertices = vec![goal];
    let mut reversed_edges = Vec::new();
    while current != start {
        let Some((parent, edge_index, sign)) = prev[current] else {
            return Err(format!("failed to reconstruct path from {start} to {goal}"));
        };
        reversed_edges.push((edge_index, sign));
        current = parent;
        reversed_vertices.push(current);
    }
    reversed_vertices.reverse();
    reversed_edges.reverse();
    Ok(ShortestPath {
        vertices: reversed_vertices,
        edges: reversed_edges,
    })
}

fn cycle_closure_residual(topology: &Complex, row: &[(usize, f64)]) -> f64 {
    let mut balance = vec![0.0; topology.vertices().len()];
    for (edge_index, weight) in row {
        let edge = topology.edges().handle_by_kidx(*edge_index);
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        balance[a] -= *weight;
        balance[b] += *weight;
    }
    balance.iter().map(|value| value.abs()).sum()
}

pub(crate) fn build_truth(
    topology: &Complex,
    coords: &MeshCoords,
    mass_u: &FeecCsr,
    harmonic_basis: &FeecMatrix,
) -> Result<FeecVector, io::Error> {
    let seed = EmbeddedDiffFormClosure::ambient_one_form(
        |p| {
            let x = p[0];
            let y = p[1];
            let z = p[2];
            FeecVector::from_column_slice(&[
                0.25 * y + 0.15 * z + 0.10 * (2.0 * x).sin(),
                -0.30 * x + 0.20 * z + 0.08 * (3.0 * y).cos(),
                0.18 * x - 0.22 * y + 0.06 * (2.0 * z).sin(),
            ])
        },
        coords.dim(),
        topology.dim(),
    );
    let seed = cochain_projection(&seed, topology, coords, None);
    let mut truth = remove_harmonic_content(&seed.coeffs, harmonic_basis, mass_u);
    for j in 0..harmonic_basis.ncols().min(HARMONIC_COEFFICIENTS.len()) {
        truth += harmonic_basis
            .column(j)
            .into_owned()
            .scale(HARMONIC_COEFFICIENTS[j]);
    }
    Ok(truth)
}

fn sample_full_matern_1form_prior(
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    config: &Genus2Torus1FormHodgeConditioningConfig,
) -> Result<FeecVector, String> {
    let precision = build_matern_precision_1form(
        topology,
        metric,
        hodge,
        Matern1FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let precision = feec_csr_to_gmrf(&precision);
    let mut prior = Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision)
        .map_err(|err| format!("failed to build full 1-form Matern prior: {err}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.raw_sample_seed);
    let sample = prior
        .sample(&mut rng)
        .map_err(|err| format!("failed to sample full 1-form Matern prior: {err}"))?;
    Ok(gmrf_vec_to_feec(&sample))
}

fn remove_harmonic_content(
    field: &FeecVector,
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> FeecVector {
    let coefficients = harmonic_coefficients_1form(field, harmonic_basis, mass_u);
    let mut harmonic_free = field.clone();
    for j in 0..harmonic_basis.ncols() {
        harmonic_free -= harmonic_basis.column(j).into_owned().scale(coefficients[j]);
    }
    harmonic_free
}

fn write_topology_summary(
    result: &Genus2Torus1FormHodgeConditioningResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    let summary = &result.topology_summary;
    writeln!(writer, "vertex_count={}", summary.vertex_count)?;
    writeln!(writer, "edge_count={}", summary.edge_count)?;
    writeln!(writer, "face_count={}", summary.face_count)?;
    writeln!(
        writer,
        "euler_characteristic={}",
        summary.euler_characteristic
    )?;
    writeln!(writer, "b0={}", summary.b0)?;
    writeln!(writer, "b1={}", summary.b1)?;
    writeln!(writer, "b2={}", summary.b2)?;
    writeln!(
        writer,
        "boundary_edge_count={}",
        summary.boundary_edge_count
    )?;
    writeln!(
        writer,
        "nonmanifold_edge_count={}",
        summary.nonmanifold_edge_count
    )?;
    writeln!(
        writer,
        "cycle_harmonic_pairing_rank={}",
        result.cycle_harmonic_pairing.rank(1e-8)
    )?;
    Ok(())
}

fn write_cycle_observations_csv(
    result: &Genus2Torus1FormHodgeConditioningResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "cycle_index,name,kind,target_vertices,path_vertices,edge_count,closure_residual_l1,observed_value"
    )?;
    for (cycle_index, cycle) in result.cycle_observations.iter().enumerate() {
        let target_vertices = cycle
            .target_vertices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|");
        let path_vertices = cycle
            .path_vertices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|");
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.12},{:.12}",
            cycle_index,
            cycle.name,
            cycle.kind.as_str(),
            target_vertices,
            path_vertices,
            cycle.edge_count,
            cycle.closure_residual_l1,
            result.observations[cycle_index]
        )?;
    }
    Ok(())
}

fn write_cycle_branch_summary_csv(
    result: &Genus2Torus1FormHodgeConditioningResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "branch,cycle_index,name,kind,observed_value,posterior_mean,residual,prior_variance,posterior_variance,variance_reduction,posterior_std,posterior_lower_95,posterior_upper_95"
    )?;
    for branch in [&result.exact, &result.coexact, &result.harmonic] {
        for (cycle_index, cycle) in result.cycle_observations.iter().enumerate() {
            let posterior_std = branch.observation_posterior_variance[cycle_index]
                .max(0.0)
                .sqrt();
            let lower_95 = branch.posterior_observation_mean[cycle_index] - 1.96 * posterior_std;
            let upper_95 = branch.posterior_observation_mean[cycle_index] + 1.96 * posterior_std;
            writeln!(
                writer,
                "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                branch.kind.as_str(),
                cycle_index,
                cycle.name,
                cycle.kind.as_str(),
                branch.observation_values[cycle_index],
                branch.posterior_observation_mean[cycle_index],
                branch.observation_residual[cycle_index],
                branch.observation_prior_variance[cycle_index],
                branch.observation_posterior_variance[cycle_index],
                branch.observation_variance_reduction[cycle_index],
                posterior_std,
                lower_95,
                upper_95
            )?;
        }
    }
    Ok(())
}

fn write_raw_prior_sample_outputs(
    result: &Genus2Torus1FormHodgeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let raw_sample = Cochain::new(1, result.raw_prior_sample.clone());
    visual_output::write_1cochain_fields(
        out_dir.join("raw_full_matern_sample_edge_fields.vtu"),
        &result.coords,
        &result.topology,
        &[("raw_full_matern_sample", &raw_sample)],
    )?;

    let raw_vectors = sample_1form_cell_vectors(&result.coords, &result.topology, &raw_sample)?;
    visual_output::write_top_cell_vector_fields(
        out_dir.join("raw_full_matern_sample_surface_vector.vtu"),
        &result.coords,
        &result.topology,
        "raw_full_matern_sample_surface_vector",
        raw_vectors.as_slice(),
        &[],
    )?;
    Ok(())
}

fn write_cycle_paths_vtu(
    result: &Genus2Torus1FormHodgeConditioningResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let path_refs = result
        .cycle_observations
        .iter()
        .map(|cycle| cycle.path_vertices.as_slice())
        .collect::<Vec<_>>();
    let cycle_index = (0..result.cycle_observations.len())
        .map(|index| index as f64)
        .collect::<Vec<_>>();
    let observed_circulation = result.observations.iter().copied().collect::<Vec<_>>();
    let harmonic_posterior_mean = result
        .harmonic
        .posterior_observation_mean
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let harmonic_posterior_std = result
        .harmonic
        .observation_posterior_variance
        .iter()
        .map(|value| value.max(0.0).sqrt())
        .collect::<Vec<_>>();
    let harmonic_posterior_lower_95 = (0..result.cycle_observations.len())
        .map(|index| {
            let std = result.harmonic.observation_posterior_variance[index]
                .max(0.0)
                .sqrt();
            result.harmonic.posterior_observation_mean[index] - 1.96 * std
        })
        .collect::<Vec<_>>();
    let harmonic_posterior_upper_95 = (0..result.cycle_observations.len())
        .map(|index| {
            let std = result.harmonic.observation_posterior_variance[index]
                .max(0.0)
                .sqrt();
            result.harmonic.posterior_observation_mean[index] + 1.96 * std
        })
        .collect::<Vec<_>>();

    visual_output::write_polyline_fields(
        path,
        "genus-2 observation cycle paths",
        &result.coords,
        &path_refs,
        &[
            ("cycle_index", cycle_index.as_slice()),
            ("observed_circulation", observed_circulation.as_slice()),
            (
                "harmonic_posterior_mean",
                harmonic_posterior_mean.as_slice(),
            ),
            ("harmonic_posterior_std", harmonic_posterior_std.as_slice()),
            (
                "harmonic_posterior_lower_95",
                harmonic_posterior_lower_95.as_slice(),
            ),
            (
                "harmonic_posterior_upper_95",
                harmonic_posterior_upper_95.as_slice(),
            ),
        ],
    )
}

fn write_branch_outputs(
    result: &Genus2Torus1FormHodgeConditioningResult,
    branch: &Hodge1FormBranchResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let branch_dir = out_dir.join(branch.kind.as_str());
    fs::create_dir_all(&branch_dir)?;

    let mut fields = vec![
        ("truth".to_string(), Cochain::new(1, result.truth.clone())),
        (
            "posterior_mean".to_string(),
            Cochain::new(1, branch.posterior_mean.clone()),
        ),
        (
            "absolute_mean_error".to_string(),
            Cochain::new(1, branch.absolute_mean_error.clone()),
        ),
        (
            "prior_variance".to_string(),
            Cochain::new(1, branch.prior_variance.clone()),
        ),
        (
            "posterior_variance".to_string(),
            Cochain::new(1, branch.posterior_variance.clone()),
        ),
        (
            "variance_reduction".to_string(),
            Cochain::new(1, branch.variance_reduction.clone()),
        ),
    ];
    for (sample_index, sample) in branch.prior_samples.iter().enumerate() {
        fields.push((
            format!("prior_sample_{sample_index}"),
            Cochain::new(1, sample.clone()),
        ));
    }
    for (sample_index, sample) in branch.posterior_samples.iter().enumerate() {
        fields.push((
            format!("posterior_sample_{sample_index}"),
            Cochain::new(1, sample.clone()),
        ));
    }
    let field_refs = fields
        .iter()
        .map(|(name, cochain)| (name.as_str(), cochain))
        .collect::<Vec<_>>();
    visual_output::write_1cochain_fields(
        branch_dir.join("edge_fields.vtu"),
        &result.coords,
        &result.topology,
        &field_refs,
    )?;
    write_branch_surface_vector_vtu(result, branch, &branch_dir)?;
    write_branch_reconstructed_component_variance_vtu(result, branch, &branch_dir)?;

    let mut summary = BufWriter::new(File::create(branch_dir.join("summary.txt"))?);
    writeln!(summary, "branch={}", branch.kind.as_str())?;
    writeln!(summary, "latent_dimension={}", branch.latent_dimension)?;
    writeln!(summary, "observation_count={}", branch.observation_count)?;
    writeln!(
        summary,
        "max_abs_observation_error={}",
        branch.max_abs_observation_error
    )?;
    writeln!(
        summary,
        "mean_abs_observation_error={}",
        branch.mean_abs_observation_error
    )?;
    writeln!(
        summary,
        "harmonic_residual_norm_truth={}",
        branch.harmonic_residual_norm_truth
    )?;
    writeln!(
        summary,
        "harmonic_residual_norm_posterior_mean={}",
        branch.harmonic_residual_norm_posterior_mean
    )?;
    writeln!(summary, "prior_sample_count={}", branch.prior_samples.len())?;
    writeln!(
        summary,
        "posterior_sample_count={}",
        branch.posterior_samples.len()
    )?;
    write_branch_variance_ratio_summary(&mut summary, branch)?;

    Ok(())
}

fn write_branch_surface_vector_vtu(
    result: &Genus2Torus1FormHodgeConditioningResult,
    branch: &Hodge1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let truth = Cochain::new(1, result.truth.clone());
    let posterior_mean = Cochain::new(1, branch.posterior_mean.clone());
    let absolute_mean_error = Cochain::new(1, branch.absolute_mean_error.clone());

    let truth_vectors = sample_1form_cell_vectors(&result.coords, &result.topology, &truth)?;
    let posterior_mean_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &posterior_mean)?;
    let absolute_mean_error_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &absolute_mean_error)?;

    let surface = &branch.reconstructed_barycenter_variance;
    let posterior_variance_vectors = surface.posterior_vtk_vectors();
    let prior_variance_vectors = surface.prior_vtk_vectors();
    let posterior_marginal_std = surface.posterior_trace.map(|value| value.max(0.0).sqrt());

    visual_output::write_top_cell_fields(
        branch_dir.join("posterior_mean_surface_vector.vtu"),
        &result.coords,
        &result.topology,
        &[
            ("truth_surface_vector", truth_vectors.as_slice()),
            (
                "posterior_mean_surface_vector",
                posterior_mean_vectors.as_slice(),
            ),
            (
                "absolute_mean_error_surface_vector",
                absolute_mean_error_vectors.as_slice(),
            ),
            (
                "posterior_directional_variance",
                posterior_variance_vectors.as_slice(),
            ),
            (
                "prior_directional_variance",
                prior_variance_vectors.as_slice(),
            ),
        ],
        &[
            ("marginal_variance", surface.posterior_trace.as_slice()),
            ("marginal_std", posterior_marginal_std.as_slice()),
            ("prior_marginal_variance", surface.prior_trace.as_slice()),
            ("marginal_variance_ratio", surface.trace_ratio.as_slice()),
        ],
    )?;
    Ok(())
}

fn write_branch_reconstructed_component_variance_vtu(
    result: &Genus2Torus1FormHodgeConditioningResult,
    branch: &Hodge1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let surface = &branch.reconstructed_barycenter_variance;
    let component_count = surface
        .prior_components
        .len()
        .min(surface.posterior_components.len());
    let mut fields = Vec::<(String, FeecVector)>::with_capacity(3 * component_count + 3);

    for component_index in 0..component_count {
        let label = ambient_component_label(component_index);
        let prior = &surface.prior_components[component_index];
        let posterior = &surface.posterior_components[component_index];
        fields.push((format!("prior_var_{label}"), prior.clone()));
        fields.push((format!("post_var_{label}"), posterior.clone()));
        fields.push((format!("ratio_{label}"), ratio_vector(posterior, prior)));
    }
    fields.push(("trace_prior".to_string(), surface.prior_trace.clone()));
    fields.push(("trace_post".to_string(), surface.posterior_trace.clone()));
    fields.push(("trace_ratio".to_string(), surface.trace_ratio.clone()));

    let field_refs = fields
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect::<Vec<_>>();
    visual_output::write_top_cell_scalar_fields(
        branch_dir.join("reconstructed_component_variance.vtu"),
        &result.coords,
        &result.topology,
        &field_refs,
    )?;
    Ok(())
}

fn write_branch_variance_ratio_summary(
    writer: &mut impl Write,
    branch: &Hodge1FormBranchResult,
) -> io::Result<()> {
    let surface = &branch.reconstructed_barycenter_variance;
    writeln!(
        writer,
        "ambient_trace_prior_variance_mean={}",
        mean_vector(&surface.prior_trace)
    )?;
    writeln!(
        writer,
        "ambient_trace_posterior_variance_mean={}",
        mean_vector(&surface.posterior_trace)
    )?;
    writeln!(
        writer,
        "ambient_trace_variance_ratio_mean={}",
        mean_vector(&surface.trace_ratio)
    )?;
    writeln!(
        writer,
        "ambient_trace_variance_ratio_min={}",
        min_vector(&surface.trace_ratio)
    )?;
    writeln!(
        writer,
        "ambient_trace_variance_ratio_max={}",
        max_vector(&surface.trace_ratio)
    )?;

    let component_count = surface
        .prior_components
        .len()
        .min(surface.posterior_components.len());
    for component_index in 0..component_count {
        let label = ambient_component_label(component_index);
        let ratio = ratio_vector(
            &surface.posterior_components[component_index],
            &surface.prior_components[component_index],
        );
        writeln!(
            writer,
            "ambient_{label}_variance_ratio_mean={}",
            mean_vector(&ratio)
        )?;
        writeln!(
            writer,
            "ambient_{label}_variance_ratio_min={}",
            min_vector(&ratio)
        )?;
        writeln!(
            writer,
            "ambient_{label}_variance_ratio_max={}",
            max_vector(&ratio)
        )?;
    }

    Ok(())
}

fn ambient_component_label(index: usize) -> String {
    match index {
        0 => "x".to_string(),
        1 => "y".to_string(),
        2 => "z".to_string(),
        _ => format!("c{index}"),
    }
}

fn ratio_vector(numerator: &FeecVector, denominator: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(
        numerator.len(),
        (0..numerator.len()).map(|i| safe_ratio(numerator[i], denominator[i])),
    )
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= EPS {
        0.0
    } else {
        numerator / denominator
    }
}

fn mean_vector(values: &FeecVector) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn min_vector(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_vector(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn genus2_1form_hodge_conditioning_builds_cycles_and_outputs() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = Genus2Torus1FormHodgeConditioningConfig::default();
        config.sample_count = 1;

        let result = run_genus2_1form_hodge_conditioning(&config)
            .expect("genus-2 Hodge conditioning should run");

        assert_eq!(result.topology_summary.euler_characteristic, -2);
        assert_eq!(result.topology_summary.b0, 1);
        assert_eq!(result.topology_summary.b1, 4);
        assert_eq!(result.topology_summary.b2, 1);
        assert_eq!(result.topology_summary.boundary_edge_count, 0);
        assert_eq!(result.topology_summary.nonmanifold_edge_count, 0);
        assert_eq!(result.cycle_observations.len(), 4);
        assert_eq!(result.cycle_harmonic_pairing.rank(1e-8), 4);
        assert_eq!(result.harmonic_basis.ncols(), 4);
        assert_eq!(result.raw_prior_sample.len(), result.topology.edges().len());
        assert!(result
            .raw_prior_sample
            .iter()
            .all(|value| value.is_finite()));

        for cycle in &result.cycle_observations {
            assert!(
                cycle.closure_residual_l1 <= 1e-10,
                "cycle {} should be closed",
                cycle.name
            );
            assert!(cycle.edge_count > 0);
            assert!(cycle.path_vertices.len() >= 2);
            assert_eq!(cycle.path_vertices.first(), cycle.path_vertices.last());
        }

        for branch in [&result.exact, &result.coexact, &result.harmonic] {
            assert_eq!(branch.observation_count, 4);
            assert_eq!(branch.prior_samples.len(), 1);
            assert_eq!(branch.posterior_samples.len(), 1);
            assert!(branch.posterior_mean.iter().all(|value| value.is_finite()));
            assert!(branch
                .observation_prior_variance
                .iter()
                .all(|value| value.is_finite()));
            assert!(branch
                .observation_posterior_variance
                .iter()
                .all(|value| value.is_finite()));
            for i in 0..branch.observation_count {
                assert!(
                    branch.observation_posterior_variance[i]
                        <= branch.observation_prior_variance[i] + 1e-8,
                    "branch {} cycle {i} posterior variance should not exceed prior variance",
                    branch.kind.as_str()
                );
            }
        }
        assert_eq!(result.harmonic.latent_dimension, 4);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!("feg_infer_genus2_hodge_{stamp}"));
        write_genus2_1form_hodge_conditioning_outputs(&result, &out_dir)
            .expect("genus-2 outputs should write");
        assert!(out_dir.join("topology_summary.txt").is_file());
        assert!(out_dir.join("cycle_observations.csv").is_file());
        assert!(out_dir.join("cycle_branch_summary.csv").is_file());
        assert!(out_dir.join("cycle_paths.vtu").is_file());
        assert!(out_dir
            .join("raw_full_matern_sample_edge_fields.vtu")
            .is_file());
        assert!(out_dir
            .join("raw_full_matern_sample_surface_vector.vtu")
            .is_file());
        let cycle_paths = fs::read_to_string(out_dir.join("cycle_paths.vtu"))
            .expect("cycle path overlay should read");
        assert!(cycle_paths.contains("VTKFile type=\"UnstructuredGrid\""));
        assert!(cycle_paths.contains("Name=\"harmonic_posterior_lower_95\""));
        assert!(cycle_paths.contains("Name=\"harmonic_posterior_upper_95\""));
        let raw_surface =
            fs::read_to_string(out_dir.join("raw_full_matern_sample_surface_vector.vtu"))
                .expect("raw sample surface vector VTU should read");
        assert!(raw_surface.contains("Name=\"raw_full_matern_sample_surface_vector\""));
        for branch in ["exact", "coexact", "harmonic"] {
            assert!(out_dir.join(branch).join("edge_fields.vtu").is_file());
            assert!(out_dir
                .join(branch)
                .join("posterior_mean_surface_vector.vtu")
                .is_file());
            let component_variance_path = out_dir
                .join(branch)
                .join("reconstructed_component_variance.vtu");
            assert!(component_variance_path.is_file());
            let component_variance =
                fs::read_to_string(component_variance_path).expect("component VTU should read");
            assert!(component_variance.contains("Name=\"ratio_x\""));
            assert!(component_variance.contains("Name=\"ratio_y\""));
            assert!(component_variance.contains("Name=\"ratio_z\""));
            assert!(component_variance.contains("Name=\"trace_ratio\""));
            assert!(out_dir.join(branch).join("summary.txt").is_file());
            let summary =
                fs::read_to_string(out_dir.join(branch).join("summary.txt")).expect("summary read");
            assert!(summary.contains("ambient_trace_variance_ratio_mean="));
        }
        fs::remove_dir_all(out_dir).expect("temporary output directory should clean up");
    }
}
