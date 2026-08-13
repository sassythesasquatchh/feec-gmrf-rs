//! Reusable FEEC de Rham helpers for explicit sparse Gaussian models.
//!
//! This module deliberately stays geometry-agnostic: it builds Whitney-form
//! Matérn precisions, exterior derivative/codifferential diagnostics, approximate
//! sparse Hodge projections, and linear integral observation operators.

use common::linalg::nalgebra::{
    bilinear_form_sparse, CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix,
    Vector as FeecVector,
};
use ddf::ManifoldComplexExt;
use exterior::ExteriorGrade;
use feg_infer::sparse::{
    add_sparse, diag_matrix, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec, invert_diag,
    lumped_diag, scale_matrix,
};
use formoniq::{assemble::assemble_galmat, operators::HodgeMassElmat};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::{SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector};
use gmrf_core::Gmrf;
use manifold::{
    geometry::{
        coord::{mesh::MeshCoords, simplex::SimplexCoords},
        metric::mesh::MeshLengths,
    },
    topology::{complex::Complex, simplex::Simplex},
};
use rand::SeedableRng;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap},
};

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FormMassInverse {
    #[default]
    Diagonal,
    RowSumLumped,
}

#[derive(Debug, Clone, Copy)]
pub struct FormMaternConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: FormMassInverse,
}

impl Default for FormMaternConfig {
    fn default() -> Self {
        Self {
            kappa: 4.0,
            tau: 1.0,
            mass_inverse: FormMassInverse::Diagonal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FormHodgeLaplacian {
    pub grade: ExteriorGrade,
    pub mass: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone)]
pub struct FormMaternSystem {
    pub grade: ExteriorGrade,
    pub mass: FeecCsr,
    pub laplacian: FeecCsr,
    pub system_matrix: FeecCsr,
    pub mass_inverse: FeecCsr,
    pub precision: FeecCsr,
}

#[derive(Debug, Clone)]
pub struct WeightedSimplexSupport {
    pub label: String,
    pub entries: Vec<(usize, f64)>,
}

impl WeightedSimplexSupport {
    pub fn new(label: impl Into<String>, entries: Vec<(usize, f64)>) -> Self {
        Self {
            label: label.into(),
            entries,
        }
    }

    pub fn l1_weight(&self) -> f64 {
        self.entries.iter().map(|(_, weight)| weight.abs()).sum()
    }
}

#[derive(Debug, Clone)]
pub struct ObservationOperator {
    pub matrix: FeecCsr,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct HodgeProjectionConfig {
    pub ridge: f64,
}

impl Default for HodgeProjectionConfig {
    fn default() -> Self {
        Self { ridge: 1e-10 }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeProjection {
    pub exact: FeecVector,
    pub coexact: FeecVector,
    pub harmonic: FeecVector,
    pub reconstruction_error: f64,
    pub orthogonality_error: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct VarianceProbeConfig {
    pub probe_count: usize,
    pub seed: u64,
}

impl Default for VarianceProbeConfig {
    fn default() -> Self {
        Self {
            probe_count: 32,
            seed: 1729,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearConditioningOutput {
    pub observations: FeecVector,
    pub posterior_precision: GmrfSparseMatrix,
    pub information: GmrfVector,
    pub posterior_mean: FeecVector,
    pub posterior_observations: FeecVector,
    pub observation_residual: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
}

pub fn build_hodge_laplacian_form(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
) -> Result<FormHodgeLaplacian, String> {
    if grade > topology.dim() {
        return Err(format!(
            "form grade {grade} exceeds topology dimension {}",
            topology.dim()
        ));
    }
    let galmats = formoniq::problems::hodge_laplace::MixedGalmats::compute(topology, metric, grade);
    let mass = galmats.mass_u_csr();
    let laplacian = hodge_laplacian_schur_complement_lumped(&galmats, mass.nrows());
    Ok(FormHodgeLaplacian {
        grade,
        mass,
        laplacian,
    })
}

pub fn build_matern_system_matrix_form(hodge: &FormHodgeLaplacian, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass, kappa2))
}

pub fn build_matern_mass_inverse_form(mass: &FeecCsr, strategy: FormMassInverse) -> FeecCsr {
    match strategy {
        FormMassInverse::Diagonal => diag_matrix(&invert_diag(&matrix_diag(mass))),
        FormMassInverse::RowSumLumped => diag_matrix(&invert_diag(&lumped_diag(mass))),
    }
}

pub fn build_matern_precision_form(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
    config: FormMaternConfig,
) -> Result<FormMaternSystem, String> {
    validate_matern_config(config)?;
    let hodge = build_hodge_laplacian_form(topology, metric, grade)?;
    let system_matrix = build_matern_system_matrix_form(&hodge, config.kappa);
    let mass_inverse = build_matern_mass_inverse_form(&hodge.mass, config.mass_inverse);
    let middle = &mass_inverse * &system_matrix;
    let mut precision = &system_matrix * &middle;
    if (config.tau - 1.0).abs() > f64::EPSILON {
        precision = scale_matrix(&precision, config.tau * config.tau);
    }

    Ok(FormMaternSystem {
        grade,
        mass: hodge.mass,
        laplacian: hodge.laplacian,
        system_matrix,
        mass_inverse,
        precision,
    })
}

pub fn mass_matrix_form(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
) -> Result<FeecCsr, String> {
    if grade > topology.dim() {
        return Err(format!(
            "form grade {grade} exceeds topology dimension {}",
            topology.dim()
        ));
    }
    Ok(FeecCsr::from(&assemble_galmat(
        topology,
        metric,
        HodgeMassElmat::new(topology.dim(), grade),
    )))
}

pub fn exterior_derivative_matrix(topology: &Complex, grade: ExteriorGrade) -> FeecCsr {
    let ncols = topology.nsimplices(grade);
    if grade >= topology.dim() {
        return FeecCsr::from(&FeecCoo::new(0, ncols));
    }
    FeecCsr::from(&topology.exterior_derivative_operator(grade))
}

pub fn derivative(topology: &Complex, grade: ExteriorGrade, values: &FeecVector) -> FeecVector {
    let d = exterior_derivative_matrix(topology, grade);
    assert_eq!(d.ncols(), values.len());
    sparse_matvec(&d, values)
}

pub fn codifferential_transform(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
) -> Result<FeecCsr, String> {
    let ncols = topology.nsimplices(grade);
    if grade == 0 {
        return Ok(FeecCsr::from(&FeecCoo::new(0, ncols)));
    }
    if grade > topology.dim() {
        return Err(format!(
            "form grade {grade} exceeds topology dimension {}",
            topology.dim()
        ));
    }

    let d_prev = exterior_derivative_matrix(topology, grade - 1);
    let mass_k = mass_matrix_form(topology, metric, grade)?;
    let mass_prev = mass_matrix_form(topology, metric, grade - 1)?;
    let dt_mass = d_prev.transpose() * mass_k;
    Ok(scale_rows(&dt_mass, &invert_diag(&lumped_diag(&mass_prev))))
}

pub fn codifferential(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
    values: &FeecVector,
) -> Result<FeecVector, String> {
    let delta = codifferential_transform(topology, metric, grade)?;
    if delta.ncols() != values.len() {
        return Err(format!(
            "codifferential input length {} does not match grade {grade} dimension {}",
            values.len(),
            delta.ncols()
        ));
    }
    Ok(sparse_matvec(&delta, values))
}

pub fn coexact_transform(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
) -> Result<FeecCsr, String> {
    if grade >= topology.dim() {
        return Ok(FeecCsr::from(&FeecCoo::new(topology.nsimplices(grade), 0)));
    }
    codifferential_transform(topology, metric, grade + 1)
}

pub fn hodge_project(
    topology: &Complex,
    metric: &MeshLengths,
    grade: ExteriorGrade,
    values: &FeecVector,
    config: HodgeProjectionConfig,
) -> Result<HodgeProjection, String> {
    if values.len() != topology.nsimplices(grade) {
        return Err(format!(
            "value length {} does not match grade {grade} dimension {}",
            values.len(),
            topology.nsimplices(grade)
        ));
    }
    let mass = mass_matrix_form(topology, metric, grade)?;

    let exact_transform = if grade == 0 {
        FeecCsr::from(&FeecCoo::new(values.len(), 0))
    } else {
        exterior_derivative_matrix(topology, grade - 1)
    };
    let exact = project_onto_transform(&mass, &exact_transform, values, config.ridge)?;

    let after_exact = values - &exact;
    let coexact_transform = coexact_transform(topology, metric, grade)?;
    let coexact = project_onto_transform(&mass, &coexact_transform, &after_exact, config.ridge)?;
    let harmonic = values - &exact - &coexact;

    let reconstructed = &exact + &coexact + &harmonic;
    let reconstruction_error = relative_norm(&(values - reconstructed), values);
    let orthogonality_error = max_pairwise_mass_orthogonality(&mass, [&exact, &coexact, &harmonic]);

    Ok(HodgeProjection {
        exact,
        coexact,
        harmonic,
        reconstruction_error,
        orthogonality_error,
    })
}

pub fn supports_to_operator(
    dimension: usize,
    supports: &[WeightedSimplexSupport],
) -> ObservationOperator {
    let mut coo = FeecCoo::new(supports.len(), dimension);
    for (row, support) in supports.iter().enumerate() {
        for &(col, weight) in &support.entries {
            if col < dimension && weight != 0.0 {
                coo.push(row, col, weight);
            }
        }
    }
    ObservationOperator {
        matrix: FeecCsr::from(&coo),
        labels: supports
            .iter()
            .map(|support| support.label.clone())
            .collect(),
    }
}

pub fn point_observation_operator(
    dimension: usize,
    labels_and_indices: &[(&str, usize)],
) -> ObservationOperator {
    let supports = labels_and_indices
        .iter()
        .map(|(label, index)| WeightedSimplexSupport::new(*label, vec![(*index, 1.0)]))
        .collect::<Vec<_>>();
    supports_to_operator(dimension, &supports)
}

pub fn volume_average_0form_operator(
    topology: &Complex,
    coords: &MeshCoords,
    label: impl Into<String>,
) -> ObservationOperator {
    let support = volume_average_0form_support(topology, coords, label, |_| true)
        .expect("whole-domain volume average should select at least one cell");
    supports_to_operator(topology.nsimplices(0), &[support])
}

pub fn volume_average_0form_support<F>(
    topology: &Complex,
    coords: &MeshCoords,
    label: impl Into<String>,
    cell_predicate: F,
) -> Result<WeightedSimplexSupport, String>
where
    F: Fn([f64; 3]) -> bool,
{
    let label = label.into();
    let mut weights = vec![0.0; topology.nsimplices(0)];
    let mut total_volume = 0.0;
    for cell in topology.cells().handle_iter() {
        let bary = simplex_barycenter(coords, &cell);
        if !cell_predicate(bary) {
            continue;
        }
        let simplex = SimplexCoords::from_simplex_and_coords(&cell, coords);
        let volume = simplex.vol();
        total_volume += volume;
        let vertex_weight = volume / (cell.vertices.len() as f64);
        for &vertex in &cell.vertices {
            weights[vertex] += vertex_weight;
        }
    }
    if total_volume > 0.0 {
        for weight in &mut weights {
            *weight /= total_volume;
        }
    }
    if total_volume <= 0.0 {
        return Err(format!(
            "0-form volume average '{}' selected no volume",
            label
        ));
    }
    let entries = weights
        .into_iter()
        .enumerate()
        .filter(|(_, weight)| weight.abs() > EPS)
        .collect::<Vec<_>>();
    Ok(WeightedSimplexSupport::new(label, entries))
}

pub fn nearest_vertex(coords: &MeshCoords, target: [f64; 3]) -> usize {
    (0..coords.nvertices())
        .min_by(|&a, &b| {
            squared_distance(coord3(coords, a), target)
                .partial_cmp(&squared_distance(coord3(coords, b), target))
                .unwrap_or(Ordering::Equal)
        })
        .expect("mesh must contain at least one vertex")
}

pub fn edge_path_integral_support(
    topology: &Complex,
    coords: &MeshCoords,
    label: impl Into<String>,
    target_points: &[[f64; 3]],
    close_path: bool,
) -> Result<WeightedSimplexSupport, String> {
    if target_points.len() < 2 {
        return Err("at least two target points are required for an edge path".to_string());
    }
    let graph = EdgeGraph::new(topology, coords)?;
    let mut target_vertices = target_points
        .iter()
        .map(|&point| nearest_vertex(coords, point))
        .collect::<Vec<_>>();
    if close_path {
        target_vertices.push(target_vertices[0]);
    }

    let mut entries = Vec::new();
    for pair in target_vertices.windows(2) {
        let path = graph.shortest_path(pair[0], pair[1])?;
        for edge_pair in path.windows(2) {
            let (edge_index, sign) = graph.oriented_edge(edge_pair[0], edge_pair[1])?;
            entries.push((edge_index, sign));
        }
    }
    Ok(WeightedSimplexSupport::new(label, entries))
}

pub fn boundary_face_integral_support<F, G>(
    topology: &Complex,
    coords: &MeshCoords,
    label: impl Into<String>,
    predicate: F,
    orientation: G,
) -> Result<WeightedSimplexSupport, String>
where
    F: Fn([f64; 3]) -> bool,
    G: Fn([f64; 3], [f64; 3]) -> f64,
{
    let label = label.into();
    if topology.dim() < 2 {
        return Err("boundary face observations require topology dimension >= 2".to_string());
    }
    let mut entries = Vec::new();
    for face in topology.facets().handle_iter() {
        if face.cocells().count() != 1 {
            continue;
        }
        let bary = simplex_barycenter(coords, &face);
        if !predicate(bary) {
            continue;
        }
        let normal = triangle_normal(coords, &face.vertices);
        let sign = orientation(bary, normal).signum();
        entries.push((face.kidx(), if sign == 0.0 { 1.0 } else { sign }));
    }
    if entries.is_empty() {
        return Err(format!(
            "boundary face observation '{}' selected no faces",
            label
        ));
    }
    Ok(WeightedSimplexSupport::new(label, entries))
}

pub fn top_cell_region_integral_support<F>(
    topology: &Complex,
    coords: &MeshCoords,
    label: impl Into<String>,
    predicate: F,
) -> Result<WeightedSimplexSupport, String>
where
    F: Fn([f64; 3]) -> bool,
{
    let label = label.into();
    let mut entries = Vec::new();
    for cell in topology.cells().handle_iter() {
        let bary = simplex_barycenter(coords, &cell);
        if predicate(bary) {
            let sign = simplex_orientation_sign(coords, &cell);
            entries.push((cell.kidx(), sign));
        }
    }
    if entries.is_empty() {
        return Err(format!(
            "top-cell observation '{}' selected no cells",
            label
        ));
    }
    Ok(WeightedSimplexSupport::new(label, entries))
}

pub fn condition_on_observations(
    prior_precision: &FeecCsr,
    observation_operator: &FeecCsr,
    truth: &FeecVector,
    noise_variance: f64,
    variance_probe: VarianceProbeConfig,
) -> Result<LinearConditioningOutput, String> {
    if observation_operator.ncols() != truth.len() {
        return Err(format!(
            "observation operator columns {} do not match truth dimension {}",
            observation_operator.ncols(),
            truth.len()
        ));
    }
    if !noise_variance.is_finite() || noise_variance <= 0.0 {
        return Err("noise variance must be finite and positive".to_string());
    }
    if variance_probe.probe_count == 0 {
        return Err("variance probe count must be positive".to_string());
    }

    let prior_precision_gmrf = feec_csr_to_gmrf(prior_precision);
    let observation_operator_gmrf = feec_csr_to_gmrf(observation_operator);
    let observations = sparse_matvec(observation_operator, truth);
    let observations_gmrf = feec_vec_to_gmrf(&observations);

    let prior_factor = prior_precision_gmrf
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let mut prior = Gmrf::from_mean_and_precision(
        GmrfVector::zeros(prior_precision.nrows()),
        prior_precision_gmrf.clone(),
    )
    .map_err(|err| format!("failed to build prior GMRF: {err}"))?
    .with_precision_sqrt(prior_factor);
    let mut prior_rng = rand::rngs::StdRng::seed_from_u64(variance_probe.seed);
    let prior_variance = prior
        .hutchinson_variances(variance_probe.probe_count, &mut prior_rng)
        .map_err(|err| format!("failed to estimate prior variances: {err}"))?;

    let (posterior_precision, information) = apply_gaussian_observations(
        &prior_precision_gmrf,
        &observation_operator_gmrf,
        &observations_gmrf,
        None,
        noise_variance,
    );
    let posterior_factor = posterior_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor posterior precision: {err}"))?;
    let mut posterior = Gmrf::from_information_and_precision_with_sqrt(
        information.clone(),
        posterior_precision.clone(),
        posterior_factor,
    )
    .map_err(|err| format!("failed to build posterior GMRF: {err}"))?;
    let posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let posterior_observations = sparse_matvec(observation_operator, &posterior_mean);
    let observation_residual = &posterior_observations - &observations;

    let mut posterior_rng = rand::rngs::StdRng::seed_from_u64(variance_probe.seed + 1);
    let posterior_variance = posterior
        .hutchinson_variances(variance_probe.probe_count, &mut posterior_rng)
        .map_err(|err| format!("failed to estimate posterior variances: {err}"))?;

    Ok(LinearConditioningOutput {
        observations,
        posterior_precision,
        information,
        posterior_mean,
        posterior_observations,
        observation_residual,
        prior_variance: gmrf_vec_to_feec(&prior_variance),
        posterior_variance: gmrf_vec_to_feec(&posterior_variance),
    })
}

pub fn sample_zero_mean_precision(precision: &FeecCsr, seed: u64) -> Result<FeecVector, String> {
    let precision = feec_csr_to_gmrf(precision);
    let factor = precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor precision for sampling: {err}"))?;
    let gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision)
        .map_err(|err| format!("failed to build sampling GMRF: {err}"))?
        .with_precision_sqrt(factor);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    gmrf.sample_one_solve(&mut rng)
        .map(|sample| gmrf_vec_to_feec(&sample))
        .map_err(|err| format!("failed to draw sample: {err}"))
}

pub fn estimate_marginal_variance(
    precision: &FeecCsr,
    probe: VarianceProbeConfig,
) -> Result<FeecVector, String> {
    if probe.probe_count == 0 {
        return Err("variance probe count must be positive".to_string());
    }
    let precision = feec_csr_to_gmrf(precision);
    let factor = precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor precision for variance estimate: {err}"))?;
    let mut gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision)
        .map_err(|err| format!("failed to build variance GMRF: {err}"))?
        .with_precision_sqrt(factor);
    let mut rng = rand::rngs::StdRng::seed_from_u64(probe.seed);
    gmrf.hutchinson_variances(probe.probe_count, &mut rng)
        .map(|variance| gmrf_vec_to_feec(&variance))
        .map_err(|err| format!("failed to estimate variance: {err}"))
}

pub fn betti_numbers(topology: &Complex) -> Vec<usize> {
    if topology.dim() == 3 {
        return betti_numbers_3d_from_boundary(topology);
    }
    (0..=topology.dim())
        .map(|grade| betti_number(topology, grade))
        .collect()
}

pub fn betti_numbers_3d_from_boundary(topology: &Complex) -> Vec<usize> {
    assert_eq!(topology.dim(), 3);
    let b0 = vertex_connected_components(topology);
    let boundary_components = boundary_connected_components(topology);
    let b2 = boundary_components.saturating_sub(b0);
    let chi = topology.nsimplices(0) as isize - topology.nsimplices(1) as isize
        + topology.nsimplices(2) as isize
        - topology.nsimplices(3) as isize;
    let b1 = (b0 as isize + b2 as isize - chi).max(0) as usize;
    vec![b0, b1, b2, 0]
}

pub fn betti_number(topology: &Complex, grade: ExteriorGrade) -> usize {
    assert!(grade <= topology.dim());
    let boundary_this = FeecMatrix::from(&topology.boundary_operator(grade));
    let cycles = boundary_this.ncols() - rank_or_zero(&boundary_this);
    let boundaries = if grade < topology.dim() {
        rank_or_zero(&FeecMatrix::from(&topology.boundary_operator(grade + 1)))
    } else {
        0
    };
    cycles - boundaries
}

pub fn sparse_matvec(matrix: &FeecCsr, vector: &FeecVector) -> FeecVector {
    assert_eq!(matrix.ncols(), vector.len());
    let mut out = FeecVector::zeros(matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        out[row] += *value * vector[col];
    }
    out
}

pub fn median_positive(values: &FeecVector) -> Option<f64> {
    let mut positives = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if positives.is_empty() {
        return None;
    }
    positives.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(positives[positives.len() / 2])
}

pub fn coord3(coords: &MeshCoords, vertex: usize) -> [f64; 3] {
    let coord = coords.coord(vertex);
    [
        coord[0],
        if coords.dim() > 1 { coord[1] } else { 0.0 },
        if coords.dim() > 2 { coord[2] } else { 0.0 },
    ]
}

pub fn simplex_barycenter(
    coords: &MeshCoords,
    simplex: &impl std::ops::Deref<Target = Simplex>,
) -> [f64; 3] {
    let simplex_coords = SimplexCoords::from_simplex_and_coords(simplex, coords);
    let bary = simplex_coords.barycenter();
    [
        bary[0],
        if bary.len() > 1 { bary[1] } else { 0.0 },
        if bary.len() > 2 { bary[2] } else { 0.0 },
    ]
}

pub fn squared_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

pub fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    squared_distance(a, b).sqrt()
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn validate_matern_config(config: FormMaternConfig) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    Ok(())
}

fn project_onto_transform(
    mass: &FeecCsr,
    transform: &FeecCsr,
    target: &FeecVector,
    ridge: f64,
) -> Result<FeecVector, String> {
    if transform.nrows() != target.len() || mass.nrows() != target.len() {
        return Err("projection dimensions do not align".to_string());
    }
    if transform.ncols() == 0 {
        return Ok(FeecVector::zeros(target.len()));
    }

    let mass_transform = mass * transform;
    let lhs = transform.transpose() * &mass_transform;
    let lhs = add_scaled_identity(&lhs, stabilized_ridge(&lhs, ridge));
    let mass_target = sparse_matvec(mass, target);
    let rhs = transform.transpose() * mass_target;

    let solution = solve_spd(&lhs, &rhs)?;
    Ok(sparse_matvec(transform, &solution))
}

fn solve_spd(matrix: &FeecCsr, rhs: &FeecVector) -> Result<FeecVector, String> {
    let factor = feec_csr_to_gmrf(matrix)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor SPD projection system: {err}"))?;
    let rhs = feec_vec_to_gmrf(rhs);
    factor
        .solve(&rhs)
        .map(|solution| gmrf_vec_to_feec(&solution))
        .map_err(|err| format!("failed to solve SPD projection system: {err}"))
}

fn stabilized_ridge(matrix: &FeecCsr, ridge: f64) -> f64 {
    if ridge <= 0.0 {
        return 0.0;
    }
    let diag = matrix_diag(matrix);
    let mean_diag = if diag.is_empty() {
        1.0
    } else {
        diag.iter().map(|value| value.abs()).sum::<f64>() / (diag.len() as f64)
    };
    ridge * mean_diag.max(1.0)
}

fn add_scaled_identity(matrix: &FeecCsr, scale: f64) -> FeecCsr {
    if scale == 0.0 {
        return matrix.clone();
    }
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for i in 0..matrix.nrows().min(matrix.ncols()) {
        coo.push(i, i, scale);
    }
    FeecCsr::from(&coo)
}

fn matrix_diag(matrix: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; matrix.nrows().min(matrix.ncols())];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diag[row] += *value;
        }
    }
    diag
}

fn rank_or_zero(matrix: &FeecMatrix) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        0
    } else {
        matrix.rank(1e-9)
    }
}

fn vertex_connected_components(topology: &Complex) -> usize {
    let mut dsu = DisjointSet::new(topology.nsimplices(0));
    for edge in topology.edges().handle_iter() {
        dsu.union(edge.vertices[0], edge.vertices[1]);
    }
    dsu.component_count()
}

fn boundary_connected_components(topology: &Complex) -> usize {
    let mut dsu = DisjointSet::new(topology.nsimplices(0));
    let mut on_boundary = vec![false; topology.nsimplices(0)];
    for face in topology.facets().handle_iter() {
        if face.cocells().count() != 1 {
            continue;
        }
        for &vertex in &face.vertices {
            on_boundary[vertex] = true;
        }
        dsu.union(face.vertices[0], face.vertices[1]);
        dsu.union(face.vertices[1], face.vertices[2]);
        dsu.union(face.vertices[2], face.vertices[0]);
    }
    let mut roots = std::collections::BTreeSet::new();
    for (vertex, is_boundary) in on_boundary.into_iter().enumerate() {
        if is_boundary {
            roots.insert(dsu.find(vertex));
        }
    }
    roots.len()
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }

    fn component_count(mut self) -> usize {
        let mut roots = std::collections::BTreeSet::new();
        for index in 0..self.parent.len() {
            roots.insert(self.find(index));
        }
        roots.len()
    }
}

fn scale_rows(matrix: &FeecCsr, row_scales: &[f64]) -> FeecCsr {
    assert_eq!(matrix.nrows(), row_scales.len());
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        let scaled = *value * row_scales[row];
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn hodge_laplacian_schur_complement_lumped(
    galmats: &formoniq::problems::hodge_laplace::MixedGalmats,
    dimension: usize,
) -> FeecCsr {
    let codifdif = if galmats.codifdif_u().nrows() == 0 {
        FeecCsr::from(&FeecCoo::new(dimension, dimension))
    } else {
        FeecCsr::from(galmats.codifdif_u())
    };

    if galmats.mass_sigma().nrows() == 0 {
        return codifdif;
    }

    let mass_sigma = FeecCsr::from(galmats.mass_sigma());
    let dif_sigma = FeecCsr::from(galmats.dif_sigma());
    let codif_u = FeecCsr::from(galmats.codif_u());
    let mass_sigma_inv = invert_diag(&lumped_diag(&mass_sigma));
    let codif_u_scaled = scale_rows(&codif_u, &mass_sigma_inv);
    let schur = &dif_sigma * &codif_u_scaled;
    add_sparse(&codifdif, &schur)
}

fn relative_norm(diff: &FeecVector, reference: &FeecVector) -> f64 {
    diff.norm() / reference.norm().max(EPS)
}

fn max_pairwise_mass_orthogonality(mass: &FeecCsr, vectors: [&FeecVector; 3]) -> f64 {
    let mut max_error: f64 = 0.0;
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let denom = mass_norm(mass, vectors[i]) * mass_norm(mass, vectors[j]);
            if denom <= EPS {
                continue;
            }
            max_error =
                max_error.max(bilinear_form_sparse(mass, vectors[i], vectors[j]).abs() / denom);
        }
    }
    max_error
}

fn mass_norm(mass: &FeecCsr, vector: &FeecVector) -> f64 {
    bilinear_form_sparse(mass, vector, vector).abs().sqrt()
}

fn triangle_normal(coords: &MeshCoords, vertices: &[usize]) -> [f64; 3] {
    assert_eq!(vertices.len(), 3);
    let p0 = coord3(coords, vertices[0]);
    let p1 = coord3(coords, vertices[1]);
    let p2 = coord3(coords, vertices[2]);
    cross(sub(p1, p0), sub(p2, p0))
}

fn simplex_orientation_sign(
    coords: &MeshCoords,
    simplex: &impl std::ops::Deref<Target = Simplex>,
) -> f64 {
    let simplex_coords = SimplexCoords::from_simplex_and_coords(simplex, coords);
    simplex_coords.det().signum()
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

#[derive(Debug, Clone)]
struct EdgeGraph {
    adjacency: Vec<Vec<(usize, f64)>>,
    edge_by_vertices: BTreeMap<(usize, usize), usize>,
}

impl EdgeGraph {
    fn new(topology: &Complex, coords: &MeshCoords) -> Result<Self, String> {
        if topology.dim() < 1 {
            return Err("edge graph requires topology dimension >= 1".to_string());
        }
        let mut adjacency = vec![Vec::new(); topology.nsimplices(0)];
        let mut edge_by_vertices = BTreeMap::new();
        for edge in topology.edges().handle_iter() {
            let a = edge.vertices[0];
            let b = edge.vertices[1];
            let length = distance(coord3(coords, a), coord3(coords, b));
            adjacency[a].push((b, length));
            adjacency[b].push((a, length));
            edge_by_vertices.insert((a.min(b), a.max(b)), edge.kidx());
        }
        Ok(Self {
            adjacency,
            edge_by_vertices,
        })
    }

    fn shortest_path(&self, start: usize, goal: usize) -> Result<Vec<usize>, String> {
        if start >= self.adjacency.len() || goal >= self.adjacency.len() {
            return Err("path endpoint is outside the vertex range".to_string());
        }
        if start == goal {
            return Ok(vec![start]);
        }

        let mut dist = vec![f64::INFINITY; self.adjacency.len()];
        let mut prev = vec![None; self.adjacency.len()];
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
            if cost > dist[vertex] {
                continue;
            }
            for &(next, weight) in &self.adjacency[vertex] {
                let next_cost = cost + weight;
                if next_cost < dist[next] {
                    dist[next] = next_cost;
                    prev[next] = Some(vertex);
                    heap.push(QueueState {
                        cost: next_cost,
                        vertex: next,
                    });
                }
            }
        }

        if !dist[goal].is_finite() {
            return Err("no path found between target vertices".to_string());
        }
        let mut path = Vec::new();
        let mut current = goal;
        path.push(current);
        while let Some(parent) = prev[current] {
            current = parent;
            path.push(current);
        }
        path.reverse();
        Ok(path)
    }

    fn oriented_edge(&self, from: usize, to: usize) -> Result<(usize, f64), String> {
        let key = (from.min(to), from.max(to));
        let edge = self
            .edge_by_vertices
            .get(&key)
            .copied()
            .ok_or_else(|| "path contains a non-edge vertex pair".to_string())?;
        let sign = if from <= to { 1.0 } else { -1.0 };
        Ok((edge, sign))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn test_mesh() -> (Complex, MeshCoords, MeshLengths) {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        (topology, coords, metric)
    }

    fn max_abs(vec: &FeecVector) -> f64 {
        vec.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    #[test]
    fn incidence_operators_square_to_zero_in_3d() {
        let (topology, _coords, _metric) = test_mesh();
        for grade in 0..2 {
            let d0 = exterior_derivative_matrix(&topology, grade);
            let d1 = exterior_derivative_matrix(&topology, grade + 1);
            let composed = &d1 * &d0;
            for (_row, _col, value) in composed.triplet_iter() {
                assert!(value.abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn matern_precision_builds_and_factorizes_for_all_3d_grades() {
        let (topology, _coords, metric) = test_mesh();
        for grade in 0..=3 {
            let system = build_matern_precision_form(
                &topology,
                &metric,
                grade,
                FormMaternConfig {
                    kappa: 2.0,
                    tau: 1.0,
                    mass_inverse: FormMassInverse::Diagonal,
                },
            )
            .expect("Matérn system should build");
            assert_eq!(system.precision.nrows(), topology.nsimplices(grade));
            assert_eq!(system.precision.ncols(), topology.nsimplices(grade));
            feec_csr_to_gmrf(&system.precision)
                .cholesky_sqrt_lower()
                .expect("precision should be SPD");
        }
    }

    #[test]
    fn codifferential_is_adjoint_with_lumped_lower_mass() {
        let (topology, _coords, metric) = test_mesh();
        for grade in 1..=3 {
            let d_prev = exterior_derivative_matrix(&topology, grade - 1);
            let mass_k = mass_matrix_form(&topology, &metric, grade).unwrap();
            let mass_prev = mass_matrix_form(&topology, &metric, grade - 1).unwrap();
            let delta = codifferential_transform(&topology, &metric, grade).unwrap();
            let a = FeecVector::from_iterator(
                topology.nsimplices(grade - 1),
                (0..topology.nsimplices(grade - 1)).map(|i| 0.25 + i as f64),
            );
            let u = FeecVector::from_iterator(
                topology.nsimplices(grade),
                (0..topology.nsimplices(grade)).map(|i| -0.5 + 0.1 * i as f64),
            );
            let lhs = bilinear_form_sparse(&mass_k, &sparse_matvec(&d_prev, &a), &u);
            let lower_lumped = diag_matrix(&lumped_diag(&mass_prev));
            let rhs = bilinear_form_sparse(&lower_lumped, &a, &sparse_matvec(&delta, &u));
            assert!((lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0));
        }
    }

    #[test]
    fn observation_operators_sum_synthetic_cochains() {
        let supports = vec![
            WeightedSimplexSupport::new("line", vec![(0, 1.0), (2, -1.0)]),
            WeightedSimplexSupport::new("patch", vec![(1, 0.5), (3, 0.25)]),
        ];
        let operator = supports_to_operator(4, &supports);
        let x = FeecVector::from_vec(vec![2.0, 4.0, 5.0, 8.0]);
        let y = sparse_matvec(&operator.matrix, &x);
        assert!((y[0] - -3.0).abs() <= 1e-12);
        assert!((y[1] - 4.0).abs() <= 1e-12);
    }

    #[test]
    fn hodge_projection_reconstructs_input() {
        let (topology, _coords, metric) = test_mesh();
        let values = FeecVector::from_iterator(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).map(|i| (i as f64 + 1.0).sin()),
        );
        let projection = hodge_project(
            &topology,
            &metric,
            1,
            &values,
            HodgeProjectionConfig::default(),
        )
        .expect("projection should succeed");
        assert!(projection.reconstruction_error <= 1e-12);
        assert!(projection.orthogonality_error.is_finite());
        assert!(projection.orthogonality_error <= 1.0);
    }

    #[test]
    fn posterior_conditioning_reduces_measured_variance_and_matches_data() {
        let (topology, _coords, metric) = test_mesh();
        let system = build_matern_precision_form(
            &topology,
            &metric,
            0,
            FormMaternConfig {
                kappa: 2.0,
                tau: 1.0,
                mass_inverse: FormMassInverse::Diagonal,
            },
        )
        .unwrap();
        let observation = point_observation_operator(system.precision.nrows(), &[("v0", 0)]);
        let truth = FeecVector::from_iterator(
            system.precision.nrows(),
            (0..system.precision.nrows()).map(|i| i as f64 + 1.0),
        );
        let result = condition_on_observations(
            &system.precision,
            &observation.matrix,
            &truth,
            1e-10,
            VarianceProbeConfig {
                probe_count: 16,
                seed: 9,
            },
        )
        .expect("conditioning should succeed");
        assert!(max_abs(&result.observation_residual) <= 1e-5);
        assert!(result.posterior_variance[0] <= result.prior_variance[0]);
    }

    #[test]
    fn betti_numbers_of_cartesian_ball_are_expected() {
        let (topology, _coords, _metric) = test_mesh();
        assert_eq!(betti_numbers(&topology), vec![1, 0, 0, 0]);
    }
}
