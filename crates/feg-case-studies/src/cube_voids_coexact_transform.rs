//! Coexact 1-form transform diagnostics on a 3D cube with spherical voids.

use crate::de_rham;
use common::linalg::nalgebra::{
    bilinear_form_sparse, CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix,
    Vector as FeecVector,
};
use feg_core::HodgeBranchKind;
use feg_infer::{
    prior::{
        hodge::{
            build_coexact_1form_transform_with_coords, build_exact_1form_transform,
            build_exact_mass_coexact_1form_transform,
            build_hodge_1form_decomposed_prior_with_coords, Hodge1FormDecomposedPrior,
            Hodge1FormPriorConfig,
        },
        matern::one_form::{
            build_hodge_laplacian_1form, MaternMassInverse as Matern1FormMassInverse,
        },
        matern::two_form::MaternMassInverse as Matern2FormMassInverse,
    },
    sparse::{add_sparse, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec, scale_matrix},
};
use formoniq::operators::HodgeMassElmat;
use gmrf_core::{observation::apply_gaussian_observations, Gmrf};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    io::gmsh::gmsh2coord_complex,
    topology::complex::Complex,
};
use rand::{seq::SliceRandom, SeedableRng};
use std::{collections::HashMap, error::Error, fs, io, path::PathBuf, process::Command};

const RELATIVE_DENOM_EPS: f64 = 1e-10;
const SUBSPACE_EIGEN_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct CubeVoidsCoexactTransformConfig {
    pub output_dir: PathBuf,
    pub mesh_path: PathBuf,
    pub geo_path: PathBuf,
    pub force_mesh: bool,
    pub mesh_size: f64,
}

impl Default for CubeVoidsCoexactTransformConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output_dir = manifest_dir.join("../../out/cube_voids_coexact_transform_diagnostics");
        Self {
            mesh_path: output_dir.join("cube_voids_coexact_transform_diagnostics.msh"),
            geo_path: output_dir.join("cube_voids_coexact_transform_diagnostics.geo"),
            output_dir,
            force_mesh: true,
            mesh_size: 0.24,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CubeVoidsCoexactTransformDiagnostics {
    pub sparse_inverse: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub cell_count: usize,
    pub b0: usize,
    pub b1: usize,
    pub b2: usize,
    pub sparse_coexact_m1_operator_norm: f64,
    pub exact_mass_coexact_m1_operator_norm: f64,
    pub sparse_coexact_codifferential_leakage: f64,
    pub exact_mass_coexact_codifferential_leakage: f64,
    pub sparse_exact_branch_mass_orthogonality: f64,
    pub exact_mass_exact_branch_mass_orthogonality: f64,
    pub sparse_vs_exact_mass_transform_relative_m1_error: f64,
    pub sparse_coexact_rank: usize,
    pub exact_mass_coexact_rank: usize,
    pub principal_cosine_min: f64,
    pub principal_cosine_mean: f64,
    pub principal_angle_max_degrees: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeVoidsCoexactTruthSource {
    MatchedSparseCoexact,
    ExactMassCoexact,
}

impl CubeVoidsCoexactTruthSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MatchedSparseCoexact => "matched_sparse_coexact",
            Self::ExactMassCoexact => "exact_mass_coexact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeVoidsObservationDesign {
    SparseEdges,
    AllEdges,
}

impl CubeVoidsObservationDesign {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseEdges => "sparse_edges",
            Self::AllEdges => "all_edges",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CubeVoidsCoexactInferenceConfig {
    pub output_dir: PathBuf,
    pub mesh_path: PathBuf,
    pub geo_path: PathBuf,
    pub force_mesh: bool,
    pub mesh_size: f64,
    pub rng_seed: u64,
    pub kappa: f64,
    pub tau: f64,
    pub observation_noise_variance: f64,
    pub sparse_edge_fraction: f64,
    pub sparse_inverses: Vec<Matern1FormMassInverse>,
    pub truth_sources: Vec<CubeVoidsCoexactTruthSource>,
    pub observation_designs: Vec<CubeVoidsObservationDesign>,
}

impl Default for CubeVoidsCoexactInferenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output_dir = manifest_dir.join("../../out/cube_voids_coexact_inference");
        Self {
            mesh_path: output_dir.join("cube_voids_coexact_inference.msh"),
            geo_path: output_dir.join("cube_voids_coexact_inference.geo"),
            output_dir,
            force_mesh: true,
            mesh_size: 0.24,
            rng_seed: 20260515,
            kappa: 2.0,
            tau: 1.0,
            observation_noise_variance: 1e-8,
            sparse_edge_fraction: 0.25,
            sparse_inverses: vec![
                Matern1FormMassInverse::RowSumLumped,
                Matern1FormMassInverse::Nc1ProjectedSparseInverse,
                Matern1FormMassInverse::BarycentricDualSparseInverse,
            ],
            truth_sources: vec![
                CubeVoidsCoexactTruthSource::MatchedSparseCoexact,
                CubeVoidsCoexactTruthSource::ExactMassCoexact,
            ],
            observation_designs: vec![
                CubeVoidsObservationDesign::SparseEdges,
                CubeVoidsObservationDesign::AllEdges,
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct CubeVoidsCoexactInferenceMetrics {
    pub sparse_inverse: String,
    pub truth_source: CubeVoidsCoexactTruthSource,
    pub observation_design: CubeVoidsObservationDesign,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub cell_count: usize,
    pub b0: usize,
    pub b1: usize,
    pub b2: usize,
    pub observation_count: usize,
    pub truth_m1_norm: f64,
    pub posterior_m1_norm: f64,
    pub relative_m1_error: f64,
    pub mass_correlation: f64,
    pub observation_rms_residual: f64,
    pub truth_lumped_delta_leakage: f64,
    pub posterior_lumped_delta_leakage: f64,
    pub posterior_lumped_delta_over_truth_norm: f64,
    pub truth_weak_delta_leakage: f64,
    pub posterior_weak_delta_leakage: f64,
    pub posterior_weak_delta_over_truth_norm: f64,
}

pub fn run_cube_voids_coexact_transform_diagnostics(
    config: &CubeVoidsCoexactTransformConfig,
) -> Result<Vec<CubeVoidsCoexactTransformDiagnostics>, Box<dyn Error>> {
    validate_config(config)?;
    ensure_cube_voids_mesh(config)?;
    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    if topology.dim() != 3 {
        return Err(invalid_data(format!(
            "expected a 3D tetrahedral mesh, got topological dimension {}",
            topology.dim()
        ))
        .into());
    }

    [
        Matern1FormMassInverse::RowSumLumped,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        Matern1FormMassInverse::BarycentricDualSparseInverse,
    ]
    .into_iter()
    .map(|strategy| {
        build_cube_voids_coexact_transform_diagnostics_for_inverse(
            &topology, &coords, &metric, strategy,
        )
    })
    .collect()
}

pub fn run_cube_voids_coexact_inference(
    config: &CubeVoidsCoexactInferenceConfig,
) -> Result<Vec<CubeVoidsCoexactInferenceMetrics>, Box<dyn Error>> {
    validate_inference_config(config)?;
    let mesh_config = CubeVoidsCoexactTransformConfig {
        output_dir: config.output_dir.clone(),
        mesh_path: config.mesh_path.clone(),
        geo_path: config.geo_path.clone(),
        force_mesh: config.force_mesh,
        mesh_size: config.mesh_size,
    };
    ensure_cube_voids_mesh(&mesh_config)?;
    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    if topology.dim() != 3 {
        return Err(invalid_data(format!(
            "expected a 3D tetrahedral mesh, got topological dimension {}",
            topology.dim()
        ))
        .into());
    }

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let mass_0form = FeecCsr::from(&formoniq::assemble::assemble_galmat(
        &topology,
        &metric,
        HodgeMassElmat::new(topology.dim(), 0),
    ));
    let d0 = build_exact_1form_transform(&topology);
    let exact_mass_coexact_transform = if config
        .truth_sources
        .contains(&CubeVoidsCoexactTruthSource::ExactMassCoexact)
    {
        Some(
            build_exact_mass_coexact_1form_transform(&topology, &metric, &hodge.mass_u)
                .map_err(invalid_data)?,
        )
    } else {
        None
    };
    let topology_summary = cube_voids_betti_numbers(&topology);
    let mut rows = Vec::new();

    for &inverse in &config.sparse_inverses {
        let prior = build_cube_voids_coexact_prior(&topology, &coords, &metric, config, inverse)?;
        for &truth_source in &config.truth_sources {
            let truth = build_cube_voids_coexact_truth(
                &prior,
                exact_mass_coexact_transform.as_ref(),
                truth_source,
                config.rng_seed
                    + inverse_seed_offset(inverse)
                    + truth_source_seed_offset(truth_source),
                &hodge.mass_u,
            )?;
            for &observation_design in &config.observation_designs {
                let selected_edges =
                    select_observed_edges(topology.edges().len(), observation_design, config);
                let observation_matrix =
                    edge_selector_operator(topology.edges().len(), &selected_edges);
                let observations = &observation_matrix * &truth;
                let posterior_mean = condition_cube_voids_coexact_prior(
                    &prior,
                    &observation_matrix,
                    &observations,
                    config.observation_noise_variance,
                )?;
                let observation_residual = &(&observation_matrix * &posterior_mean) - &observations;
                rows.push(compute_cube_voids_coexact_inference_metrics(
                    inverse,
                    truth_source,
                    observation_design,
                    &topology,
                    topology_summary,
                    &metric,
                    &hodge.mass_u,
                    &mass_0form,
                    &d0,
                    selected_edges.len(),
                    &truth,
                    &posterior_mean,
                    &observation_residual,
                )?);
            }
        }
    }

    Ok(rows)
}

fn build_cube_voids_coexact_transform_diagnostics_for_inverse(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    one_form_mass_inverse: Matern1FormMassInverse,
) -> Result<CubeVoidsCoexactTransformDiagnostics, Box<dyn Error>> {
    let hodge = build_hodge_laplacian_1form(topology, metric);
    let mass_0form = formoniq::assemble::assemble_galmat(
        topology,
        metric,
        HodgeMassElmat::new(topology.dim(), 0),
    );
    let mass_0form = FeecCsr::from(&mass_0form);
    let exact_transform = build_exact_1form_transform(topology);
    let sparse_coexact_transform = build_coexact_1form_transform_with_coords(
        topology,
        coords,
        metric,
        &hodge.mass_u,
        one_form_mass_inverse,
    )
    .map_err(invalid_data)?;
    let exact_mass_coexact_transform =
        build_exact_mass_coexact_1form_transform(topology, metric, &hodge.mass_u)
            .map_err(invalid_data)?;

    let sparse_norm = operator_mass_norm(&sparse_coexact_transform, &hodge.mass_u);
    let exact_mass_norm = operator_mass_norm(&exact_mass_coexact_transform, &hodge.mass_u);
    let difference = add_sparse(
        &sparse_coexact_transform,
        &scale_matrix(&exact_mass_coexact_transform, -1.0),
    );
    let subspace = mass_principal_angle_summary(
        &sparse_coexact_transform,
        &exact_mass_coexact_transform,
        &hodge.mass_u,
    );

    let (b0, b1, b2) = cube_voids_betti_numbers(topology);

    Ok(CubeVoidsCoexactTransformDiagnostics {
        sparse_inverse: one_form_mass_inverse_name(one_form_mass_inverse).to_string(),
        vertex_count: topology.vertices().len(),
        edge_count: topology.edges().len(),
        face_count: topology.nsimplices(2),
        cell_count: topology.cells().len(),
        b0,
        b1,
        b2,
        sparse_coexact_m1_operator_norm: sparse_norm,
        exact_mass_coexact_m1_operator_norm: exact_mass_norm,
        sparse_coexact_codifferential_leakage: relative_or_nan(
            codifferential_operator_norm(
                &hodge.mass_u,
                &mass_0form,
                &exact_transform,
                &sparse_coexact_transform,
            )?,
            sparse_norm,
        ),
        exact_mass_coexact_codifferential_leakage: relative_or_nan(
            codifferential_operator_norm(
                &hodge.mass_u,
                &mass_0form,
                &exact_transform,
                &exact_mass_coexact_transform,
            )?,
            exact_mass_norm,
        ),
        sparse_exact_branch_mass_orthogonality: mass_orthogonality_ratio(
            &exact_transform,
            &sparse_coexact_transform,
            &hodge.mass_u,
        ),
        exact_mass_exact_branch_mass_orthogonality: mass_orthogonality_ratio(
            &exact_transform,
            &exact_mass_coexact_transform,
            &hodge.mass_u,
        ),
        sparse_vs_exact_mass_transform_relative_m1_error: relative_or_nan(
            operator_mass_norm(&difference, &hodge.mass_u),
            exact_mass_norm,
        ),
        sparse_coexact_rank: subspace.left_rank,
        exact_mass_coexact_rank: subspace.right_rank,
        principal_cosine_min: subspace.min_cosine,
        principal_cosine_mean: subspace.mean_cosine,
        principal_angle_max_degrees: subspace.max_angle_degrees,
    })
}

fn validate_config(config: &CubeVoidsCoexactTransformConfig) -> io::Result<()> {
    if !config.mesh_size.is_finite() || config.mesh_size <= 0.0 {
        return Err(invalid_input("mesh_size must be finite and positive"));
    }
    Ok(())
}

fn validate_inference_config(config: &CubeVoidsCoexactInferenceConfig) -> io::Result<()> {
    if !config.mesh_size.is_finite() || config.mesh_size <= 0.0 {
        return Err(invalid_input("mesh_size must be finite and positive"));
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive"));
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive"));
    }
    if !config.observation_noise_variance.is_finite() || config.observation_noise_variance <= 0.0 {
        return Err(invalid_input(
            "observation_noise_variance must be finite and positive",
        ));
    }
    if !config.sparse_edge_fraction.is_finite()
        || config.sparse_edge_fraction <= 0.0
        || config.sparse_edge_fraction > 1.0
    {
        return Err(invalid_input(
            "sparse_edge_fraction must be finite and in (0, 1]",
        ));
    }
    if config.sparse_inverses.is_empty() {
        return Err(invalid_input("at least one sparse inverse is required"));
    }
    if config.truth_sources.is_empty() {
        return Err(invalid_input("at least one truth source is required"));
    }
    if config.observation_designs.is_empty() {
        return Err(invalid_input("at least one observation design is required"));
    }
    Ok(())
}

fn build_cube_voids_coexact_prior(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    config: &CubeVoidsCoexactInferenceConfig,
    inverse: Matern1FormMassInverse,
) -> Result<Hodge1FormDecomposedPrior, Box<dyn Error>> {
    let two_form_mass_inverse = if inverse == Matern1FormMassInverse::BarycentricDualSparseInverse {
        Matern2FormMassInverse::BarycentricDualSparseInverse
    } else {
        Matern2FormMassInverse::ExactTopDegreeDiagonalOrProjectedNc2
    };
    build_hodge_1form_decomposed_prior_with_coords(
        topology,
        coords,
        metric,
        Hodge1FormPriorConfig {
            kappa: config.kappa,
            tau: config.tau,
            branches: vec![HodgeBranchKind::Coexact],
            harmonic_dim: 0,
            harmonic_basis_override: None,
            one_form_mass_inverse: inverse,
            two_form_mass_inverse,
            ..Hodge1FormPriorConfig::default()
        },
    )
    .map_err(invalid_data)
    .map_err(Into::into)
}

fn build_cube_voids_coexact_truth(
    prior: &Hodge1FormDecomposedPrior,
    exact_mass_coexact_transform: Option<&FeecCsr>,
    truth_source: CubeVoidsCoexactTruthSource,
    seed: u64,
    mass_1form: &FeecCsr,
) -> Result<FeecVector, Box<dyn Error>> {
    let latent =
        de_rham::sample_zero_mean_precision(&prior.precision, seed).map_err(invalid_data)?;
    let raw_truth = match truth_source {
        CubeVoidsCoexactTruthSource::MatchedSparseCoexact => &prior.latent_to_ambient * &latent,
        CubeVoidsCoexactTruthSource::ExactMassCoexact => {
            let transform = exact_mass_coexact_transform.ok_or_else(|| {
                invalid_data("exact-mass coexact truth requested without exact transform")
            })?;
            transform * &latent
        }
    };
    Ok(scale_to_mass_norm(&raw_truth, mass_1form, 1.0))
}

fn condition_cube_voids_coexact_prior(
    prior: &Hodge1FormDecomposedPrior,
    observation_matrix: &FeecCsr,
    observations: &FeecVector,
    noise_variance: f64,
) -> Result<FeecVector, Box<dyn Error>> {
    let latent_observation = observation_matrix * &prior.latent_to_ambient;
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&prior.precision),
        &feec_csr_to_gmrf(&latent_observation),
        &feec_vec_to_gmrf(observations),
        None,
        noise_variance,
    );
    let posterior = Gmrf::from_information_and_precision(information, posterior_precision)?;
    let latent_mean = gmrf_vec_to_feec(posterior.mean());
    Ok(&prior.latent_to_ambient * &latent_mean)
}

#[allow(clippy::too_many_arguments)]
fn compute_cube_voids_coexact_inference_metrics(
    inverse: Matern1FormMassInverse,
    truth_source: CubeVoidsCoexactTruthSource,
    observation_design: CubeVoidsObservationDesign,
    topology: &Complex,
    topology_summary: (usize, usize, usize),
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    mass_0form: &FeecCsr,
    d0: &FeecCsr,
    observation_count: usize,
    truth: &FeecVector,
    posterior_mean: &FeecVector,
    observation_residual: &FeecVector,
) -> Result<CubeVoidsCoexactInferenceMetrics, Box<dyn Error>> {
    let error = posterior_mean - truth;
    let truth_m1_norm = mass_norm(truth, mass_1form);
    let posterior_m1_norm = mass_norm(posterior_mean, mass_1form);
    let error_m1_norm = mass_norm(&error, mass_1form);
    let mass_correlation =
        if truth_m1_norm > RELATIVE_DENOM_EPS && posterior_m1_norm > RELATIVE_DENOM_EPS {
            bilinear_form_sparse(mass_1form, truth, posterior_mean)
                / (truth_m1_norm * posterior_m1_norm)
        } else {
            f64::NAN
        };

    let truth_lumped_delta =
        de_rham::codifferential(topology, metric, 1, truth).map_err(invalid_data)?;
    let posterior_lumped_delta =
        de_rham::codifferential(topology, metric, 1, posterior_mean).map_err(invalid_data)?;
    let truth_lumped_delta_norm = mass_norm(&truth_lumped_delta, mass_0form);
    let posterior_lumped_delta_norm = mass_norm(&posterior_lumped_delta, mass_0form);
    let truth_weak_delta_norm = weak_codifferential_vector_norm(mass_1form, mass_0form, d0, truth)?;
    let posterior_weak_delta_norm =
        weak_codifferential_vector_norm(mass_1form, mass_0form, d0, posterior_mean)?;

    Ok(CubeVoidsCoexactInferenceMetrics {
        sparse_inverse: one_form_mass_inverse_name(inverse).to_string(),
        truth_source,
        observation_design,
        vertex_count: topology.vertices().len(),
        edge_count: topology.edges().len(),
        face_count: topology.nsimplices(2),
        cell_count: topology.cells().len(),
        b0: topology_summary.0,
        b1: topology_summary.1,
        b2: topology_summary.2,
        observation_count,
        truth_m1_norm,
        posterior_m1_norm,
        relative_m1_error: relative_or_nan(error_m1_norm, truth_m1_norm),
        mass_correlation,
        observation_rms_residual: rms_norm(observation_residual),
        truth_lumped_delta_leakage: relative_or_nan(truth_lumped_delta_norm, truth_m1_norm),
        posterior_lumped_delta_leakage: relative_or_nan(
            posterior_lumped_delta_norm,
            posterior_m1_norm,
        ),
        posterior_lumped_delta_over_truth_norm: relative_or_nan(
            posterior_lumped_delta_norm,
            truth_m1_norm,
        ),
        truth_weak_delta_leakage: relative_or_nan(truth_weak_delta_norm, truth_m1_norm),
        posterior_weak_delta_leakage: relative_or_nan(posterior_weak_delta_norm, posterior_m1_norm),
        posterior_weak_delta_over_truth_norm: relative_or_nan(
            posterior_weak_delta_norm,
            truth_m1_norm,
        ),
    })
}

fn select_observed_edges(
    edge_count: usize,
    observation_design: CubeVoidsObservationDesign,
    config: &CubeVoidsCoexactInferenceConfig,
) -> Vec<usize> {
    match observation_design {
        CubeVoidsObservationDesign::AllEdges => (0..edge_count).collect(),
        CubeVoidsObservationDesign::SparseEdges => {
            let count = ((edge_count as f64) * config.sparse_edge_fraction)
                .round()
                .clamp(1.0, edge_count as f64) as usize;
            let mut edges = (0..edge_count).collect::<Vec<_>>();
            let mut rng = rand::rngs::StdRng::seed_from_u64(config.rng_seed + 811);
            edges.shuffle(&mut rng);
            edges.truncate(count);
            edges.sort_unstable();
            edges
        }
    }
}

fn edge_selector_operator(edge_count: usize, selected_edges: &[usize]) -> FeecCsr {
    let mut coo = FeecCoo::new(selected_edges.len(), edge_count);
    for (row, edge) in selected_edges.iter().copied().enumerate() {
        coo.push(row, edge, 1.0);
    }
    FeecCsr::from(&coo)
}

fn weak_codifferential_vector_norm(
    mass_1form: &FeecCsr,
    mass_0form: &FeecCsr,
    d0: &FeecCsr,
    field: &FeecVector,
) -> Result<f64, Box<dyn Error>> {
    let weighted = mass_1form * field;
    let weak = d0.transpose() * &weighted;
    let factor = feec_csr_to_gmrf(mass_0form)
        .cholesky_sqrt_lower()
        .map_err(|err| invalid_data(format!("failed to factor 0-form mass matrix: {err}")))?;
    let solution = factor
        .solve(&feec_vec_to_gmrf(&weak))
        .map_err(|err| invalid_data(format!("failed to apply inverse 0-form mass: {err}")))?;
    let solution = gmrf_vec_to_feec(&solution);
    Ok(weak.dot(&solution).max(0.0).sqrt())
}

fn scale_to_mass_norm(field: &FeecVector, mass: &FeecCsr, target_norm: f64) -> FeecVector {
    let norm = mass_norm(field, mass);
    if norm <= RELATIVE_DENOM_EPS {
        field.clone()
    } else {
        field * (target_norm / norm)
    }
}

fn mass_norm(field: &FeecVector, mass: &FeecCsr) -> f64 {
    bilinear_form_sparse(mass, field, field).max(0.0).sqrt()
}

fn rms_norm(values: &FeecVector) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.dot(values) / values.len() as f64).sqrt()
    }
}

fn inverse_seed_offset(inverse: Matern1FormMassInverse) -> u64 {
    match inverse {
        Matern1FormMassInverse::RowSumLumped => 1000,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse => 2000,
        Matern1FormMassInverse::BarycentricDualSparseInverse => 3000,
    }
}

fn truth_source_seed_offset(truth_source: CubeVoidsCoexactTruthSource) -> u64 {
    match truth_source {
        CubeVoidsCoexactTruthSource::MatchedSparseCoexact => 100,
        CubeVoidsCoexactTruthSource::ExactMassCoexact => 200,
    }
}

fn ensure_cube_voids_mesh(config: &CubeVoidsCoexactTransformConfig) -> Result<(), Box<dyn Error>> {
    if !config.force_mesh && config.mesh_path.is_file() {
        return Ok(());
    }
    if let Some(parent) = config.geo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = config.mesh_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.geo_path, cube_voids_geo(config.mesh_size))?;
    let status = Command::new("gmsh")
        .arg("-3")
        .arg(&config.geo_path)
        .arg("-format")
        .arg("msh4")
        .arg("-o")
        .arg(&config.mesh_path)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "gmsh failed while generating `{}`",
            config.mesh_path.display()
        ))
        .into());
    }
    Ok(())
}

fn cube_voids_geo(mesh_size: f64) -> String {
    format!(
        r#"SetFactory("OpenCASCADE");
lc = {mesh_size:.12};
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumCirclePoints = 10;
Box(1) = {{0, 0, 0, 1, 1, 1}};
Sphere(2) = {{0.34, 0.44, 0.52, 0.16}};
Sphere(3) = {{0.68, 0.60, 0.48, 0.14}};
out[] = BooleanDifference{{ Volume{{1}}; Delete; }}{{ Volume{{2, 3}}; Delete; }};
Physical Volume("cube_with_voids") = {{out[0]}};
Mesh 3;
"#
    )
}

fn one_form_mass_inverse_name(strategy: Matern1FormMassInverse) -> &'static str {
    match strategy {
        Matern1FormMassInverse::RowSumLumped => "row_sum_lumped",
        Matern1FormMassInverse::Nc1ProjectedSparseInverse => "nc1_projected_sparse_inverse",
        Matern1FormMassInverse::BarycentricDualSparseInverse => "barycentric_dual_sparse_inverse",
    }
}

fn operator_mass_norm(transform: &FeecCsr, mass: &FeecCsr) -> f64 {
    let weighted = mass * transform;
    let gram = transform.transpose() * &weighted;
    trace_sparse(&gram).max(0.0).sqrt()
}

fn codifferential_operator_norm(
    mass_1form: &FeecCsr,
    mass_0form: &FeecCsr,
    exact_transform: &FeecCsr,
    coexact_transform: &FeecCsr,
) -> Result<f64, Box<dyn Error>> {
    let weighted = mass_1form * coexact_transform;
    let weak_codifferential = exact_transform.transpose() * &weighted;
    weak_dual_operator_norm(&weak_codifferential, mass_0form)
}

fn weak_dual_operator_norm(weak_operator: &FeecCsr, mass: &FeecCsr) -> Result<f64, Box<dyn Error>> {
    let factor = feec_csr_to_gmrf(mass)
        .cholesky_sqrt_lower()
        .map_err(|err| invalid_data(format!("failed to factor mass matrix: {err}")))?;
    let dense = FeecMatrix::from(weak_operator);
    let mut norm_squared = 0.0;
    for col in 0..dense.ncols() {
        let rhs = FeecVector::from_iterator(
            dense.nrows(),
            (0..dense.nrows()).map(|row| dense[(row, col)]),
        );
        let solution = factor
            .solve(&feec_vec_to_gmrf(&rhs))
            .map_err(|err| invalid_data(format!("failed to apply inverse mass: {err}")))?;
        let solution = gmrf_vec_to_feec(&solution);
        norm_squared += rhs.dot(&solution);
    }
    Ok(norm_squared.max(0.0).sqrt())
}

fn mass_orthogonality_ratio(left: &FeecCsr, right: &FeecCsr, mass: &FeecCsr) -> f64 {
    let weighted_right = mass * right;
    let cross = left.transpose() * &weighted_right;
    let numerator = frobenius_norm_sparse(&cross);
    let denominator = operator_mass_norm(left, mass) * operator_mass_norm(right, mass);
    relative_or_nan(numerator, denominator)
}

#[derive(Debug, Clone)]
struct OperatorSubspaceSummary {
    left_rank: usize,
    right_rank: usize,
    min_cosine: f64,
    mean_cosine: f64,
    max_angle_degrees: f64,
}

fn mass_principal_angle_summary(
    left: &FeecCsr,
    right: &FeecCsr,
    mass: &FeecCsr,
) -> OperatorSubspaceSummary {
    let left_gram = mass_cross_gram_dense(left, left, mass);
    let right_gram = mass_cross_gram_dense(right, right, mass);
    let cross_gram = mass_cross_gram_dense(left, right, mass);
    let (left_inverse_sqrt, left_rank) = inverse_sqrt_coefficients(left_gram);
    let (right_inverse_sqrt, right_rank) = inverse_sqrt_coefficients(right_gram);
    if left_rank == 0 || right_rank == 0 {
        return OperatorSubspaceSummary {
            left_rank,
            right_rank,
            min_cosine: f64::NAN,
            mean_cosine: f64::NAN,
            max_angle_degrees: f64::NAN,
        };
    }
    let cosine_matrix = left_inverse_sqrt.transpose() * cross_gram * right_inverse_sqrt;
    if cosine_matrix.nrows() == 0 || cosine_matrix.ncols() == 0 {
        return OperatorSubspaceSummary {
            left_rank,
            right_rank,
            min_cosine: f64::NAN,
            mean_cosine: f64::NAN,
            max_angle_degrees: f64::NAN,
        };
    }
    let svd = cosine_matrix.svd(false, false);
    let common_rank = left_rank.min(right_rank);
    let mut cosines = svd
        .singular_values
        .iter()
        .take(common_rank)
        .map(|value| value.abs().clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    if left_rank != right_rank {
        cosines.extend(std::iter::repeat(0.0).take(left_rank.max(right_rank) - common_rank));
    }
    let min_cosine = cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let mean_cosine = mean(cosines.iter().copied());
    let max_angle_degrees = min_cosine.acos() * 180.0 / std::f64::consts::PI;
    OperatorSubspaceSummary {
        left_rank,
        right_rank,
        min_cosine,
        mean_cosine,
        max_angle_degrees,
    }
}

fn mass_cross_gram_dense(left: &FeecCsr, right: &FeecCsr, mass: &FeecCsr) -> FeecMatrix {
    let weighted_right = mass * right;
    let gram = left.transpose() * &weighted_right;
    FeecMatrix::from(&gram)
}

fn inverse_sqrt_coefficients(gram: FeecMatrix) -> (FeecMatrix, usize) {
    let symmetric = symmetrize_dense(&gram);
    let eigen = symmetric.symmetric_eigen();
    let max_eigenvalue = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let tolerance = (SUBSPACE_EIGEN_TOLERANCE * max_eigenvalue).max(1e-14);
    let kept = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value > tolerance).then_some((index, *value)))
        .collect::<Vec<_>>();
    let mut coefficients = FeecMatrix::zeros(gram.ncols(), kept.len());
    for (out_col, (eigen_col, eigenvalue)) in kept.iter().copied().enumerate() {
        let scale = eigenvalue.sqrt().recip();
        for row in 0..gram.ncols() {
            coefficients[(row, out_col)] = eigen.eigenvectors[(row, eigen_col)] * scale;
        }
    }
    let rank = kept.len();
    (coefficients, rank)
}

fn symmetrize_dense(matrix: &FeecMatrix) -> FeecMatrix {
    let mut symmetric = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            symmetric[(row, col)] = 0.5 * (matrix[(row, col)] + matrix[(col, row)]);
        }
    }
    symmetric
}

fn cube_voids_betti_numbers(topology: &Complex) -> (usize, usize, usize) {
    let b0 = count_vertex_components(topology);
    let boundary_components = count_boundary_surface_components(topology);
    let b2 = boundary_components.saturating_sub(b0);
    let euler_characteristic = topology.nsimplices(0) as isize - topology.nsimplices(1) as isize
        + topology.nsimplices(2) as isize
        - topology.nsimplices(3) as isize;
    let b1 = (b0 as isize + b2 as isize - euler_characteristic).max(0) as usize;
    (b0, b1, b2)
}

fn count_vertex_components(topology: &Complex) -> usize {
    let mut disjoint = DisjointSet::new(topology.nsimplices(0));
    for edge in topology.edges().handle_iter() {
        if edge.vertices.len() == 2 {
            disjoint.union(edge.vertices[0], edge.vertices[1]);
        }
    }
    disjoint.component_count()
}

fn count_boundary_surface_components(topology: &Complex) -> usize {
    let boundary_faces = topology.boundary_facets();
    if boundary_faces.is_empty() {
        return 0;
    }
    let mut disjoint = DisjointSet::new(boundary_faces.len());
    let mut edge_to_face = HashMap::<(usize, usize), usize>::new();
    for (face_index, face) in boundary_faces.iter().copied().enumerate() {
        let vertices = &face.handle(topology).vertices;
        for i in 0..vertices.len() {
            for j in i + 1..vertices.len() {
                let a = vertices[i].min(vertices[j]);
                let b = vertices[i].max(vertices[j]);
                if let Some(previous_face) = edge_to_face.insert((a, b), face_index) {
                    disjoint.union(previous_face, face_index);
                }
            }
        }
    }
    disjoint.component_count()
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        let parent = self.parent[index];
        if parent != index {
            let root = self.find(parent);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
    }

    fn component_count(mut self) -> usize {
        let mut roots = Vec::<usize>::new();
        for index in 0..self.parent.len() {
            let root = self.find(index);
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        roots.len()
    }
}

fn trace_sparse(matrix: &FeecCsr) -> f64 {
    matrix
        .triplet_iter()
        .filter_map(|(row, col, value)| (row == col).then_some(*value))
        .sum()
}

fn frobenius_norm_sparse(matrix: &FeecCsr) -> f64 {
    matrix
        .triplet_iter()
        .map(|(_, _, value)| *value * *value)
        .sum::<f64>()
        .sqrt()
}

fn relative_or_nan(numerator: f64, denominator: f64) -> f64 {
    if denominator.is_finite() && denominator.abs() > RELATIVE_DENOM_EPS {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gmsh_available() -> bool {
        Command::new("gmsh").arg("-version").output().is_ok()
    }

    #[test]
    fn cube_voids_coexact_transform_diagnostics_run() {
        if !gmsh_available() {
            eprintln!("skipping cube voids coexact transform test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/cube_voids_coexact_transform_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = CubeVoidsCoexactTransformConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("cube_voids_coexact_transform_test.msh"),
            geo_path: out_dir.join("cube_voids_coexact_transform_test.geo"),
            force_mesh: true,
            mesh_size: 0.34,
        };
        let rows = run_cube_voids_coexact_transform_diagnostics(&config)
            .expect("cube voids coexact transform diagnostics should run");
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| {
            row.vertex_count > 0
                && row.edge_count > 0
                && row.face_count > 0
                && row.cell_count > 0
                && row.b0 == 1
                && row.b1 == 0
                && row.b2 == 2
                && row.sparse_coexact_codifferential_leakage.is_finite()
                && row.sparse_exact_branch_mass_orthogonality.is_finite()
                && row
                    .sparse_vs_exact_mass_transform_relative_m1_error
                    .is_finite()
                && row.exact_mass_coexact_codifferential_leakage < 1e-8
                && row.exact_mass_exact_branch_mass_orthogonality < 1e-8
                && row.sparse_coexact_rank > 0
                && row.sparse_coexact_rank == row.exact_mass_coexact_rank
                && (0.0..=1.0).contains(&row.principal_cosine_min)
                && (0.0..=1.0).contains(&row.principal_cosine_mean)
                && row.principal_angle_max_degrees.is_finite()
        }));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn cube_voids_coexact_inference_matched_dense_recovers_truth() {
        if !gmsh_available() {
            eprintln!("skipping cube voids coexact inference test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/cube_voids_coexact_inference_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = CubeVoidsCoexactInferenceConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("cube_voids_coexact_inference_test.msh"),
            geo_path: out_dir.join("cube_voids_coexact_inference_test.geo"),
            force_mesh: true,
            mesh_size: 0.42,
            sparse_inverses: vec![Matern1FormMassInverse::Nc1ProjectedSparseInverse],
            truth_sources: vec![CubeVoidsCoexactTruthSource::MatchedSparseCoexact],
            observation_designs: vec![CubeVoidsObservationDesign::AllEdges],
            ..CubeVoidsCoexactInferenceConfig::default()
        };
        let rows = run_cube_voids_coexact_inference(&config)
            .expect("cube voids coexact inference should run");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.b0, 1);
        assert_eq!(row.b1, 0);
        assert_eq!(row.b2, 2);
        assert!(row.relative_m1_error.is_finite());
        assert!(
            row.relative_m1_error < 1e-3,
            "matched dense coexact inference should recover truth, got E1={:.3e}",
            row.relative_m1_error
        );
        assert!(row.posterior_lumped_delta_leakage.is_finite());
        assert!(row.posterior_weak_delta_leakage.is_finite());
        let _ = fs::remove_dir_all(&out_dir);
    }
}
