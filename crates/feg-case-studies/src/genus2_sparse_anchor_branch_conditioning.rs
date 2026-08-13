//! Mixed-truth conditioning validation for sparse-anchor Hodge branches on a genus-2 surface.
//!
//! The validation conditions exact-only, coexact-only, harmonic-only, and joint
//! sparse-anchor priors on the same observations from a mixed exact/coexact/harmonic
//! field. The purpose is to check branch structure under misspecification and
//! topological harmonic recovery, not to make branch-only models fit mixed data perfectly.

use crate::{
    genus2_1form_hodge_conditioning::{
        build_cycle_observation_matrix, default_genus2_torus_mesh_path, validate_genus2_topology,
        Genus2TopologySummary,
    },
    genus2_topological_inverse::select_local_observation_edges,
};
use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use ddf::ManifoldComplexExt;
use feg_core::HodgeBranchKind;
use feg_infer::{
    prior::{
        matern::MaternAlpha,
        sparse_anchor_hodge::{
            build_sparse_anchor_hodge_1form_prior_with_coords, SparseAnchorBranchConfig,
            SparseAnchorGauge, SparseAnchorHodge1FormBranch, SparseAnchorHodge1FormPrior,
            SparseAnchorHodge1FormPriorConfig,
        },
    },
    sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec},
};
use gmrf_core::{observation::apply_gaussian_observations, Gmrf};
use manifold::{
    geometry::{
        coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
        metric::mesh::MeshLengths,
    },
    io::gmsh::gmsh2coord_complex,
    topology::complex::Complex,
};
use std::{fs, path::PathBuf};

const EPS: f64 = 1e-12;
const HARMONIC_COEFFICIENTS: [f64; 4] = [0.80, -0.55, 0.35, -0.25];
pub const EXACT_CLOSURE_TOLERANCE: f64 = 1e-10;
pub const COEXACT_COCLOSED_TOLERANCE: f64 = 2e-1;
pub const HARMONIC_STRUCTURE_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone)]
pub struct Genus2SparseAnchorBranchConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub alpha: MaternAlpha,
    pub harmonic_dim: usize,
    pub harmonic_precision: f64,
    pub local_observation_count: usize,
    pub local_noise_variance: f64,
    pub cycle_noise_variance: f64,
}

impl Default for Genus2SparseAnchorBranchConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_genus2_torus_mesh_path(),
            kappa: 1.0,
            tau: 1.0,
            alpha: MaternAlpha::Two,
            harmonic_dim: 4,
            harmonic_precision: 1.0,
            local_observation_count: 16,
            local_noise_variance: 1e-6,
            cycle_noise_variance: 1e-8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BranchPosteriorDiagnostics {
    pub label: String,
    pub observation_residual_relative: f64,
    pub mass_norm: f64,
    pub closure_residual_relative: f64,
    pub coclosed_residual_relative: f64,
    pub cycle_period_error_relative: f64,
    pub harmonic_latent_dimension: Option<usize>,
    pub harmonic_coefficient_error_relative: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Genus2SparseAnchorBranchConditioningReport {
    pub topology_summary: Genus2TopologySummary,
    pub observation_count: usize,
    pub local_observation_count: usize,
    pub cycle_observation_count: usize,
    pub local_noise_variance: f64,
    pub cycle_noise_variance: f64,
    pub harmonic_latent_dimension: usize,
    pub cycle_harmonic_pairing_rank: usize,
    pub truth_exact_mass_norm: f64,
    pub truth_coexact_mass_norm: f64,
    pub truth_harmonic_mass_norm: f64,
    pub truth_mixed_mass_norm: f64,
    pub truth_harmonic_coefficients: Vec<f64>,
    pub exact_only: BranchPosteriorDiagnostics,
    pub coexact_only: BranchPosteriorDiagnostics,
    pub harmonic_only: BranchPosteriorDiagnostics,
    pub joint_total: BranchPosteriorDiagnostics,
    pub joint_exact: BranchPosteriorDiagnostics,
    pub joint_coexact: BranchPosteriorDiagnostics,
    pub joint_harmonic: BranchPosteriorDiagnostics,
}

impl Genus2SparseAnchorBranchConditioningReport {
    pub fn diagnostics(&self) -> [&BranchPosteriorDiagnostics; 7] {
        [
            &self.exact_only,
            &self.coexact_only,
            &self.harmonic_only,
            &self.joint_total,
            &self.joint_exact,
            &self.joint_coexact,
            &self.joint_harmonic,
        ]
    }
}

struct MeshData {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
    topology_summary: Genus2TopologySummary,
}

struct MixedTruth {
    exact: FeecVector,
    coexact: FeecVector,
    harmonic: FeecVector,
    mixed: FeecVector,
    harmonic_coefficients: FeecVector,
}

struct ObservationRow {
    entries: Vec<(usize, f64)>,
    noise_variance: f64,
}

struct ObservationSystem {
    unscaled_selector: FeecCsr,
    scaled_selector: FeecCsr,
    unscaled_observations: FeecVector,
    scaled_observations: FeecVector,
    local_count: usize,
    cycle_count: usize,
}

struct ConditioningOutput {
    latent_mean: FeecVector,
    ambient_mean: FeecVector,
}

pub fn compute_genus2_sparse_anchor_branch_conditioning(
    config: Genus2SparseAnchorBranchConditioningConfig,
) -> Result<Genus2SparseAnchorBranchConditioningReport, String> {
    validate_config(&config)?;

    let mesh = load_mesh(&config)?;
    let exact_prior = build_prior(
        &mesh.topology,
        &mesh.coords,
        &mesh.metric,
        &config,
        [HodgeBranchKind::Exact],
    )?;
    let coexact_prior = build_prior(
        &mesh.topology,
        &mesh.coords,
        &mesh.metric,
        &config,
        [HodgeBranchKind::Coexact],
    )?;
    let harmonic_prior = build_prior(
        &mesh.topology,
        &mesh.coords,
        &mesh.metric,
        &config,
        [HodgeBranchKind::Harmonic],
    )?;
    let joint_prior = build_prior(
        &mesh.topology,
        &mesh.coords,
        &mesh.metric,
        &config,
        [
            HodgeBranchKind::Exact,
            HodgeBranchKind::Coexact,
            HodgeBranchKind::Harmonic,
        ],
    )?;

    let truth = build_mixed_truth(&mesh, &exact_prior, &coexact_prior, &joint_prior)?;
    let (cycle_selector, _) = build_cycle_observation_matrix(&mesh.topology, &mesh.coords)
        .map_err(|err| err.to_string())?;
    let observations = build_observation_system(&mesh, &cycle_selector, &truth, &config)?;

    let exact_conditioned = condition_prior(
        &exact_prior,
        &observations.scaled_selector,
        &observations.scaled_observations,
    )?;
    let coexact_conditioned = condition_prior(
        &coexact_prior,
        &observations.scaled_selector,
        &observations.scaled_observations,
    )?;
    let harmonic_conditioned = condition_prior(
        &harmonic_prior,
        &observations.scaled_selector,
        &observations.scaled_observations,
    )?;
    let joint_conditioned = condition_prior(
        &joint_prior,
        &observations.scaled_selector,
        &observations.scaled_observations,
    )?;

    let joint_exact_mean = branch_ambient_mean(
        &joint_conditioned.latent_mean,
        joint_prior
            .branch(HodgeBranchKind::Exact)
            .ok_or_else(|| "joint prior missing exact branch".to_string())?,
    );
    let joint_coexact_mean = branch_ambient_mean(
        &joint_conditioned.latent_mean,
        joint_prior
            .branch(HodgeBranchKind::Coexact)
            .ok_or_else(|| "joint prior missing coexact branch".to_string())?,
    );
    let joint_harmonic_mean = branch_ambient_mean(
        &joint_conditioned.latent_mean,
        joint_prior
            .branch(HodgeBranchKind::Harmonic)
            .ok_or_else(|| "joint prior missing harmonic branch".to_string())?,
    );

    let d1 = FeecCsr::from(&mesh.topology.exterior_derivative_operator(1));
    let d0 = FeecCsr::from(&mesh.topology.exterior_derivative_operator(0));
    let mass_1 = &joint_prior.mass_1form;
    let truth_periods = &cycle_selector * &truth.mixed;
    let cycle_harmonic_pairing = &cycle_selector * &joint_prior.harmonic_basis;
    let cycle_harmonic_pairing_rank = cycle_harmonic_pairing.rank(1e-8);

    let harmonic_only_diagnostics = add_harmonic_diagnostics(
        diagnostics(
            "harmonic_only",
            &harmonic_conditioned.ambient_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        &harmonic_conditioned.ambient_mean,
        &truth.harmonic,
        &harmonic_prior,
    );
    let joint_harmonic_diagnostics = add_harmonic_diagnostics(
        diagnostics(
            "joint_harmonic",
            &joint_harmonic_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        &joint_harmonic_mean,
        &truth.harmonic,
        &joint_prior,
    );

    Ok(Genus2SparseAnchorBranchConditioningReport {
        topology_summary: mesh.topology_summary,
        observation_count: observations.unscaled_selector.nrows(),
        local_observation_count: observations.local_count,
        cycle_observation_count: observations.cycle_count,
        local_noise_variance: config.local_noise_variance,
        cycle_noise_variance: config.cycle_noise_variance,
        harmonic_latent_dimension: joint_prior
            .branch(HodgeBranchKind::Harmonic)
            .map(|branch| branch.latent_dimension)
            .unwrap_or(0),
        cycle_harmonic_pairing_rank,
        truth_exact_mass_norm: mass_norm(&truth.exact, mass_1),
        truth_coexact_mass_norm: mass_norm(&truth.coexact, mass_1),
        truth_harmonic_mass_norm: mass_norm(&truth.harmonic, mass_1),
        truth_mixed_mass_norm: mass_norm(&truth.mixed, mass_1),
        truth_harmonic_coefficients: truth.harmonic_coefficients.iter().copied().collect(),
        exact_only: diagnostics(
            "exact_only",
            &exact_conditioned.ambient_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        coexact_only: diagnostics(
            "coexact_only",
            &coexact_conditioned.ambient_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        harmonic_only: harmonic_only_diagnostics,
        joint_total: diagnostics(
            "joint_total",
            &joint_conditioned.ambient_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        joint_exact: diagnostics(
            "joint_exact",
            &joint_exact_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        joint_coexact: diagnostics(
            "joint_coexact",
            &joint_coexact_mean,
            &observations.unscaled_selector,
            &observations.unscaled_observations,
            &cycle_selector,
            &truth_periods,
            mass_1,
            &d0,
            &d1,
        ),
        joint_harmonic: joint_harmonic_diagnostics,
    })
}

fn load_mesh(config: &Genus2SparseAnchorBranchConditioningConfig) -> Result<MeshData, String> {
    let mesh_bytes = fs::read(&config.mesh_path)
        .map_err(|err| format!("failed to read {}: {err}", config.mesh_path.display()))?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = validate_genus2_topology(&topology).map_err(|err| err.to_string())?;
    Ok(MeshData {
        topology,
        coords,
        metric,
        topology_summary,
    })
}

fn build_prior(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    config: &Genus2SparseAnchorBranchConditioningConfig,
    branches: impl IntoIterator<Item = HodgeBranchKind>,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    let branch_config = SparseAnchorBranchConfig {
        kappa: config.kappa,
        tau: config.tau,
        alpha: config.alpha,
    };
    build_sparse_anchor_hodge_1form_prior_with_coords(
        topology,
        coords,
        metric,
        SparseAnchorHodge1FormPriorConfig {
            branches: branches.into_iter().collect(),
            exact: branch_config,
            coexact: branch_config,
            harmonic_precision: config.harmonic_precision,
            harmonic_dim: Some(config.harmonic_dim),
            ..SparseAnchorHodge1FormPriorConfig::default()
        },
    )
}

fn build_mixed_truth(
    mesh: &MeshData,
    exact_prior: &SparseAnchorHodge1FormPrior,
    coexact_prior: &SparseAnchorHodge1FormPrior,
    joint_prior: &SparseAnchorHodge1FormPrior,
) -> Result<MixedTruth, String> {
    let exact_branch = exact_prior
        .branch(HodgeBranchKind::Exact)
        .ok_or_else(|| "exact prior missing exact branch".to_string())?;
    let exact_potential = anchored_potential_values(
        &vertex_potential(&mesh.coords),
        exact_branch
            .gauge
            .as_ref()
            .ok_or_else(|| "exact branch missing sparse anchor gauge".to_string())?,
    );
    let exact = mass_normalized(
        &(&exact_branch.transform * &exact_potential),
        &exact_prior.mass_1form,
    )?;

    let coexact_branch = coexact_prior
        .branch(HodgeBranchKind::Coexact)
        .ok_or_else(|| "coexact prior missing coexact branch".to_string())?;
    let coexact_potential = anchored_potential_values(
        &face_potential(&mesh.topology, &mesh.coords),
        coexact_branch
            .gauge
            .as_ref()
            .ok_or_else(|| "coexact branch missing sparse anchor gauge".to_string())?,
    );
    let coexact = mass_normalized(
        &(&coexact_branch.transform * &coexact_potential),
        &coexact_prior.mass_1form,
    )?;

    if joint_prior.harmonic_basis.ncols() != HARMONIC_COEFFICIENTS.len() {
        return Err(format!(
            "expected {} harmonic basis vectors, got {}",
            HARMONIC_COEFFICIENTS.len(),
            joint_prior.harmonic_basis.ncols()
        ));
    }
    let harmonic_coefficients = FeecVector::from_column_slice(&HARMONIC_COEFFICIENTS);
    let harmonic = mass_normalized(
        &(&joint_prior.harmonic_basis * &harmonic_coefficients),
        &joint_prior.mass_1form,
    )?;
    let harmonic_coefficients = harmonic_coefficients_in_basis(
        &harmonic,
        &joint_prior.harmonic_basis,
        &joint_prior.mass_1form,
    );
    let mixed = &(&exact + &coexact) + &harmonic;

    Ok(MixedTruth {
        exact,
        coexact,
        harmonic,
        mixed,
        harmonic_coefficients,
    })
}

fn vertex_potential(coords: &MeshCoords) -> FeecVector {
    FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|vertex| {
            let coord = coords.coord(vertex);
            coord[0] + 0.35 * coord[1] - 0.20 * coord[2] + 0.10 * (2.0 * coord[0]).sin()
        }),
    )
}

fn face_potential(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let mut values = FeecVector::zeros(topology.cells().len());
    for face in topology.cells().handle_iter() {
        let barycenter = face.coord_simplex(coords).barycenter();
        values[face.kidx()] = barycenter[1] - 0.25 * barycenter[0] * barycenter[2]
            + 0.12 * (2.0 * barycenter[2]).cos();
    }
    values
}

fn anchored_potential_values(full: &FeecVector, gauge: &SparseAnchorGauge) -> FeecVector {
    let shift = if gauge.anchors.len() == 1 {
        full[gauge.anchors[0]]
    } else {
        0.0
    };
    FeecVector::from_iterator(
        gauge.kept_dofs.len(),
        gauge.kept_dofs.iter().map(|&dof| full[dof] - shift),
    )
}

fn build_observation_system(
    mesh: &MeshData,
    cycle_selector: &FeecCsr,
    truth: &MixedTruth,
    config: &Genus2SparseAnchorBranchConditioningConfig,
) -> Result<ObservationSystem, String> {
    let local_edges = select_local_observation_edges(
        &mesh.topology,
        &mesh.coords,
        cycle_selector,
        config.local_observation_count,
    );
    let mut rows = local_edges
        .iter()
        .copied()
        .map(|edge_index| ObservationRow {
            entries: vec![(edge_index, 1.0)],
            noise_variance: config.local_noise_variance,
        })
        .collect::<Vec<_>>();
    let cycle_rows = sparse_rows(cycle_selector)
        .into_iter()
        .map(|entries| ObservationRow {
            entries,
            noise_variance: config.cycle_noise_variance,
        })
        .collect::<Vec<_>>();
    let cycle_count = cycle_rows.len();
    rows.extend(cycle_rows);

    let unscaled_selector = observation_selector(&rows, mesh.topology.edges().len(), false)?;
    let scaled_selector = observation_selector(&rows, mesh.topology.edges().len(), true)?;
    let clean_observations = &unscaled_selector * &truth.mixed;
    let unscaled_observations = observations_with_noise(&clean_observations, &rows);
    let scaled_observations = scaled_observations(&unscaled_observations, &rows);

    Ok(ObservationSystem {
        unscaled_selector,
        scaled_selector,
        unscaled_observations,
        scaled_observations,
        local_count: local_edges.len(),
        cycle_count,
    })
}

fn sparse_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if value.abs() > EPS {
            rows[row].push((col, *value));
        }
    }
    rows
}

fn observation_selector(
    rows: &[ObservationRow],
    edge_count: usize,
    scale_by_noise: bool,
) -> Result<FeecCsr, String> {
    let mut coo = FeecCoo::new(rows.len(), edge_count);
    for (row_index, row) in rows.iter().enumerate() {
        let scale = if scale_by_noise {
            row.noise_variance.sqrt()
        } else {
            1.0
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(
                "observation noise standard deviation must be finite and positive".to_string(),
            );
        }
        for (edge_index, value) in &row.entries {
            coo.push(row_index, *edge_index, *value / scale);
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn observations_with_noise(clean: &FeecVector, rows: &[ObservationRow]) -> FeecVector {
    FeecVector::from_iterator(
        clean.len(),
        clean.iter().enumerate().map(|(index, value)| {
            let pattern = ((index * 37 + 11) % 17) as f64 / 8.0 - 1.0;
            value + 0.25 * rows[index].noise_variance.sqrt() * pattern
        }),
    )
}

fn scaled_observations(unscaled: &FeecVector, rows: &[ObservationRow]) -> FeecVector {
    FeecVector::from_iterator(
        unscaled.len(),
        unscaled
            .iter()
            .enumerate()
            .map(|(index, value)| value / rows[index].noise_variance.sqrt()),
    )
}

fn condition_prior(
    prior: &SparseAnchorHodge1FormPrior,
    scaled_selector: &FeecCsr,
    scaled_observations: &FeecVector,
) -> Result<ConditioningOutput, String> {
    let latent_observation_matrix = scaled_selector * &prior.latent_to_ambient;
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&prior.precision),
        &feec_csr_to_gmrf(&latent_observation_matrix),
        &feec_vec_to_gmrf(scaled_observations),
        None,
        1.0,
    );
    let posterior = Gmrf::from_information_and_precision(information, posterior_precision)
        .map_err(|err| err.to_string())?;
    let latent_mean = gmrf_vec_to_feec(posterior.mean());
    let ambient_mean = &prior.latent_to_ambient * &latent_mean;
    Ok(ConditioningOutput {
        latent_mean,
        ambient_mean,
    })
}

fn branch_ambient_mean(
    latent_mean: &FeecVector,
    branch: &SparseAnchorHodge1FormBranch,
) -> FeecVector {
    let range = branch.latent_range();
    let branch_latent = FeecVector::from_iterator(
        branch.latent_dimension,
        range.map(|index| latent_mean[index]),
    );
    &branch.transform * &branch_latent
}

#[allow(clippy::too_many_arguments)]
fn diagnostics(
    label: impl Into<String>,
    field: &FeecVector,
    observation_selector: &FeecCsr,
    observations: &FeecVector,
    cycle_selector: &FeecCsr,
    truth_periods: &FeecVector,
    mass_1: &FeecCsr,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> BranchPosteriorDiagnostics {
    let observation_residual = observation_selector * field - observations;
    let weighted = mass_1 * field;
    let closure = d1 * field;
    let coclosed = d0.transpose() * &weighted;
    let cycle_error = cycle_selector * field - truth_periods;
    BranchPosteriorDiagnostics {
        label: label.into(),
        observation_residual_relative: observation_residual.norm() / observations.norm().max(EPS),
        mass_norm: mass_norm(field, mass_1),
        closure_residual_relative: closure.norm() / field.norm().max(EPS),
        coclosed_residual_relative: coclosed.norm() / weighted.norm().max(EPS),
        cycle_period_error_relative: cycle_error.norm() / truth_periods.norm().max(EPS),
        harmonic_latent_dimension: None,
        harmonic_coefficient_error_relative: None,
    }
}

fn add_harmonic_diagnostics(
    mut diagnostics: BranchPosteriorDiagnostics,
    field: &FeecVector,
    truth_harmonic: &FeecVector,
    prior: &SparseAnchorHodge1FormPrior,
) -> BranchPosteriorDiagnostics {
    let posterior_coefficients =
        harmonic_coefficients_in_basis(field, &prior.harmonic_basis, &prior.mass_1form);
    let truth_coefficients =
        harmonic_coefficients_in_basis(truth_harmonic, &prior.harmonic_basis, &prior.mass_1form);
    diagnostics.harmonic_latent_dimension = Some(prior.harmonic_basis.ncols());
    diagnostics.harmonic_coefficient_error_relative = Some(relative_vector_error(
        &posterior_coefficients,
        &truth_coefficients,
    ));
    diagnostics
}

fn harmonic_coefficients_in_basis(
    field: &FeecVector,
    harmonic_basis: &FeecMatrix,
    mass_1: &FeecCsr,
) -> FeecVector {
    let weighted = mass_1 * field;
    FeecVector::from_iterator(
        harmonic_basis.ncols(),
        (0..harmonic_basis.ncols()).map(|col| harmonic_basis.column(col).dot(&weighted)),
    )
}

fn relative_vector_error(value: &FeecVector, truth: &FeecVector) -> f64 {
    (value - truth).norm() / truth.norm().max(EPS)
}

fn mass_normalized(field: &FeecVector, mass: &FeecCsr) -> Result<FeecVector, String> {
    let norm = mass_norm(field, mass);
    if !norm.is_finite() || norm <= EPS {
        return Err("cannot normalize a zero or non-finite 1-form field".to_string());
    }
    Ok(field / norm)
}

fn mass_norm(field: &FeecVector, mass: &FeecCsr) -> f64 {
    let weighted = mass * field;
    field.dot(&weighted).max(0.0).sqrt()
}

fn validate_config(config: &Genus2SparseAnchorBranchConditioningConfig) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.alpha == MaternAlpha::Three {
        return Err("sparse-anchor branch conditioning supports alpha=1 or alpha=2".to_string());
    }
    if config.harmonic_dim != HARMONIC_COEFFICIENTS.len() {
        return Err(format!(
            "genus-2 sparse-anchor validation expects harmonic_dim={}",
            HARMONIC_COEFFICIENTS.len()
        ));
    }
    if !config.harmonic_precision.is_finite() || config.harmonic_precision <= 0.0 {
        return Err("harmonic precision must be finite and positive".to_string());
    }
    if config.local_observation_count == 0 {
        return Err("local observation count must be positive".to_string());
    }
    if !config.local_noise_variance.is_finite() || config.local_noise_variance <= 0.0 {
        return Err("local noise variance must be finite and positive".to_string());
    }
    if !config.cycle_noise_variance.is_finite() || config.cycle_noise_variance <= 0.0 {
        return Err("cycle noise variance must be finite and positive".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::lock_feec_harmonic_tests;

    #[test]
    fn genus2_sparse_anchor_branch_conditioning_preserves_branch_structure() {
        let _lock = lock_feec_harmonic_tests();
        let report = compute_genus2_sparse_anchor_branch_conditioning(
            Genus2SparseAnchorBranchConditioningConfig {
                local_observation_count: 12,
                ..Genus2SparseAnchorBranchConditioningConfig::default()
            },
        )
        .expect("genus-2 branch conditioning validation should run");

        eprintln!(
            "topology={:?} harmonic_dim={} cycle_harmonic_pairing_rank={} truth_harmonic_coefficients={:?}",
            report.topology_summary,
            report.harmonic_latent_dimension,
            report.cycle_harmonic_pairing_rank,
            report.truth_harmonic_coefficients
        );
        for diagnostics in report.diagnostics() {
            eprintln!("{diagnostics:?}");
            assert!(diagnostics.observation_residual_relative.is_finite());
            assert!(diagnostics.mass_norm.is_finite());
            assert!(diagnostics.mass_norm > 0.0);
            assert!(diagnostics.closure_residual_relative.is_finite());
            assert!(diagnostics.coclosed_residual_relative.is_finite());
            assert!(diagnostics.cycle_period_error_relative.is_finite());
            if let Some(error) = diagnostics.harmonic_coefficient_error_relative {
                assert!(error.is_finite());
            }
        }

        assert_eq!(report.topology_summary.b1, 4);
        assert_eq!(report.topology_summary.b2, 1);
        assert_eq!(report.harmonic_latent_dimension, 4);
        assert_eq!(report.cycle_harmonic_pairing_rank, 4);
        assert!(report.truth_exact_mass_norm > 0.0);
        assert!(report.truth_coexact_mass_norm > 0.0);
        assert!(report.truth_harmonic_mass_norm > 0.0);
        assert!(report.truth_mixed_mass_norm > 0.0);
        assert!(report.exact_only.closure_residual_relative <= EXACT_CLOSURE_TOLERANCE);
        assert!(report.joint_exact.closure_residual_relative <= EXACT_CLOSURE_TOLERANCE);
        assert!(report.coexact_only.coclosed_residual_relative <= COEXACT_COCLOSED_TOLERANCE);
        assert!(report.joint_coexact.coclosed_residual_relative <= COEXACT_COCLOSED_TOLERANCE);
        assert!(report.harmonic_only.closure_residual_relative <= HARMONIC_STRUCTURE_TOLERANCE);
        assert!(report.harmonic_only.coclosed_residual_relative <= HARMONIC_STRUCTURE_TOLERANCE);
        assert!(report.joint_harmonic.closure_residual_relative <= HARMONIC_STRUCTURE_TOLERANCE);
        assert!(report.joint_harmonic.coclosed_residual_relative <= HARMONIC_STRUCTURE_TOLERANCE);
    }
}
