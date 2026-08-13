//! Mixed-truth conditioning validation for sparse-anchor Hodge branches on S^2.
//!
//! The validation conditions exact-only, coexact-only, and joint exact+coexact
//! priors on the same observations from a mixed field. The purpose is to check
//! that posterior means retain branch structure under misspecification, not that
//! branch-only models fit mixed data perfectly.

use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
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
    dim3::mesh_sphere_surface,
    geometry::{
        coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
        metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
};

const EPS: f64 = 1e-12;
pub const EXACT_CLOSURE_TOLERANCE: f64 = 1e-10;
pub const COEXACT_COCLOSED_TOLERANCE: f64 = 2e-1;

#[derive(Debug, Clone)]
pub struct SphereSparseAnchorBranchConditioningConfig {
    pub level: usize,
    pub kappa: f64,
    pub tau: f64,
    pub alpha: MaternAlpha,
    pub noise_variance: f64,
    pub observation_count: usize,
}

impl Default for SphereSparseAnchorBranchConditioningConfig {
    fn default() -> Self {
        Self {
            level: 2,
            kappa: 1.0,
            tau: 1.0,
            alpha: MaternAlpha::Two,
            noise_variance: 1e-6,
            observation_count: 12,
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
}

#[derive(Debug, Clone)]
pub struct SphereSparseAnchorBranchConditioningReport {
    pub level: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub observation_count: usize,
    pub noise_variance: f64,
    pub truth_exact_mass_norm: f64,
    pub truth_coexact_mass_norm: f64,
    pub truth_mixed_mass_norm: f64,
    pub exact_only: BranchPosteriorDiagnostics,
    pub coexact_only: BranchPosteriorDiagnostics,
    pub joint_total: BranchPosteriorDiagnostics,
    pub joint_exact: BranchPosteriorDiagnostics,
    pub joint_coexact: BranchPosteriorDiagnostics,
}

impl SphereSparseAnchorBranchConditioningReport {
    pub fn diagnostics(&self) -> [&BranchPosteriorDiagnostics; 5] {
        [
            &self.exact_only,
            &self.coexact_only,
            &self.joint_total,
            &self.joint_exact,
            &self.joint_coexact,
        ]
    }
}

struct MeshData {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
}

struct MixedTruth {
    exact: FeecVector,
    coexact: FeecVector,
    mixed: FeecVector,
}

struct ConditioningOutput {
    latent_mean: FeecVector,
    ambient_mean: FeecVector,
}

pub fn compute_sphere_sparse_anchor_branch_conditioning(
    config: SphereSparseAnchorBranchConditioningConfig,
) -> Result<SphereSparseAnchorBranchConditioningReport, String> {
    validate_config(&config)?;

    let mesh = build_mesh(config.level);
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
    let joint_prior = build_prior(
        &mesh.topology,
        &mesh.coords,
        &mesh.metric,
        &config,
        [HodgeBranchKind::Exact, HodgeBranchKind::Coexact],
    )?;

    let truth = build_mixed_truth(&mesh, &exact_prior, &coexact_prior)?;
    let selector = edge_selector(
        mesh.topology.edges().len(),
        config.observation_count.min(mesh.topology.edges().len()),
    );
    let observations = observations_with_noise(&(&selector * &truth.mixed), config.noise_variance);

    let exact_conditioned = condition_prior(
        &exact_prior,
        &selector,
        &observations,
        config.noise_variance,
    )?;
    let coexact_conditioned = condition_prior(
        &coexact_prior,
        &selector,
        &observations,
        config.noise_variance,
    )?;
    let joint_conditioned = condition_prior(
        &joint_prior,
        &selector,
        &observations,
        config.noise_variance,
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

    let d1 = FeecCsr::from(&mesh.topology.exterior_derivative_operator(1));
    let d0 = FeecCsr::from(&mesh.topology.exterior_derivative_operator(0));
    let mass_1 = &joint_prior.mass_1form;

    Ok(SphereSparseAnchorBranchConditioningReport {
        level: config.level,
        vertex_count: mesh.topology.vertices().len(),
        edge_count: mesh.topology.edges().len(),
        face_count: mesh.topology.cells().len(),
        observation_count: selector.nrows(),
        noise_variance: config.noise_variance,
        truth_exact_mass_norm: mass_norm(&truth.exact, mass_1),
        truth_coexact_mass_norm: mass_norm(&truth.coexact, mass_1),
        truth_mixed_mass_norm: mass_norm(&truth.mixed, mass_1),
        exact_only: diagnostics(
            "exact_only",
            &exact_conditioned.ambient_mean,
            &selector,
            &observations,
            mass_1,
            &d0,
            &d1,
        ),
        coexact_only: diagnostics(
            "coexact_only",
            &coexact_conditioned.ambient_mean,
            &selector,
            &observations,
            mass_1,
            &d0,
            &d1,
        ),
        joint_total: diagnostics(
            "joint_total",
            &joint_conditioned.ambient_mean,
            &selector,
            &observations,
            mass_1,
            &d0,
            &d1,
        ),
        joint_exact: diagnostics(
            "joint_exact",
            &joint_exact_mean,
            &selector,
            &observations,
            mass_1,
            &d0,
            &d1,
        ),
        joint_coexact: diagnostics(
            "joint_coexact",
            &joint_coexact_mean,
            &selector,
            &observations,
            mass_1,
            &d0,
            &d1,
        ),
    })
}

fn build_mesh(level: usize) -> MeshData {
    let surface = mesh_sphere_surface(level);
    let (topology, coords) = surface.into_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    MeshData {
        topology,
        coords,
        metric,
    }
}

fn build_prior(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    config: &SphereSparseAnchorBranchConditioningConfig,
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
            harmonic_dim: Some(0),
            ..SparseAnchorHodge1FormPriorConfig::default()
        },
    )
}

fn build_mixed_truth(
    mesh: &MeshData,
    exact_prior: &SparseAnchorHodge1FormPrior,
    coexact_prior: &SparseAnchorHodge1FormPrior,
) -> Result<MixedTruth, String> {
    let d0 = FeecCsr::from(&mesh.topology.exterior_derivative_operator(0));
    let phi = vertex_potential(&mesh.coords);
    let exact = mass_normalized(&(&d0 * &phi), &exact_prior.mass_1form)?;

    let coexact_branch = coexact_prior
        .branch(HodgeBranchKind::Coexact)
        .ok_or_else(|| "coexact prior missing coexact branch".to_string())?;
    let face_potential = anchored_potential_values(
        &face_potential(&mesh.topology, &mesh.coords),
        coexact_branch
            .gauge
            .as_ref()
            .ok_or_else(|| "coexact branch missing sparse anchor gauge".to_string())?,
    );
    let coexact = mass_normalized(
        &(&coexact_branch.transform * &face_potential),
        &coexact_prior.mass_1form,
    )?;
    let mixed = &exact + &coexact;

    Ok(MixedTruth {
        exact,
        coexact,
        mixed,
    })
}

fn vertex_potential(coords: &MeshCoords) -> FeecVector {
    FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|vertex| {
            let coord = coords.coord(vertex);
            coord[0] + 0.25 * coord[2]
        }),
    )
}

fn face_potential(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let mut values = FeecVector::zeros(topology.cells().len());
    for face in topology.cells().handle_iter() {
        let barycenter = face.coord_simplex(coords).barycenter();
        values[face.kidx()] = barycenter[1] - 0.25 * barycenter[0] * barycenter[2];
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

fn edge_selector(edge_count: usize, observation_count: usize) -> FeecCsr {
    let indices = evenly_spaced_indices(edge_count, observation_count);
    let mut coo = FeecCoo::new(indices.len(), edge_count);
    for (row, edge) in indices.into_iter().enumerate() {
        coo.push(row, edge, 1.0);
    }
    FeecCsr::from(&coo)
}

fn observations_with_noise(clean: &FeecVector, noise_variance: f64) -> FeecVector {
    let amplitude = 0.25 * noise_variance.sqrt();
    FeecVector::from_iterator(
        clean.len(),
        clean.iter().enumerate().map(|(index, value)| {
            let pattern = ((index * 37 + 11) % 17) as f64 / 8.0 - 1.0;
            value + amplitude * pattern
        }),
    )
}

fn condition_prior(
    prior: &SparseAnchorHodge1FormPrior,
    selector: &FeecCsr,
    observations: &FeecVector,
    noise_variance: f64,
) -> Result<ConditioningOutput, String> {
    let latent_observation_matrix = selector * &prior.latent_to_ambient;
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&prior.precision),
        &feec_csr_to_gmrf(&latent_observation_matrix),
        &feec_vec_to_gmrf(observations),
        None,
        noise_variance,
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

fn diagnostics(
    label: impl Into<String>,
    field: &FeecVector,
    selector: &FeecCsr,
    observations: &FeecVector,
    mass_1: &FeecCsr,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> BranchPosteriorDiagnostics {
    let observation_residual = selector * field - observations;
    let weighted = mass_1 * field;
    let closure = d1 * field;
    let coclosed = d0.transpose() * &weighted;
    BranchPosteriorDiagnostics {
        label: label.into(),
        observation_residual_relative: observation_residual.norm() / observations.norm().max(EPS),
        mass_norm: mass_norm(field, mass_1),
        closure_residual_relative: closure.norm() / field.norm().max(EPS),
        coclosed_residual_relative: coclosed.norm() / weighted.norm().max(EPS),
    }
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

fn evenly_spaced_indices(len: usize, count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    if len <= count {
        return (0..len).collect();
    }
    (0..count).map(|i| i * len / count).collect()
}

fn validate_config(config: &SphereSparseAnchorBranchConditioningConfig) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.alpha == MaternAlpha::Three {
        return Err("sparse-anchor branch conditioning supports alpha=1 or alpha=2".to_string());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err("noise variance must be finite and positive".to_string());
    }
    if config.observation_count == 0 {
        return Err("observation count must be positive".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sparse_anchor_branch_conditioning_preserves_branch_structure() {
        let report = compute_sphere_sparse_anchor_branch_conditioning(
            SphereSparseAnchorBranchConditioningConfig {
                level: 1,
                observation_count: 10,
                ..SphereSparseAnchorBranchConditioningConfig::default()
            },
        )
        .expect("branch conditioning validation should run");

        for diagnostics in report.diagnostics() {
            eprintln!("{diagnostics:?}");
            assert!(diagnostics.observation_residual_relative.is_finite());
            assert!(diagnostics.mass_norm.is_finite());
            assert!(diagnostics.mass_norm > 0.0);
            assert!(diagnostics.closure_residual_relative.is_finite());
            assert!(diagnostics.coclosed_residual_relative.is_finite());
        }
        assert!(report.truth_exact_mass_norm > 0.0);
        assert!(report.truth_coexact_mass_norm > 0.0);
        assert!(report.truth_mixed_mass_norm > 0.0);
        assert!(report.exact_only.closure_residual_relative <= EXACT_CLOSURE_TOLERANCE);
        assert!(report.joint_exact.closure_residual_relative <= EXACT_CLOSURE_TOLERANCE);
        assert!(report.coexact_only.coclosed_residual_relative <= COEXACT_COCLOSED_TOLERANCE);
        assert!(report.joint_coexact.coclosed_residual_relative <= COEXACT_COCLOSED_TOLERANCE);
    }
}
