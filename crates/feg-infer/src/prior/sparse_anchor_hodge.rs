use crate::prior::hodge::{compute_harmonic_basis_1form, mass_orthonormalize_harmonic_basis_1form};
use crate::prior::matern::{
    build_lindgren_precision_from_system,
    one_form::{
        build_hodge_laplacian_1form, build_matern_mass_inverse_1form,
        build_matern_mass_inverse_1form_with_coords, MaternMassInverse as Matern1FormMassInverse,
    },
    two_form::{
        build_hodge_laplacian_2form, build_hodge_laplacian_2form_with_lower_mass_inverse_coords,
        build_matern_mass_inverse_2form, build_matern_mass_inverse_2form_with_coords,
        build_matern_system_matrix_2form, MaternMassInverse as Matern2FormMassInverse,
    },
    zero_form::{
        build_laplace_beltrami_0form, build_matern_mass_inverse_0form,
        build_matern_system_matrix_0form, MaternMassInverse as Matern0FormMassInverse,
    },
    MaternAlpha,
};
use crate::sparse::{
    add_sparse, block_diag_feec_csr, dense_to_feec_csr, feec_csr_to_core_triplet, feec_csr_to_gmrf,
    hstack_feec_csr, scale_matrix,
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix};
use ddf::ManifoldComplexExt;
use feg_core::{GaussianPriorSpec, HodgeBranchKind};
use formoniq::problems::hodge_laplace::{solve_hodge_laplace_harmonics_with_galmats, MixedGalmats};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::{collections::VecDeque, ops::Range};

const RANK_TOLERANCE: f64 = 1e-10;
const NULLSPACE_RESIDUAL_TOLERANCE: f64 = 1e-7;
#[cfg(test)]
const SPARSE_COEXACT_COCLOSED_RELATIVE_TOLERANCE: f64 = 2e-1;
#[cfg(test)]
const ORDINARY_POTENTIAL_3D_COEXACT_COCLOSED_RELATIVE_TOLERANCE: f64 = 1.0;

/// Matérn parameters for one non-harmonic Hodge branch.
#[derive(Debug, Clone, Copy)]
pub struct HodgeMaternBranchConfig {
    pub kappa: f64,
    pub tau: f64,
    pub alpha: MaternAlpha,
}

/// Compatibility alias for [`HodgeMaternBranchConfig`].
pub type SparseAnchorBranchConfig = HodgeMaternBranchConfig;

/// Placement of the requested Matérn spectrum in a decomposed Hodge prior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HodgeMaternSpectrum {
    /// Put the Matérn spectrum on the latent potential before `d`/`delta`.
    /// The resulting form covariance has the additional exterior-calculus
    /// eigenvalue factor `lambda`.
    Potential,
    /// Choose the potential precision so the synthesized form branch has the
    /// requested Matérn spectrum, without the additional `lambda` factor.
    Form,
}

impl Default for HodgeMaternBranchConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            tau: 1.0,
            alpha: MaternAlpha::Two,
        }
    }
}

/// Shared configuration for potential- and form-spectrum Hodge--Matérn priors.
#[derive(Debug, Clone)]
pub struct HodgeMatern1FormPriorConfig {
    pub branches: Vec<HodgeBranchKind>,
    pub exact: HodgeMaternBranchConfig,
    pub coexact: HodgeMaternBranchConfig,
    pub harmonic_precision: f64,
    pub harmonic_dim: Option<usize>,
    pub harmonic_basis_override: Option<FeecMatrix>,
    pub zero_form_mass_inverse: Matern0FormMassInverse,
    pub one_form_mass_inverse: Matern1FormMassInverse,
    pub two_form_mass_inverse: Matern2FormMassInverse,
}

impl Default for HodgeMatern1FormPriorConfig {
    fn default() -> Self {
        Self {
            branches: HodgeBranchKind::ALL.to_vec(),
            exact: HodgeMaternBranchConfig::default(),
            coexact: HodgeMaternBranchConfig::default(),
            harmonic_precision: 1.0,
            harmonic_dim: None,
            harmonic_basis_override: None,
            zero_form_mass_inverse: Matern0FormMassInverse::default(),
            one_form_mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
            two_form_mass_inverse: Matern2FormMassInverse::default(),
        }
    }
}

impl HodgeMatern1FormPriorConfig {
    pub fn selected(branches: impl IntoIterator<Item = HodgeBranchKind>) -> Self {
        Self {
            branches: branches.into_iter().collect(),
            ..Self::default()
        }
    }
}

/// Compatibility alias for [`HodgeMatern1FormPriorConfig`].
pub type SparseAnchorHodge1FormPriorConfig = HodgeMatern1FormPriorConfig;

#[derive(Debug, Clone)]
pub struct OrdinaryPotentialHodge1FormPriorConfig {
    pub branches: Vec<HodgeBranchKind>,
    pub exact: SparseAnchorBranchConfig,
    pub coexact: SparseAnchorBranchConfig,
    pub zero_form_mass_inverse: Matern0FormMassInverse,
    pub one_form_mass_inverse: Matern1FormMassInverse,
    pub two_form_mass_inverse: Matern2FormMassInverse,
}

impl Default for OrdinaryPotentialHodge1FormPriorConfig {
    fn default() -> Self {
        Self {
            branches: vec![HodgeBranchKind::Exact, HodgeBranchKind::Coexact],
            exact: SparseAnchorBranchConfig::default(),
            coexact: SparseAnchorBranchConfig::default(),
            zero_form_mass_inverse: Matern0FormMassInverse::default(),
            one_form_mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
            two_form_mass_inverse: Matern2FormMassInverse::default(),
        }
    }
}

impl OrdinaryPotentialHodge1FormPriorConfig {
    pub fn selected(branches: impl IntoIterator<Item = HodgeBranchKind>) -> Self {
        Self {
            branches: branches.into_iter().collect(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgePotentialGauge {
    pub anchors: Vec<usize>,
    pub kept_dofs: Vec<usize>,
    pub nullity: usize,
    pub max_transform_null_residual: f64,
}

/// Compatibility alias for [`HodgePotentialGauge`].
pub type SparseAnchorGauge = HodgePotentialGauge;

#[derive(Debug, Clone)]
pub struct HodgeMatern1FormBranch {
    pub kind: HodgeBranchKind,
    pub offset: usize,
    pub latent_dimension: usize,
    pub ambient_dimension: usize,
    pub precision: FeecCsr,
    pub transform: FeecCsr,
    pub gauge: Option<HodgePotentialGauge>,
}

/// Compatibility alias for [`HodgeMatern1FormBranch`].
pub type SparseAnchorHodge1FormBranch = HodgeMatern1FormBranch;

impl HodgeMatern1FormBranch {
    pub fn latent_range(&self) -> Range<usize> {
        self.offset..self.offset + self.latent_dimension
    }
}

/// A decomposed sparse Hodge--Matérn prior in its latent coordinates.
#[derive(Debug, Clone)]
pub struct HodgeMatern1FormPrior {
    pub ambient_dimension: usize,
    pub precision: FeecCsr,
    pub latent_to_ambient: FeecCsr,
    pub mass_1form: FeecCsr,
    pub harmonic_basis: FeecMatrix,
    pub branches: Vec<HodgeMatern1FormBranch>,
}

/// Compatibility alias for [`HodgeMatern1FormPrior`].
pub type SparseAnchorHodge1FormPrior = HodgeMatern1FormPrior;

impl HodgeMatern1FormPrior {
    pub fn latent_dimension(&self) -> usize {
        self.precision.nrows()
    }

    pub fn branch(&self, kind: HodgeBranchKind) -> Option<&SparseAnchorHodge1FormBranch> {
        self.branches.iter().find(|branch| branch.kind == kind)
    }

    pub fn gaussian_prior_spec(&self) -> GaussianPriorSpec {
        GaussianPriorSpec {
            mean: vec![0.0; self.latent_dimension()],
            precision: feec_csr_to_core_triplet(&self.precision),
        }
    }
}

pub fn build_hodge_matern_1form_prior(
    topology: &Complex,
    metric: &MeshLengths,
    spectrum: HodgeMaternSpectrum,
    config: HodgeMatern1FormPriorConfig,
) -> Result<HodgeMatern1FormPrior, String> {
    build_hodge_matern_1form_prior_with_optional_coords(topology, None, metric, spectrum, config)
}

pub fn build_hodge_matern_1form_prior_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    spectrum: HodgeMaternSpectrum,
    config: HodgeMatern1FormPriorConfig,
) -> Result<HodgeMatern1FormPrior, String> {
    build_hodge_matern_1form_prior_with_optional_coords(
        topology,
        Some(coords),
        metric,
        spectrum,
        config,
    )
}

pub fn build_sparse_anchor_hodge_1form_prior(
    topology: &Complex,
    metric: &MeshLengths,
    config: SparseAnchorHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    build_sparse_anchor_hodge_1form_prior_with_optional_coords(topology, None, metric, config)
}

pub fn build_sparse_anchor_hodge_1form_prior_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    config: SparseAnchorHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    build_sparse_anchor_hodge_1form_prior_with_optional_coords(
        topology,
        Some(coords),
        metric,
        config,
    )
}

pub fn build_ordinary_potential_hodge_1form_prior(
    topology: &Complex,
    metric: &MeshLengths,
    config: OrdinaryPotentialHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    build_ordinary_potential_hodge_1form_prior_with_optional_coords(topology, None, metric, config)
}

pub fn build_ordinary_potential_hodge_1form_prior_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    config: OrdinaryPotentialHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    build_ordinary_potential_hodge_1form_prior_with_optional_coords(
        topology,
        Some(coords),
        metric,
        config,
    )
}

fn build_sparse_anchor_hodge_1form_prior_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    config: SparseAnchorHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    build_hodge_matern_1form_prior_with_optional_coords(
        topology,
        coords,
        metric,
        HodgeMaternSpectrum::Form,
        config,
    )
}

fn build_hodge_matern_1form_prior_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    spectrum: HodgeMaternSpectrum,
    config: HodgeMatern1FormPriorConfig,
) -> Result<HodgeMatern1FormPrior, String> {
    validate_spectrum_config(&config, spectrum)?;
    if !(topology.dim() == 2 || topology.dim() == 3) {
        return Err(format!(
            "Hodge 1-form Matérn prior supports 2D/3D meshes, got dimension {}",
            topology.dim()
        ));
    }

    let hodge_1form = build_hodge_laplacian_1form(topology, metric);
    let ambient_dimension = hodge_1form.mass_u.nrows();
    let mut branches = Vec::with_capacity(config.branches.len());
    let mut harmonic_basis = FeecMatrix::zeros(ambient_dimension, 0);
    let mut offset = 0;

    for kind in &config.branches {
        let mut branch = match kind {
            HodgeBranchKind::Exact => match spectrum {
                HodgeMaternSpectrum::Potential => {
                    build_ordinary_potential_exact_branch(topology, metric, &config)?
                }
                HodgeMaternSpectrum::Form => build_exact_branch(topology, metric, &config)?,
            },
            HodgeBranchKind::Coexact => match spectrum {
                HodgeMaternSpectrum::Potential => build_ordinary_potential_coexact_branch(
                    topology,
                    coords,
                    metric,
                    &hodge_1form.mass_u,
                    &config,
                )?,
                HodgeMaternSpectrum::Form => {
                    build_coexact_branch(topology, coords, metric, &hodge_1form.mass_u, &config)?
                }
            },
            HodgeBranchKind::Harmonic => {
                let harmonic_dim = config
                    .harmonic_dim
                    .unwrap_or_else(|| topology.homology_dim(1));
                harmonic_basis = if let Some(basis) = &config.harmonic_basis_override {
                    validate_harmonic_basis_override(basis, ambient_dimension, harmonic_dim)?;
                    basis.clone()
                } else {
                    let basis = compute_harmonic_basis_1form(topology, metric, harmonic_dim, None)?;
                    mass_orthonormalize_harmonic_basis_1form(&basis, &hodge_1form.mass_u)?
                };
                build_harmonic_branch(&harmonic_basis, config.harmonic_precision)?
            }
        };
        branch.offset = offset;
        offset += branch.latent_dimension;
        branches.push(branch);
    }

    let precision_blocks = branches
        .iter()
        .map(|branch| &branch.precision)
        .collect::<Vec<_>>();
    let transform_blocks = branches
        .iter()
        .map(|branch| &branch.transform)
        .collect::<Vec<_>>();
    let precision = block_diag_feec_csr(&precision_blocks);
    let latent_to_ambient = hstack_feec_csr(&transform_blocks)?;

    feec_csr_to_gmrf(&precision)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("joint Hodge-Matérn precision did not factorize: {err}"))?;

    Ok(SparseAnchorHodge1FormPrior {
        ambient_dimension,
        precision,
        latent_to_ambient,
        mass_1form: hodge_1form.mass_u,
        harmonic_basis,
        branches,
    })
}

fn build_ordinary_potential_hodge_1form_prior_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    config: OrdinaryPotentialHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormPrior, String> {
    validate_ordinary_potential_config(&config)?;
    build_hodge_matern_1form_prior_with_optional_coords(
        topology,
        coords,
        metric,
        HodgeMaternSpectrum::Potential,
        SparseAnchorHodge1FormPriorConfig {
            branches: config.branches,
            exact: config.exact,
            coexact: config.coexact,
            zero_form_mass_inverse: config.zero_form_mass_inverse,
            one_form_mass_inverse: config.one_form_mass_inverse,
            two_form_mass_inverse: config.two_form_mass_inverse,
            ..SparseAnchorHodge1FormPriorConfig::default()
        },
    )
}

fn build_exact_branch(
    topology: &Complex,
    metric: &MeshLengths,
    config: &SparseAnchorHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormBranch, String> {
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let system = build_matern_system_matrix_0form(&laplace, config.exact.kappa);
    let mass_inverse =
        build_matern_mass_inverse_0form(&laplace.mass, config.zero_form_mass_inverse);
    let q_full = spectrum_matched_potential_precision(
        &system,
        &mass_inverse,
        config.exact.alpha,
        config.exact.kappa,
        config.exact.tau,
    )?;
    let transform_full = FeecCsr::from(&topology.exterior_derivative_operator(0));
    let anchors = connected_component_vertex_anchors(topology);
    let gauge = build_anchor_selector(
        q_full.nrows(),
        &anchors,
        anchors.len(),
        &transform_full,
        None,
    )?;
    let precision = restrict_precision_with_selector(&q_full, &gauge.selector);
    validate_branch_precision(HodgeBranchKind::Exact, &precision, &gauge.diagnostics)?;
    let transform = &transform_full * &gauge.selector;

    Ok(SparseAnchorHodge1FormBranch {
        kind: HodgeBranchKind::Exact,
        offset: 0,
        latent_dimension: precision.nrows(),
        ambient_dimension: transform.nrows(),
        precision,
        transform,
        gauge: Some(gauge.diagnostics),
    })
}

fn build_ordinary_potential_exact_branch(
    topology: &Complex,
    metric: &MeshLengths,
    config: &HodgeMatern1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormBranch, String> {
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let system = build_matern_system_matrix_0form(&laplace, config.exact.kappa);
    let mass_inverse =
        build_matern_mass_inverse_0form(&laplace.mass, config.zero_form_mass_inverse);
    let precision = build_lindgren_precision_from_system(
        &system,
        &mass_inverse,
        config.exact.alpha,
        config.exact.tau,
    );
    validate_ordinary_branch_precision(HodgeBranchKind::Exact, &precision)?;
    let transform = FeecCsr::from(&topology.exterior_derivative_operator(0));

    Ok(SparseAnchorHodge1FormBranch {
        kind: HodgeBranchKind::Exact,
        offset: 0,
        latent_dimension: precision.nrows(),
        ambient_dimension: transform.nrows(),
        precision,
        transform,
        gauge: None,
    })
}

fn build_coexact_branch(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &SparseAnchorHodge1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormBranch, String> {
    if topology.dim() < 2 {
        return Err("coexact 1-form branch requires a mesh dimension of at least 2".to_string());
    }

    let timing_enabled = sparse_anchor_timing_enabled();
    let started = std::time::Instant::now();
    let hodge_2form = if let Some(coords) = coords {
        build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
            topology,
            coords,
            metric,
            config.one_form_mass_inverse,
        )?
    } else if config.one_form_mass_inverse == Matern1FormMassInverse::BarycentricDualSparseInverse {
        return Err("barycentric-dual coexact branch requires coordinates".to_string());
    } else {
        build_hodge_laplacian_2form(topology, metric)?
    };
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact] hodge_2form mass_nnz={} laplacian_nnz={} elapsed={:.3}s",
            hodge_2form.mass_u.nnz(),
            hodge_2form.laplacian.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let system = build_matern_system_matrix_2form(&hodge_2form, config.coexact.kappa);
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact] system nnz={} elapsed={:.3}s",
            system.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let mass_inverse_2 = if let Some(coords) = coords {
        build_matern_mass_inverse_2form_with_coords(
            topology,
            coords,
            metric,
            &hodge_2form.mass_u,
            config.two_form_mass_inverse,
        )?
    } else {
        build_matern_mass_inverse_2form(
            topology,
            metric,
            &hodge_2form.mass_u,
            config.two_form_mass_inverse,
        )?
    };
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact] mass_inverse_2 nnz={} elapsed={:.3}s",
            mass_inverse_2.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let q_full = spectrum_matched_potential_precision(
        &system,
        &mass_inverse_2,
        config.coexact.alpha,
        config.coexact.kappa,
        config.coexact.tau,
    )?;
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact] q_full nnz={} elapsed={:.3}s",
            q_full.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let transform_full = build_coexact_transform(
        topology,
        coords,
        metric,
        mass_1form,
        config.one_form_mass_inverse,
    )?;
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact] transform_full nnz={} elapsed={:.3}s",
            transform_full.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let (precision, transform, gauge) = anchor_hodge_potential_branch(
        topology,
        metric,
        2,
        &q_full,
        &transform_full,
        HodgeBranchKind::Coexact,
    )?;
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact] anchor_branch precision_nnz={} transform_nnz={} elapsed={:.3}s",
            precision.nnz(),
            transform.nnz(),
            started.elapsed().as_secs_f64()
        );
    }

    Ok(SparseAnchorHodge1FormBranch {
        kind: HodgeBranchKind::Coexact,
        offset: 0,
        latent_dimension: precision.nrows(),
        ambient_dimension: transform.nrows(),
        precision,
        transform,
        gauge: Some(gauge),
    })
}

fn build_ordinary_potential_coexact_branch(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &HodgeMatern1FormPriorConfig,
) -> Result<SparseAnchorHodge1FormBranch, String> {
    if topology.dim() < 2 {
        return Err("coexact 1-form branch requires a mesh dimension of at least 2".to_string());
    }

    let hodge_2form = if let Some(coords) = coords {
        build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
            topology,
            coords,
            metric,
            config.one_form_mass_inverse,
        )?
    } else if config.one_form_mass_inverse == Matern1FormMassInverse::BarycentricDualSparseInverse {
        return Err("barycentric-dual coexact branch requires coordinates".to_string());
    } else {
        build_hodge_laplacian_2form(topology, metric)?
    };
    let system = build_matern_system_matrix_2form(&hodge_2form, config.coexact.kappa);
    let mass_inverse_2 = if let Some(coords) = coords {
        build_matern_mass_inverse_2form_with_coords(
            topology,
            coords,
            metric,
            &hodge_2form.mass_u,
            config.two_form_mass_inverse,
        )?
    } else {
        build_matern_mass_inverse_2form(
            topology,
            metric,
            &hodge_2form.mass_u,
            config.two_form_mass_inverse,
        )?
    };
    let precision = build_lindgren_precision_from_system(
        &system,
        &mass_inverse_2,
        config.coexact.alpha,
        config.coexact.tau,
    );
    validate_ordinary_branch_precision(HodgeBranchKind::Coexact, &precision)?;
    let transform = build_coexact_transform(
        topology,
        coords,
        metric,
        mass_1form,
        config.one_form_mass_inverse,
    )?;

    Ok(SparseAnchorHodge1FormBranch {
        kind: HodgeBranchKind::Coexact,
        offset: 0,
        latent_dimension: precision.nrows(),
        ambient_dimension: transform.nrows(),
        precision,
        transform,
        gauge: None,
    })
}

fn build_harmonic_branch(
    harmonic_basis: &FeecMatrix,
    harmonic_precision: f64,
) -> Result<SparseAnchorHodge1FormBranch, String> {
    if !harmonic_precision.is_finite() || harmonic_precision <= 0.0 {
        return Err("harmonic precision must be finite and positive".to_string());
    }
    let transform = dense_to_feec_csr(harmonic_basis, 0.0);
    let mut coo = FeecCoo::new(harmonic_basis.ncols(), harmonic_basis.ncols());
    for index in 0..harmonic_basis.ncols() {
        coo.push(index, index, harmonic_precision);
    }
    let precision = FeecCsr::from(&coo);

    Ok(SparseAnchorHodge1FormBranch {
        kind: HodgeBranchKind::Harmonic,
        offset: 0,
        latent_dimension: precision.nrows(),
        ambient_dimension: transform.nrows(),
        precision,
        transform,
        gauge: None,
    })
}

pub fn spectrum_matched_potential_precision(
    shifted_system: &FeecCsr,
    mass_inverse: &FeecCsr,
    alpha: MaternAlpha,
    kappa: f64,
    tau: f64,
) -> Result<FeecCsr, String> {
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    let next_alpha = match alpha {
        MaternAlpha::One => MaternAlpha::Two,
        MaternAlpha::Two => MaternAlpha::Three,
        MaternAlpha::Three => {
            return Err(
                "spectrum-matched decomposed branches currently support alpha=1 or alpha=2"
                    .to_string(),
            )
        }
    };
    let timing_enabled = std::env::var_os("FEG_MATERN_TIMINGS").is_some();
    let started = std::time::Instant::now();
    let r_alpha = build_lindgren_precision_from_system(shifted_system, mass_inverse, alpha, 1.0);
    if timing_enabled {
        eprintln!(
            "[spectrum_matched_potential_precision] r_alpha alpha={} nnz={} elapsed={:.3}s",
            alpha.as_u32(),
            r_alpha.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let r_next =
        build_lindgren_precision_from_system(shifted_system, mass_inverse, next_alpha, 1.0);
    if timing_enabled {
        eprintln!(
            "[spectrum_matched_potential_precision] r_next alpha={} nnz={} elapsed={:.3}s",
            next_alpha.as_u32(),
            r_next.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let unscaled = add_sparse(&r_next, &scale_matrix(&r_alpha, -(kappa * kappa)));
    if timing_enabled {
        eprintln!(
            "[spectrum_matched_potential_precision] combine nnz={} elapsed={:.3}s",
            unscaled.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(scale_matrix(&unscaled, tau * tau))
}

fn build_coexact_transform(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    mass_inverse: Matern1FormMassInverse,
) -> Result<FeecCsr, String> {
    let timing_enabled = sparse_anchor_timing_enabled();
    let started = std::time::Instant::now();
    let galmats = MixedGalmats::compute(topology, metric, 2);
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact_transform] galmats elapsed={:.3}s",
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let codif_u = FeecCsr::from(galmats.codif_u());
    let mass_inverse_1form = if let Some(coords) = coords {
        build_matern_mass_inverse_1form_with_coords(
            topology,
            coords,
            metric,
            mass_1form,
            mass_inverse,
        )?
    } else if mass_inverse == Matern1FormMassInverse::BarycentricDualSparseInverse {
        return Err("barycentric-dual coexact branch requires coordinates".to_string());
    } else {
        build_matern_mass_inverse_1form(topology, metric, mass_1form, mass_inverse)
    };
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact_transform] mass_inverse_1form nnz={} codif_nnz={} elapsed={:.3}s",
            mass_inverse_1form.nnz(),
            codif_u.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let transform = &mass_inverse_1form * &codif_u;
    if timing_enabled {
        eprintln!(
            "[sparse_anchor_coexact_transform] multiply nnz={} elapsed={:.3}s",
            transform.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(transform)
}

struct AnchorSelector {
    selector: FeecCsr,
    diagnostics: SparseAnchorGauge,
}

fn build_anchor_selector(
    dimension: usize,
    anchors: &[usize],
    nullity: usize,
    transform: &FeecCsr,
    nullspace: Option<&FeecMatrix>,
) -> Result<AnchorSelector, String> {
    let mut is_anchor = vec![false; dimension];
    for &anchor in anchors {
        if anchor >= dimension {
            return Err(format!(
                "anchor index {anchor} is out of bounds for dimension {dimension}"
            ));
        }
        if is_anchor[anchor] {
            return Err(format!("anchor index {anchor} was selected more than once"));
        }
        is_anchor[anchor] = true;
    }
    let kept_dofs = (0..dimension)
        .filter(|index| !is_anchor[*index])
        .collect::<Vec<_>>();
    let mut coo = FeecCoo::new(dimension, kept_dofs.len());
    for (reduced_col, full_row) in kept_dofs.iter().copied().enumerate() {
        coo.push(full_row, reduced_col, 1.0);
    }
    let max_transform_null_residual = nullspace
        .map(|basis| max_abs_sparse_dense_product(transform, basis))
        .unwrap_or(0.0);
    Ok(AnchorSelector {
        selector: FeecCsr::from(&coo),
        diagnostics: SparseAnchorGauge {
            anchors: anchors.to_vec(),
            kept_dofs,
            nullity,
            max_transform_null_residual,
        },
    })
}

fn restrict_precision_with_selector(precision: &FeecCsr, selector: &FeecCsr) -> FeecCsr {
    let middle = precision * selector;
    selector.transpose() * &middle
}

pub fn anchor_hodge_potential_branch(
    topology: &Complex,
    metric: &MeshLengths,
    source_degree: usize,
    precision_full: &FeecCsr,
    transform_full: &FeecCsr,
    kind: HodgeBranchKind,
) -> Result<(FeecCsr, FeecCsr, SparseAnchorGauge), String> {
    let timing_enabled = sparse_anchor_timing_enabled();
    let started = std::time::Instant::now();
    let harmonic_nullspace = source_harmonic_basis(topology, metric, source_degree)?;
    if timing_enabled {
        eprintln!(
            "[anchor_hodge_potential_branch] source_harmonic_basis rows={} cols={} elapsed={:.3}s",
            harmonic_nullspace.nrows(),
            harmonic_nullspace.ncols(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let anchors = choose_sparse_anchor_rows(&harmonic_nullspace, RANK_TOLERANCE)?;
    if timing_enabled {
        eprintln!(
            "[anchor_hodge_potential_branch] choose_anchors count={} elapsed={:.3}s",
            anchors.len(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let gauge = build_anchor_selector(
        precision_full.nrows(),
        &anchors,
        harmonic_nullspace.ncols(),
        transform_full,
        Some(&harmonic_nullspace),
    )?;
    if timing_enabled {
        eprintln!(
            "[anchor_hodge_potential_branch] build_selector elapsed={:.3}s",
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let precision = restrict_precision_with_selector(precision_full, &gauge.selector);
    if timing_enabled {
        eprintln!(
            "[anchor_hodge_potential_branch] restrict_precision nnz={} elapsed={:.3}s",
            precision.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    validate_branch_precision(kind, &precision, &gauge.diagnostics)?;
    if timing_enabled {
        eprintln!(
            "[anchor_hodge_potential_branch] validate_precision elapsed={:.3}s",
            started.elapsed().as_secs_f64()
        );
    }
    let started = std::time::Instant::now();
    let transform = transform_full * &gauge.selector;
    if timing_enabled {
        eprintln!(
            "[anchor_hodge_potential_branch] restrict_transform nnz={} elapsed={:.3}s",
            transform.nnz(),
            started.elapsed().as_secs_f64()
        );
    }
    Ok((precision, transform, gauge.diagnostics))
}

fn sparse_anchor_timing_enabled() -> bool {
    std::env::var_os("FEG_SPARSE_ANCHOR_TIMINGS").is_some()
        || std::env::var_os("FEG_MATERN_TIMINGS").is_some()
}

fn validate_branch_precision(
    kind: HodgeBranchKind,
    precision: &FeecCsr,
    gauge: &SparseAnchorGauge,
) -> Result<(), String> {
    if gauge.max_transform_null_residual > NULLSPACE_RESIDUAL_TOLERANCE {
        return Err(format!(
            "{} branch transform does not annihilate the source harmonic nullspace; max residual {:.3e}",
            kind.as_str(),
            gauge.max_transform_null_residual
        ));
    }
    feec_csr_to_gmrf(precision)
        .cholesky_sqrt_lower()
        .map(|_| ())
        .map_err(|err| {
            format!(
                "{} branch precision did not factorize after anchoring {} null modes at {:?}: {err}",
                kind.as_str(),
                gauge.nullity,
                gauge.anchors
            )
        })
}

fn validate_ordinary_branch_precision(
    kind: HodgeBranchKind,
    precision: &FeecCsr,
) -> Result<(), String> {
    feec_csr_to_gmrf(precision)
        .cholesky_sqrt_lower()
        .map(|_| ())
        .map_err(|err| {
            format!(
                "{} ordinary-potential branch precision did not factorize: {err}",
                kind.as_str()
            )
        })
}

fn source_harmonic_basis(
    topology: &Complex,
    metric: &MeshLengths,
    source_degree: usize,
) -> Result<FeecMatrix, String> {
    if source_degree > topology.dim() {
        return Err(format!(
            "source harmonic degree {source_degree} exceeds mesh dimension {}",
            topology.dim()
        ));
    }
    if source_degree == topology.dim() {
        let galmats = MixedGalmats::compute(topology, metric, source_degree);
        let codifferential_sparse = FeecCsr::from(galmats.codif_u());
        if let Some(nullspace) =
            top_degree_volume_harmonic_basis(topology, metric, &codifferential_sparse, None)
        {
            return Ok(nullspace);
        }

        let harmonic_dim = homology_dim_safe(topology, source_degree);
        if harmonic_dim == 0 {
            return Ok(FeecMatrix::zeros(topology.nsimplices(source_degree), 0));
        }
        let codifferential = FeecMatrix::from(galmats.codif_u());
        let nullspace = dense_nullspace(&codifferential, RANK_TOLERANCE);
        if nullspace.ncols() != harmonic_dim {
            return Err(format!(
                "top-degree harmonic nullspace dimension {} did not match homology dimension {}",
                nullspace.ncols(),
                harmonic_dim
            ));
        }
        return Ok(nullspace);
    }

    let harmonic_dim = homology_dim_safe(topology, source_degree);
    if harmonic_dim == 0 {
        return Ok(FeecMatrix::zeros(topology.nsimplices(source_degree), 0));
    }
    let galmats = MixedGalmats::compute(topology, metric, source_degree);
    Ok(solve_hodge_laplace_harmonics_with_galmats(
        topology,
        &galmats,
        source_degree,
        harmonic_dim,
        None,
        None,
    ))
}

fn top_degree_volume_harmonic_basis(
    topology: &Complex,
    metric: &MeshLengths,
    codifferential: &FeecCsr,
    expected_harmonic_dim: Option<usize>,
) -> Option<FeecMatrix> {
    let degree = topology.dim();
    let ncells = topology.nsimplices(degree);
    if ncells == 0 || codifferential.ncols() != ncells {
        return None;
    }

    let boundary = FeecCsr::from(&topology.boundary_operator(degree));
    let mut face_entries = vec![Vec::<(usize, f64)>::new(); boundary.nrows()];
    for (row, col, value) in boundary.triplet_iter() {
        if value.abs() > RANK_TOLERANCE {
            face_entries[row].push((col, *value));
        }
    }

    let mut adjacency = vec![Vec::<(usize, f64)>::new(); ncells];
    for entries in face_entries {
        if entries.is_empty() {
            continue;
        }
        if entries.len() != 2 {
            return None;
        }
        let (left, left_sign) = entries[0];
        let (right, right_sign) = entries[1];
        if left_sign.abs() <= RANK_TOLERANCE || right_sign.abs() <= RANK_TOLERANCE {
            return None;
        }
        adjacency[left].push((right, -left_sign / right_sign));
        adjacency[right].push((left, -right_sign / left_sign));
    }

    let mut signed_component = vec![None::<(usize, f64)>; ncells];
    let mut components = Vec::<Vec<(usize, f64)>>::new();
    for root in 0..ncells {
        if signed_component[root].is_some() {
            continue;
        }
        let component_index = components.len();
        signed_component[root] = Some((component_index, 1.0));
        let mut component = Vec::new();
        let mut queue = VecDeque::from([root]);
        while let Some(cell) = queue.pop_front() {
            let (_, cell_sign) = signed_component[cell]?;
            component.push((cell, cell_sign));
            for &(next, relation) in &adjacency[cell] {
                let next_sign = relation * cell_sign;
                if let Some((seen_component, seen_sign)) = signed_component[next] {
                    if seen_component != component_index
                        || (seen_sign - next_sign).abs() > RANK_TOLERANCE
                    {
                        return None;
                    }
                } else {
                    signed_component[next] = Some((component_index, next_sign));
                    queue.push_back(next);
                }
            }
        }
        components.push(component);
    }
    if let Some(expected_harmonic_dim) = expected_harmonic_dim {
        if components.len() != expected_harmonic_dim {
            return None;
        }
    }
    if components.is_empty() {
        return None;
    }

    let mut basis = FeecMatrix::zeros(ncells, components.len());
    for (component_index, component) in components.iter().enumerate() {
        let mut norm_squared = 0.0;
        for &(cell_index, sign) in component {
            let cell = topology.cells().handle_by_kidx(cell_index);
            let value = sign * metric.simplex_lengths(cell).vol();
            basis[(cell_index, component_index)] = value;
            norm_squared += value * value;
        }
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || norm <= RANK_TOLERANCE {
            return None;
        }
        for &(cell_index, _) in component {
            basis[(cell_index, component_index)] /= norm;
        }
    }

    (max_abs_sparse_dense_product(codifferential, &basis) <= NULLSPACE_RESIDUAL_TOLERANCE)
        .then_some(basis)
}

fn dense_nullspace(operator: &FeecMatrix, tolerance: f64) -> FeecMatrix {
    if operator.ncols() == 0 {
        return FeecMatrix::zeros(0, 0);
    }
    if operator.nrows() == 0 {
        return FeecMatrix::identity(operator.ncols(), operator.ncols());
    }
    let svd = operator.clone().svd(true, true);
    let rank = svd
        .singular_values
        .iter()
        .filter(|value| value.abs() > tolerance)
        .count();
    let nullity = operator.ncols().saturating_sub(rank);
    if nullity == 0 {
        return FeecMatrix::zeros(operator.ncols(), 0);
    }
    let v_t = svd
        .v_t
        .expect("right singular vectors were requested for nullspace computation");
    FeecMatrix::from_fn(operator.ncols(), nullity, |row, col| v_t[(rank + col, row)])
}

fn homology_dim_safe(topology: &Complex, degree: usize) -> usize {
    if degree < topology.dim() {
        return topology.homology_dim(degree);
    }
    let boundary = FeecMatrix::from(&topology.boundary_operator(degree));
    boundary.ncols() - boundary.rank(1e-12)
}

fn choose_sparse_anchor_rows(nullspace: &FeecMatrix, tolerance: f64) -> Result<Vec<usize>, String> {
    let nullity = nullspace.ncols();
    if nullity == 0 {
        return Ok(Vec::new());
    }
    let mut anchors = Vec::with_capacity(nullity);
    let mut rank = 0;
    for row in 0..nullspace.nrows() {
        let mut candidate = anchors.clone();
        candidate.push(row);
        let candidate_rank = selected_row_rank(nullspace, &candidate, tolerance);
        if candidate_rank > rank {
            anchors.push(row);
            rank = candidate_rank;
            if rank == nullity {
                break;
            }
        }
    }
    if rank != nullity {
        return Err(format!(
            "could not choose sparse anchors for nullspace of dimension {nullity}; selected rank {rank}"
        ));
    }
    Ok(anchors)
}

fn selected_row_rank(matrix: &FeecMatrix, rows: &[usize], tolerance: f64) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let submatrix = FeecMatrix::from_fn(rows.len(), matrix.ncols(), |i, j| matrix[(rows[i], j)]);
    submatrix.rank(tolerance)
}

fn connected_component_vertex_anchors(topology: &Complex) -> Vec<usize> {
    let nvertices = topology.vertices().len();
    let mut adjacency = vec![Vec::new(); nvertices];
    for edge in topology.edges().handle_iter() {
        let vertices = edge.iter().collect::<Vec<_>>();
        if vertices.len() == 2 {
            adjacency[vertices[0]].push(vertices[1]);
            adjacency[vertices[1]].push(vertices[0]);
        }
    }

    let mut seen = vec![false; nvertices];
    let mut anchors = Vec::new();
    for root in 0..nvertices {
        if seen[root] {
            continue;
        }
        anchors.push(root);
        seen[root] = true;
        let mut queue = VecDeque::from([root]);
        while let Some(vertex) = queue.pop_front() {
            for &next in &adjacency[vertex] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
    }
    anchors
}

fn max_abs_sparse_dense_product(sparse: &FeecCsr, dense: &FeecMatrix) -> f64 {
    if dense.ncols() == 0 {
        return 0.0;
    }
    assert_eq!(sparse.ncols(), dense.nrows());
    let mut product = FeecMatrix::zeros(sparse.nrows(), dense.ncols());
    for (row, col, value) in sparse.triplet_iter() {
        for dense_col in 0..dense.ncols() {
            product[(row, dense_col)] += *value * dense[(col, dense_col)];
        }
    }
    product.iter().copied().map(f64::abs).fold(0.0, f64::max)
}

fn validate_spectrum_config(
    config: &HodgeMatern1FormPriorConfig,
    spectrum: HodgeMaternSpectrum,
) -> Result<(), String> {
    if config.branches.is_empty() {
        return Err("at least one Hodge-Matérn branch must be requested".to_string());
    }
    for branch in &config.branches {
        if config
            .branches
            .iter()
            .filter(|other| *other == branch)
            .count()
            > 1
        {
            return Err(format!(
                "Hodge-Matérn branch `{}` was requested more than once",
                branch.as_str()
            ));
        }
    }
    match spectrum {
        HodgeMaternSpectrum::Potential => {
            validate_positive_branch_config("exact", config.exact)?;
            validate_positive_branch_config("coexact", config.coexact)?;
        }
        HodgeMaternSpectrum::Form => {
            validate_form_spectrum_branch_config("exact", config.exact)?;
            validate_form_spectrum_branch_config("coexact", config.coexact)?;
        }
    }
    if !config.harmonic_precision.is_finite() || config.harmonic_precision <= 0.0 {
        return Err("harmonic precision must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_harmonic_basis_override(
    basis: &FeecMatrix,
    ambient_dimension: usize,
    harmonic_dim: usize,
) -> Result<(), String> {
    if basis.nrows() != ambient_dimension {
        return Err(format!(
            "harmonic basis override has {} rows but 1-form dimension is {}",
            basis.nrows(),
            ambient_dimension
        ));
    }
    if basis.ncols() != harmonic_dim {
        return Err(format!(
            "harmonic basis override has {} columns but harmonic dimension is {}",
            basis.ncols(),
            harmonic_dim
        ));
    }
    if !basis.iter().all(|value| value.is_finite()) {
        return Err("harmonic basis override contains non-finite values".to_string());
    }
    Ok(())
}

fn validate_ordinary_potential_config(
    config: &OrdinaryPotentialHodge1FormPriorConfig,
) -> Result<(), String> {
    if config.branches.is_empty() {
        return Err("at least one ordinary-potential branch must be requested".to_string());
    }
    for branch in &config.branches {
        if config
            .branches
            .iter()
            .filter(|other| *other == branch)
            .count()
            > 1
        {
            return Err(format!(
                "ordinary-potential branch `{}` was requested more than once",
                branch.as_str()
            ));
        }
        if *branch == HodgeBranchKind::Harmonic {
            return Err(
                "ordinary-potential diagnostic prior supports exact/coexact branches only"
                    .to_string(),
            );
        }
    }
    validate_positive_branch_config("exact", config.exact)?;
    validate_positive_branch_config("coexact", config.coexact)?;
    Ok(())
}

fn validate_form_spectrum_branch_config(
    name: &str,
    config: HodgeMaternBranchConfig,
) -> Result<(), String> {
    validate_positive_branch_config(name, config)?;
    if config.alpha == MaternAlpha::Three {
        return Err(format!(
            "{name} branch alpha=3 is not yet supported by the form-spectrum construction"
        ));
    }
    Ok(())
}

fn validate_positive_branch_config(
    name: &str,
    config: HodgeMaternBranchConfig,
) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(format!("{name} branch kappa must be finite and positive"));
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(format!("{name} branch tau must be finite and positive"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::dim3::mesh_sphere_surface;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn max_abs_csr(matrix: &FeecCsr) -> f64 {
        matrix
            .triplet_iter()
            .map(|(_, _, value)| value.abs())
            .fold(0.0, f64::max)
    }

    fn frobenius_norm_csr(matrix: &FeecCsr) -> f64 {
        matrix
            .triplet_iter()
            .map(|(_, _, value)| value * value)
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn top_degree_volume_harmonic_basis_is_codifferential_null_on_sphere() {
        let surface = mesh_sphere_surface(2);
        let (topology, coords) = surface.into_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let galmats = MixedGalmats::compute(&topology, &metric, topology.dim());
        let codifferential = FeecCsr::from(galmats.codif_u());
        let basis = top_degree_volume_harmonic_basis(&topology, &metric, &codifferential, Some(1))
            .expect("sphere top-degree harmonic basis should have a volume fast path");

        assert_eq!(basis.nrows(), topology.cells().len());
        assert_eq!(basis.ncols(), 1);
        assert!(
            max_abs_sparse_dense_product(&codifferential, &basis) <= NULLSPACE_RESIDUAL_TOLERANCE,
            "top-degree volume basis should be in the codifferential nullspace"
        );
    }

    #[test]
    fn harmonic_branch_uses_supplied_basis_override() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let mut harmonic_basis = FeecMatrix::zeros(topology.edges().len(), 1);
        harmonic_basis[(0, 0)] = 2.0;

        let prior = build_hodge_matern_1form_prior(
            &topology,
            &metric,
            HodgeMaternSpectrum::Form,
            HodgeMatern1FormPriorConfig {
                branches: vec![HodgeBranchKind::Harmonic],
                harmonic_dim: Some(1),
                harmonic_basis_override: Some(harmonic_basis.clone()),
                ..HodgeMatern1FormPriorConfig::default()
            },
        )
        .expect("harmonic branch with supplied basis should build");

        assert_eq!(prior.harmonic_basis, harmonic_basis);
        assert_eq!(prior.latent_to_ambient.ncols(), 1);
        assert_eq!(prior.latent_to_ambient.nrows(), topology.edges().len());
    }

    #[test]
    fn potential_matern_alpha_two_matches_legacy_decomposed_exact_branch() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let legacy = crate::prior::hodge::build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            crate::prior::hodge::Hodge1FormPriorConfig::branch(
                1.25,
                0.75,
                HodgeBranchKind::Exact,
                0,
            ),
        )
        .expect("legacy decomposed exact prior should build");
        let canonical = build_hodge_matern_1form_prior(
            &topology,
            &metric,
            HodgeMaternSpectrum::Potential,
            HodgeMatern1FormPriorConfig {
                branches: vec![HodgeBranchKind::Exact],
                exact: HodgeMaternBranchConfig {
                    kappa: 1.25,
                    tau: 0.75,
                    alpha: MaternAlpha::Two,
                },
                harmonic_dim: Some(0),
                ..HodgeMatern1FormPriorConfig::default()
            },
        )
        .expect("canonical potential-spectrum exact prior should build");

        assert_eq!(canonical.precision, legacy.precision);
        assert_eq!(canonical.latent_to_ambient, legacy.latent_to_ambient);
    }

    #[test]
    fn alpha_three_is_valid_for_potential_but_rejected_for_form_spectrum() {
        let config = HodgeMatern1FormPriorConfig {
            branches: vec![HodgeBranchKind::Exact],
            exact: HodgeMaternBranchConfig {
                alpha: MaternAlpha::Three,
                ..HodgeMaternBranchConfig::default()
            },
            ..HodgeMatern1FormPriorConfig::default()
        };

        assert!(validate_spectrum_config(&config, HodgeMaternSpectrum::Potential).is_ok());
        assert!(validate_spectrum_config(&config, HodgeMaternSpectrum::Form).is_err());
    }

    #[test]
    fn exact_branch_anchors_one_vertex_per_sphere_component_and_is_closed() {
        let surface = mesh_sphere_surface(1);
        let (topology, coords) = surface.into_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = build_sparse_anchor_hodge_1form_prior(
            &topology,
            &metric,
            SparseAnchorHodge1FormPriorConfig::selected([HodgeBranchKind::Exact]),
        )
        .expect("exact sparse-anchor prior should build");

        let exact = prior.branch(HodgeBranchKind::Exact).unwrap();
        assert_eq!(exact.gauge.as_ref().unwrap().nullity, 1);
        assert_eq!(exact.gauge.as_ref().unwrap().anchors.len(), 1);
        assert_eq!(exact.latent_dimension, topology.vertices().len() - 1);
        feec_csr_to_gmrf(&exact.precision)
            .cholesky_sqrt_lower()
            .expect("exact anchored precision should factorize");

        let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
        let closed_residual = &d1 * &exact.transform;
        assert!(max_abs_csr(&closed_residual) <= 1e-12);
    }

    #[test]
    fn coexact_branch_anchors_source_harmonics_and_factorizes_on_sphere() {
        let surface = mesh_sphere_surface(1);
        let (topology, coords) = surface.into_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = build_sparse_anchor_hodge_1form_prior(
            &topology,
            &metric,
            SparseAnchorHodge1FormPriorConfig::selected([HodgeBranchKind::Coexact]),
        )
        .expect("coexact sparse-anchor prior should build");

        let coexact = prior.branch(HodgeBranchKind::Coexact).unwrap();
        assert_eq!(coexact.gauge.as_ref().unwrap().nullity, 1);
        assert_eq!(coexact.gauge.as_ref().unwrap().anchors.len(), 1);
        assert!(
            coexact.gauge.as_ref().unwrap().max_transform_null_residual
                <= NULLSPACE_RESIDUAL_TOLERANCE
        );
        feec_csr_to_gmrf(&coexact.precision)
            .cholesky_sqrt_lower()
            .expect("coexact anchored precision should factorize");

        let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
        let weighted_transform = &prior.mass_1form * &coexact.transform;
        let coclosed_residual = &d0.transpose() * &weighted_transform;
        let relative_coclosed_defect =
            frobenius_norm_csr(&coclosed_residual) / frobenius_norm_csr(&weighted_transform);
        assert!(
            relative_coclosed_defect <= SPARSE_COEXACT_COCLOSED_RELATIVE_TOLERANCE,
            "projected sparse inverse coexact transform coclosed defect {relative_coclosed_defect:.3e} exceeded tolerance"
        );
    }

    #[test]
    fn exact_and_coexact_joint_precision_is_block_sparse_and_factorizes() {
        let surface = mesh_sphere_surface(1);
        let (topology, coords) = surface.into_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = build_sparse_anchor_hodge_1form_prior(
            &topology,
            &metric,
            SparseAnchorHodge1FormPriorConfig::selected([
                HodgeBranchKind::Exact,
                HodgeBranchKind::Coexact,
            ]),
        )
        .expect("joint exact/coexact sparse-anchor prior should build");

        assert_eq!(prior.branches.len(), 2);
        assert_eq!(prior.ambient_dimension, topology.edges().len());
        assert_eq!(prior.latent_to_ambient.nrows(), topology.edges().len());
        assert_eq!(prior.precision.nrows(), prior.latent_dimension());
        feec_csr_to_gmrf(&prior.precision)
            .cholesky_sqrt_lower()
            .expect("joint anchored precision should factorize");
    }

    #[test]
    fn sparse_anchor_row_selection_finds_full_rank_nullspace_rows() {
        let nullspace = FeecMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                2.0, 0.0, //
                0.0, 3.0, //
                0.0, 4.0,
            ],
        );
        let anchors = choose_sparse_anchor_rows(&nullspace, 1e-12).unwrap();
        assert_eq!(anchors.len(), 2);
        assert_eq!(selected_row_rank(&nullspace, &anchors, 1e-12), 2);
    }

    #[test]
    fn ordinary_potential_branches_factorize_and_preserve_differential_structure_on_cube() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = build_ordinary_potential_hodge_1form_prior_with_coords(
            &topology,
            &coords,
            &metric,
            OrdinaryPotentialHodge1FormPriorConfig::default(),
        )
        .expect("ordinary-potential diagnostic prior should build");

        assert_eq!(prior.branches.len(), 2);
        assert_eq!(prior.ambient_dimension, topology.edges().len());
        feec_csr_to_gmrf(&prior.precision)
            .cholesky_sqrt_lower()
            .expect("ordinary-potential joint precision should factorize");

        let exact = prior.branch(HodgeBranchKind::Exact).unwrap();
        let coexact = prior.branch(HodgeBranchKind::Coexact).unwrap();
        assert!(exact.gauge.is_none());
        assert!(coexact.gauge.is_none());

        let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
        let closed_residual = &d1 * &exact.transform;
        assert!(max_abs_csr(&closed_residual) <= 1e-12);

        let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
        let weighted_transform = &prior.mass_1form * &coexact.transform;
        let coclosed_residual = &d0.transpose() * &weighted_transform;
        let relative_coclosed_defect =
            frobenius_norm_csr(&coclosed_residual) / frobenius_norm_csr(&weighted_transform);
        assert!(
            relative_coclosed_defect
                <= ORDINARY_POTENTIAL_3D_COEXACT_COCLOSED_RELATIVE_TOLERANCE,
            "ordinary-potential coexact transform coclosed defect {relative_coclosed_defect:.3e} exceeded tolerance"
        );
    }
}
