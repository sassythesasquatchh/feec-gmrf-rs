//! Reusable linear model builders for the 2D annulus H-formulation benchmark.

use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix};
use ddf::ManifoldComplexExt;
use feg_infer::prior::{
    matern::{
        zero_form::{
            build_laplace_beltrami_0form, build_matern_mass_inverse_0form,
            build_matern_system_matrix_0form, MaternMassInverse as Matern0FormMassInverse,
        },
        MaternAlpha,
    },
    sparse_anchor_hodge::spectrum_matched_potential_precision,
};
use feg_infer::sparse::{
    add_sparse, block_diag_feec_csr, dense_to_feec_csr, hstack_feec_csr, scale_matrix,
};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnnulusModelKind {
    ComponentwiseGp,
    ComponentwiseGpCorrection,
    ScalarPotentialGp,
    ScalarPotentialGpCorrection,
    ExactOnlyFeec,
    FeecSplitNoSpectralCorrection,
    FeecGmrf,
}

impl AnnulusModelKind {
    pub fn benchmark_models() -> [Self; 5] {
        [
            Self::ComponentwiseGp,
            Self::ComponentwiseGpCorrection,
            Self::ScalarPotentialGpCorrection,
            Self::FeecSplitNoSpectralCorrection,
            Self::FeecGmrf,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ComponentwiseGp => "componentwise_gp",
            Self::ComponentwiseGpCorrection => "componentwise_gp_correction",
            Self::ScalarPotentialGp => "scalar_potential_gp",
            Self::ScalarPotentialGpCorrection => "scalar_potential_gp_correction",
            Self::ExactOnlyFeec => "exact_only_feec",
            Self::FeecSplitNoSpectralCorrection => "feec_split_no_spectral_correction",
            Self::FeecGmrf => "feec_gmrf",
        }
    }
}

impl std::str::FromStr for AnnulusModelKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "componentwise_gp" | "componentwise-gp" => Ok(Self::ComponentwiseGp),
            "componentwise_gp_correction" | "componentwise-gp-correction" => {
                Ok(Self::ComponentwiseGpCorrection)
            }
            "scalar_potential_gp" | "scalar-potential-gp" => Ok(Self::ScalarPotentialGp),
            "scalar_potential_gp_correction" | "scalar-potential-gp-correction" => {
                Ok(Self::ScalarPotentialGpCorrection)
            }
            "exact_only_feec" | "exact-only-feec" => Ok(Self::ExactOnlyFeec),
            "feec_split_no_spectral_correction" | "feec-split-no-spectral-correction" => {
                Ok(Self::FeecSplitNoSpectralCorrection)
            }
            "feec_gmrf" | "feec-gmrf" => Ok(Self::FeecGmrf),
            other => Err(format!(
                "unknown annulus model `{other}`; expected componentwise_gp, componentwise_gp_correction, scalar_potential_gp, scalar_potential_gp_correction, exact_only_feec, feec_split_no_spectral_correction, or feec_gmrf"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnulusLinearModel {
    pub model_kind: AnnulusModelKind,
    pub prior_precision: FeecCsr,
    pub latent_to_h: FeecCsr,
    pub h_offset: common::linalg::nalgebra::Vector,
    pub q_column: Option<usize>,
    pub selected_kappa: f64,
    pub selected_tau0: f64,
    pub selected_tau1: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnnulusExactPriorKind {
    StandardPotential,
    SpectrumMatchedAlpha1,
    SpectrumMatchedAlpha2,
}

impl AnnulusExactPriorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StandardPotential => "standard_potential",
            Self::SpectrumMatchedAlpha1 => "spectrum_matched_alpha1",
            Self::SpectrumMatchedAlpha2 => "spectrum_matched_alpha2",
        }
    }

    pub fn spectrum_matched_alpha(self) -> Option<MaternAlpha> {
        match self {
            Self::StandardPotential => None,
            Self::SpectrumMatchedAlpha1 => Some(MaternAlpha::One),
            Self::SpectrumMatchedAlpha2 => Some(MaternAlpha::Two),
        }
    }
}

impl std::str::FromStr for AnnulusExactPriorKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "standard" | "standard-potential" | "standard_potential" => {
                Ok(Self::StandardPotential)
            }
            "spectrum-matched" | "spectrum-matched-alpha2" | "spectrum_matched_alpha2" => {
                Ok(Self::SpectrumMatchedAlpha2)
            }
            "spectrum-matched-alpha1" | "spectrum_matched_alpha1" => {
                Ok(Self::SpectrumMatchedAlpha1)
            }
            other => Err(format!(
                "unknown annulus exact prior `{other}`; expected standard-potential, spectrum-matched-alpha1, or spectrum-matched-alpha2"
            )),
        }
    }
}

impl AnnulusLinearModel {
    pub fn latent_dimension(&self) -> usize {
        self.prior_precision.nrows()
    }
}

#[derive(Debug, Clone)]
pub struct AnnulusPotentialPrior {
    pub q_phi_full: FeecCsr,
    pub q_phi_latent: FeecCsr,
    pub d0_latent: FeecCsr,
}

#[derive(Debug, Clone, Copy)]
pub struct AnnulusPotentialPriorConfig {
    pub tau0: f64,
    pub tau1: f64,
    pub sigma_q: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AnnulusComponentwiseGpConfig {
    pub kappa: f64,
    pub tau: f64,
    pub jitter_scale: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AnnulusScalarPotentialGpConfig {
    pub kappa: f64,
    pub tau: f64,
    pub jitter_scale: f64,
}

pub fn build_feec_gmrf_model(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    psi_h: &common::linalg::nalgebra::Vector,
    config: AnnulusPotentialPriorConfig,
) -> Result<AnnulusLinearModel, String> {
    build_feec_gmrf_model_with_exact_prior(
        topology,
        metric,
        mass_1form,
        psi_h,
        config,
        AnnulusExactPriorKind::SpectrumMatchedAlpha2,
    )
}

fn build_feec_gmrf_model_with_exact_prior(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    psi_h: &common::linalg::nalgebra::Vector,
    config: AnnulusPotentialPriorConfig,
    exact_prior: AnnulusExactPriorKind,
) -> Result<AnnulusLinearModel, String> {
    build_exact_plus_harmonic_model(
        AnnulusModelKind::FeecGmrf,
        topology,
        metric,
        mass_1form,
        psi_h,
        config,
        exact_prior,
    )
}

pub fn build_feec_split_no_spectral_correction_model(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    psi_h: &common::linalg::nalgebra::Vector,
    config: AnnulusPotentialPriorConfig,
) -> Result<AnnulusLinearModel, String> {
    build_exact_plus_harmonic_model(
        AnnulusModelKind::FeecSplitNoSpectralCorrection,
        topology,
        metric,
        mass_1form,
        psi_h,
        config,
        AnnulusExactPriorKind::StandardPotential,
    )
}

pub fn build_exact_only_feec_model(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: AnnulusPotentialPriorConfig,
) -> Result<AnnulusLinearModel, String> {
    build_exact_correction_model(
        AnnulusModelKind::ExactOnlyFeec,
        topology,
        metric,
        mass_1form,
        &common::linalg::nalgebra::Vector::zeros(topology.nsimplices(1)),
        config,
        AnnulusExactPriorKind::SpectrumMatchedAlpha2,
    )
}

pub fn build_componentwise_gp_model(
    topology: &Complex,
    coords: &MeshCoords,
    config: AnnulusComponentwiseGpConfig,
) -> Result<AnnulusLinearModel, String> {
    let scalar_precision = build_vertex_gp_precision(
        topology,
        coords,
        config.kappa,
        config.tau,
        config.jitter_scale,
    )?;
    Ok(AnnulusLinearModel {
        model_kind: AnnulusModelKind::ComponentwiseGp,
        prior_precision: block_diag_feec_csr(&[&scalar_precision, &scalar_precision]),
        latent_to_h: build_edge_component_operator(topology, coords),
        h_offset: common::linalg::nalgebra::Vector::zeros(topology.nsimplices(1)),
        q_column: None,
        selected_kappa: config.kappa,
        selected_tau0: config.tau,
        selected_tau1: f64::NAN,
    })
}

pub fn build_componentwise_gp_correction_model(
    topology: &Complex,
    coords: &MeshCoords,
    h_offset: &common::linalg::nalgebra::Vector,
    config: AnnulusComponentwiseGpConfig,
) -> Result<AnnulusLinearModel, String> {
    let mut model = build_componentwise_gp_model(topology, coords, config)?;
    validate_h_offset(topology, h_offset)?;
    model.model_kind = AnnulusModelKind::ComponentwiseGpCorrection;
    model.h_offset = h_offset.clone();
    Ok(model)
}

pub fn build_scalar_potential_gp_model(
    topology: &Complex,
    coords: &MeshCoords,
    config: AnnulusScalarPotentialGpConfig,
) -> Result<AnnulusLinearModel, String> {
    let q0 = build_vertex_gp_precision(
        topology,
        coords,
        config.kappa,
        config.tau,
        config.jitter_scale,
    )?;
    let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
    Ok(AnnulusLinearModel {
        model_kind: AnnulusModelKind::ScalarPotentialGp,
        prior_precision: q0,
        latent_to_h: d0,
        h_offset: common::linalg::nalgebra::Vector::zeros(topology.nsimplices(1)),
        q_column: None,
        selected_kappa: config.kappa,
        selected_tau0: config.tau,
        selected_tau1: f64::NAN,
    })
}

pub fn build_scalar_potential_gp_correction_model(
    topology: &Complex,
    coords: &MeshCoords,
    h_offset: &common::linalg::nalgebra::Vector,
    config: AnnulusScalarPotentialGpConfig,
) -> Result<AnnulusLinearModel, String> {
    let mut model = build_scalar_potential_gp_model(topology, coords, config)?;
    validate_h_offset(topology, h_offset)?;
    model.model_kind = AnnulusModelKind::ScalarPotentialGpCorrection;
    model.h_offset = h_offset.clone();
    Ok(model)
}

pub fn build_potential_prior(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: AnnulusPotentialPriorConfig,
) -> Result<AnnulusPotentialPrior, String> {
    validate_positive("potential prior tau0", config.tau0)?;
    validate_positive("potential prior tau1", config.tau1)?;
    let q_phi_full =
        build_scalar_potential_precision(topology, metric, mass_1form, config.tau0, config.tau1)?;
    let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
    Ok(AnnulusPotentialPrior {
        q_phi_latent: q_phi_full.clone(),
        q_phi_full,
        d0_latent: d0,
    })
}

pub fn build_edge_component_operator(topology: &Complex, coords: &MeshCoords) -> FeecCsr {
    let vertex_count = topology.nsimplices(0);
    let mut coo = FeecCoo::new(topology.nsimplices(1), 2 * vertex_count);
    for edge in topology.edges().handle_iter() {
        let row = edge.kidx();
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        let ca = coords.coord(a);
        let cb = coords.coord(b);
        let dx = cb[0] - ca[0];
        let dy = cb[1] - ca[1];
        coo.push(row, a, 0.5 * dx);
        coo.push(row, b, 0.5 * dx);
        coo.push(row, vertex_count + a, 0.5 * dy);
        coo.push(row, vertex_count + b, 0.5 * dy);
    }
    FeecCsr::from(&coo)
}

fn build_exact_correction_model(
    model_kind: AnnulusModelKind,
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    h_offset: &common::linalg::nalgebra::Vector,
    config: AnnulusPotentialPriorConfig,
    exact_prior: AnnulusExactPriorKind,
) -> Result<AnnulusLinearModel, String> {
    let prior = match exact_prior.spectrum_matched_alpha() {
        Some(alpha) => build_spectrum_matched_potential_prior(topology, metric, config, alpha)?,
        None => build_potential_prior(topology, metric, mass_1form, config)?,
    };
    validate_h_offset(topology, h_offset)?;
    Ok(AnnulusLinearModel {
        model_kind,
        prior_precision: prior.q_phi_latent,
        latent_to_h: prior.d0_latent,
        h_offset: h_offset.clone(),
        q_column: None,
        selected_kappa: f64::NAN,
        selected_tau0: config.tau0,
        selected_tau1: config.tau1,
    })
}

fn build_exact_plus_harmonic_model(
    model_kind: AnnulusModelKind,
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    psi_h: &common::linalg::nalgebra::Vector,
    config: AnnulusPotentialPriorConfig,
    exact_prior: AnnulusExactPriorKind,
) -> Result<AnnulusLinearModel, String> {
    validate_positive("harmonic prior sigma_q", config.sigma_q)?;
    validate_h_offset(topology, psi_h)?;
    let prior = match exact_prior.spectrum_matched_alpha() {
        Some(alpha) => build_spectrum_matched_potential_prior(topology, metric, config, alpha)?,
        None => build_potential_prior(topology, metric, mass_1form, config)?,
    };
    let q_column = prior.d0_latent.ncols();
    let q_precision = scalar_precision(1.0 / (config.sigma_q * config.sigma_q));
    let psi_column = vector_as_sparse_column(psi_h);
    Ok(AnnulusLinearModel {
        model_kind,
        prior_precision: block_diag_feec_csr(&[&prior.q_phi_latent, &q_precision]),
        latent_to_h: hstack_feec_csr(&[&prior.d0_latent, &psi_column])?,
        h_offset: common::linalg::nalgebra::Vector::zeros(topology.nsimplices(1)),
        q_column: Some(q_column),
        selected_kappa: f64::NAN,
        selected_tau0: config.tau0,
        selected_tau1: config.tau1,
    })
}

fn build_vertex_gp_precision(
    topology: &Complex,
    coords: &MeshCoords,
    kappa: f64,
    tau: f64,
    jitter_scale: f64,
) -> Result<FeecCsr, String> {
    validate_positive("vertex GP kappa", kappa)?;
    validate_positive("vertex GP tau", tau)?;
    validate_positive("vertex GP jitter scale", jitter_scale)?;

    let vertex_count = topology.nsimplices(0);
    let covariance_scale = 1.0 / (tau * tau);
    let mut covariance = FeecMatrix::zeros(vertex_count, vertex_count);
    for i in 0..vertex_count {
        let ci = coords.coord(i);
        for j in 0..=i {
            let cj = coords.coord(j);
            let dx = ci[0] - cj[0];
            let dy = ci[1] - cj[1];
            let r = (dx * dx + dy * dy).sqrt();
            let kr = kappa * r;
            let value = covariance_scale * (1.0 + kr) * (-kr).exp();
            covariance[(i, j)] = value;
            covariance[(j, i)] = value;
        }
    }
    let jitter = (jitter_scale * covariance_scale).max(1e-12);
    for i in 0..vertex_count {
        covariance[(i, i)] += jitter;
    }
    let precision = covariance.try_inverse().ok_or_else(|| {
        format!("failed to invert vertex GP covariance for kappa={kappa} tau={tau}")
    })?;
    Ok(dense_to_feec_csr(&precision, 0.0))
}

fn build_spectrum_matched_potential_prior(
    topology: &Complex,
    metric: &MeshLengths,
    config: AnnulusPotentialPriorConfig,
    alpha: MaternAlpha,
) -> Result<AnnulusPotentialPrior, String> {
    validate_positive("spectral potential tau0", config.tau0)?;
    validate_positive("spectral potential tau1", config.tau1)?;
    let kappa = (config.tau0 / config.tau1).sqrt();
    let tau = config.tau1.sqrt();
    let q_phi_full =
        build_spectrum_matched_potential_precision_full(topology, metric, alpha, kappa, tau)?;
    let selector = build_connected_component_anchor_selector(topology)?;
    let q_selector = &q_phi_full * &selector;
    let q_phi_latent = selector.transpose() * &q_selector;
    let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
    let d0_latent = &d0 * &selector;
    Ok(AnnulusPotentialPrior {
        q_phi_full,
        q_phi_latent,
        d0_latent,
    })
}

pub(crate) fn build_spectrum_matched_potential_precision_full(
    topology: &Complex,
    metric: &MeshLengths,
    alpha: MaternAlpha,
    kappa: f64,
    tau: f64,
) -> Result<FeecCsr, String> {
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let system = build_matern_system_matrix_0form(&laplace, kappa);
    let mass_inverse =
        build_matern_mass_inverse_0form(&laplace.mass, Matern0FormMassInverse::RowSumLumped);
    spectrum_matched_potential_precision(&system, &mass_inverse, alpha, kappa, tau)
}

fn build_scalar_potential_precision(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    tau0: f64,
    tau1: f64,
) -> Result<FeecCsr, String> {
    validate_positive("scalar potential tau0", tau0)?;
    validate_positive("scalar potential tau1", tau1)?;
    let mass_0form = crate::de_rham::mass_matrix_form(topology, metric, 0)?;
    let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
    let m1_d0 = mass_1form * &d0;
    let stiffness = d0.transpose() * &m1_d0;
    Ok(add_sparse(
        &scale_matrix(&mass_0form, tau0),
        &scale_matrix(&stiffness, tau1),
    ))
}

fn scalar_precision(value: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(1, 1);
    if value != 0.0 {
        coo.push(0, 0, value);
    }
    FeecCsr::from(&coo)
}

fn vector_as_sparse_column(vector: &common::linalg::nalgebra::Vector) -> FeecCsr {
    let mut coo = FeecCoo::new(vector.len(), 1);
    for (row, value) in vector.iter().copied().enumerate() {
        if value != 0.0 {
            coo.push(row, 0, value);
        }
    }
    FeecCsr::from(&coo)
}

fn build_connected_component_anchor_selector(topology: &Complex) -> Result<FeecCsr, String> {
    let dimension = topology.nsimplices(0);
    if dimension < 2 {
        return Err("sparse potential anchoring requires at least two vertices".to_string());
    }
    let anchors = connected_component_vertex_anchors(topology);
    if anchors.is_empty() || anchors.len() >= dimension {
        return Err(format!(
            "invalid sparse anchor count {} for dimension {dimension}",
            anchors.len()
        ));
    }
    let mut is_anchor = vec![false; dimension];
    for anchor in anchors {
        if anchor >= dimension {
            return Err(format!(
                "anchor index {anchor} exceeds dimension {dimension}"
            ));
        }
        is_anchor[anchor] = true;
    }
    let mut coo = FeecCoo::new(
        dimension,
        dimension - is_anchor.iter().filter(|&&flag| flag).count(),
    );
    let mut col = 0;
    for (row, anchored) in is_anchor.iter().copied().enumerate() {
        if anchored {
            continue;
        }
        coo.push(row, col, 1.0);
        col += 1;
    }
    Ok(FeecCsr::from(&coo))
}

fn connected_component_vertex_anchors(topology: &Complex) -> Vec<usize> {
    let vertex_count = topology.nsimplices(0);
    let mut adjacency = vec![Vec::new(); vertex_count];
    for edge in topology.edges().handle_iter() {
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        adjacency[a].push(b);
        adjacency[b].push(a);
    }

    let mut anchors = Vec::new();
    let mut seen = vec![false; vertex_count];
    for root in 0..vertex_count {
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

fn validate_positive(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be finite and positive"))
    }
}

fn validate_h_offset(
    topology: &Complex,
    h_offset: &common::linalg::nalgebra::Vector,
) -> Result<(), String> {
    if h_offset.len() == topology.nsimplices(1) {
        Ok(())
    } else {
        Err(format!(
            "H offset length {} does not match edge count {}",
            h_offset.len(),
            topology.nsimplices(1)
        ))
    }
}
