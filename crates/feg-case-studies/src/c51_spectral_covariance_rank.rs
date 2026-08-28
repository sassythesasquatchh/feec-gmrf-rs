//! C51 spectral covariance-rank discovery against the exact supported FEEC law.
//!
//! This study deliberately separates accuracy discovery from method timing.
//! One checked maximum mixed eigenbasis is shared across nested ranks, so every
//! row is permanently ineligible for a performance comparison.

use common::linalg::nalgebra::{CsrMatrix, Matrix};
use feec_gmrf::prelude::{
    EssentialBoundaryElimination, ExactFormOperators, ExactMaternPriorBuilder, FeecGmrfError,
    GaussianNoise, LinearGaussianModelBuilder, LinearMap, LinearObservation, MagneticFieldMaps3d,
    MagneticVolumeSensorBank3d, MaternAlpha, MaternParameters, Result, SparseMat,
    SupportedMaternPriorBuilder, WendlandC2Kernel3d,
};
#[cfg(feature = "external-solvers")]
use feec_gmrf::spectral::{
    partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative,
    PartialMixedGeneralizedEigenbasisDiagnostics, PartialMixedGeneralizedEigensolverOptions,
};
use feg_gp::{EigenBasis, SpectralMaternConfig, SpectralMaternGp};
use manifold::gen::cartesian::CartesianMeshInfo;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;

pub const C51_SPECTRAL_COVARIANCE_SCHEMA_VERSION: &str = "c51-spectral-covariance-rank-v2";
pub const C51_SPECTRAL_COVARIANCE_STUDY_ID: &str = "c51/spectral-covariance-rank";
pub const C51_SPECTRAL_RANK_GRID: [usize; 12] = [
    32, 64, 128, 256, 512, 768, 1_024, 1_280, 1_536, 1_792, 2_000, 2_048,
];
pub const C51_COVARIANCE_THRESHOLDS: [f64; 3] = [0.10, 0.05, 0.02];

const OUTPUT_DIMENSION: usize = 9;
const OBSERVATION_GRID_POINTS_PER_AXIS: usize = 6;
const OBSERVATION_CANDIDATE_COUNT: usize = 3
    * OBSERVATION_GRID_POINTS_PER_AXIS
    * OBSERVATION_GRID_POINTS_PER_AXIS
    * OBSERVATION_GRID_POINTS_PER_AXIS;
const EXACT_VARIANCE_RELATIVE_FLOOR: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct C51SpectralCovarianceRankConfig {
    pub profile: String,
    pub refinements: Vec<usize>,
    pub ranks: Vec<usize>,
    /// Optional staged-discovery cap selected from the frozen publication grid.
    pub requested_maximum_rank: Option<usize>,
    pub alphas: Vec<MaternAlpha>,
    pub kappa: f64,
    pub tau: f64,
    pub observation_noise_variance: f64,
    pub qoi_kernel_radius: f64,
    pub observation_kernel_radius: f64,
    /// Preflight cap for the retained eigenvector array alone. External
    /// process RSS and wall-time caps remain the responsibility of the runner.
    pub maximum_basis_storage_bytes: usize,
}

impl C51SpectralCovarianceRankConfig {
    /// Bounded schema/API smoke profile. It is not publication evidence.
    pub fn smoke() -> Self {
        Self {
            profile: "smoke".to_string(),
            refinements: vec![4],
            ranks: vec![8, 16, 32],
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two, MaternAlpha::Three],
            kappa: 4.0,
            tau: 1.0,
            observation_noise_variance: 0.05,
            qoi_kernel_radius: 0.18,
            observation_kernel_radius: 0.16,
            maximum_basis_storage_bytes: 256 * 1024 * 1024,
        }
    }

    /// Frozen publication discovery profile. Running it requires a fresh
    /// process with externally enforced resource caps.
    pub fn publication() -> Self {
        Self {
            profile: "publication-discovery-v1".to_string(),
            refinements: vec![8, 10],
            ranks: C51_SPECTRAL_RANK_GRID.to_vec(),
            alphas: vec![MaternAlpha::Two, MaternAlpha::Three],
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        }
    }

    /// Restrict a publication run to one mesh so n=8 can be inspected before n=10.
    pub fn for_publication_refinement(mut self, refinement: usize) -> Result<Self> {
        if !self.profile.starts_with("publication-discovery-v1") || ![8, 10].contains(&refinement) {
            return Err(invalid(
                "publication refinement must be 8 or 10 on the publication profile",
            ));
        }
        self.refinements = vec![refinement];
        self.validate()?;
        Ok(self)
    }

    /// Stage the frozen publication grid without changing any evaluated rank.
    pub fn with_maximum_rank(mut self, maximum_rank: usize) -> Result<Self> {
        if !self.profile.starts_with("publication-discovery-v1")
            || !C51_SPECTRAL_RANK_GRID.contains(&maximum_rank)
        {
            return Err(invalid(
                "publication maximum rank must be a member of the frozen rank grid",
            ));
        }
        self.ranks.retain(|rank| *rank <= maximum_rank);
        self.requested_maximum_rank = Some(maximum_rank);
        self.profile = format!("publication-discovery-v1-max-{maximum_rank}");
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.profile.trim().is_empty() {
            return Err(invalid("C51 spectral-rank profile must not be empty"));
        }
        if self.refinements.is_empty() || self.refinements.iter().any(|value| *value < 2) {
            return Err(invalid(
                "C51 spectral-rank refinements must contain values of at least two",
            ));
        }
        if self.ranks.is_empty() || self.ranks.contains(&0) {
            return Err(invalid(
                "C51 spectral-rank grid must be positive and nonempty",
            ));
        }
        if self.ranks.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                "C51 spectral-rank grid must be strictly increasing without duplicates",
            ));
        }
        if let Some(maximum_rank) = self.requested_maximum_rank {
            if !self.profile.starts_with("publication-discovery-v1-max-")
                || self.ranks.last().copied() != Some(maximum_rank)
                || !C51_SPECTRAL_RANK_GRID.contains(&maximum_rank)
            {
                return Err(invalid(
                    "staged publication profile must end at its recorded frozen-grid maximum rank",
                ));
            }
        }
        if self.profile == "publication-discovery-v1"
            && self.requested_maximum_rank.is_none()
            && (!self.ranks.contains(&2_000) || !self.ranks.contains(&2_048))
        {
            return Err(invalid(
                "publication rank discovery must retain mandatory ranks 2000 and 2048",
            ));
        }
        if self.alphas != [MaternAlpha::Two, MaternAlpha::Three] {
            return Err(invalid(
                "C51 spectral covariance study requires separate alpha=2 and alpha=3 runs",
            ));
        }
        MaternParameters::new(MaternAlpha::Two, self.kappa, self.tau)?;
        GaussianNoise::variance(self.observation_noise_variance)?;
        for (name, radius) in [
            ("QoI", self.qoi_kernel_radius),
            ("observation", self.observation_kernel_radius),
        ] {
            if !radius.is_finite() || radius <= 0.0 || radius >= 0.25 {
                return Err(invalid(format!(
                    "C51 {name} kernel radius must lie in (0, 0.25)"
                )));
            }
        }
        if self.maximum_basis_storage_bytes == 0 {
            return Err(invalid("C51 basis-storage cap must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C51SpectralCovarianceRankMetadata {
    pub command: String,
    pub root_sha: Option<String>,
    pub feec_sha: Option<String>,
    pub gmrf_sha: Option<String>,
    pub root_dirty: Option<bool>,
    pub feec_dirty: Option<bool>,
    pub gmrf_dirty: Option<bool>,
}

impl C51SpectralCovarianceRankMetadata {
    pub fn discovery(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            root_sha: None,
            feec_sha: None,
            gmrf_sha: None,
            root_dirty: None,
            feec_dirty: None,
            gmrf_dirty: None,
        }
    }

    /// Capture root and submodule revisions without silently replacing a
    /// failed query with guessed provenance.
    pub fn from_repository(command: impl Into<String>, repository_root: &Path) -> Result<Self> {
        Ok(Self {
            command: command.into(),
            root_sha: Some(git_output(repository_root, &["rev-parse", "HEAD"])?),
            feec_sha: Some(git_output(
                &repository_root.join("feec"),
                &["rev-parse", "HEAD"],
            )?),
            gmrf_sha: Some(git_output(
                &repository_root.join("gmrf-rs"),
                &["rev-parse", "HEAD"],
            )?),
            root_dirty: Some(!git_output(repository_root, &["status", "--porcelain"])?.is_empty()),
            feec_dirty: Some(
                !git_output(&repository_root.join("feec"), &["status", "--porcelain"])?.is_empty(),
            ),
            gmrf_dirty: Some(
                !git_output(&repository_root.join("gmrf-rs"), &["status", "--porcelain"])?
                    .is_empty(),
            ),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum C51CovarianceDesign {
    Prior,
    Fixed32,
    Density020,
    Density100,
}

impl C51CovarianceDesign {
    pub const ALL: [Self; 4] = [
        Self::Prior,
        Self::Fixed32,
        Self::Density020,
        Self::Density100,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prior => "prior",
            Self::Fixed32 => "fixed32",
            Self::Density020 => "density_0.02",
            Self::Density100 => "density_0.10",
        }
    }

    fn observation_count(self, state_dimension: usize) -> usize {
        match self {
            Self::Prior => 0,
            Self::Fixed32 => 32.min(OBSERVATION_CANDIDATE_COUNT),
            Self::Density020 => (0.02 * state_dimension as f64).ceil() as usize,
            Self::Density100 => (0.10 * state_dimension as f64).ceil() as usize,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51CovarianceErrorMetrics {
    pub relative_frobenius: f64,
    pub relative_spectral: f64,
    pub maximum_relative_diagonal: f64,
    pub maximum_relative_directional: f64,
    pub relative_correlation_frobenius: f64,
}

impl C51CovarianceErrorMetrics {
    pub fn primary_envelope(self) -> f64 {
        self.relative_frobenius
            .max(self.maximum_relative_diagonal)
            .max(self.maximum_relative_directional)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51SpectralCovarianceRankRow {
    pub refinement: usize,
    pub state_dimension: usize,
    pub alpha: u32,
    pub design: C51CovarianceDesign,
    pub observation_count: usize,
    pub observation_density: f64,
    pub rank: usize,
    /// Rank of the single shared basis from which this row was truncated.
    pub shared_maximum_basis_rank: usize,
    pub rank_fraction: f64,
    pub status: String,
    pub censored_stage: String,
    pub censored_reason: String,
    pub selection: Option<C51CovarianceErrorMetrics>,
    pub held_out: Option<C51CovarianceErrorMetrics>,
    /// Relative FEEC-mass norm errors for `C M p` at two fixed M-normalized
    /// primal probes. These are discrete latent diagnostics, not cross-mesh physical outputs.
    pub latent_mass_action_relative_errors: Option<[f64; 2]>,
    pub latent_mass_action_maximum_relative_error: Option<f64>,
    pub basis_storage_bytes: usize,
    pub prepared_storage_bytes: Option<usize>,
    pub method_performance_timing_eligible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C51SpectralBasisAttempt {
    pub refinement: usize,
    pub state_dimension: usize,
    pub requested_rank: usize,
    pub status: String,
    pub reason: String,
    pub diagnostics_status: String,
    pub diagnostics_unavailable_reason: String,
    pub mass_orthogonality_certificate_status: String,
    pub mass_orthogonality_certificate_reason: String,
    pub generalized_residual_certificate_status: String,
    pub generalized_residual_certificate_reason: String,
    pub external_tool_versions_status: String,
    pub external_tool_versions_reason: String,
    pub sigma_dimension: Option<usize>,
    pub converged_rank: Option<usize>,
    pub retained_rank: Option<usize>,
    pub input_nonzeros: Option<usize>,
    pub l_nonzeros: Option<usize>,
    pub d_nonzeros: Option<usize>,
    pub c_nonzeros: Option<usize>,
    pub lower_mass_nonzeros: Option<usize>,
    pub state_mass_nonzeros: Option<usize>,
    pub basis_storage_entries: Option<usize>,
    pub basis_storage_bytes: Option<usize>,
    pub clamped_negative_eigenvalues: Option<usize>,
    pub eigenpair_selection: Option<String>,
    pub lower_mass_solve: Option<String>,
    pub workspace: Option<String>,
    pub binary: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51OutputCovarianceRecord {
    pub refinement: usize,
    pub alpha: u32,
    pub design: C51CovarianceDesign,
    pub rank: Option<usize>,
    pub bank: &'static str,
    pub representation: &'static str,
    pub row: usize,
    pub column: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51SensorBankAudit {
    pub refinement: usize,
    pub bank: &'static str,
    pub kernels: Vec<WendlandC2Kernel3d>,
    pub quadrature_rule: &'static str,
    pub quadrature_normalizations: Vec<f64>,
    pub component_row_nonzeros: Vec<[usize; 3]>,
    /// Canonical composed cochain row entries, kernel-major then component-major.
    pub component_rows: Vec<[Vec<(usize, f64)>; 3]>,
    /// Pairs whose compact physical support balls overlap.
    pub overlapping_kernel_pairs: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51FullRankPlumbingGate {
    pub refinement: usize,
    pub state_dimension: usize,
    pub rank: usize,
    pub direct_supported_prior_primary_error: f64,
    pub direct_supported_posterior_primary_error: f64,
    pub spectral_supported_prior_primary_error: f64,
    pub spectral_supported_posterior_primary_error: f64,
    pub tolerance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51SpectralCovarianceRankArchive {
    pub metadata: C51SpectralCovarianceRankMetadata,
    pub config: C51SpectralCovarianceRankConfig,
    pub basis_attempts: Vec<C51SpectralBasisAttempt>,
    pub rows: Vec<C51SpectralCovarianceRankRow>,
    pub covariances: Vec<C51OutputCovarianceRecord>,
    pub sensor_banks: Vec<C51SensorBankAudit>,
    pub plumbing_gates: Vec<C51FullRankPlumbingGate>,
}

impl C51SpectralCovarianceRankArchive {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.basis_attempts.len() != self.config.refinements.len() {
            return Err(invalid(
                "C51 spectral-rank archive must contain one basis attempt per refinement",
            ));
        }
        for attempt in &self.basis_attempts {
            match attempt.diagnostics_status.as_str() {
                "measured" if attempt.diagnostics_unavailable_reason.is_empty() => {}
                "unavailable" if !attempt.diagnostics_unavailable_reason.is_empty() => {}
                _ => {
                    return Err(invalid(
                        "C51 basis diagnostics require measured or explicitly unavailable status",
                    ))
                }
            }
        }
        if self.rows.is_empty() {
            return Err(invalid("C51 spectral-rank archive has no rows"));
        }
        if self.sensor_banks.len() != 3 * self.config.refinements.len() {
            return Err(invalid(
                "C51 spectral-rank archive must contain selection, held-out, and observation sensor audits per refinement",
            ));
        }
        for bank in &self.sensor_banks {
            if bank.kernels.len() != bank.quadrature_normalizations.len()
                || bank.kernels.len() != bank.component_row_nonzeros.len()
                || bank.kernels.len() != bank.component_rows.len()
                || bank
                    .component_rows
                    .iter()
                    .enumerate()
                    .any(|(kernel, rows)| {
                        rows.iter().enumerate().any(|(component, row)| {
                            row.len() != bank.component_row_nonzeros[kernel][component]
                                || row.iter().any(|(_, weight)| !weight.is_finite())
                        })
                    })
            {
                return Err(invalid(
                    "C51 sensor audit must archive every finite composed row weight",
                ));
            }
        }
        if self
            .plumbing_gates
            .iter()
            .map(|gate| gate.refinement)
            .collect::<Vec<_>>()
            != [2, 4]
            || self.plumbing_gates.iter().any(|gate| {
                gate.rank != gate.state_dimension
                    || gate.direct_supported_prior_primary_error > gate.tolerance
                    || gate.direct_supported_posterior_primary_error > gate.tolerance
                    || gate.spectral_supported_prior_primary_error > gate.tolerance
                    || gate.spectral_supported_posterior_primary_error > gate.tolerance
            })
        {
            return Err(invalid(
                "C51 full-rank n=2/n=4 same-law plumbing gates are missing or failed",
            ));
        }
        for row in &self.rows {
            if row.method_performance_timing_eligible {
                return Err(invalid(
                    "shared-maximum spectral rank rows can never be performance eligible",
                ));
            }
            let measured = row.status == "measured";
            if measured != (row.selection.is_some() && row.held_out.is_some()) {
                return Err(invalid(
                    "measured spectral rank rows require both covariance metric banks",
                ));
            }
            if measured
                != (row.latent_mass_action_relative_errors.is_some()
                    && row.latent_mass_action_maximum_relative_error.is_some())
            {
                return Err(invalid(
                    "measured spectral rank rows require latent mass-action diagnostics",
                ));
            }
            if measured && (!row.censored_stage.is_empty() || !row.censored_reason.is_empty()) {
                return Err(invalid(
                    "measured spectral rank row carries censoring metadata",
                ));
            }
            if !measured && (row.censored_stage.is_empty() || row.censored_reason.is_empty()) {
                return Err(invalid(
                    "censored spectral rank row lacks typed stage/reason",
                ));
            }
        }
        Ok(())
    }

    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(directory)?;
        self.write_rows(&directory.join("rows.csv"))?;
        self.write_covariances(&directory.join("covariances.csv"))?;
        self.write_basis_attempts(&directory.join("basis_attempts.csv"))?;
        self.write_sensor_banks(&directory.join("sensor_banks.csv"))?;
        self.write_sensor_overlaps(&directory.join("sensor_overlaps.csv"))?;
        self.write_manifest(&directory.join("manifest.yaml"))?;
        Ok(())
    }

    fn write_rows(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "evidence_tier",
                "method_performance_timing_eligible",
                "refinement",
                "state_dimension",
                "alpha",
                "design",
                "observation_count",
                "observation_density",
                "rank",
                "shared_maximum_basis_rank",
                "rank_fraction",
                "status",
                "censored_stage",
                "censored_reason",
                "basis_storage_bytes",
                "prepared_storage_bytes",
                "selection_e_f",
                "selection_e_2",
                "selection_e_diag",
                "selection_e_dir",
                "selection_e_corr",
                "selection_e_primary",
                "heldout_e_f",
                "heldout_e_2",
                "heldout_e_diag",
                "heldout_e_dir",
                "heldout_e_corr",
                "heldout_e_primary",
                "latent_mass_action_probe_0_relative_error",
                "latent_mass_action_probe_1_relative_error",
                "latent_mass_action_maximum_relative_error",
            ])
            .map_err(csv_error)?;
        for row in &self.rows {
            let mut record = vec![
                "rank_discovery".to_string(),
                row.method_performance_timing_eligible.to_string(),
                row.refinement.to_string(),
                row.state_dimension.to_string(),
                row.alpha.to_string(),
                row.design.as_str().to_string(),
                row.observation_count.to_string(),
                format_float(row.observation_density),
                row.rank.to_string(),
                row.shared_maximum_basis_rank.to_string(),
                format_float(row.rank_fraction),
                row.status.clone(),
                row.censored_stage.clone(),
                row.censored_reason.clone(),
                row.basis_storage_bytes.to_string(),
                row.prepared_storage_bytes
                    .map_or_else(String::new, |value| value.to_string()),
            ];
            append_metrics(&mut record, row.selection);
            append_metrics(&mut record, row.held_out);
            match row.latent_mass_action_relative_errors {
                Some(errors) => record.extend([
                    format_float(errors[0]),
                    format_float(errors[1]),
                    format_float(
                        row.latent_mass_action_maximum_relative_error
                            .expect("validated measured latent action maximum"),
                    ),
                ]),
                None => record.extend([String::new(), String::new(), String::new()]),
            }
            writer.write_record(record).map_err(csv_error)?;
        }
        writer.flush()?;
        Ok(())
    }

    fn write_covariances(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "refinement",
                "alpha",
                "design",
                "rank",
                "bank",
                "representation",
                "row",
                "column",
                "value",
            ])
            .map_err(csv_error)?;
        for value in &self.covariances {
            writer
                .write_record([
                    value.refinement.to_string(),
                    value.alpha.to_string(),
                    value.design.as_str().to_string(),
                    value.rank.map_or_else(String::new, |rank| rank.to_string()),
                    value.bank.to_string(),
                    value.representation.to_string(),
                    value.row.to_string(),
                    value.column.to_string(),
                    format_float(value.value),
                ])
                .map_err(csv_error)?;
        }
        writer.flush()?;
        Ok(())
    }

    fn write_basis_attempts(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "refinement",
                "state_dimension",
                "requested_rank",
                "status",
                "reason",
                "diagnostics_status",
                "diagnostics_unavailable_reason",
                "mass_orthogonality_certificate_status",
                "mass_orthogonality_certificate_reason",
                "generalized_residual_certificate_status",
                "generalized_residual_certificate_reason",
                "external_tool_versions_status",
                "external_tool_versions_reason",
                "sigma_dimension",
                "converged_rank",
                "retained_rank",
                "input_nonzeros",
                "l_nonzeros",
                "d_nonzeros",
                "c_nonzeros",
                "lower_mass_nonzeros",
                "state_mass_nonzeros",
                "basis_storage_entries",
                "basis_storage_bytes",
                "clamped_negative_eigenvalues",
                "eigenpair_selection",
                "lower_mass_solve",
                "workspace",
                "binary",
            ])
            .map_err(csv_error)?;
        for attempt in &self.basis_attempts {
            writer
                .write_record([
                    attempt.refinement.to_string(),
                    attempt.state_dimension.to_string(),
                    attempt.requested_rank.to_string(),
                    attempt.status.clone(),
                    attempt.reason.clone(),
                    attempt.diagnostics_status.clone(),
                    attempt.diagnostics_unavailable_reason.clone(),
                    attempt.mass_orthogonality_certificate_status.clone(),
                    attempt.mass_orthogonality_certificate_reason.clone(),
                    attempt.generalized_residual_certificate_status.clone(),
                    attempt.generalized_residual_certificate_reason.clone(),
                    attempt.external_tool_versions_status.clone(),
                    attempt.external_tool_versions_reason.clone(),
                    option_string(attempt.sigma_dimension),
                    option_string(attempt.converged_rank),
                    option_string(attempt.retained_rank),
                    option_string(attempt.input_nonzeros),
                    option_string(attempt.l_nonzeros),
                    option_string(attempt.d_nonzeros),
                    option_string(attempt.c_nonzeros),
                    option_string(attempt.lower_mass_nonzeros),
                    option_string(attempt.state_mass_nonzeros),
                    option_string(attempt.basis_storage_entries),
                    option_string(attempt.basis_storage_bytes),
                    option_string(attempt.clamped_negative_eigenvalues),
                    attempt.eigenpair_selection.clone().unwrap_or_default(),
                    attempt.lower_mass_solve.clone().unwrap_or_default(),
                    attempt.workspace.clone().unwrap_or_default(),
                    attempt.binary.clone().unwrap_or_default(),
                ])
                .map_err(csv_error)?;
        }
        writer.flush()?;
        Ok(())
    }

    fn write_sensor_banks(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "refinement",
                "bank",
                "kernel_index",
                "center_x",
                "center_y",
                "center_z",
                "radius",
                "quadrature_rule",
                "quadrature_normalization",
                "component",
                "cochain_column",
                "cochain_weight",
                "component_row_nonzeros",
            ])
            .map_err(csv_error)?;
        for bank in &self.sensor_banks {
            for (index, kernel) in bank.kernels.iter().copied().enumerate() {
                let center = kernel.center();
                for component in 0..3 {
                    for &(column, weight) in &bank.component_rows[index][component] {
                        writer
                            .write_record([
                                bank.refinement.to_string(),
                                bank.bank.to_string(),
                                index.to_string(),
                                format_float(center[0]),
                                format_float(center[1]),
                                format_float(center[2]),
                                format_float(kernel.radius()),
                                bank.quadrature_rule.to_string(),
                                format_float(bank.quadrature_normalizations[index]),
                                component.to_string(),
                                column.to_string(),
                                format_float(weight),
                                bank.component_row_nonzeros[index][component].to_string(),
                            ])
                            .map_err(csv_error)?;
                    }
                }
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_sensor_overlaps(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record(["refinement", "bank", "left_kernel", "right_kernel"])
            .map_err(csv_error)?;
        for bank in &self.sensor_banks {
            for pair in &bank.overlapping_kernel_pairs {
                writer
                    .write_record([
                        bank.refinement.to_string(),
                        bank.bank.to_string(),
                        pair[0].to_string(),
                        pair[1].to_string(),
                    ])
                    .map_err(csv_error)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_manifest(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version: {C51_SPECTRAL_COVARIANCE_SCHEMA_VERSION}"
        )?;
        writeln!(writer, "study_id: {C51_SPECTRAL_COVARIANCE_STUDY_ID}")?;
        writeln!(writer, "profile: {}", self.config.profile)?;
        writeln!(writer, "evidence_tier: rank_discovery")?;
        writeln!(writer, "method_performance_timing_eligible: false")?;
        writeln!(writer, "shared_maximum_basis: true")?;
        writeln!(writer, "command: {}", yaml_string(&self.metadata.command))?;
        write_yaml_option(&mut writer, "root_sha", self.metadata.root_sha.as_deref())?;
        write_yaml_option(&mut writer, "feec_sha", self.metadata.feec_sha.as_deref())?;
        write_yaml_option(&mut writer, "gmrf_sha", self.metadata.gmrf_sha.as_deref())?;
        write_yaml_bool_option(&mut writer, "root_dirty", self.metadata.root_dirty)?;
        write_yaml_bool_option(&mut writer, "feec_dirty", self.metadata.feec_dirty)?;
        write_yaml_bool_option(&mut writer, "gmrf_dirty", self.metadata.gmrf_dirty)?;
        writeln!(
            writer,
            "refinements: {}",
            yaml_usizes(&self.config.refinements)
        )?;
        writeln!(writer, "ranks: {}", yaml_usizes(&self.config.ranks))?;
        write_yaml_usize_option(
            &mut writer,
            "requested_maximum_rank",
            self.config.requested_maximum_rank,
        )?;
        writeln!(writer, "thresholds: [0.10, 0.05, 0.02]")?;
        writeln!(writer, "alpha_values: [2, 3]")?;
        writeln!(writer, "kappa: {:.17e}", self.config.kappa)?;
        writeln!(writer, "tau: {:.17e}", self.config.tau)?;
        writeln!(
            writer,
            "observation_noise_variance: {:.17e}",
            self.config.observation_noise_variance
        )?;
        writeln!(writer, "qoi_kernel: fixed_footprint_wendland_c2")?;
        writeln!(
            writer,
            "qoi_kernel_radius: {:.17e}",
            self.config.qoi_kernel_radius
        )?;
        writeln!(
            writer,
            "observation_kernel_radius: {:.17e}",
            self.config.observation_kernel_radius
        )?;
        writeln!(
            writer,
            "observation_candidate_order: farthest_point_on_fixed_6x6x6_grid"
        )?;
        writeln!(
            writer,
            "observation_overlap_rule: overlap_permitted_and_archived"
        )?;
        writeln!(writer, "latent_diagnostic: relative_M_norm_error_of_C_M_p_for_two_fixed_M_normalized_primal_probes")?;
        writeln!(
            writer,
            "latent_diagnostic_cross_mesh_physical_output: false"
        )?;
        writeln!(
            writer,
            "maximum_basis_storage_bytes: {}",
            self.config.maximum_basis_storage_bytes
        )?;
        writeln!(writer, "resource_cap_scope: retained_basis_preflight_only")?;
        writeln!(writer, "external_process_caps_required: true")?;
        writeln!(writer, "full_rank_plumbing_gates:")?;
        for gate in &self.plumbing_gates {
            writeln!(writer, "  - refinement: {}", gate.refinement)?;
            writeln!(writer, "    evidence_tier: bounded_same_law_certificate")?;
            writeln!(writer, "    state_dimension: {}", gate.state_dimension)?;
            writeln!(writer, "    rank: {}", gate.rank)?;
            writeln!(
                writer,
                "    direct_supported_prior_primary_error: {:.17e}",
                gate.direct_supported_prior_primary_error
            )?;
            writeln!(
                writer,
                "    direct_supported_posterior_primary_error: {:.17e}",
                gate.direct_supported_posterior_primary_error
            )?;
            writeln!(
                writer,
                "    spectral_supported_prior_primary_error: {:.17e}",
                gate.spectral_supported_prior_primary_error
            )?;
            writeln!(
                writer,
                "    spectral_supported_posterior_primary_error: {:.17e}",
                gate.spectral_supported_posterior_primary_error
            )?;
            writeln!(writer, "    tolerance: {:.17e}", gate.tolerance)?;
        }
        writeln!(writer, "rows_csv: rows.csv")?;
        writeln!(writer, "covariances_csv: covariances.csv")?;
        writeln!(writer, "basis_attempts_csv: basis_attempts.csv")?;
        writeln!(writer, "sensor_banks_csv: sensor_banks.csv")?;
        writeln!(writer, "sensor_overlaps_csv: sensor_overlaps.csv")?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(feature = "external-solvers")]
pub fn run_c51_spectral_covariance_rank(
    config: &C51SpectralCovarianceRankConfig,
    metadata: C51SpectralCovarianceRankMetadata,
) -> Result<C51SpectralCovarianceRankArchive> {
    config.validate()?;
    let plumbing_gates = [2, 4]
        .into_iter()
        .map(|refinement| run_full_rank_plumbing_gate(config, refinement))
        .collect::<Result<Vec<_>>>()?;
    let mut archive = C51SpectralCovarianceRankArchive {
        metadata,
        config: config.clone(),
        basis_attempts: Vec::new(),
        rows: Vec::new(),
        covariances: Vec::new(),
        sensor_banks: Vec::new(),
        plumbing_gates,
    };
    for &refinement in &config.refinements {
        run_refinement(config, refinement, &mut archive)?;
    }
    archive.validate()?;
    Ok(archive)
}

#[cfg(not(feature = "external-solvers"))]
pub fn run_c51_spectral_covariance_rank(
    _config: &C51SpectralCovarianceRankConfig,
    _metadata: C51SpectralCovarianceRankMetadata,
) -> Result<C51SpectralCovarianceRankArchive> {
    Err(FeecGmrfError::Unsupported(
        "C51 spectral covariance-rank discovery requires external-solvers and PETSc/SLEPc"
            .to_string(),
    ))
}

#[cfg(feature = "external-solvers")]
fn run_full_rank_plumbing_gate(
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
) -> Result<C51FullRankPlumbingGate> {
    const TOLERANCE: f64 = 1.0e-8;
    if ![2, 4].contains(&refinement) {
        return Err(invalid("C51 full-rank plumbing refinement must be 2 or 4"));
    }
    let mesh = CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)?;
    let state_dimension = operators.state().dimension();
    let partial = partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
        &topology,
        &metric,
        1,
        PartialMixedGeneralizedEigensolverOptions::new(state_dimension),
    )
    .map_err(spectral_error)?;
    if partial.diagnostics().retained_rank != state_dimension {
        return Err(FeecGmrfError::Inference(format!(
            "C51 n={refinement} plumbing gate retained {} of {state_dimension} modes",
            partial.diagnostics().retained_rank
        )));
    }
    let physical = MagneticFieldMaps3d::from_feec(&topology, &coordinates)?;
    let output_full = LinearMap::stack(&physical.volume_average_components)?;
    let elimination =
        EssentialBoundaryElimination::from_layout(operators.state().boundary_layout().clone())?;
    let output_reduced = eliminate_homogeneous_map(&elimination, &output_full)?;
    let output_csr = to_csr(&output_reduced);
    let parameters = MaternParameters::new(MaternAlpha::Two, config.kappa, config.tau)?;
    let direct = ExactMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .maximum_materialized_dimension(state_dimension)?
        .build()?;
    let exact = SupportedMaternPriorBuilder::from_operators(operators)
        .parameters(parameters)
        .build()?;
    let exact_prior_covariance = exact.pushforward_covariance(&output_full)?;
    let mut direct_factor = direct.factor()?;
    let direct_prior_covariance = direct_factor.pushforward_covariance(&output_full)?;
    let direct_supported_prior_error =
        covariance_error_metrics(&direct_prior_covariance, &exact_prior_covariance)?
            .primary_envelope();
    let gp = SpectralMaternGp::from_eigenbasis(
        partial.into_basis(),
        SpectralMaternConfig {
            kappa: config.kappa,
            alpha: 2.0,
            tau: config.tau,
            k: state_dimension,
        },
    )
    .map_err(spectral_error)?;
    let spectral_prior_covariance = faer_to_rows(
        &gp.prior_output_covariance(&output_csr)
            .map_err(spectral_error)?,
    );
    let spectral_supported_prior_error =
        covariance_error_metrics(&spectral_prior_covariance, &exact_prior_covariance)?
            .primary_envelope();

    let observation = LinearObservation::new(
        output_full,
        vec![0.0; output_csr.nrows()],
        GaussianNoise::variance(config.observation_noise_variance)?,
    )?;
    let exact_posterior = exact.condition(std::slice::from_ref(&observation), &[])?;
    let mut direct_posterior = LinearGaussianModelBuilder::new(direct)
        .observe(observation)?
        .condition()?;
    let output_full = LinearMap::stack(&physical.volume_average_components)?;
    let exact_posterior_covariance = exact_posterior.pushforward_covariance(&output_full)?;
    let direct_posterior_covariance = direct_posterior.pushforward_covariance(&output_full)?;
    let direct_supported_posterior_error =
        covariance_error_metrics(&direct_posterior_covariance, &exact_posterior_covariance)?
            .primary_envelope();
    let prepared = gp
        .prepare_linear_observations(&output_csr, config.observation_noise_variance)
        .map_err(spectral_error)?;
    let spectral_posterior_covariance = faer_to_rows(
        &prepared
            .output_moments_with_covariance(&output_csr, &vec![0.0; output_csr.nrows()])
            .map_err(spectral_error)?
            .covariance,
    );
    let spectral_supported_posterior_error =
        covariance_error_metrics(&spectral_posterior_covariance, &exact_posterior_covariance)?
            .primary_envelope();
    if [
        direct_supported_prior_error,
        direct_supported_posterior_error,
        spectral_supported_prior_error,
        spectral_supported_posterior_error,
    ]
    .into_iter()
    .any(|error| error > TOLERANCE)
    {
        return Err(FeecGmrfError::Inference(format!(
            "C51 full-rank n={refinement} same-law plumbing gate failed: direct/support prior {direct_supported_prior_error:.6e}, direct/support posterior {direct_supported_posterior_error:.6e}, spectral/support prior {spectral_supported_prior_error:.6e}, spectral/support posterior {spectral_supported_posterior_error:.6e}, tolerance {TOLERANCE:.6e}"
        )));
    }
    Ok(C51FullRankPlumbingGate {
        refinement,
        state_dimension,
        rank: state_dimension,
        direct_supported_prior_primary_error: direct_supported_prior_error,
        direct_supported_posterior_primary_error: direct_supported_posterior_error,
        spectral_supported_prior_primary_error: spectral_supported_prior_error,
        spectral_supported_posterior_primary_error: spectral_supported_posterior_error,
        tolerance: TOLERANCE,
    })
}

#[cfg(feature = "external-solvers")]
fn run_refinement(
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    archive: &mut C51SpectralCovarianceRankArchive,
) -> Result<()> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)?;
    let state_dimension = operators.state().dimension();
    let elimination =
        EssentialBoundaryElimination::from_layout(operators.state().boundary_layout().clone())?;

    let selection = MagneticVolumeSensorBank3d::from_feec(
        &topology,
        &coordinates,
        &selection_kernels(config.qoi_kernel_radius)?,
    )?;
    let held_out = MagneticVolumeSensorBank3d::from_feec(
        &topology,
        &coordinates,
        &held_out_kernels(config.qoi_kernel_radius)?,
    )?;
    let candidates = MagneticVolumeSensorBank3d::from_feec(
        &topology,
        &coordinates,
        &observation_kernels(config.observation_kernel_radius)?,
    )?;
    archive
        .sensor_banks
        .push(sensor_bank_audit(refinement, "selection", &selection)?);
    archive
        .sensor_banks
        .push(sensor_bank_audit(refinement, "held_out", &held_out)?);
    archive.sensor_banks.push(sensor_bank_audit(
        refinement,
        "observation_candidates",
        &candidates,
    )?);
    let selection_full = selection.map().map().clone();
    let held_out_full = held_out.map().map().clone();
    if selection_full.output_dimension() != OUTPUT_DIMENSION
        || held_out_full.output_dimension() != OUTPUT_DIMENSION
    {
        return Err(FeecGmrfError::Dimension(
            "C51 selection and held-out banks must each have nine outputs".to_string(),
        ));
    }
    let selection_reduced = eliminate_homogeneous_map(&elimination, &selection_full)?;
    let held_out_reduced = eliminate_homogeneous_map(&elimination, &held_out_full)?;
    let selection_csr = to_csr(&selection_reduced);
    let held_out_csr = to_csr(&held_out_reduced);
    let designs = build_designs(state_dimension, candidates.map().map(), &elimination)?;

    let valid_ranks = config
        .ranks
        .iter()
        .copied()
        .filter(|rank| *rank <= state_dimension)
        .collect::<Vec<_>>();
    let Some(&requested_rank) = valid_ranks.last() else {
        archive.basis_attempts.push(C51SpectralBasisAttempt {
            refinement,
            state_dimension,
            requested_rank: 0,
            status: "censored".to_string(),
            reason: "every requested rank exceeds the active state dimension".to_string(),
            diagnostics_status: "unavailable".to_string(),
            diagnostics_unavailable_reason:
                "mixed eigensolver was not invoked because every rank was invalid".to_string(),
            mass_orthogonality_certificate_status: "unavailable".to_string(),
            mass_orthogonality_certificate_reason: "no eigenbasis was constructed".to_string(),
            generalized_residual_certificate_status: "unavailable".to_string(),
            generalized_residual_certificate_reason: "no eigenbasis was constructed".to_string(),
            external_tool_versions_status: "unavailable".to_string(),
            external_tool_versions_reason: "external helper was not invoked".to_string(),
            sigma_dimension: None,
            converged_rank: None,
            retained_rank: None,
            input_nonzeros: None,
            l_nonzeros: None,
            d_nonzeros: None,
            c_nonzeros: None,
            lower_mass_nonzeros: None,
            state_mass_nonzeros: None,
            basis_storage_entries: None,
            basis_storage_bytes: None,
            clamped_negative_eigenvalues: None,
            eigenpair_selection: None,
            lower_mass_solve: None,
            workspace: None,
            binary: None,
        });
        append_censored_rows(
            archive,
            config,
            refinement,
            state_dimension,
            "rank_validation",
            "every requested rank exceeds the active state dimension",
        );
        return Ok(());
    };
    let expected_basis_bytes = state_dimension
        .checked_mul(requested_rank)
        .and_then(|entries| entries.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| FeecGmrfError::Dimension("C51 basis storage overflow".to_string()))?;
    if expected_basis_bytes > config.maximum_basis_storage_bytes {
        let reason = format!(
            "requested basis requires {expected_basis_bytes} bytes, above the frozen {}-byte preflight cap",
            config.maximum_basis_storage_bytes
        );
        archive.basis_attempts.push(C51SpectralBasisAttempt {
            refinement,
            state_dimension,
            requested_rank,
            status: "censored".to_string(),
            reason: reason.clone(),
            diagnostics_status: "unavailable".to_string(),
            diagnostics_unavailable_reason:
                "mixed eigensolver was not invoked because the basis preflight cap failed"
                    .to_string(),
            mass_orthogonality_certificate_status: "unavailable".to_string(),
            mass_orthogonality_certificate_reason: "no eigenbasis was constructed".to_string(),
            generalized_residual_certificate_status: "unavailable".to_string(),
            generalized_residual_certificate_reason: "no eigenbasis was constructed".to_string(),
            external_tool_versions_status: "unavailable".to_string(),
            external_tool_versions_reason: "external helper was not invoked".to_string(),
            sigma_dimension: None,
            converged_rank: None,
            retained_rank: None,
            input_nonzeros: None,
            l_nonzeros: None,
            d_nonzeros: None,
            c_nonzeros: None,
            lower_mass_nonzeros: None,
            state_mass_nonzeros: None,
            basis_storage_entries: None,
            basis_storage_bytes: Some(expected_basis_bytes),
            clamped_negative_eigenvalues: None,
            eigenpair_selection: None,
            lower_mass_solve: None,
            workspace: None,
            binary: None,
        });
        append_censored_rows(
            archive,
            config,
            refinement,
            state_dimension,
            "basis_preflight",
            &reason,
        );
        return Ok(());
    }

    let partial =
        match partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
            &topology,
            &metric,
            1,
            PartialMixedGeneralizedEigensolverOptions::new(requested_rank),
        ) {
            Ok(partial) => partial,
            Err(error) => {
                let reason = error.to_string();
                archive.basis_attempts.push(C51SpectralBasisAttempt {
                    refinement,
                    state_dimension,
                    requested_rank,
                    status: "censored".to_string(),
                    reason: reason.clone(),
                    diagnostics_status: "unavailable".to_string(),
                    diagnostics_unavailable_reason:
                        "the mixed eigensolver failed before checked diagnostics were returned"
                            .to_string(),
                    mass_orthogonality_certificate_status: "unavailable".to_string(),
                    mass_orthogonality_certificate_reason: "no checked eigenbasis was returned"
                        .to_string(),
                    generalized_residual_certificate_status: "unavailable".to_string(),
                    generalized_residual_certificate_reason:
                        "mixed helper failure returned no residual certificate".to_string(),
                    external_tool_versions_status: "unavailable".to_string(),
                    external_tool_versions_reason:
                        "mixed helper failure returned binary context but no tool versions"
                            .to_string(),
                    sigma_dimension: None,
                    converged_rank: None,
                    retained_rank: None,
                    input_nonzeros: None,
                    l_nonzeros: None,
                    d_nonzeros: None,
                    c_nonzeros: None,
                    lower_mass_nonzeros: None,
                    state_mass_nonzeros: None,
                    basis_storage_entries: None,
                    basis_storage_bytes: Some(expected_basis_bytes),
                    clamped_negative_eigenvalues: None,
                    eigenpair_selection: None,
                    lower_mass_solve: None,
                    workspace: None,
                    binary: None,
                });
                append_censored_rows(
                    archive,
                    config,
                    refinement,
                    state_dimension,
                    "mixed_eigensolve",
                    &reason,
                );
                return Ok(());
            }
        };
    archive
        .basis_attempts
        .push(successful_basis_attempt(refinement, partial.diagnostics()));
    let basis = partial.into_basis();

    for &alpha in &config.alphas {
        let row_start = archive.rows.len();
        let covariance_start = archive.covariances.len();
        if let Err(error) = run_alpha(
            config,
            refinement,
            state_dimension,
            alpha,
            &operators,
            &basis,
            &selection_full,
            &held_out_full,
            &selection_csr,
            &held_out_csr,
            &designs,
            archive,
        ) {
            archive.rows.truncate(row_start);
            archive.covariances.truncate(covariance_start);
            append_censored_alpha_rows(
                archive,
                config,
                refinement,
                state_dimension,
                alpha,
                "exact_or_spectral_covariance",
                &error.to_string(),
            );
        }
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
#[allow(clippy::too_many_arguments)]
fn run_alpha(
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    operators: &ExactFormOperators,
    maximum_basis: &EigenBasis,
    selection_full: &LinearMap,
    held_out_full: &LinearMap,
    selection_csr: &CsrMatrix,
    held_out_csr: &CsrMatrix,
    designs: &[ObservationDesign],
    archive: &mut C51SpectralCovarianceRankArchive,
) -> Result<()> {
    let parameters = MaternParameters::new(alpha, config.kappa, config.tau)?;
    let latent_probes = latent_mass_probes(operators.state().mass())?;
    let latent_probe_rhs = latent_probes
        .iter()
        .map(|probe| mass_apply(operators.state().mass(), probe))
        .collect::<Result<Vec<_>>>()?;
    let exact_prior = SupportedMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .build()?;
    let prior_factor = exact_prior.factor()?;
    let mut exact_designs = vec![ExactDesignCovariance {
        design: C51CovarianceDesign::Prior,
        observation_count: 0,
        observation_reduced: None,
        selection: prior_factor.pushforward_covariance(selection_full)?,
        held_out: prior_factor.pushforward_covariance(held_out_full)?,
        latent_actions: latent_probe_rhs
            .iter()
            .map(|rhs| prior_factor.covariance_action(rhs))
            .collect::<Result<Vec<_>>>()?,
    }];
    for design in designs {
        let observation = LinearObservation::new(
            design.full.clone(),
            vec![0.0; design.count],
            GaussianNoise::variance(config.observation_noise_variance)?,
        )?;
        let posterior = exact_prior.condition(&[observation], &[])?;
        exact_designs.push(ExactDesignCovariance {
            design: design.kind,
            observation_count: design.count,
            observation_reduced: Some(to_csr(&design.reduced)),
            selection: posterior.pushforward_covariance(selection_full)?,
            held_out: posterior.pushforward_covariance(held_out_full)?,
            latent_actions: latent_probe_rhs
                .iter()
                .map(|rhs| posterior.covariance_action(rhs))
                .collect::<Result<Vec<_>>>()?,
        });
    }
    for exact in &exact_designs {
        validate_reference_covariance(&exact.selection, "selection")?;
        validate_reference_covariance(&exact.held_out, "held-out")?;
        archive_covariance(
            &mut archive.covariances,
            refinement,
            alpha,
            exact.design,
            None,
            "selection",
            "exact_supported",
            &exact.selection,
        );
        archive_covariance(
            &mut archive.covariances,
            refinement,
            alpha,
            exact.design,
            None,
            "held_out",
            "exact_supported",
            &exact.held_out,
        );
    }

    for &rank in &config.ranks {
        if rank > state_dimension || rank > maximum_basis.len() {
            append_censored_rank_rows(
                archive,
                refinement,
                state_dimension,
                alpha,
                rank,
                "rank_validation",
                "requested rank exceeds the available active eigenbasis",
            );
            continue;
        }
        let gp = match maximum_basis
            .truncate_smallest(rank)
            .map_err(spectral_error)
            .and_then(|rank_basis| {
                SpectralMaternGp::from_eigenbasis(
                    rank_basis,
                    SpectralMaternConfig {
                        kappa: config.kappa,
                        alpha: f64::from(alpha.as_u32()),
                        tau: config.tau,
                        k: rank,
                    },
                )
                .map_err(spectral_error)
            }) {
            Ok(gp) => gp,
            Err(error) => {
                append_censored_rank_rows(
                    archive,
                    refinement,
                    state_dimension,
                    alpha,
                    rank,
                    "spectral_rank_construction",
                    &error.to_string(),
                );
                continue;
            }
        };
        for exact in &exact_designs {
            let row_start = archive.rows.len();
            let covariance_start = archive.covariances.len();
            let evaluation = (|| -> Result<()> {
                let (
                    selection_covariance,
                    held_out_covariance,
                    approximate_latent_actions,
                    prepared_storage_bytes,
                ) = match &exact.observation_reduced {
                    None => (
                        faer_to_rows(
                            &gp.prior_output_covariance(selection_csr)
                                .map_err(spectral_error)?,
                        ),
                        faer_to_rows(
                            &gp.prior_output_covariance(held_out_csr)
                                .map_err(spectral_error)?,
                        ),
                        latent_probe_rhs
                            .iter()
                            .map(|rhs| gp.prior_covariance_action(rhs).map_err(spectral_error))
                            .collect::<Result<Vec<_>>>()?,
                        None,
                    ),
                    Some(observation) => {
                        let prepared = gp
                            .prepare_linear_observations(
                                observation,
                                config.observation_noise_variance,
                            )
                            .map_err(spectral_error)?;
                        let values = vec![0.0; exact.observation_count];
                        let selection = prepared
                            .output_moments_with_covariance(selection_csr, &values)
                            .map_err(spectral_error)?;
                        let held_out = prepared
                            .output_moments_with_covariance(held_out_csr, &values)
                            .map_err(spectral_error)?;
                        (
                            faer_to_rows(&selection.covariance),
                            faer_to_rows(&held_out.covariance),
                            latent_probe_rhs
                                .iter()
                                .map(|rhs| {
                                    prepared
                                        .state_covariance_action(rhs)
                                        .map_err(spectral_error)
                                })
                                .collect::<Result<Vec<_>>>()?,
                            Some(
                                prepared
                                    .diagnostics()
                                    .persistent_f64_entries
                                    .checked_mul(std::mem::size_of::<f64>())
                                    .ok_or_else(|| {
                                        FeecGmrfError::Dimension(
                                            "C51 prepared spectral storage overflow".to_string(),
                                        )
                                    })?,
                            ),
                        )
                    }
                };
                let selection_metrics =
                    covariance_error_metrics(&selection_covariance, &exact.selection)?;
                let held_out_metrics =
                    covariance_error_metrics(&held_out_covariance, &exact.held_out)?;
                let latent_mass_action_relative_errors = latent_action_errors(
                    operators.state().mass(),
                    &approximate_latent_actions,
                    &exact.latent_actions,
                )?;
                let latent_mass_action_maximum_relative_error = latent_mass_action_relative_errors
                    .into_iter()
                    .fold(0.0, f64::max);
                archive_covariance(
                    &mut archive.covariances,
                    refinement,
                    alpha,
                    exact.design,
                    Some(rank),
                    "selection",
                    "spectral",
                    &selection_covariance,
                );
                archive_covariance(
                    &mut archive.covariances,
                    refinement,
                    alpha,
                    exact.design,
                    Some(rank),
                    "held_out",
                    "spectral",
                    &held_out_covariance,
                );
                archive.rows.push(C51SpectralCovarianceRankRow {
                    refinement,
                    state_dimension,
                    alpha: alpha.as_u32(),
                    design: exact.design,
                    observation_count: exact.observation_count,
                    observation_density: exact.observation_count as f64 / state_dimension as f64,
                    rank,
                    shared_maximum_basis_rank: maximum_basis.len(),
                    rank_fraction: rank as f64 / state_dimension as f64,
                    status: "measured".to_string(),
                    censored_stage: String::new(),
                    censored_reason: String::new(),
                    selection: Some(selection_metrics),
                    held_out: Some(held_out_metrics),
                    latent_mass_action_relative_errors: Some(latent_mass_action_relative_errors),
                    latent_mass_action_maximum_relative_error: Some(
                        latent_mass_action_maximum_relative_error,
                    ),
                    basis_storage_bytes: state_dimension * rank * std::mem::size_of::<f64>(),
                    prepared_storage_bytes,
                    method_performance_timing_eligible: false,
                });
                Ok(())
            })();
            if let Err(error) = evaluation {
                archive.rows.truncate(row_start);
                archive.covariances.truncate(covariance_start);
                append_censored_design_row(
                    archive,
                    refinement,
                    state_dimension,
                    alpha,
                    rank,
                    exact.design,
                    exact.observation_count,
                    "spectral_design_covariance",
                    &error.to_string(),
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
#[derive(Clone)]
struct ObservationDesign {
    kind: C51CovarianceDesign,
    count: usize,
    full: LinearMap,
    reduced: LinearMap,
}

#[cfg(feature = "external-solvers")]
struct ExactDesignCovariance {
    design: C51CovarianceDesign,
    observation_count: usize,
    observation_reduced: Option<CsrMatrix>,
    selection: Vec<Vec<f64>>,
    held_out: Vec<Vec<f64>>,
    latent_actions: Vec<Vec<f64>>,
}

#[cfg(feature = "external-solvers")]
fn build_designs(
    state_dimension: usize,
    candidates: &LinearMap,
    elimination: &EssentialBoundaryElimination,
) -> Result<Vec<ObservationDesign>> {
    C51CovarianceDesign::ALL
        .into_iter()
        .filter(|kind| *kind != C51CovarianceDesign::Prior)
        .map(|kind| {
            let count = kind.observation_count(state_dimension);
            if count > candidates.output_dimension() {
                return Err(FeecGmrfError::Dimension(format!(
                    "C51 {} design requests {count} rows but only {} fixed candidates exist",
                    kind.as_str(),
                    candidates.output_dimension()
                )));
            }
            let full = candidates.select_outputs(&(0..count).collect::<Vec<_>>())?;
            let reduced = eliminate_homogeneous_map(elimination, &full)?;
            Ok(ObservationDesign {
                kind,
                count,
                full,
                reduced,
            })
        })
        .collect()
}

fn selection_kernels(radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    kernels_from_centers(
        &[[0.25, 0.25, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]],
        radius,
    )
}

fn held_out_kernels(radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    kernels_from_centers(
        &[[0.75, 0.75, 0.25], [0.25, 0.25, 0.75], [0.75, 0.25, 0.25]],
        radius,
    )
}

fn kernels_from_centers(centers: &[[f64; 3]], radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    centers
        .iter()
        .map(|center| WendlandC2Kernel3d::new(*center, radius))
        .collect()
}

fn sensor_bank_audit(
    refinement: usize,
    bank: &'static str,
    sensors: &MagneticVolumeSensorBank3d,
) -> Result<C51SensorBankAudit> {
    let expected_rows = 3 * sensors.kernels().len();
    if sensors.map().map().output_dimension() != expected_rows
        || sensors.quadrature_normalizations().len() != sensors.kernels().len()
    {
        return Err(FeecGmrfError::Dimension(format!(
            "C51 {bank} sensor audit has inconsistent kernel/output dimensions"
        )));
    }
    let mut rows = vec![Vec::<(usize, f64)>::new(); expected_rows];
    for (row, column, value) in sensors.map().map().matrix().triplet_iter() {
        rows[row].push((column, value));
    }
    for row in &mut rows {
        row.sort_by_key(|(column, _)| *column);
    }
    let component_rows = (0..sensors.kernels().len())
        .map(|kernel| {
            [
                rows[3 * kernel].clone(),
                rows[3 * kernel + 1].clone(),
                rows[3 * kernel + 2].clone(),
            ]
        })
        .collect::<Vec<_>>();
    let mut overlapping_kernel_pairs = Vec::new();
    for left in 0..sensors.kernels().len() {
        for right in (left + 1)..sensors.kernels().len() {
            let left_kernel = sensors.kernels()[left];
            let right_kernel = sensors.kernels()[right];
            if distance_squared(left_kernel.center(), right_kernel.center())
                < (left_kernel.radius() + right_kernel.radius()).powi(2)
            {
                overlapping_kernel_pairs.push([left, right]);
            }
        }
    }
    let mut row_nonzeros = vec![0_usize; expected_rows];
    for (row, _, _) in sensors.map().map().matrix().triplet_iter() {
        row_nonzeros[row] += 1;
    }
    Ok(C51SensorBankAudit {
        refinement,
        bank,
        kernels: sensors.kernels().to_vec(),
        quadrature_rule: "tetrahedron_cell_barycenter",
        quadrature_normalizations: sensors.quadrature_normalizations().to_vec(),
        component_row_nonzeros: row_nonzeros
            .chunks_exact(3)
            .map(|rows| [rows[0], rows[1], rows[2]])
            .collect(),
        component_rows,
        overlapping_kernel_pairs,
    })
}

fn observation_kernels(radius: f64) -> Result<Vec<WendlandC2Kernel3d>> {
    let count = OBSERVATION_GRID_POINTS_PER_AXIS;
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
        .ok_or_else(|| invalid("C51 observation candidate grid is empty"))?;
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
            .expect("nonempty candidate set has a farthest point");
        ordered.push(remaining.swap_remove(next));
    }
    kernels_from_centers(&ordered, radius)
}

fn distance_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

fn nearest_distance_squared(point: [f64; 3], selected: &[[f64; 3]]) -> f64 {
    selected
        .iter()
        .map(|selected| distance_squared(point, *selected))
        .fold(f64::INFINITY, f64::min)
}

fn eliminate_homogeneous_map(
    elimination: &EssentialBoundaryElimination,
    map: &LinearMap,
) -> Result<LinearMap> {
    elimination
        .eliminate_map(map, &vec![0.0; map.output_dimension()])
        .map(|eliminated| eliminated.reduced().clone())
}

fn to_csr(map: &LinearMap) -> CsrMatrix {
    feg_infer::sparse::core_triplet_to_feec_csr(map.matrix())
}

fn latent_mass_probes(mass: &SparseMat) -> Result<[Vec<f64>; 2]> {
    let dimension = mass.nrows();
    if dimension == 0 || mass.ncols() != dimension {
        return Err(FeecGmrfError::Dimension(
            "C51 latent mass probes require a nonempty square mass matrix".to_string(),
        ));
    }
    let first = (0..dimension)
        .map(|index| {
            let coordinate = index as f64 + 1.0;
            (coordinate * 2.0_f64.sqrt()).sin() + 0.25 * (coordinate * 3.0_f64.sqrt()).cos()
        })
        .collect::<Vec<_>>();
    let second = (0..dimension)
        .map(|index| {
            let coordinate = index as f64 + 1.0;
            (coordinate * 5.0_f64.sqrt()).cos() - 0.2 * (coordinate * 7.0_f64.sqrt()).sin()
        })
        .collect::<Vec<_>>();
    let first = normalize_mass_vector(mass, first)?;
    let projection = mass_inner_product(mass, &first, &second)?;
    let second = second
        .into_iter()
        .zip(&first)
        .map(|(value, first)| value - projection * first)
        .collect::<Vec<_>>();
    let second = normalize_mass_vector(mass, second)?;
    Ok([first, second])
}

fn normalize_mass_vector(mass: &SparseMat, mut vector: Vec<f64>) -> Result<Vec<f64>> {
    let norm = mass_inner_product(mass, &vector, &vector)?.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(FeecGmrfError::Inference(
            "C51 latent probe has non-positive FEEC mass norm".to_string(),
        ));
    }
    for value in &mut vector {
        *value /= norm;
    }
    Ok(vector)
}

fn mass_inner_product(mass: &SparseMat, left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != mass.nrows() || right.len() != mass.ncols() {
        return Err(FeecGmrfError::Dimension(
            "C51 latent mass-inner-product vector dimension mismatch".to_string(),
        ));
    }
    let weighted = mass_apply(mass, right)?;
    let value = left
        .iter()
        .zip(weighted)
        .map(|(left, weighted)| left * weighted)
        .sum::<f64>();
    if !value.is_finite() {
        return Err(FeecGmrfError::Inference(
            "C51 latent mass inner product is non-finite".to_string(),
        ));
    }
    Ok(value)
}

fn mass_apply(mass: &SparseMat, input: &[f64]) -> Result<Vec<f64>> {
    mass.apply_checked(input).map_err(FeecGmrfError::Dimension)
}

fn latent_action_errors(
    mass: &SparseMat,
    approximation: &[Vec<f64>],
    reference: &[Vec<f64>],
) -> Result<[f64; 2]> {
    if approximation.len() != 2 || reference.len() != 2 {
        return Err(FeecGmrfError::Dimension(
            "C51 latent covariance diagnostic requires exactly two paired actions".to_string(),
        ));
    }
    let mut errors = [0.0; 2];
    for index in 0..2 {
        if approximation[index].len() != reference[index].len() {
            return Err(FeecGmrfError::Dimension(
                "C51 paired latent covariance actions have different dimensions".to_string(),
            ));
        }
        let difference = approximation[index]
            .iter()
            .zip(&reference[index])
            .map(|(approximation, reference)| approximation - reference)
            .collect::<Vec<_>>();
        let numerator = mass_inner_product(mass, &difference, &difference)?
            .max(0.0)
            .sqrt();
        let denominator = mass_inner_product(mass, &reference[index], &reference[index])?
            .max(0.0)
            .sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(FeecGmrfError::Inference(format!(
                "C51 exact latent covariance action {index} has zero FEEC mass norm"
            )));
        }
        errors[index] = numerator / denominator;
    }
    Ok(errors)
}

fn validate_reference_covariance(covariance: &[Vec<f64>], bank: &str) -> Result<()> {
    validate_square_covariance(covariance)?;
    let maximum = (0..covariance.len())
        .map(|index| covariance[index][index])
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err(FeecGmrfError::Inference(format!(
            "C51 exact {bank} covariance has no positive finite diagonal"
        )));
    }
    for (index, row) in covariance.iter().enumerate() {
        let variance = row[index];
        if !variance.is_finite() || variance <= EXACT_VARIANCE_RELATIVE_FLOOR * maximum {
            return Err(FeecGmrfError::Inference(format!(
                "C51 exact {bank} variance {index} is materially near zero: {variance:.16e}"
            )));
        }
    }
    Ok(())
}

pub fn covariance_error_metrics(
    approximation: &[Vec<f64>],
    reference: &[Vec<f64>],
) -> Result<C51CovarianceErrorMetrics> {
    validate_square_covariance(approximation)?;
    validate_square_covariance(reference)?;
    if approximation.len() != reference.len() {
        return Err(FeecGmrfError::Dimension(
            "C51 covariance matrices have different dimensions".to_string(),
        ));
    }
    validate_reference_covariance(reference, "metric reference")?;
    let dimension = reference.len();
    let reference_matrix = rows_to_matrix(reference);
    let approximation_matrix = rows_to_matrix(approximation);
    let difference = &approximation_matrix - &reference_matrix;
    let reference_frobenius = reference_matrix.norm();
    let reference_spectral = spectral_norm(&reference_matrix);
    if reference_frobenius == 0.0 || reference_spectral == 0.0 {
        return Err(FeecGmrfError::Inference(
            "C51 covariance metric reference norm is zero".to_string(),
        ));
    }
    let maximum_relative_diagonal = (0..dimension)
        .map(|index| {
            (approximation[index][index] - reference[index][index]).abs() / reference[index][index]
        })
        .fold(0.0, f64::max);
    let maximum_relative_directional = covariance_directions(dimension)
        .into_iter()
        .map(|direction| {
            let reference_value = quadratic(reference, &direction);
            if !reference_value.is_finite() || reference_value <= 0.0 {
                return Err(FeecGmrfError::Inference(
                    "C51 directional reference covariance is non-positive".to_string(),
                ));
            }
            Ok((quadratic(approximation, &direction) - reference_value).abs() / reference_value)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(0.0, f64::max);
    let reference_correlation = correlation_matrix(reference)?;
    let approximation_correlation = correlation_matrix(approximation)?;
    let correlation_difference = &approximation_correlation - &reference_correlation;
    let correlation_norm = reference_correlation.norm();
    let metrics = C51CovarianceErrorMetrics {
        relative_frobenius: difference.norm() / reference_frobenius,
        relative_spectral: spectral_norm(&difference) / reference_spectral,
        maximum_relative_diagonal,
        maximum_relative_directional,
        relative_correlation_frobenius: correlation_difference.norm() / correlation_norm,
    };
    if [
        metrics.relative_frobenius,
        metrics.relative_spectral,
        metrics.maximum_relative_diagonal,
        metrics.maximum_relative_directional,
        metrics.relative_correlation_frobenius,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(FeecGmrfError::Inference(
            "C51 covariance metric is non-finite or negative".to_string(),
        ));
    }
    Ok(metrics)
}

fn validate_square_covariance(covariance: &[Vec<f64>]) -> Result<()> {
    if covariance.is_empty()
        || covariance
            .iter()
            .any(|row| row.len() != covariance.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(FeecGmrfError::Dimension(
            "C51 covariance must be nonempty, square, and finite".to_string(),
        ));
    }
    Ok(())
}

fn covariance_directions(dimension: usize) -> Vec<Vec<f64>> {
    let mut directions = (0..dimension)
        .map(|index| {
            let mut direction = vec![0.0; dimension];
            direction[index] = 1.0;
            direction
        })
        .collect::<Vec<_>>();
    if dimension == OUTPUT_DIMENSION {
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            for sign in [-1.0, 1.0] {
                let mut direction = vec![0.0; dimension];
                for kernel in 0..3 {
                    direction[3 * kernel + left] = 1.0;
                    direction[3 * kernel + right] = sign;
                }
                let norm = direction
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                for value in &mut direction {
                    *value /= norm;
                }
                directions.push(direction);
            }
        }
    }
    directions
}

fn quadratic(matrix: &[Vec<f64>], direction: &[f64]) -> f64 {
    direction
        .iter()
        .enumerate()
        .map(|(row, left)| {
            direction
                .iter()
                .enumerate()
                .map(|(column, right)| left * matrix[row][column] * right)
                .sum::<f64>()
        })
        .sum()
}

fn correlation_matrix(covariance: &[Vec<f64>]) -> Result<Matrix> {
    let dimension = covariance.len();
    let mut correlation = Matrix::zeros(dimension, dimension);
    for row in 0..dimension {
        if covariance[row][row] <= 0.0 {
            return Err(FeecGmrfError::Inference(format!(
                "C51 covariance diagonal {row} is non-positive"
            )));
        }
        for column in 0..dimension {
            if covariance[column][column] <= 0.0 {
                return Err(FeecGmrfError::Inference(format!(
                    "C51 covariance diagonal {column} is non-positive"
                )));
            }
            correlation[(row, column)] = covariance[row][column]
                / (covariance[row][row] * covariance[column][column]).sqrt();
        }
    }
    Ok(correlation)
}

fn rows_to_matrix(rows: &[Vec<f64>]) -> Matrix {
    Matrix::from_fn(rows.len(), rows.len(), |row, column| rows[row][column])
}

fn spectral_norm(matrix: &Matrix) -> f64 {
    matrix
        .clone()
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .fold(0.0, f64::max)
}

#[cfg(feature = "external-solvers")]
fn faer_to_rows(matrix: &faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}

#[cfg(feature = "external-solvers")]
fn successful_basis_attempt(
    refinement: usize,
    diagnostics: &PartialMixedGeneralizedEigenbasisDiagnostics,
) -> C51SpectralBasisAttempt {
    C51SpectralBasisAttempt {
        refinement,
        state_dimension: diagnostics.u_dimension,
        requested_rank: diagnostics.requested_rank,
        status: "measured".to_string(),
        reason: String::new(),
        diagnostics_status: "measured".to_string(),
        diagnostics_unavailable_reason: String::new(),
        mass_orthogonality_certificate_status: "validated_defect_not_archived".to_string(),
        mass_orthogonality_certificate_reason:
            "canonical EigenBasis construction enforces the requested tolerance but does not expose the measured maximum defect"
                .to_string(),
        generalized_residual_certificate_status: "unavailable".to_string(),
        generalized_residual_certificate_reason:
            "the mixed PETSc/SLEPc adapter does not return generalized eigen-residual norms"
                .to_string(),
        external_tool_versions_status: "unavailable".to_string(),
        external_tool_versions_reason:
            "the adapter archives helper binary/workspace context but does not return PETSc/SLEPc version strings"
                .to_string(),
        sigma_dimension: Some(diagnostics.sigma_dimension),
        converged_rank: Some(diagnostics.converged_rank),
        retained_rank: Some(diagnostics.retained_rank),
        input_nonzeros: Some(diagnostics.input_nonzeros),
        l_nonzeros: Some(diagnostics.l_nonzeros),
        d_nonzeros: Some(diagnostics.d_nonzeros),
        c_nonzeros: Some(diagnostics.c_nonzeros),
        lower_mass_nonzeros: Some(diagnostics.mkm1_nonzeros),
        state_mass_nonzeros: Some(diagnostics.mk_nonzeros),
        basis_storage_entries: Some(diagnostics.basis_storage_entries),
        basis_storage_bytes: Some(diagnostics.basis_storage_bytes),
        clamped_negative_eigenvalues: Some(diagnostics.clamped_negative_eigenvalues),
        eigenpair_selection: Some(format!("{:?}", diagnostics.which)),
        lower_mass_solve: Some(format!("{:?}", diagnostics.mass_solve)),
        workspace: Some(diagnostics.workspace.display().to_string()),
        binary: Some(diagnostics.binary.display().to_string()),
    }
}

fn append_censored_rows(
    archive: &mut C51SpectralCovarianceRankArchive,
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    stage: &str,
    reason: &str,
) {
    for &alpha in &config.alphas {
        append_censored_alpha_rows(
            archive,
            config,
            refinement,
            state_dimension,
            alpha,
            stage,
            reason,
        );
    }
}

fn append_censored_alpha_rows(
    archive: &mut C51SpectralCovarianceRankArchive,
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    stage: &str,
    reason: &str,
) {
    for &rank in &config.ranks {
        append_censored_rank_rows(
            archive,
            refinement,
            state_dimension,
            alpha,
            rank,
            stage,
            reason,
        );
    }
}

fn append_censored_rank_rows(
    archive: &mut C51SpectralCovarianceRankArchive,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    rank: usize,
    stage: &str,
    reason: &str,
) {
    for design in C51CovarianceDesign::ALL {
        let observation_count = design.observation_count(state_dimension);
        append_censored_design_row(
            archive,
            refinement,
            state_dimension,
            alpha,
            rank,
            design,
            observation_count,
            stage,
            reason,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn append_censored_design_row(
    archive: &mut C51SpectralCovarianceRankArchive,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    rank: usize,
    design: C51CovarianceDesign,
    observation_count: usize,
    stage: &str,
    reason: &str,
) {
    let shared_maximum_basis_rank = archive
        .basis_attempts
        .iter()
        .rev()
        .find(|attempt| attempt.refinement == refinement)
        .map_or(0, |attempt| attempt.requested_rank);
    archive.rows.push(C51SpectralCovarianceRankRow {
        refinement,
        state_dimension,
        alpha: alpha.as_u32(),
        design,
        observation_count,
        observation_density: observation_count as f64 / state_dimension as f64,
        rank,
        shared_maximum_basis_rank,
        rank_fraction: rank as f64 / state_dimension as f64,
        status: "censored".to_string(),
        censored_stage: stage.to_string(),
        censored_reason: reason.to_string(),
        selection: None,
        held_out: None,
        latent_mass_action_relative_errors: None,
        latent_mass_action_maximum_relative_error: None,
        basis_storage_bytes: state_dimension.saturating_mul(rank).saturating_mul(8),
        prepared_storage_bytes: None,
        method_performance_timing_eligible: false,
    });
}

#[cfg(feature = "external-solvers")]
#[allow(clippy::too_many_arguments)]
fn archive_covariance(
    output: &mut Vec<C51OutputCovarianceRecord>,
    refinement: usize,
    alpha: MaternAlpha,
    design: C51CovarianceDesign,
    rank: Option<usize>,
    bank: &'static str,
    representation: &'static str,
    covariance: &[Vec<f64>],
) {
    for (row, values) in covariance.iter().enumerate() {
        for (column, value) in values.iter().copied().enumerate() {
            output.push(C51OutputCovarianceRecord {
                refinement,
                alpha: alpha.as_u32(),
                design,
                rank,
                bank,
                representation,
                row,
                column,
                value,
            });
        }
    }
}

fn append_metrics(output: &mut Vec<String>, metrics: Option<C51CovarianceErrorMetrics>) {
    match metrics {
        Some(metrics) => output.extend([
            format_float(metrics.relative_frobenius),
            format_float(metrics.relative_spectral),
            format_float(metrics.maximum_relative_diagonal),
            format_float(metrics.maximum_relative_directional),
            format_float(metrics.relative_correlation_frobenius),
            format_float(metrics.primary_envelope()),
        ]),
        None => output.extend((0..6).map(|_| String::new())),
    }
}

fn format_float(value: f64) -> String {
    format!("{value:.17e}")
}

fn option_string<T: Display>(value: Option<T>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn yaml_string(value: &str) -> String {
    format!("{:?}", value)
}

fn yaml_usizes(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn write_yaml_option(
    writer: &mut impl Write,
    key: &str,
    value: Option<&str>,
) -> std::io::Result<()> {
    match value {
        Some(value) => writeln!(writer, "{key}: {}", yaml_string(value)),
        None => writeln!(writer, "{key}: null"),
    }
}

fn write_yaml_bool_option(
    writer: &mut impl Write,
    key: &str,
    value: Option<bool>,
) -> std::io::Result<()> {
    match value {
        Some(value) => writeln!(writer, "{key}: {value}"),
        None => writeln!(writer, "{key}: null"),
    }
}

fn write_yaml_usize_option(
    writer: &mut impl Write,
    key: &str,
    value: Option<usize>,
) -> std::io::Result<()> {
    match value {
        Some(value) => writeln!(writer, "{key}: {value}"),
        None => writeln!(writer, "{key}: null"),
    }
}

fn spectral_error(error: impl Display) -> FeecGmrfError {
    FeecGmrfError::Inference(format!("C51 spectral covariance error: {error}"))
}

fn csv_error(error: csv::Error) -> FeecGmrfError {
    FeecGmrfError::Io(std::io::Error::other(error))
}

fn git_output(repository: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()?;
    if !output.status.success() {
        return Err(FeecGmrfError::Inference(format!(
            "git -C {} {} failed with status {}: {}",
            repository.display(),
            arguments.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| {
            FeecGmrfError::Inference(format!(
                "git output for {} was not UTF-8: {error}",
                repository.display()
            ))
        })
}

fn invalid(message: impl Into<String>) -> FeecGmrfError {
    FeecGmrfError::InvalidParameter(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_profile_freezes_mandatory_ranks_and_regularities() {
        let config = C51SpectralCovarianceRankConfig::publication();
        config.validate().unwrap();
        assert!(config.ranks.contains(&2_000));
        assert!(config.ranks.contains(&2_048));
        assert_eq!(config.alphas, [MaternAlpha::Two, MaternAlpha::Three]);
        assert_eq!(C51_COVARIANCE_THRESHOLDS, [0.10, 0.05, 0.02]);
    }

    #[test]
    fn publication_staging_filters_only_frozen_ranks_and_one_mesh() {
        let config = C51SpectralCovarianceRankConfig::publication()
            .for_publication_refinement(8)
            .unwrap()
            .with_maximum_rank(1_024)
            .unwrap();
        assert_eq!(config.refinements, [8]);
        assert_eq!(config.ranks.last(), Some(&1_024));
        assert_eq!(config.requested_maximum_rank, Some(1_024));
        assert!(config
            .ranks
            .iter()
            .all(|rank| C51_SPECTRAL_RANK_GRID.contains(rank)));
        assert!(C51SpectralCovarianceRankConfig::publication()
            .with_maximum_rank(1_000)
            .is_err());
        assert!(C51SpectralCovarianceRankConfig::publication()
            .for_publication_refinement(6)
            .is_err());
    }

    #[test]
    fn covariance_metrics_match_identity_and_detect_scale_error() {
        let reference = (0..OUTPUT_DIMENSION)
            .map(|row| {
                (0..OUTPUT_DIMENSION)
                    .map(|column| if row == column { 2.0 } else { 0.1 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let exact = covariance_error_metrics(&reference, &reference).unwrap();
        assert!(exact.primary_envelope() < 1.0e-14);
        let scaled = reference
            .iter()
            .map(|row| row.iter().map(|value| 1.1 * value).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let metrics = covariance_error_metrics(&scaled, &reference).unwrap();
        assert!((metrics.relative_frobenius - 0.1).abs() < 1.0e-12);
        assert!((metrics.maximum_relative_diagonal - 0.1).abs() < 1.0e-12);
        assert!((metrics.maximum_relative_directional - 0.1).abs() < 1.0e-12);
        assert!(metrics.relative_correlation_frobenius < 1.0e-14);
    }

    #[test]
    fn observation_candidates_are_fixed_stratified_and_sufficient_for_n10() {
        let kernels = observation_kernels(0.16).unwrap();
        assert_eq!(kernels.len() * 3, OBSERVATION_CANDIDATE_COUNT);
        assert!(C51CovarianceDesign::Density100.observation_count(6_130) <= kernels.len() * 3);
        let centers = kernels
            .iter()
            .map(|kernel| kernel.center())
            .collect::<Vec<_>>();
        assert!(centers[0]
            .iter()
            .all(|coordinate| (0.16..=0.84).contains(coordinate)));
        assert!(distance_squared(centers[0], [0.5; 3]) < 0.02);
        assert!(distance_squared(centers[0], centers[1]) > 0.2);
    }

    #[test]
    fn censored_rows_remain_typed_and_timing_ineligible() {
        let config = C51SpectralCovarianceRankConfig::smoke();
        let mut archive = C51SpectralCovarianceRankArchive {
            metadata: C51SpectralCovarianceRankMetadata::discovery("test"),
            config: config.clone(),
            basis_attempts: vec![C51SpectralBasisAttempt {
                refinement: 4,
                state_dimension: 100,
                requested_rank: 32,
                status: "censored".to_string(),
                reason: "solver unavailable".to_string(),
                diagnostics_status: "unavailable".to_string(),
                diagnostics_unavailable_reason: "solver unavailable".to_string(),
                mass_orthogonality_certificate_status: "unavailable".to_string(),
                mass_orthogonality_certificate_reason: "solver unavailable".to_string(),
                generalized_residual_certificate_status: "unavailable".to_string(),
                generalized_residual_certificate_reason: "solver unavailable".to_string(),
                external_tool_versions_status: "unavailable".to_string(),
                external_tool_versions_reason: "solver unavailable".to_string(),
                sigma_dimension: None,
                converged_rank: None,
                retained_rank: None,
                input_nonzeros: None,
                l_nonzeros: None,
                d_nonzeros: None,
                c_nonzeros: None,
                lower_mass_nonzeros: None,
                state_mass_nonzeros: None,
                basis_storage_entries: None,
                basis_storage_bytes: None,
                clamped_negative_eigenvalues: None,
                eigenpair_selection: None,
                lower_mass_solve: None,
                workspace: None,
                binary: None,
            }],
            rows: Vec::new(),
            covariances: Vec::new(),
            sensor_banks: ["selection", "held_out", "observation_candidates"]
                .into_iter()
                .map(|bank| C51SensorBankAudit {
                    refinement: 4,
                    bank,
                    kernels: vec![WendlandC2Kernel3d::new([0.5; 3], 0.1).unwrap()],
                    quadrature_rule: "tetrahedron_cell_barycenter",
                    quadrature_normalizations: vec![1.0],
                    component_row_nonzeros: vec![[1; 3]],
                    component_rows: vec![[vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]],
                    overlapping_kernel_pairs: Vec::new(),
                })
                .collect(),
            plumbing_gates: [2, 4]
                .into_iter()
                .map(|refinement| C51FullRankPlumbingGate {
                    refinement,
                    state_dimension: 10,
                    rank: 10,
                    direct_supported_prior_primary_error: 1.0e-12,
                    direct_supported_posterior_primary_error: 1.0e-12,
                    spectral_supported_prior_primary_error: 1.0e-12,
                    spectral_supported_posterior_primary_error: 1.0e-12,
                    tolerance: 1.0e-8,
                })
                .collect(),
        };
        append_censored_rows(
            &mut archive,
            &config,
            4,
            100,
            "mixed_eigensolve",
            "solver unavailable",
        );
        archive.validate().unwrap();
        assert!(archive.rows.iter().all(|row| {
            row.status == "censored"
                && !row.method_performance_timing_eligible
                && row.selection.is_none()
        }));
    }

    #[test]
    fn sensor_weights_and_overlaps_are_archived_without_cross_product_duplication() {
        let directory =
            std::env::temp_dir().join(format!("c51-sensor-archive-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let archive = C51SpectralCovarianceRankArchive {
            metadata: C51SpectralCovarianceRankMetadata::discovery("test"),
            config: C51SpectralCovarianceRankConfig::smoke(),
            basis_attempts: Vec::new(),
            rows: Vec::new(),
            covariances: Vec::new(),
            sensor_banks: vec![C51SensorBankAudit {
                refinement: 4,
                bank: "observation_candidates",
                kernels: vec![
                    WendlandC2Kernel3d::new([0.4; 3], 0.2).unwrap(),
                    WendlandC2Kernel3d::new([0.6; 3], 0.2).unwrap(),
                ],
                quadrature_rule: "tetrahedron_cell_barycenter",
                quadrature_normalizations: vec![1.0, 1.0],
                component_row_nonzeros: vec![[1; 3], [1; 3]],
                component_rows: vec![
                    [vec![(0, 1.0)], vec![(1, 2.0)], vec![(2, 3.0)]],
                    [vec![(3, 4.0)], vec![(4, 5.0)], vec![(5, 6.0)]],
                ],
                overlapping_kernel_pairs: vec![[0, 1]],
            }],
            plumbing_gates: Vec::new(),
        };
        let weights_path = directory.join("sensor_banks.csv");
        let overlaps_path = directory.join("sensor_overlaps.csv");
        archive.write_sensor_banks(&weights_path).unwrap();
        archive.write_sensor_overlaps(&overlaps_path).unwrap();

        let weights = std::fs::read_to_string(weights_path).unwrap();
        let overlaps = std::fs::read_to_string(overlaps_path).unwrap();
        assert_eq!(weights.lines().count(), 7);
        assert!(!weights.contains("overlapping_kernel_pairs"));
        assert_eq!(overlaps.lines().count(), 2);
        assert!(overlaps.contains("4,observation_candidates,0,1"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
