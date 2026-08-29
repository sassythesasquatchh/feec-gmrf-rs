//! C51 spectral covariance-rank discovery against the exact supported FEEC law.
//!
//! This study deliberately separates accuracy discovery from method timing.
//! One checked maximum mixed eigenbasis is shared across nested ranks, so every
//! row is permanently ineligible for a performance comparison.

#[cfg(any(feature = "external-solvers", test))]
use crate::c51_magnetic_sensor_design::primary_observation_kernels;
#[cfg(feature = "external-solvers")]
use crate::c51_magnetic_sensor_design::{
    assemble_juq_performance_observation_prefix, held_out_kernels,
    juq_performance_observation_kernel_prefix, selection_kernels,
};
use crate::c51_magnetic_sensor_design::{
    C51ScalarObservationPrefixMetadata, C51_FIXED_OBSERVATION_RADIUS,
    C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS, C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT,
    C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID, C51_PRIMARY_SCALAR_OBSERVATION_COUNT,
};
#[cfg(any(feature = "external-solvers", test))]
use common::linalg::nalgebra::CsrMatrix;
use common::linalg::nalgebra::Matrix;
#[cfg(feature = "external-solvers")]
use common::linalg::petsc::{petsc_read_eigenvals, petsc_read_eigenvecs};
#[cfg(feature = "external-solvers")]
use feec_gmrf::prelude::{
    calibrate_prior_to_physical_rms, AuditedSupportedCovariance, EssentialBoundaryConditions,
    ExactFormOperators, ExactMaternPriorBuilder, LinearGaussianModelBuilder, LinearObservation,
    MagneticFieldMaps3d, MaternPriorBuilder, SupportedCovarianceSymmetryAudit,
    SupportedMaternPriorBuilder,
};
#[cfg(any(feature = "external-solvers", test))]
use feec_gmrf::prelude::{
    EssentialBoundaryElimination, LinearMap, MagneticVolumeSensorBank3d, SparseMat,
};
use feec_gmrf::prelude::{
    FeecGmrfError, GaussianNoise, MaternAlpha, MaternParameters, Result, WendlandC2Kernel3d,
    INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION,
    INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT,
};
#[cfg(feature = "external-solvers")]
use feec_gmrf::spectral::{
    partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative,
    PartialMixedGeneralizedEigenbasisDiagnostics, PartialMixedGeneralizedEigensolverOptions,
};
#[cfg(feature = "external-solvers")]
use feg_gp::{
    checked_imported_generalized_hodge_eigenbasis, EigenBasis, SpectralMaternConfig,
    SpectralMaternGp,
};
#[cfg(feature = "external-solvers")]
use manifold::gen::cartesian::CartesianMeshInfo;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;

pub const C51_SPECTRAL_COVARIANCE_SCHEMA_VERSION: &str = "c51-spectral-covariance-rank-v3";
pub const C51_SPECTRAL_COVARIANCE_STUDY_ID: &str = "c51/spectral-covariance-rank";
pub const C51_SPECTRAL_RANK_GRID: [usize; 12] = [
    32, 64, 128, 256, 512, 768, 1_024, 1_280, 1_536, 1_792, 2_000, 2_048,
];
/// Frozen post-primary alpha-2 rank grid for the `n=8` extension.
pub const C51_SPECTRAL_RANK_EXTENSION_N8_GRID: [usize; 6] =
    [2_048, 2_304, 2_560, 2_816, 3_000, 3_032];
/// Frozen post-primary alpha-2 rank grid for the `n=10` extension.
pub const C51_SPECTRAL_RANK_EXTENSION_N10_GRID: [usize; 5] = [2_048, 2_560, 3_072, 3_584, 4_096];
pub const C51_COVARIANCE_THRESHOLDS: [f64; 3] = [0.10, 0.05, 0.02];
/// Frozen scalar-observation prefixes for the pre-performance accuracy ladder.
pub const C51_HIGH_OBSERVATION_COUNTS: [usize; 7] = [32, 128, 512, 1_024, 2_048, 4_096, 5_184];
pub const C51_HIGH_OBSERVATION_RANKS_N8: [usize; 13] = [
    512, 768, 1_024, 1_280, 1_536, 1_792, 2_000, 2_048, 2_304, 2_560, 2_816, 3_000, 3_032,
];
pub const C51_HIGH_OBSERVATION_RANKS_N10: [usize; 17] = [
    512, 768, 1_024, 1_280, 1_536, 1_792, 2_000, 2_048, 2_560, 3_072, 3_584, 4_096, 4_608, 5_120,
    5_632, 6_000, 6_130,
];
/// Exploratory replay grids backed by already generated, revalidated modes.
/// They select the next proof-grade rank solve but cannot emit a performance
/// rank decision themselves.
pub const C51_HIGH_OBSERVATION_REPLAY_RANKS_N8: [usize; 13] = C51_HIGH_OBSERVATION_RANKS_N8;
pub const C51_HIGH_OBSERVATION_REPLAY_RANKS_N10: [usize; 12] = [
    512, 768, 1_024, 1_280, 1_536, 1_792, 2_000, 2_048, 2_560, 3_072, 3_584, 4_096,
];

/// Immutable CLI/archive profile for the `n=8` post-primary extension.
pub const C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8: &str = "rank-extension-v1-n8";
/// Immutable CLI/archive profile for the `n=10` post-primary extension.
pub const C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10: &str = "rank-extension-v1-n10";
/// Frozen explanation for why the extension follows, but does not mutate, the primary study.
pub const C51_SPECTRAL_RANK_EXTENSION_RATIONALE: &str = "post-primary extension frozen after publication-discovery-v1 left corrected alpha=2 5%/2% decisions unresolved; it may resolve registered thresholds but may not support method timing";
pub const C51_HIGH_OBSERVATION_PROFILE_N8: &str = "high-observation-accuracy-v1-n8";
pub const C51_HIGH_OBSERVATION_PROFILE_N10: &str = "high-observation-accuracy-v1-n10";
pub const C51_HIGH_OBSERVATION_RATIONALE: &str = "pre-performance accuracy ladder over fixed dimension-independent observation counts; shared spectral basis and exact references are permanently timing ineligible";
pub const C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8: &str = "high-observation-primary-m2048-v1-n8";
pub const C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10: &str = "high-observation-primary-m2048-v1-n10";
pub const C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8: &str =
    "high-observation-m2048-replay-v1-n8-r3032";
pub const C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10: &str =
    "high-observation-m2048-replay-v1-n10-r4096";
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8: &str =
    "high-observation-m4096-replay-v1-n8-r3032";
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10: &str =
    "high-observation-m4096-replay-v1-n10-r4096";
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10: &str =
    "high-observation-m4096-rank-extension-v1-n10-r4608";
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_N10_GRID: [usize; 2] = [4_096, 4_608];
pub const C51_HIGH_OBSERVATION_PRIMARY_COUNT: usize = 2_048;
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8: &str =
    "high-observation-confirmatory-m4096-v1-n8";
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10: &str =
    "high-observation-confirmatory-m4096-v1-n10";
pub const C51_HIGH_OBSERVATION_CONFIRMATORY_COUNT: usize = 4_096;
pub const C51_PERFORMANCE_OUTPUT_DESIGN_ID: &str =
    "rank_study_selection_and_held_out_wendland_c2_18";
pub const C51_PERFORMANCE_RANK_DECISION_SCHEMA_VERSION: &str = "c51-performance-rank-decisions-v2";
const C51_PERFORMANCE_PRIMARY_TOLERANCE: f64 = 0.05;
const C51_ACCURACY_ARCHIVE_HASH_FILES: [&str; 10] = [
    "basis_attempts.csv",
    "compact_accuracy.csv",
    "covariances.csv",
    "exact_covariance_audits.csv",
    "high_observation_kernels.csv",
    "high_observation_prefixes.csv",
    "manifest.yaml",
    "rows.csv",
    "sensor_banks.csv",
    "sensor_overlaps.csv",
];
const C51_PERF0_ARCHIVE_HASH_FILES: [&str; 2] = ["manifest.yaml", "rows.csv"];

const OUTPUT_DIMENSION: usize = 9;
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

    /// Frozen post-primary alpha=2 extension. This is a separate evidence tier
    /// and never mutates the primary publication-discovery-v1 rank grid.
    pub fn rank_extension(refinement: usize) -> Result<Self> {
        let (profile, ranks) = match refinement {
            8 => (
                C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8,
                C51_SPECTRAL_RANK_EXTENSION_N8_GRID.as_slice(),
            ),
            10 => (
                C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10,
                C51_SPECTRAL_RANK_EXTENSION_N10_GRID.as_slice(),
            ),
            _ => {
                return Err(invalid(
                    "C51 rank extension refinement must be exactly 8 or 10",
                ))
            }
        };
        let config = Self {
            profile: profile.to_string(),
            refinements: vec![refinement],
            ranks: ranks.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    /// Frozen high-observation accuracy ladder evaluated before any timing run.
    pub fn high_observation_accuracy(refinement: usize) -> Result<Self> {
        let (profile, ranks) = match refinement {
            8 => (
                C51_HIGH_OBSERVATION_PROFILE_N8,
                C51_HIGH_OBSERVATION_RANKS_N8.as_slice(),
            ),
            10 => (
                C51_HIGH_OBSERVATION_PROFILE_N10,
                C51_HIGH_OBSERVATION_RANKS_N10.as_slice(),
            ),
            _ => {
                return Err(invalid(
                    "C51 high-observation accuracy refinement must be exactly 8 or 10",
                ))
            }
        };
        let config = Self {
            profile: profile.to_string(),
            refinements: vec![refinement],
            ranks: ranks.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            observation_kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    /// Claim-bearing `m=2048` accuracy gate required before primary timing cells.
    pub fn high_observation_primary(refinement: usize) -> Result<Self> {
        let (profile, ranks) = match refinement {
            8 => (
                C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8,
                C51_HIGH_OBSERVATION_RANKS_N8.as_slice(),
            ),
            10 => (
                C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10,
                C51_HIGH_OBSERVATION_RANKS_N10.as_slice(),
            ),
            _ => {
                return Err(invalid(
                    "C51 primary high-observation refinement must be 8 or 10",
                ))
            }
        };
        let config = Self {
            profile: profile.to_string(),
            refinements: vec![refinement],
            ranks: ranks.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            observation_kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    /// Accuracy-only replay of a previously generated basis. The caller must
    /// supply the PETSc workspace, whose eigenpairs are re-normalized and
    /// checked against the currently assembled mass matrix. Replay results are
    /// exploratory and cannot qualify a performance rank.
    pub fn high_observation_replay(refinement: usize) -> Result<Self> {
        let (profile, ranks) = match refinement {
            8 => (
                C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8,
                C51_HIGH_OBSERVATION_REPLAY_RANKS_N8.as_slice(),
            ),
            10 => (
                C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10,
                C51_HIGH_OBSERVATION_REPLAY_RANKS_N10.as_slice(),
            ),
            _ => {
                return Err(invalid(
                    "C51 high-observation replay refinement must be 8 or 10",
                ))
            }
        };
        let config = Self {
            profile: profile.to_string(),
            refinements: vec![refinement],
            ranks: ranks.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            observation_kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    /// Accuracy-only `m=4096` replay of a previously generated basis. This
    /// selects a fresh exact-request confirmatory rank without attributing the
    /// cached eigensolve time or emitting a performance rank decision.
    pub fn high_observation_confirmatory_replay(refinement: usize) -> Result<Self> {
        let (profile, ranks) = match refinement {
            8 => (
                C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8,
                C51_HIGH_OBSERVATION_REPLAY_RANKS_N8.as_slice(),
            ),
            10 => (
                C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10,
                C51_HIGH_OBSERVATION_REPLAY_RANKS_N10.as_slice(),
            ),
            _ => {
                return Err(invalid(
                    "C51 confirmatory high-observation replay refinement must be 8 or 10",
                ))
            }
        };
        let config = Self {
            profile: profile.to_string(),
            refinements: vec![refinement],
            ranks: ranks.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            observation_kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    /// Fresh post-replay extension resolving the preregistered 2% `m=4096`
    /// sensitivity rank at `n=10`. The shared 4,608-mode solve is accuracy
    /// evidence only; production method timing must request the selected rank
    /// independently.
    pub fn high_observation_confirmatory_rank_extension_n10() -> Result<Self> {
        let config = Self {
            profile: C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10.to_string(),
            refinements: vec![10],
            ranks: C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_N10_GRID.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            observation_kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    /// Confirmatory `m=4096` accuracy gate. This remains separate from both
    /// the `m=2048` primary decision and the seven-count contextual ladder.
    pub fn high_observation_confirmatory(refinement: usize) -> Result<Self> {
        let (profile, ranks) = match refinement {
            8 => (
                C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8,
                C51_HIGH_OBSERVATION_RANKS_N8.as_slice(),
            ),
            10 => (
                C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10,
                C51_HIGH_OBSERVATION_RANKS_N10.as_slice(),
            ),
            _ => {
                return Err(invalid(
                    "C51 confirmatory high-observation refinement must be 8 or 10",
                ))
            }
        };
        let config = Self {
            profile: profile.to_string(),
            refinements: vec![refinement],
            ranks: ranks.to_vec(),
            requested_maximum_rank: None,
            alphas: vec![MaternAlpha::Two],
            observation_kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
            maximum_basis_storage_bytes: 512 * 1024 * 1024,
            ..Self::smoke()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn evidence_tier(&self) -> &'static str {
        if self.is_high_observation_primary() {
            "high_observation_primary_accuracy"
        } else if self.is_high_observation_replay() {
            "high_observation_replay_accuracy_only"
        } else if self.is_high_observation_confirmatory() {
            "high_observation_confirmatory_accuracy"
        } else if self.is_high_observation_confirmatory_rank_extension() {
            "high_observation_confirmatory_rank_extension"
        } else if self.is_high_observation_accuracy() {
            "high_observation_accuracy"
        } else if self.is_rank_extension() {
            "rank_extension"
        } else {
            "rank_discovery"
        }
    }

    pub fn post_primary_rationale(&self) -> Option<&'static str> {
        if self.is_high_observation_primary() {
            Some("claim-bearing pre-performance m=2048 rank gate with 5% primary and 2% sensitivity thresholds")
        } else if self.is_high_observation_replay() {
            Some("accuracy-only replay of previously generated modes to select the next proof-grade exact-request rank; timing and rank-decision export are forbidden")
        } else if self.is_high_observation_confirmatory() {
            Some("confirmatory pre-performance m=4096 rank gate with the same frozen 5% primary and 2% sensitivity thresholds")
        } else if self.is_high_observation_confirmatory_rank_extension() {
            Some("fresh post-replay extension of the unresolved preregistered 2% m=4096 n=10 sensitivity rank; shared-basis timing is forbidden")
        } else if self.is_high_observation_accuracy() {
            Some(C51_HIGH_OBSERVATION_RATIONALE)
        } else {
            self.is_rank_extension()
                .then_some(C51_SPECTRAL_RANK_EXTENSION_RATIONALE)
        }
    }

    /// Expected active Whitney dimension frozen into a mesh-specific extension.
    pub fn expected_state_dimension(&self) -> Option<usize> {
        match self.profile.as_str() {
            C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8 => Some(3_032),
            C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10 => Some(6_130),
            C51_HIGH_OBSERVATION_PROFILE_N8 => Some(3_032),
            C51_HIGH_OBSERVATION_PROFILE_N10 => Some(6_130),
            C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8 => Some(3_032),
            C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10 => Some(6_130),
            C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8 => Some(3_032),
            C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10 => Some(6_130),
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8 => Some(3_032),
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10 => Some(6_130),
            C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10 => Some(6_130),
            C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8 => Some(3_032),
            C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10 => Some(6_130),
            _ => None,
        }
    }

    fn is_rank_extension(&self) -> bool {
        matches!(
            self.profile.as_str(),
            C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8 | C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10
        )
    }

    fn is_high_observation_accuracy(&self) -> bool {
        matches!(
            self.profile.as_str(),
            C51_HIGH_OBSERVATION_PROFILE_N8 | C51_HIGH_OBSERVATION_PROFILE_N10
        )
    }

    fn is_high_observation_primary(&self) -> bool {
        matches!(
            self.profile.as_str(),
            C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8 | C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10
        )
    }

    fn is_high_observation_replay(&self) -> bool {
        matches!(
            self.profile.as_str(),
            C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8
                | C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10
                | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8
                | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10
        )
    }

    fn is_high_observation_confirmatory_replay(&self) -> bool {
        matches!(
            self.profile.as_str(),
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8
                | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10
        )
    }

    fn is_high_observation_confirmatory_rank_extension(&self) -> bool {
        self.profile == C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10
    }

    fn is_high_observation_confirmatory(&self) -> bool {
        matches!(
            self.profile.as_str(),
            C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8
                | C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10
        )
    }

    fn is_any_high_observation(&self) -> bool {
        self.is_high_observation_primary()
            || self.is_high_observation_replay()
            || self.is_high_observation_confirmatory()
            || self.is_high_observation_confirmatory_rank_extension()
            || self.is_high_observation_accuracy()
    }

    fn high_observation_counts(&self) -> &'static [usize] {
        if self.is_high_observation_primary()
            || (self.is_high_observation_replay()
                && !self.is_high_observation_confirmatory_replay())
        {
            &[C51_HIGH_OBSERVATION_PRIMARY_COUNT]
        } else if self.is_high_observation_confirmatory()
            || self.is_high_observation_confirmatory_replay()
            || self.is_high_observation_confirmatory_rank_extension()
        {
            &[C51_HIGH_OBSERVATION_CONFIRMATORY_COUNT]
        } else if self.is_high_observation_accuracy() {
            &C51_HIGH_OBSERVATION_COUNTS
        } else {
            &[]
        }
    }

    fn covariance_designs(&self) -> &'static [C51CovarianceDesign] {
        if self.is_high_observation_primary()
            || (self.is_high_observation_replay()
                && !self.is_high_observation_confirmatory_replay())
        {
            &[C51CovarianceDesign::HighM2048]
        } else if self.is_high_observation_confirmatory()
            || self.is_high_observation_confirmatory_replay()
            || self.is_high_observation_confirmatory_rank_extension()
        {
            &[C51CovarianceDesign::HighM4096]
        } else if self.is_high_observation_accuracy() {
            &C51CovarianceDesign::HIGH_OBSERVATION
        } else {
            &C51CovarianceDesign::ALL
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
        let expected_extension = match self.profile.as_str() {
            C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8 => {
                Some((8, C51_SPECTRAL_RANK_EXTENSION_N8_GRID.as_slice()))
            }
            C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10 => {
                Some((10, C51_SPECTRAL_RANK_EXTENSION_N10_GRID.as_slice()))
            }
            C51_HIGH_OBSERVATION_PROFILE_N8 => Some((8, C51_HIGH_OBSERVATION_RANKS_N8.as_slice())),
            C51_HIGH_OBSERVATION_PROFILE_N10 => {
                Some((10, C51_HIGH_OBSERVATION_RANKS_N10.as_slice()))
            }
            C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8 => {
                Some((8, C51_HIGH_OBSERVATION_RANKS_N8.as_slice()))
            }
            C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10 => {
                Some((10, C51_HIGH_OBSERVATION_RANKS_N10.as_slice()))
            }
            C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8 => {
                Some((8, C51_HIGH_OBSERVATION_REPLAY_RANKS_N8.as_slice()))
            }
            C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10 => {
                Some((10, C51_HIGH_OBSERVATION_REPLAY_RANKS_N10.as_slice()))
            }
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8 => {
                Some((8, C51_HIGH_OBSERVATION_REPLAY_RANKS_N8.as_slice()))
            }
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10 => {
                Some((10, C51_HIGH_OBSERVATION_REPLAY_RANKS_N10.as_slice()))
            }
            C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10 => Some((
                10,
                C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_N10_GRID.as_slice(),
            )),
            C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8 => {
                Some((8, C51_HIGH_OBSERVATION_RANKS_N8.as_slice()))
            }
            C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10 => {
                Some((10, C51_HIGH_OBSERVATION_RANKS_N10.as_slice()))
            }
            _ => None,
        };
        if let Some((refinement, ranks)) = expected_extension {
            if self.refinements != [refinement]
                || self.ranks != ranks
                || self.requested_maximum_rank.is_some()
                || self.alphas != [MaternAlpha::Two]
                || (self.is_any_high_observation()
                    && self.observation_kernel_radius != C51_FIXED_OBSERVATION_RADIUS)
                || (self.is_any_high_observation()
                    && (self.high_observation_counts().last().map_or(true, |count| {
                        *count > C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT
                    }) || self
                        .covariance_designs()
                        .iter()
                        .map(|design| design.observation_count(1))
                        .collect::<Vec<_>>()
                        != self.high_observation_counts()))
            {
                return Err(invalid(
                    "C51 post-primary accuracy profiles have immutable refinement, alpha, rank grid, and sensor radius",
                ));
            }
        } else if self.alphas != [MaternAlpha::Two, MaternAlpha::Three] {
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
    HighM32,
    HighM128,
    HighM512,
    HighM1024,
    HighM2048,
    HighM4096,
    HighM5184,
}

impl C51CovarianceDesign {
    pub const ALL: [Self; 4] = [
        Self::Prior,
        Self::Fixed32,
        Self::Density020,
        Self::Density100,
    ];
    pub const HIGH_OBSERVATION: [Self; 7] = [
        Self::HighM32,
        Self::HighM128,
        Self::HighM512,
        Self::HighM1024,
        Self::HighM2048,
        Self::HighM4096,
        Self::HighM5184,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prior => "prior",
            Self::Fixed32 => "fixed32",
            Self::Density020 => "density_0.02",
            Self::Density100 => "density_0.10",
            Self::HighM32 => "high_m32",
            Self::HighM128 => "high_m128",
            Self::HighM512 => "high_m512",
            Self::HighM1024 => "high_m1024",
            Self::HighM2048 => "high_m2048",
            Self::HighM4096 => "high_m4096",
            Self::HighM5184 => "high_m5184",
        }
    }

    fn observation_count(self, state_dimension: usize) -> usize {
        match self {
            Self::Prior => 0,
            Self::Fixed32 => 32.min(C51_PRIMARY_SCALAR_OBSERVATION_COUNT),
            Self::Density020 => (0.02 * state_dimension as f64).ceil() as usize,
            Self::Density100 => (0.10 * state_dimension as f64).ceil() as usize,
            Self::HighM32 => 32,
            Self::HighM128 => 128,
            Self::HighM512 => 512,
            Self::HighM1024 => 1_024,
            Self::HighM2048 => 2_048,
            Self::HighM4096 => 4_096,
            Self::HighM5184 => 5_184,
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
pub struct C51ExactCovarianceAuditRecord {
    pub refinement: usize,
    pub alpha: u32,
    pub design: C51CovarianceDesign,
    pub bank: &'static str,
    pub policy_version: &'static str,
    pub status: &'static str,
    pub residual_aware: bool,
    pub residual_status: &'static str,
    pub projection_applied: bool,
    pub dimension: usize,
    pub raw_scale: f64,
    pub raw_maximum_absolute_skew: f64,
    pub raw_maximum_relative_skew: f64,
    pub raw_maximum_skew_row: usize,
    pub raw_maximum_skew_column: usize,
    pub relative_skew_limit: f64,
    pub projection_maximum_absolute_correction: f64,
    pub projection_frobenius_correction: f64,
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
pub struct C51HighObservationDesignAudit {
    pub refinement: usize,
    pub sensor_design_id: &'static str,
    pub grid_points_per_axis: usize,
    pub kernel_radius: f64,
    pub scalar_capacity: usize,
    pub maximum_prefix_kernels: Vec<WendlandC2Kernel3d>,
    pub maximum_prefix_quadrature_normalizations: Vec<f64>,
    pub prefixes: Vec<C51ScalarObservationPrefixMetadata>,
    pub scalar_map_nonzeros: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51CompactAccuracyRow {
    pub refinement: usize,
    pub state_dimension: usize,
    pub alpha: u32,
    pub design: C51CovarianceDesign,
    pub observation_count: usize,
    pub status: String,
    pub unavailable_reason: String,
    pub selection: Option<C51CovarianceErrorMetrics>,
    pub held_out: Option<C51CovarianceErrorMetrics>,
    /// Exact-FEEC prior RMS on the frozen selection bank. This is fixed before
    /// conditioning and is never estimated from held-out outputs.
    pub calibration_target_prior_rms: Option<f64>,
    /// Projected-Whitney prior RMS on the same selection bank before scaling.
    pub calibration_uncalibrated_prior_rms: Option<f64>,
    /// Scalar multiplying the raw projected precision.
    pub calibration_precision_scale: Option<f64>,
    /// Equivalent scalar multiplying the Matérn amplitude `tau`.
    pub calibration_tau_multiplier: Option<f64>,
    /// Posterior covariance error after applying the frozen prior calibration.
    pub calibrated_selection: Option<C51CovarianceErrorMetrics>,
    /// Independent posterior covariance error after applying the frozen prior
    /// calibration; this bank is not used to choose the scalar.
    pub calibrated_held_out: Option<C51CovarianceErrorMetrics>,
    pub latent_mass_action_relative_errors: Option<[f64; 2]>,
    pub latent_mass_action_maximum_relative_error: Option<f64>,
    pub method_performance_timing_eligible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51PerformanceRankDecision {
    pub refinement: usize,
    pub state_dimension: usize,
    pub scalar_observation_count: usize,
    pub alpha: u32,
    pub relative_tolerance: f64,
    pub sensor_design_id: &'static str,
    pub output_design_id: &'static str,
    pub selected_spectral_rank: usize,
    pub lower_failing_rank: usize,
    pub lower_failing_selection_primary_error: f64,
    pub selection_primary_error: f64,
    pub held_out_primary_error: f64,
    pub validation_status: &'static str,
    pub rank_status: &'static str,
    pub compact_whitney_selection_primary_error: f64,
    pub compact_whitney_held_out_primary_error: f64,
    pub calibrated_compact_whitney_selection_primary_error: f64,
    pub calibrated_compact_whitney_held_out_primary_error: f64,
    pub compact_whitney_calibration_target_prior_rms: f64,
    pub compact_whitney_calibration_uncalibrated_prior_rms: f64,
    pub compact_whitney_calibration_precision_scale: f64,
    pub compact_whitney_calibration_tau_multiplier: f64,
    pub compact_whitney_validation_status: &'static str,
    pub calibrated_compact_whitney_validation_status: &'static str,
    pub accuracy_archive_sha256: String,
    pub timing_basis_source: &'static str,
    pub shared_basis_timing_eligible: bool,
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
    pub exact_covariance_audits: Vec<C51ExactCovarianceAuditRecord>,
    pub sensor_banks: Vec<C51SensorBankAudit>,
    pub high_observation_designs: Vec<C51HighObservationDesignAudit>,
    pub compact_accuracy_rows: Vec<C51CompactAccuracyRow>,
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
        let sensor_banks_per_refinement = if self.config.is_any_high_observation() {
            2
        } else {
            3
        };
        if self.sensor_banks.len() != sensor_banks_per_refinement * self.config.refinements.len() {
            return Err(invalid(
                "C51 spectral-rank archive has the wrong number of profile-specific sensor audits",
            ));
        }
        if self.config.is_any_high_observation() {
            if self.high_observation_designs.len() != self.config.refinements.len()
                || self.high_observation_designs.iter().any(|audit| {
                    audit.sensor_design_id != C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID
                        || audit.grid_points_per_axis != C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS
                        || audit.kernel_radius != C51_FIXED_OBSERVATION_RADIUS
                        || audit.scalar_capacity != C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT
                        || audit.maximum_prefix_kernels.len()
                            != self
                                .config
                                .high_observation_counts()
                                .last()
                                .copied()
                                .unwrap_or(0)
                                .div_ceil(3)
                        || audit.maximum_prefix_quadrature_normalizations.len()
                            != audit.maximum_prefix_kernels.len()
                        || audit.prefixes.len() != self.config.high_observation_counts().len()
                        || audit.scalar_map_nonzeros.len() != audit.prefixes.len()
                        || audit
                            .prefixes
                            .iter()
                            .map(|prefix| prefix.scalar_observation_count)
                            .collect::<Vec<_>>()
                            != self.config.high_observation_counts()
                })
            {
                return Err(invalid(
                    "C51 high-observation archive lacks its frozen nested sensor provenance",
                ));
            }
        } else if !self.high_observation_designs.is_empty() {
            return Err(invalid(
                "only high-observation profiles may archive high-observation sensor provenance",
            ));
        }
        if self.config.is_any_high_observation() {
            let expected =
                self.config.refinements.len() * self.config.high_observation_counts().len();
            if self.compact_accuracy_rows.len() != expected {
                return Err(invalid(
                    "C51 high-observation archive requires one projected Whitney-compact accuracy row per observation prefix",
                ));
            }
            for row in &self.compact_accuracy_rows {
                let measured = row.status == "measured";
                let calibration_complete = row.calibration_target_prior_rms.is_some()
                    && row.calibration_uncalibrated_prior_rms.is_some()
                    && row.calibration_precision_scale.is_some()
                    && row.calibration_tau_multiplier.is_some()
                    && row.calibrated_selection.is_some()
                    && row.calibrated_held_out.is_some();
                let calibration_valid = if !measured {
                    true
                } else if let (Some(target), Some(raw), Some(scale), Some(tau_multiplier)) = (
                    row.calibration_target_prior_rms,
                    row.calibration_uncalibrated_prior_rms,
                    row.calibration_precision_scale,
                    row.calibration_tau_multiplier,
                ) {
                    [target, raw, scale, tau_multiplier]
                        .into_iter()
                        .all(|value| value.is_finite() && value > 0.0)
                        && (tau_multiplier.powi(2) - scale).abs() <= 1.0e-12 * scale
                } else {
                    false
                };
                if row.method_performance_timing_eligible
                    || !C51_HIGH_OBSERVATION_COUNTS.contains(&row.observation_count)
                    || measured
                        != (row.selection.is_some()
                            && row.held_out.is_some()
                            && calibration_complete
                            && row.latent_mass_action_relative_errors.is_some()
                            && row.latent_mass_action_maximum_relative_error.is_some())
                    || (!measured
                        && (row.calibration_target_prior_rms.is_some()
                            || row.calibration_uncalibrated_prior_rms.is_some()
                            || row.calibration_precision_scale.is_some()
                            || row.calibration_tau_multiplier.is_some()
                            || row.calibrated_selection.is_some()
                            || row.calibrated_held_out.is_some()))
                    || !calibration_valid
                    || (measured && !row.unavailable_reason.is_empty())
                    || (!measured && row.unavailable_reason.is_empty())
                {
                    return Err(invalid(
                        "C51 projected Whitney-compact accuracy row is incomplete or timing eligible",
                    ));
                }
            }
        } else if !self.compact_accuracy_rows.is_empty() {
            return Err(invalid(
                "only high-observation profiles may archive compact accuracy rows",
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
        let mut audit_keys = BTreeSet::new();
        for audit in &self.exact_covariance_audits {
            if !audit_keys.insert((audit.refinement, audit.alpha, audit.design, audit.bank)) {
                return Err(invalid("C51 exact covariance audits must be unique"));
            }
            if !["selection", "held_out"].contains(&audit.bank)
                || audit.policy_version != INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION
                || audit.status != "accepted_interim_scale_relative_and_projected"
                || audit.residual_aware
                || audit.residual_status.is_empty()
                || !audit.projection_applied
                || audit.dimension != OUTPUT_DIMENSION
                || audit.raw_maximum_skew_row >= audit.dimension
                || audit.raw_maximum_skew_column >= audit.dimension
                || audit.relative_skew_limit != INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT
                || audit.raw_maximum_relative_skew > audit.relative_skew_limit
                || [
                    audit.raw_scale,
                    audit.raw_maximum_absolute_skew,
                    audit.raw_maximum_relative_skew,
                    audit.relative_skew_limit,
                    audit.projection_maximum_absolute_correction,
                    audit.projection_frobenius_correction,
                ]
                .into_iter()
                .any(|value| !value.is_finite() || value < 0.0)
            {
                return Err(invalid(
                    "C51 exact covariance audit does not satisfy the frozen interim policy",
                ));
            }
            for representation in ["exact_supported_raw", "exact_supported_symmetric"] {
                let entries = self
                    .covariances
                    .iter()
                    .filter(|entry| {
                        entry.refinement == audit.refinement
                            && entry.alpha == audit.alpha
                            && entry.design == audit.design
                            && entry.rank.is_none()
                            && entry.bank == audit.bank
                            && entry.representation == representation
                    })
                    .count();
                if entries != audit.dimension * audit.dimension {
                    return Err(invalid(
                        "C51 audited exact covariance must archive complete raw and symmetric matrices",
                    ));
                }
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
            if measured {
                for bank in ["selection", "held_out"] {
                    if !audit_keys.contains(&(row.refinement, row.alpha, row.design, bank)) {
                        return Err(invalid(
                            "measured C51 row lacks an accepted exact covariance audit",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(directory)?;
        self.write_rows(&directory.join("rows.csv"))?;
        self.write_covariances(&directory.join("covariances.csv"))?;
        self.write_exact_covariance_audits(&directory.join("exact_covariance_audits.csv"))?;
        self.write_basis_attempts(&directory.join("basis_attempts.csv"))?;
        self.write_sensor_banks(&directory.join("sensor_banks.csv"))?;
        self.write_sensor_overlaps(&directory.join("sensor_overlaps.csv"))?;
        self.write_high_observation_prefixes(&directory.join("high_observation_prefixes.csv"))?;
        self.write_high_observation_kernels(&directory.join("high_observation_kernels.csv"))?;
        self.write_compact_accuracy(&directory.join("compact_accuracy.csv"))?;
        self.write_manifest(&directory.join("manifest.yaml"))?;
        Ok(())
    }

    /// Export the accuracy-gated rank consumed by the separately timed C51
    /// performance campaign. The digest covers the canonical archive files in
    /// a fixed order and therefore does not depend on filesystem traversal.
    pub fn write_performance_rank_decision(
        &self,
        accuracy_archive_directory: &Path,
        perf0_archive_directory: &Path,
        output: &Path,
    ) -> Result<C51PerformanceRankDecision> {
        self.validate()?;
        validate_c51_perf0_archive(perf0_archive_directory)?;
        let digest = c51_accuracy_archive_sha256(accuracy_archive_directory)?;
        let perf0_digest = c51_named_files_sha256(
            perf0_archive_directory,
            &C51_PERF0_ARCHIVE_HASH_FILES,
            "C51 PERF-0 archive",
        )?;
        let decision = derive_performance_rank_decision(self, digest)?;
        let payload = serde_json::json!({
            "schema_version": C51_PERFORMANCE_RANK_DECISION_SCHEMA_VERSION,
            "profile": "juq-performance-v1",
            "status": "accuracy_validated",
            "decisions": [{
                "decision_id": format!(
                    "rank-n{}-m{}-r{}",
                    decision.refinement,
                    decision.scalar_observation_count,
                    decision.selected_spectral_rank
                ),
                "source_accuracy_profile": self.config.profile,
                "refinement": decision.refinement,
                "state_dimension": decision.state_dimension,
                "scalar_observation_count": decision.scalar_observation_count,
                "alpha": decision.alpha,
                "relative_tolerance": decision.relative_tolerance,
                "observation_design_id": decision.sensor_design_id,
                "physical_qoi_design_id": decision.output_design_id,
                "reference_discrete_law_id": "exact_full_metric_whitney",
                "common_h_g_fixture_sha256": decision.accuracy_archive_sha256,
                "spectral_low_rank": {
                    "rank": decision.selected_spectral_rank,
                    "lower_failing_rank": decision.lower_failing_rank,
                    "lower_failing_selection_relative_error": decision.lower_failing_selection_primary_error,
                    "selection_relative_error": decision.selection_primary_error,
                    "heldout_relative_error": decision.held_out_primary_error,
                    "validation_status": decision.validation_status,
                    "rank_status": decision.rank_status,
                    "accuracy_artifact_sha256": decision.accuracy_archive_sha256,
                    "timing_basis_source": decision.timing_basis_source,
                    "shared_maximum_basis_timing_eligible": decision.shared_basis_timing_eligible
                },
                "whitney_compact": {
                    "selection_relative_error": decision.compact_whitney_selection_primary_error,
                    "heldout_relative_error": decision.compact_whitney_held_out_primary_error,
                    "validation_status": decision.compact_whitney_validation_status,
                    "accuracy_artifact_sha256": decision.accuracy_archive_sha256,
                    "identical_observation_and_output_maps": true,
                    "method_performance_timing_eligible": false
                },
                "whitney_compact_calibrated": {
                    "selection_relative_error": decision.calibrated_compact_whitney_selection_primary_error,
                    "heldout_relative_error": decision.calibrated_compact_whitney_held_out_primary_error,
                    "validation_status": decision.calibrated_compact_whitney_validation_status,
                    "amplitude_calibration": {
                        "stage": "prior_before_conditioning",
                        "parameter_changed": "tau_only",
                        "target": "exact_feec_prior_selection_bank_physical_rms",
                        "calibration_bank": "selection",
                        "evaluation_bank": "held_out",
                        "target_prior_rms": decision.compact_whitney_calibration_target_prior_rms,
                        "uncalibrated_prior_rms": decision.compact_whitney_calibration_uncalibrated_prior_rms,
                        "precision_scale": decision.compact_whitney_calibration_precision_scale,
                        "tau_multiplier": decision.compact_whitney_calibration_tau_multiplier,
                        "heldout_used_for_calibration": false
                    },
                    "accuracy_artifact_sha256": decision.accuracy_archive_sha256,
                    "identical_observation_and_output_maps": true,
                    "method_performance_timing_eligible": false
                },
                "exact_observation_gp": {
                    "validation_status": "perf0_same_law_validated",
                    "fixture_id": "PERF-0",
                    "fixture_artifact_sha256": perf0_digest,
                    "fixture_hash_scope": "manifest.yaml+rows.csv"
                },
                "accuracy_artifact_hash_scope": "canonical_c51_accuracy_archive_files_v1"
            }]
        });
        let writer = BufWriter::new(File::create(output)?);
        serde_json::to_writer_pretty(writer, &payload)
            .map_err(|error| invalid(format!("could not serialize C51 rank decision: {error}")))?;
        Ok(decision)
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
                self.config.evidence_tier().to_string(),
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

    fn write_exact_covariance_audits(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "refinement",
                "alpha",
                "design",
                "bank",
                "policy_version",
                "status",
                "residual_aware",
                "residual_status",
                "projection_applied",
                "dimension",
                "raw_scale",
                "raw_maximum_absolute_skew",
                "raw_maximum_relative_skew",
                "raw_maximum_skew_row",
                "raw_maximum_skew_column",
                "relative_skew_limit",
                "projection_maximum_absolute_correction",
                "projection_frobenius_correction",
            ])
            .map_err(csv_error)?;
        for audit in &self.exact_covariance_audits {
            writer
                .write_record([
                    audit.refinement.to_string(),
                    audit.alpha.to_string(),
                    audit.design.as_str().to_string(),
                    audit.bank.to_string(),
                    audit.policy_version.to_string(),
                    audit.status.to_string(),
                    audit.residual_aware.to_string(),
                    audit.residual_status.to_string(),
                    audit.projection_applied.to_string(),
                    audit.dimension.to_string(),
                    format_float(audit.raw_scale),
                    format_float(audit.raw_maximum_absolute_skew),
                    format_float(audit.raw_maximum_relative_skew),
                    audit.raw_maximum_skew_row.to_string(),
                    audit.raw_maximum_skew_column.to_string(),
                    format_float(audit.relative_skew_limit),
                    format_float(audit.projection_maximum_absolute_correction),
                    format_float(audit.projection_frobenius_correction),
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

    fn write_high_observation_prefixes(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "refinement",
                "sensor_design_id",
                "grid_points_per_axis",
                "kernel_radius",
                "scalar_capacity",
                "scalar_observation_count",
                "vector_center_count",
                "bx_count",
                "by_count",
                "bz_count",
                "selected_row_rule",
                "scalar_map_nonzeros",
            ])
            .map_err(csv_error)?;
        for audit in &self.high_observation_designs {
            for (prefix, nonzeros) in audit.prefixes.iter().zip(&audit.scalar_map_nonzeros) {
                writer
                    .write_record([
                        audit.refinement.to_string(),
                        audit.sensor_design_id.to_string(),
                        audit.grid_points_per_axis.to_string(),
                        format_float(audit.kernel_radius),
                        audit.scalar_capacity.to_string(),
                        prefix.scalar_observation_count.to_string(),
                        prefix.vector_center_count.to_string(),
                        prefix.component_counts[0].to_string(),
                        prefix.component_counts[1].to_string(),
                        prefix.component_counts[2].to_string(),
                        "contiguous_zero_based_scalar_prefix".to_string(),
                        nonzeros.to_string(),
                    ])
                    .map_err(csv_error)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_high_observation_kernels(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "refinement",
                "sensor_design_id",
                "kernel_index",
                "center_x",
                "center_y",
                "center_z",
                "radius",
                "quadrature_normalization",
            ])
            .map_err(csv_error)?;
        for audit in &self.high_observation_designs {
            for (index, (kernel, normalization)) in audit
                .maximum_prefix_kernels
                .iter()
                .zip(&audit.maximum_prefix_quadrature_normalizations)
                .enumerate()
            {
                writer
                    .write_record([
                        audit.refinement.to_string(),
                        audit.sensor_design_id.to_string(),
                        index.to_string(),
                        format_float(kernel.center()[0]),
                        format_float(kernel.center()[1]),
                        format_float(kernel.center()[2]),
                        format_float(kernel.radius()),
                        format_float(*normalization),
                    ])
                    .map_err(csv_error)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_compact_accuracy(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path).map_err(csv_error)?;
        writer
            .write_record([
                "evidence_tier",
                "method",
                "method_performance_timing_eligible",
                "refinement",
                "state_dimension",
                "alpha",
                "design",
                "observation_count",
                "status",
                "unavailable_reason",
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
                "calibration_policy",
                "calibration_bank",
                "calibration_target_prior_rms",
                "calibration_uncalibrated_prior_rms",
                "calibration_precision_scale",
                "calibration_tau_multiplier",
                "calibrated_selection_e_f",
                "calibrated_selection_e_2",
                "calibrated_selection_e_diag",
                "calibrated_selection_e_dir",
                "calibrated_selection_e_corr",
                "calibrated_selection_e_primary",
                "calibrated_heldout_e_f",
                "calibrated_heldout_e_2",
                "calibrated_heldout_e_diag",
                "calibrated_heldout_e_dir",
                "calibrated_heldout_e_corr",
                "calibrated_heldout_e_primary",
                "latent_mass_action_probe_0_relative_error",
                "latent_mass_action_probe_1_relative_error",
                "latent_mass_action_maximum_relative_error",
            ])
            .map_err(csv_error)?;
        for row in &self.compact_accuracy_rows {
            let mut record = vec![
                self.config.evidence_tier().to_string(),
                "whitney_compact_projected".to_string(),
                row.method_performance_timing_eligible.to_string(),
                row.refinement.to_string(),
                row.state_dimension.to_string(),
                row.alpha.to_string(),
                row.design.as_str().to_string(),
                row.observation_count.to_string(),
                row.status.clone(),
                row.unavailable_reason.clone(),
            ];
            append_metrics(&mut record, row.selection);
            append_metrics(&mut record, row.held_out);
            if row.status == "measured" {
                record.extend([
                    "exact_feec_prior_selection_bank_physical_rms".to_string(),
                    "selection".to_string(),
                    format_float(
                        row.calibration_target_prior_rms
                            .expect("validated measured calibration target"),
                    ),
                    format_float(
                        row.calibration_uncalibrated_prior_rms
                            .expect("validated measured uncalibrated RMS"),
                    ),
                    format_float(
                        row.calibration_precision_scale
                            .expect("validated measured calibration precision scale"),
                    ),
                    format_float(
                        row.calibration_tau_multiplier
                            .expect("validated measured calibration tau multiplier"),
                    ),
                ]);
            } else {
                record.extend([
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                ]);
            }
            append_metrics(&mut record, row.calibrated_selection);
            append_metrics(&mut record, row.calibrated_held_out);
            match row.latent_mass_action_relative_errors {
                Some(errors) => record.extend([
                    format_float(errors[0]),
                    format_float(errors[1]),
                    format_float(
                        row.latent_mass_action_maximum_relative_error
                            .expect("validated measured compact latent action maximum"),
                    ),
                ]),
                None => record.extend([String::new(), String::new(), String::new()]),
            }
            writer.write_record(record).map_err(csv_error)?;
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
        writeln!(writer, "evidence_tier: {}", self.config.evidence_tier())?;
        match self.config.post_primary_rationale() {
            Some(rationale) => {
                writeln!(writer, "post_primary: true")?;
                writeln!(writer, "post_primary_rationale: {}", yaml_string(rationale))?;
            }
            None => {
                writeln!(writer, "post_primary: false")?;
                writeln!(writer, "post_primary_rationale: null")?;
            }
        }
        writeln!(writer, "method_performance_timing_eligible: false")?;
        writeln!(writer, "shared_maximum_basis: true")?;
        writeln!(
            writer,
            "observation_counts_dimension_independent: {}",
            self.config.is_any_high_observation()
        )?;
        if self.config.is_any_high_observation() {
            writeln!(
                writer,
                "sensor_design_id: {C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID}"
            )?;
            writeln!(
                writer,
                "high_observation_counts: {}",
                yaml_usizes(self.config.high_observation_counts())
            )?;
            let decision_scope = if self.config.is_high_observation_primary() {
                "primary_m2048_only"
            } else if self.config.is_high_observation_replay() {
                "exploratory_m2048_replay_no_rank_decision"
            } else if self.config.is_high_observation_confirmatory() {
                "confirmatory_m4096_only"
            } else {
                "contextual_stress_only_no_performance_rank_selection"
            };
            writeln!(writer, "performance_rank_decision_scope: {decision_scope}")?;
            write_yaml_usize_option(
                &mut writer,
                "performance_rank_decision_observation_count",
                if self.config.is_high_observation_primary() {
                    Some(C51_HIGH_OBSERVATION_PRIMARY_COUNT)
                } else if self.config.is_high_observation_replay() {
                    None
                } else if self.config.is_high_observation_confirmatory() {
                    Some(C51_HIGH_OBSERVATION_CONFIRMATORY_COUNT)
                } else {
                    None
                },
            )?;
        }
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
        write_yaml_usize_option(
            &mut writer,
            "expected_state_dimension",
            self.config.expected_state_dimension(),
        )?;
        writeln!(writer, "ranks: {}", yaml_usizes(&self.config.ranks))?;
        write_yaml_usize_option(
            &mut writer,
            "requested_maximum_rank",
            self.config.requested_maximum_rank,
        )?;
        writeln!(writer, "thresholds: [0.10, 0.05, 0.02]")?;
        writeln!(writer, "claim_bearing_physical_covariance_threshold: 0.05")?;
        writeln!(writer, "sensitivity_physical_covariance_threshold: 0.02")?;
        writeln!(writer, "contextual_physical_covariance_threshold: 0.10")?;
        writeln!(writer, "performance_rank_rule: smallest_tested_sustained_selection_rank_with_lower_failing_bracket_and_held_out_pass")?;
        writeln!(
            writer,
            "censored_endpoint_selectable_for_performance: false"
        )?;
        writeln!(
            writer,
            "state_dimension_endpoint_semantics: full_rank_required_not_low_rank_or_truncated"
        )?;
        writeln!(
            writer,
            "alpha_values: {}",
            yaml_usizes(
                &self
                    .config
                    .alphas
                    .iter()
                    .map(|alpha| alpha.as_u32() as usize)
                    .collect::<Vec<_>>()
            )
        )?;
        writeln!(writer, "kappa: {:.17e}", self.config.kappa)?;
        writeln!(writer, "tau: {:.17e}", self.config.tau)?;
        writeln!(writer, "projected_amplitude_calibration:")?;
        writeln!(writer, "  stage: prior_before_conditioning")?;
        writeln!(writer, "  parameter_changed: tau_only")?;
        writeln!(
            writer,
            "  target: exact_feec_prior_selection_bank_physical_rms"
        )?;
        writeln!(writer, "  calibration_bank: selection")?;
        writeln!(writer, "  evaluation_bank: held_out")?;
        writeln!(writer, "  held_out_used_for_calibration: false")?;
        writeln!(writer, "  raw_projected_result_retained: true")?;
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
        writeln!(
            writer,
            "exact_covariance_audits_csv: exact_covariance_audits.csv"
        )?;
        writeln!(
            writer,
            "exact_covariance_audit_policy: {}",
            INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION
        )?;
        writeln!(
            writer,
            "exact_covariance_relative_skew_limit: {:.17e}",
            INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT
        )?;
        writeln!(writer, "exact_covariance_audit_residual_aware: false")?;
        writeln!(writer, "basis_attempts_csv: basis_attempts.csv")?;
        writeln!(writer, "sensor_banks_csv: sensor_banks.csv")?;
        writeln!(writer, "sensor_overlaps_csv: sensor_overlaps.csv")?;
        writeln!(
            writer,
            "high_observation_prefixes_csv: high_observation_prefixes.csv"
        )?;
        writeln!(
            writer,
            "high_observation_kernels_csv: high_observation_kernels.csv"
        )?;
        writeln!(writer, "compact_accuracy_csv: compact_accuracy.csv")?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(feature = "external-solvers")]
pub fn run_c51_spectral_covariance_rank(
    config: &C51SpectralCovarianceRankConfig,
    metadata: C51SpectralCovarianceRankMetadata,
) -> Result<C51SpectralCovarianceRankArchive> {
    run_c51_spectral_covariance_rank_internal(config, metadata, None)
}

/// Replay a previously generated PETSc eigenbasis for accuracy exploration.
///
/// The imported vectors are checked against the current FEEC mass matrix, but
/// the mixed generalized eigen-equation is not re-certified from the cached
/// files. Consequently this route is restricted to replay profiles and cannot
/// emit the rank decision consumed by performance timing.
#[cfg(feature = "external-solvers")]
pub fn run_c51_spectral_covariance_rank_replay(
    config: &C51SpectralCovarianceRankConfig,
    metadata: C51SpectralCovarianceRankMetadata,
    petsc_workspace: &Path,
) -> Result<C51SpectralCovarianceRankArchive> {
    if !config.is_high_observation_replay() {
        return Err(invalid(
            "a precomputed PETSc workspace is accepted only by a C51 replay profile",
        ));
    }
    run_c51_spectral_covariance_rank_internal(config, metadata, Some(petsc_workspace))
}

#[cfg(feature = "external-solvers")]
fn run_c51_spectral_covariance_rank_internal(
    config: &C51SpectralCovarianceRankConfig,
    metadata: C51SpectralCovarianceRankMetadata,
    petsc_workspace: Option<&Path>,
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
        exact_covariance_audits: Vec::new(),
        sensor_banks: Vec::new(),
        high_observation_designs: Vec::new(),
        compact_accuracy_rows: Vec::new(),
        plumbing_gates,
    };
    for &refinement in &config.refinements {
        run_refinement(config, refinement, petsc_workspace, &mut archive)?;
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
    petsc_workspace: Option<&Path>,
    archive: &mut C51SpectralCovarianceRankArchive,
) -> Result<()> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)?;
    let state_dimension = operators.state().dimension();
    if let Some(expected) = config.expected_state_dimension() {
        if state_dimension != expected {
            return Err(FeecGmrfError::Dimension(format!(
                "C51 {} expected active state dimension {expected}, assembled {state_dimension}",
                config.profile
            )));
        }
    }
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
    archive
        .sensor_banks
        .push(sensor_bank_audit(refinement, "selection", &selection)?);
    archive
        .sensor_banks
        .push(sensor_bank_audit(refinement, "held_out", &held_out)?);
    let candidate_map = if config.is_any_high_observation() {
        let (map, audit) =
            high_observation_candidate_map_and_audit(config, refinement, &topology, &coordinates)?;
        archive.high_observation_designs.push(audit);
        map
    } else {
        let candidates = MagneticVolumeSensorBank3d::from_feec(
            &topology,
            &coordinates,
            &primary_observation_kernels(config.observation_kernel_radius)?,
        )?;
        archive.sensor_banks.push(sensor_bank_audit(
            refinement,
            "observation_candidates",
            &candidates,
        )?);
        candidates.map().map().clone()
    };
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
    let designs = build_designs(
        config.covariance_designs(),
        state_dimension,
        &candidate_map,
        &elimination,
    )?;

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
        append_unavailable_compact_for_config(
            archive,
            config,
            refinement,
            state_dimension,
            "spectral basis unavailable before exact/compact comparison",
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
        append_unavailable_compact_for_config(
            archive,
            config,
            refinement,
            state_dimension,
            &reason,
        );
        return Ok(());
    }

    let basis = if let Some(workspace) = petsc_workspace {
        match replay_basis_from_workspace(&operators, workspace, requested_rank, refinement) {
            Ok((basis, attempt)) => {
                archive.basis_attempts.push(attempt);
                basis
            }
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
                        "the cached PETSc basis failed import or mass validation".to_string(),
                    mass_orthogonality_certificate_status: "unavailable".to_string(),
                    mass_orthogonality_certificate_reason: reason.clone(),
                    generalized_residual_certificate_status: "unavailable".to_string(),
                    generalized_residual_certificate_reason:
                        "replay never substitutes for a proof-grade residual-validated solve"
                            .to_string(),
                    external_tool_versions_status: "unavailable".to_string(),
                    external_tool_versions_reason:
                        "cached helper execution metadata is not inferred".to_string(),
                    sigma_dimension: None,
                    converged_rank: None,
                    retained_rank: None,
                    input_nonzeros: None,
                    l_nonzeros: None,
                    d_nonzeros: None,
                    c_nonzeros: None,
                    lower_mass_nonzeros: None,
                    state_mass_nonzeros: Some(operators.state().mass().nnz()),
                    basis_storage_entries: None,
                    basis_storage_bytes: Some(expected_basis_bytes),
                    clamped_negative_eigenvalues: None,
                    eigenpair_selection: None,
                    lower_mass_solve: None,
                    workspace: Some(workspace.display().to_string()),
                    binary: None,
                });
                append_censored_rows(
                    archive,
                    config,
                    refinement,
                    state_dimension,
                    "cached_basis_revalidation",
                    &reason,
                );
                append_unavailable_compact_for_config(
                    archive,
                    config,
                    refinement,
                    state_dimension,
                    &reason,
                );
                return Ok(());
            }
        }
    } else {
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
                    append_unavailable_compact_for_config(
                        archive,
                        config,
                        refinement,
                        state_dimension,
                        &reason,
                    );
                    return Ok(());
                }
            };
        archive
            .basis_attempts
            .push(successful_basis_attempt(refinement, partial.diagnostics()));
        partial.into_basis()
    };

    for &alpha in &config.alphas {
        let row_start = archive.rows.len();
        let covariance_start = archive.covariances.len();
        let audit_start = archive.exact_covariance_audits.len();
        let compact_start = archive.compact_accuracy_rows.len();
        if let Err(error) = run_alpha(
            config,
            refinement,
            state_dimension,
            alpha,
            &topology,
            &metric,
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
            archive.exact_covariance_audits.truncate(audit_start);
            archive.compact_accuracy_rows.truncate(compact_start);
            if config.is_any_high_observation() {
                append_unavailable_compact_rows(
                    archive,
                    config,
                    refinement,
                    state_dimension,
                    alpha,
                    &error.to_string(),
                );
            }
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
fn replay_basis_from_workspace(
    operators: &ExactFormOperators,
    workspace: &Path,
    requested_rank: usize,
    refinement: usize,
) -> Result<(EigenBasis, C51SpectralBasisAttempt)> {
    let eigenvalue_path = workspace.join("out/eigenvals.bin");
    let eigenvector_path = workspace.join("out/eigenvecs_u.bin");
    let eigenvalues = petsc_read_eigenvals(path_text(&eigenvalue_path)?)
        .map_err(|error| replay_io_error(&eigenvalue_path, error))?;
    let eigenvectors = petsc_read_eigenvecs(path_text(&eigenvector_path)?)
        .map_err(|error| replay_io_error(&eigenvector_path, error))?;
    let state_dimension = operators.state().dimension();
    let converged_rank = eigenvalues.len();
    if eigenvectors.nrows() != state_dimension
        || eigenvectors.ncols() != converged_rank
        || converged_rank < requested_rank
    {
        return Err(FeecGmrfError::Dimension(format!(
            "C51 cached basis has eigenvalues {converged_rank}, vectors {}x{}, expected at least {requested_rank} vectors on state dimension {state_dimension}",
            eigenvectors.nrows(),
            eigenvectors.ncols()
        )));
    }
    let mass = feg_infer::sparse::core_triplet_to_feec_csr(operators.state().mass());
    let (basis, maximum_defect, clamped) = checked_imported_generalized_hodge_eigenbasis(
        eigenvalues,
        eigenvectors,
        &mass,
        1.0e-10,
        1.0e-8,
    )
    .map_err(spectral_error)?;
    let basis = basis
        .truncate_smallest(requested_rank)
        .map_err(spectral_error)?;
    let entries = state_dimension
        .checked_mul(requested_rank)
        .ok_or_else(|| FeecGmrfError::Dimension("C51 replay basis size overflow".to_string()))?;
    let bytes = entries
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| FeecGmrfError::Dimension("C51 replay basis bytes overflow".to_string()))?;
    let eigenvalue_sha = c51_file_sha256(&eigenvalue_path)?;
    let eigenvector_sha = c51_file_sha256(&eigenvector_path)?;
    let attempt = C51SpectralBasisAttempt {
        refinement,
        state_dimension,
        requested_rank,
        status: "replayed_accuracy_only".to_string(),
        reason: format!(
            "cached PETSc outputs mass-revalidated; eigenvals_sha256={eigenvalue_sha}; eigenvecs_u_sha256={eigenvector_sha}; maximum_mass_orthogonality_defect={maximum_defect:.17e}; not eligible for timing or rank-decision export"
        ),
        diagnostics_status: "measured".to_string(),
        diagnostics_unavailable_reason: String::new(),
        mass_orthogonality_certificate_status: "validated".to_string(),
        mass_orthogonality_certificate_reason: format!(
            "maximum defect {maximum_defect:.17e} <= 1e-8"
        ),
        generalized_residual_certificate_status: "unavailable".to_string(),
        generalized_residual_certificate_reason:
            "cached replay is exploratory; the selected production rank must be regenerated by the exact-request residual-validated route"
                .to_string(),
        external_tool_versions_status: "unavailable".to_string(),
        external_tool_versions_reason:
            "cached helper tool versions are not inferred from binary outputs".to_string(),
        sigma_dimension: None,
        converged_rank: Some(converged_rank),
        retained_rank: Some(requested_rank),
        input_nonzeros: None,
        l_nonzeros: None,
        d_nonzeros: None,
        c_nonzeros: None,
        lower_mass_nonzeros: None,
        state_mass_nonzeros: Some(operators.state().mass().nnz()),
        basis_storage_entries: Some(entries),
        basis_storage_bytes: Some(bytes),
        clamped_negative_eigenvalues: Some(clamped),
        eigenpair_selection: Some("smallest_cached_replay".to_string()),
        lower_mass_solve: Some("recorded_direct_not_reexecuted".to_string()),
        workspace: Some(workspace.display().to_string()),
        binary: None,
    };
    Ok((basis, attempt))
}

#[cfg(feature = "external-solvers")]
fn path_text(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        FeecGmrfError::InvalidParameter(format!(
            "C51 cached PETSc path is not valid UTF-8: {}",
            path.display()
        ))
    })
}

#[cfg(feature = "external-solvers")]
fn replay_io_error(path: &Path, error: std::io::Error) -> FeecGmrfError {
    FeecGmrfError::Io(std::io::Error::new(
        error.kind(),
        format!(
            "could not read cached PETSc output {}: {error}",
            path.display()
        ),
    ))
}

#[cfg(feature = "external-solvers")]
fn c51_file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    let mut digest = Sha256::new();
    digest.update(&bytes);
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(feature = "external-solvers")]
#[allow(clippy::too_many_arguments)]
fn run_alpha(
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    topology: &manifold::topology::complex::Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
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
    let compact_calibration_reference = if config.is_any_high_observation() {
        Some(
            prior_factor
                .pushforward_covariance_audited(selection_full)?
                .symmetric,
        )
    } else {
        None
    };
    let mut exact_designs = Vec::new();
    if config
        .covariance_designs()
        .contains(&C51CovarianceDesign::Prior)
    {
        exact_designs.push(ExactDesignCovariance {
            design: C51CovarianceDesign::Prior,
            observation_count: 0,
            observation_reduced: None,
            selection: prior_factor.pushforward_covariance_audited(selection_full)?,
            held_out: prior_factor.pushforward_covariance_audited(held_out_full)?,
            latent_actions: latent_probe_rhs
                .iter()
                .map(|rhs| prior_factor.covariance_action(rhs))
                .collect::<Result<Vec<_>>>()?,
        });
    }
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
            selection: posterior.pushforward_covariance_audited(selection_full)?,
            held_out: posterior.pushforward_covariance_audited(held_out_full)?,
            latent_actions: latent_probe_rhs
                .iter()
                .map(|rhs| posterior.covariance_action(rhs))
                .collect::<Result<Vec<_>>>()?,
        });
    }
    for exact in &exact_designs {
        validate_reference_covariance(&exact.selection.symmetric, "selection")?;
        validate_reference_covariance(&exact.held_out.symmetric, "held-out")?;
        archive_audited_exact_covariance(
            archive,
            refinement,
            alpha,
            exact.design,
            "selection",
            &exact.selection,
        );
        archive_audited_exact_covariance(
            archive,
            refinement,
            alpha,
            exact.design,
            "held_out",
            &exact.held_out,
        );
    }

    if config.is_any_high_observation() {
        let compact_covariance_start = archive.covariances.len();
        if let Err(error) = evaluate_compact_accuracy(
            config,
            refinement,
            state_dimension,
            alpha,
            topology,
            metric,
            operators,
            selection_full,
            held_out_full,
            compact_calibration_reference.as_deref().ok_or_else(|| {
                FeecGmrfError::Inference(
                    "C51 compact calibration reference was not constructed".to_string(),
                )
            })?,
            designs,
            &exact_designs,
            &latent_probe_rhs,
            archive,
        ) {
            archive
                .compact_accuracy_rows
                .retain(|row| !(row.refinement == refinement && row.alpha == alpha.as_u32()));
            archive.covariances.truncate(compact_covariance_start);
            append_unavailable_compact_rows(
                archive,
                config,
                refinement,
                state_dimension,
                alpha,
                &error.to_string(),
            );
        }
    }

    for &rank in &config.ranks {
        if rank > state_dimension || rank > maximum_basis.len() {
            append_censored_rank_rows(
                archive,
                config.covariance_designs(),
                CensoredRankContext {
                    refinement,
                    state_dimension,
                    alpha,
                    rank,
                    stage: "rank_validation",
                    reason: "requested rank exceeds the available active eigenbasis",
                },
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
                    config.covariance_designs(),
                    CensoredRankContext {
                        refinement,
                        state_dimension,
                        alpha,
                        rank,
                        stage: "spectral_rank_construction",
                        reason: &error.to_string(),
                    },
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
                    covariance_error_metrics(&selection_covariance, &exact.selection.symmetric)?;
                let held_out_metrics =
                    covariance_error_metrics(&held_out_covariance, &exact.held_out.symmetric)?;
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
#[allow(clippy::too_many_arguments)]
fn evaluate_compact_accuracy(
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    topology: &manifold::topology::complex::Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    operators: &ExactFormOperators,
    selection_full: &LinearMap,
    held_out_full: &LinearMap,
    exact_prior_selection_covariance: &[Vec<f64>],
    designs: &[ObservationDesign],
    exact_designs: &[ExactDesignCovariance],
    latent_probe_rhs: &[Vec<f64>],
    archive: &mut C51SpectralCovarianceRankArchive,
) -> Result<()> {
    let parameters = MaternParameters::new(alpha, config.kappa, config.tau)?;
    let compact_prior = MaternPriorBuilder::from_feec_whitney_projected(topology, metric, 1)?
        .parameters(parameters)
        .essential_boundary_conditions(EssentialBoundaryConditions::homogeneous_on_boundary(
            topology, 1,
        )?)
        .build()?;
    if compact_prior.dimension() != state_dimension {
        return Err(FeecGmrfError::Dimension(format!(
            "C51 compact prior dimension {} does not match exact active dimension {state_dimension}",
            compact_prior.dimension()
        )));
    }
    let target_prior_rms = covariance_rms(exact_prior_selection_covariance)?;
    let (calibrated_prior, calibration) =
        calibrate_prior_to_physical_rms(&compact_prior, selection_full, target_prior_rms)?;
    for design in designs {
        let exact = exact_designs
            .iter()
            .find(|exact| exact.design == design.kind)
            .ok_or_else(|| {
                FeecGmrfError::Inference(format!(
                    "C51 compact comparison lacks exact reference for {}",
                    design.kind.as_str()
                ))
            })?;
        let observation = LinearObservation::new(
            design.full.clone(),
            vec![0.0; design.count],
            GaussianNoise::variance(config.observation_noise_variance)?,
        )?;
        let mut posterior = LinearGaussianModelBuilder::new(compact_prior.clone())
            .observe(observation)?
            .condition()?;
        let calibrated_observation = LinearObservation::new(
            design.full.clone(),
            vec![0.0; design.count],
            GaussianNoise::variance(config.observation_noise_variance)?,
        )?;
        let mut calibrated_posterior = LinearGaussianModelBuilder::new(calibrated_prior.clone())
            .observe(calibrated_observation)?
            .condition()?;
        let selection_covariance = posterior.pushforward_covariance(selection_full)?;
        let held_out_covariance = posterior.pushforward_covariance(held_out_full)?;
        let calibrated_selection_covariance =
            calibrated_posterior.pushforward_covariance(selection_full)?;
        let calibrated_held_out_covariance =
            calibrated_posterior.pushforward_covariance(held_out_full)?;
        let approximate_latent_actions = latent_probe_rhs
            .iter()
            .map(|rhs| posterior.covariance_action(rhs))
            .collect::<Result<Vec<_>>>()?;
        let selection =
            covariance_error_metrics(&selection_covariance, &exact.selection.symmetric)?;
        let held_out = covariance_error_metrics(&held_out_covariance, &exact.held_out.symmetric)?;
        let calibrated_selection =
            covariance_error_metrics(&calibrated_selection_covariance, &exact.selection.symmetric)?;
        let calibrated_held_out =
            covariance_error_metrics(&calibrated_held_out_covariance, &exact.held_out.symmetric)?;
        let latent_errors = latent_action_errors(
            operators.state().mass(),
            &approximate_latent_actions,
            &exact.latent_actions,
        )?;
        archive_covariance(
            &mut archive.covariances,
            refinement,
            alpha,
            design.kind,
            None,
            "selection",
            "whitney_compact_projected",
            &selection_covariance,
        );
        archive_covariance(
            &mut archive.covariances,
            refinement,
            alpha,
            design.kind,
            None,
            "held_out",
            "whitney_compact_projected",
            &held_out_covariance,
        );
        archive_covariance(
            &mut archive.covariances,
            refinement,
            alpha,
            design.kind,
            None,
            "selection",
            "whitney_compact_projected_physical_rms_calibrated",
            &calibrated_selection_covariance,
        );
        archive_covariance(
            &mut archive.covariances,
            refinement,
            alpha,
            design.kind,
            None,
            "held_out",
            "whitney_compact_projected_physical_rms_calibrated",
            &calibrated_held_out_covariance,
        );
        archive.compact_accuracy_rows.push(C51CompactAccuracyRow {
            refinement,
            state_dimension,
            alpha: alpha.as_u32(),
            design: design.kind,
            observation_count: design.count,
            status: "measured".to_string(),
            unavailable_reason: String::new(),
            selection: Some(selection),
            held_out: Some(held_out),
            calibration_target_prior_rms: Some(calibration.target_rms),
            calibration_uncalibrated_prior_rms: Some(calibration.uncalibrated_rms),
            calibration_precision_scale: Some(calibration.precision_scale),
            calibration_tau_multiplier: Some(calibration.tau_multiplier()),
            calibrated_selection: Some(calibrated_selection),
            calibrated_held_out: Some(calibrated_held_out),
            latent_mass_action_relative_errors: Some(latent_errors),
            latent_mass_action_maximum_relative_error: Some(
                latent_errors.into_iter().fold(0.0, f64::max),
            ),
            method_performance_timing_eligible: false,
        });
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
#[cfg(feature = "external-solvers")]
fn append_unavailable_compact_rows(
    archive: &mut C51SpectralCovarianceRankArchive,
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    reason: &str,
) {
    for &design in config.covariance_designs() {
        if design == C51CovarianceDesign::Prior {
            continue;
        }
        archive.compact_accuracy_rows.push(C51CompactAccuracyRow {
            refinement,
            state_dimension,
            alpha: alpha.as_u32(),
            design,
            observation_count: design.observation_count(state_dimension),
            status: "unavailable".to_string(),
            unavailable_reason: reason.to_string(),
            selection: None,
            held_out: None,
            calibration_target_prior_rms: None,
            calibration_uncalibrated_prior_rms: None,
            calibration_precision_scale: None,
            calibration_tau_multiplier: None,
            calibrated_selection: None,
            calibrated_held_out: None,
            latent_mass_action_relative_errors: None,
            latent_mass_action_maximum_relative_error: None,
            method_performance_timing_eligible: false,
        });
    }
}

#[cfg(feature = "external-solvers")]
#[cfg(feature = "external-solvers")]
fn append_unavailable_compact_for_config(
    archive: &mut C51SpectralCovarianceRankArchive,
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    state_dimension: usize,
    reason: &str,
) {
    if config.is_any_high_observation() {
        for &alpha in &config.alphas {
            append_unavailable_compact_rows(
                archive,
                config,
                refinement,
                state_dimension,
                alpha,
                reason,
            );
        }
    }
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
    selection: AuditedSupportedCovariance,
    held_out: AuditedSupportedCovariance,
    latent_actions: Vec<Vec<f64>>,
}

#[cfg(feature = "external-solvers")]
fn build_designs(
    kinds: &[C51CovarianceDesign],
    state_dimension: usize,
    candidates: &LinearMap,
    elimination: &EssentialBoundaryElimination,
) -> Result<Vec<ObservationDesign>> {
    kinds
        .iter()
        .copied()
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

#[cfg(feature = "external-solvers")]
fn high_observation_candidate_map_and_audit(
    config: &C51SpectralCovarianceRankConfig,
    refinement: usize,
    topology: &manifold::topology::complex::Complex,
    coordinates: &manifold::geometry::coord::mesh::MeshCoords,
) -> Result<(LinearMap, C51HighObservationDesignAudit)> {
    let counts = config.high_observation_counts();
    let maximum_count = counts
        .last()
        .copied()
        .ok_or_else(|| invalid("C51 high-observation profile has no frozen observation count"))?;
    let maximum =
        assemble_juq_performance_observation_prefix(topology, coordinates, maximum_count)?;
    let mut prefixes = Vec::with_capacity(counts.len());
    let mut scalar_map_nonzeros = Vec::with_capacity(counts.len());
    for &count in counts {
        let (_, metadata) = juq_performance_observation_kernel_prefix(count)?;
        let map = maximum
            .scalar_map()
            .select_outputs(&metadata.selected_scalar_rows)?;
        if map.output_dimension() != count {
            return Err(FeecGmrfError::Dimension(format!(
                "C51 high-observation prefix {count} assembled {} scalar rows",
                map.output_dimension()
            )));
        }
        prefixes.push(metadata);
        scalar_map_nonzeros.push(map.matrix().triplet_iter().count());
    }
    let audit = C51HighObservationDesignAudit {
        refinement,
        sensor_design_id: C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID,
        grid_points_per_axis: C51_JUQ_PERFORMANCE_GRID_POINTS_PER_AXIS,
        kernel_radius: C51_FIXED_OBSERVATION_RADIUS,
        scalar_capacity: C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT,
        maximum_prefix_kernels: maximum.kernels().to_vec(),
        maximum_prefix_quadrature_normalizations: maximum.quadrature_normalizations().to_vec(),
        prefixes,
        scalar_map_nonzeros,
    };
    Ok((maximum.scalar_map().clone(), audit))
}

#[cfg(any(feature = "external-solvers", test))]
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

#[cfg(any(feature = "external-solvers", test))]
fn distance_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

#[cfg(any(feature = "external-solvers", test))]
fn eliminate_homogeneous_map(
    elimination: &EssentialBoundaryElimination,
    map: &LinearMap,
) -> Result<LinearMap> {
    elimination
        .eliminate_map(map, &vec![0.0; map.output_dimension()])
        .map(|eliminated| eliminated.reduced().clone())
}

#[cfg(any(feature = "external-solvers", test))]
fn to_csr(map: &LinearMap) -> CsrMatrix {
    feg_infer::sparse::core_triplet_to_feec_csr(map.matrix())
}

#[cfg(any(feature = "external-solvers", test))]
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

#[cfg(any(feature = "external-solvers", test))]
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

#[cfg(any(feature = "external-solvers", test))]
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

#[cfg(any(feature = "external-solvers", test))]
fn mass_apply(mass: &SparseMat, input: &[f64]) -> Result<Vec<f64>> {
    mass.apply_checked(input).map_err(FeecGmrfError::Dimension)
}

#[cfg(any(feature = "external-solvers", test))]
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

#[cfg(feature = "external-solvers")]
fn covariance_rms(covariance: &[Vec<f64>]) -> Result<f64> {
    validate_square_covariance(covariance)?;
    let mean_variance = covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .sum::<f64>()
        / covariance.len() as f64;
    if !mean_variance.is_finite() || mean_variance <= 0.0 {
        return Err(FeecGmrfError::Inference(
            "C51 calibration covariance has non-positive mean marginal variance".to_string(),
        ));
    }
    Ok(mean_variance.sqrt())
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

#[cfg(any(feature = "external-solvers", test))]
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

#[cfg(any(feature = "external-solvers", test))]
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
            config.covariance_designs(),
            CensoredRankContext {
                refinement,
                state_dimension,
                alpha,
                rank,
                stage,
                reason,
            },
        );
    }
}

#[cfg(any(feature = "external-solvers", test))]
struct CensoredRankContext<'a> {
    refinement: usize,
    state_dimension: usize,
    alpha: MaternAlpha,
    rank: usize,
    stage: &'a str,
    reason: &'a str,
}

#[cfg(any(feature = "external-solvers", test))]
fn append_censored_rank_rows(
    archive: &mut C51SpectralCovarianceRankArchive,
    designs: &[C51CovarianceDesign],
    context: CensoredRankContext<'_>,
) {
    for &design in designs {
        let observation_count = design.observation_count(context.state_dimension);
        append_censored_design_row(
            archive,
            context.refinement,
            context.state_dimension,
            context.alpha,
            context.rank,
            design,
            observation_count,
            context.stage,
            context.reason,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "external-solvers", test))]
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
fn archive_audited_exact_covariance(
    archive: &mut C51SpectralCovarianceRankArchive,
    refinement: usize,
    alpha: MaternAlpha,
    design: C51CovarianceDesign,
    bank: &'static str,
    covariance: &AuditedSupportedCovariance,
) {
    archive_covariance(
        &mut archive.covariances,
        refinement,
        alpha,
        design,
        None,
        bank,
        "exact_supported_raw",
        &covariance.raw,
    );
    archive_covariance(
        &mut archive.covariances,
        refinement,
        alpha,
        design,
        None,
        bank,
        "exact_supported_symmetric",
        &covariance.symmetric,
    );
    archive
        .exact_covariance_audits
        .push(exact_covariance_audit_record(
            refinement,
            alpha,
            design,
            bank,
            &covariance.audit,
        ));
}

#[cfg(feature = "external-solvers")]
fn exact_covariance_audit_record(
    refinement: usize,
    alpha: MaternAlpha,
    design: C51CovarianceDesign,
    bank: &'static str,
    audit: &SupportedCovarianceSymmetryAudit,
) -> C51ExactCovarianceAuditRecord {
    C51ExactCovarianceAuditRecord {
        refinement,
        alpha: alpha.as_u32(),
        design,
        bank,
        policy_version: audit.policy_version,
        status: audit.status,
        residual_aware: audit.residual_aware,
        residual_status: audit.residual_status,
        projection_applied: true,
        dimension: audit.dimension,
        raw_scale: audit.raw_scale,
        raw_maximum_absolute_skew: audit.raw_maximum_absolute_skew,
        raw_maximum_relative_skew: audit.raw_maximum_relative_skew,
        raw_maximum_skew_row: audit.raw_maximum_skew_row,
        raw_maximum_skew_column: audit.raw_maximum_skew_column,
        relative_skew_limit: audit.relative_skew_limit,
        projection_maximum_absolute_correction: audit.projection_maximum_absolute_correction,
        projection_frobenius_correction: audit.projection_frobenius_correction,
    }
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

fn derive_performance_rank_decision(
    archive: &C51SpectralCovarianceRankArchive,
    accuracy_archive_sha256: String,
) -> Result<C51PerformanceRankDecision> {
    if !(archive.config.is_high_observation_primary()
        || archive.config.is_high_observation_confirmatory())
    {
        return Err(invalid(
            "C51 performance rank decisions require a focused m=2048 or m=4096 accuracy profile",
        ));
    }
    let refinement = archive.config.refinements[0];
    let observation_count = archive.config.high_observation_counts()[0];
    let design = archive.config.covariance_designs()[0];
    let mut rows = archive
        .rows
        .iter()
        .filter(|row| {
            row.refinement == refinement
                && row.alpha == 2
                && row.design == design
                && row.observation_count == observation_count
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.rank);
    if rows.len() != archive.config.ranks.len()
        || rows.iter().map(|row| row.rank).collect::<Vec<_>>() != archive.config.ranks
    {
        return Err(invalid(
            "C51 focused rank decision lacks exactly one row for every frozen rank",
        ));
    }
    if rows.iter().any(|row| row.status != "measured") {
        return Err(invalid(
            "C51 focused rank decision cannot be exported from a censored rank ladder",
        ));
    }
    if rows
        .iter()
        .any(|row| row.method_performance_timing_eligible)
    {
        return Err(invalid(
            "C51 shared-basis accuracy rows must remain timing ineligible",
        ));
    }
    let primary_errors = rows
        .iter()
        .map(|row| {
            let selection = row
                .selection
                .ok_or_else(|| invalid("C51 measured rank row lacks selection metrics"))?
                .primary_envelope();
            let held_out = row
                .held_out
                .ok_or_else(|| invalid("C51 measured rank row lacks held-out metrics"))?
                .primary_envelope();
            if !selection.is_finite() || !held_out.is_finite() {
                return Err(invalid("C51 rank decision errors must be finite"));
            }
            Ok((selection, held_out))
        })
        .collect::<Result<Vec<_>>>()?;
    let selected_index = (1..rows.len()).find(|&index| {
        primary_errors[index].1 <= C51_PERFORMANCE_PRIMARY_TOLERANCE
            && primary_errors[index - 1].0 > C51_PERFORMANCE_PRIMARY_TOLERANCE
            && primary_errors[index..]
                .iter()
                .all(|(selection, _)| *selection <= C51_PERFORMANCE_PRIMARY_TOLERANCE)
    });
    let selected_index = selected_index.ok_or_else(|| {
        invalid(
            "C51 focused rank ladder has no 5% sustained selection pass with a lower failing bracket and held-out pass",
        )
    })?;
    let selected = rows[selected_index];
    let lower = rows[selected_index - 1];

    let compact = archive
        .compact_accuracy_rows
        .iter()
        .filter(|row| {
            row.refinement == refinement
                && row.alpha == 2
                && row.design == design
                && row.observation_count == observation_count
        })
        .collect::<Vec<_>>();
    if compact.len() != 1 || compact[0].status != "measured" {
        return Err(invalid(
            "C51 rank decision requires one measured projected Whitney-compact accuracy row on the identical design",
        ));
    }
    let compact = compact[0];
    let compact_selection = compact
        .selection
        .ok_or_else(|| invalid("C51 compact row lacks selection metrics"))?
        .primary_envelope();
    let compact_held_out = compact
        .held_out
        .ok_or_else(|| invalid("C51 compact row lacks held-out metrics"))?
        .primary_envelope();
    let calibrated_compact_selection = compact
        .calibrated_selection
        .ok_or_else(|| invalid("C51 calibrated compact row lacks selection metrics"))?
        .primary_envelope();
    let calibrated_compact_held_out = compact
        .calibrated_held_out
        .ok_or_else(|| invalid("C51 calibrated compact row lacks held-out metrics"))?
        .primary_envelope();
    if !compact_selection.is_finite()
        || !compact_held_out.is_finite()
        || compact_selection > C51_PERFORMANCE_PRIMARY_TOLERANCE
        || compact_held_out > C51_PERFORMANCE_PRIMARY_TOLERANCE
        || !calibrated_compact_selection.is_finite()
        || !calibrated_compact_held_out.is_finite()
    {
        return Err(invalid(
            "C51 projected Whitney-compact method does not satisfy the frozen 5% selection and held-out envelope",
        ));
    }
    if compact.method_performance_timing_eligible {
        return Err(invalid(
            "C51 compact accuracy evidence must remain timing ineligible",
        ));
    }

    Ok(C51PerformanceRankDecision {
        refinement,
        state_dimension: selected.state_dimension,
        scalar_observation_count: observation_count,
        alpha: 2,
        relative_tolerance: C51_PERFORMANCE_PRIMARY_TOLERANCE,
        sensor_design_id: C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID,
        output_design_id: C51_PERFORMANCE_OUTPUT_DESIGN_ID,
        selected_spectral_rank: selected.rank,
        lower_failing_rank: lower.rank,
        lower_failing_selection_primary_error: primary_errors[selected_index - 1].0,
        selection_primary_error: primary_errors[selected_index].0,
        held_out_primary_error: primary_errors[selected_index].1,
        validation_status: "heldout_validated",
        rank_status: if selected.rank == selected.state_dimension {
            "full_rank_required"
        } else {
            "low_rank"
        },
        compact_whitney_selection_primary_error: compact_selection,
        compact_whitney_held_out_primary_error: compact_held_out,
        calibrated_compact_whitney_selection_primary_error: calibrated_compact_selection,
        calibrated_compact_whitney_held_out_primary_error: calibrated_compact_held_out,
        compact_whitney_calibration_target_prior_rms: compact
            .calibration_target_prior_rms
            .ok_or_else(|| invalid("C51 compact row lacks its calibration target RMS"))?,
        compact_whitney_calibration_uncalibrated_prior_rms: compact
            .calibration_uncalibrated_prior_rms
            .ok_or_else(|| invalid("C51 compact row lacks its raw calibration RMS"))?,
        compact_whitney_calibration_precision_scale: compact
            .calibration_precision_scale
            .ok_or_else(|| invalid("C51 compact row lacks its calibration precision scale"))?,
        compact_whitney_calibration_tau_multiplier: compact
            .calibration_tau_multiplier
            .ok_or_else(|| invalid("C51 compact row lacks its calibration tau multiplier"))?,
        compact_whitney_validation_status: "heldout_validated",
        calibrated_compact_whitney_validation_status: if calibrated_compact_selection
            <= C51_PERFORMANCE_PRIMARY_TOLERANCE
            && calibrated_compact_held_out <= C51_PERFORMANCE_PRIMARY_TOLERANCE
        {
            "physical_rms_calibrated_heldout_validated"
        } else {
            "physical_rms_calibrated_measured_accuracy_context"
        },
        accuracy_archive_sha256,
        timing_basis_source: "partial_exact_requested_mixed_shell",
        shared_basis_timing_eligible: false,
    })
}

fn c51_accuracy_archive_sha256(directory: &Path) -> Result<String> {
    c51_named_files_sha256(
        directory,
        &C51_ACCURACY_ARCHIVE_HASH_FILES,
        "C51 accuracy archive",
    )
}

fn c51_named_files_sha256(directory: &Path, files: &[&str], label: &str) -> Result<String> {
    if !directory.is_dir() {
        return Err(invalid(format!(
            "{label} hash source must be a completed archive directory"
        )));
    }
    let mut digest = Sha256::new();
    for name in files {
        let path = directory.join(name);
        let bytes = fs::read(&path).map_err(|error| {
            FeecGmrfError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "could not hash required {label} file {}: {error}",
                    path.display()
                ),
            ))
        })?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(&bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn validate_c51_perf0_archive(directory: &Path) -> Result<()> {
    let rows_path = directory.join("rows.csv");
    let mut reader = csv::Reader::from_path(&rows_path).map_err(csv_error)?;
    let headers = reader.headers().map_err(csv_error)?.clone();
    let index = |name: &str| {
        headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| invalid(format!("C51 PERF-0 rows are missing column {name}")))
    };
    let method_index = index("method")?;
    let same_law_index = index("same_discrete_law_eligible")?;
    let observation_design_index = index("observation_design_id")?;
    let output_design_index = index("physical_qoi_design_id")?;
    let output_count_index = index("physical_qoi_output_count")?;
    let error_columns = [
        "physical_output_covariance_action_relative_error",
        "physical_variance_relative_error",
        "physical_transformed_variance_relative_error",
        "physical_mean_relative_error",
    ]
    .map(|name| {
        Ok((
            index(&format!("{name}_status"))?,
            index(&format!("{name}_value"))?,
        ))
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let mut exact_gp_rows = 0;
    for record in reader.records() {
        let record = record.map_err(csv_error)?;
        if record.get(method_index) != Some("exact_observation_gp") {
            continue;
        }
        exact_gp_rows += 1;
        if record.get(same_law_index) != Some("true")
            || record.get(observation_design_index) != Some(C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID)
            || record.get(output_design_index) != Some(C51_PERFORMANCE_OUTPUT_DESIGN_ID)
            || record.get(output_count_index) != Some("18")
        {
            return Err(invalid(
                "C51 PERF-0 exact-GP row does not use the frozen same-law H/G fixture",
            ));
        }
        for &(status_index, value_index) in &error_columns {
            if record.get(status_index) != Some("measured") {
                return Err(invalid(
                    "C51 PERF-0 exact-GP physical error is not measured",
                ));
            }
            let value = record
                .get(value_index)
                .ok_or_else(|| invalid("C51 PERF-0 exact-GP error value is missing"))?
                .parse::<f64>()
                .map_err(|error| invalid(format!("invalid C51 PERF-0 error: {error}")))?;
            if !value.is_finite() || value > 1.0e-8 {
                return Err(invalid(format!(
                    "C51 PERF-0 exact-GP physical error {value} exceeds 1e-8"
                )));
            }
        }
    }
    if exact_gp_rows != 1 {
        return Err(invalid(format!(
            "C51 PERF-0 requires exactly one exact-GP row, found {exact_gp_rows}"
        )));
    }
    Ok(())
}

#[cfg(any(feature = "external-solvers", test))]
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
    fn rank_extension_profiles_are_mesh_specific_post_primary_evidence() {
        let n8 = C51SpectralCovarianceRankConfig::rank_extension(8).unwrap();
        assert_eq!(n8.profile, C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8);
        assert_eq!(n8.refinements, [8]);
        assert_eq!(n8.ranks, C51_SPECTRAL_RANK_EXTENSION_N8_GRID);
        assert_eq!(n8.alphas, [MaternAlpha::Two]);
        assert_eq!(n8.expected_state_dimension(), Some(3_032));
        assert_eq!(n8.evidence_tier(), "rank_extension");
        assert!(n8.post_primary_rationale().is_some());

        let n10 = C51SpectralCovarianceRankConfig::rank_extension(10).unwrap();
        assert_eq!(n10.profile, C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10);
        assert_eq!(n10.refinements, [10]);
        assert_eq!(n10.ranks, C51_SPECTRAL_RANK_EXTENSION_N10_GRID);
        assert_eq!(n10.expected_state_dimension(), Some(6_130));
        assert_eq!(n10.evidence_tier(), "rank_extension");

        assert!(C51SpectralCovarianceRankConfig::rank_extension(6).is_err());
        let mut mutated = n8;
        mutated.ranks.pop();
        assert!(mutated.validate().is_err());
    }

    #[test]
    fn high_observation_profiles_freeze_counts_lower_brackets_and_dimension_independence() {
        let n8 = C51SpectralCovarianceRankConfig::high_observation_accuracy(8).unwrap();
        assert_eq!(n8.profile, C51_HIGH_OBSERVATION_PROFILE_N8);
        assert_eq!(n8.ranks, C51_HIGH_OBSERVATION_RANKS_N8);
        assert_eq!(n8.alphas, [MaternAlpha::Two]);
        assert_eq!(n8.evidence_tier(), "high_observation_accuracy");
        assert_eq!(
            n8.covariance_designs(),
            C51CovarianceDesign::HIGH_OBSERVATION
        );
        assert_eq!(n8.ranks[0], 512);
        assert_eq!(n8.ranks.last(), Some(&3_032));
        assert!(C51CovarianceDesign::HighM4096.observation_count(3_032) > 3_032);
        assert_eq!(
            C51CovarianceDesign::HIGH_OBSERVATION.map(|design| design.observation_count(17)),
            C51_HIGH_OBSERVATION_COUNTS
        );

        let n10 = C51SpectralCovarianceRankConfig::high_observation_accuracy(10).unwrap();
        assert_eq!(n10.ranks, C51_HIGH_OBSERVATION_RANKS_N10);
        assert_eq!(n10.ranks.last(), Some(&6_130));
        let mut mutated = n10;
        mutated.ranks.remove(0);
        assert!(mutated.validate().is_err());
        assert!(C51SpectralCovarianceRankConfig::high_observation_accuracy(6).is_err());
    }

    #[test]
    fn focused_high_observation_profiles_freeze_primary_and_confirmatory_cells() {
        for refinement in [8, 10] {
            let primary =
                C51SpectralCovarianceRankConfig::high_observation_primary(refinement).unwrap();
            assert_eq!(primary.high_observation_counts(), [2_048]);
            assert_eq!(
                primary.covariance_designs(),
                [C51CovarianceDesign::HighM2048]
            );
            assert_eq!(primary.ranks[0], 512);
            assert_eq!(primary.evidence_tier(), "high_observation_primary_accuracy");

            let confirmatory =
                C51SpectralCovarianceRankConfig::high_observation_confirmatory(refinement).unwrap();
            assert_eq!(confirmatory.high_observation_counts(), [4_096]);
            assert_eq!(
                confirmatory.covariance_designs(),
                [C51CovarianceDesign::HighM4096]
            );
            assert_eq!(confirmatory.ranks[0], 512);
            assert_eq!(
                confirmatory.evidence_tier(),
                "high_observation_confirmatory_accuracy"
            );

            let mut mutated = confirmatory;
            mutated.ranks.remove(0);
            assert!(mutated.validate().is_err());
        }
        assert!(C51SpectralCovarianceRankConfig::high_observation_primary(6).is_err());
        assert!(C51SpectralCovarianceRankConfig::high_observation_confirmatory(6).is_err());

        let extension =
            C51SpectralCovarianceRankConfig::high_observation_confirmatory_rank_extension_n10()
                .unwrap();
        assert_eq!(
            extension.profile,
            C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10
        );
        assert_eq!(
            extension.ranks,
            C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_N10_GRID
        );
        assert_eq!(extension.high_observation_counts(), [4_096]);
        assert_eq!(
            extension.covariance_designs(),
            [C51CovarianceDesign::HighM4096]
        );
        assert_eq!(
            extension.evidence_tier(),
            "high_observation_confirmatory_rank_extension"
        );
    }

    #[test]
    fn replay_profiles_are_accuracy_only_and_stop_at_cached_rank() {
        let n8 = C51SpectralCovarianceRankConfig::high_observation_replay(8).unwrap();
        assert_eq!(n8.profile, C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8);
        assert_eq!(n8.ranks, C51_HIGH_OBSERVATION_REPLAY_RANKS_N8);
        assert_eq!(n8.ranks.last(), Some(&3_032));
        assert_eq!(n8.high_observation_counts(), [2_048]);
        assert_eq!(n8.evidence_tier(), "high_observation_replay_accuracy_only");

        let n10 = C51SpectralCovarianceRankConfig::high_observation_replay(10).unwrap();
        assert_eq!(n10.profile, C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10);
        assert_eq!(n10.ranks, C51_HIGH_OBSERVATION_REPLAY_RANKS_N10);
        assert_eq!(n10.ranks.last(), Some(&4_096));
        assert_eq!(n10.high_observation_counts(), [2_048]);
        assert!(!n10.is_high_observation_primary());

        let confirmatory_n8 =
            C51SpectralCovarianceRankConfig::high_observation_confirmatory_replay(8).unwrap();
        assert_eq!(
            confirmatory_n8.profile,
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8
        );
        assert_eq!(confirmatory_n8.high_observation_counts(), [4_096]);
        assert_eq!(
            confirmatory_n8.covariance_designs(),
            [C51CovarianceDesign::HighM4096]
        );
        assert_eq!(confirmatory_n8.ranks.last(), Some(&3_032));

        let confirmatory_n10 =
            C51SpectralCovarianceRankConfig::high_observation_confirmatory_replay(10).unwrap();
        assert_eq!(
            confirmatory_n10.profile,
            C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10
        );
        assert_eq!(confirmatory_n10.high_observation_counts(), [4_096]);
        assert_eq!(confirmatory_n10.ranks.last(), Some(&4_096));

        let mut mutated = n10;
        mutated.ranks.push(4_608);
        assert!(mutated.validate().is_err());
        assert!(C51SpectralCovarianceRankConfig::high_observation_replay(6).is_err());
        assert!(C51SpectralCovarianceRankConfig::high_observation_confirmatory_replay(6).is_err());
    }

    #[test]
    fn focused_high_observation_manifest_binds_rank_decision_and_endpoint_semantics() {
        let path = std::env::temp_dir().join(format!(
            "c51-focused-high-observation-manifest-{}-{}.yaml",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let archive = C51SpectralCovarianceRankArchive {
            metadata: C51SpectralCovarianceRankMetadata::discovery("test"),
            config: C51SpectralCovarianceRankConfig::high_observation_primary(8).unwrap(),
            basis_attempts: Vec::new(),
            rows: Vec::new(),
            covariances: Vec::new(),
            exact_covariance_audits: Vec::new(),
            sensor_banks: Vec::new(),
            high_observation_designs: Vec::new(),
            compact_accuracy_rows: Vec::new(),
            plumbing_gates: Vec::new(),
        };
        archive.write_manifest(&path).unwrap();
        let manifest = std::fs::read_to_string(&path).unwrap();
        assert!(manifest.contains("performance_rank_decision_scope: primary_m2048_only"));
        assert!(manifest.contains("performance_rank_decision_observation_count: 2048"));
        assert!(manifest.contains("claim_bearing_physical_covariance_threshold: 0.05"));
        assert!(manifest.contains("sensitivity_physical_covariance_threshold: 0.02"));
        assert!(manifest.contains(
            "state_dimension_endpoint_semantics: full_rank_required_not_low_rank_or_truncated"
        ));

        let confirmatory = C51SpectralCovarianceRankArchive {
            config: C51SpectralCovarianceRankConfig::high_observation_confirmatory(10).unwrap(),
            ..archive
        };
        confirmatory.write_manifest(&path).unwrap();
        let manifest = std::fs::read_to_string(&path).unwrap();
        assert!(manifest.contains("performance_rank_decision_scope: confirmatory_m4096_only"));
        assert!(manifest.contains("performance_rank_decision_observation_count: 4096"));
        std::fs::remove_file(path).unwrap();
    }

    fn decision_test_metrics(error: f64) -> C51CovarianceErrorMetrics {
        C51CovarianceErrorMetrics {
            relative_frobenius: error,
            relative_spectral: error,
            maximum_relative_diagonal: error,
            maximum_relative_directional: error,
            relative_correlation_frobenius: error,
        }
    }

    fn decision_test_archive(selected_index: usize) -> C51SpectralCovarianceRankArchive {
        let config = C51SpectralCovarianceRankConfig::high_observation_primary(8).unwrap();
        let rows = config
            .ranks
            .iter()
            .enumerate()
            .map(|(index, &rank)| {
                let error = if index < selected_index { 0.06 } else { 0.04 };
                C51SpectralCovarianceRankRow {
                    refinement: 8,
                    state_dimension: 3_032,
                    alpha: 2,
                    design: C51CovarianceDesign::HighM2048,
                    observation_count: 2_048,
                    observation_density: 2_048.0 / 3_032.0,
                    rank,
                    shared_maximum_basis_rank: 3_032,
                    rank_fraction: rank as f64 / 3_032.0,
                    status: "measured".to_string(),
                    censored_stage: String::new(),
                    censored_reason: String::new(),
                    selection: Some(decision_test_metrics(error)),
                    held_out: Some(decision_test_metrics(0.04)),
                    latent_mass_action_relative_errors: Some([0.01, 0.01]),
                    latent_mass_action_maximum_relative_error: Some(0.01),
                    basis_storage_bytes: 0,
                    prepared_storage_bytes: None,
                    method_performance_timing_eligible: false,
                }
            })
            .collect();
        C51SpectralCovarianceRankArchive {
            metadata: C51SpectralCovarianceRankMetadata::discovery("test"),
            config,
            basis_attempts: Vec::new(),
            rows,
            covariances: Vec::new(),
            exact_covariance_audits: Vec::new(),
            sensor_banks: Vec::new(),
            high_observation_designs: Vec::new(),
            compact_accuracy_rows: vec![C51CompactAccuracyRow {
                refinement: 8,
                state_dimension: 3_032,
                alpha: 2,
                design: C51CovarianceDesign::HighM2048,
                observation_count: 2_048,
                status: "measured".to_string(),
                unavailable_reason: String::new(),
                selection: Some(decision_test_metrics(0.03)),
                held_out: Some(decision_test_metrics(0.04)),
                calibration_target_prior_rms: Some(1.0),
                calibration_uncalibrated_prior_rms: Some(0.8),
                calibration_precision_scale: Some(0.64),
                calibration_tau_multiplier: Some(0.8),
                calibrated_selection: Some(decision_test_metrics(0.02)),
                calibrated_held_out: Some(decision_test_metrics(0.03)),
                latent_mass_action_relative_errors: Some([0.01, 0.01]),
                latent_mass_action_maximum_relative_error: Some(0.01),
                method_performance_timing_eligible: false,
            }],
            plumbing_gates: Vec::new(),
        }
    }

    #[test]
    fn performance_rank_decision_requires_bracket_and_classifies_full_rank() {
        let low_rank = decision_test_archive(4);
        let decision = derive_performance_rank_decision(&low_rank, "a".repeat(64)).unwrap();
        assert_eq!(decision.selected_spectral_rank, low_rank.config.ranks[4]);
        assert_eq!(decision.lower_failing_rank, low_rank.config.ranks[3]);
        assert_eq!(decision.rank_status, "low_rank");
        assert_eq!(decision.validation_status, "heldout_validated");
        assert_eq!(
            decision.output_design_id,
            "rank_study_selection_and_held_out_wendland_c2_18"
        );
        assert_eq!(
            decision.compact_whitney_validation_status,
            "heldout_validated"
        );
        assert_eq!(
            decision.calibrated_compact_whitney_validation_status,
            "physical_rms_calibrated_heldout_validated"
        );
        assert!(!decision.shared_basis_timing_eligible);

        let full_rank = decision_test_archive(C51_HIGH_OBSERVATION_RANKS_N8.len() - 1);
        let decision = derive_performance_rank_decision(&full_rank, "b".repeat(64)).unwrap();
        assert_eq!(decision.selected_spectral_rank, 3_032);
        assert_eq!(decision.rank_status, "full_rank_required");
    }

    #[test]
    fn performance_rank_decision_rejects_censor_heldout_compact_and_missing_bracket() {
        let mut censored = decision_test_archive(4);
        censored.rows[2].status = "censored".to_string();
        assert!(derive_performance_rank_decision(&censored, "a".repeat(64)).is_err());

        let mut held_out = decision_test_archive(4);
        held_out.rows[4].held_out = Some(decision_test_metrics(0.051));
        assert!(derive_performance_rank_decision(&held_out, "a".repeat(64)).is_err());

        let mut compact = decision_test_archive(4);
        compact.compact_accuracy_rows[0].selection = Some(decision_test_metrics(0.051));
        assert!(derive_performance_rank_decision(&compact, "a".repeat(64)).is_err());

        let no_bracket = decision_test_archive(0);
        assert!(derive_performance_rank_decision(&no_bracket, "a".repeat(64)).is_err());
    }

    #[test]
    fn canonical_accuracy_archive_digest_is_order_independent_and_content_sensitive() {
        let directory = std::env::temp_dir().join(format!(
            "c51-accuracy-digest-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir_all(&directory).unwrap();
        for name in C51_ACCURACY_ARCHIVE_HASH_FILES.into_iter().rev() {
            std::fs::write(directory.join(name), name.as_bytes()).unwrap();
        }
        let first = c51_accuracy_archive_sha256(&directory).unwrap();
        let second = c51_accuracy_archive_sha256(&directory).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        std::fs::write(directory.join("rows.csv"), b"changed").unwrap();
        assert_ne!(first, c51_accuracy_archive_sha256(&directory).unwrap());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rank_extension_manifest_names_alpha_and_post_primary_tier_truthfully() {
        let path = std::env::temp_dir().join(format!(
            "c51-rank-extension-manifest-{}-{}.yaml",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let archive = C51SpectralCovarianceRankArchive {
            metadata: C51SpectralCovarianceRankMetadata::discovery("test"),
            config: C51SpectralCovarianceRankConfig::rank_extension(10).unwrap(),
            basis_attempts: Vec::new(),
            rows: Vec::new(),
            covariances: Vec::new(),
            exact_covariance_audits: Vec::new(),
            sensor_banks: Vec::new(),
            high_observation_designs: Vec::new(),
            compact_accuracy_rows: Vec::new(),
            plumbing_gates: Vec::new(),
        };
        archive.write_manifest(&path).unwrap();
        let manifest = std::fs::read_to_string(&path).unwrap();
        assert!(manifest.contains("profile: rank-extension-v1-n10"));
        assert!(manifest.contains("evidence_tier: rank_extension"));
        assert!(manifest.contains("post_primary: true"));
        assert!(manifest.contains("post_primary_rationale: \"post-primary extension"));
        assert!(manifest.contains("alpha_values: [2]"));
        assert!(manifest.contains("expected_state_dimension: 6130"));
        assert!(manifest.contains("method_performance_timing_eligible: false"));
        std::fs::remove_file(path).unwrap();
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
        let kernels = primary_observation_kernels(0.16).unwrap();
        assert_eq!(kernels.len() * 3, C51_PRIMARY_SCALAR_OBSERVATION_COUNT);
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
            exact_covariance_audits: Vec::new(),
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
            high_observation_designs: Vec::new(),
            compact_accuracy_rows: Vec::new(),
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
            exact_covariance_audits: Vec::new(),
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
            high_observation_designs: Vec::new(),
            compact_accuracy_rows: Vec::new(),
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

    #[test]
    fn exact_covariance_audit_csv_records_interim_policy_and_raw_defect() {
        let directory =
            std::env::temp_dir().join(format!("c51-exact-covariance-audit-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let archive = C51SpectralCovarianceRankArchive {
            metadata: C51SpectralCovarianceRankMetadata::discovery("test"),
            config: C51SpectralCovarianceRankConfig::smoke(),
            basis_attempts: Vec::new(),
            rows: Vec::new(),
            covariances: Vec::new(),
            exact_covariance_audits: vec![C51ExactCovarianceAuditRecord {
                refinement: 8,
                alpha: 3,
                design: C51CovarianceDesign::Prior,
                bank: "selection",
                policy_version: INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION,
                status: "accepted_interim_scale_relative_and_projected",
                residual_aware: false,
                residual_status:
                    "unavailable: gmrf-core KKT covariance-action residuals are not exposed",
                projection_applied: true,
                dimension: OUTPUT_DIMENSION,
                raw_scale: 1.369_730_048_8e-3,
                raw_maximum_absolute_skew: 4.944_989e-14,
                raw_maximum_relative_skew: 3.610_19e-11,
                raw_maximum_skew_row: 4,
                raw_maximum_skew_column: 5,
                relative_skew_limit: INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT,
                projection_maximum_absolute_correction: 2.472_494_5e-14,
                projection_frobenius_correction: 3.496_636_3e-14,
            }],
            sensor_banks: Vec::new(),
            high_observation_designs: Vec::new(),
            compact_accuracy_rows: Vec::new(),
            plumbing_gates: Vec::new(),
        };
        let path = directory.join("exact_covariance_audits.csv");
        archive.write_exact_covariance_audits(&path).unwrap();
        let csv = std::fs::read_to_string(path).unwrap();
        assert!(csv.contains(INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION));
        assert!(csv.contains("8,3,prior,selection"));
        assert!(csv.contains("false,unavailable:"));
        assert!(csv.contains("4.944989000000000"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
