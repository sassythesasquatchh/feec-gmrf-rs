//! Errors returned by the public FEEC--GMRF API.

use thiserror::Error;

/// Error returned while validating, assembling, or solving a FEEC--GMRF model.
#[derive(Debug, Error)]
pub enum FeecGmrfError {
    /// Matrix, vector, or form-space dimensions are inconsistent.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// A model parameter is non-finite or outside its mathematical domain.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// A matrix entry is non-finite, including after duplicate entries are summed.
    #[error("{matrix} contains a non-finite entry at ({row}, {column})")]
    NonFiniteMatrixEntry {
        matrix: &'static str,
        row: usize,
        column: usize,
    },
    /// A matrix declared to be symmetric differs from its transpose beyond roundoff.
    #[error(
        "{matrix} is not symmetric at ({row}, {column}): absolute transpose difference \
         {max_abs_difference:.6e} exceeds the scale-aware tolerance {tolerance:.6e}"
    )]
    MatrixNotSymmetric {
        matrix: &'static str,
        row: usize,
        column: usize,
        max_abs_difference: f64,
        tolerance: f64,
    },
    /// A precision matrix failed an unshifted sparse Cholesky factorization.
    #[error("{matrix} is not positive definite: {reason}")]
    PrecisionNotPositiveDefinite {
        matrix: &'static str,
        reason: String,
    },
    /// A deliberately materialized exact reference exceeded its configured guard.
    #[error(
        "exact eliminated reference requires dimension {dimension}, exceeding configured maximum {maximum}"
    )]
    ExactReferenceSizeExceeded { dimension: usize, maximum: usize },
    /// A proper Gaussian API was asked to construct an intrinsic or improper law.
    #[error("intrinsic model is unsupported by this proper-prior API: {0}")]
    IntrinsicModelUnsupported(String),
    /// FEEC assembly failed.
    #[error("FEEC assembly failed: {0}")]
    Assembly(String),
    /// Gaussian model construction or inference failed.
    #[error("GMRF inference failed: {0}")]
    Inference(String),
    /// A requested capability is not available for the supplied operators.
    #[error("unsupported model: {0}")]
    Unsupported(String),
    /// Reading or writing a model artifact failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl From<gmrf_core::GmrfError> for FeecGmrfError {
    fn from(value: gmrf_core::GmrfError) -> Self {
        Self::Inference(value.to_string())
    }
}

/// Result type used by the public FEEC--GMRF API.
pub type Result<T> = std::result::Result<T, FeecGmrfError>;
