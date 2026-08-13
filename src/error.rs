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
