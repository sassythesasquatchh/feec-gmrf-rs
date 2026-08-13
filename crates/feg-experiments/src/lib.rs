//! Exploratory and historical FEEC--GMRF workflows.
//!
//! Build this crate explicitly to run research prototypes and earlier
//! entrypoints. Workflows move to `feg-case-studies` once their reusable
//! mathematics is in the parent API and they have publication profiles and
//! regression tests.

/// Stability marker for downstream tooling.
pub const STABILITY: &str = "experimental";

/// Research prototypes and earlier workflow entrypoints.
pub use feg_case_studies::experimental;
