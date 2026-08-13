//! Exploratory and historical FEEC--GMRF workflows.
//!
//! This crate is intentionally excluded from the default workspace members.
//! A workflow is promoted to `feg-case-studies` only after its reusable
//! mathematics has moved into the parent API and publication profiles and
//! regression tests have been added.

/// Stability marker for downstream tooling.
pub const STABILITY: &str = "experimental";

/// Historical workflows retained outside the supported default build.
pub use feg_case_studies::experimental;
