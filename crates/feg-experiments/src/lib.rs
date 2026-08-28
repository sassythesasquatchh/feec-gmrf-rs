//! Exploratory FEEC--GMRF numerical programs.
//!
//! Build this crate explicitly to run research prototypes and earlier
//! entrypoints. Workflows move to `feg-case-studies` once their reusable
//! mathematics is in the parent API and they have publication profiles and
//! regression tests.

/// Crate version marker for tools that enumerate experiment packages.
pub const STABILITY: &str = "experimental";

/// Research programs available as example entry points.
pub use feg_case_studies::experimental;
