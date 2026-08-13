//! Stable Cargo target for opt-in external-reference checks.
//!
//! Detailed external-reference assertions live beside the relevant workflows
//! under `#[cfg(feature = "external-reference-tests")]`.

#[test]
fn external_reference_target_is_feature_gated() {
    assert!(cfg!(feature = "external-reference-tests"));
}
