//! Stable Cargo target for opt-in TEAM 13 workflow checks.
//!
//! The detailed heavy assertions live beside the workflow implementation under
//! `#[cfg(feature = "heavy-tests")]` and reuse its private setup helpers.

#[test]
fn heavy_team13_target_is_feature_gated() {
    assert!(cfg!(feature = "heavy-tests"));
}
