//! Stable Cargo target for opt-in toroidal workflow checks.
//!
//! The detailed heavy assertions live beside the workflow implementation under
//! `#[cfg(feature = "heavy-tests")]`, where they can reuse private setup
//! helpers without duplicating case-study orchestration.

#[test]
fn heavy_toroidal_target_is_feature_gated() {
    assert!(cfg!(feature = "heavy-tests"));
}
