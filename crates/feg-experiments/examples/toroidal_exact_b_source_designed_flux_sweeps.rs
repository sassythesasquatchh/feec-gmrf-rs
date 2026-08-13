fn main() -> Result<(), Box<dyn std::error::Error>> {
    use feg_case_studies::toroidal_exact_b_sweeps::{
        run_toroidal_exact_b_sweeps, ToroidalExactBSweepKind, ToroidalExactBSweepProfile,
    };
    run_toroidal_exact_b_sweeps(
        "out/examples/toroidal_exact_b_source_designed_flux_sweeps",
        ToroidalExactBSweepProfile::ThesisSubmitted,
        ToroidalExactBSweepKind::Both,
    )?;
    Ok(())
}
