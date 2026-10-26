use feg_case_studies::c51_one_form_pde_uq::run_with_observables;
use std::{error::Error, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=7).contains(&args.len()) {
        return Err("usage: c51_one_form_pde_uq <compact|supported|gp|spectral-RANK|pde> <even-refinement> <pde-sigma> <exact|mc|none|bounds> <output-directory> [fixed-precision-scale] [discrete|continuum|local] (gp: none and refinement <=8; spectral: none and refinement <=12, rank <=256 above refinement 8; bounds: compact/supported and refinement <=6; continuum/local require a fixed scale)".into());
    }
    // The development comparison uses one computation thread per process.
    faer::set_global_parallelism(faer::Par::Seq);
    if args
        .get(6)
        .is_some_and(|s| !matches!(s.as_str(), "discrete" | "continuum" | "local"))
    {
        return Err("workload must be discrete, continuum or local".into());
    }
    run_with_observables(
        &args[0],
        args[1].parse()?,
        args[2].parse()?,
        &args[3],
        Path::new(&args[4]),
        args.get(5).map(|v| v.parse()).transpose()?,
        args.get(6).is_some_and(|s| s != "discrete"),
        args.get(6).is_some_and(|s| s == "local"),
    )?;
    Ok(())
}
