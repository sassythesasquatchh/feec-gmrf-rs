use std::path::Path;

pub fn execute(run_directory: &Path, against: &str) -> Result<(), String> {
    crate::output::verify_run_manifest(run_directory, against)?;
    println!("verified {} against {against}", run_directory.display());
    Ok(())
}
