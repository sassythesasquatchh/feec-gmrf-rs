use feg_case_studies::experimental::c51_benchmark::C51RunMetadata;
use feg_case_studies::experimental::c51_thesis_normalized_compact::{
    run_c51_thesis_normalized_compact, C51ThesisNormalizedCompactConfig,
};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() != 2 {
        return Err("usage: c51_thesis_normalized_compact <8|10> <output-directory>".into());
    }
    let refinement = arguments[0].parse::<usize>()?;
    let output = PathBuf::from(&arguments[1]);
    let config = C51ThesisNormalizedCompactConfig::publication(refinement)?;
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut metadata = C51RunMetadata::from_repository(
        std::env::args().collect::<Vec<_>>().join(" "),
        &repository_root,
    )?;
    metadata.profile = config.profile.clone();
    let report = run_c51_thesis_normalized_compact(&config, metadata)?;
    report.write_directory(&output)?;
    println!(
        "wrote thesis-normalized C51 compact accuracy to {}",
        output.display()
    );
    Ok(())
}
