use feg_case_studies::experimental::c51_benchmark::{
    run_c51_benchmark, run_c51_spectral_rank_pilot, C51BenchmarkConfig, C51RunMetadata,
};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let profile = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "smoke".to_string());
    let config = match profile.as_str() {
        "smoke" => C51BenchmarkConfig::smoke(),
        "reference_ladder" => C51BenchmarkConfig::reference_ladder(),
        "spectral_rank_pilot" => C51BenchmarkConfig::spectral_rank_pilot(),
        other => {
            return Err(format!(
                "unknown C51 profile {other:?}; use smoke, reference_ladder, or spectral_rank_pilot"
            )
            .into())
        }
    };
    let output = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("out/c51-benchmark-{profile}")));
    let enabled_features = if profile == "spectral_rank_pilot" {
        "experimental,external-solvers"
    } else {
        "experimental"
    };
    let mut metadata = C51RunMetadata::smoke(format!(
        "cargo run --release -p feg-case-studies --features {enabled_features} --example c51_benchmark -- {profile}"
    ));
    metadata.profile = profile;
    let archive = if metadata.profile == "spectral_rank_pilot" {
        run_c51_spectral_rank_pilot(&config, metadata)?
    } else {
        run_c51_benchmark(&config, metadata)?
    };
    archive.write_directory(&output)?;
    println!(
        "wrote {} benchmark rows to {}",
        archive.rows.len(),
        output.display()
    );
    Ok(())
}
