use feg_case_studies::experimental::c51_benchmark::{
    run_c51_benchmark, C51BenchmarkConfig, C51RunMetadata,
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
        other => {
            return Err(
                format!("unknown C51 profile {other:?}; use smoke or reference_ladder").into(),
            )
        }
    };
    let output = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("out/c51-benchmark-{profile}")));
    let mut metadata = C51RunMetadata::smoke(format!(
        "cargo run --release -p feg-case-studies --features experimental --example c51_benchmark -- {profile}"
    ));
    metadata.profile = profile;
    let archive = run_c51_benchmark(&config, metadata)?;
    archive.write_directory(&output)?;
    println!(
        "wrote {} benchmark rows to {}",
        archive.rows.len(),
        output.display()
    );
    Ok(())
}
