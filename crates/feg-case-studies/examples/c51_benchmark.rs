use feg_case_studies::experimental::c51_benchmark::{
    run_c51_benchmark, C51BenchmarkConfig, C51RunMetadata,
};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/c51-benchmark-smoke"));
    let archive = run_c51_benchmark(
        &C51BenchmarkConfig::smoke(),
        C51RunMetadata::smoke("cargo run --release -p feg-case-studies --features experimental --example c51_benchmark"),
    )?;
    archive.write_directory(&output)?;
    println!(
        "wrote {} benchmark rows to {}",
        archive.rows.len(),
        output.display()
    );
    Ok(())
}
