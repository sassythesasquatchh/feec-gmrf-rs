use feg_case_studies::c51_spectral_covariance_rank::{
    run_c51_spectral_covariance_rank, C51SpectralCovarianceRankConfig,
    C51SpectralCovarianceRankMetadata,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let mut profile = "smoke".to_string();
    let mut output = PathBuf::from("out/c51_spectral_covariance_rank");
    let mut refinement = None;
    let mut maximum_rank = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--profile" => {
                index += 1;
                profile = arguments
                    .get(index)
                    .ok_or("--profile requires a value")?
                    .clone();
            }
            "--output" => {
                index += 1;
                output = PathBuf::from(arguments.get(index).ok_or("--output requires a value")?);
            }
            "--refinement" => {
                index += 1;
                refinement = Some(
                    arguments
                        .get(index)
                        .ok_or("--refinement requires a value")?
                        .parse::<usize>()?,
                );
            }
            "--maximum-rank" => {
                index += 1;
                maximum_rank = Some(
                    arguments
                        .get(index)
                        .ok_or("--maximum-rank requires a value")?
                        .parse::<usize>()?,
                );
            }
            flag => return Err(format!("unknown argument {flag}").into()),
        }
        index += 1;
    }

    let config = match profile.as_str() {
        "smoke" => {
            if refinement.is_some_and(|value| value != 4) || maximum_rank.is_some() {
                return Err(
                    "smoke has frozen refinement 4 and does not accept --maximum-rank".into(),
                );
            }
            C51SpectralCovarianceRankConfig::smoke()
        }
        "publication-discovery-v1" => {
            let refinement = refinement.ok_or(
                "publication runs require --refinement 8 or 10; run n=8 and inspect it before n=10",
            )?;
            let mut config = C51SpectralCovarianceRankConfig::publication()
                .for_publication_refinement(refinement)?;
            if let Some(maximum_rank) = maximum_rank {
                config = config.with_maximum_rank(maximum_rank)?;
            }
            config
        }
        _ => return Err(format!("unknown C51 spectral covariance profile {profile}").into()),
    };
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metadata =
        C51SpectralCovarianceRankMetadata::from_repository(arguments.join(" "), &repository_root)?;
    let archive = run_c51_spectral_covariance_rank(&config, metadata)?;
    archive.write_directory(&output)?;
    println!(
        "wrote {} rows for {} to {}",
        archive.rows.len(),
        config.profile,
        output.display()
    );
    Ok(())
}
