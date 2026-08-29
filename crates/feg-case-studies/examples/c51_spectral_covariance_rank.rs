use feg_case_studies::c51_spectral_covariance_rank::{
    run_c51_spectral_covariance_rank, run_c51_spectral_covariance_rank_replay,
    C51SpectralCovarianceRankConfig, C51SpectralCovarianceRankMetadata,
    C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10, C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8,
    C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10,
    C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10,
    C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8, C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10,
    C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8, C51_HIGH_OBSERVATION_PROFILE_N10,
    C51_HIGH_OBSERVATION_PROFILE_N8, C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10,
    C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8, C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10,
    C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let mut profile = "smoke".to_string();
    let mut output = PathBuf::from("out/c51_spectral_covariance_rank");
    let mut refinement = None;
    let mut maximum_rank = None;
    let mut rank_decision_output = None;
    let mut perf0_archive = None;
    let mut basis_workspace = None;
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
            "--rank-decision-output" => {
                index += 1;
                rank_decision_output = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or("--rank-decision-output requires a value")?,
                ));
            }
            "--perf0-archive" => {
                index += 1;
                perf0_archive = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or("--perf0-archive requires a value")?,
                ));
            }
            "--basis-workspace" => {
                index += 1;
                basis_workspace = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or("--basis-workspace requires a value")?,
                ));
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
        C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8 | C51_SPECTRAL_RANK_EXTENSION_PROFILE_N10 => {
            let expected_refinement = if profile == C51_SPECTRAL_RANK_EXTENSION_PROFILE_N8 {
                8
            } else {
                10
            };
            let refinement = refinement.ok_or_else(|| {
                format!(
                    "{profile} requires --refinement {expected_refinement}; the mesh-specific extension profile is immutable"
                )
            })?;
            if refinement != expected_refinement {
                return Err(format!(
                    "{profile} requires --refinement {expected_refinement}, received {refinement}"
                )
                .into());
            }
            if maximum_rank.is_some() {
                return Err(format!(
                    "{profile} has an immutable rank grid and rejects --maximum-rank"
                )
                .into());
            }
            C51SpectralCovarianceRankConfig::rank_extension(refinement)?
        }
        C51_HIGH_OBSERVATION_PROFILE_N8 | C51_HIGH_OBSERVATION_PROFILE_N10 => {
            let expected_refinement = if profile == C51_HIGH_OBSERVATION_PROFILE_N8 {
                8
            } else {
                10
            };
            let refinement = refinement
                .ok_or_else(|| format!("{profile} requires --refinement {expected_refinement}"))?;
            if refinement != expected_refinement {
                return Err(format!(
                    "{profile} requires --refinement {expected_refinement}, received {refinement}"
                )
                .into());
            }
            if maximum_rank.is_some() {
                return Err(format!(
                    "{profile} has immutable observation and rank ladders and rejects --maximum-rank"
                )
                .into());
            }
            C51SpectralCovarianceRankConfig::high_observation_accuracy(refinement)?
        }
        C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8 | C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N10 => {
            let expected_refinement = if profile == C51_HIGH_OBSERVATION_PRIMARY_PROFILE_N8 {
                8
            } else {
                10
            };
            validate_focused_profile_arguments(
                &profile,
                refinement,
                expected_refinement,
                maximum_rank,
            )?;
            C51SpectralCovarianceRankConfig::high_observation_primary(expected_refinement)?
        }
        C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8
        | C51_HIGH_OBSERVATION_REPLAY_PROFILE_N10
        | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8
        | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10 => {
            let expected_refinement = if matches!(
                profile.as_str(),
                C51_HIGH_OBSERVATION_REPLAY_PROFILE_N8
                    | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8
            ) {
                8
            } else {
                10
            };
            validate_focused_profile_arguments(
                &profile,
                refinement,
                expected_refinement,
                maximum_rank,
            )?;
            if basis_workspace.is_none() {
                return Err(format!("{profile} requires --basis-workspace").into());
            }
            if rank_decision_output.is_some() || perf0_archive.is_some() {
                return Err("a replay profile cannot emit a performance rank decision".into());
            }
            if matches!(
                profile.as_str(),
                C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N8
                    | C51_HIGH_OBSERVATION_CONFIRMATORY_REPLAY_PROFILE_N10
            ) {
                C51SpectralCovarianceRankConfig::high_observation_confirmatory_replay(
                    expected_refinement,
                )?
            } else {
                C51SpectralCovarianceRankConfig::high_observation_replay(expected_refinement)?
            }
        }
        C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8
        | C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N10 => {
            let expected_refinement = if profile == C51_HIGH_OBSERVATION_CONFIRMATORY_PROFILE_N8 {
                8
            } else {
                10
            };
            validate_focused_profile_arguments(
                &profile,
                refinement,
                expected_refinement,
                maximum_rank,
            )?;
            C51SpectralCovarianceRankConfig::high_observation_confirmatory(expected_refinement)?
        }
        C51_HIGH_OBSERVATION_CONFIRMATORY_RANK_EXTENSION_PROFILE_N10 => {
            validate_focused_profile_arguments(&profile, refinement, 10, maximum_rank)?;
            if basis_workspace.is_some()
                || rank_decision_output.is_some()
                || perf0_archive.is_some()
            {
                return Err(
                    "the m=4096 n=10 rank extension accepts no replay or rank-decision artifacts"
                        .into(),
                );
            }
            C51SpectralCovarianceRankConfig::high_observation_confirmatory_rank_extension_n10()?
        }
        _ => return Err(format!("unknown C51 spectral covariance profile {profile}").into()),
    };
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metadata =
        C51SpectralCovarianceRankMetadata::from_repository(arguments.join(" "), &repository_root)?;
    let archive = match basis_workspace.as_deref() {
        Some(workspace) => run_c51_spectral_covariance_rank_replay(&config, metadata, workspace)?,
        None => run_c51_spectral_covariance_rank(&config, metadata)?,
    };
    archive.write_directory(&output)?;
    match (rank_decision_output, perf0_archive) {
        (Some(decision_output), Some(perf0_archive)) => {
            archive.write_performance_rank_decision(&output, &perf0_archive, &decision_output)?;
            println!("wrote rank decision to {}", decision_output.display());
        }
        (None, None) => {}
        _ => {
            return Err(
                "--rank-decision-output and --perf0-archive must be supplied together".into(),
            )
        }
    }
    println!(
        "wrote {} rows for {} to {}",
        archive.rows.len(),
        config.profile,
        output.display()
    );
    Ok(())
}

fn validate_focused_profile_arguments(
    profile: &str,
    refinement: Option<usize>,
    expected_refinement: usize,
    maximum_rank: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let refinement = refinement
        .ok_or_else(|| format!("{profile} requires --refinement {expected_refinement}"))?;
    if refinement != expected_refinement {
        return Err(format!(
            "{profile} requires --refinement {expected_refinement}, received {refinement}"
        )
        .into());
    }
    if maximum_rank.is_some() {
        return Err(format!(
            "{profile} has immutable observation and rank ladders and rejects --maximum-rank"
        )
        .into());
    }
    Ok(())
}
