use feg_case_studies::experimental::c51_benchmark::{
    run_c51_benchmark, run_c51_resource, run_c51_spectral_rank_pilot, C51BenchmarkConfig,
    C51MethodId, C51ObservationClass, C51ResourceConfig, C51ResourcePhase, C51ResourceSelection,
    C51ResourceSensorTarget, C51RunMetadata,
};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|value| value == "resource") {
        return run_resource(&arguments[1..]);
    }
    run_accuracy(&arguments)
}

fn run_accuracy(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let profile = arguments
        .first()
        .cloned()
        .unwrap_or_else(|| "smoke".to_string());
    let config = match profile.as_str() {
        "smoke" => C51BenchmarkConfig::smoke(),
        "reference_ladder" => C51BenchmarkConfig::reference_ladder(),
        "spectral_rank_pilot" => C51BenchmarkConfig::spectral_rank_pilot(),
        other => {
            return Err(format!(
                "unknown C51 profile {other:?}; use smoke, reference_ladder, spectral_rank_pilot, or resource"
            )
            .into())
        }
    };
    let output = arguments
        .get(1)
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

fn run_resource(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if arguments.len() != 10 {
        return Err(
            "resource usage: c51_benchmark resource <profile> <output> <refinement> <observation-class> <count|density> <target> <D> <method> <rank|none> <cold_setup|query_full_uq>"
                .into(),
        );
    }
    let profile = &arguments[0];
    if profile != "smoke" {
        return Err(format!("resource profile {profile:?} is unavailable; use smoke").into());
    }
    let output = PathBuf::from(&arguments[1]);
    let refinement = parse_usize("refinement", &arguments[2])?;
    let observation_class = match arguments[3].as_str() {
        "local_physical_b" => C51ObservationClass::LocalPhysicalB,
        "local_selectors" => C51ObservationClass::LocalSelectors,
        "global_row" => C51ObservationClass::GlobalRow,
        value => return Err(format!("unknown C51 observation class {value:?}").into()),
    };
    let sensor_target = match arguments[4].as_str() {
        "count" => C51ResourceSensorTarget::Count(parse_usize("sensor count", &arguments[5])?),
        "density" => C51ResourceSensorTarget::Density(arguments[5].parse::<f64>()?),
        value => return Err(format!("unknown C51 sensor target kind {value:?}").into()),
    };
    let repeated_data_count = parse_usize("repeated data count", &arguments[6])?;
    let method = match arguments[7].as_str() {
        "fm_direct" => C51MethodId::FullMetricDirect,
        "fm_aug" => C51MethodId::FullMetricAugmented,
        "whitney_compact" => C51MethodId::WhitneyCompact,
        "spectral_full_rank" => C51MethodId::SpectralFullRank,
        "spectral_low_rank" => C51MethodId::SpectralLowRank,
        value => return Err(format!("unknown C51 method {value:?}").into()),
    };
    let spectral_rank = match arguments[8].as_str() {
        "none" => None,
        value => Some(parse_usize("spectral rank", value)?),
    };
    let phase = match arguments[9].as_str() {
        "cold_setup" => C51ResourcePhase::ColdSetup,
        "query_full_uq" => C51ResourcePhase::QueryFullUq,
        value => return Err(format!("unknown C51 resource phase {value:?}").into()),
    };
    let selection = C51ResourceSelection {
        refinement,
        observation_class,
        sensor_target,
        repeated_data_count,
        method,
        spectral_rank,
        phase,
    };
    let config = C51ResourceConfig::smoke(selection);
    let mut metadata = C51RunMetadata::smoke(std::env::args().collect::<Vec<_>>().join(" "));
    metadata.profile = format!("resource/{profile}");
    let archive = run_c51_resource(&config, metadata)?;
    archive.write_directory(&output)?;
    println!(
        "wrote one method-owned resource row to {}",
        output.display()
    );
    Ok(())
}

fn parse_usize(name: &str, value: &str) -> Result<usize, Box<dyn Error>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}").into())
}
