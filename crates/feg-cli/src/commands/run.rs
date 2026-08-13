use std::path::Path;

pub fn execute(
    study_id: &str,
    configuration: &crate::args::RunConfiguration,
    output: &Path,
) -> Result<(), String> {
    use feg_case_studies::study::{CustomStudyConfiguration, StudyRunProfile};

    let study = feg_case_studies::study::find_published_study(study_id)
        .ok_or_else(|| format!("unknown published study `{study_id}`"))?;
    let custom;
    let (profile, config_path) = match configuration {
        crate::args::RunConfiguration::Profile(profile) => {
            if !study.profiles.contains(&profile.as_str()) {
                return Err(format!(
                    "study `{study_id}` does not define profile `{profile}`"
                ));
            }
            (StudyRunProfile::Named(profile), None)
        }
        crate::args::RunConfiguration::Custom(path) => {
            custom = CustomStudyConfiguration::from_path(path, study_id)?;
            (StudyRunProfile::Custom(&custom), Some(path.as_path()))
        }
    };
    crate::prerequisites::check(study.requirements)?;
    std::fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let provenance = crate::provenance::Provenance::capture(
        study_id,
        profile.label(),
        study.inputs,
        config_path,
    )?;
    let started = std::time::Instant::now();
    let metrics = (study.run)(study_id, &profile, output)?;
    crate::output::write_run_manifest(output, &provenance, &metrics, started.elapsed())?;
    println!(
        "completed {study_id} ({}) in {}",
        profile.label(),
        output.display()
    );
    Ok(())
}
