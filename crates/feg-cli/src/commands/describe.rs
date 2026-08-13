pub fn execute(study_id: &str) -> Result<(), String> {
    let study = feg_case_studies::study::find_published_study(study_id)
        .ok_or_else(|| format!("unknown published study `{study_id}`"))?;
    println!(
        "{}\n\nFamily: {}\n{}",
        study.id, study.family, study.summary
    );
    println!("Profiles: {}", study.profiles.join(", "));
    let custom_keys = feg_case_studies::study::custom_configuration_keys(study_id);
    if !custom_keys.is_empty() {
        println!("Custom configuration keys: {}", custom_keys.join(", "));
    }
    if !study.requirements.is_empty() {
        println!("Requirements: {}", study.requirements.join(", "));
    }
    Ok(())
}
