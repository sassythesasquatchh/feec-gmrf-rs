pub fn execute() -> Result<(), String> {
    for study in feg_case_studies::study::published_studies() {
        println!("{:<40} {}", study.id, study.summary);
    }
    Ok(())
}
