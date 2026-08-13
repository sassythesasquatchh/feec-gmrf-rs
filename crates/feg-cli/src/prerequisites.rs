//! External-program checks declared by study descriptors.

pub fn check(requirements: &[&str]) -> Result<(), String> {
    for requirement in requirements {
        let status = std::process::Command::new(requirement)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if !matches!(status, Ok(status) if status.success()) {
            return Err(format!(
                "required external program `{requirement}` is unavailable"
            ));
        }
    }
    Ok(())
}
