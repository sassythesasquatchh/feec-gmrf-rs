//! Reproducibility provenance captured for every study run.

use std::path::Path;

#[derive(Debug, Clone)]
pub struct Provenance {
    pub study_id: String,
    pub profile: String,
    pub package_version: &'static str,
    pub root_sha: String,
    pub feec_sha: String,
    pub gmrf_sha: String,
    pub root_dirty: bool,
    pub feec_dirty: bool,
    pub gmrf_dirty: bool,
    pub rustc: String,
    pub cargo: String,
    pub backend: String,
    pub tools: Vec<(String, String)>,
    pub inputs: Vec<FileRecord>,
    pub command: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FileRecord {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

impl Provenance {
    pub fn capture(
        study_id: &str,
        profile: &str,
        input_paths: &[&str],
        custom_config: Option<&Path>,
    ) -> Result<Self, String> {
        let mut inputs = input_paths
            .iter()
            .map(|path| {
                file_record(path).ok_or_else(|| {
                    format!("declared study input `{path}` is missing or unreadable")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(path) = custom_config {
            inputs.push(file_record(path).ok_or_else(|| {
                format!(
                    "custom study configuration `{}` is missing or unreadable",
                    path.display()
                )
            })?);
        }
        Ok(Self {
            study_id: study_id.to_string(),
            profile: profile.to_string(),
            package_version: env!("CARGO_PKG_VERSION"),
            root_sha: git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
            feec_sha: git_output(&["-C", "feec", "rev-parse", "HEAD"])
                .unwrap_or_else(|| "unknown".into()),
            gmrf_sha: git_output(&["-C", "gmrf-rs", "rev-parse", "HEAD"])
                .unwrap_or_else(|| "unknown".into()),
            root_dirty: is_dirty(&[]),
            feec_dirty: is_dirty(&["-C", "feec"]),
            gmrf_dirty: is_dirty(&["-C", "gmrf-rs"]),
            rustc: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
            cargo: command_output("cargo", &["--version"]).unwrap_or_else(|| "unknown".into()),
            backend: format!(
                "explicit sparse matrices; {}",
                command_output("cargo", &["tree", "-p", "faer", "--depth", "0"])
                    .unwrap_or_else(|| "faer version unavailable".into())
            ),
            tools: ["gmsh", "mpiexec", "python3"]
                .into_iter()
                .map(|tool| {
                    (
                        tool.to_string(),
                        command_output(tool, &["--version"])
                            .unwrap_or_else(|| "unavailable".into()),
                    )
                })
                .collect(),
            inputs,
            command: std::env::args().collect(),
        })
    }
}

pub fn file_record(path: impl AsRef<Path>) -> Option<FileRecord> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    Some(FileRecord {
        path: path.display().to_string(),
        bytes: metadata.len(),
        sha256: sha256(path).unwrap_or_else(|| "unavailable".to_string()),
    })
}

fn is_dirty(prefix: &[&str]) -> bool {
    let mut args = prefix.to_vec();
    args.extend(["status", "--porcelain"]);
    git_output(&args).is_some_and(|value| !value.is_empty())
}

fn sha256(path: &Path) -> Option<String> {
    let rendered = path.to_str()?;
    command_output("sha256sum", &[rendered])
        .or_else(|| command_output("shasum", &["-a", "256", rendered]))
        .and_then(|line| line.split_whitespace().next().map(str::to_string))
}

fn git_output(args: &[&str]) -> Option<String> {
    command_output("git", args)
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
