//! Stable, dependency-light run-manifest encoding.

use crate::provenance::{FileRecord, Provenance};
use std::path::Path;
use std::time::Duration;

pub fn write_run_manifest(
    output: &Path,
    provenance: &Provenance,
    metrics: &[(String, f64)],
    elapsed: Duration,
) -> Result<(), String> {
    let mut text = String::new();
    push(&mut text, "schema", "feg-study-run-v1");
    push(&mut text, "study_id", &provenance.study_id);
    push(&mut text, "profile", &provenance.profile);
    push(&mut text, "package_version", provenance.package_version);
    push(&mut text, "root_sha", &provenance.root_sha);
    push(&mut text, "feec_sha", &provenance.feec_sha);
    push(&mut text, "gmrf_sha", &provenance.gmrf_sha);
    push(&mut text, "root_dirty", &provenance.root_dirty.to_string());
    push(&mut text, "feec_dirty", &provenance.feec_dirty.to_string());
    push(&mut text, "gmrf_dirty", &provenance.gmrf_dirty.to_string());
    push(&mut text, "rustc", &provenance.rustc);
    push(&mut text, "cargo", &provenance.cargo);
    push(&mut text, "backend", &provenance.backend);
    for (name, version) in &provenance.tools {
        push(&mut text, &format!("tool.{name}"), version);
    }
    write_file_records(&mut text, "input", &provenance.inputs);
    push(&mut text, "command", &provenance.command.join(" "));
    push(
        &mut text,
        "elapsed_seconds",
        &elapsed.as_secs_f64().to_string(),
    );
    for (name, value) in metrics {
        push(&mut text, &format!("metric.{name}"), &value.to_string());
    }
    write_file_records(&mut text, "artifact", &artifact_inventory(output));
    std::fs::write(output.join("run-manifest.tsv"), text).map_err(|error| error.to_string())
}

pub fn verify_run_manifest(run_directory: &Path, against: &str) -> Result<(), String> {
    let manifest = std::fs::read_to_string(run_directory.join("run-manifest.tsv"))
        .map_err(|error| error.to_string())?;
    let expected = format!("profile\t{against}\n");
    if !manifest.contains(&expected) {
        return Err(format!("run manifest does not use profile `{against}`"));
    }
    if against == "thesis-submitted"
        && [
            "root_dirty\ttrue\n",
            "feec_dirty\ttrue\n",
            "gmrf_dirty\ttrue\n",
        ]
        .iter()
        .any(|marker| manifest.contains(marker))
    {
        return Err("a dirty checkout cannot certify a thesis-submitted run".to_string());
    }
    if !run_directory.join("resolved-profile.toml").is_file() {
        return Err("run directory is missing resolved-profile.toml".to_string());
    }
    let fields = manifest
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .collect::<std::collections::BTreeMap<_, _>>();
    for (key, relative_path) in fields
        .iter()
        .filter(|(key, _)| key.starts_with("artifact.") && key.ends_with(".path"))
    {
        let prefix = key.trim_end_matches(".path");
        let expected_hash = fields
            .get(format!("{prefix}.sha256").as_str())
            .ok_or_else(|| format!("manifest is missing {prefix}.sha256"))?;
        let path = run_directory.join(relative_path);
        let record = crate::provenance::file_record(&path)
            .ok_or_else(|| format!("recorded artifact `{relative_path}` is missing"))?;
        if record.sha256 != *expected_hash {
            return Err(format!(
                "recorded artifact `{relative_path}` has changed since the run"
            ));
        }
    }
    for (key, value) in fields.iter().filter(|(key, _)| key.starts_with("metric.")) {
        let parsed = value
            .parse::<f64>()
            .map_err(|_| format!("manifest metric `{key}` is not numeric"))?;
        if !parsed.is_finite() {
            return Err(format!("manifest metric `{key}` is not finite"));
        }
    }
    Ok(())
}

fn artifact_inventory(output: &Path) -> Vec<FileRecord> {
    let mut records = Vec::new();
    let mut pending = vec![output.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().and_then(|name| name.to_str()) != Some("run-manifest.tsv") {
                if let Some(mut record) = crate::provenance::file_record(&path) {
                    record.path = path
                        .strip_prefix(output)
                        .unwrap_or(&path)
                        .display()
                        .to_string();
                    records.push(record);
                }
            }
        }
    }
    records.sort_by(|left, right| left.path.cmp(&right.path));
    records
}

fn write_file_records(output: &mut String, namespace: &str, records: &[FileRecord]) {
    for (index, record) in records.iter().enumerate() {
        push(output, &format!("{namespace}.{index}.path"), &record.path);
        push(
            output,
            &format!("{namespace}.{index}.bytes"),
            &record.bytes.to_string(),
        );
        push(
            output,
            &format!("{namespace}.{index}.sha256"),
            &record.sha256,
        );
    }
}

fn push(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push('\t');
    output.push_str(&value.replace(['\n', '\t'], " "));
    output.push('\n');
}
