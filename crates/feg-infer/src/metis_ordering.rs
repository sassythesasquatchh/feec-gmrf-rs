use gmrf_core::{Permutation, SparseMatrix};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{create_dir_all, read_to_string, remove_dir_all};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn metis_nested_dissection_ordering(
    precision: &SparseMatrix,
) -> Result<Option<Permutation>, String> {
    if !command_exists_on_path("ndmetis") {
        return Ok(None);
    }

    let graph = metis_graph_text(precision)?;
    let workdir = unique_metis_workdir()?;
    create_dir_all(&workdir).map_err(|err| err.to_string())?;
    let graph_path = workdir.join("precision.graph");
    let iperm_path = workdir.join("precision.graph.iperm");
    let cleanup = MetisTempDir {
        path: workdir.clone(),
    };
    std::fs::write(&graph_path, graph).map_err(|err| err.to_string())?;

    let output = match Command::new("ndmetis")
        .arg("-seed=0")
        .arg(&graph_path)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(format!("failed to run ndmetis: {err}")),
    };

    if !output.status.success() {
        return Err(format!(
            "ndmetis failed with status {}: {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let iperm = read_to_string(&iperm_path).map_err(|err| err.to_string())?;
    let permutation = parse_metis_iperm(&iperm, precision.nrows())?;
    drop(cleanup);
    Ok(Some(permutation))
}

fn command_exists_on_path(command: &str) -> bool {
    std::env::var_os("PATH")
        .as_deref()
        .is_some_and(|path| command_exists_in_path_value(command, path))
}

fn command_exists_in_path_value(command: &str, path: &OsStr) -> bool {
    std::env::split_paths(path).any(|directory| directory.join(command).is_file())
}

struct MetisTempDir {
    path: PathBuf,
}

impl Drop for MetisTempDir {
    fn drop(&mut self) {
        let _ = remove_dir_all(&self.path);
    }
}

fn unique_metis_workdir() -> Result<PathBuf, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!("feg-metis-{}-{nanos}", std::process::id())))
}

pub fn metis_graph_text(precision: &SparseMatrix) -> Result<String, String> {
    let adjacency = symmetric_graph_adjacency(precision)?;
    Ok(metis_graph_text_from_adjacency(&adjacency))
}

pub fn symmetric_graph_adjacency(precision: &SparseMatrix) -> Result<Vec<BTreeSet<usize>>, String> {
    if precision.nrows() != precision.ncols() {
        return Err("METIS ordering requires a square precision matrix".to_string());
    }
    let mut adjacency = vec![BTreeSet::new(); precision.nrows()];
    for (row, col, value) in precision.triplet_iter() {
        if row == col || *value == 0.0 {
            continue;
        }
        adjacency[row].insert(col);
        adjacency[col].insert(row);
    }
    Ok(adjacency)
}

pub fn metis_graph_text_from_adjacency(adjacency: &[BTreeSet<usize>]) -> String {
    let edge_count = adjacency
        .iter()
        .map(|neighbors| neighbors.len())
        .sum::<usize>()
        / 2;
    let mut text = format!("{} {edge_count}\n", adjacency.len());
    for neighbors in adjacency {
        let line = neighbors
            .iter()
            .map(|neighbor| (neighbor + 1).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        text.push_str(&line);
        text.push('\n');
    }
    text
}

pub fn parse_metis_iperm(contents: &str, dimension: usize) -> Result<Permutation, String> {
    let orig_to_perm = contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim()
                .parse::<usize>()
                .map_err(|err| format!("invalid ndmetis iperm entry {line:?}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if orig_to_perm.len() != dimension {
        return Err(format!(
            "ndmetis iperm length {} does not match matrix dimension {dimension}",
            orig_to_perm.len()
        ));
    }
    Permutation::from_orig_to_perm(orig_to_perm).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::types::CooMatrix;

    #[test]
    fn metis_graph_export_is_symmetric_one_based_and_loop_free() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 10.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 2, -1.0);
        coo.push(2, 2, 10.0);
        let matrix = SparseMatrix::from(&coo);

        let adjacency = symmetric_graph_adjacency(&matrix).unwrap();
        assert!(!adjacency[0].contains(&0));
        assert!(adjacency[0].contains(&1));
        assert!(adjacency[1].contains(&0));
        assert!(adjacency[1].contains(&2));
        assert!(adjacency[2].contains(&1));

        let text = metis_graph_text_from_adjacency(&adjacency);
        assert_eq!(text, "3 2\n2\n1 3\n2\n");
    }

    #[test]
    fn parses_metis_iperm_as_orig_to_perm() {
        let permutation = parse_metis_iperm("2\n0\n1\n", 3).unwrap();
        assert_eq!(permutation.orig_to_perm, vec![2, 0, 1]);
        assert_eq!(permutation.perm_to_orig, vec![1, 2, 0]);
    }

    #[test]
    fn command_path_preflight_distinguishes_present_and_absent_tools() {
        let directory = unique_metis_workdir().unwrap();
        create_dir_all(&directory).unwrap();
        let cleanup = MetisTempDir {
            path: directory.clone(),
        };
        std::fs::write(directory.join("ndmetis"), "").unwrap();

        assert!(command_exists_in_path_value(
            "ndmetis",
            directory.as_os_str()
        ));
        assert!(!command_exists_in_path_value(
            "missing-ndmetis",
            directory.as_os_str()
        ));
        drop(cleanup);
    }

    #[test]
    fn ndmetis_ordering_runs_when_available() {
        if !command_exists_on_path("ndmetis") {
            return;
        }

        let mut coo = CooMatrix::new(4, 4);
        for index in 0..4 {
            coo.push(index, index, 3.0);
            if index + 1 < 4 {
                coo.push(index, index + 1, -1.0);
                coo.push(index + 1, index, -1.0);
            }
        }
        let matrix = SparseMatrix::from(&coo);
        let Some(permutation) = metis_nested_dissection_ordering(&matrix).unwrap() else {
            return;
        };
        permutation.validate().unwrap();
        assert_eq!(permutation.dimension(), 4);
    }
}
