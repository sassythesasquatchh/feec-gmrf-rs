#!/usr/bin/env python3
"""Freeze exact-law C51 spectral ranks from preregistered 2% replay curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
import uuid
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "c51-performance-rank-decisions-v2"
PROFILE = "juq-exact-performance-v1"
SENSOR_DESIGN = "c51-magnetic-fixed-footprint-maximin-12-v1"
OUTPUT_DESIGN = "rank_study_selection_and_held_out_wendland_c2_18"
CAMPAIGN_TOLERANCE = 0.05
RANK_SELECTION_TOLERANCE = 0.02
ARCHIVE_FILES = (
    "basis_attempts.csv",
    "compact_accuracy.csv",
    "covariances.csv",
    "exact_covariance_audits.csv",
    "high_observation_kernels.csv",
    "high_observation_prefixes.csv",
    "manifest.yaml",
    "rows.csv",
    "sensor_banks.csv",
    "sensor_overlaps.csv",
)
PERF0_FILES = ("manifest.yaml", "rows.csv")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_files_hash(directory: pathlib.Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot hash required artifact {path}: {error}") from error
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def validate_perf0(directory: pathlib.Path) -> str:
    rows = load_csv(directory / "rows.csv")
    indexed = {row.get("method"): row for row in rows}
    for method in ("fm_aug", "exact_observation_gp"):
        row = indexed.get(method)
        if row is None or row.get("same_discrete_law_eligible") != "true":
            raise ValueError(f"PERF-0 lacks a same-law {method} row")
        if row.get("observation_design_id") != SENSOR_DESIGN:
            raise ValueError(f"PERF-0 {method} row has the wrong observation design")
        if row.get("physical_qoi_design_id") != OUTPUT_DESIGN:
            raise ValueError(f"PERF-0 {method} row has the wrong physical-QoI design")
        if row.get("physical_qoi_output_count") != "18":
            raise ValueError(f"PERF-0 {method} row does not certify 18 outputs")
        for metric in (
            "physical_output_covariance_action_relative_error",
            "physical_variance_relative_error",
            "physical_transformed_variance_relative_error",
            "physical_mean_relative_error",
        ):
            if row.get(f"{metric}_status") != "measured":
                raise ValueError(f"PERF-0 {method} lacks measured {metric}")
            if float(row[f"{metric}_value"]) > 1.0e-8:
                raise ValueError(f"PERF-0 {method} exceeds the same-law tolerance")
    return named_files_hash(directory, PERF0_FILES)


def derive_one(attempt: pathlib.Path, perf0_hash: str) -> dict[str, Any]:
    manifest_path = attempt / "attempt_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("outcome") != "completed" or manifest.get("raw_return_code") != 0:
        raise ValueError(f"{attempt} is not a completed replay attempt")
    if manifest.get("evidence_tier") not in {
        "high_observation_replay_accuracy_only",
        "high_observation_confirmatory_rank_extension",
    }:
        raise ValueError(f"{attempt} is not an admissible m=4096 accuracy attempt")
    profile = str(manifest.get("profile", ""))
    if "m4096" not in profile:
        raise ValueError(f"{attempt} is not an m=4096 accuracy profile")
    if manifest.get("method_performance_timing_eligible") is not False:
        raise ValueError(f"{attempt} incorrectly marks replay timing eligible")
    study = attempt / "study"
    rows = load_csv(study / "rows.csv")
    selected_rows = [
        row
        for row in rows
        if row.get("alpha") == "2"
        and row.get("design") == "high_m4096"
        and row.get("observation_count") == "4096"
    ]
    if not selected_rows:
        raise ValueError(f"{attempt} contains no alpha-2 m=4096 rank curve")
    selected_rows.sort(key=lambda row: int(row["rank"]))
    if any(row.get("status") != "measured" for row in selected_rows):
        raise ValueError(f"{attempt} contains a censored spectral rank")
    if any(row.get("method_performance_timing_eligible") != "false" for row in selected_rows):
        raise ValueError(f"{attempt} contains a timing-eligible replay row")
    candidate_index = None
    for index, row in enumerate(selected_rows):
        if index == 0:
            continue
        suffix = selected_rows[index:]
        if all(
            float(later["selection_e_primary"]) <= RANK_SELECTION_TOLERANCE
            for later in suffix
        ):
            candidate_index = index
            break
    if candidate_index is None:
        raise ValueError(f"{attempt} does not attain the preregistered 2% selection rule")
    selected = selected_rows[candidate_index]
    lower = selected_rows[candidate_index - 1]
    selection_error = float(selected["selection_e_primary"])
    heldout_error = float(selected["heldout_e_primary"])
    lower_error = float(lower["selection_e_primary"])
    if heldout_error > RANK_SELECTION_TOLERANCE:
        raise ValueError(
            f"{attempt} fails held-out at its preregistered 2% selection-locked rank"
        )
    if lower_error <= RANK_SELECTION_TOLERANCE:
        raise ValueError(f"{attempt} lacks an immediate lower 2% failing bracket")
    refinement = int(selected["refinement"])
    state_dimension = int(selected["state_dimension"])
    rank = int(selected["rank"])
    archive_hash = named_files_hash(study, ARCHIVE_FILES)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{attempt} lacks source provenance")
    repository = source.get("repository")
    submodules = source.get("submodules")
    if not isinstance(repository, dict) or not isinstance(submodules, dict):
        raise ValueError(f"{attempt} has incomplete source provenance")
    if repository.get("dirty") is not False or any(
        not isinstance(submodules.get(name), dict)
        or submodules[name].get("dirty") is not False
        for name in ("feec", "gmrf")
    ):
        raise ValueError(f"{attempt} replay source is dirty")
    return {
        "decision_id": f"exact-rank-n{refinement}-m4096-r{rank}",
        "source_accuracy_profile": profile,
        "source_attempt_manifest_sha256": sha256_file(manifest_path),
        "source_identity": {
            "root_sha": repository.get("head"),
            "feec_sha": submodules["feec"].get("head"),
            "gmrf_sha": submodules["gmrf"].get("head"),
        },
        "refinement": refinement,
        "state_dimension": state_dimension,
        "scalar_observation_count": 4096,
        "alpha": 2,
        "relative_tolerance": CAMPAIGN_TOLERANCE,
        "observation_design_id": SENSOR_DESIGN,
        "physical_qoi_design_id": OUTPUT_DESIGN,
        "reference_discrete_law_id": "exact_full_metric_whitney",
        "common_h_g_fixture_sha256": archive_hash,
        "spectral_low_rank": {
            "rank": rank,
            "lower_failing_rank": int(lower["rank"]),
            "lower_failing_selection_relative_error": lower_error,
            "selection_relative_error": selection_error,
            "heldout_relative_error": heldout_error,
            "validation_status": "heldout_validated",
            "rank_status": "full_rank_required" if rank == state_dimension else "low_rank",
            "qualification_rule": "pre_registered_2pct_sensitivity",
            "rank_selection_tolerance": RANK_SELECTION_TOLERANCE,
            "accuracy_artifact_sha256": archive_hash,
            "timing_basis_source": "partial_exact_requested_mixed_shell",
            "shared_maximum_basis_timing_eligible": False,
        },
        "exact_constrained": {
            "validation_status": "perf0_same_law_validated",
            "fixture_id": "PERF-0",
            "fixture_artifact_sha256": perf0_hash,
            "fixture_hash_scope": "manifest.yaml+rows.csv",
        },
        "exact_observation_gp": {
            "validation_status": "perf0_same_law_validated",
            "fixture_id": "PERF-0",
            "fixture_artifact_sha256": perf0_hash,
            "fixture_hash_scope": "manifest.yaml+rows.csv",
        },
        "accuracy_artifact_hash_scope": "canonical_c51_accuracy_archive_files_v1",
    }


def derive(attempts: Sequence[pathlib.Path], perf0: pathlib.Path) -> dict[str, Any]:
    if not attempts:
        raise ValueError("at least one replay attempt is required")
    perf0_hash = validate_perf0(perf0)
    decisions = [derive_one(path, perf0_hash) for path in attempts]
    decisions.sort(key=lambda item: int(item["refinement"]))
    refinements = [int(item["refinement"]) for item in decisions]
    if len(refinements) != len(set(refinements)):
        raise ValueError("replay attempts contain duplicate refinements")
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "status": "frozen_heldout_validated",
        "source_attempts": [str(path) for path in attempts],
        "perf0_archive": str(perf0),
        "decisions": decisions,
    }


def write_atomic(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf0-archive", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("attempts", type=pathlib.Path, nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        write_atomic(args.output, derive(args.attempts, args.perf0_archive))
    except (OSError, ValueError) as error:
        print(f"C51 exact rank decision failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
