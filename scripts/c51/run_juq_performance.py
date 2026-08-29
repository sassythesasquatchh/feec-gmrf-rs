#!/usr/bin/env python3
"""Orchestrate the immutable C51 JUQ matched-accuracy performance campaign.

Every method/phase cell is delegated to fresh_process_runner.py and therefore
owns one scientific process and one external process-tree RSS measurement.
Spectral ranks are never inferred here: execution requires a separate,
held-out-validated 5 percent decision file covering every frozen (n, m) key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pathlib
import random
import shlex
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import analyze_archive
import fresh_process_runner as fresh


SCHEMA_VERSION = "c51-juq-performance-campaign-v1"
PROFILE_SCHEMA_VERSION = "c51-juq-performance-profile-v1"
RANK_DECISION_SCHEMA_VERSION = "c51-performance-rank-decisions-v2"
COMPACT_PROFILE_NAME = "juq-performance-v1"
EXACT_PROFILE_NAME = "juq-exact-performance-v1"
DEFAULT_PROFILE = pathlib.Path(__file__).with_name("juq-performance-v1.json")
DEFAULT_RANK_DECISIONS = pathlib.Path(__file__).with_name(
    "juq-performance-v1-rank-decisions.template.json"
)
DEFAULT_RUNNER = pathlib.Path(__file__).with_name("fresh_process_runner.py")
COMPACT_EXPECTED_AXES = {
    "refinements": [8, 10],
    "scalar_observation_counts": [512, 2048, 4096, 5184],
    "replacement_data_counts": [1, 100],
    "methods": ["whitney_compact", "exact_observation_gp", "spectral_low_rank"],
    "workload_phases": ["cold_setup", "query_physical_uq"],
    "observation_class": "fixed_footprint_magnetic_volume",
}
COMPACT_EXPECTED_EVIDENCE_TIERS = [
    {
        "id": "claim_primary",
        "role": "claim_bearing_5pct_matched_accuracy",
        "refinements": [8, 10],
        "scalar_observation_counts": [2048],
        "replacement_data_counts": [1, 100],
        "warmup_replicates": 1,
        "measured_replicates": 3,
    },
    {
        "id": "scaling_context",
        "role": "lower_m_scaling_context_not_primary_claim",
        "refinements": [8, 10],
        "scalar_observation_counts": [512],
        "replacement_data_counts": [1, 100],
        "warmup_replicates": 0,
        "measured_replicates": 2,
    },
    {
        "id": "high_data_confirmation",
        "role": "confirmatory_high_data_5pct_matched_accuracy",
        "refinements": [8, 10],
        "scalar_observation_counts": [4096],
        "replacement_data_counts": [100],
        "warmup_replicates": 0,
        "measured_replicates": 3,
    },
    {
        "id": "capacity_stress",
        "role": "maximum_capacity_resource_stress_and_typed_censor_context",
        "refinements": [8, 10],
        "scalar_observation_counts": [5184],
        "replacement_data_counts": [100],
        "warmup_replicates": 0,
        "measured_replicates": 1,
    },
]
EXACT_EXPECTED_AXES = {
    "refinements": [8, 10],
    "scalar_observation_counts": [4096],
    "replacement_data_counts": [1, 100],
    "methods": ["fm_aug", "exact_observation_gp", "spectral_low_rank"],
    "workload_phases": ["cold_setup", "query_physical_uq"],
    "observation_class": "fixed_footprint_magnetic_volume",
}
EXACT_EXPECTED_EVIDENCE_TIERS = [
    {
        "id": "claim_primary_exact",
        "role": "claim_bearing_same_law_time_throughput",
        "refinements": [8, 10],
        "scalar_observation_counts": [4096],
        "replacement_data_counts": [1, 100],
        "warmup_replicates": 1,
        "measured_replicates": 3,
    }
]
PROFILE_CONTRACTS = {
    COMPACT_PROFILE_NAME: (COMPACT_EXPECTED_AXES, COMPACT_EXPECTED_EVIDENCE_TIERS),
    EXACT_PROFILE_NAME: (EXACT_EXPECTED_AXES, EXACT_EXPECTED_EVIDENCE_TIERS),
}
# Backward-compatible names used by the compact-v1 tests and downstream
# validation code. New code resolves its contract from the profile payload.
PROFILE_NAME = COMPACT_PROFILE_NAME
EXPECTED_AXES = COMPACT_EXPECTED_AXES
EXPECTED_EVIDENCE_TIERS = COMPACT_EXPECTED_EVIDENCE_TIERS
EXPECTED_EXECUTION = {
    "deterministic_randomization_seed": 849348622,
    "threads": 1,
    "cgroup_mode": "required",
    "continue_after_cell_failure": True,
    "require_clean_recursive_source": True,
    "fresh_process_per_method_phase_cell": True,
}
PAIR_QUERY_MEASUREMENTS = {
    "repeated_data_ns",
    "physical_qoi_ns",
    "variance_estimation_ns",
    "query_external_peak_rss_bytes",
}
PAIR_SETUP_MEASUREMENTS = set(analyze_archive.MEASUREMENT_FIELDS).difference(
    PAIR_QUERY_MEASUREMENTS
)
PAIR_KEY_FIELDS = (
    "schema_version",
    "study_id",
    "method",
    "available",
    "unavailable_reason",
    "discrete_law_id",
    "same_discrete_law_eligible",
    "common_physical_uq_workload_eligible",
    "method_performance_timing_eligible",
    "method_performance_timing_ineligibility_reason",
    "spectral_basis_source",
    "spectral_feec_assembly_reuse",
    "refinement",
    "spatial_dimension",
    "form_degree",
    "alpha",
    "kappa",
    "tau",
    "observation_class",
    "sensor_count",
    "sensor_density",
    "repeated_data_count",
    "posterior_sample_count",
    "seed",
    "latent_dimension",
    "ambient_dimension",
    "constraint_dimension",
    "observation_nnz",
    "solver",
    "ordering",
)


@dataclass(frozen=True)
class CampaignCell:
    order_index: int
    block_index: int
    evidence_tier: str
    evidence_role: str
    run_kind: str
    replicate: int
    refinement: int
    scalar_observation_count: int
    replacement_data_count: int
    method: str
    workload_phase: str
    spectral_rank: int | None
    rank_decision_id: str | None
    cell_id: str


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"profile schema must be {PROFILE_SCHEMA_VERSION}")
    profile_name = str(profile.get("profile"))
    if profile_name not in PROFILE_CONTRACTS:
        raise ValueError(f"unsupported immutable C51 profile {profile_name!r}")
    expected_axes, expected_tiers = PROFILE_CONTRACTS[profile_name]
    if profile.get("axes") != expected_axes:
        raise ValueError(f"{profile_name} axes differ from the immutable contract")
    if profile.get("evidence_tiers") != expected_tiers:
        raise ValueError(f"{profile_name} evidence tiers differ from the immutable contract")
    if profile.get("execution") != EXPECTED_EXECUTION:
        raise ValueError(f"{profile_name} execution policy differs from the immutable contract")
    law = profile.get("scientific_law")
    expected_law = {
        "geometry": "unit_cube",
        "spatial_dimension": 3,
        "form_degree": 1,
        "alpha": 2,
        "kappa": 4.0,
        "tau": 1.0,
        "observation_noise_variance": 0.05,
        "boundary_treatment": "homogeneous_full_relative",
        "physical_chain": "A_to_D1_to_B",
        "sensor_design_id": "c51-magnetic-fixed-footprint-maximin-12-v1",
        "physical_qoi_design_id": "rank_study_selection_and_held_out_wendland_c2_18",
        "sensor_radius": 0.16,
        "sensor_capacity_scalar_rows": 5184,
        "accuracy_target": {
            "metric": "held_out_physical_covariance_primary_envelope",
            "primary_relative_tolerance": 0.05,
            "context_relative_tolerance": 0.1,
            "sensitivity_relative_tolerance": 0.02,
        },
    }
    if law != expected_law:
        raise ValueError(f"{profile_name} scientific law differs from the immutable contract")


def rank_key(
    refinement: int,
    scalar_observation_count: int,
    profile: Mapping[str, Any],
) -> tuple[int, int, int, float, str]:
    law = profile["scientific_law"]
    return (
        refinement,
        scalar_observation_count,
        int(law["alpha"]),
        float(law["accuracy_target"]["primary_relative_tolerance"]),
        str(law["sensor_design_id"]),
    )


def require_sha256(value: Any, label: str) -> str:
    artifact_hash = str(value)
    if len(artifact_hash) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_hash
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return artifact_hash


def load_rank_decisions(
    path: pathlib.Path, profile: Mapping[str, Any]
) -> tuple[dict[tuple[int, int, int, float, str], dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    profile_name = str(profile["profile"])
    if payload.get("schema_version") != RANK_DECISION_SCHEMA_VERSION:
        raise ValueError(f"rank decision schema must be {RANK_DECISION_SCHEMA_VERSION}")
    if payload.get("profile") != profile_name:
        raise ValueError(f"rank decisions must target {profile_name}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("rank decision file must contain a decisions list")
    indexed: dict[tuple[int, int, int, float, str], dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"rank decision {index} must be an object")
        try:
            key = (
                int(decision["refinement"]),
                int(decision["scalar_observation_count"]),
                int(decision["alpha"]),
                float(decision["relative_tolerance"]),
                str(decision["observation_design_id"]),
            )
            spectral = decision["spectral_low_rank"]
            exact_gp = decision["exact_observation_gp"]
            rank = int(spectral["rank"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"rank decision {index} has invalid qualification fields: {error}"
            ) from error
        if key != rank_key(key[0], key[1], profile):
            raise ValueError(
                f"rank decision {index} is not a primary 5 percent frozen-law decision"
            )
        axes = profile["axes"]
        if key[0] not in axes["refinements"] or key[1] not in axes[
            "scalar_observation_counts"
        ]:
            raise ValueError(f"rank decision {index} is outside the frozen axes")
        if rank < 1:
            raise ValueError(f"rank decision {index} must have a positive rank")
        if decision.get("reference_discrete_law_id") != "exact_full_metric_whitney":
            raise ValueError(f"rank decision {index} has the wrong reference discrete law")
        if decision.get("physical_qoi_design_id") != (
            "rank_study_selection_and_held_out_wendland_c2_18"
        ):
            raise ValueError(f"rank decision {index} has the wrong physical-QoI design")
        require_sha256(
            decision.get("common_h_g_fixture_sha256", ""),
            f"rank decision {index} common H/G fixture",
        )
        if spectral.get("validation_status") != "heldout_validated":
            raise ValueError(f"rank decision {index} spectral model is not held-out validated")
        if spectral.get("timing_basis_source") != "partial_exact_requested_mixed_shell":
            raise ValueError(f"rank decision {index} does not require exact-request timing")
        try:
            spectral_error = float(spectral["heldout_relative_error"])
            spectral_selection_error = float(spectral["selection_relative_error"])
            lower_failing_error = float(spectral["lower_failing_selection_relative_error"])
            lower_failing_rank = int(spectral["lower_failing_rank"])
            state_dimension = int(decision["state_dimension"])
            rank_selection_tolerance = float(
                spectral.get("rank_selection_tolerance", key[3])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"rank decision {index} has invalid qualification metrics") from error
        if profile_name == EXACT_PROFILE_NAME:
            if spectral.get("qualification_rule") != "pre_registered_2pct_sensitivity":
                raise ValueError(
                    f"rank decision {index} exact campaign requires the preregistered 2 percent sensitivity rule"
                )
            if rank_selection_tolerance != 0.02:
                raise ValueError(
                    f"rank decision {index} exact campaign rank tolerance must be 2 percent"
                )
        elif rank_selection_tolerance != key[3]:
            raise ValueError(f"rank decision {index} changes the compact-v1 rank tolerance")
        if (
            not 0.0 <= spectral_error <= rank_selection_tolerance
            or not 0.0 <= spectral_selection_error <= rank_selection_tolerance
        ):
            raise ValueError(
                f"rank decision {index} spectral error exceeds its rank-selection tolerance"
            )
        if lower_failing_rank >= rank or lower_failing_error <= rank_selection_tolerance:
            raise ValueError(f"rank decision {index} lacks a lower failing selection bracket")
        expected_rank_status = "full_rank_required" if rank == state_dimension else "low_rank"
        if rank > state_dimension or spectral.get("rank_status") != expected_rank_status:
            raise ValueError(f"rank decision {index} has inconsistent rank endpoint semantics")
        if spectral.get("shared_maximum_basis_timing_eligible") is not False:
            raise ValueError(f"rank decision {index} does not forbid shared-basis timing")
        spectral_hash = require_sha256(
            spectral.get("accuracy_artifact_sha256", ""),
            f"rank decision {index} spectral accuracy artifact",
        )
        common_hash = require_sha256(
            decision.get("common_h_g_fixture_sha256", ""),
            f"rank decision {index} common H/G fixture",
        )
        if spectral_hash != common_hash:
            raise ValueError(f"rank decision {index} does not bind spectral accuracy to common H/G")
        if profile_name == COMPACT_PROFILE_NAME:
            try:
                compact = decision["whitney_compact"]
                compact_error = float(compact["heldout_relative_error"])
                compact_selection_error = float(compact["selection_relative_error"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"rank decision {index} lacks compact qualifications: {error}"
                ) from error
            if compact.get("validation_status") != "heldout_validated":
                raise ValueError(f"rank decision {index} compact model is not held-out validated")
            if not 0.0 <= compact_error <= key[3] or not 0.0 <= compact_selection_error <= key[3]:
                raise ValueError(f"rank decision {index} compact error exceeds 5 percent")
            compact_hash = require_sha256(
                compact.get("accuracy_artifact_sha256", ""),
                f"rank decision {index} compact accuracy artifact",
            )
            if compact_hash != common_hash:
                raise ValueError(f"rank decision {index} does not bind compact and spectral accuracy")
            if compact.get("identical_observation_and_output_maps") is not True:
                raise ValueError(f"rank decision {index} compact method lacks identical H/G maps")
            if compact.get("method_performance_timing_eligible") is not False:
                raise ValueError(f"rank decision {index} compact accuracy row is timing eligible")
        else:
            try:
                exact_constrained = decision["exact_constrained"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"rank decision {index} lacks exact constrained qualification: {error}"
                ) from error
            if exact_constrained.get("validation_status") != "perf0_same_law_validated":
                raise ValueError(
                    f"rank decision {index} exact constrained model lacks PERF-0 qualification"
                )
            if exact_constrained.get("fixture_id") != "PERF-0":
                raise ValueError(f"rank decision {index} exact constrained fixture is not PERF-0")
            exact_constrained_hash = require_sha256(
                exact_constrained.get("fixture_artifact_sha256", ""),
                f"rank decision {index} exact constrained PERF-0 artifact",
            )
        if exact_gp.get("validation_status") != "perf0_same_law_validated":
            raise ValueError(f"rank decision {index} exact GP lacks PERF-0 qualification")
        if exact_gp.get("fixture_id") != "PERF-0":
            raise ValueError(f"rank decision {index} exact GP fixture is not PERF-0")
        exact_gp_hash = require_sha256(
            exact_gp.get("fixture_artifact_sha256", ""),
            f"rank decision {index} PERF-0 fixture artifact",
        )
        if profile_name == EXACT_PROFILE_NAME and exact_constrained_hash != exact_gp_hash:
            raise ValueError(
                f"rank decision {index} exact methods do not share one PERF-0 artifact"
            )
        if key in indexed:
            raise ValueError(f"duplicate rank decision key {key}")
        normalized = dict(decision)
        normalized["rank"] = rank
        normalized["decision_id"] = str(
            decision.get("decision_id") or f"rank-{key[0]}-{key[1]}-{rank}"
        )
        indexed[key] = normalized
    return indexed, payload


def selected_evidence_tiers(
    profile: Mapping[str, Any], requested_ids: Sequence[str] | None = None
) -> list[Mapping[str, Any]]:
    """Resolve a canonical, immutable subset of the frozen evidence tiers."""
    tiers = list(profile["evidence_tiers"])
    if requested_ids is None or len(requested_ids) == 0:
        return tiers
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("each --evidence-tier may be selected at most once")
    known = {str(tier["id"]) for tier in tiers}
    unknown = set(requested_ids).difference(known)
    if unknown:
        raise ValueError(f"unknown evidence tiers: {sorted(unknown)}")
    requested = set(requested_ids)
    return [tier for tier in tiers if str(tier["id"]) in requested]


def required_rank_keys(
    profile: Mapping[str, Any], requested_tier_ids: Sequence[str] | None = None
) -> list[tuple[int, int, int, float, str]]:
    keys = {
        rank_key(refinement, count, profile)
        for tier in selected_evidence_tiers(profile, requested_tier_ids)
        for refinement, count in itertools.product(
            tier["refinements"], tier["scalar_observation_counts"]
        )
    }
    return sorted(keys)


def missing_rank_keys(
    profile: Mapping[str, Any],
    decisions: Mapping[tuple[int, int, int, float, str], Mapping[str, Any]],
    requested_tier_ids: Sequence[str] | None = None,
) -> list[tuple[int, int, int, float, str]]:
    return [
        key
        for key in required_rank_keys(profile, requested_tier_ids)
        if key not in decisions
    ]


def projected_cell_counts(
    profile: Mapping[str, Any], requested_tier_ids: Sequence[str] | None = None
) -> dict[str, int]:
    methods = len(profile["axes"]["methods"])
    phases = len(profile["axes"]["workload_phases"])
    totals = {"warmup": 0, "measured": 0}
    for tier in selected_evidence_tiers(profile, requested_tier_ids):
        block_width = (
            len(tier["refinements"])
            * len(tier["scalar_observation_counts"])
            * len(tier["replacement_data_counts"])
            * phases
            * methods
        )
        totals["warmup"] += int(tier["warmup_replicates"]) * block_width
        totals["measured"] += int(tier["measured_replicates"]) * block_width
    return totals | {"total": totals["warmup"] + totals["measured"]}


def generate_cells(
    profile: Mapping[str, Any],
    decisions: Mapping[tuple[int, int, int, float, str], Mapping[str, Any]],
    requested_tier_ids: Sequence[str] | None = None,
) -> list[CampaignCell]:
    tiers = selected_evidence_tiers(profile, requested_tier_ids)
    missing = missing_rank_keys(profile, decisions, requested_tier_ids)
    if missing:
        raise ValueError(
            f"cannot generate campaign cells with {len(missing)} missing 5 percent rank decisions"
        )
    axes = profile["axes"]
    execution = profile["execution"]
    cells: list[CampaignCell] = []
    order_index = 0
    block_index = 0
    repetition_ordinal = 0
    for tier in tiers:
        repetitions = [
            *[("warmup", index) for index in range(tier["warmup_replicates"])],
            *[("measured", index) for index in range(tier["measured_replicates"])],
        ]
        for run_kind, replicate in repetitions:
            blocks = list(
                itertools.product(
                    tier["refinements"],
                    tier["scalar_observation_counts"],
                    tier["replacement_data_counts"],
                    axes["workload_phases"],
                )
            )
            rng = random.Random(
                execution["deterministic_randomization_seed"] + repetition_ordinal
            )
            repetition_ordinal += 1
            rng.shuffle(blocks)
            for refinement, count, data_count, phase in blocks:
                methods = list(axes["methods"])
                rng.shuffle(methods)
                for method in methods:
                    decision = decisions[rank_key(refinement, count, profile)]
                    spectral_rank = (
                        int(decision["rank"])
                        if method == "spectral_low_rank"
                        else None
                    )
                    decision_id = str(decision["decision_id"])
                    phase_label = "setup" if phase == "cold_setup" else "physicaluq"
                    cell_id = (
                        f"{tier['id']}-{run_kind[:1]}{replicate:02d}-b{block_index:04d}-"
                        f"n{refinement:02d}-m{count:04d}-d{data_count:03d}-"
                        f"{phase_label}-{method}"
                    )
                    cells.append(
                        CampaignCell(
                            order_index,
                            block_index,
                            str(tier["id"]),
                            str(tier["role"]),
                            run_kind,
                            replicate,
                            refinement,
                            count,
                            data_count,
                            method,
                            phase,
                            spectral_rank,
                            decision_id,
                            cell_id,
                        )
                    )
                    order_index += 1
                block_index += 1
    return cells


def cell_command(
    cell: CampaignCell,
    repository: pathlib.Path,
    output: pathlib.Path,
    runner: pathlib.Path,
    profile: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--repository",
        str(repository),
        "--profile",
        str(profile["profile"]),
        "--replicate",
        str(cell.replicate),
        "--output",
        str(output / "cells" / cell.cell_id),
        "--threads",
        str(profile["execution"]["threads"]),
        "--cgroup-mode",
        str(profile["execution"]["cgroup_mode"]),
        "--resource-method",
        cell.method,
        "--refinement",
        str(cell.refinement),
        "--observation-class",
        str(profile["axes"]["observation_class"]),
        "--sensor-count",
        str(cell.scalar_observation_count),
        "--repeated-data",
        str(cell.replacement_data_count),
        "--workload-phase",
        cell.workload_phase,
    ]
    if cell.method == "spectral_low_rank":
        if cell.spectral_rank is None:
            raise ValueError(f"spectral cell {cell.cell_id} has no frozen 5 percent rank")
        command.extend(["--spectral-rank", str(cell.spectral_rank)])
    return command


def plan_document(
    repository: pathlib.Path,
    profile_path: pathlib.Path,
    rank_path: pathlib.Path,
    runner: pathlib.Path,
    profile: Mapping[str, Any],
    decisions: Mapping[tuple[int, int, int, float, str], Mapping[str, Any]],
    cells: Sequence[CampaignCell],
    output: pathlib.Path,
    requested_tier_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = selected_evidence_tiers(profile, requested_tier_ids)
    selected_ids = [str(tier["id"]) for tier in selected]
    omitted_ids = [
        str(tier["id"])
        for tier in profile["evidence_tiers"]
        if str(tier["id"]) not in set(selected_ids)
    ]
    missing = missing_rank_keys(profile, decisions, requested_tier_ids)
    projected = projected_cell_counts(profile, requested_tier_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "plan",
        "profile": str(profile["profile"]),
        "profile_file": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "rank_decision_file": str(rank_path),
        "rank_decision_sha256": sha256_file(rank_path),
        "runner": str(runner),
        "runner_sha256": sha256_file(runner),
        "repository": str(repository),
        "output": str(output),
        "status": "blocked_missing_rank_decisions" if missing else "ready",
        "selected_evidence_tiers": selected_ids,
        "omitted_evidence_tiers": omitted_ids,
        "selected_evidence_tier_contracts": selected,
        "primary_accuracy_tolerance": 0.05,
        "context_accuracy_tolerance": 0.1,
        "sensitivity_accuracy_tolerance": 0.02,
        "missing_rank_decisions": [
            {
                "refinement": key[0],
                "scalar_observation_count": key[1],
                "alpha": key[2],
                "relative_tolerance": key[3],
                "sensor_design_id": key[4],
            }
            for key in missing
        ],
        "cell_count": len(cells),
        "warmup_cell_count": sum(cell.run_kind == "warmup" for cell in cells),
        "measured_cell_count": sum(cell.run_kind == "measured" for cell in cells),
        "projected_cell_count": projected["total"],
        "projected_warmup_cell_count": projected["warmup"],
        "projected_measured_cell_count": projected["measured"],
        "plan_sha256": canonical_hash([asdict(cell) for cell in cells]),
        "cells": [
            asdict(cell)
            | {
                "command": (
                    cell_command(cell, repository, output, runner, profile)
                    if cell.method != "spectral_low_rank" or cell.spectral_rank is not None
                    else None
                )
            }
            for cell in cells
        ],
    }


def recursive_source_clean(repository: pathlib.Path) -> tuple[bool, dict[str, str]]:
    results: dict[str, str] = {}
    for name, path in (
        ("root", repository),
        ("feec", repository / "feec"),
        ("gmrf", repository / "gmrf-rs"),
    ):
        process = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        results[name] = (
            process.stdout if process.returncode == 0
            else f"query_failed: {process.stderr.strip()}"
        )
    return all(value == "" for value in results.values()), results


def source_identity(source: Mapping[str, Any]) -> dict[str, str]:
    """Extract a fail-closed root/FEEC/GMRF identity from runner metadata."""

    try:
        components = source["submodules"]
        values = {
            "root_sha": str(source["head"]),
            "feec_sha": str(components["feec"]["head"]),
            "gmrf_sha": str(components["gmrf"]["head"]),
        }
        dirty = {
            "root_dirty": source["dirty"],
            "feec_dirty": components["feec"]["dirty"],
            "gmrf_dirty": components["gmrf"]["dirty"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"source metadata is incomplete: {error}") from error
    for name, value in values.items():
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"source metadata has invalid {name}")
    for name, value in dirty.items():
        if value is not False:
            raise ValueError(f"publication source requires {name}=false")
    return values


def load_one_row(cell_directory: pathlib.Path) -> dict[str, str]:
    path = cell_directory / "study" / "rows.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one row, found {len(rows)}")
    return rows[0]


def set_measurement_triplet(
    target: dict[str, str], source: Mapping[str, str], name: str
) -> None:
    for suffix in ("status", "value", "reason"):
        target[f"{name}_{suffix}"] = source.get(f"{name}_{suffix}", "")


def merge_phase_rows(
    setup_row: Mapping[str, str],
    query_row: Mapping[str, str],
    setup_manifest: pathlib.Path,
    query_manifest: pathlib.Path,
) -> dict[str, str]:
    mismatches = {
        field: (setup_row.get(field), query_row.get(field))
        for field in PAIR_KEY_FIELDS
        if setup_row.get(field) != query_row.get(field)
    }
    if mismatches:
        raise ValueError(f"phase rows differ in scientific identity: {mismatches}")
    merged = dict(setup_row)
    merged["resource_scope"] = "paired_method_phases"
    merged["workload_phase"] = "combined_cold_setup_query_physical_uq"
    for name in PAIR_SETUP_MEASUREMENTS:
        set_measurement_triplet(merged, setup_row, name)
    for name in PAIR_QUERY_MEASUREMENTS:
        set_measurement_triplet(merged, query_row, name)
    merged["cold_setup_runner_manifest"] = str(setup_manifest)
    merged["cold_setup_runner_manifest_sha256"] = sha256_file(setup_manifest)
    merged["query_physical_uq_runner_manifest"] = str(query_manifest)
    merged["query_physical_uq_runner_manifest_sha256"] = sha256_file(query_manifest)
    merged["phase_merge_policy"] = (
        "setup/storage/factor_rss_from_cold_setup;"
        "replacement/physical_qoi/query_rss_from_query_physical_uq"
    )
    return merged


def pair_key(cell: CampaignCell) -> tuple[str, str, int, int, int, int, str]:
    return (
        cell.evidence_tier,
        cell.run_kind,
        cell.replicate,
        cell.refinement,
        cell.scalar_observation_count,
        cell.replacement_data_count,
        cell.method,
    )


def classify_failure(return_code: int, runner_manifest: Mapping[str, Any] | None) -> str:
    if return_code == 124:
        return "timeout_censored"
    process = runner_manifest.get("process", {}) if isinstance(runner_manifest, dict) else {}
    signal_number = process.get("terminating_signal") if isinstance(process, dict) else None
    if return_code == 137 or signal_number == 9:
        return "oom_or_sigkill_censored"
    return "failed"


def summarize_cell(
    cell: CampaignCell,
    cell_directory: pathlib.Path,
    return_code: int,
    expected_source_identity: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = cell_directory / "runner-manifest.json"
    try:
        manifest = read_json(manifest_path)
        manifest_error = None
    except ValueError as error:
        manifest = None
        manifest_error = str(error)
    summary: dict[str, Any] = {
        "cell": asdict(cell),
        "return_code": return_code,
        "runner_manifest": str(manifest_path),
        "runner_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "runner_manifest_status": manifest.get("status") if manifest else None,
        "manifest_error": manifest_error,
    }
    if return_code != 0:
        summary["status"] = classify_failure(return_code, manifest)
        return summary
    if manifest is None or manifest.get("status") != "completed":
        summary["status"] = "rejected_runner_manifest"
        summary["rejection_reasons"] = [
            "successful process lacks a completed runner manifest"
        ]
        return summary
    try:
        observed_identity = source_identity(manifest["source"])
    except (KeyError, TypeError, ValueError) as error:
        summary["status"] = "rejected_source_provenance"
        summary["rejection_reasons"] = [str(error)]
        return summary
    summary["source_identity"] = observed_identity
    if observed_identity != dict(expected_source_identity):
        summary["status"] = "rejected_source_mismatch"
        summary["rejection_reasons"] = [
            "cell root/FEEC/GMRF revisions differ from the campaign freeze"
        ]
        return summary
    try:
        row = load_one_row(cell_directory)
        reasons = fresh.performance_ineligibility_reasons(
            row,
            {
                "method": cell.method,
                "spectral_rank": cell.spectral_rank,
                "sensor_target_value": cell.scalar_observation_count,
            },
        )
    except (OSError, ValueError) as error:
        summary["status"] = "rejected"
        summary["rejection_reasons"] = [str(error)]
        return summary
    if reasons:
        summary["status"] = "rejected_performance_ineligible"
        summary["rejection_reasons"] = reasons
        return summary
    summary["status"] = "eligible_phase"
    summary["row_sha256"] = sha256_file(cell_directory / "study" / "rows.csv")
    process = manifest.get("process", {}) if manifest else {}
    summary["peak_rss"] = process.get("peak_rss_archive_process")
    return summary


def write_combined_rows(path: pathlib.Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pair_results(
    cells: Sequence[CampaignCell],
    summaries: Mapping[str, Mapping[str, Any]],
    output: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, int, int, int, int, str], dict[str, CampaignCell]] = {}
    for cell in cells:
        grouped.setdefault(pair_key(cell), {})[cell.workload_phase] = cell
    pair_records: list[dict[str, Any]] = []
    combined_rows: list[dict[str, str]] = []
    for key, phases in grouped.items():
        setup = phases.get("cold_setup")
        query = phases.get("query_physical_uq")
        record: dict[str, Any] = {"pair_key": list(key)}
        if setup is None or query is None:
            record.update(status="censored_missing_phase", reason="frozen phase pair incomplete")
            pair_records.append(record)
            continue
        setup_summary = summaries.get(setup.cell_id)
        query_summary = summaries.get(query.cell_id)
        record["cold_setup_cell"] = setup.cell_id
        record["query_physical_uq_cell"] = query.cell_id
        record["cold_setup_status"] = (
            setup_summary.get("status") if setup_summary else "not_run"
        )
        record["query_physical_uq_status"] = (
            query_summary.get("status") if query_summary else "not_run"
        )
        record["cold_setup_runner_manifest"] = (
            setup_summary.get("runner_manifest") if setup_summary else None
        )
        record["cold_setup_runner_manifest_sha256"] = (
            setup_summary.get("runner_manifest_sha256") if setup_summary else None
        )
        record["query_physical_uq_runner_manifest"] = (
            query_summary.get("runner_manifest") if query_summary else None
        )
        record["query_physical_uq_runner_manifest_sha256"] = (
            query_summary.get("runner_manifest_sha256") if query_summary else None
        )
        if (
            record["cold_setup_status"] != "eligible_phase"
            or record["query_physical_uq_status"] != "eligible_phase"
        ):
            record["status"] = "censored_phase_failure_or_ineligibility"
            pair_records.append(record)
            continue
        setup_dir = output / "cells" / setup.cell_id
        query_dir = output / "cells" / query.cell_id
        setup_manifest = setup_dir / "runner-manifest.json"
        query_manifest = query_dir / "runner-manifest.json"
        try:
            merged = merge_phase_rows(
                load_one_row(setup_dir),
                load_one_row(query_dir),
                setup_manifest,
                query_manifest,
            )
        except (OSError, ValueError) as error:
            record.update(status="rejected_phase_merge", reason=str(error))
            pair_records.append(record)
            continue
        merged["cold_setup_runner_manifest"] = str(setup_manifest.relative_to(output))
        merged["query_physical_uq_runner_manifest"] = str(query_manifest.relative_to(output))
        merged["campaign_evidence_tier"] = setup.evidence_tier
        merged["campaign_evidence_role"] = setup.evidence_role
        merged["campaign_run_kind"] = setup.run_kind
        merged["campaign_replicate"] = str(setup.replicate)
        merged["campaign_rank_decision_id"] = str(setup.rank_decision_id)
        merged["campaign_cold_setup_cell_id"] = setup.cell_id
        merged["campaign_query_physical_uq_cell_id"] = query.cell_id
        if setup.run_kind == "warmup":
            record["status"] = "warmup_completed_excluded_from_evidence"
            pair_records.append(record)
            continue
        record["status"] = "eligible_combined_row"
        record["combined_row_index"] = len(combined_rows)
        combined_rows.append(merged)
        pair_records.append(record)
    return pair_records, combined_rows


def execute(
    repository: pathlib.Path,
    output: pathlib.Path,
    runner: pathlib.Path,
    profile_path: pathlib.Path,
    rank_path: pathlib.Path,
    profile: Mapping[str, Any],
    decisions: Mapping[tuple[int, int, int, float, str], Mapping[str, Any]],
    cells: Sequence[CampaignCell],
    requested_tier_ids: Sequence[str] | None = None,
) -> int:
    missing = missing_rank_keys(profile, decisions, requested_tier_ids)
    if missing:
        raise ValueError(
            f"execution is blocked by {len(missing)} missing held-out-validated 5 percent ranks"
        )
    clean, details = recursive_source_clean(repository)
    if profile["execution"]["require_clean_recursive_source"] and not clean:
        raise ValueError(f"publication execution requires a clean recursive source: {details}")
    frozen_source = fresh.git_metadata(repository)
    frozen_identity = source_identity(frozen_source)
    if output.exists():
        raise ValueError(f"refusing to overwrite campaign directory {output}")
    output.mkdir(parents=True)
    (output / "cells").mkdir()
    plan = plan_document(
        repository,
        profile_path,
        rank_path,
        runner,
        profile,
        decisions,
        cells,
        output,
        requested_tier_ids,
    )
    atomic_write_json(output / "campaign-plan.json", plan)
    campaign: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": str(profile["profile"]),
        "status": "running",
        "plan_sha256": plan["plan_sha256"],
        "profile_sha256": plan["profile_sha256"],
        "rank_decision_sha256": plan["rank_decision_sha256"],
        "selected_evidence_tiers": plan["selected_evidence_tiers"],
        "omitted_evidence_tiers": plan["omitted_evidence_tiers"],
        "source_clean_queries": details,
        "source_identity_start": frozen_identity,
        "cell_results": [],
        "pair_results": [],
    }
    manifest_path = output / "campaign-manifest.json"
    atomic_write_json(manifest_path, campaign)
    summaries: dict[str, dict[str, Any]] = {}
    for cell in cells:
        command = cell_command(cell, repository, output, runner, profile)
        process = subprocess.run(
            command,
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        cell_directory = output / "cells" / cell.cell_id
        cell_directory.mkdir(parents=True, exist_ok=True)
        (cell_directory / "orchestrator.stdout.log").write_text(
            process.stdout, encoding="utf-8"
        )
        (cell_directory / "orchestrator.stderr.log").write_text(
            process.stderr, encoding="utf-8"
        )
        summary = summarize_cell(
            cell, cell_directory, process.returncode, frozen_identity
        )
        summary["command"] = command
        summary["command_text"] = shlex.join(command)
        summaries[cell.cell_id] = summary
        campaign["cell_results"].append(summary)
        atomic_write_json(manifest_path, campaign)
    paired, combined = pair_results(cells, summaries, output)
    campaign["pair_results"] = paired
    if combined:
        write_combined_rows(output / "combined-rows.csv", combined)
        campaign["combined_rows"] = {
            "path": "combined-rows.csv",
            "sha256": sha256_file(output / "combined-rows.csv"),
            "count": len(combined),
        }
    expected_measured_pairs = sum(cell.run_kind == "measured" for cell in cells) // 2
    final_identity = source_identity(fresh.git_metadata(repository))
    campaign["source_identity_end"] = final_identity
    campaign["status"] = (
        "source_changed_during_campaign"
        if final_identity != frozen_identity
        else (
            "completed"
            if len(combined) == expected_measured_pairs
            else "completed_with_censored_or_rejected_cells"
        )
    )
    atomic_write_json(manifest_path, campaign)
    return 0 if campaign["status"] == "completed" else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--profile-file", type=pathlib.Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--rank-decisions", type=pathlib.Path, default=DEFAULT_RANK_DECISIONS
    )
    parser.add_argument("--runner", type=pathlib.Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--evidence-tier",
        action="append",
        choices=sorted(
            {
                str(tier["id"])
                for _, tiers in PROFILE_CONTRACTS.values()
                for tier in tiers
            }
        ),
        help=(
            "run one immutable evidence tier; repeat to select several. "
            "The default selects every frozen tier"
        ),
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("out/c51/juq-performance-v1")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        repository = args.repository.resolve()
        profile_path = args.profile_file.resolve()
        rank_path = args.rank_decisions.resolve()
        runner = args.runner.resolve()
        profile = read_json(profile_path)
        validate_profile(profile)
        selected_evidence_tiers(profile, args.evidence_tier)
        decisions, _ = load_rank_decisions(rank_path, profile)
        missing = missing_rank_keys(profile, decisions, args.evidence_tier)
        cells = (
            []
            if missing
            else generate_cells(profile, decisions, args.evidence_tier)
        )
        plan = plan_document(
            repository,
            profile_path,
            rank_path,
            runner,
            profile,
            decisions,
            cells,
            args.output.resolve(),
            args.evidence_tier,
        )
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if missing:
            raise ValueError(
                f"execution is blocked by {len(missing)} missing held-out-validated 5 percent ranks"
            )
        return execute(
            repository,
            args.output.resolve(),
            runner,
            profile_path,
            rank_path,
            profile,
            decisions,
            cells,
            args.evidence_tier,
        )
    except (OSError, ValueError) as error:
        print(f"C51 JUQ performance orchestration failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
