#!/usr/bin/env python3
"""Validate and analyze one C51 accuracy--resource archive.

This tool is intentionally independent of the benchmark runner.  It treats
``unavailable`` as evidence, never as zero cost, keeps observation classes and
full-/low-rank spectral rows separate, and will not issue a publication pass
for discovery profiles such as ``smoke`` or ``reference_ladder``.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "c51-accuracy-resource-v2"
STUDY_ID = "c51/accuracy-resource"
PUBLICATION_PROFILE = re.compile(r"^juq-v[1-9][0-9]*(?:[-_.][a-z0-9][a-z0-9._-]*)?$")

MEASUREMENT_FIELDS = (
    "prior_precision_nnz",
    "observation_update_nnz",
    "posterior_precision_nnz",
    "ambient_energy_nnz",
    "constraint_nnz",
    "kkt_input_nnz",
    "factor_storage_nnz",
    "certificate_factor_storage_nnz",
    "fill_ratio",
    "spectral_rank",
    "spectral_basis_storage_values",
    "spectral_gp_storage_values",
    "spectral_prepared_storage_values",
    "spectral_dense_factor_values",
    "spectral_state_feature_values",
    "spectral_observation_feature_values",
    "spectral_sampling_normal_dimension",
    "feec_assembly_ns",
    "prior_build_ns",
    "spectral_operator_recovery_ns",
    "spectral_eigensolve_ns",
    "spectral_gp_build_ns",
    "observation_update_assembly_ns",
    "observation_update_and_factor_ns",
    "symbolic_analysis_ns",
    "numeric_factor_ns",
    "combined_factor_setup_ns",
    "repeated_data_ns",
    "covariance_action_ns",
    "physical_qoi_ns",
    "sampling_ns",
    "variance_estimation_ns",
    "factor_external_peak_rss_bytes",
    "query_external_peak_rss_bytes",
    "posterior_mean_relative_error",
    "covariance_action_relative_error",
    "physical_covariance_action_relative_error",
    "physical_variance_relative_error",
    "physical_transformed_variance_relative_error",
    "physical_mean_relative_error",
    "physical_transformed_variance_min_raw",
    "physical_transformed_variance_negative_count",
)

SIGNED_MEASUREMENT_FIELDS = {"physical_transformed_variance_min_raw"}

SPECTRAL_CONSTRUCTION_TIME_FIELDS = (
    "spectral_operator_recovery_ns",
    "spectral_eigensolve_ns",
    "spectral_gp_build_ns",
)

SPECTRAL_STORAGE_AUDIT_FIELDS = (
    "spectral_basis_storage_values",
    "spectral_gp_storage_values",
    "spectral_prepared_storage_values",
    "spectral_dense_factor_values",
    "spectral_state_feature_values",
    "spectral_observation_feature_values",
    "spectral_sampling_normal_dimension",
)

REQUIRED_BASE_FIELDS = (
    "schema_version",
    "study_id",
    "method",
    "available",
    "unavailable_reason",
    "discrete_law_id",
    "same_discrete_law_eligible",
    "matched_physical_qoi_eligible",
    "full_uq_workload_eligible",
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

EQUALITY_FIELDS = (
    "posterior_mean_relative_error",
    "covariance_action_relative_error",
    "physical_covariance_action_relative_error",
    "physical_variance_relative_error",
    "physical_transformed_variance_relative_error",
    "physical_mean_relative_error",
)

ACCURACY_FIELDS = (
    "physical_covariance_action_relative_error",
    "physical_variance_relative_error",
    "physical_transformed_variance_relative_error",
    "physical_mean_relative_error",
)

SETUP_TIME_FIELDS = (
    "feec_assembly_ns",
    "prior_build_ns",
    "observation_update_and_factor_ns",
)

QUERY_TIME_FIELDS = (
    "covariance_action_ns",
    "physical_qoi_ns",
    "sampling_ns",
    "variance_estimation_ns",
)

COLD_TIME_FIELDS = SETUP_TIME_FIELDS + ("repeated_data_ns",) + QUERY_TIME_FIELDS

REFERENCE_METHODS = {"fm_direct", "reference", "ref_full_w"}


class ArchiveError(ValueError):
    """Raised when an archive violates its declared schema or semantics."""


@dataclass(frozen=True)
class Measurement:
    status: str
    value: float | None
    reason: str


@dataclass(frozen=True)
class ResourceSummary:
    cold_ns: float
    amortized_ns_per_dataset: float
    stored_values: float
    peak_rss_bytes: float


@dataclass(frozen=True)
class Aggregate:
    group_key: tuple[str, ...]
    method_label: str
    observation_class: str
    refinement: int
    repeated_data_count: int
    row_count: int
    feasible: bool
    cold_ns: float | None
    amortized_ns_per_dataset: float | None
    stored_values: float | None
    peak_rss_bytes: float | None


@dataclass(frozen=True)
class Crossover:
    group_key: tuple[str, ...]
    observation_class: str
    refinement: int
    comparator: str
    compact_to_comparator_cold: float
    compact_to_comparator_amortized: float
    compact_to_comparator_stored: float
    compact_to_comparator_peak_rss: float
    practical_advantage: bool


@dataclass(frozen=True)
class RankPoint:
    rank: int
    rank_fraction: float
    spectral_class: str
    physical_errors: dict[str, float | None]
    joint_feasible: bool
    mandatory_reference_rank: bool


@dataclass(frozen=True)
class RankThreshold:
    metric: str
    smallest_eligible_rank: int | None
    smallest_eligible_rank_fraction: float | None
    bracket_status: str


@dataclass(frozen=True)
class RankCurve:
    group_key: tuple[str, ...]
    observation_class: str
    refinement: int
    sensor_count: int
    repeated_data_count: int
    ambient_dimension: int
    points: tuple[RankPoint, ...]
    thresholds: tuple[RankThreshold, ...]
    mandatory_reference_ranks_present: tuple[int, ...]


@dataclass
class Analysis:
    profile: str
    evidence_tier: str
    row_count: int
    availability_counts: dict[str, int]
    full_rank_checks: list[dict[str, Any]]
    rank_curves: list[RankCurve]
    aggregates: list[Aggregate]
    crossovers: list[Crossover]
    blockers: list[str]
    limitations: list[str]
    decision: str

    def to_json(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "evidence_tier": self.evidence_tier,
            "row_count": self.row_count,
            "availability_counts": self.availability_counts,
            "full_rank_checks": self.full_rank_checks,
            "rank_curves": [asdict(value) for value in self.rank_curves],
            "aggregates": [asdict(value) for value in self.aggregates],
            "crossovers": [asdict(value) for value in self.crossovers],
            "blockers": self.blockers,
            "limitations": self.limitations,
            "decision": self.decision,
        }


@dataclass(frozen=True)
class GateConfig:
    full_rank_tolerance: float = 1.0e-8
    maximum_physical_error: float = 0.10
    practical_improvement_ratio: float = 0.75
    maximum_primary_resource_ratio: float = 1.00
    maximum_other_resource_ratio: float = 1.10
    minimum_claim_refinements: int = 2


def parse_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ArchiveError(f"row {row_number}: {field} must be true or false, got {value!r}")


def parse_finite(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ArchiveError(f"row {row_number}: {field} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise ArchiveError(f"row {row_number}: {field} must be finite, got {value!r}")
    return parsed


def parse_nonnegative(value: str, field: str, row_number: int) -> float:
    parsed = parse_finite(value, field, row_number)
    if parsed < 0.0:
        raise ArchiveError(f"row {row_number}: {field} must be non-negative, got {parsed}")
    return parsed


def measurement(row: dict[str, str], name: str, row_number: int) -> Measurement:
    status = row[f"{name}_status"].strip()
    raw_value = row[f"{name}_value"].strip()
    reason = row[f"{name}_reason"].strip()
    if status == "measured":
        if not raw_value:
            raise ArchiveError(f"row {row_number}: measured {name} has no value")
        if reason:
            raise ArchiveError(f"row {row_number}: measured {name} must not have a reason")
        parser = parse_finite if name in SIGNED_MEASUREMENT_FIELDS else parse_nonnegative
        return Measurement(status, parser(raw_value, name, row_number), "")
    if status == "not_recorded":
        if raw_value or reason:
            raise ArchiveError(
                f"row {row_number}: not_recorded {name} must have empty value and reason"
            )
        return Measurement(status, None, "")
    if status == "unavailable":
        if raw_value:
            raise ArchiveError(f"row {row_number}: unavailable {name} must not have a value")
        if not reason:
            raise ArchiveError(f"row {row_number}: unavailable {name} requires a reason")
        return Measurement(status, None, reason)
    raise ArchiveError(
        f"row {row_number}: {name}_status must be measured, not_recorded, or unavailable"
    )


def measured_value(row: dict[str, str], name: str, row_number: int) -> float | None:
    return measurement(row, name, row_number).value


def require_measured(row: dict[str, str], name: str, row_number: int) -> float:
    item = measurement(row, name, row_number)
    if item.status != "measured":
        suffix = f": {item.reason}" if item.reason else ""
        raise ArchiveError(
            f"row {row_number}: required {name} is {item.status}{suffix}"
        )
    assert item.value is not None
    return item.value


def parse_manifest(path: Path) -> dict[str, Any]:
    """Parse the deliberately simple manifest subset emitted by the harness."""

    result: dict[str, Any] = {}
    section: str | None = None
    for line_number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = original.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if section:
                current = result.get(section)
                if current == {}:
                    current = []
                    result[section] = current
                if not isinstance(current, list):
                    raise ArchiveError(
                        f"manifest line {line_number}: mixed list and mapping section {section}"
                    )
                current.append(parse_yaml_scalar(stripped[2:]))
            continue
        if ":" not in stripped:
            raise ArchiveError(f"manifest line {line_number}: expected key: value")
        key, raw = stripped.split(":", 1)
        raw = raw.strip()
        if line.startswith("  "):
            if section:
                nested = result.setdefault(section, {})
                if isinstance(nested, dict):
                    nested[key] = parse_yaml_scalar(raw)
            continue
        section = key if not raw else None
        if raw:
            result[key] = parse_yaml_scalar(raw)
        elif key not in result:
            result[key] = {}
    return result


def parse_yaml_scalar(raw: str) -> Any:
    if raw == "null":
        return None
    if raw in {"true", "false"}:
        return raw == "true"
    if raw.startswith('"'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ArchiveError(f"invalid quoted manifest scalar {raw!r}") from error
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    try:
        return int(raw)
    except ValueError:
        return raw


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ArchiveError("rows.csv has no header")
        required = set(REQUIRED_BASE_FIELDS)
        for name in MEASUREMENT_FIELDS:
            required.update((f"{name}_status", f"{name}_value", f"{name}_reason"))
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ArchiveError(f"rows.csv is missing required columns: {', '.join(missing)}")
        return list(reader)


def validate_archive(manifest: dict[str, Any], rows: Sequence[dict[str, str]]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArchiveError(f"manifest schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("study_id") != STUDY_ID:
        raise ArchiveError(f"manifest study_id must be {STUDY_ID!r}")
    if manifest.get("rows_csv") != "rows.csv":
        raise ArchiveError("manifest rows_csv must be 'rows.csv'")
    frames = manifest.get("comparison_frames")
    required_frames = {
        "same_discrete_law_representation",
        "matched_physical_qoi_method",
    }
    if not isinstance(frames, list) or not required_frames.issubset(frames):
        raise ArchiveError("manifest is missing the two required comparison_frames")
    if manifest.get("row_count") != len(rows):
        raise ArchiveError(
            f"manifest row_count {manifest.get('row_count')!r} does not match {len(rows)} rows"
        )
    for row_number, row in enumerate(rows, 2):
        if row["schema_version"] != SCHEMA_VERSION or row["study_id"] != STUDY_ID:
            raise ArchiveError(f"row {row_number}: schema or study identifier mismatch")
        available = parse_bool(row["available"], "available", row_number)
        reason = row["unavailable_reason"].strip()
        if available and reason:
            raise ArchiveError(f"row {row_number}: available row has unavailable_reason")
        if not available and not reason:
            raise ArchiveError(f"row {row_number}: unavailable row requires a reason")
        for boolean_field in (
            "same_discrete_law_eligible",
            "matched_physical_qoi_eligible",
            "full_uq_workload_eligible",
        ):
            parse_bool(row[boolean_field], boolean_field, row_number)
        for numeric_field in (
            "refinement",
            "spatial_dimension",
            "form_degree",
            "alpha",
            "kappa",
            "tau",
            "sensor_count",
            "sensor_density",
            "repeated_data_count",
            "posterior_sample_count",
            "seed",
            "latent_dimension",
            "ambient_dimension",
            "constraint_dimension",
            "observation_nnz",
        ):
            parse_nonnegative(row[numeric_field], numeric_field, row_number)
        for name in MEASUREMENT_FIELDS:
            measurement(row, name, row_number)


def evidence_tier(profile: str) -> str:
    normalized = profile.strip().lower()
    if PUBLICATION_PROFILE.fullmatch(normalized):
        return "publication"
    return "discovery"


def is_spectral(row: dict[str, str]) -> bool:
    return "spect" in row["method"].lower()


def spectral_class(row: dict[str, str], row_number: int) -> str | None:
    if not is_spectral(row):
        return None
    rank = measured_value(row, "spectral_rank", row_number)
    if rank is None:
        return "spectral_unknown_rank"
    ambient = int(parse_nonnegative(row["ambient_dimension"], "ambient_dimension", row_number))
    if rank >= ambient or "full" in row["method"].lower():
        return "spectral_full_rank"
    return f"spectral_low_rank_r{int(rank)}"


def method_label(row: dict[str, str], row_number: int) -> str:
    classification = spectral_class(row, row_number)
    return f"{row['method']}:{classification}" if classification else row["method"]


def row_group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row[name]
        for name in (
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
        )
    )


def row_is_reference(row: dict[str, str]) -> bool:
    method = row["method"].lower()
    return method in REFERENCE_METHODS or "direct" in method or "reference" in method


def row_feasible(row: dict[str, str], row_number: int, config: GateConfig) -> bool:
    if not parse_bool(row["available"], "available", row_number):
        return False
    if not parse_bool(
        row["matched_physical_qoi_eligible"], "matched_physical_qoi_eligible", row_number
    ):
        return False
    if row_is_reference(row):
        return True
    for name in ACCURACY_FIELDS:
        value = measured_value(row, name, row_number)
        if value is None or value > config.maximum_physical_error:
            return False
    return True


def row_resources(row: dict[str, str], row_number: int) -> ResourceSummary:
    repeated = int(
        parse_nonnegative(row["repeated_data_count"], "repeated_data_count", row_number)
    )
    if repeated <= 0:
        raise ArchiveError(f"row {row_number}: repeated_data_count must be positive")
    setup_ns = sum(require_measured(row, name, row_number) for name in SETUP_TIME_FIELDS)
    repeated_data_ns = require_measured(row, "repeated_data_ns", row_number)
    query_ns = sum(require_measured(row, name, row_number) for name in QUERY_TIME_FIELDS)
    # The archive performs D replacement-data mean calculations and one full-UQ
    # query bundle.  Cold time estimates one replacement mean plus that bundle;
    # amortized time reports exactly the measured archive workload divided by D.
    cold_ns = setup_ns + repeated_data_ns / repeated + query_ns
    amortized_ns = (setup_ns + repeated_data_ns + query_ns) / repeated

    method = row["method"].lower()
    if is_spectral(row):
        spectral_values = {
            name: require_measured(row, name, row_number)
            for name in SPECTRAL_STORAGE_AUDIT_FIELDS
        }
        subphase_total = sum(
            require_measured(row, name, row_number)
            for name in SPECTRAL_CONSTRUCTION_TIME_FIELDS
        )
        prior_build = require_measured(row, "prior_build_ns", row_number)
        timing_scale = max(prior_build, subphase_total, 1.0)
        if abs(prior_build - subphase_total) > max(1.0, 1.0e-6 * timing_scale):
            raise ArchiveError(
                f"row {row_number}: spectral prior_build_ns ({prior_build:g}) does not equal "
                f"operator recovery + eigensolve + GP build ({subphase_total:g})"
            )
        # The prepared posterior already contains its dense factor and state/
        # observation features.  Add the eigensolver basis allocation, but do
        # not also add the discardable GP copy or double-count its components.
        stored = (
            spectral_values["spectral_basis_storage_values"]
            + spectral_values["spectral_prepared_storage_values"]
        )
    elif "aug" in method or "kkt" in method:
        factor = require_measured(row, "factor_storage_nnz", row_number)
        stored = require_measured(row, "kkt_input_nnz", row_number) + factor
    else:
        factor = require_measured(row, "factor_storage_nnz", row_number)
        stored = require_measured(row, "posterior_precision_nnz", row_number) + factor
    factor_rss = require_measured(row, "factor_external_peak_rss_bytes", row_number)
    query_rss = require_measured(row, "query_external_peak_rss_bytes", row_number)
    if cold_ns <= 0.0 or stored <= 0.0 or factor_rss <= 0.0 or query_rss <= 0.0:
        raise ArchiveError(
            f"row {row_number}: required time, stored-resource, and peak-RSS values must be positive"
        )
    return ResourceSummary(
        cold_ns=cold_ns,
        amortized_ns_per_dataset=amortized_ns,
        stored_values=stored,
        peak_rss_bytes=max(factor_rss, query_rss),
    )


def publication_metadata_blockers(manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if manifest.get("compiler_profile") != "release":
        blockers.append("publication evidence requires compiler_profile=release")
    for name in ("root_sha", "feec_sha", "gmrf_sha"):
        if not manifest.get(name):
            blockers.append(f"publication evidence requires {name}")
    for name in ("root_dirty", "feec_dirty", "gmrf_dirty"):
        if manifest.get(name) is not False:
            blockers.append(f"publication evidence requires {name}=false")
    hardware = manifest.get("hardware")
    if not isinstance(hardware, dict):
        blockers.append("publication evidence requires hardware metadata")
    else:
        for name in (
            "host",
            "cpu_model",
            "logical_cores",
            "total_memory_bytes",
            "operating_system",
            "kernel",
            "thread_affinity",
        ):
            if hardware.get(name) is None:
                blockers.append(f"publication evidence requires hardware.{name}")
    return blockers


def full_rank_checks(
    rows: Sequence[dict[str, str]], config: GateConfig
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for row_number, row in enumerate(rows, 2):
        if not parse_bool(row["available"], "available", row_number):
            continue
        if spectral_class(row, row_number) != "spectral_full_rank":
            continue
        if not parse_bool(
            row["same_discrete_law_eligible"], "same_discrete_law_eligible", row_number
        ):
            blockers.append(f"row {row_number}: full-rank spectral row is not same-law eligible")
            continue
        equality_match = any(
            candidate_number != row_number
            and parse_bool(candidate["available"], "available", candidate_number)
            and not is_spectral(candidate)
            and parse_bool(
                candidate["same_discrete_law_eligible"],
                "same_discrete_law_eligible",
                candidate_number,
            )
            and candidate["discrete_law_id"] == row["discrete_law_id"]
            and row_group_key(candidate) == row_group_key(row)
            for candidate_number, candidate in enumerate(rows, 2)
        )
        if not equality_match:
            blockers.append(
                f"row {row_number}: full-rank spectral row has no available same-law nonspectral match"
            )
        values: dict[str, float] = {}
        missing: list[str] = []
        for name in EQUALITY_FIELDS:
            value = measured_value(row, name, row_number)
            if value is None:
                missing.append(name)
            else:
                values[name] = value
        maximum = max(values.values(), default=math.inf)
        passed = not missing and maximum <= config.full_rank_tolerance
        checks.append(
            {
                "row_number": row_number,
                "observation_class": row["observation_class"],
                "refinement": int(row["refinement"]),
                "maximum_relative_error": maximum,
                "missing": missing,
                "passed": passed,
            }
        )
        if not passed:
            blockers.append(
                f"row {row_number}: full-rank spectral equality failed "
                f"(max={maximum:.6g}, missing={','.join(missing) or 'none'})"
            )
    if not checks:
        blockers.append("no available full-rank same-law spectral equality row")
    return checks, blockers


def rank_threshold(
    points: Sequence[RankPoint], metric: str, maximum_error: float
) -> RankThreshold:
    if metric == "joint":
        observed = [
            point
            for point in points
            if all(value is not None for value in point.physical_errors.values())
        ]
        passing = [point for point in observed if point.joint_feasible]
    else:
        observed = [point for point in points if point.physical_errors[metric] is not None]
        passing = [
            point
            for point in observed
            if point.physical_errors[metric] is not None
            and point.physical_errors[metric] <= maximum_error
        ]
    if not observed:
        return RankThreshold(metric, None, None, "no_measurements")
    if not passing:
        return RankThreshold(metric, None, None, "threshold_not_reached")
    smallest = min(passing, key=lambda point: point.rank)
    lower_observed = [point for point in observed if point.rank < smallest.rank]
    status = "bracketed" if lower_observed else "threshold_below_tested_grid"
    return RankThreshold(metric, smallest.rank, smallest.rank_fraction, status)


def rank_error_curves(
    rows: Sequence[dict[str, str]], config: GateConfig
) -> list[RankCurve]:
    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_number, row in enumerate(rows, 2):
        if not parse_bool(row["available"], "available", row_number) or not is_spectral(row):
            continue
        if not parse_bool(
            row["matched_physical_qoi_eligible"],
            "matched_physical_qoi_eligible",
            row_number,
        ):
            continue
        grouped[row_group_key(row)].append((row_number, row))

    curves: list[RankCurve] = []
    for group_key, members in sorted(grouped.items()):
        ambient_values = {
            int(parse_nonnegative(row["ambient_dimension"], "ambient_dimension", row_number))
            for row_number, row in members
        }
        if len(ambient_values) != 1:
            raise ArchiveError(f"spectral rank curve {group_key!r} mixes ambient dimensions")
        ambient = ambient_values.pop()
        if ambient <= 0:
            raise ArchiveError(f"spectral rank curve {group_key!r} has zero ambient dimension")
        by_rank: dict[int, list[tuple[int, dict[str, str]]]] = defaultdict(list)
        for row_number, row in members:
            raw_rank = require_measured(row, "spectral_rank", row_number)
            if not raw_rank.is_integer() or raw_rank <= 0.0 or raw_rank > ambient:
                raise ArchiveError(
                    f"row {row_number}: spectral_rank must be an integer in [1, {ambient}]"
                )
            by_rank[int(raw_rank)].append((row_number, row))

        points: list[RankPoint] = []
        for rank, rank_members in sorted(by_rank.items()):
            errors: dict[str, float | None] = {}
            for name in ACCURACY_FIELDS:
                values = [
                    measured_value(row, name, row_number)
                    for row_number, row in rank_members
                ]
                measured = [value for value in values if value is not None]
                errors[name] = statistics.median(measured) if len(measured) == len(values) else None
            points.append(
                RankPoint(
                    rank=rank,
                    rank_fraction=rank / ambient,
                    spectral_class="spectral_full_rank"
                    if rank >= ambient
                    else "spectral_low_rank",
                    physical_errors=errors,
                    joint_feasible=all(
                        value is not None and value <= config.maximum_physical_error
                        for value in errors.values()
                    ),
                    mandatory_reference_rank=rank in {2000, 2048},
                )
            )
        thresholds = tuple(
            rank_threshold(points, metric, config.maximum_physical_error)
            for metric in (*ACCURACY_FIELDS, "joint")
        )
        first = members[0][1]
        curves.append(
            RankCurve(
                group_key=group_key,
                observation_class=first["observation_class"],
                refinement=int(first["refinement"]),
                sensor_count=int(first["sensor_count"]),
                repeated_data_count=int(first["repeated_data_count"]),
                ambient_dimension=ambient,
                points=tuple(points),
                thresholds=thresholds,
                mandatory_reference_ranks_present=tuple(
                    point.rank for point in points if point.mandatory_reference_rank
                ),
            )
        )
    return curves


def aggregate_rows(
    rows: Sequence[dict[str, str]], config: GateConfig
) -> tuple[list[Aggregate], list[str]]:
    grouped: dict[tuple[tuple[str, ...], str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    blockers: list[str] = []
    for row_number, row in enumerate(rows, 2):
        if not parse_bool(row["available"], "available", row_number):
            continue
        if not parse_bool(
            row["full_uq_workload_eligible"], "full_uq_workload_eligible", row_number
        ):
            continue
        grouped[(row_group_key(row), method_label(row, row_number))].append((row_number, row))

    aggregates: list[Aggregate] = []
    for (group_key, label), members in sorted(grouped.items()):
        resources: list[ResourceSummary] = []
        feasible_values: list[bool] = []
        for row_number, row in members:
            feasible_values.append(row_feasible(row, row_number, config))
            try:
                resources.append(row_resources(row, row_number))
            except ArchiveError as error:
                blockers.append(str(error))
        complete = len(resources) == len(members)
        aggregates.append(
            Aggregate(
                group_key=group_key,
                method_label=label,
                observation_class=members[0][1]["observation_class"],
                refinement=int(members[0][1]["refinement"]),
                repeated_data_count=int(members[0][1]["repeated_data_count"]),
                row_count=len(members),
                feasible=all(feasible_values),
                cold_ns=statistics.median(value.cold_ns for value in resources)
                if complete
                else None,
                amortized_ns_per_dataset=statistics.median(
                    value.amortized_ns_per_dataset for value in resources
                )
                if complete
                else None,
                stored_values=statistics.median(value.stored_values for value in resources)
                if complete
                else None,
                peak_rss_bytes=statistics.median(value.peak_rss_bytes for value in resources)
                if complete
                else None,
            )
        )
    return aggregates, sorted(set(blockers))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def compute_crossovers(
    aggregates: Sequence[Aggregate], config: GateConfig
) -> list[Crossover]:
    grouped: dict[tuple[str, ...], list[Aggregate]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[aggregate.group_key].append(aggregate)
    output: list[Crossover] = []
    for group_key, members in sorted(grouped.items()):
        compact = next(
            (value for value in members if value.method_label == "whitney_compact" and value.feasible),
            None,
        )
        if compact is None:
            continue
        for comparator in members:
            if comparator is compact or not comparator.feasible:
                continue
            cold = safe_ratio(compact.cold_ns, comparator.cold_ns)
            amortized = safe_ratio(
                compact.amortized_ns_per_dataset, comparator.amortized_ns_per_dataset
            )
            stored = safe_ratio(compact.stored_values, comparator.stored_values)
            rss = safe_ratio(compact.peak_rss_bytes, comparator.peak_rss_bytes)
            if None in (cold, amortized, stored, rss):
                continue
            assert cold is not None and amortized is not None and stored is not None and rss is not None
            primary_ratios = (cold, amortized, rss)
            practical = (
                min(primary_ratios) <= config.practical_improvement_ratio
                and max(primary_ratios) <= config.maximum_primary_resource_ratio
                and stored <= config.maximum_other_resource_ratio
            )
            output.append(
                Crossover(
                    group_key=group_key,
                    observation_class=compact.observation_class,
                    refinement=compact.refinement,
                    comparator=comparator.method_label,
                    compact_to_comparator_cold=cold,
                    compact_to_comparator_amortized=amortized,
                    compact_to_comparator_stored=stored,
                    compact_to_comparator_peak_rss=rss,
                    practical_advantage=practical,
                )
            )
    return output


def decision_blockers(
    manifest: dict[str, Any],
    rows: Sequence[dict[str, str]],
    checks: Sequence[dict[str, Any]],
    rank_curves: Sequence[RankCurve],
    aggregates: Sequence[Aggregate],
    crossovers: Sequence[Crossover],
    config: GateConfig,
) -> list[str]:
    blockers: list[str] = []
    profile = str(manifest.get("profile") or "")
    if evidence_tier(profile) != "publication":
        blockers.append(
            f"profile {profile!r} is discovery evidence; only a frozen juq-vN profile can clear C51"
        )
    else:
        blockers.extend(publication_metadata_blockers(manifest))
    if not checks or any(not bool(check["passed"]) for check in checks):
        blockers.append("full-rank same-law spectral equality gate has not passed")

    classes = {
        row["observation_class"]
        for row_number, row in enumerate(rows, 2)
        if parse_bool(row["available"], "available", row_number)
    }
    for required_class in ("local_physical_b", "local_selectors"):
        if required_class not in classes:
            blockers.append(f"missing required observation class {required_class}")

    local_groups: dict[tuple[str, ...], list[Crossover]] = defaultdict(list)
    for crossover in crossovers:
        if crossover.observation_class == "local_physical_b":
            local_groups[crossover.group_key].append(crossover)
    winning_refinements: set[int] = set()
    for values in local_groups.values():
        has_low_rank_spectral = any(":spectral_low_rank" in value.comparator for value in values)
        if has_low_rank_spectral and values and all(value.practical_advantage for value in values):
            winning_refinements.add(values[0].refinement)
    if len(winning_refinements) < config.minimum_claim_refinements:
        blockers.append(
            "compact method lacks practical wins against every feasible comparator, including "
            f"low-rank spectral, on {config.minimum_claim_refinements} local-physical refinements"
        )
    if not any(
        aggregate.feasible and ":spectral_low_rank" in aggregate.method_label
        for aggregate in aggregates
    ):
        blockers.append("no matched-accuracy feasible low-rank spectral comparator")
    if evidence_tier(profile) == "publication":
        local_curves = [
            curve for curve in rank_curves if curve.observation_class == "local_physical_b"
        ]
        unbracketed = [
            curve
            for curve in local_curves
            if next(
                threshold for threshold in curve.thresholds if threshold.metric == "joint"
            ).bracket_status
            != "bracketed"
        ]
        if not local_curves:
            blockers.append("no empirical local-physical spectral rank-error curve")
        elif unbracketed:
            cells = ", ".join(
                f"ref={curve.refinement}/m={curve.sensor_count}/D={curve.repeated_data_count}"
                for curve in unbracketed[:6]
            )
            suffix = " ..." if len(unbracketed) > 6 else ""
            blockers.append(
                "spectral rank grid does not bracket the joint physical-accuracy threshold in "
                f"{len(unbracketed)} local-physical cells ({cells}{suffix})"
            )
    return sorted(set(blockers))


def analyze(
    manifest: dict[str, Any], rows: Sequence[dict[str, str]], config: GateConfig
) -> Analysis:
    validate_archive(manifest, rows)
    checks, equality_blockers = full_rank_checks(rows, config)
    rank_curves = rank_error_curves(rows, config)
    aggregates, resource_blockers = aggregate_rows(rows, config)
    crossovers = compute_crossovers(aggregates, config)
    blockers = equality_blockers + resource_blockers
    blockers.extend(
        decision_blockers(
            manifest, rows, checks, rank_curves, aggregates, crossovers, config
        )
    )
    availability = Counter()
    for row_number, row in enumerate(rows, 2):
        status = "available" if parse_bool(row["available"], "available", row_number) else "unavailable"
        availability[f"{row['observation_class']}|{method_label(row, row_number)}|{status}"] += 1
    limitations = [
        "ratios use archive medians; confidence intervals require replicate-aware production rows",
        "stored-value counts are not bytes and do not replace external peak RSS",
        "cold time is setup plus one replacement-data mean and one full-UQ query bundle; "
        "amortized time is the measured archive workload (D means, one query bundle) divided by D",
        "matched accuracy is evaluated only from the relative physical-error fields present in schema v2",
        "spectral stored values equal eigensolver-basis plus prepared-posterior storage; the "
        "discardable GP copy and prepared-posterior components are audited but not double-counted",
        "spectral prior_build_ns includes operator recovery, eigensolve, and GP construction; "
        "the three subphases are checked but not added a second time",
        "observation classes are never pooled; each crossover row retains its class",
    ]
    unique_blockers = sorted(set(blockers))
    return Analysis(
        profile=str(manifest.get("profile") or ""),
        evidence_tier=evidence_tier(str(manifest.get("profile") or "")),
        row_count=len(rows),
        availability_counts=dict(sorted(availability.items())),
        full_rank_checks=checks,
        rank_curves=rank_curves,
        aggregates=aggregates,
        crossovers=crossovers,
        blockers=unique_blockers,
        limitations=limitations,
        decision="PASS" if not unique_blockers else "FAIL",
    )


def markdown(analysis: Analysis) -> str:
    lines = [
        "# C51 archive decision report",
        "",
        f"- Profile: `{analysis.profile}`",
        f"- Evidence tier: **{analysis.evidence_tier}**",
        f"- Rows: {analysis.row_count}",
        f"- Decision: **{analysis.decision}**",
        "",
        "## Decision blockers",
        "",
    ]
    lines.extend(f"- {value}" for value in analysis.blockers)
    if not analysis.blockers:
        lines.append("- None.")
    lines.extend(["", "## Availability by observation class and method", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in analysis.availability_counts.items())
    lines.extend(["", "## Full-rank same-law equality", ""])
    lines.append("| row | observation | refinement | max relative error | pass |")
    lines.append("|---:|---|---:|---:|:---:|")
    for check in analysis.full_rank_checks:
        maximum = check["maximum_relative_error"]
        rendered = "missing" if not math.isfinite(maximum) else f"{maximum:.6g}"
        lines.append(
            f"| {check['row_number']} | {check['observation_class']} | "
            f"{check['refinement']} | {rendered} | {'yes' if check['passed'] else 'no'} |"
        )
    if not analysis.full_rank_checks:
        lines.append("| - | - | - | - | no rows |")
    lines.extend(["", "## Empirical spectral rank--error curves", ""])
    lines.append(
        "| observation | refinement | m | D | N | fixed r | r/N | cov action | scalar var | "
        "variance vector | mean | joint pass | mandatory 2000/2048 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for curve in analysis.rank_curves:
        for point in curve.points:
            error = point.physical_errors
            lines.append(
                f"| {curve.observation_class} | {curve.refinement} | {curve.sensor_count} | "
                f"{curve.repeated_data_count} | {curve.ambient_dimension} | {point.rank} | "
                f"{point.rank_fraction:.6g} | "
                f"{format_number(error['physical_covariance_action_relative_error'])} | "
                f"{format_number(error['physical_variance_relative_error'])} | "
                f"{format_number(error['physical_transformed_variance_relative_error'])} | "
                f"{format_number(error['physical_mean_relative_error'])} | "
                f"{'yes' if point.joint_feasible else 'no'} | "
                f"{'yes' if point.mandatory_reference_rank else 'no'} |"
            )
    if not analysis.rank_curves:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | no | no |")
    lines.extend(["", "### Smallest measured rank meeting the 10% envelope", ""])
    lines.append(
        "| observation | refinement | m | D | metric | smallest fixed r | r/N | grid status |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---|")
    for curve in analysis.rank_curves:
        for threshold in curve.thresholds:
            lines.append(
                f"| {curve.observation_class} | {curve.refinement} | {curve.sensor_count} | "
                f"{curve.repeated_data_count} | {threshold.metric} | "
                f"{format_integer(threshold.smallest_eligible_rank)} | "
                f"{format_number(threshold.smallest_eligible_rank_fraction)} | "
                f"{threshold.bracket_status} |"
            )
    lines.extend(["", "## Matched-accuracy feasible resource rows", ""])
    lines.append(
        "| observation | refinement | method | feasible | cold ms | amortized ms/data | "
        "stored values | peak RSS MiB |"
    )
    lines.append("|---|---:|---|:---:|---:|---:|---:|---:|")
    for value in analysis.aggregates:
        lines.append(
            f"| {value.observation_class} | {value.refinement} | {value.method_label} | "
            f"{'yes' if value.feasible else 'no'} | {format_scaled(value.cold_ns, 1e6)} | "
            f"{format_scaled(value.amortized_ns_per_dataset, 1e6)} | "
            f"{format_number(value.stored_values)} | {format_scaled(value.peak_rss_bytes, 2**20)} |"
        )
    lines.extend(["", "## Compact-to-comparator crossover ratios", ""])
    lines.append(
        "| observation | refinement | comparator | cold | amortized | stored | peak RSS | advantage |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|:---:|")
    for value in analysis.crossovers:
        lines.append(
            f"| {value.observation_class} | {value.refinement} | {value.comparator} | "
            f"{value.compact_to_comparator_cold:.3f} | "
            f"{value.compact_to_comparator_amortized:.3f} | "
            f"{value.compact_to_comparator_stored:.3f} | "
            f"{value.compact_to_comparator_peak_rss:.3f} | "
            f"{'yes' if value.practical_advantage else 'no'} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in analysis.limitations)
    return "\n".join(lines) + "\n"


def format_number(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6g}"


def format_integer(value: int | None) -> str:
    return "missing" if value is None else str(value)


def format_scaled(value: float | None, scale: float) -> str:
    return "missing" if value is None else f"{value / scale:.6g}"


def resolve_paths(archive: Path, manifest_name: str) -> tuple[Path, Path]:
    if archive.is_dir():
        return archive / "rows.csv", archive / manifest_name
    raise ArchiveError(f"archive path must be a directory, got {archive}")


def positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="directory containing rows.csv and manifest")
    parser.add_argument("--manifest", default="manifest.yaml", help="manifest filename")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--full-rank-tolerance", type=positive_finite, default=1.0e-8)
    parser.add_argument("--maximum-physical-error", type=positive_finite, default=0.10)
    parser.add_argument("--practical-improvement-ratio", type=positive_finite, default=0.75)
    parser.add_argument("--maximum-primary-resource-ratio", type=positive_finite, default=1.00)
    parser.add_argument("--maximum-other-resource-ratio", type=positive_finite, default=1.10)
    parser.add_argument("--minimum-claim-refinements", type=int, default=2)
    args = parser.parse_args(argv)
    if args.minimum_claim_refinements < 1:
        parser.error("--minimum-claim-refinements must be positive")
    config = GateConfig(
        full_rank_tolerance=args.full_rank_tolerance,
        maximum_physical_error=args.maximum_physical_error,
        practical_improvement_ratio=args.practical_improvement_ratio,
        maximum_primary_resource_ratio=args.maximum_primary_resource_ratio,
        maximum_other_resource_ratio=args.maximum_other_resource_ratio,
        minimum_claim_refinements=args.minimum_claim_refinements,
    )
    try:
        rows_path, manifest_path = resolve_paths(args.archive, args.manifest)
        manifest = parse_manifest(manifest_path)
        rows = load_rows(rows_path)
        result = analyze(manifest, rows, config)
    except (ArchiveError, OSError) as error:
        print(f"C51 archive error: {error}", file=sys.stderr)
        return 2
    report = markdown(result)
    if args.markdown_out:
        args.markdown_out.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.require_pass and result.decision != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
