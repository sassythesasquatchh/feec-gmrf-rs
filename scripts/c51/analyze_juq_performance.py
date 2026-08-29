#!/usr/bin/env python3
"""Adjudicate the frozen C51 JUQ matched-accuracy performance campaign."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "c51_juq_campaign", SCRIPT_DIR / "run_juq_performance.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
campaign = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = campaign
RUNNER_SPEC.loader.exec_module(campaign)

SCHEMA_VERSION = "c51-juq-performance-analysis-v1"
METHODS = ("whitney_compact", "exact_observation_gp", "spectral_low_rank")
COMPARATORS = ("exact_observation_gp", "spectral_low_rank")
REFINEMENTS = (8, 10)
DATA_COUNTS = (1, 100)
OBSERVATION_COUNT = 2048
PRACTICAL_IMPROVEMENT_RATIO = 0.75
MAXIMUM_OTHER_RESOURCE_REGRESSION_RATIO = 1.10


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class ResourcePoint:
    refinement: int
    replacement_data_count: int
    replicate: int
    method: str
    end_to_end_time_ns: float
    gross_peak_rss_bytes: float
    persistent_values: float


@dataclass(frozen=True)
class PairedComparison:
    refinement: int
    replacement_data_count: int
    comparator: str
    replicate_count: int
    time_ratio_median: float
    time_ratio_ci95_low: float
    time_ratio_ci95_high: float
    peak_rss_ratio_median: float
    peak_rss_ratio_ci95_low: float
    peak_rss_ratio_ci95_high: float
    storage_ratio_median: float
    practical_advantage: bool
    reason: str


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path} must contain a JSON object")
    return value


def measured(row: Mapping[str, str], name: str) -> float:
    if row.get(f"{name}_status") != "measured":
        reason = row.get(f"{name}_reason", "")
        raise AnalysisError(f"{row.get('method')} lacks measured {name}: {reason}")
    try:
        value = float(row[f"{name}_value"])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"{row.get('method')} has invalid {name}") from error
    if not math.isfinite(value) or value < 0.0:
        raise AnalysisError(f"{row.get('method')} has non-finite or negative {name}")
    return value


def headline_setup_time(row: Mapping[str, str]) -> float:
    if row["method"] == "exact_observation_gp":
        return measured(row, "exact_gp_end_to_end_setup_ns")
    return sum(
        measured(row, name)
        for name in (
            "feec_assembly_ns",
            "prior_build_ns",
            "observation_update_and_factor_ns",
        )
    )


def query_time(row: Mapping[str, str]) -> float:
    return sum(
        measured(row, name)
        for name in ("repeated_data_ns", "physical_qoi_ns", "variance_estimation_ns")
    )


def persistent_values(row: Mapping[str, str]) -> float:
    method = row["method"]
    if method == "fm_aug":
        return measured(row, "kkt_input_nnz")
    if method == "whitney_compact":
        return measured(row, "posterior_precision_nnz") + measured(
            row, "factor_storage_nnz"
        )
    if method == "exact_observation_gp":
        return measured(row, "exact_gp_persistent_storage_values")
    if method == "spectral_low_rank":
        return sum(
            measured(row, name)
            for name in (
                "spectral_basis_storage_values",
                "spectral_gp_storage_values",
                "spectral_prepared_storage_values",
            )
        )
    raise AnalysisError(f"unsupported method {method}")


def validate_row(
    row: Mapping[str, str],
    decisions: Mapping[tuple[int, int, int, float, str], Mapping[str, Any]],
    profile: Mapping[str, Any],
    evidence_tier: str,
    methods: Sequence[str],
    observation_count: int,
) -> ResourcePoint:
    if row.get("campaign_evidence_tier") != evidence_tier:
        raise AnalysisError(f"combined row is not in the {evidence_tier} tier")
    if row.get("available") != "true" or row.get(
        "common_physical_uq_workload_eligible"
    ) != "true":
        raise AnalysisError("combined row is unavailable or not common-workload eligible")
    if row.get("observation_design_id") != campaign.JUQ_SENSOR_DESIGN_ID:
        raise AnalysisError("combined row has the wrong observation design")
    if row.get("physical_qoi_design_id") != campaign.JUQ_PHYSICAL_QOI_DESIGN_ID:
        raise AnalysisError("combined row has the wrong physical-QoI design")
    try:
        refinement = int(row["refinement"])
        count = int(row["sensor_count"])
        data_count = int(row["repeated_data_count"])
        replicate = int(row["campaign_replicate"])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"combined row has invalid axes: {error}") from error
    method = row.get("method", "")
    if refinement not in REFINEMENTS or count != observation_count:
        raise AnalysisError("combined row is outside the frozen primary mesh/count axes")
    if data_count not in DATA_COUNTS or method not in methods:
        raise AnalysisError("combined row is outside the frozen primary data/method axes")
    key = campaign.rank_key(refinement, count, profile)
    decision = decisions.get(key)
    if decision is None:
        raise AnalysisError(f"combined row has no accuracy decision for {key}")
    if row.get("campaign_rank_decision_id") != decision["decision_id"]:
        raise AnalysisError("combined row is not bound to its frozen rank decision")
    if method == "spectral_low_rank":
        if int(measured(row, "spectral_rank")) != int(decision["rank"]):
            raise AnalysisError("spectral row rank differs from its frozen accuracy decision")
        if row.get("spectral_basis_source") != "partial_exact_requested_mixed_shell":
            raise AnalysisError("spectral timing did not use a fresh exact-request basis")
    setup_rss = measured(row, "factor_external_peak_rss_bytes")
    query_rss = measured(row, "query_external_peak_rss_bytes")
    return ResourcePoint(
        refinement=refinement,
        replacement_data_count=data_count,
        replicate=replicate,
        method=method,
        end_to_end_time_ns=headline_setup_time(row) + query_time(row),
        gross_peak_rss_bytes=max(setup_rss, query_rss),
        persistent_values=persistent_values(row),
    )


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_paired_bootstrap_median_ci(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 3:
        raise AnalysisError("paired 95% intervals require at least three measured replicates")
    bootstrapped = [
        median(sample)
        for sample in itertools.product(values, repeat=len(values))
    ]
    return percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def compare(
    points: Sequence[ResourcePoint],
    baseline: str = "whitney_compact",
    comparators: Sequence[str] = COMPARATORS,
    memory_noninferiority_required: bool = True,
) -> list[PairedComparison]:
    indexed = {
        (point.refinement, point.replacement_data_count, point.replicate, point.method): point
        for point in points
    }
    comparisons: list[PairedComparison] = []
    for refinement, data_count, comparator in itertools.product(
        REFINEMENTS, DATA_COUNTS, comparators
    ):
        replicates = sorted(
            {
                point.replicate
                for point in points
                if point.refinement == refinement
                and point.replacement_data_count == data_count
            }
        )
        if len(replicates) != 3:
            raise AnalysisError(
                f"n={refinement}, D={data_count} has {len(replicates)} replicates instead of 3"
            )
        time_ratios: list[float] = []
        rss_ratios: list[float] = []
        storage_ratios: list[float] = []
        for replicate in replicates:
            try:
                candidate = indexed[(refinement, data_count, replicate, baseline)]
                other = indexed[(refinement, data_count, replicate, comparator)]
            except KeyError as error:
                raise AnalysisError(f"missing paired method row: {error}") from error
            time_ratios.append(candidate.end_to_end_time_ns / other.end_to_end_time_ns)
            rss_ratios.append(candidate.gross_peak_rss_bytes / other.gross_peak_rss_bytes)
            storage_ratios.append(candidate.persistent_values / other.persistent_values)
        time_low, time_high = exact_paired_bootstrap_median_ci(time_ratios)
        rss_low, rss_high = exact_paired_bootstrap_median_ci(rss_ratios)
        if memory_noninferiority_required:
            time_win = (
                time_high <= PRACTICAL_IMPROVEMENT_RATIO
                and rss_high <= MAXIMUM_OTHER_RESOURCE_REGRESSION_RATIO
            )
            rss_win = (
                rss_high <= PRACTICAL_IMPROVEMENT_RATIO
                and time_high <= MAXIMUM_OTHER_RESOURCE_REGRESSION_RATIO
            )
            practical = time_win or rss_win
            reason = (
                "paired upper bounds establish >=25% time or RSS improvement and <=10% regression in the other"
                if practical
                else "frozen paired upper-bound practical-advantage rule not met"
            )
        else:
            practical = time_high <= PRACTICAL_IMPROVEMENT_RATIO
            reason = (
                "paired time-ratio upper bound establishes >=25% end-to-end time improvement; RSS is a reported tradeoff"
                if practical
                else "frozen paired upper-bound time-advantage rule not met"
            )
        comparisons.append(
            PairedComparison(
                refinement,
                data_count,
                comparator,
                len(replicates),
                median(time_ratios),
                time_low,
                time_high,
                median(rss_ratios),
                rss_low,
                rss_high,
                median(storage_ratios),
                practical,
                reason,
            )
        )
    return comparisons


def analyze(
    campaign_directory: pathlib.Path, rank_decisions: pathlib.Path
) -> dict[str, Any]:
    manifest = read_json(campaign_directory / "campaign-manifest.json")
    if manifest.get("schema_version") != campaign.SCHEMA_VERSION:
        raise AnalysisError("campaign manifest has the wrong schema")
    profile_name = str(manifest.get("profile"))
    if profile_name not in campaign.PROFILE_CONTRACTS:
        raise AnalysisError("campaign manifest has the wrong profile")
    exact_campaign = profile_name == campaign.EXACT_PROFILE_NAME
    evidence_tier = "claim_primary_exact" if exact_campaign else "claim_primary"
    methods = (
        ("fm_aug", "exact_observation_gp", "spectral_low_rank")
        if exact_campaign
        else METHODS
    )
    baseline = "fm_aug" if exact_campaign else "whitney_compact"
    comparators = COMPARATORS
    observation_count = 4096 if exact_campaign else OBSERVATION_COUNT
    if manifest.get("status") != "completed":
        raise AnalysisError(
            f"campaign is not complete without censors: {manifest.get('status')}"
        )
    if manifest.get("selected_evidence_tiers") != [evidence_tier]:
        raise AnalysisError(f"C51 adjudication requires exactly the {evidence_tier} tier")
    start_identity = manifest.get("source_identity_start")
    end_identity = manifest.get("source_identity_end")
    if not isinstance(start_identity, dict) or start_identity != end_identity:
        raise AnalysisError("campaign source identity is missing or changed during execution")
    try:
        campaign.source_identity(
            {
                "head": start_identity["root_sha"],
                "dirty": False,
                "submodules": {
                    "feec": {"head": start_identity["feec_sha"], "dirty": False},
                    "gmrf": {"head": start_identity["gmrf_sha"], "dirty": False},
                },
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError(f"invalid frozen source identity: {error}") from error
    plan = read_json(campaign_directory / "campaign-plan.json")
    if plan.get("schema_version") != campaign.SCHEMA_VERSION or plan.get("status") != "ready":
        raise AnalysisError("campaign plan is missing or was not ready")
    if plan.get("selected_evidence_tiers") != [evidence_tier]:
        raise AnalysisError(f"campaign plan does not freeze exactly {evidence_tier}")
    if plan.get("plan_sha256") != manifest.get("plan_sha256"):
        raise AnalysisError("campaign plan hash does not match the manifest")
    cells = plan.get("cells")
    if not isinstance(cells, list) or campaign.canonical_hash(
        [
            {key: value for key, value in cell.items() if key != "command"}
            for cell in cells
            if isinstance(cell, dict)
        ]
    ) != plan.get("plan_sha256") or any(not isinstance(cell, dict) for cell in cells):
        raise AnalysisError("campaign cell plan was modified after freezing")
    rank_hash = campaign.sha256_file(rank_decisions)
    if rank_hash != manifest.get("rank_decision_sha256") or rank_hash != plan.get(
        "rank_decision_sha256"
    ):
        raise AnalysisError("rank-decision artifact does not match the frozen campaign")
    profile_path = pathlib.Path(str(plan.get("profile_file", "")))
    if not profile_path.is_absolute():
        profile_path = campaign_directory / profile_path
    if not profile_path.is_file():
        raise AnalysisError("frozen performance profile is missing")
    profile_hash = campaign.sha256_file(profile_path)
    if profile_hash != manifest.get("profile_sha256") or profile_hash != plan.get(
        "profile_sha256"
    ):
        raise AnalysisError("performance profile does not match the frozen campaign")
    profile = campaign.read_json(profile_path)
    campaign.validate_profile(profile)
    if profile.get("profile") != profile_name:
        raise AnalysisError("campaign profile file does not match the manifest profile")
    decisions, decision_payload = campaign.load_rank_decisions(rank_decisions, profile)
    combined_path = campaign_directory / "combined-rows.csv"
    combined_record = manifest.get("combined_rows")
    if not isinstance(combined_record, dict):
        raise AnalysisError("campaign manifest lacks combined-row provenance")
    if (
        combined_record.get("path") != "combined-rows.csv"
        or combined_record.get("sha256") != campaign.sha256_file(combined_path)
        or combined_record.get("count") != 36
    ):
        raise AnalysisError("combined rows do not match their frozen manifest record")
    with combined_path.open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for prefix in ("cold_setup", "query_physical_uq"):
            path = pathlib.Path(row.get(f"{prefix}_runner_manifest", ""))
            if not path.is_absolute():
                path = campaign_directory / path
            expected_hash = row.get(f"{prefix}_runner_manifest_sha256")
            if not path.is_file() or campaign.sha256_file(path) != expected_hash:
                raise AnalysisError(f"{prefix} runner manifest is missing or modified")
            runner_manifest = read_json(path)
            try:
                identity = campaign.source_identity(runner_manifest["source"])
            except (KeyError, TypeError, ValueError) as error:
                raise AnalysisError(f"invalid {prefix} source provenance: {error}") from error
            if identity != start_identity:
                raise AnalysisError(f"{prefix} runner source differs from the campaign freeze")
    points = [
        validate_row(
            row,
            decisions,
            profile,
            evidence_tier,
            methods,
            observation_count,
        )
        for row in rows
    ]
    if len(points) != 36:
        raise AnalysisError(f"primary campaign requires 36 measured combined rows, found {len(points)}")
    if len(
        {
            (point.refinement, point.replacement_data_count, point.replicate, point.method)
            for point in points
        }
    ) != len(points):
        raise AnalysisError("primary campaign contains duplicate measured method cells")
    comparisons = compare(
        points,
        baseline,
        comparators,
        memory_noninferiority_required=not exact_campaign,
    )
    passed = all(comparison.practical_advantage for comparison in comparisons)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim": "C51",
        "decision": "PASS" if passed else "FAIL",
        "rule": {
            "accuracy_tolerance": 0.05,
            "required_refinements": list(REFINEMENTS),
            "required_replacement_data_counts": list(DATA_COUNTS),
            "candidate": baseline,
            "comparators": list(comparators),
            "practical_improvement_ratio": PRACTICAL_IMPROVEMENT_RATIO,
            "maximum_other_resource_regression_ratio": (
                MAXIMUM_OTHER_RESOURCE_REGRESSION_RATIO if not exact_campaign else None
            ),
            "memory_role": (
                "noninferiority gate" if not exact_campaign else "reported tradeoff"
            ),
            "paired_interval": "exact finite paired bootstrap median 95% interval",
        },
        "campaign_manifest": str(campaign_directory / "campaign-manifest.json"),
        "rank_decisions": str(rank_decisions),
        "rank_decision_status": decision_payload.get("status"),
        "point_count": len(points),
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "limitations": [
            "This is a bounded n=8,10 regime result, not an asymptotic complexity proof.",
            "Only three measured replicate pairs make the interval conservative and low-resolution.",
            "Gross process peak RSS, not analytic storage, controls the memory claim.",
        ],
    }


def write_markdown(path: pathlib.Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# C51 JUQ performance decision",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| n | D | comparator | time ratio median [95%] | RSS ratio median [95%] | pass |",
        "|---:|---:|---|---:|---:|:---:|",
    ]
    for item in result["comparisons"]:
        lines.append(
            "| {refinement} | {replacement_data_count} | {comparator} | "
            "{time_ratio_median:.3f} [{time_ratio_ci95_low:.3f}, {time_ratio_ci95_high:.3f}] | "
            "{peak_rss_ratio_median:.3f} [{peak_rss_ratio_ci95_low:.3f}, {peak_rss_ratio_ci95_high:.3f}] | {passed} |".format(
                **item, passed="yes" if item["practical_advantage"] else "no"
            )
        )
    lines.extend(["", *[f"- {item}" for item in result["limitations"]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_directory", type=pathlib.Path)
    parser.add_argument("--rank-decisions", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = analyze(arguments.campaign_directory, arguments.rank_decisions)
        output = arguments.output or arguments.campaign_directory / "c51-analysis.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_markdown(output.with_suffix(".md"), result)
        print(json.dumps({"decision": result["decision"], "output": str(output)}))
        return 0 if result["decision"] == "PASS" else 1
    except (AnalysisError, OSError, ValueError) as error:
        print(f"C51 performance analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
