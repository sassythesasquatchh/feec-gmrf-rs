#!/usr/bin/env python3
"""Analyze frozen C51 mass-weighted latent covariance-action rank thresholds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

THRESHOLDS = (0.10, 0.05, 0.02)
GROUP_COLUMNS = ("refinement", "state_dimension", "alpha", "design")
ERROR_COLUMN = "latent_mass_action_maximum_relative_error"
# Numerical reproducibility gate for the same rank extracted from two independently
# requested shared bases. This is not a scientific error threshold.
DUPLICATE_ERROR_ABSOLUTE_TOLERANCE = 5.0e-12


def load_rows(run_directories: list[Path]) -> list[dict[str, str]]:
    rows = []
    for directory in run_directories:
        path = directory / "rows.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            current = list(csv.DictReader(handle))
        required = {*GROUP_COLUMNS, "evidence_tier", "rank", "status", ERROR_COLUMN}
        missing = required.difference(current[0] if current else ())
        if missing:
            raise ValueError(f"{path} is empty or missing columns: {sorted(missing)}")
        rows.extend(current)
    return rows


def deduplicate_ranks(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], float]:
    by_rank: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_rank[int(row["rank"])].append(row)
    unique = []
    maximum_delta = 0.0
    for rank in sorted(by_rank):
        copies = by_rank[rank]
        reference = copies[0]
        for copy in copies[1:]:
            if copy["status"] != reference["status"]:
                raise ValueError(f"duplicate rank {rank} has inconsistent status")
            if reference["status"] == "measured":
                delta = abs(float(copy[ERROR_COLUMN]) - float(reference[ERROR_COLUMN]))
                maximum_delta = max(maximum_delta, delta)
                if delta > DUPLICATE_ERROR_ABSOLUTE_TOLERANCE:
                    raise ValueError(f"duplicate rank {rank} has inconsistent latent error")
        preferred = next(
            (row for row in copies if row["evidence_tier"] == "rank_discovery"),
            reference,
        )
        unique.append(preferred)
    return unique, maximum_delta


def analyze_group(rows: list[dict[str, str]], threshold: float) -> dict[str, str]:
    tiers = "+".join(sorted({row["evidence_tier"] for row in rows}))
    ordered, duplicate_delta = deduplicate_ranks(rows)
    maximum_rank = int(ordered[-1]["rank"])
    candidate = None
    for index, row in enumerate(ordered):
        suffix = ordered[index:]
        if all(
            later["status"] == "measured"
            and float(later[ERROR_COLUMN]) <= threshold
            for later in suffix
        ):
            candidate = row
            break
    result = {
        **{column: ordered[0][column] for column in GROUP_COLUMNS},
        "evidence_span": tiers,
        "maximum_duplicate_rank_absolute_error_delta": f"{duplicate_delta:.17e}",
        "duplicate_reproducibility_absolute_tolerance": (
            f"{DUPLICATE_ERROR_ABSOLUTE_TOLERANCE:.17e}"
        ),
        "threshold": f"{threshold:.17e}",
        "maximum_tested_rank": str(maximum_rank),
        "locked_rank": "",
        "rank_fraction": "",
        "latent_error_at_locked_rank": "",
        "decision_source_tier": "",
        "status": "",
        "reason": "",
    }
    if candidate is not None:
        result["locked_rank"] = candidate["rank"]
        result["rank_fraction"] = candidate["rank_fraction"]
        result["latent_error_at_locked_rank"] = candidate[ERROR_COLUMN]
        result["decision_source_tier"] = candidate["evidence_tier"]
        result["status"] = "attained"
        result["reason"] = "latent error sustained through the maximum tested rank"
    elif any(row["status"] != "measured" for row in ordered):
        result["status"] = "computationally_unavailable"
        result["reason"] = "a censored rank prevents a sustained measured suffix"
    else:
        result["status"] = "required_rank_gt_max_tested"
        result["reason"] = f"no sustained latent pass through rank {maximum_rank}"
    return result


def analyze(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in GROUP_COLUMNS)].append(row)
    return [
        analyze_group(groups[key], threshold)
        for key in sorted(groups)
        for threshold in THRESHOLDS
    ]


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.run_directories[-1] / "latent_rank_threshold_summary.csv"
    summary = analyze(load_rows(arguments.run_directories))
    write_summary(output, summary)
    print(f"wrote {len(summary)} latent threshold decisions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
