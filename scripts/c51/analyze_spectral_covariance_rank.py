#!/usr/bin/env python3
"""Apply the frozen sustained-selection/held-out C51 rank rule."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

THRESHOLDS = (0.10, 0.05, 0.02)
GROUP_COLUMNS = ("refinement", "alpha", "design")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        *GROUP_COLUMNS,
        "evidence_tier",
        "rank",
        "status",
        "selection_e_primary",
        "heldout_e_primary",
    }
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"rows CSV is empty or missing columns: {sorted(missing)}")
    return rows


def analyze_group(rows: list[dict[str, str]], threshold: float) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    maximum_rank = int(ordered[-1]["rank"])
    evidence_tiers = {row.get("evidence_tier", "rank_discovery") for row in ordered}
    if len(evidence_tiers) != 1:
        raise ValueError("a spectral rank curve cannot mix evidence tiers")
    candidate = None
    for index, row in enumerate(ordered):
        suffix = ordered[index:]
        if row["status"] != "measured" or any(
            later["status"] != "measured" for later in suffix
        ):
            continue
        if all(float(later["selection_e_primary"]) <= threshold for later in suffix):
            candidate = row
            break

    result = {
        **{column: ordered[0][column] for column in GROUP_COLUMNS},
        "evidence_tier": evidence_tiers.pop(),
        "threshold": f"{threshold:.17e}",
        "maximum_tested_rank": str(maximum_rank),
        "locked_rank": "",
        "selection_error_at_locked_rank": "",
        "heldout_error_at_locked_rank": "",
        "status": "",
        "reason": "",
    }
    if candidate is not None:
        heldout = float(candidate["heldout_e_primary"])
        result["locked_rank"] = candidate["rank"]
        result["selection_error_at_locked_rank"] = candidate["selection_e_primary"]
        result["heldout_error_at_locked_rank"] = candidate["heldout_e_primary"]
        if heldout <= threshold:
            result["status"] = "attained"
            result["reason"] = "selection sustained through maximum tested rank and held-out passed"
        else:
            result["status"] = "heldout_failed"
            result["reason"] = "held-out failed at the selection-locked rank; later ranks are not retuned"
        return result

    if any(row["status"] != "measured" for row in ordered):
        result["status"] = "computationally_unavailable"
        result["reason"] = (
            "a censored rank prevents a sustained measured prefix; this is not evidence that "
            "the required rank exceeds the maximum"
        )
    elif maximum_rank == 2_048:
        result["status"] = "required_rank_gt_2048"
        result["reason"] = "all ranks through 2048 were measured but no sustained selection pass exists"
    else:
        result["status"] = "required_rank_gt_max_tested"
        result["reason"] = (
            f"all ranks through the maximum tested rank {maximum_rank} were measured "
            "but no sustained selection pass exists"
        )
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
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    input_path = arguments.run_directory / "rows.csv"
    output_path = arguments.output or arguments.run_directory / "rank_threshold_summary.csv"
    summary = analyze(load_rows(input_path))
    write_summary(output_path, summary)
    print(f"wrote {len(summary)} threshold decisions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
