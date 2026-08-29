#!/usr/bin/env python3
"""Combine independently generated C51 rank-decision files without weakening them."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import uuid
from typing import Any, Sequence


SCHEMA_VERSION = "c51-performance-rank-decisions-v2"
PROFILE = "juq-performance-v1"


def read_payload(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read rank decisions from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} has the wrong rank-decision schema")
    if payload.get("profile") != PROFILE:
        raise ValueError(f"{path} does not target {PROFILE}")
    if not isinstance(payload.get("decisions"), list):
        raise ValueError(f"{path} has no decisions list")
    return payload


def decision_key(decision: dict[str, Any]) -> tuple[int, int, int, float, str]:
    try:
        return (
            int(decision["refinement"]),
            int(decision["scalar_observation_count"]),
            int(decision["alpha"]),
            float(decision["relative_tolerance"]),
            str(decision["observation_design_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid rank-decision key: {error}") from error


def combine(paths: Sequence[pathlib.Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one rank-decision input is required")
    indexed: dict[tuple[int, int, int, float, str], dict[str, Any]] = {}
    perf0_hashes: set[str] = set()
    for path in paths:
        payload = read_payload(path)
        for raw in payload["decisions"]:
            if not isinstance(raw, dict):
                raise ValueError(f"{path} contains a non-object decision")
            key = decision_key(raw)
            if key in indexed:
                raise ValueError(f"duplicate rank-decision key {key}")
            try:
                perf0_hashes.add(
                    str(raw["exact_observation_gp"]["fixture_artifact_sha256"])
                )
            except (KeyError, TypeError) as error:
                raise ValueError(f"{path} lacks PERF-0 provenance: {error}") from error
            indexed[key] = raw
    if len(perf0_hashes) != 1:
        raise ValueError("rank decisions do not share one immutable PERF-0 artifact")
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "status": "frozen_heldout_validated",
        "source_files": [str(path) for path in paths],
        "decisions": [indexed[key] for key in sorted(indexed)],
    }


def write_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("inputs", type=pathlib.Path, nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        write_atomic(arguments.output, combine(arguments.inputs))
    except ValueError as error:
        print(f"C51 rank-decision combination failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
