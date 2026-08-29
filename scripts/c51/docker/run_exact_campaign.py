#!/usr/bin/env python3
"""Prepare native Linux dependencies and run the exact-law C51 campaign."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Sequence


REPOSITORY = pathlib.Path("/workspace")


def run(command: Sequence[str], *, environment: dict[str, str] | None = None) -> None:
    process = subprocess.run(
        list(command),
        cwd=REPOSITORY,
        env=environment,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"command failed with status {process.returncode}: {command}")


def slepc_root() -> pathlib.Path:
    matches = sorted(
        pathlib.Path("/usr/lib").glob("slepcdir/**/lib/slepc/conf/slepc_common")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one SLEPc configuration, found {matches}")
    return matches[0].parents[3]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-decisions", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not pathlib.Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        print("C51 Linux campaign requires cgroup v2", file=sys.stderr)
        return 2
    if not os.access("/sys/fs/cgroup", os.W_OK):
        print(
            "C51 Linux campaign requires a delegated/privileged writable cgroup v2 mount",
            file=sys.stderr,
        )
        return 2
    environment = os.environ.copy()
    environment["SLEPC_DIR"] = str(slepc_root())
    environment["PETSC_SOLVER_PATH"] = str(
        REPOSITORY / "feec" / "petsc-solver" / "ghiep.out"
    )
    try:
        for directory in (REPOSITORY, REPOSITORY / "feec", REPOSITORY / "gmrf-rs"):
            run(["git", "config", "--global", "--add", "safe.directory", str(directory)])
        run(["make", "-C", "feec/petsc-solver", "clean", "all"], environment=environment)
        run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "feg-case-studies",
                "--features",
                "experimental,external-solvers",
                "--example",
                "c51_benchmark",
            ],
            environment=environment,
        )
        command = [
            "python3",
            "scripts/c51/run_juq_performance.py",
            "--profile-file",
            "scripts/c51/juq-exact-performance-v1.json",
            "--rank-decisions",
            str(args.rank_decisions),
            "--evidence-tier",
            "claim_primary_exact",
            "--output",
            str(args.output),
        ]
        if args.dry_run:
            command.append("--dry-run")
        run(command, environment=environment)
    except RuntimeError as error:
        print(f"C51 Linux campaign failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
