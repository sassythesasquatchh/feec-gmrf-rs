#!/usr/bin/env python3
"""Build and safely run one C51 spectral covariance-rank discovery attempt.

The release build is completed before scientific resource measurement begins.
The scientific binary is then executed directly in a fresh process group while
this runner samples the resident memory of that process tree, the attempt
directory size, and elapsed wall time.  Every started attempt has an atomic
``attempt_manifest.json`` even when the build, launch, monitor, or child fails.

Only Python's standard library is used.  Process inspection reads ``/proc`` on
Linux and invokes the platform ``ps`` utility on macOS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "c51-spectral-rank-attempt-v2"
PRIMARY_PROFILE = "publication-discovery-v1"
RANK_EXTENSION_PROFILES = {
    "rank-extension-v1-n8": {
        "refinement": 8,
        "state_dimension": 3032,
        "ranks": (2048, 2304, 2560, 2816, 3000, 3032),
        "evidence_tier": "rank_extension",
    },
    "rank-extension-v1-n10": {
        "refinement": 10,
        "state_dimension": 6130,
        "ranks": (2048, 2560, 3072, 3584, 4096),
        "evidence_tier": "rank_extension",
    },
    "high-observation-m4096-rank-extension-v1-n10-r4608": {
        "refinement": 10,
        "state_dimension": 6130,
        "ranks": (4096, 4608),
        "evidence_tier": "high_observation_confirmatory_rank_extension",
    },
}
HIGH_OBSERVATION_PROFILES = {
    "high-observation-primary-m2048-v1-n8": {
        "refinement": 8,
        "state_dimension": 3032,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2304, 2560, 2816, 3000, 3032),
        "evidence_tier": "high_observation_primary_accuracy",
    },
    "high-observation-primary-m2048-v1-n10": {
        "refinement": 10,
        "state_dimension": 6130,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6000, 6130),
        "evidence_tier": "high_observation_primary_accuracy",
    },
    "high-observation-confirmatory-m4096-v1-n8": {
        "refinement": 8,
        "state_dimension": 3032,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2304, 2560, 2816, 3000, 3032),
        "evidence_tier": "high_observation_confirmatory_accuracy",
    },
    "high-observation-confirmatory-m4096-v1-n10": {
        "refinement": 10,
        "state_dimension": 6130,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6000, 6130),
        "evidence_tier": "high_observation_confirmatory_accuracy",
    },
}
REPLAY_PROFILES = {
    "high-observation-m2048-replay-v1-n8-r3032": {
        "refinement": 8,
        "state_dimension": 3032,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2304, 2560, 2816, 3000, 3032),
        "evidence_tier": "high_observation_replay_accuracy_only",
    },
    "high-observation-m2048-replay-v1-n10-r4096": {
        "refinement": 10,
        "state_dimension": 6130,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2560, 3072, 3584, 4096),
        "evidence_tier": "high_observation_replay_accuracy_only",
    },
    "high-observation-m4096-replay-v1-n8-r3032": {
        "refinement": 8,
        "state_dimension": 3032,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2304, 2560, 2816, 3000, 3032),
        "evidence_tier": "high_observation_replay_accuracy_only",
    },
    "high-observation-m4096-replay-v1-n10-r4096": {
        "refinement": 10,
        "state_dimension": 6130,
        "ranks": (512, 768, 1024, 1280, 1536, 1792, 2000, 2048, 2560, 3072, 3584, 4096),
        "evidence_tier": "high_observation_replay_accuracy_only",
    },
}
RANK_EXTENSION_RATIONALE = (
    "post-primary extension frozen after publication-discovery-v1 left corrected "
    "alpha=2 5%/2% decisions unresolved; shared-basis rows are timing ineligible"
)
EXAMPLE = "c51_spectral_covariance_rank"
PACKAGE = "feg-case-studies"
FEATURES = "experimental,external-solvers"
MANIFEST_NAME = "attempt_manifest.json"
DEFAULT_WALL_TIME_SECONDS = 6.0 * 60.0 * 60.0
DEFAULT_RSS_CAP_BYTES = 16 * 1024**3
DEFAULT_DISK_CAP_BYTES = 10 * 1024**3
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "FAER_NUM_THREADS",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def command_text(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finalize_manifest(
    path: pathlib.Path, manifest: dict[str, Any], output: pathlib.Path
) -> None:
    """Write a final manifest whose disk maximum includes the final archive."""

    atomic_write_json(path, manifest)
    final_bytes = directory_bytes(output)
    recorded = int(manifest.get("maximum_output_directory_bytes", 0) or 0)
    if final_bytes > recorded:
        manifest["maximum_output_directory_bytes"] = final_bytes
        atomic_write_json(path, manifest)


def run_capture(command: Sequence[str], cwd: pathlib.Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        return {
            "status": "unavailable",
            "command": command_text(command),
            "reason": str(error),
        }
    return {
        "status": "available" if completed.returncode == 0 else "unavailable",
        "command": command_text(command),
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def successful_stdout(result: Mapping[str, Any]) -> str | None:
    if result.get("status") == "available":
        value = str(result.get("stdout", "")).strip()
        return value or None
    return None


def repository_state(path: pathlib.Path) -> dict[str, Any]:
    head = run_capture(["git", "rev-parse", "HEAD"], path)
    status = run_capture(["git", "status", "--porcelain"], path)
    branch = run_capture(["git", "branch", "--show-current"], path)
    return {
        "path": str(path),
        "head": successful_stdout(head),
        "branch": successful_stdout(branch),
        "dirty": (
            bool(str(status.get("stdout", "")))
            if status.get("status") == "available"
            else None
        ),
        "queries": {"head": head, "branch": branch, "status": status},
    }


def source_metadata(repository: pathlib.Path) -> dict[str, Any]:
    submodule_status = run_capture(
        ["git", "submodule", "status", "--recursive"], repository
    )
    return {
        "repository": repository_state(repository),
        "submodule_status": submodule_status,
        "submodules": {
            name: (
                repository_state(repository / relative)
                if (repository / relative).is_dir()
                else {
                    "path": str(repository / relative),
                    "head": None,
                    "dirty": None,
                    "status": "unavailable",
                    "reason": "submodule directory is missing",
                }
            )
            for name, relative in (("feec", "feec"), ("gmrf", "gmrf-rs"))
        },
    }


def platform_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
    }


def tool_metadata(repository: pathlib.Path, cargo: str) -> dict[str, Any]:
    return {
        "python": run_capture([sys.executable, "--version"], repository),
        "cargo": run_capture([cargo, "--version", "--verbose"], repository),
        "rustc": run_capture(["rustc", "--version", "--verbose"], repository),
        "git": run_capture(["git", "--version"], repository),
        "ps": run_capture(["ps", "--version"], repository),
    }


def thread_environment(count: int) -> dict[str, str]:
    if count < 1:
        raise ValueError("thread count must be positive")
    return {name: str(count) for name in THREAD_ENV_NAMES}


def target_directory(repository: pathlib.Path) -> pathlib.Path:
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured is None:
        return repository / "target"
    path = pathlib.Path(configured)
    return path if path.is_absolute() else repository / path


def release_binary(repository: pathlib.Path) -> pathlib.Path:
    suffix = ".exe" if os.name == "nt" else ""
    return target_directory(repository) / "release" / "examples" / f"{EXAMPLE}{suffix}"


def build_command(cargo: str) -> list[str]:
    return [
        cargo,
        "build",
        "--locked",
        "--release",
        "-p",
        PACKAGE,
        "--features",
        FEATURES,
        "--example",
        EXAMPLE,
    ]


def scientific_command(
    binary: pathlib.Path,
    profile: str,
    refinement: int,
    maximum_rank: int | None,
    study_output: pathlib.Path,
    perf0_archive: pathlib.Path | None = None,
    basis_workspace: pathlib.Path | None = None,
) -> list[str]:
    command = [
        str(binary),
        "--profile",
        profile,
        "--refinement",
        str(refinement),
        "--output",
        str(study_output),
    ]
    if maximum_rank is not None:
        command[5:5] = ["--maximum-rank", str(maximum_rank)]
    if perf0_archive is not None:
        command.extend(
            [
                "--perf0-archive",
                str(perf0_archive),
                "--rank-decision-output",
                str(study_output / "rank-decision.json"),
            ]
        )
    if basis_workspace is not None:
        command.extend(["--basis-workspace", str(basis_workspace)])
    return command


def resolve_profile(
    profile: str, refinement: int, maximum_rank: int | None
) -> dict[str, Any]:
    if profile == PRIMARY_PROFILE:
        if refinement not in (8, 10):
            raise ValueError("publication-discovery-v1 requires refinement 8 or 10")
        if maximum_rank is None:
            raise ValueError("publication-discovery-v1 requires --maximum-rank")
        return {
            "profile": profile,
            "refinement": refinement,
            "maximum_rank": maximum_rank,
            "rank_grid": None,
            "expected_state_dimension": None,
            "evidence_tier": "rank_discovery",
            "post_primary": False,
            "post_primary_rationale": None,
            "rank_decision_required": False,
        }
    if profile in RANK_EXTENSION_PROFILES:
        frozen = RANK_EXTENSION_PROFILES[profile]
        if refinement != frozen["refinement"]:
            raise ValueError(
                f"{profile} requires refinement {frozen['refinement']}, received {refinement}"
            )
        if maximum_rank is not None:
            raise ValueError(f"{profile} has an immutable grid and rejects --maximum-rank")
        return {
            "profile": profile,
            "refinement": refinement,
            "maximum_rank": frozen["ranks"][-1],
            "rank_grid": list(frozen["ranks"]),
            "expected_state_dimension": frozen["state_dimension"],
            "evidence_tier": frozen["evidence_tier"],
            "post_primary": True,
            "post_primary_rationale": RANK_EXTENSION_RATIONALE,
            "rank_decision_required": False,
        }
    if profile in HIGH_OBSERVATION_PROFILES:
        frozen = HIGH_OBSERVATION_PROFILES[profile]
        if refinement != frozen["refinement"]:
            raise ValueError(
                f"{profile} requires refinement {frozen['refinement']}, received {refinement}"
            )
        if maximum_rank is not None:
            raise ValueError(f"{profile} has an immutable grid and rejects --maximum-rank")
        return {
            "profile": profile,
            "refinement": refinement,
            "maximum_rank": frozen["ranks"][-1],
            "rank_grid": list(frozen["ranks"]),
            "expected_state_dimension": frozen["state_dimension"],
            "evidence_tier": frozen["evidence_tier"],
            "post_primary": False,
            "post_primary_rationale": "pre-performance held-out rank qualification",
            "rank_decision_required": True,
        }
    if profile in REPLAY_PROFILES:
        frozen = REPLAY_PROFILES[profile]
        if refinement != frozen["refinement"]:
            raise ValueError(
                f"{profile} requires refinement {frozen['refinement']}, received {refinement}"
            )
        if maximum_rank is not None:
            raise ValueError(f"{profile} has an immutable grid and rejects --maximum-rank")
        return {
            "profile": profile,
            "refinement": refinement,
            "maximum_rank": frozen["ranks"][-1],
            "rank_grid": list(frozen["ranks"]),
            "expected_state_dimension": frozen["state_dimension"],
            "evidence_tier": frozen["evidence_tier"],
            "post_primary": False,
            "post_primary_rationale": "accuracy-only replay selecting the next proof-grade rank solve",
            "rank_decision_required": False,
        }
    raise ValueError(f"unknown C51 spectral covariance profile {profile}")


def directory_bytes(root: pathlib.Path) -> int:
    """Return allocated artifact file sizes without following symlinks."""

    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(pathlib.Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
    return total


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    ppid: int
    process_group: int
    rss_bytes: int


def linux_process_snapshot() -> dict[int, ProcessSample]:
    snapshot: dict[int, ProcessSample] = {}
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("/proc is not mounted")
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            stat_text = (directory / "stat").read_text(encoding="ascii")
            closing = stat_text.rfind(")")
            if closing < 0:
                continue
            fields = stat_text[closing + 2 :].split()
            # fields starts at proc(5) stat field 3: state, ppid, pgrp.
            ppid = int(fields[1])
            process_group = int(fields[2])
            rss_kib = None
            for line in (directory / "status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                    break
            if rss_kib is None:
                continue
            snapshot[pid] = ProcessSample(pid, ppid, process_group, rss_kib * 1024)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return snapshot


def parse_ps_snapshot(text: str) -> dict[int, ProcessSample]:
    snapshot: dict[int, ProcessSample] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            pid, ppid, process_group, rss_kib = map(int, fields)
        except ValueError:
            continue
        snapshot[pid] = ProcessSample(pid, ppid, process_group, rss_kib * 1024)
    return snapshot


def macos_process_snapshot() -> dict[int, ProcessSample]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,rss="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ps process sampling failed: {completed.stderr.strip()}")
    return parse_ps_snapshot(completed.stdout)


def process_snapshot() -> dict[int, ProcessSample]:
    system = platform.system()
    if system == "Linux":
        return linux_process_snapshot()
    if system == "Darwin":
        return macos_process_snapshot()
    raise RuntimeError(f"process-tree RSS sampling is unsupported on {system}")


def process_tree_pids(
    root_pid: int, snapshot: Mapping[int, ProcessSample]
) -> set[int]:
    """Select descendants plus surviving members of the fresh process group."""

    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, sample in snapshot.items():
            if pid not in selected and sample.ppid in selected:
                selected.add(pid)
                changed = True
    # A descendant can outlive its immediate parent and be reparented between
    # samples.  Because the child starts a new session, its process group is an
    # unambiguous continuation of this attempt.
    selected.update(
        pid for pid, sample in snapshot.items() if sample.process_group == root_pid
    )
    return selected


def sampled_process_tree_rss(root_pid: int) -> tuple[int, list[int]]:
    snapshot = process_snapshot()
    pids = sorted(process_tree_pids(root_pid, snapshot))
    return sum(snapshot[pid].rss_bytes for pid in pids if pid in snapshot), pids


def normalized_exit(return_code: int | None) -> tuple[int | None, int | None, str | None]:
    if return_code is None:
        return None, None, None
    if return_code >= 0:
        return return_code, None, None
    number = -return_code
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = None
    return 128 + number, number, name


def terminate_process_group(
    process: subprocess.Popen[bytes], grace_seconds: float = 3.0
) -> dict[str, Any]:
    record: dict[str, Any] = {"requested_at_utc": utc_now()}
    if process.poll() is not None:
        record["action"] = "already_exited"
        return record
    try:
        os.killpg(process.pid, signal.SIGTERM)
        record["action"] = "SIGTERM_process_group"
    except ProcessLookupError:
        record["action"] = "process_group_already_gone"
        return record
    try:
        process.wait(timeout=grace_seconds)
        record["escalated"] = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            record["escalated"] = True
            record["escalation"] = "SIGKILL_process_group"
        except ProcessLookupError:
            record["escalated"] = False
    return record


@dataclass
class MonitoredResult:
    outcome: str
    reason: str
    raw_return_code: int | None
    shell_exit_status: int | None
    terminating_signal: int | None
    terminating_signal_name: str | None
    started_at_utc: str
    ended_at_utc: str
    duration_seconds: float
    peak_sampled_process_tree_rss_bytes: int
    maximum_output_directory_bytes: int
    samples: int
    last_sampled_pids: list[int]
    termination: dict[str, Any] | None
    launch_error: str | None
    monitor_error: str | None


def run_monitored_process(
    command: Sequence[str],
    cwd: pathlib.Path,
    attempt_output: pathlib.Path,
    environment: Mapping[str, str],
    wall_time_limit_seconds: float,
    rss_cap_bytes: int,
    disk_cap_bytes: int,
    sample_interval_seconds: float,
) -> MonitoredResult:
    stdout_path = attempt_output / "scientific.stdout.log"
    stderr_path = attempt_output / "scientific.stderr.log"
    started_at = utc_now()
    started = time.monotonic()
    maximum_disk = directory_bytes(attempt_output)
    peak_rss = 0
    sample_count = 0
    last_pids: list[int] = []
    termination: dict[str, Any] | None = None
    launch_error = None
    monitor_error = None
    process: subprocess.Popen[bytes] | None = None
    outcome = "launch_failed"
    reason = "scientific process did not start"

    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [str(part) for part in command],
                cwd=cwd,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while True:
                elapsed = time.monotonic() - started
                try:
                    maximum_disk = max(maximum_disk, directory_bytes(attempt_output))
                    rss, pids = sampled_process_tree_rss(process.pid)
                except (OSError, RuntimeError) as error:
                    monitor_error = str(error)
                    outcome = "monitor_failed"
                    reason = f"cannot enforce resource caps: {error}"
                    termination = terminate_process_group(process)
                    break
                peak_rss = max(peak_rss, rss)
                last_pids = pids
                sample_count += 1

                if elapsed > wall_time_limit_seconds:
                    outcome = "wall_time_cap_exceeded"
                    reason = (
                        f"wall time {elapsed:.6f}s exceeded cap "
                        f"{wall_time_limit_seconds:.6f}s"
                    )
                    termination = terminate_process_group(process)
                    break
                if rss > rss_cap_bytes:
                    outcome = "rss_cap_exceeded"
                    reason = (
                        f"sampled process-tree RSS {rss} bytes exceeded cap "
                        f"{rss_cap_bytes} bytes"
                    )
                    termination = terminate_process_group(process)
                    break
                if maximum_disk > disk_cap_bytes:
                    outcome = "disk_cap_exceeded"
                    reason = (
                        f"attempt output size {maximum_disk} bytes exceeded cap "
                        f"{disk_cap_bytes} bytes"
                    )
                    termination = terminate_process_group(process)
                    break

                return_code = process.poll()
                if return_code is not None:
                    if return_code == 0:
                        outcome = "completed"
                        reason = "scientific process exited successfully"
                    else:
                        outcome = "child_failed"
                        reason = f"scientific process exited with raw status {return_code}"
                    break
                time.sleep(sample_interval_seconds)
    except OSError as error:
        launch_error = str(error)
        outcome = "launch_failed"
        reason = f"scientific process launch failed: {error}"
    except KeyboardInterrupt:
        outcome = "interrupted"
        reason = "runner received KeyboardInterrupt"
        if process is not None:
            termination = terminate_process_group(process)

    if process is not None and process.poll() is None:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            additional = terminate_process_group(process, grace_seconds=0.5)
            termination = termination or additional
            process.wait()

    maximum_disk = max(maximum_disk, directory_bytes(attempt_output))
    raw_return_code = process.returncode if process is not None else None
    shell_status, signal_number, signal_name = normalized_exit(raw_return_code)
    duration = time.monotonic() - started
    return MonitoredResult(
        outcome=outcome,
        reason=reason,
        raw_return_code=raw_return_code,
        shell_exit_status=shell_status,
        terminating_signal=signal_number,
        terminating_signal_name=signal_name,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_seconds=duration,
        peak_sampled_process_tree_rss_bytes=peak_rss,
        maximum_output_directory_bytes=maximum_disk,
        samples=sample_count,
        last_sampled_pids=last_pids,
        termination=termination,
        launch_error=launch_error,
        monitor_error=monitor_error,
    )


def build_release(
    command: Sequence[str],
    repository: pathlib.Path,
    environment: Mapping[str, str],
    attempt_output: pathlib.Path,
) -> dict[str, Any]:
    stdout_path = attempt_output / "build.stdout.log"
    stderr_path = attempt_output / "build.stderr.log"
    started_at = utc_now()
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                [str(part) for part in command],
                cwd=repository,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        raw_return_code = completed.returncode
        launch_error = None
    except OSError as error:
        raw_return_code = 127
        launch_error = str(error)
        stdout_path.touch()
        stderr_path.write_text(f"build launch failed: {error}\n", encoding="utf-8")
    shell_status, signal_number, signal_name = normalized_exit(raw_return_code)
    return {
        "command": [str(part) for part in command],
        "command_text": command_text(command),
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "raw_return_code": raw_return_code,
        "shell_exit_status": shell_status,
        "terminating_signal": signal_number,
        "terminating_signal_name": signal_name,
        "launch_error": launch_error,
        "outside_scientific_measurement": True,
    }


def validate_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def dry_run_plan(
    args: argparse.Namespace,
    repository: pathlib.Path,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or resolve_profile(args.profile, args.refinement, args.maximum_rank)
    binary = release_binary(repository)
    output = args.output.resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run",
        "repository": str(repository),
        "attempt_output": str(output),
        "study_output": str(output / "study"),
        "profile": profile["profile"],
        "evidence_tier": profile["evidence_tier"],
        "post_primary": profile["post_primary"],
        "post_primary_rationale": profile["post_primary_rationale"],
        "refinement": args.refinement,
        "maximum_rank": profile["maximum_rank"],
        "rank_grid": profile["rank_grid"],
        "expected_state_dimension": profile["expected_state_dimension"],
        "rank_decision_required": profile["rank_decision_required"],
        "perf0_archive": str(args.perf0_archive.resolve()) if args.perf0_archive else None,
        "basis_workspace": str(args.basis_workspace.resolve()) if args.basis_workspace else None,
        "build_command": build_command(args.cargo),
        "anticipated_binary": str(binary),
        "scientific_command": scientific_command(
            binary,
            args.profile,
            args.refinement,
            args.maximum_rank,
            output / "study",
            args.perf0_archive.resolve() if args.perf0_archive else None,
            args.basis_workspace.resolve() if args.basis_workspace else None,
        ),
        "caps": {
            "wall_time_seconds": args.wall_time_limit_seconds,
            "sampled_process_tree_rss_bytes": args.rss_cap_bytes,
            "output_directory_bytes": args.disk_cap_bytes,
            "sample_interval_seconds": args.sample_interval_seconds,
        },
        "thread_environment": thread_environment(args.threads),
    }


def reserve_attempt_directory(output: pathlib.Path) -> None:
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite existing attempt directory: {output}") from error


def initial_manifest(
    args: argparse.Namespace,
    repository: pathlib.Path,
    output: pathlib.Path,
    binary: pathlib.Path,
    environment_overrides: Mapping[str, str],
    source: Mapping[str, Any],
    platform_info: Mapping[str, Any],
    tools: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = scientific_command(
        binary,
        args.profile,
        args.refinement,
        args.maximum_rank,
        output / "study",
        args.perf0_archive.resolve() if args.perf0_archive else None,
        args.basis_workspace.resolve() if args.basis_workspace else None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": "running",
        "reason": "attempt initialized",
        "profile": profile["profile"],
        "evidence_tier": profile["evidence_tier"],
        "post_primary": profile["post_primary"],
        "post_primary_rationale": profile["post_primary_rationale"],
        "method_performance_timing_eligible": False,
        "shared_maximum_basis": True,
        "refinement": args.refinement,
        "maximum_rank": profile["maximum_rank"],
        "rank_grid": profile["rank_grid"],
        "expected_state_dimension": profile["expected_state_dimension"],
        "rank_decision_required": profile["rank_decision_required"],
        "perf0_archive": str(args.perf0_archive.resolve()) if args.perf0_archive else None,
        "basis_workspace": str(args.basis_workspace.resolve()) if args.basis_workspace else None,
        "requested_maximum_rank": args.maximum_rank,
        "attempt_output": str(output),
        "study_output": str(output / "study"),
        "command": scientific,
        "command_text": command_text(scientific),
        "caps": {
            "wall_time_seconds": args.wall_time_limit_seconds,
            "sampled_process_tree_rss_bytes": args.rss_cap_bytes,
            "output_directory_bytes": args.disk_cap_bytes,
            "sample_interval_seconds": args.sample_interval_seconds,
            "rss_measurement": "sampled sum over the fresh process tree/process group",
        },
        "timestamps": {"attempt_started_at_utc": utc_now(), "attempt_ended_at_utc": None},
        "duration_seconds": None,
        "peak_sampled_process_tree_rss_bytes": 0,
        "maximum_output_directory_bytes": 0,
        "raw_return_code": None,
        "shell_exit_status": None,
        "terminating_signal": None,
        "terminating_signal_name": None,
        "source": dict(source),
        "platform": dict(platform_info),
        "tools": dict(tools),
        "environment_overrides": dict(environment_overrides),
        "runner": {
            "path": str(pathlib.Path(__file__).resolve()),
            "python_standard_library_only": True,
        },
    }


def run_attempt(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    if not (repository / "Cargo.toml").is_file():
        print(f"repository has no Cargo.toml: {repository}", file=sys.stderr)
        return 2
    try:
        validate_positive("refinement", args.refinement)
        profile = resolve_profile(args.profile, args.refinement, args.maximum_rank)
        if profile["rank_decision_required"]:
            if args.perf0_archive is None:
                raise ValueError(f"{args.profile} requires --perf0-archive")
            if not args.perf0_archive.resolve().is_dir():
                raise ValueError("--perf0-archive must name a completed archive directory")
        elif args.perf0_archive is not None:
            raise ValueError("--perf0-archive is accepted only by focused high-observation profiles")
        if args.profile in REPLAY_PROFILES:
            if args.basis_workspace is None:
                raise ValueError(f"{args.profile} requires --basis-workspace")
            workspace = args.basis_workspace.resolve()
            if not (workspace / "out" / "eigenvals.bin").is_file() or not (
                workspace / "out" / "eigenvecs_u.bin"
            ).is_file():
                raise ValueError("--basis-workspace lacks PETSc eigenvalue/vector outputs")
        elif args.basis_workspace is not None:
            raise ValueError("--basis-workspace is accepted only by replay profiles")
        validate_positive("maximum rank", profile["maximum_rank"])
        validate_positive("wall-time limit", args.wall_time_limit_seconds)
        validate_positive("RSS cap", args.rss_cap_bytes)
        validate_positive("disk cap", args.disk_cap_bytes)
        validate_positive("sample interval", args.sample_interval_seconds)
        overrides = thread_environment(args.threads)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if platform.system() not in {"Darwin", "Linux"}:
        print("resource-capped C51 runs are supported only on macOS and Linux", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, repository, profile), indent=2, sort_keys=True))
        return 0

    # Freeze source and tool state before creating a possibly unignored output
    # directory inside the checkout.  The attempt itself begins only after this
    # read-only preflight and successful exclusive directory reservation.
    try:
        captured_source = source_metadata(repository)
        captured_platform = platform_metadata()
        captured_tools = tool_metadata(repository, args.cargo)
    except KeyboardInterrupt:
        print("interrupted while collecting pre-attempt metadata", file=sys.stderr)
        return 130

    output = args.output.resolve()
    try:
        reserve_attempt_directory(output)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    manifest_path = output / MANIFEST_NAME
    binary = release_binary(repository)
    environment = os.environ.copy()
    environment.update(overrides)
    attempt_started = time.monotonic()
    try:
        manifest = initial_manifest(
            args,
            repository,
            output,
            binary,
            overrides,
            captured_source,
            captured_platform,
            captured_tools,
            profile,
        )
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "outcome": "interrupted" if interrupted else "initialization_failed",
            "reason": f"attempt metadata initialization failed: {error}",
            "profile": profile["profile"],
            "evidence_tier": profile["evidence_tier"],
            "post_primary": profile["post_primary"],
            "post_primary_rationale": profile["post_primary_rationale"],
            "method_performance_timing_eligible": False,
            "shared_maximum_basis": True,
            "refinement": args.refinement,
            "maximum_rank": profile["maximum_rank"],
            "rank_grid": profile["rank_grid"],
            "expected_state_dimension": profile["expected_state_dimension"],
            "rank_decision_required": profile["rank_decision_required"],
            "perf0_archive": str(args.perf0_archive.resolve()) if args.perf0_archive else None,
            "basis_workspace": str(args.basis_workspace.resolve()) if args.basis_workspace else None,
            "requested_maximum_rank": args.maximum_rank,
            "attempt_output": str(output),
            "study_output": str(output / "study"),
            "command": scientific_command(
                binary,
                args.profile,
                args.refinement,
                args.maximum_rank,
                output / "study",
                args.perf0_archive.resolve() if args.perf0_archive else None,
                args.basis_workspace.resolve() if args.basis_workspace else None,
            ),
            "caps": {
                "wall_time_seconds": args.wall_time_limit_seconds,
                "sampled_process_tree_rss_bytes": args.rss_cap_bytes,
                "output_directory_bytes": args.disk_cap_bytes,
                "sample_interval_seconds": args.sample_interval_seconds,
            },
            "timestamps": {
                "attempt_started_at_utc": utc_now(),
                "attempt_ended_at_utc": utc_now(),
            },
            "duration_seconds": time.monotonic() - attempt_started,
            "peak_sampled_process_tree_rss_bytes": 0,
            "maximum_output_directory_bytes": 0,
            "raw_return_code": None,
            "shell_exit_status": 130 if interrupted else None,
            "terminating_signal": None,
            "terminating_signal_name": None,
        }
        finalize_manifest(manifest_path, manifest, output)
        return 130 if interrupted else 2
    atomic_write_json(manifest_path, manifest)

    try:
        build = build_release(
            build_command(args.cargo), repository, environment, output
        )
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        manifest.update(
            {
                "outcome": "interrupted" if interrupted else "build_monitor_failed",
                "reason": f"release build did not complete: {error}",
                "raw_return_code": None,
                "shell_exit_status": 130 if interrupted else None,
            }
        )
        manifest["timestamps"]["attempt_ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.monotonic() - attempt_started
        manifest["maximum_output_directory_bytes"] = directory_bytes(output)
        finalize_manifest(manifest_path, manifest, output)
        return 130 if interrupted else 2
    manifest["build"] = build
    if build["shell_exit_status"] != 0:
        manifest.update(
            {
                "outcome": "build_failed",
                "reason": "release build failed before scientific measurement",
                "raw_return_code": build["raw_return_code"],
                "shell_exit_status": build["shell_exit_status"],
                "terminating_signal": build["terminating_signal"],
                "terminating_signal_name": build["terminating_signal_name"],
            }
        )
        manifest["timestamps"]["attempt_ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.monotonic() - attempt_started
        manifest["maximum_output_directory_bytes"] = directory_bytes(output)
        finalize_manifest(manifest_path, manifest, output)
        return int(build["shell_exit_status"] or 1)

    if not binary.is_file():
        manifest.update(
            {
                "outcome": "binary_missing",
                "reason": f"release build succeeded but binary is missing: {binary}",
            }
        )
        manifest["timestamps"]["attempt_ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.monotonic() - attempt_started
        manifest["maximum_output_directory_bytes"] = directory_bytes(output)
        finalize_manifest(manifest_path, manifest, output)
        return 2

    initial_bytes = directory_bytes(output)
    if initial_bytes > args.disk_cap_bytes:
        manifest.update(
            {
                "outcome": "disk_cap_exceeded_before_launch",
                "reason": (
                    f"build artifacts/logs in attempt directory use {initial_bytes} bytes, "
                    f"above disk cap {args.disk_cap_bytes} bytes"
                ),
                "maximum_output_directory_bytes": initial_bytes,
            }
        )
        manifest["timestamps"]["attempt_ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.monotonic() - attempt_started
        finalize_manifest(manifest_path, manifest, output)
        return 2

    command = scientific_command(
        binary,
        args.profile,
        args.refinement,
        args.maximum_rank,
        output / "study",
        args.perf0_archive.resolve() if args.perf0_archive else None,
        args.basis_workspace.resolve() if args.basis_workspace else None,
    )
    result = run_monitored_process(
        command=command,
        cwd=repository,
        attempt_output=output,
        environment=environment,
        wall_time_limit_seconds=args.wall_time_limit_seconds,
        rss_cap_bytes=args.rss_cap_bytes,
        disk_cap_bytes=args.disk_cap_bytes,
        sample_interval_seconds=args.sample_interval_seconds,
    )
    manifest.update(
        {
            "outcome": result.outcome,
            "reason": result.reason,
            "raw_return_code": result.raw_return_code,
            "shell_exit_status": result.shell_exit_status,
            "terminating_signal": result.terminating_signal,
            "terminating_signal_name": result.terminating_signal_name,
            "duration_seconds": time.monotonic() - attempt_started,
            "scientific_duration_seconds": result.duration_seconds,
            "peak_sampled_process_tree_rss_bytes": result.peak_sampled_process_tree_rss_bytes,
            "maximum_output_directory_bytes": result.maximum_output_directory_bytes,
            "resource_samples": result.samples,
            "last_sampled_process_tree_pids": result.last_sampled_pids,
            "termination": result.termination,
            "launch_error": result.launch_error,
            "monitor_error": result.monitor_error,
            "scientific_timestamps": {
                "started_at_utc": result.started_at_utc,
                "ended_at_utc": result.ended_at_utc,
            },
        }
    )
    manifest["timestamps"]["attempt_ended_at_utc"] = utc_now()
    finalize_manifest(manifest_path, manifest, output)
    if result.outcome == "completed":
        return 0
    if result.shell_exit_status not in (None, 0):
        return int(result.shell_exit_status)
    return 2


def fake_child(args: argparse.Namespace) -> int:
    if args._fake_child == "success":
        pathlib.Path(args._fake_path).write_bytes(b"ok")
        return 0
    if args._fake_child == "sleep":
        time.sleep(args._fake_seconds)
        return 0
    if args._fake_child == "disk":
        pathlib.Path(args._fake_path).write_bytes(b"x" * args._fake_bytes)
        time.sleep(args._fake_seconds)
        return 0
    if args._fake_child == "tree":
        child = subprocess.Popen(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--refinement",
                "1",
                "--maximum-rank",
                "1",
                "--output",
                str(pathlib.Path(args._fake_path).parent / "unused"),
                "--_fake-child",
                "allocate",
                "--_fake-bytes",
                str(args._fake_bytes),
                "--_fake-seconds",
                str(args._fake_seconds),
            ]
        )
        allocation = bytearray(args._fake_bytes)
        allocation[0:1] = b"x"
        child.wait()
        return 0
    if args._fake_child == "allocate":
        allocation = bytearray(args._fake_bytes)
        allocation[0:1] = b"x"
        time.sleep(args._fake_seconds)
        return 0
    return 2


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--profile",
        choices=(
            PRIMARY_PROFILE,
            *RANK_EXTENSION_PROFILES,
            *HIGH_OBSERVATION_PROFILES,
            *REPLAY_PROFILES,
        ),
        default=PRIMARY_PROFILE,
    )
    parser.add_argument("--refinement", required=True, type=int)
    parser.add_argument("--maximum-rank", type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--perf0-archive", type=pathlib.Path)
    parser.add_argument("--basis-workspace", type=pathlib.Path)
    parser.add_argument("--wall-time-limit-seconds", type=float, default=DEFAULT_WALL_TIME_SECONDS)
    parser.add_argument("--rss-cap-bytes", type=int, default=DEFAULT_RSS_CAP_BYTES)
    parser.add_argument("--disk-cap-bytes", type=int, default=DEFAULT_DISK_CAP_BYTES)
    parser.add_argument(
        "--sample-interval-seconds", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--_fake-child",
        choices=("success", "sleep", "disk", "tree", "allocate"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_fake-path", default="fake-output", help=argparse.SUPPRESS)
    parser.add_argument("--_fake-bytes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_fake-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args._fake_child is not None:
        return fake_child(args)
    return run_attempt(args)


if __name__ == "__main__":
    raise SystemExit(main())
