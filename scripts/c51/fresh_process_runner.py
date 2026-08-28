#!/usr/bin/env python3
"""Run one C51 benchmark replicate in a fresh, resource-monitored process.

The scientific benchmark remains owned by the Rust example.  This runner owns
only release-build preflight, process isolation, resource acquisition, and an
immutable JSON operations manifest.  Missing counters are represented as
explicit unavailable measurements; no resource value is inferred from another
counter without naming the selected source.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "c51-fresh-process-v2"
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "FAER_NUM_THREADS",
)
SUPPORTED_PROFILES = ("smoke",)
RESOURCE_METHODS = (
    "fm_direct",
    "fm_aug",
    "whitney_compact",
    "spectral_full_rank",
    "spectral_low_rank",
)
OBSERVATION_CLASSES = ("local_physical_b", "local_selectors", "global_row")
WORKLOAD_PHASES = ("cold_setup", "query_full_uq")


def measured(value: Any, unit: str, source: str) -> dict[str, Any]:
    return {"status": "measured", "value": value, "unit": unit, "source": source}


def unavailable(reason: str, source: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "unavailable", "reason": reason}
    if source is not None:
        result["source"] = source
    return result


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def command_text(argv: Sequence[str]) -> str:
    return shlex.join([str(value) for value in argv])


def run_capture(argv: Sequence[str], cwd: pathlib.Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(value) for value in argv],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        return {
            "status": "unavailable",
            "command": command_text(argv),
            "reason": str(error),
        }
    return {
        "status": "measured",
        "command": command_text(argv),
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def captured_stdout(result: Mapping[str, Any]) -> str | None:
    if result.get("status") == "measured" and result.get("return_code") == 0:
        return str(result.get("stdout", "")).strip() or None
    return None


def git_metadata(repository: pathlib.Path) -> dict[str, Any]:
    head = run_capture(["git", "rev-parse", "HEAD"], repository)
    status = run_capture(["git", "status", "--porcelain"], repository)
    branch = run_capture(["git", "branch", "--show-current"], repository)
    submodules = run_capture(["git", "submodule", "status", "--recursive"], repository)
    return {
        "root": str(repository),
        "head": captured_stdout(head),
        "branch": captured_stdout(branch),
        "dirty": (
            bool(str(status.get("stdout", "")))
            if status.get("status") == "measured" and status.get("return_code") == 0
            else None
        ),
        "head_query": head,
        "status_query": status,
        "submodule_status": submodules,
        "submodules": {
            name: repository_component_metadata(repository / relative)
            for name, relative in (("feec", "feec"), ("gmrf", "gmrf-rs"))
        },
    }


def repository_component_metadata(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "status": "unavailable", "reason": "directory missing"}
    head = run_capture(["git", "rev-parse", "HEAD"], path)
    status = run_capture(["git", "status", "--porcelain"], path)
    return {
        "path": str(path),
        "head": captured_stdout(head),
        "dirty": (
            bool(str(status.get("stdout", "")))
            if status.get("status") == "measured" and status.get("return_code") == 0
            else None
        ),
        "head_query": head,
        "status_query": status,
    }


def read_first(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def linux_cpu_model() -> str | None:
    text = read_first(pathlib.Path("/proc/cpuinfo"))
    if text is None:
        return None
    for line in text.splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def linux_memory_bytes() -> int | None:
    text = read_first(pathlib.Path("/proc/meminfo"))
    if text is None:
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def sysctl_value(name: str) -> str | None:
    result = run_capture(["sysctl", "-n", name], pathlib.Path.cwd())
    return captured_stdout(result)


def integer_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def hardware_metadata(thread_environment: Mapping[str, str]) -> dict[str, Any]:
    system = platform.system()
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            affinity = None
    if system == "Darwin":
        cpu_model = sysctl_value("machdep.cpu.brand_string")
        logical_cores = integer_or_none(sysctl_value("hw.logicalcpu"))
        physical_cores = integer_or_none(sysctl_value("hw.physicalcpu"))
        total_memory = integer_or_none(sysctl_value("hw.memsize"))
    else:
        cpu_model = linux_cpu_model()
        logical_cores = os.cpu_count()
        physical_cores = None
        total_memory = linux_memory_bytes()
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "operating_system": system,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "total_memory_bytes": total_memory,
        "cpu_affinity": affinity,
        "thread_environment": dict(thread_environment),
    }


def tool_metadata(repository: pathlib.Path) -> dict[str, Any]:
    commands = {
        "python": [sys.executable, "--version"],
        "rustc": ["rustc", "--version", "--verbose"],
        "cargo": ["cargo", "--version", "--verbose"],
        "git": ["git", "--version"],
    }
    return {name: run_capture(argv, repository) for name, argv in commands.items()}


def thread_environment(count: int) -> dict[str, str]:
    if count < 1:
        raise ValueError("thread count must be positive")
    return {name: str(count) for name in THREAD_ENV_NAMES}


def sha256_file(path: pathlib.Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def file_record(path: pathlib.Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as error:
        return {"path": str(path), "status": "unavailable", "reason": str(error)}
    return {
        "path": str(path),
        "status": "measured",
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class TimeTool:
    kind: str
    path: pathlib.Path | None
    metadata: dict[str, Any]

    def wrap(self, command: Sequence[str], output: pathlib.Path) -> list[str]:
        if self.path is None:
            return [str(item) for item in command]
        if self.kind == "gnu":
            return [str(self.path), "-v", "-o", str(output), *map(str, command)]
        if self.kind == "macos_bsd":
            return [str(self.path), "-l", "-o", str(output), *map(str, command)]
        raise RuntimeError(f"unsupported time tool kind {self.kind}")


def detect_time_tool(repository: pathlib.Path) -> TimeTool:
    candidate = pathlib.Path("/usr/bin/time")
    if not candidate.is_file():
        return TimeTool(
            "unavailable",
            None,
            unavailable("/usr/bin/time is not installed", "/usr/bin/time"),
        )
    if platform.system() == "Darwin":
        return TimeTool(
            "macos_bsd",
            candidate,
            {
                "status": "measured",
                "kind": "macos_bsd",
                "path": str(candidate),
                "rss_unit": "bytes",
            },
        )
    version = run_capture([str(candidate), "--version"], repository)
    stdout = str(version.get("stdout", ""))
    if version.get("return_code") == 0 and "GNU" in stdout:
        return TimeTool(
            "gnu",
            candidate,
            {
                "status": "measured",
                "kind": "gnu",
                "path": str(candidate),
                "version": stdout.strip(),
                "rss_unit": "kbytes_converted_to_bytes",
            },
        )
    return TimeTool(
        "unavailable",
        None,
        unavailable(
            "/usr/bin/time exists but is neither detected GNU time nor supported macOS BSD time",
            str(candidate),
        )
        | {"probe": version},
    )


def parse_time_measurements(kind: str, path: pathlib.Path) -> dict[str, Any]:
    text = read_first(path)
    if text is None:
        return {
            "peak_rss_bytes": unavailable("time output file was not produced", str(path)),
            "user_cpu_seconds": unavailable("time output file was not produced", str(path)),
            "system_cpu_seconds": unavailable("time output file was not produced", str(path)),
        }
    if kind == "gnu":
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        user = re.search(r"User time \(seconds\):\s*([0-9.eE+-]+)", text)
        system = re.search(r"System time \(seconds\):\s*([0-9.eE+-]+)", text)
        return {
            "peak_rss_bytes": (
                measured(int(rss.group(1)) * 1024, "bytes", "gnu_time_maximum_rss")
                if rss
                else unavailable("GNU time output has no maximum RSS field", str(path))
            ),
            "user_cpu_seconds": (
                measured(float(user.group(1)), "seconds", "gnu_time")
                if user
                else unavailable("GNU time output has no user CPU field", str(path))
            ),
            "system_cpu_seconds": (
                measured(float(system.group(1)), "seconds", "gnu_time")
                if system
                else unavailable("GNU time output has no system CPU field", str(path))
            ),
        }
    if kind == "macos_bsd":
        rss = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", text, re.MULTILINE)
        cpu = re.search(
            r"^\s*([0-9.eE+-]+)\s+real\s+([0-9.eE+-]+)\s+user\s+([0-9.eE+-]+)\s+sys\s*$",
            text,
            re.MULTILINE,
        )
        return {
            "peak_rss_bytes": (
                measured(int(rss.group(1)), "bytes", "macos_bsd_time_maximum_rss")
                if rss
                else unavailable("macOS time output has no maximum RSS field", str(path))
            ),
            "user_cpu_seconds": (
                measured(float(cpu.group(2)), "seconds", "macos_bsd_time")
                if cpu
                else unavailable("macOS time output has no user CPU field", str(path))
            ),
            "system_cpu_seconds": (
                measured(float(cpu.group(3)), "seconds", "macos_bsd_time")
                if cpu
                else unavailable("macOS time output has no system CPU field", str(path))
            ),
        }
    return {
        "peak_rss_bytes": unavailable("no supported external time tool was used"),
        "user_cpu_seconds": unavailable("no supported external time tool was used"),
        "system_cpu_seconds": unavailable("no supported external time tool was used"),
    }


@dataclass
class CgroupScope:
    path: pathlib.Path | None
    measurement: dict[str, Any]
    cleanup_reason: str | None = None


def current_cgroup_v2_path() -> pathlib.Path | None:
    membership = read_first(pathlib.Path("/proc/self/cgroup"))
    if membership is None:
        return None
    for line in membership.splitlines():
        if line.startswith("0::"):
            relative = line[3:].lstrip("/")
            return pathlib.Path("/sys/fs/cgroup") / relative
    return None


def create_cgroup_scope(mode: str) -> CgroupScope:
    if mode == "off":
        return CgroupScope(None, unavailable("cgroup measurement disabled by --cgroup-mode"))
    if platform.system() != "Linux":
        return CgroupScope(None, unavailable("cgroup-v2 memory.peak is Linux-only"))
    root = pathlib.Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        return CgroupScope(None, unavailable("the host is not exposing cgroup v2"))
    parent = current_cgroup_v2_path()
    if parent is None:
        return CgroupScope(None, unavailable("current cgroup-v2 membership is unavailable"))
    child = parent / f"feg-c51-{os.getpid()}-{uuid.uuid4().hex[:10]}"
    try:
        child.mkdir(mode=0o700)
    except OSError as error:
        return CgroupScope(
            None,
            unavailable(f"cannot create delegated child cgroup: {error}", str(parent)),
        )
    if not (child / "memory.peak").is_file():
        try:
            child.rmdir()
        except OSError:
            pass
        return CgroupScope(None, unavailable("child cgroup has no memory.peak", str(child)))
    return CgroupScope(child, {"status": "pending", "source": "cgroup_v2_memory_peak"})


def read_and_remove_cgroup(scope: CgroupScope) -> dict[str, Any]:
    if scope.path is None:
        return scope.measurement
    peak_path = scope.path / "memory.peak"
    try:
        raw = peak_path.read_text(encoding="utf-8").strip()
        value = int(raw)
        result = measured(value, "bytes", "cgroup_v2_memory_peak")
        result["cgroup"] = str(scope.path)
    except (OSError, ValueError) as error:
        result = unavailable(f"cannot read cgroup memory.peak: {error}", str(peak_path))
    try:
        scope.path.rmdir()
    except OSError as error:
        result["cleanup_warning"] = str(error)
    return result


def cgroup_exec(script: pathlib.Path, cgroup: pathlib.Path, command: Sequence[str]) -> list[str]:
    return [sys.executable, str(script), "--_cgroup-exec", str(cgroup), "--", *command]


def normalized_exit_status(return_code: int) -> tuple[int, int | None]:
    if return_code >= 0:
        return return_code, None
    signal_number = -return_code
    return 128 + signal_number, signal_number


@dataclass
class ProcessResult:
    manifest: dict[str, Any]
    shell_exit_status: int


def select_peak_rss(cgroup_value: dict[str, Any], time_value: dict[str, Any]) -> dict[str, Any]:
    if cgroup_value.get("status") == "measured":
        return {
            **cgroup_value,
            "selection_rule": "prefer cgroup-v2 memory.peak when measured",
            "other_source_status": time_value.get("status"),
        }
    if time_value.get("status") == "measured":
        return {
            **time_value,
            "selection_rule": "external time RSS selected because cgroup-v2 was explicitly unavailable",
            "cgroup_unavailable_reason": cgroup_value.get("reason"),
        }
    return unavailable(
        "neither cgroup-v2 memory.peak nor the platform time tool produced peak RSS"
    ) | {"cgroup": cgroup_value, "time": time_value}


def run_measured_process(
    command: Sequence[str],
    cwd: pathlib.Path,
    output_directory: pathlib.Path,
    environment: Mapping[str, str],
    cgroup_mode: str,
    script_path: pathlib.Path,
) -> ProcessResult:
    stdout_path = output_directory / "stdout.log"
    stderr_path = output_directory / "stderr.log"
    time_path = output_directory / "external-time.txt"
    exit_record_path = output_directory / "scientific-exit.json"
    time_tool = detect_time_tool(cwd)
    cgroup = create_cgroup_scope(cgroup_mode)
    if cgroup_mode == "required" and cgroup.path is None:
        raise RuntimeError(str(cgroup.measurement.get("reason", "cgroup-v2 unavailable")))

    scientific_command = [str(value) for value in command]
    measured_command = scientific_command
    if cgroup.path is not None:
        measured_command = cgroup_exec(script_path, cgroup.path, measured_command)
    exit_record_command = [
        sys.executable,
        str(script_path),
        "--_exit-record",
        str(exit_record_path),
        "--",
        *measured_command,
    ]
    wrapped_command = time_tool.wrap(exit_record_command, time_path)

    started_at = utc_now()
    monotonic_start = time.monotonic_ns()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.run(
                wrapped_command,
                cwd=cwd,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        measurement_wrapper_return_code = process.returncode
        launch_error = None
    except OSError as error:
        measurement_wrapper_return_code = 127
        launch_error = str(error)
        stderr_path.write_text(f"runner launch error: {error}\n", encoding="utf-8")
        if not stdout_path.exists():
            stdout_path.write_bytes(b"")
    elapsed_ns = time.monotonic_ns() - monotonic_start
    ended_at = utc_now()

    cgroup_measurement = read_and_remove_cgroup(cgroup)
    time_measurements = parse_time_measurements(time_tool.kind, time_path)
    try:
        exit_record = json.loads(exit_record_path.read_text(encoding="utf-8"))
        raw_return_code = int(exit_record["raw_return_code"])
        exit_record_status: dict[str, Any] = {
            "status": "measured",
            "path": str(exit_record_path),
            "sha256": sha256_file(exit_record_path),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raw_return_code = measurement_wrapper_return_code
        exit_record_status = unavailable(
            f"scientific exit record was not available; using resource-wrapper status: {error}",
            str(exit_record_path),
        )
    shell_status, signal_number = normalized_exit_status(raw_return_code)
    manifest = {
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "scientific_command": scientific_command,
        "measured_command": wrapped_command,
        "scientific_command_text": command_text(scientific_command),
        "measured_command_text": command_text(wrapped_command),
        "raw_return_code": raw_return_code,
        "shell_exit_status": shell_status,
        "measurement_wrapper_return_code": measurement_wrapper_return_code,
        "scientific_exit_record": exit_record_status,
        "terminating_signal": signal_number,
        "terminating_signal_name": (
            signal.Signals(signal_number).name if signal_number is not None else None
        ),
        "launch_error": launch_error,
        "wall_time": measured(elapsed_ns, "nanoseconds", "python_monotonic_ns"),
        "time_tool": time_tool.metadata,
        "time_measurements": time_measurements,
        "cgroup_peak_rss_bytes": cgroup_measurement,
        "peak_rss_archive_process": select_peak_rss(
            cgroup_measurement, time_measurements["peak_rss_bytes"]
        ),
        "peak_rss_row_attribution": unavailable(
            "the current example executes all methods and designs in one process; "
            "archive-process peak RSS cannot be attached to a method row or used for a crossover"
        ),
        "artifacts": {
            "stdout": file_record(stdout_path),
            "stderr": file_record(stderr_path),
            "external_time": file_record(time_path),
            "scientific_exit": file_record(exit_record_path),
        },
    }
    return ProcessResult(manifest, shell_status)


def cargo_target_directory(repository: pathlib.Path, environment: Mapping[str, str]) -> pathlib.Path:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repository,
        env=dict(environment),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cargo metadata failed: {result.stderr.strip()}")
    metadata = json.loads(result.stdout)
    return pathlib.Path(metadata["target_directory"])


def release_example_path(repository: pathlib.Path, environment: Mapping[str, str]) -> pathlib.Path:
    suffix = ".exe" if os.name == "nt" else ""
    return cargo_target_directory(repository, environment) / "release" / "examples" / (
        "c51_benchmark" + suffix
    )


def inventory(directory: pathlib.Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [
        file_record(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def build_release_example(
    repository: pathlib.Path,
    output_directory: pathlib.Path,
    environment: Mapping[str, str],
    cargo: str,
    features: str,
) -> dict[str, Any]:
    command = [
        cargo,
        "build",
        "--locked",
        "--release",
        "-p",
        "feg-case-studies",
        "--features",
        features,
        "--example",
        "c51_benchmark",
    ]
    stdout_path = output_directory / "build.stdout.log"
    stderr_path = output_directory / "build.stderr.log"
    started = time.monotonic_ns()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(
                command,
                cwd=repository,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        return_code = result.returncode
        launch_error = None
    except OSError as error:
        return_code = 127
        launch_error = str(error)
        stdout_path.write_bytes(b"")
        stderr_path.write_text(f"build launch error: {error}\n", encoding="utf-8")
    shell_status, signal_number = normalized_exit_status(return_code)
    return {
        "command": command,
        "command_text": command_text(command),
        "raw_return_code": return_code,
        "shell_exit_status": shell_status,
        "terminating_signal": signal_number,
        "launch_error": launch_error,
        "wall_time": measured(time.monotonic_ns() - started, "nanoseconds", "python_monotonic_ns"),
        "artifacts": {
            "stdout": file_record(stdout_path),
            "stderr": file_record(stderr_path),
        },
    }


def build_features(selection: Mapping[str, Any] | None) -> str:
    if selection is not None and selection["method"] == "spectral_low_rank":
        return "experimental,external-solvers"
    return "experimental"


def requested_resource_selection(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.resource_method is None:
        resource_values = (
            args.refinement,
            args.observation_class,
            args.sensor_count,
            args.sensor_density,
            args.repeated_data,
            args.spectral_rank,
            args.workload_phase,
        )
        if any(value is not None for value in resource_values):
            raise ValueError("resource selectors require --resource-method")
        return None
    required = {
        "--refinement": args.refinement,
        "--observation-class": args.observation_class,
        "--repeated-data": args.repeated_data,
        "--workload-phase": args.workload_phase,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"resource mode is missing {', '.join(missing)}")
    if args.refinement < 2:
        raise ValueError("resource refinement must be at least two")
    if args.repeated_data < 1:
        raise ValueError("resource repeated-data count must be positive")
    if (args.sensor_count is None) == (args.sensor_density is None):
        raise ValueError("resource mode requires exactly one of --sensor-count or --sensor-density")
    if args.sensor_count is not None and args.sensor_count < 1:
        raise ValueError("resource sensor count must be positive")
    if args.sensor_density is not None and not (0.0 < args.sensor_density <= 1.0):
        raise ValueError("resource sensor density must lie in (0, 1]")
    if args.observation_class == "global_row" and args.sensor_count != 1:
        raise ValueError("the global-row resource control requires --sensor-count 1")
    if args.resource_method == "spectral_low_rank":
        if args.spectral_rank is None or args.spectral_rank < 1:
            raise ValueError("spectral_low_rank requires a positive --spectral-rank")
    elif args.spectral_rank is not None:
        raise ValueError("only spectral_low_rank accepts --spectral-rank")
    return {
        "method": args.resource_method,
        "refinement": args.refinement,
        "observation_class": args.observation_class,
        "sensor_target_kind": "count" if args.sensor_count is not None else "density",
        "sensor_target_value": (
            args.sensor_count if args.sensor_count is not None else args.sensor_density
        ),
        "repeated_data_count": args.repeated_data,
        "spectral_rank": args.spectral_rank,
        "workload_phase": args.workload_phase,
    }


def scientific_command(
    binary: pathlib.Path,
    study_output: pathlib.Path,
    profile: str,
    selection: Mapping[str, Any] | None,
) -> list[str]:
    if selection is None:
        return [str(binary), profile, str(study_output)]
    return [
        str(binary),
        "resource",
        profile,
        str(study_output),
        str(selection["refinement"]),
        str(selection["observation_class"]),
        str(selection["sensor_target_kind"]),
        str(selection["sensor_target_value"]),
        str(selection["repeated_data_count"]),
        str(selection["method"]),
        "none" if selection["spectral_rank"] is None else str(selection["spectral_rank"]),
        str(selection["workload_phase"]),
    ]


def resource_scope(selection: Mapping[str, Any] | None) -> dict[str, Any]:
    if selection is None:
        return {
            "kind": "archive_process",
            "publication_crossover_eligible": False,
            "reason": "the all-method accuracy archive shares one process and cannot own method RSS",
        }
    return {
        "kind": "method_phase",
        "row_attribution_eligible": True,
        "publication_crossover_eligible": False,
        "selection": dict(selection),
        "reason": "one process owns one method-phase RSS measurement; publication eligibility still depends on the row's method-performance and full-UQ flags",
    }


def attach_peak_rss_to_resource_row(
    rows_path: pathlib.Path,
    selection: Mapping[str, Any],
    peak_rss: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        with rows_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read resource rows for RSS attribution: {error}") from error
    if fieldnames is None or len(rows) != 1:
        raise ValueError(
            f"resource RSS attribution requires exactly one CSV row; found {len(rows)}"
        )
    row = rows[0]
    expected = {
        "resource_scope": "method_phase",
        "workload_phase": str(selection["workload_phase"]),
        "method": str(selection["method"]),
        "refinement": str(selection["refinement"]),
        "observation_class": str(selection["observation_class"]),
        "repeated_data_count": str(selection["repeated_data_count"]),
    }
    mismatches = {
        name: {"expected": value, "actual": row.get(name)}
        for name, value in expected.items()
        if row.get(name) != value
    }
    if selection["spectral_rank"] is not None:
        actual_rank = row.get("spectral_rank_value")
        expected_rank = str(selection["spectral_rank"])
        if row.get("spectral_rank_status") != "measured" or actual_rank != expected_rank:
            mismatches["spectral_rank"] = {
                "expected": expected_rank,
                "actual": actual_rank,
                "status": row.get("spectral_rank_status"),
            }
    if selection["sensor_target_kind"] == "count":
        expected_count = str(selection["sensor_target_value"])
        if row.get("sensor_count") != expected_count:
            mismatches["sensor_count"] = {
                "expected": expected_count,
                "actual": row.get("sensor_count"),
            }
    if mismatches:
        raise ValueError(f"resource RSS row does not match selection: {mismatches}")
    if peak_rss.get("status") != "measured":
        return unavailable(
            "method-phase RSS was not attached because no measured peak was available"
        ) | {"peak_rss": dict(peak_rss), "selection": dict(selection)}
    measurement_name = (
        "factor_external_peak_rss_bytes"
        if selection["workload_phase"] == "cold_setup"
        else "query_external_peak_rss_bytes"
    )
    columns = [
        f"{measurement_name}_status",
        f"{measurement_name}_value",
        f"{measurement_name}_reason",
    ]
    missing_columns = [name for name in columns if name not in fieldnames]
    if missing_columns:
        raise ValueError(f"resource CSV is missing RSS columns: {missing_columns}")
    row[columns[0]] = "measured"
    row[columns[1]] = str(peak_rss["value"])
    row[columns[2]] = ""
    temporary = rows_path.with_name(f".{rows_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, rows_path)
    except OSError as error:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise ValueError(f"cannot atomically attach resource RSS: {error}") from error
    return measured(int(peak_rss["value"]), "bytes", str(peak_rss.get("source", "unknown"))) | {
        "resource_scope": "method_phase",
        "workload_phase": selection["workload_phase"],
        "method": selection["method"],
        "rows_csv": str(rows_path),
        "measurement_field": measurement_name,
    }


def plan(args: argparse.Namespace, repository: pathlib.Path) -> dict[str, Any]:
    selection = requested_resource_selection(args)
    features = build_features(selection)
    thread_env = thread_environment(args.threads)
    environment = os.environ.copy()
    environment.update(thread_env)
    environment["FEG_C51_PROFILE"] = args.profile
    environment["FEG_C51_REPLICATE"] = str(args.replicate)
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", repository / "target"))
    binary = target / "release" / "examples" / "c51_benchmark"
    study_output = args.output / "study"
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run",
        "profile": args.profile,
        "replicate": args.replicate,
        "profile_support": {
            "supported": list(SUPPORTED_PROFILES),
            "note": "smoke supports either the all-method accuracy archive or one method-phase resource selection",
        },
        "resource_scope": resource_scope(selection),
        "repository": str(repository),
        "output": str(args.output),
        "build_command": [
            args.cargo,
            "build",
            "--locked",
            "--release",
            "-p",
            "feg-case-studies",
            "--features",
            features,
            "--example",
            "c51_benchmark",
        ],
        "anticipated_binary": str(binary),
        "scientific_command": scientific_command(binary, study_output, args.profile, selection),
        "thread_environment": thread_env,
        "cgroup_mode": args.cgroup_mode,
    }


def run_benchmark(args: argparse.Namespace, script_path: pathlib.Path) -> int:
    repository = args.repository.resolve()
    if args.profile not in SUPPORTED_PROFILES:
        print(
            f"profile {args.profile!r} is not implemented by the current C51 Rust example; "
            f"supported profiles: {', '.join(SUPPORTED_PROFILES)}",
            file=sys.stderr,
        )
        return 2
    try:
        selection = requested_resource_selection(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.replicate < 0:
        print("replicate must be non-negative", file=sys.stderr)
        return 2
    if not (repository / "Cargo.toml").is_file():
        print(f"repository has no Cargo.toml: {repository}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(plan(args, repository), indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    if output.exists():
        print(f"refusing to overwrite existing run directory: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    study_output = output / "study"

    thread_env = thread_environment(args.threads)
    environment = os.environ.copy()
    environment.update(thread_env)
    environment["FEG_C51_PROFILE"] = args.profile
    environment["FEG_C51_REPLICATE"] = str(args.replicate)
    source = git_metadata(repository)
    tools = tool_metadata(repository)
    hardware = hardware_metadata(thread_env)
    initial_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "profile": args.profile,
        "replicate": args.replicate,
        "profile_support": {
            "supported": list(SUPPORTED_PROFILES),
            "note": "smoke supports all-method accuracy and isolated resource modes; replicate identifies the fresh-process operations replicate",
        },
        "resource_scope": resource_scope(selection),
        "runner": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "python": sys.version,
        },
        "source": source,
        "tools": tools,
        "hardware": hardware,
        "environment": {name: environment[name] for name in (*THREAD_ENV_NAMES, "FEG_C51_PROFILE", "FEG_C51_REPLICATE")},
        "requested_output": str(output),
        "started_at_utc": utc_now(),
    }
    manifest_path = output / "runner-manifest.json"
    atomic_write_json(manifest_path, initial_manifest)

    if args.skip_build:
        build = {
            "status": "skipped",
            "reason": "--skip-build requested; binary existence and source metadata still recorded",
            "shell_exit_status": 0,
        }
    else:
        build = build_release_example(
            repository,
            output,
            environment,
            args.cargo,
            build_features(selection),
        )

    try:
        binary = release_example_path(repository, environment)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        initial_manifest.update(
            {
                "status": "failed_preflight",
                "build": build,
                "preflight_error": str(error),
                "ended_at_utc": utc_now(),
            }
        )
        atomic_write_json(manifest_path, initial_manifest)
        return int(build.get("shell_exit_status", 2)) or 2

    binary_record = file_record(binary)
    if int(build.get("shell_exit_status", 0)) != 0 or not binary.is_file():
        exit_status = int(build.get("shell_exit_status", 2)) or 2
        initial_manifest.update(
            {
                "status": "build_failed" if exit_status != 0 else "binary_missing",
                "build": build,
                "binary": binary_record,
                "ended_at_utc": utc_now(),
            }
        )
        atomic_write_json(manifest_path, initial_manifest)
        return exit_status

    command = scientific_command(binary, study_output, args.profile, selection)
    try:
        result = run_measured_process(
            command,
            repository,
            output,
            environment,
            args.cgroup_mode,
            script_path,
        )
        result.manifest["resource_scope"] = resource_scope(selection)
        if result.shell_exit_status == 0 and selection is not None:
            result.manifest["peak_rss_row_attribution"] = attach_peak_rss_to_resource_row(
                study_output / "rows.csv",
                selection,
                result.manifest["peak_rss_archive_process"],
            )
        elif selection is not None:
            result.manifest["peak_rss_row_attribution"] = unavailable(
                "the selected scientific method phase failed, so no resource row was attributed"
            ) | {"selection": dict(selection)}
        initial_manifest.update(
            {
                "status": "completed" if result.shell_exit_status == 0 else "benchmark_failed",
                "build": build,
                "binary": binary_record,
                "process": result.manifest,
                "study_inventory": inventory(study_output),
                "ended_at_utc": utc_now(),
            }
        )
        atomic_write_json(manifest_path, initial_manifest)
        return result.shell_exit_status
    except (OSError, RuntimeError, ValueError) as error:
        initial_manifest.update(
            {
                "status": "runner_failed",
                "build": build,
                "binary": binary_record,
                "runner_error": str(error),
                "ended_at_utc": utc_now(),
            }
        )
        atomic_write_json(manifest_path, initial_manifest)
        print(f"runner failed: {error}", file=sys.stderr)
        return 2


def self_test_child(mode: str) -> int:
    if mode == "success":
        allocation = bytearray(8 * 1024 * 1024)
        allocation[0] = 1
        print(
            json.dumps(
                {
                    "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_NAMES},
                    "allocation_bytes": len(allocation),
                },
                sort_keys=True,
            )
        )
        print("self-test stderr sentinel", file=sys.stderr)
        time.sleep(0.03)
        return 0
    if mode == "failure":
        print("intentional self-test failure", file=sys.stderr)
        return 23
    print(f"unknown self-test child mode {mode}", file=sys.stderr)
    return 2


def exit_record_child(path: pathlib.Path, command: Sequence[str]) -> int:
    if not command:
        atomic_write_json(
            path,
            {"raw_return_code": 2, "launch_error": "exit-record wrapper received no command"},
        )
        return 2
    try:
        result = subprocess.run(list(command), check=False)
        raw_return_code = result.returncode
        launch_error = None
    except OSError as error:
        raw_return_code = 127
        launch_error = str(error)
    shell_status, signal_number = normalized_exit_status(raw_return_code)
    atomic_write_json(
        path,
        {
            "raw_return_code": raw_return_code,
            "shell_exit_status": shell_status,
            "terminating_signal": signal_number,
            "launch_error": launch_error,
        },
    )
    return shell_status


def run_self_test(script_path: pathlib.Path, cgroup_mode: str) -> int:
    with tempfile.TemporaryDirectory(prefix="c51-runner-self-test-") as temporary:
        root = pathlib.Path(temporary)
        environment = os.environ.copy()
        environment.update(thread_environment(1))
        success_dir = root / "success"
        failure_dir = root / "failure"
        success_dir.mkdir()
        failure_dir.mkdir()
        success = run_measured_process(
            [sys.executable, str(script_path), "--_self-test-child", "success"],
            pathlib.Path.cwd(),
            success_dir,
            environment,
            cgroup_mode,
            script_path,
        )
        failure = run_measured_process(
            [sys.executable, str(script_path), "--_self-test-child", "failure"],
            pathlib.Path.cwd(),
            failure_dir,
            environment,
            cgroup_mode,
            script_path,
        )
        output = json.loads((success_dir / "stdout.log").read_text(encoding="utf-8"))
        checks = {
            "success_exit_zero": success.shell_exit_status == 0,
            "failure_exit_preserved": failure.shell_exit_status == 23,
            "stdout_captured": output.get("allocation_bytes") == 8 * 1024 * 1024,
            "threads_pinned": all(
                output.get("thread_environment", {}).get(name) == "1"
                for name in THREAD_ENV_NAMES
            ),
            "rss_explicit": success.manifest["peak_rss_archive_process"].get("status")
            in {"measured", "unavailable"},
            "rss_not_row_attributed": success.manifest["peak_rss_row_attribution"].get("status")
            == "unavailable",
            "stderr_captured": "sentinel"
            in (success_dir / "stderr.log").read_text(encoding="utf-8"),
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "checks": checks,
            "success_process": success.manifest,
            "failure_process": failure.manifest,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if all(checks.values()) else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
        help="feec-gmrf-release checkout (default: inferred from this script)",
    )
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("out/c51-run"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--resource-method", choices=RESOURCE_METHODS)
    parser.add_argument("--refinement", type=int)
    parser.add_argument("--observation-class", choices=OBSERVATION_CLASSES)
    sensor_group = parser.add_mutually_exclusive_group()
    sensor_group.add_argument("--sensor-count", type=int)
    sensor_group.add_argument("--sensor-density", type=float)
    parser.add_argument("--repeated-data", type=int)
    parser.add_argument("--spectral-rank", type=int)
    parser.add_argument("--workload-phase", choices=WORKLOAD_PHASES)
    parser.add_argument(
        "--cgroup-mode",
        choices=("auto", "required", "off"),
        default="auto",
        help="use a delegated Linux cgroup-v2 child when available",
    )
    parser.add_argument("--_cgroup-exec", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--_exit-record", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--_self-test-child", choices=("success", "failure"), help=argparse.SUPPRESS)
    args, remainder = parser.parse_known_args(argv)
    args._remainder = remainder
    return args


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(sys.argv[1:] if argv is None else argv)
    script_path = pathlib.Path(__file__).resolve()
    if arguments._cgroup_exec is not None:
        command = list(arguments._remainder)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            print("cgroup executor received no command", file=sys.stderr)
            return 2
        try:
            (arguments._cgroup_exec / "cgroup.procs").write_text(
                f"{os.getpid()}\n", encoding="ascii"
            )
        except OSError as error:
            print(f"cannot enter requested cgroup: {error}", file=sys.stderr)
            return 125
        os.execvpe(command[0], command, os.environ.copy())
        return 127
    if arguments._exit_record is not None:
        command = list(arguments._remainder)
        if command and command[0] == "--":
            command = command[1:]
        return exit_record_child(arguments._exit_record, command)
    if arguments._self_test_child is not None:
        return self_test_child(arguments._self_test_child)
    if arguments._remainder:
        print(f"unexpected arguments: {arguments._remainder}", file=sys.stderr)
        return 2
    if arguments.self_test:
        return run_self_test(script_path, arguments.cgroup_mode)
    try:
        thread_environment(arguments.threads)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    return run_benchmark(arguments, script_path)


if __name__ == "__main__":
    raise SystemExit(main())
