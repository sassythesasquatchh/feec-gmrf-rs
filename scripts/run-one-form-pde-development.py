#!/usr/bin/env python3
"""Serial development runs; one fresh process per measurement, plain CSV output."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def worker(command, directory, timeout):
    start = time.perf_counter()
    with (directory / "console.log").open("w") as log:
        with subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True) as process:
            try:
                code = process.wait(timeout=timeout)
                status = "ok" if code == 0 else f"exit_{code}"
            except subprocess.TimeoutExpired:
                # The spectral adapter starts a solver helper. Bound the whole
                # process group so a timed-out eigensolve cannot remain running.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = "timeout"
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # Darwin reports bytes; Linux reports KiB. This worker has exactly one child.
    peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    record = {"status": status, "process_wall_s": time.perf_counter() - start,
              "process_peak_rss_bytes": peak}
    (directory / "process.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[4]), Path(sys.argv[2]), float(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["compact", "supported", "gp"], help="compact, supported, gp, spectral-RANK, or pde (continuum deterministic solve)")
    parser.add_argument("--fixed-scale-reference", type=Path, help="Existing RMS-calibrated run directory; use compact's scale for compact, supported's for GP and every spectral rank")
    parser.add_argument("--fixed-scale", type=float, help="One unchanged precision scale for every method and mesh")
    parser.add_argument("--workload", choices=["discrete", "continuum", "local"], default="discrete")
    parser.add_argument("--skip-warmup", action="store_true", help="Append measured repetitions after an existing pilot/warmup")
    parser.add_argument("--field", choices=["exact", "mc", "none", "bounds"], default="none")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for method in args.methods:
        if method not in {"compact", "supported", "gp", "pde"} and not (method.startswith("spectral-") and method[9:].isdigit() and int(method[9:]) > 0):
            parser.error(f"invalid method: {method}")
        if method.startswith("spectral-") and args.fixed_scale_reference is None and args.fixed_scale is None:
            parser.error("spectral methods require a fixed scale")
    if args.fixed_scale is not None and args.fixed_scale_reference is not None:
        parser.error("choose a fixed scale or a reference directory")
    if args.workload != "discrete" and args.fixed_scale is None:
        parser.error("continuum validation requires --fixed-scale, unchanged across meshes and methods")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = Path("target/release/examples/c51_one_form_pde_uq").resolve()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", RAYON_NUM_THREADS="1")
    timings = args.output / "timings.csv"
    # Additional levels/methods can join this development sweep. Existing run
    # directories remain protected by mkdir(exist_ok=False).
    rows = list(csv.DictReader(timings.open())) if timings.exists() else []
    for level in args.levels:
        for repeat in range(1 if args.skip_warmup else 0, args.repeats + 1):
            # Rotate method order across repetitions; repeat zero is warmup.
            methods = args.methods[repeat % len(args.methods):] + args.methods[:repeat % len(args.methods)]
            for method in methods:
                if method == "gp" and (level > 8 or args.field != "none"):
                    continue
                if method.startswith("spectral-") and (level > 12 or args.field != "none" or (level > 8 and int(method[9:]) > 256)):
                    continue
                if method == "supported" and args.field == "mc":
                    continue
                if method.startswith("spectral-") and int(method[9:]) > 7*level**3 - 9*level**2 + 3*level:
                    continue
                directory = args.output / f"{method}-n{level}-rep{repeat}"
                directory.mkdir(exist_ok=False)
                command = [str(binary), method, str(level), str(args.sigma), args.field, str(directory)]
                if args.fixed_scale_reference is not None:
                    reference_method = "compact" if method == "compact" else "supported"
                    reference = args.fixed_scale_reference / f"{reference_method}-n{level}-rep0/metrics.csv"
                    parameters = {row["quantity"]: float(row["value"]) for row in csv.DictReader(reference.open())}
                    if method.startswith("spectral-") and int(method[9:]) > parameters["state_dimension"]:
                        directory.rmdir()
                        continue
                    command.append(str(parameters["precision_scale"]))
                elif args.fixed_scale is not None:
                    command.append(str(args.fixed_scale))
                if args.workload != "discrete":
                    command.append(args.workload)
                result = subprocess.run([sys.executable, __file__, "--worker", str(directory), str(args.timeout), json.dumps(command)], env=env, text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                record = json.loads(result.stdout)
                record.update(method=method, refinement=level, repeat=repeat, warmup=repeat == 0, field_mode=args.field, pde_sigma=args.sigma, workload=args.workload)
                if args.fixed_scale_reference is not None:
                    record["parameter_reference"] = str(reference)
                metrics = directory / "metrics.csv"
                if metrics.exists():
                    record.update({row["quantity"]: float(row["value"]) for row in csv.DictReader(metrics.open())})
                rows.append(record)
                keys = sorted(set().union(*(row.keys() for row in rows)))
                with (args.output / "timings.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{method} n={level} rep={repeat}: {record['status']}, wall={record['process_wall_s']:.3f}s, peak={record['process_peak_rss_bytes']/2**30:.3f}GiB", flush=True)
                if record["status"] != "ok":
                    # Preserve the failing row and stop this bounded invocation.
                    raise SystemExit(1)


if __name__ == "__main__":
    main()
