# C51 fresh-process runner

Run one release-mode smoke replicate into a new directory:

```text
python3 scripts/c51/fresh_process_runner.py \
  --profile smoke \
  --replicate 0 \
  --threads 1 \
  --output out/c51/smoke-r000
```

This default mode retains the complete all-method accuracy archive. Its process
RSS is not attributed to an individual method.

For one method-owned setup measurement, select exactly one refinement,
observation, workload size, method, and phase:

```text
python3 scripts/c51/fresh_process_runner.py \
  --profile smoke \
  --replicate 0 \
  --threads 1 \
  --output out/c51/compact-setup-r000 \
  --resource-method whitney_compact \
  --refinement 2 \
  --observation-class local_physical_b \
  --sensor-count 4 \
  --repeated-data 10 \
  --workload-phase cold_setup
```

Use `--sensor-density 0.02` instead of `--sensor-count` to resolve a density
against that method's active state dimension. Use `query_full_uq` to include
replacement-data means, covariance actions, physical-QoI moments, posterior
sampling, and transformed variance. Low-rank spectral selections additionally
require `--spectral-rank`; other methods reject it.

The runner first builds the `c51_benchmark` example with `--locked --release`,
then executes the binary directly so compilation is not included in scientific
timing. It refuses to overwrite an existing run directory.

Useful checks:

```text
python3 scripts/c51/fresh_process_runner.py --dry-run \
  --profile smoke --replicate 0 --output out/c51/smoke-r000

python3 scripts/c51/fresh_process_runner.py --self-test

python3 -m unittest scripts/c51/test_fresh_process_runner.py
```

On macOS, `/usr/bin/time -l` supplies peak resident bytes. On Linux, the
runner tries to create a delegated cgroup-v2 child and reads `memory.peak`; GNU
`time -v` is recorded independently. If cgroup delegation or the platform time
tool is unavailable, the JSON manifest retains an explicit reason. Use
`--cgroup-mode required` for production Linux runs that must fail rather than
proceed without cgroup-v2 measurement.

The fresh-process runner implements only the frozen smoke scientific constants.
It rejects other profile names rather than silently substituting smoke.
`replicate` identifies the operations replicate; scientific seeds remain fixed
for paired timing.

In resource mode the Rust process constructs only the selected scientific
model and emits exactly one row with `resource_scope=method_phase`. The runner
refuses attribution unless the CSV contains exactly one matching method and
phase, then writes measured setup RSS only to
`factor_external_peak_rss_bytes` or full-UQ RSS only to
`query_external_peak_rss_bytes`. Accuracy fields are typed unavailable because
no reference model is constructed. The ordinary all-method mode remains an
archive-process discovery measurement and is never row-attributed.
