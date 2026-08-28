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

A `spectral_low_rank` resource selection is built with
`experimental,external-solvers` and performs one independently requested
sparse mixed PETSc/SLEPc solve at exactly `--spectral-rank`. It has no dense or
shared-maximum fallback. The row records requested, converged, and retained
ranks, mixed-block nonzeros, the smallest-mode/direct-mass policy, a streamed
mixed-equation residual certificate, the `M_k`-orthogonality defect, validation
time, and helper provenance. Timing eligibility fails closed when any invariant
or certificate fails. The present convenience adapter reassembles compatible
full-relative layouts and mixed blocks; that duplication is recorded and its
time remains charged to the eigensolve rather than being silently subtracted.

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

## Spectral covariance-rank discovery guard

The dedicated rank-discovery study has a separate capped runner:

```text
python3 scripts/c51/run_spectral_covariance_rank.py \
  --refinement 8 \
  --maximum-rank 2048 \
  --output out/c51/spectral-rank-n8-r2048
```

It first builds `c51_spectral_covariance_rank` with the locked release profile,
outside scientific measurement. It then executes the release binary directly
with the immutable `publication-discovery-v1` profile and exactly one requested
refinement and maximum rank. The attempt directory must not already exist.

The post-primary rank extension is deliberately separate from that discovery
profile. It was frozen only after the corrected alpha-2 primary results left
registered 5%/2% decisions unresolved. Each mesh-specific profile has an
immutable rank grid, rejects `--maximum-rank`, records
`evidence_tier: rank_extension`, and remains timing-ineligible:

```text
python3 scripts/c51/run_spectral_covariance_rank.py \
  --profile rank-extension-v1-n8 \
  --refinement 8 \
  --output out/c51/spectral-rank-extension-v1-n8

python3 scripts/c51/run_spectral_covariance_rank.py \
  --profile rank-extension-v1-n10 \
  --refinement 10 \
  --output out/c51/spectral-rank-extension-v1-n10
```

The frozen grids are `2048,2304,2560,2816,3000,3032` for `n=8` and
`2048,2560,3072,3584,4096` for `n=10`, both at alpha 2. A fully measured
unattained `n=10` threshold is reported against the actual maximum tested rank
4,096, never against the primary cap 2,048. Resource-censored attempts remain
computationally unavailable rather than rank lower bounds.

When reporting the primary and extension together, pass the original summary
to the analyzer with `--primary-summary`. The resulting
`combined_rank_threshold_summary.csv` preserves every already-attained primary
rank and any primary held-out failure; the extension can resolve only a primary
`required_rank_gt_*` censor. In particular, a later rank whose held-out value
was already inspected cannot be used to retune a failed primary lock.

The child starts in a fresh process group. The runner samples the summed RSS of
that process tree, wall time, and all file sizes in the attempt directory. It
terminates the entire group when any cap is exceeded. Defaults are six hours,
16 GiB sampled RSS, 10 GiB output, one thread, and a one-second sample period;
override them explicitly with `--wall-time-limit-seconds`, `--rss-cap-bytes`,
`--disk-cap-bytes`, `--threads`, and `--sample-interval-seconds`.

Every started attempt retains `attempt_manifest.json`, including failures and
cap terminations. It records commands, caps, timestamps, exit/signal status,
peak sampled process-tree RSS, maximum output bytes, root and submodule Git
state, platform data, and available tool versions. Build and scientific stdout
and stderr are kept separately. Preview a run without building or creating the
output directory:

```text
python3 scripts/c51/run_spectral_covariance_rank.py --dry-run \
  --refinement 8 --maximum-rank 2048 \
  --output out/c51/spectral-rank-n8-r2048
```

The monitor uses `/proc` on Linux and the platform `ps` command on macOS. If it
cannot sample the process tree, it fails closed and terminates the child rather
than continuing without enforcing the RSS cap.
