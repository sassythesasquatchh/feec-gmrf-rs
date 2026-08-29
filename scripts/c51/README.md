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

The fresh-process runner accepts the discovery `smoke` profile and the frozen
`juq-performance-v1` publication profile. It rejects any other profile name
rather than silently substituting smoke. `replicate` identifies the operations
replicate; scientific seeds remain fixed for paired timing.

In resource mode the Rust process constructs only the selected scientific
model and emits exactly one row with `resource_scope=method_phase`. The runner
refuses attribution unless the CSV contains exactly one matching method and
phase, then writes measured setup RSS only to
`factor_external_peak_rss_bytes` or query RSS only to
`query_external_peak_rss_bytes`. Accuracy fields are typed unavailable because
no reference model is constructed. The ordinary all-method mode remains an
archive-process discovery measurement and is never row-attributed.

## JUQ matched-accuracy performance campaign

The publication campaign is preregistered in `juq-performance-v1.json` and is
orchestrated by `run_juq_performance.py`. It uses the nested 12-by-12-by-12
fixed-footprint magnetic sensor design, radius 0.16, the canonical
`A -> D1 -> B` physical map, and exact scalar observation counts up to the
5,184-row capacity. The claim-bearing accuracy envelope is 5%. The 10% result
is context only and the 2% result is a sensitivity analysis; neither can be
silently substituted for the 5% rank decision.

Every method and workload phase runs in its own fresh process. The setup phase
owns assembly, storage, factorization time, and factor peak RSS. The common
`query_physical_uq` phase owns replacement-data physical-output means, the full
18-by-18 posterior physical-output covariance, its action and diagonal, plus
query peak RSS. It excludes latent covariance queries and state sampling,
which the exact observation-space GP does not provide. The
orchestrator merges only matched phase pairs, retains both source-manifest paths
and SHA-256 checksums, excludes warmups from evidence, and preserves timeout,
OOM/SIGKILL, rejected, and otherwise censored cells in the campaign manifest.
Publication execution requires Linux cgroup-v2 `memory.peak` attribution and a
clean recursive source checkout.

The frozen tiers are deliberately not one blind Cartesian campaign:

| Tier | Frozen cells | Replication role |
|---|---|---|
| `claim_primary` | n=8,10; m=2048; D=1,100 | 1 warmup + 3 measured paired replicates |
| `scaling_context` | n=8,10; m=512; D=1,100 | 2 measured paired replicates |
| `high_data_confirmation` | n=8,10; m=4096; D=100 | 3 measured paired replicates |
| `capacity_stress` | n=8,10; m=5184; D=100 | 1 measured paired stress/censor replicate |

All tiers together contain 192 fresh method-phase processes: 24 warmups and
168 measured processes, yielding 84 measured paired rows if no cell is
censored. The primary tier alone contains 96 processes and 36 measured pairs.

Ranks are inputs, not outputs, of this campaign. They are generated by the
focused held-out covariance study below and then combined without manual
editing. A dry run with the pending template reports the missing decision keys
and creates no cells:

```text
python3 scripts/c51/run_juq_performance.py --dry-run
```

By default, every tier is selected and all eight `(n,m)` rank decisions are
required. Select immutable tiers explicitly when their decisions become ready;
the plan and manifest record both selected and omitted tiers. For example, the
primary campaign can run as soon as the n=8 and n=10 decisions at m=2048 are
frozen, without inventing ranks for later tiers:

```text
python3 scripts/c51/run_juq_performance.py \
  --evidence-tier claim_primary \
  --rank-decisions path/to/frozen-rank-decisions.json \
  --output out/c51/juq-performance-v1-primary
```

After a complete uncensored campaign, adjudicate the frozen paired rule with:

```text
python3 scripts/c51/analyze_juq_performance.py \
  out/c51/juq-performance-v1-primary \
  --rank-decisions path/to/frozen-rank-decisions.json
```

For every `n=8,10`, `D=1,100`, and comparator, the upper endpoint of the
paired 95% median-ratio interval must show at least a 25% improvement in either
end-to-end time or gross peak RSS, with at most a 10% regression in the other.
Both the exact observation-space GP and the accuracy-matched spectral GP must
pass this rule. Three measured paired replicates make the interval deliberately
conservative and low-resolution.

Repeat `--evidence-tier` to run multiple ready tiers. A selected tier is
all-or-nothing: every required refinement/rank cell must be present, every
method pays its own setup in a fresh process, and missing decisions block the
entire selected tier.

### Determine the accuracy-matched spectral ranks

First run the bounded same-law `PERF-0` fixture. It checks the exact
observation-space GP against the supported FEEC law and records the compact
Whitney approximation on the identical frozen physical maps:

```text
cargo run --locked --release -p feg-case-studies --features experimental \
  --example c51_benchmark -- juq_accuracy_perf0 out/c51/perf0
```

Then run the two claim-bearing `m=2048` covariance studies. These are the
expensive runs: use the capped runner and new output directories. The selected
rank is the smallest with a lower failing bracket, a sustained 5% pass on the
selection bank, and an independent held-out-bank pass. The 2% result is
sensitivity only. The `n=10` grid reaches `r=N=6130`, so it can report
`full_rank_required` rather than the unresolved statement `>4096`.

```text
python3 scripts/c51/run_spectral_covariance_rank.py \
  --profile high-observation-primary-m2048-v1-n8 \
  --refinement 8 --perf0-archive out/c51/perf0 \
  --output out/c51/high-m-primary-n8

python3 scripts/c51/run_spectral_covariance_rank.py \
  --profile high-observation-primary-m2048-v1-n10 \
  --refinement 10 --perf0-archive out/c51/perf0 \
  --output out/c51/high-m-primary-n10
```

Each successful attempt writes `study/rank-decision.json`. Combine the two
decisions before planning the primary performance campaign:

```text
python3 scripts/c51/combine_rank_decisions.py \
  --output out/c51/primary-rank-decisions.json \
  out/c51/high-m-primary-n8/study/rank-decision.json \
  out/c51/high-m-primary-n10/study/rank-decision.json

python3 scripts/c51/run_juq_performance.py --dry-run \
  --evidence-tier claim_primary \
  --rank-decisions out/c51/primary-rank-decisions.json
```

## Four-method exact plus projected campaign

`juq-exact-projected-performance-v1.json` preserves the frozen three-method
exact campaign and adds `whitney_compact` as a fourth, explicitly
accuracy-unmatched Pareto point. The claim-bearing comparison remains
`fm_aug` versus the exact observation-space GP and the independently
2%-qualified spectral GP. Projected-Whitney resource ratios are descriptive:
they cannot clear the exact-law claim because the separately archived fixed-
`tau` covariance errors are material.

For state degree one, let `E0` and `E1` denote the FEEC projected sparse
approximations to the lower and state mass inverses. The fourth model assembles

```text
L_projected = D1^T M2 D1 + M1 D0 E0 D0^T M1
A_projected = kappa^2 M1 + L_projected
Q_projected = tau^2 A_projected E1 A_projected       (alpha = 2)
```

and conditions the ordinary `N`-coefficient GMRF through
`Q_projected + H^T R^-1 H`. There are no auxiliary variables or KKT
constraints. Both inverse sites use the canonical
`MaternPriorBuilder::from_feec_whitney_projected` path; the benchmark does not
reimplement either projection. The current submitted-policy implementation
assembles on the full cochain spaces and then applies active principal
restriction, rather than silently substituting the later boundary-first
ablation.

Derive a four-method decision artifact from the same `m=4096` accuracy
attempts and PERF-0 archive used by the exact campaign:

```text
python3 scripts/c51/derive_exact_rank_decisions.py \
  --profile juq-exact-projected-performance-v1 \
  --perf0-archive <perf0-archive> \
  --output out/c51/exact-projected-rank-decisions.json \
  <n8-m4096-attempt> <n10-m4096-attempt>

python3 scripts/c51/run_juq_performance.py \
  --profile-file scripts/c51/juq-exact-projected-performance-v1.json \
  --rank-decisions out/c51/exact-projected-rank-decisions.json \
  --evidence-tier claim_primary_exact \
  --output out/c51/juq-exact-projected-performance-v1-macos
```

The profile runs 32 warm-up and 96 measured method-phase processes. The
analyzer adjudicates only the two fair exact/spectral comparators and emits a
separate projected-Whitney table containing fixed-`tau` accuracy and paired
time, RSS, and persistent-storage ratios.

## Physically calibrated projected-Whitney campaign

`juq-exact-calibrated-projected-performance-v1.json` retains the raw projected
row and adds `whitney_compact_calibrated`. The latter uses the package-level
physical-RMS calibration pipeline before conditioning:

```text
sigma_exact^2 = tr(G_sel C_exact G_sel^T) / 9
sigma_raw^2   = tr(G_sel C_projected G_sel^T) / 9
s_Q           = sigma_raw^2 / sigma_exact^2
tau_cal       = tau * sqrt(s_Q)
Q_cal         = s_Q * Q_projected
```

`G_sel` is the frozen nine-output selection bank. The second nine-output bank
is held out from calibration and is used only for posterior validation. The
raw projected posterior is still archived, and the calibrated posterior is
reconditioned from `Q_cal`; an already-conditioned covariance is never scaled.
The fresh-process resource row recomputes the raw projected RMS through
`calibrate_prior_to_physical_rms`, charges that factorization and its nine
covariance solves to `prior_build_ns`, and includes their memory in peak RSS.

After producing new accuracy attempts with schema
`c51-spectral-covariance-rank-v3`, freeze the calibration and spectral rank:

```text
python3 scripts/c51/derive_exact_rank_decisions.py \
  --profile juq-exact-calibrated-projected-performance-v1 \
  --perf0-archive <perf0-archive> \
  --output out/c51/exact-calibrated-projected-rank-decisions.json \
  <n8-m4096-attempt> <n10-m4096-attempt>

python3 scripts/c51/run_juq_performance.py \
  --profile-file scripts/c51/juq-exact-calibrated-projected-performance-v1.json \
  --rank-decisions out/c51/exact-calibrated-projected-rank-decisions.json \
  --evidence-tier claim_primary_exact \
  --output out/c51/juq-exact-calibrated-projected-performance-v1-macos
```

The exact augmented, exact observation-space GP, and spectral comparison
remain the same-law C51 gate. The calibrated projected method is a separate
practical approximation claim: one physically meaningful amplitude parameter
is fitted, while covariance shape and independent held-out posterior accuracy
remain empirical tests.

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
