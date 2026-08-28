# C51 fresh-process runner

Run one release-mode smoke replicate into a new directory:

```text
python3 scripts/c51/fresh_process_runner.py \
  --profile smoke \
  --replicate 0 \
  --threads 1 \
  --output out/c51/smoke-r000
```

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

The current Rust example implements only `C51BenchmarkConfig::smoke()`. The
runner therefore rejects other profile names instead of silently running smoke
under another label. `replicate` identifies the fresh-process operations
replicate; the scientific smoke seeds remain fixed for paired timing.
