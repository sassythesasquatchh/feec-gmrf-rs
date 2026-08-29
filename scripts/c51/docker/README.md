# Native Linux C51 campaign

This image supplies a native ARM64 Ubuntu 24.04 environment for the
preregistered exact-law C51 performance campaign. The base image digest and
Rust toolchain are pinned. Runtime manifests still record the actual kernel,
CPU, compiler, PETSc/SLEPc, and source revisions used by every process.

Build the image from the repository root:

```sh
docker build --platform linux/arm64 \
  -t feec-gmrf-c51-linux:24.04 \
  -f scripts/c51/docker/Dockerfile scripts/c51/docker
```

First validate the resolved campaign without launching scientific cells:

```sh
docker run --rm --platform linux/arm64 --privileged \
  -v "$PWD:/workspace" \
  -v feec-gmrf-c51-linux-target:/target \
  feec-gmrf-c51-linux:24.04 \
  python3 scripts/c51/docker/run_exact_campaign.py \
  --rank-decisions out/c51/exact-rank-decisions.json \
  --output out/c51/juq-exact-performance-v1-linux \
  --dry-run
```

Remove `--dry-run` for the measured campaign. `--privileged` is required here
only because the runner creates one cgroup-v2 child per fresh process to read
that process's kernel-accounted peak memory. A missing writable cgroup-v2
mount is a hard error; the campaign never substitutes sampled RSS silently.

The rank-decision file must be derived from completed, clean, held-out spectral
accuracy attempts. Campaign output belongs under the ignored `out/` tree.
