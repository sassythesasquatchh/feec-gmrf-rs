# Clean installation and first run

This guide starts from a clean recursive checkout. The default Rust library and
most studies do not require PETSc. Hodge--Laplacian, annulus, and publication
validation workflows additionally require PETSc with MUMPS, SLEPc, Gmsh, MPI,
and Python NGSolve.

## 1. Obtain the source

Install Git and a Rust toolchain of at least 1.80. Rustup is the recommended
Rust installer: <https://rust-lang.org/tools/install/>.

Clone recursively so the exact FEEC and GMRF commits pinned by the parent are
checked out:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf.git
cd feec-gmrf
git submodule status
```

If the repository was cloned without `--recursive`, initialize it afterward:

```text
git submodule update --init --recursive
```

## 2. Build and test the portable Rust installation

The integration, FEEC, and GMRF manifests are release-tested independently.
All commands below run optimized builds:

```text
cargo build --release --workspace --exclude feg-experiments
cargo test --release --workspace --exclude feg-experiments --all-targets

cargo test --release --workspace --manifest-path feec/Cargo.toml
cargo test --release --all-targets --locked --manifest-path gmrf-rs/Cargo.toml

cargo check --release -p feg-experiments --all-targets
cargo test --release -p feg-experiments --all-targets
```

Cargo may promote in-tree path dependencies into the root workspace graph even
though `feec/` and `gmrf-rs/` are declared as exclusions. The explicit manifest
commands above are therefore part of the supported clean-install check.

## 3. Run an example without native solvers

List the registered studies and inspect a profile:

```text
cargo run --release -p feg-cli --bin feg-study -- list
cargo run --release -p feg-cli --bin feg-study -- \
  describe matern/trace-normalization
```

Run and verify a small deterministic Matérn example:

```text
cargo run --release -p feg-cli --bin feg-study -- \
  run matern/trace-normalization --profile smoke --output out/first-run
cargo run --release -p feg-cli --bin feg-study -- \
  verify out/first-run --against smoke
```

The output directory contains the resolved configuration, numerical metrics,
input hashes, command line, tool versions, and root/FEEC/GMRF commit provenance.

## 4. Install the publication solver stack

Skip this section when only the portable APIs and studies are needed. The
external workflows require:

- Gmsh;
- an MPI C compiler;
- PETSc built with MUMPS;
- a compatible SLEPc build;
- Python 3 with NGSolve; and
- Make.

### Ubuntu package installation

The publication CI starts with:

```text
sudo apt-get update
sudo apt-get install --yes \
  gmsh libopenmpi-dev libpetsc-real-dev libslepc-real-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ngsolve
```

Distribution PETSc configurations differ. The repository check below is
authoritative and will reject a package build that lacks MUMPS.

### Source installation on macOS or Linux

PETSc officially supports in-place configurations selected by `PETSC_DIR` and
`PETSC_ARCH`, and recommends `--download-PACKAGE` for external packages. See
<https://petsc.org/release/install/install/>. One suitable optimized real-scalar
configuration is:

```text
git clone -b release https://gitlab.com/petsc/petsc.git
cd petsc
export PETSC_DIR="$PWD"
export PETSC_ARCH=arch-mumps-opt
./configure \
  PETSC_ARCH="$PETSC_ARCH" \
  --with-debugging=0 \
  --download-fblaslapack \
  --download-metis \
  --download-mumps \
  --download-openmpi \
  --download-parmetis \
  --download-ptscotch \
  --download-scalapack
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all check
```

Build a compatible SLEPc release against that exact PETSc configuration. The
official SLEPc instructions require `SLEPC_DIR`, `PETSC_DIR`, and `PETSC_ARCH`
for an in-place build: <https://slepc.upv.es/release/installation/quickstart.html>.

```text
cd ..
git clone -b release https://gitlab.com/slepc/slepc.git
cd slepc
export SLEPC_DIR="$PWD"
./configure
make
make check
```

Keep these three variables set when returning to the FEEC--GMRF checkout:

```text
export PETSC_DIR=/absolute/path/to/petsc
export PETSC_ARCH=arch-mumps-opt
export SLEPC_DIR=/absolute/path/to/slepc
```

For prefix-installed PETSc/SLEPc, set `PETSC_DIR` and `SLEPC_DIR` to their
installation prefixes and leave `PETSC_ARCH` unset. When none of these variables
is set, the release scripts fall back to `pkg-config`.

If a system compiler is upgraded after PETSc was configured, regenerate its
configuration with PETSc's recorded `reconfigure-$PETSC_ARCH.py` script before
publishing. The helper builder also adds Homebrew's stable current-GCC library
directories during application linking, but reconfiguration is the durable
maintenance action. See <https://petsc.org/release/install/multibuild/>.

## 5. Validate and build the native helpers

From the FEEC--GMRF repository root, with the environment above active:

```text
bash scripts/check-publication-prerequisites.sh
bash scripts/build-petsc-helpers.sh
```

The scripts print the selected directories. Explicit environment variables take
precedence; `pkg-config` is fallback-only. The check reads `petscconf.h` from the
selected configuration and refuses to continue unless MUMPS is enabled.

Run the native integration gates:

```text
cargo test --release --manifest-path feec/Cargo.toml -p formoniq \
  --features parent-fixture-tests
cargo test --release -p feg-infer --features external-solver-tests
cargo test --release -p feg-case-studies --lib --features external-reference-tests \
  sphere_sparse_anchor_kernel_validation::tests
```

The generated executables and solver scratch directories are ignored by Git.
Remove them with:

```text
make -C feec/petsc-solver \
  PETSC_DIR="$PETSC_DIR" PETSC_ARCH="${PETSC_ARCH:-}" SLEPC_DIR="$SLEPC_DIR" clean
```

## 6. Run an external-solver example

The cube Hodge--Laplacian smoke profile exercises the compiled PETSc helper:

```text
cargo run --release -p feg-cli --bin feg-study -- \
  run hodge-laplacian/cube --profile smoke --output out/hodge-cube
cargo run --release -p feg-cli --bin feg-study -- \
  verify out/hodge-cube --against smoke
```

For all maintained smoke profiles, use:

```text
bash scripts/run-smoke-studies.sh out/smoke
```

The 15 immutable submitted-result profiles are intentionally separate because
they are expensive:

```text
bash scripts/run-publication-regressions.sh out/thesis-submitted
```

Do not treat a missing native prerequisite as a skipped pass. Both runners stop
on errors, and publication results should be accepted only from a clean checkout
whose generated manifests report the intended repository commits.
