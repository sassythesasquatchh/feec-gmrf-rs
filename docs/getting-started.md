# Clean installation and first run

This is the supported installation path for a new Apple Silicon Mac running
macOS 15 or newer, or an x86-64 computer running Ubuntu 24.04. It starts from a
recursive clone of `main` and builds PETSc, MUMPS, MPI, and SLEPc inside the
checkout. It does not use a system PETSc and does not require NGSolve.

Commands in this guide are run from a terminal. Native solver compilation can
take a substantial amount of time and disk space; the measured values from the
release clean-room runs are recorded in
[`clean-install-validation.md`](clean-install-validation.md).

## 1. Install operating-system prerequisites

### Apple Silicon macOS

Install Apple's command-line developer tools if they are not already present:

```text
xcode-select --install
```

Install [Homebrew](https://brew.sh/) if necessary, then install the required
Fortran compiler, build tools, and mesh generator:

```text
brew update
brew install gcc cmake pkg-config gmsh
```

Install Rust with Rustup and start a new shell, or source Cargo's environment
file as shown:

```text
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustup default stable
```

The supported macOS path uses Apple Clang for C/C++ and Homebrew `gfortran` for
the Fortran packages required by MUMPS.

### Ubuntu 24.04 x86-64

Install the system compiler and build prerequisites:

```text
sudo apt-get update
sudo apt-get install --yes \
  build-essential gfortran git curl ca-certificates python3 cmake pkg-config gmsh
```

Install Rust with Rustup:

```text
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustup default stable
```

On either platform, confirm that Cargo is at least version 1.80:

```text
rustc --version
cargo --version
```

## 2. Clone the complete repository

Use the public HTTPS URL and initialize both pinned submodules during cloning:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf-rs.git
cd feec-gmrf-rs
git submodule status --recursive
```

The last command must print one FEEC commit and one GMRF commit without a
leading `-` or `+`. If the repository was cloned without `--recursive`, repair
it with:

```text
git submodule update --init --recursive
```

Record the exact source being installed:

```text
git rev-parse HEAD
git -C feec rev-parse HEAD
git -C gmrf-rs rev-parse HEAD
```

## 3. Build the pinned native solver stack

The canonical installation is deliberately local to this checkout:

```text
.native/petsc
.native/petsc/arch-feec-mumps-opt
.native/slepc
```

The directories are ignored by Git. Start with no PETSc/SLEPc variables so an
old shell configuration cannot influence the build:

```text
unset PETSC_DIR PETSC_ARCH SLEPC_DIR
mkdir -p .native
git clone --depth 1 --branch v3.25.3 \
  https://gitlab.com/petsc/petsc.git .native/petsc
git clone --depth 1 --branch v3.25.1 \
  https://gitlab.com/slepc/slepc.git .native/slepc
```

Set the common build locations:

```text
repository_root="$PWD"
export PETSC_DIR="$repository_root/.native/petsc"
export PETSC_ARCH=arch-feec-mumps-opt
export SLEPC_DIR="$repository_root/.native/slepc"
```

On macOS, select Apple Clang and Homebrew Fortran:

```text
export CC=clang
export CXX=clang++
export FC=gfortran
```

On Ubuntu, select the GNU compiler suite instead:

```text
export CC=gcc
export CXX=g++
export FC=gfortran
```

Configure an optimized real-scalar PETSc. The external-package flags force the
required MPI and sparse solver stack to be built locally rather than discovered
from Homebrew or apt:

```text
cd "$PETSC_DIR"
./configure \
  PETSC_ARCH="$PETSC_ARCH" \
  --with-cc="$CC" \
  --with-cxx="$CXX" \
  --with-fc="$FC" \
  --with-debugging=0 \
  --with-scalar-type=real \
  --download-fblaslapack \
  --download-metis \
  --download-mumps \
  --download-openmpi \
  --download-parmetis \
  --download-ptscotch \
  --download-scalapack
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all check
```

Build the matching SLEPc 3.25 release against that PETSc configuration:

```text
cd "$SLEPC_DIR"
./configure
make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check
cd "$repository_root"
```

For a new terminal opened after the build, restore the three variables before
running native maintenance commands:

```text
export PETSC_DIR="$PWD/.native/petsc"
export PETSC_ARCH=arch-feec-mumps-opt
export SLEPC_DIR="$PWD/.native/slepc"
export PATH="$PETSC_DIR/$PETSC_ARCH/bin:$PATH"
```

Repository scripts automatically prefer this complete local layout over
`pkg-config`. Explicit environment variables still take first priority.

## 4. Validate the native stack and compile the helpers

Run the core native diagnostic; it checks the platform tools, the pinned local
tags, SLEPc configuration, MPI commands, and `PETSC_HAVE_MUMPS` in the selected
PETSc header:

```text
bash scripts/check-native-prerequisites.sh
```

Successful output ends with:

```text
native prerequisites are available
```

Build the three FEEC PETSc/SLEPc helper executables:

```text
bash scripts/build-petsc-helpers.sh
```

Successful output ends with:

```text
PETSc/SLEPc helpers are ready in .../feec/petsc-solver
```

NGSolve is intentionally not part of this installation. The separate
`check-publication-prerequisites.sh` command adds NGSolve only for optional
external-reference reproduction.

## 5. Build and test every Rust workspace

All supported checks use optimized builds:

```text
cargo build --release --locked --workspace --exclude feg-experiments
cargo test --release --locked --workspace --exclude feg-experiments --all-targets

cargo test --release --workspace --all-targets --manifest-path feec/Cargo.toml
cargo test --release --all-targets --locked --manifest-path gmrf-rs/Cargo.toml

cargo check --release --locked -p feg-experiments --all-targets
cargo test --release --locked -p feg-experiments --all-targets
bash scripts/test-external-consumer.sh
```

Cargo may promote in-tree path dependencies into the parent workspace graph
even though `feec/` and `gmrf-rs/` are exclusions. The explicit manifest checks
above therefore remain required.

Run the native integration gates:

```text
cargo test --release --manifest-path feec/Cargo.toml -p formoniq \
  --features parent-fixture-tests
cargo test --release -p feg-infer --features external-solver-tests
cargo test --release -p feg-case-studies --lib --features external-reference-tests \
  sphere_sparse_anchor_kernel_validation::tests
```

No test in this section may be counted as passed when it reports that a native
prerequisite was skipped.

## 6. Run the introductory examples

Run both public API examples from the repository root:

```text
cargo run --release --locked --example minimal_0form
cargo run --release --locked --example em_1form_uq
```

These examples write only to ignored output locations and require no NGSolve
reference data.

## 7. Run and verify all maintained smoke studies

The installation acceptance suite contains 15 deterministic smoke profiles:

```text
bash scripts/run-smoke-studies.sh out/smoke
```

Each study must print both `completed` and `verified`; the command must exit
with status zero. The expensive `thesis-submitted` profiles are a separate
publication regression and are not part of installation acceptance.

To rerun one failed study after correcting a prerequisite, use the commands
printed by the runner, or inspect the stable command surface with:

```text
cargo run --release -p feg-cli --bin feg-study -- list
cargo run --release -p feg-cli --bin feg-study -- describe STUDY_ID
```

## 8. Confirm the checkout remains clean

Native sources, helper executables, Cargo targets, and study output are ignored.
The final recursive status should contain no tracked changes:

```text
git status --short
git submodule foreach --recursive git status --short
```

## Troubleshooting

### The local clone is partial

If either `.native/petsc` or `.native/slepc` exists without its matching build,
the diagnostic deliberately selects the local layout and reports the missing
directory or configuration. Complete the interrupted build, or remove only the
incomplete `.native` solver directories and repeat section 3.

### The wrong PETSc is selected

`check-native-prerequisites.sh` prints its selection and all paths. For the
supported clean installation it must say `repository-local installation`. If it
says `explicit environment`, clear stale values and rerun it:

```text
unset PETSC_DIR PETSC_ARCH SLEPC_DIR
bash scripts/check-native-prerequisites.sh
```

### PETSc lacks MUMPS

The check reads the exact `petscconf.h` it prints and requires
`PETSC_HAVE_MUMPS 1`. Do not substitute a Homebrew or distribution PETSc that
fails this check; build the pinned local configuration in section 3.

### A compiler was upgraded

PETSc records compiler and runtime-library paths at configuration time. For the
repository-local installation, the durable repair is to rebuild both local
solver directories with the current compilers, then rebuild the three helpers.

### A helper cannot load an MPI or Fortran library

First restore the environment block from section 3 and confirm that
`$PETSC_DIR/$PETSC_ARCH/bin` is on `PATH`. On macOS inspect the helper with
`otool -L`; on Ubuntu use `ldd`. A path into a removed compiler installation
means PETSc and SLEPc must be rebuilt rather than patched in place.

### NGSolve is reported missing

Use `check-native-prerequisites.sh` for the supported professor installation.
NGSolve is required only by `check-publication-prerequisites.sh` and optional
external-reference workflows.
