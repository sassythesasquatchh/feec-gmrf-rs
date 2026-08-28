# Installation and first run

The core library and its introductory examples require Rust and a recursive Git
checkout. PETSc, SLEPc, MPI, MUMPS, Gmsh, and NGSolve are needed only by the
workflows identified below.

## Core Rust installation

Install Rust 1.80 or newer with [Rustup](https://rustup.rs/):

```text
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustup default stable
rustc --version
cargo --version
```

Clone the repository and initialize its FEEC and GMRF submodules:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf-rs.git
cd feec-gmrf-rs
git submodule status --recursive
```

If an existing clone is missing a submodule, initialize it with:

```text
git submodule update --init --recursive
```

Build the principal workspace:

```text
cargo build --release --locked --workspace --exclude feg-experiments
```

Run the two introductory examples:

```text
cargo run --release --locked --example minimal_0form
cargo run --release --locked --example em_1form_uq
```

Both examples use the in-process sparse solver. They write CSV and VTU output
under `out/`, which is ignored by Git.

Run the portable tests for all three workspaces:

```text
cargo test --release --locked --workspace --exclude feg-experiments --all-targets
cargo test --release --workspace --all-targets --manifest-path feec/Cargo.toml
cargo test --release --locked --all-targets --manifest-path gmrf-rs/Cargo.toml
cargo check --release --locked -p feg-experiments --all-targets
```

The parent manifest excludes the standalone FEEC and GMRF workspaces, so their
manifest checks are listed separately.

## Study command

The `feg-study` command lists and runs maintained numerical studies:

```text
cargo run --release -p feg-cli --bin feg-study -- list
cargo run --release -p feg-cli --bin feg-study -- describe matern/scalar
cargo run --release -p feg-cli --bin feg-study -- \
  run matern/scalar --profile smoke --output out/matern-scalar
cargo run --release -p feg-cli --bin feg-study -- \
  verify out/matern-scalar --against smoke
```

Each study descriptor reports the tools and data it requires. Scalar Matérn
studies and many topology studies use only Rust. Electromagnetic benchmarks
may additionally require Gmsh or the native solver stack.

See [Study reproduction](reproduction.md) for profiles, custom
configurations, and run metadata.

## Optional native solver stack

The source-build procedure below has been used on Apple Silicon macOS 15 and
Ubuntu 24.04 x86-64. It installs PETSc and SLEPc under the checkout so that a
system installation cannot silently select a different scalar type, MPI
implementation, or sparse solver.

### Operating-system packages

On macOS, install the command-line developer tools and Homebrew packages:

```text
xcode-select --install
brew update
brew install gcc cmake pkg-config gmsh
```

On Ubuntu 24.04:

```text
sudo apt-get update
sudo apt-get install --yes \
  build-essential gfortran flex git curl ca-certificates python3 cmake pkg-config \
  zlib1g-dev libfontconfig1-dev gmsh
```

### Build PETSc and SLEPc

Clone the selected releases:

```text
mkdir -p .native
git clone --depth 1 --branch v3.25.3 \
  https://gitlab.com/petsc/petsc.git .native/petsc
git clone --depth 1 --branch v3.25.1 \
  https://gitlab.com/slepc/slepc.git .native/slepc
```

Define their locations:

```text
repository_root="$PWD"
export PETSC_DIR="$repository_root/.native/petsc"
export PETSC_ARCH=arch-feec-mumps-opt
export SLEPC_DIR="$repository_root/.native/slepc"
```

On macOS:

```text
export CC=clang
export CXX=clang++
export FC=gfortran
```

On Ubuntu:

```text
export CC=gcc
export CXX=g++
export FC=gfortran
```

Configure an optimized real-scalar PETSc with the MPI and sparse direct-solver
dependencies required by the external workflows:

```text
cd "$PETSC_DIR"
./configure \
  PETSC_ARCH="$PETSC_ARCH" \
  --with-cc="$CC" \
  --with-cxx="$CXX" \
  --with-fc="$FC" \
  --with-debugging=0 \
  --with-scalar-type=real \
  --download-bison \
  --download-fblaslapack \
  --download-metis \
  --download-mumps \
  --download-openmpi \
  --download-parmetis \
  --download-ptscotch \
  --download-scalapack
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all check
```

Build SLEPc against that PETSc configuration:

```text
cd "$SLEPC_DIR"
./configure
make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check
cd "$repository_root"
```

For a later shell, restore:

```text
export PETSC_DIR="$PWD/.native/petsc"
export PETSC_ARCH=arch-feec-mumps-opt
export SLEPC_DIR="$PWD/.native/slepc"
export PATH="$PETSC_DIR/$PETSC_ARCH/bin:$PATH"
```

Validate the selected installation and compile the FEEC helper programs:

```text
bash scripts/check-native-prerequisites.sh
bash scripts/build-petsc-helpers.sh
```

The diagnostic checks that PETSc uses real scalars, that MUMPS is available,
and that SLEPc is configured against the same PETSc installation.

### Native integration tests

With the variables above set:

```text
cargo test --release --manifest-path feec/Cargo.toml -p formoniq \
  --features parent-fixture-tests
cargo test --release -p feg-gp --features external-solvers \
  --test hodge_laplace_integration
cargo test --release -p feg-infer --features external-solver-tests
cargo test --release -p feg-case-studies --lib \
  --features external-reference-tests \
  sphere_sparse_anchor_kernel_validation::tests
```

NGSolve is not required by the core library or introductory examples. It is
used only for optional comparisons with independently generated reference
solutions. `scripts/check-publication-prerequisites.sh` checks that extended
environment.

## Troubleshooting

### A submodule is missing or at the wrong commit

```text
git submodule update --init --recursive
git submodule status --recursive
```

A leading `-` means the submodule is uninitialized. A leading `+` means its
working tree is not at the commit recorded by the parent.

### The wrong PETSc is selected

Explicit `PETSC_DIR`, `PETSC_ARCH`, and `SLEPC_DIR` values take precedence.
Clear stale values before using the checkout-local installation:

```text
unset PETSC_DIR PETSC_ARCH SLEPC_DIR
export PETSC_DIR="$PWD/.native/petsc"
export PETSC_ARCH=arch-feec-mumps-opt
export SLEPC_DIR="$PWD/.native/slepc"
bash scripts/check-native-prerequisites.sh
```

### PETSc was built without MUMPS

Reconfigure PETSc with `--download-mumps` and its MPI/ScaLAPACK dependencies.
The helper build treats a missing MUMPS configuration as an error.

### SLEPc reports a PETSc mismatch

Reconfigure SLEPc after selecting the intended `PETSC_DIR` and `PETSC_ARCH`.
Both packages must use the same PETSc build.

### Gmsh is unavailable

Install Gmsh with `brew install gmsh` or `apt-get install gmsh`. Studies that
use checked-in meshes can still run; studies that generate benchmark meshes
will report the missing executable.

### Generated files appear in Git status

Build products, `.native/`, and `out/` are ignored. If tracked files appear
modified, inspect them before deleting anything:

```text
git status --short
git submodule foreach --recursive git status --short
```
