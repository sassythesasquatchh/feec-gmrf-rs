# Contributing

Run commands in release mode unless debugging specifically requires otherwise.
Do not introduce dependencies from either submodule back into the parent
workspace.

Before changing scientific code, identify the mathematical owner:

- FEEC assembly, topology, quadrature, reconstruction, and boundary reduction:
  `feec`.
- Gaussian precision algebra, conditioning, constraints, sparse solves,
  sampling, and variance estimation: `gmrf-core` in `gmrf-rs/`.
- model composition, Matérn construction, physical chains, time discretisation,
  and inference orchestration: the root package and `feg-infer`.
- geometry, material values, measurements, and reported metrics:
  `feg-case-studies`.

There must be one canonical implementation for each equation or operator. Tests
may construct fixtures, but must call production Matérn recurrences,
conditioning, KKT, sampling, variance, and pushforward code.

Required checks:

```text
cargo fmt --all --check
cargo check --release --workspace --exclude feg-experiments --all-targets
cargo test --release --workspace --exclude feg-experiments --all-targets
cargo clippy --release --workspace --exclude feg-experiments --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --workspace --exclude feg-experiments --no-deps

cargo check --release -p feg-experiments --all-targets
cargo test --release -p feg-experiments --all-targets

cargo fmt --all --check --manifest-path feec/Cargo.toml
cargo test --release --workspace --manifest-path feec/Cargo.toml
cargo clippy --release --workspace --all-targets --manifest-path feec/Cargo.toml -- -D warnings

cargo fmt --all --check --manifest-path gmrf-rs/Cargo.toml
cargo test --release --workspace --manifest-path gmrf-rs/Cargo.toml
cargo clippy --release --workspace --all-targets --manifest-path gmrf-rs/Cargo.toml -- -D warnings
```

Tests that execute the FEEC PETSc/SLEPc helper programs are intentionally gated
from the portable default suite. The publication environment requires a PETSc
build with MUMPS. Set `PETSC_DIR`, optional `PETSC_ARCH`, and `SLEPC_DIR` for an
explicit installation, then run:

```text
bash scripts/check-publication-prerequisites.sh
bash scripts/build-petsc-helpers.sh
cargo test --release --manifest-path feec/Cargo.toml -p formoniq \
  --features parent-fixture-tests
cargo test --release -p feg-infer --features external-solver-tests
cargo test --release -p feg-case-studies --lib --features external-reference-tests \
  sphere_sparse_anchor_kernel_validation::tests
```

See [`docs/getting-started.md`](docs/getting-started.md) for the clean-install
and source-build procedure.

Every new feature needs focused unit tests; integrations need integration tests.
A study is promoted from `feg-experiments` only after its reusable mathematics
has moved to the owning lower layer and it has smoke and immutable publication
profiles.

## Dependency and ownership rules

- `feec` and `gmrf-rs` are independently buildable submodules and must not
  depend on the parent integration workspace or on each other.
- Shared integration contracts belong in `feg-core`; FEEC discretization and
  assembly remain in `feec`; Gaussian algebra and sparse solves remain in
  `gmrf-core`; model composition and inference orchestration remain in the
  parent workspace.
- Linear conditioning, constraints, sparse-row composition, covariance
  actions, sampling, and variance estimation must use the canonical
  `gmrf-core` implementations rather than case-study copies.
- Physical magnetic outputs used for calibration or uncertainty reporting must
  follow the explicit `A -> D1 A -> B` FEEC pushforward.
- Numerical limitations, fallbacks, and unexplained discrepancies are release
  blockers until they are documented and reviewed.
