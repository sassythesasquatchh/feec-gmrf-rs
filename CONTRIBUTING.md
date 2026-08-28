# Contributing

Changes should preserve the mathematical separation between discretization,
statistical modelling, and Gaussian linear algebra.

## Where changes are made

- `feec/` implements mesh topology, degrees of freedom, quadrature, incidence
  and mass matrices, Hodge operators, boundary reduction, PDE residuals and
  Jacobians, and physical reconstruction.
- `gmrf-rs/` implements sparse precision storage, factorization and solves,
  Gaussian conditioning, equality constraints, covariance actions, sampling,
  and uncertainty estimators.
- The parent workspace composes those operators into Matérn, PDE, nonlinear,
  physical, and spatiotemporal statistical models.
- `crates/feg-case-studies/` supplies scientific geometries, material values,
  measurements, study configurations, and reported metrics.

The FEEC and GMRF repositories are independently buildable and do not depend on
the parent integration workspace or on each other. Shared integration
contracts are kept small and live in `feg-core`.

## Mathematical invariants

Before changing a model, identify the equation represented by every matrix and
the coordinate space on which it acts. In particular:

- use the `gmrf-core` observation, constraint, sparse-row, sampling, covariance,
  and uncertainty algorithms rather than defining application-specific
  alternatives;
- reduce essential boundary coefficients before applying an alpha-2 or
  alpha-3 Matérn recurrence;
- construct magnetic uncertainty through the explicit
  `A -> D1 A -> B` FEEC map;
- retain residual and Jacobian consistency in nonlinear models;
- document symmetry, definiteness, gauge, nullspace, and mass-inverse
  assumptions;
- report numerical limitations and solver fallbacks explicitly.

A numerical discrepancy should be explained by the mathematics, discretization,
solver, or estimator. Do not conceal it by changing tolerances or reference
values.

## Tests

New functionality requires focused unit tests. A new connection between
workspaces or model layers also requires an integration test. Commands are run
in release mode unless a debug build is needed for diagnosis.

Parent workspace:

```text
cargo fmt --all --check
cargo check --release --workspace --exclude feg-experiments --all-targets
cargo test --release --workspace --exclude feg-experiments --all-targets
cargo clippy --release --workspace --exclude feg-experiments --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release \
  --workspace --exclude feg-experiments --no-deps

cargo check --release -p feg-experiments --all-targets
cargo test --release -p feg-experiments --all-targets
```

FEEC workspace:

```text
cargo fmt --all --check --manifest-path feec/Cargo.toml
cargo test --release --workspace --manifest-path feec/Cargo.toml
cargo clippy --release --workspace --all-targets \
  --manifest-path feec/Cargo.toml -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --workspace \
  --manifest-path feec/Cargo.toml --no-deps
```

GMRF workspace:

```text
cargo fmt --all --check --manifest-path gmrf-rs/Cargo.toml
cargo test --release --workspace --manifest-path gmrf-rs/Cargo.toml
cargo clippy --release --workspace --all-targets \
  --manifest-path gmrf-rs/Cargo.toml -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release \
  --manifest-path gmrf-rs/Cargo.toml --no-deps
```

Tests that execute PETSc or SLEPc helper programs are separate because those
packages are optional. With a matching MUMPS-enabled PETSc/SLEPc environment:

```text
bash scripts/check-publication-prerequisites.sh
bash scripts/build-petsc-helpers.sh
cargo test --release --manifest-path feec/Cargo.toml -p formoniq \
  --features parent-fixture-tests
cargo test --release -p feg-infer --features external-solver-tests
cargo test --release -p feg-case-studies --lib \
  --features external-reference-tests \
  sphere_sparse_anchor_kernel_validation::tests
```

The source-build procedure is documented in
[Installation and first run](docs/getting-started.md).
