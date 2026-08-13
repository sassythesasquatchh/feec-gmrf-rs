# FEEC–GMRF

`feec-gmrf` is a Rust library for constructing Gaussian Markov random fields on
finite-element exterior-calculus spaces. It combines geometry-independent model
composition with explicit sparse FEEC operators and reusable GMRF conditioning,
constraint, sampling, and uncertainty algorithms.

This repository is the software deliverable accompanying Patrick Dowd's
master's thesis. Version 0.1.0 is distributed through GitHub source releases;
the crates are intentionally marked `publish = false`.

## Repository layout

- `feec/` is an attributed thesis fork of Luis Wirth's `formoniq` FEEC library.
- `gmrf-rs/` is the clean-history `gmrf-rs` sister repository with the
  standalone `gmrf-core` crate at its root.
- the root `feec-gmrf` crate is the supported reusable API.
- `crates/feg-case-studies` contains maintained report-backed workflows.
- `crates/feg-experiments` contains default-off exploratory workflows.
- `crates/feg-cli` provides the `feg-study` reproducibility command.

The root manifest lists only the integration crates and declares both submodule
directories as exclusions. Because Cargo can still promote in-tree path
dependencies into the root workspace graph, the release gates also invoke the
FEEC and GMRF manifests independently; each remains independently buildable.

## Quick start

Add the repository as a Git dependency (replace the revision with a released
tag or commit):

```toml
[dependencies]
feec-gmrf = { git = "https://github.com/sassythesasquatchh/feec-gmrf", tag = "v0.1.0" }
```

An operator-level scalar Matérn model needs no case-study code:

```rust
use feec_gmrf::prelude::*;

fn model() -> Result<Posterior> {
    let degree = FormDegree::new(0, 2)?;
    let mass = SparseMat::diagonal(3, 1.0);
    let laplacian = SparseMat::from_rows(3, &[
        vec![(0, 1.0), (1, -1.0)],
        vec![(0, -1.0), (1, 2.0), (2, -1.0)],
        vec![(1, -1.0), (2, 1.0)],
    ]).map_err(FeecGmrfError::Dimension)?;
    let operators = FormOperators::new(degree, 2, mass, laplacian)?;
    let prior = MaternPriorBuilder::from_operators(operators)
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.0, 1.0)?)
        .mass_inverse(MassInversePolicy::Diagonal)
        .build()?;

    let sensor = LinearMap::new(
        SparseMat::from_rows(3, &[vec![(1, 1.0)]])
            .map_err(FeecGmrfError::Dimension)?,
    )?;
    let observation = LinearObservation::new(
        sensor,
        vec![0.25],
        GaussianNoise::variance(1.0e-2)?,
    )?;
    LinearGaussianModelBuilder::new(prior)
        .observe(observation)?
        .condition()
}
```

For FEEC assembly, use `MaternPriorBuilder::from_feec(topology, metric, degree)`.
For physical magnetic output, compose the exterior derivative and flux
reconstruction with `magnetic_field_map`; the API intentionally makes the
`A -> D1 A -> B` chain explicit.

Homogeneous and prescribed essential boundaries use the same top-level
elimination path. Boundary values are folded into observations, constraints,
nonlinear residuals, and derived physical quantities, while inference remains
in active coordinates:

```rust
let boundary = EssentialBoundaryConditions::prescribed(
    vec![0, 2],
    vec![1.0, -0.5],
)?;
let prior = MaternPriorBuilder::from_feec(&topology, &metric, degree)?
    .essential_boundary_conditions(boundary)
    .build()?;

// `sensor` may use the full cochain ordering. Its fixed-value contribution is
// folded into the observation bias by the model builder.
let posterior = LinearGaussianModelBuilder::new(prior)
    .observe(LinearObservation::new(sensor, values, noise)?)?
    .condition()?;
let full_mean = posterior.cochain_mean();
```

The default feature set is the supported explicit sparse-matrix route. Enable
`spectral` for low-rank/spectral GP APIs, and additionally opt into
`external-solvers` for workflows that invoke external PETSc/SLEPc tooling.
PETSc must include MUMPS factorization support. External executables remain
runtime prerequisites and are checked explicitly by the study runner.

## Reproducing studies

```text
cargo run --release -p feg-cli --bin feg-study -- list
cargo run --release -p feg-cli --bin feg-study -- describe matern/trace-normalization
cargo run --release -p feg-cli --bin feg-study -- run matern/trace-normalization \
  --profile smoke --output out/trace-smoke
cargo run --release -p feg-cli --bin feg-study -- verify out/trace-smoke \
  --against smoke
```

See [clean installation and first run](docs/getting-started.md), [the
reproduction guide](docs/reproduction.md), [architecture and
ownership](docs/architecture.md), [scientific input inventory](docs/assets.md),
and [contribution guide](CONTRIBUTING.md). The clean-history import is recorded
in [PROVENANCE.md](PROVENANCE.md).

## License and attribution

The parent integration code is MIT licensed. The FEEC submodule retains its
upstream MIT/Apache-2.0 dual licensing and attribution. The GMRF submodule is
MIT licensed. See [THIRD_PARTY.md](THIRD_PARTY.md) and the lineage documents in
each submodule.
