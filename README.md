# FEEC–GMRF

`feec-gmrf` is a Rust library for constructing Gaussian Markov random fields on
finite-element exterior-calculus spaces. It combines FEEC operators with sparse
Gaussian conditioning, constraints, sampling, and uncertainty estimation.

This repository is the software deliverable accompanying Patrick Dowd's
master's thesis. Its reusable API is provided by the root crate, with FEEC and
GMRF implementations pinned as Git submodules. Version 0.1.0 is distributed
through GitHub source releases; the crates are intentionally marked
`publish = false`.

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

Clone the complete repository and run the two introductory workflows in release
mode:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf
cd feec-gmrf
cargo run --release --example minimal_0form
cargo run --release --example em_1form_uq
```

[`minimal_0form`](examples/minimal_0form.rs) introduces scalar Matérn
conditioning, named and ad hoc outputs, and exact and Monte Carlo variance.
[`em_1form_uq`](examples/em_1form_uq.rs) develops a mixed-boundary
electromagnetic problem with physical pushforwards, a weak PDE residual, and
engineering quantities of interest.

## Use as a dependency

Add a released tag or pinned commit to your `Cargo.toml`:

```toml
[dependencies]
feec-gmrf = { git = "https://github.com/sassythesasquatchh/feec-gmrf", tag = "v0.1.0" }
```

Once a prior and its output maps have been assembled, ordinary conditioning uses
the root builder API:

```rust
use feec_gmrf::prelude::*;

fn condition(
    prior: GaussianPrior,
    sensor: LinearMap,
    transect: LinearMap,
    observed_value: f64,
) -> Result<Posterior> {
    let observation = LinearObservation::new(
        sensor,
        vec![observed_value],
        GaussianNoise::standard_deviation(0.05)?,
    )?;

    LinearGaussianModelBuilder::new(prior)
        .observe(observation)?
        .derive(DerivedQuantity::new("transect", transect)?)?
        .condition()
}
```

The example source files show FEEC assembly, boundary treatment, physical maps,
and posterior uncertainty queries in context.

## Features

The default feature set uses explicit sparse matrices. Enable `spectral` for
low-rank and spectral GP APIs, and `external-solvers` for study workflows that
invoke PETSc or SLEPc tools.

## Further documentation

- [Example source](examples/) provides the introductory, runnable workflows.
- [Clean installation and first run](docs/getting-started.md) covers Rust and
  native solver setup from a fresh machine.
- Generate public API documentation with `cargo doc --release --open`.
- [Architecture and mathematical ownership](docs/architecture.md) describes
  dependency direction and shared mathematical invariants.
- [Reproduction guide](docs/reproduction.md) explains study profiles, provenance,
  and the `feg-study` command.
- [Scientific input inventory](docs/assets.md) records retained geometries,
  meshes, and fixtures.
- [Contribution guide](CONTRIBUTING.md) defines code ownership and required checks.
- [Changelog](CHANGELOG.md) summarizes user-visible additions and corrections.
- [Release procedure](docs/release.md) and
  [thesis result validation](docs/thesis-result-validation.md) record the release
  gates and numerical evidence.
- [Source provenance](PROVENANCE.md) records the curated import lineage.

## License and attribution

The parent integration code is MIT licensed. The FEEC submodule retains its
upstream MIT/Apache-2.0 dual licensing and attribution, and the GMRF submodule is
MIT licensed. See [THIRD_PARTY.md](THIRD_PARTY.md) and the lineage documents in
each submodule.
