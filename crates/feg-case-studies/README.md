# FEG Case Studies

This crate contains maintained thesis case-study workflows. For publication
and report reproducibility, treat
`feg_case_studies::publication` as the supported API surface.

Runnable report workflows are invoked through the registry-driven `feg-study`
CLI. Exploratory and historical entrypoints live in `feg-experiments`; this
crate does not use Cargo examples as an alternate scientific implementation
surface.

## Publication-supported workflows

- Chapter 7 validation: cube mass-inverse variance, Matérn trace
  normalization, Matérn functional convergence, sphere branch observables, and
  torus 1-form PDE workflows.
- Chapter 8 electromagnetic UQ: magnetic physical calibration, magnetic prior
  UQ comparison, annular H-formulation, and exact toroidal B/source workflows.

Reusable FEEC/GMRF logic should live in `feg-infer`, `feg-gp`, or `gmrf-core`.
These case-study modules should only orchestrate geometry, configuration,
reporting, and thesis-specific artifact generation.

Direct use of public `feg-infer` and `gmrf-core` APIs is intentional when a
study needs specialist diagnostics or lower-level control. The root
`feec-gmrf` crate remains the supported API for external users; the ownership
rule here is that studies call canonical lower-layer implementations rather
than reproducing them locally.

## Experimental workflows

Root-level modules gated by the `experimental` feature are historical or
exploratory and are re-exported through `feg-experiments`. They are not part of
the publication guarantee.

## Test Tiers

The default test suite is intended to be fast and deterministic:

```sh
rtk cargo test -p feg-case-studies --release
```

Generated-mesh, nonlinear, publication-scale, output-writing, and large
workflow regression tests are opt-in:

```sh
rtk cargo test -p feg-case-studies --release --features heavy-tests --lib -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_team13 -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_toroidal -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_planar_holes -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_magnetic_uq -- --test-threads=1
```

Tests that need external reference tools or large reference artifacts use the
external tier:

```sh
rtk cargo test -p feg-case-studies --release --features external-reference-tests --test external_references -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features external-reference-tests --lib -- --test-threads=1
```

Most heavy assertions remain as feature-gated unit tests beside the workflow
code so they can reuse private setup helpers without duplicating case-study
logic. New workflow-scale tests should use `#[cfg(feature = "heavy-tests")]`;
tests requiring PETSc, NGSolve exports, or other external reference artifacts
should use `#[cfg(feature = "external-reference-tests")]`.
