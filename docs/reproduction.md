# Reproducing the numerical studies

The `feg-study` command provides named configurations for the maintained
numerical studies. A run records its resolved parameters, source revisions,
inputs, deterministic seeds, software versions, metrics, and generated
artifacts.

Begin with [Installation and first run](getting-started.md). Use a recursive
checkout so that the FEEC and GMRF revisions recorded by the parent are
available:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf-rs.git
cd feec-gmrf-rs
```

## Discovering studies

List the registry and inspect one study:

```text
cargo run --release -p feg-cli --bin feg-study -- list
cargo run --release -p feg-cli --bin feg-study -- describe STUDY_ID
```

The description includes available profiles, accepted custom parameters,
required input files, and external executables.

## Profiles

Two profile families are included:

- `smoke` uses a small deterministic problem for checking the complete
  execution and reporting path;
- `thesis-submitted` uses the mesh levels, observations, estimator sizes, and
  other parameters associated with the submitted thesis results.

Run and verify a profile with:

```text
cargo run --release -p feg-cli --bin feg-study -- \
  run STUDY_ID --profile smoke --output out/STUDY_ID
cargo run --release -p feg-cli --bin feg-study -- \
  verify out/STUDY_ID --against smoke
```

Verification compares the run manifest with the requested profile and checks
the recorded input and artifact inventory. A dirty source tree is recorded;
verification of a `thesis-submitted` profile requires a clean tree.

The complete smoke collection can be run with:

```text
bash scripts/run-smoke-studies.sh out/smoke
```

Some full-resolution studies take substantially longer than the smoke
profiles.

## Custom configurations

A custom TOML file starts from a named profile and overrides parameters accepted
by that study:

```toml
schema = "feg-study-custom-v1"
study_id = "matern/scalar"
base_profile = "smoke"
dimension = 2
range_cells = 3
level = 12
```

Run and verify it with:

```text
cargo run --release -p feg-cli --bin feg-study -- \
  run matern/scalar --config research.toml --output out/research
cargo run --release -p feg-cli --bin feg-study -- \
  verify out/research --against custom
```

Unknown keys, duplicate keys, incompatible value types, and a mismatched
`study_id` are errors. `feg-study describe` lists the accepted keys for each
study.

## Recorded run information

Each output directory contains enough metadata to identify the computation:

- resolved profile or custom configuration;
- command line;
- parent, FEEC, and GMRF commit IDs and dirty status;
- Rust and external-tool versions;
- deterministic random seeds;
- paths and hashes of declared scientific inputs;
- metrics and artifact inventory.

This information is scientific run provenance: it describes how a numerical
result was obtained. It does not describe repository import or publication
history.

## External tools

Many studies use only the Rust sparse solver and checked-in meshes. Depending
on the problem, a descriptor may require:

- Gmsh for runtime mesh generation;
- PETSc with MUMPS for large sparse direct solves;
- SLEPc for eigenvalue and harmonic-basis calculations;
- NGSolve for an optional independent reference comparison.

Missing declared prerequisites cause the run to fail with a diagnostic. The
PETSc/SLEPc build and environment variables are described in
[Installation and first run](getting-started.md).

## Scientific inputs

Checked-in geometry definitions and mesh fixtures are listed in
[Scientific input data](assets.md). Their fixed entity ordering and physical
tags are part of a reproducible discretization, so a generated mesh should not
be substituted without recording the change.
