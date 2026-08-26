# Reproduction guide

This guide is for readers reproducing the thesis studies. It covers the pinned
checkout, study profiles, provenance records, and external prerequisites used by
the `feg-study` workflow.

For operating-system prerequisites, a source build of PETSc with MUMPS, helper
compilation, and a first verified run, begin with
[`getting-started.md`](getting-started.md).

Use a recursive checkout so the exact FEEC and GMRF commits pinned by the parent
are present:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf-rs
cd feec-gmrf-rs
```

Build the three manifests separately in release mode. The parent declares the
two submodule directories as workspace exclusions, while Cargo may still
promote in-tree path dependencies into the root workspace graph.

```text
cargo test --release --workspace
cargo test --release --workspace --manifest-path feec/Cargo.toml
cargo test --release --workspace --manifest-path gmrf-rs/Cargo.toml
```

## Profiles

- `smoke` is a cheap deterministic configuration for CI and installation
  checks.
- `thesis-submitted` is immutable and matches the submitted report settings.
  Magnetic calibration uses levels 2–8 and 512 Hutchinson probes; the sphere
  observable study includes refinement level 6.
- future publication profiles are immutable and publication-named.
- custom research configurations are recorded with a `custom` verification
  status.

Use `--config` for a strict custom research configuration. The file identifies
the study and the immutable profile whose defaults it overrides;
`feg-study describe <study-id>` lists the accepted study-specific keys. Unknown
keys and mismatched study IDs are errors.

```toml
schema = "feg-study-custom-v1"
study_id = "matern/scalar"
base_profile = "smoke"
dimension = 2
range_cells = 3
level = 12
```

```text
feg-study run matern/scalar --config research.toml --output out/research
feg-study verify out/research --against custom
```

Every run directory contains the resolved configuration, command line, root and
submodule commits, dirty status, tool versions, deterministic seeds, metrics,
input hashes, and artifact inventory. Verification refuses to certify a dirty
`thesis-submitted` run.

Some electromagnetic workflows require Gmsh, PETSc with MUMPS, or SLEPc.
Optional publication-reference comparisons additionally require NGSolve. Their
registry descriptors declare runtime prerequisites; a missing prerequisite is
an error, never a skipped successful run. The manual
publication workflow builds the FEEC helper programs with
`scripts/build-petsc-helpers.sh` and runs the gated
`feg-infer` and external-reference regression tests before executing the 15
immutable profiles.

The submitted report repository at commit `115890e` is the initial capability
and reference-results guide. It is not modified by the packaging workflow.

The checked-in geometry definitions and mesh fixtures used by maintained
profiles are catalogued in [`assets.md`](assets.md).
