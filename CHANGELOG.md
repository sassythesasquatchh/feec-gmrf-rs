# Changelog

All notable changes to the supported package are recorded here. Numerical
changes must describe their mathematical cause and may not be hidden by changing
tolerances or reference data.

## 0.1.0 — unreleased

- Added the reusable `feec_gmrf` root façade.
- Added the state-only linear FEEC--GMRF UQ façade, including canonical
  zero-form and electromagnetic one-form conditioning examples.
- Added constrained Monte Carlo uncertainty estimates in `gmrf-core` and
  exposed the FEEC boundary-orientation sign needed by physical weak-form
  assembly.
- Added grade-independent Matérn recurrence construction for alpha 1, 2, and 3.
- Added validated sparse maps, boundary layouts, linear observations,
  constraints, physical pushforwards, posterior uncertainty, and spatiotemporal
  prior construction.
- Made the FEEC and GMRF workspaces independently buildable.
- Added registry-driven maintained-study profiles and provenance manifests.
- Added strict file-backed custom research configurations based on immutable
  publication profiles.
- Centralized physical-RMS calibration through generic transformed GMRF
  covariance actions.
- Exposed exact covariance pushforwards for named derived quantities through
  the supported `Posterior` API, including hard-constraint corrections.
- Added top-level homogeneous and prescribed essential-boundary elimination.
  FEEC Matérn operators are reduced before the mass-inverse recurrence, fixed
  values are folded into all linear and nonlinear model terms, and posterior
  cochain means, variances, and samples are lifted with exact prescribed values.
- Separated publication-supported and experimental workflow surfaces.
- Removed tracked generated solver products and observation caches.
- Removed byte-identical duplicate torus and toroidal-inductor meshes while
  retaining the canonical filenames and unchanged mesh hashes.

### Numerical corrections requiring release review

- Corrected the GMRF AR(1) precision constructor for endpoint and singleton
  processes, with exact coefficient tests for each matrix position.
- Corrected the deterministic cube and torus `H_d` outputs to report the graph
  norm `sqrt(||u-u_h||^2 + ||d(u-u_h)||^2)`. The packaging baseline had
  labelled the exterior-derivative seminorm alone as `H_d`; the corrected
  values restore the submitted tables without changing the underlying solves.
- Restored `matern/scalar` to the submitted Lindgren-style diagnostic: one
  central vertex and 21 nonnegative coordinate-axis lags on the level-64 mesh.
  The initial registry profile incorrectly selected all 35,937 interior
  vertices and attempted to form their dense transformed covariance. The
  report configuration is now a first-class maintained workflow with focused
  profile tests.
- Corrected `matern/marginal-variance-4d` to run the submitted scalar
  alpha-2/alpha-3 point, line-average, and area-average study at levels 4, 8,
  12, and 16. The initial registry entry incorrectly invoked a newer all-form
  study; that workflow remains available under the experimental feature.
- Pinned the immutable `toroidal-b/canonical` profile to the exact submitted
  12/24 training/heldout flux-row split. The reusable greedy source-design
  algorithm remains the default for smoke and custom research runs; the
  publication split is recorded explicitly because one selection is a
  solver-sensitive near-tie.
- Corrected `hodge/torus-residual-weight` to run the submitted posterior
  residual-precision convergence sweep at κ=4 over weights 10^2 through
  10^12 and torus mesh levels 0 through 3. The initial registry entry
  incorrectly invoked a distinct residual-field variance-decomposition
  diagnostic, which remains available as a case-study module.
- Moved the torus resolution-bound check from the cube Hodge--Laplacian runner
  to the torus residual-weight runner where it belongs. The misplaced check
  rejected valid submitted cube refinements above 3 before any solve began;
  focused tests now cover the maintained torus mesh range.

The comparison method, exact profile settings, numerical differences, and
release dispositions are recorded in
[`docs/thesis-result-validation.md`](docs/thesis-result-validation.md).
