# Changelog

This file records user-visible capabilities and corrections that change the
mathematical or numerical result.

## 0.1.0 — unreleased

### Field models and inference

- Added Matérn precision construction for 0-, 1-, 2-, and 3-form FEEC spaces,
  with integer orders alpha 1, 2, and 3 and explicit mass-inverse policies.
- Added potential-spectrum and form-spectrum Hodge-decomposed 1-form priors
  with exact, coexact, and harmonic branches.
- Added sparse linear Gaussian observations, heteroscedastic and correlated
  noise models, hard linear constraints, derived quantities, and predictive
  diagnostics.
- Added homogeneous and prescribed essential-boundary elimination. FEEC
  operators are reduced before the Matérn recurrence, affine offsets are folded
  into every model term, and posterior results lift back to the complete
  cochain with zero variance at prescribed values.
- Added reduced linear PDE models, weak-residual conditioning, PDE-induced
  priors, nonlinear Gauss–Newton/Laplace inference, and spatiotemporal
  precision construction.
- Added block state/source uncertainty models with latent or analytically
  collapsed uncertain inputs.

### Sparse computation and uncertainty

- Added sparse matrix composition through the `LinearMap` API.
- Added reusable prior factors and prepared linear Gaussian designs for
  repeated observation values.
- Added exact, selected-inverse, Monte Carlo, Hutchinson, local
  Rao–Blackwellised, and dimension-dependent uncertainty methods.
- Added transformed covariance actions, exact small-output covariance blocks,
  weighted covariance traces, constrained variance corrections, and
  factorization diagnostics.
- Added deterministic stochastic-estimator batching, standard-error reporting,
  and explicit variance stabilization metadata.
- Added physical-RMS calibration through transformed covariance actions.

### Physical fields and scientific output

- Added orientation-aware magnetic maps following
  `A -> D1 A -> reconstructed B`, along with boundary flux and
  volume-average maps.
- Added posterior reports containing fields, quantities of interest,
  predictive diagnostics, typed CSV tables, and FEEC or top-cell VTU fields.
- Added `minimal_0form` and `em_1form_uq` as introductory scalar and
  electromagnetic workflows.
- Added named, reproducible study profiles and strict custom configurations
  through the `feg-study` command.
- Separated maintained case studies from exploratory numerical programs.

### Numerical corrections

- Corrected the AR(1) precision endpoints and singleton case in `gmrf-core`.
  Exact coefficient tests cover every matrix position.
- Corrected cube and torus `H_d` error output to use the graph norm
  `sqrt(||u-u_h||^2 + ||d(u-u_h)||^2)` rather than the exterior-derivative
  seminorm alone.
- Restored the scalar Matérn diagnostic to one central vertex and 21
  nonnegative coordinate-axis lags on the level-64 mesh, avoiding an unintended
  dense covariance over all interior vertices.
- Restored the four-dimensional marginal-variance study to the scalar alpha-2
  and alpha-3 point, line-average, and area-average configurations.
- Restored the toroidal magnetic study's submitted 12/24 training and held-out
  flux-row split. The greedy source-design method remains available for new
  experiments.
- Restored the torus residual-weight study to the posterior
  residual-precision sweep at kappa 4 and moved its resolution check to the
  torus workflow.
