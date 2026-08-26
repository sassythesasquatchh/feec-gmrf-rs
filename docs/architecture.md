# Architecture and mathematical ownership

This document is a design reference for maintainers and advanced contributors.
It records dependency direction, mathematical ownership, and the operator
constructions that must remain consistent across crates.

The package follows the direction

```text
FEEC assembly  --->  feec-gmrf composition  --->  GMRF algebra/solvers
                         |
                         v
                   case-study workflows
```

Dependencies flow from the integration workspace to the two standalone
submodules; neither submodule depends on the integration workspace. The GMRF
sister repository is mounted at `gmrf-rs/`, while its public Rust package
remains named `gmrf-core`.

## Ownership

| Mathematical object or operation | Owner |
|---|---|
| topology, degrees of freedom, quadrature, `D_k`, `M_k`, weak Hodge–Laplacian, reconstruction, boundary reduction | `feec` |
| sparse precision storage and factorization, `H^T R^-1 H`, conditioning, KKT constraints, covariance actions, sampling, variance estimators | `gmrf-core` |
| Matérn recurrence, Hodge branch composition, model construction, time discretisation, Gauss–Newton/Laplace orchestration, physical operator chains | `feec-gmrf` / `feg-infer` |
| geometry, material constants, measurement selection, metrics, artifact selection | `feg-case-studies` |

Applications build models through the public root crate. `feg-core` contains
integration-owned transport and statistical-model contracts. FEEC-owned
deterministic contracts, including `DofLayout`, live in `formoniq` and enter the
integration layer through explicit adapters.

## API and implementation crates

`gmrf-core` is the root package of the standalone `gmrf-rs` repository, and
`feg-infer` is an implementation crate in this repository. The root package
delegates Gaussian algebra to `gmrf-core` and FEEC-specific inference adapters
to `feg-infer`.

The root API carries the downstream stability guarantee. Maintained case
studies use it for generic Gaussian model composition, conditioning, sampling,
variance estimation, physical normalization, and predictive diagnostics.
Studies may use public `gmrf-core` and `feg-infer` APIs for specialist
diagnostics and FEEC assembly. Reusable operations discovered in a study move
to the lower layer that owns the corresponding mathematics. The current
migration scope and remaining gaps are recorded in
[`api-migration.md`](api-migration.md).

The CLI depends on the case-study registry and, when enabled, the experiments
registry. Scientific work is carried out by the registered workflow.

## Matérn construction

For every supported form degree, FEEC assembles a mass matrix `M_k` and weak
Hodge–Laplacian `L_k`. The integration layer forms

```text
A_k = kappa^2 M_k + L_k
Q_1 = tau^2 A_k
Q_2 = tau^2 A_k M_k^-1 A_k
Q_3 = tau^2 A_k M_k^-1 A_k M_k^-1 A_k.
```

FEEC assembly and the selected mass-inverse policy carry the form-degree
dependence. `prior::matern_recurrence` implements the shared recurrence.

## Physical quantities

Physical outputs are `LinearMap` compositions. In particular, magnetic flux
density is always constructed as

```text
A  --D1-->  magnetic 2-cochain  --Hodge/reconstruction-->  physical B.
```

Calibration, normalization, observation uncertainty, and reporting all use
this operator chain.

Physical-RMS calibration is performed by
`calibrate_prior_to_physical_rms(prior, map, target)` or its estimator-aware and
weighted variants. They evaluate the generic GMRF transformed covariance for
the supplied composed `LinearMap`, then rescale the precision. Exact and
Hutchinson trace routes share the same calibration recurrence.

## Repeated designs and uncertainty

`GaussianPrior::factor` caches a sparse factor for repeated prior uncertainty
and sampling. `LinearGaussianModelBuilder::prepare` fixes a prior, observation
operators, noise models, constraints, and derived quantities, then reuses one
posterior factor while observation values change. Exact, Monte Carlo,
Hutchinson, and dimension-switched automatic variance policies are exposed by
the root `VarianceMethod` contract.

Heteroscedastic independent observations use `LinearObservation::from_rows` or
`GaussianNoise::independent_variances`. Canonical Gaussian predictive
diagnostics live in `gmrf-core` and are re-exported by the root package.

## Essential-boundary elimination

Homogeneous and prescribed essential conditions share
`EssentialBoundaryElimination`. For active coefficients `z`, the complete
cochain is represented as

```text
x = P z + g,
```

where `P` inserts active coefficients and `g` contains prescribed values. FEEC
operators are reduced before the Matérn mass-inverse recurrence. The parent
facade then applies the same elimination to every model component:

```text
H x + b  ->  (H P) z + (b + H g)
C x = d  ->  (C P) z = d - C g
G x + c  ->  (G P) z + (c + G g)
r(x)     ->  r(P z + g), with Jacobian J(P z + g) P.
```

Posterior computation remains in active coordinates. `cochain_mean`,
`cochain_variances`, and `sample_cochain` lift results to the complete FEEC
ordering; prescribed coefficients have their requested values and exactly zero
variance. The GMRF layer receives the resulting active-coordinate problem.

`GaussianPrior::condition_on_essential_boundary` computes the exact conditional
distribution of an assembled Gaussian precision. Matérn models use
`MaternPriorBuilder::essential_boundary_conditions`, which reduces the FEEC
space before applying the recurrence. Conditioning a full alpha-2 or alpha-3
precision generally produces a different reduced precision.
