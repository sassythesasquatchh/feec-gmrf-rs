# Architecture and mathematical ownership

The package follows the direction

```text
FEEC assembly  --->  feec-gmrf composition  --->  GMRF algebra/solvers
                         |
                         v
                   case-study workflows
```

Neither standalone submodule depends on the integration workspace.
The GMRF sister repository is mounted at `gmrf-rs/`; its public Rust package
remains named `gmrf-core`.

## Canonical ownership

| Mathematical object or operation | Owner |
|---|---|
| topology, degrees of freedom, quadrature, `D_k`, `M_k`, weak Hodge–Laplacian, reconstruction, boundary reduction | `feec` |
| sparse precision storage and factorization, `H^T R^-1 H`, conditioning, KKT constraints, covariance actions, sampling, variance estimators | `gmrf-core` |
| Matérn recurrence, Hodge branch composition, model construction, time discretisation, Gauss–Newton/Laplace orchestration, physical operator chains | `feec-gmrf` / `feg-infer` |
| geometry, material constants, measurement selection, metrics, artifact selection | `feg-case-studies` |

The public root crate is the supported downstream boundary. `feg-core` contains
only integration-owned transport and statistical-model contracts. FEEC-owned
deterministic contracts, including `DofLayout`, live directly in `formoniq` and
are consumed through explicit adapters rather than duplicated.

## Dependency edges versus the supported API

`gmrf-core` is the root package of the standalone `gmrf-rs` repository and is a normal,
intentional dependency of the root package. `feg-infer` is an implementation
crate in this repository and is likewise a normal dependency of the root
package. The facade delegates to these crates instead of reproducing their
algorithms.

The supported API statement is about what downstream users can rely on, not
about forbidding dependencies between internal workspace crates. Maintained
case studies may call public `gmrf-core` or `feg-infer` APIs for specialist
diagnostics and workflow orchestration. They must not copy the algorithms those
crates own, and any generally reusable operation first discovered in a study
must be moved to its canonical lower layer. The root facade exposes the stable,
geometry-independent construction path; lower-level implementation APIs are
not automatically part of that stability promise.

The CLI has a stricter boundary because it owns no scientific logic: it depends
on the case-study registry and optionally the experiments registry, and does
not depend directly on FEEC, GMRF, or inference crates.

## Matérn construction

For every supported form degree, FEEC assembles a mass matrix `M_k` and weak
Hodge–Laplacian `L_k`. The integration layer forms

```text
A_k = kappa^2 M_k + L_k
Q_1 = tau^2 A_k
Q_2 = tau^2 A_k M_k^-1 A_k
Q_3 = tau^2 A_k M_k^-1 A_k M_k^-1 A_k.
```

Only FEEC assembly and the selected mass-inverse policy depend on form degree.
The recurrence is implemented once by `prior::matern_recurrence`.

## Physical quantities

Physical outputs are `LinearMap` compositions. In particular, magnetic flux
density is always constructed as

```text
A  --D1-->  magnetic 2-cochain  --Hodge/reconstruction-->  physical B.
```

Direct cell-curl conveniences are not substitutes for this chain in
calibration, normalization, observation uncertainty, or reporting.

Physical-RMS calibration is performed by
`calibrate_prior_to_physical_rms(prior, map, target)`. It evaluates the generic
GMRF transformed covariance for the supplied composed `LinearMap` and rescales
the precision once; case studies do not implement their own covariance or
calibration recurrence.

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
variance. GMRF algorithms therefore remain independent of FEEC boundary
semantics.

For an already assembled arbitrary Gaussian precision,
`GaussianPrior::condition_on_essential_boundary` computes its exact conditional
distribution. For a Matérn model, prefer
`MaternPriorBuilder::essential_boundary_conditions`, because reducing a full
alpha-2 or alpha-3 precision after the recurrence is not generally equivalent
to applying the recurrence on the boundary-reduced FEEC space.
