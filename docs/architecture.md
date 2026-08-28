# Mathematical and software architecture

This guide follows a model from a finite element mesh to posterior statistics.
It also states the assumptions that make the resulting precision matrices
mathematically meaningful and numerically tractable.

## From geometry to a Gaussian model

The computation has three main stages:

```text
mesh and metric
    |
    v
FEEC spaces, mass matrices, exterior derivatives, boundary reduction
    |
    v
prior precision + observations + constraints + derived maps
    |
    v
sparse GMRF factorization, posterior actions, samples, and reports
```

The `feec` workspace assembles discrete differential operators. `gmrf-core`
implements Gaussian precision algebra and sparse solves. The integration
workspace turns FEEC operators into statistical models and exposes them through
the `feec_gmrf` crate.

The dependency direction reflects the mathematics:

- FEEC does not need to know how a Gaussian posterior is computed.
- GMRF algorithms do not need to know whether a sparse matrix came from a
  mesh, a graph, or another discretization.
- The integration layer knows both the physical meaning of the FEEC operators
  and the statistical meaning of the precision and observation terms.

`feg-core` contains the small set of matrix and model specifications shared
across those boundaries. `feg-infer` contains FEEC-specific model composition,
and `feg-case-studies` supplies geometries, material parameters, measurements,
and scientific outputs.

## FEEC representation

On a simplicial complex, a discrete \(k\)-form is a vector in a cochain space
\(C^k\). The incidence matrix \(D_k\) represents the exterior derivative and
satisfies

```text
D_(k+1) D_k = 0.
```

The mass matrix \(M_k\) represents the finite element \(L^2\) inner product.
Weak codifferentials and Hodge–Laplacians are assembled from incidence and mass
matrices. Boundary reduction supplies the active degrees of freedom and all
affine contributions from prescribed values.

`FormOperators` packages the matrices needed to construct a prior without
discarding their form degree or ambient dimension. `LinearMap` is the
geometry-independent sparse map used after assembly. It can apply a map or its
transpose, compose maps, stack maps, select rows, and convert FEEC CSR
matrices into the sparse representation used by the inference layer.

Composition is performed with the sparse matrix backend. Triplet or ordered-map
assembly is reserved for cases that genuinely construct a new sparsity pattern,
such as collecting independently generated entries or merging duplicates.
This avoids repeated symbolic reconstruction when two assembled operators can
be multiplied directly.

## Gaussian precision representation

For a proper Gaussian field

```text
x ~ N(mu, Q^-1),
```

the sparse precision \(Q\) and information vector \(\eta=Q\mu\) are the primary
objects. Sparse Cholesky factors are cached and reused for means, covariance
actions, sampling, and uncertainty estimators.

For a linear observation

```text
y = Hx + b + epsilon,    epsilon ~ N(0, R),
```

the posterior terms are

```text
Q_post   = Q + H^T R^-1 H
eta_post = eta + H^T R^-1 (y - b).
```

These products are assembled in `gmrf-core`. Scalar, diagonal, and sparse
observation precisions use the same update path. Stacked observations retain
their sparse row structure.

Hard constraints \(Cx=d\) are handled by a sparse KKT factorization. Posterior
means and transformed variances include the exact low-rank covariance
correction from the constraints; they are not approximated as observations
with an arbitrarily small noise variance.

## Prior construction

### Matérn recurrence

For form degree \(k\), define

```text
A_k = kappa^2 M_k + L_k,
```

where \(L_k\) is a weak, self-adjoint, non-negative Hodge–Laplacian. The
supported integer orders use

```text
Q_1 = tau^2 A_k
Q_2 = tau^2 A_k M_k^-1 A_k
Q_3 = tau^2 A_k M_k^-1 A_k M_k^-1 A_k.
```

The mass-inverse policy is part of the discretization. It may use a lumped
diagonal, an assembled diagonal, a projected inverse, or an explicitly
supplied matrix. The choice changes the discrete covariance and must therefore
remain visible in the model configuration.

`MaternPriorBuilder::essential_boundary_conditions` reduces \(M_k\), \(L_k\),
and the selected inverse before applying this recurrence. Constructing a
full-space alpha-2 or alpha-3 precision and conditioning it afterwards is, in
general, a different Gaussian model.

### Hodge-decomposed 1-form priors

A 1-form is represented as the sum of exact, coexact, and harmonic branches:

```text
u = d phi + delta psi + h.
```

The latent precision is block diagonal across the requested branches, while a
sparse synthesis map sends the branch coordinates to the ambient 1-form
space. Observations and derived quantities are pulled back through this map.

For an eigenmode with Hodge–Laplacian eigenvalue \(\lambda\):

- a potential-spectrum branch with exponent \(a\) has ambient covariance
  proportional to
  \(\lambda(\kappa^2+\lambda)^{-a}\);
- a form-spectrum branch compensates for the derivative factor and has ambient
  covariance proportional to
  \((\kappa^2+\lambda)^{-a}\).

Sparse gauges make the potential representation proper. They do not define an
additional physical branch. Harmonic modes are represented by a computed basis
and receive their own finite-dimensional precision.

### PDE-induced priors and PDE observations

`LinearPdeSystem` wraps a boundary-reduced affine residual

```text
r(z) = A z + c.
```

There are three distinct uses:

1. `matern_prior_builder` interprets \(A\) as a weak elliptic generator
   \(L_k\) compatible with the state mass matrix and applies the Matérn
   recurrence. Self-adjointness and non-negativity are modelling assumptions.
2. `pde_induced_prior` constructs
   \(Q=A^TWA\), with residual precision \(W\). This treats the residual norm as
   the definition of the prior.
3. `LinearPdeModelBuilder::weak_residual` adds the same quadratic form as an
   observation of zero to a separately chosen prior. This changes the
   posterior, not the prior model.

Keeping these operations separate prevents an elliptic regularizer, a physical
model-discrepancy term, and a Matérn SPDE from being treated as interchangeable.

## Boundary reduction

Essential values are represented by

```text
x = P z + g,
```

where \(z\) contains active coefficients, \(P\) inserts them into the full
cochain, and \(g\) contains prescribed values. The same elimination is applied
to every model component:

```text
Hx + b  ->  (HP)z + (b + Hg)
Cx = d  ->  (CP)z = d - Cg
Gx + c  ->  (GP)z + (c + Gg)
r(x)    ->  r(Pz + g),       J_z = J_x P.
```

Posterior calculations remain in active coordinates. Full-cochain means,
variances, and samples are lifted afterwards. Prescribed coefficients retain
their specified values and have exactly zero variance.

## Physical pushforwards

A physical field is a named `LinearMap`, not a post-processing callback. For a
magnetic vector potential, the map is

```text
A cochain --D1--> magnetic flux cochain --R_B--> cellwise B vectors.
```

`D1` carries discrete orientation and topological information. `R_B` carries
the Whitney reconstruction and Euclidean vector-proxy choice. Calibration,
observations, means, covariance actions, and samples all use the composed map
\(R_BD_1\).

The flux cochain itself remains available as a derived quantity. Euclidean
reconstruction is needed only when the requested output is a vector field in
physical coordinates.

## Linear state/source models

`feg-infer::linear_pde::LinearPdeUqProblem` represents a reduced state together
with uncertain forcing or material inputs. Each input specifies a prior and a
linear map into the PDE residual.

An input can be represented in two ways:

- **latent**: append its coefficients to the joint Gaussian state and retain
  posterior statistics for the input;
- **collapsed**: analytically push its covariance through the residual map and
  condition only the physical state.

`RepresentationPreference` selects or permits these alternatives. Joint
measurements and derived quantities provide separate sparse blocks for the
state and each latent input. The solver assembles one joint precision, applies
physical observations and the weak PDE residual, factors it, and then extracts
state, input, residual, and derived uncertainty.

This workflow remains in `feg-infer` because the block layout, input
representation, scaling, and source elimination are explicit modelling
choices. Its Gaussian updates, sparse rows, factorizations, and variance
estimators come from `gmrf-core`.

## Nonlinear Laplace inference

A nonlinear residual term supplies a residual vector \(r(x)\), a sparse
Jacobian \(J(x)\), and a residual precision. At each Gauss–Newton iteration the
integration layer forms

```text
H = Q_prior + sum J^T W J
g = Q_prior (x - mu) + sum J^T W r
```

and obtains the update from a sparse solve. Damping and line search control the
step. Convergence is checked using the actual nonlinear objective and residual
evaluations. The final Hessian approximation defines the local Laplace
posterior.

FEEC assembles the residual and Jacobian; the integration layer controls the
iteration; `gmrf-core` performs the sparse Gaussian algebra.

## Spatiotemporal models

`SpacetimePriorBuilder` combines spatial FEEC operators with a time
discretization. For an implicit Euler step, the transition residual has the
form

```text
G x_(t+1) - M x_t.
```

The initial precision and process precision generate a block-tridiagonal
spatiotemporal precision. The integration layer chooses the time grid and
forms the blocks; `gmrf-core` stores and solves the resulting Gaussian model.

## Uncertainty and repeated computation

The main uncertainty routes are:

- exact covariance solves for small or moderate outputs;
- selected-inverse methods for compatible sparsity patterns;
- seeded posterior Monte Carlo;
- Hutchinson diagonal estimation;
- local Rao–Blackwellised Monte Carlo for `LinearPdeUqProblem`;
- automatic exact/Hutchinson selection based on latent dimension.

Stochastic estimators report batch standard errors and stabilization metadata.
Variance floors correct negative roundoff or estimator noise but do not erase
the raw negative count or raw minimum.

`GaussianPrior::factor` caches a prior factor. A
`PreparedLinearGaussianModel` also caches the posterior design and
factorization when observation values change but operators, noise, and
constraints remain fixed. Transformed means and variances use covariance
actions through the requested map rather than constructing a dense latent
covariance.

## Reporting

`PosteriorReportBuilder` requests latent, full-cochain, derived, or ad hoc
mapped fields. It retains the estimator metadata, computes small exact
quantity-of-interest covariance blocks, evaluates Gaussian predictive
diagnostics, and produces typed tables and VTU fields.

Scientific interpretation remains explicit in the calling code: units,
reference solutions, truth fields, acceptance thresholds, and selected
artifacts are supplied by the application.
