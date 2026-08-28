//! Sparse linear maps and FEEC form-operator bundles.

use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feg_core::{SparseTriplet, SparseTripletMatrix};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// Sparse matrix representation used by public model and map types.
pub type SparseMat = SparseTripletMatrix;

/// A validated linear map between finite-dimensional spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearMap {
    matrix: SparseMat,
}

impl LinearMap {
    /// Wrap a sparse matrix after validating all stored entries.
    pub fn new(matrix: SparseMat) -> Result<Self> {
        validate_sparse(&matrix)?;
        Ok(Self { matrix })
    }

    /// Convert a FEEC CSR matrix into the sparse map representation.
    pub fn from_feec_csr(matrix: &FeecCsr) -> Result<Self> {
        Self::new(sparse_mat_from_feec_csr(matrix))
    }

    /// Construct an identity map.
    pub fn identity(dimension: usize) -> Self {
        Self {
            matrix: SparseMat::diagonal(dimension, 1.0),
        }
    }

    /// Select coefficients from an input vector in the requested output order.
    ///
    /// Repeated indices represent repeated measurements.
    pub fn selector(input_dimension: usize, indices: &[usize]) -> Result<Self> {
        let rows = indices
            .iter()
            .map(|&index| vec![(index, 1.0)])
            .collect::<Vec<_>>();
        Self::weighted_rows(input_dimension, &rows)
    }

    /// Construct a single sparse weighted row.
    pub fn weighted_row(input_dimension: usize, entries: &[(usize, f64)]) -> Result<Self> {
        Self::weighted_rows(input_dimension, &[entries.to_vec()])
    }

    /// Construct sparse weighted rows, merging duplicate columns in each row.
    pub fn weighted_rows(input_dimension: usize, rows: &[Vec<(usize, f64)>]) -> Result<Self> {
        let mut matrix = SparseMat::new(rows.len(), input_dimension);
        for (row_index, row) in rows.iter().enumerate() {
            let mut merged = BTreeMap::<usize, f64>::new();
            for &(column, value) in row {
                if column >= input_dimension {
                    return Err(FeecGmrfError::Dimension(format!(
                        "weighted row column {column} lies outside input dimension {input_dimension}"
                    )));
                }
                if !value.is_finite() {
                    return Err(FeecGmrfError::InvalidParameter(
                        "weighted-row entries must be finite".to_string(),
                    ));
                }
                *merged.entry(column).or_default() += value;
            }
            for (column, value) in merged {
                if value != 0.0 {
                    matrix.push(row_index, column, value);
                }
            }
        }
        Self::new(matrix)
    }

    /// Borrow the sparse matrix representation.
    pub fn matrix(&self) -> &SparseMat {
        &self.matrix
    }

    /// Consume the map and return its sparse matrix.
    pub fn into_matrix(self) -> SparseMat {
        self.matrix
    }

    /// Dimension of the map's codomain.
    pub fn output_dimension(&self) -> usize {
        self.matrix.nrows()
    }

    /// Dimension of the map's domain.
    pub fn input_dimension(&self) -> usize {
        self.matrix.ncols()
    }

    /// Apply the map to a vector.
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>> {
        self.matrix
            .apply_checked(input)
            .map_err(FeecGmrfError::Dimension)
    }

    /// Apply the transpose of the map to a vector.
    pub fn apply_transpose(&self, input: &[f64]) -> Result<Vec<f64>> {
        self.matrix
            .transpose()
            .apply_checked(input)
            .map_err(FeecGmrfError::Dimension)
    }

    /// Return `self` after `inner`.
    pub fn compose(&self, inner: &Self) -> Result<Self> {
        if inner.output_dimension() != self.input_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "cannot compose {}x{} and {}x{} operators",
                self.output_dimension(),
                self.input_dimension(),
                inner.output_dimension(),
                inner.input_dimension()
            )));
        }
        Self::new(multiply_sparse(&self.matrix, &inner.matrix)?)
    }

    /// Stack maps with a shared domain vertically.
    pub fn stack(maps: &[Self]) -> Result<Self> {
        let Some(first) = maps.first() else {
            return Err(FeecGmrfError::InvalidParameter(
                "at least one map is required for stacking".to_string(),
            ));
        };
        let ncols = first.input_dimension();
        if maps.iter().any(|map| map.input_dimension() != ncols) {
            return Err(FeecGmrfError::Dimension(
                "all stacked maps must have the same input dimension".to_string(),
            ));
        }
        let nrows = maps.iter().map(Self::output_dimension).sum();
        let mut output = SparseMat::new(nrows, ncols);
        let mut row_offset = 0;
        for map in maps {
            for (row, col, value) in map.matrix.triplet_iter() {
                output.push(row_offset + row, col, value);
            }
            row_offset += map.output_dimension();
        }
        Self::new(output)
    }

    /// Restrict the codomain to the selected rows.
    pub fn select_outputs(&self, rows: &[usize]) -> Result<Self> {
        Self::new(
            self.matrix
                .select_rows(rows)
                .map_err(FeecGmrfError::Dimension)?,
        )
    }
}

/// Convert a FEEC CSR matrix into the public sparse matrix representation.
pub fn sparse_mat_from_feec_csr(matrix: &FeecCsr) -> SparseMat {
    SparseMat::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: *value,
            }),
    )
}

/// A validated differential-form degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FormDegree(usize);

impl FormDegree {
    /// Validate a form degree against the intrinsic complex dimension.
    pub fn new(value: usize, complex_dimension: usize) -> Result<Self> {
        if value > complex_dimension {
            return Err(FeecGmrfError::Dimension(format!(
                "form degree {value} exceeds complex dimension {complex_dimension}"
            )));
        }
        Ok(Self(value))
    }

    /// Return the numeric form degree.
    pub fn get(self) -> usize {
        self.0
    }
}

/// Relationship between an assembled boundary-reduced form space and its full
/// coefficient space.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryLayout {
    full_dimension: usize,
    active_dofs: Vec<usize>,
    fixed_dofs: Vec<(usize, f64)>,
}

impl BoundaryLayout {
    /// Construct a validated boundary layout.
    pub fn new(
        full_dimension: usize,
        active_dofs: Vec<usize>,
        fixed_dofs: Vec<(usize, f64)>,
    ) -> Result<Self> {
        let active = active_dofs.iter().copied().collect::<BTreeSet<_>>();
        let fixed = fixed_dofs
            .iter()
            .map(|(index, _)| *index)
            .collect::<BTreeSet<_>>();
        if active.len() != active_dofs.len() || fixed.len() != fixed_dofs.len() {
            return Err(FeecGmrfError::InvalidParameter(
                "boundary layout degrees of freedom must be unique".to_string(),
            ));
        }
        if active
            .iter()
            .chain(&fixed)
            .any(|index| *index >= full_dimension)
        {
            return Err(FeecGmrfError::Dimension(
                "boundary layout contains a degree of freedom outside the full space".to_string(),
            ));
        }
        if active.iter().any(|index| fixed.contains(index)) {
            return Err(FeecGmrfError::InvalidParameter(
                "a boundary degree of freedom cannot be both active and fixed".to_string(),
            ));
        }
        if fixed_dofs.iter().any(|(_, value)| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "fixed boundary values must be finite".to_string(),
            ));
        }
        Ok(Self {
            full_dimension,
            active_dofs,
            fixed_dofs,
        })
    }

    /// Identity layout with every coefficient active.
    pub fn identity(dimension: usize) -> Self {
        Self {
            full_dimension: dimension,
            active_dofs: (0..dimension).collect(),
            fixed_dofs: Vec::new(),
        }
    }

    /// Dimension before boundary elimination.
    pub fn full_dimension(&self) -> usize {
        self.full_dimension
    }

    /// Active coefficient indices in reduced ordering.
    pub fn active_dofs(&self) -> &[usize] {
        &self.active_dofs
    }

    /// Fixed coefficient indices and values.
    pub fn fixed_dofs(&self) -> &[(usize, f64)] {
        &self.fixed_dofs
    }

    fn is_identity(&self) -> bool {
        self.fixed_dofs.is_empty()
            && self.active_dofs.len() == self.full_dimension
            && self.active_dofs.iter().copied().eq(0..self.full_dimension)
    }

    fn validate_complete_homogeneous(&self, space: &'static str) -> Result<()> {
        if self.active_dofs.len() + self.fixed_dofs.len() != self.full_dimension {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "{space} boundary layout must classify every full-space coefficient"
            )));
        }
        if self.fixed_dofs.iter().any(|(_, value)| *value != 0.0) {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "{space} exact boundary-domain layout must be homogeneous"
            )));
        }
        Ok(())
    }
}

/// One active FEEC form space together with its weak L2 metric.
#[derive(Debug, Clone, PartialEq)]
pub struct FormSpaceMetric {
    degree: FormDegree,
    mass: SparseMat,
    boundary_layout: BoundaryLayout,
}

impl FormSpaceMetric {
    /// Construct a positive-definite form-space metric on a complete active/fixed layout.
    pub fn new(
        degree: FormDegree,
        mass: SparseMat,
        boundary_layout: BoundaryLayout,
    ) -> Result<Self> {
        if mass.nrows() != boundary_layout.active_dofs().len() {
            return Err(FeecGmrfError::Dimension(format!(
                "degree-{} mass dimension {} does not match {} active boundary-layout coefficients",
                degree.get(),
                mass.nrows(),
                boundary_layout.active_dofs().len()
            )));
        }
        boundary_layout.validate_complete_homogeneous("form-space")?;
        let mass = if mass.nrows() == 0 && mass.ncols() == 0 {
            validate_sparse(&mass)?;
            mass
        } else {
            validate_proper_precision("form-space mass", mass)?
        };
        Ok(Self {
            degree,
            mass,
            boundary_layout,
        })
    }

    /// Form degree of this space.
    pub fn degree(&self) -> FormDegree {
        self.degree
    }

    /// Weak L2 mass matrix in active coordinates.
    pub fn mass(&self) -> &SparseMat {
        &self.mass
    }

    /// Active/fixed coefficient layout relative to the full cochain space.
    pub fn boundary_layout(&self) -> &BoundaryLayout {
        &self.boundary_layout
    }

    /// Active coefficient dimension.
    pub fn dimension(&self) -> usize {
        self.mass.nrows()
    }
}

/// The incoming coefficient derivative `D_(k-1): C^(k-1) -> C^k`.
#[derive(Debug, Clone, PartialEq)]
pub struct IncomingFormOperator {
    lower: FormSpaceMetric,
    active_derivative: LinearMap,
    full_derivative: Option<LinearMap>,
}

impl IncomingFormOperator {
    /// Construct an incoming derivative. Cross-space dimensions are validated
    /// when it is attached to [`ExactFormOperators`].
    pub fn new(
        lower: FormSpaceMetric,
        active_derivative: LinearMap,
        full_derivative: Option<LinearMap>,
    ) -> Result<Self> {
        if active_derivative.input_dimension() != lower.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "incoming derivative input dimension {} does not match lower-space dimension {}",
                active_derivative.input_dimension(),
                lower.dimension()
            )));
        }
        if let Some(full) = &full_derivative {
            if full.input_dimension() != lower.boundary_layout().full_dimension() {
                return Err(FeecGmrfError::Dimension(format!(
                    "full incoming derivative input dimension {} does not match lower full dimension {}",
                    full.input_dimension(),
                    lower.boundary_layout().full_dimension()
                )));
            }
        }
        Ok(Self {
            lower,
            active_derivative,
            full_derivative,
        })
    }

    pub fn lower(&self) -> &FormSpaceMetric {
        &self.lower
    }

    pub fn active_derivative(&self) -> &LinearMap {
        &self.active_derivative
    }

    pub fn full_derivative(&self) -> Option<&LinearMap> {
        self.full_derivative.as_ref()
    }
}

/// The outgoing coefficient derivative `D_k: C^k -> C^(k+1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingFormOperator {
    upper: FormSpaceMetric,
    active_derivative: LinearMap,
    full_derivative: Option<LinearMap>,
}

impl OutgoingFormOperator {
    /// Construct an outgoing derivative. Cross-space dimensions are validated
    /// when it is attached to [`ExactFormOperators`].
    pub fn new(
        upper: FormSpaceMetric,
        active_derivative: LinearMap,
        full_derivative: Option<LinearMap>,
    ) -> Result<Self> {
        if active_derivative.output_dimension() != upper.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "outgoing derivative output dimension {} does not match upper-space dimension {}",
                active_derivative.output_dimension(),
                upper.dimension()
            )));
        }
        if let Some(full) = &full_derivative {
            if full.output_dimension() != upper.boundary_layout().full_dimension() {
                return Err(FeecGmrfError::Dimension(format!(
                    "full outgoing derivative output dimension {} does not match upper full dimension {}",
                    full.output_dimension(),
                    upper.boundary_layout().full_dimension()
                )));
            }
        }
        Ok(Self {
            upper,
            active_derivative,
            full_derivative,
        })
    }

    pub fn upper(&self) -> &FormSpaceMetric {
        &self.upper
    }

    pub fn active_derivative(&self) -> &LinearMap {
        &self.active_derivative
    }

    pub fn full_derivative(&self) -> Option<&LinearMap> {
        self.full_derivative.as_ref()
    }
}

/// Provenance of the adjacent matrices used by an exact finite construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormOperatorProvenance {
    /// Assembled by this package from lowest-order Whitney spaces on the
    /// homogeneous full-relative subcomplex.
    FeecWhitneyHomogeneousFullRelative,
    /// Supplied by a downstream caller and validated as a compatible complex.
    UserSuppliedAdjacentOperators { label: String },
}

/// Boundary-domain meaning certified for an exact adjacent operator bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactBoundaryDomain {
    HomogeneousFullRelative,
    UserSuppliedCompatibleSubcomplex { name: String },
}

/// Adjacent masses, derivatives, and layouts for the exact weak Hodge operator.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactFormOperators {
    complex_dimension: usize,
    state: FormSpaceMetric,
    incoming: Option<IncomingFormOperator>,
    outgoing: Option<OutgoingFormOperator>,
    boundary_domain: ExactBoundaryDomain,
    provenance: FormOperatorProvenance,
}

impl ExactFormOperators {
    /// Validate caller-supplied adjacent operators as one compatible complex.
    pub fn new(
        complex_dimension: usize,
        state: FormSpaceMetric,
        incoming: Option<IncomingFormOperator>,
        outgoing: Option<OutgoingFormOperator>,
        boundary_domain: ExactBoundaryDomain,
        provenance_label: impl Into<String>,
    ) -> Result<Self> {
        Self::validate_and_construct(
            complex_dimension,
            state,
            incoming,
            outgoing,
            boundary_domain,
            FormOperatorProvenance::UserSuppliedAdjacentOperators {
                label: provenance_label.into(),
            },
        )
    }

    /// Assemble lowest-order Whitney adjacent spaces on the homogeneous
    /// full-relative subcomplex before either inverse is formed.
    pub fn from_feec_homogeneous_full_relative(
        topology: &Complex,
        geometry: &MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        let assembled = formoniq::relative::assemble_whitney_full_relative_operators(
            topology, geometry, degree,
        )
        .map_err(|error| FeecGmrfError::Assembly(error.to_string()))?;
        let state = form_space_metric_from_relative(assembled.state(), topology.dim())?;
        let incoming = assembled
            .incoming()
            .zip(assembled.lower())
            .map(|(derivative, lower)| {
                IncomingFormOperator::new(
                    form_space_metric_from_relative(lower, topology.dim())?,
                    LinearMap::from_feec_csr(derivative.active())?,
                    Some(LinearMap::from_feec_csr(derivative.full())?),
                )
            })
            .transpose()?;
        let outgoing = assembled
            .outgoing()
            .zip(assembled.upper())
            .map(|(derivative, upper)| {
                OutgoingFormOperator::new(
                    form_space_metric_from_relative(upper, topology.dim())?,
                    LinearMap::from_feec_csr(derivative.active())?,
                    Some(LinearMap::from_feec_csr(derivative.full())?),
                )
            })
            .transpose()?;
        Self::validate_and_construct(
            topology.dim(),
            state,
            incoming,
            outgoing,
            ExactBoundaryDomain::HomogeneousFullRelative,
            FormOperatorProvenance::FeecWhitneyHomogeneousFullRelative,
        )
    }

    fn validate_and_construct(
        complex_dimension: usize,
        state: FormSpaceMetric,
        incoming: Option<IncomingFormOperator>,
        outgoing: Option<OutgoingFormOperator>,
        boundary_domain: ExactBoundaryDomain,
        provenance: FormOperatorProvenance,
    ) -> Result<Self> {
        let degree = state.degree().get();
        if state.dimension() == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "the exact Matérn state space must have positive dimension".to_string(),
            ));
        }
        if degree > complex_dimension {
            return Err(FeecGmrfError::Dimension(format!(
                "form degree {degree} exceeds complex dimension {complex_dimension}"
            )));
        }
        match (degree, &incoming) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "degree-zero exact operators must not contain an incoming branch".to_string(),
                ))
            }
            (_, None) => {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "degree-{degree} exact operators require the degree-{} incoming branch",
                    degree - 1
                )))
            }
            (_, Some(branch)) => {
                if branch.lower().degree().get() + 1 != degree
                    || branch.active_derivative().output_dimension() != state.dimension()
                {
                    return Err(FeecGmrfError::Dimension(
                        "incoming exact derivative does not connect the adjacent active degrees"
                            .to_string(),
                    ));
                }
                validate_full_derivative_compatibility(
                    "incoming exterior derivative",
                    branch.full_derivative(),
                    branch.active_derivative(),
                    branch.lower().boundary_layout(),
                    state.boundary_layout(),
                )?;
            }
        }
        match (degree == complex_dimension, &outgoing) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "top-degree exact operators must not contain an outgoing branch".to_string(),
                ))
            }
            (false, None) => {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "degree-{degree} exact operators require the degree-{} outgoing branch",
                    degree + 1
                )))
            }
            (false, Some(branch)) => {
                if branch.upper().degree().get() != degree + 1
                    || branch.active_derivative().input_dimension() != state.dimension()
                {
                    return Err(FeecGmrfError::Dimension(
                        "outgoing exact derivative does not connect the adjacent active degrees"
                            .to_string(),
                    ));
                }
                validate_full_derivative_compatibility(
                    "outgoing exterior derivative",
                    branch.full_derivative(),
                    branch.active_derivative(),
                    state.boundary_layout(),
                    branch.upper().boundary_layout(),
                )?;
            }
        }
        if let (Some(incoming), Some(outgoing)) = (&incoming, &outgoing) {
            let composition = multiply_sparse(
                outgoing.active_derivative().matrix(),
                incoming.active_derivative().matrix(),
            )?;
            validate_numerical_zero("adjacent exterior derivatives", &composition)?;
        }
        Ok(Self {
            complex_dimension,
            state,
            incoming,
            outgoing,
            boundary_domain,
            provenance,
        })
    }

    pub fn complex_dimension(&self) -> usize {
        self.complex_dimension
    }

    pub fn degree(&self) -> FormDegree {
        self.state.degree()
    }

    pub fn state(&self) -> &FormSpaceMetric {
        &self.state
    }

    pub fn incoming(&self) -> Option<&IncomingFormOperator> {
        self.incoming.as_ref()
    }

    pub fn outgoing(&self) -> Option<&OutgoingFormOperator> {
        self.outgoing.as_ref()
    }

    pub fn boundary_domain(&self) -> &ExactBoundaryDomain {
        &self.boundary_domain
    }

    pub fn provenance(&self) -> &FormOperatorProvenance {
        &self.provenance
    }
}

fn form_space_metric_from_relative(
    space: &formoniq::relative::FullRelativeWhitneySpace,
    complex_dimension: usize,
) -> Result<FormSpaceMetric> {
    let degree = FormDegree::new(space.degree(), complex_dimension)?;
    let layout = BoundaryLayout::new(
        space.layout().full_dimension,
        space.layout().active_dofs.clone(),
        space
            .layout()
            .prescribed_dofs
            .iter()
            .map(|dof| (dof.index, dof.value))
            .collect(),
    )?;
    FormSpaceMetric::new(degree, sparse_mat_from_feec_csr(space.mass()), layout)
}

fn validate_full_derivative_compatibility(
    name: &'static str,
    full: Option<&LinearMap>,
    active: &LinearMap,
    source: &BoundaryLayout,
    target: &BoundaryLayout,
) -> Result<()> {
    if active.input_dimension() != source.active_dofs().len()
        || active.output_dimension() != target.active_dofs().len()
    {
        return Err(FeecGmrfError::Dimension(format!(
            "{name} active dimensions do not match its source and target layouts"
        )));
    }
    if full.is_none() && (!source.is_identity() || !target.is_identity()) {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "{name} requires a full derivative to certify non-identity boundary layouts"
        )));
    }
    let Some(full) = full else {
        return Ok(());
    };
    if full.input_dimension() != source.full_dimension()
        || full.output_dimension() != target.full_dimension()
    {
        return Err(FeecGmrfError::Dimension(format!(
            "{name} full dimensions do not match its source and target layouts"
        )));
    }

    let source_reduced = source
        .active_dofs()
        .iter()
        .copied()
        .enumerate()
        .map(|(reduced, full)| (full, reduced))
        .collect::<BTreeMap<_, _>>();
    let mut difference = BTreeMap::<(usize, usize), f64>::new();
    for (row, column, value) in full.matrix().triplet_iter() {
        if let Some(reduced_column) = source_reduced.get(&column) {
            *difference.entry((row, *reduced_column)).or_default() += value;
        }
    }
    for (row, column, value) in active.matrix().triplet_iter() {
        let full_row = target.active_dofs()[row];
        *difference.entry((full_row, column)).or_default() -= value;
    }
    validate_difference_zero(name, difference.values().copied())
}

fn validate_numerical_zero(name: &'static str, matrix: &SparseMat) -> Result<()> {
    validate_difference_zero(name, matrix.triplet_iter().map(|(_, _, value)| value))
}

fn validate_difference_zero(
    name: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<()> {
    let values = values.into_iter().collect::<Vec<_>>();
    let maximum = values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let tolerance = 64.0 * f64::EPSILON * maximum.max(1.0);
    if maximum > tolerance {
        return Err(FeecGmrfError::Assembly(format!(
            "{name} compatibility residual {maximum:.6e} exceeds tolerance {tolerance:.6e}"
        )));
    }
    Ok(())
}

/// FEEC operators required to construct a Gaussian prior on one form space.
#[derive(Debug, Clone, PartialEq)]
pub struct FormOperators {
    degree: FormDegree,
    complex_dimension: usize,
    mass: SparseMat,
    hodge_laplacian: SparseMat,
    boundary_layout: BoundaryLayout,
    exterior_derivative_in: Option<LinearMap>,
    exterior_derivative_out: Option<LinearMap>,
}

impl FormOperators {
    /// Construct a form-operator bundle from assembled matrices.
    pub fn new(
        degree: FormDegree,
        complex_dimension: usize,
        mass: SparseMat,
        hodge_laplacian: SparseMat,
    ) -> Result<Self> {
        validate_square_same_shape(&mass, &hodge_laplacian)?;
        validate_sparse(&mass)?;
        validate_sparse(&hodge_laplacian)?;
        if degree.get() > complex_dimension {
            return Err(FeecGmrfError::Dimension(format!(
                "form degree {} exceeds complex dimension {complex_dimension}",
                degree.get()
            )));
        }
        let dimension = mass.nrows();
        Ok(Self {
            degree,
            complex_dimension,
            mass,
            hodge_laplacian,
            boundary_layout: BoundaryLayout::identity(dimension),
            exterior_derivative_in: None,
            exterior_derivative_out: None,
        })
    }

    /// Attach the layout used to eliminate essential-boundary coefficients.
    pub fn with_boundary_layout(mut self, layout: BoundaryLayout) -> Result<Self> {
        if layout.active_dofs.len() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "boundary layout has {} active coefficients but operators have dimension {}",
                layout.active_dofs.len(),
                self.dimension()
            )));
        }
        self.boundary_layout = layout;
        Ok(self)
    }

    /// Attach the exterior derivative entering this form space.
    pub fn with_incoming_exterior_derivative(mut self, derivative: LinearMap) -> Result<Self> {
        if derivative.output_dimension() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "incoming exterior derivative output dimension {} does not match form dimension {}",
                derivative.output_dimension(),
                self.dimension()
            )));
        }
        self.exterior_derivative_in = Some(derivative);
        Ok(self)
    }

    /// Attach the exterior derivative leaving this form space.
    pub fn with_exterior_derivative(mut self, derivative: LinearMap) -> Result<Self> {
        if derivative.input_dimension() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "exterior derivative input dimension {} does not match form dimension {}",
                derivative.input_dimension(),
                self.dimension()
            )));
        }
        self.exterior_derivative_out = Some(derivative);
        Ok(self)
    }

    /// Form degree represented by these operators.
    pub fn degree(&self) -> FormDegree {
        self.degree
    }

    /// Intrinsic dimension of the underlying complex.
    pub fn complex_dimension(&self) -> usize {
        self.complex_dimension
    }

    /// Number of coefficients in the form space.
    pub fn dimension(&self) -> usize {
        self.mass.nrows()
    }

    /// Weak mass matrix.
    pub fn mass(&self) -> &SparseMat {
        &self.mass
    }

    /// Weak Hodge--Laplacian matrix.
    pub fn hodge_laplacian(&self) -> &SparseMat {
        &self.hodge_laplacian
    }

    /// Boundary reduction used by the assembled operators.
    pub fn boundary_layout(&self) -> &BoundaryLayout {
        &self.boundary_layout
    }

    /// Exterior derivative entering this form space, when supplied.
    pub fn incoming_exterior_derivative(&self) -> Option<&LinearMap> {
        self.exterior_derivative_in.as_ref()
    }

    /// Exterior derivative leaving this form space, when supplied.
    pub fn exterior_derivative(&self) -> Option<&LinearMap> {
        self.exterior_derivative_out.as_ref()
    }
}

pub(crate) fn validate_sparse(matrix: &SparseMat) -> Result<()> {
    for (row, col, value) in matrix.triplet_iter() {
        if row >= matrix.nrows() || col >= matrix.ncols() {
            return Err(FeecGmrfError::Dimension(format!(
                "sparse entry ({row}, {col}) lies outside {}x{} matrix",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
        if !value.is_finite() {
            return Err(FeecGmrfError::InvalidParameter(
                "sparse entries must be finite".to_string(),
            ));
        }
    }
    Ok(())
}

/// Validate and canonicalize a public proper-Gaussian precision.
///
/// Duplicate coordinates are summed, exact zeros are removed, and entries are
/// sorted by `(row, column)`. This preserves the represented linear operator;
/// it does not symmetrize, clip, jitter, or otherwise alter its values. The
/// returned precision has passed an unshifted sparse Cholesky factorization.
pub(crate) fn validate_proper_precision(
    matrix_name: &'static str,
    matrix: SparseMat,
) -> Result<SparseMat> {
    if matrix.nrows() != matrix.ncols() {
        return Err(FeecGmrfError::Dimension(format!(
            "{matrix_name} must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }

    let mut entries = BTreeMap::<(usize, usize), f64>::new();
    for (row, column, value) in matrix.triplet_iter() {
        if row >= matrix.nrows() || column >= matrix.ncols() {
            return Err(FeecGmrfError::Dimension(format!(
                "sparse entry ({row}, {column}) lies outside {}x{} {matrix_name}",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
        if !value.is_finite() {
            return Err(FeecGmrfError::NonFiniteMatrixEntry {
                matrix: matrix_name,
                row,
                column,
            });
        }
        let total = entries.entry((row, column)).or_default();
        *total += value;
        if !total.is_finite() {
            return Err(FeecGmrfError::NonFiniteMatrixEntry {
                matrix: matrix_name,
                row,
                column,
            });
        }
    }
    entries.retain(|_, value| *value != 0.0);

    let entry_scale = entries
        .values()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    // Sparse products assembled through different traversal orders can differ
    // by a few ulps. Keep this tolerance strictly relative to matrix scale so
    // rescaling a valid precision does not change the symmetry decision.
    let symmetry_tolerance = 64.0 * f64::EPSILON * entry_scale;
    let (asymmetry_row, asymmetry_column, max_abs_difference) =
        entries
            .iter()
            .fold((0, 0, 0.0_f64), |current, (&(row, column), &value)| {
                let difference =
                    (value - entries.get(&(column, row)).copied().unwrap_or(0.0)).abs();
                if difference > current.2 {
                    (row, column, difference)
                } else {
                    current
                }
            });
    if max_abs_difference > symmetry_tolerance {
        return Err(FeecGmrfError::MatrixNotSymmetric {
            matrix: matrix_name,
            row: asymmetry_row,
            column: asymmetry_column,
            max_abs_difference,
            tolerance: symmetry_tolerance,
        });
    }

    let canonical = SparseMat::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        entries
            .into_iter()
            .map(|((row, col), value)| SparseTriplet { row, col, value }),
    );
    let backend = feg_infer::sparse::core_triplet_to_gmrf(&canonical);
    backend
        .cholesky_sqrt_lower()
        .map_err(|error| FeecGmrfError::PrecisionNotPositiveDefinite {
            matrix: matrix_name,
            reason: error.to_string(),
        })?;
    Ok(canonical)
}

pub(crate) fn validate_square_same_shape(lhs: &SparseMat, rhs: &SparseMat) -> Result<()> {
    if lhs.nrows() != lhs.ncols() || rhs.nrows() != rhs.ncols() {
        return Err(FeecGmrfError::Dimension(
            "form mass and Hodge--Laplacian matrices must be square".to_string(),
        ));
    }
    if lhs.nrows() != rhs.nrows() {
        return Err(FeecGmrfError::Dimension(format!(
            "operator dimensions differ: {} and {}",
            lhs.nrows(),
            rhs.nrows()
        )));
    }
    Ok(())
}

pub(crate) fn add_sparse(lhs: &SparseMat, rhs: &SparseMat) -> Result<SparseMat> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err(FeecGmrfError::Dimension(
            "sparse addition requires equal shapes".to_string(),
        ));
    }
    let lhs = feg_infer::sparse::core_triplet_to_gmrf(lhs);
    let rhs = feg_infer::sparse::core_triplet_to_gmrf(rhs);
    Ok(feg_infer::sparse::gmrf_sparse_to_core_triplet(
        &gmrf_core::add_sparse(&lhs, &rhs),
    ))
}

pub(crate) fn multiply_sparse(lhs: &SparseMat, rhs: &SparseMat) -> Result<SparseMat> {
    let lhs = feg_infer::sparse::core_triplet_to_gmrf(lhs);
    let rhs = feg_infer::sparse::core_triplet_to_gmrf(rhs);
    let product = gmrf_core::multiply_sparse(&lhs, &rhs)?;
    Ok(feg_infer::sparse::gmrf_sparse_to_core_triplet(&product))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_and_stacking_are_dimension_checked() {
        let scale = LinearMap::new(SparseMat::diagonal(2, 2.0)).unwrap();
        let composed = scale.compose(&scale).unwrap();
        assert_eq!(composed.apply(&[1.0, 3.0]).unwrap(), vec![4.0, 12.0]);

        let stacked = LinearMap::stack(&[scale.clone(), scale]).unwrap();
        assert_eq!(
            stacked.apply(&[1.0, 3.0]).unwrap(),
            vec![2.0, 6.0, 2.0, 6.0]
        );
    }

    #[test]
    fn rectangular_sparse_composition_matches_sequential_application() {
        let inner = LinearMap::weighted_rows(
            4,
            &[
                vec![(0, 1.0), (1, 2.0)],
                vec![(1, -1.0), (2, 3.0)],
                vec![(0, 2.0), (3, 1.0)],
            ],
        )
        .unwrap();
        let outer =
            LinearMap::weighted_rows(3, &[vec![(0, 2.0), (1, 1.0)], vec![(0, -1.0), (2, 0.5)]])
                .unwrap();
        let input = [1.0, -2.0, 0.5, 4.0];

        let composed = outer.compose(&inner).unwrap();
        let sequential = outer.apply(&inner.apply(&input).unwrap()).unwrap();

        assert_eq!(composed.output_dimension(), 2);
        assert_eq!(composed.input_dimension(), 4);
        assert_eq!(composed.apply(&input).unwrap(), sequential);
    }

    #[test]
    fn sparse_addition_cancels_entries_after_backend_conversion() {
        let left = SparseMat::from_triplets(
            2,
            2,
            [
                SparseTriplet {
                    row: 0,
                    col: 1,
                    value: 3.0,
                },
                SparseTriplet {
                    row: 1,
                    col: 0,
                    value: 2.0,
                },
            ],
        );
        let right = SparseMat::from_triplets(
            2,
            2,
            [
                SparseTriplet {
                    row: 0,
                    col: 1,
                    value: -3.0,
                },
                SparseTriplet {
                    row: 1,
                    col: 1,
                    value: 5.0,
                },
            ],
        );

        let sum = add_sparse(&left, &right).unwrap();
        assert_eq!(sum.apply_checked(&[2.0, 4.0]).unwrap(), [0.0, 24.0]);
    }

    #[test]
    fn selector_preserves_order_and_repeated_indices() {
        let selector = LinearMap::selector(3, &[2, 0, 2]).unwrap();
        assert_eq!(selector.apply(&[1.0, 2.0, 3.0]).unwrap(), [3.0, 1.0, 3.0]);
        assert!(LinearMap::selector(3, &[3]).is_err());
    }

    #[test]
    fn weighted_rows_merge_duplicate_columns() {
        let map =
            LinearMap::weighted_rows(3, &[vec![(0, 1.0), (0, 2.0), (2, -1.0)], vec![(1, 4.0)]])
                .unwrap();
        assert_eq!(map.apply(&[2.0, 3.0, 5.0]).unwrap(), [1.0, 12.0]);
        assert!(LinearMap::weighted_row(2, &[(0, f64::NAN)]).is_err());
    }
}
