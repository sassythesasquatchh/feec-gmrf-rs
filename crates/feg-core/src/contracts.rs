use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryTreatment {
    HardEssential,
    SoftEssential { variance: f64 },
    Natural,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryRegionSpec {
    pub name: String,
    pub dofs: Vec<usize>,
    pub values: Vec<f64>,
    pub treatment: BoundaryTreatment,
}

impl BoundaryRegionSpec {
    pub fn new(
        name: impl Into<String>,
        dofs: Vec<usize>,
        values: Vec<f64>,
        treatment: BoundaryTreatment,
    ) -> Self {
        assert_eq!(
            dofs.len(),
            values.len(),
            "boundary dof and value counts must match"
        );
        Self {
            name: name.into(),
            dofs,
            values,
            treatment,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundarySpec {
    pub state_regions: Vec<BoundaryRegionSpec>,
    pub auxiliary_regions: Vec<BoundaryRegionSpec>,
}

impl BoundarySpec {
    pub fn with_state_region(mut self, region: BoundaryRegionSpec) -> Self {
        self.state_regions.push(region);
        self
    }

    pub fn with_auxiliary_region(mut self, region: BoundaryRegionSpec) -> Self {
        self.auxiliary_regions.push(region);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseTriplet {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTripletMatrix {
    nrows: usize,
    ncols: usize,
    triplets: Vec<SparseTriplet>,
}

impl SparseTripletMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }

    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: impl IntoIterator<Item = SparseTriplet>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            triplets: triplets.into_iter().collect(),
        }
    }

    pub fn diagonal(dimension: usize, value: f64) -> Self {
        Self::from_triplets(
            dimension,
            dimension,
            (0..dimension)
                .filter(move |_| value != 0.0)
                .map(move |index| SparseTriplet {
                    row: index,
                    col: index,
                    value,
                }),
        )
    }

    pub fn from_rows(ncols: usize, rows: &[Vec<(usize, f64)>]) -> Result<Self, String> {
        let mut matrix = Self::new(rows.len(), ncols);
        for (row, entries) in rows.iter().enumerate() {
            for &(col, value) in entries {
                if col >= ncols {
                    return Err(format!(
                        "row {row} references column {col}, outside matrix width {ncols}"
                    ));
                }
                if !value.is_finite() {
                    return Err("sparse triplet row contains a non-finite value".to_string());
                }
                if value != 0.0 {
                    matrix.push(row, col, value);
                }
            }
        }
        Ok(matrix)
    }

    pub fn from_columns(nrows: usize, columns: &[Vec<(usize, f64)>]) -> Result<Self, String> {
        let mut matrix = Self::new(nrows, columns.len());
        for (col, entries) in columns.iter().enumerate() {
            for &(row, value) in entries {
                if row >= nrows {
                    return Err(format!(
                        "column {col} references row {row}, outside matrix height {nrows}"
                    ));
                }
                if !value.is_finite() {
                    return Err("sparse triplet column contains a non-finite value".to_string());
                }
                if value != 0.0 {
                    matrix.push(row, col, value);
                }
            }
        }
        Ok(matrix)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nnz(&self) -> usize {
        self.triplets.len()
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        self.triplets.push(SparseTriplet { row, col, value });
    }

    pub fn triplet_iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.triplets
            .iter()
            .map(|entry| (entry.row, entry.col, entry.value))
    }

    pub fn transpose(&self) -> Self {
        Self::from_triplets(
            self.ncols,
            self.nrows,
            self.triplets.iter().map(|entry| SparseTriplet {
                row: entry.col,
                col: entry.row,
                value: entry.value,
            }),
        )
    }

    pub fn scaled(&self, scale: f64) -> Self {
        Self::from_triplets(
            self.nrows,
            self.ncols,
            self.triplets.iter().filter_map(|entry| {
                let value = scale * entry.value;
                (value != 0.0).then_some(SparseTriplet {
                    row: entry.row,
                    col: entry.col,
                    value,
                })
            }),
        )
    }

    pub fn select_rows(&self, rows: &[usize]) -> Result<Self, String> {
        let mut selected = BTreeMap::new();
        for (new_row, old_row) in rows.iter().copied().enumerate() {
            if old_row >= self.nrows {
                return Err(format!(
                    "selected row {old_row} is outside matrix height {}",
                    self.nrows
                ));
            }
            if selected.insert(old_row, new_row).is_some() {
                return Err(format!("selected row {old_row} appears more than once"));
            }
        }
        Ok(Self::from_triplets(
            rows.len(),
            self.ncols,
            self.triplets.iter().filter_map(|entry| {
                Some(SparseTriplet {
                    row: *selected.get(&entry.row)?,
                    col: entry.col,
                    value: entry.value,
                })
            }),
        ))
    }

    pub fn apply_checked(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        if values.len() != self.ncols {
            return Err(format!(
                "input vector length {} does not match matrix width {}",
                values.len(),
                self.ncols
            ));
        }
        let mut output = vec![0.0; self.nrows];
        for entry in &self.triplets {
            if entry.row >= self.nrows || entry.col >= self.ncols {
                return Err(format!(
                    "triplet ({}, {}) is outside matrix shape {}x{}",
                    entry.row, entry.col, self.nrows, self.ncols
                ));
            }
            output[entry.row] += entry.value * values[entry.col];
        }
        Ok(output)
    }

    pub fn quadratic_form_checked(&self, values: &[f64]) -> Result<f64, String> {
        if self.nrows != self.ncols {
            return Err("quadratic form requires a square matrix".to_string());
        }
        Ok(values
            .iter()
            .zip(self.apply_checked(values)?)
            .map(|(left, right)| left * right)
            .sum())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussianMeasurementSpec {
    pub name: String,
    pub operator: SparseTripletMatrix,
    pub observations: Vec<f64>,
    pub bias: Vec<f64>,
    pub variance: f64,
}

impl LinearGaussianMeasurementSpec {
    pub fn validate(&self, state_dimension: usize) -> Result<(), String> {
        if self.operator.ncols() != state_dimension {
            return Err(format!("measurement `{}` operator column count {} must match state dimension {state_dimension}", self.name, self.operator.ncols()));
        }
        if self.operator.nrows() != self.observations.len()
            || self.operator.nrows() != self.bias.len()
        {
            return Err(format!(
                "measurement `{}` row, observation, and bias counts must match",
                self.name
            ));
        }
        if !self.variance.is_finite() || self.variance <= 0.0 {
            return Err(format!(
                "measurement `{}` variance must be finite and positive",
                self.name
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearResidualEvaluation {
    pub residual: Vec<f64>,
    pub jacobian: SparseTripletMatrix,
}

impl NonlinearResidualEvaluation {
    pub fn validate(
        &self,
        residual_dimension: usize,
        state_dimension: usize,
    ) -> Result<(), String> {
        if self.residual.len() != residual_dimension
            || self.jacobian.nrows() != residual_dimension
            || self.jacobian.ncols() != state_dimension
        {
            return Err(
                "nonlinear residual/Jacobian dimensions do not match the model".to_string(),
            );
        }
        Ok(())
    }
}

pub trait NonlinearResidualModel {
    fn state_dimension(&self) -> usize;
    fn residual_dimension(&self) -> usize;

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        self.residual_and_jacobian(state)
            .map(|evaluation| evaluation.residual)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String>;
}
