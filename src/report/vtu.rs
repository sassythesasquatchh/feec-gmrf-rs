use super::FieldReport;
use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use manifold::geometry::coord::mesh::MeshCoords;
use manifold::topology::complex::Complex;
use std::collections::BTreeSet;
use std::path::Path;

/// Layout of a flattened three-component vector field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorLayout3 {
    /// `[x0, y0, z0, x1, y1, z1, ...]`.
    Interleaved,
    /// `[x0, x1, ..., y0, y1, ..., z0, z1, ...]`.
    ComponentMajor,
}

impl VectorLayout3 {
    /// Convert a validated flattened vector field to VTU's three-vector rows.
    pub fn to_vectors(self, values: &[f64]) -> Result<Vec<[f64; 3]>> {
        validate_finite("vector field", values)?;
        if values.len() % 3 != 0 {
            return Err(FeecGmrfError::Dimension(format!(
                "three-vector field length {} is not divisible by 3",
                values.len()
            )));
        }
        let count = values.len() / 3;
        Ok(match self {
            Self::Interleaved => values
                .chunks_exact(3)
                .map(|value| [value[0], value[1], value[2]])
                .collect(),
            Self::ComponentMajor => (0..count)
                .map(|index| {
                    [
                        values[index],
                        values[count + index],
                        values[2 * count + index],
                    ]
                })
                .collect(),
        })
    }

    /// Flatten VTU three-vector rows using this layout.
    pub fn from_vectors(self, vectors: &[[f64; 3]]) -> Result<Vec<f64>> {
        if vectors.iter().flatten().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "vector field must contain only finite values".to_string(),
            ));
        }
        Ok(match self {
            Self::Interleaved => vectors.iter().flat_map(|value| *value).collect(),
            Self::ComponentMajor => (0..3)
                .flat_map(|component| vectors.iter().map(move |value| value[component]))
                .collect(),
        })
    }

    /// Split a validated flattened vector field into x/y/z component arrays.
    pub fn to_components(self, values: &[f64]) -> Result<[Vec<f64>; 3]> {
        let vectors = self.to_vectors(values)?;
        Ok(std::array::from_fn(|component| {
            vectors.iter().map(|value| value[component]).collect()
        }))
    }
}

/// Builds a VTU bundle of named fields on one FEEC cochain space.
#[derive(Debug, Clone)]
pub struct CochainVtuBuilder {
    form_degree: usize,
    dimension: Option<usize>,
    fields: Vec<(String, Cochain)>,
    names: BTreeSet<String>,
}

impl CochainVtuBuilder {
    pub fn new(form_degree: usize) -> Self {
        Self {
            form_degree,
            dimension: None,
            fields: Vec::new(),
            names: BTreeSet::new(),
        }
    }

    pub fn form_degree(&self) -> usize {
        self.form_degree
    }

    pub fn add_values(&mut self, name: impl Into<String>, values: Vec<f64>) -> Result<&mut Self> {
        validate_finite("cochain field", &values)?;
        let cochain = Cochain::new(self.form_degree, FeecVector::from_vec(values));
        self.add_cochain(name, cochain)
    }

    pub fn add_cochain(&mut self, name: impl Into<String>, cochain: Cochain) -> Result<&mut Self> {
        let name = name.into();
        validate_name(&name)?;
        if cochain.dim() != self.form_degree {
            return Err(FeecGmrfError::Dimension(format!(
                "cochain field `{name}` has degree {}, expected {}",
                cochain.dim(),
                self.form_degree
            )));
        }
        validate_finite("cochain field", cochain.coeffs().as_slice())?;
        let dimension = cochain.len();
        if let Some(expected) = self.dimension {
            if dimension != expected {
                return Err(FeecGmrfError::Dimension(format!(
                    "cochain field `{name}` has length {dimension}, expected {expected}"
                )));
            }
        } else {
            self.dimension = Some(dimension);
        }
        if !self.names.insert(name.clone()) {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "duplicate VTU field name `{name}`"
            )));
        }
        self.fields.push((name, cochain));
        Ok(self)
    }

    /// Add `<prefix>_mean`, `<prefix>_variance`, and
    /// `<prefix>_standard_deviation` from a field report.
    pub fn add_field_report(&mut self, prefix: &str, report: &FieldReport) -> Result<&mut Self> {
        self.add_values(format!("{prefix}_mean"), report.mean.clone())?;
        self.add_values(format!("{prefix}_variance"), report.variance.values.clone())?;
        self.add_values(
            format!("{prefix}_standard_deviation"),
            report.standard_deviations.clone(),
        )?;
        Ok(self)
    }

    pub fn write(
        &self,
        path: impl AsRef<Path>,
        coords: &MeshCoords,
        topology: &Complex,
    ) -> Result<()> {
        if self.fields.is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "at least one cochain VTU field is required".to_string(),
            ));
        }
        let fields = self
            .fields
            .iter()
            .map(|(name, cochain)| (name.as_str(), cochain))
            .collect::<Vec<_>>();
        formoniq::io::write_cochain_vtu_fields(path, coords, topology, self.form_degree, &fields)?;
        Ok(())
    }
}

/// Builds a VTU bundle of named scalar and three-vector top-cell fields.
#[derive(Debug, Clone, Default)]
pub struct TopCellVtuBuilder {
    dimension: Option<usize>,
    vectors: Vec<(String, Vec<[f64; 3]>)>,
    scalars: Vec<(String, Vec<f64>)>,
    names: BTreeSet<String>,
}

impl TopCellVtuBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_scalar(&mut self, name: impl Into<String>, values: Vec<f64>) -> Result<&mut Self> {
        let name = name.into();
        validate_finite("top-cell scalar field", &values)?;
        self.validate_field(&name, values.len())?;
        self.scalars.push((name, values));
        Ok(self)
    }

    pub fn add_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<[f64; 3]>,
    ) -> Result<&mut Self> {
        let name = name.into();
        if values.iter().flatten().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "top-cell vector field must contain only finite values".to_string(),
            ));
        }
        self.validate_field(&name, values.len())?;
        self.vectors.push((name, values));
        Ok(self)
    }

    pub fn add_flat_vector(
        &mut self,
        name: impl Into<String>,
        values: &[f64],
        layout: VectorLayout3,
    ) -> Result<&mut Self> {
        self.add_vector(name, layout.to_vectors(values)?)
    }

    /// Add scalar mean, variance, and standard-deviation top-cell fields.
    pub fn add_scalar_field_report(
        &mut self,
        prefix: &str,
        report: &FieldReport,
    ) -> Result<&mut Self> {
        self.add_scalar(format!("{prefix}_mean"), report.mean.clone())?;
        self.add_scalar(format!("{prefix}_variance"), report.variance.values.clone())?;
        self.add_scalar(
            format!("{prefix}_standard_deviation"),
            report.standard_deviations.clone(),
        )?;
        Ok(self)
    }

    pub fn write(
        &self,
        path: impl AsRef<Path>,
        coords: &MeshCoords,
        topology: &Complex,
    ) -> Result<()> {
        if self.vectors.is_empty() && self.scalars.is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "at least one top-cell VTU field is required".to_string(),
            ));
        }
        let vectors = self
            .vectors
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
            .collect::<Vec<_>>();
        let scalars = self
            .scalars
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
            .collect::<Vec<_>>();
        formoniq::io::write_top_cell_vtu_fields(path, coords, topology, &vectors, &scalars)?;
        Ok(())
    }

    fn validate_field(&mut self, name: &str, dimension: usize) -> Result<()> {
        validate_name(name)?;
        if let Some(expected) = self.dimension {
            if dimension != expected {
                return Err(FeecGmrfError::Dimension(format!(
                    "top-cell field `{name}` has length {dimension}, expected {expected}"
                )));
            }
        } else {
            self.dimension = Some(dimension);
        }
        if !self.names.insert(name.to_string()) {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "duplicate VTU field name `{name}`"
            )));
        }
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.trim().is_empty() || name.chars().any(char::is_control) {
        return Err(FeecGmrfError::InvalidParameter(
            "VTU field names must be non-empty and contain no control characters".to_string(),
        ));
    }
    Ok(())
}

fn validate_finite(name: &str, values: &[f64]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}
