use crate::{
    boundary::adapt_boundary_spec,
    prior::matern::{build_lindgren_precision_from_system, MaternAlpha},
    sparse::{add_sparse, feec_csr_to_gmrf, scale_matrix},
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr};
use feg_core::{BoundarySpec, LinearGaussianMeasurementSpec};
use formoniq::{
    problems::reduced_linear::{assemble_reduced_hodge_operators, MassInverseApproximation},
    reduction::DofLayout,
};
use gmrf_core::{BlockTridiagonalPrecision, GmrfError};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};

#[derive(Debug, Clone)]
pub struct SpacetimePriorConfig {
    pub times: Vec<f64>,
}

impl SpacetimePriorConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.times.is_empty() {
            return Err("time grid must contain at least one state".to_string());
        }
        if !self.times.iter().all(|time| time.is_finite()) {
            return Err("time grid must contain only finite values".to_string());
        }
        for pair in self.times.windows(2) {
            if pair[1] <= pair[0] {
                return Err(format!(
                    "time grid must be strictly increasing, found {} followed by {}",
                    pair[0], pair[1]
                ));
            }
        }
        Ok(())
    }

    pub fn slice_count(&self) -> usize {
        self.times.len()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarPriorConfig {
    pub kappa: f64,
    pub tau: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Hodge1PriorConfig {
    pub kappa: f64,
    pub tau: f64,
    pub lower_mass_inverse: MassInverseApproximation,
    pub state_mass_inverse: MassInverseApproximation,
}

#[derive(Debug, Clone, Copy)]
pub struct Hodge2PriorConfig {
    pub kappa: f64,
    pub tau: f64,
    pub lower_mass_inverse: MassInverseApproximation,
    pub state_mass_inverse: MassInverseApproximation,
}

#[derive(Debug, Clone)]
pub(crate) struct SpatialEvolutionOperators {
    pub mass: FeecCsr,
    pub drift: FeecCsr,
    pub initial_precision: FeecCsr,
    pub driving_noise_precision: FeecCsr,
    pub layout: DofLayout,
    pub soft_observations: Vec<LinearGaussianMeasurementSpec>,
}

impl SpatialEvolutionOperators {
    fn state_dimension(&self) -> usize {
        self.mass.nrows()
    }
}

#[derive(Debug, Clone)]
pub struct SpacetimePrior {
    operators: SpatialEvolutionOperators,
    pub precision: BlockTridiagonalPrecision,
}

impl SpacetimePrior {
    pub fn state_dimension(&self) -> usize {
        self.operators.state_dimension()
    }

    pub fn layout(&self) -> &DofLayout {
        &self.operators.layout
    }

    pub fn mass(&self) -> &FeecCsr {
        &self.operators.mass
    }

    pub fn drift(&self) -> &FeecCsr {
        &self.operators.drift
    }

    pub fn initial_precision(&self) -> &FeecCsr {
        &self.operators.initial_precision
    }

    pub fn driving_noise_precision(&self) -> &FeecCsr {
        &self.operators.driving_noise_precision
    }

    pub fn soft_observations(&self) -> &[LinearGaussianMeasurementSpec] {
        &self.operators.soft_observations
    }
}

fn build_spacetime_prior_from_operators(
    operators: SpatialEvolutionOperators,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    config.validate()?;
    let dimension = operators.state_dimension();
    validate_square("mass", &operators.mass, dimension)?;
    validate_square("drift", &operators.drift, dimension)?;
    validate_square("initial precision", &operators.initial_precision, dimension)?;
    validate_square(
        "driving-noise precision",
        &operators.driving_noise_precision,
        dimension,
    )?;

    let mut diagonal_blocks = vec![zero_matrix(dimension); config.slice_count()];
    let mut lower_blocks = Vec::with_capacity(config.slice_count().saturating_sub(1));
    diagonal_blocks[0] = add_sparse(&diagonal_blocks[0], &operators.initial_precision);
    let mass_t = operators.mass.transpose();
    for pair in config.times.windows(2) {
        let dt = pair[1] - pair[0];
        let inv_dt = 1.0 / dt;
        let g = add_sparse(&operators.mass, &scale_matrix(&operators.drift, dt));
        let mt_qw_m = scaled_triple_product(
            &mass_t,
            &operators.driving_noise_precision,
            &operators.mass,
            inv_dt,
        );
        let gt_qw_g = scaled_triple_product(
            &g.transpose(),
            &operators.driving_noise_precision,
            &g,
            inv_dt,
        );
        let gt_qw_m = scaled_triple_product(
            &g.transpose(),
            &operators.driving_noise_precision,
            &operators.mass,
            -inv_dt,
        );
        let step = lower_blocks.len();
        diagonal_blocks[step] = add_sparse(&diagonal_blocks[step], &mt_qw_m);
        diagonal_blocks[step + 1] = add_sparse(&diagonal_blocks[step + 1], &gt_qw_g);
        lower_blocks.push(gt_qw_m);
    }

    let precision = BlockTridiagonalPrecision::new(
        diagonal_blocks.iter().map(feec_csr_to_gmrf).collect(),
        lower_blocks.iter().map(feec_csr_to_gmrf).collect(),
    )
    .map_err(gmrf_error_to_string)?;
    Ok(SpacetimePrior {
        operators,
        precision,
    })
}

pub fn build_0form_spacetime_prior(
    topology: &Complex,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: ScalarPriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    validate_matern_parameters(spatial.kappa, spatial.tau)?;
    let adapted = adapt_boundary_spec(boundary, topology.nsimplices(0), 0)?;
    let assembled = assemble_reduced_hodge_operators(
        topology,
        geometry,
        None,
        0,
        &adapted.essential,
        None,
        MassInverseApproximation::RowSumLumped,
    )?;
    build_from_assembled(
        assembled.mass,
        assembled.laplacian,
        assembled.state_mass_inverse,
        assembled.layout,
        adapted.soft_state_measurements,
        spatial.kappa,
        spatial.tau,
        config,
    )
}

pub fn build_1form_spacetime_prior(
    topology: &Complex,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: Hodge1PriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    build_kform_spacetime_prior(
        topology,
        None,
        geometry,
        boundary,
        1,
        spatial.kappa,
        spatial.tau,
        spatial.lower_mass_inverse,
        spatial.state_mass_inverse,
        config,
    )
}

pub fn build_1form_spacetime_prior_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: Hodge1PriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    build_kform_spacetime_prior(
        topology,
        Some(coords),
        geometry,
        boundary,
        1,
        spatial.kappa,
        spatial.tau,
        spatial.lower_mass_inverse,
        spatial.state_mass_inverse,
        config,
    )
}

pub fn build_2form_spacetime_prior(
    topology: &Complex,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: Hodge2PriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    build_kform_spacetime_prior(
        topology,
        None,
        geometry,
        boundary,
        2,
        spatial.kappa,
        spatial.tau,
        spatial.lower_mass_inverse,
        spatial.state_mass_inverse,
        config,
    )
}

pub fn build_2form_spacetime_prior_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: Hodge2PriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    build_kform_spacetime_prior(
        topology,
        Some(coords),
        geometry,
        boundary,
        2,
        spatial.kappa,
        spatial.tau,
        spatial.lower_mass_inverse,
        spatial.state_mass_inverse,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_kform_spacetime_prior(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    grade: usize,
    kappa: f64,
    tau: f64,
    lower_mass_inverse: MassInverseApproximation,
    state_mass_inverse: MassInverseApproximation,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    validate_matern_parameters(kappa, tau)?;
    let adapted = adapt_boundary_spec(
        boundary,
        topology.nsimplices(grade),
        topology.nsimplices(grade - 1),
    )?;
    let assembled = assemble_reduced_hodge_operators(
        topology,
        geometry,
        coords,
        grade,
        &adapted.essential,
        Some(lower_mass_inverse),
        state_mass_inverse,
    )?;
    build_from_assembled(
        assembled.mass,
        assembled.laplacian,
        assembled.state_mass_inverse,
        assembled.layout,
        adapted.soft_state_measurements,
        kappa,
        tau,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_from_assembled(
    mass: FeecCsr,
    laplacian: FeecCsr,
    state_mass_inverse: FeecCsr,
    layout: DofLayout,
    soft_observations: Vec<LinearGaussianMeasurementSpec>,
    kappa: f64,
    tau: f64,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    let drift = add_sparse(&laplacian, &scale_matrix(&mass, kappa * kappa));
    let precision =
        build_lindgren_precision_from_system(&drift, &state_mass_inverse, MaternAlpha::Two, tau);
    let operators = SpatialEvolutionOperators {
        mass,
        drift,
        initial_precision: precision.clone(),
        driving_noise_precision: precision,
        layout,
        soft_observations,
    };
    build_spacetime_prior_from_operators(operators, config)
}

fn validate_matern_parameters(kappa: f64, tau: f64) -> Result<(), String> {
    if !kappa.is_finite() || kappa < 0.0 {
        return Err("spatial kappa must be finite and nonnegative".to_string());
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err("spatial tau must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_square(label: &str, matrix: &FeecCsr, dimension: usize) -> Result<(), String> {
    if matrix.nrows() != dimension || matrix.ncols() != dimension {
        return Err(format!(
            "{label} must be {dimension}x{dimension}, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    Ok(())
}

fn zero_matrix(dimension: usize) -> FeecCsr {
    FeecCsr::from(&FeecCoo::new(dimension, dimension))
}

fn scaled_triple_product(left: &FeecCsr, middle: &FeecCsr, right: &FeecCsr, scale: f64) -> FeecCsr {
    scale_matrix(&(left * middle * right), scale)
}

fn gmrf_error_to_string(err: GmrfError) -> String {
    err.to_string()
}
