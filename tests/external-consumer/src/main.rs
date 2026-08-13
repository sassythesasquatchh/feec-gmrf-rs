use feec_gmrf::prelude::*;

struct QuadraticResidual;

impl ResidualModel for QuadraticResidual {
    fn state_dimension(&self) -> usize {
        2
    }

    fn residual_dimension(&self) -> usize {
        1
    }

    fn residual_and_jacobian(
        &self,
        state: &[f64],
    ) -> std::result::Result<NonlinearResidualEvaluation, String> {
        if state.len() != 2 {
            return Err("quadratic residual expects two coefficients".to_string());
        }
        Ok(NonlinearResidualEvaluation {
            residual: vec![state[0] * state[0] + state[1] - 1.0],
            jacobian: SparseMat::from_rows(2, &[vec![(0, 2.0 * state[0]), (1, 1.0)]])?,
        })
    }
}

fn operator_prior(degree: usize) -> Result<GaussianPrior> {
    let degree = FormDegree::new(degree, 2)?;
    let operators = FormOperators::new(
        degree,
        2,
        SparseMat::diagonal(2, 1.0),
        SparseMat::diagonal(2, 0.5),
    )?;
    MaternPriorBuilder::from_operators(operators)
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.25, 0.8)?)
        .build()
}

fn main() -> Result<()> {
    let _scalar_prior = operator_prior(0)?;
    let form_prior = operator_prior(1)?;

    let sensor = LinearMap::new(
        SparseMat::from_rows(2, &[vec![(0, 1.0)]]).map_err(FeecGmrfError::Dimension)?,
    )?;
    let observation =
        LinearObservation::new(sensor.clone(), vec![0.2], GaussianNoise::variance(0.05)?)?;
    let posterior = LinearGaussianModelBuilder::new(form_prior.clone())
        .observe(observation)?
        .derive(DerivedQuantity::new("custom_sensor", sensor)?)?
        .condition()?;
    assert_eq!(posterior.mean().len(), 2);

    let boundary_prior = form_prior.clone().condition_on_essential_boundary(
        EssentialBoundaryConditions::prescribed(vec![1], vec![1.5])?,
    )?;
    let full_sensor = LinearMap::new(
        SparseMat::from_rows(2, &[vec![(0, 1.0), (1, 2.0)]]).map_err(FeecGmrfError::Dimension)?,
    )?;
    let mut boundary_posterior = LinearGaussianModelBuilder::new(boundary_prior)
        .derive(DerivedQuantity::new("boundary_sensor", full_sensor)?)?
        .condition()?;
    assert_eq!(boundary_posterior.cochain_mean()[1], 1.5);
    assert_eq!(boundary_posterior.cochain_variances()?[1], 0.0);

    let d1 = LinearMap::new(SparseMat::diagonal(2, 2.0))?;
    let reconstruction = LinearMap::new(SparseMat::diagonal(2, 0.5))?;
    let magnetic = magnetic_field_map(&d1, &reconstruction)?;
    assert_eq!(magnetic.apply(&[1.0, -1.0])?, vec![1.0, -1.0]);
    let (_calibrated_prior, calibration) =
        calibrate_prior_to_physical_rms(&form_prior, magnetic.map(), 0.25)?;
    assert!(calibration.precision_scale.is_finite());

    let residual = QuadraticResidual;
    let nonlinear_term =
        NonlinearResidualTerm::zero("custom_physics", &residual, GaussianNoise::variance(0.1)?)?;
    let _nonlinear_model =
        NonlinearLaplaceModelBuilder::new(form_prior).residual(nonlinear_term)?;
    Ok(())
}
