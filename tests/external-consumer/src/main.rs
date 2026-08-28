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
        .recurrence_mass_inverse(RecurrenceMassInversePolicy::RowSumLumped)
        .build()
}

struct ExactOneFormFixture {
    operators: ExactFormOperators,
    exterior_derivative_1: LinearMap,
    flux_reconstruction: LinearMap,
}

fn exact_one_form_fixture() -> Result<ExactOneFormFixture> {
    let lower = FormSpaceMetric::new(
        FormDegree::new(0, 3)?,
        SparseMat::diagonal(1, 1.8),
        BoundaryLayout::identity(1),
    )?;
    let state = FormSpaceMetric::new(
        FormDegree::new(1, 3)?,
        SparseMat::from_rows(
            3,
            &[
                vec![(0, 2.0), (1, 0.25), (2, 0.10)],
                vec![(0, 0.25), (1, 1.7), (2, 0.15)],
                vec![(0, 0.10), (1, 0.15), (2, 1.4)],
            ],
        )
        .map_err(FeecGmrfError::Dimension)?,
        BoundaryLayout::identity(3),
    )?;
    let upper = FormSpaceMetric::new(
        FormDegree::new(2, 3)?,
        SparseMat::from_rows(2, &[vec![(0, 1.5), (1, 0.10)], vec![(0, 0.10), (1, 1.2)]])
            .map_err(FeecGmrfError::Dimension)?,
        BoundaryLayout::identity(2),
    )?;

    // D0 maps into ker(D1), so this is a small nontrivial cochain complex.
    let exterior_derivative_0 = LinearMap::new(
        SparseMat::from_rows(1, &[vec![(0, 1.0)], vec![(0, 1.0)], vec![(0, 1.0)]])
            .map_err(FeecGmrfError::Dimension)?,
    )?;
    let exterior_derivative_1 = LinearMap::new(
        SparseMat::from_rows(3, &[vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0), (2, -1.0)]])
            .map_err(FeecGmrfError::Dimension)?,
    )?;
    let incoming = IncomingFormOperator::new(lower, exterior_derivative_0, None)?;
    let outgoing = OutgoingFormOperator::new(upper, exterior_derivative_1.clone(), None)?;
    let operators = ExactFormOperators::new(
        3,
        state,
        Some(incoming),
        Some(outgoing),
        ExactBoundaryDomain::UserSuppliedCompatibleSubcomplex {
            name: "external three-dimensional one-form fixture".to_string(),
        },
        "external-consumer exact adjacent operators",
    )?;
    let flux_reconstruction = LinearMap::new(
        SparseMat::from_rows(
            2,
            &[
                vec![(0, 1.0), (1, 0.15)],
                vec![(0, -0.20), (1, 0.9)],
                vec![(0, 0.35), (1, -0.25)],
            ],
        )
        .map_err(FeecGmrfError::Dimension)?,
    )?;
    Ok(ExactOneFormFixture {
        operators,
        exterior_derivative_1,
        flux_reconstruction,
    })
}

fn vector_infinity_norm(values: &[f64]) -> f64 {
    values.iter().copied().map(f64::abs).fold(0.0, f64::max)
}

fn dense_infinity_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().copied().map(f64::abs).sum::<f64>())
        .fold(0.0, f64::max)
}

fn sparse_infinity_norm(matrix: &SparseMat) -> f64 {
    let mut row_sums = vec![0.0; matrix.nrows()];
    for (row, _, value) in matrix.triplet_iter() {
        row_sums[row] += value.abs();
    }
    vector_infinity_norm(&row_sums)
}

fn assert_condition_aware_close(
    actual: &[f64],
    expected: &[f64],
    condition_proxy: f64,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let scale = vector_infinity_norm(actual)
        .max(vector_infinity_norm(expected))
        .max(1.0);
    // Both routes use backward-stable direct factorizations, but execute
    // different Cholesky/KKT elimination sequences. This multiplier permits
    // small sparse-backend/platform accumulation while retaining an entirely
    // epsilon- and condition-scaled tolerance with no absolute error floor.
    const FACTORIZATION_BACKWARD_ERROR_MULTIPLIER: f64 = 1_024.0;
    let relative_budget =
        FACTORIZATION_BACKWARD_ERROR_MULTIPLIER * f64::EPSILON * condition_proxy.max(1.0);
    let tolerance = scale * relative_budget;
    assert!(
        error <= tolerance,
        "{context}: infinity-norm error {error:.16e} exceeds condition-aware tolerance \
         {tolerance:.16e} (condition proxy {condition_proxy:.8e})"
    );
}

fn validate_exact_supported_physical_consumer() -> Result<()> {
    for alpha in [MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three] {
        let fixture = exact_one_form_fixture()?;
        let parameters = MaternParameters::new(alpha, 1.15, 0.85)?;
        let exact = ExactMaternPriorBuilder::from_operators(fixture.operators.clone())
            .parameters(parameters)
            .build()?;
        let supported = SupportedMaternPriorBuilder::from_operators(fixture.operators)
            .parameters(parameters)
            .build()?;
        let observation = LinearObservation::new(
            LinearMap::weighted_row(3, &[(0, 0.6), (1, -0.4), (2, 0.8)])?,
            vec![0.35],
            GaussianNoise::variance(0.07)?,
        )?;

        let mut exact_posterior = LinearGaussianModelBuilder::new(exact)
            .observe(observation.clone())?
            .condition()?;
        let supported_posterior = supported.condition(&[observation], &[])?;

        // Dense covariance is deliberate here only because this is a fixed
        // three-coordinate external-consumer oracle. It estimates cond_inf(Q)
        // for the comparison tolerance; neither production route requires it.
        let covariance = exact_posterior.pushforward_covariance(&LinearMap::identity(3))?;
        let condition_proxy =
            sparse_infinity_norm(exact_posterior.precision()) * dense_infinity_norm(&covariance);
        assert!(condition_proxy.is_finite() && condition_proxy >= 1.0);
        assert_condition_aware_close(
            supported_posterior.mean(),
            exact_posterior.mean(),
            condition_proxy,
            &format!("{alpha:?} conditioned state mean"),
        );

        for (index, rhs) in [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.75, -1.25, 0.5],
        ]
        .iter()
        .enumerate()
        {
            let supported_action = supported_posterior.covariance_action(rhs)?;
            let exact_action = exact_posterior.covariance_action(rhs)?;
            assert_condition_aware_close(
                &supported_action,
                &exact_action,
                condition_proxy,
                &format!("{alpha:?} covariance action {index}"),
            );
        }

        let magnetic =
            magnetic_field_map(&fixture.exterior_derivative_1, &fixture.flux_reconstruction)?;
        let exact_physical_covariance = exact_posterior.pushforward_covariance(magnetic.map())?;
        let exact_physical_variances = exact_physical_covariance
            .iter()
            .enumerate()
            .map(|(index, row)| row[index])
            .collect::<Vec<_>>();
        let supported_physical_variances =
            supported_posterior.pushforward_variances(magnetic.map())?;
        assert_condition_aware_close(
            &supported_physical_variances,
            &exact_physical_variances,
            condition_proxy,
            &format!("{alpha:?} magnetic pushforward variances"),
        );
    }
    Ok(())
}

fn run_external_consumer() -> Result<()> {
    let _scalar_prior = operator_prior(0)?;
    let form_prior = operator_prior(1)?;
    let metadata = form_prior
        .matern_metadata()
        .ok_or_else(|| FeecGmrfError::Inference("missing Matérn metadata".to_string()))?;
    assert_eq!(
        metadata.representation(),
        MaternRepresentation::ProvidedCollapsedEliminated
    );
    assert_eq!(
        metadata.lower_adjoint_inverse(),
        InverseUse::Applied(MassInverseKind::EmbeddedInCollapsedHodge)
    );
    assert_eq!(
        metadata.recurrence_inverse(),
        InverseUse::Applied(MassInverseKind::RowSumLumped)
    );

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
    validate_exact_supported_physical_consumer()?;
    Ok(())
}

fn main() -> Result<()> {
    run_external_consumer()
}

#[cfg(test)]
mod tests {
    #[test]
    fn public_external_consumer_workflow() {
        super::run_external_consumer().unwrap();
    }
}
