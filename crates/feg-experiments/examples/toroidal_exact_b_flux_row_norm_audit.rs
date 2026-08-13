use feg_case_studies::toroidal_inductor::{
    toroidal_exact_b_field_recovery_observation_indices, toroidal_exact_b_surface_flux_row_norms,
    ToroidalExactBRecoveryConfig,
};
use std::{collections::BTreeSet, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = ToroidalExactBRecoveryConfig::default();
    config.surface_flux_azimuth_count = 16;
    config.observation_noise_std = 1.0e-9;
    let rows = toroidal_exact_b_surface_flux_row_norms(&config)?;
    let train6 = training_set(6)?;
    let train12 = training_set(12)?;
    let train24 = training_set(24)?;
    let train36 = training_set(36)?;

    println!(
        "row,name,nnz,l2_norm,max_abs_entry,max_diag_contribution,train6,train12,train24,train36,heldout"
    );
    for row in &rows {
        let heldout = toroidal_exact_b_field_recovery_observation_indices(16, 36)?
            .heldout_indices
            .contains(&row.row_index);
        println!(
            "{},{},{},{:.16e},{:.16e},{:.16e},{},{},{},{},{}",
            row.row_index,
            row.name,
            row.nnz,
            row.l2_norm,
            row.max_abs_entry,
            row.max_diagonal_contribution,
            train6.contains(&row.row_index),
            train12.contains(&row.row_index),
            train24.contains(&row.row_index),
            train36.contains(&row.row_index),
            heldout
        );
    }
    Ok(())
}

fn training_set(training_count: usize) -> Result<BTreeSet<usize>, Box<dyn Error>> {
    Ok(
        toroidal_exact_b_field_recovery_observation_indices(16, training_count)?
            .training_indices
            .into_iter()
            .collect(),
    )
}
