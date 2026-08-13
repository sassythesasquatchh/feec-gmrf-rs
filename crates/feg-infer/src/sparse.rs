use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use feg_core::{SparseTriplet, SparseTripletMatrix};
use formoniq::reduction::{DofLayout, PrescribedDof};
use gmrf_core::types::{CooMatrix as GmrfCoo, SparseMatrix as GmrfSparse, Vector as GmrfVector};
use gmrf_core::SparseRowOperator;
use std::collections::BTreeMap;

pub fn feec_csr_to_gmrf(mat: &FeecCsr) -> GmrfSparse {
    let mut coo = GmrfCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        coo.push(row, col, *value);
    }
    GmrfSparse::from(&coo)
}

/// Convert the integration sparse contract to the GMRF backend.
///
/// This adapter deliberately lives in the integration crate so `gmrf-core`
/// remains independent of FEEC and `feg-core`.
pub fn core_triplet_to_gmrf(matrix: &SparseTripletMatrix) -> GmrfSparse {
    feec_csr_to_gmrf(&core_triplet_to_feec_csr(matrix))
}

/// Convert a GMRF sparse matrix to the integration sparse contract.
pub fn gmrf_sparse_to_core_triplet(matrix: &GmrfSparse) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
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

pub fn feec_vec_to_gmrf(vec: &FeecVector) -> GmrfVector {
    GmrfVector::from_vec(vec.iter().copied().collect())
}

pub fn gmrf_vec_to_feec(vec: &GmrfVector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}

pub fn core_triplet_to_feec_csr(matrix: &SparseTripletMatrix) -> FeecCsr {
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, value);
    }
    FeecCsr::from(&coo)
}

pub fn feec_csr_to_core_triplet(matrix: &FeecCsr) -> SparseTripletMatrix {
    let mut triplet = SparseTripletMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            triplet.push(row, col, *value);
        }
    }
    triplet
}

pub fn identity_feec_csr(dimension: usize, value: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(dimension, dimension);
    if value != 0.0 {
        for index in 0..dimension {
            coo.push(index, index, value);
        }
    }
    FeecCsr::from(&coo)
}

pub fn feec_csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            rows[row].push((col, *value));
        }
    }
    rows
}

pub fn sparse_row_operator_from_feec_csr(matrix: &FeecCsr) -> Result<SparseRowOperator, String> {
    SparseRowOperator::from_sparse_matrix(&feec_csr_to_gmrf(matrix)).map_err(|err| err.to_string())
}

pub fn sparse_row_operator_from_feec_csr_with_tolerance(
    matrix: &FeecCsr,
    drop_tolerance: f64,
) -> Result<SparseRowOperator, String> {
    if !drop_tolerance.is_finite() {
        return Err("sparse row conversion drop tolerance must be finite".to_string());
    }
    let tol = drop_tolerance.abs();
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if value.abs() > tol {
            rows[row].push((col, *value));
        }
    }
    SparseRowOperator::new(matrix.ncols(), rows).map_err(|err| err.to_string())
}

pub fn sparse_row_operator_from_feec_dense(
    matrix: &FeecMatrix,
    drop_tolerance: f64,
) -> Result<SparseRowOperator, String> {
    let dense = gmrf_core::types::DenseMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        matrix[(i, j)]
    });
    SparseRowOperator::from_dense_matrix(&dense, drop_tolerance).map_err(|err| err.to_string())
}

pub fn sparse_row_operator_apply_feec(
    operator: &SparseRowOperator,
    input: &FeecVector,
) -> Result<FeecVector, String> {
    let gmrf_input = feec_vec_to_gmrf(input);
    operator
        .apply(&gmrf_input)
        .map(|value| gmrf_vec_to_feec(&value))
        .map_err(|err| err.to_string())
}

pub fn sparse_row_to_triplet(ncols: usize, row: &[(usize, f64)]) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        1,
        ncols,
        row.iter()
            .copied()
            .filter(|(_, value)| *value != 0.0)
            .map(|(col, value)| SparseTriplet { row: 0, col, value }),
    )
}

pub fn sparse_rows_to_triplet(
    nrows: usize,
    ncols: usize,
    rows: &[Vec<(usize, f64)>],
) -> SparseTripletMatrix {
    let mut matrix = SparseTripletMatrix::new(nrows, ncols);
    for (row_index, row) in rows.iter().enumerate() {
        for (col, value) in row {
            if *value != 0.0 {
                matrix.push(row_index, *col, *value);
            }
        }
    }
    matrix
}

pub fn sparse_row_operator_to_triplet(
    operator: &SparseRowOperator,
) -> Result<SparseTripletMatrix, String> {
    SparseTripletMatrix::from_rows(operator.ncols, &operator.rows)
}

pub fn triplet_to_sparse_row_operator(
    matrix: &SparseTripletMatrix,
) -> Result<SparseRowOperator, String> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if value != 0.0 {
            rows[row].push((col, value));
        }
    }
    SparseRowOperator::new(matrix.ncols(), rows).map_err(|err| err.to_string())
}

pub fn select_triplet_rows(
    matrix: &SparseTripletMatrix,
    rows: &[usize],
) -> Result<SparseTripletMatrix, String> {
    matrix.select_rows(rows)
}

pub fn restrict_triplet_columns_and_fold_fixed(
    full_operator: &SparseTripletMatrix,
    bias: &[f64],
    layout: &DofLayout,
) -> Result<(SparseTripletMatrix, Vec<f64>), String> {
    if full_operator.ncols() != layout.full_dimension {
        return Err(format!(
            "full operator column count {} does not match layout dimension {}",
            full_operator.ncols(),
            layout.full_dimension
        ));
    }
    if full_operator.nrows() != bias.len() {
        return Err(format!(
            "bias length {} does not match operator row count {}",
            bias.len(),
            full_operator.nrows()
        ));
    }

    let reduced_map = reduced_index_map(layout);
    let fixed_by_full = layout
        .prescribed_dofs
        .iter()
        .map(|entry| (entry.index, entry.value))
        .collect::<BTreeMap<_, _>>();
    let mut reduced = SparseTripletMatrix::new(full_operator.nrows(), layout.reduced_dimension());
    let mut folded_bias = bias.to_vec();
    for (row, col, value) in full_operator.triplet_iter() {
        if let Some(reduced_col) = reduced_map[col] {
            reduced.push(row, reduced_col, value);
        } else if let Some(fixed_value) = fixed_by_full.get(&col) {
            folded_bias[row] += value * fixed_value;
        }
    }
    Ok((reduced, folded_bias))
}

pub fn restrict_sparse_row_operator_columns_and_fold_fixed(
    full_operator: &SparseRowOperator,
    bias: &[f64],
    layout: &DofLayout,
) -> Result<(SparseRowOperator, Vec<f64>), String> {
    if full_operator.ncols != layout.full_dimension {
        return Err(format!(
            "full operator column count {} does not match layout dimension {}",
            full_operator.ncols, layout.full_dimension
        ));
    }
    if full_operator.nrows() != bias.len() {
        return Err(format!(
            "bias length {} does not match operator row count {}",
            bias.len(),
            full_operator.nrows()
        ));
    }

    let reduced_map = reduced_index_map(layout);
    let fixed_by_full = layout
        .prescribed_dofs
        .iter()
        .map(|entry| (entry.index, entry.value))
        .collect::<BTreeMap<_, _>>();
    let mut rows = vec![Vec::new(); full_operator.nrows()];
    let mut folded_bias = bias.to_vec();
    for (row_index, row) in full_operator.rows.iter().enumerate() {
        for (col, value) in row {
            if let Some(reduced_col) = reduced_map[*col] {
                rows[row_index].push((reduced_col, *value));
            } else if let Some(fixed_value) = fixed_by_full.get(col) {
                folded_bias[row_index] += *value * fixed_value;
            }
        }
    }
    SparseRowOperator::new(layout.reduced_dimension(), rows)
        .map(|operator| (operator, folded_bias))
        .map_err(|err| err.to_string())
}

pub fn restrict_sparse_row_operator_columns(
    full_operator: &SparseRowOperator,
    layout: &DofLayout,
) -> Result<SparseRowOperator, String> {
    let zero_bias = vec![0.0; full_operator.nrows()];
    let (operator, bias) =
        restrict_sparse_row_operator_columns_and_fold_fixed(full_operator, &zero_bias, layout)?;
    if bias.iter().any(|value| value.abs() > 0.0) {
        return Err("column restriction would produce a nonzero fixed-dof bias".to_string());
    }
    Ok(operator)
}

pub fn select_square_triplet_rows_cols(
    matrix: &SparseTripletMatrix,
    rows: &[usize],
) -> Result<SparseTripletMatrix, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "row/column selection requires a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut index_by_row = BTreeMap::new();
    for (selected_index, row) in rows.iter().copied().enumerate() {
        if row >= matrix.nrows() {
            return Err(format!(
                "selected row {row} is outside matrix dimension {}",
                matrix.nrows()
            ));
        }
        if index_by_row.insert(row, selected_index).is_some() {
            return Err(format!("selected row {row} appears more than once"));
        }
    }
    Ok(SparseTripletMatrix::from_triplets(
        rows.len(),
        rows.len(),
        matrix.triplet_iter().filter_map(|(row, col, value)| {
            let selected_row = index_by_row.get(&row)?;
            let selected_col = index_by_row.get(&col)?;
            Some(SparseTriplet {
                row: *selected_row,
                col: *selected_col,
                value,
            })
        }),
    ))
}

pub fn apply_sparse_row(row: &[(usize, f64)], values: &[f64]) -> Result<f64, String> {
    let mut output = 0.0;
    for (col, weight) in row {
        let Some(value) = values.get(*col) else {
            return Err("sparse row references a column outside the vector".to_string());
        };
        output += *weight * *value;
    }
    Ok(output)
}

pub fn reduce_vector_with_layout(
    layout: &DofLayout,
    full: &FeecVector,
) -> Result<FeecVector, String> {
    if full.len() != layout.full_dimension {
        return Err(format!(
            "full vector length {} does not match layout dimension {}",
            full.len(),
            layout.full_dimension
        ));
    }
    Ok(FeecVector::from_iterator(
        layout.reduced_dimension(),
        layout.active_dofs.iter().map(|&index| full[index]),
    ))
}

pub fn lift_vector_with_layout(
    layout: &DofLayout,
    reduced: &FeecVector,
) -> Result<FeecVector, String> {
    if reduced.len() != layout.reduced_dimension() {
        return Err(format!(
            "reduced vector length {} does not match layout reduced dimension {}",
            reduced.len(),
            layout.reduced_dimension()
        ));
    }
    let mut full = FeecVector::zeros(layout.full_dimension);
    for (reduced_index, full_index) in layout.active_dofs.iter().copied().enumerate() {
        full[full_index] = reduced[reduced_index];
    }
    for PrescribedDof { index, value } in &layout.prescribed_dofs {
        full[*index] = *value;
    }
    Ok(full)
}

pub fn reduced_index_map(layout: &DofLayout) -> Vec<Option<usize>> {
    let mut map = vec![None; layout.full_dimension];
    for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
        map[full] = Some(reduced);
    }
    map
}

pub fn restrict_columns_and_fold_fixed(
    full_block: &FeecCsr,
    bias: &FeecVector,
    layout: &DofLayout,
) -> Result<(FeecCsr, FeecVector), String> {
    if full_block.ncols() != layout.full_dimension {
        return Err(format!(
            "full observation block column count {} does not match layout dimension {}",
            full_block.ncols(),
            layout.full_dimension
        ));
    }
    if full_block.nrows() != bias.len() {
        return Err(format!(
            "bias length {} does not match observation row count {}",
            bias.len(),
            full_block.nrows()
        ));
    }

    let mut reduced = FeecCoo::new(full_block.nrows(), layout.reduced_dimension());
    let mut reduced_bias = bias.clone();
    let reduced_map = reduced_index_map(layout);
    for (row, col, value) in full_block.triplet_iter() {
        if let Some(reduced_col) = reduced_map[col] {
            reduced.push(row, reduced_col, *value);
        } else if let Some(fixed) = layout
            .prescribed_dofs
            .iter()
            .find(|entry| entry.index == col)
        {
            reduced_bias[row] += *value * fixed.value;
        }
    }
    Ok((FeecCsr::from(&reduced), reduced_bias))
}

pub fn feec_csr_to_dense(mat: &FeecCsr) -> FeecMatrix {
    let mut dense = FeecMatrix::zeros(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        dense[(row, col)] += *value;
    }
    dense
}

pub fn dense_to_feec_csr(mat: &FeecMatrix, drop_tolerance: f64) -> FeecCsr {
    let tol = drop_tolerance.abs();
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let value = mat[(i, j)];
            if value.abs() > tol {
                coo.push(i, j, value);
            }
        }
    }
    FeecCsr::from(&coo)
}

pub fn block_diag_feec_csr(blocks: &[&FeecCsr]) -> FeecCsr {
    let total_rows = blocks.iter().map(|block| block.nrows()).sum();
    let total_cols = blocks.iter().map(|block| block.ncols()).sum();
    let mut coo = FeecCoo::new(total_rows, total_cols);
    let mut row_offset = 0;
    let mut col_offset = 0;
    for block in blocks {
        for (row, col, value) in block.triplet_iter() {
            if *value != 0.0 {
                coo.push(row_offset + row, col_offset + col, *value);
            }
        }
        row_offset += block.nrows();
        col_offset += block.ncols();
    }
    FeecCsr::from(&coo)
}

pub fn hstack_feec_csr(blocks: &[&FeecCsr]) -> Result<FeecCsr, String> {
    let Some(first) = blocks.first() else {
        return Ok(FeecCsr::from(&FeecCoo::new(0, 0)));
    };
    let row_count = first.nrows();
    if let Some(block) = blocks.iter().find(|block| block.nrows() != row_count) {
        return Err(format!(
            "cannot horizontally stack blocks with row counts {} and {}",
            row_count,
            block.nrows()
        ));
    }

    let total_cols = blocks.iter().map(|block| block.ncols()).sum();
    let mut coo = FeecCoo::new(row_count, total_cols);
    let mut col_offset = 0;
    for block in blocks {
        for (row, col, value) in block.triplet_iter() {
            if *value != 0.0 {
                coo.push(row, col_offset + col, *value);
            }
        }
        col_offset += block.ncols();
    }
    Ok(FeecCsr::from(&coo))
}

pub fn restrict_rows_with_layout(matrix: &FeecCsr, layout: &DofLayout) -> Result<FeecCsr, String> {
    if matrix.nrows() != layout.full_dimension {
        return Err(format!(
            "matrix row count {} does not match layout full dimension {}",
            matrix.nrows(),
            layout.full_dimension
        ));
    }

    let mut row_map = vec![None; layout.full_dimension];
    for (reduced_row, full_row) in layout.active_dofs.iter().copied().enumerate() {
        row_map[full_row] = Some(reduced_row);
    }

    let mut coo = FeecCoo::new(layout.reduced_dimension(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if let Some(reduced_row) = row_map[row] {
            if *value != 0.0 {
                coo.push(reduced_row, col, *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

pub fn restrict_square_with_layout(
    matrix: &FeecCsr,
    layout: &DofLayout,
) -> Result<FeecCsr, String> {
    if matrix.nrows() != layout.full_dimension || matrix.ncols() != layout.full_dimension {
        return Err(format!(
            "square matrix dimensions {}x{} do not match layout dimension {}",
            matrix.nrows(),
            matrix.ncols(),
            layout.full_dimension
        ));
    }

    let map = reduced_index_map(layout);
    let mut coo = FeecCoo::new(layout.reduced_dimension(), layout.reduced_dimension());
    for (row, col, value) in matrix.triplet_iter() {
        if let (Some(row), Some(col)) = (map[row], map[col]) {
            if *value != 0.0 {
                coo.push(row, col, *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

pub fn symmetrize_feec_csr(matrix: &FeecCsr) -> FeecCsr {
    let mut values = BTreeMap::<(usize, usize), f64>::new();
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            *values.entry((row, col)).or_insert(0.0) += *value;
        } else {
            *values.entry((row, col)).or_insert(0.0) += 0.5 * *value;
            *values.entry((col, row)).or_insert(0.0) += 0.5 * *value;
        }
    }
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for ((row, col), value) in values {
        if value != 0.0 {
            coo.push(row, col, value);
        }
    }
    FeecCsr::from(&coo)
}

pub fn identity_triplet_matrix(size: usize, value: f64) -> SparseTripletMatrix {
    let mut matrix = SparseTripletMatrix::new(size, size);
    if value != 0.0 {
        for index in 0..size {
            matrix.push(index, index, value);
        }
    }
    matrix
}

pub fn lumped_diag(mat: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, _col, value) in mat.triplet_iter() {
        diag[row] += *value;
    }
    diag
}

pub fn matrix_diag(mat: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, col, value) in mat.triplet_iter() {
        if row == col {
            diag[row] += *value;
        }
    }
    diag
}

pub fn invert_diag(diag: &[f64]) -> Vec<f64> {
    let eps = 1e-12;
    diag.iter()
        .map(|v| if v.abs() < eps { 0.0 } else { 1.0 / v })
        .collect()
}

pub fn diag_matrix(diag: &[f64]) -> FeecCsr {
    let mut coo = FeecCoo::new(diag.len(), diag.len());
    for (i, value) in diag.iter().copied().enumerate() {
        if value != 0.0 {
            coo.push(i, i, value);
        }
    }
    FeecCsr::from(&coo)
}

pub fn add_feec_diagonal_shift(matrix: &FeecCsr, shift: f64) -> Result<FeecCsr, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "diagonal shift requires a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if !shift.is_finite() {
        return Err("diagonal shift must be finite".to_string());
    }

    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            coo.push(row, col, *value);
        }
    }
    if shift != 0.0 {
        for index in 0..matrix.nrows() {
            coo.push(index, index, shift);
        }
    }
    Ok(FeecCsr::from(&coo))
}

pub fn stabilize_feec_diagonal(matrix: &FeecCsr, floor: f64) -> Result<FeecCsr, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "diagonal stabilization requires a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if !floor.is_finite() {
        return Err("diagonal floor must be finite".to_string());
    }

    let diagonal = matrix_diag(matrix);
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            coo.push(row, col, *value);
        }
    }
    for (index, value) in diagonal.into_iter().enumerate() {
        let shift = floor - value;
        if shift > 0.0 {
            coo.push(index, index, shift);
        }
    }
    Ok(FeecCsr::from(&coo))
}

pub fn scale_matrix(mat: &FeecCsr, scale: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        let scaled = *value * scale;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

pub fn add_sparse(a: &FeecCsr, b: &FeecCsr) -> FeecCsr {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());

    let mut coo = FeecCoo::new(a.nrows(), a.ncols());
    for (row, col, value) in a.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (row, col, value) in b.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn feec_csr_dense_roundtrip_preserves_entries() {
        let mut coo = FeecCoo::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 2, -1.5);
        coo.push(1, 1, 3.0);
        coo.push(2, 0, 0.25);
        let csr = FeecCsr::from(&coo);

        let dense = feec_csr_to_dense(&csr);
        let roundtrip = dense_to_feec_csr(&dense, 0.0);

        assert_eq!(csr.nrows(), roundtrip.nrows());
        assert_eq!(csr.ncols(), roundtrip.ncols());
        let dense_roundtrip = feec_csr_to_dense(&roundtrip);
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert!(approx_eq(dense[(i, j)], dense_roundtrip[(i, j)]));
            }
        }
    }

    #[test]
    fn dense_to_feec_csr_drops_small_entries() {
        let dense = FeecMatrix::from_row_slice(2, 2, &[1.0, 1e-16, -1e-18, 2.0]);
        let csr = dense_to_feec_csr(&dense, 1e-14);
        let roundtrip = feec_csr_to_dense(&csr);

        assert!(approx_eq(roundtrip[(0, 0)], 1.0));
        assert!(approx_eq(roundtrip[(1, 1)], 2.0));
        assert!(approx_eq(roundtrip[(0, 1)], 0.0));
        assert!(approx_eq(roundtrip[(1, 0)], 0.0));
    }

    #[test]
    fn triplet_column_restriction_folds_fixed_values_into_bias() {
        let layout = DofLayout::new(
            3,
            vec![0, 2],
            vec![PrescribedDof {
                index: 1,
                value: 4.0,
            }],
        );
        let mut full = SparseTripletMatrix::new(1, 3);
        full.push(0, 0, 2.0);
        full.push(0, 1, 3.0);
        full.push(0, 2, 5.0);

        let (reduced, bias) =
            restrict_triplet_columns_and_fold_fixed(&full, &[7.0], &layout).unwrap();
        assert_eq!(bias, vec![19.0]);
        let triplets = reduced.triplet_iter().collect::<Vec<_>>();
        assert!(triplets.contains(&(0, 0, 2.0)));
        assert!(triplets.contains(&(0, 1, 5.0)));
    }

    #[test]
    fn feec_csr_rows_preserve_sparse_row_structure() {
        let mut coo = FeecCoo::new(2, 3);
        coo.push(0, 2, 1.5);
        coo.push(1, 0, -2.0);
        coo.push(1, 2, 0.25);
        let csr = FeecCsr::from(&coo);

        let rows = feec_csr_rows(&csr);

        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains(&(2, 1.5)));
        assert!(rows[1].contains(&(0, -2.0)));
        assert!(rows[1].contains(&(2, 0.25)));
    }

    #[test]
    fn sparse_row_operator_converts_to_triplets_and_selects_rows() {
        let operator = SparseRowOperator::new(
            4,
            vec![vec![(0, 1.0), (2, -1.0)], vec![(1, 3.0)], vec![(3, 2.5)]],
        )
        .unwrap();

        let triplet = sparse_row_operator_to_triplet(&operator).unwrap();
        let selected = select_triplet_rows(&triplet, &[2, 0]).unwrap();

        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected.ncols(), 4);
        let entries = selected.triplet_iter().collect::<Vec<_>>();
        assert!(entries.contains(&(0, 3, 2.5)));
        assert!(entries.contains(&(1, 0, 1.0)));
        assert!(entries.contains(&(1, 2, -1.0)));
    }

    #[test]
    fn sparse_row_operator_restriction_folds_fixed_values() {
        let layout = DofLayout::new(
            4,
            vec![0, 2],
            vec![
                PrescribedDof {
                    index: 1,
                    value: 5.0,
                },
                PrescribedDof {
                    index: 3,
                    value: -2.0,
                },
            ],
        );
        let operator =
            SparseRowOperator::new(4, vec![vec![(0, 1.0), (1, 2.0), (2, 3.0)], vec![(3, -4.0)]])
                .unwrap();

        let (restricted, bias) =
            restrict_sparse_row_operator_columns_and_fold_fixed(&operator, &[7.0, 11.0], &layout)
                .unwrap();

        assert_eq!(restricted.ncols, 2);
        assert_eq!(restricted.rows[0], vec![(0, 1.0), (1, 3.0)]);
        assert_eq!(restricted.rows[1], Vec::<(usize, f64)>::new());
        assert_eq!(bias, vec![17.0, 19.0]);
    }

    #[test]
    fn square_triplet_row_col_selection_remaps_entries() {
        let mut matrix = SparseTripletMatrix::new(3, 3);
        matrix.push(0, 0, 1.0);
        matrix.push(0, 2, 2.0);
        matrix.push(1, 1, 3.0);
        matrix.push(2, 0, 4.0);
        matrix.push(2, 2, 5.0);

        let selected = select_square_triplet_rows_cols(&matrix, &[2, 0]).unwrap();
        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected.ncols(), 2);
        let triplets = selected.triplet_iter().collect::<Vec<_>>();
        assert!(triplets.contains(&(0, 0, 5.0)));
        assert!(triplets.contains(&(0, 1, 4.0)));
        assert!(triplets.contains(&(1, 0, 2.0)));
        assert!(triplets.contains(&(1, 1, 1.0)));
    }

    #[test]
    fn feec_diagonal_shift_and_stabilization_adjust_only_diagonal() {
        let mut coo = FeecCoo::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(0, 1, 0.5);
        coo.push(1, 1, -1.0);
        let matrix = FeecCsr::from(&coo);

        let shifted = add_feec_diagonal_shift(&matrix, 2.0).unwrap();
        let shifted_dense = feec_csr_to_dense(&shifted);
        assert!(approx_eq(shifted_dense[(0, 0)], 3.0));
        assert!(approx_eq(shifted_dense[(0, 1)], 0.5));
        assert!(approx_eq(shifted_dense[(1, 1)], 1.0));

        let stabilized = stabilize_feec_diagonal(&matrix, 0.25).unwrap();
        let stabilized_dense = feec_csr_to_dense(&stabilized);
        assert!(approx_eq(stabilized_dense[(0, 0)], 1.0));
        assert!(approx_eq(stabilized_dense[(0, 1)], 0.5));
        assert!(approx_eq(stabilized_dense[(1, 1)], 0.25));
    }
}
