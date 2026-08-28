use feec_gmrf::prelude::*;
use manifold::gen::cartesian::CartesianMeshInfo;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const STRUCTURAL_ZERO: f64 = 0.0;
const MAGNITUDE_THRESHOLDS: [f64; 3] = [1.0e-12, 1.0e-10, 1.0e-8];

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReachReport {
    retained_nonlocal_pairs: usize,
    maximum_local_graph_distance: usize,
    distance_histogram: BTreeMap<usize, usize>,
}

#[derive(Debug)]
struct LevelReport {
    cells_per_axis: usize,
    state_dimension: usize,
    structural: ReachReport,
    thresholded: Vec<(f64, ReachReport)>,
}

fn perturbed_square(
    cells_per_axis: usize,
) -> (
    manifold::topology::complex::Complex,
    manifold::geometry::metric::mesh::MeshLengths,
) {
    let mesh = CartesianMeshInfo::new_unit(2, cells_per_axis);
    let (topology, mut coordinates) = mesh.compute_coord_complex();
    let h = 1.0 / cells_per_axis as f64;

    for vertex in 0..coordinates.nvertices() {
        let cartesian = mesh.vertex_cart_idx(vertex);
        if cartesian
            .iter()
            .all(|&index| index != 0 && index != cells_per_axis)
        {
            let x = coordinates.matrix()[(0, vertex)];
            let y = coordinates.matrix()[(1, vertex)];
            coordinates.matrix_mut()[(0, vertex)] += 0.071 * h * (11.0 * x + 7.0 * y + 0.3).sin();
            coordinates.matrix_mut()[(1, vertex)] += 0.053 * h * (5.0 * x - 13.0 * y + 0.2).cos();
        }
    }

    let metric = coordinates.to_edge_lengths(&topology);
    (topology, metric)
}

fn accumulated_entries(matrix: &SparseMat) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for (row, column, value) in matrix.triplet_iter() {
        *entries.entry((row, column)).or_default() += value;
    }
    entries
}

fn row_supports(map: &LinearMap) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new(); map.output_dimension()];
    for (row, column, value) in map.matrix().triplet_iter() {
        if value != STRUCTURAL_ZERO {
            rows[row].push(column);
        }
    }
    for row in &mut rows {
        row.sort_unstable();
        row.dedup();
    }
    rows
}

/// Graph of couplings already exposed by the state metric and outgoing branch.
///
/// The metric-weighted outgoing map is formed through the public composition
/// API. Cross-support within each upper-space row gives the possible support
/// of the local outgoing quadratic form, without reproducing its coefficients.
fn local_state_outgoing_graph(operators: &ExactFormOperators) -> Vec<BTreeSet<usize>> {
    let dimension = operators.state().dimension();
    let mut graph = vec![BTreeSet::new(); dimension];

    for ((row, column), value) in accumulated_entries(operators.state().mass()) {
        if row != column && value != STRUCTURAL_ZERO {
            graph[row].insert(column);
            graph[column].insert(row);
        }
    }

    if let Some(outgoing) = operators.outgoing() {
        let upper_metric = LinearMap::new(outgoing.upper().mass().clone()).unwrap();
        let metric_weighted = upper_metric.compose(outgoing.active_derivative()).unwrap();
        let derivative_rows = row_supports(outgoing.active_derivative());
        let weighted_rows = row_supports(&metric_weighted);
        for (derivative, weighted) in derivative_rows.iter().zip(&weighted_rows) {
            for &left in derivative {
                for &right in weighted {
                    if left != right {
                        graph[left].insert(right);
                        graph[right].insert(left);
                    }
                }
            }
        }
    }

    graph
}

fn shortest_distance(graph: &[BTreeSet<usize>], start: usize, goal: usize) -> Option<usize> {
    let mut distance = vec![usize::MAX; graph.len()];
    distance[start] = 0;
    let mut queue = VecDeque::from([start]);
    while let Some(current) = queue.pop_front() {
        if current == goal {
            return Some(distance[current]);
        }
        for &next in &graph[current] {
            if distance[next] == usize::MAX {
                distance[next] = distance[current] + 1;
                queue.push_back(next);
            }
        }
    }
    None
}

fn reach_report(
    precision: &SparseMat,
    local_graph: &[BTreeSet<usize>],
    relative_threshold: f64,
) -> ReachReport {
    let entries = accumulated_entries(precision);
    let maximum_magnitude = entries.values().copied().map(f64::abs).fold(0.0, f64::max);
    let cutoff = relative_threshold * maximum_magnitude;
    let mut histogram = BTreeMap::new();

    for (&(row, column), &value) in &entries {
        if row >= column || value.abs() <= cutoff || local_graph[row].contains(&column) {
            continue;
        }
        let distance = shortest_distance(local_graph, row, column)
            .expect("the local state/outgoing graph should connect the active FEEC state");
        *histogram.entry(distance).or_default() += 1;
    }

    ReachReport {
        retained_nonlocal_pairs: histogram.values().sum(),
        maximum_local_graph_distance: histogram.keys().next_back().copied().unwrap_or_default(),
        distance_histogram: histogram,
    }
}

fn permute_matrix(matrix: &SparseMat, new_to_old: &[usize]) -> SparseMat {
    let old_to_new = new_to_old
        .iter()
        .copied()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect::<BTreeMap<_, _>>();
    SparseMat::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, column, value)| feg_core::SparseTriplet {
                row: old_to_new[&row],
                col: old_to_new[&column],
                value,
            }),
    )
}

fn permute_columns(matrix: &SparseMat, new_to_old: &[usize]) -> SparseMat {
    let old_to_new = new_to_old
        .iter()
        .copied()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect::<BTreeMap<_, _>>();
    SparseMat::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, column, value)| feg_core::SparseTriplet {
                row,
                col: old_to_new[&column],
                value,
            }),
    )
}

fn relabel_exact_operators(
    operators: &ExactFormOperators,
    new_to_old: &[usize],
) -> ExactFormOperators {
    let old_layout = operators.state().boundary_layout();
    let active_dofs = new_to_old
        .iter()
        .map(|&old| old_layout.active_dofs()[old])
        .collect();
    let state = FormSpaceMetric::new(
        operators.degree(),
        permute_matrix(operators.state().mass(), new_to_old),
        BoundaryLayout::new(
            old_layout.full_dimension(),
            active_dofs,
            old_layout.fixed_dofs().to_vec(),
        )
        .unwrap(),
    )
    .unwrap();
    let incoming = operators.incoming().map(|incoming| {
        IncomingFormOperator::new(
            incoming.lower().clone(),
            incoming
                .active_derivative()
                .select_outputs(new_to_old)
                .unwrap(),
            incoming.full_derivative().cloned(),
        )
        .unwrap()
    });
    let outgoing = operators.outgoing().map(|outgoing| {
        OutgoingFormOperator::new(
            outgoing.upper().clone(),
            LinearMap::new(permute_columns(
                outgoing.active_derivative().matrix(),
                new_to_old,
            ))
            .unwrap(),
            outgoing.full_derivative().cloned(),
        )
        .unwrap()
    });
    ExactFormOperators::new(
        operators.complex_dimension(),
        state,
        incoming,
        outgoing,
        ExactBoundaryDomain::UserSuppliedCompatibleSubcomplex {
            name: "relabelled homogeneous full-relative subcomplex".to_string(),
        },
        "deterministically relabelled FEEC fixture",
    )
    .unwrap()
}

fn assert_matrices_close(actual: &SparseMat, expected: &SparseMat) {
    let actual = accumulated_entries(actual);
    let expected = accumulated_entries(expected);
    let coordinates = actual
        .keys()
        .chain(expected.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    for coordinate in coordinates {
        let actual = actual.get(&coordinate).copied().unwrap_or_default();
        let expected = expected.get(&coordinate).copied().unwrap_or_default();
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= 1.0e-10 * scale,
            "permuted precision entry {coordinate:?} was {actual:.16e}, expected {expected:.16e}"
        );
    }
}

fn permute_graph(graph: &[BTreeSet<usize>], new_to_old: &[usize]) -> Vec<BTreeSet<usize>> {
    let old_to_new = new_to_old
        .iter()
        .copied()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect::<BTreeMap<_, _>>();
    let mut permuted = vec![BTreeSet::new(); graph.len()];
    for (old, neighbours) in graph.iter().enumerate() {
        let new = old_to_new[&old];
        permuted[new].extend(
            neighbours
                .iter()
                .map(|old_neighbour| old_to_new[old_neighbour]),
        );
    }
    permuted
}

fn deterministic_relabelling(dimension: usize) -> Vec<usize> {
    (0..dimension)
        .filter(|index| index % 2 == 1)
        .rev()
        .chain((0..dimension).filter(|index| index % 2 == 0).rev())
        .collect()
}

#[test]
fn exact_one_form_precision_has_mesh_growing_nonlocal_graph_reach() {
    let mut reports = Vec::new();

    for cells_per_axis in 2..=5 {
        let (topology, metric) = perturbed_square(cells_per_axis);
        let operators =
            ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1).unwrap();
        let local_graph = local_state_outgoing_graph(&operators);
        let prior = ExactMaternPriorBuilder::from_operators(operators.clone())
            .parameters(MaternParameters::new(MaternAlpha::One, 1.37, 1.0).unwrap())
            .build()
            .unwrap();

        let structural = reach_report(prior.precision(), &local_graph, STRUCTURAL_ZERO);
        assert!(
            structural.retained_nonlocal_pairs > 0,
            "refinement {cells_per_axis} retained no coupling beyond the local graph"
        );
        assert!(
            structural.maximum_local_graph_distance >= 2,
            "refinement {cells_per_axis} did not reach beyond one local graph edge"
        );

        let relabelling = deterministic_relabelling(prior.dimension());
        let permuted_precision = permute_matrix(prior.precision(), &relabelling);
        let permuted_graph = permute_graph(&local_graph, &relabelling);
        let relabelled_prior = ExactMaternPriorBuilder::from_operators(relabel_exact_operators(
            &operators,
            &relabelling,
        ))
        .parameters(MaternParameters::new(MaternAlpha::One, 1.37, 1.0).unwrap())
        .build()
        .unwrap();
        assert_matrices_close(relabelled_prior.precision(), &permuted_precision);
        assert_eq!(
            reach_report(&permuted_precision, &permuted_graph, STRUCTURAL_ZERO),
            structural,
            "graph diagnostics must be invariant under deterministic coefficient relabelling"
        );

        reports.push(LevelReport {
            cells_per_axis,
            state_dimension: prior.dimension(),
            structural,
            thresholded: MAGNITUDE_THRESHOLDS
                .into_iter()
                .map(|threshold| {
                    (
                        threshold,
                        reach_report(prior.precision(), &local_graph, threshold),
                    )
                })
                .collect(),
        });
    }

    for report in &reports {
        eprintln!(
            "cells={} state={} structural_pairs={} structural_reach={}",
            report.cells_per_axis,
            report.state_dimension,
            report.structural.retained_nonlocal_pairs,
            report.structural.maximum_local_graph_distance
        );
        for (threshold, thresholded) in &report.thresholded {
            assert!(
                thresholded.retained_nonlocal_pairs <= report.structural.retained_nonlocal_pairs
            );
            eprintln!(
                "  relative_threshold={threshold:.0e} pairs={} reach={}",
                thresholded.retained_nonlocal_pairs, thresholded.maximum_local_graph_distance
            );
        }
    }
    assert!(reports
        .windows(2)
        .all(|levels| levels[1].state_dimension > levels[0].state_dimension));
    assert!(
        reports
            .windows(2)
            .all(|levels| levels[1].structural.maximum_local_graph_distance
                >= levels[0].structural.maximum_local_graph_distance),
        "structurally retained graph reach should not decrease across these refinements"
    );
    assert!(
        reports
            .last()
            .unwrap()
            .structural
            .maximum_local_graph_distance
            > reports
                .first()
                .unwrap()
                .structural
                .maximum_local_graph_distance,
        "the exact marginal graph reach should grow as the mesh is refined"
    );
}
