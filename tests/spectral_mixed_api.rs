#![cfg(feature = "external-solvers")]

use feec_gmrf::operator::ExactFormOperators;
use feec_gmrf::spectral::{
    partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative, GpError,
    PartialMixedGeneralizedEigensolverOptions,
};
use manifold::gen::cartesian::CartesianMeshInfo;

#[test]
fn mixed_spectral_facade_derives_exact_operator_boundary_ordering() {
    let mesh = CartesianMeshInfo::new_unit(2, 2);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let exact =
        ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1).unwrap();
    let relative =
        formoniq::relative::assemble_whitney_full_relative_operators(&topology, &metric, 1)
            .unwrap();

    assert_eq!(
        exact.state().boundary_layout().active_dofs(),
        relative.state().layout().active_dofs
    );
    assert_eq!(
        exact
            .incoming()
            .unwrap()
            .lower()
            .boundary_layout()
            .active_dofs(),
        relative.lower().unwrap().layout().active_dofs
    );

    let error = partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
        &topology,
        &metric,
        1,
        PartialMixedGeneralizedEigensolverOptions::new(0),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        GpError::InvalidPartialGeneralizedEigenbasisRank { requested: 0, .. }
    ));
}
