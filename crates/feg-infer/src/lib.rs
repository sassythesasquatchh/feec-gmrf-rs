pub mod adapters;
pub mod boundary;
pub mod sparse;

pub use sparse::{core_triplet_to_feec_csr, lift_vector_with_layout, reduce_vector_with_layout};

pub mod conditioning;
pub mod diagnostics;
pub mod linear_pde;
#[doc(hidden)]
pub mod metis_ordering;
pub mod model;
pub mod nonlinear;
pub mod physical;
pub mod prior;
pub mod vtk;
