// Closed genus-2 surface: two solid tori fused by a small direct overlap.
// There is no connecting cylinder.
//
// Surface mesh:
//   gmsh -2 geometries/genus2_torus_touching.geo -format msh41 -o meshes/genus2_torus_touching.msh
// Volume mesh:
//   gmsh -3 geometries/genus2_torus_touching.geo -format msh41 -o meshes/genus2_torus_touching_vol.msh
//
// Topology note:
//   overlap_depth must be positive. Exact tangency is singular, while a small
//   positive overlap makes the Boolean union connected through a small disk-like
//   contact patch. Keep overlap_depth small compared with r so the union remains
//   the connected sum of two solid tori, with genus-2 boundary.

SetFactory("OpenCASCADE");

Mesh.MshFileVersion = 4.1;
Mesh.ElementOrder = 1;
Mesh.RecombineAll = 0;

// Distance from each torus center to the center of its tube.
R = 0.80;

// Radius of each torus tube.
r = 0.28;

// Amount by which the two facing torus tubes overlap.
// 0 gives a singular point contact. Values like 0.04--0.10 give a small,
// clean overlap for these R,r values. Do not make this comparable to r.
overlap_depth = 0.07;

// Distance between the two torus centers.
// Facing tube-center separation is D - 2*R = 2*r - overlap_depth.
D = 2*R + 2*r - overlap_depth;

// Characteristic mesh sizes.
// The original fine values, lc_min = 0.03 and lc_max = 0.11, generate a
// presentation-quality mesh but make the release Hodge eigensolve too slow for
// the checked example. These coarser values preserve the same touching-torus
// geometry while keeping the experiment runnable.
lc_min = 0.13;
lc_max = 0.25;

// Two solid tori. In Gmsh/OpenCASCADE, Torus creates a volume.
// The tori are placed so their facing tubes overlap slightly near the origin.
Torus(1) = {-D/2, 0, 0, R, r};
Torus(2) = { D/2, 0, 0, R, r};

// Move the OpenCASCADE periodic seams away from the overlap patch.
Rotate {{0, 0, 1}, {-D/2, 0, 0}, Pi/2} { Volume{1}; }
Rotate {{0, 0, 1}, { D/2, 0, 0}, Pi/2} { Volume{2}; }

// Fuse into a single genus-2 solid handlebody.
BooleanUnion(100) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

// Boundary surface of the fused handlebody: this is the genus-2 torus.
s() = Boundary{ Volume{100}; };

// Physical groups.
Physical Surface("genus2_surface", 1) = s();
Physical Volume("genus2_solid", 2) = {100};

// Mesh controls.
Mesh.MeshSizeMin = lc_min;
Mesh.MeshSizeMax = lc_max;
Mesh.MeshSizeFromCurvature = 10;
Mesh.Optimize = 1;
Geometry.NumSubEdges = 100;
