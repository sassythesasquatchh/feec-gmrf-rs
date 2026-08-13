// De Rham ladder demonstration volume.
//
// Topology target:
//   b0 = 1 connected component
//   b1 = 3 through-tunnels
//   b2 = 2 enclosed cavities
//   b3 = 0 for a bounded domain with boundary
//
// Generate with:
//   gmsh -3 geometries/derham_ladder_handlebody.geo -format msh41 -o meshes/derham_ladder_handlebody.msh

SetFactory("OpenCASCADE");

Mesh.MshFileVersion = 4.1;
Mesh.ElementOrder = 1;
Mesh.RecombineAll = 0;
Mesh.SaveAll = 0;

If (!Exists(MeshScale))
  MeshScale = 1.0;
EndIf

lc_min = 0.18 * MeshScale;
lc_max = 0.28 * MeshScale;

// Normalized CAD-like solid.
Box(1) = {-1.0, -1.0, -1.0, 2.0, 2.0, 2.0};

// Three mutually disjoint through-tunnels.
tunnel_r0 = 0.18;
tunnel_r1 = 0.17;
tunnel_r2 = 0.16;
Cylinder(10) = {-1.25, -0.65, -0.65, 2.50, 0.00, 0.00, tunnel_r0, 2*Pi};
Cylinder(11) = { 0.65, -1.25,  0.00, 0.00, 2.50, 0.00, tunnel_r1, 2*Pi};
Cylinder(12) = {-0.55,  0.65, -1.25, 0.00, 0.00, 2.50, tunnel_r2, 2*Pi};

// Two interior voids. They do not touch each other, the tunnels, or the outer box.
cavity_r0 = 0.18;
cavity_r1 = 0.17;
Sphere(20) = { 0.15,  0.00, 0.60, cavity_r0};
Sphere(21) = {-0.45, -0.25, 0.15, cavity_r1};

BooleanDifference(100) = { Volume{1}; Delete; }{ Volume{10, 11, 12, 20, 21}; Delete; };

Physical Volume("derham_ladder_domain", 1) = {100};

Mesh.MeshSizeMin = lc_min;
Mesh.MeshSizeMax = lc_max;
Mesh.MeshSizeFromCurvature = 12;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.Optimize = 1;
