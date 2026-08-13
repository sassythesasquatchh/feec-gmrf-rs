SetFactory("OpenCASCADE");

If (!Exists(FullDomain))
  FullDomain = 0;
EndIf
If (!Exists(MeshScale))
  MeshScale = 1.0;
EndIf
If (!Exists(SteelGap))
  SteelGap = 0.0005;
EndIf

// TEAM 13 dimensions from the NGSolve reference, in meters.
air_x = 0.5;
air_y = 0.5;
air_z = 0.5;
coil_outer = 0.200;
coil_inner = 0.150;
coil_height = 0.100;
coil_corner_outer = 0.050;
coil_corner_inner = 0.025;
sheet_height = 0.1264;
sheet_thickness = 0.0032;
c_gap = sheet_thickness + 2 * SteelGap;

z_min = -air_z / 2;
solid_z_min = -coil_height / 2;
sheet_z_min = -sheet_height / 2;
If (FullDomain == 0)
  z_min = 0.0;
  solid_z_min = 0.0;
  sheet_z_min = 0.0;
EndIf
z_len = air_z / 2 - z_min;
solid_z_len = coil_height / 2 - solid_z_min;
sheet_z_len = sheet_height / 2 - sheet_z_min;

Mesh.ElementOrder = 1;
Mesh.RecombineAll = 0;
Mesh.MshFileVersion = 4.1;
Mesh.SaveAll = 1;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeMin = 0.00005 * MeshScale;
Mesh.MeshSizeMax = 0.045 * MeshScale;

// Outer truncation box.
Box(1) = {-air_x / 2, -air_y / 2, z_min, air_x, air_y, z_len};

// Coil racetrack: four straight bricks and four annular quadrant corners.
Box(10) = {-0.050,  0.075, solid_z_min, 0.100, 0.025, solid_z_len}; // back
Box(11) = {-0.050, -0.100, solid_z_min, 0.100, 0.025, solid_z_len}; // front
Box(12) = {-0.100, -0.050, solid_z_min, 0.025, 0.100, solid_z_len}; // left
Box(13) = { 0.075, -0.050, solid_z_min, 0.025, 0.100, solid_z_len}; // right

Cylinder(20) = { 0.050,  0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_outer, 2*Pi};
Cylinder(21) = { 0.050,  0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_inner, 2*Pi};
ann20[] = BooleanDifference{ Volume{20}; Delete; }{ Volume{21}; Delete; };
Box(22) = { 0.050,  0.050, solid_z_min, 0.050, 0.050, solid_z_len};
crb[] = BooleanIntersection{ Volume{ann20[]}; Delete; }{ Volume{22}; Delete; };

Cylinder(30) = {-0.050,  0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_outer, 2*Pi};
Cylinder(31) = {-0.050,  0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_inner, 2*Pi};
ann30[] = BooleanDifference{ Volume{30}; Delete; }{ Volume{31}; Delete; };
Box(32) = {-0.100,  0.050, solid_z_min, 0.050, 0.050, solid_z_len};
clb[] = BooleanIntersection{ Volume{ann30[]}; Delete; }{ Volume{32}; Delete; };

Cylinder(40) = {-0.050, -0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_outer, 2*Pi};
Cylinder(41) = {-0.050, -0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_inner, 2*Pi};
ann40[] = BooleanDifference{ Volume{40}; Delete; }{ Volume{41}; Delete; };
Box(42) = {-0.100, -0.100, solid_z_min, 0.050, 0.050, solid_z_len};
clf[] = BooleanIntersection{ Volume{ann40[]}; Delete; }{ Volume{42}; Delete; };

Cylinder(50) = { 0.050, -0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_outer, 2*Pi};
Cylinder(51) = { 0.050, -0.050, solid_z_min, 0, 0, solid_z_len, coil_corner_inner, 2*Pi};
ann50[] = BooleanDifference{ Volume{50}; Delete; }{ Volume{51}; Delete; };
Box(52) = { 0.050, -0.100, solid_z_min, 0.050, 0.050, solid_z_len};
crf[] = BooleanIntersection{ Volume{ann50[]}; Delete; }{ Volume{52}; Delete; };

coilVols[] = {10, 11, 12, 13, crb[], clb[], clf[], crf[]};

// Iron sheets: vertical sheet plus two C-shaped laminated channels.
Box(60) = {-sheet_thickness / 2, -0.025, sheet_z_min, sheet_thickness, 0.050, sheet_z_len};

Box(70) = {-c_gap / 2 - 0.120 - sheet_thickness, -0.065, sheet_z_min,
           0.120 + sheet_thickness, 0.050, sheet_z_len};
Box(71) = {-c_gap / 2 - 0.120, -0.065, sheet_z_min + sheet_thickness,
           0.120, 0.050, sheet_z_len - 2 * sheet_thickness};
leftC[] = BooleanDifference{ Volume{70}; Delete; }{ Volume{71}; Delete; };

Box(80) = {c_gap / 2, 0.015, sheet_z_min,
           0.120 + sheet_thickness, 0.050, sheet_z_len};
Box(81) = {c_gap / 2, 0.015, sheet_z_min + sheet_thickness,
           0.120, 0.050, sheet_z_len - 2 * sheet_thickness};
rightC[] = BooleanDifference{ Volume{80}; Delete; }{ Volume{81}; Delete; };

ironVols[] = {60, leftC[], rightC[]};

frags[] = BooleanFragments{ Volume{1}; Delete; }{ Volume{coilVols[], ironVols[]}; Delete; };

Physical Volume("team13_domain") = {frags[]};

// The thin 3.2 mm steel sheets and the 4.2 mm channel gap are too sharp for a
// purely global mesh size.  Refine the central TEAM 13 feature region in the
// spirit of the NGSolve/mufem meshes, while keeping the far air box coarse.
feature_h = 0.00040 * MeshScale;
gap_h = 0.00020 * MeshScale;
far_h = 0.020 * MeshScale;

Field[1] = Box;
Field[1].VIn = feature_h;
Field[1].VOut = far_h;
Field[1].XMin = -0.130;
Field[1].XMax =  0.130;
Field[1].YMin = -0.080;
Field[1].YMax =  0.080;
Field[1].ZMin = sheet_z_min - 0.002;
Field[1].ZMax = sheet_height / 2 + 0.002;

Field[2] = Box;
Field[2].VIn = gap_h;
Field[2].VOut = far_h;
Field[2].XMin = -0.008;
Field[2].XMax =  0.008;
Field[2].YMin = -0.080;
Field[2].YMax =  0.080;
Field[2].ZMin = sheet_z_min - 0.002;
Field[2].ZMax = sheet_height / 2 + 0.002;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;

Mesh 3;
