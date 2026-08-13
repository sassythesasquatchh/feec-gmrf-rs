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

measurement_h = 0.00040 * MeshScale;

// TEAM 13 steel measurement patches 1-7: Bz on z-normal mid-sheet surfaces.
Point(100010) = {0.0000, -0.025, 0.000, measurement_h};
Point(100011) = {0.0016, -0.025, 0.000, measurement_h};
Point(100012) = {0.0016,  0.025, 0.000, measurement_h};
Point(100013) = {0.0000,  0.025, 0.000, measurement_h};
Line(110010) = {100010, 100011}; Line(110011) = {100011, 100012}; Line(110012) = {100012, 100013}; Line(110013) = {100013, 100010};
Curve Loop(115001) = {110010, 110011, 110012, 110013}; Plane Surface(120001) = {115001};

Point(100020) = {0.0000, -0.025, 0.010, measurement_h};
Point(100021) = {0.0016, -0.025, 0.010, measurement_h};
Point(100022) = {0.0016,  0.025, 0.010, measurement_h};
Point(100023) = {0.0000,  0.025, 0.010, measurement_h};
Line(110020) = {100020, 100021}; Line(110021) = {100021, 100022}; Line(110022) = {100022, 100023}; Line(110023) = {100023, 100020};
Curve Loop(115002) = {110020, 110021, 110022, 110023}; Plane Surface(120002) = {115002};

Point(100030) = {0.0000, -0.025, 0.020, measurement_h};
Point(100031) = {0.0016, -0.025, 0.020, measurement_h};
Point(100032) = {0.0016,  0.025, 0.020, measurement_h};
Point(100033) = {0.0000,  0.025, 0.020, measurement_h};
Line(110030) = {100030, 100031}; Line(110031) = {100031, 100032}; Line(110032) = {100032, 100033}; Line(110033) = {100033, 100030};
Curve Loop(115003) = {110030, 110031, 110032, 110033}; Plane Surface(120003) = {115003};

Point(100040) = {0.0000, -0.025, 0.030, measurement_h};
Point(100041) = {0.0016, -0.025, 0.030, measurement_h};
Point(100042) = {0.0016,  0.025, 0.030, measurement_h};
Point(100043) = {0.0000,  0.025, 0.030, measurement_h};
Line(110040) = {100040, 100041}; Line(110041) = {100041, 100042}; Line(110042) = {100042, 100043}; Line(110043) = {100043, 100040};
Curve Loop(115004) = {110040, 110041, 110042, 110043}; Plane Surface(120004) = {115004};

Point(100050) = {0.0000, -0.025, 0.040, measurement_h};
Point(100051) = {0.0016, -0.025, 0.040, measurement_h};
Point(100052) = {0.0016,  0.025, 0.040, measurement_h};
Point(100053) = {0.0000,  0.025, 0.040, measurement_h};
Line(110050) = {100050, 100051}; Line(110051) = {100051, 100052}; Line(110052) = {100052, 100053}; Line(110053) = {100053, 100050};
Curve Loop(115005) = {110050, 110051, 110052, 110053}; Plane Surface(120005) = {115005};

Point(100060) = {0.0000, -0.025, 0.050, measurement_h};
Point(100061) = {0.0016, -0.025, 0.050, measurement_h};
Point(100062) = {0.0016,  0.025, 0.050, measurement_h};
Point(100063) = {0.0000,  0.025, 0.050, measurement_h};
Line(110060) = {100060, 100061}; Line(110061) = {100061, 100062}; Line(110062) = {100062, 100063}; Line(110063) = {100063, 100060};
Curve Loop(115006) = {110060, 110061, 110062, 110063}; Plane Surface(120006) = {115006};

Point(100070) = {0.0000, -0.025, 0.060, measurement_h};
Point(100071) = {0.0016, -0.025, 0.060, measurement_h};
Point(100072) = {0.0016,  0.025, 0.060, measurement_h};
Point(100073) = {0.0000,  0.025, 0.060, measurement_h};
Line(110070) = {100070, 100071}; Line(110071) = {100071, 100072}; Line(110072) = {100072, 100073}; Line(110073) = {100073, 100070};
Curve Loop(115007) = {110070, 110071, 110072, 110073}; Plane Surface(120007) = {115007};

// TEAM 13 steel measurement patches 8-18: Bx on x-normal top-channel surfaces.
Point(100080) = {0.0021, 0.015, 0.0600, measurement_h};
Point(100081) = {0.0021, 0.065, 0.0600, measurement_h};
Point(100082) = {0.0021, 0.065, 0.0632, measurement_h};
Point(100083) = {0.0021, 0.015, 0.0632, measurement_h};
Line(110080) = {100080, 100081}; Line(110081) = {100081, 100082}; Line(110082) = {100082, 100083}; Line(110083) = {100083, 100080};
Curve Loop(115008) = {110080, 110081, 110082, 110083}; Plane Surface(120008) = {115008};

Point(100090) = {0.0100, 0.015, 0.0600, measurement_h};
Point(100091) = {0.0100, 0.065, 0.0600, measurement_h};
Point(100092) = {0.0100, 0.065, 0.0632, measurement_h};
Point(100093) = {0.0100, 0.015, 0.0632, measurement_h};
Line(110090) = {100090, 100091}; Line(110091) = {100091, 100092}; Line(110092) = {100092, 100093}; Line(110093) = {100093, 100090};
Curve Loop(115009) = {110090, 110091, 110092, 110093}; Plane Surface(120009) = {115009};

Point(100100) = {0.0200, 0.015, 0.0600, measurement_h};
Point(100101) = {0.0200, 0.065, 0.0600, measurement_h};
Point(100102) = {0.0200, 0.065, 0.0632, measurement_h};
Point(100103) = {0.0200, 0.015, 0.0632, measurement_h};
Line(110100) = {100100, 100101}; Line(110101) = {100101, 100102}; Line(110102) = {100102, 100103}; Line(110103) = {100103, 100100};
Curve Loop(115010) = {110100, 110101, 110102, 110103}; Plane Surface(120010) = {115010};

Point(100110) = {0.0300, 0.015, 0.0600, measurement_h};
Point(100111) = {0.0300, 0.065, 0.0600, measurement_h};
Point(100112) = {0.0300, 0.065, 0.0632, measurement_h};
Point(100113) = {0.0300, 0.015, 0.0632, measurement_h};
Line(110110) = {100110, 100111}; Line(110111) = {100111, 100112}; Line(110112) = {100112, 100113}; Line(110113) = {100113, 100110};
Curve Loop(115011) = {110110, 110111, 110112, 110113}; Plane Surface(120011) = {115011};

Point(100120) = {0.0400, 0.015, 0.0600, measurement_h};
Point(100121) = {0.0400, 0.065, 0.0600, measurement_h};
Point(100122) = {0.0400, 0.065, 0.0632, measurement_h};
Point(100123) = {0.0400, 0.015, 0.0632, measurement_h};
Line(110120) = {100120, 100121}; Line(110121) = {100121, 100122}; Line(110122) = {100122, 100123}; Line(110123) = {100123, 100120};
Curve Loop(115012) = {110120, 110121, 110122, 110123}; Plane Surface(120012) = {115012};

Point(100130) = {0.0500, 0.015, 0.0600, measurement_h};
Point(100131) = {0.0500, 0.065, 0.0600, measurement_h};
Point(100132) = {0.0500, 0.065, 0.0632, measurement_h};
Point(100133) = {0.0500, 0.015, 0.0632, measurement_h};
Line(110130) = {100130, 100131}; Line(110131) = {100131, 100132}; Line(110132) = {100132, 100133}; Line(110133) = {100133, 100130};
Curve Loop(115013) = {110130, 110131, 110132, 110133}; Plane Surface(120013) = {115013};

Point(100140) = {0.0600, 0.015, 0.0600, measurement_h};
Point(100141) = {0.0600, 0.065, 0.0600, measurement_h};
Point(100142) = {0.0600, 0.065, 0.0632, measurement_h};
Point(100143) = {0.0600, 0.015, 0.0632, measurement_h};
Line(110140) = {100140, 100141}; Line(110141) = {100141, 100142}; Line(110142) = {100142, 100143}; Line(110143) = {100143, 100140};
Curve Loop(115014) = {110140, 110141, 110142, 110143}; Plane Surface(120014) = {115014};

Point(100150) = {0.0800, 0.015, 0.0600, measurement_h};
Point(100151) = {0.0800, 0.065, 0.0600, measurement_h};
Point(100152) = {0.0800, 0.065, 0.0632, measurement_h};
Point(100153) = {0.0800, 0.015, 0.0632, measurement_h};
Line(110150) = {100150, 100151}; Line(110151) = {100151, 100152}; Line(110152) = {100152, 100153}; Line(110153) = {100153, 100150};
Curve Loop(115015) = {110150, 110151, 110152, 110153}; Plane Surface(120015) = {115015};

Point(100160) = {0.1000, 0.015, 0.0600, measurement_h};
Point(100161) = {0.1000, 0.065, 0.0600, measurement_h};
Point(100162) = {0.1000, 0.065, 0.0632, measurement_h};
Point(100163) = {0.1000, 0.015, 0.0632, measurement_h};
Line(110160) = {100160, 100161}; Line(110161) = {100161, 100162}; Line(110162) = {100162, 100163}; Line(110163) = {100163, 100160};
Curve Loop(115016) = {110160, 110161, 110162, 110163}; Plane Surface(120016) = {115016};

Point(100170) = {0.1100, 0.015, 0.0600, measurement_h};
Point(100171) = {0.1100, 0.065, 0.0600, measurement_h};
Point(100172) = {0.1100, 0.065, 0.0632, measurement_h};
Point(100173) = {0.1100, 0.015, 0.0632, measurement_h};
Line(110170) = {100170, 100171}; Line(110171) = {100171, 100172}; Line(110172) = {100172, 100173}; Line(110173) = {100173, 100170};
Curve Loop(115017) = {110170, 110171, 110172, 110173}; Plane Surface(120017) = {115017};

Point(100180) = {0.1221, 0.015, 0.0600, measurement_h};
Point(100181) = {0.1221, 0.065, 0.0600, measurement_h};
Point(100182) = {0.1221, 0.065, 0.0632, measurement_h};
Point(100183) = {0.1221, 0.015, 0.0632, measurement_h};
Line(110180) = {100180, 100181}; Line(110181) = {100181, 100182}; Line(110182) = {100182, 100183}; Line(110183) = {100183, 100180};
Curve Loop(115018) = {110180, 110181, 110182, 110183}; Plane Surface(120018) = {115018};

// TEAM 13 steel measurement patches 19-25: Bz on z-normal right-edge surfaces.
Point(100190) = {0.1221, 0.015, 0.060, measurement_h};
Point(100191) = {0.1253, 0.015, 0.060, measurement_h};
Point(100192) = {0.1253, 0.065, 0.060, measurement_h};
Point(100193) = {0.1221, 0.065, 0.060, measurement_h};
Line(110190) = {100190, 100191}; Line(110191) = {100191, 100192}; Line(110192) = {100192, 100193}; Line(110193) = {100193, 100190};
Curve Loop(115019) = {110190, 110191, 110192, 110193}; Plane Surface(120019) = {115019};

Point(100200) = {0.1221, 0.015, 0.050, measurement_h};
Point(100201) = {0.1253, 0.015, 0.050, measurement_h};
Point(100202) = {0.1253, 0.065, 0.050, measurement_h};
Point(100203) = {0.1221, 0.065, 0.050, measurement_h};
Line(110200) = {100200, 100201}; Line(110201) = {100201, 100202}; Line(110202) = {100202, 100203}; Line(110203) = {100203, 100200};
Curve Loop(115020) = {110200, 110201, 110202, 110203}; Plane Surface(120020) = {115020};

Point(100210) = {0.1221, 0.015, 0.040, measurement_h};
Point(100211) = {0.1253, 0.015, 0.040, measurement_h};
Point(100212) = {0.1253, 0.065, 0.040, measurement_h};
Point(100213) = {0.1221, 0.065, 0.040, measurement_h};
Line(110210) = {100210, 100211}; Line(110211) = {100211, 100212}; Line(110212) = {100212, 100213}; Line(110213) = {100213, 100210};
Curve Loop(115021) = {110210, 110211, 110212, 110213}; Plane Surface(120021) = {115021};

Point(100220) = {0.1221, 0.015, 0.030, measurement_h};
Point(100221) = {0.1253, 0.015, 0.030, measurement_h};
Point(100222) = {0.1253, 0.065, 0.030, measurement_h};
Point(100223) = {0.1221, 0.065, 0.030, measurement_h};
Line(110220) = {100220, 100221}; Line(110221) = {100221, 100222}; Line(110222) = {100222, 100223}; Line(110223) = {100223, 100220};
Curve Loop(115022) = {110220, 110221, 110222, 110223}; Plane Surface(120022) = {115022};

Point(100230) = {0.1221, 0.015, 0.020, measurement_h};
Point(100231) = {0.1253, 0.015, 0.020, measurement_h};
Point(100232) = {0.1253, 0.065, 0.020, measurement_h};
Point(100233) = {0.1221, 0.065, 0.020, measurement_h};
Line(110230) = {100230, 100231}; Line(110231) = {100231, 100232}; Line(110232) = {100232, 100233}; Line(110233) = {100233, 100230};
Curve Loop(115023) = {110230, 110231, 110232, 110233}; Plane Surface(120023) = {115023};

Point(100240) = {0.1221, 0.015, 0.010, measurement_h};
Point(100241) = {0.1253, 0.015, 0.010, measurement_h};
Point(100242) = {0.1253, 0.065, 0.010, measurement_h};
Point(100243) = {0.1221, 0.065, 0.010, measurement_h};
Line(110240) = {100240, 100241}; Line(110241) = {100241, 100242}; Line(110242) = {100242, 100243}; Line(110243) = {100243, 100240};
Curve Loop(115024) = {110240, 110241, 110242, 110243}; Plane Surface(120024) = {115024};

Point(100250) = {0.1221, 0.015, 0.000, measurement_h};
Point(100251) = {0.1253, 0.015, 0.000, measurement_h};
Point(100252) = {0.1253, 0.065, 0.000, measurement_h};
Point(100253) = {0.1221, 0.065, 0.000, measurement_h};
Line(110250) = {100250, 100251}; Line(110251) = {100251, 100252}; Line(110252) = {100252, 100253}; Line(110253) = {100253, 100250};
Curve Loop(115025) = {110250, 110251, 110252, 110253}; Plane Surface(120025) = {115025};

measurementSurfaces[] = {
  120001, 120002, 120003, 120004, 120005,
  120006, 120007, 120008, 120009, 120010,
  120011, 120012, 120013, 120014, 120015,
  120016, 120017, 120018, 120019, 120020,
  120021, 120022, 120023, 120024, 120025
};

frags[] = BooleanFragments{ Volume{1, coilVols[], ironVols[]}; Delete; }{
  Surface{measurementSurfaces[]}; Delete;
};

domain_eps = 1e-9;
domainVols[] = Volume In BoundingBox{-air_x / 2 - domain_eps, -air_y / 2 - domain_eps, z_min - domain_eps,
                                      air_x / 2 + domain_eps,  air_y / 2 + domain_eps, air_z / 2 + domain_eps};
Physical Volume("team13_domain") = {domainVols[]};

Physical Surface("team13_measurement_planes") = {
  120001, 120002, 120003, 120004, 120005,
  120006, 120007, 120008, 120009, 120010,
  120011, 120012, 120013, 120014, 120015,
  120016, 120017, 120018, 120019, 120020,
  120021, 120022, 120023, 120024, 120025
};

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
