// Conforming linear tetrahedra: a permeable sphere and surrounding air box.
// gmsh -3 geometries/permeable_sphere.geo -setnumber MeshScale 1 -format msh41 -o sphere.msh
SetFactory("OpenCASCADE");
If (!Exists(Radius))
  Radius = 1;
EndIf
If (!Exists(OuterRadius))
  OuterRadius = 3 * Radius;
EndIf
If (!Exists(MeshScale))
  MeshScale = 1;
EndIf
If (!Exists(SensorRefinement))
  SensorRefinement = 1;
EndIf
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 4.1;
Mesh.SaveAll = 1;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.Optimize = 1;
Box(1) = {-OuterRadius, -OuterRadius, -OuterRadius,
          2*OuterRadius, 2*OuterRadius, 2*OuterRadius};
Sphere(2) = {0, 0, 0, Radius};
domains[] = BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; };
eps = Radius * 1e-6;
iron[] = Volume In BoundingBox{-Radius-eps,-Radius-eps,-Radius-eps,
                                Radius+eps, Radius+eps, Radius+eps};
air[] = domains[];
air[] -= iron[];
interface[] = Boundary{ Volume{iron[]}; };
outer[] = CombinedBoundary{ Volume{domains[]}; };
Physical Volume("sphere", 1) = {iron[]};
Physical Volume("air", 2) = {air[]};
Physical Surface("interface", 3) = {interface[]};
Physical Surface("outer", 4) = {outer[]};
Field[1] = Distance;
Field[1].SurfacesList = {interface[]};
Field[1].Sampling = 150;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = Radius * MeshScale / 8;
Field[2].SizeMax = Radius * MeshScale;
Field[2].DistMin = 0;
Field[2].DistMax = Radius;
Field[3] = Box;
Field[3].VIn = Radius * MeshScale / 3;
Field[3].VOut = Radius * MeshScale;
Field[3].XMin = -Radius;
Field[3].XMax = Radius;
Field[3].YMin = -Radius;
Field[3].YMax = Radius;
Field[3].ZMin = -Radius;
Field[3].ZMax = Radius;
Field[3].Thickness = Radius / 2;
Field[4] = Min;
Field[4].FieldsList = {2,3};
// Pointwise observations also need local resolution in air. These fixed
// positions match default_sensors(); they are independent of truth and noise.
If (SensorRefinement)
  sx[] = {1.42,-1.53,0.24,-0.37,0.29,-0.21};
  sy[] = {0.27,0.31,1.47,-1.61,-0.23,0.38};
  sz[] = {0.36,-0.22,0.41,0.19,1.54,-1.43};
  For k In {0:5}
    Field[10+k] = Ball;
    Field[10+k].XCenter = Radius*sx[k];
    Field[10+k].YCenter = Radius*sy[k];
    Field[10+k].ZCenter = Radius*sz[k];
    Field[10+k].Radius = Radius*0.22;
    Field[10+k].Thickness = Radius*0.35;
    Field[10+k].VIn = Radius*MeshScale/16;
    Field[10+k].VOut = Radius*MeshScale;
    // Seed the localized sizing region, without putting an observation on a
    // mesh vertex. Otherwise a coarse initial tetrahedralization can miss a
    // small interior Ball field entirely.
    px[] = {0.14,-0.07,-0.05,-0.01};
    py[] = {0.01,0.13,-0.09,-0.03};
    pz[] = {0.0,0.02,0.11,-0.15};
    For j In {0:3}
      seed = newp;
      Point(seed) = {Radius*(sx[k]+MeshScale*px[j]/2),
                     Radius*(sy[k]+MeshScale*py[j]/2),
                     Radius*(sz[k]+MeshScale*pz[j]/2),Radius*MeshScale/16};
      Point{seed} In Volume{air[0]};
    EndFor
  EndFor
  Field[4].FieldsList = {2,3,10,11,12,13,14,15};
EndIf
Background Field = 4;
