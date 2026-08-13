SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 4.1;
Mesh.ElementOrder = 1;
Mesh.MeshSizeMin = 0.12;
Mesh.MeshSizeMax = 0.12;

Box(1) = {-0.2, -0.2, -0.2, 0.7, 0.7, 0.7};
Box(2) = {0.0, 0.0, 0.0, 0.294, 0.294, 0.019};
Box(3) = {0.018, 0.018, -0.001, 0.108, 0.108, 0.021};
BooleanDifference{ Volume{2}; Delete; }{ Volume{3}; Delete; }

Box(10) = {0.094, 0.000, 0.049, 0.050, 0.200, 0.100};
Box(11) = {0.244, 0.000, 0.049, 0.050, 0.200, 0.100};
Box(12) = {0.144, 0.000, 0.049, 0.100, 0.050, 0.100};
Box(13) = {0.144, 0.150, 0.049, 0.100, 0.050, 0.100};

BooleanFragments{ Volume{1}; Delete; }{ Volume{2,10,11,12,13}; Delete; }
