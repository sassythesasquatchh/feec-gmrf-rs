# Scientific input inventory

Checked-in meshes are intentional, versioned inputs for tests and maintained
study profiles. They are not generated build output. Keeping them fixes entity
ordering, physical tags, and input hashes independently of the installed Gmsh
version.

| Geometry source | Checked-in or generated mesh use |
|---|---|
| `geometries/derham_ladder_handlebody.geo` | `meshes/derham_ladder_handlebody.msh` |
| `geometries/genus2_torus_touching.geo` | `meshes/genus2_torus_touching.msh` |
| `geometries/torus_shell.geo` | `meshes/torus_shell_coarse.msh` and resolution 0--3 fixtures |
| `geometries/team7.geo` | `meshes/team7.msh` and TEAM 7 regeneration workflows |
| `geometries/team13_linear.geo` | Runtime TEAM 13 benchmark meshes |
| `geometries/team13_linear_measurement_planes.geo` | Runtime measurement-plane variants |
| `feec/geometries/toroidal_inductor.geo` | `meshes/toroidal_inductor.msh` |

Maintained study descriptors record the exact files they consume and include
their hashes in run provenance. Generated meshes and solver scratch files must
remain under ignored output or temporary directories.
