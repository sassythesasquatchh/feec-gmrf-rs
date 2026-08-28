# Scientific input data

Checked-in meshes are versioned numerical inputs. Keeping them in the
repository fixes simplex ordering, orientation, physical tags, and input hashes
independently of the installed Gmsh version.

| Geometry source | Mesh use |
|---|---|
| `geometries/derham_ladder_handlebody.geo` | `meshes/derham_ladder_handlebody.msh` |
| `geometries/genus2_torus_touching.geo` | `meshes/genus2_torus_touching.msh` |
| `geometries/torus_shell.geo` | `meshes/torus_shell_coarse.msh` and resolution 0–3 fixtures |
| `geometries/team7.geo` | `meshes/team7.msh` and TEAM 7 regeneration studies |
| `geometries/team13_linear.geo` | Runtime TEAM 13 benchmark meshes |
| `geometries/team13_linear_measurement_planes.geo` | TEAM 13 measurement-plane variants |
| `feec/geometries/toroidal_inductor.geo` | `meshes/toroidal_inductor.msh` |

Study descriptors identify the files they consume, and `feg-study` records
their hashes in the run manifest. Runtime meshes, solver scratch files, and
generated visualization data belong under ignored output or temporary
directories.
