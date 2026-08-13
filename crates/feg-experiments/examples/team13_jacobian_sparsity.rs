use feg_case_studies::team13::{
    run_team13_jacobian_sparsity_audit, Team13DomainMode, Team13JacobianSparsityAuditConfig,
    Team13NonlinearMaterialKind,
};
use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let workspace = workspace_root()?;
    let mut audit = config;
    audit.mesh_path = absolutize(&workspace, &audit.mesh_path);
    let result = run_team13_jacobian_sparsity_audit(&audit)?;

    println!("metric,value");
    println!("domain,{}", result.domain_mode.as_str());
    println!("mesh,{}", result.mesh_path.display());
    println!("material_kind,{}", result.material_kind.as_str());
    println!("vertices,{}", result.vertices);
    println!("edges,{}", result.edges);
    println!("cells,{}", result.cells);
    println!("active_dofs,{}", result.active_dofs);
    println!("boundary_edge_dofs,{}", result.boundary_edge_dofs);
    println!("residual_dimension,{}", result.residual_dimension);
    println!("jacobian_rows,{}", result.jacobian_rows);
    println!("jacobian_cols,{}", result.jacobian_cols);
    println!("jacobian_nnz,{}", result.jacobian_nnz);
    println!(
        "jacobian_lower_triangle_nnz,{}",
        result.jacobian_lower_triangle_nnz
    );
    println!("jacobian_density,{:.12e}", result.jacobian_density);
    println!(
        "jacobian_lower_triangle_density,{:.12e}",
        result.jacobian_lower_triangle_density
    );
    println!("normal_rows,{}", result.normal_rows);
    println!("normal_cols,{}", result.normal_cols);
    println!("normal_nnz,{}", result.normal_nnz);
    println!(
        "normal_lower_triangle_nnz,{}",
        result.normal_lower_triangle_nnz
    );
    println!("normal_density,{:.12e}", result.normal_density);
    println!(
        "normal_lower_triangle_density,{:.12e}",
        result.normal_lower_triangle_density
    );
    println!(
        "normal_to_jacobian_nnz_ratio,{:.12e}",
        result.normal_to_jacobian_nnz_ratio
    );
    println!(
        "reduced_physical_rhs_norm,{:.12e}",
        result.reduced_physical_rhs_norm
    );
    println!(
        "linear_mean_residual_norm,{:.12e}",
        result.linear_mean_residual_norm
    );
    println!(
        "nonlinear_residual_at_linear_mean_norm,{:.12e}",
        result.nonlinear_residual_at_linear_mean_norm
    );
    println!("jacobian_seconds,{:.12e}", result.jacobian_seconds);
    println!(
        "normal_product_seconds,{:.12e}",
        result.normal_product_seconds
    );
    Ok(())
}

fn parse_args() -> Result<Team13JacobianSparsityAuditConfig, Box<dyn Error>> {
    let mut config = Team13JacobianSparsityAuditConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--mesh" => config.mesh_path = PathBuf::from(next_arg(&mut args, "--mesh")?),
            "--full-domain" => config.domain_mode = Team13DomainMode::Full,
            "--material-kind" | "--material" => {
                config.material_kind = parse_material_kind(&next_arg(&mut args, arg.as_str())?)?
            }
            "--ngsolve-tabulated-material" => {
                config.material_kind = Team13NonlinearMaterialKind::NgsolveTabulatedLinear
            }
            "--smooth-material" => {
                config.material_kind = Team13NonlinearMaterialKind::SmoothQuadratic
            }
            "--beta-iron" => config.beta_iron = next_arg(&mut args, "--beta-iron")?.parse()?,
            "--b-scale" => config.b_scale_tesla = next_arg(&mut args, "--b-scale")?.parse()?,
            "--ampere-turns" => {
                config.ampere_turns = next_arg(&mut args, "--ampere-turns")?.parse()?
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    Ok(config)
}

fn parse_material_kind(value: &str) -> Result<Team13NonlinearMaterialKind, Box<dyn Error>> {
    match value {
        "ngsolve" | "ngsolve-table" | "ngsolve-tabulated-linear" | "tabulated" => {
            Ok(Team13NonlinearMaterialKind::NgsolveTabulatedLinear)
        }
        "smooth" | "smooth-quadratic" => Ok(Team13NonlinearMaterialKind::SmoothQuadratic),
        other => Err(format!(
            "unknown material kind `{other}`; expected ngsolve-tabulated-linear or smooth-quadratic"
        )
        .into()),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let mut dir = env::current_dir()?;
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("crates/feg-case-studies").is_dir() {
            return Ok(dir);
        }
        if !dir.pop() {
            return Err("could not locate workspace root".into());
        }
    }
}

fn absolutize(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-case-studies --example team13_jacobian_sparsity -- [options]\n\
         \n\
         Options:\n\
           --mesh PATH             Use an existing TEAM13 mesh\n\
           --full-domain           Treat the mesh as the full z-symmetric domain\n\
           --material-kind VALUE   ngsolve-tabulated-linear or smooth-quadratic\n\
           --ampere-turns VALUE    Coil excitation, default 1000\n\
           --beta-iron VALUE       Smooth nonlinear iron beta, default 10\n\
           --b-scale VALUE         B scale in tesla, default 1"
    );
}
