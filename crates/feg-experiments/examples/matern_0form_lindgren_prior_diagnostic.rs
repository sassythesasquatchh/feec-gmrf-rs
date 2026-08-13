use feg_case_studies::matern_scalar::{
    run_scalar_matern_validation, write_scalar_matern_validation_outputs,
    ScalarMaternValidationConfig,
};
use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output_dir = PathBuf::from("out/matern_0form_lindgren_prior_diagnostic");
    let mut dimension = 2;
    let mut cases = vec![(10, 256), (100, 512)];
    let mut cases_overridden = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--out-dir" => {
                output_dir = args
                    .next()
                    .ok_or("--out-dir requires a value")?
                    .into();
            }
            "--dimension" => {
                dimension = args
                    .next()
                    .ok_or("--dimension requires a value")?
                    .parse()?;
            }
            "--cases" => {
                cases_overridden = true;
                cases = parse_cases(&args.next().ok_or("--cases requires a value")?)?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: matern_0form_lindgren_prior_diagnostic [--dimension 2|3] [--out-dir PATH] [--cases RANGE:LEVEL,...]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }
    if dimension == 3 && !cases_overridden {
        cases = vec![(10, 64)];
    }

    for (range_cells, level) in cases {
        let config = ScalarMaternValidationConfig {
            dimension,
            range_cells,
            level,
        };
        let report = run_scalar_matern_validation(config)?;
        let case_dir = output_dir.join(format!(
            "dim{dimension}_range{range_cells}_level{level}"
        ));
        write_scalar_matern_validation_outputs(&report, case_dir)?;
        println!(
            "dimension={} range={} level={} ndofs={} correlation_rmse={:.6e} variance_error={:.3}% total={:.2}s",
            dimension,
            range_cells,
            level,
            report.ndofs,
            report.correlation_rmse,
            100.0 * report.variance_relative_error,
            report.total_seconds,
        );
    }
    Ok(())
}

fn parse_cases(value: &str) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| {
            let (range, level) = part
                .split_once(':')
                .ok_or("cases must be RANGE:LEVEL pairs")?;
            Ok((range.parse()?, level.parse()?))
        })
        .collect()
}
