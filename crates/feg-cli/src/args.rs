//! Dependency-light parsing for `feg-study` arguments.

use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    List,
    Describe {
        study_id: String,
    },
    Run {
        study_id: String,
        configuration: RunConfiguration,
        output: PathBuf,
    },
    Verify {
        run_directory: PathBuf,
        against: String,
    },
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunConfiguration {
    Profile(String),
    Custom(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arguments {
    pub command: Command,
}

impl Arguments {
    pub fn parse_env() -> Self {
        Self::parse(std::env::args().skip(1)).unwrap_or_else(|message| {
            eprintln!("error: {message}\n");
            Self {
                command: Command::Help,
            }
        })
    }

    pub fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args = args.into_iter().map(Into::into);
        let Some(command) = args.next() else {
            return Ok(Self {
                command: Command::Help,
            });
        };
        let command = match command.as_str() {
            "list" => Command::List,
            "describe" => Command::Describe {
                study_id: args.next().ok_or("describe requires <study-id>")?,
            },
            "run" => {
                let study_id = args.next().ok_or("run requires <study-id>")?;
                let mut profile = None;
                let mut config = None;
                let mut output = None;
                while let Some(flag) = args.next() {
                    match flag.as_str() {
                        "--profile" => profile = args.next(),
                        "--config" => config = args.next().map(PathBuf::from),
                        "--output" => output = args.next().map(PathBuf::from),
                        unknown => return Err(format!("unknown run option `{unknown}`")),
                    }
                }
                let configuration = match (profile, config) {
                    (Some(profile), None) => RunConfiguration::Profile(profile),
                    (None, Some(path)) => RunConfiguration::Custom(path),
                    (None, None) => {
                        return Err("run requires either --profile <profile> or --config <path>"
                            .to_string())
                    }
                    (Some(_), Some(_)) => {
                        return Err("--profile and --config are mutually exclusive".to_string())
                    }
                };
                Command::Run {
                    study_id,
                    configuration,
                    output: output.ok_or("run requires --output <directory>")?,
                }
            }
            "verify" => {
                let run_directory =
                    PathBuf::from(args.next().ok_or("verify requires <run-directory>")?);
                if args.next().as_deref() != Some("--against") {
                    return Err("verify requires --against <profile>".to_string());
                }
                let against = args.next().ok_or("verify requires --against <profile>")?;
                Command::Verify {
                    run_directory,
                    against,
                }
            }
            "help" | "--help" | "-h" => Command::Help,
            unknown => return Err(format!("unknown command `{unknown}`")),
        };
        Ok(Self { command })
    }
}

pub const HELP: &str = "FEEC-GMRF reproducible study runner\n\n\
Usage:\n  feg-study list\n  feg-study describe <study-id>\n  \
feg-study run <study-id> --profile <profile> --output <directory>\n  \
feg-study run <study-id> --config <path> --output <directory>\n  \
feg-study verify <run-directory> --against <profile>\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stable_run_surface() {
        let args = Arguments::parse([
            "run",
            "matern/scalar",
            "--profile",
            "smoke",
            "--output",
            "out",
        ])
        .unwrap();
        assert!(matches!(args.command, Command::Run { .. }));
    }

    #[test]
    fn parses_custom_run_surface() {
        let args = Arguments::parse([
            "run",
            "matern/scalar",
            "--config",
            "research.toml",
            "--output",
            "out",
        ])
        .unwrap();
        assert!(matches!(
            args.command,
            Command::Run {
                configuration: RunConfiguration::Custom(_),
                ..
            }
        ));
    }

    #[test]
    fn rejects_profile_and_custom_config_together() {
        let error = Arguments::parse([
            "run",
            "matern/scalar",
            "--profile",
            "smoke",
            "--config",
            "research.toml",
            "--output",
            "out",
        ])
        .unwrap_err();
        assert!(error.contains("mutually exclusive"));
    }
}
