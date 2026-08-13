//! Implementations of the stable CLI commands.

mod describe;
mod list;
mod run;
mod verify;

use crate::args::{Arguments, Command, HELP};

pub fn execute(arguments: Arguments) -> Result<(), String> {
    match arguments.command {
        Command::List => list::execute(),
        Command::Describe { study_id } => describe::execute(&study_id),
        Command::Run {
            study_id,
            configuration,
            output,
        } => run::execute(&study_id, &configuration, &output),
        Command::Verify {
            run_directory,
            against,
        } => verify::execute(&run_directory, &against),
        Command::Help => {
            print!("{HELP}");
            Ok(())
        }
    }
}
