fn main() {
    if let Err(error) = feg_cli::commands::execute(feg_cli::args::Arguments::parse_env()) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
