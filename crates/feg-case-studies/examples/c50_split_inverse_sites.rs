use feg_case_studies::c50_split_inverse_sites::{
    run_c50_split_inverse_site_study, C50_SPLIT_INVERSE_SITE_REFINEMENTS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let refinements = std::env::args()
        .skip(1)
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let refinements = if refinements.is_empty() {
        C50_SPLIT_INVERSE_SITE_REFINEMENTS.to_vec()
    } else {
        refinements
    };
    for row in run_c50_split_inverse_site_study(&refinements)? {
        println!("{row:?}");
    }
    Ok(())
}
