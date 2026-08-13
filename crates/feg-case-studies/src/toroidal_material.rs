//! Toroidal-inductor constitutive policy.
//!
//! The torus geometry and its material partition are benchmark choices, not
//! FEEC assembly concepts, so they live with the case study.

use formoniq::problems::nonlinear_magnetostatic::SpatialReluctivity;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToroidalReluctivityLaw {
    pub major_radius: f64,
    pub core_minor_radius: f64,
    pub nu_air: f64,
    pub nu_core0: f64,
    pub beta_core: f64,
}

impl ToroidalReluctivityLaw {
    pub fn new(
        major_radius: f64,
        core_minor_radius: f64,
        nu_air: f64,
        nu_core0: f64,
        beta_core: f64,
    ) -> Result<Self, String> {
        let law = Self {
            major_radius,
            core_minor_radius,
            nu_air,
            nu_core0,
            beta_core,
        };
        law.validate()?;
        Ok(law)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.major_radius.is_finite() || self.major_radius <= 0.0 {
            return Err("toroidal reluctivity major_radius must be finite and positive".into());
        }
        if !self.core_minor_radius.is_finite() || self.core_minor_radius <= 0.0 {
            return Err(
                "toroidal reluctivity core_minor_radius must be finite and positive".into(),
            );
        }
        if !self.nu_air.is_finite() || self.nu_air <= 0.0 {
            return Err("toroidal reluctivity nu_air must be finite and positive".into());
        }
        if !self.nu_core0.is_finite() || self.nu_core0 <= 0.0 {
            return Err("toroidal reluctivity nu_core0 must be finite and positive".into());
        }
        if !self.beta_core.is_finite() || self.beta_core < 0.0 {
            return Err("toroidal reluctivity beta_core must be finite and nonnegative".into());
        }
        Ok(())
    }

    pub fn is_core_point(&self, point: [f64; 3]) -> bool {
        let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
        ((rho - self.major_radius).powi(2) + point[2] * point[2]).sqrt() <= self.core_minor_radius
    }

    pub fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        if self.is_core_point(point) {
            self.nu_core0 * (1.0 + self.beta_core * magnetic_flux_squared)
        } else {
            self.nu_air
        }
    }

    pub fn d_nu_ds(&self, point: [f64; 3], _magnetic_flux_squared: f64) -> f64 {
        if self.is_core_point(point) {
            self.nu_core0 * self.beta_core
        } else {
            0.0
        }
    }
}

impl SpatialReluctivity for ToroidalReluctivityLaw {
    fn validate(&self) -> Result<(), String> {
        ToroidalReluctivityLaw::validate(self)
    }

    fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        ToroidalReluctivityLaw::nu(self, point, magnetic_flux_squared)
    }

    fn d_nu_ds(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        ToroidalReluctivityLaw::d_nu_ds(self, point, magnetic_flux_squared)
    }

    fn linear_reference_reluctivity(&self, point: [f64; 3]) -> f64 {
        self.nu(point, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_core_and_air_and_preserves_linear_reference() {
        let law = ToroidalReluctivityLaw::new(1.0, 0.25, 10.0, 2.0, 3.0).unwrap();
        let core = [1.0, 0.0, 0.0];
        let air = [0.0, 0.0, 0.0];
        assert_eq!(law.nu(core, 4.0), 26.0);
        assert_eq!(law.d_nu_ds(core, 4.0), 6.0);
        assert_eq!(law.nu(air, 4.0), 10.0);
        assert_eq!(law.d_nu_ds(air, 4.0), 0.0);
        assert_eq!(law.linear_reference_reluctivity(core), 2.0);
    }
}
