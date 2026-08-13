//! TEAM 13 benchmark geometry and constitutive laws.
//!
//! These laws encode the benchmark's iron geometry, B-H table, and calibration
//! coordinates. The case-study layer owns this benchmark-specific material
//! model.

use formoniq::problems::nonlinear_magnetostatic::SpatialReluctivity;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Team13SmoothIronReluctivityLaw {
    pub nu_air: f64,
    pub nu_iron0: f64,
    pub beta_iron: f64,
    pub b_scale: f64,
    pub log_iron_nu_scale: f64,
}

impl Team13SmoothIronReluctivityLaw {
    pub fn new(nu_air: f64, nu_iron0: f64, beta_iron: f64, b_scale: f64) -> Result<Self, String> {
        let law = Self {
            nu_air,
            nu_iron0,
            beta_iron,
            b_scale,
            log_iron_nu_scale: 0.0,
        };
        law.validate()?;
        Ok(law)
    }

    pub fn with_log_iron_nu_scale(mut self, value: f64) -> Result<Self, String> {
        self.log_iron_nu_scale = value;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.nu_air.is_finite() || self.nu_air <= 0.0 {
            return Err("TEAM13 air reluctivity must be finite and positive".into());
        }
        if !self.nu_iron0.is_finite() || self.nu_iron0 <= 0.0 {
            return Err("TEAM13 iron reluctivity must be finite and positive".into());
        }
        if !self.beta_iron.is_finite() || self.beta_iron < 0.0 {
            return Err("TEAM13 iron nonlinear beta must be finite and nonnegative".into());
        }
        if !self.b_scale.is_finite() || self.b_scale <= 0.0 {
            return Err("TEAM13 nonlinear B scale must be finite and positive".into());
        }
        if !self.log_iron_nu_scale.is_finite() || !self.iron_nu_scale().is_finite() {
            return Err("TEAM13 iron reluctivity log-scale must be finite".into());
        }
        Ok(())
    }

    pub fn iron_nu_scale(&self) -> f64 {
        self.log_iron_nu_scale.exp()
    }

    pub fn is_iron_point(&self, point: [f64; 3]) -> bool {
        team13_is_iron_point(point)
    }

    pub fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        if self.is_iron_point(point) {
            self.iron_nu_scale()
                * self.nu_iron0
                * (1.0 + self.beta_iron * magnetic_flux_squared / (self.b_scale * self.b_scale))
        } else {
            self.nu_air
        }
    }

    pub fn d_nu_ds(&self, point: [f64; 3], _magnetic_flux_squared: f64) -> f64 {
        if self.is_iron_point(point) {
            self.iron_nu_scale() * self.nu_iron0 * self.beta_iron / (self.b_scale * self.b_scale)
        } else {
            0.0
        }
    }

    pub fn linear_reference_law(&self) -> Self {
        Self {
            beta_iron: 0.0,
            ..*self
        }
    }
}

impl SpatialReluctivity for Team13SmoothIronReluctivityLaw {
    fn validate(&self) -> Result<(), String> {
        Team13SmoothIronReluctivityLaw::validate(self)
    }

    fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        Team13SmoothIronReluctivityLaw::nu(self, point, magnetic_flux_squared)
    }

    fn d_nu_ds(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        Team13SmoothIronReluctivityLaw::d_nu_ds(self, point, magnetic_flux_squared)
    }

    fn linear_reference_reluctivity(&self, point: [f64; 3]) -> f64 {
        self.nu(point, 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Team13BhSample {
    pub b_tesla: f64,
    pub h_ampere_per_meter: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Team13TabulatedReluctivityLaw {
    pub nu_air: f64,
    pub linear_nu_iron: f64,
    pub samples: &'static [Team13BhSample],
    pub log_iron_nu_scale: f64,
    pub log_h_shape_anchors_tesla: [f64; 3],
    pub log_h_shape_values: [f64; 3],
}

impl Team13TabulatedReluctivityLaw {
    pub fn new(
        nu_air: f64,
        linear_nu_iron: f64,
        samples: &'static [Team13BhSample],
    ) -> Result<Self, String> {
        let law = Self {
            nu_air,
            linear_nu_iron,
            samples,
            log_iron_nu_scale: 0.0,
            log_h_shape_anchors_tesla: [0.5, 1.7, 2.3],
            log_h_shape_values: [0.0; 3],
        };
        law.validate()?;
        Ok(law)
    }

    pub fn with_log_iron_nu_scale(mut self, value: f64) -> Result<Self, String> {
        self.log_iron_nu_scale = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_log_h_shape(
        mut self,
        anchors_tesla: [f64; 3],
        values: [f64; 3],
    ) -> Result<Self, String> {
        self.log_h_shape_anchors_tesla = anchors_tesla;
        self.log_h_shape_values = values;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.nu_air.is_finite() || self.nu_air <= 0.0 {
            return Err("TEAM13 tabulated air reluctivity must be finite and positive".into());
        }
        if !self.linear_nu_iron.is_finite() || self.linear_nu_iron <= 0.0 {
            return Err(
                "TEAM13 tabulated linear iron reluctivity must be finite and positive".into(),
            );
        }
        if self.samples.len() < 2 {
            return Err("TEAM13 tabulated B-H law requires at least two samples".into());
        }
        if !self.log_iron_nu_scale.is_finite() || !self.iron_nu_scale().is_finite() {
            return Err("TEAM13 tabulated iron reluctivity log-scale must be finite".into());
        }
        if !self
            .log_h_shape_anchors_tesla
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(
                "TEAM13 tabulated log-H shape anchors must be finite and nonnegative".into(),
            );
        }
        if !(self.log_h_shape_anchors_tesla[0] < self.log_h_shape_anchors_tesla[1]
            && self.log_h_shape_anchors_tesla[1] < self.log_h_shape_anchors_tesla[2])
        {
            return Err("TEAM13 tabulated log-H shape anchors must be strictly increasing".into());
        }
        if !self
            .log_h_shape_values
            .iter()
            .all(|value| value.is_finite() && (self.log_iron_nu_scale + *value).exp().is_finite())
        {
            return Err("TEAM13 tabulated log-H shape values must be finite".into());
        }
        let mut previous_b = f64::NEG_INFINITY;
        let mut previous_h = f64::NEG_INFINITY;
        for (index, sample) in self.samples.iter().enumerate() {
            if !sample.b_tesla.is_finite() || sample.b_tesla < 0.0 {
                return Err(format!(
                    "TEAM13 B-H sample {index} has invalid B value {}",
                    sample.b_tesla
                ));
            }
            if !sample.h_ampere_per_meter.is_finite() || sample.h_ampere_per_meter < 0.0 {
                return Err(format!(
                    "TEAM13 B-H sample {index} has invalid H value {}",
                    sample.h_ampere_per_meter
                ));
            }
            if sample.b_tesla < previous_b {
                return Err("TEAM13 B-H samples must have nondecreasing B values".into());
            }
            if sample.b_tesla == previous_b && sample.h_ampere_per_meter != previous_h {
                return Err("TEAM13 B-H duplicate B samples must have identical H values".into());
            }
            if sample.h_ampere_per_meter < previous_h {
                return Err("TEAM13 B-H samples must have nondecreasing H values".into());
            }
            previous_b = sample.b_tesla;
            previous_h = sample.h_ampere_per_meter;
        }
        Ok(())
    }

    pub fn is_iron_point(&self, point: [f64; 3]) -> bool {
        team13_is_iron_point(point)
    }

    pub fn iron_nu_scale(&self) -> f64 {
        self.log_iron_nu_scale.exp()
    }

    pub fn log_h_shape_basis(&self, b: f64) -> [f64; 3] {
        let [b0, b1, b2] = self.log_h_shape_anchors_tesla;
        if b <= b0 {
            [1.0, 0.0, 0.0]
        } else if b < b1 {
            let t = (b - b0) / (b1 - b0);
            [1.0 - t, t, 0.0]
        } else if b < b2 {
            let t = (b - b1) / (b2 - b1);
            [0.0, 1.0 - t, t]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    pub fn h_ampere_per_meter(&self, b: f64) -> f64 {
        let (h, _) = self.h_and_slope(b.max(0.0));
        self.iron_h_scale(b.max(0.0)) * h
    }

    pub fn d_nu_d_log_h_shape_values(
        &self,
        point: [f64; 3],
        magnetic_flux_squared: f64,
    ) -> [f64; 3] {
        if !self.is_iron_point(point) {
            return [0.0; 3];
        }
        let b = magnetic_flux_squared.max(0.0).sqrt();
        let nu = self.nu(point, magnetic_flux_squared);
        let basis = self.log_h_shape_basis(b);
        [basis[0] * nu, basis[1] * nu, basis[2] * nu]
    }

    pub fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        if !self.is_iron_point(point) {
            return self.nu_air;
        }
        let b = magnetic_flux_squared.max(0.0).sqrt();
        let (h, slope) = self.h_and_slope(b);
        self.iron_h_scale(b) * if b <= 1e-14 { slope } else { h / b }
    }

    pub fn d_nu_ds(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        if !self.is_iron_point(point) {
            return 0.0;
        }
        let b = magnetic_flux_squared.max(0.0).sqrt();
        if b <= 1e-10 {
            return 0.0;
        }
        let (h, slope) = self.h_and_slope(b);
        let (shape, shape_derivative) = self.log_h_shape_and_derivative(b);
        let scale = (self.log_iron_nu_scale + shape).exp();
        let perturbed_h_derivative = scale * (slope + h * shape_derivative);
        let perturbed_h = scale * h;
        (perturbed_h_derivative * b - perturbed_h) / (2.0 * b * b * b)
    }

    pub fn linear_reference_law(&self) -> Team13SmoothIronReluctivityLaw {
        Team13SmoothIronReluctivityLaw {
            nu_air: self.nu_air,
            nu_iron0: self.linear_nu_iron,
            beta_iron: 0.0,
            b_scale: 1.0,
            log_iron_nu_scale: self.log_iron_nu_scale + self.log_h_shape_values[0],
        }
    }

    fn iron_h_scale(&self, b: f64) -> f64 {
        let (shape, _) = self.log_h_shape_and_derivative(b);
        (self.log_iron_nu_scale + shape).exp()
    }

    fn log_h_shape_and_derivative(&self, b: f64) -> (f64, f64) {
        let [b0, b1, b2] = self.log_h_shape_anchors_tesla;
        let [v0, v1, v2] = self.log_h_shape_values;
        if b <= b0 {
            (v0, 0.0)
        } else if b < b1 {
            let inv_width = 1.0 / (b1 - b0);
            let t = (b - b0) * inv_width;
            ((1.0 - t) * v0 + t * v1, (v1 - v0) * inv_width)
        } else if b < b2 {
            let inv_width = 1.0 / (b2 - b1);
            let t = (b - b1) * inv_width;
            ((1.0 - t) * v1 + t * v2, (v2 - v1) * inv_width)
        } else {
            (v2, 0.0)
        }
    }

    fn h_and_slope(&self, b: f64) -> (f64, f64) {
        let first = self.samples[0];
        if b <= first.b_tesla {
            let slope = if first.b_tesla > 0.0 {
                first.h_ampere_per_meter / first.b_tesla
            } else {
                segment_slope(first, self.samples[1])
            };
            return (slope * b, slope);
        }
        for window in self.samples.windows(2) {
            let left = window[0];
            let right = window[1];
            if right.b_tesla <= left.b_tesla {
                continue;
            }
            if b <= right.b_tesla {
                let slope = segment_slope(left, right);
                return (left.h_ampere_per_meter + slope * (b - left.b_tesla), slope);
            }
        }
        let n = self.samples.len();
        let left = self.samples[n - 2];
        let right = self.samples[n - 1];
        let slope = segment_slope(left, right);
        (
            right.h_ampere_per_meter + slope * (b - right.b_tesla),
            slope,
        )
    }
}

impl SpatialReluctivity for Team13TabulatedReluctivityLaw {
    fn validate(&self) -> Result<(), String> {
        Team13TabulatedReluctivityLaw::validate(self)
    }

    fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        Team13TabulatedReluctivityLaw::nu(self, point, magnetic_flux_squared)
    }

    fn d_nu_ds(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
        Team13TabulatedReluctivityLaw::d_nu_ds(self, point, magnetic_flux_squared)
    }

    fn linear_reference_reluctivity(&self, point: [f64; 3]) -> f64 {
        if self.is_iron_point(point) {
            self.linear_nu_iron * (self.log_iron_nu_scale + self.log_h_shape_values[0]).exp()
        } else {
            self.nu_air
        }
    }
}

fn segment_slope(left: Team13BhSample, right: Team13BhSample) -> f64 {
    (right.h_ampere_per_meter - left.h_ampere_per_meter) / (right.b_tesla - left.b_tesla)
}

pub fn team13_is_iron_point(point: [f64; 3]) -> bool {
    team13_is_vertical_sheet(point)
        || team13_is_left_c_sheet(point)
        || team13_is_right_c_sheet(point)
}

fn team13_is_vertical_sheet(point: [f64; 3]) -> bool {
    team13_in_range(point[0], (-0.0016, 0.0016))
        && team13_in_range(point[1], (-0.025, 0.025))
        && team13_in_range(point[2], (-0.0632, 0.0632))
}

fn team13_is_left_c_sheet(point: [f64; 3]) -> bool {
    let outer = team13_in_range(point[0], (-0.1253, -0.0021))
        && team13_in_range(point[1], (-0.065, -0.015))
        && team13_in_range(point[2], (-0.0632, 0.0632));
    let inner = team13_in_range(point[0], (-0.1221, -0.0021))
        && team13_in_range(point[2], (-0.0600, 0.0600));
    outer && !inner
}

fn team13_is_right_c_sheet(point: [f64; 3]) -> bool {
    let outer = team13_in_range(point[0], (0.0021, 0.1253))
        && team13_in_range(point[1], (0.015, 0.065))
        && team13_in_range(point[2], (-0.0632, 0.0632));
    let inner =
        team13_in_range(point[0], (0.0021, 0.1221)) && team13_in_range(point[2], (-0.0600, 0.0600));
    outer && !inner
}

fn team13_in_range(value: f64, range: (f64, f64)) -> bool {
    const EPS: f64 = 1e-12;
    value >= range.0 - EPS && value <= range.1 + EPS
}

#[cfg(test)]
mod tests {
    use super::*;

    static SAMPLES: &[Team13BhSample] = &[
        Team13BhSample {
            b_tesla: 0.0,
            h_ampere_per_meter: 0.0,
        },
        Team13BhSample {
            b_tesla: 1.0,
            h_ampere_per_meter: 2.0,
        },
        Team13BhSample {
            b_tesla: 2.0,
            h_ampere_per_meter: 8.0,
        },
    ];

    #[test]
    fn tabulated_law_preserves_nu_derivative_and_linear_reference() {
        let law = Team13TabulatedReluctivityLaw::new(10.0, 3.0, SAMPLES).unwrap();
        let iron = [0.0, 0.0, 0.0];
        let air = [0.2, 0.2, 0.2];
        assert_eq!(law.nu(iron, 1.0), 2.0);
        assert_eq!(law.d_nu_ds(iron, 1.0), 0.0);
        assert_eq!(law.nu(air, 1.0), 10.0);
        assert_eq!(law.linear_reference_reluctivity(iron), 3.0);
    }

    #[test]
    fn smooth_law_sensitivity_matches_finite_difference() {
        let law = Team13SmoothIronReluctivityLaw::new(10.0, 2.0, 3.0, 2.0).unwrap();
        let point = [0.0, 0.0, 0.0];
        let s = 4.0;
        let h = 1e-6;
        let finite_difference = (law.nu(point, s + h) - law.nu(point, s - h)) / (2.0 * h);
        assert!((finite_difference - law.d_nu_ds(point, s)).abs() < 1e-8);
    }
}
