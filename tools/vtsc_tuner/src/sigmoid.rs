//! VTSC sigmoid math.
//!
//! Mirrors `_physics_based_lateral_acceleration` in
//! `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`:
//!
//!     a_lat(κ) = A / (1 + exp(B · (κ − C))) + D      clamped to [MIN, MAX]
//!
//! with the sign conventions (A < 0, B < 0) enforced in `vision_turn_params.py`.
//! Downstream the speed target is `v = sqrt(a_lat / κ)`.

use serde::{Deserialize, Serialize};

pub const MS_TO_MPH: f64 = 2.236_936_292_054_4;
pub const MPH_TO_MS: f64 = 0.447_04;

/// Six raw sigmoid parameters. A and B are stored with their natural (negative)
/// sign; all clamping happens at the UI layer so the math here stays literal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SigmoidParams {
  pub a: f64,
  pub b: f64,
  pub c: f64,
  pub d: f64,
  pub min_lat: f64,
  pub max_lat: f64,
}

impl Default for SigmoidParams {
  /// Repo defaults from `vision_turn_controller.py` lines 818-823.
  fn default() -> Self {
    Self {
      a: -3.260,
      b: -6270.0,
      c: 0.005_010,
      d: 5.607,
      min_lat: 1.8,
      max_lat: 4.478,
    }
  }
}

impl SigmoidParams {
  /// Evaluate the lateral-acceleration ceiling for a given curvature (1/m).
  /// Returns m/s² clamped to [`min_lat`, `max_lat`]. Matches the Python
  /// implementation exactly (no `sigmoid_scale` here; that's a runtime knob).
  pub fn eval(&self, curvature: f64) -> f64 {
    let kappa = curvature.clamp(1e-8, 1.0);
    let exponent = self.b * (kappa - self.c);
    // Guard against exp overflow at extreme (B·Δκ) — saturates to the asymptote.
    let raw = if exponent > 700.0 {
      self.d
    } else if exponent < -700.0 {
      self.a + self.d
    } else {
      self.a / (1.0 + exponent.exp()) + self.d
    };
    raw.clamp(self.min_lat, self.max_lat)
  }

  /// Given a desired lateral-acceleration ceiling at a given *speed* (mph),
  /// invert `v = sqrt(a/κ)` to recover the implied curvature.
  pub fn kappa_at(speed_mph: f64, a_lat: f64) -> f64 {
    let v_mps = speed_mph * MPH_TO_MS;
    if v_mps < 1e-3 {
      1.0
    } else {
      (a_lat / (v_mps * v_mps)).clamp(1e-8, 1.0)
    }
  }
}

/// One point on the rendered (speed, a_lat) curve.
#[derive(Debug, Clone, Copy)]
pub struct CurveSample {
  #[allow(dead_code)] // useful for future exports / debug
  pub kappa: f64,
  pub speed_mph: f64,
  pub a_lat: f64,
}

/// Parametric (speed_mph, a_lat) samples along the sigmoid, sweeping curvature
/// on a log scale. Bands compose on top multiplicatively.
pub fn sample_curve(
  params: &SigmoidParams,
  bands: &[Band],
  n: usize,
  kappa_min: f64,
  kappa_max: f64,
) -> Vec<CurveSample> {
  let n = n.max(2);
  let log_min = kappa_min.max(1e-12).log10();
  let log_max = kappa_max.max(kappa_min * 10.0).log10();
  let prepped = prepare_bands(params, bands);
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    let t = i as f64 / (n - 1) as f64;
    let log_k = log_max + (log_min - log_max) * t; // high κ → low κ → high speed
    let kappa = 10f64.powf(log_k);
    let base = params.eval(kappa);
    let composed = apply_prepared_bands(kappa, base, &prepped).clamp(params.min_lat, params.max_lat);
    let v_mps = (composed / kappa).sqrt();
    out.push(CurveSample {
      kappa,
      speed_mph: v_mps * MS_TO_MPH,
      a_lat: composed,
    });
  }
  out
}

// ---------------------------------------------------------------------------
// EQ-style local shaping bands — stacked multiplicatively on top of the sigmoid.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Band {
  /// Anchor speed in mph (what the user sees on the X axis).
  pub center_speed_mph: f64,
  /// Gain in decibels applied to a_lat at the band centre.
  /// +3 dB ≈ +41%, −3 dB ≈ −29%.
  pub gain_db: f64,
  /// Bandwidth control. Q = 0.3 wide, Q = 4 narrow. Maps to σ_logκ = 0.5/Q.
  pub q: f64,
  pub enabled: bool,
}

impl Default for Band {
  fn default() -> Self {
    Self {
      center_speed_mph: 45.0,
      gain_db: 0.0,
      q: 1.0,
      enabled: true,
    }
  }
}

impl Band {
  /// Return the band's centre curvature given the current sigmoid (so the band
  /// tracks the curve when macro params move).
  pub fn kappa_center(&self, params: &SigmoidParams) -> f64 {
    // Iterate the fixed point κ ↔ a(κ) a few times since a is cheap.
    let mut a = 0.5 * (params.min_lat + params.max_lat);
    for _ in 0..8 {
      let kappa = SigmoidParams::kappa_at(self.center_speed_mph, a);
      a = params.eval(kappa);
    }
    SigmoidParams::kappa_at(self.center_speed_mph, a)
  }
}

/// A band pre-resolved to a fixed log₁₀(κ_center) against the current sigmoid.
/// Call `prepare_bands` once per curve redraw, then reuse across sample points.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBand {
  pub log_kc: f64,
  pub gain_db: f64,
  pub sigma: f64,
}

pub fn prepare_bands(params: &SigmoidParams, bands: &[Band]) -> Vec<PreparedBand> {
  bands
    .iter()
    .filter(|b| b.enabled)
    .map(|b| {
      let kc = b.kappa_center(params);
      PreparedBand {
        log_kc: kc.max(1e-12).log10(),
        gain_db: b.gain_db,
        sigma: 0.5 / b.q.max(0.1),
      }
    })
    .collect()
}

/// Apply prepared bands multiplicatively in dB space at a single curvature.
pub fn apply_prepared_bands(kappa: f64, base_a_lat: f64, prepped: &[PreparedBand]) -> f64 {
  if prepped.is_empty() {
    return base_a_lat;
  }
  let log_k = kappa.max(1e-12).log10();
  let mut log_gain_db = 0.0_f64;
  for band in prepped {
    let z = (log_k - band.log_kc) / band.sigma;
    let bell = (-0.5 * z * z).exp();
    log_gain_db += band.gain_db * bell;
  }
  base_a_lat * 10f64.powf(log_gain_db / 20.0)
}

// ---------------------------------------------------------------------------
// Export helpers — build a `Q_CURVE_POINTS` list equivalent to the composed
// (sigmoid × bands) output, so the existing runtime path at
// `vtsc_curve_tuning.py` can consume a tune produced here.
// ---------------------------------------------------------------------------

/// Dump `(κ, q_speed)` control points sampled at log-uniform curvatures.
/// `q_speed = sqrt((sigmoid × bands) / sigmoid)` — since speed goes as
/// `sqrt(a/κ)`, a gain in lateral accel maps to `sqrt(gain)` in speed.
#[allow(dead_code)] // reserved for Q_CURVE_POINTS export in a later version
pub fn bands_as_q_curve_points(
  params: &SigmoidParams,
  bands: &[Band],
  n: usize,
) -> Vec<(f64, f64)> {
  let n = n.max(2);
  let prepped = prepare_bands(params, bands);
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    let t = i as f64 / (n - 1) as f64;
    let log_k = -5.0 + 5.0 * t; // 1e-5 → 1e0
    let kappa = 10f64.powf(log_k);
    let base = params.eval(kappa);
    if base < 1e-6 {
      out.push((kappa, 1.0));
      continue;
    }
    let composed = apply_prepared_bands(kappa, base, &prepped);
    let q_speed = (composed / base).sqrt().clamp(0.5, 1.5);
    out.push((kappa, q_speed));
  }
  out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
  }

  #[test]
  fn matches_python_reference_at_midpoint() {
    // From vision_turn_controller.py defaults.
    let p = SigmoidParams::default();
    let a_mid = p.a / (1.0 + 0.0_f64.exp()) + p.d; // κ == C → exp(0) == 1 → A/2 + D
    assert!(approx(a_mid, -3.26 / 2.0 + 5.607, 1e-9));
    assert!(approx(p.eval(0.005_010), a_mid.clamp(1.8, 4.478), 1e-9));
  }

  #[test]
  fn clamps_at_asymptotes() {
    let p = SigmoidParams::default();
    // Straight road: κ → 0 → exponent → +∞ → sigmoid collapses to 0 → result = D, clamped to max.
    assert!(approx(p.eval(1e-8), p.max_lat, 1e-9));
    // Tight curve: κ → 1 → exponent → very negative → sigmoid → 1 → result = A + D, clamped to min if below.
    let tight = p.eval(1.0);
    assert!(tight >= p.min_lat - 1e-9);
    assert!(tight <= p.max_lat + 1e-9);
  }

  #[test]
  fn kappa_at_round_trip() {
    // At 60 mph and 3.5 m/s², κ = 3.5 / (60 * 0.44704)² ≈ 0.00487
    let k = SigmoidParams::kappa_at(60.0, 3.5);
    let v = (3.5_f64 / k).sqrt() * MS_TO_MPH;
    assert!(approx(v, 60.0, 1e-6));
  }

  #[test]
  fn band_centered_adds_expected_gain() {
    let p = SigmoidParams::default();
    let bands = [Band {
      center_speed_mph: 60.0,
      gain_db: 6.0,
      q: 2.0,
      enabled: true,
    }];
    let prepped = prepare_bands(&p, &bands);
    let kc = 10f64.powf(prepped[0].log_kc);
    let base = p.eval(kc);
    let composed = apply_prepared_bands(kc, base, &prepped);
    let gain = composed / base;
    assert!(approx(gain, 10f64.powf(6.0 / 20.0), 1e-2));
    // No-op when disabled.
    let disabled = [Band {
      enabled: false,
      ..bands[0]
    }];
    let empty = prepare_bands(&p, &disabled);
    assert!(empty.is_empty());
    assert!(approx(apply_prepared_bands(kc, base, &empty), base, 1e-9));
  }

  #[test]
  fn sample_curve_monotonic_in_speed() {
    let p = SigmoidParams::default();
    let samples = sample_curve(&p, &[], 200, 1e-5, 1.0);
    let speeds: Vec<f64> = samples.iter().map(|s| s.speed_mph).collect();
    // As κ decreases (straighter road), speed should rise monotonically.
    for w in speeds.windows(2) {
      assert!(w[1] >= w[0] - 1e-6, "speed decreased: {} → {}", w[0], w[1]);
    }
  }
}
