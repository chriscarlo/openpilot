//! Plain-English knob bindings over `SigmoidParams`.
//!
//! The raw sigmoid has six parameters (A, B, C, D, MIN, MAX) whose units mean
//! nothing to a driver. This module exposes four intuitive knobs:
//!
//!   * Tight-Curve Ceiling  (m/s²)   — lat-accel in slow, tight corners
//!   * Straight-Road Ceiling (m/s²)  — lat-accel on highway-gentle curves
//!   * Transition Speed     (mph)   — where the curve inflects
//!   * Sharpness            (0..10) — how abrupt the transition feels
//!
//! The mapping chosen makes "what you see is what you get":
//!   high-κ asymptote ≡ Tight-Curve Ceiling  ⇒  A + D = low_ceiling
//!   low-κ asymptote  ≡ Straight-Road Ceiling ⇒  D     = high_ceiling
//!   sigmoid midpoint κ is solved from Transition Speed at the mean a_lat.
//!
//! Clip ranges mirror `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`
//! so everything produced here will survive the runtime re-clamp.

use serde::{Deserialize, Serialize};

use crate::sigmoid::{MPH_TO_MS, MS_TO_MPH, SigmoidParams};

/// Clip ranges matching the runtime in `vision_turn_params.py`.
pub mod clip {
  pub const D: (f64, f64) = (2.0, 6.5);
  pub const ABS_A: (f64, f64) = (0.2, 5.0);
  #[allow(dead_code)] // surfaced in the advanced-view raw slider range
  pub const ABS_B: (f64, f64) = (100.0, 1.0e5);
  pub const C: (f64, f64) = (1.0e-5, 0.1);
  pub const MIN_LAT: (f64, f64) = (1.0, 3.0);
  pub const MAX_LAT: (f64, f64) = (2.0, 5.5);
  pub const TRANSITION_MPH: (f64, f64) = (8.0, 120.0);
  pub const SHARPNESS: (f64, f64) = (0.0, 10.0);
}

/// Maps `[0, 10]` → `|B| ∈ [100, 1e5]` log-linearly.
fn sharpness_to_abs_b(sharpness: f64) -> f64 {
  let s = sharpness.clamp(clip::SHARPNESS.0, clip::SHARPNESS.1);
  let log_b = 2.0 + (s / 10.0) * 3.0; // 10² → 10⁵
  10f64.powf(log_b)
}

fn abs_b_to_sharpness(abs_b: f64) -> f64 {
  let log_b = abs_b.max(1.0).log10();
  ((log_b - 2.0) / 3.0 * 10.0).clamp(clip::SHARPNESS.0, clip::SHARPNESS.1)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlainKnobs {
  pub tight_curve_accel: f64,    // m/s²
  pub straight_road_accel: f64,  // m/s²
  pub transition_speed_mph: f64, // mph
  pub sharpness: f64,            // 0..10
}

impl Default for PlainKnobs {
  fn default() -> Self {
    Self::from_sigmoid(&SigmoidParams::default())
  }
}

impl PlainKnobs {
  /// Project the four plain knobs down to the six raw sigmoid parameters.
  pub fn to_sigmoid(&self) -> SigmoidParams {
    // Enforce low ≤ high with a thin minimum gap so |A| stays valid.
    let low = self.tight_curve_accel.clamp(clip::MIN_LAT.0, clip::MIN_LAT.1);
    let high = self
      .straight_road_accel
      .clamp(clip::MAX_LAT.0, clip::MAX_LAT.1)
      .max(low + clip::ABS_A.0);

    // Raw sigmoid asymptotes: low-κ ≈ D, high-κ ≈ A + D.
    let d = high.clamp(clip::D.0, clip::D.1);
    let a_abs = (high - low).clamp(clip::ABS_A.0, clip::ABS_A.1);
    let a = -a_abs;

    // Midpoint lat-accel is the mean of the asymptotes; solve
    //     v² = a_mid / C   ⇒   C = a_mid / v²
    // then clamp into the runtime's allowed range.
    let a_mid = 0.5 * (low + high);
    let v_mps = self
      .transition_speed_mph
      .clamp(clip::TRANSITION_MPH.0, clip::TRANSITION_MPH.1)
      * MPH_TO_MS;
    let c = (a_mid / (v_mps * v_mps)).clamp(clip::C.0, clip::C.1);

    let b = -sharpness_to_abs_b(self.sharpness);

    SigmoidParams {
      a,
      b,
      c,
      d,
      min_lat: low,
      max_lat: high,
    }
  }

  /// Recover plain knobs from a raw sigmoid (best effort; loses the extra
  /// degree of freedom between clamp vs asymptote).
  pub fn from_sigmoid(s: &SigmoidParams) -> Self {
    let low = (s.a + s.d).max(s.min_lat);
    let high = s.d.min(s.max_lat);
    let a_mid = 0.5 * (low + high);
    let v_mps = (a_mid / s.c.max(1e-9)).sqrt();
    Self {
      tight_curve_accel: low,
      straight_road_accel: high,
      transition_speed_mph: (v_mps * MS_TO_MPH)
        .clamp(clip::TRANSITION_MPH.0, clip::TRANSITION_MPH.1),
      sharpness: abs_b_to_sharpness(s.b.abs()),
    }
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn round_trip_default_is_stable() {
    let raw = SigmoidParams::default();
    let plain = PlainKnobs::from_sigmoid(&raw);
    let raw2 = plain.to_sigmoid();
    // Because min_lat default (1.8) is below the raw A+D asymptote (2.347),
    // to_sigmoid snaps min_lat up to low=2.347. That's an intentional "what
    // you see is what you get" collapse: the re-round-tripped plain is
    // identical to the original.
    let plain2 = PlainKnobs::from_sigmoid(&raw2);
    assert!((plain.tight_curve_accel - plain2.tight_curve_accel).abs() < 1e-6);
    assert!((plain.straight_road_accel - plain2.straight_road_accel).abs() < 1e-6);
    assert!((plain.transition_speed_mph - plain2.transition_speed_mph).abs() < 0.1);
    assert!((plain.sharpness - plain2.sharpness).abs() < 1e-6);
  }

  #[test]
  fn sharpness_log_mapping() {
    assert!((sharpness_to_abs_b(0.0) - 100.0).abs() < 1e-6);
    assert!((sharpness_to_abs_b(10.0) - 1.0e5).abs() < 1.0);
    // Monotonic.
    let s = [0.0, 2.5, 5.0, 7.5, 10.0];
    let b: Vec<f64> = s.iter().copied().map(sharpness_to_abs_b).collect();
    for w in b.windows(2) {
      assert!(w[1] > w[0]);
    }
  }

  #[test]
  fn transition_speed_places_midpoint_at_requested_v() {
    let k = PlainKnobs {
      tight_curve_accel: 2.0,
      straight_road_accel: 4.5,
      transition_speed_mph: 55.0,
      sharpness: 5.0,
    };
    let s = k.to_sigmoid();
    let a_mid = 0.5 * (k.tight_curve_accel + k.straight_road_accel);
    let v_mph = (a_mid / s.c).sqrt() * MS_TO_MPH;
    assert!((v_mph - 55.0).abs() < 0.5);
  }

  #[test]
  fn clamps_are_respected() {
    let k = PlainKnobs {
      tight_curve_accel: 0.1,      // below MIN_LAT.0
      straight_road_accel: 10.0,   // above MAX_LAT.1
      transition_speed_mph: 500.0, // absurd
      sharpness: -3.0,             // below range
    };
    let s = k.to_sigmoid();
    assert!(s.min_lat >= clip::MIN_LAT.0 - 1e-9);
    assert!(s.max_lat <= clip::MAX_LAT.1 + 1e-9);
    assert!(s.d >= clip::D.0 - 1e-9 && s.d <= clip::D.1 + 1e-9);
    assert!(s.a.abs() >= clip::ABS_A.0 - 1e-9 && s.a.abs() <= clip::ABS_A.1 + 1e-9);
    assert!(s.b.abs() >= clip::ABS_B.0 - 1e-9 && s.b.abs() <= clip::ABS_B.1 + 1e-9);
    assert!(s.c >= clip::C.0 - 1e-12 && s.c <= clip::C.1 + 1e-12);
  }
}
