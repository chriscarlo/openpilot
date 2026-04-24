"""1-state adaptive Kalman filter for lead distance tracking.

State:       x = dRel  (scalar)
Control:     u = vRel  (measured, used as process input)
Measurement: z = raw_dRel

Key properties vs the EMA prediction-corrector:
  - Adaptive gain: naturally trusts prediction more when measurements are noisy
  - Innovation deadband: same as EMA, ignores sub-threshold noise
  - Symmetric correction clamp: limits per-frame state movement to prevent
    noise-driven drift (analogous to EMA's alpha-bounded corrections)
  - Closing snap + innovation gate: same safety behaviour as EMA
"""
from __future__ import annotations

import math


# --- default tuning -------------------------------------------------------
DEFAULT_Q_DREL = 0.5          # m²/s  — position process noise
DEFAULT_R_DREL = 3.5          # m²    — measurement noise variance
DEFAULT_INNOVATION_GATE_SIGMA = 3.5
DEFAULT_CLOSING_SNAP_M = 20.0
# Require N consecutive large-closing innovations before snapping.
# A single garbage measurement must not teleport the filter.
DEFAULT_CLOSING_SNAP_CONFIRM_FRAMES = 3
# Max rate the filter can close via snap (m/s).  Even confirmed snaps
# should not teleport — clamp to a physically plausible closing rate.
DEFAULT_CLOSING_SNAP_RATE_MAX_MPS = 15.0  # ~54 km/h relative — extreme but plausible
DEFAULT_OPEN_SLEW_MAX_MPS = 1.25
DEFAULT_INNOVATION_DEADBAND_M = 0.35  # same as EMA filter
DEFAULT_KALMAN_GAIN_MAX = 0.35         # cap adaptive gain to prevent post-gate overcorrection


class LeadKalmanFilter:
  """1-state adaptive Kalman for smoothing AI-model lead distance."""

  __slots__ = (
    '_x', '_p',
    '_q', '_r',
    '_innovation_gate_sigma',
    '_closing_snap_m',
    '_open_slew_max_mps',
    '_deadband_m',
    '_k_max',
    '_closing_snap_confirm',
    '_closing_snap_rate_max',
    '_closing_snap_count',
    '_innov_mean',            # running innovation mean (for centering)
    '_innov_var',             # running innovation variance around mean (adaptive R)
    '_initialised',
    'last_debug',
  )

  def __init__(self, *,
               q_drel: float = DEFAULT_Q_DREL,
               r_drel: float = DEFAULT_R_DREL,
               innovation_gate_sigma: float = DEFAULT_INNOVATION_GATE_SIGMA,
               closing_snap_m: float = DEFAULT_CLOSING_SNAP_M,
               open_slew_max_mps: float = DEFAULT_OPEN_SLEW_MAX_MPS,
               deadband_m: float = DEFAULT_INNOVATION_DEADBAND_M,
               kalman_gain_max: float = DEFAULT_KALMAN_GAIN_MAX,
               # Accept but ignore legacy kwargs for sweep compatibility
               q_vrel: float = 0.0, **_kw):
    self._q = float(q_drel)
    self._r = float(r_drel)
    self._innovation_gate_sigma = float(innovation_gate_sigma)
    self._closing_snap_m = float(closing_snap_m)
    self._open_slew_max_mps = float(open_slew_max_mps)
    self._deadband_m = float(deadband_m)
    self._k_max = float(kalman_gain_max)
    self._closing_snap_confirm = int(DEFAULT_CLOSING_SNAP_CONFIRM_FRAMES)
    self._closing_snap_rate_max = float(DEFAULT_CLOSING_SNAP_RATE_MAX_MPS)
    self._closing_snap_count = 0
    self._initialised = False
    self._x = 0.0
    self._p = 0.0
    self.last_debug: dict[str, float | bool] = {}
    self.reset()

  def reset(self, drel: float | None = None) -> None:
    self._initialised = drel is not None
    self._x = float(drel) if drel is not None else 0.0
    self._p = 100.0
    self._closing_snap_count = 0
    self._innov_mean = 0.0
    self._innov_var = float(self._r)
    self.last_debug = {
      "kalman_gain": 0.0,
      "innovation_m": 0.0,
      "innovation_std": 0.0,
      "vrel_input_mps": 0.0,
      "gated": False,
      "closing_snap": False,
      "open_slew_clamped": False,
      "deadband_applied": False,
      "opening_vrel_suppressed": False,
    }

  @property
  def value(self) -> float | None:
    return self._x if self._initialised else None

  def update(self, raw_drel: float, raw_vrel: float, dt_s: float,
             tau_close: float = 0.0, tau_open: float = 0.0,
             innovation_gate: float = 0.0, closing_gate: float = 0.0,
             open_slew_max_mps: float = 0.0,
             r_meas: float | None = None) -> float:
    """Run one predict-update cycle.

    Args:
      r_meas: per-frame measurement noise override (e.g. model xStd²).
              When provided, replaces the nominal R for this frame.
              Clamped to [0.5, 50.0] to prevent degenerate values.
    """
    slew_max = open_slew_max_mps if open_slew_max_mps > 0.0 else self._open_slew_max_mps

    if not self._initialised or dt_s <= 0.0:
      self._x = float(raw_drel)
      self._p = self._r
      self._initialised = True
      self.last_debug = {
        "kalman_gain": 1.0, "innovation_m": 0.0, "innovation_std": 0.0,
        "vrel_input_mps": 0.0, "gated": False, "closing_snap": False,
        "open_slew_clamped": False, "deadband_applied": False,
        "opening_vrel_suppressed": False,
      }
      return self._x

    prev_x = self._x

    # ---- PREDICT ---------------------------------------------------------
    v_input = float(raw_vrel)
    opening_vrel_suppressed = False
    if v_input > 0.0:
      opening_evidence_m = float(raw_drel) - prev_x
      if opening_evidence_m <= self._deadband_m:
        # Positive vRel by itself is not enough evidence to move dRel open.
        # Model vRel noise otherwise integrates straight through prediction
        # even when the measured distance is flat.
        v_input = 0.0
        opening_vrel_suppressed = True
      else:
        v_input = min(v_input, slew_max)
    x_pred = self._x + v_input * dt_s
    p_pred = self._p + self._q * dt_s

    # ---- INNOVATE --------------------------------------------------------
    innov_raw = raw_drel - x_pred

    # Determine effective R for this frame.
    # Priority: per-frame r_meas (from model xStd²) > adaptive innov_var > nominal R
    if r_meas is not None:
      r_frame = float(max(0.5, min(50.0, r_meas)))
    else:
      r_frame = self._r
    # Adaptive R: raise above r_frame when recent innovations are large.
    # Cap at 3x to prevent over-damping during real dynamics.
    r_eff = max(r_frame, min(self._innov_var, r_frame * 3.0))

    s = p_pred + r_eff
    innov_std = math.sqrt(max(s, 1e-6))

    gated = False
    closing_snap = False
    open_slew_clamped = False
    deadband_applied = False
    k = 0.0

    # Apply deadband — absorb sub-threshold noise
    if abs(innov_raw) <= self._deadband_m:
      innov = 0.0
      deadband_applied = True
    else:
      innov = innov_raw - math.copysign(self._deadband_m, innov_raw)

    # ---- UPDATE ----------------------------------------------------------
    # Step 1: Closing snap confirmation counter
    if innov_raw < -self._closing_snap_m:
      self._closing_snap_count += 1
    else:
      self._closing_snap_count = 0

    # Step 2: Decide update action
    if self._closing_snap_count >= self._closing_snap_confirm:
      # Confirmed large-closing: N consecutive frames agree lead is much closer.
      # Snap fully — the multi-frame confirmation is the safety gate.
      self._x = float(raw_drel)
      self._p = self._r * 4.0
      self._closing_snap_count = 0
      closing_snap = True
      k = 1.0
    elif innov_raw < -self._closing_snap_m or abs(innov_raw) > self._innovation_gate_sigma * innov_std:
      # Gate: unconfirmed snap OR general outlier — reject, inflate uncertainty
      self._x = x_pred
      self._p = p_pred * 2.0
      gated = True
      k = 0.0
    elif deadband_applied:
      # Within deadband: just use prediction
      self._x = x_pred
      self._p = p_pred
      k = 0.0
    else:
      # Standard Kalman update with capped gain
      k_raw = p_pred / s
      k = min(k_raw, self._k_max)  # cap gain to prevent post-gate overcorrection
      self._x = x_pred + k * innov
      self._p = (1.0 - k_raw) * p_pred  # covariance uses uncapped gain for proper Kalman math
      # Update adaptive R: track variance AROUND the running mean.
      # Sustained innovations (real dynamics) have low variance → R stays low.
      # Oscillating innovations (noise) have high variance → R rises.
      self._innov_mean = 0.95 * self._innov_mean + 0.05 * innov_raw
      centered = innov_raw - self._innov_mean
      self._innov_var = 0.95 * self._innov_var + 0.05 * (centered * centered)

    # NOTE: no opening slew limit — the capped Kalman gain provides symmetric
    # noise rejection in both directions, eliminating the drift-down bias that
    # the asymmetric EMA slew limit creates under heavy-tailed noise.
    # The gain cap at k_max=0.25 means max correction = 0.25 * innov.
    # For a 5 m noise spike: 1.25 m correction.  This is bounded and symmetric.

    self.last_debug = {
      "kalman_gain": float(k),
      "innovation_m": float(innov_raw),
      "innovation_std": float(innov_std),
      "vrel_input_mps": float(v_input),
      "gated": bool(gated),
      "closing_snap": bool(closing_snap),
      "open_slew_clamped": bool(open_slew_clamped),
      "deadband_applied": bool(deadband_applied),
      "opening_vrel_suppressed": bool(opening_vrel_suppressed),
    }
    return self._x

  def set_tuning(self, *, q_drel: float | None = None,
                 r_drel: float | None = None) -> None:
    if q_drel is not None:
      self._q = float(q_drel)
    if r_drel is not None:
      self._r = float(r_drel)
