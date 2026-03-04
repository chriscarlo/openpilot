import numpy as np
import time
import math
import json
import os
from dataclasses import dataclass
from enum import IntEnum

from cereal import custom
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.numpy_fast import clip
from opendbc.car.common.conversions import Conversions as CV
try:
  from opendbc.car.vehicle_model import VehicleModel
except Exception:
  VehicleModel = None
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants
from .vision_turn_params import update_vtsc_params
try:
  from .vtsc_curve_tuning import Q_CURVE_ENABLED, Q_CURVE_POINTS
except Exception:
  Q_CURVE_ENABLED = False
  Q_CURVE_POINTS = []

VisionTurnControllerState = custom.LongitudinalPlanSP.VisionTurnSpeedControl.VisionTurnSpeedControlState

N_POINTS = int(min(33, len(ModelConstants.T_IDXS)))  # Use available trajectory points

# ===== Freeway Fail-Open Guard Tunables =====
# If path is straight, visibility is long, and confidence is good, ignore occlusion effects.
FREEWAY_CURV_EPS = 1e-5       # effectively straight (1/m)
FREEWAY_MIN_VISIBLE_M = 120.0 # visible horizon long enough (m)
FREEWAY_MIN_CONF = 0.60       # path/model confidence threshold

# ===== Freeway Cap Hold (Planner Response Latency) =====
# At freeway speeds, the VTSC cap can briefly dip for a single model frame as curvature predictions
# fluctuate. The longitudinal planner/MPC typically cannot respond meaningfully to these sub-0.5s
# pulses, which presents as "VTSC is late to slow". Hold material cap reductions briefly so the
# planner sees a stable target and begins braking sooner.
VTURN_HOLD_MIN_V_MPS = 27.0    # only engage hold above ~60 mph
VTURN_HOLD_DELTA_MPS = 1.0    # only hold when cap reduces cruise by ≥ this
VTURN_HOLD_S = 1.2            # tuned from rlogs: typical planner response ≈ 0.9–1.2s
# Under degraded vision we still want a hold, but make it shorter to avoid "sticky" slowdowns after
# a curve ends (especially when the UI state machine keeps predicted curvature slightly non-zero).
VTURN_HOLD_S_OCCLUDED = 0.85
# Minimum predicted lateral acceleration (m/s^2) to treat as real turn evidence for hold gating.
_ENTERING_PRED_LAT_ACC_TH = 1.3

# Global model-horizon phase advance (seconds). Shifts both braking onset and post-apex
# release earlier to compensate planner/actuation latency.
VTSC_TRAJECTORY_PHASE_ADVANCE_S = 1.0

# ===== Feature Flags & Thresholds =====
# Use a large sentinel for "no cap" speed contributions when disabling a channel
INF_SPEED = 1e9

# ===== Steering-curvature fallback =====
# If the model curvature stays near-flat while the steering input indicates a real curve,
# use steering-derived curvature as a last-resort signal to avoid entering a curve at cruise.
STEER_CURVATURE_FALLBACK_MODEL_KAPPA_MAX = 0.003  # 1/m: model says "straight-ish"
STEER_CURVATURE_FALLBACK_MIN_KAPPA = 0.003        # 1/m: car is actually turning
STEER_CURVATURE_FALLBACK_MIN_V_MPS = 13.0         # only consider at ~29 mph+

# ===== Severe-confidence overshoot conservatism =====
# If lane-line confidence is extremely low, the model often "discovers" tight off-ramp curvature late.
# Make overshoot detection slightly more conservative so VTSC begins slowing earlier in SEVERE/LOST,
# without changing the baseline curvature→speed mapping used in good visibility.
SEVERE_OVERSHOOT_SPEED_SCALE_MIN = 0.90  # multiplicative on safe speeds (lower => more conservative)

# Hidden-turn early deceleration feature flag (disabled fully per request)
HIDDEN_TURN_ENABLED = False

# Highway override threshold: start any bypass/relax behavior at 55 mph
HIGHWAY_MIN_MPH = 55.0
HIGHWAY_MIN_MPS = float(HIGHWAY_MIN_MPH * CV.MPH_TO_MS)

# ===== Low-speed occlusion margin relax =====
# At very low speeds on effectively-straight roads, VTSC's occlusion math can trip "negative margin"
# and enforce a decel/hold that feels like a crawl. Allow a small override in this regime.
LOW_SPEED_MARGIN_MAX_V_MPS = 12.5      # taper ends ≈28 mph (dominates town speeds)
LOW_SPEED_MARGIN_CURV_THRESH = 3.5e-4  # below this treat as effectively straight

# Lead-bypass headway floor: avoid inflated headway at crawl speeds behind a lead
OCCL_BYPASS_HEADWAY_V_FLOOR_MPS = 5.0  # ~11 mph
# Lead-bypass low-speed close-lead fallback
OCCL_BYPASS_LOW_SPEED_V_MPS = 7.0      # ~16 mph
OCCL_BYPASS_LEAD_D_REL_MAX_M = 27.0    # ~89 ft

# ===== PSI / Occlusion arbitration tunables (defaults; overridden via Params) =====
# PSI gate to qualify occlusion influence during FOV occlusion
PSI_THRESH_RAD = 0.020     # default gate open threshold (radians)
PSI_HYST_RAD  = 0.005      # hysteresis
# Double-cap guard: if pre-cap target already ≤ occl vmin + eps, don't re-apply occlusion cap
DOUBLE_CAP_EPS_MPS = 0.30
# fov_exit recovery when confidence is near-zero and psi gate is closed
OCCL_CONF_FLOOR    = 0.05
FOV_EXIT_RELAX_S   = 0.60
OCCL_VMIN_NUDGE_MPS = 0.50

# ===== Map lookahead helpers =====
EARTH_R_M = 6371007.2

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
  c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
  return EARTH_R_M * c

def _xy_from_latlon_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
  """Equirectangular approximation in meters in an (east, north) local frame around (lat0, lon0)."""
  dlat = math.radians(lat - lat0)
  dlon = math.radians(lon - lon0)
  x_east = EARTH_R_M * dlon * math.cos(math.radians(lat0))
  y_north = EARTH_R_M * dlat
  return float(x_east), float(y_north)

# ===== ADAPTIVE DECELERATION SYSTEM =====
# Physics-based deceleration management for vision update lag scenarios
# Goal: Target comfort rates, but escalate to minimum decel/jerk needed to reach target speed at curve

# Comfort deceleration limits (m/s²) - primary targets
COMFORT_DECEL_LIMIT = -1.47  # -0.15g - comfortable deceleration
COMFORT_JERK_LIMIT = -2.0    # m/s³ - comfortable jerk

# Safety limits for adaptive escalation (m/s²)
MAX_ADAPTIVE_DECEL = -6.0    # System maximum deceleration
MAX_ADAPTIVE_JERK = -6.0     # System maximum jerk

# Default noise filtering parameters
DEFAULT_FILTER_ALPHA = 0.3      # EMA filter coefficient (0.1-0.9)
DEFAULT_HYSTERESIS_THRESHOLD = 0.15  # Hysteresis threshold (0.1-0.5)
DEFAULT_SAFETY_BIAS = 0.1        # Safety bias factor (0.0-0.5)

# ===== VISION OCCLUSION HANDLING (SIMPLIFIED) =====
# Smoothed confidence with a small hysteresis band; hold last curvature during occlusion
CONF_ALPHA = 0.28       # Faster EMA to track recovery
CONF_GOOD_TH = 0.70     # Threshold to (re)enter good vision
CONF_BAD_TH = 0.65      # Threshold to enter occlusion
# Public hysteresis thresholds (compatibility for tests)
CONFIDENCE_ENTER_PARTIAL = CONF_BAD_TH  # align status with control gate
CONFIDENCE_EXIT_TO_FULL = CONF_GOOD_TH  # align status with control gate
CONFIDENCE_ENTER_SEVERE = 0.45
CONFIDENCE_EXIT_TO_PARTIAL = 0.55

class VisionStatus(IntEnum):
    FULL_VISIBILITY = 0
    PARTIAL_OCCLUSION = 1
    SEVERE_OCCLUSION = 2
    VISION_LOST = 3


@dataclass
class VisionOcclusionState:
    """Smoothed confidence with monotonic-decay occlusion handling.

    Behavior while occluded (vision_good == False):
    - Pre-apex trend (tightening): grow curvature conservatively with distance using gamma_per_m.
    - Post-apex trend (easing): bounded decay toward a fraction of entry curvature to avoid crawl.
    Always maintains monotonic speed by blocking positive acceleration (enforced upstream).
    """
    last_valid_curvature: float = 0.0
    smoothed_confidence: float = 1.0
    vision_good: bool = True
    # Compatibility fields expected by tests
    vision_status: VisionStatus = VisionStatus.FULL_VISIBILITY
    confidence_decay_factor: float = 1.0
    extrapolated_curvature: float = 0.0
    good_vision_frames: int = 0
    occlusion_start_time: float = 0.0
    updated_once: bool = False
    alpha: float = CONF_ALPHA
    good_threshold: float = CONF_GOOD_TH
    bad_threshold: float = CONF_BAD_TH
    # Monotonic occlusion model state
    prev_curvature_good: float = 0.0
    occluded_since_time: float = 0.0
    reacquired_at: float = 0.0
    tail_started: bool = False
    tail_start_time: float = 0.0
    entry_curvature: float = 0.0
    est_curvature: float = 0.0
    trend_sign: int = 0
    last_time: float = 0.0
    distance_since_m: float = 0.0
    mode_monotonic: bool = True
    gamma_per_m: float = 2.5e-4
    lat_jerk_cap: float = 2.0
    vis_horizon_s: float = 1.2
    envelope_horizon_s: float = 1.0
    # Two-stage decay
    decay_tau_fast_s: float = 1.2
    decay_tau_slow_s: float = 2.0
    min_frac_initial: float = 0.6
    min_frac: float = 0.20
    # Trend multipliers (decreasing multiplier set to 1.0 per design)
    increasing_trend_mul: float = 1.0
    decreasing_trend_mul: float = 1.0
    # Dwell timers
    enter_dwell_s: float = 0.20
    exit_dwell_s: float = 0.10
    below_bad_time_s: float = 0.0
    above_good_time_s: float = 0.0
    # Last-known-good (LKG) anchor for occlusion re-anchoring
    _lkg_kappa: float = 0.0
    _lkg_time: float = 0.0
    _lkg_speed: float = 0.0
    _lkg_ramp_s: float = 0.9  # ramp time to allow occluded estimate to exceed LKG

    def update(self, current_curvature: float, vision_confidence: float, v_ego_or_tm, tm=None):
        # Support both 3-arg (tm) and 4-arg (v_ego, tm) signatures
        if tm is None:
            tm = float(v_ego_or_tm)
            v_ego = 0.0
        else:
            v_ego = float(v_ego_or_tm)
        # EMA smoothing of confidence
        self.smoothed_confidence = (1.0 - self.alpha) * self.smoothed_confidence + self.alpha * vision_confidence

        # Track time and distance progression
        dt = 0.0 if self.last_time == 0.0 else max(0.0, tm - self.last_time)
        self.last_time = tm
        # Dwell accumulation based on tri-state thresholds
        if self.smoothed_confidence < self.bad_threshold:
            self.below_bad_time_s += dt
            self.above_good_time_s = 0.0
        elif self.smoothed_confidence > self.good_threshold:
            self.above_good_time_s += dt
            self.below_bad_time_s = 0.0
        else:
            # Borderline band: reset both dwell timers
            self.below_bad_time_s = 0.0
            self.above_good_time_s = 0.0

        # State transitions with dwell
        if self.vision_good:
            # Enter occlusion only after dwell below bad
            if self.below_bad_time_s >= self.enter_dwell_s:
                self.vision_good = False
                self.occluded_since_time = tm
                self.occlusion_start_time = tm
                self.distance_since_m = 0.0
                self.tail_started = False
                self.tail_start_time = 0.0
                base = self.last_valid_curvature if self.updated_once else current_curvature
                self.entry_curvature = max(0.0, float(base))
                self.est_curvature = self.entry_curvature
                dcur = current_curvature - self.prev_curvature_good
                self.trend_sign = 1 if dcur > 0.0 else (-1 if dcur < 0.0 else 0)
                # reset enter dwell
                self.below_bad_time_s = 0.0
            else:
                # Vision is good but not entering occlusion: defer last_valid_curvature updates
                # to the compatibility section (requires 3 strong-good frames)
                self.prev_curvature_good = current_curvature
                self.updated_once = True
        else:
            # Exit occlusion only after dwell above good
            if self.above_good_time_s >= self.exit_dwell_s:
                self.vision_good = True
                self.last_valid_curvature = current_curvature
                # Reacquired: mark timestamp for downstream smoothing aids
                self.reacquired_at = tm
                self.prev_curvature_good = current_curvature
                self.updated_once = True
                self.occluded_since_time = 0.0
                self.distance_since_m = 0.0
                self.tail_started = False
                self.tail_start_time = 0.0
                self.entry_curvature = 0.0
                self.est_curvature = 0.0
                self.trend_sign = 0
                self.above_good_time_s = 0.0
            else:
                # Remain occluded: update distance and estimate curvature for the tail only
                self.distance_since_m += v_ego * dt
                if not self.mode_monotonic:
                    self.est_curvature = self.last_valid_curvature
                else:
                    elapsed = max(0.0, tm - self.occluded_since_time)
                    # Visible horizon in meters
                    s_vis = max(0.0, self.vis_horizon_s * max(0.0, v_ego))
                    s_tail_raw = max(0.0, self.distance_since_m - s_vis)
                    # Tail growth window timing
                    if (s_tail_raw > 1e-3) and (not self.tail_started):
                        self.tail_started = True
                        self.tail_start_time = tm
                    tail_elapsed = (tm - self.tail_start_time) if self.tail_started else 0.0
                    # Curvature- and speed-aware tail allowance
                    k_now_for_window = max(0.0, max(self.entry_curvature, self.est_curvature))
                    # Tail window selection: prioritize safety but ensure sufficient growth at moderate speeds
                    # Baseline: very short for sweepers, long for tight curves
                    if k_now_for_window <= 0.0035:
                        t_allow = 0.10
                    else:
                        t_allow = 1.20
                    # Production alignment: at mountain speeds (≤ ~30 m/s), allow a larger tail window
                    # so curvature growth can reflect an upcoming bend even when entry curvature was near-zero.
                    if v_ego <= 30.0:
                        t_allow = max(t_allow, 0.80)
                    # Clamp to configured envelope horizon
                    t_allow = max(0.10, min(t_allow, self.envelope_horizon_s))
                    s_tail_allow = max(0.0, v_ego) * t_allow
                    s_tail = min(s_tail_raw, s_tail_allow)

                    if self.trend_sign >= 0:
                        # Growth only beyond visible horizon; allow conservative growth with jerk-capped gamma
                        if (s_tail > 0.0) and (tail_elapsed <= self.envelope_horizon_s):
                            # Speed-based cap (piecewise, interpolated) + jerk cap
                            v_cap_speed = 22.0  # soften jerk cap below ~49 mph to preserve mountain decel
                            eff_v = min(max(0.0, v_ego), v_cap_speed)
                            gamma_cap_jerk = getattr(self, 'lat_jerk_cap', 2.0) / max(eff_v**3, 1e-3)
                            # Piecewise speed cap table (m/s -> gamma cap)
                            sp = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0]
                            gp = [8e-4,6e-4,5e-4,4e-4,3e-4,1.8e-4,1.2e-4,1.0e-4]
                            vv = max(0.0, v_ego)
                            if vv <= sp[0]:
                                gamma_cap_speed = gp[0]
                            elif vv >= sp[-1]:
                                gamma_cap_speed = gp[-1]
                            else:
                                for i in range(len(sp)-1):
                                    if sp[i] <= vv <= sp[i+1]:
                                        t = (vv - sp[i]) / max(1e-6, (sp[i+1] - sp[i]))
                                        gamma_cap_speed = gp[i] + (gp[i+1] - gp[i]) * t
                                        break
                            # Allow growth up to configured gamma with both speed and jerk caps
                            gamma_eff = min(self.gamma_per_m, gamma_cap_speed, gamma_cap_jerk)
                            # Optional trend multipliers (default 1.0)
                            try:
                                inc_mul = float(getattr(self, 'increasing_trend_mul', 1.0))
                            except Exception:
                                inc_mul = 1.0
                            self.est_curvature = max(0.0, self.entry_curvature + inc_mul * (gamma_eff * s_tail))
                        else:
                            # Freeze growth beyond tail window
                            self.est_curvature = max(0.0, self.est_curvature)
                    else:
                        # Two-stage decay, with time-to-floor behavior
                        if self.entry_curvature <= 0.0:
                            self.est_curvature = 0.0
                        else:
                            # Floor ramps down after 0.7s occluded
                            floor_frac = self.min_frac if elapsed >= 0.7 else self.min_frac_initial
                            floor_val = floor_frac * self.entry_curvature
                            tau = self.decay_tau_fast_s if elapsed <= 0.8 else self.decay_tau_slow_s
                            decay = math.exp(-elapsed / max(1e-3, tau))
                            try:
                                dec_mul = float(getattr(self, 'decreasing_trend_mul', 1.0))
                            except Exception:
                                dec_mul = 1.0
                            self.est_curvature = max(floor_val, (self.entry_curvature * decay) * dec_mul)


        # ===== Compatibility: expose VisionStatus with hysteresis thresholds =====
        c = float(vision_confidence)
        prev = self.vision_status
        # Immediate escalate to SEVERE on very low confidence
        if c < CONFIDENCE_ENTER_SEVERE:
            self.vision_status = VisionStatus.SEVERE_OCCLUSION
        else:
            if prev == VisionStatus.FULL_VISIBILITY:
                # Leave FULL only below 0.75
                self.vision_status = VisionStatus.PARTIAL_OCCLUSION if c < CONFIDENCE_ENTER_PARTIAL else VisionStatus.FULL_VISIBILITY
            elif prev == VisionStatus.PARTIAL_OCCLUSION:
                # Return to FULL at 0.85; otherwise remain PARTIAL
                if c >= CONFIDENCE_EXIT_TO_FULL:
                    self.vision_status = VisionStatus.FULL_VISIBILITY
                else:
                    self.vision_status = VisionStatus.PARTIAL_OCCLUSION
            elif prev == VisionStatus.SEVERE_OCCLUSION:
                # Recover to PARTIAL at 0.55
                self.vision_status = VisionStatus.PARTIAL_OCCLUSION if c >= CONFIDENCE_EXIT_TO_PARTIAL else VisionStatus.SEVERE_OCCLUSION
            else:
                # VISION_LOST -> treat as severe until recovery
                self.vision_status = VisionStatus.PARTIAL_OCCLUSION if c >= CONFIDENCE_EXIT_TO_PARTIAL else VisionStatus.VISION_LOST
        # Track compat timestamps
        if self.vision_status != VisionStatus.FULL_VISIBILITY and self.occlusion_start_time == 0.0:
            self.occlusion_start_time = tm
        if self.vision_status == VisionStatus.FULL_VISIBILITY:
            self.reacquired_at = tm
# ===== Compatibility: last_valid_curvature update after 3 strong good frames =====
        # Count strong-good frames regardless of current status for compatibility
        if c >= CONFIDENCE_EXIT_TO_FULL:
            self.good_vision_frames = int(self.good_vision_frames) + 1
        else:
            self.good_vision_frames = 0
        if self.good_vision_frames >= 3:
            self.last_valid_curvature = float(current_curvature)
            self.updated_once = True

        # ===== Compatibility: no-decay fix =====
        self.confidence_decay_factor = 1.0
        # Vision-first extrapolated curvature with LKG anchoring during occlusion.
        # - When vision is good: update LKG anchor and use current curvature.
        # - When occluded: blend from LKG toward occluded estimate with a short time ramp.
        now_ts = float(tm)
        if self.vision_good:
            try:
                self._lkg_kappa = float(abs(current_curvature))
                self._lkg_time = now_ts
                # curvature_to_speed defined below; safe to call at runtime
                self._lkg_speed = float(curvature_to_speed(max(1e-8, self._lkg_kappa)))
            except Exception:
                pass
            self.extrapolated_curvature = float(current_curvature)
        else:
            try:
                k_occ = float(max(0.0, self.est_curvature))
                k_lkg = float(getattr(self, '_lkg_kappa', k_occ))
                age_s = max(0.0, now_ts - float(getattr(self, '_lkg_time', 0.0)))
                ramp_s = float(getattr(self, '_lkg_ramp_s', 1.5))
                alpha = min(1.0, age_s / max(0.1, ramp_s))
                k_blend = k_lkg + alpha * max(0.0, k_occ - k_lkg)
                self.extrapolated_curvature = float(k_blend)
            except Exception:
                self.extrapolated_curvature = float(max(0.0, self.est_curvature))

# ===== ORIGINAL PHYSICS-BASED VTSC CONSTANTS =====
_MIN_V = 2.24  # Do not operate under 5mph (was 5.6 m/s = 12.5mph)

_DEBUG = False

# Advanced vision-based functions extracted from chauffeur_vtsc.py

# Constants for advanced curvature-based speed calculation
MAX_SPEED_DEFAULT = 70.0  # m/s, fallback for straight roads (overridden by param)
SPEED_INCREASE_FACTOR = 1.0  # Global multiplier on target speeds (overridden by param)

# Physics sigmoid tunables (overridden by params)
PHYSICS_A = -2.300000    # Amplitude (deepened to keep tight-turn lat_accel ≈ D+A ≈ 1.97 m/s²)
PHYSICS_B = -2000.000000 # Steepness
PHYSICS_C = 0.004778     # Transition center (1/m)
PHYSICS_D = 4.270000     # Baseline (m/s²; raised to target ~70 mph at k≈0.004 sweeper curvature)
PHYSICS_MIN_LAT_ACCEL = 1.8
PHYSICS_MAX_LAT_ACCEL = 3.90  # raised to permit 70 mph at k=0.004 (was 3.12 → 61 mph)

# Low-speed bias (applied as +Δ mph under a taper)
LOW_SPEED_BIAS_MPH = 5.0         # +speed boost at tight curves (tapers to 0 by END_MPH)
LOW_SPEED_BIAS_END_MPH = 55.0    # taper covers up to ~55 mph base speed (was 50.0)

# ===== Hidden-turn early deceleration trigger (occlusion-only, sub-65 mph) =====
# Allows jerk-limited early braking when a short-horizon physics deficit is provably large
# despite a transiently positive visible-margin condition.
# Align hidden-turn speed gate with highway threshold (~55 mph)
HIDDEN_TURN_V_MAX_MPS = HIGHWAY_MIN_MPS  # ~55 mph; above this we run pure physics
HIDDEN_TURN_T_H_S = 1.8        # short horizon (~40 m at 50 mph)
HIDDEN_TURN_DELTA_V_MPS = 2.0  # ~6 mph speed gap
HIDDEN_TURN_MIN_OCC_S = 0.30   # require persisting occlusion ≥ 600 ms
HIDDEN_TURN_AVAIL_SCALE = 0.50  # slight nudge for earlier activation
HIDDEN_TURN_PHASE_S = 2.0      # only within first ~2 s of occlusion
HIDDEN_TURN_HEADING_WIN_S = 1.2
HIDDEN_TURN_VIS_HEADING_MAX_RAD = math.radians(6.0)  # ~6°, "straight enough"

def _physics_based_lateral_acceleration(curvature: float) -> float:
    """
    Continuous sigmoid-based lateral acceleration function (scipy optimized)
    
    Replaces piecewise function with smooth continuous alternative that provides:
    - Highway zone (≤0.0029): Maintains ~3.12 m/s² performance  
    - Transition zone: Smooth exponential decay
    - Tight curves (>0.0053): 20%+ more aggressive than original (1.8-2.04 vs 1.5-1.7 m/s²)
    
    Mathematical model: Optimized sigmoid with R² = 0.9747
    Benefits: Perfect continuity, no discontinuous jumps, more aggressive low-speed cornering
    """
    curvature = max(1e-8, min(curvature, 1.0))
    # Use globally-tunable sigmoid parameters
    result = PHYSICS_A / (1.0 + math.exp(PHYSICS_B * (curvature - PHYSICS_C))) + PHYSICS_D
    return max(PHYSICS_MIN_LAT_ACCEL, min(result, PHYSICS_MAX_LAT_ACCEL))

def _q_curve_multiplier(abs_curvature_meters: float) -> float:
    if not Q_CURVE_ENABLED or len(Q_CURVE_POINTS) < 2:
        return 1.0
    if not (abs_curvature_meters > 0.0 and math.isfinite(abs_curvature_meters)):
        return 1.0

    pts = []
    for k, q in Q_CURVE_POINTS:
        try:
            kf = float(k)
            qf = float(q)
        except Exception:
            continue
        if not (kf > 0.0 and math.isfinite(qf)):
            continue
        pts.append((kf, qf))
    if len(pts) < 2:
        return 1.0
    pts.sort(key=lambda kv: kv[0])

    kappa = clip(float(abs_curvature_meters), pts[0][0], pts[-1][0])
    logk = math.log10(max(kappa, 1e-12))

    for i in range(len(pts) - 1):
        k0, q0 = pts[i]
        k1, q1 = pts[i + 1]
        if kappa <= k1:
            log0 = math.log10(max(k0, 1e-12))
            log1 = math.log10(max(k1, 1e-12))
            if log1 <= log0:
                return clip(q0, 0.5, 1.5)
            t = (logk - log0) / (log1 - log0)
            q = q0 + (q1 - q0) * t
            return clip(q, 0.5, 1.5)
    return clip(pts[-1][1], 0.5, 1.5)

def curvature_to_speed(abs_curvature_meters: float) -> float:
    """FIXED: Calculates target speed (m/s) directly from curvature with NO SCALING HACK"""
    if abs_curvature_meters < 1e-7:  # Handle straight roads
        return MAX_SPEED_DEFAULT

    # Get safe lateral acceleration using tuned sigmoid
    safe_lat_accel = _physics_based_lateral_acceleration(abs_curvature_meters)

    # Calculate speed using physics formula v = sqrt(a / k) with CONSISTENT curvature
    try:
        base_speed_mps = math.sqrt(safe_lat_accel / abs_curvature_meters)
    except (ValueError, ZeroDivisionError):
        base_speed_mps = 0.0

    # Apply simple low-speed bias in mph with taper
    base_speed_mph = base_speed_mps * CV.MS_TO_MPH
    if LOW_SPEED_BIAS_MPH != 0.0 and base_speed_mph < LOW_SPEED_BIAS_END_MPH:
        taper = 1.0 - (base_speed_mph / max(LOW_SPEED_BIAS_END_MPH, 1e-3))
        base_speed_mph = base_speed_mph + LOW_SPEED_BIAS_MPH * max(0.0, min(1.0, taper))
        base_speed_mps = max(0.0, base_speed_mph * CV.MPH_TO_MS)

    # Apply speed increase factor and clip to reasonable maximum
    target_speed_mps = base_speed_mps * SPEED_INCREASE_FACTOR
    target_speed_mps = clip(target_speed_mps, 0.0, MAX_SPEED_DEFAULT)

    # Optional post-scale for tuning the curvature→speed relationship (multiplicative in speed)
    q = _q_curve_multiplier(abs_curvature_meters)
    return clip(target_speed_mps * q, 0.0, MAX_SPEED_DEFAULT)

def find_apexes_enhanced(curvature_array: np.ndarray, threshold: float = 5e-5, min_prominence: float = 1e-4) -> list:
    """
    Find local maxima (apex points) in curvature data with noise filtering.
    
    Args:
        curvature_array: Array of curvature values
        threshold: Minimum curvature to consider as potential apex
        min_prominence: Minimum peak prominence to filter noise
        
    Returns:
        List of indices where apexes are detected
    """
    if len(curvature_array) < 3:
        return []

    # Apply light smoothing (3-point moving average) to reduce noise
    # Use 'same' mode to maintain array size
    smoothed = np.convolve(curvature_array, [0.25, 0.5, 0.25], mode='same')

    apex_indices = []
    for i in range(1, len(smoothed) - 1):
        # Check if this is a local maximum above threshold
        if (smoothed[i] > threshold and
            smoothed[i] >= smoothed[i + 1] and
            smoothed[i] > smoothed[i - 1]):
            # Calculate prominence (peak height above neighbors)
            prominence = smoothed[i] - min(smoothed[i-1], smoothed[i+1])
            if prominence > min_prominence:
                apex_indices.append(i)

    return apex_indices

def calculate_anticipation_time(v_ego_ms: float, target_speed_ms: float, max_pred_lat_acc: float, aggressiveness: float = 1.0) -> float:
    """
    OPTIMIZED anticipation time calculation with research-validated parameters.
    17.6% improvement over original algorithm through Bayesian optimization.
    
    Optimized through synthetic testing across 108 scenarios covering:
    - Speed ranges: 5-85 mph across parking, residential, urban, highway contexts
    - Research-validated comfort deceleration limits (0.295g)
    - Human factors timing expectations from literature
    
    Key improvements:
    - Fixes "eternity at low speed" (parking: -5.5% timing)  
    - Fixes "insufficient buffer at high speed" (highway: +29% timing)
    - Context-aware scaling based on driving environment
    
    Args:
        v_ego_ms: Current vehicle speed (m/s)
        target_speed_ms: Target speed for curve (m/s)  
        max_pred_lat_acc: Maximum predicted lateral acceleration (m/s²)
        aggressiveness: User-configurable multiplier (0.5-2.0, default 1.0)
        
    Returns:
        Optimized anticipation time in seconds
    """

    # Optimized parameters from research study
    reaction_time_base = 1.185        # vs original 1.5s - faster response
    speed_normalization = 15.0        # vs original 20.0 - tuned for city driving
    speed_factor_min = 0.642          # vs original 0.7
    speed_factor_max = 1.975          # vs original 1.5 - wider range
    delta_factor_gain = 0.683         # vs original 0.5 - more sensitive
    delta_factor_max = 1.665          # vs original 1.5
    severity_normalization = 2.424    # vs original 1.5 - less lat acc impact
    timing_min = 0.565               # vs original 1.0 - allows faster reactions
    timing_max = 8.0                 # vs original 3.0 - wider range

    # Context-aware multipliers based on speed
    v_ego_mph = v_ego_ms * CV.MS_TO_MPH
    if v_ego_mph <= 15:
        context_multiplier = 0.973      # Parking: slightly faster
    elif v_ego_mph <= 35:
        context_multiplier = 1.144      # Residential: moderate increase
    elif v_ego_mph <= 55:
        context_multiplier = 1.200      # Urban: efficiency balance
    else:
        context_multiplier = 1.384      # Highway: maximum safety margin

    # Speed factor: Optimized scaling
    speed_factor = clip(v_ego_ms / speed_normalization, speed_factor_min, speed_factor_max)

    # Speed reduction factor: Enhanced sensitivity
    if v_ego_ms > 0.1:
        delta_ratio = (v_ego_ms - target_speed_ms) / v_ego_ms
        delta_factor = clip(1.0 + delta_ratio * delta_factor_gain, 1.0, delta_factor_max)
    else:
        delta_factor = 1.0

    # Curve severity factor: Simplified (optimization showed minimal impact)
    severity_factor = clip(max_pred_lat_acc / severity_normalization, 1.0, 1.0)

    # Calculate optimized timing
    base_timing = reaction_time_base * speed_factor * delta_factor * severity_factor
    timing = base_timing * context_multiplier * aggressiveness

    # Adjust max timing limit based on aggressiveness to allow more pre-emptive slowing
    adjusted_timing_max = timing_max * aggressiveness

    return clip(timing, timing_min, adjusted_timing_max)

def _debug(msg):
  if not _DEBUG:
    return
  print(msg)

def _description_for_state(turn_controller_state):
  if turn_controller_state == VisionTurnControllerState.disabled:
    return 'DISABLED'
  if turn_controller_state == VisionTurnControllerState.entering:
    return 'ENTERING'
  if turn_controller_state == VisionTurnControllerState.turning:
    return 'TURNING'
  if turn_controller_state == VisionTurnControllerState.leaving:
    return 'LEAVING'


class VisionTurnController:
  def __init__(self, CP):
    self._params = Params()
    try:
      self._mem_params = Params("/dev/shm/params")
    except Exception:
      self._mem_params = self._params
    self._CP = CP
    self._vm = None
    try:
      if VehicleModel is not None:
        self._vm = VehicleModel(CP)
    except Exception:
      self._vm = None
    self._steering_angle_deg = 0.0
    self._op_enabled = False
    self._gas_pressed = False
    # Defaults; Params applied by update_vtsc_params(force=True) below
    self._is_enabled = False
    # User-configurable aggressiveness for pre-emptive slowing (0.5-2.0, default 1.0)
    # Higher values = earlier/more conservative slowing before curves
    self._aggressiveness = 1.0

    # Optional fixed lead time override (seconds). 0.0 = disabled
    self._fixed_lead_time_s = 0.0
    # Signed timing offsets (seconds): 0 = default timing, lower = earlier, higher = later.
    # Curve offset affects horizon interpretation; overshoot offset affects braking onset timing;
    # apex exit offset affects when post-apex acceleration logic begins.
    self._curve_phase_offset_s = 0.0
    self._overshoot_phase_offset_s = 0.0
    self._apex_exit_phase_offset_s = 0.0
    
    # ===== ADAPTIVE DECELERATION PARAMETERS =====
    # User-configurable noise filtering parameters
    self._filter_alpha = DEFAULT_FILTER_ALPHA
    self._base_filter_alpha = self._filter_alpha
    
    self._hysteresis_threshold = DEFAULT_HYSTERESIS_THRESHOLD
    
    self._safety_bias = DEFAULT_SAFETY_BIAS
    
    self._last_params_update = 0.
    self._v_cruise_setpoint = 0.
    self._v_ego = 0.
    self._a_ego = 0.
    self._a_target = 0.
    self._v_overshoot = 0.
    self._state = VisionTurnControllerState.disabled

    # ===== ADAPTIVE DECELERATION SYSTEM =====
    self._current_decel = 0.0
    self._filtered_decel_requirement = 0.0
    self._decel_hysteresis_state = False

    # ===== VISION OCCLUSION HANDLING =====
    self._occlusion_state = VisionOcclusionState()
    # Fast reacquisition window management
    self._base_filter_alpha = DEFAULT_FILTER_ALPHA
    self._fast_reacq_until = 0.0
    self._fast_reacq_alpha = 0.85
    self._fast_reacq_window_s = 0.90
    # Visibility barrier params (defaults; Params override in update)
    self._vis_horizon_s = 1.4
    self._vis_margin_m = 10.0
    self._gamma_per_meter = 0.00035
    self._lat_jerk_cap = 2.0
    # Sentinel: <= 0 disables cap in tests
    if self._lat_jerk_cap <= 0.0:
      self._lat_jerk_cap = 1e9
    # Anticipation moderation state
    self._prev_smoothed_conf = 1.0
    self._prev_filtered_curvature = 0.0
    self._anticipation_budget_window_start = 0.0
    self._cum_anticipation_reduction = 0.0
    self._last_high_conf_target_speed = 0.0
    self._anticipation_max_reduction_mps = 2.0

    # ===== Freeway cap hold (avoid flicker) =====
    self._v_turn_hold_until = 0.0
    self._v_turn_hold_min = float(INF_SPEED)

    # Advanced controller state
    self._current_accel = 0.0
    self._prev_target_speed = 0.0
    # Smoothing bounds (tunable)
    self._max_decel = 3.5  # VisionTurnSpeedControlSmoothingMaxDecel
    self._max_jerk = 6.0   # VisionTurnSpeedControlSmoothingMaxJerk
    self._accel_to_decel_ratio = 1.3  # VisionTurnSpeedControlAccelToDecelRatio
    self._jerk_accel_multiplier = 2.0 # VisionTurnSpeedControlJerkAccelMultiplier
    self._max_accel = self._accel_to_decel_ratio * self._max_decel
    self._max_jerk_accel = self._jerk_accel_multiplier * self._max_jerk

    # EMA filtering for curvature (tunable)
    self._curvature_ema_ratio = 0.3  # VisionTurnSpeedControlCurvatureEMAFactor
    self._filtered_curvature = 0.0

    # Anticipatory deceleration state
    self._is_decelerating_for_curve = False

    # Anticipation/Overshoot planning tunables
    self._planning_decel_limit = 3.5            # VisionTurnSpeedControlPlanningDecelLimit (m/s²)
    self._overshoot_safety_margin = 1.2         # VisionTurnSpeedControlOvershootSafetyMargin (multiplier)
    self._overshoot_min_distance = 10.0         # VisionTurnSpeedControlOvershootMinDistance (m)
    self._anticipation_target_reduction = 0.95  # VisionTurnSpeedControlAnticipationTargetReduction
    # Time-to-brake trigger for planner-facing overshoot cap.
    # <= 0 means we should already be applying overshoot braking.
    self._overshoot_trigger_in_s = float('inf')
    self._overshoot_cap_active = False

    # Apex detection and tracking
    self._apex_indices = []  # Indices of detected apexes in trajectory
    self._last_apex_passed_time = 0.0  # For hysteresis
    self._distance_past_apex = 0.0  # Meters past most recent apex
    self._apex_exit_ready = False
    self._apex_trigger_idx = 0
    self._curve_sample_idx = 0
    # Detection
    self._apex_threshold = 5e-5          # VisionTurnSpeedControlApexThreshold
    self._apex_prominence = 1e-4         # VisionTurnSpeedControlApexProminence
    self._apex_hysteresis_time = 2.0     # VisionTurnSpeedControlApexHysteresisTime (s)
    self._apex_meters_per_index = 2.0    # VisionTurnSpeedControlApexMetersPerIndex (m)
    self._apex_near_index = 3            # VisionTurnSpeedControlApexNearIndex (indices)
    # Boost
    self._apex_boost_distance = 50.0     # VisionTurnSpeedControlApexBoostDistance (m)
    self._apex_boost_factor = 0.1        # VisionTurnSpeedControlApexBoostFactor (0..0.2)
    self._apex_boost_min_lat_accel = 1.0 # VisionTurnSpeedControlApexBoostMinLatAccel (m/s²)
    self._apex_boost_center = 2.0        # VisionTurnSpeedControlApexBoostCenter (m/s²)
    self._apex_boost_width = 0.5         # VisionTurnSpeedControlApexBoostWidth (m/s²)
    self._boost_safety_curvature_scale = 0.7 # VisionTurnSpeedControlBoostSafetyCurvatureScale

    # Comfort/adaptive deceleration limits (tunable)
    self._comfort_decel_limit = COMFORT_DECEL_LIMIT
    self._comfort_jerk_limit = COMFORT_JERK_LIMIT
    self._max_adaptive_decel = MAX_ADAPTIVE_DECEL
    self._max_adaptive_jerk = MAX_ADAPTIVE_JERK
    self._curvature_trajectory = []  # Store curvature array for apex detection

    self._reset()
    # Force an initial params refresh to ensure runtime knobs reflect latest Params
    try:
      update_vtsc_params(self, force=True)
    except Exception:
      # Safe to continue with defaults if Params not available at startup
      pass

    # Map lookahead cache/state
    self._map_curv_cache_raw = None
    self._map_curv_cache = []
    self._map_curv_last_ts = 0.0
    self._map_tail_active = False
    self._map_tail_last_cap = None
    self._map_tail_last_start = 0.0
    self._map_tail_last_coverage = 0.0
    # Debug-only map lookahead diagnostics (why map cap is inactive this frame)
    self._map_tail_reason = "init"
    self._map_tail_compute_reason = "init"

    # Rally co-pilot / HUD curve preview derived from VTSC's map lookahead inputs.
    # The HUD must not compute curves; it only renders these fields.
    self._curve_preview_valid = False
    self._curve_preview_distance_m = 0.0
    self._curve_preview_time_to_s = 0.0
    self._curve_preview_kappa_max = 0.0
    self._curve_preview_direction = 0  # VisionTurnSpeedControl.TurnDirection (unknown=0)
    self._curve_preview_severity = 0   # VisionTurnSpeedControl.CurveSeverity (unknown=0)
    self._curve_preview_points: list[tuple[float, float]] = []
    self._curve_preview_last_ts = 0.0
    self._curve_preview_last_cache_raw = None
    self._curve_preview_last_latlon: tuple[float, float] | None = None

    # Lead-aware occlusion bypass
    self._occl_bypass_with_lead = True
    self._occl_bypass_headway_s = 3.0
    self._lead_present = False
    self._lead_headway_s = 99.0
    self._occl_lead_bypass_active = False

    # Telemetry/debug controls
    self._dbg_enabled = False
    self._dbg_emit_interval_s = 0.5  # ~2 Hz
    self._dbg_next_emit_ts = 0.0
    self._dbg_refresh_ts = 0.0
    self._dbg_write_file = False
    # Snapshot fields
    self._dbg_k_model = 0.0
    self._dbg_k_steer = 0.0
    self._dbg_steer_fallback_active = False
    self._dbg_target_raw = 0.0
    self._dbg_target_final = 0.0
    self._dbg_occl_positive_margin = False
    self._dbg_early_no_raise = False
    self._dbg_tail_frac = 0.0
    self._dbg_s_tail = 0.0
    self._dbg_jerk_cmd = 0.0
    # FOV gating + units diagnostics
    self._psi_fov_rad = 0.49
    # Unify margin to ~5° across all code paths
    self._psi_margin_rad = 0.087  # ~5 deg
    self._fov_occluded = False
    self._fov_on_cnt = 0
    self._fov_off_cnt = 0
    self._fov_reason = ''
    self._fov_pretrigger_time_s = 1.5
    # Disable onset stickiness/overshoot windows (no-raise window effectively 0)
    self._fov_onset_boost_frames = 0
    self._fov_overshoot_frames = 0
    self._fov_boost_left = 0
    self._fov_overshoot_left = 0
    self._fov_ewma_tau_s = 0.4
    self._fov_kappa_ewma = 0.0
    self._fov_N_on = 2
    self._fov_N_off = 12
    self._dbg_psi_vis = 0.0
    self._dbg_psi_thresh = 0.0
    self._dbg_ttfov_s = 0.0
    self._dbg_units_ok = True
    self._dbg_gamma_eff = 0.0
    # Occlusion arbitration breadcrumbs (defaults)
    self._dbg_psi_est = 0.0
    self._dbg_consider_occl = False
    self._dbg_double_cap_guard = False
    # Onset tracking for occlusion window and early no-raise
    self._occlusion_prev = False
    self._occlusion_onset_timer_s = 0.0
    self._v_cap_active_at_onset_mps = 0.0
    self._onset_no_raise_active = False
    # Cap selection + freeway guard debug fields
    self._dbg_active_cap = ""
    self._dbg_cap_visible_vmin = 0.0
    self._dbg_cap_occl_vmin = 0.0
    self._dbg_cap_map_vmin = 0.0
    self._dbg_vtsc_cmd = 0.0
    self._dbg_kappa_vis = 0.0
    self._dbg_s_visible_m = 0.0
    self._dbg_path_conf = 0.0
    self._dbg_fail_open = False
    self._freeway_failopen_active = False

    # Vision-floor and dropout discrimination (aggressive bias)
    # 0=off, 1=TTL floor after last-known-good, 2=strict floor always
    self._vision_floor_ttl_s = 3.0
    self._vision_floor_mult = 1.00
    # Treat short model frame loss as transient dropout; suppress occlusion pretrigger briefly
    self._dropout_grace_s = 0.40
    self._dropout_until = 0.0

  # ===== Internal Param Helpers (decode/parse/clip) =====
  def _get_float_param(self, key: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Read a float param from Params with robust decoding and optional clipping.

    - Accepts bytes or string; falls back to default on any parse error.
    - If bounds provided, applies clip to [lo, hi].
    """
    try:
      raw = self._params.get(key)
      if raw is None:
        val = float(default)
      else:
        s = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else raw
        val = float(s)
    except Exception:
      # Be robust to UnknownKeyName or any decode/parse failure in off-road/harness contexts
      val = float(default)
    if lo is not None and hi is not None:
      return clip(val, lo, hi)
    return val

  def _get_bool_param(self, key: str, default: bool = False) -> bool:
    """Read a boolean param; returns default if underlying access fails."""
    try:
      return bool(self._params.get_bool(key))
    except Exception:
      return bool(default)

  def _should_emit_debug(self, now_s: float) -> bool:
    try:
      if now_s >= float(getattr(self, '_dbg_refresh_ts', 0.0)):
        self._dbg_enabled = bool(self._get_bool_param('VTSCVerboseDebug', False))
        # refresh file writer toggle alongside verbose debug
        self._dbg_write_file = bool(self._get_bool_param('VTSCWriteSnapshotFile', False))
        self._dbg_refresh_ts = now_s + 2.0
    except Exception:
      self._dbg_enabled = False
    if not self._dbg_enabled:
      return False
    nxt = float(getattr(self, '_dbg_next_emit_ts', 0.0))
    if now_s >= nxt:
      self._dbg_next_emit_ts = now_s + float(getattr(self, '_dbg_emit_interval_s', 0.5))
      return True
    return False

  def _vision_status_str(self) -> str:
    try:
      vs = getattr(self._occlusion_state, 'vision_status', None)
    except Exception:
      vs = None
    if vs == VisionStatus.FULL_VISIBILITY:
      return 'FULL'
    if vs == VisionStatus.PARTIAL_OCCLUSION:
      return 'PARTIAL'
    if vs == VisionStatus.SEVERE_OCCLUSION:
      return 'SEVERE'
    if vs == VisionStatus.VISION_LOST:
      return 'LOST'
    return 'UNKNOWN'

  def _append_snapshot_to_file(self, snap: dict, now_s: float) -> None:
    """Append a compact JSON snapshot line to a small rotating file on device."""
    try:
      base_dir = "/data/media/0/VTSCDebug"
      path = os.path.join(base_dir, "vtsc_snapshots.jsonl")
      os.makedirs(base_dir, exist_ok=True)
      # Attach timestamp to snapshot
      snap_out = dict(snap)
      snap_out['ts'] = float(now_s)
      line = json.dumps(snap_out, separators=(',', ':')) + "\n"
      # Rotate if file grows beyond ~512 KB (simple strategy)
      try:
        if os.path.exists(path) and os.path.getsize(path) > 512 * 1024:
          # Truncate by replacing with empty file; keep a single backup
          try:
            os.replace(path, path + ".1")
          except Exception:
            pass
      except Exception:
        pass
      with open(path, 'a', encoding='utf-8') as f:
        f.write(line)
    except Exception:
      # Never raise from telemetry path
      pass

  def snapshot_debug_state(self) -> dict:
    try:
      v_ego = float(getattr(self, '_v_ego', 0.0))
      v_cruise = float(getattr(self, '_v_cruise_setpoint', 0.0))
      lead = bool(getattr(self, '_lead_present', False))
      hw = float(getattr(self, '_lead_headway_s', 99.0))
      conf = float(getattr(self._occlusion_state, 'smoothed_confidence', 0.0))
      k_model = float(getattr(self, '_dbg_k_model', 0.0))
      k_steer = float(getattr(self, '_dbg_k_steer', 0.0))
      steer_fallback_active = bool(getattr(self, '_dbg_steer_fallback_active', False))
      k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      k_vis = float(getattr(self._occlusion_state, 'last_valid_curvature', 0.0))
      is_easing = bool(getattr(self, '_is_easing', False))
      abs_cr = float(getattr(self, '_abs_curvature_rate', 0.0))
      # Speeds
      v_phys_base = float(min(v_cruise, curvature_to_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
      v_occ = float(curvature_to_speed(max(1e-8, k_est)))
      v_vis = float(curvature_to_speed(max(1e-8, k_vis)))
      raw = float(getattr(self, '_dbg_target_raw', 0.0))
      final = float(getattr(self, '_dbg_target_final', raw))
      # Occlusion gating
      occl_margin = bool(getattr(self, '_dbg_occl_positive_margin', False))
      bypass = bool(getattr(self, '_occl_lead_bypass_active', False))
      low_speed_margin = bool(getattr(self, '_dbg_low_speed_margin', False))
      fov_occ = bool(getattr(self, '_fov_occluded', False))
      vis_h = float(getattr(self, '_vis_horizon_s', 1.4))
      tail_frac = float(getattr(self, '_dbg_tail_frac', 0.0))
      s_tail = float(getattr(self, '_dbg_s_tail', 0.0))
      enr = bool(getattr(self, '_dbg_early_no_raise', False))
      # Lookahead
      map_active = bool(getattr(self, '_map_tail_active', False))
      map_cap = float(getattr(self, '_map_tail_last_cap', 0.0) or 0.0)
      map_start = float(getattr(self, '_map_tail_last_start', 0.0) or 0.0)
      map_cov = float(getattr(self, '_map_tail_last_coverage', 0.0) or 0.0)
      map_reason = str(getattr(self, '_map_tail_reason', '') or '')
      map_compute_reason = str(getattr(self, '_map_tail_compute_reason', '') or '')
      # Limits and commands
      comfort = float(getattr(self, '_comfort_decel_limit', -1.47))
      max_adapt = float(getattr(self, '_max_adaptive_decel', -6.0))
      decel_cmd = float(getattr(self, '_current_decel', 0.0))
      jerk_cmd = float(getattr(self, '_dbg_jerk_cmd', 0.0))
      a_cmd = float(getattr(self, '_a_target', 0.0))
      # Newly added diagnostics populated in _update_solution
      cap_vis = float(getattr(self, '_dbg_cap_visible_vmin', 0.0))
      cap_occ = float(getattr(self, '_dbg_cap_occl_vmin', 0.0))
      cap_map = float(getattr(self, '_dbg_cap_map_vmin', 0.0))
      active_cap = str(getattr(self, '_dbg_active_cap', '') or '')
      vtsc_cmd = float(getattr(self, '_dbg_vtsc_cmd', 0.0) or 0.0)
      s_vis_m = float(getattr(self, '_dbg_s_visible_m', 0.0))
      fail_open = bool(getattr(self, '_dbg_fail_open', False))
      # FOV/units helpers (may be unset on older builds; default sensibly)
      psi_fov = float(getattr(self, '_psi_fov_rad', 0.49))
      psi_margin = float(getattr(self, '_psi_margin_rad', 0.087))
      ttfov = float(getattr(self, '_dbg_ttfov_s', 0.0))
      psi_vis = float(abs(k_vis) * max(0.0, s_vis_m))
      psi_thresh = float(max(0.0, psi_fov - psi_margin))
      occl_reason = str(getattr(self, '_fov_reason', '') or '')
      occl_on = int(getattr(self, '_fov_on_cnt', 0))
      occl_off = int(getattr(self, '_fov_off_cnt', 0))
      gamma_eff = float(getattr(self, '_dbg_gamma_eff', 0.0))
      units_ok = bool(getattr(self, '_dbg_units_ok', True))
      boost_left = int(getattr(self, '_fov_boost_left', 0))
      overshoot_left = int(getattr(self, '_fov_overshoot_left', 0))
      cap_hold_active = bool(getattr(self, '_dbg_vturn_hold_active', False))
      cap_hold_min = float(getattr(self, '_dbg_vturn_hold_min', 0.0) or 0.0)
      return {
        'v': v_ego, 'cruise': v_cruise, 'lead': lead, 'hw': hw,
        'conf': conf, 'vision_status': self._vision_status_str(),
        'k_model': k_model, 'k_steer': k_steer, 'steer_fallback_active': steer_fallback_active,
        'k_occ': k_est, 'k_vis_last': k_vis,
        'is_easing': is_easing, 'abs_curv_rate': abs_cr,
        'v_base': v_phys_base, 'v_occ': v_occ, 'v_vis': v_vis,
        'raw': raw, 'final': final,
        'occl_positive_margin': occl_margin, 'occl_lead_bypass_active': bypass,
        'low_speed_margin_override': low_speed_margin,
        'fov_occluded': fov_occ,
        'vis_horizon_s': vis_h, 'tail_frac': tail_frac, 's_tail': s_tail, 'early_no_raise': enr,
        'map_tail_active': map_active, 'map_tail_cap': map_cap, 'map_tail_start_m': map_start, 'map_tail_coverage': map_cov,
        'map_tail_reason': map_reason, 'map_tail_compute_reason': map_compute_reason,
        'comfort_decel': comfort, 'max_adaptive_decel': max_adapt, 'decel_cmd': decel_cmd, 'jerk_cmd': jerk_cmd, 'a_cmd': a_cmd,
        # New fields for quick triage on road
        'active_cap': active_cap, 'vtsc_cmd': vtsc_cmd,
        'cap_source': str(getattr(self, '_dbg_cap_source', '')),
        'cap_visible_vmin': cap_vis, 'cap_occl_vmin': cap_occ, 'cap_map_vmin': cap_map,
        's_visible_m': s_vis_m, 'kappa_vis': k_vis, 'path_conf': conf,
        'occluded': bool(not getattr(self._occlusion_state, 'vision_good', True)),
        'fail_open': fail_open,
        'psi_vis': psi_vis, 'psi_thresh': psi_thresh, 'ttfov_s': ttfov, 'psi_fov_rad': psi_fov, 'psi_margin_rad': psi_margin,
        'occlusion_reason': occl_reason, 'occl_on_cnt': occl_on, 'occl_off_cnt': occl_off, 'onset_boost_left': boost_left, 'overshoot_left': overshoot_left,
        'cap_hold_active': cap_hold_active, 'cap_hold_min': cap_hold_min,
        # κ-bias diagnostics
        'onset_bias_active': bool(getattr(self, '_dbg_onset_bias_active', False)),
        'onset_gate_reason': getattr(self, '_dbg_onset_gate_reason', None),
        'gamma_eff': gamma_eff, 'units_ok': units_ok,
        # Occlusion arbitration breadcrumbs
        'psi_gate_est': float(getattr(self, '_dbg_psi_est', 0.0)),
        'psi_gate_thresh': float(getattr(self, '_psi_thresh_rad', PSI_THRESH_RAD)),
        'consider_occl_gate': bool(getattr(self, '_dbg_consider_occl', False)),
        'double_cap_guard': bool(getattr(self, '_dbg_double_cap_guard', False)),
        'pre_cap_target': float(getattr(self, '_pre_cap_target_speed', 0.0)),
        # Phase-offset diagnostics
        'curve_sample_idx': int(getattr(self, '_curve_sample_idx', 0)),
        'overshoot_trigger_in_s': float(getattr(self, '_overshoot_trigger_in_s', 0.0)),
        'overshoot_cap_active': bool(getattr(self, '_overshoot_cap_active', False)),
        'apex_trigger_idx': int(getattr(self, '_apex_trigger_idx', 0)),
        'apex_exit_ready': bool(getattr(self, '_apex_exit_ready', False)),
        # Duplicated with _dbg_* names for watcher compatibility
        '_dbg_psi_est': float(getattr(self, '_dbg_psi_est', 0.0)),
        '_dbg_psi_thresh': float(getattr(self, '_psi_thresh_rad', PSI_THRESH_RAD)),
        '_dbg_consider_occl': bool(getattr(self, '_dbg_consider_occl', False)),
        '_dbg_double_cap_guard': bool(getattr(self, '_dbg_double_cap_guard', False)),
        '_dbg_pre_cap_target': float(getattr(self, '_pre_cap_target_speed', 0.0)),
      }
    except Exception:
      return {}

  # Pure FOV-based occlusion gating helper
  @staticmethod
  def occlusion_gate(kappa_vis: float, s_visible_m: float, path_conf: float,
                     psi_fov_rad: float, psi_margin_rad: float,
                     k_freeway: float = FREEWAY_CURV_EPS,
                     k_min: float = 2e-4, s_long: float = 120.0,
                     state: dict | None = None) -> tuple[bool, dict, str, dict]:
    state = dict(state or {})
    on_cnt = int(state.get('on_cnt', 0))
    off_cnt = int(state.get('off_cnt', 0))
    occluded = bool(state.get('occluded', False))
    psi_vis = abs(float(kappa_vis)) * max(0.0, float(s_visible_m))
    psi_thresh = max(0.0, float(psi_fov_rad) - float(psi_margin_rad))
    onset = (abs(kappa_vis) >= k_min) and (psi_vis >= psi_thresh)
    clear = (abs(kappa_vis) < k_freeway) or ((s_visible_m >= s_long) and (psi_vis < psi_thresh) and (path_conf >= 0.6))
    N_on, N_off = 5, 10
    reason = 'none'
    if onset:
      on_cnt += 1
      off_cnt = 0
      if on_cnt >= N_on:
        occluded = True
        reason = 'fov_exit'
    elif clear:
      off_cnt += 1
      on_cnt = 0
      if off_cnt >= N_off:
        occluded = False
        reason = 'freeway' if abs(kappa_vis) < k_freeway else 'short_vis'
    else:
      on_cnt = max(0, on_cnt - 1)
      off_cnt = max(0, off_cnt - 1)
    return occluded, {'on_cnt': on_cnt, 'off_cnt': off_cnt, 'occluded': occluded}, reason, {'psi_vis': psi_vis, 'psi_thresh': psi_thresh}

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'TVC: TurnVisionController state: {_description_for_state(value)}')
    self._state = value

  @property
  def adaptive_decel_active(self):
    """Adaptive deceleration system status for external monitoring."""
    return abs(self._current_decel) > abs(self._comfort_decel_limit) * 1.1

  @property
  def decel_requirement(self):
    """Current deceleration requirement for external monitoring."""
    return self._filtered_decel_requirement

  @property
  def a_target(self):
    return self._a_target if self.is_active else self._a_ego

  @property
  def v_turn(self):
    # VTSC output for longitudinal planner ingestion.
    #
    # IMPORTANT:
    # - The longitudinal planner consumes `v_turn` as a *speed cap* (min-of-sources).
    # - `v_turn` must therefore represent VTSC's latest computed recommendation, not a
    #   speed trajectory integrated from VTSC's internal accel state (which can create
    #   a feedback loop where the cap sticks near `v_ego` and prevents recovery).
    #
    # `_v_turn_output` is set each update() in `_update_solution()`.
    try:
      v_out = float(getattr(self, '_v_turn_output', 0.0) or 0.0)
    except Exception:
      v_out = 0.0
    if v_out > 0.0:
      return v_out
    # Fallback for very early init / offline contexts.
    return float(getattr(self, '_v_cruise_setpoint', 0.0) or 0.0)

  @property
  def current_lat_acc(self):
    return self._current_lat_acc

  @property
  def max_pred_lat_acc(self):
    return self._max_pred_lat_acc

  @property
  def is_active(self):
    # SIMPLIFIED: Always active when system is enabled - let longitudinal planner's min() decide usage
    return self._op_enabled and self._is_enabled and not self._gas_pressed

  @property
  def is_entering(self):
    return self._state == VisionTurnControllerState.entering

  @property
  def is_turning(self):
    return self._state == VisionTurnControllerState.turning

  @property
  def is_leaving(self):
    return self._state == VisionTurnControllerState.leaving

  @property
  def distance(self):
    """Distance to lateral acceleration overshoot point."""
    return self._v_overshoot_distance if hasattr(self, '_v_overshoot_distance') else 200.0

  # ===== Rally co-pilot / HUD curve preview (map-enriched) =====
  @property
  def curve_preview_valid(self) -> bool:
    return bool(getattr(self, '_curve_preview_valid', False))

  @property
  def curve_preview_distance_m(self) -> float:
    return float(getattr(self, '_curve_preview_distance_m', 0.0) or 0.0)

  @property
  def curve_preview_time_to_s(self) -> float:
    return float(getattr(self, '_curve_preview_time_to_s', 0.0) or 0.0)

  @property
  def curve_preview_kappa_max(self) -> float:
    return float(getattr(self, '_curve_preview_kappa_max', 0.0) or 0.0)

  @property
  def curve_preview_direction(self) -> int:
    # capnp enum value for VisionTurnSpeedControl.TurnDirection
    return int(getattr(self, '_curve_preview_direction', 0) or 0)

  @property
  def curve_preview_severity(self) -> int:
    # capnp enum value for VisionTurnSpeedControl.CurveSeverity
    return int(getattr(self, '_curve_preview_severity', 0) or 0)

  @property
  def curve_preview_points(self) -> list[tuple[float, float]]:
    pts = getattr(self, '_curve_preview_points', None)
    return list(pts) if isinstance(pts, list) else []

  def getCurrentLateralAccel(self):
    """Return current lateral acceleration for HUD display."""
    return self._current_lat_acc

  def _reset(self):
    self._current_lat_acc = 0.
    self._max_v_for_current_curvature = 0.
    self._max_pred_lat_acc = 0.
    self._v_overshoot_distance = 200.
    self._lat_acc_overshoot_ahead = False
    self._overshoot_trigger_in_s = float('inf')
    self._overshoot_cap_active = False

    # Reset adaptive deceleration system
    self._current_decel = 0.0
    self._filtered_decel_requirement = 0.0
    self._decel_hysteresis_state = False

    # Reset vision occlusion state
    self._occlusion_state = VisionOcclusionState()

    # Reset apex tracking
    self._apex_indices = []
    self._distance_past_apex = 0.0
    self._apex_exit_ready = False
    self._apex_trigger_idx = 0
    self._curve_sample_idx = 0
    self._curvature_trajectory = []

    # Reset advanced controller state (preserve current_accel to avoid jerk spikes)
    # Do not zero _current_accel here; preserve continuity across state transitions
    self._prev_target_speed = self._v_ego if hasattr(self, '_v_ego') else 0.0
    self._filtered_curvature = 0.0
    # Track cruise setpoint changes (e.g., speed-limit steps)
    self._prev_v_cruise_setpoint = getattr(self, '_prev_v_cruise_setpoint', 0.0)
    self._limit_step_until = 0.0
    self._suppress_raise_due_to_limit = False

    # Reset anticipatory deceleration state
    self._is_decelerating_for_curve = False

  def _apply_freeway_v_turn_hold(self, v_cap: float) -> float:
    """Hold material VTSC cap reductions briefly to bridge model flicker.

    Motivation: The model curvature horizon can flicker, producing short (<0.5s) cap dips. The
    longitudinal planner/MPC often cannot react within that window, so braking begins late and the
    driver intervenes. Holding the lowest cap briefly makes the cap persistent enough to be acted
    upon, while keeping release responsive.
    """
    try:
      now = float(time.time())
    except Exception:
      now = 0.0

    # Default: no hold
    self._dbg_vturn_hold_active = False

    try:
      v_ego = float(self._v_ego)
    except Exception:
      v_ego = 0.0

    # Gate hold behavior:
    # - At freeway speeds with good vision, hold helps when the horizon "pulses" a curve.
    # - Under degraded vision (occlusion), hold helps at any speed when a real cap appears briefly.
    try:
      vision_good = bool(getattr(self._occlusion_state, 'vision_good', True))
    except Exception:
      vision_good = True
    try:
      failopen = bool(getattr(self, '_freeway_failopen_active', False))
    except Exception:
      failopen = False
    fov_occluded = bool(getattr(self, '_fov_occluded', False))
    try:
      turn_evidence = float(getattr(self, '_max_pred_lat_acc', 0.0)) >= float(_ENTERING_PRED_LAT_ACC_TH)
    except Exception:
      turn_evidence = False

    freeway_clean = (v_ego >= float(VTURN_HOLD_MIN_V_MPS)) and vision_good and (not fov_occluded)
    # Under degraded vision, allow triggering a hold at any speed when there is clear turn evidence.
    # Do not trigger this while the FOV-occlusion latch is active: that subsystem already enforces
    # monotonic caps and can legitimately hold a cap after a curve until geometry clears.
    occluded_trigger_ok = (not vision_good) and (not failopen) and (not fov_occluded) and turn_evidence

    try:
      v_cruise = float(self._v_cruise_setpoint)
    except Exception:
      v_cruise = float(v_cap)

    # Update/extend hold window only when VTSC is asking for a meaningful reduction.
    overspeed = (v_ego - float(v_cap)) >= 0.5
    if (freeway_clean or occluded_trigger_ok) and ((v_cruise - float(v_cap)) >= float(VTURN_HOLD_DELTA_MPS)) and (overspeed or freeway_clean):
      hold_until = float(getattr(self, '_v_turn_hold_until', 0.0) or 0.0)
      hold_min = float(getattr(self, '_v_turn_hold_min', float(v_cap)) or float(v_cap))
      if now >= hold_until:
        hold_min = float(v_cap)
      else:
        hold_min = min(hold_min, float(v_cap))
      self._v_turn_hold_min = hold_min
      hold_s = float(VTURN_HOLD_S_OCCLUDED) if occluded_trigger_ok else float(VTURN_HOLD_S)
      self._v_turn_hold_until = now + hold_s

    # Apply hold if active
    hold_until2 = float(getattr(self, '_v_turn_hold_until', 0.0) or 0.0)
    hold_min2 = float(getattr(self, '_v_turn_hold_min', float(v_cap)) or float(v_cap))
    if now < hold_until2:
      self._dbg_vturn_hold_active = True
      self._dbg_vturn_hold_min = float(hold_min2)
      return float(min(float(v_cap), hold_min2))

    # Hold expired: allow immediate release
    self._v_turn_hold_min = float(INF_SPEED)
    return float(v_cap)

  def _update_params(self):
    # Delegate to shared reader to avoid duplicating logic here
    update_vtsc_params(self)

  def _calculate_required_deceleration(self, v_current: float, v_target: float, distance: float) -> float:
    """Calculate minimum deceleration required using physics: a = (v_f² - v_i²) / (2d)"""
    if distance <= 0.1:  # Avoid division by zero
      return self._max_adaptive_decel
    
    # Physics formula: a = (v_target² - v_current²) / (2 × distance)
    required_decel = (v_target * v_target - v_current * v_current) / (2.0 * distance)
    
    # Apply safety bias - slightly more aggressive to ensure we reach target
    safety_biased_decel = required_decel * (1.0 + self._safety_bias)
    
    # Clamp to system limits
    return max(safety_biased_decel, self._max_adaptive_decel)

  def _get_optimal_deceleration(self, raw_decel: float, dt: float) -> float:
    """Get optimal deceleration using adaptive physics-based approach with noise filtering."""
    
    # Step 1: Apply EMA filtering to smooth deceleration requirements
    # Initialize filter to first value if not yet initialized (was 0)
    if self._filtered_decel_requirement == 0.0 and raw_decel != 0.0:
        self._filtered_decel_requirement = raw_decel
    else:
        self._filtered_decel_requirement = ((1.0 - self._filter_alpha) * self._filtered_decel_requirement +
                                           self._filter_alpha * raw_decel)
    
    # Step 2: Determine if we should use comfort or adaptive deceleration
    comfort_sufficient = (abs(self._filtered_decel_requirement) <= abs(self._comfort_decel_limit))
    
    # Step 3: Apply hysteresis to prevent oscillation between comfort/adaptive modes
    if comfort_sufficient and not self._decel_hysteresis_state:
        # Comfort deceleration is sufficient and we're not in adaptive mode
        target_decel = max(self._filtered_decel_requirement, self._comfort_decel_limit)
        target_jerk_limit = abs(self._comfort_jerk_limit)
    elif not comfort_sufficient and not self._decel_hysteresis_state:
        # Need to switch to adaptive mode
        self._decel_hysteresis_state = True
        # Use physics-based calculation with system limits
        target_decel = max(self._filtered_decel_requirement, self._max_adaptive_decel)
        target_jerk_limit = abs(self._max_adaptive_jerk)
    elif self._decel_hysteresis_state:
        # Currently in adaptive mode - check if we can return to comfort with hysteresis
        hysteresis_threshold = abs(self._comfort_decel_limit) * (1.0 - self._hysteresis_threshold)
        if abs(self._filtered_decel_requirement) <= hysteresis_threshold:
            self._decel_hysteresis_state = False
            target_decel = max(self._filtered_decel_requirement, self._comfort_decel_limit)
            target_jerk_limit = abs(self._comfort_jerk_limit)
        else:
            # Stay in adaptive mode
            target_decel = max(self._filtered_decel_requirement, self._max_adaptive_decel)
            target_jerk_limit = abs(self._max_adaptive_jerk)
    else:
        # Default case
        target_decel = max(self._filtered_decel_requirement, self._comfort_decel_limit)
        target_jerk_limit = abs(self._comfort_jerk_limit)
    
    # Step 4: Apply jerk limiting for smooth transitions
    max_decel_change = abs(target_jerk_limit) * dt  # Ensure positive limit
    decel_change = target_decel - self._current_decel
    
    if abs(decel_change) > max_decel_change:
        if decel_change > 0:
            self._current_decel += max_decel_change
        else:
            self._current_decel -= max_decel_change
    else:
        self._current_decel = target_decel
    
    return self._current_decel

  def _update_vision_occlusion(self, model_data, current_time: float):
    """Update vision occlusion state and return curvature to use under monotonic occlusion mode."""
    # Dropout detection: if the model publishes too few orientationRate frames, consider it a transient
    # and open a short grace window during which we avoid pretrigger occlusion.
    try:
      z = getattr(getattr(model_data, 'orientationRate', None), 'z', None)
      n_frames = int(len(z)) if z is not None else 0
      if n_frames and n_frames < 33:
        self._dropout_until = float(time.time()) + float(getattr(self, '_dropout_grace_s', 0.40))
    except Exception:
      pass
    try:
      setattr(self._occlusion_state, 'dropout_active', bool(time.time() < float(getattr(self, '_dropout_until', 0.0))))
    except Exception:
      pass
    # Estimate vision confidence
    if model_data is None:
        vision_confidence = 0.0
    else:
        if hasattr(model_data, 'laneLineProbs') and model_data.laneLineProbs:
            vision_confidence = float(np.mean(model_data.laneLineProbs))
        else:
            vision_confidence = 1.0

    # Candidate current curvature is the filtered curvature we track
    current_curvature = self._filtered_curvature

    # Remember vision_good before update to detect reacquisition
    prev_good = self._occlusion_state.vision_good

    # Update occlusion state with vehicle speed and time
    self._occlusion_state.update(current_curvature, vision_confidence, self._v_ego, current_time)
    # While vision is good, keep last_valid_curvature fresh at controller level (compatibility with acceptance tests)
    if self._occlusion_state.vision_good:
      self._occlusion_state.last_valid_curvature = current_curvature

    # On reacquisition, arm a fast filtering window to improve recovery time
    if (not prev_good) and self._occlusion_state.vision_good:
      # Fast window; effective alpha increased later when applied
      now_t = time.time()
      self._fast_reacq_until = now_t + float(getattr(self, '_fast_reacq_window_s', 0.9))
      # Record precise reacquisition moment for accel floor logic
      try:
        self._occlusion_state.reacquired_at = float(now_t)
      except Exception:
        pass
      # Immediately clear any FOV-gated occlusion artifacts to let fresh vision take over
      try:
        self._fov_occluded = False
        self._fov_on_cnt = 0
        self._fov_off_cnt = 0
        self._fov_boost_left = 0
        self._fov_overshoot_left = 0
        self._onset_no_raise_active = False
      except Exception:
        pass

    # If vision is good, use current curvature; otherwise, use estimated curvature under monotonic model
    return current_curvature if self._occlusion_state.vision_good else self._occlusion_state.extrapolated_curvature

  def _monitor_adaptive_deceleration(self, required_decel: float, remaining_distance: float) -> bool:
    """Monitor adaptive deceleration system performance and detect extreme scenarios."""
    # Check if we're using maximum system deceleration
    is_max_decel = abs(self._current_decel) >= abs(self._max_adaptive_decel) * 0.95
    
    # Check if distance is critically short
    is_critical_distance = remaining_distance < 20.0  # meters
    
    # Log adaptive deceleration activation for debugging
    if self._decel_hysteresis_state:
        _debug(f'VTSC: Adaptive decel active - current: {self._current_decel:.2f}, required: {required_decel:.2f}, distance: {remaining_distance:.1f}m')
    
    # Return true if we're in a challenging scenario (for external monitoring)
    return is_max_decel and is_critical_distance

  def _update_calculations(self, sm):
    """Advanced vision-based curvature calculation using direct model outputs."""
    # Be tolerant of lightweight SM stubs in offline tests
    try:
      model_data = sm['modelV2'] if getattr(sm, 'valid', {}).get('modelV2', False) else None
    except Exception:
      model_data = getattr(sm, 'modelV2', None)
      if model_data is None:
        data = getattr(sm, '_data', None)
        if isinstance(data, dict):
          model_data = data.get('modelV2', None)
    # Lead presence/headway estimation from radarState (if available)
    try:
      try:
        rs = sm['radarState'] if getattr(sm, 'valid', {}).get('radarState', False) else None
      except Exception:
        rs = getattr(sm, 'radarState', None)
        if rs is None:
          data = getattr(sm, '_data', None)
          if isinstance(data, dict):
            rs = data.get('radarState', None)
      lead = getattr(rs, 'leadOne', None) if rs is not None else None
      status = bool(getattr(lead, 'status', False)) if lead is not None else False
      d_rel = float(getattr(lead, 'dRel', 1e9)) if lead is not None else 1e9
      v_ego_safe = max(0.1, float(self._v_ego))
      v_floor = max(float(OCCL_BYPASS_HEADWAY_V_FLOOR_MPS), float(_MIN_V))
      v_for_headway = max(v_ego_safe, v_floor)
      headway_s = float(d_rel) / v_for_headway
      self._lead_present = status
      self._lead_d_rel_m = float(d_rel)
      self._lead_headway_s = headway_s
    except Exception:
      self._lead_present = False
      self._lead_headway_s = 99.0
    current_time = time.time()

    # Initialize defaults for edge cases (use the last filtered curvature if present).
    # NOTE: vision occlusion state is updated *after* we update `_filtered_curvature` for this frame.
    # This avoids a 1-frame lag where `last_valid_curvature` can be stuck at ~0 when entering a curve,
    # which in turn breaks FOV-occlusion gating and fail-open logic.
    current_curvature_signed = 0.0
    current_curvature = float(getattr(self, '_filtered_curvature', 0.0))
    max_pred_curvature = current_curvature
    # Recomputed per-frame; used to gate planner-facing overshoot cap timing.
    self._overshoot_trigger_in_s = float('inf')
    self._curve_sample_idx = 0

    # Lead-aware occlusion bypass activation
    try:
      # Lead-bypass is intended to prevent the occlusion subsystem from interfering while following
      # a lead at close headway (e.g., lane-line confidence drops behind a lead should not "stick"
      # VTSC in occlusion/no-raise behavior on otherwise benign segments).
      low_speed_lead_close = bool(
        self._lead_present and (v_ego_safe <= float(OCCL_BYPASS_LOW_SPEED_V_MPS)) and (float(self._lead_d_rel_m) <= float(OCCL_BYPASS_LEAD_D_REL_MAX_M))
      )
      self._occl_lead_bypass_active = bool(
        self._occl_bypass_with_lead and self._lead_present and (
          (self._lead_headway_s <= float(self._occl_bypass_headway_s)) or low_speed_lead_close
        )
      )
    except Exception:
      self._occl_lead_bypass_active = False

    # Use advanced method: direct model data access whenever model is available
    if (model_data is not None and
        hasattr(model_data, 'orientationRate') and hasattr(model_data, 'velocity') and
        model_data.orientationRate.z is not None and model_data.velocity.x is not None):

      orientation_rate_raw = model_data.orientationRate.z
      velocity_pred_raw = model_data.velocity.x

      MIN_POINTS = 3
      if (len(orientation_rate_raw) >= MIN_POINTS and len(velocity_pred_raw) >= MIN_POINTS):
        # Use direct model outputs for curvature calculation
        n_points = int(min(len(orientation_rate_raw), len(velocity_pred_raw), N_POINTS))
        # Ensure n_points is a pure Python int for Cap'n Proto compatibility
        n_points = int(n_points)
        # Advance model-time interpretation to remove systematic VTSC lag.
        # User offset convention: lower values start earlier, higher values start later.
        times_nominal = np.array(ModelConstants.T_IDXS[:n_points], dtype=float)
        base_phase_advance_s = float(VTSC_TRAJECTORY_PHASE_ADVANCE_S)
        curve_phase_offset_s = float(getattr(self, '_curve_phase_offset_s', 0.0))
        phase_advance_s = float(clip(base_phase_advance_s - curve_phase_offset_s, 0.0, float(times_nominal[-1])))
        lead_idx = int(np.searchsorted(times_nominal, phase_advance_s, side='left'))
        lead_idx = int(min(max(lead_idx, 0), n_points - 1))
        self._curve_sample_idx = int(lead_idx)
        times_for_planning = np.maximum(0.0, times_nominal - phase_advance_s)
        # FIXED: Preserve sign information - don't use np.abs() here!
        orientation_rate_signed = np.array(list(orientation_rate_raw)[:n_points], dtype=float)
        velocity_pred = np.array(list(velocity_pred_raw)[:n_points], dtype=float)

        # Compute curvature array with SIGNED values.
        # Model orientationRate.z is yaw rate (rad/s). Curvature κ = yaw_rate / speed (1/m).
        # Use predicted velocity to convert; clamp very low speeds to avoid blow-ups.
        v_clip = np.clip(velocity_pred, 0.1, None)
        curvature_array_signed = orientation_rate_signed / v_clip
        # For max calculations, use absolute values
        curvature_array_abs = np.abs(curvature_array_signed)
        max_pred_curvature = float(np.max(curvature_array_abs))
        # expose for debug snapshot
        self._dbg_k_model = max_pred_curvature

        # Store curvature trajectory and detect apexes (use absolute values for apex detection)
        self._curvature_trajectory = curvature_array_abs.tolist()
        raw_apex_indices = find_apexes_enhanced(curvature_array_abs, self._apex_threshold, self._apex_prominence)
        if lead_idx > 0 and raw_apex_indices:
          self._apex_indices = sorted({max(0, int(i) - lead_idx) for i in raw_apex_indices})
        else:
          self._apex_indices = raw_apex_indices
        _debug(f'TVC: Found {len(self._apex_indices)} apexes at indices: {self._apex_indices}')

        # Calculate lateral acceleration using model-predicted curvature
        # This is more accurate than steering angle at highway speeds
        # Use the current model-predicted curvature WITH SIGN preserved
        if len(curvature_array_signed) > 0:
          sample_idx = int(min(max(self._curve_sample_idx, 0), len(curvature_array_signed) - 1))
          current_curvature = float(curvature_array_abs[sample_idx])  # Absolute value for calculations
          current_curvature_signed = float(curvature_array_signed[sample_idx])  # Signed value for lateral accel

        # Steering-curvature fallback when the model says "straight" but steering indicates a
        # real curve. This avoids "fail open" behavior on sharp bends when the model curvature
        # momentarily flattens.
        self._dbg_k_steer = 0.0
        self._dbg_steer_fallback_active = False
        try:
          llp = getattr(model_data, 'laneLineProbs', None)
          vision_confidence = float(np.mean(llp)) if llp else 1.0
        except Exception:
          vision_confidence = 1.0
        try:
          if (float(max_pred_curvature) <= STEER_CURVATURE_FALLBACK_MODEL_KAPPA_MAX and
              self._vm is not None and float(self._v_ego) >= STEER_CURVATURE_FALLBACK_MIN_V_MPS):
            sa_rad = math.radians(float(getattr(self, '_steering_angle_deg', 0.0)))
            kappa_steer = float(self._vm.calc_curvature(sa_rad, float(self._v_ego), 0.0))
            kappa_steer_abs = abs(kappa_steer)
            self._dbg_k_steer = float(kappa_steer_abs)
            if kappa_steer_abs >= STEER_CURVATURE_FALLBACK_MIN_KAPPA:
              current_curvature = max(float(current_curvature), float(kappa_steer_abs))
              current_curvature_signed = float(kappa_steer)
              self._dbg_steer_fallback_active = True
        except Exception:
          pass

        # Update filtered curvature using EMA of the NEAR-TERM curvature, not the horizon max.
        # Using the max across the horizon makes the "visible" path act like an occlusion cap
        # and causes premature, persistent overslow. Filter toward the instantaneous curvature instead.
        self._filtered_curvature = ((1 - self._curvature_ema_ratio) * self._filtered_curvature +
                                   self._curvature_ema_ratio * current_curvature)

        # Calculate lateral accelerations using model predictions (not steering angle)
        self._current_lat_acc = current_curvature_signed * self._v_ego**2
        self._max_pred_lat_acc = self._v_ego**2 * max_pred_curvature

        # Calculate safe speed using advanced physics-based method
        self._max_v_for_current_curvature = curvature_to_speed(current_curvature) if current_curvature > 0 else V_CRUISE_MAX * CV.KPH_TO_MS

        # Check for overshoot using curvature_to_speed method (use absolute values)
        safe_speeds = np.array([curvature_to_speed(curv) for curv in curvature_array_abs])
        # Under very low lane-line confidence, be mildly conservative when deciding whether we need
        # to start slowing for a curve ahead. This helps blind off-ramps where the model curvature
        # estimate can rise sharply only very late in the approach.
        try:
          conf_for_scale = float(vision_confidence)
        except Exception:
          conf_for_scale = 1.0
        try:
          lead_bypass = bool(self._occl_lead_bypass_active)
        except Exception:
          lead_bypass = False
        if (not lead_bypass) and (conf_for_scale < CONFIDENCE_EXIT_TO_PARTIAL):
          try:
            denom = float(max(1e-6, CONFIDENCE_EXIT_TO_PARTIAL - CONFIDENCE_ENTER_SEVERE))
            t = float(clip((conf_for_scale - CONFIDENCE_ENTER_SEVERE) / denom, 0.0, 1.0))
          except Exception:
            t = 0.0
          scale = float(SEVERE_OVERSHOOT_SPEED_SCALE_MIN + (1.0 - SEVERE_OVERSHOOT_SPEED_SCALE_MIN) * t)
          safe_speeds = safe_speeds * scale
        overshoot_mask = safe_speeds < self._v_ego
        self._lat_acc_overshoot_ahead = np.any(overshoot_mask)

        # Update occlusion state using the curvature we've just computed/filtered for this frame.
        # This keeps `last_valid_curvature` and confidence gating aligned with the *current* frame's
        # `_filtered_curvature`, avoiding a 1-frame lag at curve entry.
        _ = self._update_vision_occlusion(model_data, current_time)

        if self._lat_acc_overshoot_ahead:
          # PROPER FIX: Consider ALL points requiring deceleration, not just first or tightest
          # Calculate which points need immediate action based on deceleration requirements
          overshoot_indices = np.where(overshoot_mask)[0]

          # For each point that needs slowing, calculate if we need to start NOW
          max_decel = max(0.1, float(self._planning_decel_limit))  # m/s² planning decel limit
          immediate_requirements = []

          for idx in overshoot_indices:
            # How much distance do we need to slow down to this point's safe speed?
            speed_diff_sq = safe_speeds[idx]**2 - self._v_ego**2
            decel_distance_needed = abs(speed_diff_sq) / max(2e-3, (2 * max_decel))

            # How far away is this point?
            point_distance = times_for_planning[idx] * self._v_ego

            # Do we need to start slowing NOW for this point?
            if point_distance <= decel_distance_needed * max(1.0, float(self._overshoot_safety_margin)):
              immediate_requirements.append((idx, safe_speeds[idx], point_distance))

          if immediate_requirements:
            # Among all points needing immediate action, target the MINIMUM safe speed
            # This ensures we plan for the tightest part of the curve
            min_required_speed = min([speed for _, speed, _ in immediate_requirements])
            # Find the index with that minimum speed
            for idx, speed, dist in immediate_requirements:
              if speed == min_required_speed:
                overshoot_idx = idx
                self._v_overshoot_distance = dist
                break
          else:
            # FIX: No immediate requirements, but still plan for the TIGHTEST point ahead
            # Don't just use the first overshoot - find the point with minimum safe speed
            tightest_idx = overshoot_indices[np.argmin(safe_speeds[overshoot_indices])]
            overshoot_idx = tightest_idx
            self._v_overshoot_distance = times_for_planning[overshoot_idx] * self._v_ego

          self._v_overshoot = min(safe_speeds[overshoot_idx], self._v_cruise_setpoint)
          # Distance already set above based on immediate requirements or tightest point
          # Ensure minimum distance for safety
          self._v_overshoot_distance = max(self._v_overshoot_distance, float(self._overshoot_min_distance))
          # Calculate anticipation time for early deceleration
          anticipation_time = calculate_anticipation_time(
              self._v_ego,
              self._v_overshoot,
              max_pred_curvature * self._v_ego**2,
              self._aggressiveness
          )

          # Optional override: use fixed lead time in seconds if configured (> 0)
          if getattr(self, '_fixed_lead_time_s', 0.0) > 0.0:
            anticipation_time = clip(self._fixed_lead_time_s, 0.1, 10.0)

          # Adjust the overshoot distance to start deceleration earlier
          # This makes us reach target speed BEFORE the apex
          anticipation_distance = anticipation_time * self._v_ego
          raw_trigger_distance = float(self._v_overshoot_distance - anticipation_distance)
          # Signed user timing offset for overshoot-based braking onset.
          # Lower values (negative) begin slowing earlier; higher values delay onset.
          overshoot_phase_offset_s = float(getattr(self, '_overshoot_phase_offset_s', 0.0))
          self._overshoot_trigger_in_s = raw_trigger_distance / max(self._v_ego, 0.1) + overshoot_phase_offset_s
          self._v_overshoot_distance = max(raw_trigger_distance, float(self._overshoot_min_distance))

          _debug(
            f"TVC: Advanced High LatAcc. Dist: {self._v_overshoot_distance:.2f}, "
            f"v: {self._v_overshoot * CV.MS_TO_KPH:.2f}, anticipation: {anticipation_time:.1f}s"
          )

        return  # Successfully processed vision data

    # Vision not good or model not available: use held curvature (adjusted_curvature)
    adjusted_curvature = self._update_vision_occlusion(model_data, current_time)
    current_curvature = max(0.0, float(adjusted_curvature))
    current_curvature_signed = 0.0
    max_pred_curvature = current_curvature

    # Update filtered curvature and dependent quantities
    self._filtered_curvature = ((1 - self._curvature_ema_ratio) * self._filtered_curvature +
                               self._curvature_ema_ratio * max_pred_curvature)
    self._current_lat_acc = current_curvature_signed * self._v_ego**2
    self._max_pred_lat_acc = self._v_ego**2 * max_pred_curvature
    self._max_v_for_current_curvature = curvature_to_speed(current_curvature) if current_curvature > 0 else V_CRUISE_MAX * CV.KPH_TO_MS
    self._lat_acc_overshoot_ahead = (self._max_v_for_current_curvature < self._v_ego)
    self._v_overshoot = min(self._max_v_for_current_curvature, self._v_cruise_setpoint)
    # Conservative default distance handling
    if self._lat_acc_overshoot_ahead:
      # Default conservative distance handling when vision not good: ensure a reasonable floor
      default_floor = max(20.0, 2.0 * float(self._overshoot_min_distance))
      self._v_overshoot_distance = max(getattr(self, '_v_overshoot_distance', default_floor), default_floor)
      self._overshoot_trigger_in_s = self._v_overshoot_distance / max(self._v_ego, 0.1)
    else:
      self._v_overshoot_distance = getattr(self, '_v_overshoot_distance', 200.0)
      self._overshoot_trigger_in_s = float('inf')
    
    # Track curvature change rate for anticipation moderation (20 Hz assumed)
    try:
      prev = self._prev_filtered_curvature
    except AttributeError:
      prev = self._filtered_curvature
    rate = (abs(self._filtered_curvature) - abs(prev)) / 0.05
    self._abs_curvature_rate = rate
    self._is_easing = (rate <= 0.0)
    self._prev_filtered_curvature = self._filtered_curvature
  def _state_transition(self):
    """Compatibility shell for legacy VTSC state telemetry.

    The state machine no longer drives VTSC behavior; it is retained only so traces and
    tooling that expect this method/field continue to function.
    """
    # System-level disable conditions still clear hold state.
    if not self._op_enabled or not self._is_enabled or self._gas_pressed:
      self._v_turn_hold_until = 0.0
      self._v_turn_hold_min = float(INF_SPEED)
      self.state = VisionTurnControllerState.disabled
      return
    self.state = VisionTurnControllerState.disabled

  def _update_solution(self):
    """SIMPLIFIED: Always run physics calculations - let longitudinal planner decide usage."""
    dt = 0.05  # 20Hz

    # Apply temporary fast reacquisition filter alpha if armed
    now = time.time()
    # Time since occlusion started (for early-phase behaviors)
    try:
      _t0_occ = float(getattr(self._occlusion_state, 'occluded_since_time', 0.0) or 0.0)
    except Exception:
      _t0_occ = 0.0
    occ_age = max(0.0, now - _t0_occ)
    if now < getattr(self, '_fast_reacq_until', 0.0):
      self._filter_alpha = max(self._base_filter_alpha, float(getattr(self, '_fast_reacq_alpha', 0.85)))
    else:
      self._filter_alpha = self._base_filter_alpha
    occl_bypass = bool(getattr(self, '_occl_lead_bypass_active', False))

    # SIMPLIFIED: Always run advanced planning logic - no activation thresholds
    # On straight roads: will return cruise setpoint, longitudinal planner ignores (other sources lower)
    # On curves: will return physics speed, longitudinal planner uses it (lowest source)
    # Calculate target speed using advanced planning
    raw_target = self._plan_advanced_speed_trajectory()
    if raw_target is None:
      raw_target = self._prev_target_speed if hasattr(self, '_prev_target_speed') else self._v_ego
    # Debug: record raw target prior to map caps
    try:
      self._dbg_target_raw = float(raw_target)
    except Exception:
      self._dbg_target_raw = float(self._prev_target_speed if hasattr(self, '_prev_target_speed') else self._v_ego)

    # Keep jerk scaling constant to respect caps
    scale_jerk = 1.0

    # Optional: apply map-based lookahead cap to extend horizon.
    # Map is "advance warning only": once vision has confident in-range turn evidence,
    # suppress map capping so vision remains the source of truth.
    self._map_tail_reason = "toggle_off"
    try:
      if self._get_bool_param('MTSCLookaheadEnabled', False):
        self._map_tail_reason = "enabled_no_cap"
        v_cap, s_start, coverage = self._map_tail_cap()
        if v_cap is not None:
          v_cap_f = float(v_cap)
          s_start_f = float(s_start)
          coverage_f = float(coverage)
          # Keep latest map diagnostics even when map cap is suppressed.
          self._map_tail_last_cap = v_cap_f
          self._map_tail_last_start = s_start_f
          self._map_tail_last_coverage = coverage_f

          map_cap_allowed = True
          try:
            v_ego_local = float(max(0.0, self._v_ego))
            s_visible = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4)) * v_ego_local)
            vis_margin = float(max(0.0, getattr(self, '_vis_margin_m', 10.0)))
            k_turn_min = float(max(1e-6, getattr(self, '_fov_k_min', 2e-4)))
            k_now = float(abs(getattr(self, '_filtered_curvature', 0.0)))
            turn_visible_now = bool(k_now >= k_turn_min)
            turn_visible_ahead = bool(
              bool(getattr(self, '_lat_acc_overshoot_ahead', False)) and
              (float(getattr(self, '_v_overshoot_distance', 1e9)) <= (s_visible + vis_margin))
            )
            vision_good = bool(getattr(self._occlusion_state, 'vision_good', True))
            # Once vision has eyes-on turn evidence, map should no longer tighten VTSC.
            map_cap_allowed = not bool(vision_good and (turn_visible_now or turn_visible_ahead))
          except Exception:
            map_cap_allowed = True

          if map_cap_allowed:
            raw_target = min(raw_target, v_cap_f)
            self._map_tail_active = True
            self._map_tail_reason = "applied"
          else:
            self._map_tail_active = False
            self._map_tail_reason = "vision_suppressed"
        else:
          self._map_tail_active = False
          self._map_tail_reason = str(getattr(self, '_map_tail_compute_reason', '') or 'no_cap')
      else:
        # Ensure HUD preview does not persist when map lookahead is disabled.
        self._clear_curve_preview()
    except Exception:
      self._map_tail_active = False
      self._map_tail_reason = "exception"
    # Debug: record final target after map caps
    try:
      self._dbg_target_final = float(raw_target)
    except Exception:
      self._dbg_target_final = float(self._prev_target_speed if hasattr(self, '_prev_target_speed') else self._v_ego)

    # Track the speed cap VTSC intends to publish to the planner. This starts from the
    # base/vision target and is tightened by occlusion/map/other constraints.
    try:
      v_target_cap = float(raw_target)
    except Exception:
      v_target_cap = float(self._v_cruise_setpoint)
    # Overshoot cap timing gate:
    # - Engage once computed "time-to-start-braking" is reached.
    # - Release this extra cap once apex-exit logic says we're past apex in an easing phase, so the
    #   planner can start accelerating out while still obeying the visible-curve cap.
    self._overshoot_cap_active = False
    try:
      if bool(getattr(self, '_lat_acc_overshoot_ahead', False)):
        trigger_in_s = float(getattr(self, '_overshoot_trigger_in_s', float('inf')))
        should_start = bool(trigger_in_s <= 0.0)
        apex_release = bool(getattr(self, '_apex_exit_ready', False) and getattr(self, '_is_easing', False))
        if should_start and not apex_release:
          v_target_cap = min(v_target_cap, float(getattr(self, '_v_overshoot', v_target_cap)))
          self._overshoot_cap_active = True
    except Exception:
      self._overshoot_cap_active = False

    # ===== Freeway sanity guard (fail-open) =====
    try:
      kappa_vis = float(abs(getattr(self._occlusion_state, 'last_valid_curvature', 0.0)))
      s_visible_m = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4) * max(0.0, self._v_ego)))
      path_conf = float(getattr(self._occlusion_state, 'smoothed_confidence', 0.0))
    except Exception:
      kappa_vis, s_visible_m, path_conf = 0.0, 0.0, 0.0
    # Param override to force fail-open during on-road triage
    failopen_param = bool(self._get_bool_param('VTSCFailOpen', False))
    fail_open = bool((kappa_vis <= FREEWAY_CURV_EPS) and (s_visible_m >= FREEWAY_MIN_VISIBLE_M) and (path_conf >= FREEWAY_MIN_CONF))
    self._freeway_failopen_active = bool(fail_open or failopen_param)
    # Snapshot fields for triage
    self._dbg_kappa_vis = kappa_vis
    self._dbg_s_visible_m = s_visible_m
    self._dbg_path_conf = path_conf
    self._dbg_fail_open = bool(self._freeway_failopen_active)

    # FOV-based occlusion gate computation with pre-trigger and stickiness
    try:
      psi_fov = float(getattr(self, '_psi_fov_rad', 0.49))
      psi_margin = float(getattr(self, '_psi_margin_rad', 0.087))
    except Exception:
      psi_fov, psi_margin = 0.49, 0.087
    # instantaneous psi (use filtered curvature to anticipate FOV exit)
    try:
      kappa_gate = float(getattr(self, '_filtered_curvature', 0.0))
    except Exception:
      kappa_gate = kappa_vis
    self._dbg_psi_vis = float(abs(kappa_gate) * max(0.0, s_visible_m))
    self._dbg_psi_thresh = float(max(0.0, psi_fov - psi_margin))
    # Pre-trigger by time-to-FOV-exit (TTFOV)
    k_min = float(getattr(self, '_fov_k_min', 2e-4))
    k_free = float(getattr(self, '_fov_k_freeway', FREEWAY_CURV_EPS))
    try:
      delta_psi = max(0.0, self._dbg_psi_thresh - self._dbg_psi_vis)
      delta_s_to_exit = delta_psi / max(abs(kappa_gate), 1e-9)
      ttfov_s = delta_s_to_exit / max(self._v_ego, 0.1)
    except Exception:
      ttfov_s = 999.0
    self._dbg_ttfov_s = float(ttfov_s)
    pretrigger_time = float(getattr(self, '_fov_pretrigger_time_s', 1.2))
    # Only allow pretrigger when confidence is actually degraded; otherwise, favor fresh vision
    try:
      _conf_s = float(getattr(self._occlusion_state, 'smoothed_confidence', 1.0))
      _conf_bad = float(getattr(self._occlusion_state, 'bad_threshold', 0.65))
      _conf_good = float(getattr(self._occlusion_state, 'good_threshold', 0.70))
      _vis_good = bool(getattr(self._occlusion_state, 'vision_good', True)) and (_conf_s >= _conf_good)
      # Track slope to suppress pretrigger while confidence is rising from borderline
      try:
        _conf_prev = float(getattr(self._occlusion_state, 'prev_smoothed_conf', _conf_s))
      except Exception:
        _conf_prev = _conf_s
      setattr(self._occlusion_state, 'prev_smoothed_conf', _conf_s)
      conf_rising = (_conf_s >= _conf_prev + 0.005)  # ~0.5% absolute rise per update
    except Exception:
      _conf_s, _conf_bad, _conf_good, _vis_good, conf_rising = 1.0, 0.65, 0.70, True, False
    pretrigger = (ttfov_s <= pretrigger_time) and (abs(kappa_gate) >= k_min) and (_conf_s < _conf_bad) and (not conf_rising)
    # Suppress pretrigger during short model dropouts to avoid overreacting
    try:
      if bool(getattr(self._occlusion_state, 'dropout_active', False)):
        pretrigger = False
    except Exception:
      pass
    # Hysteretic onset/clear
    onset_geom = (abs(kappa_gate) >= k_min) and (self._dbg_psi_vis >= self._dbg_psi_thresh)
    onset = (onset_geom or pretrigger) and (not _vis_good)
    # Clear when geometry says we're no longer near a field-of-view exit.
    #
    # Rationale:
    # - Real roads often have mediocre lane-line confidence for benign reasons (wear, glare, lead vehicle).
    # - VTSC is an assistant; once the path is safely within FoV again, we should recover like a
    #   reasonable human would, instead of "sticking" in occlusion due to confidence alone.
    clear = (abs(kappa_vis) < k_free) or (self._dbg_psi_vis < self._dbg_psi_thresh) or _vis_good
    # If vision is good, forcibly clear occlusion state and reset counters immediately
    if _vis_good and self._fov_occluded:
      self._fov_occluded = False
      self._fov_on_cnt = 0
      self._fov_off_cnt = 0
      self._fov_boost_left = 0
      self._fov_overshoot_left = 0
      self._onset_no_raise_active = False
    if onset and not self._fov_occluded:
      self._fov_on_cnt = int(self._fov_on_cnt) + 1
      self._fov_off_cnt = 0
      if self._fov_on_cnt >= int(getattr(self, '_fov_N_on', 5)):
        self._fov_occluded = True
        self._fov_reason = 'pretrigger' if pretrigger else 'fov_exit'
        self._fov_boost_left = int(getattr(self, '_fov_onset_boost_frames', 10))
        self._fov_overshoot_left = int(getattr(self, '_fov_overshoot_frames', 10))
        # initialize EWMA curvature at current filtered
        try:
          self._fov_kappa_ewma = float(getattr(self, '_filtered_curvature', 0.0))
        except Exception:
          self._fov_kappa_ewma = 0.0
    elif clear and self._fov_occluded:
      # During onset stickiness window, do not clear on small margin
      if int(getattr(self, '_fov_boost_left', 0)) <= 0:
        self._fov_off_cnt = int(self._fov_off_cnt) + 1
        self._fov_on_cnt = 0
        if self._fov_off_cnt >= int(getattr(self, '_fov_N_off', 10)):
          self._fov_occluded = False
          self._fov_reason = 'freeway' if abs(kappa_vis) < k_free else 'short_vis'
    else:
      self._fov_on_cnt = max(0, int(self._fov_on_cnt) - 1)
      self._fov_off_cnt = max(0, int(self._fov_off_cnt) - 1)
    # decay windows
    if int(getattr(self, '_fov_boost_left', 0)) > 0:
      self._fov_boost_left -= 1
    if int(getattr(self, '_fov_overshoot_left', 0)) > 0:
      self._fov_overshoot_left -= 1

    # Compute acceleration command to drive current speed toward target
    accel_cmd = (raw_target - self._v_ego) / dt
    # ===== Severe confidence occlusion guard (non-FOV) =====
    # If vision confidence is extremely low (SEVERE/LOST), do not allow increasing speed.
    #
    # This is intentionally *not* tied to FOV geometry; it protects against cases where the
    # model has effectively no reliable lane-line confidence (e.g., glare/washed-out markings),
    # but the FOV gate might not trigger (gentle curvature / low psi).
    #
    # Lead bypass MUST override this, since lead presence can reduce lane-line confidence
    # without representing a true visibility occlusion.
    try:
      vs = getattr(self._occlusion_state, 'vision_status', None)
      severe_conf = (vs == VisionStatus.SEVERE_OCCLUSION) or (vs == VisionStatus.VISION_LOST)
    except Exception:
      severe_conf = False
    # Only apply this guard when there is meaningful curvature (i.e., when VTSC is relevant).
    # On straight roads, low lane-line confidence can happen for reasons unrelated to "can't see around a bend"
    # (e.g., worn paint, glare, lead vehicle covering lane lines), and we don't want VTSC to interfere.
    try:
      k_gate_abs = abs(float(kappa_gate))
    except Exception:
      k_gate_abs = 0.0
    try:
      min_operating_v = float(_MIN_V)
    except Exception:
      min_operating_v = 0.0
    low_speed_guard = bool(self._v_ego <= max(0.0, min_operating_v))
    severe_conf_no_raise = bool(
      severe_conf and (not occl_bypass) and (not self._freeway_failopen_active)
      and (k_gate_abs >= float(k_min)) and (not low_speed_guard)
    )
    if severe_conf_no_raise:
      accel_cmd = min(accel_cmd, 0.0)
      try:
        v_target_cap = min(float(v_target_cap), float(self._v_ego))
      except Exception:
        pass
    # Fast reacquisition nudge: only if physics base supports acceleration above current speed
    try:
      now_t = time.time()
      if bool(getattr(self._occlusion_state, 'vision_good', True)) and (now_t <= float(getattr(self, '_fast_reacq_until', 0.0))):
        try:
          base_cap = float(min(self._v_cruise_setpoint, curvature_to_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
        except Exception:
          base_cap = float(self._v_cruise_setpoint)
        # Require a small margin to ensure this is a raise scenario
        if (base_cap > (self._v_ego + 0.05)) and (float(getattr(self, '_prev_target_speed', self._v_ego)) < 0.98 * base_cap):
          accel_cmd = max(accel_cmd, 0.18)
    except Exception:
      pass
    # If FOV-gated occlusion is active, fold in a conservative occlusion cap immediately
    self._dbg_low_speed_margin = False
    if self._fov_occluded and (not occl_bypass) and (not self._freeway_failopen_active):
      try:
        k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      except Exception:
        k_est = 0.0
      k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
      # During early overshoot window, use EWMA to be conservative
      if int(getattr(self, '_fov_overshoot_left', 0)) > 0:
        try:
          tau = float(getattr(self, '_fov_ewma_tau_s', 0.5))
          alpha = 1.0 - math.exp(-dt / max(1e-3, tau))
        except Exception:
          alpha = 0.5
        self._fov_kappa_ewma = (1.0 - alpha) * float(getattr(self, '_fov_kappa_ewma', k_filt)) + alpha * k_filt
        k_cons = max(1e-8, min(self._fov_kappa_ewma, k_est))
      else:
        k_cons = max(1e-8, min(k_filt, k_est))
      # === κ-bias at occluded onset: briefly bias occlusion cap to ensure decel starts ===
      # Use existing onset stickiness window (_fov_boost_left) as a safe timing window.
      try:
        psi_margin_deg = float(getattr(self, '_psi_margin_rad', 0.087)) * 57.2957795
      except Exception:
        psi_margin_deg = 0.0
      try:
        ttfov_s = float(getattr(self, '_dbg_ttfov_s', 999.0))
      except Exception:
        ttfov_s = 999.0
      pretrigger_time = float(getattr(self, '_fov_pretrigger_time_s', 1.5))
      slack_s = 0.10
      v_max_mps = 36.0
      try:
        boost_left = int(getattr(self, '_fov_boost_left', 0))
      except Exception:
        boost_left = 0
      onset_gate = (boost_left > 0) and (psi_margin_deg >= 5.0) and (self._v_ego <= v_max_mps) and (ttfov_s <= (pretrigger_time + slack_s))
      try:
        if onset_gate:
          # Disable curvature inflation at onset; do not alter k_cons
          self._dbg_onset_bias_active = False
          self._dbg_onset_gate_reason = {
            'psi_ok': True,
            'speed_ok': True,
            'ttfov_ok': True,
            'boost_left': int(boost_left),
          }
        else:
          self._dbg_onset_bias_active = False
          self._dbg_onset_gate_reason = {
            'psi_ok': bool(psi_margin_deg >= 5.0),
            'speed_ok': bool(self._v_ego <= v_max_mps),
            'ttfov_ok': bool(ttfov_s <= (pretrigger_time + slack_s)),
            'boost_left': int(boost_left),
          }
      except Exception:
        self._dbg_onset_bias_active = False
        self._dbg_onset_gate_reason = {'error': True}
      v_occ_cap = float(curvature_to_speed(k_cons))
      accel_cmd = min(accel_cmd, (v_occ_cap - self._v_ego) / dt)
      # Also constrain the published cap to reflect this occlusion cap.
      try:
        v_target_cap = min(float(v_target_cap), float(v_occ_cap))
      except Exception:
        pass

      # === Consolidated curvature cap and early no-raise during onset window ===
      # Maintain onset timers on rising edge of occlusion
      if self._fov_occluded and not getattr(self, '_occlusion_prev', False):
        self._occlusion_onset_timer_s = 0.0
        # Estimate visible cap at onset for reference (min of cruise and filtered-curvature speed)
        try:
          cap_vis_onset = float(min(self._v_cruise_setpoint, curvature_to_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
        except Exception:
          cap_vis_onset = float(self._v_cruise_setpoint)
        # Active cap approx at onset
        self._v_cap_active_at_onset_mps = float(min(cap_vis_onset, v_occ_cap))
        # Arm a short vision-floor TTL to avoid immediate depression below LKG
        try:
          self._vision_floor_until = float(time.time()) + float(getattr(self, '_vision_floor_ttl_s', 2.5))
        except Exception:
          self._vision_floor_until = 0.0
      if self._fov_occluded:
        self._occlusion_onset_timer_s += dt
      self._occlusion_prev = bool(self._fov_occluded)

      # Onset window remains visible in debug breadcrumbs; no-raise behavior is disabled.
      try:
        psi_margin_deg = float(getattr(self, '_psi_margin_rad', 0.087)) * 57.2957795
      except Exception:
        psi_margin_deg = 0.0
      v_max_mps = 36.0
      window_s = 0.8
      # Relax TTFOV gating inside onset window to ensure assist engages
      onset_gate = (self._fov_occluded and (psi_margin_deg >= 5.0) and (self._v_ego <= v_max_mps) and (self._occlusion_onset_timer_s <= window_s))
      if onset_gate:
        # Disable onset curvature inflation and decel floors; rely on v_occ_cap from k_cons only
        self._dbg_cap_source = 'occluded_onset_disabled'
      self._onset_no_raise_active = False

    # Occlusion-time accel gating: allow positive accel only with positive margin
    occl_positive_margin = False
    early_no_raise = False  # suppress positive accel in early hidden-turn phase
    occl_effects_active = bool(self._fov_occluded and (not occl_bypass) and (not self._freeway_failopen_active))
    if occl_effects_active:
      v_gate_hi = HIGHWAY_MIN_MPS  # ~55 mph
      if self._v_ego < v_gate_hi:
        # Gradual bias toward pure physics mode between ~50 and 65 mph (no hard bypass)
        # Compute barrier context to determine margin (near vs. far)
        try:
          v_vis = curvature_to_speed(max(1e-8, float(self._occlusion_state.last_valid_curvature)))
          # When occluded, do NOT let the filtered/model curvature drive the near cap;
          # rely on last-visible curvature (v_vis) for the near bound.
          v_near = min(v_vis, self._v_cruise_setpoint)
          # Use the more conservative (lower speed) of occlusion est curvature and filtered curvature
          try:
            k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
          except Exception:
            k_est = 0.0
          k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
          v_occ_raw = min(curvature_to_speed(max(1e-8, k_est)), curvature_to_speed(k_filt))
          s_vis = max(0.0, float(getattr(self, '_vis_horizon_s', 1.4)) * max(0.0, self._v_ego))
          a_cap = abs(float(self._comfort_decel_limit))
          v_now = max(self._prev_target_speed, self._v_ego)
          # Tail-aware far bound for gating
          try:
            dist_since = float(getattr(self._occlusion_state, 'distance_since_m', 0.0))
          except Exception:
            dist_since = 0.0
          s_tail = max(0.0, dist_since - s_vis)
          try:
            k_now = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
          except Exception:
            k_now = 0.0
          if k_now <= 0.004:
            tail_frac = 0.10
          elif k_now >= 0.008:
            tail_frac = 0.90
          else:
            # Interpolate from 0.10 at 0.004 to 0.90 at 0.008 (match harness diagnostics)
            tail_frac = 0.10 + 0.80 * ((k_now - 0.004) / 0.004)
          # Speed-based gating: ignore tail floor above ~55 mph (pure physics) and taper between 50–55 mph
          v_gate_lo = 22.35  # m/s ~50 mph
          v_gate_hi = 24.5872  # m/s ~55 mph
          v_now_for_gate = max(0.0, self._v_ego)
          blend = (
            0.0 if v_now_for_gate <= v_gate_lo else
            (1.0 if v_now_for_gate >= v_gate_hi else (v_now_for_gate - v_gate_lo) / max(1e-6, (v_gate_hi - v_gate_lo)))
          )
          tail_frac_eff = tail_frac * (1.0 - blend)
          try:
            v_cap_tail_eff = math.sqrt(max(0.0, v_now * v_now - 2.0 * a_cap * (tail_frac_eff * s_tail)))
          except Exception:
            v_cap_tail_eff = v_now
          # Compute two far bounds: one with speed-blended tail (for internal use) and one matching harness (for gating)
          v_far_gate = min(v_occ_raw, v_cap_tail_eff, self._v_cruise_setpoint)
          # Harness-equivalent far bound (no speed-based blend on tail fraction)
          try:
            v_cap_tail_h = math.sqrt(max(0.0, v_now * v_now - 2.0 * a_cap * (tail_frac * s_tail)))
          except Exception:
            v_cap_tail_h = v_now
          v_far_h = min(v_occ_raw, v_cap_tail_h, self._v_cruise_setpoint)
          d_req_h = max(0.0, (v_now * v_now - v_far_h * v_far_h) / max(2e-3, 2.0 * a_cap))
          margin_dist = float(getattr(self, '_vis_margin_m', 10.0))
          # Use harness-equivalent margin for gating decisions with small buffer (≈2 m)
          positive_margin = (d_req_h <= (s_vis - (margin_dist + 2.0)))
          early_phase = int(getattr(self, '_fov_on_cnt', 0)) <= 10
          occl_positive_margin = bool(positive_margin and not early_phase)
          # Snapshot for telemetry
          try:
            self._dbg_s_tail = float(s_tail)
            self._dbg_tail_frac = float(tail_frac)
          except Exception:
            self._dbg_s_tail = 0.0
            self._dbg_tail_frac = 0.0
          # ===== Hidden-turn early deceleration trigger (short-horizon critical deficit) =====
          if HIDDEN_TURN_ENABLED and (self._v_ego <= HIDDEN_TURN_V_MAX_MPS):
            try:
              now = time.time()
            except Exception:
              now = 0.0
            t0 = float(getattr(self._occlusion_state, 'occluded_since_time', 0.0) or 0.0)
            occ_elapsed = max(0.0, now - t0)
            if occ_elapsed >= HIDDEN_TURN_MIN_OCC_S and occ_elapsed <= HIDDEN_TURN_PHASE_S:
              v_req_hidden = float(v_cap_tail_eff)
              deficit = max(0.0, v_now - v_req_hidden)
              try:
                vs = getattr(self._occlusion_state, 'vision_status', None)
              except Exception:
                vs = None
              use_shed = (vs == VisionStatus.SEVERE_OCCLUSION or vs == VisionStatus.VISION_LOST)
              if deficit >= HIDDEN_TURN_DELTA_V_MPS or use_shed:
                d_avail_time = max(0.0, self._v_ego * HIDDEN_TURN_T_H_S)
                d_avail_vis = max(0.0, s_vis - margin_dist)
                d_avail = min(d_avail_time, d_avail_vis)
                if use_shed:
                  v_req_min = max(0.0, v_now - HIDDEN_TURN_DELTA_V_MPS)
                  d_req_hidden = max(0.0, (v_now * v_now - v_req_min * v_req_min) / max(2e-3, 2.0 * a_cap))
                else:
                  d_req_hidden = max(0.0, (v_now * v_now - v_req_hidden * v_req_hidden) / max(2e-3, 2.0 * a_cap))
                # Soft early-phase visible-heading tightening for hidden-turns
                try:
                  entry_kappa = abs(float(getattr(self._occlusion_state, 'entry_curvature', 0.0) or 0.0))
                except Exception:
                  entry_kappa = 0.0
                s_head = max(6.0, self._v_ego * HIDDEN_TURN_HEADING_WIN_S)
                vis_heading_rad = entry_kappa * s_head
                straightness_gain = max(0.0, min(1.0,
                  (HIDDEN_TURN_VIS_HEADING_MAX_RAD - vis_heading_rad) / max(1e-6, HIDDEN_TURN_VIS_HEADING_MAX_RAD)))
                phase_progress = max(0.0, min(1.0, occ_elapsed / max(1e-6, HIDDEN_TURN_PHASE_S)))
                early_tighten = 0.55 * straightness_gain * (1.0 - phase_progress)
                short_h_avail = (HIDDEN_TURN_AVAIL_SCALE * d_avail) * (1.0 - early_tighten)
                if d_req_hidden > short_h_avail:
                  v_occ_raw = min(v_occ_raw, v_req_hidden)
                  positive_margin = False
                  # Suppress positive accel during early hidden-turn phase (~1.5s)
                  early_no_raise = True
          reachable_cap = v_near
        except Exception:
          positive_margin = False
          reachable_cap = self._v_cruise_setpoint
        # Under positive margin allow non-negative acceleration; otherwise decelerate toward barrier target
        if positive_margin:
          # Drive a gentle raise toward reachable_cap using existing jerk limits
          pos_limit = max(0.0, float(getattr(self, '_max_accel', 1.0)))
          desired_target = max(self._v_ego, min(reachable_cap, self._v_ego + pos_limit * 0.05))
          if not early_no_raise:
            accel_cmd = max(accel_cmd, (desired_target - self._v_ego) / 0.05)
          # Never decelerate while margin is positive
          accel_cmd = max(accel_cmd, 0.0)
        else:
          # Produce barrier target using near/far policy and fold into accel (favor decel)
          # Include far-field bound for:
          # - Highway (>36 m/s)
          # - Moderate/mountain speeds when curvature is meaningful (k_now ≥ 0.004)
          # - All sub-30 m/s regimes to ensure timely slowing for hidden/abrupt turns outside FoV
          use_far = (self._v_ego > 36.0) or (self._v_ego <= 36.0 and k_now >= 0.004) or (self._v_ego <= 30.0)
          if use_far:
            barrier_target_speed = min(min(v_near, v_far_gate), v_now)
          else:
            # Very low curvature at low speeds: stick to near bound to avoid crawl
            barrier_target_speed = min(v_near, v_now)
          # Fold barrier target into commanded deceleration (respect jerk limits downstream):
          # Pull toward barrier target; prefer more conservative (more negative) acceleration
          accel_cmd = min(accel_cmd, (barrier_target_speed - self._v_ego) / 0.05)
          # Constrain published cap for planner ingestion.
          try:
            v_target_cap = min(float(v_target_cap), float(barrier_target_speed))
          except Exception:
            pass

        if (not occl_positive_margin) and (self._v_ego <= LOW_SPEED_MARGIN_MAX_V_MPS) and (k_now <= LOW_SPEED_MARGIN_CURV_THRESH):
          occl_positive_margin = True
          self._dbg_low_speed_margin = True
      else:
        # Highway-speed occlusion path: compute near/far context locally.
        # NOTE: Tail-floor terms are intentionally omitted here; above ~55 mph we operate in
        # "pure physics" mode (v_occ_raw + last-visible curvature), and rely on downstream
        # occlusion "no-raise" gating to prevent inappropriate acceleration.
        try:
          v_vis = curvature_to_speed(max(1e-8, float(self._occlusion_state.last_valid_curvature)))
        except Exception:
          v_vis = float(self._v_cruise_setpoint)
        v_near = min(float(v_vis), float(self._v_cruise_setpoint))
        try:
          k_now = abs(float(getattr(self._occlusion_state, 'est_curvature', 0.0) or 0.0))
        except Exception:
          k_now = 0.0
        try:
          k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
        except Exception:
          k_est = 0.0
        k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
        v_occ_raw = min(curvature_to_speed(max(1e-8, k_est)), curvature_to_speed(k_filt))
        v_now = max(float(getattr(self, '_prev_target_speed', self._v_ego)), float(self._v_ego))
        v_far_gate = min(float(v_occ_raw), float(self._v_cruise_setpoint))

        # Produce barrier target using near/far policy and fold into accel (favor decel)
        # Include far-field bound for:
        # - Highway (>36 m/s)
        # - Moderate/mountain speeds when curvature is meaningful (k_now ≥ 0.004)
        # - All sub-30 m/s regimes to ensure timely slowing for hidden/abrupt turns outside FoV
        use_far = (self._v_ego > 36.0) or (self._v_ego <= 36.0 and k_now >= 0.004) or (self._v_ego <= 30.0)
        if use_far:
          barrier_target_speed = min(min(v_near, v_far_gate), v_now)
        else:
          barrier_target_speed = min(v_near, v_now)
        # Fold barrier target into commanded deceleration (respect jerk limits downstream):
        # Pull toward barrier target; prefer more conservative (more negative) acceleration
        accel_cmd = min(accel_cmd, (barrier_target_speed - self._v_ego) / 0.05)
        # Constrain published cap for planner ingestion.
        try:
          v_target_cap = min(float(v_target_cap), float(barrier_target_speed))
        except Exception:
          pass
    # ===== APPLY ADAPTIVE DECELERATION SYSTEM =====
    # Enforce no positive acceleration while occluded unless positive margin exists.
    # Additionally, suppress positive accel in early hidden-turn phase.
    if occl_effects_active:
      # Suppress raising during recent speed-limit step down while occluded
      try:
        if time.time() < getattr(self, '_limit_step_until', 0.0):
          early_no_raise = True
      except Exception:
        pass
      # If a speed-limit down-step occurred, suppress raising entirely until vision is good again
      if getattr(self, '_suppress_raise_due_to_limit', False):
        early_no_raise = True
      # Straight-road exemption: when curvature is near-zero, there is no hidden turn
      # to protect against.  Holding the cap at v_ego on a straight road after a lead
      # car turns off traps the system at low speed with no hazard justification.
      try:
        _k_for_noraise = float(abs(getattr(self, '_filtered_curvature', 0.0)))
      except Exception:
        _k_for_noraise = 1.0  # fail conservative
      straight_road = (_k_for_noraise < 5e-4)  # ~2000m radius — effectively straight
      if (not occl_positive_margin) or early_no_raise:
        if not straight_road:
          accel_cmd = min(accel_cmd, 0.0)
          # Mirror the "no-raise" behavior in the published speed cap:
          # if we are occluded and disallowing positive accel, we must not publish a cap above v_ego
          # (otherwise the planner/MPC will accelerate).
          try:
            v_target_cap = min(float(v_target_cap), float(self._v_ego))
          except Exception:
            pass
    # record for telemetry
    self._dbg_occl_positive_margin = bool(occl_positive_margin)
    self._dbg_early_no_raise = bool(early_no_raise)
    # Check if deceleration is required
    if accel_cmd < 0:
        # For curve scenarios, use physics-based calculation if needed
        if self._lat_acc_overshoot_ahead:
            remaining_distance = self._v_overshoot_distance
            physics_required_decel = self._calculate_required_deceleration(
                self._v_ego, self._v_overshoot, remaining_distance)
            # Use the more conservative (more negative) of commanded or physics-required decel
            accel_cmd = min(accel_cmd, physics_required_decel)

        # While occluded, avoid over-braking: cap to comfort decel limit
        if occl_effects_active:
            accel_cmd = max(accel_cmd, self._comfort_decel_limit)

        # Apply adaptive deceleration system with noise filtering
        accel_cmd = self._get_optimal_deceleration(accel_cmd, dt)
        # While occluded, ensure decel command does not exceed comfort cap after filtering
        if occl_effects_active:
          try:
            self._current_decel = max(self._current_decel, self._comfort_decel_limit)
          except Exception:
            pass

        # Monitor adaptive deceleration performance
        remaining_distance = self._v_overshoot_distance if self._lat_acc_overshoot_ahead else 100.0
        self._monitor_adaptive_deceleration(accel_cmd, remaining_distance)
    else:
      # Clear suppression after reacquisition
      self._suppress_raise_due_to_limit = False
      # For acceleration, use normal limits
      pos_limit = self._max_accel
      # Apply a small fast-reacquisition acceleration floor for up to _fast_reacq_window_s
      now = time.time()
      # If just reacquired within 0.65s, ensure a small additional push to close gap sooner
      if getattr(self._occlusion_state, 'reacquired_at', 0.0) > 0.0 and (now - self._occlusion_state.reacquired_at) <= 0.65:
        accel_cmd = max(accel_cmd, 0.18)
        if self._occlusion_state.vision_good and now < getattr(self, '_fast_reacq_until', 0.0):
          accel_cmd = max(accel_cmd, 0.18)
        # Positive-margin uplift while occluded: after an initial dwell, apply a modest floor
        if self._fov_occluded and occl_positive_margin and occ_age > 1.5 and not getattr(self, '_onset_no_raise_active', False):
          accel_cmd = max(accel_cmd, 0.22)
      accel_cmd = min(accel_cmd, pos_limit)

      # Gradually decay filter during acceleration instead of hard reset
      # This preserves filter memory for smoother transitions
      self._current_decel = 0.0
      self._filtered_decel_requirement *= 0.95  # Decay filter by 5% per update
      # Only reset hysteresis state when filter is nearly zero
      if abs(self._filtered_decel_requirement) < 0.1:
        self._decel_hysteresis_state = False

    # Jerk-limit the change in acceleration
    accel_diff = accel_cmd - self._current_accel

    prev_accel_val = float(self._current_accel)
    if accel_diff > 0:
      # Cap positive jerk to 2.5 m/s^3 to meet comfort bounds in tests
      max_jerk_pos = min(self._max_jerk_accel * scale_jerk, 2.5)
      max_delta = max_jerk_pos * dt
      if accel_diff > max_delta:
        self._current_accel += max_delta
      else:
        self._current_accel = accel_cmd
    elif accel_diff < 0:
      max_delta = (self._max_jerk * scale_jerk) * dt
      if accel_diff < -max_delta:
        self._current_accel -= max_delta
      else:
        self._current_accel = accel_cmd
    else:
      self._current_accel = accel_cmd
    # compute jerk for telemetry (m/s^3)
    try:
      self._dbg_jerk_cmd = float((self._current_accel - prev_accel_val) / dt)
    except Exception:
      self._dbg_jerk_cmd = 0.0

    # Hard clamp: after a speed-limit step while occluded, disallow any positive acceleration
    if occl_effects_active and getattr(self, '_suppress_raise_due_to_limit', False) and self._current_accel > 0.0:
      self._current_accel = 0.0
    # Fast reacquisition acceleration floor: ensure a small positive nudge upon recovery
    try:
      now_ts2 = time.time()
    except Exception:
      now_ts2 = 0.0
    try:
      fast_reacq_until = float(getattr(self, '_fast_reacq_until', 0.0))
    except Exception:
      fast_reacq_until = 0.0
    try:
      now_ts2 = float(time.time())
    except Exception:
      now_ts2 = 0.0
    if getattr(self._occlusion_state, 'vision_good', True) and (now_ts2 <= fast_reacq_until):
      # Only enforce positive accel floor when physics base supports a raise
      try:
        base_cap2 = float(min(self._v_cruise_setpoint, curvature_to_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
      except Exception:
        base_cap2 = float(self._v_cruise_setpoint)
      if (base_cap2 > (self._v_ego + 0.05)) and (float(getattr(self, '_prev_target_speed', self._v_ego)) < 0.98 * base_cap2):
        self._current_accel = max(self._current_accel, 0.18)
    # Update target acceleration for compatibility
    self._a_target = self._current_accel
    # Remove global occlusion decel floor: allow target accel to follow physics and margin

    # Update previous target speed by integrating the commanded acceleration.
    # This makes the controller's internal target track what we actually commanded.
    self._prev_target_speed = max(0.0, self._prev_target_speed + self._current_accel * dt)

    # Publish cap to the planner: clamp to cruise setpoint and keep non-negative.
    try:
      v_publish = float(max(0.0, min(float(v_target_cap), float(self._v_cruise_setpoint))))
      self._v_turn_output = float(self._apply_freeway_v_turn_hold(v_publish))
    except Exception:
      self._v_turn_output = float(getattr(self, '_v_cruise_setpoint', 0.0) or 0.0)

    # ===== Determine winning cap for telemetry =====
    # Visible-cap should reflect only what is actually visible. While occluded,
    # use the last-known-good visible curvature instead of the filtered/model value.
    try:
      if occl_effects_active:
        k_vis_only = float(max(1e-8, float(getattr(self._occlusion_state, 'last_valid_curvature', 0.0))))
        cap_visible_vmin = float(min(self._v_cruise_setpoint, curvature_to_speed(k_vis_only)))
      else:
        k_filt_only = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
        cap_visible_vmin = float(min(self._v_cruise_setpoint, curvature_to_speed(k_filt_only)))
    except Exception:
      cap_visible_vmin = float(self._v_cruise_setpoint)
    try:
      try:
        k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      except Exception:
        k_est = 0.0
      k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
      if int(getattr(self, '_fov_overshoot_left', 0)) > 0:
        try:
          tau = float(getattr(self, '_fov_ewma_tau_s', 0.5))
          alpha = 1.0 - math.exp(-dt / max(1e-3, tau))
        except Exception:
          alpha = 0.5
        self._fov_kappa_ewma = (1.0 - alpha) * float(getattr(self, '_fov_kappa_ewma', k_filt)) + alpha * k_filt
        k_cons = max(1e-8, min(self._fov_kappa_ewma, k_est))
        cap_occl_vmin = float(curvature_to_speed(k_cons))
      else:
        cap_occl_vmin = float(curvature_to_speed(max(1e-8, k_est)))
    except Exception:
      cap_occl_vmin = 0.0
    try:
      cap_map_vmin = float(getattr(self, '_map_tail_last_cap', 0.0) or 0.0) if bool(getattr(self, '_map_tail_active', False)) else 0.0
    except Exception:
      cap_map_vmin = 0.0
    self._dbg_cap_visible_vmin = cap_visible_vmin
    self._dbg_cap_occl_vmin = cap_occl_vmin
    self._dbg_cap_map_vmin = cap_map_vmin
    # ===== Arbitration: PSI-gated occlusion, optional relax, and double-cap guard =====
    caps = [("visible", cap_visible_vmin)]

    consider_occl = bool(occl_effects_active)
    # Do not allow occlusion to depress below visible when we have good vision correlation
    try:
      if bool(getattr(self._occlusion_state, 'vision_good', True)):
        consider_occl = False
    except Exception:
      pass
    psi_gate_open = True
    psi_est = 0.0
    if consider_occl:
      # Estimate visible-horizon heading change (psi) from local curvature and visible horizon
      try:
        v_ego = float(max(0.0, self._v_ego))
      except Exception:
        v_ego = 0.0
      try:
        s_vis = float(max(0.0, getattr(self._occlusion_state, "vis_horizon_s", 1.2)) * v_ego)
      except Exception:
        s_vis = 0.0
      try:
        kappa = float(max(0.0, abs(getattr(self, "_filtered_curvature", 0.0))))
      except Exception:
        kappa = 0.0
      psi_est = kappa * s_vis
      try:
        psi_th = float(getattr(self, "_psi_thresh_rad", PSI_THRESH_RAD))
      except Exception:
        psi_th = float(PSI_THRESH_RAD)
      try:
        psi_hyst = float(getattr(self, "_psi_hyst_rad", PSI_HYST_RAD))
      except Exception:
        psi_hyst = float(PSI_HYST_RAD)
      # Simple hysteresis on the gate
      psi_gate_open = psi_est >= (psi_th - psi_hyst)
      consider_occl = consider_occl and psi_gate_open
      # Export debug breadcrumbs
      try:
        self._dbg_psi_est = float(psi_est)
        self._dbg_psi_gate_thresh = float(psi_th)
      except Exception:
        pass

    # Enough-vision predicate: skip occlusion capping when correlation is viable
    try:
      v_ego_local = float(max(0.0, self._v_ego))
    except Exception:
      v_ego_local = 0.0
    try:
      s_vis_m_local = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4)) * v_ego_local)
    except Exception:
      s_vis_m_local = 0.0
    try:
      conf_local = float(getattr(self._occlusion_state, 'smoothed_confidence', 1.0))
    except Exception:
      conf_local = 1.0
    enough_s = float(getattr(self, '_enough_s_visible_m', 35.0))
    enough_vision = bool(getattr(self._occlusion_state, 'vision_good', True)) or ((s_vis_m_local >= enough_s) and (conf_local >= FREEWAY_MIN_CONF))
    if enough_vision:
      consider_occl = False

    if consider_occl:
      # If confidence is near-zero and we've been in fov_exit for a while, relax occlusion vmin upward (bounded by visible)
      try:
        conf = float(getattr(self._occlusion_state, "smoothed_confidence", 1.0))
        occ_start = float(getattr(self._occlusion_state, "occlusion_start_time", 0.0))
        now_t = time.time()
        if (
          conf <= float(getattr(self, "_occl_conf_floor", OCCL_CONF_FLOOR))
          and (now_t - occ_start >= float(getattr(self, "_fov_exit_relax_s", FOV_EXIT_RELAX_S)))
        ):
          cap_occl_vmin = min(cap_visible_vmin, cap_occl_vmin + float(getattr(self, "_occl_vmin_nudge_mps", OCCL_VMIN_NUDGE_MPS)))
      except Exception:
        pass
      # Vision floor TTL: lift occl cap to at least a fraction of LKG speed for a short window
      try:
        now_t2 = time.time()
        if now_t2 <= float(getattr(self, '_vision_floor_until', 0.0)):
          try:
            k_lkg = float(abs(getattr(self._occlusion_state, 'last_valid_curvature', 0.0)))
          except Exception:
            k_lkg = 0.0
          v_lkg = float(curvature_to_speed(max(1e-8, k_lkg)))
          floor_mult = float(getattr(self, '_vision_floor_mult', 0.98))
          v_floor = floor_mult * v_lkg
          cap_occl_vmin = max(cap_occl_vmin, min(cap_visible_vmin, v_floor))
      except Exception:
        pass
      # Double-cap guard: skip occlusion if pre-cap target already ≤ occlusion vmin + eps
      try:
        raw_pre = float(getattr(self, "_pre_cap_target_speed", getattr(self, "_prev_target_speed", 0.0)))
      except Exception:
        raw_pre = float(getattr(self, "_prev_target_speed", 0.0))
      try:
        eps = float(getattr(self, "_double_cap_eps_mps", DOUBLE_CAP_EPS_MPS))
      except Exception:
        eps = float(DOUBLE_CAP_EPS_MPS)
      if raw_pre <= cap_occl_vmin + eps:
        try:
          self._dbg_double_cap_guard = True
        except Exception:
          pass
        consider_occl = False

    try:
      self._dbg_consider_occl = bool(consider_occl)
    except Exception:
      pass
    if consider_occl:
      caps.append(("occlusion", cap_occl_vmin))
    if bool(getattr(self, '_map_tail_active', False)) and cap_map_vmin > 0.0:
      caps.append(("map", cap_map_vmin))
    try:
      active_cap, _ = min(caps, key=lambda kv: kv[1])
    except Exception:
      active_cap = "none"
    self._dbg_active_cap = str(active_cap)
    # Expose commanded min speed approximation for telemetry
    try:
      self._dbg_vtsc_cmd = float(self.v_turn)
    except Exception:
      self._dbg_vtsc_cmd = float(self._prev_target_speed)

  def _plan_advanced_speed_trajectory(self) -> float:
    """SIMPLIFIED: Always calculate physics-based speed, let longitudinal planner handle activation."""
    self._apex_exit_ready = False
    self._apex_trigger_idx = int(getattr(self, '_apex_near_index', 3))

    # Always calculate physics-based speed regardless of curvature amount
    # On straight roads: will return cruise setpoint, longitudinal planner ignores
    # On curves: will return physics speed, longitudinal planner uses it

    # Calculate safe speed using curvature_to_speed (physics-based)
    physics_safe_speed = curvature_to_speed(self._filtered_curvature)
    base_target = min(self._v_cruise_setpoint, physics_safe_speed)
    # Expose a "pre-cap" baseline so central arbitration can detect double-capping
    try:
      self._pre_cap_target_speed = float(base_target)
    except Exception:
      self._pre_cap_target_speed = float(base_target)

    # CONSENSUS FIX: Physics-based boost using lateral acceleration, not cruise setpoint
    # Calculate actual lateral acceleration from current curvature
    lateral_accel = abs(self._current_lat_acc)  # Already calculated as curvature * v_ego^2

    # Smooth boost factor using sigmoid to avoid hard switching
    boost_center = float(self._apex_boost_center)
    boost_width = max(1e-3, float(self._apex_boost_width))
    boost_amp = max(0.0, float(self._apex_boost_factor))

    # Sigmoid function: smoothly transitions based on lateral acceleration
    boost_factor = 1.0 + boost_amp / (1 + np.exp(-(lateral_accel - boost_center) / boost_width))

    # IMPROVED APEX DETECTION: Use actual geometric apexes, not crude ratio
    is_past_apex = False
    apply_boost = False

    # Check if we have detected apexes and are past one
    if self._apex_indices and len(self._apex_indices) > 0:
      # Vehicle is always at index 0, apexes are ahead in trajectory
      # Estimate meters per index based on typical trajectory spacing (about 1-2m)
      # T_IDXS gives us time stamps, convert to distance using current speed
      meters_per_index = float(self._apex_meters_per_index)

      # Find the nearest apex
      nearest_apex_idx = self._apex_indices[0]

      # Check if we've passed this apex (index would be negative in vehicle frame)
      # Since vehicle is at 0 and trajectory extends ahead, an apex at index 5
      # means it's 5*meters_per_index ahead. As we move, this decreases.
      # We track this with hysteresis to avoid re-triggering

      current_time = time.time()

      # Convert signed exit offset into a dynamic trigger index.
      # Lower values (negative) move acceleration onset earlier; higher values delay it.
      idx_per_second = float(self._v_ego) / max(0.1, meters_per_index)
      apex_exit_offset_s = float(getattr(self, '_apex_exit_phase_offset_s', 0.0))
      base_apex_idx = int(self._apex_near_index)
      trigger_apex_idx = int(round(base_apex_idx - apex_exit_offset_s * idx_per_second))
      trigger_apex_idx = int(clip(trigger_apex_idx, 1, 50))
      self._apex_trigger_idx = int(trigger_apex_idx)

      # Simple heuristic: if apex is in first trigger indices, we're very close or past it
      if nearest_apex_idx < trigger_apex_idx:
        # Check hysteresis - don't re-trigger same apex within 2 seconds
        if current_time - self._last_apex_passed_time > float(self._apex_hysteresis_time):
          is_past_apex = True
          self._last_apex_passed_time = current_time
          self._distance_past_apex = max(0.0, (trigger_apex_idx - nearest_apex_idx) * meters_per_index)
        else:
          # Still in boost window from previous detection
          is_past_apex = True
          self._distance_past_apex += self._v_ego * 0.05  # Update distance (20Hz update rate)

      # Apply boost if we're 0-50m past apex and in a real curve
      if is_past_apex and self._distance_past_apex < float(self._apex_boost_distance):
        apply_boost = True
      self._apex_exit_ready = bool(is_past_apex)

    if apply_boost and lateral_accel > float(self._apex_boost_min_lat_accel):  # Only boost if actually in a curve
      # Apply physics-based boost for acceleration out of apex
      # This creates the desired "kick" feeling without referencing cruise setpoint
      target_speed = base_target * boost_factor

      # Clamp to reasonable physics limits, NOT cruise setpoint
      # Allow speed to naturally reach what physics permits
      max_physics_speed = curvature_to_speed(self._filtered_curvature * float(self._boost_safety_curvature_scale))
      target_speed = clip(target_speed, _MIN_V, max_physics_speed)

      # Clear deceleration state when past apex
      self._is_decelerating_for_curve = False
    else:
      # BEFORE APEX or ON STRAIGHT: Use base physics speed
      # Check if we should start decelerating early
      if self._lat_acc_overshoot_ahead and not self._is_decelerating_for_curve:
        # Mark that we've started anticipatory deceleration
        self._is_decelerating_for_curve = True

      # Use the physics-based calculation
      target_speed = base_target

      # If we're in anticipatory deceleration mode and haven't reached target yet
      # Do not apply extra reduction when occluded to avoid over-braking
      # Also moderate reduction when confidence is near threshold or trending down
      if (self._is_decelerating_for_curve and self._v_ego > base_target + 0.5
          and self._occlusion_state.vision_good):
        conf = self._occlusion_state.smoothed_confidence
        good = float(self._occlusion_state.good_threshold)
        near_thresh = conf < (good + 0.02)
        now = time.time()
        dabs = getattr(self, '_abs_curvature_rate', 0.0)
        flattening_or_easing = (dabs <= 1e-5) or getattr(self, '_is_easing', False)
        # Confidence-independent moderation near apex: if curvature growth is small or negative, freeze
        if (flattening_or_easing) or (near_thresh and (flattening_or_easing or dabs <= 0.0)):
          # Freeze extra anticipation near threshold to avoid digging deeper
          target_speed = base_target
        else:
          # Apply reduction but cap the budget when confidence is degrading
          reduction_factor = clip(float(self._anticipation_target_reduction), 0.9, 1.0)
          proposed = base_target * reduction_factor
          # Global cap on anticipatory reduction depth relative to base
          cap_min = base_target - float(getattr(self, '_anticipation_max_reduction_mps', 2.0))
          proposed = max(proposed, cap_min)
          # Budget window: 0.8s while confidence trending downward
          if conf < self._prev_smoothed_conf - 1e-3:
            if self._anticipation_budget_window_start == 0.0 or (now - self._anticipation_budget_window_start) > 0.8:
              self._anticipation_budget_window_start = now
              self._cum_anticipation_reduction = 0.0
              self._last_high_conf_target_speed = base_target
            # Remaining budget in m/s
            remaining = max(0.0, 2.0 - self._cum_anticipation_reduction)
            # Limit additional reduction
            allowed_target = max(base_target - remaining, proposed)
            actual_reduction = max(0.0, base_target - allowed_target)
            self._cum_anticipation_reduction += actual_reduction
            target_speed = allowed_target
          else:
            target_speed = proposed
        self._prev_smoothed_conf = conf

      target_speed = clip(target_speed, _MIN_V, self._v_cruise_setpoint)

    # Post-reacquisition nudge: slightly bias toward physics base for faster convergence
    try:
      now = time.time()
    except Exception:
      now = 0.0
    if self._occlusion_state.vision_good and now < getattr(self, '_fast_reacq_until', 0.0):
      target_speed = min(self._v_cruise_setpoint, max(target_speed, base_target * 1.02))

    # Note: occlusion barriers and onset handling are applied centrally in _update_solution.
    # Avoid duplicating those effects here to prevent double-clamping.

    return float(target_speed)

  def _get_last_gps(self) -> tuple[float, float] | None:
    try:
      raw = self._mem_params.get('LastGPSPosition') or self._params.get('LastGPSPosition')
      if not raw:
        return None
      obj = json.loads(raw if isinstance(raw, str) else raw.decode('utf-8'))
      lat = float(obj.get('latitude', 0.0))
      lon = float(obj.get('longitude', 0.0))
      if lat == 0.0 and lon == 0.0:
        return None
      return (lat, lon)
    except Exception:
      return None

  def _load_map_curvatures(self) -> list[tuple[float, float, float]]:
    """Return list of (lat, lon, curvature) from mapd Params, decimated to ~80 points."""
    now = time.time()
    # refresh at most 5 Hz
    if (now - self._map_curv_last_ts) < 0.2 and self._map_curv_cache:
      return self._map_curv_cache
    try:
      raw = self._mem_params.get('MapCurvatures') or self._params.get('MapCurvatures')
      if not raw:
        self._map_curv_cache = []
        self._map_curv_last_ts = now
        return []
      s = raw if isinstance(raw, str) else raw.decode('utf-8')
      if s == self._map_curv_cache_raw:
        self._map_curv_last_ts = now
        return self._map_curv_cache
      arr = json.loads(s)
      pts = []
      for it in arr:
        try:
          lat = float(it.get('latitude', it.get('lat', 0.0)))
          lon = float(it.get('longitude', it.get('lon', 0.0)))
          k = float(it.get('curvature', 0.0))
          pts.append((lat, lon, max(0.0, k)))
        except Exception:
          continue
      # decimate to <= 80 samples to keep things cheap
      n = len(pts)
      if n > 80:
        step = max(1, n // 80)
        pts = pts[::step]
      self._map_curv_cache_raw = s
      self._map_curv_cache = pts
      self._map_curv_last_ts = now
      return pts
    except Exception:
      self._map_curv_cache = []
      self._map_curv_last_ts = now
      return []

  def _clear_curve_preview(self) -> None:
    self._curve_preview_valid = False
    self._curve_preview_distance_m = 0.0
    self._curve_preview_time_to_s = 0.0
    self._curve_preview_kappa_max = 0.0
    self._curve_preview_direction = 0
    self._curve_preview_severity = 0
    self._curve_preview_points = []

  def _update_curve_preview_from_map(self, *, gps_lat: float, gps_lon: float, pts: list[tuple[float, float, float]], i0: int) -> None:
    """Update HUD curve preview from mapd curvature samples.

    This is intentionally display-oriented: the HUD must not run its own curve math.
    """
    now = time.time()
    cache_raw = getattr(self, '_map_curv_cache_raw', None)
    last_raw = getattr(self, '_curve_preview_last_cache_raw', None)
    last_latlon = getattr(self, '_curve_preview_last_latlon', None)

    moved_far = True
    try:
      if last_latlon is not None:
        moved_far = _haversine_m(float(gps_lat), float(gps_lon), float(last_latlon[0]), float(last_latlon[1])) > 5.0
    except Exception:
      moved_far = True

    # Recompute at most 5 Hz unless map changed or ego moved materially.
    if (now - float(getattr(self, '_curve_preview_last_ts', 0.0) or 0.0)) < 0.2 and (cache_raw == last_raw) and (not moved_far):
      # Still refresh time-to-curve using latest speed.
      try:
        self._curve_preview_time_to_s = float(self._curve_preview_distance_m) / max(0.1, float(self._v_ego))
      except Exception:
        pass
      return

    # Default to invalid; set valid only when we can build a sane preview.
    self._clear_curve_preview()

    # Build a forward window scaled to ~10s lookahead at current speed.
    PREVIEW_S_MAX_M = max(200.0, min(float(self._v_ego) * 10.0, 600.0))
    MAX_POINTS = 48
    KAPPA_MIN = 1.0e-3  # 1/m, ~1000 m radius (detect gentler curves)
    RUN = 2             # consecutive samples to start/end a curve

    try:
      i0 = int(max(0, min(len(pts) - 1, i0)))
    except Exception:
      i0 = 0

    lat_ref = float(pts[i0][0])
    lon_ref = float(pts[i0][1])
    w_lat: list[float] = [lat_ref]
    w_lon: list[float] = [lon_ref]
    w_k: list[float] = [max(0.0, float(pts[i0][2]))]
    s_pts: list[float] = [0.0]

    s = 0.0
    # Hard cap on iterations to keep this bounded even if points are dense.
    for j in range(i0, min(len(pts) - 1, i0 + 120)):
      ds = _haversine_m(float(pts[j][0]), float(pts[j][1]), float(pts[j+1][0]), float(pts[j+1][1]))
      if not (ds > 0.05 and math.isfinite(ds)):
        continue
      s += float(ds)
      w_lat.append(float(pts[j+1][0]))
      w_lon.append(float(pts[j+1][1]))
      w_k.append(max(0.0, float(pts[j+1][2])))
      s_pts.append(float(s))
      if s >= PREVIEW_S_MAX_M:
        break

    if len(w_lat) < 4:
      return

    # Convert to local EN (east, north) and find a stable initial tangent.
    en: list[tuple[float, float]] = []
    for la, lo in zip(w_lat, w_lon, strict=False):
      en.append(_xy_from_latlon_m(float(la), float(lo), lat_ref, lon_ref))

    fdx = fdy = 0.0
    for idx in range(1, len(en)):
      dx, dy = float(en[idx][0]), float(en[idx][1])
      if math.hypot(dx, dy) > 1.0:
        fdx, fdy = dx, dy
        break
    norm = math.hypot(fdx, fdy)
    if not (norm > 1e-3 and math.isfinite(norm)):
      return
    fx, fy = fdx / norm, fdy / norm
    lx, ly = -fy, fx

    # Rotate into ego-local (x forward, y left).
    fwd_left: list[tuple[float, float]] = []
    for (xe, yn) in en:
      x_fwd = float(xe) * fx + float(yn) * fy
      y_left = float(xe) * lx + float(yn) * ly
      fwd_left.append((x_fwd, y_left))

    # Find the next curve region by curvature threshold persistence.
    start_idx = None
    consec = 0
    for idx, k in enumerate(w_k):
      if float(k) >= KAPPA_MIN:
        consec += 1
      else:
        consec = 0
      if consec >= RUN:
        start_idx = idx - (RUN - 1)
        break
    if start_idx is None:
      # No curve detected — emit the full road polyline with zero metadata so
      # the HUD has points pre-cached for an instant transition when a curve
      # does appear.  The HUD gates visibility on kappa_max, so this won't show.
      pts_out = fwd_left
      if len(pts_out) > MAX_POINTS:
        step = float(len(pts_out) - 1) / float(MAX_POINTS - 1)
        idxs = [int(round(i * step)) for i in range(MAX_POINTS)]
        uniq, last = [], -1
        for ii in idxs:
          ii = max(0, min(len(pts_out) - 1, int(ii)))
          if ii != last:
            uniq.append(ii)
            last = ii
        pts_out = [pts_out[ii] for ii in uniq]
      try:
        self._curve_preview_valid = True
        self._curve_preview_distance_m = 0.0
        self._curve_preview_time_to_s = 0.0
        self._curve_preview_kappa_max = 0.0
        self._curve_preview_direction = 0
        self._curve_preview_severity = 0
        self._curve_preview_points = [(float(x), float(y)) for (x, y) in pts_out]
        self._curve_preview_last_ts = float(now)
        self._curve_preview_last_cache_raw = cache_raw
        self._curve_preview_last_latlon = (float(gps_lat), float(gps_lon))
      except Exception:
        self._clear_curve_preview()
      return

    end_idx = len(w_k) - 1
    consec_below = 0
    for idx in range(int(start_idx), len(w_k)):
      if float(w_k[idx]) < KAPPA_MIN:
        consec_below += 1
      else:
        consec_below = 0
      if consec_below >= RUN:
        end_idx = max(int(start_idx), idx - RUN)
        break

    # Direction from polyline turning (cross product sign).
    cross_sum = 0.0
    for idx in range(int(start_idx), max(int(start_idx), int(end_idx) - 2)):
      x1, y1 = fwd_left[idx]
      x2, y2 = fwd_left[idx + 1]
      x3, y3 = fwd_left[idx + 2]
      v1x, v1y = (x2 - x1), (y2 - y1)
      v2x, v2y = (x3 - x2), (y3 - y2)
      cross_sum += (v1x * v2y - v1y * v2x)
    if cross_sum > 1e-3:
      direction = 1  # left
    elif cross_sum < -1e-3:
      direction = 2  # right
    else:
      direction = 0  # unknown

    # Peak curvature (abs) within curve region for simple display gating.
    try:
      kappa_max = 0.0
      for idx in range(int(start_idx), int(end_idx) + 1):
        kappa_max = max(kappa_max, float(w_k[idx]))
      if not (kappa_max > 0.0 and math.isfinite(kappa_max)):
        kappa_max = 0.0
    except Exception:
      kappa_max = 0.0

    # Severity from min safe speed within the curve region.
    try:
      vs_min = float('inf')
      for idx in range(int(start_idx), int(end_idx) + 1):
        vs_min = min(vs_min, float(curvature_to_speed(float(w_k[idx]))))
      if not math.isfinite(vs_min):
        severity = 0
      elif vs_min < 12.0:
        severity = 3  # tight
      elif vs_min < 20.0:
        severity = 2  # medium
      else:
        severity = 1  # gentle
    except Exception:
      severity = 0

    # Decimate points to <= MAX_POINTS, preserving endpoints.
    pts_out = fwd_left
    if len(pts_out) > MAX_POINTS:
      step = float(len(pts_out) - 1) / float(MAX_POINTS - 1)
      idxs = []
      for i in range(MAX_POINTS):
        idxs.append(int(round(i * step)))
      # ensure monotonic unique indices
      uniq = []
      last = -1
      for ii in idxs:
        ii = max(0, min(len(pts_out) - 1, int(ii)))
        if ii != last:
          uniq.append(ii)
          last = ii
      pts_out = [pts_out[ii] for ii in uniq]

    # Publish preview fields (used by HUD only).
    try:
      self._curve_preview_valid = True
      self._curve_preview_distance_m = float(s_pts[int(start_idx)])
      self._curve_preview_time_to_s = float(self._curve_preview_distance_m) / max(0.1, float(self._v_ego))
      self._curve_preview_kappa_max = float(kappa_max)
      self._curve_preview_direction = int(direction)
      self._curve_preview_severity = int(severity)
      self._curve_preview_points = [(float(x), float(y)) for (x, y) in pts_out]
      self._curve_preview_last_ts = float(now)
      self._curve_preview_last_cache_raw = cache_raw
      self._curve_preview_last_latlon = (float(gps_lat), float(gps_lon))
    except Exception:
      self._clear_curve_preview()

  def _map_tail_cap(self) -> tuple[float | None, float, float]:
    """
    Compute a comfort-reachable cap on current speed from map curvature tail.

    Returns (v_cap_mps|None, start_distance_m, coverage_frac)
    """
    self._map_tail_compute_reason = "unknown"
    gps = self._get_last_gps()
    if gps is None:
      self._map_tail_compute_reason = "no_gps"
      self._clear_curve_preview()
      return (None, 0.0, 0.0)
    lat0, lon0 = gps
    pts = self._load_map_curvatures()
    if len(pts) < 3:
      self._map_tail_compute_reason = "no_map_curvatures"
      self._clear_curve_preview()
      return (None, 0.0, 0.0)

    # Find nearest index to ego
    try:
      dists = [ _haversine_m(lat0, lon0, p[0], p[1]) for p in pts ]
      i0 = int(np.argmin(dists))
    except Exception:
      i0 = 0

    # Build forward along-track distances from nearest point
    s_list = [0.0]
    for i in range(i0, len(pts)-1):
      s_list.append(s_list[-1] + _haversine_m(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]))
    # Remove duplicate 0 entry alignment
    if len(s_list) > 0:
      s_list = s_list[1:]
    k_list = [max(0.0, pts[j][2]) for j in range(i0+1, len(pts))]
    if not s_list or not k_list:
      self._map_tail_compute_reason = "insufficient_map_points"
      self._clear_curve_preview()
      return (None, 0.0, 0.0)

    # Update HUD preview from the same map lookahead inputs VTSC already uses.
    try:
      self._update_curve_preview_from_map(gps_lat=float(lat0), gps_lon=float(lon0), pts=pts, i0=i0)
    except Exception:
      # Never let preview failures affect longitudinal behavior.
      self._clear_curve_preview()

    # Limit horizon to ~800 m
    S_MAX = 800.0
    L = len(s_list)
    cut = L
    for idx, s in enumerate(s_list):
      if s >= S_MAX:
        cut = idx+1
        break
    s_list = s_list[:cut]
    k_list = k_list[:cut]

    # Compute vsafe from curvature
    vsafe = [ curvature_to_speed(k) for k in k_list ]

    # Determine start distance: visible horizon + margin
    #
    # NOTE:
    # Map tail is primarily meant to cover *beyond* what vision can see.
    # When vision confidence is extremely low (SEVERE/LOST), we may not have reliable near-horizon
    # curvature from the model either. In that case, allow map curvature to influence the cap
    # immediately (starting at the margin distance) so short, sharp off-ramp curves inside the
    # usual "visible horizon" aren't ignored.
    try:
      vs = getattr(self._occlusion_state, 'vision_status', VisionStatus.FULL_VISIBILITY)
      severe_vision = bool(int(vs) >= int(VisionStatus.SEVERE_OCCLUSION))
    except Exception:
      severe_vision = False
    vis_margin = float(getattr(self, '_vis_margin_m', 10.0))
    if severe_vision:
      s_start = max(0.0, vis_margin)
    else:
      s_start = max(0.0, self._v_ego * float(getattr(self, '_vis_horizon_s', 1.4)) + vis_margin)
    # Planning decel: use half of comfort decel for the reachable-cap so braking
    # begins earlier and more gently, instead of last-second emergency braking.
    a_comf_full = float(max(0.1, getattr(self, '_max_decel', 3.5)))
    a_plan = a_comf_full * 0.5

    # Reachable cap: start from cruise setpoint, not v_ego.  Starting from v_ego
    # creates a one-way ratchet that pins the cap at current speed when exiting a
    # curve, preventing acceleration even when the next curve is far ahead.
    v_now = float(self._v_ego)
    v_cap = float(self._v_cruise_setpoint)
    any_future = False
    for vi, di in zip(vsafe, s_list, strict=False):
      if di < s_start:
        continue
      any_future = True
      d = max(0.0, di - s_start)
      try:
        v_allow = math.sqrt(max(0.0, vi*vi + 2.0 * a_plan * d))
      except Exception:
        v_allow = v_now
      v_cap = min(v_cap, v_allow)
    if not any_future:
      self._map_tail_compute_reason = "no_future_points_beyond_start"
      return (None, s_start, float(min(1.0, s_list[-1] / max(1e-3, s_start))))

    coverage = float(min(1.0, (s_list[-1] - s_start) / max(1e-3, (S_MAX - s_start)))) if s_list[-1] > s_start else 0.0
    self._map_tail_compute_reason = "cap_available"
    return (max(0.0, v_cap), s_start, coverage)

  def update(self, sm, enabled, v_ego, a_ego, v_cruise_setpoint, v_cruise_cluster_setpoint=None):
    self._op_enabled = enabled
    # Be defensive about SM shape in offline/testing environments
    try:
      cs = sm['carState']
    except Exception:
      try:
        cs = getattr(sm, 'carState', None)
        if cs is None:
          data = getattr(sm, '_data', None)
          if isinstance(data, dict):
            cs = data.get('carState', None)
      except Exception:
        cs = None
    self._gas_pressed = bool(getattr(cs, 'gasPressed', False))
    try:
      self._steering_angle_deg = float(getattr(cs, 'steeringAngleDeg', 0.0))
    except Exception:
      self._steering_angle_deg = 0.0
    self._v_ego = v_ego
    self._a_ego = a_ego
    # Use cluster speed as source of truth if available, otherwise fall back to v_cruise
    prev_limit = getattr(self, '_v_cruise_setpoint', 0.0)
    self._v_cruise_setpoint = v_cruise_cluster_setpoint if v_cruise_cluster_setpoint is not None else v_cruise_setpoint
    try:
      if self._v_cruise_setpoint < prev_limit - 0.2:
        self._limit_step_until = time.time() + 1.0
        self._suppress_raise_due_to_limit = True
    except Exception:
      pass

    # Initialize advanced controller state on first run or when speed changes significantly
    if (self._prev_target_speed == 0.0 or
        abs(self._prev_target_speed - v_ego) > 5.0):
      self._prev_target_speed = v_ego
      self._current_accel = a_ego

    self._update_params()
    self._update_calculations(sm)
    self._state_transition()
    self._update_solution()
    # Emit compact debug snapshot if enabled and rate allows
    try:
      now_s = float(getattr(time, 'monotonic', time.time)())
    except Exception:
      now_s = time.time()
    if self._should_emit_debug(now_s):
      try:
        snap = self.snapshot_debug_state()
        if snap:
          cloudlog.debug("VTSCDBG %s", json.dumps(snap, separators=(',', ':')))
          if bool(getattr(self, '_dbg_write_file', False)):
            self._append_snapshot_to_file(snap, now_s)
      except Exception:
        pass
