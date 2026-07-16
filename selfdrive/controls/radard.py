#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import capnp
from cereal import messaging, log, car, custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import (
  LeadResponseTuningConfig,
  read_lead_response_tuning_config,
)

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame

MODEL_LEAD_TRACK_ID_START = -1001
MODEL_LEAD_TRACK_MAX_MISSES = 6
MODEL_LEAD_TRACK_MAX_COUNT = 4
MODEL_LEAD_PARAM_REFRESH_DT_S = 1.0
MODEL_LEAD_ASSOC_Y_GATE_M = 3.0
MODEL_LEAD_ASSOC_VREL_GATE_MPS = 8.0
MODEL_LEAD_DUPLICATE_PATH_GATE_M = 0.8
MODEL_LEAD_DUPLICATE_VREL_GATE_MPS = 2.0
MODEL_LEAD_DUPLICATE_DREL_GATE_M = 35.0
MODEL_LEAD_DUPLICATE_CLOSER_KEEP_SEPARATE_M = 12.0
MODEL_LEAD_DUPLICATE_CLOSING_KEEP_SEPARATE_MPS = 2.5
MODEL_LEAD_SAME_SLOT_RECOVER_DREL_GATE_M = 80.0
MODEL_LEAD_CLOSE_INNOVATION_M = 2.5
MODEL_LEAD_FAST_CLOSE_TAU_S = 0.12
MODEL_LEAD_STRONG_CLOSING_MPS = 2.5
MODEL_LEAD_NOISE_CLOSE_SLEW_MPS = 1.0
# Urgency blend degeneracy guards: both endpoints are independently live-tunable,
# so a degenerate span must collapse to u=0 (current behavior), never sign-flip.
MODEL_LEAD_BLEND_MIN_SPAN = 1e-2
MODEL_LEAD_BLEND_TTC_MIN_CLOSING_MPS = 0.3
MODEL_LEAD_VREL_TAU_S = 0.40
MODEL_LEAD_FAST_VREL_TAU_S = 0.16
MODEL_LEAD_LAT_TAU_S = 0.45
MODEL_LEAD_ACCEL_TAU_S = 0.60
MODEL_LEAD_PROB_TAU_S = 0.80
# Opening governor: published-closing TTC below this vetoes any relax — near a
# genuinely-closing lead the pipeline's pessimism stands, regardless of what
# the position window claims.
OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S = 6.0
# A held opening correction may bridge mild model-velocity pessimism, but a
# genuinely fast current closure always wins independently of live veto knobs.
OPENING_GOVERNOR_HARD_CLOSING_MPS = MODEL_LEAD_STRONG_CLOSING_MPS
# A position-only CD9 latch is considered weak only while its windowed velocity
# evidence remains below this fixed ceiling. This classification is deliberately
# not live-tunable: permissive diagnostic values must never turn a sustained
# closure into an opening-governor comfort case.
OPENING_GOVERNOR_WEAK_CLOSING_MAX_MPS = 1.0
# A held CD9 clamp may reconcile to calm current velocity only after a robust
# longer position horizon independently agrees the closure is mild. The
# original braking-lead road case closed 2.4-6 m/s on position; today's steady
# lead false-closing plateaus stayed at or below ~1.1 m/s on this horizon.
CLOSING_GOVERNOR_RECOVERY_POSITION_WINDOW_S = 1.25
CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS = 1.25
CLOSING_GOVERNOR_RECOVERY_MIN_POSITION_SPAN_S = 1.0
CLOSING_GOVERNOR_RECOVERY_MIN_POSITION_SAMPLES = 16
CLOSING_GOVERNOR_RECOVERY_MAX_SAMPLE_GAP_S = 0.075
CLOSING_GOVERNOR_RECOVERY_MIN_RAW_TTC_S = 12.0
MODEL_LEAD_CENTER_PATH_GATE_M = 2.6
MODEL_LEAD_CUTIN_VLAT_MPS = 0.7
LEAD_TRACK_PROB_DROPOUT_MIN_SPEED_MPS = 4.0
LEAD_TRACK_PROB_DROPOUT_MIN_DREL_M = 1.0
LEAD_TRACK_PROB_DROPOUT_NEAR_DREL_M = 25.0
LEAD_TRACK_PROB_DROPOUT_NEAR_HEADWAY_S = 1.6
LEAD_TRACK_PROB_DROPOUT_CENTER_Y_ABS_M = 1.5
LEAD_TRACK_PROB_DROPOUT_PULLING_AWAY_MAX_MPS = 0.5
LEAD_TRACK_PROB_DROPOUT_URGENT_CLOSING_MPS = 1.0
LEAD_TRACK_PROB_DROPOUT_URGENT_TTC_S = 8.0
# FCW-corroboration vote history depth (frames). The live-tunable vote window
# (ModelLeadFcwCorrobWindow) is clamped to this; history is seeded all-True so
# a freshly acquired track (a genuine sudden cut-in) is never suppressed.
MODEL_LEAD_FCW_CORROB_HIST_LEN = 8
# Evidence-only steady-lead parity proof. This path never reshapes RadarState
# kinematics: it publishes a bounded candidate for the planner to consume on a
# private working copy after applying the exact configured-gap gate.
STEADY_PARITY_WINDOW_S = 2.0
STEADY_PARITY_MIN_SAMPLES = 24
STEADY_PARITY_MIN_SPAN_S = 1.6
STEADY_PARITY_MAX_SAMPLE_GAP_S = 0.075
STEADY_PARITY_MIN_PAIR_SPAN_S = 0.5
STEADY_PARITY_MIN_POSITION_SLOPE_MPS = -0.25
STEADY_PARITY_MAX_POSITION_SLOPE_MPS = 0.75
STEADY_PARITY_ALEAD_VETO_MPS2 = 0.20
STEADY_PARITY_MAX_RAW_CLOSING_MPS = 1.5
STEADY_PARITY_MIN_RAW_TTC_S = 12.0
STEADY_PARITY_MAX_DPATH_M = 1.5
STEADY_PARITY_MAX_VLAT_MPS = 0.7
STEADY_PARITY_VLAT_OFFCENTER_DPATH_M = 1.25
STEADY_PARITY_MIN_MODEL_PROB = 0.60
STEADY_PARITY_IDENTITY_DREL_JUMP_M = 4.0
# Raw vRel alone is not enough to attest a genuine threat after a private
# steady-parity correction. A same-frame range step must independently confirm
# at least this much closure before the planner may bypass comfort carryover.
STEADY_PARITY_CURRENT_THREAT_MIN_POSITION_CLOSING_MPS = 1.25
# Independent proof for the MPC's private lead-acceleration correlation
# amplifier. Unlike steady parity, a velocity-only spike does not deassert this
# proof: that spike is precisely the measurement the amplifier would otherwise
# treat as corroboration. The dense raw-position slope, current published
# aLeadK, and same-frame threat gates must all remain calm.
ACCEL_CORR_CALM_POSITION_WINDOW_S = 2.0
ACCEL_CORR_CALM_POSITION_MIN_SAMPLES = 24
ACCEL_CORR_CALM_POSITION_MIN_SPAN_S = 1.6
ACCEL_CORR_CALM_POSITION_MAX_SAMPLE_GAP_S = 0.075
ACCEL_CORR_CALM_POSITION_MIN_PAIR_SPAN_S = 0.5
ACCEL_CORR_CALM_POSITION_MIN_SLOPE_MPS = -1.25
ACCEL_CORR_CALM_POSITION_MAX_SLOPE_MPS = 0.75
ACCEL_CORR_CALM_POSITION_MAX_ALEAD_ABS_MPS2 = 0.20
# The intended false-onset route frame reached raw aLead=-0.3506 while the
# known genuine CD3 truth-deficit reports -0.48..-0.54. A -0.20 gate would
# erase the fix; this fixed hard gate preserves that separation and ensures a
# meaningful genuine raw-decel onset clears veto authority in the same frame.
ACCEL_CORR_CALM_POSITION_RAW_ALEAD_HARD_VETO_MPS2 = 0.40
ACCEL_CORR_CALM_POSITION_MAX_PUBLISHED_CLOSING_MPS = 2.5
ACCEL_CORR_CALM_POSITION_MIN_PUBLISHED_TTC_S = 12.0

# Version zero is the Cap'n Proto default and therefore means no producer-side
# replay contract was emitted.
_REPLAY_INPUTS_VERSION = 2


def _finite_float(value: Any, default: float = 0.0) -> float:
  try:
    ret = float(value)
    return ret if math.isfinite(ret) else float(default)
  except Exception:
    return float(default)


def _ema_alpha(dt_s: float, tau_s: float) -> float:
  if tau_s <= 1e-3:
    return 1.0
  return float(1.0 - math.exp(-max(0.0, dt_s) / max(1e-3, tau_s)))


@dataclass
class ModelLeadTrack:
  identifier: int
  dRel: float
  yRel: float
  vRel: float
  vLead: float
  vLeadK: float
  aLeadK: float
  aLeadTau: float
  modelProb: float
  dPath: float
  vLat: float
  last_t: float
  age: int = 1
  missed: int = 0
  closer_confirm_frames: int = 0
  # Consecutive frames the raw measurement has been beyond the innovation gate
  # on the OPENING side; corroborates that a much-too-close internal state
  # (e.g. a wrongly adopted inward outlier) should be healed.
  opening_confirm_frames: int = 0
  last_slot: int = -1
  # Effective closing-side dRel filter delay of the most recent update; caps the
  # publish-side lag compensation so a fast-adopting regime is not over-corrected.
  drel_lag_s: float = 0.0
  # FCW corroboration: per-frame votes on whether the raw measurement agrees
  # with the filtered dRel (raw NOT materially farther than the filter). A
  # phantom-collapsed state (filtered dRel far below what the model keeps
  # measuring) loses the vote and is published with fcwSuppressed=True so the
  # planner cannot escalate it into crash_cnt/FCW. Seeded all-True: new tracks
  # (genuine sudden cut-ins) and the missed-frame hold keep legacy FCW timing.
  fcw_agree_hist: deque = field(default_factory=lambda: deque([True] * MODEL_LEAD_FCW_CORROB_HIST_LEN,
                                                              maxlen=MODEL_LEAD_FCW_CORROB_HIST_LEN))
  fcw_suppressed: bool = False
  # CD4 opening-only published-dRel step guard (publish-only, mirrors lag-comp):
  # the last value actually PUBLISHED for this track. None until the track's
  # first publish so the guard never binds on frame 1 (a fresh cut-in publishes
  # its true close dRel unclamped). The guard also caches which frame it last
  # evaluated so a duplicate slot's second same-frame get_RadarState call reuses
  # the first call's result instead of comparing against (and re-clamping) the
  # value that same-frame call just wrote.
  last_published_drel: float | None = None
  _step_guard_eval_t: float | None = None
  _step_guard_last_out: float | None = None
  # CD8 far-range stopped-traffic vLead optimism clamp state. Recent RAW model
  # vLead samples (v_ego + raw vRel) to detect a SUSTAINED monotonic decline (a
  # stopping/decelerating lead), and a (time, internal-dRel) history so the
  # position-derived vLead d(dRel)/dt + v_ego can be estimated over the WHOLE
  # confirm window (endpoint slope) rather than a single fragile frame - a
  # single-frame internal-dRel snap (filter innovation) cannot produce a garbage
  # velocity that way. Publish-time clamp only; internal EMA state is untouched
  # and the clamp only ever makes the published vLead SLOWER (more urgent).
  raw_vlead_hist: deque = field(default_factory=lambda: deque(maxlen=8))
  drel_hist: deque = field(default_factory=lambda: deque(maxlen=8))
  # CD9 corroborated-closing governor (road 205-6 Event B) evidence window:
  # (t, raw_drel, raw_vrel, raw_alead) per measured frame. Windowed means over
  # the RAW streams detect a sustained closure the slow publish-side EMAs are
  # lagging; heavy-tail single-frame dRel outliers cannot dominate the
  # k-endpoint means. governor_closing_mps caches the least-aggressive
  # corroborated closure for the publish clamp while the latch holds.
  closing_evidence: deque = field(default_factory=lambda: deque(maxlen=64))
  governor_hold_until_t: float = -1.0
  governor_closing_mps: float = 0.0
  governor_active: bool = False
  governor_reason: str = "inactive"
  governor_threat_corroborated: bool = False
  governor_calm_recovery_mode: bool = False
  governor_calm_recovery_applied: bool = False
  governor_recovery_position_closing_mps: float | None = None
  governor_recovery_vrel_floor_mps: float | None = None
  # Opening governor (CD9's mirror): publish-time one-directional vRel relax
  # floor while the raw position window PROVES sustained opening that the
  # closing-biased publish pipeline is contradicting. The separate hold state
  # bridges short proof-window dropouts without feeding the published result
  # back into its own deadline. None = not armed.
  opening_relax_vrel: float | None = None
  opening_relax_hold_vrel: float | None = None
  opening_relax_hold_until_t: float = -1.0
  opening_last_raw_proof_t: float = -1.0
  opening_relax_held: bool = False
  # Longer raw-position history used only as a bridge veto. The 0.6 s CD9
  # evidence deque is destructively trimmed and can reverse on high-frequency
  # x noise; this independent horizon distinguishes that local reversal from a
  # genuinely closing gap. It never arms or refreshes the hold by itself.
  opening_position_evidence: deque = field(default_factory=lambda: deque(maxlen=64))
  opening_long_position_slope_mps: float | None = None
  opening_bridge_position_safe: bool = False
  # Samples at/before this identity/threat boundary cannot establish a new
  # opening hold. Closing evidence remains intact for CD9 safety; only the
  # less-urgent opening proof must be earned again on fresh frames.
  opening_evidence_epoch_t: float = -1.0
  # Independent 2 s raw-position history for the planner-only steady-parity
  # candidate. It must not reuse CD9's destructively trimmed 0.6 s deque.
  steady_parity_evidence: deque = field(default_factory=lambda: deque(maxlen=64))
  steady_parity_candidate_valid: bool = False
  steady_parity_position_slope_mps: float = 0.0
  steady_parity_vrel_floor_mps: float = 0.0
  steady_parity_held: bool = False
  steady_parity_hold_floor_mps: float | None = None
  steady_parity_hold_until_t: float = -1.0
  steady_parity_reason: str = "inactive"
  steady_parity_sample_count: int = 0
  steady_parity_window_span_s: float = 0.0
  steady_parity_max_sample_gap_s: float = 0.0
  # One-frame, current-measurement attestation for the planner's stateful
  # comfort bypass. This never changes the steady-parity proof/candidate state.
  steady_parity_current_threat: bool = False
  # Separate raw-position history for the private aLead correlation-amplifier
  # veto. It must survive the very velocity-only contradiction that clears
  # steady-parity correction authority.
  accel_corr_position_evidence: deque = field(default_factory=lambda: deque(maxlen=64))
  accel_corr_calm_position_dense_valid: bool = False
  accel_corr_calm_position_valid: bool = False
  accel_corr_calm_position_slope_mps: float = 0.0
  accel_corr_calm_position_reason: str = "inactive"
  accel_corr_calm_position_sample_count: int = 0
  accel_corr_calm_position_window_span_s: float = 0.0
  accel_corr_calm_position_max_sample_gap_s: float = 0.0

  @classmethod
  def from_lead_dict(cls, identifier: int, lead_dict: dict[str, Any], now: float, lead_slot: int) -> "ModelLeadTrack":
    return cls(
      identifier=identifier,
      dRel=_finite_float(lead_dict.get("dRel")),
      yRel=_finite_float(lead_dict.get("yRel")),
      vRel=_finite_float(lead_dict.get("vRel")),
      vLead=_finite_float(lead_dict.get("vLead")),
      vLeadK=_finite_float(lead_dict.get("vLeadK", lead_dict.get("vLead"))),
      aLeadK=_finite_float(lead_dict.get("aLeadK")),
      aLeadTau=_finite_float(lead_dict.get("aLeadTau"), 0.3),
      modelProb=_finite_float(lead_dict.get("modelProb")),
      dPath=_finite_float(lead_dict.get("dPath", lead_dict.get("yRel"))),
      vLat=_finite_float(lead_dict.get("vLat")),
      last_t=float(now),
      last_slot=int(lead_slot),
    )

  def predict_drel(self, now: float) -> float:
    dt_s = float(np.clip(float(now) - self.last_t, 0.0, 1.0))
    return float(max(0.0, self.dRel + self.vRel * dt_s))

  def _clear_opening_relax(self, *, rearm_after_t: float | None = None) -> None:
    """Clear every less-urgent opening-governor state atomically."""
    self.opening_relax_vrel = None
    self.opening_relax_hold_vrel = None
    self.opening_relax_hold_until_t = -1.0
    self.opening_last_raw_proof_t = -1.0
    self.opening_relax_held = False
    if rearm_after_t is not None:
      self.opening_evidence_epoch_t = max(self.opening_evidence_epoch_t, float(rearm_after_t))

  def _clear_steady_parity(self, reason: str, *, clear_history: bool = True) -> None:
    self.steady_parity_candidate_valid = False
    self.steady_parity_held = False
    self.steady_parity_hold_floor_mps = None
    self.steady_parity_hold_until_t = -1.0
    self.steady_parity_vrel_floor_mps = 0.0
    self.steady_parity_reason = str(reason)
    if clear_history:
      self.steady_parity_evidence.clear()
      self.steady_parity_position_slope_mps = 0.0
      self.steady_parity_sample_count = 0
      self.steady_parity_window_span_s = 0.0
      self.steady_parity_max_sample_gap_s = 0.0

  def _clear_accel_corr_calm_position(self, reason: str, *, clear_history: bool = True) -> None:
    self.accel_corr_calm_position_dense_valid = False
    self.accel_corr_calm_position_valid = False
    self.accel_corr_calm_position_reason = str(reason)
    if clear_history:
      self.accel_corr_position_evidence.clear()
      self.accel_corr_calm_position_slope_mps = 0.0
      self.accel_corr_calm_position_sample_count = 0
      self.accel_corr_calm_position_window_span_s = 0.0
      self.accel_corr_calm_position_max_sample_gap_s = 0.0

  def _update_accel_corr_calm_position(self, now: float, raw_drel: float, raw_vrel: float,
                                       raw_alead: float, raw_prob: float, raw_dpath: float,
                                       raw_vlat: float, v_ego: float) -> None:
    """Attest dense calm position evidence without changing lead kinematics.

    A noisy published-vLead derivative can make the private MPC correlation
    amplifier deepen a mild model decel into a false hard brake. This producer
    proof intentionally ignores that derivative and asks an independent 2 s
    raw-position history whether the same physical lead is actually closing.
    Any current published braking, identity/lateral/probability boundary, or
    dense closing slope clears the proof in the same frame. A sub-identity-gate
    range step remains visible in untouched public kinematics and cannot by
    itself authorize this positive-only skip of extra amplification.
    """
    self.accel_corr_calm_position_dense_valid = False
    self.accel_corr_calm_position_valid = False

    if self.accel_corr_position_evidence:
      prev_t, prev_drel = self.accel_corr_position_evidence[-1]
      dt_s = float(now) - float(prev_t)
      expected_step_m = float(raw_vrel) * max(0.0, dt_s)
      if (dt_s <= 1e-3 or dt_s > 0.25 or
          abs((float(raw_drel) - float(prev_drel)) - expected_step_m) > STEADY_PARITY_IDENTITY_DREL_JUMP_M):
        self._clear_accel_corr_calm_position("identity_boundary")

    raw_vlead = float(v_ego) + float(raw_vrel)
    near_threat = float(raw_drel) <= max(10.0, 0.55 * max(0.0, float(v_ego)))
    hard_reason = None
    if float(raw_alead) < -ACCEL_CORR_CALM_POSITION_RAW_ALEAD_HARD_VETO_MPS2:
      hard_reason = "raw_hard_braking"
    elif abs(float(self.aLeadK)) > ACCEL_CORR_CALM_POSITION_MAX_ALEAD_ABS_MPS2:
      hard_reason = "published_accel"
    elif raw_vlead < 0.0:
      hard_reason = "oncoming"
    elif near_threat:
      hard_reason = "near_threat"
    elif (abs(float(raw_dpath)) > STEADY_PARITY_MAX_DPATH_M or
          (abs(float(raw_dpath)) > STEADY_PARITY_VLAT_OFFCENTER_DPATH_M and
           abs(float(raw_vlat)) >= STEADY_PARITY_MAX_VLAT_MPS)):
      hard_reason = "lateral_ambiguity"
    elif float(raw_prob) < STEADY_PARITY_MIN_MODEL_PROB:
      hard_reason = "low_probability"
    if hard_reason is not None:
      self._clear_accel_corr_calm_position(hard_reason)
      return

    self.accel_corr_position_evidence.append((float(now), float(raw_drel)))
    while (self.accel_corr_position_evidence and
           (float(now) - self.accel_corr_position_evidence[0][0]) > ACCEL_CORR_CALM_POSITION_WINDOW_S):
      self.accel_corr_position_evidence.popleft()

    samples = list(self.accel_corr_position_evidence)
    span_s = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    gaps_s = [b[0] - a[0] for a, b in zip(samples, samples[1:], strict=False)]
    max_gap_s = max(gaps_s) if gaps_s else 0.0
    self.accel_corr_calm_position_sample_count = len(samples)
    self.accel_corr_calm_position_window_span_s = float(span_s)
    self.accel_corr_calm_position_max_sample_gap_s = float(max_gap_s)
    dense = bool(
      len(samples) >= ACCEL_CORR_CALM_POSITION_MIN_SAMPLES and
      span_s >= ACCEL_CORR_CALM_POSITION_MIN_SPAN_S and
      gaps_s and max_gap_s <= ACCEL_CORR_CALM_POSITION_MAX_SAMPLE_GAP_S
    )
    pair_slopes = [
      (later[1] - earlier[1]) / (later[0] - earlier[0])
      for i, earlier in enumerate(samples)
      for later in samples[i + 1:]
      if (later[0] - earlier[0]) >= ACCEL_CORR_CALM_POSITION_MIN_PAIR_SPAN_S
    ]
    position_slope = float(np.median(pair_slopes)) if pair_slopes else None
    if position_slope is not None:
      self.accel_corr_calm_position_slope_mps = position_slope

    slope_in_band = bool(
      position_slope is not None and
      ACCEL_CORR_CALM_POSITION_MIN_SLOPE_MPS <= position_slope <= ACCEL_CORR_CALM_POSITION_MAX_SLOPE_MPS
    )
    if dense and not slope_in_band:
      self._clear_accel_corr_calm_position("position_slope_veto")
      return
    if dense and slope_in_band:
      self.accel_corr_calm_position_dense_valid = True
      self.accel_corr_calm_position_reason = "position_proven"
      return
    self.accel_corr_calm_position_reason = "sparse_window"

  def _update_steady_parity(self, now: float, raw_drel: float, raw_vrel: float,
                            raw_alead: float, raw_prob: float, raw_dpath: float,
                            raw_vlat: float, v_ego: float,
                            cfg: LeadResponseTuningConfig) -> None:
    """Publish evidence for a planner-only steady-lead vRel correction.

    The proof is intentionally stricter and longer than the existing opening
    governor. A robust Theil-Sen slope over 2 s must agree the gap is near
    parity while every independently urgent current/window signal stays calm.
    No kinematic field on this ModelLeadTrack is changed here.
    """
    # One-frame telemetry only: never let an earlier threat survive a calm,
    # missing, or identity-only publication.
    self.steady_parity_current_threat = False
    trust_deficit = float(getattr(cfg, 'steady_parity_trust_deficit_mps', 99.0))
    if trust_deficit >= 99.0:
      self._clear_steady_parity("disabled")
      return

    # A same-track association may survive a large position discontinuity. It
    # is still an identity boundary for less-urgent evidence and must earn a
    # completely fresh 2 s epoch.
    recent_position_closing_mps: float | None = None
    if self.steady_parity_evidence:
      prev_t, prev_drel, *_ = self.steady_parity_evidence[-1]
      dt_s = float(now) - float(prev_t)
      expected_step_m = float(raw_vrel) * max(0.0, dt_s)
      if (dt_s <= 1e-3 or dt_s > 0.25 or
          abs((float(raw_drel) - float(prev_drel)) - expected_step_m) > STEADY_PARITY_IDENTITY_DREL_JUMP_M):
        self._clear_steady_parity("identity_boundary")
      elif dt_s <= STEADY_PARITY_MAX_SAMPLE_GAP_S:
        recent_position_closing_mps = max(
          0.0, -(float(raw_drel) - float(prev_drel)) / dt_s,
        )

    raw_closing_mps = max(0.0, -float(raw_vrel))
    raw_ttc_s = float(raw_drel) / max(raw_closing_mps, 0.1)
    raw_vlead = float(v_ego) + float(raw_vrel)
    near_threat = float(raw_drel) <= max(10.0, 0.55 * max(0.0, float(v_ego)))
    current_raw_braking = float(raw_alead) < -STEADY_PARITY_ALEAD_VETO_MPS2
    current_fast_close = raw_closing_mps >= STEADY_PARITY_MAX_RAW_CLOSING_MPS
    current_short_ttc = raw_closing_mps > 0.3 and raw_ttc_s <= STEADY_PARITY_MIN_RAW_TTC_S
    hard_reason = None
    if float(self.aLeadK) < -STEADY_PARITY_ALEAD_VETO_MPS2:
      hard_reason = "published_braking"
    elif current_short_ttc:
      hard_reason = "short_raw_ttc"
    elif raw_vlead < 0.0:
      hard_reason = "oncoming"
    elif near_threat:
      hard_reason = "near_threat"
    elif (abs(float(raw_dpath)) > STEADY_PARITY_MAX_DPATH_M or
          (abs(float(raw_dpath)) > STEADY_PARITY_VLAT_OFFCENTER_DPATH_M and
           abs(float(raw_vlat)) >= STEADY_PARITY_MAX_VLAT_MPS)):
      hard_reason = "lateral_ambiguity"
    elif float(raw_prob) < STEADY_PARITY_MIN_MODEL_PROB:
      hard_reason = "low_probability"
    if hard_reason is not None:
      self.steady_parity_current_threat = hard_reason in (
        "published_braking", "short_raw_ttc", "oncoming", "near_threat",
      )
      self._clear_steady_parity(hard_reason)
      return

    def append_and_measure_position_window() -> tuple[list[tuple], bool, float | None, float]:
      self.steady_parity_evidence.append((
        float(now), float(raw_drel), float(raw_vrel), float(raw_alead),
        float(raw_dpath), float(raw_vlat), float(raw_prob),
      ))
      while self.steady_parity_evidence and (float(now) - self.steady_parity_evidence[0][0]) > STEADY_PARITY_WINDOW_S:
        self.steady_parity_evidence.popleft()

      samples = list(self.steady_parity_evidence)
      span_s = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
      gaps_s = [b[0] - a[0] for a, b in zip(samples, samples[1:], strict=False)]
      max_gap_s = max(gaps_s) if gaps_s else 0.0
      self.steady_parity_sample_count = len(samples)
      self.steady_parity_window_span_s = float(span_s)
      self.steady_parity_max_sample_gap_s = float(max_gap_s)
      dense = bool(
        len(samples) >= STEADY_PARITY_MIN_SAMPLES and
        span_s >= STEADY_PARITY_MIN_SPAN_S and
        gaps_s and max_gap_s <= STEADY_PARITY_MAX_SAMPLE_GAP_S
      )

      pair_slopes = [
        (later[1] - earlier[1]) / (later[0] - earlier[0])
        for i, earlier in enumerate(samples)
        for later in samples[i + 1:]
        if (later[0] - earlier[0]) >= STEADY_PARITY_MIN_PAIR_SPAN_S
      ]
      position_slope = float(np.median(pair_slopes)) if pair_slopes else None
      if position_slope is not None:
        self.steady_parity_position_slope_mps = position_slope
      alead_mean = sum(sample[3] for sample in samples) / len(samples)
      return samples, dense, position_slope, float(alead_mean)

    # A far/high-TTC raw-vRel spike removes private correction immediately and
    # clears the complete proof epoch. It is attested as a genuine threat only
    # when raw braking or an independent same-frame range step corroborates it.
    # The rejected short proof-preservation experiment deliberately does not
    # survive in production: a calm frame must earn a fresh dense position proof.
    if current_fast_close:
      self.steady_parity_current_threat = bool(
        current_raw_braking or
        (recent_position_closing_mps is not None and
         recent_position_closing_mps > STEADY_PARITY_CURRENT_THREAT_MIN_POSITION_CLOSING_MPS)
      )
      self._clear_steady_parity("current_raw_braking" if current_raw_braking else "fast_close")
      return

    self.steady_parity_candidate_valid = False
    self.steady_parity_held = False
    samples, dense, position_slope, alead_mean = append_and_measure_position_window()
    # A current raw braking sample always removes correction in the same frame,
    # but remains in the evidence window so isolated model-accel noise is judged
    # by the specified window mean. Sustained braking drives that mean below the
    # veto; published aLead crossing the same threshold clears the epoch above.
    if current_raw_braking:
      self.steady_parity_current_threat = True
      self._clear_steady_parity("current_raw_braking", clear_history=False)
      return
    if alead_mean < -STEADY_PARITY_ALEAD_VETO_MPS2:
      self.steady_parity_current_threat = True
      self._clear_steady_parity("window_braking", clear_history=False)
      return
    slope_in_band = bool(
      position_slope is not None and
      STEADY_PARITY_MIN_POSITION_SLOPE_MPS <= position_slope <= STEADY_PARITY_MAX_POSITION_SLOPE_MPS
    )

    if dense and slope_in_band:
      floor_mps = min(0.0, float(position_slope) - max(0.0, trust_deficit))
      self.steady_parity_candidate_valid = True
      self.steady_parity_vrel_floor_mps = float(floor_mps)
      self.steady_parity_hold_floor_mps = float(floor_mps)
      hold_s = float(np.clip(float(getattr(cfg, 'steady_parity_hold_s', 0.0)), 0.0, 1.5))
      self.steady_parity_hold_until_t = float(now) + hold_s
      self.steady_parity_reason = "position_proven"
      return

    # Hold bridges only a density dropout. A dense out-of-band slope is fresh
    # contradictory position evidence and immediately clears authority.
    hold_active = bool(
      not dense and self.steady_parity_hold_floor_mps is not None and
      float(now) <= self.steady_parity_hold_until_t and
      float(getattr(cfg, 'steady_parity_hold_s', 0.0)) > 0.0 and
      (position_slope is None or slope_in_band)
    )
    if hold_active:
      self.steady_parity_candidate_valid = True
      self.steady_parity_vrel_floor_mps = float(self.steady_parity_hold_floor_mps)
      self.steady_parity_held = True
      self.steady_parity_reason = "density_hold"
      return

    self.steady_parity_hold_floor_mps = None
    self.steady_parity_hold_until_t = -1.0
    self.steady_parity_vrel_floor_mps = 0.0
    self.steady_parity_reason = "sparse_window" if not dense else "position_slope_veto"

  def _fast_closing_supported(self, raw_drel: float, raw_vrel: float, raw_dpath: float,
                              raw_vlat: float, innovation_m: float, v_ego: float,
                              cfg: LeadResponseTuningConfig) -> bool:
    if innovation_m >= -MODEL_LEAD_CLOSE_INNOVATION_M:
      self.closer_confirm_frames = max(0, self.closer_confirm_frames - 1)
      return False

    centered = abs(raw_dpath) <= MODEL_LEAD_CENTER_PATH_GATE_M
    cutin_like = abs(raw_dpath) <= 3.5 and abs(raw_vlat) >= MODEL_LEAD_CUTIN_VLAT_MPS
    closing_speed = max(0.0, -raw_vrel)
    ttc_s = raw_drel / max(closing_speed, 0.1)
    very_close = raw_drel <= max(10.0, float(v_ego) * 0.55)
    low_ttc = closing_speed > 0.5 and ttc_s <= float(cfg.model_lead_filter_safe_ttc_s)
    strong_closing = closing_speed >= MODEL_LEAD_STRONG_CLOSING_MPS

    if centered or cutin_like or low_ttc or strong_closing or very_close:
      self.closer_confirm_frames += 1
    else:
      self.closer_confirm_frames = max(0, self.closer_confirm_frames - 1)

    # Corroboration before adoption: a single heavy-tailed inward dRel outlier
    # can satisfy every gate above on its own (the TTC gate is computed from
    # the raw sample and strong_closing holds for ANY approach > 2.5 m/s), so
    # one noise frame used to collapse the track in a single 50 ms step. A real
    # cut-in / close threat keeps measuring beyond the gate frame after frame,
    # so requiring N qualifying frames (default 2 = one extra 50 ms frame,
    # during which the closing-urgency blend below still adopts pessimistically
    # at the boosted close slew) filters isolated outliers at ~0.01% of their
    # single-frame rate. ConfirmFrames=1 restores the legacy instant adoption.
    #
    # Confirmation-starvation band (genuine same-track jumps of roughly
    # 2.5-3.5 m): frame-1's urgency-blend adoption pulls the next frame's
    # innovation back UNDER the 2.5 m gate, so this counter decrements and fast
    # adoption never fires for that band — the worst-case adoption latency is
    # NOT 2 frames there. Convergence is via the blend over ~0.3-0.6 s
    # (measured: 2.4 m optimism -> 1.0 m in 0.3 s at 6 m/s ego). This matches
    # legacy risk acceptance: legacy carried the same optimism magnitude for
    # within-gate <= 2.5 m jumps, which never reached the fast path either.
    required_frames = max(1, int(round(float(cfg.model_lead_filter_fast_close_confirm_frames))))
    return bool(
      (low_ttc or strong_closing or cutin_like) and
      self.closer_confirm_frames >= required_frames
    )

  @staticmethod
  def _closing_urgency(raw_drel: float, raw_vrel: float, innovation_m: float,
                       cfg: LeadResponseTuningConfig) -> float:
    # Continuous urgency in [0, 1] filling the cliff between the slow filter and
    # the hard fast-close gates (which stay verbatim as the u=1 short-circuit).
    # Pessimistic direction only: nonzero only when the measurement says the
    # lead is CLOSER than predicted, so the blend can only speed convergence
    # toward a closer lead, never toward a farther one.
    if innovation_m >= 0.0:
      return 0.0
    # BlendTauFloorS >= ModelLeadFilterTauS is the documented disable: force u=0
    # so the vRel-tau blend and slew boost are disabled too (exact legacy).
    if float(cfg.model_lead_filter_blend_tau_floor_s) >= float(cfg.model_lead_filter_tau_s):
      return 0.0
    blend_min_span = float(getattr(cfg, 'model_lead_blend_min_span', MODEL_LEAD_BLEND_MIN_SPAN))
    ttc_min_closing_mps = float(getattr(cfg, 'model_lead_blend_ttc_min_closing_mps', MODEL_LEAD_BLEND_TTC_MIN_CLOSING_MPS))
    closing_speed = max(0.0, -raw_vrel)
    urgency = 0.0
    close_lo = float(cfg.model_lead_filter_blend_close_lo_mps)
    close_span = MODEL_LEAD_STRONG_CLOSING_MPS - close_lo
    if close_span >= blend_min_span:
      urgency = float(np.clip((closing_speed - close_lo) / close_span, 0.0, 1.0))
    ttc_hi = float(cfg.model_lead_filter_blend_ttc_hi_s)
    ttc_span = ttc_hi - float(cfg.model_lead_filter_safe_ttc_s)
    if closing_speed > ttc_min_closing_mps and ttc_span >= blend_min_span:
      ttc_s = raw_drel / max(closing_speed, 0.1)
      urgency = max(urgency, float(np.clip((ttc_hi - ttc_s) / ttc_span, 0.0, 1.0)))
    return urgency

  def update(self, lead_dict: dict[str, Any], now: float, v_ego: float,
             cfg: LeadResponseTuningConfig, lead_slot: int) -> dict[str, Any]:
    raw_drel = max(0.0, _finite_float(lead_dict.get("dRel"), self.dRel))
    raw_yrel = _finite_float(lead_dict.get("yRel"), self.yRel)
    raw_vrel = _finite_float(lead_dict.get("vRel"), self.vRel)
    raw_alead = _finite_float(lead_dict.get("aLeadK"), self.aLeadK)
    raw_prob = _finite_float(lead_dict.get("modelProb"), self.modelProb)
    raw_dpath = _finite_float(lead_dict.get("dPath", raw_yrel), raw_yrel)
    raw_vlat = _finite_float(lead_dict.get("vLat"), self.vLat)

    # A slot/identity transition must earn fresh raw opening proof. Association
    # may legitimately preserve the physical track, but a less-urgent hold must
    # never transfer across a reordered lead hypothesis.
    if self.last_slot >= 0 and int(lead_slot) != self.last_slot:
      self._clear_opening_relax(rearm_after_t=now)
      self.opening_position_evidence.clear()
      self._clear_steady_parity("slot_change")
      self._clear_accel_corr_calm_position("slot_change")

    dt_s = float(np.clip(float(now) - self.last_t, 0.0, 0.25))
    predicted_drel = self.predict_drel(now)
    innovation_m = raw_drel - predicted_drel
    # The arming gate is intentionally smaller than the 2.5 m close-innovation
    # gate: besides deep phantom adoptions, the closing-urgency blend rectifies
    # symmetric short-range noise into a persistent ~1-2 m pessimistic dRel bias
    # during the last meters of a stop (closing side adopts at the urgency tau,
    # opening side is slew-capped), and that bias must also heal.
    open_recovery_gate_m = float(cfg.model_lead_filter_open_recovery_innov_gate_m)
    if innovation_m > open_recovery_gate_m > 0.0:
      self.opening_confirm_frames += 1
    else:
      self.opening_confirm_frames = 0
    fast_closing = self._fast_closing_supported(raw_drel, raw_vrel, raw_dpath, raw_vlat, innovation_m, v_ego, cfg)
    self._update_steady_parity(
      now, raw_drel, raw_vrel, raw_alead, raw_prob, raw_dpath, raw_vlat, v_ego, cfg,
    )

    urgency = 0.0 if fast_closing else self._closing_urgency(raw_drel, raw_vrel, innovation_m, cfg)
    # CD9: a corroborated sustained closure forces full urgency (the existing
    # fast-tau/slew-boost machinery) even while the per-frame gates read calm -
    # the road event sat below BlendCloseLoMps for the entire first second.
    governor_active = self._update_closing_governor(
      now, raw_drel, raw_vrel, raw_alead, cfg,
      raw_prob=raw_prob, raw_dpath=raw_dpath, raw_vlat=raw_vlat,
    )
    if governor_active and not fast_closing:
      urgency = 1.0
    self._update_opening_governor(
      now, governor_active, cfg,
      raw_drel=raw_drel, raw_vrel=raw_vrel, raw_alead=raw_alead,
      raw_dpath=raw_dpath, raw_vlat=raw_vlat, v_ego=v_ego,
    )

    if fast_closing:
      drel_alpha = max(0.65, _ema_alpha(dt_s, MODEL_LEAD_FAST_CLOSE_TAU_S))
      next_drel = predicted_drel + drel_alpha * innovation_m
      self.drel_lag_s = MODEL_LEAD_FAST_CLOSE_TAU_S
    else:
      close_slew_mps = MODEL_LEAD_NOISE_CLOSE_SLEW_MPS + max(0.0, -raw_vrel) * 0.4
      open_slew_mps = float(cfg.model_lead_filter_open_slew_max_mps) + max(0.0, raw_vrel)
      tau_s = float(cfg.model_lead_filter_tau_s) * (1.6 if innovation_m < 0.0 else 1.0)
      if urgency > 0.0:
        # Geometric blend toward the urgency tau floor; floor_eff never exceeds
        # tau_s so blended closing adoption is never slower than legacy, and the
        # slew boost keeps the close-slew clamp from binding at high urgency.
        floor_eff = min(float(cfg.model_lead_filter_blend_tau_floor_s), tau_s)
        tau_s = tau_s ** (1.0 - urgency) * floor_eff ** urgency
        close_slew_mps += float(cfg.model_lead_filter_blend_slew_boost_mps) * urgency
      drel_alpha = _ema_alpha(dt_s, tau_s)
      open_step_max_m = max(0.0, open_slew_mps) * dt_s
      # Corroborated low-speed symmetric recovery: if the raw measurement has
      # been beyond the innovation gate on the OPENING side for N consecutive
      # frames (isolated opening outliers at the measured 2% rate cannot chain
      # that long), the internal state is wrong-too-close — e.g. a phantom
      # inward adoption — and must heal before standstill instead of ratcheting
      # against the still-closing prediction at the 1.2 m/s opening slew.
      # At low ego speed the repeated-distance corroboration is sufficient. At
      # higher speed, recovery additionally requires a non-braking raw lead and
      # a safe raw-side TTC. This closes the freeway failure mode where one
      # adopted inward outlier left dRel 10-15 m wrong-too-close indefinitely,
      # while preserving the pessimistic state for any nearby or braking threat.
      # MaxEgoMps > 0.0 guard makes the documented kill switch exact: with the
      # knob at 0.0, 'v_ego <= 0.0' would still arm the recovery at standstill,
      # so A/B isolation with OpenRecoveryMaxEgoMps=0 would not be bit-exact
      # legacy while stopped.
      open_recovery_required = max(1, int(round(float(cfg.model_lead_filter_open_recovery_confirm_frames))))
      open_recovery_max_ego_mps = float(cfg.model_lead_filter_open_recovery_max_ego_mps)
      raw_closing_mps = max(0.0, -raw_vrel)
      raw_ttc_s = raw_drel / max(raw_closing_mps, 0.1)
      high_speed_recovery_safe = (
        raw_alead >= -float(getattr(cfg, 'opening_governor_alead_veto_mps2', 0.2)) and
        raw_ttc_s >= OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S
      )
      if (innovation_m > 0.0 and
          self.opening_confirm_frames >= open_recovery_required and
          open_recovery_max_ego_mps > 0.0 and
          (float(v_ego) <= open_recovery_max_ego_mps or high_speed_recovery_safe)):
        recovery_alpha = _ema_alpha(dt_s, float(cfg.model_lead_filter_open_recovery_tau_s))
        drel_alpha = max(drel_alpha, recovery_alpha)
        open_step_max_m = max(open_step_max_m, recovery_alpha * innovation_m)
      target_drel = predicted_drel + drel_alpha * innovation_m
      step_m = float(np.clip(
        target_drel - predicted_drel,
        -max(0.0, close_slew_mps) * dt_s,
        open_step_max_m,
      ))
      next_drel = predicted_drel + step_m
      self.drel_lag_s = float(tau_s)

    if fast_closing:
      vrel_tau_s = float(cfg.model_lead_filter_fast_vrel_tau_s)
    elif urgency > 0.0:
      vrel_tau_s = ((1.0 - urgency) * float(cfg.model_lead_filter_vrel_tau_s)
                    + urgency * float(cfg.model_lead_filter_fast_vrel_tau_s))
    else:
      vrel_tau_s = float(cfg.model_lead_filter_vrel_tau_s)
    vrel_alpha = _ema_alpha(dt_s, vrel_tau_s)
    lat_alpha = _ema_alpha(dt_s, MODEL_LEAD_LAT_TAU_S)
    # CD9: while the governor is latched the aLeadK EMA runs at the fast tau so
    # the published lead decel converges to the model's measurement ~3x sooner
    # (road: the 0.60 s tau halved the published decel through the whole event).
    accel_tau_s = float(getattr(cfg, 'closing_governor_alead_tau_s', MODEL_LEAD_ACCEL_TAU_S)) \
      if governor_active else MODEL_LEAD_ACCEL_TAU_S
    accel_alpha = _ema_alpha(dt_s, accel_tau_s)
    prob_alpha = _ema_alpha(dt_s, MODEL_LEAD_PROB_TAU_S)

    self.dRel = float(max(0.0, next_drel))
    self._update_fcw_corroboration(raw_drel, raw_vrel, cfg)
    self.yRel = float(self.yRel + lat_alpha * (raw_yrel - self.yRel))
    self.dPath = float(self.dPath + lat_alpha * (raw_dpath - self.dPath))
    self.vLat = float(self.vLat + lat_alpha * (raw_vlat - self.vLat))
    self.vRel = float(self.vRel + vrel_alpha * (raw_vrel - self.vRel))
    self.vLead = float(v_ego + self.vRel)
    self.vLeadK = self.vLead
    self._apply_far_range_vlead_optimism_clamp(raw_vrel, v_ego, next_drel, now, cfg)
    # CD9 clamp is PUBLISH-TIME ONLY (get_RadarState): mutating the tracker's
    # vRel state here corrupted association (the assoc vRel gate compares track
    # state to measurements) and fed the prediction, causing track-id churn.
    self.governor_active = bool(governor_active)
    self.aLeadK = float(self.aLeadK + accel_alpha * (raw_alead - self.aLeadK))
    self.aLeadTau = 0.3
    self.modelProb = float(self.modelProb + prob_alpha * (raw_prob - self.modelProb))
    self._update_accel_corr_calm_position(
      now, raw_drel, raw_vrel, raw_alead, raw_prob, raw_dpath, raw_vlat, v_ego,
    )
    self.last_t = float(now)
    self.last_slot = int(lead_slot)
    self.age += 1
    self.missed = 0
    return self.get_RadarState(cfg)

  def _update_closing_governor(self, now: float, raw_drel: float, raw_vrel: float,
                               raw_alead: float, cfg: LeadResponseTuningConfig, *,
                               raw_prob: float = 1.0, raw_dpath: float = 0.0,
                               raw_vlat: float = 0.0) -> bool:
    """CD9 corroborated-closing governor (road 205-6 Event B).

    The road near-collision was pure publish-side latency: the model's RAW
    streams showed a braking lead on time (sustained raw aLead -0.55 from
    onset, raw vRel closing, raw dRel collapsing), but the published state ran
    a compounded EMA lag behind them - vRel tau 0.60 with the urgency blend
    gated below BlendCloseLoMps, dRel close-tau 2.8x1.6 under a ~1 m/s slew
    clamp, aLeadK tau 0.60 halving the decel - so the planner's brake ramp
    trailed the closure by ~1.9 s and the driver had to stomp at THW 1.10 s.

    This governor watches the RAW evidence over a short window and, when the
    window CORROBORATES a sustained closure two independent ways, latches a
    fast-filter regime (urgency forced to 1 -> existing fast taus; fast aLeadK
    tau) plus a one-directional publish clamp of vLead toward the LEAST
    aggressive corroborated closure. Two arm paths, BOTH requiring the
    windowed raw vRel to agree the lead is closing (> MinClosingMps):

      position-excess: the endpoint-mean slope of the raw dRel window closes
        faster than the currently PUBLISHED closing speed by MarginMps - the
        tracker's own position stream is outrunning what it publishes.
      sustained-decel: the windowed mean raw lead accel is below
        -AccelOnsetMps2 - the earliest reliable road signal (raw aLead mean
        separated -0.55 vs -0.10 steady ~0.5 s before the position slope).

    Noise immunity is structural: single heavy-tail dRel outliers (road:
    +-1.5-3 m frames) cannot dominate k-endpoint means over the window;
    symmetric vRel noise cannot hold a windowed mean past MinClosingMps while
    ALSO faking a position excess / sustained decel; an opening or steady
    follow fails the vRel-agreement gate outright. The clamp only ever LOWERS
    the published vLead (more urgent), never raises it, and the taus it forces
    still filter the model's own measurements - nothing is fabricated.

    Rollback sentinels: ClosingGovernorMarginMps >= 99 disables the governor
    entirely (exact legacy publish); ClosingGovernorAccelOnsetMps2 >= 99
    disables acceleration-derived arming and full position trust while leaving
    position-excess, velocity-earned, and short-raw-TTC authority available.
    """
    self.closing_evidence.append((float(now), float(raw_drel), float(raw_vrel), float(raw_alead)))
    self.opening_position_evidence.append((float(now), float(raw_drel)))
    self.governor_calm_recovery_applied = False
    self.governor_recovery_position_closing_mps = None
    self.governor_recovery_vrel_floor_mps = None

    margin = float(getattr(cfg, 'closing_governor_margin_mps', 99.0))
    if margin >= 99.0:
      self.governor_hold_until_t = -1.0
      self.governor_closing_mps = 0.0
      self.governor_reason = "disabled"
      self.governor_threat_corroborated = False
      self.governor_calm_recovery_mode = False
      return False

    window_s = max(0.2, float(getattr(cfg, 'closing_governor_window_s', 0.7)))
    while self.closing_evidence and (now - self.closing_evidence[0][0]) > window_s:
      self.closing_evidence.popleft()

    active = now <= self.governor_hold_until_t
    samples = list(self.closing_evidence)
    span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    # Demand real coverage of the window (missed/dropped frames leave holes):
    # a sparse window must not arm the fast regime.
    if len(samples) < 8 or span < 0.5 * window_s:
      self.governor_calm_recovery_mode = False
      if not active:
        self.governor_reason = "inactive"
        self.governor_threat_corroborated = False
      return active

    min_closing = float(getattr(cfg, 'closing_governor_min_closing_mps', 0.30))
    vrel_closing = -(sum(s[2] for s in samples) / len(samples))
    # A weak, position-only latch may start while velocity evidence is calm,
    # then outlive the arm condition as the same dense window develops a real
    # closure. Upgrade that already-active latch before any early return. This
    # ceiling is deliberately fixed: a permissive opening raw-closing tune
    # must never let a previously earned opening hold mask sustained closure.
    if (active and not self.governor_threat_corroborated and
        vrel_closing > OPENING_GOVERNOR_WEAK_CLOSING_MAX_MPS):
      self.governor_threat_corroborated = True
      self.governor_reason = "velocity_corroborated"
    # The hold bridges brief evidence dropouts during a continuing closure, but
    # it must not preserve a stale closing clamp after the SAME window has
    # settled near parity. That was happening on-road and in the EV6 loop: raw
    # vRel had recovered to roughly parity while the held governor continued
    # publishing a 1.8 m/s closure, so the MPC braked an already-opening gap and
    # drove ego below lead speed. Releasing here only removes the extra
    # publish-side pessimism; the tracker's filtered state and every MPC safety
    # path remain in force. A braking lead vetoes the early release.
    opening_alead_veto = float(getattr(cfg, 'opening_governor_alead_veto_mps2', 0.2))
    alead_mean = sum(s[3] for s in samples) / len(samples)
    raw_closing = max(0.0, -float(raw_vrel))
    raw_collision_ttc_s = float(raw_drel) / max(raw_closing, 0.1)
    current_braking = float(raw_alead) < -opening_alead_veto
    current_short_ttc = raw_closing > min_closing and raw_collision_ttc_s <= OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S
    current_fast_close = raw_closing >= OPENING_GOVERNOR_HARD_CLOSING_MPS
    if active and (current_braking or current_short_ttc or current_fast_close):
      self.governor_threat_corroborated = True
      self.governor_calm_recovery_mode = False
      if current_braking:
        self.governor_reason = "current_braking"
      elif current_short_ttc:
        self.governor_reason = "short_raw_ttc"
      else:
        self.governor_reason = "fast_close"

    # A model-velocity burst can earn a legitimate closing hold, then recover
    # while that stale clamp remains frozen for up to one second. Reconcile only
    # an ALREADY-ACTIVE hold after two actual consecutive calm measurements and
    # a dense, full one-second position history agree the closure is mild. Using
    # the measured samples themselves (rather than a mutable counter) makes a
    # miss or scheduler gap fail closed. The base filtered track and the earned
    # hold deadline stay intact. The evidence defines a ceiling on CD9's EXTRA
    # one-directional clamp; the ordinary filtered vRel path remains live and
    # can still publish a stronger closure as current evidence rises.
    long_position_closing: float | None = None
    long_samples = [
      sample for sample in self.opening_position_evidence
      if (now - sample[0]) <= CLOSING_GOVERNOR_RECOVERY_POSITION_WINDOW_S
    ]
    long_span = long_samples[-1][0] - long_samples[0][0] if len(long_samples) >= 2 else 0.0
    long_gaps = [b[0] - a[0] for a, b in zip(long_samples, long_samples[1:], strict=False)]
    long_dense = bool(
      len(long_samples) >= CLOSING_GOVERNOR_RECOVERY_MIN_POSITION_SAMPLES and
      long_span >= CLOSING_GOVERNOR_RECOVERY_MIN_POSITION_SPAN_S and
      long_gaps and max(long_gaps) <= 2.0 * CLOSING_GOVERNOR_RECOVERY_MAX_SAMPLE_GAP_S
    )
    if long_dense:
      long_k = max(2, len(long_samples) // 4)
      long_first, long_last = long_samples[:long_k], long_samples[-long_k:]
      long_t_first = sum(sample[0] for sample in long_first) / long_k
      long_t_last = sum(sample[0] for sample in long_last) / long_k
      if (long_t_last - long_t_first) > 1e-3:
        long_d_first = sum(sample[1] for sample in long_first) / long_k
        long_d_last = sum(sample[1] for sample in long_last) / long_k
        long_position_closing = max(0.0, -(long_d_last - long_d_first) / (long_t_last - long_t_first))

    # Recovery's acceleration veto is never weakenable past the fixed 0.2 m/s²
    # safety boundary, even if an opening-governor tune is made permissive.
    recovery_alead_veto = min(max(0.0, opening_alead_veto), 0.2)
    recent_samples = samples[-2:]
    consecutive_measured = bool(
      len(recent_samples) == 2 and
      1e-3 < (recent_samples[1][0] - recent_samples[0][0]) <= CLOSING_GOVERNOR_RECOVERY_MAX_SAMPLE_GAP_S
    )

    # The long-window position estimate owns the recovery floor, but it must
    # agree with the newest raw range step in this same publication. This veto
    # catches a genuine closure before the robust one-second window can react;
    # it never supplies the floor itself, so a single noisy step cannot create
    # less-urgent evidence.
    recent_position_safe = False
    recent_position_samples = list(self.opening_position_evidence)[-2:]
    if len(recent_position_samples) == 2:
      (recent_t0, recent_d0), (recent_t1, recent_d1) = recent_position_samples
      recent_dt = float(recent_t1) - float(recent_t0)
      if (math.isfinite(recent_dt) and math.isfinite(float(recent_d0)) and math.isfinite(float(recent_d1)) and
          1e-3 < recent_dt <= CLOSING_GOVERNOR_RECOVERY_MAX_SAMPLE_GAP_S):
        recent_position_closing = max(0.0, (float(recent_d0) - float(recent_d1)) / recent_dt)
        recent_position_ttc_s = float(raw_drel) / max(recent_position_closing, 0.1)
        recent_position_safe = bool(
          recent_position_closing <= CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS and
          recent_position_ttc_s > CLOSING_GOVERNOR_RECOVERY_MIN_RAW_TTC_S
        )

    def calm_recovery_sample(sample: tuple[float, float, float, float]) -> bool:
      _, sample_drel, sample_vrel, sample_alead = sample
      sample_closing = max(0.0, -float(sample_vrel))
      sample_ttc_s = float(sample_drel) / max(sample_closing, 0.1)
      return bool(
        float(sample_alead) >= -recovery_alead_veto and
        sample_closing < OPENING_GOVERNOR_HARD_CLOSING_MPS and
        sample_ttc_s > OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S
      )

    def numeric_recovery_sample_safe(sample: tuple[float, float, float, float]) -> bool:
      _, sample_drel, sample_vrel, sample_alead = sample
      sample_closing = max(0.0, -float(sample_vrel))
      sample_ttc_s = float(sample_drel) / max(sample_closing, 0.1)
      return bool(
        float(sample_alead) >= -recovery_alead_veto and
        sample_closing < OPENING_GOVERNOR_HARD_CLOSING_MPS and
        sample_ttc_s > CLOSING_GOVERNOR_RECOVERY_MIN_RAW_TTC_S
      )

    raw_recovery_context_safe = bool(
      float(raw_prob) >= STEADY_PARITY_MIN_MODEL_PROB and
      abs(float(raw_dpath)) <= STEADY_PARITY_MAX_DPATH_M and
      not (abs(float(raw_dpath)) > STEADY_PARITY_VLAT_OFFCENTER_DPATH_M and
           abs(float(raw_vlat)) >= STEADY_PARITY_MAX_VLAT_MPS)
    )
    calm_recovery = bool(
      active and
      self.governor_threat_corroborated and
      consecutive_measured and
      all(calm_recovery_sample(sample) for sample in recent_samples) and
      alead_mean >= -recovery_alead_veto and
      long_position_closing is not None and
      long_position_closing <= CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS
    )
    numeric_recovery_safe = bool(
      raw_recovery_context_safe and
      recent_position_safe and
      all(numeric_recovery_sample_safe(sample) for sample in recent_samples) and
      raw_collision_ttc_s > CLOSING_GOVERNOR_RECOVERY_MIN_RAW_TTC_S
    )
    recovery_closing = None if long_position_closing is None else max(raw_closing, float(long_position_closing))
    enters_recovery = bool(
      calm_recovery and recovery_closing is not None and
      self.governor_closing_mps > recovery_closing + 1e-3
    )
    if calm_recovery and recovery_closing is not None and (self.governor_calm_recovery_mode or enters_recovery):
      # Once reconciliation starts, stale CD9 authority may only decay while
      # the calm cross-signal contract remains true. Raising this extra clamp
      # to chase a noisy-but-subcritical raw vRel recreated braking in route
      # 422; the base filtered vRel already handles that current measurement.
      self.governor_closing_mps = min(self.governor_closing_mps, recovery_closing)
      self.governor_reason = "calm_recovery_capped"
      self.governor_calm_recovery_mode = True
      self.governor_calm_recovery_applied = True
      self.governor_recovery_position_closing_mps = float(long_position_closing)
      # Raw/model vRel is the signal that the independent dense position window
      # has disproven. Publish the position-only correction for the planner's
      # positive brake-release seam; public vRel and the governor's own clamp
      # remain conservative at max(raw, position) above.
      if numeric_recovery_safe:
        self.governor_recovery_vrel_floor_mps = -float(long_position_closing)
      return True
    self.governor_calm_recovery_mode = False

    release_closing_max = 0.5 * min_closing
    if vrel_closing <= release_closing_max and alead_mean >= -opening_alead_veto:
      self.governor_hold_until_t = -1.0
      self.governor_closing_mps = 0.0
      self.governor_reason = "inactive"
      self.governor_threat_corroborated = False
      self.governor_calm_recovery_mode = False
      return False
    if vrel_closing <= min_closing:
      return active

    k = max(2, len(samples) // 4)
    first_k, last_k = samples[:k], samples[-k:]
    t_first = sum(s[0] for s in first_k) / k
    t_last = sum(s[0] for s in last_k) / k
    if (t_last - t_first) <= 1e-3:
      return active
    d_first = sum(s[1] for s in first_k) / k
    d_last = sum(s[1] for s in last_k) / k
    pos_closing = -(d_last - d_first) / (t_last - t_first)

    published_closing = max(0.0, -float(self.vRel))
    armed_position = pos_closing > published_closing + margin and pos_closing > min_closing

    accel_onset = float(getattr(cfg, 'closing_governor_accel_onset_mps2', 99.0))
    armed_decel = accel_onset < 99.0 and alead_mean < -accel_onset

    if armed_position or armed_decel:
      hold_s = float(getattr(cfg, 'closing_governor_hold_s', 1.0))
      self.governor_hold_until_t = float(now) + max(0.1, hold_s)
      # Corroborated closure for the publish clamp: the position stream may
      # LEAD the velocity evidence by a bounded trust headroom (the road's raw
      # v-stream itself lied optimistic against the model's own position
      # stream), but never beyond it - a pure position phantom stays capped
      # near the gated vRel evidence.
      # Position may lead the windowed velocity stream only by independently
      # corroborated authority. A TTC derived from this same dRel slope is not
      # independent: the captured EV6 raw-x collapse produced both a 7.16 m/s
      # position closure and a 5.99 s "collision" TTC while raw vRel/aLead were
      # calm, spending the full +1.5 m/s allowance from one noisy signal.
      #
      # Sustained lead braking and a short TTC from the CURRENT raw velocity
      # retain the full allowance. A single raw-aLead frame is not enough: the
      # road corpus has isolated threshold crossings, so acceleration must be
      # confirmed by the window mean or two consecutive model-cadence samples.
      # Full position trust uses the sustained-decel onset threshold, not the
      # smaller opening-recovery veto. The 2026-07-15 20:38:27 capture had two
      # mild -0.30/-0.24 m/s^2 samples trip the 0.20 opening veto and spend the
      # entire +1.5 m/s position allowance even as raw range was opening. Keep
      # that state threat-corroborated (current_braking above still uses the
      # opening veto), but do not amplify position noise until aLead reaches
      # the independently tuned sustained-braking threshold.
      # Otherwise windowed velocity corroboration earns position excess
      # continuously, one-for-one above MarginMps and capped by the tune.
      # This removes the binary full-trust cliff at vrel_closing == margin.
      max_pos_trust = max(0.0, float(getattr(cfg, 'closing_governor_pos_trust_excess_mps', 0.0)))
      recent_alead_decel = bool(
        accel_onset < 99.0 and len(samples) >= 2 and
        all(s[3] < -accel_onset for s in samples[-2:])
      )
      braking_corroborated = armed_decel or recent_alead_decel
      short_raw_ttc = raw_closing > min_closing and raw_collision_ttc_s <= OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S
      if braking_corroborated or short_raw_ttc:
        pos_trust = max_pos_trust
      else:
        velocity_trust = max(0.0, vrel_closing - max(0.0, margin))
        # Before the second decel sample arrives, aLead may contribute only its
        # continuous magnitude beyond the veto, never the full tuned allowance.
        # Isolated near-threshold corpus crossings therefore add only a small,
        # bounded amount while a real -0.7 m/s^2 onset is not ignored for 50 ms.
        unconfirmed_alead_cap = max(0.0, float(getattr(cfg, 'closing_governor_unconfirmed_alead_trust_mps', 0.0)))
        unconfirmed_alead_trust = min(unconfirmed_alead_cap,
                                      max(0.0, -float(raw_alead) - opening_alead_veto))
        pos_trust = min(max_pos_trust, max(velocity_trust, unconfirmed_alead_trust))
      self.governor_closing_mps = float(max(0.0, min(pos_closing, vrel_closing + max(0.0, pos_trust))))
      threat_corroborated = bool(
        armed_decel or braking_corroborated or current_braking or short_raw_ttc or
        current_fast_close or vrel_closing > OPENING_GOVERNOR_WEAK_CLOSING_MAX_MPS
      )
      # Corroboration is sticky for the whole CD9 hold. A real braking onset may
      # recover its current aLead on later frames, but the opening side must not
      # reinterpret that recovery as permission to override the active threat.
      self.governor_threat_corroborated = bool(
        (active and self.governor_threat_corroborated) or threat_corroborated
      )
      if self.governor_threat_corroborated:
        if armed_decel or braking_corroborated:
          self.governor_reason = "lead_braking"
        elif short_raw_ttc:
          self.governor_reason = "short_raw_ttc"
        elif current_fast_close:
          self.governor_reason = "fast_close"
        else:
          self.governor_reason = "velocity_corroborated"
      else:
        self.governor_reason = "position_only_weak"
      return True
    if not active:
      self.governor_reason = "inactive"
      self.governor_threat_corroborated = False
      self.governor_calm_recovery_mode = False
    return active

  def _update_opening_governor(self, now: float, closing_governor_active: bool,
                               cfg: LeadResponseTuningConfig, *,
                               raw_drel: float | None = None,
                               raw_vrel: float | None = None,
                               raw_alead: float | None = None,
                               raw_dpath: float | None = None,
                               raw_vlat: float | None = None,
                               v_ego: float | None = None) -> None:
    """Opening governor: CD9's exact mirror (2026-07-08 phantom-closing runs).

    The publish pipeline's safety asymmetry is deliberate — lag comp boosts
    vRel closing-ward, CD9 clamps vLead slower-ward, fast recovery exists for
    the CLOSING direction only (the opening recovery path is gated below
    8 m/s ego and opening dRel is slew-clamped) — so above ~18 mph, opening
    truth can only leak back through the slow main vRel EMA. Measured on the
    2026-07-08 drive: 22.3% of lead-tracking frames published vRel <= -1.0
    while the position stream showed the gap OPENING >= 0.2 m/s, in runs up to
    6.9 s, with the planner braking through 27% of them — the felt "rides the
    brakes until a 3 second gap" and most of the ~2 s follow floor (the MPC's
    desired distance inflates quadratically with published closing speed).

    While the SAME raw evidence window CD9 trusts proves a sustained opening
    (k-endpoint mean slope of raw dRel), and nothing threatening is in the
    window, the published vRel gets a one-directional relax FLOOR:

      floor = min(pos_opening - TrustDeficitMps, 0.0)

    max() semantics at publish: it can only ever make the published lead
    FASTER / less urgent, and never past parity (0.0) — the mirror image of
    CD9's min() clamp, with the same position-primacy rationale (CD9's own
    road evidence showed the raw v-stream lying against the position stream).

    Vetoes (any -> no relax; ambiguity resolves toward MORE braking):
      - threat-corroborated CD9 latched or holding
      - windowed raw vRel mean shows closing beyond RawClosingVetoMps
      - windowed raw aLead mean below -ALeadVetoMps2 (braking lead)
      - published closing TTC under OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S
      - sparse window (same coverage demand as CD9)

    OpeningGovernorHoldS extends an earned correction across a short proof
    dropout only while a longer raw-position trend remains non-closing. A weak,
    position-only CD9 latch may coexist with that already-earned hold, but can
    neither create nor refresh it. Current braking, fast/short-TTC closure,
    oncoming/lateral/near threats, misses, and identity changes clear all
    less-urgent state immediately.

    Rollback sentinels: OpeningGovernorTrustDeficitMps >= 99 disables entirely;
    OpeningGovernorHoldS = 0 restores exact per-frame behavior.
    """
    self.opening_relax_vrel = None
    self.opening_relax_held = False

    trust_deficit = float(getattr(cfg, 'opening_governor_trust_deficit_mps', 99.0))
    if trust_deficit >= 99.0:
      self._clear_opening_relax()
      return

    hold_s = float(getattr(cfg, 'opening_governor_hold_s', 0.0))
    legacy_per_frame = hold_s <= 0.0
    closing_governor_engaged = bool(closing_governor_active or now <= self.governor_hold_until_t)
    if legacy_per_frame:
      # Exact rollback: no retained state, and any CD9 latch has the historical
      # unconditional precedence over a same-frame opening proof.
      self.opening_relax_hold_vrel = None
      self.opening_relax_hold_until_t = -1.0
      self.opening_last_raw_proof_t = -1.0
      if closing_governor_engaged:
        return

    samples_all = list(self.closing_evidence)
    last_sample = samples_all[-1] if samples_all else (float(now), float(self.dRel), float(self.vRel), float(self.aLeadK))
    current_drel = max(0.0, float(last_sample[1] if raw_drel is None else raw_drel))
    current_vrel = float(last_sample[2] if raw_vrel is None else raw_vrel)
    current_alead = float(last_sample[3] if raw_alead is None else raw_alead)
    current_dpath = float(self.dPath if raw_dpath is None else raw_dpath)
    current_vlat = float(self.vLat if raw_vlat is None else raw_vlat)
    current_vego = max(0.0, float(self.vLead - self.vRel if v_ego is None else v_ego))

    if not legacy_per_frame:
      alead_veto = float(getattr(cfg, 'opening_governor_alead_veto_mps2', 0.2))
      # Tuning may make the ordinary window veto more conservative, but it may
      # not disable the same-frame braking escape used by the held state.
      hard_alead_veto = min(max(0.0, alead_veto), 0.2)
      raw_closing = max(0.0, -current_vrel)
      raw_ttc_s = current_drel / max(raw_closing, 0.1)
      raw_vlead = current_vego + current_vrel
      near_threat = current_drel <= max(10.0, 0.55 * current_vego)
      lateral_threat = bool(
        abs(current_vlat) >= MODEL_LEAD_CUTIN_VLAT_MPS or
        abs(current_dpath - float(self.dPath)) >= MODEL_LEAD_DUPLICATE_PATH_GATE_M
      )
      current_hard_threat = bool(
        current_alead < -hard_alead_veto or
        raw_closing >= OPENING_GOVERNOR_HARD_CLOSING_MPS or
        (raw_closing > 0.3 and raw_ttc_s <= OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S) or
        raw_vlead < 0.0 or lateral_threat or near_threat or
        (closing_governor_engaged and self.governor_threat_corroborated)
      )
      if current_hard_threat:
        self._clear_opening_relax(rearm_after_t=now)
        return

    published_closing = max(0.0, -float(self.vRel))
    if published_closing <= 1e-3:
      if legacy_per_frame:
        return  # publish already agrees: nothing to relax
    if published_closing > 1e-3 and (float(self.dRel) / published_closing) < OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S:
      if not legacy_per_frame:
        self._clear_opening_relax(rearm_after_t=now)
      return

    window_s = max(0.2, float(getattr(cfg, 'closing_governor_window_s', 0.7)))
    samples = [
      s for s in self.closing_evidence
      if (now - s[0]) <= window_s and (legacy_per_frame or s[0] > self.opening_evidence_epoch_t)
    ]
    span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    dense_window = len(samples) >= 8 and span >= 0.5 * window_s

    # A deadline alone is not enough to retain a less-urgent publish. Bridge a
    # noisy short-window dropout only while a longer RAW-position trend remains
    # non-closing. This history cannot arm or extend the hold; it can only end
    # one early. The route-232 oscillation stays positive on this horizon, while
    # the braking-lead twin's genuine closure turns it negative before onset.
    long_window_s = min(1.5, max(1.0, hold_s + 0.25))
    long_samples = [
      s for s in self.opening_position_evidence
      if (now - s[0]) <= long_window_s and s[0] > self.opening_evidence_epoch_t
    ]
    long_span = long_samples[-1][0] - long_samples[0][0] if len(long_samples) >= 2 else 0.0
    self.opening_long_position_slope_mps = None
    self.opening_bridge_position_safe = False
    if len(long_samples) >= 8 and long_span >= 0.5:
      long_k = max(2, len(long_samples) // 4)
      long_first, long_last = long_samples[:long_k], long_samples[-long_k:]
      long_t_first = sum(s[0] for s in long_first) / long_k
      long_t_last = sum(s[0] for s in long_last) / long_k
      if (long_t_last - long_t_first) > 1e-3:
        long_d_first = sum(s[1] for s in long_first) / long_k
        long_d_last = sum(s[1] for s in long_last) / long_k
        self.opening_long_position_slope_mps = float(
          (long_d_last - long_d_first) / (long_t_last - long_t_first)
        )
        self.opening_bridge_position_safe = self.opening_long_position_slope_mps >= -0.05

    raw_closing_veto = float(getattr(cfg, 'opening_governor_raw_closing_veto_mps', 1.0))
    vrel_closing = -(sum(s[2] for s in samples) / len(samples)) if samples else 0.0
    if dense_window and vrel_closing > raw_closing_veto:
      if not legacy_per_frame:
        self._clear_opening_relax(rearm_after_t=now)
      return
    alead_veto = float(getattr(cfg, 'opening_governor_alead_veto_mps2', 0.2))
    alead_mean = sum(s[3] for s in samples) / len(samples) if samples else 0.0
    if dense_window and alead_mean < -alead_veto:
      if not legacy_per_frame:
        self._clear_opening_relax(rearm_after_t=now)
      return

    hold_active = bool(
      not legacy_per_frame and self.opening_relax_hold_vrel is not None and
      now <= self.opening_relax_hold_until_t
    )
    if closing_governor_engaged:
      # A weak position-only CD9 latch may coexist only with a hold that was
      # already earned. It cannot arm or refresh one, and CD9 automatically
      # regains publish precedence at the absolute raw-proof deadline.
      if hold_active and not self.governor_threat_corroborated and self.opening_bridge_position_safe:
        self.opening_relax_vrel = float(min(self.opening_relax_hold_vrel, 0.0))
        self.opening_relax_held = True
      else:
        self._clear_opening_relax(
          rearm_after_t=now if self.governor_threat_corroborated else None,
        )
      return

    if not dense_window:
      if hold_active and self.opening_bridge_position_safe:
        self.opening_relax_vrel = float(min(self.opening_relax_hold_vrel, 0.0))
        self.opening_relax_held = True
      elif not legacy_per_frame:
        self._clear_opening_relax()
      return

    k = max(2, len(samples) // 4)
    first_k, last_k = samples[:k], samples[-k:]
    t_first = sum(s[0] for s in first_k) / k
    t_last = sum(s[0] for s in last_k) / k
    if (t_last - t_first) <= 1e-3:
      return
    d_first = sum(s[1] for s in first_k) / k
    d_last = sum(s[1] for s in last_k) / k
    pos_opening = (d_last - d_first) / (t_last - t_first)

    min_opening = float(getattr(cfg, 'opening_governor_min_opening_mps', 0.2))
    if pos_opening < min_opening:
      if hold_active and self.opening_bridge_position_safe:
        self.opening_relax_vrel = float(min(self.opening_relax_hold_vrel, 0.0))
        self.opening_relax_held = True
      elif not legacy_per_frame:
        self._clear_opening_relax()
      return
    proof_floor = float(min(pos_opening - trust_deficit, 0.0))
    self.opening_relax_vrel = proof_floor
    if not legacy_per_frame:
      if hold_active and self.opening_relax_hold_vrel is not None:
        # Retain the least-pessimistic floor earned inside this uninterrupted
        # proof episode; all threat exits above still revoke it immediately.
        proof_floor = max(proof_floor, float(self.opening_relax_hold_vrel))
      self.opening_relax_vrel = float(min(proof_floor, 0.0))
      self.opening_relax_hold_vrel = self.opening_relax_vrel
      self.opening_relax_hold_until_t = float(now) + min(max(0.0, hold_s), 1.5)
      self.opening_last_raw_proof_t = float(now)

  def _apply_far_range_vlead_optimism_clamp(self, raw_vrel: float, v_ego: float,
                                            next_drel: float, now: float,
                                            cfg: LeadResponseTuningConfig) -> None:
    """CD8 far-range stopped-traffic vLead optimism clamp (road 200-13 EDGE2).

    While a far, newly-acquired lead is still stopping/slow, the model's
    published vLead runs biased HIGH (road: ~+4 m/s vs position-derived truth),
    so the kinematic stopping-need term (compute_lead_stopping_need_decel, which
    uses vLead) computes a much smaller required decel than reality and the
    cruise->lead handoff / braking starts late, forcing a concentrated hard stop.

    When the RAW model lead velocity is declining monotonically across recent
    frames (a stopping/decelerating lead) AND range is beyond a live-tunable far
    threshold, publish min(model vLead, position-derived vLead), where the
    position-derived velocity is estimated from the internal-filter dRel trend
    (d(dRel)/dt + v_ego). This lets the car see the truth (a slower/stopping
    lead) sooner, so the existing stopping-need handoff leg engages ~2 s earlier.

    SAFETY: this is publish-time only (internal EMA state untouched, no feedback
    lag) and STRICTLY one-directional - it only ever makes the published vLead
    SLOWER / more urgent (min), never faster / less urgent. It is tightly guarded
    so it does NOT fire on a genuinely moving/steady lead (raw vLead not
    monotonically declining), on noise (a SUSTAINED decline over N frames is
    required, not one frame), or on a near/normal follow (range below the far
    threshold). Rollback sentinel: LeadVLeadOptimismClampRangeM = 1e9 (range
    unreachable) OR LeadVLeadOptimismClampGain = 0 disables the clamp entirely.

    NOTE ON aLeadTau: the published clamp lowers vLead but does NOT synthesize a
    lead decel into aLeadK - the stopping-need trigger uses vLead directly (the
    understated closure is what delayed the handoff), so correcting vLead is
    sufficient and avoids fabricating a decel the model never measured.
    """
    far_range_m = float(getattr(cfg, 'lead_vlead_optimism_clamp_range_m', 1e9))
    gain = float(getattr(cfg, 'lead_vlead_optimism_clamp_gain', 0.0))
    confirm_frames = max(1, int(round(float(getattr(cfg, 'lead_vlead_optimism_clamp_confirm_frames', 3.0)))))

    # Record the RAW model vLead (v_ego + raw vRel) so the monotonic-decline
    # detector sees the model's own measurement stream, and the (time, internal
    # dRel) so the windowed position-derivative can be taken. `now` is THIS
    # frame's timestamp; self.last_t is not updated until after this method
    # returns, so the current time must come from `now`.
    raw_vlead = float(v_ego) + float(raw_vrel)
    self.raw_vlead_hist.append(raw_vlead)
    self.drel_hist.append((float(now), float(next_drel)))

    # Disable sentinels: an unreachable far range OR a zero gain restores exact
    # legacy publish (no clamp, no state effect beyond the cheap history above).
    if far_range_m >= 1e9 or gain <= 0.0:
      return
    if float(self.dRel) < far_range_m:
      return
    # Sustained monotonic decline: require confirm_frames+1 samples that are
    # each strictly (with a tiny noise deadband) lower than the previous. One
    # noisy frame cannot chain this; a genuinely stopping lead does frame after
    # frame. A steady/moving lead (flat or rising raw vLead) never qualifies.
    hist = list(self.raw_vlead_hist)
    if len(hist) < confirm_frames + 1:
      return
    recent = hist[-(confirm_frames + 1):]
    declining = all((recent[i + 1] - recent[i]) < -0.05 for i in range(len(recent) - 1))
    if not declining:
      return

    # Slow-lead gate on the MODEL's OWN reported vLead: the current raw vLead must
    # itself be a genuinely slow/stopping lead (at/below slow_lead_frac * ego).
    # This is the primary discriminator between CD8 (a real stopped lead the model
    # reports at ~1-5 m/s while ego is much faster) and a fast steady lead
    # suffering a transient vLead dip (CD6 vLeadK rollover: a 34 m/s lead dips to
    # ~30 m/s, ratio ~0.92, and is excluded even though it is momentarily
    # declining). It uses the model measurement directly, so it does not depend on
    # the fragile internal-dRel slope.
    slow_lead_frac = float(getattr(cfg, 'lead_vlead_optimism_clamp_slow_lead_frac', 0.5))
    if raw_vlead > float(v_ego) * slow_lead_frac:
      return

    # Position-derived vLead from the internal-filter dRel trend, taken over the
    # WHOLE confirm window (endpoint slope), NOT a single frame: a single-frame
    # internal-dRel snap (filter innovation gate) would otherwise yield a garbage
    # velocity. Needs a window that spans a real interval.
    drel_pts = list(self.drel_hist)[-(confirm_frames + 1):]
    if len(drel_pts) < confirm_frames + 1:
      return
    dt_pos = drel_pts[-1][0] - drel_pts[0][0]
    if dt_pos <= 1e-3:
      return
    d_drel_dt = (drel_pts[-1][1] - drel_pts[0][1]) / dt_pos
    # Physically sane clamp on the closure rate: a tracked forward lead's dRel
    # cannot open/close faster than a modest bound around the ego closure, so a
    # residual dRel discontinuity cannot fabricate an extreme position velocity.
    d_drel_dt = float(np.clip(d_drel_dt, -float(v_ego) - 2.0, 2.0))
    # A forward stopped-traffic lead's position-derived speed lives in [0, v_ego];
    # clamp there so the target is always a real, non-negative slow-lead speed.
    v_lead_pos = float(np.clip(float(v_ego) + d_drel_dt, 0.0, float(v_ego)))

    # Position-derived slow-lead gate (defense-in-depth alongside the raw-vLead
    # gate above): the position-derived velocity must ALSO be a real slow/stopping
    # lead (at/below slow_lead_frac * ego). A 34 m/s lead whose gap holds has
    # d(dRel)/dt ~0 so v_lead_pos ~v_ego (ratio ~1) and is excluded even across a
    # transient vLead dip; a genuine stopped lead reads ~0.
    if v_lead_pos > float(v_ego) * slow_lead_frac:
      return

    # Only ever make the published lead SLOWER / more urgent: blend the internal
    # vLead toward the (lower) position-derived value by the gain, then take the
    # min so a HIGHER position-derived estimate (noise) can never speed it up.
    clamped_vlead = self.vLead + gain * (v_lead_pos - self.vLead)
    new_vlead = min(self.vLead, clamped_vlead)
    if new_vlead < self.vLead:
      self.vLead = float(new_vlead)
      self.vLeadK = float(new_vlead)
      self.vRel = float(new_vlead - float(v_ego))

  def _update_fcw_corroboration(self, raw_drel: float, raw_vrel: float,
                                cfg: LeadResponseTuningConfig) -> None:
    """Vote on whether the raw model measurement corroborates the filtered dRel.

    Two regimes make filtered dRel sit BELOW raw by more than the tolerance:

    (1) Phantom collapse: a wrongly-adopted inward outlier whose one-way
        open-slew recovery cannot heal it. The raw stream keeps measuring the
        lead FAR AWAY and NOT closing (raw is itself safe), so the filtered
        closeness is uncorroborated and must be suppressed from FCW.

    (2) Deliberate closing-urgency pessimism: on a GENUINE fast close the
        blend/lag-comp publish path intentionally runs the filtered dRel
        several metres more pessimistic than the (optimistic) raw model x
        (road 200-13: filtered 11 m vs raw 16 m while closing 7 m/s). Here the
        raw stream ITSELF corroborates an imminent threat, so suppressing FCW
        silences a real collision course (CD2).

    The false invariant the old veto rested on was 'filtered >= raw on genuine
    threats'. That holds only for the pure closing-EMA lag; it is violated
    exactly by the deliberate pessimism above. So before the raw-vs-filter
    disagreement vote can suppress, an escape hatch checks the RAW kinematics
    directly: if raw is genuinely closing (>= RawClosingMinMps) with a raw-side
    TTC at/under RawTtcMaxS, the raw measurement corroborates the threat on its
    own and FCW is never suppressed, regardless of the filtered-vs-raw delta.
    The deep phantom (raw not closing, raw far) fails this gate and stays
    suppressed; the genuine urgency-pessimism close passes it and stays FCW
    eligible. RawClosingMinMps <= 0 OR RawTtcMaxS <= 0 disables the escape
    (exact legacy raw-vs-filter veto).

    A majority vote over a short window still bridges isolated outward
    measurement outliers (~3% of frames at close range). MinAgree <= 0 disables
    suppression entirely (legacy).
    """
    tol_m = float(getattr(cfg, 'model_lead_fcw_corrob_tol_m', 2.5))
    min_agree = int(round(float(getattr(cfg, 'model_lead_fcw_corrob_min_agree', 2.0))))
    window = int(round(float(getattr(cfg, 'model_lead_fcw_corrob_window', 3.0))))
    self.fcw_agree_hist.append(bool((raw_drel - self.dRel) <= tol_m))
    if min_agree <= 0:
      self.fcw_suppressed = False
      return

    # Raw-kinematic threat escape: the raw stream independently corroborates an
    # imminent collision, so the deliberate closing-urgency pessimism must not
    # be misread as a phantom collapse (CD2).
    raw_closing_min_mps = float(getattr(cfg, 'model_lead_fcw_corrob_raw_closing_min_mps', 1.0))
    raw_ttc_max_s = float(getattr(cfg, 'model_lead_fcw_corrob_raw_ttc_max_s', 3.5))
    raw_closing = max(0.0, -float(raw_vrel))
    if (raw_closing_min_mps > 0.0 and raw_ttc_max_s > 0.0 and
        raw_closing >= raw_closing_min_mps and
        (float(raw_drel) / max(raw_closing, 0.1)) <= raw_ttc_max_s):
      self.fcw_suppressed = False
      return

    window = int(np.clip(window, min_agree, MODEL_LEAD_FCW_CORROB_HIST_LEN))
    recent = list(self.fcw_agree_hist)[-window:]
    self.fcw_suppressed = sum(recent) < min_agree

  def get_RadarState(self, cfg: LeadResponseTuningConfig | None = None) -> dict[str, Any]:
    published_drel = float(self.dRel)
    if cfg is not None:
      # Closing-only group-delay compensation on the PUBLISHED dRel; internal
      # filter state is untouched (no feedback). Publication only ever moves
      # CLOSER, never farther, so a held/filtered lead is never more optimistic
      # than the internal state. Capped at the active regime's actual filter
      # delay so the fast-adopting path is not over-corrected at high closing.
      lag_comp_s = min(float(cfg.model_lead_filter_lag_comp_s), float(self.drel_lag_s))
      if lag_comp_s > 0.0:
        # Stopping-regime fade: below FadeLo ego speed the compensation is off,
        # above FadeHi it is full, linear ramp between. At low ego speed the
        # filter lag error is proportionally tiny (closing speeds are small) so
        # compensation buys little safety while pushing the stop point back.
        # Degenerate span (FadeHi <= FadeLo, e.g. both 0) disables the fade and
        # restores full compensation everywhere — the pessimistic direction.
        fade_hi_mps = float(cfg.model_lead_filter_lag_comp_fade_hi_mps)
        fade_lo_mps = float(cfg.model_lead_filter_lag_comp_fade_lo_mps)
        fade_span_mps = fade_hi_mps - fade_lo_mps
        blend_min_span = float(getattr(cfg, 'model_lead_blend_min_span', MODEL_LEAD_BLEND_MIN_SPAN))
        if fade_span_mps >= blend_min_span:
          v_ego_est = max(0.0, float(self.vLead) - float(self.vRel))
          lag_comp_s *= float(np.clip((v_ego_est - fade_lo_mps) / fade_span_mps, 0.0, 1.0))
      if lag_comp_s > 0.0:
        closing_mps = max(0.0, -self.vRel - float(cfg.model_lead_filter_lag_comp_deadzone_mps))
        published_drel = max(0.0, published_drel - closing_mps * lag_comp_s)

    # CD4 published-dRel single-frame OPENING-ONLY step guard (defense-in-depth).
    # Runs AFTER lag-comp on the post-lag-comp published value and stores the
    # (clamped) result, so the reference tracks the actual publication. It clamps
    # ONLY the OPENING (farther, smaller-urgency) direction: a step that moves the
    # lead CLOSER is never touched, so it can never delay or attenuate emergency
    # braking (the closing/fast-close/FCW path bypasses this rate-limit by
    # construction -- the clamp branch is unreachable for closing steps). It also
    # never touches internal filter state (publish-only, like lag-comp), so it
    # adds no feedback lag. Evaluated exactly ONCE per frame per track (keyed on
    # last_t, which the same-frame duplicate early-return path also observes) so a
    # second same-frame call reuses the first result instead of double-clamping.
    # Never binds on the first publish (last_published_drel is None) so a fresh
    # genuine cut-in publishes its true close dRel with zero attenuation.
    if cfg is not None:
      published_drel = self._apply_step_guard(published_drel, cfg)

    # CD9 corroborated-closing governor publish clamp (one-directional, publish
    # only - internal EMA/association/prediction state untouched): while the
    # governor is latched, the published vLead may not exceed the corroborated
    # closure below the ego estimate. min() semantics: it can only ever make
    # the published lead SLOWER / more urgent, never faster.
    published_vrel = float(self.vRel)
    published_vlead = float(self.vLead)
    published_vleadk = float(self.vLeadK)
    if self.governor_active and self.governor_closing_mps > 0.0:
      v_ego_est = max(0.0, float(self.vLead) - float(self.vRel))
      target_vlead = max(0.0, v_ego_est - float(self.governor_closing_mps))
      if target_vlead < published_vlead:
        published_vlead = target_vlead
        published_vleadk = target_vlead
        published_vrel = target_vlead - v_ego_est
    # Opening governor publish relax (CD9's mirror, one-directional, publish
    # only): while the raw position window PROVES sustained opening and no
    # threat veto stands (see _update_opening_governor), the published vRel is
    # floored at min(pos_opening - trust_deficit, 0.0). max() semantics: only
    # ever makes the published lead FASTER / less urgent, never past parity.
    # Mutually exclusive with the CD9 clamp by construction (CD9 latched or
    # holding vetoes the relax before it arms).
    relax_floor = self.opening_relax_vrel
    if relax_floor is not None and published_vrel < relax_floor:
      v_ego_est = max(0.0, float(self.vLead) - float(self.vRel))
      published_vrel = float(relax_floor)
      published_vlead = max(0.0, v_ego_est + published_vrel)
      published_vleadk = published_vlead
    published_closing_mps = max(0.0, -float(published_vrel))
    published_ttc_s = float(published_drel) / max(published_closing_mps, 0.1)
    self.accel_corr_calm_position_valid = bool(
      self.accel_corr_calm_position_dense_valid and
      self.missed == 0 and
      abs(float(self.aLeadK)) <= ACCEL_CORR_CALM_POSITION_MAX_ALEAD_ABS_MPS2 and
      published_closing_mps < ACCEL_CORR_CALM_POSITION_MAX_PUBLISHED_CLOSING_MPS and
      published_ttc_s > ACCEL_CORR_CALM_POSITION_MIN_PUBLISHED_TTC_S and
      published_vlead >= 0.0
    )
    if self.accel_corr_calm_position_dense_valid and not self.accel_corr_calm_position_valid:
      if self.missed != 0:
        self.accel_corr_calm_position_reason = "missed_frame"
      elif abs(float(self.aLeadK)) > ACCEL_CORR_CALM_POSITION_MAX_ALEAD_ABS_MPS2:
        self.accel_corr_calm_position_reason = "published_accel"
      elif published_closing_mps >= ACCEL_CORR_CALM_POSITION_MAX_PUBLISHED_CLOSING_MPS:
        self.accel_corr_calm_position_reason = "published_fast_close"
      elif published_ttc_s <= ACCEL_CORR_CALM_POSITION_MIN_PUBLISHED_TTC_S:
        self.accel_corr_calm_position_reason = "published_short_ttc"
      else:
        self.accel_corr_calm_position_reason = "oncoming"
    return {
      "dRel": published_drel,
      "yRel": float(self.yRel),
      "vRel": published_vrel,
      "vLead": published_vlead,
      "vLeadK": published_vleadk,
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau),
      "fcw": False,
      "fcwSuppressed": bool(self.fcw_suppressed),
      "closingGovernorRecovery": bool(self.governor_calm_recovery_mode),
      "closingGovernorRecoveryPositionClosingMps": float(self.governor_recovery_position_closing_mps or 0.0),
      "closingGovernorRecoveryVRelFloorMps": float(self.governor_recovery_vrel_floor_mps or 0.0),
      "closingGovernorRecoveryNumericValid": bool(
        self.governor_calm_recovery_mode and
        self.governor_recovery_position_closing_mps is not None and
        self.governor_recovery_vrel_floor_mps is not None
      ),
      "steadyParityCandidateValid": bool(self.steady_parity_candidate_valid),
      "steadyParityPositionSlopeMps": float(self.steady_parity_position_slope_mps),
      "steadyParityVRelFloorMps": float(self.steady_parity_vrel_floor_mps),
      "steadyParityHeld": bool(self.steady_parity_held),
      "steadyParityCurrentThreat": bool(self.steady_parity_current_threat),
      "accelCorrCalmPositionValid": bool(self.accel_corr_calm_position_valid),
      "accelCorrCalmPositionSlopeMps": float(self.accel_corr_calm_position_slope_mps),
      "modelProb": float(self.modelProb),
      "status": True,
      "radar": False,
      "radarTrackId": int(self.identifier),
      "dPath": float(self.dPath),
      "vLat": float(self.vLat),
    }

  def _apply_step_guard(self, published_drel: float, cfg: LeadResponseTuningConfig) -> float:
    abs_m = float(getattr(cfg, 'model_lead_step_guard_abs_m', 0.0))
    frac = float(getattr(cfg, 'model_lead_step_guard_frac', 0.0))
    # Disable sentinel: either term at/below zero restores exact legacy publish.
    if abs_m <= 0.0 or frac <= 0.0:
      self.last_published_drel = float(published_drel)
      self._step_guard_eval_t = None
      self._step_guard_last_out = None
      return float(published_drel)

    # Once-per-frame idempotency: a duplicate slot calls get_RadarState twice in
    # one frame (once in update(), once via the already-updated early-return).
    # Both calls share last_t (the track was updated this frame), so on the
    # second same-frame call reuse the first call's clamped output instead of
    # comparing published_drel against the value the first call just stored (which
    # would let a genuine opening continuation get clamped a second time).
    now_key = float(self.last_t)
    if self._step_guard_eval_t is not None and self._step_guard_eval_t == now_key \
       and self._step_guard_last_out is not None:
      return float(self._step_guard_last_out)

    prev = self.last_published_drel
    out = float(published_drel)
    # Only guard when the same track persisted with a prior published value
    # (never bind on the first publish -> a fresh cut-in is unclamped on frame 1).
    if prev is not None:
      opening_step = out - float(prev)
      if opening_step > 0.0:  # OPENING (farther) only; closing is never clamped.
        bound = max(abs_m, frac * out)
        if opening_step > bound:
          out = float(prev) + bound

    self.last_published_drel = out
    self._step_guard_eval_t = now_key
    self._step_guard_last_out = out
    return out


class ModelLeadTracker:
  def __init__(self, params: Params | None = None):
    self.params = params if params is not None else Params()
    self._cfg = LeadResponseTuningConfig.defaults()
    self._last_param_refresh_t = -1e9
    self._tracks: dict[int, ModelLeadTrack] = {}
    self._updated_track_ids: set[int] = set()
    self._next_identifier = MODEL_LEAD_TRACK_ID_START
    self._frame_active = False

  @property
  def tracks(self) -> dict[int, ModelLeadTrack]:
    return self._tracks

  def begin_frame(self, now: float) -> None:
    now = float(now)
    self._refresh_config(now)
    self._updated_track_ids.clear()
    self._frame_active = True

  def end_frame(self) -> None:
    if not self._frame_active:
      return
    for identifier, track in list(self._tracks.items()):
      if identifier not in self._updated_track_ids:
        track.missed += 1
        track.steady_parity_current_threat = False
        track.governor_calm_recovery_mode = False
        track.governor_calm_recovery_applied = False
        track.governor_recovery_position_closing_mps = None
        track.governor_recovery_vrel_floor_mps = None
        # A missed frame means no fresh position evidence: never carry a stale
        # opening relax onto a held/coasted publish (less-urgent direction).
        track._clear_opening_relax(rearm_after_t=track.last_t)
        track._clear_steady_parity("missed_frame")
        track._clear_accel_corr_calm_position("missed_frame")
      if track.missed > MODEL_LEAD_TRACK_MAX_MISSES:
        self._tracks.pop(identifier, None)
    self._frame_active = False

  def _refresh_config(self, now: float) -> None:
    if (float(now) - self._last_param_refresh_t) < MODEL_LEAD_PARAM_REFRESH_DT_S:
      return
    try:
      self._cfg = read_lead_response_tuning_config(self.params)
    except Exception:
      self._cfg = LeadResponseTuningConfig.defaults()
    self._last_param_refresh_t = float(now)

  def _new_identifier(self) -> int:
    identifier = self._next_identifier
    self._next_identifier -= 1
    return identifier

  def _association_score(self, track: ModelLeadTrack, lead_dict: dict[str, Any], now: float, lead_slot: int) -> float | None:
    raw_drel = _finite_float(lead_dict.get("dRel"))
    raw_dpath = _finite_float(lead_dict.get("dPath", lead_dict.get("yRel")))
    raw_yrel = _finite_float(lead_dict.get("yRel"))
    raw_vrel = _finite_float(lead_dict.get("vRel"))
    pred_drel = track.predict_drel(now)
    same_slot_gate = 35.0 if track.last_slot == int(lead_slot) else float(self._cfg.model_lead_filter_assoc_drel_m)
    drel_gate = max(same_slot_gate, 0.18 * max(pred_drel, raw_drel, 1.0))
    drel_err = abs(pred_drel - raw_drel)
    path_err = abs(track.dPath - raw_dpath)
    y_err = abs(track.yRel - raw_yrel)
    vrel_err = abs(track.vRel - raw_vrel)
    # CD4: gate lateral continuity PRIMARILY on the path-relative dPath, which
    # stays lane-relevant through curves, and give the RAW yRel a separate, LARGER
    # tolerance. The legacy shared 3.0 m gate keyed on raw yRel, which a
    # curve-induced yRel drift (+1.93 -> -8.3 m, dPath still < 0.9 m) repeatedly
    # blew on the SAME physical lead, churning the published track id. Rollback:
    # set both knobs to 3.0 to restore the legacy shared 3.0 m gate.
    dpath_gate = float(self._cfg.model_lead_assoc_dpath_gate_m)
    # Closing-side safety (amendment 1): the raw-y widening applies only to the
    # OPENING/lane-relevant case. When the candidate is CLOSING (raw vRel < 0) hold
    # the raw-y tolerance at the legacy 3.0 m so a slow-closing near lead whose
    # dPath momentarily reads inside the dPath gate while its true yRel is 3-7 m
    # cannot be masked into a farther track. Fast-close/short-TTC/cut-in cases are
    # additionally protected downstream (closer_safety_candidate + per-track
    # fast-close/closing-urgency adoption), so this is belt-and-suspenders.
    y_raw_tol = float(self._cfg.model_lead_assoc_y_raw_tol_m)
    if raw_vrel < 0.0:
      y_raw_tol = min(y_raw_tol, MODEL_LEAD_ASSOC_Y_GATE_M)
    # CD4 rollback-completeness: the fix dropped the legacy y_err clause from the
    # same-frame-duplicate merge. To let the documented rollback sentinel (set
    # ModelLeadAssocDPathGateM AND ModelLeadAssocYRawTolM both to 3.0 =
    # MODEL_LEAD_ASSOC_Y_GATE_M) FULLY restore the pre-fix churn, gate that clause on
    # the SAME knob. Key on the CONFIGURED raw-y tolerance (NOT the vrel-narrowed
    # y_raw_tol above, which is 3.0 for any closing candidate even at the default):
    # only when the configured knob itself is at/below the legacy 3.0 m gate (the
    # rollback sentinel) require y_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M for a
    # same-frame duplicate, exactly as the pre-CD4 code did. At the default (7.0) the
    # clause stays dropped, so default behavior is bit-identical to a67737c0b.
    cfg_y_raw_tol = float(self._cfg.model_lead_assoc_y_raw_tol_m)
    duplicate_y_ok = (cfg_y_raw_tol > MODEL_LEAD_ASSOC_Y_GATE_M or
                      y_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M)
    same_frame_duplicate = (
      track.identifier in self._updated_track_ids and
      track.last_slot != int(lead_slot) and
      path_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M and
      duplicate_y_ok and
      vrel_err <= MODEL_LEAD_DUPLICATE_VREL_GATE_MPS
    )
    closer_safety_candidate = (
      raw_drel < (pred_drel - MODEL_LEAD_DUPLICATE_CLOSER_KEEP_SEPARATE_M) and
      raw_vrel < -MODEL_LEAD_DUPLICATE_CLOSING_KEEP_SEPARATE_MPS
    )
    if same_frame_duplicate and not closer_safety_candidate:
      drel_gate = max(drel_gate, MODEL_LEAD_DUPLICATE_DREL_GATE_M)

    if drel_err > drel_gate or path_err > dpath_gate or y_err > y_raw_tol:
      return None
    if vrel_err > MODEL_LEAD_ASSOC_VREL_GATE_MPS:
      return None
    updated_penalty = 0.25 if track.identifier in self._updated_track_ids else 0.0
    # Normalize each term by its own gate so the closest-dPath track still wins
    # and the widened raw-y contribution is softened (not weighted at the old
    # 3.0 m scale, which would let raw-yRel drift dominate the ranking).
    return float(
      drel_err / max(drel_gate, 1e-3) +
      path_err / max(dpath_gate, 1e-3) +
      y_err / max(y_raw_tol, 1e-3) +
      vrel_err / MODEL_LEAD_ASSOC_VREL_GATE_MPS +
      updated_penalty
    )

  def _match_track(self, lead_dict: dict[str, Any], now: float, lead_slot: int) -> ModelLeadTrack | None:
    best: tuple[float, ModelLeadTrack] | None = None
    for track in self._tracks.values():
      score = self._association_score(track, lead_dict, now, lead_slot)
      if score is None:
        continue
      if best is None or score < best[0]:
        best = (score, track)
    if best is not None:
      return best[1]

    raw_drel = _finite_float(lead_dict.get("dRel"))
    raw_dpath = _finite_float(lead_dict.get("dPath", lead_dict.get("yRel")))
    raw_yrel = _finite_float(lead_dict.get("yRel"))
    raw_vrel = _finite_float(lead_dict.get("vRel"))
    same_slot_best: tuple[float, ModelLeadTrack] | None = None
    for track in self._tracks.values():
      if track.last_slot != int(lead_slot):
        continue
      pred_drel = track.predict_drel(now)
      drel_err = abs(pred_drel - raw_drel)
      path_err = abs(track.dPath - raw_dpath)
      y_err = abs(track.yRel - raw_yrel)
      vrel_err = abs(track.vRel - raw_vrel)
      # CD4: same dPath-primary / raw-y-widened treatment as _association_score so
      # the recovery path cannot reject a genuine same-slot continuation on a
      # curve-induced yRel excursion either. Raw-y stays at the legacy 3.0 m when
      # the candidate is closing (see _association_score amendment-1 rationale).
      dpath_gate = float(self._cfg.model_lead_assoc_dpath_gate_m)
      y_raw_tol = float(self._cfg.model_lead_assoc_y_raw_tol_m)
      if raw_vrel < 0.0:
        y_raw_tol = min(y_raw_tol, MODEL_LEAD_ASSOC_Y_GATE_M)
      if (
        drel_err > MODEL_LEAD_SAME_SLOT_RECOVER_DREL_GATE_M or
        path_err > dpath_gate or
        y_err > y_raw_tol or
        vrel_err > MODEL_LEAD_ASSOC_VREL_GATE_MPS
      ):
        continue
      score = drel_err + 4.0 * path_err + 4.0 * y_err + vrel_err
      if same_slot_best is None or score < same_slot_best[0]:
        same_slot_best = (score, track)
    return None if same_slot_best is None else same_slot_best[1]

  def update_from_vision(self, lead_dict: dict[str, Any], *, now: float | None, v_ego: float,
                         lead_slot: int = 0) -> dict[str, Any]:
    now = 0.0 if now is None else float(now)
    if not self._frame_active:
      self._refresh_config(now)

    track = self._match_track(lead_dict, now, lead_slot)
    if track is None:
      track = ModelLeadTrack.from_lead_dict(self._new_identifier(), lead_dict, now, lead_slot)
      self._tracks[track.identifier] = track
      while len(self._tracks) > MODEL_LEAD_TRACK_MAX_COUNT:
        stale_identifier = max(self._tracks.values(), key=lambda t: (t.missed, -t.age)).identifier
        self._tracks.pop(stale_identifier, None)

    if track.identifier in self._updated_track_ids:
      return track.get_RadarState(self._cfg)

    self._updated_track_ids.add(track.identifier)
    return track.update(lead_dict, now, v_ego, self._cfg, lead_slot)


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead
    self.measured = measured   # measured or estimate

    # computed velocity and accelerations
    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    # Learn if constant acceleration
    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": self.is_potential_fcw(model_prob),
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than 0.75, are almost always glitches on toyota radars
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track]):
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This isn't exactly right, but it's a good heuristic
    return prob_d * prob_y * prob_v

  track = max(tracks.values(), key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 10) or (v_ego + track.vRel > 3)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_msg.prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def _get_model_path_xy(model_msg: capnp._DynamicStructReader) -> tuple[np.ndarray, np.ndarray] | None:
  try:
    path_x = np.asarray(model_msg.position.x, dtype=float)
    path_y = np.asarray(model_msg.position.y, dtype=float)
  except Exception:
    return None

  if path_x.size < 2 or path_x.size != path_y.size:
    return None

  finite = np.isfinite(path_x) & np.isfinite(path_y)
  path_x = path_x[finite]
  path_y = path_y[finite]
  if path_x.size < 2:
    return None

  # Model path points are published in increasing longitudinal order; guard against
  # malformed inputs to keep interpolation stable.
  if np.any(np.diff(path_x) < 0.0):
    order = np.argsort(path_x)
    path_x = path_x[order]
    path_y = path_y[order]

  return path_x, path_y


def get_path_y_rel(model_msg: capnp._DynamicStructReader, d_rel: float) -> float:
  path_xy = _get_model_path_xy(model_msg)
  if path_xy is None or not math.isfinite(d_rel):
    return 0.0

  path_x, path_y = path_xy
  x_device = float(np.clip(d_rel + RADAR_TO_CAMERA, path_x[0], path_x[-1]))
  return float(-np.interp(x_device, path_x, path_y))


def get_path_relative_lead_metrics(lead_dict: dict[str, Any], model_msg: capnp._DynamicStructReader,
                                   lead_msg: capnp._DynamicStructReader | None = None) -> tuple[float, float]:
  y_rel = float(lead_dict.get("yRel", 0.0) or 0.0)
  d_rel = float(lead_dict.get("dRel", 0.0) or 0.0)
  d_path = y_rel - get_path_y_rel(model_msg, d_rel)
  v_lat = 0.0

  if lead_msg is not None:
    try:
      times = np.asarray(lead_msg.t, dtype=float)
      xs = np.asarray(lead_msg.x, dtype=float)
      ys = np.asarray(lead_msg.y, dtype=float)
      valid = np.isfinite(times) & np.isfinite(xs) & np.isfinite(ys)
      if np.any(valid):
        times = times[valid]
        xs = xs[valid]
        ys = ys[valid]
        future_idxs = np.where(times > (times[0] + 1e-3))[0]
        if future_idxs.size > 0:
          i = int(future_idxs[0])
          future_d_rel = float(xs[i] - RADAR_TO_CAMERA)
          future_y_rel = float(-ys[i])
          future_d_path = future_y_rel - get_path_y_rel(model_msg, future_d_rel)
          dt = float(times[i] - times[0])
          if dt > 1e-3:
            v_lat = float((future_d_path - d_path) / dt)
    except Exception:
      v_lat = 0.0

  return d_path, v_lat


def add_path_relative_lead_metrics(lead_dict: dict[str, Any], model_msg: capnp._DynamicStructReader,
                                   lead_msg: capnp._DynamicStructReader | None = None) -> dict[str, Any]:
  d_path, v_lat = get_path_relative_lead_metrics(lead_dict, model_msg, lead_msg)
  lead_dict["dPath"] = float(d_path)
  lead_dict["vLat"] = float(v_lat)
  return lead_dict


def _is_lead_prob_accepted(lead_prob: float, prev_latched: bool,
                           prob_enter: float, prob_exit: float) -> bool:
  """Asymmetric Schmitt trigger on vision lead prob. Previous latch state decides
  whether the current prob clears the enter or the exit threshold. Defaults to
  the classic `prob > 0.5` behavior when both thresholds collapse to 0.5."""
  if not math.isfinite(lead_prob):
    return False
  lower = min(prob_enter, prob_exit)
  upper = max(prob_enter, prob_exit)
  return lead_prob > (lower if prev_latched else upper)


def _hyundai_scc_track_without_lateral(CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> bool:
  return bool(
    getattr(CP, "brand", "") == "hyundai" and
    (
      int(getattr(CP_SP, "flags", 0)) & int(HyundaiFlagsSP.ENHANCED_SCC) or
      int(getattr(CP, "flags", 0)) & int(HyundaiFlags.CAMERA_SCC | HyundaiFlags.CANFD_CAMERA_SCC)
    )
  )


def _lead_msg_y_rel(lead_msg: capnp._DynamicStructReader) -> float:
  try:
    y_rel = float(-lead_msg.y[0])
    return y_rel if math.isfinite(y_rel) else math.nan
  except Exception:
    return math.nan


def _select_prob_dropout_track(v_ego: float, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
                               CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> Track | None:
  """Retain a measured close/closing track when an already-latched model lead prob dips."""
  if float(v_ego) < LEAD_TRACK_PROB_DROPOUT_MIN_SPEED_MPS:
    return None

  allow_missing_lateral = _hyundai_scc_track_without_lateral(CP, CP_SP)
  fallback_y_rel = _lead_msg_y_rel(lead_msg)
  best: tuple[float, Track] | None = None
  near_drel = max(LEAD_TRACK_PROB_DROPOUT_NEAR_DREL_M,
                  LEAD_TRACK_PROB_DROPOUT_NEAR_HEADWAY_S * max(float(v_ego), 0.0))

  for track in tracks.values():
    d_rel = _finite_float(getattr(track, "dRel", math.nan), math.nan)
    y_rel = _finite_float(getattr(track, "yRel", math.nan), math.nan)
    v_rel = _finite_float(getattr(track, "vRel", math.nan), math.nan)
    measured = bool(getattr(track, "measured", False))
    if (not measured or not math.isfinite(d_rel) or not math.isfinite(v_rel)):
      continue
    if not math.isfinite(y_rel):
      if not allow_missing_lateral or not math.isfinite(fallback_y_rel):
        continue
      y_rel = fallback_y_rel
    if d_rel < LEAD_TRACK_PROB_DROPOUT_MIN_DREL_M or abs(y_rel) > LEAD_TRACK_PROB_DROPOUT_CENTER_Y_ABS_M:
      continue

    closing_speed = max(0.0, -v_rel)
    near_following = d_rel <= near_drel and v_rel <= LEAD_TRACK_PROB_DROPOUT_PULLING_AWAY_MAX_MPS
    urgent_ttc = d_rel / max(closing_speed, 1e-3)
    urgent_closing = (
      closing_speed >= LEAD_TRACK_PROB_DROPOUT_URGENT_CLOSING_MPS and
      urgent_ttc <= LEAD_TRACK_PROB_DROPOUT_URGENT_TTC_S
    )
    if not (near_following or urgent_closing):
      continue

    score = d_rel - 2.0 * closing_speed
    if best is None or score < best[0]:
      best = (score, track)

  return None if best is None else best[1]


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, CP: structs.CarParams, CP_SP: structs.CarParamsSP, model_msg: capnp._DynamicStructReader,
             low_speed_override: bool = True, prev_latched: bool = False,
             prob_enter: float = 0.5, prob_exit: float = 0.5,
             model_lead_tracker: ModelLeadTracker | None = None,
             lead_slot: int = 0, now: float | None = None) -> dict[str, Any]:
  # Determine leads, this is where the essential logic happens
  prob_accepted = _is_lead_prob_accepted(float(lead_msg.prob), prev_latched, prob_enter, prob_exit)
  if len(tracks) > 0 and ready and prob_accepted:
    track = match_vision_to_track(v_ego, lead_msg, tracks)
  else:
    track = None

  lead_dict = {'status': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_msg.prob)
    lead_dict = get_custom_yrel(CP, CP_SP, lead_dict, lead_msg)
  elif (track is None) and ready and prob_accepted:
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)
    if model_lead_tracker is not None:
      lead_dict = add_path_relative_lead_metrics(lead_dict, model_msg, lead_msg)
      lead_dict = model_lead_tracker.update_from_vision(lead_dict, now=now, v_ego=v_ego, lead_slot=lead_slot)
  elif (track is None) and ready and prev_latched and not prob_accepted:
    dropout_track = _select_prob_dropout_track(v_ego, tracks, lead_msg, CP, CP_SP)
    if dropout_track is not None:
      lead_dict = dropout_track.get_RadarState(lead_msg.prob)
      lead_dict = get_custom_yrel(CP, CP_SP, lead_dict, lead_msg)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  if lead_dict.get("status", False):
    lead_dict = add_path_relative_lead_metrics(lead_dict, model_msg, lead_msg if ready else None)

  return lead_dict


def get_custom_yrel(CP: structs.CarParams, CP_SP: structs.CarParamsSP, lead_dict: dict[str, Any],
                    lead_msg: capnp._DynamicStructReader) -> dict[str, Any]:
  if CP.brand == "hyundai" and (CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC or
                                CP.flags & (HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC)):
    lead_dict['yRel'] = float(-lead_msg.y[0])

  return lead_dict


class RadarD:
  LEAD_PROB_REFRESH_S = 1.0

  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParams, delay: float = 0.0):
    self.CP = CP
    self.CP_SP = CP_SP

    self.current_time = 0.0

    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)
    self.model_lead_tracker = ModelLeadTracker()

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

    self._lead_latched = [False, False]
    self._params = Params()
    self._lead_prob_enter = 0.5
    self._lead_prob_exit = 0.5
    self._last_prob_refresh_t = 0.0

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks(service_list=['modelV2', 'liveTracks'])
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']
    replay_inputs = self.radar_state.replayInputs
    replay_inputs.valid = True
    replay_inputs.version = _REPLAY_INPUTS_VERSION
    replay_inputs.modelV2MonoTimeNs = sm.logMonoTime['modelV2']
    replay_inputs.carStateMonoTimeNs = sm.logMonoTime['carState']
    replay_inputs.liveTracksMonoTimeNs = sm.logMonoTime['liveTracks']

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    self._maybe_refresh_lead_prob_thresholds()

    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      self.model_lead_tracker.begin_frame(self.current_time)
      lead_one = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego,
                          self.CP, self.CP_SP, sm['modelV2'], low_speed_override=True,
                          prev_latched=self._lead_latched[0],
                          prob_enter=self._lead_prob_enter, prob_exit=self._lead_prob_exit,
                          model_lead_tracker=self.model_lead_tracker,
                          lead_slot=0, now=self.current_time)
      self._populate_governor_replay_debug(replay_inputs.leadOneGovernor, lead_one)
      lead_two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego,
                          self.CP, self.CP_SP, sm['modelV2'], low_speed_override=False,
                          prev_latched=self._lead_latched[1],
                          prob_enter=self._lead_prob_enter, prob_exit=self._lead_prob_exit,
                          model_lead_tracker=self.model_lead_tracker,
                          lead_slot=1, now=self.current_time)
      self._populate_governor_replay_debug(replay_inputs.leadTwoGovernor, lead_two)
      self.radar_state.leadOne = lead_one
      self.radar_state.leadTwo = lead_two
      self._lead_latched = [bool(lead_one.get("status", False)), bool(lead_two.get("status", False))]
      self.model_lead_tracker.end_frame()

  def _populate_governor_replay_debug(self, debug, lead: dict[str, Any]) -> None:
    """Snapshot the exact model-track governor state used for this lead."""
    radar_track_id = int(lead.get("radarTrackId", -1)) if lead.get("status", False) else -1
    track = self.model_lead_tracker.tracks.get(radar_track_id)
    debug.radarTrackId = radar_track_id
    if track is None:
      debug.valid = False
      return

    debug.valid = True
    debug.active = bool(track.governor_active)
    debug.closingMps = float(track.governor_closing_mps)
    debug.holdRemainingS = max(0.0, float(track.governor_hold_until_t) - self.current_time)
    debug.reason = str(track.governor_reason)
    debug.threatCorroborated = bool(track.governor_threat_corroborated)
    debug.calmRecoveryMode = bool(track.governor_calm_recovery_mode)
    debug.calmRecoveryApplied = bool(track.governor_calm_recovery_applied)
    if track.governor_recovery_position_closing_mps is not None:
      debug.recoveryPositionClosingMps = float(track.governor_recovery_position_closing_mps)
      debug.recoveryPositionClosingValid = True
    if track.governor_recovery_vrel_floor_mps is not None:
      debug.recoveryVRelFloorMps = float(track.governor_recovery_vrel_floor_mps)
      debug.recoveryVRelFloorValid = True
    debug.steadyParityCandidateValid = bool(track.steady_parity_candidate_valid)
    debug.steadyParityPositionSlopeMps = float(track.steady_parity_position_slope_mps)
    debug.steadyParityVRelFloorMps = float(track.steady_parity_vrel_floor_mps)
    debug.steadyParityHeld = bool(track.steady_parity_held)
    debug.steadyParitySampleCount = int(track.steady_parity_sample_count)
    debug.steadyParityWindowSpanS = float(track.steady_parity_window_span_s)
    debug.steadyParityMaxSampleGapS = float(track.steady_parity_max_sample_gap_s)
    debug.steadyParityReason = str(track.steady_parity_reason)
    debug.accelCorrCalmPositionValid = bool(track.accel_corr_calm_position_valid)
    debug.accelCorrCalmPositionSlopeMps = float(track.accel_corr_calm_position_slope_mps)
    debug.accelCorrCalmPositionReason = str(track.accel_corr_calm_position_reason)

  def _maybe_refresh_lead_prob_thresholds(self) -> None:
    if (self.current_time - self._last_prob_refresh_t) < self.LEAD_PROB_REFRESH_S:
      return
    self._last_prob_refresh_t = self.current_time
    def _read(key: str, default: float) -> float:
      try:
        raw = self._params.get(key)
        if raw is None:
          return float(default)
        val = float(raw)
        return val if math.isfinite(val) else float(default)
      except Exception:
        return float(default)
    self._lead_prob_enter = max(0.0, min(1.0, _read("Longitudinal.LiveTune.LeadProbEnter", 0.6)))
    self._lead_prob_exit = max(0.0, min(1.0, _read("Longitudinal.LiveTune.LeadProbExit", 0.35)))

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(6, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(Params().get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("radard got CarParamsSP")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'liveTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP, CP_SP, CP.radarDelay)

  while 1:
    sm.update()

    RD.update(sm, sm['liveTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
