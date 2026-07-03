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

    urgency = 0.0 if fast_closing else self._closing_urgency(raw_drel, raw_vrel, innovation_m, cfg)

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
      # Restricted to low ego speed (MaxEgoMps, 0 disables) where a wrongly
      # optimistic dRel costs little stopping distance and the phantom-collapse
      # ratchet does its damage.
      # MaxEgoMps > 0.0 guard makes the documented kill switch exact: with the
      # knob at 0.0, 'v_ego <= 0.0' would still arm the recovery at standstill,
      # so A/B isolation with OpenRecoveryMaxEgoMps=0 would not be bit-exact
      # legacy while stopped.
      open_recovery_required = max(1, int(round(float(cfg.model_lead_filter_open_recovery_confirm_frames))))
      open_recovery_max_ego_mps = float(cfg.model_lead_filter_open_recovery_max_ego_mps)
      if (innovation_m > 0.0 and
          self.opening_confirm_frames >= open_recovery_required and
          open_recovery_max_ego_mps > 0.0 and
          float(v_ego) <= open_recovery_max_ego_mps):
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
    accel_alpha = _ema_alpha(dt_s, MODEL_LEAD_ACCEL_TAU_S)
    prob_alpha = _ema_alpha(dt_s, MODEL_LEAD_PROB_TAU_S)

    self.dRel = float(max(0.0, next_drel))
    self._update_fcw_corroboration(raw_drel, raw_vrel, cfg)
    self.yRel = float(self.yRel + lat_alpha * (raw_yrel - self.yRel))
    self.dPath = float(self.dPath + lat_alpha * (raw_dpath - self.dPath))
    self.vLat = float(self.vLat + lat_alpha * (raw_vlat - self.vLat))
    self.vRel = float(self.vRel + vrel_alpha * (raw_vrel - self.vRel))
    self.vLead = float(v_ego + self.vRel)
    self.vLeadK = self.vLead
    self.aLeadK = float(self.aLeadK + accel_alpha * (raw_alead - self.aLeadK))
    self.aLeadTau = 0.3
    self.modelProb = float(self.modelProb + prob_alpha * (raw_prob - self.modelProb))
    self.last_t = float(now)
    self.last_slot = int(lead_slot)
    self.age += 1
    self.missed = 0
    return self.get_RadarState(cfg)

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
    return {
      "dRel": published_drel,
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau),
      "fcw": False,
      "fcwSuppressed": bool(self.fcw_suppressed),
      "modelProb": float(self.modelProb),
      "status": True,
      "radar": False,
      "radarTrackId": int(self.identifier),
      "dPath": float(self.dPath),
      "vLat": float(self.vLat),
    }


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
    same_frame_duplicate = (
      track.identifier in self._updated_track_ids and
      track.last_slot != int(lead_slot) and
      path_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M and
      y_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M and
      vrel_err <= MODEL_LEAD_DUPLICATE_VREL_GATE_MPS
    )
    closer_safety_candidate = (
      raw_drel < (pred_drel - MODEL_LEAD_DUPLICATE_CLOSER_KEEP_SEPARATE_M) and
      raw_vrel < -MODEL_LEAD_DUPLICATE_CLOSING_KEEP_SEPARATE_MPS
    )
    if same_frame_duplicate and not closer_safety_candidate:
      drel_gate = max(drel_gate, MODEL_LEAD_DUPLICATE_DREL_GATE_M)

    if drel_err > drel_gate or path_err > MODEL_LEAD_ASSOC_Y_GATE_M or y_err > MODEL_LEAD_ASSOC_Y_GATE_M:
      return None
    if vrel_err > MODEL_LEAD_ASSOC_VREL_GATE_MPS:
      return None
    updated_penalty = 0.25 if track.identifier in self._updated_track_ids else 0.0
    return float(
      drel_err / max(drel_gate, 1e-3) +
      path_err / MODEL_LEAD_ASSOC_Y_GATE_M +
      y_err / MODEL_LEAD_ASSOC_Y_GATE_M +
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
      if (
        drel_err > MODEL_LEAD_SAME_SLOT_RECOVER_DREL_GATE_M or
        path_err > MODEL_LEAD_ASSOC_Y_GATE_M or
        y_err > MODEL_LEAD_ASSOC_Y_GATE_M or
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
      lead_two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego,
                          self.CP, self.CP_SP, sm['modelV2'], low_speed_override=False,
                          prev_latched=self._lead_latched[1],
                          prob_enter=self._lead_prob_enter, prob_exit=self._lead_prob_exit,
                          model_lead_tracker=self.model_lead_tracker,
                          lead_slot=1, now=self.current_time)
      self.radar_state.leadOne = lead_one
      self.radar_state.leadTwo = lead_two
      self._lead_latched = [bool(lead_one.get("status", False)), bool(lead_two.get("status", False))]
      self.model_lead_tracker.end_frame()

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
