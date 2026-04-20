#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

try:
  from openpilot.common.params import Params
except Exception:
  # Local test/dev fallback when openpilot runtime deps are unavailable.
  class Params:  # type: ignore[no-redef]
    def get(self, _key: str):
      return None

    def put(self, _key: str, _value: Any) -> None:
      raise KeyError(_key)

    def remove(self, _key: str) -> None:
      raise KeyError(_key)


@dataclass(frozen=True)
class LeadResponseTuneSpec:
  attr: str
  key: str
  cli_name: str
  label: str
  default: float
  minimum: float
  maximum: float
  description: str

  def clamp(self, value: float) -> float:
    return float(min(max(float(value), self.minimum), self.maximum))


LEAD_RESPONSE_TUNE_SPECS = (
  LeadResponseTuneSpec(
    attr="lead_preview_strength",
    key="Longitudinal.LiveTune.LeadPreviewStrength",
    cli_name="lead-preview-strength",
    label="preview_strength",
    default=1.8,
    minimum=0.0,
    maximum=2.0,
    description="Scale factor for how early a newly recognized slower lead starts shaping decel.",
  ),
  LeadResponseTuneSpec(
    attr="lead_preview_gap_min_m",
    key="Longitudinal.LiveTune.LeadPreviewGapMinM",
    cli_name="lead-preview-gap-min-m",
    label="preview_gap_min_m",
    default=1.5,
    minimum=0.0,
    maximum=10.0,
    description="Minimum extra slack above nominal headway before preview logic activates.",
  ),
  LeadResponseTuneSpec(
    attr="lead_preview_max_buffer_m",
    key="Longitudinal.LiveTune.LeadPreviewMaxBufferM",
    cli_name="lead-preview-max-buffer-m",
    label="preview_max_buffer_m",
    default=12.0,
    minimum=0.0,
    maximum=25.0,
    description="Upper cap on how much closer the previewed lead obstacle can be pulled.",
  ),
  LeadResponseTuneSpec(
    attr="lead_acquire_window_s",
    key="Longitudinal.LiveTune.LeadAcquireWindowS",
    cli_name="lead-acquire-window-s",
    label="lead_acquire_window_s",
    default=1.25,
    minimum=0.0,
    maximum=3.0,
    description="Short boost window after a lead appears or jumps materially closer/slower.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_strength",
    key="Longitudinal.LiveTune.GapReclaimStrength",
    cli_name="gap-reclaim-strength",
    label="reclaim_strength",
    default=1.5,
    minimum=0.0,
    maximum=2.0,
    description="Scale factor for how eagerly ACC closes a safe extra gap when the lead is pulling away.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_gap_min_m",
    key="Longitudinal.LiveTune.GapReclaimGapMinM",
    cli_name="gap-reclaim-gap-min-m",
    label="reclaim_gap_min_m",
    default=0.0,
    minimum=0.0,
    maximum=10.0,
    description="Minimum extra slack above nominal headway before gap reclaim is allowed.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_max_accel",
    key="Longitudinal.LiveTune.GapReclaimMaxAccel",
    cli_name="gap-reclaim-max-accel",
    label="reclaim_max_accel",
    default=0.24,
    minimum=0.0,
    maximum=0.75,
    description="Cap on the positive accel floor used to close a safe extra gap.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_duration_s",
    key="Longitudinal.LiveTune.CutInSettleDurationS",
    cli_name="cutin-settle-duration-s",
    label="cutin_settle_duration_s",
    default=7.0,
    minimum=0.0,
    maximum=12.0,
    description="Grace-window length for a benign cut-in before the planner fully returns to nominal headway.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_max_decel",
    key="Longitudinal.LiveTune.CutInSettleMaxDecel",
    cli_name="cutin-settle-max-decel",
    label="cutin_settle_max_decel",
    default=0.30,
    minimum=0.0,
    maximum=0.80,
    description="Maximum braking magnitude allowed during the cut-in grace window after it ramps in.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_max_closing_speed_mps",
    key="Longitudinal.LiveTune.CutInSettleMaxClosingSpeedMps",
    cli_name="cutin-settle-max-closing-speed-mps",
    label="cutin_settle_max_closing_speed_mps",
    default=2.5,
    minimum=0.5,
    maximum=6.0,
    description="Largest ego-minus-lead closing speed that can still qualify for cut-in grace.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_accel_bias_mps2",
    key="Longitudinal.LiveTune.CutInSettleAccelBiasMps2",
    cli_name="cutin-settle-accel-bias",
    label="cutin_settle_accel_bias",
    default=0.20,
    minimum=0.0,
    maximum=0.30,
    description="Positive accel bias added to settle floor to counteract EV regen braking on coast.",
  ),
  LeadResponseTuneSpec(
    attr="virtual_lead_slow_tau_s",
    key="Longitudinal.LiveTune.VirtualLeadSlowTauS",
    cli_name="virtual-lead-slow-tau",
    label="vl_slow_tau",
    default=1.00,
    minimum=0.10,
    maximum=3.0,
    description="EMA time constant for lead kinematics (aLeadK) in the safe/noise-rejection direction. "
                "Lower = faster response to lead accel changes, more model noise passed through. "
                "Sign transitions (decel-to-accel) always use a faster fixed tau regardless of this value.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_tau_close_s",
    key="Longitudinal.LiveTune.DRelFilterTauCloseS",
    cli_name="drel-filter-tau-close",
    label="drel_tau_close",
    default=0.30,
    minimum=0.05,
    maximum=2.0,
    description="dRel filter time constant for closing (lead appears nearer). Lower = faster safety response.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_tau_open_s",
    key="Longitudinal.LiveTune.DRelFilterTauOpenS",
    cli_name="drel-filter-tau-open",
    label="drel_tau_open",
    default=1.00,
    minimum=0.10,
    maximum=5.0,
    description="dRel filter time constant for opening (lead appears farther). Higher = more noise rejection.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_open_slew_max_mps",
    key="Longitudinal.LiveTune.DRelFilterOpenSlewMaxMps",
    cli_name="drel-filter-open-slew-max-mps",
    label="drel_open_slew_max_mps",
    default=1.25,
    minimum=0.25,
    maximum=5.0,
    description="Max opening-side dRel motion the filter will admit per second before correction.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_innovation_gate_m",
    key="Longitudinal.LiveTune.DRelFilterInnovationGateM",
    cli_name="drel-filter-ig",
    label="drel_ig",
    default=30.0,
    minimum=5.0,
    maximum=60.0,
    description="Innovation gate: snap to raw when prediction error exceeds this (meters).",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_closing_gate_m",
    key="Longitudinal.LiveTune.DRelFilterClosingGateM",
    cli_name="drel-filter-cg",
    label="drel_cg",
    default=20.0,
    minimum=5.0,
    maximum=40.0,
    description="Closing gate: snap to raw when lead appears this much closer than predicted (meters).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_reacquire_pos_jerk_limit",
    key="Longitudinal.LiveTune.CruiseReacquirePosJerkLimit",
    cli_name="cruise-reacquire-pos-jerk-limit",
    label="cruise_reacquire_pos_jerk_limit",
    default=0.6,
    minimum=0.0,
    maximum=5.0,
    description="Max upward jerk (m/s^3) on planner output during cruise after a lead drops. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_reacquire_jerk_window_s",
    key="Longitudinal.LiveTune.CruiseReacquireJerkWindowS",
    cli_name="cruise-reacquire-jerk-window-s",
    label="cruise_reacquire_jerk_window_s",
    default=1.5,
    minimum=0.0,
    maximum=3.0,
    description="Duration (s) the cruise_reacquire_pos_jerk_limit is enforced after a lead drops. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="lead_prob_enter",
    key="Longitudinal.LiveTune.LeadProbEnter",
    cli_name="lead-prob-enter",
    label="lead_prob_enter",
    default=0.6,
    minimum=0.0,
    maximum=1.0,
    description="vision lead prob required to latch a slot on (Schmitt trigger). Raise to reject flicker.",
  ),
  LeadResponseTuneSpec(
    attr="lead_prob_exit",
    key="Longitudinal.LiveTune.LeadProbExit",
    cli_name="lead-prob-exit",
    label="lead_prob_exit",
    default=0.35,
    minimum=0.0,
    maximum=1.0,
    description="vision lead prob below which a latched slot releases. Lower than enter = hysteresis band.",
  ),
  LeadResponseTuneSpec(
    attr="lead_source_acquire_frames",
    key="Longitudinal.LiveTune.LeadSourceAcquireFrames",
    cli_name="lead-source-acquire-frames",
    label="lead_source_acquire_frames",
    default=1.0,
    minimum=1.0,
    maximum=20.0,
    description="Consecutive valid-lead frames required at the MPC before switching source FROM cruise TO lead. 1 = no dwell.",
  ),
  LeadResponseTuneSpec(
    attr="lead_source_release_frames",
    key="Longitudinal.LiveTune.LeadSourceReleaseFrames",
    cli_name="lead-source-release-frames",
    label="lead_source_release_frames",
    default=1.0,
    minimum=1.0,
    maximum=40.0,
    description="Consecutive invalid-lead frames required at the MPC before switching source FROM lead TO cruise. 1 = no dwell.",
  ),
  LeadResponseTuneSpec(
    attr="phantom_lead_hold_s",
    key="Longitudinal.LiveTune.PhantomLeadHoldS",
    cli_name="phantom-lead-hold-s",
    label="phantom_lead_hold_s",
    default=0.0,
    minimum=0.0,
    maximum=1.5,
    description="Duration (s) the last-known lead is extrapolated after status goes False. 0 disables (default off).",
  ),
  LeadResponseTuneSpec(
    attr="phantom_lead_stable_frames",
    key="Longitudinal.LiveTune.PhantomLeadStableFrames",
    cli_name="phantom-lead-stable-frames",
    label="phantom_lead_stable_frames",
    default=5.0,
    minimum=1.0,
    maximum=40.0,
    description="Consecutive stable frames required before a dropped lead is eligible for phantom hold.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_detect_transitions",
    key="Longitudinal.LiveTune.FlutterDetectTransitions",
    cli_name="flutter-detect-transitions",
    label="flutter_detect_transitions",
    default=2.0,
    minimum=1.0,
    maximum=10.0,
    description="Source-transition count within FlutterDetectWindowS that triggers bidirectional jerk clamp.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_detect_window_s",
    key="Longitudinal.LiveTune.FlutterDetectWindowS",
    cli_name="flutter-detect-window-s",
    label="flutter_detect_window_s",
    default=1.0,
    minimum=0.1,
    maximum=5.0,
    description="Rolling window length (s) for flutter-detection transition count.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_clamp_jerk_mps3",
    key="Longitudinal.LiveTune.FlutterClampJerkMps3",
    cli_name="flutter-clamp-jerk-mps3",
    label="flutter_clamp_jerk_mps3",
    default=0.8,
    minimum=0.0,
    maximum=5.0,
    description="Bidirectional jerk cap (m/s^3) applied to planner output while flutter mode is active. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_clamp_bypass_decel_mps2",
    key="Longitudinal.LiveTune.FlutterClampBypassDecelMps2",
    cli_name="flutter-clamp-bypass-decel-mps2",
    label="flutter_clamp_bypass_decel_mps2",
    default=1.5,
    minimum=0.0,
    maximum=5.0,
    description="If modelAccel < -this, flutter clamp is bypassed so hard braking is not delayed.",
  ),
)

LEAD_RESPONSE_TUNE_SPECS_BY_ATTR = {spec.attr: spec for spec in LEAD_RESPONSE_TUNE_SPECS}


@dataclass(frozen=True)
class LeadResponseTuningConfig:
  lead_preview_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_strength"].default
  lead_preview_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_gap_min_m"].default
  lead_preview_max_buffer_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_max_buffer_m"].default
  lead_acquire_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_acquire_window_s"].default
  gap_reclaim_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_strength"].default
  gap_reclaim_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_gap_min_m"].default
  gap_reclaim_max_accel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_max_accel"].default
  cutin_settle_duration_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_duration_s"].default
  cutin_settle_max_decel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_max_decel"].default
  cutin_settle_max_closing_speed_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_max_closing_speed_mps"].default
  cutin_settle_accel_bias_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_accel_bias_mps2"].default
  virtual_lead_slow_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["virtual_lead_slow_tau_s"].default
  drel_filter_tau_close_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_tau_close_s"].default
  drel_filter_tau_open_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_tau_open_s"].default
  drel_filter_open_slew_max_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_open_slew_max_mps"].default
  drel_filter_innovation_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_innovation_gate_m"].default
  drel_filter_closing_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_closing_gate_m"].default
  cruise_reacquire_pos_jerk_limit: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_reacquire_pos_jerk_limit"].default
  cruise_reacquire_jerk_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_reacquire_jerk_window_s"].default
  lead_prob_enter: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_prob_enter"].default
  lead_prob_exit: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_prob_exit"].default
  lead_source_acquire_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_source_acquire_frames"].default
  lead_source_release_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_source_release_frames"].default
  phantom_lead_hold_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["phantom_lead_hold_s"].default
  phantom_lead_stable_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["phantom_lead_stable_frames"].default
  flutter_detect_transitions: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_detect_transitions"].default
  flutter_detect_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_detect_window_s"].default
  flutter_clamp_jerk_mps3: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_clamp_jerk_mps3"].default
  flutter_clamp_bypass_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_clamp_bypass_decel_mps2"].default

  @classmethod
  def defaults(cls) -> LeadResponseTuningConfig:
    return cls()

  def as_dict(self) -> dict[str, float]:
    return asdict(self)


def clamp_lead_response_tuning_value(attr: str, value: float) -> float:
  return LEAD_RESPONSE_TUNE_SPECS_BY_ATTR[attr].clamp(float(value))


def _read_float(params: Params, key: str, default: float) -> float:
  try:
    raw = params.get(key)
    if raw is None:
      return float(default)
    value = float(raw)
    return value if math.isfinite(value) else float(default)
  except Exception:
    return float(default)


def build_lead_response_tuning_config(values: dict[str, Any] | None = None) -> LeadResponseTuningConfig:
  values = {} if values is None else dict(values)
  clamped: dict[str, float] = {}
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    raw_value = values.get(spec.attr, spec.default)
    try:
      numeric_value = float(raw_value)
    except Exception:
      numeric_value = spec.default
    if not math.isfinite(numeric_value):
      numeric_value = spec.default
    clamped[spec.attr] = spec.clamp(numeric_value)
  return LeadResponseTuningConfig(**clamped)


def read_lead_response_tuning_config(params: Params) -> LeadResponseTuningConfig:
  values = {
    spec.attr: _read_float(params, spec.key, spec.default)
    for spec in LEAD_RESPONSE_TUNE_SPECS
  }
  return build_lead_response_tuning_config(values)


def set_lead_response_tuning_params(params: Params, overrides: dict[str, float]) -> dict[str, float]:
  applied: dict[str, float] = {}
  for attr, requested in overrides.items():
    spec = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR[attr]
    applied_value = spec.clamp(float(requested))
    params.put(spec.key, applied_value)
    applied[attr] = applied_value
  return applied


def reset_lead_response_tuning_params(params: Params) -> None:
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    params.remove(spec.key)


def get_lead_response_tune_rows(params: Params) -> list[dict[str, Any]]:
  effective = read_lead_response_tuning_config(params)
  rows: list[dict[str, Any]] = []
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    rows.append({
      "attr": spec.attr,
      "cli_name": spec.cli_name,
      "label": spec.label,
      "key": spec.key,
      "description": spec.description,
      "default": spec.default,
      "minimum": spec.minimum,
      "maximum": spec.maximum,
      "stored": params.get(spec.key),
      "effective": getattr(effective, spec.attr),
    })
  return rows


def format_lead_response_tune_summary(config: LeadResponseTuningConfig) -> str:
  return " ".join(
    f"{spec.label}={getattr(config, spec.attr):.3f}"
    for spec in LEAD_RESPONSE_TUNE_SPECS
  )
