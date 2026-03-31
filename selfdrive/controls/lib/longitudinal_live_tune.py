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
    default=1.0,
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
    attr="gap_reclaim_strength",
    key="Longitudinal.LiveTune.GapReclaimStrength",
    cli_name="gap-reclaim-strength",
    label="reclaim_strength",
    default=1.0,
    minimum=0.0,
    maximum=2.0,
    description="Scale factor for how eagerly ACC closes a safe extra gap when the lead is pulling away.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_gap_min_m",
    key="Longitudinal.LiveTune.GapReclaimGapMinM",
    cli_name="gap-reclaim-gap-min-m",
    label="reclaim_gap_min_m",
    default=1.5,
    minimum=0.0,
    maximum=10.0,
    description="Minimum extra slack above nominal headway before gap reclaim is allowed.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_max_accel",
    key="Longitudinal.LiveTune.GapReclaimMaxAccel",
    cli_name="gap-reclaim-max-accel",
    label="reclaim_max_accel",
    default=0.36,
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
    default=0.10,
    minimum=0.0,
    maximum=0.30,
    description="Positive accel bias added to settle floor to counteract EV regen braking on coast.",
  ),
)

LEAD_RESPONSE_TUNE_SPECS_BY_ATTR = {spec.attr: spec for spec in LEAD_RESPONSE_TUNE_SPECS}


@dataclass(frozen=True)
class LeadResponseTuningConfig:
  lead_preview_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_strength"].default
  lead_preview_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_gap_min_m"].default
  lead_preview_max_buffer_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_max_buffer_m"].default
  gap_reclaim_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_strength"].default
  gap_reclaim_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_gap_min_m"].default
  gap_reclaim_max_accel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_max_accel"].default
  cutin_settle_duration_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_duration_s"].default
  cutin_settle_max_decel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_max_decel"].default
  cutin_settle_max_closing_speed_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_max_closing_speed_mps"].default
  cutin_settle_accel_bias_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_accel_bias_mps2"].default

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
