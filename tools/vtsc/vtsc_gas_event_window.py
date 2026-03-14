#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Any, Dict, Mapping, Sequence


DEFAULT_GAS_EVENT_WINDOW_HALF_S = 1.0
GAS_EVENT_NOT_CONSTRAINING_GAP_MPS = -0.5
GAS_EVENT_PRESSING_THROUGH_CAP_GAP_MPS = 0.75
GAS_EVENT_ACTIVE_SHARE_MIN = 0.25
GAS_EVENT_TIGHTEN_SCALE_DELTA_MIN = 0.01
GAS_EVENT_RELAX_SCALE_DELTA_MIN = 0.005


def _as_float(value: Any) -> float | None:
  try:
    out = float(value)
  except Exception:
    return None
  return out if math.isfinite(out) else None


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  if value is None:
    return False
  if isinstance(value, str):
    return value.strip().lower() in {"1", "true", "yes", "on"}
  try:
    return bool(value)
  except Exception:
    return False


def classify_gas_event_constraint(*, gap_median: float | None, gap_max: float | None) -> str:
  if gap_median is None and gap_max is None:
    return ""
  if gap_median is not None and gap_median <= GAS_EVENT_NOT_CONSTRAINING_GAP_MPS:
    return "not_constraining"
  if gap_median is not None and gap_median >= GAS_EVENT_PRESSING_THROUGH_CAP_GAP_MPS:
    return "pressing_through_cap"
  if gap_max is not None and gap_max >= (GAS_EVENT_PRESSING_THROUGH_CAP_GAP_MPS + 0.5):
    return "pressing_through_cap"
  return "borderline_cap"


def classify_gas_event_calibration(summary: Mapping[str, Any]) -> str:
  constraint_label = str(summary.get("constraint_label") or "")
  active_share = _as_float(summary.get("active_share"))
  scale_min = _as_float(summary.get("scale_min"))
  scale_max = _as_float(summary.get("scale_max"))

  if not constraint_label:
    return ""
  if constraint_label == "not_constraining":
    return "not_constraining"
  if active_share is None or active_share < GAS_EVENT_ACTIVE_SHARE_MIN:
    return "inactive_or_weak"

  tighten = scale_min is not None and scale_min <= (1.0 - GAS_EVENT_TIGHTEN_SCALE_DELTA_MIN)
  relax = scale_max is not None and scale_max >= (1.0 + GAS_EVENT_RELAX_SCALE_DELTA_MIN)
  if tighten and relax:
    return "mixed_bias"
  if tighten:
    return "tighten_bias"
  if relax:
    return "relax_candidate"
  return "neutral_or_weak"


def summarize_gas_event_window(
  rows: Sequence[Mapping[str, Any]],
  *,
  half_s: float = DEFAULT_GAS_EVENT_WINDOW_HALF_S,
) -> Dict[str, Any]:
  window_rows = []
  gap_vals = []
  scale_vals = []
  reason_counts: Counter[str] = Counter()
  cap_counts: Counter[str] = Counter()
  active_count = 0

  for row in rows:
    dt = _as_float(row.get("dt"))
    if dt is None or abs(dt) > float(half_s):
      continue
    window_rows.append(row)

    v_ego = _as_float(row.get("vEgo"))
    vtsc_vel = _as_float(row.get("vtscVelMps"))
    if v_ego is not None and vtsc_vel is not None:
      gap_vals.append(float(v_ego - vtsc_vel))

    scale = _as_float(row.get("lowSpeedCalScale"))
    if scale is not None:
      scale_vals.append(float(scale))

    if _as_bool(row.get("lowSpeedCalActive")):
      active_count += 1

    reason = str(row.get("lowSpeedCalReason") or "").strip()
    if reason:
      reason_counts[reason] += 1

    cap = str(row.get("activeCap") or "").strip()
    if cap:
      cap_counts[cap] += 1

  if not window_rows:
    return {
      "window_half_s": float(half_s),
      "samples": 0,
      "active_share": None,
      "gap_median": None,
      "gap_max": None,
      "scale_min": None,
      "scale_max": None,
      "reason_mode": "",
      "cap_mode": "",
      "constraint_label": "",
      "calibration_label": "",
    }

  summary: Dict[str, Any] = {
    "window_half_s": float(half_s),
    "samples": len(window_rows),
    "active_share": float(active_count / len(window_rows)),
    "gap_median": (float(statistics.median(gap_vals)) if gap_vals else None),
    "gap_max": (float(max(gap_vals)) if gap_vals else None),
    "scale_min": (float(min(scale_vals)) if scale_vals else None),
    "scale_max": (float(max(scale_vals)) if scale_vals else None),
    "reason_mode": reason_counts.most_common(1)[0][0] if reason_counts else "",
    "cap_mode": cap_counts.most_common(1)[0][0] if cap_counts else "",
  }
  summary["constraint_label"] = classify_gas_event_constraint(
    gap_median=summary["gap_median"],
    gap_max=summary["gap_max"],
  )
  summary["calibration_label"] = classify_gas_event_calibration(summary)
  return summary
