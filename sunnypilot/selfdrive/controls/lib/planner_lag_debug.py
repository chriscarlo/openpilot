from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

ENABLE_PARAM = "VTSCWriteSnapshotFile"
REALDATA_DIR = Path("/data/media/0/realdata")
EVENTS_DIR_DEFAULT = Path("/data/media/0/VTSCDebug/planner_lag_events")

TRACE_SECONDS = 20.0
EXPECTED_HZ = 20.0
REFRESH_INTERVAL_S = 2.0
STARTUP_GRACE_S = 5.0
DUMP_COOLDOWN_S = 8.0

LOW_CADENCE_GAP_S = 1.0 / 18.0
LOW_CADENCE_LOOP_S = 1.0 / 18.0
LOW_CADENCE_STREAK_TRIGGER = 6
HARD_PUBLISH_GAP_S = 0.080
HARD_LOOP_DT_S = 0.070

SPAN_PLANNER_UPDATE_TOTAL = "planner.update_total"
SPAN_UPDATE_V_CRUISE_TOTAL = "planner.update_v_cruise_total"
SPAN_RESPONSE_MODEL_LOOKUP = "planner.get_cruise_response_model"
SPAN_MPC_UPDATE = "planner.mpc_update"
SPAN_PUBLISH_LONGITUDINAL_PLAN_SP = "planner.publish_longitudinal_plan_sp"
SPAN_PREVIEW_ENCODE = "planner.preview_encode"
SPAN_DRIVER_ASSISTANCE_PUBLISH = "planner.driver_assistance_publish"
SPAN_VTSC_UPDATE = "vtsc.update_total"
SPAN_MAP_TAIL_CAP = "vtsc.map_tail_cap_total"
SPAN_PREVIEW_FROM_MAP = "vtsc.preview_from_map"
SPAN_PREVIEW_BRANCH_STUBS = "vtsc.preview_branch_stubs"
SPAN_MAP_CAP_ADVISORY = "vtsc.compute_map_cap.advisory"
SPAN_MAP_CAP_STRATEGIC = "vtsc.compute_map_cap.strategic"
SPAN_HELPER_PREDICT = "helper.predict_average_decel_for_cruise_cap"
SPAN_HELPER_CRUISE_CAP = "helper.cruise_cap_for_required_average_decel"

TIMING_KEYS_BASE = (
  SPAN_PLANNER_UPDATE_TOTAL,
  SPAN_UPDATE_V_CRUISE_TOTAL,
  SPAN_RESPONSE_MODEL_LOOKUP,
  SPAN_MPC_UPDATE,
  SPAN_PUBLISH_LONGITUDINAL_PLAN_SP,
  SPAN_PREVIEW_ENCODE,
  SPAN_DRIVER_ASSISTANCE_PUBLISH,
  SPAN_VTSC_UPDATE,
  SPAN_MAP_TAIL_CAP,
  SPAN_PREVIEW_FROM_MAP,
  SPAN_PREVIEW_BRANCH_STUBS,
  SPAN_MAP_CAP_ADVISORY,
  SPAN_MAP_CAP_STRATEGIC,
  SPAN_HELPER_PREDICT,
  SPAN_HELPER_CRUISE_CAP,
)

DERIVED_COMPONENTS = (
  "helper_total_ms",
  "preview_total_ms",
  "strategic_outside_helper_ms",
  "map_tail_other_ms",
  "vtsc_non_map_tail_ms",
  "planner_update_non_vtsc_ms",
  "publish_non_preview_encode_ms",
  "planner_loop_other_ms",
)

SUSPECT_COMPONENTS = (
  "strategic_outside_helper_ms",
  "helper_total_ms",
  "preview_total_ms",
  "vtsc_non_map_tail_ms",
  "planner_update_non_vtsc_ms",
  "publish_non_preview_encode_ms",
  "planner_loop_other_ms",
)

SUMMARY_COMPONENTS = (
  "planner_loop_dt_ms",
  "planner_publish_gap_ms",
  *TIMING_KEYS_BASE,
  *DERIVED_COMPONENTS,
)

TOP_BAD_CYCLE_FIELDS = (
  "frame",
  "planner_publish_gap_ms",
  "planner_loop_dt_ms",
  SPAN_PLANNER_UPDATE_TOTAL,
  SPAN_UPDATE_V_CRUISE_TOTAL,
  SPAN_RESPONSE_MODEL_LOOKUP,
  SPAN_MPC_UPDATE,
  SPAN_VTSC_UPDATE,
  SPAN_MAP_TAIL_CAP,
  SPAN_PREVIEW_FROM_MAP,
  SPAN_PREVIEW_BRANCH_STUBS,
  SPAN_MAP_CAP_ADVISORY,
  SPAN_MAP_CAP_STRATEGIC,
  SPAN_HELPER_PREDICT,
  SPAN_HELPER_CRUISE_CAP,
  "helper_total_ms",
  "preview_total_ms",
  "strategic_outside_helper_ms",
  "map_tail_other_ms",
  "vtsc_non_map_tail_ms",
  "planner_update_non_vtsc_ms",
  "publish_non_preview_encode_ms",
  "planner_loop_other_ms",
  "strategy_mode",
  "strategy_state",
  "map_floor_active",
  "map_floor_reason",
  "map_tail_reason",
  "map_tail_compute_reason",
  "map_tail_active",
  "map_tail_coverage",
  "map_advisory_cap",
  "map_strategic_cap",
  "map_selected_cap",
  "map_anchor_dist_m",
  "curve_preview_valid",
  "curve_preview_points",
  "curve_preview_branch_stubs",
  "curve_distance_m",
  "curve_time_to_s",
  "curve_max_curvature",
  "vtsc_velocity",
  "vtsc_state",
  "vtsc_state_name",
  "active_cap",
  "map_geometry_valid",
  "nearby_segment_count",
  "lag_reasons",
  "timing_counts",
)

_ACTIVE_RECORDER = threading.local()


def _active_cycle_recorder() -> PlannerLagRecorder | None:
  recorder = getattr(_ACTIVE_RECORDER, "recorder", None)
  if recorder is None or recorder.current_cycle is None:
    return None
  return recorder


def _set_active_cycle_recorder(recorder: PlannerLagRecorder | None) -> None:
  _ACTIVE_RECORDER.recorder = recorder


def start_span(name: str):
  recorder = _active_cycle_recorder()
  if recorder is None:
    return None
  return (recorder, str(name), time.perf_counter_ns())


def end_span(token) -> None:
  if token is None:
    return
  recorder, name, t0_ns = token
  recorder.record_span_ns(name, max(0, time.perf_counter_ns() - int(t0_ns)))


def record_fields(**fields: Any) -> None:
  recorder = _active_cycle_recorder()
  if recorder is None:
    return
  recorder.record_fields(**fields)


def current_cycle_enabled() -> bool:
  return _active_cycle_recorder() is not None


def _json_safe(value: Any) -> Any:
  if value is None or isinstance(value, (bool, int, str)):
    return value
  if isinstance(value, float):
    return float(value) if math.isfinite(value) else None
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  try:
    if hasattr(value, "name"):
      return str(value.name)
  except Exception:
    pass
  try:
    return int(value)
  except Exception:
    pass
  try:
    return float(value)
  except Exception:
    return str(value)


def _round_floats(obj: Any, digits: int = 4) -> Any:
  if isinstance(obj, float):
    return round(obj, digits) if math.isfinite(obj) else None
  if isinstance(obj, list):
    return [_round_floats(v, digits=digits) for v in obj]
  if isinstance(obj, dict):
    return {str(k): _round_floats(v, digits=digits) for k, v in obj.items()}
  return obj


def _percentile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  vals = sorted(float(v) for v in values)
  if len(vals) == 1:
    return vals[0]
  pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
  lo = int(math.floor(pos))
  hi = int(math.ceil(pos))
  if lo == hi:
    return vals[lo]
  frac = pos - lo
  return vals[lo] + (vals[hi] - vals[lo]) * frac


def _stats(values: list[float]) -> dict[str, Any]:
  vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
  if not vals:
    return {"count": 0}
  return {
    "count": len(vals),
    "mean_ms": float(sum(vals) / len(vals)),
    "p50_ms": float(_percentile(vals, 0.50)),
    "p90_ms": float(_percentile(vals, 0.90)),
    "p95_ms": float(_percentile(vals, 0.95)),
    "max_ms": float(max(vals)),
    "sum_ms": float(sum(vals)),
  }


def _component_breakdown_ms(cycle: dict[str, Any]) -> dict[str, float]:
  helper_predict_ms = float(cycle.get(SPAN_HELPER_PREDICT, 0.0) or 0.0)
  helper_cruise_cap_ms = float(cycle.get(SPAN_HELPER_CRUISE_CAP, 0.0) or 0.0)
  helper_total_ms = helper_predict_ms + helper_cruise_cap_ms

  preview_from_map_ms = float(cycle.get(SPAN_PREVIEW_FROM_MAP, 0.0) or 0.0)
  preview_branch_stubs_ms = float(cycle.get(SPAN_PREVIEW_BRANCH_STUBS, 0.0) or 0.0)
  preview_total_ms = preview_from_map_ms + preview_branch_stubs_ms

  strategic_total_ms = float(cycle.get(SPAN_MAP_CAP_STRATEGIC, 0.0) or 0.0)
  advisory_total_ms = float(cycle.get(SPAN_MAP_CAP_ADVISORY, 0.0) or 0.0)
  map_tail_total_ms = float(cycle.get(SPAN_MAP_TAIL_CAP, 0.0) or 0.0)
  if map_tail_total_ms <= 0.0 and (preview_total_ms + advisory_total_ms + strategic_total_ms) > 0.0:
    map_tail_total_ms = preview_total_ms + advisory_total_ms + strategic_total_ms
  vtsc_update_ms = float(cycle.get(SPAN_VTSC_UPDATE, 0.0) or 0.0)
  planner_update_ms = float(cycle.get(SPAN_PLANNER_UPDATE_TOTAL, 0.0) or 0.0)
  update_v_cruise_ms = float(cycle.get(SPAN_UPDATE_V_CRUISE_TOTAL, 0.0) or 0.0)
  planner_loop_dt_ms = float(cycle.get("planner_loop_dt_ms", 0.0) or 0.0)
  publish_sp_ms = float(cycle.get(SPAN_PUBLISH_LONGITUDINAL_PLAN_SP, 0.0) or 0.0)
  preview_encode_ms = float(cycle.get(SPAN_PREVIEW_ENCODE, 0.0) or 0.0)
  mpc_update_ms = float(cycle.get(SPAN_MPC_UPDATE, 0.0) or 0.0)
  driver_assist_ms = float(cycle.get(SPAN_DRIVER_ASSISTANCE_PUBLISH, 0.0) or 0.0)

  strategic_outside_helper_ms = max(0.0, strategic_total_ms - helper_total_ms)
  map_tail_other_ms = max(0.0, map_tail_total_ms - preview_total_ms - advisory_total_ms - strategic_total_ms)
  vtsc_non_map_tail_ms = max(0.0, vtsc_update_ms - map_tail_total_ms)
  planner_update_non_vtsc_ms = max(0.0, planner_update_ms - update_v_cruise_ms - mpc_update_ms)
  publish_non_preview_encode_ms = max(0.0, publish_sp_ms - preview_encode_ms)
  planner_loop_other_ms = max(0.0, planner_loop_dt_ms - planner_update_ms - publish_sp_ms - driver_assist_ms)

  return {
    "helper_total_ms": float(helper_total_ms),
    "preview_total_ms": float(preview_total_ms),
    "strategic_outside_helper_ms": float(strategic_outside_helper_ms),
    "map_tail_other_ms": float(map_tail_other_ms),
    "vtsc_non_map_tail_ms": float(vtsc_non_map_tail_ms),
    "planner_update_non_vtsc_ms": float(planner_update_non_vtsc_ms),
    "publish_non_preview_encode_ms": float(publish_non_preview_encode_ms),
    "planner_loop_other_ms": float(planner_loop_other_ms),
  }


def summarize_window(cycles: list[dict[str, Any]], trigger_cycle: dict[str, Any] | None = None) -> dict[str, Any]:
  normalized: list[dict[str, Any]] = []
  for cycle in cycles:
    row = dict(cycle)
    row.update(_component_breakdown_ms(row))
    normalized.append(row)

  bad_cycles = [row for row in normalized if bool(row.get("is_bad_cycle", False))]
  trigger = dict(trigger_cycle or (normalized[-1] if normalized else {}))
  if trigger:
    trigger.update(_component_breakdown_ms(trigger))

  component_stats = {
    name: _stats([float(row.get(name, 0.0) or 0.0) for row in bad_cycles])
    for name in SUMMARY_COMPONENTS
  }
  ranked_components = [
    {"name": name, "mean_ms": float(stats["mean_ms"])}
    for name, stats in component_stats.items()
    if stats.get("count", 0) > 0
  ]
  ranked_components.sort(key=lambda item: item["mean_ms"], reverse=True)
  suspect_rank = [
    {"name": name, "mean_ms": float(component_stats[name]["mean_ms"])}
    for name in SUSPECT_COMPONENTS
    if component_stats[name].get("count", 0) > 0
  ]
  suspect_rank.sort(key=lambda item: item["mean_ms"], reverse=True)

  top_bad_cycles = sorted(
    bad_cycles,
    key=lambda row: (
      float(row.get("planner_publish_gap_ms", 0.0) or 0.0),
      float(row.get("planner_loop_dt_ms", 0.0) or 0.0),
    ),
    reverse=True,
  )[:8]
  compact_bad_cycles = []
  for row in top_bad_cycles:
    compact_bad_cycles.append({field: _json_safe(row.get(field)) for field in TOP_BAD_CYCLE_FIELDS if field in row})

  return {
    "window_cycles": len(normalized),
    "bad_cycles": len(bad_cycles),
    "trigger_reasons": list(trigger.get("lag_reasons", []) or []),
    "dominant_component_bad_cycles": (suspect_rank[0]["name"] if suspect_rank else None),
    "component_stats_bad_cycles_ms": component_stats,
    "component_mean_rank_bad_cycles_ms": ranked_components,
    "suspect_split_rank_bad_cycles_ms": suspect_rank,
    "top_bad_cycles": compact_bad_cycles,
  }


class PlannerLagRecorder:
  def __init__(
    self,
    *,
    params: Params | None = None,
    events_dir: str | Path = EVENTS_DIR_DEFAULT,
    time_fn=time.monotonic,
  ):
    self._params = params if params is not None else Params()
    self._events_dir = Path(events_dir)
    self._time_fn = time_fn
    self._trace: collections.deque[dict[str, Any]] = collections.deque(
      maxlen=max(240, int(math.ceil(TRACE_SECONDS * EXPECTED_HZ * 1.5))),
    )
    self._current_cycle: dict[str, Any] | None = None
    self._enabled = False
    self._refresh_at_s = 0.0
    self._enabled_since_s = 0.0
    self._cooldown_until_s = 0.0
    self._low_cadence_streak = 0
    self._last_cycle_start_s: float | None = None
    self._last_publish_end_s: float | None = None
    self._last_model_logmono_ns: int | None = None

  @property
  def current_cycle(self) -> dict[str, Any] | None:
    return self._current_cycle

  def _read_enabled(self) -> bool:
    try:
      return bool(self._params.get_bool(ENABLE_PARAM))
    except Exception:
      return False

  def _refresh_enabled(self, now_s: float) -> bool:
    if now_s < self._refresh_at_s:
      return self._enabled
    self._refresh_at_s = now_s + REFRESH_INTERVAL_S
    enabled = self._read_enabled()
    if enabled == self._enabled:
      return enabled
    self._enabled = enabled
    self._trace.clear()
    self._current_cycle = None
    self._low_cadence_streak = 0
    self._cooldown_until_s = 0.0
    self._last_cycle_start_s = None
    self._last_publish_end_s = None
    self._last_model_logmono_ns = None
    self._enabled_since_s = now_s if enabled else 0.0
    if not enabled:
      _set_active_cycle_recorder(None)
    return enabled

  def begin_cycle(self, *, frame: int | None = None, model_logmono_ns: int | None = None) -> bool:
    now_s = float(self._time_fn())
    if not self._refresh_enabled(now_s):
      _set_active_cycle_recorder(None)
      return False

    cycle: dict[str, Any] = {
      "t_monotonic_s": float(now_s),
      "frame": (int(frame) if frame is not None else None),
      "timings_ms": {},
      "timing_counts": {},
      "lag_reasons": [],
    }

    if self._last_cycle_start_s is not None:
      cycle["planner_cycle_start_gap_ms"] = float(max(0.0, now_s - self._last_cycle_start_s) * 1000.0)
    if model_logmono_ns is not None and self._last_model_logmono_ns is not None:
      model_gap_ns = int(model_logmono_ns) - int(self._last_model_logmono_ns)
      if model_gap_ns >= 0:
        cycle["model_gap_ms"] = float(model_gap_ns / 1e6)

    self._last_cycle_start_s = now_s
    self._last_model_logmono_ns = (int(model_logmono_ns) if model_logmono_ns is not None else None)
    self._current_cycle = cycle
    _set_active_cycle_recorder(self)
    return True

  def record_span_ns(self, name: str, dt_ns: int) -> None:
    cycle = self._current_cycle
    if cycle is None:
      return
    timings = cycle.setdefault("timings_ms", {})
    counts = cycle.setdefault("timing_counts", {})
    timings[str(name)] = float(timings.get(str(name), 0.0) + (max(0, int(dt_ns)) / 1e6))
    counts[str(name)] = int(counts.get(str(name), 0) + 1)

  def record_fields(self, **fields: Any) -> None:
    cycle = self._current_cycle
    if cycle is None:
      return
    for key, value in fields.items():
      cycle[str(key)] = _json_safe(value)

  def finish_cycle(self, *, planner_loop_dt_s: float, publish_end_s: float | None = None) -> dict[str, Any] | None:
    cycle = self._current_cycle
    _set_active_cycle_recorder(None)
    self._current_cycle = None
    if cycle is None:
      return None

    end_s = float(self._time_fn()) if publish_end_s is None else float(publish_end_s)
    cycle["planner_loop_dt_ms"] = float(max(0.0, float(planner_loop_dt_s)) * 1000.0)
    cycle["planner_publish_gap_ms"] = (
      float(max(0.0, end_s - self._last_publish_end_s) * 1000.0)
      if self._last_publish_end_s is not None else 0.0
    )
    self._last_publish_end_s = end_s

    cycle.update(cycle.pop("timings_ms", {}))
    cycle["timing_counts"] = {k: int(v) for k, v in cycle.get("timing_counts", {}).items()}
    cycle.update(_component_breakdown_ms(cycle))

    publish_gap_s = float(cycle.get("planner_publish_gap_ms", 0.0) or 0.0) / 1000.0
    planner_loop_s = float(cycle.get("planner_loop_dt_ms", 0.0) or 0.0) / 1000.0
    low_cadence = publish_gap_s >= LOW_CADENCE_GAP_S or planner_loop_s >= LOW_CADENCE_LOOP_S
    self._low_cadence_streak = (self._low_cadence_streak + 1) if low_cadence else 0
    cycle["low_cadence_streak"] = int(self._low_cadence_streak)
    cycle["is_bad_cycle"] = bool(low_cadence)

    lag_reasons: list[str] = []
    if publish_gap_s >= HARD_PUBLISH_GAP_S:
      lag_reasons.append("publish_gap_hard")
    if planner_loop_s >= HARD_LOOP_DT_S:
      lag_reasons.append("planner_loop_hard")
    if self._low_cadence_streak >= LOW_CADENCE_STREAK_TRIGGER:
      lag_reasons.append("low_cadence_streak")
    cycle["lag_reasons"] = lag_reasons

    self._trace.append(cycle)
    if lag_reasons and (end_s - self._enabled_since_s) >= STARTUP_GRACE_S and end_s >= self._cooldown_until_s:
      self._cooldown_until_s = end_s + DUMP_COOLDOWN_S
      self._schedule_dump(trigger_cycle=cycle, dump_t=end_s)

    return cycle

  def _schedule_dump(self, *, trigger_cycle: dict[str, Any], dump_t: float) -> None:
    window = [dict(row) for row in list(self._trace)]
    trigger = dict(trigger_cycle)
    route = self._safe_read_current_route()
    seg_guess = self._guess_current_segment(route) if route else None

    event_tag = dt.datetime.utcfromtimestamp(time.time()).strftime("%Y%m%dT%H%M%SZ")
    event_id = f"{event_tag}_planner_lag"
    if route:
      event_id += f"_{route}"
    if seg_guess is not None:
      event_id += f"_seg{int(seg_guess):03d}"
    event_dir = self._events_dir / event_id

    meta = {
      "event_id": event_id,
      "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
      "enable_param": ENABLE_PARAM,
      "route": route,
      "seg_guess": seg_guess,
      "dump_t_monotonic_s": float(dump_t),
      "trigger_cycle": trigger,
    }

    threading.Thread(
      target=self._write_event_bundle,
      args=(event_dir, meta, window, trigger),
      daemon=True,
      name="planner-lag-dump",
    ).start()

  def _write_event_bundle(self, event_dir: Path, meta: dict[str, Any], window: list[dict[str, Any]], trigger_cycle: dict[str, Any]) -> None:
    try:
      event_dir.mkdir(parents=True, exist_ok=True)
      summary = summarize_window(window, trigger_cycle=trigger_cycle)
      self._write_json(event_dir / "event.json", {**meta, "summary": summary})
      self._write_json(event_dir / "summary.json", summary)
      self._write_json(event_dir / "trigger_cycle.json", trigger_cycle)
      self._write_jsonl(event_dir / "trace_20s.jsonl", window)
      cloudlog.warning(
        "VTSC planner lag dump",
        event_id=str(meta.get("event_id", "")),
        route=str(meta.get("route", "")),
        seg_guess=int(meta["seg_guess"]) if meta.get("seg_guess") is not None else None,
        reasons=list(trigger_cycle.get("lag_reasons", []) or []),
        planner_publish_gap_ms=round(float(trigger_cycle.get("planner_publish_gap_ms", 0.0) or 0.0), 2),
        planner_loop_dt_ms=round(float(trigger_cycle.get("planner_loop_dt_ms", 0.0) or 0.0), 2),
        strategic_outside_helper_ms=round(float(trigger_cycle.get("strategic_outside_helper_ms", 0.0) or 0.0), 2),
        helper_total_ms=round(float(trigger_cycle.get("helper_total_ms", 0.0) or 0.0), 2),
        preview_total_ms=round(float(trigger_cycle.get("preview_total_ms", 0.0) or 0.0), 2),
      )
    except Exception:
      cloudlog.exception("VTSC planner lag dump failed")

  @staticmethod
  def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(_round_floats(obj), f, indent=2, sort_keys=True)
      f.write("\n")
    os.replace(tmp, path)

  @staticmethod
  def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
      for row in rows:
        f.write(json.dumps(_round_floats(row), separators=(",", ":")) + "\n")
    os.replace(tmp, path)

  def _safe_read_current_route(self) -> str:
    try:
      raw = self._params.get("CurrentRoute")
      if raw is None:
        return ""
      if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="ignore")
      return str(raw)
    except Exception:
      return ""

  @staticmethod
  def _guess_current_segment(route: str) -> int | None:
    if not route:
      return None
    prefix = f"{route}--"
    segs: list[int] = []
    try:
      for name in os.listdir(REALDATA_DIR):
        if not name.startswith(prefix):
          continue
        try:
          segs.append(int(name[len(prefix):]))
        except Exception:
          continue
    except Exception:
      return None
    return max(segs) if segs else None
