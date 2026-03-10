#!/usr/bin/env python3
from __future__ import annotations

"""
Build a VTSC intervention RCA workbook from locally cached rlogs.

Inputs
- Event bundles under:
    <base>/events_offline/<event_id>/event.json
  The event.json must include:
    - route (e.g. "00000007--38c928758f")
    - seg (int)
    - t0 (float, monotonic seconds; typically logMonoTime/1e9)

- Matching rlogs under:
    <base>/realdata/<route>--<seg>/rlog.zst

Outputs
- Per-event trace (rlog-synced, 20s window):
    <event_dir>/trace_rlog_20s_plus.jsonl

- RCA workbook:
    <base>/vtsc_rca.xlsx  (default)

This is intentionally self-contained: it uses pycapnp + zstandard and loads cereal
schemas from a provided repo root (default: /home/chris/repos/chauffeur-dev4).
"""

import argparse
import bz2
import dataclasses
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import zstandard as zstd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DEFAULT_REPO_ROOT = Path("/home/chris/repos/chauffeur-dev4")


def _read_bytes_maybe_decompress(path: Path) -> bytes:
  dat = path.read_bytes()
  # bz2 magic: BZh
  if dat.startswith(b"BZh"):
    return bz2.decompress(dat)
  # zstd magic: 28 B5 2F FD
  if dat.startswith(b"\x28\xB5\x2F\xFD"):
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(dat) as reader:
      return reader.read()
  return dat


def _iter_capnp_events(path: Path, capnp_log) -> Iterator[Any]:
  dat = _read_bytes_maybe_decompress(path)
  try:
    ents = capnp_log.Event.read_multiple_bytes(dat)
  except Exception as e:
    raise RuntimeError(f"Failed to parse capnp events from {path}: {e}") from e

  # read_multiple_bytes yields a generator; wrap in try so corruption doesn't crash everything
  for evt in ents:
    yield evt


def _safe_float(x: Any) -> Optional[float]:
  try:
    v = float(x)
  except Exception:
    return None
  if not math.isfinite(v):
    return None
  return v


def _safe_int(x: Any) -> Optional[int]:
  # pycapnp enums are _DynamicEnum; use `.raw` to get the underlying integer.
  try:
    raw = getattr(x, "raw")
  except Exception:
    raw = None
  if raw is not None:
    try:
      return int(raw)
    except Exception:
      return None
  try:
    return int(x)
  except Exception:
    return None


def _safe_str(x: Any) -> Optional[str]:
  if x is None:
    return None
  if isinstance(x, bytes):
    try:
      return x.decode("utf-8", errors="replace")
    except Exception:
      return str(x)
  try:
    return str(x)
  except Exception:
    return None


def _mean_finite(xs: Iterable[Any]) -> Optional[float]:
  vals: List[float] = []
  for x in xs:
    v = _safe_float(x)
    if v is None:
      continue
    vals.append(v)
  if not vals:
    return None
  return float(statistics.fmean(vals))


def _max_abs(xs: Iterable[Any]) -> Optional[float]:
  m: Optional[float] = None
  for x in xs:
    v = _safe_float(x)
    if v is None:
      continue
    a = abs(v)
    if m is None or a > m:
      m = a
  return m


def _yaw_rate_abs_max3(z_list: Iterable[Any]) -> Optional[float]:
  # ModelDataV2.orientationRate.{t,z} is a fixed-horizon prediction.
  # "3" here intentionally means the first 3 predicted points (not 3 seconds),
  # matching prior VTSC debug traces used for quick curvature confidence checks.
  vals: List[float] = []
  for z in list(z_list)[:3]:
    zv = _safe_float(z)
    if zv is None:
      continue
    vals.append(abs(zv))
  return max(vals) if vals else None


@dataclasses.dataclass(frozen=True)
class EventBundle:
  event_id: str
  action: str
  route: str
  seg: int
  t0: float
  pre_s: float
  post_s: float
  dt_s: float
  event_dir: Path


def _load_event_bundle(event_dir: Path) -> EventBundle:
  event_path = event_dir / "event.json"
  d = json.loads(event_path.read_text(encoding="utf-8"))
  event_id = str(d.get("event_id") or event_dir.name)
  action = str(d.get("action") or "")
  route = str(d.get("route") or "")
  seg = int(d.get("seg") if d.get("seg") is not None else d.get("scan_hit", {}).get("seg", -1))
  t0 = float(d.get("t0") if d.get("t0") is not None else d.get("scan_hit", {}).get("t", 0.0))
  window = d.get("window_s") or {}
  pre_s = float(window.get("pre", 10.0))
  post_s = float(window.get("post", 10.0))
  dt_s = float(window.get("dt", 0.05))
  return EventBundle(
    event_id=event_id,
    action=action,
    route=route,
    seg=seg,
    t0=t0,
    pre_s=pre_s,
    post_s=post_s,
    dt_s=dt_s,
    event_dir=event_dir,
  )


def _vtsc_state_name(v: Optional[int]) -> str:
  return {
    0: "disabled",
    1: "entering",
    2: "turning",
    3: "leaving",
  }.get(int(v) if v is not None else -1, "unknown")


def _model_conf_name(v: Optional[int]) -> str:
  return {0: "red", 1: "yellow", 2: "green"}.get(int(v) if v is not None else -1, "unknown")


def _lp_source_name(v: Optional[int]) -> str:
  return {0: "cruise", 1: "lead0", 2: "lead1", 3: "lead2", 4: "e2e"}.get(int(v) if v is not None else -1, "unknown")


def _round_dt(dt: float) -> float:
  # Stable string/lookup key; avoid -0.0 and floating drift.
  return float(f"{dt:.2f}")


def _build_trace_rows_for_segment(
  *,
  rlog_path: Path,
  events: List[EventBundle],
  capnp_log,
) -> Dict[str, List[Dict[str, Any]]]:
  # Prepare per-event targets.
  targets_by_id: Dict[str, List[float]] = {}
  idx_by_id: Dict[str, int] = {}
  out_by_id: Dict[str, List[Dict[str, Any]]] = {}

  for ev in events:
    n = int(round((ev.pre_s + ev.post_s) / ev.dt_s)) + 1
    targets = [float(ev.t0 - ev.pre_s + i * ev.dt_s) for i in range(n)]
    targets_by_id[ev.event_id] = targets
    idx_by_id[ev.event_id] = 0
    out_by_id[ev.event_id] = []

  # Segment-wide bounds for early stop.
  seg_t_min = min(ts[0] for ts in targets_by_id.values())
  seg_t_max = max(ts[-1] for ts in targets_by_id.values())

  latest: Dict[str, Dict[str, Any]] = {}

  def _get_latest(k: str) -> Dict[str, Any]:
    return latest.get(k, {})

  def _row_for_target(target_t: float, ev: EventBundle) -> Dict[str, Any]:
    # Read latest snapshots.
    cs = _get_latest("carState")
    cc = _get_latest("carControl")
    sds = _get_latest("selfdriveState")
    ctrls = _get_latest("controlsState")
    lp_sp = _get_latest("longitudinalPlanSP")
    model = _get_latest("modelV2")
    lp = _get_latest("longitudinalPlan")
    gps = _get_latest("gpsLocation")
    map_sp = _get_latest("liveMapDataSP")

    v_ego = cs.get("vEgo")
    vtsc_v = lp_sp.get("vtscVelMps")
    lp_v0 = lp.get("lpV0")

    dt = float(target_t - ev.t0)
    row: Dict[str, Any] = {
      "t": float(f"{target_t:.6f}".rstrip("0").rstrip(".")),
      "dt": _round_dt(dt),
      "vEgo": v_ego,
      "aEgo": cs.get("aEgo"),
      "vCruiseKph": cs.get("vCruiseKph"),
      "gasPressed": cs.get("gasPressed"),
      "brakePressed": cs.get("brakePressed"),
      "brake": cs.get("brake"),
      "enabled": sds.get("enabled"),
      "latActive": cc.get("latActive"),
      "longActive": cc.get("longActive"),
      "actAccel": cc.get("actAccel"),
      "actLongState": cc.get("actLongState"),
      "uiAccelCmd": ctrls.get("uiAccelCmd"),
      "upAccelCmd": ctrls.get("upAccelCmd"),
      "ufAccelCmd": ctrls.get("ufAccelCmd"),
      "forceDecel": ctrls.get("forceDecel"),
      "ctrlCurvature": ctrls.get("ctrlCurvature"),
      "ctrlDesiredCurvature": ctrls.get("ctrlDesiredCurvature"),
      "vtscState": lp_sp.get("vtscState"),
      "vtscVelMps": vtsc_v,
      "vtscMaxPredLatAcc": lp_sp.get("vtscMaxPredLatAcc"),
      "vtscCurLatAcc": lp_sp.get("vtscCurLatAcc"),
      "llProbMean": model.get("llProbMean"),
      "modelConf": model.get("modelConf"),
      "modelFrameDropPerc": model.get("modelFrameDropPerc"),
      "oriRateLen": model.get("oriRateLen"),
      "yawRateAbsMax3": model.get("yawRateAbsMax3"),
      "yawRateAbsMax": model.get("yawRateAbsMax"),
      "lpSource": lp.get("lpSource"),
      "lpATarget": lp.get("lpATarget"),
      "lpV0": lp_v0,
      "lpVMin": lp.get("lpVMin"),
      "lpAllowBrake": lp.get("lpAllowBrake"),
      "lpAllowThrottle": lp.get("lpAllowThrottle"),
      "lpHasLead": lp.get("lpHasLead"),
      # GPS/map helper observability for MTSC RCA
      "gpsService": gps.get("gpsService"),
      "gpsLat": gps.get("gpsLat"),
      "gpsLon": gps.get("gpsLon"),
      "gpsAlt": gps.get("gpsAlt"),
      "gpsSpeed": gps.get("gpsSpeed"),
      "gpsBearingDeg": gps.get("gpsBearingDeg"),
      "gpsAccuracy": gps.get("gpsAccuracy"),
      "gpsHasFix": gps.get("gpsHasFix"),
      "gpsAge": (float(target_t - gps.get("_t")) if gps.get("_t") is not None else None),
      "mapSpeedLimitValid": map_sp.get("mapSpeedLimitValid"),
      "mapSpeedLimit": map_sp.get("mapSpeedLimit"),
      "mapSpeedLimitAheadValid": map_sp.get("mapSpeedLimitAheadValid"),
      "mapSpeedLimitAhead": map_sp.get("mapSpeedLimitAhead"),
      "mapSpeedLimitAheadDistance": map_sp.get("mapSpeedLimitAheadDistance"),
      "mapRoadName": map_sp.get("mapRoadName"),
      "mapRoadGeometryValid": map_sp.get("mapRoadGeometryValid"),
      "mapCurrentWayId": map_sp.get("mapCurrentWayId"),
      "mapDataAge": (float(target_t - map_sp.get("_t")) if map_sp.get("_t") is not None else None),
    }

    # Names + derived deltas.
    row["vtscStateName"] = _vtsc_state_name(_safe_int(row.get("vtscState")))
    row["modelConfName"] = _model_conf_name(_safe_int(row.get("modelConf")))
    row["lpSourceName"] = _lp_source_name(_safe_int(row.get("lpSource")))

    row["dv_vtsc"] = (float(v_ego) - float(vtsc_v)) if v_ego is not None and vtsc_v is not None else None
    row["dv_lp0"] = (float(v_ego) - float(lp_v0)) if v_ego is not None and lp_v0 is not None else None
    return row

  for evt in _iter_capnp_events(rlog_path, capnp_log):
    try:
      t = float(evt.logMonoTime) * 1e-9
    except Exception:
      continue

    # Fast skip until we're close to the earliest target.
    if t < (seg_t_min - 2.0):
      continue

    which = None
    try:
      which = evt.which()
    except Exception:
      which = None

    if which == "carState":
      m = evt.carState
      latest["carState"] = {
        "vEgo": _safe_float(getattr(m, "vEgo", None)),
        "aEgo": _safe_float(getattr(m, "aEgo", None)),
        "vCruiseKph": _safe_float(getattr(m, "vCruise", None)),
        "gasPressed": bool(getattr(m, "gasPressed", False)),
        "brakePressed": bool(getattr(m, "brakePressed", False)),
        "brake": _safe_float(getattr(m, "brake", None)),
      }
    elif which == "carControl":
      m = evt.carControl
      act = getattr(m, "actuators", None)
      latest["carControl"] = {
        "latActive": bool(getattr(m, "latActive", False)),
        "longActive": bool(getattr(m, "longActive", False)),
        "actAccel": _safe_float(getattr(act, "accel", None)) if act is not None else None,
        "actLongState": _safe_int(getattr(act, "longControlState", None)) if act is not None else None,
      }
    elif which == "selfdriveState":
      m = evt.selfdriveState
      latest["selfdriveState"] = {"enabled": bool(getattr(m, "enabled", False))}
    elif which == "controlsState":
      m = evt.controlsState
      latest["controlsState"] = {
        "uiAccelCmd": _safe_float(getattr(m, "uiAccelCmd", None)),
        "upAccelCmd": _safe_float(getattr(m, "upAccelCmd", None)),
        "ufAccelCmd": _safe_float(getattr(m, "ufAccelCmd", None)),
        "forceDecel": bool(getattr(m, "forceDecel", False)),
        "ctrlCurvature": _safe_float(getattr(m, "curvature", None)),
        "ctrlDesiredCurvature": _safe_float(getattr(m, "desiredCurvature", None)),
      }
    elif which == "longitudinalPlanSP":
      m = evt.longitudinalPlanSP
      vtsc = getattr(m, "visionTurnSpeedControl", None)
      if vtsc is not None:
        latest["longitudinalPlanSP"] = {
          "vtscState": _safe_int(getattr(vtsc, "state", None)),
          "vtscVelMps": _safe_float(getattr(vtsc, "velocity", None)),
          "vtscMaxPredLatAcc": _safe_float(getattr(vtsc, "maxPredictedLateralAccel", None)),
          "vtscCurLatAcc": _safe_float(getattr(vtsc, "currentLateralAccel", None)),
        }
    elif which == "modelV2":
      m = evt.modelV2
      lane_probs = getattr(m, "laneLineProbs", [])
      ll_mean = _mean_finite(lane_probs) if lane_probs is not None else None
      conf = _safe_int(getattr(m, "confidence", None))
      fd = _safe_float(getattr(m, "frameDropPerc", None))
      ori = getattr(m, "orientationRate", None)
      t_list = getattr(ori, "t", []) if ori is not None else []
      z_list = getattr(ori, "z", []) if ori is not None else []
      latest["modelV2"] = {
        "llProbMean": ll_mean,
        "modelConf": conf,
        "modelFrameDropPerc": fd,
        "oriRateLen": len(list(t_list)) if t_list is not None else None,
        "yawRateAbsMax3": _yaw_rate_abs_max3(z_list) if z_list is not None else None,
        "yawRateAbsMax": _max_abs(z_list) if z_list is not None else None,
      }
    elif which == "longitudinalPlan":
      m = evt.longitudinalPlan
      speeds = list(getattr(m, "speeds", []) or [])
      v0 = speeds[0] if speeds else None
      vmin = min(speeds) if speeds else None
      latest["longitudinalPlan"] = {
        "lpSource": _safe_int(getattr(m, "longitudinalPlanSource", None)),
        "lpATarget": _safe_float(getattr(m, "aTarget", None)),
        "lpV0": _safe_float(v0),
        "lpVMin": _safe_float(vmin),
        "lpAllowBrake": bool(getattr(m, "allowBrake", True)),
        "lpAllowThrottle": bool(getattr(m, "allowThrottle", True)),
        "lpHasLead": bool(getattr(m, "hasLead", False)),
      }
    elif which in ("gpsLocation", "gpsLocationExternal"):
      m = getattr(evt, which)
      latest["gpsLocation"] = {
        "gpsService": which,
        "gpsLat": _safe_float(getattr(m, "latitude", None)),
        "gpsLon": _safe_float(getattr(m, "longitude", None)),
        "gpsAlt": _safe_float(getattr(m, "altitude", None)),
        "gpsSpeed": _safe_float(getattr(m, "speed", None)),
        "gpsBearingDeg": _safe_float(getattr(m, "bearingDeg", None)),
        "gpsAccuracy": _safe_float(getattr(m, "accuracy", None)),
        "gpsHasFix": bool(getattr(m, "hasFix", False)),
        "_t": t,
      }
    elif which == "liveMapDataSP":
      m = evt.liveMapDataSP
      current_way_id = None
      try:
        current_way_id = _safe_int(getattr(getattr(m, "currentRoadSegment", None), "wayId", None))
      except Exception:
        current_way_id = None
      latest["liveMapDataSP"] = {
        "mapSpeedLimitValid": bool(getattr(m, "speedLimitValid", False)),
        "mapSpeedLimit": _safe_float(getattr(m, "speedLimit", None)),
        "mapSpeedLimitAheadValid": bool(getattr(m, "speedLimitAheadValid", False)),
        "mapSpeedLimitAhead": _safe_float(getattr(m, "speedLimitAhead", None)),
        "mapSpeedLimitAheadDistance": _safe_float(getattr(m, "speedLimitAheadDistance", None)),
        "mapRoadName": _safe_str(getattr(m, "roadName", None)),
        "mapRoadGeometryValid": bool(getattr(m, "roadGeometryValid", False)) if hasattr(m, "roadGeometryValid") else None,
        "mapCurrentWayId": current_way_id,
        "_t": t,
      }

    # Flush targets for any events whose next target <= current time.
    for ev in events:
      targets = targets_by_id[ev.event_id]
      i = idx_by_id[ev.event_id]
      while i < len(targets) and targets[i] <= t:
        out_by_id[ev.event_id].append(_row_for_target(targets[i], ev))
        i += 1
      idx_by_id[ev.event_id] = i

    # Early stop once we've filled all events and passed the max.
    if t > (seg_t_max + 0.5) and all(idx_by_id[e.event_id] >= len(targets_by_id[e.event_id]) for e in events):
      break

  # If any event didn't fully populate (e.g. log ended), pad remaining rows with last-known values.
  for ev in events:
    targets = targets_by_id[ev.event_id]
    i = idx_by_id[ev.event_id]
    while i < len(targets):
      out_by_id[ev.event_id].append(_row_for_target(targets[i], ev))
      i += 1
    idx_by_id[ev.event_id] = i

  return out_by_id


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, separators=(",", ":")) + "\n")
  tmp.replace(path)


def _load_trace(path: Path) -> pd.DataFrame:
  rows: List[Dict[str, Any]] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  df = pd.DataFrame(rows)
  # Ensure numeric dt for filtering (it's stored already rounded to 2dp).
  if "dt" in df.columns:
    df["dt"] = pd.to_numeric(df["dt"], errors="coerce")
  return df


def _value_at_dt(df: pd.DataFrame, dt: float, col: str) -> float | None:
  if df.empty or col not in df.columns or "dt" not in df.columns:
    return None
  want = float(f"{dt:.2f}")
  sub = df[df["dt"].sub(want).abs() < 1e-9]
  if sub.empty:
    # fallback: nearest
    i = (df["dt"] - want).abs().idxmin()
    try:
      v = df.loc[i, col]
    except Exception:
      return None
  else:
    v = sub.iloc[0][col]
  try:
    v = float(v)
  except Exception:
    return None
  return v if math.isfinite(v) else None


def _first_dt_where(df: pd.DataFrame, col: str, pred) -> float | None:
  if df.empty or col not in df.columns or "dt" not in df.columns:
    return None
  for _idx, row in df.sort_values("dt").iterrows():
    try:
      v = row[col]
    except Exception:
      continue
    if pred(v):
      try:
        return float(row["dt"])
      except Exception:
        return None
  return None


def _window_df(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
  if df.empty or "dt" not in df.columns:
    return df
  return df[(df["dt"] >= lo) & (df["dt"] <= hi)].copy()


def _compute_summary_row(ev: EventBundle, df: pd.DataFrame) -> Dict[str, Any]:
  pre = _window_df(df, -2.0, -0.1)
  row: Dict[str, Any] = {
    "event_id": ev.event_id,
    "action": ev.action,
    "route": ev.route,
    "seg": ev.seg,
    "t0": ev.t0,
    "vEgo@0": _value_at_dt(df, 0.0, "vEgo"),
    "vEgo@-0.5": _value_at_dt(df, -0.5, "vEgo"),
    "vtscVel@-0.5": _value_at_dt(df, -0.5, "vtscVelMps"),
    "dv_vtsc@-0.5": _value_at_dt(df, -0.5, "dv_vtsc"),
    "vtscVel@-2": _value_at_dt(df, -2.0, "vtscVelMps"),
    "vtscVelDrop(-2->-0.5)": None,
    "vtscVelMin[-2..0]": None,
    "vtscVelDtMin[-2..0]": None,
    "dv_vtscMax[-2..0]": None,
    "dv_vtscDtMax[-2..0]": None,
    "predLatAccMax[-2..0]": None,
    "predLatAccDtMax[-2..0]": None,
    "llProbMin[-2..0]": None,
    "llProbDtMin[-2..0]": None,
    "llProb@-2": _value_at_dt(df, -2.0, "llProbMean"),
    "llProb@-0.5": _value_at_dt(df, -0.5, "llProbMean"),
    "modelFrameDropMax[-2..0]": None,
    "aTarget@-0.5": _value_at_dt(df, -0.5, "lpATarget"),
    "aTargetFirstNegDt[-2..0]": None,
    "actAccel@-0.5": _value_at_dt(df, -0.5, "actAccel"),
    "actAccelFirstNegDt[-2..0]": None,
    "lpVMin@-0.5": _value_at_dt(df, -0.5, "lpVMin"),
    "lpVMinMin[-2..0]": None,
    "lpVMinDtMin[-2..0]": None,
    "gpsAge@-0.5": _value_at_dt(df, -0.5, "gpsAge"),
    "gpsAgeMax[-2..0]": None,
    "mapDataAge@-0.5": _value_at_dt(df, -0.5, "mapDataAge"),
    "mapDataAgeMax[-2..0]": None,
    "mapRoadNameChanges[-2..0]": None,
    "mapSpeedValidChanges[-2..0]": None,
    "mapRoadGeomValidAny[-2..0]": None,
  }

  vtsc_m2 = row["vtscVel@-2"]
  vtsc_m05 = row["vtscVel@-0.5"]
  if vtsc_m2 is not None and vtsc_m05 is not None:
    row["vtscVelDrop(-2->-0.5)"] = float(vtsc_m2) - float(vtsc_m05)

  # predLatAcc max and dt in [-2,0]
  if not pre.empty and "vtscMaxPredLatAcc" in pre.columns:
    i = pre["vtscMaxPredLatAcc"].astype(float).idxmax()
    try:
      row["predLatAccMax[-2..0]"] = float(pre.loc[i, "vtscMaxPredLatAcc"])
      row["predLatAccDtMax[-2..0]"] = float(pre.loc[i, "dt"])
    except Exception:
      pass

  # vtscVel min and dt in [-2,0]
  if not pre.empty and "vtscVelMps" in pre.columns:
    try:
      pre_v = pre["vtscVelMps"].astype(float)
      i = pre_v.idxmin()
      row["vtscVelMin[-2..0]"] = float(pre.loc[i, "vtscVelMps"])
      row["vtscVelDtMin[-2..0]"] = float(pre.loc[i, "dt"])
    except Exception:
      pass

  # dv_vtsc max and dt in [-2,0]
  if not pre.empty and "dv_vtsc" in pre.columns:
    try:
      pre_dv = pre["dv_vtsc"].astype(float)
      i = pre_dv.idxmax()
      row["dv_vtscMax[-2..0]"] = float(pre.loc[i, "dv_vtsc"])
      row["dv_vtscDtMax[-2..0]"] = float(pre.loc[i, "dt"])
    except Exception:
      pass

  # llProb min and dt in [-2,0]
  if not pre.empty and "llProbMean" in pre.columns:
    i = pre["llProbMean"].astype(float).idxmin()
    try:
      row["llProbMin[-2..0]"] = float(pre.loc[i, "llProbMean"])
      row["llProbDtMin[-2..0]"] = float(pre.loc[i, "dt"])
    except Exception:
      pass

  # lpVMin min and dt in [-2,0]
  if not pre.empty and "lpVMin" in pre.columns:
    try:
      pre_vmin = pre["lpVMin"].astype(float)
      i = pre_vmin.idxmin()
      row["lpVMinMin[-2..0]"] = float(pre.loc[i, "lpVMin"])
      row["lpVMinDtMin[-2..0]"] = float(pre.loc[i, "dt"])
    except Exception:
      pass

  if not pre.empty and "gpsAge" in pre.columns:
    try:
      row["gpsAgeMax[-2..0]"] = float(pre["gpsAge"].astype(float).max())
    except Exception:
      pass

  if not pre.empty and "mapDataAge" in pre.columns:
    try:
      row["mapDataAgeMax[-2..0]"] = float(pre["mapDataAge"].astype(float).max())
    except Exception:
      pass

  if not pre.empty and "mapRoadName" in pre.columns:
    try:
      names = [str(x) for x in pre.sort_values("dt")["mapRoadName"].tolist()]
      row["mapRoadNameChanges[-2..0]"] = int(sum(1 for a, b in zip(names, names[1:]) if a != b))
    except Exception:
      pass

  if not pre.empty and "mapSpeedLimitValid" in pre.columns:
    try:
      valid = [bool(x) for x in pre.sort_values("dt")["mapSpeedLimitValid"].tolist()]
      row["mapSpeedValidChanges[-2..0]"] = int(sum(1 for a, b in zip(valid, valid[1:]) if a != b))
    except Exception:
      pass

  if not pre.empty and "mapRoadGeometryValid" in pre.columns:
    try:
      vals = [bool(x) for x in pre["mapRoadGeometryValid"].tolist()]
      row["mapRoadGeomValidAny[-2..0]"] = bool(any(vals))
    except Exception:
      pass

  if not pre.empty and "modelFrameDropPerc" in pre.columns:
    try:
      row["modelFrameDropMax[-2..0]"] = float(pre["modelFrameDropPerc"].astype(float).max())
    except Exception:
      pass

  row["aTargetFirstNegDt[-2..0]"] = _first_dt_where(pre, "lpATarget", lambda v: _safe_float(v) is not None and float(v) < -0.10)
  row["actAccelFirstNegDt[-2..0]"] = _first_dt_where(pre, "actAccel", lambda v: _safe_float(v) is not None and float(v) < -0.10)
  return row


def _style_sheet(ws) -> None:
  ws.freeze_panes = "A2"
  header_fill = PatternFill("solid", fgColor="1F2937")  # slate-ish
  header_font = Font(color="FFFFFF", bold=True)
  for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

  # Basic column sizing (auto-fit-ish)
  for col in range(1, ws.max_column + 1):
    letter = get_column_letter(col)
    max_len = 0
    for row in range(1, min(ws.max_row, 3000) + 1):
      v = ws[f"{letter}{row}"].value
      if v is None:
        continue
      max_len = max(max_len, len(str(v)))
    ws.column_dimensions[letter].width = min(60, max(10, max_len + 2))


def _write_workbook(summary_rows: List[Dict[str, Any]], trace_rows: pd.DataFrame, out_path: Path) -> None:
  wb = Workbook()

  # Summary
  ws = wb.active
  ws.title = "Summary"
  if summary_rows:
    cols = list(summary_rows[0].keys())
    ws.append(cols)
    for r in summary_rows:
      ws.append([r.get(c) for c in cols])
  _style_sheet(ws)

  # Trace
  ws2 = wb.create_sheet("Trace")
  if not trace_rows.empty:
    cols2 = list(trace_rows.columns)
    ws2.append(cols2)
    for _idx, r in trace_rows.iterrows():
      ws2.append([r.get(c) for c in cols2])
  _style_sheet(ws2)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  wb.save(out_path)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--base", type=str, default=str(Path(__file__).resolve().parents[1]))
  ap.add_argument("--repo-root", type=str, default=str(DEFAULT_REPO_ROOT))
  ap.add_argument("--events-subdir", type=str, default="events_offline")
  ap.add_argument("--realdata-subdir", type=str, default="realdata")
  ap.add_argument("--out-xlsx", type=str, default="")
  ap.add_argument("--overwrite-traces", action="store_true")
  args = ap.parse_args()

  base = Path(args.base).resolve()
  repo_root = Path(args.repo_root).resolve()
  events_root = base / args.events_subdir
  realdata_root = base / args.realdata_subdir
  out_xlsx = Path(args.out_xlsx).resolve() if args.out_xlsx else (base / "vtsc_rca.xlsx")

  if not events_root.is_dir():
    raise SystemExit(f"events dir missing: {events_root}")

  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
  try:
    from cereal import log as capnp_log  # type: ignore
  except Exception as e:
    raise SystemExit(f"Failed to import cereal from repo_root={repo_root}: {e}")

  bundles: List[EventBundle] = []
  for p in sorted(events_root.iterdir()):
    if not p.is_dir():
      continue
    try:
      bundles.append(_load_event_bundle(p))
    except Exception:
      continue
  if not bundles:
    raise SystemExit(f"No event bundles found under {events_root}")

  # Group by segment rlog.
  by_seg: Dict[Tuple[str, int], List[EventBundle]] = {}
  for b in bundles:
    by_seg.setdefault((b.route, b.seg), []).append(b)

  # Generate traces per segment.
  for (route, seg), evs in sorted(by_seg.items(), key=lambda t: (t[0][0], t[0][1])):
    rlog = realdata_root / f"{route}--{seg}" / "rlog.zst"
    if not rlog.exists():
      print(f"[WARN] missing rlog for route={route} seg={seg}: {rlog}", file=sys.stderr)
      continue

    # Only generate those missing unless overwrite requested.
    need = []
    for ev in evs:
      out_trace = ev.event_dir / "trace_rlog_20s_plus.jsonl"
      if args.overwrite_traces or (not out_trace.exists()):
        need.append(ev)
    if not need:
      continue

    out = _build_trace_rows_for_segment(rlog_path=rlog, events=need, capnp_log=capnp_log)
    for ev in need:
      rows = out.get(ev.event_id, [])
      _write_jsonl(ev.event_dir / "trace_rlog_20s_plus.jsonl", rows)
      print(f"[trace] {ev.event_id} rows={len(rows)} rlog={rlog}")

  # Build workbook (use trace files on disk for simplicity).
  summary_rows: List[Dict[str, Any]] = []
  traces: List[pd.DataFrame] = []
  for ev in bundles:
    trace_path = ev.event_dir / "trace_rlog_20s_plus.jsonl"
    if not trace_path.exists():
      continue
    df = _load_trace(trace_path)
    df.insert(0, "event_id", ev.event_id)
    df.insert(1, "route", ev.route)
    df.insert(2, "seg", ev.seg)
    df.insert(3, "action", ev.action)
    traces.append(df)
    summary_rows.append(_compute_summary_row(ev, df))

  trace_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
  _write_workbook(summary_rows, trace_df, out_xlsx)
  print(f"[xlsx] wrote {out_xlsx} (events={len(summary_rows)} trace_rows={len(trace_df)})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
