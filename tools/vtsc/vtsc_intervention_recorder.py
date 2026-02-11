#!/usr/bin/env python3
"""
VTSC Intervention Recorder (on-device)

Goal
- Detect longitudinal driver interventions (gas/brake press) while VTSC is the
  active limiting source (slowest speed recommendation).
- Save a self-contained "event bundle" for offline VTSC tuning:
  - 10s pre + 10s post of a lightweight computed trace (JSONL)
  - VTSC snapshot window (JSONL) from /data/media/0/VTSCDebug/vtsc_snapshots.jsonl
  - rlog/qlog segments (prev/current/next) copied once the segment closes (no .lock)

This runs safely alongside openpilot. It does NOT capture camera/video.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Allow running without `pip install -e .` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cereal import messaging
from opendbc.car.common.conversions import Conversions as CV


REALDATA_DIR = Path("/data/media/0/realdata")
VTSC_DEBUG_DIR = Path("/data/media/0/VTSCDebug")
EVENTS_DIR_DEFAULT = Path("/data/media/0/VTSCTuner/events")

# Conservative defaults: keep bundles, but avoid unbounded growth.
DEFAULT_MAX_TOTAL_MB = 2048


def _now_utc_tag() -> str:
  return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _safe_read_current_route() -> str:
  try:
    from openpilot.common.params import Params
    route = Params().get("CurrentRoute")
    if route is None:
      return ""
    if isinstance(route, (bytes, bytearray)):
      return route.decode("utf-8", errors="ignore")
    return str(route)
  except Exception:
    return ""


def _list_route_segments(route: str) -> list[int]:
  if not route:
    return []
  segs: list[int] = []
  try:
    prefix = f"{route}--"
    for name in os.listdir(REALDATA_DIR):
      if not name.startswith(prefix):
        continue
      # name: <route>--<seg>
      tail = name[len(prefix):]
      try:
        segs.append(int(tail))
      except Exception:
        continue
  except Exception:
    return []
  return sorted(set(segs))


def _guess_current_segment(route: str) -> int | None:
  segs = _list_route_segments(route)
  return max(segs) if segs else None


def _segment_dir(route: str, seg: int) -> Path:
  return REALDATA_DIR / f"{route}--{seg}"


def _segment_complete(seg_dir: Path) -> bool:
  # loggerd creates lock files while still writing.
  if not seg_dir.is_dir():
    return False
  if (seg_dir / "rlog.lock").exists():
    return False
  # rlog should exist and be non-empty to be useful
  rlog = seg_dir / "rlog.zst"
  if not rlog.exists():
    return False
  try:
    return rlog.stat().st_size > 0
  except Exception:
    return False


def _wait_for_segment(route: str, seg: int, *, timeout_s: float) -> Path | None:
  t0 = time.monotonic()
  seg_dir = _segment_dir(route, seg)
  while time.monotonic() - t0 <= timeout_s:
    if _segment_complete(seg_dir):
      return seg_dir
    time.sleep(0.5)
  return None


def _copy_if_exists(src: Path, dst: Path) -> None:
  try:
    if src.exists():
      dst.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src, dst)
  except Exception:
    pass


def _dir_size_bytes(path: Path) -> int:
  total = 0
  try:
    for root, _dirs, files in os.walk(path):
      for f in files:
        try:
          total += (Path(root) / f).stat().st_size
        except Exception:
          pass
  except Exception:
    pass
  return total


def _prune_old_events(events_root: Path, *, max_total_mb: int) -> None:
  try:
    events_root.mkdir(parents=True, exist_ok=True)
  except Exception:
    return

  max_bytes = int(max_total_mb) * 1024 * 1024
  # Sort by mtime (oldest first)
  items: list[tuple[float, Path, int]] = []
  total = 0
  for p in events_root.iterdir():
    if not p.is_dir():
      continue
    sz = _dir_size_bytes(p)
    total += sz
    try:
      mt = p.stat().st_mtime
    except Exception:
      mt = 0.0
    items.append((mt, p, sz))
  items.sort(key=lambda t: t[0])

  while total > max_bytes and items:
    _mt, p, sz = items.pop(0)
    try:
      shutil.rmtree(p, ignore_errors=True)
    except Exception:
      pass
    total -= sz


def _write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
    f.write("\n")
  os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, separators=(",", ":")) + "\n")
  os.replace(tmp, path)


def _extract_vtsc_snapshots_window(out_path: Path, *, t0_mono: float, pre_s: float, post_s: float) -> int:
  src = VTSC_DEBUG_DIR / "vtsc_snapshots.jsonl"
  if not src.exists():
    return 0
  t_lo = float(t0_mono - pre_s)
  t_hi = float(t0_mono + post_s)
  rows: list[dict[str, Any]] = []
  try:
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
      for line in f:
        try:
          d = json.loads(line)
        except Exception:
          continue
        try:
          ts = float(d.get("ts", -1.0))
        except Exception:
          continue
        if t_lo <= ts <= t_hi:
          rows.append(d if isinstance(d, dict) else {})
  except Exception:
    return 0
  if rows:
    _write_jsonl(out_path, rows)
  return len(rows)


def _tail_text_files(glob_paths: list[str], *, grep_pat: str, max_lines: int) -> list[str]:
  out: list[str] = []
  rx = re.compile(grep_pat)
  for p in glob_paths:
    try:
      with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max_lines:]
      for ln in lines:
        if rx.search(ln):
          out.append(f"{p}:{ln.rstrip()}")
    except Exception:
      continue
  return out[-max_lines:]


def _compute_sources(sample: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
  # Returns: (min_source_name, min_speed_mps, all_sources)
  INF = 1e9
  sources: dict[str, float] = {}
  for k, v in sample.get("sources", {}).items():
    try:
      vv = float(v)
    except Exception:
      continue
    if vv <= 0.0:
      continue
    sources[str(k)] = vv
  if not sources:
    return "none", INF, {}
  name, val = min(sources.items(), key=lambda kv: kv[1])
  return name, float(val), sources


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--events-dir", type=str, default=str(EVENTS_DIR_DEFAULT))
  ap.add_argument("--pre-seconds", type=float, default=10.0)
  ap.add_argument("--post-seconds", type=float, default=10.0)
  ap.add_argument("--buffer-seconds", type=float, default=30.0)
  ap.add_argument("--max-total-mb", type=int, default=DEFAULT_MAX_TOTAL_MB)
  ap.add_argument("--min-vtsc-delta-mps", type=float, default=0.5, help="Require VTSC to be this much below the next-best source.")
  ap.add_argument("--min-pred-lat-accel", type=float, default=0.8, help="Minimum maxPredictedLateralAccel to consider 'real turn'.")
  ap.add_argument("--cooldown-seconds", type=float, default=6.0)
  args = ap.parse_args()

  events_root = Path(args.events_dir)
  pre_s = float(args.pre_seconds)
  post_s = float(args.post_seconds)
  buf_s = float(args.buffer_seconds)

  # Subscriptions: keep minimal but sufficient to decide "VTSC is limiting".
  services = [
    "carState",
    "carControl",
    "controlsState",
    "selfdriveState",
    "longitudinalPlanSP",
    "rtiStateSP",
  ]
  sm = messaging.SubMaster(services)

  # Rolling trace buffer for precise 10s pre/post without depending on route segmentation.
  dt = 0.05  # 20 Hz
  buf_len = int(max(20, buf_s / dt))
  trace = collections.deque(maxlen=buf_len)

  gas_prev = False
  brake_prev = False
  last_trigger_t = -1e9
  # Cache CurrentRoute reads to avoid hitting Params at 20Hz.
  route_cache = ""
  next_route_check_t = 0.0

  pending: dict[str, Any] | None = None

  def _start_segment_copy_worker(event_dir: Path, route: str, seg0: int) -> None:
    # Copy prev/current/next segments once they close (no rlog.lock).
    segs = [seg0 - 1, seg0, seg0 + 1]
    out_seg_dir = event_dir / "segments"
    for seg in segs:
      if seg < 0:
        continue
      # Wait longer for current/next to close/appear.
      timeout = 15.0 if seg == seg0 - 1 else 150.0
      sd = _wait_for_segment(route, seg, timeout_s=timeout)
      if sd is None:
        continue
      dst = out_seg_dir / f"{seg:03d}"
      _copy_if_exists(sd / "rlog.zst", dst / "rlog.zst")
      _copy_if_exists(sd / "qlog.zst", dst / "qlog.zst")

  def _kick_copy_worker(event_dir: Path, route: str, seg0: int) -> None:
    # Best-effort background thread, but avoid importing threading until needed.
    try:
      import threading
      t = threading.Thread(target=_start_segment_copy_worker, args=(event_dir, route, seg0), daemon=True)
      t.start()
    except Exception:
      pass

  print("[vtsc_intervention] running. waiting for interventions...", flush=True)

  while True:
    sm.update(int(dt * 1000))
    t_mono = float(time.monotonic())

    if t_mono >= next_route_check_t:
      route_cache = _safe_read_current_route()
      next_route_check_t = t_mono + 2.0

    try:
      cs = sm["carState"]
    except Exception:
      continue
    try:
      cc = sm["carControl"]
    except Exception:
      cc = None
    try:
      sds = sm["selfdriveState"]
    except Exception:
      sds = None
    try:
      lp = sm["longitudinalPlanSP"]
    except Exception:
      lp = None
    try:
      rti = sm["rtiStateSP"]
    except Exception:
      rti = None
    try:
      ctrls = sm["controlsState"]
    except Exception:
      ctrls = None

    gas = bool(getattr(cs, "gasPressed", False))
    brake = bool(getattr(cs, "brakePressed", False))

    # Engagement gating (avoid spurious offroad presses).
    long_active = bool(getattr(cc, "longActive", False)) if cc is not None else False
    lat_active = bool(getattr(cc, "latActive", False)) if cc is not None else False
    enabled = bool(getattr(sds, "enabled", False)) if sds is not None else False

    # Cruise setpoint is in kph; convert to m/s.
    try:
      v_cruise_mps = float(getattr(cs, "vCruise", 0.0)) * float(CV.KPH_TO_MS)
    except Exception:
      v_cruise_mps = 0.0
    try:
      v_ego = float(getattr(cs, "vEgo", 0.0))
    except Exception:
      v_ego = 0.0
    try:
      a_ego = float(getattr(cs, "aEgo", 0.0))
    except Exception:
      a_ego = 0.0
    try:
      brake_val = float(getattr(cs, "brake", 0.0))
    except Exception:
      brake_val = 0.0
    try:
      regen = bool(getattr(cs, "regenBraking", False))
    except Exception:
      regen = False

    # ControlsState fields are helpful for offline triage (curvature + accel commands).
    ctrl_curv = None
    ctrl_des_curv = None
    up_accel = None
    ui_accel = None
    uf_accel = None
    long_ctrl_state = None
    force_decel = None
    if ctrls is not None:
      try:
        ctrl_curv = float(getattr(ctrls, "curvature", 0.0))
      except Exception:
        ctrl_curv = None
      try:
        ctrl_des_curv = float(getattr(ctrls, "desiredCurvature", 0.0))
      except Exception:
        ctrl_des_curv = None
      try:
        up_accel = float(getattr(ctrls, "upAccelCmd", 0.0))
      except Exception:
        up_accel = None
      try:
        ui_accel = float(getattr(ctrls, "uiAccelCmd", 0.0))
      except Exception:
        ui_accel = None
      try:
        uf_accel = float(getattr(ctrls, "ufAccelCmd", 0.0))
      except Exception:
        uf_accel = None
      try:
        long_ctrl_state = int(getattr(ctrls, "longControlState", 0))
      except Exception:
        long_ctrl_state = None
      try:
        force_decel = bool(getattr(ctrls, "forceDecel", False))
      except Exception:
        force_decel = None

    # VTSC + SLC details from longitudinalPlanSP (published every planner cycle).
    vtsc_state = None
    vtsc_vel = None
    pred_lat_acc = None
    cur_lat_acc = None
    slc_active = False
    slc_offseted = None
    if lp is not None:
      try:
        vtsc = lp.visionTurnSpeedControl
        vtsc_state = int(getattr(vtsc, "state", 0))
        vtsc_vel = float(getattr(vtsc, "velocity", 0.0))
        pred_lat_acc = float(getattr(vtsc, "maxPredictedLateralAccel", 0.0))
        cur_lat_acc = float(getattr(vtsc, "currentLateralAccel", 0.0))
      except Exception:
        pass
      try:
        slc = lp.slc
        slc_active = bool(getattr(slc, "active", False))
        if slc_active:
          slc_offseted = float(getattr(slc, "speedLimit", 0.0)) + float(getattr(slc, "speedLimitOffset", 0.0))
      except Exception:
        pass

    # RTI recommendation (m/s) when active.
    rti_reco = None
    if rti is not None:
      try:
        reco = float(getattr(rti, "recommendedSpeed", 0.0))
        if reco > 0.1:
          rti_reco = reco
      except Exception:
        pass

    # Build sources and determine whether VTSC is the winning limiter.
    sources: dict[str, float] = {"cruise": float(v_cruise_mps)}
    if vtsc_vel is not None and vtsc_vel > 0.0:
      sources["vtsc"] = float(vtsc_vel)
    if slc_offseted is not None and slc_offseted > 0.0:
      sources["slc"] = float(slc_offseted)
    if rti_reco is not None and rti_reco > 0.0:
      sources["rti"] = float(rti_reco)

    # Determine best and second-best sources (lowest speeds).
    src_sorted = sorted(sources.items(), key=lambda kv: kv[1])
    min_src, min_v = src_sorted[0] if src_sorted else ("none", 1e9)
    second_v = src_sorted[1][1] if len(src_sorted) >= 2 else 1e9
    vtsc_limiting = (min_src == "vtsc") and (float(second_v) - float(min_v) >= float(args.min_vtsc_delta_mps))

    # Lightweight trace row for later slicing.
    row = {
      "t": t_mono,
      "route": route_cache,
      "vEgo": v_ego,
      "aEgo": a_ego,
      "gas": gas,
      "brake": brake,
      "enabled": enabled,
      "latActive": lat_active,
      "longActive": long_active,
      "brakeVal": brake_val,
      "regenBraking": regen,
      "vCruiseMps": v_cruise_mps,
      "ctrlCurvature": ctrl_curv,
      "ctrlDesiredCurvature": ctrl_des_curv,
      "ctrlUpAccelCmd": up_accel,
      "ctrlUiAccelCmd": ui_accel,
      "ctrlUfAccelCmd": uf_accel,
      "ctrlLongControlState": long_ctrl_state,
      "ctrlForceDecel": force_decel,
      "vtscState": vtsc_state,
      "vtscVelMps": vtsc_vel,
      "vtscMaxPredLatAcc": pred_lat_acc,
      "vtscCurLatAcc": cur_lat_acc,
      "slcActive": slc_active,
      "slcOffsetedMps": slc_offseted,
      "rtiRecoMps": rti_reco,
      "sources": sources,
      "minSource": min_src,
      "minSpeedMps": float(min_v),
      "vtscLimiting": bool(vtsc_limiting),
    }
    trace.append(row)

    # Finalize a pending event after the post window.
    if pending is not None and t_mono >= float(pending["t0"]) + post_s:
      ev_dir = Path(pending["event_dir"])
      t0 = float(pending["t0"])
      # Extract window from trace buffer (best effort).
      window = [r for r in list(trace) if (t0 - pre_s) <= float(r.get("t", 0.0)) <= (t0 + post_s)]
      _write_jsonl(ev_dir / "trace_20s.jsonl", window)
      n_snap = _extract_vtsc_snapshots_window(ev_dir / "vtsc_snapshots_20s.jsonl", t0_mono=t0, pre_s=pre_s, post_s=post_s)
      # Best-effort tails for quick human triage
      try:
        vtsc_watch_tail = []
        watch_path = VTSC_DEBUG_DIR / "vtsc_watch.log"
        if watch_path.exists():
          vtsc_watch_tail = watch_path.read_text(errors="ignore").splitlines()[-300:]
        (ev_dir / "vtsc_watch_tail.txt").write_text("\n".join(vtsc_watch_tail) + "\n", encoding="utf-8")
      except Exception:
        pass
      try:
        swag_globs = [str(p) for p in sorted(Path("/data/log").glob("swaglog.*"))[-3:]]
        hits = _tail_text_files(swag_globs, grep_pat=r"VTSCDBG|VisionTurn|vtsc", max_lines=1200)
        (ev_dir / "swaglog_vtsc_tail.txt").write_text("\n".join(hits) + "\n", encoding="utf-8")
      except Exception:
        pass

      # Rotation (local to events dir)
      _prune_old_events(events_root, max_total_mb=int(args.max_total_mb))

      print(f"[vtsc_intervention] event finalized {ev_dir.name} (snapshots={n_snap}, trace_rows={len(window)})", flush=True)
      pending = None

    # Trigger on rising edge (with cooldown).
    gas_rise = gas and not gas_prev
    brake_rise = brake and not brake_prev
    gas_prev = gas
    brake_prev = brake

    if not (gas_rise or brake_rise):
      continue
    if (t_mono - last_trigger_t) < float(args.cooldown_seconds):
      continue

    # Only care while OP is actively controlling longitudinal, and VTSC is the limiting source.
    #
    # NOTE: a pedal intervention can flip longActive/latActive in the same cycle, so use a short
    # trailing window as "pre-intervention" truth.
    recent_n = int(max(1, 0.75 / dt))  # ~0.75s window
    recent = list(trace)[-recent_n:]
    was_engaged = any(bool(r.get("enabled")) and bool(r.get("latActive")) and bool(r.get("longActive")) for r in recent)
    was_vtsc_limiting = any(bool(r.get("vtscLimiting")) for r in recent)
    if not (was_engaged and was_vtsc_limiting):
      continue

    # Ensure this is actually a "turny" scenario; avoid false positives on straight + other constraints.
    if not vtsc_limiting:
      continue
    if pred_lat_acc is not None and float(pred_lat_acc) < float(args.min_pred_lat_accel):
      # Still allow if VTSC is materially below cruise (turn cap), even if predicted lat acc not populated.
      if not (vtsc_vel is not None and (v_cruise_mps - float(vtsc_vel)) >= 1.0):
        continue

    last_trigger_t = t_mono
    action = "gas" if gas_rise else "brake"
    route = _safe_read_current_route()
    seg0 = _guess_current_segment(route) if route else None

    event_id = f"{_now_utc_tag()}_{action}"
    if route:
      event_id += f"_{route}"
    if seg0 is not None:
      event_id += f"_seg{seg0:03d}"

    ev_dir = events_root / event_id
    ev_dir.mkdir(parents=True, exist_ok=True)

    meta = {
      "event_id": event_id,
      "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
      "t0_monotonic_s": t_mono,
      "action": action,
      "route": route,
      "seg_guess": seg0,
      "enabled": enabled,
      "latActive": lat_active,
      "longActive": long_active,
      "vEgo": v_ego,
      "vCruiseMps": v_cruise_mps,
      "vtscState": vtsc_state,
      "vtscVelMps": vtsc_vel,
      "vtscMaxPredLatAcc": pred_lat_acc,
      "vtscCurLatAcc": cur_lat_acc,
      "sources": sources,
      "minSource": min_src,
      "minSpeedMps": float(min_v),
    }
    _write_json(ev_dir / "event.json", meta)

    print(f"[vtsc_intervention] TRIGGER {event_id} sources={sources} vEgo={v_ego:.2f} vCruise={v_cruise_mps:.2f}", flush=True)

    # Start background segment copier (eventually consistent once segments close).
    if route and seg0 is not None:
      _kick_copy_worker(ev_dir, route, int(seg0))

    # Mark as pending until we have post-window samples.
    pending = {"t0": t_mono, "event_dir": str(ev_dir)}

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
