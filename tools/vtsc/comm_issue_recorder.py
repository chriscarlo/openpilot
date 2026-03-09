#!/usr/bin/env python3
"""
VTSC Comm-Issue Recorder (on-device)

Goal
- Detect onroad comm/readiness degradations without manual prompting.
- Save a self-contained event bundle for offline RCA:
  - 10s pre + 10s post lightweight trace (JSONL)
  - summarized readiness / service-health timeline
  - trigger-time process/core snapshot and short per-core CPU sample
  - filtered swaglog tail
  - prev/current/next rlog/qlog segments copied once closed

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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cereal import messaging  # noqa: E402


REALDATA_DIR = Path("/data/media/0/realdata")
EVENTS_DIR_DEFAULT = Path("/data/media/0/VTSCTuner/comm_issues")
DEFAULT_MAX_TOTAL_MB = 768

COMM_EVENT_NAMES = {"commIssue", "commIssueAvgFreq"}
TRIGGER_IGNORED_MANAGER_PROCESSES = {"loggerd", "mapd"}
PROCESS_SNAPSHOT_NAMES = (
  "plannerd",
  "radard",
  "selfdrived",
  "controlsd",
  "camerad",
  "modeld",
  "locationd",
  "paramsd",
  "calibrationd",
  "lagd",
  "torqued",
  "card",
  "mapd",
)
SERVICE_HEALTH_NAMES = [
  "selfdriveStateSP",
  "onroadEvents",
  "managerState",
  "deviceState",
  "pandaStates",
  "peripheralState",
  "roadCameraState",
  "driverCameraState",
  "wideRoadCameraState",
  "modelV2",
  "livePose",
  "liveCalibration",
  "liveParameters",
  "radarState",
  "driverMonitoringState",
  "controlsState",
  "carOutput",
  "carControl",
  "longitudinalPlan",
  "driverAssistance",
]
CRITICAL_RECORDER_SERVICES = {"selfdriveStateSP", "onroadEvents", "managerState"}
SWAGLOG_PATTERN = re.compile(
  r"commIssue|cameraFrameRate|VTSC slow update|VTSC preview encode slow|"
  r"plannerd|radard|selfdrived|camerad|modeld|not_alive|not_freq_ok",
  re.IGNORECASE,
)


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
  if not seg_dir.is_dir():
    return False
  if (seg_dir / "rlog.lock").exists():
    return False
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


def _write_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "w", encoding="utf-8") as f:
    f.write(text)
  os.replace(tmp, path)


def _safe_float(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except Exception:
    return float(default)


def _status_name(status: int) -> str:
  return {
    0: "RED",
    1: "YELLOW",
    2: "GREEN",
  }.get(int(status), f"UNKNOWN({status})")


def extract_event_names(events_msg: Any) -> list[str]:
  try:
    return sorted({str(e.name) for e in events_msg})
  except Exception:
    return []


def extract_subsystem_statuses(ss_sp: Any) -> dict[str, int]:
  out: dict[str, int] = {}
  try:
    for st in ss_sp.subsystemStatuses:
      out[str(st.name)] = int(st.status)
  except Exception:
    pass
  return out


def extract_manager_processes(manager_state: Any) -> tuple[list[dict[str, Any]], list[str]]:
  procs: list[dict[str, Any]] = []
  down: list[str] = []
  try:
    for proc in manager_state.processes:
      entry = {
        "name": str(proc.name),
        "pid": int(proc.pid),
        "running": bool(proc.running),
        "shouldBeRunning": bool(proc.shouldBeRunning),
        "exitCode": int(proc.exitCode),
      }
      procs.append(entry)
      if entry["shouldBeRunning"] and not entry["running"]:
        down.append(entry["name"])
  except Exception:
    return [], []
  return procs, sorted(set(down))


def build_service_health(sm: Any, services: list[str]) -> dict[str, dict[str, bool]]:
  out: dict[str, dict[str, bool]] = {}
  for svc in services:
    out[str(svc)] = {
      "alive": bool(sm.alive.get(svc, True)),
      "freq_ok": bool(sm.freq_ok.get(svc, True)),
      "valid": bool(sm.valid.get(svc, True)),
    }
  return out


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
  subsystems = row.get("subsystems", {})
  service_health = row.get("serviceHealth", {})
  manager_not_running = list(row.get("managerNotRunning", []))
  bad_subsystems = sorted(name for name, status in subsystems.items() if int(status) != 2)
  red_subsystems = sorted(name for name, status in subsystems.items() if int(status) == 0)
  yellow_subsystems = sorted(name for name, status in subsystems.items() if int(status) == 1)
  service_not_alive = sorted(name for name, health in service_health.items() if not bool(health.get("alive", True)))
  service_not_freq_ok = sorted(name for name, health in service_health.items() if not bool(health.get("freq_ok", True)))
  service_invalid = sorted(name for name, health in service_health.items() if not bool(health.get("valid", True)))
  critical_service_faults = sorted(
    name for name in CRITICAL_RECORDER_SERVICES
    if name in service_health and (
      not bool(service_health[name].get("alive", True)) or
      not bool(service_health[name].get("freq_ok", True)) or
      not bool(service_health[name].get("valid", True))
    )
  )
  trigger_manager_down = sorted(name for name in manager_not_running if name not in TRIGGER_IGNORED_MANAGER_PROCESSES)
  comm_events = sorted(name for name in row.get("onroadEvents", []) if name in COMM_EVENT_NAMES)

  degraded = bool(bad_subsystems or comm_events or trigger_manager_down or critical_service_faults)
  trigger_reasons: list[str] = []
  if bad_subsystems:
    trigger_reasons.append("subsystem_not_green")
  if comm_events:
    trigger_reasons.append("comm_event")
  if trigger_manager_down:
    trigger_reasons.append("manager_process_down")
  if critical_service_faults:
    trigger_reasons.append("critical_service_fault")

  evaluated = dict(row)
  evaluated.update({
    "badSubsystems": bad_subsystems,
    "redSubsystems": red_subsystems,
    "yellowSubsystems": yellow_subsystems,
    "serviceNotAlive": service_not_alive,
    "serviceNotFreqOk": service_not_freq_ok,
    "serviceInvalid": service_invalid,
    "criticalServiceFaults": critical_service_faults,
    "triggerManagerNotRunning": trigger_manager_down,
    "commEvents": comm_events,
    "degraded": degraded,
    "triggerReasons": trigger_reasons,
  })
  return evaluated


def summarize_trace_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
  bad_subsystems = collections.Counter()
  red_subsystems = collections.Counter()
  yellow_subsystems = collections.Counter()
  comm_events = collections.Counter()
  service_not_alive = collections.Counter()
  service_not_freq_ok = collections.Counter()
  service_invalid = collections.Counter()
  manager_not_running = collections.Counter()
  critical_service_faults = collections.Counter()
  degraded_rows = 0
  v_ego_vals: list[float] = []
  t_vals: list[float] = []
  degraded_t: list[float] = []

  for row in rows:
    t = _safe_float(row.get("t"), -1.0)
    if t >= 0.0:
      t_vals.append(t)
    v_ego = _safe_float(row.get("vEgo"), -1.0)
    if v_ego >= 0.0:
      v_ego_vals.append(v_ego)
    if bool(row.get("degraded")):
      degraded_rows += 1
      if t >= 0.0:
        degraded_t.append(t)
    for name in row.get("badSubsystems", []):
      bad_subsystems[name] += 1
    for name in row.get("redSubsystems", []):
      red_subsystems[name] += 1
    for name in row.get("yellowSubsystems", []):
      yellow_subsystems[name] += 1
    for name in row.get("commEvents", []):
      comm_events[name] += 1
    for name in row.get("serviceNotAlive", []):
      service_not_alive[name] += 1
    for name in row.get("serviceNotFreqOk", []):
      service_not_freq_ok[name] += 1
    for name in row.get("serviceInvalid", []):
      service_invalid[name] += 1
    for name in row.get("managerNotRunning", []):
      manager_not_running[name] += 1
    for name in row.get("criticalServiceFaults", []):
      critical_service_faults[name] += 1

  def _most_common(counter: collections.Counter[str]) -> list[dict[str, Any]]:
    return [{"name": k, "samples": int(v)} for k, v in counter.most_common()]

  return {
    "rows": int(len(rows)),
    "window_duration_s": (max(t_vals) - min(t_vals)) if len(t_vals) >= 2 else None,
    "degraded_rows": int(degraded_rows),
    "degraded_duration_s": (max(degraded_t) - min(degraded_t)) if len(degraded_t) >= 2 else (0.0 if degraded_t else None),
    "bad_subsystems": _most_common(bad_subsystems),
    "red_subsystems": _most_common(red_subsystems),
    "yellow_subsystems": _most_common(yellow_subsystems),
    "comm_events": _most_common(comm_events),
    "service_not_alive": _most_common(service_not_alive),
    "service_not_freq_ok": _most_common(service_not_freq_ok),
    "service_invalid": _most_common(service_invalid),
    "manager_not_running": _most_common(manager_not_running),
    "critical_service_faults": _most_common(critical_service_faults),
    "vEgo_min": min(v_ego_vals) if v_ego_vals else None,
    "vEgo_max": max(v_ego_vals) if v_ego_vals else None,
  }


def format_summary_report(*, meta: dict[str, Any], summary: dict[str, Any], trigger_row: dict[str, Any]) -> str:
  lines = [
    f"event_id: {meta.get('event_id', '')}",
    f"created_utc: {meta.get('created_utc', '')}",
    f"route: {meta.get('route', '')}",
    f"seg_guess: {meta.get('seg_guess', '')}",
    f"trigger_monotonic_s: {meta.get('t0_monotonic_s', '')}",
    f"trigger_reasons: {', '.join(trigger_row.get('triggerReasons', [])) or '-'}",
    f"bad_subsystems_at_trigger: {', '.join(trigger_row.get('badSubsystems', [])) or '-'}",
    f"red_subsystems_at_trigger: {', '.join(trigger_row.get('redSubsystems', [])) or '-'}",
    f"yellow_subsystems_at_trigger: {', '.join(trigger_row.get('yellowSubsystems', [])) or '-'}",
    f"comm_events_at_trigger: {', '.join(trigger_row.get('commEvents', [])) or '-'}",
    f"manager_not_running_at_trigger: {', '.join(trigger_row.get('managerNotRunning', [])) or '-'}",
    f"service_not_alive_at_trigger: {', '.join(trigger_row.get('serviceNotAlive', [])) or '-'}",
    f"service_not_freq_ok_at_trigger: {', '.join(trigger_row.get('serviceNotFreqOk', [])) or '-'}",
    f"service_invalid_at_trigger: {', '.join(trigger_row.get('serviceInvalid', [])) or '-'}",
    f"vEgo_at_trigger_mps: {trigger_row.get('vEgo', '')}",
    "",
    f"rows: {summary.get('rows')}",
    f"window_duration_s: {summary.get('window_duration_s')}",
    f"degraded_rows: {summary.get('degraded_rows')}",
    f"degraded_duration_s: {summary.get('degraded_duration_s')}",
    f"vEgo_window_min_mps: {summary.get('vEgo_min')}",
    f"vEgo_window_max_mps: {summary.get('vEgo_max')}",
    "",
  ]

  def _append_counter(title: str, key: str) -> None:
    lines.append(f"{title}:")
    items = summary.get(key, [])
    if not items:
      lines.append("  -")
      return
    for item in items:
      lines.append(f"  {item['name']}: {item['samples']}")

  _append_counter("bad_subsystems", "bad_subsystems")
  _append_counter("red_subsystems", "red_subsystems")
  _append_counter("yellow_subsystems", "yellow_subsystems")
  _append_counter("comm_events", "comm_events")
  _append_counter("service_not_alive", "service_not_alive")
  _append_counter("service_not_freq_ok", "service_not_freq_ok")
  _append_counter("service_invalid", "service_invalid")
  _append_counter("manager_not_running", "manager_not_running")
  _append_counter("critical_service_faults", "critical_service_faults")

  return "\n".join(lines) + "\n"


def _run_text_command(args: list[str]) -> str:
  try:
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
  except Exception as e:
    return f"command failed: {' '.join(args)}\n{e}\n"
  out = proc.stdout or ""
  err = proc.stderr or ""
  if err:
    out += ("\n" if out else "") + err
  return out


def _capture_process_snapshot() -> dict[str, Any]:
  raw_text = _run_text_command(["ps", "-eo", "pid,comm,cls,rtprio,pri,psr,pcpu,args", "--sort=-pcpu"])
  lines = raw_text.splitlines()
  hot_lines: list[str] = []
  per_core: dict[str, list[dict[str, Any]]] = {}
  header = lines[0] if lines else ""
  for line in lines[1:]:
    parts = line.strip().split(None, 7)
    if len(parts) < 8:
      continue
    try:
      pid = int(parts[0])
    except Exception:
      continue
    comm = parts[1]
    cls = parts[2]
    rtprio = parts[3]
    pri = parts[4]
    psr = parts[5]
    try:
      pcpu = float(parts[6])
    except Exception:
      pcpu = 0.0
    args = parts[7]
    if any(name in args or name == comm for name in PROCESS_SNAPSHOT_NAMES):
      hot_lines.append(line)
    core_key = str(psr)
    per_core.setdefault(core_key, []).append({
      "pid": pid,
      "comm": comm,
      "cls": cls,
      "rtprio": rtprio,
      "pri": pri,
      "psr": int(psr) if str(psr).isdigit() else psr,
      "pcpu": pcpu,
      "args": args,
    })

  for core_entries in per_core.values():
    core_entries.sort(key=lambda e: float(e.get("pcpu", 0.0)), reverse=True)
    del core_entries[8:]

  filtered_text = "\n".join(([header] if header else []) + hot_lines) + ("\n" if hot_lines else "")
  return {
    "raw_text": raw_text,
    "filtered_text": filtered_text,
    "per_core": per_core,
  }


def _read_proc_stat() -> dict[str, list[int]]:
  out: dict[str, list[int]] = {}
  try:
    with open("/proc/stat", "r", encoding="utf-8") as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) < 5 or not parts[0].startswith("cpu"):
          continue
        name = parts[0]
        vals: list[int] = []
        for raw in parts[1:]:
          try:
            vals.append(int(raw))
          except Exception:
            vals.append(0)
        out[name] = vals
  except Exception:
    return {}
  return out


def _sample_cpu_load(interval_s: float = 0.15) -> dict[str, float]:
  start = _read_proc_stat()
  if not start:
    return {}
  time.sleep(max(0.05, float(interval_s)))
  end = _read_proc_stat()
  out: dict[str, float] = {}
  for name, a in start.items():
    b = end.get(name)
    if b is None:
      continue
    n = min(len(a), len(b))
    if n == 0:
      continue
    total_a = sum(a[:n])
    total_b = sum(b[:n])
    idle_a = a[3] + (a[4] if n > 4 else 0)
    idle_b = b[3] + (b[4] if n > 4 else 0)
    total_d = total_b - total_a
    idle_d = idle_b - idle_a
    if total_d <= 0:
      continue
    out[name] = round(100.0 * (1.0 - (idle_d / total_d)), 2)
  return out


def _capture_host_snapshot() -> dict[str, Any]:
  head = ""
  try:
    head = _run_text_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).strip()
  except Exception:
    head = ""
  loadavg = ""
  try:
    with open("/proc/loadavg", "r", encoding="utf-8") as f:
      loadavg = f.read().strip()
  except Exception:
    loadavg = ""
  return {
    "head": head,
    "loadavg": loadavg,
    "cpu_load_percent": _sample_cpu_load(),
  }


def _capture_filtered_swaglog_tail(*, max_lines: int = 1200) -> str:
  hits: list[str] = []
  try:
    files = sorted(Path("/data/log").glob("swaglog.*"))[-5:]
  except Exception:
    files = []
  for path in files:
    try:
      with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-2000:]
    except Exception:
      continue
    for line in lines:
      if SWAGLOG_PATTERN.search(line):
        hits.append(f"{path}:{line.rstrip()}")
  return "\n".join(hits[-max_lines:]) + ("\n" if hits else "")


def _start_segment_copy_worker(event_dir: Path, route: str, seg0: int) -> None:
  segs = [seg0 - 1, seg0, seg0 + 1]
  out_seg_dir = event_dir / "segments"
  for seg in segs:
    if seg < 0:
      continue
    timeout = 15.0 if seg == seg0 - 1 else 150.0
    sd = _wait_for_segment(route, seg, timeout_s=timeout)
    if sd is None:
      continue
    dst = out_seg_dir / f"{seg:03d}"
    _copy_if_exists(sd / "rlog.zst", dst / "rlog.zst")
    _copy_if_exists(sd / "qlog.zst", dst / "qlog.zst")


def _kick_copy_worker(event_dir: Path, route: str, seg0: int) -> None:
  try:
    import threading
    t = threading.Thread(target=_start_segment_copy_worker, args=(event_dir, route, seg0), daemon=True)
    t.start()
  except Exception:
    pass


def _build_row(sm: Any, *, route: str) -> dict[str, Any]:
  try:
    ss_sp = sm["selfdriveStateSP"]
  except Exception:
    ss_sp = None
  try:
    events_msg = sm["onroadEvents"]
  except Exception:
    events_msg = None
  try:
    manager_state = sm["managerState"]
  except Exception:
    manager_state = None
  try:
    device_state = sm["deviceState"]
  except Exception:
    device_state = None
  try:
    car_state = sm["carState"]
  except Exception:
    car_state = None

  subsystems = extract_subsystem_statuses(ss_sp)
  manager_processes, manager_not_running = extract_manager_processes(manager_state)
  row = {
    "t": float(time.monotonic()),
    "route": route,
    "started": bool(getattr(device_state, "started", False)) if device_state is not None else False,
    "allSystemsReady": bool(getattr(ss_sp, "allSystemsReady", False)) if ss_sp is not None else False,
    "subsystems": subsystems,
    "subsystemStatusNames": {name: _status_name(status) for name, status in subsystems.items()},
    "onroadEvents": extract_event_names(events_msg),
    "managerNotRunning": manager_not_running,
    "managerProcesses": manager_processes,
    "serviceHealth": build_service_health(sm, SERVICE_HEALTH_NAMES),
    "vEgo": _safe_float(getattr(car_state, "vEgo", 0.0) if car_state is not None else 0.0),
  }
  return evaluate_row(row)


def _finalize_event(*, event_dir: Path, t0: float, trace: collections.deque[dict[str, Any]], pre_s: float,
                    post_s: float, meta: dict[str, Any], max_total_mb: int, events_root: Path) -> None:
  window = [r for r in list(trace) if (t0 - pre_s) <= float(r.get("t", 0.0)) <= (t0 + post_s)]
  summary = summarize_trace_window(window)
  trigger_row = next((r for r in window if abs(float(r.get("t", 0.0)) - t0) < 0.101), meta.get("trigger_row", {}))
  if not trigger_row and window:
    trigger_row = min(window, key=lambda r: abs(float(r.get("t", 0.0)) - t0))

  _write_jsonl(event_dir / "trace_20s.jsonl", window)
  _write_json(event_dir / "summary.json", summary)
  _write_json(event_dir / "trigger_row.json", trigger_row)
  _write_text(event_dir / "report.txt", format_summary_report(meta=meta, summary=summary, trigger_row=trigger_row))
  _write_text(event_dir / "swaglog_tail.txt", _capture_filtered_swaglog_tail())
  meta_out = dict(meta)
  meta_out["summary"] = summary
  _write_json(event_dir / "event.json", meta_out)

  _prune_old_events(events_root, max_total_mb=max_total_mb)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--events-dir", type=str, default=str(EVENTS_DIR_DEFAULT))
  ap.add_argument("--pre-seconds", type=float, default=10.0)
  ap.add_argument("--post-seconds", type=float, default=10.0)
  ap.add_argument("--buffer-seconds", type=float, default=40.0)
  ap.add_argument("--sample-hz", type=float, default=10.0)
  ap.add_argument("--max-total-mb", type=int, default=DEFAULT_MAX_TOTAL_MB)
  ap.add_argument("--cooldown-seconds", type=float, default=8.0)
  ap.add_argument("--startup-grace-seconds", type=float, default=5.0)
  args = ap.parse_args()

  events_root = Path(args.events_dir)
  pre_s = float(args.pre_seconds)
  post_s = float(args.post_seconds)
  dt = 1.0 / max(1.0, float(args.sample_hz))
  buf_len = int(max(20, float(args.buffer_seconds) / dt))

  services = [
    "selfdriveStateSP",
    "onroadEvents",
    "managerState",
    "deviceState",
    "carState",
    "pandaStates",
    "peripheralState",
    "roadCameraState",
    "driverCameraState",
    "wideRoadCameraState",
    "modelV2",
    "livePose",
    "liveCalibration",
    "liveParameters",
    "radarState",
    "driverMonitoringState",
    "controlsState",
    "carOutput",
    "carControl",
    "longitudinalPlan",
    "driverAssistance",
  ]
  sm = messaging.SubMaster(services, poll="selfdriveStateSP")

  trace: collections.deque[dict[str, Any]] = collections.deque(maxlen=buf_len)
  next_sample_t = 0.0
  next_route_check_t = 0.0
  route_cache = ""
  last_trigger_t = -1e9
  prev_degraded = False
  started_since = None
  pending: dict[str, Any] | None = None

  print("[comm_issue_recorder] running. waiting for readiness/comm events...", flush=True)

  while True:
    now = time.monotonic()
    if now < next_sample_t:
      time.sleep(min(0.02, next_sample_t - now))
      continue

    sm.update(0)
    sample_t = time.monotonic()
    next_sample_t = sample_t + dt

    if sample_t >= next_route_check_t:
      route_cache = _safe_read_current_route()
      next_route_check_t = sample_t + 2.0

    row = _build_row(sm, route=route_cache)
    trace.append(row)

    started = bool(row.get("started", False))
    if started:
      if started_since is None:
        started_since = sample_t
    else:
      started_since = None

    if pending is not None:
      if sample_t >= float(pending["t0"]) + post_s or (not started and sample_t >= float(pending["t0"]) + 1.0):
        _finalize_event(
          event_dir=Path(pending["event_dir"]),
          t0=float(pending["t0"]),
          trace=trace,
          pre_s=pre_s,
          post_s=post_s,
          meta=dict(pending["meta"]),
          max_total_mb=int(args.max_total_mb),
          events_root=events_root,
        )
        print(f"[comm_issue_recorder] event finalized {Path(pending['event_dir']).name}", flush=True)
        pending = None

    if not started:
      prev_degraded = False
      continue

    startup_grace = started_since is None or (sample_t - started_since) < float(args.startup_grace_seconds)
    degraded = bool(row.get("degraded", False))
    rising_edge = degraded and not prev_degraded
    prev_degraded = degraded

    if startup_grace or not rising_edge:
      continue
    if (sample_t - last_trigger_t) < float(args.cooldown_seconds):
      continue

    last_trigger_t = sample_t
    route = _safe_read_current_route()
    seg0 = _guess_current_segment(route) if route else None
    event_id = f"{_now_utc_tag()}_comm"
    if route:
      event_id += f"_{route}"
    if seg0 is not None:
      event_id += f"_seg{seg0:03d}"

    event_dir = events_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    host_snapshot = _capture_host_snapshot()
    process_snapshot = _capture_process_snapshot()
    _write_json(event_dir / "host_snapshot_trigger.json", host_snapshot)
    _write_json(event_dir / "cpu_process_snapshot_trigger.json", process_snapshot["per_core"])
    _write_text(event_dir / "cpu_process_snapshot_trigger.txt", process_snapshot["filtered_text"] or process_snapshot["raw_text"])
    _write_text(event_dir / "ps_full_trigger.txt", process_snapshot["raw_text"])

    meta = {
      "event_id": event_id,
      "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
      "t0_monotonic_s": sample_t,
      "route": route,
      "seg_guess": seg0,
      "trigger_row": row,
      "trigger_reasons": row.get("triggerReasons", []),
      "bad_subsystems": row.get("badSubsystems", []),
      "red_subsystems": row.get("redSubsystems", []),
      "yellow_subsystems": row.get("yellowSubsystems", []),
      "comm_events": row.get("commEvents", []),
      "manager_not_running": row.get("managerNotRunning", []),
      "service_not_alive": row.get("serviceNotAlive", []),
      "service_not_freq_ok": row.get("serviceNotFreqOk", []),
      "service_invalid": row.get("serviceInvalid", []),
      "vEgo": row.get("vEgo"),
      "host_snapshot": host_snapshot,
    }
    _write_json(event_dir / "event.json", meta)
    _write_json(event_dir / "trigger_row.json", row)

    if route and seg0 is not None:
      _kick_copy_worker(event_dir, route, int(seg0))

    pending = {"t0": sample_t, "event_dir": str(event_dir), "meta": meta}
    print(
      f"[comm_issue_recorder] TRIGGER {event_id} reasons={row.get('triggerReasons', [])} "
      f"bad={row.get('badSubsystems', [])} comm={row.get('commEvents', [])}",
      flush=True,
    )

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
