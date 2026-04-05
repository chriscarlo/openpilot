#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cereal.messaging import SubMaster
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params

THERMAL_STATUS_NAMES = {
  0: "green",
  1: "yellow",
  2: "red",
  3: "danger",
}

WATCH_PATTERNS = {
  "objectd": ("sunnypilot.objectd.objectd",),
  "modeld": ("selfdrive.modeld.modeld", "sunnypilot.modeld.modeld"),
  "camerad": ("./camerad", "camerad"),
  "plannerd": ("selfdrive.controls.plannerd",),
  "controlsd": ("selfdrive.controls.controlsd",),
  "card": ("selfdrive.car.card",),
  "ui": ("./ui",),
  "hardwared": ("system.hardware.hardwared",),
  "dmonitoringmodeld": ("selfdrive.modeld.dmonitoringmodeld",),
  "dmonitoringd": ("selfdrive.monitoring.dmonitoringd",),
}

LOG_KEYWORDS = ("objectd", "objectHazard", "SNPE", "hazard", "VisionIPC", "modeld")
CORE_COLLISION_PAIRS = (
  ("objectd", "modeld"),
  ("objectd", "camerad"),
  ("objectd", "plannerd"),
)


def mean(values: list[float]) -> float:
  return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return float(ordered[0])
  rank = (len(ordered) - 1) * pct
  low = int(rank)
  high = min(low + 1, len(ordered) - 1)
  frac = rank - low
  return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def safe_float(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except Exception:
    return default


def safe_bool(value: Any, default: bool = False) -> bool:
  try:
    return bool(value)
  except Exception:
    return default


def safe_text(value: Any, default: str = "") -> str:
  try:
    return str(value)
  except Exception:
    return default


def proc_name(proc: Any) -> str:
  name = safe_text(proc.name, "")
  if len(proc.cmdline):
    name = safe_text(proc.cmdline[0], name)
  if len(proc.exe):
    name = f"{safe_text(proc.exe)} - {name}"
  return name


def proc_cpu_time(proc: Any) -> float:
  return (
    safe_float(proc.cpuUser) +
    safe_float(proc.cpuSystem) +
    safe_float(proc.cpuChildrenUser) +
    safe_float(proc.cpuChildrenSystem)
  )


def canonical_process_name(name: str) -> str | None:
  for canonical, patterns in WATCH_PATTERNS.items():
    if any(pattern in name for pattern in patterns):
      return canonical
  return None


def read_gpubusy_percent() -> float | None:
  path = Path("/sys/class/kgsl/kgsl-3d0/gpubusy")
  if not path.exists():
    return None
  try:
    busy_raw, total_raw = path.read_text().strip().split()[:2]
    busy = int(busy_raw)
    total = int(total_raw)
    if total <= 0:
      return None
    return round((busy / total) * 100.0, 1)
  except Exception:
    return None


def run_ps_snapshot() -> dict[str, dict[str, Any]]:
  result = {name: {"cpu_pct": 0.0, "rss_mb": 0.0, "psr_counts": Counter()} for name in WATCH_PATTERNS}
  try:
    output = subprocess.check_output(
      ["ps", "-eo", "pid,psr,pcpu,rss,args", "--no-headers"],
      text=True,
      stderr=subprocess.DEVNULL,
    )
  except Exception:
    return result

  for line in output.splitlines():
    parts = line.split(None, 4)
    if len(parts) != 5:
      continue
    _, psr, pcpu, rss, args = parts
    canonical = canonical_process_name(args)
    if canonical is None:
      continue
    try:
      result[canonical]["cpu_pct"] += float(pcpu)
    except ValueError:
      pass
    try:
      result[canonical]["rss_mb"] += int(rss) / 1024.0
    except ValueError:
      pass
    try:
      result[canonical]["psr_counts"][int(psr)] += 1
    except ValueError:
      pass

  return result


def build_services() -> list[str]:
  wanted = [
    "deviceState",
    "procLog",
    "longitudinalPlan",
    "longitudinalPlanSP",
    "objectHazardStateSP",
    "logMessage",
    "managerState",
  ]
  return [service for service in wanted if service in SERVICE_LIST]


def collect_object_hazard(sm: SubMaster) -> dict[str, Any] | None:
  if "objectHazardStateSP" not in sm.data or not sm.seen.get("objectHazardStateSP", False):
    return None
  state = sm["objectHazardStateSP"]
  return {
    "enabled": safe_bool(state.enabled),
    "active": safe_bool(state.active),
    "model_ready": safe_bool(state.modelReady),
    "backend": safe_text(state.backend),
    "hazard_on_path": safe_bool(state.hazardOnPath),
    "stop_required": safe_bool(state.stopRequired),
    "recommended_speed": safe_float(state.recommendedSpeed),
    "hazard_distance_m": safe_float(state.hazardDistanceM),
    "hazard_confidence": safe_float(state.hazardConfidence),
    "hazard_class": safe_text(state.hazardClass),
    "source_frame_id": int(getattr(state, "sourceFrameId", 0)),
  }


def collect_planner_hazard(sm: SubMaster) -> dict[str, Any] | None:
  if "longitudinalPlanSP" not in sm.data or not sm.seen.get("longitudinalPlanSP", False):
    return None
  state = sm["longitudinalPlanSP"]
  try:
    control = state.objectHazardControl
  except Exception:
    return None

  return {
    "enabled": safe_bool(control.enabled),
    "active": safe_bool(control.active),
    "recommended_speed": safe_float(control.recommendedSpeed),
    "stop_required": safe_bool(control.stopRequired),
    "hazard_distance_m": safe_float(control.hazardDistanceM),
    "hazard_confidence": safe_float(control.hazardConfidence),
    "hazard_class": safe_text(control.hazardClass),
  }


def update_all_process_summaries(
  proc_log: Any,
  prev_proc_log: Any | None,
  dt: float,
  all_cpu_totals: dict[str, float],
  all_cpu_peaks: dict[str, float],
  all_rss_peaks: dict[str, float],
  watched_cpu: dict[str, float],
  watched_rss: dict[str, float],
) -> None:
  prev_by_pid = {}
  if prev_proc_log is not None:
    prev_by_pid = {proc.pid: proc for proc in prev_proc_log.procs}

  for proc in proc_log.procs:
    name = proc_name(proc)
    canonical = canonical_process_name(name)
    rss_mb = safe_float(proc.memRss) / 1e6
    all_rss_peaks[name] = max(all_rss_peaks.get(name, 0.0), rss_mb)
    if canonical is not None:
      watched_rss[canonical] += rss_mb

    cpu_pct = 0.0
    prev_proc = prev_by_pid.get(proc.pid)
    if prev_proc is not None and dt > 0.0:
      cpu_pct = max(0.0, (proc_cpu_time(proc) - proc_cpu_time(prev_proc)) / dt * 100.0)
      all_cpu_totals[name] += cpu_pct
      all_cpu_peaks[name] = max(all_cpu_peaks.get(name, 0.0), cpu_pct)
      if canonical is not None:
        watched_cpu[canonical] += cpu_pct


def dominant_core(psr_counts: Counter[int]) -> int | None:
  if not psr_counts:
    return None
  return psr_counts.most_common(1)[0][0]


def main() -> None:
  parser = argparse.ArgumentParser(description="Collect live object-hazard and resource telemetry on-device.")
  parser.add_argument("--duration", type=float, default=45.0)
  parser.add_argument("--interval", type=float, default=1.0)
  parser.add_argument("--top-n", type=int, default=8)
  args = parser.parse_args()

  services = build_services()
  sm = SubMaster(services)
  params = Params()
  object_hazard_enabled = params.get_bool("ObjectHazardEnabled")

  recent_logs: deque[str] = deque(maxlen=20)
  tail_samples: deque[dict[str, Any]] = deque(maxlen=5)

  memory_samples: list[float] = []
  gpu_samples: list[float] = []
  gpubusy_samples: list[float] = []
  cpu_core_samples: list[list[float]] = []
  cpu_temp_samples: list[float] = []
  gpu_temp_samples: list[float] = []
  thermal_samples: list[int] = []

  watched_cpu_series: dict[str, list[float]] = defaultdict(list)
  watched_rss_series: dict[str, list[float]] = defaultdict(list)
  watched_core_counts: dict[str, Counter[int]] = defaultdict(Counter)
  collision_counts: Counter[str] = Counter()

  all_cpu_totals: dict[str, float] = defaultdict(float)
  all_cpu_peaks: dict[str, float] = defaultdict(float)
  all_rss_peaks: dict[str, float] = defaultdict(float)
  proc_intervals = 0

  samples = 0
  started_samples = 0
  hazard_seen_samples = 0
  hazard_active_samples = 0
  hazard_model_ready_samples = 0
  planner_hazard_active_samples = 0
  planner_hazard_stop_samples = 0
  long_should_stop_samples = 0
  objectd_present_samples = 0
  backends_seen: set[str] = set()
  mismatch_counts = Counter()

  prev_proc_log = None
  prev_proc_log_time = None

  warmup_deadline = time.monotonic() + 10.0
  while time.monotonic() < warmup_deadline and not (sm.seen.get("deviceState", False) and sm.seen.get("procLog", False)):
    sm.update(1000)

  deadline = time.monotonic() + max(args.duration, args.interval)
  while time.monotonic() < deadline:
    sm.update(int(max(args.interval, 0.2) * 1000))

    device = sm["deviceState"] if sm.seen.get("deviceState", False) else None
    proc_log = sm["procLog"] if sm.seen.get("procLog", False) else None

    if "logMessage" in sm.data and sm.updated.get("logMessage", False):
      log_line = safe_text(sm["logMessage"].logMessage).strip()
      if log_line and any(keyword in log_line for keyword in LOG_KEYWORDS):
        recent_logs.append(log_line)

    gpubusy_pct = read_gpubusy_percent()
    ps_snapshot = run_ps_snapshot()
    watched_cpu = defaultdict(float)
    watched_rss = defaultdict(float)

    dt = 0.0
    if proc_log is not None and prev_proc_log is not None and prev_proc_log_time is not None:
      dt = max(0.0, (sm.logMonoTime["procLog"] - prev_proc_log_time) / 1e9)
      if dt > 0.0:
        proc_intervals += 1
        update_all_process_summaries(
          proc_log,
          prev_proc_log,
          dt,
          all_cpu_totals,
          all_cpu_peaks,
          all_rss_peaks,
          watched_cpu,
          watched_rss,
        )

    if proc_log is not None:
      prev_proc_log = proc_log
      prev_proc_log_time = sm.logMonoTime["procLog"]

    for name, stats in ps_snapshot.items():
      watched_rss[name] = max(watched_rss[name], safe_float(stats["rss_mb"]))
      watched_cpu[name] = max(watched_cpu[name], safe_float(stats["cpu_pct"]))
      core = dominant_core(stats["psr_counts"])
      if core is not None:
        watched_core_counts[name][core] += 1

    for name in WATCH_PATTERNS:
      watched_cpu_series[name].append(round(watched_cpu.get(name, 0.0), 2))
      watched_rss_series[name].append(round(watched_rss.get(name, 0.0), 2))

    objectd_core = dominant_core(ps_snapshot["objectd"]["psr_counts"])
    if objectd_core is not None:
      objectd_present_samples += 1
    for left, right in CORE_COLLISION_PAIRS:
      left_core = dominant_core(ps_snapshot[left]["psr_counts"])
      right_core = dominant_core(ps_snapshot[right]["psr_counts"])
      if left_core is not None and left_core == right_core:
        collision_counts[f"{left}/{right}"] += 1

    object_hazard = collect_object_hazard(sm)
    planner_hazard = collect_planner_hazard(sm)
    should_stop = None
    if "longitudinalPlan" in sm.data and sm.seen.get("longitudinalPlan", False):
      should_stop = safe_bool(sm["longitudinalPlan"].shouldStop)
      if should_stop:
        long_should_stop_samples += 1

    if device is not None:
      samples += 1
      started = safe_bool(device.started)
      if started:
        started_samples += 1
      cpu_per_core = [safe_float(v) for v in device.cpuUsagePercent]
      memory_pct = safe_float(device.memoryUsagePercent)
      gpu_pct = safe_float(device.gpuUsagePercent)
      cpu_temp = max([safe_float(v) for v in device.cpuTempC] or [0.0])
      gpu_temp = max([safe_float(v) for v in device.gpuTempC] or [0.0])
      thermal = int(getattr(device, "thermalStatus", 0))

      memory_samples.append(memory_pct)
      gpu_samples.append(gpu_pct)
      cpu_core_samples.append(cpu_per_core)
      cpu_temp_samples.append(cpu_temp)
      gpu_temp_samples.append(gpu_temp)
      thermal_samples.append(thermal)
      if gpubusy_pct is not None:
        gpubusy_samples.append(gpubusy_pct)

      if object_hazard is not None:
        hazard_seen_samples += 1
        if object_hazard["active"]:
          hazard_active_samples += 1
        if object_hazard["model_ready"]:
          hazard_model_ready_samples += 1
        if object_hazard["backend"]:
          backends_seen.add(object_hazard["backend"])

      if planner_hazard is not None:
        if planner_hazard["active"]:
          planner_hazard_active_samples += 1
        if planner_hazard["stop_required"]:
          planner_hazard_stop_samples += 1

      if object_hazard is not None and object_hazard["active"] and (planner_hazard is None or not planner_hazard["active"]):
        mismatch_counts["hazard_not_reaching_planner"] += 1
      if planner_hazard is not None and planner_hazard["stop_required"] and not should_stop:
        mismatch_counts["planner_stop_not_reaching_longitudinal_plan"] += 1
      if object_hazard is not None and object_hazard["enabled"] and not object_hazard["model_ready"]:
        mismatch_counts["objectd_enabled_but_model_not_ready"] += 1

      tail_samples.append({
        "started": started,
        "memory_pct": round(memory_pct, 1),
        "gpu_pct": round(gpu_pct, 1),
        "gpubusy_pct": gpubusy_pct,
        "cpu_pct_per_core": [round(v, 1) for v in cpu_per_core],
        "thermal_status": THERMAL_STATUS_NAMES.get(thermal, str(thermal)),
        "cpu_temp_max_c": round(cpu_temp, 1),
        "gpu_temp_max_c": round(gpu_temp, 1),
        "object_hazard": object_hazard,
        "planner_hazard": planner_hazard,
        "longitudinal_should_stop": should_stop,
        "watched_processes": {
          name: {
            "cpu_pct": round(watched_cpu.get(name, 0.0), 2),
            "rss_mb": round(watched_rss.get(name, 0.0), 2),
            "psr": dominant_core(ps_snapshot[name]["psr_counts"]),
          }
          for name in WATCH_PATTERNS
        },
      })

  core_columns = list(zip(*cpu_core_samples)) if cpu_core_samples else []
  device_summary = {
    "sample_count": samples,
    "started_samples": started_samples,
    "memory_mean_pct": round(mean(memory_samples), 1),
    "memory_p95_pct": round(percentile(memory_samples, 0.95), 1),
    "memory_peak_pct": round(max(memory_samples) if memory_samples else 0.0, 1),
    "gpu_mean_pct": round(mean(gpu_samples), 1),
    "gpu_p95_pct": round(percentile(gpu_samples, 0.95), 1),
    "gpu_peak_pct": round(max(gpu_samples) if gpu_samples else 0.0, 1),
    "gpubusy_mean_pct": round(mean(gpubusy_samples), 1) if gpubusy_samples else None,
    "gpubusy_p95_pct": round(percentile(gpubusy_samples, 0.95), 1) if gpubusy_samples else None,
    "gpubusy_peak_pct": round(max(gpubusy_samples), 1) if gpubusy_samples else None,
    "cpu_core_mean_pct": [round(mean(list(column)), 1) for column in core_columns],
    "cpu_core_p95_pct": [round(percentile(list(column), 0.95), 1) for column in core_columns],
    "cpu_core_peak_pct": [round(max(list(column)), 1) for column in core_columns],
    "cpu_temp_peak_c": round(max(cpu_temp_samples) if cpu_temp_samples else 0.0, 1),
    "gpu_temp_peak_c": round(max(gpu_temp_samples) if gpu_temp_samples else 0.0, 1),
    "thermal_peak": max(thermal_samples) if thermal_samples else 0,
    "thermal_peak_name": THERMAL_STATUS_NAMES.get(max(thermal_samples) if thermal_samples else 0, "green"),
  }

  watched_process_summary = {}
  for name in WATCH_PATTERNS:
    watched_process_summary[name] = {
      "avg_cpu_pct": round(mean(watched_cpu_series[name]), 2),
      "p95_cpu_pct": round(percentile(watched_cpu_series[name], 0.95), 2),
      "peak_cpu_pct": round(max(watched_cpu_series[name]) if watched_cpu_series[name] else 0.0, 2),
      "peak_rss_mb": round(max(watched_rss_series[name]) if watched_rss_series[name] else 0.0, 2),
      "dominant_psr": dominant_core(watched_core_counts[name]),
      "samples_seen": len(watched_cpu_series[name]),
    }

  cpu_rank = sorted(
    ({
      "name": name,
      "avg_cpu_pct": round(total / proc_intervals, 2),
      "peak_cpu_pct": round(all_cpu_peaks.get(name, 0.0), 2),
    } for name, total in all_cpu_totals.items()),
    key=lambda item: item["avg_cpu_pct"],
    reverse=True,
  )[:args.top_n] if proc_intervals else []

  rss_rank = sorted(
    ({
      "name": name,
      "peak_rss_mb": round(rss, 2),
    } for name, rss in all_rss_peaks.items()),
    key=lambda item: item["peak_rss_mb"],
    reverse=True,
  )[:args.top_n]

  result = {
    "meta": {
      "repo_path": str(Path.cwd()),
      "duration_s": args.duration,
      "interval_s": args.interval,
      "services_monitored": services,
      "service_presence": {service: service in SERVICE_LIST for service in services},
      "object_hazard_param_enabled": object_hazard_enabled,
      "capture_end_monotonic_s": round(time.monotonic(), 3),
    },
    "summary": {
      "device": device_summary,
      "hazard": {
        "service_present": "objectHazardStateSP" in services,
        "service_seen_samples": hazard_seen_samples,
        "active_samples": hazard_active_samples,
        "model_ready_samples": hazard_model_ready_samples,
        "planner_active_samples": planner_hazard_active_samples,
        "planner_stop_required_samples": planner_hazard_stop_samples,
        "longitudinal_should_stop_samples": long_should_stop_samples,
        "objectd_present_samples": objectd_present_samples,
        "backends_seen": sorted(backends_seen),
        "mismatch_counts": dict(mismatch_counts),
      },
      "watched_processes": watched_process_summary,
      "core_collisions": {
        pair: {
          "same_core_samples": count,
          "same_core_ratio": round(count / samples, 3) if samples else 0.0,
        }
        for pair, count in collision_counts.items()
      },
      "top_cpu_processes": cpu_rank,
      "top_rss_processes": rss_rank,
    },
    "tail_samples": list(tail_samples),
    "log_tail": list(recent_logs),
  }
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
