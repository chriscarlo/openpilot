#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REMOTE_PROBE = Path(__file__).with_name("remote_probe.py")


def build_remote_command(remote_repo: str, duration: float, interval: float, top_n: int) -> str:
  return (
    "set -euo pipefail; "
    f"cd {shlex.quote(remote_repo)}; "
    "source /usr/local/venv/bin/activate; "
    f"python3 - --duration {duration:.3f} --interval {interval:.3f} --top-n {int(top_n)}"
  )


def run_remote_probe(ssh_profile: str, remote_repo: str, duration: float, interval: float, top_n: int) -> dict[str, Any]:
  cmd = [
    "ssh",
    ssh_profile,
    f"bash -lc {shlex.quote(build_remote_command(remote_repo, duration, interval, top_n))}",
  ]
  result = subprocess.run(
    cmd,
    input=REMOTE_PROBE.read_text(),
    text=True,
    capture_output=True,
    timeout=max(30.0, duration + 30.0),
  )
  if result.returncode != 0:
    stderr = result.stderr.strip() or "<no stderr>"
    stdout = result.stdout.strip()
    hint = ""
    if result.returncode == 255:
      hint = "\nHint: verify `ssh {profile} true` works first and that the configured identity file exists.".format(
        profile=ssh_profile,
      )
    raise RuntimeError(
      f"remote probe failed with exit {result.returncode}{hint}\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
    )
  try:
    return json.loads(result.stdout)
  except json.JSONDecodeError as err:
    raise RuntimeError(f"remote probe returned invalid JSON: {err}\nRaw output:\n{result.stdout}") from err


def fmt_percent(value: float | None) -> str:
  if value is None:
    return "n/a"
  return f"{value:.1f}%"


def fmt_temp(value: float | None) -> str:
  if value is None:
    return "n/a"
  return f"{value:.1f}C"


def render_device_summary(summary: dict[str, Any]) -> list[str]:
  device = summary["device"]
  lines = [
    f"Device: started {device['started_samples']}/{device['sample_count']} samples, "
    f"memory {fmt_percent(device['memory_mean_pct'])} mean / {fmt_percent(device['memory_p95_pct'])} p95 / {fmt_percent(device['memory_peak_pct'])} peak, "
    f"GPU {fmt_percent(device['gpu_mean_pct'])} mean / {fmt_percent(device['gpu_p95_pct'])} p95 / {fmt_percent(device['gpu_peak_pct'])} peak.",
    f"GPU busy: {fmt_percent(device['gpubusy_mean_pct'])} mean / {fmt_percent(device['gpubusy_p95_pct'])} p95 / {fmt_percent(device['gpubusy_peak_pct'])} peak. "
    f"Thermals: {device['thermal_peak_name']} peak, CPU {fmt_temp(device['cpu_temp_peak_c'])}, GPU {fmt_temp(device['gpu_temp_peak_c'])}.",
  ]
  hot_cores = [
    f"cpu{i}={device['cpu_core_p95_pct'][i]:.1f}%"
    for i in range(len(device["cpu_core_p95_pct"]))
    if device["cpu_core_p95_pct"][i] >= 70.0
  ]
  if hot_cores:
    lines.append("Hot cores by p95: " + ", ".join(hot_cores))
  return lines


def render_hazard_summary(summary: dict[str, Any], meta: dict[str, Any]) -> list[str]:
  hazard = summary["hazard"]
  backends = ", ".join(hazard["backends_seen"]) if hazard["backends_seen"] else "none"
  lines = [
    f"Hazard path: param enabled={meta['object_hazard_param_enabled']}, service present={hazard['service_present']}, "
    f"objectd seen in {hazard['objectd_present_samples']} samples, model ready in {hazard['model_ready_samples']} samples, backends seen={backends}.",
    f"Hazard activity: object active={hazard['active_samples']} samples, planner active={hazard['planner_active_samples']}, "
    f"planner stopRequired={hazard['planner_stop_required_samples']}, longitudinal shouldStop={hazard['longitudinal_should_stop_samples']}.",
  ]
  if hazard["mismatch_counts"]:
    mismatch_text = ", ".join(f"{name}={count}" for name, count in sorted(hazard["mismatch_counts"].items()))
    lines.append("Mismatches: " + mismatch_text)
  return lines


def render_process_summary(summary: dict[str, Any]) -> list[str]:
  watched = summary["watched_processes"]
  interesting = []
  for name in ("modeld", "objectd", "camerad", "plannerd", "ui", "hardwared"):
    proc = watched.get(name)
    if not proc:
      continue
    interesting.append(
      f"{name}: avg CPU {proc['avg_cpu_pct']:.2f}%, p95 {proc['p95_cpu_pct']:.2f}%, "
      f"peak RSS {proc['peak_rss_mb']:.1f} MB, core {proc['dominant_psr']}"
    )
  return interesting


def build_recommendations(report: dict[str, Any]) -> list[str]:
  meta = report["meta"]
  summary = report["summary"]
  device = summary["device"]
  hazard = summary["hazard"]
  watched = summary["watched_processes"]
  collisions = summary["core_collisions"]
  recommendations: list[str] = []

  if device["started_samples"] == 0:
    recommendations.append(
      "The device never appeared onroad during the capture. Force onroad or attach to the car before interpreting modeld or objectd resource numbers."
    )
    return recommendations

  if not meta["object_hazard_param_enabled"]:
    recommendations.append(
      "Enable `ObjectHazardEnabled` first. The pipeline is gated before manager/process work, so performance tuning is premature."
    )

  if not hazard["service_present"]:
    recommendations.append(
      "The deployed repo does not advertise `objectHazardStateSP`. Point the device at the experimental branch or deploy the worktree before debugging runtime behavior."
    )
    return recommendations

  if hazard["objectd_present_samples"] == 0:
    recommendations.append(
      "`objectd` never appeared in the process snapshot. Check manager gating, process registration, and whether the device was actually started onroad."
    )

  if hazard["model_ready_samples"] == 0:
    recommendations.append(
      "`objectd` stayed unready for the whole capture. Fix detector assets or accelerator runtime before touching planner thresholds."
    )

  if hazard["mismatch_counts"].get("hazard_not_reaching_planner", 0) > 0:
    recommendations.append(
      "Hazard state became active without planner-side `objectHazardControl` following it. Fix message subscription or planner merge wiring before changing detection sensitivity."
    )

  if hazard["mismatch_counts"].get("planner_stop_not_reaching_longitudinal_plan", 0) > 0:
    recommendations.append(
      "Planner `stopRequired` did not consistently reach `longitudinalPlan.shouldStop`. Fix the longitudinal merge path before tuning slowdown math."
    )

  hot_core_p95 = max(device["cpu_core_p95_pct"] or [0.0])
  gpu_p95 = max(
    [value for value in (device["gpu_p95_pct"], device["gpubusy_p95_pct"]) if value is not None] or [0.0]
  )

  collision_ratio = max((entry["same_core_ratio"] for entry in collisions.values()), default=0.0)
  if hot_core_p95 >= 85.0:
    if collision_ratio >= 0.25:
      recommendations.append(
        "One core is hot and `objectd` repeatedly shares a core with a critical process. Lower `objectd` cadence or detector input size first; only move scheduler placement if the cheaper reductions do not clear the collision."
      )
    else:
      recommendations.append(
        "CPU pressure is high without a strong single-core collision signal. Reduce `objectd` cadence, preprocessing cost, or debug bookkeeping before touching planner code."
      )

  if gpu_p95 >= 70.0:
    backend_text = ", ".join(hazard["backends_seen"]) if hazard["backends_seen"] else "unknown"
    recommendations.append(
      f"GPU pressure is high with backend(s) {backend_text}. Lower auxiliary cadence or input size first; if that is not enough, test a DSP backend. Do not add CPU inference as a fallback."
    )

  if device["memory_p95_pct"] >= 65.0:
    recommendations.append(
      "Memory pressure is above the usual onroad comfort band. Trim detection debug output, cached frame data, or other per-frame allocations before changing model behavior."
    )

  if device["thermal_peak"] >= 1:
    recommendations.append(
      "Thermal status reached yellow or worse. Shorten the run and lower auxiliary load before treating the observed behavior as a semantic bug."
    )

  objectd = watched.get("objectd", {})
  modeld = watched.get("modeld", {})
  if (
    not recommendations and
    objectd.get("p95_cpu_pct", 0.0) < 20.0 and
    modeld.get("p95_cpu_pct", 0.0) < 30.0 and
    gpu_p95 < 60.0 and
    device["memory_p95_pct"] < 65.0 and
    device["thermal_peak"] == 0
  ):
    recommendations.append(
      "The capture stayed inside a reasonable resource envelope. The next change should target hazard semantics or planner thresholds, not runtime load shedding."
    )

  return recommendations


def render_text_report(report: dict[str, Any]) -> str:
  lines: list[str] = []
  meta = report["meta"]
  summary = report["summary"]
  lines.extend(render_device_summary(summary))
  lines.extend(render_hazard_summary(summary, meta))
  process_lines = render_process_summary(summary)
  if process_lines:
    lines.append("Watched processes:")
    lines.extend(f"- {line}" for line in process_lines)

  top_cpu = summary["top_cpu_processes"][:5]
  if top_cpu:
    lines.append("Top CPU processes:")
    lines.extend(
      f"- {entry['name']}: avg {entry['avg_cpu_pct']:.2f}% / peak {entry['peak_cpu_pct']:.2f}%"
      for entry in top_cpu
    )

  log_tail = report.get("log_tail", [])[-5:]
  if log_tail:
    lines.append("Relevant log tail:")
    lines.extend(f"- {line}" for line in log_tail)

  recommendations = build_recommendations(report)
  if recommendations:
    lines.append("Recommended next actions:")
    lines.extend(f"{idx}. {text}" for idx, text in enumerate(recommendations, start=1))

  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Run the tici object-hazard live monitor over SSH.")
  parser.add_argument("--ssh-profile", default="commaHome")
  parser.add_argument("--remote-repo", default="/data/openpilot")
  parser.add_argument("--duration", type=float, default=45.0)
  parser.add_argument("--interval", type=float, default=1.0)
  parser.add_argument("--top-n", type=int, default=8)
  parser.add_argument("--json", action="store_true", help="Print the raw JSON report.")
  parser.add_argument("--save-json", type=Path, help="Write the raw JSON report to this path.")
  args = parser.parse_args()

  try:
    report = run_remote_probe(args.ssh_profile, args.remote_repo, args.duration, args.interval, args.top_n)
  except Exception as err:
    print(str(err), file=sys.stderr)
    return 1

  if args.save_json is not None:
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

  if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
  else:
    print(render_text_report(report))

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
