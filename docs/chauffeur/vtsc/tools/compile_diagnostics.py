#!/usr/bin/env python3
# stdlib only
import os, json, glob, argparse
from typing import Dict, Any, List, Tuple

EXPECTED_TOPICS = [
  "liveCalibration","driverMonitoringState","longitudinalPlan",
  "liveDelay","liveParameters","radarState",
  "liveTorqueParameters","driverAssistance","mapTurnSpeedControlSP"
]

def load_json(path: str) -> Any:
  try:
    with open(path, "r") as f: return json.load(f)
  except FileNotFoundError:
    return None

def latest_vtsc_dir(base: str) -> str:
  cands = sorted([p for p in glob.glob(os.path.join(base, "vtsc_good_data_*")) if os.path.isdir(p)])
  return cands[-1] if cands else base

def summarize_rlog_rates(rlog_rates: Dict[str, Any]) -> Tuple[List[Tuple[str, float]], int]:
  """Return offenders sorted by worst p95 gap (desc), and count of topics with p95>=1s."""
  worst: Dict[str, float] = {}
  for rlog, payload in (rlog_rates or {}).items():
    topics = (payload or {}).get("topics", {})
    for t, stats in topics.items():
      p95 = stats.get("p95_gap_s")
      if p95 is None: continue
      if t not in worst or (p95 > worst[t]):
        worst[t] = p95
  offenders = sorted(worst.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1), reverse=True)
  num_bad = sum(1 for _, v in offenders if v is not None and v >= 1.0)
  return offenders, num_bad

def is_rt(cls: str, rtprio: str) -> bool:
  c = (cls or "").upper()
  rp = (rtprio or "").strip()
  if "FF" in c or "FIFO" in c or c == "RT": return True
  if rp and rp not in ("0", "-", "None", ""): return True
  return False

def verdict_from(runtime: Dict[str, Any], services: Dict[str, Any],
                 live_probe: Dict[str, Any], rlog_rates: Dict[str, Any],
                 load_hint: Dict[str, Any]) -> Tuple[str, str]:
  mtscd = (runtime or {}).get("mtscd", {}) or {}
  cls = str(mtscd.get("cls") or "")
  rtprio = str(mtscd.get("rtprio") or "")
  cpu_pct = float(mtscd.get("cpu_pct") or 0.0)
  rt_throttling = bool((runtime or {}).get("rt_throttling") or False)
  stale = list((live_probe or {}).get("stale_after_15s") or [])
  dupes = list((services or {}).get("duplicate_ports") or [])
  load_m = float((load_hint or {}).get("mtscd_cpu_pct") or cpu_pct)
  load_l = float((load_hint or {}).get("loggerd_cpu_pct") or 0.0)
  loggerd_hot = bool((load_hint or {}).get("loggerd_cpu_hot") or (load_l >= 25.0))
  mtscd_hot = bool((load_hint or {}).get("mtscd_cpu_hot") or (load_m >= 25.0))

  offenders, num_bad = summarize_rlog_rates(rlog_rates)

  # Heuristics
  starv = (is_rt(cls, rtprio) or rt_throttling or mtscd_hot) and (num_bad >= 3 or len(stale) >= 3)
  svc_issue = bool(dupes) or ("mapTurnSpeedControlSP" not in EXPECTED_TOPICS)  # placeholder; we rely on dupes
  # "logging pressure" = heavy loggerd and high mapTurnSpeedControlSP rate causing gaps elsewhere
  map_rates = []
  for _, payload in (rlog_rates or {}).items():
    t = (payload or {}).get("topics", {}).get("mapTurnSpeedControlSP")
    if t: map_rates.append(float(t.get("hz_med") or 0.0))
  high_map_rate = (max(map_rates) if map_rates else 0.0) >= 10.0
  log_pressure = loggerd_hot and high_map_rate and num_bad >= 2

  notes: List[str] = []
  if starv:
    worst5 = [f"{t}:{g:.3f}s" for t,g in offenders[:5] if g is not None]
    throttles = (runtime or {}).get("rt_throttling_lines") or []
    notes.append(f"Starvation indicators: cls={cls or 'unknown'}, rtprio={rtprio or 'unknown'}, mtscd_cpu={load_m:.1f}%")
    if rt_throttling: notes.append("Kernel reported RT throttling (see runtime.rt_throttling_lines).")
    if worst5: notes.append("Worst p95 gap offenders: " + ", ".join(worst5))
    if stale: notes.append("Live probe stale topics: " + ", ".join(stale))
    return "starvation", "\n".join(f"- {n}" for n in notes)

  if svc_issue:
    if dupes: notes.append(f"Duplicate service ports: {dupes}")
    return "service_schema", "\n".join(f"- {n}" for n in (notes or ["Potential service/schema configuration issue."]))

  if log_pressure:
    notes.append(f"loggerd hot ({load_l:.1f}%), mapTurnSpeedControlSP median Hz ≈ {max(map_rates):.1f}")
    notes.append(f"Worst p95 gap offenders count (>=1s): {num_bad}")
    return "logging_pressure", "\n".join(f"- {n}" for n in notes)

  return "inconclusive", "- Evidence insufficient for a clear verdict. Provide runtime/services/live_probe/load_hint if missing."

def compile_diagnostics(data_dir: str) -> Dict[str, Any]:
  rlog_rates = load_json(os.path.join(data_dir, "rlog_rates.json")) or {}
  runtime = load_json(os.path.join(data_dir, "runtime.json")) or {"mtscd": {"cls": None, "rtprio": None, "cpu_pct": None, "affinity": None}, "rt_throttling": False}
  services = load_json(os.path.join(data_dir, "services.json")) or {"mapTurnSpeedControlSP": None, "duplicate_ports": []}
  live_probe = load_json(os.path.join(data_dir, "live_probe.json")) or {"stale_after_15s": [], "last_age_s": {}}
  load_hint = load_json(os.path.join(data_dir, "load_hint.json")) or {"loggerd_cpu_hot": False, "mtscd_cpu_hot": False}

  verdict, notes = verdict_from(runtime, services, live_probe, rlog_rates, load_hint)

  final = {
    "runtime": runtime,
    "services": services,
    "live_probe": live_probe,
    "load_hint": load_hint,
    "verdict": verdict,
    "notes": notes
  }
  return final

def main():
  ap = argparse.ArgumentParser(description="Compile MTSC comm diagnostics into a single verdict JSON.")
  ap.add_argument("--data-dir", default=None, help="Path to vtsc_good_data_*/")
  ap.add_argument("--write", action="store_true", help="Write diagnostics.json to the data dir")
  args = ap.parse_args()

  base = os.path.join(os.getcwd(), "docs", "chauffeur", "vtsc")
  data_dir = args.data_dir or latest_vtsc_dir(base)
  if not os.path.isdir(data_dir):
    raise SystemExit(f"Data dir not found: {data_dir}")

  final = compile_diagnostics(data_dir)

  if args.write:
    out = os.path.join(data_dir, "diagnostics.json")
    with open(out, "w") as f: json.dump(final, f, indent=2)
    print(out)
  else:
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
  main()

