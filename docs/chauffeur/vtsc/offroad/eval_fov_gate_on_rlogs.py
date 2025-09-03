#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-road evaluation of VTSC FOV gating using existing rlogs.
- Parses VTSCDBG from swaglog ("VTSCDBG {...}").
- Recomputes occlusion_gate per frame (with hysteresis) to predict occlusion_after.
- Summarizes freeway occlusion %, crawl %, and hidden-turn recall (proxy based on psi).
Outputs metrics.json and by_log.jsonl under --outdir/<timestamp>/.
"""

from __future__ import annotations

import os, re, sys, json, glob, math, argparse, datetime, statistics
from typing import Dict, Any, Iterable, List, Tuple

# Try both import paths for LogReader (repo-local vs module path)
LogReader = None
try:
  from openpilot.tools.lib.logreader import LogReader as _LR  # type: ignore
  LogReader = _LR
except Exception:
  try:
    from tools.lib.logreader import LogReader as _LR2  # type: ignore
    LogReader = _LR2
  except Exception as e:
    print("ERROR: cannot import tools.lib.logreader; run inside openpilot repo.", file=sys.stderr)
    raise

# Try import the VTSC occlusion_gate; otherwise use fallback copy
HAVE_VTSC_GATE = False
try:
  from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController  # type: ignore
  HAVE_VTSC_GATE = hasattr(VisionTurnController, 'occlusion_gate')
except Exception:
  HAVE_VTSC_GATE = False


def occlusion_gate_fallback(kappa_vis: float, s_visible_m: float, path_conf: float,
                            psi_fov_rad: float, psi_margin_rad: float,
                            k_freeway: float = 1e-5, k_min: float = 2e-4,
                            s_long: float = 120.0,
                            state: Dict[str, Any] | None = None,
                            n_on: int = 5, n_off: int = 10) -> Tuple[bool, Dict[str, Any], str, Dict[str, float]]:
  if state is None:
    state = {"on_cnt": 0, "off_cnt": 0, "occluded": False}
  occluded = bool(state.get("occluded", False))
  on_cnt = int(state.get("on_cnt", 0))
  off_cnt = int(state.get("off_cnt", 0))

  ka = abs(kappa_vis or 0.0)
  s = float(s_visible_m or 0.0)
  conf = float(path_conf or 0.0)
  psi_vis = ka * s
  psi_thresh = max(0.0, psi_fov_rad - psi_margin_rad)

  onset = (ka >= k_min) and (psi_vis >= psi_thresh)
  clear = (ka < k_freeway) or ((s >= s_long) and (psi_vis < psi_thresh) and (conf >= 0.6))

  reason = "none"
  if onset and not occluded:
    on_cnt += 1
    off_cnt = 0
    if on_cnt >= n_on:
      occluded = True
      reason = "fov_exit"
  elif clear and occluded:
    off_cnt += 1
    on_cnt = 0
    if off_cnt >= n_off:
      occluded = False
      reason = "freeway"
  else:
    on_cnt = max(0, on_cnt - 1) if not onset else on_cnt
    off_cnt = max(0, off_cnt - 1) if not clear else off_cnt

  state.update({"occluded": occluded, "on_cnt": on_cnt, "off_cnt": off_cnt})
  dbg = {"psi_vis": psi_vis, "psi_thresh": psi_thresh, "kappa_abs": ka, "s_visible_m": s}
  return occluded, state, (reason if reason != "none" else ("fov_exit" if onset else "freeway" if clear else "none")), dbg


def gated(ka, s, conf, psi_fov_rad, psi_margin_rad, state, n_on, n_off):
  if HAVE_VTSC_GATE:
    try:
      return VisionTurnController.occlusion_gate(ka, s, conf, psi_fov_rad, psi_margin_rad,
                                                 state=state,
                                                 k_freeway=1e-5, k_min=2e-4, s_long=120.0)
    except TypeError:
      pass
    except Exception:
      pass
  return occlusion_gate_fallback(ka, s, conf, psi_fov_rad, psi_margin_rad, state=state, n_on=n_on, n_off=n_off)


VTSCDBG_RE = re.compile(r'VTSCDBG[:\s]*(\{.*\})')


def parse_vtscdbg_from_logmessage(raw: Any) -> Dict[str, Any] | None:
  """raw is often a JSON string; try to load and read 'msg', else regex search."""
  if not isinstance(raw, str):
    try:
      s = str(raw)
    except Exception:
      return None
  else:
    s = raw
  try:
    outer = json.loads(s)
    msg = outer.get('msg', '')
    m = VTSCDBG_RE.search(msg)
    if m:
      return json.loads(m.group(1))
  except Exception:
    pass
  m = VTSCDBG_RE.search(s)
  if not m:
    return None
  try:
    return json.loads(m.group(1))
  except Exception:
    return None


def iter_frames(rlog_path: str) -> Iterable[Dict[str, Any]]:
  v_ego = None
  for ev in LogReader(rlog_path):
    w = ev.which()
    if w == "carState":
      try:
        v_ego = float(ev.carState.vEgo)
      except Exception:
        pass
    elif w == "logMessage":
      try:
        dbg = parse_vtscdbg_from_logmessage(getattr(ev, 'logMessage'))
        if dbg is None:
          continue
        t = float(getattr(ev, "logMonoTime", 0)) * 1e-9
        yield {"t": t, "v_ego": v_ego, "dbg": dbg}
      except Exception:
        continue


def classify_freeway(dbg: Dict[str, Any]) -> bool:
  kappa = abs(float(dbg.get("kappa_vis", 0.0)))
  svis = float(dbg.get("s_visible_m", 0.0))
  conf = float(dbg.get("path_conf", 0.0))
  return (kappa <= 1e-5) and (svis >= 120.0) and (conf >= 0.6)


def analyze_log(rlog_path: str, psi_fov_deg: float, psi_margin_deg: float,
                n_on: int, n_off: int) -> Dict[str, Any]:
  psi_fov = math.radians(psi_fov_deg)
  psi_margin = math.radians(psi_margin_deg)
  state = {"occluded": False, "on_cnt": 0, "off_cnt": 0}

  total = freeway_total = 0
  occl_orig_total = occl_after_total = 0
  freeway_occl_orig = freeway_occl_after = 0
  crawl_after = crawl_orig = 0
  crawl_sequences_after = 0
  hidden_pos_total = hidden_pos_recalled = 0

  last_t = None
  running_crawl = 0.0

  for fr in iter_frames(rlog_path):
    dbg = fr["dbg"]; t = fr["t"]; v_ego = float(fr.get("v_ego") or 0.0)
    total += 1

    kappa = float(dbg.get("kappa_vis", 0.0))
    svis = float(dbg.get("s_visible_m", 0.0))
    conf = float(dbg.get("path_conf", 0.0))
    cap_vis = dbg.get("cap_visible_vmin")
    cap_occ = dbg.get("cap_occl_vmin")
    cap_map = dbg.get("cap_map_vmin")
    vtsc_cmd_orig = dbg.get("vtsc_cmd")
    v_set = dbg.get("v_set") or dbg.get("cruise") or v_ego

    occl_orig = bool(dbg.get("occluded", False))
    occl_after, state, reason, fdbg = gated(kappa, svis, conf, psi_fov, psi_margin, state, n_on, n_off)
    occl_orig_total += int(occl_orig)
    occl_after_total += int(occl_after)

    freeway = classify_freeway(dbg)
    if freeway:
      freeway_total += 1
      freeway_occl_orig += int(occl_orig)
      freeway_occl_after += int(occl_after)

    def min_ignore_none(vals):
      vals = [x for x in vals if x is not None]
      return min(vals) if vals else None
    cap_list_after = [cap_vis]
    if occl_after and (cap_occ is not None):
      cap_list_after.append(cap_occ)
    if cap_map is not None:
      cap_list_after.append(cap_map)
    vtsc_cmd_after = min_ignore_none(cap_list_after)

    def is_crawl(vcmd, vset):
      if vcmd is None or vset is None:
        return False
      vcmd = float(vcmd); vset = float(vset)
      return (vcmd < 0.85 * vset) or (vcmd < 20.0)

    if is_crawl(vtsc_cmd_after, v_set):
      crawl_after += 1
      if last_t is not None:
        running_crawl += (t - last_t)
    else:
      if running_crawl >= 2.0:
        crawl_sequences_after += 1
      running_crawl = 0.0

    if is_crawl(vtsc_cmd_orig, v_set):
      crawl_orig += 1

    psi_vis = fdbg.get("psi_vis", abs(kappa) * svis)
    psi_thresh = fdbg.get("psi_thresh", max(0.0, psi_fov - psi_margin))
    is_positive = (psi_vis >= psi_thresh) and (svis > 0.0)
    if is_positive:
      hidden_pos_total += 1
      hidden_pos_recalled += int(occl_after)

    last_t = t

  if running_crawl >= 2.0:
    crawl_sequences_after += 1

  def pct(n, d):
    return (100.0 * n / d) if d else 0.0

  return {
    "rlog": rlog_path,
    "frames_total": total,
    "freeway_frames": freeway_total,
    "occluded_orig_pct": round(pct(occl_orig_total, total), 3),
    "occluded_after_pct": round(pct(occl_after_total, total), 3),
    "freeway_occluded_orig_pct": round(pct(freeway_occl_orig, freeway_total), 3),
    "freeway_occluded_after_pct": round(pct(freeway_occl_after, freeway_total), 3),
    "crawl_orig_pct": round(pct(crawl_orig, total), 3),
    "crawl_after_pct": round(pct(crawl_after, total), 3),
    "crawl_sequences_after_ge2s": int(crawl_sequences_after),
    "hidden_pos_total": int(hidden_pos_total),
    "hidden_recall_after_pct": round(pct(hidden_pos_recalled, hidden_pos_total), 3),
  }


def load_manifest(path: str) -> List[str]:
  try:
    with open(path, "r") as f:
      m = json.load(f)
    for key in ("rlogs", "files"):
      if key in m and isinstance(m[key], list):
        return [p for p in m[key] if isinstance(p, str) and (p.endswith('rlog.zst') or p.endswith('rlog.bz2') or 'rlog' in p)]
  except Exception:
    pass
  if os.path.isdir(path):
    return sorted(glob.glob(os.path.join(path, "**", "rlog.*"), recursive=True))
  return []


def main():
  ap = argparse.ArgumentParser(description="Off-road VTSC FOV gate evaluation on existing rlogs.")
  ap.add_argument("--manifest", nargs="*", default=[], help="Path(s) to MANIFEST.json or folders with rlogs")
  ap.add_argument("--glob", nargs="*", default=[], help='Glob(s), e.g. /data/media/0/realdata/*/*/rlog.*')
  ap.add_argument("--psi-fov-deg", type=float, default=28.0)
  ap.add_argument("--psi-margin-deg", type=float, default=5.0)
  ap.add_argument("--n-on", type=int, default=5)
  ap.add_argument("--n-off", type=int, default=10)
  ap.add_argument("--outdir", default="docs/chauffeur/vtsc/offroad/reports")
  args = ap.parse_args()

  rlogs: List[str] = []
  for m in args.manifest:
    rlogs.extend(load_manifest(m))
  for g in args.glob:
    rlogs.extend(glob.glob(g))
  rlogs = sorted(set(rlogs))
  if not rlogs:
    print("No rlogs found. Provide --manifest or --glob.", file=sys.stderr)
    sys.exit(2)

  ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  outdir = os.path.join(args.outdir, f"vtsc_offroad_{ts}")
  os.makedirs(outdir, exist_ok=True)

  per_log: List[Dict[str, Any]] = []
  for r in rlogs:
    try:
      s = analyze_log(r, args.psi_fov_deg, args.psi_margin_deg, args.n_on, args.n_off)
      print(json.dumps({k: s[k] for k in ("rlog","freeway_occluded_after_pct","crawl_after_pct","hidden_recall_after_pct")}))
      per_log.append(s)
    except KeyboardInterrupt:
      raise
    except Exception as e:
      print(f"ERROR analyzing {r}: {e}", file=sys.stderr)

  hp = [x["hidden_recall_after_pct"] for x in per_log if x["hidden_pos_total"]>0]
  agg = {
    "count_logs": len(per_log),
    "psi_fov_deg": args.psi_fov_deg,
    "psi_margin_deg": args.psi_margin_deg,
    "n_on": args.n_on, "n_off": args.n_off,
    "freeway_occluded_after_pct_med": (statistics.median([x["freeway_occluded_after_pct"] for x in per_log]) if per_log else None),
    "crawl_after_pct_med": (statistics.median([x["crawl_after_pct"] for x in per_log]) if per_log else None),
    "hidden_recall_after_pct_med": (statistics.median(hp) if hp else None),
    "logs_worse_than_spec": {
      "freeway_occluded_after_pct>2": [x["rlog"] for x in per_log if x["freeway_occluded_after_pct"] > 2.0],
      "crawl_after_pct>1": [x["rlog"] for x in per_log if x["crawl_after_pct"] > 1.0],
      "hidden_recall_after_pct<90": [x["rlog"] for x in per_log if x["hidden_pos_total"]>0 and x["hidden_recall_after_pct"] < 90.0],
    }
  }

  with open(os.path.join(outdir, "by_log.jsonl"), "w") as f:
    for s in per_log:
      f.write(json.dumps(s) + "\n")
  with open(os.path.join(outdir, "metrics.json"), "w") as f:
    json.dump({"aggregate": agg, "logs": per_log}, f, indent=2)

  print("\n=== VTSC OFF-ROAD SUMMARY ===")
  print(json.dumps(agg, indent=2))
  print(f"\nArtifacts written to: {outdir}")


if __name__ == "__main__":
  main()
