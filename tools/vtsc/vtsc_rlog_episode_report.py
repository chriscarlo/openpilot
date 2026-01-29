#!/usr/bin/env python3
from __future__ import annotations

"""
VTSC rlog "cap episode" reporter (before/after replay).

Goal:
- Quantify how VTSC behaves over a set of rlogs by detecting "cap episodes" and computing:
  - cap episode duration (s)
  - time-to-first-decel (s) within each episode

This is intended for offline tuning and regression checks when iterating on VTSC logic.

Typical usage (local cache):
  python3 tools/vtsc/vtsc_rlog_episode_report.py .cache/commaCar/00000214--01d7104b54

Compare a code-path toggle / constant (e.g. severe overshoot scaling):
  python3 tools/vtsc/vtsc_rlog_episode_report.py .cache/... --before-scale-min 1.0 --after-scale-min 0.90

Notes:
- This tool replays the VisionTurnController against the recorded modelV2 + carState signals.
- It uses a deterministic Params patch to avoid depending on the host's Params database.
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

# Allow running without `pip install -e .` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.tools.lib.logreader import LogReader
import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController


@dataclasses.dataclass(frozen=True)
class Sample:
  t: float
  v_ego: float
  a_ego: float
  v_turn: float
  a_target: float


@dataclasses.dataclass(frozen=True)
class Episode:
  idx: int
  start_t: float
  end_t: float
  duration_s: float
  ttfdecel_s: float | None
  min_v_turn: float
  max_v_ego: float


def _discover_rlogs(inputs: List[str]) -> List[Path]:
  paths = [Path(p).expanduser() for p in inputs]
  out: List[Path] = []
  for p in paths:
    if p.is_file():
      out.append(p)
      continue
    if not p.is_dir():
      continue
    # Support both layouts:
    # - route_dir/<seg>/rlog.zst
    # - route_dir/<seg>.rlog.zst
    patterns = [
      "**/*.rlog.zst",
      "**/*.rlog.bz2",
      "**/rlog.zst",
      "**/rlog.bz2",
      "**/rlog_*.zst",
      "**/rlog_*.bz2",
    ]
    for pat in patterns:
      out.extend(sorted(p.glob(pat)))
  # Dedup + stable sort
  uniq = sorted({pp.resolve() for pp in out})
  return uniq


def _extract_route_and_segment(path: Path) -> Tuple[str, str]:
  # Try to infer from common layouts.
  name = path.name
  seg = ""
  if name.endswith(".rlog.zst") or name.endswith(".rlog.bz2"):
    seg = name.split(".")[0]
  # If parent dir is numeric, that's the segment.
  if not seg and path.parent.name.isdigit():
    seg = path.parent.name
  # Route: nearest parent with a '--' in name (e.g. 00000214--01d7104b54)
  route = ""
  for parent in [path.parent] + list(path.parents):
    if "--" in parent.name:
      route = parent.name
      break
  if not route:
    route = path.parent.name
  if not seg:
    seg = "?"
  return route, seg


def _mk_controller_deterministic() -> VisionTurnController:
  # Minimal car params (enough to build VehicleModel in dev tests).
  class MockCP:
    mass = 1600.0
    rotationalInertia = 2500.0
    wheelbase = 2.75
    centerToFront = 1.20
    steerRatio = 15.0
    steerRatioRear = 0.0
    tireStiffnessFront = 80000.0
    tireStiffnessRear = 80000.0

  with patch("sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params") as MockParams:
    mp = MagicMock()

    def _get_bool(key: str) -> bool:
      # Keep core VTSC enabled; everything else deterministic/off unless the controller hard-requires it.
      if key in ("VisionTurnSpeedControl", "VisionTurnSpeedControlOcclBypassWithLead"):
        return True
      return False

    mp.get_bool.side_effect = _get_bool

    def _get(key: str):
      # Defaults chosen to match test harness expectations.
      if key.endswith("Aggressiveness"):
        return b"1.0"
      if key.endswith("FilterAlpha"):
        return b"0.3"
      if key.endswith("HysteresisThreshold"):
        return b"0.2"
      if key.endswith("SafetyBias"):
        return b"0.1"
      return None

    mp.get.side_effect = _get
    MockParams.return_value = mp
    return VisionTurnController(MockCP())


def _estimate_v_cruise_from_vtscdbg(rlog_path: Path) -> float | None:
  # If VTSCDBG is enabled, it includes the cruise setpoint in m/s as 'cruise'.
  import json
  for m in LogReader(str(rlog_path)):
    if m.which() != "logMessage":
      continue
    try:
      outer = json.loads(m.logMessage)
    except Exception:
      continue
    msg = outer.get("msg")
    if not (isinstance(msg, str) and msg.startswith("VTSCDBG ")):
      continue
    try:
      d = json.loads(msg.split("VTSCDBG ", 1)[1])
    except Exception:
      continue
    try:
      v = float(d.get("cruise"))
    except Exception:
      v = None
    if v is not None and math.isfinite(v) and v > 0.1:
      return v
  return None


def _replay_samples(
  rlog_path: Path,
  *,
  severe_scale_min: float,
  v_cruise_mps: float | None,
) -> List[Sample]:
  # Temporarily override the constant for this run.
  prev = float(getattr(vtc, "SEVERE_OVERSHOOT_SPEED_SCALE_MIN", 1.0))
  setattr(vtc, "SEVERE_OVERSHOOT_SPEED_SCALE_MIN", float(severe_scale_min))
  try:
    ctrl = _mk_controller_deterministic()

    class SM:
      def __init__(self):
        self.valid: Dict[str, bool] = {"modelV2": True, "carState": True}
        self._data: Dict[str, Any] = {}

      def __getitem__(self, key):
        return self._data.get(key)

    sm = SM()

    # State updated by messages; used when we see modelV2.
    v_ego = 0.0
    a_ego = 0.0
    gas_pressed = False
    steer_deg = 0.0
    last_radar_state = None
    have_radar = False

    t0_ns: int | None = None
    v_cruise_eff = float(v_cruise_mps) if v_cruise_mps is not None else None
    if v_cruise_eff is None:
      v_cruise_eff = _estimate_v_cruise_from_vtscdbg(rlog_path)
    if v_cruise_eff is None:
      # Safe-ish default for analysis; caller can override via CLI.
      v_cruise_eff = 33.0

    out: List[Sample] = []
    for m in LogReader(str(rlog_path)):
      which = m.which()
      if which == "carState":
        cs = m.carState
        try:
          v_ego = float(cs.vEgo)
        except Exception:
          pass
        try:
          a_ego = float(cs.aEgo)
        except Exception:
          pass
        try:
          gas_pressed = bool(cs.gasPressed)
        except Exception:
          pass
        try:
          steer_deg = float(cs.steeringAngleDeg)
        except Exception:
          pass
        continue

      if which == "radarState":
        last_radar_state = m.radarState
        have_radar = True
        continue

      if which != "modelV2":
        continue

      if t0_ns is None:
        t0_ns = int(m.logMonoTime)
      t = (int(m.logMonoTime) - int(t0_ns)) * 1e-9

      sm._data["modelV2"] = m.modelV2
      sm._data["carState"] = SimpleNamespace(gasPressed=gas_pressed, steeringAngleDeg=steer_deg)
      if have_radar and last_radar_state is not None:
        sm._data["radarState"] = last_radar_state
        sm.valid["radarState"] = True
      else:
        sm._data.pop("radarState", None)
        sm.valid.pop("radarState", None)

      # Make controller time deterministic based on rlog monotime.
      with patch("sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: t), \
           patch("sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: t):
        ctrl.update(sm, True, float(v_ego), float(a_ego), float(v_cruise_eff))

      try:
        v_turn = float(ctrl.v_turn)
      except Exception:
        v_turn = 0.0
      try:
        a_target = float(ctrl.a_target)
      except Exception:
        a_target = 0.0

      out.append(Sample(t=float(t), v_ego=float(v_ego), a_ego=float(a_ego), v_turn=float(v_turn), a_target=float(a_target)))

    return out
  finally:
    setattr(vtc, "SEVERE_OVERSHOOT_SPEED_SCALE_MIN", prev)


def _detect_episodes(
  samples: List[Sample],
  *,
  vdiff_mps: float,
  decel_a_threshold: float,
  gap_s: float,
) -> List[Episode]:
  if len(samples) < 2:
    return []

  # Precompute dt per sample for duration estimation.
  ts = [s.t for s in samples]
  dts = []
  for i in range(len(ts) - 1):
    dts.append(max(0.0, float(ts[i + 1] - ts[i])))
  dt_med = sorted(dts)[len(dts) // 2] if dts else 0.05
  dt_tail = max(0.01, min(0.15, float(dt_med) if math.isfinite(dt_med) else 0.05))

  active = [bool(s.v_turn <= (s.v_ego - vdiff_mps)) for s in samples]
  episodes: List[Episode] = []
  ep_start_idx: int | None = None
  last_active_t: float | None = None

  def close_episode(ep_end_idx: int) -> None:
    nonlocal ep_start_idx
    if ep_start_idx is None:
      return
    start = samples[ep_start_idx]
    end = samples[ep_end_idx]
    dur = 0.0
    min_v_turn = float("inf")
    max_v_ego = 0.0
    ttf = None

    for i in range(ep_start_idx, ep_end_idx + 1):
      s = samples[i]
      min_v_turn = min(min_v_turn, float(s.v_turn))
      max_v_ego = max(max_v_ego, float(s.v_ego))
      if ttf is None and float(s.a_target) <= -abs(float(decel_a_threshold)):
        ttf = float(s.t - start.t)
      # duration accumulate where active; last sample uses dt_tail
      if active[i]:
        dt_i = dt_tail if i >= (len(samples) - 1) else (samples[i + 1].t - samples[i].t if i < len(samples) - 1 else dt_tail)
        dur += max(0.0, float(dt_i))

    episodes.append(Episode(
      idx=len(episodes),
      start_t=float(start.t),
      end_t=float(end.t),
      duration_s=float(dur),
      ttfdecel_s=None if ttf is None else float(ttf),
      min_v_turn=float(min_v_turn if math.isfinite(min_v_turn) else 0.0),
      max_v_ego=float(max_v_ego),
    ))
    ep_start_idx = None

  for i, (s, is_active) in enumerate(zip(samples, active, strict=False)):
    if is_active:
      if ep_start_idx is None:
        ep_start_idx = i
      last_active_t = float(s.t)
      continue
    # Not active: close if we were active and the gap exceeded.
    if ep_start_idx is not None and last_active_t is not None:
      if (float(s.t) - float(last_active_t)) >= float(gap_s):
        close_episode(i - 1)
        last_active_t = None

  # close trailing
  if ep_start_idx is not None:
    close_episode(len(samples) - 1)

  return episodes


def _fmt(x: float | None) -> str:
  if x is None:
    return ""
  if not math.isfinite(float(x)):
    return ""
  return f"{float(x):.3f}"


def main() -> int:
  ap = argparse.ArgumentParser(description="VTSC rlog cap-episode reporter (before/after replay).")
  ap.add_argument("inputs", nargs="+", help="rlog files or directories (will be searched recursively)")
  ap.add_argument("--before-scale-min", type=float, default=1.0, help="SEVERE_OVERSHOOT_SPEED_SCALE_MIN for baseline")
  ap.add_argument("--after-scale-min", type=float, default=0.90, help="SEVERE_OVERSHOOT_SPEED_SCALE_MIN for tuned run")
  ap.add_argument("--v-cruise-mps", type=float, default=None, help="Override cruise setpoint used during replay (m/s)")
  ap.add_argument("--vdiff-mps", type=float, default=0.25, help="Episode active if v_turn <= v_ego - vdiff")
  ap.add_argument("--decel-a-threshold", type=float, default=0.10, help="First decel when a_target <= -threshold (m/s^2)")
  ap.add_argument("--gap-s", type=float, default=0.20, help="Gap tolerance between active samples (s)")
  ap.add_argument("--summary", action="store_true", help="Print per-file summary instead of per-episode rows")
  ap.add_argument("--out", type=str, default=None, help="Write TSV to this file (also prints to stdout)")
  args = ap.parse_args()

  rlogs = _discover_rlogs(list(args.inputs))
  if not rlogs:
    raise SystemExit("No rlog files found under provided inputs.")

  lines: List[str] = []
  if args.summary:
    header = [
      "route", "segment", "file",
      "n_ep_before", "n_ep_after",
      "total_cap_s_before", "total_cap_s_after",
      "median_ep_s_before", "median_ep_s_after",
      "median_ttfdecel_s_before", "median_ttfdecel_s_after",
      "first_ep_start_s_before", "first_ep_start_s_after",
    ]
  else:
    header = [
      "route", "segment", "file", "ep_idx",
      "cap_s_before", "ttfdecel_s_before", "start_s_before",
      "cap_s_after", "ttfdecel_s_after", "start_s_after",
      "delta_cap_s", "delta_start_s",
    ]
  lines.append("\t".join(header))

  for rlog in rlogs:
    route, seg = _extract_route_and_segment(rlog)

    before_samples = _replay_samples(
      rlog,
      severe_scale_min=float(args.before_scale_min),
      v_cruise_mps=args.v_cruise_mps,
    )
    after_samples = _replay_samples(
      rlog,
      severe_scale_min=float(args.after_scale_min),
      v_cruise_mps=args.v_cruise_mps,
    )
    before_eps = _detect_episodes(
      before_samples,
      vdiff_mps=float(args.vdiff_mps),
      decel_a_threshold=float(args.decel_a_threshold),
      gap_s=float(args.gap_s),
    )
    after_eps = _detect_episodes(
      after_samples,
      vdiff_mps=float(args.vdiff_mps),
      decel_a_threshold=float(args.decel_a_threshold),
      gap_s=float(args.gap_s),
    )

    if args.summary:
      def _median(vals: List[float]) -> float | None:
        if not vals:
          return None
        vv = sorted(vals)
        return float(vv[len(vv) // 2])
      def _sum(vals: List[float]) -> float:
        return float(sum(vals))

      b_durs = [e.duration_s for e in before_eps]
      a_durs = [e.duration_s for e in after_eps]
      b_ttf = [e.ttfdecel_s for e in before_eps if e.ttfdecel_s is not None]
      a_ttf = [e.ttfdecel_s for e in after_eps if e.ttfdecel_s is not None]
      b_first = before_eps[0].start_t if before_eps else None
      a_first = after_eps[0].start_t if after_eps else None

      row = [
        route, seg, str(rlog),
        str(len(before_eps)), str(len(after_eps)),
        _fmt(_sum(b_durs)), _fmt(_sum(a_durs)),
        _fmt(_median(b_durs)), _fmt(_median(a_durs)),
        _fmt(_median(b_ttf)), _fmt(_median(a_ttf)),
        _fmt(b_first), _fmt(a_first),
      ]
      lines.append("\t".join(row))
      continue

    # Per-episode rows: align by index (good enough when comparing same file)
    n = max(len(before_eps), len(after_eps))
    for i in range(n):
      b = before_eps[i] if i < len(before_eps) else None
      a = after_eps[i] if i < len(after_eps) else None
      b_cap = b.duration_s if b else None
      a_cap = a.duration_s if a else None
      b_start = b.start_t if b else None
      a_start = a.start_t if a else None
      delta_cap = (a_cap - b_cap) if (a_cap is not None and b_cap is not None) else None
      delta_start = (a_start - b_start) if (a_start is not None and b_start is not None) else None
      row = [
        route,
        seg,
        str(rlog),
        str(i),
        _fmt(b_cap),
        _fmt(b.ttfdecel_s) if b else "",
        _fmt(b_start),
        _fmt(a_cap),
        _fmt(a.ttfdecel_s) if a else "",
        _fmt(a_start),
        _fmt(delta_cap),
        _fmt(delta_start),
      ]
      lines.append("\t".join(row))

  tsv = "\n".join(lines) + "\n"
  print(tsv, end="")
  if args.out:
    Path(args.out).expanduser().write_text(tsv, encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
