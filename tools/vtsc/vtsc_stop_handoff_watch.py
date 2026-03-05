#!/usr/bin/env python3
"""
VTSC stop/handoff watcher for on-road debugging.

Focus:
- Is VTSC blocking acceleration from stop/low-speed launch?
- Is lead-follow logic the likely limiter instead?
- Is there a launch handoff oscillation (positive->negative->positive aTarget)?

This subscribes directly to live messaging and prints compact status lines.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path
import sys

try:
  from cereal import messaging
except ModuleNotFoundError:
  # Support running from /tmp on-device or other non-repo working dirs.
  sys.path.append("/data/openpilot")
  try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
  except Exception:
    pass
  from cereal import messaging


def _sign_with_deadband(x: float, deadband: float = 0.10) -> int:
  if x > deadband:
    return 1
  if x < -deadband:
    return -1
  return 0


def _fmt(x: float | int | None, n: int = 2) -> str:
  if x is None:
    return "?"
  try:
    xf = float(x)
    if not math.isfinite(xf):
      return "?"
    return f"{xf:.{n}f}"
  except Exception:
    return "?"


def _bool(v: bool) -> str:
  return "Y" if bool(v) else "N"


def _lead_dup(lead1, lead2) -> bool:
  try:
    if not (bool(lead1.status) and bool(lead2.status)):
      return False
    d_close = abs(float(lead1.dRel) - float(lead2.dRel)) < 2.0
    p_close = abs(float(lead1.modelProb) - float(lead2.modelProb)) < 0.10
    same_sensor = bool(lead1.radar) == bool(lead2.radar)
    same_track = int(lead1.radarTrackId) == int(lead2.radarTrackId)
    return d_close and p_close and same_sensor and same_track
  except Exception:
    return False


def _lcs_name(v: int) -> str:
  return {
    0: "off",
    1: "pid",
    2: "stopping",
    3: "starting",
  }.get(int(v), f"unk{v}")


def _enum_name(v) -> str:
  """Best-effort name extraction for capnp DynamicEnum / python enums / scalars."""
  try:
    s = str(v)
    if s:
      return s
  except Exception:
    pass
  return "?"


def _lcs_code(v) -> int:
  try:
    return int(v)
  except Exception:
    pass
  name = _enum_name(v).strip().lower()
  return {
    "off": 0,
    "pid": 1,
    "stopping": 2,
    "starting": 3,
  }.get(name, -1)


def main() -> int:
  ap = argparse.ArgumentParser(description="Watch VTSC stop/launch handoff behavior live.")
  ap.add_argument("--hz", type=float, default=10.0, help="Render/update rate in Hz (default: 10)")
  ap.add_argument("--duration", type=float, default=0.0, help="Run for N seconds; 0 = run until Ctrl-C")
  ap.add_argument("--launch-window-s", type=float, default=8.0, help="Seconds after standstill release to watch handoff")
  ap.add_argument("--only-alerts", action="store_true", help="Only print suspicious lines/alerts")
  ap.add_argument("--max-speed", type=float, default=12.0, help="When not alert-only, print rows only below this speed (m/s)")
  args = ap.parse_args()

  period = 1.0 / max(1.0, args.hz)
  sm = messaging.SubMaster(
    ["carState", "radarState", "longitudinalPlan", "longitudinalPlanSP", "modelV2", "selfdriveState", "controlsState"],
    poll="longitudinalPlanSP",
  )

  prev_standstill = True
  launch_active = False
  launch_t0 = 0.0
  launch_signs: deque[tuple[float, int]] = deque(maxlen=32)
  launch_a_target_samples: deque[float] = deque(maxlen=256)
  launch_block_vtsc = False
  launch_block_lead = False
  launch_handoff = False
  launch_any_enabled = False
  launch_lcs_path: list[int] = []
  last_print_t = 0.0
  t_start = time.monotonic()

  print("watching stop/handoff: Ctrl-C to stop")
  print("t v aT aE vtsc lead[d,hw] dup mdlL block handoff notes")

  try:
    while True:
      now = time.monotonic()
      if args.duration > 0 and (now - t_start) >= args.duration:
        return 0

      sm.update(int(period * 1000))
      if not sm.updated["longitudinalPlanSP"]:
        continue

      cs = sm["carState"]
      rs = sm["radarState"]
      lp = sm["longitudinalPlan"]
      lpsp = sm["longitudinalPlanSP"]
      m = sm["modelV2"]
      sd = sm["selfdriveState"]
      ctrl = sm["controlsState"]

      try:
        v_ego = float(cs.vEgo)
      except Exception:
        v_ego = 0.0
      try:
        a_ego = float(cs.aEgo)
      except Exception:
        a_ego = 0.0
      try:
        a_target = float(lp.aTarget)
      except Exception:
        a_target = 0.0
      try:
        vtsc_v = float(lpsp.visionTurnSpeedControl.velocity)
      except Exception:
        vtsc_v = float("nan")
      try:
        vtsc_state = int(lpsp.visionTurnSpeedControl.state)
      except Exception:
        vtsc_state = -1
      try:
        long_state_raw = ctrl.longControlState
      except Exception:
        long_state_raw = "?"
      long_state = _lcs_code(long_state_raw)
      long_state_name = _lcs_name(long_state) if long_state >= 0 else _enum_name(long_state_raw)

      standstill = bool(getattr(cs, "standstill", False)) or (v_ego < 0.10)
      enabled = bool(getattr(sd, "enabled", False))
      active = bool(getattr(sd, "active", False))
      sd_state_name = _enum_name(getattr(sd, "state", "?"))
      sd_state_norm = sd_state_name.strip().lower()
      control_on = enabled or active or (sd_state_norm in ("enabled", "softdisabling", "overriding"))

      lead1 = rs.leadOne
      lead2 = rs.leadTwo
      lead_status = bool(getattr(lead1, "status", False))
      lead_d_rel = float(getattr(lead1, "dRel", 0.0) or 0.0)
      headway_s = lead_d_rel / max(v_ego, 0.30) if lead_status else 99.0
      lead_close = lead_status and ((v_ego > 2.0 and headway_s < 1.30) or (v_ego <= 2.0 and lead_d_rel < 6.0))
      lead_dup = _lead_dup(lead1, lead2)

      model_active_leads = 0
      try:
        for ld in m.leadsV3:
          prob = float(getattr(ld, "prob", 0.0) or 0.0)
          x0 = float(ld.x[0]) if len(ld.x) else float("nan")
          if prob >= 0.50 and math.isfinite(x0) and x0 > 0.0:
            model_active_leads += 1
      except Exception:
        pass

      # VTSC likely limiting if it publishes a cap close to current speed while below usual cruise.
      vtsc_cap_near_ego = math.isfinite(vtsc_v) and (vtsc_v <= v_ego + 0.50)
      planner_not_accel = a_target <= 0.05

      block_by_vtsc = control_on and (v_ego < 5.0) and planner_not_accel and vtsc_cap_near_ego
      block_by_lead = control_on and (v_ego < 5.0) and planner_not_accel and lead_close and not vtsc_cap_near_ego
      # Diagnostic versions that ignore enabled-state to help root-cause odd handoff behavior.
      block_by_vtsc_diag = (v_ego < 5.0) and planner_not_accel and vtsc_cap_near_ego
      block_by_lead_diag = (v_ego < 5.0) and planner_not_accel and lead_close and not vtsc_cap_near_ego

      # Stop->launch handoff tracking
      if prev_standstill and (not standstill):
        launch_active = True
        launch_t0 = now
        launch_signs.clear()
        launch_a_target_samples.clear()
        launch_block_vtsc = False
        launch_block_lead = False
        launch_handoff = False
        launch_any_enabled = False
        launch_lcs_path = [long_state]
        print(
          f"{time.strftime('%H:%M:%S')} "
          f"LAUNCH_START v={_fmt(v_ego,2)} lead={_bool(lead_status)} "
          f"d={_fmt(lead_d_rel,1)} hw={_fmt(headway_s,2)} dup={_bool(lead_dup)} mdlL={model_active_leads} lcs={long_state_name}"
        )

      if standstill:
        # Reset when fully stopped again.
        launch_active = False
        launch_signs.clear()
        launch_a_target_samples.clear()
        launch_any_enabled = False
        launch_lcs_path = []

      handoff_glitch = False
      if launch_active:
        dt_launch = now - launch_t0
        if dt_launch <= args.launch_window_s:
          launch_a_target_samples.append(a_target)
          if control_on:
            launch_any_enabled = True
          if (not launch_lcs_path) or (launch_lcs_path[-1] != long_state):
            launch_lcs_path.append(long_state)
          if block_by_vtsc_diag:
            launch_block_vtsc = True
          if block_by_lead_diag:
            launch_block_lead = True
          s = _sign_with_deadband(a_target, deadband=0.10)
          if s != 0:
            if not launch_signs or launch_signs[-1][1] != s:
              launch_signs.append((now, s))
          # Positive -> negative -> positive (or inverse) while still slow and no close lead:
          if len(launch_signs) >= 3 and v_ego < 4.0 and (not lead_close):
            s1 = launch_signs[-3][1]
            s2 = launch_signs[-2][1]
            s3 = launch_signs[-1][1]
            if s1 == s3 and s1 != s2:
              handoff_glitch = True
              launch_handoff = True
        else:
          # Emit compact launch summary when launch window closes.
          if launch_a_target_samples:
            a_min = min(launch_a_target_samples)
            a_max = max(launch_a_target_samples)
          else:
            a_min = 0.0
            a_max = 0.0
          summary_notes = []
          if launch_block_vtsc:
            summary_notes.append("BLOCK_VTSC")
          if launch_block_lead:
            summary_notes.append("BLOCK_LEAD")
          if launch_handoff:
            summary_notes.append("HANDOFF_GLITCH")
          # If we handoff from starting->pid and immediately command non-accel at crawl with no close lead,
          # flag it as potential start->pid handoff stall.
          if len(launch_lcs_path) >= 2 and (3 in launch_lcs_path) and (1 in launch_lcs_path) and (a_max <= 0.20):
            summary_notes.append("HANDOFF_STARTING_PID_STALL")
          if not launch_any_enabled:
            summary_notes.append("ENG_OFF")
          lcs_path_txt = "->".join(_lcs_name(s) for s in launch_lcs_path) if launch_lcs_path else "-"
          print(
            f"{time.strftime('%H:%M:%S')} "
            f"LAUNCH_SUMMARY aT[min,max]=[{_fmt(a_min,2)},{_fmt(a_max,2)}] "
            f"lcs_path={lcs_path_txt} flags={','.join(summary_notes) if summary_notes else '-'}"
          )
          launch_active = False
          launch_signs.clear()
          launch_a_target_samples.clear()
          launch_lcs_path = []

      notes = []
      alert_notes = []
      if lead_close:
        notes.append("lead_close")
      if lead_dup:
        notes.append("lead_dup")
      if vtsc_cap_near_ego:
        notes.append("vtsc_near_ego")
        if planner_not_accel and control_on and v_ego < 8.0:
          alert_notes.append("VTSC_NEAR_EGO_NO_ACCEL")
      if block_by_vtsc:
        notes.append("BLOCK_VTSC")
        alert_notes.append("BLOCK_VTSC")
      if block_by_lead:
        notes.append("BLOCK_LEAD")
        alert_notes.append("BLOCK_LEAD")
      if handoff_glitch:
        notes.append("HANDOFF_GLITCH")
        alert_notes.append("HANDOFF_GLITCH")
      if (vtsc_state >= 0) and (vtsc_state != 0):
        notes.append(f"vtsc_state={vtsc_state}")
      if long_state >= 0:
        notes.append(f"lcs={long_state_name}")
      notes.append(f"sd={sd_state_name}")
      if not control_on:
        notes.append("ENG_OFF")

      should_print = ((not args.only_alerts) and (v_ego <= args.max_speed)) or bool(alert_notes)
      if should_print and (now - last_print_t >= period):
        last_print_t = now
        ts = time.strftime("%H:%M:%S")
        print(
          f"{ts} "
          f"v={_fmt(v_ego,2)} aT={_fmt(a_target,2)} aE={_fmt(a_ego,2)} "
          f"vtsc={_fmt(vtsc_v,2)} "
          f"lead={_bool(lead_status)}[{_fmt(lead_d_rel,1)},{_fmt(headway_s,2)}] "
          f"dup={_bool(lead_dup)} mdlL={model_active_leads} lcs={long_state_name} "
          f"eng={_bool(control_on)} blockV={_bool(block_by_vtsc)} handoff={_bool(handoff_glitch)} "
          f"notes={','.join(notes) if notes else '-'}"
        )

      prev_standstill = standstill

  except KeyboardInterrupt:
    return 0


if __name__ == "__main__":
  raise SystemExit(main())
