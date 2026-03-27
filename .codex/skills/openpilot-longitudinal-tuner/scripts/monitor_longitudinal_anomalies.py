#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _ensure_repo_on_path() -> Path | None:
  candidates: list[Path] = []
  try:
    candidates.append(Path(__file__).resolve().parents[4])
  except Exception:
    pass
  candidates.append(Path("/data/openpilot"))

  for root in candidates:
    if root.exists() and (root / "cereal").exists():
      root_str = str(root)
      if root_str not in sys.path:
        sys.path.insert(0, root_str)
      return root
  return None


REPO_ROOT = _ensure_repo_on_path()

from cereal import car, custom, messaging  # noqa: E402
from openpilot.common.params import Params  # noqa: E402
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import (  # noqa: E402
  format_lead_response_tune_summary,
  read_lead_response_tuning_config,
)

try:  # noqa: E402
  from opendbc.car.hyundai.values import HyundaiFlags
except Exception:  # pragma: no cover - best effort only
  HyundaiFlags = None


FORWARD_GEARS = {
  "drive",
  "eco",
  "sport",
  "low",
  "brake",
  "manumatic",
}


def _fmt(value: Any, digits: int = 2) -> str:
  try:
    number = float(value)
  except Exception:
    return "?"
  if not math.isfinite(number):
    return "?"
  return f"{number:.{digits}f}"


def _bool(value: bool) -> str:
  return "Y" if bool(value) else "N"


def _enum_leaf(value: Any) -> str:
  try:
    text = str(value)
  except Exception:
    return "?"
  if not text:
    return "?"
  return text.split(".")[-1].lower()


def _normalize_gear(raw: Any) -> str:
  try:
    text = str(raw or "").strip().lower()
  except Exception:
    return "unknown"
  return text or "unknown"


def _monitoring_enabled(*, started: bool, gear: Any) -> bool:
  return bool(started) and _normalize_gear(gear) in FORWARD_GEARS


def _load_struct_from_params(params: Params, key: str, schema: Any) -> Any | None:
  try:
    raw = params.get(key)
    if raw:
      return messaging.log_from_bytes(raw, schema)
  except Exception:
    return None
  return None


def _load_bundle_generation(params: Params) -> int | None:
  try:
    bundle = params.get("ModelManager_ActiveBundle") or {}
    generation = bundle.get("generation", None)
    return int(generation) if generation is not None else None
  except Exception:
    return None


def _decode_hyundai_topology(cp: Any) -> list[str]:
  if cp is None or HyundaiFlags is None:
    return []
  try:
    if getattr(cp, "brand", "") != "hyundai":
      return []
    flags = int(getattr(cp, "flags", 0))
  except Exception:
    return []

  bits: list[str] = []
  if flags & int(HyundaiFlags.CANFD):
    bits.append("canfd")
  if flags & int(HyundaiFlags.EV):
    bits.append("ev")
  if flags & int(HyundaiFlags.CANFD_LKA_STEERING):
    bits.append("lka-steer")
  else:
    bits.append("lfa-steer")
  if flags & int(HyundaiFlags.CANFD_CAMERA_SCC):
    bits.append("camera-scc")
  if flags & int(HyundaiFlags.CANFD_ALT_BUTTONS):
    bits.append("alt-buttons")
  return bits


def _find_delayed_command(history: deque[tuple[float, float]], target_t: float) -> float | None:
  for ts, accel in reversed(history):
    if ts <= target_t:
      return accel
  return history[0][1] if history else None


def _count(counter: dict[str, int], key: str, condition: bool) -> int:
  counter[key] = counter.get(key, 0) + 1 if condition else 0
  return counter[key]


def _lead_text(lead: Any, v_ego: float) -> str:
  try:
    if not bool(lead.status):
      return "none"
    d_rel = float(lead.dRel)
    headway = d_rel / max(v_ego, 0.3)
    return f"{d_rel:.0f}m/{headway:.1f}s"
  except Exception:
    return "?"


def _mode_text(sd: Any, lpsp: Any) -> str:
  try:
    dec = lpsp.dec
    if bool(getattr(dec, "active", False)):
      return _enum_leaf(getattr(dec, "state", "acc"))
  except Exception:
    pass
  try:
    return "blended" if bool(getattr(sd, "experimentalMode", False)) else "acc"
  except Exception:
    return "?"


def _cap_text(lpsp: Any, rti: Any) -> tuple[str, float | None]:
  caps: list[tuple[str, float]] = []

  try:
    vtsc = lpsp.visionTurnSpeedControl
    vtsc_state = _enum_leaf(getattr(vtsc, "state", "disabled"))
    vtsc_speed = float(getattr(vtsc, "velocity", 0.0) or 0.0)
    if vtsc_state != "disabled" and vtsc_speed > 0.0:
      caps.append(("vtsc", vtsc_speed))
  except Exception:
    pass

  try:
    slc = lpsp.slc
    slc_active = bool(getattr(slc, "active", False))
    slc_speed = float(getattr(slc, "speedLimit", 0.0) or 0.0)
    slc_offset = float(getattr(slc, "speedLimitOffset", 0.0) or 0.0)
    if slc_active and slc_speed > 0.0:
      caps.append(("slc", slc_speed + slc_offset))
  except Exception:
    pass

  try:
    threat_ahead = bool(getattr(rti, "threatAhead", False))
    recommended = float(getattr(rti, "recommendedSpeed", 0.0) or 0.0)
    if threat_ahead and recommended > 0.0:
      caps.append(("rti", recommended))
  except Exception:
    pass

  if not caps:
    return "-", None

  label, speed = min(caps, key=lambda item: item[1])
  return f"{label}:{speed:.1f}", speed


def main() -> int:
  parser = argparse.ArgumentParser(description="Watch longitudinal anomalies live.")
  parser.add_argument("--hz", type=float, default=5.0, help="render/update rate in Hz")
  parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds; 0 runs until Ctrl-C")
  parser.add_argument("--only-alerts", action="store_true", help="print only alerting rows")
  parser.add_argument("--all-gears", action="store_true", help="do not auto-pause offroad or outside a forward gear")
  parser.add_argument("--jsonl-out", type=str, default="", help="optional JSONL output path")
  parser.add_argument("--show-live-tune", action="store_true", help="print effective live lead-tune values at startup")
  args = parser.parse_args()

  period = 1.0 / max(1.0, float(args.hz))
  timeout_ms = max(50, int(period * 1000))

  params = Params()
  cp = _load_struct_from_params(params, "CarParams", car.CarParams)
  cp_sp = _load_struct_from_params(params, "CarParamsSP", custom.CarParamsSP)
  generation = _load_bundle_generation(params)

  brand = getattr(cp, "brand", "?") if cp is not None else "?"
  fingerprint = getattr(cp, "carFingerprint", "?") if cp is not None else "?"
  owns_long = bool(getattr(cp, "openpilotLongitudinalControl", False)) if cp is not None else False
  actuator_delay = float(getattr(cp, "longitudinalActuatorDelay", 0.5)) if cp is not None else 0.5
  hyundai_topology = ",".join(_decode_hyundai_topology(cp)) or "-"
  hyundai_tuning = str(params.get("HyundaiLongitudinalTuning") or "0") if brand == "hyundai" else "-"
  sp_flags = int(getattr(cp_sp, "flags", 0)) if cp_sp is not None else 0

  print(
    f"car={fingerprint} brand={brand} op_long={_bool(owns_long)} "
    f"delay={_fmt(actuator_delay, 2)} bundle_gen={generation if generation is not None else '?'} "
    f"hyundai={hyundai_topology} hy_long_tune={hyundai_tuning} sp_flags={sp_flags}"
  )
  if args.show_live_tune:
    print(f"live_tune {format_lead_response_tune_summary(read_lead_response_tuning_config(params))}")
  print("watching longitudinal anomalies: Ctrl-C to stop")
  print("t v aE lp cc can mdl src mode cap lead notes")

  jsonl_file = None
  if args.jsonl_out:
    out_path = Path(args.jsonl_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = out_path.open("a", encoding="utf-8")

  services = [
    "carState",
    "carControl",
    "carOutput",
    "carStateSP",
    "controlsState",
    "deviceState",
    "longitudinalPlan",
    "longitudinalPlanSP",
    "modelV2",
    "radarState",
    "rtiStateSP",
    "selfdriveState",
  ]
  sm = messaging.SubMaster(services, poll="longitudinalPlan")

  counters: dict[str, int] = defaultdict(int)
  history: deque[tuple[float, float]] = deque(maxlen=max(128, int(args.hz * 20)))
  start_t = time.monotonic()
  last_monitor_state: tuple[bool, bool, str] | None = None

  try:
    while True:
      now = time.monotonic()
      if args.duration > 0.0 and (now - start_t) >= args.duration:
        return 0

      sm.update(timeout_ms)
      if not sm.updated["longitudinalPlan"]:
        continue

      cs = sm["carState"]
      cc = sm["carControl"]
      co = sm["carOutput"]
      ctrl = sm["controlsState"]
      lp = sm["longitudinalPlan"]
      lpsp = sm["longitudinalPlanSP"]
      md = sm["modelV2"]
      rs = sm["radarState"]
      rti = sm["rtiStateSP"]
      sd = sm["selfdriveState"]
      device = sm["deviceState"]

      gear = _normalize_gear(getattr(cs, "gearShifter", "unknown"))
      started = bool(getattr(device, "started", False))
      monitor_active = True if args.all_gears else _monitoring_enabled(started=started, gear=gear)
      monitor_state = (monitor_active, started, gear)
      if monitor_state != last_monitor_state:
        status = "active" if monitor_active else "paused"
        print(f"{time.strftime('%H:%M:%S')} monitor={status} started={_bool(started)} gear={gear}")
        last_monitor_state = monitor_state

      if not monitor_active:
        counters.clear()
        history.clear()
        continue

      v_ego = float(getattr(cs, "vEgo", 0.0) or 0.0)
      a_ego = float(getattr(cs, "aEgo", 0.0) or 0.0)
      standstill = bool(getattr(cs, "standstill", False)) or v_ego < 0.1
      gas_pressed = bool(getattr(cs, "gasPressed", False))
      brake_pressed = bool(getattr(cs, "brakePressed", False))
      pedal_override = gas_pressed or brake_pressed

      plan_a_target = float(getattr(lp, "aTarget", 0.0) or 0.0)
      allow_throttle = bool(getattr(lp, "allowThrottle", True))
      should_stop = bool(getattr(lp, "shouldStop", False))
      plan_source = _enum_leaf(getattr(lp, "longitudinalPlanSource", "?"))

      model_accel = float(getattr(getattr(md, "action", object()), "desiredAcceleration", 0.0) or 0.0)
      model_should_stop = bool(getattr(getattr(md, "action", object()), "shouldStop", False))
      throttle_prob = None
      try:
        throttle_prob = float(md.meta.disengagePredictions.gasPressProbs[1])
      except Exception:
        throttle_prob = None

      cc_accel = float(getattr(getattr(cc, "actuators", object()), "accel", 0.0) or 0.0)
      can_accel = float(getattr(getattr(co, "actuatorsOutput", object()), "accel", 0.0) or 0.0)
      history.append((now, can_accel))
      delayed_can = _find_delayed_command(history, now - actuator_delay)

      long_state = _enum_leaf(getattr(ctrl, "longControlState", "?"))
      control_on = bool(getattr(cc, "longActive", False)) or bool(getattr(sd, "enabled", False))

      mode = _mode_text(sd, lpsp)
      cap_label, cap_speed = _cap_text(lpsp, rti)
      lead_text = _lead_text(getattr(rs, "leadOne", object()), v_ego)

      blendish = mode == "blended" or bool(getattr(sd, "experimentalMode", False))

      throttle_block = control_on and not pedal_override and not allow_throttle and plan_a_target > 0.15 and v_ego > 2.5
      plan_model_gap = control_on and not pedal_override and blendish and abs(plan_a_target - model_accel) > 0.75 and v_ego > 2.0
      shaping_gap = owns_long and control_on and not pedal_override and abs(cc_accel - can_accel) > 0.35
      tracking_gap = (
        owns_long and control_on and not pedal_override and delayed_can is not None and
        abs(delayed_can - a_ego) > 0.75 and v_ego > 1.0
      )
      stop_mismatch = (
        owns_long and control_on and v_ego < 5.0 and
        (should_stop != (long_state == "stopping"))
      )

      throttle_block_n = _count(counters, "throttle_block", throttle_block)
      plan_model_gap_n = _count(counters, "plan_model_gap", plan_model_gap)
      shaping_gap_n = _count(counters, "shaping_gap", shaping_gap)
      tracking_gap_n = _count(counters, "tracking_gap", tracking_gap)
      stop_mismatch_n = _count(counters, "stop_mismatch", stop_mismatch)

      alerts: list[str] = []
      notes: list[str] = []

      if throttle_block_n >= 3:
        notes.append("throttle-block")
      elif not allow_throttle:
        notes.append("thr-gated")

      if plan_model_gap_n >= 3:
        notes.append("plan-vs-model")

      if shaping_gap_n >= max(3, int(args.hz)):
        notes.append("shape")

      if tracking_gap_n >= max(3, int(max(1.0, actuator_delay * 2.0) * args.hz)):
        alerts.append("tracking-gap")

      if stop_mismatch_n >= 2:
        alerts.append("stop-mismatch")

      if cap_speed is not None and control_on and plan_a_target > 0.1 and v_ego > cap_speed + 1.0:
        notes.append("external-cap")

      if pedal_override:
        notes.append("pedal")
      if should_stop:
        notes.append("plan-stop")
      if model_should_stop and not should_stop:
        notes.append("model-stop-only")
      if throttle_prob is not None and not allow_throttle:
        notes.append(f"gasProb={throttle_prob:.2f}")

      row = {
        "t": time.time(),
        "vEgo": v_ego,
        "aEgo": a_ego,
        "planATarget": plan_a_target,
        "carControlAccel": cc_accel,
        "carOutputAccel": can_accel,
        "delayedCarOutputAccel": delayed_can,
        "modelAccel": model_accel,
        "planSource": plan_source,
        "mode": mode,
        "cap": cap_label,
        "lead": lead_text,
        "longControlState": long_state,
        "allowThrottle": allow_throttle,
        "shouldStop": should_stop,
        "modelShouldStop": model_should_stop,
        "gasPressed": gas_pressed,
        "brakePressed": brake_pressed,
        "notes": notes,
        "alerts": alerts,
      }

      if jsonl_file is not None:
        jsonl_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        jsonl_file.flush()

      if args.only_alerts and not alerts:
        continue

      prefix = "ALERT" if alerts else "INFO "
      note_text = ",".join(alerts + [n for n in notes if n not in alerts]) if (alerts or notes) else "-"
      line = (
        f"{prefix} {time.strftime('%H:%M:%S')} "
        f"v={_fmt(v_ego, 1)} aE={_fmt(a_ego, 2)} lp={_fmt(plan_a_target, 2)} "
        f"cc={_fmt(cc_accel, 2)} can={_fmt(can_accel, 2)} mdl={_fmt(model_accel, 2)} "
        f"src={plan_source} mode={mode} cap={cap_label} lead={lead_text} notes={note_text}"
      )
      print(line)
  except KeyboardInterrupt:
    return 0
  finally:
    if jsonl_file is not None:
      jsonl_file.close()


if __name__ == "__main__":
  raise SystemExit(main())
