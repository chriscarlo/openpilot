from __future__ import annotations

from collections.abc import Iterable


LEAD_EVENT_OVERSHOOT_WINDOW_S = 2.0
LEAD_EVENT_NAMES = {"lead_reveal", "handoff_reveal"}
HANDOFF_PREREVEAL_WINDOW_S = 0.75


def summarize_trace(trace: list[dict], *, vehicle: dict, scenario_name: str, noise_profile: str) -> dict:
  if not trace:
    return {"scenario": scenario_name, "noiseProfile": noise_profile, "vehicle": vehicle}

  follow_rows = [row for row in trace if row["t_s"] >= 0.5 and _row_follow_target_speed(row) is not None]
  if follow_rows:
    deltas = [row["v_ego_true_mps"] - _row_follow_target_speed(row) for row in follow_rows]
    follow_overshoot = max(max(deltas), 0.0)
    follow_undershoot = max(max(-delta for delta in deltas), 0.0)
  else:
    follow_overshoot = 0.0
    follow_undershoot = 0.0

  active_gap_rows = [row for row in trace if row["true_min_gap_m"] is not None]
  final_gap_m = active_gap_rows[-1]["true_min_gap_m"] if active_gap_rows else None
  final_headway_s = None
  if final_gap_m is not None:
    final_headway_s = final_gap_m / max(trace[-1]["v_ego_true_mps"], 1.0)

  reclaim_delay_s = _max_latency(
    _event_latencies(
      trace,
      "pullaway_start",
      lambda row: row["longcontrol_accel_mps2"] > 0.15,
    ),
  )
  lead_acquire_latency_s = _max_latency(
    _event_latencies(
      trace,
      "lead_reveal",
      lambda row: bool(row["planner_source"]) and row["planner_source"] != "cruise",
    ),
  )
  handoff_brake_latency_s = _max_latency(
    _event_latencies(
      trace,
      "handoff_reveal",
      lambda row: row["planner_accel_mps2"] <= -0.2,
    ),
  )
  lead_event_overshoot_growth_mps = _max_lead_event_overshoot_growth(trace)
  handoff_prereveal_mean_overshoot_mps = _max_handoff_prereveal_mean_overshoot(trace)
  handoff_prereveal_speed_loss_mps = _max_handoff_prereveal_speed_loss(trace)
  handoff_prereveal_cruise_fraction = _max_handoff_prereveal_cruise_fraction(trace)
  lead_control_rows = _lead_control_rows(trace)
  lead_control_fraction = float(len(lead_control_rows)) / float(len(trace))
  lead_control_accel_sign_reversals = _sign_reversals(lead_control_rows, "controller_accel_mps2", epsilon=0.05)
  max_lead_control_accel_jerk_mps3 = _max_abs_rate(lead_control_rows, "controller_accel_mps2", max_step_s=0.06)

  return {
    "scenario": scenario_name,
    "noiseProfile": noise_profile,
    "vehicle": vehicle,
    "peakPlannerBrakeMps2": min(row["planner_accel_mps2"] for row in trace),
    "peakPlannerAccelMps2": max(row["planner_accel_mps2"] for row in trace),
    "peakLongControlBrakeMps2": min(row["longcontrol_accel_mps2"] for row in trace),
    "peakLongControlAccelMps2": max(row["longcontrol_accel_mps2"] for row in trace),
    "peakControllerBrakeMps2": min(row["controller_accel_mps2"] for row in trace),
    "peakControllerAccelMps2": max(row["controller_accel_mps2"] for row in trace),
    "peakRealizedBrakeMps2": min(row["realized_accel_mps2"] for row in trace),
    "peakRealizedAccelMps2": max(row["realized_accel_mps2"] for row in trace),
    "maxFollowOvershootMps": follow_overshoot,
    "leadEventOvershootGrowthMps": lead_event_overshoot_growth_mps,
    "handoffPrerevealMeanOvershootMps": handoff_prereveal_mean_overshoot_mps,
    "handoffPrerevealSpeedLossMps": handoff_prereveal_speed_loss_mps,
    "handoffPrerevealCruiseFraction": handoff_prereveal_cruise_fraction,
    "maxFollowUndershootMps": follow_undershoot,
    "leadControlFraction": lead_control_fraction,
    "leadControlDropoutFraction": 1.0 - lead_control_fraction,
    "leadControlAccelSignReversals": lead_control_accel_sign_reversals,
    "maxLeadControlAccelJerkMps3": max_lead_control_accel_jerk_mps3,
    "minTrueGapM": min((row["true_min_gap_m"] for row in active_gap_rows), default=None),
    "finalTrueGapM": final_gap_m,
    "finalHeadwayS": final_headway_s,
    "plannerControllerDivergenceMps2": max(abs(row["planner_accel_mps2"] - row["controller_accel_mps2"]) for row in trace),
    "hyundaiControllerShapingDivergenceMps2": max(abs(row["longcontrol_accel_mps2"] - row["controller_accel_mps2"]) for row in trace),
    "longControlRealizedDivergenceMps2": max(abs(row["longcontrol_accel_mps2"] - row["realized_accel_mps2"]) for row in trace),
    "reclaimDelayS": reclaim_delay_s,
    "leadAcquireLatencyS": lead_acquire_latency_s,
    "handoffBrakeLatencyS": handoff_brake_latency_s,
  }


def _first_latency(trace: Iterable[dict], event_t_s: float | None, predicate) -> float | None:
  if event_t_s is None:
    return None
  for row in trace:
    if row["t_s"] >= event_t_s and predicate(row):
      return row["t_s"] - event_t_s
  return None


def _event_latencies(trace: Iterable[dict], event_name: str, predicate) -> list[float]:
  rows = list(trace)
  latencies: list[float] = []
  for row in rows:
    if row.get("event") != event_name:
      continue
    latency = _first_latency(rows, float(row["t_s"]), predicate)
    if latency is not None:
      latencies.append(float(latency))
  return latencies


def _max_latency(latencies: Iterable[float]) -> float | None:
  latencies = [float(latency) for latency in latencies]
  if not latencies:
    return None
  return max(latencies)


def _row_follow_target_speed(row: dict) -> float | None:
  control_speed = row.get("control_lead_speed_mps")
  if control_speed is not None:
    return float(control_speed)
  if row.get("planner_source") in ("lead0", "lead1") and row.get("active_lead_speed_mps") is not None:
    return float(row["active_lead_speed_mps"])
  return None


def _row_follow_overshoot(row: dict) -> float | None:
  target_speed = _row_follow_target_speed(row)
  if target_speed is None:
    return None
  return max(0.0, float(row["v_ego_true_mps"]) - target_speed)


def _lead_control_rows(trace: Iterable[dict]) -> list[dict]:
  return [row for row in trace if row.get("planner_source") in ("lead0", "lead1")]


def _sign_reversals(rows: Iterable[dict], field: str, *, epsilon: float) -> int:
  previous_sign = 0
  reversals = 0
  for row in rows:
    value = float(row[field])
    sign = 1 if value > epsilon else -1 if value < -epsilon else 0
    if sign == 0:
      continue
    if previous_sign != 0 and sign != previous_sign:
      reversals += 1
    previous_sign = sign
  return reversals


def _max_abs_rate(rows: Iterable[dict], field: str, *, max_step_s: float) -> float:
  previous_t: float | None = None
  previous_value: float | None = None
  max_rate = 0.0
  for row in rows:
    current_t = float(row["t_s"])
    current_value = float(row[field])
    if previous_t is not None and previous_value is not None:
      dt_s = current_t - previous_t
      if 0.0 < dt_s <= max_step_s:
        max_rate = max(max_rate, abs(current_value - previous_value) / dt_s)
    previous_t = current_t
    previous_value = current_value
  return max_rate


def _max_lead_event_overshoot_growth(trace: list[dict]) -> float:
  if not trace:
    return 0.0

  growths: list[float] = []
  lead_event_rows = [row for row in trace if row.get("event") in LEAD_EVENT_NAMES]

  for idx, event_row in enumerate(lead_event_rows):
    event_t = float(event_row["t_s"])
    next_event_t = None
    if idx + 1 < len(lead_event_rows):
      next_event_t = float(lead_event_rows[idx + 1]["t_s"])

    event_overshoot = _row_follow_overshoot(event_row)
    if event_overshoot is None:
      event_overshoot = 0.0

    window_end_t = event_t + LEAD_EVENT_OVERSHOOT_WINDOW_S
    if next_event_t is not None:
      window_end_t = min(window_end_t, next_event_t)

    max_overshoot = event_overshoot
    for row in trace:
      row_t = float(row["t_s"])
      if row_t < event_t:
        continue
      if row_t > window_end_t:
        break
      row_overshoot = _row_follow_overshoot(row)
      if row_overshoot is not None:
        max_overshoot = max(max_overshoot, row_overshoot)

    growths.append(max(0.0, max_overshoot - event_overshoot))

  return max(growths, default=0.0)


def _max_handoff_prereveal_mean_overshoot(trace: list[dict]) -> float:
  if not trace:
    return 0.0

  event_means: list[float] = []
  for event_row in trace:
    if event_row.get("event") != "handoff_reveal":
      continue

    reveal_target_speed = _row_follow_target_speed(event_row)
    if reveal_target_speed is None:
      continue

    event_t = float(event_row["t_s"])
    window_start_t = max(0.0, event_t - HANDOFF_PREREVEAL_WINDOW_S)
    prereveal_rows = [
      row for row in trace
      if window_start_t <= float(row["t_s"]) < event_t
    ]
    if not prereveal_rows:
      continue

    overshoots = [max(0.0, float(row["v_ego_true_mps"]) - float(reveal_target_speed)) for row in prereveal_rows]
    event_means.append(sum(overshoots) / len(overshoots))

  return max(event_means, default=0.0)


def _max_handoff_prereveal_speed_loss(trace: list[dict]) -> float:
  if not trace:
    return 0.0

  event_losses: list[float] = []
  for event_row in trace:
    if event_row.get("event") != "handoff_reveal":
      continue

    reveal_source = str(event_row.get("planner_source") or "")
    incoming_status_key = {
      "lead0": "lead_one_status",
      "lead1": "lead_two_status",
    }.get(reveal_source)

    event_t = float(event_row["t_s"])
    window_start_t = max(0.0, event_t - HANDOFF_PREREVEAL_WINDOW_S)
    prereveal_rows = [
      row for row in trace
      if window_start_t <= float(row["t_s"]) < event_t
    ]
    fallback_rows = list(prereveal_rows)
    if incoming_status_key is not None:
      prereveal_rows = [row for row in prereveal_rows if bool(row.get(incoming_status_key))]
    else:
      prereveal_rows = [row for row in prereveal_rows if bool(row.get("lead_one_status")) or bool(row.get("lead_two_status"))]
    if len(prereveal_rows) < 2:
      prereveal_rows = fallback_rows
    if len(prereveal_rows) < 2:
      continue

    speeds = [float(row["v_ego_true_mps"]) for row in prereveal_rows]
    event_losses.append(max(speeds) - min(speeds))

  return max(event_losses, default=0.0)


def _max_handoff_prereveal_cruise_fraction(trace: list[dict]) -> float:
  if not trace:
    return 0.0

  event_fractions: list[float] = []
  for event_row in trace:
    if event_row.get("event") != "handoff_reveal":
      continue

    reveal_source = str(event_row.get("planner_source") or "")
    incoming_status_key = {
      "lead0": "lead_one_status",
      "lead1": "lead_two_status",
    }.get(reveal_source)

    event_t = float(event_row["t_s"])
    window_start_t = max(0.0, event_t - HANDOFF_PREREVEAL_WINDOW_S)
    prereveal_rows = [
      row for row in trace
      if window_start_t <= float(row["t_s"]) < event_t
    ]
    if incoming_status_key is not None:
      prereveal_rows = [row for row in prereveal_rows if bool(row.get(incoming_status_key))]
    else:
      prereveal_rows = [row for row in prereveal_rows if bool(row.get("lead_one_status")) or bool(row.get("lead_two_status"))]
    if not prereveal_rows:
      continue

    cruise_rows = [row for row in prereveal_rows if row.get("planner_source") == "cruise"]
    event_fractions.append(float(len(cruise_rows)) / float(len(prereveal_rows)))

  return max(event_fractions, default=0.0)
