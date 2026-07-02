"""Repro: brake slam on a calm, noise-free approach to a stopped lead.

User report: while slowly, calmly approaching a natural stopping point behind a
stopped lead, the car suddenly slams the brakes well short of where it should
stop. Reproduced with ZERO perception noise in the tici-fidelity loop (device
controller mode + radard perception stage): ego 5 m/s, stopped lead revealed at
45 m, steady-state required decel only ~0.32 m/s^2, yet the planner ramps
-1.37 -> -4.00 m/s^2 in 0.7 s (worst 0.3 s window: -2.42 -> -4.00) at ~8.7 m
true gap and the plant realizes ~-3.9 m/s^2.

Mechanism (measured, not inferred): on every slam frame the planner output
exactly equals mpc.lead_slowdown_accel_ceiling — the analytic lead-slowdown law
get_lead_slowdown_accel_ceiling (selfdrive/controls/lib/longitudinal_mpc_lib/
long_mpc.py) applied as min() over the MPC output at
selfdrive/controls/lib/longitudinal_planner.py:421-423. Its danger term divides
closing_speed^2 by 2*max(danger_surplus, 0.3) where danger_surplus =
dRel - 0.75*headway_gap (long_mpc.py:1012-1015): the denominator collapses near
the natural stop point on ANY final approach, so commanded decel blows up to
the live-tunable cap lead_slowdown_max_decel (=4.0).
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 12.0
EGO_V0_MPS = 5.0
CRUISE_SPEED_MPS = 5.0
INITIAL_GAP_M = 45.0
LEAD_MODEL_PROB = 0.98
MPC_STOP_DISTANCE_M = 6.0
# Steady-state decel needed to stop MPC_STOP_DISTANCE_M behind the lead: this is
# what makes the scenario "calm" — any human-plausible plan needs < 0.35 m/s^2.
CALM_REQUIRED_DECEL_MPS2 = EGO_V0_MPS ** 2 / (2.0 * (INITIAL_GAP_M - MPC_STOP_DISTANCE_M))

# Slam signature bounds (task spec: accel drop >= 1.5 m/s^2 within <= 0.3 s while
# still moving, well before the natural stop point).
SLAM_WINDOW_S = 0.3
MAX_WINDOW_DROP_MPS2 = 1.5
MOVING_V_MPS = 2.0
WELL_BEFORE_STOP_GAP_M = 7.0
# A calm approach must never need more than this much planner brake.
MAX_PLANNER_BRAKE_MPS2 = -3.0
# Safety floor: a "fix" must not trade the slam for a collision or near-miss.
MIN_TRUE_GAP_FLOOR_M = 4.0


def _build_steps() -> list[StepInput]:
  steps = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    lead = LeadDirective(
      status=True,
      v_lead_mps=0.0,
      model_prob_target=LEAD_MODEL_PROB,
      d_rel_override_m=INITIAL_GAP_M if i == 0 else None,
      acquisition_reset=i == 0,
    )
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead,
                           note="calm noise-free approach to stopped lead"))
  return steps


@functools.lru_cache(maxsize=1)
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="stop_slam_ceiling_noise_free",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  # 100 Hz control trace; the planner updates every DT_MDL (5 control ticks).
  return result.trace[::5]


def _approach_rows(result: SimulationResult) -> list[dict]:
  return [row for row in _planner_rows(result)
          if row["v_ego_true_mps"] > MOVING_V_MPS
          and row["lead_one_true_d_rel_m"] is not None
          and row["lead_one_true_d_rel_m"] > WELL_BEFORE_STOP_GAP_M]


def _worst_window_drop(result: SimulationResult) -> tuple[float, dict | None]:
  rows = _planner_rows(result)
  worst = 0.0
  worst_row = None
  for i, row in enumerate(rows):
    if row["v_ego_true_mps"] <= MOVING_V_MPS:
      continue
    gap = row["lead_one_true_d_rel_m"]
    if gap is None or gap <= WELL_BEFORE_STOP_GAP_M:
      continue
    a0 = row["planner_accel_mps2"]
    for j in range(i + 1, len(rows)):
      if rows[j]["t_s"] - row["t_s"] > SLAM_WINDOW_S + 1e-6:
        break
      drop = a0 - rows[j]["planner_accel_mps2"]
      if drop > worst:
        worst = drop
        worst_row = row
  return worst, worst_row


def test_calm_approach_scenario_wiring() -> None:
  result = _run()

  # Tici-fidelity loop resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"

  # The scenario is genuinely calm: stopping 6 m behind the lead from the reveal
  # needs well under comfort-level decel.
  assert CALM_REQUIRED_DECEL_MPS2 < 0.6

  # The lead is acquired and published continuously once latched, and the MPC
  # follows it as lead0 through the approach.
  published = [row for row in result.trace if row["lead_one_published_d_rel_m"] is not None]
  assert published and published[0]["t_s"] < 1.0
  assert any(row["planner_source"] == "lead0" for row in result.trace)

  # Ego really approaches and stops behind the lead without collision.
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)
  assert stop_row is not None, "ego never stopped inside the scenario window"
  assert stop_row["true_min_gap_m"] > 0.0

  # Mechanism identification: every deep-brake frame during the approach is
  # owned by the analytic lead-slowdown ceiling (planner output == ceiling).
  # Vacuously true once the slam is fixed and no such frames remain.
  slam_frames = [row for row in _approach_rows(result) if row["planner_accel_mps2"] < -2.5]
  assert all(
    row["mpc_acc_source_debug"].get("lead_slowdown_accel_ceiling") is not None
    and abs(row["planner_accel_mps2"] - row["mpc_acc_source_debug"]["lead_slowdown_accel_ceiling"]) < 1e-6
    for row in slam_frames
  ), "deep-brake frames not owned by lead_slowdown_accel_ceiling: mechanism changed, re-investigate"


def test_no_brake_slam_on_calm_noise_free_approach() -> None:
  result = _run()

  worst_drop, worst_row = _worst_window_drop(result)
  approach = _approach_rows(result)
  peak_planner_brake = min(row["planner_accel_mps2"] for row in approach)
  peak_realized_brake = min(row["realized_accel_mps2"] for row in result.trace)
  min_true_gap = result.summary["minTrueGapM"]
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)

  no_jerk_spike = worst_drop < MAX_WINDOW_DROP_MPS2
  no_overbrake = peak_planner_brake >= MAX_PLANNER_BRAKE_MPS2
  safe_gap = min_true_gap >= MIN_TRUE_GAP_FLOOR_M

  physics = (
    f"calm noise-free approach (ego {EGO_V0_MPS} m/s, stopped lead at {INITIAL_GAP_M} m, "
    f"steady required decel {CALM_REQUIRED_DECEL_MPS2:.2f} m/s^2):\n"
    f"  worst planner drop in any {SLAM_WINDOW_S} s window while v_ego > {MOVING_V_MPS} and "
    f"true gap > {WELL_BEFORE_STOP_GAP_M} m: {worst_drop:.2f} m/s^2 (bound < {MAX_WINDOW_DROP_MPS2})"
    + (f" at t={worst_row['t_s']:.2f} v_ego={worst_row['v_ego_true_mps']:.2f} "
       f"gap={worst_row['lead_one_true_d_rel_m']:.2f} m" if worst_row else "") + "\n"
    f"  peak planner brake during approach: {peak_planner_brake:.2f} m/s^2 (bound >= {MAX_PLANNER_BRAKE_MPS2}); "
    f"peak realized brake: {peak_realized_brake:.2f} m/s^2\n"
    f"  min true gap: {min_true_gap:.2f} m (floor {MIN_TRUE_GAP_FLOOR_M} m); "
    f"stop gap: {None if stop_row is None else round(stop_row['true_min_gap_m'], 2)} m"
  )
  assert no_jerk_spike and no_overbrake and safe_gap, physics
