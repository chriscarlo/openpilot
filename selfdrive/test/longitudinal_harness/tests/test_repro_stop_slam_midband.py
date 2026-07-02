"""Repro oracle: mid-band (~18 mph) brake slam on a calm approach to a stopped lead.

User report: steady ~8 m/s (17.9 mph) approach to a stopped lead with plenty of
warning still ended in a sudden deep brake and a too-close stop. Reproduced with
ZERO perception noise in the tici-fidelity loop (device controller mode + radard
perception stage): ego 8 m/s, stopped lead revealed at 80 m. Steady-state decel
needed to stop 6 m short is only 0.43 m/s^2, yet the planner held exactly 0.0
until ~21.5 m true gap, then ramped -0.75 -> -4.00 in 1.15 s (worst 0.3 s window
1.46 m/s^2), saturated the analytic lead-slowdown ceiling at -4.00, and stopped
2.56 m behind the lead. Bit-identical with every M1/M2/M3/phantom/fade rollback
knob flipped: none of those mechanisms participate.

Mechanism (measured via acc_source_debug, not inferred): on the Hyundai
AI-lead-stability path the MPC sees ONLY the active obstacle — while
_acc_obstacle_mode == 'cruise' the stopped lead is invisible to the solver
regardless of distance (long_mpc.py _select_acc_obstacle returns
cruise_obstacle; stock openpilot instead takes the elementwise min of lead and
cruise obstacle trajectories). For a stopped lead approached at 6-8 m/s every
early cruise->lead handoff leg was structurally blocked:
  - low_speed_queue_hold needs v_ego <= HYUNDAI_LOW_SPEED_QUEUE_V_EGO_MAX (6.0);
  - approach_reacquire needs closing_speed in [1.5, 4.0] m/s (closing to a
    stopped lead equals v_ego ~ 8);
  - raw_gap_hold needs gap_surplus <= 2.5 m (that IS the slam);
  - raw_obstacle_hold needs lead_obstacle[0] <= cruise_obstacle[0] - 1.0, i.e.
    dRel <= ~21.8 m at 8 m/s with no preview shift.
The lead-approach preview (compute_lead_approach_preview) is hard-gated at
v_ego < LEAD_APPROACH_PREVIEW_MIN_SPEED (8.0) on the drifting v_desired filter
state, so at exactly 8 m/s it dies mid-approach, and at 7 m/s it is never
eligible at all — which is why this oracle covers BOTH 8.0 m/s (the preview
knife-edge) and 7.0 m/s (the structurally-blocked core of the band, worst
measured point pre-fix: 1.61 m/s^2 step, peak -4.00, stop gap 2.61 m).

FIX (landed with this oracle flip): the kinematic stopping-need handoff leg in
_select_acc_obstacle (`stopping_need_hold`, live-tunable via
LeadHandoffStoppingNeedDecelMps2 / LeadHandoffStoppingNeedRefSpeedMps,
rollback 1e9 = exact legacy handoff) hands the solver the lead obstacle as soon
as stopping STOP_DISTANCE short of the lead kinematically requires the
threshold decel, using the M1 kinematic-bound two-branch physics with inverted
oncoming semantics. Post-fix at composed HEAD defaults (with the preview
fade): 8/80 -> worst 0.3 s step 0.46, peak -2.64, stop gap 6.51 m;
7/80 -> 0.53 / -2.81 / 5.88 m.

Calm-human reference from 8 m/s with 80 m warning: steady decel 0.43 m/s^2,
peak well under 1.5 m/s^2 with early onset, stopping ~STOP_DISTANCE (6 m)
behind the lead — hence the bounds below (no 0.3 s step > 1.0 m/s^2, peak
brake >= -3.0 m/s^2, stop gap 4.0-7.5 m).
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 22.0
# 8.0 m/s: the preview knife-edge (v_desired filter drifts below the 8.0 gate
# mid-approach). 7.0 m/s: the structurally-blocked core of the band (preview
# never eligible), which pre-fix also resisted a preview-gate-only partial fix
# (patched-preview 7/80 peak was -3.87). Same three-conjunct bounds for both.
EGO_V0_CASES_MPS = (8.0, 7.0)
INITIAL_GAP_M = 80.0
LEAD_MODEL_PROB = 0.98
MPC_STOP_DISTANCE_M = STOP_DISTANCE

# Slam signature bounds, tuned to a calm human stop from 8 m/s with 80 m
# warning (steady 0.43 m/s^2): jerk steps stay under 1.0 m/s^2 per 0.3 s,
# braking never needs -3.0 m/s^2, and the car stops near the nominal 6 m
# stop distance.
SLAM_WINDOW_S = 0.3
MAX_WINDOW_DROP_MPS2 = 1.0
MOVING_V_MPS = 2.0
WELL_BEFORE_STOP_GAP_M = 7.0
MAX_PLANNER_BRAKE_MPS2 = -3.0
STOP_GAP_MIN_M = 4.0
STOP_GAP_MAX_M = 7.5
# Mechanism fingerprint: pre-fix the cruise->lead obstacle handoff landed at
# ~21.5 m true gap; any handoff later than this at 8 m/s forces physics past
# comfort. Used only inside the slam-conditioned mechanism guard below, so it
# is vacuous once the slam is fixed.
LATE_HANDOFF_GAP_M = 25.0


def _calm_required_decel(v0_mps: float) -> float:
  # Steady-state decel to stop MPC_STOP_DISTANCE_M behind the lead from the
  # reveal: what makes this scenario "calm" — any human-plausible plan needs
  # well under comfort-level decel.
  return v0_mps ** 2 / (2.0 * (INITIAL_GAP_M - MPC_STOP_DISTANCE_M))


def _build_steps(v0_mps: float) -> list[StepInput]:
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
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=v0_mps, lead_one=lead,
                           note="mid-band calm noise-free approach to stopped lead"))
  return steps


@functools.lru_cache(maxsize=len(EGO_V0_CASES_MPS))
def _run(v0_mps: float) -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name=f"stop_slam_midband_noise_free_{v0_mps:g}",
    steps=_build_steps(v0_mps),
    initial_speed_mps=v0_mps,
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


def _handoff_row(result: SimulationResult) -> dict | None:
  # First planner frame where the MPC obstacle machine hands the solver the
  # lead obstacle instead of the cruise obstacle.
  return next((row for row in _planner_rows(result)
               if row["mpc_acc_source_debug"].get("active_mode") == "lead"), None)


@pytest.mark.parametrize("v0_mps", EGO_V0_CASES_MPS)
def test_midband_calm_approach_scenario_wiring(v0_mps: float) -> None:
  result = _run(v0_mps)

  # Tici-fidelity loop resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"

  # The scenario is genuinely calm: stopping 6 m behind the lead from the
  # reveal needs well under comfort-level decel.
  assert _calm_required_decel(v0_mps) < 0.6

  # The lead is acquired and published early, and the MPC eventually follows
  # it as lead0.
  published = [row for row in result.trace if row["lead_one_published_d_rel_m"] is not None]
  assert published and published[0]["t_s"] < 1.0
  assert any(row["planner_source"] == "lead0" for row in result.trace)

  # Ego really approaches and stops behind the lead without collision.
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)
  assert stop_row is not None, "ego never stopped inside the scenario window"
  assert stop_row["true_min_gap_m"] > 0.0

  # Stopping-need handoff leg fingerprint: the fix hands the solver the lead
  # obstacle from long range on this approach, and the gap-reclaim machinery
  # (never exercised at ~40-75 m gap surpluses before this leg existed) must
  # stay quiet on every stopping-need-owned frame: no reclaim accel floor, no
  # keep-up floor, no reclaim obstacle push.
  stopping_need_rows = [row for row in _planner_rows(result)
                        if row["mpc_acc_source_debug"].get("reason") == "stopping_need_hold"]
  assert stopping_need_rows, "stopping-need handoff leg never fired on the midband approach"
  assert all(
    abs(row["planner_gap_reclaim_floor_mps2"]) < 1e-9
    and abs(row["mpc_acc_source_debug"].get("lead_keepup_accel_floor", 0.0) or 0.0) < 1e-9
    and abs(row["mpc_acc_source_debug"].get("gap_reclaim_obstacle_push_m", 0.0) or 0.0) < 1e-9
    for row in stopping_need_rows
  ), "gap-reclaim machinery engaged on a stopping-need handoff frame"

  # Mechanism identification (vacuous once the slam is fixed): every
  # slam-grade brake frame during the approach happens with the analytic
  # lead-slowdown ceiling engaged and at/below it (the ceiling owns the frame
  # or the MPC demands even deeper against the late-handoff obstacle), and the
  # slam only exists when the cruise->lead obstacle handoff landed late with
  # the lead-approach preview inactive at handoff.
  deep_frames = [row for row in _approach_rows(result) if row["planner_accel_mps2"] < -2.5]
  assert all(
    row["mpc_acc_source_debug"].get("lead_slowdown_accel_ceiling") is not None
    and row["planner_accel_mps2"] <= row["mpc_acc_source_debug"]["lead_slowdown_accel_ceiling"] + 1e-6
    for row in deep_frames
  ), "deep-brake frames no longer bounded by lead_slowdown_accel_ceiling: mechanism changed, re-investigate"

  worst_drop, _ = _worst_window_drop(result)
  slam_frames = [row for row in _approach_rows(result)
                 if row["planner_accel_mps2"] < MAX_PLANNER_BRAKE_MPS2]
  if worst_drop >= MAX_WINDOW_DROP_MPS2 or slam_frames:
    handoff = _handoff_row(result)
    assert handoff is not None, "slam present but no cruise->lead handoff frame: mechanism changed, re-investigate"
    assert handoff["lead_one_true_d_rel_m"] < LATE_HANDOFF_GAP_M, (
      "slam present but handoff was early: mechanism changed, re-investigate"
    )
    preview = handoff["mpc_lead_preview_debug"].get("lead0", {})
    assert not preview.get("active", False), (
      "slam present with lead-approach preview active at handoff: mechanism changed, re-investigate"
    )


@pytest.mark.parametrize("v0_mps", EGO_V0_CASES_MPS)
def test_no_slam_on_midband_calm_noise_free_approach(v0_mps: float) -> None:
  result = _run(v0_mps)

  worst_drop, worst_row = _worst_window_drop(result)
  approach = _approach_rows(result)
  peak_planner_brake = min(row["planner_accel_mps2"] for row in approach)
  peak_realized_brake = min(row["realized_accel_mps2"] for row in result.trace)
  min_true_gap = result.summary["minTrueGapM"]
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)
  stop_gap = None if stop_row is None else stop_row["true_min_gap_m"]
  handoff = _handoff_row(result)

  no_jerk_spike = worst_drop < MAX_WINDOW_DROP_MPS2
  no_overbrake = peak_planner_brake >= MAX_PLANNER_BRAKE_MPS2
  human_stop_gap = stop_gap is not None and STOP_GAP_MIN_M <= stop_gap <= STOP_GAP_MAX_M

  physics = (
    f"mid-band calm noise-free approach (ego {v0_mps} m/s, stopped lead at {INITIAL_GAP_M} m, "
    f"steady required decel {_calm_required_decel(v0_mps):.2f} m/s^2):\n"
    f"  worst planner drop in any {SLAM_WINDOW_S} s window while v_ego > {MOVING_V_MPS} and "
    f"true gap > {WELL_BEFORE_STOP_GAP_M} m: {worst_drop:.2f} m/s^2 (bound < {MAX_WINDOW_DROP_MPS2})"
    + (f" at t={worst_row['t_s']:.2f} v_ego={worst_row['v_ego_true_mps']:.2f} "
       f"gap={worst_row['lead_one_true_d_rel_m']:.2f} m" if worst_row else "") + "\n"
    f"  peak planner brake during approach: {peak_planner_brake:.2f} m/s^2 (bound >= {MAX_PLANNER_BRAKE_MPS2}); "
    f"peak realized brake: {peak_realized_brake:.2f} m/s^2\n"
    f"  stop gap: {None if stop_gap is None else round(stop_gap, 2)} m "
    f"(calm-human window {STOP_GAP_MIN_M}-{STOP_GAP_MAX_M} m); min true gap: {min_true_gap:.2f} m\n"
    f"  cruise->lead obstacle handoff at "
    + (f"t={handoff['t_s']:.2f} true gap={handoff['lead_one_true_d_rel_m']:.1f} m" if handoff else "<never>")
  )
  assert no_jerk_spike and no_overbrake and human_stop_gap, physics
