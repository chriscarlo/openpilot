"""Repro: FCW visual-alert emergency override slams to -5.5 m/s^2 below the planner.

Runtime chain (all cited lines verified in this tree):
  - the phantom published-gap collapse (see test_repro_stop_slam_phantom_noise)
    drives mpc.crash_cnt past 2 (long_mpc.py: predicted gap < CRASH_DISTANCE with
    modelProb > 0.9 on consecutive frames),
  - longitudinal_planner.py:363 sets longitudinalPlan.fcw,
  - selfdrive/selfdrived/selfdrived.py raises EventName.fcw (planner_fcw branch),
  - selfdrive/selfdrived/events.py gives it VisualAlert.fcw for 2.0 s,
  - selfdrive/controls/controlsd.py forwards it as hudControl.visualAlert,
  - opendbc/sunnypilot/car/hyundai/longitudinal/controller.py update() routes
    visualAlert == fcw to emergency_control(), which sets actual_accel to
    max(ACCEL_MIN, car_config.accel_min) = -5.5 m/s^2 for the EV6 IN ONE 20 ms
    TICK, bypassing both the no-radar EMA and all jerk limiting, regardless of
    the HyundaiLongitudinalTuning param.

The closed-loop harness models this chain (closed_loop.py FCW visual-alert
fidelity block). Measured on ev6_measured seed 99 (before fix): crash_cnt
reaches 48, planner fcw first fires at t=4.00 s with v_ego 6.03 m/s and TRUE
gap 15.8 m (published 4.28 m), and the controller steps -2.36 -> -5.50 m/s^2 in
one tick. Stock openpilot treats FCW as alert-only; this fork turns it into
full braking — the user's suspected 'separate stopping logic'.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from opendbc.car.structs import CarControl
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import NoiseSeeds, resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

VisualAlert = CarControl.HUDControl.VisualAlert

DURATION_S = 12.0
EGO_V0_MPS = 6.0
CRUISE_SPEED_MPS = 8.0
INITIAL_GAP_M = 40.0
LEAD_MODEL_PROB = 0.98
NOISE_BASE_SEED = 99

# The EV6 car-config emergency floor commanded by emergency_control.
EMERGENCY_ACCEL_MPS2 = -5.5
# No calm approach to a stopped lead with this much true gap should ever raise
# a forward-collision warning, let alone emergency braking.
FCW_TRUE_GAP_M = 8.0
MOVING_V_MPS = 2.0
# The controller must stay within the planner's authority envelope.
MAX_CONTROLLER_BRAKE_MPS2 = -4.5
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
                           note="calm approach to stopped lead under measured EV6 noise"))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config()


@functools.lru_cache(maxsize=1)
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name="stop_slam_fcw_override_seed99",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    noise_seeds=NoiseSeeds.from_base(NOISE_BASE_SEED),
    seed=NOISE_BASE_SEED,
    perception_filter="auto",
  )


def test_fcw_emergency_override_chain_is_real() -> None:
  """Guard: the below-planner mechanism exists in the runtime carcontroller,
  independent of any planner/perception fix — visualAlert=fcw steps the
  commanded accel to the emergency floor in one controller update."""
  from opendbc.sunnypilot.car.hyundai.longitudinal.controller import LongitudinalController

  cfg = _vehicle_config()
  assert cfg.describe()["resolvedControllerMode"] == "device"
  controller = LongitudinalController(cfg.cp, cfg.cp_sp)
  cc_sp = SimpleNamespace(params=cfg.cc_sp_params, flags=cfg.cp_sp.flags)
  cs = SimpleNamespace(out=SimpleNamespace(vEgo=6.0, aEgo=-0.5), aBasis=-0.5)

  def cc(alert):
    return SimpleNamespace(
      actuators=SimpleNamespace(accel=-0.5, longControlState=LongCtrlState.pid),
      longActive=True,
      enabled=True,
      hudControl=SimpleNamespace(visualAlert=alert),
    )

  # Settle on a calm -0.5 m/s^2 demand without any alert.
  for _ in range(50):
    controller.update(cc(None), cc_sp, cs)
  settled_accel = float(controller.actual_accel)
  assert settled_accel > -1.5

  # One update with VisualAlert.fcw steps straight to the emergency floor.
  controller.update(cc(VisualAlert.fcw), cc_sp, cs)
  assert float(controller.actual_accel) == pytest.approx(EMERGENCY_ACCEL_MPS2, abs=1e-6), (
    f"emergency_control commanded {controller.actual_accel:+.2f}, expected {EMERGENCY_ACCEL_MPS2:+.2f} "
    f"in one 20 ms tick from {settled_accel:+.2f}"
  )


def test_fcw_scenario_wiring() -> None:
  result = _run()
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "ev6_measured"
  # The harness FCW chain fields are recorded on every trace row.
  assert all("planner_fcw" in row and "fcw_visual_alert_active" in row and "mpc_crash_cnt" in row
             for row in result.trace)
  # The true kinematics stay calm: stopped lead the whole time, ego stops
  # without collision.
  lead_speeds = {row["active_lead_speed_mps"] for row in result.trace if row["has_any_lead"]}
  assert lead_speeds == {0.0}
  assert result.summary["minTrueGapM"] > 0.0
  # When the planner raises fcw the alert chain engages within one planner step
  # (vacuously true once nothing fires anymore).
  fcw_rows = [i for i, row in enumerate(result.trace) if row["planner_fcw"]]
  assert all(result.trace[min(i + 5, len(result.trace) - 1)]["fcw_visual_alert_active"] for i in fcw_rows)


@pytest.mark.xfail(strict=True, reason="Phantom gap collapse drives mpc.crash_cnt > 2 on a calm approach; "
                                       "longitudinalPlan.fcw -> VisualAlert.fcw -> Hyundai emergency_control "
                                       "commands -5.5 m/s^2 in one 20 ms tick below the planner, bypassing "
                                       "EMA and jerk limits (controller.py update()/emergency_control)")
def test_no_fcw_emergency_braking_on_calm_approach() -> None:
  result = _run()

  false_fcw = next(
    (row for row in result.trace
     if row["planner_fcw"] and row["v_ego_true_mps"] > MOVING_V_MPS
     and row["true_min_gap_m"] is not None and row["true_min_gap_m"] > FCW_TRUE_GAP_M),
    None,
  )
  max_crash_cnt = max(row["mpc_crash_cnt"] for row in result.trace)
  min_controller_accel = min(row["controller_accel_mps2"] for row in result.trace)
  min_true_gap = result.summary["minTrueGapM"]

  no_false_fcw = false_fcw is None
  controller_bounded = min_controller_accel >= MAX_CONTROLLER_BRAKE_MPS2
  safe_gap = min_true_gap >= MIN_TRUE_GAP_FLOOR_M

  physics = (
    f"ev6_measured seed {NOISE_BASE_SEED}, ego {EGO_V0_MPS} m/s cruise {CRUISE_SPEED_MPS} m/s, "
    f"stopped lead at {INITIAL_GAP_M} m:\n"
    f"  false FCW (planner fcw while moving with true gap > {FCW_TRUE_GAP_M} m): "
    + ("none" if false_fcw is None else
       f"t={false_fcw['t_s']:.2f} v_ego={false_fcw['v_ego_true_mps']:.2f} "
       f"true_gap={false_fcw['true_min_gap_m']:.2f} m "
       f"published={false_fcw['lead_one_published_d_rel_m']} m") + "\n"
    f"  max mpc.crash_cnt: {max_crash_cnt:.0f} (fcw threshold > 2)\n"
    f"  min controller accel: {min_controller_accel:+.2f} m/s^2 (bound >= {MAX_CONTROLLER_BRAKE_MPS2}; "
    f"emergency floor {EMERGENCY_ACCEL_MPS2})\n"
    f"  min true gap: {min_true_gap:.2f} m (floor {MIN_TRUE_GAP_FLOOR_M} m)"
  )
  assert no_false_fcw and controller_bounded and safe_gap, physics
