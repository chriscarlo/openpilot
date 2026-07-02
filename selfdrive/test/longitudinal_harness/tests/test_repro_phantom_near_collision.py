"""Repro R11: near-collision through a phantom-lead prob dropout while the lead brakes hard.

Scenario (docs/chauffeur/longitudinal/test_runtime_gap_audit_20260701.md, R11):
a lead braking at -3 m/s^2 loses model probability for 0.7 s. radard drops the
published lead (Schmitt exit, LeadProbExit=0.25), and the MPC lead stabilizer
serves a phantom for the whole window (PhantomLeadHoldS=0.80 > 0.7 s) whose vRel
is frozen at the mild drop-time value and whose aLeadK decays toward 0
(selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py _stabilize_raw_leads),
so commanded decel stalls near coast while the true closing rate keeps growing,
and on reacquire the planner panic-corrects against a far closer/slower lead.

Full tici-fidelity loop: device controller mode + radard perception stage,
noise off to isolate the mechanism.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

# Trajectory from the audit doc R11 harness design (20 Hz StepInputs).
DURATION_S = 14.0
CRUISE_SPEED_MPS = 30.0
EGO_V0_MPS = 30.0
INITIAL_GAP_M = 45.0
LEAD_V0_MPS = 30.0
LEAD_DECEL_MPS2 = -3.0
LEAD_V_FLOOR_MPS = 12.0
LEAD_MODEL_PROB = 0.9
BRAKE_START_S = 4.0
# 0.7 s dropout: must stay inside PhantomLeadHoldS=0.80 (device livetune snapshot,
# docs/chauffeur/longitudinal/device_livetune_snapshot_20260701.txt) so the
# stabilizer phantom - not unlatch - owns the whole window.
DROPOUT_START_S = 4.5
DROPOUT_END_S = 5.2

# Must-fail bounds from the audit doc R11 expected failure signal.
ONSET_ACCEL_MPS2 = -1.5
MAX_ONSET_DELAY_S = 1.3
MIN_TRUE_GAP_FLOOR_M = 8.0
# Stall signature: while the true lead brakes at -3 through the window, the
# commanded decel must keep deepening (audit doc R5 assertion (1), same phantom).
WINDOW_MIN_DEEPENING_MPS2 = 0.3
WINDOW_PRE_T_S = DROPOUT_START_S - DT_MDL  # last planner step fed the real lead
WINDOW_END_T_S = DROPOUT_END_S - DT_MDL    # last planner step inside the dropout


def _lead_speed(t_s: float) -> float:
  if t_s < BRAKE_START_S:
    return LEAD_V0_MPS
  return max(LEAD_V_FLOOR_MPS, LEAD_V0_MPS + LEAD_DECEL_MPS2 * (t_s - BRAKE_START_S))


def _build_steps(dropout: bool) -> list[StepInput]:
  steps = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    v_lead = _lead_speed(t_s)
    braking = t_s >= BRAKE_START_S and v_lead > LEAD_V_FLOOR_MPS
    in_dropout = dropout and DROPOUT_START_S <= t_s < DROPOUT_END_S
    # The dropout is expressed as model prob 0.0 with the track kept alive: the
    # radard stage's Schmitt latch drops the published lead (the real vision
    # dropout path) while the plant-side track keeps integrating ground truth.
    lead = LeadDirective(
      status=True,
      v_lead_mps=v_lead,
      model_prob_target=0.0 if in_dropout else LEAD_MODEL_PROB,
      a_lead_k_mps2=LEAD_DECEL_MPS2 if braking else 0.0,
      d_rel_override_m=INITIAL_GAP_M if i == 0 else None,
    )
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config()


@functools.lru_cache(maxsize=2)
def _run(dropout: bool) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name="phantom_near_collision_dropout" if dropout else "phantom_near_collision_clean",
    steps=_build_steps(dropout),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _row_at(trace: list[dict], t_s: float) -> dict:
  return min(trace, key=lambda row: abs(row["t_s"] - t_s))


def _measure(result: SimulationResult) -> dict[str, float | None]:
  onset_t = next((row["t_s"] for row in result.trace if row["planner_accel_mps2"] <= ONSET_ACCEL_MPS2), None)
  return {
    "brake_onset_t_s": onset_t,
    "brake_onset_delay_s": None if onset_t is None else onset_t - BRAKE_START_S,
    "window_pre_planner_accel_mps2": _row_at(result.trace, WINDOW_PRE_T_S)["planner_accel_mps2"],
    "window_end_planner_accel_mps2": _row_at(result.trace, WINDOW_END_T_S)["planner_accel_mps2"],
    "min_true_gap_m": result.summary["minTrueGapM"],
    "peak_planner_brake_mps2": result.summary["peakPlannerBrakeMps2"],
    "peak_realized_brake_mps2": result.summary["peakRealizedBrakeMps2"],
    "min_v_ego_mps": min(row["v_ego_true_mps"] for row in result.trace),
  }


def test_dropout_scenario_wiring() -> None:
  clean = _run(False)
  dropout = _run(True)

  # Tici-fidelity loop resolved as intended.
  for result in (clean, dropout):
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"
  # 0.7 s dropout < phantom hold: the phantom, not unlatch, owns the window.
  assert float(_vehicle_config().params["Longitudinal.LiveTune.PhantomLeadHoldS"]) == pytest.approx(0.8)

  window = [row for row in dropout.trace if DROPOUT_START_S <= row["t_s"] < DROPOUT_END_S]
  assert window
  # radard dropped the published lead for the whole window while ground truth
  # kept flowing from the plant-side track...
  assert all(row["lead_one_published_d_rel_m"] is None for row in window)
  assert all(row["true_min_gap_m"] is not None for row in window)
  # ...yet the MPC still ran against a lead: only the stabilizer phantom can be
  # feeding source lead0 while radard publishes nothing.
  assert all(row["planner_source"] == "lead0" for row in window)
  # Lead reacquired right after the window at its true kinematic position.
  reacquired = _row_at(dropout.trace, DROPOUT_END_S + 0.2)
  assert reacquired["lead_one_published_d_rel_m"] is not None
  assert reacquired["lead_one_published_d_rel_m"] == pytest.approx(reacquired["lead_one_true_d_rel_m"], abs=1.0)
  # Matched pair: both runs share the same trajectory up to the dropout.
  assert _row_at(clean.trace, WINDOW_PRE_T_S)["planner_accel_mps2"] == pytest.approx(
    _row_at(dropout.trace, WINDOW_PRE_T_S)["planner_accel_mps2"])
  # The lead genuinely brakes: the true gap collapses well below the start gap.
  assert dropout.summary["minTrueGapM"] < INITIAL_GAP_M - 20.0


@pytest.mark.xfail(strict=True, reason="Lead stabilizer phantom freezes vRel and decays aLeadK during a prob dropout "
                                       "while the lead brakes, stalling commanded decel and delaying brake onset "
                                       "(test_runtime_gap_audit_20260701.md R11 / GAP 4)")
def test_phantom_dropout_brake_onset_and_gap() -> None:
  clean = _measure(_run(False))
  dropout = _measure(_run(True))

  onset_ok = dropout["brake_onset_delay_s"] is not None and dropout["brake_onset_delay_s"] <= MAX_ONSET_DELAY_S
  gap_ok = dropout["min_true_gap_m"] >= MIN_TRUE_GAP_FLOOR_M
  # During the 0.7 s window the true lead brakes at -3 the whole time: the
  # commanded decel must keep deepening, not stall on the phantom's erased decel.
  window_deepening_ok = (dropout["window_end_planner_accel_mps2"]
                         <= dropout["window_pre_planner_accel_mps2"] - WINDOW_MIN_DEEPENING_MPS2)

  physics = (
    f"phantom dropout run vs clean run (lead -3 m/s^2 from t={BRAKE_START_S}s, "
    f"prob dropout t=[{DROPOUT_START_S}, {DROPOUT_END_S})s):\n"
    f"  brake onset (planner <= {ONSET_ACCEL_MPS2} m/s^2): dropout t={dropout['brake_onset_t_s']}s "
    f"(delay {dropout['brake_onset_delay_s']}s, bound {MAX_ONSET_DELAY_S}s), clean t={clean['brake_onset_t_s']}s "
    f"(delay {clean['brake_onset_delay_s']}s)\n"
    f"  planner accel across dropout window: dropout {dropout['window_pre_planner_accel_mps2']:.3f} -> "
    f"{dropout['window_end_planner_accel_mps2']:.3f} m/s^2 (must deepen by >= {WINDOW_MIN_DEEPENING_MPS2}), "
    f"clean {clean['window_pre_planner_accel_mps2']:.3f} -> {clean['window_end_planner_accel_mps2']:.3f} m/s^2\n"
    f"  min true gap: dropout {dropout['min_true_gap_m']:.2f} m (floor {MIN_TRUE_GAP_FLOOR_M} m), "
    f"clean {clean['min_true_gap_m']:.2f} m\n"
    f"  peak planner brake: dropout {dropout['peak_planner_brake_mps2']:.2f}, clean {clean['peak_planner_brake_mps2']:.2f} m/s^2; "
    f"peak realized brake: dropout {dropout['peak_realized_brake_mps2']:.2f}, clean {clean['peak_realized_brake_mps2']:.2f} m/s^2\n"
    f"  min v_ego: dropout {dropout['min_v_ego_mps']:.2f}, clean {clean['min_v_ego_mps']:.2f} m/s"
  )
  assert onset_ok and gap_ok and window_deepening_ok, physics
