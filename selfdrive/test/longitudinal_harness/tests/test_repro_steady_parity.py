"""Full EV6 regression for planner-only steady-lead parity reconciliation.

The road failure is a cross-signal contradiction: raw measured range is steady
while model/published velocity says ego is continuously closing. RadarD must
publish only robust position evidence; its control kinematics stay identical.
The Hyundai MPC may use that evidence on a private copy only above the exact
configured gap, with an unshaped FCW backstop and immediate threat restore.
"""
from __future__ import annotations

import functools
import statistics

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput


DURATION_S = 6.0
EGO_V0_MPS = 29.0
LEAD_V_MPS = 29.0
CRUISE_V_MPS = 34.0
MEASURED_DREL_M = 55.0
MEASURED_VREL_MPS = -0.8
MODEL_PROB = 0.98
FIX_SLEW_MPS2 = 0.8
ROLLBACK_SLEW_MPS2 = 0.0
EVENT_T_S = 2.5


def _lead(*, frame: int, d_path_m: float = 0.0, v_lat_mps: float = 0.0,
          v_rel_mps: float = MEASURED_VREL_MPS,
          a_lead_mps2: float = 0.0) -> LeadDirective:
  return LeadDirective(
    status=True,
    v_lead_mps=LEAD_V_MPS,
    model_prob_target=MODEL_PROB,
    d_rel_override_m=MEASURED_DREL_M if frame == 0 else None,
    measured_d_rel_m=MEASURED_DREL_M,
    measured_v_rel_mps=v_rel_mps,
    a_lead_k_mps2=a_lead_mps2,
    y_rel_m=d_path_m,
    d_path_m=d_path_m,
    v_lat_mps=v_lat_mps,
    acquisition_reset=(frame == 0),
  )


def _steady_steps(duration_s: float = DURATION_S) -> list[StepInput]:
  return [
    StepInput(
      t_s=frame * DT_MDL,
      cruise_speed_mps=CRUISE_V_MPS,
      lead_one=_lead(frame=frame),
      recorded_v_ego_mps=EGO_V0_MPS,
      recorded_a_ego_mps2=0.0,
      note="flat raw dRel contradicts persistent negative raw/published vRel",
    )
    for frame in range(int(round(duration_s / DT_MDL)))
  ]


def _event_steps(event: str) -> list[StepInput]:
  duration_s = 3.2
  steps: list[StepInput] = []
  for frame in range(int(round(duration_s / DT_MDL))):
    t_s = frame * DT_MDL
    if t_s < EVENT_T_S:
      lead = _lead(frame=frame)
      note = "qualify steady-parity evidence"
    elif event == "braking":
      lead = _lead(frame=frame, v_rel_mps=-3.0, a_lead_mps2=-1.0)
      note = "current braking and fast close"
    elif event == "half_out":
      lead = _lead(frame=frame, d_path_m=1.3, v_lat_mps=5.0)
      note = "high lateral velocity while meaningfully off center"
    else:
      raise ValueError(event)
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_V_MPS,
      lead_one=lead,
      recorded_v_ego_mps=EGO_V0_MPS,
      recorded_a_ego_mps2=0.0,
      note=note,
    ))
  return steps


@functools.lru_cache(maxsize=4)
def _vehicle_config(slew_mps2: float, trust_deficit_mps: float = 0.20):
  overrides = {
    f"VibeTune.Follow.Standard.Headway{idx}": "1.70"
    for idx in range(4)
  }
  overrides.update({
    "LongitudinalPersonality": "1",
    "Longitudinal.LiveTune.SteadyParityTrustDeficitMps": f"{trust_deficit_mps:g}",
    "Longitudinal.LiveTune.SteadyParityVRelSlewMps2": f"{slew_mps2:g}",
    "Longitudinal.LiveTune.SteadyParityHoldS": "0.50",
  })
  return resolve_ev6_vehicle_config(param_overrides=overrides)


@functools.lru_cache(maxsize=4)
def _run_steady(slew_mps2: float, trust_deficit_mps: float = 0.20) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(slew_mps2, trust_deficit_mps),
    scenario_name=f"steady_parity_slew_{slew_mps2:g}_trust_{trust_deficit_mps:g}",
    steps=_steady_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=1,
    perception_filter="auto",
  )


@functools.lru_cache(maxsize=2)
def _run_event(event: str) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(FIX_SLEW_MPS2),
    scenario_name=f"steady_parity_exit_{event}",
    steps=_event_steps(event),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=1,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  return result.trace[::5]


def _assert_full_ev6_path(result: SimulationResult) -> list[dict]:
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["radarUnavailable"] is True
  rows = _planner_rows(result)
  assert rows
  assert all(row["planner_source"] == "lead0" for row in rows[2:])
  assert len({row["lead_one_radard_debug"]["track_id"] for row in rows[2:]}) == 1
  return rows


def test_steady_parity_changes_only_mpc_copy_and_reduces_false_braking() -> None:
  fix = _assert_full_ev6_path(_run_steady(FIX_SLEW_MPS2))
  rollback = _assert_full_ev6_path(_run_steady(ROLLBACK_SLEW_MPS2))
  master_rollback = _assert_full_ev6_path(_run_steady(FIX_SLEW_MPS2, 99.0))

  # The producer evidence must be identical between twins, as must every
  # RadarD-published control kinematic. Only the planner-private copy changes.
  for fixed, base in zip(fix, rollback, strict=True):
    assert fixed["lead_one_raw_d_rel_m"] == pytest.approx(base["lead_one_raw_d_rel_m"], abs=1e-9)
    assert fixed["lead_one_raw_v_rel_mps"] == pytest.approx(base["lead_one_raw_v_rel_mps"], abs=1e-9)
    assert fixed["lead_one_published_d_rel_m"] == pytest.approx(base["lead_one_published_d_rel_m"], abs=1e-9)
    assert fixed["lead_one_published_v_rel_mps"] == pytest.approx(base["lead_one_published_v_rel_mps"], abs=1e-9)
    assert fixed["lead_one_published_v_lead_mps"] == pytest.approx(base["lead_one_published_v_lead_mps"], abs=1e-9)
    assert fixed["lead_one_radard_debug"]["steady_parity_candidate_valid"] == \
      base["lead_one_radard_debug"]["steady_parity_candidate_valid"]

  first_proof = next(row for row in fix if row["lead_one_radard_debug"]["steady_parity_candidate_valid"])
  assert first_proof["t_s"] == pytest.approx(1.70, abs=DT_MDL)
  assert first_proof["lead_one_radard_debug"]["steady_parity_position_slope_mps"] == pytest.approx(0.0, abs=1e-6)
  assert first_proof["lead_one_radard_debug"]["steady_parity_vrel_floor_mps"] == pytest.approx(-0.2, abs=1e-6)

  changed = [
    row for row in fix
    if row["mpc_steady_parity_debug"]["slot0"]["corrected_vrel_mps"] >
       row["mpc_steady_parity_debug"]["slot0"]["original_vrel_mps"] + 1e-6
  ]
  assert changed
  assert max(
    row["mpc_steady_parity_debug"]["slot0"]["corrected_vrel_mps"] -
    row["mpc_steady_parity_debug"]["slot0"]["original_vrel_mps"]
    for row in changed
  ) >= 0.55
  assert all(row["mpc_steady_parity_debug"]["slot0"]["corrected_vrel_mps"] <= 1e-9 for row in changed)
  assert all(
    row["mpc_steady_parity_debug"]["slot0"]["reason"] == "rollback_disabled"
    for row in rollback
  )
  assert all(
    not row["lead_one_radard_debug"]["steady_parity_candidate_valid"]
    for row in master_rollback
  )
  assert all(
    row["mpc_steady_parity_debug"]["slot0"]["reason"] == "rollback_disabled"
    for row in master_rollback
  )
  # The two independent master controls both restore the same baseline output.
  for slew_off, trust_off in zip(rollback, master_rollback, strict=True):
    assert slew_off["lead_one_published_d_rel_m"] == pytest.approx(trust_off["lead_one_published_d_rel_m"], abs=1e-9)
    assert slew_off["lead_one_published_v_rel_mps"] == pytest.approx(trust_off["lead_one_published_v_rel_mps"], abs=1e-9)
    assert slew_off["planner_accel_mps2"] == pytest.approx(trust_off["planner_accel_mps2"], abs=1e-9)

  # In the matched full loop, rollback continues braking for the fictional
  # closure while the fixed planner gently closes still-positive gap surplus.
  fixed_tail = [row["planner_accel_mps2"] for row in fix if row["t_s"] >= 3.0]
  rollback_tail = [row["planner_accel_mps2"] for row in rollback if row["t_s"] >= 3.0]
  assert statistics.median(fixed_tail) >= statistics.median(rollback_tail) + 0.10
  assert fix[-1]["true_min_gap_m"] <= rollback[-1]["true_min_gap_m"] - 0.30
  assert fix[-1]["true_min_gap_m"] >= 50.0


@pytest.mark.parametrize(("event", "expected_reason"), [
  ("braking", "fast_close"),
  ("half_out", "lateral_ambiguity"),
])
def test_steady_parity_restores_unshaped_lead_same_frame_for_threats(event: str, expected_reason: str) -> None:
  rows = _assert_full_ev6_path(_run_event(event))
  event_row = next(row for row in rows if row["t_s"] >= EVENT_T_S - 1e-9)
  pre_event = rows[rows.index(event_row) - 1]

  assert pre_event["mpc_steady_parity_debug"]["slot0"]["active"] is True
  assert pre_event["mpc_steady_parity_debug"]["slot0"]["corrected_vrel_mps"] > \
    pre_event["mpc_steady_parity_debug"]["slot0"]["original_vrel_mps"]
  assert pre_event["lead_one_radard_debug"]["track_id"] == event_row["lead_one_radard_debug"]["track_id"]

  assert event_row["lead_one_radard_debug"]["steady_parity_candidate_valid"] is False
  assert event_row["lead_one_radard_debug"]["steady_parity_reason"] == expected_reason
  debug = event_row["mpc_steady_parity_debug"]["slot0"]
  assert debug["active"] is False
  assert debug["corrected_vrel_mps"] == pytest.approx(debug["original_vrel_mps"], abs=1e-9)

  # A reset is real, not a one-frame blink: no stale evidence may return in the
  # immediate following frames.
  start = rows.index(event_row)
  for row in rows[start:start + 5]:
    assert row["lead_one_radard_debug"]["steady_parity_candidate_valid"] is False
    slot = row["mpc_steady_parity_debug"]["slot0"]
    assert slot["corrected_vrel_mps"] == pytest.approx(slot["original_vrel_mps"], abs=1e-9)
