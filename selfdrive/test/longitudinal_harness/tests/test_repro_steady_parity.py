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
from types import SimpleNamespace

import pytest

from cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import add_path_relative_lead_metrics, get_RadarState_from_vision
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput
from openpilot.selfdrive.test.longitudinal_harness.radard_stage import _fill_lead_v3


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


def test_radard_stage_reconstructs_distinct_dpath_and_signed_vlat() -> None:
  model = messaging.new_message("modelV2")
  model.modelV2.position.x = [0.0, 100.0]
  model.modelV2.position.y = [0.0, 0.0]
  model.modelV2.init("leadsV3", 1)
  entry = model.modelV2.leadsV3[0]
  raw = SimpleNamespace(
    status=True,
    dRel=55.0,
    yRel=2.75,
    dPath=-1.25,
    vRel=-0.4,
    vLat=-0.8,
    aLeadK=-0.1,
    modelProb=0.98,
  )

  _fill_lead_v3(entry, raw, model_v_ego=EGO_V0_MPS)
  reconstructed = get_RadarState_from_vision(entry, EGO_V0_MPS, EGO_V0_MPS)
  add_path_relative_lead_metrics(reconstructed, model.modelV2, entry)

  assert reconstructed["yRel"] == pytest.approx(raw.dPath, abs=1e-6)
  assert reconstructed["yRel"] != pytest.approx(raw.yRel, abs=1e-6)
  assert reconstructed["dPath"] == pytest.approx(raw.dPath, abs=1e-6)
  assert reconstructed["vLat"] == pytest.approx(raw.vLat, abs=1e-6)


def _lead(*, frame: int, d_path_m: float = 0.0, v_lat_mps: float = 0.0,
          v_rel_mps: float = MEASURED_VREL_MPS,
          a_lead_mps2: float = 0.0,
          d_rel_m: float = MEASURED_DREL_M) -> LeadDirective:
  return LeadDirective(
    status=True,
    v_lead_mps=LEAD_V_MPS,
    model_prob_target=MODEL_PROB,
    d_rel_override_m=d_rel_m if frame == 0 or d_rel_m != MEASURED_DREL_M else None,
    measured_d_rel_m=d_rel_m,
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
    elif event == "near":
      lead = _lead(frame=frame, d_rel_m=15.0)
      note = "raw range enters the near-threat gate"
    elif event == "crash":
      lead = _lead(frame=frame, v_rel_mps=-20.0, a_lead_mps2=-3.0)
      note = "crash-positive genuine lead after active parity correction"
    elif event == "velocity_spike" and t_s < EVENT_T_S + DT_MDL - 1e-9:
      lead = _lead(frame=frame, v_rel_mps=-2.0)
      note = "one-frame raw-vRel-only contradiction"
    elif event == "velocity_spike":
      lead = _lead(frame=frame)
      note = "calm position-consistent rearm"
    elif event == "velocity_timeout":
      lead = _lead(frame=frame, v_rel_mps=-2.0)
      note = "sustained velocity-only contradiction"
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


@functools.lru_cache(maxsize=4)
def _run_event(event: str, trust_deficit_mps: float = 0.20) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(FIX_SLEW_MPS2, trust_deficit_mps),
    scenario_name=f"steady_parity_exit_{event}_trust_{trust_deficit_mps:g}",
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
  ("braking", "current_raw_braking"),
  ("near", "near_threat"),
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


@pytest.mark.parametrize("event", ["braking", "near"])
def test_genuine_threat_bypasses_carryover_and_preserves_recorded_input_twin(event: str) -> None:
  fixed = _assert_full_ev6_path(_run_event(event))
  rollback = _assert_full_ev6_path(_run_event(event, 99.0))
  event_row = next(row for row in fixed if row["t_s"] >= EVENT_T_S - 1e-9)
  event_index = fixed.index(event_row)
  rollback_row = rollback[event_index]
  pre_event = fixed[event_index - 1]

  assert pre_event["mpc_steady_parity_debug"]["slot0"]["active"] is True
  slot = event_row["mpc_steady_parity_debug"]["slot0"]
  assert slot["was_active"] is True
  assert slot["same_track"] is True
  assert slot["producer_current_threat"] is True
  assert slot["current_kinematic_threat"] is True
  assert event_row["planner_comfort_jerk_debug"]["bypassed"] is True
  assert event_row["planner_comfort_jerk_debug"]["bypass_reason"] == "steady_parity_current_threat"

  safety = event_row["planner_steady_parity_threat_debug"]
  assert safety["active"] is True
  assert safety["bound_satisfied"] is True
  assert event_row["planner_accel_mps2"] <= safety["pre_comfort_limiter_output_mps2"] + 1e-9
  # This fixture empirically lands within 0.01 m/s^2 of the master-disabled
  # twin. It is a regression measurement, not a universal rollback theorem:
  # recurrent MPC state differs unless a shadow planner is run.
  assert event_row["planner_accel_mps2"] <= rollback_row["planner_accel_mps2"] + 0.01

  # Recorded replay holds both ego and every public/raw lead input identical;
  # only recurrent planner history differs between the two runs.
  for field in (
    "planner_input_v_ego_mps",
    "planner_input_a_ego_mps2",
    "lead_one_raw_d_rel_m",
    "lead_one_raw_v_rel_mps",
    "lead_one_published_d_rel_m",
    "lead_one_published_v_rel_mps",
    "lead_one_published_v_lead_mps",
  ):
    assert event_row[field] == pytest.approx(rollback_row[field], abs=1e-9)

  # One-shot semantics: the next frame resumes ordinary comfort evolution from
  # the actual threat-frame output rather than persistently bypassing to a much
  # deeper raw MPC request.
  following = fixed[event_index + 1]
  assert following["mpc_steady_parity_debug"]["slot0"]["current_kinematic_threat"] is False
  assert following["planner_steady_parity_threat_debug"]["active"] is False


def test_velocity_only_spike_interrupts_same_frame_and_requires_fresh_epoch() -> None:
  rows = _assert_full_ev6_path(_run_event("velocity_spike"))
  rollback = _assert_full_ev6_path(_run_event("velocity_spike", 99.0))
  event_row = next(row for row in rows if row["t_s"] >= EVENT_T_S - 1e-9)
  event_index = rows.index(event_row)
  pre_event = rows[event_index - 1]
  calm_one = rows[event_index + 1]
  calm_two = rows[event_index + 2]

  assert pre_event["mpc_steady_parity_debug"]["slot0"]["active"] is True
  assert event_row["lead_one_radard_debug"]["steady_parity_candidate_valid"] is False
  assert event_row["lead_one_radard_debug"]["steady_parity_reason"] == "fast_close"
  event_debug = event_row["mpc_steady_parity_debug"]["slot0"]
  assert event_debug["active"] is False
  assert event_debug["corrected_vrel_mps"] == pytest.approx(event_debug["original_vrel_mps"], abs=1e-9)
  assert event_debug["producer_current_threat"] is False
  assert event_debug["current_kinematic_threat"] is False
  assert event_row["planner_steady_parity_threat_debug"]["active"] is False
  assert event_row["planner_comfort_jerk_debug"].get("bypass_reason", "") != "steady_parity_current_threat"

  assert calm_one["lead_one_radard_debug"]["steady_parity_candidate_valid"] is False
  assert calm_one["lead_one_radard_debug"]["steady_parity_reason"] == "sparse_window"
  calm_one_debug = calm_one["mpc_steady_parity_debug"]["slot0"]
  assert calm_one_debug["corrected_vrel_mps"] == pytest.approx(calm_one_debug["original_vrel_mps"], abs=1e-9)

  assert calm_two["lead_one_radard_debug"]["steady_parity_candidate_valid"] is False
  assert calm_two["lead_one_radard_debug"]["steady_parity_reason"] == "sparse_window"
  assert calm_two["mpc_steady_parity_debug"]["slot0"]["active"] is False
  assert all(
    not row["lead_one_radard_debug"]["steady_parity_candidate_valid"]
    for row in rows[event_index:]
  )

  # Proof reset is evidence bookkeeping only; a master-disabled producer twin
  # follows identical raw and public control kinematics frame-for-frame.
  for fixed, base in zip(rows, rollback, strict=True):
    for field in (
      "lead_one_raw_d_rel_m",
      "lead_one_raw_v_rel_mps",
      "lead_one_published_d_rel_m",
      "lead_one_published_v_rel_mps",
      "lead_one_published_v_lead_mps",
    ):
      assert fixed[field] == pytest.approx(base[field], abs=1e-9)


def test_sustained_velocity_only_close_never_becomes_a_threat_bypass() -> None:
  rows = _assert_full_ev6_path(_run_event("velocity_timeout"))
  event_rows = [row for row in rows if row["t_s"] >= EVENT_T_S - 1e-9]
  assert event_rows
  assert all(row["lead_one_radard_debug"]["steady_parity_reason"] == "fast_close" for row in event_rows)
  assert all(not row["mpc_steady_parity_debug"]["slot0"]["producer_current_threat"] for row in event_rows)
  assert all(not row["mpc_steady_parity_debug"]["slot0"]["current_kinematic_threat"] for row in event_rows)
  assert all(not row["planner_steady_parity_threat_debug"]["active"] for row in event_rows)


def test_crash_positive_fcw_sequence_is_frame_identical_to_master_rollback() -> None:
  fixed_result = _run_event("crash")
  rollback_result = _run_event("crash", 99.0)
  assert fixed_result.vehicle["resolvedControllerMode"] == "device"
  assert fixed_result.vehicle["perceptionFilter"] == "radard"
  assert fixed_result.vehicle["radarUnavailable"] is True
  fixed = _planner_rows(fixed_result)
  rollback = _planner_rows(rollback_result)
  fixed_event = [row for row in fixed if row["t_s"] >= EVENT_T_S - 1e-9]
  rollback_event = [row for row in rollback if row["t_s"] >= EVENT_T_S - 1e-9]

  assert [row["mpc_crash_cnt"] for row in fixed_event[:5]] == [1.0, 2.0, 3.0, 4.0, 5.0]
  assert [row["mpc_crash_cnt"] for row in fixed_event] == [row["mpc_crash_cnt"] for row in rollback_event]
  assert [row["planner_fcw"] for row in fixed_event] == [row["planner_fcw"] for row in rollback_event]
  assert [row["planner_fcw"] for row in fixed_event[:3]] == [False, False, True]
