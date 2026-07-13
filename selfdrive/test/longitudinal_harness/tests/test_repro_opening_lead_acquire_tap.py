"""Regression: a first-visible pulling-away lead must not inject a brake tap.

The 2026-07-12 surface-street drive exposed a planner-local transient distinct
from the earlier radard position-collapse loop and from the stop-launch
lead->cruise release: while cruise was accelerating, a newly published lead was
already pulling away, but the newly appearing lead-present cruise cap collapsed
the delayed cruise profile and changed aTarget from +1.499 to -0.702 m/s^2 for
one frame.  Source stayed ``cruise`` throughout.

This full EV6 device-controller + real-radard harness scenario pins that edge and
pairs it with a dangerous closing-lead acquisition.  The comfort limiter may
smooth only the benign opening case; genuine braking must retain immediate raw
authority.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 7.0
EGO_V0_MPS = 11.0
CRUISE_SPEED_MPS = 19.0
REVEAL_T_S = 3.0
LEAD_PROB = 0.98

OPENING_LEAD_V_MPS = 15.0
OPENING_GAP_M = 24.0
CLOSING_LEAD_V_MPS = 8.0
CLOSING_GAP_M = 18.0
DECEL_LEAD_V0_MPS = 13.0
DECEL_LEAD_ACCEL_MPS2 = -1.2
DECEL_LEAD_GAP_M = OPENING_GAP_M

MAX_BENIGN_ONE_FRAME_DROP_MPS2 = 0.40
MIN_DANGEROUS_BRAKE_MPS2 = -2.0


def _build_steps(*, lead_v_mps: float, gap_m: float, lead_accel_mps2: float = 0.0) -> list[StepInput]:
  steps: list[StepInput] = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    if t_s < REVEAL_T_S:
      lead = LeadDirective()
    else:
      reveal_now = abs(t_s - REVEAL_T_S) < DT_MDL * 0.5
      v_lead = max(0.0, lead_v_mps + lead_accel_mps2 * (t_s - REVEAL_T_S))
      lead = LeadDirective(
        status=True,
        v_lead_mps=v_lead,
        model_prob_target=LEAD_PROB,
        a_lead_k_mps2=lead_accel_mps2,
        d_rel_override_m=gap_m if reveal_now else None,
        acquisition_reset=reveal_now,
      )
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_SPEED_MPS,
      lead_one=lead,
      event="lead_reveal" if abs(t_s - REVEAL_T_S) < DT_MDL * 0.5 else None,
      note="first-visible lead during cruise acceleration",
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _vehicle_config(handoff_window_s: float = 0.4):
  # Pin the mechanism defaults so a later device snapshot cannot silently turn
  # this regression into a tune comparison. EDGE1 is unrelated to this edge.
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.4",
    "Longitudinal.LiveTune.HandoffLimitWindowS": f"{handoff_window_s:g}",
    "Longitudinal.LiveTune.HandoffLimitMaxDeltaMps2": "0.3",
    "Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2": "10.0",
  })


def _run(*, lead_v_mps: float, gap_m: float, name: str,
         lead_accel_mps2: float = 0.0, handoff_window_s: float = 0.4) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(handoff_window_s),
    scenario_name=name,
    steps=_build_steps(lead_v_mps=lead_v_mps, gap_m=gap_m, lead_accel_mps2=lead_accel_mps2),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


@functools.lru_cache(maxsize=1)
def _run_opening() -> SimulationResult:
  return _run(lead_v_mps=OPENING_LEAD_V_MPS, gap_m=OPENING_GAP_M, name="opening_lead_acquire_tap")


@functools.lru_cache(maxsize=1)
def _run_opening_rollback() -> SimulationResult:
  return _run(lead_v_mps=OPENING_LEAD_V_MPS, gap_m=OPENING_GAP_M,
              name="opening_lead_acquire_tap_rollback", handoff_window_s=0.0)


@functools.lru_cache(maxsize=1)
def _run_closing() -> SimulationResult:
  return _run(lead_v_mps=CLOSING_LEAD_V_MPS, gap_m=CLOSING_GAP_M, name="closing_lead_acquire_safety")


@functools.lru_cache(maxsize=1)
def _run_decelerating() -> SimulationResult:
  return _run(lead_v_mps=DECEL_LEAD_V0_MPS, gap_m=DECEL_LEAD_GAP_M,
              lead_accel_mps2=DECEL_LEAD_ACCEL_MPS2, name="decelerating_lead_acquire_safety")


@functools.lru_cache(maxsize=1)
def _run_decelerating_rollback() -> SimulationResult:
  return _run(lead_v_mps=DECEL_LEAD_V0_MPS, gap_m=DECEL_LEAD_GAP_M,
              lead_accel_mps2=DECEL_LEAD_ACCEL_MPS2,
              name="decelerating_lead_acquire_safety_rollback", handoff_window_s=0.0)


def _planner_rows(result: SimulationResult) -> list[dict]:
  return result.trace[::5]


def _first_published_edge(result: SimulationResult) -> tuple[dict, dict]:
  rows = _planner_rows(result)
  edge = next(((prev, row) for prev, row in zip(rows[:-1], rows[1:], strict=False)
               if prev["lead_one_published_d_rel_m"] is None
               and row["lead_one_published_d_rel_m"] is not None), None)
  assert edge is not None, "radard never published the revealed lead"
  return edge


def test_opening_lead_scenario_wiring() -> None:
  result = _run_opening()
  prev, row = _first_published_edge(result)

  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert prev["planner_source"] == row["planner_source"] == "cruise"
  assert row["lead_one_raw_v_rel_mps"] > 0.5
  assert row["lead_one_published_v_rel_mps"] > 0.5
  assert row["lead_one_raw_a_lead_k_mps2"] >= 0.0

  # This really is an opening/nonshrinking geometry around publication, not a
  # closing lead whose braking demand the comfort path would be hiding.
  event_rows = [r for r in _planner_rows(result) if row["t_s"] <= r["t_s"] <= row["t_s"] + 0.15]
  true_gaps = [r["lead_one_true_d_rel_m"] for r in event_rows]
  assert true_gaps == sorted(true_gaps)


def test_first_visible_opening_lead_does_not_tap_brakes() -> None:
  result = _run_opening()
  rollback = _run_opening_rollback()
  prev, row = _first_published_edge(result)
  roll_prev, roll_row = _first_published_edge(rollback)
  event_rows = [r for r in _planner_rows(result) if row["t_s"] <= r["t_s"] <= row["t_s"] + 0.25]
  pairs = list(zip([prev, *event_rows[:-1]], event_rows, strict=True))
  worst_drop = min(cur["planner_accel_mps2"] - prior["planner_accel_mps2"] for prior, cur in pairs)
  min_accel = min(r["planner_accel_mps2"] for r in event_rows)

  physics = (
    f"first-visible opening lead at t={row['t_s']:.2f}s, raw/published vRel "
    f"{row['lead_one_raw_v_rel_mps']:+.3f}/{row['lead_one_published_v_rel_mps']:+.3f} m/s, "
    f"dRel {row['lead_one_published_d_rel_m']:.2f} m: aTarget "
    f"{prev['planner_accel_mps2']:+.3f} -> {row['planner_accel_mps2']:+.3f}, "
    f"minimum {min_accel:+.3f}, worst 50 ms drop {worst_drop:+.3f} m/s^2 "
    f"(pre-fix publication frame +1.499 -> -0.702)"
  )
  assert row["planner_handoff_limit_debug"]["opening_cap_appeared"] is True, physics
  assert row["planner_handoff_limit_debug"]["clipped"] is True, physics
  assert min_accel >= 0.0, physics
  assert worst_drop >= -MAX_BENIGN_ONE_FRAME_DROP_MPS2, physics
  assert roll_row["planner_accel_mps2"] < -0.5, physics
  assert roll_row["planner_accel_mps2"] - roll_prev["planner_accel_mps2"] < -1.0, physics
  assert row["planner_accel_mps2"] > roll_row["planner_accel_mps2"] + 1.0, physics


def test_dangerous_closing_lead_bypasses_comfort_limit() -> None:
  """Matched reveal timing; only dangerous lead kinematics/gap differ."""
  result = _run_closing()
  prev, row = _first_published_edge(result)

  assert row["lead_one_raw_v_rel_mps"] < -5.0
  assert row["lead_one_published_v_rel_mps"] < -5.0
  assert row["planner_handoff_limit_debug"]["down_bypassed"] is True
  assert row["planner_handoff_limit_debug"]["bypass_reason"] in ("requested_decel", "kinematic")
  assert row["planner_accel_mps2"] <= MIN_DANGEROUS_BRAKE_MPS2
  assert row["planner_accel_mps2"] - prev["planner_accel_mps2"] < -1.0


def test_decelerating_lead_matches_unlimited_braking_twin() -> None:
  """Same 24 m reveal as the opening case; genuine aLeadK braking differs."""
  current = _run_decelerating()
  rollback = _run_decelerating_rollback()
  prev, row = _first_published_edge(current)
  roll_prev, roll_row = _first_published_edge(rollback)

  assert row["planner_source"] == roll_row["planner_source"] == "lead0"
  assert row["lead_one_published_v_rel_mps"] < -1.0
  assert row["lead_one_published_a_lead_k_mps2"] <= DECEL_LEAD_ACCEL_MPS2
  assert row["planner_handoff_limit_debug"]["down_bypassed"] is True
  assert row["planner_handoff_limit_debug"]["bypass_reason"] == "lead_decel"
  assert row["planner_handoff_limit_debug"]["clipped"] is False
  assert row["planner_accel_mps2"] < -0.5
  assert row["planner_accel_mps2"] == pytest.approx(roll_row["planner_accel_mps2"], abs=1e-9)
  assert row["planner_accel_mps2"] - prev["planner_accel_mps2"] == pytest.approx(
    roll_row["planner_accel_mps2"] - roll_prev["planner_accel_mps2"], abs=1e-9)
