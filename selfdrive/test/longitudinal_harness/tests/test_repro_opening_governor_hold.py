"""Regression: EV6 model-lead opening truth must not chatter at publication.

The 2026-07-13 segment-6 freeway capture held one stable lead track while the
model's position stream opened by roughly 1.3 m/s and its velocity stream still
reported a mild close.  The stateless opening governor repeatedly published
exact parity, revoked it on a noisy 0.6 s position window, then re-armed.  That
0 <-> negative-vRel edge drove the lead keep-up floor and final planner target
back and forth even though the MPC source stayed on lead0.

This file keeps the regression at full tici fidelity: synthesized leadsV3 pass
through the real RadarD/ModelLeadTracker before the EV6 planner, LongControl,
Hyundai device controller, and plant.  OpeningGovernorHoldS=0 is the exact
stateless rollback twin.  A matched braking/short-TTC twin proves the comfort
hold yields immediately to current threat evidence.
"""
from __future__ import annotations

import functools
import math

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput


DURATION_S = 12.0
SETTLE_S = 2.0
EGO_V0_MPS = 31.3
LEAD_V_MPS = 31.3
CRUISE_V_MPS = 34.0
INITIAL_GAP_M = 45.0
LEAD_MODEL_PROB = 0.98

# Segment-6 kinematic contradiction, with deterministic model-position noise.
POSITION_OPENING_MPS = 1.3
POSITION_NOISE_AMPLITUDE_M = 0.5
VREL_MEAN_MPS = -0.34
VREL_BAND_MPS = 0.35

FIX_HOLD_S = 1.0
ROLLBACK_HOLD_S = 0.0
PARITY_EPS_MPS = 1e-5
REVERSAL_EXCURSION_MPS2 = 0.025

THREAT_T_S = 2.0
THREAT_DREL_M = 18.0
THREAT_VREL_MPS = -6.0
THREAT_ALEAD_MPS2 = -2.0


def _measured_drel(t_s: float) -> float:
  return (INITIAL_GAP_M + POSITION_OPENING_MPS * t_s +
          POSITION_NOISE_AMPLITUDE_M * math.sin(2.0 * math.pi * 0.4 * t_s) +
          POSITION_NOISE_AMPLITUDE_M * math.sin(2.0 * math.pi * 2.7 * t_s + 0.7))


def _measured_vrel(t_s: float) -> float:
  return VREL_MEAN_MPS + VREL_BAND_MPS * math.sin(2.0 * math.pi * 1.4 * t_s + 0.3)


def _build_chatter_steps() -> list[StepInput]:
  steps: list[StepInput] = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_V_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=LEAD_V_MPS,
        model_prob_target=LEAD_MODEL_PROB,
        d_rel_override_m=INITIAL_GAP_M if i == 0 else None,
        measured_d_rel_m=_measured_drel(t_s),
        measured_v_rel_mps=_measured_vrel(t_s),
        a_lead_k_mps2=0.0,
        acquisition_reset=(i == 0),
      ),
      note="segment-6 opening-position / mildly-closing-velocity contradiction",
    ))
  return steps


def _build_threat_steps() -> list[StepInput]:
  """Arm the opening correction, then present braking at a three-second TTC."""
  duration_s = 4.0
  steps: list[StepInput] = []
  for i in range(int(round(duration_s / DT_MDL))):
    t_s = i * DT_MDL
    if t_s < THREAT_T_S:
      d_rel = INITIAL_GAP_M + 1.0 * t_s
      v_rel = -0.2
      a_lead = 0.0
      v_lead = LEAD_V_MPS
      note = "dense opening proof before threat"
    else:
      threat_age_s = t_s - THREAT_T_S
      d_rel = max(4.0, THREAT_DREL_M + THREAT_VREL_MPS * threat_age_s)
      v_rel = THREAT_VREL_MPS
      a_lead = THREAT_ALEAD_MPS2
      v_lead = LEAD_V_MPS + THREAT_VREL_MPS
      note = "current braking plus short-TTC closure"

    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_V_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=v_lead,
        model_prob_target=LEAD_MODEL_PROB,
        d_rel_override_m=INITIAL_GAP_M if i == 0 else None,
        measured_d_rel_m=d_rel,
        measured_v_rel_mps=v_rel,
        a_lead_k_mps2=a_lead,
        acquisition_reset=(i == 0),
      ),
      note=note,
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _vehicle_config(hold_s: float):
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.OpeningGovernorHoldS": f"{hold_s:g}",
    # Captured device value: velocity disagreement alone did not veto position
    # truth.  The production hold still has non-bypassable current-threat exits.
    "Longitudinal.LiveTune.OpeningGovernorRawClosingVetoMps": "99",
    "Longitudinal.LiveTune.OpeningGovernorTrustDeficitMps": "0.3",
    "Longitudinal.LiveTune.OpeningGovernorMinOpeningMps": "0.2",
    "Longitudinal.LiveTune.ComfortJerkLimitMps3": "0.4",
    "Longitudinal.LiveTune.LeadKeepUpMaxAccel": "0.22",
  })


@functools.lru_cache(maxsize=2)
def _run_chatter(hold_s: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(hold_s),
    scenario_name=f"opening_governor_chatter_hold_{hold_s:g}",
    steps=_build_chatter_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=1,
    perception_filter="auto",
  )


@functools.lru_cache(maxsize=2)
def _run_threat(hold_s: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(hold_s),
    scenario_name=f"opening_governor_threat_hold_{hold_s:g}",
    steps=_build_threat_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=1,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  # Harness traces every 100 Hz control tick; RadarD/planner update at 20 Hz.
  return result.trace[::5]


def _post_settle_rows(result: SimulationResult) -> list[dict]:
  return [row for row in _planner_rows(result) if row["t_s"] >= SETTLE_S]


def _parity_negative_edges(rows: list[dict]) -> int:
  at_parity = [row["lead_one_published_v_rel_mps"] >= -PARITY_EPS_MPS for row in rows]
  return sum(prev != cur for prev, cur in zip(at_parity, at_parity[1:], strict=False))


def _accel_direction_reversals(rows: list[dict]) -> int:
  """Count direction changes after a felt-size acceleration excursion."""
  values = [row["planner_accel_mps2"] for row in rows]
  if len(values) < 2:
    return 0

  reversals = 0
  last_extreme = values[0]
  direction = 0
  for i in range(1, len(values)):
    delta = values[i] - values[i - 1]
    if abs(delta) < 1e-6:
      continue
    next_direction = 1 if delta > 0.0 else -1
    if (direction != 0 and next_direction != direction and
        abs(values[i - 1] - last_extreme) >= REVERSAL_EXCURSION_MPS2):
      reversals += 1
      last_extreme = values[i - 1]
    direction = next_direction
  return reversals


def _assert_full_path_and_stable_lead(result: SimulationResult) -> list[dict]:
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["radarUnavailable"] is True

  rows = _post_settle_rows(result)
  assert rows
  assert all(row["planner_source"] == "lead0" for row in rows)
  assert not any(prev["planner_source"] != cur["planner_source"]
                 for prev, cur in zip(rows, rows[1:], strict=False))
  assert all(abs(row["lead_one_raw_a_lead_k_mps2"]) <= 1e-9 for row in rows)
  assert result.summary["minTrueGapM"] >= 30.0
  return rows


def test_opening_governor_hold_reduces_parity_chatter_and_accel_reversals() -> None:
  fix_rows = _assert_full_path_and_stable_lead(_run_chatter(FIX_HOLD_S))
  rollback_rows = _assert_full_path_and_stable_lead(_run_chatter(ROLLBACK_HOLD_S))

  # The opening correction is conservative: it may relax false closure only to
  # parity, never fabricate a pulling-away lead.
  assert max(row["lead_one_published_v_rel_mps"] for row in fix_rows) <= PARITY_EPS_MPS
  assert max(row["lead_one_published_v_rel_mps"] for row in rollback_rows) <= PARITY_EPS_MPS

  fix_edges = _parity_negative_edges(fix_rows)
  rollback_edges = _parity_negative_edges(rollback_rows)
  fix_reversals = _accel_direction_reversals(fix_rows)
  rollback_reversals = _accel_direction_reversals(rollback_rows)

  physics = (
    "segment-6 position/velocity contradiction through real RadarD, stable lead0:\n"
    f"  hold={FIX_HOLD_S:g}s: parity/negative edges {fix_edges}, "
    f"planner accel reversals {fix_reversals}\n"
    f"  hold={ROLLBACK_HOLD_S:g}s rollback: parity/negative edges {rollback_edges}, "
    f"planner accel reversals {rollback_reversals}"
  )

  # Scenario-validity guards ensure a future fixture drift cannot make both
  # twins trivially smooth and pass the comparison.
  assert rollback_edges >= 8, physics
  assert rollback_reversals >= 4, physics

  # Material, not exact-count, expectations: tolerate small downstream changes
  # while requiring the stateful hold to remove most publication chatter and at
  # least two felt-size planner direction reversals.
  assert fix_edges <= max(2, rollback_edges // 2), physics
  assert fix_reversals <= rollback_reversals - 2, physics


def test_opening_governor_hold_yields_to_current_braking_and_short_ttc() -> None:
  fix_rows = _planner_rows(_run_threat(FIX_HOLD_S))
  rollback_rows = _planner_rows(_run_threat(ROLLBACK_HOLD_S))
  rollback_by_t = {row["t_s"]: row for row in rollback_rows}

  threat_row = next(row for row in fix_rows if row["t_s"] >= THREAT_T_S - 1e-9)
  rollback_threat = rollback_by_t[threat_row["t_s"]]
  threat_i = fix_rows.index(threat_row)
  pre_threat = fix_rows[threat_i - 1]

  # Prove the safety event hits the same established track while the opening
  # correction was active on the immediately preceding frame.
  assert pre_threat["lead_one_radard_debug"]["opening_relax_vrel_mps"] is not None
  assert pre_threat["lead_one_radard_debug"]["track_id"] == threat_row["lead_one_radard_debug"]["track_id"]
  assert threat_row["lead_one_raw_d_rel_m"] == pytest.approx(THREAT_DREL_M, abs=1e-6)
  assert threat_row["lead_one_raw_v_rel_mps"] == pytest.approx(THREAT_VREL_MPS, abs=1e-6)
  assert threat_row["lead_one_raw_a_lead_k_mps2"] == pytest.approx(THREAT_ALEAD_MPS2, abs=1e-6)
  assert THREAT_DREL_M / abs(THREAT_VREL_MPS) <= 3.0

  # Current braking/short-TTC evidence is a hard same-frame exit. The held
  # comfort state must not make RadarD or the planner less urgent than the exact
  # stateless rollback twin.
  assert threat_row["lead_one_radard_debug"]["opening_relax_vrel_mps"] is None
  assert threat_row["lead_one_radard_debug"]["closing_governor_active"] is True
  assert threat_row["lead_one_published_v_rel_mps"] < -1.0
  assert threat_row["lead_one_published_v_rel_mps"] <= rollback_threat["lead_one_published_v_rel_mps"] + 1e-6
  assert threat_row["planner_accel_mps2"] < 0.0
  assert threat_row["planner_accel_mps2"] <= rollback_threat["planner_accel_mps2"] + 0.02

  # The hold may not reappear while the threat continues.
  for row in fix_rows[threat_i:threat_i + 5]:
    assert row["lead_one_radard_debug"]["opening_relax_vrel_mps"] is None

  # Matched-twin sanity: both runs use identical raw threat measurements.
  for row in fix_rows[threat_i:threat_i + 5]:
    rollback_row = rollback_by_t[row["t_s"]]
    assert row["lead_one_raw_d_rel_m"] == pytest.approx(rollback_row["lead_one_raw_d_rel_m"], abs=1e-9)
    assert row["lead_one_raw_v_rel_mps"] == pytest.approx(rollback_row["lead_one_raw_v_rel_mps"], abs=1e-9)
    assert row["lead_one_raw_a_lead_k_mps2"] == pytest.approx(rollback_row["lead_one_raw_a_lead_k_mps2"], abs=1e-9)
