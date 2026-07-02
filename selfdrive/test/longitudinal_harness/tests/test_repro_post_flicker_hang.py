"""Repro R7 (docs/chauffeur/longitudinal/test_runtime_gap_audit_20260701.md):

After a brief lead flicker followed by lead departure, ego hangs back with almost
no accel recovery toward set speed. Two planner clamps at the shipped live-tune
values interact:
  - flutter clamp: two MPC source transitions inside FlutterDetectWindowS=1.0
    latch flutter mode and cap +/- slew at FlutterClampJerkMps3=0.12
    (selfdrive/controls/lib/longitudinal_planner.py _apply_flutter_mode_clamp)
  - cruise-reacquire jerk limit: the departure's lead0->cruise transition arms a
    CruiseReacquireJerkWindowS=3.0 window whose slew ceiling is anchored at the
    pre-departure follow accel (~-0.1) and grows at
    CruiseReacquirePosJerkLimit=0.08 m/s^3 (_apply_cruise_reacquire_jerk_limit)
so output_a_target can gain at most ~+0.24 m/s^2 over the whole window.

The audit's designed flicker (a 0.15 s modelProb dip with status held True) can
no longer flip the MPC source on this branch: the MPC lead stabilizer keys only
on lead status and bridges sub-PhantomLeadHoldS dropouts with a phantom lead
(long_mpc.py _stabilize_raw_leads). On the real radar-less EV6 a modelProb
collapse below radard's Schmitt exit drops radarState status entirely, so the
faithful flicker at this harness's direct radarState seam is a status dropout
slightly longer than the 0.8 s phantom hold — that produces the audit's two
source transitions inside the 1.0 s flutter window through branch-runtime code.
"""
from __future__ import annotations

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

# Shipped clamp values, pinned by the device snapshot
# (docs/chauffeur/longitudinal/device_livetune_snapshot_20260701.txt; identical
# to the common/params_keys.h:399-409 defaults).
FLUTTER_DETECT_WINDOW_S = 1.0     # Longitudinal.LiveTune.FlutterDetectWindowS
FLUTTER_DETECT_TRANSITIONS = 2    # Longitudinal.LiveTune.FlutterDetectTransitions
REACQUIRE_POS_JERK_MPS3 = 0.08    # Longitudinal.LiveTune.CruiseReacquirePosJerkLimit
REACQUIRE_WINDOW_S = 3.0          # Longitudinal.LiveTune.CruiseReacquireJerkWindowS
PHANTOM_LEAD_HOLD_S = 0.8         # Longitudinal.LiveTune.PhantomLeadHoldS

FLICKER_START_S = 3.0
# 0.9 s dropout: just past the 0.8 s phantom hold so the source genuinely flips
# lead0->cruise before the lead reappears (a shorter flicker is bridged by the
# phantom lead and never reaches the flutter detector).
FLICKER_END_S = 3.9
DEPARTURE_S = 4.3
DURATION_S = 12.0
CRUISE_SPEED_MPS = 33.0
LEAD_SPEED_MPS = 26.0
INITIAL_GAP_M = 40.0
# True gap when the lead reappears at t=3.9 (ego has closed ~2.5 m on the
# slightly-slower lead by then); keeps the follow situation continuous.
REACQUIRE_GAP_M = 37.5


def _build_flicker_then_departure(duration_s: float = DURATION_S, dt_s: float = DT_MDL) -> list[StepInput]:
  steps: list[StepInput] = []
  for idx in range(int(round(duration_s / dt_s))):
    t_s = idx * dt_s
    flickering = FLICKER_START_S <= t_s < FLICKER_END_S
    if t_s >= DEPARTURE_S or flickering:
      lead = LeadDirective()
      event = None
      if abs(t_s - FLICKER_START_S) < (dt_s * 0.5):
        event = "flicker_dropout"
      elif abs(t_s - DEPARTURE_S) < (dt_s * 0.5):
        event = "lead_departure"
      note = "lead dropped" if flickering else "lead departed"
    else:
      reacquire = abs(t_s - FLICKER_END_S) < (dt_s * 0.5)
      lead = LeadDirective(
        status=True,
        v_lead_mps=LEAD_SPEED_MPS,
        model_prob_target=0.9,
        d_rel_override_m=INITIAL_GAP_M if idx == 0 else (REACQUIRE_GAP_M if reacquire else None),
        acquisition_reset=idx == 0 or reacquire,
      )
      event = "flicker_reacquire" if reacquire else None
      note = "steady follow"
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead, event=event, note=note))
  return steps


@pytest.fixture(scope="module")
def flicker_trace() -> list[dict]:
  # Planner-side mechanism: passthrough loop per the audit's harness_design (the
  # device-mode Hyundai EMA stage would only add lag on top of the same clamps).
  vehicle = resolve_ev6_vehicle_config(controller_mode="passthrough")
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="flicker_then_departure",
    steps=_build_flicker_then_departure(),
    initial_speed_mps=27.0,
    initial_accel_mps2=0.0,
    noise_profile="off",
    seed=42,
  )
  return result.trace


def _source_transitions(trace: list[dict]) -> list[tuple[float, str, str]]:
  transitions = []
  prev = None
  for row in trace:
    src = row["planner_source"]
    if prev is not None and src != prev:
      transitions.append((row["t_s"], prev, src))
    prev = src
  return transitions


def _row_at(trace: list[dict], t_s: float) -> dict:
  return min(trace, key=lambda row: abs(row["t_s"] - t_s))


def test_flicker_latches_flutter_and_departure_arms_reacquire(flicker_trace: list[dict]) -> None:
  # Scenario-validity guard (not the behavioral xfail): prove the trace goes
  # through the audited mechanism — flutter latched by the flicker, reacquire
  # window armed by the departure — so the xfail below cannot rot into failing
  # for an unrelated reason.
  transitions = _source_transitions(flicker_trace)
  flicker_flips = [t for t in transitions if FLICKER_START_S <= t[0] < DEPARTURE_S]
  # lead0->cruise once the phantom hold expires, cruise->lead0 on reacquisition:
  # two transitions inside the shipped 1.0 s flutter window => flutter latches.
  assert len(flicker_flips) >= FLUTTER_DETECT_TRANSITIONS, f"flicker never flipped the MPC source: {transitions}"
  assert flicker_flips[0][1:] == ("lead0", "cruise")
  assert flicker_flips[1][1:] == ("cruise", "lead0")
  assert flicker_flips[1][0] - flicker_flips[0][0] < FLUTTER_DETECT_WINDOW_S

  departure_flips = [t for t in transitions if t[0] >= DEPARTURE_S]
  assert departure_flips, f"lead departure never returned the source to cruise: {transitions}"
  assert departure_flips[0][1:] == ("lead0", "cruise")
  # The phantom lead also delays the cruise handoff itself by ~PhantomLeadHoldS.
  assert departure_flips[0][0] == pytest.approx(DEPARTURE_S + PHANTOM_LEAD_HOLD_S, abs=0.3)


def test_accel_recovers_within_two_seconds_of_lead_departure(flicker_trace: list[dict]) -> None:
  transitions = _source_transitions(flicker_trace)
  post_departure = [row for row in flicker_trace if DEPARTURE_S <= row["t_s"] <= DEPARTURE_S + 2.0]
  max_accel_2s = max(row["planner_accel_mps2"] for row in post_departure)
  min_accel_2s = min(row["planner_accel_mps2"] for row in post_departure)

  accel_at_departure = _row_at(flicker_trace, DEPARTURE_S)["planner_accel_mps2"]
  recovery_rows = [row for row in flicker_trace if row["t_s"] >= DEPARTURE_S and row["planner_accel_mps2"] >= 0.25]
  recovery_t = recovery_rows[0]["t_s"] if recovery_rows else None

  v_at_departure = _row_at(flicker_trace, DEPARTURE_S)["v_ego_true_mps"]
  v_after_3s = _row_at(flicker_trace, DEPARTURE_S + 3.0)["v_ego_true_mps"]

  physics = (
    f"lead departed t={DEPARTURE_S:.2f}s with planner_accel={accel_at_departure:+.3f} m/s^2, "
    f"v_ego={v_at_departure:.3f} m/s, cruise set {CRUISE_SPEED_MPS:.1f} m/s; "
    f"planner_accel over the next 2.0 s stayed in [{min_accel_2s:+.3f}, {max_accel_2s:+.3f}] m/s^2, "
    f"first reached +0.25 m/s^2 at t={recovery_t if recovery_t is not None else '>end'} s "
    f"(expected within {DEPARTURE_S + 2.0:.1f} s); "
    f"v_ego 3.0 s after departure = {v_after_3s:.3f} m/s (sagged {v_after_3s - v_at_departure:+.3f}); "
    f"source transitions: {[(round(t, 2), a, b) for t, a, b in transitions]}"
  )

  # Audit R7 assertion: within 2 s of the lead departing, the planner must be
  # commanding meaningful accel toward the 33 m/s set speed. Guarded by the
  # CruiseReacquireJerkRamp escalation of the reacquire jerk allowance — with a
  # fixed 0.08 m/s^3 limit the ceiling stays anchored at the pre-departure
  # follow accel and cannot reach +0.25 inside the 3.0 s window.
  assert max_accel_2s >= 0.25, f"post-departure accel hang: {physics}"
  # Secondary: no continued speed sag toward a lead that no longer exists.
  assert v_after_3s >= v_at_departure, f"ego kept slowing after the lead departed: {physics}"
