"""Repro: FAILED LAUNCH road 205-13 (2026-07-04 Event A) - stop-latch release lag + weak launch.

Field evidence (docs/chauffeur/longitudinal/road_incidents_20260704.md Event A;
full-rate extraction of realdata/00000205--63a5523547--13 rlog, t=0..6.5 s):

Ego stopped behind a stopped lead (published gap 4.15 m, raw model x ~5.7 m),
longControlState=stopping holding the full stopAccel -2.0. The lead visibly
launched at t~1.05 (raw leadsV3 v 0.3 -> 1.4 m/s by 1.5 s, raw a +0.8..+1.9,
prob 1.0 throughout) and departed at ~+1.9 m/s^2. What followed:

  1. RELEASE LAG (~1.2 s of held full brake after real departure). The
     should_release_stop_for_lead_launch chain is three serial gates:
     published pullaway > 0.1 (met ~1.2 s), MPC a_target > 0.0 (met ~1.6 s),
     and the ABSOLUTE dRel >= 5.0 m arming gate - the binding one. The stop
     settled at a published 4.15 m, so the lead had to open 0.85 m of
     PUBLISHED gap (~2.5 m raw, the opening side is slew-capped at 1.2 m/s)
     before the release could even arm: armed 2.16 s, +0.1 s hold, released
     2.26 s. A stop that settles at 5.1 m would arm instantly; one at 3.6 m
     would need 1.4 m. Departure evidence, not an absolute range, must arm it.
  2. WEAK LAUNCH. On release, `starting` outputs the flat CP.startAccel=+1.0
     (ignoring the plan), the EV6 takes ~0.85 s to physically move (brake
     bleed + torque build, co=+1.0 the whole time), and the MPC's jerk-shaped
     ramp from standstill was still only asking +1.02 when the driver gave up
     at t=3.96 - with the lead 12.5 m ahead pulling away at +5.2 m/s. The same
     MPC asked +2.7 two seconds later: the demand exists, the launch window
     never gets it. get_low_speed_launch_follow_max_accel only RAISES THE
     CLIP CEILING (to ~2.3 here) - nothing lifts the demand into it.

Scenario synthesis (road-derived): ego stopped, lead stopped at a true 4.5 m
gap; after a 3.0 s settle (longcontrol fully ramped to stopAccel -2.0, road
held -2.0) the lead launches at a true +1.9 m/s^2 up to 7.5 m/s with the model
reporting accel truthfully (road raw a tracked the launch well). Noise off to
keep the frame-precise release-gate timing deterministic (road prob was 1.0
throughout - this failure has no perception-dropout component). Live-tune
deltas of the drive are seeded (VRelTauS=0.60 also slows the published
pullaway ~0.45 s; EDGE1 cap disabled).

Full tici-fidelity loop: device controller mode + radard perception stage +
device livetune snapshot. Requires the closed_loop cruiseState.standstill
fidelity fix (EV6 CAN-FD op-long hardwires it False - carstate.py) or the
state machine can never leave `stopping` at all.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 12.0
CRUISE_SPEED_MPS = 15.0            # ~34 mph street; road cruise was above the launch window speeds
TRUE_GAP_M = 4.5                   # road: published 4.15 at the stop (raw x ~5.7)
LEAD_MODEL_PROB = 1.0              # road prob 1.0 the entire launch

LAUNCH_T_S = 3.0                   # settle long enough for longcontrol to ramp to stopAccel -2.0
LEAD_LAUNCH_ACCEL_MPS2 = 1.9       # road raw a +0.8..+2.1 through the departure
LEAD_V_CEIL_MPS = 7.5              # road lead reached ~7+ m/s while still tracked

# Onset = first step the TRUE lead speed crosses this (road t~1.05 rel).
ONSET_LEAD_V_MPS = 0.3

# Live-tune deltas in force during the drive (NOT the committed defaults).
DRIVE_LIVETUNE_OVERRIDES = {
  "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.60",
  "Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2": "10.0",
}

# Desired-behavior bounds (road-derived).
MAX_RELEASE_DELAY_S = 0.6          # road: 1.2 s of held -2.0 after real departure
MIN_LAUNCH_PLANNER_ACCEL_MPS2 = 1.2  # road: planner peaked +1.02 before the driver quit
LAUNCH_ACCEL_WITHIN_S = 1.25       # ...measured within this window after release
MIN_V_AT_ONSET_PLUS_3P5_MPS = 2.5  # road (driver-aided!) ~3.0; pre-fix harness must sit below
MIN_TRUE_GAP_M = 3.5               # launch must never eat into the stopped gap
MAX_LAUNCH_PLANNER_ACCEL_MPS2 = 2.6  # comfort containment: below ACCEL_MAX, no slam-launch


def _lead_speed(t_s: float) -> float:
  if t_s < LAUNCH_T_S:
    return 0.0
  return min(LEAD_V_CEIL_MPS, LEAD_LAUNCH_ACCEL_MPS2 * (t_s - LAUNCH_T_S))


def _build_steps() -> list[StepInput]:
  steps = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    v_lead = _lead_speed(t_s)
    launching = LAUNCH_T_S <= t_s and v_lead < LEAD_V_CEIL_MPS
    lead = LeadDirective(
      status=True,
      v_lead_mps=v_lead,
      model_prob_target=LEAD_MODEL_PROB,
      a_lead_k_mps2=LEAD_LAUNCH_ACCEL_MPS2 if launching else 0.0,
      d_rel_override_m=TRUE_GAP_M if i == 0 else None,
      acquisition_reset=i == 0,
    )
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_SPEED_MPS,
      lead_one=lead,
      event="lead_launch" if abs(t_s - LAUNCH_T_S) < (DT_MDL * 0.5) else None,
      note="stopped behind stopped lead" if t_s < LAUNCH_T_S else "lead departs +1.9",
    ))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config(param_overrides=dict(DRIVE_LIVETUNE_OVERRIDES))


@functools.lru_cache(maxsize=1)
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name="stop_launch_release",
    steps=_build_steps(),
    initial_speed_mps=0.0,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _onset_t(trace: list[dict]) -> float | None:
  return next((row["t_s"] for row in trace
               if row["active_lead_speed_mps"] is not None
               and row["active_lead_speed_mps"] >= ONSET_LEAD_V_MPS), None)


def _release_t(trace: list[dict]) -> float | None:
  # First stop-release AFTER the launch (the latch must be engaged pre-launch).
  return next((row["t_s"] for row in trace
               if row["t_s"] >= LAUNCH_T_S and not row["planner_should_stop"]), None)


def _measure(result: SimulationResult) -> dict:
  trace = result.trace
  onset_t = _onset_t(trace)
  release_t = _release_t(trace)
  launch_window = [] if release_t is None else [
    row for row in trace if release_t <= row["t_s"] <= release_t + LAUNCH_ACCEL_WITHIN_S]
  peak_launch_accel = max((row["planner_accel_mps2"] for row in launch_window), default=None)
  v_probe_row = None
  if onset_t is not None:
    v_probe_row = next((row for row in trace if row["t_s"] >= onset_t + 3.5), None)
  post_launch = [row for row in trace if row["t_s"] >= LAUNCH_T_S]
  return {
    "onset_t_s": onset_t,
    "release_t_s": release_t,
    "release_delay_s": None if (onset_t is None or release_t is None) else release_t - onset_t,
    "peak_launch_planner_accel_mps2": peak_launch_accel,
    "v_at_onset_plus_3p5_mps": None if v_probe_row is None else v_probe_row["v_ego_true_mps"],
    "min_true_gap_m": min((row["true_min_gap_m"] for row in post_launch
                           if row["true_min_gap_m"] is not None), default=None),
    "peak_planner_accel_mps2": max(row["planner_accel_mps2"] for row in trace),
    "relatched": any(row["planner_should_stop"] for row in trace
                     if release_t is not None and row["t_s"] > release_t),
  }


def test_stop_launch_scenario_wiring() -> None:
  result = _run()

  # Tici-fidelity loop resolved as intended, with the drive's live-tune deltas.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"

  # The stop actually engages: standstill with the stop latch in and longcontrol
  # in `stopping`, holding the CP stopAccel by the end of the settle (road: -2.0).
  pre_launch = [row for row in result.trace if 2.0 <= row["t_s"] < LAUNCH_T_S]
  assert pre_launch
  assert all(row["v_ego_true_mps"] < 0.05 for row in pre_launch)
  assert all(row["planner_should_stop"] for row in pre_launch)
  assert all(row["longcontrol_state_name"] == "stopping" for row in pre_launch)
  assert min(row["longcontrol_accel_mps2"] for row in pre_launch) <= -1.9

  # The lead is continuously tracked through the stop and launch (road prob 1.0).
  settled = [row for row in result.trace if row["t_s"] >= 1.5]
  assert all(row["lead_one_published_d_rel_m"] is not None for row in settled)

  # The state machine can leave `stopping` at all (EV6 op-long cruiseState
  # standstill fidelity): once the stop latch releases, `starting` must appear.
  release_t = _release_t(result.trace)
  if release_t is not None:
    after = [row for row in result.trace if row["t_s"] >= release_t]
    assert any(row["longcontrol_state_name"] == "starting" for row in after)

  # Comfort containment holds regardless of fix state: no slam-launch.
  assert max(row["planner_accel_mps2"] for row in result.trace) <= MAX_LAUNCH_PLANNER_ACCEL_MPS2

  # Launch safety: the commanded launch never eats into the stopped gap.
  m = _measure(result)
  assert m["min_true_gap_m"] is not None and m["min_true_gap_m"] >= MIN_TRUE_GAP_M


@pytest.mark.xfail(strict=True,
                   reason="road 205-13 Event A: stop latch held ~1.2 s after real lead departure "
                          "(absolute dRel>=5.0 arming gate) and the launch demand never exceeded "
                          "+1.0 while the lead departed at +5 m/s (driver pedaled)")
def test_stop_release_and_launch_track_departing_lead() -> None:
  m = _measure(_run())

  release_ok = m["release_delay_s"] is not None and m["release_delay_s"] <= MAX_RELEASE_DELAY_S
  launch_ok = (m["peak_launch_planner_accel_mps2"] is not None
               and m["peak_launch_planner_accel_mps2"] >= MIN_LAUNCH_PLANNER_ACCEL_MPS2)
  progress_ok = (m["v_at_onset_plus_3p5_mps"] is not None
                 and m["v_at_onset_plus_3p5_mps"] >= MIN_V_AT_ONSET_PLUS_3P5_MPS)
  no_relatch = not m["relatched"]

  physics = (
    f"stopped-lead departure (true +{LEAD_LAUNCH_ACCEL_MPS2} m/s^2 from t={LAUNCH_T_S}s, gap {TRUE_GAP_M} m):\n"
    f"  lead-motion onset t={m['onset_t_s']}s, stop-latch release t={m['release_t_s']}s "
    f"(delay {m['release_delay_s']}s vs bound {MAX_RELEASE_DELAY_S}s; road held -2.0 for ~1.2 s)\n"
    f"  peak planner accel within {LAUNCH_ACCEL_WITHIN_S}s of release: {m['peak_launch_planner_accel_mps2']} "
    f"(floor {MIN_LAUNCH_PLANNER_ACCEL_MPS2}; road peaked +1.02 before the driver pedaled)\n"
    f"  v_ego at onset+3.5s: {m['v_at_onset_plus_3p5_mps']} m/s (floor {MIN_V_AT_ONSET_PLUS_3P5_MPS}; "
    f"road driver-aided ~3.0)\n"
    f"  relatched after release: {m['relatched']}, min true gap {m['min_true_gap_m']} m"
  )
  assert release_ok and launch_ok and progress_ok and no_relatch, physics
