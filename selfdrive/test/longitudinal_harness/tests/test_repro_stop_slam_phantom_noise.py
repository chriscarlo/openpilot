"""Repro: phantom close-lead latch slams the brakes during a calm noisy approach.

User report: sudden unexpected brake slam well short of the natural stop point
while calmly approaching a stopped lead. Under the repo's own calibrated EV6
noise model (ev6_measured, NoiseSeeds.from_base(99)), a single inward dRel
outlier during the approach trips radard ModelLeadTracker's fast-close gates
(selfdrive/controls/radard.py _fast_closing_supported: the TTC gate is computed
from the raw outlier itself, and strong_closing is true for any approach faster
than 2.5 m/s), is adopted at alpha >= 0.65 in one 50 ms frame, and the
opening-side recovery is capped at ModelLeadFilterOpenSlewMaxMps against a
prediction still closing at vRel — so the published dRel collapses to ~1.8 m
while the true gap is ~12.8 m. The planner steps -0.93 -> -3.97 m/s^2 within
0.3 s at ~18 m true gap and the car full-brakes to a stop several meters short.

Measured on this scenario (before fix): published dRel 1.85 m at t=4.5 s with
true gap 12.83 m; planner window step -0.93 -> -3.97 m/s^2 at t=3.6-3.9 s with
true gap 18.2 m and v_ego 6.03 m/s.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import NoiseSeeds, resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 12.0
EGO_V0_MPS = 6.0
CRUISE_SPEED_MPS = 8.0
INITIAL_GAP_M = 40.0
LEAD_MODEL_PROB = 0.98
NOISE_BASE_SEED = 99

# Phantom signature: the published gap must never collapse far below the truth.
PHANTOM_PUBLISHED_M = 2.0
PHANTOM_TRUE_GAP_M = 8.0
# Slam signature: one <= 0.3 s planner step this deep, starting from an
# established calm decel, while the true gap is still comfortable.
SLAM_WINDOW_S = 0.3
SLAM_STEP_MPS2 = 2.0
CALM_BASELINE_MPS2 = -1.5
SLAM_TRUE_GAP_M = 10.0
MOVING_V_MPS = 2.0
# Safety floor: a fix must not trade the phantom slam for a real near-miss.
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
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="stop_slam_phantom_noise_seed99",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    noise_seeds=NoiseSeeds.from_base(NOISE_BASE_SEED),
    seed=NOISE_BASE_SEED,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  return result.trace[::5]


def _phantom_collapse(result: SimulationResult) -> dict | None:
  for row in _planner_rows(result):
    pub = row["lead_one_published_d_rel_m"]
    true_gap = row["lead_one_true_d_rel_m"]
    if pub is not None and true_gap is not None and pub < PHANTOM_PUBLISHED_M and true_gap > PHANTOM_TRUE_GAP_M:
      return row
  return None


def _slam_step(result: SimulationResult) -> tuple[dict, dict] | None:
  rows = _planner_rows(result)
  for i, row in enumerate(rows):
    a0 = row["planner_accel_mps2"]
    gap = row["lead_one_true_d_rel_m"]
    if a0 < CALM_BASELINE_MPS2 or row["v_ego_true_mps"] <= MOVING_V_MPS or gap is None or gap <= SLAM_TRUE_GAP_M:
      continue
    for j in range(i + 1, len(rows)):
      if rows[j]["t_s"] - row["t_s"] > SLAM_WINDOW_S + 1e-6:
        break
      if a0 - rows[j]["planner_accel_mps2"] >= SLAM_STEP_MPS2:
        return row, rows[j]
  return None


def test_noisy_calm_approach_scenario_wiring() -> None:
  result = _run()

  # Tici-fidelity loop + calibrated noise resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "ev6_measured"
  assert result.vehicle["noiseSeeds"]["drel"] == NOISE_BASE_SEED

  # True kinematics are calm: the lead is stopped the whole time and the gap
  # closes at cruise speed, nothing in the ground truth demands hard braking.
  lead_speeds = {row["active_lead_speed_mps"] for row in result.trace if row["has_any_lead"]}
  assert lead_speeds == {0.0}

  # The lead is acquired and published before any braking decision.
  published = [row for row in result.trace if row["lead_one_published_d_rel_m"] is not None]
  assert published and published[0]["t_s"] < 1.0

  # A calm planner baseline is established before the step (gentle decel only
  # in the first 3.5 s of the approach).
  early = [row for row in _planner_rows(result) if 1.0 <= row["t_s"] <= 3.5]
  assert min(row["planner_accel_mps2"] for row in early) > -1.6

  # Ego stops without collision (the phantom stops the car early, not late).
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)
  assert stop_row is not None
  assert stop_row["true_min_gap_m"] > 0.0


def test_m2_no_phantom_collapse_and_no_slam_step_under_measured_noise() -> None:
  """GREEN pin of the M2 fix on the exact repro scenario.

  The strict-xfail test below stays xfail on the min-true-gap floor conjunct
  (owned by the M1/M3 stopping chain), so it cannot catch an M2 regression:
  it would stay xfail whether the phantom returns or not. This test enforces
  the two M2 criteria — no phantom collapse and no mid-approach slam step —
  as hard green assertions on the same seed-99 run.
  """
  result = _run()

  phantom = _phantom_collapse(result)
  assert phantom is None, (
    f"phantom collapse returned: t={phantom['t_s']:.2f} "
    f"published={phantom['lead_one_published_d_rel_m']:.2f} m "
    f"true={phantom['lead_one_true_d_rel_m']:.2f} m"
  )

  slam = _slam_step(result)
  assert slam is None, (
    f"slam step returned: {slam[0]['planner_accel_mps2']:+.2f} -> "
    f"{slam[1]['planner_accel_mps2']:+.2f} m/s^2 over "
    f"[{slam[0]['t_s']:.2f}, {slam[1]['t_s']:.2f}] s at true gap "
    f"{slam[0]['lead_one_true_d_rel_m']:.2f} m"
  )


@pytest.mark.xfail(strict=True, reason="Phantom mechanism FIXED (fast-close corroboration + corroborated opening "
                                       "recovery in radard.py ModelLeadTrack): no phantom collapse and no slam "
                                       "step under measured noise. Remaining conjunct: minTrueGapM 3.74 < 4.0 "
                                       "floor, owned by the composed-tree M1/M3 stopping chain — given accurate "
                                       "published gaps it stops 2.9-4.0 m short on 11/14 ev6_measured seeds, and "
                                       "clean no-phantom seeds show the identical short stops with M2 fully "
                                       "disabled (FastCloseConfirmFrames=1, OpenRecoveryMaxEgoMps=0), so do NOT "
                                       "re-diagnose the fast-close outlier adoption; see the stopping-chain "
                                       "residual owner task in docs/chauffeur/live_tunable_params.md")
def test_no_phantom_collapse_slam_under_measured_noise() -> None:
  result = _run()

  phantom = _phantom_collapse(result)
  slam = _slam_step(result)
  min_true_gap = result.summary["minTrueGapM"]
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)
  peak_planner_brake = min(row["planner_accel_mps2"] for row in result.trace)

  no_phantom = phantom is None
  no_slam_step = slam is None
  safe_gap = min_true_gap >= MIN_TRUE_GAP_FLOOR_M

  physics = (
    f"ev6_measured seed {NOISE_BASE_SEED}, ego {EGO_V0_MPS} m/s cruise {CRUISE_SPEED_MPS} m/s, "
    f"stopped lead at {INITIAL_GAP_M} m:\n"
    f"  phantom collapse (published < {PHANTOM_PUBLISHED_M} m while true gap > {PHANTOM_TRUE_GAP_M} m): "
    + ("none" if phantom is None else
       f"t={phantom['t_s']:.2f} published={phantom['lead_one_published_d_rel_m']:.2f} m "
       f"true={phantom['lead_one_true_d_rel_m']:.2f} m v_ego={phantom['v_ego_true_mps']:.2f}") + "\n"
    f"  slam step (>= {SLAM_STEP_MPS2} m/s^2 within {SLAM_WINDOW_S} s from calmer than {CALM_BASELINE_MPS2} "
    f"while true gap > {SLAM_TRUE_GAP_M} m): "
    + ("none" if slam is None else
       f"{slam[0]['planner_accel_mps2']:+.2f} -> {slam[1]['planner_accel_mps2']:+.2f} m/s^2 over "
       f"[{slam[0]['t_s']:.2f}, {slam[1]['t_s']:.2f}] s at true gap {slam[0]['lead_one_true_d_rel_m']:.2f} m "
       f"v_ego {slam[0]['v_ego_true_mps']:.2f} m/s") + "\n"
    f"  peak planner brake: {peak_planner_brake:.2f} m/s^2; min true gap: {min_true_gap:.2f} m "
    f"(floor {MIN_TRUE_GAP_FLOOR_M} m); stop gap: "
    f"{None if stop_row is None else round(stop_row['true_min_gap_m'], 2)} m"
  )
  assert no_phantom and no_slam_step and safe_gap, physics
