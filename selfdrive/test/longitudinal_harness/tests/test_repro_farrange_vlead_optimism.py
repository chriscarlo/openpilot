"""Repro: CD8 (SEV-3) - far-range stopped-traffic vLead OPTIMISM concentrates a
late hard stop, with no clamp on the published-vLead the kinematic stopping-need
term reads.

Field evidence (road_forensics.json, forensics 200-13 EDGE2 - the successful-but-
late FCW stop the USER specifically asked to spread earlier):

  - first raw sighting prob 0.05 at 78.8 m, 2.8 s to a solid track;
  - published vLead ran ~+4 m/s OPTIMISTIC (high) vs position-derived truth
    during the 43-48 s window;
  - aTarget plateaued -1.5 for 2 s then dove to -4.76 with carOutput saturating
    -5.50 for 1.6 s;
  - 42% of the kinetic energy was shed in the LAST 3 s vs a -1.91 m/s^2
    constant-decel ideal.

Root cause: while a far, newly-acquired lead is still stopping/slow, the model's
published vLead is biased high, so the kinematic stopping-need term
(compute_lead_stopping_need_decel, which uses vLead to size the closure) computes
a much smaller required decel than reality and the cruise->lead handoff / braking
starts late, forcing a concentrated hard stop.

Scenario synthesis (road-derived, tuned so the constant-decel-from-first-solid-
sight ideal is genuinely spreadable - ~1.4 m/s^2, in the road's -1.91 band): ego
16 m/s approaching a STOPPED (~1 m/s) lead first visible at 105 m, with a scripted
modelProb ramp (0.05 -> 0.90, crossing the 0.6 Schmitt-enter band ~0.65 s in while
the true gap is still ~95 m) and the published/raw vLead biased HIGH early (a +5
m/s RAW injection, EMA-attenuated by the real tracker to a ~+4 m/s PUBLISHED
optimism - the road figure), decaying to truth by ~52 m (0.5 * reveal). The bias
is injected via LeadDirective.v_lead_bias_mps, which perturbs ONLY the raw
measured vRel the
tracker sees (leadsV3.v = model_v_ego + vRel, radard_stage.py _fill_lead_v3
entry.v) - exactly as measured_d_rel_bias_m perturbs the raw x - so the REAL
radard ModelLeadTracker EMAs the optimistic velocity into the published vLead the
MPC's stopping-need trigger extrapolates with. Ground-truth kinematics (the plant
lead speed) are untouched, so the position-derived truth the fix recovers really
IS slower than the published optimism.

CD8 desired behavior (task spec, road-derived): on a far, genuinely-stopped-lead
approach whose constant-decel-from-first-solid-sight ideal is spreadable, the mean
decel over the FIRST HALF of the approach must reach >= 60% of that ideal (the
brake starts early, not deferred) AND the peak decel must stay shallow
(<= -3.5 m/s^2 - the stop is SPREAD, not concentrated into a late slam). The
optimistic published vLead defeats both today: the stopping-need handoff engages
late so the first half coasts and a hard concentrated stop follows.

Full tici-fidelity loop: device controller mode + the REAL radard ModelLeadTracker
perception stage, noise off to isolate the vLead-optimism mechanism.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 16.0
EGO_V0_MPS = 16.0                 # road ego ~16 m/s onto the far stopped queue
CRUISE_SPEED_MPS = 30.0           # cruise pull (lead is what governs)
REVEAL_GAP_M = 105.0             # first visible far (road: prob 0.05 at 78.8 m)
LEAD_V_MPS = 1.0                 # ~1 m/s (task: stopped or ~1 m/s)
MPC_STOP_DISTANCE_M = STOP_DISTANCE

# modelProb ramp (road: 0.05 -> solid over 2.8 s). Crosses the 0.6 Schmitt-enter
# band ~0.65 s in, while the true gap is still ~95 m, so the lead is latched with
# room to spread; the pathology is the optimism DELAYING the brake, not a late
# latch.
PROB_RAMP_S = 1.0
PROB_LO = 0.05
PROB_HI = 0.90

# Published/raw vLead biased HIGH early (road ~+4 m/s vs position-derived truth),
# decaying linearly to truth by 0.5 * reveal (~52 m). The +5 m/s RAW injection is
# EMA-attenuated by the real radard ModelLeadTracker vRel filter to a ~+4 m/s
# PUBLISHED optimism at the tracker (measured), matching the road figure.
VLEAD_BIAS_MPS = 5.0
BIAS_DECAY_GAP_M = REVEAL_GAP_M * 0.50

# Behavioral bounds (task spec, road-derived). The stop must be SPREAD: the mean
# decel over the first half of the approach reaches a good fraction of the
# constant-decel-from-first-solid-sight ideal, and the peak stays shallow.
FIRST_HALF_MIN_RATIO = 0.60      # first-half mean decel >= 60% of the ideal
MAX_PEAK_DECEL_MPS2 = -3.5       # peak decel must stay >= -3.5 (spread, not slam)

MOVING_V_MPS = 2.0
WELL_BEFORE_STOP_GAP_M = 7.0


def _prob(t_s: float) -> float:
  if t_s <= 0.0:
    return PROB_LO
  if t_s >= PROB_RAMP_S:
    return PROB_HI
  return PROB_LO + (PROB_HI - PROB_LO) * (t_s / PROB_RAMP_S)


def _bias(gap_m: float) -> float:
  if gap_m >= REVEAL_GAP_M:
    return VLEAD_BIAS_MPS
  if gap_m <= BIAS_DECAY_GAP_M:
    return 0.0
  return VLEAD_BIAS_MPS * (gap_m - BIAS_DECAY_GAP_M) / (REVEAL_GAP_M - BIAS_DECAY_GAP_M)


def _build_steps(vlead_bias: bool = True) -> list[StepInput]:
  steps: list[StepInput] = []
  # Approximate the true gap for the bias schedule (ego closes at ~EGO-LEAD, lead
  # ~stopped); the schedule only needs to hold the bias high while far and fade it
  # by ~58 m, which this open-loop estimate does faithfully.
  gap = REVEAL_GAP_M
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    lead = LeadDirective(
      status=True,
      v_lead_mps=LEAD_V_MPS,
      model_prob_target=_prob(t_s),
      d_rel_override_m=REVEAL_GAP_M if i == 0 else None,
      v_lead_bias_mps=(_bias(gap) if vlead_bias else 0.0),
      acquisition_reset=(i == 0),
    )
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead,
                           note="cd8 far-range stopped-traffic vLead optimism"))
    gap = max(0.0, gap - (EGO_V0_MPS - LEAD_V_MPS) * DT_MDL)
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config()


@functools.lru_cache(maxsize=1)
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name="farrange_vlead_optimism",
    steps=_build_steps(vlead_bias=True),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


# CD8 fix knob: the far-range published-vLead optimism clamp in radard. Range =
# 55 (shipped default, arms the clamp) vs 1e9 (rollback sentinel: range
# unreachable, clamp disabled -> the pre-fix optimistic publish is restored).
FIX_CLAMP_RANGE_M = 55.0
ROLLBACK_CLAMP_RANGE_M = 1e9


@functools.lru_cache(maxsize=2)
def _vehicle_config_clamp(clamp_range_m: float):
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.LeadVLeadOptimismClampRangeM": f"{clamp_range_m:g}",
  })


@functools.lru_cache(maxsize=2)
def _run_clamp(clamp_range_m: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config_clamp(clamp_range_m),
    scenario_name=f"farrange_vlead_optimism_clamp_{clamp_range_m:g}",
    steps=_build_steps(vlead_bias=True),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  # 100 Hz control trace; the planner updates every DT_MDL (5 control ticks).
  return result.trace[::5]


def _first_solid_sight(result: SimulationResult) -> dict:
  # The perception event: the first frame where radard latches and publishes the
  # lead (prob crosses the Schmitt-enter band). This is BEFORE the planner starts
  # braking and is identical across the fix and its rollback (the clamp only
  # changes the published vLead, not the dRel/prob latch), so it is a stable
  # run-independent reference for the constant-decel ideal.
  return next(row for row in _planner_rows(result)
              if row["lead_one_published_d_rel_m"] is not None
              and row["lead_one_true_d_rel_m"] is not None)


def _constant_decel_ideal(result: SimulationResult) -> float:
  solid = _first_solid_sight(result)
  gap = solid["lead_one_true_d_rel_m"]
  v = solid["v_ego_true_mps"]
  return v ** 2 / (2.0 * max(gap - MPC_STOP_DISTANCE_M, 1e-3))


def _approach_rows(result: SimulationResult) -> list[dict]:
  # The braking approach: from first solid sight (the perception latch, when a
  # human/plan would begin shaping the stop) until near the stop point, while
  # still moving. Starting at the latch (NOT the earlier cruise-accel phase)
  # keeps the first-half mean a faithful measure of how early the STOP is shaped.
  solid_t = _first_solid_sight(result)["t_s"]
  return [row for row in _planner_rows(result)
          if row["t_s"] >= solid_t
          and row["v_ego_true_mps"] > MOVING_V_MPS
          and row["lead_one_true_d_rel_m"] is not None
          and row["lead_one_true_d_rel_m"] > WELL_BEFORE_STOP_GAP_M]


def _first_half_rows(result: SimulationResult) -> list[dict]:
  # First half of the approach BY DISTANCE covered (true gap from solid-sight to
  # the last moving frame), so the split tracks the physical approach geometry
  # rather than the frame count (which the fix's earlier/slower profile would
  # otherwise skew). A deferred stop coasts through this first half; a spread
  # stop is already braking.
  approach = _approach_rows(result)
  if not approach:
    return []
  gap_hi = approach[0]["lead_one_true_d_rel_m"]
  gap_lo = approach[-1]["lead_one_true_d_rel_m"]
  gap_mid = 0.5 * (gap_hi + gap_lo)
  return [row for row in approach if row["lead_one_true_d_rel_m"] >= gap_mid]


def _first_half_mean_decel(result: SimulationResult) -> float:
  half = _first_half_rows(result)
  if not half:
    return 0.0
  # Mean BRAKING magnitude (only the decel part) over the first half of the
  # approach: a deferred stop coasts here (near 0), a spread stop already brakes.
  return sum(-min(0.0, row["planner_accel_mps2"]) for row in half) / len(half)


def _peak_decel(result: SimulationResult) -> float:
  approach = _approach_rows(result)
  return min((row["planner_accel_mps2"] for row in approach), default=0.0)


def _published_vlead_optimism_m(result: SimulationResult) -> float:
  # Max amount the published vLead ran ABOVE the true lead speed while far and
  # latched (the injected optimism reaching the tracker).
  worst = 0.0
  for row in _planner_rows(result):
    pub = row["lead_one_published_v_lead_mps"]
    true_v = row["active_lead_speed_mps"]
    gap = row["lead_one_true_d_rel_m"]
    if pub is None or true_v is None or gap is None or gap < BIAS_DECAY_GAP_M:
      continue
    worst = max(worst, pub - true_v)
  return worst


def test_farrange_vlead_optimism_scenario_wiring() -> None:
  result = _run()

  # Tici-fidelity loop resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"

  # Scenario-validity guard #1: the lead is genuinely STOPPED/slow and the
  # approach is a real closing (ego >> lead), so this is a true stopping approach.
  moving = [row for row in _planner_rows(result)
            if row["v_ego_true_mps"] > MOVING_V_MPS and row["active_lead_speed_mps"] is not None]
  assert moving, "no moving-approach frames"
  assert all(row["active_lead_speed_mps"] <= LEAD_V_MPS + 0.5 for row in moving), (
    "lead is not genuinely stopped/slow; scenario is not the CD8 stopped-traffic case")
  assert EGO_V0_MPS >= LEAD_V_MPS + 10.0, "approach is not a real closing (ego >> lead)"

  # Scenario-validity guard #2: the lead is latched from far range (a real min
  # true gap is computed and never collapses to a collision), so any late/hard
  # stop is a brake-shaping defect, not an unavoidable emergency.
  solid = _first_solid_sight(result)
  assert solid["lead_one_true_d_rel_m"] > 60.0, (
    f"lead first latched at only {solid['lead_one_true_d_rel_m']:.1f} m; not a far-range acquisition")
  assert result.summary["minTrueGapM"] > 0.0, "ego collided with the lead"

  # Scenario-validity guard #3: the constant-decel-from-first-solid-sight ideal is
  # genuinely SPREADABLE (a calm human decel), so a concentrated hard stop is a
  # defect, not physics. (Road ideal -1.91 m/s^2.)
  ideal = _constant_decel_ideal(result)
  assert 0.8 <= ideal <= 2.5, (
    f"constant-decel ideal {ideal:.2f} m/s^2 is not in the spreadable band; retune the scenario")

  # Scenario-validity guard #4: the injected vLead optimism actually reaches the
  # published tracker (published vLead runs materially ABOVE the true lead speed
  # while far), so the mechanism under test is genuinely exercised.
  optimism = _published_vlead_optimism_m(result)
  assert optimism >= 1.5, (
    f"published vLead ran only {optimism:.2f} m/s above truth; the injected optimism did not reach the tracker")

  # Ego really approaches and stops behind the lead.
  stop_row = next((row for row in result.trace if row["t_s"] > 1.0 and row["v_ego_true_mps"] < 0.05), None)
  assert stop_row is not None, "ego never stopped inside the scenario window"


@pytest.mark.xfail(strict=True, reason="CD8: far-range published-vLead optimism clamp not yet implemented (fix pending)")
def test_farrange_stopped_traffic_stop_is_spread() -> None:
  result = _run()

  ideal = _constant_decel_ideal(result)
  fh_mean = _first_half_mean_decel(result)
  peak = _peak_decel(result)
  ratio = fh_mean / ideal if ideal > 0.0 else 0.0
  solid = _first_solid_sight(result)

  first_half_ok = ratio >= FIRST_HALF_MIN_RATIO
  peak_ok = peak >= MAX_PEAK_DECEL_MPS2

  physics = (
    f"far-range stopped-traffic approach (ego {EGO_V0_MPS} m/s, ~stopped lead first visible at "
    f"{REVEAL_GAP_M} m, published vLead biased +{VLEAD_BIAS_MPS} m/s early):\n"
    f"  first solid sight: t={solid['t_s']:.2f}s true gap {solid['lead_one_true_d_rel_m']:.1f} m "
    f"v_ego {solid['v_ego_true_mps']:.2f} m/s\n"
    f"  constant-decel-from-first-solid-sight ideal: {ideal:.2f} m/s^2 (road -1.91)\n"
    f"  first-half mean decel: {fh_mean:.2f} m/s^2 = {ratio:.0%} of ideal "
    f"(bound >= {FIRST_HALF_MIN_RATIO:.0%})\n"
    f"  peak decel during approach: {peak:.2f} m/s^2 (bound >= {MAX_PEAK_DECEL_MPS2}; road dove to -4.76)\n"
    f"  min true gap: {result.summary['minTrueGapM']:.2f} m"
  )
  assert first_half_ok and peak_ok, physics


@pytest.mark.xfail(strict=True, reason="CD8: LeadVLeadOptimismClampRangeM clamp not yet implemented (fix pending)")
def test_vlead_optimism_clamp_knob_fix_vs_rollback() -> None:
  """CD8 fix-knob oracle (the NEW threshold's rollback sentinel).

  The SAFETY RULE requires EVERY new threshold to have an oracled rollback knob.
  This test exercises the CD8 clamp's own knob directly: LeadVLeadOptimismClampRangeM,
  the far range beyond which the published-vLead optimism clamp arms.

    range = 55 (the fix, shipped default): while the far, stopping lead's raw
      vLead declines monotonically beyond 55 m, the published vLead is pulled to
      the position-derived truth (min only - always slower / more urgent), so the
      stopping-need handoff engages earlier and the brake SPREADS: the first-half
      mean decel reaches a good fraction of the ideal and the peak stays shallow.
    range = 1e9 (rollback sentinel): the far range is unreachable, so the clamp is
      disabled and the pre-fix optimistic published vLead is restored - the road
      pathology reappears (the first half coasts and a concentrated hard stop
      follows, peak past the comfort bound).

  The injected true kinematics are identical in both twins; only the clamp range
  differs, so any behavioral divergence is attributable to the CD8 fix alone."""
  fix = _run_clamp(FIX_CLAMP_RANGE_M)
  rollback = _run_clamp(ROLLBACK_CLAMP_RANGE_M)

  # Matched twins: identical true lead kinematics (only the clamp range differs;
  # the injected directives + noise are identical).
  for r_fix, r_roll in zip(fix.trace, rollback.trace, strict=True):
    assert r_fix["active_lead_speed_mps"] == pytest.approx(r_roll["active_lead_speed_mps"], abs=1e-9)

  # The perception latch (first solid sight) is identical across the twins - the
  # clamp changes only the published vLead, not the dRel/prob latch - so the
  # constant-decel ideal is a shared reference.
  ideal_fix = _constant_decel_ideal(fix)
  ideal_roll = _constant_decel_ideal(rollback)

  fix_ratio = _first_half_mean_decel(fix) / ideal_fix if ideal_fix > 0 else 0.0
  roll_ratio = _first_half_mean_decel(rollback) / ideal_roll if ideal_roll > 0 else 0.0
  fix_peak = _peak_decel(fix)
  roll_peak = _peak_decel(rollback)

  physics = (
    f"CD8 vLead-optimism clamp knob (LeadVLeadOptimismClampRangeM), far-range stopped-lead approach:\n"
    f"  range={FIX_CLAMP_RANGE_M} (fix):      first-half mean {fix_ratio:.0%} of ideal "
    f"(bound >= {FIRST_HALF_MIN_RATIO:.0%}), peak {fix_peak:.2f} m/s^2 (bound >= {MAX_PEAK_DECEL_MPS2})\n"
    f"  range={ROLLBACK_CLAMP_RANGE_M:g} (rollback): first-half mean {roll_ratio:.0%} of ideal, "
    f"peak {roll_peak:.2f} m/s^2 (pre-fix pathology; road dove to -4.76 in the last 3 s)"
  )

  # The fix: the stop is spread (first half reaches the ideal fraction, peak shallow).
  assert fix_ratio >= FIRST_HALF_MIN_RATIO, physics
  assert fix_peak >= MAX_PEAK_DECEL_MPS2, physics

  # The rollback sentinel restores the pre-fix pathology: the first half coasts
  # below the ideal fraction AND the peak exceeds the comfort bound.
  assert roll_ratio < FIRST_HALF_MIN_RATIO, physics
  assert roll_peak < MAX_PEAK_DECEL_MPS2, physics

  # And the fix is strictly safer than its own rollback on both headline metrics
  # (more brake early, shallower peak).
  assert fix_ratio > roll_ratio, physics
  assert fix_peak > roll_peak, physics
