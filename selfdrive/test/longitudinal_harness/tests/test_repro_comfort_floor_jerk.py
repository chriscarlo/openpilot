"""Repro: CD7 (SEV-3) - MPC snaps to the comfort floor in one frame under a
steady single-source follow, with NO anti-jerk envelope on the planner output.

Field evidence (road_forensics.json, forensics 200-10 / 200-9 tap1 / 201-9):

  200-10: aTarget stepped -0.31 -> -1.00 in 0.15 s (~4.6 m/s^3) with ZERO source
    flips, after ~8 s of vision vRel noise (a -0.5..-2.4 m/s band, aLeadK ~0)
    masking a slow mean close. Nothing hazardous was happening; the MPC QP output
    simply step-changed hard on a single noisy frame.
  200-9 tap1: a 0.86 m/s^2 aTarget sign reversal in 0.8 s with an unchanged track
    and benign kinematics.
  201-9: two 50 ms self-correcting aTarget spikes (-1.2 delta) mid-cruise.

This is the audit's "no closed-loop test asserts any jerk/safety envelope": the
MPC QP output can step-change hard on a single noisy frame even when nothing
hazardous is happening. It is part of the felt THROTTLE_BLIP_JERK / VACILLATION.

Scenario synthesis (all numbers road-derived). A near-constant-speed lead at
30 m/s in a steady single-source lead0 follow at target headway, ego 30 m/s,
run 60 s. The perception vRel carries an injected +/-1.1 m/s band around a small
mean close (-0.3 m/s) - faithful to the road's -0.5..-2.4 m/s vRel band that
masked the slow mean close - on TOP of the ev6_measured distance-banded dRel
noise (the real EV6 lead-noise characterization). aLeadK is ~0 the whole run.
Shipped device costs (ObstacleCost=2.0, AccelChangeCost=400, UseKalmanDRelFilter=1)
are the seeded device-livetune snapshot the tici-fidelity harness loads by default.

The injection is faithful to the device pipeline: the harness raw lead's measured
vRel becomes leadsV3.v = v_ego + vRel (radard_stage.py _fill_lead_v3 entry.v),
which the REAL radard ModelLeadTracker EMAs into the published vLeadK/vRel the MPC
extrapolates with - i.e. exactly the noisy vRel entering the real Schmitt-latched
tracker, with no synthetic short-circuit.

CD7 desired behavior (task spec, road-derived): ABSENT any hazard/urgency gate
(no fast-close / short-TTC / FCW, benign kinematics), the peak one-frame |aTarget|
jerk over the run must stay under a live-tunable comfort bound (~1.0 m/s^3) and
the count of >=0.5 m/s^2-amplitude aTarget sign reversals per 60 s must stay small.
Real braking under ANY hazard signal must fully BYPASS the envelope (validated by
the genuine-threat / reclaim suites), so this bound is a pure comfort envelope on
benign single-source follow. This repro is strict-xfail until that envelope lands.

Full tici-fidelity loop: device controller mode + the REAL radard ModelLeadTracker
perception stage, ev6_measured lead noise + the injected vRel band.
"""
from __future__ import annotations

import functools
import random

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 60.0
EGO_V0_MPS = 30.0                 # road ~30 m/s steady cruise
LEAD_V0_MPS = 30.0               # near-constant-speed lead
T_FOLLOW_S = 1.45                # EV6 standard-personality headway
TARGET_GAP_M = EGO_V0_MPS * T_FOLLOW_S   # ~43.5 m -> steady follow at target headway
LEAD_MODEL_PROB = 0.98          # continuous vision track, never dropped
SEED = 42

# Injected perception vRel band (faithful to the road's -0.5..-2.4 m/s band that
# masked the slow mean close). Small mean close so the true gap holds sane, the
# noise is the +/-1.1 m/s vRel band the tracker EMAs and the MPC extrapolates.
VREL_MEAN_MPS = -0.3
VREL_BAND_MPS = 1.1

# Ignore the initial lead-acquisition / follow-settle transient; the CD7 defect
# is a STEADY-STATE one-frame jerk with the source already latched on lead0.
SETTLE_S = 8.0

# Desired-behavior bounds (task spec, road-derived). These are the strict-xfail
# oracle: they must FAIL on shipped HEAD (the anti-jerk envelope is absent) and
# PASS once the graded-onset comfort envelope lands.
#
# The oracle targets the DOWNWARD comfort-BRAKE one-frame jerk - the road's
# headline CD7 defect (200-10: aTarget stepped -0.31 -> -1.00, a sudden unnecessary
# BRAKE) and the safety-relevant felt jerk (an unexpected brake blip on a benign
# steady follow). The envelope is deliberately ASYMMETRIC (down-leg only), matching
# the CD5 relatch blend / flutter clamp precedent: the UPWARD leg (brake-RELEASE and
# re-accel toward a followed lead) is always-safe and must stay fast, so it is left
# free. So the bounded quantity is the worst one-frame DECREASE in aTarget.
MAX_ONE_FRAME_JERK_MPS3 = 1.0    # road step ~4.6 m/s^3 down; benign comfort bound ~1.0
MAX_SIGN_REVERSALS_PER_60S = 2   # road 200-9 tap1: a 0.86 m/s^2 reversal on a benign track
REVERSAL_AMPLITUDE_MPS2 = 0.5    # only count reversals whose full swing exceeds this

# Scenario-validity floors (benign kinematics; asserted unconditionally).
MIN_TRUE_GAP_FLOOR_M = 20.0      # steady follow: the gap must never collapse
FCW_TTC_WINDOW_S = 3.5           # no short-TTC / fast-close hazard gate may fire


def _band_vrel(rng: random.Random) -> float:
  return VREL_MEAN_MPS + rng.uniform(-VREL_BAND_MPS, VREL_BAND_MPS)


def _build_steps() -> list[StepInput]:
  rng = random.Random(SEED)
  steps: list[StepInput] = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    lead = LeadDirective(
      status=True,
      v_lead_mps=LEAD_V0_MPS,
      model_prob_target=LEAD_MODEL_PROB,
      d_rel_override_m=TARGET_GAP_M if i == 0 else None,
      # Noisy perception vRel band (road -0.5..-2.4 m/s) entering the REAL radard
      # ModelLeadTracker verbatim (radard_stage.py _fill_lead_v3 entry.v).
      measured_v_rel_mps=_band_vrel(rng),
      a_lead_k_mps2=0.0,   # aLeadK ~0 the whole run (road: benign, no lead decel)
      acquisition_reset=(i == 0),
    )
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=EGO_V0_MPS, lead_one=lead,
                           note="steady single-source follow, vRel noise band"))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config()


@functools.lru_cache(maxsize=1)
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name="comfort_floor_jerk",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    seed=SEED,
    perception_filter="auto",
  )


# CD7 fix knob: the graded-onset anti-jerk envelope bounds the per-frame
# |delta output_a_target| to ComfortJerkLimitMps3 * dt while NO hazard/urgency
# gate is active. Rollback sentinel = a large limit (spec max 50) disables it.
FIX_COMFORT_JERK_MPS3 = 0.8          # shipped default (bounds 0.04 m/s^2/frame)
ROLLBACK_COMFORT_JERK_MPS3 = 50.0    # rollback sentinel: envelope disabled


@functools.lru_cache(maxsize=2)
def _vehicle_config_jerk(jerk_limit: float):
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.ComfortJerkLimitMps3": f"{jerk_limit:g}",
  })


@functools.lru_cache(maxsize=2)
def _run_jerk(jerk_limit: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config_jerk(jerk_limit),
    scenario_name=f"comfort_floor_jerk_limit_{jerk_limit:g}",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    seed=SEED,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  # One row per 20 Hz planner step (the harness logs every control tick; the
  # planner updates once per 5 ticks at DT_MDL).
  return result.trace[::5]


def _post_settle_rows(result: SimulationResult) -> list[dict]:
  return [r for r in _planner_rows(result) if r["t_s"] >= SETTLE_S]


def _worst_one_frame_jerk(rows: list[dict]) -> tuple[float, float, float, float]:
  """Largest single-frame |delta aTarget| / dt over the rows (m/s^3), with the
  frame time and the aTarget pair straddling it (bidirectional; used only by the
  scenario-wiring guard to prove the worst blip is on a fully-tracked lead)."""
  worst = (0.0, None, 0.0, 0.0)
  for a, b in zip(rows[:-1], rows[1:], strict=False):
    jerk = abs(b["planner_accel_mps2"] - a["planner_accel_mps2"]) / DT_MDL
    if jerk > worst[0]:
      worst = (jerk, b["t_s"], a["planner_accel_mps2"], b["planner_accel_mps2"])
  return worst


def _worst_downward_jerk(rows: list[dict]) -> tuple[float, float, float, float]:
  """Largest single-frame DECREASE in aTarget / dt (m/s^3) - the comfort-BRAKE
  one-frame jerk (the road's headline CD7 defect, a sudden unnecessary brake blip
  on a benign steady follow). Only the downward leg is bounded by the envelope;
  the upward (brake-release / re-accel) leg is always-safe and left free."""
  worst = (0.0, None, 0.0, 0.0)
  for a, b in zip(rows[:-1], rows[1:], strict=False):
    drop = a["planner_accel_mps2"] - b["planner_accel_mps2"]  # positive when braking harder
    if drop <= 0.0:
      continue
    jerk = drop / DT_MDL
    if jerk > worst[0]:
      worst = (jerk, b["t_s"], a["planner_accel_mps2"], b["planner_accel_mps2"])
  return worst


def _sign_reversals(rows: list[dict], amplitude: float) -> int:
  """Count aTarget direction reversals whose full swing from the previous
  extremum exceeds `amplitude` (m/s^2) - the vacillation signature."""
  a = [r["planner_accel_mps2"] for r in rows]
  if len(a) < 2:
    return 0
  reversals = 0
  last_ext = a[0]
  direction = 0
  for i in range(1, len(a)):
    d = a[i] - a[i - 1]
    if abs(d) < 1e-6:
      continue
    nd = 1 if d > 0 else -1
    if direction != 0 and nd != direction and abs(a[i - 1] - last_ext) > amplitude:
      reversals += 1
      last_ext = a[i - 1]
    direction = nd
  return reversals


def _source_flips(rows: list[dict]) -> list[float]:
  return [b["t_s"] for a, b in zip(rows[:-1], rows[1:], strict=False)
          if a["planner_source"] != b["planner_source"]]


def _ttc_s(row: dict) -> float | None:
  gap = row["true_min_gap_m"]
  v_lead = row["active_lead_speed_mps"]
  if gap is None or v_lead is None:
    return None
  closing = row["v_ego_true_mps"] - v_lead
  if closing <= 0.3:
    return None
  return gap / closing


def test_comfort_floor_jerk_scenario_wiring() -> None:
  result = _run()

  # Tici-fidelity loop resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "ev6_measured"

  rows = _planner_rows(result)
  post = _post_settle_rows(result)
  assert post, "no post-settle planner rows"

  # Scenario-validity guard #1: SOURCE STAYS lead0 (no flips) once settled. The
  # road's CD7 jerk fired with ZERO source flips - this is NOT a handoff/flutter
  # defect (those are CD5/CD6). Asserted unconditionally so a scenario that
  # secretly flips source can never masquerade as a CD7 pass.
  post_flips = _source_flips(post)
  assert not post_flips, (
    f"source flipped post-settle at t={[round(x, 2) for x in post_flips[:8]]}; "
    f"scenario no longer isolates the single-source one-frame jerk")
  assert all(r["planner_source"] == "lead0" for r in post), (
    f"post-settle source not steadily lead0 "
    f"(sources: {sorted({r['planner_source'] for r in post})})")

  # Scenario-validity guard #2: benign kinematics - no fast-close / short-TTC /
  # FCW hazard gate fires the entire run. The one-frame jerk must be a pure
  # comfort artifact, not a real braking response.
  assert result.summary["minTrueGapM"] >= MIN_TRUE_GAP_FLOOR_M, (
    f"true gap collapsed to {result.summary['minTrueGapM']:.1f} m; not benign")
  min_ttc = min((ttc for r in rows if (ttc := _ttc_s(r)) is not None), default=None)
  assert min_ttc is None or min_ttc >= FCW_TTC_WINDOW_S, (
    f"a short-TTC hazard window opened (min TTC {min_ttc:.2f} s < {FCW_TTC_WINDOW_S} s); "
    f"scenario is not benign")
  assert not any(r["planner_fcw"] for r in rows), "FCW fired; scenario is not benign"

  # Scenario-validity guard #3: aLeadK stays ~0 (benign lead, no real decel) and
  # the track is present at high prob for the vast majority of the run. The
  # ev6_measured profile injects real ~0.15 Hz x 0.7 s vision prob dropouts; the
  # MPC lead stabilizer phantom-holds through them (source never flips off lead0,
  # guard #1), so the follow stays single-source. Allow the inherent dropout
  # fraction but require the track present most of the time.
  settled = [r for r in rows if r["t_s"] >= 1.0]
  present_frac = sum(1 for r in settled if r["lead_one_published_d_rel_m"] is not None) / len(settled)
  assert present_frac >= 0.85, (
    f"published lead present only {present_frac:.0%} of settled frames; track not continuous enough")
  raw_alead = [r["lead_one_a_lead_k_mps2"] for r in settled
               if r["lead_one_a_lead_k_mps2"] is not None]
  assert raw_alead and max(abs(v) for v in raw_alead) <= 0.05, (
    "injected aLeadK is not ~0; scenario is not the benign-lead CD7 case")

  # Scenario-validity guard #3b: the WORST one-frame jerk frame is itself on a
  # FULLY-TRACKED lead (high prob) - i.e. the jerk is a pure steady-follow noise
  # blip, NOT a prob-dropout recovery transient. This pins the CD7 mechanism.
  post = _post_settle_rows(result)
  _, worst_t, _, _ = _worst_one_frame_jerk(post)
  worst_row = next(r for r in post if r["t_s"] == worst_t)
  assert worst_row["lead_one_model_prob"] >= 0.9, (
    f"worst-jerk frame at t={worst_t}s is on a dropped/low-prob track "
    f"(prob {worst_row['lead_one_model_prob']:.2f}); not the steady-follow CD7 blip")

  # Scenario-validity guard #4: the noisy vRel band actually reaches the tracker
  # (the published vRel wobbles), so the one-frame jerk is genuinely noise-driven.
  pub_vrel = [r["lead_one_published_v_rel_mps"] for r in settled
              if r["lead_one_published_v_rel_mps"] is not None]
  assert pub_vrel and (max(pub_vrel) - min(pub_vrel)) >= 0.5, (
    "the injected vRel band did not reach the published tracker; noise not exercised")


@pytest.mark.xfail(strict=True, reason="CD7: no comfort anti-jerk envelope on planner output (fix pending)")
def test_comfort_floor_one_frame_jerk_bounded() -> None:
  result = _run()
  post = _post_settle_rows(result)

  jerk, jerk_t, prev_a, jerk_a = _worst_downward_jerk(post)
  reversals = _sign_reversals(post, REVERSAL_AMPLITUDE_MPS2)

  jerk_ok = jerk <= MAX_ONE_FRAME_JERK_MPS3
  reversals_ok = reversals <= MAX_SIGN_REVERSALS_PER_60S

  physics = (
    f"steady single-source lead0 follow (ego {EGO_V0_MPS} m/s, lead {LEAD_V0_MPS} m/s at "
    f"~{TARGET_GAP_M:.1f} m target headway, aLeadK ~0), vRel noise band "
    f"{VREL_MEAN_MPS}+/-{VREL_BAND_MPS} m/s on ev6_measured dRel noise:\n"
    f"  worst one-frame comfort-BRAKE jerk: {jerk:.2f} m/s^3 at t={jerk_t}s "
    f"({prev_a:+.3f} -> {jerk_a:+.3f} m/s^2 in one {DT_MDL*1000:.0f} ms frame; "
    f"bound <= {MAX_ONE_FRAME_JERK_MPS3} m/s^3; road stepped -0.31 -> -1.00 = ~4.6 m/s^3)\n"
    f"  >={REVERSAL_AMPLITUDE_MPS2} m/s^2 aTarget sign reversals in {DURATION_S:.0f}s: "
    f"{reversals} (bound <= {MAX_SIGN_REVERSALS_PER_60S}; road 200-9 tap1: a 0.86 m/s^2 reversal)\n"
    f"  post-settle source flips: {len(_source_flips(post))} (ZERO expected - not a handoff defect)"
  )
  assert jerk_ok and reversals_ok, physics


@pytest.mark.xfail(strict=True, reason="CD7: ComfortJerkLimitMps3 envelope not yet implemented (fix pending)")
def test_comfort_jerk_limit_knob_fix_vs_rollback() -> None:
  """CD7 fix-knob oracle (the NEW threshold's rollback sentinel).

  The SAFETY RULE requires EVERY new threshold to have an oracled rollback knob.
  This test exercises the CD7 envelope's own knob directly: ComfortJerkLimitMps3,
  the per-frame |delta output_a_target| jerk bound applied while NO hazard gate is
  active.

    limit = 0.8 (the fix, shipped default): the graded-onset envelope bounds the
      downward comfort-brake one-frame jerk to ComfortJerkLimitMps3 * dt =
      0.04 m/s^2/frame, so the steady-follow noise no longer step-changes aTarget
      hard into a brake - the brake jerk drops under the comfort bound and the
      vacillation reversals collapse.
    limit = 50.0 (rollback sentinel): the envelope is effectively disabled, so the
      pre-fix un-enveloped MPC output is restored and the road pathology reappears
      (downward comfort-brake jerk >> the comfort bound).

  The injected true kinematics + noise are identical in both twins; only the jerk
  limit differs, so any behavioral divergence is attributable to the CD7 fix alone.
  """
  fix = _run_jerk(FIX_COMFORT_JERK_MPS3)
  rollback = _run_jerk(ROLLBACK_COMFORT_JERK_MPS3)

  fix_post = _post_settle_rows(fix)
  roll_post = _post_settle_rows(rollback)

  # Matched twins: identical true lead kinematics (the noise seed + directives are
  # identical; only the comfort jerk limit differs).
  for r_fix, r_roll in zip(fix.trace, rollback.trace, strict=True):
    assert r_fix["active_lead_speed_mps"] == pytest.approx(r_roll["active_lead_speed_mps"], abs=1e-9)
    assert r_fix["lead_one_a_lead_k_mps2"] == pytest.approx(r_roll["lead_one_a_lead_k_mps2"], abs=1e-9)

  fix_jerk, *_ = _worst_downward_jerk(fix_post)
  roll_jerk, *_ = _worst_downward_jerk(roll_post)
  fix_rev = _sign_reversals(fix_post, REVERSAL_AMPLITUDE_MPS2)
  roll_rev = _sign_reversals(roll_post, REVERSAL_AMPLITUDE_MPS2)

  physics = (
    f"CD7 comfort jerk-limit knob (ComfortJerkLimitMps3), steady single-source follow:\n"
    f"  limit={FIX_COMFORT_JERK_MPS3} (fix):      peak comfort-brake jerk {fix_jerk:.2f} m/s^3 "
    f"(bound {MAX_ONE_FRAME_JERK_MPS3}), reversals {fix_rev}\n"
    f"  limit={ROLLBACK_COMFORT_JERK_MPS3} (rollback): peak comfort-brake jerk {roll_jerk:.2f} m/s^3, "
    f"reversals {roll_rev} (pre-fix pathology; road stepped ~4.6 m/s^3)"
  )

  # The fix: peak comfort-brake jerk holds under the comfort bound and reversals stay small.
  assert fix_jerk <= MAX_ONE_FRAME_JERK_MPS3, physics
  assert fix_rev <= MAX_SIGN_REVERSALS_PER_60S, physics

  # The rollback sentinel restores the pre-fix un-enveloped pathology (jerk far
  # above the comfort bound).
  assert roll_jerk > MAX_ONE_FRAME_JERK_MPS3, physics

  # And the fix is strictly smoother than its own rollback.
  assert fix_jerk < roll_jerk, physics
