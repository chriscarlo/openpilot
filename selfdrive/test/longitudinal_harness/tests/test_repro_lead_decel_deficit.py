"""Repro: NEAR_COLLISION #1 (road unit 200-13 EDGE1) - lead-decel truth deficit + silenced FCW.

Field evidence (road_forensics.json unit 200-13, seg 00000200--8cbf2c9481--12@47..13@7,
driver stomp at 13@2.34):

CD3 (lead-decel truth deficit): the lead braked at a true (position-derived)
-1.3..-2.3 m/s^2 sustained ~6 s, but the raw model leadsV3 'a' reported only
-0.1..-0.7 (a 0.1-0.5x underreport), radard's MODEL_LEAD_ACCEL_TAU_S=0.6 EMA
halved it again (published aLeadK peak -0.48), and aLeadTau=0.3 decayed that
near-zero value across the MPC horizon. Seeing a barely-decelerating lead the
MPC ramped aTarget at ~0.2 m/s^3 while THW fell 1.55 -> 0.45 s; the driver
disengaged at THW 0.45 s (d 11.1 m, TTC 1.9 s, vRel -5.96) and braked -4.2.

CD2 (FCW corroboration veto false invariant): radard.py _update_fcw_corroboration
votes 'phantom' whenever raw_drel - filtered dRel > ModelLeadFcwCorrobTolM=2.5 in
>= 2 of 3 frames, assuming filtered >= raw on genuine threats. But on this real
fast close the closing-urgency blend deliberately published dRel 5-7 m MORE
pessimistic than the raw model x (13@2.30: filtered 11.1 vs raw 16.9; 13@2.84:
8.5 vs 15.8 - the raw model x itself ran optimistic against ground truth), so
fcwSuppressed=True held through the deepest 3 s (13@2.84-5.84) and reset
mpc.crash_cnt (long_mpc.py FCW block). Zero FCW fired the entire drive.

Scenario synthesis (all numbers road-derived, see _measure/physics for the
side-by-side): steady follow at 24 m/s (road ego 24.8 at the tap, vCruise 120
kph) at THW 1.88 s (road steady 1.88), cruise pull erodes THW to ~1.5 by decel
onset (road: 1.55 after the cruise-source phase); the lead then brakes at a true
-1.8 m/s^2 for 6 s from 24 m/s while the synthesized leadsV3 accel reports only
0.3x of it (road ratio band 0.1-0.5x). Injection point: the harness lead
directive's a_lead_k_mps2 flows into the raw lead's aLeadK, which the radard
stage writes verbatim into the leadsV3 synthesis -
selfdrive/test/longitudinal_harness/radard_stage.py _fill_lead_v3:
`entry.a = [float(raw_lead.aLeadK), float(raw_lead.aLeadK)]` - i.e. exactly the
model's underreported accel measurement entering the REAL radard pipeline
(Schmitt latch -> ModelLeadTracker 0.6 s accel EMA -> published aLeadK/aLeadTau).
During the deep close the raw measured x additionally carries the road-measured
model far-x optimism (+6 m; road raw-minus-filtered +5.8..+7.3 m) via
LeadDirective.measured_d_rel_bias_m so the corroboration veto sees the same
raw-vs-filter disagreement it saw on the road.

A matched truthful-accel pair run (identical true kinematics, reported accel
= true -1.8) isolates the mechanism: it keeps min THW 0.99 s / min TTC 4.8 s,
while the deficit run collapses to min THW 0.49 s / min TTC 2.4 s (road: 0.45 s
/ 1.9 s at the driver stomp).

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

DURATION_S = 18.0
EGO_V0_MPS = 24.0                 # road ego 24.80 m/s at the driver stomp
CRUISE_SPEED_MPS = 120.0 / 3.6    # road vCruise 120 kph
LEAD_V0_MPS = 24.0                # task/road: true decel starts from 24 m/s
INITIAL_GAP_M = 45.0              # THW 1.88 s (road steady follow 1.88 s)
LEAD_MODEL_PROB = 0.99            # road: vision-only track, prob 0.98-1.00, never dropped

TRUE_DECEL_MPS2 = -1.8            # road position-derived -1.3..-2.3 sustained
DECEL_START_S = 6.0
DECEL_DURATION_S = 6.0
LEAD_V_FLOOR_MPS = LEAD_V0_MPS + TRUE_DECEL_MPS2 * DECEL_DURATION_S  # 13.2 m/s
REPORTED_ACCEL_RATIO = 0.3        # road-measured raw-model/true ratio band 0.1-0.5x
ROAD_RATIO_BAND = (0.1, 0.5)

# Road-measured raw model far-x optimism during the deep close: raw leadsV3 x ran
# +5.8 m (13@2.30) to +7.3 m (13@2.84) above the filtered/true gap; fcwSuppressed
# was voted at 13@0.5-0.7 and continuously 13@2.84-5.84. Mapped onto the harness
# timeline (decel onset t=6.0 ~ road 12@56.5): ramp in ~4 s after onset, hold
# through the deepest close, fade as the closing rate dies.
RAW_X_OPTIMISM_M = 6.0
RAW_X_RAMP_START_S = 10.0
RAW_X_RAMP_END_S = 11.5
RAW_X_HOLD_END_S = 14.5
RAW_X_FADE_END_S = 16.0

# Desired-behavior bounds (task spec, derived from the road kinematics).
ONSET_ACCEL_MPS2 = -1.0
MAX_ONSET_DELAY_S = 1.5
MIN_THW_FLOOR_S = 0.9             # road THW fell to 0.45 s at the driver stomp
FCW_TTC_WINDOW_S = 3.5            # road TTC 1.9 s at the tap, min 1.14 s
MOVING_V_MPS = 2.0
MIN_CLOSING_MPS = 0.3

FCW_CORROB_TOL_M = 2.5            # ModelLeadFcwCorrobTolM (device livetune 2026-07-02)


def _lead_speed(t_s: float) -> float:
  if t_s < DECEL_START_S:
    return LEAD_V0_MPS
  return max(LEAD_V_FLOOR_MPS, LEAD_V0_MPS + TRUE_DECEL_MPS2 * (t_s - DECEL_START_S))


def _raw_x_optimism(t_s: float) -> float:
  if t_s < RAW_X_RAMP_START_S:
    return 0.0
  if t_s < RAW_X_RAMP_END_S:
    return RAW_X_OPTIMISM_M * (t_s - RAW_X_RAMP_START_S) / (RAW_X_RAMP_END_S - RAW_X_RAMP_START_S)
  if t_s < RAW_X_HOLD_END_S:
    return RAW_X_OPTIMISM_M
  if t_s < RAW_X_FADE_END_S:
    return RAW_X_OPTIMISM_M * (RAW_X_FADE_END_S - t_s) / (RAW_X_FADE_END_S - RAW_X_HOLD_END_S)
  return 0.0


def _build_steps(reported_ratio: float) -> list[StepInput]:
  steps = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    v_lead = _lead_speed(t_s)
    braking = DECEL_START_S <= t_s < DECEL_START_S + DECEL_DURATION_S
    lead = LeadDirective(
      status=True,
      v_lead_mps=v_lead,
      model_prob_target=LEAD_MODEL_PROB,
      # MODEL-UNDERREPORTED lead accel: this is the raw leadsV3 'a' measurement
      # (radard_stage.py _fill_lead_v3 writes it verbatim into entry.a).
      a_lead_k_mps2=(TRUE_DECEL_MPS2 * reported_ratio) if braking else 0.0,
      d_rel_override_m=INITIAL_GAP_M if i == 0 else None,
      measured_d_rel_bias_m=_raw_x_optimism(t_s),
      acquisition_reset=i == 0,
    )
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead,
                           note="lead brakes -1.8 true, model reports 0.3x" if braking else "steady follow"))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config()


@functools.lru_cache(maxsize=2)
def _run(reported_ratio: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name=f"lead_decel_deficit_ratio_{reported_ratio:g}",
    steps=_build_steps(reported_ratio),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


# CD3 fix knob: the MPC lead stabilizer amplifies the underreported aLeadK toward
# the corroborating vLead-trend finite-difference (_apply_lead_accel_corr_bound
# amplify branch). Gain 1.0 = shipped fix; gain 0.0 = pre-fix rollback sentinel.
FIX_AMPLIFY_GAIN = 1.0
ROLLBACK_AMPLIFY_GAIN = 0.0


@functools.lru_cache(maxsize=2)
def _vehicle_config_amplify(amplify_gain: float):
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.LeadAccelCorrAmplifyGain": f"{amplify_gain:g}",
  })


@functools.lru_cache(maxsize=2)
def _run_amplify(amplify_gain: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config_amplify(amplify_gain),
    scenario_name=f"lead_decel_deficit_amplify_{amplify_gain:g}",
    steps=_build_steps(REPORTED_ACCEL_RATIO),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _thw_s(row: dict) -> float | None:
  gap = row["true_min_gap_m"]
  if gap is None or row["v_ego_true_mps"] <= 0.5:
    return None
  return gap / row["v_ego_true_mps"]


def _ttc_s(row: dict) -> float | None:
  gap = row["true_min_gap_m"]
  v_lead = row["active_lead_speed_mps"]
  if gap is None or v_lead is None:
    return None
  closing = row["v_ego_true_mps"] - v_lead
  if closing <= MIN_CLOSING_MPS:
    return None
  return gap / closing


def _deep_ttc_window(trace: list[dict]) -> list[dict]:
  window = []
  for row in trace:
    ttc = _ttc_s(row)
    if ttc is not None and ttc < FCW_TTC_WINDOW_S and row["v_ego_true_mps"] > MOVING_V_MPS:
      window.append(row)
  return window


def _measure(result: SimulationResult) -> dict:
  trace = result.trace
  onset_t = next((row["t_s"] for row in trace
                  if row["t_s"] >= DECEL_START_S and row["planner_accel_mps2"] <= ONSET_ACCEL_MPS2), None)
  thws = [thw for row in trace if (thw := _thw_s(row)) is not None]
  ttcs = [ttc for row in trace if (ttc := _ttc_s(row)) is not None]
  window = _deep_ttc_window(trace)
  pub_alead = [row["lead_one_published_a_lead_k_mps2"] for row in trace
               if row["lead_one_published_a_lead_k_mps2"] is not None]
  return {
    "brake_onset_t_s": onset_t,
    "brake_onset_delay_s": None if onset_t is None else onset_t - DECEL_START_S,
    "min_thw_s": min(thws),
    "min_ttc_s": min(ttcs) if ttcs else None,
    "min_true_gap_m": result.summary["minTrueGapM"],
    "peak_planner_brake_mps2": result.summary["peakPlannerBrakeMps2"],
    "published_a_lead_k_peak_mps2": min(pub_alead),
    "fcw_count": sum(1 for row in trace if row["planner_fcw"]),
    "fcw_in_window_count": sum(1 for row in window if row["planner_fcw"]),
    "window_n": len(window),
    "window_t_span_s": (window[0]["t_s"], window[-1]["t_s"]) if window else None,
    "window_suppressed_n": sum(1 for row in window if row["lead_one_fcw_suppressed"]),
    "max_crash_cnt": max(row["mpc_crash_cnt"] for row in trace),
    "max_raw_minus_published_m": max(
      (row["lead_one_measured_d_rel_m"] - row["lead_one_published_d_rel_m"])
      for row in trace
      if row["lead_one_measured_d_rel_m"] is not None and row["lead_one_published_d_rel_m"] is not None),
  }


def test_lead_decel_deficit_scenario_wiring() -> None:
  deficit = _run(REPORTED_ACCEL_RATIO)
  truthful = _run(1.0)

  # Tici-fidelity loop resolved as intended.
  for result in (deficit, truthful):
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"

  decel_rows = [row for row in deficit.trace if DECEL_START_S + 0.5 <= row["t_s"] < DECEL_START_S + DECEL_DURATION_S]
  assert decel_rows
  # True (plant-side) lead decel is the full -1.8: measured from the ground-truth
  # lead speed slope, independent of anything the perception/planner stack does.
  row_a = min(decel_rows, key=lambda r: r["t_s"])
  row_b = max(decel_rows, key=lambda r: r["t_s"])
  true_decel = (row_b["active_lead_speed_mps"] - row_a["active_lead_speed_mps"]) / (row_b["t_s"] - row_a["t_s"])
  assert true_decel == pytest.approx(TRUE_DECEL_MPS2, abs=0.05)
  # ...while the injected raw model leadsV3 accel reports only 0.3x of it
  # (radard_stage.py _fill_lead_v3 entry.a), inside the road-measured 0.1-0.5x band.
  raw_reported = [row["lead_one_a_lead_k_mps2"] for row in decel_rows]
  assert all(v == pytest.approx(TRUE_DECEL_MPS2 * REPORTED_ACCEL_RATIO, abs=0.05) for v in raw_reported)
  assert ROAD_RATIO_BAND[0] <= REPORTED_ACCEL_RATIO <= ROAD_RATIO_BAND[1]
  truthful_reported = [row["lead_one_a_lead_k_mps2"] for row in truthful.trace
                       if DECEL_START_S + 0.5 <= row["t_s"] < DECEL_START_S + DECEL_DURATION_S]
  assert all(v == pytest.approx(TRUE_DECEL_MPS2, abs=0.05) for v in truthful_reported)

  # Corroborated continuous track (road: prob 0.98-1.00, never dropped): the
  # published lead exists on every planner step once the Schmitt latch is in.
  settled = [row for row in deficit.trace if row["t_s"] >= 1.0]
  assert all(row["lead_one_published_d_rel_m"] is not None for row in settled)
  assert all(row["lead_one_model_prob"] >= 0.9 for row in settled)

  # The road-measured raw model far-x optimism reaches the tracker: raw measured
  # dRel runs ~+6 m above ground truth through the deep close (road +5.8..+7.3 m).
  hold_rows = [row for row in deficit.trace if RAW_X_RAMP_END_S <= row["t_s"] < RAW_X_HOLD_END_S]
  assert hold_rows
  for row in hold_rows:
    assert row["lead_one_measured_d_rel_m"] - row["lead_one_true_d_rel_m"] == pytest.approx(RAW_X_OPTIMISM_M, abs=0.6)

  # Matched pair: identical true kinematics, only the reported accel differs.
  for r_def, r_tru in zip(deficit.trace, truthful.trace, strict=True):
    assert r_def["active_lead_speed_mps"] == pytest.approx(r_tru["active_lead_speed_mps"], abs=1e-9)

  # Mechanism containment: if a silent deep-TTC excursion exists, the FCW
  # corroboration veto must be what is holding FCW down (road: fcwSuppressed=True
  # through 13@2.84-5.84). Green today via the veto; stays green when a fix
  # either fires FCW inside the window or prevents the window entirely; goes RED
  # if the deep silent window ever exists through some OTHER mechanism.
  window = _deep_ttc_window(deficit.trace)
  fcw_in_window = any(row["planner_fcw"] for row in window)
  veto_engaged_in_window = any(row["lead_one_fcw_suppressed"] for row in window)
  assert (not window) or fcw_in_window or veto_engaged_in_window


def test_control_brake_onset_and_thw_floor() -> None:
  # FIXED (CD3 lead-decel truth deficit, road 200-13 EDGE1): the raw model
  # leadsV3.a still reports only 0.3x of the true -1.8 m/s^2 decel and radard's
  # 0.6 s accel EMA still halves it, but the MPC lead stabilizer now amplifies
  # aLeadK toward the corroborating vLead-trend finite-difference
  # (_apply_lead_accel_corr_bound amplify branch, LeadAccelCorrAmplifyGain).
  # With both the model and the trend agreeing the lead brakes, the MPC
  # extrapolates the true decel and brakes early enough to hold THW >= 0.9 s.
  deficit = _measure(_run(REPORTED_ACCEL_RATIO))
  truthful = _measure(_run(1.0))

  onset_ok = deficit["brake_onset_delay_s"] is not None and deficit["brake_onset_delay_s"] <= MAX_ONSET_DELAY_S
  thw_ok = deficit["min_thw_s"] >= MIN_THW_FLOOR_S

  physics = (
    f"model-underreported lead decel (true {TRUE_DECEL_MPS2} m/s^2 x {DECEL_DURATION_S}s from t={DECEL_START_S}s, "
    f"raw leadsV3.a = {REPORTED_ACCEL_RATIO}x = {TRUE_DECEL_MPS2 * REPORTED_ACCEL_RATIO:+.2f}) vs matched truthful run:\n"
    f"  brake onset (planner <= {ONSET_ACCEL_MPS2} m/s^2): deficit t={deficit['brake_onset_t_s']}s "
    f"(delay {deficit['brake_onset_delay_s']}s, bound {MAX_ONSET_DELAY_S}s), "
    f"truthful t={truthful['brake_onset_t_s']}s (delay {truthful['brake_onset_delay_s']}s)\n"
    f"  min THW: deficit {deficit['min_thw_s']:.3f}s (floor {MIN_THW_FLOOR_S}s; road fell to 0.45s), "
    f"truthful {truthful['min_thw_s']:.3f}s\n"
    f"  min TTC: deficit {deficit['min_ttc_s']:.2f}s (road 1.9s at the stomp, 1.14s min), "
    f"truthful {truthful['min_ttc_s']:.2f}s\n"
    f"  min true gap: deficit {deficit['min_true_gap_m']:.2f} m (road 3.9 m after the driver's -4.2 stomp), "
    f"truthful {truthful['min_true_gap_m']:.2f} m\n"
    f"  published aLeadK peak: deficit {deficit['published_a_lead_k_peak_mps2']:+.2f} m/s^2 vs true {TRUE_DECEL_MPS2} "
    f"(road: published peak -0.48 vs true -1.3..-2.3), truthful {truthful['published_a_lead_k_peak_mps2']:+.2f}\n"
    f"  peak planner brake: deficit {deficit['peak_planner_brake_mps2']:.2f}, truthful {truthful['peak_planner_brake_mps2']:.2f} m/s^2"
  )
  assert onset_ok and thw_ok, physics


def test_fcw_fires_inside_deep_ttc_window() -> None:
  # FIXED via CD3 (lead-decel truth deficit): the assertion has two legitimate
  # pass paths - "FCW fires inside the deep-TTC window" OR "the planner never
  # lets TTC dip into the window at all". The CD3 aLeadK amplify gives the MPC a
  # truthful lead-decel signal, so the near-collision never develops: min TTC
  # stays ~4.9 s (> the 3.5 s window) and min THW ~0.99 s. The deep silent
  # window that the CD2 FCW-corroboration veto used to hold open no longer
  # exists on this scenario, so no_deep_window is True. (The separate CD2 veto
  # fix is validated by test_repro_phantom_near_collision and the fcw_override
  # phantom-suppression guard, both green.)
  deficit = _run(REPORTED_ACCEL_RATIO)
  m = _measure(deficit)

  window = _deep_ttc_window(deficit.trace)
  # Desired behavior on a corroborated, continuously-tracked genuine collision
  # course: either the planner never lets TTC dip below the FCW window at all,
  # or FCW fires inside it. (Road: TTC fell to 1.14 s and FCW stayed silent.)
  no_deep_window = not window
  fcw_fired_in_window = m["fcw_in_window_count"] > 0

  physics = (
    f"deep-TTC window (TTC < {FCW_TTC_WINDOW_S}s while moving): "
    + (f"t={m['window_t_span_s'][0]:.2f}..{m['window_t_span_s'][1]:.2f}s, n={m['window_n']} ticks"
       if window else "none") + "\n"
    f"  min TTC: {m['min_ttc_s']:.2f}s, min THW {m['min_thw_s']:.3f}s, min true gap {m['min_true_gap_m']:.2f} m "
    f"(road: TTC 1.9s/THW 0.45s/gap 11.1 m at the driver stomp)\n"
    f"  longitudinalPlan.fcw count: {m['fcw_count']} total, {m['fcw_in_window_count']} inside the window "
    f"(road: zero FCW the entire drive)\n"
    f"  fcwSuppressed ticks inside window: {m['window_suppressed_n']}/{m['window_n']} "
    f"(road: fcwSuppressed=True continuously 13@2.84-5.84, the deepest 3 s)\n"
    f"  max raw-minus-published dRel: {m['max_raw_minus_published_m']:.2f} m vs corroboration tol {FCW_CORROB_TOL_M} m "
    f"(road: raw model x ran +5.8..+7.3 m above the filtered/true gap)\n"
    f"  max mpc.crash_cnt: {m['max_crash_cnt']:.0f} (fcw needs > 2; the veto resets it AND the 0.3x-extrapolated lead "
    f"never yields a predicted crash)"
  )
  assert no_deep_window or fcw_fired_in_window, physics


def test_phantom_collapse_still_never_fires_fcw() -> None:
  """Green guard (phantom-suppression direction must hold): the existing
  phantom-collapse scenario - the very case the corroboration veto exists for -
  still never raises FCW. Reuses the fcw_override scenario helper verbatim."""
  from selfdrive.test.longitudinal_harness.tests.test_repro_stop_slam_fcw_override import (
    FCW_TRUE_GAP_M,
    MOVING_V_MPS as PHANTOM_MOVING_V_MPS,
    _run as _run_phantom_collapse,
  )

  result = _run_phantom_collapse()
  false_fcw = next(
    (row for row in result.trace
     if row["planner_fcw"] and row["v_ego_true_mps"] > PHANTOM_MOVING_V_MPS
     and row["true_min_gap_m"] is not None and row["true_min_gap_m"] > FCW_TRUE_GAP_M),
    None,
  )
  assert false_fcw is None, (
    f"phantom-collapse scenario raised FCW at t={false_fcw['t_s']:.2f}s "
    f"(v_ego {false_fcw['v_ego_true_mps']:.2f} m/s, true gap {false_fcw['true_min_gap_m']:.2f} m)"
  )


def test_amplify_gain_knob_fix_vs_rollback() -> None:
  """CD3 fix-knob oracle (the NEW threshold's rollback sentinel).

  The SAFETY RULE requires EVERY new threshold to have an oracled rollback knob.
  CD1 encodes this via test_project_gain_knob_fix_vs_rollback and CD2 via its
  raw-closing escape; this test exercises the CD3 fix's own knob directly:
  LeadAccelCorrAmplifyGain, which scales how strongly the MPC lead stabilizer
  amplifies the underreported aLeadK toward the corroborating vLead-trend
  finite-difference (_apply_lead_accel_corr_bound amplify branch).

    gain = 1.0 (the fix): the MPC extrapolates the true lead decel from the
      model+trend agreement and brakes early enough that the near-collision never
      develops - min THW holds >= 0.9 s and TTC never dips into the deep-TTC
      window (road pathology absent).
    gain = 0.0 (rollback sentinel): the pre-fix un-amplified aLeadK is restored,
      so the MPC only sees the 0.3x-underreported decel; the road pathology
      reappears - THW collapses to ~0.49 s / TTC ~2.36 s and the deep-TTC window
      opens (road: THW 0.45 s / TTC 1.9 s at the driver stomp).

  The injected true kinematics are identical in both twins; only the amplify gain
  differs, so any behavioral divergence is attributable to the CD3 fix alone."""
  fix = _run_amplify(FIX_AMPLIFY_GAIN)
  rollback = _run_amplify(ROLLBACK_AMPLIFY_GAIN)
  m_fix = _measure(fix)
  m_roll = _measure(rollback)

  # Matched twins: identical true lead kinematics, only the amplify gain differs.
  for r_fix, r_roll in zip(fix.trace, rollback.trace, strict=True):
    assert r_fix["active_lead_speed_mps"] == pytest.approx(r_roll["active_lead_speed_mps"], abs=1e-9)
    assert r_fix["lead_one_a_lead_k_mps2"] == pytest.approx(r_roll["lead_one_a_lead_k_mps2"], abs=1e-9)

  fix_window = _deep_ttc_window(fix.trace)
  roll_window = _deep_ttc_window(rollback.trace)

  physics = (
    f"CD3 amplify-gain knob (LeadAccelCorrAmplifyGain), lead brakes {TRUE_DECEL_MPS2} m/s^2 true "
    f"while raw leadsV3.a reports only {REPORTED_ACCEL_RATIO}x:\n"
    f"  gain={FIX_AMPLIFY_GAIN} (fix):      min THW {m_fix['min_thw_s']:.3f} s (floor {MIN_THW_FLOOR_S} s), "
    f"min TTC {m_fix['min_ttc_s']:.2f} s, deep-TTC window {len(fix_window)} ticks\n"
    f"  gain={ROLLBACK_AMPLIFY_GAIN} (rollback): min THW {m_roll['min_thw_s']:.3f} s, "
    f"min TTC {m_roll['min_ttc_s']:.2f} s, deep-TTC window {len(roll_window)} ticks "
    f"(pre-fix pathology; road THW 0.45 s / TTC 1.9 s at the driver stomp)"
  )

  # The fix: THW holds at/above the floor and TTC never enters the deep window.
  assert m_fix["min_thw_s"] >= MIN_THW_FLOOR_S, physics
  assert not fix_window, physics

  # The rollback sentinel restores the road pathology: THW collapses below the
  # floor and the deep-TTC window opens.
  assert m_roll["min_thw_s"] < MIN_THW_FLOOR_S, physics
  assert roll_window, physics

  # And the fix is strictly safer than its own rollback on the two headline
  # road-derived metrics (larger THW headroom, larger TTC margin).
  assert m_fix["min_thw_s"] > m_roll["min_thw_s"], physics
  assert m_fix["min_ttc_s"] > m_roll["min_ttc_s"], physics


def test_recovery_taint_immediately_before_genuine_brake_preserves_cd3_safety(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A calm-recovery epoch immediately before a real brake must not make CD3
  miss its established onset/THW contract while fresh correlation re-settles."""
  from openpilot.selfdrive.controls.radard import ModelLeadTrack

  original_get_radar_state = ModelLeadTrack.get_RadarState
  marked_frames = 0

  def get_radar_state_with_pre_brake_recovery(self, cfg=None):
    nonlocal marked_frames
    out = original_get_radar_state(self, cfg)
    if DECEL_START_S - 1.0 <= float(self.last_t) < DECEL_START_S:
      out["closingGovernorRecovery"] = True
      marked_frames += 1
    return out

  monkeypatch.setattr(ModelLeadTrack, "get_RadarState", get_radar_state_with_pre_brake_recovery)
  result = run_harness(
    vehicle_config=_vehicle_config_amplify(FIX_AMPLIFY_GAIN),
    scenario_name="lead_decel_deficit_pre_brake_recovery_taint",
    steps=_build_steps(REPORTED_ACCEL_RATIO),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )
  measured = _measure(result)
  first_amplified = next(
    (
      row for row in result.trace[::5]
      if row["t_s"] >= DECEL_START_S
      and row["mpc_acc_source_debug"].get("approach_reacquire_lead_decel_mps2", 0.0) >= 1.0
    ),
    None,
  )
  deep_ttc_window = _deep_ttc_window(result.trace)

  assert marked_frames > 0
  assert measured["brake_onset_delay_s"] is not None
  assert measured["brake_onset_delay_s"] <= MAX_ONSET_DELAY_S
  assert measured["min_thw_s"] >= MIN_THW_FLOOR_S
  assert not deep_ttc_window or measured["fcw_in_window_count"] > 0
  assert first_amplified is not None
  # Recovery deliberately discards the shaped derivative history; the normal
  # 2*tau contract therefore resumes CD3 after one fresh ~0.6 s epoch. The
  # model/governor paths keep the absolute onset and THW/TTC safety bounds green
  # during that bounded re-settle.
  assert first_amplified["t_s"] <= DECEL_START_S + 0.65
