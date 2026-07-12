"""Repro: CD6 (road 200-6) - one-frame cruise<->lead0 handoff sign-flip on a
vLeadK rollover, plus the EDGE1 direction (cruise accelerating INTO a sub-target
lead).

Field evidence (road_forensics.json G3 / CD6, seg 00000200--8cbf2c9481--6):

  aTarget flipped +0.40 -> -0.41 (t=31.305->31.354) and +0.58 -> -0.56
  (t=46.908->46.952) in single 50 ms frames, each triggered by vLeadK dropping
  2.5-2.7 m/s in <2 s while dRel > 75 m barely moved (the vRel EMA leads the
  distance evidence). The longitudinalPlanSource flip coincided frame-for-frame
  with the aTarget discontinuity. The no-radar 50 Hz EMA device controller branch
  amplified it: cc vs co divergence 1.14-1.18 m/s^2 at the flip frames. The
  flutter clamp never engaged (it needs >= 2 source flips in 1 s; these are single
  flips) and CruiseReacquire* only limits the POSITIVE leg, so nothing bounds the
  first braking frame after a vLeadK-driven cruise-side dive.

  200-13 EDGE1 phase-1 is the same handoff boundary failing the OTHER way: the
  cruise source accelerated INTO a sub-target lead (+0.49 aTarget) while THW
  eroded 1.94 -> 1.55 s, before the lead braked - headway spent on the cruise
  source over a lead already inside the desired follow distance.

Scenario synthesis (all numbers road-derived). Full tici-fidelity loop: device
controller mode + the REAL radard ModelLeadTracker perception stage, noise OFF so
every number is deterministic.

SIGN-FLIP scenario (road 200-6):
  Cruise-accel with the lead OUTSIDE the follow envelope: ego 27 m/s, set speed
  34 m/s (cruise pulling hard, aTarget ~+0.56 like the road's +0.58), a lead
  cruising at the set speed ~70 m ahead (dRel > target_gap + 15 m; the true gap
  holds ~99 m the whole run - the lead never actually closes). At t=6.0 s the
  raw perception vRel dips 5.5 m/s over 0.8 s and recovers; the real radard EMA
  attenuates that to the road's ~3.5 m/s published vLeadK rollover while the
  distance evidence says nothing changed. The rollover briefly makes the lead0
  obstacle win, forcing a cruise->lead0->cruise source transition, and the
  single-frame cruise-side dive slams aTarget negative.

  The injection is faithful to the device pipeline: the harness raw lead's
  measured vRel becomes leadsV3.v = v_ego + vRel (radard_stage.py _fill_lead_v3
  entry.v), which the REAL radard ModelLeadTracker EMAs into the published
  vLeadK/vRel the planner extrapolates with - i.e. exactly the vRel-EMA-leads-
  distance rollover the road exhibited, entering the real Schmitt-latched tracker.

EDGE1 scenario (road 200-13 phase-1):
  ego 24 m/s well below a 28 m/s set speed (cruise accelerating at ~+0.85), a
  slower 22 m/s lead resolves already INSIDE the desired follow distance (gap
  38 m, THW ~1.53 s - the road's eroded 1.55 s) with a closing (negative) vRel
  trend. For the source-boundary frames before lead0 latches, the cruise source
  keeps spending headway - positive aTarget into a sub-target lead (road: +0.49).

CD6 desired behavior (task spec, road-derived): rate-limit the handoff both signs
for the first 0.3-0.5 s after ANY source transition so no single frame flips
aTarget more than ~0.4 m/s^2, and with a valid lead inside the desired follow
distance on a negative-vRel trend cap positive aTarget <= +0.1 (do not spend
headway on the cruise source). This repro is strict-xfail until that fix lands.
"""
from __future__ import annotations

import functools

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import desired_follow_distance

# ---- SIGN-FLIP scenario (road 200-6) -------------------------------------------
SF_DURATION_S = 16.0
SF_EGO_V0_MPS = 27.0              # road ego ~27 m/s under the cruise pull
SF_CRUISE_MPS = 34.0             # set speed above ego: cruise accelerates (road aTarget +0.58)
SF_LEAD_TRUE_V_MPS = 34.0        # lead cruises at set speed -> true gap holds, lead never closes
SF_GAP0_M = 70.0                 # OUTSIDE the follow envelope (dRel > target_gap + 15 m; road dRel > 75 m)
SF_LEAD_PROB = 0.97             # road: continuous vision track, never dropped
SF_DROP_START_S = 6.0
SF_DROP_DUR_S = 0.8             # <2 s (road: vLeadK dropped 2.5-2.7 m/s in <2 s)
SF_DROP_DV_MPS = 5.5           # raw dip -> ~3.5 m/s published vLeadK rollover after real radard

# Desired-behavior bounds (task spec, road-derived).
SF_MAX_ONE_FRAME_DELTA_A_MPS2 = 0.4   # road: single-frame |delta aTarget| 0.8-2.7
SF_MAX_CC_CO_DIVERGENCE_MPS2 = 0.5    # road: cc vs co divergence 1.14-1.18 at the flip frame
SF_MIN_TRUE_GAP_FLOOR_M = 60.0        # the lead never really closes (true gap holds ~99 m)


def _sf_measured_vrel(t_s: float) -> float | None:
  """Perception vRel artifact: a transient dip and recovery (the vLeadK rollover).

  None outside the dip window -> the harness computes measured vRel from the true
  kinematics (lead at set speed), so away from the rollover the tracker sees the
  honest closing rate.
  """
  if t_s < SF_DROP_START_S:
    return None
  td = t_s - SF_DROP_START_S
  if td < SF_DROP_DUR_S:
    return -SF_DROP_DV_MPS * (td / SF_DROP_DUR_S)
  if td < 2.0 * SF_DROP_DUR_S:
    return -SF_DROP_DV_MPS + SF_DROP_DV_MPS * ((td - SF_DROP_DUR_S) / SF_DROP_DUR_S)
  return None


def _sf_steps() -> list[StepInput]:
  steps: list[StepInput] = []
  for i in range(int(round(SF_DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    in_rollover = SF_DROP_START_S <= t_s < SF_DROP_START_S + 2.0 * SF_DROP_DUR_S
    lead = LeadDirective(
      status=True,
      v_lead_mps=SF_LEAD_TRUE_V_MPS,
      model_prob_target=SF_LEAD_PROB,
      d_rel_override_m=SF_GAP0_M if i == 0 else None,
      # Pin the reported gap so the rollover is a pure vLeadK/vRel artifact with the
      # distance evidence unchanged (road: dRel > 75 m barely moved).
      measured_d_rel_m=SF_GAP0_M,
      measured_v_rel_mps=_sf_measured_vrel(t_s),
      acquisition_reset=i == 0,
    )
    steps.append(StepInput(
      t_s=t_s, cruise_speed_mps=SF_CRUISE_MPS, lead_one=lead,
      note="vLeadK rollover on a far cruise lead" if in_rollover else "cruise-accel toward set speed, far lead",
    ))
  return steps


# ---- EDGE1 scenario (road 200-13 phase-1) --------------------------------------
E1_DURATION_S = 8.0
E1_EGO_V0_MPS = 24.0             # road ego ~24 m/s
E1_CRUISE_MPS = 28.0            # set speed above ego: cruise accelerating (~+0.85)
E1_LEAD_V_MPS = 22.0           # slower lead (negative vRel; ego closing)
E1_REVEAL_T_S = 2.0
E1_GAP_AT_REVEAL_M = 38.0      # already INSIDE the desired follow distance; THW ~1.53 s (road eroded 1.55)
E1_LEAD_PROB = 0.98

# Desired-behavior bound (task spec): a valid lead inside the desired follow
# distance on a negative-vRel trend must cap positive aTarget (road: +0.49 spent).
E1_MAX_POSITIVE_A_INSIDE_MPS2 = 0.1
E1_MIN_TRUE_GAP_FLOOR_M = 20.0
E1_T_FOLLOW_S = 1.45          # EV6 standard-personality headway used for the df test


def _e1_steps() -> list[StepInput]:
  steps: list[StepInput] = []
  for i in range(int(round(E1_DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    if t_s < E1_REVEAL_T_S:
      lead = LeadDirective()
      note = "cruise-accel below set speed, no lead"
    else:
      reveal_now = abs(t_s - E1_REVEAL_T_S) < (DT_MDL * 0.5)
      lead = LeadDirective(
        status=True,
        v_lead_mps=E1_LEAD_V_MPS,
        model_prob_target=E1_LEAD_PROB,
        d_rel_override_m=E1_GAP_AT_REVEAL_M if reveal_now else None,
        acquisition_reset=reveal_now,
      )
      note = "sub-target lead resolves inside desired follow distance"
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=E1_CRUISE_MPS, lead_one=lead, note=note))
  return steps


@functools.lru_cache(maxsize=1)
def _signflip_vehicle_config():
  # Isolate CD6. The road/device snapshot deliberately disables EDGE1 after it
  # misfired on ghost model leads; enabling it here makes the synthetic vRel
  # rollover trigger that separate positive-accel cap before the source flips.
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2": "10.0",
    "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.4",
  })


@functools.lru_cache(maxsize=1)
def _edge1_vehicle_config():
  # The device snapshot may carry the driver's live deltas (2026-07-04:
  # HandoffInsideDfPositiveCapMps2=10.0 - EDGE1 cap deliberately DISABLED on
  # the car after it misfired on ghost model leads). This test validates the
  # EDGE1 mechanism itself, so pin the cap at its committed default.
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2": "0.1",
    # Calibration tune of this file's frame-delta/divergence bounds (the
    # device snapshot may carry the driver's live VRelTauS=0.60 delta).
    "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.4",
  })


@functools.lru_cache(maxsize=1)
def _run_signflip() -> SimulationResult:
  return run_harness(
    vehicle_config=_signflip_vehicle_config(),
    scenario_name="handoff_signflip",
    steps=_sf_steps(),
    initial_speed_mps=SF_EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


@functools.lru_cache(maxsize=1)
def _run_edge1() -> SimulationResult:
  return run_harness(
    vehicle_config=_edge1_vehicle_config(),
    scenario_name="handoff_edge1_cruise_into_lead",
    steps=_e1_steps(),
    initial_speed_mps=E1_EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _planner_rows(result: SimulationResult) -> list[dict]:
  # One row per 20 Hz planner step (the harness logs every control tick; the
  # planner updates once per 5 ticks at DT_MDL).
  return result.trace[::5]


def _source_transition_frames(rows: list[dict]) -> list[tuple[dict, dict]]:
  return [(a, b) for a, b in zip(rows[:-1], rows[1:], strict=False)
          if a["planner_source"] != b["planner_source"]]


def _sf_flip_frame(rows: list[dict]) -> dict:
  """The single frame with the largest one-step |delta aTarget| in the rollover
  neighborhood - the cruise-side dive that precedes/coincides with the source
  flip (road: nothing bounds the first braking frame after the dive)."""
  window = [r for r in rows if SF_DROP_START_S - 0.5 <= r["t_s"] <= SF_DROP_START_S + 2.5]
  worst = None
  worst_delta = 0.0
  for a, b in zip(window[:-1], window[1:], strict=False):
    delta = b["planner_accel_mps2"] - a["planner_accel_mps2"]
    if abs(delta) > abs(worst_delta):
      worst_delta = delta
      worst = (a, b)
  assert worst is not None
  a, b = worst
  return {
    "prev_t_s": a["t_s"],
    "flip_t_s": b["t_s"],
    "prev_a_mps2": a["planner_accel_mps2"],
    "flip_a_mps2": b["planner_accel_mps2"],
    "one_frame_delta_a_mps2": worst_delta,
    "cc_co_divergence_mps2": abs(b["controller_accel_mps2"] - b["longcontrol_accel_mps2"]),
  }


def test_signflip_scenario_wiring() -> None:
  result = _run_signflip()

  # Tici-fidelity loop resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"

  rows = _planner_rows(result)

  # Scenario-validity guard #1: the lead genuinely never closes - the true gap
  # holds well outside the follow envelope the entire run. The negative aTarget
  # dive is therefore an artifact of the vLeadK rollover, not real proximity.
  # Asserted unconditionally (NOT only inside the strict-xfail oracle, where a
  # violation would silently register as an "expected failure").
  assert result.summary["minTrueGapM"] >= SF_MIN_TRUE_GAP_FLOOR_M, (
    f"true gap collapsed to {result.summary['minTrueGapM']:.1f} m - the lead "
    f"actually closed; scenario no longer isolates the vLeadK-rollover handoff")

  # Scenario-validity guard #2: cruise is genuinely accelerating hard just before
  # the rollover (the flip is FROM a positive cruise-accel state, road +0.58).
  pre = [r for r in rows if SF_DROP_START_S - 0.6 <= r["t_s"] < SF_DROP_START_S]
  assert pre and pre[-1]["planner_source"] == "cruise"
  assert pre[-1]["planner_accel_mps2"] >= 0.3, (
    f"cruise not accelerating into the rollover (aTarget {pre[-1]['planner_accel_mps2']:.3f}); "
    f"scenario does not set up the sign flip")

  # Scenario-validity guard #3: a cruise<->lead0 source transition actually occurs
  # at the rollover (the handoff whose discontinuity CD6 is about). Both sources
  # must appear, including a cruise->lead0 hand.
  transitions = _source_transition_frames(rows)
  sources_seen = {r["planner_source"] for r in rows}
  assert "cruise" in sources_seen and "lead0" in sources_seen, (
    f"no cruise<->lead0 transition occurred (sources seen: {sources_seen})")
  assert any(a["planner_source"] == "cruise" and b["planner_source"] == "lead0"
             for a, b in transitions), "cruise->lead0 handoff never happened"

  # Scenario-validity guard #4: the rollover really is a vLeadK artifact - the
  # published vRel dives (the tracker's EMA reports the lead braking) while the
  # published dRel stays essentially pinned (road: dRel > 75 m barely moved).
  roll = [r for r in rows if SF_DROP_START_S <= r["t_s"] <= SF_DROP_START_S + 2.0 * SF_DROP_DUR_S
          and r["lead_one_published_v_rel_mps"] is not None]
  assert roll and min(r["lead_one_published_v_rel_mps"] for r in roll) <= -1.0, (
    "published vRel never dived - the vLeadK rollover did not reach the tracker")
  pub_drels = [r["lead_one_published_d_rel_m"] for r in roll if r["lead_one_published_d_rel_m"] is not None]
  assert pub_drels and (max(pub_drels) - min(pub_drels)) < 30.0, (
    "published dRel moved too much during the rollover; not a pure vLeadK artifact")


def test_signflip_one_frame_delta_and_divergence_bounded() -> None:
  result = _run_signflip()
  flip = _sf_flip_frame(_planner_rows(result))

  one_frame_ok = abs(flip["one_frame_delta_a_mps2"]) <= SF_MAX_ONE_FRAME_DELTA_A_MPS2
  divergence_ok = flip["cc_co_divergence_mps2"] <= SF_MAX_CC_CO_DIVERGENCE_MPS2

  physics = (
    f"cruise<->lead0 handoff on a vLeadK rollover (lead at set speed {SF_LEAD_TRUE_V_MPS} m/s "
    f"~{SF_GAP0_M} m ahead, perception vRel dips {SF_DROP_DV_MPS} m/s over {SF_DROP_DUR_S}s at t={SF_DROP_START_S}s):\n"
    f"  cruise aTarget just before the flip: {flip['prev_a_mps2']:+.3f} m/s^2 at t={flip['prev_t_s']:.3f}s "
    f"(road +0.58 under the cruise pull)\n"
    f"  one-frame aTarget across the flip: {flip['prev_a_mps2']:+.3f} -> {flip['flip_a_mps2']:+.3f} m/s^2 "
    f"(delta {flip['one_frame_delta_a_mps2']:+.3f}, bound |.| <= {SF_MAX_ONE_FRAME_DELTA_A_MPS2}; road 0.8-2.7)\n"
    f"  cc vs co divergence at the flip frame: {flip['cc_co_divergence_mps2']:.3f} m/s^2 "
    f"(bound <= {SF_MAX_CC_CO_DIVERGENCE_MPS2}; road 1.14-1.18 - the 50 Hz device EMA lags the planner dive)\n"
    f"  min true gap over the run: {result.summary['minTrueGapM']:.1f} m (the lead never actually closed)"
  )
  assert one_frame_ok and divergence_ok, physics


def test_edge1_scenario_wiring() -> None:
  result = _run_edge1()

  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"

  rows = _planner_rows(result)

  # Scenario-validity guard: no crash - the lead stays a safe distance ahead the
  # whole run (asserted unconditionally, outside the strict-xfail oracle).
  assert result.summary["minTrueGapM"] >= E1_MIN_TRUE_GAP_FLOOR_M, (
    f"true gap collapsed to {result.summary['minTrueGapM']:.1f} m")

  # The lead resolves already inside the desired follow distance with a negative
  # (closing) vRel trend, and the cruise source is the one in control at the
  # reveal frame (the handoff boundary the EDGE1 defect lives on).
  reveal = [r for r in rows if E1_REVEAL_T_S <= r["t_s"] < E1_REVEAL_T_S + 0.15
            and r["true_min_gap_m"] is not None]
  assert reveal, "lead never resolved at the reveal"
  first = reveal[0]
  df = desired_follow_distance(first["v_ego_true_mps"], E1_LEAD_V_MPS, t_follow=E1_T_FOLLOW_S)
  assert first["true_min_gap_m"] < df, (
    f"lead resolved OUTSIDE the desired follow distance (gap {first['true_min_gap_m']:.1f} m vs df {df:.1f} m); "
    f"scenario does not exercise the inside-df EDGE1 direction")
  assert first["v_ego_true_mps"] - E1_LEAD_V_MPS > 0.2, "vRel not closing (no negative trend)"
  assert first["planner_source"] == "cruise", (
    f"cruise did not own the reveal frame (source {first['planner_source']}); "
    f"scenario does not exercise the cruise-into-lead boundary")


def test_edge1_positive_accel_capped_inside_desired_follow() -> None:
  result = _run_edge1()
  rows = _planner_rows(result)

  # The worst positive aTarget spent by the cruise source while a valid lead is
  # inside the desired follow distance on a closing (negative-vRel) trend.
  worst_positive = None
  worst_row = None
  for r in rows:
    gap = r["true_min_gap_m"]
    v = r["v_ego_true_mps"]
    if gap is None or v <= 1.0 or r["t_s"] < E1_REVEAL_T_S:
      continue
    df = desired_follow_distance(v, E1_LEAD_V_MPS, t_follow=E1_T_FOLLOW_S)
    closing = v - E1_LEAD_V_MPS
    if gap < df and closing > 0.1:
      if worst_positive is None or r["planner_accel_mps2"] > worst_positive:
        worst_positive = r["planner_accel_mps2"]
        worst_row = r

  assert worst_positive is not None, "no inside-df closing frame was observed"

  physics = (
    f"sub-target lead ({E1_LEAD_V_MPS} m/s) resolves inside the desired follow distance while cruise owns "
    f"(ego {E1_EGO_V0_MPS} m/s toward set speed {E1_CRUISE_MPS} m/s):\n"
    f"  worst positive aTarget inside df on a closing trend: {worst_positive:+.3f} m/s^2 at "
    f"t={worst_row['t_s']:.3f}s, source={worst_row['planner_source']} "
    f"(bound <= {E1_MAX_POSITIVE_A_INSIDE_MPS2}; road +0.49 while THW eroded 1.94->1.55 s)\n"
    f"  true gap / THW there: {worst_row['true_min_gap_m']:.1f} m / "
    f"{worst_row['true_min_gap_m'] / worst_row['v_ego_true_mps']:.2f} s\n"
    f"  min true gap over the run: {result.summary['minTrueGapM']:.1f} m"
  )
  assert worst_positive <= E1_MAX_POSITIVE_A_INSIDE_MPS2, physics
