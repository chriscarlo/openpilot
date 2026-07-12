"""Repro: NEAR_COLLISION #2 (road unit 200-15-17 TAP 1) - lead-decel-blind
brake-release floor vs a lead braking to a stop.

Field evidence (road_forensics.json unit 200-15-17-stop-cluster, TAP 1 at
seg 00000200--8cbf2c9481--15@51.87, driver stomp; one of the drive's two
driver-intervention NEAR_COLLISIONs):

CD1 (brake-release floor is lead-decel-blind): get_lead_brake_release_accel_floor
(longitudinal_planner.py :86-256, applied at :509-510 as
`output_a_target = max(output_a_target, floor)`, introduced by 1c976dda3)
computes its release floor from the INSTANTANEOUS closing speed only and never
projects the lead's own deceleration forward. On this tap the ego was following
at 6.7 m/s (steady THW ~1.77, bank target ~1.45) when the lead began braking to
a full stop at aLeadK -0.42..-0.63 m/s^2 from ~6.15 m/s. Every release guard was
satisfied the whole time: the published/tracker aLeadK stayed in the
-0.42..-0.63 band and NEVER crossed the -0.75 veto
(LeadBrakeReleaseLeadDecelMinMps2), and the closing rate at the floor-active
moment (0.57-0.71 m/s) sat under the 0.75 m/s near-target gate
(LeadBrakeReleaseNearTargetMaxClosingMps), with vEgo > the 5.0 m/s min-speed
gate. So the floor sat in its `closing_to_target` (floor = -closing^2/(2*gap_err))
and `near_target` (floor = -0.05) branches and clipped the MPC's ramping brake
request for ~3.5 s while the lead decelerated toward a stop. Realized decel ran
mean -0.20 m/s^2 through the first half vs the -0.53 m/s^2 a constant-decel plan
from brake onset would have needed to stop ~4 m behind; the planner peaked only
-2.40 and the driver stomped to aEgo -5.53, leaving a min gap of 1.13 m.

Scenario synthesis (all numbers road-derived, see _measure/physics for the
side-by-side): steady follow at 6.7 m/s (road ego 6.67 at the tap) at THW ~1.46
(road steady 1.76 vs bank target 1.45), gap 11.4 m; the lead then brakes at a
true/reported -0.5 m/s^2 (road aLeadK band -0.42..-0.63, INSIDE the -0.75 veto)
from 6.15 m/s to a full stop. Injection point: the harness lead directive's
a_lead_k_mps2 flows through the radard stage's 0.6 s accel EMA and is published
as aLeadK ~-0.50 at peak (road-measured -0.42..-0.63), which the release floor
reads verbatim as `lead.aLeadK` - i.e. exactly the shallow decel that keeps the
-0.75 lead-decel veto from ever tripping.

The rollback-knob twin isolates the mechanism: raising
Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelMinMps2 from the shipped -0.75 to
-0.20 (the documented interim mitigation direction) trips the lead-decel veto the
moment filtered aLeadK crosses -0.20, so the floor cannot bind after the veto is
actually satisfied; the MPC then keeps min gap >= 4.0 m. The prior test counted
the EMA ramp between 0 and -0.20 as though the veto had already fired.

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

DURATION_S = 17.0
EGO_V0_MPS = 6.7                 # road ego 6.67 m/s at the tap
CRUISE_SPEED_MPS = 20.0          # road vCruise 72 kph -> never binding at this speed
LEAD_V0_MPS = 6.15               # road vLead 6.15 at brake onset
INITIAL_GAP_M = 11.4             # steady THW ~1.46 (road 1.76 vs bank target 1.45)
LEAD_MODEL_PROB = 0.99           # road: continuous vision track through the tap

# Lead brakes to a FULL stop at a shallow decel that stays INSIDE the -0.75 veto
# band (road aLeadK -0.42..-0.63). The radard 0.6 s accel EMA publishes ~-0.50
# at peak - never crossing LeadBrakeReleaseLeadDecelMinMps2.
TRUE_DECEL_MPS2 = -0.5
DECEL_START_S = 3.0
LEAD_STOP_V_MPS = 0.0

# Rollback knob (the floor's documented lead-decel veto): the shipped value is
# -0.75 (floor stays active on this -0.5 lead). Raising it toward zero trips the
# veto and disables the floor once filtered aLeadK reaches that threshold.
# -0.20 gives a clean twin while retaining the real radard EMA crossing delay.
FLOOR_VETO_ROLLBACK_MPS2 = -0.20
SHIPPED_FLOOR_VETO_MPS2 = -0.75  # LeadBrakeReleaseLeadDecelMinMps2 default

# CD1 fix knob (the projection gain that adds the lead's own decel to the
# required ego decel). Default 1.0 = exact relative-frame requirement (the fix);
# 0.0 = pre-fix instantaneous-closing floor (rollback sentinel). This is the
# knob the CD1 fix introduced, exercised at BOTH the -0.75 shipped veto so the
# floor stays armed on the decelerating lead in both twins — only the projection
# term differs. Per the SAFETY RULE every new threshold gets an oracled rollback.
FIX_PROJECT_GAIN = 1.0
ROLLBACK_PROJECT_GAIN = 0.0

# Desired-behavior bounds (task spec, derived from the road kinematics).
MIN_TRUE_GAP_FLOOR_M = 4.0       # road min gap collapsed to 1.13 m at the stomp
NEAR_TARGET_MAX_CLOSING_MPS = 0.75  # LeadBrakeReleaseNearTargetMaxClosingMps
LEAD_DECEL_VETO_BAND = (-0.75, 0.0)  # road aLeadK -0.42..-0.63 sat inside this


def _lead_speed(t_s: float) -> float:
  if t_s < DECEL_START_S:
    return LEAD_V0_MPS
  return max(LEAD_STOP_V_MPS, LEAD_V0_MPS + TRUE_DECEL_MPS2 * (t_s - DECEL_START_S))


def _build_steps() -> list[StepInput]:
  steps = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    v_lead = _lead_speed(t_s)
    braking = DECEL_START_S <= t_s and v_lead > 0.0
    lead = LeadDirective(
      status=True,
      v_lead_mps=v_lead,
      model_prob_target=LEAD_MODEL_PROB,
      # Shallow reported lead accel (road aLeadK -0.42..-0.63): the radard 0.6 s
      # EMA publishes ~-0.50 at peak, verbatim into the lead the release floor
      # reads as lead.aLeadK - never crossing the -0.75 veto.
      a_lead_k_mps2=(TRUE_DECEL_MPS2 if braking else 0.0),
      d_rel_override_m=INITIAL_GAP_M if i == 0 else None,
      acquisition_reset=i == 0,
    )
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead,
                           note="lead brakes -0.5 to a stop (inside -0.75 veto)" if braking else "steady follow"))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config(floor_veto_mps2: float):
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelMinMps2": f"{floor_veto_mps2:g}",
  })


@functools.lru_cache(maxsize=2)
def _run(floor_veto_mps2: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(floor_veto_mps2),
    scenario_name=f"release_floor_stopping_lead_veto_{floor_veto_mps2:g}",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


@functools.lru_cache(maxsize=1)
def _vehicle_config_project_gain(project_gain: float):
  # Hold the lead-decel veto at the SHIPPED -0.75 so the floor stays armed on the
  # decelerating lead in both twins; only the CD1 projection gain differs.
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelMinMps2": f"{SHIPPED_FLOOR_VETO_MPS2:g}",
    "Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelProjectGain": f"{project_gain:g}",
  })


@functools.lru_cache(maxsize=2)
def _run_project_gain(project_gain: float) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config_project_gain(project_gain),
    scenario_name=f"release_floor_stopping_lead_projgain_{project_gain:g}",
    steps=_build_steps(),
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


def _floor_active_rows(trace: list[dict]) -> list[dict]:
  return [row for row in trace if row["planner_lead_brake_release_debug"].get("active")]


def _floor_clip_rows(trace: list[dict]) -> list[dict]:
  """Ticks where the release floor is the binding constraint on the planner
  output (planner accel == the negative floor, i.e. max(mpc, floor) == floor)."""
  clipped = []
  for row in trace:
    floor = row["planner_lead_brake_release_floor_mps2"]
    if floor < -1e-6 and abs(row["planner_accel_mps2"] - floor) < 1e-6:
      clipped.append(row)
  return clipped


def _floor_clip_on_decel_above_demand(shipped: list[dict], rollback: list[dict]) -> list[dict]:
  """Ticks where, while the lead is decelerating, the floor clips the shipped
  planner brake ABOVE (shallower than) the unclamped-MPC demand. The rollback
  twin (floor vetoed) is the unclamped-demand proxy: identical true kinematics,
  only the floor disabled."""
  out = []
  for row_s, row_r in zip(shipped, rollback, strict=True):
    floor = row_s["planner_lead_brake_release_floor_mps2"]
    lead_accel = row_s["planner_lead_brake_release_debug"].get("lead_accel_mps2") or 0.0
    binding = floor < -1e-6 and abs(row_s["planner_accel_mps2"] - floor) < 1e-6
    if binding and lead_accel < 0.0 and row_s["planner_accel_mps2"] > row_r["planner_accel_mps2"] + 1e-6:
      out.append(row_s)
  return out


def _measure(result: SimulationResult) -> dict:
  trace = result.trace
  thws = [thw for row in trace if (thw := _thw_s(row)) is not None]
  pub_alead = [row["lead_one_published_a_lead_k_mps2"] for row in trace
               if row["lead_one_published_a_lead_k_mps2"] is not None
               and row["t_s"] >= DECEL_START_S]
  floor_active = _floor_active_rows(trace)
  return {
    "min_thw_s": min(thws),
    "min_true_gap_m": result.summary["minTrueGapM"],
    "peak_planner_brake_mps2": result.summary["peakPlannerBrakeMps2"],
    "published_a_lead_k_peak_mps2": min(pub_alead) if pub_alead else None,
    "floor_active_n": len(floor_active),
    "floor_clip_n": len(_floor_clip_rows(trace)),
  }


def test_release_floor_scenario_wiring() -> None:
  shipped = _run(SHIPPED_FLOOR_VETO_MPS2)
  rollback = _run(FLOOR_VETO_ROLLBACK_MPS2)

  # Tici-fidelity loop resolved as intended.
  for result in (shipped, rollback):
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"

  # Corroborated continuous track (road: continuous vision track through the
  # tap): the published lead exists on every settled planner step.
  settled = [row for row in shipped.trace if row["t_s"] >= 1.0]
  assert all(row["lead_one_published_d_rel_m"] is not None for row in settled)
  assert all(row["lead_one_model_prob"] >= 0.9 for row in settled)

  # SCENARIO-VALIDITY GUARD #1: the release floor actually activates on this
  # decelerating lead in the shipped configuration - proven via its own debug
  # trace signature (planner.lead_brake_release_debug.active). If a future change
  # stops arming the floor here, this guard goes RED so the behavioral xfail
  # below can never silently pass on a scenario that no longer exercises CD1.
  shipped_active = _floor_active_rows(shipped.trace)
  assert shipped_active, "release floor never activated in the shipped run (scenario no longer exercises CD1)"
  # ...and it reaches one of the exact CD1 projection branches while the lead is
  # visibly decelerating. With real radard timing, the filtered aLead crosses
  # -0.2 just after gap_error enters near_target; requiring closing_to_target
  # specifically made the old validity guard impossible despite exercising the
  # same lead-decel projection term.
  cd1_projection_active = [row for row in shipped_active
                           if row["t_s"] >= DECEL_START_S
                           and row["planner_lead_brake_release_debug"].get("reason") in ("closing_to_target", "near_target")
                           and (row["planner_lead_brake_release_debug"].get("lead_accel_mps2") or 0.0) < -0.2]
  assert cd1_projection_active, "release floor never armed in a CD1 lead-decel projection branch"

  # SCENARIO-VALIDITY GUARD #2: every release guard the road reported as
  # satisfied is satisfied here too. The lead-decel veto NEVER trips in the
  # shipped run: the published aLeadK stays inside the -0.75 band the whole time
  # (road -0.42..-0.63), which is the whole reason the floor stays armed.
  decel_active = [row for row in shipped_active if row["t_s"] >= DECEL_START_S
                  and (row["planner_lead_brake_release_debug"].get("lead_accel_mps2") or 0.0) < -0.2]
  assert decel_active, "floor never active while the lead was visibly decelerating"
  for row in decel_active:
    dbg = row["planner_lead_brake_release_debug"]
    lead_accel = dbg.get("lead_accel_mps2")
    # veto not tripped: lead_accel sits inside (above) the shipped -0.75 veto band
    assert LEAD_DECEL_VETO_BAND[0] < lead_accel < LEAD_DECEL_VETO_BAND[1], (
      f"lead_accel {lead_accel:.3f} left the road-measured veto band {LEAD_DECEL_VETO_BAND} at t={row['t_s']:.2f}s"
    )
    # never dropped into the veto/deficit branches while it clipped the MPC:
    # the floor sits in the closing_to_target / near_target branches.
    assert dbg.get("reason") in ("closing_to_target", "near_target", "gap_recovered", "projected_recovery")

  # The rollback twin, by contrast, trips the veto the instant the lead brakes:
  # the floor never activates on the decelerating lead there.
  rollback_decel_active = [row for row in _floor_active_rows(rollback.trace)
                           if row["t_s"] >= DECEL_START_S
                           and (row["planner_lead_brake_release_debug"].get("lead_accel_mps2") or 0.0) < -0.2]
  assert not rollback_decel_active, "rollback knob failed to veto the floor on the decelerating lead"

  # Matched pair: identical true kinematics, only the floor's veto knob differs.
  for r_ship, r_roll in zip(shipped.trace, rollback.trace, strict=True):
    assert r_ship["active_lead_speed_mps"] == pytest.approx(r_roll["active_lead_speed_mps"], abs=1e-9)

  # Published aLeadK stayed shallow (road-measured -0.42..-0.63, never <= -0.75).
  m = _measure(shipped)
  assert LEAD_DECEL_VETO_BAND[0] < m["published_a_lead_k_peak_mps2"] < LEAD_DECEL_VETO_BAND[1]


# CD1 FIXED (was strict-xfail): get_lead_brake_release_accel_floor now adds the
# lead's own deceleration magnitude to the required ego decel in its closing and
# near-target branches (LeadBrakeReleaseLeadDecelProjectGain, default 1.0), so on
# a lead braking to a stop at -0.42..-0.63 m/s^2 (inside the -0.75 veto) with
# 0.57-0.71 m/s closing (inside the 0.75 near-target gate) the floor is sized
# deeper than the MPC's ramping brake and no longer clips it. Marker removed per
# the oracle discipline: the fix flipped this XPASS -> plain green.
def test_control_release_floor_does_not_clip_stopping_lead_brake() -> None:
  shipped = _run(SHIPPED_FLOOR_VETO_MPS2)
  rollback = _run(FLOOR_VETO_ROLLBACK_MPS2)
  m_ship = _measure(shipped)
  m_roll = _measure(rollback)

  # Behavior #1: the true gap must not collapse below the road-derived floor.
  gap_ok = m_ship["min_true_gap_m"] >= MIN_TRUE_GAP_FLOOR_M
  # Behavior #2: the release floor must never clip the planner brake above the
  # unclamped-MPC demand while the lead is decelerating. The rollback twin (floor
  # vetoed) is the unclamped demand: identical kinematics, floor off.
  clips = _floor_clip_on_decel_above_demand(shipped.trace, rollback.trace)
  clip_ok = len(clips) == 0

  first_clip = clips[0] if clips else None
  physics = (
    f"lead brakes {TRUE_DECEL_MPS2} m/s^2 (published aLeadK peak "
    f"{m_ship['published_a_lead_k_peak_mps2']:+.2f}, road -0.42..-0.63, veto {SHIPPED_FLOOR_VETO_MPS2}) to a stop "
    f"from t={DECEL_START_S}s; shipped floor-active run vs rollback-veto twin "
    f"(LeadBrakeReleaseLeadDecelMinMps2 {SHIPPED_FLOOR_VETO_MPS2} -> {FLOOR_VETO_ROLLBACK_MPS2}):\n"
    f"  min true gap: shipped {m_ship['min_true_gap_m']:.2f} m (floor {MIN_TRUE_GAP_FLOOR_M} m; road collapsed to "
    f"1.13 m at the stomp), rollback {m_roll['min_true_gap_m']:.2f} m\n"
    f"  min THW: shipped {m_ship['min_thw_s']:.3f} s, rollback {m_roll['min_thw_s']:.3f} s "
    f"(road steady 1.76 -> 0.93 at the tap)\n"
    f"  peak planner brake: shipped {m_ship['peak_planner_brake_mps2']:.2f} m/s^2 "
    f"(road planner peak -2.40, ideal constant -0.53 from onset), rollback {m_roll['peak_planner_brake_mps2']:.2f}\n"
    f"  release-floor active ticks: shipped {m_ship['floor_active_n']}, rollback {m_roll['floor_active_n']}\n"
    f"  floor-clip-above-demand ticks (while lead decel): {len(clips)}"
    + (f", first at t={first_clip['t_s']:.2f}s (planner {first_clip['planner_accel_mps2']:+.3f} vs floor "
       f"{first_clip['planner_lead_brake_release_floor_mps2']:+.3f}, "
       f"leadacc {first_clip['planner_lead_brake_release_debug'].get('lead_accel_mps2'):+.3f})" if first_clip else "")
  )
  assert gap_ok and clip_ok, physics


def test_project_gain_knob_fix_vs_rollback() -> None:
  """CD1 projection-gain oracle at the current post-M1 composition.

  The `test_rollback_knob_restores_unclamped_braking` twin above exercises the
  OLDER LeadBrakeReleaseLeadDecelMinMps2 veto (disables the floor entirely). The
  SAFETY RULE requires EVERY new threshold to have an oracled rollback knob, so
  this test exercises the CD1 fix's own knob directly:
  LeadBrakeReleaseLeadDecelProjectGain at the shipped -0.75 veto (floor stays
  ARMED on the decelerating lead in both twins).

    gain = 1.0 (the fix): the lead's own decel is added to the required ego decel,
      sizing the floor DEEPER than the MPC's ramping brake, so it never clips the
      brake above the demand and the gap holds >= 4.0 m.
    gain = 0.0 (rollback sentinel): the pre-fix instantaneous-closing floor
      candidate is restored bit-for-bit. The later M1 threat ceiling retains
      final authority, but the shallower candidate still changes the integrated
      trajectory and collapses the gap below 4 m.

  The later M1 threat ceiling now correctly wins over this release floor on a
  braking lead, so final-output equality with the floor is no longer a valid
  binding oracle. Measure the floor candidate itself, then retain the physical
  minimum-gap comparison between matched gain twins."""
  fix = _run_project_gain(FIX_PROJECT_GAIN)
  rollback = _run_project_gain(ROLLBACK_PROJECT_GAIN)
  m_fix = _measure(fix)
  m_roll = _measure(rollback)

  # Matched twins: identical true kinematics, only the projection gain differs.
  for r_fix, r_roll in zip(fix.trace, rollback.trace, strict=True):
    assert r_fix["active_lead_speed_mps"] == pytest.approx(r_roll["active_lead_speed_mps"], abs=1e-9)

  # In BOTH twins the floor is armed on the decelerating lead (the -0.75 veto is
  # held, not tripped) — this is what makes the gain the sole difference.
  for result in (fix, rollback):
    decel_active = [row for row in _floor_active_rows(result.trace)
                    if row["t_s"] >= DECEL_START_S
                    and (row["planner_lead_brake_release_debug"].get("lead_accel_mps2") or 0.0) < -0.2]
    assert decel_active, "floor should stay armed on the decelerating lead at the shipped -0.75 veto"

  projected_rows = []
  for row_fix, row_roll in zip(fix.trace, rollback.trace, strict=True):
    dbg_fix = row_fix["planner_lead_brake_release_debug"]
    dbg_roll = row_roll["planner_lead_brake_release_debug"]
    if (dbg_fix.get("active") and dbg_roll.get("active") and
        (dbg_fix.get("lead_accel_mps2") or 0.0) < -0.2):
      projected_rows.append((row_fix, row_roll))
  assert projected_rows, "projection-gain twins never reached the decelerating-lead floor branch"

  floor_separation = max(
    row_roll["planner_lead_brake_release_floor_mps2"] - row_fix["planner_lead_brake_release_floor_mps2"]
    for row_fix, row_roll in projected_rows
  )

  physics = (
    f"CD1 projection-gain knob (LeadBrakeReleaseLeadDecelProjectGain) at the shipped "
    f"{SHIPPED_FLOOR_VETO_MPS2} veto, lead brakes {TRUE_DECEL_MPS2} m/s^2 to a stop:\n"
    f"  gain={FIX_PROJECT_GAIN} (fix):      min gap {m_fix['min_true_gap_m']:.2f} m\n"
    f"  gain={ROLLBACK_PROJECT_GAIN} (rollback): min gap {m_roll['min_true_gap_m']:.2f} m "
    f"(pre-fix pathology; road collapsed to 1.13 m)\n"
    f"  maximum rollback-minus-fix floor separation: {floor_separation:.3f} m/s^2"
  )

  # The fix adds the lead-decel term to the candidate and holds the true gap.
  assert floor_separation > 0.15, physics
  assert m_fix["min_true_gap_m"] >= MIN_TRUE_GAP_FLOOR_M, physics

  # The rollback sentinel removes that projected decel and restores the smaller gap.
  assert m_roll["min_true_gap_m"] < MIN_TRUE_GAP_FLOOR_M, physics

  # And the fix is strictly safer than its own rollback: deeper minimum gap.
  assert m_fix["min_true_gap_m"] > m_roll["min_true_gap_m"], physics


def test_rollback_knob_restores_unclamped_braking() -> None:
  """Rollback-knob twin (green today; must stay green): raising the floor's
  lead-decel veto LeadBrakeReleaseLeadDecelMinMps2 from the shipped -0.75 to
  -0.20 disables the floor on the visibly-braking lead - the documented interim
  mitigation. With the floor out of the way the MPC brakes unclamped, keeping min
  true gap >= 4.0 m with zero floor-clip ticks on the decelerating lead. This
  encodes the rollback path and proves the collapse is owned by the floor, not
  the MPC solve."""
  rollback = _run(FLOOR_VETO_ROLLBACK_MPS2)
  m = _measure(rollback)

  # No floor-clip ticks while the lead decelerates in the rollback run.
  rollback_decel_clips = [row for row in _floor_clip_rows(rollback.trace)
                          if (row["planner_lead_brake_release_debug"].get("lead_accel_mps2") or 0.0)
                          < FLOOR_VETO_ROLLBACK_MPS2]
  assert not rollback_decel_clips, (
    f"rollback run still clipped the brake on a decelerating lead "
    f"(first at t={rollback_decel_clips[0]['t_s']:.2f}s)"
  )
  assert m["min_true_gap_m"] >= MIN_TRUE_GAP_FLOOR_M, (
    f"rollback run min true gap {m['min_true_gap_m']:.2f} m < floor {MIN_TRUE_GAP_FLOOR_M} m "
    f"(the floor is not the sole owner of the collapse)"
  )
