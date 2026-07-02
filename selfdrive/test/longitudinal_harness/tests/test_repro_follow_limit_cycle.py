"""Repro: the steady-follow limit cycle (user seat report, 2026-07-02).

Seat report (2023 Kia EV6 GT, strong regen; verbatim intent): during steady-ish
follow the car would drift too close to the lead, then instead of gradually
easing to coast/mild regen it would SUDDENLY slow to create distance, then STAY
ON THE BRAKE TOO LONG, then fail to re-accelerate to match lead speed before
reaching the desired follow distance, then have to accelerate harder, OVERSHOOT
(gap collapses again), slow again -- repeating 1-2 more cycles before settling.

Measured in the tici-fidelity loop (device controller mode + radard perception,
noise OFF, so every number below is deterministic):

Highway dip (ego=lead=30 m/s, gap settled at target, lead dips 1.5 m/s over
1.5 s and recovers):
  - brake-hold: planner accel stays < -0.15 m/s^2 for 0.95 s AFTER the gap has
    recovered past the MPC's own desired gap while ego is no longer closing
    (realized accel holds < -0.15 for 1.9 s through the actuation chain).
  - late re-accel: ego is still 1.57 m/s slower than the lead after the gap
    regains target; planner re-accel crawls up at ~0.1 m/s^3 to a ~0.32 cap.
  - overshoot: the reclaimed gap then collapses to 3.6 m BELOW target.
Mechanism (measured, not inferred): during the entire brake-hold window the
lead-brake-release path (get_lead_brake_release_accel_floor,
selfdrive/controls/lib/longitudinal_planner.py) returns reason
'brake_authority_deficit' -- its headway-based gap error (computed on the
lag-filtered published dRel, no vRel term) sits at ~-3.0 m against
LeadBrakeReleaseBrakeDeficitMarginM=1.5 -- and once it finally activates its
floor (-0.05) is already above the MPC output, so the release never binds and
the hold is owned by the MPC's own slow unwind (AccelChangeCost=400).

City ease (ego=lead=13.5 m/s, lead eases 0.4 m/s^2 for 3.5 s and resumes):
  - the gap error rings twice (2 zero crossings with amplitude > 15% of
    target) and takes 21.1 s to stay within 10% of target for 5 s.
Attribution: the c55a1758d lag-comp fade (FadeLo=12/FadeHi=18) removes ~75% of
the closing lag compensation at 13.5 m/s; with the fade disabled
(FadeLo=FadeHi=0, the legacy-emulation setting) the identical scenario settles
in 14.9 s with 1 crossing. All six landed mechanism fixes together
(LeadSlowdownKinematicHeadroom / FastClose+OpenRecovery / FcwCorrob /
PhantomLeadDecelHold / fade) change NOTHING else in this regime: the 30 m/s
runs are bit-identical current-vs-legacy under both noise-off and ev6_measured.
"""
from __future__ import annotations

import functools

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

SETTLE_S = 14.0
BRAKE_HOLD_THRESHOLD_MPS2 = -0.15
SETTLE_BAND_FRAC = 0.10
SETTLE_WINDOW_S = 5.0
CYCLE_AMPLITUDE_FRAC = 0.15

# Highway dip bounds (today: brake_hold 0.95 s planner / 1.90 s realized,
# max deficit 1.57 m/s, overshoot -3.61 m).
HW_V0_MPS = 30.0
HW_CRUISE_MPS = 33.0
HW_GAP0_M = 42.0  # equal-speed target at t_follow ~1.2: 1.2*30 + 6
HW_DURATION_S = 50.0
MAX_BRAKE_HOLD_PLANNER_S = 0.4
MAX_BRAKE_HOLD_REALIZED_S = 1.2
MAX_REACCEL_DEFICIT_MPS = 1.0
MIN_OVERSHOOT_GAP_ERR_M = -2.5

# City ease bounds (today: 2 crossings, settle 21.1 s; with the lag-comp fade
# disabled -- the legacy emulation -- the same run measures 1 crossing and
# 14.9 s, so these bounds are demonstrably achievable).
CITY_V0_MPS = 13.5
CITY_CRUISE_MPS = 16.0
CITY_GAP0_M = 22.2
CITY_DURATION_S = 55.0
MAX_CYCLE_COUNT = 1
MAX_SETTLE_S = 16.0

# Safety floor shared by every leg: a fix must not trade ringing for proximity.
MIN_TRUE_GAP_FLOOR_M = {HW_V0_MPS: 30.0, CITY_V0_MPS: 10.0}

# Calm-approach (SUDDEN-slow) leg of the seat report, encoded as always-green
# guards (asserted in the wiring test so they can never hide inside an
# expected failure): for these MILD lead slowdowns (a 1.5 m/s dip / a
# 0.4 m/s^2 ease) the braking onset must stay within calm EV6 regen feel.
# Measured on the fixed tree (noise-off): dip 0.212 m/s^2-per-0.3s worst step
# and -0.476 m/s^2 planner minimum; ease 0.039 and -0.380. Bounds leave
# headroom but trip if a future change converts these mild dips into stabby
# braking. They bound the response to a MILD disturbance only — genuine
# threats are owned by the kinematic ceiling / FCW paths, which these
# scenarios never engage (min true gap stays >= the floors above).
MAX_CALM_ONSET_STEP_MPS2_PER_0P3S = {HW_V0_MPS: 0.35, CITY_V0_MPS: 0.25}
MIN_CALM_PLANNER_ACCEL_MPS2 = {HW_V0_MPS: -0.80, CITY_V0_MPS: -0.65}


def _lead_speed_dip(t_s: float, v0: float) -> float:
  t_d = t_s - SETTLE_S
  if t_d < 0.0:
    return v0
  if t_d < 1.5:
    return v0 - 1.0 * t_d
  if t_d < 2.5:
    return v0 - 1.5
  if t_d < 4.0:
    return v0 - 1.5 + 1.0 * (t_d - 2.5)
  return v0


def _lead_speed_ease(t_s: float, v0: float) -> float:
  t_d = t_s - SETTLE_S
  if t_d < 0.0:
    return v0
  if t_d < 3.5:
    return v0 - 0.4 * t_d
  return min(v0, (v0 - 0.4 * 3.5) + 0.4 * (t_d - 3.5))


def _build_steps(profile, v0: float, gap0: float, cruise: float, duration_s: float) -> list[StepInput]:
  steps: list[StepInput] = []
  for i in range(int(round(duration_s / DT_MDL))):
    t_s = i * DT_MDL
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=cruise,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=profile(t_s, v0),
        model_prob_target=0.95,
        d_rel_override_m=gap0 if i == 0 else None,
        acquisition_reset=i == 0,
      ),
      note="steady follow with a transient lead slowdown",
    ))
  return steps


@functools.lru_cache(maxsize=None)
def _run(case: str) -> SimulationResult:
  profile, v0, gap0, cruise, duration = {
    "highway_dip": (_lead_speed_dip, HW_V0_MPS, HW_GAP0_M, HW_CRUISE_MPS, HW_DURATION_S),
    "city_ease": (_lead_speed_ease, CITY_V0_MPS, CITY_GAP0_M, CITY_CRUISE_MPS, CITY_DURATION_S),
  }[case]
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name=f"follow_limit_cycle_{case}",
    steps=_build_steps(profile, v0, gap0, cruise, duration),
    initial_speed_mps=v0,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _post_rows(result: SimulationResult) -> list[dict]:
  # 20 Hz planner cadence, lead-owned rows after the disturbance begins.
  return [r for r in result.trace[::5]
          if r["t_s"] >= SETTLE_S
          and r["control_true_gap_error_m"] is not None
          and r["control_lead_speed_mps"] is not None]


def _first_recover_t(post: list[dict]) -> float | None:
  below = [r for r in post if r["control_true_gap_error_m"] < 0.0]
  if not below:
    return None
  first_below_t = below[0]["t_s"]
  return next((r["t_s"] for r in post
               if r["t_s"] > first_below_t and r["control_true_gap_error_m"] >= 0.0), None)


def _measure(result: SimulationResult) -> dict[str, float | int | None]:
  post = _post_rows(result)
  recover_t = _first_recover_t(post)

  brake_hold_planner_s = 0.0
  brake_hold_realized_s = 0.0
  reaccel_deficit = None
  overshoot_m = None
  if recover_t is not None:
    for r in post:
      if r["t_s"] < recover_t:
        continue
      closing = r["v_ego_true_mps"] - r["control_lead_speed_mps"]
      if r["control_true_gap_error_m"] >= 0.0 and closing <= 0.0:
        if r["planner_accel_mps2"] < BRAKE_HOLD_THRESHOLD_MPS2:
          brake_hold_planner_s += DT_MDL
        if r["realized_accel_mps2"] < BRAKE_HOLD_THRESHOLD_MPS2:
          brake_hold_realized_s += DT_MDL
    after = [r for r in post if r["t_s"] >= recover_t]
    reaccel_deficit = max(r["control_lead_speed_mps"] - r["v_ego_true_mps"] for r in after)
    overshoot_m = min(r["control_true_gap_error_m"] for r in after)

  # Settle time: first instant after the disturbance from which |gap error|
  # stays inside the 10% band for a full 5 s window.
  settle_t = None
  for r in post:
    window = [q for q in post if r["t_s"] <= q["t_s"] <= r["t_s"] + SETTLE_WINDOW_S]
    if (window and post[-1]["t_s"] >= r["t_s"] + SETTLE_WINDOW_S - 1e-6
        and all(abs(q["control_true_gap_error_m"]) < SETTLE_BAND_FRAC * q["planner_desired_true_gap_m"] for q in window)):
      settle_t = r["t_s"]
      break

  # Cycle count: gap-error zero crossings whose preceding excursion exceeded
  # 15% of the target, counted until settled.
  cycles = 0
  prev_sign = 0
  peak = 0.0
  end_t = settle_t if settle_t is not None else post[-1]["t_s"]
  for r in post:
    if r["t_s"] > end_t:
      break
    err = r["control_true_gap_error_m"]
    sign = 1 if err > 0.0 else -1 if err < 0.0 else 0
    peak = max(peak, abs(err))
    if sign != 0 and prev_sign != 0 and sign != prev_sign:
      if peak > CYCLE_AMPLITUDE_FRAC * r["planner_desired_true_gap_m"]:
        cycles += 1
      peak = abs(err)
    if sign != 0:
      prev_sign = sign

  return {
    "recover_t_s": recover_t,
    "brake_hold_planner_s": round(brake_hold_planner_s, 2),
    "brake_hold_realized_s": round(brake_hold_realized_s, 2),
    "reaccel_deficit_mps": None if reaccel_deficit is None else round(reaccel_deficit, 3),
    "overshoot_gap_err_m": None if overshoot_m is None else round(overshoot_m, 2),
    "cycle_count": cycles,
    "settle_s": None if settle_t is None else round(settle_t - SETTLE_S, 1),
    "min_true_gap_m": round(result.summary["minTrueGapM"], 2),
  }


def _fmt(metrics: dict) -> str:
  return " ".join(f"{k}={v}" for k, v in metrics.items())


def test_follow_limit_cycle_scenario_wiring() -> None:
  for case, v0 in (("highway_dip", HW_V0_MPS), ("city_ease", CITY_V0_MPS)):
    result = _run(case)

    # Tici-fidelity loop resolved as intended.
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"
    assert result.vehicle["noiseProfile"] == "off"

    # Safety floor, asserted unconditionally (NOT only inside the strict-xfail
    # behavior oracles, where a violation would silently register as an
    # "expected failure"): no interim change may trade ringing for proximity.
    assert result.summary["minTrueGapM"] >= MIN_TRUE_GAP_FLOOR_M[v0], (
      f"{case}: min true gap {result.summary['minTrueGapM']:.2f} m breached the "
      f"safety floor {MIN_TRUE_GAP_FLOOR_M[v0]} m")

    # Calm-approach (SUDDEN-slow) leg: the braking onset for this MILD lead
    # slowdown must stay within calm EV6 regen feel — no stab, no deep brake.
    post = _post_rows(result)
    below = [r for r in post if r["control_true_gap_error_m"] < 0.0]
    if below:
      first_below_t = below[0]["t_s"]
      recover_t = next((r["t_s"] for r in post
                        if r["t_s"] > first_below_t and r["control_true_gap_error_m"] >= 0.0),
                       post[-1]["t_s"])
      seg = [r for r in post if first_below_t <= r["t_s"] <= recover_t]
      min_accel = min(r["planner_accel_mps2"] for r in seg)
      worst_step = 0.0
      for i, r in enumerate(seg):
        for q in seg[i + 1:]:
          if q["t_s"] - r["t_s"] > 0.3 + 1e-6:
            break
          worst_step = max(worst_step, r["planner_accel_mps2"] - q["planner_accel_mps2"])
      assert worst_step <= MAX_CALM_ONSET_STEP_MPS2_PER_0P3S[v0], (
        f"{case}: braking onset step {worst_step:.3f} m/s^2 per 0.3 s exceeds the calm bound "
        f"{MAX_CALM_ONSET_STEP_MPS2_PER_0P3S[v0]} for a mild lead slowdown")
      assert min_accel >= MIN_CALM_PLANNER_ACCEL_MPS2[v0], (
        f"{case}: planner decel {min_accel:.3f} m/s^2 exceeds the calm bound "
        f"{MIN_CALM_PLANNER_ACCEL_MPS2[v0]} for a mild lead slowdown")

    # The follow is genuinely settled at target when the disturbance begins and
    # the lead stays in MPC control throughout.
    pre = [r for r in result.trace[::5]
           if 10.0 <= r["t_s"] < SETTLE_S and r["control_true_gap_error_m"] is not None]
    assert pre, f"{case}: lead never owned before the disturbance"
    assert all(abs(r["control_true_gap_error_m"]) < 0.15 * r["planner_desired_true_gap_m"] for r in pre), (
      f"{case}: gap not settled at target before the disturbance")
    post = _post_rows(result)
    assert post and post[-1]["t_s"] > SETTLE_S + 20.0, f"{case}: lost lead ownership after the disturbance"

    # Mechanism pin for the brake-hold leg: today the lead-brake-release floor
    # never binds during hold frames (it is gated out as brake_authority_deficit
    # or already above the MPC output). Vacuously true once the hold is fixed.
    recover_t = _first_recover_t(post)
    if recover_t is not None:
      hold_rows = [r for r in post
                   if r["t_s"] >= recover_t
                   and r["control_true_gap_error_m"] >= 0.0
                   and (r["v_ego_true_mps"] - r["control_lead_speed_mps"]) <= 0.0
                   and r["planner_accel_mps2"] < BRAKE_HOLD_THRESHOLD_MPS2]
      assert all(
        (not r["planner_lead_brake_release_debug"].get("active", False))
        or r["planner_lead_brake_release_debug"]["floor_mps2"] <= r["planner_accel_mps2"] + 1e-6
        for r in hold_rows
      ), f"{case}: release floor bound during a hold frame: mechanism changed, re-investigate"


def test_highway_dip_releases_brake_and_matches_lead_speed() -> None:
  metrics = _measure(_run("highway_dip"))
  physics = (
    f"highway dip (ego=lead={HW_V0_MPS} m/s, gap settled at target, lead dips 1.5 m/s and recovers):\n"
    f"  brake-hold after gap recovered & not closing: planner {metrics['brake_hold_planner_s']} s "
    f"(bound <= {MAX_BRAKE_HOLD_PLANNER_S}), realized {metrics['brake_hold_realized_s']} s "
    f"(bound <= {MAX_BRAKE_HOLD_REALIZED_S})\n"
    f"  max speed deficit vs lead after gap regained target: {metrics['reaccel_deficit_mps']} m/s "
    f"(bound <= {MAX_REACCEL_DEFICIT_MPS})\n"
    f"  rebound gap minimum vs target: {metrics['overshoot_gap_err_m']} m (bound >= {MIN_OVERSHOOT_GAP_ERR_M})\n"
    f"  min true gap: {metrics['min_true_gap_m']} m (floor {MIN_TRUE_GAP_FLOOR_M[HW_V0_MPS]})\n"
    f"  full: {_fmt(metrics)}"
  )
  assert metrics["recover_t_s"] is not None, physics
  assert metrics["min_true_gap_m"] >= MIN_TRUE_GAP_FLOOR_M[HW_V0_MPS], physics
  assert (metrics["brake_hold_planner_s"] <= MAX_BRAKE_HOLD_PLANNER_S
          and metrics["brake_hold_realized_s"] <= MAX_BRAKE_HOLD_REALIZED_S
          and metrics["reaccel_deficit_mps"] <= MAX_REACCEL_DEFICIT_MPS
          and metrics["overshoot_gap_err_m"] >= MIN_OVERSHOOT_GAP_ERR_M), physics


def test_city_ease_settles_without_ringing() -> None:
  metrics = _measure(_run("city_ease"))
  physics = (
    f"city ease (ego=lead={CITY_V0_MPS} m/s, lead eases 0.4 m/s^2 for 3.5 s and resumes):\n"
    f"  gap-error zero crossings with amplitude > {CYCLE_AMPLITUDE_FRAC:.0%} of target before settling: "
    f"{metrics['cycle_count']} (bound <= {MAX_CYCLE_COUNT})\n"
    f"  settle time (|gap error| < {SETTLE_BAND_FRAC:.0%} of target for {SETTLE_WINDOW_S} s): "
    f"{metrics['settle_s']} s (bound <= {MAX_SETTLE_S})\n"
    f"  min true gap: {metrics['min_true_gap_m']} m (floor {MIN_TRUE_GAP_FLOOR_M[CITY_V0_MPS]})\n"
    f"  full: {_fmt(metrics)}"
  )
  assert metrics["min_true_gap_m"] >= MIN_TRUE_GAP_FLOOR_M[CITY_V0_MPS], physics
  assert (metrics["cycle_count"] <= MAX_CYCLE_COUNT
          and metrics["settle_s"] is not None
          and metrics["settle_s"] <= MAX_SETTLE_S), physics
