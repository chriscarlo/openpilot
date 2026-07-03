"""Repro: CD5 (SEV-2) - cruise-reacquire jerk ramp composes unsafely with a
fresh-lead relatch and has NO lost-vs-departed memory.

Field evidence (road_forensics.json G3 handoff cluster, drive 00000200--8cbf2c9481
and 000001ff--59f2aa2d1d):

  200-9 tap2 (00000200--8cbf2c9481--9@33.850): a lead status drop entered the
  CruiseReacquireJerkRamp, which accelerated cruise to +0.64 m/s^2; then a
  flapping relatch of the SAME lead at dRel ~84 m / TTC > 100 s slammed aTarget
  to -2.08 m/s^2 - a 2.7 m/s^2 swing in 270 ms. The lead was never a threat
  (TTC > 30 s the whole time): the slam is the abrupt fresh-obstacle ObstacleCost
  landing on top of a ramped-up cruise accel with no blend.

  ff4 (000001ff--59f2aa2d1d--4@258.87): after a *prob-collapse* track break (the
  vision prob fell through radard's Schmitt band with the lead physically still
  there - NOT a genuine departure), the SAME reacquire ramp surged to +0.72 m/s^2
  within 1.8 s of a -0.97 m/s^2 brake, "with nothing ahead-status known." The
  ramp has no memory of WHY the lead exited: a phantom prob-collapse and a real
  departure escalate the jerk allowance identically.

Mechanism (selfdrive/controls/lib/longitudinal_planner.py
_apply_cruise_reacquire_jerk_limit, ~:578-635): on a lead0/lead1 -> cruise source
transition the ramp arms for CruiseReacquireJerkWindowS and lets the allowed
positive jerk escalate as `CruiseReacquirePosJerkLimit + CruiseReacquireJerkRamp *
elapsed_s`. Nothing bounds the FIRST braking frame after the lead relatches
(the clamp only clips the positive leg), and the escalation is blind to whether
the exit was a genuine departure or a recoverable prob-collapse.

Two parts, both tici-fidelity (device controller + real radard pipeline, noise
off to isolate the mechanism):

(a) Steady follow at 27 m/s; drop lead STATUS for the reacquire window (ramp
    climbs to ~+1.2 m/s^2); re-present the SAME lead mid-ramp at a non-threatening
    relatch (true TTC >> 30 s). STRICT-XFAIL: the fresh-obstacle relatch slams
    the planner past |1.0| m/s^2 (road -2.08) even though the lead is not a threat.

(b) Same steady follow, but the lead's PERCEPTION collapses via a scripted
    modelProb ramp 0.9 -> 0.10 through radard's Schmitt exit band, with dRel/vRel
    physically CONTINUOUS the whole time; held low past PhantomLeadHoldS so the
    source genuinely exits to cruise. STRICT-XFAIL: the reacquire jerk ceiling
    escalates above the CruiseReacquirePosJerkLimit floor within ~2 s of the
    collapse-exit exactly as it does after a genuine STATUS departure - proving
    the ramp has no lost-vs-departed memory (road ff4 +0.72 surge). A matched
    departure-exit twin shows the same escalation, and a ramp=0 rollback twin
    shows the escalation IS the ramp.

Safety invariant (stays GREEN, must never regress): an emergency relatch
(short TTC / fast close / FCW) is braked within one frame - the reacquire clamp
only limits the positive leg, so no rate-limit ever delays real braking.
"""
from __future__ import annotations

import functools

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

# ---- shared kinematics -------------------------------------------------------
EGO_V0_MPS = 27.0                # road ego ~27 m/s through the G3 handoff cluster
CRUISE_SPEED_MPS = 45.0          # well above ego: the ramp has headroom to climb
DURATION_S = 15.0

# Reacquire ramp floor / ramp defaults (common/params_keys.h, live-tunable):
#   CruiseReacquirePosJerkLimit = 0.08 m/s^3  (the "floor")
#   CruiseReacquireJerkWindowS  = 3.0 s
#   CruiseReacquireJerkRamp     = 0.8 m/s^3 per s
REACQUIRE_JERK_FLOOR_MPS3 = 0.08
REACQUIRE_JERK_FLOOR_EPS_MPS3 = 1e-3

# ==== PART (a): fresh-lead relatch slam ======================================
A_DROP_START_S = 4.0
A_DROP_END_S = 6.8               # 2.8 s status drop -> reacquire ramp climbs to ~+0.65
A_INITIAL_GAP_M = 45.0          # steady FOLLOW gap (lead0 is the MPC source pre-drop)
A_RELATCH_GAP_M = 42.0          # SAME lead re-presented at 42 m mid-ramp
A_RELATCH_VREL_MPS = -0.5       # road relatch closing ~-0.5 m/s
# True relatch geometry is non-threatening: TTC = 42 / 0.5 = 84 s (road ~100 s) >> 30 s.
A_RELATCH_TRUE_TTC_S = A_RELATCH_GAP_M / abs(A_RELATCH_VREL_MPS)
A_RELATCH_FLAP_S = 0.6          # brief Schmitt flap at relatch (road: flapping relatch)
# Spec (a) asserts |peak decel| <= 1.0; the planner peak is comfort-clamped at
# exactly -1.0 here, so the road-derived quantity we bound is the DISCONTINUITY
# itself: the one-frame aTarget swing at the relatch (road 200-9 tap2: 2.7 m/s^2
# swing in 270 ms). A comfortable non-threatening relatch should move aTarget by
# far less than this in a single 50 ms frame.
A_PEAK_DECEL_BOUND_MPS2 = 1.0   # planner peak is clamped at this; kept for context
A_RELATCH_SWING_BOUND_MPS2 = 0.5  # max acceptable one-frame aTarget swing (comfort)

# ==== PART (b): prob-collapse-exit vs departure-exit =========================
B_COLLAPSE_START_S = 4.0
B_COLLAPSE_END_S = 5.0          # prob ramps 0.9 -> 0.10 over 1 s (through Schmitt exit 0.25)
B_HOLD_LOW_UNTIL_S = 6.5        # hold prob low past PhantomLeadHoldS=0.80 -> genuine cruise exit
B_GAP_M = 45.0                 # lead PHYSICALLY present at ~45 m the whole time
B_LEAD_VREL_MPS = -0.3         # gentle continuous closing; kinematics never jump
B_PROB_HI = 0.9
B_PROB_LO = 0.10
# Lookback window after the exit over which the ramp must NOT escalate (spec ~2 s).
B_LOOKBACK_S = 2.0


# ---- part (a) scenario -------------------------------------------------------
def _build_steps_relatch(*, flap: bool = True) -> list[StepInput]:
  steps: list[StepInput] = []
  n = int(round(DURATION_S / DT_MDL))
  for i in range(n):
    t = i * DT_MDL
    if t < A_DROP_START_S:
      lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS, model_prob_target=0.98,
                           d_rel_override_m=A_INITIAL_GAP_M if i == 0 else None,
                           acquisition_reset=i == 0)
      note = "steady follow at 27 m/s"
    elif t < A_DROP_END_S:
      lead = LeadDirective(status=False)
      note = "lead status dropped -> reacquire ramp arms"
    else:
      tr = t - A_DROP_END_S
      # Brief Schmitt flap at the relatch instant (road: flapping relatch).
      prob = 0.98
      if flap and tr < A_RELATCH_FLAP_S:
        prob = 0.98 if int(tr / 0.1) % 2 == 0 else 0.10
      lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS + A_RELATCH_VREL_MPS,
                           model_prob_target=prob,
                           d_rel_override_m=A_RELATCH_GAP_M if abs(t - A_DROP_END_S) < 1e-6 else None,
                           acquisition_reset=abs(t - A_DROP_END_S) < 1e-6)
      note = "SAME lead relatched at 45 m / TTC 90 s mid-ramp"
    steps.append(StepInput(t_s=t, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead, note=note))
  return steps


@functools.lru_cache(maxsize=1)
def _run_relatch() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="reacquire_relatch_slam",
    steps=_build_steps_relatch(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


# ---- part (b) scenarios ------------------------------------------------------
def _collapse_prob(t: float) -> float:
  if t < B_COLLAPSE_START_S:
    return B_PROB_HI
  if t < B_COLLAPSE_END_S:
    frac = (t - B_COLLAPSE_START_S) / (B_COLLAPSE_END_S - B_COLLAPSE_START_S)
    return B_PROB_HI + (B_PROB_LO - B_PROB_HI) * frac
  if t < B_HOLD_LOW_UNTIL_S:
    return B_PROB_LO
  return B_PROB_HI


def _build_steps_collapse() -> list[StepInput]:
  """Prob-collapse exit: status stays True (plant keeps integrating true gap) so
  dRel/vRel are physically CONTINUOUS; only modelProb falls through the Schmitt
  band and is held low past PhantomLeadHoldS to force a genuine cruise exit."""
  steps: list[StepInput] = []
  n = int(round(DURATION_S / DT_MDL))
  for i in range(n):
    t = i * DT_MDL
    lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS + B_LEAD_VREL_MPS,
                         model_prob_target=_collapse_prob(t),
                         d_rel_override_m=B_GAP_M if i == 0 else None,
                         acquisition_reset=i == 0)
    steps.append(StepInput(t_s=t, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead,
                           note="prob collapse (continuous kinematics)"))
  return steps


def _build_steps_departure() -> list[StepInput]:
  """Genuine departure exit twin: identical timing but the lead LEAVES (status
  False) over the same interval, then the SAME lead returns."""
  steps: list[StepInput] = []
  n = int(round(DURATION_S / DT_MDL))
  for i in range(n):
    t = i * DT_MDL
    if t < B_COLLAPSE_START_S or t >= B_HOLD_LOW_UNTIL_S:
      lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS + B_LEAD_VREL_MPS,
                           model_prob_target=B_PROB_HI,
                           d_rel_override_m=B_GAP_M if i == 0 else None,
                           acquisition_reset=(i == 0) or abs(t - B_HOLD_LOW_UNTIL_S) < 1e-6)
    else:
      lead = LeadDirective(status=False)
    steps.append(StepInput(t_s=t, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead,
                           note="genuine status departure"))
  return steps


@functools.lru_cache(maxsize=1)
def _run_collapse() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="reacquire_prob_collapse_exit",
    steps=_build_steps_collapse(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off", seed=42, perception_filter="auto",
  )


@functools.lru_cache(maxsize=1)
def _run_departure() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="reacquire_departure_exit",
    steps=_build_steps_departure(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off", seed=42, perception_filter="auto",
  )


@functools.lru_cache(maxsize=1)
def _run_collapse_ramp_zero() -> SimulationResult:
  # Rollback sentinel for the reacquire ramp: CruiseReacquireJerkRamp=0 pins the
  # allowed jerk at CruiseReacquirePosJerkLimit for the whole window. This is
  # exactly the ceiling the CD5 fix must hold after a COLLAPSE-exit (selectively,
  # gated on exit cause). Proves the escalation is owned by the ramp knob.
  cfg = resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.CruiseReacquireJerkRamp": "0.0",
  })
  return run_harness(
    vehicle_config=cfg,
    scenario_name="reacquire_prob_collapse_exit_ramp0",
    steps=_build_steps_collapse(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off", seed=42, perception_filter="auto",
  )


# ---- measurement helpers -----------------------------------------------------
def _first_lead_to_cruise_exit_s(result: SimulationResult, after_s: float) -> float | None:
  prev = None
  for row in result.trace:
    src = row["planner_source"]
    if prev in ("lead0", "lead1") and src == "cruise" and row["t_s"] >= after_s:
      return float(row["t_s"])
    prev = src
  return None


def _ramp_peak_before(result: SimulationResult, before_s: float) -> float:
  peak = 0.0
  for row in result.trace:
    if row["t_s"] < before_s and row["planner_accel_mps2"] > peak:
      peak = row["planner_accel_mps2"]
  return peak


def _peak_decel_after(result: SimulationResult, after_s: float) -> tuple[float, float | None]:
  peak = 0.0
  when = None
  for row in result.trace:
    if row["t_s"] >= after_s and row["planner_accel_mps2"] < peak:
      peak = row["planner_accel_mps2"]
      when = float(row["t_s"])
  return peak, when


def _max_one_frame_swing_after(result: SimulationResult, after_s: float) -> tuple[float, float | None]:
  prev = None
  worst = 0.0
  when = None
  for row in result.trace:
    a = row["planner_accel_mps2"]
    if row["t_s"] >= after_s and prev is not None and abs(a - prev) > worst:
      worst = abs(a - prev)
      when = float(row["t_s"])
    prev = a
  return worst, when


def _max_allowed_jerk_in_window(result: SimulationResult, start_s: float, end_s: float) -> tuple[float, float | None]:
  worst = 0.0
  when = None
  for row in result.trace:
    d = row["planner_cruise_reacquire_debug"]
    if start_s <= row["t_s"] <= end_s and d.get("active") and d["allowed_jerk_mps3"] > worst:
      worst = d["allowed_jerk_mps3"]
      when = float(row["t_s"])
  return worst, when


# =============================================================================
# SCENARIO-VALIDITY GUARDS (must stay GREEN): if these go red the xfails below
# are vacuous and the oracle is no longer exercising CD5.
# =============================================================================
def test_relatch_scenario_wiring() -> None:
  result = _run_relatch()
  # Tici-fidelity loop resolved as intended.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"

  # The reacquire ramp actually arms on the status drop and climbs well above its
  # floor before the relatch (this is the ramped-up state the relatch lands on).
  ramp_peak = _ramp_peak_before(result, A_DROP_END_S)
  assert ramp_peak > 0.4, f"reacquire ramp did not climb (peak {ramp_peak:+.3f} m/s^2) - scenario no longer exercises CD5"
  ramp_active = [row for row in result.trace
                 if A_DROP_START_S <= row["t_s"] < A_DROP_END_S
                 and row["planner_cruise_reacquire_debug"].get("active")]
  assert ramp_active, "reacquire jerk clamp never armed during the status drop"

  # The relatch re-presents the SAME lead at the intended non-threatening gap
  # (true TTC 90 s): a genuine, corroborated, high-prob lead - not a threat.
  post = [row for row in result.trace if row["t_s"] >= A_DROP_END_S + 0.7]
  latched = [row for row in post if row["lead_one_published_d_rel_m"] is not None
             and row["lead_one_model_prob"] >= 0.9]
  assert latched, "lead never relatched after the drop"
  assert A_RELATCH_TRUE_TTC_S > 30.0, "part (a) requires relatch TTC > 30 s"


def test_collapse_and_departure_both_exit_to_cruise() -> None:
  collapse = _run_collapse()
  departure = _run_departure()
  for result in (collapse, departure):
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"

  # Both exit paths genuinely hand the source lead0 -> cruise (so the reacquire
  # ramp arms in BOTH) - that is what makes exit CAUSE the only difference.
  collapse_exit = _first_lead_to_cruise_exit_s(collapse, B_COLLAPSE_START_S)
  departure_exit = _first_lead_to_cruise_exit_s(departure, B_COLLAPSE_START_S)
  assert collapse_exit is not None, "prob-collapse never produced a cruise exit (phantom hold bridged it)"
  assert departure_exit is not None, "status departure never produced a cruise exit"

  # In the COLLAPSE run the underlying kinematics are physically CONTINUOUS: the
  # published dRel before and after the collapse is on the same ~45 m track with
  # no fabricated jump (the collapse is a perception event, not a real departure).
  pre = [row for row in collapse.trace if 3.0 <= row["t_s"] < B_COLLAPSE_START_S
         and row["lead_one_published_d_rel_m"] is not None]
  postc = [row for row in collapse.trace if B_HOLD_LOW_UNTIL_S + 0.3 <= row["t_s"] <= B_HOLD_LOW_UNTIL_S + 1.0
           and row["lead_one_published_d_rel_m"] is not None]
  assert pre and postc, "collapse run missing pre/post published-lead samples"
  # gap drifts only by the gentle true closing (~-0.3 m/s over the gap window), never a step
  assert abs(pre[-1]["lead_one_published_d_rel_m"] - postc[0]["lead_one_published_d_rel_m"]) < 6.0, (
    "collapse run published dRel jumped - kinematics were not continuous"
  )


# =============================================================================
# PART (a) STRICT-XFAIL: fresh-lead relatch slam on a non-threatening lead
# =============================================================================
def test_relatch_slam_bounded_on_nonthreatening_lead() -> None:
  result = _run_relatch()

  ramp_peak = _ramp_peak_before(result, A_DROP_END_S)
  peak_decel, peak_t = _peak_decel_after(result, A_DROP_END_S)
  swing, swing_t = _max_one_frame_swing_after(result, A_DROP_END_S)

  physics = (
    f"status drop [{A_DROP_START_S}, {A_DROP_END_S}) s -> reacquire ramp to {ramp_peak:+.3f} m/s^2; "
    f"SAME lead relatched at {A_RELATCH_GAP_M:.0f} m / vRel {A_RELATCH_VREL_MPS:+.1f} "
    f"(true TTC {A_RELATCH_TRUE_TTC_S:.0f} s >> 30 s, non-threatening):\n"
    f"  worst one-frame aTarget swing after relatch: {swing:.3f} m/s^2 @ t={swing_t}s "
    f"(bound {A_RELATCH_SWING_BOUND_MPS2}; road 200-9 tap2: 2.7 m/s^2 in 270 ms)\n"
    f"  peak planner decel after relatch: {peak_decel:+.3f} m/s^2 @ t={peak_t}s "
    f"(comfort-clamped at |{A_PEAK_DECEL_BOUND_MPS2:.1f}|; road aTarget slam -2.08)"
  )
  # A relatch onto a lead that is not a threat (TTC 84 s) must not jerk aTarget in
  # a single frame: the fresh obstacle must be blended in, not slammed on.
  assert swing <= A_RELATCH_SWING_BOUND_MPS2, physics


# =============================================================================
# PART (b) STRICT-XFAIL: no lost-vs-departed memory - a recoverable prob-collapse
# escalates the reacquire jerk exactly like a genuine departure.
# =============================================================================
def test_collapse_exit_holds_reacquire_jerk_floor() -> None:
  collapse = _run_collapse()
  departure = _run_departure()
  ramp_zero = _run_collapse_ramp_zero()

  collapse_exit = _first_lead_to_cruise_exit_s(collapse, B_COLLAPSE_START_S)
  departure_exit = _first_lead_to_cruise_exit_s(departure, B_COLLAPSE_START_S)
  assert collapse_exit is not None and departure_exit is not None

  c_jerk, c_when = _max_allowed_jerk_in_window(collapse, collapse_exit, collapse_exit + B_LOOKBACK_S)
  d_jerk, d_when = _max_allowed_jerk_in_window(departure, departure_exit, departure_exit + B_LOOKBACK_S)
  z_jerk, _ = _max_allowed_jerk_in_window(ramp_zero,
                                          _first_lead_to_cruise_exit_s(ramp_zero, B_COLLAPSE_START_S) or collapse_exit,
                                          (_first_lead_to_cruise_exit_s(ramp_zero, B_COLLAPSE_START_S) or collapse_exit) + B_LOOKBACK_S)

  physics = (
    f"prob-collapse exit vs genuine departure exit (both hand lead0 -> cruise), "
    f"CruiseReacquirePosJerkLimit floor {REACQUIRE_JERK_FLOOR_MPS3:.2f} m/s^3:\n"
    f"  COLLAPSE-exit @ t={collapse_exit}s: reacquire allowed_jerk escalated to "
    f"{c_jerk:.3f} m/s^3 within {B_LOOKBACK_S} s (peak @ t={c_when}s) "
    f"- should hold at the floor (road ff4 +0.72 surge)\n"
    f"  DEPARTURE-exit @ t={departure_exit}s: allowed_jerk escalated to {d_jerk:.3f} m/s^3 "
    f"(peak @ t={d_when}s) - escalation is legitimate here\n"
    f"  ramp=0 rollback sentinel on the collapse run: allowed_jerk stayed at "
    f"{z_jerk:.3f} m/s^3 (proves the escalation IS the ramp; the fix pins this "
    f"ceiling after a collapse-exit)"
  )
  # After a recoverable prob-COLLAPSE exit the reacquire jerk must NOT escalate
  # above its floor for the lookback window (the ramp should stay lost-aware).
  assert c_jerk <= REACQUIRE_JERK_FLOOR_MPS3 + REACQUIRE_JERK_FLOOR_EPS_MPS3, physics


# =============================================================================
# SAFETY INVARIANT (stays GREEN, must never regress): the reacquire rate-limit
# only clips the POSITIVE leg, so an emergency relatch is braked within one
# frame - no rate-limit ever delays genuine braking.
# =============================================================================
EMERG_DROP_START_S = 4.0
EMERG_DROP_END_S = 6.8
EMERG_RELATCH_GAP_M = 18.0
EMERG_RELATCH_VREL_MPS = -8.0    # TTC 2.25 s, fast close + FCW
EMERG_BRAKE_THRESHOLD_MPS2 = -2.0
EMERG_MAX_ONSET_DELAY_S = 0.2    # must brake within ~one planner frame of relatch


def _build_steps_emergency() -> list[StepInput]:
  steps: list[StepInput] = []
  n = int(round(12.0 / DT_MDL))
  for i in range(n):
    t = i * DT_MDL
    if t < EMERG_DROP_START_S:
      lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS, model_prob_target=0.98,
                           d_rel_override_m=A_INITIAL_GAP_M if i == 0 else None,
                           acquisition_reset=i == 0)
    elif t < EMERG_DROP_END_S:
      lead = LeadDirective(status=False)
    else:
      lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS + EMERG_RELATCH_VREL_MPS,
                           model_prob_target=0.98, fcw=True,
                           d_rel_override_m=EMERG_RELATCH_GAP_M if abs(t - EMERG_DROP_END_S) < 1e-6 else None,
                           acquisition_reset=abs(t - EMERG_DROP_END_S) < 1e-6)
    steps.append(StepInput(t_s=t, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead))
  return steps


def test_emergency_relatch_not_rate_limited() -> None:
  """SAFETY: a mid-ramp emergency relatch (18 m, vRel -8, TTC 2.25 s, FCW) must
  be braked within one frame - the reacquire clamp must never delay real braking.
  This is the invariant the eventual CD5 rate-limit fix must not break."""
  result = run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="reacquire_emergency_relatch",
    steps=_build_steps_emergency(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off", seed=42, perception_filter="auto",
  )

  onset_delay = None
  peak_decel = 0.0
  for row in result.trace:
    if row["t_s"] >= EMERG_DROP_END_S:
      a = row["planner_accel_mps2"]
      if onset_delay is None and a <= EMERG_BRAKE_THRESHOLD_MPS2:
        onset_delay = float(row["t_s"]) - EMERG_DROP_END_S
      if a < peak_decel:
        peak_decel = a

  physics = (
    f"emergency relatch at {EMERG_RELATCH_GAP_M:.0f} m / vRel {EMERG_RELATCH_VREL_MPS:+.1f} "
    f"(TTC {EMERG_RELATCH_GAP_M / abs(EMERG_RELATCH_VREL_MPS):.2f} s, FCW) mid reacquire ramp:\n"
    f"  brake (<= {EMERG_BRAKE_THRESHOLD_MPS2} m/s^2) onset after relatch: {onset_delay} s "
    f"(bound {EMERG_MAX_ONSET_DELAY_S} s); peak planner decel {peak_decel:.2f} m/s^2"
  )
  assert onset_delay is not None and onset_delay <= EMERG_MAX_ONSET_DELAY_S, physics
  assert peak_decel <= -3.0, physics


# =============================================================================
# SAFETY INVARIANT (stays GREEN): a MODERATE cut-in toward a decelerating lead
# at long range must NOT be blend-delayed. The relatch blend's urgency-bypass
# predicate is thin here - closing 2.0 < urgent-closing, TTC 15 s > urgent-TTC,
# no FCW - so the aLeadK <= CruiseRelatchUrgentLeadDecelMps2 bypass (and the
# cut-in track-identity guard) must let anticipatory braking pass within one
# frame of the pre-fix (blend-off) path. This covers the CD2/lead_decel gap the
# extreme-FCW emergency test does not exercise.
# =============================================================================
MOD_CUTIN_APPEAR_S = 5.0
MOD_CUTIN_GAP_M = 30.0
MOD_CUTIN_VREL_MPS = -2.0        # closing 2.0 < urgent-closing 2.5
MOD_CUTIN_ALEADK_MPS2 = -1.0     # lead already braking; TTC 15 s > urgent-TTC 4 s
MOD_CUTIN_TRUE_TTC_S = MOD_CUTIN_GAP_M / abs(MOD_CUTIN_VREL_MPS)
MOD_ONSET_TOLERANCE_S = 0.10     # blend must not delay onset beyond ~one frame


def _build_steps_moderate_cutin() -> list[StepInput]:
  steps: list[StepInput] = []
  n = int(round(12.0 / DT_MDL))
  for i in range(n):
    t = i * DT_MDL
    if t < MOD_CUTIN_APPEAR_S:
      # No lead: plain cruise, so the appearance is a genuine cruise->lead
      # transition (relatch-arming edge) onto a NEW track.
      lead = LeadDirective(status=False)
    else:
      lead = LeadDirective(status=True, v_lead_mps=EGO_V0_MPS + MOD_CUTIN_VREL_MPS,
                           model_prob_target=0.98,
                           a_lead_k_mps2=MOD_CUTIN_ALEADK_MPS2,
                           d_rel_override_m=MOD_CUTIN_GAP_M if abs(t - MOD_CUTIN_APPEAR_S) < 1e-6 else None,
                           acquisition_reset=abs(t - MOD_CUTIN_APPEAR_S) < 1e-6)
    steps.append(StepInput(t_s=t, cruise_speed_mps=CRUISE_SPEED_MPS, lead_one=lead))
  return steps


def _moderate_cutin_brake_onset(*, blend_s: str) -> tuple[float | None, float]:
  cfg = resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.CruiseRelatchBlendS": blend_s,
  })
  result = run_harness(
    vehicle_config=cfg,
    scenario_name=f"reacquire_moderate_cutin_blend_{blend_s}",
    steps=_build_steps_moderate_cutin(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off", seed=42, perception_filter="auto",
  )
  onset = None
  peak = 0.0
  for row in result.trace:
    if row["t_s"] >= MOD_CUTIN_APPEAR_S:
      a = row["planner_accel_mps2"]
      if onset is None and a <= -0.5:
        onset = float(row["t_s"]) - MOD_CUTIN_APPEAR_S
      if a < peak:
        peak = a
  return onset, peak


def test_moderate_cutin_decel_lead_not_blend_delayed() -> None:
  """SAFETY: a moderate cut-in (30 m, closing 2.0 < urgent, TTC 15 s > urgent,
  no FCW) toward an already-decelerating lead (aLeadK -1.0) must brake within one
  frame of the blend-OFF path - the aLeadK lead-decel bypass (and cut-in guard)
  keep the relatch blend from clipping anticipatory braking."""
  on_onset, on_peak = _moderate_cutin_brake_onset(blend_s="1.5")   # blend active (default)
  off_onset, off_peak = _moderate_cutin_brake_onset(blend_s="0.0")  # rollback sentinel

  physics = (
    f"moderate cut-in at {MOD_CUTIN_GAP_M:.0f} m / vRel {MOD_CUTIN_VREL_MPS:+.1f} "
    f"(closing {abs(MOD_CUTIN_VREL_MPS):.1f} < urgent 2.5, TTC {MOD_CUTIN_TRUE_TTC_S:.0f} s > urgent 4, "
    f"no FCW), lead aLeadK {MOD_CUTIN_ALEADK_MPS2:+.1f} (already braking):\n"
    f"  brake (<= -0.5) onset: blend-ON {on_onset} s vs blend-OFF {off_onset} s "
    f"(tolerance {MOD_ONSET_TOLERANCE_S} s)\n"
    f"  peak decel: blend-ON {on_peak:.3f} vs blend-OFF {off_peak:.3f} m/s^2"
  )
  assert on_onset is not None and off_onset is not None, physics
  # Blend must not delay braking onset beyond ~one frame vs the pre-fix path.
  assert on_onset <= off_onset + MOD_ONSET_TOLERANCE_S, physics
  # And the anticipatory brake magnitude must not be materially clipped by the
  # blend (the aLeadK bypass removes the large-TTC decel cap for a decel lead).
  assert on_peak <= off_peak + 0.15, physics
