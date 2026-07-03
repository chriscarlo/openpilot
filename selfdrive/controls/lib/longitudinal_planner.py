#!/usr/bin/env python3
import math
from collections import deque
import numpy as np
from openpilot.common.params import Params
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_BRAKE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_headway_follow_distance
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_low_speed_launch_follow_max_accel
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.planner_lag_debug import (
  SPAN_MPC_UPDATE,
  SPAN_PLANNER_UPDATE_TOTAL,
  end_span,
  record_fields,
  start_span,
)
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

LON_MPC_STEP = 0.2  # first step is 0.2s
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
LEAD_LAUNCH_RELEASE_HOLD_S = 0.10
LEAD_LAUNCH_RELEASE_MIN_DREL_M = 5.0

# Follow-regime gap reclaim (F2 of the limit-cycle fix): the raise above the
# coast bias only phases in once the lead is genuinely pulling away, so a
# settled equal-speed follow keeps today's coast floor bit-identically.
GAP_RECLAIM_FOLLOW_PULLAWAY_BP = [0.35, 1.25]
GAP_RECLAIM_FOLLOW_PULLAWAY_V = [0.0, 1.0]
GAP_RECLAIM_FOLLOW_PROJECT_HORIZON_S = 1.2
GAP_RECLAIM_FOLLOW_MIN_GAP_DIV_M = 0.5

# Lookup table for turns
# Allow higher total accel (lateral+longitudinal) at low speeds and taper with speed
# Shape: 0 m/s -> 4.0 m/s^2, 20 m/s -> 2.0 m/s^2, 40 m/s+ -> 1.2 m/s^2
_A_TOTAL_MAX_V = [4.0, 2.0, 1.2]
_A_TOTAL_MAX_BP = [0., 20., 40.]


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def should_release_stop_for_lead_launch(CP, *, standstill: bool, v_ego: float,
                                        a_target: float, lead_source: str,
                                        control_leads) -> bool:
  if (not bool(standstill) or
      float(a_target) <= 0.0 or
      lead_source not in ("lead0", "lead1")):
    return False

  lead_idx = 0 if lead_source == "lead0" else 1
  if lead_idx >= len(control_leads):
    return False

  lead = control_leads[lead_idx]
  if lead is None or not bool(getattr(lead, "status", False)):
    return False

  lead_vrel = float(getattr(lead, "vRel", 0.0) or 0.0)
  lead_v = float(getattr(lead, "vLead", v_ego) or v_ego)
  lead_drel = float(getattr(lead, "dRel", 0.0) or 0.0)
  if lead_drel < LEAD_LAUNCH_RELEASE_MIN_DREL_M:
    return False

  lead_pullaway_speed = max(0.0, lead_vrel, lead_v - float(v_ego))
  return bool(lead_pullaway_speed > max(float(getattr(CP, "vEgoStarting", 0.0)), 0.1))


def get_lead_brake_release_accel_floor(mpc, *, v_ego: float, lead_source: str, control_leads):
  debug = {
    "active": False,
    "reason": "inactive",
    "floor_mps2": None,
    "gap_error_m": None,
    "vrel_credit_m": None,
    "pullaway_mps": None,
    "time_to_target_s": None,
    "danger_surplus_m": None,
    "brake_authority_decel_mps2": None,
    "brake_authority_gap_m": None,
    "brake_authority_surplus_m": None,
  }
  vibe_controller = getattr(mpc, "vibe_controller", None)
  if vibe_controller is None or not bool(vibe_controller.is_follow_enabled()):
    debug["reason"] = "vibe_follow_disabled"
    return None, debug
  tuning = getattr(mpc, "_live_tune_cfg", None)
  tuning = tuning if tuning is not None else LeadResponseTuningConfig.defaults()

  if lead_source not in ("lead0", "lead1") or float(v_ego) < tuning.lead_brake_release_min_speed_mps:
    return None, debug

  lead_idx = 0 if lead_source == "lead0" else 1
  if lead_idx >= len(control_leads):
    debug["reason"] = "missing_lead"
    return None, debug

  lead = control_leads[lead_idx]
  if lead is None or not bool(getattr(lead, "status", False)):
    debug["reason"] = "invalid_lead"
    return None, debug

  t_follow = float(getattr(mpc, "current_t_follow", 0.0) or 0.0)
  if t_follow <= 0.0:
    debug["reason"] = "missing_t_follow"
    return None, debug

  lead_v = max(0.0, float(getattr(lead, "vLead", v_ego) or v_ego))
  lead_drel = max(0.0, float(getattr(lead, "dRel", 0.0) or 0.0))
  lead_accel = float(getattr(lead, "aLeadK", 0.0) or 0.0)
  lead_vrel = float(getattr(lead, "vRel", lead_v - float(v_ego)) or 0.0)
  pullaway_speed = max(lead_vrel, lead_v - float(v_ego))
  closing_speed = max(0.0, -lead_vrel, float(v_ego) - lead_v)

  # vRel-aware gap error (variant A of the limit-cycle brake-hold fix): measure
  # recovery against the same target the MPC itself uses —
  # desired_follow_distance(v_ego, v_lead, t_follow) — instead of the bare
  # equal-speed headway gap. The stopped-equivalence difference
  # (v_lead^2 - v_ego^2) / (2 * COMFORT_BRAKE) is granted only as CREDIT
  # (lead genuinely faster -> the headway target the MPC has already recovered
  # past is closer than the equal-speed one), never as extra pessimism, and:
  #   - the lead-speed basis is the more pessimistic of vLead and
  #     v_ego + vRel, so a closing state (either signal) gets zero credit and
  #     keeps today's headway-based (more-braking) eligibility bit-identically;
  #   - the credit is capped at lead_brake_release_vrel_credit_cap_m so a
  #     misassociated much-faster lead cannot buy a coast floor at any
  #     distance. Cap = 0 is the rollback sentinel (exact legacy gap error).
  vrel_credit = 0.0
  credit_cap_m = float(tuning.lead_brake_release_vrel_credit_cap_m)
  if credit_cap_m > 0.0:
    lead_v_pessimistic = max(0.0, min(lead_v, float(v_ego) + lead_vrel))
    # Recovery projection: the tracker's aLeadK leads its vRel by several
    # hundred ms, so a lead that has finished a transient slowdown reads
    # "accelerating hard" well before its published speed crosses back over
    # ego's. Project the pessimistic lead-speed basis forward by the POSITIVE
    # part of aLeadK only — a decelerating or steady lead gets nothing, and
    # the total credit stays under the same cap — so the release floor swings
    # up within the actuation lag of true recovery instead of a half second
    # late. Rollback sentinel: LeadBrakeReleaseRecoveryProjS = 0.
    lead_v_pessimistic += max(0.0, lead_accel) * float(tuning.lead_brake_release_recovery_proj_s)
    vrel_credit = min(
      credit_cap_m,
      max(0.0, (lead_v_pessimistic ** 2 - float(v_ego) ** 2) / (2.0 * COMFORT_BRAKE)),
    )
  gap_error = lead_drel + vrel_credit - get_headway_follow_distance(float(v_ego), t_follow)
  debug["gap_error_m"] = float(gap_error)
  debug["vrel_credit_m"] = float(vrel_credit)
  debug["pullaway_mps"] = float(pullaway_speed)
  debug["closing_mps"] = float(closing_speed)
  debug["lead_accel_mps2"] = float(lead_accel)

  params = getattr(mpc, "params", None)
  if params is None:
    debug["reason"] = "missing_mpc_geometry"
    return None, debug

  brake_decel = max(1e-3, abs(min(0.0, float(params[0, 0]) if float(params[0, 0]) < 0.0 else ACCEL_MIN)))
  relative_stop_extra_m = (closing_speed ** 2) / (2.0 * brake_decel)
  brake_authority_gap = get_headway_follow_distance(float(v_ego), t_follow) + relative_stop_extra_m
  # Eligibility uses the same vRel-aware recovery measure as gap_error: credit
  # is zero in every closing state, so no closing state gets looser eligibility
  # than the legacy headway-based gate.
  brake_authority_surplus = lead_drel + vrel_credit - brake_authority_gap
  debug["brake_authority_decel_mps2"] = float(brake_decel)
  debug["brake_authority_gap_m"] = float(brake_authority_gap)
  debug["brake_authority_surplus_m"] = float(brake_authority_surplus)
  debug["danger_surplus_m"] = float(brake_authority_surplus)
  if brake_authority_surplus < -tuning.lead_brake_release_brake_deficit_margin_m:
    debug["reason"] = "brake_authority_deficit"
    return None, debug

  if lead_accel < tuning.lead_brake_release_lead_decel_min_mps2:
    debug["reason"] = "lead_decelerating"
    return None, debug

  # CD1 fix — lead-decel-aware brake-release floor (road 200-15-17 TAP 1). The
  # closing / near-target branches below sized the floor from the INSTANTANEOUS
  # closing speed only, treating the lead as holding its current speed. A lead
  # braking to a stop at a decel shallow enough to stay inside the -0.75
  # lead_brake_release_lead_decel_min veto (road aLeadK -0.42..-0.63) keeps
  # closing faster every tick, so the true required ego decel is the legacy
  # relative-closure term PLUS the lead's own deceleration (relative-frame
  # kinematics: for the relative velocity to null out inside the gap surplus the
  # ego decel must overcome a_lead as well). Adding that term deepens the floor
  # exactly when the lead is decelerating so the floor stops clipping the MPC's
  # correct ramping brake; a steady or accelerating lead contributes zero, so
  # every non-decelerating state keeps today's floor bit-identically.
  # lead_decel_extra is the magnitude (>= 0) the lead's braking adds to the
  # required ego decel, scaled by the live-tunable projection gain
  # (lead_brake_release_lead_decel_project_gain; 0 = exact legacy rollback).
  lead_decel_extra = (max(0.0, -lead_accel) *
                      float(tuning.lead_brake_release_lead_decel_project_gain))
  debug["lead_decel_extra_mps2"] = float(lead_decel_extra)
  near_target_branch = False
  if gap_error >= 0.0:
    if closing_speed > 0.0:
      decel_needed = (closing_speed ** 2) / (2.0 * max(gap_error, 0.5)) + lead_decel_extra
      release_floor = -min(decel_needed, brake_decel)
    else:
      release_floor = tuning.lead_brake_release_coast_bias_mps2
      # Follow-regime gap reclaim (F2): once the vRel-aware target is
      # recovered and the lead is pulling away, raise the floor toward
      # gap_reclaim_follow_max_accel so ego matches lead speed BEFORE the
      # equal-speed gap is regained, instead of crawling up at the MPC's
      # unwind jerk and then having to accelerate harder and overshoot.
      # The raise is tapered by the kinematic overshoot bound: project the
      # closing speed that the CURRENT commanded accel produces over a short
      # horizon; the decel that projected closure would need to shed over the
      # remaining (vRel-aware) gap surplus scales the extra authority away.
      # Rollback sentinel: GapReclaimFollowMaxAccel <= the coast bias
      # disables the raise entirely (exact pre-fix coast floor);
      # GapReclaimTaperGain = 0 removes only the taper (naive raise).
      follow_cap = float(tuning.gap_reclaim_follow_max_accel)
      if follow_cap > release_floor:
        pullaway_ramp = float(np.interp(pullaway_speed, GAP_RECLAIM_FOLLOW_PULLAWAY_BP, GAP_RECLAIM_FOLLOW_PULLAWAY_V))
        if pullaway_ramp > 0.0:
          horizon_s = GAP_RECLAIM_FOLLOW_PROJECT_HORIZON_S
          ego_accel = float(getattr(mpc, "x0", (0.0, 0.0, 0.0))[2])
          projected_closing = max(
            0.0,
            (float(v_ego) + ego_accel * horizon_s) - (lead_v + lead_accel * horizon_s),
          )
          overshoot_decel_need = (projected_closing ** 2) / (2.0 * max(gap_error, GAP_RECLAIM_FOLLOW_MIN_GAP_DIV_M))
          overshoot_taper = float(np.clip(1.0 - float(tuning.gap_reclaim_taper_gain) * overshoot_decel_need, 0.0, 1.0))
          release_floor += (follow_cap - release_floor) * pullaway_ramp * overshoot_taper
          debug["follow_reclaim_taper"] = float(overshoot_taper)
          debug["follow_reclaim_pullaway_ramp"] = float(pullaway_ramp)
    time_to_target_s = 0.0
  elif (gap_error >= -tuning.lead_brake_release_near_target_margin_m and
        closing_speed <= tuning.lead_brake_release_near_target_max_closing_mps):
    # Near-target hold at the shallow near-target floor, but lower it by the
    # lead's own deceleration so a lead braking hard inside the near-target band
    # cannot be pinned above the decel its own braking demands. Steady lead ->
    # lead_decel_extra = 0 -> exact legacy near-target floor.
    release_floor = min(tuning.lead_brake_release_near_target_floor_mps2,
                        tuning.lead_brake_release_near_target_floor_mps2 - lead_decel_extra)
    release_floor = max(release_floor, -brake_decel)
    near_target_branch = True
    time_to_target_s = 0.0
  else:
    if pullaway_speed <= tuning.lead_brake_release_min_pullaway_mps:
      debug["reason"] = "not_opening"
      return None, debug
    time_to_target_s = -gap_error / max(pullaway_speed, 1e-3)
    if time_to_target_s > tuning.lead_brake_release_lookahead_s:
      debug["time_to_target_s"] = float(time_to_target_s)
      debug["reason"] = "target_too_far"
      return None, debug
    progress = 1.0 - float(np.clip(time_to_target_s / tuning.lead_brake_release_lookahead_s, 0.0, 1.0))
    release_floor = float(np.interp(
      progress,
      [0.0, 1.0],
      [tuning.lead_brake_release_approach_floor_mps2, tuning.lead_brake_release_coast_bias_mps2],
    ))

  debug["active"] = True
  if gap_error >= 0.0:
    debug["reason"] = "closing_to_target" if closing_speed > 0.0 else "gap_recovered"
  elif near_target_branch:
    debug["reason"] = "near_target"
  else:
    debug["reason"] = "projected_recovery"
  debug["floor_mps2"] = float(release_floor)
  debug["time_to_target_s"] = float(time_to_target_s)
  return float(release_floor), debug


def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt, CP=CP)
    # TODO remove mpc modes when TR released
    self.mpc.mode = 'acc'
    LongitudinalPlannerSP.__init__(self, self.CP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False
    self._lead_launch_release_counter = 0
    self._prev_mpc_source: str = ""
    self._cruise_pos_jerk_frames_left: int = 0
    self._cruise_pos_jerk_prev_a: float = 0.0
    self._flutter_prev_source: str = ""
    self._source_transition_frames: deque = deque()
    self._flutter_clamp_prev_a: float = 0.0
    self._flutter_mode_active: bool = False
    self.lead_brake_release_accel_floor = 0.0
    self.lead_brake_release_debug = {"active": False, "reason": "init"}
    # Observability for the cruise-reacquire jerk ramp (read-only; does not
    # affect behavior). Lets the longitudinal harness prove whether the ramp is
    # escalating above its jerk floor and how far into the window it is.
    self.cruise_reacquire_debug: dict = {
      "active": False,
      "frames_left": 0,
      "elapsed_s": 0.0,
      "allowed_jerk_mps3": 0.0,
      "jerk_floor_mps3": 0.0,
      "slew_ceiling_mps2": 0.0,
      "clipped": False,
    }

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0

    self.params = Params()
    self.param_read_counter = 0

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    total_span = start_span(SPAN_PLANNER_UPDATE_TOTAL)
    self.mode = 'blended' if sm['selfdriveState'].experimentalMode else 'acc'
    if not self.mlsim:
      self.mpc.mode = self.mode
    LongitudinalPlannerSP.update(self, sm)
    if dec_mpc_mode := self.get_mpc_mode():
      self.mode = dec_mpc_mode
      if not self.mlsim:
        self.mpc.mode = dec_mpc_mode

    self.handle_mode_transition(self.mode)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if self.mode == 'acc':
      if self.vibe_controller.is_accel_enabled():
        # Only get max acceleration from vibe controller, use default ACCEL_MIN for minimum
        accel_limits = self.vibe_controller.get_accel_limits(v_ego)
        if accel_limits is not None:
          max_accel = accel_limits[1]
          accel_clip = [ACCEL_MIN, max_accel]
        else:
          # Fallback to stock if vibe controller returns None
          accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
        # VTSC: Disabled turn acceleration limiting - conflicts with Vision Turn Speed Controller
        # steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
        # accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)
      else:
        accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
        # VTSC: Disabled turn acceleration limiting - conflicts with Vision Turn Speed Controller
        # steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
        # accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)
    else:
      # For mode != 'acc' ('blended')
      if self.vibe_controller.is_accel_enabled():
        accel_limits = self.vibe_controller.get_accel_limits(v_ego)
        if accel_limits is not None:
          max_accel = accel_limits[1]
          accel_clip = [ACCEL_MIN, max_accel]
        else:
          accel_clip = [ACCEL_MIN, ACCEL_MAX]
      else:
        accel_clip = [ACCEL_MIN, ACCEL_MAX]

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])
      self._lead_launch_release_counter = 0

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    x, v, a, j, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Get new v_cruise from Speed Limit Control
    self._planner_output_accel_limits = (float(accel_clip[0]), float(accel_clip[1]))
    v_cruise = LongitudinalPlannerSP.update_v_cruise(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    mpc_update_span = start_span(SPAN_MPC_UPDATE)
    try:
      self.mpc.update(sm['radarState'], v_cruise, x, v, a, j, personality=sm['selfdriveState'].personality)
    finally:
      end_span(mpc_update_span)
    record_fields(
      mpc_solve_time_ms=float(getattr(self.mpc, 'solve_time', 0.0) or 0.0) * 1000.0,
      planner_mode=str(self.mode),
    )

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                      action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    lead_launch_release_ready = output_should_stop_mpc and should_release_stop_for_lead_launch(
      self.CP,
      standstill=sm['carState'].standstill,
      v_ego=v_ego,
      a_target=output_a_target_mpc,
      lead_source=str(getattr(self.mpc, "source", "")),
      control_leads=getattr(self.mpc, "control_leads", ()),
    )
    if lead_launch_release_ready:
      self._lead_launch_release_counter += 1
      release_hold_frames = max(1, int(np.ceil(LEAD_LAUNCH_RELEASE_HOLD_S / max(self.dt, 1e-3))))
      if self._lead_launch_release_counter >= release_hold_frames:
        output_should_stop_mpc = False
    else:
      self._lead_launch_release_counter = 0

    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    if self.mode == 'acc' or not self.mlsim:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc
    else:
      output_a_target = self.blend_accel_transition(output_a_target_mpc, output_a_target_e2e, v_ego)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc

    if self.object_hazard.is_active and self.object_hazard.stop_required:
      self.output_should_stop = True

    gap_reclaim_floor = float(getattr(self.mpc, 'gap_reclaim_accel_floor', 0.0) or 0.0)
    lead_keepup_floor = float(getattr(self.mpc, 'lead_keepup_accel_floor', 0.0) or 0.0)
    if (lead_keepup_floor > 0.0 and
        not self.output_should_stop and
        output_a_target >= -0.12):
      output_a_target = max(output_a_target, lead_keepup_floor)
    if (gap_reclaim_floor > 0.0 and
        not bool(getattr(self.mpc, 'use_upstream_gap_reclaim', False)) and
        not self.output_should_stop and
        output_a_target >= -0.05):
      output_a_target = max(output_a_target, gap_reclaim_floor)

    cutin_settle_floor = float(getattr(self.mpc, 'cutin_settle_accel_floor', 0.0) or 0.0)
    if (bool(getattr(self.mpc, 'cutin_settle_active', False)) and
        not self.output_should_stop):
      output_a_target = max(output_a_target, cutin_settle_floor)

    lead_slowdown_ceiling = getattr(self.mpc, 'lead_slowdown_accel_ceiling', None)
    if lead_slowdown_ceiling is not None:
      output_a_target = min(output_a_target, float(lead_slowdown_ceiling))

    lead_source = str(getattr(self.mpc, "source", ""))
    control_leads = getattr(self.mpc, "control_leads", ())
    lead_brake_release_floor, lead_brake_release_debug = get_lead_brake_release_accel_floor(
      self.mpc,
      v_ego=v_ego,
      lead_source=lead_source,
      control_leads=control_leads,
    )
    self.lead_brake_release_accel_floor = float(lead_brake_release_floor or 0.0)
    self.lead_brake_release_debug = lead_brake_release_debug
    if lead_brake_release_floor is not None and not self.output_should_stop:
      output_a_target = max(output_a_target, float(lead_brake_release_floor))
      # The M1 kinematic slowdown ceiling must win over the release floor while
      # the lead is a corroborated threat: in the overlap state (lead
      # decelerating between the slowdown onset threshold and the release
      # path's -0.75 m/s^2 gate, or any closing state) the kinematic bound may
      # demand more braking than the floor would allow. Gate the re-clamp on
      # the threat signals themselves so the ceiling's slew-limited RELEASE
      # tail (a comfort mechanism, not a threat bound) cannot re-brake a
      # recovered, opening gap below the bound release floor.
      lead_is_threatening = (
        float(lead_brake_release_debug.get("lead_accel_mps2") or 0.0) < 0.0 or
        float(lead_brake_release_debug.get("closing_mps") or 0.0) > 0.0)
      if lead_slowdown_ceiling is not None and lead_is_threatening:
        output_a_target = min(output_a_target, float(lead_slowdown_ceiling))

    cruise_owned_accel_cap = getattr(self.mpc, "cruise_owned_accel_cap", None)
    if lead_source == "cruise" and cruise_owned_accel_cap is not None:
      output_a_target = min(output_a_target, float(cruise_owned_accel_cap))

    if lead_source in ("lead0", "lead1"):
      lead_idx = 0 if lead_source == "lead0" else 1
      if lead_idx < len(control_leads):
        launch_follow_max_accel = get_low_speed_launch_follow_max_accel(
          v_ego,
          control_leads[lead_idx],
          getattr(self.mpc, "current_t_follow", 0.0),
          float(accel_clip[1]),
        )
        accel_clip[1] = max(float(accel_clip[1]), float(launch_follow_max_accel))

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self._planner_output_accel_limits = (float(accel_clip[0]), float(accel_clip[1]))
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

    model_accel_for_flutter = float(sm['modelV2'].action.desiredAcceleration) if hasattr(sm['modelV2'], 'action') else 0.0
    self._apply_cruise_reacquire_jerk_limit(lead_source)
    self._apply_flutter_mode_clamp(lead_source, model_accel_for_flutter)

    end_span(total_span)

  def _apply_cruise_reacquire_jerk_limit(self, lead_source: str) -> None:
    # When the MPC's selected source transitions from lead-follow to cruise, the
    # planner's output accel can jump sharply as it seeks the set speed. Clamp the
    # upward slew for a short window so the transition feels less abrupt. Braking
    # and lead-follow acceleration are unaffected. Window ends early if output
    # reaches the cruise accel cap, so steady-state cruise is never constrained.
    tune_cfg = getattr(self.mpc, "_live_tune_cfg", None)
    jerk_limit = float(getattr(tune_cfg, "cruise_reacquire_pos_jerk_limit", 0.0) or 0.0)
    window_s = float(getattr(tune_cfg, "cruise_reacquire_jerk_window_s", 0.0) or 0.0)
    jerk_ramp = float(getattr(tune_cfg, "cruise_reacquire_jerk_ramp_mps3_per_s", 0.0) or 0.0)

    source_is_lead = lead_source in ("lead0", "lead1")
    dt = float(max(self.dt, 1e-3))

    if (self._prev_mpc_source in ("lead0", "lead1") and not source_is_lead and
        jerk_limit > 0.0 and window_s > 0.0):
      self._cruise_pos_jerk_frames_left = int(math.ceil(window_s / dt))
      # _cruise_pos_jerk_prev_a carries the prior frame's clipped output_a_target,
      # so we intentionally do not overwrite it here — it is our slew anchor.

    if source_is_lead:
      self._cruise_pos_jerk_frames_left = 0

    if self._cruise_pos_jerk_frames_left > 0 and jerk_limit > 0.0:
      # Allowed jerk escalates the longer the handoff persists: the first frames
      # stay as soft as the base limit (suppressing the abrupt post-flicker
      # surge this clamp exists for), but recovery toward set speed is no longer
      # pinned near the pre-departure follow accel for the whole window.
      window_frames = max(1, int(math.ceil(window_s / dt))) if window_s > 0.0 else self._cruise_pos_jerk_frames_left
      elapsed_s = max(0, window_frames - self._cruise_pos_jerk_frames_left) * dt
      allowed_jerk = jerk_limit + max(jerk_ramp, 0.0) * elapsed_s
      slew_ceiling = self._cruise_pos_jerk_prev_a + allowed_jerk * dt
      clipped = self.output_a_target > slew_ceiling
      if clipped:
        self.output_a_target = slew_ceiling
      self.cruise_reacquire_debug = {
        "active": True,
        "frames_left": int(self._cruise_pos_jerk_frames_left),
        "elapsed_s": float(elapsed_s),
        "allowed_jerk_mps3": float(allowed_jerk),
        "jerk_floor_mps3": float(jerk_limit),
        "slew_ceiling_mps2": float(slew_ceiling),
        "clipped": bool(clipped),
      }
      self._cruise_pos_jerk_frames_left -= 1
      if self.output_a_target >= float(self._planner_output_accel_limits[1]) - 1e-3:
        self._cruise_pos_jerk_frames_left = 0
    else:
      self.cruise_reacquire_debug = {
        "active": False,
        "frames_left": int(self._cruise_pos_jerk_frames_left),
        "elapsed_s": 0.0,
        "allowed_jerk_mps3": 0.0,
        "jerk_floor_mps3": float(jerk_limit),
        "slew_ceiling_mps2": 0.0,
        "clipped": False,
      }

    self._cruise_pos_jerk_prev_a = float(self.output_a_target)
    self._prev_mpc_source = lead_source

  def _apply_flutter_mode_clamp(self, lead_source: str, model_accel: float) -> None:
    # Bidirectional jerk clamp when the MPC source is flip-flopping at the
    # edge of lead acquisition. Counts source transitions in a rolling window;
    # when count >= N, both + and - slew are clamped. Clamp is bypassed on
    # strong modelAccel disagreement so hard braking is never delayed.
    tune_cfg = getattr(self.mpc, "_live_tune_cfg", None)
    n_trans_threshold = int(max(1, round(float(getattr(tune_cfg, "flutter_detect_transitions", 2.0) or 2.0))))
    window_s = float(getattr(tune_cfg, "flutter_detect_window_s", 1.0) or 1.0)
    jerk_cap = float(getattr(tune_cfg, "flutter_clamp_jerk_mps3", 0.0) or 0.0)
    bypass_decel = float(getattr(tune_cfg, "flutter_clamp_bypass_decel_mps2", 1.5) or 1.5)
    brake_jerk_cap = float(getattr(tune_cfg, "flutter_clamp_brake_jerk_mps3", 0.0) or 0.0)

    if jerk_cap <= 0.0 or window_s <= 0.0:
      self._source_transition_frames.clear()
      self._flutter_mode_active = False
      self._flutter_clamp_prev_a = float(self.output_a_target)
      return

    now_s = float(self.dt) * 0.0  # per-frame pseudo-time; use frame count / 1/dt
    # We don't have a wall clock here — use dt-based virtual time tracked via len
    max_frames = max(1, int(math.ceil(window_s / max(self.dt, 1e-3))))

    # Record transitions only when both old and new sources are real (skip the
    # init-empty-string -> first-source "transition" that would otherwise fire
    # on the very first frame).
    if self._flutter_prev_source and self._flutter_prev_source != lead_source:
      self._source_transition_frames.append(max_frames)
    self._flutter_prev_source = lead_source

    # Decay counters by 1 each call; drop expired.
    self._source_transition_frames = deque(
      [count - 1 for count in self._source_transition_frames if count - 1 > 0]
    )

    transitions_in_window = len(self._source_transition_frames)
    want_flutter_mode = transitions_in_window >= n_trans_threshold
    if want_flutter_mode:
      self._flutter_mode_active = True
    elif transitions_in_window == 0:
      self._flutter_mode_active = False

    if self._flutter_mode_active and jerk_cap > 0.0:
      # Bypass if the model strongly wants to brake — we should not delay real braking.
      if not (bypass_decel > 0.0 and (model_accel < -abs(bypass_decel) or self.output_a_target < -abs(bypass_decel))):
        dt = float(max(self.dt, 1e-3))
        max_step = jerk_cap * dt
        # Braking gets its own, never-tighter jerk allowance: taps are suppressed
        # by the slow release (positive cap), not by throttling brake onset.
        max_brake_step = max(brake_jerk_cap, jerk_cap) * dt
        delta = self.output_a_target - self._flutter_clamp_prev_a
        if delta > max_step:
          self.output_a_target = self._flutter_clamp_prev_a + max_step
        elif delta < -max_brake_step:
          self.output_a_target = self._flutter_clamp_prev_a - max_brake_step

    self._flutter_clamp_prev_a = float(self.output_a_target)

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
