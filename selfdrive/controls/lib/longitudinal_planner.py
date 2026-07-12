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
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import desired_follow_distance
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (get_low_speed_launch_follow_factor,
                                                                            get_low_speed_launch_follow_max_accel)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import compute_relatch_required_decel
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

# Mirror of selfdrive/controls/radard.py RADAR_TO_CAMERA (RADAR ~1.5 m ahead of
# the camera/mesh frame). modelV2.leadsV3[*].x[0] is measured in the camera frame,
# so subtract this to recover the radar-frame dRel the planner reasons about. Kept
# as a local constant to avoid pulling radard's import chain into the planner.
_RADAR_TO_CAMERA_M = 1.52
# Minimum raw model-lead prob for the CD6 EDGE1 cap to treat a leadsV3 slot as a
# real approaching lead. Below the radard Schmitt enter band (default 0.6) but well
# above sensor/model noise, so a genuinely-present lead whose prob is still ramping
# up (road 200-13: the headway is spent on exactly those pre-latch frames) arms the
# positive cap while a spurious far blip does not. The cap only ever REDUCES a
# positive accel, so a false positive here can never delay braking.
_HANDOFF_EDGE1_MODEL_PROB_FLOOR = 0.15
# The CD6 EDGE1 cap fires only when the lead is CLEARLY inside the follow distance
# (gap < this fraction of df), not merely skimming the df boundary. A lead spending
# headway (road 200-13: gap 38 m vs df ~67 m, ratio ~0.57) is deep inside; a far
# lead momentarily reading inside df only because a vLeadK rollover inflated df
# (road 200-6: gap 70 m vs df ~71 m, ratio ~0.98 at the artifact peak) is a boundary
# skim and must NOT trip the positive cap — otherwise the cap's own engagement adds
# a one-frame accel drop that the handoff sign-flip bound would then flag.
_HANDOFF_EDGE1_INSIDE_DF_FRACTION = 0.9
# Single-frame drop in the MPC cruise-owned accel cap that flags the rollover-driven
# pre-handoff cruise dive (road 200-6: cap collapses 0.86 -> 0.0). Large enough that
# only a genuine collapse — not the cap's gradual pre-collapse taper — arms the
# symmetric window.
_HANDOFF_CAP_COLLAPSE_DROP_MPS2 = 0.3
# CD7 comfort jerk envelope: a per-frame step bound at/above this (m/s^2/frame) is
# treated as effectively unbounded, so a large ComfortJerkLimitMps3 rollback
# sentinel (e.g. 50 m/s^3 * 0.05 s = 2.5 m/s^2/frame) disables the envelope
# without touching any real single-frame move (the whole accel range is ~[-4, 2]).
_COMFORT_JERK_DISABLE_STEP_MPS2 = 2.0

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
LEAD_BRAKE_RELEASE_CLOSING_COAST_MIN_ALEAD_MPS2 = -0.15

# Lookup table for turns
# Allow higher total accel (lateral+longitudinal) at low speeds and taper with speed
# Shape: 0 m/s -> 4.0 m/s^2, 20 m/s -> 2.0 m/s^2, 40 m/s+ -> 1.2 m/s^2
_A_TOTAL_MAX_V = [4.0, 2.0, 1.2]
_A_TOTAL_MAX_BP = [0., 20., 40.]


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def should_release_stop_for_lead_launch(CP, *, standstill: bool, v_ego: float,
                                        a_target: float, lead_source: str,
                                        control_leads, tuning=None,
                                        stop_min_drel: float | None = None) -> bool:
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

  # Arming gate (road 205-13 Event A): the legacy ABSOLUTE dRel gate made the
  # release latency depend on where the stop happened to settle - a published
  # 4.15 m stop needed the lead to open 0.85 m of slew-lagged published gap
  # (~1.0 s of held -2.0) before release could even arm. Departure evidence is
  # relative: arm once the published gap has RISEN LaunchReleaseDepartGateM
  # above the minimum seen during THIS stop, capped by the absolute gate
  # (whichever is smaller), with a hard 2.0 m floor so a release can never arm
  # on top of a bumper. Depart-gate sentinel >= 99 restores the pure absolute
  # gate (exact legacy arming).
  abs_gate_m = float(getattr(tuning, "launch_release_min_drel_m", LEAD_LAUNCH_RELEASE_MIN_DREL_M))
  depart_gate_m = float(getattr(tuning, "launch_release_depart_gate_m", 99.0))
  arm_gate_m = abs_gate_m
  if depart_gate_m < 99.0 and stop_min_drel is not None:
    arm_gate_m = min(abs_gate_m, float(stop_min_drel) + depart_gate_m)
  if lead_drel < max(2.0, arm_gate_m):
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
  release_reason = None
  if gap_error >= 0.0:
    if closing_speed > 0.0:
      time_to_target_s = gap_error / max(closing_speed, 1e-3)
      debug["time_to_target_s"] = float(time_to_target_s)
      if (time_to_target_s > tuning.lead_brake_release_lookahead_s and
          closing_speed <= tuning.lead_brake_release_near_target_max_closing_mps and
          lead_accel >= LEAD_BRAKE_RELEASE_CLOSING_COAST_MIN_ALEAD_MPS2):
        # If the equal-speed headway target is still several seconds away,
        # do not spend the whole surplus feather-braking. Hold near coast and
        # let the closing speed naturally reclaim the oversized gap; close or
        # decelerating leads fall through to the kinematic brake floor below.
        release_floor = tuning.lead_brake_release_coast_bias_mps2
        release_reason = "closing_coast_window"
      else:
        decel_needed = (closing_speed ** 2) / (2.0 * max(gap_error, 0.5)) + lead_decel_extra
        release_floor = -min(decel_needed, brake_decel)
        release_reason = "closing_to_target"
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
      release_reason = "gap_recovered"
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
  if release_reason is not None:
    debug["reason"] = release_reason
  elif gap_error >= 0.0:
    debug["reason"] = "closing_to_target" if closing_speed > 0.0 else "gap_recovered"
  elif near_target_branch:
    debug["reason"] = "near_target"
  else:
    debug["reason"] = "projected_recovery"
  debug["floor_mps2"] = float(release_floor)
  debug["time_to_target_s"] = float(time_to_target_s)
  return float(release_floor), debug


def should_apply_lead_brake_release_accel_floor(output_a_target: float,
                                                release_floor: float | None,
                                                release_debug: dict) -> bool:
  return release_floor is not None


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
    # Minimum published control-lead dRel seen during the CURRENT stop; the
    # departure-relative release arming gate measures gap RISE against it.
    self._stop_min_lead_drel: float | None = None
    self._prev_mpc_source: str = ""
    self._cruise_pos_jerk_frames_left: int = 0
    self._cruise_pos_jerk_prev_a: float = 0.0
    self._flutter_prev_source: str = ""
    self._source_transition_frames: deque = deque()
    self._flutter_clamp_prev_a: float = 0.0
    self._flutter_mode_active: bool = False
    # --- CD5: lost-vs-departed memory for the cruise-reacquire ramp (part b) and
    # the fresh-lead relatch obstacle blend (part a). ---
    # Ring buffer of the lead-owned slot's recent (status, modelProb, dRel) used
    # to classify WHY a lead0/lead1 -> cruise exit happened (a recoverable
    # prob-collapse vs a genuine departure). Bounded length; oldest dropped.
    self._exit_lookback: deque = deque(maxlen=32)
    # Accumulators tracked while the source is lead-owned (reset on cruise): the
    # largest single-frame drop in the PUBLISHED radarState modelProb (a gradual
    # prob-collapse stays small; a genuine departure shows an abrupt cliff) and
    # whether the published dRel ever dropped out mid-track.
    self._exit_max_prob_drop: float = 0.0
    self._exit_prev_pub_prob: float | None = None
    self._exit_drel_dropout: bool = False
    self._exit_peak_prob: float = 0.0
    # Published modelProb at the most recent frame the published lead's status
    # was True: a prob-COLLAPSE fades gradually so this is LOW at the status
    # flip; a genuine DEPARTURE vanishes abruptly with this still HIGH. This is
    # the primary collapse-vs-departure discriminator.
    self._exit_last_status_true_prob: float | None = None
    self._reacquire_exit_cause: str = "none"
    self._collapse_holdback_frames_left: int = 0
    # The physical identity (radarTrackId) and last continuous dRel of the lead
    # that owned the source just before a cruise exit. Used to arm the relatch
    # blend ONLY on a same-physical-lead re-presentation (not a fresh cut-in).
    self._exit_lead_track_id: int = -1
    self._exit_lead_last_drel: float | None = None
    # Identity of the slot that currently owns the source (snapshotted into the
    # exit-lead identity only at a lead->cruise handoff). Pending flag ensures the
    # relatch blend fires only after a genuine handoff, not on first acquisition.
    self._last_lead_owned_track_id: int = -1
    self._last_lead_owned_drel: float | None = None
    self._reacquire_armed_pending: bool = False
    # Relatch obstacle blend state (part a): a self-contained negative-leg slew
    # clamp on output_a_target for a short window after a non-urgent, same-lead
    # cruise -> lead relatch, so the fresh ObstacleCost cannot yank aTarget in a
    # single frame. CD6-independent (does not import/depend on any CD6 symbol).
    self._relatch_blend_frames_left: int = 0
    self._relatch_blend_prev_a: float = 0.0
    self._relatch_prev_src: str = ""
    # Read-only observability for the relatch blend (surfaced by the harness).
    self.relatch_blend_debug: dict = {
      "active": False,
      "frames_left": 0,
      "bypassed": False,
      "bypass_reason": "",
      "neg_cap_mps2": 0.0,
      "clipped": False,
    }
    # --- CD6: symmetric post-transition handoff limiter (road 200-6 / 200-13). ---
    # After ANY cruise<->lead source transition, a SYMMETRIC per-frame delta clamp
    # bounds |output_a_target - prev_a| for a short window so a vLeadK-rollover
    # handoff cannot sign-flip aTarget in one frame. The UPWARD leg always applies
    # (limiting accel is always safe); the DOWNWARD (braking) leg is bypassed under
    # the shared relatch urgency signal so emergency braking is never delayed. A
    # separate always-on EDGE1 cap limits positive aTarget while the cruise source
    # accelerates into a lead already inside the desired follow distance.
    self._handoff_prev_src: str = ""
    self._handoff_limit_frames_left: int = 0
    self._handoff_prev_a: float = 0.0
    # Previous frame's MPC cruise-owned accel cap. A large single-frame DROP in it
    # is the narrow signal of the rollover-driven pre-handoff cruise dive; a far
    # slow lead the MPC steadily suppresses keeps the cap ~0 (no drop) and so does
    # not arm the window.
    self._handoff_prev_cruise_cap: float | None = None
    # EDGE1 cap release-slew state: a hard positive cap keyed on a flickering model
    # prob can toggle off in one frame and let the accel jump back up. Rate-limit
    # only the cap's RELEASE (upward) leg so transient toggling cannot inject a
    # one-frame positive swing. Engagement (reducing accel) is never delayed.
    self._handoff_edge1_active: bool = False
    self._handoff_edge1_prev_out: float = 0.0
    # Read-only observability for the handoff limiter (surfaced by the harness).
    self.handoff_limit_debug: dict = {
      "active": False,
      "frames_left": 0,
      "down_bypassed": False,
      "bypass_reason": "",
      "edge1_capped": False,
      "clipped": False,
    }
    # --- CD7: graded-onset comfort anti-jerk envelope (road 200-10 / 200-9 / 201-9). ---
    # The truly-FINAL limiter (runs after the CD6 handoff limiter). While NO
    # hazard/urgency gate is active, it bounds |output_a_target - prev_a| per frame
    # to ComfortJerkLimitMps3 * dt so a single noisy vRel frame under a benign
    # steady follow cannot step-change aTarget hard. Symmetric when benign; fully
    # bypassed under any hazard/urgency signal (the shared _relatch_urgency_bypass)
    # so real braking is NEVER rate-limited. Anchors on the previous frame's final
    # output (the same value published last frame).
    self._comfort_jerk_prev_a: float = 0.0
    self._comfort_jerk_prev_src: str = ""
    self.comfort_jerk_debug: dict = {
      "active": False,
      "bypassed": False,
      "bypass_reason": "",
      "gated_reason": "",
      "max_step_mps2": 0.0,
      "clipped": False,
    }
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
      "exit_cause": "none",
      "collapse_holdback_frames_left": 0,
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
      self._stop_min_lead_drel = None

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
    launch_tuning = getattr(self.mpc, "_live_tune_cfg", None)
    launch_tuning = launch_tuning if launch_tuning is not None else LeadResponseTuningConfig.defaults()
    # Track the stop-settle gap for the departure-relative release gate: the
    # minimum published control-lead dRel seen while stopped. Reset whenever
    # the car is not at a standstill so each stop measures its own settle.
    if bool(sm['carState'].standstill):
      release_leads = getattr(self.mpc, "control_leads", ())
      release_source = str(getattr(self.mpc, "source", ""))
      release_idx = {"lead0": 0, "lead1": 1}.get(release_source)
      if release_idx is not None and release_idx < len(release_leads):
        release_lead = release_leads[release_idx]
        if release_lead is not None and bool(getattr(release_lead, "status", False)):
          release_drel = float(getattr(release_lead, "dRel", 0.0) or 0.0)
          if release_drel > 0.0:
            self._stop_min_lead_drel = release_drel if self._stop_min_lead_drel is None \
              else min(self._stop_min_lead_drel, release_drel)
    else:
      self._stop_min_lead_drel = None

    lead_launch_release_ready = output_should_stop_mpc and should_release_stop_for_lead_launch(
      self.CP,
      standstill=sm['carState'].standstill,
      v_ego=v_ego,
      a_target=output_a_target_mpc,
      lead_source=str(getattr(self.mpc, "source", "")),
      control_leads=getattr(self.mpc, "control_leads", ()),
      tuning=launch_tuning,
      stop_min_drel=self._stop_min_lead_drel,
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
    if (not self.output_should_stop and
        should_apply_lead_brake_release_accel_floor(
          output_a_target,
          lead_brake_release_floor,
          lead_brake_release_debug,
        )):
      output_a_target = max(output_a_target, float(lead_brake_release_floor))
      # The M1 kinematic slowdown ceiling must win over the release floor while
      # the lead is a corroborated threat: in the overlap state (lead
      # decelerating between the slowdown onset threshold and the release
      # path's -0.75 m/s^2 gate, or closure above the near-target limit) the
      # kinematic bound may demand more braking than the floor would allow.
      # Smaller lagged closing estimates are already safety-gated by the release
      # floor's brake-authority calculation and must not reapply the ceiling's
      # comfort-release tail to an otherwise recovered gap.
      lead_is_threatening = (
        float(lead_brake_release_debug.get("lead_accel_mps2") or 0.0) < 0.0 or
        float(lead_brake_release_debug.get("closing_mps") or 0.0) >
        float(getattr(self.mpc._live_tune_cfg, "lead_brake_release_near_target_max_closing_mps", 0.75)))
      if lead_slowdown_ceiling is not None and lead_is_threatening:
        output_a_target = min(output_a_target, float(lead_slowdown_ceiling))

    cruise_owned_accel_cap = getattr(self.mpc, "cruise_owned_accel_cap", None)
    if lead_source == "cruise" and cruise_owned_accel_cap is not None:
      output_a_target = min(output_a_target, float(cruise_owned_accel_cap))

    # Launch-follow accel FLOOR (road 205-13 Event A): once the stop latch
    # releases on a departing lead, the MPC's jerk-shaped ramp from standstill
    # takes seconds to ASK for real accel (road: demand peaked +1.02 while the
    # lead departed at +5 m/s and the driver pedaled) even though the launch
    # clip raise already PERMITS ~2.3 - and the M1 slowdown ceiling's
    # slew-limited RELEASE tail (a comfort mechanism) otherwise caps the
    # whole launch at its ramp rate. Floor the demand at the existing
    # low-speed launch-follow factor (scales by ego speed, lead speed,
    # pullaway and gap surplus; exactly 0 unless the lead is genuinely
    # pulling away) times a live-tunable ceiling. Applied AFTER the slowdown
    # ceiling and only while the lead is NOT threatening (same predicate as
    # the brake-release floor: no lead decel, no closing), so any real threat
    # keeps the ceiling's authority untouched. shouldStop gates it off
    # entirely; the factor's speed term fades it out by ~10 m/s ego.
    # Rollback sentinel: LaunchFollowAccelFloorMaxMps2 = 0 disables the floor
    # (exact legacy demand).
    launch_floor_max = float(getattr(launch_tuning, "launch_follow_accel_floor_max_mps2", 0.0) or 0.0)
    if (launch_floor_max > 0.0 and
        not self.output_should_stop and
        lead_source in ("lead0", "lead1")):
      floor_idx = 0 if lead_source == "lead0" else 1
      if floor_idx < len(control_leads):
        floor_lead = control_leads[floor_idx]
        lead_not_threatening = (
          floor_lead is not None and bool(getattr(floor_lead, "status", False)) and
          float(getattr(floor_lead, "aLeadK", 0.0) or 0.0) >= 0.0 and
          float(getattr(floor_lead, "vRel", 0.0) or 0.0) >= 0.0)
        if lead_not_threatening:
          launch_factor = get_low_speed_launch_follow_factor(
            v_ego, floor_lead, float(getattr(self.mpc, "current_t_follow", 0.0) or 0.0))
          if launch_factor > 0.0:
            # (The starting-state passthrough in longcontrol.py covers the
            # sub-vEgoStarting window at >= startAccel; this floor owns the
            # demand from there up as the factor grows with the departure.)
            output_a_target = max(output_a_target, launch_factor * launch_floor_max)

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

    # Published-radarState snapshot of the source-owned slot for exit-cause
    # classification (CD5 part b). The RAW published modelProb — not the MPC's
    # stabilized control_lead prob, which is pinned high through a collapse — is
    # what distinguishes a gradual prob-collapse (small per-frame decay) from a
    # genuine departure (an abrupt single-frame prob cliff). Snapshot the slot
    # that owns the source; fall back to leadOne.
    published_lead = None
    try:
      rs = sm['radarState']
      published_lead = rs.leadTwo if lead_source == "lead1" else rs.leadOne
    except Exception:
      published_lead = None

    model_accel_for_flutter = float(sm['modelV2'].action.desiredAcceleration) if hasattr(sm['modelV2'], 'action') else 0.0
    self._apply_cruise_reacquire_jerk_limit(lead_source, control_leads, published_lead)
    self._apply_flutter_mode_clamp(lead_source, model_accel_for_flutter)
    # CD5(a): the relatch obstacle blend is the FINAL negative-leg authority, so
    # there is exactly one binding downward slew clamp per frame. It anchors on
    # the previous frame's final output (shared prev_a semantics with the flutter
    # clamp) and only ever tightens the DOWNWARD move; braking under any urgency
    # signal bypasses it entirely (proven safe: emergency relatch test).
    self._apply_relatch_obstacle_blend(lead_source, control_leads)
    # CD6: the symmetric post-transition handoff limiter is the FINAL composed
    # limiter — it runs AFTER the relatch blend so it bounds the fully composed
    # output_a_target. It anchors on the previous frame's final output and, while
    # armed after a source flip, symmetrically bounds the per-frame delta (the
    # upward leg always applies; the downward leg is urgency-bypassed). It also
    # applies the always-on EDGE1 positive-accel cap on the cruise source.
    #
    # The EDGE1 cap keys on the raw MODEL leads (modelV2.leadsV3). On road 200-13
    # the cruise source spends positive headway on the frames where a slower lead
    # is already inside the follow distance but its model prob is still ramping
    # BELOW the radard Schmitt enter band — so it is NOT yet a control_lead or a
    # published radarState lead. The raw model lead is the only signal available
    # then, and the planner receives it (modelV2.leadsV3) exactly as in production.
    # Reduce each raw model lead to (d_rel, v_lead, prob); prob gates presence.
    model_leads: list[tuple[float, float, float]] = []
    try:
      for lv in sm['modelV2'].leadsV3:
        prob = float(getattr(lv, "prob", 0.0) or 0.0)
        xs = getattr(lv, "x", None)
        vs = getattr(lv, "v", None)
        if xs is None or vs is None or len(xs) == 0 or len(vs) == 0:
          continue
        d_rel = float(xs[0]) - _RADAR_TO_CAMERA_M
        v_lead = float(vs[0])
        model_leads.append((d_rel, v_lead, prob))
    except Exception:
      model_leads = []
    self._apply_handoff_transition_limit(lead_source, control_leads, model_leads, v_ego)
    # CD7: the graded-onset comfort anti-jerk envelope is the truly-FINAL limiter.
    # It runs AFTER the handoff limiter so it bounds the fully composed
    # output_a_target, and it is fully bypassed under any hazard/urgency signal so
    # real braking (already passed by every earlier limiter's urgency bypass) is
    # never rate-limited.
    self._apply_comfort_jerk_envelope(lead_source, control_leads)

    end_span(total_span)

  @staticmethod
  def _lead_owned_slot(lead_source: str, control_leads):
    # The _StabilizedLead object that currently owns the MPC source, or None.
    if lead_source not in ("lead0", "lead1"):
      return None
    idx = 0 if lead_source == "lead0" else 1
    if idx < len(control_leads):
      return control_leads[idx]
    return None

  def _classify_exit_cause(self, exit_prob_lo: float, abrupt_prob_drop: float) -> str:
    # Classify WHY the lead-owned slot handed off to cruise, from the PUBLISHED
    # radarState modelProb history accumulated while the slot owned the source.
    #
    # Key signal (verified on the CD5 oracle): the MPC's stabilized control_lead
    # prob is pinned high through a collapse and so cannot discriminate; the RAW
    # published modelProb does — a prob-COLLAPSE decays GRADUALLY (largest single
    # frame drop ~0.04), while a genuine DEPARTURE shows an ABRUPT cliff (a single
    # frame drop ~0.9 when the track is lost). We track the largest single-frame
    # published-prob drop plus a dRel-dropout flag while lead-owned.
    #   "collapse": no abrupt prob cliff (max single-frame drop < abrupt_prob_drop)
    #     AND the lead was genuinely latched at some point (peak prob above the
    #     exit band) AND the published dRel never dropped out AND the prob ended
    #     at/below the Schmitt exit band. Perception faded with the lead still
    #     there (road ff4).
    #   "departure": an abrupt prob cliff, a mid-track dRel dropout, or the lead
    #     never having been solidly latched — an abrupt track loss.
    # Fail SAFE toward braking on ambiguity: anything not matching the collapse
    # pattern returns "departure" (full positive ramp; the relatch blend
    # separately refuses to arm a negative-leg clamp on an unknown exit lead).
    if self._exit_last_status_true_prob is None or self._exit_prev_pub_prob is None:
      return "departure"

    was_latched = self._exit_peak_prob > exit_prob_lo
    # The collapse fingerprint: the lead was solidly latched at some point, then
    # its published prob FADED to (roughly) the Schmitt exit band by the last
    # status-True frame. `1 - abrupt_prob_drop` is the "still high" line: a
    # last-status-True prob at/above it means the track vanished abruptly with
    # the prob still high -> departure. Below it (faded) -> collapse.
    faded = self._exit_last_status_true_prob <= (1.0 - max(abrupt_prob_drop, 1e-6))
    if was_latched and faded:
      return "collapse"
    return "departure"

  def _apply_cruise_reacquire_jerk_limit(self, lead_source: str, control_leads=(), published_lead=None) -> None:
    # When the MPC's selected source transitions from lead-follow to cruise, the
    # planner's output accel can jump sharply as it seeks the set speed. Clamp the
    # upward slew for a short window so the transition feels less abrupt. Braking
    # and lead-follow acceleration are unaffected. Window ends early if output
    # reaches the cruise accel cap, so steady-state cruise is never constrained.
    tune_cfg = getattr(self.mpc, "_live_tune_cfg", None)
    jerk_limit = float(getattr(tune_cfg, "cruise_reacquire_pos_jerk_limit", 0.0) or 0.0)
    window_s = float(getattr(tune_cfg, "cruise_reacquire_jerk_window_s", 0.0) or 0.0)
    jerk_ramp = float(getattr(tune_cfg, "cruise_reacquire_jerk_ramp_mps3_per_s", 0.0) or 0.0)
    holdback_s = float(getattr(tune_cfg, "cruise_collapse_holdback_s", 0.0) or 0.0)
    lookback_frames = int(max(1, round(float(getattr(tune_cfg, "cruise_exit_lookback_frames", 7.0) or 7.0))))
    abrupt_prob_drop = float(getattr(tune_cfg, "cruise_exit_abrupt_prob_drop", 0.3) or 0.3)
    # Radard's live Schmitt exit band edge (default 0.25). Cite the live-tuned
    # value rather than a stale hardcode; the collapse classifier keys on prob
    # ending below this band after a gradual decay.
    exit_prob_lo = float(getattr(tune_cfg, "lead_prob_exit", 0.25) or 0.25)

    source_is_lead = lead_source in ("lead0", "lead1")
    dt = float(max(self.dt, 1e-3))

    lead_slot = self._lead_owned_slot(lead_source, control_leads)

    # Track the exit-cause signals from the PUBLISHED radarState modelProb while
    # the source is lead-owned: the largest single-frame prob drop (a gradual
    # prob-collapse stays small, a departure shows an abrupt cliff), the peak
    # prob (was it ever solidly latched), a mid-track dRel dropout, and a short
    # tail of the published (status, prob, dRel) so the classifier can confirm
    # the prob ended below the Schmitt exit band. All accumulators reset on
    # cruise so a stale departed history can never leak into a later exit.
    if self._exit_lookback.maxlen != max(1, lookback_frames):
      self._exit_lookback = deque(list(self._exit_lookback)[-lookback_frames:], maxlen=max(1, lookback_frames))
    if source_is_lead and published_lead is not None:
      pub_status = bool(getattr(published_lead, "status", False))
      pub_prob = float(getattr(published_lead, "modelProb", 0.0) or 0.0)
      pub_drel_raw = getattr(published_lead, "dRel", None)
      pub_drel = None if pub_drel_raw is None else float(pub_drel_raw)
      # A dRel dropout mid-track: status false / dRel unavailable / dRel ~0 while
      # we had previously seen a real gap.
      if (not pub_status) or pub_drel is None or (pub_drel is not None and pub_drel <= 0.5 and self._exit_peak_prob > 0.0):
        self._exit_drel_dropout = True
      if self._exit_prev_pub_prob is not None:
        drop = self._exit_prev_pub_prob - pub_prob
        if drop > self._exit_max_prob_drop:
          self._exit_max_prob_drop = drop
      self._exit_prev_pub_prob = pub_prob
      self._exit_peak_prob = max(self._exit_peak_prob, pub_prob)
      # Latch the published prob at the last frame the published status was True
      # (and the gap was real): the primary collapse-vs-departure discriminator.
      if pub_status and pub_drel is not None and pub_drel > 0.5:
        self._exit_last_status_true_prob = pub_prob
      self._exit_lookback.append((pub_status, pub_prob, pub_drel))
    # Track the CONTROL-lead identity of the CURRENTLY lead-owned slot. This is
    # snapshotted into the exit-lead identity only at the lead->cruise EXIT below
    # (NOT every lead frame), so the relatch same-physical-lead guard compares a
    # relatched lead against the DEPARTED lead — never against itself, which
    # would spuriously arm the blend on the very first lead acquisition.
    if source_is_lead and lead_slot is not None:
      self._last_lead_owned_track_id = int(getattr(lead_slot, "radarTrackId", -1) or -1)
      self._last_lead_owned_drel = float(getattr(lead_slot, "dRel", 0.0) or 0.0)

    if (self._prev_mpc_source in ("lead0", "lead1") and not source_is_lead and
        jerk_limit > 0.0 and window_s > 0.0):
      self._cruise_pos_jerk_frames_left = int(math.ceil(window_s / dt))
      # _cruise_pos_jerk_prev_a carries the prior frame's clipped output_a_target,
      # so we intentionally do not overwrite it here — it is our slew anchor.
      # Snapshot the DEPARTING lead's identity so the relatch blend can require a
      # same-physical-lead re-presentation, and arm the pending-relatch flag so
      # the blend only fires on a relatch that follows a genuine lead->cruise
      # handoff (never on the first lead acquisition).
      self._exit_lead_track_id = int(self._last_lead_owned_track_id)
      self._exit_lead_last_drel = (None if self._last_lead_owned_drel is None
                                   else float(self._last_lead_owned_drel))
      self._reacquire_armed_pending = True
      # Classify the exit cause from the pre-exit published-prob signals; on a
      # recoverable prob-collapse, pin the ramp at its floor for the holdback
      # window so a phantom perception dropout does not license the full
      # re-acceleration escalation a genuine departure would (road ff4).
      # Fail-safe: ambiguity returns "departure" (full ramp).
      self._reacquire_exit_cause = self._classify_exit_cause(exit_prob_lo, abrupt_prob_drop)
      if self._reacquire_exit_cause == "collapse" and holdback_s > 0.0:
        self._collapse_holdback_frames_left = int(math.ceil(holdback_s / dt))
      else:
        self._collapse_holdback_frames_left = 0

    if source_is_lead:
      self._cruise_pos_jerk_frames_left = 0
      self._reacquire_exit_cause = "none"
      # A real corroborated closing/threatening lead relatch clears the holdback
      # (a collapse followed by a genuine re-approach must NOT be held back).
      if lead_slot is not None:
        closing = -float(getattr(lead_slot, "vRel", 0.0) or 0.0)
        a_lead = float(getattr(lead_slot, "aLeadK", 0.0) or 0.0)
        if closing > 0.0 or a_lead < 0.0 or bool(getattr(lead_slot, "fcw", False)):
          self._collapse_holdback_frames_left = 0
    elif self._cruise_pos_jerk_frames_left == 0:
      # Steady cruise (not in an active reacquire window): drop the departed
      # lead history / accumulators so they cannot leak into a later exit.
      self._exit_lookback.clear()
      self._exit_max_prob_drop = 0.0
      self._exit_prev_pub_prob = None
      self._exit_drel_dropout = False
      self._exit_peak_prob = 0.0
      self._exit_last_status_true_prob = None
      # The reacquire window expired without a relatch: forget the departed lead
      # identity and disarm the pending-relatch flag so a much later, unrelated
      # cruise->lead acquisition is never treated as a relatch of this lead.
      self._reacquire_armed_pending = False
      self._exit_lead_track_id = -1
      self._exit_lead_last_drel = None

    if self._collapse_holdback_frames_left > 0:
      effective_ramp = 0.0
    else:
      effective_ramp = max(jerk_ramp, 0.0)

    if self._cruise_pos_jerk_frames_left > 0 and jerk_limit > 0.0:
      # Allowed jerk escalates the longer the handoff persists: the first frames
      # stay as soft as the base limit (suppressing the abrupt post-flicker
      # surge this clamp exists for), but recovery toward set speed is no longer
      # pinned near the pre-departure follow accel for the whole window. After a
      # prob-collapse exit the ramp term is held at 0 (pinned at the floor) for
      # the holdback window.
      window_frames = max(1, int(math.ceil(window_s / dt))) if window_s > 0.0 else self._cruise_pos_jerk_frames_left
      elapsed_s = max(0, window_frames - self._cruise_pos_jerk_frames_left) * dt
      allowed_jerk = jerk_limit + effective_ramp * elapsed_s
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
        "exit_cause": str(self._reacquire_exit_cause),
        "collapse_holdback_frames_left": int(self._collapse_holdback_frames_left),
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
        "exit_cause": str(self._reacquire_exit_cause),
        "collapse_holdback_frames_left": int(self._collapse_holdback_frames_left),
      }

    if self._collapse_holdback_frames_left > 0:
      self._collapse_holdback_frames_left -= 1

    self._cruise_pos_jerk_prev_a = float(self.output_a_target)
    self._prev_mpc_source = lead_source

  def _relatch_urgency_bypass(self, lead_slot, cfg) -> tuple[bool, str]:
    # Shared urgency predicate evaluated on the relatch ARMING frame BEFORE any
    # clamp is applied. If ANY threat signal is present, the blend is disarmed
    # and full braking passes THIS frame — no rate-limit ever delays real
    # braking. Ambiguity resolves toward MORE braking.
    # Plain getattr defaults (no `x or default`): every one of these knobs has an
    # explicit 0-disables sentinel guarded below, and `0.0 or default` silently
    # resurrects the default — the sentinel was unreachable through this read.
    urgent_ttc = float(getattr(cfg, "cruise_relatch_urgent_ttc_s", 4.0))
    urgent_closing = float(getattr(cfg, "cruise_relatch_urgent_closing_mps", 8.0))
    bypass_decel = float(getattr(cfg, "cruise_relatch_bypass_decel_mps2", -1.5))
    urgent_lead_decel = float(getattr(cfg, "cruise_relatch_urgent_lead_decel_mps2", -1.0))

    # Requested decel already at/below the bypass floor: let it through.
    if bypass_decel < 0.0 and self.output_a_target <= bypass_decel:
      return True, "requested_decel"
    if lead_slot is None:
      # No lead object to reason about -> do not risk suppressing braking.
      return True, "no_lead_obj"
    if bool(getattr(lead_slot, "fcw", False)):
      return True, "fcw"
    closing = -float(getattr(lead_slot, "vRel", 0.0) or 0.0)
    # Kinematic urgency: physics demanding at/beyond |bypass_decel| passes with
    # full authority regardless of geometry — the same floor the requested-decel
    # test honors, so "decels >= 1.5 m/s^2 are never gated" holds whether the MPC
    # requests them or the approach kinematics demand them. This replaces the
    # closing-alone test as the primary closing gate: acquiring a lead inherently
    # means closing, so raw closing speed without a distance/surplus qualifier
    # classified every routine freeway acquire (2.6-5.3 m/s at 12-21 s TTC on the
    # 2026-07-08 traces) as a threat and voided the relatch clamp exactly where
    # it was needed. Ego speed is recovered from the slot's own kinematics
    # (vRel = vLead - vEgo by radard definition) so the predicate needs no new
    # plumbing.
    if bypass_decel < 0.0:
      v_ego_slot = float(getattr(lead_slot, "vLead", 0.0) or 0.0) - float(getattr(lead_slot, "vRel", 0.0) or 0.0)
      t_follow = float(getattr(self.mpc, "current_t_follow", 1.45) or 1.45)
      required = compute_relatch_required_decel(v_ego_slot, lead_slot, t_follow, cfg)
      if required >= -bypass_decel:
        return True, "kinematic"
    # Raw-closing backstop for sensor-odd cases the surplus math may not cover
    # (default raised 2.5 -> 8.0 now that the kinematic test owns the routine
    # range; crossing it is no longer a comfort cliff because the kinematic cap
    # below has already opened proportional authority on the way there).
    if urgent_closing > 0.0 and closing >= urgent_closing:
      return True, "closing"
    d_rel = float(getattr(lead_slot, "dRel", 0.0) or 0.0)
    if closing > 1e-3:
      ttc = d_rel / max(closing, 1e-3)
      if urgent_ttc > 0.0 and ttc <= urgent_ttc:
        return True, "ttc"
    a_lead = float(getattr(lead_slot, "aLeadK", 0.0) or 0.0)
    if urgent_lead_decel < 0.0 and a_lead <= urgent_lead_decel:
      # Lead has begun braking at long range: TTC/closing/FCW lag it. Bypass the
      # blend AND the large-TTC decel cap so anticipatory braking is not clipped.
      return True, "lead_decel"
    return False, ""

  def _apply_relatch_obstacle_blend(self, lead_source: str, control_leads=()) -> None:
    # CD5(a): after a cruise -> lead relatch, the fresh ObstacleCost can slam
    # output_a_target down in a single frame (road 200-9 tap2: 2.7 m/s^2 in
    # 270 ms) even when the relatched lead is NOT a threat. Blend the NEW
    # obstacle's downward pull in over CruiseRelatchBlendS by slew-limiting only
    # the DOWNWARD move of output_a_target, and cap the relatch peak decel while
    # TTC is large. Self-contained negative-leg clamp (no CD6 dependency). The
    # upward leg is untouched (brake RELEASE is never slowed). Bypassed entirely
    # under any urgency signal so genuine braking is never delayed.
    cfg = getattr(self.mpc, "_live_tune_cfg", None)
    blend_s = float(getattr(cfg, "cruise_relatch_blend_s", 0.0) or 0.0)
    blend_jerk = float(getattr(cfg, "cruise_relatch_blend_jerk_mps3", 0.0) or 0.0)
    release_jerk = float(getattr(cfg, "cruise_relatch_release_jerk_mps3", 0.0) or 0.0)
    max_decel = float(getattr(cfg, "cruise_relatch_max_decel_mps2", 0.0) or 0.0)
    dt = float(max(self.dt, 1e-3))

    source_is_lead = lead_source in ("lead0", "lead1")
    lead_slot = self._lead_owned_slot(lead_source, control_leads)

    # Default debug (overwritten below when the blend is active or bypassed).
    debug = {"active": False, "frames_left": int(self._relatch_blend_frames_left),
             "bypassed": False, "bypass_reason": "", "neg_cap_mps2": 0.0, "clipped": False}

    # Detect a relatch: previous frame NOT lead-owned, this frame lead-owned.
    # _prev_mpc_source is updated inside _apply_cruise_reacquire_jerk_limit (runs
    # before this), so it already holds THIS frame's source; track our own edge
    # via _relatch_prev_src instead.
    prev_src = self._relatch_prev_src
    # A relatch is a cruise -> lead edge that FOLLOWS a genuine lead -> cruise
    # handoff (the reacquire window armed and is still pending). This gate is
    # what keeps the very first lead ACQUISITION (and any cruise->lead edge that
    # was not preceded by a real lead departure) from being treated as a relatch.
    is_relatch = ((prev_src not in ("lead0", "lead1")) and source_is_lead and
                  self._reacquire_armed_pending)

    if is_relatch and blend_s > 0.0 and (blend_jerk > 0.0 or release_jerk > 0.0):
      # Consume the pending flag: this handoff's relatch has now been evaluated.
      self._reacquire_armed_pending = False
      # Track-identity / cut-in guard: arm ONLY on a same-physical-lead
      # re-presentation. A fresh cut-in (new radarTrackId AND a dRel
      # discontinuity vs the pre-cruise lead, or unknown exit identity) must
      # never be negative-leg-blended — full braking passes.
      new_track = int(getattr(lead_slot, "radarTrackId", -1) or -1) if lead_slot is not None else -1
      new_drel = float(getattr(lead_slot, "dRel", 0.0) or 0.0) if lead_slot is not None else None
      same_track = (self._exit_lead_track_id >= 0 and new_track >= 0 and
                    new_track == self._exit_lead_track_id)
      drel_continuous = (self._exit_lead_last_drel is not None and new_drel is not None and
                         abs(new_drel - self._exit_lead_last_drel) <= max(3.0, 0.1 * abs(self._exit_lead_last_drel)))
      same_physical_lead = same_track or drel_continuous

      bypassed, reason = self._relatch_urgency_bypass(lead_slot, cfg)
      if same_physical_lead and not bypassed:
        self._relatch_blend_frames_left = int(math.ceil(blend_s / dt))
      else:
        self._relatch_blend_frames_left = 0
        debug = {"active": False, "frames_left": 0,
                 "bypassed": True,
                 "bypass_reason": (reason if bypassed else "cut_in"),
                 "neg_cap_mps2": 0.0, "clipped": False}

    # While armed, re-check urgency EACH frame (a lead that starts braking mid
    # blend must brake immediately) and slew-limit only the downward leg. The
    # window RIDES THROUGH brief source flaps (lead <-> cruise Schmitt chatter at
    # the relatch instant): a momentarily-absent lead_slot is NOT treated as
    # urgent, because the fresh-obstacle downward pull that this clamp exists to
    # spread persists in the MPC output across the flap. Only a present-lead
    # threat signal (FCW / closing / short-TTC / lead-decel) or a requested decel
    # at/below the bypass floor disarms.
    if self._relatch_blend_frames_left > 0 and (blend_jerk > 0.0 or release_jerk > 0.0):
      if lead_slot is not None:
        bypassed, reason = self._relatch_urgency_bypass(lead_slot, cfg)
      else:
        # Transient flap: no lead object this frame. Do not disarm on absence;
        # keep the downward slew clamp. Still honor the requested-decel bypass so
        # a genuinely hard MPC brake is never throttled.
        bypass_decel = float(getattr(cfg, "cruise_relatch_bypass_decel_mps2", -1.5))
        bypassed = bypass_decel < 0.0 and self.output_a_target <= bypass_decel
        reason = "requested_decel" if bypassed else ""
      if bypassed:
        # Disarm: full braking passes unmodified this frame.
        self._relatch_blend_frames_left = 0
        debug = {"active": False, "frames_left": 0, "bypassed": True,
                 "bypass_reason": reason or "urgent", "neg_cap_mps2": 0.0, "clipped": False}
      else:
        clipped = False
        # DOWNWARD (brake-onset) leg: jerk-capped so the fresh ObstacleCost cannot
        # slam aTarget in one frame. Safety-critical; the urgency bypass above has
        # already let any genuine threat past unmodified.
        max_down_step = blend_jerk * dt
        neg_floor = self._relatch_blend_prev_a - max_down_step
        if blend_jerk > 0.0 and self.output_a_target < neg_floor:
          self.output_a_target = neg_floor
          clipped = True
        # UPWARD (brake-RELEASE) leg: jerk-capped so the abrupt release blip as the
        # obstacle cost settles out is smoothed. ALWAYS-SAFE — it only ever keeps
        # MORE brake (delays release), never reduces braking or delays onset — so
        # it is NOT urgency-bypassed. Skips the arming frame's initial approach
        # (prev_a is the pre-relatch cruise accel, so the first downward move is
        # governed by the down leg, not this).
        if release_jerk > 0.0:
          pos_ceiling = self._relatch_blend_prev_a + release_jerk * dt
          if self.output_a_target > pos_ceiling:
            self.output_a_target = pos_ceiling
            clipped = True
        # Kinematic decel cap: while the blend is active on a non-urgent relatch,
        # peak decel is bounded by the deeper of the flat comfort floor and
        # K x the decel the approach kinematically requires. Continuous in the
        # requirement — a routine far acquire gets the tiny flat floor, a real
        # approach opens exactly proportional authority, and by the time any
        # binary bypass threshold is crossed the cap has already converged to
        # what the MPC wants, so no crossing produces a comfort cliff. The
        # urgency bypass above already removed this entirely under any threat.
        # Sentinel: CruiseRelatchKinematicHeadroom = 0 restores the flat cap.
        neg_cap = max_decel
        headroom = float(getattr(cfg, "cruise_relatch_kinematic_headroom", 0.0))
        if neg_cap < 0.0 and headroom > 0.0 and lead_slot is not None:
          v_ego_slot = float(getattr(lead_slot, "vLead", 0.0) or 0.0) - float(getattr(lead_slot, "vRel", 0.0) or 0.0)
          t_follow = float(getattr(self.mpc, "current_t_follow", 1.45) or 1.45)
          required = compute_relatch_required_decel(v_ego_slot, lead_slot, t_follow, cfg)
          if math.isfinite(required):
            neg_cap = min(neg_cap, -headroom * required)
          else:
            neg_cap = -float("inf")
        if neg_cap < 0.0 and self.output_a_target < neg_cap:
          self.output_a_target = neg_cap
          clipped = True
        debug = {"active": True, "frames_left": int(self._relatch_blend_frames_left),
                 "bypassed": False, "bypass_reason": "",
                 "neg_cap_mps2": float(neg_floor), "clipped": bool(clipped)}
        self._relatch_blend_frames_left -= 1

    self.relatch_blend_debug = debug

    # Shared prev_a anchor: the relatch blend is the FINAL negative-leg clamp, so
    # its anchor is the previous frame's final output — the same value the
    # flutter clamp will anchor on next frame. Exactly one binding downward slew
    # per frame (this one; the flutter clamp precedes it and, on a single-relatch
    # frame, cannot be active because flutter mode needs >=2 transitions).
    self._relatch_blend_prev_a = float(self.output_a_target)
    self._relatch_prev_src = lead_source

  def _apply_handoff_transition_limit(self, lead_source: str, control_leads=(),
                                      model_leads=(), v_ego: float = 0.0) -> None:
    # CD6 (road 200-6): a vLeadK rollover on a FAR non-hazard lead flips the
    # cruise<->lead0 source and the fresh obstacle sign-flips output_a_target in a
    # single 50 ms frame (+0.58 -> -0.56). This is the felt VACILLATION. Bound the
    # per-frame delta SYMMETRICALLY for a short window after ANY source flip so no
    # single frame moves aTarget more than HandoffLimitMaxDeltaMps2.
    #
    # CRITICAL ASYMMETRY IN THE BYPASS: the UPWARD (accel-increasing) leg ALWAYS
    # applies — limiting acceleration is always safe. The DOWNWARD (braking) leg is
    # bypassed under the EXACT relatch urgency signal (_relatch_urgency_bypass:
    # FCW / short-TTC / fast-close on the owned lead, or a requested hard decel) so
    # a genuine close/closing lead is braked with full authority within one frame
    # of the transition — emergency braking is NEVER delayed by this limiter.
    #
    # This is the FINAL composed limiter (runs after the relatch blend), so it
    # anchors on _handoff_prev_a (the previous frame's final output).
    cfg = getattr(self.mpc, "_live_tune_cfg", None)
    window_s = float(getattr(cfg, "handoff_limit_window_s", 0.0) or 0.0)
    max_delta = float(getattr(cfg, "handoff_limit_max_delta_mps2", 0.0) or 0.0)
    inside_df_cap = float(getattr(cfg, "handoff_inside_df_positive_cap_mps2", 10.0))
    dt = float(max(self.dt, 1e-3))

    # A source transition is a cruise<->lead OR lead0<->lead1 flip vs the previous
    # frame. Skip the init-empty-string -> first-source edge (not a real handoff).
    prev_src = self._handoff_prev_src
    real_flip = bool(prev_src) and prev_src != lead_source

    # The cruise<->lead0 handoff on a vLeadK rollover (road 200-6) is preceded by a
    # cruise-SIDE dive: as the rolled-over vRel makes a far lead momentarily look
    # like a fast-closing obstacle, the MPC's cruise-owned accel cap COLLAPSES from
    # a high value to ~0 in one frame and slams the cruise output negative — one to
    # a few frames BEFORE the source LABEL flips to lead0. Arming only on the label
    # flip misses that dive. The precise, narrow signal is that sudden cap collapse:
    # a large single-frame DROP in cruise_owned_accel_cap. This distinguishes the
    # rollover slam (cap 0.86 -> 0.0) from a far slow lead the MPC is legitimately
    # and steadily suppressing (cap already ~0, no collapse — must NOT arm, or the
    # legitimate suppression to ~0 would be held positive). Only meaningful while
    # cruise owns the source.
    cur_cruise_cap = getattr(self.mpc, "cruise_owned_accel_cap", None)
    cap_collapsed = False
    if (lead_source == "cruise" and cur_cruise_cap is not None and
        self._handoff_prev_cruise_cap is not None and
        self._handoff_prev_cruise_cap - float(cur_cruise_cap) > _HANDOFF_CAP_COLLAPSE_DROP_MPS2):
      cap_collapsed = True

    if (real_flip or cap_collapsed) and window_s > 0.0 and max_delta > 0.0:
      self._handoff_limit_frames_left = int(math.ceil(window_s / dt))
    self._handoff_prev_cruise_cap = None if cur_cruise_cap is None else float(cur_cruise_cap)

    # The lead whose threat state governs the DOWNWARD-leg bypass: the source-owned
    # slot when a lead owns the source, else (cruise-side dive) the closest closing
    # control lead. Passing the real lead (not None) matters — _relatch_urgency_bypass
    # returns "no_lead_obj" (an unconditional bypass) for None, which would let the
    # pre-handoff cruise dive through unclamped. With the real lead its FCW / TTC /
    # closing / lead-decel tests plus the requested-decel floor still bypass any
    # genuine emergency braking.
    urgency_lead = self._lead_owned_slot(lead_source, control_leads)
    if urgency_lead is None:
      for lead in control_leads:
        if lead is None or not bool(getattr(lead, "status", False)):
          continue
        if float(getattr(lead, "vRel", 0.0) or 0.0) < 0.0 or (v_ego - float(getattr(lead, "vLead", 0.0) or 0.0)) > 0.1:
          urgency_lead = lead
          break

    down_bypassed = False
    bypass_reason = ""
    windowed_clipped = False
    if self._handoff_limit_frames_left > 0 and window_s > 0.0 and max_delta > 0.0:
      target = float(self.output_a_target)
      delta = target - self._handoff_prev_a
      if delta > max_delta:
        # UPWARD (accel-increasing) leg — ALWAYS applies (limiting accel is safe).
        self.output_a_target = self._handoff_prev_a + max_delta
        windowed_clipped = True
      elif delta < -max_delta and target < 0.0:
        # DOWNWARD (braking) leg. The CD6 defect is a SIGN FLIP into braking, so
        # only bound the descent once the target actually enters braking territory
        # (target < 0). A drop that merely REDUCES positive accel toward a floor
        # (e.g. the MPC's own lead-present cruise-accel cap suppressing accel to ~0
        # as a far slow lead appears) is legitimate comfort behavior — never a felt
        # slam — and passes unclamped. Braking under the shared relatch urgency
        # signal is bypassed so genuine emergency braking is never delayed.
        bypassed, reason = self._relatch_urgency_bypass(urgency_lead, cfg)
        down_bypassed = bool(bypassed)
        bypass_reason = reason
        if not bypassed:
          self.output_a_target = self._handoff_prev_a - max_delta
          windowed_clipped = True
      self._handoff_limit_frames_left -= 1
      # End the window early once the transition has settled (the raw move is
      # already within the per-frame bound), so steady state is never constrained.
      if abs(delta) <= max_delta:
        self._handoff_limit_frames_left = 0

    # EDGE1 cap (always-on, NOT windowed; road 200-13 phase-1): while the cruise
    # source owns control and a slower lead is already INSIDE the desired follow
    # distance on a closing (ego-faster) trend, cap positive output_a_target so the
    # planner stops spending headway accelerating into the sub-target lead before it
    # latches. It keys on the raw MODEL leads (modelV2.leadsV3), NOT the stabilized
    # control_leads or the radard-published radarState leads: on the road the
    # headway is spent precisely on the frames where the lead's model prob is still
    # ramping BELOW the Schmitt enter band, so it is not yet a control lead nor a
    # published radarState lead — a control_leads-only check misses exactly the
    # frames that spend it. A modest prob floor keeps far spurious model blips from
    # arming the cap. Uses desired_follow_distance from the same module the oracle
    # imports. Positive-only (never adds braking), so it cannot delay any decel.
    # Rollback: a large cap (spec max 10) makes this unreachable.
    edge1_capped = False
    t_follow = float(getattr(self.mpc, "current_t_follow", 0.0) or 0.0)
    if lead_source == "cruise" and self.output_a_target > inside_df_cap and v_ego > 1.0:
      for d_rel, v_lead, prob in model_leads:
        # Presence floor: ignore absent / very-low-confidence model blips.
        if prob < _HANDOFF_EDGE1_MODEL_PROB_FLOOR:
          continue
        # Closing = ego faster than the model lead speed (the negative-vRel trend
        # the road exhibited). A lead pulling away is never capped.
        closing = v_ego - v_lead
        if closing <= 0.1:
          continue
        df = desired_follow_distance(v_ego, v_lead, t_follow=t_follow)
        if 0.0 < d_rel < df * _HANDOFF_EDGE1_INSIDE_DF_FRACTION:
          self.output_a_target = min(self.output_a_target, inside_df_cap)
          edge1_capped = True
          break

    # EDGE1 RELEASE slew: when the cap disengages (e.g. a flickering model prob dips
    # below the presence floor for a frame, or the lead crosses the df boundary),
    # the accel would otherwise jump straight back to the uncapped cruise value in
    # one frame. Rate-limit ONLY that upward release by max_delta so a transient
    # toggle cannot inject a positive one-frame swing. Release-only and upward-only:
    # it never reduces accel and never delays braking. Uses the same max_delta as
    # the windowed limiter; if the limiter is disabled (window/max_delta 0) there is
    # nothing to smooth against, so it no-ops.
    if (not edge1_capped and self._handoff_edge1_active and max_delta > 0.0 and
        self.output_a_target > self._handoff_edge1_prev_out + max_delta):
      self.output_a_target = self._handoff_edge1_prev_out + max_delta
      edge1_release_slewed = True
    else:
      edge1_release_slewed = False
    # The cap is "active" for release-slew purposes while it clips OR while its
    # release is still being slewed toward the uncapped value.
    self._handoff_edge1_active = bool(edge1_capped or edge1_release_slewed)
    self._handoff_edge1_prev_out = float(self.output_a_target)

    self.handoff_limit_debug = {
      "active": bool(self._handoff_limit_frames_left > 0 or windowed_clipped),
      "frames_left": int(self._handoff_limit_frames_left),
      "down_bypassed": bool(down_bypassed),
      "bypass_reason": str(bypass_reason),
      "edge1_capped": bool(edge1_capped),
      "clipped": bool(windowed_clipped or edge1_capped),
    }

    self._handoff_prev_a = float(self.output_a_target)
    self._handoff_prev_src = lead_source

  def _apply_comfort_jerk_envelope(self, lead_source: str, control_leads=()) -> None:
    # CD7 (road 200-10 / 200-9 tap1 / 201-9): under a benign STEADY single-source
    # LEAD follow the MPC QP output can step-change output_a_target hard on a single
    # noisy vRel frame (road: -0.31 -> -1.00 in 0.15 s, ~4.6 m/s^3) even though
    # nothing hazardous is happening - the felt THROTTLE_BLIP_JERK / VACILLATION.
    # No earlier limiter bounds this: the handoff/relatch/flutter clamps only arm
    # on a source flip or flutter, and none of those is happening here. Bound
    # |output_a_target - prev_a| per frame to ComfortJerkLimitMps3 * dt as the
    # truly-FINAL composed limiter.
    #
    # SCOPE GATE (narrow, non-regressing): the envelope engages ONLY in the exact
    # steady-lead-follow regime the CD7 blip lives in - a lead0/lead1 source with
    # NO other composed limiter or cap doing legitimate fast work this frame. It is
    # gated OFF (raw output passes) whenever the source is cruise-owned (the cruise
    # accel cap does legitimate one-frame accel SUPPRESSION toward ~0 on a far slow
    # lead), a source flip just occurred, or the CD6 handoff limiter / EDGE1 cap /
    # CD5 relatch blend / flutter clamp is engaged this frame. Those limiters make
    # deliberate fast moves (a graded handoff delta, a cap engagement) that an
    # unconditional per-frame bound would over-smooth and defeat - so the envelope
    # defers to them and only shaves the pure steady-follow spike they never touch.
    #
    # CRITICAL SAFETY: even inside its regime the envelope is FULLY BYPASSED
    # whenever ANY hazard/urgency signal is active - the SAME _relatch_urgency_bypass
    # predicate CD5/CD6 use (FCW / short-TTC / fast-close on the owned lead, or a
    # requested hard decel). So genuine braking and cut-in response pass THIS frame
    # unmodified; only the benign steady-follow comfort-brake blip is graded.
    #
    # ASYMMETRIC (down-leg only): the bound applies ONLY to the DOWNWARD
    # (comfort-braking-onset) move - the road's headline defect (a sudden unnecessary
    # brake, 200-10: -0.31 -> -1.00) and the safety-relevant felt jerk. The UPWARD
    # leg (brake-RELEASE and re-accel toward a followed lead) is always-safe and must
    # stay fast, so it is left free (matching CD5's relatch blend and the flutter
    # clamp). This is what keeps the legitimate managed release/re-accel moves from
    # being blunted while still killing the felt comfort-brake blip.
    #
    # It only grades the ONSET of the sustained downward move (a step into a brake):
    # once the output holds the value for a couple frames prev_a catches up and
    # Delta -> 0, so no sustained comfort floor is lowered.
    cfg = getattr(self.mpc, "_live_tune_cfg", None)
    jerk_limit = float(getattr(cfg, "comfort_jerk_limit_mps3", 0.0) or 0.0)
    bypass_decel = float(getattr(cfg, "comfort_jerk_bypass_decel_mps2", -1.5) or -1.5)
    dt = float(max(self.dt, 1e-3))
    max_step = jerk_limit * dt

    prev_src = self._comfort_jerk_prev_src
    self._comfort_jerk_prev_src = lead_source

    # Disabled (rollback sentinel: jerk_limit <= 0, or a large value whose
    # per-frame bound is unreachable): pass through, keep the anchor fresh.
    if jerk_limit <= 0.0 or max_step >= _COMFORT_JERK_DISABLE_STEP_MPS2:
      self.comfort_jerk_debug = {"active": False, "bypassed": False, "bypass_reason": "",
                                 "gated_reason": "disabled", "max_step_mps2": float(max_step), "clipped": False}
      self._comfort_jerk_prev_a = float(self.output_a_target)
      return

    # SCOPE GATE: engage only in the steady-lead-follow regime. Any of these means
    # another mechanism is legitimately shaping the output this frame - defer.
    gated_reason = ""
    if lead_source not in ("lead0", "lead1"):
      gated_reason = "not_lead_source"       # cruise cap does legitimate suppression
    elif prev_src != lead_source:
      gated_reason = "source_transition"     # a handoff frame the CD6 limiter owns
    elif bool(self.handoff_limit_debug.get("active")) or bool(self.handoff_limit_debug.get("clipped")):
      gated_reason = "handoff_limiter"       # CD6 windowed clamp / EDGE1 cap engaged
    elif bool(self.relatch_blend_debug.get("active")):
      gated_reason = "relatch_blend"         # CD5 relatch blend engaged
    elif bool(self._flutter_mode_active):
      gated_reason = "flutter_mode"          # flutter clamp engaged
    if gated_reason:
      self.comfort_jerk_debug = {"active": False, "bypassed": False, "bypass_reason": "",
                                 "gated_reason": gated_reason, "max_step_mps2": float(max_step), "clipped": False}
      self._comfort_jerk_prev_a = float(self.output_a_target)
      return

    # In-regime: the source is lead-owned, so the urgency lead IS the source-owned
    # slot. Passing the real lead (not None) matters - _relatch_urgency_bypass
    # returns an unconditional bypass for None, which would defeat the envelope.
    urgency_lead = self._lead_owned_slot(lead_source, control_leads)

    # Hazard/urgency bypass (the SAME predicate as CD5/CD6). If ANY threat signal
    # is present, the envelope is disarmed and the raw output passes unmodified so
    # genuine braking is never rate-limited. With no lead object this frame (a
    # transient slot flap while still lead-source), honor only the requested-decel
    # floor so a hard MPC brake is never throttled; a benign flap is still graded.
    if urgency_lead is not None:
      bypassed, reason = self._relatch_urgency_bypass(urgency_lead, cfg)
      # _relatch_urgency_bypass uses cruise_relatch_bypass_decel_mps2 for its
      # requested-decel floor; also honor the CD7-specific floor so the envelope's
      # own configured hard-brake bypass applies.
      if not bypassed and bypass_decel < 0.0 and self.output_a_target <= bypass_decel:
        bypassed, reason = True, "requested_decel"
    else:
      bypassed = bypass_decel < 0.0 and self.output_a_target <= bypass_decel
      reason = "requested_decel" if bypassed else ""

    clipped = False
    if bypassed:
      self.comfort_jerk_debug = {"active": True, "bypassed": True, "bypass_reason": reason or "urgent",
                                 "gated_reason": "", "max_step_mps2": float(max_step), "clipped": False}
    else:
      # ASYMMETRIC (down-leg only): bound only the DOWNWARD (comfort-braking-onset)
      # move - the road's headline CD7 defect (a sudden unnecessary brake blip,
      # 200-10: -0.31 -> -1.00) and the safety-relevant felt jerk. The UPWARD leg
      # (brake-RELEASE and re-accel toward a followed lead) is always-safe (it only
      # ever reduces braking / matches a lead) and must stay fast, so it is left
      # free - matching the CD5 relatch blend and flutter clamp precedent, and so
      # the legitimate managed release/re-accel moves are never blunted.
      delta = self.output_a_target - self._comfort_jerk_prev_a
      if delta < -max_step:
        self.output_a_target = self._comfort_jerk_prev_a - max_step
        clipped = True
      self.comfort_jerk_debug = {"active": True, "bypassed": False, "bypass_reason": "",
                                 "gated_reason": "", "max_step_mps2": float(max_step), "clipped": bool(clipped)}

    self._comfort_jerk_prev_a = float(self.output_a_target)

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
