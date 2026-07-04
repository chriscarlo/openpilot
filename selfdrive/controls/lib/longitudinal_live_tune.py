#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

try:
  from openpilot.common.params import Params
except Exception:
  # Local test/dev fallback when openpilot runtime deps are unavailable.
  class Params:  # type: ignore[no-redef]
    def get(self, _key: str):
      return None

    def put(self, _key: str, _value: Any) -> None:
      raise KeyError(_key)

    def remove(self, _key: str) -> None:
      raise KeyError(_key)


@dataclass(frozen=True)
class LeadResponseTuneSpec:
  attr: str
  key: str
  cli_name: str
  label: str
  default: float
  minimum: float
  maximum: float
  description: str

  def clamp(self, value: float) -> float:
    return float(min(max(float(value), self.minimum), self.maximum))


LEAD_RESPONSE_TUNE_SPECS = (
  LeadResponseTuneSpec(
    attr="lead_preview_strength",
    key="Longitudinal.LiveTune.LeadPreviewStrength",
    cli_name="lead-preview-strength",
    label="preview_strength",
    default=1.5,
    minimum=0.0,
    maximum=2.0,
    description="Scale factor for how early a newly recognized slower lead starts shaping decel.",
  ),
  LeadResponseTuneSpec(
    attr="lead_preview_gap_min_m",
    key="Longitudinal.LiveTune.LeadPreviewGapMinM",
    cli_name="lead-preview-gap-min-m",
    label="preview_gap_min_m",
    default=1.0,
    minimum=0.0,
    maximum=10.0,
    description="Minimum extra slack above nominal headway before preview logic activates.",
  ),
  LeadResponseTuneSpec(
    attr="lead_preview_max_buffer_m",
    key="Longitudinal.LiveTune.LeadPreviewMaxBufferM",
    cli_name="lead-preview-max-buffer-m",
    label="preview_max_buffer_m",
    default=10.0,
    minimum=0.0,
    maximum=25.0,
    description="Upper cap on how much closer the previewed lead obstacle can be pulled.",
  ),
  LeadResponseTuneSpec(
    attr="lead_preview_min_speed_mps",
    key="Longitudinal.LiveTune.LeadPreviewMinSpeedMps",
    cli_name="lead-preview-min-speed",
    label="preview_min_speed",
    default=6.0,
    minimum=0.0,
    maximum=8.0,
    description="Ego speed (m/s) at which the lead-approach preview fades to zero; it ramps linearly up to full strength "
                "at 8.0 m/s (LEAD_APPROACH_PREVIEW_MIN_SPEED). Replaces the historical hard cut at 8.0, which stepped the "
                "previewed obstacle by up to LeadPreviewMaxBufferM instantly when the planner's filtered speed wobbled "
                "across 8.0. Activity superset at defaults: behavior at/above 8.0 m/s is unchanged; below it the fade only "
                "adds preview that used to be zero. Rollback sentinel: 8.0 (the spec maximum) reproduces the legacy hard "
                "cut exactly.",
  ),
  LeadResponseTuneSpec(
    attr="lead_acquire_window_s",
    key="Longitudinal.LiveTune.LeadAcquireWindowS",
    cli_name="lead-acquire-window-s",
    label="lead_acquire_window_s",
    default=1.5,
    minimum=0.0,
    maximum=3.0,
    description="Short boost window after a lead appears or jumps materially closer/slower.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_strength",
    key="Longitudinal.LiveTune.GapReclaimStrength",
    cli_name="gap-reclaim-strength",
    label="reclaim_strength",
    default=0.55,
    minimum=0.0,
    maximum=2.0,
    description="Scale factor for how eagerly ACC closes a safe extra gap when the lead is pulling away.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_gap_min_m",
    key="Longitudinal.LiveTune.GapReclaimGapMinM",
    cli_name="gap-reclaim-gap-min-m",
    label="reclaim_gap_min_m",
    default=3.0,
    minimum=0.0,
    maximum=10.0,
    description="Minimum extra slack above nominal headway before gap reclaim is allowed.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_max_accel",
    key="Longitudinal.LiveTune.GapReclaimMaxAccel",
    cli_name="gap-reclaim-max-accel",
    label="reclaim_max_accel",
    default=0.30,
    minimum=0.0,
    maximum=0.75,
    description="Cap on the positive accel floor used to close a safe extra gap.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_follow_max_accel",
    key="Longitudinal.LiveTune.GapReclaimFollowMaxAccel",
    cli_name="gap-reclaim-follow-max-accel",
    label="reclaim_follow_max_accel",
    default=0.32,
    minimum=0.0,
    maximum=1.5,
    description="Follow-regime cap on the positive accel floor used to match a pulling-away lead's speed once the "
                "vRel-aware follow target is recovered (the brake-release path's recovered branch). Separate from the "
                "shared GapReclaimMaxAccel (which stays at its default for the launch/stoplight reclaim and the Hyundai "
                "lead-to-cruise transition floor). The raise above the coast bias phases in with lead pullaway speed and "
                "is tapered by the kinematic overshoot bound (GapReclaimTaperGain). Rollback sentinel: any value at or "
                "below LeadBrakeReleaseCoastBiasMps2 (e.g. 0) disables the raise, restoring the pre-fix coast floor.",
  ),
  LeadResponseTuneSpec(
    attr="gap_reclaim_taper_gain",
    key="Longitudinal.LiveTune.GapReclaimTaperGain",
    cli_name="gap-reclaim-taper-gain",
    label="reclaim_taper_gain",
    default=2.0,
    minimum=0.0,
    maximum=20.0,
    description="Kinematic overshoot taper for the follow-regime gap reclaim floor, in 1/(m/s^2): the extra authority "
                "above the coast bias is scaled by clip(1 - gain * c_proj^2 / (2 * gap_surplus), 0, 1) where c_proj is "
                "the closing speed projected from the current commanded accel over 1.2 s. Larger = extra authority "
                "fades sooner as projected closure grows (strictly less accel than a naive raise). 0 disables only the "
                "taper (naive raise, diagnostic); use GapReclaimFollowMaxAccel = 0 to roll the whole raise back.",
  ),
  LeadResponseTuneSpec(
    attr="lead_keepup_strength",
    key="Longitudinal.LiveTune.LeadKeepUpStrength",
    cli_name="lead-keepup-strength",
    label="keepup_strength",
    default=1.15,
    minimum=0.0,
    maximum=2.0,
    description="Scale for the keep-up accel floor: tiny at first, then able to climb to its cap as pull-away is confirmed.",
  ),
  LeadResponseTuneSpec(
    attr="lead_keepup_gap_min_m",
    key="Longitudinal.LiveTune.LeadKeepUpGapMinM",
    cli_name="lead-keepup-gap-min-m",
    label="keepup_gap_min_m",
    default=0.10,
    minimum=0.0,
    maximum=5.0,
    description="Extra gap above nominal headway before keep-up distance bias starts; pull-away speed can still start it softly.",
  ),
  LeadResponseTuneSpec(
    attr="lead_keepup_max_accel",
    key="Longitudinal.LiveTune.LeadKeepUpMaxAccel",
    cli_name="lead-keepup-max-accel",
    label="keepup_max_accel",
    default=0.22,
    minimum=0.0,
    maximum=5.0,
    description="Cap on the keep-up accel floor. Planner/personality accel limits still apply, so leave the default tiny for EV comfort.",
  ),
  LeadResponseTuneSpec(
    attr="lead_slowdown_strength",
    key="Longitudinal.LiveTune.LeadSlowdownStrength",
    cli_name="lead-slowdown-strength",
    label="slowdown_strength",
    default=0.25,
    minimum=0.0,
    maximum=2.0,
    description="Scale for the slower/braking-lead accel ceiling: soft at onset, full authority for confirmed stop threats.",
  ),
  LeadResponseTuneSpec(
    attr="lead_slowdown_max_decel",
    key="Longitudinal.LiveTune.LeadSlowdownMaxDecel",
    cli_name="lead-slowdown-max-decel",
    label="slowdown_max_decel",
    default=4.0,
    minimum=0.0,
    maximum=6.0,
    description="Maximum braking magnitude the slower/braking-lead ceiling may request before vehicle/controller limits apply.",
  ),
  LeadResponseTuneSpec(
    attr="lead_slowdown_kinematic_headroom",
    key="Longitudinal.LiveTune.LeadSlowdownKinematicHeadroom",
    cli_name="lead-slowdown-kinematic-headroom",
    label="slowdown_kinematic_headroom",
    default=1.5,
    minimum=1.0,
    maximum=5.0,
    description="Multiple of the kinematically required stop decel the slowdown danger term may demand; caps the danger-surplus collapse on calm stops without limiting genuine short-gap threats.",
  ),
  LeadResponseTuneSpec(
    attr="lead_slowdown_kinematic_margin_m",
    key="Longitudinal.LiveTune.LeadSlowdownKinematicMarginM",
    cli_name="lead-slowdown-kinematic-margin-m",
    label="slowdown_kinematic_margin",
    default=4.0,
    minimum=1.0,
    maximum=5.0,
    description="Gap reserve for the slowdown kinematic bound: the danger term is uncapped (full authority) once the lead is "
                "projected to stop inside this distance. Range is enforced again in code (1.0 to STOP_DISTANCE-1.0 = 5.0): at "
                "6 m the bound inflates near the natural stop point and readmits the calm-stop slam; below 1 m the bound gives "
                "LESS ceiling braking and the inside-margin full-authority restoration is unreachable.",
  ),
  LeadResponseTuneSpec(
    attr="lead_slowdown_kinematic_oncoming_vlead_mps",
    key="Longitudinal.LiveTune.LeadSlowdownKinematicOncomingVLeadMps",
    cli_name="lead-slowdown-kinematic-oncoming-vlead",
    label="slowdown_kinematic_oncoming_vlead",
    default=-2.5,
    minimum=-6.0,
    maximum=-1.5,
    description="Published vLead (m/s) below which the slowdown kinematic bound is bypassed entirely (full legacy danger "
                "authority for oncoming/reversing leads, whose true closure the max(0, vLead) clamp would understate). Must "
                "stay clearly below the ~-1.2 m/s near-stop vRel-boost artifact (max -1.5) or the calm-stop fix is defeated; "
                "more negative than -6 would deny genuinely oncoming leads their uncapped authority.",
  ),
  LeadResponseTuneSpec(
    attr="lead_handoff_stopping_need_decel_mps2",
    key="Longitudinal.LiveTune.LeadHandoffStoppingNeedDecelMps2",
    cli_name="lead-handoff-stopping-need-decel",
    label="handoff_stopping_need_decel",
    default=0.80,
    minimum=0.05,
    maximum=1e9,
    description="Kinematic stopping-need cruise->lead obstacle handoff threshold (m/s^2) at/below the reference speed: "
                "hand the MPC the lead obstacle as soon as stopping STOP_DISTANCE short of the lead requires at least "
                "this decel (scaled by max(1, v_ego/LeadHandoffStoppingNeedRefSpeedMps) above the reference speed). "
                "Fixes the midband (~18 mph) stop slam where every early handoff leg is structurally blocked and "
                "braking starts at ~21 m. OR'd into raw_requires_owner, so it can only make the handoff earlier. "
                "Legacy rollback sentinel: 1e9 (the spec maximum deliberately admits it) makes the leg unreachable and "
                "restores the pre-fix handoff exactly.",
  ),
  LeadResponseTuneSpec(
    attr="lead_handoff_stopping_need_ref_speed_mps",
    key="Longitudinal.LiveTune.LeadHandoffStoppingNeedRefSpeedMps",
    cli_name="lead-handoff-stopping-need-ref-speed",
    label="handoff_stopping_need_ref_speed",
    default=8.0,
    minimum=1.0,
    maximum=1e9,
    description="Reference ego speed (m/s) for the stopping-need handoff threshold: below it the threshold is the flat "
                "base value; above it the threshold scales by v_ego/ref (a constant-time-headway trigger for stopped "
                "leads). Bounds ghost/false-positive exposure at highway speed to ranges where the required decel is "
                "genuinely proportional, and keeps 9-10 m/s calm stops inside the human stop-gap window. Sentinel: 1e9 "
                "(the spec maximum deliberately admits it) disables scaling entirely (flat threshold at every speed).",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_min_speed_mps",
    key="Longitudinal.LiveTune.LeadBrakeReleaseMinSpeedMps",
    cli_name="lead-brake-release-min-speed",
    label="release_min_speed",
    default=5.0,
    minimum=0.0,
    maximum=20.0,
    description="Minimum ego speed for the Vibe follow-gap brake-release floor.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_brake_deficit_margin_m",
    key="Longitudinal.LiveTune.LeadBrakeReleaseBrakeDeficitMarginM",
    cli_name="lead-brake-release-deficit-margin",
    label="release_deficit_margin",
    default=1.5,
    minimum=0.0,
    maximum=10.0,
    description="Allowed relative-braking-distance deficit before the release floor stays disabled.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_lookahead_s",
    key="Longitudinal.LiveTune.LeadBrakeReleaseLookaheadS",
    cli_name="lead-brake-release-lookahead",
    label="release_lookahead",
    default=2.0,
    minimum=0.1,
    maximum=6.0,
    description="Lookahead window for easing decel when an opening gap is projected to recover the Vibe headway target.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_min_pullaway_mps",
    key="Longitudinal.LiveTune.LeadBrakeReleaseMinPullawayMps",
    cli_name="lead-brake-release-min-pullaway",
    label="release_min_pullaway",
    default=0.10,
    minimum=0.0,
    maximum=3.0,
    description="Minimum opening speed before projected Vibe target recovery can ease continued braking.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_near_target_margin_m",
    key="Longitudinal.LiveTune.LeadBrakeReleaseNearTargetMarginM",
    cli_name="lead-brake-release-near-target-margin",
    label="release_near_target_margin",
    default=1.5,
    minimum=0.0,
    maximum=8.0,
    description="Vibe headway deficit that can be treated as near target when closing speed is small.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_near_target_max_closing_mps",
    key="Longitudinal.LiveTune.LeadBrakeReleaseNearTargetMaxClosingMps",
    cli_name="lead-brake-release-near-target-closing",
    label="release_near_target_closing",
    default=0.75,
    minimum=0.0,
    maximum=4.0,
    description="Maximum closing speed still eligible for the near-target decel floor.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_near_target_floor_mps2",
    key="Longitudinal.LiveTune.LeadBrakeReleaseNearTargetFloorMps2",
    cli_name="lead-brake-release-near-target-floor",
    label="release_near_target_floor",
    default=-0.05,
    minimum=-2.0,
    maximum=0.5,
    description="Accel floor applied near the recovered Vibe headway target; can be slightly positive to counter regen.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_lead_decel_min_mps2",
    key="Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelMinMps2",
    cli_name="lead-brake-release-lead-decel-min",
    label="release_lead_decel_min",
    default=-0.75,
    minimum=-6.0,
    maximum=0.0,
    description="Disable brake release when the lead is braking harder than this threshold.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_approach_floor_mps2",
    key="Longitudinal.LiveTune.LeadBrakeReleaseApproachFloorMps2",
    cli_name="lead-brake-release-approach-floor",
    label="release_approach_floor",
    default=-0.60,
    minimum=-6.0,
    maximum=0.0,
    description="Most braking allowed by the projected-recovery release floor before it ramps toward near-coast.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_lead_decel_project_gain",
    key="Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelProjectGain",
    cli_name="lead-brake-release-lead-decel-project-gain",
    label="release_lead_decel_project_gain",
    default=1.0,
    minimum=0.0,
    maximum=2.0,
    description="CD1 fix (road 200-15-17 TAP 1): scale on the lead's own deceleration magnitude added to the required "
                "ego decel in the brake-release floor's closing and near-target branches, so a lead braking to a stop "
                "at a decel too shallow to trip LeadBrakeReleaseLeadDecelMinMps2 (road aLeadK -0.42..-0.63, inside the "
                "-0.75 veto) can no longer clip the MPC's ramping brake above what the still-decelerating lead demands. "
                "At 1.0 the floor uses the exact relative-frame requirement (legacy closure decel + |aLeadK|); a steady "
                "or accelerating lead contributes zero regardless of the gain, so all non-decelerating states keep the "
                "shipped floor bit-identically. Rollback sentinel: 0 restores the pre-fix instantaneous-closing floor.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_vrel_credit_cap_m",
    key="Longitudinal.LiveTune.LeadBrakeReleaseVrelCreditCapM",
    cli_name="lead-brake-release-vrel-credit-cap",
    label="release_vrel_credit_cap",
    default=10.0,
    minimum=0.0,
    maximum=20.0,
    description="Cap (m) on the vRel-aware gap-error credit max(0, vLead^2 - vEgo^2) / (2 * COMFORT_BRAKE) granted to the "
                "brake-release path when the lead is corroborated faster than ego (both vLead and vEgo + vRel must agree, "
                "so closing states get zero credit and keep the legacy headway-based eligibility bit-identically). Aligns "
                "release eligibility with the MPC's own vRel-aware desired_follow_distance target so the planner stops "
                "holding brake after the MPC's gap is already recovered; the cap keeps a misassociated much-faster lead "
                "from buying a coast floor at any distance. Rollback sentinel: 0 restores the pure headway gap error.",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_recovery_proj_s",
    key="Longitudinal.LiveTune.LeadBrakeReleaseRecoveryProjS",
    cli_name="lead-brake-release-recovery-proj",
    label="release_recovery_proj",
    default=3.0,
    minimum=0.0,
    maximum=5.0,
    description="Horizon (s) by which the brake-release vRel-credit basis projects the lead speed forward using the "
                "POSITIVE part of aLeadK only (a decelerating or steady lead gets zero projection). Compensates the "
                "tracker's vRel lag behind aLeadK when a lead finishes a transient slowdown, so the release floor stops "
                "holding brake ~0.5 s late; the projected credit still saturates at LeadBrakeReleaseVrelCreditCapM. "
                "Rollback sentinel: 0 disables the projection (speed-signal-only credit).",
  ),
  LeadResponseTuneSpec(
    attr="lead_brake_release_coast_bias_mps2",
    key="Longitudinal.LiveTune.LeadBrakeReleaseCoastBiasMps2",
    cli_name="lead-brake-release-coast-bias",
    label="release_coast_bias",
    default=0.05,
    minimum=-0.5,
    maximum=0.8,
    description="Accel floor once the Vibe headway target is recovered and ego is no longer closing.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_duration_s",
    key="Longitudinal.LiveTune.CutInSettleDurationS",
    cli_name="cutin-settle-duration-s",
    label="cutin_settle_duration_s",
    default=6.0,
    minimum=0.0,
    maximum=12.0,
    description="Grace-window length for a benign cut-in before the planner fully returns to nominal headway.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_max_decel",
    key="Longitudinal.LiveTune.CutInSettleMaxDecel",
    cli_name="cutin-settle-max-decel",
    label="cutin_settle_max_decel",
    default=0.15,
    minimum=0.0,
    maximum=0.80,
    description="Maximum braking magnitude allowed during the cut-in grace window after it ramps in.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_max_closing_speed_mps",
    key="Longitudinal.LiveTune.CutInSettleMaxClosingSpeedMps",
    cli_name="cutin-settle-max-closing-speed-mps",
    label="cutin_settle_max_closing_speed_mps",
    default=2.2,
    minimum=0.5,
    maximum=6.0,
    description="Largest ego-minus-lead closing speed that can still qualify for cut-in grace.",
  ),
  LeadResponseTuneSpec(
    attr="cutin_settle_accel_bias_mps2",
    key="Longitudinal.LiveTune.CutInSettleAccelBiasMps2",
    cli_name="cutin-settle-accel-bias",
    label="cutin_settle_accel_bias",
    default=0.12,
    minimum=0.0,
    maximum=0.30,
    description="Positive accel bias added to settle floor to counteract EV regen braking on coast.",
  ),
  LeadResponseTuneSpec(
    attr="virtual_lead_slow_tau_s",
    key="Longitudinal.LiveTune.VirtualLeadSlowTauS",
    cli_name="virtual-lead-slow-tau",
    label="vl_slow_tau",
    default=1.30,
    minimum=0.10,
    maximum=3.0,
    description="EMA time constant for lead kinematics (aLeadK) in the safe/noise-rejection direction. "
                "Lower = faster response to lead accel changes, more model noise passed through. "
                "Sign transitions (decel-to-accel) always use a faster fixed tau regardless of this value.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_tau_close_s",
    key="Longitudinal.LiveTune.DRelFilterTauCloseS",
    cli_name="drel-filter-tau-close",
    label="drel_tau_close",
    default=0.30,
    minimum=0.05,
    maximum=2.0,
    description="dRel filter time constant for closing (lead appears nearer). Lower = faster safety response.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_tau_open_s",
    key="Longitudinal.LiveTune.DRelFilterTauOpenS",
    cli_name="drel-filter-tau-open",
    label="drel_tau_open",
    default=0.80,
    minimum=0.10,
    maximum=5.0,
    description="dRel filter time constant for opening (lead appears farther). Higher = more noise rejection.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_open_slew_max_mps",
    key="Longitudinal.LiveTune.DRelFilterOpenSlewMaxMps",
    cli_name="drel-filter-open-slew-max-mps",
    label="drel_open_slew_max_mps",
    default=1.80,
    minimum=0.25,
    maximum=5.0,
    description="Max opening-side dRel motion the filter will admit per second before correction.",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_innovation_gate_m",
    key="Longitudinal.LiveTune.DRelFilterInnovationGateM",
    cli_name="drel-filter-ig",
    label="drel_ig",
    default=30.0,
    minimum=5.0,
    maximum=60.0,
    description="Innovation gate: snap to raw when prediction error exceeds this (meters).",
  ),
  LeadResponseTuneSpec(
    attr="drel_filter_closing_gate_m",
    key="Longitudinal.LiveTune.DRelFilterClosingGateM",
    cli_name="drel-filter-cg",
    label="drel_cg",
    default=12.0,
    minimum=5.0,
    maximum=40.0,
    description="Closing gate: snap to raw when lead appears this much closer than predicted (meters).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_reacquire_pos_jerk_limit",
    key="Longitudinal.LiveTune.CruiseReacquirePosJerkLimit",
    cli_name="cruise-reacquire-pos-jerk-limit",
    label="cruise_reacquire_pos_jerk_limit",
    default=0.08,
    minimum=0.0,
    maximum=5.0,
    description="Max upward jerk (m/s^3) on planner output during cruise after a lead drops. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_reacquire_jerk_window_s",
    key="Longitudinal.LiveTune.CruiseReacquireJerkWindowS",
    cli_name="cruise-reacquire-jerk-window-s",
    label="cruise_reacquire_jerk_window_s",
    default=3.0,
    minimum=0.0,
    maximum=3.0,
    description="Duration (s) the cruise_reacquire_pos_jerk_limit is enforced after a lead drops. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_reacquire_jerk_ramp_mps3_per_s",
    key="Longitudinal.LiveTune.CruiseReacquireJerkRamp",
    cli_name="cruise-reacquire-jerk-ramp",
    label="cruise_reacquire_jerk_ramp",
    default=0.8,
    minimum=0.0,
    maximum=5.0,
    description="Growth rate (m/s^3 per s) of the reacquire jerk allowance after a lead drops; first frames stay at CruiseReacquirePosJerkLimit. 0 = fixed limit for the whole window.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_collapse_holdback_s",
    key="Longitudinal.LiveTune.CruiseCollapseHoldbackS",
    cli_name="cruise-collapse-holdback-s",
    label="cruise_collapse_holdback_s",
    default=2.0,
    minimum=0.0,
    maximum=5.0,
    description="Seconds after a prob-COLLAPSE lead->cruise exit to pin the reacquire jerk at CruiseReacquirePosJerkLimit (no ramp escalation). A genuine departure keeps the full ramp. 0 = pre-CD5 (ramp escalates regardless of exit cause).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_exit_lookback_frames",
    key="Longitudinal.LiveTune.CruiseExitLookbackFrames",
    cli_name="cruise-exit-lookback-frames",
    label="cruise_exit_lookback_frames",
    default=7.0,
    minimum=1.0,
    maximum=20.0,
    description="Frames of departing lead (status,modelProb,dRel) published history retained at a lead->cruise exit (used to confirm the prob ended below the Schmitt exit band).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_exit_abrupt_prob_drop",
    key="Longitudinal.LiveTune.CruiseExitAbruptProbDrop",
    cli_name="cruise-exit-abrupt-prob-drop",
    label="cruise_exit_abrupt_prob_drop",
    default=0.3,
    minimum=0.05,
    maximum=1.0,
    description="Largest single-frame PUBLISHED modelProb drop (while lead-owned) at/above which a lead->cruise exit is classified a genuine DEPARTURE (abrupt track cliff) rather than a recoverable prob-COLLAPSE (gradual decay). Only 'collapse' pins the reacquire ramp at its floor. Raise to make collapse-classification (and the holdback) less eager.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_blend_s",
    key="Longitudinal.LiveTune.CruiseRelatchBlendS",
    cli_name="cruise-relatch-blend-s",
    label="cruise_relatch_blend_s",
    default=1.5,
    minimum=0.0,
    maximum=3.0,
    description="Duration (s) the relatch obstacle blend stays armed after a fresh, non-urgent, same-lead cruise->lead relatch. Spans the obstacle-cost settle: the downward pull is slew-blended in AND the brake-release blip at settle-out is smoothed. 0 = pre-CD5 hard obstacle swap.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_blend_jerk_mps3",
    key="Longitudinal.LiveTune.CruiseRelatchBlendJerkMps3",
    cli_name="cruise-relatch-blend-jerk",
    label="cruise_relatch_blend_jerk",
    default=2.0,
    minimum=0.0,
    maximum=10.0,
    description="Negative-leg (brake-onset) jerk cap (m/s^3) during the relatch blend window; spreads the fresh-obstacle brake step. Urgency-bypassed. 0 = no downward slew (blend disabled).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_release_jerk_mps3",
    key="Longitudinal.LiveTune.CruiseRelatchReleaseJerkMps3",
    cli_name="cruise-relatch-release-jerk",
    label="cruise_relatch_release_jerk",
    default=2.0,
    minimum=0.0,
    maximum=10.0,
    description="Positive-leg (brake-RELEASE) jerk cap (m/s^3) during the relatch blend window; smooths the abrupt release blip as the obstacle cost settles. Always-safe (only ever keeps MORE brake, never delays brake onset), so it is NOT urgency-bypassed. 0 = release leg untouched (pre-CD5).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_urgent_ttc_s",
    key="Longitudinal.LiveTune.CruiseRelatchUrgentTtcS",
    cli_name="cruise-relatch-urgent-ttc-s",
    label="cruise_relatch_urgent_ttc_s",
    default=4.0,
    minimum=0.0,
    maximum=15.0,
    description="Relatch TTC (s) at/below which the blend AND large-TTC decel cap are bypassed (full braking passes immediately). Shares the emergency bypass.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_urgent_closing_mps",
    key="Longitudinal.LiveTune.CruiseRelatchUrgentClosingMps",
    cli_name="cruise-relatch-urgent-closing-mps",
    label="cruise_relatch_urgent_closing_mps",
    default=2.5,
    minimum=0.0,
    maximum=20.0,
    description="Relatch closing speed (m/s) at/above which the blend AND large-TTC decel cap are bypassed.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_bypass_decel_mps2",
    key="Longitudinal.LiveTune.CruiseRelatchBypassDecelMps2",
    cli_name="cruise-relatch-bypass-decel-mps2",
    label="cruise_relatch_bypass_decel_mps2",
    default=-1.5,
    minimum=-5.0,
    maximum=0.0,
    description="Requested-decel (m/s^2) at/below which the relatch blend is bypassed (mirrors the flutter bypass). Full braking passes the arming frame.",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_urgent_lead_decel_mps2",
    key="Longitudinal.LiveTune.CruiseRelatchUrgentLeadDecelMps2",
    cli_name="cruise-relatch-urgent-lead-decel-mps2",
    label="cruise_relatch_urgent_lead_decel_mps2",
    default=-1.0,
    minimum=-5.0,
    maximum=0.0,
    description="Relatched lead aLeadK (m/s^2) at/below which the blend AND large-TTC decel cap are bypassed (anticipatory braking toward a decelerating lead; TTC/closing/FCW lag a lead that just began braking at long range).",
  ),
  LeadResponseTuneSpec(
    attr="cruise_relatch_max_decel_mps2",
    key="Longitudinal.LiveTune.CruiseRelatchMaxDecelMps2",
    cli_name="cruise-relatch-max-decel-mps2",
    label="cruise_relatch_max_decel_mps2",
    default=-0.8,
    minimum=-5.0,
    maximum=0.0,
    description="Cap on relatch peak decel (m/s^2) while the blend window is active on a non-urgent, large-TTC relatch. Removed by the urgency bypass. 0 = no cap.",
  ),
  LeadResponseTuneSpec(
    attr="handoff_limit_window_s",
    key="Longitudinal.LiveTune.HandoffLimitWindowS",
    cli_name="handoff-limit-window-s",
    label="handoff_limit_window_s",
    default=0.40,
    minimum=0.0,
    maximum=1.0,
    description="CD6 (road 200-6): duration (s) a SYMMETRIC per-frame delta clamp on the planner output is armed after any "
                "cruise<->lead source transition, so a vLeadK-rollover handoff cannot sign-flip aTarget in a single frame. "
                "The upward (accel-increasing) leg always applies (limiting acceleration is always safe); the downward "
                "(braking) leg is bypassed under the shared relatch urgency signal (fast-close / short-TTC / FCW / "
                "requested hard decel) so emergency braking is never delayed. Rollback sentinel: 0 disables the windowed "
                "limiter entirely (pre-CD6 hard handoff).",
  ),
  LeadResponseTuneSpec(
    attr="handoff_limit_max_delta_mps2",
    key="Longitudinal.LiveTune.HandoffLimitMaxDeltaMps2",
    cli_name="handoff-limit-max-delta-mps2",
    label="handoff_limit_max_delta_mps2",
    default=0.30,
    minimum=0.0,
    maximum=2.0,
    description="CD6: maximum |output_a_target - prev_a| (m/s^2) allowed per frame while the handoff limiter window is "
                "active (comfortably under the oracle's 0.4 one-frame bound). Symmetric bound; the downward leg is "
                "urgency-bypassed. Only meaningful when HandoffLimitWindowS > 0.",
  ),
  LeadResponseTuneSpec(
    attr="handoff_inside_df_positive_cap_mps2",
    key="Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2",
    cli_name="handoff-inside-df-positive-cap-mps2",
    label="handoff_inside_df_positive_cap_mps2",
    default=0.10,
    minimum=0.0,
    maximum=10.0,
    description="CD6 EDGE1 (road 200-13 phase-1): always-on cap on positive output_a_target (m/s^2) while the source is "
                "cruise AND a valid control lead is inside desired_follow_distance(v_ego, v_lead, t_follow) on a closing "
                "(ego-faster / negative-vRel) trend, so the planner stops spending headway accelerating into a sub-target "
                "lead before lead0 latches. Not windowed. Rollback sentinel: a large value (e.g. 10 = the spec maximum) "
                "disables the cap.",
  ),
  LeadResponseTuneSpec(
    attr="comfort_jerk_limit_mps3",
    key="Longitudinal.LiveTune.ComfortJerkLimitMps3",
    cli_name="comfort-jerk-limit-mps3",
    label="comfort_jerk_limit_mps3",
    default=0.8,
    minimum=0.0,
    maximum=50.0,
    description="CD7 (road 200-10 / 200-9 tap1 / 201-9): graded-onset comfort anti-jerk envelope on the planner's FINAL "
                "output_a_target. Bounds the per-frame DOWNWARD (comfort-braking-onset) delta output_a_target to "
                "comfort_jerk_limit_mps3 * dt (0.8 m/s^3 * 0.05 s = 0.04 m/s^2/frame) so a single noisy vRel frame under "
                "a benign steady LEAD follow cannot step-change aTarget hard into a brake (road: -0.31 -> -1.00 in "
                "0.15 s, ~4.6 m/s^3) - the felt unnecessary-braking blip. ASYMMETRIC (down-leg only): the UPWARD leg "
                "(brake-RELEASE and re-accel toward a followed lead) is always-safe and left FREE so managed release/"
                "re-accel moves are never blunted (matches the CD5 relatch blend / flutter clamp precedent). SCOPE-GATED "
                "to the steady-lead-follow regime: engages only when a lead0/lead1 source owns control with NO source "
                "flip, CD6 handoff limiter/EDGE1 cap, CD5 relatch blend, or flutter clamp active this frame. FULLY "
                "BYPASSED whenever ANY hazard/urgency signal is active - the SAME _relatch_urgency_bypass signal CD5/CD6 "
                "use (FCW / short-TTC / fast-close on the owned lead, or a requested hard decel) - so real braking is "
                "NEVER rate-limited. Runs AFTER the CD6 handoff limiter as the truly-final composed limiter. Only grades "
                "the ONSET of a downward step (no sustained floor is lowered once prev_a catches up). Rollback sentinel: "
                "a large value (e.g. 50 = the spec maximum) makes the per-frame bound unreachable and disables the "
                "envelope (pre-CD7 un-enveloped output); 0 also disables it.",
  ),
  LeadResponseTuneSpec(
    attr="comfort_jerk_bypass_decel_mps2",
    key="Longitudinal.LiveTune.ComfortJerkBypassDecelMps2",
    cli_name="comfort-jerk-bypass-decel-mps2",
    label="comfort_jerk_bypass_decel_mps2",
    default=-1.5,
    minimum=-5.0,
    maximum=0.0,
    description="CD7: requested-decel floor (m/s^2) at/below which the comfort jerk envelope is bypassed regardless of the "
                "lead-object urgency tests (mirrors the flutter/relatch bypass floor). A raw output_a_target at/below this "
                "passes unmodified this frame so a hard MPC brake is never throttled even if no lead object is present to "
                "evaluate. Rollback sentinel: 0 disables this floor (lead-object urgency signal only).",
  ),
  LeadResponseTuneSpec(
    attr="lead_prob_enter",
    key="Longitudinal.LiveTune.LeadProbEnter",
    cli_name="lead-prob-enter",
    label="lead_prob_enter",
    default=0.6,
    minimum=0.0,
    maximum=1.0,
    description="vision lead prob required to latch a slot on (Schmitt trigger). Raise to reject flicker.",
  ),
  LeadResponseTuneSpec(
    attr="lead_prob_exit",
    key="Longitudinal.LiveTune.LeadProbExit",
    cli_name="lead-prob-exit",
    label="lead_prob_exit",
    default=0.25,
    minimum=0.0,
    maximum=1.0,
    description="vision lead prob below which a latched slot releases. Lower than enter = hysteresis band.",
  ),
  LeadResponseTuneSpec(
    attr="lead_source_acquire_frames",
    key="Longitudinal.LiveTune.LeadSourceAcquireFrames",
    cli_name="lead-source-acquire-frames",
    label="lead_source_acquire_frames",
    default=1.0,
    minimum=1.0,
    maximum=20.0,
    description="Consecutive valid-lead frames required at the MPC before switching source FROM cruise TO lead. 1 = no dwell.",
  ),
  LeadResponseTuneSpec(
    attr="lead_source_release_frames",
    key="Longitudinal.LiveTune.LeadSourceReleaseFrames",
    cli_name="lead-source-release-frames",
    label="lead_source_release_frames",
    default=20.0,
    minimum=1.0,
    maximum=40.0,
    description="Consecutive invalid-lead frames required at the MPC before switching source FROM lead TO cruise. 1 = no dwell.",
  ),
  LeadResponseTuneSpec(
    attr="phantom_lead_hold_s",
    key="Longitudinal.LiveTune.PhantomLeadHoldS",
    cli_name="phantom-lead-hold-s",
    label="phantom_lead_hold_s",
    default=0.80,
    minimum=0.0,
    maximum=1.5,
    description="Duration (s) the last-known lead is extrapolated after status goes False. 0 disables (default off).",
  ),
  LeadResponseTuneSpec(
    attr="phantom_lead_stable_frames",
    key="Longitudinal.LiveTune.PhantomLeadStableFrames",
    cli_name="phantom-lead-stable-frames",
    label="phantom_lead_stable_frames",
    default=3.0,
    minimum=1.0,
    maximum=40.0,
    description="Consecutive stable frames required before a dropped lead is eligible for phantom hold.",
  ),
  LeadResponseTuneSpec(
    attr="phantom_lead_decel_hold_factor",
    key="Longitudinal.LiveTune.PhantomLeadDecelHoldFactor",
    cli_name="phantom-lead-decel-hold-factor",
    label="phantom_lead_decel_hold_factor",
    default=1.0,
    minimum=0.0,
    maximum=1.0,
    description="Fraction of the last measured lead decel (aLeadK<0) held through the phantom window. 1 = full hold; 0 = legacy linear decay to zero. Positive aLeadK always decays.",
  ),
  LeadResponseTuneSpec(
    attr="phantom_lead_decel_trend_gain",
    key="Longitudinal.LiveTune.PhantomLeadDecelTrendGain",
    cli_name="phantom-lead-decel-trend-gain",
    label="phantom_lead_decel_trend_gain",
    default=1.0,
    minimum=0.0,
    maximum=1.0,
    description="Fraction of the measured pre-drop d(aLeadK)/dt continued through the phantom window (deepening trends only; relaxing trends are never extrapolated). 0 = hold constant.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_detect_transitions",
    key="Longitudinal.LiveTune.FlutterDetectTransitions",
    cli_name="flutter-detect-transitions",
    label="flutter_detect_transitions",
    default=2.0,
    minimum=1.0,
    maximum=10.0,
    description="Source-transition count within FlutterDetectWindowS that triggers bidirectional jerk clamp.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_detect_window_s",
    key="Longitudinal.LiveTune.FlutterDetectWindowS",
    cli_name="flutter-detect-window-s",
    label="flutter_detect_window_s",
    default=1.0,
    minimum=0.1,
    maximum=5.0,
    description="Rolling window length (s) for flutter-detection transition count.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_clamp_jerk_mps3",
    key="Longitudinal.LiveTune.FlutterClampJerkMps3",
    cli_name="flutter-clamp-jerk-mps3",
    label="flutter_clamp_jerk_mps3",
    default=0.12,
    minimum=0.0,
    maximum=5.0,
    description="Bidirectional jerk cap (m/s^3) applied to planner output while flutter mode is active. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_clamp_bypass_decel_mps2",
    key="Longitudinal.LiveTune.FlutterClampBypassDecelMps2",
    cli_name="flutter-clamp-bypass-decel-mps2",
    label="flutter_clamp_bypass_decel_mps2",
    default=1.5,
    minimum=0.0,
    maximum=5.0,
    description="If modelAccel < -this, flutter clamp is bypassed so hard braking is not delayed.",
  ),
  LeadResponseTuneSpec(
    attr="flutter_clamp_brake_jerk_mps3",
    key="Longitudinal.LiveTune.FlutterClampBrakeJerkMps3",
    cli_name="flutter-clamp-brake-jerk-mps3",
    label="flutter_clamp_brake_jerk_mps3",
    default=1.5,
    minimum=0.0,
    maximum=5.0,
    description="Downward jerk cap (m/s^3) while flutter mode is active; never tighter than FlutterClampJerkMps3. 0 = symmetric legacy clamp.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_tau_s",
    key="Longitudinal.LiveTune.ModelLeadFilterTauS",
    cli_name="model-lead-filter-tau",
    label="model_lead_tau",
    default=2.80,
    minimum=0.20,
    maximum=8.0,
    description="Source-side model-only lead dRel filter time constant in radard. Higher = more source-noise rejection.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_open_slew_max_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterOpenSlewMaxMps",
    cli_name="model-lead-open-slew-max-mps",
    label="model_lead_open_slew",
    default=1.20,
    minimum=0.10,
    maximum=6.0,
    description="Max source-side opening dRel motion admitted per second before model velocity support.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_safe_ttc_s",
    key="Longitudinal.LiveTune.ModelLeadFilterSafeTtcS",
    cli_name="model-lead-safe-ttc",
    label="model_lead_safe_ttc",
    default=4.00,
    minimum=1.0,
    maximum=10.0,
    description="Low-TTC threshold that lets source-side model-lead filtering fast-adopt a closer measurement.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_assoc_drel_m",
    key="Longitudinal.LiveTune.ModelLeadFilterAssocDRelM",
    cli_name="model-lead-assoc-drel",
    label="model_lead_assoc_drel",
    default=12.0,
    minimum=3.0,
    maximum=35.0,
    description="Max dRel difference for associating model-only leads to a stable synthetic radard track.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_vrel_tau_s",
    key="Longitudinal.LiveTune.ModelLeadFilterVRelTauS",
    cli_name="model-lead-vrel-tau",
    label="model_lead_vrel_tau",
    default=0.40,
    minimum=0.10,
    maximum=2.0,
    description="Source-side model-only lead relative-velocity filter tau in radard. Lower = faster accel/decel recognition.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_fast_vrel_tau_s",
    key="Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS",
    cli_name="model-lead-fast-vrel-tau",
    label="model_lead_fast_vrel_tau",
    default=0.16,
    minimum=0.05,
    maximum=1.0,
    description="Relative-velocity filter tau used when model-lead gating admits a low-TTC or strongly closing event.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_blend_tau_floor_s",
    key="Longitudinal.LiveTune.ModelLeadFilterBlendTauFloorS",
    cli_name="model-lead-blend-tau-floor",
    label="model_lead_blend_tau_floor",
    default=0.30,
    minimum=0.05,
    maximum=8.0,
    description="dRel filter tau at full closing urgency (geometric blend from ModelLeadFilterTauS). "
                ">= ModelLeadFilterTauS forces urgency to 0 and disables the whole blend (exact legacy).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_blend_close_lo_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterBlendCloseLoMps",
    cli_name="model-lead-blend-close-lo",
    label="model_lead_blend_close_lo",
    default=1.0,
    minimum=0.0,
    maximum=2.4,
    description="Closing speed where dRel-filter urgency starts; hi endpoint is the fixed 2.5 m/s strong-closing gate, "
                "so the maximum stays below 2.5 to keep the blend span positive.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_blend_ttc_hi_s",
    key="Longitudinal.LiveTune.ModelLeadFilterBlendTtcHiS",
    cli_name="model-lead-blend-ttc-hi",
    label="model_lead_blend_ttc_hi",
    default=12.0,
    minimum=10.5,
    maximum=30.0,
    description="TTC where dRel-filter urgency starts; low endpoint is ModelLeadFilterSafeTtcS (max 10.0), "
                "so the minimum stays above it to keep the blend span positive.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_blend_slew_boost_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterBlendSlewBoostMps",
    cli_name="model-lead-blend-slew-boost",
    label="model_lead_blend_slew_boost",
    default=6.0,
    minimum=0.0,
    maximum=20.0,
    description="Extra closing dRel slew allowance (m/s) at full urgency so the close-slew clamp cannot bind. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_lag_comp_s",
    key="Longitudinal.LiveTune.ModelLeadFilterLagCompS",
    cli_name="model-lead-lag-comp",
    label="model_lead_lag_comp",
    default=0.6,
    minimum=0.0,
    maximum=2.0,
    description="Closing-only group-delay compensation (s) on the PUBLISHED model-lead dRel; capped at the active "
                "filter regime's delay. Publication only moves closer, never farther. 0 disables.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_lag_comp_deadzone_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterLagCompDeadzoneMps",
    cli_name="model-lead-lag-comp-deadzone",
    label="model_lead_lag_comp_deadzone",
    default=0.5,
    minimum=0.0,
    maximum=5.0,
    description="Closing speed ignored by the lag compensation; keeps steady-noise vRel jitter out of published dRel.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_lag_comp_fade_lo_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterLagCompFadeLoMps",
    cli_name="model-lead-lag-comp-fade-lo",
    label="model_lead_lag_comp_fade_lo",
    default=12.0,
    minimum=0.0,
    maximum=30.0,
    description="Ego speed (m/s) at/below which the lag compensation is fully faded out (stopping regime). "
                "FadeHi <= FadeLo disables the fade entirely (full compensation at all speeds).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_lag_comp_fade_hi_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterLagCompFadeHiMps",
    cli_name="model-lead-lag-comp-fade-hi",
    label="model_lead_lag_comp_fade_hi",
    default=18.0,
    minimum=0.0,
    maximum=40.0,
    description="Ego speed (m/s) at/above which the lag compensation is fully active. Linear ramp from FadeLo. "
                "Set both to 0 to disable the fade (full compensation everywhere, pre-fade behavior).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_fast_close_confirm_frames",
    key="Longitudinal.LiveTune.ModelLeadFilterFastCloseConfirmFrames",
    cli_name="model-lead-fast-close-confirm-frames",
    label="model_lead_fast_close_confirm",
    default=2.0,
    minimum=1.0,
    maximum=6.0,
    description="Consecutive qualifying frames (50 ms each) of beyond-gate inward dRel innovation required before the "
                "fast-close path adopts a much-closer measurement. 1 = legacy single-frame adoption; 2 filters isolated "
                "heavy-tail outliers while adding 50 ms to genuine cut-in fast adoption (urgency blend still reacts in frame 1).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_open_recovery_confirm_frames",
    key="Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryConfirmFrames",
    cli_name="model-lead-open-recovery-confirm-frames",
    label="model_lead_open_recovery_confirm",
    default=4.0,
    minimum=1.0,
    maximum=12.0,
    description="Consecutive frames of beyond-gate OPENING dRel innovation required before the corroborated low-speed "
                "recovery engages to heal a wrong-too-close track state. Higher = more conservative healing.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_open_recovery_tau_s",
    key="Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryTauS",
    cli_name="model-lead-open-recovery-tau",
    label="model_lead_open_recovery_tau",
    default=0.5,
    minimum=0.05,
    maximum=8.0,
    description="dRel filter tau used while the corroborated opening recovery is engaged (bypasses the opening slew "
                "cap up to this tau's step). >= ModelLeadFilterTauS effectively restores the legacy slew-only recovery.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_open_recovery_max_ego_mps",
    key="Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryMaxEgoMps",
    cli_name="model-lead-open-recovery-max-ego",
    label="model_lead_open_recovery_max_ego",
    default=8.0,
    minimum=0.0,
    maximum=40.0,
    description="Ego speed (m/s) at/below which the corroborated opening recovery may engage. 0 disables the recovery "
                "entirely (legacy opening behavior at all speeds).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_filter_open_recovery_innov_gate_m",
    key="Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryInnovGateM",
    cli_name="model-lead-open-recovery-innov-gate",
    label="model_lead_open_recovery_innov_gate",
    default=2.5,
    minimum=0.1,
    maximum=10.0,
    description="Opening dRel innovation (m) a frame must exceed to count toward the recovery confirmation. Default "
                "mirrors the 2.5 m close gate so only deeply-wrong states heal; lowering toward 1.0 also heals the "
                "~1-2 m pessimistic noise-rectification bias near stops (measured 0.1-0.4 m shorter true stop gaps).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_fcw_corrob_tol_m",
    key="Longitudinal.LiveTune.ModelLeadFcwCorrobTolM",
    cli_name="model-lead-fcw-corrob-tol",
    label="model_lead_fcw_corrob_tol",
    default=2.5,
    minimum=0.5,
    maximum=50.0,
    description="FCW-corroboration tolerance (m): a raw model dRel more than this ABOVE the filtered dRel counts as a "
                "disagreement vote (raw says the lead is farther, i.e. the filtered closeness is uncorroborated). "
                "Closing-lag disagreement is impossible by sign, so genuine threats always agree; base noise at crash "
                "range is sigma ~0.5 m, so 2.5 m = 5 sigma against false disagreement.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_fcw_corrob_min_agree",
    key="Longitudinal.LiveTune.ModelLeadFcwCorrobMinAgree",
    cli_name="model-lead-fcw-corrob-min-agree",
    label="model_lead_fcw_corrob_min_agree",
    default=2.0,
    minimum=0.0,
    maximum=8.0,
    description="Minimum agreeing frames within the FCW-corroboration window for a model lead to stay FCW-eligible "
                "(fcwSuppressed=False). 0 disables suppression entirely (legacy: any predicted crash with prob > 0.9 "
                "accrues crash_cnt regardless of raw-measurement corroboration).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_fcw_corrob_window",
    key="Longitudinal.LiveTune.ModelLeadFcwCorrobWindow",
    cli_name="model-lead-fcw-corrob-window",
    label="model_lead_fcw_corrob_window",
    default=3.0,
    minimum=1.0,
    maximum=8.0,
    description="FCW-corroboration vote window (frames, 50 ms each). Majority vote (MinAgree of Window) bridges "
                "isolated outward measurement outliers (~3% of close-range frames) so genuine FCW timing is untouched, "
                "while a phantom-collapsed track (raw persistently far above the filter) is suppressed within one frame.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_fcw_corrob_raw_closing_min_mps",
    key="Longitudinal.LiveTune.ModelLeadFcwCorrobRawClosingMinMps",
    cli_name="model-lead-fcw-corrob-raw-closing-min",
    label="model_lead_fcw_corrob_raw_closing_min",
    default=1.0,
    minimum=0.0,
    maximum=20.0,
    description="Raw-kinematic FCW-corroboration escape: minimum raw closing speed (m/s) for the raw model "
                "measurement to independently corroborate an imminent threat and hold FCW eligible even when the "
                "filtered dRel runs more pessimistic than raw (the deliberate closing-urgency blend, road 200-13 CD2). "
                "A phantom collapse measures raw NOT closing, so it fails this gate and stays suppressed. 0 disables "
                "the escape (exact legacy raw-vs-filter veto; rollback knob).",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_fcw_corrob_raw_ttc_max_s",
    key="Longitudinal.LiveTune.ModelLeadFcwCorrobRawTtcMaxS",
    cli_name="model-lead-fcw-corrob-raw-ttc-max",
    label="model_lead_fcw_corrob_raw_ttc_max",
    default=3.5,
    minimum=0.0,
    maximum=15.0,
    description="Raw-kinematic FCW-corroboration escape: maximum raw-side TTC (s, computed on the raw model dRel and "
                "raw closing speed) at/under which the raw measurement independently corroborates an imminent threat "
                "and holds FCW eligible regardless of the filtered-vs-raw delta (CD2). Aligns with the road-derived "
                "deep-close TTC band (200-13: TTC 1.9 s at the stomp). 0 disables the escape (rollback knob).",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_margin_mps2",
    key="Longitudinal.LiveTune.LeadAccelCorrMarginMps2",
    cli_name="lead-accel-corr-margin",
    label="lead_accel_corr_margin",
    default=0.5,
    minimum=0.0,
    maximum=10.0,
    description="Max uncorroborated lead decel (m/s^2) below the measured vLead trend passed to the MPC in "
                "non-dangerous, fresh-measurement states. >= 10 disables the bound entirely.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_meas_tau_s",
    key="Longitudinal.LiveTune.LeadAccelCorrMeasTauS",
    cli_name="lead-accel-corr-meas-tau",
    label="lead_accel_corr_meas_tau",
    default=0.3,
    minimum=0.1,
    maximum=2.0,
    description="Low-pass tau for the measured vLead trend used to corroborate aLeadK. 0.6 measurably delayed "
                "hard-brake onset in the design sweep - do not raise casually.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_ttc_guard_s",
    key="Longitudinal.LiveTune.LeadAccelCorrTtcGuardS",
    cli_name="lead-accel-corr-ttc-guard",
    label="lead_accel_corr_ttc_guard",
    default=8.0,
    minimum=2.0,
    maximum=20.0,
    description="Corroboration bound is bypassed (full aLeadK passes) at or below this TTC.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_closing_guard_mps",
    key="Longitudinal.LiveTune.LeadAccelCorrClosingGuardMps",
    cli_name="lead-accel-corr-closing-guard",
    label="lead_accel_corr_closing_guard",
    default=1.5,
    minimum=0.0,
    maximum=10.0,
    description="Corroboration bound is bypassed at or above this closing speed.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_near_headway_s",
    key="Longitudinal.LiveTune.LeadAccelCorrNearHeadwayS",
    cli_name="lead-accel-corr-near-headway",
    label="lead_accel_corr_near_headway",
    default=1.2,
    minimum=0.0,
    maximum=4.0,
    description="Corroboration bound is bypassed inside this headway (s) of gap.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_closing_rearm_mps",
    key="Longitudinal.LiveTune.LeadAccelCorrClosingRearmMps",
    cli_name="lead-accel-corr-closing-rearm",
    label="lead_accel_corr_closing_rearm",
    default=0.5,
    minimum=0.0,
    maximum=5.0,
    description="Dangerous-state bypass hysteresis: closing speed must drop this far below ClosingGuardMps before "
                "the bypass can disengage.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_ttc_rearm_s",
    key="Longitudinal.LiveTune.LeadAccelCorrTtcRearmS",
    cli_name="lead-accel-corr-ttc-rearm",
    label="lead_accel_corr_ttc_rearm",
    default=2.0,
    minimum=0.0,
    maximum=10.0,
    description="Dangerous-state bypass hysteresis: TTC must rise this far above TtcGuardS before the bypass can "
                "disengage.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_headway_rearm_m",
    key="Longitudinal.LiveTune.LeadAccelCorrHeadwayRearmM",
    cli_name="lead-accel-corr-headway-rearm",
    label="lead_accel_corr_headway_rearm",
    default=2.0,
    minimum=0.0,
    maximum=10.0,
    description="Dangerous-state bypass hysteresis: gap must exceed the near-headway gate by this many meters "
                "before the bypass can disengage.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_settle_tau_mult",
    key="Longitudinal.LiveTune.LeadAccelCorrSettleTauMult",
    cli_name="lead-accel-corr-settle-tau-mult",
    label="lead_accel_corr_settle_tau_mult",
    default=2.0,
    minimum=0.5,
    maximum=5.0,
    description="Multiple of MeasTauS of same-track vLead history required before the corroboration bound can "
                "clamp aLeadK.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_max_dt_s",
    key="Longitudinal.LiveTune.LeadAccelCorrMaxDtS",
    cli_name="lead-accel-corr-max-dt",
    label="lead_accel_corr_max_dt",
    default=0.5,
    minimum=0.05,
    maximum=2.0,
    description="Max frame-to-frame dt admitted as a same-track vLead measurement; a larger gap resets the "
                "corroboration low-pass (treated as a track identity change).",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_amplify_gain",
    key="Longitudinal.LiveTune.LeadAccelCorrAmplifyGain",
    cli_name="lead-accel-corr-amplify-gain",
    label="lead_accel_corr_amplify_gain",
    default=1.0,
    minimum=0.0,
    maximum=1.0,
    description="CD3 lead-decel truth deficit: fraction of the way to pull the model's underreported aLeadK toward "
                "the measured vLead trend (corr_a_meas_lp) per frame, when BOTH the model and the trend agree the lead "
                "is braking. 0 disables (rollback to the downward bound only). Only ever DEEPENS an already-negative "
                "model aLeadK; never fabricates decel from a coasting report.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_amplify_deadband_mps2",
    key="Longitudinal.LiveTune.LeadAccelCorrAmplifyDeadbandMps2",
    cli_name="lead-accel-corr-amplify-deadband",
    label="lead_accel_corr_amplify_deadband",
    default=0.35,
    minimum=0.0,
    maximum=3.0,
    description="The vLead trend must be this many m/s^2 MORE negative than the model aLeadK before amplify engages. "
                "Rejects the finite-difference jitter of a steady/lightly-braking lead (ev6_measured vRel noise + prob "
                "dropouts) so amplify cannot chatter aLeadK on a non-threat.",
  ),
  LeadResponseTuneSpec(
    attr="lead_accel_corr_amplify_cap_mps2",
    key="Longitudinal.LiveTune.LeadAccelCorrAmplifyCapMps2",
    cli_name="lead-accel-corr-amplify-cap",
    label="lead_accel_corr_amplify_cap",
    default=2.0,
    minimum=0.0,
    maximum=10.0,
    description="Max m/s^2 that amplify may deepen aLeadK below the model report in a single frame; bounds the effect "
                "of one noisy vLead-trend sample.",
  ),
  LeadResponseTuneSpec(
    attr="lead_stabilizer_trend_tau_s",
    key="Longitudinal.LiveTune.LeadStabilizerTrendTauS",
    cli_name="lead-stabilizer-trend-tau",
    label="lead_stabilizer_trend_tau",
    default=0.20,
    minimum=0.05,
    maximum=1.0,
    description="EMA time constant for the measured d(aLeadK)/dt used by the phantom trend hold. Lower = faster "
                "trend response, more measurement noise passed through.",
  ),
  LeadResponseTuneSpec(
    attr="lead_stabilizer_trend_drel_jump_m",
    key="Longitudinal.LiveTune.LeadStabilizerTrendDRelJumpM",
    cli_name="lead-stabilizer-trend-drel-jump",
    label="lead_stabilizer_trend_drel_jump",
    default=3.0,
    minimum=1.0,
    maximum=10.0,
    description="Identity gate for the trend measurement: a dRel step this far off the propagated position between "
                "consecutive valid frames is treated as a track swap, not a measurement.",
  ),
  LeadResponseTuneSpec(
    attr="lead_stabilizer_trend_yrel_jump_m",
    key="Longitudinal.LiveTune.LeadStabilizerTrendYRelJumpM",
    cli_name="lead-stabilizer-trend-yrel-jump",
    label="lead_stabilizer_trend_yrel_jump",
    default=1.5,
    minimum=0.3,
    maximum=5.0,
    description="Identity gate for the trend measurement: a lateral (yRel) jump this large between consecutive "
                "valid frames is treated as a track swap, not a measurement.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_blend_min_span",
    key="Longitudinal.LiveTune.ModelLeadBlendMinSpan",
    cli_name="model-lead-blend-min-span",
    label="model_lead_blend_min_span",
    default=1e-2,
    minimum=1e-3,
    maximum=1.0,
    description="Degeneracy guard (radard): a closing-urgency blend span (closing-speed, TTC, or lag-comp fade) "
                "narrower than this collapses to disabled (u=0) instead of risking a sign flip.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_blend_ttc_min_closing_mps",
    key="Longitudinal.LiveTune.ModelLeadBlendTtcMinClosingMps",
    cli_name="model-lead-blend-ttc-min-closing",
    label="model_lead_blend_ttc_min_closing",
    default=0.3,
    minimum=0.0,
    maximum=3.0,
    description="Minimum closing speed (radard) before the TTC-based closing-urgency term is evaluated at all; "
                "guards TTC=dRel/closing against blowing up near zero closing speed.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_assoc_dpath_gate_m",
    key="Longitudinal.LiveTune.ModelLeadAssocDPathGateM",
    cli_name="model-lead-assoc-dpath-gate",
    label="model_lead_assoc_dpath_gate",
    default=1.8,
    minimum=0.5,
    maximum=6.0,
    description="CD4 (radard): PRIMARY path-relative lateral-continuity gate for ModelLeadTracker association. A model "
                "lead whose path-relative dPath differs from a track's filtered dPath by more than this is treated as a "
                "different physical lead (spawns its own track). Replaces the shared 3.0 m gate that keyed on RAW yRel, "
                "which a curve-induced yRel drift (+1.93 -> -8.3 m, dPath still < 0.9 m) repeatedly blew, churning the "
                "published track id. Genuine adjacent/cut-in leads at a truly different offset still exceed this and stay "
                "separate. Rollback sentinel: set to 3.0 alongside ModelLeadAssocYRawTolM=3.0 to restore the legacy "
                "shared 3.0 m gate exactly.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_assoc_y_raw_tol_m",
    key="Longitudinal.LiveTune.ModelLeadAssocYRawTolM",
    cli_name="model-lead-assoc-y-raw-tol",
    label="model_lead_assoc_y_raw_tol",
    default=7.0,
    minimum=3.0,
    maximum=12.0,
    description="CD4 (radard): separate, LARGER tolerance for the RAW yRel term in ModelLeadTracker association, "
                "replacing the 3.0 m hard reject that a curve-induced raw-yRel excursion tripped on the same physical "
                "lead. Continuity is now gated PRIMARILY on dPath (ModelLeadAssocDPathGateM); raw y_err only rejects when "
                "it exceeds this wider tolerance. Safety: when the candidate is CLOSING (raw vRel < 0) the raw-y "
                "tolerance is held at the legacy 3.0 m so a slow-closing near lead in a momentary low-dPath band cannot "
                "be masked into a farther track (the widening applies only to the opening/lane-relevant case). Rollback "
                "sentinel: set to 3.0 (alongside ModelLeadAssocDPathGateM=3.0) to restore the legacy shared 3.0 m gate.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_step_guard_abs_m",
    key="Longitudinal.LiveTune.ModelLeadStepGuardAbsM",
    cli_name="model-lead-step-guard-abs",
    label="model_lead_step_guard_abs",
    default=3.0,
    minimum=0.0,
    maximum=20.0,
    description="CD4 (radard): absolute floor (m) of the OPENING-ONLY published-dRel single-frame step bound. While a "
                "track persists with a prior published value, a published dRel jump FARTHER than "
                "max(this, ModelLeadStepGuardFrac*dRel) in one frame is clamped to that bound (defense-in-depth against "
                "any residual fabricated opening step). SAFETY: the guard is one-sided by construction -- it clamps ONLY "
                "the opening (farther) direction; a closing (nearer) reading is NEVER clamped, so it can never delay or "
                "attenuate emergency braking and is exempt from the rate-limit safety rule. Never binds on a track's "
                "first publish. Rollback sentinel: 0.0 (or ModelLeadStepGuardFrac=0.0) disables the guard entirely.",
  ),
  LeadResponseTuneSpec(
    attr="model_lead_step_guard_frac",
    key="Longitudinal.LiveTune.ModelLeadStepGuardFrac",
    cli_name="model-lead-step-guard-frac",
    label="model_lead_step_guard_frac",
    default=0.10,
    minimum=0.0,
    maximum=1.0,
    description="CD4 (radard): dRel-proportional term of the OPENING-ONLY published-dRel single-frame step bound "
                "max(ModelLeadStepGuardAbsM, this*dRel). See ModelLeadStepGuardAbsM: opening-only, never clamps a closing "
                "reading, exempt from the emergency-braking rate-limit rule. Rollback sentinel: 0.0 (or "
                "ModelLeadStepGuardAbsM=0.0) disables the guard entirely.",
  ),
  LeadResponseTuneSpec(
    attr="lead_vlead_optimism_clamp_range_m",
    key="Longitudinal.LiveTune.LeadVLeadOptimismClampRangeM",
    cli_name="lead-vlead-optimism-clamp-range",
    label="vlead_optimism_clamp_range",
    default=55.0,
    minimum=20.0,
    maximum=1e9,
    description="CD8 (radard, road 200-13 EDGE2): far-range (m) beyond which the published-vLead optimism clamp is armed. "
                "While a far, newly-acquired lead is still stopping/slowing, the model's published vLead runs biased HIGH "
                "(road ~+4 m/s vs position-derived truth), so the kinematic stopping-need handoff term underestimates the "
                "required decel and braking starts late, forcing a concentrated hard stop. Beyond this range AND when the "
                "RAW model vLead is declining monotonically for LeadVLeadOptimismClampConfirmFrames+1 frames (a stopping "
                "lead), the published vLead is pulled toward the position-derived value d(dRel)/dt + v_ego (min only - it "
                "only ever makes the lead SLOWER / more urgent, never faster). Publish-time only; internal filter state "
                "untouched. Rollback sentinel: 1e9 (the spec maximum deliberately admits it) makes the range unreachable "
                "and disables the clamp exactly (or set LeadVLeadOptimismClampGain=0).",
  ),
  LeadResponseTuneSpec(
    attr="lead_vlead_optimism_clamp_gain",
    key="Longitudinal.LiveTune.LeadVLeadOptimismClampGain",
    cli_name="lead-vlead-optimism-clamp-gain",
    label="vlead_optimism_clamp_gain",
    default=1.0,
    minimum=0.0,
    maximum=1.0,
    description="CD8 (radard): fraction of the way the published vLead is pulled from the optimistic model value toward "
                "the position-derived d(dRel)/dt + v_ego estimate when the far-range optimism clamp fires. 1.0 = publish "
                "the full position-derived (truth) velocity; the result is still min()'d against the model vLead so the "
                "clamp can only ever LOWER the published vLead (bias toward earlier braking), never raise it. Rollback "
                "sentinel: 0.0 disables the clamp entirely (exact legacy publish, alongside the range sentinel).",
  ),
  LeadResponseTuneSpec(
    attr="lead_vlead_optimism_clamp_confirm_frames",
    key="Longitudinal.LiveTune.LeadVLeadOptimismClampConfirmFrames",
    cli_name="lead-vlead-optimism-clamp-confirm-frames",
    label="vlead_optimism_clamp_confirm_frames",
    default=3.0,
    minimum=1.0,
    maximum=6.0,
    description="CD8 (radard): consecutive RAW-model-vLead decline frames required (this many decreasing steps, i.e. "
                "this+1 samples) before the far-range optimism clamp arms, so a single noisy frame cannot trigger it - a "
                "genuinely stopping/decelerating lead declines frame after frame, isolated noise does not. A flat or "
                "rising raw vLead (a steady/moving/pulling-away lead) never qualifies, so the clamp is inert on normal "
                "following. Raise to demand a longer sustained decline; 1 = a single decline step arms it.",
  ),
  LeadResponseTuneSpec(
    attr="lead_vlead_optimism_clamp_slow_lead_frac",
    key="Longitudinal.LiveTune.LeadVLeadOptimismClampSlowLeadFrac",
    cli_name="lead-vlead-optimism-clamp-slow-lead-frac",
    label="vlead_optimism_clamp_slow_lead_frac",
    default=0.5,
    minimum=0.0,
    maximum=1.0,
    description="CD8 (radard): the far-range vLead optimism clamp fires ONLY when the position-derived lead velocity is a "
                "genuinely slow/stopping lead - at/below this fraction of ego speed - so it targets the stopped-traffic "
                "approach and NOT a fast steady lead that suffered a transient perception vLead dip (a vLeadK rollover, "
                "CD6 non-regression: a 34 m/s lead momentarily reading 31 m/s has ratio ~0.97 and is excluded, while a "
                "stopping lead reads ~0). Raise toward 1.0 to admit any lead slower than ego (least restrictive); lower to "
                "demand a more nearly-stopped lead. Only meaningful when the clamp is otherwise armed.",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_margin_mps",
    key="Longitudinal.LiveTune.ClosingGovernorMarginMps",
    cli_name="closing-governor-margin",
    label="closing_governor_margin",
    default=0.75,
    minimum=0.05,
    maximum=100.0,
    description="CD9 (radard, road 205-6 Event B): corroborated-closing governor MASTER arm margin (m/s). The governor "
                "latches when the windowed position-derived closure of the RAW model dRel stream exceeds the currently "
                "published closing speed by at least this margin AND the windowed RAW vRel agrees the lead is closing "
                "(> ClosingGovernorMinClosingMps). While latched it forces the closing-side fast taus (dRel blend floor, "
                "fast vRel tau, fast aLeadK tau) and one-directionally clamps the published vLead toward the LEAST "
                "aggressive corroborated closure - the road defect was ~1.2 s of publish-side EMA lag against a braking "
                "lead while the tracker's own raw stream showed the truth. Steady-follow noise cannot latch it: both "
                "windowed means must agree beyond the margin. Rollback sentinel: >= 99 disables the governor entirely "
                "(exact legacy publish).",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_min_closing_mps",
    key="Longitudinal.LiveTune.ClosingGovernorMinClosingMps",
    cli_name="closing-governor-min-closing",
    label="closing_governor_min_closing",
    default=0.30,
    minimum=0.05,
    maximum=5.0,
    description="CD9 (radard): minimum windowed RAW-vRel closing speed (m/s) required to corroborate EITHER governor arm "
                "path (position-excess or sustained-lead-decel). Below this the lead is not provably closing and the "
                "governor stays inert regardless of position slope - this is what keeps opening/steady follows and pure "
                "position-noise runs from latching. Raise to demand a harder closure before the fast path engages.",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_window_s",
    key="Longitudinal.LiveTune.ClosingGovernorWindowS",
    cli_name="closing-governor-window",
    label="closing_governor_window",
    default=0.60,
    minimum=0.20,
    maximum=2.0,
    description="CD9 (radard): evidence window (s) for the corroborated-closing governor's raw-stream means (endpoint-mean "
                "position slope, mean raw vRel, mean raw lead accel). Longer = more noise immunity but later latch and a "
                "larger inherent estimate lag (~window/2 x closure accel). Harness sweep on the road-205-6 repro: 0.6 "
                "holds min THW 1.04 with a -0.26 worst steady-noise excursion across seeds; 0.5 buys THW 1.14 but puts "
                "the steady heavy-tail excursion at -0.53, uncomfortably near the -0.6 phantom-brake line. Single "
                "heavy-tail raw dRel outliers (road: +-1.5-3 m frames) cannot dominate the k-endpoint means.",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_accel_onset_mps2",
    key="Longitudinal.LiveTune.ClosingGovernorAccelOnsetMps2",
    cli_name="closing-governor-accel-onset",
    label="closing_governor_accel_onset",
    default=0.35,
    minimum=0.05,
    maximum=100.0,
    description="CD9 (radard): sustained-lead-decel arm path - the governor also latches when the windowed mean RAW model "
                "lead accel is below -this (m/s^2) while the windowed raw vRel corroborates closing. On the road event the "
                "raw aLead mean separated cleanly (-0.55 sustained during the brake vs -0.10 steady phase) a full ~0.5 s "
                "before the position slope confirmed - this is the earliest reliable signal. Rollback sentinel: >= 99 "
                "disables this arm path only (position-excess path unaffected).",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_pos_trust_excess_mps",
    key="Longitudinal.LiveTune.ClosingGovernorPosTrustExcessMps",
    cli_name="closing-governor-pos-trust-excess",
    label="closing_governor_pos_trust",
    default=1.5,
    minimum=0.0,
    maximum=100.0,
    description="CD9 (radard): how far (m/s) the position-derived closure may LEAD the windowed raw-vRel closing evidence "
                "in the governor's publish clamp: clamp closure = min(position closure, vRel closure + this). The road "
                "event's raw v-stream itself ran ~+1.5-2.3 m/s optimistic against the model's own position stream, so a "
                "pure min(pos, vRel) cap republished most of the lie; the position slope was the truth-teller. This bounds "
                "how much the clamp trusts position beyond what velocity corroborates - a pure position phantom with "
                "minimal vRel agreement is still capped near the (gated) vRel evidence. 0.0 = strict min(pos, vRel) "
                "(most conservative).",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_hold_s",
    key="Longitudinal.LiveTune.ClosingGovernorHoldS",
    cli_name="closing-governor-hold",
    label="closing_governor_hold",
    default=1.00,
    minimum=0.10,
    maximum=5.0,
    description="CD9 (radard): latch hold (s) after the last frame the governor's arm conditions were satisfied. Keeps the "
                "fast-tau regime engaged across single non-qualifying frames mid-closure so the response does not chatter "
                "between fast and slow filtering while the threat is still developing.",
  ),
  LeadResponseTuneSpec(
    attr="closing_governor_alead_tau_s",
    key="Longitudinal.LiveTune.ClosingGovernorALeadTauS",
    cli_name="closing-governor-alead-tau",
    label="closing_governor_alead_tau",
    default=0.18,
    minimum=0.05,
    maximum=2.0,
    description="CD9 (radard): aLeadK EMA tau (s) while the corroborated-closing governor is latched, replacing the fixed "
                "0.60 s MODEL_LEAD_ACCEL_TAU_S that halved the published lead decel through the whole road event (pub -0.3 "
                "vs raw -0.7 at onset; -1.3 vs -2.1 late). Only the time constant changes - the published value is still "
                "an EMA of the model's own accel measurement, never fabricated.",
  ),
  LeadResponseTuneSpec(
    attr="launch_release_min_drel_m",
    key="Longitudinal.LiveTune.LaunchReleaseMinDrelM",
    cli_name="launch-release-min-drel",
    label="launch_release_min_drel",
    default=5.0,
    minimum=2.0,
    maximum=30.0,
    description="Event A (road 205-13): ABSOLUTE published-dRel arming gate (m) for releasing the stop latch on a lead "
                "launch - the legacy hardcoded 5.0. The release arms once published dRel exceeds EITHER this absolute "
                "range OR stop-settle dRel + LaunchReleaseDepartGateM (departure-relative), whichever is smaller; a hard "
                "floor of 2.0 m always applies. On the road the stop settled at a published 4.15 m, so this absolute gate "
                "alone forced the lead to open 0.85 m of published gap (~1.0 s of held -2.0 full brake) before release "
                "could even arm.",
  ),
  LeadResponseTuneSpec(
    attr="launch_release_depart_gate_m",
    key="Longitudinal.LiveTune.LaunchReleaseDepartGateM",
    cli_name="launch-release-depart-gate",
    label="launch_release_depart_gate",
    default=0.20,
    minimum=0.0,
    maximum=100.0,
    description="Event A (road 205-13): DEPARTURE-RELATIVE arming gate (m) - the published dRel rise above the minimum "
                "published dRel seen during THIS stop that proves the lead is genuinely departing. Arms the stop-latch "
                "release as min(absolute gate, stop-min + this), so a stop that settles at 4.1 m does not need the lead "
                "to reach an arbitrary 5.0 m before the car can begin releasing the brake. Works with the existing "
                "pullaway-speed and hold-frame conditions unchanged. Rollback sentinel: >= 99 restores the pure absolute "
                "gate (exact legacy arming).",
  ),
  LeadResponseTuneSpec(
    attr="launch_follow_accel_floor_max_mps2",
    key="Longitudinal.LiveTune.LaunchFollowAccelFloorMaxMps2",
    cli_name="launch-follow-accel-floor-max",
    label="launch_follow_accel_floor_max",
    default=1.8,
    minimum=0.0,
    maximum=2.5,
    description="Event A (road 205-13): low-speed launch-follow accel FLOOR ceiling (m/s^2). The planner floors its output "
                "accel at get_low_speed_launch_follow_factor(...) x this while a lead is departing at low ego speed, so "
                "the launch DEMAND actually rises toward the departing lead instead of relying on the MPC's jerk-shaped "
                "ramp from standstill (road: demand peaked +1.02 while the lead departed at +5 m/s and the driver "
                "pedaled). The existing factor already scales by ego speed, lead speed, pullaway and gap surplus, and the "
                "existing launch clip raise still bounds the ceiling; the floor never applies while shouldStop or while "
                "the lead is closing. Rollback sentinel: 0.0 disables the floor entirely (exact legacy demand).",
  ),
)

LEAD_RESPONSE_TUNE_SPECS_BY_ATTR = {spec.attr: spec for spec in LEAD_RESPONSE_TUNE_SPECS}


@dataclass(frozen=True)
class LeadResponseTuningConfig:
  lead_preview_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_strength"].default
  lead_preview_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_gap_min_m"].default
  lead_preview_max_buffer_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_max_buffer_m"].default
  lead_preview_min_speed_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_min_speed_mps"].default
  lead_acquire_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_acquire_window_s"].default
  gap_reclaim_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_strength"].default
  gap_reclaim_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_gap_min_m"].default
  gap_reclaim_max_accel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_max_accel"].default
  gap_reclaim_follow_max_accel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_follow_max_accel"].default
  gap_reclaim_taper_gain: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_taper_gain"].default
  lead_keepup_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_keepup_strength"].default
  lead_keepup_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_keepup_gap_min_m"].default
  lead_keepup_max_accel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_keepup_max_accel"].default
  lead_slowdown_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_slowdown_strength"].default
  lead_slowdown_max_decel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_slowdown_max_decel"].default
  lead_slowdown_kinematic_headroom: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_slowdown_kinematic_headroom"].default
  lead_slowdown_kinematic_margin_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_slowdown_kinematic_margin_m"].default
  lead_slowdown_kinematic_oncoming_vlead_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_slowdown_kinematic_oncoming_vlead_mps"].default
  lead_handoff_stopping_need_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_handoff_stopping_need_decel_mps2"].default
  lead_handoff_stopping_need_ref_speed_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_handoff_stopping_need_ref_speed_mps"].default
  lead_brake_release_min_speed_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_min_speed_mps"].default
  lead_brake_release_brake_deficit_margin_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_brake_deficit_margin_m"].default
  lead_brake_release_lookahead_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_lookahead_s"].default
  lead_brake_release_min_pullaway_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_min_pullaway_mps"].default
  lead_brake_release_near_target_margin_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_near_target_margin_m"].default
  lead_brake_release_near_target_max_closing_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_near_target_max_closing_mps"].default
  lead_brake_release_near_target_floor_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_near_target_floor_mps2"].default
  lead_brake_release_lead_decel_min_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_lead_decel_min_mps2"].default
  lead_brake_release_approach_floor_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_approach_floor_mps2"].default
  lead_brake_release_lead_decel_project_gain: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_lead_decel_project_gain"].default
  lead_brake_release_vrel_credit_cap_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_vrel_credit_cap_m"].default
  lead_brake_release_recovery_proj_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_recovery_proj_s"].default
  lead_brake_release_coast_bias_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_brake_release_coast_bias_mps2"].default
  cutin_settle_duration_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_duration_s"].default
  cutin_settle_max_decel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_max_decel"].default
  cutin_settle_max_closing_speed_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_max_closing_speed_mps"].default
  cutin_settle_accel_bias_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cutin_settle_accel_bias_mps2"].default
  virtual_lead_slow_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["virtual_lead_slow_tau_s"].default
  drel_filter_tau_close_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_tau_close_s"].default
  drel_filter_tau_open_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_tau_open_s"].default
  drel_filter_open_slew_max_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_open_slew_max_mps"].default
  drel_filter_innovation_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_innovation_gate_m"].default
  drel_filter_closing_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["drel_filter_closing_gate_m"].default
  cruise_reacquire_pos_jerk_limit: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_reacquire_pos_jerk_limit"].default
  cruise_reacquire_jerk_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_reacquire_jerk_window_s"].default
  cruise_reacquire_jerk_ramp_mps3_per_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_reacquire_jerk_ramp_mps3_per_s"].default
  cruise_collapse_holdback_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_collapse_holdback_s"].default
  cruise_exit_lookback_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_exit_lookback_frames"].default
  cruise_exit_abrupt_prob_drop: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_exit_abrupt_prob_drop"].default
  cruise_relatch_blend_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_blend_s"].default
  cruise_relatch_blend_jerk_mps3: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_blend_jerk_mps3"].default
  cruise_relatch_release_jerk_mps3: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_release_jerk_mps3"].default
  cruise_relatch_urgent_ttc_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_urgent_ttc_s"].default
  cruise_relatch_urgent_closing_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_urgent_closing_mps"].default
  cruise_relatch_bypass_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_bypass_decel_mps2"].default
  cruise_relatch_urgent_lead_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_urgent_lead_decel_mps2"].default
  cruise_relatch_max_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["cruise_relatch_max_decel_mps2"].default
  handoff_limit_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["handoff_limit_window_s"].default
  handoff_limit_max_delta_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["handoff_limit_max_delta_mps2"].default
  handoff_inside_df_positive_cap_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["handoff_inside_df_positive_cap_mps2"].default
  comfort_jerk_limit_mps3: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["comfort_jerk_limit_mps3"].default
  comfort_jerk_bypass_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["comfort_jerk_bypass_decel_mps2"].default
  lead_prob_enter: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_prob_enter"].default
  lead_prob_exit: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_prob_exit"].default
  lead_source_acquire_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_source_acquire_frames"].default
  lead_source_release_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_source_release_frames"].default
  phantom_lead_hold_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["phantom_lead_hold_s"].default
  phantom_lead_stable_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["phantom_lead_stable_frames"].default
  phantom_lead_decel_hold_factor: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["phantom_lead_decel_hold_factor"].default
  phantom_lead_decel_trend_gain: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["phantom_lead_decel_trend_gain"].default
  flutter_detect_transitions: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_detect_transitions"].default
  flutter_detect_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_detect_window_s"].default
  flutter_clamp_jerk_mps3: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_clamp_jerk_mps3"].default
  flutter_clamp_bypass_decel_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_clamp_bypass_decel_mps2"].default
  flutter_clamp_brake_jerk_mps3: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["flutter_clamp_brake_jerk_mps3"].default
  model_lead_filter_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_tau_s"].default
  model_lead_filter_open_slew_max_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_open_slew_max_mps"].default
  model_lead_filter_safe_ttc_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_safe_ttc_s"].default
  model_lead_filter_assoc_drel_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_assoc_drel_m"].default
  model_lead_filter_vrel_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_vrel_tau_s"].default
  model_lead_filter_fast_vrel_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_fast_vrel_tau_s"].default
  model_lead_filter_blend_tau_floor_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_blend_tau_floor_s"].default
  model_lead_filter_blend_close_lo_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_blend_close_lo_mps"].default
  model_lead_filter_blend_ttc_hi_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_blend_ttc_hi_s"].default
  model_lead_filter_blend_slew_boost_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_blend_slew_boost_mps"].default
  model_lead_filter_lag_comp_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_lag_comp_s"].default
  model_lead_filter_lag_comp_deadzone_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_lag_comp_deadzone_mps"].default
  model_lead_filter_lag_comp_fade_lo_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_lag_comp_fade_lo_mps"].default
  model_lead_filter_lag_comp_fade_hi_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_lag_comp_fade_hi_mps"].default
  model_lead_filter_fast_close_confirm_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_fast_close_confirm_frames"].default
  model_lead_filter_open_recovery_confirm_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_open_recovery_confirm_frames"].default
  model_lead_filter_open_recovery_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_open_recovery_tau_s"].default
  model_lead_filter_open_recovery_max_ego_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_open_recovery_max_ego_mps"].default
  model_lead_filter_open_recovery_innov_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_filter_open_recovery_innov_gate_m"].default
  model_lead_fcw_corrob_tol_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_fcw_corrob_tol_m"].default
  model_lead_fcw_corrob_min_agree: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_fcw_corrob_min_agree"].default
  model_lead_fcw_corrob_window: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_fcw_corrob_window"].default
  model_lead_fcw_corrob_raw_closing_min_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_fcw_corrob_raw_closing_min_mps"].default
  model_lead_fcw_corrob_raw_ttc_max_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_fcw_corrob_raw_ttc_max_s"].default
  lead_accel_corr_margin_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_margin_mps2"].default
  lead_accel_corr_meas_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_meas_tau_s"].default
  lead_accel_corr_ttc_guard_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_ttc_guard_s"].default
  lead_accel_corr_closing_guard_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_closing_guard_mps"].default
  lead_accel_corr_near_headway_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_near_headway_s"].default
  lead_accel_corr_closing_rearm_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_closing_rearm_mps"].default
  lead_accel_corr_ttc_rearm_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_ttc_rearm_s"].default
  lead_accel_corr_headway_rearm_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_headway_rearm_m"].default
  lead_accel_corr_settle_tau_mult: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_settle_tau_mult"].default
  lead_accel_corr_max_dt_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_max_dt_s"].default
  lead_accel_corr_amplify_gain: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_amplify_gain"].default
  lead_accel_corr_amplify_deadband_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_amplify_deadband_mps2"].default
  lead_accel_corr_amplify_cap_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_accel_corr_amplify_cap_mps2"].default
  lead_stabilizer_trend_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_stabilizer_trend_tau_s"].default
  lead_stabilizer_trend_drel_jump_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_stabilizer_trend_drel_jump_m"].default
  lead_stabilizer_trend_yrel_jump_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_stabilizer_trend_yrel_jump_m"].default
  model_lead_blend_min_span: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_blend_min_span"].default
  model_lead_blend_ttc_min_closing_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_blend_ttc_min_closing_mps"].default
  model_lead_assoc_dpath_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_assoc_dpath_gate_m"].default
  model_lead_assoc_y_raw_tol_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_assoc_y_raw_tol_m"].default
  model_lead_step_guard_abs_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_step_guard_abs_m"].default
  model_lead_step_guard_frac: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_step_guard_frac"].default
  lead_vlead_optimism_clamp_range_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_vlead_optimism_clamp_range_m"].default
  lead_vlead_optimism_clamp_gain: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_vlead_optimism_clamp_gain"].default
  lead_vlead_optimism_clamp_confirm_frames: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_vlead_optimism_clamp_confirm_frames"].default
  lead_vlead_optimism_clamp_slow_lead_frac: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_vlead_optimism_clamp_slow_lead_frac"].default
  closing_governor_margin_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_margin_mps"].default
  closing_governor_min_closing_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_min_closing_mps"].default
  closing_governor_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_window_s"].default
  closing_governor_accel_onset_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_accel_onset_mps2"].default
  closing_governor_pos_trust_excess_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_pos_trust_excess_mps"].default
  closing_governor_hold_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_hold_s"].default
  closing_governor_alead_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["closing_governor_alead_tau_s"].default
  launch_release_min_drel_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["launch_release_min_drel_m"].default
  launch_release_depart_gate_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["launch_release_depart_gate_m"].default
  launch_follow_accel_floor_max_mps2: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["launch_follow_accel_floor_max_mps2"].default

  @classmethod
  def defaults(cls) -> LeadResponseTuningConfig:
    return cls()

  def as_dict(self) -> dict[str, float]:
    return asdict(self)


def clamp_lead_response_tuning_value(attr: str, value: float) -> float:
  return LEAD_RESPONSE_TUNE_SPECS_BY_ATTR[attr].clamp(float(value))


def _read_float(params: Params, key: str, default: float) -> float:
  try:
    raw = params.get(key)
    if raw is None:
      return float(default)
    value = float(raw)
    return value if math.isfinite(value) else float(default)
  except Exception:
    return float(default)


def build_lead_response_tuning_config(values: dict[str, Any] | None = None) -> LeadResponseTuningConfig:
  values = {} if values is None else dict(values)
  clamped: dict[str, float] = {}
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    raw_value = values.get(spec.attr, spec.default)
    try:
      numeric_value = float(raw_value)
    except Exception:
      numeric_value = spec.default
    if not math.isfinite(numeric_value):
      numeric_value = spec.default
    clamped[spec.attr] = spec.clamp(numeric_value)
  return LeadResponseTuningConfig(**clamped)


def read_lead_response_tuning_config(params: Params) -> LeadResponseTuningConfig:
  values = {
    spec.attr: _read_float(params, spec.key, spec.default)
    for spec in LEAD_RESPONSE_TUNE_SPECS
  }
  return build_lead_response_tuning_config(values)


def set_lead_response_tuning_params(params: Params, overrides: dict[str, float]) -> dict[str, float]:
  applied: dict[str, float] = {}
  for attr, requested in overrides.items():
    spec = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR[attr]
    applied_value = spec.clamp(float(requested))
    params.put(spec.key, applied_value)
    applied[attr] = applied_value
  return applied


def reset_lead_response_tuning_params(params: Params) -> None:
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    params.remove(spec.key)


def get_lead_response_tune_rows(params: Params) -> list[dict[str, Any]]:
  effective = read_lead_response_tuning_config(params)
  rows: list[dict[str, Any]] = []
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    rows.append({
      "attr": spec.attr,
      "cli_name": spec.cli_name,
      "label": spec.label,
      "key": spec.key,
      "description": spec.description,
      "default": spec.default,
      "minimum": spec.minimum,
      "maximum": spec.maximum,
      "stored": params.get(spec.key),
      "effective": getattr(effective, spec.attr),
    })
  return rows


def format_lead_response_tune_summary(config: LeadResponseTuningConfig) -> str:
  return " ".join(
    f"{spec.label}={getattr(config, spec.attr):.3f}"
    for spec in LEAD_RESPONSE_TUNE_SPECS
  )
