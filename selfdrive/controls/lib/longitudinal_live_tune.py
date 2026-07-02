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
    default=0.12,
    minimum=0.0,
    maximum=0.75,
    description="Cap on the positive accel floor used to close a safe extra gap.",
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
    default=0.095,
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
)

LEAD_RESPONSE_TUNE_SPECS_BY_ATTR = {spec.attr: spec for spec in LEAD_RESPONSE_TUNE_SPECS}


@dataclass(frozen=True)
class LeadResponseTuningConfig:
  lead_preview_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_strength"].default
  lead_preview_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_gap_min_m"].default
  lead_preview_max_buffer_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_max_buffer_m"].default
  lead_acquire_window_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_acquire_window_s"].default
  gap_reclaim_strength: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_strength"].default
  gap_reclaim_gap_min_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_gap_min_m"].default
  gap_reclaim_max_accel: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_max_accel"].default
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
  lead_stabilizer_trend_tau_s: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_stabilizer_trend_tau_s"].default
  lead_stabilizer_trend_drel_jump_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_stabilizer_trend_drel_jump_m"].default
  lead_stabilizer_trend_yrel_jump_m: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_stabilizer_trend_yrel_jump_m"].default
  model_lead_blend_min_span: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_blend_min_span"].default
  model_lead_blend_ttc_min_closing_mps: float = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["model_lead_blend_ttc_min_closing_mps"].default

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
