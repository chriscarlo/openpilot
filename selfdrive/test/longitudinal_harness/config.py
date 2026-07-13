from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.hyundai.hyundaicanfd import CanBus
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.radar_interface import RADAR_START_ADDR
from opendbc.car.hyundai.values import CAR, HyundaiFlags
from opendbc.sunnypilot.car.hyundai.longitudinal.helpers import LongitudinalTuningType
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP, HyundaiSafetyFlagsSP

# The EV6's only lead source is vision: radard publishes aLeadTau=0.3 on every model
# lead (selfdrive/controls/radard.py ModelLeadTrack.update / get_RadarState_from_vision),
# not the radar-track default _LEAD_ACCEL_TAU=1.5 the legacy plant/harness hardcoded.
EV6_MODEL_LEAD_A_LEAD_TAU_S = 0.3

# Live-tune params dumped from the real device (2026-07-04 post CD9/Event-A deploy,
# dongle CHAUFFEUR_DEV_e521630c; dumped via the Params API, not file cat - lazily
# seeded keys read empty at the file level). Includes the driver's deliberate live
# deltas: ModelLeadFilterVRelTauS=0.60, HandoffInsideDfPositiveCapMps2=10.0.
# Drop in a newer dump by pointing resolve_ev6_vehicle_config(livetune_snapshot=...) at it.
DEVICE_LIVETUNE_SNAPSHOT_PATH = (
  Path(__file__).resolve().parents[3] / "docs" / "chauffeur" / "longitudinal" / "device_livetune_snapshot_20260704.txt"
)


def load_livetune_snapshot(source: str | Path | dict[str, Any] | None = None) -> dict[str, str]:
  """Load a device param dump as a {param_name: value} mapping.

  Accepts a dict (returned normalized) or a path to a dump of "Key = value" lines,
  the format produced by the re-dump command documented in
  docs/chauffeur/longitudinal/test_runtime_gap_audit_20260701.md.
  """
  if source is None:
    source = DEVICE_LIVETUNE_SNAPSHOT_PATH
  if isinstance(source, dict):
    return {str(key): str(value) for key, value in source.items()}

  values: dict[str, str] = {}
  for line in Path(source).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, _, value = line.partition("=")
    values[key.strip()] = value.strip()
  return values


FRIENDLY_PARAM_NAMES = {
  "obstacle_cost": "Longitudinal.LiveTune.ObstacleCost",
  "accel_change_cost": "Longitudinal.LiveTune.AccelChangeCost",
  "accel_cost": "Longitudinal.LiveTune.AccelCost",
  "lead_preview_strength": "Longitudinal.LiveTune.LeadPreviewStrength",
  "lead_preview_gap_min_m": "Longitudinal.LiveTune.LeadPreviewGapMinM",
  "lead_preview_max_buffer_m": "Longitudinal.LiveTune.LeadPreviewMaxBufferM",
  "lead_preview_min_speed_mps": "Longitudinal.LiveTune.LeadPreviewMinSpeedMps",
  "lead_acquire_window_s": "Longitudinal.LiveTune.LeadAcquireWindowS",
  "gap_reclaim_strength": "Longitudinal.LiveTune.GapReclaimStrength",
  "gap_reclaim_gap_min_m": "Longitudinal.LiveTune.GapReclaimGapMinM",
  "gap_reclaim_max_accel": "Longitudinal.LiveTune.GapReclaimMaxAccel",
  "gap_reclaim_follow_max_accel": "Longitudinal.LiveTune.GapReclaimFollowMaxAccel",
  "gap_reclaim_taper_gain": "Longitudinal.LiveTune.GapReclaimTaperGain",
  "lead_keepup_strength": "Longitudinal.LiveTune.LeadKeepUpStrength",
  "lead_keepup_gap_min_m": "Longitudinal.LiveTune.LeadKeepUpGapMinM",
  "lead_keepup_max_accel": "Longitudinal.LiveTune.LeadKeepUpMaxAccel",
  "lead_slowdown_strength": "Longitudinal.LiveTune.LeadSlowdownStrength",
  "lead_slowdown_max_decel": "Longitudinal.LiveTune.LeadSlowdownMaxDecel",
  "lead_slowdown_kinematic_headroom": "Longitudinal.LiveTune.LeadSlowdownKinematicHeadroom",
  "lead_slowdown_kinematic_margin_m": "Longitudinal.LiveTune.LeadSlowdownKinematicMarginM",
  "lead_slowdown_kinematic_oncoming_vlead_mps": "Longitudinal.LiveTune.LeadSlowdownKinematicOncomingVLeadMps",
  "lead_handoff_stopping_need_decel_mps2": "Longitudinal.LiveTune.LeadHandoffStoppingNeedDecelMps2",
  "lead_handoff_stopping_need_ref_speed_mps": "Longitudinal.LiveTune.LeadHandoffStoppingNeedRefSpeedMps",
  "lead_brake_release_vrel_credit_cap_m": "Longitudinal.LiveTune.LeadBrakeReleaseVrelCreditCapM",
  "lead_brake_release_recovery_proj_s": "Longitudinal.LiveTune.LeadBrakeReleaseRecoveryProjS",
  "cutin_settle_duration_s": "Longitudinal.LiveTune.CutInSettleDurationS",
  "cutin_settle_max_decel": "Longitudinal.LiveTune.CutInSettleMaxDecel",
  "cutin_settle_max_closing_speed_mps": "Longitudinal.LiveTune.CutInSettleMaxClosingSpeedMps",
  "cutin_settle_accel_bias_mps2": "Longitudinal.LiveTune.CutInSettleAccelBiasMps2",
  "drel_filter_tau_close_s": "Longitudinal.LiveTune.DRelFilterTauCloseS",
  "drel_filter_tau_open_s": "Longitudinal.LiveTune.DRelFilterTauOpenS",
  "drel_filter_open_slew_max_mps": "Longitudinal.LiveTune.DRelFilterOpenSlewMaxMps",
  "drel_filter_innovation_gate_m": "Longitudinal.LiveTune.DRelFilterInnovationGateM",
  "drel_filter_closing_gate_m": "Longitudinal.LiveTune.DRelFilterClosingGateM",
  "cruise_reacquire_pos_jerk_limit": "Longitudinal.LiveTune.CruiseReacquirePosJerkLimit",
  "cruise_reacquire_jerk_window_s": "Longitudinal.LiveTune.CruiseReacquireJerkWindowS",
  "cruise_reacquire_jerk_ramp": "Longitudinal.LiveTune.CruiseReacquireJerkRamp",
  "lead_prob_enter": "Longitudinal.LiveTune.LeadProbEnter",
  "lead_prob_exit": "Longitudinal.LiveTune.LeadProbExit",
  "lead_source_acquire_frames": "Longitudinal.LiveTune.LeadSourceAcquireFrames",
  "lead_source_release_frames": "Longitudinal.LiveTune.LeadSourceReleaseFrames",
  "phantom_lead_hold_s": "Longitudinal.LiveTune.PhantomLeadHoldS",
  "phantom_lead_stable_frames": "Longitudinal.LiveTune.PhantomLeadStableFrames",
  "phantom_lead_decel_hold_factor": "Longitudinal.LiveTune.PhantomLeadDecelHoldFactor",
  "phantom_lead_decel_trend_gain": "Longitudinal.LiveTune.PhantomLeadDecelTrendGain",
  "flutter_detect_transitions": "Longitudinal.LiveTune.FlutterDetectTransitions",
  "flutter_detect_window_s": "Longitudinal.LiveTune.FlutterDetectWindowS",
  "flutter_clamp_jerk_mps3": "Longitudinal.LiveTune.FlutterClampJerkMps3",
  "flutter_clamp_bypass_decel_mps2": "Longitudinal.LiveTune.FlutterClampBypassDecelMps2",
  "flutter_clamp_brake_jerk_mps3": "Longitudinal.LiveTune.FlutterClampBrakeJerkMps3",
  "model_lead_filter_tau_s": "Longitudinal.LiveTune.ModelLeadFilterTauS",
  "model_lead_filter_open_slew_max_mps": "Longitudinal.LiveTune.ModelLeadFilterOpenSlewMaxMps",
  "model_lead_filter_safe_ttc_s": "Longitudinal.LiveTune.ModelLeadFilterSafeTtcS",
  "model_lead_filter_assoc_drel_m": "Longitudinal.LiveTune.ModelLeadFilterAssocDRelM",
  "model_lead_filter_vrel_tau_s": "Longitudinal.LiveTune.ModelLeadFilterVRelTauS",
  "model_lead_filter_fast_vrel_tau_s": "Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS",
  "model_lead_filter_blend_tau_floor_s": "Longitudinal.LiveTune.ModelLeadFilterBlendTauFloorS",
  "model_lead_filter_blend_close_lo_mps": "Longitudinal.LiveTune.ModelLeadFilterBlendCloseLoMps",
  "model_lead_filter_blend_ttc_hi_s": "Longitudinal.LiveTune.ModelLeadFilterBlendTtcHiS",
  "model_lead_filter_blend_slew_boost_mps": "Longitudinal.LiveTune.ModelLeadFilterBlendSlewBoostMps",
  "model_lead_filter_lag_comp_s": "Longitudinal.LiveTune.ModelLeadFilterLagCompS",
  "model_lead_filter_lag_comp_deadzone_mps": "Longitudinal.LiveTune.ModelLeadFilterLagCompDeadzoneMps",
  "model_lead_filter_lag_comp_fade_lo_mps": "Longitudinal.LiveTune.ModelLeadFilterLagCompFadeLoMps",
  "model_lead_filter_lag_comp_fade_hi_mps": "Longitudinal.LiveTune.ModelLeadFilterLagCompFadeHiMps",
  "model_lead_filter_fast_close_confirm_frames": "Longitudinal.LiveTune.ModelLeadFilterFastCloseConfirmFrames",
  "model_lead_filter_open_recovery_confirm_frames": "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryConfirmFrames",
  "model_lead_filter_open_recovery_tau_s": "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryTauS",
  "model_lead_filter_open_recovery_max_ego_mps": "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryMaxEgoMps",
  "model_lead_filter_open_recovery_innov_gate_m": "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryInnovGateM",
  "model_lead_fcw_corrob_tol_m": "Longitudinal.LiveTune.ModelLeadFcwCorrobTolM",
  "model_lead_fcw_corrob_min_agree": "Longitudinal.LiveTune.ModelLeadFcwCorrobMinAgree",
  "model_lead_fcw_corrob_window": "Longitudinal.LiveTune.ModelLeadFcwCorrobWindow",
  "lead_accel_corr_margin_mps2": "Longitudinal.LiveTune.LeadAccelCorrMarginMps2",
  "lead_accel_corr_meas_tau_s": "Longitudinal.LiveTune.LeadAccelCorrMeasTauS",
  "lead_accel_corr_ttc_guard_s": "Longitudinal.LiveTune.LeadAccelCorrTtcGuardS",
  "lead_accel_corr_closing_guard_mps": "Longitudinal.LiveTune.LeadAccelCorrClosingGuardMps",
  "lead_accel_corr_near_headway_s": "Longitudinal.LiveTune.LeadAccelCorrNearHeadwayS",
  "lead_accel_corr_amplify_model_decel_min_mps2": "Longitudinal.LiveTune.LeadAccelCorrAmplifyModelDecelMinMps2",
  "use_kalman_drel_filter": "Longitudinal.LiveTune.UseKalmanDRelFilter",
  "kalman_drel_q": "Longitudinal.LiveTune.KalmanDRelQ",
  "kalman_drel_r": "Longitudinal.LiveTune.KalmanDRelR",
  "kalman_drel_gain_max": "Longitudinal.LiveTune.KalmanDRelGainMax",
  "kalman_drel_deadband_m": "Longitudinal.LiveTune.KalmanDRelDeadbandM",
  "accel_personality": "AccelPersonality",
  "longitudinal_personality": "LongitudinalPersonality",
  "vibe_enabled": "VibePersonalityEnabled",
  "vibe_follow_enabled": "VibeFollowPersonalityEnabled",
  "vibe_accel_enabled": "VibeAccelPersonalityEnabled",
  "hyundai_tuning_mode": "HyundaiLongitudinalTuning",
  "long_tuning_custom_toggle": "LongTuningCustomToggle",
  "long_tuning_accel_min": "LongTuningAccelMin",
  "long_tuning_accel_max": "LongTuningAccelMax",
  "long_tuning_v_ego_stopping": "LongTuningVEgoStopping",
  "long_tuning_stopping_decel_rate": "LongTuningStoppingDecelRate",
  "long_tuning_min_upper_jerk": "LongTuningMinUpperJerk",
  "long_tuning_min_lower_jerk": "LongTuningMinLowerJerk",
  "long_tuning_jerk_limits": "LongTuningJerkLimits",
}

DEFAULT_PARAM_VALUES = {
  "DynamicExperimentalControl": "0",
  "ExperimentalMode": "0",
  "AccelPersonality": "0",
  "LongitudinalPersonality": "1",
  "Longitudinal.LiveTune.ObstacleCost": "2.0",
  "Longitudinal.LiveTune.AccelChangeCost": "400.0",
  "Longitudinal.LiveTune.AccelCost": "1.0",
  "Longitudinal.LiveTune.LeadPreviewStrength": "1.5",
  "Longitudinal.LiveTune.LeadPreviewGapMinM": "1.0",
  "Longitudinal.LiveTune.LeadPreviewMaxBufferM": "10.0",
  "Longitudinal.LiveTune.LeadPreviewMinSpeedMps": "6.0",
  "Longitudinal.LiveTune.LeadAcquireWindowS": "1.5",
  "Longitudinal.LiveTune.GapReclaimStrength": "0.55",
  "Longitudinal.LiveTune.GapReclaimGapMinM": "3.0",
  "Longitudinal.LiveTune.GapReclaimMaxAccel": "0.30",
  "Longitudinal.LiveTune.GapReclaimFollowMaxAccel": "0.32",
  "Longitudinal.LiveTune.GapReclaimTaperGain": "2.0",
  "Longitudinal.LiveTune.LeadKeepUpStrength": "1.15",
  "Longitudinal.LiveTune.LeadKeepUpGapMinM": "0.10",
  "Longitudinal.LiveTune.LeadKeepUpMaxAccel": "0.22",
  "Longitudinal.LiveTune.LeadSlowdownStrength": "0.35",
  "Longitudinal.LiveTune.LeadSlowdownMaxDecel": "4.0",
  "Longitudinal.LiveTune.LeadSlowdownKinematicHeadroom": "1.5",
  "Longitudinal.LiveTune.LeadSlowdownKinematicMarginM": "4.0",
  "Longitudinal.LiveTune.LeadSlowdownKinematicOncomingVLeadMps": "-2.5",
  "Longitudinal.LiveTune.LeadHandoffStoppingNeedDecelMps2": "0.80",
  "Longitudinal.LiveTune.LeadHandoffStoppingNeedRefSpeedMps": "8.0",
  "Longitudinal.LiveTune.LeadBrakeReleaseMinSpeedMps": "5.0",
  "Longitudinal.LiveTune.LeadBrakeReleaseBrakeDeficitMarginM": "1.5",
  "Longitudinal.LiveTune.LeadBrakeReleaseLookaheadS": "2.0",
  "Longitudinal.LiveTune.LeadBrakeReleaseMinPullawayMps": "0.10",
  "Longitudinal.LiveTune.LeadBrakeReleaseNearTargetMarginM": "1.5",
  "Longitudinal.LiveTune.LeadBrakeReleaseNearTargetMaxClosingMps": "0.75",
  "Longitudinal.LiveTune.LeadBrakeReleaseNearTargetFloorMps2": "-0.05",
  "Longitudinal.LiveTune.LeadBrakeReleaseLeadDecelMinMps2": "-0.75",
  "Longitudinal.LiveTune.LeadBrakeReleaseApproachFloorMps2": "-0.60",
  "Longitudinal.LiveTune.LeadBrakeReleaseVrelCreditCapM": "10.0",
  "Longitudinal.LiveTune.LeadBrakeReleaseRecoveryProjS": "3.0",
  "Longitudinal.LiveTune.LeadBrakeReleaseCoastBiasMps2": "0.05",
  "Longitudinal.LiveTune.CutInSettleDurationS": "6.0",
  "Longitudinal.LiveTune.CutInSettleMaxDecel": "0.15",
  "Longitudinal.LiveTune.CutInSettleMaxClosingSpeedMps": "2.2",
  "Longitudinal.LiveTune.CutInSettleAccelBiasMps2": "0.12",
  "Longitudinal.LiveTune.VirtualLeadSlowTauS": "1.30",
  "Longitudinal.LiveTune.UseKalmanDRelFilter": "1",
  "Longitudinal.LiveTune.KalmanDRelQ": "0.5",
  "Longitudinal.LiveTune.KalmanDRelR": "6.0",
  "Longitudinal.LiveTune.KalmanDRelGainMax": "0.25",
  "Longitudinal.LiveTune.KalmanDRelDeadbandM": "0.75",
  "Longitudinal.LiveTune.DRelFilterTauCloseS": "0.30",
  "Longitudinal.LiveTune.DRelFilterTauOpenS": "0.80",
  "Longitudinal.LiveTune.DRelFilterOpenSlewMaxMps": "1.80",
  "Longitudinal.LiveTune.DRelFilterInnovationGateM": "30.0",
  "Longitudinal.LiveTune.DRelFilterClosingGateM": "12.0",
  "Longitudinal.LiveTune.CruiseReacquirePosJerkLimit": "0.08",
  "Longitudinal.LiveTune.CruiseReacquireJerkWindowS": "3.0",
  "Longitudinal.LiveTune.CruiseReacquireJerkRamp": "0.8",
  "Longitudinal.LiveTune.LeadProbEnter": "0.60",
  "Longitudinal.LiveTune.LeadProbExit": "0.25",
  "Longitudinal.LiveTune.LeadSourceAcquireFrames": "1.0",
  "Longitudinal.LiveTune.LeadSourceReleaseFrames": "20.0",
  "Longitudinal.LiveTune.PhantomLeadHoldS": "0.80",
  "Longitudinal.LiveTune.PhantomLeadStableFrames": "3.0",
  "Longitudinal.LiveTune.PhantomLeadDecelHoldFactor": "1.0",
  "Longitudinal.LiveTune.PhantomLeadDecelTrendGain": "1.0",
  "Longitudinal.LiveTune.FlutterDetectTransitions": "2.0",
  "Longitudinal.LiveTune.FlutterDetectWindowS": "1.0",
  "Longitudinal.LiveTune.FlutterClampJerkMps3": "0.12",
  "Longitudinal.LiveTune.FlutterClampBypassDecelMps2": "1.5",
  "Longitudinal.LiveTune.FlutterClampBrakeJerkMps3": "1.5",
  "Longitudinal.LiveTune.ModelLeadFilterTauS": "2.80",
  "Longitudinal.LiveTune.ModelLeadFilterOpenSlewMaxMps": "1.20",
  "Longitudinal.LiveTune.ModelLeadFilterSafeTtcS": "4.00",
  "Longitudinal.LiveTune.ModelLeadFilterAssocDRelM": "12.0",
  "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.40",
  "Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS": "0.16",
  "Longitudinal.LiveTune.ModelLeadFilterBlendTauFloorS": "0.30",
  "Longitudinal.LiveTune.ModelLeadFilterBlendCloseLoMps": "1.0",
  "Longitudinal.LiveTune.ModelLeadFilterBlendTtcHiS": "12.0",
  "Longitudinal.LiveTune.ModelLeadFilterBlendSlewBoostMps": "6.0",
  "Longitudinal.LiveTune.ModelLeadFilterLagCompS": "0.6",
  "Longitudinal.LiveTune.ModelLeadFilterLagCompDeadzoneMps": "0.5",
  "Longitudinal.LiveTune.ModelLeadFilterLagCompFadeLoMps": "12.0",
  "Longitudinal.LiveTune.ModelLeadFilterLagCompFadeHiMps": "18.0",
  "Longitudinal.LiveTune.ModelLeadFilterFastCloseConfirmFrames": "2.0",
  "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryConfirmFrames": "4.0",
  "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryTauS": "0.5",
  "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryMaxEgoMps": "8.0",
  "Longitudinal.LiveTune.ModelLeadFilterOpenRecoveryInnovGateM": "2.5",
  "Longitudinal.LiveTune.ModelLeadFcwCorrobTolM": "2.5",
  "Longitudinal.LiveTune.ModelLeadFcwCorrobMinAgree": "2.0",
  "Longitudinal.LiveTune.ModelLeadFcwCorrobWindow": "3.0",
  "Longitudinal.LiveTune.LeadAccelCorrMarginMps2": "0.5",
  "Longitudinal.LiveTune.LeadAccelCorrMeasTauS": "0.3",
  "Longitudinal.LiveTune.LeadAccelCorrTtcGuardS": "8.0",
  "Longitudinal.LiveTune.LeadAccelCorrClosingGuardMps": "1.5",
  "Longitudinal.LiveTune.LeadAccelCorrNearHeadwayS": "1.2",
  "Longitudinal.LiveTune.LeadAccelCorrAmplifyModelDecelMinMps2": "0.10",
  "HyundaiLongitudinalTuning": str(LongitudinalTuningType.OFF),
  "LongTuningCustomToggle": "0",
  "LongTuningAccelMin": "-5.5",
  "LongTuningAccelMax": "2.0",
  "LongTuningVEgoStopping": "0.25",
  "LongTuningStoppingDecelRate": "0.40",
  "LongTuningMinUpperJerk": "0.5",
  "LongTuningMinLowerJerk": "0.5",
  "LongTuningJerkLimits": "4.0",
  "VibePersonalityEnabled": "1",
  "VibeFollowPersonalityEnabled": "1",
  "VibeAccelPersonalityEnabled": "1",
  "VibeTune.Follow.Relaxed.Headway0": "1.25",
  "VibeTune.Follow.Relaxed.Headway1": "1.60",
  "VibeTune.Follow.Relaxed.Headway2": "1.85",
  "VibeTune.Follow.Relaxed.Headway3": "2.20",
  "VibeTune.Follow.Standard.Headway0": "1.25",
  "VibeTune.Follow.Standard.Headway1": "1.30",
  "VibeTune.Follow.Standard.Headway2": "1.38",
  "VibeTune.Follow.Standard.Headway3": "1.40",
  "VibeTune.Follow.Aggressive.Headway0": "1.19",
  "VibeTune.Follow.Aggressive.Headway1": "1.19",
  "VibeTune.Follow.Aggressive.Headway2": "1.29",
  "VibeTune.Follow.Aggressive.Headway3": "1.29",
  "VibeTune.Brake.Relaxed.Decel0": "-0.50",
  "VibeTune.Brake.Relaxed.Decel1": "-0.80",
  "VibeTune.Brake.Relaxed.Decel2": "-1.20",
  "VibeTune.Brake.Relaxed.Decel3": "-1.20",
  "VibeTune.Brake.Standard.Decel0": "-1.05",
  "VibeTune.Brake.Standard.Decel1": "-1.15",
  "VibeTune.Brake.Standard.Decel2": "-1.30",
  "VibeTune.Brake.Standard.Decel3": "-1.30",
  "VibeTune.Brake.Aggressive.Decel0": "-1.10",
  "VibeTune.Brake.Aggressive.Decel1": "-1.25",
  "VibeTune.Brake.Aggressive.Decel2": "-1.40",
  "VibeTune.Brake.Aggressive.Decel3": "-1.40",
  "VibeTune.Accel.Eco.Max0": "1.10",
  "VibeTune.Accel.Eco.Max1": "1.00",
  "VibeTune.Accel.Eco.Max2": "0.85",
  "VibeTune.Accel.Eco.Max3": "0.76",
  "VibeTune.Accel.Eco.Max4": "0.58",
  "VibeTune.Accel.Eco.Max5": "0.46",
  "VibeTune.Accel.Eco.Max6": "0.365",
  "VibeTune.Accel.Eco.Max7": "0.317",
  "VibeTune.Accel.Eco.Max8": "0.089",
  "VibeTune.Accel.Normal.Max0": "2.00",
  "VibeTune.Accel.Normal.Max1": "2.00",
  "VibeTune.Accel.Normal.Max2": "1.42",
  "VibeTune.Accel.Normal.Max3": "1.10",
  "VibeTune.Accel.Normal.Max4": "0.65",
  "VibeTune.Accel.Normal.Max5": "0.56",
  "VibeTune.Accel.Normal.Max6": "0.43",
  "VibeTune.Accel.Normal.Max7": "0.36",
  "VibeTune.Accel.Normal.Max8": "0.12",
  "VibeTune.Accel.Sport.Max0": "4.00",
  "VibeTune.Accel.Sport.Max1": "4.00",
  "VibeTune.Accel.Sport.Max2": "3.80",
  "VibeTune.Accel.Sport.Max3": "3.50",
  "VibeTune.Accel.Sport.Max4": "2.00",
  "VibeTune.Accel.Sport.Max5": "1.75",
  "VibeTune.Accel.Sport.Max6": "1.325",
  "VibeTune.Accel.Sport.Max7": "1.15",
  "VibeTune.Accel.Sport.Max8": "0.50",
  "VisionTurnSpeedControl": "1",
  "SpeedLimitControl": "1",
  "RTIEnabled": "1",
  "WeatherAwareControlEnabled": "1",
}


@dataclass(frozen=True)
class NoiseProfile:
  name: str
  drel_frac: float
  drel_floor_m: float
  vrel_factor: float
  vrel_floor_mps: float
  # Measured EV6 heavy-tail dRel model: distance-band sigmas plus occasional large
  # outlier jumps, ported from the calibrated generator in
  # selfdrive/controls/lib/tests/test_lead_filter_ab.py (fitted to real EV6 captures).
  # When True the band model replaces the fractional gaussian above.
  drel_measured_bands: bool = False
  drel_outlier_extra_sigma_m: float = 0.0
  # Lateral jitter on the published yRel/dPath (runtime dPath is model-path derived
  # and noisy; it gates leads in/out of MPC control via the role classifier).
  y_rel_sigma_m: float = 0.0
  d_path_sigma_m: float = 0.0
  # Transient model-lead probability dropouts (modelProb forced to 0 for the window),
  # exercising the phantom-hold / source-hysteresis stack.
  prob_dropout_rate_hz: float = 0.0
  prob_dropout_duration_s: float = 0.0


@dataclass(frozen=True)
class NoiseSeeds:
  drel: int
  vrel: int
  aego: int
  vego: int
  lat: int = 4
  prob: int = 5

  @classmethod
  def from_base(cls, base_seed: int) -> NoiseSeeds:
    return cls(
      drel=int(base_seed),
      vrel=int(base_seed) + 1,
      aego=int(base_seed) + 2,
      vego=int(base_seed) + 3,
      lat=int(base_seed) + 4,
      prob=int(base_seed) + 5,
    )

  def offset(self, delta: int) -> NoiseSeeds:
    return NoiseSeeds(
      drel=self.drel + delta,
      vrel=self.vrel + delta,
      aego=self.aego + delta,
      vego=self.vego + delta,
      lat=self.lat + delta,
      prob=self.prob + delta,
    )

  def with_overrides(self,
                     *,
                     drel: int | None = None,
                     vrel: int | None = None,
                     aego: int | None = None,
                     vego: int | None = None,
                     lat: int | None = None,
                     prob: int | None = None) -> NoiseSeeds:
    return NoiseSeeds(
      drel=self.drel if drel is None else drel,
      vrel=self.vrel if vrel is None else vrel,
      aego=self.aego if aego is None else aego,
      vego=self.vego if vego is None else vego,
      lat=self.lat if lat is None else lat,
      prob=self.prob if prob is None else prob,
    )

  def as_dict(self) -> dict[str, int]:
    return asdict(self)


NOISE_PROFILES: dict[str, NoiseProfile] = {
  "off": NoiseProfile("off", 0.0, 0.0, 0.0, 0.0),
  "realistic": NoiseProfile("realistic", 0.03, 0.75, 0.05, 0.05),
  "stress": NoiseProfile("stress", 0.07, 1.5, 0.10, 0.2),
  # Repo-measured EV6 lead-noise characterization in the loop:
  # - dRel band sigmas + 8 m extra-sigma outliers and vRel sigma 0.3 m/s from
  #   selfdrive/controls/lib/tests/test_lead_filter_ab.py (fitted to real EV6
  #   captures 2026-04-08; 2% of frames >12 m jumps).
  # - yRel/dPath jitter 0.35 m from the model-lead yStd used by
  #   .codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py.
  # - prob dropouts ~0.15 Hz x 0.7 s from the audit's real-log corroboration
  #   (11 lead status flips in 33 s of calm traffic; ~0.7 s vision dropouts) in
  #   docs/chauffeur/longitudinal/test_runtime_gap_audit_20260701.md.
  "ev6_measured": NoiseProfile(
    "ev6_measured", 0.0, 0.0, 0.0, 0.3,
    drel_measured_bands=True,
    drel_outlier_extra_sigma_m=8.0,
    y_rel_sigma_m=0.35,
    d_path_sigma_m=0.35,
    prob_dropout_rate_hz=0.15,
    prob_dropout_duration_s=0.7,
  ),
}


@dataclass(frozen=True)
class VehiclePlantConfig:
  command_delay_s: float
  accel_rise_tau_s: float = 0.30
  regen_tau_s: float = 0.40
  brake_tau_s: float = 0.20
  regen_threshold_mps2: float = -1.0
  aego_measure_delay_s: float = 0.10
  aego_measure_noise_std: float = 0.0
  vego_measure_noise_std: float = 0.0

  def as_dict(self) -> dict[str, float]:
    return asdict(self)


@dataclass
class ResolvedVehicleConfig:
  topology: str
  requested_controller_mode: str
  resolved_controller_mode: str
  hyundai_tuning_mode: int
  tune_source: str
  fidelity_source: str
  params: dict[str, str]
  cp: structs.CarParams
  cp_sp: structs.CarParamsSP
  cc_sp_params: list[structs.CarControlSP.Param]
  plant_config: VehiclePlantConfig
  # aLeadTau published on synthesized leads; EV6 model leads always carry 0.3
  # (radard.py). Non-EV6/legacy consumers must set their value explicitly.
  a_lead_tau_s: float = EV6_MODEL_LEAD_A_LEAD_TAU_S
  # Lead perception stage between synthesized ground truth and the planner:
  # "direct" fabricates radarState from directives (legacy), "radard" routes the
  # leads through the real radard pipeline (Schmitt latch + ModelLeadTracker).
  perception_filter: str = "direct"
  metadata: dict[str, Any] = field(default_factory=dict)

  def describe(self) -> dict[str, Any]:
    return {
      "candidate": str(self.cp.carFingerprint),
      "topology": self.topology,
      "requestedControllerMode": self.requested_controller_mode,
      "resolvedControllerMode": self.resolved_controller_mode,
      "perceptionFilter": self.perception_filter,
      "aLeadTauS": float(self.a_lead_tau_s),
      "hyundaiTuningMode": self.hyundai_tuning_mode,
      "tuneSource": self.tune_source,
      "fidelitySource": self.fidelity_source,
      "openpilotLongitudinalControl": bool(self.cp.openpilotLongitudinalControl),
      "pcmCruise": bool(self.cp.pcmCruise),
      "radarUnavailable": bool(self.cp.radarUnavailable),
      "cpFlags": int(self.cp.flags),
      "spFlags": int(self.cp_sp.flags),
      "spSafetyParam": int(self.cp_sp.safetyParam),
      "longitudinalActuatorDelay": float(self.cp.longitudinalActuatorDelay),
      "vEgoStopping": float(self.cp.vEgoStopping),
      "startingState": bool(self.cp.startingState),
      "plantConfig": self.plant_config.as_dict(),
      **self.metadata,
    }


def normalize_param_overrides(overrides: dict[str, Any] | None) -> dict[str, str]:
  if overrides is None:
    return {}

  normalized: dict[str, str] = {}
  for raw_key, raw_value in overrides.items():
    key = FRIENDLY_PARAM_NAMES.get(raw_key, raw_key)
    if isinstance(raw_value, bool):
      normalized[key] = "1" if raw_value else "0"
    else:
      normalized[key] = str(raw_value)
  return normalized


def build_cc_sp_params(params: dict[str, str]) -> list[structs.CarControlSP.Param]:
  return [structs.CarControlSP.Param(key=key, value=str(value)) for key, value in sorted(params.items())]


def build_synthetic_ev6_inputs(topology: str) -> tuple[dict[int, dict[int, int]], list[structs.CarParams.CarFw]]:
  topology = topology.lower()
  if topology not in ("lka", "lfa"):
    raise ValueError(f"unsupported EV6 topology '{topology}'")

  fingerprint = gen_empty_fingerprint()
  fingerprint[1][RADAR_START_ADDR] = 8
  car_fw: list[structs.CarParams.CarFw] = []

  if topology == "lka":
    cam_can = CanBus(None, fingerprint).CAM
    fingerprint[cam_can][0x50] = 8
    fingerprint[cam_can][0x110] = 8
    car_fw.append(structs.CarParams.CarFw(ecu=structs.CarParams.Ecu.adas))

  return fingerprint, car_fw


def _apply_hyundai_tuning(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: dict[str, str]) -> None:
  hyundai_tuning_mode = int(params.get("HyundaiLongitudinalTuning", str(LongitudinalTuningType.OFF)))
  tuning_mask = HyundaiFlagsSP.LONG_TUNING_DYNAMIC.value | HyundaiFlagsSP.LONG_TUNING_PREDICTIVE.value
  CP_SP.flags &= ~tuning_mask

  if hyundai_tuning_mode == LongitudinalTuningType.DYNAMIC:
    CP_SP.flags |= HyundaiFlagsSP.LONG_TUNING_DYNAMIC.value
  elif hyundai_tuning_mode == LongitudinalTuningType.PREDICTIVE:
    CP_SP.flags |= HyundaiFlagsSP.LONG_TUNING_PREDICTIVE.value

  CarInterface.get_longitudinal_tuning_sp(CP, CP_SP, params)


def resolve_ev6_vehicle_config(*,
                               topology: str = "lka",
                               controller_mode: str = "device",
                               tune_source: str = "defaults",
                               param_overrides: dict[str, Any] | None = None,
                               hyundai_tuning_mode: int | None = None,
                               snapshot_vehicle: dict[str, Any] | None = None,
                               snapshot_params: dict[str, Any] | None = None,
                               plant_overrides: dict[str, Any] | None = None,
                               livetune_snapshot: str | Path | dict[str, Any] | None = DEVICE_LIVETUNE_SNAPSHOT_PATH,
                               a_lead_tau_s: float | None = None,
                               perception_filter: str = "auto") -> ResolvedVehicleConfig:
  if controller_mode not in ("auto", "device", "passthrough", "shaped"):
    raise ValueError(f"unsupported controller_mode '{controller_mode}'")
  if perception_filter not in ("auto", "direct", "radard"):
    raise ValueError(f"unsupported perception_filter '{perception_filter}'")

  fingerprint, car_fw = build_synthetic_ev6_inputs(topology)
  params = dict(DEFAULT_PARAM_VALUES)
  if livetune_snapshot is not None:
    params.update(normalize_param_overrides(load_livetune_snapshot(livetune_snapshot)))
  params.update(normalize_param_overrides(snapshot_params))
  params.update(normalize_param_overrides(param_overrides))

  if hyundai_tuning_mode is not None:
    params["HyundaiLongitudinalTuning"] = str(int(hyundai_tuning_mode))

  if snapshot_vehicle and snapshot_vehicle.get("controllerMode") and controller_mode == "auto":
    controller_mode = str(snapshot_vehicle["controllerMode"])

  CP = CarInterface.get_params(CAR.KIA_EV6, fingerprint, car_fw, True, False, False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.KIA_EV6, fingerprint, car_fw, True, False)
  CP.openpilotLongitudinalControl = True
  CP.pcmCruise = False
  CP_SP.flags |= HyundaiFlagsSP.LONGITUDINAL_MAIN_CRUISE_TOGGLEABLE.value
  CP_SP.safetyParam |= HyundaiSafetyFlagsSP.LONG_MAIN_CRUISE_TOGGLEABLE

  if snapshot_vehicle and "spFlags" in snapshot_vehicle:
    CP_SP.flags = int(snapshot_vehicle["spFlags"])
  if snapshot_vehicle and "spSafetyParam" in snapshot_vehicle:
    CP_SP.safetyParam = int(snapshot_vehicle["spSafetyParam"])

  if controller_mode == "shaped" and int(params.get("HyundaiLongitudinalTuning", "0")) == LongitudinalTuningType.OFF:
    params["HyundaiLongitudinalTuning"] = str(LongitudinalTuningType.DYNAMIC)

  if controller_mode in ("device", "passthrough"):
    CP.radarUnavailable = True
  elif controller_mode == "shaped":
    CP.radarUnavailable = False
  elif controller_mode == "auto" and snapshot_vehicle and "radarUnavailable" in snapshot_vehicle:
    CP.radarUnavailable = bool(snapshot_vehicle["radarUnavailable"])

  _apply_hyundai_tuning(CP, CP_SP, params)

  if controller_mode == "device":
    # tici fidelity: the real EV6 CarController always routes actuators.accel through
    # LongitudinalController with CP.radarUnavailable=True, taking the no-radar EMA
    # branch (opendbc/sunnypilot/car/hyundai/longitudinal/controller.py calculate_accel)
    # regardless of the Hyundai tuning toggle.
    resolved_controller_mode = "device"
  else:
    resolved_controller_mode = (
      "shaped"
      if (int(params.get("HyundaiLongitudinalTuning", "0")) != LongitudinalTuningType.OFF and not CP.radarUnavailable)
      else "passthrough"
    )
  if controller_mode == "shaped" and resolved_controller_mode != "shaped":
    raise ValueError("requested shaped EV6 controller mode, but Hyundai runtime shaping is not active")
  if controller_mode == "passthrough" and resolved_controller_mode != "passthrough":
    raise ValueError("requested passthrough EV6 controller mode, but config resolved to shaped mode")

  if perception_filter == "auto":
    if snapshot_vehicle and snapshot_vehicle.get("perceptionFilter"):
      # Snapshot bundles record radarState as published on device, i.e. already
      # radard-filtered; route_extract labels them "direct" to avoid double-filtering.
      perception_filter = str(snapshot_vehicle["perceptionFilter"])
    elif resolved_controller_mode == "device":
      # tici fidelity: on the radar-less EV6 the only lead source is
      # modelV2.leadsV3 through radard's Schmitt prob latch + ModelLeadTracker
      # (selfdrive/controls/radard.py) before radarState reaches the planner.
      perception_filter = "radard"
    else:
      perception_filter = "direct"

  plant_config = VehiclePlantConfig(command_delay_s=float(CP.longitudinalActuatorDelay))
  if snapshot_vehicle and snapshot_vehicle.get("plantConfig"):
    plant_config = VehiclePlantConfig(**{**plant_config.as_dict(), **snapshot_vehicle["plantConfig"]})
  if plant_overrides:
    plant_config = VehiclePlantConfig(**{**plant_config.as_dict(), **plant_overrides})

  if a_lead_tau_s is None:
    if snapshot_vehicle and "aLeadTauS" in snapshot_vehicle:
      a_lead_tau_s = float(snapshot_vehicle["aLeadTauS"])
    else:
      a_lead_tau_s = EV6_MODEL_LEAD_A_LEAD_TAU_S

  metadata = {}
  if snapshot_vehicle:
    metadata.update({k: v for k, v in snapshot_vehicle.items() if k not in {"plantConfig"}})
  if livetune_snapshot is not None:
    metadata["livetuneSource"] = "dict" if isinstance(livetune_snapshot, dict) else str(livetune_snapshot)

  return ResolvedVehicleConfig(
    topology=topology,
    requested_controller_mode=controller_mode,
    resolved_controller_mode=resolved_controller_mode,
    hyundai_tuning_mode=int(params.get("HyundaiLongitudinalTuning", "0")),
    tune_source=tune_source,
    fidelity_source="snapshot" if snapshot_vehicle else "synthetic",
    params=params,
    cp=CP,
    cp_sp=CP_SP,
    cc_sp_params=build_cc_sp_params(params),
    plant_config=plant_config,
    a_lead_tau_s=float(a_lead_tau_s),
    perception_filter=perception_filter,
    metadata=metadata,
  )
