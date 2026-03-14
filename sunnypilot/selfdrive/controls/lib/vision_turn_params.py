import time
import importlib

from openpilot.common.numpy_fast import clip
from .vtsc_map_strategy import DEFAULT_MAP_STRATEGY, normalize_map_strategy

PARAM_REFRESH_S = 0.2  # 5 Hz live-tuning refresh


def update_vtsc_params(ctrl, *, force: bool = False) -> None:
  """Refresh VisionTurnController tunables from Params.

  This mirrors the previous inline _update_params implementation, but is
  consolidated in a separate module to keep the controller focused on
  control logic. It updates instance fields on `ctrl` and writes the small
  set of module-level physics knobs on the controller module.
  """
  # Use controller module's patched time.monotonic if tests patched it; fallback to stdlib otherwise.
  # When force=True, skip debounce to avoid depending on patched time during construction.
  vtc_mod = importlib.import_module(ctrl.__class__.__module__)
  if not force:
    try:
      monotonic = getattr(getattr(vtc_mod, "time", None), "monotonic", time.monotonic)
    except Exception:
      monotonic = time.monotonic
    tm = float(monotonic())
    if tm <= float(getattr(ctrl, "_last_params_update", 0.0)) + PARAM_REFRESH_S:
      return
  else:
    tm = float(time.monotonic())

  # Convenience accessors from the controller
  getf = ctrl._get_float_param
  getb = ctrl._get_bool_param
  gets = ctrl._get_string_param
  P = ctrl._params
  expert_enabled = bool(getb("VTSCExpertModeEnabled", getattr(ctrl, "_expert_mode_enabled", False)))
  ctrl._expert_mode_enabled = expert_enabled

  def expf(key: str, default: float, lo: float, hi: float, mod_attr=None) -> float:
    if not expert_enabled:
      return float(default)
    base = float(getattr(vtc_mod, mod_attr, default)) if mod_attr is not None else float(default)
    return float(getf(key, base, lo, hi))

  def expb(key: str, default: bool, mod_attr=None) -> bool:
    if not expert_enabled:
      return bool(default)
    base = bool(getattr(vtc_mod, mod_attr, default)) if mod_attr is not None else bool(default)
    return bool(getb(key, base))

  # Enable + high-level knobs
  ctrl._is_enabled = getb("VisionTurnSpeedControl", False)
  ctrl._map_strategy_mode = normalize_map_strategy(gets("VTSCMapStrategy", getattr(ctrl, "_map_strategy_mode", DEFAULT_MAP_STRATEGY)))
  ctrl._aggressiveness = getf("VisionTurnSpeedControlAggressiveness", getattr(ctrl, "_aggressiveness", 1.0), 0.5, 2.0)
  ctrl._fixed_lead_time_s = getf("VisionTurnSpeedControlFixedLeadTimeSeconds", getattr(ctrl, "_fixed_lead_time_s", 0.0), 0.0, 10.0)
  ctrl._curve_phase_offset_s = getf("VisionTurnSpeedControlCurvePhaseOffsetS", getattr(ctrl, "_curve_phase_offset_s", 0.0), -3.0, 3.0)
  ctrl._overshoot_phase_offset_s = getf("VisionTurnSpeedControlOvershootPhaseOffsetS", getattr(ctrl, "_overshoot_phase_offset_s", 0.0), -3.0, 3.0)
  ctrl._apex_exit_phase_offset_s = getf("VisionTurnSpeedControlApexExitPhaseOffsetS", getattr(ctrl, "_apex_exit_phase_offset_s", 0.0), -3.0, 3.0)
  # Lead-aware occlusion bypass
  ctrl._occl_bypass_with_lead = getb("VisionTurnSpeedControlOcclBypassWithLead", getattr(ctrl, "_occl_bypass_with_lead", True))
  ctrl._occl_bypass_headway_s = getf("VisionTurnSpeedControlOcclBypassHeadwayS", getattr(ctrl, "_occl_bypass_headway_s", 3.0), 0.5, 5.0)

  # Adaptive decel filtering + safety bias
  ctrl._filter_alpha = getf("VisionTurnSpeedControlFilterAlpha", getattr(ctrl, "_filter_alpha", 0.3), 0.1, 0.9)
  ctrl._hysteresis_threshold = getf("VisionTurnSpeedControlHysteresisThreshold", getattr(ctrl, "_hysteresis_threshold", 0.15), 0.1, 0.5)
  ctrl._safety_bias = getf("VisionTurnSpeedControlSafetyBias", getattr(ctrl, "_safety_bias", 0.1), 0.0, 0.5)
  ctrl._base_filter_alpha = ctrl._filter_alpha

  # Curvature EMA factor
  ctrl._curvature_ema_ratio = getf(
    "VisionTurnSpeedControlCurvatureEMAFactor",
    getattr(ctrl, "_curvature_ema_ratio", 0.3),
    0.1, 0.5,
  )

  # Smoothing bounds
  sm_max_decel_b = P.get("VisionTurnSpeedControlSmoothingMaxDecel")
  try:
    sm_max_decel = float(sm_max_decel_b.decode("utf-8") if isinstance(sm_max_decel_b, (bytes, bytearray)) else sm_max_decel_b) if sm_max_decel_b else ctrl._max_decel
  except Exception:
    sm_max_decel = ctrl._max_decel
  ctrl._max_decel = clip(sm_max_decel, 1.0, 7.0)

  sm_max_jerk_b = P.get("VisionTurnSpeedControlSmoothingMaxJerk")
  try:
    sm_max_jerk = float(sm_max_jerk_b.decode("utf-8") if isinstance(sm_max_jerk_b, (bytes, bytearray)) else sm_max_jerk_b) if sm_max_jerk_b else ctrl._max_jerk
  except Exception:
    sm_max_jerk = ctrl._max_jerk
  ctrl._max_jerk = clip(sm_max_jerk, 1.0, 12.0)

  accel_to_decel_b = P.get("VisionTurnSpeedControlAccelToDecelRatio")
  try:
    accel_to_decel = float(accel_to_decel_b.decode("utf-8") if isinstance(accel_to_decel_b, (bytes, bytearray)) else accel_to_decel_b) if accel_to_decel_b else ctrl._accel_to_decel_ratio
  except Exception:
    accel_to_decel = ctrl._accel_to_decel_ratio
  ctrl._accel_to_decel_ratio = clip(accel_to_decel, 1.0, 1.6)

  jerk_accel_mult_b = P.get("VisionTurnSpeedControlJerkAccelMultiplier")
  try:
    jerk_accel_mult = float(jerk_accel_mult_b.decode("utf-8") if isinstance(jerk_accel_mult_b, (bytes, bytearray)) else jerk_accel_mult_b) if jerk_accel_mult_b else ctrl._jerk_accel_multiplier
  except Exception:
    jerk_accel_mult = ctrl._jerk_accel_multiplier
  ctrl._jerk_accel_multiplier = clip(jerk_accel_mult, 1.0, 3.0)

  # Derived smoothing
  ctrl._max_accel = ctrl._accel_to_decel_ratio * ctrl._max_decel
  ctrl._max_jerk_accel = ctrl._jerk_accel_multiplier * ctrl._max_jerk

  # Visibility barrier and occlusion growth knobs on controller
  ctrl._vis_horizon_s = getf("VisionTurnSpeedControlVisHorizonS", getattr(ctrl, "_vis_horizon_s", 1.4))
  ctrl._vis_margin_m = getf("VisionTurnSpeedControlVisMarginM", getattr(ctrl, "_vis_margin_m", 10.0))
  ctrl._gamma_per_meter = getf("VisionTurnSpeedControlGammaPerMeter", getattr(ctrl, "_gamma_per_meter", 0.00025))
  ctrl._lat_jerk_cap = getf("VisionTurnSpeedControlLatJerkCap", getattr(ctrl, "_lat_jerk_cap", 2.0))
  if ctrl._lat_jerk_cap <= 0.0:
    ctrl._lat_jerk_cap = 1e9
  # FOV gating parameters
  try:
    ctrl._psi_fov_rad = getf("VisionTurnSpeedControlPsiFOVRad", getattr(ctrl, "_psi_fov_rad", 0.49), 0.1, 1.2)
  except Exception:
    ctrl._psi_fov_rad = getattr(ctrl, "_psi_fov_rad", 0.49)
  try:
    ctrl._psi_margin_rad = getf("VisionTurnSpeedControlPsiMarginRad", getattr(ctrl, "_psi_margin_rad", 0.087), 0.0, 0.5)
  except Exception:
    ctrl._psi_margin_rad = getattr(ctrl, "_psi_margin_rad", 0.087)
  # Occlusion arbitration PSI gate (separate from FOV psi)
  try:
    ctrl._psi_thresh_rad = float(getf("VTSC.PsiThreshRad", getattr(ctrl, "_psi_thresh_rad", 0.020)))
  except Exception:
    ctrl._psi_thresh_rad = getattr(ctrl, "_psi_thresh_rad", 0.020)
  try:
    ctrl._psi_hyst_rad = float(getf("VTSC.PsiHystRad", getattr(ctrl, "_psi_hyst_rad", 0.005)))
  except Exception:
    ctrl._psi_hyst_rad = getattr(ctrl, "_psi_hyst_rad", 0.005)
  # Additional FOV gate tunables
  ctrl._fov_k_min = getf("VisionTurnSpeedControlFOVKMin", getattr(ctrl, "_fov_k_min", 2e-4), 1e-6, 1e-2)
  ctrl._fov_k_freeway = getf("VisionTurnSpeedControlFOVKFreeway", getattr(ctrl, "_fov_k_freeway", 1e-5), 1e-7, 1e-3)
  ctrl._fov_s_long_m = getf("VisionTurnSpeedControlFOVSLongM", getattr(ctrl, "_fov_s_long_m", 120.0), 10.0, 400.0)
  ctrl._fov_pretrigger_time_s = getf("VisionTurnSpeedControlFOVPretriggerTimeS", getattr(ctrl, "_fov_pretrigger_time_s", 1.2), 0.1, 3.0)
  ctrl._fov_onset_boost_frames = int(getf("VisionTurnSpeedControlFOVOnsetBoostFrames", getattr(ctrl, "_fov_onset_boost_frames", 10), 0.0, 60.0))
  ctrl._fov_overshoot_frames = int(getf("VisionTurnSpeedControlFOVOvershootFrames", getattr(ctrl, "_fov_overshoot_frames", 10), 0.0, 60.0))
  ctrl._fov_ewma_tau_s = getf("VisionTurnSpeedControlFOVEWMATauS", getattr(ctrl, "_fov_ewma_tau_s", 0.5), 0.05, 3.0)
  ctrl._fov_N_on = int(getf("VisionTurnSpeedControlFOVNOn", getattr(ctrl, "_fov_N_on", 5), 1.0, 30.0))
  ctrl._fov_N_off = int(getf("VisionTurnSpeedControlFOVNOff", getattr(ctrl, "_fov_N_off", 10), 1.0, 60.0))

  # Anticipation & overshoot
  plan_decel_b = P.get("VisionTurnSpeedControlPlanningDecelLimit")
  try:
    plan_decel = float(plan_decel_b.decode("utf-8") if isinstance(plan_decel_b, (bytes, bytearray)) else plan_decel_b) if plan_decel_b else ctrl._planning_decel_limit
  except Exception:
    plan_decel = ctrl._planning_decel_limit
  ctrl._planning_decel_limit = clip(plan_decel, 1.0, 7.0)

  overshoot_safety_b = P.get("VisionTurnSpeedControlOvershootSafetyMargin")
  try:
    overshoot_safety = float(overshoot_safety_b.decode("utf-8") if isinstance(overshoot_safety_b, (bytes, bytearray)) else overshoot_safety_b) if overshoot_safety_b else ctrl._overshoot_safety_margin
  except Exception:
    overshoot_safety = ctrl._overshoot_safety_margin
  ctrl._overshoot_safety_margin = clip(overshoot_safety, 1.0, 1.5)

  overshoot_min_dist_b = P.get("VisionTurnSpeedControlOvershootMinDistance")
  try:
    overshoot_min_dist = float(overshoot_min_dist_b.decode("utf-8") if isinstance(overshoot_min_dist_b, (bytes, bytearray)) else overshoot_min_dist_b) if overshoot_min_dist_b else ctrl._overshoot_min_distance
  except Exception:
    overshoot_min_dist = ctrl._overshoot_min_distance
  ctrl._overshoot_min_distance = clip(overshoot_min_dist, 1.0, 200.0)

  anticip_red_b = P.get("VisionTurnSpeedControlAnticipationTargetReduction")
  try:
    anticip_red = float(anticip_red_b.decode("utf-8") if isinstance(anticip_red_b, (bytes, bytearray)) else anticip_red_b) if anticip_red_b else ctrl._anticipation_target_reduction
  except Exception:
    anticip_red = ctrl._anticipation_target_reduction
  ctrl._anticipation_target_reduction = clip(anticip_red, 0.9, 1.0)

  # Apex detection & boost
  apex_th_b = P.get("VisionTurnSpeedControlApexThreshold")
  try:
    apex_th = float(apex_th_b.decode("utf-8") if isinstance(apex_th_b, (bytes, bytearray)) else apex_th_b) if apex_th_b else ctrl._apex_threshold
  except Exception:
    apex_th = ctrl._apex_threshold
  ctrl._apex_threshold = clip(apex_th, 1e-6, 1e-3)

  apex_prom_b = P.get("VisionTurnSpeedControlApexProminence")
  try:
    apex_prom = float(apex_prom_b.decode("utf-8") if isinstance(apex_prom_b, (bytes, bytearray)) else apex_prom_b) if apex_prom_b else ctrl._apex_prominence
  except Exception:
    apex_prom = ctrl._apex_prominence
  ctrl._apex_prominence = clip(apex_prom, 1e-6, 1e-2)

  apex_hyst_b = P.get("VisionTurnSpeedControlApexHysteresisTime")
  try:
    apex_hyst = float(apex_hyst_b.decode("utf-8") if isinstance(apex_hyst_b, (bytes, bytearray)) else apex_hyst_b) if apex_hyst_b else ctrl._apex_hysteresis_time
  except Exception:
    apex_hyst = ctrl._apex_hysteresis_time
  ctrl._apex_hysteresis_time = clip(apex_hyst, 0.1, 10.0)

  apex_mpi_b = P.get("VisionTurnSpeedControlApexMetersPerIndex")
  try:
    apex_mpi = float(apex_mpi_b.decode("utf-8") if isinstance(apex_mpi_b, (bytes, bytearray)) else apex_mpi_b) if apex_mpi_b else ctrl._apex_meters_per_index
  except Exception:
    apex_mpi = ctrl._apex_meters_per_index
  ctrl._apex_meters_per_index = clip(apex_mpi, 0.5, 5.0)

  apex_near_idx_b = P.get("VisionTurnSpeedControlApexNearIndex")
  try:
    apex_near_idx = int(float(apex_near_idx_b.decode("utf-8") if isinstance(apex_near_idx_b, (bytes, bytearray)) else apex_near_idx_b)) if apex_near_idx_b else ctrl._apex_near_index
  except Exception:
    apex_near_idx = ctrl._apex_near_index
  ctrl._apex_near_index = int(clip(apex_near_idx, 1, 10))

  apex_boost_dist_b = P.get("VisionTurnSpeedControlApexBoostDistance")
  try:
    apex_boost_dist = float(apex_boost_dist_b.decode("utf-8") if isinstance(apex_boost_dist_b, (bytes, bytearray)) else apex_boost_dist_b) if apex_boost_dist_b else ctrl._apex_boost_distance
  except Exception:
    apex_boost_dist = ctrl._apex_boost_distance
  ctrl._apex_boost_distance = clip(apex_boost_dist, 0.0, 300.0)

  apex_boost_factor_b = P.get("VisionTurnSpeedControlApexBoostFactor")
  try:
    apex_boost_factor = float(apex_boost_factor_b.decode("utf-8") if isinstance(apex_boost_factor_b, (bytes, bytearray)) else apex_boost_factor_b) if apex_boost_factor_b else ctrl._apex_boost_factor
  except Exception:
    apex_boost_factor = ctrl._apex_boost_factor
  ctrl._apex_boost_factor = clip(apex_boost_factor, 0.0, 0.5)

  apex_boost_min_lat_b = P.get("VisionTurnSpeedControlApexBoostMinLatAccel")
  try:
    apex_boost_min_lat = float(apex_boost_min_lat_b.decode("utf-8") if isinstance(apex_boost_min_lat_b, (bytes, bytearray)) else apex_boost_min_lat_b) if apex_boost_min_lat_b else ctrl._apex_boost_min_lat_accel
  except Exception:
    apex_boost_min_lat = ctrl._apex_boost_min_lat_accel
  ctrl._apex_boost_min_lat_accel = clip(apex_boost_min_lat, 0.0, 5.0)

  apex_boost_center_b = P.get("VisionTurnSpeedControlApexBoostCenter")
  try:
    apex_boost_center = float(apex_boost_center_b.decode("utf-8") if isinstance(apex_boost_center_b, (bytes, bytearray)) else apex_boost_center_b) if apex_boost_center_b else ctrl._apex_boost_center
  except Exception:
    apex_boost_center = ctrl._apex_boost_center
  ctrl._apex_boost_center = clip(apex_boost_center, 0.0, 5.0)

  apex_boost_width_b = P.get("VisionTurnSpeedControlApexBoostWidth")
  try:
    apex_boost_width = float(apex_boost_width_b.decode("utf-8") if isinstance(apex_boost_width_b, (bytes, bytearray)) else apex_boost_width_b) if apex_boost_width_b else ctrl._apex_boost_width
  except Exception:
    apex_boost_width = ctrl._apex_boost_width
  ctrl._apex_boost_width = clip(apex_boost_width, 0.05, 5.0)

  ctrl._boost_safety_curvature_scale = getf(
    "VisionTurnSpeedControlBoostSafetyCurvatureScale",
    getattr(ctrl, "_boost_safety_curvature_scale", 0.7),
    0.5, 1.0,
  )

  # Comfort/adaptive limits
  comfort_decel_b = P.get("VisionTurnSpeedControlComfortDecelLimit")
  try:
    comfort_decel = float(comfort_decel_b.decode("utf-8") if isinstance(comfort_decel_b, (bytes, bytearray)) else comfort_decel_b) if comfort_decel_b else ctrl._comfort_decel_limit
  except Exception:
    comfort_decel = ctrl._comfort_decel_limit
  ctrl._comfort_decel_limit = -abs(clip(abs(comfort_decel), 1.0, 3.0))

  comfort_jerk = getf("VisionTurnSpeedControlComfortJerkLimit", getattr(ctrl, "_comfort_jerk_limit", -2.0))
  ctrl._comfort_jerk_limit = -abs(clip(abs(comfort_jerk), 1.0, 4.0))

  max_adapt_decel = getf("VisionTurnSpeedControlMaxAdaptiveDecel", getattr(ctrl, "_max_adaptive_decel", -6.0))
  ctrl._max_adaptive_decel = -abs(clip(abs(max_adapt_decel), 3.0, 9.0))

  max_adapt_jerk = getf("VisionTurnSpeedControlMaxAdaptiveJerk", getattr(ctrl, "_max_adaptive_jerk", -6.0))
  ctrl._max_adaptive_jerk = -abs(clip(abs(max_adapt_jerk), 3.0, 10.0))

  # Vision occlusion thresholds
  ctrl._occlusion_state.alpha = getf(
    "VisionTurnSpeedControlVisionConfAlpha",
    getattr(ctrl._occlusion_state, "alpha", 0.28),
    0.01, 0.9,
  )
  ctrl._occlusion_state.good_threshold = getf(
    "VisionTurnSpeedControlVisionConfGoodThreshold",
    getattr(ctrl._occlusion_state, "good_threshold", 0.70),
    0.5, 0.99,
  )
  ctrl._occlusion_state.bad_threshold = getf(
    "VisionTurnSpeedControlVisionConfBadThreshold",
    getattr(ctrl._occlusion_state, "bad_threshold", 0.65),
    0.1, ctrl._occlusion_state.good_threshold,
  )
  # Keep occlusion model in sync with controller-level visibility knobs
  ctrl._occlusion_state.vis_horizon_s = ctrl._vis_horizon_s
  ctrl._occlusion_state.gamma_per_m = ctrl._gamma_per_meter
  ctrl._occlusion_state.lat_jerk_cap = ctrl._lat_jerk_cap

  # Global speed scaling and caps (written to controller module globals)
  inc_factor_b = P.get("VisionTurnSpeedControlSpeedIncreaseFactor")
  try:
    inc_factor = float(inc_factor_b.decode("utf-8") if isinstance(inc_factor_b, (bytes, bytearray)) else inc_factor_b) if inc_factor_b else getattr(vtc_mod, "SPEED_INCREASE_FACTOR", 1.0)
  except Exception:
    inc_factor = getattr(vtc_mod, "SPEED_INCREASE_FACTOR", 1.0)
  setattr(vtc_mod, "SPEED_INCREASE_FACTOR", clip(inc_factor, 0.5, 1.5))

  setattr(vtc_mod, "MAX_SPEED_DEFAULT", getf(
    "VisionTurnSpeedControlMaxSpeed",
    getattr(vtc_mod, "MAX_SPEED_DEFAULT", 70.0),
    10.0, 90.0,
  ))

  setattr(vtc_mod, "_MIN_V", getf(
    "VisionTurnSpeedControlMinOperatingSpeed",
    getattr(vtc_mod, "_MIN_V", 2.24),
    0.5, 10.0,
  ))

  # Low-speed bias (mph)
  setattr(vtc_mod, "LOW_SPEED_BIAS_MPH", getf(
    "VisionTurnSpeedControlLowSpeedSpeedBiasMph",
    getattr(vtc_mod, "LOW_SPEED_BIAS_MPH", 0.0),
    -5.0, 5.0,
  ))

  setattr(vtc_mod, "LOW_SPEED_BIAS_END_MPH", getf(
    "VisionTurnSpeedControlLowSpeedBiasEndMph",
    getattr(vtc_mod, "LOW_SPEED_BIAS_END_MPH", 50.0),
    10.0, 80.0,
  ))

  ctrl._low_speed_calibration_high_end_mph = getf(
    "VisionTurnSpeedControlLowSpeedLearnedHighEndMph",
    getattr(ctrl, "_low_speed_calibration_high_end_mph", getattr(vtc_mod, "LOW_SPEED_CALIB_TARGET_END_MPH", 40.0)),
    20.0, 60.0,
  )

  # Physics sigmoid knobs
  phys_base = getf("VisionTurnSpeedControlPhysicsBaseline", getattr(vtc_mod, "PHYSICS_D", 3.144734))
  setattr(vtc_mod, "PHYSICS_D", clip(phys_base, 2.0, 4.0))

  phys_amp = getf("VisionTurnSpeedControlPhysicsAmplitude", getattr(vtc_mod, "PHYSICS_A", -1.1751))
  setattr(vtc_mod, "PHYSICS_A", -abs(clip(abs(phys_amp), 0.2, 2.5)))

  phys_steep = getf("VisionTurnSpeedControlPhysicsSteepness", getattr(vtc_mod, "PHYSICS_B", -2000.0))
  setattr(vtc_mod, "PHYSICS_B", -abs(clip(abs(phys_steep), 100.0, 1e5)))

  phys_center = getf("VisionTurnSpeedControlPhysicsCenter", getattr(vtc_mod, "PHYSICS_C", 0.004778))
  setattr(vtc_mod, "PHYSICS_C", clip(phys_center, 1e-5, 0.1))

  phys_min_lat = getf("VisionTurnSpeedControlPhysicsMinLatAccel", getattr(vtc_mod, "PHYSICS_MIN_LAT_ACCEL", 1.8))
  setattr(vtc_mod, "PHYSICS_MIN_LAT_ACCEL", clip(phys_min_lat, 1.0, 3.0))

  phys_max_lat = getf("VisionTurnSpeedControlPhysicsMaxLatAccel", getattr(vtc_mod, "PHYSICS_MAX_LAT_ACCEL", 3.12))
  setattr(vtc_mod, "PHYSICS_MAX_LAT_ACCEL", clip(phys_max_lat, 2.0, 4.0))

  # Ensure cross-key constraint: min <= max
  try:
    _min = float(getattr(vtc_mod, "PHYSICS_MIN_LAT_ACCEL", 1.8))
    _max = float(getattr(vtc_mod, "PHYSICS_MAX_LAT_ACCEL", 3.12))
    if _min > _max:
      setattr(vtc_mod, "PHYSICS_MIN_LAT_ACCEL", _max)
      setattr(vtc_mod, "PHYSICS_MAX_LAT_ACCEL", _min)
  except Exception:
    pass

  # Expert-only module-level overrides. Disabled expert mode restores defaults.
  setattr(vtc_mod, "FREEWAY_CURV_EPS", expf("VTSC.Expert.FreewayCurvEps", 1e-5, 1e-7, 1e-2, "FREEWAY_CURV_EPS"))
  setattr(vtc_mod, "FREEWAY_MIN_VISIBLE_M", expf("VTSC.Expert.FreewayMinVisibleM", 120.0, 20.0, 400.0, "FREEWAY_MIN_VISIBLE_M"))
  setattr(vtc_mod, "FREEWAY_MIN_CONF", expf("VTSC.Expert.FreewayMinConf", 0.60, 0.05, 0.99, "FREEWAY_MIN_CONF"))
  highway_min_mps = expf("VTSC.Expert.HighwayMinMps", 24.5872, 8.0, 45.0, "HIGHWAY_MIN_MPS")
  setattr(vtc_mod, "HIGHWAY_MIN_MPS", highway_min_mps)
  setattr(vtc_mod, "VTURN_HOLD_MIN_V_MPS", expf("VTSC.Expert.VTurnHoldMinVMps", 27.0, 8.0, 45.0, "VTURN_HOLD_MIN_V_MPS"))
  setattr(vtc_mod, "VTURN_HOLD_DELTA_MPS", expf("VTSC.Expert.VTurnHoldDeltaMps", 1.0, 0.1, 8.0, "VTURN_HOLD_DELTA_MPS"))
  setattr(vtc_mod, "VTURN_HOLD_S", expf("VTSC.Expert.VTurnHoldS", 1.2, 0.1, 5.0, "VTURN_HOLD_S"))
  setattr(vtc_mod, "VTURN_HOLD_S_OCCLUDED", expf("VTSC.Expert.VTurnHoldSOccluded", 0.85, 0.05, 3.0, "VTURN_HOLD_S_OCCLUDED"))
  setattr(vtc_mod, "_ENTERING_PRED_LAT_ACC_TH", expf("VTSC.Expert.EnteringPredLatAccTh", 1.3, 0.2, 5.0, "_ENTERING_PRED_LAT_ACC_TH"))
  setattr(vtc_mod, "VTSC_TRAJECTORY_PHASE_ADVANCE_S", expf("VTSC.Expert.TrajectoryPhaseAdvanceS", 1.0, -1.0, 4.0, "VTSC_TRAJECTORY_PHASE_ADVANCE_S"))
  setattr(vtc_mod, "STEER_CURVATURE_FALLBACK_MODEL_KAPPA_MAX", expf("VTSC.Expert.SteerFallbackModelKappaMax", 0.003, 1e-5, 0.02, "STEER_CURVATURE_FALLBACK_MODEL_KAPPA_MAX"))
  setattr(vtc_mod, "STEER_CURVATURE_FALLBACK_MIN_KAPPA", expf("VTSC.Expert.SteerFallbackMinKappa", 0.003, 1e-5, 0.03, "STEER_CURVATURE_FALLBACK_MIN_KAPPA"))
  setattr(vtc_mod, "STEER_CURVATURE_FALLBACK_MIN_V_MPS", expf("VTSC.Expert.SteerFallbackMinVMps", 13.0, 1.0, 35.0, "STEER_CURVATURE_FALLBACK_MIN_V_MPS"))
  setattr(vtc_mod, "SEVERE_OVERSHOOT_SPEED_SCALE_MIN", expf("VTSC.Expert.SevereOvershootSpeedScaleMin", 0.90, 0.5, 1.0, "SEVERE_OVERSHOOT_SPEED_SCALE_MIN"))
  setattr(vtc_mod, "HIDDEN_TURN_ENABLED", expb("VTSC.Expert.HiddenTurnEnabled", False, "HIDDEN_TURN_ENABLED"))
  setattr(vtc_mod, "HIDDEN_TURN_V_MAX_MPS", expf("VTSC.Expert.HiddenTurnVMaxMps", highway_min_mps, 5.0, 45.0, "HIDDEN_TURN_V_MAX_MPS"))
  setattr(vtc_mod, "HIDDEN_TURN_T_H_S", expf("VTSC.Expert.HiddenTurnTHS", 1.8, 0.1, 5.0, "HIDDEN_TURN_T_H_S"))
  setattr(vtc_mod, "HIDDEN_TURN_DELTA_V_MPS", expf("VTSC.Expert.HiddenTurnDeltaVMps", 2.0, 0.1, 10.0, "HIDDEN_TURN_DELTA_V_MPS"))
  setattr(vtc_mod, "HIDDEN_TURN_MIN_OCC_S", expf("VTSC.Expert.HiddenTurnMinOccS", 0.30, 0.0, 5.0, "HIDDEN_TURN_MIN_OCC_S"))
  setattr(vtc_mod, "HIDDEN_TURN_AVAIL_SCALE", expf("VTSC.Expert.HiddenTurnAvailScale", 0.50, 0.0, 2.0, "HIDDEN_TURN_AVAIL_SCALE"))
  setattr(vtc_mod, "HIDDEN_TURN_PHASE_S", expf("VTSC.Expert.HiddenTurnPhaseS", 2.0, 0.1, 10.0, "HIDDEN_TURN_PHASE_S"))
  setattr(vtc_mod, "HIDDEN_TURN_HEADING_WIN_S", expf("VTSC.Expert.HiddenTurnHeadingWinS", 1.2, 0.1, 5.0, "HIDDEN_TURN_HEADING_WIN_S"))
  setattr(vtc_mod, "HIDDEN_TURN_VIS_HEADING_MAX_RAD", expf("VTSC.Expert.HiddenTurnVisHeadingMaxRad", 0.10472, 0.01, 0.8, "HIDDEN_TURN_VIS_HEADING_MAX_RAD"))
  setattr(vtc_mod, "LOW_SPEED_MARGIN_MAX_V_MPS", expf("VTSC.Expert.LowSpeedMarginMaxVMps", 12.5, 1.0, 30.0, "LOW_SPEED_MARGIN_MAX_V_MPS"))
  setattr(vtc_mod, "LOW_SPEED_MARGIN_CURV_THRESH", expf("VTSC.Expert.LowSpeedMarginCurvThresh", 3.5e-4, 1e-6, 0.01, "LOW_SPEED_MARGIN_CURV_THRESH"))
  setattr(vtc_mod, "OCCL_BYPASS_HEADWAY_V_FLOOR_MPS", expf("VTSC.Expert.OcclBypassHeadwayVFloorMps", 5.0, 0.1, 20.0, "OCCL_BYPASS_HEADWAY_V_FLOOR_MPS"))
  setattr(vtc_mod, "OCCL_BYPASS_LOW_SPEED_V_MPS", expf("VTSC.Expert.OcclBypassLowSpeedVMps", 7.0, 0.1, 25.0, "OCCL_BYPASS_LOW_SPEED_V_MPS"))
  setattr(vtc_mod, "OCCL_BYPASS_LEAD_D_REL_MAX_M", expf("VTSC.Expert.OcclBypassLeadDRelMaxM", 27.0, 5.0, 200.0, "OCCL_BYPASS_LEAD_D_REL_MAX_M"))
  conf_enter_severe = expf("VTSC.Expert.ConfidenceEnterSevere", 0.45, 0.10, 0.90, "CONFIDENCE_ENTER_SEVERE")
  conf_exit_partial = expf("VTSC.Expert.ConfidenceExitToPartial", 0.55, max(conf_enter_severe + 1e-3, 0.11), 0.99, "CONFIDENCE_EXIT_TO_PARTIAL")
  setattr(vtc_mod, "CONFIDENCE_ENTER_SEVERE", conf_enter_severe)
  setattr(vtc_mod, "CONFIDENCE_EXIT_TO_PARTIAL", conf_exit_partial)

  # Occlusion dwell and tuning
  ctrl._occlusion_state.enter_dwell_s = getf(
    "VisionTurnSpeedControlOcclEnterDwellS",
    getattr(ctrl._occlusion_state, "enter_dwell_s", 0.2),
    0.0, 2.0,
  )

  exit_dwell_b = P.get("VisionTurnSpeedControlOcclExitDwellS")
  try:
    exit_dwell = float(exit_dwell_b.decode("utf-8") if isinstance(exit_dwell_b, (bytes, bytearray)) else exit_dwell_b) if exit_dwell_b else getattr(ctrl._occlusion_state, "exit_dwell_s", 0.1)
  except Exception:
    exit_dwell = getattr(ctrl._occlusion_state, "exit_dwell_s", 0.1)
  ctrl._occlusion_state.exit_dwell_s = clip(exit_dwell, 0.0, 2.0)

  ctrl._occlusion_state.gamma_per_m = getf(
    "VisionTurnSpeedControlCurvatureGrowthPerMeter",
    getattr(ctrl._occlusion_state, "gamma_per_m", 2.5e-4),
    0.0, 0.01,
  )

  ctrl._occlusion_state.envelope_horizon_s = getf(
    "VisionTurnSpeedControlEnvelopeHorizonS",
    getattr(ctrl._occlusion_state, "envelope_horizon_s", 1.0),
    0.1, 5.0,
  )

  ctrl._occlusion_state.decay_tau_fast_s = getf(
    "VisionTurnSpeedControlOcclusionDecayTauFastS",
    getattr(ctrl._occlusion_state, "decay_tau_fast_s", 1.2),
    0.1, 5.0,
  )

  ctrl._occlusion_state.decay_tau_slow_s = getf(
    "VisionTurnSpeedControlOcclusionDecayTauSlowS",
    getattr(ctrl._occlusion_state, "decay_tau_slow_s", 2.0),
    0.1, 10.0,
  )

  ctrl._occlusion_state.min_frac = getf(
    "VisionTurnSpeedControlOcclusionMinFrac",
    getattr(ctrl._occlusion_state, "min_frac", 0.2),
    0.05, 0.9,
  )

  # Fast reacquisition window tuning
  ctrl._fast_reacq_alpha = getf(
    "VisionTurnSpeedControlFastReacqAlpha",
    getattr(ctrl, "_fast_reacq_alpha", 0.85),
    0.3, 0.99,
  )
  ctrl._fast_reacq_window_s = getf(
    "VisionTurnSpeedControlFastReacqWindowS",
    getattr(ctrl, "_fast_reacq_window_s", 0.9),
    0.1, 3.0,
  )

  # Vision-floor and dropout discrimination knobs
  ctrl._vision_floor_ttl_s = getf(
    "VisionTurnSpeedControlVisionFloorTtlS",
    getattr(ctrl, "_vision_floor_ttl_s", 3.0),
    0.0, 10.0,
  )
  ctrl._vision_floor_mult = getf(
    "VisionTurnSpeedControlVisionFloorMult",
    getattr(ctrl, "_vision_floor_mult", 1.00),
    0.5, 1.5,
  )
  ctrl._dropout_grace_s = getf(
    "VisionTurnSpeedControlDropoutGraceS",
    getattr(ctrl, "_dropout_grace_s", 0.40),
    0.0, 2.0,
  )

  if hasattr(ctrl, "_sync_low_speed_calibration_param"):
    ctrl._sync_low_speed_calibration_param()

  # Double-cap guard and fov_exit relax tunables
  try:
    ctrl._double_cap_eps_mps = float(getf("VTSC.DoubleCapEpsMps", getattr(ctrl, "_double_cap_eps_mps", 0.30)))
  except Exception:
    ctrl._double_cap_eps_mps = getattr(ctrl, "_double_cap_eps_mps", 0.30)
  try:
    ctrl._occl_conf_floor = float(getf("VTSC.OcclConfFloor", getattr(ctrl, "_occl_conf_floor", 0.05)))
  except Exception:
    ctrl._occl_conf_floor = getattr(ctrl, "_occl_conf_floor", 0.05)
  try:
    ctrl._fov_exit_relax_s = float(getf("VTSC.FovExitRelaxS", getattr(ctrl, "_fov_exit_relax_s", 0.60)))
  except Exception:
    ctrl._fov_exit_relax_s = getattr(ctrl, "_fov_exit_relax_s", 0.60)
  try:
    ctrl._occl_vmin_nudge_mps = float(getf("VTSC.OcclVminNudgeMps", getattr(ctrl, "_occl_vmin_nudge_mps", 0.50)))
  except Exception:
    ctrl._occl_vmin_nudge_mps = getattr(ctrl, "_occl_vmin_nudge_mps", 0.50)

  ctrl._last_params_update = float(tm)
