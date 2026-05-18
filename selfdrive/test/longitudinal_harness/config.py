from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.hyundai.hyundaicanfd import CanBus
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.radar_interface import RADAR_START_ADDR
from opendbc.car.hyundai.values import CAR, HyundaiFlags
from opendbc.sunnypilot.car.hyundai.longitudinal.helpers import LongitudinalTuningType
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP

FRIENDLY_PARAM_NAMES = {
  "obstacle_cost": "Longitudinal.LiveTune.ObstacleCost",
  "accel_change_cost": "Longitudinal.LiveTune.AccelChangeCost",
  "accel_cost": "Longitudinal.LiveTune.AccelCost",
  "lead_preview_strength": "Longitudinal.LiveTune.LeadPreviewStrength",
  "lead_preview_gap_min_m": "Longitudinal.LiveTune.LeadPreviewGapMinM",
  "lead_preview_max_buffer_m": "Longitudinal.LiveTune.LeadPreviewMaxBufferM",
  "lead_acquire_window_s": "Longitudinal.LiveTune.LeadAcquireWindowS",
  "gap_reclaim_strength": "Longitudinal.LiveTune.GapReclaimStrength",
  "gap_reclaim_gap_min_m": "Longitudinal.LiveTune.GapReclaimGapMinM",
  "gap_reclaim_max_accel": "Longitudinal.LiveTune.GapReclaimMaxAccel",
  "lead_keepup_strength": "Longitudinal.LiveTune.LeadKeepUpStrength",
  "lead_keepup_gap_min_m": "Longitudinal.LiveTune.LeadKeepUpGapMinM",
  "lead_keepup_max_accel": "Longitudinal.LiveTune.LeadKeepUpMaxAccel",
  "lead_slowdown_strength": "Longitudinal.LiveTune.LeadSlowdownStrength",
  "lead_slowdown_max_decel": "Longitudinal.LiveTune.LeadSlowdownMaxDecel",
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
  "lead_prob_enter": "Longitudinal.LiveTune.LeadProbEnter",
  "lead_prob_exit": "Longitudinal.LiveTune.LeadProbExit",
  "lead_source_acquire_frames": "Longitudinal.LiveTune.LeadSourceAcquireFrames",
  "lead_source_release_frames": "Longitudinal.LiveTune.LeadSourceReleaseFrames",
  "phantom_lead_hold_s": "Longitudinal.LiveTune.PhantomLeadHoldS",
  "phantom_lead_stable_frames": "Longitudinal.LiveTune.PhantomLeadStableFrames",
  "flutter_detect_transitions": "Longitudinal.LiveTune.FlutterDetectTransitions",
  "flutter_detect_window_s": "Longitudinal.LiveTune.FlutterDetectWindowS",
  "flutter_clamp_jerk_mps3": "Longitudinal.LiveTune.FlutterClampJerkMps3",
  "flutter_clamp_bypass_decel_mps2": "Longitudinal.LiveTune.FlutterClampBypassDecelMps2",
  "model_lead_filter_tau_s": "Longitudinal.LiveTune.ModelLeadFilterTauS",
  "model_lead_filter_open_slew_max_mps": "Longitudinal.LiveTune.ModelLeadFilterOpenSlewMaxMps",
  "model_lead_filter_safe_ttc_s": "Longitudinal.LiveTune.ModelLeadFilterSafeTtcS",
  "model_lead_filter_assoc_drel_m": "Longitudinal.LiveTune.ModelLeadFilterAssocDRelM",
  "model_lead_filter_vrel_tau_s": "Longitudinal.LiveTune.ModelLeadFilterVRelTauS",
  "model_lead_filter_fast_vrel_tau_s": "Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS",
  "use_kalman_drel_filter": "Longitudinal.LiveTune.UseKalmanDRelFilter",
  "kalman_drel_q": "Longitudinal.LiveTune.KalmanDRelQ",
  "kalman_drel_r": "Longitudinal.LiveTune.KalmanDRelR",
  "kalman_drel_gain_max": "Longitudinal.LiveTune.KalmanDRelGainMax",
  "kalman_drel_deadband_m": "Longitudinal.LiveTune.KalmanDRelDeadbandM",
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
  "Longitudinal.LiveTune.ObstacleCost": "2.0",
  "Longitudinal.LiveTune.AccelChangeCost": "400.0",
  "Longitudinal.LiveTune.AccelCost": "1.0",
  "Longitudinal.LiveTune.LeadPreviewStrength": "1.5",
  "Longitudinal.LiveTune.LeadPreviewGapMinM": "1.0",
  "Longitudinal.LiveTune.LeadPreviewMaxBufferM": "10.0",
  "Longitudinal.LiveTune.LeadAcquireWindowS": "1.5",
  "Longitudinal.LiveTune.GapReclaimStrength": "0.55",
  "Longitudinal.LiveTune.GapReclaimGapMinM": "3.0",
  "Longitudinal.LiveTune.GapReclaimMaxAccel": "0.12",
  "Longitudinal.LiveTune.LeadKeepUpStrength": "1.15",
  "Longitudinal.LiveTune.LeadKeepUpGapMinM": "0.10",
  "Longitudinal.LiveTune.LeadKeepUpMaxAccel": "0.095",
  "Longitudinal.LiveTune.LeadSlowdownStrength": "0.25",
  "Longitudinal.LiveTune.LeadSlowdownMaxDecel": "4.0",
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
  "Longitudinal.LiveTune.LeadProbEnter": "0.60",
  "Longitudinal.LiveTune.LeadProbExit": "0.25",
  "Longitudinal.LiveTune.LeadSourceAcquireFrames": "1.0",
  "Longitudinal.LiveTune.LeadSourceReleaseFrames": "20.0",
  "Longitudinal.LiveTune.PhantomLeadHoldS": "0.80",
  "Longitudinal.LiveTune.PhantomLeadStableFrames": "3.0",
  "Longitudinal.LiveTune.FlutterDetectTransitions": "2.0",
  "Longitudinal.LiveTune.FlutterDetectWindowS": "1.0",
  "Longitudinal.LiveTune.FlutterClampJerkMps3": "0.12",
  "Longitudinal.LiveTune.FlutterClampBypassDecelMps2": "1.5",
  "Longitudinal.LiveTune.ModelLeadFilterTauS": "2.80",
  "Longitudinal.LiveTune.ModelLeadFilterOpenSlewMaxMps": "1.20",
  "Longitudinal.LiveTune.ModelLeadFilterSafeTtcS": "4.00",
  "Longitudinal.LiveTune.ModelLeadFilterAssocDRelM": "12.0",
  "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.40",
  "Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS": "0.16",
  "HyundaiLongitudinalTuning": str(LongitudinalTuningType.OFF),
  "LongTuningCustomToggle": "0",
  "LongTuningAccelMin": "-5.5",
  "LongTuningAccelMax": "2.0",
  "LongTuningVEgoStopping": "0.25",
  "LongTuningStoppingDecelRate": "0.40",
  "LongTuningMinUpperJerk": "0.5",
  "LongTuningMinLowerJerk": "0.5",
  "LongTuningJerkLimits": "4.0",
  "VibePersonalityEnabled": "0",
  "VibeFollowPersonalityEnabled": "0",
  "VibeAccelPersonalityEnabled": "0",
  "VisionTurnSpeedControl": "0",
  "SpeedLimitControl": "0",
  "RTIEnabled": "0",
  "WeatherAwareControlEnabled": "0",
}


@dataclass(frozen=True)
class NoiseProfile:
  name: str
  drel_frac: float
  drel_floor_m: float
  vrel_factor: float
  vrel_floor_mps: float


@dataclass(frozen=True)
class NoiseSeeds:
  drel: int
  vrel: int
  aego: int
  vego: int

  @classmethod
  def from_base(cls, base_seed: int) -> NoiseSeeds:
    return cls(
      drel=int(base_seed),
      vrel=int(base_seed) + 1,
      aego=int(base_seed) + 2,
      vego=int(base_seed) + 3,
    )

  def offset(self, delta: int) -> NoiseSeeds:
    return NoiseSeeds(
      drel=self.drel + delta,
      vrel=self.vrel + delta,
      aego=self.aego + delta,
      vego=self.vego + delta,
    )

  def with_overrides(self,
                     *,
                     drel: int | None = None,
                     vrel: int | None = None,
                     aego: int | None = None,
                     vego: int | None = None) -> NoiseSeeds:
    return NoiseSeeds(
      drel=self.drel if drel is None else drel,
      vrel=self.vrel if vrel is None else vrel,
      aego=self.aego if aego is None else aego,
      vego=self.vego if vego is None else vego,
    )

  def as_dict(self) -> dict[str, int]:
    return asdict(self)


NOISE_PROFILES: dict[str, NoiseProfile] = {
  "off": NoiseProfile("off", 0.0, 0.0, 0.0, 0.0),
  "realistic": NoiseProfile("realistic", 0.03, 0.75, 0.05, 0.05),
  "stress": NoiseProfile("stress", 0.07, 1.5, 0.10, 0.2),
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
  metadata: dict[str, Any] = field(default_factory=dict)

  def describe(self) -> dict[str, Any]:
    return {
      "topology": self.topology,
      "requestedControllerMode": self.requested_controller_mode,
      "resolvedControllerMode": self.resolved_controller_mode,
      "hyundaiTuningMode": self.hyundai_tuning_mode,
      "tuneSource": self.tune_source,
      "fidelitySource": self.fidelity_source,
      "openpilotLongitudinalControl": bool(self.cp.openpilotLongitudinalControl),
      "pcmCruise": bool(self.cp.pcmCruise),
      "radarUnavailable": bool(self.cp.radarUnavailable),
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
                               topology: str = "lfa",
                               controller_mode: str = "auto",
                               tune_source: str = "defaults",
                               param_overrides: dict[str, Any] | None = None,
                               hyundai_tuning_mode: int | None = None,
                               snapshot_vehicle: dict[str, Any] | None = None,
                               snapshot_params: dict[str, Any] | None = None,
                               plant_overrides: dict[str, Any] | None = None) -> ResolvedVehicleConfig:
  if controller_mode not in ("auto", "passthrough", "shaped"):
    raise ValueError(f"unsupported controller_mode '{controller_mode}'")

  fingerprint, car_fw = build_synthetic_ev6_inputs(topology)
  params = dict(DEFAULT_PARAM_VALUES)
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

  if controller_mode == "shaped" and int(params.get("HyundaiLongitudinalTuning", "0")) == LongitudinalTuningType.OFF:
    params["HyundaiLongitudinalTuning"] = str(LongitudinalTuningType.DYNAMIC)

  if controller_mode == "passthrough":
    CP.radarUnavailable = True
  elif controller_mode == "shaped":
    CP.radarUnavailable = False
  elif controller_mode == "auto" and snapshot_vehicle and "radarUnavailable" in snapshot_vehicle:
    CP.radarUnavailable = bool(snapshot_vehicle["radarUnavailable"])

  _apply_hyundai_tuning(CP, CP_SP, params)

  resolved_controller_mode = (
    "shaped"
    if (int(params.get("HyundaiLongitudinalTuning", "0")) != LongitudinalTuningType.OFF and not CP.radarUnavailable)
    else "passthrough"
  )
  if controller_mode == "shaped" and resolved_controller_mode != "shaped":
    raise ValueError("requested shaped EV6 controller mode, but Hyundai runtime shaping is not active")
  if controller_mode == "passthrough" and resolved_controller_mode != "passthrough":
    raise ValueError("requested passthrough EV6 controller mode, but config resolved to shaped mode")

  plant_config = VehiclePlantConfig(command_delay_s=float(CP.longitudinalActuatorDelay))
  if snapshot_vehicle and snapshot_vehicle.get("plantConfig"):
    plant_config = VehiclePlantConfig(**{**plant_config.as_dict(), **snapshot_vehicle["plantConfig"]})
  if plant_overrides:
    plant_config = VehiclePlantConfig(**{**plant_config.as_dict(), **plant_overrides})

  metadata = {}
  if snapshot_vehicle:
    metadata.update({k: v for k, v in snapshot_vehicle.items() if k not in {"plantConfig"}})

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
    metadata=metadata,
  )
