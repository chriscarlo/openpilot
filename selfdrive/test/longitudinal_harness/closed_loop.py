from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from cereal import log, messaging
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from opendbc.car.structs import CarControlSP
from openpilot.common.gps import get_gps_location_service
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_params import update_vtsc_params

from .config import NOISE_PROFILES, NoiseProfile, NoiseSeeds, ResolvedVehicleConfig
from .inputs import LeadDirective, SnapshotBundle, StepInput
from .metrics import summarize_trace


class HarnessParams:
  def __init__(self, params: dict[str, str]):
    self.params = dict(params)

  def get(self, key: str, block: bool = False, encoding: str | None = None, return_default: bool = False):
    return self.params.get(key)

  def get_bool(self, key: str) -> bool:
    raw = self.params.get(key)
    if raw is None:
      return False
    raw_text = str(raw).strip().lower()
    if raw_text in ("1", "true", "t", "yes", "y", "on"):
      return True
    if raw_text in ("0", "false", "f", "no", "n", "off"):
      return False
    return False

  def put(self, key: str, value: Any) -> None:
    self.params[key] = str(value)

  def put_bool(self, key: str, value: bool) -> None:
    self.params[key] = "1" if value else "0"

  def put_nonblocking(self, key: str, value: Any) -> None:
    self.put(key, value)

  def remove(self, key: str) -> None:
    self.params.pop(key, None)


def _bind_planner_params(planner: LongitudinalPlanner, harness_params: HarnessParams) -> None:
  planner.params = harness_params
  planner.mpc._live_tune_params = harness_params
  planner.mpc._refresh_live_tune(now=0.0, force=True)
  planner.mpc.lead_role_classifier._params = harness_params
  planner.mpc.lead_role_classifier._last_refresh_t = 0.0
  planner.mpc.vibe_controller.params = harness_params
  planner.mpc.vibe_controller._last_param_refresh_t = float("-inf")
  planner.mpc.vibe_controller._update_tuning_profiles(force=True)
  planner.mpc.vibe_controller._update_from_params()

  planner.dec._params = harness_params
  planner.dec._frame = 0
  planner.dec._read_params()

  planner.vibe_controller.params = harness_params
  planner.vibe_controller._last_param_refresh_t = float("-inf")
  planner.vibe_controller._update_tuning_profiles(force=True)
  planner.vibe_controller._update_from_params()

  planner.v_tsc._params = harness_params
  planner.v_tsc._mem_params = harness_params
  update_vtsc_params(planner.v_tsc, force=True)
  planner.v_tsc._sync_low_speed_calibration_param(force=True)

  planner.slc._params = harness_params
  planner.slc._resolver._gps_location_service = get_gps_location_service(harness_params)
  planner.slc._current_time = 1e9
  planner.slc._last_params_update = -1e9
  planner.slc._update_params()

  planner.rti.params = harness_params
  planner.rti._load_user_params()
  planner.rti._reset_state()

  planner.weather.params = harness_params
  planner.weather._last_param_read = -1e9
  planner.weather._load_user_params()
  planner.weather._reset()


class SubMasterStub(dict):
  def __init__(self, values: dict[str, Any], *, valid_overrides: dict[str, bool] | None = None):
    super().__init__(values)
    self.valid = {key: False for key in values}
    self.alive = {key: True for key in values}
    self.updated = {key: True for key in values}
    self.logMonoTime = {key: 0 for key in values}
    self.frame = 0
    if valid_overrides:
      self.valid.update(valid_overrides)

  def all_checks(self, service_list: list[str] | None = None) -> bool:
    service_list = list(self.keys()) if service_list is None else service_list
    return all(self.valid.get(service, False) for service in service_list)


@dataclass
class VehiclePlantState:
  time_s: float
  true_distance_m: float
  true_speed_mps: float
  true_accel_mps2: float
  measured_speed_mps: float
  measured_accel_mps2: float


@dataclass
class LeadTrackState:
  active: bool = False
  distance_m: float = 0.0
  speed_mps: float = 0.0
  model_prob: float = 0.0
  acquisition_age_s: float = 0.0
  measured_d_rel_m: float | None = None
  a_lead_k_mps2: float = 0.0
  last_speed_for_accel: float = 0.0
  just_acquired: bool = False


@dataclass
class SimulationResult:
  vehicle: dict[str, Any]
  summary: dict[str, Any]
  trace: list[dict[str, Any]]


def _to_builtin(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(k): _to_builtin(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_to_builtin(v) for v in value]
  if isinstance(value, np.generic):
    return value.item()
  return value


def _long_control_state_name(state: int) -> str:
  if state == LongCtrlState.off:
    return "off"
  if state == LongCtrlState.stopping:
    return "stopping"
  if state == LongCtrlState.starting:
    return "starting"
  if state == LongCtrlState.pid:
    return "pid"
  return f"unknown:{state}"


class DelayedVehiclePlant:
  def __init__(self, *,
               dt_s: float,
               config,
               initial_speed_mps: float,
               initial_accel_mps2: float = 0.0,
               aego_seed: int = 0,
               vego_seed: int = 1):
    self.dt_s = dt_s
    self.config = config
    self.true_distance_m = 0.0
    self.true_speed_mps = initial_speed_mps
    self.true_accel_mps2 = initial_accel_mps2
    self.command_history = [initial_accel_mps2]
    self.accel_history = [initial_accel_mps2]
    self.speed_history = [initial_speed_mps]
    self.aego_rng = np.random.default_rng(aego_seed)
    self.vego_rng = np.random.default_rng(vego_seed)

  def step(self, commanded_accel_mps2: float) -> tuple[float, float, float, float]:
    self.command_history.append(commanded_accel_mps2)
    delayed_command = self._lookup(self.command_history, self.config.command_delay_s)

    target_tau = self.config.accel_rise_tau_s
    if delayed_command < self.true_accel_mps2:
      target_tau = self.config.regen_tau_s if delayed_command >= self.config.regen_threshold_mps2 else self.config.brake_tau_s
    alpha = min(1.0, self.dt_s / max(target_tau, self.dt_s))
    self.true_accel_mps2 += alpha * (delayed_command - self.true_accel_mps2)
    self.true_speed_mps = max(0.0, self.true_speed_mps + self.true_accel_mps2 * self.dt_s)
    self.true_distance_m += self.true_speed_mps * self.dt_s

    self.accel_history.append(self.true_accel_mps2)
    self.speed_history.append(self.true_speed_mps)
    measured_accel = self._lookup(self.accel_history, self.config.aego_measure_delay_s)
    measured_speed = self._lookup(self.speed_history, self.config.aego_measure_delay_s)
    if self.config.aego_measure_noise_std > 0.0:
      measured_accel += float(self.aego_rng.normal(0.0, self.config.aego_measure_noise_std))
    if self.config.vego_measure_noise_std > 0.0:
      measured_speed += float(self.vego_rng.normal(0.0, self.config.vego_measure_noise_std))
    return delayed_command, self.true_accel_mps2, measured_accel, max(0.0, measured_speed)

  def _lookup(self, history: list[float], delay_s: float) -> float:
    steps = max(0.0, delay_s / self.dt_s)
    low = int(np.floor(steps))
    frac = steps - low
    idx_a = max(0, len(history) - 1 - low)
    idx_b = max(0, idx_a - 1)
    if frac <= 1e-6:
      return history[idx_a]
    return float((1.0 - frac) * history[idx_a] + frac * history[idx_b])


def run_harness(*,
                vehicle_config: ResolvedVehicleConfig,
                scenario_name: str,
                steps: list[StepInput],
                initial_speed_mps: float,
                initial_accel_mps2: float = 0.0,
                noise_profile: str = "realistic",
                seed: int = 42,
                noise_seeds: NoiseSeeds | None = None) -> SimulationResult:
  planner_dt_s = DT_MDL
  control_dt_s = DT_CTRL
  profile = NOISE_PROFILES[noise_profile]
  control_ticks_per_step = max(1, int(round(planner_dt_s / control_dt_s)))
  seeds = noise_seeds or NoiseSeeds.from_base(seed)
  drel_rng = np.random.default_rng(seeds.drel)
  vrel_rng = np.random.default_rng(seeds.vrel)

  planner = LongitudinalPlanner(vehicle_config.cp, init_v=initial_speed_mps, init_a=initial_accel_mps2)
  sim_time_s = [0.0]
  planner.mpc._time_fn = lambda: sim_time_s[0]
  harness_params = HarnessParams(vehicle_config.params)
  _bind_planner_params(planner, harness_params)

  long_control = LongControl(vehicle_config.cp)
  hyundai_controller = None
  if vehicle_config.cp.brand == "hyundai":
    from opendbc.sunnypilot.car.hyundai.longitudinal.controller import LongitudinalController
    hyundai_controller = LongitudinalController(vehicle_config.cp, vehicle_config.cp_sp)

  plant = DelayedVehiclePlant(
    dt_s=control_dt_s,
    config=vehicle_config.plant_config,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    aego_seed=seeds.aego,
    vego_seed=seeds.vego,
  )
  state = VehiclePlantState(
    time_s=0.0,
    true_distance_m=0.0,
    true_speed_mps=initial_speed_mps,
    true_accel_mps2=initial_accel_mps2,
    measured_speed_mps=initial_speed_mps,
    measured_accel_mps2=initial_accel_mps2,
  )
  lead_tracks = {"leadOne": LeadTrackState(), "leadTwo": LeadTrackState()}
  trace: list[dict[str, Any]] = []
  planner_accel = float(initial_accel_mps2)
  planner_source = "cruise"
  planner_should_stop = False

  for step in steps:
    sim_time_s[0] = float(step.t_s)
    radar_state, _ = _build_radar_state(lead_tracks, step, state, profile, planner_dt_s, drel_rng, vrel_rng)
    sm = _build_submaster(step, state, radar_state, long_control.long_control_state, bool(vehicle_config.cp.openpilotLongitudinalControl))
    planner.update(sm)
    planner_accel = float(planner.output_a_target)
    planner_source = str(getattr(planner.mpc, "source", ""))
    planner_should_stop = bool(planner.output_should_stop)

    for control_idx in range(control_ticks_per_step):
      long_active = bool(vehicle_config.cp.openpilotLongitudinalControl)
      accel_limits = CarInterface.get_pid_accel_limits(vehicle_config.cp, state.measured_speed_mps, step.cruise_speed_mps)
      cs_loc = SimpleNamespace(
        vEgo=state.measured_speed_mps,
        aEgo=state.measured_accel_mps2,
        brakePressed=False,
        cruiseState=SimpleNamespace(standstill=state.true_speed_mps < 0.01),
      )
      longcontrol_accel = float(long_control.update(long_active, cs_loc, planner_accel, planner_should_stop, accel_limits))

      controller_accel = longcontrol_accel
      controller_jerk_upper = 0.0
      controller_jerk_lower = 0.0
      if hyundai_controller is not None:
        CC = SimpleNamespace(
          actuators=SimpleNamespace(accel=longcontrol_accel, longControlState=long_control.long_control_state),
          longActive=long_active,
          enabled=True,
          hudControl=SimpleNamespace(visualAlert=None),
        )
        CC_SP = SimpleNamespace(params=vehicle_config.cc_sp_params, flags=vehicle_config.cp_sp.flags)
        CC_SP.leadOne = radar_state.leadOne
        CC_SP.leadTwo = radar_state.leadTwo
        CS = SimpleNamespace(
          out=SimpleNamespace(vEgo=state.measured_speed_mps, aEgo=state.measured_accel_mps2),
          aBasis=state.measured_accel_mps2,
        )
        hyundai_controller.update(CC, CC_SP, CS)
        controller_accel = float(hyundai_controller.actual_accel)
        controller_jerk_upper = float(hyundai_controller.jerk_upper)
        controller_jerk_lower = float(hyundai_controller.jerk_lower)

      delayed_command, realized_accel, measured_accel, measured_speed = plant.step(controller_accel)
      tick_t_s = float(step.t_s + (control_idx * control_dt_s))
      state = VehiclePlantState(
        time_s=tick_t_s,
        true_distance_m=plant.true_distance_m,
        true_speed_mps=plant.true_speed_mps,
        true_accel_mps2=realized_accel,
        measured_speed_mps=measured_speed,
        measured_accel_mps2=measured_accel,
      )
      _advance_lead_tracks(lead_tracks, control_dt_s)

      lead_meta = _current_lead_meta(lead_tracks, state)
      active_true_gaps = [meta["true_d_rel_m"] for meta in lead_meta.values() if meta["true_d_rel_m"] is not None]
      active_leads = [(name, meta) for name, meta in lead_meta.items() if meta["true_d_rel_m"] is not None]
      active_lead_speed = None
      if active_leads:
        active_lead_speed = min(active_leads, key=lambda item: item[1]["true_d_rel_m"])[1]["v_lead_mps"]
      control_slot_name = {"lead0": "leadOne", "lead1": "leadTwo"}.get(planner_source)
      control_meta = lead_meta.get(control_slot_name or "", {})
      control_lead_speed = control_meta.get("v_lead_mps")
      control_true_gap = control_meta.get("true_d_rel_m")

      trace.append({
        "t_s": tick_t_s,
        "event": step.event if control_idx == 0 else None,
        "note": step.note,
        "planner_source": planner_source,
        "planner_accel_mps2": planner_accel,
        "planner_should_stop": planner_should_stop,
        "planner_gap_reclaim_floor_mps2": float(getattr(planner.mpc, "gap_reclaim_accel_floor", 0.0) or 0.0),
        "planner_cutin_settle_floor_mps2": float(getattr(planner.mpc, "cutin_settle_accel_floor", 0.0) or 0.0),
        "planner_lead_present_cruise_cap_mps2": float(getattr(planner.mpc, "lead_present_cruise_accel_cap", 0.0) or 0.0),
        "planner_accel_clip_min_mps2": float(getattr(planner, "_planner_output_accel_limits", (0.0, 0.0))[0]),
        "planner_accel_clip_max_mps2": float(getattr(planner, "_planner_output_accel_limits", (0.0, 0.0))[1]),
        "longcontrol_accel_mps2": longcontrol_accel,
        "longcontrol_state": int(long_control.long_control_state),
        "longcontrol_state_name": _long_control_state_name(long_control.long_control_state),
        "controller_accel_mps2": controller_accel,
        "controller_jerk_upper_mps3": controller_jerk_upper,
        "controller_jerk_lower_mps3": controller_jerk_lower,
        "delayed_command_mps2": delayed_command,
        "realized_accel_mps2": realized_accel,
        "measured_accel_mps2": measured_accel,
        "v_ego_true_mps": state.true_speed_mps,
        "v_ego_measured_mps": state.measured_speed_mps,
        "has_any_lead": bool(active_leads),
        "active_lead_speed_mps": active_lead_speed,
        "has_control_lead": control_lead_speed is not None,
        "control_lead_speed_mps": control_lead_speed,
        "control_true_gap_m": control_true_gap,
        "true_min_gap_m": min(active_true_gaps) if active_true_gaps else None,
        "lead_one_status": lead_meta["leadOne"]["status"],
        "lead_two_status": lead_meta["leadTwo"]["status"],
        "lead_one_true_d_rel_m": lead_meta["leadOne"]["true_d_rel_m"],
        "lead_two_true_d_rel_m": lead_meta["leadTwo"]["true_d_rel_m"],
        "lead_one_measured_d_rel_m": lead_meta["leadOne"]["measured_d_rel_m"],
        "lead_two_measured_d_rel_m": lead_meta["leadTwo"]["measured_d_rel_m"],
        "lead_one_a_lead_k_mps2": lead_meta["leadOne"]["a_lead_k_mps2"],
        "lead_two_a_lead_k_mps2": lead_meta["leadTwo"]["a_lead_k_mps2"],
        "lead_one_model_prob": lead_meta["leadOne"]["model_prob"],
        "lead_two_model_prob": lead_meta["leadTwo"]["model_prob"],
        "mpc_acc_source_debug": _to_builtin(getattr(planner.mpc, "acc_source_debug", {})),
        "mpc_cutin_settle_debug": _to_builtin(getattr(planner.mpc, "cutin_settle_debug", {})),
        "mpc_lead_preview_debug": _to_builtin(getattr(planner.mpc, "lead_approach_preview_debug", {})),
        "mpc_adjacent_awareness_preview_debug": _to_builtin(getattr(planner.mpc, "adjacent_awareness_preview_debug", {})),
        "mpc_hyundai_virtual_lead_debug": _to_builtin(getattr(planner.mpc, "hyundai_virtual_lead_debug", {})),
      })

  vehicle_description = vehicle_config.describe()
  vehicle_description["noiseProfile"] = profile.name
  vehicle_description["noiseSeeds"] = seeds.as_dict()
  summary = summarize_trace(trace, vehicle=vehicle_description, scenario_name=scenario_name, noise_profile=profile.name)
  return SimulationResult(vehicle=vehicle_description, summary=summary, trace=trace)


def _build_submaster(step: StepInput, state: VehiclePlantState, radar_state, long_control_state, long_active: bool) -> SubMasterStub:
  radar = messaging.new_message("radarState")
  radar.radarState = radar_state
  control = messaging.new_message("controlsState")
  control.controlsState.longControlState = long_control_state
  control.controlsState.forceDecel = step.force_decel
  ss = messaging.new_message("selfdriveState")
  ss.selfdriveState.enabled = True
  ss.selfdriveState.experimentalMode = step.experimental_mode
  ss.selfdriveState.personality = int(log.LongitudinalPersonality.standard)
  car_state = messaging.new_message("carState")
  car_state.carState.vEgo = float(state.measured_speed_mps)
  car_state.carState.aEgo = float(state.measured_accel_mps2)
  car_state.carState.standstill = bool(state.true_speed_mps < 0.01)
  car_state.carState.vCruise = float(step.cruise_speed_mps * 3.6)
  car_control = messaging.new_message("carControl")
  car_control.carControl.orientationNED = [0.0, float(step.pitch_rad), 0.0]
  car_control.carControl.longActive = bool(long_active)

  model = messaging.new_message("modelV2")
  position = log.XYZTData.new_message()
  velocity = log.XYZTData.new_message()
  acceleration = log.XYZTData.new_message()
  position.x = [float(x) for x in (state.measured_speed_mps + 0.5) * np.array(ModelConstants.T_IDXS)]
  velocity.x = [float(x) for x in (state.measured_speed_mps + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
  velocity.x[0] = float(state.measured_speed_mps)
  acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
  model.modelV2.position = position
  model.modelV2.velocity = velocity
  model.modelV2.acceleration = acceleration
  model.modelV2.action.desiredAcceleration = float(state.measured_accel_mps2 + 0.1)
  model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0 for _ in range(6)]

  live_parameters = messaging.new_message("liveParameters")
  car_state_sp = messaging.new_message("carStateSP")
  live_map_data_sp = messaging.new_message("liveMapDataSP")
  gps_location = messaging.new_message("gpsLocation")
  selfdrive_state_sp = messaging.new_message("selfdriveStateSP")
  rti_state_sp = messaging.new_message("rtiStateSP")

  return SubMasterStub(
    {
      "radarState": radar.radarState,
      "carState": car_state.carState,
      "carControl": car_control.carControl,
      "controlsState": control.controlsState,
      "selfdriveState": ss.selfdriveState,
      "liveParameters": live_parameters.liveParameters,
      "modelV2": model.modelV2,
      "carStateSP": car_state_sp.carStateSP,
      "liveMapDataSP": live_map_data_sp.liveMapDataSP,
      "gpsLocation": gps_location.gpsLocation,
      "selfdriveStateSP": selfdrive_state_sp.selfdriveStateSP,
      "rtiStateSP": rti_state_sp.rtiStateSP,
    },
    valid_overrides={
      "radarState": True,
      "carState": True,
      "carControl": True,
      "controlsState": True,
      "selfdriveState": True,
      "modelV2": True,
    },
  )


def _build_radar_state(lead_tracks: dict[str, LeadTrackState],
                       step: StepInput,
                       state: VehiclePlantState,
                       noise_profile: NoiseProfile,
                       dt_s: float,
                       drel_rng: np.random.Generator,
                       vrel_rng: np.random.Generator) -> tuple[log.RadarState, dict[str, dict[str, Any]]]:
  radar_state = log.RadarState.new_message()
  lead_meta = {}
  for slot_name, directive in (("leadOne", step.lead_one), ("leadTwo", step.lead_two)):
    lead_message, lead_meta[slot_name] = _build_lead(
      slot_name,
      lead_tracks[slot_name],
      directive,
      state,
      noise_profile,
      dt_s,
      drel_rng,
      vrel_rng,
    )
    setattr(radar_state, slot_name, lead_message)
  return radar_state, lead_meta


def _build_lead(slot_name: str,
                track: LeadTrackState,
                directive: LeadDirective,
                state: VehiclePlantState,
                noise_profile: NoiseProfile,
                dt_s: float,
                drel_rng: np.random.Generator,
                vrel_rng: np.random.Generator) -> tuple[log.RadarState.LeadData, dict[str, Any]]:
  lead = log.RadarState.LeadData.new_message()
  true_d_rel = None
  measured_d_rel = None

  if not directive.status:
    track.active = False
    track.model_prob = 0.0
    track.acquisition_age_s = 0.0
    track.measured_d_rel_m = None
    track.a_lead_k_mps2 = 0.0
    return lead, {
      "status": False,
      "true_d_rel_m": None,
      "measured_d_rel_m": None,
      "v_lead_mps": None,
      "a_lead_k_mps2": None,
      "model_prob": 0.0,
    }

  if not track.active or directive.acquisition_reset:
    track.active = True
    track.just_acquired = True
    track.acquisition_age_s = 0.0
    if directive.d_rel_override_m is not None:
      track.distance_m = state.true_distance_m + directive.d_rel_override_m
    elif directive.measured_d_rel_m is not None:
      track.distance_m = state.true_distance_m + directive.measured_d_rel_m
    elif track.distance_m <= state.true_distance_m:
      track.distance_m = state.true_distance_m + 200.0
    track.model_prob = 0.0
    track.last_speed_for_accel = directive.v_lead_mps
  else:
    track.just_acquired = False
    if directive.d_rel_override_m is not None:
      track.distance_m = state.true_distance_m + directive.d_rel_override_m

  track.speed_mps = directive.v_lead_mps

  true_d_rel = max(0.0, track.distance_m - state.true_distance_m)
  if directive.measured_d_rel_m is not None:
    measured_d_rel = directive.measured_d_rel_m
  else:
    measured_d_rel = _apply_distance_noise(true_d_rel, noise_profile, drel_rng)

  if directive.measured_v_rel_mps is not None:
    measured_v_rel = directive.measured_v_rel_mps
  else:
    std = max(noise_profile.vrel_floor_mps, noise_profile.vrel_factor * max(true_d_rel, 0.0))
    measured_v_rel = (track.speed_mps - state.true_speed_mps) + (_rng_normal(vrel_rng, std) if std > 0.0 else 0.0)

  target_prob = directive.model_prob_target
  if target_prob > 0.0 and track.acquisition_age_s < 0.25:
    track.model_prob = min(target_prob, max(track.model_prob, (track.acquisition_age_s + dt_s) / 0.25))
  else:
    track.model_prob = target_prob

  if directive.a_lead_k_mps2 is not None:
    a_lead_k = directive.a_lead_k_mps2
  elif track.just_acquired:
    a_lead_k = 0.0
  else:
    a_lead_k = (track.speed_mps - track.last_speed_for_accel) / dt_s

  v_lead_k = directive.v_lead_k_mps if directive.v_lead_k_mps is not None else track.speed_mps
  track.last_speed_for_accel = v_lead_k
  track.measured_d_rel_m = measured_d_rel
  track.a_lead_k_mps2 = a_lead_k

  lead.status = True
  lead.dRel = float(measured_d_rel)
  lead.yRel = float(directive.y_rel_m)
  lead.vRel = float(measured_v_rel)
  lead.aRel = float(a_lead_k - state.measured_accel_mps2)
  lead.vLead = float(track.speed_mps)
  lead.vLeadK = float(v_lead_k)
  lead.aLeadK = float(a_lead_k)
  lead.aLeadTau = float(_LEAD_ACCEL_TAU)
  lead.modelProb = float(track.model_prob)
  lead.dPath = float(directive.d_path_m if directive.d_path_m is not None else directive.y_rel_m)
  lead.vLat = float(directive.v_lat_mps)
  lead.fcw = bool(directive.fcw)
  lead.radar = bool(directive.radar)
  lead.radarTrackId = int(directive.radar_track_id)

  return lead, {
    "status": True,
    "true_d_rel_m": true_d_rel,
    "measured_d_rel_m": measured_d_rel,
    "v_lead_mps": track.speed_mps,
    "a_lead_k_mps2": a_lead_k,
    "model_prob": track.model_prob,
  }


def _advance_lead_tracks(lead_tracks: dict[str, LeadTrackState], dt_s: float) -> None:
  for track in lead_tracks.values():
    if track.active:
      track.distance_m += track.speed_mps * dt_s
      track.acquisition_age_s += dt_s


def _current_lead_meta(lead_tracks: dict[str, LeadTrackState], state: VehiclePlantState) -> dict[str, dict[str, Any]]:
  meta = {}
  for slot_name, track in lead_tracks.items():
    if not track.active:
      meta[slot_name] = {
        "status": False,
        "true_d_rel_m": None,
        "measured_d_rel_m": None,
        "v_lead_mps": None,
        "a_lead_k_mps2": None,
        "model_prob": 0.0,
      }
      continue

    meta[slot_name] = {
      "status": True,
      "true_d_rel_m": max(0.0, track.distance_m - state.true_distance_m),
      "measured_d_rel_m": track.measured_d_rel_m,
      "v_lead_mps": track.speed_mps,
      "a_lead_k_mps2": track.a_lead_k_mps2,
      "model_prob": track.model_prob,
    }
  return meta


def _apply_distance_noise(true_d_rel: float, profile: NoiseProfile, rng: np.random.Generator) -> float:
  if profile.name == "off" or profile.drel_frac <= 0.0:
    return true_d_rel
  std = max(profile.drel_floor_m, profile.drel_frac * max(true_d_rel, 0.0))
  return max(0.0, true_d_rel + _rng_normal(rng, std))


def _rng_normal(rng: np.random.Generator, std: float) -> float:
  return float(rng.normal(0.0, std))
