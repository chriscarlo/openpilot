"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import time

from cereal import messaging, custom
from opendbc.car import structs
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit_controller.speed_limit_controller import SpeedLimitController
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
from openpilot.sunnypilot.selfdrive.controls.lib.rti_controller import RTIController
from openpilot.sunnypilot.selfdrive.controls.lib.weather_controller import WeatherController
from openpilot.sunnypilot.models.helpers import get_active_bundle

from openpilot.sunnypilot.selfdrive.controls.lib.vibe_personality.vibe_personality import VibePersonalityController
DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, mpc):
    self.CP = CP
    self.events_sp = EventsSP()
    self.transition_init()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.vibe_controller = VibePersonalityController()
    self.v_tsc = VisionTurnController(CP)
    self.slc = SpeedLimitController(CP)
    self.rti = RTIController(CP)
    self.weather = WeatherController()
    model_bundle = get_active_bundle()
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None

  @property
  def mlsim(self) -> bool:
    # If we don't have a generation set, we assume its default model. Which as of today are mlsim.
    return bool(self.generation is None or self.generation >= 11)

  def get_mpc_mode(self) -> str | None:
    if not self.dec.active():
      return None

    return self.dec.mode()

  def update_v_cruise(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> float:
    self.events_sp.clear()

    self.slc.update(sm, v_ego, a_ego, v_cruise, self.events_sp)

    v_cruise_slc = self.slc.speed_limit_offseted if self.slc.is_active else V_CRUISE_UNSET

    # VTSC (with map enrichment) should compute continuously onroad so the HUD can
    # show curve previews even when openpilot is not engaged. Only apply VTSC as a
    # speed source when longitudinal control is active.
    apply_vtsc = bool(sm['carControl'].longActive)
    # When not engaged, there may be no meaningful cruise setpoint. Use a high cap
    # so VTSC produces physics-based advisory speeds instead of collapsing to 0.
    v_cruise_for_vtsc = float(v_cruise if apply_vtsc else (V_CRUISE_MAX * CV.KPH_TO_MS))
    try:
      planner_accel_limits = getattr(self, '_planner_output_accel_limits', None)
      response_model = self.mpc.get_cruise_response_model(
        v_ego,
        actuation_delay_s=float(getattr(self.CP, 'longitudinalActuatorDelay', 0.0)) + DT_MDL,
        planner_accel_limits=planner_accel_limits,
      )
    except Exception:
      response_model = None
    self.v_tsc.set_longitudinal_response_model(response_model)
    vt_update_t0 = time.monotonic()
    self.v_tsc.update(sm, True, v_ego, a_ego, v_cruise_for_vtsc)
    vt_update_dt = time.monotonic() - vt_update_t0
    if vt_update_dt >= 0.03 or (bool(getattr(self.v_tsc, '_apex_exit_ready', False)) and vt_update_dt >= 0.015):
      try:
        map_data = sm['liveMapDataSP']
        nearby_segments = len(getattr(map_data, 'nearbyRoadSegments', []))
        road_geometry_valid = bool(getattr(map_data, 'roadGeometryValid', False))
      except Exception:
        nearby_segments = 0
        road_geometry_valid = False
      try:
        cloudlog.warning(
          "VTSC slow update",
          dt_ms=round(vt_update_dt * 1000.0, 2),
          v_ego=float(v_ego),
          state=str(self.v_tsc.state),
          apex_exit_ready=bool(getattr(self.v_tsc, '_apex_exit_ready', False)),
          strategy_mode=str(getattr(self.v_tsc, '_dbg_strategy_mode', 'unknown') or 'unknown'),
          strategy_state=str(getattr(self.v_tsc, '_dbg_strategy_state', 'idle') or 'idle'),
          map_tail_reason=str(getattr(self.v_tsc, '_map_tail_reason', '') or ''),
          curve_preview_valid=bool(self.v_tsc.curve_preview_valid),
          curve_preview_points=len(self.v_tsc.curve_preview_points),
          curve_preview_branch_stubs=len(self.v_tsc.curve_preview_branch_stubs),
          nearby_segments=int(nearby_segments),
          road_geometry_valid=road_geometry_valid,
        )
      except Exception:
        pass
    v_cruise_v_tsc = self.v_tsc.v_turn if (apply_vtsc and self.v_tsc.is_active) else V_CRUISE_UNSET

    # Update RTI controller
    self.rti.update(sm, v_ego, a_ego, v_cruise, posted_speed_limit=self.slc.speed_limit)
    v_cruise_rti = self.rti.speed_recommendation if self.rti.is_active else V_CRUISE_UNSET

    # Update Weather controller
    self.weather.update(v_ego, v_cruise)
    v_cruise_weather = self.weather.speed_recommendation if self.weather.is_active else V_CRUISE_UNSET

    cruise_speeds = [v_cruise]

    # MTSC publisher deprecated: VTSC handles map lookahead internally

    if self.v_tsc.is_active and v_cruise_v_tsc != V_CRUISE_UNSET:
      cruise_speeds.append(v_cruise_v_tsc)
    if self.slc.is_active and v_cruise_slc != V_CRUISE_UNSET:
      cruise_speeds.append(v_cruise_slc)
    if self.rti.is_active and v_cruise_rti != V_CRUISE_UNSET:
      cruise_speeds.append(v_cruise_rti)
    if self.weather.is_active and v_cruise_weather != V_CRUISE_UNSET:
      cruise_speeds.append(v_cruise_weather)

    v_cruise_final = min(cruise_speeds)
    return v_cruise_final

  def transition_init(self) -> None:
    self._transition_counter: int = 0
    self._transition_steps: int = 15
    self._last_mode = 'acc'

  def handle_mode_transition(self, mode: str) -> None:
    if self._last_mode != mode:
      if mode == 'blended':
        self._transition_counter = 0
      self._last_mode = mode

  def blend_accel_transition(self, mpc_accel: float, e2e_accel: float, v_ego: float) -> float:
    if self.dec.enabled():
      if self._transition_counter < self._transition_steps:
        self._transition_counter += 1
        progress = self._transition_counter / self._transition_steps
        if v_ego > 5.0 and e2e_accel < 0.0:
          if mpc_accel < 0.0 and e2e_accel > mpc_accel:
            return mpc_accel
          # use k4.0 and normalize midpoint at 0.4
          sigmoid = 1.0 / (1.0 + math.exp(-4.0 * (abs(e2e_accel / ACCEL_MIN) - 0.4)))
          blend_factor = 1.0 - (1.0 - progress) * (1.0 - sigmoid)
          blended = mpc_accel + (e2e_accel - mpc_accel) * blend_factor
          return blended
    return min(mpc_accel, e2e_accel)

  def update(self, sm: messaging.SubMaster) -> None:
    self.dec.update(sm)
    self.vibe_controller.update()

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control
    dec = longitudinalPlanSP.dec
    dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
    dec.enabled = self.dec.enabled()
    dec.active = self.dec.active()

    # Vision Turn Speed Control
    visionTurnSpeedControl = longitudinalPlanSP.visionTurnSpeedControl
    visionTurnSpeedControl.state = self.v_tsc.state
    visionTurnSpeedControl.velocity = float(self.v_tsc.v_turn)
    visionTurnSpeedControl.currentLateralAccel = float(self.v_tsc.current_lat_acc)
    visionTurnSpeedControl.maxPredictedLateralAccel = float(self.v_tsc.max_pred_lat_acc)
    # Rally co-pilot curve preview (map-enriched). HUD-only telemetry.
    preview_encode_t0 = time.monotonic()
    preview_pts_count = 0
    preview_stub_count = 0
    try:
      visionTurnSpeedControl.curvePreviewValid = bool(self.v_tsc.curve_preview_valid)
      visionTurnSpeedControl.curveDistanceM = float(self.v_tsc.curve_preview_distance_m)
      visionTurnSpeedControl.curveTimeToS = float(self.v_tsc.curve_preview_time_to_s)
      visionTurnSpeedControl.curveMaxCurvature = float(self.v_tsc.curve_preview_kappa_max)
      visionTurnSpeedControl.curveDirection = int(self.v_tsc.curve_preview_direction)
      visionTurnSpeedControl.curveSeverity = int(self.v_tsc.curve_preview_severity)
      pts = self.v_tsc.curve_preview_points
      preview_pts_count = len(pts)
      if pts:
        out_pts = visionTurnSpeedControl.init('curvePreviewPoints', len(pts))
        for i, (x_fwd, y_left) in enumerate(pts):
          out_pts[i].xFwdM = float(x_fwd)
          out_pts[i].yLeftM = float(y_left)
      branch_stubs = self.v_tsc.curve_preview_branch_stubs
      preview_stub_count = len(branch_stubs)
      if branch_stubs:
        out_stubs = visionTurnSpeedControl.init('curvePreviewBranchStubs', len(branch_stubs))
        for i, stub in enumerate(branch_stubs):
          out_stubs[i].highlighted = bool(stub.get('highlighted', False))
          stub_pts = stub.get('points', [])
          if stub_pts:
            out_stub_pts = out_stubs[i].init('points', len(stub_pts))
            for j, (x_fwd, y_left) in enumerate(stub_pts):
              out_stub_pts[j].xFwdM = float(x_fwd)
              out_stub_pts[j].yLeftM = float(y_left)
    except Exception:
      # Backward compatibility if capnp/python bindings are older.
      pass
    preview_encode_dt = time.monotonic() - preview_encode_t0
    if preview_encode_dt >= 0.01 or (bool(getattr(self.v_tsc, '_apex_exit_ready', False)) and preview_encode_dt >= 0.005):
      try:
        cloudlog.warning(
          "VTSC preview encode slow",
          dt_ms=round(preview_encode_dt * 1000.0, 2),
          state=str(self.v_tsc.state),
          apex_exit_ready=bool(getattr(self.v_tsc, '_apex_exit_ready', False)),
          curve_preview_valid=bool(self.v_tsc.curve_preview_valid),
          curve_preview_points=int(preview_pts_count),
          curve_preview_branch_stubs=int(preview_stub_count),
        )
      except Exception:
        pass

    # Speed Limit Control
    slc = longitudinalPlanSP.slc
    slc.state = self.slc.state
    slc.enabled = self.slc.is_enabled
    slc.active = self.slc.is_active
    slc.speedLimit = float(self.slc.speed_limit)
    slc.speedLimitOffset = float(self.slc.speed_limit_offset)
    slc.distToSpeedLimit = float(self.slc.distance)
    # Publish selected SLC source explicitly (avoid UI heuristics)
    try:
      if self.slc.source == 1:  # Source.car_state
        slc.source = custom.LongitudinalPlanSP.SlcSource.car
      elif self.slc.source == 2:  # Source.map_data
        slc.source = custom.LongitudinalPlanSP.SlcSource.map
      else:
        slc.source = custom.LongitudinalPlanSP.SlcSource.none
    except Exception:
      # Backward compatibility if older custom.capnp without source field
      pass

    pm.send('longitudinalPlanSP', plan_sp_send)
