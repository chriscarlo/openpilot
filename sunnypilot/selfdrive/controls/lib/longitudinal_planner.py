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
from openpilot.sunnypilot.selfdrive.controls.lib.object_hazard_controller import ObjectHazardController
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_mark_recorder import MarkRecorder
from openpilot.sunnypilot.selfdrive.controls.lib.planner_lag_debug import (
  PlannerLagRecorder,
  SPAN_PUBLISH_LONGITUDINAL_PLAN_SP,
  SPAN_PREVIEW_ENCODE,
  SPAN_RESPONSE_MODEL_LOOKUP,
  SPAN_UPDATE_V_CRUISE_TOTAL,
  SPAN_VTSC_UPDATE,
  end_span,
  record_fields,
  start_span,
)
from openpilot.sunnypilot.models.helpers import get_active_bundle

from openpilot.sunnypilot.selfdrive.controls.lib.vibe_personality.vibe_personality import VibePersonalityController
DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState


def _finite_diagnostic_float(value, default: float = 0.0) -> float:
  try:
    number = float(value)
  except (TypeError, ValueError, OverflowError):
    return float(default)
  return number if math.isfinite(number) else float(default)


def _diagnostic_int(value, default: int = 0) -> int:
  try:
    return int(value)
  except (TypeError, ValueError, OverflowError):
    return int(default)


def _lead_state_diagnostic(payload) -> dict:
  data = payload if isinstance(payload, dict) else {}
  return {
    "status": bool(data.get("status", False)),
    "dRelM": _finite_diagnostic_float(data.get("dRel")),
    "vRelMps": _finite_diagnostic_float(data.get("vRel")),
    "aRelMps2": _finite_diagnostic_float(data.get("aRel")),
    "vLeadMps": _finite_diagnostic_float(data.get("vLead")),
    "vLeadKMps": _finite_diagnostic_float(data.get("vLeadK")),
    "aLeadKMps2": _finite_diagnostic_float(data.get("aLeadK")),
    "modelProb": _finite_diagnostic_float(data.get("modelProb")),
    "radar": bool(data.get("radar", False)),
    "radarTrackId": _diagnostic_int(data.get("radarTrackId", -1) or -1, -1),
    "fcw": bool(data.get("fcw", False)),
    "closingGovernorRecovery": bool(data.get("closingGovernorRecovery", False)),
    "steadyParityCurrentThreat": bool(data.get("steadyParityCurrentThreat", False)),
    "accelCorrRawHardBraking": bool(data.get("accelCorrRawHardBraking", False)),
  }


def build_lead_diagnostics_snapshot(planner) -> dict:
  """Build the versioned logging payload without affecting planner behavior."""
  mpc = getattr(planner, "mpc", None)
  virtual_debug = getattr(mpc, "hyundai_virtual_lead_debug", {}) if mpc is not None else {}
  virtual_debug = virtual_debug if isinstance(virtual_debug, dict) else {}
  acc_debug = getattr(mpc, "acc_source_debug", {}) if mpc is not None else {}
  acc_debug = acc_debug if isinstance(acc_debug, dict) else {}
  opening = virtual_debug.get("opening_recovery", {})
  opening = opening if isinstance(opening, dict) else {}
  source = str(virtual_debug.get("source") or getattr(mpc, "source", "") or "")
  slot = {"lead0": "slot0", "lead1": "slot1"}.get(source)
  stability_debug = getattr(mpc, "lead_stability_debug", {}) if mpc is not None else {}
  stability = stability_debug.get(slot, {}) if slot and isinstance(stability_debug, dict) else {}
  stability = stability if isinstance(stability, dict) else {}
  release = getattr(planner, "lead_brake_release_debug", {})
  release = release if isinstance(release, dict) else {}
  release_floor = release.get("floor_mps2")
  slowdown_ceiling = getattr(mpc, "lead_slowdown_accel_ceiling", None) if mpc is not None else None
  input_state = _lead_state_diagnostic(virtual_debug.get("input"))
  virtual_state = _lead_state_diagnostic(virtual_debug.get("filtered"))
  confirm_frames = max(0, min(65535, _diagnostic_int(opening.get("confirm_frames", 0) or 0)))
  return {
    "valid": bool(virtual_debug.get("active", False) and input_state["status"] and virtual_state["status"]),
    "version": 1,
    "source": source,
    "reason": str(acc_debug.get("reason", virtual_debug.get("reset_reason") or "inactive")),
    "input": input_state,
    "virtual": virtual_state,
    "inputObstacleM": _finite_diagnostic_float(acc_debug.get("best_lead_obstacle")),
    "virtualObstacleM": _finite_diagnostic_float(acc_debug.get("filtered_lead_obstacle")),
    "selectedObstacleM": _finite_diagnostic_float(getattr(mpc, "selected_obstacle_m", 0.0)),
    "cruiseObstacleM": _finite_diagnostic_float(acc_debug.get("cruise_obstacle")),
    "inputGapSurplusM": _finite_diagnostic_float(acc_debug.get("raw_gap_surplus_m")),
    "virtualGapSurplusM": _finite_diagnostic_float(acc_debug.get("filtered_gap_surplus_m")),
    "accelCorrClamped": bool(stability.get("accel_corr_clamped", False)),
    "accelCorrAmplified": bool(stability.get("accel_corr_amplified", False)),
    "slowdownCeilingValid": slowdown_ceiling is not None,
    "slowdownCeilingMps2": _finite_diagnostic_float(slowdown_ceiling),
    "releaseFloorValid": release_floor is not None,
    "releaseFloorMps2": _finite_diagnostic_float(release_floor),
    "releaseReason": str(release.get("reason", "inactive")),
    "openingRecoveryActive": bool(opening.get("active", False)),
    "openingRecoveryCandidate": bool(opening.get("candidate", False)),
    "openingRecoveryConfirmFrames": confirm_frames,
    "openingRecoveryTauS": _finite_diagnostic_float(opening.get("tau_s"), 1.0),
    "openingRecoveryReason": str(opening.get("reason", "inactive")),
  }


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
    self.object_hazard = ObjectHazardController()
    self.planner_lag_debug = PlannerLagRecorder()
    self.mark_recorder = MarkRecorder()
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
    total_span = start_span(SPAN_UPDATE_V_CRUISE_TOTAL)
    try:
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
      response_model_span = start_span(SPAN_RESPONSE_MODEL_LOOKUP)
      try:
        planner_accel_limits = getattr(self, '_planner_output_accel_limits', None)
        response_model = self.mpc.get_cruise_response_model(
          v_ego,
          actuation_delay_s=float(getattr(self.CP, 'longitudinalActuatorDelay', 0.0)) + DT_MDL,
          planner_accel_limits=planner_accel_limits,
        )
      except Exception:
        response_model = None
      finally:
        end_span(response_model_span)
      self.v_tsc.set_longitudinal_response_model(response_model)
      vt_update_t0 = time.monotonic()
      vt_update_span = start_span(SPAN_VTSC_UPDATE)
      try:
        self.v_tsc.update(sm, True, v_ego, a_ego, v_cruise_for_vtsc)
      finally:
        end_span(vt_update_span)
      vt_update_dt = time.monotonic() - vt_update_t0
      try:
        map_data = sm['liveMapDataSP']
        nearby_segments = len(getattr(map_data, 'nearbyRoadSegments', []))
        road_geometry_valid = bool(getattr(map_data, 'roadGeometryValid', False))
      except Exception:
        nearby_segments = 0
        road_geometry_valid = False
      record_fields(
        strategy_mode=str(getattr(self.v_tsc, '_dbg_strategy_mode', 'unknown') or 'unknown'),
        strategy_state=str(getattr(self.v_tsc, '_dbg_strategy_state', 'idle') or 'idle'),
        map_floor_active=bool(getattr(self.v_tsc, '_dbg_map_floor_active', False)),
        map_floor_reason=str(getattr(self.v_tsc, '_dbg_map_floor_reason', '') or ''),
        map_tail_active=bool(getattr(self.v_tsc, '_map_tail_active', False)),
        map_tail_reason=str(getattr(self.v_tsc, '_map_tail_reason', '') or ''),
        map_tail_compute_reason=str(getattr(self.v_tsc, '_map_tail_compute_reason', '') or ''),
        map_tail_coverage=float(getattr(self.v_tsc, '_map_tail_last_coverage', 0.0) or 0.0),
        map_advisory_cap=float(getattr(self.v_tsc, '_dbg_map_advisory_cap', 0.0) or 0.0),
        map_strategic_cap=float(getattr(self.v_tsc, '_dbg_map_strategic_cap', 0.0) or 0.0),
        map_selected_cap=float(getattr(self.v_tsc, '_dbg_selected_cap', 0.0) or 0.0),
        map_anchor_dist_m=float(getattr(self.v_tsc, '_dbg_map_anchor_dist_m', 0.0) or 0.0),
        active_cap=str(getattr(self.v_tsc, '_dbg_active_cap', '') or ''),
        curve_preview_valid=bool(self.v_tsc.curve_preview_valid),
        curve_preview_points=len(self.v_tsc.curve_preview_points),
        curve_preview_tiles=len(self.v_tsc.curve_preview_tiles),
        curve_preview_branch_stubs=len(self.v_tsc.curve_preview_branch_stubs),
        curve_distance_m=float(self.v_tsc.curve_preview_distance_m),
        curve_time_to_s=float(self.v_tsc.curve_preview_time_to_s),
        curve_max_curvature=float(self.v_tsc.curve_preview_kappa_max),
        vtsc_velocity=float(self.v_tsc.v_turn),
        vtsc_state=int(self.v_tsc.state),
        vtsc_state_name=str(self.v_tsc.state),
        map_geometry_valid=road_geometry_valid,
        nearby_segment_count=int(nearby_segments),
        response_model_present=bool(response_model is not None),
      )
      if vt_update_dt >= 0.03 or (bool(getattr(self.v_tsc, '_apex_exit_ready', False)) and vt_update_dt >= 0.015):
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
            curve_preview_tiles=len(self.v_tsc.curve_preview_tiles),
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

      # Update object hazard controller
      self.object_hazard.update(sm, v_ego, a_ego, v_cruise)
      v_cruise_object_hazard = self.object_hazard.speed_recommendation if self.object_hazard.is_active else V_CRUISE_UNSET

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
      if self.object_hazard.is_active and v_cruise_object_hazard != V_CRUISE_UNSET:
        cruise_speeds.append(v_cruise_object_hazard)

      v_cruise_final = min(cruise_speeds)
      return v_cruise_final
    finally:
      end_span(total_span)

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
    publish_span = start_span(SPAN_PUBLISH_LONGITUDINAL_PLAN_SP)
    try:
      plan_sp_send = messaging.new_message('longitudinalPlanSP')

      plan_sp_send.valid = sm.all_checks(service_list=['controlsState'])

      longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
      longitudinalPlanSP.events = self.events_sp.to_msg()
      longitudinalPlanSP.leadDiagnostics = build_lead_diagnostics_snapshot(self)

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
      # Preserve the map/vision arbitration decision in every route log. The
      # controller's legacy `state` remains compatibility-only, so it cannot
      # attribute whether the final VTSC cap came from map lookahead or vision.
      visionTurnSpeedControl.mapStrategyState = str(getattr(self.v_tsc, '_dbg_strategy_state', 'idle') or 'idle')
      visionTurnSpeedControl.mapStrategyMode = str(getattr(self.v_tsc, '_dbg_strategy_mode', 'strategic') or 'strategic')
      visionTurnSpeedControl.mapFloorActive = bool(getattr(self.v_tsc, '_dbg_map_floor_active', False))
      visionTurnSpeedControl.mapFloorReason = str(getattr(self.v_tsc, '_dbg_map_floor_reason', '') or '')
      visionTurnSpeedControl.visionRelaxAllowed = bool(getattr(self.v_tsc, '_dbg_vision_relax_allowed', False))
      visionTurnSpeedControl.visionRelaxReason = str(getattr(self.v_tsc, '_dbg_vision_relax_reason', '') or '')
      visionTurnSpeedControl.mapAdvisoryCap = float(getattr(self.v_tsc, '_dbg_map_advisory_cap', 0.0) or 0.0)
      visionTurnSpeedControl.mapStrategicCap = float(getattr(self.v_tsc, '_dbg_map_strategic_cap', 0.0) or 0.0)
      visionTurnSpeedControl.visionLocalCap = float(getattr(self.v_tsc, '_dbg_vision_local_cap', 0.0) or 0.0)
      visionTurnSpeedControl.selectedCap = float(getattr(self.v_tsc, '_dbg_selected_cap', 0.0) or 0.0)
      visionTurnSpeedControl.mapAnchorDistanceM = float(getattr(self.v_tsc, '_dbg_map_anchor_dist_m', 0.0) or 0.0)
      visionTurnSpeedControl.mapAnchorCurvature = float(getattr(self.v_tsc, '_dbg_map_anchor_k', 0.0) or 0.0)
      visionTurnSpeedControl.mapAnchorIndex = int(getattr(self.v_tsc, '_map_tail_anchor_index', -1))
      visionTurnSpeedControl.mapTakeoverDwellS = float(getattr(self.v_tsc, '_dbg_map_takeover_dwell_s', 0.0) or 0.0)
      visionTurnSpeedControl.mapCounterevidenceDwellS = float(getattr(self.v_tsc, '_dbg_map_counterevidence_dwell_s', 0.0) or 0.0)
      # Rally co-pilot curve preview (map-enriched). HUD-only telemetry.
      preview_encode_t0 = time.monotonic()
      preview_encode_span = start_span(SPAN_PREVIEW_ENCODE)
      preview_pts_count = 0
      preview_tile_count = 0
      preview_stub_count = 0
      try:
        visionTurnSpeedControl.curvePreviewValid = bool(self.v_tsc.curve_preview_valid)
        visionTurnSpeedControl.curveDistanceM = float(self.v_tsc.curve_preview_distance_m)
        visionTurnSpeedControl.curveTimeToS = float(self.v_tsc.curve_preview_time_to_s)
        visionTurnSpeedControl.curveMaxCurvature = float(self.v_tsc.curve_preview_kappa_max)
        visionTurnSpeedControl.curveDirection = int(self.v_tsc.curve_preview_direction)
        visionTurnSpeedControl.curveSeverity = int(self.v_tsc.curve_preview_severity)
        visionTurnSpeedControl.mapWindingValid = bool(getattr(self.v_tsc, '_mapd_winding_valid', False))
        visionTurnSpeedControl.mapWindingLevel = int(getattr(self.v_tsc, '_mapd_winding_level', 0) or 0)
        visionTurnSpeedControl.mapWindingScore = int(getattr(self.v_tsc, '_mapd_winding_score', 0) or 0)
        visionTurnSpeedControl.mapWindingConfidence = int(getattr(self.v_tsc, '_mapd_winding_confidence', 0) or 0)
        visionTurnSpeedControl.mapWindingCurrentLevel = int(getattr(self.v_tsc, '_mapd_winding_current_level', 0) or 0)
        visionTurnSpeedControl.mapWindingCurrentScore = int(getattr(self.v_tsc, '_mapd_winding_current_score', 0) or 0)
        visionTurnSpeedControl.mapWindingCurrentConfidence = int(getattr(self.v_tsc, '_mapd_winding_current_confidence', 0) or 0)
        visionTurnSpeedControl.mapWindingWayCount = int(getattr(self.v_tsc, '_mapd_winding_way_count', 0) or 0)
        visionTurnSpeedControl.windingContextActive = bool(getattr(self.v_tsc, '_winding_context_active', False))
        visionTurnSpeedControl.windingContextLevel = int(getattr(self.v_tsc, '_winding_context_level', 0) or 0)
        visionTurnSpeedControl.windingContextScore = float(getattr(self.v_tsc, '_winding_context_score', 0.0) or 0.0)
        visionTurnSpeedControl.windingContextConfidence = float(getattr(self.v_tsc, '_winding_context_confidence', 0.0) or 0.0)
        winding_context_source = str(getattr(self.v_tsc, '_winding_context_source', 'none') or 'none')
        source_enum = custom.LongitudinalPlanSP.VisionTurnSpeedControl.WindingContextSource.none
        if winding_context_source == 'local':
          source_enum = custom.LongitudinalPlanSP.VisionTurnSpeedControl.WindingContextSource.local
        elif winding_context_source == 'mapd':
          source_enum = custom.LongitudinalPlanSP.VisionTurnSpeedControl.WindingContextSource.mapd
        elif winding_context_source == 'blended':
          source_enum = custom.LongitudinalPlanSP.VisionTurnSpeedControl.WindingContextSource.blended
        visionTurnSpeedControl.windingContextSource = source_enum
        pts = self.v_tsc.curve_preview_points
        preview_pts_count = len(pts)
        if pts:
          out_pts = visionTurnSpeedControl.init('curvePreviewPoints', len(pts))
          for i, (x_fwd, y_left) in enumerate(pts):
            out_pts[i].xFwdM = float(x_fwd)
            out_pts[i].yLeftM = float(y_left)
        tiles = self.v_tsc.curve_preview_tiles
        preview_tile_count = len(tiles)
        if tiles:
          out_tiles = visionTurnSpeedControl.init('curvePreviewTiles', len(tiles))
          for i, tile in enumerate(tiles):
            out_tiles[i].tileId = int(tile.get('id', 0) or 0)
            out_tiles[i].distanceM = float(tile.get('distance_m', 0.0) or 0.0)
            out_tiles[i].timeToS = float(tile.get('time_to_s', 0.0) or 0.0)
            out_tiles[i].direction = int(tile.get('direction', 0) or 0)
            out_tiles[i].severity = int(tile.get('severity', 0) or 0)
            out_tiles[i].maxCurvature = float(tile.get('max_curvature', 0.0) or 0.0)
            out_tiles[i].advisorySpeedMps = float(tile.get('advisory_speed_mps', 0.0) or 0.0)
            tile_pts = tile.get('points', [])
            if tile_pts:
              out_tile_pts = out_tiles[i].init('points', len(tile_pts))
              for j, (x_fwd, y_left) in enumerate(tile_pts):
                out_tile_pts[j].xFwdM = float(x_fwd)
                out_tile_pts[j].yLeftM = float(y_left)
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
      finally:
        end_span(preview_encode_span)
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
            curve_preview_tiles=int(preview_tile_count),
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

      objectHazardControl = longitudinalPlanSP.objectHazardControl
      objectHazardControl.enabled = bool(self.object_hazard.enabled)
      objectHazardControl.active = bool(self.object_hazard.is_active)
      objectHazardControl.recommendedSpeed = float(
        0.0 if self.object_hazard.speed_recommendation == V_CRUISE_UNSET else self.object_hazard.speed_recommendation
      )
      objectHazardControl.stopRequired = bool(self.object_hazard.stop_required)
      objectHazardControl.hazardDistanceM = float(self.object_hazard.hazard_distance_m)
      objectHazardControl.hazardConfidence = float(self.object_hazard.hazard_confidence)
      objectHazardControl.hazardClass = self.object_hazard.hazard_class

      pm.send('longitudinalPlanSP', plan_sp_send)
    finally:
      end_span(publish_span)
