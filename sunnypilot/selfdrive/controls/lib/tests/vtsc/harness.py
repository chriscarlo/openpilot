#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
from unittest.mock import MagicMock, patch
from cereal import log

# Local import of the controller under test
from openpilot.tools.lib.logreader import LogReader

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
  VisionTurnController,
  curvature_to_speed,
)


@dataclass
class Step:
  # One simulation step of inputs
  curvature: float           # model curvature (1/m)
  confidence: float          # model vision confidence [0..1]
  curvature_ahead: Optional[float] = None  # optional "ahead" curvature (1/m) for horizon points > 0
  lead_d_rel_m: Optional[float] = None  # if provided, simulates a lead at this distance
  steering_angle_deg: float = 0.0  # steering wheel angle (deg), used by steering-curvature fallback
  dt: Optional[float] = None           # optional per-step dt override (seconds)
  live_map_data: Optional[Dict[str, Any]] = None  # optional liveMapDataSP fields for mapd telemetry tests
  lane_change_state: int = int(log.LaneChangeState.off)
  lane_change_direction: int = int(log.LaneChangeDirection.none)
  left_blinker: bool = False
  right_blinker: bool = False
  desired_curvature: Optional[float] = None
  actual_curvature: Optional[float] = None
  lateral_output: Optional[float] = None
  lateral_saturated: Optional[bool] = None
  lateral_active: Optional[bool] = None


def _mk_sm(curvature: float, curvature_ahead: Optional[float], v_pred: float, confidence: float,
           lead_d_rel_m: Optional[float], steering_angle_deg: float, live_map_data: Optional[Dict[str, Any]] = None,
           lane_change_state: int = int(log.LaneChangeState.off),
           lane_change_direction: int = int(log.LaneChangeDirection.none),
           left_blinker: bool = False, right_blinker: bool = False,
           desired_curvature: Optional[float] = None, actual_curvature: Optional[float] = None,
           lateral_output: Optional[float] = None, lateral_saturated: Optional[bool] = None,
           lateral_active: Optional[bool] = None):
  """Create a minimal SM stub with modelV2 and optional radarState.leadOne."""
  # modelV2.orientationRate.z is yaw rate (rad/s), not curvature.
  # Curvature κ (1/m) = yaw_rate / speed, so yaw_rate = κ * v.
  v_pred = float(max(0.0, v_pred))
  k_now = float(curvature)
  k_ahead = float(curvature_ahead) if curvature_ahead is not None else k_now
  # Use a simple 2-level profile: current point at k_now, all future points at k_ahead.
  k_points = [k_now] + [k_ahead] * 32
  yaw_rate_points = [k * v_pred for k in k_points]
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=yaw_rate_points),
    velocity=SimpleNamespace(x=[v_pred] * 33),
    laneLineProbs=[confidence] * 4,
    meta=SimpleNamespace(
      laneChangeState=lane_change_state,
      laneChangeDirection=lane_change_direction,
    ),
  )

  # Optional simple radarState lead stub
  if lead_d_rel_m is not None:
    lead_one = SimpleNamespace(status=True, dRel=float(lead_d_rel_m))
    radar_state = SimpleNamespace(leadOne=lead_one)
    valid = {'modelV2': True, 'radarState': True}
  else:
    radar_state = None
    valid = {'modelV2': True}

  live_map_msg = SimpleNamespace(**(live_map_data or {})) if live_map_data is not None else None
  if live_map_msg is not None:
    valid['liveMapDataSP'] = True

  controls_state = None
  if any(v is not None for v in (desired_curvature, actual_curvature, lateral_output, lateral_saturated, lateral_active)):
    desired_curvature_f = float(curvature if desired_curvature is None else desired_curvature)
    actual_curvature_f = float(desired_curvature_f if actual_curvature is None else actual_curvature)
    lat_state = SimpleNamespace(
      active=True if lateral_active is None else bool(lateral_active),
      output=0.0 if lateral_output is None else float(lateral_output),
      saturated=False if lateral_saturated is None else bool(lateral_saturated),
    )
    lateral_control_state = SimpleNamespace(which=lambda: 'torqueState', torqueState=lat_state)
    controls_state = SimpleNamespace(
      desiredCurvature=float(desired_curvature_f),
      curvature=float(actual_curvature_f),
      lateralControlState=lateral_control_state,
    )
    valid['controlsState'] = True

  class SM:
    def __init__(self, model, radar_state, valid):
      self.valid = valid
      self._data = {
        'modelV2': model,
        'carState': SimpleNamespace(
          gasPressed=False,
          steeringAngleDeg=float(steering_angle_deg),
          leftBlinker=bool(left_blinker),
          rightBlinker=bool(right_blinker),
        ),
      }
      if radar_state is not None:
        self._data['radarState'] = radar_state
      if live_map_msg is not None:
        self._data['liveMapDataSP'] = live_map_msg
      if controls_state is not None:
        self._data['controlsState'] = controls_state
    def __getitem__(self, key):
      return self._data.get(key)

  return SM(model, radar_state, valid)


def mk_vtsc_with_params(
  aggressiveness: float = 1.0,
  alpha: float = 0.3,
  hysteresis: float = 0.2,
  safety_bias: float = 0.1,
  bool_overrides: Optional[Dict[str, bool]] = None,
) -> VisionTurnController:
  """Instantiate VisionTurnController with Params patched to specified values."""
  class MockCP:  # minimal car params (enough to build VehicleModel in dev tests)
    mass = 1600.0
    rotationalInertia = 2500.0
    wheelbase = 2.75
    centerToFront = 1.20
    steerRatio = 15.0
    steerRatioRear = 0.0
    tireStiffnessFront = 80000.0
    tireStiffnessRear = 80000.0
  with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
    mp = MagicMock()
    # Be explicit about booleans: the old "return True for everything" masked behavior
    # by accidentally enabling debug/fail-open/testing toggles.
    def _get_bool(key: str) -> bool:
      if bool_overrides and key in bool_overrides:
        return bool(bool_overrides[key])
      # Keep core VTSC enabled for controller tests by default.
      if key in ('VisionTurnSpeedControl', 'VisionTurnSpeedControlOcclBypassWithLead'):
        return True
      return False
    mp.get_bool.side_effect = _get_bool
    def _get(key: str):
      if key.endswith('Aggressiveness'):
        return str(aggressiveness).encode()
      if key.endswith('FilterAlpha'):
        return str(alpha).encode()
      if key.endswith('HysteresisThreshold'):
        return str(hysteresis).encode()
      if key.endswith('SafetyBias'):
        return str(safety_bias).encode()
      return None
    mp.get.side_effect = _get
    MockParams.return_value = mp
    return VisionTurnController(MockCP())


def simulate_sequence(
  steps: Iterable[Step],
  vtsc: Optional[VisionTurnController] = None,
  v0_mps: float = 25.0,
  v_cruise_mps: float = 30.0,
  dt: float = 0.05,
  capture_history: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], List[Dict[str, Any]]]:
  """Run a simple time-stepped simulation and return the final debug snapshot.

  - Uses VTSC.update on each step with a synthetic clock.
  - Integrates acceleration to update v_ego.
  - Returns the controller snapshot_debug_state() from the final step.
  """
  ctrl = vtsc or mk_vtsc_with_params()
  v_ego = float(v0_mps)
  a_ego = 0.0
  t = 0.0

  history: List[Dict[str, Any]] = [] if capture_history else None

  for st in steps:
    step_dt = float(getattr(st, 'dt', dt) or dt)
    sm = _mk_sm(st.curvature, st.curvature_ahead, v_ego, st.confidence, st.lead_d_rel_m,
                st.steering_angle_deg, st.live_map_data, st.lane_change_state,
                st.lane_change_direction, st.left_blinker, st.right_blinker,
                st.desired_curvature, st.actual_curvature, st.lateral_output,
                st.lateral_saturated, st.lateral_active)
    # Patch time used inside controller to advance deterministically
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: t), \
         patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: t):
      ctrl.update(sm, True, v_ego, a_ego, v_cruise_mps)

    # Integrate acceleration to update speed for next step
    a_cmd = float(ctrl.a_target)
    v_ego = max(0.0, v_ego + a_cmd * step_dt)
    a_ego = a_cmd
    t += step_dt
    if history is not None:
      snap_step = ctrl.snapshot_debug_state() or {}
      history.append({
        't': t,
        'a_cmd': a_cmd,
        'vision_status': snap_step.get('vision_status'),
        'fov_occluded': snap_step.get('fov_occluded'),
        'occl_positive_margin': snap_step.get('occl_positive_margin'),
        'low_speed_margin_override': snap_step.get('low_speed_margin_override'),
      })

  # Final snapshot for assertions; augment with last a_target for sign checks
  snap = ctrl.snapshot_debug_state() or {}
  try:
    snap['a_last'] = float(ctrl.a_target)
  except Exception:
    snap['a_last'] = 0.0
  if history is not None:
    return snap, history
  return snap


def load_steps_from_rlog(path: str, limit: Optional[int] = 300) -> List[Step]:
  """Load VTSC simulation steps from a recorded rlog.

  - Uses modelV2 orientationRate/velocity to derive curvature.
  - Averages lane line probabilities for confidence.
  - Tracks latest radarState.leadOne distance for lead context.
  """
  steps: List[Step] = []
  last_lead: Optional[float] = None
  prev_model_time: Optional[int] = None

  for msg in LogReader(path):
    which = msg.which()
    if which == 'radarState':
      lead = msg.radarState.leadOne
      if lead is not None and bool(getattr(lead, 'status', False)):
        try:
          last_lead = float(getattr(lead, 'dRel', None))
        except Exception:
          last_lead = None
      else:
        last_lead = None
    elif which == 'modelV2':
      model = msg.modelV2
      try:
        vel = float(model.velocity.x[0]) if len(model.velocity.x) > 0 else 0.0
        rate = float(model.orientationRate.z[0]) if len(model.orientationRate.z) > 0 else 0.0
      except Exception:
        continue
      if vel <= 1e-3:
        vel = 1e-3
      curvature = rate / vel
      try:
        probs = list(model.laneLineProbs)
        confidence = float(sum(probs) / len(probs)) if probs else 1.0
      except Exception:
        confidence = 1.0
      step_dt = None
      if prev_model_time is not None:
        raw_dt = max(0.0, (msg.logMonoTime - prev_model_time) * 1e-9)
        # Clamp dt to a sane control range (20 Hz nominal)
        step_dt = min(0.15, max(0.01, raw_dt))
      prev_model_time = msg.logMonoTime
      steps.append(Step(curvature=curvature, confidence=confidence, lead_d_rel_m=last_lead, dt=step_dt))
      if limit is not None and len(steps) >= limit:
        break

  return steps

def simulate_sequence_trace(
  steps: Iterable[Step],
  vtsc: Optional[VisionTurnController] = None,
  v0_mps: float = 25.0,
  v_cruise_mps: float = 30.0,
  dt: float = 0.05,
  integrate_ego: bool = True,
) -> List[Dict[str, Any]]:
  """Run the simulation and return a per-step trace for timing assertions.

  Each trace entry includes:
  - t: simulated time (s)
  - v_ego: ego speed before integration (m/s)
  - a_target: VTSC accel target used for integration (m/s²)
  - v_turn: VTSC speed cap output (m/s)
  - fov_occluded: internal FOV-occlusion latch state (bool)
  - occl_lead_bypass_active: lead-bypass active flag (bool)
  """
  ctrl = vtsc or mk_vtsc_with_params()
  v_ego = float(v0_mps)
  a_ego = 0.0
  t = 0.0

  trace: List[Dict[str, Any]] = []

  for st in steps:
    sm = _mk_sm(st.curvature, st.curvature_ahead, v_ego, st.confidence, st.lead_d_rel_m,
                st.steering_angle_deg, st.live_map_data, st.lane_change_state,
                st.lane_change_direction, st.left_blinker, st.right_blinker,
                st.desired_curvature, st.actual_curvature, st.lateral_output,
                st.lateral_saturated, st.lateral_active)
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: t), \
         patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: t):
      ctrl.update(sm, True, v_ego, a_ego, v_cruise_mps)

    a_cmd = float(ctrl.a_target)
    try:
      v_turn = float(ctrl.v_turn)
    except Exception:
      v_turn = 0.0
    snap = ctrl.snapshot_debug_state() or {}
    trace.append({
      't': float(t),
      'v_ego': float(v_ego),
      'a_target': float(a_cmd),
      'v_turn': float(v_turn),
      'fov_occluded': bool(getattr(ctrl, '_fov_occluded', False)),
      'occl_lead_bypass_active': bool(getattr(ctrl, '_occl_lead_bypass_active', False)),
      'conf': float(snap.get('conf', 0.0) or 0.0),
      'vision_status': str(snap.get('vision_status', 'UNKNOWN') or 'UNKNOWN'),
    })

    if integrate_ego:
      v_ego = max(0.0, v_ego + a_cmd * dt)
      a_ego = a_cmd
    else:
      a_ego = 0.0
    t += dt

  return trace
