#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.common.params import Params

from .pipeline_harness import (
  Step,
  import_longitudinal_planner_sp,
  run_vtsc_min_of_sources,
)


class _NoOpSLC:
  """SpeedLimitController stub for VTSC-focused tests."""
  def __init__(self, CP):
    self._enabled = False

  def update(self, *args, **kwargs) -> None:
    pass

  @property
  def state(self):
    return 0

  @property
  def is_enabled(self) -> bool:
    return False

  @property
  def is_active(self) -> bool:
    return False

  @property
  def speed_limit_offseted(self) -> float:
    return 0.0

  @property
  def speed_limit(self) -> float:
    return 0.0

  @property
  def speed_limit_offset(self) -> float:
    return 0.0

  @property
  def distance(self) -> float:
    return 0.0

  @property
  def source(self) -> int:
    return 0


class _NoOpRTI:
  """RTIController stub for VTSC-focused tests."""
  def __init__(self, CP):
    pass

  def update(self, *args, **kwargs) -> None:
    pass

  @property
  def is_active(self) -> bool:
    return False

  @property
  def speed_recommendation(self) -> float:
    return 0.0

  @property
  def state(self) -> int:
    return 0


class _NoOpDEC:
  def __init__(self, CP, mpc):
    pass

  def update(self, sm) -> None:
    pass

  def active(self) -> bool:
    return False

  def enabled(self) -> bool:
    return False

  def mode(self) -> str:
    return 'acc'


class _NoOpVibe:
  def update(self) -> None:
    pass

  def is_accel_enabled(self) -> bool:
    return False

  def get_accel_limits(self, v_ego):
    return None


class _MockCP:
  # Minimal CP shape required by LongitudinalPlannerSP constructor and related stubs.
  pass


@pytest.fixture()
def planner_sp(monkeypatch):
  # Import module with model-runner deps stubbed out.
  mod = import_longitudinal_planner_sp()

  # Patch out non-VTSC controllers so update_v_cruise doesn't require extra messages.
  monkeypatch.setattr(mod, 'SpeedLimitController', _NoOpSLC, raising=True)
  monkeypatch.setattr(mod, 'RTIController', _NoOpRTI, raising=True)
  monkeypatch.setattr(mod, 'DynamicExperimentalController', _NoOpDEC, raising=True)
  monkeypatch.setattr(mod, 'VibePersonalityController', _NoOpVibe, raising=True)

  # Ensure VTSC is enabled via Params (use real Params store for fidelity).
  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  # Keep defaults for everything else unless explicitly needed by tests.
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  # Construct planner after Params are configured so VTSC reads enable state on init.
  dummy_mpc = object()
  planner = mod.LongitudinalPlannerSP(_MockCP(), dummy_mpc)

  return planner


def test_vtsc_cap_is_selected_when_lower_than_cruise(planner_sp):
  # Straight-ish road with a moderate curve ahead should drop VTSC below cruise.
  # Use curvature large enough to force a meaningful cap.
  v0 = 30.0
  v_cruise = 33.0
  steps = [
    Step(curvature=0.006, confidence=0.95, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True),
  ] * 40  # ~2.0s worth of updates

  hist = run_vtsc_min_of_sources(planner_sp, steps=steps, dt=0.05, start_t=0.0)
  assert hist

  last = hist[-1]
  assert last['vtsc_active'] is True
  assert last['vtsc_v_turn'] > 0.0
  # With SLC/RTI disabled, v_cruise_final should be min(v_cruise, vtsc_v_turn).
  assert last['v_cruise_final'] == pytest.approx(min(v_cruise, last['vtsc_v_turn']), abs=1e-6)


def test_longitudinal_plan_sp_publishes_vtsc_velocity(planner_sp):
  class _FakePM:
    def __init__(self):
      self.sent = {}
    def send(self, name, msg) -> None:
      self.sent[name] = msg

  # Run one update so VTSC has populated v_turn.
  v0 = 30.0
  v_cruise = 33.0
  steps = [Step(curvature=0.01, confidence=0.95, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True)] * 5
  hist = run_vtsc_min_of_sources(planner_sp, steps=steps, dt=0.05, start_t=0.0)
  assert hist

  # Publish and inspect the emitted message.
  pm = _FakePM()
  # Reuse a minimal SM (all_checks is provided by FakeSubMaster inside run_vtsc_min_of_sources).
  # The publisher uses internal controller state for VTSC fields.
  sm_last = None
  from .pipeline_harness import FakeSubMaster, make_model_v2, make_radar_state, make_car_state, make_car_control
  model = make_model_v2(curvature=0.01, v_pred=v0, confidence=0.95)
  sm_last = FakeSubMaster(
    data={
      'modelV2': model,
      'radarState': make_radar_state(lead_d_rel_m=None),
      'carState': make_car_state(gas_pressed=False),
      'carControl': make_car_control(long_active=True),
      'controlsState': SimpleNamespace(),  # only used for validity; all_checks() returns True
      'controlsStateSP': SimpleNamespace(),
    },
    valid={'modelV2': True, 'radarState': True},
  )

  planner_sp.publish_longitudinal_plan_sp(sm_last, pm)
  assert 'longitudinalPlanSP' in pm.sent

  msg = pm.sent['longitudinalPlanSP']
  vtsc_pub = float(msg.longitudinalPlanSP.visionTurnSpeedControl.velocity)
  assert vtsc_pub == pytest.approx(float(planner_sp.v_tsc.v_turn), abs=1e-6)


def test_longitudinal_plan_sp_publishes_winding_context(planner_sp):
  class _FakePM:
    def __init__(self):
      self.sent = {}
    def send(self, name, msg) -> None:
      self.sent[name] = msg

  planner_sp.v_tsc._mapd_winding_valid = True
  planner_sp.v_tsc._mapd_winding_level = 4
  planner_sp.v_tsc._mapd_winding_score = 208
  planner_sp.v_tsc._mapd_winding_confidence = 196
  planner_sp.v_tsc._mapd_winding_current_level = 2
  planner_sp.v_tsc._mapd_winding_current_score = 124
  planner_sp.v_tsc._mapd_winding_current_confidence = 180
  planner_sp.v_tsc._mapd_winding_way_count = 3
  planner_sp.v_tsc._winding_context_active = True
  planner_sp.v_tsc._winding_context_level = 4
  planner_sp.v_tsc._winding_context_score = 0.82
  planner_sp.v_tsc._winding_context_confidence = 0.91
  planner_sp.v_tsc._winding_context_source = 'blended'

  from .pipeline_harness import FakeSubMaster, make_model_v2, make_radar_state, make_car_state, make_car_control
  sm_last = FakeSubMaster(
    data={
      'modelV2': make_model_v2(curvature=0.0, v_pred=20.0, confidence=0.95),
      'radarState': make_radar_state(lead_d_rel_m=None),
      'carState': make_car_state(gas_pressed=False),
      'carControl': make_car_control(long_active=True),
      'controlsState': SimpleNamespace(),
      'controlsStateSP': SimpleNamespace(),
    },
    valid={'modelV2': True, 'radarState': True},
  )

  pm = _FakePM()
  planner_sp.publish_longitudinal_plan_sp(sm_last, pm)
  vtsc = pm.sent['longitudinalPlanSP'].longitudinalPlanSP.visionTurnSpeedControl
  assert bool(vtsc.mapWindingValid) is True
  assert int(vtsc.mapWindingLevel) == 4
  assert int(vtsc.mapWindingScore) == 208
  assert int(vtsc.mapWindingConfidence) == 196
  assert int(vtsc.mapWindingCurrentLevel) == 2
  assert int(vtsc.mapWindingCurrentScore) == 124
  assert int(vtsc.mapWindingCurrentConfidence) == 180
  assert int(vtsc.mapWindingWayCount) == 3
  assert bool(vtsc.windingContextActive) is True
  assert int(vtsc.windingContextLevel) == 4
  assert float(vtsc.windingContextScore) == pytest.approx(0.82, abs=1e-6)
  assert float(vtsc.windingContextConfidence) == pytest.approx(0.91, abs=1e-6)
  assert str(vtsc.windingContextSource) == 'blended'


def test_vtsc_does_not_apply_when_long_inactive(planner_sp):
  v0 = 30.0
  v_cruise = 33.0
  steps = [
    Step(curvature=0.01, confidence=0.95, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=False),
  ] * 10
  hist = run_vtsc_min_of_sources(planner_sp, steps=steps, dt=0.05, start_t=0.0)
  assert hist

  last = hist[-1]
  # If longitudinal is inactive, VTSC should not constrain cruise.
  assert last['v_cruise_final'] == pytest.approx(v_cruise, abs=1e-6)


def test_vtsc_recovers_back_to_cruise_after_curve(planner_sp):
  v0 = 28.0
  v_cruise = 33.0
  dt = 0.05

  # Phase 1: curve present, should cap below cruise.
  steps = [Step(curvature=0.01, confidence=0.95, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True)] * 40
  # Phase 2: curve gone, confidence good; cap should wind back up smoothly.
  steps += [Step(curvature=0.0, confidence=0.95, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True)] * 60

  hist = run_vtsc_min_of_sources(planner_sp, steps=steps, dt=dt, start_t=0.0, integrate_ego=True)
  assert len(hist) == len(steps)

  post = hist[40:]
  assert post
  peak_step = max(max(0.0, float(nxt['v_cruise_final']) - float(cur['v_cruise_final'])) for cur, nxt in zip(post, post[1:], strict=False))
  assert peak_step <= 0.25
  assert float(post[-1]['v_cruise_final']) >= float(post[0]['v_cruise_final']) + 3.0


def test_vtsc_recovers_after_occlusion_reacquisition(planner_sp):
  v0 = 25.0
  v_cruise = 30.0
  dt = 0.05

  # Enter occlusion: low confidence for long enough to trigger dwell.
  steps = [Step(curvature=0.008, confidence=0.40, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True)] * 30
  # Curve goes away but confidence is still poor; VTSC may still be conservative.
  steps += [Step(curvature=0.0, confidence=0.40, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True)] * 20
  # Reacquire: confidence returns to full.
  steps += [Step(curvature=0.0, confidence=0.95, v_ego=v0, a_ego=0.0, v_cruise=v_cruise, long_active=True)] * 60

  hist = run_vtsc_min_of_sources(planner_sp, steps=steps, dt=dt, start_t=0.0, integrate_ego=True)
  assert len(hist) == len(steps)

  # Once confidence is back and curvature is zero, v_cruise_final should return to cruise.
  recover_idx = None
  for i in range(50, len(hist)):
    if hist[i]['v_cruise_final'] >= v_cruise - 1e-3:
      recover_idx = i
      break

  assert recover_idx is not None, "VTSC did not recover to cruise after vision reacquisition"
  assert (recover_idx - 50) * dt <= 3.0
