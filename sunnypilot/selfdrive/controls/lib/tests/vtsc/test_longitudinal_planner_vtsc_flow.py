#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace
import importlib
import sys

import pytest

from openpilot.common.params import Params
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, LongControl

from .pipeline_harness import (
  FakeSubMaster,
  install_fake_long_mpc,
  make_car_control,
  make_model_v2,
  make_radar_state,
)


@pytest.fixture(autouse=True)
def _isolate_vision_flow_from_persistent_map_lookahead():
  """These tests exercise the vision-to-planner path, not persistent MTSC input."""
  params = Params()
  previous = params.get('MTSCLookaheadEnabled')
  params.put_bool('MTSCLookaheadEnabled', False)
  try:
    yield
  finally:
    if previous is None:
      params.remove('MTSCLookaheadEnabled')
    else:
      params.put('MTSCLookaheadEnabled', previous)


@pytest.fixture(autouse=True)
def _restore_hijacked_modules():
  """install_fake_long_mpc replaces the real long_mpc in sys.modules and the
  planner is then re-imported bound to the fake. Without restoration, any
  later-collected suite that imports the real module surface (e.g. the
  longitudinal harness importing get_safe_obstacle_distance) gets the fake and
  errors — visible as order-dependent collection failures under
  pytest-randomly. Snapshot both modules and put the originals back."""
  names = (
    'openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc',
    'openpilot.selfdrive.controls.lib.longitudinal_planner',
  )
  saved = {name: sys.modules.get(name) for name in names}
  try:
    yield
  finally:
    for name, mod in saved.items():
      if mod is not None:
        sys.modules[name] = mod
      else:
        sys.modules.pop(name, None)


class _NoOpSLC:
  def __init__(self, CP):
    pass
  def update(self, *args, **kwargs) -> None:
    pass
  @property
  def is_active(self) -> bool:
    return False
  @property
  def speed_limit_offseted(self) -> float:
    return 0.0
  @property
  def state(self) -> int:
    return 0
  @property
  def is_enabled(self) -> bool:
    return False
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
  # Fields used by LongitudinalPlanner.update
  openpilotLongitudinalControl = True
  longitudinalActuatorDelay = 0.15
  vEgoStopping = 0.25
  notCar = False


class _MockLongControlCP:
  # Minimal CP used by LongControl in tests.
  vEgoStarting = 0.5
  startingState = False
  stopAccel = -2.0
  stoppingDecelRate = 0.2
  startAccel = 1.0
  longitudinalTuning = SimpleNamespace(
    kpBP=[0.0],
    kpV=[0.0],
    kiBP=[0.0],
    kiV=[0.0],
    kf=1.0,  # feed-forward makes output≈a_target for stable tests
  )


def _import_longitudinal_planner_module(monkeypatch):
  install_fake_long_mpc()

  # Import LongitudinalPlannerSP module first so we can patch its dependencies before the main planner imports it.
  sp_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner')

  # Patch out non-VTSC controllers to keep inputs minimal.
  monkeypatch.setattr(sp_mod, 'SpeedLimitController', _NoOpSLC, raising=True)
  monkeypatch.setattr(sp_mod, 'RTIController', _NoOpRTI, raising=True)
  monkeypatch.setattr(sp_mod, 'DynamicExperimentalController', _NoOpDEC, raising=True)
  monkeypatch.setattr(sp_mod, 'VibePersonalityController', _NoOpVibe, raising=True)

  # Now import the main planner module. It imports LongitudinalPlannerSP and LongitudinalMpc at import-time.
  sys.modules.pop('openpilot.selfdrive.controls.lib.longitudinal_planner', None)
  main_mod = importlib.import_module('openpilot.selfdrive.controls.lib.longitudinal_planner')
  return main_mod


def _mk_sm(*,
           curvature: float,
           curvature_ahead: float | None,
           confidence: float,
           v_ego: float,
           a_ego: float,
           v_cruise_mps: float,
           long_active: bool,
           lead_d_rel_m: float | None = None) -> object:
  # carState.vCruise is in kph in this fork's planner.
  car_state = SimpleNamespace(
    vEgo=float(v_ego),
    aEgo=float(a_ego),
    standstill=bool(v_ego < 0.01),
    vCruise=float(v_cruise_mps * CV.MS_TO_KPH),
    gasPressed=False,
    brakePressed=False,
    cruiseState=SimpleNamespace(standstill=False),
  )
  controls_state = SimpleNamespace(
    longControlState=LongCtrlState.pid,
    forceDecel=False,
  )
  selfdrive_state = SimpleNamespace(
    experimentalMode=False,
    personality=0,
    enabled=True,
  )
  # VTSC uses modelV2.orientationRate.z + velocity.x + laneLineProbs
  model = make_model_v2(curvature=curvature, curvature_ahead=curvature_ahead, v_pred=v_ego, confidence=confidence)
  return FakeSubMaster(
    data={
      'modelV2': model,
      'radarState': make_radar_state(lead_d_rel_m=lead_d_rel_m),
      'carState': car_state,
      'carControl': make_car_control(long_active=long_active),
      'controlsState': controls_state,
      'selfdriveState': selfdrive_state,
      'liveParameters': SimpleNamespace(angleOffsetDeg=0.0),
    },
    valid={'modelV2': True, 'radarState': True},
  )


def test_vtsc_cap_enters_mpc_as_min_of_sources(monkeypatch):
  mod = _import_longitudinal_planner_module(monkeypatch)

  # Ensure VTSC enabled.
  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  planner = mod.LongitudinalPlanner(_MockCP(), init_v=30.0, init_a=0.0)

  # Grab fake MPC instance and ensure it exposes last_v_cruise.
  assert hasattr(planner.mpc, 'last_v_cruise')

  dt = 0.05
  t = 0.0

  # Phase 1: curve active -> VTSC cap below cruise -> expect decel.
  v_ego = 30.0
  a_ego = 0.0
  v_cruise = 33.0
  for _ in range(40):
    sm = _mk_sm(curvature=0.01, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise, long_active=True)
    vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
    from unittest.mock import patch
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)
    # Update dynamics (simple integration of commanded accel)
    a_ego = float(planner.output_a_target)
    v_ego = max(0.0, v_ego + a_ego * dt)
    t += dt

  # The MPC should receive the min-of cruise speeds; with SLC/RTI disabled that's min(cruise, vtsc).
  vtsc_v = float(planner.v_tsc.v_turn) if getattr(planner.v_tsc, 'is_active', False) else v_cruise
  assert planner.mpc.last_v_cruise == pytest.approx(min(v_cruise, vtsc_v), abs=1e-6)
  assert float(planner.output_a_target) <= 1e-3

  # If VTSC continues to limit, MPC should still receive the exact min-of source value.
  # This assertion is specifically about "ingestion correctness" (no unit mismatch, no stale value).
  assert planner.mpc.last_v_cruise == pytest.approx(min(v_cruise, float(planner.v_tsc.v_turn)), abs=1e-6)


def test_vtsc_release_recovers_mpc_to_cruise(monkeypatch):
  mod = _import_longitudinal_planner_module(monkeypatch)

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  planner = mod.LongitudinalPlanner(_MockCP(), init_v=30.0, init_a=0.0)
  assert hasattr(planner.mpc, 'last_v_cruise')

  dt = 0.05
  t = 0.0
  v_ego = 30.0
  a_ego = 0.0
  v_cruise = 33.0
  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
  from unittest.mock import patch

  # Induce a cap for ~2s.
  for _ in range(40):
    sm = _mk_sm(curvature=0.01, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise, long_active=True)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)
    a_ego = float(planner.output_a_target)
    v_ego = max(0.0, v_ego + a_ego * dt)
    t += dt

  # Clear the curve and keep good confidence; expect cap to return to cruise.
  for _ in range(120):  # 6s window
    sm = _mk_sm(curvature=0.0, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise, long_active=True)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)
    a_ego = float(planner.output_a_target)
    v_ego = max(0.0, v_ego + a_ego * dt)
    t += dt

  assert planner.mpc.last_v_cruise == pytest.approx(v_cruise, abs=1e-3)


def test_vtsc_release_recovers_actuator_accel(monkeypatch):
  # Full-ish chain test:
  # VTSC -> min-of cruise -> MPC input -> planner.output_a_target -> LongControl -> actuator accel.
  #
  # This guards the user-facing failure mode:
  # - VTSC decelerates for a curve, but then "refuses to return" / won't accelerate after the curve ends.
  mod = _import_longitudinal_planner_module(monkeypatch)

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  planner = mod.LongitudinalPlanner(_MockCP(), init_v=30.0, init_a=0.0)
  assert hasattr(planner.mpc, 'last_v_cruise')

  long_control = LongControl(_MockLongControlCP())

  dt = 0.05
  t = 0.0
  v_ego = 30.0
  a_ego = 0.0
  v_cruise = 33.0

  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
  from unittest.mock import patch

  saw_decel = False
  saw_accel_after_release = False

  # Phase 1: curve active -> expect decel.
  for _ in range(40):  # 2.0s
    sm = _mk_sm(curvature=0.01, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise, long_active=True)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)

    a_target = float(planner.output_a_target)
    act = float(long_control.update(True, sm['carState'], a_target, should_stop=False, accel_limits=(-3.5, 2.0)))

    saw_decel = saw_decel or (act < -0.05)
    a_ego = act
    v_ego = max(0.0, v_ego + a_ego * dt)
    t += dt

  assert planner.mpc.last_v_cruise < v_cruise - 1e-3
  assert saw_decel, "Expected some deceleration while VTSC cap is active"

  # Phase 2: curve gone -> expect cap release to cruise and positive accel at some point.
  for _ in range(120):  # 6.0s window
    sm = _mk_sm(curvature=0.0, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise, long_active=True)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)

    a_target = float(planner.output_a_target)
    act = float(long_control.update(True, sm['carState'], a_target, should_stop=False, accel_limits=(-3.5, 2.0)))

    if planner.mpc.last_v_cruise >= v_cruise - 1e-3:
      saw_accel_after_release = saw_accel_after_release or (act > 0.05)

    a_ego = act
    v_ego = max(0.0, v_ego + a_ego * dt)
    t += dt

  assert planner.mpc.last_v_cruise == pytest.approx(v_cruise, abs=1e-3)
  assert saw_accel_after_release, "Expected positive actuator accel after VTSC released back to cruise"


def test_vtsc_curve_exit_stays_on_normal_path_end_to_end(monkeypatch):
  # End-to-end regression after removing occlusion:
  # low-confidence curve exit should never latch FOV occlusion or any lead-bypass path,
  # but the planner should still accelerate once the cap releases.
  mod = _import_longitudinal_planner_module(monkeypatch)

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)
  p.put_bool('VisionTurnSpeedControlOcclBypassWithLead', True)
  p.put('VisionTurnSpeedControlOcclBypassHeadwayS', 3.0)

  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
  from unittest.mock import patch

  dt = 0.05
  v_cruise = 30.0
  conf = 0.55
  headway_s = 2.0
  n_curve = 10
  n_straight = 320
  k_curve = 0.012
  k_ahead = 0.0

  def run(*, with_lead: bool) -> dict[str, object]:
    planner = mod.LongitudinalPlanner(_MockCP(), init_v=25.0, init_a=0.0)
    long_control = LongControl(_MockLongControlCP())

    t = 0.0
    v_ego = 25.0
    a_ego = 0.0

    fov_latched = False
    fov_cleared_idx: int | None = None
    bypass_seen = False
    saw_positive_accel_after_clear = False

    for i in range(n_curve + n_straight):
      if i < n_curve:
        curv = k_curve
        curv_ahead = None
      else:
        curv = 0.0
        curv_ahead = k_ahead

      lead_d = None
      if with_lead:
        lead_d = headway_s * max(0.1, float(v_ego))

      sm = _mk_sm(
        curvature=curv,
        curvature_ahead=curv_ahead,
        confidence=conf,
        v_ego=v_ego,
        a_ego=a_ego,
        v_cruise_mps=v_cruise,
        long_active=True,
        lead_d_rel_m=lead_d,
      )

      with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
        planner.update(sm)

      fov_now = bool(getattr(planner.v_tsc, '_fov_occluded', False))
      if i < n_curve:
        fov_latched = fov_latched or fov_now
      elif fov_cleared_idx is None and (not fov_now):
        fov_cleared_idx = i
      bypass_seen = bypass_seen or bool(getattr(planner.v_tsc, '_occl_lead_bypass_active', False))

      a_target = float(planner.output_a_target)
      act = float(long_control.update(True, sm['carState'], a_target, should_stop=False, accel_limits=(-3.5, 2.0)))

      if (fov_cleared_idx is not None) and (act > 0.05):
        saw_positive_accel_after_clear = True

      a_ego = act
      v_ego = max(0.0, v_ego + a_ego * dt)
      t += dt

    return {
      'fov_latched': fov_latched,
      'fov_cleared_idx': fov_cleared_idx,
      'bypass_seen': bypass_seen,
      'saw_positive_accel_after_clear': saw_positive_accel_after_clear,
    }

  out_no_lead = run(with_lead=False)
  assert out_no_lead['fov_latched'] is False
  assert out_no_lead['bypass_seen'] is False
  assert out_no_lead['fov_cleared_idx'] is not None
  assert int(out_no_lead['fov_cleared_idx']) == n_curve
  assert out_no_lead['saw_positive_accel_after_clear'] is True

  out_lead = run(with_lead=True)
  assert out_lead['fov_latched'] is False
  assert out_lead['bypass_seen'] is False
  assert out_lead['fov_cleared_idx'] is not None
  assert int(out_lead['fov_cleared_idx']) == n_curve
  assert out_lead['saw_positive_accel_after_clear'] is True

  assert abs(int(out_lead['fov_cleared_idx']) - int(out_no_lead['fov_cleared_idx'])) == 0


def test_throttle_prob_gate_can_prevent_accel_after_vtsc_release(monkeypatch):
  # Pipeline deviation test (important for debugging "won't return to speed"):
  #
  # Even if VTSC winds its cap back up enough to permit acceleration, the main longitudinal planner can still
  # refuse to accelerate if `allow_throttle` is false. This happens when model
  # `meta.disengagePredictions.gasPressProbs[1]` is low (throttle_prob <= 0.4) at speed.
  #
  # This test makes the non-VTSC confounder explicit, so future regressions can be triaged quickly.
  mod = _import_longitudinal_planner_module(monkeypatch)

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
  from unittest.mock import patch

  dt = 0.05
  # Pick a cruise delta large enough that the fake MPC requests a noticeable positive accel,
  # so the `allow_throttle` clip is observable in this synthetic environment.
  v_cruise = 40.0
  n_curve = 10
  n_straight = 320

  def run(throttle_prob: float) -> dict[str, object]:
    planner = mod.LongitudinalPlanner(_MockCP(), init_v=25.0, init_a=0.0)

    t = 0.0
    # Keep ego dynamics fixed in this test: we want to isolate the planner's accel clipping behavior,
    # not the full closed-loop response. (Closed-loop tests already exist elsewhere in this suite.)
    v_ego = 25.0
    a_ego = 0.0

    saw_release = False
    accel_after_release: list[float] = []
    allow_throttle_seen = True

    for i in range(n_curve + n_straight):
      curv = 0.01 if i < n_curve else 0.0
      sm = _mk_sm(curvature=curv, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego,
                  v_cruise_mps=v_cruise, long_active=True)
      # Control allow_throttle via model meta (see LongitudinalPlanner.parse_model()).
      try:
        sm['modelV2'].meta.disengagePredictions.gasPressProbs[1] = float(throttle_prob)
      except Exception:
        pass

      with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
        planner.update(sm)

      allow_throttle_seen = allow_throttle_seen and bool(getattr(planner, 'allow_throttle', True))

      # Once curvature is zero, VTSC should eventually wind its cap high enough that throttle is relevant.
      if i >= n_curve and float(getattr(planner.v_tsc, 'v_turn', 0.0)) >= v_ego + 0.5:
        saw_release = True
        accel_after_release.append(float(getattr(planner, 'output_a_target', 0.0)))
      t += dt

    return {
      'saw_release': saw_release,
      'accel_after_release': accel_after_release,
      'allow_throttle_seen': allow_throttle_seen,
    }

  # Baseline: with throttle allowed, once VTSC has wound its cap high enough we should see positive accel.
  ok = run(throttle_prob=1.0)
  assert ok['saw_release'] is True
  accel_ok = [float(a) for a in (ok['accel_after_release'] or [])]
  assert accel_ok, "Expected to record accel samples after VTSC release shaping"
  assert accel_ok[-1] > 0.05

  # Confounder: with throttle forbidden, even after VTSC releases, planner accel must stay at/below coast.
  blocked = run(throttle_prob=0.0)
  assert blocked['saw_release'] is True
  assert blocked['allow_throttle_seen'] is False
  accel_blocked = [float(a) for a in (blocked['accel_after_release'] or [])]
  assert accel_blocked, "Expected to record accel samples after VTSC release"
  # End of run should be meaningfully lower than the throttle-allowed case and should settle at/below coast.
  assert accel_blocked[-1] <= accel_ok[-1] - 0.20
  assert accel_blocked[-1] <= -0.05
  # NOTE: In this synthetic harness, accel envelope smoothing dominates and the clamp effect is
  # gradual; verify that low throttle_prob flips the gate and drives a monotonic ramp-down.
  tail_diffs = [b - a for a, b in zip(accel_blocked[-10:], accel_blocked[-9:])]
  assert all(d <= 1e-6 for d in tail_diffs)


def test_mpc_cruise_clipping_softens_large_vtsc_step_down(monkeypatch):
  # Confounder / sanity check:
  #
  # Even when VTSC correctly drives `v_cruise_final` down (min-of sources), the long MPC
  # internally clips the cruise profile inside a decel envelope (`v_lower`/`v_upper`).
  #
  # That means a sharp VTSC cap step-down (e.g., 33 m/s -> 12 m/s) cannot be represented
  # instantly at the front of the MPC horizon; early-horizon points remain near v_ego and
  # only converge to the new cap over time.
  #
  # This test prevents mis-triage: "VTSC asked for X but the car didn't immediately do X"
  # is often an MPC envelope effect, not a VTSC ingestion bug.
  mod = _import_longitudinal_planner_module(monkeypatch)

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  planner = mod.LongitudinalPlanner(_MockCP(), init_v=30.0, init_a=0.0)

  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
  from unittest.mock import patch

  dt = 0.05
  t = 0.0
  v_ego = 30.0
  a_ego = 0.0
  v_cruise = 33.0

  # Use a tight curve with good confidence to produce a large VTSC cap step-down.
  for _ in range(8):
    sm = _mk_sm(curvature=0.012, curvature_ahead=None, confidence=0.95, v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise, long_active=True)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)
    t += dt

  vtsc_cap = float(getattr(planner.v_tsc, 'v_turn', v_cruise))
  assert vtsc_cap < v_cruise - 5.0, "Scenario did not generate a meaningful VTSC step-down"

  # Planner passes the min-of sources into MPC.
  assert planner.mpc.last_v_cruise == pytest.approx(min(v_cruise, vtsc_cap), abs=1e-6)

  # But MPC's *internal* cruise envelope clips the early horizon points.
  v_clip = getattr(planner.mpc, 'last_v_cruise_clipped', None)
  assert v_clip is not None
  assert len(v_clip) > 2
  # At horizon start, envelope forces cruise to equal the MPC's current state speed
  # (cannot instantaneously change). We use the stored envelope for this assertion to avoid
  # coupling to the planner's internal filtering of v_ego.
  v_lower = getattr(planner.mpc, 'last_v_lower', None)
  assert v_lower is not None
  assert float(v_clip[0]) == pytest.approx(float(v_lower[0]), abs=1e-6)
  # The clipped profile must eventually reach the requested cruise cap.
  assert float(min(v_clip)) == pytest.approx(float(planner.mpc.last_v_cruise), abs=1e-6)
  # Early horizon should still be materially above the VTSC cap (demonstrating the softening).
  assert float(v_clip[5]) >= float(planner.mpc.last_v_cruise) + 5.0
