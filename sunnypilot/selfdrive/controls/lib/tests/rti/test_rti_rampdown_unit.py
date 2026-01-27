#!/usr/bin/env python3
"""
Focused unit tests for RTIController ramp-down and filtering behavior.

This test avoids the repository's global conftest by running with:
  pytest --confcutdir=sunnypilot/selfdrive/controls/lib/tests -q \
         sunnypilot/selfdrive/controls/lib/tests/rti/test_rti_rampdown_unit.py
"""

import sys
import types
import math
from pathlib import Path


def make_stubs():
    # Provide minimal cereal.messaging stub to satisfy type hints
    cereal_pkg = types.ModuleType('cereal')
    sys.modules['cereal'] = cereal_pkg
    msg_stub = types.ModuleType('cereal.messaging')
    # Expose a SubMaster attribute to satisfy annotations
    msg_stub.SubMaster = object
    sys.modules['cereal.messaging'] = msg_stub
    # Also stub cereal.log to satisfy hardware imports during params init
    log_stub = types.ModuleType('cereal.log')
    sys.modules['cereal.log'] = log_stub
    # Stub minimal cereal.car to satisfy cruise imports
    car_stub = types.ModuleType('cereal.car')
    class _ButtonEvent: pass
    class _ButtonEventType: pass
    _ButtonEvent.Type = _ButtonEventType
    class _CarState: pass
    _CarState.ButtonEvent = _ButtonEvent
    car_stub.CarState = _CarState
    sys.modules['cereal.car'] = car_stub

    # Stub minimal opendbc Conversions constants used by logging
    opendbc_pkg = types.ModuleType('opendbc')
    car_pkg = types.ModuleType('opendbc.car')
    common_pkg = types.ModuleType('opendbc.car.common')
    conv_mod = types.ModuleType('opendbc.car.common.conversions')
    class _Conversions:
        MS_TO_KPH = 3.6
        KPH_TO_MS = 1.0 / 3.6
    conv_mod.Conversions = _Conversions
    sys.modules['opendbc'] = opendbc_pkg
    sys.modules['opendbc.car'] = car_pkg
    sys.modules['opendbc.car.common'] = common_pkg
    sys.modules['opendbc.car.common.conversions'] = conv_mod

    # Stub cruise constants module to avoid importing full car.cruise
    cruise_mod = types.ModuleType('openpilot.selfdrive.car.cruise')
    cruise_mod.V_CRUISE_UNSET = 255
    cruise_mod.V_CRUISE_MAX = 145.0
    sys.modules['openpilot.selfdrive.car.cruise'] = cruise_mod

    # Stub swaglog to avoid pulling hardware paths/DeviceState
    swaglog_mod = types.ModuleType('openpilot.common.swaglog')
    class _Logger:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def debug(self, *a, **k): pass
    swaglog_mod.cloudlog = _Logger()
    sys.modules['openpilot.common.swaglog'] = swaglog_mod


def force_params_stub():
    """Force using the lightweight Python Params stub instead of the compiled extension.

    This avoids importing openpilot.common.swaglog and hardware paths during tests.
    """
    import types as _types
    import sys as _sys
    modname = 'openpilot.common.params_pyx'
    if modname in _sys.modules:
        return
    stub_path = Path('openpilot/common/params_pyx.py')
    if stub_path.exists():
        code = stub_path.read_text()
        module = _types.ModuleType(modname)
        exec(compile(code, str(stub_path), 'exec'), module.__dict__)
        _sys.modules[modname] = module


def test_rti_posted_rampdown_and_no_overshoot(monkeypatch):
    make_stubs()
    force_params_stub()
    from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController
    from types import SimpleNamespace

    class FakeThreat:
        def __init__(self, distance, direction, type_name, speed_limit_ms, confidence=0.9):
            self.distance = distance
            self.direction = direction
            self.type = type_name
            self.speedLimitMs = speed_limit_ms
            self.confidence = confidence

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0

    ctrl = RTIController(CP=None)

    # Force params for reproducibility
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIDecelRate', 1.5)  # m/s^2
    ctrl.params.put('RTIThreatFilter', '1')  # Police-only

    # Ego and cruise at ~75 mph -> 33.53 m/s; posted = 65 mph -> 29.06 m/s
    ctrl._v_cruise = 33.53
    v = 33.53
    posted_ms = 29.06
    state = FakeState([FakeThreat(distance=1200, direction='ahead', type_name='police', speed_limit_ms=posted_ms)])

    # Simulate 5 seconds at 10Hz
    dt = 0.1
    sm = SimpleNamespace(valid={'rtiStateSP': True})
    recs = []
    for _ in range(50):
        ctrl._v_ego = v
        ctrl._process_rti_state(state, dt=dt)
        r = ctrl.speed_recommendation
        assert r > 0
        recs.append(r)
        # Update ego speed assuming realized decel roughly follows the requested ramp
        v = max(0.0, v - 1.5 * dt)

    # 1) The final recommendation should be at or very near the posted limit (within small tol)
    assert abs(recs[-1] - posted_ms) < 0.5

    # 2) The recommendation should be monotonically non-increasing until it reaches posted
    def nonincreasing_until_target(seq, target):
        last = seq[0]
        for x in seq[1:]:
            # once we're within tolerance of target, allow small jitter
            if x <= target + 0.3:
                return True
            assert x <= last + 1e-6
            last = x
        return True

    assert nonincreasing_until_target(recs, posted_ms)


def test_rti_filtering_respected(monkeypatch):
    make_stubs()
    force_params_stub()
    from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController
    from types import SimpleNamespace

    class FakeThreat:
        def __init__(self, distance, direction, type_name, speed_limit_ms, confidence=0.9):
            self.distance = distance
            self.direction = direction
            self.type = type_name
            self.speedLimitMs = speed_limit_ms
            self.confidence = confidence

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0

    ctrl = RTIController(CP=None)
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIDecelRate', 1.4)

    # Select Cameras-only (2) and present a police + speed camera ahead
    ctrl.params.put('RTIThreatFilter', '2')

    ctrl._v_cruise = 30.0
    # Ensure v_ego is slightly below v_cruise so the controller can become active on the first step.
    ctrl._v_ego = 29.9
    sm = SimpleNamespace(valid={'rtiStateSP': True})

    police = FakeThreat(500, 'ahead', 'police', 25.0)
    camera = FakeThreat(600, 'ahead', 'speedCamera', 24.0)
    state = FakeState([police, camera])

    # Run a few seconds of updates; only camera should be considered for the base target.
    dt = 0.1
    for _ in range(50):
        ctrl._process_rti_state(state, dt=dt)

    assert ctrl.is_active
    assert ctrl.threat_type == 'speedCamera'
    # Because filter excludes police, the target should ramp down toward the camera's posted limit (24 m/s)
    assert ctrl.speed_recommendation <= 24.0 + 0.3


def test_rti_no_speed_spike_on_direction_transition_jitter(monkeypatch):
    """Regression: avoid brief RTI dropouts when threat direction becomes ambiguous at the alert location.

    Real-world symptom: as ego passes directly over the threat coordinate, bearing math can briefly classify
    the threat as left/right instead of ahead/behind. If RTI drops for a cycle, the planner falls back to
    v_cruise_console, causing a momentary speed increase. Controller should stay active through the jitter.
    """
    make_stubs()
    force_params_stub()
    from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController

    class FakeThreat:
        def __init__(self, threat_id, distance, direction, type_name, speed_limit_ms, confidence=0.9):
            self.id = threat_id
            self.distance = distance
            self.direction = direction
            self.type = type_name
            self.speedLimitMs = speed_limit_ms
            self.confidence = confidence

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0

    ctrl = RTIController(CP=None)
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIDecelRate', 1.5)  # m/s^2
    ctrl.params.put('RTIThreatFilter', '1')  # Police-only

    # Driver setpoint (v_cruise_console) > ego (common steady-state condition)
    ctrl._v_cruise = 33.53  # ~75 mph in m/s
    ctrl._v_ego = 33.0
    posted_ms = 29.06  # ~65 mph in m/s

    threat = FakeThreat(threat_id="police-001", distance=200.0, direction='ahead', type_name='police', speed_limit_ms=posted_ms)
    state = FakeState([threat])

    # Approaching: RTI should be active
    ctrl._process_rti_state(state, dt=0.1)
    assert ctrl.is_active
    assert ctrl.speed_recommendation != 255  # V_CRUISE_UNSET (stub)

    # At / passing: direction briefly jitters sideways (right/left). Keep slowing anyway.
    threat.distance = 1.0
    threat.direction = 'right'
    ctrl._process_rti_state(state, dt=0.1)
    assert ctrl.is_active, "RTI should not drop out at the alert location"
    assert ctrl.speed_recommendation != 255

    # After passing: behind within resume distance should also stay active
    threat.distance = 10.0
    threat.direction = 'behind'
    ctrl._process_rti_state(state, dt=0.1)
    assert ctrl.is_active
    assert ctrl.speed_recommendation != 255
