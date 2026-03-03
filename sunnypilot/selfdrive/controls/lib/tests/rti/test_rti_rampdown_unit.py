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
    """Force using a lightweight Params stub instead of the compiled extension.

    This test suite is intended to run without compiled extensions or full runtime deps.
    """
    import types as _types
    import sys as _sys

    modname = 'openpilot.common.params_pyx'
    if modname in _sys.modules:
        return

    try:
        __import__(modname)
        return
    except Exception:
        pass

    module = _types.ModuleType(modname)

    class UnknownKeyName(Exception):
        pass

    class ParamKeyFlag:
        pass

    class ParamKeyType:
        pass

    class Params:
        def __init__(self):
            self._vals = {}

        def check_key(self, key):
            return True

        def get(self, key, block=False):
            val = self._vals.get(key, None)
            if val is None:
                return None
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
            if isinstance(val, str):
                return val.encode('utf-8')
            return val

        def put(self, key, val):
            self._vals[key] = val

        def get_bool(self, key):
            val = self._vals.get(key, False)
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = val.decode('utf-8')
                except Exception:
                    return bool(val)
            if isinstance(val, str):
                return val.strip().lower() in ("1", "true", "t", "yes", "y", "on")
            return bool(val)

        def put_bool(self, key, val):
            self._vals[key] = bool(val)

    module.Params = Params
    module.ParamKeyFlag = ParamKeyFlag
    module.ParamKeyType = ParamKeyType
    module.UnknownKeyName = UnknownKeyName
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


def test_rti_ignores_offroad_threats_even_if_ahead(monkeypatch):
    make_stubs()
    force_params_stub()
    from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController

    class FakeThreat:
        def __init__(self, distance, direction, type_name, speed_limit_ms, on_same_road, confidence=0.9):
            self.id = "threat-offroad"
            self.distance = distance
            self.direction = direction
            self.type = type_name
            self.speedLimitMs = speed_limit_ms
            self.onSameRoad = on_same_road
            self.confidence = confidence

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0

    ctrl = RTIController(CP=None)
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIThreatFilter', '1')  # police

    ctrl._v_cruise = 33.53
    ctrl._v_ego = 30.0

    # Ahead + close + matching type, but explicitly off-road.
    threat = FakeThreat(200.0, 'ahead', 'police', 25.0, on_same_road=False)
    state = FakeState([threat])
    ctrl._process_rti_state(state, dt=0.1)

    assert not ctrl.is_active
    assert ctrl.speed_recommendation == 255  # V_CRUISE_UNSET in stub


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


def test_rti_activates_when_v_ego_equals_v_cruise_in_posted_mode(monkeypatch):
    """Regression: RTI must begin slowing even when v_ego == v_cruise.

    Real-world symptom: when cruising exactly at the set speed (common with SLC),
    RTI never becomes active because it only "activates" once the recommendation
    is strictly below v_cruise, but the initial ramp starts at v_cruise.
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
    ctrl.params.put('RTIThreatFilter', '1')  # police
    ctrl.params.put('RTIDecelRate', 1.5)

    # Steady-state: ego exactly at cruise (no lead, flat road)
    v_cruise = 31.29  # ~70 mph in m/s
    posted_ms = 29.06  # ~65 mph in m/s
    ctrl._v_cruise = v_cruise
    ctrl._v_ego = v_cruise

    threat = FakeThreat("police-002", distance=500.0, direction='ahead', type_name='police', speed_limit_ms=posted_ms)
    state = FakeState([threat])

    # Even without ego speed already dropping, RTI should activate and start ramping down.
    ctrl._process_rti_state(state, dt=0.1)
    assert ctrl.is_active, "RTI should activate when posted limit < v_cruise, even if v_ego == v_cruise"
    assert ctrl.speed_recommendation != 255  # V_CRUISE_UNSET (stub)


def test_rti_uses_slc_posted_limit_when_alert_missing_speed_limit(monkeypatch):
    """RTI should pin to SLC's posted limit even when alert payload has no speedLimit."""
    make_stubs()
    force_params_stub()
    from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController

    class FakeThreat:
        def __init__(self, threat_id, distance, direction, type_name, confidence=0.9):
            self.id = threat_id
            self.distance = distance
            self.direction = direction
            self.type = type_name
            self.confidence = confidence
            self.speedLimitMs = 0.0  # simulate "no speedLimit" in alert

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0  # posted mode in rtid would emit 0 when no speed limit is available

    ctrl = RTIController(CP=None)
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIThreatFilter', '1')  # police
    ctrl.params.put('RTIDecelRate', 1.5)

    v_cruise = 31.29  # ~70 mph in m/s (e.g., speed limit + offset)
    posted_ms = 29.06  # ~65 mph in m/s (actual posted limit from SLC source)

    # What planner passes in via RTIController.update(... posted_speed_limit=slc.speed_limit)
    ctrl._posted_speed_limit = posted_ms
    ctrl._v_cruise = v_cruise
    ctrl._v_ego = v_cruise

    threat = FakeThreat("police-003", distance=400.0, direction='ahead', type_name='police')
    state = FakeState([threat])

    # First step (dt=0 is common right after activation), controller should still go active.
    ctrl._process_rti_state(state, dt=0.0)
    assert ctrl.is_active
    assert ctrl.speed_recommendation != 255  # V_CRUISE_UNSET (stub)

    # Then it should ramp down toward posted limit even if ego speed hasn't started dropping yet.
    for _ in range(50):  # 5 seconds at 10Hz
        ctrl._v_ego = v_cruise
        ctrl._process_rti_state(state, dt=0.1)

    assert abs(ctrl.speed_recommendation - posted_ms) < 0.5


def test_rti_stays_inactive_when_threat_outside_user_windows(monkeypatch):
    """RTI must not constrain cruise when threats are outside approach/resume windows."""
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
            self.onSameRoad = True

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0

    ctrl = RTIController(CP=None)
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIThreatFilter', '1')  # police
    ctrl.params.put('RTIForwardSlowdownRange', 200)  # meters
    ctrl.params.put('RTIResumeSpeedDistance', 150)   # meters
    ctrl._load_user_params()

    v_cruise = 31.29  # ~70 mph in m/s
    posted_ms = 26.82  # ~60 mph in m/s
    ctrl._posted_speed_limit = posted_ms
    ctrl._v_cruise = v_cruise
    ctrl._v_ego = v_cruise

    # Both directions are outside their respective user windows.
    ahead_outside = FakeThreat("police-ahead", distance=260.0, direction='ahead', type_name='police', speed_limit_ms=0.0)
    behind_outside = FakeThreat("police-behind", distance=220.0, direction='behind', type_name='police', speed_limit_ms=0.0)

    ctrl._process_rti_state(FakeState([ahead_outside, behind_outside]), dt=0.1)
    assert not ctrl.is_active
    assert ctrl.speed_recommendation == 255  # V_CRUISE_UNSET in stub


def test_rti_deactivates_after_leaving_user_windows(monkeypatch):
    """RTI should release control once a threat exits approach/resume distance windows."""
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
            self.onSameRoad = True

    class FakeState:
        def __init__(self, threats):
            self.threats = threats
            self.recommendedSpeed = 0.0

    ctrl = RTIController(CP=None)
    ctrl.params.put_bool('RTIEnabled', True)
    ctrl.params.put('RTISpeedReductionMode', 'posted')
    ctrl.params.put('RTIThreatFilter', '1')  # police
    ctrl.params.put('RTIForwardSlowdownRange', 200)  # meters
    ctrl.params.put('RTIResumeSpeedDistance', 150)   # meters
    ctrl._load_user_params()

    v_cruise = 31.29
    posted_ms = 26.82
    ctrl._posted_speed_limit = posted_ms
    ctrl._v_cruise = v_cruise
    ctrl._v_ego = v_cruise

    threat = FakeThreat("police-window", distance=180.0, direction='ahead', type_name='police', speed_limit_ms=0.0)

    # Inside approach window -> RTI should activate.
    ctrl._process_rti_state(FakeState([threat]), dt=0.1)
    assert ctrl.is_active
    assert ctrl.speed_recommendation != 255

    # Move outside both windows -> RTI should release and stop constraining cruise.
    threat.distance = 260.0
    threat.direction = 'ahead'
    ctrl._process_rti_state(FakeState([threat]), dt=0.1)
    assert not ctrl.is_active
    assert ctrl.speed_recommendation == 255
