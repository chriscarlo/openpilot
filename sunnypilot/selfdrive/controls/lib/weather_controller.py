#!/usr/bin/env python3
"""
Weather Controller — reads WeatherCondition param from weatherd (Pirate
Weather point-forecast) and computes a speed-cap recommendation for the
longitudinal planner.

The reduction curve is a C1-continuous smoothstep through four anchor
points. Knots removed the piecewise-linear "hard breakpoints" feel of the
previous version: the speed cap now ramps up gently through drizzle, hits
each user-configured reduction at the corresponding precipitation threshold,
and only saturates at red_heavy once the radar observation looks like a
proper downpour.

Follows the same pattern as RTIController: Params → speed_recommendation + is_active.
"""

import json
import time

from openpilot.common.params import Params
from opendbc.car.common.conversions import Conversions as CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET

# Speed reductions stored in mph; converted to m/s at runtime.
MPH_TO_MS = CV.MPH_TO_MS

# Ramp rates (m/s^2). Decel is gentle so severity onset is smooth; accel is
# faster so that when rain clears (or the driver raises cruise) the weather
# cap releases the driver-commanded set speed promptly.
WEATHER_DECEL_RATE = 1.0
WEATHER_ACCEL_RATE = 2.5

# How old weather data can be before we ignore it (seconds)
MAX_DATA_AGE_S = 600  # 10 minutes

# Precipitation intensity anchor points (mm/hour). These are placed deeper
# into each meteorological rain category than the literal thresholds so that
# drizzle doesn't instantly jump to the full "light rain" reduction:
#
#   meteorological "light"   : 0.5 – 2.5 mm/hr  → LIGHT anchor placed at 1.5
#   meteorological "moderate": 2.5 – 7.6 mm/hr  → MODERATE anchor placed at 5.0
#   meteorological "heavy"   : > 7.6 mm/hr      → HEAVY anchor placed at 10.0
#
# Between anchors the reduction is a cubic smoothstep (3t^2 - 2t^3) of the
# two adjacent user-configured reductions. Smoothstep has f'(0)=f'(1)=0,
# which removes the slope discontinuity the old linear interpolation had at
# each knot.
PRECIP_NONE = 0.0
PRECIP_LIGHT = 1.5
PRECIP_MODERATE = 5.0
PRECIP_HEAVY = 10.0

# Minimum precipitation to activate. Below this threshold the smoothstep
# curve would produce a sub-0.1 mph reduction that isn't worth a control
# cycle — reset the controller instead so other constraints take over.
MIN_PRECIP_MM_PER_HR = 0.1


def _lerp(a: float, b: float, t: float) -> float:
  """Linear interpolation: a when t=0, b when t=1, clamped."""
  t = max(0.0, min(1.0, t))
  return a + (b - a) * t


def _smoothstep(t: float) -> float:
  """Cubic Hermite smoothstep. f(0)=0, f(1)=1, f'(0)=f'(1)=0.

  Produces a C1-continuous transition between two anchor values so the
  speed-reduction curve has no slope kinks where segments meet.
  """
  t = max(0.0, min(1.0, t))
  return t * t * (3.0 - 2.0 * t)


def _interpolate_reduction(precip_mm: float,
                           red_none: float, red_light: float,
                           red_moderate: float, red_heavy: float) -> float:
  """Compute a C1-continuous, monotonically-increasing speed reduction (m/s).

  Smoothstep-interpolates between four anchor points:
    0 mm             → red_none     (0 m/s)
    PRECIP_LIGHT     → red_light
    PRECIP_MODERATE  → red_moderate
    PRECIP_HEAVY     → red_heavy    (clamped above)
  """
  if precip_mm <= PRECIP_NONE:
    return red_none
  if precip_mm >= PRECIP_HEAVY:
    return red_heavy
  if precip_mm <= PRECIP_LIGHT:
    t = _smoothstep((precip_mm - PRECIP_NONE) / (PRECIP_LIGHT - PRECIP_NONE))
    return _lerp(red_none, red_light, t)
  if precip_mm <= PRECIP_MODERATE:
    t = _smoothstep((precip_mm - PRECIP_LIGHT) / (PRECIP_MODERATE - PRECIP_LIGHT))
    return _lerp(red_light, red_moderate, t)
  t = _smoothstep((precip_mm - PRECIP_MODERATE) / (PRECIP_HEAVY - PRECIP_MODERATE))
  return _lerp(red_moderate, red_heavy, t)


def _extract_precipitation_mm(condition: dict) -> float:
  return max(0.0, float(condition.get("precipitation_mm", 0.0)))


class WeatherController:
  """Weather-aware speed reduction with continuous precipitation interpolation."""

  def __init__(self):
    self.params = Params()

    # State
    self._enabled = False
    self._is_active = False
    self._speed_recommendation = V_CRUISE_UNSET
    self._severity = "none"

    # Ramp-down state
    self._ramped_speed: float | None = None
    self._last_update_ts: float | None = None

    # Cached user settings (re-read periodically)
    self._red_light = 0.0   # m/s reduction at light anchor
    self._red_moderate = 0.0
    self._red_heavy = 0.0
    self._last_param_read = 0.0
    self._param_read_interval = 10.0  # re-read user prefs every 10s

  def _load_user_params(self) -> None:
    """Load speed-reduction settings from Params (stored as mph integers)."""
    def _read_mph(key: str, default: int) -> float:
      raw = self.params.get(key)
      if raw is not None:
        try:
          return int(raw) * MPH_TO_MS
        except (ValueError, TypeError):
          pass
      return default * MPH_TO_MS

    self._red_light = _read_mph("WeatherSpeedReductionLight", 5)
    self._red_moderate = _read_mph("WeatherSpeedReductionModerate", 10)
    self._red_heavy = _read_mph("WeatherSpeedReductionHeavy", 15)

  def update(self, v_ego: float, v_cruise: float) -> None:
    """Update weather speed recommendation.

    Args:
      v_ego: Current ego velocity in m/s
      v_cruise: Current cruise setpoint in m/s
    """
    self._enabled = self.params.get_bool("WeatherAwareControlEnabled")
    if not self._enabled:
      self._reset()
      return

    # Periodically re-read user reduction settings
    now = time.monotonic()
    if now - self._last_param_read > self._param_read_interval:
      self._load_user_params()
      self._last_param_read = now

    # Read weather condition from weatherd
    raw = self.params.get("WeatherCondition")
    if not raw:
      self._reset()
      return

    try:
      condition = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
      self._reset()
      return

    # Check data freshness
    data_ts = condition.get("timestamp", 0)
    if time.time() - data_ts > MAX_DATA_AGE_S:
      self._reset()
      return

    severity = condition.get("severity", "none")
    self._severity = severity
    precip_mm = _extract_precipitation_mm(condition)

    # Require non-trivial precipitation to activate. The signal is a doppler
    # radar observation at ego position, so a non-zero value means rain is
    # physically falling here right now.
    if precip_mm < MIN_PRECIP_MM_PER_HR:
      self._reset()
      return

    # Compute interpolated reduction from continuous precipitation value
    reduction_ms = _interpolate_reduction(
      precip_mm,
      red_none=0.0,
      red_light=self._red_light,
      red_moderate=self._red_moderate,
      red_heavy=self._red_heavy,
    )

    if reduction_ms <= 0.0:
      self._reset()
      return

    # Compute target speed: cruise minus interpolated reduction
    target_speed = v_cruise - reduction_ms

    # Don't go below a safe minimum (5 mph ~ 2.24 m/s)
    target_speed = max(target_speed, 2.24)

    # Only activate if we'd actually be reducing speed
    if target_speed >= v_cruise:
      self._reset()
      return

    # Smooth ramp-down
    dt = 0.0
    if self._last_update_ts is not None:
      dt = max(0.0, min(1.0, now - self._last_update_ts))
    self._last_update_ts = now

    if self._ramped_speed is None or not self._is_active:
      # Start ramp from current cruise
      self._ramped_speed = v_cruise
    else:
      if self._ramped_speed > target_speed and dt > 0.0:
        max_drop = WEATHER_DECEL_RATE * dt
        self._ramped_speed = max(target_speed, self._ramped_speed - max_drop)
      elif self._ramped_speed < target_speed and dt > 0.0:
        # Severity decreased or driver raised cruise — release the cap at the
        # faster accel rate so we don't artificially hold the driver below
        # their set speed.
        max_rise = WEATHER_ACCEL_RATE * dt
        self._ramped_speed = min(target_speed, self._ramped_speed + max_rise)

    self._speed_recommendation = min(self._ramped_speed, v_cruise)
    self._is_active = True

  def _reset(self) -> None:
    self._is_active = False
    self._speed_recommendation = V_CRUISE_UNSET
    self._severity = "none"
    self._ramped_speed = None

  @property
  def is_active(self) -> bool:
    return self._is_active

  @property
  def speed_recommendation(self) -> float:
    if self._is_active:
      return self._speed_recommendation
    return V_CRUISE_UNSET

  @property
  def severity(self) -> str:
    return self._severity

  @property
  def enabled(self) -> bool:
    return self._enabled
