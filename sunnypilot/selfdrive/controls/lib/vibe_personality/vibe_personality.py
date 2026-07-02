"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import time

from cereal import log, custom
import numpy as np
from openpilot.common.params import Params

LongPersonality = log.LongitudinalPersonality
AccelPersonality = custom.LongitudinalPlanSP.AccelerationPersonality

LONG_PERSONALITIES = [LongPersonality.relaxed, LongPersonality.standard, LongPersonality.aggressive]
ACCEL_PERSONALITIES = [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]

LONG_MODE_NAMES = {
  LongPersonality.relaxed: "Relaxed",
  LongPersonality.standard: "Standard",
  LongPersonality.aggressive: "Aggressive",
}

ACCEL_MODE_NAMES = {
  AccelPersonality.eco: "Eco",
  AccelPersonality.normal: "Normal",
  AccelPersonality.sport: "Sport",
}

# Acceleration Profiles mapped to AccelPersonality (eco/normal/sport)
DEFAULT_MAX_ACCEL_PROFILES = {
  AccelPersonality.eco:       (1.10, 1.00, 0.85, 0.76, 0.58, 0.46, 0.365, 0.317, 0.089),
  AccelPersonality.normal:    (2.00, 2.00, 1.42, 1.10, 0.65, 0.56, 0.43, 0.36, 0.12),
  AccelPersonality.sport:     (4.00, 4.00, 3.80, 3.50, 2.00, 1.75, 1.325, 1.15, 0.50),
}
MAX_ACCEL_PROFILES = DEFAULT_MAX_ACCEL_PROFILES
MAX_ACCEL_BREAKPOINTS = [0.0, 6.0, 9.0, 11.0, 16.0, 20.0, 25.0, 30.0, 55.0]

# Braking profiles mapped to LongPersonality (relaxed/standard/aggressive)
DEFAULT_MIN_ACCEL_PROFILES = {
  LongPersonality.relaxed:    (-0.50, -0.80, -1.20, -1.20),
  LongPersonality.standard:   (-1.05, -1.15, -1.30, -1.30),
  LongPersonality.aggressive: (-1.10, -1.25, -1.40, -1.40),
}
MIN_ACCEL_PROFILES = DEFAULT_MIN_ACCEL_PROFILES
MIN_ACCEL_BREAKPOINTS = [0.0, 10.0, 25.0, 50.0]

# Following Distance Profiles mapped to LongPersonality (relaxed/standard/aggressive)
# EV6 retune: three distinct banks, relaxed tops at 1.8s; standard/aggressive tighter.
# Anchors are 0/44/50/90 mph; kept in sync with VibeTune.Follow.* defaults in params_keys.h.
DEFAULT_FOLLOW_DISTANCE_PROFILES = {
  LongPersonality.relaxed: {
    'x_vel':  [0.0, 19.7, 22.2, 40.0],
    'y_dist': (1.60, 1.65, 1.70, 1.80),
  },
  LongPersonality.standard: {
    'x_vel':  [0.0, 19.7, 22.2, 40.0],
    'y_dist': (1.45, 1.50, 1.55, 1.65),
  },
  LongPersonality.aggressive: {
    'x_vel':  [0.0, 19.7, 22.2, 40.0],
    'y_dist': (1.30, 1.35, 1.40, 1.50),
  },
}
FOLLOW_DISTANCE_PROFILES = DEFAULT_FOLLOW_DISTANCE_PROFILES


def _make_follow_param_key(personality: int, idx: int) -> str:
  return f"VibeTune.Follow.{LONG_MODE_NAMES[personality]}.Headway{idx}"


def _make_brake_param_key(personality: int, idx: int) -> str:
  return f"VibeTune.Brake.{LONG_MODE_NAMES[personality]}.Decel{idx}"


def _make_accel_param_key(personality: int, idx: int) -> str:
  return f"VibeTune.Accel.{ACCEL_MODE_NAMES[personality]}.Max{idx}"


FOLLOW_DISTANCE_PARAM_KEYS = {
  personality: tuple(_make_follow_param_key(personality, idx) for idx in range(len(DEFAULT_FOLLOW_DISTANCE_PROFILES[personality]['y_dist'])))
  for personality in LONG_PERSONALITIES
}

MIN_ACCEL_PARAM_KEYS = {
  personality: tuple(_make_brake_param_key(personality, idx) for idx in range(len(DEFAULT_MIN_ACCEL_PROFILES[personality])))
  for personality in LONG_PERSONALITIES
}

MAX_ACCEL_PARAM_KEYS = {
  personality: tuple(_make_accel_param_key(personality, idx) for idx in range(len(DEFAULT_MAX_ACCEL_PROFILES[personality])))
  for personality in ACCEL_PERSONALITIES
}

PARAM_REFRESH_S = 1.0  # 1 Hz param refresh — profiles change rarely


class VibePersonalityController:
  """
  Controller for managing separated acceleration and distance controls:
  - AccelPersonality controls acceleration behavior (eco, normal, sport)
  - LongPersonality controls braking and following distance (relaxed, standard, aggressive)
  """

  def __init__(self):
    self.params = Params()
    self._last_param_refresh_t = 0.0

    # Separate personalities for acceleration and distance control
    self.accel_personality = AccelPersonality.normal
    self.long_personality = LongPersonality.standard

    # Cached toggle states (refreshed by _update_from_params)
    self._enabled = True
    self._accel_enabled = True
    self._follow_enabled = True

    # Parameter keys
    self.param_keys = {
      'accel_personality': 'AccelPersonality',        # eco=0, normal=1, sport=2
      'long_personality': 'LongitudinalPersonality',  # aggressive=0, standard=1, relaxed=2
      'enabled': 'VibePersonalityEnabled',
      'accel_enabled': 'VibeAccelPersonalityEnabled',
      'follow_enabled': 'VibeFollowPersonalityEnabled'
    }

    self.max_accel_profiles = {personality: tuple(values) for personality, values in DEFAULT_MAX_ACCEL_PROFILES.items()}
    self.min_accel_profiles = {personality: tuple(values) for personality, values in DEFAULT_MIN_ACCEL_PROFILES.items()}
    self.follow_distance_profiles = {
      personality: {
        'x_vel': list(profile['x_vel']),
        'y_dist': tuple(profile['y_dist']),
      }
      for personality, profile in DEFAULT_FOLLOW_DISTANCE_PROFILES.items()
    }

    self._precompute_slopes()
    self._update_tuning_profiles(force=True)

  def _precompute_slopes(self):
    """Precompute all interpolation slopes for efficiency"""
    self.max_accel_slopes = {}
    self.min_accel_slopes = {}
    self.follow_distance_slopes = {}

    # Precompute for AccelPersonality (acceleration)
    for personality in ACCEL_PERSONALITIES:
      if personality in self.max_accel_profiles:
        self.max_accel_slopes[personality] = self._compute_slopes(MAX_ACCEL_BREAKPOINTS, self.max_accel_profiles[personality])

    # Precompute for LongPersonality (braking and following)
    for personality in LONG_PERSONALITIES:
      if personality in self.min_accel_profiles:
        self.min_accel_slopes[personality] = self._compute_slopes(MIN_ACCEL_BREAKPOINTS, self.min_accel_profiles[personality])

      if personality in self.follow_distance_profiles:
        profile = self.follow_distance_profiles[personality]
        self.follow_distance_slopes[personality] = self._compute_slopes(profile['x_vel'], profile['y_dist'])

  def _read_profile_values(self, keys: tuple[str, ...], defaults: tuple[float, ...]) -> tuple[float, ...]:
    values = []
    for key, default in zip(keys, defaults, strict=True):
      try:
        value = self.params.get(key, return_default=True)
      except Exception:
        value = default
      if value is None:
        value = default
      values.append(float(value))
    return tuple(values)

  def _update_tuning_profiles(self, force: bool = False):
    updated = force

    for personality in ACCEL_PERSONALITIES:
      values = self._read_profile_values(MAX_ACCEL_PARAM_KEYS[personality], DEFAULT_MAX_ACCEL_PROFILES[personality])
      if force or values != self.max_accel_profiles[personality]:
        self.max_accel_profiles[personality] = values
        updated = True

    for personality in LONG_PERSONALITIES:
      min_values = self._read_profile_values(MIN_ACCEL_PARAM_KEYS[personality], DEFAULT_MIN_ACCEL_PROFILES[personality])
      if force or min_values != self.min_accel_profiles[personality]:
        self.min_accel_profiles[personality] = min_values
        updated = True

      follow_values = self._read_profile_values(FOLLOW_DISTANCE_PARAM_KEYS[personality], DEFAULT_FOLLOW_DISTANCE_PROFILES[personality]['y_dist'])
      if force or follow_values != self.follow_distance_profiles[personality]['y_dist']:
        self.follow_distance_profiles[personality] = {
          'x_vel': list(DEFAULT_FOLLOW_DISTANCE_PROFILES[personality]['x_vel']),
          'y_dist': follow_values,
        }
        updated = True

    if updated:
      self._precompute_slopes()

  def _update_from_params(self):
    """Update personalities from params (rate limited via wall-clock debounce)."""
    now = time.monotonic()
    if (now - self._last_param_refresh_t) < PARAM_REFRESH_S:
      return
    self._last_param_refresh_t = now

    # Update AccelPersonality
    try:
      accel_personality_val = self.params.get(self.param_keys['accel_personality'])
      if accel_personality_val is not None:
        accel_personality_int = int(accel_personality_val)
        if accel_personality_int in [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]:
          self.accel_personality = accel_personality_int
    except (ValueError, TypeError):
      pass

    self._update_tuning_profiles()

    # Update LongPersonality
    try:
      long_personality_val = self.params.get(self.param_keys['long_personality'])
      if long_personality_val is not None:
        long_personality_int = int(long_personality_val)
        if long_personality_int in [LongPersonality.relaxed, LongPersonality.standard, LongPersonality.aggressive]:
          self.long_personality = long_personality_int
    except (ValueError, TypeError):
      pass

    # Refresh cached toggle states
    self._enabled = self._read_toggle('enabled')
    self._accel_enabled = self._read_toggle('accel_enabled')
    self._follow_enabled = self._read_toggle('follow_enabled')

  def _read_toggle(self, key: str, default: bool = True) -> bool:
    """Read toggle state from Params (call only inside _update_from_params)."""
    try:
      return self.params.get_bool(self.param_keys[key]) if key in self.param_keys else default
    except Exception:
      return default

  def _set_toggle_state(self, key: str, value: bool):
    """Set toggle state in params and update cache."""
    if key in self.param_keys:
      self.params.put_bool(self.param_keys[key], value)
      # Update cached value immediately
      if key == 'enabled':
        self._enabled = value
      elif key == 'accel_enabled':
        self._accel_enabled = value
      elif key == 'follow_enabled':
        self._follow_enabled = value

  # AccelPersonality Management (for acceleration)
  def set_accel_personality(self, personality: int) -> bool:
    """Set AccelPersonality (eco=0, normal=1, sport=2)"""
    if personality in [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]:
      self.accel_personality = personality
      self.params.put(self.param_keys['accel_personality'], str(personality))
      return True
    return False

  def cycle_accel_personality(self) -> int:
    """Cycle through AccelPersonality: eco -> normal -> sport -> eco"""
    personalities = [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]
    current_idx = personalities.index(self.accel_personality)
    next_personality = personalities[(current_idx + 1) % len(personalities)]
    self.set_accel_personality(next_personality)
    return int(next_personality)

  def get_accel_personality(self) -> int:
    """Get current AccelPersonality"""
    self._update_from_params()
    return int(self.accel_personality)

  # LongPersonality Management (for braking and following distance)
  def set_long_personality(self, personality: int) -> bool:
    """Set LongPersonality (relaxed=0, standard=1, aggressive=2)"""
    if personality in [LongPersonality.relaxed, LongPersonality.standard, LongPersonality.aggressive]:
      self.long_personality = personality
      self.params.put(self.param_keys['long_personality'], str(personality))
      return True
    return False

  def cycle_long_personality(self) -> int:
    """Cycle through LongPersonality: relaxed -> standard -> aggressive -> relaxed"""
    personalities = [LongPersonality.relaxed, LongPersonality.standard, LongPersonality.aggressive]
    current_idx = personalities.index(self.long_personality)
    next_personality = personalities[(current_idx + 1) % len(personalities)]
    self.set_long_personality(next_personality)
    return int(next_personality)

  def get_long_personality(self) -> int:
    """Get current LongPersonality"""
    self._update_from_params()
    return int(self.long_personality)

  # Toggle Functions
  def toggle_personality(self): return self._toggle_flag('enabled')
  def toggle_accel_personality(self): return self._toggle_flag('accel_enabled')
  def toggle_follow_distance_personality(self): return self._toggle_flag('follow_enabled')

  def _toggle_flag(self, key):
    current = self._read_toggle(key)
    self._set_toggle_state(key, not current)
    return not current

  def set_personality_enabled(self, enabled: bool): self._set_toggle_state('enabled', enabled)

  # Feature-specific enable checks (use cached toggle values)
  def is_accel_enabled(self) -> bool:
    self._update_from_params()
    return self._enabled and self._accel_enabled

  def is_follow_enabled(self) -> bool:
    self._update_from_params()
    return self._enabled and self._follow_enabled

  def is_enabled(self) -> bool:
    self._update_from_params()
    return self._enabled and (self._accel_enabled or self._follow_enabled)

  def get_accel_limits(self, v_ego: float) -> tuple[float, float] | None:
    """
    Get acceleration limits based on current personalities.
    - Max acceleration from AccelPersonality (eco/normal/sport)
    - Min acceleration (braking) from LongPersonality (relaxed/standard/aggressive)
    Returns None if controller is disabled.
    """
    self._update_from_params()
    if not self.is_accel_enabled():
      return None

    try:
      # Max acceleration from AccelPersonality
      max_a = self._interpolate(v_ego, MAX_ACCEL_BREAKPOINTS, self.max_accel_profiles[self.accel_personality],
                                self.max_accel_slopes[self.accel_personality])

      # Min acceleration (braking) from LongPersonality
      min_a = self._interpolate(v_ego, MIN_ACCEL_BREAKPOINTS, self.min_accel_profiles[self.long_personality],
                                self.min_accel_slopes[self.long_personality])

      return float(min_a), float(max_a)
    except (KeyError, IndexError):
      return None

  def get_follow_distance_multiplier(self, v_ego: float) -> float | None:
    """Get following distance multiplier based on LongPersonality only"""
    self._update_from_params()
    if not self.is_follow_enabled():
      return None

    try:
      profile = self.follow_distance_profiles[self.long_personality]
      multiplier = float(self._interpolate(v_ego, profile['x_vel'], profile['y_dist'],
                                           self.follow_distance_slopes[self.long_personality]))
      return multiplier
    except (KeyError, IndexError):
      return None

  def get_personality_info(self) -> dict:
    """Get comprehensive info about current personalities and settings"""
    self._update_from_params()

    accel_names = {AccelPersonality.eco: "Eco", AccelPersonality.normal: "Normal", AccelPersonality.sport: "Sport"}
    long_names = {LongPersonality.relaxed: "Relaxed", LongPersonality.standard: "Standard", LongPersonality.aggressive: "Aggressive"}

    info = {
      "accel_personality": accel_names.get(self.accel_personality, "Unknown"),
      "accel_personality_int": self.accel_personality,
      "long_personality": long_names.get(self.long_personality, "Unknown"),
      "long_personality_int": self.long_personality,
      "enabled": self._enabled,
      "accel_enabled": self._accel_enabled,
      "follow_enabled": self._follow_enabled,
      "accel_description": f"Acceleration: {accel_names.get(self.accel_personality, 'Unknown')}",
      "long_description": f"Following/Braking: {long_names.get(self.long_personality, 'Unknown')}",
    }

    return info

  def get_min_accel(self, v_ego: float) -> float | None:
    """Get minimum acceleration (braking) from distance mode"""
    limits = self.get_accel_limits(v_ego)
    return limits[0] if limits else None

  def get_max_accel(self, v_ego: float) -> float | None:
    """Get maximum acceleration from drive mode"""
    limits = self.get_accel_limits(v_ego)
    return limits[1] if limits else None

  def reset(self):
    """Reset to default modes"""
    self.accel_personality = AccelPersonality.normal
    self.long_personality = LongPersonality.standard
    self._last_param_refresh_t = 0.0

  def update(self):
    """No-op — kept for caller compatibility. Rate limiting is wall-clock based."""
    pass

  def _compute_slopes(self, x, y):
    """Compute slopes for Hermite interpolation using symmetric difference method."""
    n = len(x)
    if n < 2:
      raise ValueError("At least two points required")

    m = np.zeros(n)
    for i in range(n):
      if i == 0:
        m[i] = (y[1] - y[0]) / (x[1] - x[0])
      elif i == n-1:
        m[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
      else:
        m[i] = ((y[i+1] - y[i]) / (x[i+1] - x[i]) + (y[i] - y[i-1]) / (x[i] - x[i-1])) / 2
    return m

  def _interpolate(self, x, xp, yp, slopes):
    """Perform cubic Hermite interpolation."""
    x = np.clip(x, xp[0], xp[-1])
    idx = np.clip(np.searchsorted(xp, x) - 1, 0, len(slopes) - 2)

    x0, x1 = xp[idx], xp[idx+1]
    y0, y1 = yp[idx], yp[idx+1]
    m0, m1 = slopes[idx], slopes[idx+1]

    t = (x - x0) / (x1 - x0)
    h = [2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2]

    return h[0]*y0 + h[1]*(x1 - x0)*m0 + h[2]*y1 + h[3]*(x1 - x0)*m1
