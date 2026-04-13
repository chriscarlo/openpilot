"""Regression tests for the weather overlay heading source.

Background
----------
The weather overlay PNG is composed north-up (east at the right) by
``weather_overlayd`` and rotated to heading-up at draw time by
``HudRendererSP::drawWeatherOverlay``. The correctness of the rotation
depends entirely on the heading value passed to ``QPainter::rotate`` being
a real compass bearing (0 = true north, + = clockwise toward east).

A previous revision of the HUD used ``livePose.orientationNED.z`` as that
heading. That is wrong: openpilot's ``locationd`` ``pose_kf`` has no absolute
yaw observation (no magnetometer, no GPS-bearing observation). It only
observes gyro and camera-odometry rotation *rates*, and the accelerometer
only constrains roll/pitch via gravity. The filter starts at yaw=0 and
integrates from there. The yaw state is a compass bearing *only* if the
device happens to start perfectly facing north AND the gyro has zero drift
— neither holds in practice.

These tests encode that invariant so nobody re-introduces the same bug.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openpilot.selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from openpilot.selfdrive.locationd.models.pose_kf import PoseKalman, States


def _run_stationary_kf(duration_s: float) -> tuple[float, float]:
  """Run pose_kf with stationary IMU for `duration_s`, return (yaw_rad, yaw_std_rad)."""
  kf = PoseKalman(GENERATED_DIR, 0.8)
  kf.init_state(PoseKalman.initial_x, covs=PoseKalman.initial_P, filter_time=0.0)
  steps = int(100 * duration_s)  # 100 Hz
  for step in range(steps):
    t = 0.01 * (step + 1)
    kf.predict_and_observe(t, ObservationKind.PHONE_GYRO, np.array([0.0, 0.0, 0.0]))
    kf.predict_and_observe(t, ObservationKind.PHONE_ACCEL, np.array([0.0, 0.0, -9.81]))
  yaw = float(kf.x[States.NED_ORIENTATION][2])
  yaw_std = float(np.sqrt(kf.P[2, 2]))
  return yaw, yaw_std


def test_pose_kf_yaw_is_not_compass_bearing_unobservable_std_explodes():
  """After a long stationary run, pose_kf's yaw std must grow unboundedly.

  If the filter observed absolute yaw from any source, a stationary run
  would *shrink* yaw std toward sensor noise. Instead the std blows up,
  which proves the yaw state is unobservable — not a compass bearing.

  This is the invariant the weather overlay depends on: any code that
  treats ``livePose.orientationNED.z`` as ``p.rotate`` input for a
  heading-up map rotation is wrong.
  """
  _, yaw_std = _run_stationary_kf(60.0)
  # Initial std is ~0.01 rad (~0.57 deg). After 60 s with no absolute
  # reference, the std must be orders of magnitude larger — one full
  # rotation (~360 deg) is the hard floor for "completely unobservable".
  assert math.degrees(yaw_std) > 360.0, (
    f"pose_kf yaw std only grew to {math.degrees(yaw_std):.1f} deg after 60 s. "
    f"If yaw were observed against an absolute reference the std would shrink, "
    f"not grow. A yaw state this noisy cannot be used as a compass bearing."
  )


def test_pose_kf_yaw_ignores_actual_compass_heading_straight_drive():
  """A car "actually" heading east but the pose_kf yaw stays at 0.

  This is the same experimental control used to prove the root cause of
  the weather-overlay east/west flip. The filter has no way to know the
  device's true orientation in the NED frame, so after 60 s of straight
  driving (no gyro rotation), yaw is still whatever it was initialised
  to — zero — regardless of the car's real compass heading.
  """
  yaw, _ = _run_stationary_kf(60.0)
  # Yaw must remain at the initial zero, even though the car could be
  # pointing any direction. This is what produces the arbitrary offset
  # the HUD's rotation code used to consume.
  assert abs(yaw) < 1e-6, (
    f"Expected pose_kf yaw to stay at 0 after a straight-drive stationary "
    f"simulation, got {math.degrees(yaw):.6f} deg. If this assertion fires, "
    f"someone may have wired an absolute yaw observation into pose_kf — "
    f"in which case the weather overlay can go back to using livePose."
  )


def test_pose_kf_yaw_integrates_gyro_rate_only():
  """Positive control: gyro yaw rate does move pose_kf yaw, proving the
  filter is operating — it's the *absolute* reference that's missing."""
  kf = PoseKalman(GENERATED_DIR, 0.8)
  kf.init_state(PoseKalman.initial_x, covs=PoseKalman.initial_P, filter_time=0.0)

  # 60 s straight, then a 5 s 90-deg right turn at 18 deg/s.
  for step in range(100 * 60):
    t = 0.01 * (step + 1)
    kf.predict_and_observe(t, ObservationKind.PHONE_GYRO, np.array([0.0, 0.0, 0.0]))
    kf.predict_and_observe(t, ObservationKind.PHONE_ACCEL, np.array([0.0, 0.0, -9.81]))
  rate = math.radians(18.0)
  for step in range(100 * 5):
    t = 60.0 + 0.01 * (step + 1)
    kf.predict_and_observe(t, ObservationKind.PHONE_GYRO, np.array([0.0, 0.0, rate]))
    kf.predict_and_observe(t, ObservationKind.PHONE_ACCEL, np.array([0.0, 0.0, -9.81]))
  yaw_deg = math.degrees(float(kf.x[States.NED_ORIENTATION][2]))
  # Allow 1 deg slop for filter dynamics.
  assert 88.0 < yaw_deg < 92.0, (
    f"Expected pose_kf to integrate gyro rate to ~+90 deg after a 90-deg "
    f"right turn, got {yaw_deg:.2f} deg. If this fails the KF itself is "
    f"broken, not the weather-overlay heading source assumption."
  )


# --- Rotation semantics check --------------------------------------------
# The C++ HUD rotates the composited PNG with ``p.rotate(-heading_deg)``
# where heading_deg is a compass bearing. Document that math here with a
# Python translation so regressions are caught even when Qt is unavailable
# in CI.

def _compass_bearing_to_qt_rotation_deg(bearing_deg: float) -> float:
  """Mirror of the C++ rotation: a compass bearing maps to ``-bearing`` in
  Qt's positive-CW ``QPainter::rotate`` convention."""
  return -bearing_deg


@pytest.mark.parametrize(
  "bearing_deg,expected_up_direction",
  [
    (0.0, "north"),   # car heading north → north stays up (source is north-up)
    (90.0, "east"),   # car heading east  → east must rotate to the top
    (180.0, "south"), # car heading south → south must rotate to the top
    (270.0, "west"),  # car heading west  → west must rotate to the top
  ],
)
def test_rotation_puts_compass_bearing_at_top(bearing_deg: float, expected_up_direction: str):
  """Rotate a unit 'forward' vector by the Qt rotation that the HUD applies,
  then check that the resulting screen-frame vector points up (=-Y in Qt)."""
  rotation_deg = _compass_bearing_to_qt_rotation_deg(bearing_deg)
  # In source coords (before rotation) "north" is -Y, "east" is +X, "south"
  # is +Y, "west" is -X. Pick the unit vector corresponding to the direction
  # the car is heading on the real-world compass.
  direction_vectors = {
    "north": (0.0, -1.0),
    "east":  (1.0,  0.0),
    "south": (0.0,  1.0),
    "west":  (-1.0, 0.0),
  }
  src_x, src_y = direction_vectors[expected_up_direction]

  # Qt's ``QTransform::rotate(θ)`` applies the standard math CCW rotation
  # matrix ``R(θ) = [[cos -sin], [sin cos]]``. Because Qt's Y axis points
  # *down*, that matrix *visually* rotates points clockwise on screen.
  # Applied to a source point (x, y) in the local (post-translate-to-anchor)
  # frame, the screen coordinate is:
  #     screen_x = x·cos(θ) - y·sin(θ)
  #     screen_y = x·sin(θ) + y·cos(θ)
  theta = math.radians(rotation_deg)
  cos_t = math.cos(theta)
  sin_t = math.sin(theta)
  screen_x = src_x * cos_t - src_y * sin_t
  screen_y = src_x * sin_t + src_y * cos_t

  # "Up" on screen is -Y. Require the rotated forward direction to land
  # very near (0, -1).
  assert screen_x == pytest.approx(0.0, abs=1e-9), (
    f"bearing={bearing_deg} {expected_up_direction}: screen_x={screen_x:.6f}, "
    f"expected ~0"
  )
  assert screen_y == pytest.approx(-1.0, abs=1e-9), (
    f"bearing={bearing_deg} {expected_up_direction}: screen_y={screen_y:.6f}, "
    f"expected -1 (up)"
  )
