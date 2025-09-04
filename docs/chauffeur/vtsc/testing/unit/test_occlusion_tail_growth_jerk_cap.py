import math
import time

import pytest

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionOcclusionState


def step(state: VisionOcclusionState, curv: float, conf: float, v: float, t: float):
  """Convenience wrapper to drive the VisionOcclusionState.update signature."""
  state.update(curv, conf, v, t)


def test_tail_growth_respects_lateral_jerk_cap():
  """
  When the jerk cap is the tightest bound, occlusion tail growth must be
  limited by gamma_cap_jerk = lat_jerk_cap / v^3.
  """
  st = VisionOcclusionState()
  # Make jerk cap limiting and allow quick tail onset
  st.gamma_per_m = 1.0            # ensure not limiting
  st.lat_jerk_cap = 0.5           # small -> tight jerk cap
  st.vis_horizon_s = 0.5          # short visible horizon
  st.envelope_horizon_s = 2.0     # sufficient tail window

  v = 20.0  # m/s
  t = 0.0

  # Prime with strong-good vision frames so last_valid_curvature is updated
  for _ in range(4):
    step(st, curv=0.0, conf=0.9, v=v, t=t)
    t += 0.1

  # Drive confidence low to enter occlusion after dwell (0.2 s)
  for _ in range(10):
    step(st, curv=0.0, conf=0.1, v=v, t=t)
    t += 0.1
    if not st.vision_good:
      break

  assert st.vision_good is False, "Expected to be in occlusion state"
  entry = float(st.entry_curvature)

  # Advance during occlusion to accumulate distance and tail
  for _ in range(20):  # 2.0 s -> distance_since ≈ 40 m
    step(st, curv=0.0, conf=0.1, v=v, t=t)
    t += 0.1

  # Compute expected bounds from the same logic in the update()
  s_vis = max(0.0, st.vis_horizon_s * max(0.0, v))
  s_tail_raw = max(0.0, st.distance_since_m - s_vis)

  k_now_for_window = max(0.0, max(st.entry_curvature, st.est_curvature))
  if k_now_for_window <= 0.0035:
    t_allow = 0.10
  else:
    t_allow = 1.20
  if v <= 30.0:
    t_allow = max(t_allow, 0.80)
  t_allow = max(0.10, min(t_allow, st.envelope_horizon_s))
  s_tail_allow = max(0.0, v) * t_allow
  s_tail = min(s_tail_raw, s_tail_allow)

  # Jerk-based gamma cap should be limiting
  eff_v = min(max(0.0, v), 22.0)
  gamma_cap_jerk = st.lat_jerk_cap / max(eff_v ** 3, 1e-3)

  upper = entry + gamma_cap_jerk * s_tail

  assert st.est_curvature <= upper + 1e-9, (
    f"est_curvature {st.est_curvature:.8f} exceeds jerk-limited bound {upper:.8f}"
  )
