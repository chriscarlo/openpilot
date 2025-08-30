#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def visibility_fov(kappa: np.ndarray, fov_half_rad: float = np.pi/6) -> np.ndarray:
  kappa = np.asarray(kappa, dtype=float)
  k = np.abs(kappa)
  s = np.empty_like(k)
  k_eps = 1e-8
  mask = k > k_eps
  s[mask] = float(fov_half_rad) / k[mask]
  s[~mask] = 1e6
  return s


def visibility_walls(
  kappa: np.ndarray,
  lane_width_m: float = 11.0 * 0.3048,
  shoulder_width_m: float = 8.0 * 0.3048,
  right_lane: bool = True,
) -> np.ndarray:
  kappa = np.asarray(kappa, dtype=float)
  k = np.abs(kappa)
  s = np.zeros_like(k)
  k_eps = 1e-8

  for i, (kk, ks) in enumerate(zip(k, kappa)):
    if kk < k_eps:
      s[i] = 1e6
      continue
    R_c = 1.0 / kk
    R_wall = max(1e-6, R_c - (lane_width_m + shoulder_width_m))
    # Camera at lane center; choose inner side based on turning direction
    if ks > 0.0:  # turning right (positive curvature by our test convention)
      R_cam = R_c - (lane_width_m / 2.0)
    else:  # turning left
      R_cam = R_c + (lane_width_m / 2.0)

    if R_cam <= R_wall + 1e-6:
      s[i] = 1e6
      continue

    ratio = np.clip(R_wall / max(R_cam, 1e-6), -1.0, 1.0)
    phi = float(np.arccos(ratio))
    s[i] = R_c * phi

  return s


def pos_margin_mask(
  v_cmd: np.ndarray,
  v_phys: np.ndarray,
  s_vis: np.ndarray,
  comfort_decel_abs: float,
  margin_m: float,
) -> np.ndarray:
  v_cmd = np.asarray(v_cmd, dtype=float)
  v_phys = np.asarray(v_phys, dtype=float)
  s_vis = np.asarray(s_vis, dtype=float)
  a_cap = float(max(1e-6, comfort_decel_abs))
  d_req = np.maximum(0.0, (v_cmd*v_cmd - v_phys*v_phys) / max(2e-3, 2.0*a_cap))
  return d_req <= (s_vis - float(margin_m))

