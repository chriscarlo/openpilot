#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _ensure_repo_on_path() -> Path:
  candidates: list[Path] = []
  try:
    candidates.append(Path(__file__).resolve().parents[4])
  except Exception:
    pass
  candidates.append(Path.cwd())
  candidates.append(Path("/data/openpilot"))

  for root in candidates:
    if root.exists() and (root / "cereal").exists() and (root / "selfdrive").exists():
      # Some Windows dev shells have old openpilot checkouts or PyPI packages
      # such as `msgq.py` ahead of the current repo. Force this checkout first.
      for path in (root / "opendbc_repo", root / "msgq_repo", root):
        path_str = str(path)
        sys.path[:] = [entry for entry in sys.path if entry.lower() != path_str.lower()]
        sys.path.insert(0, path_str)
      return root
  raise RuntimeError("could not locate openpilot repo root")


REPO_ROOT = _ensure_repo_on_path()

from cereal import log  # noqa: E402
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import build_lead_response_tuning_config  # noqa: E402
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as long_mpc_module  # noqa: E402
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, N  # noqa: E402
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA, ModelLeadTracker, get_lead  # noqa: E402


EV6_LIVE_SOFT_TUNE: dict[str, float] = {
  "lead_preview_strength": 0.80,
  "lead_preview_gap_min_m": 3.0,
  "lead_preview_max_buffer_m": 5.0,
  "lead_acquire_window_s": 0.20,
  "gap_reclaim_strength": 0.35,
  "gap_reclaim_gap_min_m": 5.0,
  "gap_reclaim_max_accel": 0.08,
  "lead_keepup_strength": 1.0,
  "lead_keepup_gap_min_m": 0.80,
  "lead_keepup_max_accel": 0.06,
  "lead_slowdown_strength": 0.65,
  "lead_slowdown_max_decel": 6.0,
  "cutin_settle_duration_s": 10.0,
  "cutin_settle_max_decel": 0.08,
  "cutin_settle_max_closing_speed_mps": 5.0,
  "cutin_settle_accel_bias_mps2": 0.15,
  "virtual_lead_slow_tau_s": 1.30,
  "drel_filter_tau_close_s": 0.45,
  "drel_filter_tau_open_s": 1.80,
  "drel_filter_open_slew_max_mps": 0.70,
  "drel_filter_innovation_gate_m": 30.0,
  "drel_filter_closing_gate_m": 12.0,
  "cruise_reacquire_pos_jerk_limit": 0.08,
  "cruise_reacquire_jerk_window_s": 3.0,
  "lead_prob_enter": 0.60,
  "lead_prob_exit": 0.35,
  "lead_source_acquire_frames": 2.0,
  "lead_source_release_frames": 8.0,
  "phantom_lead_hold_s": 0.40,
  "phantom_lead_stable_frames": 5.0,
  "flutter_detect_transitions": 2.0,
  "flutter_detect_window_s": 1.0,
  "flutter_clamp_jerk_mps3": 0.18,
  "flutter_clamp_bypass_decel_mps2": 1.5,
  "model_lead_filter_tau_s": 2.8,
  "model_lead_filter_open_slew_max_mps": 1.2,
  "model_lead_filter_safe_ttc_s": 4.0,
  "model_lead_filter_assoc_drel_m": 12.0,
  "model_lead_filter_vrel_tau_s": 0.55,
  "model_lead_filter_fast_vrel_tau_s": 0.22,
}


LEAD_TUNE_PROFILES: dict[str, dict[str, float]] = {
  "ev6-live-soft": EV6_LIVE_SOFT_TUNE,
  "ev6-half-keepup": {
    **EV6_LIVE_SOFT_TUNE,
    "lead_keepup_max_accel": 0.03,
  },
}


SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
  "ev6-highway-acquisition": {
    "duration_s": 55.0,
    "hz": 20.0,
    "seed": 15,
    "true_gap": 108.0,
    "v_ego": 35.6,
    "v_lead": 30.6,
    "v_cruise": 40.0,
    "model_prob": 0.97,
    "t_follow": 1.3,
    "source_noise_std_m": 3.5,
    "source_noise_tau_s": 1.1,
    "white_noise_std_m": 0.35,
    "vrel_noise_std_mps": 0.40,
    "vrel_noise_tau_s": 0.8,
    "vrel_white_noise_std_mps": 0.08,
    "spike_prob_per_s": 0.08,
    "spike_min_m": 8.0,
    "spike_max_m": 24.0,
    "spike_hold_min_s": 0.10,
    "spike_hold_max_s": 0.30,
    "duplicate": True,
    "duplicate_std_m": 1.2,
    "closed_loop": True,
    "full_mpc": True,
  },
  "ev6-highway-goldilocks": {
    "duration_s": 140.0,
    "hz": 20.0,
    "seed": 16,
    "true_gap": 44.5,
    "v_ego": 29.0,
    "v_lead": 29.0,
    "v_cruise": 29.0,
    "model_prob": 0.96,
    "t_follow": 1.3,
    "source_noise_std_m": 1.2,
    "source_noise_tau_s": 1.1,
    "white_noise_std_m": 0.30,
    "vrel_noise_std_mps": 0.22,
    "vrel_noise_tau_s": 0.8,
    "vrel_white_noise_std_mps": 0.06,
    "spike_prob_per_s": 0.02,
    "spike_min_m": 2.0,
    "spike_max_m": 4.0,
    "spike_hold_min_s": 0.10,
    "spike_hold_max_s": 0.25,
    "duplicate": False,
    "closed_loop": False,
    "full_mpc": True,
  },
  "ev6-highway-goldilocks-flap": {
    "duration_s": 60.0,
    "hz": 10.0,
    "seed": 17,
    "true_gap": 47.0,
    "v_ego": 29.0,
    "v_lead": 29.0,
    "v_cruise": 29.0,
    "model_prob": 0.96,
    "t_follow": 1.3,
    "source_noise_std_m": 0.9,
    "source_noise_tau_s": 1.1,
    "white_noise_std_m": 0.20,
    "vrel_noise_std_mps": 0.28,
    "vrel_noise_tau_s": 0.8,
    "vrel_white_noise_std_mps": 0.08,
    "spike_prob_per_s": 0.02,
    "spike_min_m": 2.0,
    "spike_max_m": 4.0,
    "spike_hold_min_s": 0.10,
    "spike_hold_max_s": 0.25,
    "duplicate": False,
    "closed_loop": False,
    "full_mpc": False,
  },
}


@dataclass(frozen=True)
class NoiseConfig:
  source_std_m: float
  source_tau_s: float
  white_std_m: float
  vrel_std_mps: float
  vrel_tau_s: float
  vrel_white_std_mps: float
  spike_prob_per_s: float
  spike_min_m: float
  spike_max_m: float
  spike_hold_min_s: float
  spike_hold_max_s: float
  dropout_prob_per_s: float
  duplicate: bool
  duplicate_std_m: float
  duplicate_y_offset_m: float


@dataclass
class Sample:
  frame: int
  t: float
  true_gap_m: float
  source_noise_m: float
  source_vrel_noise_mps: float
  raw_drel_m: float
  radard_drel_m: float
  radard_error_m: float
  radar_track_id: int
  filtered_drel_m: float
  filtered_error_m: float
  mpc_source: str
  virtual_source: str
  reset_reason: str | None
  snap_to_raw: bool
  open_slew_clamped: bool
  drel_consistency_m: float
  a_solution_mps2: float
  a_command_mps2: float
  gap_reclaim_floor_mps2: float
  lead_keepup_floor_mps2: float
  lead_slowdown_ceiling_mps2: float
  lead_bias_command_mps2: float
  acc_reason: str | None
  duplicate_active: bool


class Clock:
  def __init__(self) -> None:
    self.t = 0.0

  def monotonic(self) -> float:
    return self.t


class LeadNoiseGenerator:
  def __init__(self, rng: random.Random, cfg: NoiseConfig):
    self.rng = rng
    self.cfg = cfg
    self.ou_m = 0.0
    self.spike_m = 0.0
    self.spike_remaining_s = 0.0

  def step(self, dt_s: float) -> float:
    tau_s = max(1e-3, float(self.cfg.source_tau_s))
    decay = math.exp(-dt_s / tau_s)
    self.ou_m = decay * self.ou_m + self.cfg.source_std_m * math.sqrt(max(0.0, 1.0 - decay * decay)) * self.rng.gauss(0.0, 1.0)

    if self.spike_remaining_s > 0.0:
      self.spike_remaining_s = max(0.0, self.spike_remaining_s - dt_s)
      if self.spike_remaining_s <= 0.0:
        self.spike_m = 0.0
    elif self.cfg.spike_prob_per_s > 0.0 and self.rng.random() < self.cfg.spike_prob_per_s * dt_s:
      sign = -1.0 if self.rng.random() < 0.5 else 1.0
      mag = self.rng.uniform(max(0.0, self.cfg.spike_min_m), max(self.cfg.spike_min_m, self.cfg.spike_max_m))
      self.spike_m = sign * mag
      self.spike_remaining_s = self.rng.uniform(max(0.0, self.cfg.spike_hold_min_s), max(self.cfg.spike_hold_min_s, self.cfg.spike_hold_max_s))

    white_m = self.rng.gauss(0.0, max(0.0, self.cfg.white_std_m))
    return self.ou_m + self.spike_m + white_m

  def visible(self, dt_s: float) -> bool:
    return not (self.cfg.dropout_prob_per_s > 0.0 and self.rng.random() < self.cfg.dropout_prob_per_s * dt_s)


class OUNoiseGenerator:
  def __init__(self, rng: random.Random, *, std: float, tau_s: float, white_std: float) -> None:
    self.rng = rng
    self.std = max(0.0, float(std))
    self.tau_s = max(1e-3, float(tau_s))
    self.white_std = max(0.0, float(white_std))
    self.ou = 0.0

  def step(self, dt_s: float) -> float:
    decay = math.exp(-float(dt_s) / self.tau_s)
    self.ou = decay * self.ou + self.std * math.sqrt(max(0.0, 1.0 - decay * decay)) * self.rng.gauss(0.0, 1.0)
    return self.ou + self.rng.gauss(0.0, self.white_std)


def _model_path() -> SimpleNamespace:
  return SimpleNamespace(position=SimpleNamespace(
    x=np.linspace(0.0, 160.0, 33),
    y=np.zeros(33),
  ))


def _model_lead(*, drel_m: float, v_ego_mps: float, v_lead_mps: float, prob: float,
                y_m: float = 0.0, a_lead_mps2: float = 0.0) -> SimpleNamespace:
  x0 = max(0.0, float(drel_m)) + RADAR_TO_CAMERA
  return SimpleNamespace(
    t=[0.0, 1.0],
    x=[x0, x0 + float(v_lead_mps)],
    y=[-float(y_m), -float(y_m)],
    v=[float(v_lead_mps), float(v_lead_mps)],
    a=[float(a_lead_mps2), float(a_lead_mps2)],
    xStd=[1.0, 1.0],
    yStd=[0.35, 0.35],
    vStd=[1.0, 1.0],
    prob=float(prob),
    probTime=0.0,
  )


def _lead_namespace(lead_dict: dict[str, Any]) -> SimpleNamespace:
  return SimpleNamespace(
    status=bool(lead_dict.get("status", False)),
    dRel=float(lead_dict.get("dRel", 0.0) or 0.0),
    yRel=float(lead_dict.get("yRel", 0.0) or 0.0),
    vRel=float(lead_dict.get("vRel", 0.0) or 0.0),
    aRel=float(lead_dict.get("aRel", 0.0) or 0.0),
    vLead=float(lead_dict.get("vLead", 0.0) or 0.0),
    dPath=float(lead_dict.get("dPath", lead_dict.get("yRel", 0.0)) or 0.0),
    vLat=float(lead_dict.get("vLat", 0.0) or 0.0),
    vLeadK=float(lead_dict.get("vLeadK", lead_dict.get("vLead", 0.0)) or 0.0),
    aLeadK=float(lead_dict.get("aLeadK", 0.0) or 0.0),
    fcw=bool(lead_dict.get("fcw", False)),
    aLeadTau=float(lead_dict.get("aLeadTau", 1.5) or 1.5),
    modelProb=float(lead_dict.get("modelProb", 0.0) or 0.0),
    radar=bool(lead_dict.get("radar", False)),
    radarTrackId=int(lead_dict.get("radarTrackId", -1) or -1),
  )


def _inactive_lead() -> SimpleNamespace:
  return _lead_namespace({"status": False})


def _sim_t_follow(mpc: LongitudinalMpc, v_ego: float, args: argparse.Namespace) -> float:
  if float(args.t_follow) > 0.0:
    return float(args.t_follow)
  if mpc.vibe_controller.is_follow_enabled():
    desired_headway = mpc.vibe_controller.get_follow_distance_multiplier(v_ego)
    if desired_headway is not None:
      return max(0.5, float(desired_headway) - long_mpc_module.STOP_DISTANCE / max(float(v_ego), 1.0))
  return float(long_mpc_module.get_T_FOLLOW(log.LongitudinalPersonality.standard))


def _apply_sim_follow_override(mpc: LongitudinalMpc, args: argparse.Namespace) -> None:
  if float(args.t_follow) <= 0.0:
    return

  mpc.vibe_controller.is_follow_enabled = lambda: True
  mpc.vibe_controller.get_follow_distance_multiplier = (
    lambda v_ego: float(args.t_follow) + long_mpc_module.STOP_DISTANCE / max(float(v_ego), 1.0)
  )


def _apply_lead_tune_profile(mpc: LongitudinalMpc, args: argparse.Namespace) -> None:
  profile = str(getattr(args, "lead_tune_profile", "params"))
  if profile == "params":
    return

  cfg = build_lead_response_tuning_config(LEAD_TUNE_PROFILES[profile])
  mpc._live_tune_cfg = cfg
  mpc._refresh_live_tune = lambda now, force=False: None


def _fast_update_virtual_lead(mpc: LongitudinalMpc, radarstate: SimpleNamespace, *,
                              now: float, v_ego: float, a_ego: float, args: argparse.Namespace) -> None:
  mpc._refresh_live_tune(now, force=(now <= 0.0))
  mpc.set_cur_state(float(v_ego), float(a_ego))
  mpc.current_t_follow = _sim_t_follow(mpc, float(v_ego), args)

  raw_control_lead0, raw_control_lead1, lead_role_debug = mpc.lead_role_classifier.classify(
    float(v_ego), radarstate.leadOne, radarstate.leadTwo, now=now,
  )
  mpc.control_leads, lead_role_debug = mpc._stabilize_control_leads(
    radarstate.leadOne,
    radarstate.leadTwo,
    raw_control_lead0,
    raw_control_lead1,
    lead_role_debug,
  )
  mpc.lead_role_debug = lead_role_debug
  mpc.status = bool(getattr(mpc.control_leads[0], "status", False) or getattr(mpc.control_leads[1], "status", False))
  mpc._update_cutin_settle_state(now, float(v_ego), lead_role_debug)

  lead_0_obstacle = mpc._build_lead_obstacle(mpc.control_leads[0])
  lead_1_obstacle = mpc._build_lead_obstacle(mpc.control_leads[1])
  v_cruise_profile = np.full(long_mpc_module.N + 1, float(args.v_cruise))
  cruise_obstacle = (
    np.cumsum(long_mpc_module.T_DIFFS * v_cruise_profile) +
    long_mpc_module.get_safe_obstacle_distance(v_cruise_profile, mpc.current_t_follow)
  )
  mpc._select_acc_obstacle(lead_0_obstacle, lead_1_obstacle, cruise_obstacle, now)


def _full_mpc_update(mpc: LongitudinalMpc, radarstate: SimpleNamespace, *,
                     v_ego: float, a_ego: float, args: argparse.Namespace) -> None:
  x = np.zeros(N + 1)
  v = np.full(N + 1, v_ego)
  a = np.full(N + 1, a_ego)
  j = np.zeros(N + 1)
  mpc.set_cur_state(v_ego, a_ego)
  mpc.update(
    radarstate,
    float(args.v_cruise),
    x,
    v,
    a,
    j,
    personality=log.LongitudinalPersonality.standard,
  )


def _rolling_ranges(values: list[float], window: int) -> list[float]:
  if not values:
    return []
  window = max(1, int(window))
  q: deque[float] = deque(maxlen=window)
  ranges: list[float] = []
  for value in values:
    if math.isfinite(value):
      q.append(value)
    if q:
      ranges.append(max(q) - min(q))
  return ranges


def _pct(values: list[float], percentile: float) -> float:
  finite = [float(v) for v in values if math.isfinite(float(v))]
  if not finite:
    return float("nan")
  return float(np.percentile(np.asarray(finite), percentile))


def _transition_count(values: list[str]) -> int:
  return sum(1 for idx in range(1, len(values)) if values[idx] != values[idx - 1])


def _sign_flip_count(values: list[float], *, deadband: float = 0.03) -> int:
  signs: list[int] = []
  for value in values:
    if not math.isfinite(value) or abs(value) < deadband:
      continue
    signs.append(1 if value > 0.0 else -1)
  return sum(1 for idx in range(1, len(signs)) if signs[idx] != signs[idx - 1])


def _finite_series(samples: list[Sample], attr: str) -> list[float]:
  values = [float(getattr(sample, attr)) for sample in samples]
  return [value for value in values if math.isfinite(value)]


def _ev_micro_flap_counts(values: list[float], *, positive_threshold: float, zero_band: float) -> tuple[int, int, int]:
  prev_state: str | None = None
  positive_to_zero = 0
  zero_to_positive = 0
  for value in values:
    if value > positive_threshold:
      state = "positive"
    elif abs(value) <= zero_band:
      state = "zero"
    elif value < -positive_threshold:
      state = "negative"
    else:
      state = "middle"

    if prev_state == "positive" and state in ("zero", "negative"):
      positive_to_zero += 1
    elif prev_state in ("zero", "negative") and state == "positive":
      zero_to_positive += 1
    if state != "middle":
      prev_state = state

  return positive_to_zero + zero_to_positive, positive_to_zero, zero_to_positive


def summarize(samples: list[Sample], *, hz: float, ev_positive_threshold: float = 0.03,
              ev_zero_band: float = 0.005) -> dict[str, Any]:
  raw_errors = [s.raw_drel_m - s.true_gap_m for s in samples if math.isfinite(s.raw_drel_m)]
  radard_errors = [s.radard_error_m for s in samples if math.isfinite(s.radard_error_m)]
  radard_values = [s.radard_drel_m for s in samples if math.isfinite(s.radard_drel_m)]
  radard_steps = [abs(radard_values[i] - radard_values[i - 1]) for i in range(1, len(radard_values))]
  radard_rolling_3s = _rolling_ranges(radard_values, int(round(max(1.0, hz * 3.0))))
  filtered_errors = [s.filtered_error_m for s in samples if math.isfinite(s.filtered_error_m)]
  filtered_values = [s.filtered_drel_m for s in samples if math.isfinite(s.filtered_drel_m)]
  filtered_steps = [abs(filtered_values[i] - filtered_values[i - 1]) for i in range(1, len(filtered_values))]
  rolling_3s = _rolling_ranges(filtered_values, int(round(max(1.0, hz * 3.0))))
  source_counts = Counter(s.mpc_source for s in samples)
  virtual_source_counts = Counter(s.virtual_source for s in samples)
  reset_counts = Counter(s.reset_reason for s in samples if s.reset_reason)
  mpc_sources = [s.mpc_source for s in samples]
  virtual_sources = [s.virtual_source for s in samples]
  accel_values = [s.a_solution_mps2 for s in samples if math.isfinite(s.a_solution_mps2)]
  accel_steps = [accel_values[i] - accel_values[i - 1] for i in range(1, len(accel_values))]
  accel_slews = [abs(step) * float(hz) for step in accel_steps]
  accel_abs = [abs(value) for value in accel_values]
  command_values = [s.a_command_mps2 for s in samples if math.isfinite(s.a_command_mps2)]
  command_steps = [command_values[i] - command_values[i - 1] for i in range(1, len(command_values))]
  command_slews = [abs(step) * float(hz) for step in command_steps]
  command_abs = [abs(value) for value in command_values]
  lead_bias_values = _finite_series(samples, "lead_bias_command_mps2")
  lead_bias_steps = [lead_bias_values[i] - lead_bias_values[i - 1] for i in range(1, len(lead_bias_values))]
  lead_bias_slews = [abs(step) * float(hz) for step in lead_bias_steps]
  lead_bias_abs = [abs(value) for value in lead_bias_values]
  ev_flaps, ev_pos_to_zero, ev_zero_to_pos = _ev_micro_flap_counts(
    lead_bias_values,
    positive_threshold=max(0.0, float(ev_positive_threshold)),
    zero_band=max(0.0, float(ev_zero_band)),
  )

  return {
    "frames": len(samples),
    "duration_s": samples[-1].t if samples else 0.0,
    "raw_error_abs_p50_m": _pct([abs(v) for v in raw_errors], 50),
    "raw_error_abs_p95_m": _pct([abs(v) for v in raw_errors], 95),
    "raw_error_range_m": (max(raw_errors) - min(raw_errors)) if raw_errors else float("nan"),
    "radard_error_abs_p50_m": _pct([abs(v) for v in radard_errors], 50),
    "radard_error_abs_p95_m": _pct([abs(v) for v in radard_errors], 95),
    "radard_error_range_m": (max(radard_errors) - min(radard_errors)) if radard_errors else float("nan"),
    "radard_drel_range_m": (max(radard_values) - min(radard_values)) if radard_values else float("nan"),
    "radard_rolling_3s_range_p95_m": _pct(radard_rolling_3s, 95),
    "radard_rolling_3s_range_max_m": max(radard_rolling_3s) if radard_rolling_3s else float("nan"),
    "radard_step_abs_p95_m": _pct(radard_steps, 95),
    "radard_step_abs_max_m": max(radard_steps) if radard_steps else float("nan"),
    "filtered_error_abs_p50_m": _pct([abs(v) for v in filtered_errors], 50),
    "filtered_error_abs_p95_m": _pct([abs(v) for v in filtered_errors], 95),
    "filtered_error_range_m": (max(filtered_errors) - min(filtered_errors)) if filtered_errors else float("nan"),
    "filtered_drel_range_m": (max(filtered_values) - min(filtered_values)) if filtered_values else float("nan"),
    "filtered_rolling_3s_range_p95_m": _pct(rolling_3s, 95),
    "filtered_rolling_3s_range_max_m": max(rolling_3s) if rolling_3s else float("nan"),
    "filtered_step_abs_p95_m": _pct(filtered_steps, 95),
    "filtered_step_abs_max_m": max(filtered_steps) if filtered_steps else float("nan"),
    "snap_to_raw_count": sum(1 for s in samples if s.snap_to_raw),
    "open_slew_clamped_count": sum(1 for s in samples if s.open_slew_clamped),
    "duplicate_active_count": sum(1 for s in samples if s.duplicate_active),
    "mpc_source_transition_count": _transition_count(mpc_sources),
    "virtual_source_transition_count": _transition_count(virtual_sources),
    "a_solution_sample_count": len(accel_values),
    "a_solution_min_mps2": min(accel_values) if accel_values else float("nan"),
    "a_solution_max_mps2": max(accel_values) if accel_values else float("nan"),
    "a_solution_abs_p95_mps2": _pct(accel_abs, 95),
    "a_solution_step_abs_p95_mps2": _pct([abs(step) for step in accel_steps], 95),
    "a_solution_step_abs_max_mps2": max([abs(step) for step in accel_steps]) if accel_steps else float("nan"),
    "a_solution_slew_abs_p95_mps3": _pct(accel_slews, 95),
    "a_solution_slew_abs_max_mps3": max(accel_slews) if accel_slews else float("nan"),
    "a_solution_sign_flip_count": _sign_flip_count(accel_values),
    "a_command_sample_count": len(command_values),
    "a_command_min_mps2": min(command_values) if command_values else float("nan"),
    "a_command_max_mps2": max(command_values) if command_values else float("nan"),
    "a_command_abs_p95_mps2": _pct(command_abs, 95),
    "a_command_step_abs_p95_mps2": _pct([abs(step) for step in command_steps], 95),
    "a_command_step_abs_max_mps2": max([abs(step) for step in command_steps]) if command_steps else float("nan"),
    "a_command_slew_abs_p95_mps3": _pct(command_slews, 95),
    "a_command_slew_abs_max_mps3": max(command_slews) if command_slews else float("nan"),
    "a_command_sign_flip_count": _sign_flip_count(command_values),
    "lead_bias_sample_count": len(lead_bias_values),
    "lead_bias_min_mps2": min(lead_bias_values) if lead_bias_values else float("nan"),
    "lead_bias_max_mps2": max(lead_bias_values) if lead_bias_values else float("nan"),
    "lead_bias_abs_p95_mps2": _pct(lead_bias_abs, 95),
    "lead_bias_step_abs_p95_mps2": _pct([abs(step) for step in lead_bias_steps], 95),
    "lead_bias_step_abs_max_mps2": max([abs(step) for step in lead_bias_steps]) if lead_bias_steps else float("nan"),
    "lead_bias_slew_abs_p95_mps3": _pct(lead_bias_slews, 95),
    "lead_bias_slew_abs_max_mps3": max(lead_bias_slews) if lead_bias_slews else float("nan"),
    "lead_bias_sign_flip_count": _sign_flip_count(lead_bias_values),
    "ev_positive_threshold_mps2": max(0.0, float(ev_positive_threshold)),
    "ev_zero_band_mps2": max(0.0, float(ev_zero_band)),
    "ev_micro_flap_count": ev_flaps,
    "ev_micro_pos_to_zero_count": ev_pos_to_zero,
    "ev_micro_zero_to_pos_count": ev_zero_to_pos,
    "gap_reclaim_floor_max_mps2": max(_finite_series(samples, "gap_reclaim_floor_mps2"), default=float("nan")),
    "lead_keepup_floor_max_mps2": max(_finite_series(samples, "lead_keepup_floor_mps2"), default=float("nan")),
    "source_counts": dict(source_counts),
    "virtual_source_counts": dict(virtual_source_counts),
    "reset_counts": {str(k): v for k, v in reset_counts.items()},
  }


def run_simulation(args: argparse.Namespace, cfg: NoiseConfig) -> tuple[dict[str, Any], list[Sample]]:
  rng = random.Random(int(args.seed))
  primary_noise = LeadNoiseGenerator(rng, cfg)
  duplicate_noise = LeadNoiseGenerator(random.Random(int(args.seed) + 100_003), cfg)
  primary_vrel_noise = OUNoiseGenerator(
    random.Random(int(args.seed) + 200_003),
    std=cfg.vrel_std_mps,
    tau_s=cfg.vrel_tau_s,
    white_std=cfg.vrel_white_std_mps,
  )
  duplicate_vrel_noise = OUNoiseGenerator(
    random.Random(int(args.seed) + 300_007),
    std=cfg.vrel_std_mps,
    tau_s=cfg.vrel_tau_s,
    white_std=cfg.vrel_white_std_mps,
  )

  cp = SimpleNamespace(brand="hyundai", radarUnavailable=True, openpilotLongitudinalControl=True, flags=0)
  cp_sp = SimpleNamespace(flags=0)
  model_msg = _model_path()
  mpc = LongitudinalMpc(CP=cp)
  mpc.mode = "acc"
  _apply_sim_follow_override(mpc, args)
  _apply_lead_tune_profile(mpc, args)
  model_lead_tracker = None if bool(args.disable_model_lead_tracker) else ModelLeadTracker()

  dt_s = 1.0 / max(1.0, float(args.hz))
  frames = max(1, int(round(float(args.duration_s) / dt_s)))
  clock = Clock()
  original_monotonic = long_mpc_module.time.monotonic
  samples: list[Sample] = []

  v_ego = float(args.v_ego)
  a_ego = 0.0
  v_lead = float(args.v_lead)
  true_gap = float(args.true_gap)

  try:
    long_mpc_module.time.monotonic = clock.monotonic
    for frame in range(frames):
      clock.t = frame * dt_s

      if bool(args.closed_loop) and frame > 0:
        true_gap = max(0.0, true_gap + (v_lead - v_ego) * dt_s)

      source_noise = primary_noise.step(dt_s)
      source_vrel_noise = primary_vrel_noise.step(dt_s)
      noisy_v_lead = max(0.0, v_lead + source_vrel_noise)
      raw_drel = max(float(args.min_drel), true_gap + source_noise)
      primary_prob = float(args.model_prob) if primary_noise.visible(dt_s) else 0.0
      lead0_msg = _model_lead(drel_m=raw_drel, v_ego_mps=v_ego, v_lead_mps=noisy_v_lead, prob=primary_prob)
      if model_lead_tracker is not None:
        model_lead_tracker.begin_frame(clock.t)
      lead0 = _lead_namespace(get_lead(
        v_ego,
        primary_prob > 0.5,
        {},
        lead0_msg,
        v_ego,
        cp,
        cp_sp,
        model_msg,
        low_speed_override=False,
        model_lead_tracker=model_lead_tracker,
        lead_slot=0,
        now=clock.t,
      ))

      if cfg.duplicate:
        dup_noise = duplicate_noise.step(dt_s)
        dup_vrel_noise = duplicate_vrel_noise.step(dt_s)
        dup_v_lead = max(0.0, v_lead + dup_vrel_noise)
        dup_drel = max(float(args.min_drel), true_gap + dup_noise + rng.gauss(0.0, cfg.duplicate_std_m))
        dup_prob = max(0.51, min(0.99, float(args.model_prob) - 0.03 + rng.gauss(0.0, 0.02)))
        lead1_msg = _model_lead(
          drel_m=dup_drel,
          v_ego_mps=v_ego,
          v_lead_mps=dup_v_lead,
          prob=dup_prob,
          y_m=cfg.duplicate_y_offset_m,
        )
        lead1 = _lead_namespace(get_lead(
          v_ego,
          dup_prob > 0.5,
          {},
          lead1_msg,
          v_ego,
          cp,
          cp_sp,
          model_msg,
          low_speed_override=False,
          model_lead_tracker=model_lead_tracker,
          lead_slot=1,
          now=clock.t,
        ))
      else:
        lead1 = _inactive_lead()
      if model_lead_tracker is not None:
        model_lead_tracker.end_frame()

      radarstate = SimpleNamespace(leadOne=lead0, leadTwo=lead1)
      if bool(args.full_mpc):
        _full_mpc_update(mpc, radarstate, v_ego=v_ego, a_ego=a_ego, args=args)
      else:
        _fast_update_virtual_lead(mpc, radarstate, now=clock.t, v_ego=v_ego, a_ego=a_ego, args=args)

      virtual_debug = mpc.hyundai_virtual_lead_debug or {}
      filtered_payload = virtual_debug.get("filtered", {}) if isinstance(virtual_debug, dict) else {}
      filter_debug = virtual_debug.get("filter", {}) if isinstance(virtual_debug, dict) else {}
      filtered_drel = float(filtered_payload.get("dRel", float("nan")))
      duplicate_debug = mpc.lead_role_debug.get("virtual_duplicate", {}) if isinstance(mpc.lead_role_debug, dict) else {}
      acc_reason = None
      if isinstance(mpc.acc_source_debug, dict):
        raw_reason = mpc.acc_source_debug.get("reason")
        acc_reason = None if raw_reason is None else str(raw_reason)
      a_solution = float(mpc.a_solution[1] if bool(args.full_mpc) and len(mpc.a_solution) > 1 else float("nan"))
      a_command = (
        float(np.clip(a_solution, float(args.min_accel), float(args.max_accel)))
        if math.isfinite(a_solution) else float("nan")
      )
      gap_reclaim_floor = float(getattr(mpc, "gap_reclaim_accel_floor", 0.0) or 0.0)
      lead_keepup_floor = float(getattr(mpc, "lead_keepup_accel_floor", 0.0) or 0.0)
      slowdown_ceiling_raw = getattr(mpc, "lead_slowdown_accel_ceiling", None)
      lead_slowdown_ceiling = float(slowdown_ceiling_raw) if slowdown_ceiling_raw is not None else float("nan")
      lead_bias_command = max(gap_reclaim_floor, lead_keepup_floor)
      if math.isfinite(lead_slowdown_ceiling) and lead_slowdown_ceiling < lead_bias_command:
        lead_bias_command = lead_slowdown_ceiling

      samples.append(Sample(
        frame=frame,
        t=clock.t,
        true_gap_m=float(true_gap),
        source_noise_m=float(source_noise),
        source_vrel_noise_mps=float(source_vrel_noise),
        raw_drel_m=float(raw_drel) if bool(lead0.status) else float("nan"),
        radard_drel_m=float(lead0.dRel) if bool(lead0.status) else float("nan"),
        radard_error_m=float(lead0.dRel) - float(true_gap) if bool(lead0.status) else float("nan"),
        radar_track_id=int(lead0.radarTrackId) if bool(lead0.status) else -1,
        filtered_drel_m=filtered_drel,
        filtered_error_m=filtered_drel - float(true_gap) if math.isfinite(filtered_drel) else float("nan"),
        mpc_source=str(mpc.source),
        virtual_source=str(virtual_debug.get("source", "")),
        reset_reason=virtual_debug.get("reset_reason") if isinstance(virtual_debug, dict) else None,
        snap_to_raw=bool(filter_debug.get("snap_to_raw", False)),
        open_slew_clamped=bool(filter_debug.get("open_slew_clamped", False)),
        drel_consistency_m=float(virtual_debug.get("drel_consistency_m", float("nan"))) if isinstance(virtual_debug, dict) else float("nan"),
        a_solution_mps2=a_solution,
        a_command_mps2=a_command,
        gap_reclaim_floor_mps2=gap_reclaim_floor,
        lead_keepup_floor_mps2=lead_keepup_floor,
        lead_slowdown_ceiling_mps2=lead_slowdown_ceiling,
        lead_bias_command_mps2=lead_bias_command,
        acc_reason=acc_reason,
        duplicate_active=bool(duplicate_debug.get("active", False)),
      ))

      if bool(args.closed_loop):
        a_cmd = a_command if math.isfinite(a_command) else 0.0
        v_ego = max(0.0, v_ego + a_cmd * dt_s)
        a_ego = a_cmd
  finally:
    long_mpc_module.time.monotonic = original_monotonic

  summary = summarize(
    samples,
    hz=float(args.hz),
    ev_positive_threshold=float(args.ev_positive_threshold_mps2),
    ev_zero_band=float(args.ev_zero_band_mps2),
  )
  summary["scenario"] = str(args.scenario)
  summary["lead_tune_profile"] = str(args.lead_tune_profile)
  summary["seed"] = int(args.seed)
  summary["noise_config"] = asdict(cfg)
  summary["closed_loop"] = bool(args.closed_loop)
  summary["full_mpc"] = bool(args.full_mpc)
  summary["model_lead_tracker_enabled"] = not bool(args.disable_model_lead_tracker)
  summary["live_tune"] = mpc._live_tune_cfg.as_dict()
  return summary, samples


def _write_json(path: str, payload: dict[str, Any]) -> None:
  out = Path(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str, samples: list[Sample]) -> None:
  out = Path(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(asdict(samples[0]).keys()) if samples else [])
    if samples:
      writer.writeheader()
      for sample in samples:
        writer.writerow(asdict(sample))


def _parse_float_list(raw: str) -> list[float]:
  return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _config_from_args(args: argparse.Namespace, *, source_std_m: float | None = None) -> NoiseConfig:
  return NoiseConfig(
    source_std_m=float(args.source_noise_std_m if source_std_m is None else source_std_m),
    source_tau_s=float(args.source_noise_tau_s),
    white_std_m=float(args.white_noise_std_m),
    vrel_std_mps=float(args.vrel_noise_std_mps),
    vrel_tau_s=float(args.vrel_noise_tau_s),
    vrel_white_std_mps=float(args.vrel_white_noise_std_mps),
    spike_prob_per_s=float(args.spike_prob_per_s),
    spike_min_m=float(args.spike_min_m),
    spike_max_m=float(args.spike_max_m),
    spike_hold_min_s=float(args.spike_hold_min_s),
    spike_hold_max_s=float(args.spike_hold_max_s),
    dropout_prob_per_s=float(args.dropout_prob_per_s),
    duplicate=bool(args.duplicate),
    duplicate_std_m=float(args.duplicate_std_m),
    duplicate_y_offset_m=float(args.duplicate_y_offset_m),
  )


def run_sweep(args: argparse.Namespace) -> int:
  rows: list[dict[str, Any]] = []
  base_seed = int(args.seed)
  std_values = _parse_float_list(args.sweep_stds)
  for source_std in std_values:
    for seed_offset in range(max(1, int(args.sweep_seeds))):
      args.seed = base_seed + seed_offset
      summary, _samples = run_simulation(args, _config_from_args(args, source_std_m=source_std))
      metric = float(summary["filtered_rolling_3s_range_p95_m"])
      rows.append({
        "source_std_m": source_std,
        "seed": int(args.seed),
        "filtered_error_abs_p95_m": summary["filtered_error_abs_p95_m"],
        "radard_rolling_3s_range_p95_m": summary["radard_rolling_3s_range_p95_m"],
        "filtered_rolling_3s_range_p95_m": summary["filtered_rolling_3s_range_p95_m"],
        "filtered_rolling_3s_range_max_m": summary["filtered_rolling_3s_range_max_m"],
        "filtered_step_abs_max_m": summary["filtered_step_abs_max_m"],
        "a_solution_slew_abs_p95_mps3": summary["a_solution_slew_abs_p95_mps3"],
        "a_solution_sign_flip_count": summary["a_solution_sign_flip_count"],
        "a_command_slew_abs_p95_mps3": summary["a_command_slew_abs_p95_mps3"],
        "a_command_sign_flip_count": summary["a_command_sign_flip_count"],
        "lead_bias_slew_abs_p95_mps3": summary["lead_bias_slew_abs_p95_mps3"],
        "lead_bias_sign_flip_count": summary["lead_bias_sign_flip_count"],
        "ev_micro_flap_count": summary["ev_micro_flap_count"],
        "snap_to_raw_count": summary["snap_to_raw_count"],
        "open_slew_clamped_count": summary["open_slew_clamped_count"],
        "source_counts": summary["source_counts"],
        "in_target_band": float(args.target_min_m) <= metric <= float(args.target_max_m),
      })

  print("source_std seed filt_abs_p95 radard_roll3s_p95 roll3s_p95 roll3s_max step_max bias_slew_p95 ev_flaps snap clamp target")
  for row in rows:
    print(
      f"{row['source_std_m']:9.2f} {row['seed']:4d} "
      f"{row['filtered_error_abs_p95_m']:12.2f} {row['radard_rolling_3s_range_p95_m']:18.2f} "
      f"{row['filtered_rolling_3s_range_p95_m']:10.2f} "
      f"{row['filtered_rolling_3s_range_max_m']:10.2f} {row['filtered_step_abs_max_m']:8.2f} "
      f"{row['lead_bias_slew_abs_p95_mps3']:13.2f} {row['ev_micro_flap_count']:8d} "
      f"{row['snap_to_raw_count']:4d} {row['open_slew_clamped_count']:5d} {str(row['in_target_band']):>6s}"
    )

  target_rows = [row for row in rows if row["in_target_band"]]
  if target_rows:
    best = min(target_rows, key=lambda row: (row["filtered_rolling_3s_range_p95_m"], row["snap_to_raw_count"]))
    print(
      "selected "
      f"source_std={best['source_std_m']:.2f} seed={best['seed']} "
      f"roll3s_p95={best['filtered_rolling_3s_range_p95_m']:.2f}m "
      f"abs_p95={best['filtered_error_abs_p95_m']:.2f}m"
    )
  else:
    print("selected none: no sweep row entered target band")

  if args.json_out:
    _write_json(args.json_out, {"sweep": rows, "target_min_m": args.target_min_m, "target_max_m": args.target_max_m})
  return 0


def print_summary(summary: dict[str, Any], samples: list[Sample], args: argparse.Namespace) -> None:
  print("ai-lead dRel source-noise simulation: PASS")
  print(
    f"scenario={summary['scenario']} tune={summary['lead_tune_profile']} "
    f"frames={summary['frames']} duration={summary['duration_s']:.2f}s seed={summary['seed']} "
    f"source_std={summary['noise_config']['source_std_m']:.2f}m "
    f"vrel_std={summary['noise_config']['vrel_std_mps']:.2f}mps duplicate={summary['noise_config']['duplicate']} "
    f"tracker={summary['model_lead_tracker_enabled']} full_mpc={summary['full_mpc']}"
  )
  print(
    f"raw_abs_p95={summary['raw_error_abs_p95_m']:.2f}m raw_range={summary['raw_error_range_m']:.2f}m "
    f"radard_abs_p95={summary['radard_error_abs_p95_m']:.2f}m radard_range={summary['radard_error_range_m']:.2f}m "
    f"filtered_abs_p95={summary['filtered_error_abs_p95_m']:.2f}m "
    f"filtered_range={summary['filtered_error_range_m']:.2f}m"
  )
  print(
    f"radard_roll3s_p95={summary['radard_rolling_3s_range_p95_m']:.2f}m "
    f"radard_roll3s_max={summary['radard_rolling_3s_range_max_m']:.2f}m "
    f"radard_step_p95={summary['radard_step_abs_p95_m']:.2f}m radard_step_max={summary['radard_step_abs_max_m']:.2f}m"
  )
  print(
    f"filtered_roll3s_p95={summary['filtered_rolling_3s_range_p95_m']:.2f}m "
    f"filtered_roll3s_max={summary['filtered_rolling_3s_range_max_m']:.2f}m "
    f"step_p95={summary['filtered_step_abs_p95_m']:.2f}m step_max={summary['filtered_step_abs_max_m']:.2f}m"
  )
  print(
    f"snap_to_raw={summary['snap_to_raw_count']} open_slew_clamped={summary['open_slew_clamped_count']} "
    f"source_transitions={summary['mpc_source_transition_count']} source_counts={summary['source_counts']} "
    f"reset_counts={summary['reset_counts']}"
  )
  print(
    f"a_raw_range=[{summary['a_solution_min_mps2']:.2f}, {summary['a_solution_max_mps2']:.2f}]mps2 "
    f"a_cmd_samples={summary['a_command_sample_count']} "
    f"a_cmd_range=[{summary['a_command_min_mps2']:.2f}, {summary['a_command_max_mps2']:.2f}]mps2 "
    f"a_cmd_abs_p95={summary['a_command_abs_p95_mps2']:.2f}mps2 "
    f"a_cmd_step_p95={summary['a_command_step_abs_p95_mps2']:.3f}mps2 "
    f"a_cmd_slew_p95={summary['a_command_slew_abs_p95_mps3']:.2f}mps3 "
    f"a_cmd_slew_max={summary['a_command_slew_abs_max_mps3']:.2f}mps3 "
    f"a_cmd_sign_flips={summary['a_command_sign_flip_count']}"
  )
  print(
    f"lead_bias_range=[{summary['lead_bias_min_mps2']:.3f}, {summary['lead_bias_max_mps2']:.3f}]mps2 "
    f"lead_bias_abs_p95={summary['lead_bias_abs_p95_mps2']:.3f}mps2 "
    f"lead_bias_slew_p95={summary['lead_bias_slew_abs_p95_mps3']:.2f}mps3 "
    f"lead_bias_slew_max={summary['lead_bias_slew_abs_max_mps3']:.2f}mps3 "
    f"lead_bias_sign_flips={summary['lead_bias_sign_flip_count']} "
    f"ev_flaps={summary['ev_micro_flap_count']} "
    f"pos_to_zero={summary['ev_micro_pos_to_zero_count']} "
    f"zero_to_pos={summary['ev_micro_zero_to_pos_count']} "
    f"ev_pos_thresh={summary['ev_positive_threshold_mps2']:.3f} "
    f"reclaim_max={summary['gap_reclaim_floor_max_mps2']:.3f}mps2 "
    f"keepup_max={summary['lead_keepup_floor_max_mps2']:.3f}mps2"
  )
  print(
    "note: EV6/no-radar model leads bypass radard Track Kalman filtering; "
    "this measures source model dRel -> radard model-lead tracker -> Hyundai virtual lead filtering. "
    "Use --full-mpc for slower solver-backed accel output."
  )

  if args.print_samples:
    stride = max(1, int(round(float(args.hz) * float(args.sample_period_s))))
    print("frame time true raw radard filtered filt_err track source a_sol a_cmd bias reset snap clamp reason")
    for sample in samples[::stride]:
      print(
        f"{sample.frame:5d} {sample.t:6.2f} {sample.true_gap_m:6.2f} "
        f"{sample.raw_drel_m:6.2f} {sample.radard_drel_m:7.2f} "
        f"{sample.filtered_drel_m:8.2f} {sample.filtered_error_m:8.2f} "
        f"{sample.radar_track_id:5d} {sample.mpc_source:6s} {sample.a_solution_mps2:6.2f} "
        f"{sample.a_command_mps2:6.2f} {sample.lead_bias_command_mps2:6.2f} "
        f"{str(sample.reset_reason):>14s} "
        f"{str(sample.snap_to_raw):>5s} {str(sample.open_slew_clamped):>5s} {sample.acc_reason}"
      )


def _provided_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
  provided: set[str] = set()
  for action in parser._actions:
    for option in action.option_strings:
      if any(arg == option or arg.startswith(f"{option}=") for arg in argv):
        provided.add(action.dest)
  return provided


def apply_scenario_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace, argv: list[str]) -> None:
  defaults = SCENARIO_DEFAULTS.get(str(args.scenario), {})
  provided = _provided_dests(parser, argv)
  for dest, value in defaults.items():
    if dest not in provided:
      setattr(args, dest, value)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Inject source-side AI/model lead dRel noise before radard.get_lead and measure post-MPC filtered dRel.",
  )
  parser.add_argument(
    "--scenario",
    choices=("steady", *SCENARIO_DEFAULTS.keys()),
    default="steady",
    help="preset log-shaped road scenario; explicit CLI values override preset defaults",
  )
  parser.add_argument("--duration-s", type=float, default=60.0)
  parser.add_argument("--hz", type=float, default=20.0)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--true-gap", type=float, default=42.0)
  parser.add_argument("--v-ego", type=float, default=29.0)
  parser.add_argument("--v-lead", type=float, default=29.0)
  parser.add_argument("--v-cruise", type=float, default=36.0)
  parser.add_argument("--model-prob", type=float, default=0.96)
  parser.add_argument("--t-follow", type=float, default=0.0, help="override follow time; 0 uses current Vibe/standard logic")
  parser.add_argument(
    "--lead-tune-profile",
    choices=("params", *LEAD_TUNE_PROFILES.keys()),
    default="params",
    help="override live Params with a named lead-response profile for repeatable what-if runs",
  )
  parser.add_argument("--min-drel", type=float, default=4.0)
  parser.add_argument("--source-noise-std-m", type=float, default=8.0)
  parser.add_argument("--source-noise-tau-s", type=float, default=1.6)
  parser.add_argument("--white-noise-std-m", type=float, default=0.45)
  parser.add_argument("--vrel-noise-std-mps", type=float, default=0.0)
  parser.add_argument("--vrel-noise-tau-s", type=float, default=0.8)
  parser.add_argument("--vrel-white-noise-std-mps", type=float, default=0.0)
  parser.add_argument("--spike-prob-per-s", type=float, default=0.10)
  parser.add_argument("--spike-min-m", type=float, default=4.0)
  parser.add_argument("--spike-max-m", type=float, default=14.0)
  parser.add_argument("--spike-hold-min-s", type=float, default=0.10)
  parser.add_argument("--spike-hold-max-s", type=float, default=0.45)
  parser.add_argument("--dropout-prob-per-s", type=float, default=0.0)
  parser.add_argument("--duplicate", action="store_true")
  parser.add_argument("--duplicate-std-m", type=float, default=1.0)
  parser.add_argument("--duplicate-y-offset-m", type=float, default=0.08)
  parser.add_argument("--closed-loop", action="store_true")
  parser.add_argument("--full-mpc", action="store_true", help="run the full LongitudinalMpc solver path; slow on Windows")
  parser.add_argument("--disable-model-lead-tracker", action="store_true", help="use the old raw model-lead radard path")
  parser.add_argument("--min-accel", type=float, default=-2.0)
  parser.add_argument("--max-accel", type=float, default=1.0)
  parser.add_argument("--ev-positive-threshold-mps2", type=float, default=0.03)
  parser.add_argument("--ev-zero-band-mps2", type=float, default=0.005)
  parser.add_argument("--print-samples", action="store_true")
  parser.add_argument("--sample-period-s", type=float, default=2.0)
  parser.add_argument("--json-out", type=str, default="")
  parser.add_argument("--csv-out", type=str, default="")
  parser.add_argument("--sweep", action="store_true")
  parser.add_argument("--sweep-stds", type=str, default="2,4,6,8,10,12,15,18,22")
  parser.add_argument("--sweep-seeds", type=int, default=3)
  parser.add_argument("--target-min-m", type=float, default=1.0)
  parser.add_argument("--target-max-m", type=float, default=15.0)
  return parser


def main() -> int:
  parser = build_parser()
  args = parser.parse_args()
  apply_scenario_defaults(parser, args, sys.argv[1:])
  if args.sweep:
    return run_sweep(args)

  summary, samples = run_simulation(args, _config_from_args(args))
  print_summary(summary, samples, args)
  if args.json_out:
    _write_json(args.json_out, {"summary": summary, "samples": [asdict(sample) for sample in samples]})
  if args.csv_out:
    _write_csv(args.csv_out, samples)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
