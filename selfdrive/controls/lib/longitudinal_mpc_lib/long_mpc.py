#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import platform
import time
from typing import Any
import numpy as np
from cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
from openpilot.selfdrive.controls.lib.lead_role_classifier import ControlLead, LeadRoleClassifier
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.lead_kalman_filter import LeadKalmanFilter
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import (
  LeadResponseTuningConfig,
  read_lead_response_tuning_config,
)
from openpilot.selfdrive.controls.lib.longitudinal_response_model import (
  DEFAULT_COMFORT_BRAKE,
  DEFAULT_CRUISE_MAX_ACCEL,
  DEFAULT_CRUISE_MIN_ACCEL,
  CruiseResponseModel,
  build_cruise_response_model,
  clip_cruise_speed_profile,
)

from openpilot.sunnypilot.selfdrive.controls.lib.vibe_personality.vibe_personality import VibePersonalityController

if __name__ != '__main__':
  try:
    from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
  except ModuleNotFoundError:
    if platform.system() != "Windows":
      raise
    from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.windows_acados_stub import AcadosOcpSolverCython

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']
REAL_MONOTONIC = time.monotonic

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 4.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 5.0
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'
LEAD_APPROACH_PREVIEW_MIN_SPEED = 8.0
LEAD_APPROACH_PREVIEW_CLOSING_MIN_MPS = 0.20
LEAD_APPROACH_PREVIEW_TIME_BP = [0.0, 2.0, 5.0, 10.0]
LEAD_APPROACH_PREVIEW_TIME_V = [0.0, 0.15, 0.50, 0.95]
LEAD_APPROACH_PREVIEW_DECAY_TAU = 1.75
LEAD_APPROACH_PREVIEW_ACQUIRE_GAP_FRACTION = 0.35
LEAD_APPROACH_PREVIEW_PROJECTED_DEFICIT_GAIN = 0.85
LEAD_APPROACH_PREVIEW_GAP_CAP_FRACTION = 0.70
LEAD_APPROACH_PREVIEW_ACQUIRE_GAP_CAP_FRACTION = 0.95
LEAD_APPROACH_PREVIEW_ACQUIRE_GAIN = 1.45
LEAD_APPROACH_PREVIEW_ACQUIRE_TIGHT_GAP_GAIN = 0.35
LEAD_APPROACH_PREVIEW_ACQUIRE_TIGHT_GAP_MAX_FRACTION = 0.20
LEAD_APPROACH_PREVIEW_ANTICIPATORY_CLOSING_MIN_MPS = 1.25
LEAD_APPROACH_PREVIEW_ANTICIPATORY_TTC_BP = [6.0, 9.0, 12.0, 14.0]
LEAD_APPROACH_PREVIEW_ANTICIPATORY_GAP_FRACTION_V = [0.55, 0.30, 0.10, 0.0]
LEAD_APPROACH_PREVIEW_DECEL_BP = [0.0, 0.5, 1.5, 3.0]
LEAD_APPROACH_PREVIEW_DECEL_V = [1.0, 1.05, 1.20, 1.35]
LEAD_HANDOFF_DANGER_MIN_SPEED = 12.0
LEAD_HANDOFF_DANGER_CLOSING_MIN_MPS = 0.75
LEAD_HANDOFF_DANGER_HORIZON_S = 1.2
LEAD_HANDOFF_DANGER_MAX_FACTOR = 1.0
LEAD_HANDOFF_DANGER_DEFICIT_REF_M = 6.0
LEAD_HANDOFF_DANGER_CLOSING_BP = [0.75, 1.5, 3.0, 6.0]
LEAD_HANDOFF_DANGER_CLOSING_V = [0.0, 0.35, 0.75, 1.0]
GAP_RECLAIM_MIN_SPEED = 8.0
GAP_RECLAIM_HEADWAY_SURPLUS_BP = [0.0, 0.08, 0.18, 0.35]
GAP_RECLAIM_HEADWAY_SURPLUS_V = [0.0, 0.06, 0.20, 0.30]
GAP_RECLAIM_PULLAWAY_BP = [0.0, 0.5, 1.5, 3.0]
GAP_RECLAIM_PULLAWAY_V = [0.0, 0.04, 0.12, 0.20]
GAP_RECLAIM_LEAD_ACCEL_BP = [0.0, 0.5, 1.5]
GAP_RECLAIM_LEAD_ACCEL_V = [0.0, 0.03, 0.06]
GAP_RECLAIM_PERSONALITY_SURPLUS_BP = [0.0, 0.06, 0.18, 0.35]
GAP_RECLAIM_PERSONALITY_SURPLUS_V = [0.0, 0.0, 0.40, 1.0]
GAP_RECLAIM_PERSONALITY_PULLAWAY_BP = [0.0, 0.3, 0.8, 1.6]
GAP_RECLAIM_PERSONALITY_PULLAWAY_V = [0.0, 0.12, 0.45, 1.0]
GAP_RECLAIM_PROJECT_HORIZON_S = 1.2
GAP_RECLAIM_PROJECT_GAP_WEIGHT = 0.35
GAP_RECLAIM_PROJECT_PULLAWAY_WEIGHT = 0.65
GAP_RECLAIM_PROJECT_EGO_ACCEL_BP = [0.0, 0.25, 0.55, 0.90]
GAP_RECLAIM_PROJECT_EGO_ACCEL_V = [0.0, 0.0, 0.45, 1.0]
GAP_RECLAIM_BLEND_RISE_TAU_S = 0.60
GAP_RECLAIM_BLEND_FALL_TAU_S = 1.20
GAP_RECLAIM_HORIZON_RAMP_TAU_S = 1.00
GAP_RECLAIM_RELAX_ROOM_FRACTION = 0.85
GAP_RECLAIM_RELAX_ROOM_MAX_M = 18.0
LEAD_KEEPUP_MIN_SPEED = 4.0
LEAD_KEEPUP_SOFT_PULLAWAY_BP = [0.0, 0.12, 0.45, 1.0]
LEAD_KEEPUP_SOFT_PULLAWAY_V = [0.0, 0.00, 0.005, 0.010]
LEAD_KEEPUP_CAP_PULLAWAY_BP = [0.0, 0.45, 1.0, 1.6, 2.5]
LEAD_KEEPUP_CAP_PULLAWAY_V = [0.0, 0.00, 0.18, 0.55, 1.00]
LEAD_KEEPUP_CAP_GAP_SURPLUS_BP = [0.0, 1.0, 3.0, 6.0, 10.0]
LEAD_KEEPUP_CAP_GAP_SURPLUS_V = [0.0, 0.00, 0.08, 0.35, 1.00]
LEAD_KEEPUP_ACCEL_GATE_PULLAWAY_BP = [0.0, 0.45, 1.0, 1.5]
LEAD_KEEPUP_ACCEL_GATE_PULLAWAY_V = [0.0, 0.00, 0.65, 1.00]
LEAD_KEEPUP_ACCEL_GATE_GAP_BP = [0.0, 1.0, 3.0, 6.0]
LEAD_KEEPUP_ACCEL_GATE_GAP_V = [0.0, 0.15, 0.65, 1.00]
LEAD_KEEPUP_ACCEL_OVERSHOOT = 1.05
LEAD_KEEPUP_TOO_CLOSE_MARGIN_M = 0.75
LEAD_KEEPUP_CLOSING_BLOCK_MPS = 0.10
LEAD_KEEPUP_LEAD_DECEL_BLOCK_MPS2 = -0.25
# Terminal rollout is handled by LongControl.stopping. Keep the lead slowdown
# ceiling out of crawl speeds so it cannot bypass the gentle stop ramp.
LEAD_SLOWDOWN_MIN_SPEED = 2.0
LEAD_SLOWDOWN_MIN_CLOSING_MPS = 0.10
LEAD_SLOWDOWN_MIN_LEAD_DECEL_MPS2 = 0.15
LEAD_SLOWDOWN_HORIZON_S = 1.25
LEAD_SLOWDOWN_SOFT_ACCEL_CAP = 0.08
LEAD_SLOWDOWN_ONSET_CLOSING_BP = [0.10, 0.25, 0.80]
LEAD_SLOWDOWN_ONSET_CLOSING_V = [0.0, 0.20, 1.0]
LEAD_SLOWDOWN_ONSET_DECEL_BP = [0.15, 0.50, 1.50]
LEAD_SLOWDOWN_ONSET_DECEL_V = [0.0, 0.15, 1.0]
LEAD_SLOWDOWN_HEADWAY_DEFICIT_BP = [0.0, 0.5, 2.0, 5.0]
LEAD_SLOWDOWN_HEADWAY_DEFICIT_V = [0.0, 0.12, 0.55, 1.0]
LEAD_SLOWDOWN_DANGER_DEFICIT_BP = [0.0, 0.3, 2.5]
LEAD_SLOWDOWN_DANGER_DEFICIT_V = [0.0, 0.55, 1.0]
LEAD_SLOWDOWN_TTC_HEADWAY_BP = [0.7, 1.5, 3.0, 6.0]
LEAD_SLOWDOWN_TTC_HEADWAY_V = [1.0, 0.75, 0.30, 0.0]
LEAD_SLOWDOWN_TTC_DANGER_BP = [0.4, 1.0, 2.0, 4.0]
LEAD_SLOWDOWN_TTC_DANGER_V = [1.0, 0.80, 0.35, 0.0]
LEAD_SLOWDOWN_TTC_COLLISION_BP = [0.8, 1.5, 3.0, 5.0]
LEAD_SLOWDOWN_TTC_COLLISION_V = [1.0, 0.70, 0.20, 0.0]
LEAD_SLOWDOWN_GAP_GATE_BP = [-2.0, 0.0, 3.0, 8.0]
LEAD_SLOWDOWN_GAP_GATE_V = [1.0, 1.0, 0.50, 0.0]
LEAD_SLOWDOWN_CLOSING_MATCH_BP = [0.0, 0.5, 1.5]
LEAD_SLOWDOWN_CLOSING_MATCH_V = [0.0, 0.35, 1.0]
LEAD_SLOWDOWN_ANTICIPATORY_CLOSING_MIN_MPS = 1.25
LEAD_SLOWDOWN_ANTICIPATORY_LEAD_DECEL_GAIN = 1.0
LEAD_SLOWDOWN_ANTICIPATORY_STRENGTH_FLOOR = 1.0
LEAD_SLOWDOWN_DANGER_MOTION_CLOSING_BP = [0.30, 1.0, 3.0]
LEAD_SLOWDOWN_DANGER_MOTION_CLOSING_V = [0.0, 0.35, 1.0]
LEAD_SLOWDOWN_DANGER_MOTION_DECEL_BP = [0.50, 1.5, 4.0]
LEAD_SLOWDOWN_DANGER_MOTION_DECEL_V = [0.0, 0.35, 1.0]
LEAD_SLOWDOWN_HARD_BRAKE_DECEL_BP = [3.0, 5.0, 6.0]
LEAD_SLOWDOWN_HARD_BRAKE_DECEL_V = [0.0, 0.50, 1.0]
LEAD_SLOWDOWN_LEAD_DECEL_OVERSHOOT = 1.05
# Hard floor for the energy-consistency bound's gap reserve (see
# get_lead_slowdown_accel_ceiling): below ~1 m the margin gives LESS ceiling
# braking (wrong direction for the permissive end of a safety tunable) and the
# inside-margin full-authority restoration becomes unreachable. The matching
# hard ceiling is STOP_DISTANCE - 1.0, enforced at the same clip site.
LEAD_SLOWDOWN_KINEMATIC_MARGIN_MIN_M = 1.0
LEAD_SLOWDOWN_COMFORT_DECEL_CAP = 1.0
LEAD_SLOWDOWN_MIN_DECEL_OUTPUT = 0.03
LEAD_SLOWDOWN_CEILING_RELEASE_RATE_MPS3 = 0.45
LEAD_SLOWDOWN_CEILING_HOLD_GAP_SURPLUS_M = 1.5
LEAD_SLOWDOWN_CEILING_HOLD_CLOSING_MPS = 0.25
LEAD_SLOWDOWN_CEILING_HOLD_PULLAWAY_RELEASE_MPS = 0.8
LOW_SPEED_LAUNCH_FACTOR_V_EGO_BP = [0.0, 2.0, 6.0, 10.0]
LOW_SPEED_LAUNCH_FACTOR_V_EGO_V = [1.0, 1.0, 0.60, 0.0]
LOW_SPEED_LAUNCH_FACTOR_V_LEAD_BP = [0.0, 0.2, 1.0, 3.0, 6.0]
LOW_SPEED_LAUNCH_FACTOR_V_LEAD_V = [0.0, 0.0, 0.35, 0.80, 1.0]
LOW_SPEED_LAUNCH_FACTOR_PULLAWAY_BP = [0.0, 0.15, 0.5, 1.5, 3.0]
LOW_SPEED_LAUNCH_FACTOR_PULLAWAY_V = [0.0, 0.0, 0.25, 0.75, 1.0]
LOW_SPEED_LAUNCH_FACTOR_GAP_BP = [0.0, 0.5, 2.0, 5.0, 9.0]
LOW_SPEED_LAUNCH_FACTOR_GAP_V = [0.0, 0.05, 0.25, 0.70, 1.0]
LOW_SPEED_LAUNCH_GAP_BUFFER_M = 7.0
LOW_SPEED_LAUNCH_QUEUE_SPEED_EXTRA_MPS = 3.0
LOW_SPEED_LAUNCH_QUEUE_VLEAD_EXTRA_MPS = 3.0
LOW_SPEED_LAUNCH_QUEUE_PULLAWAY_EXTRA_MPS = 3.0
LOW_SPEED_LAUNCH_MAX_ACCEL = 2.4
# Keep in sync with selfdrive/controls/lib/longitudinal_planner.py:get_max_accel.
# LongitudinalMpc cannot import that module directly because planner imports MPC.
GAP_RECLAIM_BASE_MAX_ACCEL_BP = [0.0, 10.0, 25.0, 40.0]
GAP_RECLAIM_BASE_MAX_ACCEL_V = [1.6, 1.2, 0.8, 0.6]
CUTIN_SETTLE_MIN_SPEED = 15.0
CUTIN_SETTLE_DETECT_DREL_MAX = 70.0
CUTIN_SETTLE_DETECT_PATH_ABS_MIN = 0.8
CUTIN_SETTLE_DETECT_TOWARD_CENTER_MIN_MPS = 0.35
CUTIN_SETTLE_DANGER_MARGIN_M = 2.0
CUTIN_SETTLE_LEAD_ACCEL_MIN = -0.5
CUTIN_SETTLE_PROGRESS_BP = [0.0, 0.15, 0.5, 1.0]
CUTIN_SETTLE_PROGRESS_V = [0.0, 0.10, 0.45, 1.0]
HYUNDAI_DUPLICATE_PATH_SWITCH_M = 0.10
HYUNDAI_DUPLICATE_MODEL_PROB_SWITCH = 0.10
HYUNDAI_DUPLICATE_VLAT_SWITCH_MPS = 1.0
HYUNDAI_DUPLICATE_DREL_SWITCH_M = 0.75
HYUNDAI_VIRTUAL_LEAD_FAST_TAU_S = 0.20
HYUNDAI_VIRTUAL_LEAD_SLOW_TAU_S = 1.00
HYUNDAI_VIRTUAL_LEAD_SIGN_TRANSITION_TAU_S = 0.30
HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S = 0.45
HYUNDAI_VIRTUAL_LEAD_MODEL_PROB_TAU_S = 0.60
HYUNDAI_VIRTUAL_LEAD_RESET_DREL_M = 8.0
HYUNDAI_VIRTUAL_LEAD_RESET_DPATH_M = 1.75
HYUNDAI_VIRTUAL_LEAD_RETAIN_GAP_SURPLUS_M = 2.5
HYUNDAI_VIRTUAL_LEAD_REACQUIRE_GAP_SURPLUS_M = 3.0
HYUNDAI_VIRTUAL_LEAD_RELEASE_GAP_SURPLUS_M = 4.0
HYUNDAI_VIRTUAL_LEAD_RELEASE_PULLAWAY_MPS = 0.35
HYUNDAI_VIRTUAL_LEAD_RELEASE_DWELL_S = 1.30
HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_GAP_SURPLUS_M = 9.0
HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_PULLAWAY_MPS = 1.00
HYUNDAI_VIRTUAL_LEAD_RELEASE_RAW_GAP_SURPLUS_M = 2.5
HYUNDAI_VIRTUAL_LEAD_RELEASE_RAW_PULLAWAY_MPS = 0.15
HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_RAW_GAP_SURPLUS_M = 4.5
HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_RAW_PULLAWAY_MPS = 0.45
HYUNDAI_VIRTUAL_LEAD_RELEASE_AGREEMENT_MAX_DREL_ERR_M = 3.0
HYUNDAI_VIRTUAL_LEAD_RAW_OBSTACLE_MARGIN_M = 1.00
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_MIN_SPEED = 8.0
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_CLOSING_MPS = 1.5
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_CLOSING_MAX_MPS = 4.0
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_GAP_SURPLUS_M = 20.0
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_OBSTACLE_MARGIN_M = 12.0
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_NEAR_GAP_SURPLUS_M = 8.0
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_TTC_HEADWAY_S = 7.5
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_LOW_SPEED_TTC_HEADWAY_S = 9.5
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_LOW_SPEED_MPS = 12.0
HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_HIGH_SPEED_MPS = 25.0
HYUNDAI_CRUISE_CAP_RAW_LEAD_MAX_PATH_ABS_M = 2.0
HYUNDAI_CRUISE_CAP_RAW_LEAD_MAX_DREL_M = 120.0
HYUNDAI_VIRTUAL_LEAD_DROPOUT_STABLE_MIN_S = 3.0
HYUNDAI_VIRTUAL_LEAD_DROPOUT_HOLD_S = 0.75
HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_ABS_VREL_MPS = 0.35
HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_DREL_ERR_M = 1.5
HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_GAP_SURPLUS_M = 3.0
HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_PATH_ABS_M = 0.85
# Classifier demotion hold: shorter, raw-corroborated fallback when the lead
# role classifier briefly rejects a steady-follow lead (e.g. path_abs spike,
# one-frame validity failure). Distinct from dropout hold in three ways:
# (1) it fires only when the previous cycle was actively following,
# (2) it requires fresh raw-lead corroboration in radarState so no stale
#     state is held through a genuine cut-out,
# (3) its max duration is shorter, since the failure mode it addresses is
#     single-frame classifier flicker rather than sustained sensor dropout.
HYUNDAI_CLASSIFIER_DEMOTION_HOLD_S = 0.30
HYUNDAI_CLASSIFIER_DEMOTION_HOLD_MIN_STABLE_S = 1.00
HYUNDAI_CLASSIFIER_DEMOTION_HOLD_MAX_PATH_ABS_M = 1.20
HYUNDAI_CLASSIFIER_DEMOTION_HOLD_MAX_DREL_ERR_M = 2.00
HYUNDAI_CLASSIFIER_DEMOTION_HOLD_CORROB_DREL_M = 10.0
HYUNDAI_SETTLED_FOLLOW_MAX_GAP_SURPLUS_M = 12.0
HYUNDAI_SETTLED_FOLLOW_MAX_ABS_VREL_MPS = 0.35
HYUNDAI_SETTLED_FOLLOW_MAX_ABS_ALEAD_MPS2 = 0.25
HYUNDAI_LOW_SPEED_QUEUE_V_EGO_MAX = 6.0
HYUNDAI_LOW_SPEED_QUEUE_DREL_MAX = 22.0
HYUNDAI_LOW_SPEED_QUEUE_VLEAD_MAX = 8.0
HYUNDAI_LOW_SPEED_QUEUE_PULLAWAY_MPS_MAX = 2.5
HYUNDAI_RECLAIM_PULLAWAY_SUPPORT_MPS = 0.15
HYUNDAI_RECLAIM_GAP_HOLD_EXTRA_M = 1.0
HYUNDAI_RECLAIM_DYNAMIC_GAP_SUPPORT_M = 3.0
HYUNDAI_RECLAIM_NOISE_SUPPRESS_CLOSING_MPS = 0.25
HYUNDAI_RECLAIM_NOISE_SUPPRESS_PULLAWAY_MPS = 0.25
HYUNDAI_RECLAIM_RAW_SAFETY_MARGIN_M = 1.75
HYUNDAI_RECLAIM_RAW_SAFETY_CLOSING_MPS = 0.60
HYUNDAI_RECLAIM_RAW_SAFETY_DECEL_MPS2 = -0.8
HYUNDAI_RECLAIM_RAW_SAFETY_DECEL_CLOSING_MPS = 0.25
HYUNDAI_LEAD_TO_CRUISE_TRANSITION_RAMP_S = 3.0
HYUNDAI_LEAD_TO_CRUISE_TRANSITION_MIN_ACCEL = 0.25
# Close-range lead safety memory: if a lead was seen within this distance
# in the last N seconds, cap cruise accel even if the source flickers to cruise.
CLOSE_LEAD_MEMORY_DREL_M = 25.0       # leads within this distance are remembered
CLOSE_LEAD_MEMORY_HOLD_S = 5.0        # remember for this long after last sighting
CLOSE_LEAD_MEMORY_ACCEL_CAP = 0.30    # max cruise accel while memory is active (m/s²)
LEAD_PRESENT_CRUISE_SPEED_CAP_BP = [0.0, 2.0, 6.0, 10.0, 15.0, 25.0]
LEAD_PRESENT_CRUISE_SPEED_CAP_V = [0.7, 0.9, 1.3, 1.9, 2.4, ACCEL_MAX]
LEAD_PRESENT_CRUISE_SURPLUS_BP = [0.0, 2.0, 8.0, 16.0, 28.0]
LEAD_PRESENT_CRUISE_SURPLUS_V = [0.0, 0.0, 0.35, 0.70, 1.0]
LEAD_PRESENT_CRUISE_PULLAWAY_BP = [0.0, 0.4, 1.0, 2.0]
LEAD_PRESENT_CRUISE_PULLAWAY_V = [0.0, 0.08, 0.35, 1.0]
LEAD_PRESENT_CRUISE_MIN_SPEED = 4.0
LEAD_PRESENT_CRUISE_CLOSING_TIGHTEN_BP = [0.0, 1.0, 3.0, 5.0]
LEAD_PRESENT_CRUISE_CLOSING_TIGHTEN_V = [1.0, 0.65, 0.25, 0.10]
LEAD_PRESENT_CRUISE_CLOSING_PROJECTION_S = 3.0
LEAD_PRESENT_CRUISE_COAST_CLOSING_MPS = 1.0
LEAD_PRESENT_CRUISE_COAST_ACCEL_CAP = 0.0

# ---------------------------------------------------------------------------
# Lead distance prediction-corrector filter
# Smooths noisy AI-model dRel while preserving fast response to real changes.
# Parameters from simulation sweep (.cache/lead_filter_sim.py).
# ---------------------------------------------------------------------------
DREL_FILTER_TAU_CLOSE_S = 0.30   # fast response when lead appears closer (safety)
DREL_FILTER_TAU_OPEN_S = 1.00    # slow response when lead appears further (noise rejection)
DREL_FILTER_INNOVATION_GATE_M = 30.0
DREL_FILTER_CLOSING_GATE_M = 20.0
DREL_FILTER_INNOVATION_DEADBAND_M = 0.35
DREL_FILTER_OPEN_SLEW_MAX_MPS = 1.25
DREL_FILTER_SNAP_HOLD_FRAMES = 4
DREL_FILTER_ALPHA_FAST = 0.5


class _StabilizedLead:
  """Mutable duck-type of cereal.RadarState.LeadData.Reader. Mirrors the subset
  of attributes MPC callers read, so downstream code is oblivious to whether
  it's looking at a raw capnp reader or a phantom-extrapolated snapshot."""
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'aLeadK', 'modelProb',
               'dPath', 'vLat', 'aLeadTau', 'aRel', 'vLeadK', 'fcw',
               'fcwSuppressed', 'radar', 'radarTrackId')

  def __init__(self, status=False, dRel=0.0, yRel=0.0, vRel=0.0, vLead=0.0,
                aLeadK=0.0, modelProb=0.0, dPath=0.0, vLat=0.0, aLeadTau=0.0,
                aRel=0.0, vLeadK=0.0, fcw=False, fcwSuppressed=False, radar=False,
                radarTrackId=-1):
    self.status = bool(status)
    self.dRel = float(dRel)
    self.yRel = float(yRel)
    self.vRel = float(vRel)
    self.vLead = float(vLead)
    self.aLeadK = float(aLeadK)
    self.modelProb = float(modelProb)
    self.dPath = float(dPath)
    self.vLat = float(vLat)
    self.aLeadTau = float(aLeadTau)
    self.aRel = float(aRel)
    self.vLeadK = float(vLeadK)
    self.fcw = bool(fcw)
    self.fcwSuppressed = bool(fcwSuppressed)
    self.radar = bool(radar)
    self.radarTrackId = int(radarTrackId)

  @staticmethod
  def _safe_attr(src: Any, name: str, default: float = 0.0) -> float:
    try:
      val = getattr(src, name, default)
      val = float(val) if val is not None else float(default)
      return val if math.isfinite(val) else float(default)
    except Exception:
      return float(default)

  @classmethod
  def from_reader(cls, rd: Any) -> _StabilizedLead:
    if rd is None:
      return cls(status=False)
    return cls(
      status=bool(getattr(rd, 'status', False)),
      dRel=cls._safe_attr(rd, 'dRel'),
      yRel=cls._safe_attr(rd, 'yRel'),
      vRel=cls._safe_attr(rd, 'vRel'),
      vLead=cls._safe_attr(rd, 'vLead'),
      aLeadK=cls._safe_attr(rd, 'aLeadK'),
      modelProb=cls._safe_attr(rd, 'modelProb'),
      dPath=cls._safe_attr(rd, 'dPath'),
      vLat=cls._safe_attr(rd, 'vLat'),
      aLeadTau=cls._safe_attr(rd, 'aLeadTau'),
      aRel=cls._safe_attr(rd, 'aRel'),
      vLeadK=cls._safe_attr(rd, 'vLeadK'),
      fcw=bool(getattr(rd, 'fcw', False)),
      fcwSuppressed=bool(getattr(rd, 'fcwSuppressed', False)),
      radar=bool(getattr(rd, 'radar', False)),
      radarTrackId=int(getattr(rd, 'radarTrackId', -1) or -1),
    )


# Smoothing for the measured d(aLeadK)/dt used by the phantom trend hold.
LEAD_STABILIZER_TREND_TAU_S = 0.20
# Identity gates for the trend measurement: a same-slot track swap (cut-in
# replacing the tracked lead) steps aLeadK across two different physical cars,
# which is not a measurement. Detected as dRel far off the propagated position
# or a lateral jump between consecutive valid frames.
LEAD_STABILIZER_TREND_DREL_JUMP_M = 3.0
LEAD_STABILIZER_TREND_YREL_JUMP_M = 1.5

# aLeadK corroboration bound (uncorroborated transient lead-decel guard).
# LeadAccelCorrMarginMps2 at/above the disable value passes aLeadK through.
LEAD_ACCEL_CORR_DISABLE_MARGIN_MPS2 = 10.0
# A vLead finite-difference beyond this physical-accel gate is a track identity
# change (lead swap), not a measurement: reset and re-settle, never clamp with it.
LEAD_ACCEL_CORR_MEAS_A_GATE_MPS2 = 10.0
# Bound stays inactive until this many taus of same-track vLead history exist.
LEAD_ACCEL_CORR_SETTLE_TAU_MULT = 2.0
# Dangerous-state bypass hysteresis: once latched, the bypass only disengages
# when closing/TTC/gap clear the guards by these margins, so noisy stabilized
# vRel hovering at a guard boundary cannot chatter aLeadK in the MPC horizon.
LEAD_ACCEL_CORR_CLOSING_REARM_MPS = 0.5
LEAD_ACCEL_CORR_TTC_REARM_S = 2.0
LEAD_ACCEL_CORR_HEADWAY_REARM_M = 2.0
LEAD_ACCEL_CORR_MAX_DT_S = 0.5
# aLeadK amplify (CD3 lead-decel truth deficit). When the model already reports
# braking and the corroborating vLead trend is meaningfully deeper, pull aLeadK
# toward the measured trend. Gain 0 disables (rollback to bound-only). Deadband
# rejects steady/lightly-braking finite-difference jitter; cap bounds how far a
# single noisy trend sample can deepen aLeadK in one frame.
LEAD_ACCEL_CORR_AMPLIFY_GAIN = 0.0
LEAD_ACCEL_CORR_AMPLIFY_DEADBAND_MPS2 = 0.35
LEAD_ACCEL_CORR_AMPLIFY_CAP_MPS2 = 2.0


class _LeadStabilityState:
  """Per-slot state for the acquire/release dwell + phantom extrapolation filter."""
  __slots__ = ('latched', 'valid_streak', 'invalid_streak',
               'latched_valid_streak', 'last_valid', 'last_valid_t',
               'a_lead_k_trend', 'corr_meas_t', 'corr_meas_v',
               'corr_a_meas_lp', 'corr_settled_s', 'corr_track_id',
               'corr_danger_latched')

  def __init__(self):
    self.latched = False
    self.valid_streak = 0
    self.invalid_streak = 0
    self.latched_valid_streak = 0
    self.last_valid: _StabilizedLead | None = None
    self.last_valid_t: float | None = None
    # EMA-smoothed d(aLeadK)/dt over valid frames; lets the phantom continue
    # the measured convergence of the (lagged) lead-accel estimate.
    self.a_lead_k_trend = 0.0
    # Low-passed finite-difference of stabilized vLead used to corroborate
    # negative aLeadK; frozen (not decayed) while the slot has no fresh
    # measurement, reset on track identity changes.
    self.corr_meas_t: float | None = None
    self.corr_meas_v = 0.0
    self.corr_a_meas_lp = 0.0
    self.corr_settled_s = 0.0
    self.corr_track_id: int | None = None
    self.corr_danger_latched = False


class LeadDistanceFilter:
  __slots__ = ('_filtered', '_frames_since_snap', 'last_debug')

  def __init__(self):
    self._filtered: float | None = None
    self._frames_since_snap: int = DREL_FILTER_SNAP_HOLD_FRAMES + 1
    self.last_debug: dict[str, float | bool] = {}
    self.reset()

  def reset(self, drel: float | None = None) -> None:
    self._filtered = drel
    self._frames_since_snap = DREL_FILTER_SNAP_HOLD_FRAMES + 1
    self.last_debug = {
      "predicted_vrel_mps": 0.0,
      "innovation_raw_m": 0.0,
      "innovation_used_m": 0.0,
      "deadband_applied": False,
      "open_slew_clamped": False,
      "snap_to_raw": False,
      "opening_vrel_suppressed": False,
    }

  @property
  def value(self) -> float | None:
    return self._filtered

  def update(self, raw_drel: float, raw_vrel: float, dt_s: float,
             tau_close: float = DREL_FILTER_TAU_CLOSE_S,
             tau_open: float = DREL_FILTER_TAU_OPEN_S,
             innovation_gate: float = DREL_FILTER_INNOVATION_GATE_M,
             closing_gate: float = DREL_FILTER_CLOSING_GATE_M,
             open_slew_max_mps: float = DREL_FILTER_OPEN_SLEW_MAX_MPS) -> float:
    if self._filtered is None or dt_s <= 0.0:
      self._filtered = raw_drel
      self._frames_since_snap = DREL_FILTER_SNAP_HOLD_FRAMES + 1
      self.last_debug = {
        "predicted_vrel_mps": float(raw_vrel),
        "innovation_raw_m": 0.0,
        "innovation_used_m": 0.0,
        "deadband_applied": False,
        "open_slew_clamped": False,
        "snap_to_raw": True,
        "opening_vrel_suppressed": False,
      }
      return raw_drel

    prev_filtered = float(self._filtered)
    predicted_vrel = float(raw_vrel)
    opening_vrel_suppressed = False
    if predicted_vrel > 0.0:
      opening_evidence_m = float(raw_drel) - prev_filtered
      if opening_evidence_m <= DREL_FILTER_INNOVATION_DEADBAND_M:
        predicted_vrel = 0.0
        opening_vrel_suppressed = True
      else:
        predicted_vrel = min(predicted_vrel, max(0.0, float(open_slew_max_mps)))

    d_pred = prev_filtered + predicted_vrel * dt_s
    innov_raw = raw_drel - d_pred
    deadband = DREL_FILTER_INNOVATION_DEADBAND_M
    if abs(innov_raw) <= deadband:
      innov = 0.0
      deadband_applied = True
    else:
      innov = float(np.sign(innov_raw) * (abs(innov_raw) - deadband))
      deadband_applied = False
    open_slew_clamped = False
    snapped = False

    if abs(innov_raw) > innovation_gate or innov_raw < -closing_gate:
      self._filtered = raw_drel
      self._frames_since_snap = 0
      snapped = True
    elif self._frames_since_snap < DREL_FILTER_SNAP_HOLD_FRAMES:
      self._filtered = d_pred + DREL_FILTER_ALPHA_FAST * innov
      self._frames_since_snap += 1
    else:
      tau = tau_close if innov_raw < 0.0 else tau_open
      alpha = float(1.0 - np.exp(-dt_s / max(tau, 1e-3)))
      self._filtered = d_pred + alpha * innov
      if raw_drel >= prev_filtered:
        max_open_step = max(0.0, float(open_slew_max_mps)) * dt_s
        max_open_value = prev_filtered + max_open_step
        if self._filtered > max_open_value:
          self._filtered = max_open_value
          open_slew_clamped = True

    self.last_debug = {
      "predicted_vrel_mps": float(predicted_vrel),
      "innovation_raw_m": float(innov_raw),
      "innovation_used_m": float(innov),
      "deadband_applied": bool(deadband_applied),
      "open_slew_clamped": bool(open_slew_clamped),
      "snap_to_raw": bool(snapped),
      "opening_vrel_suppressed": bool(opening_vrel_suppressed),
    }

    return self._filtered


# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = DEFAULT_COMFORT_BRAKE
STOP_DISTANCE = 6.0
CRUISE_MIN_ACCEL = DEFAULT_CRUISE_MIN_ACCEL
CRUISE_MAX_ACCEL = DEFAULT_CRUISE_MAX_ACCEL

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.6
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.80
  elif personality==log.LongitudinalPersonality.standard:
    return 1.50
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.20
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def get_headway_follow_distance(v_ego, t_follow):
  return STOP_DISTANCE + t_follow * v_ego

def desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)


def compute_lead_approach_preview(v_ego, lead, t_follow,
                                  tuning: LeadResponseTuningConfig | None = None,
                                  *,
                                  acquire_window_active: bool = False) -> tuple[float, dict[str, float | bool | str | None]]:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  debug: dict[str, float | bool | str | None] = {
    "active": False,
    "mode": "inactive",
    "acquire_window_active": bool(acquire_window_active),
    "closing_speed_mps": 0.0,
    "gap_surplus_m": 0.0,
    "headway_gap_m": 0.0,
    "ttc_to_headway_s": None,
    "projected_deficit_m": 0.0,
    "preview_time_s": 0.0,
    "anticipatory_ttc_blend": 0.0,
    "anticipatory_buffer_m": 0.0,
    "preview_buffer_m": 0.0,
  }
  if lead is None or not getattr(lead, 'status', False):
    return 0.0, debug

  # Low-speed preview fade (knife-edge hardening): the historical hard cut at
  # v_ego < LEAD_APPROACH_PREVIEW_MIN_SPEED steps the previewed lead obstacle
  # by up to lead_preview_max_buffer_m (10 m) instantaneously whenever the
  # drifting v_desired filter state wobbles across 8.0 m/s — a latent jerk
  # source whenever the lead already owns the obstacle. Instead, fade the
  # preview linearly from zero at LeadPreviewMinSpeedMps up to full strength
  # at LEAD_APPROACH_PREVIEW_MIN_SPEED. Activity superset at defaults: at or
  # above 8.0 m/s the fade is exactly 1.0 (bit-identical to the hard gate);
  # below it the fade only ADDS preview that used to be zero. Rollback:
  # LeadPreviewMinSpeedMps = 8.0 reproduces the hard cut exactly. The acquire
  # window path is untouched (fade forced to 1.0 there, as before).
  min_speed_fade = 1.0
  if not acquire_window_active:
    fade_lo = float(np.clip(float(tuning.lead_preview_min_speed_mps), 0.0, LEAD_APPROACH_PREVIEW_MIN_SPEED))
    if fade_lo >= LEAD_APPROACH_PREVIEW_MIN_SPEED:
      min_speed_fade = 1.0 if v_ego >= LEAD_APPROACH_PREVIEW_MIN_SPEED else 0.0
    else:
      min_speed_fade = float(np.interp(v_ego, [fade_lo, LEAD_APPROACH_PREVIEW_MIN_SPEED], [0.0, 1.0]))
    if min_speed_fade <= 0.0:
      debug["min_speed_fade"] = 0.0
      return 0.0, debug

  low_speed_acquire = bool(acquire_window_active and float(v_ego) < LEAD_APPROACH_PREVIEW_MIN_SPEED)

  v_lead_raw = float(getattr(lead, 'vLead', v_ego) or v_ego)
  v_lead = max(0.0, v_lead_raw)
  v_rel = float(getattr(lead, 'vRel', v_lead_raw - float(v_ego)) or (v_lead_raw - float(v_ego)))
  closing_speed = max(0.0, float(v_ego) - v_lead)
  if low_speed_acquire:
    closing_speed = max(closing_speed, -v_rel)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  headway_gap = get_headway_follow_distance(float(v_ego), t_follow)
  gap_surplus = d_rel - headway_gap
  preview_time = float(np.interp(closing_speed, LEAD_APPROACH_PREVIEW_TIME_BP, LEAD_APPROACH_PREVIEW_TIME_V))
  gap_min_m = float(tuning.lead_preview_gap_min_m)
  if acquire_window_active:
    gap_min_m *= LEAD_APPROACH_PREVIEW_ACQUIRE_GAP_FRACTION
  if closing_speed <= LEAD_APPROACH_PREVIEW_CLOSING_MIN_MPS and gap_surplus <= gap_min_m:
    return 0.0, debug

  lead_decel = max(0.0, -float(getattr(lead, 'aLeadK', 0.0) or 0.0))
  projected_horizon_s = preview_time + (float(tuning.lead_acquire_window_s) if acquire_window_active else 0.0)
  projected_gap = max(0.0, d_rel - closing_speed * projected_horizon_s - 0.5 * lead_decel * (projected_horizon_s ** 2))
  projected_deficit = max(0.0, headway_gap - projected_gap)
  if gap_surplus > 0.0 and closing_speed > LEAD_APPROACH_PREVIEW_CLOSING_MIN_MPS:
    ttc_to_headway = gap_surplus / max(closing_speed, 1e-3)
  else:
    ttc_to_headway = None

  max_buffer_fraction = LEAD_APPROACH_PREVIEW_ACQUIRE_GAP_CAP_FRACTION if acquire_window_active else LEAD_APPROACH_PREVIEW_GAP_CAP_FRACTION
  max_buffer = min(tuning.lead_preview_max_buffer_m, max(0.0, gap_surplus) * max_buffer_fraction)
  if max_buffer <= 0.0 and low_speed_acquire:
    tight_gap_deficit = max(max(0.0, headway_gap - d_rel), projected_deficit)
    max_buffer = min(
      tuning.lead_preview_max_buffer_m * LEAD_APPROACH_PREVIEW_ACQUIRE_TIGHT_GAP_MAX_FRACTION,
      tight_gap_deficit * LEAD_APPROACH_PREVIEW_ACQUIRE_TIGHT_GAP_GAIN,
    )
  if max_buffer <= 0.0:
    return 0.0, debug

  base_preview = closing_speed * preview_time
  projected_preview = projected_deficit * LEAD_APPROACH_PREVIEW_PROJECTED_DEFICIT_GAIN
  anticipatory_ttc_blend = 0.0
  anticipatory_buffer = 0.0
  if (ttc_to_headway is not None and
      closing_speed >= LEAD_APPROACH_PREVIEW_ANTICIPATORY_CLOSING_MIN_MPS and
      gap_surplus > gap_min_m):
    anticipatory_ttc_blend = float(np.interp(
      float(ttc_to_headway),
      LEAD_APPROACH_PREVIEW_ANTICIPATORY_TTC_BP,
      LEAD_APPROACH_PREVIEW_ANTICIPATORY_GAP_FRACTION_V,
    ))
    anticipatory_buffer = max(0.0, gap_surplus) * max(0.0, anticipatory_ttc_blend)
  preview_buffer = max(base_preview, projected_preview, anticipatory_buffer)
  lead_decel_scale = float(np.interp(lead_decel, LEAD_APPROACH_PREVIEW_DECEL_BP, LEAD_APPROACH_PREVIEW_DECEL_V))
  if acquire_window_active:
    preview_buffer *= LEAD_APPROACH_PREVIEW_ACQUIRE_GAIN
  preview_buffer *= lead_decel_scale * tuning.lead_preview_strength
  preview_buffer *= min_speed_fade
  preview_buffer = float(np.clip(preview_buffer, 0.0, max_buffer))
  debug = {
    "active": bool(preview_buffer > 0.0),
    "mode": "acquire" if acquire_window_active and preview_buffer > 0.0 else ("base" if preview_buffer > 0.0 else "inactive"),
    "acquire_window_active": bool(acquire_window_active),
    "closing_speed_mps": float(closing_speed),
    "gap_surplus_m": float(gap_surplus),
    "headway_gap_m": float(headway_gap),
    "ttc_to_headway_s": None if ttc_to_headway is None else float(ttc_to_headway),
    "projected_deficit_m": float(projected_deficit),
    "preview_time_s": float(preview_time),
    "anticipatory_ttc_blend": float(anticipatory_ttc_blend),
    "anticipatory_buffer_m": float(anticipatory_buffer),
    "min_speed_fade": float(min_speed_fade),
    "preview_buffer_m": float(preview_buffer),
  }
  return preview_buffer, debug


def get_lead_approach_preview_buffer(v_ego, lead, t_follow,
                                     tuning: LeadResponseTuningConfig | None = None,
                                     *,
                                     acquire_window_active: bool = False) -> float:
  preview_buffer, _ = compute_lead_approach_preview(
    v_ego,
    lead,
    t_follow,
    tuning,
    acquire_window_active=acquire_window_active,
  )
  return preview_buffer


def apply_lead_approach_preview(lead_obstacle, preview_buffer_m):
  if preview_buffer_m <= 0.0:
    return lead_obstacle

  decay = np.exp(-T_IDXS / LEAD_APPROACH_PREVIEW_DECAY_TAU)
  return np.maximum(lead_obstacle - preview_buffer_m * decay, 0.0)


def compute_lead_handoff_danger_factor(v_ego, lead, t_follow,
                                       tuning: LeadResponseTuningConfig | None = None,
                                       *,
                                       handoff_remaining_s: float = 0.0) -> tuple[float, dict[str, float | bool | None]]:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  debug: dict[str, float | bool | None] = {
    "active": False,
    "danger_factor": float(LEAD_DANGER_FACTOR),
    "closing_speed_mps": 0.0,
    "current_deficit_m": 0.0,
    "projected_deficit_m": 0.0,
    "window_scale": 0.0,
    "severity_scale": 0.0,
  }
  if lead is None or not getattr(lead, 'status', False) or handoff_remaining_s <= 0.0:
    return float(LEAD_DANGER_FACTOR), debug
  if v_ego < LEAD_HANDOFF_DANGER_MIN_SPEED:
    return float(LEAD_DANGER_FACTOR), debug

  v_lead_raw = float(getattr(lead, 'vLead', v_ego) or v_ego)
  v_lead = max(0.0, v_lead_raw)
  v_rel = float(getattr(lead, 'vRel', v_lead_raw - float(v_ego)) or (v_lead_raw - float(v_ego)))
  closing_speed = max(0.0, float(v_ego) - v_lead, -v_rel)
  if closing_speed <= LEAD_HANDOFF_DANGER_CLOSING_MIN_MPS:
    return float(LEAD_DANGER_FACTOR), debug

  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  headway_gap = get_headway_follow_distance(float(v_ego), t_follow)
  current_deficit = max(0.0, headway_gap - d_rel)
  lead_decel = max(0.0, -float(getattr(lead, 'aLeadK', 0.0) or 0.0))
  projected_gap = max(
    0.0,
    d_rel - closing_speed * LEAD_HANDOFF_DANGER_HORIZON_S - 0.5 * lead_decel * (LEAD_HANDOFF_DANGER_HORIZON_S ** 2),
  )
  projected_deficit = max(0.0, headway_gap - projected_gap)
  severity_deficit = max(current_deficit, projected_deficit)
  if severity_deficit <= 0.0:
    return float(LEAD_DANGER_FACTOR), debug

  default_preview_strength = LeadResponseTuningConfig.defaults().lead_preview_strength
  window_scale = float(np.clip(handoff_remaining_s / max(float(tuning.lead_acquire_window_s), 1e-3), 0.0, 1.0))
  severity_scale = float(np.clip(severity_deficit / LEAD_HANDOFF_DANGER_DEFICIT_REF_M, 0.0, 1.0))
  closing_scale = float(np.interp(closing_speed, LEAD_HANDOFF_DANGER_CLOSING_BP, LEAD_HANDOFF_DANGER_CLOSING_V))
  gain_scale = float(np.clip(float(tuning.lead_preview_strength) / max(default_preview_strength, 1e-3), 0.5, 1.5))
  boost_scale = float(np.clip(window_scale * severity_scale * closing_scale * gain_scale, 0.0, 1.0))
  danger_factor = float(np.clip(
    LEAD_DANGER_FACTOR + (LEAD_HANDOFF_DANGER_MAX_FACTOR - LEAD_DANGER_FACTOR) * boost_scale,
    LEAD_DANGER_FACTOR,
    LEAD_HANDOFF_DANGER_MAX_FACTOR,
  ))
  debug = {
    "active": bool(danger_factor > LEAD_DANGER_FACTOR),
    "danger_factor": float(danger_factor),
    "closing_speed_mps": float(closing_speed),
    "current_deficit_m": float(current_deficit),
    "projected_deficit_m": float(projected_deficit),
    "window_scale": float(window_scale),
    "severity_scale": float(severity_scale),
  }
  return danger_factor, debug


def get_lead_handoff_danger_factor(v_ego, lead, t_follow,
                                   tuning: LeadResponseTuningConfig | None = None,
                                   *,
                                   handoff_remaining_s: float = 0.0) -> float:
  danger_factor, _ = compute_lead_handoff_danger_factor(
    v_ego,
    lead,
    t_follow,
    tuning,
    handoff_remaining_s=handoff_remaining_s,
  )
  return danger_factor


def get_gap_reclaim_effective_cap(v_ego, lead, t_follow,
                                  tuning: LeadResponseTuningConfig | None = None,
                                  personality_max_accel: float | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  comfort_cap = max(float(tuning.gap_reclaim_max_accel), 1e-3)
  low_speed_launch_factor = get_low_speed_launch_follow_factor(v_ego, lead, t_follow)
  launch_active = low_speed_launch_factor > 0.0
  if (lead is None or not getattr(lead, 'status', False) or
      (v_ego < GAP_RECLAIM_MIN_SPEED and not launch_active) or
      personality_max_accel is None):
    return comfort_cap

  personality_cap = max(comfort_cap, float(personality_max_accel))
  if launch_active:
    personality_cap = max(
      personality_cap,
      get_low_speed_launch_follow_max_accel(v_ego, lead, t_follow, comfort_cap),
    )
  if personality_cap <= comfort_cap:
    return comfort_cap

  v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  gap_surplus = d_rel - get_headway_follow_distance(float(v_ego), t_follow)
  if gap_surplus <= tuning.gap_reclaim_gap_min_m:
    return comfort_cap

  headway_surplus = gap_surplus / max(float(v_ego), GAP_RECLAIM_MIN_SPEED)
  pullaway_speed = max(0.0, v_lead - float(v_ego))
  surplus_blend = float(np.interp(headway_surplus, GAP_RECLAIM_PERSONALITY_SURPLUS_BP, GAP_RECLAIM_PERSONALITY_SURPLUS_V))
  pullaway_blend = float(np.interp(pullaway_speed, GAP_RECLAIM_PERSONALITY_PULLAWAY_BP, GAP_RECLAIM_PERSONALITY_PULLAWAY_V))
  surplus_activation = float(np.interp(headway_surplus, GAP_RECLAIM_PERSONALITY_SURPLUS_BP, [0.0, 0.15, 0.55, 1.0]))
  cap_blend = max(surplus_blend, pullaway_blend * surplus_activation)
  if v_ego < GAP_RECLAIM_MIN_SPEED:
    cap_blend = max(cap_blend, low_speed_launch_factor)
  return float(comfort_cap + (personality_cap - comfort_cap) * cap_blend)


def get_gap_reclaim_projection_scale(v_ego, lead, t_follow, ego_accel: float = 0.0) -> float:
  if lead is None or not getattr(lead, 'status', False) or v_ego < GAP_RECLAIM_MIN_SPEED:
    return 1.0

  ego_accel = max(float(ego_accel), 0.0)
  accel_activation = float(np.interp(ego_accel, GAP_RECLAIM_PROJECT_EGO_ACCEL_BP, GAP_RECLAIM_PROJECT_EGO_ACCEL_V))
  if accel_activation <= 0.0:
    return 1.0

  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
  lead_accel = float(getattr(lead, 'aLeadK', 0.0) or 0.0)
  gap_surplus = max(0.0, d_rel - get_headway_follow_distance(float(v_ego), t_follow))
  if gap_surplus <= 0.0:
    return 1.0

  pullaway_speed = max(0.0, v_lead - float(v_ego))
  horizon_s = GAP_RECLAIM_PROJECT_HORIZON_S
  projected_gap_surplus = max(
    0.0,
    gap_surplus +
    (v_lead - float(v_ego)) * horizon_s +
    0.5 * (lead_accel - ego_accel) * (horizon_s ** 2),
  )
  gap_scale = float(np.clip(projected_gap_surplus / max(gap_surplus, 1e-3), 0.0, 1.0))

  if pullaway_speed > 0.05:
    projected_pullaway_speed = max(0.0, pullaway_speed + (lead_accel - ego_accel) * horizon_s)
    pullaway_scale = float(np.clip(projected_pullaway_speed / max(pullaway_speed, 1e-3), 0.0, 1.0))
  else:
    pullaway_scale = gap_scale

  projection_scale = (
    GAP_RECLAIM_PROJECT_GAP_WEIGHT * gap_scale +
    GAP_RECLAIM_PROJECT_PULLAWAY_WEIGHT * pullaway_scale
  )
  return float(np.clip(1.0 - accel_activation * (1.0 - projection_scale), 0.0, 1.0))


def _lead_float(lead, attr: str, default: float) -> float:
  value = getattr(lead, attr, default)
  if value is None:
    return float(default)
  return float(value)


def get_gap_reclaim_accel_floor(v_ego, lead, t_follow,
                                tuning: LeadResponseTuningConfig | None = None,
                                personality_max_accel: float | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  low_speed_launch_factor = get_low_speed_launch_follow_factor(v_ego, lead, t_follow)
  launch_active = low_speed_launch_factor > 0.0
  if lead is None or not getattr(lead, 'status', False) or (v_ego < GAP_RECLAIM_MIN_SPEED and not launch_active):
    return 0.0

  v_lead = _lead_float(lead, 'vLead', float(v_ego))
  if v_lead < float(v_ego) - 0.3:
    return 0.0

  lead_accel = float(getattr(lead, 'aLeadK', 0.0) or 0.0)
  if lead_accel < -0.4:
    return 0.0

  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  gap_surplus = d_rel - get_headway_follow_distance(float(v_ego), t_follow)
  if gap_surplus <= tuning.gap_reclaim_gap_min_m:
    return 0.0

  headway_surplus = gap_surplus / max(float(v_ego), GAP_RECLAIM_MIN_SPEED)
  pullaway_speed = max(0.0, v_lead - float(v_ego))
  gap_term = float(np.interp(headway_surplus, GAP_RECLAIM_HEADWAY_SURPLUS_BP, GAP_RECLAIM_HEADWAY_SURPLUS_V))
  pullaway_term = float(np.interp(pullaway_speed, GAP_RECLAIM_PULLAWAY_BP, GAP_RECLAIM_PULLAWAY_V))
  accel_term = float(np.interp(max(0.0, lead_accel), GAP_RECLAIM_LEAD_ACCEL_BP, GAP_RECLAIM_LEAD_ACCEL_V))
  comfort_cap = max(float(tuning.gap_reclaim_max_accel), 1e-3)
  intent = (max(gap_term, pullaway_term) + accel_term) * tuning.gap_reclaim_strength
  intent_fraction = float(np.clip(intent / comfort_cap, 0.0, 1.0))
  effective_cap = get_gap_reclaim_effective_cap(
    v_ego,
    lead,
    t_follow,
    tuning,
    personality_max_accel=personality_max_accel,
  )
  if v_ego < GAP_RECLAIM_MIN_SPEED:
    effective_cap = float(comfort_cap + (effective_cap - comfort_cap) * low_speed_launch_factor)
  floor = intent_fraction * effective_cap
  return float(np.clip(floor, 0.0, effective_cap))


def get_lead_keepup_accel_floor(v_ego, lead, t_follow,
                                tuning: LeadResponseTuningConfig | None = None,
                                personality_max_accel: float | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  v_ego = float(v_ego)
  if lead is None or not getattr(lead, 'status', False) or v_ego < LEAD_KEEPUP_MIN_SPEED:
    return 0.0

  v_lead = _lead_float(lead, 'vLead', v_ego)
  v_rel = _lead_float(lead, 'vRel', v_lead - v_ego)
  lead_accel = float(getattr(lead, 'aLeadK', 0.0) or 0.0)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  gap_surplus = d_rel - get_headway_follow_distance(v_ego, t_follow)
  closing_speed = max(0.0, v_ego - v_lead, -v_rel)
  pullaway_speed = max(0.0, v_lead - v_ego, v_rel)

  if gap_surplus < -LEAD_KEEPUP_TOO_CLOSE_MARGIN_M:
    return 0.0
  if closing_speed > LEAD_KEEPUP_CLOSING_BLOCK_MPS:
    return 0.0
  if lead_accel < LEAD_KEEPUP_LEAD_DECEL_BLOCK_MPS2:
    return 0.0

  gap_active = max(0.0, gap_surplus - float(tuning.lead_keepup_gap_min_m))
  if pullaway_speed <= LEAD_KEEPUP_SOFT_PULLAWAY_BP[1] and gap_active <= 0.0:
    return 0.0

  cap = max(float(tuning.lead_keepup_max_accel), 0.0)
  if personality_max_accel is not None:
    cap = min(cap, max(0.0, float(personality_max_accel)))
  if gap_surplus < 0.0:
    cap *= float(np.clip((gap_surplus + LEAD_KEEPUP_TOO_CLOSE_MARGIN_M) / LEAD_KEEPUP_TOO_CLOSE_MARGIN_M, 0.0, 1.0))

  soft_floor = float(np.interp(pullaway_speed, LEAD_KEEPUP_SOFT_PULLAWAY_BP, LEAD_KEEPUP_SOFT_PULLAWAY_V))
  pullaway_cap_scale = float(np.interp(pullaway_speed, LEAD_KEEPUP_CAP_PULLAWAY_BP, LEAD_KEEPUP_CAP_PULLAWAY_V))
  gap_cap_scale = float(np.interp(gap_active, LEAD_KEEPUP_CAP_GAP_SURPLUS_BP, LEAD_KEEPUP_CAP_GAP_SURPLUS_V))
  speed_floor = cap * max(pullaway_cap_scale, gap_cap_scale)

  pullaway_accel_gate = float(np.interp(pullaway_speed, LEAD_KEEPUP_ACCEL_GATE_PULLAWAY_BP, LEAD_KEEPUP_ACCEL_GATE_PULLAWAY_V))
  gap_accel_gate = float(np.interp(gap_active, LEAD_KEEPUP_ACCEL_GATE_GAP_BP, LEAD_KEEPUP_ACCEL_GATE_GAP_V))
  accel_gate = max(pullaway_accel_gate, gap_accel_gate)
  accel_floor = min(cap, max(0.0, lead_accel) * LEAD_KEEPUP_ACCEL_OVERSHOOT) * accel_gate

  intent = max(soft_floor, speed_floor, accel_floor) * float(tuning.lead_keepup_strength)
  return float(np.clip(intent, 0.0, cap))


def get_lead_slowdown_accel_ceiling(v_ego, lead, t_follow,
                                    tuning: LeadResponseTuningConfig | None = None,
                                    min_accel: float = ACCEL_MIN,
                                    max_accel: float = ACCEL_MAX,
                                    anticipatory_enabled: bool = True) -> float | None:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  v_ego = float(v_ego)
  if lead is None or not getattr(lead, 'status', False) or v_ego < LEAD_SLOWDOWN_MIN_SPEED:
    return None

  max_decel = min(abs(float(min_accel)), max(0.0, float(tuning.lead_slowdown_max_decel)))
  if max_decel <= 0.0:
    return None

  v_lead_raw = _lead_float(lead, 'vLead', v_ego)
  v_lead = max(0.0, v_lead_raw)
  v_rel = _lead_float(lead, 'vRel', v_lead_raw - v_ego)
  closing_speed = max(0.0, v_ego - v_lead, -v_rel)
  pullaway_speed = max(0.0, v_lead - v_ego, v_rel)
  lead_decel = max(0.0, -float(getattr(lead, 'aLeadK', 0.0) or 0.0))
  if closing_speed < LEAD_SLOWDOWN_MIN_CLOSING_MPS and lead_decel < LEAD_SLOWDOWN_MIN_LEAD_DECEL_MPS2:
    return None

  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  headway_gap = get_headway_follow_distance(v_ego, t_follow)
  danger_gap = LEAD_DANGER_FACTOR * headway_gap
  gap_surplus = d_rel - headway_gap
  danger_surplus = d_rel - danger_gap
  if pullaway_speed > 0.5 and closing_speed <= LEAD_SLOWDOWN_MIN_CLOSING_MPS:
    return LEAD_SLOWDOWN_SOFT_ACCEL_CAP if gap_surplus < 0.0 else None

  horizon_s = LEAD_SLOWDOWN_HORIZON_S
  projected_gap = max(0.0, d_rel - closing_speed * horizon_s - 0.5 * lead_decel * (horizon_s ** 2))
  projected_headway_deficit = max(0.0, headway_gap - projected_gap)
  projected_danger_deficit = max(0.0, danger_gap - projected_gap)
  if pullaway_speed > 0.5 and gap_surplus > 0.0 and projected_headway_deficit <= 0.0:
    return None

  if closing_speed > 0.0:
    if gap_surplus > 0.0:
      ttc_headway = gap_surplus / max(closing_speed, 1e-3)
    else:
      ttc_headway = 0.0
    if danger_surplus > 0.0:
      ttc_danger = danger_surplus / max(closing_speed, 1e-3)
    else:
      ttc_danger = 0.0
    ttc_collision = d_rel / max(closing_speed, 1e-3)
  else:
    ttc_headway = 1e6
    ttc_danger = 1e6
    ttc_collision = 1e6

  gap_gate = float(np.interp(gap_surplus, LEAD_SLOWDOWN_GAP_GATE_BP, LEAD_SLOWDOWN_GAP_GATE_V))
  headway_deficit_gate = float(np.interp(projected_headway_deficit, LEAD_SLOWDOWN_HEADWAY_DEFICIT_BP, LEAD_SLOWDOWN_HEADWAY_DEFICIT_V))
  closing_match_gate = float(np.interp(closing_speed, LEAD_SLOWDOWN_CLOSING_MATCH_BP, LEAD_SLOWDOWN_CLOSING_MATCH_V))
  lead_match_gate = max(gap_gate * closing_match_gate, headway_deficit_gate)
  if gap_surplus <= 0.0:
    lead_match_gate = 1.0
  lead_match_decel = lead_decel * LEAD_SLOWDOWN_LEAD_DECEL_OVERSHOOT * lead_match_gate

  if closing_speed > 0.0:
    headway_required_gap = gap_surplus if gap_surplus > 0.0 else max(d_rel, 1.0)
    headway_required_decel = (closing_speed ** 2) / (2.0 * max(headway_required_gap, 0.5))
  else:
    headway_required_decel = 0.0
  ttc_headway_gate = float(np.interp(ttc_headway, LEAD_SLOWDOWN_TTC_HEADWAY_BP, LEAD_SLOWDOWN_TTC_HEADWAY_V))
  closing_decel = headway_required_decel * max(headway_deficit_gate, ttc_headway_gate)
  anticipatory_headway_gate = 0.0
  if (anticipatory_enabled and
      closing_speed >= LEAD_SLOWDOWN_ANTICIPATORY_CLOSING_MIN_MPS and
      gap_surplus > 0.0):
    anticipatory_headway_gate = float(np.interp(
      ttc_headway,
      LEAD_APPROACH_PREVIEW_ANTICIPATORY_TTC_BP,
      LEAD_APPROACH_PREVIEW_ANTICIPATORY_GAP_FRACTION_V,
    ))
  anticipatory_decel = (
    headway_required_decel + lead_decel * LEAD_SLOWDOWN_ANTICIPATORY_LEAD_DECEL_GAIN
  ) * max(0.0, anticipatory_headway_gate)

  danger_required_decel = 0.0
  if closing_speed > 0.0 or lead_decel > 0.0:
    danger_required_gap = danger_surplus if danger_surplus > 0.0 else max(d_rel - CRASH_DISTANCE, 1.0)
    danger_required_decel = (closing_speed ** 2) / (2.0 * max(danger_required_gap, 0.3)) + lead_decel * LEAD_SLOWDOWN_LEAD_DECEL_OVERSHOOT
  # Energy-consistency bound (calm-stop slam fix): danger_surplus measures the
  # distance to 0.75*headway, a line the natural stop point sits INSIDE at low
  # speed, so on every normal stop the denominator above collapses to its 0.3 m
  # floor and the danger term saturates to lead_slowdown_max_decel. Physics is
  # the collapse-proof reference instead: the gated danger demand may never
  # exceed `headroom` times the decel actually required to stop `margin` short
  # of the lead (including the lead's remaining braking distance when it is
  # slowing). Genuine threats keep full authority by construction: when the
  # available stopping distance is truly small the physical requirement itself
  # is large (and the bound disappears once the lead is projected to stop
  # inside `margin`), and matching a braking lead's own decel (with overshoot)
  # is never capped. Applied AFTER the danger gates (see danger_decel below) so
  # it only trims demands that exceed K x physics, never the gate softening.
  kinematic_headroom = max(1.0, float(tuning.lead_slowdown_kinematic_headroom))
  # Margin is clamped in code, not just in the tunable spec: values at/above
  # STOP_DISTANCE (6 m) inflate the bound near the natural stop point enough to
  # fully readmit the calm-stop slam (verified: margin=6.0 reproduces the legacy
  # -4.0 saturation), and margin below ~1 m starves the ceiling of its stop-gap
  # reserve AND makes the inside-margin full-authority restoration unreachable.
  # No live-tunable (or direct param) value may cross either line.
  kinematic_margin = float(np.clip(float(tuning.lead_slowdown_kinematic_margin_m),
                                   LEAD_SLOWDOWN_KINEMATIC_MARGIN_MIN_M,
                                   STOP_DISTANCE - 1.0))
  # Closing speed for the bound comes from v_lead = max(0, vLead), NOT from the
  # published closing_speed above. radard publishes vLead = v_ego + vRel
  # (radard.py ModelLeadTrack.update), so for a MOVING lead this difference is
  # still the boosted -vRel (lag comp / urgency blend overshoot) — the bound is
  # physics-true only through the zero clamp on v_lead: exact for stopped /
  # near-stopped leads (the calm-stop collapse case, where the boost artifact
  # publishes vLead ~ -1.2 m/s and the clamp discards it), and conservative
  # (over-braking side) for moving leads. The boost still fully drives the
  # danger DEMAND above; only this BOUND discards it. Known open case: a
  # creeping lead (~0.8 m/s) approached at 8 m/s noise-off still saturates the
  # ceiling to max decel — pre-existing behavior (legacy emulation identical),
  # NOT covered by this bound.
  # Oncoming/reversing bypass: the max(0, vLead) clamp would credit a genuinely
  # oncoming lead as merely stationary and trim a ceiling the true closure rate
  # (v_ego + |vLead|) fully justifies. Below the bypass threshold (default
  # -2.5 m/s, clearly beyond the ~-1.2 m/s near-stop boost artifact) the bound
  # is skipped entirely and the danger term keeps full legacy authority.
  kinematic_closing = max(0.0, v_ego - v_lead)
  match_avail_gap = d_rel - kinematic_margin
  required_kinematic_decel = None
  if v_lead_raw >= float(tuning.lead_slowdown_kinematic_oncoming_vlead_mps) and match_avail_gap > 0.05:
    required_kinematic_decel = (kinematic_closing ** 2) / (2.0 * match_avail_gap)
    if lead_decel > 0.05:
      lead_stop_dist = (v_lead ** 2) / (2.0 * lead_decel)
      stop_avail_gap = d_rel + lead_stop_dist - kinematic_margin
      if stop_avail_gap > 0.05:
        required_kinematic_decel = max(required_kinematic_decel, (v_ego ** 2) / (2.0 * stop_avail_gap))
      else:
        required_kinematic_decel = None
  danger_decel_cap = None
  if required_kinematic_decel is not None:
    danger_decel_cap = max(kinematic_headroom * required_kinematic_decel,
                           lead_decel * LEAD_SLOWDOWN_LEAD_DECEL_OVERSHOOT)
  danger_deficit_gate = float(np.interp(projected_danger_deficit, LEAD_SLOWDOWN_DANGER_DEFICIT_BP, LEAD_SLOWDOWN_DANGER_DEFICIT_V))
  ttc_danger_gate = float(np.interp(ttc_danger, LEAD_SLOWDOWN_TTC_DANGER_BP, LEAD_SLOWDOWN_TTC_DANGER_V))
  ttc_collision_gate = float(np.interp(ttc_collision, LEAD_SLOWDOWN_TTC_COLLISION_BP, LEAD_SLOWDOWN_TTC_COLLISION_V))
  hard_brake_gate = float(np.interp(lead_decel, LEAD_SLOWDOWN_HARD_BRAKE_DECEL_BP, LEAD_SLOWDOWN_HARD_BRAKE_DECEL_V))
  danger_motion_gate = max(
    float(np.interp(closing_speed, LEAD_SLOWDOWN_DANGER_MOTION_CLOSING_BP, LEAD_SLOWDOWN_DANGER_MOTION_CLOSING_V)),
    float(np.interp(lead_decel, LEAD_SLOWDOWN_DANGER_MOTION_DECEL_BP, LEAD_SLOWDOWN_DANGER_MOTION_DECEL_V)),
  )
  danger_gate = max(
    ttc_collision_gate,
    max(danger_deficit_gate, ttc_danger_gate) * danger_motion_gate * hard_brake_gate,
  )
  danger_decel = danger_required_decel * danger_gate
  if danger_decel_cap is not None:
    danger_decel = min(danger_decel, danger_decel_cap)

  proximity_gate = max(gap_gate, headway_deficit_gate, ttc_headway_gate)
  onset = max(
    float(np.interp(closing_speed, LEAD_SLOWDOWN_ONSET_CLOSING_BP, LEAD_SLOWDOWN_ONSET_CLOSING_V)) * proximity_gate,
    float(np.interp(lead_decel, LEAD_SLOWDOWN_ONSET_DECEL_BP, LEAD_SLOWDOWN_ONSET_DECEL_V)) * max(0.20, lead_match_gate),
    headway_deficit_gate,
  )
  positive_accel_ceiling = float(np.interp(
    np.clip(onset, 0.0, 1.0),
    [0.0, 0.20, 1.0],
    [float(max_accel), LEAD_SLOWDOWN_SOFT_ACCEL_CAP, LEAD_SLOWDOWN_SOFT_ACCEL_CAP],
  ))

  strength = max(0.0, float(tuning.lead_slowdown_strength))
  anticipatory_strength = 0.0 if strength <= 0.0 else max(strength, LEAD_SLOWDOWN_ANTICIPATORY_STRENGTH_FLOOR)
  comfort_decel = min(
    LEAD_SLOWDOWN_COMFORT_DECEL_CAP,
    max(max(lead_match_decel, closing_decel) * strength, anticipatory_decel * anticipatory_strength),
  )
  decel_mag = min(max_decel, max(comfort_decel, danger_decel))
  if decel_mag < LEAD_SLOWDOWN_MIN_DECEL_OUTPUT:
    if positive_accel_ceiling >= float(max_accel) - 1e-3:
      return None
    return float(positive_accel_ceiling)

  return float(np.clip(min(positive_accel_ceiling, -decel_mag), -max_decel, float(max_accel)))


def compute_lead_stopping_need_decel(v_ego, lead,
                                     tuning: LeadResponseTuningConfig | None = None) -> float:
  """Decel magnitude (m/s^2, positive) kinematically required to stop
  STOP_DISTANCE short of the lead, for the cruise->lead handoff trigger.

  Mirrors the M1 kinematic-bound two-branch physics in
  get_lead_slowdown_accel_ceiling (max(0, vLead) clamp, closing-speed match
  branch, and the lead-stop-extended v_ego branch for a decelerating lead) so
  the trigger and the ceiling bound cannot disagree about what physics demands.

  Two deliberate divergences from the M1 bound:
  - Oncoming semantics are INVERTED: M1's oncoming bypass returns None to grant
    the ceiling full authority; a trigger must FIRE for oncoming leads instead,
    so below the oncoming vLead threshold the need is computed with the full
    closure rate v_ego + |vLead|.
  - The margin is STOP_DISTANCE (the MPC's own stop offset), not the tunable
    kinematic margin: the trigger asks "does stopping where the MPC would stop
    already require this much decel", and a larger margin only fires earlier
    (the safe direction).

  Returns inf when ego is already inside the margin (or the decelerating lead's
  projected stop point is): the handoff leg must fire unconditionally there.
  In practice those frames are already owned by raw_gap_hold, which is what
  keeps the 1e9 rollback value behaviorally identical to the legacy handoff.
  """
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  if lead is None or not getattr(lead, 'status', False):
    return 0.0

  v_ego = float(v_ego)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  v_lead_raw = _lead_float(lead, 'vLead', v_ego)
  oncoming = v_lead_raw < float(tuning.lead_slowdown_kinematic_oncoming_vlead_mps)
  v_lead = max(0.0, v_lead_raw)
  # Same clamp as the M1 bound for normal leads; full closure for oncoming.
  closing = max(0.0, v_ego - (v_lead_raw if oncoming else v_lead))

  match_avail_gap = d_rel - STOP_DISTANCE
  if match_avail_gap <= 0.05:
    return float('inf')
  required = (closing ** 2) / (2.0 * match_avail_gap)

  lead_decel = max(0.0, -float(getattr(lead, 'aLeadK', 0.0) or 0.0))
  if lead_decel > 0.05:
    lead_stop_dist = (v_lead ** 2) / (2.0 * lead_decel)
    stop_avail_gap = d_rel + lead_stop_dist - STOP_DISTANCE
    if stop_avail_gap > 0.05:
      required = max(required, (v_ego ** 2) / (2.0 * stop_avail_gap))
    else:
      required = float('inf')
  return float(required)


def get_low_speed_launch_follow_factor(v_ego, lead, t_follow) -> float:
  if lead is None or not getattr(lead, 'status', False):
    return 0.0

  v_ego = float(v_ego)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
  v_rel = float(getattr(lead, 'vRel', 0.0) or 0.0)
  pullaway_speed = max(0.0, v_lead - v_ego, v_rel)
  if pullaway_speed <= 0.0:
    return 0.0

  gap_surplus = max(0.0, d_rel - get_headway_follow_distance(v_ego, t_follow))
  speed_term = float(np.interp(v_ego, LOW_SPEED_LAUNCH_FACTOR_V_EGO_BP, LOW_SPEED_LAUNCH_FACTOR_V_EGO_V))
  lead_speed_term = float(np.interp(v_lead, LOW_SPEED_LAUNCH_FACTOR_V_LEAD_BP, LOW_SPEED_LAUNCH_FACTOR_V_LEAD_V))
  pullaway_term = float(np.interp(pullaway_speed, LOW_SPEED_LAUNCH_FACTOR_PULLAWAY_BP, LOW_SPEED_LAUNCH_FACTOR_PULLAWAY_V))
  gap_term = float(max(0.0, np.interp(gap_surplus, LOW_SPEED_LAUNCH_FACTOR_GAP_BP, LOW_SPEED_LAUNCH_FACTOR_GAP_V)))
  return float(np.clip(speed_term * lead_speed_term * max(pullaway_term, gap_term), 0.0, 1.0))


def get_low_speed_launch_follow_max_accel(v_ego, lead, t_follow, base_max_accel: float) -> float:
  factor = get_low_speed_launch_follow_factor(v_ego, lead, t_follow)
  return float(base_max_accel + (LOW_SPEED_LAUNCH_MAX_ACCEL - base_max_accel) * factor)


def get_lead_present_cruise_accel_cap(v_ego, lead, t_follow,
                                      tuning: LeadResponseTuningConfig | None = None,
                                      personality_max_accel: float | None = None) -> float | None:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  if lead is None or not getattr(lead, 'status', False) or float(v_ego) < LEAD_PRESENT_CRUISE_MIN_SPEED:
    return None

  comfort_cap = max(float(tuning.gap_reclaim_max_accel), 1e-3)
  personality_cap = ACCEL_MAX if personality_max_accel is None else max(comfort_cap, float(personality_max_accel))

  d_rel_raw = getattr(lead, 'dRel', 0.0)
  v_lead_raw = getattr(lead, 'vLead', None)
  v_rel_raw = getattr(lead, 'vRel', 0.0)
  d_rel = float(0.0 if d_rel_raw is None else d_rel_raw)
  v_lead = float(v_ego if v_lead_raw is None else v_lead_raw)
  v_rel = float(0.0 if v_rel_raw is None else v_rel_raw)
  closing_speed = max(0.0, float(v_ego) - v_lead)
  gap_surplus_raw = max(0.0, d_rel - get_headway_follow_distance(float(v_ego), t_follow))
  # Project gap surplus forward: if closing, the gap is shrinking
  closing_reduction = closing_speed * LEAD_PRESENT_CRUISE_CLOSING_PROJECTION_S
  gap_surplus = max(0.0, gap_surplus_raw - closing_reduction)
  pullaway_speed = max(0.0, v_lead - float(v_ego), v_rel)

  speed_cap = float(np.interp(float(v_ego), LEAD_PRESENT_CRUISE_SPEED_CAP_BP, LEAD_PRESENT_CRUISE_SPEED_CAP_V))
  accel_cap = min(personality_cap, speed_cap)
  if accel_cap <= comfort_cap:
    return comfort_cap

  gap_blend = float(np.interp(gap_surplus, LEAD_PRESENT_CRUISE_SURPLUS_BP, LEAD_PRESENT_CRUISE_SURPLUS_V))
  pullaway_blend = float(np.interp(pullaway_speed, LEAD_PRESENT_CRUISE_PULLAWAY_BP, LEAD_PRESENT_CRUISE_PULLAWAY_V))
  closing_tighten = float(np.interp(closing_speed, LEAD_PRESENT_CRUISE_CLOSING_TIGHTEN_BP, LEAD_PRESENT_CRUISE_CLOSING_TIGHTEN_V))
  blend = max(gap_blend, pullaway_blend) * closing_tighten
  cap = float(comfort_cap + (accel_cap - comfort_cap) * blend)
  if closing_speed >= LEAD_PRESENT_CRUISE_COAST_CLOSING_MPS and pullaway_speed <= 0.0:
    cap = min(cap, LEAD_PRESENT_CRUISE_COAST_ACCEL_CAP)
  return cap


def should_start_cutin_settle_event(prev_role: str, prev_control_active: bool, current_role: str,
                                    lead, *, cutin_promoted: bool, toward_center_mps: float,
                                    path_abs_m: float, v_ego: float) -> bool:
  if lead is None or not getattr(lead, 'status', False) or float(v_ego) < CUTIN_SETTLE_MIN_SPEED:
    return False

  d_rel = float(getattr(lead, 'dRel', 1e9) or 1e9)
  if d_rel > CUTIN_SETTLE_DETECT_DREL_MAX:
    return False

  became_center_from_adjacent = (
    current_role == LeadRoleClassifier.CENTER_CONTROL and
    prev_role in (LeadRoleClassifier.ADJ_LEFT, LeadRoleClassifier.ADJ_RIGHT)
  )
  new_control = bool(getattr(lead, 'status', False)) and (not prev_control_active)
  lateral_hint = (
    float(toward_center_mps) >= CUTIN_SETTLE_DETECT_TOWARD_CENTER_MIN_MPS or
    float(path_abs_m) >= CUTIN_SETTLE_DETECT_PATH_ABS_MIN
  )
  return bool(cutin_promoted or became_center_from_adjacent or (new_control and lateral_hint))


def get_cutin_settle_accel_floor(v_ego, lead, t_follow, age_s,
                                 tuning: LeadResponseTuningConfig | None = None) -> float | None:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  if (lead is None or not getattr(lead, 'status', False) or
      float(v_ego) < CUTIN_SETTLE_MIN_SPEED or
      tuning.cutin_settle_duration_s <= 0.0):
    return None

  age_s = float(age_s)
  if age_s < 0.0 or age_s > tuning.cutin_settle_duration_s:
    return None

  v_lead = max(0.0, float(getattr(lead, 'vLead', v_ego) or v_ego))
  closing_speed = max(0.0, float(v_ego) - v_lead)
  if closing_speed > tuning.cutin_settle_max_closing_speed_mps:
    return None

  lead_accel = float(getattr(lead, 'aLeadK', 0.0) or 0.0)
  if lead_accel < CUTIN_SETTLE_LEAD_ACCEL_MIN:
    return None

  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  danger_distance = LEAD_DANGER_FACTOR * desired_follow_distance(float(v_ego), v_lead, t_follow)
  if d_rel <= (danger_distance + CUTIN_SETTLE_DANGER_MARGIN_M):
    return None

  progress = float(np.clip(age_s / tuning.cutin_settle_duration_s, 0.0, 1.0))
  progress_scale = float(np.interp(progress, CUTIN_SETTLE_PROGRESS_BP, CUTIN_SETTLE_PROGRESS_V))
  closing_scale = float(np.clip(closing_speed / tuning.cutin_settle_max_closing_speed_mps, 0.0, 1.0))
  floor_mag = tuning.cutin_settle_max_decel * progress_scale * closing_scale
  bias = float(getattr(tuning, 'cutin_settle_accel_bias_mps2', 0.0) or 0.0)
  return bias - float(np.clip(floor_mag, 0.0, tuning.cutin_settle_max_decel))


def gen_long_model():
  from casadi import SX, vertcat
  from openpilot.third_party.acados.acados_template import AcadosModel

  model = AcadosModel()
  model.name = MODEL_NAME

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  prev_a = SX.sym('prev_a')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model.p = vertcat(a_min, a_max, x_obstacle, prev_a, lead_t_follow, lead_danger_factor)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_ocp():
  from casadi import vertcat
  from openpilot.third_party.acados.acados_template import AcadosOcp

  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  prev_a = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           v_ego,
           a_ego,
           a_ego - prev_a,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([CRUISE_MIN_ACCEL, CRUISE_MAX_ACCEL, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR])


  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  LIVE_TUNE_REFRESH_DT_S = 0.50
  LEAD_STABILIZER_PHANTOM_YREL_KILL_M = 1.75

  def __init__(self, mode='acc', dt=DT_MDL, CP=None):
    self.mode = mode
    self.dt = dt
    self._time_fn = time.monotonic
    self._hyundai_ai_lead_stability_enabled = bool(getattr(CP, 'brand', None) == 'hyundai')
    self.use_upstream_gap_reclaim = self._hyundai_ai_lead_stability_enabled
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self._live_tune_params = Params()
    self._last_live_tune_refresh_t = 0.0
    self._live_tune_cfg = LeadResponseTuningConfig.defaults()
    self._live_obstacle_cost = float(X_EGO_OBSTACLE_COST)
    self._live_a_change_cost = float(A_CHANGE_COST)
    self._live_a_ego_cost = float(A_EGO_COST)
    self.reset()
    self.source = SOURCES[2]
    self.vibe_controller = VibePersonalityController()
    self.lead_role_classifier = LeadRoleClassifier()
    self.lead_role_debug = {}
    self.last_lead_role_log_t = 0.0
    self._lead_stability_state = [_LeadStabilityState(), _LeadStabilityState()]
    self._lead_stability_phantom_slots: tuple[bool, bool] = (False, False)
    self.lead_stability_debug: dict[str, Any] = {}

  def reset(self):
    # self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.solver.reset()
    # self.solver.options_set('print_level', 2)
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.prev_a = np.array(self.a_solution)
    self.j_solution = np.zeros(N)
    self.yref = np.zeros((N+1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N,1))
    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.solution_status = 0
    self.last_cruise_response_model = None
    self.last_v_lower = None
    self.last_v_upper = None
    self.last_v_cruise_clipped = None
    self.current_t_follow = float(get_T_FOLLOW())
    self.control_leads = (None, None)
    self.lead_approach_preview = (0.0, 0.0)
    self.lead_approach_preview_debug = {
      "lead0": {"active": False, "mode": "inactive", "acquire_window_active": False, "preview_buffer_m": 0.0},
      "lead1": {"active": False, "mode": "inactive", "acquire_window_active": False, "preview_buffer_m": 0.0},
    }
    self.adjacent_awareness_preview_debug = {"active": False}
    self.lead_handoff_danger_factor = float(LEAD_DANGER_FACTOR)
    self.lead_handoff_danger_debug = {"active": False, "danger_factor": float(LEAD_DANGER_FACTOR)}
    self.gap_reclaim_accel_floor = 0.0
    self.lead_keepup_accel_floor = 0.0
    self.lead_slowdown_accel_ceiling = None
    self._lead_slowdown_accel_ceiling_last = None
    self._lead_slowdown_accel_ceiling_last_t = None
    self.gap_reclaim_obstacle_push = 0.0
    self.gap_reclaim_stabilization_push = 0.0
    self.gap_reclaim_projection_scale = 1.0
    self.gap_reclaim_personality_max_accel = 0.0
    self.gap_reclaim_effective_cap = 0.0
    self.lead_present_cruise_accel_cap = 0.0
    self.cruise_owned_accel_cap = None
    self._gap_reclaim_blend = 0.0
    self._gap_reclaim_last_t = None
    self._raw_reclaim_safety_override_active = False
    self.cutin_settle_active = False
    self.cutin_settle_accel_floor = 0.0
    self.cutin_settle_debug = {}
    self._virtual_cutin_event_t = None
    self._prev_virtual_lead_role = LeadRoleClassifier.INVALID
    self._prev_virtual_lead_control_active = False
    self._hyundai_duplicate_selected_raw_slot = None
    self._hyundai_virtual_lead = None
    self._hyundai_virtual_lead_source = None
    self._hyundai_virtual_lead_last_t = None
    self._hyundai_virtual_lead_identity_changed = False
    self._hyundai_virtual_lead_reset_reason = None
    self._hyundai_virtual_lead_stable_since_t = None
    self._hyundai_virtual_lead_last_drel_error_m = 1e9
    self._hyundai_virtual_lead_dropout_until_t = None
    self._drel_filter = LeadDistanceFilter()
    self._drel_kalman = LeadKalmanFilter()
    self._use_kalman_drel = False
    self._use_kalman_drel_prev = False
    self.hyundai_virtual_lead_debug = {"active": False}
    self._hyundai_reclaim_lead = None
    self._hyundai_reclaim_last_t = None
    self._acc_obstacle_mode = 'cruise'
    self._acc_obstacle_candidate_mode = None
    self._acc_obstacle_candidate_t = None
    self._classifier_demotion_hold_until_t: float | None = None
    self._last_raw_radar_leads: tuple = (None, None)
    self._lead_to_cruise_transition_t = None
    self._lead_to_cruise_transition_source = None
    # Close-range lead memory: safety cap on cruise accel when a lead was
    # recently visible nearby, even if the model is currently flickering.
    self._close_lead_last_seen_t: float | None = None
    self._close_lead_last_drel: float = 1e9
    self._lead_handoff_until_t = None
    self._lead_handoff_from_source = None
    self._lead_handoff_to_source = None
    self._lead_acquire_until_t = {'lead0': None, 'lead1': None}
    self._lead_acquire_last_reason = {'lead0': None, 'lead1': None}
    self._lead_prev_obs = {
      'lead0': {"status": False, "dRel": None, "vLead": None},
      'lead1': {"status": False, "dRel": None, "vLead": None},
    }
    self.acc_source_debug = {}
    # timers
    self.solve_time = 0.0
    self.time_qp_solution = 0.0
    self.time_linearization = 0.0
    self.time_integrator = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def _refresh_live_tune(self, now: float, force: bool = False) -> None:
    if not force and (now - self._last_live_tune_refresh_t) < self.LIVE_TUNE_REFRESH_DT_S:
      return
    self._last_live_tune_refresh_t = now
    self._live_tune_cfg = read_lead_response_tuning_config(self._live_tune_params)
    try:
      raw = self._live_tune_params.get("Longitudinal.LiveTune.ObstacleCost")
      self._live_obstacle_cost = float(raw) if raw is not None else float(X_EGO_OBSTACLE_COST)
    except Exception:
      self._live_obstacle_cost = float(X_EGO_OBSTACLE_COST)
    try:
      raw = self._live_tune_params.get("Longitudinal.LiveTune.AccelChangeCost")
      self._live_a_change_cost = float(raw) if raw is not None else float(A_CHANGE_COST)
    except Exception:
      self._live_a_change_cost = float(A_CHANGE_COST)
    try:
      raw = self._live_tune_params.get("Longitudinal.LiveTune.AccelCost")
      self._live_a_ego_cost = float(raw) if raw is not None else float(A_EGO_COST)
    except Exception:
      self._live_a_ego_cost = float(A_EGO_COST)
    # Kalman dRel filter toggle and tuning
    try:
      self._use_kalman_drel = bool(self._live_tune_params.get_bool("Longitudinal.LiveTune.UseKalmanDRelFilter"))
    except Exception:
      self._use_kalman_drel = False
    # On toggle: seed the newly-active filter from the outgoing filter's state
    if self._use_kalman_drel != self._use_kalman_drel_prev:
      current_drel = (self._drel_filter.value if self._use_kalman_drel_prev is False else self._drel_kalman.value)
      if self._use_kalman_drel:
        self._drel_kalman.reset(current_drel)
      else:
        self._drel_filter.reset(current_drel)
      self._use_kalman_drel_prev = self._use_kalman_drel
    try:
      q = self._live_tune_params.get("Longitudinal.LiveTune.KalmanDRelQ")
      r = self._live_tune_params.get("Longitudinal.LiveTune.KalmanDRelR")
      km = self._live_tune_params.get("Longitudinal.LiveTune.KalmanDRelGainMax")
      db = self._live_tune_params.get("Longitudinal.LiveTune.KalmanDRelDeadbandM")
      self._drel_kalman.set_tuning(
        q_drel=float(q) if q is not None else None,
        r_drel=float(r) if r is not None else None,
      )
      if km is not None:
        self._drel_kalman._k_max = float(km)
      if db is not None:
        self._drel_kalman._deadband_m = float(db)
    except Exception:
      pass

  def get_live_tune_config(self) -> LeadResponseTuningConfig:
    return self._live_tune_cfg

  @staticmethod
  def _lead_attr(lead, attr: str, default: float = 0.0) -> float:
    return float(getattr(lead, attr, default) or default)

  @staticmethod
  def _lead_debug_payload(lead) -> dict[str, float | bool]:
    if lead is None:
      return {
        "status": False,
        "dRel": 0.0,
        "yRel": 0.0,
        "dPath": 0.0,
        "vLat": 0.0,
        "vRel": 0.0,
        "modelProb": 0.0,
      }
    return {
      "status": bool(getattr(lead, "status", False)),
      "dRel": LongitudinalMpc._lead_attr(lead, "dRel"),
      "yRel": LongitudinalMpc._lead_attr(lead, "yRel"),
      "dPath": LongitudinalMpc._lead_attr(lead, "dPath", LongitudinalMpc._lead_attr(lead, "yRel")),
      "vLat": LongitudinalMpc._lead_attr(lead, "vLat"),
      "vRel": LongitudinalMpc._lead_attr(lead, "vRel"),
      "modelProb": LongitudinalMpc._lead_attr(lead, "modelProb"),
    }

  def _update_lead_acquire_state(self, now: float) -> dict[str, dict[str, float | bool | str | None]]:
    debug: dict[str, dict[str, float | bool | str | None]] = {}
    acquire_window_s = max(0.0, float(getattr(self._live_tune_cfg, 'lead_acquire_window_s', 0.0)))
    v_ego = float(self.x0[1])

    for slot_idx, lead in enumerate(self.control_leads):
      slot_key = f"lead{slot_idx}"
      prev = self._lead_prev_obs[slot_key]
      status = bool(lead is not None and getattr(lead, 'status', False))
      activated = False
      activation_reason = self._lead_acquire_last_reason[slot_key]

      if status:
        d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
        v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
        closing_speed = max(0.0, v_ego - v_lead)
        reason = None
        if not bool(prev["status"]):
          reason = "new_lead"
        else:
          prev_drel = prev["dRel"]
          prev_vlead = prev["vLead"]
          if prev_drel is not None and (float(prev_drel) - d_rel) >= max(4.0, closing_speed * 0.45):
            reason = "closer_jump"
          elif prev_vlead is not None and (float(prev_vlead) - v_lead) >= 1.0 and closing_speed > LEAD_APPROACH_PREVIEW_CLOSING_MIN_MPS:
            reason = "slower_jump"

        if reason is not None and acquire_window_s > 0.0:
          self._lead_acquire_until_t[slot_key] = now + acquire_window_s
          self._lead_acquire_last_reason[slot_key] = reason
          activation_reason = reason
          activated = True

        remaining_s = 0.0
        if self._lead_acquire_until_t[slot_key] is not None:
          remaining_s = max(0.0, float(self._lead_acquire_until_t[slot_key]) - now)
        active = remaining_s > 0.0
        prev.update({"status": True, "dRel": d_rel, "vLead": v_lead})
      else:
        self._lead_acquire_until_t[slot_key] = None
        self._lead_acquire_last_reason[slot_key] = None
        activation_reason = None
        active = False
        remaining_s = 0.0
        prev.update({"status": False, "dRel": None, "vLead": None})

      debug[slot_key] = {
        "active": bool(active),
        "activated": bool(activated),
        "remaining_s": float(remaining_s),
        "reason": activation_reason,
      }

    return debug

  @staticmethod
  def _empty_lead_debug_payload() -> dict[str, float | bool]:
    return {
      "status": False,
      "dRel": 0.0,
      "yRel": 0.0,
      "dPath": 0.0,
      "vLat": 0.0,
      "vRel": 0.0,
      "modelProb": 0.0,
    }

  def _compute_adjacent_awareness_preview_obstacle(self, raw_leads: dict[str, object],
                                                   lead_role_debug: dict[str, object],
                                                   v_ego: float,
                                                   t_follow: float) -> tuple[np.ndarray | None, dict[str, float | bool | str | None]]:
    debug: dict[str, float | bool | str | None] = {"active": False}
    awareness_entries = lead_role_debug.get("awareness", [])
    if not awareness_entries:
      return None, debug

    center_exit_m = (
      float(self.lead_role_classifier._cfg.get("center_y_abs_max_m", 2.2)) +
      float(self.lead_role_classifier._cfg.get("center_hyst_m", 0.35))
    )
    center_enter_m = float(self.lead_role_classifier._cfg.get("center_y_abs_min_m", 1.2))
    best_obstacle = None

    for entry in awareness_entries:
      slot_idx = int(entry.get("slot", -1))
      slot_key = f"lead{slot_idx}"
      if slot_key not in raw_leads:
        continue

      raw_lead = raw_leads[slot_key]
      if raw_lead is None or not getattr(raw_lead, 'status', False):
        continue

      role = str(entry.get("role", LeadRoleClassifier.INVALID))
      if role not in (LeadRoleClassifier.ADJ_LEFT, LeadRoleClassifier.ADJ_RIGHT):
        continue

      path_abs_m = abs(float(entry.get("dPath", self._lead_attr(raw_lead, "dPath", self._lead_attr(raw_lead, "yRel"))) or 0.0))
      toward_center_gate_mps = CUTIN_SETTLE_DETECT_TOWARD_CENTER_MIN_MPS
      toward_center_mps = float(lead_role_debug.get("toward_center_mps", {}).get(slot_key, 0.0) or 0.0)
      model_prob = float(np.clip(getattr(raw_lead, 'modelProb', 0.0) or 0.0, 0.0, 1.0))
      if path_abs_m > center_exit_m or toward_center_mps < toward_center_gate_mps or model_prob <= 0.0:
        continue

      preview_buffer_raw, preview_debug = compute_lead_approach_preview(
        v_ego,
        raw_lead,
        t_follow,
        self._live_tune_cfg,
        acquire_window_active=True,
      )
      if preview_buffer_raw <= 0.0:
        continue

      preview_buffer_m = float(preview_buffer_raw) * model_prob
      if preview_buffer_m <= 0.0:
        continue

      lead_xv = self.process_lead(raw_lead)
      handoff_remaining_s = max(0.0, path_abs_m - center_enter_m) / max(toward_center_mps, 1e-3)
      projected_deficit_m = float(preview_debug.get("projected_deficit_m", 0.0) or 0.0)
      path_scale = float(np.interp(path_abs_m, [center_enter_m, center_exit_m], [1.0, 0.55]))
      handoff_scale = float(np.interp(handoff_remaining_s, [0.0, 1.5], [1.0, 0.65]))
      application_blend = float(np.clip(model_prob * path_scale * handoff_scale, 0.0, 1.0))
      obstacle = apply_lead_approach_preview(
        lead_xv[:, 0] + get_stopped_equivalence_factor(lead_xv[:, 1]),
        preview_buffer_m,
      )
      if best_obstacle is None or float(obstacle[0]) < float(best_obstacle[0]):
        debug = {
          "active": True,
          "slot": slot_key,
          "role": role,
          "path_abs_m": float(path_abs_m),
          "toward_center_mps": float(toward_center_mps),
          "toward_center_gate_mps": float(toward_center_gate_mps),
          "model_prob": float(model_prob),
          "confidence_scale": float(model_prob),
          "path_scale": float(path_scale),
          "handoff_scale": float(handoff_scale),
          "application_blend": float(application_blend),
          "preview_buffer_raw_m": float(preview_buffer_raw),
          "preview_buffer_m": float(preview_buffer_m),
          "preview_obstacle_m": float(obstacle[0]),
          "handoff_remaining_s": float(handoff_remaining_s),
          "mode": str(preview_debug.get("mode", "inactive")),
          "closing_speed_mps": float(preview_debug.get("closing_speed_mps", 0.0) or 0.0),
          "gap_surplus_m": float(preview_debug.get("gap_surplus_m", 0.0) or 0.0),
          "projected_deficit_m": float(projected_deficit_m),
        }
        best_obstacle = obstacle

    return best_obstacle, debug

  @staticmethod
  def _lead_follow_metrics(v_ego: float, t_follow: float, lead, obstacle_0: float | None = None) -> dict[str, float]:
    if lead is None or not getattr(lead, 'status', False):
      return {
        "gap_surplus": 1e9,
        "pullaway_speed": 0.0,
        "closing_speed": 0.0,
        "obstacle_0": float(obstacle_0 if obstacle_0 is not None else 1e9),
        "ttc_to_headway_s": 1e6,
      }

    gap_surplus = float(getattr(lead, 'dRel', 0.0) or 0.0) - get_headway_follow_distance(v_ego, t_follow)
    v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
    v_rel = float(getattr(lead, 'vRel', 0.0) or 0.0)
    closing_speed = float(max(0.0, v_ego - v_lead, -v_rel))
    return {
      "gap_surplus": float(gap_surplus),
      "pullaway_speed": float(max(0.0, v_lead - v_ego, v_rel)),
      "closing_speed": closing_speed,
      "obstacle_0": float(obstacle_0 if obstacle_0 is not None else 1e9),
      "ttc_to_headway_s": LongitudinalMpc._time_to_headway(gap_surplus, closing_speed),
    }

  @staticmethod
  def _time_to_headway(gap_surplus_m: float, closing_speed_mps: float) -> float:
    if gap_surplus_m <= 0.0:
      return 0.0
    if closing_speed_mps <= 1e-3:
      return 1e6
    return float(gap_surplus_m / closing_speed_mps)

  @staticmethod
  def _approach_reacquire_ttc_threshold(v_ego: float) -> float:
    return float(np.interp(
      float(v_ego),
      [HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_LOW_SPEED_MPS,
       HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_HIGH_SPEED_MPS],
      [HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_LOW_SPEED_TTC_HEADWAY_S,
       HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_TTC_HEADWAY_S],
    ))

  @staticmethod
  def _is_plausible_cruise_cap_raw_lead(lead) -> bool:
    if lead is None or not getattr(lead, 'status', False):
      return False
    d_rel = LongitudinalMpc._lead_attr(lead, "dRel", 1e9)
    if d_rel <= 0.0 or d_rel > HYUNDAI_CRUISE_CAP_RAW_LEAD_MAX_DREL_M:
      return False
    d_path = LongitudinalMpc._lead_attr(lead, "dPath", LongitudinalMpc._lead_attr(lead, "yRel"))
    return abs(d_path) <= HYUNDAI_CRUISE_CAP_RAW_LEAD_MAX_PATH_ABS_M

  @staticmethod
  def _is_hyundai_settled_follow(raw_lead, filtered_lead,
                                 raw_metrics: dict[str, float],
                                 filtered_metrics: dict[str, float]) -> bool:
    if raw_lead is None or filtered_lead is None:
      return False
    if not getattr(raw_lead, 'status', False) or not getattr(filtered_lead, 'status', False):
      return False

    abs_vrel = max(
      abs(float(getattr(raw_lead, 'vRel', 0.0) or 0.0)),
      abs(float(getattr(filtered_lead, 'vRel', 0.0) or 0.0)),
    )
    abs_alead = max(
      abs(float(getattr(raw_lead, 'aLeadK', 0.0) or 0.0)),
      abs(float(getattr(filtered_lead, 'aLeadK', 0.0) or 0.0)),
    )
    gap_surplus = min(float(raw_metrics["gap_surplus"]), float(filtered_metrics["gap_surplus"]))
    return bool(
      gap_surplus <= HYUNDAI_SETTLED_FOLLOW_MAX_GAP_SURPLUS_M and
      abs_vrel <= HYUNDAI_SETTLED_FOLLOW_MAX_ABS_VREL_MPS and
      abs_alead <= HYUNDAI_SETTLED_FOLLOW_MAX_ABS_ALEAD_MPS2
    )

  @staticmethod
  def _ema_alpha(dt_s: float, tau_s: float) -> float:
    if dt_s <= 0.0 or tau_s <= 0.0:
      return 1.0
    return float(np.clip(1.0 - np.exp(-dt_s / max(tau_s, 1e-3)), 0.0, 1.0))

  def _filter_metric(self, prev: float, current: float, dt_s: float, *,
                     danger_if_lower: bool, fast_tau_s: float = HYUNDAI_VIRTUAL_LEAD_FAST_TAU_S,
                     slow_tau_s: float = HYUNDAI_VIRTUAL_LEAD_SLOW_TAU_S,
                     sign_transition_tau_s: float | None = None) -> float:
    use_fast = current <= prev if danger_if_lower else current >= prev
    if use_fast:
      tau_s = fast_tau_s
    elif sign_transition_tau_s is not None and prev * current < 0.0:
      # genuine positive↔negative crossing (neither side is zero)
      tau_s = sign_transition_tau_s
    else:
      tau_s = slow_tau_s
    alpha = self._ema_alpha(dt_s, tau_s)
    return float(prev + alpha * (current - prev))

  def _filter_symmetric_metric(self, prev: float, current: float, dt_s: float, tau_s: float) -> float:
    alpha = self._ema_alpha(dt_s, tau_s)
    return float(prev + alpha * (current - prev))

  def _clear_hyundai_virtual_lead_dropout_hold(self) -> None:
    self._hyundai_virtual_lead_dropout_until_t = None

  def _get_hyundai_virtual_lead_dropout_debug(self, now: float) -> dict[str, float | bool]:
    stable_age_s = 0.0 if self._hyundai_virtual_lead_stable_since_t is None else max(0.0, now - float(self._hyundai_virtual_lead_stable_since_t))
    hold_remaining_s = 0.0 if self._hyundai_virtual_lead_dropout_until_t is None else max(0.0, float(self._hyundai_virtual_lead_dropout_until_t) - now)
    lead = self._hyundai_virtual_lead
    metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, lead)
    abs_vrel = abs(float(getattr(lead, 'vRel', 0.0) or 0.0)) if lead is not None else 1e9
    path_abs = abs(float(getattr(lead, 'dPath', getattr(lead, 'yRel', 0.0)) or 0.0)) if lead is not None else 1e9
    drel_consistency = float(self._hyundai_virtual_lead_last_drel_error_m)
    eligible = bool(
      lead is not None and
      getattr(lead, 'status', False) and
      self._hyundai_virtual_lead_source in ('lead0', 'lead1') and
      stable_age_s >= HYUNDAI_VIRTUAL_LEAD_DROPOUT_STABLE_MIN_S and
      abs_vrel <= HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_ABS_VREL_MPS and
      drel_consistency <= HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_DREL_ERR_M and
      metrics["gap_surplus"] <= HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_GAP_SURPLUS_M and
      path_abs <= HYUNDAI_VIRTUAL_LEAD_DROPOUT_MAX_PATH_ABS_M
    )
    return {
      "active": bool(hold_remaining_s > 0.0),
      "eligible": eligible,
      "remaining_s": float(hold_remaining_s),
      "stable_age_s": float(stable_age_s),
      "abs_vrel_mps": float(abs_vrel),
      "gap_surplus_m": float(metrics["gap_surplus"]),
      "drel_consistency_m": float(drel_consistency),
      "path_abs_m": float(path_abs),
    }

  def _maybe_hold_hyundai_virtual_lead_dropout(self, now: float) -> tuple[ControlLead | None, dict[str, float | bool]]:
    debug = self._get_hyundai_virtual_lead_dropout_debug(now)
    activated = False
    if self._hyundai_virtual_lead_dropout_until_t is None and bool(debug["eligible"]):
      self._hyundai_virtual_lead_dropout_until_t = now + HYUNDAI_VIRTUAL_LEAD_DROPOUT_HOLD_S
      debug = self._get_hyundai_virtual_lead_dropout_debug(now)
      activated = True
    debug["activated"] = bool(activated)
    if not bool(debug["active"]) or self._hyundai_virtual_lead is None:
      return None, debug
    held_lead = copy.deepcopy(self._hyundai_virtual_lead)
    held_lead.status = True
    return held_lead, debug

  def _has_fresh_raw_radar_corroboration(self, stored_lead) -> tuple[bool, dict[str, object]]:
    """Is there still a valid raw radar lead near the stored virtual lead?

    Used by _maybe_hold_hyundai_classifier_demotion to verify that the physical
    object is still present in the sensor frame before holding stored state.
    This is what separates "classifier briefly demoted a lead that is still
    there" from "lead genuinely disappeared".
    """
    debug: dict[str, object] = {"matched": False}
    if stored_lead is None:
      debug["reason"] = "no_stored_lead"
      return False, debug
    stored_d = float(getattr(stored_lead, 'dRel', 0.0) or 0.0)
    best_delta: float | None = None
    best_slot: int | None = None
    for idx, raw in enumerate(self._last_raw_radar_leads):
      if raw is None:
        continue
      try:
        if not bool(getattr(raw, 'status', False)):
          continue
        raw_d = float(getattr(raw, 'dRel', 0.0) or 0.0)
        raw_yrel = float(getattr(raw, 'yRel', 0.0) or 0.0)
        raw_vrel = float(getattr(raw, 'vRel', 0.0) or 0.0)
      except (TypeError, ValueError):
        continue
      if not (math.isfinite(raw_d) and math.isfinite(raw_yrel) and math.isfinite(raw_vrel)):
        continue
      delta = abs(raw_d - stored_d)
      if delta > HYUNDAI_CLASSIFIER_DEMOTION_HOLD_CORROB_DREL_M:
        continue
      if best_delta is None or delta < best_delta:
        best_delta = delta
        best_slot = idx
    if best_delta is None:
      debug["reason"] = "no_raw_match_within_tolerance"
      return False, debug
    debug["matched"] = True
    debug["matched_slot"] = int(best_slot) if best_slot is not None else None
    debug["dRel_delta_m"] = float(best_delta)
    return True, debug

  def _maybe_hold_hyundai_classifier_demotion(self, now: float) -> tuple[ControlLead | None, dict[str, object]]:
    """Short hold when the lead role classifier briefly demotes a steady-follow lead.

    This is distinct from _maybe_hold_hyundai_virtual_lead_dropout:
      - fires only when the previous cycle was actively following a lead
      - requires fresh raw radar corroboration (stored state alone is not enough)
      - uses a shorter max-hold duration than dropout hold
      - permits a wider gap-surplus envelope than dropout hold, since highway
        steady-state following naturally sits above the tight-gap threshold

    Safety: if no raw radar lead is still present near the stored position,
    this returns None and the caller releases to cruise immediately — no
    stale state is held through a genuine cut-out or lateral departure.
    """
    debug: dict[str, object] = {
      "active": False,
      "eligible": False,
      "remaining_s": 0.0,
    }
    if self._acc_obstacle_mode != 'lead':
      debug["reason"] = "not_following"
      self._classifier_demotion_hold_until_t = None
      return None, debug
    stored_lead = self._hyundai_virtual_lead
    if stored_lead is None or not bool(getattr(stored_lead, 'status', False)):
      debug["reason"] = "no_stored_lead"
      self._classifier_demotion_hold_until_t = None
      return None, debug

    source_key = str(self._hyundai_virtual_lead_source or "")
    debug["source"] = source_key
    role_reasons = self.lead_role_debug.get("reasons", {}) if isinstance(self.lead_role_debug, dict) else {}
    if str(role_reasons.get(source_key, "")) == "raw_lateral_departure":
      debug["reason"] = "raw_lateral_departure"
      self._classifier_demotion_hold_until_t = None
      return None, debug

    stable_age_s = (
      0.0 if self._hyundai_virtual_lead_stable_since_t is None
      else max(0.0, now - float(self._hyundai_virtual_lead_stable_since_t))
    )
    path_abs = abs(float(getattr(stored_lead, 'dPath', getattr(stored_lead, 'yRel', 0.0)) or 0.0))
    drel_consistency = float(self._hyundai_virtual_lead_last_drel_error_m)
    debug["stable_age_s"] = float(stable_age_s)
    debug["path_abs_m"] = float(path_abs)
    debug["drel_consistency_m"] = float(drel_consistency)

    preconditions_ok = (
      stable_age_s >= HYUNDAI_CLASSIFIER_DEMOTION_HOLD_MIN_STABLE_S and
      path_abs <= HYUNDAI_CLASSIFIER_DEMOTION_HOLD_MAX_PATH_ABS_M and
      drel_consistency <= HYUNDAI_CLASSIFIER_DEMOTION_HOLD_MAX_DREL_ERR_M
    )
    if not preconditions_ok:
      debug["reason"] = "preconditions_failed"
      self._classifier_demotion_hold_until_t = None
      return None, debug

    corroborated, corrob_debug = self._has_fresh_raw_radar_corroboration(stored_lead)
    debug["corroboration"] = corrob_debug
    if not corroborated:
      debug["reason"] = "no_raw_corroboration"
      self._classifier_demotion_hold_until_t = None
      return None, debug

    if self._classifier_demotion_hold_until_t is None:
      self._classifier_demotion_hold_until_t = now + HYUNDAI_CLASSIFIER_DEMOTION_HOLD_S

    remaining = float(self._classifier_demotion_hold_until_t) - now
    if remaining <= 0.0:
      debug["reason"] = "expired"
      self._classifier_demotion_hold_until_t = None
      return None, debug

    debug["eligible"] = True
    debug["active"] = True
    debug["remaining_s"] = float(remaining)
    held = copy.deepcopy(stored_lead)
    held.status = True
    return held, debug

  def _get_lead_to_cruise_transition_accel_cap(self, now: float, v_ego: float,
                                               personality_max_accel: float | None) -> float | None:
    if self._lead_to_cruise_transition_t is None or self.source != 'cruise':
      return None
    elapsed_s = max(0.0, now - float(self._lead_to_cruise_transition_t))
    if elapsed_s >= HYUNDAI_LEAD_TO_CRUISE_TRANSITION_RAMP_S:
      return None

    speed_cap = float(np.interp(float(v_ego), LEAD_PRESENT_CRUISE_SPEED_CAP_BP, LEAD_PRESENT_CRUISE_SPEED_CAP_V))
    if personality_max_accel is None:
      target_cap = speed_cap
    else:
      target_cap = min(speed_cap, max(0.0, float(personality_max_accel)))
    start_cap = min(
      target_cap,
      max(float(self._live_tune_cfg.gap_reclaim_max_accel), HYUNDAI_LEAD_TO_CRUISE_TRANSITION_MIN_ACCEL),
    )
    progress = float(np.clip(elapsed_s / HYUNDAI_LEAD_TO_CRUISE_TRANSITION_RAMP_S, 0.0, 1.0))
    return float(start_cap + (target_cap - start_cap) * progress)

  def _reset_lead_slowdown_ceiling_release_limit(self) -> None:
    self._lead_slowdown_accel_ceiling_last = None
    self._lead_slowdown_accel_ceiling_last_t = None

  def _limit_lead_slowdown_ceiling_release(self, ceiling: float | None,
                                           raw_metrics: dict[str, float],
                                           filtered_metrics: dict[str, float],
                                           now: float) -> float | None:
    min_gap_surplus = min(float(raw_metrics["gap_surplus"]), float(filtered_metrics["gap_surplus"]))
    max_closing_speed = max(float(raw_metrics["closing_speed"]), float(filtered_metrics["closing_speed"]))
    max_pullaway_speed = max(float(raw_metrics["pullaway_speed"]), float(filtered_metrics["pullaway_speed"]))
    hold_needed = (
      max_closing_speed >= LEAD_SLOWDOWN_CEILING_HOLD_CLOSING_MPS or
      min_gap_surplus <= LEAD_SLOWDOWN_CEILING_HOLD_GAP_SURPLUS_M
    )
    if max_pullaway_speed >= LEAD_SLOWDOWN_CEILING_HOLD_PULLAWAY_RELEASE_MPS and min_gap_surplus > 0.0:
      hold_needed = False
    if not hold_needed:
      self._lead_slowdown_accel_ceiling_last = ceiling
      self._lead_slowdown_accel_ceiling_last_t = now
      return ceiling

    prev_ceiling = self._lead_slowdown_accel_ceiling_last
    prev_t = self._lead_slowdown_accel_ceiling_last_t
    if prev_ceiling is None or prev_t is None:
      self._lead_slowdown_accel_ceiling_last = ceiling
      self._lead_slowdown_accel_ceiling_last_t = now
      return ceiling

    target = float(ACCEL_MAX if ceiling is None else ceiling)
    prev = float(prev_ceiling)
    if target > prev:
      dt = max(0.0, float(now) - float(prev_t))
      target = min(target, prev + LEAD_SLOWDOWN_CEILING_RELEASE_RATE_MPS3 * dt)
    result = None if target >= ACCEL_MAX - 1e-3 else float(target)
    self._lead_slowdown_accel_ceiling_last = result
    self._lead_slowdown_accel_ceiling_last_t = now
    return result

  def _reset_hyundai_virtual_lead(self, reason: str) -> None:
    self._hyundai_virtual_lead = None
    self._hyundai_virtual_lead_source = None
    self._hyundai_virtual_lead_last_t = None
    self._hyundai_virtual_lead_identity_changed = False
    self._hyundai_virtual_lead_reset_reason = reason
    self._hyundai_virtual_lead_stable_since_t = None
    self._hyundai_virtual_lead_last_drel_error_m = 1e9
    self._clear_hyundai_virtual_lead_dropout_hold()
    self._drel_filter.reset()
    self._drel_kalman.reset()
    self._hyundai_reclaim_lead = None
    self._hyundai_reclaim_last_t = None
    self.hyundai_virtual_lead_debug = {
      "active": False,
      "reset_reason": reason,
    }

  @staticmethod
  def _synthetic_model_track_id(lead) -> int | None:
    try:
      track_id = int(getattr(lead, 'radarTrackId', -1) or -1)
      radar = bool(getattr(lead, 'radar', False))
    except Exception:
      return None
    return track_id if (not radar and track_id <= -1001) else None

  def _should_reset_hyundai_virtual_lead(self, lead_source: str, lead) -> tuple[bool, str | None]:
    if lead is None or not getattr(lead, 'status', False):
      return True, "no_control_lead"
    if self._hyundai_virtual_lead is None or self._hyundai_virtual_lead_source is None:
      return True, "init"
    if lead_source != self._hyundai_virtual_lead_source:
      prev_model_track_id = self._synthetic_model_track_id(self._hyundai_virtual_lead)
      new_model_track_id = self._synthetic_model_track_id(lead)
      if prev_model_track_id is None or prev_model_track_id != new_model_track_id:
        return True, "source_switch"
    raw_drel = float(getattr(lead, 'dRel', 0.0) or 0.0)
    filtered_drel = float(self._hyundai_virtual_lead.dRel)
    drel_delta = raw_drel - filtered_drel
    raw_dpath = float(getattr(lead, 'dPath', getattr(lead, 'yRel', 0.0)) or 0.0)
    filtered_dpath = float(self._hyundai_virtual_lead.dPath)
    path_delta = abs(raw_dpath - filtered_dpath)
    if drel_delta < -HYUNDAI_VIRTUAL_LEAD_RESET_DREL_M:
      return True, "drel_jump_closer"
    if (
      drel_delta > HYUNDAI_VIRTUAL_LEAD_RESET_DREL_M and
      (
        float(getattr(lead, 'vRel', 0.0) or 0.0) > HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_PULLAWAY_MPS or
        path_delta > HYUNDAI_VIRTUAL_LEAD_RESET_DPATH_M * 0.7
      )
    ):
      return True, "drel_jump_opening"
    if path_delta > HYUNDAI_VIRTUAL_LEAD_RESET_DPATH_M:
      return True, "dpath_jump"
    return False, None

  def _build_lead_obstacle(self, lead) -> np.ndarray:
    lead_xv = self.process_lead(lead)
    return lead_xv[:,0] + get_stopped_equivalence_factor(lead_xv[:,1])

  def _update_gap_reclaim_blend(self, target_blend: float, now: float) -> float:
    target_blend = float(np.clip(target_blend, 0.0, 1.0))
    if self._gap_reclaim_last_t is None:
      self._gap_reclaim_last_t = now
      self._gap_reclaim_blend = target_blend
      return self._gap_reclaim_blend

    dt_s = float(np.clip(now - self._gap_reclaim_last_t, 0.0, 1.0))
    self._gap_reclaim_last_t = now
    tau_s = GAP_RECLAIM_BLEND_RISE_TAU_S if target_blend > self._gap_reclaim_blend else GAP_RECLAIM_BLEND_FALL_TAU_S
    alpha = self._ema_alpha(dt_s, tau_s)
    self._gap_reclaim_blend = float(self._gap_reclaim_blend + alpha * (target_blend - self._gap_reclaim_blend))
    return self._gap_reclaim_blend

  def _get_gap_reclaim_personality_max_accel(self, v_ego: float) -> float | None:
    max_accel = self.vibe_controller.get_max_accel(v_ego)
    if max_accel is not None:
      return float(max_accel)
    return float(np.interp(float(v_ego), GAP_RECLAIM_BASE_MAX_ACCEL_BP, GAP_RECLAIM_BASE_MAX_ACCEL_V))

  def _apply_hyundai_gap_reclaim(self, raw_lead_obstacle: np.ndarray, filtered_lead_obstacle: np.ndarray,
                                 raw_lead, filtered_lead, raw_metrics: dict[str, float],
                                 settled_follow: bool, now: float,
                                 anticipatory_slowdown_enabled: bool = False) -> tuple[np.ndarray, bool]:
    self.gap_reclaim_accel_floor = 0.0
    self.lead_keepup_accel_floor = 0.0
    self.gap_reclaim_obstacle_push = 0.0
    self.gap_reclaim_stabilization_push = 0.0
    self.gap_reclaim_projection_scale = 1.0
    self.gap_reclaim_personality_max_accel = 0.0
    self.gap_reclaim_effective_cap = 0.0
    self.lead_present_cruise_accel_cap = 0.0
    self._raw_reclaim_safety_override_active = False

    if not self._hyundai_ai_lead_stability_enabled:
      self._gap_reclaim_blend = 0.0
      self._gap_reclaim_last_t = now
      self.gap_reclaim_effective_cap = 0.0
      self.lead_keepup_accel_floor = 0.0
      self.lead_slowdown_accel_ceiling = None
      self._reset_lead_slowdown_ceiling_release_limit()
      return np.minimum(raw_lead_obstacle, filtered_lead_obstacle), False

    reclaim_lead = self._update_hyundai_reclaim_lead(now, raw_lead, filtered_lead, settled_follow=settled_follow)
    filtered_metrics = self._lead_follow_metrics(
      float(self.x0[1]),
      self.current_t_follow,
      filtered_lead,
      float(filtered_lead_obstacle[0]),
    )
    self.gap_reclaim_projection_scale = get_gap_reclaim_projection_scale(
      float(self.x0[1]),
      reclaim_lead,
      self.current_t_follow,
      ego_accel=float(self.x0[2]),
    )
    personality_max_accel = self._get_gap_reclaim_personality_max_accel(float(self.x0[1]))
    self.gap_reclaim_personality_max_accel = float(personality_max_accel or 0.0)
    self.gap_reclaim_effective_cap = get_gap_reclaim_effective_cap(
      float(self.x0[1]),
      reclaim_lead,
      self.current_t_follow,
      self._live_tune_cfg,
      personality_max_accel=personality_max_accel,
    )

    reclaim_intent = get_gap_reclaim_accel_floor(
      float(self.x0[1]),
      reclaim_lead,
      self.current_t_follow,
      self._live_tune_cfg,
      personality_max_accel=personality_max_accel,
    )
    keepup_intent = max(
      get_lead_keepup_accel_floor(
        float(self.x0[1]),
        raw_lead,
        self.current_t_follow,
        self._live_tune_cfg,
        personality_max_accel=personality_max_accel,
      ),
      get_lead_keepup_accel_floor(
        float(self.x0[1]),
        reclaim_lead,
        self.current_t_follow,
        self._live_tune_cfg,
        personality_max_accel=personality_max_accel,
      ),
    )
    self.gap_reclaim_accel_floor = float(reclaim_intent)
    self.lead_keepup_accel_floor = float(keepup_intent)
    slowdown_ceilings = [
      get_lead_slowdown_accel_ceiling(
        float(self.x0[1]),
        lead,
        self.current_t_follow,
        self._live_tune_cfg,
        min_accel=ACCEL_MIN,
        max_accel=ACCEL_MAX,
        anticipatory_enabled=anticipatory_slowdown_enabled,
      )
      for lead in (raw_lead, filtered_lead)
    ]
    active_slowdown_ceilings = [float(ceiling) for ceiling in slowdown_ceilings if ceiling is not None]
    slowdown_ceiling = min(active_slowdown_ceilings) if active_slowdown_ceilings else None
    self.lead_slowdown_accel_ceiling = self._limit_lead_slowdown_ceiling_release(
      slowdown_ceiling,
      raw_metrics,
      filtered_metrics,
      now,
    )
    max_intent = max(
      float(self.gap_reclaim_effective_cap),
      float(self._live_tune_cfg.lead_keepup_max_accel),
      1e-3,
    )
    combined_intent = max(reclaim_intent, keepup_intent)
    target_blend = combined_intent / max_intent if max_intent > 0.0 else 0.0
    blend = self._update_gap_reclaim_blend(target_blend, now)
    obstacle_delta = np.maximum(raw_lead_obstacle - filtered_lead_obstacle, 0.0)
    horizon_ramp = 1.0 - np.exp(-T_IDXS / GAP_RECLAIM_HORIZON_RAMP_TAU_S)
    stabilization_push = obstacle_delta * blend * horizon_ramp
    stabilization_push_suppressed = bool(
      settled_follow and
      float(raw_metrics["closing_speed"]) < HYUNDAI_RECLAIM_NOISE_SUPPRESS_CLOSING_MPS and
      float(raw_metrics["pullaway_speed"]) < HYUNDAI_RECLAIM_NOISE_SUPPRESS_PULLAWAY_MPS
    )
    if stabilization_push_suppressed:
      stabilization_push = np.zeros_like(stabilization_push)

    gap_surplus = max(0.0, float(reclaim_lead.dRel) - get_headway_follow_distance(float(self.x0[1]), self.current_t_follow))
    raw_gap_surplus = max(0.0, float(raw_metrics["gap_surplus"]))
    room_gap_surplus = max(gap_surplus, raw_gap_surplus if keepup_intent > 0.0 else 0.0)
    room_gap_min_m = float(self._live_tune_cfg.gap_reclaim_gap_min_m)
    if keepup_intent > 0.0:
      room_gap_min_m = min(room_gap_min_m, float(self._live_tune_cfg.lead_keepup_gap_min_m))
    reclaim_room_max = min(
      GAP_RECLAIM_RELAX_ROOM_MAX_M,
      max(0.0, room_gap_surplus - room_gap_min_m) * GAP_RECLAIM_RELAX_ROOM_FRACTION,
    )
    effective_reclaim_intent = max_intent * blend
    base_reclaim_room = np.minimum(0.5 * effective_reclaim_intent * np.square(T_IDXS), reclaim_room_max * horizon_ramp)
    reclaim_room = base_reclaim_room * float(self.gap_reclaim_projection_scale)

    obstacle_push = stabilization_push + reclaim_room
    self.gap_reclaim_obstacle_push = float(np.max(obstacle_push))
    self.gap_reclaim_stabilization_push = float(np.max(stabilization_push))
    target_obstacle = np.minimum(raw_lead_obstacle, filtered_lead_obstacle + stabilization_push) + reclaim_room

    raw_obstacle_margin = float(np.min(filtered_lead_obstacle - raw_lead_obstacle))
    raw_lead_accel = float(getattr(raw_lead, 'aLeadK', 0.0) or 0.0)
    raw_safety_override = (
      raw_obstacle_margin >= HYUNDAI_RECLAIM_RAW_SAFETY_MARGIN_M or
      float(raw_metrics["closing_speed"]) > HYUNDAI_RECLAIM_RAW_SAFETY_CLOSING_MPS or
      (
        raw_lead_accel < HYUNDAI_RECLAIM_RAW_SAFETY_DECEL_MPS2 and
        float(raw_metrics["closing_speed"]) > HYUNDAI_RECLAIM_RAW_SAFETY_DECEL_CLOSING_MPS
      )
    )
    self._raw_reclaim_safety_override_active = bool(raw_safety_override)
    if raw_safety_override:
      self._hyundai_reclaim_lead = copy.deepcopy(filtered_lead)
      self._hyundai_reclaim_last_t = now
      target_obstacle = np.minimum(target_obstacle, raw_lead_obstacle)

    return target_obstacle, stabilization_push_suppressed

  def _update_hyundai_reclaim_lead(self, now: float, raw_lead, filtered_lead, *,
                                   settled_follow: bool) -> ControlLead:
    raw_control = ControlLead.from_lead(raw_lead) if raw_lead is not None else ControlLead()
    raw_metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, raw_control)
    dynamic_gap_supported = (
      raw_metrics["gap_surplus"] > float(self._live_tune_cfg.gap_reclaim_gap_min_m) + HYUNDAI_RECLAIM_DYNAMIC_GAP_SUPPORT_M and
      raw_metrics["closing_speed"] < HYUNDAI_RECLAIM_PULLAWAY_SUPPORT_MPS and
      float(raw_control.aLeadK) > -0.6
    )
    dynamic_pullaway_supported = (
      raw_metrics["pullaway_speed"] > HYUNDAI_RECLAIM_PULLAWAY_SUPPORT_MPS or
      float(raw_control.aLeadK) > 0.10 or
      dynamic_gap_supported
    )
    distance_hold_supported = (
      raw_metrics["gap_surplus"] > float(self._live_tune_cfg.gap_reclaim_gap_min_m) + HYUNDAI_RECLAIM_GAP_HOLD_EXTRA_M or
      dynamic_pullaway_supported
    )

    optimistic = copy.deepcopy(filtered_lead)
    if distance_hold_supported and not settled_follow:
      optimistic.dRel = max(float(filtered_lead.dRel), float(raw_control.dRel))
    if dynamic_pullaway_supported:
      optimistic.vRel = max(float(filtered_lead.vRel), float(raw_control.vRel))
      optimistic.aRel = max(float(filtered_lead.aRel), float(raw_control.aRel))
      optimistic.vLead = max(float(filtered_lead.vLead), float(raw_control.vLead))
      optimistic.vLeadK = max(float(filtered_lead.vLeadK), float(raw_control.vLeadK))
      optimistic.aLeadK = max(float(filtered_lead.aLeadK), float(raw_control.aLeadK))
    optimistic.modelProb = max(float(filtered_lead.modelProb), float(raw_control.modelProb))

    if self._hyundai_virtual_lead_identity_changed or self._hyundai_reclaim_lead is None or self._hyundai_reclaim_last_t is None:
      self._hyundai_reclaim_lead = copy.deepcopy(optimistic)
      self._hyundai_reclaim_last_t = now
      return copy.deepcopy(self._hyundai_reclaim_lead)

    dt_s = float(np.clip(now - float(self._hyundai_reclaim_last_t), 0.0, 1.0))
    prev = self._hyundai_reclaim_lead
    reclaim = copy.deepcopy(prev)
    reclaim.status = optimistic.status
    reclaim.dRel = self._filter_metric(prev.dRel, optimistic.dRel, dt_s, danger_if_lower=not distance_hold_supported)
    reclaim.yRel = filtered_lead.yRel
    reclaim.dPath = filtered_lead.dPath
    reclaim.vLat = filtered_lead.vLat
    reclaim.vRel = self._filter_metric(prev.vRel, optimistic.vRel, dt_s, danger_if_lower=not dynamic_pullaway_supported)
    reclaim.aRel = self._filter_metric(prev.aRel, optimistic.aRel, dt_s, danger_if_lower=not dynamic_pullaway_supported)
    reclaim.vLead = self._filter_metric(prev.vLead, optimistic.vLead, dt_s, danger_if_lower=not dynamic_pullaway_supported)
    reclaim.vLeadK = self._filter_metric(prev.vLeadK, optimistic.vLeadK, dt_s, danger_if_lower=not dynamic_pullaway_supported)
    reclaim.aLeadK = self._filter_metric(prev.aLeadK, optimistic.aLeadK, dt_s, danger_if_lower=not dynamic_pullaway_supported,
                                         slow_tau_s=self._live_tune_cfg.virtual_lead_slow_tau_s,
                                         sign_transition_tau_s=HYUNDAI_VIRTUAL_LEAD_SIGN_TRANSITION_TAU_S)
    reclaim.fcw = optimistic.fcw
    reclaim.fcwSuppressed = bool(getattr(optimistic, 'fcwSuppressed', False))
    reclaim.aLeadTau = optimistic.aLeadTau
    reclaim.modelProb = self._filter_symmetric_metric(prev.modelProb, optimistic.modelProb, dt_s, HYUNDAI_VIRTUAL_LEAD_MODEL_PROB_TAU_S)
    reclaim.radar = optimistic.radar
    reclaim.radarTrackId = optimistic.radarTrackId
    self._hyundai_reclaim_lead = reclaim
    self._hyundai_reclaim_last_t = now
    return copy.deepcopy(reclaim)

  def _update_hyundai_virtual_lead(self, now: float, lead_source: str | None, lead):
    self._hyundai_virtual_lead_identity_changed = False

    if not self._hyundai_ai_lead_stability_enabled:
      self.hyundai_virtual_lead_debug = {"active": False}
      return None

    raw_lead = ControlLead.from_lead(lead) if lead is not None else ControlLead()
    if not raw_lead.status:
      held_lead, dropout_hold_debug = self._maybe_hold_hyundai_virtual_lead_dropout(now)
      if held_lead is not None:
        metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, held_lead)
        self.hyundai_virtual_lead_debug = {
          "active": True,
          "source": str(self._hyundai_virtual_lead_source),
          "identity_changed": False,
          "reset_reason": "dropout_hold",
          "filtered": self._lead_debug_payload(held_lead),
          "metrics": metrics,
          "drel_consistency_m": float(self._hyundai_virtual_lead_last_drel_error_m),
          "filter": dict((self._drel_kalman if self._use_kalman_drel else self._drel_filter).last_debug),
          "drel_filter_type": "kalman" if self._use_kalman_drel else "ema",
          "dropout_hold": dropout_hold_debug,
        }
        return held_lead
    else:
      self._clear_hyundai_virtual_lead_dropout_hold()

    should_reset, reset_reason = self._should_reset_hyundai_virtual_lead(lead_source, lead)

    if should_reset:
      self._hyundai_virtual_lead = raw_lead
      self._hyundai_virtual_lead_source = lead_source if raw_lead.status else None
      self._hyundai_virtual_lead_last_t = now if raw_lead.status else None
      self._hyundai_virtual_lead_identity_changed = bool(raw_lead.status)
      self._hyundai_virtual_lead_reset_reason = reset_reason
      self._hyundai_virtual_lead_stable_since_t = now if raw_lead.status else None
      self._hyundai_virtual_lead_last_drel_error_m = 0.0 if raw_lead.status else 1e9
      self._drel_filter.reset(raw_lead.dRel if raw_lead.status else None)
      self._drel_kalman.reset(raw_lead.dRel if raw_lead.status else None)
      if raw_lead.status and reset_reason == "drel_jump_closer":
        self._drel_filter.last_debug.update({
          "predicted_vrel_mps": float(getattr(raw_lead, 'vRel', 0.0) or 0.0),
          "innovation_raw_m": 0.0,
          "innovation_used_m": 0.0,
          "deadband_applied": False,
          "open_slew_clamped": False,
          "snap_to_raw": True,
        })
      if not raw_lead.status:
        self._reset_hyundai_virtual_lead(reset_reason or "no_control_lead")
        return None
    else:
      dt_s = float(np.clip(now - float(self._hyundai_virtual_lead_last_t or now), 0.0, 1.0))
      prev = self._hyundai_virtual_lead
      filtered = copy.deepcopy(prev)
      filtered.status = raw_lead.status
      cfg = self._live_tune_cfg
      raw_vrel = float(getattr(raw_lead, 'vRel', 0.0) or 0.0)
      if self._use_kalman_drel:
        filtered.dRel = self._drel_kalman.update(raw_lead.dRel, raw_vrel, dt_s)
      else:
        filtered.dRel = self._drel_filter.update(
          raw_lead.dRel, raw_vrel, dt_s,
          tau_close=getattr(cfg, 'drel_filter_tau_close_s', DREL_FILTER_TAU_CLOSE_S),
          tau_open=getattr(cfg, 'drel_filter_tau_open_s', DREL_FILTER_TAU_OPEN_S),
          innovation_gate=getattr(cfg, 'drel_filter_innovation_gate_m', DREL_FILTER_INNOVATION_GATE_M),
          closing_gate=getattr(cfg, 'drel_filter_closing_gate_m', DREL_FILTER_CLOSING_GATE_M),
          open_slew_max_mps=getattr(cfg, 'drel_filter_open_slew_max_mps', DREL_FILTER_OPEN_SLEW_MAX_MPS),
        )
      filtered.yRel = self._filter_symmetric_metric(prev.yRel, raw_lead.yRel, dt_s, HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S)
      filtered.vRel = self._filter_metric(prev.vRel, raw_lead.vRel, dt_s, danger_if_lower=True)
      filtered.aRel = self._filter_metric(prev.aRel, raw_lead.aRel, dt_s, danger_if_lower=True)
      filtered.vLead = self._filter_metric(prev.vLead, raw_lead.vLead, dt_s, danger_if_lower=True)
      filtered.dPath = self._filter_symmetric_metric(prev.dPath, raw_lead.dPath, dt_s, HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S)
      filtered.vLat = self._filter_symmetric_metric(prev.vLat, raw_lead.vLat, dt_s, HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S)
      filtered.vLeadK = self._filter_metric(prev.vLeadK, raw_lead.vLeadK, dt_s, danger_if_lower=True)
      filtered.aLeadK = self._filter_metric(prev.aLeadK, raw_lead.aLeadK, dt_s, danger_if_lower=True,
                                             slow_tau_s=cfg.virtual_lead_slow_tau_s,
                                             sign_transition_tau_s=HYUNDAI_VIRTUAL_LEAD_SIGN_TRANSITION_TAU_S)
      filtered.fcw = raw_lead.fcw
      filtered.fcwSuppressed = bool(getattr(raw_lead, 'fcwSuppressed', False))
      filtered.aLeadTau = raw_lead.aLeadTau
      filtered.modelProb = self._filter_symmetric_metric(prev.modelProb, raw_lead.modelProb, dt_s, HYUNDAI_VIRTUAL_LEAD_MODEL_PROB_TAU_S)
      filtered.radar = raw_lead.radar
      filtered.radarTrackId = raw_lead.radarTrackId
      # A stabilizer phantom is synthetic, noiseless extrapolation: smoothing it
      # only lags the propagated closure, making the MPC's lead MORE optimistic
      # than the held measurement. Clamp the filtered view to be no more
      # optimistic than the phantom while the source slot is phantom-held.
      slot_idx = {"lead0": 0, "lead1": 1}.get(str(lead_source), None)
      if slot_idx is not None and self._lead_stability_phantom_slots[slot_idx]:
        filtered.dRel = min(filtered.dRel, raw_lead.dRel)
        filtered.vRel = min(filtered.vRel, raw_lead.vRel)
        filtered.vLead = min(filtered.vLead, raw_lead.vLead)
        filtered.vLeadK = min(filtered.vLeadK, raw_lead.vLeadK)
        filtered.aLeadK = min(filtered.aLeadK, raw_lead.aLeadK)
      self._hyundai_virtual_lead_last_drel_error_m = abs(float(raw_lead.dRel) - float(filtered.dRel))
      self._hyundai_virtual_lead = filtered
      self._hyundai_virtual_lead_source = lead_source
      self._hyundai_virtual_lead_last_t = now
      self._hyundai_virtual_lead_reset_reason = None

    metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, self._hyundai_virtual_lead)
    dropout_hold_debug = self._get_hyundai_virtual_lead_dropout_debug(now)
    dropout_hold_debug["activated"] = False
    self.hyundai_virtual_lead_debug = {
      "active": True,
      "source": str(self._hyundai_virtual_lead_source),
      "identity_changed": bool(self._hyundai_virtual_lead_identity_changed),
      "reset_reason": self._hyundai_virtual_lead_reset_reason,
      "filtered": self._lead_debug_payload(self._hyundai_virtual_lead),
      "metrics": metrics,
      "drel_consistency_m": float(self._hyundai_virtual_lead_last_drel_error_m),
      "filter": dict((self._drel_kalman if self._use_kalman_drel else self._drel_filter).last_debug),
      "drel_filter_type": "kalman" if self._use_kalman_drel else "ema",
      "dropout_hold": dropout_hold_debug,
    }
    return self._hyundai_virtual_lead

  def _duplicate_candidate_metrics(self, lead) -> dict[str, float]:
    return {
      "dPath": abs(self._lead_attr(lead, "dPath", self._lead_attr(lead, "yRel"))),
      "modelProb": self._lead_attr(lead, "modelProb"),
      "vLat": abs(self._lead_attr(lead, "vLat")),
      "dRel": self._lead_attr(lead, "dRel", 1e9),
    }

  def _duplicate_candidate_sort_key(self, lead) -> tuple[float, float, float, float]:
    metrics = self._duplicate_candidate_metrics(lead)
    return (metrics["dPath"], -metrics["modelProb"], metrics["vLat"], metrics["dRel"])

  def _is_materially_better_duplicate_candidate(self, challenger, incumbent) -> bool:
    challenger_metrics = self._duplicate_candidate_metrics(challenger)
    incumbent_metrics = self._duplicate_candidate_metrics(incumbent)

    if challenger_metrics["dPath"] + HYUNDAI_DUPLICATE_PATH_SWITCH_M < incumbent_metrics["dPath"]:
      return True
    if (challenger_metrics["modelProb"] >
        incumbent_metrics["modelProb"] + HYUNDAI_DUPLICATE_MODEL_PROB_SWITCH and
        challenger_metrics["dPath"] <= incumbent_metrics["dPath"] + HYUNDAI_DUPLICATE_PATH_SWITCH_M):
      return True
    if (challenger_metrics["vLat"] + HYUNDAI_DUPLICATE_VLAT_SWITCH_MPS < incumbent_metrics["vLat"] and
        challenger_metrics["dPath"] <= incumbent_metrics["dPath"] + HYUNDAI_DUPLICATE_PATH_SWITCH_M):
      return True
    if (challenger_metrics["dRel"] + HYUNDAI_DUPLICATE_DREL_SWITCH_M < incumbent_metrics["dRel"] and
        challenger_metrics["dPath"] <= incumbent_metrics["dPath"] + HYUNDAI_DUPLICATE_PATH_SWITCH_M):
      return True
    return False

  def _should_use_hyundai_virtual_duplicate(self, lead_role_debug: dict[str, object]) -> bool:
    if not self._hyundai_ai_lead_stability_enabled:
      return False
    if not bool(lead_role_debug.get("duplicate_pair", False)):
      return False
    roles = lead_role_debug.get("roles", {})
    return (
      roles.get("lead0") == LeadRoleClassifier.CENTER_CONTROL and
      roles.get("lead1") == LeadRoleClassifier.CENTER_CONTROL
    )

  def _apply_lead_accel_corr_bound(self, slot: int, lead: _StabilizedLead, raw_valid: bool, now: float) -> bool:
    """Bound uncorroborated negative aLeadK by the measured vLead trend at the
    single lead ingress feeding role classifier, previews, process_lead and the
    brake-release floor. Applied ONLY on fresh-measurement, non-phantom frames
    in non-dangerous states: a phantom/held lead keeps its held decel (never
    made more optimistic than the last measured state), and any dangerous state
    passes full aLeadK through. Returns True when the bound trimmed aLeadK."""
    state = self._lead_stability_state[slot]
    cfg = self._live_tune_cfg
    margin = float(getattr(cfg, 'lead_accel_corr_margin_mps2', LEAD_ACCEL_CORR_DISABLE_MARGIN_MPS2))
    if margin >= LEAD_ACCEL_CORR_DISABLE_MARGIN_MPS2:
      state.corr_meas_t = None
      state.corr_danger_latched = False
      return False
    if not lead.status:
      state.corr_meas_t = None
      state.corr_a_meas_lp = 0.0
      state.corr_settled_s = 0.0
      state.corr_track_id = None
      state.corr_danger_latched = False
      return False

    fresh = raw_valid and not self._lead_stability_phantom_slots[slot]
    if not fresh:
      # No fresh measurement (phantom hold / stale latch): freeze the low-pass
      # (a decay toward zero would fabricate corroboration) and never clamp.
      return False

    meas_tau = max(0.1, float(getattr(cfg, 'lead_accel_corr_meas_tau_s', 0.3)))
    max_dt_s = float(getattr(cfg, 'lead_accel_corr_max_dt_s', LEAD_ACCEL_CORR_MAX_DT_S))
    if state.corr_meas_t is None:
      identity_change = True
      dt = 0.0
      a_fd = 0.0
    else:
      dt = float(now) - float(state.corr_meas_t)
      a_fd = (float(lead.vLead) - float(state.corr_meas_v)) / dt if dt > 1e-3 else 0.0
      identity_change = (
        state.corr_track_id != int(lead.radarTrackId) or
        not (1e-3 < dt < max_dt_s) or
        abs(a_fd) > LEAD_ACCEL_CORR_MEAS_A_GATE_MPS2
      )
    if identity_change:
      state.corr_a_meas_lp = 0.0
      state.corr_settled_s = 0.0
    else:
      alpha = dt / (dt + meas_tau)
      state.corr_a_meas_lp += alpha * (a_fd - state.corr_a_meas_lp)
      state.corr_settled_s += dt
    state.corr_meas_t = float(now)
    state.corr_meas_v = float(lead.vLead)
    state.corr_track_id = int(lead.radarTrackId)

    v_ego = float(self.x0[1])
    closing = max(0.0, v_ego - float(lead.vLead))
    ttc = float(lead.dRel) / max(closing, 0.1)
    ttc_guard = float(getattr(cfg, 'lead_accel_corr_ttc_guard_s', 8.0))
    closing_guard = float(getattr(cfg, 'lead_accel_corr_closing_guard_mps', 1.5))
    near_gap_m = float(getattr(cfg, 'lead_accel_corr_near_headway_s', 1.2)) * v_ego
    closing_rearm_mps = float(getattr(cfg, 'lead_accel_corr_closing_rearm_mps', LEAD_ACCEL_CORR_CLOSING_REARM_MPS))
    ttc_rearm_s = float(getattr(cfg, 'lead_accel_corr_ttc_rearm_s', LEAD_ACCEL_CORR_TTC_REARM_S))
    headway_rearm_m = float(getattr(cfg, 'lead_accel_corr_headway_rearm_m', LEAD_ACCEL_CORR_HEADWAY_REARM_M))
    settle_tau_mult = float(getattr(cfg, 'lead_accel_corr_settle_tau_mult', LEAD_ACCEL_CORR_SETTLE_TAU_MULT))
    if ttc <= ttc_guard or closing >= closing_guard or float(lead.dRel) <= near_gap_m:
      state.corr_danger_latched = True
    elif (closing <= max(0.0, closing_guard - closing_rearm_mps) and
          ttc >= ttc_guard + ttc_rearm_s and
          float(lead.dRel) > near_gap_m + headway_rearm_m):
      state.corr_danger_latched = False

    settled = state.corr_settled_s >= settle_tau_mult * meas_tau

    # Corroborated aLeadK AMPLIFY (CD3 lead-decel truth deficit). The model's
    # leadsV3 'a' chronically underreports real lead braking (road 200-13:
    # published aLeadK peaked -0.48 while the position-derived truth was
    # -1.3..-2.3 m/s^2), and radard's 0.6 s accel EMA halves it again. The
    # vLead-trend finite-difference (corr_a_meas_lp, already low-passed here)
    # measures the real decel from vLead alone: in that exact event it tracked
    # -1.7..-1.9 while aLeadK sat at -0.54. Pull aLeadK toward that measured
    # trend when BOTH agree the lead is braking, so the MPC extrapolates a
    # truthful decel instead of a barely-moving lead.
    #
    # Safety-shaped gates (why this cannot fabricate phantom braking):
    #  - lead.aLeadK < 0: the MODEL must already report braking. A coasting or
    #    accelerating model report is never overridden into decel; the trend
    #    can only DEEPEN an already-reported brake, never invent one.
    #  - corr_a_meas_lp < lead.aLeadK - deadband: the kinematic trend must be
    #    meaningfully MORE negative than the model. The deadband rejects the
    #    finite-difference jitter of a steady/lightly-braking lead (ev6_measured
    #    vRel noise + prob dropouts) that would otherwise chatter aLeadK.
    #  - settled: same-track vLead history >= settle_tau_mult * meas_tau, so a
    #    fresh acquisition / track swap cannot inject a spurious first-frame trend.
    #  - runs while danger-latched (unlike the downward bound): a genuine close
    #    is exactly when the deficit is dangerous, and the model-already-negative
    #    gate keeps it corroborated.
    # Gain / deadband / cap are all live-tunable; gain 0 restores the old
    # bound-only behavior exactly (kill switch).
    amplify_gain = float(getattr(cfg, 'lead_accel_corr_amplify_gain',
                                 LEAD_ACCEL_CORR_AMPLIFY_GAIN))
    amplify_deadband = float(getattr(cfg, 'lead_accel_corr_amplify_deadband_mps2',
                                     LEAD_ACCEL_CORR_AMPLIFY_DEADBAND_MPS2))
    amplify_cap = float(getattr(cfg, 'lead_accel_corr_amplify_cap_mps2',
                                LEAD_ACCEL_CORR_AMPLIFY_CAP_MPS2))
    if (amplify_gain > 0.0 and settled and lead.aLeadK < 0.0 and
        float(state.corr_a_meas_lp) < float(lead.aLeadK) - amplify_deadband):
      # Never pull past the measured trend, and never deepen by more than the
      # per-frame cap below the current aLeadK (bounds a single noisy trend
      # sample). Sign is preserved: target is always <= aLeadK < 0.
      target = max(float(state.corr_a_meas_lp), float(lead.aLeadK) - amplify_cap)
      amplified = float(lead.aLeadK) + amplify_gain * (target - float(lead.aLeadK))
      if amplified < float(lead.aLeadK):
        lead.aLeadK = amplified
        return True

    if (state.corr_danger_latched or lead.aLeadK >= 0.0 or not settled):
      return False
    bounded = max(float(lead.aLeadK), min(0.0, float(state.corr_a_meas_lp)) - margin)
    if bounded <= float(lead.aLeadK):
      return False
    lead.aLeadK = bounded
    return True

  def _stabilize_raw_leads(self, raw_lead0: Any, raw_lead1: Any, now: float) -> tuple[_StabilizedLead, _StabilizedLead]:
    """Apply acquire/release dwell + phantom extrapolation to each raw lead.
    Returns duck-typed _StabilizedLead objects that downstream MPC code treats
    as if they were capnp readers. At shipped defaults (PhantomLeadHoldS=0.8)
    the phantom owns release: the release_frames unlatch branch only runs when
    phantom_hold_s=0. Set acquire_frames=1, release_frames=1, phantom_hold_s=0
    to collapse to raw passthrough."""
    cfg = self._live_tune_cfg
    acquire_frames = int(max(1, round(float(getattr(cfg, 'lead_source_acquire_frames', 1.0) or 1.0))))
    release_frames = int(max(1, round(float(getattr(cfg, 'lead_source_release_frames', 1.0) or 1.0))))
    phantom_hold_s = float(max(0.0, float(getattr(cfg, 'phantom_lead_hold_s', 0.0) or 0.0)))
    stable_frames = int(max(1, round(float(getattr(cfg, 'phantom_lead_stable_frames', 1.0) or 1.0))))

    outs: list[_StabilizedLead] = []
    phantom_slots = [False, False]
    raw_valids = [False, False]
    for slot, raw in enumerate((raw_lead0, raw_lead1)):
      state = self._lead_stability_state[slot]
      raw_valid = bool(getattr(raw, 'status', False)) if raw is not None else False
      raw_valids[slot] = raw_valid

      if raw_valid:
        state.valid_streak += 1
        state.invalid_streak = 0
        if state.latched:
          state.latched_valid_streak += 1
        new_valid = _StabilizedLead.from_reader(raw)
        if state.last_valid is not None and state.last_valid_t is not None:
          trend_dt = float(now) - float(state.last_valid_t)
          drel_jump_gate_m = float(getattr(cfg, 'lead_stabilizer_trend_drel_jump_m', LEAD_STABILIZER_TREND_DREL_JUMP_M))
          yrel_jump_gate_m = float(getattr(cfg, 'lead_stabilizer_trend_yrel_jump_m', LEAD_STABILIZER_TREND_YREL_JUMP_M))
          trend_tau_s = float(getattr(cfg, 'lead_stabilizer_trend_tau_s', LEAD_STABILIZER_TREND_TAU_S))
          expected_drel = state.last_valid.dRel + state.last_valid.vRel * trend_dt
          identity_jump = (abs(new_valid.dRel - expected_drel) > drel_jump_gate_m
                           or abs(new_valid.yRel - state.last_valid.yRel) > yrel_jump_gate_m)
          if identity_jump or not (1e-3 < trend_dt < 0.5):
            state.a_lead_k_trend = 0.0
          else:
            trend_raw = (new_valid.aLeadK - state.last_valid.aLeadK) / trend_dt
            alpha = trend_dt / (trend_dt + trend_tau_s)
            state.a_lead_k_trend += alpha * (trend_raw - state.a_lead_k_trend)
        else:
          state.a_lead_k_trend = 0.0
        state.last_valid = new_valid
        state.last_valid_t = now
      else:
        state.invalid_streak += 1
        state.valid_streak = 0

      if state.latched:
        if state.invalid_streak >= release_frames and not (phantom_hold_s > 0.0):
          state.latched = False
          state.latched_valid_streak = 0
      else:
        if state.valid_streak >= acquire_frames:
          state.latched = True
          state.latched_valid_streak = state.valid_streak

      if state.latched and raw_valid:
        outs.append(_StabilizedLead.from_reader(raw))
        continue

      if (state.latched and not raw_valid and state.last_valid is not None
          and state.last_valid_t is not None and phantom_hold_s > 0.0):
        age = max(0.0, now - state.last_valid_t)
        if age > phantom_hold_s or state.latched_valid_streak < stable_frames:
          state.latched = False
          state.latched_valid_streak = 0
          outs.append(_StabilizedLead(status=False))
          continue
        # Kill phantom if a DIFFERENT candidate appears in the other slot at a
        # materially different yRel — suggests our lead left and a new car came in.
        other_idx = 1 - slot
        other = (raw_lead0, raw_lead1)[other_idx]
        if other is not None and bool(getattr(other, 'status', False)):
          other_y = _StabilizedLead._safe_attr(other, 'yRel')
          if abs(other_y - state.last_valid.yRel) > self.LEAD_STABILIZER_PHANTOM_YREL_KILL_M:
            state.latched = False
            state.latched_valid_streak = 0
            outs.append(_StabilizedLead(status=False))
            continue
        decay = max(0.0, 1.0 - age / max(phantom_hold_s, 1e-3))
        # A held lead must never be kinematically more optimistic than its last
        # measurement: hold measured decel through the phantom window (scaled by
        # phantom_decel_hold_factor; 0 restores legacy decay-to-zero), continue
        # the measured deepening trend of the lagged aLeadK estimate (never the
        # relaxing trend), and propagate vRel/vLead/dRel with it. Positive accel
        # still decays — an extrapolated pull-away is the optimistic direction.
        hold_factor = float(min(1.0, max(0.0, float(getattr(cfg, 'phantom_lead_decel_hold_factor', 1.0) or 0.0))))
        trend_gain = float(min(1.0, max(0.0, float(getattr(cfg, 'phantom_lead_decel_trend_gain', 1.0) or 0.0))))
        a_meas = state.last_valid.aLeadK
        if a_meas < 0.0:
          a_base = a_meas * (hold_factor + (1.0 - hold_factor) * decay)
          trend = min(0.0, state.a_lead_k_trend) * trend_gain
          a_hold = max(-10.0, a_base + trend * age)
        else:
          a_base = a_meas * decay
          trend = 0.0
          a_hold = a_base
        a_prop = min(a_base, 0.0)
        phantom = _StabilizedLead(
          status=True,
          dRel=max(1.0, state.last_valid.dRel + state.last_valid.vRel * age
                   + 0.5 * a_prop * age * age + trend * age ** 3 / 6.0),
          yRel=state.last_valid.yRel,
          vRel=state.last_valid.vRel + a_prop * age + 0.5 * trend * age * age,
          vLead=max(0.0, state.last_valid.vLead + a_prop * age + 0.5 * trend * age * age),
          aLeadK=a_hold,
          modelProb=state.last_valid.modelProb * decay,
          dPath=state.last_valid.dPath,
          vLat=state.last_valid.vLat,
          aLeadTau=state.last_valid.aLeadTau,
          aRel=state.last_valid.aRel,
          vLeadK=state.last_valid.vLeadK,
          fcw=state.last_valid.fcw,
          fcwSuppressed=bool(getattr(state.last_valid, 'fcwSuppressed', False)),
          radar=state.last_valid.radar,
          radarTrackId=state.last_valid.radarTrackId,
        )
        phantom_slots[slot] = True
        outs.append(phantom)
        continue

      outs.append(_StabilizedLead(status=False))

    self._lead_stability_phantom_slots = tuple(phantom_slots)
    corr_clamped = [
      self._apply_lead_accel_corr_bound(slot, outs[slot], raw_valids[slot], now)
      for slot in range(2)
    ]
    self.lead_stability_debug = {
      f"slot{slot}": {
        "latched": bool(state.latched),
        "valid_streak": int(state.valid_streak),
        "invalid_streak": int(state.invalid_streak),
        "latched_valid_streak": int(state.latched_valid_streak),
        "phantom_age_s": (float(now - state.last_valid_t) if state.last_valid_t is not None else None),
        "phantom_active": bool(phantom_slots[slot]),
        "out_status": bool(outs[slot].status),
        "accel_corr_clamped": bool(corr_clamped[slot]),
        "accel_corr_a_meas_lp": float(state.corr_a_meas_lp),
        "accel_corr_danger_latched": bool(state.corr_danger_latched),
      }
      for slot, state in enumerate(self._lead_stability_state)
    }
    return outs[0], outs[1]

  def _stabilize_control_leads(self, raw_lead0, raw_lead1,
                               control_lead0: ControlLead,
                               control_lead1: ControlLead,
                               lead_role_debug: dict[str, object]) -> tuple[tuple[ControlLead, ControlLead], dict[str, object]]:
    if not self._should_use_hyundai_virtual_duplicate(lead_role_debug):
      self._hyundai_duplicate_selected_raw_slot = None
      debug = copy.deepcopy(lead_role_debug)
      debug["virtual_duplicate"] = {"active": False}
      return (control_lead0, control_lead1), debug

    raw_leads = {0: raw_lead0, 1: raw_lead1}
    current_best_slot = min(raw_leads, key=lambda idx: self._duplicate_candidate_sort_key(raw_leads[idx]))
    previous_slot = self._hyundai_duplicate_selected_raw_slot
    selected_slot = current_best_slot
    held_previous = False
    if previous_slot in raw_leads and previous_slot != current_best_slot:
      if not self._is_materially_better_duplicate_candidate(raw_leads[current_best_slot], raw_leads[previous_slot]):
        selected_slot = previous_slot
        held_previous = True
    self._hyundai_duplicate_selected_raw_slot = selected_slot

    selected_lead = ControlLead.from_lead(raw_leads[selected_slot])
    raw_cutin_promoted = lead_role_debug.get("cutin_promoted", {})
    raw_toward_center = lead_role_debug.get("toward_center_mps", {})
    raw_roles = lead_role_debug.get("roles", {})
    raw_reasons = lead_role_debug.get("reasons", {})
    raw_control_status = lead_role_debug.get("control_status", {})

    debug = copy.deepcopy(lead_role_debug)
    debug["raw_duplicate_model"] = {
      "roles": raw_roles,
      "reasons": raw_reasons,
      "control_status": raw_control_status,
      "dropped_slot": lead_role_debug.get("dropped_slot"),
    }
    debug["virtual_duplicate"] = {
      "active": True,
      "selected_raw_slot": int(selected_slot),
      "suppressed_raw_slot": int(1 - selected_slot),
      "held_previous_slot": bool(held_previous),
      "previous_raw_slot": None if previous_slot is None else int(previous_slot),
    }
    debug["roles"] = {"lead0": LeadRoleClassifier.CENTER_CONTROL, "lead1": LeadRoleClassifier.INVALID}
    debug["reasons"] = {"lead0": f"virtual_duplicate_raw_{selected_slot}", "lead1": "suppressed_duplicate"}
    debug["toward_center_mps"] = {
      "lead0": float(max(
        float(raw_toward_center.get("lead0", 0.0) or 0.0),
        float(raw_toward_center.get("lead1", 0.0) or 0.0),
      )),
      "lead1": 0.0,
    }
    debug["cutin_promoted"] = {
      "lead0": bool(raw_cutin_promoted.get("lead0", False) or raw_cutin_promoted.get("lead1", False)),
      "lead1": False,
    }
    debug["control_status"] = {"lead0": True, "lead1": False}
    debug["dropped_slot"] = 1
    debug["raw"] = {
      "lead0": self._lead_debug_payload(raw_leads[selected_slot]),
      "lead1": self._empty_lead_debug_payload(),
    }
    awareness = [
      entry for entry in debug.get("awareness", [])
      if int(entry.get("slot", -1)) != selected_slot
    ]
    awareness.append({
      "slot": 1,
      "role": "suppressed_duplicate",
      "dRel": self._lead_attr(raw_leads[1 - selected_slot], "dRel"),
      "yRel": self._lead_attr(raw_leads[1 - selected_slot], "yRel"),
      "dPath": self._lead_attr(raw_leads[1 - selected_slot], "dPath", self._lead_attr(raw_leads[1 - selected_slot], "yRel")),
      "vLat": self._lead_attr(raw_leads[1 - selected_slot], "vLat"),
      "vRel": self._lead_attr(raw_leads[1 - selected_slot], "vRel"),
      "dropped_duplicate": True,
    })
    debug["awareness"] = awareness
    return (selected_lead, ControlLead()), debug

  def _reset_acc_obstacle_candidate(self) -> None:
    self._acc_obstacle_candidate_mode = None
    self._acc_obstacle_candidate_t = None

  def _get_best_control_lead(self) -> tuple[str | None, ControlLead | None]:
    if self.control_leads[0] is not None and getattr(self.control_leads[0], 'status', False):
      return 'lead0', self.control_leads[0]
    if self.control_leads[1] is not None and getattr(self.control_leads[1], 'status', False):
      return 'lead1', self.control_leads[1]
    return None, None

  def _advance_acc_obstacle_candidate(self, candidate_mode: str, now: float, dwell_s: float, default_mode: str) -> str:
    if self._acc_obstacle_candidate_mode != candidate_mode:
      self._acc_obstacle_candidate_mode = candidate_mode
      self._acc_obstacle_candidate_t = now
      return default_mode
    if self._acc_obstacle_candidate_t is None or (now - self._acc_obstacle_candidate_t) < dwell_s:
      return default_mode
    self._acc_obstacle_mode = candidate_mode
    self._reset_acc_obstacle_candidate()
    return candidate_mode

  def _select_acc_obstacle(self, lead_0_obstacle, lead_1_obstacle, cruise_obstacle, now: float) -> np.ndarray:
    lead_candidates: list[tuple[str, np.ndarray]] = []
    if self.control_leads[0] is not None and getattr(self.control_leads[0], 'status', False):
      lead_candidates.append(('lead0', lead_0_obstacle))
    if self.control_leads[1] is not None and getattr(self.control_leads[1], 'status', False):
      lead_candidates.append(('lead1', lead_1_obstacle))

    if not self._hyundai_ai_lead_stability_enabled:
      self.hyundai_virtual_lead_debug = {"active": False}
      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
      self.source = SOURCES[np.argmin(x_obstacles[0])]
      self.acc_source_debug = {
        "active_mode": "lead" if self.source in ('lead0', 'lead1') else "cruise",
        "delta_m": None,
        "candidate_mode": None,
        "used_hysteresis": False,
      }
      return np.min(x_obstacles, axis=1)

    if not lead_candidates:
      # Snapshot virtual-lead state before calling _update_hyundai_virtual_lead,
      # which resets that state as a side effect if dropout hold fails. The
      # classifier demotion hold needs the pre-reset snapshot to evaluate.
      snap_virtual_lead = copy.deepcopy(self._hyundai_virtual_lead) if self._hyundai_virtual_lead is not None else None
      snap_virtual_lead_source = self._hyundai_virtual_lead_source
      snap_virtual_lead_last_t = self._hyundai_virtual_lead_last_t
      snap_virtual_lead_reset_reason = self._hyundai_virtual_lead_reset_reason
      snap_virtual_lead_stable_since_t = self._hyundai_virtual_lead_stable_since_t
      snap_virtual_lead_last_drel_error_m = self._hyundai_virtual_lead_last_drel_error_m
      snap_virtual_lead_dropout_until_t = self._hyundai_virtual_lead_dropout_until_t
      snap_drel_filter = copy.deepcopy(self._drel_filter)
      snap_drel_kalman = copy.deepcopy(self._drel_kalman)

      held_lead = self._update_hyundai_virtual_lead(now, None, None)
      self.gap_reclaim_accel_floor = 0.0
      self.lead_keepup_accel_floor = 0.0
      self.lead_slowdown_accel_ceiling = None
      self._reset_lead_slowdown_ceiling_release_limit()
      self.gap_reclaim_obstacle_push = 0.0
      self.gap_reclaim_projection_scale = 1.0
      self.gap_reclaim_personality_max_accel = 0.0
      self.gap_reclaim_effective_cap = 0.0
      self._raw_reclaim_safety_override_active = False
      self.lead_present_cruise_accel_cap = 0.0
      self._gap_reclaim_blend = 0.0
      self._gap_reclaim_last_t = now
      self._hyundai_reclaim_lead = None
      self._hyundai_reclaim_last_t = None
      if held_lead is not None and getattr(held_lead, 'status', False):
        self.status = True
        self._acc_obstacle_mode = 'lead'
        self._reset_acc_obstacle_candidate()
        self._classifier_demotion_hold_until_t = None
        self.source = str(self._hyundai_virtual_lead_source or 'lead0')
        active_obstacle = self._build_lead_obstacle(held_lead)
        self.acc_source_debug = {
          "active_mode": "lead",
          "best_lead_source": str(self._hyundai_virtual_lead_source),
          "candidate_mode": None,
          "reason": "dropout_hold",
          "used_hysteresis": True,
        }
        return active_obstacle

      # Dropout hold did not engage; _update_hyundai_virtual_lead cleared the
      # stored virtual-lead state. Restore the snapshot and see whether the
      # classifier demotion hold can ride through a one-frame classifier reject
      # with fresh raw-radar corroboration.
      self._hyundai_virtual_lead = snap_virtual_lead
      self._hyundai_virtual_lead_source = snap_virtual_lead_source
      self._hyundai_virtual_lead_last_t = snap_virtual_lead_last_t
      self._hyundai_virtual_lead_reset_reason = snap_virtual_lead_reset_reason
      self._hyundai_virtual_lead_stable_since_t = snap_virtual_lead_stable_since_t
      self._hyundai_virtual_lead_last_drel_error_m = snap_virtual_lead_last_drel_error_m
      self._hyundai_virtual_lead_dropout_until_t = snap_virtual_lead_dropout_until_t
      self._drel_filter = snap_drel_filter
      self._drel_kalman = snap_drel_kalman

      demotion_held_lead, demotion_debug = self._maybe_hold_hyundai_classifier_demotion(now)
      if demotion_held_lead is not None:
        self.status = True
        self._acc_obstacle_mode = 'lead'
        self._reset_acc_obstacle_candidate()
        self.source = str(self._hyundai_virtual_lead_source or 'lead0')
        active_obstacle = self._build_lead_obstacle(demotion_held_lead)
        self.acc_source_debug = {
          "active_mode": "lead",
          "best_lead_source": str(self._hyundai_virtual_lead_source),
          "candidate_mode": None,
          "reason": "classifier_demotion_hold",
          "used_hysteresis": True,
          "classifier_demotion_hold": demotion_debug,
        }
        self.hyundai_virtual_lead_debug = {
          "active": True,
          "source": str(self._hyundai_virtual_lead_source),
          "reset_reason": "classifier_demotion_hold",
        }
        return active_obstacle

      # Genuine loss: neither dropout hold nor demotion hold engaged.
      # Re-clear state since the snapshot restore above was provisional.
      self._reset_hyundai_virtual_lead("no_control_lead")
      self._classifier_demotion_hold_until_t = None
      self._acc_obstacle_mode = 'cruise'
      self._reset_acc_obstacle_candidate()
      self.source = 'cruise'
      self.acc_source_debug = {
        "active_mode": "cruise",
        "best_lead_source": None,
        "candidate_mode": None,
        "reason": "no_control_lead",
        "used_hysteresis": True,
      }
      return cruise_obstacle

    # Lead candidates present — classifier is happy with at least one slot.
    # Clear the opportunistic classifier-demotion hold so it starts fresh the
    # next time the classifier transiently rejects.
    self._classifier_demotion_hold_until_t = None
    best_lead_source, best_lead_obstacle = min(lead_candidates, key=lambda item: item[1][0])
    best_lead = self.control_leads[0] if best_lead_source == 'lead0' else self.control_leads[1]
    # Update close-range lead memory — only when we're actively following a lead
    # (not when cruise already owns the lead, to avoid overriding cruise's taper)
    raw_drel = float(getattr(best_lead, 'dRel', 1e9) or 1e9)
    if raw_drel < CLOSE_LEAD_MEMORY_DREL_M and self._acc_obstacle_mode != 'cruise':
      self._close_lead_last_seen_t = now
      self._close_lead_last_drel = raw_drel
    filtered_lead = self._update_hyundai_virtual_lead(now, best_lead_source, best_lead)
    if filtered_lead is None or not getattr(filtered_lead, 'status', False):
      self._acc_obstacle_mode = 'cruise'
      self._reset_acc_obstacle_candidate()
      self.source = 'cruise'
      self.gap_reclaim_accel_floor = 0.0
      self.lead_keepup_accel_floor = 0.0
      self.lead_slowdown_accel_ceiling = None
      self._reset_lead_slowdown_ceiling_release_limit()
      self.gap_reclaim_obstacle_push = 0.0
      self.gap_reclaim_stabilization_push = 0.0
      self.gap_reclaim_effective_cap = 0.0
      self._raw_reclaim_safety_override_active = False
      self.acc_source_debug = {
        "active_mode": "cruise",
        "best_lead_source": None,
        "candidate_mode": None,
        "reason": "no_filtered_lead",
        "used_hysteresis": True,
      }
      return cruise_obstacle

    filtered_lead_obstacle = self._build_lead_obstacle(filtered_lead)
    raw_metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, best_lead, float(best_lead_obstacle[0]))
    filtered_metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, filtered_lead, float(filtered_lead_obstacle[0]))
    raw_filtered_drel_error_m = float(self._hyundai_virtual_lead_last_drel_error_m)
    steady_follow = self._is_hyundai_settled_follow(best_lead, filtered_lead, raw_metrics, filtered_metrics)
    low_speed_launch_factor = get_low_speed_launch_follow_factor(float(self.x0[1]), filtered_lead, self.current_t_follow)
    launch_gap_buffer_m = LOW_SPEED_LAUNCH_GAP_BUFFER_M * low_speed_launch_factor
    raw_gap_surplus_for_release = raw_metrics["gap_surplus"] - launch_gap_buffer_m
    filtered_gap_surplus_for_release = filtered_metrics["gap_surplus"] - launch_gap_buffer_m
    queue_speed_cap = HYUNDAI_LOW_SPEED_QUEUE_V_EGO_MAX + LOW_SPEED_LAUNCH_QUEUE_SPEED_EXTRA_MPS * low_speed_launch_factor
    queue_vlead_cap = HYUNDAI_LOW_SPEED_QUEUE_VLEAD_MAX + LOW_SPEED_LAUNCH_QUEUE_VLEAD_EXTRA_MPS * low_speed_launch_factor
    queue_pullaway_cap = HYUNDAI_LOW_SPEED_QUEUE_PULLAWAY_MPS_MAX + LOW_SPEED_LAUNCH_QUEUE_PULLAWAY_EXTRA_MPS * low_speed_launch_factor
    low_speed_queue_hold = (
      float(self.x0[1]) <= queue_speed_cap and
      float(getattr(best_lead, 'dRel', 1e9) or 1e9) <= HYUNDAI_LOW_SPEED_QUEUE_DREL_MAX and
      float(getattr(best_lead, 'vLead', self.x0[1]) or self.x0[1]) <= queue_vlead_cap and
      raw_metrics["pullaway_speed"] <= queue_pullaway_cap
    )
    approach_obstacle_delta_m = raw_metrics["obstacle_0"] - float(cruise_obstacle[0])
    approach_reacquire_ttc_threshold_s = self._approach_reacquire_ttc_threshold(float(self.x0[1]))
    raw_ttc_to_headway_s = self._time_to_headway(raw_gap_surplus_for_release, raw_metrics["closing_speed"])
    raw_near_target_for_reacquire = (
      raw_gap_surplus_for_release <= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_NEAR_GAP_SURPLUS_M or
      raw_ttc_to_headway_s <= approach_reacquire_ttc_threshold_s
    )
    approach_reacquire = (
      float(self.x0[1]) >= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_MIN_SPEED and
      raw_metrics["closing_speed"] >= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_CLOSING_MPS and
      raw_metrics["closing_speed"] <= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_CLOSING_MAX_MPS and
      raw_gap_surplus_for_release <= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_GAP_SURPLUS_M and
      approach_obstacle_delta_m <= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_OBSTACLE_MARGIN_M and
      raw_near_target_for_reacquire
    )
    # Stopping-need handoff (midband-slam fix): on this Hyundai AI-lead-stability
    # path the solver sees ONLY the active obstacle, so while cruise owns it a
    # stopped/slowing lead is invisible at any distance. Hand the solver the
    # lead obstacle as soon as the kinematic decel required to stop
    # STOP_DISTANCE short of the lead reaches the live-tunable threshold —
    # OR'd into raw_requires_owner so it can only make the handoff EARLIER,
    # never delay or weaken any existing handoff/release/ceiling/FCW path.
    # Rollback: LeadHandoffStoppingNeedDecelMps2 = 1e9 makes the leg
    # unreachable (exact legacy handoff behavior).
    stopping_need_decel_mps2 = compute_lead_stopping_need_decel(
      float(self.x0[1]), best_lead, self._live_tune_cfg)
    # The threshold scales with ego speed above the reference (midband) speed:
    # at the 6-8 m/s slam band it is the flat base value, while at highway
    # speed the same base would fire for any latched stopped lead out to
    # ~700+ m equivalent, so scaling (a) bounds ghost/false-positive exposure
    # at 20-30 m/s to ranges where physics genuinely demands proportional
    # decel, and (b) keeps the 9-10 m/s calm-stop gap inside the human window
    # (the perception dRel bias grows with the length of the gentle decel
    # phase). Base * max(1, v/ref) with ref=8.0 is a constant-time-headway
    # trigger for the stopped-lead case. RefSpeed=1e9 => flat threshold.
    stopping_need_ref_speed = max(1.0, float(self._live_tune_cfg.lead_handoff_stopping_need_ref_speed_mps))
    stopping_need_threshold_mps2 = (
      float(self._live_tune_cfg.lead_handoff_stopping_need_decel_mps2) *
      max(1.0, float(self.x0[1]) / stopping_need_ref_speed)
    )
    stopping_need_hold = stopping_need_decel_mps2 >= stopping_need_threshold_mps2
    raw_obstacle_requires_owner = (
      raw_metrics["obstacle_0"] <= (float(cruise_obstacle[0]) - HYUNDAI_VIRTUAL_LEAD_RAW_OBSTACLE_MARGIN_M) and
      raw_near_target_for_reacquire
    )
    raw_requires_owner = (
      low_speed_queue_hold or
      approach_reacquire or
      raw_gap_surplus_for_release <= HYUNDAI_VIRTUAL_LEAD_RETAIN_GAP_SURPLUS_M or
      raw_obstacle_requires_owner or
      stopping_need_hold
    )
    release_ready = (
      filtered_gap_surplus_for_release >= HYUNDAI_VIRTUAL_LEAD_RELEASE_GAP_SURPLUS_M and
      filtered_metrics["pullaway_speed"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_PULLAWAY_MPS
    )
    raw_release_ready = (
      raw_gap_surplus_for_release >= HYUNDAI_VIRTUAL_LEAD_RELEASE_RAW_GAP_SURPLUS_M and
      raw_metrics["pullaway_speed"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_RAW_PULLAWAY_MPS
    )
    release_agreement_ok = bool(
      raw_release_ready and
      (not steady_follow or raw_filtered_drel_error_m <= HYUNDAI_VIRTUAL_LEAD_RELEASE_AGREEMENT_MAX_DREL_ERR_M)
    )
    immediate_release = (
      filtered_gap_surplus_for_release >= HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_GAP_SURPLUS_M and
      filtered_metrics["pullaway_speed"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_PULLAWAY_MPS
    )
    raw_immediate_release_ready = (
      raw_gap_surplus_for_release >= HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_RAW_GAP_SURPLUS_M and
      raw_metrics["pullaway_speed"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_RAW_PULLAWAY_MPS
    )
    reacquire_lead = (
      raw_requires_owner or
      filtered_gap_surplus_for_release <= HYUNDAI_VIRTUAL_LEAD_REACQUIRE_GAP_SURPLUS_M
    )
    active_mode = self._acc_obstacle_mode
    reason = "filtered_hold"
    stabilization_push_suppressed = False
    far_closing_cruise_ok = (
      active_mode == 'lead' and
      raw_metrics["closing_speed"] >= HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_CLOSING_MPS and
      raw_gap_surplus_for_release > HYUNDAI_VIRTUAL_LEAD_APPROACH_REACQUIRE_NEAR_GAP_SURPLUS_M and
      raw_ttc_to_headway_s > approach_reacquire_ttc_threshold_s and
      not low_speed_queue_hold and
      not stopping_need_hold
    )
    if raw_requires_owner:
      active_mode = 'lead'
      self._acc_obstacle_mode = 'lead'
      self._reset_acc_obstacle_candidate()
      if low_speed_queue_hold:
        reason = "low_speed_queue_hold"
      elif raw_metrics["gap_surplus"] <= HYUNDAI_VIRTUAL_LEAD_RETAIN_GAP_SURPLUS_M:
        reason = "raw_gap_hold"
      elif approach_reacquire:
        reason = "approach_reacquire"
      elif raw_obstacle_requires_owner:
        reason = "raw_obstacle_hold"
      else:
        reason = "stopping_need_hold"
    elif active_mode == 'lead':
      if far_closing_cruise_ok:
        active_mode = 'cruise'
        self._acc_obstacle_mode = 'cruise'
        self._reset_acc_obstacle_candidate()
        reason = "far_closing_cruise"
      elif immediate_release and raw_immediate_release_ready and release_agreement_ok:
        active_mode = 'cruise'
        self._acc_obstacle_mode = 'cruise'
        self._reset_acc_obstacle_candidate()
        reason = "filtered_pullaway_immediate"
      elif release_ready and release_agreement_ok:
        active_mode = self._advance_acc_obstacle_candidate(
          'cruise', now, HYUNDAI_VIRTUAL_LEAD_RELEASE_DWELL_S, default_mode='lead',
        )
        reason = "filtered_pullaway_dwell"
      elif release_ready:
        self._reset_acc_obstacle_candidate()
        reason = "release_agreement_hold"
      else:
        self._reset_acc_obstacle_candidate()
        reason = "filtered_hold"
    else:
      if reacquire_lead:
        active_mode = 'lead'
        self._acc_obstacle_mode = 'lead'
        self._reset_acc_obstacle_candidate()
        if raw_requires_owner:
          reason = "lead_reacquire_raw"
        else:
          reason = "lead_reacquire_filtered"
      else:
        self._reset_acc_obstacle_candidate()
        reason = "cruise_hold"

    if active_mode == 'lead':
      self._acc_obstacle_mode = 'lead'
      self.source = best_lead_source
      active_obstacle, stabilization_push_suppressed = self._apply_hyundai_gap_reclaim(
        best_lead_obstacle,
        filtered_lead_obstacle,
        best_lead,
        filtered_lead,
        raw_metrics,
        steady_follow,
        now,
        anticipatory_slowdown_enabled=approach_reacquire,
      )
    else:
      self._acc_obstacle_mode = 'cruise'
      self.source = 'cruise'
      self.gap_reclaim_accel_floor = 0.0
      self.lead_keepup_accel_floor = 0.0
      self.lead_slowdown_accel_ceiling = None
      self._reset_lead_slowdown_ceiling_release_limit()
      self.gap_reclaim_obstacle_push = 0.0
      self.gap_reclaim_stabilization_push = 0.0
      self.gap_reclaim_effective_cap = 0.0
      self._raw_reclaim_safety_override_active = False
      active_obstacle = cruise_obstacle

    self.acc_source_debug = {
      "active_mode": str(self._acc_obstacle_mode),
      "best_lead_source": str(best_lead_source),
      "best_lead_obstacle": float(best_lead_obstacle[0]),
      "filtered_lead_obstacle": float(filtered_lead_obstacle[0]),
      "cruise_obstacle": float(cruise_obstacle[0]),
      "raw_gap_surplus_m": float(raw_metrics["gap_surplus"]),
      "filtered_gap_surplus_m": float(filtered_metrics["gap_surplus"]),
      "launch_gap_buffer_m": float(launch_gap_buffer_m),
      "raw_gap_surplus_for_release_m": float(raw_gap_surplus_for_release),
      "filtered_gap_surplus_for_release_m": float(filtered_gap_surplus_for_release),
      "raw_pullaway_mps": float(raw_metrics["pullaway_speed"]),
      "filtered_pullaway_mps": float(filtered_metrics["pullaway_speed"]),
      "raw_closing_mps": float(raw_metrics["closing_speed"]),
      "filtered_closing_mps": float(filtered_metrics["closing_speed"]),
      "approach_obstacle_delta_m": float(approach_obstacle_delta_m),
      "approach_reacquire": bool(approach_reacquire),
      "approach_reacquire_ttc_threshold_s": float(approach_reacquire_ttc_threshold_s),
      "raw_ttc_to_headway_s": float(raw_ttc_to_headway_s),
      "raw_near_target_for_reacquire": bool(raw_near_target_for_reacquire),
      "raw_obstacle_requires_owner": bool(raw_obstacle_requires_owner),
      "far_closing_cruise_ok": bool(far_closing_cruise_ok),
      "low_speed_launch_factor": float(low_speed_launch_factor),
      "low_speed_queue_speed_cap_mps": float(queue_speed_cap),
      "low_speed_queue_vlead_cap_mps": float(queue_vlead_cap),
      "low_speed_queue_pullaway_cap_mps": float(queue_pullaway_cap),
      "raw_filtered_drel_error_m": float(raw_filtered_drel_error_m),
      "steady_follow": bool(steady_follow),
      "filtered_release_ready": bool(release_ready),
      "raw_release_ready": bool(raw_release_ready),
      "release_agreement_ok": bool(release_agreement_ok),
      "low_speed_queue_hold": bool(low_speed_queue_hold),
      "stopping_need_decel_mps2": float(stopping_need_decel_mps2),
      "stopping_need_threshold_mps2": float(stopping_need_threshold_mps2),
      "stopping_need_hold": bool(stopping_need_hold),
      "gap_reclaim_blend": float(self._gap_reclaim_blend),
      "lead_keepup_accel_floor": float(self.lead_keepup_accel_floor),
      "lead_slowdown_accel_ceiling": None if self.lead_slowdown_accel_ceiling is None else float(self.lead_slowdown_accel_ceiling),
      "gap_reclaim_obstacle_push_m": float(self.gap_reclaim_obstacle_push),
      "stabilization_push_m": float(self.gap_reclaim_stabilization_push),
      "stabilization_push_suppressed": bool(stabilization_push_suppressed),
      "gap_reclaim_projection_scale": float(self.gap_reclaim_projection_scale),
      "gap_reclaim_effective_cap": float(self.gap_reclaim_effective_cap),
      "gap_reclaim_personality_max_accel": float(self.gap_reclaim_personality_max_accel),
      "lead_present_cruise_accel_cap": float(self.lead_present_cruise_accel_cap),
      "raw_reclaim_safety_override": bool(self._raw_reclaim_safety_override_active),
      "candidate_mode": self._acc_obstacle_candidate_mode,
      "reason": reason,
      "used_hysteresis": True,
      "lead_handoff_danger_factor": float(self.lead_handoff_danger_factor),
      "lead_handoff_danger_active": bool(self.lead_handoff_danger_debug.get("active", False)),
      "lead_handoff_danger_from": self.lead_handoff_danger_debug.get("from_source"),
      "lead_handoff_danger_to": self.lead_handoff_danger_debug.get("to_source"),
    }
    return active_obstacle

  def _update_cutin_settle_state(self, now: float, v_ego: float, lead_role_debug: dict[str, object]) -> None:
    slot_key, lead = self._get_best_control_lead()
    if slot_key is None or lead is None or not bool(getattr(lead, 'status', False)):
      self._virtual_cutin_event_t = None
      self._prev_virtual_lead_role = LeadRoleClassifier.INVALID
      self._prev_virtual_lead_control_active = False
      return

    raw = lead_role_debug.get("raw", {}).get(slot_key, {})
    path_abs_m = abs(float(raw.get("dPath", 0.0) or 0.0))
    toward_center_mps = float(lead_role_debug.get("toward_center_mps", {}).get(slot_key, 0.0) or 0.0)
    cutin_promoted = bool(lead_role_debug.get("cutin_promoted", {}).get(slot_key, False))
    current_role = str(lead_role_debug.get("roles", {}).get(slot_key, LeadRoleClassifier.INVALID))
    prev_role = LeadRoleClassifier.INVALID if self._hyundai_virtual_lead_identity_changed else self._prev_virtual_lead_role
    prev_control_active = False if self._hyundai_virtual_lead_identity_changed else self._prev_virtual_lead_control_active

    if should_start_cutin_settle_event(
      prev_role,
      prev_control_active,
      current_role,
      lead,
      cutin_promoted=cutin_promoted,
      toward_center_mps=toward_center_mps,
      path_abs_m=path_abs_m,
      v_ego=v_ego,
    ):
      self._virtual_cutin_event_t = now

    if self._virtual_cutin_event_t is not None and (now - float(self._virtual_cutin_event_t)) > self._live_tune_cfg.cutin_settle_duration_s:
      self._virtual_cutin_event_t = None

    self._prev_virtual_lead_role = current_role
    self._prev_virtual_lead_control_active = bool(getattr(lead, 'status', False))

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
    jerk_factor = get_jerk_factor(personality)
    if self.mode == 'acc':
      a_change_cost = self._live_a_change_cost if prev_accel_constraint else 0
      cost_weights = [self._live_obstacle_cost, X_EGO_COST, V_EGO_COST, self._live_a_ego_cost, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    elif self.mode == 'blended':
      a_change_cost = 40.0 if prev_accel_constraint else 0
      cost_weights = [0., 0.1, 0.2, 5.0, a_change_cost, 1.0]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner cost set')
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(N+1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def get_gap_reclaim_floor(self) -> float:
    if self.mode != 'acc' or self.last_v_cruise_clipped is None or len(self.last_v_cruise_clipped) < 2:
      return 0.0

    v_ego = float(self.x0[1])
    if float(np.max(self.last_v_cruise_clipped)) <= v_ego + 0.05:
      return 0.0

    if self._hyundai_ai_lead_stability_enabled:
      if self.source not in ('lead0', 'lead1') or self._hyundai_reclaim_lead is None:
        return 0.0
      return float(self.gap_reclaim_accel_floor)

    personality_max_accel = self._get_gap_reclaim_personality_max_accel(v_ego)
    return float(max(
      get_gap_reclaim_accel_floor(
        v_ego,
        lead,
        self.current_t_follow,
        self._live_tune_cfg,
        personality_max_accel=personality_max_accel,
      )
      for lead in self.control_leads
    ))

  def get_lead_keepup_floor(self) -> float:
    if self.mode != 'acc' or self.last_v_cruise_clipped is None or len(self.last_v_cruise_clipped) < 2:
      return 0.0

    v_ego = float(self.x0[1])
    if float(np.max(self.last_v_cruise_clipped)) <= v_ego + 0.05:
      return 0.0

    if self._hyundai_ai_lead_stability_enabled:
      if self.source not in ('lead0', 'lead1'):
        return 0.0
      return float(self.lead_keepup_accel_floor)

    personality_max_accel = self._get_gap_reclaim_personality_max_accel(v_ego)
    return float(max(
      get_lead_keepup_accel_floor(
        v_ego,
        lead,
        self.current_t_follow,
        self._live_tune_cfg,
        personality_max_accel=personality_max_accel,
      )
      for lead in self.control_leads
    ))

  def get_lead_slowdown_ceiling(self) -> float | None:
    if self.mode != 'acc':
      return None

    v_ego = float(self.x0[1])
    if self._hyundai_ai_lead_stability_enabled:
      if self.source not in ('lead0', 'lead1'):
        return None
      return self.lead_slowdown_accel_ceiling

    ceilings = [
      get_lead_slowdown_accel_ceiling(
        v_ego,
        lead,
        self.current_t_follow,
        self._live_tune_cfg,
        min_accel=ACCEL_MIN,
        max_accel=ACCEL_MAX,
      )
      for lead in self.control_leads
    ]
    active_ceilings = [float(ceiling) for ceiling in ceilings if ceiling is not None]
    return min(active_ceilings) if active_ceilings else None

  def get_cutin_settle_floor(self, now: float) -> float:
    self.cutin_settle_active = False
    self.cutin_settle_accel_floor = 0.0
    self.cutin_settle_debug = {}

    if self.mode != 'acc' or self.source not in ('lead0', 'lead1'):
      return 0.0

    slot_idx = 0 if self.source == 'lead0' else 1
    slot_key = f"lead{slot_idx}"
    lead = self.control_leads[slot_idx]
    event_t = self._virtual_cutin_event_t
    if event_t is None:
      return 0.0

    age_s = max(0.0, float(now) - float(event_t))
    floor = get_cutin_settle_accel_floor(
      float(self.x0[1]),
      lead,
      self.current_t_follow,
      age_s,
      self._live_tune_cfg,
    )
    if floor is None:
      return 0.0

    self.cutin_settle_active = True
    self.cutin_settle_accel_floor = float(floor)
    self.cutin_settle_debug = {
      "slot": slot_key,
      "age_s": float(age_s),
      "floor": float(floor),
      "dRel": float(getattr(lead, 'dRel', 0.0) or 0.0),
      "vLead": float(getattr(lead, 'vLead', 0.0) or 0.0),
      "aLeadK": float(getattr(lead, 'aLeadK', 0.0) or 0.0),
    }
    return float(floor)

  def get_cruise_response_model(self, v_ego: float, *, actuation_delay_s: float = 0.0,
                                planner_accel_limits: tuple[float, float] | None = None) -> CruiseResponseModel:
    if self.vibe_controller.is_accel_enabled():
      accel_limits = self.vibe_controller.get_accel_limits(v_ego)
      if accel_limits is not None:
        min_accel = float(accel_limits[0])
      else:
        min_accel = CRUISE_MIN_ACCEL
    else:
      min_accel = CRUISE_MIN_ACCEL
    if planner_accel_limits is not None:
      planner_accel_min = float(planner_accel_limits[0])
      planner_accel_max = float(planner_accel_limits[1])
    else:
      planner_accel_min = ACCEL_MIN
      planner_accel_max = ACCEL_MAX
    cruise_max_accel = min(CRUISE_MAX_ACCEL, planner_accel_max)
    return build_cruise_response_model(
      min_accel_mps2=min_accel,
      max_accel_mps2=cruise_max_accel,
      comfort_brake_mps2=COMFORT_BRAKE,
      actuation_delay_s=actuation_delay_s,
      planner_output_min_accel_mps2=planner_accel_min,
      planner_output_max_accel_mps2=planner_accel_max,
    )

  def update(self, radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard):
    v_ego = self.x0[1]
    now = float(self._time_fn())
    self._refresh_live_tune(now)
    prev_source = self.source

    # Get following distance
    if self.vibe_controller.is_follow_enabled():
      desired_headway = self.vibe_controller.get_follow_distance_multiplier(v_ego)
      if desired_headway is not None:
        # Compensate for STOP_DISTANCE so the user's headway setting matches
        # displayed headway (dRel/v_ego) at steady state.
        t_follow = max(0.5, float(desired_headway) - STOP_DISTANCE / max(float(v_ego), 1.0))
      else:
        t_follow = get_T_FOLLOW(personality)
    else:
      t_follow = get_T_FOLLOW(personality)
    self.current_t_follow = float(t_follow)

    stabilized_lead0, stabilized_lead1 = self._stabilize_raw_leads(
      getattr(radarstate, 'leadOne', None),
      getattr(radarstate, 'leadTwo', None),
      now,
    )

    raw_control_lead0, raw_control_lead1, lead_role_debug = self.lead_role_classifier.classify(
      v_ego, stabilized_lead0, stabilized_lead1, now=now,
    )
    self.control_leads, lead_role_debug = self._stabilize_control_leads(
      stabilized_lead0, stabilized_lead1, raw_control_lead0, raw_control_lead1, lead_role_debug,
    )
    # Snapshot stabilized leads so _maybe_hold_hyundai_classifier_demotion can
    # corroborate against the same view the rest of update() uses. Keeps
    # dwell/phantom decisions internally consistent across update().
    self._last_raw_radar_leads = (stabilized_lead0, stabilized_lead1)
    control_lead0, control_lead1 = self.control_leads
    self.lead_role_debug = lead_role_debug
    self.status = control_lead0.status or control_lead1.status
    self._update_cutin_settle_state(now, v_ego, lead_role_debug)
    lead_acquire_debug = self._update_lead_acquire_state(now)

    response_model = self.get_cruise_response_model(v_ego)
    self.last_cruise_response_model = response_model

    lead_xv_0 = self.process_lead(control_lead0)
    lead_xv_1 = self.process_lead(control_lead1)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])
    lead_0_preview, lead_0_preview_debug = compute_lead_approach_preview(
      v_ego,
      control_lead0,
      t_follow,
      self._live_tune_cfg,
      acquire_window_active=bool(lead_acquire_debug["lead0"]["active"]),
    )
    lead_1_preview, lead_1_preview_debug = compute_lead_approach_preview(
      v_ego,
      control_lead1,
      t_follow,
      self._live_tune_cfg,
      acquire_window_active=bool(lead_acquire_debug["lead1"]["active"]),
    )
    lead_0_preview_debug["acquire"] = lead_acquire_debug["lead0"]
    lead_1_preview_debug["acquire"] = lead_acquire_debug["lead1"]
    lead_0_obstacle = apply_lead_approach_preview(lead_0_obstacle, lead_0_preview)
    lead_1_obstacle = apply_lead_approach_preview(lead_1_obstacle, lead_1_preview)
    self.lead_approach_preview = (lead_0_preview, lead_1_preview)
    self.lead_approach_preview_debug = {
      "lead0": lead_0_preview_debug,
      "lead1": lead_1_preview_debug,
    }
    adjacent_awareness_preview_obstacle, adjacent_awareness_preview_debug = self._compute_adjacent_awareness_preview_obstacle(
      {"lead0": stabilized_lead0, "lead1": stabilized_lead1},
      lead_role_debug,
      float(v_ego),
      t_follow,
    )
    self.adjacent_awareness_preview_debug = adjacent_awareness_preview_debug

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX

    # Update in ACC mode or ACC/e2e blend
    if self.mode == 'acc':
      lead_for_cruise_cap = None
      lead_for_cruise_cap_source = None
      lead_for_cruise_obstacle = 1e9
      if control_lead0.status:
        lead_for_cruise_cap = control_lead0
        lead_for_cruise_cap_source = "lead0_control"
        lead_for_cruise_obstacle = float(lead_0_obstacle[0])
      if control_lead1.status and float(lead_1_obstacle[0]) < lead_for_cruise_obstacle:
        lead_for_cruise_cap = control_lead1
        lead_for_cruise_cap_source = "lead1_control"
        lead_for_cruise_obstacle = float(lead_1_obstacle[0])
      if lead_for_cruise_cap is None:
        for raw_source, raw_lead in (("lead0_raw_path", stabilized_lead0), ("lead1_raw_path", stabilized_lead1)):
          if not self._is_plausible_cruise_cap_raw_lead(raw_lead):
            continue
          raw_drel = self._lead_attr(raw_lead, "dRel", 1e9)
          if raw_drel < lead_for_cruise_obstacle:
            lead_for_cruise_cap = raw_lead
            lead_for_cruise_cap_source = raw_source
            lead_for_cruise_obstacle = raw_drel

      personality_max_accel = self._get_gap_reclaim_personality_max_accel(float(v_ego))
      lead_present_cruise_cap = get_lead_present_cruise_accel_cap(
        float(v_ego),
        lead_for_cruise_cap,
        self.current_t_follow,
        self._live_tune_cfg,
        personality_max_accel=personality_max_accel,
      )
      self.lead_present_cruise_accel_cap = float(lead_present_cruise_cap or 0.0)
      planner_accel_limits = None if lead_present_cruise_cap is None else (ACCEL_MIN, float(lead_present_cruise_cap))
      cruise_owned_accel_cap = None
      self.cruise_owned_accel_cap = None
      response_model = self.get_cruise_response_model(v_ego, planner_accel_limits=planner_accel_limits)
      self.last_cruise_response_model = response_model

      # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
      # when the leads are no factor.
      v_lower, v_upper, v_cruise_clipped = clip_cruise_speed_profile(
        v_ego=v_ego,
        v_cruise=v_cruise,
        t_idxs=T_IDXS,
        response_model=response_model,
      )
      self.last_v_lower = v_lower
      self.last_v_upper = v_upper
      self.last_v_cruise_clipped = v_cruise_clipped
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow)
      active_obstacle = self._select_acc_obstacle(lead_0_obstacle, lead_1_obstacle, cruise_obstacle, now)
      # Enforce v_cruise set speed as a hard ceiling even while following a lead.
      # The Hyundai-stabilized lead path in _select_acc_obstacle returns a lead-only
      # obstacle, so without this clamp the MPC will follow a lead that exceeds the
      # driver's set speed. The non-stabilized path already does np.min with
      # cruise_obstacle; this line restores the same invariant.
      active_obstacle = np.minimum(active_obstacle, cruise_obstacle)
      adjacent_preview_applied = False
      adjacent_preview_application_blend = 0.0
      if (adjacent_awareness_preview_obstacle is not None and
          float(adjacent_awareness_preview_obstacle[0]) < float(active_obstacle[0])):
        adjacent_preview_application_blend = float(np.clip(
          self.adjacent_awareness_preview_debug.get("application_blend", 0.0) or 0.0,
          0.0,
          1.0,
        ))
        if adjacent_preview_application_blend > 0.0:
          preview_target = np.minimum(active_obstacle, adjacent_awareness_preview_obstacle)
          active_obstacle = active_obstacle - (active_obstacle - preview_target) * adjacent_preview_application_blend
          adjacent_preview_applied = True
      self.adjacent_awareness_preview_debug = {
        **self.adjacent_awareness_preview_debug,
        "applied": bool(adjacent_preview_applied),
        "applied_blend": float(adjacent_preview_application_blend),
        "active_obstacle_m": float(active_obstacle[0]),
      }
      self.lead_handoff_danger_factor = float(LEAD_DANGER_FACTOR)
      self.lead_handoff_danger_debug = {"active": False, "danger_factor": float(LEAD_DANGER_FACTOR)}
      self._lead_handoff_until_t = None
      self._lead_handoff_from_source = None
      self._lead_handoff_to_source = None
      if prev_source in ('lead0', 'lead1') and self.source == 'cruise':
        self._lead_to_cruise_transition_t = now
        self._lead_to_cruise_transition_source = prev_source
      elif self.source != 'cruise':
        self._lead_to_cruise_transition_t = None
        self._lead_to_cruise_transition_source = None
        # Clear close-lead memory once we're stably on a lead — no longer needed
        if self._close_lead_last_seen_t is not None:
          self._close_lead_last_seen_t = None
          self._close_lead_last_drel = 1e9

      # Close-range lead safety memory: if a lead was seen nearby recently,
      # hard-cap cruise accel even if the model is currently flickering.
      # This prevents accelerating toward a car the model saw 0.5s ago.
      close_lead_memory_active = (
        self._close_lead_last_seen_t is not None and
        (now - float(self._close_lead_last_seen_t)) < CLOSE_LEAD_MEMORY_HOLD_S
      )
      if self.source == 'cruise' and lead_present_cruise_cap is not None:
        cruise_owned_accel_cap = float(lead_present_cruise_cap)

      transition_accel_cap = self._get_lead_to_cruise_transition_accel_cap(now, float(v_ego), personality_max_accel)
      # Close-lead memory: if the normal transition cap expired but a lead was
      # recently seen nearby, apply the safety cap as a backstop.
      if self.source == 'cruise' and close_lead_memory_active and transition_accel_cap is None:
        transition_accel_cap = CLOSE_LEAD_MEMORY_ACCEL_CAP
      if self.source == 'cruise' and transition_accel_cap is not None:
        capped_max_accel = float(transition_accel_cap)
        if lead_present_cruise_cap is not None:
          capped_max_accel = min(capped_max_accel, float(lead_present_cruise_cap))
        cruise_owned_accel_cap = float(capped_max_accel)
        response_model = self.get_cruise_response_model(v_ego, planner_accel_limits=(ACCEL_MIN, capped_max_accel))
        self.last_cruise_response_model = response_model
        v_lower, v_upper, v_cruise_clipped = clip_cruise_speed_profile(
          v_ego=v_ego,
          v_cruise=v_cruise,
          t_idxs=T_IDXS,
          response_model=response_model,
        )
        self.last_v_lower = v_lower
        self.last_v_upper = v_upper
        self.last_v_cruise_clipped = v_cruise_clipped
        cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow)
        active_obstacle = cruise_obstacle
      elif (self._lead_to_cruise_transition_t is not None and
            (now - float(self._lead_to_cruise_transition_t)) >= HYUNDAI_LEAD_TO_CRUISE_TRANSITION_RAMP_S):
        self._lead_to_cruise_transition_t = None
        self._lead_to_cruise_transition_source = None

      if self.mode == 'acc' and self.source == 'cruise' and self.acc_source_debug:
        transition_active = bool(self._lead_to_cruise_transition_t is not None and transition_accel_cap is not None)
        transition_elapsed_s = None if self._lead_to_cruise_transition_t is None else max(0.0, now - float(self._lead_to_cruise_transition_t))
        self.acc_source_debug["source_transition_active"] = transition_active
        self.acc_source_debug["source_transition_from"] = self._lead_to_cruise_transition_source
        self.acc_source_debug["source_transition_elapsed_s"] = transition_elapsed_s
        self.acc_source_debug["close_lead_memory_active"] = bool(close_lead_memory_active)
        self.acc_source_debug["close_lead_memory_drel"] = float(self._close_lead_last_drel) if close_lead_memory_active else None
        self.acc_source_debug["source_transition_accel_cap"] = None if transition_accel_cap is None else float(transition_accel_cap)
      elif self.acc_source_debug:
        self.acc_source_debug["source_transition_active"] = False
        self.acc_source_debug["source_transition_from"] = None
        self.acc_source_debug["source_transition_elapsed_s"] = None
        self.acc_source_debug["source_transition_accel_cap"] = None
      if self.source == 'cruise' and cruise_owned_accel_cap is not None:
        self.cruise_owned_accel_cap = float(cruise_owned_accel_cap)
        self.params[:,1] = np.minimum(self.params[:,1], float(cruise_owned_accel_cap))
      else:
        self.cruise_owned_accel_cap = None

      if self.acc_source_debug:
        self.acc_source_debug["adjacent_awareness_preview_active"] = bool(self.adjacent_awareness_preview_debug.get("active", False))
        self.acc_source_debug["adjacent_awareness_preview_applied"] = bool(self.adjacent_awareness_preview_debug.get("applied", False))
        self.acc_source_debug["adjacent_awareness_preview_slot"] = self.adjacent_awareness_preview_debug.get("slot")
        self.acc_source_debug["cruise_owned_accel_cap"] = None if cruise_owned_accel_cap is None else float(cruise_owned_accel_cap)
        self.acc_source_debug["lead_present_cruise_accel_cap_source"] = lead_for_cruise_cap_source
        self.acc_source_debug["lead_present_cruise_accel_cap_drel_m"] = (
          None if lead_for_cruise_cap is None else self._lead_attr(lead_for_cruise_cap, "dRel", 0.0)
        )

      self.gap_reclaim_accel_floor = self.get_gap_reclaim_floor()
      self.lead_keepup_accel_floor = self.get_lead_keepup_floor()
      self.lead_slowdown_accel_ceiling = self.get_lead_slowdown_ceiling()
      self.cutin_settle_accel_floor = self.get_cutin_settle_floor(now)
      self.params[:,5] = LEAD_DANGER_FACTOR

      # These are not used in ACC mode
      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

    elif self.mode == 'blended':
      self._lead_to_cruise_transition_t = None
      self._lead_to_cruise_transition_source = None
      self.last_v_lower = None
      self.last_v_upper = None
      self.last_v_cruise_clipped = None
      self.params[:,5] = 1.0
      self.hyundai_virtual_lead_debug = {"active": False, "reset_reason": "non_acc_mode"}

      x_obstacles = np.column_stack([lead_0_obstacle,
                                     lead_1_obstacle])
      cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0]
      xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1])
      x = np.cumsum(np.insert(xforward, 0, x[0]))

      x_and_cruise = np.column_stack([x, cruise_target])
      x = np.min(x_and_cruise, axis=1)

      self.source = 'e2e' if x_and_cruise[1,0] < x_and_cruise[1,1] else 'cruise'
      self.acc_source_debug = {
        "active_mode": str(self.source),
        "delta_m": None,
        "candidate_mode": None,
        "used_hysteresis": False,
      }
      self.gap_reclaim_accel_floor = 0.0
      self.lead_keepup_accel_floor = 0.0
      self.lead_slowdown_accel_ceiling = None
      self._reset_lead_slowdown_ceiling_release_limit()
      self.gap_reclaim_obstacle_push = 0.0
      self.gap_reclaim_stabilization_push = 0.0
      self.gap_reclaim_projection_scale = 1.0
      self.gap_reclaim_personality_max_accel = 0.0
      self.gap_reclaim_effective_cap = 0.0
      self.lead_present_cruise_accel_cap = 0.0
      self.cruise_owned_accel_cap = None
      self._gap_reclaim_blend = 0.0
      self._gap_reclaim_last_t = now
      self.cutin_settle_active = False
      self.cutin_settle_accel_floor = 0.0
      self.cutin_settle_debug = {}
      self.lead_handoff_danger_factor = float(LEAD_DANGER_FACTOR)
      self.lead_handoff_danger_debug = {"active": False, "danger_factor": float(LEAD_DANGER_FACTOR)}
      self.adjacent_awareness_preview_debug = {"active": False, "applied": False}
      self._lead_handoff_until_t = None
      self._lead_handoff_from_source = None
      self._lead_handoff_to_source = None

    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

    self.yref[:,1] = x
    self.yref[:,2] = v
    self.yref[:,3] = a
    self.yref[:,5] = j
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    if self.mode == 'acc':
      self.params[:,2] = active_obstacle
      speed_constraint_ub = np.full(CONSTR_DIM, 1e4)
      speed_constraint_ub[0] = max(float(v_cruise), float(v_ego)) + 0.05
    else:
      self.params[:,2] = np.min(x_obstacles, axis=1)
      speed_constraint_ub = np.full(CONSTR_DIM, 1e4)
    self.params[:,3] = np.copy(self.prev_a)
    self.params[:,4] = t_follow
    for i in range(N):
      self.solver.constraints_set(i, "uh", speed_constraint_ub)

    self.run()

    fcw_lead_xv = None
    fcw_model_prob = 0.0
    fcw_suppressed = False
    if self.mode == 'acc':
      if self.source == 'lead0' and control_lead0.status:
        fcw_lead_xv = lead_xv_0
        fcw_model_prob = float(control_lead0.modelProb)
        fcw_suppressed = bool(getattr(control_lead0, 'fcwSuppressed', False))
      elif self.source == 'lead1' and control_lead1.status:
        fcw_lead_xv = lead_xv_1
        fcw_model_prob = float(control_lead1.modelProb)
        fcw_suppressed = bool(getattr(control_lead1, 'fcwSuppressed', False))
    else:
      if self.source == 'lead0' and control_lead0.status:
        fcw_lead_xv = lead_xv_0
        fcw_model_prob = float(control_lead0.modelProb)
        fcw_suppressed = bool(getattr(control_lead0, 'fcwSuppressed', False))
      elif self.source == 'lead1' and control_lead1.status:
        fcw_lead_xv = lead_xv_1
        fcw_model_prob = float(control_lead1.modelProb)
        fcw_suppressed = bool(getattr(control_lead1, 'fcwSuppressed', False))

    # fcwSuppressed is the producing tracker's veto: the raw measurement stream
    # does NOT corroborate the filtered closeness (phantom-collapsed lead), so a
    # predicted crash against it must not accrue toward FCW / the Hyundai
    # emergency-braking override. Default False everywhere it is not computed,
    # which preserves legacy behavior for radar tracks and fabricated leads; a
    # corroborated genuine threat is never suppressed (raw <= filtered + tol on
    # a real collision course), so genuine FCW timing is frame-identical.
    if (fcw_lead_xv is not None and
            np.any(fcw_lead_xv[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            fcw_model_prob > 0.9 and
            not fcw_suppressed):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

    # Check if it got within lead comfort range
    # TODO This should be done cleaner
    if self.mode == 'blended':
      if any((lead_0_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow))- self.x_sol[:,0] < 0.0):
        self.source = 'lead0'
      if any((lead_1_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow))- self.x_sol[:,0] < 0.0) and \
         (lead_1_obstacle[0] - lead_0_obstacle[0]):
        self.source = 'lead1'

    if lead_role_debug.get("debug_log_enabled", False):
      raw = lead_role_debug.get("raw", {})
      has_raw_lead = bool(raw.get("lead0", {}).get("status")) or bool(raw.get("lead1", {}).get("status"))
      log_period_s = 0.25 if lead_role_debug.get("duplicate_pair", False) else 1.0
      should_log = has_raw_lead and (now - self.last_lead_role_log_t) >= log_period_s
      if should_log:
        self.last_lead_role_log_t = now
        dbg_payload = {
          "vEgo": float(v_ego),
          "source": str(self.source),
          "gate_active": bool(lead_role_debug.get("gate_active", False)),
          "low_speed_bypass": bool(lead_role_debug.get("low_speed_bypass", False)),
          "roles": lead_role_debug.get("roles", {}),
          "reasons": lead_role_debug.get("reasons", {}),
          "cutin_promoted": lead_role_debug.get("cutin_promoted", {}),
          "toward_center_mps": lead_role_debug.get("toward_center_mps", {}),
          "duplicate_pair": bool(lead_role_debug.get("duplicate_pair", False)),
          "dropped_slot": lead_role_debug.get("dropped_slot", None),
          "control_status": lead_role_debug.get("control_status", {}),
          "cutin_settle": {
            "active": bool(self.cutin_settle_active),
            "floor": float(self.cutin_settle_accel_floor),
            **self.cutin_settle_debug,
          },
          "lead_handoff_danger": self.lead_handoff_danger_debug,
          "lead_preview": self.lead_approach_preview_debug,
          "source_hysteresis": self.acc_source_debug,
          "filtered_virtual_lead": self.hyundai_virtual_lead_debug,
          "virtual_duplicate": lead_role_debug.get("virtual_duplicate", {"active": False}),
          "raw_duplicate_model": lead_role_debug.get("raw_duplicate_model", {}),
          "raw": lead_role_debug.get("raw", {}),
          "awareness": lead_role_debug.get("awareness", []),
          "lead_stability": self.lead_stability_debug,
        }
        cloudlog.info(f"LEADROLEDBG {json.dumps(dbg_payload, separators=(',', ':'), sort_keys=True)}")

  def run(self):
    # t0 = time.monotonic()
    # reset = 0
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])
    self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
    self.time_linearization = float(self.solver.get_stats('time_lin')[0])
    self.time_integrator = float(self.solver.get_stats('time_sim')[0])

    # qp_iter = self.solver.get_stats('statistics')[-1][-1] # SQP_RTI specific
    # print(f"long_mpc timings: tot {self.solve_time:.2e}, qp {self.time_qp_solution:.2e}, lin {self.time_linearization:.2e}, \
    # integrator {self.time_integrator:.2e}, qp_iter {qp_iter}")
    # res = self.solver.get_residuals()
    # print(f"long_mpc residuals: {res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}")
    # self.solver.print_statistics()

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.prev_a = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    # Keep solver warning throttling on a real wall clock so tests can inject
    # self._time_fn without also accelerating unrelated bookkeeping.
    t = REAL_MONOTONIC()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()
      # reset = 1
    # print(f"long_mpc timings: total internal {self.solve_time:.2e}, external: {(time.monotonic() - t0):.2e} qp {self.time_qp_solution:.2e}, \
    # lin {self.time_linearization:.2e} qp_iter {qp_iter}, reset {reset}")


if __name__ == "__main__":
  from openpilot.third_party.acados.acados_template import AcadosOcpSolver

  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
