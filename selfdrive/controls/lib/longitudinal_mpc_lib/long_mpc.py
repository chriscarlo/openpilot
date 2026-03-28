#!/usr/bin/env python3
import copy
import json
import os
import time
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

if __name__ == '__main__':  # generating code
  from openpilot.third_party.acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.
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
LEAD_APPROACH_PREVIEW_TIME_BP = [0.0, 2.0, 5.0, 10.0]
LEAD_APPROACH_PREVIEW_TIME_V = [0.0, 0.15, 0.50, 0.95]
LEAD_APPROACH_PREVIEW_DECAY_TAU = 1.75
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
HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S = 0.45
HYUNDAI_VIRTUAL_LEAD_MODEL_PROB_TAU_S = 0.60
HYUNDAI_VIRTUAL_LEAD_RESET_DREL_M = 8.0
HYUNDAI_VIRTUAL_LEAD_RESET_DPATH_M = 1.75
HYUNDAI_VIRTUAL_LEAD_RETAIN_GAP_SURPLUS_M = 2.5
HYUNDAI_VIRTUAL_LEAD_REACQUIRE_GAP_SURPLUS_M = 3.0
HYUNDAI_VIRTUAL_LEAD_RELEASE_GAP_SURPLUS_M = 4.0
HYUNDAI_VIRTUAL_LEAD_RELEASE_PULLAWAY_MPS = 0.35
HYUNDAI_VIRTUAL_LEAD_RELEASE_DWELL_S = 1.00
HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_GAP_SURPLUS_M = 7.0
HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_PULLAWAY_MPS = 1.00
HYUNDAI_VIRTUAL_LEAD_RAW_OBSTACLE_MARGIN_M = 1.00
HYUNDAI_LOW_SPEED_QUEUE_V_EGO_MAX = 6.0
HYUNDAI_LOW_SPEED_QUEUE_DREL_MAX = 22.0
HYUNDAI_LOW_SPEED_QUEUE_VLEAD_MAX = 8.0
HYUNDAI_LOW_SPEED_QUEUE_PULLAWAY_MPS_MAX = 2.5


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


def get_lead_approach_preview_buffer(v_ego, lead, t_follow,
                                     tuning: LeadResponseTuningConfig | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  if lead is None or not getattr(lead, 'status', False) or v_ego < LEAD_APPROACH_PREVIEW_MIN_SPEED:
    return 0.0

  v_lead = max(0.0, float(getattr(lead, 'vLead', v_ego) or v_ego))
  closing_speed = max(0.0, float(v_ego) - v_lead)
  if closing_speed <= 0.5:
    return 0.0

  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  headway_gap = get_headway_follow_distance(float(v_ego), t_follow)
  gap_surplus = d_rel - headway_gap
  if gap_surplus <= tuning.lead_preview_gap_min_m:
    return 0.0

  preview_time = float(np.interp(closing_speed, LEAD_APPROACH_PREVIEW_TIME_BP, LEAD_APPROACH_PREVIEW_TIME_V))
  preview_buffer = closing_speed * preview_time
  lead_accel = max(0.0, float(getattr(lead, 'aLeadK', 0.0) or 0.0))
  accel_scale = float(np.interp(lead_accel, GAP_RECLAIM_LEAD_ACCEL_BP, [1.0, 0.75, 0.5]))
  max_buffer = min(tuning.lead_preview_max_buffer_m, gap_surplus * 0.7)
  preview_buffer *= accel_scale * tuning.lead_preview_strength
  return float(np.clip(preview_buffer, 0.0, max_buffer))


def apply_lead_approach_preview(lead_obstacle, preview_buffer_m):
  if preview_buffer_m <= 0.0:
    return lead_obstacle

  decay = np.exp(-T_IDXS / LEAD_APPROACH_PREVIEW_DECAY_TAU)
  return np.maximum(lead_obstacle - preview_buffer_m * decay, 0.0)


def get_gap_reclaim_effective_cap(v_ego, lead, t_follow,
                                  tuning: LeadResponseTuningConfig | None = None,
                                  personality_max_accel: float | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  comfort_cap = max(float(tuning.gap_reclaim_max_accel), 1e-3)
  if (lead is None or not getattr(lead, 'status', False) or v_ego < GAP_RECLAIM_MIN_SPEED or
      personality_max_accel is None):
    return comfort_cap

  personality_cap = max(comfort_cap, float(personality_max_accel))
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


def get_gap_reclaim_accel_floor(v_ego, lead, t_follow,
                                tuning: LeadResponseTuningConfig | None = None,
                                personality_max_accel: float | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  if lead is None or not getattr(lead, 'status', False) or v_ego < GAP_RECLAIM_MIN_SPEED:
    return 0.0

  v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
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
  floor = intent_fraction * effective_cap
  return float(np.clip(floor, 0.0, effective_cap))


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
  return -float(np.clip(floor_mag, 0.0, tuning.cutin_settle_max_decel))


def gen_long_model():
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

  def __init__(self, mode='acc', dt=DT_MDL, CP=None):
    self.mode = mode
    self.dt = dt
    self._hyundai_ai_lead_stability_enabled = bool(getattr(CP, 'brand', None) == 'hyundai')
    self.use_upstream_gap_reclaim = self._hyundai_ai_lead_stability_enabled
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self._live_tune_params = Params()
    self._last_live_tune_refresh_t = 0.0
    self._live_tune_cfg = LeadResponseTuningConfig.defaults()
    self.reset()
    self.source = SOURCES[2]
    self.vibe_controller = VibePersonalityController()
    self.lead_role_classifier = LeadRoleClassifier()
    self.lead_role_debug = {}
    self.last_lead_role_log_t = 0.0

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
    self.gap_reclaim_accel_floor = 0.0
    self.gap_reclaim_obstacle_push = 0.0
    self.gap_reclaim_projection_scale = 1.0
    self.gap_reclaim_personality_max_accel = 0.0
    self.gap_reclaim_effective_cap = 0.0
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
    self.hyundai_virtual_lead_debug = {"active": False}
    self._hyundai_reclaim_lead = None
    self._hyundai_reclaim_last_t = None
    self._acc_obstacle_mode = 'cruise'
    self._acc_obstacle_candidate_mode = None
    self._acc_obstacle_candidate_t = None
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

  @staticmethod
  def _lead_follow_metrics(v_ego: float, t_follow: float, lead, obstacle_0: float | None = None) -> dict[str, float]:
    if lead is None or not getattr(lead, 'status', False):
      return {
        "gap_surplus": 1e9,
        "pullaway_speed": 0.0,
        "closing_speed": 0.0,
        "obstacle_0": float(obstacle_0 if obstacle_0 is not None else 1e9),
      }

    gap_surplus = float(getattr(lead, 'dRel', 0.0) or 0.0) - get_headway_follow_distance(v_ego, t_follow)
    v_lead = float(getattr(lead, 'vLead', v_ego) or v_ego)
    v_rel = float(getattr(lead, 'vRel', 0.0) or 0.0)
    return {
      "gap_surplus": float(gap_surplus),
      "pullaway_speed": float(max(0.0, v_lead - v_ego, v_rel)),
      "closing_speed": float(max(0.0, v_ego - v_lead, -v_rel)),
      "obstacle_0": float(obstacle_0 if obstacle_0 is not None else 1e9),
    }

  @staticmethod
  def _ema_alpha(dt_s: float, tau_s: float) -> float:
    if dt_s <= 0.0 or tau_s <= 0.0:
      return 1.0
    return float(np.clip(1.0 - np.exp(-dt_s / max(tau_s, 1e-3)), 0.0, 1.0))

  def _filter_metric(self, prev: float, current: float, dt_s: float, *,
                     danger_if_lower: bool, fast_tau_s: float = HYUNDAI_VIRTUAL_LEAD_FAST_TAU_S,
                     slow_tau_s: float = HYUNDAI_VIRTUAL_LEAD_SLOW_TAU_S) -> float:
    use_fast = current <= prev if danger_if_lower else current >= prev
    alpha = self._ema_alpha(dt_s, fast_tau_s if use_fast else slow_tau_s)
    return float(prev + alpha * (current - prev))

  def _filter_symmetric_metric(self, prev: float, current: float, dt_s: float, tau_s: float) -> float:
    alpha = self._ema_alpha(dt_s, tau_s)
    return float(prev + alpha * (current - prev))

  def _reset_hyundai_virtual_lead(self, reason: str) -> None:
    self._hyundai_virtual_lead = None
    self._hyundai_virtual_lead_source = None
    self._hyundai_virtual_lead_last_t = None
    self._hyundai_virtual_lead_identity_changed = False
    self._hyundai_virtual_lead_reset_reason = reason
    self._hyundai_reclaim_lead = None
    self._hyundai_reclaim_last_t = None
    self.hyundai_virtual_lead_debug = {
      "active": False,
      "reset_reason": reason,
    }

  def _should_reset_hyundai_virtual_lead(self, lead_source: str, lead) -> tuple[bool, str | None]:
    if lead is None or not getattr(lead, 'status', False):
      return True, "no_control_lead"
    if self._hyundai_virtual_lead is None or self._hyundai_virtual_lead_source is None:
      return True, "init"
    if lead_source != self._hyundai_virtual_lead_source:
      return True, "source_switch"
    if abs(float(getattr(lead, 'dRel', 0.0) or 0.0) - float(self._hyundai_virtual_lead.dRel)) > HYUNDAI_VIRTUAL_LEAD_RESET_DREL_M:
      return True, "drel_jump"
    if abs(float(getattr(lead, 'dPath', getattr(lead, 'yRel', 0.0)) or 0.0) - float(self._hyundai_virtual_lead.dPath)) > HYUNDAI_VIRTUAL_LEAD_RESET_DPATH_M:
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
    if max_accel is None:
      return None
    return float(max_accel)

  def _apply_hyundai_gap_reclaim(self, raw_lead_obstacle: np.ndarray, filtered_lead_obstacle: np.ndarray,
                                 raw_lead, filtered_lead, now: float) -> np.ndarray:
    self.gap_reclaim_accel_floor = 0.0
    self.gap_reclaim_obstacle_push = 0.0
    self.gap_reclaim_projection_scale = 1.0
    self.gap_reclaim_personality_max_accel = 0.0
    self.gap_reclaim_effective_cap = 0.0
    self._raw_reclaim_safety_override_active = False

    if not self._hyundai_ai_lead_stability_enabled:
      self._gap_reclaim_blend = 0.0
      self._gap_reclaim_last_t = now
      self.gap_reclaim_effective_cap = 0.0
      return np.minimum(raw_lead_obstacle, filtered_lead_obstacle)

    reclaim_lead = self._update_hyundai_reclaim_lead(now, raw_lead, filtered_lead)
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
    self.gap_reclaim_accel_floor = float(reclaim_intent)
    max_intent = max(float(self.gap_reclaim_effective_cap), 1e-3)
    target_blend = reclaim_intent / max_intent if max_intent > 0.0 else 0.0
    blend = self._update_gap_reclaim_blend(target_blend, now)
    obstacle_delta = np.maximum(raw_lead_obstacle - filtered_lead_obstacle, 0.0)
    horizon_ramp = 1.0 - np.exp(-T_IDXS / GAP_RECLAIM_HORIZON_RAMP_TAU_S)
    stabilization_push = obstacle_delta * blend * horizon_ramp

    gap_surplus = max(0.0, float(reclaim_lead.dRel) - get_headway_follow_distance(float(self.x0[1]), self.current_t_follow))
    reclaim_room_max = min(
      GAP_RECLAIM_RELAX_ROOM_MAX_M,
      max(0.0, gap_surplus - float(self._live_tune_cfg.gap_reclaim_gap_min_m)) * GAP_RECLAIM_RELAX_ROOM_FRACTION,
    )
    effective_reclaim_intent = max_intent * blend
    base_reclaim_room = np.minimum(0.5 * effective_reclaim_intent * np.square(T_IDXS), reclaim_room_max * horizon_ramp)
    reclaim_room = base_reclaim_room * float(self.gap_reclaim_projection_scale)

    obstacle_push = stabilization_push + reclaim_room
    self.gap_reclaim_obstacle_push = float(np.max(obstacle_push))
    target_obstacle = np.minimum(raw_lead_obstacle, filtered_lead_obstacle + stabilization_push) + reclaim_room

    raw_obstacle_margin = float(np.min(filtered_lead_obstacle - raw_lead_obstacle))
    raw_closing_speed = float(self.x0[1]) - float(getattr(raw_lead, 'vLead', self.x0[1]) or self.x0[1])
    raw_lead_accel = float(getattr(raw_lead, 'aLeadK', 0.0) or 0.0)
    raw_safety_override = (
      raw_obstacle_margin >= HYUNDAI_VIRTUAL_LEAD_RAW_OBSTACLE_MARGIN_M or
      raw_closing_speed > 0.3 or
      (raw_lead_accel < -0.6 and raw_closing_speed > 0.1)
    )
    self._raw_reclaim_safety_override_active = bool(raw_safety_override)
    if raw_safety_override:
      self._hyundai_reclaim_lead = copy.deepcopy(filtered_lead)
      self._hyundai_reclaim_last_t = now
      target_obstacle = np.minimum(target_obstacle, raw_lead_obstacle)

    return target_obstacle

  def _update_hyundai_reclaim_lead(self, now: float, raw_lead, filtered_lead) -> ControlLead:
    raw_control = ControlLead.from_lead(raw_lead) if raw_lead is not None else ControlLead()
    optimistic = copy.deepcopy(filtered_lead)
    optimistic.dRel = max(float(filtered_lead.dRel), float(raw_control.dRel))
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
    reclaim.dRel = self._filter_metric(prev.dRel, optimistic.dRel, dt_s, danger_if_lower=False)
    reclaim.yRel = filtered_lead.yRel
    reclaim.dPath = filtered_lead.dPath
    reclaim.vLat = filtered_lead.vLat
    reclaim.vRel = self._filter_metric(prev.vRel, optimistic.vRel, dt_s, danger_if_lower=False)
    reclaim.aRel = self._filter_metric(prev.aRel, optimistic.aRel, dt_s, danger_if_lower=False)
    reclaim.vLead = self._filter_metric(prev.vLead, optimistic.vLead, dt_s, danger_if_lower=False)
    reclaim.vLeadK = self._filter_metric(prev.vLeadK, optimistic.vLeadK, dt_s, danger_if_lower=False)
    reclaim.aLeadK = self._filter_metric(prev.aLeadK, optimistic.aLeadK, dt_s, danger_if_lower=False)
    reclaim.fcw = optimistic.fcw
    reclaim.aLeadTau = optimistic.aLeadTau
    reclaim.modelProb = self._filter_symmetric_metric(prev.modelProb, optimistic.modelProb, dt_s, HYUNDAI_VIRTUAL_LEAD_MODEL_PROB_TAU_S)
    reclaim.radar = optimistic.radar
    reclaim.radarTrackId = optimistic.radarTrackId
    self._hyundai_reclaim_lead = reclaim
    self._hyundai_reclaim_last_t = now
    return copy.deepcopy(reclaim)

  def _update_hyundai_virtual_lead(self, now: float, lead_source: str, lead):
    self._hyundai_virtual_lead_identity_changed = False

    if not self._hyundai_ai_lead_stability_enabled:
      self.hyundai_virtual_lead_debug = {"active": False}
      return None

    should_reset, reset_reason = self._should_reset_hyundai_virtual_lead(lead_source, lead)
    raw_lead = ControlLead.from_lead(lead) if lead is not None else ControlLead()

    if should_reset:
      self._hyundai_virtual_lead = raw_lead
      self._hyundai_virtual_lead_source = lead_source if raw_lead.status else None
      self._hyundai_virtual_lead_last_t = now if raw_lead.status else None
      self._hyundai_virtual_lead_identity_changed = bool(raw_lead.status)
      self._hyundai_virtual_lead_reset_reason = reset_reason
      if not raw_lead.status:
        self._reset_hyundai_virtual_lead(reset_reason or "no_control_lead")
        return None
    else:
      dt_s = float(np.clip(now - float(self._hyundai_virtual_lead_last_t or now), 0.0, 1.0))
      prev = self._hyundai_virtual_lead
      filtered = copy.deepcopy(prev)
      filtered.status = raw_lead.status
      filtered.dRel = self._filter_metric(prev.dRel, raw_lead.dRel, dt_s, danger_if_lower=True)
      filtered.yRel = self._filter_symmetric_metric(prev.yRel, raw_lead.yRel, dt_s, HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S)
      filtered.vRel = self._filter_metric(prev.vRel, raw_lead.vRel, dt_s, danger_if_lower=True)
      filtered.aRel = self._filter_metric(prev.aRel, raw_lead.aRel, dt_s, danger_if_lower=True)
      filtered.vLead = self._filter_metric(prev.vLead, raw_lead.vLead, dt_s, danger_if_lower=True)
      filtered.dPath = self._filter_symmetric_metric(prev.dPath, raw_lead.dPath, dt_s, HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S)
      filtered.vLat = self._filter_symmetric_metric(prev.vLat, raw_lead.vLat, dt_s, HYUNDAI_VIRTUAL_LEAD_PATH_TAU_S)
      filtered.vLeadK = self._filter_metric(prev.vLeadK, raw_lead.vLeadK, dt_s, danger_if_lower=True)
      filtered.aLeadK = self._filter_metric(prev.aLeadK, raw_lead.aLeadK, dt_s, danger_if_lower=True)
      filtered.fcw = raw_lead.fcw
      filtered.aLeadTau = raw_lead.aLeadTau
      filtered.modelProb = self._filter_symmetric_metric(prev.modelProb, raw_lead.modelProb, dt_s, HYUNDAI_VIRTUAL_LEAD_MODEL_PROB_TAU_S)
      filtered.radar = raw_lead.radar
      filtered.radarTrackId = raw_lead.radarTrackId
      self._hyundai_virtual_lead = filtered
      self._hyundai_virtual_lead_source = lead_source
      self._hyundai_virtual_lead_last_t = now
      self._hyundai_virtual_lead_reset_reason = None

    metrics = self._lead_follow_metrics(float(self.x0[1]), self.current_t_follow, self._hyundai_virtual_lead)
    self.hyundai_virtual_lead_debug = {
      "active": True,
      "source": str(self._hyundai_virtual_lead_source),
      "identity_changed": bool(self._hyundai_virtual_lead_identity_changed),
      "reset_reason": self._hyundai_virtual_lead_reset_reason,
      "filtered": self._lead_debug_payload(self._hyundai_virtual_lead),
      "metrics": metrics,
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
      self._reset_hyundai_virtual_lead("no_control_lead")
      self._acc_obstacle_mode = 'cruise'
      self._reset_acc_obstacle_candidate()
      self.gap_reclaim_obstacle_push = 0.0
      self.gap_reclaim_effective_cap = 0.0
      self._gap_reclaim_blend = 0.0
      self._gap_reclaim_last_t = now
      self.source = 'cruise'
      self.acc_source_debug = {
        "active_mode": "cruise",
        "delta_m": None,
        "candidate_mode": None,
        "used_hysteresis": True,
      }
      return cruise_obstacle

    best_lead_source, best_lead_obstacle = min(lead_candidates, key=lambda item: item[1][0])
    best_lead = self.control_leads[0] if best_lead_source == 'lead0' else self.control_leads[1]
    filtered_lead = self._update_hyundai_virtual_lead(now, best_lead_source, best_lead)
    if filtered_lead is None or not getattr(filtered_lead, 'status', False):
      self._acc_obstacle_mode = 'cruise'
      self._reset_acc_obstacle_candidate()
      self.source = 'cruise'
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
    low_speed_queue_hold = (
      float(self.x0[1]) <= HYUNDAI_LOW_SPEED_QUEUE_V_EGO_MAX and
      float(getattr(best_lead, 'dRel', 1e9) or 1e9) <= HYUNDAI_LOW_SPEED_QUEUE_DREL_MAX and
      float(getattr(best_lead, 'vLead', self.x0[1]) or self.x0[1]) <= HYUNDAI_LOW_SPEED_QUEUE_VLEAD_MAX and
      raw_metrics["pullaway_speed"] <= HYUNDAI_LOW_SPEED_QUEUE_PULLAWAY_MPS_MAX
    )
    raw_requires_owner = (
      low_speed_queue_hold or
      raw_metrics["gap_surplus"] <= HYUNDAI_VIRTUAL_LEAD_RETAIN_GAP_SURPLUS_M or
      raw_metrics["obstacle_0"] <= (float(cruise_obstacle[0]) - HYUNDAI_VIRTUAL_LEAD_RAW_OBSTACLE_MARGIN_M)
    )
    release_ready = (
      filtered_metrics["gap_surplus"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_GAP_SURPLUS_M and
      filtered_metrics["pullaway_speed"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_PULLAWAY_MPS
    )
    immediate_release = (
      filtered_metrics["gap_surplus"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_GAP_SURPLUS_M and
      filtered_metrics["pullaway_speed"] >= HYUNDAI_VIRTUAL_LEAD_RELEASE_IMMEDIATE_PULLAWAY_MPS
    )
    reacquire_lead = (
      raw_requires_owner or
      filtered_metrics["gap_surplus"] <= HYUNDAI_VIRTUAL_LEAD_REACQUIRE_GAP_SURPLUS_M
    )
    active_mode = self._acc_obstacle_mode
    reason = "filtered_hold"

    if raw_requires_owner:
      active_mode = 'lead'
      self._acc_obstacle_mode = 'lead'
      self._reset_acc_obstacle_candidate()
      if low_speed_queue_hold:
        reason = "low_speed_queue_hold"
      elif raw_metrics["gap_surplus"] <= HYUNDAI_VIRTUAL_LEAD_RETAIN_GAP_SURPLUS_M:
        reason = "raw_gap_hold"
      else:
        reason = "raw_obstacle_hold"
    elif active_mode == 'lead':
      if immediate_release:
        active_mode = 'cruise'
        self._acc_obstacle_mode = 'cruise'
        self._reset_acc_obstacle_candidate()
        reason = "filtered_pullaway_immediate"
      elif release_ready:
        active_mode = self._advance_acc_obstacle_candidate(
          'cruise', now, HYUNDAI_VIRTUAL_LEAD_RELEASE_DWELL_S, default_mode='lead',
        )
        reason = "filtered_pullaway_dwell"
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
      active_obstacle = self._apply_hyundai_gap_reclaim(best_lead_obstacle, filtered_lead_obstacle, best_lead, filtered_lead, now)
    else:
      self._acc_obstacle_mode = 'cruise'
      self.source = 'cruise'
      self.gap_reclaim_obstacle_push = 0.0
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
      "raw_pullaway_mps": float(raw_metrics["pullaway_speed"]),
      "filtered_pullaway_mps": float(filtered_metrics["pullaway_speed"]),
      "low_speed_queue_hold": bool(low_speed_queue_hold),
      "gap_reclaim_blend": float(self._gap_reclaim_blend),
      "gap_reclaim_obstacle_push_m": float(self.gap_reclaim_obstacle_push),
      "gap_reclaim_projection_scale": float(self.gap_reclaim_projection_scale),
      "gap_reclaim_effective_cap": float(self.gap_reclaim_effective_cap),
      "gap_reclaim_personality_max_accel": float(self.gap_reclaim_personality_max_accel),
      "raw_reclaim_safety_override": bool(self._raw_reclaim_safety_override_active),
      "candidate_mode": self._acc_obstacle_candidate_mode,
      "reason": reason,
      "used_hysteresis": True,
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
      a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
      cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
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
    if float(self.last_v_cruise_clipped[1]) <= v_ego + 0.05:
      return 0.0

    if self._hyundai_ai_lead_stability_enabled:
      if self.source not in ('lead0', 'lead1') or self._hyundai_reclaim_lead is None:
        return 0.0
      personality_max_accel = self._get_gap_reclaim_personality_max_accel(v_ego)
      return float(get_gap_reclaim_accel_floor(
        v_ego,
        self._hyundai_reclaim_lead,
        self.current_t_follow,
        self._live_tune_cfg,
        personality_max_accel=personality_max_accel,
      ))

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
    return build_cruise_response_model(
      min_accel_mps2=min_accel,
      max_accel_mps2=CRUISE_MAX_ACCEL,
      comfort_brake_mps2=COMFORT_BRAKE,
      actuation_delay_s=actuation_delay_s,
      planner_output_min_accel_mps2=planner_accel_min,
      planner_output_max_accel_mps2=planner_accel_max,
    )

  def update(self, radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard):
    v_ego = self.x0[1]
    now = time.monotonic()
    self._refresh_live_tune(now)

    # Get following distance
    if self.vibe_controller.is_follow_enabled():
      t_follow = self.vibe_controller.get_follow_distance_multiplier(v_ego)
      if t_follow is None:
        # Fallback to stock behavior when vibe controller can't provide a value
        t_follow = get_T_FOLLOW(personality)
    else:
      t_follow = get_T_FOLLOW(personality)
    self.current_t_follow = float(t_follow)

    raw_control_lead0, raw_control_lead1, lead_role_debug = self.lead_role_classifier.classify(
      v_ego, radarstate.leadOne, radarstate.leadTwo, now=now,
    )
    self.control_leads, lead_role_debug = self._stabilize_control_leads(
      radarstate.leadOne, radarstate.leadTwo, raw_control_lead0, raw_control_lead1, lead_role_debug,
    )
    control_lead0, control_lead1 = self.control_leads
    self.lead_role_debug = lead_role_debug
    self.status = control_lead0.status or control_lead1.status
    self._update_cutin_settle_state(now, v_ego, lead_role_debug)

    response_model = self.get_cruise_response_model(v_ego)
    self.last_cruise_response_model = response_model

    lead_xv_0 = self.process_lead(control_lead0)
    lead_xv_1 = self.process_lead(control_lead1)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])
    lead_0_preview = get_lead_approach_preview_buffer(v_ego, control_lead0, t_follow, self._live_tune_cfg)
    lead_1_preview = get_lead_approach_preview_buffer(v_ego, control_lead1, t_follow, self._live_tune_cfg)
    lead_0_obstacle = apply_lead_approach_preview(lead_0_obstacle, lead_0_preview)
    lead_1_obstacle = apply_lead_approach_preview(lead_1_obstacle, lead_1_preview)
    self.lead_approach_preview = (lead_0_preview, lead_1_preview)

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX

    # Update in ACC mode or ACC/e2e blend
    if self.mode == 'acc':
      self.params[:,5] = LEAD_DANGER_FACTOR

      response_model = self.get_cruise_response_model(v_ego)
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
      self.gap_reclaim_accel_floor = self.get_gap_reclaim_floor()
      self.cutin_settle_accel_floor = self.get_cutin_settle_floor(now)

      # These are not used in ACC mode
      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

    elif self.mode == 'blended':
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
      self.gap_reclaim_obstacle_push = 0.0
      self.gap_reclaim_projection_scale = 1.0
      self.gap_reclaim_personality_max_accel = 0.0
      self.gap_reclaim_effective_cap = 0.0
      self._gap_reclaim_blend = 0.0
      self._gap_reclaim_last_t = now
      self.cutin_settle_active = False
      self.cutin_settle_accel_floor = 0.0
      self.cutin_settle_debug = {}

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
    else:
      self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.prev_a)
    self.params[:,4] = t_follow

    self.run()

    fcw_lead_xv = None
    fcw_model_prob = 0.0
    if self.mode == 'acc':
      if self.source == 'lead0' and control_lead0.status:
        fcw_lead_xv = lead_xv_0
        fcw_model_prob = float(control_lead0.modelProb)
      elif self.source == 'lead1' and control_lead1.status:
        fcw_lead_xv = lead_xv_1
        fcw_model_prob = float(control_lead1.modelProb)
    else:
      if self.source == 'lead0' and control_lead0.status:
        fcw_lead_xv = lead_xv_0
        fcw_model_prob = float(control_lead0.modelProb)
      elif self.source == 'lead1' and control_lead1.status:
        fcw_lead_xv = lead_xv_1
        fcw_model_prob = float(control_lead1.modelProb)

    if (fcw_lead_xv is not None and
            np.any(fcw_lead_xv[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            fcw_model_prob > 0.9):
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
          "source_hysteresis": self.acc_source_debug,
          "filtered_virtual_lead": self.hyundai_virtual_lead_debug,
          "virtual_duplicate": lead_role_debug.get("virtual_duplicate", {"active": False}),
          "raw_duplicate_model": lead_role_debug.get("raw_duplicate_model", {}),
          "raw": lead_role_debug.get("raw", {}),
          "awareness": lead_role_debug.get("awareness", []),
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

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()
      # reset = 1
    # print(f"long_mpc timings: total internal {self.solve_time:.2e}, external: {(time.monotonic() - t0):.2e} qp {self.time_qp_solution:.2e}, \
    # lin {self.time_linearization:.2e} qp_iter {qp_iter}, reset {reset}")


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
