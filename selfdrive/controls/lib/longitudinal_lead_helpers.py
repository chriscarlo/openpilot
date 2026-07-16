#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig


COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
LOW_SPEED_LAUNCH_FACTOR_V_EGO_BP = [0.0, 2.0, 6.0, 10.0]
LOW_SPEED_LAUNCH_FACTOR_V_EGO_V = [1.0, 1.0, 0.60, 0.0]
LOW_SPEED_LAUNCH_FACTOR_V_LEAD_BP = [0.0, 0.2, 1.0, 3.0, 6.0]
LOW_SPEED_LAUNCH_FACTOR_V_LEAD_V = [0.0, 0.0, 0.35, 0.80, 1.0]
LOW_SPEED_LAUNCH_FACTOR_PULLAWAY_BP = [0.0, 0.15, 0.5, 1.5, 3.0]
LOW_SPEED_LAUNCH_FACTOR_PULLAWAY_V = [0.0, 0.0, 0.25, 0.75, 1.0]
LOW_SPEED_LAUNCH_FACTOR_GAP_BP = [0.0, 0.5, 2.0, 5.0, 9.0]
LOW_SPEED_LAUNCH_FACTOR_GAP_V = [0.0, 0.05, 0.25, 0.70, 1.0]
LOW_SPEED_LAUNCH_MAX_ACCEL = 2.4


def _lead_float(lead, attr: str, default: float) -> float:
  value = getattr(lead, attr, default)
  if value is None:
    return float(default)
  return float(value)


def get_headway_follow_distance(v_ego, t_follow) -> float:
  return STOP_DISTANCE + float(t_follow) * float(v_ego)


def compute_lead_stopping_need_decel(v_ego, lead,
                                     tuning: LeadResponseTuningConfig | None = None) -> float:
  tuning = LeadResponseTuningConfig.defaults() if tuning is None else tuning
  if lead is None or not getattr(lead, 'status', False):
    return 0.0

  v_ego = float(v_ego)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  v_lead_raw = _lead_float(lead, 'vLead', v_ego)
  oncoming = v_lead_raw < float(tuning.lead_slowdown_kinematic_oncoming_vlead_mps)
  v_lead = max(0.0, v_lead_raw)
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


def compute_relatch_required_decel(v_ego, lead, t_follow,
                                   tuning: LeadResponseTuningConfig | None = None) -> float:
  if lead is None or not getattr(lead, 'status', False):
    return 0.0
  v_ego = float(v_ego)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  v_lead_raw = _lead_float(lead, 'vLead', v_ego)
  v_rel = _lead_float(lead, 'vRel', v_lead_raw - v_ego)
  closing = max(0.0, v_ego - max(0.0, v_lead_raw), -v_rel)
  match_required = 0.0
  if closing > 0.0:
    gap_surplus = d_rel - get_headway_follow_distance(v_ego, t_follow)
    match_required = (closing ** 2) / (2.0 * max(gap_surplus, 0.5))
  return float(max(match_required, compute_lead_stopping_need_decel(v_ego, lead, tuning)))


def get_low_speed_launch_follow_factor(v_ego, lead, t_follow) -> float:
  if lead is None or not getattr(lead, 'status', False):
    return 0.0

  v_ego = float(v_ego)
  d_rel = float(getattr(lead, 'dRel', 0.0) or 0.0)
  v_lead = _lead_float(lead, 'vLead', v_ego)
  v_rel = _lead_float(lead, 'vRel', 0.0)
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
