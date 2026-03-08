from __future__ import annotations

import math
from dataclasses import dataclass

from openpilot.selfdrive.controls.lib.longitudinal_response_model import (
  CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2,
  CruiseResponseModel,
  cruise_cap_for_required_average_decel,
  predict_average_decel_for_cruise_cap,
)


MAP_STRATEGY_ADVISORY = "advisory"
MAP_STRATEGY_STRATEGIC = "strategic"
DEFAULT_MAP_STRATEGY = MAP_STRATEGY_STRATEGIC
STRATEGIC_OVERSHOOT_DELTA_MPS = 1.0


def normalize_map_strategy(raw: str | bytes | None) -> str:
  if isinstance(raw, (bytes, bytearray)):
    try:
      raw = raw.decode("utf-8")
    except Exception:
      raw = ""
  value = str(raw or "").strip().lower()
  if value in (MAP_STRATEGY_ADVISORY, MAP_STRATEGY_STRATEGIC):
    return value
  return DEFAULT_MAP_STRATEGY


@dataclass
class MapCapCandidate:
  mode: str
  cap_mps: float | None
  start_m: float
  coverage: float
  reason: str
  anchor_dist_m: float | None = None
  anchor_vsafe_mps: float | None = None
  anchor_curvature: float | None = None
  anchor_index: int | None = None


@dataclass
class MapStrategyState:
  release_latched: bool = False
  takeover_since: float = 0.0
  counterevidence_since: float = 0.0
  release_reason: str = ""
  strategy_state: str = "idle"

  def reset(self) -> None:
    self.release_latched = False
    self.takeover_since = 0.0
    self.counterevidence_since = 0.0
    self.release_reason = ""
    self.strategy_state = "idle"


@dataclass
class MapStrategyDecision:
  apply_map_cap: bool
  map_floor_active: bool
  map_reason: str
  strategy_state: str
  vision_relax_allowed: bool
  vision_relax_reason: str
  takeover_dwell_s: float
  counterevidence_dwell_s: float


def advisory_start_distance(
  *,
  v_ego: float,
  vis_horizon_s: float,
  vis_margin_m: float,
  severe_vision: bool,
  partial_vision: bool,
  vision_confidence: float,
  conf_lo: float,
  conf_hi: float,
) -> float:
  s_start_full = max(0.0, float(v_ego) * float(vis_horizon_s) + float(vis_margin_m))
  if severe_vision:
    return max(0.0, float(vis_margin_m))
  if partial_vision:
    denom = max(1e-3, float(conf_hi) - float(conf_lo))
    blend = min(1.0, max(0.0, (float(vision_confidence) - float(conf_lo)) / denom))
    return max(0.0, float(vis_margin_m) + blend * (s_start_full - float(vis_margin_m)))
  return s_start_full


def _strategic_frontier(
  *,
  s_list: list[float],
  k_list: list[float],
  vsafe_list: list[float],
  abs_indices: list[int],
  s_start: float,
):
  best_vsafe = float('inf')
  for di, ki, vi, abs_idx in zip(s_list, k_list, vsafe_list, abs_indices, strict=False):
    if float(di) < float(s_start):
      continue
    vsafe = float(vi)
    if vsafe < best_vsafe - 1e-6:
      best_vsafe = vsafe
      yield float(di), float(ki), vsafe, int(abs_idx)


def compute_map_cap_candidate(
  *,
  mode: str,
  s_list: list[float],
  k_list: list[float],
  vsafe_list: list[float],
  abs_indices: list[int],
  v_ego: float,
  v_cruise: float,
  vis_horizon_s: float,
  vis_margin_m: float,
  severe_vision: bool,
  partial_vision: bool,
  vision_confidence: float,
  conf_lo: float,
  conf_hi: float,
  max_decel: float,
  horizon_limit_m: float,
  response_model: CruiseResponseModel | None = None,
  fixed_lead_time_s: float = 0.0,
  curve_phase_offset_s: float = 0.0,
  overshoot_phase_offset_s: float = 0.0,
  reference_speed_mps: float | None = None,
) -> MapCapCandidate:
  strategy_mode = normalize_map_strategy(mode)
  if strategy_mode == MAP_STRATEGY_STRATEGIC:
    s_start = 0.0
  else:
    s_start = advisory_start_distance(
      v_ego=v_ego,
      vis_horizon_s=vis_horizon_s,
      vis_margin_m=vis_margin_m,
      severe_vision=severe_vision,
      partial_vision=partial_vision,
      vision_confidence=vision_confidence,
      conf_lo=conf_lo,
      conf_hi=conf_hi,
    )

  a_plan = max(0.1, float(max_decel)) * 0.5
  any_future = False
  v_cap = float(v_cruise)
  anchor_dist_m = None
  anchor_vsafe_mps = None
  anchor_curvature = None
  anchor_index = None
  first_candidate = None
  direct_cap = float(v_cruise)
  direct_anchor = None
  response_constraints: list[tuple[float, float, float, float, int, int]] = []
  reference_speed = float(reference_speed_mps) if reference_speed_mps is not None else float(v_ego)
  timing_speed = max(0.1, float(v_ego))

  if strategy_mode == MAP_STRATEGY_STRATEGIC:
    candidate_points = _strategic_frontier(
      s_list=s_list,
      k_list=k_list,
      vsafe_list=vsafe_list,
      abs_indices=abs_indices,
      s_start=s_start,
    )
  else:
    candidate_points = (
      (float(di), float(ki), float(vi), int(abs_idx))
      for di, ki, vi, abs_idx in zip(s_list, k_list, vsafe_list, abs_indices, strict=False)
      if float(di) >= s_start
    )

  for seq, (di, ki, vi, abs_idx) in enumerate(candidate_points):
    any_future = True
    vsafe = float(vi)
    if first_candidate is None:
      first_candidate = (float(di), float(vsafe), float(ki), int(abs_idx))
    effective_distance = float(di)
    if strategy_mode == MAP_STRATEGY_STRATEGIC:
      # Reuse the driver's existing VTSC timing semantics for map planning too:
      # fixed lead time means "be at the anchor speed before the anchor by N seconds"
      # and should tighten the reachable cap even when the signed phase offsets are zero.
      effective_distance -= max(0.0, float(fixed_lead_time_s)) * timing_speed
      # curve timing moves the nominal "arrive at anchor speed" point earlier/later,
      # while overshoot timing only biases anchors that are materially tighter than the
      # current local vision target.
      effective_distance += float(curve_phase_offset_s) * timing_speed
      if vsafe + STRATEGIC_OVERSHOOT_DELTA_MPS < float(reference_speed):
        effective_distance += float(overshoot_phase_offset_s) * timing_speed
      effective_distance = max(0.0, effective_distance)
    if strategy_mode == MAP_STRATEGY_STRATEGIC and response_model is not None:
      try:
        braking_distance = max(0.0, effective_distance - float(v_ego) * float(response_model.actuation_delay_s))
        if braking_distance <= 1e-3:
          v_allow = float(vsafe)
          if v_allow < direct_cap - 1e-6:
            direct_cap = float(v_allow)
            direct_anchor = (float(di), float(vsafe), float(ki), int(abs_idx))
        elif vsafe >= float(v_ego) - 1e-6:
          v_allow = float(v_cruise)
        else:
          required_decel = max(0.0, (float(v_ego) * float(v_ego) - vsafe * vsafe) / (2.0 * braking_distance))
          response_constraints.append((float(required_decel), float(di), float(vsafe), float(ki), int(abs_idx), int(seq)))
          v_allow = float(v_cruise)
      except Exception:
        v_allow = float(v_ego)
    else:
      d = max(0.0, effective_distance - s_start)
      try:
        v_allow = math.sqrt(max(0.0, vsafe * vsafe + 2.0 * a_plan * d))
      except Exception:
        v_allow = float(v_ego)
    if anchor_dist_m is None or v_allow < v_cap - 1e-6:
      anchor_dist_m = float(di)
      anchor_vsafe_mps = float(vsafe)
      anchor_curvature = float(ki)
      anchor_index = int(abs_idx)
    v_cap = min(v_cap, v_allow)

  if strategy_mode == MAP_STRATEGY_STRATEGIC and response_model is not None:
    def _apply_anchor(anchor):
      nonlocal anchor_dist_m, anchor_vsafe_mps, anchor_curvature, anchor_index
      if anchor is None:
        return
      anchor_dist_m, anchor_vsafe_mps, anchor_curvature, anchor_index = anchor

    v_cap = float(v_cruise)
    if direct_anchor is not None:
      v_cap = float(direct_cap)
      _apply_anchor(direct_anchor)

    if response_constraints:
      max_supported_required_decel = predict_average_decel_for_cruise_cap(
        v_ego=float(v_ego),
        cruise_cap=0.0,
        response_model=response_model,
      ) + float(CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2)

      feasible_constraint = None
      infeasible_constraint = None
      for req, di, vsafe, ki, abs_idx, seq in response_constraints:
        if float(req) <= max_supported_required_decel + 1e-9:
          if (feasible_constraint is None or
              float(req) > float(feasible_constraint[0]) + 1e-9 or
              (abs(float(req) - float(feasible_constraint[0])) <= 1e-9 and int(seq) < int(feasible_constraint[5]))):
            feasible_constraint = (float(req), float(di), float(vsafe), float(ki), int(abs_idx), int(seq))
        elif (infeasible_constraint is None or
              float(vsafe) < float(infeasible_constraint[2]) - 1e-9 or
              (abs(float(vsafe) - float(infeasible_constraint[2])) <= 1e-9 and int(seq) < int(infeasible_constraint[5]))):
          infeasible_constraint = (float(req), float(di), float(vsafe), float(ki), int(abs_idx), int(seq))

      if feasible_constraint is not None:
        req, di, vsafe, ki, abs_idx, _ = feasible_constraint
        possible_cap = cruise_cap_for_required_average_decel(
          v_ego=float(v_ego),
          required_decel_mps2=float(req),
          response_model=response_model,
          v_cruise_upper=float(v_cruise),
        )
        if possible_cap < v_cap - 1e-6:
          v_cap = float(possible_cap)
          _apply_anchor((float(di), float(vsafe), float(ki), int(abs_idx)))

      if infeasible_constraint is not None:
        _, imp_di, imp_vsafe, imp_ki, imp_abs_idx, _ = infeasible_constraint
        fallback_cap = min(float(v_cruise), float(imp_vsafe))
        if fallback_cap < v_cap - 1e-6:
          v_cap = float(fallback_cap)
          _apply_anchor((float(imp_di), float(imp_vsafe), float(imp_ki), int(imp_abs_idx)))

    if anchor_dist_m is None and first_candidate is not None:
      _apply_anchor(first_candidate)

  if not any_future:
    coverage = 0.0
    if s_list:
      coverage = float(min(1.0, float(s_list[-1]) / max(1e-3, s_start or 1.0)))
    return MapCapCandidate(
      mode=strategy_mode,
      cap_mps=None,
      start_m=float(s_start),
      coverage=coverage,
      reason="no_future_points_beyond_start",
    )

  coverage = 0.0
  if s_list and float(s_list[-1]) > s_start:
    coverage = float(min(1.0, (float(s_list[-1]) - s_start) / max(1e-3, (float(horizon_limit_m) - s_start))))
  return MapCapCandidate(
    mode=strategy_mode,
    cap_mps=max(0.0, float(v_cap)),
    start_m=float(s_start),
    coverage=coverage,
    reason="cap_available",
    anchor_dist_m=anchor_dist_m,
    anchor_vsafe_mps=anchor_vsafe_mps,
    anchor_curvature=anchor_curvature,
    anchor_index=anchor_index,
  )


def evaluate_map_strategy(
  *,
  mode: str,
  state: MapStrategyState,
  candidate: MapCapCandidate | None,
  raw_target_pre_map: float,
  full_visibility: bool,
  vision_good: bool,
  turn_visible: bool,
  s_visible_m: float,
  vis_margin_m: float,
  now_s: float,
  apex_exit_ready: bool,
  takeover_eps_mps: float = 0.25,
  counterevidence_delta_mps: float = 0.75,
  takeover_dwell_s: float = 0.35,
  counterevidence_dwell_s: float = 0.75,
  rearm_margin_m: float = 15.0,
  rearm_delta_mps: float = 0.50,
) -> MapStrategyDecision:
  strategy_mode = normalize_map_strategy(mode)
  if candidate is None or candidate.cap_mps is None:
    state.reset()
    return MapStrategyDecision(
      apply_map_cap=False,
      map_floor_active=False,
      map_reason="no_cap",
      strategy_state="idle",
      vision_relax_allowed=False,
      vision_relax_reason="no_cap",
      takeover_dwell_s=0.0,
      counterevidence_dwell_s=0.0,
    )

  if strategy_mode == MAP_STRATEGY_ADVISORY:
    state.reset()
    map_still_tighter = bool(float(candidate.cap_mps) + takeover_eps_mps < float(raw_target_pre_map))
    map_cap_allowed = not bool(full_visibility and vision_good and turn_visible and (not map_still_tighter))
    return MapStrategyDecision(
      apply_map_cap=map_cap_allowed,
      map_floor_active=map_cap_allowed,
      map_reason="applied" if map_cap_allowed else "vision_suppressed",
      strategy_state="map_owns" if map_cap_allowed else "vision_owns",
      vision_relax_allowed=not map_cap_allowed,
      vision_relax_reason="" if map_cap_allowed else "full_visibility_takeover",
      takeover_dwell_s=0.0,
      counterevidence_dwell_s=0.0,
    )

  if not full_visibility or not vision_good:
    state.release_latched = False
    state.release_reason = ""
    state.takeover_since = 0.0
    state.counterevidence_since = 0.0

  anchor_dist_m = float(candidate.anchor_dist_m if candidate.anchor_dist_m is not None else 1e9)
  takeover_zone_m = max(0.0, float(s_visible_m) + float(vis_margin_m))
  in_takeover_zone = bool(full_visibility and vision_good and turn_visible and anchor_dist_m <= takeover_zone_m)

  if in_takeover_zone:
    if float(raw_target_pre_map) <= float(candidate.cap_mps) + takeover_eps_mps:
      if state.takeover_since <= 0.0:
        state.takeover_since = float(now_s)
    else:
      state.takeover_since = 0.0

    if float(raw_target_pre_map) >= float(candidate.cap_mps) + counterevidence_delta_mps:
      if state.counterevidence_since <= 0.0:
        state.counterevidence_since = float(now_s)
    else:
      state.counterevidence_since = 0.0
  else:
    state.takeover_since = 0.0
    state.counterevidence_since = 0.0
    if anchor_dist_m > (takeover_zone_m + float(rearm_margin_m)):
      state.release_latched = False
      state.release_reason = ""

  takeover_elapsed = max(0.0, float(now_s) - float(state.takeover_since)) if state.takeover_since > 0.0 else 0.0
  counterevidence_elapsed = max(0.0, float(now_s) - float(state.counterevidence_since)) if state.counterevidence_since > 0.0 else 0.0

  vision_relax_reason = ""
  if apex_exit_ready and full_visibility and turn_visible:
    state.release_latched = True
    state.release_reason = "post_apex_release"
  elif takeover_elapsed >= float(takeover_dwell_s):
    state.release_latched = True
    state.release_reason = "takeover_dwell"
  elif counterevidence_elapsed >= float(counterevidence_dwell_s):
    state.release_latched = True
    state.release_reason = "counterevidence_dwell"

  if state.release_latched:
    if in_takeover_zone:
      vision_relax_reason = state.release_reason or "vision_takeover"
    elif float(candidate.cap_mps) + float(rearm_delta_mps) < float(raw_target_pre_map) and anchor_dist_m > (takeover_zone_m + float(rearm_margin_m)):
      state.release_latched = False
      state.release_reason = ""

  if state.release_latched:
    state.strategy_state = "vision_owns"
    return MapStrategyDecision(
      apply_map_cap=False,
      map_floor_active=False,
      map_reason=state.release_reason or "vision_suppressed",
      strategy_state=state.strategy_state,
      vision_relax_allowed=True,
      vision_relax_reason=vision_relax_reason or state.release_reason or "vision_takeover",
      takeover_dwell_s=takeover_elapsed,
      counterevidence_dwell_s=counterevidence_elapsed,
    )

  if in_takeover_zone:
    state.strategy_state = "takeover_zone"
  elif full_visibility and vision_good:
    state.strategy_state = "vision_clear_waiting"
  else:
    state.strategy_state = "map_owns"
  return MapStrategyDecision(
    apply_map_cap=True,
    map_floor_active=True,
    map_reason="applied",
    strategy_state=state.strategy_state,
    vision_relax_allowed=False,
    vision_relax_reason="map_floor_tighter",
    takeover_dwell_s=takeover_elapsed,
    counterevidence_dwell_s=counterevidence_elapsed,
  )
