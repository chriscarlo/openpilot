from __future__ import annotations

import math
from dataclasses import dataclass

from openpilot.sunnypilot.selfdrive.controls.lib.planner_lag_debug import (
  SPAN_MAP_CAP_ADVISORY,
  SPAN_MAP_CAP_STRATEGIC,
  end_span,
  start_span,
)
from openpilot.selfdrive.controls.lib.longitudinal_response_model import (
  CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2,
  CruiseResponseModel,
  cruise_cap_for_required_average_decel,
  predict_average_decel_for_cruise_cap,
)


MAP_STRATEGY_ADVISORY = "advisory"
MAP_STRATEGY_STRATEGIC = "strategic"
DEFAULT_MAP_STRATEGY = MAP_STRATEGY_STRATEGIC
CURVE_PHASE_OFFSET_ZERO_BASELINE_S = -3.0
STRATEGIC_OVERSHOOT_DELTA_MPS = 1.0
STRATEGIC_CHAIN_LOCAL_MIN_EPS_MPS = 0.05
STRATEGIC_CHAIN_REARM_RISE_MPS = 0.75
WINDING_ROAD_LOOKAHEAD_M = 325.0
WINDING_ROAD_BASELINE_QUANTILE = 0.85
WINDING_ROAD_MIN_DROP_MPS = 2.0
WINDING_ROAD_DROP_RATIO = 0.12
WINDING_ROAD_SHORT_GAP_MAX_M = 120.0
WINDING_ROAD_CURVE_FRACTION_START = 0.20
WINDING_ROAD_CURVE_FRACTION_FULL = 0.50
WINDING_ROAD_ACTIVE_SCORE = 0.55
WINDING_BEHAVIOR_MAX_LEVEL = 5
WINDING_BEHAVIOR_MIN_SIGNAL = 0.20
WINDING_BEHAVIOR_SOFT_SIGNAL = 0.35
WINDING_BEHAVIOR_LOCAL_LEVEL4_SCORE = 0.78
WINDING_BEHAVIOR_LOCAL_LEVEL5_SCORE = 0.90


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


def effective_curve_phase_offset_s(curve_phase_offset_s: float, *, extra_adjust_s: float = 0.0) -> float:
  # Keep the user-facing knob centered at 0 while preserving the earlier
  # timing that used to require dialing CurvePhaseOffsetS down to -3.0.
  return float(curve_phase_offset_s) + CURVE_PHASE_OFFSET_ZERO_BASELINE_S + float(extra_adjust_s)


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


@dataclass
class WindingRoadContext:
  active: bool = False
  score: float = 0.0
  horizon_m: float = 0.0
  reference_vsafe_mps: float = 0.0
  min_anchor_vsafe_mps: float | None = None
  anchor_count: int = 0
  short_gap_count: int = 0
  curve_distance_m: float = 0.0


@dataclass(frozen=True)
class WindingBehaviorProfile:
  level: int = 0
  name: str = "normal"
  fixed_lead_time_adjust_s: float = 0.0
  curve_phase_offset_adjust_s: float = 0.0
  overshoot_phase_offset_adjust_s: float = 0.0
  apex_release_lat_acc_ratio: float = 0.92
  apex_release_hold_s: float = 0.60
  takeover_dwell_s: float = 0.35
  counterevidence_dwell_s: float = 0.75
  rearm_margin_m: float = 15.0
  rearm_delta_mps: float = 0.50
  allow_immediate_post_apex_release: bool = True
  v_turn_release_up_slew_mps2: float = float('inf')


WINDING_BEHAVIOR_PROFILES: dict[int, WindingBehaviorProfile] = {
  0: WindingBehaviorProfile(
    level=0,
    name="normal",
    apex_release_lat_acc_ratio=0.88,
    apex_release_hold_s=0.60,
    allow_immediate_post_apex_release=True,
    v_turn_release_up_slew_mps2=4.5,
  ),
  1: WindingBehaviorProfile(
    level=1,
    name="gentle_curvy",
    curve_phase_offset_adjust_s=0.03,
    overshoot_phase_offset_adjust_s=0.02,
    apex_release_lat_acc_ratio=0.80,
    apex_release_hold_s=0.80,
    takeover_dwell_s=0.40,
    counterevidence_dwell_s=0.85,
    rearm_margin_m=13.0,
    rearm_delta_mps=0.45,
    allow_immediate_post_apex_release=True,
    v_turn_release_up_slew_mps2=4.0,
  ),
  2: WindingBehaviorProfile(
    level=2,
    name="sustained_curvy",
    curve_phase_offset_adjust_s=0.14,
    overshoot_phase_offset_adjust_s=0.10,
    apex_release_lat_acc_ratio=0.62,
    apex_release_hold_s=0.75,
    takeover_dwell_s=0.55,
    counterevidence_dwell_s=1.00,
    rearm_margin_m=10.0,
    rearm_delta_mps=0.38,
    allow_immediate_post_apex_release=False,
    v_turn_release_up_slew_mps2=3.6,
  ),
  3: WindingBehaviorProfile(
    level=3,
    name="tight_winding",
    curve_phase_offset_adjust_s=0.22,
    overshoot_phase_offset_adjust_s=0.16,
    apex_release_lat_acc_ratio=0.48,
    apex_release_hold_s=0.50,
    takeover_dwell_s=0.70,
    counterevidence_dwell_s=1.20,
    rearm_margin_m=8.0,
    rearm_delta_mps=0.32,
    allow_immediate_post_apex_release=False,
    v_turn_release_up_slew_mps2=3.2,
  ),
  4: WindingBehaviorProfile(
    level=4,
    name="switchback_zone",
    curve_phase_offset_adjust_s=0.30,
    overshoot_phase_offset_adjust_s=0.22,
    apex_release_lat_acc_ratio=0.40,
    apex_release_hold_s=0.35,
    takeover_dwell_s=0.85,
    counterevidence_dwell_s=1.35,
    rearm_margin_m=6.0,
    rearm_delta_mps=0.26,
    allow_immediate_post_apex_release=False,
    v_turn_release_up_slew_mps2=2.8,
  ),
  5: WindingBehaviorProfile(
    level=5,
    name="hairpin_extreme",
    curve_phase_offset_adjust_s=0.40,
    overshoot_phase_offset_adjust_s=0.30,
    apex_release_lat_acc_ratio=0.30,
    apex_release_hold_s=0.25,
    takeover_dwell_s=1.00,
    counterevidence_dwell_s=1.50,
    rearm_margin_m=4.0,
    rearm_delta_mps=0.20,
    allow_immediate_post_apex_release=False,
    v_turn_release_up_slew_mps2=2.4,
  ),
}
DEFAULT_WINDING_BEHAVIOR_PROFILE = WINDING_BEHAVIOR_PROFILES[0]


def _clip01(value: float) -> float:
  return min(1.0, max(0.0, float(value)))


def _clamp_winding_level(level: int | float) -> int:
  try:
    raw = int(level)
  except Exception:
    raw = 0
  return min(max(raw, 0), int(WINDING_BEHAVIOR_MAX_LEVEL))


def _local_winding_behavior_level(*, active: bool, score: float) -> int:
  if not bool(active):
    return 0
  score_f = _clip01(score)
  if score_f >= float(WINDING_BEHAVIOR_LOCAL_LEVEL5_SCORE):
    return 5
  if score_f >= float(WINDING_BEHAVIOR_LOCAL_LEVEL4_SCORE):
    return 4
  return 3


def _mapd_winding_behavior_level(*, level: int, score: float, confidence: float) -> int:
  eff_level = _clamp_winding_level(level)
  if eff_level <= 0:
    return 0

  signal = min(1.0, max(0.0, 0.65 * _clip01(score) + 0.35 * _clip01(confidence)))
  if signal < float(WINDING_BEHAVIOR_MIN_SIGNAL):
    return 0
  if signal < float(WINDING_BEHAVIOR_SOFT_SIGNAL):
    return max(1, eff_level - 1)
  return eff_level


def resolve_winding_behavior_profile(
  *,
  active: bool,
  level: int,
  score: float,
  confidence: float,
  source: str,
  local_active: bool = False,
  local_score: float = 0.0,
  mapd_level: int = 0,
  mapd_score: float = 0.0,
  mapd_confidence: float = 0.0,
) -> WindingBehaviorProfile:
  local_level = _local_winding_behavior_level(
    active=bool(local_active),
    score=float(local_score),
  )
  mapd_level_eff = _mapd_winding_behavior_level(
    level=int(mapd_level),
    score=float(mapd_score),
    confidence=float(mapd_confidence),
  )

  fused_level = _clamp_winding_level(level)
  fused_score = _clip01(score)
  fused_confidence = _clip01(confidence)
  source_value = str(source or "none").strip().lower()

  if source_value == "blended":
    selected_level = max(fused_level, mapd_level_eff, local_level)
  elif source_value == "mapd":
    selected_level = max(mapd_level_eff, min(fused_level, mapd_level_eff))
  elif source_value == "local":
    selected_level = max(local_level, fused_level)
  else:
    selected_level = max(mapd_level_eff, local_level, fused_level if bool(active) else 0)

  if selected_level <= 0:
    signal = max(fused_score, fused_confidence, _clip01(mapd_score), _clip01(local_score))
    if bool(active) and signal >= float(WINDING_BEHAVIOR_SOFT_SIGNAL):
      selected_level = max(1, fused_level)

  return WINDING_BEHAVIOR_PROFILES.get(_clamp_winding_level(selected_level), DEFAULT_WINDING_BEHAVIOR_PROFILE)


def _max_entry_speed_for_target(
  *,
  target_speed: float,
  distance_m: float,
  response_model: CruiseResponseModel,
) -> float:
  target = max(0.0, float(target_speed))
  distance = max(0.0, float(distance_m))
  if distance <= 1e-3:
    return target

  decel = max(1e-3, float(response_model.planning_decel_mps2))
  delay = max(0.0, float(response_model.actuation_delay_s))
  delay_term = decel * delay
  radicand = (delay_term * delay_term) + (target * target) + (2.0 * decel * distance)
  return max(0.0, -delay_term + math.sqrt(max(0.0, radicand)))


def _strategic_chain_envelope_cap(
  *,
  frontier_points: list[tuple[float, float, float, float, int]],
  response_model: CruiseResponseModel,
  v_cruise: float,
) -> tuple[float, tuple[float, float, float, int] | None]:
  if not frontier_points:
    return float(v_cruise), None

  ordered = sorted(frontier_points, key=lambda row: (float(row[0]), float(row[1]), int(row[4])))
  caps: list[float] = [float(v_cruise)] * len(ordered)
  anchors: list[tuple[float, float, float, int] | None] = [None] * len(ordered)

  for idx in range(len(ordered) - 1, -1, -1):
    eff_dist, abs_dist, vsafe, curvature, abs_idx = ordered[idx]
    point_anchor = (float(abs_dist), float(vsafe), float(curvature), int(abs_idx))
    if idx == len(ordered) - 1:
      caps[idx] = min(float(v_cruise), float(vsafe))
      anchors[idx] = point_anchor
      continue

    next_eff_dist = float(ordered[idx + 1][0])
    carry_cap = _max_entry_speed_for_target(
      target_speed=float(caps[idx + 1]),
      distance_m=max(0.0, next_eff_dist - float(eff_dist)),
      response_model=response_model,
    )
    if float(vsafe) <= carry_cap + 1e-6:
      caps[idx] = min(float(v_cruise), float(vsafe))
      anchors[idx] = point_anchor
    else:
      caps[idx] = min(float(v_cruise), float(carry_cap))
      anchors[idx] = anchors[idx + 1]

  current_cap = _max_entry_speed_for_target(
    target_speed=float(caps[0]),
    distance_m=max(0.0, float(ordered[0][0])),
    response_model=response_model,
  )
  return min(float(v_cruise), float(current_cap)), anchors[0]


def _strategic_chain_anchor_points(
  *,
  source_points: list[tuple[float, float, float, float, int]],
) -> list[tuple[float, float, float, float, int]]:
  if not source_points:
    return []

  ordered = sorted(source_points, key=lambda row: (float(row[0]), float(row[1]), int(row[4])))
  eps = float(STRATEGIC_CHAIN_LOCAL_MIN_EPS_MPS)
  minima_runs: list[list[int]] = []
  cur_run: list[int] = []

  def _flush_run() -> None:
    nonlocal cur_run
    if cur_run:
      minima_runs.append(list(cur_run))
      cur_run = []

  for idx, row in enumerate(ordered):
    v = float(row[2])
    prev_v = float(ordered[idx - 1][2]) if idx > 0 else float('inf')
    next_v = float(ordered[idx + 1][2]) if idx + 1 < len(ordered) else float('inf')
    local_min = bool(
      (v <= prev_v + eps) and (v <= next_v + eps) and
      ((idx == 0) or (idx + 1 == len(ordered)) or (v < prev_v - eps) or (v < next_v - eps))
    )
    if local_min:
      cur_run.append(int(idx))
    else:
      _flush_run()
  _flush_run()

  if not minima_runs:
    return []

  candidate_indices: list[int] = []
  for run in minima_runs:
    best_idx = min(
      run,
      key=lambda idx: (
        float(ordered[idx][2]),
        float(ordered[idx][0]),
        float(ordered[idx][1]),
        int(ordered[idx][4]),
      ),
    )
    candidate_indices.append(int(best_idx))

  accepted_indices: list[int] = []
  rise_delta = float(STRATEGIC_CHAIN_REARM_RISE_MPS)
  for cand_idx in candidate_indices:
    if not accepted_indices:
      accepted_indices.append(int(cand_idx))
      continue

    last_idx = int(accepted_indices[-1])
    last_v = float(ordered[last_idx][2])
    cand_v = float(ordered[cand_idx][2])
    peak_v = max(float(row[2]) for row in ordered[last_idx:cand_idx + 1])
    if (cand_v < last_v - eps) or (peak_v >= last_v + rise_delta):
      accepted_indices.append(int(cand_idx))

  return [ordered[idx] for idx in accepted_indices]


def classify_winding_road_context(
  *,
  s_list: list[float],
  vsafe_list: list[float],
  abs_indices: list[int],
  lookahead_m: float = WINDING_ROAD_LOOKAHEAD_M,
) -> WindingRoadContext:
  # This intentionally classifies "curve density / chained bends ahead" from the same unsigned
  # map speed profile the cap logic already trusts. Signed left/right alternation can be layered
  # on later from the preview polyline without changing this behavior-neutral summary.
  points: list[tuple[float, float, int]] = []
  max_lookahead = max(0.0, float(lookahead_m))
  for di, vi, abs_idx in zip(s_list, vsafe_list, abs_indices, strict=False):
    dist_m = float(di)
    if dist_m < 0.0:
      continue
    if dist_m > max_lookahead:
      break
    points.append((dist_m, float(vi), int(abs_idx)))

  if not points:
    return WindingRoadContext()

  horizon_m = float(points[-1][0])
  baseline_samples = sorted(float(vsafe) for _dist, vsafe, _abs_idx in points)
  q_idx = int(round((len(baseline_samples) - 1) * float(WINDING_ROAD_BASELINE_QUANTILE)))
  q_idx = min(max(q_idx, 0), len(baseline_samples) - 1)
  reference_vsafe = float(baseline_samples[q_idx])
  anchor_drop = max(float(WINDING_ROAD_MIN_DROP_MPS), float(reference_vsafe) * float(WINDING_ROAD_DROP_RATIO))
  curve_threshold = float(reference_vsafe) - float(anchor_drop)

  curve_distance_m = 0.0
  first_dist, first_vsafe, _ = points[0]
  if float(first_vsafe) <= curve_threshold + 1e-6:
    curve_distance_m += max(0.0, float(first_dist))
  prev_dist = float(first_dist)
  prev_vsafe = float(first_vsafe)
  for dist_m, vsafe, _ in points[1:]:
    dist_m = float(dist_m)
    vsafe = float(vsafe)
    if min(float(prev_vsafe), float(vsafe)) <= curve_threshold + 1e-6:
      curve_distance_m += max(0.0, float(dist_m) - float(prev_dist))
    prev_dist = float(dist_m)
    prev_vsafe = float(vsafe)

  anchor_points = _strategic_chain_anchor_points(
    source_points=[(float(dist_m), float(dist_m), float(vsafe), 0.0, int(abs_idx)) for dist_m, vsafe, abs_idx in points],
  )
  meaningful_anchors = [row for row in anchor_points if float(row[2]) <= curve_threshold + 1e-6]
  anchor_count = len(meaningful_anchors)
  short_gap_count = 0
  for prev_row, next_row in zip(meaningful_anchors, meaningful_anchors[1:], strict=False):
    gap_m = float(next_row[1]) - float(prev_row[1])
    if gap_m <= float(WINDING_ROAD_SHORT_GAP_MAX_M) + 1e-6:
      short_gap_count += 1

  min_anchor_vsafe = min((float(row[2]) for row in meaningful_anchors), default=None)
  curve_fraction = (float(curve_distance_m) / max(1e-3, float(horizon_m))) if horizon_m > 1e-6 else 0.0
  anchor_score = _clip01((float(anchor_count) - 1.0) / 1.5)
  gap_score = _clip01(float(short_gap_count) / max(1.0, float(anchor_count - 1)))
  density_score = _clip01(
    (float(curve_fraction) - float(WINDING_ROAD_CURVE_FRACTION_START)) /
    max(1e-3, float(WINDING_ROAD_CURVE_FRACTION_FULL) - float(WINDING_ROAD_CURVE_FRACTION_START))
  )
  depth_score = 0.0
  if min_anchor_vsafe is not None:
    depth_score = _clip01((float(reference_vsafe) - float(min_anchor_vsafe) - float(anchor_drop)) / 4.0)

  score = (
    0.35 * float(anchor_score) +
    0.30 * float(gap_score) +
    0.20 * float(density_score) +
    0.15 * float(depth_score)
  )
  active = bool(
    anchor_count >= 2 and
    short_gap_count >= 1 and
    score >= float(WINDING_ROAD_ACTIVE_SCORE)
  )
  return WindingRoadContext(
    active=active,
    score=float(score),
    horizon_m=float(horizon_m),
    reference_vsafe_mps=float(reference_vsafe),
    min_anchor_vsafe_mps=None if min_anchor_vsafe is None else float(min_anchor_vsafe),
    anchor_count=int(anchor_count),
    short_gap_count=int(short_gap_count),
    curve_distance_m=float(curve_distance_m),
  )


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
  winding_profile: WindingBehaviorProfile | None = None,
) -> MapCapCandidate:
  strategy_mode = normalize_map_strategy(mode)
  profile = winding_profile or DEFAULT_WINDING_BEHAVIOR_PROFILE
  fixed_lead_time = max(0.0, float(fixed_lead_time_s) + float(profile.fixed_lead_time_adjust_s))
  curve_phase_offset = effective_curve_phase_offset_s(
    curve_phase_offset_s,
    extra_adjust_s=float(profile.curve_phase_offset_adjust_s),
  )
  overshoot_phase_offset = float(overshoot_phase_offset_s) + float(profile.overshoot_phase_offset_adjust_s)
  total_span = start_span(SPAN_MAP_CAP_STRATEGIC if strategy_mode == MAP_STRATEGY_STRATEGIC else SPAN_MAP_CAP_ADVISORY)
  try:
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
    strategic_chain_source_points: list[tuple[float, float, float, float, int]] = []
    reference_speed = float(reference_speed_mps) if reference_speed_mps is not None else float(v_ego)
    timing_speed = max(0.1, float(v_ego))

    all_strategic_points = []
    if strategy_mode == MAP_STRATEGY_STRATEGIC:
      all_strategic_points = [
        (float(di), float(ki), float(vi), int(abs_idx))
        for di, ki, vi, abs_idx in zip(s_list, k_list, vsafe_list, abs_indices, strict=False)
        if float(di) >= s_start
      ]

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
        effective_distance -= float(fixed_lead_time) * timing_speed
        # curve timing moves the nominal "arrive at anchor speed" point earlier/later,
        # while overshoot timing only biases anchors that are materially tighter than the
        # current local vision target.
        effective_distance += float(curve_phase_offset) * timing_speed
        if vsafe + STRATEGIC_OVERSHOOT_DELTA_MPS < float(reference_speed):
          effective_distance += float(overshoot_phase_offset) * timing_speed
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

      if all_strategic_points:
        for di, ki, vi, abs_idx in all_strategic_points:
          effective_distance = float(di)
          effective_distance -= float(fixed_lead_time) * timing_speed
          effective_distance += float(curve_phase_offset) * timing_speed
          if float(vi) + STRATEGIC_OVERSHOOT_DELTA_MPS < float(reference_speed):
            effective_distance += float(overshoot_phase_offset) * timing_speed
          effective_distance = max(0.0, effective_distance)
          strategic_chain_source_points.append((float(effective_distance), float(di), float(vi), float(ki), int(abs_idx)))

      strategic_chain_points = _strategic_chain_anchor_points(
        source_points=strategic_chain_source_points,
      )
      if strategic_chain_points:
        envelope_cap, envelope_anchor = _strategic_chain_envelope_cap(
          frontier_points=strategic_chain_points,
          response_model=response_model,
          v_cruise=float(v_cruise),
        )
        if envelope_cap < v_cap - 1e-6:
          v_cap = float(envelope_cap)
          _apply_anchor(envelope_anchor)

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
  finally:
    end_span(total_span)


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
  winding_profile: WindingBehaviorProfile | None = None,
) -> MapStrategyDecision:
  strategy_mode = normalize_map_strategy(mode)
  profile = winding_profile or DEFAULT_WINDING_BEHAVIOR_PROFILE
  takeover_dwell_s = max(float(takeover_dwell_s), float(profile.takeover_dwell_s))
  counterevidence_dwell_s = max(float(counterevidence_dwell_s), float(profile.counterevidence_dwell_s))
  if int(profile.level) > 0:
    rearm_margin_m = min(float(rearm_margin_m), float(profile.rearm_margin_m))
    rearm_delta_mps = min(float(rearm_delta_mps), float(profile.rearm_delta_mps))
  allow_immediate_post_apex_release = bool(profile.allow_immediate_post_apex_release)
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
  takeover_zone_m = max(0.0, float(s_visible_m))
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
  if allow_immediate_post_apex_release and apex_exit_ready and full_visibility and turn_visible:
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
