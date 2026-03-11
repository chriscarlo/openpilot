from dataclasses import dataclass

import numpy as np

DEFAULT_LANE_WIDTH = 3.7
MAX_FIT_DISTANCE_M = 15.0
MIN_VALID_LANE_WIDTH_M = 2.8
MAX_VALID_LANE_WIDTH_M = 4.8
MAX_WIDTH_SPREAD_M = 0.45
MIN_VALID_SAMPLES = 3


@dataclass(frozen=True)
class LaneCenterEstimate:
  left_y: float
  right_y: float
  left_prob: float
  right_prob: float
  center_y: float
  lane_width: float
  center_prob: float
  center_valid: bool


def _to_array(values) -> np.ndarray:
  return np.asarray(list(values), dtype=np.float32)


def _weighted_intercept(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
  weight_sum = float(np.sum(weights))
  if weight_sum <= 0.0:
    return float(y[0]) if len(y) else 0.0

  x_mean = float(np.sum(weights * x) / weight_sum)
  y_mean = float(np.sum(weights * y) / weight_sum)
  denom = float(np.sum(weights * np.square(x - x_mean)))
  if denom <= 1e-6:
    return y_mean

  slope = float(np.sum(weights * (x - x_mean) * (y - y_mean)) / denom)
  return float(y_mean - slope * x_mean)


def estimate_lane_center(lane_lines, lane_line_probs) -> LaneCenterEstimate:
  left_line = lane_lines[1]
  right_line = lane_lines[2]

  left_y = _to_array(left_line.y)
  right_y = _to_array(right_line.y)
  x = _to_array(left_line.x)

  left_y0 = float(left_y[0]) if len(left_y) else 0.0
  right_y0 = float(right_y[0]) if len(right_y) else 0.0
  left_prob = float(lane_line_probs[1]) if len(lane_line_probs) > 1 else 0.0
  right_prob = float(lane_line_probs[2]) if len(lane_line_probs) > 2 else 0.0

  raw_center_y = 0.5 * (left_y0 + right_y0)
  raw_lane_width = right_y0 - left_y0

  sample_count = min(len(x), len(left_y), len(right_y))
  if sample_count < MIN_VALID_SAMPLES:
    return LaneCenterEstimate(left_y0, right_y0, left_prob, right_prob, raw_center_y, raw_lane_width, 0.0, False)

  x = x[:sample_count]
  left_y = left_y[:sample_count]
  right_y = right_y[:sample_count]
  lane_width = right_y - left_y
  centerline = 0.5 * (left_y + right_y)

  mask = (
    np.isfinite(x) &
    np.isfinite(left_y) &
    np.isfinite(right_y) &
    np.isfinite(lane_width) &
    np.isfinite(centerline) &
    (x <= MAX_FIT_DISTANCE_M) &
    (lane_width >= MIN_VALID_LANE_WIDTH_M) &
    (lane_width <= MAX_VALID_LANE_WIDTH_M)
  )

  if int(np.count_nonzero(mask)) < MIN_VALID_SAMPLES:
    return LaneCenterEstimate(left_y0, right_y0, left_prob, right_prob, raw_center_y, raw_lane_width, 0.0, False)

  x_fit = x[mask]
  center_fit = centerline[mask]
  width_fit = lane_width[mask]
  weights = 1.0 / (1.0 + x_fit)

  center_y = _weighted_intercept(x_fit, center_fit, weights)
  lane_width_m = float(np.average(width_fit, weights=weights))
  width_spread = float(np.sqrt(np.average(np.square(width_fit - lane_width_m), weights=weights)))

  prob_score = float(np.clip(np.sqrt(max(left_prob, 0.0) * max(right_prob, 0.0)), 0.0, 1.0))
  width_score = float(np.clip(1.0 - abs(lane_width_m - DEFAULT_LANE_WIDTH) / 1.2, 0.0, 1.0))
  spread_score = float(np.clip(1.0 - width_spread / MAX_WIDTH_SPREAD_M, 0.0, 1.0))
  center_prob = prob_score * width_score * spread_score
  center_valid = bool(
    prob_score >= 0.5 and
    width_score > 0.0 and
    spread_score > 0.0 and
    MIN_VALID_LANE_WIDTH_M <= lane_width_m <= MAX_VALID_LANE_WIDTH_M
  )

  return LaneCenterEstimate(left_y0, right_y0, left_prob, right_prob, center_y, lane_width_m, center_prob, center_valid)


def fill_lane_line_meta(builder, lane_lines, lane_line_probs) -> LaneCenterEstimate:
  estimate = estimate_lane_center(lane_lines, lane_line_probs)
  builder.leftY = estimate.left_y
  builder.leftProb = estimate.left_prob
  builder.rightY = estimate.right_y
  builder.rightProb = estimate.right_prob
  builder.centerY = estimate.center_y
  builder.laneWidth = estimate.lane_width
  builder.centerProb = estimate.center_prob
  builder.centerValid = estimate.center_valid
  return estimate

