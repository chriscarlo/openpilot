import numpy as np

from sunnypilot.modeld_v2.parse_model_outputs_split import Parser
from sunnypilot.models.runners.tinygrad.split_outputs import merge_split_model_outputs
from sunnypilot.models.split_model_constants import SplitModelConstants


def test_parse_policy_outputs_parses_standalone_planplus():
  raw = np.zeros((1, SplitModelConstants.IDX_N * SplitModelConstants.PLAN_WIDTH * 2), dtype=np.float32)

  parsed = Parser().parse_policy_outputs({'planplus': raw.copy()})

  expected_shape = (1, SplitModelConstants.IDX_N, SplitModelConstants.PLAN_WIDTH)
  assert parsed['planplus'].shape == expected_shape
  assert parsed['planplus_stds'].shape == expected_shape


def test_merge_split_model_outputs_includes_off_policy_outputs_without_folding_planplus():
  plan = np.ones((1, SplitModelConstants.IDX_N, SplitModelConstants.PLAN_WIDTH), dtype=np.float32)
  planplus = np.full_like(plan, 2.0)
  lane_lines = np.zeros((1, SplitModelConstants.NUM_LANE_LINES, SplitModelConstants.IDX_N, SplitModelConstants.LANE_LINES_WIDTH), dtype=np.float32)
  pose = np.zeros((1, SplitModelConstants.POSE_WIDTH), dtype=np.float32)

  merged = merge_split_model_outputs(
    {'planplus': planplus},
    {'pose': pose},
    {'plan': plan, 'lane_lines': lane_lines},
  )

  assert 'plan' in merged
  assert 'lane_lines' in merged
  assert 'planplus' in merged
  np.testing.assert_array_equal(merged['plan'], plan)
  np.testing.assert_array_equal(merged['planplus'], planplus)
