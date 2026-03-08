from types import SimpleNamespace

import pytest

from sunnypilot.models.runners.tinygrad.compat import get_captured_input_info


def test_get_captured_input_info_prefers_current_field():
  captured = SimpleNamespace(
    expected_input_info=[("shape", "st", "new_dtype", "new_device")],
    expected_st_vars_dtype_device=[("shape", "st", "old_dtype", "old_device")],
  )
  assert get_captured_input_info(captured) == captured.expected_input_info


def test_get_captured_input_info_falls_back_to_legacy_field():
  captured = SimpleNamespace(
    expected_st_vars_dtype_device=[("shape", "st", "legacy_dtype", "legacy_device")],
  )
  assert get_captured_input_info(captured) == captured.expected_st_vars_dtype_device


def test_get_captured_input_info_requires_metadata():
  with pytest.raises(AttributeError, match="missing expected input metadata"):
    get_captured_input_info(SimpleNamespace())
