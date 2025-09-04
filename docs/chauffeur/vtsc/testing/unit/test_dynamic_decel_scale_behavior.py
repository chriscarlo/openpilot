import math

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import dynamic_decel_scale


def test_dynamic_decel_scale_range_and_shape():
  # Range bounds
  assert math.isclose(dynamic_decel_scale(3.0), 9.0, rel_tol=0, abs_tol=1e-6)
  assert math.isclose(dynamic_decel_scale(35.0), 2.0, rel_tol=0, abs_tol=1e-6)

  # Monotonic decrease in operating range
  s10 = dynamic_decel_scale(10.0)
  s20 = dynamic_decel_scale(20.0)
  s30 = dynamic_decel_scale(30.0)
  assert 9.0 >= s10 > s20 > s30 >= 2.0

