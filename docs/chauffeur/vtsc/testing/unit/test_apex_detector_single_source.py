import importlib


def test_only_enhanced_apex_detector_present():
  mod = importlib.import_module("sunnypilot.selfdrive.controls.lib.vision_turn_controller")
  assert hasattr(mod, "find_apexes_enhanced"), "Enhanced apex finder missing"
  assert not hasattr(mod, "find_apexes"), "Legacy apex finder should be removed to avoid drift"

