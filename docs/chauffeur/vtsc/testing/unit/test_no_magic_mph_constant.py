from pathlib import Path


def test_no_literal_mph_magic_constant():
  """Ensure the VTSC module uses CV.MS_TO_MPH instead of a hardcoded 2.237."""
  path = Path("sunnypilot/selfdrive/controls/lib/vision_turn_controller.py")
  text = path.read_text(encoding="utf-8")
  assert "2.237" not in text, "Found hardcoded 2.237 MPH conversion in VTSC module"

