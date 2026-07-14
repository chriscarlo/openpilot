from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from openpilot.tools.vtsc.apply_physics_params import PHYSICS_KEYS, install_physics_params


REQUESTED = {
  "VisionTurnSpeedControlPhysicsAmplitude": -1.658965,
  "VisionTurnSpeedControlPhysicsSteepness": -1395.055546,
  "VisionTurnSpeedControlPhysicsCenter": 0.005397,
  "VisionTurnSpeedControlPhysicsBaseline": 4.107103,
  "VisionTurnSpeedControlPhysicsMinLatAccel": 2.4481,
  "VisionTurnSpeedControlPhysicsMaxLatAccel": 4.1071,
}


class FakeParams:
  def __init__(self, *, offroad: bool = True, fail_key: str | None = None):
    self.offroad = offroad
    self.fail_key = fail_key
    self.values = {key: 1.0 for key in PHYSICS_KEYS}

  def get_bool(self, key: str) -> bool:
    assert key == "IsOffroad"
    return self.offroad

  def get(self, key: str):
    return self.values.get(key)

  def put(self, key: str, value: float) -> None:
    self.values[key] = float(value)
    if key == self.fail_key:
      self.values[key] += 0.25
      self.fail_key = None

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


def test_installs_and_verifies_complete_sigmoid() -> None:
  params = FakeParams()
  actual = install_physics_params(params, REQUESTED)
  assert actual == REQUESTED
  assert params.values == REQUESTED


def test_refuses_to_write_while_onroad() -> None:
  params = FakeParams(offroad=False)
  before = params.values.copy()
  with pytest.raises(RuntimeError, match="offroad"):
    install_physics_params(params, REQUESTED)
  assert params.values == before


def test_readback_failure_rolls_back_all_six_values() -> None:
  failed_key = "VisionTurnSpeedControlPhysicsCenter"
  params = FakeParams(fail_key=failed_key)
  before = params.values.copy()
  with pytest.raises(RuntimeError, match="read-back"):
    install_physics_params(params, REQUESTED)
  assert params.values == before


def test_device_entrypoint_imports_with_only_repo_pythonpath() -> None:
  repository_root = Path(__file__).resolve().parents[3]
  environment = {"PATH": os.environ["PATH"], "PYTHONPATH": str(repository_root)}
  result = subprocess.run(
    [sys.executable, "tools/vtsc/apply_physics_params.py", "--help"],
    cwd=repository_root,
    env=environment,
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stderr
