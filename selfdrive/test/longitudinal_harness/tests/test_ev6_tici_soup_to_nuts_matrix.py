"""Full active-tici EV6 topology smoke matrix for every synthetic scenario.

Scenario-specific physics oracles live beside their focused regressions. This
matrix prevents any scenario from quietly falling back to direct perception,
planner-only output, or a passthrough controller while still being cited as
end-to-end evidence.
"""
from __future__ import annotations

import math

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import SCENARIO_NAMES, build_synthetic_scenario

SCENARIO_DURATION_S = 30.0


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_every_synthetic_scenario_runs_through_active_ev6_tici_path(scenario_name: str) -> None:
  initial_speed, initial_accel, steps = build_synthetic_scenario(
    scenario_name,
    duration_s=SCENARIO_DURATION_S,
    dt_s=DT_MDL,
  )
  result = run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name=f"ev6_tici_soup_to_nuts_{scenario_name}",
    steps=steps,
    initial_speed_mps=initial_speed,
    initial_accel_mps2=initial_accel,
    noise_profile="ev6_measured",
    seed=0,
  )

  assert result.vehicle["candidate"] == "KIA_EV6"
  assert result.vehicle["topology"] == "lka"
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "ev6_measured"
  assert result.vehicle["radarUnavailable"] is True
  assert result.vehicle["openpilotLongitudinalControl"] is True
  assert result.vehicle["pcmCruise"] is False
  assert result.vehicle["longitudinalActuatorDelay"] == 0.5

  rows = result.trace
  assert rows
  assert any(r["has_any_lead"] for r in rows)
  assert any(r["lead_one_radard_debug"]["track_id"] is not None or
             r["lead_two_radard_debug"]["track_id"] is not None for r in rows)
  assert any(r["lead_one_published_d_rel_m"] is not None or
             r["lead_two_published_d_rel_m"] is not None for r in rows)

  # Each layer produced finite runtime output. Scenario-specific tests assert
  # the correct physical response; this guards the topology itself.
  for row in rows:
    for key in (
      "planner_accel_mps2",
      "longcontrol_accel_mps2",
      "controller_accel_mps2",
      "delayed_command_mps2",
      "realized_accel_mps2",
      "v_ego_true_mps",
    ):
      assert math.isfinite(float(row[key])), f"{scenario_name}: non-finite {key} at t={row['t_s']}"
