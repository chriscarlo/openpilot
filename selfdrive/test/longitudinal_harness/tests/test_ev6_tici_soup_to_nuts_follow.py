"""Soup-to-nuts EV6 follow regression through the complete active tici path.

The fixture intentionally includes measured no-radar model noise and traverses:
modelV2 lead -> real radard tracker -> Hyundai lead classification/MPC ->
post-MPC planner limits -> LongControl -> Hyundai no-radar accel shaping ->
command delay/vehicle response.  A direct-lead or planner-only simulation is
not an acceptable substitute for this test.
"""
from __future__ import annotations

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

EGO_V0_MPS = 27.5
LEAD_V_MPS = 25.5
CRUISE_MPS = 32.0
GAP0_M = 115.0
DISPLAY_HEADWAY_S = 1.70
DURATION_S = 50.0


def _steps() -> list[StepInput]:
  return [
    StepInput(
      t_s=i * DT_MDL,
      cruise_speed_mps=CRUISE_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=LEAD_V_MPS,
        model_prob_target=0.95,
        d_rel_override_m=GAP0_M if i == 0 else None,
        acquisition_reset=i == 0,
      ),
      note="steady far slower lead through complete EV6 tici path",
    )
    for i in range(round(DURATION_S / DT_MDL))
  ]


def test_ev6_tici_path_converges_to_displayed_headway_without_weakening_safety() -> None:
  params = {f"VibeTune.Follow.Standard.Headway{i}": str(DISPLAY_HEADWAY_S) for i in range(4)}
  result = run_harness(
    vehicle_config=resolve_ev6_vehicle_config(param_overrides=params),
    scenario_name="ev6_tici_soup_to_nuts_follow",
    steps=_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    seed=0,
  )
  rows = result.trace[::5]

  # Active tici configuration and every runtime layer must be present.
  assert result.vehicle["candidate"] == "KIA_EV6"
  assert result.vehicle["topology"] == "lka"
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "ev6_measured"
  assert result.vehicle["radarUnavailable"] is True
  assert result.vehicle["openpilotLongitudinalControl"] is True
  assert result.vehicle["pcmCruise"] is False
  assert result.vehicle["longitudinalActuatorDelay"] == 0.5

  # Real radard filtering, LongControl, Hyundai shaping, delay, and plant
  # response are all exercised rather than short-circuited by the fixture.
  published = [r for r in rows if r["lead_one_published_d_rel_m"] is not None]
  assert published
  assert max(abs(r["lead_one_raw_d_rel_m"] - r["lead_one_published_d_rel_m"]) for r in published) > 5.0
  assert max(abs(r["longcontrol_accel_mps2"] - r["planner_accel_mps2"]) for r in rows) > 0.05
  assert max(abs(r["controller_accel_mps2"] - r["longcontrol_accel_mps2"]) for r in rows) > 0.05
  assert max(abs(r["delayed_command_mps2"] - r["controller_accel_mps2"]) for r in rows) > 0.05
  assert max(abs(r["realized_accel_mps2"] - r["delayed_command_mps2"]) for r in rows) > 0.05

  # Weak, non-braking position noise must not fabricate a persistent closing
  # lead and make the post-MPC slowdown ceiling hold an oversized gap. The full
  # stopping-equivalence geometry and corroborated-closing authority remain in
  # force, while the complete path converges to the displayed follow target.
  tail = [r for r in rows if r["t_s"] >= DURATION_S - 5.0]
  tail_thw = [r["lead_one_true_d_rel_m"] / r["v_ego_true_mps"] for r in tail]
  assert sum(tail_thw) / len(tail_thw) <= DISPLAY_HEADWAY_S + 0.20
  assert tail_thw[-1] <= DISPLAY_HEADWAY_S + 0.20

  # Do not buy closer following by spending collision margin or creating a
  # catch-up overshoot into the lead.
  assert result.summary["minTrueGapM"] >= 40.0
  assert max(r["mpc_crash_cnt"] for r in rows) == 0.0
  assert not any(r["planner_fcw"] for r in rows)
  assert min(r["v_ego_true_mps"] for r in rows) >= LEAD_V_MPS - 0.35
