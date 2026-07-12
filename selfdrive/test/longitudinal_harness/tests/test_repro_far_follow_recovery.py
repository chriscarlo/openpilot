"""Regression for the EV6 far-lead brake/open-gap/crawl-back pathology.

Before the fix, this deterministic tici-fidelity run adopted one measured
far-range inward dRel outlier, retained a roughly 12 m wrong-too-close track,
held a fabricated 1.8 m/s closure after raw vRel recovered, slowed ego to
24.22 m/s behind a steady 25.5 m/s lead, and ended roughly 12 m beyond its
1.70 s target. The direct-perception sim hid both radard state failures.
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
TARGET_HEADWAY_S = 1.70
DURATION_S = 40.0
SEED = 0


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
      note="steady far slower lead with measured EV6 perception noise",
    )
    for i in range(round(DURATION_S / DT_MDL))
  ]


def _target_gap(row: dict) -> float:
  return TARGET_HEADWAY_S * float(row["v_ego_true_mps"]) + 6.0


def test_far_lead_outlier_recovers_without_braking_below_lead_speed() -> None:
  params = {f"VibeTune.Follow.Standard.Headway{i}": str(TARGET_HEADWAY_S) for i in range(4)}
  result = run_harness(
    vehicle_config=resolve_ev6_vehicle_config(param_overrides=params),
    scenario_name="far_follow_recovery",
    steps=_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    seed=SEED,
  )
  rows = result.trace[::5]

  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "ev6_measured"

  published_rows = [r for r in rows if r["lead_one_published_d_rel_m"] is not None]
  deficits = [r["lead_one_published_d_rel_m"] - r["lead_one_true_d_rel_m"] for r in published_rows]
  assert min(deficits) < -10.0, "fixture did not exercise the measured inward dRel collapse"
  collapse_idx = next(
    i for i, (row, deficit) in enumerate(zip(published_rows, deficits, strict=True))
    if row["t_s"] > 25.0 and deficit < -8.0
  )
  collapse_t = published_rows[collapse_idx]["t_s"]
  recovery_t = next(
    r["t_s"] for r in published_rows[collapse_idx:]
    if r["lead_one_published_d_rel_m"] - r["lead_one_true_d_rel_m"] > -5.0
  )
  assert recovery_t - collapse_t <= 1.0

  braking_open_gap = [
    r for r in rows
    if r["lead_one_true_d_rel_m"] - _target_gap(r) > 2.0
    and r["v_ego_true_mps"] <= LEAD_V_MPS
    and r["planner_accel_mps2"] < -0.1
  ]
  assert not braking_open_gap
  assert min(r["v_ego_true_mps"] for r in rows) >= LEAD_V_MPS - 0.35

  final = rows[-1]
  assert final["v_ego_true_mps"] > LEAD_V_MPS
  assert final["lead_one_true_d_rel_m"] - _target_gap(final) <= 6.0
  assert result.summary["minTrueGapM"] >= 45.0
