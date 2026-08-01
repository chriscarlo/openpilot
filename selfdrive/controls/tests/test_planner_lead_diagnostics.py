from types import SimpleNamespace

import pytest

from cereal import messaging
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import build_lead_diagnostics_snapshot


def _lead_payload(*, v_rel: float, v_lead: float, track_id: int = -1032):
  return {
    "status": True,
    "dRel": 40.5,
    "vRel": v_rel,
    "aRel": 0.1,
    "vLead": v_lead,
    "vLeadK": v_lead - 0.1,
    "aLeadK": 0.02,
    "modelProb": 0.98,
    "radar": False,
    "radarTrackId": track_id,
    "fcw": False,
    "closingGovernorRecovery": False,
    "steadyParityCurrentThreat": False,
    "accelCorrRawHardBraking": False,
  }


def test_planner_private_lead_diagnostics_serialize_input_virtual_and_decisions():
  mpc = SimpleNamespace(
    source="lead0",
    selected_obstacle_m=58.0,
    hyundai_virtual_lead_debug={
      "active": True,
      "source": "lead0",
      "input": _lead_payload(v_rel=0.35, v_lead=30.35),
      "filtered": _lead_payload(v_rel=-1.55, v_lead=28.45),
      "opening_recovery": {
        "active": True,
        "candidate": True,
        "confirm_frames": 3,
        "tau_s": 0.30,
        "reason": "confirmed_opening",
      },
    },
    acc_source_debug={
      "reason": "lead_hold",
      "best_lead_obstacle": 251.386,
      "filtered_lead_obstacle": 233.845,
      "cruise_obstacle": 257.204,
      "raw_gap_surplus_m": -4.214,
      "filtered_gap_surplus_m": -5.100,
    },
    lead_stability_debug={
      "slot0": {
        "accel_corr_clamped": False,
        "accel_corr_amplified": False,
      },
    },
    lead_slowdown_accel_ceiling=0.08,
  )
  planner = SimpleNamespace(
    mpc=mpc,
    lead_brake_release_debug={"reason": "inside_target", "floor_mps2": -0.05},
  )

  snapshot = build_lead_diagnostics_snapshot(planner)
  message = messaging.new_message("longitudinalPlanSP")
  message.longitudinalPlanSP.leadDiagnostics = snapshot
  diagnostics = message.as_reader().longitudinalPlanSP.leadDiagnostics

  assert diagnostics.valid is True
  assert diagnostics.version == 1
  assert diagnostics.source == "lead0"
  assert diagnostics.input.vRelMps == pytest.approx(0.35)
  assert diagnostics.virtual.vRelMps == pytest.approx(-1.55)
  assert diagnostics.inputObstacleM == pytest.approx(251.386, abs=1e-4)
  assert diagnostics.virtualObstacleM == pytest.approx(233.845, abs=1e-4)
  assert diagnostics.selectedObstacleM == pytest.approx(58.0)
  assert diagnostics.inputGapSurplusM == pytest.approx(-4.214, abs=1e-4)
  assert diagnostics.slowdownCeilingValid is True
  assert diagnostics.slowdownCeilingMps2 == pytest.approx(0.08)
  assert diagnostics.releaseFloorValid is True
  assert diagnostics.releaseFloorMps2 == pytest.approx(-0.05)
  assert diagnostics.openingRecoveryActive is True
  assert diagnostics.openingRecoveryConfirmFrames == 3
  assert diagnostics.openingRecoveryTauS == pytest.approx(0.30)
  assert diagnostics.openingRecoveryReason == "confirmed_opening"
