from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_lead_brake_release_accel_floor
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_headway_follow_distance


V_EGO = 29.0
TRACK_ID = -1001


def _cfg(**updates) -> LeadResponseTuningConfig:
  return dataclasses.replace(LeadResponseTuningConfig.defaults(), **updates)


def _lead(**updates):
  values = {
    "status": True,
    "dRel": 64.0,
    "yRel": 0.0,
    "vRel": -1.2,
    "vLead": V_EGO - 1.2,
    "vLeadK": V_EGO - 1.2,
    "aLeadK": 0.0,
    "modelProb": 0.98,
    "dPath": 0.0,
    "vLat": 0.0,
    "radar": False,
    "radarTrackId": TRACK_ID,
    "fcw": False,
  }
  values.update(updates)
  return SimpleNamespace(**values)


def _wire_values(**updates):
  values = {
    "status": True,
    "dRel": 64.0,
    "yRel": 0.0,
    "vRel": -1.2,
    "vLead": V_EGO - 1.2,
    "vLeadK": V_EGO - 1.2,
    "aLeadK": 0.0,
    "modelProb": 0.98,
    "dPath": 0.0,
    "vLat": 0.0,
    "radar": False,
    "radarTrackId": TRACK_ID,
    "fcw": False,
    "fcwSuppressed": False,
    "closingGovernorRecovery": True,
    "closingGovernorRecoveryPositionClosingMps": 0.10,
    "closingGovernorRecoveryVRelFloorMps": -0.10,
    "closingGovernorRecoveryNumericValid": True,
  }
  values.update(updates)
  return values


def _radarstate(**updates):
  msg = log.RadarState.new_message()
  msg.leadOne = _wire_values(**updates)
  return msg


def _plain_radarstate(**updates):
  """Keep exact binary64 boundary values out of the Float32 wire fixture."""
  return SimpleNamespace(
    leadOne=SimpleNamespace(**_wire_values(**updates)),
    leadTwo=SimpleNamespace(status=False, radarTrackId=-1),
  )


def _duplicate_radarstate(**lead_two_updates):
  msg = _radarstate()
  msg.leadTwo = _wire_values(**lead_two_updates)
  return msg


def _mpc(*, t_follow: float = 1.5, **cfg_updates):
  return SimpleNamespace(
    vibe_controller=SimpleNamespace(is_follow_enabled=lambda: True),
    _live_tune_cfg=_cfg(**cfg_updates),
    _hyundai_ai_lead_stability_enabled=True,
    current_t_follow=t_follow,
    params=np.array([[-4.0]], dtype=float),
    x0=(0.0, V_EGO, 0.0),
    crash_cnt=0,
    lead_role_debug={
      "virtual_duplicate": {"active": False},
      "reasons": {"lead0": "center_lane", "lead1": "invalid"},
      "cutin_promoted": {"lead0": False, "lead1": False},
    },
    cutin_settle_active=False,
  )


def _apply(*, mpc=None, lead=None, radarstate=None, lead_source="lead0",
           previous_lead_source="lead0", previous_track_id=TRACK_ID,
           radarstate_updated=True, planner_fcw=False):
  mpc = _mpc(closing_recovery_bridge_max_position_closing_mps=1.25) if mpc is None else mpc
  lead = _lead() if lead is None else lead
  radarstate = _radarstate() if radarstate is None else radarstate
  control_leads = (lead, None) if lead_source == "lead0" else (None, lead)
  return get_lead_brake_release_accel_floor(
    mpc,
    v_ego=V_EGO,
    lead_source=lead_source,
    control_leads=control_leads,
    radarstate=radarstate,
    radarstate_updated=radarstate_updated,
    previous_lead_source=previous_lead_source,
    previous_track_id=previous_track_id,
    planner_fcw=planner_fcw,
  )


def test_bridge_is_default_off_and_exactly_preserves_baseline() -> None:
  default_mpc = _mpc()
  floor_default, debug_default = _apply(mpc=default_mpc)
  floor_explicit_off, debug_explicit_off = _apply(
    mpc=_mpc(closing_recovery_bridge_max_position_closing_mps=0.0),
  )

  assert floor_default == pytest.approx(floor_explicit_off)
  assert floor_default < 0.0
  assert debug_default["reason"] == debug_explicit_off["reason"] == "closing_to_target"
  assert debug_default["closing_recovery_bridge"]["reason"] == "rollback_disabled"


def test_bridge_raises_only_the_positive_release_floor_and_leaves_wire_untouched() -> None:
  radarstate = _radarstate()
  original = (float(radarstate.leadOne.vRel), float(radarstate.leadOne.vLead))

  floor, debug = _apply(radarstate=radarstate)
  bridge = debug["closing_recovery_bridge"]

  assert floor < 0.0
  assert debug["reason"] == "closing_to_target"
  assert debug["closing_mps"] == pytest.approx(1.2)
  assert bridge["candidate"] and not bridge["applied"]
  assert bridge["candidate_floor_mps2"] == pytest.approx(0.05)
  assert bridge["effective_closing_mps"] == pytest.approx(0.1)
  assert bridge["baseline_floor_mps2"] < 0.0
  assert (float(radarstate.leadOne.vRel), float(radarstate.leadOne.vLead)) == pytest.approx(original)


@pytest.mark.parametrize(("call_updates", "reason"), [
  ({"radarstate_updated": False}, "stale_radarstate"),
  ({"previous_lead_source": "cruise"}, "source_change"),
  ({"previous_track_id": -1002}, "identity_change"),
  ({"planner_fcw": True}, "fcw_or_crash"),
])
def test_frame_continuity_and_fcw_vetoes(call_updates, reason) -> None:
  floor, debug = _apply(**call_updates)
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == reason


def test_crash_and_unproven_virtual_duplicate_vetoes() -> None:
  crash_mpc = _mpc(closing_recovery_bridge_max_position_closing_mps=1.25)
  crash_mpc.crash_cnt = 1
  floor, debug = _apply(mpc=crash_mpc)
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "fcw_or_crash"

  duplicate_mpc = _mpc(closing_recovery_bridge_max_position_closing_mps=1.25)
  duplicate_mpc.lead_role_debug["virtual_duplicate"]["active"] = True
  floor, debug = _apply(mpc=duplicate_mpc)
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "unproven_virtual_duplicate"


@pytest.mark.parametrize(("wire_updates", "reason"), [
  ({"closingGovernorRecovery": False}, "inactive"),
  ({"closingGovernorRecoveryNumericValid": False}, "invalid_numeric_provenance"),
  ({"aLeadK": -0.21}, "current_braking"),
  ({"modelProb": 0.59}, "low_probability"),
  ({"dPath": 1.51}, "lateral_ambiguity"),
  ({"dPath": 1.30, "vLat": 0.70}, "lateral_ambiguity"),
  ({"closingGovernorRecoveryPositionClosingMps": 1.26}, "position_closing_veto"),
  ({"closingGovernorRecoveryVRelFloorMps": 0.10}, "recovery_floor_veto"),
  ({"closingGovernorRecoveryVRelFloorMps": -2.50}, "recovery_floor_veto"),
  ({"closingGovernorRecoveryPositionClosingMps": 0.50,
    "closingGovernorRecoveryVRelFloorMps": -0.20}, "inconsistent_recovery_floor"),
  ({"closingGovernorRecoveryVRelFloorMps": -0.80}, "recovery_close_too_fast"),
])
def test_same_frame_wire_threats_fail_closed(wire_updates, reason) -> None:
  floor, debug = _apply(radarstate=_radarstate(**wire_updates))
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == reason


def test_nonfinite_numeric_provenance_fails_closed() -> None:
  floor, debug = _apply(radarstate=_radarstate(closingGovernorRecoveryVRelFloorMps=float("nan")))
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "nonfinite_numeric_provenance"


def test_short_published_ttc_fails_closed() -> None:
  mpc = _mpc(t_follow=0.5, closing_recovery_bridge_max_position_closing_mps=1.25)
  lead = _lead(dRel=24.0, vRel=-2.0, vLead=V_EGO - 2.0)
  radarstate = _radarstate(dRel=24.0, vRel=-2.0, vLead=V_EGO - 2.0)
  floor, debug = _apply(mpc=mpc, lead=lead, radarstate=radarstate)
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["published_ttc_s"] is None
  assert debug["closing_recovery_bridge"]["reason"] == "control_short_ttc"

  # A calmer control copy cannot override a more urgent current wire copy.
  lead = _lead(dRel=30.0, vRel=-2.0, vLead=V_EGO - 2.0)
  floor, debug = _apply(mpc=mpc, lead=lead, radarstate=radarstate)
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["published_ttc_s"] == pytest.approx(12.0)
  assert debug["closing_recovery_bridge"]["reason"] == "short_published_ttc"


def test_inside_target_near_threat_and_recovery_target_vetoes() -> None:
  target = get_headway_follow_distance(V_EGO, 1.5)

  floor, debug = _apply(
    lead=_lead(dRel=target - 0.1),
    radarstate=_radarstate(dRel=target - 0.1),
  )
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "control_inside_target"

  floor, debug = _apply(
    mpc=_mpc(t_follow=0.2, closing_recovery_bridge_max_position_closing_mps=1.25),
    lead=_lead(dRel=15.0),
    radarstate=_radarstate(dRel=15.0),
  )
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "control_near_threat"

  near_recovery_drel = target + 0.35
  floor, debug = _apply(
    lead=_lead(dRel=near_recovery_drel),
    radarstate=_radarstate(
      dRel=near_recovery_drel,
      closingGovernorRecoveryPositionClosingMps=0.20,
      closingGovernorRecoveryVRelFloorMps=-0.20,
    ),
  )
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "recovery_target_near"


def test_track_provenance_must_be_unique_model_owned_and_need_correction() -> None:
  radarstate = _radarstate()
  radarstate.leadTwo = {
    "status": True,
    "dRel": 64.0,
    "vRel": -1.2,
    "vLead": V_EGO - 1.2,
    "radarTrackId": TRACK_ID,
  }
  floor, debug = _apply(radarstate=radarstate)
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "ambiguous_wire_identity"

  floor, debug = _apply(lead=_lead(radarTrackId=-1))
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "identity_change"

  floor, debug = _apply(lead=_lead(vRel=-0.1, vLead=V_EGO - 0.1))
  assert floor > 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "already_at_recovery_floor"
  assert not debug["closing_recovery_bridge"]["applied"]


@pytest.mark.parametrize(("lead_updates", "reason"), [
  ({"aLeadK": -0.30}, "control_lead_braking"),
  ({"modelProb": 0.59}, "control_low_probability"),
  ({"dPath": 1.51}, "control_lateral_ambiguity"),
  ({"dPath": 1.30, "vLat": 0.70}, "control_lateral_ambiguity"),
  ({"fcw": True}, "control_lead_fcw"),
  ({"aLeadK": float("nan")}, "invalid_control_evidence"),
  ({"vRel": float("nan")}, "invalid_control_evidence"),
  ({"vLead": float("nan")}, "invalid_control_evidence"),
  ({"dRel": float("nan")}, "invalid_control_evidence"),
])
def test_more_urgent_control_view_vetoes_calm_wire(lead_updates, reason) -> None:
  floor, debug = _apply(lead=_lead(**lead_updates))
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == reason
  assert not debug["closing_recovery_bridge"]["candidate"]


def test_control_geometry_and_brake_authority_veto_candidate() -> None:
  target = get_headway_follow_distance(V_EGO, 1.5)
  floor, debug = _apply(
    lead=_lead(dRel=target - 0.1),
    radarstate=_radarstate(dRel=64.0),
  )
  assert floor is None or floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "control_inside_target"

  mpc = _mpc(closing_recovery_bridge_max_position_closing_mps=1.25)
  mpc.params[0, 0] = -0.5
  d_rel = target + 2.0
  lead = _lead(dRel=d_rel, vRel=-2.4, vLead=V_EGO - 2.4)
  radarstate = _radarstate(
    dRel=d_rel,
    vRel=-2.4,
    vLead=V_EGO - 2.4,
    closingGovernorRecoveryPositionClosingMps=0.10,
    closingGovernorRecoveryVRelFloorMps=-0.10,
  )
  floor, debug = _apply(mpc=mpc, lead=lead, radarstate=radarstate)
  assert floor is None
  assert debug["reason"] == "brake_authority_deficit"
  assert debug["closing_recovery_bridge"]["reason"] == "baseline_brake_authority_deficit"
  assert not debug["closing_recovery_bridge"]["candidate"]


def _proven_duplicate_mpc():
  mpc = _mpc(closing_recovery_bridge_max_position_closing_mps=1.25)
  mpc.lead_role_debug = {
    "virtual_duplicate": {
      "active": True,
      "selected_raw_slot": 0,
      "suppressed_raw_slot": 1,
    },
    "reasons": {"lead0": "virtual_duplicate_raw_0", "lead1": "suppressed_duplicate"},
    "cutin_promoted": {"lead0": False, "lead1": False},
  }
  return mpc


def test_only_route_proven_slot_zero_duplicate_is_reconciled() -> None:
  floor, debug = _apply(mpc=_proven_duplicate_mpc(), radarstate=_duplicate_radarstate(vLat=3.5))
  bridge = debug["closing_recovery_bridge"]
  assert floor < 0.0
  assert bridge["candidate"]
  assert bridge["duplicate_reconciled"]
  assert bridge["wire_slot"] == 0


@pytest.mark.parametrize("mutation", [
  "slot_one_selected",
  "wrong_source_reason",
  "cutin_promoted",
  "cutin_settle",
  "physical_mismatch",
  "numeric_valid_mismatch",
  "recovery_floor_mismatch",
  "fcw_suppression_mismatch",
  "current_threat_mismatch",
])
def test_every_duplicate_proof_invariant_fails_closed(mutation) -> None:
  mpc = _proven_duplicate_mpc()
  radarstate = _duplicate_radarstate(vLat=3.5)
  if mutation == "slot_one_selected":
    mpc.lead_role_debug["virtual_duplicate"].update(selected_raw_slot=1, suppressed_raw_slot=0)
  elif mutation == "wrong_source_reason":
    mpc.lead_role_debug["reasons"]["lead0"] = "center_lane"
  elif mutation == "cutin_promoted":
    mpc.lead_role_debug["cutin_promoted"]["lead0"] = True
  elif mutation == "cutin_settle":
    mpc.cutin_settle_active = True
  elif mutation == "physical_mismatch":
    radarstate.leadTwo.vRel = -1.1
  elif mutation == "numeric_valid_mismatch":
    radarstate.leadTwo.closingGovernorRecoveryNumericValid = False
  elif mutation == "recovery_floor_mismatch":
    radarstate.leadTwo.closingGovernorRecoveryVRelFloorMps = -0.11
  elif mutation == "fcw_suppression_mismatch":
    radarstate.leadTwo.fcwSuppressed = True
  elif mutation == "current_threat_mismatch":
    radarstate.leadTwo.steadyParityCurrentThreat = True

  floor, debug = _apply(mpc=mpc, radarstate=radarstate)
  assert floor < 0.0
  assert debug["closing_recovery_bridge"]["reason"] == "unproven_virtual_duplicate"
  assert not debug["closing_recovery_bridge"]["candidate"]


def _output_planner(*, output=-0.60, should_stop=False, fcw=False,
                    release_elapsed_s=0.05):
  return SimpleNamespace(
    lead_brake_release_debug={
      "closing_recovery_bridge": {
        "candidate": True,
        "applied": False,
        "reason": "candidate",
        "candidate_floor_mps2": 0.05,
        "output_uplift_applied_mps2": 0.0,
      },
    },
    output_should_stop=should_stop,
    fcw=fcw,
    output_a_target=output,
    mpc=_mpc(closing_recovery_bridge_max_position_closing_mps=1.25),
    _planner_output_accel_limits=(-4.0, 2.0),
    dt=0.05,
    _positive_release_elapsed_s=release_elapsed_s,
  )


def _apply_output(planner, *, previous=-0.70, force_decel=False):
  LongitudinalPlanner._apply_closing_recovery_output_uplift(
    planner,
    previous_published_output=previous,
    force_slow_decel=force_decel,
  )


def test_post_limiter_uplift_is_positive_capped_and_release_slewed() -> None:
  planner = _output_planner()
  _apply_output(planner, previous=-0.60)
  assert planner.output_a_target == pytest.approx(-0.50)
  assert planner.lead_brake_release_debug["closing_recovery_bridge"]["applied"]

  planner = _output_planner()
  _apply_output(planner, previous=-0.65)
  assert planner.output_a_target == pytest.approx(-0.55)
  assert -0.60 <= planner.output_a_target <= -0.50


def test_post_limiter_uplift_uses_elapsed_positive_release_step() -> None:
  planner = _output_planner(release_elapsed_s=0.028)
  _apply_output(planner, previous=-0.60)

  debug = planner.lead_brake_release_debug["closing_recovery_bridge"]
  assert planner.output_a_target == pytest.approx(-0.60 + 2.0 * 0.028)
  assert debug["release_elapsed_s"] == pytest.approx(0.028)
  assert debug["release_max_step_mps2"] == pytest.approx(2.0 * 0.028)
  assert debug["applied"]


def test_post_limiter_steady_parity_threat_veto_preserves_exact_rollback() -> None:
  planner = _output_planner()
  rollback = float(planner.output_a_target)

  LongitudinalPlanner._apply_closing_recovery_output_uplift(
    planner,
    previous_published_output=-0.70,
    steady_parity_current_threat=True,
  )

  debug = planner.lead_brake_release_debug["closing_recovery_bridge"]
  assert planner.output_a_target == rollback
  assert debug["reason"] == "steady_parity_current_threat"
  assert not debug["applied"]


def test_bridge_safety_threshold_equalities_and_adjacent_values() -> None:
  target_gap = get_headway_follow_distance(V_EGO, 1.5)

  _, equal_gap = _apply(lead=_lead(dRel=target_gap))
  _, above_gap = _apply(lead=_lead(dRel=math.nextafter(target_gap, math.inf)))
  assert equal_gap["closing_recovery_bridge"]["reason"] == "control_inside_target"
  assert above_gap["closing_recovery_bridge"]["reason"] == "control_recovery_target_near"

  _, equal_close = _apply(lead=_lead(vRel=-2.5, vLead=V_EGO))
  _, below_close = _apply(lead=_lead(vRel=-math.nextafter(2.5, 0.0), vLead=V_EGO))
  assert equal_close["closing_recovery_bridge"]["reason"] == "control_fast_close"
  assert below_close["closing_recovery_bridge"]["reason"] == "candidate"

  _, equal_calm = _apply(lead=_lead(aLeadK=-0.15))
  _, below_calm = _apply(lead=_lead(aLeadK=math.nextafter(-0.15, -math.inf)))
  assert equal_calm["closing_recovery_bridge"]["reason"] == "candidate"
  assert below_calm["closing_recovery_bridge"]["reason"] == "control_lead_braking"

  equal_hard = _output_planner(output=-1.5)
  _apply_output(equal_hard, previous=-1.5)
  assert equal_hard.output_a_target == -1.5
  assert equal_hard.lead_brake_release_debug["closing_recovery_bridge"]["reason"] == "hard_requested_decel"

  above_hard = _output_planner(output=math.nextafter(-1.5, math.inf))
  _apply_output(above_hard, previous=above_hard.output_a_target)
  assert above_hard.output_a_target > -1.5
  assert above_hard.lead_brake_release_debug["closing_recovery_bridge"]["reason"] == "applied"


def test_position_closing_limit_includes_equality_and_rejects_smaller_tune() -> None:
  lead = _lead(vRel=-1.5, vLead=V_EGO - 1.5)
  radarstate = _plain_radarstate(
    vRel=-1.5,
    vLead=V_EGO - 1.5,
    closingGovernorRecoveryPositionClosingMps=1.25,
    closingGovernorRecoveryVRelFloorMps=-1.25,
  )
  equal_limit = _mpc(
    closing_recovery_bridge_max_position_closing_mps=1.25,
    lead_brake_release_near_target_max_closing_mps=1.25,
  )
  _, equal_debug = _apply(mpc=equal_limit, lead=lead, radarstate=radarstate)
  assert equal_debug["closing_recovery_bridge"]["reason"] == "candidate"

  below_limit = _mpc(
    closing_recovery_bridge_max_position_closing_mps=math.nextafter(1.25, 0.0),
    lead_brake_release_near_target_max_closing_mps=1.25,
  )
  _, below_debug = _apply(mpc=below_limit, lead=lead, radarstate=radarstate)
  assert below_debug["closing_recovery_bridge"]["reason"] == "position_closing_veto"


@pytest.mark.parametrize(("updates", "force_decel", "reason"), [
  ({"output": -2.0}, False, "hard_requested_decel"),
  ({"should_stop": True}, False, "should_stop"),
  ({"fcw": True}, False, "planner_fcw"),
  ({}, True, "force_decel"),
])
def test_post_limiter_safety_vetoes_preserve_exact_rollback(updates, force_decel, reason) -> None:
  planner = _output_planner(**updates)
  rollback = float(planner.output_a_target)
  _apply_output(planner, force_decel=force_decel)
  assert planner.output_a_target == rollback
  assert planner.lead_brake_release_debug["closing_recovery_bridge"]["reason"] == reason
  assert not planner.lead_brake_release_debug["closing_recovery_bridge"]["applied"]


@pytest.mark.parametrize("bypass_floor", [0.0, -5.0])
def test_fixed_hard_brake_veto_cannot_be_disabled_by_comfort_tunes(bypass_floor) -> None:
  planner = _output_planner(output=-2.0)
  planner.mpc._live_tune_cfg = _cfg(
    closing_recovery_bridge_max_position_closing_mps=1.25,
    comfort_jerk_bypass_decel_mps2=bypass_floor,
    cruise_relatch_bypass_decel_mps2=bypass_floor,
  )
  _apply_output(planner, previous=-2.0)
  assert planner.output_a_target == -2.0
  assert planner.lead_brake_release_debug["closing_recovery_bridge"]["reason"] == "hard_requested_decel"
  assert not planner.lead_brake_release_debug["closing_recovery_bridge"]["applied"]
