from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


DT = 0.05
COMFORT_JERK = 0.4


def _make_stub(*, prev_a: float = 0.0, owner: str = "", source: str = "lead0",
               keepup_max_accel: float = 0.22):
  cfg = replace(
    LeadResponseTuningConfig.defaults(),
    comfort_jerk_limit_mps3=COMFORT_JERK,
    comfort_jerk_bypass_decel_mps2=-1.5,
    cruise_relatch_bypass_decel_mps2=-1.5,
    lead_keepup_max_accel=keepup_max_accel,
  )
  lead = SimpleNamespace(
    status=True,
    fcw=False,
    dRel=60.0,
    vRel=0.0,
    vLead=27.0,
    aLeadK=0.0,
  )
  stub = SimpleNamespace(
    dt=DT,
    output_a_target=prev_a,
    _comfort_jerk_prev_a=prev_a,
    _comfort_jerk_prev_src=source,
    _comfort_upward_floor_owner=owner,
    _comfort_upward_slew_frames=0,
    _comfort_upward_prev_owner="",
    _flutter_mode_active=False,
    handoff_limit_debug={"active": False, "clipped": False},
    relatch_blend_debug={"active": False},
    comfort_jerk_debug={},
    mpc=SimpleNamespace(_live_tune_cfg=cfg, current_t_follow=1.45),
  )
  stub._lead_owned_slot = LongitudinalPlanner._lead_owned_slot
  stub._relatch_urgency_bypass = MethodType(LongitudinalPlanner._relatch_urgency_bypass, stub)
  stub._apply_comfort_jerk_envelope = MethodType(LongitudinalPlanner._apply_comfort_jerk_envelope, stub)
  return stub, (lead,)


@pytest.mark.parametrize("owner", ["lead_keepup", "brake_release"])
def test_discrete_floor_owned_upward_step_is_comfort_slewed(owner: str) -> None:
  stub, leads = _make_stub(owner=owner)
  stub.output_a_target = 0.22

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(COMFORT_JERK * DT)
  assert stub.comfort_jerk_debug["clipped"] is True
  assert stub.comfort_jerk_debug["upward_floor_owner"] == owner


def test_ordinary_mpc_upward_step_remains_unmodified() -> None:
  stub, leads = _make_stub(owner="")
  stub.output_a_target = 0.22

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(0.22)
  assert stub.comfort_jerk_debug["clipped"] is False


def test_large_floor_owned_recovery_passes_immediately() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="brake_release")
  stub.output_a_target = 0.2

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(0.2)
  assert stub.comfort_jerk_debug["clipped"] is False


def test_live_keepup_raise_cannot_expand_micro_recovery_ceiling() -> None:
  stub, leads = _make_stub(owner="lead_keepup", keepup_max_accel=1.0)
  stub.output_a_target = 0.4

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(0.4)
  assert stub.comfort_jerk_debug["upward_max_delta_mps2"] == pytest.approx(0.30)
  assert stub.comfort_jerk_debug["clipped"] is False


def test_persistent_micro_floor_rollon_ramps_to_full_authority_quickly() -> None:
  stub, leads = _make_stub(owner="lead_keepup")
  outputs = []
  for _ in range(4):
    stub.output_a_target = 0.22
    stub._apply_comfort_jerk_envelope("lead0", leads)
    outputs.append(stub.output_a_target)

  assert outputs == pytest.approx([0.02, 0.08, 0.18, 0.22])


def test_benign_downward_step_remains_comfort_slewed() -> None:
  stub, leads = _make_stub(owner="")
  stub.output_a_target = -0.5

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(-COMFORT_JERK * DT)
  assert stub.comfort_jerk_debug["clipped"] is True


def test_requested_emergency_brake_bypasses_envelope() -> None:
  stub, leads = _make_stub(prev_a=0.2, owner="")
  stub.output_a_target = -2.0

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(-2.0)
  assert stub.comfort_jerk_debug["bypassed"] is True
  assert stub.comfort_jerk_debug["bypass_reason"] == "requested_decel"
