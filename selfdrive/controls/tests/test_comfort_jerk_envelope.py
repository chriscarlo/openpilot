from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


DT = 0.05
COMFORT_JERK = 0.4


def _make_stub(*, prev_a: float = 0.0, owner: str = "", source: str = "lead0",
               keepup_max_accel: float = 0.22, release_jerk: float = 2.0,
               track_id: int = -1035, hyundai_enabled: bool = True,
               release_elapsed_s: float | None = DT):
  cfg = replace(
    LeadResponseTuningConfig.defaults(),
    comfort_jerk_limit_mps3=COMFORT_JERK,
    comfort_jerk_bypass_decel_mps2=-1.5,
    cruise_relatch_bypass_decel_mps2=-1.5,
    lead_keepup_max_accel=keepup_max_accel,
    lead_brake_release_jerk_mps3=release_jerk,
  )
  lead = SimpleNamespace(
    status=True,
    fcw=False,
    dRel=60.0,
    vRel=0.0,
    vLead=27.0,
    aLeadK=0.0,
    radarTrackId=track_id,
  )
  stub = SimpleNamespace(
    dt=DT,
    output_a_target=prev_a,
    _comfort_jerk_prev_a=prev_a,
    _comfort_jerk_prev_src=source,
    _comfort_jerk_prev_track_id=track_id,
    _brake_release_slew_active=False,
    _positive_release_elapsed_s=release_elapsed_s,
    _comfort_upward_floor_owner=owner,
    _flutter_mode_active=False,
    handoff_limit_debug={"active": False, "clipped": False},
    relatch_blend_debug={"active": False},
    comfort_jerk_debug={},
    mpc=SimpleNamespace(
      _live_tune_cfg=cfg,
      current_t_follow=1.45,
      _hyundai_ai_lead_stability_enabled=hyundai_enabled,
    ),
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


def test_low_speed_legacy_large_floor_owned_recovery_passes_immediately() -> None:
  # Omitting v_ego intentionally exercises the method's 0 m/s default: below
  # LeadBrakeReleaseMinSpeedMps the new release leg is disabled and the legacy
  # large-recovery pass-through remains exact.
  stub, leads = _make_stub(prev_a=-0.8, owner="brake_release")
  stub.output_a_target = 0.2

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(0.2)
  assert stub.comfort_jerk_debug["clipped"] is False


def test_non_hyundai_brake_release_remains_exact_legacy() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="", hyundai_enabled=False)
  stub.output_a_target = 0.2

  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)

  assert stub.output_a_target == pytest.approx(0.2)
  assert stub._brake_release_slew_active is False
  assert stub.comfort_jerk_debug["release_slew_clipped"] is False


def test_same_track_brake_release_uses_positive_only_release_jerk() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="brake_release")
  outputs = []
  for _ in range(9):
    stub.output_a_target = 0.05
    stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
    outputs.append(stub.output_a_target)

  assert outputs == pytest.approx([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05])
  assert max(b - a for a, b in zip([-0.8, *outputs[:-1]], outputs, strict=True)) == pytest.approx(2.0 * DT)
  assert stub.comfort_jerk_debug["release_slew_active"] is False
  assert stub.comfort_jerk_debug["release_slew_clipped"] is False


def test_same_track_brake_release_uses_elapsed_cycle_time() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="", release_elapsed_s=0.028)
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
  first_output = stub.output_a_target

  assert first_output == pytest.approx(-0.8 + 2.0 * 0.028)
  assert stub.comfort_jerk_debug["release_elapsed_s"] == pytest.approx(0.028)
  assert stub.comfort_jerk_debug["release_max_step_mps2"] == pytest.approx(2.0 * 0.028)

  stub._positive_release_elapsed_s = 0.071
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)

  assert stub.output_a_target == pytest.approx(first_output + 2.0 * 0.071)
  assert stub.comfort_jerk_debug["release_elapsed_s"] == pytest.approx(0.071)


@pytest.mark.parametrize("elapsed_s", [None, 0.0, -0.01, 0.201, float("inf"), float("nan")])
def test_brake_release_invalid_or_stale_elapsed_uses_nominal_fallback(elapsed_s) -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="", release_elapsed_s=elapsed_s)
  stub.output_a_target = 0.2

  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)

  assert stub.output_a_target == pytest.approx(-0.8 + 2.0 * DT)
  assert stub.comfort_jerk_debug["release_elapsed_s"] == pytest.approx(DT)


def test_positive_release_clock_uses_model_monotonic_cadence_and_resets_gaps() -> None:
  stub = SimpleNamespace(
    dt=DT,
    _positive_release_prev_model_mono_ns=0,
    _positive_release_elapsed_s=DT,
  )

  LongitudinalPlanner._refresh_positive_release_elapsed(
    stub, SimpleNamespace(logMonoTime={"modelV2": 1_000_000_000}),
  )
  assert stub._positive_release_elapsed_s == pytest.approx(DT)

  LongitudinalPlanner._refresh_positive_release_elapsed(
    stub, SimpleNamespace(logMonoTime={"modelV2": 1_028_000_000}),
  )
  assert stub._positive_release_elapsed_s == pytest.approx(0.028)

  # A stale gap and a non-monotonic clock both retain the conservative nominal
  # step, while anchoring the next valid cycle to the latest positive clock.
  LongitudinalPlanner._refresh_positive_release_elapsed(
    stub, SimpleNamespace(logMonoTime={"modelV2": 1_500_000_000}),
  )
  assert stub._positive_release_elapsed_s == pytest.approx(DT)
  LongitudinalPlanner._refresh_positive_release_elapsed(
    stub, SimpleNamespace(logMonoTime={"modelV2": 1_400_000_000}),
  )
  assert stub._positive_release_elapsed_s == pytest.approx(DT)
  LongitudinalPlanner._refresh_positive_release_elapsed(
    stub, SimpleNamespace(logMonoTime={"modelV2": 1_450_000_000}),
  )
  assert stub._positive_release_elapsed_s == pytest.approx(DT)

  LongitudinalPlanner._refresh_positive_release_elapsed(
    stub, SimpleNamespace(logMonoTime={}),
  )
  assert stub._positive_release_elapsed_s == pytest.approx(DT)
  assert stub._positive_release_prev_model_mono_ns == 0


def test_road_brake_to_floor_fix_rollback_and_emergency_twins() -> None:
  # 2026-07-14 17:23:22.852: same lead0/track -1035 jumped from
  # -0.8156906 straight to the +0.05 coast-bias floor in one model frame.
  recorded_brake = -0.8156905770301819
  recorded_floor = 0.05000000074505806

  fix, leads = _make_stub(prev_a=recorded_brake, owner="brake_release", release_jerk=2.0)
  rollback, rollback_leads = _make_stub(prev_a=recorded_brake, owner="brake_release", release_jerk=0.0)
  emergency, emergency_leads = _make_stub(prev_a=recorded_brake, owner="brake_release", release_jerk=2.0)

  for stub, stub_leads in ((fix, leads), (rollback, rollback_leads), (emergency, emergency_leads)):
    stub.output_a_target = recorded_floor
    stub._apply_comfort_jerk_envelope("lead0", stub_leads, v_ego=22.0)

  assert fix.output_a_target == pytest.approx(recorded_brake + 2.0 * DT)
  assert fix.comfort_jerk_debug["release_slew_active"] is True
  assert fix.comfort_jerk_debug["release_slew_clipped"] is True

  # The documented zero sentinel restores the exact ungraded floor jump.
  assert rollback.output_a_target == pytest.approx(recorded_floor)
  assert rollback.comfort_jerk_debug["release_slew_active"] is False
  assert rollback.comfort_jerk_debug["release_slew_clipped"] is False

  # Matched emergency twin: a renewed hard-brake request on the next frame is
  # never rate-limited, even while the positive release episode is latched.
  emergency.output_a_target = -2.0
  emergency._apply_comfort_jerk_envelope("lead0", emergency_leads, v_ego=22.0)
  assert emergency.output_a_target == pytest.approx(-2.0)
  assert emergency._brake_release_slew_active is False
  assert emergency.comfort_jerk_debug["bypassed"] is True
  assert emergency.comfort_jerk_debug["bypass_reason"] == "requested_decel"


def test_brake_release_slew_never_delays_renewed_braking() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="", release_elapsed_s=0.028)
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
  assert stub.output_a_target == pytest.approx(-0.8 + 2.0 * 0.028)
  assert stub._brake_release_slew_active is True

  stub.output_a_target = -2.0
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
  assert stub.output_a_target == pytest.approx(-2.0)
  assert stub._brake_release_slew_active is False


def test_brake_release_slew_requires_same_track_and_normal_road_speed() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="")
  leads[0].radarTrackId = -1036
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
  assert stub.output_a_target == pytest.approx(0.2)

  stub, leads = _make_stub(prev_a=-0.8, owner="")
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=2.0)
  assert stub.output_a_target == pytest.approx(0.2)


def test_track_change_disarms_an_active_release_and_refreshes_anchor() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="")
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
  assert stub.output_a_target == pytest.approx(-0.7)
  assert stub._brake_release_slew_active is True

  leads[0].radarTrackId = -1036
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)
  assert stub.output_a_target == pytest.approx(0.2)
  assert stub._brake_release_slew_active is False
  assert stub._comfort_jerk_prev_a == pytest.approx(0.2)
  assert stub._comfort_jerk_prev_track_id == -1036


def test_unknown_track_id_zero_fails_open_for_release_smoothing() -> None:
  stub, leads = _make_stub(prev_a=-0.8, owner="", track_id=0)
  stub.output_a_target = 0.2
  stub._apply_comfort_jerk_envelope("lead0", leads, v_ego=22.0)

  assert stub.output_a_target == pytest.approx(0.2)
  assert stub.comfort_jerk_debug["release_track_id"] == -1
  assert stub.comfort_jerk_debug["release_slew_clipped"] is False


def test_live_keepup_raise_cannot_expand_micro_recovery_ceiling() -> None:
  stub, leads = _make_stub(owner="lead_keepup", keepup_max_accel=1.0)
  stub.output_a_target = 0.4

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(0.4)
  assert stub.comfort_jerk_debug["upward_max_delta_mps2"] == pytest.approx(0.30)
  assert stub.comfort_jerk_debug["clipped"] is False


def test_persistent_micro_floor_rollon_uses_constant_comfort_jerk() -> None:
  stub, leads = _make_stub(owner="lead_keepup")
  outputs = []
  for _ in range(11):
    stub.output_a_target = 0.22
    stub._apply_comfort_jerk_envelope("lead0", leads)
    outputs.append(stub.output_a_target)

  assert outputs == pytest.approx([0.02 * i for i in range(1, 12)])
  assert stub.comfort_jerk_debug["upward_step_mps2"] == pytest.approx(COMFORT_JERK * DT)


def test_benign_downward_step_remains_comfort_slewed() -> None:
  stub, leads = _make_stub(owner="", release_elapsed_s=0.028)
  stub.output_a_target = -0.5

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(-COMFORT_JERK * DT)
  assert stub.comfort_jerk_debug["clipped"] is True
  assert stub.comfort_jerk_debug["max_step_mps2"] == pytest.approx(COMFORT_JERK * DT)
  assert stub.comfort_jerk_debug["release_elapsed_s"] == pytest.approx(0.028)


def test_same_track_steady_parity_threat_bypasses_downward_comfort_carryover() -> None:
  stub, leads = _make_stub(prev_a=0.05, owner="")
  leads[0].steadyParityThreatRestore = True
  stub.output_a_target = -0.083013

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(-0.083013)
  assert stub._comfort_jerk_prev_a == pytest.approx(-0.083013)
  assert stub.comfort_jerk_debug["bypassed"] is True
  assert stub.comfort_jerk_debug["bypass_reason"] == "steady_parity_current_threat"

  # The producer leaves raw-vRel-only invalidation un-attested, so the identical
  # comfort step remains graded instead of reintroducing a phantom brake tap.
  calm_stub, calm_leads = _make_stub(prev_a=0.05, owner="")
  calm_stub.output_a_target = -0.083013
  calm_stub._apply_comfort_jerk_envelope("lead0", calm_leads)
  assert calm_stub.output_a_target == pytest.approx(0.05 - COMFORT_JERK * DT)
  assert calm_stub.comfort_jerk_debug["bypassed"] is False


def test_requested_emergency_brake_bypasses_envelope() -> None:
  stub, leads = _make_stub(prev_a=0.2, owner="")
  stub.output_a_target = -2.0

  stub._apply_comfort_jerk_envelope("lead0", leads)

  assert stub.output_a_target == pytest.approx(-2.0)
  assert stub.comfort_jerk_debug["bypassed"] is True
  assert stub.comfort_jerk_debug["bypass_reason"] == "requested_decel"
