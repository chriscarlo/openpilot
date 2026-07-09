import math
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE,
  compute_relatch_required_decel,
)

# 2026-07-08 drive-home rlog anchors (route 0000021b/21c): every acquisition
# bypassed the old closing-alone urgency test (closing 2.65-5.28 m/s at TTC
# 12-21 s) and braked -1.0..-2.0 on arrival, producing the measured "never
# follows closer than 2.5 s" floor. These tests pin the kinematic replacement.
T_FOLLOW = 1.3
V_EGO = 21.6


def _lead(*, v_ego=V_EGO, closing=0.0, surplus_m=15.0, a_lead=0.0, fcw=False,
          track_id=7):
  d_rel = STOP_DISTANCE + T_FOLLOW * v_ego + surplus_m
  v_lead = v_ego - closing
  return SimpleNamespace(
    status=True,
    dRel=d_rel,
    vRel=v_lead - v_ego,
    vLead=v_lead,
    aLeadK=a_lead,
    fcw=fcw,
    radarTrackId=track_id,
  )


def _make_cfg(**overrides):
  # Real tuning dataclass so the kinematic helper sees genuine spec defaults
  # (8.0 closing backstop, -0.15 flat cap, 1.5 headroom, -1.5 bypass floor).
  return replace(LeadResponseTuningConfig.defaults(), **overrides)


def _make_stub(cfg=None, *, output_a=0.0, dt=0.05):
  cfg = cfg or _make_cfg()
  stub = SimpleNamespace(
    output_a_target=output_a,
    dt=dt,
    _relatch_blend_frames_left=0,
    _relatch_blend_prev_a=0.0,
    _relatch_prev_src="cruise",
    _reacquire_armed_pending=True,
    _exit_lead_track_id=7,
    _exit_lead_last_drel=None,
    relatch_blend_debug={},
    mpc=SimpleNamespace(_live_tune_cfg=cfg, current_t_follow=T_FOLLOW),
  )
  stub._relatch_urgency_bypass = types.MethodType(
    LongitudinalPlanner._relatch_urgency_bypass, stub,
  )
  stub._lead_owned_slot = LongitudinalPlanner._lead_owned_slot
  stub._apply_relatch_obstacle_blend = types.MethodType(
    LongitudinalPlanner._apply_relatch_obstacle_blend, stub,
  )
  return stub


class TestComputeRelatchRequiredDecel:
  def test_trace_routine_far_close_requires_little(self):
    # 17:04:30 acquire analog: closing 2.72 at 14.7 m surplus.
    lead = _lead(closing=2.72, surplus_m=14.7)
    req = compute_relatch_required_decel(V_EGO, lead, T_FOLLOW)
    assert req == pytest.approx(2.72**2 / (2 * 14.7), rel=0.05)
    assert req < 0.3

  def test_inside_target_closing_requires_multiple_mps2(self):
    lead = _lead(closing=2.72, surplus_m=-5.0)
    req = compute_relatch_required_decel(V_EGO, lead, T_FOLLOW)
    assert req >= 1.5  # surplus floor binds -> full-authority regime

  def test_pullaway_requires_nothing(self):
    lead = _lead(closing=-1.0, surplus_m=10.0)
    assert compute_relatch_required_decel(V_EGO, lead, T_FOLLOW) == 0.0

  def test_braking_lead_raises_requirement_via_stopping_need(self):
    calm = compute_relatch_required_decel(V_EGO, _lead(closing=1.0, surplus_m=30.0), T_FOLLOW)
    braking = compute_relatch_required_decel(V_EGO, _lead(closing=1.0, surplus_m=30.0, a_lead=-2.5), T_FOLLOW)
    assert braking > calm


class TestRelatchUrgencyBypassKinematic:
  def test_routine_freeway_acquire_no_longer_urgent(self):
    # Pre-fix this returned (True, "closing") for every drive-home acquire.
    stub = _make_stub()
    lead = _lead(closing=2.72, surplus_m=14.7)
    bypassed, reason = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert not bypassed, reason

  def test_trace_faster_close_still_not_urgent_at_big_surplus(self):
    # 17:06:04 analog: closing 4.16 at 21.7 m surplus (TTC 13.6 s) braked -1.99.
    stub = _make_stub()
    lead = _lead(closing=4.16, surplus_m=21.7)
    bypassed, reason = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert not bypassed, reason

  def test_same_closing_with_small_surplus_is_kinematically_urgent(self):
    stub = _make_stub()
    lead = _lead(closing=4.16, surplus_m=5.0)  # required = 17.3/10 = 1.73 >= 1.5
    bypassed, reason = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert bypassed and reason == "kinematic"

  def test_raw_closing_backstop_fires_at_extreme_close(self):
    stub = _make_stub()
    lead = _lead(closing=9.0, surplus_m=160.0)  # required small, closing >= 8.0
    bypassed, reason = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert bypassed and reason == "closing"

  def test_short_ttc_still_bypasses(self):
    stub = _make_stub()
    lead = _lead(closing=2.0, surplus_m=-27.0)  # dRel ~7 m -> ttc ~3.5 s
    bypassed, _ = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert bypassed

  def test_braking_lead_still_bypasses(self):
    stub = _make_stub()
    lead = _lead(closing=0.5, surplus_m=60.0, a_lead=-1.2)
    bypassed, _ = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert bypassed

  def test_requested_hard_decel_still_bypasses(self):
    stub = _make_stub(output_a=-1.6)
    lead = _lead(closing=0.5, surplus_m=60.0)
    bypassed, reason = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert bypassed and reason == "requested_decel"

  def test_bypass_decel_zero_disables_kinematic_leg(self):
    stub = _make_stub(_make_cfg(cruise_relatch_bypass_decel_mps2=0.0))
    lead = _lead(closing=4.16, surplus_m=5.0)
    bypassed, _ = stub._relatch_urgency_bypass(lead, stub.mpc._live_tune_cfg)
    assert not bypassed


def _run_blend_frames(stub, lead, demand, n):
  outs = []
  for _ in range(n):
    stub.output_a_target = demand
    stub._apply_relatch_obstacle_blend("lead0", (lead,))
    outs.append(stub.output_a_target)
  return outs


class TestRelatchKinematicCap:
  def test_trace_routine_acquire_glides_at_kinematic_cap(self):
    # Demand -1.10 (observed 17:04:30) must settle at -K*required ~= -0.38,
    # not the raw demand and not the old flat -0.8.
    stub = _make_stub()
    lead = _lead(closing=2.72, surplus_m=14.7)
    outs = _run_blend_frames(stub, lead, -1.10, 12)
    expected = -1.5 * (2.72**2 / (2 * 14.7))
    assert min(outs) == pytest.approx(expected, abs=0.02)

  def test_trace_faster_close_opens_proportional_authority(self):
    # 17:06:04 geometry (closing 4.16, surplus 21.7). Demands beyond the -1.5
    # requested floor legitimately bypass (never fight a hard MPC brake), so
    # the cap governs the comfort band: a -1.4 demand settles at -K*required
    # ~= -0.60, not the raw demand and not the old flat -0.8.
    stub = _make_stub()
    lead = _lead(closing=4.16, surplus_m=21.7)
    outs = _run_blend_frames(stub, lead, -1.4, 16)
    expected = -1.5 * (4.16**2 / (2 * 21.7))
    assert min(outs) == pytest.approx(expected, abs=0.02)

  def test_demand_beyond_requested_floor_always_passes(self):
    # Safety invariant: the cap never suppresses a demand at/below the -1.5
    # requested-decel floor — that leg bypasses the whole blend.
    stub = _make_stub()
    lead = _lead(closing=4.16, surplus_m=21.7)
    outs = _run_blend_frames(stub, lead, -1.99, 4)
    assert min(outs) == pytest.approx(-1.99)

  def test_tiny_requirement_holds_flat_floor(self):
    stub = _make_stub()
    lead = _lead(closing=0.8, surplus_m=40.0)  # required ~0.008
    outs = _run_blend_frames(stub, lead, -0.9, 12)
    assert min(outs) == pytest.approx(-0.15, abs=1e-6)

  def test_headroom_zero_sentinel_restores_flat_cap(self):
    stub = _make_stub(_make_cfg(cruise_relatch_kinematic_headroom=0.0))
    lead = _lead(closing=4.16, surplus_m=21.7)
    outs = _run_blend_frames(stub, lead, -1.4, 16)
    assert min(outs) == pytest.approx(-0.15, abs=1e-6)

  def test_urgent_arming_frame_passes_full_braking(self):
    stub = _make_stub()
    lead = _lead(closing=4.16, surplus_m=5.0)  # kinematic bypass on arming
    outs = _run_blend_frames(stub, lead, -1.2, 1)
    assert outs[0] == pytest.approx(-1.2)
    assert stub.relatch_blend_debug["bypassed"] is True
    assert stub.relatch_blend_debug["bypass_reason"] == "kinematic"

  def test_down_leg_jerk_still_shapes_the_approach_to_the_cap(self):
    # First frames must descend at blend_jerk*dt per frame, not step to the cap.
    stub = _make_stub()
    lead = _lead(closing=4.16, surplus_m=21.7)
    outs = _run_blend_frames(stub, lead, -1.4, 6)
    steps = [outs[0]] + [outs[i] - outs[i - 1] for i in range(1, len(outs))]
    assert all(s >= -(2.0 * 0.05) - 1e-9 for s in steps)

  def test_cap_reopens_frame_by_frame_as_surplus_shrinks(self):
    # Continuity: the same closing at shrinking surplus must monotonically
    # deepen the cap (no cliff anywhere along the approach).
    stub = _make_stub(_make_cfg(cruise_relatch_blend_jerk_mps3=50.0))
    caps = []
    for surplus in (25.0, 18.0, 12.0, 8.0, 5.5):
      lead = _lead(closing=3.0, surplus_m=surplus)
      stub.output_a_target = -1.4
      stub._apply_relatch_obstacle_blend("lead0", (lead,))
      caps.append(stub.output_a_target)
    assert all(caps[i + 1] < caps[i] for i in range(len(caps) - 1))
    assert not math.isinf(caps[-1])
