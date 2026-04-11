from types import SimpleNamespace

import pytest

from cereal import log
from openpilot.common.params import Params
from opendbc.car.hyundai.values import CAR
from opendbc.car.hyundai.interface import CarInterface
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_DANGER_FACTOR,
  get_cutin_settle_accel_floor,
  get_gap_reclaim_effective_cap,
  get_gap_reclaim_accel_floor,
  get_gap_reclaim_projection_scale,
  get_lead_handoff_danger_factor,
  get_lead_present_cruise_accel_cap,
  get_lead_approach_preview_buffer,
  should_start_cutin_settle_event,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


def _make_lead(*, status=True, d_rel=60.0, v_lead=33.5, a_lead=0.0):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    vLead=v_lead,
    aLeadK=a_lead,
  )


def _configure_vibe_follow(headway=1.3):
  params = Params()
  params.put_bool('VibePersonalityEnabled', True)
  params.put_bool('VibeFollowPersonalityEnabled', True)
  params.put_bool('VibeAccelPersonalityEnabled', False)
  params.put('LongitudinalPersonality', int(log.LongitudinalPersonality.standard))
  for idx in range(4):
    params.put(f'VibeTune.Follow.Standard.Headway{idx}', float(headway))


def _make_ev6_hkg_cp():
  cp = CarInterface.get_non_essential_params(CAR.KIA_EV6)
  cp.openpilotLongitudinalControl = True
  cp.pcmCruise = False
  return cp


def _run_pullaway_scenario():
  plant = Plant(
    lead_relevancy=True,
    speed=33.5,
    distance_lead=55.0,
    personality=int(log.LongitudinalPersonality.standard),
  )
  rows = []
  for _ in range(220):
    t = plant.current_time
    v_lead = 33.5 if t < 4.0 else 34.2
    logrow = plant.step(v_lead=v_lead, prob_lead=1.0, v_cruise=40.0, prob_throttle=1.0)
    rows.append({
      "t": t,
      "accel": float(logrow["acceleration"]),
      "d_rel": float(logrow["distance_lead"] - logrow["distance"]),
      "gap_floor": float(plant.planner.mpc.gap_reclaim_accel_floor),
    })
  return rows


def _run_new_lead_scenario(*, plant_kwargs=None, distance_lead=90.0, recognition_t=1.5,
                           ego_speed=33.5, lead_speed=27.0):
  plant_kwargs = {} if plant_kwargs is None else dict(plant_kwargs)
  plant = Plant(
    lead_relevancy=True,
    speed=ego_speed,
    distance_lead=distance_lead,
    personality=int(log.LongitudinalPersonality.standard),
    **plant_kwargs,
  )
  rows = []
  for _ in range(120):
    t = plant.current_time
    prob_lead = 0.0 if t < recognition_t else 1.0
    logrow = plant.step(v_lead=lead_speed, prob_lead=prob_lead, v_cruise=40.0, prob_throttle=1.0)
    preview_debug = plant.planner.mpc.lead_approach_preview_debug["lead0"]
    rows.append({
      "t": t,
      "accel": float(logrow["acceleration"]),
      "planner_accel": float(logrow["planner_acceleration"]),
      "controller_accel": float(logrow["controller_acceleration"]),
      "preview": float(plant.planner.mpc.lead_approach_preview[0]),
      "preview_mode": str(preview_debug["mode"]),
      "acquire_window_active": bool(preview_debug["acquire"]["active"]),
      "source": str(plant.planner.mpc.source),
    })
  return rows


@pytest.fixture(autouse=True)
def _planner_test_setup(monkeypatch):
  _configure_vibe_follow()
  monkeypatch.setattr(
    LongitudinalPlannerSP,
    "update_v_cruise",
    lambda self, sm, v_ego, a_ego, v_cruise: v_cruise,
  )


class TestLeadInteractionHeuristics:
  def test_gap_reclaim_floor_only_appears_for_safe_pullaway(self):
    safe_pullaway = _make_lead(d_rel=58.0, v_lead=34.3, a_lead=0.2)
    slower_lead = _make_lead(d_rel=58.0, v_lead=32.8, a_lead=0.0)
    braking_lead = _make_lead(d_rel=58.0, v_lead=34.3, a_lead=-0.8)

    floor = get_gap_reclaim_accel_floor(33.5, safe_pullaway, 1.3)

    assert floor > 0.1
    assert get_gap_reclaim_accel_floor(33.5, slower_lead, 1.3) == pytest.approx(0.0)
    assert get_gap_reclaim_accel_floor(33.5, braking_lead, 1.3) == pytest.approx(0.0)

  def test_gap_reclaim_effective_cap_expands_toward_personality_accel_for_large_surplus_gap(self):
    wide_pullaway = _make_lead(d_rel=72.0, v_lead=35.2, a_lead=0.2)

    comfort_cap = get_gap_reclaim_effective_cap(33.5, wide_pullaway, 1.3)
    sport_cap = get_gap_reclaim_effective_cap(33.5, wide_pullaway, 1.3, personality_max_accel=1.15)
    comfort_floor = get_gap_reclaim_accel_floor(33.5, wide_pullaway, 1.3)
    sport_floor = get_gap_reclaim_accel_floor(33.5, wide_pullaway, 1.3, personality_max_accel=1.15)

    assert sport_cap > comfort_cap + 0.25
    assert sport_cap <= 1.15 + 1e-6
    assert sport_floor > comfort_floor + 0.15

  def test_gap_reclaim_effective_cap_stays_close_to_comfort_near_target_gap(self):
    near_target_pullaway = _make_lead(d_rel=51.5, v_lead=34.1, a_lead=0.1)

    comfort_cap = get_gap_reclaim_effective_cap(33.5, near_target_pullaway, 1.3)
    sport_cap = get_gap_reclaim_effective_cap(33.5, near_target_pullaway, 1.3, personality_max_accel=1.15)

    assert sport_cap - comfort_cap < 0.10

  def test_lead_present_cruise_accel_cap_stays_moderate_in_traffic(self):
    traffic_pullaway = _make_lead(d_rel=28.0, v_lead=9.8, a_lead=0.1)
    setattr(traffic_pullaway, "vRel", 0.6)

    cap = get_lead_present_cruise_accel_cap(9.0, traffic_pullaway, 1.3, personality_max_accel=3.5)

    assert 0.55 < cap < 1.20

  def test_lead_present_cruise_accel_cap_can_expand_for_far_open_gap(self):
    far_pullaway = _make_lead(d_rel=60.0, v_lead=11.5, a_lead=0.2)
    setattr(far_pullaway, "vRel", 1.2)

    cap = get_lead_present_cruise_accel_cap(9.0, far_pullaway, 1.3, personality_max_accel=3.5)

    assert cap > 1.5

  def test_lead_present_cruise_accel_cap_tightens_for_clearly_slower_leads(self):
    slow_lead = _make_lead(d_rel=100.0, v_lead=0.0, a_lead=0.0)
    setattr(slow_lead, "vRel", -9.0)

    cap = get_lead_present_cruise_accel_cap(9.0, slow_lead, 1.3, personality_max_accel=3.5)

    # With high closing speed, cap should be tight (near comfort cap) not None
    assert cap is not None
    assert cap < 1.0

  def test_gap_reclaim_projection_scale_tapers_room_when_ego_accel_is_already_closing_gap(self):
    mid_pullaway = _make_lead(d_rel=58.0, v_lead=33.9, a_lead=0.1)

    low_ego_scale = get_gap_reclaim_projection_scale(33.5, mid_pullaway, 1.3, ego_accel=0.0)
    high_ego_scale = get_gap_reclaim_projection_scale(33.5, mid_pullaway, 1.3, ego_accel=0.9)

    assert low_ego_scale == pytest.approx(1.0)
    assert high_ego_scale < 0.45

  def test_approach_preview_only_appears_when_closing_outside_headway(self):
    closing_far = _make_lead(d_rel=79.0, v_lead=27.0, a_lead=0.0)
    near_headway = _make_lead(d_rel=49.0, v_lead=27.0, a_lead=0.0)
    non_closing = _make_lead(d_rel=79.0, v_lead=34.5, a_lead=0.0)

    preview = get_lead_approach_preview_buffer(34.5, closing_far, 1.3)

    assert preview > 3.0
    assert get_lead_approach_preview_buffer(34.5, near_headway, 1.3) == pytest.approx(0.0)
    assert get_lead_approach_preview_buffer(34.5, non_closing, 1.3) == pytest.approx(0.0)

  def test_lead_handoff_danger_factor_engages_for_high_speed_cutin_inside_current_headway(self):
    inside_headway = _make_lead(d_rel=23.0, v_lead=24.0, a_lead=0.0)

    factor = get_lead_handoff_danger_factor(27.0, inside_headway, 1.3, handoff_remaining_s=1.25)

    assert factor > 0.93

  def test_lead_handoff_danger_factor_supports_projected_slower_far_lead(self):
    slower_far_lead = _make_lead(d_rel=56.0, v_lead=22.0, a_lead=0.0)

    factor = get_lead_handoff_danger_factor(35.0, slower_far_lead, 1.3, handoff_remaining_s=1.25)

    assert factor > 0.90

  def test_lead_handoff_danger_factor_fades_as_window_expires(self):
    inside_headway = _make_lead(d_rel=23.0, v_lead=24.0, a_lead=0.0)

    early = get_lead_handoff_danger_factor(27.0, inside_headway, 1.3, handoff_remaining_s=1.25)
    late = get_lead_handoff_danger_factor(27.0, inside_headway, 1.3, handoff_remaining_s=0.20)

    assert early > late > LEAD_DANGER_FACTOR

  def test_lead_handoff_danger_factor_ignores_low_speed_cutin(self):
    inside_headway = _make_lead(d_rel=12.0, v_lead=4.0, a_lead=0.0)

    factor = get_lead_handoff_danger_factor(8.0, inside_headway, 1.3, handoff_remaining_s=1.25)

    assert factor == pytest.approx(LEAD_DANGER_FACTOR)

  def test_cutin_settle_event_starts_for_adjacent_to_control_transition(self):
    lead = _make_lead(d_rel=44.0, v_lead=33.0, a_lead=0.0)

    assert should_start_cutin_settle_event(
      prev_role="adjacent_awareness_left",
      prev_control_active=False,
      current_role="center_control",
      lead=lead,
      cutin_promoted=False,
      toward_center_mps=0.2,
      path_abs_m=0.6,
      v_ego=33.5,
    ) is True

  def test_cutin_settle_event_ignores_straight_new_center_lead_without_lateral_hint(self):
    lead = _make_lead(d_rel=44.0, v_lead=33.0, a_lead=0.0)

    assert should_start_cutin_settle_event(
      prev_role="invalid",
      prev_control_active=False,
      current_role="center_control",
      lead=lead,
      cutin_promoted=False,
      toward_center_mps=0.0,
      path_abs_m=0.1,
      v_ego=33.5,
    ) is False

  def test_cutin_settle_floor_only_appears_for_benign_recent_cutin(self):
    benign = _make_lead(d_rel=47.0, v_lead=33.0, a_lead=0.0)
    dangerous = _make_lead(d_rel=38.0, v_lead=30.0, a_lead=-0.8)

    floor = get_cutin_settle_accel_floor(33.5, benign, 1.3, age_s=4.0)

    assert floor is not None
    assert 0.15 < floor < 0.20  # positive due to the stronger default regen bias (+0.20)
    assert get_cutin_settle_accel_floor(33.5, dangerous, 1.3, age_s=1.0) is None

  def test_cutin_settle_floor_blocks_braking_for_same_speed_merge(self):
    same_speed = _make_lead(d_rel=46.0, v_lead=33.5, a_lead=0.0)

    floor = get_cutin_settle_accel_floor(33.5, same_speed, 1.3, age_s=1.0)

    assert floor == pytest.approx(0.20)  # pure regen bias, no decel (closing_speed=0)


class TestLeadInteractionScenarios:
  def test_pullaway_gap_reclaim_nudges_accel_without_large_gap_growth(self):
    rows = _run_pullaway_scenario()

    accel_at_4p8s = next(row["accel"] for row in rows if row["t"] >= 4.8)
    gap_at_6s = next(row["d_rel"] for row in rows if row["t"] >= 6.0)
    max_gap_floor = max(row["gap_floor"] for row in rows)

    assert accel_at_4p8s >= 0.15
    assert max_gap_floor >= 0.20
    assert gap_at_6s < 55.0

  def test_new_slower_lead_uses_preview_and_starts_releasing_accel_earlier(self):
    rows = _run_new_lead_scenario()

    accel_at_2s = next(row["accel"] for row in rows if row["t"] >= 2.0)
    max_preview = max(row["preview"] for row in rows)

    assert max_preview >= 5.0
    assert accel_at_2s < 0.30

  def test_ev6_hkg_new_lead_acquisition_uses_real_cp_and_no_radar_damping(self):
    rows = _run_new_lead_scenario(
      plant_kwargs={"CP": _make_ev6_hkg_cp(), "hyundai_controller": True},
    )

    max_preview = max(row["preview"] for row in rows)
    acquire_active = any(row["acquire_window_active"] for row in rows if row["t"] >= 1.5)
    acquire_mode_seen = any(row["preview_mode"] == "acquire" for row in rows if row["t"] >= 1.5)
    accel_at_2s = next(row["accel"] for row in rows if row["t"] >= 2.0)
    planner_accel_at_2s = next(row["planner_accel"] for row in rows if row["t"] >= 2.0)
    accel_at_2p25s = next(row["accel"] for row in rows if row["t"] >= 2.25)
    accel_at_2p3s = next(row["accel"] for row in rows if row["t"] >= 2.3)
    source_after_acquire = next(row["source"] for row in rows if row["t"] >= 2.0)
    planner_vs_controller = max(abs(row["planner_accel"] - row["controller_accel"]) for row in rows if row["t"] >= 1.5)

    assert acquire_active is True
    assert acquire_mode_seen is True
    assert max_preview >= 6.0
    assert planner_accel_at_2s < -0.10
    assert accel_at_2s > 0.40
    assert accel_at_2p25s <= 0.05
    assert accel_at_2p3s < 0.0
    assert source_after_acquire == "lead0"
    assert planner_vs_controller > 0.5
