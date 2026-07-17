from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import build_lead_response_tuning_config
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  ACCEL_MAX,
  STOP_DISTANCE,
  LongitudinalMpc,
  N,
  get_gap_reclaim_accel_floor,
  get_lead_keepup_accel_floor,
  get_lead_slowdown_accel_ceiling,
)


def _make_lead(*, status=True, d_rel=44.0, y_rel=0.0, d_path=None, v_lat=0.0, v_rel=0.0,
               v_lead=29.0, a_lead=0.0, model_prob=0.95, radar=False, radar_track_id=-1):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    yRel=y_rel,
    vRel=v_rel,
    aRel=0.0,
    vLead=v_lead,
    dPath=y_rel if d_path is None else d_path,
    vLat=v_lat,
    vLeadK=v_lead,
    aLeadK=a_lead,
    fcw=False,
    aLeadTau=1.5,
    modelProb=model_prob,
    radar=radar,
    radarTrackId=radar_track_id,
  )


def _configure_vibe_follow(headway=1.3):
  params = Params()
  params.put_bool('VibePersonalityEnabled', True)
  params.put_bool('VibeFollowPersonalityEnabled', True)
  params.put_bool('VibeAccelPersonalityEnabled', False)
  params.put('LongitudinalPersonality', int(log.LongitudinalPersonality.standard))
  for idx in range(4):
    params.put(f'VibeTune.Follow.Standard.Headway{idx}', float(headway))


def _configure_vibe_accel(*, enabled: bool, personality: int = 0):
  params = Params()
  params.put_bool('VibePersonalityEnabled', True)
  params.put_bool('VibeAccelPersonalityEnabled', enabled)
  params.put('AccelPersonality', str(int(personality)))


def _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, *, time_fn=None):
  mpc = LongitudinalMpc(CP=SimpleNamespace(brand='hyundai'))
  mpc.mode = 'acc'
  mpc.set_cur_state(v_ego, a_ego)
  if time_fn is not None:
    mpc._time_fn = time_fn
  return mpc


def _run_update(mpc: LongitudinalMpc, lead0, lead1, *, v_cruise=40.0):
  radarstate = SimpleNamespace(leadOne=lead0, leadTwo=lead1)
  x = np.zeros(N + 1)
  v = np.zeros(N + 1)
  a = np.zeros(N + 1)
  j = np.zeros(N + 1)
  mpc.update(radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard)


def _run_update_with_state(mpc: LongitudinalMpc, lead0, lead1, *, v_ego, a_ego=0.0, v_cruise=40.0):
  mpc.set_cur_state(v_ego, a_ego)
  _run_update(mpc, lead0, lead1, v_cruise=v_cruise)


class _MonotonicStub:
  def __init__(self, start=100.0, step=0.2):
    self.value = start
    self.step = step

  def __call__(self):
    current = self.value
    self.value += self.step
    return current


@pytest.fixture(autouse=True)
def _planner_test_setup():
  _configure_vibe_follow()
  params = Params()
  params.put_bool("VTSC.Expert.AdjLeadControlEnabled", True)
  params.put("VTSC.Expert.AdjLeadCutInDRelMaxM", 60.0)


class TestHyundaiAiLeadStability:
  def test_steady_model_lead_reacquires_when_stop_sign_closing_before_gap_collapses(self):
    v_cruise_30mph = 48.0 / 3.6
    mpc = _make_hyundai_mpc(v_ego=13.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.2))

    for _ in range(20):
      _run_update_with_state(
        mpc,
        _make_lead(d_rel=49.0, y_rel=0.04, d_path=0.04, v_rel=0.0, v_lead=13.0, a_lead=0.0,
                   model_prob=0.99, radar=False, radar_track_id=-1006),
        _make_lead(status=False),
        v_ego=13.0,
        v_cruise=v_cruise_30mph,
      )
    assert mpc.source == "cruise"

    loglike_samples = (
      (13.02, 47.19, -0.10, 12.92, -0.02),
      (13.08, 46.56, -1.41, 11.62, -0.09),
      (13.15, 44.64, -1.06, 12.12, -0.27),
      (13.25, 40.63, -1.21, 12.02, -0.21),
      (13.23, 39.62, -1.33, 11.93, -0.36),
      (13.20, 36.48, -1.62, 11.62, -0.38),
      (13.24, 33.92, -1.93, 11.31, -0.38),
    )
    for v_ego, d_rel, v_rel, v_lead, a_lead in loglike_samples:
      _run_update_with_state(
        mpc,
        _make_lead(d_rel=d_rel, y_rel=0.04, d_path=0.04, v_rel=v_rel, v_lead=v_lead,
                   a_lead=a_lead, model_prob=0.99, radar=False, radar_track_id=-1006),
        _make_lead(status=False),
        v_ego=v_ego,
        v_cruise=v_cruise_30mph,
      )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "approach_reacquire"
    assert mpc.acc_source_debug["approach_reacquire"] is True
    assert mpc.acc_source_debug["raw_closing_mps"] >= 1.9
    assert mpc.lead_slowdown_accel_ceiling < -0.15

  def test_same_synthetic_model_track_can_switch_slots_without_source_reset(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc()
    track_id = -1001

    _run_update(
      mpc,
      _make_lead(d_rel=42.0, y_rel=0.03, d_path=0.03, radar_track_id=track_id),
      _make_lead(status=False),
    )

    assert mpc.hyundai_virtual_lead_debug["reset_reason"] == "init"
    assert mpc._hyundai_virtual_lead_source == "lead0"

    _run_update(
      mpc,
      _make_lead(status=False),
      _make_lead(d_rel=42.2, y_rel=0.04, d_path=0.04, radar_track_id=track_id),
    )

    assert mpc._hyundai_virtual_lead_source == "lead1"
    assert mpc.hyundai_virtual_lead_debug["reset_reason"] is None
    assert mpc.hyundai_virtual_lead_debug["identity_changed"] is False
    assert mpc._hyundai_virtual_lead.radarTrackId == track_id

  def test_duplicate_pair_keeps_virtual_winner_stable_across_small_slot_jitter(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc()

    samples = [
      (
        _make_lead(d_rel=44.90, y_rel=0.05, d_path=0.05, v_lat=0.45, v_rel=-0.1, model_prob=0.96),
        _make_lead(d_rel=44.85, y_rel=0.08, d_path=0.08, v_lat=4.20, v_rel=-0.08, model_prob=0.92),
      ),
      (
        _make_lead(d_rel=44.92, y_rel=0.04, d_path=0.04, v_lat=0.55, v_rel=-0.1, model_prob=0.95),
        _make_lead(d_rel=44.80, y_rel=0.06, d_path=0.06, v_lat=4.05, v_rel=-0.09, model_prob=0.93),
      ),
      (
        _make_lead(d_rel=44.88, y_rel=0.05, d_path=0.05, v_lat=0.50, v_rel=-0.1, model_prob=0.95),
        _make_lead(d_rel=44.82, y_rel=0.07, d_path=0.07, v_lat=4.10, v_rel=-0.08, model_prob=0.93),
      ),
    ]

    selected_slots = []
    for lead0, lead1 in samples:
      _run_update(mpc, lead0, lead1)
      assert mpc.lead_role_debug["virtual_duplicate"]["active"] is True
      selected_slots.append(mpc.lead_role_debug["virtual_duplicate"]["selected_raw_slot"])

    assert selected_slots == [0, 0, 0]
    assert mpc.control_leads[0].status is True
    assert mpc.control_leads[1].status is False

  def test_source_hysteresis_holds_lead_through_brief_jitter_then_releases_after_sustained_gap_growth(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    warmup_lead0 = _make_lead(d_rel=35.6, y_rel=0.05, d_path=0.05, v_lat=0.45, v_rel=-0.1, model_prob=0.96)
    warmup_lead1 = _make_lead(d_rel=35.55, y_rel=0.08, d_path=0.08, v_lat=4.10, v_rel=-0.08, model_prob=0.92)
    for _ in range(4):
      _run_update(mpc, warmup_lead0, warmup_lead1)
    assert mpc.source == "lead0"

    jitter_sources = []
    for d_rel, v_rel, v_lead in (
      (38.7, 0.55, 29.55),
      (39.0, 0.72, 29.72),
      (38.8, 0.18, 29.18),
      (39.1, 0.64, 29.64),
    ):
      _run_update(
        mpc,
        _make_lead(d_rel=d_rel, y_rel=0.05, d_path=0.05, v_lat=0.50, v_rel=v_rel, v_lead=v_lead, model_prob=0.95),
        _make_lead(d_rel=d_rel - 0.04, y_rel=0.08, d_path=0.08, v_lat=4.00, v_rel=v_rel - 0.02, v_lead=v_lead, model_prob=0.93),
      )
      jitter_sources.append(mpc.source)

    assert jitter_sources == ["lead0", "lead0", "lead0", "lead0"]
    assert mpc.hyundai_virtual_lead_debug["active"] is True

    max_reclaim_push = 0.0
    for _ in range(12):
      _run_update(
        mpc,
        _make_lead(d_rel=43.0, y_rel=0.05, d_path=0.05, v_lat=0.45, v_rel=0.85, v_lead=29.85, model_prob=0.95),
        _make_lead(d_rel=42.95, y_rel=0.07, d_path=0.07, v_lat=4.05, v_rel=0.82, v_lead=29.85, model_prob=0.93),
      )
      max_reclaim_push = max(max_reclaim_push, float(mpc.gap_reclaim_obstacle_push))

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["used_hysteresis"] is True
    assert mpc.acc_source_debug["reason"] in ("filtered_pullaway_dwell", "filtered_pullaway_immediate", "cruise_hold")
    assert mpc.acc_source_debug["raw_release_ready"] is True
    assert mpc.acc_source_debug["release_agreement_ok"] is True
    assert max_reclaim_push > 0.5

  def test_filtered_release_waits_for_raw_agreement_before_leaving_lead(self):
    mpc = _make_hyundai_mpc(time_fn=_MonotonicStub(step=0.2))

    stable_lead = _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96)
    for _ in range(4):
      _run_update(mpc, stable_lead, _make_lead(status=False))

    for _ in range(3):
      _run_update(
        mpc,
        _make_lead(d_rel=46.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=1.00, v_lead=30.00, a_lead=0.05, model_prob=0.96),
        _make_lead(status=False),
      )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["candidate_mode"] is None
    assert mpc.acc_source_debug["filtered_release_ready"] is False
    assert mpc.acc_source_debug["raw_release_ready"] is True
    assert mpc.acc_source_debug["release_agreement_ok"] is True

    _run_update(
      mpc,
      _make_lead(d_rel=46.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
      _make_lead(status=False),
    )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "filtered_hold"
    assert mpc.acc_source_debug["candidate_mode"] is None
    assert mpc.acc_source_debug["raw_release_ready"] is False
    assert mpc.acc_source_debug["release_agreement_ok"] is False

  def test_settled_follow_suppresses_raw_stabilization_push_from_gap_breathing(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    stable_lead = _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96)
    for _ in range(4):
      _run_update(mpc, stable_lead, _make_lead(status=False))

    _run_update(
      mpc,
      _make_lead(d_rel=49.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
      _make_lead(status=False),
    )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["steady_follow"] is True
    assert mpc.acc_source_debug["stabilization_push_suppressed"] is True
    assert mpc.acc_source_debug["stabilization_push_m"] == pytest.approx(0.0)
    assert mpc.acc_source_debug["raw_reclaim_safety_override"] is False

  def test_keepup_floor_engages_before_large_gap_reclaim_threshold(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    params = Params()
    saved_gap_min = params.get("Longitudinal.LiveTune.GapReclaimGapMinM")
    try:
      params.put("Longitudinal.LiveTune.GapReclaimGapMinM", 4.0)
      mpc = _make_hyundai_mpc(v_ego=33.5, a_ego=0.0)

      for _ in range(2):
        _run_update(
          mpc,
          _make_lead(d_rel=43.7, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.97),
          _make_lead(status=False),
          v_cruise=40.0,
        )
      _run_update(
        mpc,
        _make_lead(d_rel=44.2, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.45, v_lead=33.95, a_lead=0.0, model_prob=0.97),
        _make_lead(status=False),
        v_cruise=40.0,
      )

      assert mpc.source == "lead0"
      assert mpc.gap_reclaim_accel_floor == pytest.approx(0.0)
      assert 0.0 < mpc.lead_keepup_accel_floor < 0.05
      assert mpc.acc_source_debug["lead_keepup_accel_floor"] == pytest.approx(mpc.lead_keepup_accel_floor)
    finally:
      if saved_gap_min is None:
        params.remove("Longitudinal.LiveTune.GapReclaimGapMinM")
      else:
        params.put("Longitudinal.LiveTune.GapReclaimGapMinM", saved_gap_min)

  def test_ev6_soft_follow_profile_reduces_goldilocks_pullaway_floors(self):
    pre_soften_tune = build_lead_response_tuning_config({
      "gap_reclaim_strength": 0.35,
      "gap_reclaim_gap_min_m": 4.0,
      "gap_reclaim_max_accel": 0.10,
      "lead_keepup_gap_min_m": 0.40,
      "lead_keepup_max_accel": 0.10,
    })
    softened_tune = build_lead_response_tuning_config({
      "gap_reclaim_strength": 0.35,
      "gap_reclaim_gap_min_m": 5.0,
      "gap_reclaim_max_accel": 0.08,
      "lead_keepup_gap_min_m": 0.80,
      "lead_keepup_max_accel": 0.06,
    })

    near_goldilocks_lead = _make_lead(d_rel=47.0, v_rel=0.25, v_lead=29.25)
    wide_goldilocks_lead = _make_lead(d_rel=54.0, v_rel=0.25, v_lead=29.25)

    pre_near_keepup = get_lead_keepup_accel_floor(29.0, near_goldilocks_lead, 1.3, pre_soften_tune)
    soft_near_keepup = get_lead_keepup_accel_floor(29.0, near_goldilocks_lead, 1.3, softened_tune)
    pre_wide_reclaim = get_gap_reclaim_accel_floor(29.0, wide_goldilocks_lead, 1.3, pre_soften_tune)
    soft_wide_reclaim = get_gap_reclaim_accel_floor(29.0, wide_goldilocks_lead, 1.3, softened_tune)
    pre_wide_keepup = get_lead_keepup_accel_floor(29.0, wide_goldilocks_lead, 1.3, pre_soften_tune)
    soft_wide_keepup = get_lead_keepup_accel_floor(29.0, wide_goldilocks_lead, 1.3, softened_tune)

    assert get_gap_reclaim_accel_floor(29.0, near_goldilocks_lead, 1.3, softened_tune) == pytest.approx(0.0)
    assert soft_near_keepup < 0.005
    assert soft_wide_reclaim == pytest.approx(0.08)
    assert soft_wide_reclaim < pre_wide_reclaim
    assert 0.05 < soft_wide_keepup <= 0.06
    assert soft_wide_keepup < pre_wide_keepup

  def test_slowdown_ceiling_reaches_full_decel_for_close_braking_lead(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=33.5, a_ego=0.0)

    _run_update(
      mpc,
      _make_lead(d_rel=10.0, y_rel=0.04, d_path=0.04, v_rel=-15.5, v_lead=18.0, a_lead=-5.0, model_prob=0.98),
      _make_lead(status=False),
      v_cruise=40.0,
    )

    assert mpc.source == "lead0"
    assert mpc.lead_slowdown_accel_ceiling == pytest.approx(-4.0)
    assert mpc.acc_source_debug["lead_slowdown_accel_ceiling"] == pytest.approx(-4.0)

  def test_slowdown_ceiling_stays_out_of_terminal_rollout(self):
    tune = build_lead_response_tuning_config({"lead_slowdown_max_decel": 4.0})
    crawl_lead = _make_lead(d_rel=0.5, v_rel=-0.5, v_lead=0.0, a_lead=-4.0)
    active_lead = _make_lead(d_rel=0.5, v_rel=-2.1, v_lead=0.0, a_lead=-4.0)

    assert get_lead_slowdown_accel_ceiling(
      0.5, crawl_lead, 1.45, tune, min_accel=-6.0, max_accel=1.6,
    ) is None
    assert get_lead_slowdown_accel_ceiling(
      2.1, active_lead, 1.45, tune, min_accel=-6.0, max_accel=1.6,
    ) < -1.0

  def test_slowdown_ceiling_release_is_rate_limited_while_close_and_closing(self):
    mpc = _make_hyundai_mpc(v_ego=28.0, a_ego=0.0)
    mpc._lead_slowdown_accel_ceiling_last = -1.0
    mpc._lead_slowdown_accel_ceiling_last_t = 100.0
    raw_metrics = {"gap_surplus": -0.29, "closing_speed": 2.2, "pullaway_speed": 0.0}
    filtered_metrics = {"gap_surplus": -0.38, "closing_speed": 2.1, "pullaway_speed": 0.0}

    limited = mpc._limit_lead_slowdown_ceiling_release(-0.05, raw_metrics, filtered_metrics, 100.2)

    assert limited < -0.85

  def test_lead_to_cruise_transition_cap_ramps_for_three_seconds(self):
    mpc = _make_hyundai_mpc(v_ego=23.6, a_ego=0.0)
    mpc.source = "cruise"
    mpc._lead_to_cruise_transition_t = 100.0

    assert mpc._get_lead_to_cruise_transition_accel_cap(100.0, 23.6, 1.6) <= 0.30
    assert mpc._get_lead_to_cruise_transition_accel_cap(102.0, 23.6, 1.6) < 1.20
    assert mpc._get_lead_to_cruise_transition_accel_cap(103.0, 23.6, 1.6) is None

  def test_brief_total_lead_dropout_holds_stable_virtual_lead_before_releasing(self):
    mpc = _make_hyundai_mpc(time_fn=_MonotonicStub(step=0.2))

    stable_lead = _make_lead(d_rel=35.6, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.02, v_lead=29.02, model_prob=0.98)
    for _ in range(32):
      _run_update(mpc, stable_lead, _make_lead(status=False))

    assert mpc.source == "lead0"

    held_sources = []
    for _ in range(7):
      _run_update(mpc, _make_lead(status=False), _make_lead(status=False))
      held_sources.append(mpc.source)
      stabilizer_active = bool(mpc.lead_stability_debug["slot0"]["out_status"])
      dropout_active = bool(mpc.hyundai_virtual_lead_debug.get("dropout_hold", {}).get("active", False))
      assert mpc.acc_source_debug["reason"] in ("dropout_hold", "raw_gap_hold")
      assert stabilizer_active or dropout_active

    assert held_sources == ["lead0"] * 7

    _run_update(mpc, _make_lead(status=False), _make_lead(status=False))

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"

  def test_center_control_grace_rides_through_one_frame_path_spike(self):
    # Stable center-control lead, then one-frame path spike just beyond the
    # center-lane exit hysteresis. The classifier should preserve ownership via
    # center_lane_grace, so the planner-side demotion hold never needs to fire.
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))

    stable_lead = _make_lead(d_rel=35.6, y_rel=1.0, d_path=1.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    for _ in range(10):
      _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert mpc._hyundai_virtual_lead is not None
    assert abs(mpc._hyundai_virtual_lead.dPath) > 0.85

    # Same dRel/vRel, but dPath/yRel jump above the classifier's center-lane
    # exit hysteresis. The classifier should hold CENTER_CONTROL instead of
    # letting control_lead0 go empty.
    flicker_lead = _make_lead(d_rel=35.6, y_rel=3.0, d_path=3.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    _run_update(mpc, flicker_lead, _make_lead(status=False))

    assert mpc.source == "lead0"
    assert mpc.lead_role_debug["reasons"]["lead0"] == "center_lane_grace"
    assert mpc.lead_role_debug["control_status"]["lead0"] is True
    assert mpc.lead_role_debug["center_grace"]["lead0"]["active"] is True
    assert mpc.acc_source_debug["reason"] != "classifier_demotion_hold"
    assert mpc.acc_source_debug["source_transition_active"] is False
    assert mpc._classifier_demotion_hold_until_t is None

    _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert mpc._classifier_demotion_hold_until_t is None

  def test_planner_demotion_backstop_handles_larger_path_spike(self):
    # If the path jump is large enough that the classifier-side grace refuses
    # it, the planner-side classifier_demotion_hold remains as the secondary
    # backstop and should preserve lead ownership for the brief spike.
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))

    # Keep the stored path within the planner demotion backstop envelope
    # (<= 1.2m) but outside dropout_hold's stricter path check (> 0.85m),
    # so dropout_hold cannot short-circuit this case.
    stable_lead = _make_lead(d_rel=35.6, y_rel=1.0, d_path=1.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    for _ in range(12):
      _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert mpc._hyundai_virtual_lead is not None
    assert 0.85 < abs(mpc._hyundai_virtual_lead.dPath) <= 1.2
    assert mpc._hyundai_virtual_lead_stable_since_t is not None

    flicker_lead = _make_lead(d_rel=35.6, y_rel=3.4, d_path=3.4, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    _run_update(mpc, flicker_lead, _make_lead(status=False))

    assert mpc.lead_role_debug["reasons"]["lead0"] == "adjacent_lane"
    assert mpc.lead_role_debug["control_status"]["lead0"] is False
    assert mpc.lead_role_debug["center_grace"]["lead0"]["active"] is False
    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "classifier_demotion_hold"
    demotion_debug = mpc.acc_source_debug["classifier_demotion_hold"]
    assert demotion_debug["active"] is True
    assert demotion_debug["eligible"] is True
    assert demotion_debug["corroboration"]["matched"] is True
    assert demotion_debug["corroboration"]["dRel_delta_m"] <= 2.0
    assert mpc.acc_source_debug["source_transition_active"] is False

  def test_raw_lateral_departure_bypasses_classifier_demotion_hold(self):
    # The planner demotion backstop is only for brief classifier path spikes.
    # If the classifier explicitly flags an extreme raw lateral departure, do
    # not re-hold the previous virtual lead just because dRel still matches.
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))

    stable_lead = _make_lead(d_rel=35.6, y_rel=0.2, d_path=0.2, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    for _ in range(15):
      _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert mpc._hyundai_virtual_lead is not None

    departing_lead = _make_lead(
      d_rel=35.6,
      y_rel=-8.0,
      d_path=0.2,
      v_lat=12.0,
      v_rel=0.0,
      v_lead=29.0,
      model_prob=0.97,
    )
    _run_update(mpc, departing_lead, _make_lead(status=False))

    assert mpc.lead_role_debug["reasons"]["lead0"] == "raw_lateral_departure"
    assert mpc.lead_role_debug["control_status"]["lead0"] is False
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"
    assert mpc._classifier_demotion_hold_until_t is None

  def test_new_raw_lateral_departure_cannot_cap_cruise(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))

    departing_path_lead = _make_lead(
      d_rel=75.0,
      y_rel=-8.0,
      d_path=0.4,
      v_lat=12.0,
      v_rel=-2.0,
      v_lead=27.0,
      model_prob=0.97,
      radar_track_id=-1036,
    )
    _run_update(mpc, departing_path_lead, _make_lead(status=False))

    assert mpc.lead_role_debug["reasons"]["lead0"] == "raw_lateral_departure"
    assert mpc.lead_role_debug["control_status"]["lead0"] is False
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "raw_lateral_departure"
    assert mpc.cruise_owned_accel_cap is None
    assert mpc.params[0, 1] == pytest.approx(ACCEL_MAX)

  def test_stable_adjacent_raw_path_requires_dwell_before_capping_cruise(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    absent = _make_lead(status=False)
    stable_demoted = _make_lead(
      d_rel=75.0,
      y_rel=1.7,
      d_path=1.7,
      v_lat=0.0,
      v_rel=-2.0,
      v_lead=27.0,
      model_prob=0.97,
      radar_track_id=-2001,
    )

    for _ in range(6):
      _run_update(mpc, stable_demoted, absent)
      assert mpc.lead_role_debug["reasons"]["lead0"] == "adjacent_lane"
      assert mpc.source == "cruise"
      assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None
      assert mpc.params[0, 1] == pytest.approx(ACCEL_MAX)

    _run_update(mpc, stable_demoted, absent)
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "dwell_complete"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead0_raw_path"
    assert mpc.cruise_owned_accel_cap == pytest.approx(0.0)
    assert mpc.params[0, 1] == pytest.approx(0.0)

  @pytest.mark.parametrize(
    ("decoy_track_id", "decoy_reason", "decoy_phantom"),
    (
      (-1, "adjacent_lane", False),
      (-9101, "adjacent_lane", True),
      (-9102, "raw_lateral_departure", False),
    ),
    ids=("missing_identity", "phantom", "raw_lateral_departure"),
  )
  def test_farther_stable_raw_candidate_accumulates_dwell_despite_nearer_rejected_decoy(
      self, decoy_track_id, decoy_reason, decoy_phantom):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0)
    nearer_decoy = _make_lead(
      d_rel=40.0, y_rel=0.4, d_path=0.4, v_rel=-2.0, v_lead=27.0,
      model_prob=0.99, radar_track_id=decoy_track_id,
    )
    farther_valid = _make_lead(
      d_rel=75.0, y_rel=1.7, d_path=1.7, v_rel=-2.0, v_lead=27.0,
      model_prob=0.97, radar_track_id=-9200,
    )
    role_debug = {"reasons": {"lead0": decoy_reason, "lead1": "adjacent_lane"}}

    selected = None
    source = None
    for sample_idx in range(7):
      mpc._lead_stability_phantom_slots = (decoy_phantom, False)
      selected, source = mpc._select_cruise_cap_raw_lead(
        (nearer_decoy, farther_valid), role_debug, 100.0 + sample_idx * 0.1, "cruise", None,
      )
      assert mpc.raw_cruise_cap_debug["source"] == "lead1_raw_path"
      assert mpc.raw_cruise_cap_debug["track_id"] == -9200
      if sample_idx < 6:
        assert selected is None
        assert source is None

    assert selected is farther_valid
    assert source == "lead1_raw_path"
    assert mpc.raw_cruise_cap_debug["reason"] == "dwell_complete"

  def test_farther_stable_raw_path_caps_cruise_despite_nearer_lateral_decoy(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    nearer_decoy = _make_lead(
      d_rel=40.0, y_rel=-8.0, d_path=0.4, v_lat=12.0,
      v_rel=-2.0, v_lead=27.0, model_prob=0.99, radar_track_id=-9300,
    )
    farther_valid = _make_lead(
      d_rel=75.0, y_rel=1.7, d_path=1.7, v_lat=0.0,
      v_rel=-2.0, v_lead=27.0, model_prob=0.97, radar_track_id=-9301,
    )

    for _ in range(6):
      _run_update(mpc, nearer_decoy, farther_valid)
      assert mpc.lead_role_debug["reasons"]["lead0"] == "raw_lateral_departure"
      assert mpc.lead_role_debug["reasons"]["lead1"] == "adjacent_lane"
      assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None

    _run_update(mpc, nearer_decoy, farther_valid)
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["track_id"] == -9301
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "dwell_complete"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead1_raw_path"
    assert mpc.cruise_owned_accel_cap == pytest.approx(0.0)
    assert mpc.params[0, 1] == pytest.approx(0.0)

  def test_exact_release_track_beats_nearer_rejected_decoy_across_slots(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0)
    nearer_missing_id_decoy = _make_lead(
      d_rel=30.0, y_rel=0.4, d_path=0.4, v_rel=-1.0, v_lead=28.0,
      model_prob=0.99, radar_track_id=-1,
    )
    farther_departing_release_track = _make_lead(
      d_rel=42.0, y_rel=-8.0, d_path=0.4, v_lat=12.0,
      v_rel=0.5, v_lead=29.5, model_prob=0.99, radar_track_id=-9400,
    )
    role_debug = {"reasons": {"lead0": "center", "lead1": "raw_lateral_departure"}}

    selected, source = mpc._select_cruise_cap_raw_lead(
      (nearer_missing_id_decoy, farther_departing_release_track),
      role_debug, 100.0, "lead1", -9400,
    )
    assert selected is farther_departing_release_track
    assert source == "lead1_raw_path"
    assert mpc.raw_cruise_cap_debug["reason"] == "previously_controlled_release"
    assert mpc.raw_cruise_cap_debug["track_id"] == -9400

    # The same identity remains authoritative for the bounded release even if
    # its next stabilized sample is a phantom and source has become cruise.
    mpc._lead_stability_phantom_slots = (False, True)
    selected, source = mpc._select_cruise_cap_raw_lead(
      (nearer_missing_id_decoy, farther_departing_release_track),
      role_debug, 100.1, "cruise", None,
    )
    assert selected is farther_departing_release_track
    assert source == "lead1_raw_path"
    assert mpc.raw_cruise_cap_debug["reason"] == "previously_controlled_release"
    assert mpc.raw_cruise_cap_debug["phantom"] is True

  def test_missing_track_identity_never_accumulates_raw_cruise_cap_dwell(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    absent = _make_lead(status=False)
    unknown_identity = _make_lead(
      d_rel=75.0,
      y_rel=1.7,
      d_path=1.7,
      v_lat=0.0,
      v_rel=-2.0,
      v_lead=27.0,
      model_prob=0.97,
      radar_track_id=-1,
    )

    for _ in range(20):
      _run_update(mpc, unknown_identity, absent)
      assert mpc.source == "cruise"
      assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None
      assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "missing_track_identity"
      assert mpc._raw_cruise_cap_candidate_key is None
      assert mpc.params[0, 1] == pytest.approx(ACCEL_MAX)

  def test_zero_track_identity_is_preserved_and_can_qualify(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    absent = _make_lead(status=False)
    valid_zero_identity = _make_lead(
      d_rel=75.0,
      y_rel=1.7,
      d_path=1.7,
      v_lat=0.0,
      v_rel=-2.0,
      v_lead=27.0,
      model_prob=0.97,
      radar_track_id=0,
    )

    for _ in range(6):
      _run_update(mpc, valid_zero_identity, absent)
      assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None

    _run_update(mpc, valid_zero_identity, absent)
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["track_id"] == 0
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "dwell_complete"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead0_raw_path"

  def test_non_hyundai_unknown_track_keeps_legacy_raw_cap_passthrough(self):
    mpc = LongitudinalMpc()
    unknown_identity = _make_lead(
      d_rel=42.0,
      y_rel=0.2,
      d_path=0.2,
      radar_track_id=-1,
    )

    selected, source = mpc._select_cruise_cap_raw_lead(
      (unknown_identity, _make_lead(status=False)), {}, 100.0, "cruise", None,
    )

    assert selected is unknown_identity
    assert source == "lead0_raw_path"
    assert mpc.raw_cruise_cap_debug["reason"] == "non_hyundai_passthrough"

  def test_live_tuned_raw_cap_dwell_is_used_without_restart(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    mpc._live_tune_cfg = build_lead_response_tuning_config({
      "cruise_cap_raw_lead_acquire_dwell_s": 0.2,
    })
    # Keep the test on this already-refreshed snapshot; production refreshes the
    # same dataclass at the ordinary 0.5 s cadence.
    mpc._last_live_tune_refresh_t = float("inf")
    absent = _make_lead(status=False)
    stable_demoted = _make_lead(
      d_rel=75.0,
      y_rel=1.7,
      d_path=1.7,
      v_lat=0.0,
      v_rel=-2.0,
      v_lead=27.0,
      model_prob=0.97,
      radar_track_id=-2002,
    )

    for _ in range(2):
      _run_update(mpc, stable_demoted, absent)
      assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None

    _run_update(mpc, stable_demoted, absent)
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["required_s"] == pytest.approx(0.2)
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "dwell_complete"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead0_raw_path"

  def test_20260716_0954_far_lateral_track_churn_does_not_flap_cruise_cap(self):
    # Measured radarState samples around the two strongest 09:54 PDT flaps.
    # On-road, -1036 stepped aTarget from +0.256 to -0.176 m/s^2 in 57 ms and
    # -1041 stepped it from +0.281 to -0.227 in 39 ms while source remained
    # cruise.  All hypotheses were several metres lateral despite |dPath|<=2m.
    mpc = _make_hyundai_mpc(v_ego=31.7, a_ego=0.25, time_fn=_MonotonicStub(step=0.05))
    absent = _make_lead(status=False)
    for _ in range(20):
      _run_update_with_state(mpc, absent, absent, v_ego=31.7, a_ego=0.25, v_cruise=32.5)

    # repeat count and following dropout count preserve the important native
    # dwell/status pattern, including -1036 lasting 0.55 s (just below the new
    # 0.60 s proof) and -1040 lasting 1.8 s while still explicitly lateral.
    road_runs = (
      ((-1036, 83.4, 5.4, 0.06, -9.5, -1.03), 12, 2),
      ((-1037, 104.0, 9.1, -1.99, -14.2, -0.95), 6, 14),
      ((-1038, 98.6, 9.5, -1.06, -13.7, -1.23), 18, 0),
      ((-1039, 102.7, 9.9, -2.00, -13.5, -1.25), 8, 14),
      ((-1040, 90.1, 8.4, 0.05, -12.4, -0.20), 36, 0),
      ((-1041, 94.8, 9.0, -1.20, -13.7, -0.12), 15, 5),
      ((-1042, 70.9, 6.8, 0.50, -11.0, -1.70), 2, 5),
      ((-1043, 88.8, 8.4, -0.19, -12.8, -0.68), 11, 16),
      ((-1044, 88.2, 8.5, -0.02, -12.0, -2.36), 9, 5),
    )
    cap_sources = []
    accel_limits = []
    first_step_jerks = []
    for (track_id, d_rel, y_rel, d_path, v_lat, v_rel), repeat_count, dropout_count in road_runs:
      for _ in range(repeat_count):
        lead0 = _make_lead(
          d_rel=d_rel, y_rel=y_rel, d_path=d_path, v_lat=v_lat,
          v_rel=v_rel, v_lead=31.7 + v_rel, model_prob=0.98,
          radar_track_id=track_id,
        )
        lead1 = _make_lead(
          d_rel=d_rel, y_rel=y_rel, d_path=d_path, v_lat=v_lat + 5.0,
          v_rel=v_rel, v_lead=31.7 + v_rel, model_prob=0.96,
          radar_track_id=track_id,
        )
        _run_update_with_state(mpc, lead0, lead1, v_ego=31.7, a_ego=0.25, v_cruise=32.5)
        assert mpc.source == "cruise"
        cap_sources.append(mpc.acc_source_debug["lead_present_cruise_accel_cap_source"])
        accel_limits.append(float(mpc.params[0, 1]))
        first_step_jerks.append(float(mpc.j_solution[0]))
      for _ in range(dropout_count):
        _run_update_with_state(mpc, absent, absent, v_ego=31.7, a_ego=0.25, v_cruise=32.5)
        assert mpc.source == "cruise"
        cap_sources.append(mpc.acc_source_debug["lead_present_cruise_accel_cap_source"])
        accel_limits.append(float(mpc.params[0, 1]))
        first_step_jerks.append(float(mpc.j_solution[0]))

    assert len(cap_sources) == 178
    assert cap_sources == [None] * len(cap_sources)
    assert accel_limits == pytest.approx([ACCEL_MAX] * len(accel_limits))
    assert min(first_step_jerks) > -0.20

  def test_previously_controlled_departure_keeps_bounded_same_track_cap(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    absent = _make_lead(status=False)
    controlled = _make_lead(
      d_rel=36.0, y_rel=0.1, d_path=0.1, v_lat=0.0,
      v_rel=0.0, v_lead=29.0, model_prob=0.99, radar_track_id=-3001,
    )
    for _ in range(12):
      _run_update(mpc, controlled, absent)
    assert mpc.source == "lead0"

    departing = _make_lead(
      d_rel=36.5, y_rel=-8.0, d_path=0.4, v_lat=12.0,
      v_rel=0.5, v_lead=29.5, model_prob=0.99, radar_track_id=-3001,
    )
    _run_update(mpc, departing, absent)

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "previously_controlled_release"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead0_raw_path"
    assert mpc.acc_source_debug["source_transition_active"] is True
    assert mpc.cruise_owned_accel_cap is not None
    assert mpc.params[0, 1] < ACCEL_MAX

    for _ in range(6):
      _run_update(mpc, departing, absent)
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] is None
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "raw_lateral_departure"

  @pytest.mark.parametrize(
    "lead",
    (
      _make_lead(
        d_rel=36.0, y_rel=0.1, d_path=0.1, v_lat=0.0,
        v_rel=-1.0, v_lead=28.0, model_prob=0.99, radar_track_id=-4001,
      ),
      _make_lead(
        d_rel=42.0, y_rel=3.2, d_path=3.2, v_lat=-1.2,
        v_rel=-1.0, v_lead=28.0, model_prob=0.99, radar_track_id=-4002,
      ),
    ),
    ids=("centered_close", "converging_cutin"),
  )
  def test_genuine_centered_and_cutin_leads_keep_immediate_control(self, lead):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    _run_update(mpc, lead, _make_lead(status=False))

    assert mpc.lead_role_debug["control_status"]["lead0"] is True
    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead0_control"
    assert mpc.acc_source_debug["raw_cruise_cap_candidate"]["reason"] == "control_lead_selected"
    assert mpc.params[0, 1] == pytest.approx(ACCEL_MAX)

  def test_far_closing_lead_stays_cruise_owned_until_target_gap_approaches(self):
    mpc = _make_hyundai_mpc(v_ego=28.5, a_ego=0.0, time_fn=_MonotonicStub(step=0.2))

    _run_update_with_state(
      mpc,
      _make_lead(d_rel=65.5, y_rel=-0.24, d_path=0.40, v_lat=0.78,
                 v_rel=-1.93, v_lead=26.57, model_prob=0.97),
      _make_lead(status=False),
      v_ego=28.5,
    )

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "cruise_hold"
    assert mpc.acc_source_debug["raw_obstacle_requires_owner"] is False
    assert mpc.acc_source_debug["raw_near_target_for_reacquire"] is False
    assert mpc.acc_source_debug["raw_ttc_to_headway_s"] > mpc.acc_source_debug["approach_reacquire_ttc_threshold_s"]
    assert mpc.cruise_owned_accel_cap == pytest.approx(0.0)

    _run_update_with_state(
      mpc,
      _make_lead(d_rel=51.6, y_rel=0.35, d_path=-0.04, v_lat=-2.04,
                 v_rel=-4.27, v_lead=23.32, model_prob=0.97),
      _make_lead(status=False),
      v_ego=27.6,
    )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "raw_obstacle_hold"
    assert mpc.acc_source_debug["raw_obstacle_requires_owner"] is True
    assert mpc.acc_source_debug["raw_ttc_to_headway_s"] <= mpc.acc_source_debug["approach_reacquire_ttc_threshold_s"]

  def test_steady_freeway_closing_lead_stays_cruise_owned_outside_target_gap(self):
    mpc = _make_hyundai_mpc(v_ego=17.4, a_ego=0.0, time_fn=_MonotonicStub(step=0.2))

    _run_update_with_state(
      mpc,
      _make_lead(d_rel=44.1, y_rel=-0.3, d_path=-0.1,
                 v_rel=-2.9, v_lead=14.5, a_lead=0.0, model_prob=0.97),
      _make_lead(status=False),
      v_ego=17.4,
    )

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "cruise_hold"
    assert mpc.acc_source_debug["raw_near_target_for_reacquire"] is False
    assert mpc.acc_source_debug["raw_obstacle_requires_owner"] is False
    assert mpc.acc_source_debug["raw_gap_surplus_m"] > 20.0
    assert mpc.acc_source_debug["raw_ttc_to_headway_s"] > mpc.acc_source_debug["approach_reacquire_ttc_threshold_s"]

  def test_far_closing_release_holds_ownership_inside_ttc_hysteresis_band(self):
    # 2026-07-06 freeway trace: raw TTC-to-headway frame jitter is ~2.5 s p90,
    # and the bc6c853f9 threshold split made release (far_closing_cruise) and
    # reacquire share one TTC boundary — observed as 0.1 s ownership
    # double-flips mid-approach. The release threshold must now sit a full
    # hysteresis gap above the reacquire threshold, holding the current owner
    # inside the dead band in BOTH directions.
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.2))

    # Warmup frame to learn the effective follow gap, then place the lead by
    # gap SURPLUS so TTC-to-headway = surplus / closing is exact by design.
    def closing_lead(surplus_m):
      d_rel = STOP_DISTANCE + mpc.current_t_follow * 29.0 + surplus_m
      return _make_lead(d_rel=d_rel, y_rel=0.04, d_path=0.04, v_rel=-2.0, v_lead=27.0, model_prob=0.97)

    _run_update_with_state(mpc, closing_lead(42.0), _make_lead(status=False), v_ego=29.0)

    # Acquire: sustained closing approach inside the reacquire TTC threshold
    # (surplus 8 m at closing 2 m/s -> TTC 4.0 s <= 4.5 s steady threshold).
    for _ in range(8):
      _run_update_with_state(mpc, closing_lead(8.0), _make_lead(status=False), v_ego=29.0)
    assert mpc.source == "lead0"

    # Dead band: TTC 6.0 s sits between the reacquire threshold (4.5 s) and
    # release threshold (4.5 + 2.5 s) — lead ownership must hold.
    _run_update_with_state(mpc, closing_lead(12.0), _make_lead(status=False), v_ego=29.0)
    reacquire_thr = mpc.acc_source_debug["approach_reacquire_ttc_threshold_s"]
    release_thr = mpc.acc_source_debug["far_closing_release_ttc_threshold_s"]
    assert release_thr == pytest.approx(reacquire_thr + 2.5)
    assert reacquire_thr < mpc.acc_source_debug["raw_ttc_to_headway_s"] <= release_thr
    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] != "far_closing_cruise"

    # Beyond the release threshold (TTC 9.0 s > 7.0 s) the still-closing lead
    # genuinely releases to cruise.
    _run_update_with_state(mpc, closing_lead(18.0), _make_lead(status=False), v_ego=29.0)
    assert mpc.acc_source_debug["raw_ttc_to_headway_s"] > mpc.acc_source_debug["far_closing_release_ttc_threshold_s"]
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "far_closing_cruise"

    # Back inside the dead band from the cruise side: no reacquire either —
    # the band holds whichever side currently owns the obstacle.
    _run_update_with_state(mpc, closing_lead(11.0), _make_lead(status=False), v_ego=29.0)
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["approach_reacquire"] is False

  def test_far_closing_release_hysteresis_zero_sentinel_restores_shared_threshold(self):
    params = Params()
    params.put("Longitudinal.LiveTune.ApproachReleaseTtcHysteresisS", 0.0)
    try:
      mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.2))

      def closing_lead(surplus_m):
        d_rel = STOP_DISTANCE + mpc.current_t_follow * 29.0 + surplus_m
        return _make_lead(d_rel=d_rel, y_rel=0.04, d_path=0.04, v_rel=-2.0, v_lead=27.0, model_prob=0.97)

      _run_update_with_state(mpc, closing_lead(42.0), _make_lead(status=False), v_ego=29.0)
      for _ in range(8):
        _run_update_with_state(mpc, closing_lead(8.0), _make_lead(status=False), v_ego=29.0)
      assert mpc.source == "lead0"

      # With the sentinel the same TTC 6.0 s frame that the hysteresis test
      # holds through releases immediately (legacy shared-threshold behavior).
      _run_update_with_state(mpc, closing_lead(12.0), _make_lead(status=False), v_ego=29.0)
      assert mpc.acc_source_debug["far_closing_release_ttc_threshold_s"] == pytest.approx(
        mpc.acc_source_debug["approach_reacquire_ttc_threshold_s"])
      assert mpc.source == "cruise"
      assert mpc.acc_source_debug["reason"] == "far_closing_cruise"
    finally:
      params.remove("Longitudinal.LiveTune.ApproachReleaseTtcHysteresisS")

  def test_classifier_demotion_hold_releases_when_raw_radar_is_also_gone(self, monkeypatch):
    # Safety backstop: if the raw radarstate has no lead anywhere,
    # corroboration fails and the demotion hold must NOT engage — otherwise
    # stale state would be held through a genuine cut-out.
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0)

    stable_lead = _make_lead(d_rel=35.6, y_rel=1.0, d_path=1.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    for _ in range(10):
      _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert abs(mpc._hyundai_virtual_lead.dPath) > 0.85

    # Total sensor loss on both slots — no raw lead anywhere in radarstate.
    _run_update(mpc, _make_lead(status=False), _make_lead(status=False))

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"

  def test_classifier_demotion_hold_releases_when_raw_lead_is_different_object(self, monkeypatch):
    # Safety backstop: if the only fresh raw lead is at a very different
    # dRel from the stored virtual lead, corroboration must fail so the
    # planner does not hold a phantom of the previous lead while an
    # unrelated object appears far ahead.
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0)

    stable_lead = _make_lead(d_rel=35.6, y_rel=1.0, d_path=1.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    for _ in range(10):
      _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert abs(mpc._hyundai_virtual_lead.dPath) > 0.85

    # Raw lead appears at 100m (dRel delta ~65m from stored ~35m), well
    # beyond the 10m corroboration tolerance, and also out of center lane.
    far_lead = _make_lead(d_rel=100.0, y_rel=3.0, d_path=3.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    _run_update(mpc, far_lead, _make_lead(status=False))

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"

  def test_center_control_grace_expires_after_sustained_demotion(self):
    # The classifier grace is a brief demotion mask only. If the lead stays
    # outside the center-lane gate, ownership must release to cruise after the
    # grace window expires.
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))

    stable_lead = _make_lead(d_rel=35.6, y_rel=1.0, d_path=1.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    for _ in range(10):
      _run_update(mpc, stable_lead, _make_lead(status=False))
    assert mpc.source == "lead0"
    assert abs(mpc._hyundai_virtual_lead.dPath) > 0.85

    # Sustained demotion with the same moderate path spike that the classifier
    # will grace briefly but not indefinitely.
    demoted_lead = _make_lead(d_rel=35.6, y_rel=3.0, d_path=3.0, v_lat=0.0, v_rel=0.0, v_lead=29.0, model_prob=0.97)
    sources = []
    for _ in range(5):
      _run_update(mpc, demoted_lead, _make_lead(status=False))
      sources.append(mpc.source)

    assert sources[0] == "lead0", f"first demotion frame should hold, got {sources}"
    assert sources[-1] == "cruise", f"sustained demotion should eventually release, got {sources}"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"
    held_frames = sum(1 for s in sources if s == "lead0")
    assert held_frames <= 2, f"center grace should not hold beyond the 0.25s window, got {sources}"

  def test_dropout_hold_rejects_real_pullaway_and_releases_to_cruise_immediately(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    stable_lead = _make_lead(d_rel=35.4, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.00, v_lead=29.0, model_prob=0.98)
    for _ in range(16):
      _run_update(mpc, stable_lead, _make_lead(status=False))

    for d_rel in (35.8, 36.1, 36.4):
      _run_update(
        mpc,
        _make_lead(d_rel=d_rel, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=1.50, v_lead=30.50, model_prob=0.98),
        _make_lead(status=False),
      )
    assert mpc.source == "lead0"

    _run_update(mpc, _make_lead(status=False), _make_lead(status=False))

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "no_control_lead"
    assert mpc.acc_source_debug["source_transition_active"] is True
    assert mpc.acc_source_debug["close_lead_memory_active"] is False
    assert mpc.hyundai_virtual_lead_debug["active"] is False

  def test_lead_to_cruise_transition_caps_accel_then_expires(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    stable_lead = _make_lead(d_rel=35.4, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.00, v_lead=29.0, model_prob=0.98)
    for _ in range(16):
      _run_update(mpc, stable_lead, _make_lead(status=False))

    for d_rel in (35.9, 36.2, 36.5):
      _run_update(
        mpc,
        _make_lead(d_rel=d_rel, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=1.50, v_lead=30.50, model_prob=0.98),
        _make_lead(status=False),
      )
    _run_update(mpc, _make_lead(status=False), _make_lead(status=False))

    first_cap = float(mpc.acc_source_debug["source_transition_accel_cap"])
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["source_transition_active"] is True
    assert 0.20 <= first_cap <= 0.35
    assert mpc.last_cruise_response_model is not None
    assert mpc.last_cruise_response_model.max_accel_mps2 == pytest.approx(first_cap)

    for _ in range(6):
      _run_update(mpc, _make_lead(status=False), _make_lead(status=False))

    assert mpc.acc_source_debug["source_transition_active"] is False
    assert mpc.last_cruise_response_model is not None
    assert mpc.last_cruise_response_model.max_accel_mps2 > first_cap + 0.20

  def test_rlog_245_closing_follow_dropout_keeps_cruise_from_accelerating(self):
    """A July-16 freeway lead dropout occurred at normal headway, not 25 m.

    The logged lead was centered and closing at roughly 36.8 m while ego was
    18.8 m/s.  The association then vanished long enough for the ordinary
    source-transition ramp to expire, and cruise started accelerating toward
    the still-stopped physical car.  A closing lead inside the dynamic follow
    envelope must keep the zero-accel safety memory armed through that gap.
    """
    mpc = _make_hyundai_mpc(v_ego=18.8, a_ego=-0.1, time_fn=_MonotonicStub(step=0.2))
    absent = _make_lead(status=False)
    closing_lead = _make_lead(
      d_rel=36.8, y_rel=0.04, d_path=0.04, v_lat=0.10,
      v_rel=-5.0, v_lead=13.8, a_lead=-0.5, model_prob=0.97,
      radar_track_id=-245,
    )

    for _ in range(16):
      _run_update_with_state(mpc, closing_lead, absent, v_ego=18.8, a_ego=-0.1, v_cruise=40.0)
    assert mpc.source == "lead0"

    # Outlast both the raw phantom and the three-second normal transition cap.
    for _ in range(20):
      _run_update_with_state(mpc, absent, absent, v_ego=18.8, a_ego=-0.1, v_cruise=40.0)

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["closing_dropout_memory_active"] is True
    assert mpc.acc_source_debug["source_transition_composed_accel_cap"] == pytest.approx(0.0)
    assert mpc.last_cruise_response_model is not None
    assert mpc.last_cruise_response_model.max_accel_mps2 == pytest.approx(0.0)

  @staticmethod
  def _run_low_speed_persistent_lead_release(*, release_a_lead: float,
                                             urgent_lead_decel: float | None = None) -> LongitudinalMpc:
    mpc = _make_hyundai_mpc(v_ego=6.0, a_ego=1.0, time_fn=_MonotonicStub(step=0.2))
    absent = _make_lead(status=False)
    queue_lead = _make_lead(
      d_rel=10.0, y_rel=0.04, d_path=0.04, v_rel=0.0,
      v_lead=6.0, a_lead=0.0, model_prob=0.98, radar_track_id=7,
    )
    for _ in range(16):
      _run_update_with_state(mpc, queue_lead, absent, v_ego=6.0, a_ego=1.0, v_cruise=19.0)
    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "low_speed_queue_hold"
    if urgent_lead_decel is not None:
      mpc._live_tune_cfg = replace(
        mpc._live_tune_cfg,
        cruise_relatch_urgent_lead_decel_mps2=urgent_lead_decel,
      )
      mpc._last_live_tune_refresh_t = float("inf")

    departing_lead = _make_lead(
      d_rel=25.0, y_rel=0.04, d_path=0.04, v_rel=3.0,
      v_lead=9.0, a_lead=release_a_lead, model_prob=0.98, radar_track_id=7,
    )
    _run_update_with_state(mpc, departing_lead, absent, v_ego=6.0, a_ego=1.0, v_cruise=19.0)
    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["reason"] == "filtered_pullaway_immediate"
    return mpc

  def test_persistent_benign_departing_lead_uses_live_cap_not_dropout_cap(self):
    mpc = self._run_low_speed_persistent_lead_release(release_a_lead=0.0)

    assert mpc.acc_source_debug["source_transition_accel_cap"] == pytest.approx(0.3)
    assert mpc.acc_source_debug["source_transition_live_lead_braking_guard"] is False
    assert mpc.acc_source_debug["source_transition_composed_accel_cap"] == pytest.approx(1.3)
    assert mpc.cruise_owned_accel_cap == pytest.approx(1.3)
    assert mpc.last_cruise_response_model.max_accel_mps2 == pytest.approx(1.3)
    assert mpc.params[0, 1] == pytest.approx(1.3)

  def test_persistent_departing_but_braking_lead_retains_transition_cap(self):
    # Positive vRel is not permission to chase a lead already braking hard.
    # Stopping-need is deliberately below its 0.8 threshold in this geometry;
    # this oracle therefore proves the explicit braking guard owns the result.
    mpc = self._run_low_speed_persistent_lead_release(release_a_lead=-1.0)

    assert mpc.acc_source_debug["stopping_need_decel_mps2"] < mpc.acc_source_debug["stopping_need_threshold_mps2"]
    assert mpc.acc_source_debug["source_transition_accel_cap"] == pytest.approx(0.3)
    assert mpc.acc_source_debug["source_transition_live_lead_braking_guard"] is True
    assert mpc.acc_source_debug["source_transition_composed_accel_cap"] == pytest.approx(0.3)
    assert mpc.cruise_owned_accel_cap == pytest.approx(0.3)
    assert mpc.last_cruise_response_model.max_accel_mps2 == pytest.approx(0.3)
    assert mpc.params[0, 1] == pytest.approx(0.3)

  def test_zero_urgent_lead_decel_sentinel_does_not_classify_coasting_as_braking(self):
    mpc = self._run_low_speed_persistent_lead_release(
      release_a_lead=0.0,
      urgent_lead_decel=0.0,
    )

    assert mpc.acc_source_debug["source_transition_live_lead_match"] is True
    assert mpc.acc_source_debug["source_transition_live_lead_braking_guard"] is False
    assert mpc.acc_source_debug["source_transition_composed_accel_cap"] == pytest.approx(1.3)

  def test_different_surviving_lead_cannot_cancel_released_lead_transition_cap(self):
    mpc = _make_hyundai_mpc(v_ego=6.0, a_ego=1.0, time_fn=_MonotonicStub(step=0.2))
    absent = _make_lead(status=False)
    queue_lead = _make_lead(
      d_rel=10.0, y_rel=0.04, d_path=0.04, v_rel=0.0,
      v_lead=6.0, a_lead=0.0, model_prob=0.98, radar_track_id=7,
    )
    for _ in range(16):
      _run_update_with_state(mpc, queue_lead, absent, v_ego=6.0, a_ego=1.0, v_cruise=19.0)
    other_lead = _make_lead(
      d_rel=25.0, y_rel=0.04, d_path=0.04, v_rel=3.0,
      v_lead=9.0, a_lead=0.0, model_prob=0.98, radar_track_id=8,
    )
    for _ in range(4):
      _run_update_with_state(mpc, absent, other_lead, v_ego=6.0, a_ego=1.0, v_cruise=19.0)

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["source_transition_from"] == "lead0"
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead1_control"
    assert mpc.acc_source_debug["source_transition_live_lead_match"] is False
    assert mpc.acc_source_debug["source_transition_accel_cap"] == pytest.approx(0.3)
    assert mpc.acc_source_debug["source_transition_composed_accel_cap"] == pytest.approx(0.3)
    assert mpc.cruise_owned_accel_cap == pytest.approx(0.3)

  def test_same_slot_replacement_cannot_self_certify_as_released_lead(self):
    mpc = _make_hyundai_mpc(v_ego=6.0, a_ego=1.0, time_fn=_MonotonicStub(step=0.2))
    absent = _make_lead(status=False)
    queue_lead = _make_lead(
      d_rel=10.0, y_rel=0.04, d_path=0.04, v_rel=0.0,
      v_lead=6.0, a_lead=0.0, model_prob=0.98, radar_track_id=7,
    )
    for _ in range(16):
      _run_update_with_state(mpc, queue_lead, absent, v_ego=6.0, a_ego=1.0, v_cruise=19.0)

    replacement = _make_lead(
      d_rel=25.0, y_rel=0.04, d_path=0.04, v_rel=3.0,
      v_lead=9.0, a_lead=0.0, model_prob=0.98, radar_track_id=8,
    )
    _run_update_with_state(mpc, replacement, absent, v_ego=6.0, a_ego=1.0, v_cruise=19.0)

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["source_transition_from"] == "lead0"
    assert mpc.acc_source_debug["source_transition_track_id"] == 7
    assert mpc.acc_source_debug["lead_present_cruise_accel_cap_source"] == "lead0_control"
    assert mpc.acc_source_debug["source_transition_live_lead_match"] is False
    assert mpc.acc_source_debug["source_transition_accel_cap"] == pytest.approx(0.3)
    assert mpc.acc_source_debug["source_transition_composed_accel_cap"] == pytest.approx(0.3)
    assert mpc.cruise_owned_accel_cap == pytest.approx(0.3)

  def test_cutin_promotion_reaches_virtual_duplicate_lead(self, monkeypatch):
    monotonic = _MonotonicStub(step=0.2)
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      monotonic,
    )
    mpc = _make_hyundai_mpc()

    _run_update(
      mpc,
      _make_lead(d_rel=43.0, y_rel=3.8, d_path=3.8, v_lat=0.0, v_rel=-0.8, model_prob=0.94),
      _make_lead(d_rel=43.1, y_rel=3.9, d_path=3.9, v_lat=0.0, v_rel=-0.8, model_prob=0.92),
    )

    _run_update(
      mpc,
      _make_lead(d_rel=42.6, y_rel=3.2, d_path=3.2, v_lat=-1.2, v_rel=-0.8, model_prob=0.94),
      _make_lead(d_rel=42.7, y_rel=3.3, d_path=3.3, v_lat=-1.0, v_rel=-0.8, model_prob=0.92),
    )

    assert mpc.lead_role_debug["virtual_duplicate"]["active"] is True
    assert mpc._virtual_cutin_event_t is not None

  def test_raw_near_gap_reacquires_lead_from_cruise_before_filtered_state_catches_up(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    for _ in range(4):
      _run_update(
        mpc,
        _make_lead(d_rel=44.0, y_rel=0.04, d_path=0.04, v_lat=0.40, v_rel=0.6, v_lead=29.7, model_prob=0.96),
        _make_lead(d_rel=43.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.6, v_lead=29.7, model_prob=0.93),
      )

    assert mpc.source == "cruise"

    _run_update(
      mpc,
      _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.40, v_rel=-0.4, v_lead=28.6, model_prob=0.96),
      _make_lead(d_rel=38.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=-0.4, v_lead=28.6, model_prob=0.93),
    )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "raw_gap_hold"

  def test_low_speed_slow_lead_stays_owned_even_with_large_time_headway(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=1.6, a_ego=0.27)

    _run_update(
      mpc,
      _make_lead(d_rel=18.3, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.5, v_lead=2.11, a_lead=0.03, model_prob=0.96),
      _make_lead(status=False),
    )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["reason"] == "low_speed_queue_hold"
    assert mpc.acc_source_debug["low_speed_queue_hold"] is True

  def test_low_speed_distant_lead_can_still_release_to_cruise(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=1.8, a_ego=0.1)

    _run_update(
      mpc,
      _make_lead(d_rel=30.0, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.8, v_lead=2.6, a_lead=0.05, model_prob=0.96),
      _make_lead(status=False),
    )

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["low_speed_queue_hold"] is False

  def test_cruise_owned_lead_present_uses_tapered_accel_cap(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    _configure_vibe_accel(enabled=True, personality=0)
    mpc = _make_hyundai_mpc(v_ego=9.0, a_ego=0.1)

    _run_update(
      mpc,
      _make_lead(d_rel=19.0, y_rel=0.04, d_path=0.04, v_lat=0.10, v_rel=0.6, v_lead=9.8, a_lead=0.1, model_prob=0.96),
      _make_lead(status=False),
    )

    assert mpc.source == "cruise"
    # Mild 0.8 m/s pullaway with aLead 0.1: kinematic chase grants a modest
    # proportional margin above the gentle cap (exact value depends on the
    # vibe-mapped t_follow); the structural assertions below are the point.
    assert 0.30 < mpc.lead_present_cruise_accel_cap < 0.55
    assert mpc.last_cruise_response_model is not None
    assert mpc.last_cruise_response_model.max_accel_mps2 == pytest.approx(mpc.lead_present_cruise_accel_cap)
    assert mpc.params[0, 1] == pytest.approx(mpc.lead_present_cruise_accel_cap)

  def test_near_gap_follow_keeps_full_accel_limit(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc()

    _run_update(
      mpc,
      _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.40, v_rel=-0.4, v_lead=28.6, model_prob=0.96),
      _make_lead(d_rel=38.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=-0.4, v_lead=28.6, model_prob=0.93),
    )

    assert mpc.source == "lead0"
    assert mpc.params[0, 1] == pytest.approx(ACCEL_MAX)

  def test_reclaim_stays_active_through_benign_raw_lead_accel_jitter(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=38.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )
    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=54.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=53.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )

    assert mpc.source == "lead0"

    _run_update(
      mpc,
      _make_lead(d_rel=54.6, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=-0.55, model_prob=0.96),
      _make_lead(d_rel=54.55, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=-0.55, model_prob=0.93),
    )

    assert mpc.source == "lead0"
    assert mpc.gap_reclaim_accel_floor > 0.0
    assert mpc.acc_source_debug["raw_reclaim_safety_override"] is False

  def test_reclaim_relaxes_active_obstacle_even_when_raw_and_filtered_align(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=38.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )
    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=54.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=53.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )

    assert mpc.source == "lead0"
    assert mpc.gap_reclaim_accel_floor > 0.0
    assert mpc.gap_reclaim_obstacle_push > 0.15
    assert mpc.acc_source_debug["stabilization_push_suppressed"] is True
    assert mpc.acc_source_debug["raw_reclaim_safety_override"] is False

  def test_reclaim_blend_tapers_when_personality_cap_is_higher_than_comfort_cap(self, monkeypatch):
    _configure_vibe_accel(enabled=True, personality=0)
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=38.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )

    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=54.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.4, v_lead=29.4, a_lead=0.1, model_prob=0.96),
        _make_lead(d_rel=53.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.4, v_lead=29.4, a_lead=0.1, model_prob=0.93),
      )

    assert mpc.gap_reclaim_effective_cap > mpc._live_tune_cfg.gap_reclaim_max_accel
    assert 0.0 < mpc._gap_reclaim_blend < 1.0

  def test_reclaim_uses_base_planner_max_accel_when_vibe_accel_is_disabled(self):
    _configure_vibe_accel(enabled=False)
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0)

    assert mpc._get_gap_reclaim_personality_max_accel(29.0) == pytest.approx(get_max_accel(29.0))

  def test_reclaim_room_tapers_when_ego_accel_is_already_built(self, monkeypatch):
    _configure_vibe_accel(enabled=True, personality=0)
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    cold_mpc = _make_hyundai_mpc(v_ego=33.5, a_ego=0.0)
    loaded_mpc = _make_hyundai_mpc(v_ego=33.5, a_ego=0.9)

    for mpc in (cold_mpc, loaded_mpc):
      for _ in range(2):
        _run_update(
          mpc,
          _make_lead(d_rel=44.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.96),
          _make_lead(d_rel=44.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.93),
        )
      for _ in range(2):
        _run_update(
          mpc,
          _make_lead(d_rel=58.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.4, v_lead=33.9, a_lead=0.1, model_prob=0.96),
          _make_lead(d_rel=57.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.4, v_lead=33.9, a_lead=0.1, model_prob=0.93),
        )

    assert cold_mpc.acc_source_debug["gap_reclaim_projection_scale"] == pytest.approx(1.0)
    assert loaded_mpc.acc_source_debug["gap_reclaim_projection_scale"] < 0.45
    assert loaded_mpc.gap_reclaim_obstacle_push < cold_mpc.gap_reclaim_obstacle_push - 0.5

  def test_reclaim_dynamic_optimism_releases_quickly_when_pullaway_stops(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=33.5, a_ego=0.5)

    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=44.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=44.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.93),
      )
    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=58.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.5, v_lead=34.0, a_lead=0.1, model_prob=0.96),
        _make_lead(d_rel=57.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.5, v_lead=34.0, a_lead=0.1, model_prob=0.93),
      )

    optimistic_push = mpc.gap_reclaim_obstacle_push
    optimistic_reclaim_vlead = float(mpc._hyundai_reclaim_lead.vLead)

    _run_update(
      mpc,
      _make_lead(d_rel=55.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.96),
      _make_lead(d_rel=54.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=33.5, a_lead=0.0, model_prob=0.93),
    )

    assert mpc.gap_reclaim_obstacle_push < optimistic_push - 0.5
    assert float(mpc._hyundai_reclaim_lead.vLead) < optimistic_reclaim_vlead - 0.10
    assert float(mpc._hyundai_reclaim_lead.vLead) <= float(mpc._hyundai_virtual_lead.vLead) + 0.35

  def test_reclaim_raw_safety_override_still_engages_for_real_closing(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=38.5, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=38.45, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )
    for _ in range(2):
      _run_update(
        mpc,
        _make_lead(d_rel=54.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.96),
        _make_lead(d_rel=53.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.93),
      )

    _run_update(
      mpc,
      _make_lead(d_rel=52.0, y_rel=0.04, d_path=0.04, v_lat=0.35, v_rel=-1.1, v_lead=27.9, a_lead=-0.8, model_prob=0.96),
      _make_lead(d_rel=51.95, y_rel=0.07, d_path=0.07, v_lat=4.00, v_rel=-1.1, v_lead=27.9, a_lead=-0.8, model_prob=0.93),
    )

    assert mpc.source == "lead0"
    assert mpc.acc_source_debug["raw_reclaim_safety_override"] is True

  def test_new_slower_lead_sets_acquire_window_and_preview_mode(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=33.5, a_ego=0.0)

    _run_update(mpc, _make_lead(status=False), _make_lead(status=False))
    _run_update(
      mpc,
      _make_lead(d_rel=90.0, y_rel=0.02, d_path=0.02, v_lat=0.2, v_rel=-6.5, v_lead=27.0, a_lead=0.0, model_prob=0.97),
      _make_lead(status=False),
      v_cruise=40.0,
    )

    preview_debug = mpc.lead_approach_preview_debug["lead0"]

    assert preview_debug["acquire"]["active"] is True
    assert preview_debug["mode"] == "acquire"
    assert float(preview_debug["preview_buffer_m"]) >= 5.9
    assert mpc.lead_approach_preview[0] >= 5.9
    assert mpc.source == "lead0"

  def test_adjacent_awareness_preview_lowers_obstacle_before_control_handoff(self):
    baseline_mpc = _make_hyundai_mpc(v_ego=34.73, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    preview_mpc = _make_hyundai_mpc(v_ego=34.73, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))

    primary_lead = _make_lead(
      d_rel=73.55,
      y_rel=0.0,
      d_path=0.0,
      v_lat=0.15,
      v_rel=-2.73,
      v_lead=32.0,
      a_lead=0.0,
      model_prob=1.0,
      radar_track_id=-1001,
    )
    adjacent_lead_prime_0 = _make_lead(
      d_rel=81.0,
      y_rel=1.9,
      d_path=1.9,
      v_lat=-0.5,
      v_rel=-12.73,
      v_lead=22.0,
      a_lead=0.0,
      model_prob=0.92,
      radar_track_id=-1002,
    )
    adjacent_lead_prime = _make_lead(
      d_rel=80.0,
      y_rel=1.85,
      d_path=1.85,
      v_lat=-0.5,
      v_rel=-12.73,
      v_lead=22.0,
      a_lead=0.0,
      model_prob=0.92,
      radar_track_id=-1002,
    )
    adjacent_lead = _make_lead(
      d_rel=79.0,
      y_rel=1.8,
      d_path=1.8,
      v_lat=-0.5,
      v_rel=-12.73,
      v_lead=22.0,
      a_lead=0.0,
      model_prob=0.92,
      radar_track_id=-1002,
    )

    for mpc in (baseline_mpc, preview_mpc):
      _run_update(mpc, primary_lead, _make_lead(status=False), v_cruise=40.0)

    _run_update(baseline_mpc, primary_lead, _make_lead(status=False), v_cruise=40.0)
    _run_update(preview_mpc, primary_lead, adjacent_lead_prime_0, v_cruise=40.0)
    _run_update(baseline_mpc, primary_lead, _make_lead(status=False), v_cruise=40.0)
    _run_update(preview_mpc, primary_lead, adjacent_lead_prime, v_cruise=40.0)
    _run_update(baseline_mpc, primary_lead, _make_lead(status=False), v_cruise=40.0)
    _run_update(preview_mpc, primary_lead, adjacent_lead, v_cruise=40.0)

    assert baseline_mpc.source == "cruise"
    assert preview_mpc.source == "cruise"
    assert baseline_mpc.cruise_owned_accel_cap == pytest.approx(0.0)
    assert preview_mpc.cruise_owned_accel_cap == pytest.approx(0.0)
    assert preview_mpc.adjacent_awareness_preview_debug["active"] is True
    assert preview_mpc.adjacent_awareness_preview_debug["applied"] is True
    assert preview_mpc.adjacent_awareness_preview_debug["slot"] == "lead1"
    assert preview_mpc.adjacent_awareness_preview_debug["toward_center_hist_mps"] == pytest.approx(0.5)
    assert preview_mpc.adjacent_awareness_preview_debug["toward_center_hist_confirm_frames"] >= 2
    assert preview_mpc.adjacent_awareness_preview_debug["history_identity_match"] is True
    assert preview_mpc.lead_approach_preview_debug["lead1"]["active"] is False
    assert float(preview_mpc.params[0, 2]) < float(baseline_mpc.params[0, 2]) - 3.0

  def test_adjacent_awareness_preview_requires_observed_path_convergence(self):
    mpc = _make_hyundai_mpc(v_ego=34.73, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    primary_lead = _make_lead(
      d_rel=73.55,
      y_rel=0.0,
      d_path=0.0,
      v_lat=0.15,
      v_rel=-2.73,
      v_lead=32.0,
      model_prob=1.0,
      radar_track_id=-1001,
    )
    parallel_adjacent = _make_lead(
      d_rel=80.0,
      y_rel=1.8,
      d_path=1.8,
      v_lat=-0.5,
      v_rel=-12.73,
      v_lead=22.0,
      model_prob=0.92,
      radar_track_id=-1002,
    )

    _run_update(mpc, primary_lead, _make_lead(status=False), v_cruise=40.0)
    _run_update(mpc, primary_lead, parallel_adjacent, v_cruise=40.0)
    _run_update(mpc, primary_lead, parallel_adjacent, v_cruise=40.0)

    assert mpc.lead_role_debug["roles"]["lead1"] == "adjacent_awareness_left"
    assert mpc.lead_role_debug["toward_center_mps"]["lead1"] == pytest.approx(0.5)
    assert mpc.lead_role_debug["toward_center_model_mps"]["lead1"] == pytest.approx(0.5)
    assert mpc.lead_role_debug["toward_center_hist_mps"]["lead1"] == pytest.approx(0.0)
    assert mpc.adjacent_awareness_preview_debug["active"] is False
    assert mpc.adjacent_awareness_preview_debug["applied"] is False

    converging_adjacent = _make_lead(
      d_rel=79.0,
      y_rel=1.75,
      d_path=1.75,
      v_lat=-0.5,
      v_rel=-12.73,
      v_lead=22.0,
      model_prob=0.92,
      radar_track_id=-1002,
    )
    _run_update(mpc, primary_lead, converging_adjacent, v_cruise=40.0)

    assert mpc.adjacent_awareness_preview_debug["active"] is False
    assert mpc.lead_role_debug["toward_center_hist_confirm_frames"]["lead1"] == 1

    converging_adjacent_2 = _make_lead(
      d_rel=78.0,
      y_rel=1.70,
      d_path=1.70,
      v_lat=-0.5,
      v_rel=-12.73,
      v_lead=22.0,
      model_prob=0.92,
      radar_track_id=-1002,
    )
    _run_update(mpc, primary_lead, converging_adjacent_2, v_cruise=40.0)

    assert mpc.lead_role_debug["roles"]["lead1"] == "adjacent_awareness_left"
    assert mpc.lead_role_debug["toward_center_hist_mps"]["lead1"] == pytest.approx(0.5)
    assert mpc.adjacent_awareness_preview_debug["active"] is True
    assert mpc.adjacent_awareness_preview_debug["applied"] is True
    assert mpc.adjacent_awareness_preview_debug["toward_center_hist_mps"] == pytest.approx(0.5)
    assert mpc.adjacent_awareness_preview_debug["toward_center_model_mps"] == pytest.approx(0.5)
    assert mpc.adjacent_awareness_preview_debug["toward_center_hist_confirm_frames"] >= 2
    assert mpc.adjacent_awareness_preview_debug["history_identity_match"] is True

  def test_adjacent_awareness_preview_rejects_jitter_and_same_slot_replacement(self):
    mpc = _make_hyundai_mpc(v_ego=34.73, a_ego=0.0, time_fn=_MonotonicStub(step=0.1))
    primary_lead = _make_lead(
      d_rel=73.55, y_rel=0.0, d_path=0.0, v_lat=0.15,
      v_rel=-2.73, v_lead=32.0, model_prob=1.0, radar_track_id=-1001,
    )

    _run_update(mpc, primary_lead, _make_lead(status=False), v_cruise=40.0)
    _run_update(
      mpc, primary_lead,
      _make_lead(d_rel=80.0, y_rel=2.45, d_path=2.45, v_lat=0.0,
                 v_rel=-12.73, v_lead=22.0, model_prob=0.92, radar_track_id=-1002),
      v_cruise=40.0,
    )
    _run_update(
      mpc, primary_lead,
      _make_lead(d_rel=79.8, y_rel=2.41, d_path=2.41, v_lat=0.0,
                 v_rel=-12.73, v_lead=22.0, model_prob=0.92, radar_track_id=-1002),
      v_cruise=40.0,
    )

    assert mpc.lead_role_debug["history_identity_match"]["lead1"] is True
    assert mpc.lead_role_debug["toward_center_hist_mps"]["lead1"] == pytest.approx(0.4)
    assert mpc.lead_role_debug["toward_center_hist_confirm_frames"]["lead1"] == 1
    assert mpc.adjacent_awareness_preview_debug["active"] is False

    _run_update(
      mpc, primary_lead,
      _make_lead(d_rel=79.6, y_rel=2.45, d_path=2.45, v_lat=0.0,
                 v_rel=-12.73, v_lead=22.0, model_prob=0.92, radar_track_id=-1002),
      v_cruise=40.0,
    )
    _run_update(
      mpc, primary_lead,
      _make_lead(d_rel=79.4, y_rel=1.80, d_path=1.80, v_lat=0.0,
                 v_rel=-12.73, v_lead=22.0, model_prob=0.92, radar_track_id=-1003),
      v_cruise=40.0,
    )

    assert mpc.lead_role_debug["history_identity_match"]["lead1"] is False
    assert mpc.lead_role_debug["toward_center_hist_mps"]["lead1"] > 6.0
    assert mpc.lead_role_debug["toward_center_observed_mps"]["lead1"] == 0.0
    assert mpc.lead_role_debug["toward_center_hist_confirm_frames"]["lead1"] == 0
    assert mpc.adjacent_awareness_preview_debug["active"] is False
    assert mpc.adjacent_awareness_preview_debug["applied"] is False

  def test_low_speed_hostile_new_lead_still_gets_acquire_preview(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=0.83, a_ego=0.0)

    _run_update(mpc, _make_lead(status=False), _make_lead(status=False), v_cruise=40.0)
    _run_update(
      mpc,
      _make_lead(
        d_rel=9.56,
        y_rel=0.15,
        d_path=0.15,
        v_rel=-20.24,
        v_lead=-19.76,
        a_lead=-0.04,
        model_prob=0.20,
      ),
      _make_lead(status=False),
      v_cruise=40.0,
    )

    preview_debug = mpc.lead_approach_preview_debug["lead0"]

    assert preview_debug["acquire"]["active"] is True
    assert preview_debug["mode"] == "acquire"
    assert float(preview_debug["preview_buffer_m"]) > 1.0
    assert mpc.lead_approach_preview[0] > 1.0
    assert mpc.source == "lead0"

  def test_new_tight_gap_lead_gets_capped_acquire_preview_inside_headway(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=6.75, a_ego=0.0)

    _run_update(mpc, _make_lead(status=False), _make_lead(status=False), v_cruise=40.0)
    _run_update(
      mpc,
      _make_lead(
        d_rel=6.53,
        y_rel=-0.36,
        d_path=-0.36,
        v_rel=-6.82,
        v_lead=-0.07,
        a_lead=0.0,
        model_prob=0.20,
      ),
      _make_lead(status=False),
      v_cruise=40.0,
    )

    preview_debug = mpc.lead_approach_preview_debug["lead0"]

    assert preview_debug["acquire"]["active"] is True
    assert preview_debug["mode"] == "acquire"
    assert float(preview_debug["preview_buffer_m"]) > 0.5
    assert mpc.lead_approach_preview[0] > 0.5
    assert mpc.source == "lead0"

  def test_opening_noise_is_slew_clamped_without_forcing_reset(self):
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0, time_fn=_MonotonicStub(step=0.2))

    for _ in range(3):
      _run_update(
        mpc,
        _make_lead(d_rel=44.0, y_rel=0.03, d_path=0.03, v_lat=0.15, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.97),
        _make_lead(status=False),
      )

    prev_filtered_drel = float(mpc.hyundai_virtual_lead_debug["filtered"]["dRel"])
    _run_update(
      mpc,
      _make_lead(d_rel=52.5, y_rel=0.04, d_path=0.04, v_lat=0.20, v_rel=0.12, v_lead=29.12, a_lead=0.0, model_prob=0.97),
      _make_lead(status=False),
    )

    filtered_drel = float(mpc.hyundai_virtual_lead_debug["filtered"]["dRel"])
    filter_debug = mpc.hyundai_virtual_lead_debug["filter"]

    assert mpc.hyundai_virtual_lead_debug["reset_reason"] is None
    assert filter_debug["open_slew_clamped"] is True
    assert filter_debug["snap_to_raw"] is False
    assert filtered_drel > prev_filtered_drel
    assert filtered_drel < 45.0

  def test_real_closing_jump_still_snaps_filter_to_raw(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=29.0, a_ego=0.0)

    for _ in range(3):
      _run_update(
        mpc,
        _make_lead(d_rel=52.0, y_rel=0.03, d_path=0.03, v_lat=0.15, v_rel=0.0, v_lead=29.0, a_lead=0.0, model_prob=0.97),
        _make_lead(status=False),
      )

    _run_update(
      mpc,
      _make_lead(d_rel=25.0, y_rel=0.04, d_path=0.04, v_lat=0.18, v_rel=-6.0, v_lead=23.0, a_lead=-1.2, model_prob=0.98),
      _make_lead(status=False),
    )

    filter_debug = mpc.hyundai_virtual_lead_debug["filter"]

    assert filter_debug["snap_to_raw"] is True
    assert mpc.hyundai_virtual_lead_debug["filtered"]["dRel"] == pytest.approx(25.0)

  def test_duplicate_slot_jitter_does_not_start_cutin_settle(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc()

    for lead0, lead1 in (
      (
        _make_lead(d_rel=44.7, y_rel=0.05, d_path=0.05, v_lat=0.40, v_rel=-0.1, model_prob=0.96),
        _make_lead(d_rel=44.66, y_rel=0.08, d_path=0.08, v_lat=4.00, v_rel=-0.08, model_prob=0.92),
      ),
      (
        _make_lead(d_rel=44.8, y_rel=0.04, d_path=0.04, v_lat=0.50, v_rel=-0.1, model_prob=0.96),
        _make_lead(d_rel=44.74, y_rel=0.07, d_path=0.07, v_lat=4.10, v_rel=-0.08, model_prob=0.92),
      ),
      (
        _make_lead(d_rel=44.75, y_rel=0.06, d_path=0.06, v_lat=0.45, v_rel=-0.1, model_prob=0.95),
        _make_lead(d_rel=44.70, y_rel=0.09, d_path=0.09, v_lat=4.05, v_rel=-0.08, model_prob=0.92),
      ),
    ):
      _run_update(mpc, lead0, lead1)

    assert mpc.lead_role_debug["virtual_duplicate"]["active"] is True
    assert mpc._virtual_cutin_event_t is None

  def test_fcw_counter_ignores_cruise_owned_pullaway_lead(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=10.976, a_ego=0.726)

    for _ in range(10):
      _run_update(
        mpc,
        _make_lead(d_rel=25.854, y_rel=0.05, d_path=0.05, v_lat=0.10, v_rel=0.488, v_lead=11.392, a_lead=0.334, model_prob=0.99),
        _make_lead(d_rel=25.975, y_rel=0.08, d_path=0.08, v_lat=0.12, v_rel=0.498, v_lead=11.402, a_lead=0.325, model_prob=0.99),
        v_cruise=70.833336,
      )

    assert mpc.source == "cruise"
    assert mpc.crash_cnt == 0

  def test_fcw_counter_still_accumulates_for_active_closing_lead(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc(v_ego=12.0, a_ego=0.3)

    for _ in range(10):
      _run_update(
        mpc,
        _make_lead(d_rel=8.0, y_rel=0.05, d_path=0.05, v_lat=0.10, v_rel=-6.0, v_lead=4.0, a_lead=-2.0, model_prob=0.99),
        _make_lead(d_rel=8.1, y_rel=0.08, d_path=0.08, v_lat=0.12, v_rel=-5.98, v_lead=4.02, a_lead=-2.0, model_prob=0.99),
        v_cruise=35.0,
      )

    assert mpc.source == "lead0"
    assert mpc.crash_cnt > 2

  def test_lead_faster_than_cruise_does_not_exceed_set_speed(self, monkeypatch):
    # Regression: ego must not accelerate past v_cruise when following a lead
    # that is faster than the set speed. The Hyundai-stabilized lead path in
    # _select_acc_obstacle returns only the lead obstacle; without a cruise
    # ceiling clamp in the caller, the MPC plans velocities above v_cruise.
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    v_cruise = 33.5  # 75 mph
    mpc = _make_hyundai_mpc(v_ego=v_cruise, a_ego=0.0)

    for _ in range(20):
      _run_update(
        mpc,
        _make_lead(d_rel=45.0, y_rel=0.0, d_path=0.0, v_lat=0.0,
                   v_rel=2.5, v_lead=36.0, a_lead=0.3, model_prob=0.99),
        _make_lead(status=False),
        v_cruise=v_cruise,
      )

    v_solution_max = float(np.max(mpc.v_solution))
    a_solution_max = float(np.max(mpc.a_solution))
    assert v_solution_max <= v_cruise + 0.1, (
      f"MPC planned v={v_solution_max:.2f} m/s above v_cruise={v_cruise:.2f} m/s "
      f"(max accel={a_solution_max:.2f} m/s²)"
    )
