from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import ACCEL_MAX, LongitudinalMpc, N


def _make_lead(*, status=True, d_rel=44.0, y_rel=0.0, d_path=None, v_lat=0.0, v_rel=0.0,
               v_lead=29.0, a_lead=0.0, model_prob=0.95):
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
    radar=False,
    radarTrackId=-1,
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


def _make_hyundai_mpc(v_ego=29.0, a_ego=0.0):
  mpc = LongitudinalMpc(CP=SimpleNamespace(brand='hyundai'))
  mpc.mode = 'acc'
  mpc.set_cur_state(v_ego, a_ego)
  return mpc


def _run_update(mpc: LongitudinalMpc, lead0, lead1, *, v_cruise=40.0):
  radarstate = SimpleNamespace(leadOne=lead0, leadTwo=lead1)
  x = np.zeros(N + 1)
  v = np.zeros(N + 1)
  a = np.zeros(N + 1)
  j = np.zeros(N + 1)
  mpc.update(radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard)


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


class TestHyundaiAiLeadStability:
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
    assert max_reclaim_push > 0.5

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
    assert 0.55 < mpc.lead_present_cruise_accel_cap < 1.20
    assert mpc.last_cruise_response_model is not None
    assert mpc.last_cruise_response_model.max_accel_mps2 == pytest.approx(mpc.lead_present_cruise_accel_cap)

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
    assert mpc.gap_reclaim_obstacle_push > 1.0
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
    assert float(mpc._hyundai_reclaim_lead.vLead) < optimistic_reclaim_vlead - 0.15
    assert float(mpc._hyundai_reclaim_lead.vLead) <= float(mpc._hyundai_virtual_lead.vLead) + 0.1

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
