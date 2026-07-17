import dataclasses
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from cereal import car, custom, log
from opendbc.car.hyundai.values import CAR, HyundaiFlags
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.radard import (
  CLOSING_GOVERNOR_RECOVERY_MIN_RAW_TTC_S,
  CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS,
  CLOSING_GOVERNOR_STALE_DECAY_MPS2,
  KalmanParams,
  ModelLeadTrack,
  ModelLeadTracker,
  OPENING_GOVERNOR_HARD_CLOSING_MPS,
  RADAR_TO_CAMERA,
  RadarD,
  Track,
  get_lead,
)


class _NoParams:
  def get(self, _key):
    return None


def _model_path():
  return SimpleNamespace(position=SimpleNamespace(
    x=np.linspace(0.0, 160.0, 33),
    y=np.zeros(33),
  ))


def _model_lead(*, d_rel, v_ego=29.0, v_lead=29.0, prob=0.96, y_rel=0.0, v_lat=0.0, a_lead=0.0):
  x0 = float(d_rel) + RADAR_TO_CAMERA
  y0 = -float(y_rel)
  return SimpleNamespace(
    t=[0.0, 1.0],
    x=[x0, x0 + float(v_lead)],
    y=[y0, y0 - float(v_lat)],
    v=[float(v_lead), float(v_lead)],
    a=[float(a_lead), float(a_lead)],
    xStd=[1.0, 1.0],
    yStd=[0.35, 0.35],
    vStd=[1.0, 1.0],
    prob=float(prob),
    probTime=0.0,
  )


def _cp():
  return SimpleNamespace(brand="hyundai", flags=0)


def _camera_scc_cp():
  return SimpleNamespace(brand="hyundai", flags=int(HyundaiFlags.CANFD_CAMERA_SCC))


def _cp_sp():
  return SimpleNamespace(flags=0)


def _tracked_lead(tracker, *, now, d_rel, v_ego=29.0, v_lead=29.0, slot=0, y_rel=0.0, v_lat=0.0):
  tracker.begin_frame(now)
  lead = get_lead(
    v_ego,
    True,
    {},
    _model_lead(d_rel=d_rel, v_ego=v_ego, v_lead=v_lead, y_rel=y_rel, v_lat=v_lat),
    v_ego,
    _cp(),
    _cp_sp(),
    _model_path(),
    low_speed_override=False,
    model_lead_tracker=tracker,
    lead_slot=slot,
    now=now,
  )
  tracker.end_frame()
  return lead


def _rolling_ranges(values, window):
  return [max(values[max(0, i - window + 1):i + 1]) - min(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


class TestRadardModelLeadFilter:
  def test_model_only_drel_noise_is_filtered_before_radarstate(self):
    tracker = ModelLeadTracker(params=_NoParams())
    rng = random.Random(9)
    raw_values = []
    tracked_values = []
    track_ids = []

    for frame in range(180):
      now = frame * 0.05
      alternating = 5.5 if frame % 2 else -5.5
      noise = alternating + 1.3 * math.sin(frame * 0.31) + rng.gauss(0.0, 0.25)
      raw = 42.0 + noise
      lead = _tracked_lead(tracker, now=now, d_rel=raw)
      raw_values.append(raw)
      tracked_values.append(float(lead["dRel"]))
      track_ids.append(int(lead["radarTrackId"]))

    warm_raw = raw_values[30:]
    warm_tracked = tracked_values[30:]
    raw_range = max(warm_raw) - min(warm_raw)
    tracked_range = max(warm_tracked) - min(warm_tracked)
    tracked_roll3s_p95 = float(np.percentile(_rolling_ranges(warm_tracked, 60), 95))

    assert raw_range > 11.0
    assert tracked_range < raw_range * 0.35
    assert tracked_roll3s_p95 < 3.0
    assert len(set(track_ids[30:])) == 1
    assert track_ids[-1] <= -1001

  def test_opening_jump_is_slew_limited_without_model_velocity_support(self):
    tracker = ModelLeadTracker(params=_NoParams())
    for frame in range(20):
      lead = _tracked_lead(tracker, now=frame * 0.05, d_rel=42.0)
    before = float(lead["dRel"])

    opened = _tracked_lead(tracker, now=1.05, d_rel=62.0, v_lead=29.0)

    assert float(opened["dRel"]) - before < 0.25
    assert int(opened["radarTrackId"]) <= -1001

  def test_low_ttc_closer_lead_is_adopted_quickly(self):
    tracker = ModelLeadTracker(params=_NoParams())
    for frame in range(20):
      _tracked_lead(tracker, now=frame * 0.05, d_rel=42.0, v_lead=29.0)

    adopted = None
    for frame in range(20, 23):
      adopted = _tracked_lead(tracker, now=frame * 0.05, d_rel=18.0, v_lead=20.0)

    assert adopted is not None
    assert float(adopted["dRel"]) < 22.5
    assert float(adopted["vRel"]) < -3.0

  def test_same_speed_distance_only_close_dip_is_not_fast_adopted(self):
    tracker = ModelLeadTracker(params=_NoParams())
    for frame in range(20):
      _tracked_lead(tracker, now=frame * 0.05, d_rel=42.0, v_lead=29.0)

    dipped = None
    for frame in range(20, 24):
      dipped = _tracked_lead(tracker, now=frame * 0.05, d_rel=14.0, v_lead=29.0)

    assert dipped is not None
    assert float(dipped["dRel"]) > 41.5
    assert int(dipped["radarTrackId"]) <= -1001

  def test_synthetic_model_track_id_survives_slot_reorder_and_duplicate_hypotheses(self):
    tracker = ModelLeadTracker(params=_NoParams())
    model_msg = _model_path()
    tracker.begin_frame(0.0)
    lead0 = get_lead(29.0, True, {}, _model_lead(d_rel=42.0), 29.0, _cp(), _cp_sp(), model_msg,
                     low_speed_override=False, model_lead_tracker=tracker, lead_slot=0, now=0.0)
    lead1 = get_lead(29.0, True, {}, _model_lead(d_rel=42.4, y_rel=0.04), 29.0, _cp(), _cp_sp(), model_msg,
                     low_speed_override=False, model_lead_tracker=tracker, lead_slot=1, now=0.0)
    tracker.end_frame()

    track_id = int(lead0["radarTrackId"])
    assert track_id <= -1001
    assert int(lead1["radarTrackId"]) == track_id

    slot1_only = _tracked_lead(tracker, now=0.05, d_rel=42.2, slot=1, y_rel=0.04)

    assert int(slot1_only["radarTrackId"]) == track_id

  def test_same_frame_duplicate_hypothesis_collapses_even_with_large_drel_noise(self):
    tracker = ModelLeadTracker(params=_NoParams())
    model_msg = _model_path()

    tracker.begin_frame(0.0)
    lead0 = get_lead(29.0, True, {}, _model_lead(d_rel=42.0), 29.0, _cp(), _cp_sp(), model_msg,
                     low_speed_override=False, model_lead_tracker=tracker, lead_slot=0, now=0.0)
    lead1 = get_lead(29.0, True, {}, _model_lead(d_rel=61.0, y_rel=0.05), 29.0, _cp(), _cp_sp(), model_msg,
                     low_speed_override=False, model_lead_tracker=tracker, lead_slot=1, now=0.0)
    tracker.end_frame()

    assert int(lead1["radarTrackId"]) == int(lead0["radarTrackId"])
    assert float(lead1["dRel"]) == pytest.approx(float(lead0["dRel"]))

  def test_real_radar_track_still_copies_raw_drel(self):
    track = Track(12, 29.0, KalmanParams(0.05))

    track.update(12.0, 0.0, 0.0, 29.0, True)
    track.update(30.0, 0.0, 0.0, 29.0, True)

    assert track.get_RadarState()["dRel"] == pytest.approx(30.0)

  def test_latched_close_closing_track_survives_model_prob_dropout(self):
    v_ego = 30.0 * 0.44704
    track = Track(42, v_ego, KalmanParams(0.05))
    track.update(17.4, 0.0, -0.8, v_ego - 0.8, True)

    lead = get_lead(
      v_ego,
      True,
      {42: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego - 0.8, prob=0.10),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert lead["status"]
    assert lead["radar"]
    assert lead["radarTrackId"] == 42
    assert lead["dRel"] == pytest.approx(17.4)
    assert lead["vRel"] == pytest.approx(-0.8)

  def test_latched_close_steady_track_survives_model_prob_dropout(self):
    v_ego = 30.0 * 0.44704
    track = Track(48, v_ego, KalmanParams(0.05))
    track.update(17.4, 0.0, 0.0, v_ego, True)

    lead = get_lead(
      v_ego,
      True,
      {48: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego, prob=0.10),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert lead["status"]
    assert lead["radarTrackId"] == 48
    assert lead["dRel"] == pytest.approx(17.4)
    assert lead["vRel"] == pytest.approx(0.0)

  def test_latched_close_fast_pulling_away_track_drops_on_model_prob_dropout(self):
    v_ego = 30.0 * 0.44704
    track = Track(49, v_ego, KalmanParams(0.05))
    track.update(17.4, 0.0, 1.0, v_ego + 1.0, True)

    lead = get_lead(
      v_ego,
      True,
      {49: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego + 1.0, prob=0.10),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert not lead["status"]

  def test_unlatched_track_does_not_acquire_on_low_model_prob(self):
    v_ego = 30.0 * 0.44704
    track = Track(45, v_ego, KalmanParams(0.05))
    track.update(17.4, 0.0, -0.8, v_ego - 0.8, True)

    lead = get_lead(
      v_ego,
      True,
      {45: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego - 0.8, prob=0.10),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=False,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert not lead["status"]

  def test_off_path_track_still_drops_when_model_prob_drops(self):
    v_ego = 30.0 * 0.44704
    track = Track(43, v_ego, KalmanParams(0.05))
    track.update(17.4, 2.2, -0.8, v_ego - 0.8, True)

    lead = get_lead(
      v_ego,
      True,
      {43: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego - 0.8, prob=0.10, y_rel=2.2),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert not lead["status"]

  def test_far_nonurgent_track_still_drops_when_model_prob_drops(self):
    v_ego = 30.0 * 0.44704
    track = Track(44, v_ego, KalmanParams(0.05))
    track.update(75.0, 0.0, -0.6, v_ego - 0.6, True)

    lead = get_lead(
      v_ego,
      True,
      {44: track},
      _model_lead(d_rel=75.0, v_ego=v_ego, v_lead=v_ego - 0.6, prob=0.10),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert not lead["status"]

  def test_camera_scc_track_without_lateral_can_bridge_model_prob_dropout(self):
    v_ego = 30.0 * 0.44704
    track = Track(46, v_ego, KalmanParams(0.05))
    track.update(17.4, math.nan, -0.8, v_ego - 0.8, True)

    lead = get_lead(
      v_ego,
      True,
      {46: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego - 0.8, prob=0.10),
      v_ego,
      _camera_scc_cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert lead["status"]
    assert lead["radarTrackId"] == 46
    assert math.isfinite(lead["yRel"])

  def test_non_scc_track_without_lateral_still_drops_on_model_prob_dropout(self):
    v_ego = 30.0 * 0.44704
    track = Track(47, v_ego, KalmanParams(0.05))
    track.update(17.4, math.nan, -0.8, v_ego - 0.8, True)

    lead = get_lead(
      v_ego,
      True,
      {47: track},
      _model_lead(d_rel=17.4, v_ego=v_ego, v_lead=v_ego - 0.8, prob=0.10),
      v_ego,
      _cp(),
      _cp_sp(),
      _model_path(),
      low_speed_override=True,
      prev_latched=True,
      prob_enter=0.60,
      prob_exit=0.25,
    )

    assert not lead["status"]


DT_FRAME_S = 0.05


def _cfg(**overrides) -> LeadResponseTuningConfig:
  cfg = LeadResponseTuningConfig()
  return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _lead_dict(d_rel, v_rel, v_ego):
  return {
    "dRel": float(d_rel),
    "yRel": 0.0,
    "vRel": float(v_rel),
    "vLead": float(v_ego + v_rel),
    "vLeadK": float(v_ego + v_rel),
    "aLeadK": 0.0,
    "aLeadTau": 0.3,
    "modelProb": 0.95,
    "dPath": 0.0,
    "vLat": 0.0,
  }


def _approach_track(cfg, *, v_ego=6.0, d0=30.0, frames=20):
  """Settled track approaching a stopped lead: raw dRel tracks truth each frame."""
  v_rel = -float(v_ego)
  track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d0, v_rel, v_ego), 0.0, 0)
  t, d = 0.0, float(d0)
  for _ in range(frames):
    t += DT_FRAME_S
    d = max(0.0, d + v_rel * DT_FRAME_S)
    track.update(_lead_dict(d, v_rel, v_ego), t, v_ego, cfg, 0)
  return track, t, d


class TestModelLeadFastCloseCorroborationAndOpenRecovery:
  """Pins the M2 fix (corroboration-before-adoption + corroborated opening
  recovery) with green tests: the strict-xfail phantom repro stays xfail on an
  unrelated stopping-chain conjunct, so it cannot catch an M2 regression."""

  def test_isolated_inward_outlier_is_not_fast_adopted(self):
    # The exact phantom mechanism: one heavy-tailed inward dRel outlier during
    # a > 2.5 m/s approach (strong_closing always true) must NOT collapse the
    # track in a single 50 ms frame.
    cfg = _cfg()
    v_ego = 6.0
    track, t, d = _approach_track(cfg, v_ego=v_ego)
    pre_drel = track.dRel

    t += DT_FRAME_S
    d -= v_ego * DT_FRAME_S
    outlier = d - 10.0
    track.update(_lead_dict(outlier, -v_ego, v_ego), t, v_ego, cfg, 0)

    # Partial pessimistic adoption only (urgency blend at the boosted close
    # slew, ~0.5 m/frame) -- nowhere near the raw outlier.
    assert track.dRel > outlier + 8.0
    assert pre_drel - track.dRel < 1.2

    # Truth resumes: the track re-converges instead of ratcheting.
    for _ in range(5):
      t += DT_FRAME_S
      d -= v_ego * DT_FRAME_S
      track.update(_lead_dict(d, -v_ego, v_ego), t, v_ego, cfg, 0)
    assert abs(track.dRel - d) < 1.0

  def test_two_consecutive_qualifying_frames_are_fast_adopted(self):
    # Latency bound for genuine same-track close threats beyond the gate: the
    # second consecutive qualifying frame reaches the fast path (alpha >= 0.65)
    # and the track converges onto the raw measurement within 3 frames.
    cfg = _cfg()
    v_ego = 6.0
    track, t, d = _approach_track(cfg, v_ego=v_ego)

    drels = []
    raw = None
    for _ in range(3):
      t += DT_FRAME_S
      d -= v_ego * DT_FRAME_S
      raw = d - 10.0
      track.update(_lead_dict(raw, -v_ego, v_ego), t, v_ego, cfg, 0)
      drels.append(track.dRel)

    # Frame 1 is partial (slew-clamped), frame 2 is the fast adoption jump.
    assert drels[0] - drels[1] > 5.0
    assert abs(drels[2] - raw) < 1.5

  def test_legacy_kill_switch_confirm_frames_one_restores_single_frame_adoption(self):
    # ConfirmFrames=1 must restore the legacy instant fast-close adoption.
    cfg = _cfg(model_lead_filter_fast_close_confirm_frames=1.0)
    v_ego = 6.0
    track, t, d = _approach_track(cfg, v_ego=v_ego)

    t += DT_FRAME_S
    d -= v_ego * DT_FRAME_S
    outlier = d - 10.0
    track.update(_lead_dict(outlier, -v_ego, v_ego), t, v_ego, cfg, 0)

    assert track.dRel < outlier + 4.0  # alpha >= 0.65 on ~-10 m innovation

  def _collapse_then_truth(self, cfg, *, v_ego, truth_frames, collapse_m=12.0):
    """Fast-collapse a settled track, then resume truthful measurements.

    Returns (track, per-frame (raw, dRel, step_above_prediction) list, final raw d)."""
    v_rel = -6.0  # closing 6 m/s in every variant (lead speed = v_ego - 6)
    track, t, d = _approach_track(cfg, v_ego=v_ego, d0=40.0)
    for _ in range(3):
      t += DT_FRAME_S
      d += v_rel * DT_FRAME_S
      track.update(_lead_dict(d - collapse_m, v_rel, v_ego), t, v_ego, cfg, 0)
    assert track.dRel < d - collapse_m + 4.0  # state is wrong-too-close

    rows = []
    for _ in range(truth_frames):
      t += DT_FRAME_S
      d += v_rel * DT_FRAME_S
      predicted = track.predict_drel(t)
      track.update(_lead_dict(d, v_rel, v_ego), t, v_ego, cfg, 0)
      rows.append((d, track.dRel, track.dRel - predicted))
    return track, rows, d

  def test_corroborated_opening_recovery_heals_wrong_too_close_state(self):
    cfg = _cfg()
    open_slew_step_m = float(cfg.model_lead_filter_open_slew_max_mps) * DT_FRAME_S
    track, rows, d = self._collapse_then_truth(cfg, v_ego=6.0, truth_frames=30)

    # Before OpenRecoveryConfirmFrames (4) consecutive beyond-gate opening
    # frames, healing is still opening-slew limited.
    for _raw, _drel, step in rows[:3]:
      assert step <= open_slew_step_m + 1e-6
    # Once armed, the recovery bypasses the slew cap toward the measurement...
    assert rows[3][2] > open_slew_step_m * 5.0
    # ...heals within ~tau (0.5 s): < 2.5 m error well inside 1.5 s...
    assert d - track.dRel < 2.5
    # ...and never steps PAST the raw measurement on any frame.
    for raw, drel, _ in rows:
      assert drel <= raw + 1e-6

  def test_high_speed_opening_recovery_stays_blocked_for_short_raw_ttc(self):
    cfg = _cfg()
    open_slew_step_m = float(cfg.model_lead_filter_open_slew_max_mps) * DT_FRAME_S
    # Above OpenRecoveryMaxEgoMps, repeated farther measurements are not enough
    # on their own: a short raw TTC keeps the pessimistic state and slew limit.
    track, rows, d = self._collapse_then_truth(cfg, v_ego=12.0, truth_frames=30, collapse_m=22.0)

    for _raw, _drel, step in rows:
      assert step <= open_slew_step_m + 1e-6
    assert d - track.dRel > 8.0  # still wrong-too-close: slew-only recovery

  def test_high_speed_opening_recovery_heals_safe_persistent_inward_collapse(self):
    cfg = _cfg()
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(45.0, -2.0, 24.0), 0.0, 0)
    rows = []
    t = 0.0
    for _ in range(35):
      t += DT_FRAME_S
      predicted = track.predict_drel(t)
      track.update(_lead_dict(60.0, -2.0, 24.0), t, 24.0, cfg, 0)
      rows.append((60.0, track.dRel, track.dRel - predicted))

    # Raw truth remains far away with a long TTC and no lead braking, so after
    # four corroborating frames the freeway-speed state heals just like the
    # low-speed path instead of retaining the false 12 m deficit indefinitely.
    assert rows[3][2] > float(cfg.model_lead_filter_open_slew_max_mps) * DT_FRAME_S * 5.0
    assert 60.0 - track.dRel < 2.5

  def test_high_speed_opening_recovery_stays_blocked_for_braking_lead(self):
    cfg = _cfg()
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(40.0, -2.0, 24.0), 0.0, 0)
    t = 0.0
    for _ in range(3):
      t += DT_FRAME_S
      track.update(_lead_dict(28.0, -2.0, 24.0), t, 24.0, cfg, 0)
    for _ in range(12):
      t += DT_FRAME_S
      predicted = track.predict_drel(t)
      lead = _lead_dict(40.0, -2.0, 24.0)
      lead["aLeadK"] = -0.5
      track.update(lead, t, 24.0, cfg, 0)
      assert track.dRel - predicted <= float(cfg.model_lead_filter_open_slew_max_mps) * DT_FRAME_S + 1e-6

  def test_open_recovery_max_ego_zero_is_exact_disable_even_at_standstill(self):
    # Kill switch: OpenRecoveryMaxEgoMps=0 must not arm at v_ego exactly 0.0
    # (0.0 <= 0.0 would otherwise hold), so A/B isolation is bit-exact legacy.
    for max_ego, expect_recovery in ((0.0, False), (8.0, True)):
      cfg = _cfg(model_lead_filter_open_recovery_max_ego_mps=max_ego)
      open_slew_step_m = float(cfg.model_lead_filter_open_slew_max_mps) * DT_FRAME_S
      track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(5.0, 0.0, 0.0), 0.0, 0)
      t = 0.0
      for _ in range(10):
        t += DT_FRAME_S
        predicted = track.predict_drel(t)
        track.update(_lead_dict(17.0, 0.0, 0.0), t, 0.0, cfg, 0)
        step = track.dRel - predicted
        if not expect_recovery:
          assert step <= open_slew_step_m + 1e-6
      if expect_recovery:
        assert track.dRel > 8.0
      else:
        assert track.dRel <= 5.0 + 10 * open_slew_step_m + 1e-6


def _cd4_lead(d_rel, *, y_rel=0.0, d_path=0.0, v_rel=0.0, v_ego=20.0, prob=0.97):
  return {
    "dRel": float(d_rel),
    "yRel": float(y_rel),
    "vRel": float(v_rel),
    "vLead": float(v_ego + v_rel),
    "vLeadK": float(v_ego + v_rel),
    "aLeadK": 0.0,
    "aLeadTau": 0.3,
    "modelProb": float(prob),
    "dPath": float(d_path),
    "vLat": 0.0,
    "status": True,
  }


class TestModelLeadCD4AssociationAndStepGuard:
  """CD4: association keys lateral continuity on the path-relative dPath (widened
  raw-yRel tolerance), and an opening-only published-dRel step guard. Proves the
  fix holds continuity through a raw-yRel excursion, does NOT mask a
  closer/closing adjacent lead, and never clamps a closing (nearer) step."""

  def test_raw_yrel_excursion_holds_track_id_when_dpath_stays_in_lane(self):
    # One physical lead through a curve: raw yRel drifts well past the legacy
    # 3.0 m gate while dPath stays in-lane. The track id must stay constant.
    tracker = ModelLeadTracker(params=_NoParams())
    v_ego = 20.0
    ids = []
    for frame in range(40):
      now = frame * DT_FRAME_S
      # yRel ramps -1.5 -> -8.0 m; dPath stays inside +/-0.6 m the whole time.
      y_rel = -1.5 - 6.5 * (frame / 39.0)
      d_path = 0.4 * math.sin(frame * 0.3)
      tracker.begin_frame(now)
      out = tracker.update_from_vision(
        _cd4_lead(45.0, y_rel=y_rel, d_path=d_path, v_ego=v_ego),
        now=now, v_ego=v_ego, lead_slot=0,
      )
      tracker.end_frame()
      ids.append(int(out["radarTrackId"]))
    # Legacy would have churned the id when yRel crossed the 3.0 m gate.
    assert len(set(ids)) == 1
    assert ids[-1] <= -1001

  def test_closing_adjacent_lead_in_widened_band_stays_separate(self):
    # Amendment 1: a genuinely CLOSER + CLOSING lead whose raw yRel sits in the
    # widened band (3 m < |y| < 7 m) but at a real path offset must NOT be
    # absorbed into the established far track. It must spawn its own track so a
    # slow-closing near threat is never masked into a farther one.
    tracker = ModelLeadTracker(params=_NoParams())
    v_ego = 20.0
    far_id = None
    for frame in range(20):
      now = frame * DT_FRAME_S
      tracker.begin_frame(now)
      far = tracker.update_from_vision(
        _cd4_lead(60.0, y_rel=0.0, d_path=0.0, v_rel=0.0, v_ego=v_ego),
        now=now, v_ego=v_ego, lead_slot=0,
      )
      tracker.end_frame()
      far_id = int(far["radarTrackId"])

    # New lead in slot 1: 30 m (much closer than the 60 m track), closing 5 m/s,
    # raw yRel 5 m AND dPath 4 m (a real different-lane offset).
    now = 20 * DT_FRAME_S
    tracker.begin_frame(now)
    tracker.update_from_vision(
      _cd4_lead(60.0, y_rel=0.0, d_path=0.0, v_rel=0.0, v_ego=v_ego),
      now=now, v_ego=v_ego, lead_slot=0,
    )
    near = tracker.update_from_vision(
      _cd4_lead(30.0, y_rel=5.0, d_path=4.0, v_rel=-5.0, v_ego=v_ego),
      now=now, v_ego=v_ego, lead_slot=1,
    )
    tracker.end_frame()
    assert int(near["radarTrackId"]) != far_id
    assert float(near["dRel"]) < 33.0  # publishes its true close dRel, not the far one

  def test_step_guard_first_publish_is_unclamped(self):
    # A fresh cut-in must publish its true close dRel on frame 1 with zero
    # attenuation: the guard never binds without a prior published value.
    cfg = _cfg()  # defaults: guard enabled
    track = ModelLeadTrack.from_lead_dict(-1001, _cd4_lead(18.0, v_rel=-6.0, v_ego=20.0), 0.0, 0)
    published = track.get_RadarState(cfg)
    assert float(published["dRel"]) == pytest.approx(18.0, abs=1e-6)

  def test_step_guard_clamps_opening_step_only(self):
    # Establish a settled published value, then jump the internal dRel FARTHER by
    # far more than max(3, 0.1*dRel) in one frame. The published step must be
    # clamped to the opening bound.
    cfg = _cfg()
    v_ego = 20.0
    track = ModelLeadTrack.from_lead_dict(-1001, _cd4_lead(40.0, v_rel=0.0, v_ego=v_ego), 0.0, 0)
    # Prime last_published_drel via a publish at ~40 m.
    track.get_RadarState(cfg)
    prev = track.last_published_drel
    assert prev is not None
    # Force the internal filtered dRel to a far jump (simulate a fabricated step)
    # and advance the frame key so the guard evaluates this as a new frame.
    track.dRel = 70.0
    track.vRel = 0.0
    track.last_t = 0.05
    out = track.get_RadarState(cfg)
    # Guard bound is computed on the pre-clamp published value (70 m, no lag comp
    # since vRel=0): max(3, 0.1*70) = 7, so the publish is held to prev + 7.
    bound = max(cfg.model_lead_step_guard_abs_m, cfg.model_lead_step_guard_frac * 70.0)
    assert float(out["dRel"]) == pytest.approx(float(prev) + bound, abs=1e-6)
    assert float(out["dRel"]) < 70.0  # actually clamped

  def test_step_guard_never_clamps_closing_step(self):
    # A step that moves the lead CLOSER (the only direction that can trigger
    # braking) is NEVER clamped, so emergency braking is never delayed.
    cfg = _cfg()
    v_ego = 20.0
    track = ModelLeadTrack.from_lead_dict(-1001, _cd4_lead(60.0, v_rel=0.0, v_ego=v_ego), 0.0, 0)
    track.get_RadarState(cfg)
    # Jump the filtered dRel much CLOSER in one frame.
    track.dRel = 20.0
    track.last_t = 0.05
    out = track.get_RadarState(cfg)
    assert float(out["dRel"]) == pytest.approx(20.0, abs=1e-6)

  def test_step_guard_disable_sentinel_restores_legacy_publish(self):
    # Either knob at 0 disables the guard: a far opening step publishes unclamped.
    for cfg in (_cfg(model_lead_step_guard_abs_m=0.0), _cfg(model_lead_step_guard_frac=0.0)):
      track = ModelLeadTrack.from_lead_dict(-1001, _cd4_lead(40.0, v_rel=0.0, v_ego=20.0), 0.0, 0)
      track.get_RadarState(cfg)
      track.dRel = 70.0
      track.last_t = 0.05
      out = track.get_RadarState(cfg)
      assert float(out["dRel"]) == pytest.approx(70.0, abs=1e-6)

  def test_step_guard_idempotent_within_frame_no_double_clamp(self):
    # A duplicate slot calls get_RadarState twice in one frame. The second call
    # must reuse the first clamped output, not re-clamp against the just-stored
    # value (which would ratchet a genuine opening continuation shorter).
    cfg = _cfg()
    track = ModelLeadTrack.from_lead_dict(-1001, _cd4_lead(40.0, v_rel=0.0, v_ego=20.0), 0.0, 0)
    track.get_RadarState(cfg)
    track.dRel = 70.0
    track.last_t = 0.05
    first = float(track.get_RadarState(cfg)["dRel"])
    second = float(track.get_RadarState(cfg)["dRel"])
    assert second == pytest.approx(first, abs=1e-9)

  def test_widened_raw_y_tol_absorbs_jump_that_legacy_gate_rejects(self):
    # A single-frame raw-yRel jump of ~5 m (dPath in-lane, opening lead so the
    # closing-side clamp does not apply) is ABSORBED into the same track under
    # the widened 7 m tolerance, but REJECTED (new track) once both gate knobs are
    # rolled back to the legacy shared 3.0 m -- proving the rollback sentinel and
    # that the fix is what admits the continuity.
    def run(cfg):
      tracker = ModelLeadTracker(params=_NoParams())
      tracker._refresh_config = lambda now: None
      tracker._cfg = cfg
      v_ego = 20.0
      # Settle a track at yRel ~ -1.0 (opening lead: pull-away, vRel +0.2).
      for frame in range(15):
        now = frame * DT_FRAME_S
        tracker.begin_frame(now)
        out = tracker.update_from_vision(
          _cd4_lead(45.0, y_rel=-1.0, d_path=0.2, v_rel=0.2, v_ego=v_ego),
          now=now, v_ego=v_ego, lead_slot=0,
        )
        tracker.end_frame()
      settled_id = int(out["radarTrackId"])
      # Abrupt raw yRel jump to -6.0 in one frame (y_err ~5 m > 3.0, < 7.0);
      # dPath stays in-lane. Opening (vRel > 0) so raw-y widening applies.
      now = 15 * DT_FRAME_S
      tracker.begin_frame(now)
      jumped = tracker.update_from_vision(
        _cd4_lead(45.0, y_rel=-6.0, d_path=0.3, v_rel=0.2, v_ego=v_ego),
        now=now, v_ego=v_ego, lead_slot=0,
      )
      tracker.end_frame()
      return settled_id, int(jumped["radarTrackId"])

    settled_fix, jumped_fix = run(_cfg())  # widened default
    assert jumped_fix == settled_fix  # absorbed, continuity held

    settled_legacy, jumped_legacy = run(
      _cfg(model_lead_assoc_dpath_gate_m=3.0, model_lead_assoc_y_raw_tol_m=3.0)
    )
    assert jumped_legacy != settled_legacy  # legacy 3.0 m gate spawns a new id


class TestClosingGovernorCalmRecovery:
  """Fail-closed contract for reconciling a stale CD9 closing hold."""

  NOW = 10.0
  DREL_M = 100.0

  @staticmethod
  def _cfg(**overrides):
    base = {
      "closing_governor_margin_mps": 0.75,
      # Keep the ordinary decel arm from refreshing the deadline in veto
      # tests; recovery's fixed braking veto remains independently active.
      "closing_governor_accel_onset_mps2": 99.0,
    }
    base.update(overrides)
    return dataclasses.replace(LeadResponseTuningConfig.defaults(), **base)

  @classmethod
  def _track(cls, *, active=True, closing_mps=1.8, drel_m=None, threat_corroborated=True):
    drel = cls.DREL_M if drel_m is None else float(drel_m)
    tr = ModelLeadTrack.from_lead_dict(
      7,
      {
        "dRel": drel,
        "yRel": 0.0,
        "vRel": -2.0,
        "vLead": 18.0,
        "aLeadK": 0.0,
        "modelProb": 0.9,
      },
      cls.NOW - 1.3,
      0,
    )
    tr.vRel = -2.0
    tr.vLead = 18.0
    tr.vLeadK = 18.0
    tr.dRel = drel
    tr.governor_hold_until_t = cls.NOW + 1.0 if active else cls.NOW - 0.01
    tr.governor_closing_mps = float(closing_mps)
    tr.governor_threat_corroborated = bool(threat_corroborated)
    tr.governor_reason = "latched"
    return tr

  @classmethod
  def _seed_histories(
    cls,
    tr,
    *,
    now=None,
    drel_m=None,
    position_closing_mps=0.6,
    raw_closing_mps=0.4,
    recent_gap_s=0.05,
    previous_raw_closing_mps=None,
    window_alead_mps2=0.0,
    previous_alead_mps2=None,
    position_times=None,
  ):
    now = cls.NOW if now is None else float(now)
    drel = cls.DREL_M if drel_m is None else float(drel_m)
    tr.closing_evidence.clear()
    tr.opening_position_evidence.clear()

    if position_times is None:
      position_times = [
        now - 2.0 + 0.05 * i
        for i in range(41)
        if now - 2.0 + 0.05 * i <= now - recent_gap_s + 1e-9
      ]
    for sample_t in position_times:
      sample_drel = drel + float(position_closing_mps) * (now - float(sample_t))
      tr.opening_position_evidence.append((float(sample_t), sample_drel))

    closing_times = [
      now - 0.60 + 0.05 * i
      for i in range(13)
      if now - 0.60 + 0.05 * i <= now - recent_gap_s + 1e-9
    ]
    for idx, sample_t in enumerate(closing_times):
      is_last = idx == len(closing_times) - 1
      sample_closing = (
        float(previous_raw_closing_mps)
        if is_last and previous_raw_closing_mps is not None
        else float(raw_closing_mps)
      )
      sample_alead = (
        float(previous_alead_mps2)
        if is_last and previous_alead_mps2 is not None
        else float(window_alead_mps2)
      )
      sample_drel = drel + float(position_closing_mps) * (now - float(sample_t))
      tr.closing_evidence.append((float(sample_t), sample_drel, -sample_closing, sample_alead))

  @staticmethod
  def _update(tr, *, now, drel_m, raw_closing_mps, raw_alead_mps2=0.0,
              raw_prob=1.0, raw_dpath=0.0, raw_vlat=0.0, cfg=None,
              allow_calm_recovery=True):
    return tr._update_closing_governor(
      float(now),
      raw_drel=float(drel_m),
      raw_vrel=-float(raw_closing_mps),
      raw_alead=float(raw_alead_mps2),
      cfg=cfg if cfg is not None else TestClosingGovernorCalmRecovery._cfg(),
      raw_prob=float(raw_prob),
      raw_dpath=float(raw_dpath),
      raw_vlat=float(raw_vlat),
      allow_calm_recovery=bool(allow_calm_recovery),
    )

  @staticmethod
  def _production_radard(*, brand, fingerprint, radar_unavailable):
    cp = car.CarParams.new_message()
    cp.brand = str(brand)
    cp.carFingerprint = str(fingerprint)
    cp.radarUnavailable = bool(radar_unavailable)
    return RadarD(cp, custom.CarParamsSP.new_message())

  @staticmethod
  def _lead_wire_bytes(lead):
    state = log.RadarState.new_message()
    state.leadOne = lead
    return state.to_bytes()

  def _update_through_tracker(self, tracker, tr):
    # Exercise the same ModelLeadTracker -> ModelLeadTrack propagation RadarD
    # uses in production while keeping this regression independent of device
    # Params. A direct tracker constructor remains the explicit reference seam.
    tr.last_t = self.NOW - 0.05
    tracker._tracks[tr.identifier] = tr
    tracker._cfg = self._cfg()
    tracker._last_param_refresh_t = self.NOW
    tracker.begin_frame(self.NOW)
    lead = tracker.update_from_vision(
      {
        "dRel": self.DREL_M,
        "yRel": 0.0,
        "vRel": -0.4,
        "vLead": 19.6,
        "vLeadK": 19.6,
        "aLeadK": 0.0,
        "aLeadTau": 0.3,
        "modelProb": 0.9,
        "dPath": 0.0,
        "vLat": 0.0,
      },
      now=self.NOW,
      v_ego=20.0,
      lead_slot=0,
    )
    tracker.end_frame()
    return lead

  def test_non_hyundai_production_radard_keeps_recovery_disabled_and_byte_exact(self):
    rd = self._production_radard(
      brand="toyota",
      fingerprint="TOYOTA_RAV4",
      radar_unavailable=True,
    )
    assert not rd.model_lead_tracker.allow_closing_governor_calm_recovery

    production = self._track()
    reference = self._track()
    for tr in (production, reference):
      self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)
      # Isolate CD9's publish clamp from the ordinary filtered vRel. With calm
      # recovery disabled, the earned 1.8 m/s clamp must remain byte/kinematic
      # identical to the explicit recovery-disabled reference.
      tr.vRel = -0.4
      tr.vLead = 19.6
      tr.vLeadK = 19.6

    production_lead = self._update_through_tracker(rd.model_lead_tracker, production)
    reference_tracker = ModelLeadTracker(
      params=_NoParams(),
      allow_closing_governor_calm_recovery=False,
    )
    reference_lead = self._update_through_tracker(reference_tracker, reference)

    assert production_lead["closingGovernorRecovery"] is False
    assert production_lead["closingGovernorRecoveryNumericValid"] is False
    assert production.governor_closing_mps == pytest.approx(1.8)
    assert production_lead["vRel"] == pytest.approx(-1.8)
    assert self._lead_wire_bytes(production_lead) == self._lead_wire_bytes(reference_lead)

  @pytest.mark.parametrize(
    ("brand", "fingerprint", "radar_unavailable"),
    [
      ("hyundai", CAR.KIA_EV6, False),
      ("hyundai", "HYUNDAI_IONIQ_5", True),
      ("toyota", CAR.KIA_EV6, True),
    ],
  )
  def test_production_recovery_scope_requires_exact_ev6_vision_topology(
    self, brand, fingerprint, radar_unavailable,
  ):
    rd = self._production_radard(
      brand=brand,
      fingerprint=fingerprint,
      radar_unavailable=radar_unavailable,
    )
    assert not rd.model_lead_tracker.allow_closing_governor_calm_recovery

  def test_ev6_radar_unavailable_production_radard_still_activates_recovery(self):
    assert ModelLeadTracker(params=_NoParams()).allow_closing_governor_calm_recovery
    rd = self._production_radard(
      brand="hyundai",
      fingerprint=CAR.KIA_EV6,
      radar_unavailable=True,
    )
    assert rd.model_lead_tracker.allow_closing_governor_calm_recovery

    tr = self._track()
    self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)
    tr.vRel = -0.4
    tr.vLead = 19.6
    tr.vLeadK = 19.6
    lead = self._update_through_tracker(rd.model_lead_tracker, tr)

    assert tr.governor_calm_recovery_applied
    assert tr.governor_closing_mps == pytest.approx(0.6)
    assert lead["closingGovernorRecovery"] is True
    assert lead["closingGovernorRecoveryNumericValid"] is True
    assert lead["vRel"] == pytest.approx(-0.6)

  def test_active_hold_recovers_with_two_nonbraking_high_ttc_frames_and_full_position_span(self):
    tr = self._track()
    self._seed_histories(tr, previous_raw_closing_mps=2.5)
    deadline = tr.governor_hold_until_t

    # The immediately preceding measured frame is exactly at the fixed
    # fast-close boundary, but it is non-braking at a 40 s raw TTC. The robust
    # range proof may therefore remove only stale CD9 excess while retaining
    # that frame's complete raw velocity authority.
    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert tr.governor_calm_recovery_applied
    assert tr.governor_closing_mps == pytest.approx(0.6)
    assert tr.governor_hold_until_t == pytest.approx(deadline)

    # The next 20 Hz sample retains the earned mode. The two-second position
    # history still spans 1.6 s after window trimming.
    now = self.NOW + 0.05
    drel = self.DREL_M - 0.6 * 0.05
    assert self._update(tr, now=now, drel_m=drel, raw_closing_mps=0.4)
    assert tr.governor_calm_recovery_applied
    assert tr.governor_reason == "calm_recovery_capped"
    assert tr.governor_recovery_position_closing_mps == pytest.approx(0.6)
    assert tr.governor_recovery_vrel_floor_mps == pytest.approx(-0.6)
    assert tr.governor_closing_mps == pytest.approx(0.6)
    assert tr.governor_hold_until_t == pytest.approx(deadline)
    assert tr.get_RadarState(self._cfg())["closingGovernorRecovery"] is True

  def test_recovery_provenance_clears_on_safety_veto(self):
    tr = self._track()
    self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)
    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert tr.get_RadarState(self._cfg())["closingGovernorRecovery"] is True

    # Current braking exits recovery immediately; the published field follows
    # the behavior state rather than the logging-only replay debug payload.
    assert self._update(
      tr, now=self.NOW + 0.05, drel_m=self.DREL_M - 0.03,
      raw_closing_mps=0.4, raw_alead_mps2=-0.21,
    )
    assert tr.governor_calm_recovery_mode is False
    assert tr.get_RadarState(self._cfg())["closingGovernorRecovery"] is False

  @pytest.mark.parametrize(
    ("context", "expected_safe"),
    [
      ({"raw_prob": 0.59}, False),
      ({"raw_dpath": 1.51}, False),
      ({"raw_dpath": 1.30, "raw_vlat": 0.70}, False),
      ({"raw_prob": 0.60, "raw_dpath": 1.25, "raw_vlat": 0.70}, True),
    ],
  )
  def test_numeric_recovery_requires_current_raw_probability_and_lateral_context(self, context, expected_safe):
    tr = self._track()
    self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)

    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4, **context,
    )
    state = tr.get_RadarState(self._cfg())
    # Probability/lateral context gates only the numeric bridge proof. The
    # public CD9 clamp remains conservative at max(raw, robust position), and
    # the planner fails closed when numeric provenance is absent.
    assert state["closingGovernorRecovery"] is True
    assert state["closingGovernorRecoveryNumericValid"] is expected_safe
    assert tr.governor_recovery_position_closing_mps is not None
    assert (tr.governor_recovery_vrel_floor_mps is not None) is expected_safe

  def test_missed_frame_clears_numeric_recovery_provenance(self):
    tr = self._track()
    self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)
    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert tr.get_RadarState(self._cfg())["closingGovernorRecoveryNumericValid"] is True

    tracker = ModelLeadTracker(params=_NoParams())
    tracker._tracks[tr.identifier] = tr
    tracker.begin_frame(self.NOW + 0.05)
    tracker.end_frame()

    state = tr.get_RadarState(self._cfg())
    assert state["closingGovernorRecovery"] is False
    assert state["closingGovernorRecoveryNumericValid"] is False
    assert tr.governor_recovery_position_closing_mps is None
    assert tr.governor_recovery_vrel_floor_mps is None

  def test_slot_transition_clears_position_history_before_recovery(self):
    tr = self._track()
    self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)
    assert len(tr.opening_position_evidence) > 16

    tr.update(
      {
        "dRel": self.DREL_M,
        "yRel": 0.0,
        "vRel": -0.4,
        "vLead": 19.6,
        "aLeadK": 0.0,
        "modelProb": 0.9,
        "dPath": 0.0,
        "vLat": 0.0,
      },
      self.NOW,
      v_ego=20.0,
      cfg=self._cfg(),
      lead_slot=1,
    )
    assert len(tr.opening_position_evidence) == 1
    assert tr.opening_position_evidence[0][0] == pytest.approx(self.NOW)
    assert tr.governor_calm_recovery_mode is False
    assert tr.governor_calm_recovery_applied is False
    # recovery stays out; only the bounded threat-free stale decay applies
    assert tr.governor_closing_mps == pytest.approx(1.8 - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05)

  @pytest.mark.parametrize(
    "raw_closing_mps,position_closing_mps,expected_mps",
    [
      (0.9, 0.4, 0.9),
      (0.4, 0.8, 0.8),
    ],
  )
  def test_recovery_entry_replaces_clamp_with_exact_max_of_raw_and_long_position(
    self, raw_closing_mps, position_closing_mps, expected_mps,
  ):
    tr = self._track()
    self._seed_histories(
      tr,
      raw_closing_mps=raw_closing_mps,
      position_closing_mps=position_closing_mps,
    )
    deadline = tr.governor_hold_until_t

    assert self._update(
      tr,
      now=self.NOW,
      drel_m=self.DREL_M,
      raw_closing_mps=raw_closing_mps,
    )
    assert tr.governor_calm_recovery_applied
    assert tr.governor_recovery_position_closing_mps == pytest.approx(position_closing_mps)
    # The public governor clamp stays conservative at max(raw, position), while
    # the planner's positive-only release proof is independently position-only.
    assert tr.governor_recovery_vrel_floor_mps == pytest.approx(-position_closing_mps)
    assert tr.governor_closing_mps == pytest.approx(expected_mps)
    assert tr.governor_hold_until_t == pytest.approx(deadline)

  def test_recovery_rejects_same_frame_raw_position_close_above_limit(self):
    tr = self._track()
    self._seed_histories(
      tr,
      raw_closing_mps=0.4,
      position_closing_mps=0.6,
    )

    # The retained robust window is mild, but the newest raw range step closes
    # 2.0 m/s. It must veto recovery before the long estimator can catch up.
    assert self._update(
      tr,
      now=self.NOW,
      drel_m=self.DREL_M - 0.07,
      raw_closing_mps=0.4,
    )
    assert tr.governor_calm_recovery_mode is False
    assert tr.governor_recovery_vrel_floor_mps is None

  def test_recovery_mode_tracks_current_raw_closing_without_restoring_stale_extra_clamp(self):
    tr = self._track()
    self._seed_histories(tr, raw_closing_mps=0.4, position_closing_mps=0.6)
    deadline = tr.governor_hold_until_t

    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert tr.governor_calm_recovery_mode
    assert tr.governor_closing_mps == pytest.approx(0.6)

    # Current raw closing rises to 1.69 m/s while the robust position history
    # stays mild. Recovery follows that current raw urgency exactly, but does
    # not restore the older 1.8 m/s stale CD9 clamp.
    now = self.NOW + 0.05
    assert self._update(
      tr,
      now=now,
      drel_m=self.DREL_M - 0.6 * 0.05,
      raw_closing_mps=1.69,
    )
    assert tr.governor_calm_recovery_mode
    assert tr.governor_calm_recovery_applied
    assert tr.governor_recovery_position_closing_mps == pytest.approx(0.6)
    assert tr.governor_closing_mps == pytest.approx(1.69)
    assert tr.governor_hold_until_t == pytest.approx(deadline)

  def test_inactive_hold_cannot_enter_calm_recovery(self):
    tr = self._track(active=False)
    self._seed_histories(tr)
    deadline = tr.governor_hold_until_t

    active = self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )

    assert not active
    assert not tr.governor_calm_recovery_applied
    assert tr.governor_recovery_position_closing_mps is None
    assert tr.governor_closing_mps == pytest.approx(1.8)
    assert tr.governor_hold_until_t == pytest.approx(deadline)

  def test_active_but_uncorroborated_hold_cannot_enter_calm_recovery(self):
    tr = self._track(threat_corroborated=False)
    self._seed_histories(tr)

    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert not tr.governor_calm_recovery_mode
    assert not tr.governor_calm_recovery_applied
    assert tr.governor_closing_mps == pytest.approx(1.8 - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05)

  def test_missed_model_frame_vetoes_recovery_on_the_next_measurement(self):
    tr = self._track()
    self._seed_histories(tr)
    tr.governor_calm_recovery_mode = True
    tr.governor_calm_recovery_applied = True
    tracker = ModelLeadTracker(params=_NoParams())
    tracker._tracks[tr.identifier] = tr
    tracker.begin_frame(self.NOW)
    tracker.end_frame()
    assert tr.missed == 1
    assert not tr.governor_calm_recovery_mode
    assert not tr.governor_calm_recovery_applied

    now = self.NOW + 0.05
    active = self._update(
      tr,
      now=now,
      drel_m=self.DREL_M - 0.6 * 0.05,
      raw_closing_mps=0.4,
    )

    assert active
    assert not tr.governor_calm_recovery_applied
    assert tr.governor_recovery_position_closing_mps is None
    # the decay step spans the missed model frame (0.10 s sample gap)
    assert tr.governor_closing_mps == pytest.approx(1.8 - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.10)

  @pytest.mark.parametrize("veto", ("current_braking", "short_ttc", "position"))
  def test_any_safety_veto_exits_existing_recovery_mode(self, veto):
    tr = self._track(closing_mps=0.6)
    tr.governor_calm_recovery_mode = True
    if veto == "current_braking":
      drel_m, raw_closing_mps, raw_alead_mps2, position_closing_mps = self.DREL_M, 0.4, -0.21, 0.6
    elif veto == "short_ttc":
      drel_m, raw_closing_mps, raw_alead_mps2, position_closing_mps = 2.4, 0.4, 0.0, 0.1
    else:
      drel_m, raw_closing_mps, raw_alead_mps2, position_closing_mps = self.DREL_M, 0.4, 0.0, 1.251
    self._seed_histories(
      tr,
      drel_m=drel_m,
      raw_closing_mps=raw_closing_mps,
      position_closing_mps=position_closing_mps,
    )

    assert self._update(
      tr,
      now=self.NOW,
      drel_m=drel_m,
      raw_closing_mps=raw_closing_mps,
      raw_alead_mps2=raw_alead_mps2,
    )
    assert not tr.governor_calm_recovery_mode
    assert not tr.governor_calm_recovery_applied

  def test_far_high_ttc_fast_close_keeps_existing_recovery_and_full_raw_urgency(self):
    tr = self._track(closing_mps=0.6)
    tr.governor_calm_recovery_mode = True
    self._seed_histories(
      tr,
      raw_closing_mps=3.2,
      position_closing_mps=0.6,
    )

    assert self._update(
      tr,
      now=self.NOW,
      drel_m=self.DREL_M,
      raw_closing_mps=3.2,
    )
    assert tr.governor_calm_recovery_mode
    assert tr.governor_calm_recovery_applied
    assert tr.governor_closing_mps == pytest.approx(3.2)

  @pytest.mark.parametrize("history_shape", ("position_gap", "sparse", "short_span"))
  def test_position_history_gap_sparse_or_short_span_vetoes_recovery(self, history_shape):
    tr = self._track()
    full_times = [self.NOW - 2.0 + 0.05 * i for i in range(40)]
    if history_shape == "position_gap":
      position_times = [
        sample_t for sample_t in full_times
        if not self.NOW - 0.95 <= sample_t <= self.NOW - 0.85
      ]
    elif history_shape == "sparse":
      position_times = list(np.linspace(self.NOW - 2.0, self.NOW - 0.05, 20))
    else:
      position_times = [self.NOW - 1.55 + 0.05 * i for i in range(31)]
    self._seed_histories(tr, position_times=position_times)

    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert not tr.governor_calm_recovery_applied
    assert tr.governor_recovery_position_closing_mps is None
    assert tr.governor_closing_mps == pytest.approx(1.8 - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05)

  @pytest.mark.parametrize("braking_location", ("current", "window"))
  def test_current_or_windowed_braking_vetoes_recovery(self, braking_location):
    tr = self._track()
    if braking_location == "current":
      self._seed_histories(tr)
      current_alead = -0.21
    else:
      # The two current samples are calm, but the complete 0.6 s evidence
      # window remains just beyond the fixed braking boundary.
      self._seed_histories(
        tr,
        window_alead_mps2=-0.25,
        previous_alead_mps2=0.0,
      )
      current_alead = 0.0

    cfg = self._cfg(opening_governor_alead_veto_mps2=100.0)
    assert self._update(
      tr,
      now=self.NOW,
      drel_m=self.DREL_M,
      raw_closing_mps=0.4,
      raw_alead_mps2=current_alead,
      cfg=cfg,
    )
    assert not tr.governor_calm_recovery_applied
    assert tr.governor_recovery_position_closing_mps is None
    assert tr.governor_closing_mps == pytest.approx(1.8)

  @pytest.mark.parametrize(
    "drel_m,expected_recovery",
    [
      (4.8, False),       # exactly 12.0 s raw TTC: fail closed
      (4.8004, True),     # 12.001 s raw TTC: strictly beyond the veto
    ],
  )
  def test_recovery_raw_ttc_boundary_is_strictly_greater_than_twelve_seconds(self, drel_m, expected_recovery):
    assert CLOSING_GOVERNOR_RECOVERY_MIN_RAW_TTC_S == pytest.approx(12.0)
    tr = self._track(drel_m=drel_m)
    self._seed_histories(
      tr,
      drel_m=drel_m,
      position_closing_mps=0.1,
      raw_closing_mps=0.4,
    )

    assert self._update(
      tr, now=self.NOW, drel_m=drel_m, raw_closing_mps=0.4,
    )
    assert tr.governor_calm_recovery_applied is expected_recovery
    assert tr.governor_closing_mps == pytest.approx(0.4 if expected_recovery else 1.8 - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05)
    assert (tr.governor_recovery_vrel_floor_mps is not None) is expected_recovery

  @pytest.mark.parametrize(
    "raw_closing_mps,expected_recovery",
    [
      (2.49, True),
      (2.50, False),
    ],
  )
  def test_raw_fast_close_boundary_2p49_vs_2p50(self, raw_closing_mps, expected_recovery):
    assert OPENING_GOVERNOR_HARD_CLOSING_MPS == pytest.approx(2.5)
    tr = self._track(closing_mps=3.0)
    self._seed_histories(
      tr,
      position_closing_mps=0.5,
      raw_closing_mps=raw_closing_mps,
    )

    assert self._update(
      tr,
      now=self.NOW,
      drel_m=self.DREL_M,
      raw_closing_mps=raw_closing_mps,
    )
    assert tr.governor_calm_recovery_applied is expected_recovery
    assert tr.governor_closing_mps == pytest.approx(raw_closing_mps if expected_recovery else 3.0)

  @pytest.mark.parametrize(
    "position_closing_mps,expected_recovery",
    [
      (1.25, True),
      (1.251, False),
    ],
  )
  def test_long_position_boundary_1p25_vs_1p251(self, position_closing_mps, expected_recovery):
    assert CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS == pytest.approx(1.25)
    tr = self._track()
    self._seed_histories(
      tr,
      position_closing_mps=position_closing_mps,
      raw_closing_mps=0.4,
    )

    assert self._update(
      tr, now=self.NOW, drel_m=self.DREL_M, raw_closing_mps=0.4,
    )
    assert tr.governor_calm_recovery_applied is expected_recovery
    assert tr.governor_closing_mps == pytest.approx(position_closing_mps if expected_recovery else 1.8 - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05)

  def test_first_braking_onset_vetoes_recovery_before_window_or_position_can_catch_up(self):
    tr = self._track()
    self._seed_histories(
      tr,
      position_closing_mps=0.2,
      raw_closing_mps=0.4,
      window_alead_mps2=0.0,
    )
    deadline = tr.governor_hold_until_t

    # Adversarial onset-lag shape: all historical evidence looks calm and the
    # long position estimator has not reacted, but the current raw aLead is the
    # first braking sample. Current evidence must win immediately.
    assert self._update(
      tr,
      now=self.NOW,
      drel_m=self.DREL_M,
      raw_closing_mps=0.4,
      raw_alead_mps2=-0.21,
      cfg=self._cfg(),
    )
    assert not tr.governor_calm_recovery_applied
    assert tr.governor_reason == "current_braking"
    assert tr.governor_threat_corroborated
    assert tr.governor_closing_mps == pytest.approx(1.8)
    assert tr.governor_hold_until_t == pytest.approx(deadline)


class TestClosingGovernorStaleDecayAndSignificance:
  """CD9 phantom hardening (2026-07-16 review): an active-but-unrearmed hold
  bleeds toward current velocity evidence at a bounded rate instead of
  freezing at its historical worst (captured false-closing episodes held
  -2.9 m/s published closing for 0.6+ s while raw vRel had flipped positive),
  and position-only arming must clear the measured slope noise of its own
  window, not just the fixed margin (+-1-1.5 m dRel jitter at 40+ m is
  1.5-3 m/s of pure endpoint-slope noise).  Both are EV6-scoped via
  allow_calm_recovery; legacy topologies stay byte-exact."""

  NOW = 10.0

  @staticmethod
  def _cfg(**overrides):
    base = {
      "closing_governor_margin_mps": 0.75,
      "closing_governor_min_closing_mps": 0.30,
      "closing_governor_hold_s": 1.0,
      "closing_governor_accel_onset_mps2": 99.0,
      "closing_governor_pos_trust_excess_mps": 1.5,
    }
    base.update(overrides)
    return dataclasses.replace(LeadResponseTuningConfig.defaults(), **base)

  def _track(self, *, published_vrel=-0.2, drel_state=44.0):
    tr = ModelLeadTrack.from_lead_dict(
      9, {"dRel": drel_state, "yRel": 0.0, "vRel": published_vrel,
          "vLead": 20.0 + published_vrel, "aLeadK": 0.0, "modelProb": 0.9},
      self.NOW - 1.0, 0)
    tr.vRel = float(published_vrel)
    tr.vLead = 20.0 + float(published_vrel)
    tr.vLeadK = tr.vLead
    tr.dRel = float(drel_state)
    return tr

  def _seed_closing_burst(self, tr, *, drel0=44.0, closing=2.8, n=13):
    t0 = self.NOW - 0.6
    for i in range(n):
      t = t0 + 0.05 * i
      d = drel0 + closing * (self.NOW - t)
      tr.closing_evidence.append((t, d, -closing, 0.0))
      tr.opening_position_evidence.append((t, d))

  def _update(self, tr, *, now, drel_m, raw_closing_mps, raw_alead_mps2=0.0,
              allow_calm_recovery=True, cfg=None):
    return tr._update_closing_governor(
      float(now), raw_drel=float(drel_m), raw_vrel=-float(raw_closing_mps),
      raw_alead=float(raw_alead_mps2), cfg=cfg or self._cfg(),
      allow_calm_recovery=bool(allow_calm_recovery),
    )

  def _arm_velocity_burst(self, *, allow=True):
    tr = self._track()
    self._seed_closing_burst(tr)
    assert self._update(tr, now=self.NOW, drel_m=44.0, raw_closing_mps=2.8,
                        allow_calm_recovery=allow)
    assert tr.governor_closing_mps == pytest.approx(2.8, abs=0.05)
    # published state caught up during the burst (governor forces fast taus),
    # so the stale window slope cannot re-arm the position path afterwards
    tr.vRel = -2.7
    tr.vLead = 20.0 - 2.7
    tr.vLeadK = tr.vLead
    return tr

  def test_stale_clamp_decays_to_evidence_and_releases_early(self):
    tr = self._arm_velocity_burst()
    armed = float(tr.governor_closing_mps)
    hold_until = float(tr.governor_hold_until_t)

    # first threat-free calm frame: decay is rate-limited AND floored by the
    # draining window mean (13x2.8 + 1x0.0)/14 = 2.6
    active = self._update(tr, now=self.NOW + 0.05, drel_m=44.0, raw_closing_mps=0.0)
    assert active
    assert tr.governor_reason == "stale_decay"
    assert tr.governor_closing_mps == pytest.approx(
      max(armed - CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05, 2.8 * 13.0 / 14.0), abs=0.02)

    last = float(tr.governor_closing_mps)
    released_at = None
    for j in range(2, 18):
      self._update(tr, now=self.NOW + 0.05 * j, drel_m=44.0, raw_closing_mps=0.0)
      assert tr.governor_closing_mps <= last + 1e-9
      if j == 5:
        # mid-hold the clamp has bled to the draining evidence level instead
        # of the armed worst (the legacy pin below holds 2.8 at this point)
        assert tr.governor_closing_mps <= armed - 4 * CLOSING_GOVERNOR_STALE_DECAY_MPS2 * 0.05 + 1e-6
        # the hold itself must stay alive through the decay: it carries the
        # same-frame threat-upgrade path
        assert tr.governor_hold_until_t == pytest.approx(hold_until)
      last = float(tr.governor_closing_mps)
      if tr.governor_hold_until_t < 0.0:
        released_at = self.NOW + 0.05 * j
        break
    # once the window mean drains under the release threshold the ordinary
    # release path clears the hold, well before the natural 1.0 s expiry
    assert released_at is not None and released_at < hold_until
    assert tr.governor_closing_mps == pytest.approx(0.0)
    assert tr.governor_reason == "inactive"
    assert not tr.governor_threat_corroborated

  def test_threat_during_decayed_hold_restores_full_authority_same_frame(self):
    tr = self._arm_velocity_burst()
    for j in range(1, 4):
      self._update(tr, now=self.NOW + 0.05 * j, drel_m=44.0, raw_closing_mps=0.0)
    assert tr.governor_reason == "stale_decay"
    decayed = float(tr.governor_closing_mps)
    assert decayed < 2.8
    # short raw TTC lands mid-hold: same-frame upgrade with authority restored
    # from the current raw closure, never softened by the decayed value
    active = self._update(tr, now=self.NOW + 0.20, drel_m=12.0, raw_closing_mps=3.0)
    assert active
    assert tr.governor_reason == "short_raw_ttc"
    assert tr.governor_threat_corroborated
    assert tr.governor_closing_mps >= 3.0 - 1e-6

  def test_stale_clamp_stays_frozen_on_legacy_topologies(self):
    tr = self._arm_velocity_burst(allow=False)
    armed = float(tr.governor_closing_mps)
    for j in range(1, 11):
      active = self._update(tr, now=self.NOW + 0.05 * j, drel_m=44.0,
                            raw_closing_mps=0.0, allow_calm_recovery=False)
      assert active
      assert tr.governor_closing_mps == pytest.approx(armed)
    # natural expiry only
    assert not self._update(tr, now=self.NOW + 1.05, drel_m=44.0,
                            raw_closing_mps=0.0, allow_calm_recovery=False)

  def test_stale_decay_freezes_through_evidence_dropout(self):
    tr = self._arm_velocity_burst()
    armed = float(tr.governor_closing_mps)
    # a genuine evidence dropout leaves a sparse window: the density gate
    # early-returns and the hold bridges with the clamp exactly frozen
    active = self._update(tr, now=self.NOW + 0.55, drel_m=44.0, raw_closing_mps=0.0)
    assert active
    assert tr.governor_closing_mps == pytest.approx(armed)

  def test_braking_lead_keeps_clamp_frozen_during_hold(self):
    tr = self._arm_velocity_burst()
    armed = float(tr.governor_closing_mps)
    active = self._update(tr, now=self.NOW + 0.05, drel_m=44.0,
                          raw_closing_mps=0.0, raw_alead_mps2=-0.25)
    assert active
    assert tr.governor_closing_mps == pytest.approx(armed)

  @staticmethod
  def _jittered_window(tr, now, *, drel0=45.0, jitter):
    # mild real slope (-0.3 m/s) plus scatter whose endpoint means read
    # ~2.1 m/s of phantom closing while the residuals scatter +-1.2 m: the
    # excess clears the fixed margin but not the measured slope noise
    offsets = [0.5, 0.7, 0.15, -1.2, 1.1, -0.9, 1.2, -1.1, 0.9, -1.2, -0.5, -0.1, -0.75]
    t0 = now - 0.6
    for i in range(13):
      t = t0 + 0.05 * i
      d = drel0 - 0.3 * (t - t0) + (offsets[i] if jitter else 0.0)
      tr.closing_evidence.append((t, d, -0.5, 0.0))
      tr.opening_position_evidence.append((t, d))

  def test_position_arm_requires_significance_under_jitter(self):
    # jittered window: legacy arms on the phantom slope, EV6 must not
    tr = self._track()
    self._jittered_window(tr, self.NOW, jitter=True)
    assert self._update(tr, now=self.NOW, drel_m=44.0, raw_closing_mps=0.5,
                        allow_calm_recovery=False)
    tr_fixed = self._track()
    self._jittered_window(tr_fixed, self.NOW, jitter=True)
    assert not self._update(tr_fixed, now=self.NOW, drel_m=44.0, raw_closing_mps=0.5,
                            allow_calm_recovery=True)

  def test_position_arm_keeps_legacy_sensitivity_on_clean_stream(self):
    # clean closing stream: residuals ~0, the fixed margin still binds and the
    # EV6 path arms exactly like legacy
    tr = self._track()
    self._seed_closing_burst(tr, closing=1.6)
    assert self._update(tr, now=self.NOW, drel_m=44.0, raw_closing_mps=1.6,
                        allow_calm_recovery=True)
    assert tr.governor_closing_mps > 0.0


class TestOpeningGovernor:
  """Opening governor (CD9's mirror): publish-time one-directional vRel relax
  while the raw position window proves sustained opening. Anchored to the
  2026-07-08 phantom-closing runs (published vRel <= -1.0 while raw dRel opened
  >= 0.2 m/s, 22.3% of lead frames, runs up to 6.9 s)."""

  NOW = 10.0

  @staticmethod
  def _cfg(**overrides):
    # CD9 margin sentinel keeps the closing governor out of the way so these
    # tests isolate the opening side.
    base = {"closing_governor_margin_mps": 99.0}
    base.update(overrides)
    return dataclasses.replace(LeadResponseTuningConfig.defaults(), **base)

  def _track(self, *, vrel_state=-2.0, drel_state=40.0, slope=0.7, raw_vrel=-0.4,
             raw_alead=0.0, n_samples=14):
    tr = ModelLeadTrack.from_lead_dict(
      7, {"dRel": drel_state, "yRel": 0.0, "vRel": vrel_state, "vLead": 20.0 + vrel_state,
          "aLeadK": 0.0, "modelProb": 0.9}, self.NOW - 1.0, 0)
    tr.vRel = float(vrel_state)
    tr.vLead = 20.0 + float(vrel_state)
    tr.vLeadK = tr.vLead
    tr.dRel = float(drel_state)
    t0 = self.NOW - 0.55
    for i in range(n_samples):
      t = t0 + 0.55 * i / (n_samples - 1)
      d = drel_state + slope * (t - self.NOW)
      tr.closing_evidence.append((t, d, raw_vrel, raw_alead))
      tr.opening_position_evidence.append((t, d))
    return tr

  @staticmethod
  def _replace_evidence(tr, *, now, drel, slope, raw_vrel=-0.4, raw_alead=0.0, n_samples=14):
    tr.closing_evidence.clear()
    t0 = now - 0.55
    for i in range(n_samples):
      t = t0 + 0.55 * i / (n_samples - 1)
      tr.closing_evidence.append((t, drel + slope * (t - now), raw_vrel, raw_alead))

  def test_phantom_closing_relax_arms_and_floors_publish(self):
    # Trace anchor: published vRel ~-2 while position opens ~0.7 m/s.
    tr = self._track()
    cfg = self._cfg()
    tr._update_opening_governor(self.NOW, False, cfg)
    assert tr.opening_relax_vrel == pytest.approx(0.0)
    rs = tr.get_RadarState(cfg)
    assert rs["vRel"] == pytest.approx(0.0)
    assert rs["vLead"] == pytest.approx(20.0)  # v_ego_est + relaxed vRel

  def test_opening_hold_bridges_ambiguous_raw_position_window(self):
    tr = self._track()
    cfg = self._cfg()
    tr._update_opening_governor(self.NOW, False, cfg)
    deadline = tr.opening_relax_hold_until_t

    now = self.NOW + 0.2
    self._replace_evidence(tr, now=now, drel=40.0, slope=0.1)
    tr._update_opening_governor(now, False, cfg, raw_drel=40.0, raw_vrel=-0.4,
                                raw_alead=0.0, raw_dpath=0.0, raw_vlat=0.0, v_ego=20.0)

    assert tr.opening_relax_vrel == pytest.approx(0.0)
    assert tr.opening_relax_held
    assert tr.opening_relax_hold_until_t == pytest.approx(deadline)

  def test_opening_hold_zero_restores_per_frame_rollback(self):
    tr = self._track()
    cfg = self._cfg(opening_governor_hold_s=0.0)
    tr._update_opening_governor(self.NOW, False, cfg)
    assert tr.opening_relax_vrel == pytest.approx(0.0)

    now = self.NOW + 0.2
    self._replace_evidence(tr, now=now, drel=40.0, slope=0.1)
    tr._update_opening_governor(now, False, cfg)

    assert tr.opening_relax_vrel is None
    assert tr.opening_relax_hold_vrel is None

  def test_opening_hold_survives_only_weak_position_governor_until_deadline(self):
    tr = self._track(vrel_state=-0.2, drel_state=60.0)
    cfg = self._cfg(closing_governor_margin_mps=0.75)
    tr._update_opening_governor(self.NOW, False, cfg)
    deadline = tr.opening_relax_hold_until_t

    now = self.NOW + 0.05
    self._replace_evidence(tr, now=now, drel=60.0, slope=-2.0)
    active = tr._update_closing_governor(
      now, raw_drel=60.0, raw_vrel=-0.4, raw_alead=0.0, cfg=cfg,
    )
    tr.governor_active = active
    tr._update_opening_governor(now, active, cfg, raw_drel=60.0, raw_vrel=-0.4,
                                raw_alead=0.0, raw_dpath=0.0, raw_vlat=0.0, v_ego=20.0)

    assert active
    assert tr.governor_reason == "position_only_weak"
    assert not tr.governor_threat_corroborated
    assert tr.opening_relax_vrel == pytest.approx(0.0)
    assert tr.get_RadarState(cfg)["vRel"] == pytest.approx(0.0)

    expired = deadline + 1e-3
    tr._update_opening_governor(expired, True, cfg, raw_drel=60.0, raw_vrel=-0.4,
                                raw_alead=0.0, raw_dpath=0.0, raw_vlat=0.0, v_ego=20.0)
    assert tr.opening_relax_vrel is None
    assert tr.get_RadarState(cfg)["vRel"] == pytest.approx(-tr.governor_closing_mps)

  def test_active_weak_governor_upgrades_when_velocity_window_turns_closing(self):
    tr = self._track(vrel_state=-0.2, drel_state=60.0)
    cfg = self._cfg(
      closing_governor_margin_mps=0.75,
      opening_governor_raw_closing_veto_mps=99.0,
    )
    tr._update_opening_governor(self.NOW, False, cfg)
    assert tr.opening_relax_hold_vrel is not None

    weak_now = self.NOW + 0.05
    self._replace_evidence(tr, now=weak_now, drel=60.0, slope=-2.0)
    assert tr._update_closing_governor(
      weak_now, raw_drel=60.0, raw_vrel=-0.4, raw_alead=0.0, cfg=cfg,
    )
    assert tr.governor_reason == "position_only_weak"
    assert not tr.governor_threat_corroborated

    # The position arm disappears, but a dense raw-velocity window now proves
    # sustained closure. The fixed safety ceiling must override the permissive
    # tunable veto and revoke the opening hold in this same frame.
    closing_now = weak_now + 0.05
    self._replace_evidence(
      tr, now=closing_now, drel=60.0, slope=0.0, raw_vrel=-1.2,
    )
    active = tr._update_closing_governor(
      closing_now, raw_drel=60.0, raw_vrel=-1.2, raw_alead=0.0, cfg=cfg,
    )
    assert active
    assert tr.governor_reason == "velocity_corroborated"
    assert tr.governor_threat_corroborated

    tr._update_opening_governor(
      closing_now, active, cfg, raw_drel=60.0, raw_vrel=-1.2,
      raw_alead=0.0, raw_dpath=0.0, raw_vlat=0.0, v_ego=20.0,
    )
    assert tr.opening_relax_vrel is None
    assert tr.opening_relax_hold_vrel is None

  def test_weak_position_governor_cannot_create_opening_hold(self):
    tr = self._track(vrel_state=-0.2, drel_state=60.0, slope=-2.0)
    cfg = self._cfg(closing_governor_margin_mps=0.75)
    active = tr._update_closing_governor(
      self.NOW, raw_drel=60.0, raw_vrel=-0.4, raw_alead=0.0, cfg=cfg,
    )
    tr._update_opening_governor(self.NOW, active, cfg, raw_drel=60.0, raw_vrel=-0.4,
                                raw_alead=0.0, raw_dpath=0.0, raw_vlat=0.0, v_ego=20.0)

    assert active
    assert tr.governor_reason == "position_only_weak"
    assert tr.opening_relax_vrel is None
    assert tr.opening_relax_hold_vrel is None

  def test_correlated_closing_governor_stays_authoritative_through_hold(self):
    tr = self._track(vrel_state=-0.2, drel_state=60.0, slope=-2.0, raw_alead=-0.5)
    cfg = self._cfg(closing_governor_margin_mps=0.75)
    assert tr._update_closing_governor(
      self.NOW, raw_drel=60.0, raw_vrel=-0.4, raw_alead=-0.5, cfg=cfg,
    )
    assert tr.governor_threat_corroborated

    now = self.NOW + 0.05
    self._replace_evidence(tr, now=now, drel=60.0, slope=-2.0, raw_alead=0.0)
    active = tr._update_closing_governor(
      now, raw_drel=60.0, raw_vrel=-0.4, raw_alead=0.0, cfg=cfg,
    )

    assert active
    assert tr.governor_threat_corroborated
    assert tr.governor_reason != "position_only_weak"
    tr._update_opening_governor(now, active, cfg)
    assert tr.opening_relax_vrel is None

  def test_miss_and_slot_change_clear_hold_and_require_fresh_proof(self):
    cfg = self._cfg()

    missed = self._track()
    missed._update_opening_governor(self.NOW, False, cfg)
    tracker = ModelLeadTracker(params=_NoParams())
    tracker._tracks[missed.identifier] = missed
    tracker.begin_frame(self.NOW + 0.05)
    tracker.end_frame()
    assert missed.opening_relax_hold_vrel is None
    assert missed.opening_evidence_epoch_t >= missed.last_t

    reordered = self._track()
    reordered._update_opening_governor(self.NOW, False, cfg)
    rs = reordered.update(
      {"dRel": 40.0, "yRel": 0.0, "vRel": -0.4, "aLeadK": 0.0,
       "modelProb": 0.9, "dPath": 0.0, "vLat": 0.0},
      self.NOW + 0.05, 20.0, cfg, 1,
    )
    assert reordered.opening_relax_hold_vrel is None
    assert reordered.opening_evidence_epoch_t == pytest.approx(self.NOW + 0.05)
    assert rs["vRel"] < -0.05

  @pytest.mark.parametrize(
    "raw_drel,raw_vrel,raw_alead,raw_vlat,v_ego",
    [
      (40.0, -0.4, -0.5, 0.0, 20.0),  # one current braking sample
      (18.0, -4.0, 0.0, 0.0, 20.0),   # short raw TTC / fast close
      (60.0, -1.0, 0.0, 0.0, 0.5),   # negative raw vLead (oncoming)
      (40.0, -0.4, 0.0, 0.8, 20.0),  # same-track lateral migration
    ],
  )
  def test_current_threat_clears_opening_hold_same_frame(self, raw_drel, raw_vrel, raw_alead,
                                                         raw_vlat, v_ego):
    tr = self._track()
    cfg = self._cfg(opening_governor_raw_closing_veto_mps=100.0,
                    opening_governor_alead_veto_mps2=100.0)
    tr._update_opening_governor(self.NOW, False, cfg)
    assert tr.opening_relax_hold_vrel is not None

    tr._update_opening_governor(
      self.NOW + 0.05, False, cfg,
      raw_drel=raw_drel, raw_vrel=raw_vrel, raw_alead=raw_alead,
      raw_dpath=0.0, raw_vlat=raw_vlat, v_ego=v_ego,
    )

    assert tr.opening_relax_vrel is None
    assert tr.opening_relax_hold_vrel is None
    assert tr.opening_relax_hold_until_t < self.NOW

  def test_stale_closing_hold_releases_when_window_is_near_parity(self):
    tr = self._track(raw_vrel=-0.1, raw_alead=0.0)
    tr.governor_hold_until_t = self.NOW + 1.0
    tr.governor_closing_mps = 1.8

    active = tr._update_closing_governor(
      self.NOW, raw_drel=40.0, raw_vrel=-0.1, raw_alead=0.0, cfg=self._cfg(closing_governor_margin_mps=0.35),
    )

    assert not active
    assert tr.governor_hold_until_t < self.NOW
    assert tr.governor_closing_mps == 0.0

  def test_stale_closing_hold_does_not_release_while_lead_is_braking(self):
    tr = self._track(raw_vrel=-0.1, raw_alead=-0.5)
    tr.governor_hold_until_t = self.NOW + 1.0
    tr.governor_closing_mps = 1.8

    active = tr._update_closing_governor(
      self.NOW, raw_drel=40.0, raw_vrel=-0.1, raw_alead=-0.5, cfg=self._cfg(closing_governor_margin_mps=0.35),
    )

    assert active
    assert tr.governor_closing_mps == pytest.approx(1.8)

  def test_weak_nonbraking_position_slope_cannot_spend_position_trust(self):
    # Rlog/soup-to-nuts shelf signature: position says ~2 m/s closing while
    # the windowed raw velocity stream says only 0.4 and raw aLead is calm.
    # The governor may arm, but it must not fabricate another 1.5 m/s closure
    # from position noise alone.
    tr = self._track(vrel_state=-0.2, drel_state=60.0, slope=-2.0, raw_vrel=-0.4)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    active = tr._update_closing_governor(
      self.NOW, raw_drel=60.0, raw_vrel=-0.4, raw_alead=0.0, cfg=cfg,
    )

    assert active
    tr.governor_active = active
    assert tr.governor_closing_mps == pytest.approx(0.4)
    assert tr.get_RadarState(cfg)["vRel"] == pytest.approx(-0.4)

  def test_captured_raw_x_collapse_position_ttc_cannot_spend_full_trust(self):
    # 23:16:51.876 capture fingerprint: raw x collapsed with a 7.16 m/s
    # position slope and 5.99 s position TTC, while raw/windowed vRel showed
    # only 0.46 m/s closure and raw aLead remained calm. Position TTC is made
    # from the same x collapse, so it is not independent corroboration.
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.02)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    active = tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=0.02, cfg=cfg,
    )

    assert active
    assert tr.governor_closing_mps == pytest.approx(0.46)

  def test_isolated_current_raw_alead_does_not_spend_full_position_trust(self):
    # A single raw-aLead threshold crossing is not debounced corroboration.
    # The corpus contains seven isolated crossings, so this frame receives
    # only bounded magnitude-based bridge authority, never the full +1.5.
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.02)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    active = tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=-0.5, cfg=cfg,
    )

    assert active
    assert tr.governor_closing_mps == pytest.approx(0.76)
    assert tr.governor_closing_mps < 0.46 + cfg.closing_governor_pos_trust_excess_mps

  def test_unconfirmed_current_raw_alead_authority_is_capped(self):
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.02)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    active = tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=-1.0, cfg=cfg,
    )

    assert active
    assert tr.governor_closing_mps == pytest.approx(0.87)

  def test_unconfirmed_current_raw_alead_authority_has_rollback_sentinel(self):
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.02)
    cfg = self._cfg(
      closing_governor_margin_mps=0.75,
      closing_governor_unconfirmed_alead_trust_mps=0.0,
    )

    active = tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=-1.0, cfg=cfg,
    )

    assert active
    assert tr.governor_closing_mps == pytest.approx(0.46)

  def test_sustained_raw_alead_preserves_full_trust_after_one_frame_confirmation(self):
    # Paired safety twin: the same x-collapse/weak-vRel fingerprint with two
    # consecutive braking samples (50 ms at model cadence) gets full authority.
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.02)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    assert tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=-0.5, cfg=cfg,
    )
    assert tr.governor_closing_mps == pytest.approx(0.76)
    active = tr._update_closing_governor(
      self.NOW + 0.05, raw_drel=42.9 - 7.16 * 0.05,
      raw_vrel=-0.46, raw_alead=-0.5, cfg=cfg,
    )

    assert active
    expected = min(7.16, 0.46 + cfg.closing_governor_pos_trust_excess_mps)
    assert tr.governor_closing_mps == pytest.approx(expected)

  def test_two_mild_raw_alead_samples_do_not_spend_full_position_trust(self):
    # 2026-07-15 20:38:27 road fingerprint: two mild aLead samples crossed the
    # opening-recovery veto (-0.20) but not the sustained-braking onset (-0.35).
    # They must keep the threat latch conservative without turning an opening
    # raw-position artifact into the full +1.5 m/s published closing clamp.
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.0)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    assert tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=-0.30, cfg=cfg,
    )
    active = tr._update_closing_governor(
      self.NOW + 0.05, raw_drel=42.9 - 7.16 * 0.05,
      raw_vrel=-0.46, raw_alead=-0.24, cfg=cfg,
    )

    assert active
    assert tr.governor_threat_corroborated
    assert tr.governor_closing_mps < 0.46 + cfg.closing_governor_pos_trust_excess_mps
    assert 0.46 <= tr.governor_closing_mps <= 0.56 + 1e-9

  def test_accel_onset_sentinel_disables_accel_derived_full_position_trust(self):
    tr = self._track(vrel_state=-0.35, drel_state=42.9, slope=-7.16,
                     raw_vrel=-0.46, raw_alead=0.0)
    cfg = self._cfg(
      closing_governor_margin_mps=0.75,
      closing_governor_accel_onset_mps2=100.0,
    )

    assert tr._update_closing_governor(
      self.NOW, raw_drel=42.9, raw_vrel=-0.46, raw_alead=-1.0, cfg=cfg,
    )
    active = tr._update_closing_governor(
      self.NOW + 0.05, raw_drel=42.9 - 7.16 * 0.05,
      raw_vrel=-0.46, raw_alead=-1.0, cfg=cfg,
    )

    assert active
    assert tr.governor_threat_corroborated
    assert tr.governor_closing_mps < 0.46 + cfg.closing_governor_pos_trust_excess_mps

  def test_windowed_closure_margin_has_no_full_position_trust_cliff(self):
    # The current >= 0.75 rule jumps from no excess to the full +1.5 m/s for
    # a 0.02 m/s evidence change. Authority must instead grow continuously.
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    outputs = []
    for raw_vrel in (-0.74, -0.76):
      tr = self._track(vrel_state=-0.2, drel_state=60.0, slope=-4.0,
                       raw_vrel=raw_vrel, raw_alead=0.0)
      assert tr._update_closing_governor(
        self.NOW, raw_drel=60.0, raw_vrel=raw_vrel, raw_alead=0.0, cfg=cfg,
      )
      outputs.append(tr.governor_closing_mps)

    assert outputs == pytest.approx([0.74, 0.77])
    assert outputs[1] - outputs[0] < 0.05

  def test_short_raw_velocity_ttc_preserves_full_position_trust(self):
    # Current raw vRel is independent of the position slope. A genuinely short
    # raw-velocity TTC must retain the full safety allowance even before the
    # window mean catches up.
    tr = self._track(vrel_state=-0.2, drel_state=40.0, slope=-6.0,
                     raw_vrel=-0.4, raw_alead=0.0)
    cfg = self._cfg(closing_governor_margin_mps=2.0)

    active = tr._update_closing_governor(
      self.NOW, raw_drel=40.0, raw_vrel=-8.0, raw_alead=0.0, cfg=cfg,
    )

    assert active
    samples = list(tr.closing_evidence)
    windowed_closing = -(sum(s[2] for s in samples) / len(samples))
    expected = min(6.0, windowed_closing + cfg.closing_governor_pos_trust_excess_mps)
    assert tr.governor_closing_mps == pytest.approx(expected)

  @pytest.mark.parametrize("raw_vrel,raw_alead,expected", [(-0.8, 0.0, 0.85), (-0.4, -0.5, 1.9)])
  def test_velocity_trust_is_continuous_while_lead_decel_preserves_full_trust(self, raw_vrel, raw_alead, expected):
    # Velocity evidence earns only continuous excess above MarginMps;
    # sustained braking aLead retains the full configured authority.
    tr = self._track(vrel_state=-0.2, drel_state=60.0, slope=-2.0,
                     raw_vrel=raw_vrel, raw_alead=raw_alead)
    cfg = self._cfg(closing_governor_margin_mps=0.75)

    active = tr._update_closing_governor(
      self.NOW, raw_drel=60.0, raw_vrel=raw_vrel, raw_alead=raw_alead, cfg=cfg,
    )

    assert active
    tr.governor_active = active
    assert tr.governor_closing_mps == pytest.approx(expected)
    assert tr.get_RadarState(cfg)["vRel"] == pytest.approx(-expected)

  def test_relax_stays_behind_position_evidence_by_trust_deficit(self):
    tr = self._track(slope=0.25)
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel == pytest.approx(-0.05)
    rs = tr.get_RadarState(self._cfg())
    assert rs["vRel"] == pytest.approx(-0.05)

  def test_relax_never_deepens_an_already_open_publish(self):
    tr = self._track(vrel_state=0.5, slope=0.7)
    tr._update_opening_governor(self.NOW, False, self._cfg())
    rs = tr.get_RadarState(self._cfg())
    assert rs["vRel"] == pytest.approx(0.5)  # max() semantics: never lowered

  def test_raw_closing_veto_blocks_relax(self):
    tr = self._track(raw_vrel=-1.5)
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel is None
    assert tr.get_RadarState(self._cfg())["vRel"] == pytest.approx(-2.0)

  def test_braking_lead_veto_blocks_relax(self):
    tr = self._track(raw_alead=-0.5)
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel is None

  def test_cd9_latch_and_hold_veto_relax(self):
    tr = self._track()
    tr._update_opening_governor(self.NOW, True, self._cfg())
    assert tr.opening_relax_vrel is None
    tr.governor_hold_until_t = self.NOW + 0.5
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel is None

  def test_short_published_ttc_vetoes_relax(self):
    tr = self._track(drel_state=8.0)  # closing 2.0 -> TTC 4 s < 6 s gate
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel is None

  def test_slope_below_min_opening_blocks_relax(self):
    tr = self._track(slope=0.1)
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel is None

  def test_sparse_window_blocks_relax(self):
    tr = self._track(n_samples=4)
    tr._update_opening_governor(self.NOW, False, self._cfg())
    assert tr.opening_relax_vrel is None

  def test_trust_deficit_sentinel_disables_governor(self):
    tr = self._track()
    tr._update_opening_governor(self.NOW, False, self._cfg(opening_governor_trust_deficit_mps=99.0))
    assert tr.opening_relax_vrel is None

  def test_update_recomputes_relax_every_frame(self):
    tr = self._track(raw_vrel=-1.5)  # veto condition present in evidence
    tr.opening_relax_vrel = 0.0  # stale relax from a prior frame
    cfg = self._cfg()
    tr.update({"dRel": 40.0, "yRel": 0.0, "vRel": -1.5, "aLeadK": 0.0,
               "modelProb": 0.9, "dPath": 0.0, "vLat": 0.0}, self.NOW + 0.05, 20.0, cfg, 0)
    assert tr.opening_relax_vrel is None

  def test_end_to_end_phantom_run_is_relaxed_via_update(self):
    # Real closing approach, then the gap physically opens while raw vRel stays
    # mildly pessimistic (the 21:01:51 trace shape). Published vRel must be
    # lifted to parity instead of riding ~-1 for seconds.
    cfg = self._cfg()
    tr = ModelLeadTrack.from_lead_dict(
      7, {"dRel": 46.0, "yRel": 0.0, "vRel": -2.0, "vLead": 18.0,
          "aLeadK": 0.0, "modelProb": 0.9}, 0.0, 0)
    now = 0.0
    d = 46.0
    for _ in range(12):  # closing phase
      now += 0.05
      d -= 2.0 * 0.05
      tr.update({"dRel": d, "yRel": 0.0, "vRel": -2.0, "aLeadK": 0.0,
                 "modelProb": 0.9, "dPath": 0.0, "vLat": 0.0}, now, 20.0, cfg, 0)
    rs = None
    for _ in range(16):  # opening phase with pessimistic raw vRel
      now += 0.05
      d += 0.7 * 0.05
      rs = tr.update({"dRel": d, "yRel": 0.0, "vRel": -0.3, "aLeadK": 0.0,
                      "modelProb": 0.9, "dPath": 0.0, "vLat": 0.0}, now, 20.0, cfg, 0)
    assert tr.opening_relax_vrel is not None
    assert rs["vRel"] > -0.05
