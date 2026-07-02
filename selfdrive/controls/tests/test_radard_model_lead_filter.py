import dataclasses
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.radard import (
  KalmanParams,
  ModelLeadTrack,
  ModelLeadTracker,
  RADAR_TO_CAMERA,
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

  def test_opening_recovery_does_not_engage_above_speed_gate(self):
    cfg = _cfg()
    open_slew_step_m = float(cfg.model_lead_filter_open_slew_max_mps) * DT_FRAME_S
    # v_ego 12 > OpenRecoveryMaxEgoMps 8: recovery must never arm.
    track, rows, d = self._collapse_then_truth(cfg, v_ego=12.0, truth_frames=30)

    for _raw, _drel, step in rows:
      assert step <= open_slew_step_m + 1e-6
    assert d - track.dRel > 8.0  # still wrong-too-close: slew-only recovery

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
