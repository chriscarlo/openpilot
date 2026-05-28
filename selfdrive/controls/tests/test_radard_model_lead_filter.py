import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.selfdrive.controls.radard import (
  KalmanParams,
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
