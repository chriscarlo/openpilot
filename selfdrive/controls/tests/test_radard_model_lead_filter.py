import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

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
