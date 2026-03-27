from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.radard import (
  RADAR_TO_CAMERA,
  add_path_relative_lead_metrics,
  get_path_relative_lead_metrics,
  get_path_y_rel,
)


def _model_path(xs, ys):
  return SimpleNamespace(position=SimpleNamespace(x=xs, y=ys))


def _lead_msg(ts, xs, ys):
  return SimpleNamespace(t=ts, x=xs, y=ys)


class TestRadardPathMetrics:
  def test_get_path_y_rel_interpolates_model_path_in_radar_frame(self):
    model_msg = _model_path(
      [0.0, 10.0, 20.0, 30.0],
      [0.0, -1.0, -2.0, -3.0],
    )

    y_rel = get_path_y_rel(model_msg, 20.0 - RADAR_TO_CAMERA)

    assert y_rel == pytest.approx(2.0)

  def test_path_relative_offset_is_near_zero_for_curve_following_lead(self):
    model_msg = _model_path(
      [0.0, 10.0, 20.0, 30.0],
      [0.0, -1.0, -2.0, -3.0],
    )
    lead_dict = {
      "status": True,
      "dRel": 20.0 - RADAR_TO_CAMERA,
      "yRel": 2.0,
    }

    d_path, v_lat = get_path_relative_lead_metrics(lead_dict, model_msg)

    assert d_path == pytest.approx(0.0, abs=1e-6)
    assert v_lat == pytest.approx(0.0)

  def test_path_relative_vlat_tracks_cutin_toward_ego_path(self):
    model_msg = _model_path(
      [0.0, 10.0, 20.0, 30.0],
      [0.0, 0.0, 0.0, 0.0],
    )
    lead_dict = {
      "status": True,
      "dRel": 20.0 - RADAR_TO_CAMERA,
      "yRel": 3.2,
    }
    lead_msg = _lead_msg(
      [0.0, 1.0],
      [20.0, 28.0],
      [-3.2, -2.2],
    )

    d_path, v_lat = get_path_relative_lead_metrics(lead_dict, model_msg, lead_msg)

    assert d_path == pytest.approx(3.2)
    assert v_lat == pytest.approx(-1.0)

  def test_add_path_relative_metrics_preserves_lead_dict_shape(self):
    model_msg = _model_path(
      [0.0, 10.0, 20.0, 30.0],
      [0.0, -1.0, -2.0, -3.0],
    )
    lead_dict = {
      "status": True,
      "dRel": 20.0 - RADAR_TO_CAMERA,
      "yRel": 2.0,
      "vRel": 0.0,
    }

    updated = add_path_relative_lead_metrics(lead_dict, model_msg)

    assert updated["dPath"] == pytest.approx(0.0, abs=1e-6)
    assert updated["vLat"] == pytest.approx(0.0)
    assert updated["vRel"] == pytest.approx(0.0)
