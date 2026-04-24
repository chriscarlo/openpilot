import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.lead_kalman_filter import LeadKalmanFilter


class TestLeadKalmanFilter:
  def test_positive_vrel_without_opening_drel_does_not_drift_distance(self):
    filt = LeadKalmanFilter(r_drel=6.0, deadband_m=0.75, kalman_gain_max=0.25)
    filt.update(50.0, 0.0, 0.1)

    for _ in range(100):
      filt.update(50.0, 0.4, 0.05)

    assert filt.value == pytest.approx(50.0, abs=0.05)
    assert filt.last_debug["opening_vrel_suppressed"] is True
    assert filt.last_debug["vrel_input_mps"] == pytest.approx(0.0)

  def test_positive_vrel_is_used_once_drel_also_opens(self):
    filt = LeadKalmanFilter(r_drel=6.0, deadband_m=0.35, kalman_gain_max=0.25)
    filt.update(50.0, 0.0, 0.1)

    raw_drel = 50.0
    for _ in range(60):
      raw_drel += 0.5 * 0.05
      filt.update(raw_drel, 0.5, 0.05)

    assert filt.value is not None
    assert filt.value > 50.7
    assert filt.last_debug["opening_vrel_suppressed"] is False
    assert filt.last_debug["vrel_input_mps"] > 0.0
