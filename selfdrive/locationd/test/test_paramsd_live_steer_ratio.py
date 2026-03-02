import numpy as np

from cereal import car, messaging
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.paramsd import retrieve_initial_vehicle_params


def test_live_steer_ratio_override_takes_precedence_over_cached_value():
  params = Params()
  CP = car.CarParams.new_message()
  CP.carFingerprint = "TEST_CAR"
  CP.steerRatio = 13.43

  msg = messaging.new_message("liveParameters")
  msg.liveParameters.steerRatio = 15.2
  msg.liveParameters.stiffnessFactor = 0.97
  msg.liveParameters.angleOffsetAverageDeg = -0.31

  override_sr = 13.42
  params.put("LiveParametersV2", msg.to_bytes())
  params.put("CarParamsPrevRoute", CP.to_bytes())
  params.put("LiveSteerRatio", override_sr)

  try:
    sr, sf, offset, _, base_sr = retrieve_initial_vehicle_params(params, CP, replay=True, debug=False)

    np.testing.assert_allclose(base_sr, override_sr)
    np.testing.assert_allclose(sr, override_sr)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
  finally:
    params.remove("LiveSteerRatio")
