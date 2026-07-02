from __future__ import annotations

from typing import Any

from cereal import log, messaging
from opendbc.car import structs
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA, RadarD

# Second trajectory sample offset used to encode lead vLat into leadsV3
# (radard derives vLat from the first future trajectory point).
_LEAD_TRAJECTORY_DT_S = 0.5


class _RadardSubMaster(dict):
  """Minimal SubMaster stand-in feeding RadarD.update (modelV2/carState/liveTracks)."""

  def __init__(self, values: dict[str, Any], *, now_ns: int, frame: int):
    super().__init__(values)
    self.seen = dict.fromkeys(values, True)
    self.valid = dict.fromkeys(values, True)
    self.alive = dict.fromkeys(values, True)
    self.updated = dict.fromkeys(values, True)
    self.logMonoTime = dict.fromkeys(values, int(now_ns))
    self.recv_frame = dict.fromkeys(values, int(frame))

  def all_checks(self, service_list: list[str] | None = None) -> bool:
    service_list = list(self.keys()) if service_list is None else service_list
    return all(self.valid.get(service, False) for service in service_list)


class RadardPerceptionStage:
  """Routes the harness's synthesized raw leads through the REAL radard pipeline.

  On the radar-less EV6 every lead is vision-derived: modelV2.leadsV3 -> radard's
  Schmitt prob latch (Longitudinal.LiveTune.LeadProbEnter/Exit) ->
  get_RadarState_from_vision -> ModelLeadTracker EMA/slew smoothing -> radarState.
  The stage runs RadarD itself (selfdrive/controls/radard.py) once per 20 Hz planner
  step, converting the raw fabricated LeadData back into leadsV3-shaped inputs so
  the planner receives leads with the same perception lag the car produces.
  """

  def __init__(self, cp: structs.CarParams, cp_sp: structs.CarParamsSP, harness_params) -> None:
    self.radard = RadarD(cp, cp_sp, float(getattr(cp, "radarDelay", 0.0) or 0.0))
    # Test seam: bind the harness param store in place of device Params and force
    # the first update to read the seeded thresholds (mirrors _bind_planner_params).
    self.radard._params = harness_params
    self.radard._last_prob_refresh_t = float("-inf")
    self.radard.model_lead_tracker.params = harness_params
    self._frame = 0
    self._empty_tracks = messaging.new_message("liveTracks").liveTracks.as_reader()

  def update(self, raw_radar_state, *, now_s: float, measured_speed_mps: float):
    """Filter one frame; returns the radarState RadarD would publish on device."""
    model = self._build_model_msg(raw_radar_state, measured_speed_mps)
    car_state = messaging.new_message("carState")
    car_state.carState.vEgo = float(measured_speed_mps)
    self._frame += 1
    sm = _RadardSubMaster(
      {
        "modelV2": model.modelV2,
        "carState": car_state.carState,
        "liveTracks": self._empty_tracks,
      },
      now_ns=int(float(now_s) * 1e9),
      frame=self._frame,
    )
    self.radard.update(sm, self._empty_tracks)
    return self.radard.radar_state

  def _build_model_msg(self, raw_radar_state, measured_speed_mps: float):
    model = messaging.new_message("modelV2")
    position = log.XYZTData.new_message()
    # Straight model path (y == 0): radard's dPath computation reduces to yRel,
    # matching the frame the harness's direct-mode leads are expressed in.
    position.x = [0.0, 250.0]
    position.y = [0.0, 0.0]
    velocity = log.XYZTData.new_message()
    velocity.x = [float(measured_speed_mps)]
    model.modelV2.position = position
    model.modelV2.velocity = velocity
    leads = model.modelV2.init("leadsV3", 2)
    for slot, raw_lead in enumerate((raw_radar_state.leadOne, raw_radar_state.leadTwo)):
      _fill_lead_v3(leads[slot], raw_lead, model_v_ego=float(measured_speed_mps))
    return model


def _fill_lead_v3(entry, raw_lead, *, model_v_ego: float) -> None:
  """Invert get_RadarState_from_vision: raw LeadData -> leadsV3 measurement."""
  if not raw_lead.status:
    entry.prob = 0.0
    entry.t = [0.0]
    entry.x = [RADAR_TO_CAMERA]
    entry.y = [0.0]
    entry.v = [0.0]
    entry.a = [0.0]
    return

  dt = _LEAD_TRAJECTORY_DT_S
  x0 = float(raw_lead.dRel) + RADAR_TO_CAMERA
  y0 = -float(raw_lead.yRel)
  v0 = model_v_ego + float(raw_lead.vRel)
  entry.prob = float(raw_lead.modelProb)
  entry.t = [0.0, dt]
  entry.x = [x0, x0]
  # With a straight path, v_lat = (future_d_path - d_path) / dt = -(y[1]-y[0])/dt.
  entry.y = [y0, y0 - float(raw_lead.vLat) * dt]
  entry.v = [v0, v0]
  entry.a = [float(raw_lead.aLeadK), float(raw_lead.aLeadK)]
