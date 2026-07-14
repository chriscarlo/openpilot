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

  def __init__(self, values: dict[str, Any], *, log_mono_time_ns: dict[str, int], recv_frame: dict[str, int]):
    super().__init__(values)
    if values.keys() != log_mono_time_ns.keys() or values.keys() != recv_frame.keys():
      raise ValueError("RadarD SubMaster values, service clocks, and receive frames must cover identical services")
    self.seen = dict.fromkeys(values, True)
    self.valid = dict.fromkeys(values, True)
    self.alive = dict.fromkeys(values, True)
    self.updated = dict.fromkeys(values, True)
    self.logMonoTime = {service: int(log_mono_time_ns[service]) for service in values}
    self.recv_frame = {service: int(recv_frame[service]) for service in values}
    self.frame = max(self.recv_frame.values(), default=0)

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
    self._empty_tracks = messaging.new_message("liveTracks").liveTracks.as_reader()
    self._last_service_log_mono_time_ns: dict[str, int | None] = {
      "modelV2": None,
      "carState": None,
      "liveTracks": None,
    }
    self._service_recv_frame = dict.fromkeys(self._last_service_log_mono_time_ns, 0)
    self.last_service_log_mono_time_ns = dict.fromkeys(self._last_service_log_mono_time_ns, 0)

  def update(self, raw_radar_state, *, now_s: float, measured_speed_mps: float,
             raw_model: dict[str, Any] | None = None,
             model_v2_log_mono_time_ns: int | None = None,
             car_state_log_mono_time_ns: int | None = None,
             live_tracks_log_mono_time_ns: int | None = None):
    """Filter one frame; returns the radarState RadarD would publish on device."""
    model = build_model_message(raw_model) if raw_model is not None else self._build_model_msg(raw_radar_state, measured_speed_mps)
    car_state = messaging.new_message("carState")
    car_state.carState.vEgo = float(measured_speed_mps)
    service_log_mono_time_ns = self._resolve_service_log_mono_times(
      now_s=now_s,
      raw_model=raw_model,
      model_v2_log_mono_time_ns=model_v2_log_mono_time_ns,
      car_state_log_mono_time_ns=car_state_log_mono_time_ns,
      live_tracks_log_mono_time_ns=live_tracks_log_mono_time_ns,
    )
    for service, service_time_ns in service_log_mono_time_ns.items():
      if service_time_ns != self._last_service_log_mono_time_ns[service]:
        self._service_recv_frame[service] += 1
        self._last_service_log_mono_time_ns[service] = service_time_ns
    self.last_service_log_mono_time_ns = dict(service_log_mono_time_ns)
    sm = _RadardSubMaster(
      {
        "modelV2": model.modelV2,
        "carState": car_state.carState,
        "liveTracks": self._empty_tracks,
      },
      log_mono_time_ns=service_log_mono_time_ns,
      recv_frame=self._service_recv_frame,
    )
    self.radard.update(sm, self._empty_tracks)
    return self.radard.radar_state

  def tracker_debug(self, radar_state) -> dict[str, dict[str, Any]]:
    """Expose the real ModelLeadTracker governor state in harness traces."""
    tracks = self.radard.model_lead_tracker.tracks
    now_s = float(self.radard.current_time)
    debug: dict[str, dict[str, Any]] = {}
    for slot in ("leadOne", "leadTwo"):
      lead = getattr(radar_state, slot)
      track = tracks.get(int(lead.radarTrackId)) if lead.status else None
      wire_governor = getattr(radar_state.replayInputs, f"{slot}Governor")
      exact_governor = bool(wire_governor.valid)
      debug[slot] = {
        "track_id": None if track is None else int(track.identifier),
        "closing_governor_debug_exact": exact_governor,
        "closing_governor_active": bool(wire_governor.active) if exact_governor else bool(track is not None and track.governor_active),
        "closing_governor_hold_remaining_s": (
          float(wire_governor.holdRemainingS) if exact_governor
          else 0.0 if track is None else max(0.0, float(track.governor_hold_until_t) - float(now_s))
        ),
        "closing_governor_closing_mps": (
          float(wire_governor.closingMps) if exact_governor
          else 0.0 if track is None else float(track.governor_closing_mps)
        ),
        "closing_governor_reason": str(wire_governor.reason) if exact_governor else "inactive" if track is None else str(track.governor_reason),
        "closing_governor_threat_corroborated": (
          bool(wire_governor.threatCorroborated) if exact_governor
          else bool(track is not None and track.governor_threat_corroborated)
        ),
        "closing_governor_calm_recovery_mode": (
          bool(wire_governor.calmRecoveryMode) if exact_governor
          else bool(track is not None and track.governor_calm_recovery_mode)
        ),
        "closing_governor_calm_recovery_applied": (
          bool(wire_governor.calmRecoveryApplied) if exact_governor
          else bool(track is not None and track.governor_calm_recovery_applied)
        ),
        "closing_governor_recovery_position_closing_mps": (
          float(wire_governor.recoveryPositionClosingMps)
          if exact_governor and wire_governor.recoveryPositionClosingValid
          else None if exact_governor or track is None else track.governor_recovery_position_closing_mps
        ),
        "opening_relax_vrel_mps": None if track is None or track.opening_relax_vrel is None else float(track.opening_relax_vrel),
        "opening_relax_held": bool(track is not None and track.opening_relax_held),
        "opening_relax_hold_remaining_s": 0.0 if track is None else max(0.0, float(track.opening_relax_hold_until_t) - float(now_s)),
        "opening_last_raw_proof_age_s": (
          None if track is None or track.opening_last_raw_proof_t < 0.0
          else max(0.0, float(now_s) - float(track.opening_last_raw_proof_t))
        ),
        "opening_long_position_slope_mps": None if track is None else track.opening_long_position_slope_mps,
        "opening_bridge_position_safe": bool(track is not None and track.opening_bridge_position_safe),
      }
    return debug

  @staticmethod
  def _resolve_service_log_mono_times(
    *,
    now_s: float,
    raw_model: dict[str, Any] | None,
    model_v2_log_mono_time_ns: int | None,
    car_state_log_mono_time_ns: int | None,
    live_tracks_log_mono_time_ns: int | None,
  ) -> dict[str, int]:
    payload_model_time_ns = None
    if raw_model is not None and raw_model.get("logMonoTimeNs") is not None:
      payload_model_time_ns = int(raw_model["logMonoTimeNs"])

    if (
      model_v2_log_mono_time_ns is not None and payload_model_time_ns is not None and
      int(model_v2_log_mono_time_ns) != payload_model_time_ns
    ):
      raise ValueError(
        "recorded modelV2 service clock disagrees with rawModel.logMonoTimeNs "
        f"({int(model_v2_log_mono_time_ns)} != {payload_model_time_ns})"
      )

    recorded_timing = any(clock is not None for clock in (
      model_v2_log_mono_time_ns,
      car_state_log_mono_time_ns,
      live_tracks_log_mono_time_ns,
    )) or payload_model_time_ns is not None
    if not recorded_timing:
      synthetic_now_ns = int(float(now_s) * 1e9)
      return dict.fromkeys(("modelV2", "carState", "liveTracks"), synthetic_now_ns)

    resolved_model_time_ns = (
      int(model_v2_log_mono_time_ns)
      if model_v2_log_mono_time_ns is not None
      else payload_model_time_ns
    )
    if resolved_model_time_ns is None or resolved_model_time_ns <= 0:
      raise ValueError("recorded RadarD replay requires a positive modelV2 service logMonoTime")
    if car_state_log_mono_time_ns is None or int(car_state_log_mono_time_ns) <= 0:
      raise ValueError(
        "recorded RadarD replay requires recordedCarStateLogMonoTimeNs; regenerate the snapshot from its rlog"
      )
    if live_tracks_log_mono_time_ns is not None and int(live_tracks_log_mono_time_ns) <= 0:
      raise ValueError("recorded liveTracks service logMonoTime must be positive when provided")

    return {
      "modelV2": resolved_model_time_ns,
      "carState": int(car_state_log_mono_time_ns),
      # A radarless or partial log can lack liveTracks. Zero keeps it from
      # inventing a synthetic publish-time clock while preserving RadarD's max.
      "liveTracks": 0 if live_tracks_log_mono_time_ns is None else int(live_tracks_log_mono_time_ns),
    }

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


def build_model_message(payload: dict[str, Any]):
  """Rebuild the modelV2 subset preserved by a route snapshot."""
  model = messaging.new_message("modelV2")
  for section_name in ("position", "velocity", "acceleration"):
    section = log.XYZTData.new_message()
    for field_name, values in payload.get(section_name, {}).items():
      if field_name in section.schema.fields:
        setattr(section, field_name, [float(value) for value in values])
    setattr(model.modelV2, section_name, section)

  action = payload.get("action", {})
  model.modelV2.action.desiredCurvature = float(action.get("desiredCurvature", 0.0))
  model.modelV2.action.desiredAcceleration = float(action.get("desiredAcceleration", 0.0))
  model.modelV2.action.shouldStop = bool(action.get("shouldStop", False))

  disengage = payload.get("disengagePredictions", {})
  model.modelV2.meta.disengagePredictions.t = [float(value) for value in disengage.get("t", [])]
  model.modelV2.meta.disengagePredictions.gasPressProbs = [float(value) for value in disengage.get("gasPressProbs", [])]

  lead_payloads = list(payload.get("leadsV3", []))
  leads = model.modelV2.init("leadsV3", len(lead_payloads))
  for entry, lead_payload in zip(leads, lead_payloads, strict=True):
    for field_name, value in lead_payload.items():
      if field_name not in entry.schema.fields:
        continue
      if field_name in ("prob", "probTime"):
        setattr(entry, field_name, float(value))
      else:
        setattr(entry, field_name, [float(item) for item in value])
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
