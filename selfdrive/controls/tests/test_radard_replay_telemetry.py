import pytest

from cereal import car, custom, log, messaging
from openpilot.selfdrive.controls.radard import ModelLeadTrack, RadarD


class _SubMaster(dict):
  def __init__(self, log_mono_time) -> None:
    model = messaging.new_message("modelV2").modelV2
    car_state = messaging.new_message("carState").carState
    live_tracks = messaging.new_message("liveTracks").liveTracks
    super().__init__({"modelV2": model, "carState": car_state, "liveTracks": live_tracks})
    self.logMonoTime = log_mono_time
    self.seen = {"modelV2": True}
    self.recv_frame = {"carState": 1}

  def all_checks(self, service_list=None) -> bool:
    return True


class _PubMaster:
  def __init__(self) -> None:
    self.message = None

  def send(self, service: str, msg) -> None:
    assert service == "radarState"
    self.message = msg.as_reader()


def test_radard_publishes_versioned_exact_input_clocks() -> None:
  # Keep max(clock) below RadarD's one-second live-tune refresh boundary; this
  # test exercises only update/publication telemetry.
  clocks = {
    "modelV2": 100_000_001,
    "carState": 100_000_002,
    "liveTracks": 100_000_003,
  }
  sm = _SubMaster(clocks)
  rd = RadarD(car.CarParams.new_message(), custom.CarParamsSP.new_message())
  rd.update(sm, sm["liveTracks"])
  pm = _PubMaster()

  rd.publish(pm)

  assert pm.message is not None
  replay = pm.message.radarState.replayInputs
  assert replay.valid
  assert replay.version == 1
  assert replay.modelV2MonoTimeNs == clocks["modelV2"]
  assert replay.carStateMonoTimeNs == clocks["carState"]
  assert replay.liveTracksMonoTimeNs == clocks["liveTracks"]
  assert not replay.leadOneGovernor.valid
  assert not replay.leadTwoGovernor.valid
  assert rd.current_time == max(clocks.values()) * 1e-9


def test_radard_replay_contract_records_exact_governor_state() -> None:
  rd = RadarD(car.CarParams.new_message(), custom.CarParamsSP.new_message())
  track = ModelLeadTrack.from_lead_dict(
    -1001,
    {"dRel": 40.0, "vRel": -0.4, "vLead": 19.6, "aLeadK": 0.0, "modelProb": 0.99},
    0.0,
    0,
  )
  track.governor_active = True
  track.governor_hold_until_t = 0.75
  track.governor_closing_mps = 0.6
  track.governor_reason = "calm_recovery_capped"
  track.governor_threat_corroborated = True
  track.governor_calm_recovery_mode = True
  track.governor_calm_recovery_applied = True
  track.governor_recovery_position_closing_mps = 0.38
  rd.model_lead_tracker.tracks[track.identifier] = track
  lead = {"status": True, "radarTrackId": track.identifier}

  state = log.RadarState.new_message()
  rd.radar_state = state
  rd.current_time = 0.1
  rd._populate_governor_replay_debug(state.replayInputs.leadOneGovernor, lead)

  debug = state.replayInputs.leadOneGovernor
  assert debug.valid
  assert debug.radarTrackId == track.identifier
  assert debug.active
  assert debug.closingMps == pytest.approx(0.6)
  assert debug.holdRemainingS == pytest.approx(0.65)
  assert debug.reason == "calm_recovery_capped"
  assert debug.threatCorroborated
  assert debug.calmRecoveryMode
  assert debug.calmRecoveryApplied
  assert debug.recoveryPositionClosingValid
  assert debug.recoveryPositionClosingMps == pytest.approx(0.38)


def test_recovery_provenance_round_trips_on_behavioral_lead_data() -> None:
  track = ModelLeadTrack.from_lead_dict(
    -1002,
    {"dRel": 40.0, "vRel": -0.4, "vLead": 19.6, "aLeadK": 0.0, "modelProb": 0.99},
    0.0,
    0,
  )
  track.governor_calm_recovery_mode = True
  state = log.RadarState.new_message()
  state.leadOne = track.get_RadarState()

  reader = state.as_reader()
  assert reader.leadOne.closingGovernorRecovery
  # Producers that do not set the new behavior field retain the schema default.
  assert not reader.leadTwo.closingGovernorRecovery
