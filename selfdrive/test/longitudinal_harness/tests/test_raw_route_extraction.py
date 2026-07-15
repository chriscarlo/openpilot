from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cereal import log, messaging
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.tools.lib.logreader import save_log
from selfdrive.test.longitudinal_harness.catalog import get_route_rows, open_catalog
from selfdrive.test.longitudinal_harness.closed_loop import run_harness
from selfdrive.test.longitudinal_harness.config import REPLAY_PARAM_DEFAULT_VALUES, resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.fidelity import NOT_EVALUATED, evaluate_fidelity
from selfdrive.test.longitudinal_harness.inputs import load_snapshot_bundle
from selfdrive.test.longitudinal_harness.route_extract import (
  detect_episode_candidates,
  extract_ev6_episodes,
  index_ev6_routes,
  load_route_scan,
  write_episode_bundle,
)


_DT_NS = 50_000_000
_FRAME_COUNT = 451
_EVENT_FRAME = 320
_EVENT_END_FRAME = 340
_EGO_SPEED_MPS = 25.0
_LEAD_GAP_M = 45.0
_TRACK_ID = 4242
_DISTRACTOR_EGO_SPEED_MPS = 41.0


def _make_car_params(*, radar_unavailable: bool = True):
  cp = CarInterface.get_non_essential_params(CAR.KIA_EV6)
  cp.carFingerprint = "KIA_EV6"
  cp.openpilotLongitudinalControl = True
  cp.radarUnavailable = radar_unavailable
  if len(cp.safetyConfigs):
    # Matches the existing synthetic EV6 catalog fixture and infers LKA/HDA2.
    cp.safetyConfigs[0].safetyParam = 21
  return cp


def _append_model(
  messages: list[Any],
  *,
  mono_time_ns: int,
  raw_v_rel_mps: float,
  lead_gap_m: float,
  marker_accel_mps2: float,
) -> None:
  model = messaging.new_message("modelV2")
  model.logMonoTime = mono_time_ns
  model.valid = True

  position = log.XYZTData.new_message()
  position.x = [0.0, 50.0, 100.0]
  position.y = [0.0, 0.0, 0.0]
  position.z = [0.0, 0.0, 0.0]
  position.t = [0.0, 0.5, 1.0]
  velocity = log.XYZTData.new_message()
  velocity.x = [_EGO_SPEED_MPS, _EGO_SPEED_MPS, _EGO_SPEED_MPS]
  velocity.y = [0.0, 0.0, 0.0]
  velocity.z = [0.0, 0.0, 0.0]
  velocity.t = [0.0, 0.5, 1.0]
  acceleration = log.XYZTData.new_message()
  acceleration.x = [0.0, 0.0, 0.0]
  acceleration.y = [0.0, 0.0, 0.0]
  acceleration.z = [0.0, 0.0, 0.0]
  acceleration.t = [0.0, 0.5, 1.0]
  model.modelV2.position = position
  model.modelV2.velocity = velocity
  model.modelV2.acceleration = acceleration
  model.modelV2.action.desiredAcceleration = marker_accel_mps2
  model.modelV2.meta.disengagePredictions.t = [0.0, 1.0]
  model.modelV2.meta.disengagePredictions.gasPressProbs = [0.0, 0.0]

  leads = model.modelV2.init("leadsV3", 3)
  for slot, lead in enumerate(leads):
    lead.t = [0.0, 0.5]
    lead.x = [lead_gap_m + RADAR_TO_CAMERA, lead_gap_m + RADAR_TO_CAMERA]
    lead.y = [0.0, 0.0]
    lead.v = [_EGO_SPEED_MPS + raw_v_rel_mps, _EGO_SPEED_MPS + raw_v_rel_mps]
    lead.a = [0.0, 0.0]
    lead.prob = 0.97 if slot == 0 else 0.0
    lead.probTime = 0.05
  messages.append(model.as_reader())


def _append_published_lead(radar_state, *, published_v_rel_mps: float) -> None:
  lead = radar_state.radarState.leadOne
  lead.status = True
  lead.dRel = _LEAD_GAP_M
  lead.vRel = published_v_rel_mps
  lead.vLead = _EGO_SPEED_MPS + published_v_rel_mps
  lead.vLeadK = _EGO_SPEED_MPS + published_v_rel_mps
  lead.aLeadK = 0.0
  lead.modelProb = 0.97
  lead.yRel = 0.0
  lead.radar = False
  lead.radarTrackId = _TRACK_ID
  radar_state.radarState.leadTwo.status = False


def _write_raw_route(
  route_root: Path,
  *,
  genuine_closure: bool,
  frame_count: int = _FRAME_COUNT,
  missing_car_state_reference_at: int | None = None,
  zero_service_timestamps_at: int | None = None,
  radar_unavailable: bool = True,
  v1_contracts: bool = False,
  radar_contract_model_mismatch_at: int | None = None,
  planner_missing_controls_at: int | None = None,
  live_tracks_not_received_at: int | None = None,
  object_hazard_active_at: int | None = None,
  param_change_at: int | None = None,
  duplicate_car_state_at: int | None = None,
  conflicting_car_state_duplicate: bool = False,
  duplicate_controls_state_at: int | None = None,
  conflicting_controls_state_duplicate: bool = False,
  event_frame: int = _EVENT_FRAME,
  planner_radar_lag_frames: int = 0,
) -> dict[str, int]:
  segment_dir = route_root / "0"
  segment_dir.mkdir(parents=True, exist_ok=True)
  rlog_path = segment_dir / "rlog.zst"
  messages: list[Any] = []

  init_data = messaging.new_message("initData")
  init_data.logMonoTime = 900_000_000
  init_data.valid = True
  captured_params = {
    **REPLAY_PARAM_DEFAULT_VALUES,
    "Longitudinal.LiveTune.ModelLeadFilterTauS": "2.80",
    "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.40",
    "Longitudinal.LiveTune.LeadProbEnter": "0.60",
    "VisionTurnSpeedControl": "0",
    "RTIEnabled": "0",
    "WeatherAwareControlEnabled": "0",
  }
  entries = init_data.initData.params.init("entries", len(captured_params))
  for entry, (key, value) in zip(entries, sorted(captured_params.items()), strict=True):
    entry.key = key
    entry.value = value.encode()
  messages.append(init_data.as_reader())

  car_params = messaging.new_message("carParams")
  car_params.logMonoTime = 950_000_000
  car_params.valid = True
  car_params.carParams = _make_car_params(radar_unavailable=radar_unavailable)
  messages.append(car_params.as_reader())

  car_control_sp = messaging.new_message("carControlSP")
  car_control_sp.logMonoTime = 975_000_000
  car_control_sp.valid = True
  car_control_sp.carControlSP.init("params", 1)
  car_control_sp.carControlSP.params[0].key = "HyundaiLongitudinalTuning"
  car_control_sp.carControlSP.params[0].value = "0"
  messages.append(car_control_sp.as_reader())

  expected_event_model_mono_time = 0
  expected_event_radar_mono_time = 0
  for frame_idx in range(frame_count):
    frame_base = 1_000_000_000 + frame_idx * _DT_NS
    incident = event_frame <= frame_idx < event_frame + (_EVENT_END_FRAME - _EVENT_FRAME)
    if genuine_closure:
      raw_v_rel_mps = -3.0
    else:
      # Warm RadarD with a sustained closing estimate, then recover the exact
      # raw model at the incident while the published track remains pessimistic.
      raw_v_rel_mps = 0.0 if frame_idx >= event_frame else -3.0
    published_v_rel_mps = -3.0 if (frame_idx < event_frame or incident) else 0.0
    planner_accel_mps2 = -1.0 if incident else 0.2

    car_state = messaging.new_message("carState")
    car_state.logMonoTime = frame_base
    car_state.valid = True
    car_state.carState.vEgo = _EGO_SPEED_MPS
    car_state.carState.aEgo = 0.0
    car_state.carState.vCruise = 108.0
    messages.append(car_state.as_reader())
    if frame_idx == duplicate_car_state_at:
      duplicate_car_state = messaging.new_message("carState")
      duplicate_car_state.logMonoTime = int(car_state.logMonoTime)
      duplicate_car_state.valid = True
      duplicate_car_state.carState.vEgo = (
        _DISTRACTOR_EGO_SPEED_MPS if conflicting_car_state_duplicate else _EGO_SPEED_MPS
      )
      duplicate_car_state.carState.aEgo = 0.0
      duplicate_car_state.carState.vCruise = 108.0
      messages.append(duplicate_car_state.as_reader())

    controls_state = messaging.new_message("controlsState")
    controls_state.logMonoTime = frame_base + 2_000_000
    controls_state.valid = True
    controls_state.controlsState.longControlState = "pid"
    controls_state.controlsState.forceDecel = False
    messages.append(controls_state.as_reader())
    if frame_idx == duplicate_controls_state_at:
      duplicate_controls = messaging.new_message("controlsState")
      duplicate_controls.logMonoTime = int(controls_state.logMonoTime)
      duplicate_controls.valid = True
      duplicate_controls.controlsState.longControlState = "pid"
      duplicate_controls.controlsState.forceDecel = conflicting_controls_state_duplicate
      messages.append(duplicate_controls.as_reader())

    selfdrive_state = messaging.new_message("selfdriveState")
    selfdrive_state.logMonoTime = frame_base + 4_000_000
    selfdrive_state.valid = True
    selfdrive_state.selfdriveState.experimentalMode = False
    selfdrive_state.selfdriveState.personality = "standard"
    messages.append(selfdrive_state.as_reader())

    car_control = messaging.new_message("carControl")
    car_control.logMonoTime = frame_base + 6_000_000
    car_control.valid = True
    car_control.carControl.longActive = True
    car_control.carControl.orientationNED = [0.0, 0.01, 0.0]
    messages.append(car_control.as_reader())

    live_tracks = messaging.new_message("liveTracks")
    live_tracks.logMonoTime = frame_base + 8_000_000
    live_tracks.valid = True
    messages.append(live_tracks.as_reader())

    intended_model_time = frame_base + 10_000_000
    _append_model(
      messages,
      mono_time_ns=intended_model_time,
      raw_v_rel_mps=raw_v_rel_mps,
      lead_gap_m=_LEAD_GAP_M,
      marker_accel_mps2=-0.125,
    )
    # This newer model is deliberately closer in log order/time. A latest-frame
    # association would select it; mdMonoTime must select the intended model.
    _append_model(
      messages,
      mono_time_ns=frame_base + 12_000_000,
      raw_v_rel_mps=-12.0,
      lead_gap_m=120.0,
      marker_accel_mps2=1.75,
    )

    # A newer carState is deliberately closer in log order/time. The extractor
    # must use radarState.carStateMonoTime, not this latest state.
    distractor_car_state = messaging.new_message("carState")
    distractor_car_state.logMonoTime = frame_base + 13_000_000
    distractor_car_state.valid = True
    distractor_car_state.carState.vEgo = _DISTRACTOR_EGO_SPEED_MPS
    distractor_car_state.carState.aEgo = 3.0
    distractor_car_state.carState.vCruise = 144.0
    messages.append(distractor_car_state.as_reader())

    # Logger order alone is insufficient: this track publication occurs before
    # radarState is logged, but after the model/carState clocks RadarD snapped.
    # Exact extraction must retain the earlier <= input-clock publication.
    too_new_live_tracks = messaging.new_message("liveTracks")
    too_new_live_tracks.logMonoTime = frame_base + 14_000_000
    too_new_live_tracks.valid = True
    messages.append(too_new_live_tracks.as_reader())

    radar_state = messaging.new_message("radarState")
    radar_state.logMonoTime = frame_base + 15_000_000
    radar_state.valid = True
    radar_state.radarState.mdMonoTime = 0 if frame_idx == zero_service_timestamps_at else intended_model_time
    radar_state.radarState.carStateMonoTime = (
      0 if frame_idx == zero_service_timestamps_at else
      frame_base - 123 if frame_idx == missing_car_state_reference_at else frame_base
    )
    if v1_contracts:
      replay_inputs = radar_state.radarState.replayInputs
      replay_inputs.valid = True
      replay_inputs.version = 1
      replay_inputs.modelV2MonoTimeNs = (
        intended_model_time + 1 if frame_idx == radar_contract_model_mismatch_at else intended_model_time
      )
      replay_inputs.carStateMonoTimeNs = frame_base
      replay_inputs.liveTracksMonoTimeNs = (
        0 if frame_idx == live_tracks_not_received_at else frame_base + 8_000_000
      )
    _append_published_lead(radar_state, published_v_rel_mps=published_v_rel_mps)
    messages.append(radar_state.as_reader())

    if frame_idx == param_change_at:
      param_change = messaging.new_message("carControlSP")
      param_change.logMonoTime = frame_base + 16_000_000
      param_change.valid = True
      param_change.carControlSP.init("params", 1)
      param_change.carControlSP.params[0].key = "Longitudinal.LiveTune.ModelLeadFilterTauS"
      param_change.carControlSP.params[0].value = "1.75"
      messages.append(param_change.as_reader())

    # longitudinalPlan is intentionally published after radarState. The scan
    # must post-join it by modelMonoTime instead of retaining the previous plan.
    longitudinal_plan = messaging.new_message("longitudinalPlan")
    longitudinal_plan.logMonoTime = frame_base + 20_000_000
    longitudinal_plan.valid = True
    longitudinal_plan.longitudinalPlan.modelMonoTime = intended_model_time
    longitudinal_plan.longitudinalPlan.aTarget = planner_accel_mps2
    longitudinal_plan.longitudinalPlan.longitudinalPlanSource = "lead0" if incident else "cruise"
    # Deliberately conflicting deprecated values: v1 SP telemetry is the sole
    # exact writer contract.
    longitudinal_plan.longitudinalPlan.radarStateMonoTimeDEPRECATED = int(radar_state.logMonoTime) + 1
    longitudinal_plan.longitudinalPlan.vCruiseDEPRECATED = 31.0
    messages.append(longitudinal_plan.as_reader())

    longitudinal_plan_sp = messaging.new_message("longitudinalPlanSP")
    longitudinal_plan_sp.logMonoTime = frame_base + 21_000_000
    longitudinal_plan_sp.valid = True
    if v1_contracts:
      replay_inputs = longitudinal_plan_sp.longitudinalPlanSP.replayInputs
      replay_inputs.valid = True
      replay_inputs.version = 1
      replay_radar_frame = max(0, frame_idx - planner_radar_lag_frames)
      replay_inputs.radarStateMonoTimeNs = 1_000_000_000 + replay_radar_frame * _DT_NS + 15_000_000
      replay_inputs.effectiveCruiseMps = 23.25
      replay_inputs.longitudinalPlanMonoTimeNs = int(longitudinal_plan.logMonoTime)
      # Planner deliberately consumes the later distractor carState; RadarD
      # consumed frame_base. Exact replay must keep these domains separate.
      replay_inputs.carStateMonoTimeNs = frame_base + 13_000_000
      replay_inputs.carControlMonoTimeNs = frame_base + 6_000_000
      replay_inputs.controlsStateMonoTimeNs = (
        frame_base + 2_000_001 if frame_idx == planner_missing_controls_at else frame_base + 2_000_000
      )
      replay_inputs.selfdriveStateMonoTimeNs = frame_base + 4_000_000
      replay_inputs.modelV2MonoTimeNs = intended_model_time
      replay_inputs.liveParametersMonoTimeNs = 0
      replay_inputs.liveMapDataSPMonoTimeNs = 0
      replay_inputs.carStateSPMonoTimeNs = 0
      replay_inputs.rtiStateSPMonoTimeNs = 0
      replay_inputs.objectHazardStateSPMonoTimeNs = 0
      replay_inputs.gpsLocationMonoTimeNs = 0
      replay_inputs.gpsLocationExternalMonoTimeNs = 0
    longitudinal_plan_sp.longitudinalPlanSP.objectHazardControl.active = (
      frame_idx == object_hazard_active_at
    )
    messages.append(longitudinal_plan_sp.as_reader())

    if frame_idx == event_frame:
      expected_event_model_mono_time = intended_model_time
      expected_event_radar_mono_time = int(radar_state.logMonoTime)

  save_log(str(rlog_path), messages)
  return {
    "event_model_mono_time": expected_event_model_mono_time,
    "event_radar_mono_time": expected_event_radar_mono_time,
    "event_car_state_mono_time": 1_000_000_000 + event_frame * _DT_NS,
    "event_live_tracks_mono_time": 1_000_000_000 + event_frame * _DT_NS + 8_000_000,
    "event_planner_car_state_mono_time": 1_000_000_000 + event_frame * _DT_NS + 13_000_000,
  }


def test_raw_route_extraction_uses_exact_associations_and_replays_bundle(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  expected = _write_raw_route(log_root / "false_closing_route", genuine_closure=False)
  _write_raw_route(log_root / "genuine_closure_route", genuine_closure=True)
  db_path = tmp_path / "catalog.sqlite3"
  bundle_root = tmp_path / "snapshots"

  conn = open_catalog(db_path)
  try:
    indexed = index_ev6_routes(conn, roots=[log_root])
    assert {entry["routeKey"] for entry in indexed} == {"false_closing_route", "genuine_closure_route"}

    route_rows = get_route_rows(conn, route_keys=["false_closing_route", "genuine_closure_route"])
    scans = {str(row["route_key"]): load_route_scan(conn, row) for row in route_rows}
    false_scan = scans["false_closing_route"]
    genuine_scan = scans["genuine_closure_route"]

    incident_frame = false_scan.frames[_EVENT_FRAME]
    assert incident_frame.log_mono_time == expected["event_radar_mono_time"]
    assert incident_frame.v_ego_mps == pytest.approx(_EGO_SPEED_MPS)
    assert incident_frame.a_ego_mps2 == pytest.approx(0.0)
    assert incident_frame.model_v2_log_mono_time_ns == expected["event_model_mono_time"]
    assert incident_frame.car_state_log_mono_time_ns == expected["event_car_state_mono_time"]
    assert incident_frame.live_tracks_log_mono_time_ns == expected["event_live_tracks_mono_time"]
    assert incident_frame.raw_model is not None
    assert incident_frame.raw_model["logMonoTimeNs"] == expected["event_model_mono_time"]
    assert len(incident_frame.raw_model["leadsV3"]) == 3
    assert incident_frame.raw_model["action"]["desiredAcceleration"] == pytest.approx(-0.125)
    assert incident_frame.raw_lead_one is not None
    assert incident_frame.raw_lead_one.v_rel_mps == pytest.approx(0.0)
    assert incident_frame.raw_lead_one.d_rel_m == pytest.approx(_LEAD_GAP_M)
    # The plan arrived after this radar frame and is only correct after the exact
    # modelMonoTime post-join. The previous frame's plan is +0.2/cruise.
    assert incident_frame.planner_accel_mps2 == pytest.approx(-1.0)
    assert incident_frame.planner_source == "lead0"
    assert incident_frame.planner_context_status == "legacy_derived"
    assert incident_frame.planner_inputs["carState"]["vEgoMps"] == pytest.approx(_EGO_SPEED_MPS)
    assert incident_frame.planner_inputs["modelV2"]["rawModel"]["logMonoTimeNs"] == expected["event_model_mono_time"]
    assert incident_frame.planner_service_log_mono_time_ns == {
      "carState": expected["event_car_state_mono_time"],
      "controlsState": expected["event_car_state_mono_time"] + 2_000_000,
      "selfdriveState": expected["event_car_state_mono_time"] + 4_000_000,
      "carControl": expected["event_car_state_mono_time"] + 6_000_000,
      "modelV2": expected["event_model_mono_time"],
    }
    assert all(
      incident_frame.planner_service_association_provenance[service]["status"] == "inferred"
      for service in ("carState", "controlsState", "carControl", "selfdriveState", "modelV2")
    )
    assert incident_frame.radard_service_association_status == "inferred"
    assert incident_frame.radard_service_association_provenance["modelV2"]["status"] == "exact"
    assert incident_frame.radard_service_association_provenance["carState"]["status"] == "exact"
    assert incident_frame.radard_service_association_provenance["liveTracks"]["status"] == "inferred"
    assert incident_frame.radard_service_association_provenance["liveTracks"]["emptyPayloadValid"] is True
    assert incident_frame.radard_gate_eligible is False

    false_candidates = [
      candidate for candidate in detect_episode_candidates(false_scan)
      if candidate.episode_type == "false_closing"
    ]
    genuine_candidates = [
      candidate for candidate in detect_episode_candidates(genuine_scan)
      if candidate.episode_type == "false_closing"
    ]
    assert len(false_candidates) == 1
    # The detector opens a 10-frame evidence window and accepts once three
    # future frames qualify, so the candidate anchor precedes raw recovery by
    # seven frames. The route includes enough earlier frames for a full 15 s.
    assert false_candidates[0].event_t_s == pytest.approx(15.65, abs=0.051)
    assert false_candidates[0].metrics["publishedClosingExcessMps"] >= 2.9
    assert genuine_candidates == []

    partial_frames = list(false_candidates[0].frames)
    partial_frames[len(partial_frames) // 2] = replace(partial_frames[len(partial_frames) // 2], raw_model=None)
    partial_candidate = replace(false_candidates[0], frames=partial_frames)
    with pytest.raises(ValueError, match="partial raw-model coverage"):
      write_episode_bundle(false_scan, partial_candidate, tmp_path / "partial_snapshots")

    partial_clock_frames = list(false_candidates[0].frames)
    partial_clock_frames[len(partial_clock_frames) // 2] = replace(
      partial_clock_frames[len(partial_clock_frames) // 2],
      car_state_log_mono_time_ns=None,
    )
    partial_clock_candidate = replace(false_candidates[0], frames=partial_clock_frames)
    with pytest.raises(ValueError, match="partial required RadarD service-clock coverage"):
      write_episode_bundle(false_scan, partial_clock_candidate, tmp_path / "partial_clock_snapshots")

    recorded = extract_ev6_episodes(
      conn,
      route_keys=["false_closing_route"],
      bundle_root=bundle_root,
    )
  finally:
    conn.close()

  false_entries = [entry for entry in recorded if entry["episodeType"] == "false_closing"]
  assert len(false_entries) == 1
  bundle = load_snapshot_bundle(false_entries[0]["bundlePath"])

  assert bundle.vehicle["perceptionFilter"] == "radard"
  assert bundle.vehicle["egoReplayMode"] == "recorded"
  assert bundle.vehicle["rawModelReplay"] is True
  assert bundle.vehicle["warmupS"] == pytest.approx(15.65, abs=0.051)
  assert bundle.vehicle["radardDependencyWarmupS"] == pytest.approx(15.0)
  assert bundle.vehicle["radardGateEligible"] is False
  assert bundle.vehicle["radardServiceAssociationStatus"] == "inferred"
  assert bundle.vehicle["radardServiceAssociationProvenance"]["representativeFrame"]["liveTracks"]["status"] == "inferred"
  assert bundle.vehicle["radardGateEligibleFrameCount"] == 0
  assert bundle.vehicle["radardServiceAssociationStatusCounts"]["inferred"] == len(bundle.timeline)
  assert bundle.vehicle["paramSource"] == "initData+carControlSP"
  assert bundle.vehicle["paramManifest"]["complete"] is True
  assert bundle.vehicle["paramManifest"]["missingKeys"] == []
  assert bundle.params["Longitudinal.LiveTune.ModelLeadFilterTauS"] == "2.80"
  assert bundle.params["Longitudinal.LiveTune.ModelLeadFilterVRelTauS"] == "0.40"
  assert bundle.params["HyundaiLongitudinalTuning"] == "0"
  assert bundle.timeline
  assert all(step.raw_model is not None and len(step.raw_model["leadsV3"]) == 3 for step in bundle.timeline)
  assert all(step.recorded_v_ego_mps is not None and step.recorded_a_ego_mps2 is not None for step in bundle.timeline)
  assert all(step.recorded_model_v2_log_mono_time_ns is not None for step in bundle.timeline)
  assert all(step.recorded_car_state_log_mono_time_ns is not None for step in bundle.timeline)
  assert all(step.recorded_live_tracks_log_mono_time_ns is not None for step in bundle.timeline)

  event_step = next(step for step in bundle.timeline if step.event == "false_closing_start")
  assert event_step.t_s == pytest.approx(15.65, abs=0.051)
  assert bundle.timeline[0].replay_warmup_status == "warmup"
  assert bundle.timeline[0].replay_reference["plannerRadarResolution"] == "unscorable"
  assert "advancing dependency state" in bundle.timeline[0].replay_reference["plannerRadarResolutionReason"]
  assert "radarState" not in bundle.timeline[0].replay_reference
  assert event_step.replay_warmup_status == "ready"
  assert event_step.radard_gate_eligible is False
  assert event_step.lead_one.exact_model_prob is True
  incident_step = next(
    step for step in bundle.timeline
    if step.raw_model is not None and step.raw_model["logMonoTimeNs"] == expected["event_model_mono_time"]
  )
  assert incident_step.t_s == pytest.approx(16.0, abs=0.051)
  assert incident_step.recorded_model_v2_log_mono_time_ns == expected["event_model_mono_time"]
  assert incident_step.recorded_car_state_log_mono_time_ns == expected["event_car_state_mono_time"]
  assert incident_step.recorded_live_tracks_log_mono_time_ns == expected["event_live_tracks_mono_time"]
  assert incident_step.replay_reference["serviceLogMonoTimeNs"] == {
    "modelV2": expected["event_model_mono_time"],
    "carState": expected["event_car_state_mono_time"],
    "liveTracks": expected["event_live_tracks_mono_time"],
  }
  assert incident_step.replay_reference["longitudinalPlan"]["aTargetMps2"] == pytest.approx(-1.0)
  assert incident_step.replay_reference["longitudinalPlan"]["source"] == "lead0"
  assert incident_step.planner_context_status == "legacy_derived"
  assert incident_step.recorded_planner_inputs["carState"]["vEgoMps"] == pytest.approx(_EGO_SPEED_MPS)
  assert incident_step.recorded_planner_service_log_mono_time_ns["modelV2"] == expected["event_model_mono_time"]
  assert incident_step.replay_reference["radardServiceAssociationStatus"] == "inferred"
  assert incident_step.replay_reference["radardGateEligible"] is False
  assert "radarState" not in incident_step.replay_reference
  assert "radarStateDiagnostic" in incident_step.replay_reference

  vehicle = resolve_ev6_vehicle_config(
    topology=str(bundle.vehicle["topology"]),
    controller_mode="auto",
    tune_source="snapshot",
    snapshot_vehicle=bundle.vehicle,
    snapshot_params=bundle.params,
    livetune_snapshot=None,
  )
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name=bundle.name,
    steps=bundle.timeline,
    initial_speed_mps=bundle.initial_speed_mps,
    initial_accel_mps2=bundle.initial_accel_mps2,
    noise_profile="off",
    perception_filter="auto",
    ego_replay_mode="auto",
  )

  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["egoReplayMode"] == "recorded"
  assert result.trace
  replay_event = next(row for row in result.trace if row["event"] == "false_closing_start")
  assert replay_event["recorded_input_replay"] is True
  assert replay_event["planner_input_v_ego_mps"] == pytest.approx(_EGO_SPEED_MPS)
  replay_incident = next(
    row for row in result.trace
    if row["replay_reference"].get("logMonoTimeNs") == expected["event_radar_mono_time"]
  )
  assert replay_incident["lead_one_raw_v_rel_mps"] == pytest.approx(0.0)
  assert replay_incident["radard_model_v2_log_mono_time_ns"] == expected["event_model_mono_time"]
  assert replay_incident["radard_car_state_log_mono_time_ns"] == expected["event_car_state_mono_time"]
  assert replay_incident["radard_live_tracks_log_mono_time_ns"] == expected["event_live_tracks_mono_time"]
  assert replay_incident["radard_service_association_status"] == "inferred"
  assert replay_incident["radard_gate_eligible"] is False
  assert replay_incident["planner_fidelity_scorable"] is False
  fidelity = evaluate_fidelity(result.trace)
  assert fidelity["radar"]["status"] == NOT_EVALUATED
  assert fidelity["planner"]["status"] == NOT_EVALUATED


def test_raw_route_extraction_rejects_missing_exact_car_state_join(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "missing_car_state_route",
    genuine_closure=False,
    frame_count=1,
    missing_car_state_reference_at=0,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["missing_car_state_route"])[0]
    with pytest.warns(RuntimeWarning, match="Dropped 1/1 radarState frames"):
      scan = load_route_scan(conn, route_row)
    assert scan.frames == []
    assert scan.service_join_diagnostics["droppedMissingCarStateJoinCount"] == 1
    assert scan.service_join_diagnostics["missingCarStateJoinExamples"] == [{
      "radarStateLogMonoTimeNs": 1_015_000_000,
      "carStateLogMonoTimeNs": 999_999_877,
      "segment": 0,
    }]
    with pytest.raises(ValueError, match="exact carState is missing"):
      load_route_scan(conn, route_row, strict_service_joins=True)
  finally:
    conn.close()


def test_zero_service_timestamps_remain_inferred_not_exact(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "zero_timestamp_route",
    genuine_closure=False,
    frame_count=1,
    zero_service_timestamps_at=0,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["zero_timestamp_route"])[0]
    scan = load_route_scan(conn, route_row)
  finally:
    conn.close()

  assert len(scan.frames) == 1
  frame = scan.frames[0]
  assert frame.radard_service_association_status == "inferred"
  assert frame.radard_service_association_provenance["modelV2"]["status"] == "inferred"
  assert "zero mdMonoTime" in frame.radard_service_association_provenance["modelV2"]["reason"]
  assert frame.radard_service_association_provenance["carState"]["status"] == "inferred"
  assert "zero carStateMonoTime" in frame.radard_service_association_provenance["carState"]["reason"]
  assert frame.radard_gate_eligible is False


def test_v1_contracts_join_exact_inputs_and_keep_planner_car_state_separate(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  expected = _write_raw_route(
    log_root / "v1_route",
    genuine_closure=False,
    frame_count=_EVENT_FRAME + 1,
    v1_contracts=True,
    live_tracks_not_received_at=_EVENT_FRAME - 3,
    radar_contract_model_mismatch_at=_EVENT_FRAME - 2,
    planner_missing_controls_at=_EVENT_FRAME - 1,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["v1_route"])[0]
    scan = load_route_scan(conn, route_row)
  finally:
    conn.close()

  explicit_zero = scan.frames[_EVENT_FRAME - 3]
  assert explicit_zero.live_tracks_log_mono_time_ns == 0
  assert explicit_zero.radard_service_association_provenance["liveTracks"]["status"] == "exact"
  assert "not yet received" in explicit_zero.radard_service_association_provenance["liveTracks"]["reason"]
  assert explicit_zero.radard_gate_eligible is True

  mismatch = scan.frames[_EVENT_FRAME - 2]
  assert mismatch.radard_service_association_status == "mismatch"
  assert mismatch.radard_service_association_provenance["modelV2"]["status"] == "mismatch"
  assert mismatch.radard_gate_eligible is False

  missing_planner_target = scan.frames[_EVENT_FRAME - 1]
  assert missing_planner_target.planner_context_status == "unscorable"
  assert missing_planner_target.planner_service_association_provenance["controlsState"]["status"] == "missing"
  assert "controlsState target" in missing_planner_target.planner_context_reason

  exact = scan.frames[_EVENT_FRAME]
  assert exact.radard_service_association_status == "exact"
  assert exact.radard_service_association_provenance["contract"] == {
    "status": "exact",
    "valid": True,
    "version": 1,
    "reason": "RadarState.replayInputs v1",
  }
  assert exact.radard_gate_eligible is True
  assert exact.planner_radar_resolution == "exact"
  assert exact.planner_context_status == "exact"
  assert exact.recorded_effective_cruise_status == "exact"
  assert exact.recorded_effective_cruise_mps == pytest.approx(23.25)
  assert exact.v_ego_mps == pytest.approx(_EGO_SPEED_MPS)
  assert exact.car_state_log_mono_time_ns == expected["event_car_state_mono_time"]
  assert exact.planner_service_log_mono_time_ns["carState"] == expected["event_planner_car_state_mono_time"]
  assert exact.planner_inputs["carState"]["vEgoMps"] == pytest.approx(_DISTRACTOR_EGO_SPEED_MPS)
  assert exact.planner_inputs["carState"]["aEgoMps2"] == pytest.approx(3.0)
  assert all(
    exact.planner_service_association_provenance[service]["status"] == "exact"
    for service in ("carState", "controlsState", "carControl", "selfdriveState", "modelV2", "radarState")
  )
  assert exact.planner_service_association_provenance["params"]["status"] == "exact"


def test_bundle_seeds_warmup_predecessor_and_covers_strict_boundary(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  event_frame = 700
  _write_raw_route(
    log_root / "v1_warmup_route",
    genuine_closure=False,
    frame_count=event_frame + 101,
    v1_contracts=True,
    event_frame=event_frame,
    planner_radar_lag_frames=1,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["v1_warmup_route"])[0]
    scan = load_route_scan(conn, route_row)
    candidate = next(
      candidate for candidate in detect_episode_candidates(scan)
      if candidate.episode_type == "false_closing"
    )
    bundle_path = write_episode_bundle(scan, candidate, tmp_path / "snapshots")
  finally:
    conn.close()

  bundle = load_snapshot_bundle(bundle_path)
  assert bundle.vehicle["dependencyHistoryS"] >= 15.0
  assert bundle.timeline[0].planner_radar_resolution == "missing"
  assert bundle.timeline[0].planner_radar_state_log_mono_time_ns is None
  assert "scheduler_seed" in (bundle.timeline[0].note or "")

  radar_publish_times = {
    step.recorded_radar_state_log_mono_time_ns
    for step in bundle.timeline
    if step.recorded_radar_state_log_mono_time_ns is not None
  }
  assert all(
    step.planner_radar_state_log_mono_time_ns in radar_publish_times
    for step in bundle.timeline
    if step.planner_radar_resolution in ("exact", "legacy_timing_unique")
  )

  evaluation_start_rel = float(bundle.vehicle["evaluationTStartS"]) - float(bundle.vehicle["tStartS"])
  evaluation_steps = [step for step in bundle.timeline if step.t_s + 1e-9 >= evaluation_start_rel]
  assert evaluation_steps
  assert all(step.replay_warmup_status == "ready" for step in evaluation_steps)
  assert all(step.radard_gate_eligible for step in evaluation_steps)

  vehicle = resolve_ev6_vehicle_config(
    topology=str(bundle.vehicle["topology"]),
    controller_mode="auto",
    tune_source="snapshot",
    snapshot_vehicle=bundle.vehicle,
    snapshot_params=bundle.params,
    livetune_snapshot=None,
  )
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name=bundle.name,
    steps=bundle.timeline,
    initial_speed_mps=bundle.initial_speed_mps,
    initial_accel_mps2=bundle.initial_accel_mps2,
    noise_profile="off",
    perception_filter="auto",
    ego_replay_mode="auto",
  )
  evaluation_trace = [row for row in result.trace if row["t_s"] + 1e-9 >= evaluation_start_rel]
  assert evaluation_trace
  assert all(row["planner_state_initialization_exact"] is False for row in evaluation_trace)
  assert all(row["planner_fidelity_scorable"] is False for row in evaluation_trace)
  assert all(
    row["planner_state_initialization_provenance"]["status"] == "missing"
    for row in evaluation_trace
  )


def test_identical_route_wide_core_input_duplicates_keep_v1_join_exact(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "identical_duplicate_route",
    genuine_closure=False,
    frame_count=1,
    v1_contracts=True,
    duplicate_controls_state_at=0,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["identical_duplicate_route"])[0]
    scan = load_route_scan(conn, route_row)
  finally:
    conn.close()

  assert scan.frames[0].planner_context_status == "exact"
  assert scan.frames[0].planner_service_association_provenance["controlsState"]["status"] == "exact"
  assert scan.service_join_diagnostics["coreInputIdenticalDuplicateCounts"] == {"controlsState": 1}
  assert scan.service_join_diagnostics["coreInputConflictingDuplicateClocks"] == {}


def test_conflicting_route_wide_core_input_duplicates_fail_closed(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "conflicting_duplicate_route",
    genuine_closure=False,
    frame_count=1,
    v1_contracts=True,
    duplicate_controls_state_at=0,
    conflicting_controls_state_duplicate=True,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["conflicting_duplicate_route"])[0]
    scan = load_route_scan(conn, route_row)
  finally:
    conn.close()

  frame = scan.frames[0]
  assert frame.radard_service_association_status == "exact"
  assert frame.planner_context_status == "unscorable"
  assert frame.planner_service_association_provenance["controlsState"]["status"] == "conflict"
  assert "conflicting duplicate payloads" in frame.planner_context_reason
  assert scan.service_join_diagnostics["coreInputConflictingDuplicateClocks"] == {
    "controlsState": [1_002_000_000],
  }


def test_conflicting_car_state_duplicate_is_not_counted_as_exact_join(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "conflicting_car_state_duplicate_route",
    genuine_closure=False,
    frame_count=1,
    v1_contracts=True,
    duplicate_car_state_at=0,
    conflicting_car_state_duplicate=True,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["conflicting_car_state_duplicate_route"])[0]
    scan = load_route_scan(conn, route_row)
  finally:
    conn.close()

  assert scan.frames[0].radard_service_association_status == "conflict"
  assert scan.frames[0].radard_service_association_provenance["carState"]["status"] == "conflict"
  assert scan.service_join_diagnostics["exactCarStateJoinCount"] == 0
  assert scan.service_join_diagnostics["coreInputConflictingDuplicateClocks"] == {
    "carState": [1_000_000_000],
  }


def test_v1_planner_context_is_unscorable_near_recorded_param_change(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "param_change_route",
    genuine_closure=False,
    frame_count=1,
    v1_contracts=True,
    param_change_at=0,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["param_change_route"])[0]
    frame = load_route_scan(conn, route_row).frames[0]
  finally:
    conn.close()

  assert frame.planner_context_status == "unscorable"
  assert frame.planner_service_association_provenance["params"]["status"] == "unstable"
  assert "stable through dependency warmup" in frame.planner_context_reason


def test_radar_capable_capture_rejects_empty_live_tracks_substitution(tmp_path: Path) -> None:
  log_root = tmp_path / "logs"
  _write_raw_route(
    log_root / "radar_capable_route",
    genuine_closure=False,
    radar_unavailable=False,
  )
  conn = open_catalog(tmp_path / "catalog.sqlite3")
  try:
    index_ev6_routes(conn, roots=[log_root])
    route_row = get_route_rows(conn, route_keys=["radar_capable_route"])[0]
    scan = load_route_scan(conn, route_row)
    candidate = next(
      candidate for candidate in detect_episode_candidates(scan)
      if candidate.episode_type == "false_closing"
    )
    assert candidate.frames[0].radard_service_association_provenance["liveTracks"]["status"] == "missing"
    assert candidate.frames[0].radard_service_association_provenance["liveTracks"]["emptyPayloadValid"] is False
    with pytest.raises(ValueError, match="cannot substitute empty liveTracks"):
      write_episode_bundle(scan, candidate, tmp_path / "snapshots")
  finally:
    conn.close()
