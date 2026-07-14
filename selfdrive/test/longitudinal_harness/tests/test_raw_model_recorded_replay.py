from __future__ import annotations

import pytest

from cereal import log, messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import (
  HarnessParams,
  LeadTrackState,
  NoiseStreams,
  VehiclePlantState,
  _build_lead,
  run_harness,
)
from selfdrive.test.longitudinal_harness.config import (
  NOISE_PROFILES,
  NoiseSeeds,
  resolve_ev6_vehicle_config,
)
from selfdrive.test.longitudinal_harness.inputs import (
  LeadDirective,
  StepInput,
  serialize_model_frame,
)
from selfdrive.test.longitudinal_harness.radard_stage import RadardPerceptionStage, build_model_message


def _raw_model_message():
  model = messaging.new_message("modelV2")

  for section_idx, section_name in enumerate(("position", "velocity", "acceleration"), start=1):
    section = log.XYZTData.new_message()
    section.x = [section_idx + 0.125, section_idx + 1.25, section_idx + 2.5]
    section.y = [-section_idx - 0.25, -section_idx - 0.5, -section_idx - 0.75]
    section.z = [0.01 * section_idx, 0.02 * section_idx, 0.03 * section_idx]
    section.t = [0.0, 0.5, 1.0]
    section.xStd = [0.1 * section_idx, 0.2 * section_idx, 0.3 * section_idx]
    section.yStd = [0.4 * section_idx, 0.5 * section_idx, 0.6 * section_idx]
    section.zStd = [0.7 * section_idx, 0.8 * section_idx, 0.9 * section_idx]
    setattr(model.modelV2, section_name, section)

  model.modelV2.action.desiredCurvature = 0.0125
  model.modelV2.action.desiredAcceleration = -0.875
  model.modelV2.action.shouldStop = True
  model.modelV2.meta.disengagePredictions.t = [0.0, 1.0, 2.0]
  model.modelV2.meta.disengagePredictions.gasPressProbs = [0.05, 0.25, 0.75]

  probabilities = (0.9875, 0.6125, 0.2375)
  leads = model.modelV2.init("leadsV3", len(probabilities))
  for idx, (lead, probability) in enumerate(zip(leads, probabilities, strict=True)):
    base = float(idx + 1)
    lead.prob = probability
    lead.probTime = 0.025 * base
    lead.t = [0.0, 0.5, 1.0]
    lead.x = [20.0 * base, 20.5 * base, 21.0 * base]
    lead.xStd = [0.5 * base, 0.6 * base, 0.7 * base]
    lead.y = [-0.2 * base, -0.1 * base, 0.0]
    lead.yStd = [0.1 * base, 0.2 * base, 0.3 * base]
    lead.v = [12.0 * base, 12.25 * base, 12.5 * base]
    lead.vStd = [0.4 * base, 0.5 * base, 0.6 * base]
    lead.a = [-0.5 * base, -0.4 * base, -0.3 * base]
    lead.aStd = [0.2 * base, 0.25 * base, 0.3 * base]

  return model


def _direct_passthrough_vehicle():
  return resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    perception_filter="direct",
    livetune_snapshot=None,
  )


def test_raw_model_roundtrip_preserves_all_three_leads_and_exact_probabilities() -> None:
  payload = serialize_model_frame(_raw_model_message().modelV2)
  rebuilt = build_model_message(payload)
  rebuilt_payload = serialize_model_frame(rebuilt.modelV2)

  assert rebuilt_payload == payload
  assert len(rebuilt.modelV2.leadsV3) == 3
  assert [lead.prob for lead in rebuilt.modelV2.leadsV3] == pytest.approx([0.9875, 0.6125, 0.2375])
  assert [lead.probTime for lead in rebuilt.modelV2.leadsV3] == pytest.approx([0.025, 0.05, 0.075])
  assert rebuilt.modelV2.action.desiredAcceleration == pytest.approx(-0.875)
  assert rebuilt.modelV2.action.shouldStop is True


def test_recorded_service_clocks_roundtrip_and_reach_radard_exactly() -> None:
  model_time_ns = 8_123_456_789
  car_state_time_ns = 8_120_000_111
  live_tracks_time_ns = 8_125_000_222
  raw_model = serialize_model_frame(_raw_model_message().modelV2)
  raw_model["logMonoTimeNs"] = model_time_ns
  step = StepInput(
    t_s=1.25,
    cruise_speed_mps=20.0,
    raw_model=raw_model,
    recorded_model_v2_log_mono_time_ns=model_time_ns,
    recorded_car_state_log_mono_time_ns=car_state_time_ns,
    recorded_live_tracks_log_mono_time_ns=live_tracks_time_ns,
    radard_service_association_status="exact",
    radard_service_association_provenance={
      "modelV2": {"status": "exact", "reason": "synthetic explicit pointer"},
      "carState": {"status": "exact", "reason": "synthetic explicit pointer"},
      "liveTracks": {"status": "exact", "reason": "synthetic explicit pointer", "emptyPayloadValid": True},
      "capture": {"radarUnavailable": True, "reason": "synthetic EV6 fixture"},
    },
    radard_gate_eligible=True,
    replay_warmup_status="ready",
    replay_warmup_reason="synthetic state pre-seeded",
    recorded_planner_inputs={
      "carState": {"vEgoMps": 31.0, "aEgoMps2": -0.25, "vCruiseKph": 120.0, "standstill": False},
    },
    recorded_planner_service_log_mono_time_ns={
      "modelV2": model_time_ns,
      "carState": car_state_time_ns + 1,
      "radarState": 8_130_000_333,
    },
    planner_service_association_provenance={
      "carState": {"status": "exact", "clockNs": car_state_time_ns + 1},
    },
  )

  rebuilt_step = StepInput.from_json(step.to_json())
  assert rebuilt_step.recorded_model_v2_log_mono_time_ns == model_time_ns
  assert rebuilt_step.recorded_car_state_log_mono_time_ns == car_state_time_ns
  assert rebuilt_step.recorded_live_tracks_log_mono_time_ns == live_tracks_time_ns
  assert rebuilt_step.radard_service_association_status == "exact"
  assert rebuilt_step.radard_service_association_provenance["liveTracks"]["status"] == "exact"
  assert rebuilt_step.radard_gate_eligible is True
  assert rebuilt_step.replay_warmup_status == "ready"
  assert rebuilt_step.recorded_planner_inputs["carState"]["vEgoMps"] == pytest.approx(31.0)
  assert rebuilt_step.recorded_planner_service_log_mono_time_ns["carState"] == car_state_time_ns + 1
  assert rebuilt_step.planner_service_association_provenance["carState"]["status"] == "exact"

  vehicle = _direct_passthrough_vehicle()
  stage = RadardPerceptionStage(vehicle.cp, vehicle.cp_sp, HarnessParams(vehicle.params))
  raw_radar_state = messaging.new_message("radarState").radarState
  radar_state = stage.update(
    raw_radar_state,
    now_s=step.t_s,
    measured_speed_mps=12.0,
    raw_model=rebuilt_step.raw_model,
    model_v2_log_mono_time_ns=rebuilt_step.recorded_model_v2_log_mono_time_ns,
    car_state_log_mono_time_ns=rebuilt_step.recorded_car_state_log_mono_time_ns,
    live_tracks_log_mono_time_ns=rebuilt_step.recorded_live_tracks_log_mono_time_ns,
  )

  assert int(radar_state.mdMonoTime) == model_time_ns
  assert int(radar_state.carStateMonoTime) == car_state_time_ns
  assert stage.last_service_log_mono_time_ns == {
    "modelV2": model_time_ns,
    "carState": car_state_time_ns,
    "liveTracks": live_tracks_time_ns,
  }
  assert stage.radard.current_time == pytest.approx(live_tracks_time_ns / 1e9)


def test_recorded_radard_timing_rejects_missing_or_conflicting_service_clock() -> None:
  vehicle = _direct_passthrough_vehicle()
  stage = RadardPerceptionStage(vehicle.cp, vehicle.cp_sp, HarnessParams(vehicle.params))
  raw_radar_state = messaging.new_message("radarState").radarState
  raw_model = serialize_model_frame(_raw_model_message().modelV2)
  raw_model["logMonoTimeNs"] = 2_000_000_000

  with pytest.raises(ValueError, match="requires recordedCarStateLogMonoTimeNs"):
    stage.update(
      raw_radar_state,
      now_s=2.1,
      measured_speed_mps=12.0,
      raw_model=raw_model,
    )

  with pytest.raises(ValueError, match="disagrees with rawModel"):
    stage.update(
      raw_radar_state,
      now_s=2.1,
      measured_speed_mps=12.0,
      raw_model=raw_model,
      model_v2_log_mono_time_ns=2_000_000_001,
      car_state_log_mono_time_ns=1_999_000_000,
    )


def test_exact_model_probability_bypasses_synthetic_acquisition_ramp() -> None:
  state = VehiclePlantState(
    time_s=0.0,
    true_distance_m=0.0,
    true_speed_mps=20.0,
    true_accel_mps2=0.0,
    measured_speed_mps=20.0,
    measured_accel_mps2=0.0,
  )
  exact, _ = _build_lead(
    "leadOne",
    LeadTrackState(),
    LeadDirective(
      status=True,
      v_lead_mps=20.0,
      model_prob_target=0.83,
      d_rel_override_m=40.0,
      acquisition_reset=True,
      exact_model_prob=True,
    ),
    state,
    NOISE_PROFILES["off"],
    DT_MDL,
    NoiseStreams.from_seeds(NoiseSeeds.from_base(10)),
    0.3,
  )
  ramped, _ = _build_lead(
    "leadOne",
    LeadTrackState(),
    LeadDirective(
      status=True,
      v_lead_mps=20.0,
      model_prob_target=0.83,
      d_rel_override_m=40.0,
      acquisition_reset=True,
    ),
    state,
    NOISE_PROFILES["off"],
    DT_MDL,
    NoiseStreams.from_seeds(NoiseSeeds.from_base(10)),
    0.3,
  )

  assert exact.modelProb == pytest.approx(0.83)
  assert ramped.modelProb == pytest.approx(DT_MDL / 0.25)


def test_recorded_ego_auto_mode_isolated_from_counterfactual_plant() -> None:
  recorded_speeds = (8.0, 8.2, 8.4)
  steps = [
    StepInput(
      t_s=idx * DT_MDL,
      cruise_speed_mps=15.0,
      recorded_v_ego_mps=speed,
      recorded_a_ego_mps2=-0.2,
      note="recorded road input must not close over the simulated plant",
    )
    for idx, speed in enumerate(recorded_speeds)
  ]

  recorded_result = run_harness(
    vehicle_config=_direct_passthrough_vehicle(),
    scenario_name="recorded_ego_isolation",
    steps=steps,
    initial_speed_mps=25.0,
    noise_profile="off",
    ego_replay_mode="auto",
  )
  plant_result = run_harness(
    vehicle_config=_direct_passthrough_vehicle(),
    scenario_name="recorded_ego_plant_override",
    steps=steps,
    initial_speed_mps=25.0,
    noise_profile="off",
    ego_replay_mode="plant",
  )

  ticks_per_step = round(DT_MDL / DT_CTRL)
  assert recorded_result.vehicle["egoReplayMode"] == "recorded"
  assert all(row["recorded_input_replay"] for row in recorded_result.trace)
  for idx, speed in enumerate(recorded_speeds):
    rows = recorded_result.trace[idx * ticks_per_step:(idx + 1) * ticks_per_step]
    assert rows
    assert all(row["planner_input_v_ego_mps"] == pytest.approx(speed) for row in rows)
    assert all(row["planner_input_a_ego_mps2"] == pytest.approx(-0.2) for row in rows)

  # The simulated vehicle starts at 25 m/s and remains a counterfactual output;
  # it cannot feed its state back into the 8 m/s recorded planner inputs.
  assert recorded_result.trace[0]["v_ego_true_mps"] > 20.0
  assert abs(recorded_result.trace[0]["v_ego_true_mps"] - recorded_result.trace[0]["planner_input_v_ego_mps"]) > 10.0

  assert plant_result.vehicle["egoReplayMode"] == "plant"
  assert not any(row["recorded_input_replay"] for row in plant_result.trace)
  assert plant_result.trace[0]["planner_input_v_ego_mps"] == pytest.approx(25.0)


def test_ego_replay_mode_validation_rejects_invalid_or_missing_recorded_inputs() -> None:
  legacy_step = StepInput(t_s=0.0, cruise_speed_mps=20.0)

  with pytest.raises(ValueError, match="unsupported ego_replay_mode"):
    run_harness(
      vehicle_config=_direct_passthrough_vehicle(),
      scenario_name="invalid_ego_mode",
      steps=[legacy_step],
      initial_speed_mps=20.0,
      noise_profile="off",
      ego_replay_mode="invalid",
    )

  with pytest.raises(ValueError, match="requires recorded ego fields"):
    run_harness(
      vehicle_config=_direct_passthrough_vehicle(),
      scenario_name="missing_recorded_ego",
      steps=[legacy_step],
      initial_speed_mps=20.0,
      noise_profile="off",
      ego_replay_mode="recorded",
    )

  with pytest.raises(ValueError, match="on every timeline frame"):
    run_harness(
      vehicle_config=_direct_passthrough_vehicle(),
      scenario_name="partial_recorded_ego",
      steps=[StepInput(t_s=0.0, cruise_speed_mps=20.0, recorded_v_ego_mps=19.5)],
      initial_speed_mps=20.0,
      noise_profile="off",
      ego_replay_mode="auto",
    )


def test_legacy_timeline_payload_keeps_original_defaults_and_runs_in_plant_mode() -> None:
  legacy_payload = {
    "t": 0.0,
    "cruiseSpeedMps": 20.0,
    "leadOne": {
      "status": True,
      "v_lead_mps": 18.0,
      "model_prob_target": 0.9,
      "d_rel_override_m": 45.0,
      "acquisition_reset": True,
    },
    "leadTwo": {},
    "note": "pre-raw-model snapshot schema",
  }
  step = StepInput.from_json(legacy_payload)

  assert step.raw_model is None
  assert step.recorded_v_ego_mps is None
  assert step.recorded_a_ego_mps2 is None
  assert step.param_updates == {}
  assert step.replay_reference == {}
  assert step.lead_one.exact_model_prob is False

  result = run_harness(
    vehicle_config=_direct_passthrough_vehicle(),
    scenario_name="legacy_timeline_compatibility",
    steps=[step],
    initial_speed_mps=20.0,
    noise_profile="off",
  )

  assert result.vehicle["egoReplayMode"] == "plant"
  assert result.vehicle["perceptionFilter"] == "direct"
  assert result.trace
  assert not any(row["recorded_input_replay"] for row in result.trace)


def test_explicit_param_override_survives_conflicting_recorded_update() -> None:
  protected_key = "Longitudinal.LiveTune.ModelLeadFilterTauS"
  replayed_key = "Longitudinal.LiveTune.ModelLeadFilterVRelTauS"
  vehicle = resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    perception_filter="direct",
    livetune_snapshot=None,
    param_overrides={protected_key: "1.23"},
  )
  step = StepInput(
    t_s=0.0,
    cruise_speed_mps=20.0,
    param_updates={
      protected_key: "9.99",
      replayed_key: "0.77",
    },
  )

  run_harness(
    vehicle_config=vehicle,
    scenario_name="recorded_param_update_precedence",
    steps=[step],
    initial_speed_mps=20.0,
    noise_profile="off",
  )

  assert protected_key in vehicle.param_override_keys
  assert vehicle.params[protected_key] == "1.23"
  assert vehicle.params[replayed_key] == "0.77"
