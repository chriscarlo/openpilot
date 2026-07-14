from __future__ import annotations

from dataclasses import replace

import pytest

from cereal import messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.test.longitudinal_harness.closed_loop import VehiclePlantState, _build_submaster, run_harness
from selfdrive.test.longitudinal_harness.config import REPLAY_PARAM_DEFAULT_VALUES, resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import StepInput
from selfdrive.test.longitudinal_harness.route_extract import (
  _pair_longitudinal_plans_with_sp,
  CachedLongitudinalPlan,
  CachedLongitudinalPlanSP,
  EffectiveCruiseContext,
  PlannerRadarAssociation,
  RouteReplayIndex,
  derive_effective_cruise_context,
  resolve_planner_context,
  resolve_planner_radar_association,
)


def _plan(*, a_target: float = 0.0, source: str = "cruise", pointer: int = 0,
          plan_sp: CachedLongitudinalPlanSP | None = None) -> CachedLongitudinalPlan:
  return CachedLongitudinalPlan(
    log_mono_time_ns=120,
    model_mono_time_ns=100,
    solver_execution_time_s=5e-9,
    radar_state_mono_time_ns=pointer,
    v_cruise_deprecated_mps=0.0,
    a_target_mps2=a_target,
    source=source,
    plan_sp=plan_sp,
  )


def _vehicle():
  return resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    perception_filter="direct",
    livetune_snapshot=None,
  )


def _complete_params() -> dict[str, str]:
  return dict(REPLAY_PARAM_DEFAULT_VALUES)


def _complete_legacy_index(
  plan: CachedLongitudinalPlan,
  *,
  param_change_mono_times_ns: tuple[int, ...] = (),
) -> RouteReplayIndex:
  return RouteReplayIndex(
    car_state_by_mono_time={},
    planner_inputs_by_service={
      "carState": {90: {"marker": "car"}},
      "controlsState": {91: {"marker": "controls"}},
      "carControl": {92: {"marker": "control"}},
      "selfdriveState": {93: {"marker": "selfdrive"}},
      "modelV2": {100: {"marker": "model"}},
    },
    live_tracks_mono_times=(),
    radar_state_mono_times=(90,),
    plans_by_model_mono_time={100: plan},
    param_change_mono_times_ns=param_change_mono_times_ns,
    rti_zero_threats_proven=False,
    rti_state_count=0,
  )


def _plan_sp(*, pointer: int = 0, cap_mps: float = 0.0, valid: bool = True,
             version: int = 1, object_hazard_active: bool = False,
             clocks: dict[str, int] | None = None,
             plan_pointer: int = 120) -> CachedLongitudinalPlanSP:
  replay_clocks = dict(clocks or {})
  replay_clocks.setdefault("radarState", pointer)
  replay_clocks.setdefault("modelV2", 100)
  return CachedLongitudinalPlanSP(
    log_mono_time_ns=125,
    slc_active=False,
    slc_state="inactive",
    slc_speed_limit_mps=0.0,
    slc_speed_limit_offset_mps=0.0,
    vtsc_state="disabled",
    vtsc_velocity_mps=0.0,
    object_hazard_active=object_hazard_active,
    replay_inputs_valid=valid,
    replay_inputs_version=version,
    replay_effective_cruise_mps=cap_mps,
    replay_plan_log_mono_time_ns=plan_pointer,
    replay_input_clocks_ns=replay_clocks,
  )


def test_explicit_pointer_selects_previous_even_when_current_was_published_before_plan() -> None:
  # The deprecated pointer deliberately conflicts; only the v1 SP contract is authoritative.
  association = resolve_planner_radar_association(
    _plan(pointer=110, plan_sp=_plan_sp(pointer=90)),
    (90, 110),
  )
  assert association.resolution == "exact"
  assert association.target_log_mono_time_ns == 90
  assert association.candidate_log_mono_times_ns == (90,)


def test_deprecated_pointer_is_ambiguous_without_v1_marker() -> None:
  association = resolve_planner_radar_association(_plan(pointer=90), (90, 110))
  assert association.resolution == "legacy_ambiguous"
  assert association.target_log_mono_time_ns is None
  assert "legacy" in association.reason

  unsupported = resolve_planner_radar_association(
    _plan(pointer=90, plan_sp=_plan_sp(pointer=90, version=2)),
    (90, 110),
  )
  assert unsupported.resolution != "exact"


def test_duplicate_plan_sp_publications_fail_closed_instead_of_granting_v1_exactness() -> None:
  plan = _plan()
  first = replace(_plan_sp(pointer=90), log_mono_time_ns=125)
  second = replace(_plan_sp(pointer=110), log_mono_time_ns=126)

  paired = _pair_longitudinal_plans_with_sp([plan], [first, second])

  assert paired[0].plan_sp is None
  association = resolve_planner_radar_association(paired[0], (90, 110))
  assert association.resolution != "exact"


@pytest.mark.parametrize(
  ("plan_sp", "reason_fragment"),
  [
    (_plan_sp(pointer=90, plan_pointer=121), "longitudinalPlan pointer mismatch"),
    (_plan_sp(pointer=90, clocks={"modelV2": 101}), "modelV2 pointer mismatch"),
  ],
)
def test_v1_plan_sp_requires_direct_plan_and_model_pointers(
  plan_sp: CachedLongitudinalPlanSP,
  reason_fragment: str,
) -> None:
  plan = _plan(plan_sp=plan_sp)
  assert resolve_planner_radar_association(plan, (90,)).resolution != "exact"
  context = resolve_planner_context(
    plan=plan,
    replay_index=RouteReplayIndex(
      car_state_by_mono_time={},
      planner_inputs_by_service={},
      live_tracks_mono_times=(),
      radar_state_mono_times=(90,),
      plans_by_model_mono_time={100: plan},
      param_change_mono_times_ns=(),
      rti_zero_threats_proven=False,
      rti_state_count=0,
    ),
    radar_association=PlannerRadarAssociation(90, (90,), "legacy_timing_unique", "diagnostic"),
    effective_cruise=EffectiveCruiseContext(20.0, "legacy", "diagnostic", "legacy_derived", "diagnostic"),
    params_snapshot=_complete_params(),
  )
  assert context.status == "unscorable"
  assert reason_fragment in context.reason


def test_plan_sp_at_next_plan_clock_belongs_only_to_next_plan() -> None:
  first_plan = _plan()
  second_plan = replace(_plan(), log_mono_time_ns=130, model_mono_time_ns=110)
  second_sp = replace(
    _plan_sp(pointer=110, clocks={"modelV2": 110}, plan_pointer=130),
    log_mono_time_ns=130,
  )

  paired = _pair_longitudinal_plans_with_sp([first_plan, second_plan], [second_sp])

  assert paired[0].plan_sp is None
  assert paired[1].plan_sp == second_sp


def test_mixed_current_previous_current_radar_targets_replay_in_order() -> None:
  publish_times = (100, 200, 300)
  target_times = (100, 100, 300)
  steps = [
    StepInput(
      t_s=idx * DT_MDL,
      cruise_speed_mps=20.0,
      recorded_radar_state_log_mono_time_ns=publish_time,
      planner_radar_state_log_mono_time_ns=target_time,
      planner_radar_state_candidates_ns=[target_time],
      planner_radar_resolution="exact",
      planner_context_status="unscorable",
      planner_context_reason="synthetic scheduler-only test",
    )
    for idx, (publish_time, target_time) in enumerate(zip(publish_times, target_times, strict=True))
  ]

  result = run_harness(
    vehicle_config=_vehicle(),
    scenario_name="mixed_planner_radar_targets",
    steps=steps,
    initial_speed_mps=20.0,
    noise_profile="off",
  )

  ticks_per_step = round(DT_MDL / DT_CTRL)
  assert [result.trace[idx * ticks_per_step]["planner_radar_state_log_mono_time_ns"] for idx in range(3)] == list(target_times)
  assert all(result.trace[idx * ticks_per_step]["planner_scheduler_scorable"] for idx in range(3))


def test_closed_loop_rejects_old_exact_params_label_without_complete_manifest() -> None:
  publish_clock = 100
  step = StepInput(
    t_s=0.0,
    cruise_speed_mps=20.0,
    recorded_radar_state_log_mono_time_ns=publish_clock,
    planner_radar_state_log_mono_time_ns=publish_clock,
    planner_radar_state_candidates_ns=[publish_clock],
    planner_radar_resolution="exact",
    planner_context_status="exact",
    recorded_effective_cruise_mps=20.0,
    recorded_effective_cruise_status="exact",
    radard_service_association_status="exact",
    radard_service_association_provenance={
      "contract": {"status": "exact", "version": 1},
      "modelV2": {"status": "exact"},
      "carState": {"status": "exact"},
      "liveTracks": {"status": "exact", "emptyPayloadValid": True},
    },
    radard_gate_eligible=True,
    replay_warmup_status="ready",
    planner_service_association_provenance={
      "carState": {"status": "exact"},
      "controlsState": {"status": "exact"},
      "carControl": {"status": "exact"},
      "selfdriveState": {"status": "exact"},
      "modelV2": {"status": "exact"},
      "radarState": {"status": "exact", "clockNs": publish_clock},
      "params": {"status": "exact"},
    },
  )

  result = run_harness(
    vehicle_config=_vehicle(),
    scenario_name="old_params_manifest_claim",
    steps=[step],
    initial_speed_mps=20.0,
    noise_profile="off",
  )

  assert result.trace[0]["planner_scheduler_scorable"] is True
  assert result.trace[0]["planner_context_scorable"] is False
  assert result.trace[0]["planner_fidelity_scorable"] is False


def test_missing_exact_radar_target_fails_with_pre_roll_message() -> None:
  step = StepInput(
    t_s=0.0,
    cruise_speed_mps=20.0,
    recorded_radar_state_log_mono_time_ns=100,
    planner_radar_state_log_mono_time_ns=50,
    planner_radar_state_candidates_ns=[50],
    planner_radar_resolution="exact",
  )
  with pytest.raises(ValueError, match="missing required pre-roll"):
    run_harness(
      vehicle_config=_vehicle(),
      scenario_name="missing_planner_radar_target",
      steps=[step],
      initial_speed_mps=20.0,
      noise_profile="off",
    )


def test_legacy_ambiguity_does_not_depend_on_recorded_planner_output() -> None:
  first = resolve_planner_radar_association(_plan(a_target=-3.0, source="lead0"), (90, 110))
  second = resolve_planner_radar_association(_plan(a_target=1.0, source="cruise"), (90, 110))
  assert first == second
  assert first.resolution == "legacy_ambiguous"
  assert first.target_log_mono_time_ns is None
  assert first.candidate_log_mono_times_ns == (90, 110)

  boundary = resolve_planner_radar_association(_plan(), (110,))
  assert boundary.resolution == "missing"
  assert "predecessor" in boundary.reason


def test_force_decel_and_proven_slc_caps_are_legacy_derived() -> None:
  forced = derive_effective_cruise_context(
    plan=None,
    raw_cruise_mps=33.0,
    force_decel=True,
    long_active=True,
    gas_pressed=False,
    params={},
    rti_zero_threats_proven=False,
  )
  assert forced.status == "legacy_derived"
  assert forced.speed_mps == 0.0
  assert forced.limiter == "forceDecel"

  plan_sp = CachedLongitudinalPlanSP(
    log_mono_time_ns=125,
    slc_active=True,
    slc_state="active",
    slc_speed_limit_mps=17.881599,
    slc_speed_limit_offset_mps=5.36448,
    vtsc_state="disabled",
    vtsc_velocity_mps=30.0,
    object_hazard_active=False,
  )
  slc = derive_effective_cruise_context(
    plan=_plan(plan_sp=plan_sp),
    raw_cruise_mps=33.7222,
    force_decel=False,
    long_active=True,
    gas_pressed=False,
    params={
      "VisionTurnSpeedControl": "0",
      "RTIEnabled": "1",
      "WeatherAwareControlEnabled": "1",
      "WeatherSpeedReductionLight": "5",
      "WeatherSpeedReductionModerate": "8",
      "WeatherSpeedReductionHeavy": "10",
    },
    rti_zero_threats_proven=True,
  )
  assert slc.status == "legacy_derived"
  assert slc.speed_mps == pytest.approx(23.246079)
  assert slc.limiter == "speedLimitControl"
  assert "weather configured floor" in (slc.provenance or "")


def test_v1_writer_cap_wins_over_conflicting_deprecated_cap_including_zero() -> None:
  for cap_mps in (0.0, 7.5):
    plan = _plan(pointer=999, plan_sp=_plan_sp(pointer=90, cap_mps=cap_mps))
    plan = replace(plan, v_cruise_deprecated_mps=31.0)
    context = derive_effective_cruise_context(
      plan=plan,
      raw_cruise_mps=33.0,
      force_decel=False,
      long_active=True,
      gas_pressed=False,
      params={},
      rti_zero_threats_proven=False,
    )
    assert context.status == "exact"
    assert context.speed_mps == pytest.approx(cap_mps)


def test_v1_exact_context_requires_complete_captured_parameter_manifest() -> None:
  clocks = {
    "carState": 1,
    "controlsState": 2,
    "carControl": 3,
    "selfdriveState": 4,
    "modelV2": 100,
    "radarState": 90,
  }
  plan = _plan(plan_sp=_plan_sp(pointer=90, cap_mps=12.0, clocks=clocks))
  index = RouteReplayIndex(
    car_state_by_mono_time={},
    planner_inputs_by_service={
      service: {clock: {"marker": service}}
      for service, clock in clocks.items()
      if service != "radarState"
    },
    live_tracks_mono_times=(),
    radar_state_mono_times=(90,),
    plans_by_model_mono_time={100: plan},
    param_change_mono_times_ns=(),
    rti_zero_threats_proven=False,
    rti_state_count=0,
  )
  params = _complete_params()
  missing_key = "Longitudinal.LiveTune.ModelLeadFilterVRelTauS"
  params.pop(missing_key)

  context = resolve_planner_context(
    plan=plan,
    replay_index=index,
    radar_association=PlannerRadarAssociation(90, (90,), "exact", "v1"),
    effective_cruise=EffectiveCruiseContext(12.0, "writerContractV1", "v1", "exact", "v1"),
    params_snapshot=params,
  )

  assert context.status == "unscorable"
  assert context.service_provenance["params"]["status"] == "incomplete"
  assert context.service_provenance["params"]["complete"] is False
  assert context.service_provenance["params"]["missingKeys"] == [missing_key]
  assert "parameter manifest is incomplete" in context.reason


def test_active_object_hazard_keeps_v1_planner_context_unscorable() -> None:
  clocks = {
    "carState": 1,
    "controlsState": 2,
    "carControl": 3,
    "selfdriveState": 4,
    "modelV2": 100,
    "radarState": 90,
    "objectHazardStateSP": 5,
  }
  plan = _plan(plan_sp=_plan_sp(
    pointer=90,
    cap_mps=12.0,
    object_hazard_active=True,
    clocks=clocks,
  ))
  index = RouteReplayIndex(
    car_state_by_mono_time={},
    planner_inputs_by_service={
      service: {clock: {"marker": service}}
      for service, clock in clocks.items()
      if service not in ("radarState", "objectHazardStateSP")
    },
    live_tracks_mono_times=(),
    radar_state_mono_times=(90,),
    plans_by_model_mono_time={100: plan},
    param_change_mono_times_ns=(),
    rti_zero_threats_proven=False,
    rti_state_count=0,
  )
  context = resolve_planner_context(
    plan=plan,
    replay_index=index,
    radar_association=PlannerRadarAssociation(90, (90,), "exact", "v1"),
    effective_cruise=EffectiveCruiseContext(12.0, "writerContractV1", "v1", "exact", "v1"),
    params_snapshot=_complete_params(),
  )
  assert context.status == "unscorable"
  assert "object hazard" in context.reason


def test_legacy_planner_context_joins_recorded_inputs_without_claiming_exactness() -> None:
  plan = _plan(plan_sp=_plan_sp(valid=False))
  snapshots = {
    "carState": {
      105: {"marker": "after-model"},
      90: {"marker": "car-before-model"},
    },
    "controlsState": {91: {"marker": "controls"}},
    "carControl": {92: {"marker": "control"}},
    "selfdriveState": {93: {"marker": "selfdrive"}},
    "modelV2": {
      99: {"marker": "wrong-model"},
      100: {"marker": "trigger-model"},
    },
  }
  index = RouteReplayIndex(
    car_state_by_mono_time={},
    planner_inputs_by_service=snapshots,
    live_tracks_mono_times=(),
    radar_state_mono_times=(90, 110),
    plans_by_model_mono_time={100: plan},
    param_change_mono_times_ns=(),
    rti_zero_threats_proven=False,
    rti_state_count=0,
    planner_input_mono_times_by_service={
      service: tuple(sorted(service_snapshots))
      for service, service_snapshots in snapshots.items()
    },
  )

  context = resolve_planner_context(
    plan=plan,
    replay_index=index,
    radar_association=PlannerRadarAssociation(None, (90, 110), "legacy_ambiguous", "scheduler race"),
    effective_cruise=EffectiveCruiseContext(20.0, "legacy", "diagnostic", "legacy_derived", "diagnostic"),
    params_snapshot=_complete_params(),
  )

  assert context.status == "legacy_derived"
  assert context.inputs["carState"]["marker"] == "car-before-model"
  assert context.inputs["modelV2"]["marker"] == "trigger-model"
  assert context.service_log_mono_time_ns == {
    "carState": 90,
    "controlsState": 91,
    "carControl": 92,
    "selfdriveState": 93,
    "modelV2": 100,
  }
  assert context.service_provenance["radarState"]["status"] == "ambiguous"
  assert all(
    association.get("status") != "exact"
    for association in context.service_provenance.values()
    if isinstance(association, dict)
  )


def test_legacy_planner_context_is_unscorable_when_a_core_payload_is_missing() -> None:
  plan = _plan(plan_sp=_plan_sp(valid=False))
  index = RouteReplayIndex(
    car_state_by_mono_time={},
    planner_inputs_by_service={
      "carState": {90: {"marker": "car"}},
      "controlsState": {91: {"marker": "controls"}},
      "carControl": {92: {"marker": "control"}},
      "selfdriveState": {},
      "modelV2": {100: {"marker": "model"}},
    },
    live_tracks_mono_times=(),
    radar_state_mono_times=(90,),
    plans_by_model_mono_time={100: plan},
    param_change_mono_times_ns=(),
    rti_zero_threats_proven=False,
    rti_state_count=0,
  )

  context = resolve_planner_context(
    plan=plan,
    replay_index=index,
    radar_association=PlannerRadarAssociation(90, (90,), "legacy_timing_unique", "unique timing"),
    effective_cruise=EffectiveCruiseContext(20.0, "legacy", "diagnostic", "legacy_derived", "diagnostic"),
    params_snapshot=_complete_params(),
  )

  assert context.status == "unscorable"
  assert context.service_provenance["selfdriveState"]["status"] == "missing"
  assert context.service_provenance["radarState"]["status"] == "inferred"


def test_legacy_planner_context_rejects_nearby_parameter_changes() -> None:
  plan = _plan(plan_sp=_plan_sp(valid=False))
  context = resolve_planner_context(
    plan=plan,
    replay_index=_complete_legacy_index(plan, param_change_mono_times_ns=(119,)),
    radar_association=PlannerRadarAssociation(90, (90,), "legacy_timing_unique", "unique timing"),
    effective_cruise=EffectiveCruiseContext(20.0, "legacy", "diagnostic", "legacy_derived", "diagnostic"),
    params_snapshot=_complete_params(),
  )

  assert context.status == "unscorable"
  assert context.service_provenance["params"]["status"] == "unstable"
  assert "stable through dependency warmup" in context.reason


def test_legacy_object_hazard_reason_survives_context_resolution() -> None:
  plan = _plan(plan_sp=_plan_sp(valid=False, object_hazard_active=True))
  context = resolve_planner_context(
    plan=plan,
    replay_index=_complete_legacy_index(plan),
    radar_association=PlannerRadarAssociation(90, (90,), "legacy_timing_unique", "unique timing"),
    effective_cruise=EffectiveCruiseContext(
      None,
      None,
      None,
      "unscorable",
      "object hazard was active but its planner input state was not captured",
    ),
    params_snapshot=_complete_params(),
  )

  assert context.status == "unscorable"
  assert "object hazard" in context.reason


def test_unsupported_planner_replay_version_cannot_fall_back_to_legacy_join() -> None:
  plan = _plan(plan_sp=_plan_sp(version=2))
  context = resolve_planner_context(
    plan=plan,
    replay_index=_complete_legacy_index(plan),
    radar_association=PlannerRadarAssociation(90, (90,), "legacy_timing_unique", "unique timing"),
    effective_cruise=EffectiveCruiseContext(20.0, "legacy", "diagnostic", "legacy_derived", "diagnostic"),
    params_snapshot=_complete_params(),
  )

  assert context.status == "unscorable"
  assert context.inputs == {}
  assert context.service_log_mono_time_ns == {}
  assert context.service_provenance["contract"]["status"] == "unsupported"


def test_build_submaster_uses_planner_car_state_not_radard_car_state() -> None:
  step = StepInput(
    t_s=0.0,
    cruise_speed_mps=30.0,
    recorded_planner_inputs={
      "carState": {
        "vEgoMps": 41.0,
        "aEgoMps2": 3.0,
        "vCruiseKph": 144.0,
        "gasPressed": True,
        "standstill": False,
      },
      "controlsState": {"longControlState": int(LongCtrlState.pid), "forceDecel": True},
      "carControl": {"longActive": True, "orientationNED": [0.0, 0.02, 0.0]},
      "selfdriveState": {"enabled": True, "experimentalMode": True, "personality": 2},
    },
    recorded_planner_service_log_mono_time_ns={
      "carState": 11,
      "controlsState": 12,
      "carControl": 13,
      "selfdriveState": 14,
      "modelV2": 15,
      "radarState": 16,
    },
  )
  radard_state = VehiclePlantState(0.0, 0.0, 25.0, 0.0, 25.0, 0.0)
  radar = messaging.new_message("radarState").radarState
  sm = _build_submaster(step, radard_state, radar, LongCtrlState.off, False)

  assert sm["carState"].vEgo == pytest.approx(41.0)
  assert sm["carState"].aEgo == pytest.approx(3.0)
  assert sm["carState"].vCruise == pytest.approx(144.0)
  assert sm["controlsState"].forceDecel is True
  assert sm["carControl"].orientationNED[1] == pytest.approx(0.02)
  assert sm["selfdriveState"].experimentalMode is True
  assert {service: sm.logMonoTime[service] for service in step.recorded_planner_service_log_mono_time_ns} == (
    step.recorded_planner_service_log_mono_time_ns
  )


def test_recorded_cap_override_runs_real_overlay_and_preserves_raw_cruise() -> None:
  step = StepInput(
    t_s=0.0,
    cruise_speed_mps=20.0,
    recorded_effective_cruise_mps=7.5,
    recorded_effective_cruise_limiter="writerContract",
    recorded_effective_cruise_provenance="synthetic exact writer contract",
    planner_context_status="exact",
    planner_context_reason="synthetic exact writer contract",
  )
  result = run_harness(
    vehicle_config=_vehicle(),
    scenario_name="effective_cap_override",
    steps=[step],
    initial_speed_mps=20.0,
    noise_profile="off",
  )
  row = result.trace[0]
  assert row["planner_raw_cruise_mps"] == pytest.approx(20.0)
  assert row["planner_generated_effective_cruise_mps"] is not None
  assert row["planner_applied_effective_cruise_mps"] == pytest.approx(7.5)
  assert row["planner_effective_cruise_override_delta_mps"] == pytest.approx(
    7.5 - row["planner_generated_effective_cruise_mps"]
  )
