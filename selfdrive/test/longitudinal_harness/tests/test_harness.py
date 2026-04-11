from __future__ import annotations

from pathlib import Path

import pytest

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import run_harness
from selfdrive.test.longitudinal_harness.config import NoiseSeeds, resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import (
  build_synthetic_scenario,
  load_snapshot_bundle,
  write_snapshot_bundle,
)
from selfdrive.test.longitudinal_harness.metrics import summarize_trace
from selfdrive.test.longitudinal_harness.sweep import DEFAULT_SCORE_WEIGHTS, SweepCandidate, enumerate_candidates, run_sweep


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "testdata" / "ev6_lka_snapshot"


def _resolve_preview_handoff_vehicle():
  return resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    param_overrides={
      "VTSC.Expert.AdjLeadControlEnabled": True,
      "VTSC.Expert.AdjLeadCutInDRelMaxM": 60.0,
    },
  )


def test_default_score_weights_include_follow_overshoot() -> None:
  assert DEFAULT_SCORE_WEIGHTS["maxFollowOvershootMps"] == pytest.approx(3.0)
  assert DEFAULT_SCORE_WEIGHTS["handoffPrerevealMeanOvershootMps"] == pytest.approx(0.0)
  assert DEFAULT_SCORE_WEIGHTS["handoffPrerevealCruiseFraction"] == pytest.approx(0.0)


def test_resolve_ev6_controller_modes() -> None:
  shaped = resolve_ev6_vehicle_config(topology="lka", controller_mode="shaped")
  assert shaped.resolved_controller_mode == "shaped"
  assert not bool(shaped.cp.radarUnavailable)
  assert shaped.hyundai_tuning_mode != 0

  passthrough = resolve_ev6_vehicle_config(topology="lfa", controller_mode="passthrough")
  assert passthrough.resolved_controller_mode == "passthrough"
  assert bool(passthrough.cp.radarUnavailable)
  assert passthrough.hyundai_tuning_mode == 0


def test_handoff_scenario_uses_distinct_lead_slots() -> None:
  _, _, steps = build_synthetic_scenario("handoff", duration_s=4.0, dt_s=DT_MDL)
  assert any(step.lead_one.status and not step.lead_two.status for step in steps)
  assert any(step.lead_one.status and step.lead_two.status for step in steps)
  assert any((not step.lead_one.status) and step.lead_two.status for step in steps)


def test_handoff_previewable_exposes_adjacent_lead_before_reveal() -> None:
  vehicle = _resolve_preview_handoff_vehicle()
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("handoff_previewable", duration_s=4.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="handoff_previewable",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=19,
  )

  reveal_t = next(row["t_s"] for row in result.trace if row["event"] == "handoff_reveal")
  prereveal_rows = [row for row in result.trace if row["t_s"] < reveal_t and row["lead_two_status"]]
  preview_rows = [row for row in prereveal_rows if row["mpc_adjacent_awareness_preview_debug"].get("active", False)]

  assert prereveal_rows
  assert len(preview_rows) >= int(len(prereveal_rows) * 0.9)
  assert all(row["planner_source"] == "lead0" for row in prereveal_rows)
  assert all(row["mpc_adjacent_awareness_preview_debug"].get("applied", False) for row in preview_rows)
  assert any(row["planner_accel_mps2"] < -0.5 for row in prereveal_rows)
  assert prereveal_rows[-1]["lead_two_true_d_rel_m"] is not None
  assert preview_rows[0]["mpc_adjacent_awareness_preview_debug"]["slot"] == "lead1"


def test_handoff_previewable_early_deficit_strengthens_prereveal_signal() -> None:
  vehicle = _resolve_preview_handoff_vehicle()

  base_initial_speed_mps, base_initial_accel_mps2, base_steps = build_synthetic_scenario("handoff_previewable", duration_s=4.0, dt_s=DT_MDL)
  base_result = run_harness(
    vehicle_config=vehicle,
    scenario_name="handoff_previewable",
    steps=base_steps,
    initial_speed_mps=base_initial_speed_mps,
    initial_accel_mps2=base_initial_accel_mps2,
    noise_profile="off",
    seed=29,
  )

  deficit_initial_speed_mps, deficit_initial_accel_mps2, deficit_steps = build_synthetic_scenario("handoff_previewable_early_deficit", duration_s=4.0, dt_s=DT_MDL)
  deficit_result = run_harness(
    vehicle_config=vehicle,
    scenario_name="handoff_previewable_early_deficit",
    steps=deficit_steps,
    initial_speed_mps=deficit_initial_speed_mps,
    initial_accel_mps2=deficit_initial_accel_mps2,
    noise_profile="off",
    seed=29,
  )

  base_reveal_t = next(row["t_s"] for row in base_result.trace if row["event"] == "handoff_reveal")
  deficit_reveal_t = next(row["t_s"] for row in deficit_result.trace if row["event"] == "handoff_reveal")
  base_prereveal_rows = [row for row in base_result.trace if row["t_s"] < base_reveal_t and row["lead_two_status"]]
  deficit_prereveal_rows = [row for row in deficit_result.trace if row["t_s"] < deficit_reveal_t and row["lead_two_status"]]
  base_preview_rows = [row for row in base_prereveal_rows if row["mpc_adjacent_awareness_preview_debug"].get("active", False)]
  deficit_preview_rows = [row for row in deficit_prereveal_rows if row["mpc_adjacent_awareness_preview_debug"].get("active", False)]

  base_summary = summarize_trace(base_result.trace, vehicle=base_result.vehicle, scenario_name="handoff_previewable", noise_profile="off")
  deficit_summary = summarize_trace(deficit_result.trace, vehicle=deficit_result.vehicle, scenario_name="handoff_previewable_early_deficit", noise_profile="off")

  assert len(deficit_prereveal_rows) > len(base_prereveal_rows)
  assert deficit_preview_rows
  assert all(row["planner_source"] == "lead0" for row in deficit_prereveal_rows)
  assert deficit_prereveal_rows[-1]["lead_two_true_d_rel_m"] is not None
  assert any(row["planner_accel_mps2"] < -1.0 for row in deficit_prereveal_rows)
  assert deficit_summary["handoffPrerevealMeanOvershootMps"] > base_summary["handoffPrerevealMeanOvershootMps"]
  assert deficit_summary["leadEventOvershootGrowthMps"] >= base_summary["leadEventOvershootGrowthMps"]


def test_handoff_previewable_cruise_release_exposes_late_prereveal_source_gap() -> None:
  vehicle = _resolve_preview_handoff_vehicle()
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("handoff_previewable_cruise_release", duration_s=4.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="handoff_previewable_cruise_release",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=29,
  )

  reveal_t = next(row["t_s"] for row in result.trace if row["event"] == "handoff_reveal")
  all_prereveal_rows = [row for row in result.trace if row["t_s"] < reveal_t and row["lead_two_status"]]
  metric_window_rows = [row for row in all_prereveal_rows if row["t_s"] >= reveal_t - 0.75]
  preview_rows = [row for row in all_prereveal_rows if row["mpc_adjacent_awareness_preview_debug"].get("active", False)]
  cruise_rows = [row for row in metric_window_rows if row["planner_source"] == "cruise"]

  assert preview_rows
  assert cruise_rows
  assert any(row["planner_source"] == "lead0" for row in metric_window_rows)
  assert any(row["mpc_acc_source_debug"]["reason"] == "filtered_pullaway_immediate" for row in metric_window_rows)
  assert result.summary["handoffPrerevealCruiseFraction"] > 0.0


def test_harness_uses_vehicle_param_overrides_for_adjacent_lead_classifier() -> None:
  params = Params()
  params.put_bool("VTSC.Expert.AdjLeadControlEnabled", True)

  vehicle = resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    param_overrides={"VTSC.Expert.AdjLeadControlEnabled": False},
  )
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("handoff_previewable", duration_s=4.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="handoff_previewable",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=21,
  )

  prereveal_rows = [row for row in result.trace if row["t_s"] < 3.0 and row["lead_two_status"]]

  assert prereveal_rows
  assert all(row["control_lead_speed_mps"] == pytest.approx(22.0) for row in prereveal_rows)
  assert all(row["planner_source"] == "lead1" for row in prereveal_rows)
  assert all(not row["mpc_adjacent_awareness_preview_debug"].get("active", False) for row in prereveal_rows)


def test_cutin_lead_acquisition_smoothing() -> None:
  vehicle = resolve_ev6_vehicle_config(topology="lfa", controller_mode="passthrough")
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("cutin", duration_s=4.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="cutin",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=7,
  )

  reveal_row = next(row for row in result.trace if row["event"] == "lead_reveal")
  later_row = next(row for row in result.trace if row["t_s"] >= reveal_row["t_s"] + 0.10 and row["lead_one_status"])

  assert reveal_row["lead_one_a_lead_k_mps2"] == pytest.approx(0.0, abs=1e-6)
  assert 0.0 < reveal_row["lead_one_model_prob"] < 1.0
  assert later_row["lead_one_model_prob"] > reveal_row["lead_one_model_prob"]


def test_pullaway_close_engages_reclaim_path() -> None:
  vehicle = resolve_ev6_vehicle_config(topology="lfa", controller_mode="passthrough")
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("pullaway_close", duration_s=8.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="pullaway_close",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=9,
  )

  event_t = next(row["t_s"] for row in result.trace if row["event"] == "pullaway_start")
  after_rows = [row for row in result.trace if row["t_s"] >= event_t]
  lead_source_rows = [row for row in after_rows if row["planner_source"] in ("lead0", "lead1")]

  assert len(lead_source_rows) > 200
  assert any(row["planner_gap_reclaim_floor_mps2"] > 0.01 for row in after_rows)
  assert result.summary["reclaimDelayS"] is not None


def test_multi_cutin_repeats_lead_reveals_under_lead_control() -> None:
  vehicle = resolve_ev6_vehicle_config(topology="lfa", controller_mode="passthrough")
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("multi_cutin", duration_s=9.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="multi_cutin",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=13,
  )

  reveal_events = [step for step in steps if step.event == "lead_reveal"]
  lead_source_rows = [row for row in result.trace if row["planner_source"] in ("lead0", "lead1")]

  assert len(reveal_events) == 2
  assert len(lead_source_rows) == len(result.trace)
  assert result.summary["maxFollowOvershootMps"] > 1.0
  assert result.summary["maxFollowUndershootMps"] > 1.0


def test_accordion_close_stays_in_lead_control() -> None:
  vehicle = resolve_ev6_vehicle_config(topology="lfa", controller_mode="passthrough")
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("accordion_close", duration_s=10.0, dt_s=DT_MDL)
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="accordion_close",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=17,
  )

  lead_source_rows = [row for row in result.trace if row["planner_source"] in ("lead0", "lead1")]

  assert len(lead_source_rows) > int(len(result.trace) * 0.85)
  assert result.summary["maxFollowOvershootMps"] > 1.0
  assert result.summary["maxFollowUndershootMps"] > 1.0


def test_hyundai_controller_overlay_differs_from_passthrough() -> None:
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("approach", duration_s=2.0, dt_s=DT_MDL)

  shaped = run_harness(
    vehicle_config=resolve_ev6_vehicle_config(topology="lfa", controller_mode="shaped"),
    scenario_name="approach",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=3,
  )
  passthrough = run_harness(
    vehicle_config=resolve_ev6_vehicle_config(topology="lfa", controller_mode="passthrough"),
    scenario_name="approach",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="off",
    seed=3,
  )

  assert shaped.summary["hyundaiControllerShapingDivergenceMps2"] > 0.1
  assert passthrough.summary["hyundaiControllerShapingDivergenceMps2"] == pytest.approx(0.0, abs=1e-9)


def test_noise_seeds_allow_radar_and_ego_noise_isolation() -> None:
  initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario("approach", duration_s=2.0, dt_s=DT_MDL)
  vehicle = resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    plant_overrides={"aego_measure_noise_std": 0.15, "vego_measure_noise_std": 0.10},
  )

  baseline = run_harness(
    vehicle_config=vehicle,
    scenario_name="approach",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="realistic",
    noise_seeds=NoiseSeeds(drel=7, vrel=8, aego=9, vego=10),
  )
  changed_drel = run_harness(
    vehicle_config=vehicle,
    scenario_name="approach",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="realistic",
    noise_seeds=NoiseSeeds(drel=17, vrel=8, aego=9, vego=10),
  )
  changed_aego = run_harness(
    vehicle_config=vehicle,
    scenario_name="approach",
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile="realistic",
    noise_seeds=NoiseSeeds(drel=7, vrel=8, aego=19, vego=10),
  )

  assert baseline.trace[0]["lead_one_measured_d_rel_m"] != pytest.approx(changed_drel.trace[0]["lead_one_measured_d_rel_m"])
  assert baseline.trace[0]["lead_one_measured_d_rel_m"] == pytest.approx(changed_aego.trace[0]["lead_one_measured_d_rel_m"])
  assert baseline.trace[0]["measured_accel_mps2"] != pytest.approx(changed_aego.trace[0]["measured_accel_mps2"])


def test_snapshot_fixture_runs_with_snapshot_fidelity() -> None:
  bundle = load_snapshot_bundle(FIXTURE_DIR)
  vehicle = resolve_ev6_vehicle_config(
    topology=str(bundle.vehicle["topology"]),
    controller_mode="auto",
    tune_source="snapshot",
    snapshot_vehicle=bundle.vehicle,
    snapshot_params=bundle.params,
  )
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name=bundle.name,
    steps=bundle.timeline,
    initial_speed_mps=bundle.initial_speed_mps,
    initial_accel_mps2=bundle.initial_accel_mps2,
    noise_profile="off",
    seed=11,
  )

  assert result.vehicle["fidelitySource"] == "snapshot"
  assert result.vehicle["topology"] == "lka"
  assert result.summary["scenario"] == "ev6_lka_snapshot_fixture"
  assert len(result.trace) == len(bundle.timeline) * int(round(DT_MDL / DT_CTRL))


def test_snapshot_bundle_roundtrip(tmp_path: Path) -> None:
  bundle = load_snapshot_bundle(FIXTURE_DIR)
  out_dir = tmp_path / "roundtrip"
  write_snapshot_bundle(bundle, out_dir)
  reloaded = load_snapshot_bundle(out_dir)

  assert reloaded.vehicle["name"] == bundle.vehicle["name"]
  assert reloaded.params["HyundaiLongitudinalTuning"] == bundle.params["HyundaiLongitudinalTuning"]
  assert len(reloaded.timeline) == len(bundle.timeline)


def test_enumerate_candidates_cartesian_product() -> None:
  candidates = enumerate_candidates(
    {"Longitudinal.LiveTune.ObstacleCost": "4.0"},
    {
      "Longitudinal.LiveTune.AccelChangeCost": ["90", "115"],
      "Longitudinal.LiveTune.LeadPreviewStrength": ["1.0", "1.25"],
    },
  )

  assert len(candidates) == 4
  assert candidates[0].overrides["Longitudinal.LiveTune.ObstacleCost"] == "4.0"


def test_run_sweep_ranks_candidates() -> None:
  candidates = enumerate_candidates({}, {"Longitudinal.LiveTune.AccelChangeCost": ["90", "115"]})
  payload = run_sweep(
    candidates=candidates,
    scenarios=["approach"],
    mode="synthetic-ev6",
    snapshot=None,
    topology="lfa",
    controller_mode="shaped",
    hyundai_tuning_mode=None,
    noise="off",
    duration_s=1.0,
    seed=5,
    score_weights={
      "excessBrakeMps2": 1.0,
      "maxFollowOvershootMps": 1.0,
      "maxFollowUndershootMps": 1.0,
      "reclaimDelayS": 1.0,
      "leadAcquireLatencyS": 1.0,
      "handoffBrakeLatencyS": 1.0,
      "longControlRealizedDivergenceMps2": 1.0,
    },
  )

  assert payload["candidateCount"] == 2
  assert payload["rankedCandidates"][0]["score"] <= payload["rankedCandidates"][1]["score"]
  assert payload["rankedCandidates"][0]["scenarioSummaries"]["approach"]["scenario"] == "approach"


def test_run_sweep_target_result_is_batch_order_invariant() -> None:
  target = SweepCandidate({
    "Longitudinal.LiveTune.LeadPreviewStrength": "1.8",
    "Longitudinal.LiveTune.LeadPreviewMaxBufferM": "16",
  })
  companion_a = SweepCandidate({
    "Longitudinal.LiveTune.LeadPreviewStrength": "1.6",
    "Longitudinal.LiveTune.LeadPreviewMaxBufferM": "14",
  })
  companion_b = SweepCandidate({
    "Longitudinal.LiveTune.LeadPreviewStrength": "2.0",
    "Longitudinal.LiveTune.LeadPreviewMaxBufferM": "14",
  })

  payload_a = run_sweep(
    candidates=[companion_a, target],
    scenarios=[],
    mode="snapshot",
    snapshot=FIXTURE_DIR,
    topology="lfa",
    controller_mode="auto",
    hyundai_tuning_mode=None,
    noise="off",
    duration_s=0.0,
    seed=42,
    score_weights=dict(DEFAULT_SCORE_WEIGHTS),
  )
  payload_b = run_sweep(
    candidates=[target, companion_b],
    scenarios=[],
    mode="snapshot",
    snapshot=FIXTURE_DIR,
    topology="lfa",
    controller_mode="auto",
    hyundai_tuning_mode=None,
    noise="off",
    duration_s=0.0,
    seed=42,
    score_weights=dict(DEFAULT_SCORE_WEIGHTS),
  )

  target_overrides = target.overrides
  result_a = next(entry for entry in payload_a["rankedCandidates"] if entry["overrides"] == target_overrides)
  result_b = next(entry for entry in payload_b["rankedCandidates"] if entry["overrides"] == target_overrides)

  assert result_a["score"] == pytest.approx(result_b["score"])
  assert result_a["weightedPenalties"] == pytest.approx(result_b["weightedPenalties"])
  summary_a = result_a["scenarioSummaries"]["ev6_lka_snapshot_fixture"]
  summary_b = result_b["scenarioSummaries"]["ev6_lka_snapshot_fixture"]
  for key in (
    "peakPlannerBrakeMps2",
    "peakControllerBrakeMps2",
    "peakLongControlAccelMps2",
    "peakPlannerAccelMps2",
    "maxFollowOvershootMps",
    "maxFollowUndershootMps",
    "longControlRealizedDivergenceMps2",
  ):
    assert summary_a[key] == pytest.approx(summary_b[key])


def test_summary_uses_worst_latency_for_repeated_events() -> None:
  trace = [
    {
      "t_s": 0.0,
      "event": "lead_reveal",
      "has_any_lead": True,
      "active_lead_speed_mps": 20.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 35.0,
      "planner_accel_mps2": 0.0,
      "longcontrol_accel_mps2": 0.0,
      "controller_accel_mps2": 0.0,
      "realized_accel_mps2": 0.0,
      "planner_source": "cruise",
    },
    {
      "t_s": 0.1,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 20.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 35.0,
      "planner_accel_mps2": 0.0,
      "longcontrol_accel_mps2": 0.0,
      "controller_accel_mps2": 0.0,
      "realized_accel_mps2": 0.0,
      "planner_source": "lead0",
    },
    {
      "t_s": 2.0,
      "event": "lead_reveal",
      "has_any_lead": True,
      "active_lead_speed_mps": 18.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 30.0,
      "planner_accel_mps2": 0.0,
      "longcontrol_accel_mps2": 0.0,
      "controller_accel_mps2": 0.0,
      "realized_accel_mps2": 0.0,
      "planner_source": "cruise",
    },
    {
      "t_s": 2.4,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 18.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 30.0,
      "planner_accel_mps2": 0.0,
      "longcontrol_accel_mps2": 0.0,
      "controller_accel_mps2": 0.0,
      "realized_accel_mps2": 0.0,
      "planner_source": "lead0",
    },
  ]

  summary = summarize_trace(
    trace,
    vehicle={"name": "fixture"},
    scenario_name="multi_event_fixture",
    noise_profile="off",
  )

  assert summary["leadAcquireLatencyS"] == pytest.approx(0.4)


def test_summary_tracks_post_event_overshoot_growth_separately_from_reveal_delta() -> None:
  trace = [
    {
      "t_s": 0.0,
      "event": "handoff_reveal",
      "has_any_lead": True,
      "active_lead_speed_mps": 10.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 35.0,
      "planner_accel_mps2": 0.0,
      "longcontrol_accel_mps2": 0.0,
      "controller_accel_mps2": 0.0,
      "realized_accel_mps2": 0.0,
      "planner_source": "lead1",
    },
    {
      "t_s": 0.4,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 12.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 34.0,
      "planner_accel_mps2": -0.8,
      "longcontrol_accel_mps2": -0.8,
      "controller_accel_mps2": -0.8,
      "realized_accel_mps2": -0.7,
      "planner_source": "lead1",
    },
    {
      "t_s": 3.0,
      "event": "lead_reveal",
      "has_any_lead": True,
      "active_lead_speed_mps": 18.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 30.0,
      "planner_accel_mps2": 0.0,
      "longcontrol_accel_mps2": 0.0,
      "controller_accel_mps2": 0.0,
      "realized_accel_mps2": 0.0,
      "planner_source": "lead0",
    },
    {
      "t_s": 3.5,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 16.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 28.0,
      "planner_accel_mps2": -0.5,
      "longcontrol_accel_mps2": -0.5,
      "controller_accel_mps2": -0.5,
      "realized_accel_mps2": -0.4,
      "planner_source": "lead0",
    },
  ]

  summary = summarize_trace(
    trace,
    vehicle={"name": "fixture"},
    scenario_name="event_overshoot_fixture",
    noise_profile="off",
  )

  assert summary["maxFollowOvershootMps"] == pytest.approx(4.0)
  assert summary["leadEventOvershootGrowthMps"] == pytest.approx(2.0)


def test_summary_tracks_handoff_prereveal_mean_overshoot_against_reveal_target() -> None:
  trace = [
    {
      "t_s": 0.4,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 10.0,
      "control_lead_speed_mps": 30.0,
      "v_ego_true_mps": 28.0,
      "true_min_gap_m": 35.0,
      "planner_accel_mps2": -0.2,
      "longcontrol_accel_mps2": -0.2,
      "controller_accel_mps2": -0.2,
      "realized_accel_mps2": -0.1,
      "planner_source": "lead0",
    },
    {
      "t_s": 0.8,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 10.0,
      "control_lead_speed_mps": 30.0,
      "v_ego_true_mps": 24.0,
      "true_min_gap_m": 32.0,
      "planner_accel_mps2": -0.6,
      "longcontrol_accel_mps2": -0.6,
      "controller_accel_mps2": -0.6,
      "realized_accel_mps2": -0.5,
      "planner_source": "lead0",
    },
    {
      "t_s": 1.0,
      "event": "handoff_reveal",
      "has_any_lead": True,
      "active_lead_speed_mps": 18.0,
      "control_lead_speed_mps": 18.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 30.0,
      "planner_accel_mps2": -0.8,
      "longcontrol_accel_mps2": -0.8,
      "controller_accel_mps2": -0.8,
      "realized_accel_mps2": -0.7,
      "planner_source": "lead1",
    },
  ]

  summary = summarize_trace(
    trace,
    vehicle={"name": "fixture"},
    scenario_name="handoff_prereveal_fixture",
    noise_profile="off",
  )

  assert summary["maxFollowOvershootMps"] == pytest.approx(2.0)
  assert summary["handoffPrerevealMeanOvershootMps"] == pytest.approx(8.0)


def test_summary_ignores_preview_only_adjacent_lead_until_control_handoff() -> None:
  trace = [
    {
      "t_s": 0.6,
      "event": None,
      "has_any_lead": True,
      "active_lead_speed_mps": 10.0,
      "control_lead_speed_mps": 30.0,
      "v_ego_true_mps": 30.0,
      "true_min_gap_m": 35.0,
      "planner_accel_mps2": -0.3,
      "longcontrol_accel_mps2": -0.3,
      "controller_accel_mps2": -0.3,
      "realized_accel_mps2": -0.2,
      "planner_source": "lead0",
    },
    {
      "t_s": 1.0,
      "event": "handoff_reveal",
      "has_any_lead": True,
      "active_lead_speed_mps": 18.0,
      "control_lead_speed_mps": 18.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 30.0,
      "planner_accel_mps2": -0.6,
      "longcontrol_accel_mps2": -0.6,
      "controller_accel_mps2": -0.6,
      "realized_accel_mps2": -0.5,
      "planner_source": "lead1",
    },
  ]

  summary = summarize_trace(
    trace,
    vehicle={"name": "fixture"},
    scenario_name="preview_handoff_fixture",
    noise_profile="off",
  )

  assert summary["maxFollowOvershootMps"] == pytest.approx(2.0)
  assert summary["handoffPrerevealMeanOvershootMps"] == pytest.approx(12.0)


def test_summary_tracks_prereveal_cruise_fraction_before_handoff() -> None:
  trace = [
    {
      "t_s": 0.35,
      "event": None,
      "has_any_lead": True,
      "lead_one_status": True,
      "lead_two_status": True,
      "active_lead_speed_mps": 10.0,
      "control_lead_speed_mps": 30.0,
      "v_ego_true_mps": 28.0,
      "true_min_gap_m": 35.0,
      "planner_accel_mps2": -0.2,
      "longcontrol_accel_mps2": -0.2,
      "controller_accel_mps2": -0.2,
      "realized_accel_mps2": -0.1,
      "planner_source": "lead0",
    },
    {
      "t_s": 0.7,
      "event": None,
      "has_any_lead": True,
      "lead_one_status": True,
      "lead_two_status": True,
      "active_lead_speed_mps": 10.0,
      "control_lead_speed_mps": None,
      "v_ego_true_mps": 24.0,
      "true_min_gap_m": 32.0,
      "planner_accel_mps2": -0.1,
      "longcontrol_accel_mps2": -0.1,
      "controller_accel_mps2": -0.1,
      "realized_accel_mps2": -0.1,
      "planner_source": "cruise",
    },
    {
      "t_s": 1.0,
      "event": "handoff_reveal",
      "has_any_lead": True,
      "lead_one_status": False,
      "lead_two_status": True,
      "active_lead_speed_mps": 18.0,
      "control_lead_speed_mps": 18.0,
      "v_ego_true_mps": 20.0,
      "true_min_gap_m": 30.0,
      "planner_accel_mps2": -0.8,
      "longcontrol_accel_mps2": -0.8,
      "controller_accel_mps2": -0.8,
      "realized_accel_mps2": -0.7,
      "planner_source": "lead1",
    },
  ]

  summary = summarize_trace(
    trace,
    vehicle={"name": "fixture"},
    scenario_name="handoff_prereveal_release_fixture",
    noise_profile="off",
  )

  assert summary["handoffPrerevealCruiseFraction"] == pytest.approx(0.5)
