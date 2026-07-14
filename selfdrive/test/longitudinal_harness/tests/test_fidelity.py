from __future__ import annotations

from copy import deepcopy

import pytest

from openpilot.selfdrive.test.longitudinal_harness.config import REPLAY_PARAM_MANIFEST_KEYS
from openpilot.selfdrive.test.longitudinal_harness.fidelity import (
  FAIL,
  NOT_EVALUATED,
  PASS,
  evaluate_diagnostic_fidelity,
  evaluate_fidelity,
)


_COMMIT = "a" * 40
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_SHORT_COVERAGE_THRESHOLDS = {
  "radar": {
    "min_kinematic_samples": 1,
    "min_scorable_samples": 1,
    "min_scorable_fraction": 1.0,
    "min_contiguous_duration_s": 0.0,
  },
  "planner": {
    "min_scorable_samples": 1,
    "min_scorable_fraction": 1.0,
    "min_contiguous_duration_s": 0.0,
  },
}


def _metadata(commit: str = _COMMIT) -> dict[str, object]:
  return {
    "gitCommit": commit,
    "gitBranch": "chauffeur-exp01",
    "gitRemote": "git@github.com:example/chauffeur.git",
    "gitDirty": False,
    "gitDiffSha256": _EMPTY_SHA256,
    "gitDiffEmpty": True,
  }


def _evaluate(rows: list[dict], *, thresholds=None) -> dict:
  return evaluate_fidelity(
    rows,
    thresholds=thresholds,
    captured_metadata=_metadata(),
    replay_metadata=_metadata(),
  )


def _row(
  index: int,
  *,
  expected_d_rel_m: float = 40.0,
  actual_d_rel_m: float | None = 40.0,
  expected_v_rel_mps: float = -1.0,
  actual_v_rel_mps: float | None = -1.0,
  expected_source: str = "cruise",
  actual_source: str = "cruise",
  expected_a_target_mps2: float = 0.1,
  actual_a_target_mps2: float = 0.1,
  association: str | None = None,
  context: str | None = None,
  nested_statuses: bool = False,
) -> dict:
  plan_reference = {
    "aTargetMps2": expected_a_target_mps2,
    "source": expected_source,
  }
  reference = {
    "logMonoTimeNs": 1_000_000_000 + (index * 50_000_000),
    "radarState": {
      "leadOne": {
        "status": True,
        "dRelM": expected_d_rel_m,
        "vRelMps": expected_v_rel_mps,
      },
      "leadTwo": {
        "status": False,
        "dRelM": 0.0,
        "vRelMps": 0.0,
      },
    },
    "longitudinalPlan": plan_reference,
    "radardGateEligible": True,
    "radardServiceAssociationProvenance": {
      "contract": {"status": "exact", "version": 1},
      "modelV2": {"status": "exact"},
      "carState": {"status": "exact"},
      "liveTracks": {"status": "exact", "emptyPayloadValid": True},
    },
  }
  if nested_statuses:
    if association is not None:
      plan_reference["associationStatus"] = association
    if context is not None:
      plan_reference["contextStatus"] = context
  else:
    if association is not None:
      reference["plannerRadarResolution"] = association
    if context is not None:
      reference["plannerContextStatus"] = context

  if association == "exact" and context == "exact":
    target_clock = reference["logMonoTimeNs"] - 1
    plan_reference["radarStateLogMonoTimeNs"] = target_clock
    plan_reference["radarStateCandidatesNs"] = [target_clock]
    reference["plannerServiceLogMonoTimeNs"] = {"radarState": target_clock}
    reference["plannerServiceAssociationProvenance"] = {
      "contract": {"status": "exact", "version": 1},
      "carState": {"status": "exact"},
      "controlsState": {"status": "exact"},
      "carControl": {"status": "exact"},
      "selfdriveState": {"status": "exact"},
      "modelV2": {"status": "exact"},
      "radarState": {"status": "exact", "clockNs": target_clock},
      "params": {
        "status": "exact",
        "complete": True,
        "requiredKeyCount": len(REPLAY_PARAM_MANIFEST_KEYS),
        "capturedRequiredKeyCount": len(REPLAY_PARAM_MANIFEST_KEYS),
        "missingKeys": [],
      },
    }
    reference["plannerFidelity"] = {"scorable": True}

  return {
    "t_s": index * 0.05,
    "replay_reference": reference,
    "lead_one_published_d_rel_m": actual_d_rel_m,
    "lead_one_published_v_rel_mps": actual_v_rel_mps,
    "lead_two_published_d_rel_m": None,
    "lead_two_published_v_rel_mps": None,
    "planner_accel_mps2": actual_a_target_mps2,
    "planner_source": actual_source,
  }


def test_radar_can_pass_while_planner_is_not_evaluated() -> None:
  result = _evaluate([_row(index) for index in range(40)])

  assert result["radar"]["status"] == PASS
  assert result["planner"]["status"] == NOT_EVALUATED
  assert result["planner"]["sample_counts"]["scorable"] == 0
  assert result["planner"]["exclusion_reasons"] == {
    "missing_planner_context_status": 40,
    "missing_planner_radar_resolution": 40,
  }
  assert result["overall"]["status"] == NOT_EVALUATED
  assert result["overall"]["passed"] is False


def test_explicit_exact_planner_metadata_passes() -> None:
  rows = []
  for index in range(40):
    phase = index % 4
    expected_source = "lead0" if phase >= 2 else "cruise"
    expected_accel = -0.3 if phase == 2 else (-0.2 if phase == 3 else 0.1)
    accel_error = 0.02 if phase == 2 else (0.01 if phase == 3 else 0.0)
    rows.append(_row(
      index,
      expected_source=expected_source,
      actual_source=expected_source,
      expected_a_target_mps2=expected_accel,
      actual_a_target_mps2=expected_accel + accel_error,
      association="exact",
      context="exact",
      nested_statuses=index == 2,
    ))

  result = _evaluate(rows)

  assert result["radar"]["status"] == PASS
  assert result["planner"]["status"] == PASS
  assert result["planner"]["sample_counts"] == {"candidate": 40, "eligible": 40, "scorable": 40}
  assert result["planner"]["metrics"]["a_target_error_mps2"]["mae"] == pytest.approx(0.0075)
  assert result["planner"]["metrics"]["source_agreement"] == 1.0
  assert result["planner"]["metrics"]["source_transitions"]["sequence_match"] is True
  assert result["planner"]["metrics"]["source_transitions"]["timing_max_error_s"] == 0.0
  assert result["overall"]["status"] == PASS


def test_old_exact_params_label_without_complete_manifest_is_not_scorable() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    row["replay_reference"]["plannerServiceAssociationProvenance"]["params"] = {"status": "exact"}

  result = _evaluate(rows)

  assert result["planner"]["status"] == NOT_EVALUATED
  assert result["planner"]["sample_counts"]["scorable"] == 0
  assert result["planner"]["exclusion_reasons"] == {"inexact_planner_replay_contract": 40}
  assert result["overall"]["status"] == NOT_EVALUATED


def test_legacy_timing_association_is_diagnostic_only() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  rows[10]["replay_reference"]["plannerRadarResolution"] = "legacy_timing_unique"

  result = _evaluate(rows)

  assert result["radar"]["status"] == PASS
  assert result["planner"]["status"] == NOT_EVALUATED
  assert result["planner"]["exclusion_reasons"] == {"inexact_planner_radar_resolution": 1}
  assert result["planner"]["coverage"]["scorable_fraction"] == pytest.approx(39 / 40)
  assert result["overall"]["status"] == NOT_EVALUATED


def test_exact_labels_without_v1_contract_are_not_scorable() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    reference = row["replay_reference"]
    reference.pop("radardGateEligible")
    reference.pop("radardServiceAssociationProvenance")
    reference.pop("plannerFidelity")
    reference.pop("plannerServiceLogMonoTimeNs")
    reference.pop("plannerServiceAssociationProvenance")

  result = _evaluate(rows)

  assert result["radar"]["status"] == NOT_EVALUATED
  assert result["radar"]["exclusion_reasons"] == {"inexact_radard_replay_contract": 80}
  assert result["planner"]["status"] == NOT_EVALUATED
  assert result["planner"]["exclusion_reasons"] == {"inexact_planner_replay_contract": 40}
  assert result["overall"]["status"] == NOT_EVALUATED


def test_legacy_diagnostic_lane_reports_metrics_but_is_never_gate_eligible() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    reference = row["replay_reference"]
    reference["radarStateDiagnostic"] = reference.pop("radarState")
    reference["plannerRadarResolution"] = "unscorable"
    reference["plannerContextStatus"] = "legacy_derived"
    reference.pop("radardGateEligible")
    reference.pop("radardServiceAssociationProvenance")
    reference.pop("plannerFidelity")
    reference.pop("plannerServiceLogMonoTimeNs")
    reference.pop("plannerServiceAssociationProvenance")
    row["planner_radar_resolution"] = "legacy_timing_unique"

  strict = _evaluate(rows)
  diagnostic = evaluate_diagnostic_fidelity(rows)

  assert strict["overall"]["status"] == NOT_EVALUATED
  assert diagnostic["mode"] == "legacy_diagnostic_non_gating"
  assert diagnostic["radar"]["status"] == PASS
  assert diagnostic["planner"]["status"] == PASS
  assert diagnostic["status"] == PASS
  assert diagnostic["diagnosticPassed"] is True
  assert diagnostic["gateEligible"] is False
  assert diagnostic["planner"]["association_status_counts"] == {"legacy_timing_unique": 40}


def test_legacy_diagnostic_lane_labels_ambiguous_synchronous_scheduler_variant() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    reference = row["replay_reference"]
    reference["radarStateDiagnostic"] = reference.pop("radarState")
    reference["plannerRadarResolution"] = "unscorable"
    reference["plannerContextStatus"] = "legacy_derived"
    row["planner_radar_resolution"] = "legacy_ambiguous"

  diagnostic = evaluate_diagnostic_fidelity(rows)

  assert diagnostic["planner"]["status"] == PASS
  assert diagnostic["planner"]["association_status_counts"] == {"legacy_ambiguous_synchronous": 40}
  assert diagnostic["ambiguityPresent"] is True
  assert diagnostic["status"] == PASS
  assert diagnostic["gateEligible"] is False


def test_legacy_diagnostic_lane_keeps_active_object_hazard_context_unscorable() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    reference = row["replay_reference"]
    reference["radarStateDiagnostic"] = reference.pop("radarState")
    reference["plannerRadarResolution"] = "unscorable"
    reference["plannerContextStatus"] = "unscorable"
    reference["plannerContextReason"] = "active object hazard state is not exactly replayed"
    row["planner_radar_resolution"] = "legacy_ambiguous"

  diagnostic = evaluate_diagnostic_fidelity(rows)

  assert diagnostic["planner"]["status"] == NOT_EVALUATED
  assert diagnostic["planner"]["exclusion_reasons"] == {"inexact_planner_context_status": 40}
  assert diagnostic["status"] == NOT_EVALUATED


@pytest.mark.parametrize(
  "reason",
  [
    "selfdriveState diagnostic payload is missing",
    "captured parameter snapshot is missing",
  ],
)
def test_legacy_diagnostic_lane_does_not_promote_explicitly_unscorable_context(reason: str) -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    reference = row["replay_reference"]
    reference["radarStateDiagnostic"] = reference.pop("radarState")
    reference["plannerRadarResolution"] = "unscorable"
    reference["plannerContextStatus"] = "unscorable"
    reference["plannerContextReason"] = reason
    row["planner_radar_resolution"] = "legacy_ambiguous"

  diagnostic = evaluate_diagnostic_fidelity(rows)

  assert diagnostic["planner"]["status"] == NOT_EVALUATED
  assert diagnostic["planner"]["exclusion_reasons"] == {"inexact_planner_context_status": 40}
  assert diagnostic["status"] == NOT_EVALUATED


def test_radar_worst_error_drives_p95_failure_and_reports_max() -> None:
  errors = [0.0, 0.0, 0.0, 0.0, 1.0]
  rows = [
    _row(
      index,
      actual_d_rel_m=40.0 + error,
      association="exact",
      context="exact",
    )
    for index, error in enumerate(errors)
  ]

  result = _evaluate(rows, thresholds=_SHORT_COVERAGE_THRESHOLDS)

  d_rel = result["radar"]["metrics"]["d_rel_error_m"]
  assert d_rel["mae"] == pytest.approx(0.2)
  assert d_rel["p95"] == pytest.approx(0.8)
  assert d_rel["max"] == pytest.approx(1.0)
  assert result["radar"]["checks"]["d_rel_p95"] is False
  assert result["radar"]["status"] == FAIL
  assert result["overall"]["status"] == FAIL


def test_planner_worst_error_drives_p95_failure_and_reports_max() -> None:
  errors = [0.0, 0.0, 0.0, 0.0, 1.0]
  rows = [
    _row(
      index,
      actual_a_target_mps2=0.1 + error,
      association="exact",
      context="exact",
    )
    for index, error in enumerate(errors)
  ]

  result = _evaluate(rows, thresholds=_SHORT_COVERAGE_THRESHOLDS)

  a_target = result["planner"]["metrics"]["a_target_error_mps2"]
  assert a_target["mae"] == pytest.approx(0.2)
  assert a_target["p95"] == pytest.approx(0.8)
  assert a_target["max"] == pytest.approx(1.0)
  assert result["planner"]["checks"]["a_target_p95"] is False
  assert result["planner"]["status"] == FAIL


def test_single_catastrophic_frame_cannot_be_averaged_into_a_pass() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  rows[-1]["lead_one_published_d_rel_m"] = 41.0
  rows[-1]["lead_one_published_v_rel_mps"] = 0.0
  rows[-1]["planner_accel_mps2"] = 3.1

  result = _evaluate(rows)

  assert result["radar"]["metrics"]["d_rel_error_m"]["mae"] == pytest.approx(0.025)
  assert result["radar"]["metrics"]["d_rel_error_m"]["p95"] == 0.0
  assert result["radar"]["checks"]["d_rel_max"] is False
  assert result["radar"]["checks"]["v_rel_max"] is False
  assert result["planner"]["metrics"]["a_target_error_mps2"]["mae"] == pytest.approx(0.075)
  assert result["planner"]["metrics"]["a_target_error_mps2"]["p95"] == 0.0
  assert result["planner"]["checks"]["a_target_max"] is False
  assert result["overall"]["status"] == FAIL


def test_inconsistent_duplicate_reference_group_fails_trace_integrity() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  duplicate = deepcopy(rows[-1])
  duplicate["t_s"] += 0.01
  duplicate["planner_accel_mps2"] = -3.0
  rows.append(duplicate)

  result = _evaluate(rows)

  assert result["unique_replay_row_count"] == 40
  assert result["duplicate_trace_row_count"] == 1
  assert result["trace_integrity"]["status"] == FAIL
  assert result["trace_integrity"]["conflicting_reference_group_count"] == 1
  assert result["overall"]["status"] == FAIL


def test_consistent_duplicate_reference_rows_remain_scorable() -> None:
  unique_rows = [_row(index, association="exact", context="exact") for index in range(40)]
  rows = []
  for row in unique_rows:
    rows.extend((row, deepcopy(row)))

  result = _evaluate(rows)

  assert result["duplicate_trace_row_count"] == 40
  assert result["trace_integrity"]["status"] == PASS
  assert result["overall"]["status"] == PASS


@pytest.mark.parametrize("invalid_clock", [None, 0, -1])
def test_missing_or_nonpositive_replay_reference_clock_never_passes(invalid_clock: int | None) -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows:
    if invalid_clock is None:
      del row["replay_reference"]["logMonoTimeNs"]
    else:
      row["replay_reference"]["logMonoTimeNs"] = invalid_clock

  result = _evaluate(rows)

  assert result["trace_integrity"]["status"] == NOT_EVALUATED
  assert result["trace_integrity"]["missing_or_invalid_reference_id_row_count"] == 40
  assert result["overall"]["status"] == NOT_EVALUATED


def test_one_active_lead_sample_cannot_certify_radar_kinematics() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows[:-1]:
    row["replay_reference"]["radarState"]["leadOne"] = {
      "status": False,
      "dRelM": 0.0,
      "vRelMps": 0.0,
    }
    row["lead_one_published_d_rel_m"] = None
    row["lead_one_published_v_rel_mps"] = None

  result = _evaluate(rows)

  assert result["radar"]["coverage"]["active_lead_candidate_samples"] == 1
  assert result["radar"]["coverage"]["d_rel_kinematic_samples"] == 1
  assert result["radar"]["checks"]["min_d_rel_kinematic_samples"] is False
  assert result["radar"]["checks"]["min_v_rel_kinematic_samples"] is False
  assert result["radar"]["status"] == NOT_EVALUATED
  assert result["overall"]["status"] == NOT_EVALUATED


def test_source_transition_sequence_mismatch_fails() -> None:
  rows = [
    _row(0, association="exact", context="exact"),
    _row(
      1,
      expected_source="lead0",
      actual_source="lead1",
      association="exact",
      context="exact",
    ),
    _row(
      2,
      expected_source="lead0",
      actual_source="lead1",
      association="exact",
      context="exact",
    ),
  ]

  result = _evaluate(rows, thresholds=_SHORT_COVERAGE_THRESHOLDS)
  transitions = result["planner"]["metrics"]["source_transitions"]

  assert transitions["sequence_match"] is False
  assert transitions["timing_within_tolerance"] is False
  assert transitions["timing_max_error_s"] is None
  assert result["planner"]["checks"]["source_transition_sequence"] is False
  assert result["planner"]["status"] == FAIL


def test_source_transition_over_100ms_fails_timing() -> None:
  expected_sources = ["cruise", "lead0", "lead0", "lead0", "lead0"]
  actual_sources = ["cruise", "cruise", "cruise", "cruise", "lead0"]
  rows = [
    _row(
      index,
      expected_source=expected,
      actual_source=actual,
      association="exact",
      context="exact",
    )
    for index, (expected, actual) in enumerate(zip(expected_sources, actual_sources, strict=True))
  ]

  result = _evaluate(rows, thresholds=_SHORT_COVERAGE_THRESHOLDS)
  transitions = result["planner"]["metrics"]["source_transitions"]

  assert transitions["sequence_match"] is True
  assert transitions["timing_max_error_s"] == pytest.approx(0.15)
  assert transitions["timing_within_tolerance"] is False
  assert result["planner"]["checks"]["source_transition_timing"] is False
  assert result["planner"]["status"] == FAIL


def test_planner_exclusion_reason_counts_distinguish_missing_ambiguous_and_na() -> None:
  rows = [
    _row(0, context="exact"),
    _row(1, association="ambiguous", context="exact"),
    _row(2, association="N/A", context="exact"),
    _row(3, association="exact"),
    _row(4, association="exact", context="ambiguous"),
    _row(5, association="exact", context="not_applicable"),
    _row(6, association="previous", context="exact"),
  ]

  result = _evaluate(rows, thresholds=_SHORT_COVERAGE_THRESHOLDS)

  assert result["planner"]["sample_counts"]["scorable"] == 0
  assert result["planner"]["exclusion_reasons"] == {
    "ambiguous_planner_context_status": 1,
    "ambiguous_planner_radar_resolution": 1,
    "inexact_planner_radar_resolution": 1,
    "missing_planner_context_status": 1,
    "missing_planner_radar_resolution": 1,
    "not_applicable_planner_context_status": 1,
    "not_applicable_planner_radar_resolution": 1,
  }
  assert result["planner"]["status"] == NOT_EVALUATED
  assert result["overall"]["status"] == NOT_EVALUATED


def test_empty_or_zero_scorable_input_never_silently_passes() -> None:
  empty = _evaluate([])
  assert empty["radar"]["status"] == NOT_EVALUATED
  assert empty["planner"]["status"] == NOT_EVALUATED
  assert empty["overall"]["passed"] is False

  ambiguous = _evaluate(
    [_row(0, association="ambiguous", context="exact")],
    thresholds=_SHORT_COVERAGE_THRESHOLDS,
  )
  assert ambiguous["radar"]["status"] == PASS
  assert ambiguous["planner"]["sample_counts"]["scorable"] == 0
  assert ambiguous["planner"]["status"] == NOT_EVALUATED
  assert ambiguous["overall"]["status"] != PASS


def test_default_gate_rejects_trivial_perfect_coverage() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(4)]

  result = _evaluate(rows)

  for stage_name in ("radar", "planner"):
    stage = result[stage_name]
    assert stage["status"] == NOT_EVALUATED
    assert stage["checks"]["min_scorable_samples"] is False
    assert stage["checks"]["min_scorable_fraction"] is True
    assert stage["checks"]["min_contiguous_duration"] is False
    assert stage["coverage"]["candidate_samples"] == 4
    assert stage["coverage"]["scorable_samples"] == 4
    assert stage["coverage"]["longest_contiguous_run_samples"] == 4
    assert stage["coverage"]["longest_contiguous_duration_s"] == pytest.approx(0.15)
    assert any("below minimum 40" in reason for reason in stage["gate_reasons"])
  assert result["overall"]["passed"] is False


def test_scorable_fraction_is_measured_against_all_candidate_rows() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  for row in rows[-10:]:
    reference = row["replay_reference"]
    del reference["radarState"]
    del reference["plannerContextStatus"]
  thresholds = {
    "radar": {
      "min_scorable_samples": 20,
      "min_scorable_fraction": 1.0,
      "min_contiguous_duration_s": 1.0,
    },
    "planner": {
      "min_scorable_samples": 20,
      "min_scorable_fraction": 1.0,
      "min_contiguous_duration_s": 1.0,
    },
  }

  result = _evaluate(rows, thresholds=thresholds)

  for stage_name in ("radar", "planner"):
    stage = result[stage_name]
    assert stage["coverage"]["candidate_samples"] == 40
    assert stage["coverage"]["scorable_samples"] == 30
    assert stage["coverage"]["scorable_fraction"] == pytest.approx(0.75)
    assert stage["checks"]["min_scorable_samples"] is True
    assert stage["checks"]["min_scorable_fraction"] is False
    assert stage["checks"]["min_contiguous_duration"] is True
    assert stage["status"] == NOT_EVALUATED
    assert any("scorable fraction 0.750000" in reason for reason in stage["gate_reasons"])


def test_timestamp_gap_breaks_contiguous_scored_run_conservatively() -> None:
  indices = [*range(20), *range(40, 60)]
  rows = [_row(index, association="exact", context="exact") for index in indices]

  result = _evaluate(rows)

  for stage_name in ("radar", "planner"):
    stage = result[stage_name]
    assert stage["coverage"]["scorable_samples"] == 40
    assert stage["coverage"]["scorable_fraction"] == 1.0
    assert stage["coverage"]["longest_contiguous_run_samples"] == 20
    assert stage["coverage"]["longest_contiguous_duration_s"] == pytest.approx(0.95)
    assert stage["checks"]["min_scorable_samples"] is True
    assert stage["checks"]["min_scorable_fraction"] is True
    assert stage["checks"]["min_contiguous_duration"] is False
    assert stage["checks"]["complete_contiguous_window"] is False
    assert stage["status"] == NOT_EVALUATED


def test_full_window_gap_cannot_fabricate_an_exact_source_transition() -> None:
  indices = [*range(40), *range(60, 100)]
  rows = [
    _row(
      index,
      expected_source="cruise" if index < 60 else "lead0",
      actual_source="cruise" if index < 60 else "lead0",
      association="exact",
      context="exact",
    )
    for index in indices
  ]

  result = _evaluate(rows)

  assert result["planner"]["metrics"]["source_transitions"]["timing_max_error_s"] == 0.0
  assert result["planner"]["coverage"]["longest_contiguous_duration_s"] == pytest.approx(1.95)
  assert result["planner"]["coverage"]["max_observed_gap_s"] == pytest.approx(1.05)
  assert result["planner"]["checks"]["complete_contiguous_window"] is False
  assert result["planner"]["status"] == NOT_EVALUATED
  assert result["overall"]["status"] == NOT_EVALUATED


def test_unique_out_of_order_reference_clock_fails_trace_integrity() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  rows[20]["replay_reference"]["logMonoTimeNs"] = rows[19]["replay_reference"]["logMonoTimeNs"] - 1

  result = _evaluate(rows)

  assert result["trace_integrity"]["status"] == FAIL
  assert result["trace_integrity"]["out_of_order_reference_row_count"] == 1
  assert result["overall"]["status"] == FAIL


def test_default_complete_coverage_cannot_exclude_exact_transition_intervals() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(50)]
  for row in rows[:10]:
    reference = row["replay_reference"]
    del reference["radarState"]
    del reference["plannerContextStatus"]

  result = _evaluate(rows)

  for stage_name in ("radar", "planner"):
    stage = result[stage_name]
    assert stage["coverage"]["scorable_samples"] == 40
    assert stage["coverage"]["scorable_fraction"] == pytest.approx(0.8)
    assert stage["checks"]["min_scorable_fraction"] is False
    assert stage["status"] == NOT_EVALUATED
  assert result["overall"]["status"] == NOT_EVALUATED


def test_absent_provenance_blocks_otherwise_passing_fidelity() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]

  result = evaluate_fidelity(rows)

  assert result["radar"]["status"] == PASS
  assert result["planner"]["status"] == PASS
  assert result["provenance"]["status"] == "unknown"
  assert result["provenance"]["gateEligible"] is False
  assert result["overall"]["status"] == NOT_EVALUATED
  assert result["overall"]["passed"] is False
  assert any("capture git commit is missing" in reason for reason in result["overall"]["gate_reasons"])


def test_mismatched_provenance_is_non_gating_even_when_counterfactual_is_explicit() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  capture = _metadata("a" * 40)
  replay = _metadata("b" * 40)

  rejected = evaluate_fidelity(rows, captured_metadata=capture, replay_metadata=replay)
  counterfactual = evaluate_fidelity(
    rows,
    captured_metadata=capture,
    replay_metadata=replay,
    counterfactual=True,
  )

  assert rejected["provenance"]["status"] == "unknown"
  assert rejected["provenance"]["accepted"] is False
  assert rejected["overall"]["status"] == NOT_EVALUATED
  assert counterfactual["provenance"]["status"] == "counterfactual"
  assert counterfactual["provenance"]["accepted"] is True
  assert counterfactual["provenance"]["gateEligible"] is False
  assert counterfactual["radar"]["status"] == PASS
  assert counterfactual["planner"]["status"] == PASS
  assert counterfactual["overall"]["status"] == NOT_EVALUATED


def test_dirty_capture_without_verified_diff_blocks_fidelity_gate() -> None:
  rows = [_row(index, association="exact", context="exact") for index in range(40)]
  dirty_capture = {
    **_metadata(),
    "gitDirty": True,
    "gitDiffEmpty": False,
    "gitDiffSha256": None,
  }

  result = evaluate_fidelity(
    rows,
    captured_metadata=dirty_capture,
    replay_metadata=_metadata(),
    acknowledge_instrumentation_only=True,
  )

  assert result["provenance"]["status"] == "unknown"
  assert result["provenance"]["accepted"] is False
  assert result["provenance"]["gateEligible"] is False
  assert result["overall"]["status"] == NOT_EVALUATED
  assert any("git diff digest is missing or invalid" in reason for reason in result["overall"]["gate_reasons"])
