from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any

from openpilot.selfdrive.test.longitudinal_harness.config import exact_replay_param_manifest
from openpilot.selfdrive.test.longitudinal_harness.provenance import (
  classify_planner_runtime_provenance,
  classify_replay_provenance,
  exact_planner_state_initialization,
  well_formed_planner_state_initialization_claim,
)
from openpilot.selfdrive.test.longitudinal_harness.replay_contracts import (
  is_exact_supported_radard_replay_contract,
)


PASS = "pass"
FAIL = "fail"
NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class RadarFidelityThresholds:
  status_agreement_min: float = 0.99
  d_rel_mae_max_m: float = 0.10
  d_rel_p95_max_m: float = 0.25
  d_rel_abs_max_m: float = 0.50
  v_rel_mae_max_mps: float = 0.10
  v_rel_p95_max_mps: float = 0.25
  v_rel_abs_max_mps: float = 0.50
  min_kinematic_samples: int = 40
  min_scorable_samples: int = 40
  min_scorable_fraction: float = 1.0
  min_contiguous_duration_s: float = 1.50
  max_contiguous_gap_s: float = 0.075


@dataclass(frozen=True)
class PlannerFidelityThresholds:
  a_target_mae_max_mps2: float = 0.10
  a_target_p95_max_mps2: float = 0.20
  a_target_abs_max_mps2: float = 0.35
  source_agreement_min: float = 0.98
  transition_timing_max_s: float = 0.10
  min_scorable_samples: int = 40
  min_scorable_fraction: float = 1.0
  min_contiguous_duration_s: float = 1.50
  max_contiguous_gap_s: float = 0.075


@dataclass(frozen=True)
class FidelityThresholds:
  radar: RadarFidelityThresholds = field(default_factory=RadarFidelityThresholds)
  planner: PlannerFidelityThresholds = field(default_factory=PlannerFidelityThresholds)


DEFAULT_THRESHOLDS = FidelityThresholds()

_PLANNER_ASSOCIATION_ACCEPTED = frozenset(("exact",))
_DIAGNOSTIC_PLANNER_ASSOCIATION_ACCEPTED = frozenset((
  "exact",
  "legacy_timing_unique",
  "legacy_synchronous",
  "legacy_ambiguous_synchronous",
))
_DIAGNOSTIC_PLANNER_CONTEXT_ACCEPTED = frozenset(("exact", "legacy_derived", "inferred", "diagnostic_derived"))
_NOT_APPLICABLE_TOKENS = frozenset((
  "n_a",
  "na",
  "not_applicable",
  "not_available",
  "not_evaluated",
  "unavailable",
))
_AMBIGUOUS_TOKEN_PARTS = ("ambiguous", "multiple", "non_unique", "unresolved")
_MISSING = object()


def evaluate_fidelity(
  trace_rows: Iterable[Mapping[str, Any]],
  *,
  thresholds: FidelityThresholds | Mapping[str, Any] | None = None,
  captured_metadata: Mapping[str, Any] | None = None,
  replay_metadata: Mapping[str, Any] | None = None,
  current_commit: str | None = None,
  current_diff_sha256: str | None = None,
  current_dirty: bool | None = None,
  current_diff_empty: bool | None = None,
  counterfactual: bool = False,
  acknowledge_instrumentation_only: bool = False,
) -> dict[str, Any]:
  """Public gate: row metadata alone can never verify planner state restoration."""
  return _evaluate_fidelity_impl(
    trace_rows,
    thresholds=thresholds,
    captured_metadata=captured_metadata,
    replay_metadata=replay_metadata,
    current_commit=current_commit,
    current_diff_sha256=current_diff_sha256,
    current_dirty=current_dirty,
    current_diff_empty=current_diff_empty,
    counterfactual=counterfactual,
    acknowledge_instrumentation_only=acknowledge_instrumentation_only,
    planner_state_restoration_verified=False,
  )


def evaluate_harness_fidelity(
  simulation_result: Any,
  *,
  thresholds: FidelityThresholds | Mapping[str, Any] | None = None,
  captured_metadata: Mapping[str, Any] | None = None,
  replay_metadata: Mapping[str, Any] | None = None,
  current_commit: str | None = None,
  current_diff_sha256: str | None = None,
  current_dirty: bool | None = None,
  current_diff_empty: bool | None = None,
  counterfactual: bool = False,
  acknowledge_instrumentation_only: bool = False,
) -> dict[str, Any]:
  """Formal gate for state restoration independently verified by run_harness.

  Unlike ``evaluate_fidelity``, this does not trust a row-authored claim. It
  requires the exact claim returned by the harness after validating the full
  route-prefix digest and production constructor inputs, and requires every
  replay row to carry that same immutable claim.
  """
  rows = getattr(simulation_result, "trace", None)
  claim = getattr(simulation_result, "planner_state_initialization_provenance", None)
  verified = bool(
    getattr(simulation_result, "planner_state_restoration_verified", False) is True and
    isinstance(rows, list) and
    isinstance(claim, Mapping) and
    exact_planner_state_initialization(claim, restoration_verified=True) and
    rows and
    all(
      isinstance(row, Mapping) and
      isinstance(row.get("replay_reference"), Mapping) and
      row["replay_reference"].get("plannerStateInitializationProvenance") == claim
      for row in rows
    )
  )
  return _evaluate_fidelity_impl(
    rows or [],
    thresholds=thresholds,
    captured_metadata=captured_metadata,
    replay_metadata=replay_metadata,
    current_commit=current_commit,
    current_diff_sha256=current_diff_sha256,
    current_dirty=current_dirty,
    current_diff_empty=current_diff_empty,
    counterfactual=counterfactual,
    acknowledge_instrumentation_only=acknowledge_instrumentation_only,
    planner_state_restoration_verified=verified,
  )


def _evaluate_fidelity_with_test_only_verified_planner_state(
  trace_rows: Iterable[Mapping[str, Any]],
  **kwargs: Any,
) -> dict[str, Any]:
  """Exercise downstream exact-planner quality checks without exposing a public bypass."""
  return _evaluate_fidelity_impl(
    trace_rows,
    planner_state_restoration_verified=True,
    **kwargs,
  )


def _evaluate_fidelity_impl(
  trace_rows: Iterable[Mapping[str, Any]],
  *,
  thresholds: FidelityThresholds | Mapping[str, Any] | None = None,
  captured_metadata: Mapping[str, Any] | None = None,
  replay_metadata: Mapping[str, Any] | None = None,
  current_commit: str | None = None,
  current_diff_sha256: str | None = None,
  current_dirty: bool | None = None,
  current_diff_empty: bool | None = None,
  counterfactual: bool = False,
  acknowledge_instrumentation_only: bool = False,
  planner_state_restoration_verified: bool,
) -> dict[str, Any]:
  """Evaluate independently replayed RadarD and longitudinal-planner output.

  ``run_harness`` emits one row per 100 Hz control tick while a recorded replay
  normally has one reference frame per 20 Hz planner tick. Rows carrying the
  same ``replay_reference.logMonoTimeNs`` are therefore collapsed before
  scoring, so each recorded frame receives equal weight.

  Planner output is intentionally stricter than radar output. It is scored only
  when the reference carries a supported RadarD dependency contract, the planner
  v1 external-input contract, an exact radar association and planner context,
  an explicitly applied captured planner/MPC
  state seed, and matching numerical-runtime provenance. Missing, ambiguous,
  and N/A metadata
  is reported as exclusion evidence and cannot turn a zero-sample planner stage
  into a pass. Both stages must also satisfy minimum sample count, scorable
  fraction, and contiguous-duration coverage. Finally, overall fidelity can pass
  only when captured and replay git provenance classifies as gate-eligible exact.
  """
  resolved_thresholds = _coerce_thresholds(thresholds)
  _validate_thresholds(resolved_thresholds)
  input_rows = [row for row in trace_rows if isinstance(row, Mapping)]
  replay_rows, duplicate_count, trace_integrity = _unique_replay_rows(input_rows)

  radar = _evaluate_radar(replay_rows, resolved_thresholds.radar)
  planner_runtime = classify_planner_runtime_provenance(captured_metadata, replay_metadata)
  planner = _evaluate_planner(
    replay_rows,
    resolved_thresholds.planner,
    planner_runtime_gate_eligible=planner_runtime.gate_eligible,
    planner_state_restoration_verified=planner_state_restoration_verified,
  )
  planner["runtime_provenance"] = planner_runtime.as_dict()
  provenance = classify_replay_provenance(
    captured_metadata,
    replay_metadata,
    current_commit=current_commit,
    current_diff_sha256=current_diff_sha256,
    current_dirty=current_dirty,
    current_diff_empty=current_diff_empty,
    counterfactual=counterfactual,
    acknowledge_instrumentation_only=acknowledge_instrumentation_only,
  )
  statuses = (radar["status"], planner["status"])
  if FAIL in statuses:
    overall_status = FAIL
  elif NOT_EVALUATED in statuses:
    overall_status = NOT_EVALUATED
  else:
    overall_status = PASS

  # This is an explicit invariant, even if future stage-status logic changes.
  if planner["sample_counts"]["scorable"] == 0:
    overall_status = NOT_EVALUATED if radar["status"] != FAIL else FAIL

  if overall_status == PASS and not provenance.gate_eligible:
    overall_status = NOT_EVALUATED

  if trace_integrity["status"] == FAIL:
    overall_status = FAIL
  elif trace_integrity["status"] == NOT_EVALUATED:
    overall_status = NOT_EVALUATED

  overall_gate_reasons: list[str] = []
  if radar["status"] != PASS:
    overall_gate_reasons.append(f"radar stage status is {radar['status']}")
    overall_gate_reasons.extend(f"radar: {reason}" for reason in radar["gate_reasons"])
  if planner["status"] != PASS:
    overall_gate_reasons.append(f"planner stage status is {planner['status']}")
    overall_gate_reasons.extend(f"planner: {reason}" for reason in planner["gate_reasons"])
  if not provenance.gate_eligible:
    overall_gate_reasons.extend(f"provenance: {reason}" for reason in provenance.reasons)
  if trace_integrity["status"] == FAIL:
    overall_gate_reasons.append(
      "trace integrity: duplicate replay references contain inconsistent expected or observed fidelity values"
    )
  elif trace_integrity["status"] == NOT_EVALUATED:
    overall_gate_reasons.append(
      "trace integrity: every fidelity row requires a positive replay-reference logMonoTimeNs"
    )

  return {
    "schema_version": 4,
    "status": overall_status,
    "overall": {
      "status": overall_status,
      "passed": overall_status == PASS,
      "gate_reasons": overall_gate_reasons,
    },
    "provenance": provenance.as_dict(),
    "thresholds": asdict(resolved_thresholds),
    "input_row_count": len(input_rows),
    "unique_replay_row_count": len(replay_rows),
    "duplicate_trace_row_count": duplicate_count,
    "trace_integrity": trace_integrity,
    "radar": radar,
    "planner": planner,
  }


def evaluate_diagnostic_fidelity(
  trace_rows: Iterable[Mapping[str, Any]],
  *,
  thresholds: FidelityThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Measure legacy replay agreement without promoting inferred inputs into an exact gate.

  Historical rlogs predate the v1 writer contracts. This lane deliberately
  exposes their numerical RadarD/planner agreement while preserving the reason
  it cannot certify causal equivalence. It may use ``radarStateDiagnostic`` and
  a synchronous legacy planner association, but ambiguity and known missing
  active-object-hazard context remain unscorable.
  """
  resolved_thresholds = _coerce_thresholds(thresholds)
  _validate_thresholds(resolved_thresholds)
  input_rows = [row for row in trace_rows if isinstance(row, Mapping)]
  replay_rows, duplicate_count, trace_integrity = _unique_replay_rows(input_rows)
  radar = _evaluate_radar(
    replay_rows,
    resolved_thresholds.radar,
    require_exact_contract=False,
    allow_diagnostic_reference=True,
  )
  planner = _evaluate_planner(
    replay_rows,
    resolved_thresholds.planner,
    require_exact_contract=False,
    diagnostic=True,
  )
  statuses = (radar["status"], planner["status"])
  if FAIL in statuses:
    status = FAIL
  elif NOT_EVALUATED in statuses:
    status = NOT_EVALUATED
  else:
    status = PASS
  if trace_integrity["status"] == FAIL:
    status = FAIL
  elif trace_integrity["status"] == NOT_EVALUATED:
    status = NOT_EVALUATED

  reasons: list[str] = []
  for stage_name, stage in (("radar", radar), ("planner", planner)):
    if stage["status"] != PASS:
      reasons.append(f"{stage_name} diagnostic stage status is {stage['status']}")
      reasons.extend(f"{stage_name}: {reason}" for reason in stage["gate_reasons"])
  if trace_integrity["status"] != PASS:
    reasons.append(f"trace integrity status is {trace_integrity['status']}")

  return {
    "schema_version": 1,
    "mode": "legacy_diagnostic_non_gating",
    "status": status,
    "diagnosticPassed": status == PASS,
    "gateEligible": False,
    "ambiguityPresent": planner["association_status_counts"].get("legacy_ambiguous_synchronous", 0) > 0,
    "reasons": reasons,
    "limitations": [
      "diagnostic results do not certify that replay consumed the same asynchronous service snapshots",
      "legacy planner rows may use a synchronous or timing-inferred RadarD association",
      "legacy_ambiguous_synchronous rows score one explicit scheduler variant and do not resolve the recorded race",
      "only a state-seeded exact replay on a matching planner runtime can authorize production behavior changes",
    ],
    "thresholds": asdict(resolved_thresholds),
    "input_row_count": len(input_rows),
    "unique_replay_row_count": len(replay_rows),
    "duplicate_trace_row_count": duplicate_count,
    "trace_integrity": trace_integrity,
    "radar": radar,
    "planner": planner,
  }


def _coerce_thresholds(value: FidelityThresholds | Mapping[str, Any] | None) -> FidelityThresholds:
  if value is None:
    return DEFAULT_THRESHOLDS
  if isinstance(value, FidelityThresholds):
    return value
  if not isinstance(value, Mapping):
    raise TypeError("thresholds must be FidelityThresholds, a mapping, or None")

  radar_payload = value.get("radar", {})
  planner_payload = value.get("planner", {})
  if not isinstance(radar_payload, Mapping) or not isinstance(planner_payload, Mapping):
    raise TypeError("thresholds['radar'] and thresholds['planner'] must be mappings")
  return FidelityThresholds(
    radar=RadarFidelityThresholds(**{**asdict(DEFAULT_THRESHOLDS.radar), **dict(radar_payload)}),
    planner=PlannerFidelityThresholds(**{**asdict(DEFAULT_THRESHOLDS.planner), **dict(planner_payload)}),
  )


def _validate_thresholds(thresholds: FidelityThresholds) -> None:
  for label, stage in (("radar", thresholds.radar), ("planner", thresholds.planner)):
    if (
      not isinstance(stage.min_scorable_samples, int)
      or isinstance(stage.min_scorable_samples, bool)
      or stage.min_scorable_samples < 1
    ):
      raise ValueError(f"{label}.min_scorable_samples must be at least 1")
    if not _is_finite_number(stage.min_scorable_fraction) or not 0.0 < stage.min_scorable_fraction <= 1.0:
      raise ValueError(f"{label}.min_scorable_fraction must be in (0, 1]")
    if not _is_finite_number(stage.min_contiguous_duration_s) or stage.min_contiguous_duration_s < 0.0:
      raise ValueError(f"{label}.min_contiguous_duration_s must be non-negative")
    if not _is_finite_number(stage.max_contiguous_gap_s) or stage.max_contiguous_gap_s <= 0.0:
      raise ValueError(f"{label}.max_contiguous_gap_s must be positive")
  if (
    not isinstance(thresholds.radar.min_kinematic_samples, int)
    or isinstance(thresholds.radar.min_kinematic_samples, bool)
    or thresholds.radar.min_kinematic_samples < 1
  ):
    raise ValueError("radar.min_kinematic_samples must be at least 1")

  bounded_minimums = {
    "radar.status_agreement_min": thresholds.radar.status_agreement_min,
    "planner.source_agreement_min": thresholds.planner.source_agreement_min,
  }
  for label, value in bounded_minimums.items():
    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
      raise ValueError(f"{label} must be in [0, 1]")

  nonnegative_maximums = {
    "radar.d_rel_mae_max_m": thresholds.radar.d_rel_mae_max_m,
    "radar.d_rel_p95_max_m": thresholds.radar.d_rel_p95_max_m,
    "radar.d_rel_abs_max_m": thresholds.radar.d_rel_abs_max_m,
    "radar.v_rel_mae_max_mps": thresholds.radar.v_rel_mae_max_mps,
    "radar.v_rel_p95_max_mps": thresholds.radar.v_rel_p95_max_mps,
    "radar.v_rel_abs_max_mps": thresholds.radar.v_rel_abs_max_mps,
    "planner.a_target_mae_max_mps2": thresholds.planner.a_target_mae_max_mps2,
    "planner.a_target_p95_max_mps2": thresholds.planner.a_target_p95_max_mps2,
    "planner.a_target_abs_max_mps2": thresholds.planner.a_target_abs_max_mps2,
    "planner.transition_timing_max_s": thresholds.planner.transition_timing_max_s,
  }
  for label, value in nonnegative_maximums.items():
    if not _is_finite_number(value) or value < 0.0:
      raise ValueError(f"{label} must be non-negative")


def _is_finite_number(value: Any) -> bool:
  return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _unique_replay_rows(
  rows: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int, dict[str, Any]]:
  unique_rows: list[Mapping[str, Any]] = []
  fingerprints_by_reference_time: dict[int, Any] = {}
  duplicate_count = 0
  conflicting_reference_times: set[int] = set()
  out_of_order_reference_times: list[int] = []
  missing_or_invalid_reference_id_count = 0
  last_reference_time: int | None = None
  for row in rows:
    reference = row.get("replay_reference")
    log_mono_time = reference.get("logMonoTimeNs") if isinstance(reference, Mapping) else None
    if isinstance(log_mono_time, int) and not isinstance(log_mono_time, bool) and log_mono_time > 0:
      fingerprint = _fidelity_fingerprint(row)
      if log_mono_time in fingerprints_by_reference_time:
        duplicate_count += 1
        if last_reference_time != log_mono_time:
          out_of_order_reference_times.append(log_mono_time)
        if fingerprints_by_reference_time[log_mono_time] != fingerprint:
          conflicting_reference_times.add(log_mono_time)
        continue
      if last_reference_time is not None and log_mono_time <= last_reference_time:
        out_of_order_reference_times.append(log_mono_time)
      fingerprints_by_reference_time[log_mono_time] = fingerprint
      last_reference_time = log_mono_time
    else:
      missing_or_invalid_reference_id_count += 1
    unique_rows.append(row)
  conflicting_times = sorted(conflicting_reference_times)
  integrity_status = (
    FAIL if conflicting_times or out_of_order_reference_times
    else NOT_EVALUATED if missing_or_invalid_reference_id_count
    else PASS
  )
  return unique_rows, duplicate_count, {
    "status": integrity_status,
    "passed": integrity_status == PASS,
    "duplicate_row_count": duplicate_count,
    "missing_or_invalid_reference_id_row_count": missing_or_invalid_reference_id_count,
    "conflicting_reference_group_count": len(conflicting_times),
    "conflicting_reference_log_mono_times_ns": conflicting_times[:20],
    "conflicting_reference_times_truncated": len(conflicting_times) > 20,
    "out_of_order_reference_row_count": len(out_of_order_reference_times),
    "out_of_order_reference_log_mono_times_ns": out_of_order_reference_times[:20],
    "out_of_order_reference_times_truncated": len(out_of_order_reference_times) > 20,
  }


def _fidelity_fingerprint(row: Mapping[str, Any]) -> Any:
  """Return values that must be stable across 100 Hz rows sharing one replay oracle."""
  return _freeze_for_compare({
    "replay_reference": row.get("replay_reference"),
    "planner_accel_mps2": _first_value(row, "planner_accel_mps2", "planner_a_target_mps2"),
    "planner_source": _first_value(row, "planner_source"),
    "lead_one_status": _actual_radar_status(row, "leadOne", "lead_one"),
    "lead_two_status": _actual_radar_status(row, "leadTwo", "lead_two"),
    "lead_one_d_rel_m": _first_value(row, "lead_one_published_d_rel_m"),
    "lead_two_d_rel_m": _first_value(row, "lead_two_published_d_rel_m"),
    "lead_one_v_rel_mps": _first_value(row, "lead_one_published_v_rel_mps"),
    "lead_two_v_rel_mps": _first_value(row, "lead_two_published_v_rel_mps"),
  })


def _freeze_for_compare(value: Any) -> Any:
  if isinstance(value, Mapping):
    return tuple(sorted((str(key), _freeze_for_compare(item)) for key, item in value.items()))
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_for_compare(item) for item in value)
  if isinstance(value, float) and math.isnan(value):
    return ("float", "nan")
  try:
    hash(value)
  except TypeError:
    return repr(value)
  return value


def _row_time_s(row: Mapping[str, Any]) -> float | None:
  reference = row.get("replay_reference")
  if isinstance(reference, Mapping):
    log_mono_time = reference.get("logMonoTimeNs")
    if isinstance(log_mono_time, int) and not isinstance(log_mono_time, bool) and log_mono_time > 0:
      return log_mono_time / 1e9
  return _finite_float(_first_value(row, "t_s", "time_s"))


def _coverage_summary(
  rows: list[Mapping[str, Any]],
  scorable_mask: list[bool],
  *,
  max_contiguous_gap_s: float,
) -> dict[str, float | int | None]:
  if len(rows) != len(scorable_mask):
    raise ValueError("coverage mask must have one entry per replay row")

  scorable_count = sum(scorable_mask)
  scorable_fraction = scorable_count / len(rows) if rows else None
  candidate_times = [_row_time_s(row) for row in rows]
  candidate_gaps = [
    current - previous
    for previous, current in zip(candidate_times[:-1], candidate_times[1:], strict=True)
    if previous is not None and current is not None
  ]
  complete_contiguous_window = bool(rows) and all(value is not None for value in candidate_times)
  if complete_contiguous_window:
    complete_contiguous_window = all(
      gap > 0.0 and gap <= max_contiguous_gap_s + 1e-12
      for gap in candidate_gaps
    )
  timed_scorable_count = 0
  longest_run_samples = 0
  longest_run_duration_s = 0.0
  run_start_s: float | None = None
  run_last_s: float | None = None
  run_samples = 0

  for row, scorable in zip(rows, scorable_mask, strict=True):
    sample_time_s = _row_time_s(row) if scorable else None
    if sample_time_s is None:
      run_start_s = None
      run_last_s = None
      run_samples = 0
      continue

    timed_scorable_count += 1
    gap_s = sample_time_s - run_last_s if run_last_s is not None else None
    if run_last_s is None or gap_s is None or gap_s <= 0.0 or gap_s > max_contiguous_gap_s + 1e-12:
      run_start_s = sample_time_s
      run_samples = 1
    else:
      run_samples += 1
    run_last_s = sample_time_s

    # A run's duration is bounded only by observed timestamps. Do not infer an
    # extra sample period before its first or after its final row.
    run_duration_s = sample_time_s - run_start_s if run_start_s is not None else 0.0
    if run_duration_s > longest_run_duration_s + 1e-12:
      longest_run_duration_s = run_duration_s
      longest_run_samples = run_samples
    elif abs(run_duration_s - longest_run_duration_s) <= 1e-12:
      longest_run_samples = max(longest_run_samples, run_samples)

  return {
    "candidate_samples": len(rows),
    "scorable_samples": scorable_count,
    "scorable_fraction": scorable_fraction,
    "timed_scorable_samples": timed_scorable_count,
    "longest_contiguous_run_samples": longest_run_samples,
    "longest_contiguous_duration_s": longest_run_duration_s,
    "max_contiguous_gap_s": max_contiguous_gap_s,
    "max_observed_gap_s": max(candidate_gaps, default=0.0) if complete_contiguous_window or candidate_gaps else None,
    "complete_contiguous_window": complete_contiguous_window,
  }


def _coverage_checks(
  coverage: Mapping[str, float | int | None],
  *,
  min_scorable_samples: int,
  min_scorable_fraction: float,
  min_contiguous_duration_s: float,
) -> dict[str, bool | None]:
  scorable_samples = int(coverage["scorable_samples"] or 0)
  scorable_fraction = _finite_float(coverage.get("scorable_fraction"))
  longest_duration_s = _finite_float(coverage.get("longest_contiguous_duration_s"))
  return {
    "min_scorable_samples": scorable_samples >= min_scorable_samples,
    "min_scorable_fraction": _minimum_check(scorable_fraction, min_scorable_fraction),
    "min_contiguous_duration": _minimum_check(longest_duration_s, min_contiguous_duration_s),
    "complete_contiguous_window": coverage.get("complete_contiguous_window") is True,
  }


def _coverage_reasons(
  coverage: Mapping[str, float | int | None],
  checks: Mapping[str, bool | None],
  *,
  min_scorable_samples: int,
  min_scorable_fraction: float,
  min_contiguous_duration_s: float,
) -> list[str]:
  reasons: list[str] = []
  if int(coverage["candidate_samples"] or 0) == 0:
    reasons.append("no candidate replay rows")
  if checks["min_scorable_samples"] is not True:
    reasons.append(
      f"scorable samples {coverage['scorable_samples']} below minimum {min_scorable_samples}"
    )
  if checks["min_scorable_fraction"] is not True:
    fraction = coverage["scorable_fraction"]
    rendered = "unknown" if fraction is None else f"{float(fraction):.6f}"
    reasons.append(f"scorable fraction {rendered} below minimum {min_scorable_fraction:.6f}")
  if checks["min_contiguous_duration"] is not True:
    longest_duration_s = float(coverage["longest_contiguous_duration_s"] or 0.0)
    reasons.append(
      f"longest contiguous scored duration {longest_duration_s:.6f}s "
      + f"below minimum {min_contiguous_duration_s:.6f}s"
    )
  if checks["complete_contiguous_window"] is not True:
    max_gap_s = coverage.get("max_observed_gap_s")
    rendered_gap = "unknown" if max_gap_s is None else f"{float(max_gap_s):.6f}s"
    reasons.append(
      f"evaluation window is not completely contiguous; maximum observed gap is {rendered_gap}"
    )
  return reasons


def _quality_reasons(checks: Mapping[str, bool | None]) -> list[str]:
  reasons: list[str] = []
  for name, result in checks.items():
    if result is False:
      reasons.append(f"quality check failed: {name}")
    elif result is None:
      reasons.append(f"quality check not evaluated: {name}")
  return reasons


def _stage_status_with_coverage(
  quality_checks: Iterable[bool | None],
  coverage_checks: Iterable[bool | None],
) -> str:
  quality_status = _stage_status(quality_checks)
  if quality_status != PASS:
    return quality_status
  # Insufficient coverage is absence of enough evidence, not evidence of an
  # output mismatch. It must block a pass without misreporting a fidelity fail.
  return PASS if _stage_status(coverage_checks) == PASS else NOT_EVALUATED


def _evaluate_radar(
  rows: list[Mapping[str, Any]],
  thresholds: RadarFidelityThresholds,
  *,
  require_exact_contract: bool = True,
  allow_diagnostic_reference: bool = False,
) -> dict[str, Any]:
  exclusions: Counter[str] = Counter()
  status_matches: list[bool] = []
  d_rel_errors: list[float] = []
  v_rel_errors: list[float] = []
  active_lead_candidate_count = 0
  scorable_mask: list[bool] = []

  for row in rows:
    row_scorable = True
    scorable_mask.append(False)
    reference = row.get("replay_reference")
    if not isinstance(reference, Mapping):
      exclusions["missing_replay_reference"] += 2
      continue
    if require_exact_contract and not _radard_contract_exact(reference):
      exclusions["inexact_radard_replay_contract"] += 2
      continue
    radar_reference = reference.get("radarState")
    if allow_diagnostic_reference and not isinstance(radar_reference, Mapping):
      radar_reference = reference.get("radarStateDiagnostic")
    if not isinstance(radar_reference, Mapping):
      # Exclusion counts use lead samples as their unit for radar.
      exclusions["missing_radar_reference"] += 2
      continue

    for slot, field_stem in (("leadOne", "lead_one"), ("leadTwo", "lead_two")):
      lead_reference = radar_reference.get(slot)
      if not isinstance(lead_reference, Mapping):
        exclusions["missing_radar_lead_reference"] += 1
        row_scorable = False
        continue

      expected_status = _bool_value(_first_value(lead_reference, "status"))
      actual_status = _actual_radar_status(row, slot, field_stem)
      if expected_status is None:
        exclusions["missing_expected_radar_status"] += 1
        row_scorable = False
        continue
      if actual_status is None:
        exclusions["missing_actual_radar_status"] += 1
        row_scorable = False
        continue

      status_matches.append(actual_status == expected_status)
      if not expected_status:
        continue
      active_lead_candidate_count += 1
      if not actual_status:
        exclusions["actual_radar_lead_absent"] += 1
        row_scorable = False
        continue

      expected_d_rel = _finite_float(_first_value(lead_reference, "dRelM", "dRel", "d_rel_m"))
      actual_d_rel = _finite_float(_first_value(row, f"{field_stem}_published_d_rel_m"))
      if expected_d_rel is None:
        exclusions["missing_expected_radar_d_rel"] += 1
        row_scorable = False
      elif actual_d_rel is None:
        exclusions["missing_actual_radar_d_rel"] += 1
        row_scorable = False
      else:
        d_rel_errors.append(abs(actual_d_rel - expected_d_rel))

      expected_v_rel = _finite_float(_first_value(lead_reference, "vRelMps", "vRel", "v_rel_mps"))
      actual_v_rel = _finite_float(_first_value(row, f"{field_stem}_published_v_rel_mps"))
      if expected_v_rel is None:
        exclusions["missing_expected_radar_v_rel"] += 1
        row_scorable = False
      elif actual_v_rel is None:
        exclusions["missing_actual_radar_v_rel"] += 1
        row_scorable = False
      else:
        v_rel_errors.append(abs(actual_v_rel - expected_v_rel))

    scorable_mask[-1] = row_scorable

  status_agreement = _agreement(status_matches)
  d_rel_stats = _error_stats(d_rel_errors)
  v_rel_stats = _error_stats(v_rel_errors)
  quality_checks: dict[str, bool | None] = {
    "status_agreement": _minimum_check(status_agreement, thresholds.status_agreement_min),
    "d_rel_mae": _maximum_check(d_rel_stats["mae"], thresholds.d_rel_mae_max_m),
    "d_rel_p95": _maximum_check(d_rel_stats["p95"], thresholds.d_rel_p95_max_m),
    "d_rel_max": _maximum_check(d_rel_stats["max"], thresholds.d_rel_abs_max_m),
    "v_rel_mae": _maximum_check(v_rel_stats["mae"], thresholds.v_rel_mae_max_mps),
    "v_rel_p95": _maximum_check(v_rel_stats["p95"], thresholds.v_rel_p95_max_mps),
    "v_rel_max": _maximum_check(v_rel_stats["max"], thresholds.v_rel_abs_max_mps),
  }
  coverage = _coverage_summary(rows, scorable_mask, max_contiguous_gap_s=thresholds.max_contiguous_gap_s)
  coverage_checks = _coverage_checks(
    coverage,
    min_scorable_samples=thresholds.min_scorable_samples,
    min_scorable_fraction=thresholds.min_scorable_fraction,
    min_contiguous_duration_s=thresholds.min_contiguous_duration_s,
  )
  coverage = {
    **coverage,
    "active_lead_candidate_samples": active_lead_candidate_count,
    "d_rel_kinematic_samples": len(d_rel_errors),
    "v_rel_kinematic_samples": len(v_rel_errors),
  }
  coverage_checks = {
    **coverage_checks,
    "min_d_rel_kinematic_samples": len(d_rel_errors) >= thresholds.min_kinematic_samples,
    "min_v_rel_kinematic_samples": len(v_rel_errors) >= thresholds.min_kinematic_samples,
  }
  checks = {**quality_checks, **coverage_checks}
  status = _stage_status_with_coverage(quality_checks.values(), coverage_checks.values())
  gate_reasons = _quality_reasons(quality_checks) + _coverage_reasons(
    coverage,
    coverage_checks,
    min_scorable_samples=thresholds.min_scorable_samples,
    min_scorable_fraction=thresholds.min_scorable_fraction,
    min_contiguous_duration_s=thresholds.min_contiguous_duration_s,
  )
  if coverage_checks["min_d_rel_kinematic_samples"] is not True:
    gate_reasons.append(
      f"dRel kinematic samples {len(d_rel_errors)} below minimum {thresholds.min_kinematic_samples}"
    )
  if coverage_checks["min_v_rel_kinematic_samples"] is not True:
    gate_reasons.append(
      f"vRel kinematic samples {len(v_rel_errors)} below minimum {thresholds.min_kinematic_samples}"
    )
  return {
    "status": status,
    "passed": status == PASS,
    "sample_counts": {
      "status": len(status_matches),
      "d_rel": len(d_rel_errors),
      "v_rel": len(v_rel_errors),
    },
    "excluded_sample_count": sum(exclusions.values()),
    "exclusion_reasons": dict(sorted(exclusions.items())),
    "coverage": coverage,
    "gate_reasons": gate_reasons,
    "metrics": {
      "status_agreement": status_agreement,
      "d_rel_error_m": d_rel_stats,
      "v_rel_error_mps": v_rel_stats,
    },
    "checks": checks,
  }


def _actual_radar_status(row: Mapping[str, Any], slot: str, field_stem: str) -> bool | None:
  explicit = _bool_value(_first_value(
    row,
    f"{field_stem}_published_status",
    f"radar_{field_stem}_status",
  ))
  if explicit is not None:
    return explicit

  for container_key in ("radarState", "radar_state"):
    container = row.get(container_key)
    lead = container.get(slot) if isinstance(container, Mapping) else None
    nested = _bool_value(_first_value(lead, "status")) if isinstance(lead, Mapping) else None
    if nested is not None:
      return nested

  d_rel_key = f"{field_stem}_published_d_rel_m"
  v_rel_key = f"{field_stem}_published_v_rel_mps"
  if d_rel_key in row or v_rel_key in row:
    return _finite_float(row.get(d_rel_key)) is not None or _finite_float(row.get(v_rel_key)) is not None

  # Compatibility for prospective compact trace producers. The full
  # run_harness trace takes the published-field branch above.
  return _bool_value(_first_value(row, f"{field_stem}_status"))


def _evaluate_planner(
  rows: list[Mapping[str, Any]],
  thresholds: PlannerFidelityThresholds,
  *,
  require_exact_contract: bool = True,
  diagnostic: bool = False,
  planner_runtime_gate_eligible: bool = True,
  planner_state_restoration_verified: bool = False,
) -> dict[str, Any]:
  exclusions: Counter[str] = Counter()
  association_statuses: Counter[str] = Counter()
  context_statuses: Counter[str] = Counter()
  eligible_count = 0
  samples: list[dict[str, Any]] = []
  scorable_mask: list[bool] = []

  for row in rows:
    scorable_mask.append(False)
    reference = row.get("replay_reference")
    if not isinstance(reference, Mapping):
      exclusions["missing_replay_reference"] += 1
      continue

    plan_reference = reference.get("longitudinalPlan")
    plan_mapping = plan_reference if isinstance(plan_reference, Mapping) else {}
    association = _planner_status(
      reference,
      plan_mapping,
      top_level_keys=("plannerRadarResolution", "plannerRadarAssociationStatus"),
      nested_keys=(
        "plannerRadarResolution",
        "radarAssociationStatus",
        "associationStatus",
        "radarAssociation",
        "association",
      ),
    )
    context = _planner_status(
      reference,
      plan_mapping,
      top_level_keys=("plannerContextStatus", "plannerContext"),
      nested_keys=("plannerContextStatus", "contextStatus", "plannerContext", "context"),
    )
    if diagnostic and isinstance(plan_reference, Mapping):
      association, context = _diagnostic_planner_statuses(row, reference, association, context)
    association_statuses[association or "missing"] += 1
    context_statuses[context or "missing"] += 1

    association_reason = _planner_association_exclusion(
      association,
      accepted=_DIAGNOSTIC_PLANNER_ASSOCIATION_ACCEPTED if diagnostic else _PLANNER_ASSOCIATION_ACCEPTED,
    )
    context_reason = _planner_context_exclusion(
      context,
      accepted=_DIAGNOSTIC_PLANNER_CONTEXT_ACCEPTED if diagnostic else frozenset(("exact",)),
    )
    if association_reason is not None or context_reason is not None:
      if association_reason is not None:
        exclusions[association_reason] += 1
      if context_reason is not None:
        exclusions[context_reason] += 1
      continue

    state_initialization = reference.get("plannerStateInitializationProvenance")
    if require_exact_contract and not well_formed_planner_state_initialization_claim(state_initialization):
      state_status = _status_token(state_initialization)
      exclusion = (
        "missing_planner_state_initialization"
        if state_status in (None, "missing", "unavailable", "not_available")
        else "inexact_planner_state_initialization"
      )
      exclusions[exclusion] += 1
      continue

    if require_exact_contract and not planner_state_restoration_verified:
      exclusions["unverified_planner_state_restoration"] += 1
      continue

    if require_exact_contract and not planner_runtime_gate_eligible:
      exclusions["inexact_planner_runtime_provenance"] += 1
      continue

    if require_exact_contract and not _planner_v1_contract_exact(
      reference,
      plan_mapping,
      planner_state_restoration_verified=planner_state_restoration_verified,
    ):
      exclusions["inexact_planner_replay_contract"] += 1
      continue

    eligible_count += 1
    row_reasons: list[str] = []
    if not isinstance(plan_reference, Mapping):
      row_reasons.append("missing_longitudinal_plan_reference")
      plan_mapping = {}

    expected_a_target = _finite_float(_first_value(plan_mapping, "aTargetMps2", "aTarget", "a_target_mps2"))
    actual_a_target = _finite_float(_first_value(row, "planner_accel_mps2", "planner_a_target_mps2"))
    expected_source = _source_value(_first_value(plan_mapping, "source"))
    actual_source = _source_value(_first_value(row, "planner_source"))
    sample_time = _row_time_s(row)

    if expected_a_target is None:
      row_reasons.append("missing_expected_planner_a_target")
    if actual_a_target is None:
      row_reasons.append("missing_actual_planner_a_target")
    if expected_source is None:
      row_reasons.append("missing_expected_planner_source")
    if actual_source is None:
      row_reasons.append("missing_actual_planner_source")
    if sample_time is None:
      row_reasons.append("missing_planner_sample_time")
    if row_reasons:
      exclusions.update(row_reasons)
      continue

    samples.append({
      "time_s": sample_time,
      "expected_a_target_mps2": expected_a_target,
      "actual_a_target_mps2": actual_a_target,
      "expected_source": expected_source,
      "actual_source": actual_source,
    })
    scorable_mask[-1] = True

  a_target_errors = [
    abs(sample["actual_a_target_mps2"] - sample["expected_a_target_mps2"])
    for sample in samples
  ]
  source_matches = [sample["actual_source"] == sample["expected_source"] for sample in samples]
  a_target_stats = _error_stats(a_target_errors)
  source_agreement = _agreement(source_matches)
  expected_transitions = _source_transitions(samples, "expected_source")
  actual_transitions = _source_transitions(samples, "actual_source")
  expected_sequence = [(transition["from"], transition["to"]) for transition in expected_transitions]
  actual_sequence = [(transition["from"], transition["to"]) for transition in actual_transitions]
  transition_sequence_match = expected_sequence == actual_sequence
  if transition_sequence_match:
    transition_timing_errors = [
      abs(actual["time_s"] - expected["time_s"])
      for expected, actual in zip(expected_transitions, actual_transitions, strict=True)
    ]
    transition_timing_max = max(transition_timing_errors, default=0.0)
    transition_timing_within_tolerance = transition_timing_max <= thresholds.transition_timing_max_s + 1e-12
  else:
    transition_timing_errors = []
    transition_timing_max = None
    transition_timing_within_tolerance = False

  has_samples = bool(samples)
  quality_checks: dict[str, bool | None] = {
    "a_target_mae": _maximum_check(a_target_stats["mae"], thresholds.a_target_mae_max_mps2),
    "a_target_p95": _maximum_check(a_target_stats["p95"], thresholds.a_target_p95_max_mps2),
    "a_target_max": _maximum_check(a_target_stats["max"], thresholds.a_target_abs_max_mps2),
    "source_agreement": _minimum_check(source_agreement, thresholds.source_agreement_min),
    "source_transition_sequence": transition_sequence_match if has_samples else None,
    "source_transition_timing": transition_timing_within_tolerance if has_samples else None,
  }
  coverage = _coverage_summary(rows, scorable_mask, max_contiguous_gap_s=thresholds.max_contiguous_gap_s)
  coverage_checks = _coverage_checks(
    coverage,
    min_scorable_samples=thresholds.min_scorable_samples,
    min_scorable_fraction=thresholds.min_scorable_fraction,
    min_contiguous_duration_s=thresholds.min_contiguous_duration_s,
  )
  checks = {**quality_checks, **coverage_checks}
  status = _stage_status_with_coverage(quality_checks.values(), coverage_checks.values())
  gate_reasons = _quality_reasons(quality_checks) + _coverage_reasons(
    coverage,
    coverage_checks,
    min_scorable_samples=thresholds.min_scorable_samples,
    min_scorable_fraction=thresholds.min_scorable_fraction,
    min_contiguous_duration_s=thresholds.min_contiguous_duration_s,
  )
  return {
    "status": status,
    "passed": status == PASS,
    "sample_counts": {
      "candidate": len(rows),
      "eligible": eligible_count,
      "scorable": len(samples),
    },
    "excluded_row_count": len(rows) - len(samples),
    "exclusion_reasons": dict(sorted(exclusions.items())),
    "association_status_counts": dict(sorted(association_statuses.items())),
    "context_status_counts": dict(sorted(context_statuses.items())),
    "coverage": coverage,
    "gate_reasons": gate_reasons,
    "metrics": {
      "a_target_error_mps2": a_target_stats,
      "source_agreement": source_agreement,
      "source_transitions": {
        "expected": expected_transitions,
        "actual": actual_transitions,
        "sequence_match": transition_sequence_match if has_samples else None,
        "timing_errors_s": transition_timing_errors,
        "timing_max_error_s": transition_timing_max,
        "timing_within_tolerance": transition_timing_within_tolerance if has_samples else None,
      },
    },
    "checks": checks,
  }


def _planner_status(
  reference: Mapping[str, Any],
  plan_reference: Mapping[str, Any],
  *,
  top_level_keys: tuple[str, ...],
  nested_keys: tuple[str, ...],
) -> str | None:
  for key in top_level_keys:
    if key in reference:
      return _status_token(reference.get(key))
  for key in nested_keys:
    if key in plan_reference:
      return _status_token(plan_reference.get(key))
  return None


def _status_token(value: Any) -> str | None:
  if isinstance(value, Mapping):
    for key in ("status", "resolution", "value", "kind"):
      if key in value:
        return _status_token(value.get(key))
    return None
  if value is None or isinstance(value, bool):
    return None
  text = str(value).strip()
  if not text:
    return None
  text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
  return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _planner_association_exclusion(
  status: str | None,
  *,
  accepted: frozenset[str] = _PLANNER_ASSOCIATION_ACCEPTED,
) -> str | None:
  if status in accepted:
    return None
  return _status_exclusion_reason(status, "planner_radar_resolution")


def _planner_context_exclusion(
  status: str | None,
  *,
  accepted: frozenset[str] = frozenset(("exact",)),
) -> str | None:
  if status in accepted:
    return None
  return _status_exclusion_reason(status, "planner_context_status")


def _diagnostic_planner_statuses(
  row: Mapping[str, Any],
  reference: Mapping[str, Any],
  association: str | None,
  context: str | None,
) -> tuple[str | None, str | None]:
  traced_association = _status_token(row.get("planner_radar_resolution"))
  if traced_association in _DIAGNOSTIC_PLANNER_ASSOCIATION_ACCEPTED:
    association = traced_association
  elif (
    _known_ambiguous_status(traced_association)
    and isinstance(reference.get("radarStateDiagnostic", reference.get("radarState")), Mapping)
  ):
    association = "legacy_ambiguous_synchronous"
  elif (
    association in (None, "missing", "untracked", "unscorable")
    and isinstance(reference.get("radarStateDiagnostic", reference.get("radarState")), Mapping)
  ):
    association = "legacy_synchronous"

  context_reason = str(reference.get("plannerContextReason", "")).lower()
  unsafe_context = "object hazard" in context_reason or "objecthazard" in context_reason
  if (
    not unsafe_context
    and context in (None, "untracked")
  ):
    context = "diagnostic_derived"
  return association, context


def _known_ambiguous_status(status: str | None) -> bool:
  return status is not None and any(part in status for part in _AMBIGUOUS_TOKEN_PARTS)


def _radard_contract_exact(reference: Mapping[str, Any]) -> bool:
  if reference.get("radardGateEligible") is not True:
    return False
  provenance = reference.get("radardServiceAssociationProvenance")
  if not isinstance(provenance, Mapping):
    return False
  contract = provenance.get("contract")
  if not is_exact_supported_radard_replay_contract(contract):
    return False
  for service in ("modelV2", "carState", "liveTracks"):
    service_provenance = provenance.get(service)
    if not isinstance(service_provenance, Mapping) or service_provenance.get("status") != "exact":
      return False
  live_tracks = provenance.get("liveTracks")
  return isinstance(live_tracks, Mapping) and live_tracks.get("emptyPayloadValid") is True


def _planner_v1_contract_exact(
  reference: Mapping[str, Any],
  plan_reference: Mapping[str, Any],
  *,
  planner_state_restoration_verified: bool,
) -> bool:
  fidelity = reference.get("plannerFidelity")
  if not isinstance(fidelity, Mapping) or fidelity.get("scorable") is not True:
    return False
  if not exact_planner_state_initialization(
    reference.get("plannerStateInitializationProvenance"),
    restoration_verified=planner_state_restoration_verified,
  ):
    return False
  provenance = reference.get("plannerServiceAssociationProvenance")
  if not isinstance(provenance, Mapping):
    return False
  contract = provenance.get("contract")
  if not isinstance(contract, Mapping) or contract.get("status") != "exact" or contract.get("version") != 1:
    return False
  for service in ("carState", "controlsState", "carControl", "selfdriveState", "modelV2", "radarState"):
    service_provenance = provenance.get(service)
    if not isinstance(service_provenance, Mapping) or service_provenance.get("status") != "exact":
      return False
  params_provenance = provenance.get("params")
  if not exact_replay_param_manifest(params_provenance):
    return False

  target_clock = plan_reference.get("radarStateLogMonoTimeNs")
  candidates = plan_reference.get("radarStateCandidatesNs")
  service_clocks = reference.get("plannerServiceLogMonoTimeNs")
  if (
    not isinstance(target_clock, int)
    or isinstance(target_clock, bool)
    or target_clock <= 0
    or not isinstance(candidates, list)
    or candidates != [target_clock]
    or not isinstance(service_clocks, Mapping)
    or service_clocks.get("radarState") != target_clock
  ):
    return False
  radar_provenance = provenance.get("radarState")
  return isinstance(radar_provenance, Mapping) and radar_provenance.get("clockNs") == target_clock


def _status_exclusion_reason(status: str | None, label: str) -> str:
  if status is None:
    return f"missing_{label}"
  if status in _NOT_APPLICABLE_TOKENS:
    return f"not_applicable_{label}"
  if any(part in status for part in _AMBIGUOUS_TOKEN_PARTS):
    return f"ambiguous_{label}"
  return f"inexact_{label}"


def _source_transitions(samples: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
  transitions: list[dict[str, Any]] = []
  if not samples:
    return transitions
  previous = str(samples[0][key])
  for sample in samples[1:]:
    current = str(sample[key])
    if current != previous:
      transitions.append({
        "time_s": float(sample["time_s"]),
        "from": previous,
        "to": current,
      })
      previous = current
  return transitions


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
  for key in keys:
    if key in mapping:
      return mapping[key]
  return _MISSING


def _finite_float(value: Any) -> float | None:
  if value is _MISSING or value is None or isinstance(value, bool):
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _bool_value(value: Any) -> bool | None:
  if value is _MISSING or value is None:
    return None
  if isinstance(value, bool):
    return value
  if isinstance(value, int) and value in (0, 1):
    return bool(value)
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
      return True
    if normalized in ("false", "0", "no"):
      return False
  return None


def _source_value(value: Any) -> str | None:
  if value is _MISSING or value is None:
    return None
  source = str(value).strip()
  return source or None


def _agreement(matches: list[bool]) -> float | None:
  if not matches:
    return None
  return sum(bool(match) for match in matches) / len(matches)


def _error_stats(errors: list[float]) -> dict[str, float | int | None]:
  if not errors:
    return {"count": 0, "mae": None, "p95": None, "max": None}
  return {
    "count": len(errors),
    "mae": sum(errors) / len(errors),
    "p95": _percentile(errors, 0.95),
    "max": max(errors),
  }


def _percentile(values: list[float], quantile: float) -> float:
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  position = (len(ordered) - 1) * quantile
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _minimum_check(value: float | None, threshold: float) -> bool | None:
  return None if value is None else value + 1e-12 >= threshold


def _maximum_check(value: float | None, threshold: float) -> bool | None:
  return None if value is None else value <= threshold + 1e-12


def _stage_status(checks: Iterable[bool | None]) -> str:
  checks = list(checks)
  if any(check is False for check in checks):
    return FAIL
  if not checks or any(check is None for check in checks):
    return NOT_EVALUATED
  return PASS


__all__ = [
  "DEFAULT_THRESHOLDS",
  "FAIL",
  "FidelityThresholds",
  "NOT_EVALUATED",
  "PASS",
  "PlannerFidelityThresholds",
  "RadarFidelityThresholds",
  "evaluate_fidelity",
  "evaluate_harness_fidelity",
  "evaluate_diagnostic_fidelity",
]
