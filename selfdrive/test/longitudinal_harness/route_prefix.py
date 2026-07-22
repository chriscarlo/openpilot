from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any

from openpilot.selfdrive.test.longitudinal_harness.closed_loop import run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import captured_param_manifest
from openpilot.selfdrive.test.longitudinal_harness.fidelity import (
  FAIL,
  NOT_EVALUATED,
  PASS,
  evaluate_diagnostic_fidelity,
  evaluate_fidelity,
  evaluate_harness_fidelity,
)
from openpilot.selfdrive.test.longitudinal_harness.fidelity_runner import resolve_fidelity_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import SnapshotBundle


_RECURRENT_DEBUG_FIELDS = (
  "mpc_acc_source_debug",
  "mpc_lead_role_debug",
  "mpc_steady_parity_debug",
  "mpc_cutin_settle_debug",
  "mpc_lead_preview_debug",
  "planner_lead_brake_release_debug",
  "planner_cruise_reacquire_debug",
  "planner_relatch_blend_debug",
  "planner_handoff_limit_debug",
  "planner_comfort_jerk_debug",
  "planner_steady_parity_threat_debug",
  "mpc_adjacent_awareness_preview_debug",
  "mpc_hyundai_virtual_lead_debug",
)


@dataclass(frozen=True)
class MarkerWindow:
  marker_id: str
  marker_log_mono_time_ns: int
  start_before_s: float = 10.0
  end_before_s: float = 0.2
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if isinstance(self.marker_log_mono_time_ns, bool) or self.marker_log_mono_time_ns <= 0:
      raise ValueError("marker_log_mono_time_ns must be a positive integer")
    if not math.isfinite(self.start_before_s) or not math.isfinite(self.end_before_s):
      raise ValueError("marker window offsets must be finite")
    if self.start_before_s <= self.end_before_s or self.end_before_s < 0.0:
      raise ValueError("marker window requires start_before_s > end_before_s >= 0")

  @property
  def start_log_mono_time_ns(self) -> int:
    return self.marker_log_mono_time_ns - int(round(self.start_before_s * 1e9))

  @property
  def end_log_mono_time_ns(self) -> int:
    return self.marker_log_mono_time_ns - int(round(self.end_before_s * 1e9))


@dataclass(frozen=True)
class RecurrentDebugThresholds:
  min_scorable_samples: int = 20
  min_scorable_fraction: float = 0.95
  min_contiguous_duration_s: float = 0.75
  max_contiguous_gap_s: float = 0.075
  agreement_fraction_min: float = 0.98
  numeric_abs_tolerance: float = 1e-6


def load_marker_windows(
  path: str | Path,
  *,
  collection_key: str = "taps",
  timestamp_key: str = "tapMonoNs",
  qualification_key: str | None = "qualifiedActiveLead",
  label_key: str | None = "tapPdt",
  start_before_s: float = 10.0,
  end_before_s: float = 0.2,
) -> list[MarkerWindow]:
  """Load marker windows without coupling the runner to a route or report schema.

  The defaults consume the brake-tap reducer's JSON, while callers may name
  different collection/timestamp/qualification keys for another route corpus.
  """
  payload = json.loads(Path(path).read_text())
  items = payload.get(collection_key) if isinstance(payload, Mapping) else None
  if not isinstance(items, list):
    raise ValueError(f"marker report must contain a list at '{collection_key}'")

  markers = []
  for index, item in enumerate(items):
    if not isinstance(item, Mapping):
      raise ValueError(f"marker item {index} is not an object")
    if qualification_key is not None and item.get(qualification_key) is not True:
      continue
    timestamp = item.get(timestamp_key)
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
      raise ValueError(f"qualified marker item {index} has no positive integer '{timestamp_key}'")
    label_value = item.get(label_key) if label_key is not None else None
    marker_id = str(label_value) if label_value not in (None, "") else f"marker_{index:03d}"
    markers.append(MarkerWindow(
      marker_id=marker_id,
      marker_log_mono_time_ns=timestamp,
      start_before_s=start_before_s,
      end_before_s=end_before_s,
      metadata=dict(item),
    ))
  if not markers:
    raise ValueError("marker report contains no qualified markers")
  return markers


def select_marker_trace_rows(
  trace_rows: Iterable[Mapping[str, Any]],
  marker: MarkerWindow,
  *,
  pre_roll_s: float | None = None,
) -> list[Mapping[str, Any]]:
  if pre_roll_s is None:
    start_ns = marker.start_log_mono_time_ns
    end_ns = marker.end_log_mono_time_ns
    include_end = True
  else:
    if not math.isfinite(pre_roll_s) or pre_roll_s <= 0.0:
      raise ValueError("pre_roll_s must be finite and positive")
    end_ns = marker.start_log_mono_time_ns
    start_ns = end_ns - int(round(pre_roll_s * 1e9))
    include_end = False

  selected = []
  for row in trace_rows:
    timestamp = _reference_time_ns(row)
    if timestamp is None or timestamp < start_ns:
      continue
    if timestamp < end_ns or (include_end and timestamp == end_ns):
      selected.append(row)
  return selected


def evaluate_recurrent_debug_convergence(
  trace_rows: Iterable[Mapping[str, Any]],
  *,
  thresholds: RecurrentDebugThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Compare replay-private recurrent debug against an explicit recorded oracle.

  Current writer-v1 rlogs do not carry ``plannerRecurrentDebug``. That absence
  is intentionally NOT_EVALUATED, never inferred from source/aTarget agreement.
  A future writer may put a mapping at
  ``replay_reference.plannerRecurrentDebug``; expected mappings may be partial,
  but every named value must agree with the corresponding replay debug value.
  """
  resolved = _coerce_recurrent_thresholds(thresholds)
  rows = _unique_reference_rows(trace_rows)
  scorable: list[tuple[int, bool]] = []
  exclusions: Counter[str] = Counter()
  compared_fields: set[str] = set()
  for row in rows:
    reference = row.get("replay_reference")
    expected = reference.get("plannerRecurrentDebug") if isinstance(reference, Mapping) else None
    if not isinstance(expected, Mapping) or not expected:
      exclusions["missing_recorded_recurrent_debug"] += 1
      continue
    actual = {key: row.get(key) for key in _RECURRENT_DEBUG_FIELDS}
    missing = [key for key in expected if key not in actual or actual[key] is None]
    if missing:
      exclusions["missing_replayed_recurrent_debug"] += 1
      continue
    timestamp = _reference_time_ns(row)
    if timestamp is None:
      exclusions["missing_reference_time"] += 1
      continue
    compared_fields.update(str(key) for key in expected)
    agrees = _expected_subset_agrees(expected, actual, abs_tolerance=resolved.numeric_abs_tolerance)
    scorable.append((timestamp, agrees))

  candidate_count = len(rows)
  scorable_count = len(scorable)
  recorded_oracle_count = candidate_count - exclusions.get("missing_recorded_recurrent_debug", 0)
  scorable_fraction = scorable_count / candidate_count if candidate_count else None
  agreement_fraction = sum(agrees for _, agrees in scorable) / scorable_count if scorable_count else None
  longest_samples, longest_duration_s, max_gap_s = _contiguous_coverage(
    [timestamp for timestamp, _ in scorable],
    max_gap_s=resolved.max_contiguous_gap_s,
  )
  checks = {
    "min_scorable_samples": None if not recorded_oracle_count else scorable_count >= resolved.min_scorable_samples,
    "min_scorable_fraction": (
      None if not recorded_oracle_count or scorable_fraction is None
      else scorable_fraction >= resolved.min_scorable_fraction
    ),
    "min_contiguous_duration": None if not recorded_oracle_count else longest_duration_s >= resolved.min_contiguous_duration_s,
    "agreement_fraction": (
      None if not recorded_oracle_count or agreement_fraction is None
      else agreement_fraction >= resolved.agreement_fraction_min
    ),
  }
  status = _checks_status(checks.values())
  reasons = []
  if status != PASS:
    if exclusions.get("missing_recorded_recurrent_debug"):
      reasons.append(
        "capture has no plannerRecurrentDebug oracle; writer-v1 external inputs cannot prove hidden recurrent state"
      )
    reasons.extend(f"check '{name}' is {value}" for name, value in checks.items() if value is not True)
  return {
    "status": status,
    "passed": status == PASS,
    "gateEligible": False,
    "sampleCounts": {"candidate": candidate_count, "scorable": scorable_count},
    "exclusionReasons": dict(sorted(exclusions.items())),
    "comparedFields": sorted(compared_fields),
    "coverage": {
      "scorableFraction": scorable_fraction,
      "longestContiguousRunSamples": longest_samples,
      "longestContiguousDurationS": longest_duration_s,
      "maxObservedGapS": max_gap_s,
    },
    "agreementFraction": agreement_fraction,
    "checks": checks,
    "reasons": reasons,
    "thresholds": asdict(resolved),
  }


def evaluate_route_prefix_trace(
  trace_rows: Iterable[Mapping[str, Any]],
  markers: Iterable[MarkerWindow],
  *,
  captured_metadata: Mapping[str, Any] | None,
  replay_metadata: Mapping[str, Any] | None,
  provenance_mode: str = "exact",
  thresholds: Any = None,
  convergence_pre_roll_s: float = 2.0,
  recurrent_thresholds: RecurrentDebugThresholds | Mapping[str, Any] | None = None,
  simulation_result: Any | None = None,
) -> dict[str, Any]:
  if provenance_mode not in ("exact", "instrumentation_only", "counterfactual"):
    raise ValueError(f"unsupported provenance_mode '{provenance_mode}'")
  rows = [row for row in trace_rows if isinstance(row, Mapping)]
  marker_list = list(markers)
  if not marker_list:
    raise ValueError("route-prefix evaluation requires at least one marker")

  window_results = []
  for marker in marker_list:
    window_rows = select_marker_trace_rows(rows, marker)
    pre_roll_rows = select_marker_trace_rows(rows, marker, pre_roll_s=convergence_pre_roll_s)
    fidelity_kwargs = {
      "thresholds": thresholds,
      "captured_metadata": captured_metadata,
      "replay_metadata": replay_metadata,
      "counterfactual": provenance_mode == "counterfactual",
      "acknowledge_instrumentation_only": provenance_mode == "instrumentation_only",
    }
    formal = (
      evaluate_harness_fidelity(replace(simulation_result, trace=list(window_rows)), **fidelity_kwargs)
      if simulation_result is not None else
      evaluate_fidelity(window_rows, **fidelity_kwargs)
    )
    diagnostic = evaluate_diagnostic_fidelity(window_rows, thresholds=thresholds)
    pre_roll_diagnostic = evaluate_diagnostic_fidelity(pre_roll_rows, thresholds=thresholds)
    planner = pre_roll_diagnostic["planner"]
    source_convergence = _planner_check_group(
      planner,
      ("source_agreement", "source_transition_sequence", "source_transition_timing"),
    )
    accel_convergence = _planner_check_group(
      planner,
      ("a_target_mae", "a_target_p95", "a_target_max"),
    )
    recurrent_convergence = evaluate_recurrent_debug_convergence(
      pre_roll_rows,
      thresholds=recurrent_thresholds,
    )
    convergence_status = _combined_status(
      source_convergence["status"],
      accel_convergence["status"],
      recurrent_convergence["status"],
    )
    formal_gate_eligible = bool(formal["overall"]["passed"])
    window_status = formal["status"] if simulation_result is not None else _combined_status(diagnostic["status"], convergence_status)
    selection_status = "selected" if window_rows else NOT_EVALUATED
    window_results.append({
      "marker": {
        "id": marker.marker_id,
        "logMonoTimeNs": marker.marker_log_mono_time_ns,
        "metadata": dict(marker.metadata),
      },
      "window": {
        "startLogMonoTimeNs": marker.start_log_mono_time_ns,
        "endLogMonoTimeNs": marker.end_log_mono_time_ns,
        "startBeforeS": marker.start_before_s,
        "endBeforeS": marker.end_before_s,
        "selectionStatus": selection_status,
        "traceRowCount": len(window_rows),
      },
      "status": window_status,
      "gateEligible": formal_gate_eligible,
      "formalFidelity": formal,
      "diagnosticFidelity": diagnostic,
      "convergence": {
        "status": convergence_status,
        "gateEligible": False,
        "preRollS": convergence_pre_roll_s,
        "traceRowCount": len(pre_roll_rows),
        "source": source_convergence,
        "aTarget": accel_convergence,
        "recurrentDebug": recurrent_convergence,
      },
    })

  statuses = Counter(result["status"] for result in window_results)
  diagnostic_statuses = Counter(result["diagnosticFidelity"]["status"] for result in window_results)
  formal_planner_statuses = Counter(result["formalFidelity"]["planner"]["status"] for result in window_results)
  recurrent_statuses = Counter(result["convergence"]["recurrentDebug"]["status"] for result in window_results)
  formal_mode = simulation_result is not None
  return {
    "schemaVersion": 1,
    "mode": "route_prefix_formal" if formal_mode else "route_prefix_diagnostic_non_gating",
    "status": _combined_status(*(result["status"] for result in window_results)),
    "gateEligible": bool(formal_mode and window_results and all(result["gateEligible"] for result in window_results)),
    "runCount": 1,
    "traceRowCount": len(rows),
    "uniqueReferenceRowCount": len(_unique_reference_rows(rows)),
    "prefixCoverage": _prefix_coverage(rows),
    "markerCount": len(marker_list),
    "statusCounts": dict(sorted(statuses.items())),
    "diagnosticStatusCounts": dict(sorted(diagnostic_statuses.items())),
    "formalPlannerStatusCounts": dict(sorted(formal_planner_statuses.items())),
    "recurrentDebugStatusCounts": dict(sorted(recurrent_statuses.items())),
    "formalFidelityCaveat": (
      "Formal planner replay requires a writer-attested complete input stream and a restorable planner checkpoint."
      if formal_mode else
      "Route-start replay remains diagnostic because rlogs do not attest publication completeness; a writer checkpoint is required."
    ),
    "windows": window_results,
  }


def run_route_prefix_convergence(
  bundle: SnapshotBundle,
  markers: Iterable[MarkerWindow],
  *,
  replay_metadata: Mapping[str, Any],
  provenance_mode: str = "exact",
  thresholds: Any = None,
  convergence_pre_roll_s: float = 2.0,
  recurrent_thresholds: RecurrentDebugThresholds | Mapping[str, Any] | None = None,
  seed: int = 42,
) -> dict[str, Any]:
  """Warm one harness instance over a route prefix and score every marker window."""
  marker_list = list(markers)
  coverage_error = _bundle_marker_coverage_error(bundle, marker_list, convergence_pre_roll_s)
  if coverage_error is not None:
    return build_not_evaluated_route_prefix_report(
      marker_list,
      reason=coverage_error,
      scenario_name=bundle.name or bundle.path.name,
      bundle_path=bundle.path,
    )

  vehicle_config = resolve_fidelity_vehicle_config(bundle)
  simulation = run_harness(
    vehicle_config=vehicle_config,
    scenario_name=bundle.name or bundle.path.name,
    steps=bundle.timeline,
    initial_speed_mps=bundle.initial_speed_mps,
    initial_accel_mps2=bundle.initial_accel_mps2,
    noise_profile="off",
    seed=seed,
    perception_filter="auto",
    ego_replay_mode="auto",
  )
  formal_simulation = simulation if getattr(simulation, "planner_state_restoration_verified", False) is True else None
  report = evaluate_route_prefix_trace(
    simulation.trace,
    marker_list,
    captured_metadata=bundle.vehicle,
    replay_metadata=replay_metadata,
    provenance_mode=provenance_mode,
    thresholds=thresholds,
    convergence_pre_roll_s=convergence_pre_roll_s,
    recurrent_thresholds=recurrent_thresholds,
    simulation_result=formal_simulation,
  )
  report.update({
    "scenarioName": bundle.name or bundle.path.name,
    "bundlePath": str(bundle.path),
    "capturedParamManifest": captured_param_manifest(bundle.params),
    "replayProvenance": dict(replay_metadata),
    "summary": simulation.summary,
  })
  return report


def build_not_evaluated_route_prefix_report(
  markers: Iterable[MarkerWindow],
  *,
  reason: str,
  scenario_name: str = "",
  bundle_path: str | Path | None = None,
  prefix_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Describe an artifact-level block without inventing replay observations."""
  marker_list = list(markers)
  unavailable_stage = {
    "status": NOT_EVALUATED,
    "passed": False,
    "gateEligible": False,
    "reasons": [reason],
  }
  windows = []
  for marker in marker_list:
    windows.append({
      "marker": {
        "id": marker.marker_id,
        "logMonoTimeNs": marker.marker_log_mono_time_ns,
        "metadata": dict(marker.metadata),
      },
      "window": {
        "startLogMonoTimeNs": marker.start_log_mono_time_ns,
        "endLogMonoTimeNs": marker.end_log_mono_time_ns,
        "startBeforeS": marker.start_before_s,
        "endBeforeS": marker.end_before_s,
        "selectionStatus": NOT_EVALUATED,
        "traceRowCount": 0,
      },
      "status": NOT_EVALUATED,
      "gateEligible": False,
      "reason": reason,
      "formalFidelity": {
        "status": NOT_EVALUATED,
        "planner": dict(unavailable_stage),
        "overall": dict(unavailable_stage),
      },
      "diagnosticFidelity": {
        "status": NOT_EVALUATED,
        "diagnosticPassed": False,
        "gateEligible": False,
        "reasons": [reason],
        "planner": dict(unavailable_stage),
      },
      "convergence": {
        "status": NOT_EVALUATED,
        "gateEligible": False,
        "traceRowCount": 0,
        "source": dict(unavailable_stage),
        "aTarget": dict(unavailable_stage),
        "recurrentDebug": dict(unavailable_stage),
      },
    })
  return {
    "schemaVersion": 1,
    "mode": "route_prefix_diagnostic_non_gating",
    "status": NOT_EVALUATED,
    "gateEligible": False,
    "runCount": 0,
    "scenarioName": scenario_name,
    "bundlePath": None if bundle_path is None else str(bundle_path),
    "markerCount": len(marker_list),
    "statusCounts": {NOT_EVALUATED: len(marker_list)},
    "diagnosticStatusCounts": {NOT_EVALUATED: len(marker_list)},
    "formalPlannerStatusCounts": {NOT_EVALUATED: len(marker_list)},
    "recurrentDebugStatusCounts": {NOT_EVALUATED: len(marker_list)},
    "prefixCoverage": dict(prefix_coverage or {}),
    "reason": reason,
    "formalFidelityCaveat": (
      "No route-prefix result was inferred because the available artifact cannot produce one complete replay."
    ),
    "windows": windows,
  }


def write_route_prefix_report(report: Mapping[str, Any], path: str | Path) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _planner_check_group(planner: Mapping[str, Any], quality_names: tuple[str, ...]) -> dict[str, Any]:
  checks = planner.get("checks") if isinstance(planner.get("checks"), Mapping) else {}
  coverage_names = (
    "complete_contiguous_window",
    "min_contiguous_duration",
    "min_scorable_fraction",
    "min_scorable_samples",
  )
  selected = {name: checks.get(name) for name in (*quality_names, *coverage_names)}
  metrics = planner.get("metrics") if isinstance(planner.get("metrics"), Mapping) else {}
  metric_subset = {
    "sourceAgreement": metrics.get("source_agreement"),
    "sourceTransitions": metrics.get("source_transitions"),
    "aTargetErrorMps2": metrics.get("a_target_error_mps2"),
  }
  return {
    "status": _checks_status(selected.values()),
    "passed": all(value is True for value in selected.values()),
    "checks": selected,
    "sampleCounts": dict(planner.get("sample_counts", {})),
    "coverage": dict(planner.get("coverage", {})),
    "metrics": metric_subset,
    "gateReasons": list(planner.get("gate_reasons", [])),
  }


def _coerce_recurrent_thresholds(
  value: RecurrentDebugThresholds | Mapping[str, Any] | None,
) -> RecurrentDebugThresholds:
  if value is None:
    resolved = RecurrentDebugThresholds()
  elif isinstance(value, RecurrentDebugThresholds):
    resolved = value
  elif isinstance(value, Mapping):
    resolved = RecurrentDebugThresholds(**dict(value))
  else:
    raise TypeError("recurrent thresholds must be RecurrentDebugThresholds, a mapping, or None")
  if resolved.min_scorable_samples < 1:
    raise ValueError("min_scorable_samples must be positive")
  for name in ("min_scorable_fraction", "agreement_fraction_min"):
    number = getattr(resolved, name)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
      raise ValueError(f"{name} must be within [0, 1]")
  for name in ("min_contiguous_duration_s", "max_contiguous_gap_s", "numeric_abs_tolerance"):
    number = getattr(resolved, name)
    if not math.isfinite(number) or number < 0.0:
      raise ValueError(f"{name} must be finite and non-negative")
  return resolved


def _expected_subset_agrees(expected: Any, actual: Any, *, abs_tolerance: float) -> bool:
  if isinstance(expected, Mapping):
    return isinstance(actual, Mapping) and all(
      key in actual and _expected_subset_agrees(value, actual[key], abs_tolerance=abs_tolerance)
      for key, value in expected.items()
    )
  if isinstance(expected, (list, tuple)):
    return isinstance(actual, (list, tuple)) and len(expected) == len(actual) and all(
      _expected_subset_agrees(left, right, abs_tolerance=abs_tolerance)
      for left, right in zip(expected, actual, strict=True)
    )
  if (
    isinstance(expected, (int, float)) and not isinstance(expected, bool) and
    isinstance(actual, (int, float)) and not isinstance(actual, bool)
  ):
    return math.isfinite(float(expected)) and math.isfinite(float(actual)) and abs(float(expected) - float(actual)) <= abs_tolerance
  return type(expected) is type(actual) and expected == actual


def _unique_reference_rows(trace_rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
  unique = []
  seen: set[int] = set()
  for row in trace_rows:
    timestamp = _reference_time_ns(row)
    if timestamp is not None:
      if timestamp in seen:
        continue
      seen.add(timestamp)
    unique.append(row)
  return unique


def _reference_time_ns(row: Mapping[str, Any]) -> int | None:
  reference = row.get("replay_reference")
  value = reference.get("logMonoTimeNs") if isinstance(reference, Mapping) else None
  return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _contiguous_coverage(timestamps_ns: list[int], *, max_gap_s: float) -> tuple[int, float, float | None]:
  if not timestamps_ns:
    return 0, 0.0, None
  longest_samples = run_samples = 1
  longest_duration_s = 0.0
  run_start = timestamps_ns[0]
  max_gap = 0.0
  previous = timestamps_ns[0]
  for timestamp in timestamps_ns[1:]:
    gap_s = (timestamp - previous) / 1e9
    max_gap = max(max_gap, gap_s)
    if 0.0 < gap_s <= max_gap_s + 1e-12:
      run_samples += 1
    else:
      run_start = timestamp
      run_samples = 1
    run_duration_s = max(0.0, (timestamp - run_start) / 1e9)
    if run_samples > longest_samples or (run_samples == longest_samples and run_duration_s > longest_duration_s):
      longest_samples = run_samples
      longest_duration_s = run_duration_s
    previous = timestamp
  return longest_samples, longest_duration_s, max_gap


def _checks_status(values: Iterable[bool | None]) -> str:
  resolved = list(values)
  if any(value is False for value in resolved):
    return FAIL
  if not resolved or any(value is None for value in resolved):
    return NOT_EVALUATED
  return PASS


def _combined_status(*statuses: str) -> str:
  if any(status == FAIL for status in statuses):
    return FAIL
  if not statuses or any(status != PASS for status in statuses):
    return NOT_EVALUATED
  return PASS


def _prefix_coverage(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
  timestamps = [_reference_time_ns(row) for row in _unique_reference_rows(rows)]
  valid = [timestamp for timestamp in timestamps if timestamp is not None]
  gaps = [(current - previous) / 1e9 for previous, current in zip(valid[:-1], valid[1:], strict=True)]
  return {
    "firstLogMonoTimeNs": valid[0] if valid else None,
    "lastLogMonoTimeNs": valid[-1] if valid else None,
    "validReferenceRowCount": len(valid),
    "missingReferenceRowCount": len(timestamps) - len(valid),
    "maxObservedGapS": max(gaps, default=None),
    "gapCountOver75Ms": sum(gap > 0.075 + 1e-12 for gap in gaps),
  }


def _bundle_marker_coverage_error(
  bundle: SnapshotBundle,
  markers: list[MarkerWindow],
  convergence_pre_roll_s: float,
) -> str | None:
  if not markers:
    return "route-prefix bundle has no marker windows"
  if not bundle.timeline:
    return "route-prefix bundle timeline is empty"
  timestamps = [
    value for value in (
      step.replay_reference.get("logMonoTimeNs")
      if isinstance(step.replay_reference, Mapping) else None
      for step in bundle.timeline
    )
    if isinstance(value, int) and not isinstance(value, bool) and value > 0
  ]
  if not timestamps:
    return "route-prefix bundle has no replay-reference clocks"
  required_start = min(marker.start_log_mono_time_ns for marker in markers) - int(round(convergence_pre_roll_s * 1e9))
  required_end = max(marker.end_log_mono_time_ns for marker in markers)
  if min(timestamps) > required_start:
    return f"route-prefix bundle starts at {min(timestamps)}, after required clean pre-roll {required_start}"
  if max(timestamps) < required_end:
    return f"route-prefix bundle ends at {max(timestamps)}, before required window end {required_end}"
  return None


__all__ = [
  "MarkerWindow",
  "RecurrentDebugThresholds",
  "build_not_evaluated_route_prefix_report",
  "evaluate_recurrent_debug_convergence",
  "evaluate_route_prefix_trace",
  "load_marker_windows",
  "run_route_prefix_convergence",
  "select_marker_trace_rows",
  "write_route_prefix_report",
]
