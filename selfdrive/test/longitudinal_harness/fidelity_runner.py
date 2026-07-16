from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
import hashlib
from pathlib import Path
import platform
import subprocess
from typing import Any

from openpilot.system.hardware import HARDWARE

from openpilot.selfdrive.test.longitudinal_harness.closed_loop import run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import captured_param_manifest, resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.fidelity import (
  evaluate_diagnostic_fidelity,
  evaluate_fidelity,
  evaluate_harness_fidelity,
)
from openpilot.selfdrive.test.longitudinal_harness.inputs import SnapshotBundle, StepInput, load_snapshot_bundle
from openpilot.selfdrive.test.longitudinal_harness.provenance import classify_replay_provenance


def collect_replay_git_metadata(repo_root: str | Path) -> dict[str, Any]:
  """Describe the source revision and numerical runtime used by a replay.

  This intentionally uses the same full tracked-worktree command that updated
  writes into ``GitDiff`` on device. Untracked artifacts are absent from both.
  A scoped diff is not sufficient for an exact gate because it cannot be
  compared byte-for-byte with the capture's full dirty-tree identity.
  """
  root = Path(repo_root).resolve()
  commit = _git_output(root, "rev-parse", "HEAD").decode().strip()
  diff = _git_output(root, "diff", "--submodule=diff")
  runtime_os_version = _read_runtime_file(
    Path("/VERSION"),
    fallback=HARDWARE.get_os_version() or platform.mac_ver()[0] or platform.version(),
  )
  runtime_kernel_version = _read_runtime_file(Path("/proc/version"), fallback=platform.version())
  return {
    "gitCommit": commit,
    "gitDirty": bool(diff),
    "gitDiffEmpty": not bool(diff),
    "gitDiffSha256": hashlib.sha256(diff).hexdigest(),
    "runtimeDeviceType": str(HARDWARE.get_device_type()),
    "runtimePlatform": platform.system().lower(),
    "runtimeMachine": platform.machine().lower(),
    "runtimeOsVersion": runtime_os_version,
    "runtimeKernelVersion": runtime_kernel_version,
  }


def select_evaluation_trace_rows(
  bundle: SnapshotBundle,
  trace_rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
  """Remove dependency warmup while retaining the recorded evaluation window."""
  rows = [row for row in trace_rows if isinstance(row, Mapping)]
  t_start_abs = _finite_float(bundle.vehicle.get("tStartS"))
  evaluation_start_abs = _finite_float(bundle.vehicle.get("evaluationTStartS"))
  evaluation_end_abs = _finite_float(bundle.vehicle.get("tEndS"))
  if t_start_abs is None or evaluation_start_abs is None or evaluation_end_abs is None:
    raise ValueError("snapshot is missing finite tStartS, evaluationTStartS, or tEndS window boundaries")
  if not t_start_abs <= evaluation_start_abs <= evaluation_end_abs:
    raise ValueError("snapshot evaluation window boundaries are not ordered")

  start_rel = max(0.0, evaluation_start_abs - t_start_abs)
  end_rel = None if evaluation_end_abs is None else max(start_rel, evaluation_end_abs - t_start_abs)
  selected = []
  for row in rows:
    row_t = _finite_float(row.get("t_s", row.get("time_s")))
    if row_t is None:
      raise ValueError("harness trace row is missing a finite t_s/time_s timestamp")
    if row_t + 1e-9 < start_rel:
      continue
    if end_rel is not None and row_t - 1e-9 > end_rel:
      continue
    selected.append(row)
  if not selected:
    raise ValueError("no harness trace rows fall inside the recorded evaluation window")
  return selected


def resolve_fidelity_vehicle_config(bundle: SnapshotBundle):
  """Build capture-backed replay config without any developer-device backfill."""
  return resolve_ev6_vehicle_config(
    topology=str(bundle.vehicle.get("topology", "lka")),
    controller_mode="auto",
    tune_source="snapshot",
    snapshot_vehicle=bundle.vehicle,
    snapshot_params=bundle.params,
    # A fidelity replay may use only values captured in this bundle. The
    # developer's July-4 device snapshot belongs to synthetic tuning runs and
    # must never backfill a missing recorded parameter.
    livetune_snapshot=None,
  )


def run_snapshot_fidelity(
  snapshot_path: str | Path,
  *,
  replay_metadata: Mapping[str, Any],
  provenance_mode: str = "exact",
  thresholds: Any = None,
  seed: int = 42,
  diagnostic_scheduler_variants: bool = False,
) -> dict[str, Any]:
  """Run one recorded bundle and return a provenance-aware fidelity result."""
  if provenance_mode not in ("exact", "instrumentation_only", "counterfactual"):
    raise ValueError(f"unsupported provenance_mode '{provenance_mode}'")

  bundle = load_snapshot_bundle(snapshot_path)
  param_manifest = captured_param_manifest(bundle.params)
  provenance = classify_replay_provenance(
    bundle.vehicle,
    replay_metadata,
    counterfactual=provenance_mode == "counterfactual",
    acknowledge_instrumentation_only=provenance_mode == "instrumentation_only",
  )
  vehicle_config = resolve_fidelity_vehicle_config(bundle)
  simulation = run_harness(
    vehicle_config=vehicle_config,
    scenario_name=bundle.name or Path(snapshot_path).name,
    steps=bundle.timeline,
    initial_speed_mps=bundle.initial_speed_mps,
    initial_accel_mps2=bundle.initial_accel_mps2,
    noise_profile="off",
    seed=seed,
    perception_filter="auto",
    ego_replay_mode="auto",
  )
  window_selection: dict[str, Any]
  try:
    evaluation_rows = select_evaluation_trace_rows(bundle, simulation.trace)
    window_selection = {"status": "selected", "reason": None}
  except ValueError as exc:
    evaluation_rows = []
    window_selection = {"status": "not_evaluated", "reason": str(exc)}
  fidelity_kwargs = {
    "thresholds": thresholds,
    "captured_metadata": bundle.vehicle,
    "replay_metadata": replay_metadata,
    "counterfactual": provenance_mode == "counterfactual",
    "acknowledge_instrumentation_only": provenance_mode == "instrumentation_only",
  }
  fidelity = (
    evaluate_harness_fidelity(replace(simulation, trace=list(evaluation_rows)), **fidelity_kwargs)
    if getattr(simulation, "planner_state_restoration_verified", False) is True else
    evaluate_fidelity(evaluation_rows, **fidelity_kwargs)
  )
  diagnostic_fidelity = evaluate_diagnostic_fidelity(evaluation_rows, thresholds=thresholds)
  if not param_manifest["complete"]:
    manifest_reason = (
      f"captured parameter manifest is missing {len(param_manifest['missingKeys'])} replay-relevant values"
    )
    fidelity["status"] = "not_evaluated"
    fidelity["planner"]["status"] = "not_evaluated"
    fidelity["planner"]["passed"] = False
    fidelity["planner"]["gate_reasons"].append(manifest_reason)
    fidelity["overall"]["status"] = "not_evaluated"
    fidelity["overall"]["passed"] = False
    fidelity["overall"]["gate_reasons"].append(f"planner: {manifest_reason}")
  if window_selection["status"] == "not_evaluated":
    fidelity["status"] = "not_evaluated"
    fidelity["overall"]["status"] = "not_evaluated"
    fidelity["overall"]["passed"] = False
    fidelity["overall"]["gate_reasons"].append(f"window selection: {window_selection['reason']}")
    diagnostic_fidelity["status"] = "not_evaluated"
    diagnostic_fidelity["diagnosticPassed"] = False
    diagnostic_fidelity["reasons"].append(f"window selection: {window_selection['reason']}")
  result = {
    "snapshotPath": str(Path(snapshot_path)),
    "scenarioName": bundle.name or Path(snapshot_path).name,
    "captureProvenance": _capture_provenance(bundle.vehicle),
    "capturedParamManifest": param_manifest,
    "replayProvenance": dict(replay_metadata),
    "provenance": provenance.as_dict(),
    "traceRowCount": len(simulation.trace),
    "evaluationTraceRowCount": len(evaluation_rows),
    "windowSelection": window_selection,
    "summary": simulation.summary,
    "fidelity": fidelity,
    "diagnosticFidelity": diagnostic_fidelity,
  }
  if diagnostic_scheduler_variants:
    result["schedulerVariantDiagnostics"] = _run_scheduler_variant_diagnostics(
      bundle,
      current_simulation=simulation,
      current_evaluation_rows=evaluation_rows,
      thresholds=thresholds,
      seed=seed,
      window_selection=window_selection,
    )
  return result


def build_scheduler_variant_timeline(
  timeline: Iterable[StepInput],
  variant: str,
) -> list[StepInput]:
  """Select one explicit legacy scheduler candidate without using planner output as an input."""
  if variant not in ("earliest_candidate", "latest_candidate"):
    raise ValueError(f"unsupported diagnostic scheduler variant '{variant}'")
  rewritten: list[StepInput] = []
  available_radar_clocks: set[int] = set()
  for step in timeline:
    publish_clock = step.recorded_radar_state_log_mono_time_ns
    if publish_clock is not None and publish_clock > 0:
      available_radar_clocks.add(int(publish_clock))
    if step.planner_radar_resolution != "legacy_ambiguous":
      rewritten.append(step)
      continue
    candidates = [
      int(clock)
      for clock in step.planner_radar_state_candidates_ns
      if int(clock) > 0 and int(clock) in available_radar_clocks
    ]
    if not candidates:
      rewritten.append(step)
      continue
    target = min(candidates) if variant == "earliest_candidate" else max(candidates)
    rewritten.append(replace(
      step,
      planner_radar_state_log_mono_time_ns=target,
      planner_radar_resolution="legacy_timing_unique",
    ))
  return rewritten


def _run_scheduler_variant_diagnostics(
  bundle: SnapshotBundle,
  *,
  current_simulation: Any,
  current_evaluation_rows: list[Mapping[str, Any]],
  thresholds: Any,
  seed: int,
  window_selection: Mapping[str, Any],
) -> dict[str, Any]:
  variants: dict[str, Any] = {
    "synchronous_current": _scheduler_variant_result(
      current_simulation,
      current_evaluation_rows,
      thresholds=thresholds,
      window_selection=window_selection,
    ),
  }
  for variant_name in ("earliest_candidate", "latest_candidate"):
    timeline = build_scheduler_variant_timeline(bundle.timeline, variant_name)
    vehicle_config = resolve_fidelity_vehicle_config(bundle)
    simulation = run_harness(
      vehicle_config=vehicle_config,
      scenario_name=f"{bundle.name or bundle.path.name}:{variant_name}",
      steps=timeline,
      initial_speed_mps=bundle.initial_speed_mps,
      initial_accel_mps2=bundle.initial_accel_mps2,
      noise_profile="off",
      seed=seed,
      perception_filter="auto",
      ego_replay_mode="auto",
    )
    try:
      rows = select_evaluation_trace_rows(bundle, simulation.trace)
      variant_window = {"status": "selected", "reason": None}
    except ValueError as exc:
      rows = []
      variant_window = {"status": "not_evaluated", "reason": str(exc)}
    variants[variant_name] = _scheduler_variant_result(
      simulation,
      rows,
      thresholds=thresholds,
      window_selection=variant_window,
    )

  ranked = []
  for name, variant in variants.items():
    planner = variant["diagnosticFidelity"]["planner"]
    mae = planner["metrics"]["a_target_error_mps2"]["mae"]
    if mae is not None:
      ranked.append((float(mae), name))
  ranked.sort()
  return {
    "mode": "legacy_scheduler_counterfactual_non_gating",
    "gateEligible": False,
    "oracleBestVariantByPlannerATargetMae": None if not ranked else ranked[0][1],
    "selectionWarning": (
      "oracleBestVariant is chosen using recorded planner output and is diagnostic only; " +
      "it must never seed an exact replay"
    ),
    "variants": variants,
  }


def _scheduler_variant_result(
  simulation: Any,
  evaluation_rows: list[Mapping[str, Any]],
  *,
  thresholds: Any,
  window_selection: Mapping[str, Any],
) -> dict[str, Any]:
  diagnostic = evaluate_diagnostic_fidelity(evaluation_rows, thresholds=thresholds)
  if window_selection.get("status") != "selected":
    diagnostic["status"] = "not_evaluated"
    diagnostic["diagnosticPassed"] = False
    diagnostic["reasons"].append(f"window selection: {window_selection.get('reason')}")
  return {
    "traceRowCount": len(simulation.trace),
    "evaluationTraceRowCount": len(evaluation_rows),
    "windowSelection": dict(window_selection),
    "summary": simulation.summary,
    "diagnosticFidelity": diagnostic,
  }


def summarize_corpus(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
  result_list = [dict(result) for result in results]
  status_counts: dict[str, int] = {}
  radar_status_counts: dict[str, int] = {}
  planner_status_counts: dict[str, int] = {}
  provenance_counts: dict[str, int] = {}
  diagnostic_status_counts: dict[str, int] = {}
  for result in result_list:
    fidelity = result.get("fidelity") if isinstance(result.get("fidelity"), Mapping) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
    diagnostic = result.get("diagnosticFidelity") if isinstance(result.get("diagnosticFidelity"), Mapping) else {}
    _increment(status_counts, str(fidelity.get("status", "unknown")))
    radar = fidelity.get("radar") if isinstance(fidelity.get("radar"), Mapping) else {}
    planner = fidelity.get("planner") if isinstance(fidelity.get("planner"), Mapping) else {}
    _increment(radar_status_counts, str(radar.get("status", "unknown")))
    _increment(planner_status_counts, str(planner.get("status", "unknown")))
    _increment(provenance_counts, str(provenance.get("status", "unknown")))
    _increment(diagnostic_status_counts, str(diagnostic.get("status", "unknown")))
  return {
    "schemaVersion": 1,
    "caseCount": len(result_list),
    "statusCounts": dict(sorted(status_counts.items())),
    "radarStatusCounts": dict(sorted(radar_status_counts.items())),
    "plannerStatusCounts": dict(sorted(planner_status_counts.items())),
    "provenanceStatusCounts": dict(sorted(provenance_counts.items())),
    "diagnosticStatusCounts": dict(sorted(diagnostic_status_counts.items())),
    "allPassed": bool(result_list) and all(
      isinstance(result.get("fidelity"), Mapping) and result["fidelity"].get("status") == "pass"
      for result in result_list
    ),
    "results": result_list,
  }


def _capture_provenance(vehicle: Mapping[str, Any]) -> dict[str, Any]:
  return {
    key: vehicle.get(key)
    for key in (
      "gitCommit", "gitBranch", "gitRemote", "gitDirty", "gitDiffEmpty", "gitDiffSha256",
      "runtimeDeviceType", "runtimePlatform", "runtimeMachine", "runtimeKernelVersion", "runtimeOsVersion",
    )
  }


def _git_output(repo_root: Path, *args: str) -> bytes:
  return subprocess.run(
    ("git", "-C", str(repo_root), *args),
    check=True,
    capture_output=True,
  ).stdout


def _read_runtime_file(path: Path, *, fallback: str | None) -> str:
  try:
    value = path.read_text()
  except OSError:
    value = fallback or ""
  return str(value).strip()


def _finite_float(value: Any) -> float | None:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if number == number and abs(number) != float("inf") else None


def _increment(counts: dict[str, int], key: str) -> None:
  counts[key] = counts.get(key, 0) + 1


__all__ = [
  "collect_replay_git_metadata",
  "build_scheduler_variant_timeline",
  "resolve_fidelity_vehicle_config",
  "run_snapshot_fidelity",
  "select_evaluation_trace_rows",
  "summarize_corpus",
]
