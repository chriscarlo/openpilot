from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.test.longitudinal_harness import fidelity_runner
from openpilot.selfdrive.test.longitudinal_harness.config import REPLAY_PARAM_DEFAULT_VALUES
from openpilot.selfdrive.test.longitudinal_harness.fidelity_runner import (
  build_scheduler_variant_timeline,
  collect_replay_git_metadata,
  select_evaluation_trace_rows,
  summarize_corpus,
)
from openpilot.selfdrive.test.longitudinal_harness.inputs import SnapshotBundle, StepInput


def test_select_evaluation_trace_rows_excludes_dependency_warmup(tmp_path: Path) -> None:
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={"tStartS": 100.0, "evaluationTStartS": 115.0, "tEndS": 120.0},
    params={},
    timeline=[],
    initial_speed_mps=0.0,
  )
  rows = [{"time_s": value} for value in (0.0, 14.99, 15.0, 17.0, 20.0, 20.01)]
  assert select_evaluation_trace_rows(bundle, rows) == rows[2:5]


def test_select_evaluation_trace_rows_uses_actual_harness_t_s_key(tmp_path: Path) -> None:
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={"tStartS": 100.0, "evaluationTStartS": 115.0, "tEndS": 120.0},
    params={},
    timeline=[],
    initial_speed_mps=0.0,
  )
  rows = [{"t_s": value} for value in (0.0, 14.99, 15.0, 17.0, 20.0, 20.01)]
  assert select_evaluation_trace_rows(bundle, rows) == rows[2:5]


def test_select_evaluation_trace_rows_rejects_missing_window_boundaries(tmp_path: Path) -> None:
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={},
    params={},
    timeline=[],
    initial_speed_mps=0.0,
  )
  with pytest.raises(ValueError, match="window boundaries"):
    select_evaluation_trace_rows(bundle, [{"t_s": 0.0}])


def test_select_evaluation_trace_rows_rejects_untimed_trace_rows(tmp_path: Path) -> None:
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={"tStartS": 100.0, "evaluationTStartS": 100.0, "tEndS": 101.0},
    params={},
    timeline=[],
    initial_speed_mps=0.0,
  )
  with pytest.raises(ValueError, match="missing a finite"):
    select_evaluation_trace_rows(bundle, [{"t_s": 0.0}, {}])


def test_scheduler_variants_select_only_candidates_already_in_the_radar_cache() -> None:
  timeline = [
    StepInput(
      t_s=0.0,
      cruise_speed_mps=20.0,
      recorded_radar_state_log_mono_time_ns=100,
      planner_radar_state_candidates_ns=[50, 100],
      planner_radar_resolution="legacy_ambiguous",
    ),
    StepInput(
      t_s=0.05,
      cruise_speed_mps=20.0,
      recorded_radar_state_log_mono_time_ns=200,
      planner_radar_state_candidates_ns=[100, 200],
      planner_radar_resolution="legacy_ambiguous",
    ),
  ]

  earliest = build_scheduler_variant_timeline(timeline, "earliest_candidate")
  latest = build_scheduler_variant_timeline(timeline, "latest_candidate")

  # Clock 50 is outside the loaded dependency history, so the first row can
  # only use its current publication. The second row has both candidates.
  assert [step.planner_radar_state_log_mono_time_ns for step in earliest] == [100, 100]
  assert [step.planner_radar_state_log_mono_time_ns for step in latest] == [100, 200]
  assert all(step.planner_radar_resolution == "legacy_timing_unique" for step in earliest + latest)


def test_scheduler_variant_builder_rejects_unknown_mode() -> None:
  with pytest.raises(ValueError, match="unsupported diagnostic scheduler variant"):
    build_scheduler_variant_timeline([], "oracle")


def test_collect_replay_git_metadata_scopes_dirty_diff(tmp_path: Path) -> None:
  _git(tmp_path, "init")
  _git(tmp_path, "config", "user.email", "test@example.com")
  _git(tmp_path, "config", "user.name", "Test")
  production = tmp_path / "selfdrive" / "controls"
  production.mkdir(parents=True)
  source = production / "planner.py"
  source.write_text("value = 1\n")
  _git(tmp_path, "add", str(source.relative_to(tmp_path)))
  _git(tmp_path, "commit", "-m", "initial")

  # An unrelated untracked artifact must not taint production provenance.
  (tmp_path / "notes.txt").write_text("scratch\n")
  clean = collect_replay_git_metadata(tmp_path)
  assert clean["gitDirty"] is False
  assert clean["gitDiffEmpty"] is True
  assert clean["runtimeDeviceType"]
  assert clean["runtimePlatform"]
  assert clean["runtimeMachine"]
  assert clean["runtimeOsVersion"]
  assert clean["runtimeKernelVersion"]

  source.write_text("value = 2\n")
  dirty = collect_replay_git_metadata(tmp_path)
  assert dirty["gitCommit"] == clean["gitCommit"]
  assert dirty["gitDirty"] is True
  assert dirty["gitDiffEmpty"] is False
  assert dirty["gitDiffSha256"] != clean["gitDiffSha256"]


def test_summarize_corpus_keeps_not_evaluated_distinct_from_failure() -> None:
  summary = summarize_corpus([
    {
      "fidelity": {
        "status": "not_evaluated",
        "radar": {"status": "not_evaluated"},
        "planner": {"status": "not_evaluated"},
      },
      "provenance": {"status": "instrumentation_only"},
    },
    {
      "fidelity": {
        "status": "fail",
        "radar": {"status": "pass"},
        "planner": {"status": "fail"},
      },
      "provenance": {"status": "exact"},
    },
  ])
  assert summary["statusCounts"] == {"fail": 1, "not_evaluated": 1}
  assert summary["radarStatusCounts"] == {"not_evaluated": 1, "pass": 1}
  assert summary["plannerStatusCounts"] == {"fail": 1, "not_evaluated": 1}
  assert summary["diagnosticStatusCounts"] == {"unknown": 2}
  assert summary["allPassed"] is False


def test_run_snapshot_fidelity_recomputes_manifest_and_rejects_old_exact_claim(monkeypatch, tmp_path: Path) -> None:
  params = dict(REPLAY_PARAM_DEFAULT_VALUES)
  missing_key = "Longitudinal.LiveTune.ModelLeadFilterVRelTauS"
  params.pop(missing_key)
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={"topology": "lka", "tStartS": 0.0, "evaluationTStartS": 0.0, "tEndS": 0.0},
    params=params,
    timeline=[],
    initial_speed_mps=0.0,
  )
  passing_fidelity = {
    "status": "pass",
    "planner": {"status": "pass", "passed": True, "gate_reasons": []},
    "overall": {"status": "pass", "passed": True, "gate_reasons": []},
  }
  monkeypatch.setattr(fidelity_runner, "load_snapshot_bundle", lambda _path: bundle)
  monkeypatch.setattr(
    fidelity_runner,
    "classify_replay_provenance",
    lambda *args, **kwargs: SimpleNamespace(as_dict=lambda: {"status": "exact"}),
  )
  monkeypatch.setattr(fidelity_runner, "resolve_fidelity_vehicle_config", lambda _bundle: object())
  monkeypatch.setattr(
    fidelity_runner,
    "run_harness",
    lambda **kwargs: SimpleNamespace(trace=[{"t_s": 0.0}], summary={}),
  )
  monkeypatch.setattr(fidelity_runner, "evaluate_fidelity", lambda *args, **kwargs: passing_fidelity)
  monkeypatch.setattr(
    fidelity_runner,
    "evaluate_diagnostic_fidelity",
    lambda *args, **kwargs: {"status": "pass", "diagnosticPassed": True, "reasons": []},
  )

  result = fidelity_runner.run_snapshot_fidelity(tmp_path, replay_metadata={})

  assert result["capturedParamManifest"]["complete"] is False
  assert result["capturedParamManifest"]["missingKeys"] == [missing_key]
  assert result["fidelity"]["status"] == "not_evaluated"
  assert result["fidelity"]["planner"]["status"] == "not_evaluated"
  assert "captured parameter manifest is missing 1" in result["fidelity"]["planner"]["gate_reasons"][-1]


def _git(root: Path, *args: str) -> None:
  subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
