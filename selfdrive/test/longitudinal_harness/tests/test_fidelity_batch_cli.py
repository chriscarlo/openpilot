from __future__ import annotations

import json
from pathlib import Path

import pytest

from openpilot.selfdrive.test.longitudinal_harness import fidelity_batch_cli


def _snapshot(path: Path) -> Path:
  path.mkdir(parents=True)
  for filename in fidelity_batch_cli.SNAPSHOT_FILENAMES:
    (path / filename).write_text("\n")
  return path.resolve()


def test_discovery_is_sorted_deduplicated_and_requires_complete_bundles(tmp_path: Path) -> None:
  root = tmp_path / "root"
  second = _snapshot(root / "z" / "second")
  first = _snapshot(root / "a" / "first")
  incomplete = root / "incomplete"
  incomplete.mkdir()
  (incomplete / "vehicle.json").write_text("{}\n")

  discovered = fidelity_batch_cli.discover_snapshot_paths([second], [root, root / "a"])

  assert discovered == sorted((first, second), key=str)
  with pytest.raises(ValueError, match="missing required files"):
    fidelity_batch_cli.discover_snapshot_paths([incomplete], [])


def test_run_batch_collects_replay_metadata_once_and_runs_sorted(monkeypatch, tmp_path: Path) -> None:
  second = _snapshot(tmp_path / "z")
  first = _snapshot(tmp_path / "a")
  calls: list[tuple[str, object]] = []

  def collect(repo_root):
    calls.append(("collect", Path(repo_root)))
    return {"gitCommit": "a" * 40}

  def run(path, *, replay_metadata, provenance_mode, diagnostic_scheduler_variants):
    calls.append(("run", Path(path)))
    assert replay_metadata == {"gitCommit": "a" * 40}
    assert provenance_mode == "instrumentation_only"
    assert diagnostic_scheduler_variants is False
    return {"snapshotPath": str(path), "fidelity": {"status": "pass"}}

  def summarize(results):
    result_list = list(results)
    calls.append(("summarize", len(result_list)))
    return {"allPassed": True, "results": result_list}

  monkeypatch.setattr(fidelity_batch_cli, "collect_replay_git_metadata", collect)
  monkeypatch.setattr(fidelity_batch_cli, "run_snapshot_fidelity", run)
  monkeypatch.setattr(fidelity_batch_cli, "summarize_corpus", summarize)

  summary = fidelity_batch_cli.run_batch(
    [second, first, second],
    repo_root=tmp_path,
    provenance_mode="instrumentation_only",
  )

  assert calls == [
    ("collect", tmp_path),
    ("run", first),
    ("run", second),
    ("summarize", 2),
  ]
  assert summary["allPassed"] is True


@pytest.mark.parametrize(
  ("audit_only", "all_passed", "expected_exit"),
  ((False, False, 1), (True, False, 0), (False, True, 0)),
)
def test_main_exit_policy_and_json_output(
  monkeypatch,
  tmp_path: Path,
  capsys,
  audit_only: bool,
  all_passed: bool,
  expected_exit: int,
) -> None:
  snapshot = _snapshot(tmp_path / "snapshot")
  output = tmp_path / "result.json"
  expected = {"schemaVersion": 1, "caseCount": 1, "allPassed": all_passed, "results": []}
  calls = []

  def run(paths, *, repo_root, provenance_mode, diagnostic_scheduler_variants):
    calls.append((list(paths), Path(repo_root), provenance_mode, diagnostic_scheduler_variants))
    return expected

  monkeypatch.setattr(fidelity_batch_cli, "run_batch", run)
  argv = [
    "--snapshot", str(snapshot),
    "--repo-root", str(tmp_path),
    "--provenance-mode", "counterfactual",
    "--output-json", str(output),
  ]
  if audit_only:
    argv.append("--audit-only")

  assert fidelity_batch_cli.main(argv) == expected_exit
  assert calls == [([snapshot], tmp_path, "counterfactual", False)]
  assert json.loads(capsys.readouterr().out) == expected
  assert json.loads(output.read_text()) == expected


def test_require_pass_compatibility_flag_cannot_weaken_default_gate(monkeypatch, tmp_path: Path) -> None:
  snapshot = _snapshot(tmp_path / "snapshot")
  monkeypatch.setattr(
    fidelity_batch_cli,
    "run_batch",
    lambda *args, **kwargs: {"allPassed": False, "results": []},
  )
  assert fidelity_batch_cli.main(["--snapshot", str(snapshot), "--require-pass"]) == 1


def test_main_requires_a_snapshot_source() -> None:
  with pytest.raises(SystemExit) as exc_info:
    fidelity_batch_cli.main([])
  assert exc_info.value.code == 2
