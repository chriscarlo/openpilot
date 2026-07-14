from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from openpilot.selfdrive.test.longitudinal_harness.fidelity_runner import (
  collect_replay_git_metadata,
  run_snapshot_fidelity,
  summarize_corpus,
)


SNAPSHOT_FILENAMES = ("vehicle.json", "params.json", "timeline.jsonl")
PROVENANCE_MODES = ("exact", "instrumentation_only", "counterfactual")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Batch EV6 longitudinal snapshot fidelity audit")
  parser.add_argument(
    "--snapshot",
    action="append",
    type=Path,
    default=[],
    help="Snapshot bundle directory; repeat for multiple bundles",
  )
  parser.add_argument(
    "--snapshot-root",
    action="append",
    type=Path,
    default=[],
    help="Recursively discover snapshot bundle directories; repeat for multiple roots",
  )
  parser.add_argument("--repo-root", type=Path, default=Path.cwd())
  parser.add_argument("--provenance-mode", choices=PROVENANCE_MODES, default="exact")
  parser.add_argument("--output-json", type=Path, default=None)
  parser.add_argument(
    "--diagnostic-scheduler-variants",
    action="store_true",
    help="Also replay legacy ambiguous planner rows with earliest/latest RadarD candidates (non-gating)",
  )
  parser.add_argument(
    "--require-pass",
    action="store_true",
    help="Deprecated compatibility flag; passing is already required by default",
  )
  parser.add_argument(
    "--audit-only",
    action="store_true",
    help="Report FAIL/NOT_EVALUATED cases but exit zero; never use this for an automated gate",
  )
  return parser


def discover_snapshot_paths(
  snapshots: Iterable[str | Path],
  snapshot_roots: Iterable[str | Path],
) -> list[Path]:
  """Return resolved snapshot directories in stable order without duplicates."""
  discovered: dict[str, Path] = {}
  for snapshot in snapshots:
    candidate = Path(snapshot).expanduser().resolve()
    _validate_snapshot_directory(candidate)
    discovered[str(candidate)] = candidate

  for root_value in snapshot_roots:
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
      raise ValueError(f"snapshot root is not a directory: {root}")
    candidates = [root] if _is_snapshot_directory(root) else []
    candidates.extend(path.parent for path in root.rglob("vehicle.json"))
    for candidate in candidates:
      resolved = candidate.resolve()
      if _is_snapshot_directory(resolved):
        discovered[str(resolved)] = resolved

  return [discovered[key] for key in sorted(discovered)]


def run_batch(
  snapshot_paths: Iterable[str | Path],
  *,
  repo_root: str | Path,
  provenance_mode: str,
  diagnostic_scheduler_variants: bool = False,
) -> dict[str, Any]:
  """Run a deterministic corpus using one scoped replay-provenance snapshot."""
  if provenance_mode not in PROVENANCE_MODES:
    raise ValueError(f"unsupported provenance mode '{provenance_mode}'")
  paths = sorted({Path(path).expanduser().resolve() for path in snapshot_paths}, key=str)
  replay_metadata = collect_replay_git_metadata(repo_root)
  results = [
    run_snapshot_fidelity(
      path,
      replay_metadata=replay_metadata,
      provenance_mode=provenance_mode,
      diagnostic_scheduler_variants=diagnostic_scheduler_variants,
    )
    for path in paths
  ]
  return summarize_corpus(results)


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if not args.snapshot and not args.snapshot_root:
    parser.error("at least one --snapshot or --snapshot-root is required")
  try:
    snapshot_paths = discover_snapshot_paths(args.snapshot, args.snapshot_root)
  except ValueError as exc:
    parser.error(str(exc))
  if not snapshot_paths:
    parser.error("no snapshot bundles found")

  summary = run_batch(
    snapshot_paths,
    repo_root=args.repo_root,
    provenance_mode=args.provenance_mode,
    diagnostic_scheduler_variants=args.diagnostic_scheduler_variants,
  )
  payload = json.dumps(summary, indent=2, sort_keys=True)
  print(payload)
  if args.output_json is not None:
    args.output_json.write_text(payload + "\n")
  gate_required = args.require_pass or not args.audit_only
  return 1 if gate_required and not summary["allPassed"] else 0


def _is_snapshot_directory(path: Path) -> bool:
  return path.is_dir() and all((path / filename).is_file() for filename in SNAPSHOT_FILENAMES)


def _validate_snapshot_directory(path: Path) -> None:
  if not _is_snapshot_directory(path):
    required = ", ".join(SNAPSHOT_FILENAMES)
    raise ValueError(f"snapshot directory is missing required files ({required}): {path}")


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = [
  "PROVENANCE_MODES",
  "SNAPSHOT_FILENAMES",
  "build_parser",
  "discover_snapshot_paths",
  "main",
  "run_batch",
]
