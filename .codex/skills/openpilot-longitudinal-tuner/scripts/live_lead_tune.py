#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_on_path() -> Path | None:
  candidates: list[Path] = []
  try:
    candidates.append(Path(__file__).resolve().parents[4])
  except Exception:
    pass
  candidates.append(Path("/data/openpilot"))

  for root in candidates:
    if root.exists() and (root / "cereal").exists():
      root_str = str(root)
      if root_str not in sys.path:
        sys.path.insert(0, root_str)
      return root
  return None


REPO_ROOT = _ensure_repo_on_path()

from openpilot.common.params import Params  # noqa: E402
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import (  # noqa: E402
  LEAD_RESPONSE_TUNE_SPECS,
  LEAD_RESPONSE_TUNE_SPECS_BY_ATTR,
  format_lead_response_tune_summary,
  get_lead_response_tune_rows,
  read_lead_response_tuning_config,
  reset_lead_response_tuning_params,
  set_lead_response_tuning_params,
)


def _format_value(value: Any) -> str:
  if value is None:
    return "-"
  return f"{float(value):.3f}"


def _rows_by_attr(params: Params) -> dict[str, dict[str, Any]]:
  return {
    row["attr"]: row
    for row in get_lead_response_tune_rows(params)
  }


def _print_table(rows: list[dict[str, Any]]) -> None:
  print("label                            stored   default effective bounds")
  for row in rows:
    bounds = f"[{row['minimum']:.3f},{row['maximum']:.3f}]"
    print(
      f"{row['label']:<32} "
      f"{_format_value(row['stored']):>7} "
      f"{row['default']:>8.3f} "
      f"{row['effective']:>9.3f} "
      f"{bounds}"
    )


def _add_tune_flags(parser: argparse.ArgumentParser) -> None:
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    parser.add_argument(
      f"--{spec.cli_name}",
      dest=spec.attr,
      type=float,
      default=None,
      help=f"{spec.description} Range: {spec.minimum:.3f}..{spec.maximum:.3f}.",
    )


def _show_command(params: Params, *, json_output: bool, shell_summary: bool) -> int:
  if shell_summary:
    print(format_lead_response_tune_summary(read_lead_response_tuning_config(params)))
    return 0

  rows = get_lead_response_tune_rows(params)
  if json_output:
    print(json.dumps(_rows_by_attr(params), indent=2, sort_keys=True))
    return 0

  _print_table(rows)
  print(format_lead_response_tune_summary(read_lead_response_tuning_config(params)))
  return 0


def _set_command(args: argparse.Namespace, params: Params) -> int:
  requested = {
    spec.attr: getattr(args, spec.attr)
    for spec in LEAD_RESPONSE_TUNE_SPECS
    if getattr(args, spec.attr) is not None
  }
  if not requested:
    raise SystemExit("set requires at least one knob override")

  applied = set_lead_response_tuning_params(params, requested)
  for attr, requested_value in requested.items():
    spec = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR[attr]
    applied_value = applied[attr]
    if abs(float(requested_value) - applied_value) > 1e-6:
      print(
        f"clamped {spec.label}: requested={float(requested_value):.3f} "
        f"effective={applied_value:.3f} bounds=[{spec.minimum:.3f},{spec.maximum:.3f}]"
      )
  print(format_lead_response_tune_summary(read_lead_response_tuning_config(params)))
  return 0


def _reset_command(params: Params) -> int:
  reset_lead_response_tuning_params(params)
  print(format_lead_response_tune_summary(read_lead_response_tuning_config(params)))
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description="Show or edit live lead-response tuning params.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  show_parser = subparsers.add_parser("show", help="print current live tune values")
  show_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
  show_parser.add_argument("--shell-summary", action="store_true", help="emit a single-line summary")

  set_parser = subparsers.add_parser("set", help="set one or more live tune values")
  _add_tune_flags(set_parser)

  subparsers.add_parser("reset", help="remove this feature's overrides and fall back to defaults")

  args = parser.parse_args()
  params = Params()

  if args.command == "show":
    return _show_command(params, json_output=args.json, shell_summary=args.shell_summary)
  if args.command == "set":
    return _set_command(args, params)
  if args.command == "reset":
    return _reset_command(params)
  raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
  raise SystemExit(main())
