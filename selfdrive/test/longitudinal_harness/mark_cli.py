"""CLI for driver marks: find the presses, then read one incident frame by frame.

  python -m openpilot.selfdrive.test.longitudinal_harness.mark_cli list
  python -m openpilot.selfdrive.test.longitudinal_harness.mark_cli show <markId>

``list`` sweeps route roots (qlog first, rlog only where a mark actually is) and
prints one row per press. ``show`` reads the segment rlog -- and its neighbouring
segments when the window crosses a boundary -- and prints the -20 s / +6 s window
in the column vocabulary used by docs/chauffeur/longitudinal/, plus the
planner-internal sidecar when one was pulled off the device.

Two things ``show`` will always tell you rather than hide:

* whether the window it printed is the full width it advertises. A press early
  in a segment whose predecessor was not pulled gets a short pre-roll, and every
  metric on the line above the trace is then computed over that partial window.
* how to get the planner-internal sidecar if it is missing (the literal ``adb``
  command, see ``marks.MARK_SIDECAR_PULL_COMMAND``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpilot.selfdrive.test.longitudinal_harness.marks import (
  DEFAULT_MARK_SIDECAR_ROOTS,
  DriverMark,
  MARK_SIDECAR_PULL_COMMAND,
  POST_WINDOW_S,
  PRE_WINDOW_S,
  TraceWindow,
  compute_mark_metrics,
  find_mark_sidecar,
  format_ascii_trace,
  join_sidecar_rows,
  load_mark_sidecar,
  load_trace_window,
  scan_root_marks,
)


# tSegRelS is segment-relative on purpose; see marks.DriverMark.
LIST_HEADER = ("routeKey", "segIdx", "pressMonoTimeNs", "tSegRelS", "services", "payload", "status", "markId")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="EV6 longitudinal harness driver marks")
  subparsers = parser.add_subparsers(dest="command", required=True)

  list_parser = subparsers.add_parser("list", help="List driver marks found under the route roots")
  list_parser.add_argument("--root", action="append", default=[], help="Route root to scan (repeatable)")
  list_parser.add_argument("--route-key", default=None, help="Only scan segments of this route")
  list_parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

  show_parser = subparsers.add_parser("show", help="Print the per-frame trace around one driver mark")
  show_parser.add_argument("mark_id", help="markId from `list`, or a bare pressMonoTimeNs")
  show_parser.add_argument("--root", action="append", default=[], help="Route root to scan (repeatable)")
  show_parser.add_argument("--route-key", default=None, help="Only scan segments of this route")
  show_parser.add_argument("--pre-s", type=float, default=abs(PRE_WINDOW_S), help="Seconds of trace before the press")
  show_parser.add_argument("--post-s", type=float, default=POST_WINDOW_S, help="Seconds of trace after the press")
  show_parser.add_argument("--sidecar-root", action="append", default=[], help="Planner-debug sidecar root (repeatable)")
  show_parser.add_argument("--no-sidecar", action="store_true", help="Skip the Phase 2 planner-internal sidecar join")
  show_parser.add_argument("--json", action="store_true", help="Emit JSON instead of an ASCII trace")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  roots = [Path(root) for root in args.root] or None
  marks = scan_root_marks(roots, route_key=args.route_key)

  if args.command == "list":
    return _run_list(marks, as_json=args.json)
  if args.command == "show":
    return _run_show(marks, args)
  raise SystemExit(f"unsupported command '{args.command}'")


def _run_list(marks: list[DriverMark], *, as_json: bool) -> int:
  if as_json:
    print(json.dumps([mark.to_dict() for mark in marks], indent=2, sort_keys=True))
    return 0
  rows: list[tuple[str, ...]] = [LIST_HEADER]
  for mark in marks:
    rows.append((
      mark.route_key,
      str(mark.seg_idx),
      str(mark.press_log_mono_time_ns),
      "--" if mark.t_seg_rel_s is None else f"{mark.t_seg_rel_s:.2f}",
      "+".join(mark.services),
      "-" if mark.payload is None else json.dumps(mark.payload, separators=(",", ":"), sort_keys=True),
      mark.status,
      mark.mark_id,
    ))
  print(_format_table(rows))
  if not marks:
    print("(no driver marks found; check --root)")
  return 0


def _run_show(marks: list[DriverMark], args: argparse.Namespace) -> int:
  mark = _resolve_mark(marks, args.mark_id)
  if mark is None:
    raise SystemExit(f"no driver mark matches '{args.mark_id}'; run `mark_cli list` to see what was found")
  if mark.rlog_path is None:
    raise SystemExit(f"driver mark {mark.mark_id} has no rlog on disk; only the qlog press was found")

  window = load_trace_window(mark.rlog_path, mark.press_log_mono_time_ns, pre_s=args.pre_s, post_s=args.post_s)
  trace = window.rows
  metrics = compute_mark_metrics(trace, window=window)

  sidecar_path: Path | None = None
  sidecar_header: dict[str, Any] | None = None
  matched_rows = 0
  if not args.no_sidecar:
    sidecar_roots = [Path(root) for root in args.sidecar_root] or list(DEFAULT_MARK_SIDECAR_ROOTS)
    sidecar_path = find_mark_sidecar(mark.route_key, mark.press_log_mono_time_ns, sidecar_roots)
    if sidecar_path is not None:
      loaded = load_mark_sidecar(sidecar_path)
      if loaded is not None:
        sidecar_header, sidecar_rows = loaded
        matched_rows = join_sidecar_rows(trace, sidecar_rows)

  if args.json:
    print(json.dumps({
      "mark": mark.to_dict(),
      "metrics": metrics,
      "window": window.summary(),
      "sidecar": _sidecar_summary(sidecar_path, sidecar_header, matched_rows),
      "trace": trace,
    }, indent=2, sort_keys=True, default=str))
    return 0

  print(f"mark {mark.mark_id}  status={mark.status}  services={'+'.join(mark.services)}")
  print(f"  rlog {mark.rlog_path}")
  if mark.payload is not None:
    print(f"  payload {json.dumps(mark.payload, sort_keys=True)}")
  print("  metrics " + json.dumps(metrics, sort_keys=True))
  print("  " + _window_line(window))
  print("  " + _sidecar_line(sidecar_path, sidecar_header, matched_rows, no_sidecar=args.no_sidecar))
  print("")
  print(format_ascii_trace(trace, title=_trace_title(window, pre_s=args.pre_s, post_s=args.post_s)))
  return 0


def _trace_title(window: TraceWindow, *, pre_s: float, post_s: float) -> str:
  title = f"window {-abs(pre_s):+.1f} s .. {abs(post_s):+.1f} s around the press"
  if not window.truncated:
    return title
  covered_pre = "?" if window.covered_pre_s is None else f"{-window.covered_pre_s:+.1f}"
  covered_post = "?" if window.covered_post_s is None else f"{window.covered_post_s:+.1f}"
  # Never print a full-width header over a partial window: every metric above
  # was computed on exactly these rows.
  return f"{title} -- WINDOW TRUNCATED, actually covers {covered_pre} s .. {covered_post} s"


def _window_line(window: TraceWindow) -> str:
  segments = "+".join(str(seg_idx) for seg_idx in window.segments_read) or "none"
  if not window.truncated:
    return f"window full ({len(window.rows)} rows, segments {segments})"
  missing = ", ".join(str(seg_idx) for seg_idx in window.missing_segments) or "none (the pulled segments simply stop here)"
  return (
    f"window TRUNCATED at {'+'.join(window.truncated_edges)} ({len(window.rows)} rows, segments {segments}; " +
    f"missing neighbour segments: {missing}) -- every metric above is computed over this partial window"
  )


def _resolve_mark(marks: list[DriverMark], wanted: str) -> DriverMark | None:
  for mark in marks:
    if mark.mark_id == wanted:
      return mark
  if wanted.isdigit():
    press_ns = int(wanted)
    for mark in marks:
      if mark.press_log_mono_time_ns == press_ns:
        return mark
  return None


def _sidecar_summary(path: Path | None, header: dict[str, Any] | None, matched_rows: int) -> dict[str, Any]:
  if path is None:
    return {"present": False}
  return {
    "present": True,
    "path": str(path),
    "matchedTraceRows": matched_rows,
    "rowCount": None if header is None else header.get("rowCount"),
    "route": None if header is None else header.get("route"),
    "segment": None if header is None else header.get("segment"),
    "schemaVersion": None if header is None else header.get("schemaVersion"),
    "truncated": None if header is None else header.get("truncated"),
    "declaredTruncated": None if header is None else header.get("declaredTruncated"),
    "integrityStatus": None if header is None else header.get("integrityStatus"),
    "integrityErrors": None if header is None else header.get("integrityErrors"),
    "droppedMarks": None if header is None else header.get("droppedMarks"),
    "liveTunePresent": bool(header is not None and header.get("liveTune")),
  }


def _sidecar_line(path: Path | None, header: dict[str, Any] | None, matched_rows: int, *, no_sidecar: bool) -> str:
  if no_sidecar:
    return "sidecar skipped (--no-sidecar)"
  if path is None:
    return (
      "sidecar none (planner-internal detail unavailable; the mark itself is unaffected). " +
      f"Pull them with: {MARK_SIDECAR_PULL_COMMAND}"
    )
  if header is None:
    return f"sidecar {path} (header unreadable)"
  return (
    f"sidecar {path} rows={header.get('rowCount')} matched={matched_rows} " +
    f"route={header.get('route')} seg={header.get('segment')} truncated={header.get('truncated')} " +
    f"integrity={header.get('integrityStatus', 'unknown')}"
  )


def _format_table(rows: list[tuple[str, ...]]) -> str:
  widths = [max(len(str(row[col])) for row in rows) for col in range(len(rows[0]))]
  return "\n".join("  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)).rstrip() for row in rows)


if __name__ == "__main__":
  raise SystemExit(main())
