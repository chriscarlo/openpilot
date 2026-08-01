"""Driver-mark discovery, tracing and offline triage.

A driver mark is a single deliberate flag-button press taken while driving. The
press crosses the wire as ``bookmarkButton`` and is echoed by ``feedbackd`` as
``userBookmark``; both are logged at decimation 1, so a press is always present
in both rlog and qlog. Nothing about the press carries a category -- the driver
gets one tap and no UI -- so everything downstream of the press is derived here.

Two conventions matter for anyone extending this module:

* Logs are opened with ``LogReader`` pointed straight at a file. ``tools.lib.route.Route``
  is deliberately not used: it dereferences ``metadata['url']`` and reaches the
  comma API, and it raises on dongle-less directory names.
* ``compute_mark_metrics`` is a TRIAGE HINT and never an oracle. It exists so a
  human can sort thirty marks by "probably harsh braking" before opening any of
  them, not so a test can assert on ``derivedCategory``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import warnings

from openpilot.tools.lib.logreader import LogReader

from openpilot.selfdrive.test.longitudinal_harness.route_extract import (
  DEFAULT_ROUTE_ROOTS,
  DRIVER_MARK_FRAME_TOLERANCE_S,
  DRIVER_MARK_PAYLOAD_WINDOW_S,
  DRIVER_MARK_SERVICES,
  WINDOWS_BY_TYPE,
  _parse_longflag,
  collapse_driver_mark_echoes,
  is_driver_mark_press_group,
  parse_segment_identity,
)


PRE_WINDOW_S, POST_WINDOW_S = WINDOWS_BY_TYPE["driver_mark"]
# The device directory is /data/media/0/LongMarks -- the literal value of
# longitudinal_mark_recorder.MARKS_DIR_DEFAULT. Pull it with, verbatim:
#
#   mkdir -p .cache && adb pull /data/media/0/LongMarks .cache/
#
# which lands the sidecars in .cache/LongMarks. The case must match the device
# name exactly or a case-sensitive filesystem will never find the pulled
# directory and `mark_cli show` degrades to "sidecar none" with no explanation.
# The snake_case spelling is kept only so sidecars pulled before this was fixed
# are still found. Route roots are tolerated too, so a sidecar pulled alongside
# its rlogs is found.
MARK_SIDECAR_DEVICE_DIR = "/data/media/0/LongMarks"
MARK_SIDECAR_PULL_COMMAND = f"mkdir -p .cache && adb pull {MARK_SIDECAR_DEVICE_DIR} .cache/"
DEFAULT_MARK_SIDECAR_ROOTS: tuple[Path, ...] = (
  Path(".cache/LongMarks"),
  Path(".cache/long_marks"),
  *DEFAULT_ROUTE_ROOTS,
)
MARK_SIDECAR_GLOB = "longmark__*.jsonl"
# The press logMonoTime is globally unique within a boot, so an exact integer
# match is the contract. The tolerance exists only for the case where the
# Phase 1 press-time note has not landed and the nearest frame clock is used.
MARK_SIDECAR_JOIN_TOLERANCE_NS = 150_000_000
# vRel below this is indistinguishable from published-track noise, so TTC built
# on it is meaningless rather than merely large.
CLOSING_SPEED_FLOOR_MPS = 0.5
HARSH_BRAKE_ACCEL_MPS2 = -2.5
HARSH_BRAKE_JERK_MPS3 = -5.0
LATE_BRAKE_TTC_S = 2.0
LATE_BRAKE_GAP_M = 12.0
SLUGGISH_RESUME_LAG_S = 2.0
LEAD_OPENING_V_REL_MPS = 0.5


@dataclass(frozen=True)
class DriverMark:
  """One collapsed driver flag press located in a log on disk.

  ``t_seg_rel_s`` is SEGMENT-relative: seconds from the first ``radarState`` in
  the one file that was scanned. It is deliberately NOT called ``t_rel_s``,
  because ``route_extract.DriverMarkPress.t_s`` is ROUTE-relative (seconds from
  the first ``radarState`` of segment 0 of the whole route) and the two are only
  equal for a mark in segment 0. Nothing may cross-reference them by value; the
  cross-tool join key is ``press_log_mono_time_ns``, which is a single boot-wide
  clock and is exact.
  """
  route_key: str
  seg_idx: int
  press_log_mono_time_ns: int
  t_seg_rel_s: float | None
  services: tuple[str, ...]
  payload: dict[str, Any] | None
  status: str
  log_path: Path
  rlog_path: Path | None = None

  @property
  def mark_id(self) -> str:
    return f"{self.route_key}--{self.seg_idx}--{self.press_log_mono_time_ns}"

  def to_dict(self) -> dict[str, Any]:
    return {
      "markId": self.mark_id,
      "routeKey": self.route_key,
      "segIdx": self.seg_idx,
      "pressMonoTimeNs": self.press_log_mono_time_ns,
      "tSegRelS": self.t_seg_rel_s,
      "services": list(self.services),
      "payload": self.payload,
      "status": self.status,
      "logPath": str(self.log_path),
      "rlogPath": None if self.rlog_path is None else str(self.rlog_path),
    }


def scan_segment_marks(log_path: str | Path,
                       *,
                       route_key: str | None = None,
                       seg_idx: int | None = None,
                       rlog_path: str | Path | None = None) -> list[DriverMark]:
  """Find every driver mark in one segment log.

  Works on either a qlog or an rlog. ``bookmarkButton``/``userBookmark`` are
  decimation-1 so both files carry every press, but ``logMessage`` is rlog-only,
  so LONGFLAG payloads are only ever recovered from an rlog.

  ``tSegRelS`` is relative to the first ``radarState`` publication in the ONE
  scanned file, i.e. segment-relative. See ``DriverMark`` for why that is not
  interchangeable with ``route_extract.DriverMarkPress.t_s``.
  """
  path = Path(log_path)
  presses: list[dict[str, Any]] = []
  payloads: list[dict[str, Any]] = []
  radar_times: list[int] = []
  first_radar_ns: int | None = None
  for msg in LogReader(str(path)):
    which = msg.which()
    if which in DRIVER_MARK_SERVICES:
      presses.append({
        "segIdx": 0 if seg_idx is None else int(seg_idx),
        "logMonoTimeNs": int(msg.logMonoTime),
        "service": which,
      })
    elif which == "logMessage":
      payload = _parse_longflag(msg.logMessage)
      if payload is not None:
        payloads.append({"logMonoTimeNs": int(msg.logMonoTime), "payload": payload})
    elif which == "radarState":
      radar_ns = int(msg.logMonoTime)
      radar_times.append(radar_ns)
      if first_radar_ns is None:
        first_radar_ns = radar_ns

  if not presses:
    return []

  radar_times.sort()
  resolved_route_key = route_key if route_key is not None else path.parent.name
  resolved_rlog = None if rlog_path is None else Path(rlog_path)
  marks: list[DriverMark] = []
  for group in collapse_driver_mark_echoes(presses):
    # A lone userBookmark is one of feedbackd's LKAS/audio paths, not a flag
    # press. See route_extract.is_driver_mark_press_group.
    if not is_driver_mark_press_group(group):
      continue
    press_ns = int(group["logMonoTimeNs"])
    payload = _nearest_payload(payloads, press_ns)
    anchored = _has_frame_within(radar_times, press_ns, DRIVER_MARK_FRAME_TOLERANCE_S)
    status = "ok" if (anchored and payload is not None) else ("no_payload" if anchored else "orphan")
    marks.append(DriverMark(
      route_key=resolved_route_key,
      seg_idx=int(group["segIdx"]),
      press_log_mono_time_ns=press_ns,
      t_seg_rel_s=None if first_radar_ns is None else (press_ns - first_radar_ns) / 1e9,
      services=tuple(group["services"]),
      payload=payload,
      status=status,
      log_path=path,
      rlog_path=resolved_rlog if resolved_rlog is not None else path,
    ))
  return marks


def scan_root_marks(roots: Iterable[str | Path] | None = None,
                    *,
                    route_key: str | None = None,
                    resolve_payloads: bool = True) -> list[DriverMark]:
  """Scan route roots for driver marks, preferring qlogs for the initial pass.

  A qlog is roughly two orders of magnitude smaller than its rlog and still
  carries every press, so the sweep reads qlogs first and only falls back to the
  rlog for segments that actually contain a mark (or that have no qlog at all).
  """
  scan_roots = [Path(root) for root in roots] if roots is not None else list(DEFAULT_ROUTE_ROOTS)
  marks: list[DriverMark] = []
  for root in scan_roots:
    if not root.exists():
      continue
    for rlog_path in sorted(root.rglob("*.zst")):
      identity = parse_segment_identity(rlog_path, root)
      if identity is None:
        continue
      _, segment_route_key, seg_idx, qlog_path = identity
      if route_key is not None and segment_route_key != route_key:
        continue
      scan_path = qlog_path if (qlog_path is not None and qlog_path.exists()) else rlog_path
      try:
        segment_marks = scan_segment_marks(
          scan_path,
          route_key=segment_route_key,
          seg_idx=seg_idx,
          rlog_path=rlog_path,
        )
        if segment_marks and resolve_payloads and scan_path != rlog_path:
          # A mark is present, so the rlog re-read is now worth its cost: it is
          # the only place LONGFLAG payloads exist.
          rescanned = scan_segment_marks(
            rlog_path,
            route_key=segment_route_key,
            seg_idx=seg_idx,
            rlog_path=rlog_path,
          )
          if rescanned:
            segment_marks = rescanned
      # One unreadable segment must not abort a whole-root sweep.
      except Exception as exc:
        warnings.warn(f"driver-mark scan skipped {scan_path}: {exc!r}", RuntimeWarning, stacklevel=2)
        continue
      marks.extend(segment_marks)
  marks.sort(key=lambda mark: (mark.route_key, mark.seg_idx, mark.press_log_mono_time_ns))
  return marks


@dataclass(frozen=True)
class TraceWindow:
  """One press's trace plus an honest statement of what the window covers.

  ``truncated`` is the load-bearing field: a truncated window has fewer seconds
  of history than ``pre_s``/``post_s`` advertise, so every scalar
  ``compute_mark_metrics`` derives from it (minATarget, peakDecelJerk, minTTC,
  throttleResumeLag, derivedCategory) is computed over a partial incident and
  must not be read as if it were complete.
  """
  rows: list[dict[str, Any]]
  press_log_mono_time_ns: int
  window_start_ns: int
  window_end_ns: int
  covered_start_ns: int | None
  covered_end_ns: int | None
  segments_read: tuple[int, ...]
  missing_segments: tuple[int, ...]

  @property
  def truncated_start(self) -> bool:
    return self.covered_start_ns is None or self.covered_start_ns > self.window_start_ns

  @property
  def truncated_end(self) -> bool:
    return self.covered_end_ns is None or self.covered_end_ns < self.window_end_ns

  @property
  def truncated(self) -> bool:
    return self.truncated_start or self.truncated_end

  @property
  def truncated_edges(self) -> list[str]:
    edges = []
    if self.truncated_start:
      edges.append("start")
    if self.truncated_end:
      edges.append("end")
    return edges

  @property
  def covered_pre_s(self) -> float | None:
    if self.covered_start_ns is None:
      return None
    return max(0.0, (self.press_log_mono_time_ns - max(self.covered_start_ns, self.window_start_ns)) / 1e9)

  @property
  def covered_post_s(self) -> float | None:
    if self.covered_end_ns is None:
      return None
    return max(0.0, (min(self.covered_end_ns, self.window_end_ns) - self.press_log_mono_time_ns) / 1e9)

  def summary(self) -> dict[str, Any]:
    return {
      "truncated": self.truncated,
      "truncatedEdges": self.truncated_edges,
      "requestedPreS": (self.press_log_mono_time_ns - self.window_start_ns) / 1e9,
      "requestedPostS": (self.window_end_ns - self.press_log_mono_time_ns) / 1e9,
      "coveredPreS": self.covered_pre_s,
      "coveredPostS": self.covered_post_s,
      "segmentsRead": list(self.segments_read),
      "missingSegments": list(self.missing_segments),
    }


def resolve_segment_rlog_chain(rlog_path: str | Path) -> dict[int, Path]:
  """Map ``seg_idx -> rlog path`` for one segment and its immediate neighbours.

  Segments are ``SEGMENT_LENGTH`` = 60 s (``system/loggerd/loggerd.h``), and the
  driver-mark window is 26 s, so a press can only ever pull in seg-1 or seg+1 --
  never further. Neighbours are found with the same ``parse_segment_identity``
  the extractor uses, so every directory layout it accepts works here too.
  """
  path = Path(rlog_path)
  identity = parse_segment_identity(path, path.parent)
  if identity is None:
    return {}
  _, route_key, seg_idx, _ = identity
  chain: dict[int, Path] = {seg_idx: path}
  search_root = path.parent.parent
  if not search_root.exists():
    return chain
  wanted = {seg_idx - 1, seg_idx + 1}
  # Both layouts the extractor produces: siblings one directory down
  # (route/<seg>/rlog.zst, route--<seg>/rlog.zst) and siblings in the same
  # directory as the mark's own rlog (route/rlog_<seg>.zst).
  for candidate in sorted([*search_root.glob("*/*.zst"), *search_root.glob("*.zst")]):
    if not wanted - set(chain):
      break
    candidate_identity = parse_segment_identity(candidate, search_root)
    if candidate_identity is None:
      continue
    _, candidate_route_key, candidate_seg_idx, _ = candidate_identity
    if candidate_route_key != route_key or candidate_seg_idx not in wanted or candidate_seg_idx in chain:
      continue
    chain[candidate_seg_idx] = candidate
  return chain


def load_trace_window(rlog_path: str | Path,
                      press_log_mono_time_ns: int,
                      *,
                      pre_s: float = -PRE_WINDOW_S,
                      post_s: float = POST_WINDOW_S) -> TraceWindow:
  """Build a per-frame trace around one press, crossing segment boundaries.

  Rows are keyed on ``longitudinalPlan`` publications (20 Hz) and carry the most
  recent state of every other service. Read from an rlog: ``longitudinalPlan``
  is decimated 10x into qlog, which is far too sparse to see a brake build.

  The press's own segment usually does not contain the whole window: a press 4 s
  into a 60 s segment has only 4 s of pre-roll in that file, and the 20 s of
  brake build-up the window exists to capture is in the PREVIOUS segment. So the
  neighbouring segment rlogs are read when the window reaches into them. When a
  needed neighbour is genuinely absent (route start, partial pull, deleted
  segment) the returned ``TraceWindow`` says so instead of silently handing back
  a short window under a full-width header.

  Row keys are snake_case so the same rows can be fed to ``compute_mark_metrics``
  as a ``closed_loop.SimulationResult.trace``. Row ``t_rel_s`` is PRESS-relative
  (unrelated to ``DriverMark.t_seg_rel_s``).
  """
  path = Path(rlog_path)
  start_ns = press_log_mono_time_ns - int(abs(pre_s) * 1e9)
  end_ns = press_log_mono_time_ns + int(abs(post_s) * 1e9)

  chain = resolve_segment_rlog_chain(path)
  own_seg_idx = next((seg_idx for seg_idx, seg_path in chain.items() if seg_path == path), 0)

  read_order: list[tuple[int, Path]] = []
  missing: list[int] = []
  previous_path = chain.get(own_seg_idx - 1)
  # The peek costs one extra open of the mark's own rlog, so it is only paid when
  # there is actually a previous segment that could be read.
  if previous_path is not None and start_ns < _first_log_mono_time_ns(path):
    read_order.append((own_seg_idx - 1, previous_path))
  read_order.append((own_seg_idx, path))

  state = _TraceState()
  rows: list[dict[str, Any]] = []
  segments_read: list[int] = []
  covered_start_ns: int | None = None
  covered_end_ns: int | None = None
  for seg_idx, seg_path in read_order:
    first_ns, last_ns = _read_trace_segment(seg_path, state, rows, start_ns, end_ns, press_log_mono_time_ns)
    segments_read.append(seg_idx)
    covered_start_ns = _min_optional(covered_start_ns, first_ns)
    covered_end_ns = _max_optional(covered_end_ns, last_ns)

  # Coverage, not the peek, is what proves the backward neighbour was needed and
  # unavailable: a previous segment that exists but is itself short lands here too.
  if covered_start_ns is not None and covered_start_ns > start_ns and (own_seg_idx - 1) not in segments_read:
    missing.append(own_seg_idx - 1)

  # Only now is the own segment's end known, so the forward neighbour is decided
  # here rather than up front. State carries over naturally: the reader below
  # continues with the state left at the end of the own segment.
  if covered_end_ns is not None and covered_end_ns < end_ns:
    next_path = chain.get(own_seg_idx + 1)
    if next_path is not None:
      first_ns, last_ns = _read_trace_segment(next_path, state, rows, start_ns, end_ns, press_log_mono_time_ns)
      segments_read.append(own_seg_idx + 1)
      covered_start_ns = _min_optional(covered_start_ns, first_ns)
      covered_end_ns = _max_optional(covered_end_ns, last_ns)
    else:
      missing.append(own_seg_idx + 1)

  rows.sort(key=lambda row: int(row["log_mono_time_ns"]))
  return TraceWindow(
    rows=rows,
    press_log_mono_time_ns=press_log_mono_time_ns,
    window_start_ns=start_ns,
    window_end_ns=end_ns,
    covered_start_ns=covered_start_ns,
    covered_end_ns=covered_end_ns,
    segments_read=tuple(segments_read),
    missing_segments=tuple(sorted(missing)),
  )


def load_trace(rlog_path: str | Path,
               press_log_mono_time_ns: int,
               *,
               pre_s: float = -PRE_WINDOW_S,
               post_s: float = POST_WINDOW_S) -> list[dict[str, Any]]:
  """``load_trace_window(...).rows``. Prefer ``load_trace_window`` -- it also
  reports whether the window it returns is the full width it was asked for."""
  return load_trace_window(rlog_path, press_log_mono_time_ns, pre_s=pre_s, post_s=post_s).rows


def _min_optional(current: int | None, candidate: int | None) -> int | None:
  if candidate is None:
    return current
  return candidate if current is None else min(current, candidate)


def _max_optional(current: int | None, candidate: int | None) -> int | None:
  if candidate is None:
    return current
  return candidate if current is None else max(current, candidate)


class _TraceState:
  """Most recent message per service, carried across segment boundaries."""
  def __init__(self) -> None:
    self.car_state: Any = None
    self.car_control: Any = None
    self.car_output: Any = None
    self.controls_state: Any = None
    self.selfdrive_state: Any = None
    self.radar_state: Any = None


def _first_log_mono_time_ns(path: Path) -> int:
  """logMonoTime of the first message in a log, or 0 if it cannot be read."""
  try:
    for msg in LogReader(str(path)):
      return int(msg.logMonoTime)
  except Exception as exc:
    warnings.warn(f"driver-mark trace could not read {path}: {exc!r}", RuntimeWarning, stacklevel=2)
  return 0


def _read_trace_segment(path: Path,
                        state: _TraceState,
                        rows: list[dict[str, Any]],
                        start_ns: int,
                        end_ns: int,
                        press_ns: int) -> tuple[int | None, int | None]:
  """Append this segment's in-window rows. Returns its (first, last) logMonoTime.

  The bounds are taken over EVERY message, not only the rows kept, because they
  are what proves whether the requested window was actually covered.
  """
  first_ns: int | None = None
  last_ns: int | None = None
  try:
    for msg in LogReader(str(path)):
      msg_ns = int(msg.logMonoTime)
      if first_ns is None:
        first_ns = msg_ns
      if last_ns is None or msg_ns > last_ns:
        last_ns = msg_ns
      which = msg.which()
      if which == "carState":
        state.car_state = msg.carState
      elif which == "carControl":
        state.car_control = msg.carControl
      elif which == "carOutput":
        state.car_output = msg.carOutput
      elif which == "controlsState":
        state.controls_state = msg.controlsState
      elif which == "selfdriveState":
        state.selfdrive_state = msg.selfdriveState
      elif which == "radarState":
        state.radar_state = msg.radarState
      elif which == "longitudinalPlan":
        if msg_ns < start_ns or msg_ns > end_ns:
          continue
        rows.append(_build_trace_row(
          plan_ns=msg_ns,
          press_ns=press_ns,
          plan=msg.longitudinalPlan,
          car_state=state.car_state,
          car_control=state.car_control,
          car_output=state.car_output,
          controls_state=state.controls_state,
          selfdrive_state=state.selfdrive_state,
          radar_state=state.radar_state,
        ))
  except Exception as exc:
    warnings.warn(f"driver-mark trace truncated at {path}: {exc!r}", RuntimeWarning, stacklevel=2)
  return first_ns, last_ns


def compute_mark_metrics(rows: list[dict[str, Any]], *, window: TraceWindow | None = None) -> dict[str, Any]:
  """Summarise one mark's trace into triage scalars.

  ``derivedCategory`` is a sorting hint for a human, not a classification. Rows
  without a planner accel (a seed frame, or a ``closed_loop`` row from before
  the planner produced output) are skipped rather than treated as zero.

  Pass ``window`` whenever the rows came from ``load_trace_window``: every scalar
  here is computed over whatever rows exist, so a caller must be told when that
  is less than the window it asked for. ``windowTruncated`` is None only when the
  caller did not supply a window and coverage is therefore unknown.
  """
  usable: list[tuple[float, dict[str, Any]]] = []
  for row in rows:
    row_t_s = _row_time(row)
    if row.get("planner_accel_mps2") is None or row_t_s is None:
      continue
    usable.append((row_t_s, row))

  min_accel: float | None = None
  min_accel_t_s: float | None = None
  peak_jerk: float | None = None
  previous: tuple[float, float] | None = None
  for t_s, row in usable:
    accel = float(row["planner_accel_mps2"])
    if min_accel is None or accel < min_accel:
      min_accel = accel
      min_accel_t_s = t_s
    if previous is not None:
      dt_s = t_s - previous[0]
      if dt_s > 0.0:
        jerk = (accel - previous[1]) / dt_s
        if peak_jerk is None or jerk < peak_jerk:
          peak_jerk = jerk
    previous = (t_s, accel)

  gaps = [float(row["lead_d_rel_m"]) for row in rows if row.get("lead_d_rel_m") is not None]
  min_gap_m = min(gaps) if gaps else None
  min_ttc_s = _min_ttc_s(rows)
  source_transitions = _source_transition_count(rows)
  resume_lag_s = _throttle_resume_lag_s(usable)
  intervened = _driver_intervened(rows)

  return {
    "rowCount": len(rows),
    "windowTruncated": None if window is None else window.truncated,
    "windowCoverage": None if window is None else window.summary(),
    "minATargetMps2": min_accel,
    "minATargetTRelS": min_accel_t_s,
    "peakDecelJerkMps3": peak_jerk,
    "minGapM": min_gap_m,
    "minTtcS": min_ttc_s,
    "plannerSourceTransitionCount": source_transitions,
    "throttleResumeLagS": resume_lag_s,
    "driverIntervened": intervened,
    "derivedCategory": _derive_category(
      min_accel=min_accel,
      peak_jerk=peak_jerk,
      min_ttc_s=min_ttc_s,
      min_gap_m=min_gap_m,
      resume_lag_s=resume_lag_s,
    ),
  }


TRACE_LEGEND = (
  "t=press-relative s, v=vEgo, aE=aEgo, lng=carControl.longActive, ss=standstill, " +
  "aTgt=longitudinalPlan.aTarget, cc=carControl.actuators.accel, co=carOutput.actuatorsOutput.accel, " +
  "lc=controlsState.longControlState, src=longitudinalPlanSource, stop=longitudinalPlan.shouldStop, " +
  "lead=(dRel, vRel, aLeadK, modelProb)"
)


def format_ascii_trace(rows: list[dict[str, Any]], *, title: str | None = None) -> str:
  """Render a trace in the column vocabulary of docs/chauffeur/longitudinal/.

  ``lng`` and ``ss`` keep their documented meanings; ``src`` and ``stop`` are the
  additions for driver marks. ``aLeadK`` and ``modelProb`` are restored to the
  lead tuple because a mark is usually about whether the lead was believed.
  """
  lines: list[str] = []
  if title is not None:
    lines.append(title)
  lines.append(TRACE_LEGEND)
  if not rows:
    lines.append("(no frames in window)")
    return "\n".join(lines)
  for row in rows:
    lines.append(_format_trace_row(row))
  return "\n".join(lines)


def find_mark_sidecar(route_key: str,
                      press_log_mono_ns: int,
                      roots: Iterable[str | Path] = DEFAULT_MARK_SIDECAR_ROOTS) -> Path | None:
  """Locate the on-device planner-internal sidecar for one driver mark.

  Primary key is the ``bookmarkButton`` envelope logMonoTime, which is globally
  unique within a boot. ``route_key`` is a tiebreak only: it is derived from
  whatever directory layout the logs were pulled into and does not always equal
  the device's ``CurrentRoute`` string.

  Returns None when no sidecar exists -- Phase 2 detail is optional, a mark is
  never lost for want of it.
  """
  best: tuple[int, Path] | None = None
  for raw_root in roots:
    root = Path(raw_root)
    if not root.exists():
      continue
    for path in sorted(root.rglob(MARK_SIDECAR_GLOB)):
      header = read_sidecar_header(path)
      if header is None:
        continue
      try:
        delta = abs(int(header.get("pressLogMonoTime", -1)) - int(press_log_mono_ns))
      except (TypeError, ValueError):
        continue
      if delta == 0:
        return path
      if delta <= MARK_SIDECAR_JOIN_TOLERANCE_NS and str(header.get("route", "")) in (route_key, ""):
        if best is None or delta < best[0]:
          best = (delta, path)
  return None if best is None else best[1]


def read_sidecar_header(path: str | Path) -> dict[str, Any] | None:
  """Read only line 1 of a sidecar. Returns None for anything malformed."""
  try:
    with open(path, encoding="utf-8") as handle:
      header = json.loads(handle.readline())
  except (OSError, ValueError):
    return None
  if not isinstance(header, dict) or header.get("type") != "header":
    return None
  return header


def load_mark_sidecar(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
  """Load a sidecar into (header, rows-as-dicts), or None if unreadable.

  Rows on disk are positional arrays whose order is declared once by
  ``header["columns"]``; they are widened to dicts here so callers never depend
  on column order.
  """
  header = read_sidecar_header(path)
  if header is None:
    return None
  columns = header.get("columns")
  if not isinstance(columns, list) or not columns:
    return None
  rows: list[dict[str, Any]] = []
  integrity_errors: list[str] = []
  truncation_markers: list[dict[str, Any]] = []
  try:
    with open(path, encoding="utf-8") as handle:
      handle.readline()
      for line in handle:
        line = line.strip()
        if not line:
          continue
        try:
          record = json.loads(line)
        except ValueError:
          integrity_errors.append("file contains an invalid JSONL record")
          continue
        if isinstance(record, dict):
          if record.get("type") == "truncated":
            truncation_markers.append(record)
          else:
            integrity_errors.append("file contains an unexpected object record")
          continue
        if not isinstance(record, list):
          integrity_errors.append("file contains a non-array row")
          continue
        if len(record) != len(columns):
          integrity_errors.append(
            f"row has {len(record)} values but header declares {len(columns)} columns"
          )
          continue
        rows.append(dict(zip(columns, record, strict=False)))
  except OSError:
    return None

  declared_truncated = bool(header.get("truncated", False))
  declared_count = header.get("rowCount")
  if not isinstance(declared_count, int) or isinstance(declared_count, bool):
    integrity_errors.append("header rowCount is missing or not an integer")
  elif declared_count != len(rows):
    integrity_errors.append(f"rowCount declares {declared_count} but file contains {len(rows)} valid rows")
  if len(truncation_markers) > 1:
    integrity_errors.append("file contains multiple truncation markers")
  if truncation_markers and not declared_truncated:
    integrity_errors.append("file contains a truncation marker but header declares truncated false")
  if header.get("truncatedAtByte") is not None and not truncation_markers:
    integrity_errors.append("header declares byte truncation but the trailing marker is missing")

  # Return an enriched copy so callers cannot accidentally print a declared
  # complete status over a short/corrupt body. Preserve the device declaration
  # separately for diagnosis.
  checked_header = dict(header)
  checked_header["declaredTruncated"] = declared_truncated
  checked_header["integrityStatus"] = "truncated" if integrity_errors else "ok"
  checked_header["integrityErrors"] = integrity_errors
  checked_header["truncated"] = declared_truncated or bool(integrity_errors)
  return checked_header, rows


def join_sidecar_rows(rows: list[dict[str, Any]], sidecar_rows: list[dict[str, Any]]) -> int:
  """Attach sidecar planner-internal state to trace rows by modelV2 logMonoTime.

  ``longitudinalPlan.modelMonoTime`` is byte-identical to the sidecar's
  ``modelLogMonoTime``, so this is an exact integer join with no tolerance.
  Mutates ``rows`` in place and returns the number of rows matched.
  """
  by_model_ns: dict[int, dict[str, Any]] = {}
  for sidecar_row in sidecar_rows:
    model_ns = sidecar_row.get("modelLogMonoTime")
    if isinstance(model_ns, int):
      by_model_ns[model_ns] = sidecar_row
  matched = 0
  for row in rows:
    trace_model_ns = row.get("model_log_mono_time_ns")
    if not isinstance(trace_model_ns, int):
      continue
    joined = by_model_ns.get(trace_model_ns)
    if joined is not None:
      row["sidecar"] = joined
      matched += 1
  return matched


SIDECAR_SUMMARY_COLUMNS = (
  ("relatchActive", "relatch"),
  ("handoffActive", "handoff"),
  ("comfortActive", "comfort"),
  ("brakeReleaseActive", "release"),
  ("bridgeApplied", "bridge"),
  ("steadyParityActive", "parity"),
  ("cruiseReacquireActive", "reacq"),
)


# The scalar columns are looked up by a list of candidate names on purpose: the
# on-device schema is owned by longitudinal_mark_recorder.COLUMNS and has already
# renamed two of these once. A rename must degrade to a missing field, never a
# crash and never a silently blank line.
SIDECAR_SCALAR_COLUMNS = (
  (("accSourceActiveMode", "accSourceState"), "acc"),
  (("accSourceReason",), "accWhy"),
  (("leadRoleLead0", "leadRoleSource"), "role"),
  (("mpcSource",), "mpc"),
)


def format_sidecar_row(sidecar_row: dict[str, Any]) -> str:
  """One compact line of planner-internal gate state for a joined trace row."""
  active = [label for column, label in SIDECAR_SUMMARY_COLUMNS if sidecar_row.get(column)]
  parts = [f"gates={','.join(active) if active else '-'}"]
  for candidates, label in SIDECAR_SCALAR_COLUMNS:
    for column in candidates:
      value = sidecar_row.get(column)
      if value is not None:
        parts.append(f"{label}={value}")
        break
  return "      dbg " + " ".join(parts)


def _build_trace_row(*,
                     plan_ns: int,
                     press_ns: int,
                     plan: Any,
                     car_state: Any,
                     car_control: Any,
                     car_output: Any,
                     controls_state: Any,
                     selfdrive_state: Any,
                     radar_state: Any) -> dict[str, Any]:
  lead = _primary_lead(radar_state)
  return {
    "t_rel_s": (plan_ns - press_ns) / 1e9,
    "log_mono_time_ns": plan_ns,
    "model_log_mono_time_ns": int(plan.modelMonoTime),
    "planner_accel_mps2": float(plan.aTarget),
    "planner_should_stop": bool(plan.shouldStop),
    "planner_source": str(plan.longitudinalPlanSource),
    "planner_fcw": bool(plan.fcw),
    "planner_has_lead": bool(plan.hasLead),
    "v_ego_mps": None if car_state is None else float(car_state.vEgo),
    "a_ego_mps2": None if car_state is None else float(car_state.aEgo),
    "gas_pressed": None if car_state is None else bool(car_state.gasPressed),
    "brake_pressed": None if car_state is None else bool(car_state.brakePressed),
    "standstill": None if car_state is None else bool(car_state.standstill),
    "long_active": None if car_control is None else bool(car_control.longActive),
    "controller_accel_mps2": None if car_control is None else float(car_control.actuators.accel),
    "output_accel_mps2": None if car_output is None else float(car_output.actuatorsOutput.accel),
    "long_control_state": None if controls_state is None else str(controls_state.longControlState),
    "enabled": None if selfdrive_state is None else bool(selfdrive_state.enabled),
    "lead_d_rel_m": None if lead is None else float(lead.dRel),
    "lead_v_rel_mps": None if lead is None else float(lead.vRel),
    "lead_a_lead_k_mps2": None if lead is None else float(lead.aLeadK),
    "lead_model_prob": None if lead is None else float(lead.modelProb),
  }


def _primary_lead(radar_state: Any) -> Any:
  if radar_state is None:
    return None
  leads = [lead for lead in (radar_state.leadOne, radar_state.leadTwo) if lead.status]
  if not leads:
    return None
  return min(leads, key=lambda lead: float(lead.dRel))


def _format_trace_row(row: dict[str, Any]) -> str:
  parts = [
    f"{_fmt(_row_time(row), '{:6.2f}')}",
    f"v={_fmt(row.get('v_ego_mps'), '{:.1f}')}",
    f"aE={_fmt(row.get('a_ego_mps2'), '{:+.2f}')}",
    f"lng={_flag(row.get('long_active'))}",
    f"ss={_flag(row.get('standstill'))}",
    f"aTgt={_fmt(row.get('planner_accel_mps2'), '{:+.2f}')}",
    f"cc={_fmt(row.get('controller_accel_mps2'), '{:+.2f}')}",
    f"co={_fmt(row.get('output_accel_mps2'), '{:+.2f}')}",
    f"lc={row.get('long_control_state') or '-'}",
    f"src={row.get('planner_source') or '-'}",
    f"stop={_flag(row.get('planner_should_stop'))}",
  ]
  if row.get("gas_pressed"):
    parts.append("gas=1")
  if row.get("brake_pressed"):
    parts.append("brake=1")
  lead_d_rel = row.get("lead_d_rel_m")
  if lead_d_rel is None:
    parts.append("lead=none")
  else:
    parts.append(
      "lead=(" +
      f"{_fmt(lead_d_rel, '{:.1f}')}, " +
      f"{_fmt(row.get('lead_v_rel_mps'), '{:+.2f}')}, " +
      f"{_fmt(row.get('lead_a_lead_k_mps2'), '{:+.2f}')}, " +
      f"{_fmt(row.get('lead_model_prob'), '{:.2f}')})"
    )
  line = " ".join(parts)
  sidecar_row = row.get("sidecar")
  if isinstance(sidecar_row, dict):
    line = line + "\n" + format_sidecar_row(sidecar_row)
  return line


def _fmt(value: Any, spec: str) -> str:
  if value is None:
    return "--"
  try:
    return spec.format(float(value))
  except (TypeError, ValueError):
    return "--"


def _flag(value: Any) -> str:
  if value is None:
    return "-"
  return "1" if value else "0"


def _row_time(row: dict[str, Any]) -> float | None:
  for key in ("t_rel_s", "t_s"):
    value = row.get(key)
    if value is not None:
      return float(value)
  return None


def _min_ttc_s(rows: list[dict[str, Any]]) -> float | None:
  best: float | None = None
  for row in rows:
    d_rel = row.get("lead_d_rel_m")
    v_rel = row.get("lead_v_rel_mps")
    if d_rel is None or v_rel is None:
      continue
    closing = -float(v_rel)
    if closing < CLOSING_SPEED_FLOOR_MPS:
      continue
    ttc = float(d_rel) / closing
    if not math.isfinite(ttc):
      continue
    if best is None or ttc < best:
      best = ttc
  return best


def _source_transition_count(rows: list[dict[str, Any]]) -> int:
  transitions = 0
  previous: str | None = None
  for row in rows:
    source = row.get("planner_source")
    if source is None:
      continue
    if previous is not None and source != previous:
      transitions += 1
    previous = source
  return transitions


def _throttle_resume_lag_s(timed_rows: list[tuple[float, dict[str, Any]]]) -> float | None:
  """Seconds between the lead visibly opening and the planner commanding accel.

  Deliberately anchored after the deepest braking of the window: the question a
  'sluggish' mark asks is how long the car stayed on the brake after the reason
  for it went away.
  """
  if not timed_rows:
    return None
  min_idx = min(range(len(timed_rows)), key=lambda idx: float(timed_rows[idx][1]["planner_accel_mps2"]))
  open_idx: int | None = None
  for idx in range(min_idx, len(timed_rows)):
    v_rel = timed_rows[idx][1].get("lead_v_rel_mps")
    if v_rel is not None and float(v_rel) >= LEAD_OPENING_V_REL_MPS:
      open_idx = idx
      break
  if open_idx is None:
    return None
  for idx in range(open_idx, len(timed_rows)):
    if float(timed_rows[idx][1]["planner_accel_mps2"]) >= 0.0:
      return max(0.0, timed_rows[idx][0] - timed_rows[open_idx][0])
  return None


def _driver_intervened(rows: list[dict[str, Any]]) -> bool:
  previous_long_active: bool | None = None
  for row in rows:
    if row.get("gas_pressed") or row.get("brake_pressed"):
      return True
    long_active = row.get("long_active")
    if previous_long_active is True and long_active is False:
      return True
    if long_active is not None:
      previous_long_active = bool(long_active)
  return False


def _derive_category(*,
                     min_accel: float | None,
                     peak_jerk: float | None,
                     min_ttc_s: float | None,
                     min_gap_m: float | None,
                     resume_lag_s: float | None) -> str:
  if (min_accel is not None and min_accel <= HARSH_BRAKE_ACCEL_MPS2) or \
     (peak_jerk is not None and peak_jerk <= HARSH_BRAKE_JERK_MPS3):
    return "harsh_brake"
  if (min_ttc_s is not None and min_ttc_s <= LATE_BRAKE_TTC_S) or \
     (min_gap_m is not None and min_gap_m <= LATE_BRAKE_GAP_M):
    return "late_brake"
  if resume_lag_s is not None and resume_lag_s >= SLUGGISH_RESUME_LAG_S:
    return "sluggish"
  return "unspecified"


def _nearest_payload(payloads: list[dict[str, Any]], press_ns: int) -> dict[str, Any] | None:
  window_ns = int(DRIVER_MARK_PAYLOAD_WINDOW_S * 1e9)
  best: tuple[int, dict[str, Any]] | None = None
  for entry in payloads:
    delta = abs(int(entry["logMonoTimeNs"]) - press_ns)
    if delta > window_ns:
      continue
    if best is None or delta < best[0]:
      best = (delta, entry["payload"])
  return None if best is None else best[1]


def _has_frame_within(frame_times_ns: list[int], press_ns: int, tolerance_s: float) -> bool:
  tolerance_ns = int(tolerance_s * 1e9)
  return any(abs(time_ns - press_ns) <= tolerance_ns for time_ns in frame_times_ns)
