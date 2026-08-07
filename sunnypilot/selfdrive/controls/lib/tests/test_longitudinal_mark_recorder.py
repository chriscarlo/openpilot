"""Tests for the Phase 2 driver-mark planner-internal recorder.

Mirrors the _FakeParams / _FakeClock / injected-thread idiom already established
by sunnypilot/selfdrive/controls/lib/tests/vtsc/test_planner_lag_debug.py. Every
case is hermetic: no device, no thread, and no disk outside tmp_path.
"""

from __future__ import annotations

import ast
import builtins
import errno
import gc
import inspect
import json
import logging
import os
import queue
import textwrap
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import longitudinal_mark_recorder as mr

# Imported at MODULE level on purpose. sunnypilot/selfdrive/controls/lib/tests/rti/
# test_rti_rampdown_unit.py installs never-restored sys.modules stubs for 'cereal'
# and 'opendbc.car' from inside its test bodies, so a same-session run of the whole
# directory would break these imports if they were deferred into a fixture.
# Collection (and therefore this import) happens before any test body runs.
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner

TICK_S = 1.0 / mr.EXPECTED_HZ
TICK_NS = int(TICK_S * 1e9)

# Stands in for plannerd's SCHED_FIFO loop thread, so a test can assert that a
# given call NEVER happens on it.
RT_THREAD_NAME = "planner-rt-standin"


class _RecordingCloudlog:
  """Records ``(method, thread name)`` for every cloudlog call the module makes."""

  def __init__(self):
    self.calls: list[tuple[str, str]] = []

  def __getattr__(self, name: str):
    def _fn(*_a, **_k):
      self.calls.append((name, threading.current_thread().name))
    return _fn

  def on(self, thread_name: str) -> list[str]:
    return [method for method, thread in self.calls if thread == thread_name]


def _on_rt_thread(fn):
  """Run ``fn`` on a thread named RT_THREAD_NAME and re-raise whatever it raised."""
  box: list[Any] = []

  def _body():
    try:
      box.append(fn())
    except BaseException as exc:  # re-raised on the calling thread below
      box.append(exc)

  thread = threading.Thread(target=_body, name=RT_THREAD_NAME)
  thread.start()
  thread.join(30.0)
  assert not thread.is_alive(), "realtime stand-in thread hung"
  if box and isinstance(box[0], BaseException):
    raise box[0]
  return box[0] if box else None


def _live_writer_threads() -> int:
  return sum(1 for t in threading.enumerate() if t.name == mr.WRITER_THREAD_NAME and t.is_alive())


def _wait_for_writer_threads(target: int, timeout_s: float = 10.0) -> int:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if _live_writer_threads() <= target:
      break
    time.sleep(0.02)
  return _live_writer_threads()


class _FakeParams:
  def __init__(self, *, enabled: bool = True, route: str = "000001a3--c20ba54385"):
    self.enabled = enabled
    self.route = route
    self.get_bool_calls = 0

  def get_bool(self, key: str) -> bool:
    assert key == mr.ENABLE_PARAM
    self.get_bool_calls += 1
    return bool(self.enabled)

  def get(self, key: str):
    if key == "CurrentRoute":
      return self.route
    if key == "GitCommit":
      return "deadbeef"
    if key == "GitBranch":
      return "chauffeur-exp01"
    return None


class _FakeClock:
  """Exact clock: accumulates in integer nanoseconds so N ticks of 50 ms is
  bit-exactly N*0.05 s. Repeated float addition drifts enough to move a flush by
  a whole tick, which would make the window/rate-limit assertions luck-based."""

  def __init__(self, now_s: float = 0.0):
    self._ns = int(round(now_s * 1e9))

  def now(self) -> float:
    return self._ns / 1e9

  def advance(self, dt_s: float) -> None:
    self._ns += int(round(dt_s * 1e9))


class _DeadThread:
  """thread_factory whose start() is a no-op: simulates a writer that never runs."""

  started = False

  def __init__(self, *, target=None, daemon=None, name=None, **_ignored):
    self._target = target

  def start(self) -> None:
    _DeadThread.started = True


class _InlineThread:
  """Runs nothing on start(); the test drains the queue explicitly."""

  def __init__(self, *, target=None, daemon=None, name=None, **_ignored):
    self._target = target

  def start(self) -> None:
    pass


def _counting_thread_factory(starts: list[str]):
  """thread_factory that records every successful start(), so a test can prove
  no Thread.start() ever lands on the flush/press path."""

  class _Counting:
    def __init__(self, *, target=None, daemon=None, name=None, **_ignored):
      self._name = name

    def start(self) -> None:
      starts.append(self._name)

  return _Counting


class _Obj:
  def __init__(self, **kw):
    self.__dict__.update(kw)


class _FakeTuneCfg:
  @staticmethod
  def as_dict() -> dict[str, Any]:
    return {"lead_acquire_window_s": 0.75, "gap_reclaim_max_accel": 0.8}


class _FakeSM:
  """Minimal SubMaster stand-in with the exact surface _build_row touches."""

  def __init__(self):
    self.frame = 0
    self.logMonoTime = {"modelV2": 0, "radarState": 0, "carState": 0, "bookmarkButton": 0}
    self.updated = {"modelV2": True, "radarState": True, "carState": True, "bookmarkButton": False}
    self._data = {
      "carState": _Obj(vEgo=20.0, aEgo=0.1, gasPressed=False),
      "carControl": _Obj(longActive=True),
    }

  def __getitem__(self, s: str):
    return self._data[s]

  def tick(self, mono_ns: int) -> None:
    self.frame += 1
    self.logMonoTime["modelV2"] = mono_ns
    self.logMonoTime["radarState"] = mono_ns - 3_000_000
    self.logMonoTime["carState"] = mono_ns - 1_000_000
    self.updated["bookmarkButton"] = False

  def press(self, mono_ns: int) -> None:
    self.updated["bookmarkButton"] = True
    self.logMonoTime["bookmarkButton"] = mono_ns


def _fake_planner(*, comfort: dict | None = None, release: dict | None = None) -> Any:
  mpc = _Obj(
    source="lead0",
    mode="acc",
    lead_role_debug={
      "gate_active": True,
      "duplicate_pair": False,
      "dropped_slot": None,
      "roles": {"lead0": "center_control", "lead1": "invalid"},
      "cutin_promoted": {"lead0": False, "lead1": False},
    },
    acc_source_debug={"active_mode": "lead", "reason": "dropout_hold"},
    control_leads=(
      _Obj(status=True, dRel=29.0, vRel=-0.31, modelProb=0.95),
      _Obj(status=False, dRel=0.0, vRel=0.0, modelProb=0.0),
    ),
    _live_tune_cfg=_FakeTuneCfg(),
  )
  if comfort is None:
    comfort = {"active": True, "bypassed": False, "bypass_reason": "", "gated_reason": "",
               "max_step_mps2": 0.06, "clipped": False, "upward_floor_owner": "brake_release",
               "upward_max_delta_mps2": 0.4, "upward_step_mps2": 0.06,
               "release_slew_active": False, "release_slew_clipped": False}
  if release is None:
    release = {"active": True, "reason": "closing_recovery", "output_bound": True,
               "closing_recovery_bridge": {"reason": "applied", "applied": True,
                                           "desired_output_mps2": -0.2,
                                           "output_uplift_applied_mps2": 0.1}}
  return _Obj(
    mpc=mpc,
    CP=_Obj(carFingerprint="KIA_EV6"),
    relatch_blend_debug={"active": False, "frames_left": 0, "bypassed": False,
                         "bypass_reason": "", "neg_cap_mps2": -3.5, "clipped": False},
    handoff_limit_debug={"active": False, "frames_left": 0, "down_bypassed": False,
                         "bypass_reason": "", "opening_cap_appeared": False,
                         "edge1_capped": False, "clipped": False},
    comfort_jerk_debug=comfort,
    lead_brake_release_debug=release,
    steady_parity_threat_debug={"active": False, "pre_comfort_limiter_output_mps2": None,
                                "final_output_mps2": -0.31, "safety_cap_applied": False},
    cruise_reacquire_debug={"active": False, "frames_left": 0, "allowed_jerk_mps3": 0.0,
                            "slew_ceiling_mps2": 0.0, "clipped": False, "exit_cause": ""},
    lead_brake_release_accel_floor=-0.3,
    lead_slowdown_arbitration_debug={
      "reason": "uncorroborated_false_brake_mpc_authority",
      "urgent": False,
      "raw_ceiling_mps2": -1.0,
      "effective_ceiling_mps2": -0.25,
      "mpc_accel_mps2": -0.25,
      "model_accel_mps2": 0.05,
    },
    prev_accel_clip=[-3.5, 1.6],
    effective_v_cruise_mps=29.0,
    output_a_target=-0.31,
    output_should_stop=False,
    a_desired=-0.3,
    v_desired_filter=_Obj(x=19.8),
    fcw=False,
  )


def _make(tmp_path: Path, *, params=None, clock=None, thread_factory=_InlineThread):
  params = params if params is not None else _FakeParams()
  clock = clock if clock is not None else _FakeClock()
  rec = mr.MarkRecorder(
    params=params,
    marks_dir=tmp_path / "LongMarks",
    realdata_dir=tmp_path / "realdata",
    time_fn=clock.now,
    thread_factory=thread_factory,
  )
  return rec, params, clock


def _run(rec, sm, planner, clock, *, ticks: int, press_at: int | None = None, start_ns: int = 10_000_000_000):
  """Drive `ticks` 20 Hz cycles; optionally press the flag button on tick `press_at`."""
  press_ns = None
  for i in range(ticks):
    mono_ns = start_ns + i * TICK_NS
    sm.tick(mono_ns)
    if press_at is not None and i == press_at:
      press_ns = mono_ns
      sm.press(mono_ns)
    rec.update(planner, sm)
    clock.advance(TICK_S)
  return press_ns


# --------------------------------------------------------------- schema/ring


def test_columns_arity_matches_row(tmp_path):
  rec, _, _ = _make(tmp_path)
  sm = _FakeSM()
  sm.tick(10_000_000_000)
  row = mr.MarkRecorder._build_row(_fake_planner(), sm)
  assert len(row) == len(mr.COLUMNS)
  assert len(set(mr.COLUMNS)) == len(mr.COLUMNS)
  assert mr.COLUMNS[0] == "modelLogMonoTime"
  assert rec is not None


def test_slowdown_arbitration_is_recorded_with_the_mark() -> None:
  sm = _FakeSM()
  sm.tick(10_000_000_000)
  row = mr.MarkRecorder._build_row(_fake_planner(), sm)

  assert row[mr.COLUMNS.index("slowdownArbitrationReason")] == "uncorroborated_false_brake_mpc_authority"
  assert row[mr.COLUMNS.index("slowdownArbitrationUrgent")] is False
  assert row[mr.COLUMNS.index("slowdownRawCeilingMps2")] == -1.0
  assert row[mr.COLUMNS.index("slowdownEffectiveCeilingMps2")] == -0.25
  assert row[mr.COLUMNS.index("slowdownMpcAccelMps2")] == -0.25
  assert row[mr.COLUMNS.index("slowdownModelAccelMps2")] == 0.05


def test_ring_is_bounded(tmp_path):
  import math

  assert mr.RING_MAXLEN == max(600, math.ceil((mr.PRE_WINDOW_S + mr.POST_WINDOW_S) * mr.EXPECTED_HZ * 1.5))
  assert mr.RING_MAXLEN == 780
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=5000)
  assert len(rec._ring) == mr.RING_MAXLEN


def test_no_unreachable_row_clamp(tmp_path):
  """MAX_ROWS_PER_MARK == RING_MAXLEN made ``if len(rows) > MAX_ROWS_PER_MARK``
  dead code: rows is filtered out of a deque whose maxlen IS RING_MAXLEN."""
  assert not hasattr(mr, "MAX_ROWS_PER_MARK")
  assert "MAX_ROWS_PER_MARK" not in inspect.getsource(mr.MarkRecorder._service_pending)
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=3000, press_at=2000)
  _mark, rows = rec._queue.get_nowait()
  assert 0 < len(rows) <= mr.RING_MAXLEN


def test_rows_join_to_model_mono_time(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  _run(rec, sm, planner, clock, ticks=50, start_ns=start)
  idx = mr.COLUMNS.index("modelLogMonoTime")
  assert [r[idx] for r in rec._ring] == [start + i * TICK_NS for i in range(50)]
  ridx = mr.COLUMNS.index("radarStateLogMonoTime")
  assert rec._ring[0][ridx] == start - 3_000_000


# ------------------------------------------------------------------- enable


def test_disabled_param_captures_nothing(tmp_path):
  _DeadThread.started = False
  rec, params, clock = _make(tmp_path, params=_FakeParams(enabled=False), thread_factory=_DeadThread)
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=200, press_at=50)
  assert len(rec._ring) == 0
  assert rec._pending == []
  assert rec._queue.empty()
  # The writer is spawned unconditionally now: it is the ONLY thread allowed to
  # read Params, so gating the spawn on the enable read is precisely what pushed
  # Thread.start() onto the realtime thread on the next successful refresh.
  assert _DeadThread.started is True
  assert rec._writer is not None


def test_param_reread_cadence(tmp_path):
  """The Params round-trip lives on the writer thread. The realtime loop must do
  none at all, and the writer must not do one per wakeup either."""
  rec, params, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=200)  # 10 s of simulated time on the RT thread
  assert params.get_bool_calls == 1        # exactly the construction read

  # 10 s of writer wakeups at WRITER_POLL_INTERVAL_S = one poll per REFRESH_INTERVAL_S.
  for _ in range(int(10.0 / mr.WRITER_POLL_INTERVAL_S)):
    rec._writer_housekeeping()
    clock.advance(mr.WRITER_POLL_INTERVAL_S)
  assert params.get_bool_calls - 1 == int(10.0 / mr.REFRESH_INTERVAL_S)


def test_live_toggle_off_clears_state(tmp_path):
  rec, params, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=100, press_at=90)
  assert len(rec._ring) > 0 and len(rec._pending) == 1
  params.enabled = False
  rec._writer_housekeeping()               # the writer observes the toggle...
  assert rec._enabled_requested is False
  assert len(rec._ring) > 0                # ...but only the RT thread may clear its own ring
  _run(rec, sm, planner, clock, ticks=60, start_ns=10_000_000_000 + 100 * TICK_NS)
  assert rec._enabled is False
  assert len(rec._ring) == 0
  assert rec._pending == []


# ------------------------------------------------------------------ windows


def test_window_bounds_exact(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  press_ns = _run(rec, sm, planner, clock, ticks=1000, press_at=500, start_ns=start)
  mark, rows = rec._queue.get_nowait()
  assert mark["pressLogMonoTime"] == press_ns
  assert len(rows) == 521
  assert rows[0][0] == press_ns - int(mr.PRE_WINDOW_S * 1e9)
  assert rows[-1][0] == press_ns + int(mr.POST_WINDOW_S * 1e9)


def test_flush_waits_for_post_window(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  # 500 pre-roll ticks then the press, then exactly 119 more ticks (5.95 s).
  _run(rec, sm, planner, clock, ticks=500 + 1 + 119, press_at=500, start_ns=start)
  assert rec._queue.empty()
  assert len(rec._pending) == 1
  # One more tick crosses press + 6.0 s.
  _run(rec, sm, planner, clock, ticks=1, start_ns=start + 620 * TICK_NS)
  assert rec._queue.qsize() == 1
  assert rec._pending == []


def test_short_window_at_route_start(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  # Press 5 s (100 ticks) after the ring starts filling.
  _run(rec, sm, planner, clock, ticks=100 + 1 + 120, press_at=100, start_ns=start)
  mark, rows = rec._queue.get_nowait()
  assert len(rows) == 221          # 100 pre + press + 120 post
  assert rows[0][0] == start       # only 5 s of pre-roll existed
  rec._write_mark(mark, rows)
  path = next((tmp_path / "LongMarks").glob("longmark__*.jsonl"))
  header = json.loads(path.read_text().splitlines()[0])
  assert header["rowCount"] == 221
  assert header["firstModelLogMonoTime"] == start
  assert header["lastModelLogMonoTime"] == mark["pressLogMonoTime"] + int(mr.POST_WINDOW_S * 1e9)
  assert header["truncated"] is True
  assert header["truncatedEdges"] == ["start"]


# ------------------------------------------------------------------ debounce


def test_debounce_collapses_sub_second_presses(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  for i in range(500):
    mono_ns = start + i * TICK_NS
    sm.tick(mono_ns)
    if i in (400, 408):  # 0.4 s apart
      sm.press(mono_ns)
    rec.update(planner, sm)
    clock.advance(TICK_S)
  assert len(rec._pending) == 1
  assert rec._pending[0]["pressLogMonoTime"] == start + 400 * TICK_NS


def test_first_press_is_not_debounced_against_the_zero_sentinel(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  press_ns = 500_000_000
  sm.tick(press_ns)
  sm.press(press_ns)
  rec.update(planner, sm)
  clock.advance(TICK_S)

  assert len(rec._pending) == 1
  assert rec._pending[0]["pressLogMonoTime"] == press_ns


def test_two_presses_over_one_second_are_two_marks(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  first, second = 500, 540  # 2.0 s apart
  for i in range(900):
    mono_ns = start + i * TICK_NS
    sm.tick(mono_ns)
    if i in (first, second):
      sm.press(mono_ns)
    rec.update(planner, sm)
    clock.advance(TICK_S)
  assert rec._queue.qsize() == 2
  mark_a, rows_a = rec._queue.get_nowait()
  mark_b, rows_b = rec._queue.get_nowait()
  assert mark_a["pressLogMonoTime"] == start + first * TICK_NS
  assert mark_b["pressLogMonoTime"] == start + second * TICK_NS
  assert len(rows_a) == 521 and len(rows_b) == 521


def test_pending_cap_drops_extra_presses(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  presses = {500, 525, 550, 575, 600}  # five presses inside 5 s, all > 1 s apart
  for i in range(605):
    mono_ns = start + i * TICK_NS
    sm.tick(mono_ns)
    if i in presses:
      sm.press(mono_ns)
    rec.update(planner, sm)
    clock.advance(TICK_S)
  assert len(rec._pending) == mr.MAX_PENDING_MARKS
  assert rec._dropped_marks == 1


# ------------------------------------------------------- missing-key defense


def test_missing_debug_keys_do_not_raise(tmp_path):
  """The gated comfort branch lacks upward_floor_owner / upward_max_delta / upward_step."""
  gated = {"active": False, "bypassed": False, "bypass_reason": "", "gated_reason": "handoff_limiter",
           "max_step_mps2": 0.06, "clipped": False, "release_slew_active": False,
           "release_slew_clipped": False}
  sm = _FakeSM()
  sm.tick(10_000_000_000)
  row = mr.MarkRecorder._build_row(_fake_planner(comfort=gated), sm)
  assert len(row) == len(mr.COLUMNS)
  for col in ("comfortUpwardFloorOwner", "comfortUpwardMaxDeltaMps2", "comfortUpwardStepMps2"):
    assert row[mr.COLUMNS.index(col)] is None
  assert row[mr.COLUMNS.index("comfortGatedReason")] == "handoff_limiter"


def test_nested_bridge_absent_does_not_raise(tmp_path):
  """The __init__ shape of lead_brake_release_debug has no closing_recovery_bridge."""
  sm = _FakeSM()
  sm.tick(10_000_000_000)
  row = mr.MarkRecorder._build_row(_fake_planner(release={"active": False, "reason": "init"}), sm)
  assert len(row) == len(mr.COLUMNS)
  for col in ("bridgeReason", "bridgeApplied", "bridgeDesiredOutputMps2", "bridgeOutputUpliftMps2"):
    assert row[mr.COLUMNS.index(col)] is None
  assert row[mr.COLUMNS.index("brakeReleaseReason")] == "init"
  assert row[mr.COLUMNS.index("brakeReleaseOutputBound")] is None


# ------------------------------------------------------------ realtime safety


def test_capture_does_no_io_and_survives_dead_writer(tmp_path, monkeypatch):
  """The realtime-safety test: no file I/O on the capture path, and a writer that
  never runs degrades the hook to a bounded, logged no-op."""

  def _boom(*_a, **_k):
    raise AssertionError("I/O on realtime path")

  # Path.open and os.open are load-bearing members of this list, not padding: a
  # leak through either does real disk I/O on the SCHED_FIFO thread while
  # builtins.open stays untouched. Proven by mutation.
  monkeypatch.setattr(builtins, "open", _boom)
  monkeypatch.setattr(os, "open", _boom)
  monkeypatch.setattr(os, "replace", _boom)
  monkeypatch.setattr(os, "listdir", _boom)
  monkeypatch.setattr(os, "makedirs", _boom)
  monkeypatch.setattr(Path, "mkdir", _boom)
  monkeypatch.setattr(Path, "open", _boom)

  _DeadThread.started = False
  rec, _, clock = _make(tmp_path, thread_factory=_DeadThread)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  # Track the PEAK error count, not just the final one: update() swallows its own
  # exceptions and a later good tick resets the counter, so a leaked open() in the
  # capture path would otherwise be invisible at the end of the run.
  peak_errors = 0
  for i in range(1000):
    mono_ns = start + i * TICK_NS
    sm.tick(mono_ns)
    if i == 500:
      sm.press(mono_ns)
    assert rec.update(planner, sm) is None
    peak_errors = max(peak_errors, rec._capture_errors)
    clock.advance(TICK_S)

  assert peak_errors == 0                # nothing raised on the capture path at any point
  assert rec._dead is False
  assert rec._capture_errors == 0
  assert rec._queue.qsize() == 1
  assert rec._dropped_marks == 0
  assert _DeadThread.started is True     # the thread was created, it just never ran
  ring_len_before = len(rec._ring)

  # Five more marks against a writer that never drains: 3 fit, 2 are dropped.
  next_start = start + 1000 * TICK_NS
  for k in range(5):
    seg_start = next_start + k * 200 * TICK_NS
    for i in range(200):
      mono_ns = seg_start + i * TICK_NS
      sm.tick(mono_ns)
      if i == 0:
        sm.press(mono_ns)
      rec.update(planner, sm)
      clock.advance(TICK_S)

  assert rec._queue.qsize() == mr.FLUSH_QUEUE_MAXSIZE
  assert rec._dropped_marks == 2
  assert rec._dead is False
  assert len(rec._ring) == ring_len_before == mr.RING_MAXLEN


def test_realtime_path_takes_no_logging_lock_and_calls_no_cloudlog(tmp_path, monkeypatch):
  """Measured on the pre-fix code driving this exact scenario: 2 logging-handler
  RLock acquisitions and 1 handler emit on the planner thread.

  cloudlog is a stdlib Logger — emitting acquires the StreamHandler and
  UnixDomainSocketHandler RLocks and writes to stderr. A SCHED_FIFO thread that
  blocks on a lock a SCHED_OTHER thread holds is exactly the unbounded stall the
  writer demotion made MORE likely, not less.
  """
  log = _RecordingCloudlog()
  monkeypatch.setattr(mr, "cloudlog", log)

  acquires: list[str] = []
  real_acquire = logging.Handler.acquire

  def _acquire(self):
    acquires.append(threading.current_thread().name)
    return real_acquire(self)

  monkeypatch.setattr(logging.Handler, "acquire", _acquire)

  rec, _, clock = _make(tmp_path)             # construction is off the realtime thread
  sm, planner = _FakeSM(), _fake_planner()
  for _ in range(mr.FLUSH_QUEUE_MAXSIZE):     # force the queue-full drop path
    rec._queue.put_nowait(({"pressLogMonoTime": 1}, []))

  def _drive():
    start = 10_000_000_000
    for i in range(300):
      mono_ns = start + i * TICK_NS
      sm.tick(mono_ns)
      if i in (0, 40, 80, 120):
        sm.press(mono_ns)
        rec.latch_press(sm)
      rec.update(planner, sm)
      clock.advance(TICK_S)
    # ...and the self-disable path on top, which used to cloudlog.exception().
    for _ in range(mr.MAX_CONSECUTIVE_CAPTURE_ERRORS + 2):
      rec.update(_Obj(), sm)

  _on_rt_thread(_drive)

  assert rec._dropped_marks >= 4              # the scenario really did drop marks
  assert rec._dead is True                    # and really did hit the self-disable path
  assert log.on(RT_THREAD_NAME) == []         # no cloudlog call at all from the planner thread
  assert [t for t in acquires if t == RT_THREAD_NAME] == []

  # None of it is silent: the writer reports both, off the realtime thread.
  clock.advance(mr.DROP_LOG_INTERVAL_S + 1.0)
  rec._writer_housekeeping()
  reported = log.on(threading.current_thread().name)
  assert reported.count("error") >= 2         # the death one-shot plus the drop counter


def test_realtime_methods_never_name_cloudlog_params_or_thread_start():
  """Static companion to the runtime measurement above: the measurement can only
  prove the paths it drives, this covers every branch of the realtime call graph
  including the ones a test never reaches."""
  banned = {"cloudlog", "_read_enabled", "_start_writer", "_poll_enable", "_drain_deferred_logs",
            "_write_mark", "_params", "close"}
  for method in (mr.MarkRecorder.update, mr.MarkRecorder.drain_press_socket, mr.MarkRecorder._latch_press_mono,
                 mr.MarkRecorder.latch_press, mr.MarkRecorder._apply_enable_request, mr.MarkRecorder._arm,
                 mr.MarkRecorder._service_pending, mr.MarkRecorder._build_row):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    leaked = sorted(banned & names)
    assert leaked == [], f"{method.__qualname__} reaches {leaked} on the realtime thread"


def test_queue_full_drops_are_counted_on_rt_and_reported_by_the_writer(tmp_path, monkeypatch):
  """The realtime thread only counts; the writer emits. cloudlog is a stdlib
  Logger, so emitting on the SCHED_FIFO thread means taking the StreamHandler and
  UnixDomainSocketHandler RLocks against a SCHED_OTHER holder."""
  calls: list[tuple] = []
  monkeypatch.setattr(mr.cloudlog, "error", lambda *a, **k: calls.append(a))

  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  for _ in range(mr.FLUSH_QUEUE_MAXSIZE):
    rec._queue.put_nowait(({"pressLogMonoTime": 1}, []))
  assert rec._queue.full()

  start = 10_000_000_000
  # Presses at t=0.0 s, 1.5 s and 5.0 s flush at 6.0 s, 7.5 s and 11.0 s: the
  # first two land inside this loop, the third in the next one.
  for i in range(200):
    mono_ns = start + i * TICK_NS
    sm.tick(mono_ns)
    if i in (0, 30, 100):
      sm.press(mono_ns)
    assert rec.update(planner, sm) is None
    clock.advance(TICK_S)
  assert rec._dropped_marks == 2
  assert rec._dropped_queue_full == 2
  assert calls == []                       # nothing was logged from the realtime path

  rec._writer_housekeeping()               # clock is at 10.0 s
  assert len(calls) == 1
  assert calls[0][1] == 2                  # new since the last report
  assert calls[0][2] == 2                  # running total

  # The third drop, inside DROP_LOG_INTERVAL_S: rate limited but still counted.
  for i in range(200, 240):
    mono_ns = start + i * TICK_NS
    sm.tick(mono_ns)
    assert rec.update(planner, sm) is None
    clock.advance(TICK_S)
  rec._writer_housekeeping()               # clock is at 12.0 s, < 10 s since the last log
  assert rec._dropped_marks == 3
  assert len(calls) == 1

  # ...and reported, with the true totals, on the next drain past the interval.
  clock.advance(mr.DROP_LOG_INTERVAL_S)
  rec._writer_housekeeping()
  assert len(calls) == 2
  assert calls[1][1] == 1
  assert calls[1][2] == 3


# ------------------------------------------------------------------- writing


def test_header_carries_route_segment_and_monotime(tmp_path):
  """Regression test for the five prior sidecars, none of which recorded
  route / segment / logMonoTime and were therefore un-joinable to rlog."""
  realdata = tmp_path / "realdata"
  route = "000001a3--c20ba54385"
  for seg in (5, 6, 7):
    (realdata / f"{route}--{seg}").mkdir(parents=True)
  (realdata / "unrelated--dir--x").mkdir(parents=True)

  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  press_ns = _run(rec, sm, planner, clock, ticks=1000, press_at=500, start_ns=start)
  mark, rows = rec._queue.get_nowait()
  rec._write_mark(mark, rows)

  path = next((tmp_path / "LongMarks").glob("longmark__*.jsonl"))
  assert path.name == f"longmark__{route}--7__{press_ns}.jsonl"
  lines = path.read_text().splitlines()
  header = json.loads(lines[0])
  assert header["type"] == "header"
  assert header["route"] == route
  assert header["segment"] == 7
  assert header["pressLogMonoTime"] == press_ns
  assert header["schemaVersion"] == mr.SCHEMA_VERSION
  assert header["columns"] == list(mr.COLUMNS)
  assert header["carFingerprint"] == "KIA_EV6"
  assert header["gitBranch"] == "chauffeur-exp01"
  assert header["liveTune"]["gap_reclaim_max_accel"] == 0.8
  assert header["rowCount"] == len(rows) == len(lines) - 1
  assert header["rowCountRequested"] == len(rows)
  assert header["droppedRows"] == 0
  assert header["truncated"] is False
  assert header["truncatedAtByte"] is None

  first = json.loads(lines[1])
  assert len(first) == len(mr.COLUMNS)
  assert first[mr.COLUMNS.index("modelLogMonoTime")] == press_ns - int(mr.PRE_WINDOW_S * 1e9)
  assert first[mr.COLUMNS.index("mpcSource")] == "lead0"
  assert first[mr.COLUMNS.index("accSourceActiveMode")] == "lead"
  assert first[mr.COLUMNS.index("leadRoleLead0")] == "center_control"
  assert not list((tmp_path / "LongMarks").glob("*.tmp"))


def test_unreadable_route_degrades_filename(tmp_path):
  class _NoRoute(_FakeParams):
    def get(self, key: str):
      raise RuntimeError("params unavailable")

  rec, _, _ = _make(tmp_path, params=_NoRoute())
  rec._write_mark({"pressLogMonoTime": 4242, "carFingerprint": "", "liveTuneCfg": None}, [])
  path = next((tmp_path / "LongMarks").glob("longmark__*.jsonl"))
  assert path.name == "longmark__unknownroute--X__4242.jsonl"
  header = json.loads(path.read_text().splitlines()[0])
  assert header["pressLogMonoTime"] == 4242
  assert header["rowCount"] == 0
  assert header["liveTune"] is None


def test_write_is_atomic_and_prunes(tmp_path):
  rec, _, _ = _make(tmp_path)
  marks_dir = tmp_path / "LongMarks"
  base_row = mr.MarkRecorder._build_row(_fake_planner(), _to_ticked_sm(10_000_000_000))
  for i in range(55):
    rec._write_mark({"pressLogMonoTime": 1000 + i, "carFingerprint": "KIA_EV6", "liveTuneCfg": None},
                    [base_row])
    # Deterministic mtime ordering; APFS/ext4 resolution is finer than the loop.
    for j, p in enumerate(sorted(marks_dir.glob("longmark__*.jsonl"))):
      os.utime(p, (1_700_000_000 + j, 1_700_000_000 + j))
  survivors = sorted(marks_dir.glob("longmark__*.jsonl"))
  assert len(survivors) == mr.MAX_MARK_FILES
  assert not list(marks_dir.glob("*.tmp"))
  # Oldest (lowest press ns) are gone.
  press_values = sorted(int(p.stem.rsplit("__", 1)[1]) for p in survivors)
  assert press_values[0] == 1000 + 55 - mr.MAX_MARK_FILES


def test_oserror_backs_off(tmp_path, monkeypatch):
  rec, _, clock = _make(tmp_path)
  real_open = builtins.open
  opens: list[str] = []

  def _enospc(*a, **k):
    opens.append(str(a[0]))
    raise OSError(errno.ENOSPC, "No space left on device")

  monkeypatch.setattr(builtins, "open", _enospc)
  monkeypatch.setattr(mr.cloudlog, "exception", lambda *a, **k: None)
  rec._write_mark({"pressLogMonoTime": 7, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert len(opens) == 1
  assert rec._disk_backoff_until_s > clock.now()

  clock.advance(30.0)
  rec._write_mark({"pressLogMonoTime": 8, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert len(opens) == 1  # short-circuited: open() was not even called

  clock.advance(31.0)
  monkeypatch.setattr(builtins, "open", real_open)
  rec._write_mark({"pressLogMonoTime": 9, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert (tmp_path / "LongMarks" / "longmark__000001a3--c20ba54385--X__9.jsonl").exists()


def test_truncated_header_matches_the_bytes_actually_written(tmp_path, monkeypatch):
  """The header is line 1 and cannot be revised once written, so the truncation
  decision has to be made before it. Measured on the pre-fix code with the same
  2048-byte budget: header rowCount=200 / truncated=False for a file holding 5
  data rows, and mark_cli reports header["truncated"] verbatim."""
  monkeypatch.setattr(mr, "MAX_BYTES_PER_MARK", 2048)
  rec, _, _ = _make(tmp_path)
  row = mr.MarkRecorder._build_row(_fake_planner(), _to_ticked_sm(10_000_000_000))
  rec._write_mark({"pressLogMonoTime": 11, "carFingerprint": "", "liveTuneCfg": None}, [row] * 200)
  path = next((tmp_path / "LongMarks").glob("longmark__*.jsonl"))
  lines = path.read_text().splitlines()
  parsed = [json.loads(line) for line in lines]  # still valid JSONL end to end
  header, marker = parsed[0], parsed[-1]
  data_rows = [record for record in parsed if isinstance(record, list)]

  assert 0 < len(data_rows) < 200
  assert header["truncated"] is True
  assert header["rowCount"] == len(data_rows)          # the count the file can back up
  assert header["rowCountRequested"] == 200
  assert header["droppedRows"] == 200 - len(data_rows)
  assert header["firstModelLogMonoTime"] == 10_000_000_000
  assert header["lastModelLogMonoTime"] == 10_000_000_000
  assert marker["type"] == "truncated"
  assert marker["rowCount"] == header["rowCount"]
  assert marker["rowCountRequested"] == 200
  assert marker["atByte"] == header["truncatedAtByte"] > mr.MAX_BYTES_PER_MARK
  # The budget really was enforced on the rows that were kept (+1 per line for
  # the newline splitlines() stripped).
  body_bytes = sum(len(line) + 1 for line in lines[1:1 + len(data_rows)])
  assert body_bytes <= mr.MAX_BYTES_PER_MARK


def test_a_single_oversized_row_reports_zero_rows_not_a_full_capture(tmp_path, monkeypatch):
  """The degenerate edge of the same bug: nothing fits, so the header must say so
  rather than inherit len(rows)."""
  monkeypatch.setattr(mr, "MAX_BYTES_PER_MARK", 8)
  rec, _, _ = _make(tmp_path)
  row = mr.MarkRecorder._build_row(_fake_planner(), _to_ticked_sm(10_000_000_000))
  rec._write_mark({"pressLogMonoTime": 12, "carFingerprint": "", "liveTuneCfg": None}, [row] * 3)
  path = next((tmp_path / "LongMarks").glob("longmark__*.jsonl"))
  header = json.loads(path.read_text().splitlines()[0])
  assert header["rowCount"] == 0
  assert header["rowCountRequested"] == 3
  assert header["droppedRows"] == 3
  assert header["truncated"] is True
  assert header["firstModelLogMonoTime"] is None
  assert header["lastModelLogMonoTime"] is None


def test_writer_loop_survives_a_failing_write(tmp_path, monkeypatch):
  rec, _, _ = _make(tmp_path)
  seen: list[str] = []
  monkeypatch.setattr(mr.cloudlog, "exception", lambda *a, **k: seen.append(str(a[0])))
  monkeypatch.setattr(rec, "_write_mark", _raiser)
  rec._queue.put_nowait(({"pressLogMonoTime": 1}, []))
  rec._queue.put_nowait(None)
  rec._writer_loop()          # returns on the sentinel, not on the failure
  assert len(seen) == 1


def _raiser(*_a, **_k):
  raise ValueError("boom")


def _to_ticked_sm(mono_ns: int) -> _FakeSM:
  sm = _FakeSM()
  sm.tick(mono_ns)
  return sm


def test_json_safe_coerces_non_primitives():
  assert mr._json_safe(None) is None
  assert mr._json_safe(True) is True
  assert mr._json_safe(3) == 3
  assert mr._json_safe(1.23456789) == 1.2346
  assert mr._json_safe(float("nan")) is None
  assert mr._json_safe(float("inf")) is None
  assert mr._json_safe(object()).startswith("<object")
  np = pytest.importorskip("numpy")
  assert mr._json_safe(np.bool_(True)) is True
  assert mr._json_safe(np.float32(0.5)) == 0.5


def test_capture_never_raises_and_self_disables(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm = _FakeSM()
  sm.tick(10_000_000_000)
  broken = _Obj()  # no planner attributes at all
  for i in range(mr.MAX_CONSECUTIVE_CAPTURE_ERRORS - 1):
    assert rec.update(broken, sm) is None
    assert rec._dead is False, i
  assert rec.update(broken, sm) is None
  assert rec._dead is True
  # Once dead it is a pure no-op forever.
  before = len(rec._ring)
  assert rec.update(_fake_planner(), sm) is None
  assert len(rec._ring) == before


def test_transient_capture_error_resets_counter(tmp_path):
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  sm.tick(10_000_000_000)
  rec.update(_Obj(), sm)
  assert rec._capture_errors == 1
  rec.update(planner, sm)
  assert rec._capture_errors == 0
  assert rec._dead is False


def test_queue_module_is_bounded():
  assert mr.FLUSH_QUEUE_MAXSIZE == 4
  q: queue.Queue = queue.Queue(maxsize=mr.FLUSH_QUEUE_MAXSIZE)
  for _ in range(mr.FLUSH_QUEUE_MAXSIZE):
    q.put_nowait(1)
  with pytest.raises(queue.Full):
    q.put_nowait(1)


# ------------------------------------------------- writer thread lifecycle (RT)


def test_writer_loop_demotes_itself_before_anything_else(tmp_path, monkeypatch):
  """The writer inherits SCHED_FIFO prio 51 pinned to core 5 from plannerd
  (PTHREAD_INHERIT_SCHED + clone() copying cpus_allowed). It must get off both
  before it ever holds the GIL for a JSON dump.

  Deliberately does NOT stub _demote_this_thread: the previous version installed
  its own no-op over it, so it pinned call ORDER only and passed with the real
  demotion body gutted. Stubbing the SYSCALLS instead makes the assertion fail
  unless the real body actually runs.
  """
  order: list[str] = []
  monkeypatch.setattr(mr.sys, "platform", "linux")
  monkeypatch.setattr(mr, "PC", False)
  monkeypatch.setattr(mr.os, "SCHED_OTHER", 0, raising=False)
  monkeypatch.setattr(mr.os, "sched_param", lambda p: p, raising=False)
  monkeypatch.setattr(mr.os, "sched_setscheduler", lambda *a: order.append("sched_setscheduler"), raising=False)
  monkeypatch.setattr(mr.os, "sched_setaffinity", lambda *a: order.append("sched_setaffinity"), raising=False)

  rec, _, _ = _make(tmp_path)
  monkeypatch.setattr(rec, "_write_mark", lambda *a, **k: order.append("write"))
  rec._queue.put_nowait(({"pressLogMonoTime": 1}, []))
  rec._queue.put_nowait(None)
  rec._writer_loop()
  assert order == ["sched_setscheduler", "sched_setaffinity", "write"]


def test_demote_unpins_to_all_cores_and_survives_a_failing_syscall(monkeypatch):
  """Each syscall is guarded on its own: EPERM on sched_setscheduler must not
  stop the unpin, and neither may kill the thread."""
  affinity: list[list[int]] = []
  monkeypatch.setattr(mr.sys, "platform", "linux")
  monkeypatch.setattr(mr, "PC", False)
  monkeypatch.setattr(mr.cloudlog, "exception", lambda *a, **k: None)

  def _eperm(*_a, **_k):
    raise PermissionError(errno.EPERM, "Operation not permitted")

  monkeypatch.setattr(mr.os, "SCHED_OTHER", 0, raising=False)
  monkeypatch.setattr(mr.os, "sched_param", lambda p: p, raising=False)
  monkeypatch.setattr(mr.os, "sched_setscheduler", _eperm, raising=False)
  monkeypatch.setattr(mr.os, "sched_setaffinity", lambda _pid, cores: affinity.append(sorted(cores)), raising=False)

  mr.MarkRecorder._demote_this_thread()          # must not raise
  assert affinity == [sorted(range(os.cpu_count() or 1))]
  assert len(affinity[0]) > 1 or (os.cpu_count() or 1) == 1


def test_demote_is_a_noop_off_linux(monkeypatch):
  monkeypatch.setattr(mr.sys, "platform", "darwin")
  monkeypatch.setattr(mr.os, "sched_setscheduler", _raiser, raising=False)
  monkeypatch.setattr(mr.os, "sched_setaffinity", _raiser, raising=False)
  mr.MarkRecorder._demote_this_thread()


def test_writer_is_spawned_off_the_press_path(tmp_path):
  """threading.Thread.start() ends in an unbounded self._started.wait(). It must
  never run inside the 20 Hz planner loop's flush path, which is reached exactly
  when the driver presses the flag button."""
  starts: list[str] = []
  rec, _, clock = _make(tmp_path, thread_factory=_counting_thread_factory(starts))
  assert starts == ["long-mark-writer"]      # spawned at construction, off the loop
  assert rec._writer is not None

  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=1000, press_at=500)
  assert rec._queue.qsize() == 1
  assert starts == ["long-mark-writer"]      # no second start() anywhere near the press


def test_disabled_param_still_spawns_the_writer(tmp_path):
  """The writer is the only thread allowed to touch Params, so it must exist even
  when the feature is off — otherwise the only thing that can ever observe a live
  enable is the realtime thread, which is exactly the Thread.start()-on-RT bug."""
  starts: list[str] = []
  rec, _, _ = _make(tmp_path, params=_FakeParams(enabled=False),
                    thread_factory=_counting_thread_factory(starts))
  assert starts == [mr.WRITER_THREAD_NAME]
  assert rec._writer is not None
  assert rec._enabled is False


def test_failed_thread_start_goes_loudly_inert_and_is_never_retried(tmp_path, monkeypatch):
  """RuntimeError('can't start new thread') under the tici's process load is
  retried WRITER_SPAWN_ATTEMPTS times, ALL inside __init__.

  There is deliberately no retry from the flush path any more: Thread.start() ends
  in an unbounded self._started.wait() and must never run on the SCHED_FIFO planner
  thread. A recorder that ends up with no writer says so once and goes inert,
  rather than quietly filling a consumerless queue for the whole drive.
  """
  attempts: list[str] = []

  class _Flaky:
    def __init__(self, *, target=None, daemon=None, name=None, **_ignored):
      self._name = name

    def start(self) -> None:
      attempts.append(threading.current_thread().name)
      raise RuntimeError("can't start new thread")

  errors: list[tuple] = []
  monkeypatch.setattr(mr.cloudlog, "error", lambda *a, **k: errors.append(a))

  rec, _, clock = _make(tmp_path, thread_factory=_Flaky)
  assert len(attempts) == mr.WRITER_SPAWN_ATTEMPTS
  assert set(attempts) == {threading.current_thread().name}   # all off the realtime thread
  assert rec._writer is None                 # not wedged on a thread that never ran
  assert rec._dead is True
  assert len(errors) == 1
  assert "could not be started" in errors[0][0]

  sm, planner = _FakeSM(), _fake_planner()
  _on_rt_thread(lambda: _run(rec, sm, planner, clock, ticks=1000, press_at=500))
  assert len(attempts) == mr.WRITER_SPAWN_ATTEMPTS            # no retry anywhere near the press
  assert rec._queue.empty()
  assert len(rec._ring) == 0
  assert len(errors) == 1


def test_construction_time_param_failure_still_spawns_the_writer(tmp_path):
  """The proven realtime-spawn path: _read_enabled() returns None on a transient
  Params failure, bool(None) is False, so gating the spawn on it left NO writer at
  construction — and the next successful refresh then called _ensure_writer() from
  inside update(), i.e. on the SCHED_FIFO planner thread."""
  starts: list[tuple[str, str]] = []

  class _Counting:
    def __init__(self, *, target=None, daemon=None, name=None, **_ignored):
      self._name = name

    def start(self) -> None:
      starts.append((self._name, threading.current_thread().name))

  class _Flaky(_FakeParams):
    fail = True

    def get_bool(self, key: str) -> bool:
      if self.fail:
        raise OSError(errno.EINTR, "Interrupted system call")
      return super().get_bool(key)

  params = _Flaky()
  clock = _FakeClock()
  rec = mr.MarkRecorder(params=params, marks_dir=tmp_path / "LongMarks",
                        realdata_dir=tmp_path / "realdata", time_fn=clock.now,
                        thread_factory=_Counting)
  assert rec._enabled is False                                 # fail closed, as before
  assert rec._writer is not None                               # but the writer exists anyway
  assert starts == [(mr.WRITER_THREAD_NAME, threading.current_thread().name)]

  # The later successful read enables capture with no further Thread.start().
  params.fail = False
  clock.advance(mr.REFRESH_INTERVAL_S)
  rec._writer_housekeeping()
  sm, planner = _FakeSM(), _fake_planner()
  _on_rt_thread(lambda: _run(rec, sm, planner, clock, ticks=100))
  assert rec._enabled is True
  assert len(rec._ring) == 100
  assert len(starts) == 1


def test_no_thread_start_ever_runs_on_the_realtime_thread(tmp_path):
  """Drives every realtime entry point, across a live enable transition and a
  press flush, on a thread standing in for the planner loop."""
  starts: list[tuple[str, str]] = []

  class _Counting:
    def __init__(self, *, target=None, daemon=None, name=None, **_ignored):
      self._name = name

    def start(self) -> None:
      starts.append((self._name, threading.current_thread().name))

  clock = _FakeClock()
  rec = mr.MarkRecorder(params=_FakeParams(enabled=False), marks_dir=tmp_path / "LongMarks",
                        realdata_dir=tmp_path / "realdata", time_fn=clock.now,
                        thread_factory=_Counting)
  assert len(starts) == 1
  rec._enabled_requested = True              # exactly what the writer's poll publishes

  sm, planner = _FakeSM(), _fake_planner()

  def _drive():
    start = 10_000_000_000
    for i in range(1000):
      mono_ns = start + i * TICK_NS
      sm.tick(mono_ns)
      if i in (100, 300, 500):
        sm.press(mono_ns)
        rec.latch_press(sm)
      rec.update(planner, sm)
      clock.advance(TICK_S)

  _on_rt_thread(_drive)
  assert rec._enabled is True
  assert rec._queue.qsize() == 3             # the presses really did flush
  assert [thread for _name, thread in starts if thread == RT_THREAD_NAME] == []


def test_repeated_construction_and_close_leaves_no_writer_threads(tmp_path):
  """Measured before the fix: 25 constructions left 25 live 'long-mark-writer'
  daemon threads, still alive after del + gc.collect(), each holding the recorder
  and its 780-slot ring alive through the bound self._writer_loop target."""
  before = _live_writer_threads()
  recs = [mr.MarkRecorder(params=_FakeParams(), marks_dir=tmp_path / "LongMarks",
                          realdata_dir=tmp_path / "realdata") for _ in range(25)]
  assert _live_writer_threads() - before == 25
  for rec in recs:
    rec.close()
  del recs
  gc.collect()
  assert _wait_for_writer_threads(before) == before


def test_a_dropped_recorder_lets_its_writer_thread_and_ring_go(tmp_path):
  """No close() at all: the thread holds only a weakref, so dropping the recorder
  has to be enough. The bound self._writer_loop target used to pin both forever."""
  before = _live_writer_threads()
  rec = mr.MarkRecorder(params=_FakeParams(), marks_dir=tmp_path / "LongMarks",
                        realdata_dir=tmp_path / "realdata")
  ref = weakref.ref(rec)
  assert _live_writer_threads() == before + 1
  del rec
  for _ in range(200):
    gc.collect()
    if ref() is None:
      break
    time.sleep(0.01)
  assert ref() is None                       # the ~1 MiB ring is collectable
  assert _wait_for_writer_threads(before) == before


def test_close_is_idempotent_and_the_context_manager_closes(tmp_path):
  before = _live_writer_threads()
  with mr.MarkRecorder(params=_FakeParams(), marks_dir=tmp_path / "LongMarks",
                       realdata_dir=tmp_path / "realdata") as rec:
    assert _live_writer_threads() == before + 1
  assert _wait_for_writer_threads(before) == before
  rec.close()
  rec.close()
  assert _live_writer_threads() == before


# ------------------------------------------------------- disk backoff / orphans


def test_disk_backoff_drops_are_counted_and_logged(tmp_path, monkeypatch):
  """_service_pending has already dropped the mark from _pending by the time
  _write_mark runs, so a silent early return loses a flagged incident forever."""
  errors: list[tuple] = []
  rec, _, clock = _make(tmp_path)
  monkeypatch.setattr(mr.cloudlog, "exception", lambda *a, **k: None)
  monkeypatch.setattr(mr.cloudlog, "error", lambda *a, **k: errors.append(a))

  def _enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")

  monkeypatch.setattr(builtins, "open", _enospc)
  rec._write_mark({"pressLogMonoTime": 7, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert rec._disk_backoff_until_s > clock.now()
  assert rec._disk_dropped_marks == 0        # this one failed, it was not backoff-dropped
  assert errors == []

  clock.advance(1.0)
  rec._write_mark({"pressLogMonoTime": 8, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert rec._disk_dropped_marks == 1
  assert len(errors) == 1
  assert "disk backoff" in errors[0][0]

  clock.advance(1.0)                         # rate limited, still counted
  rec._write_mark({"pressLogMonoTime": 9, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert rec._disk_dropped_marks == 2
  assert len(errors) == 1

  clock.advance(mr.DROP_LOG_INTERVAL_S + 1.0)
  rec._write_mark({"pressLogMonoTime": 10, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert rec._disk_dropped_marks == 3
  assert len(errors) == 2
  assert errors[1][2] == 3


def test_failed_write_leaves_no_orphan_tmp(tmp_path, monkeypatch):
  """The realistic ENOSPC shape: open() succeeds and the failure surfaces at
  write(), leaving a partial .jsonl.tmp that no glob would ever clean up."""
  rec, _, _ = _make(tmp_path)
  monkeypatch.setattr(mr.cloudlog, "exception", lambda *a, **k: None)
  real_open = builtins.open

  class _NoSpaceFile:
    def __init__(self, f):
      self._f = f

    def write(self, _s):
      raise OSError(errno.ENOSPC, "No space left on device")

    def __enter__(self):
      return self

    def __exit__(self, *_a):
      self._f.close()
      return False

  monkeypatch.setattr(builtins, "open", lambda *a, **k: _NoSpaceFile(real_open(*a, **k)))
  rec._write_mark({"pressLogMonoTime": 5, "carFingerprint": "", "liveTuneCfg": None}, [])

  marks_dir = tmp_path / "LongMarks"
  assert marks_dir.is_dir()
  assert list(marks_dir.iterdir()) == []     # no orphan .jsonl.tmp, no partial .jsonl


def test_prune_sweeps_and_accounts_for_orphan_tmp(tmp_path):
  rec, _, _ = _make(tmp_path)
  marks_dir = tmp_path / "LongMarks"
  marks_dir.mkdir(parents=True)
  orphan = marks_dir / "longmark__oldroute--1__1.jsonl.tmp"
  orphan.write_text("x" * 4096)

  rec._write_mark({"pressLogMonoTime": 2, "carFingerprint": "", "liveTuneCfg": None}, [])
  assert not orphan.exists()
  assert not list(marks_dir.glob("*.tmp"))
  assert len(list(marks_dir.glob("longmark__*.jsonl"))) == 1


# ------------------------------------------------------------ press latching


def test_press_survives_a_missed_modelv2_poll_window(tmp_path):
  """bookmarkButton is a non-polled service under poll='modelV2', so SubMaster
  drains its socket every iteration and clears sm.updated on the next one. A
  press landing on a cycle where modelV2 was late must still be captured."""
  start = 10_000_000_000
  press_ns = start + 500 * TICK_NS

  def _drive(rec, clock, *, latch: bool):
    sm, planner = _FakeSM(), _fake_planner()
    for i in range(500):
      sm.tick(start + i * TICK_NS)
      if latch:
        rec.latch_press(sm)
      rec.update(planner, sm)
      clock.advance(TICK_S)
    # plannerd iteration where modelV2 did NOT update: only latch_press runs.
    sm.press(press_ns)
    if latch:
      rec.latch_press(sm)
    # Next iteration: SubMaster.update_msgs() has cleared sm.updated.
    sm.tick(start + 501 * TICK_NS)
    if latch:
      rec.latch_press(sm)
    rec.update(planner, sm)

  rec, _, clock = _make(tmp_path)
  _drive(rec, clock, latch=True)
  assert len(rec._pending) == 1
  assert rec._pending[0]["pressLogMonoTime"] == press_ns

  # Proof the latch is load-bearing: without it the press is silently discarded.
  rec2, _, clock2 = _make(tmp_path)
  _drive(rec2, clock2, latch=False)
  assert rec2._pending == []


def test_latch_press_is_inert_when_disabled_and_never_raises(tmp_path):
  rec, _, _ = _make(tmp_path, params=_FakeParams(enabled=False))
  sm = _FakeSM()
  sm.press(1234)
  rec.latch_press(sm)
  assert len(rec._latched_presses) == 0
  # A malformed sm must not raise into the plannerd loop.
  assert rec.latch_press(_Obj()) is None


def test_live_disable_clears_a_latched_press(tmp_path):
  rec, params, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=10)
  sm.press(10_000_000_000 + 10 * TICK_NS)
  rec.latch_press(sm)
  assert len(rec._latched_presses) == 1
  params.enabled = False
  clock.advance(mr.REFRESH_INTERVAL_S)
  rec._writer_housekeeping()
  sm.tick(10_000_000_000 + 11 * TICK_NS)
  rec.update(planner, sm)
  assert len(rec._latched_presses) == 0
  assert rec._pending == []


def test_two_presses_during_a_modelv2_stall_are_both_armed(tmp_path):
  """Proven loss with the old single-slot latch: two presses 1.5 s apart (past
  DEBOUNCE_S, i.e. two separate incidents) while modelV2 was stalled — the second
  overwrote the first, dropped stayed 0 and nothing was logged."""
  rec, _, clock = _make(tmp_path)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  _run(rec, sm, planner, clock, ticks=100, start_ns=start)

  first = start + 100 * TICK_NS
  second = first + int(1.5e9)
  sm.press(first)
  rec.latch_press(sm)                        # modelV2 stalled: update() does not run at all
  clock.advance(1.5)
  sm.press(second)
  rec.latch_press(sm)
  clock.advance(TICK_S)
  assert list(rec._latched_presses) == [first, second]

  sm.tick(second + TICK_NS)                  # modelV2 comes back
  rec.update(planner, sm)
  assert [m["pressLogMonoTime"] for m in rec._pending] == [first, second]
  assert rec._dropped_marks == 0


def test_nonconflated_press_socket_retains_two_presses_during_a_model_stall(tmp_path, monkeypatch):
  """The production socket must be non-conflated and drained directly.

  SubMaster always constructs conflate=True sockets. Its old bookmarkButton
  entry could therefore expose only the newest press after a >1 s model stall,
  no matter how many slots the recorder's deque had.
  """
  rec, _, _ = _make(tmp_path)
  first, second = 10_000_000_000, 11_500_000_000
  messages = iter([_Obj(logMonoTime=first), _Obj(logMonoTime=second), None])
  monkeypatch.setattr(mr.messaging, "recv_one_or_none", lambda _sock: next(messages))

  rec.drain_press_socket(_Obj())
  assert list(rec._latched_presses) == [first, second]

  plannerd_source = (Path(__file__).parents[5] / "selfdrive/controls/plannerd.py").read_text(encoding="utf-8")
  assert 'sub_sock("bookmarkButton", conflate=False)' in plannerd_source
  assert "mark_recorder.drain_press_socket(bookmark_sock)" in plannerd_source


def test_latch_overflow_is_counted_and_reported(tmp_path, monkeypatch):
  """Beyond MAX_LATCHED_PRESSES the newest press is refused (matching _arm's
  MAX_PENDING_MARKS rule) — counted on the realtime thread, logged by the writer."""
  errors: list[tuple] = []
  monkeypatch.setattr(mr.cloudlog, "error", lambda *a, **k: errors.append(a))

  rec, _, clock = _make(tmp_path)
  sm = _FakeSM()
  base, step = 10_000_000_000, int(1.5e9)
  for i in range(mr.MAX_LATCHED_PRESSES + 3):
    sm.press(base + i * step)
    rec.latch_press(sm)

  assert list(rec._latched_presses) == [base + i * step for i in range(mr.MAX_LATCHED_PRESSES)]
  assert rec._dropped_latch_overflow == 3
  assert rec._dropped_marks == 3
  assert errors == []                        # nothing logged from the realtime thread

  clock.advance(mr.DROP_LOG_INTERVAL_S + 1.0)
  rec._writer_housekeeping()
  assert len(errors) == 1
  assert "presses latched" in errors[0][0]
  assert errors[0][2] == 3                   # new overflows
  assert errors[0][3] == 3                   # running total dropped


# ------------------------------------------------------------- param failures


def test_transient_param_read_failure_keeps_the_ring_and_pending(tmp_path, monkeypatch):
  """One EINTR'd Params read must not be mistaken for a live disable; that would
  destroy 20 s of pre-window plus every already-armed mark."""
  errors: list[tuple] = []
  monkeypatch.setattr(mr.cloudlog, "error", lambda *a, **k: errors.append(a))

  class _Flaky(_FakeParams):
    fail = False

    def get_bool(self, key: str) -> bool:
      if self.fail:
        raise OSError(errno.EINTR, "Interrupted system call")
      return super().get_bool(key)

  params = _Flaky()
  rec, _, clock = _make(tmp_path, params=params)
  sm, planner = _FakeSM(), _fake_planner()
  start = 10_000_000_000
  _run(rec, sm, planner, clock, ticks=100, press_at=90, start_ns=start)
  assert len(rec._ring) == 100
  assert len(rec._pending) == 1

  params.fail = True
  rec._writer_housekeeping()                 # the failing read, on the writer thread
  assert rec._param_read_errors == 1
  # Absorbed, but never silent: the writer reports it where it happened.
  assert len(errors) == 1
  assert "param read failed" in errors[0][0]

  _run(rec, sm, planner, clock, ticks=100, start_ns=start + 100 * TICK_NS)
  assert rec._enabled is True
  assert rec._enabled_requested is True
  assert len(rec._ring) == 200               # pre-window intact
  assert len(rec._pending) == 1              # armed mark intact
  assert rec._dead is False
  assert len(errors) == 1                    # and nothing was logged from the realtime loop


def test_unknown_param_key_logs_once_and_stays_inert(tmp_path, monkeypatch):
  """Deploying the Python half without rebuilding common/params_pyx leaves the
  recorder permanently inert; that must not be silent."""
  errors: list[tuple] = []
  monkeypatch.setattr(mr.cloudlog, "error", lambda *a, **k: errors.append(a))

  class _NoKey(_FakeParams):
    def get_bool(self, key: str) -> bool:
      raise mr.UnknownKeyName(key)

  starts: list[str] = []
  rec, _, clock = _make(tmp_path, params=_NoKey(), thread_factory=_counting_thread_factory(starts))
  sm, planner = _FakeSM(), _fake_planner()
  _run(rec, sm, planner, clock, ticks=400, press_at=100)

  assert rec._enabled is False               # fail closed
  assert len(rec._ring) == 0
  assert starts == [mr.WRITER_THREAD_NAME]   # the writer still exists; only capture is inert
  assert len(errors) == 1                    # emitted from __init__, off the realtime thread
  assert errors[0][1] == mr.ENABLE_PARAM
  assert "scons" in errors[0][0]

  # Every later writer poll re-observes the same missing key and must not re-log.
  for _ in range(20):
    rec._writer_housekeeping()
    clock.advance(mr.REFRESH_INTERVAL_S)
  assert rec._unknown_key_errors > 1
  assert len(errors) == 1


# ------------------------------------------------- binding to the REAL planner


@pytest.fixture(scope="module")
def real_planner():
  """A real LongitudinalPlanner for the 2023 Kia EV6.

  Every other test in this file drives a hand-built fake whose attribute names
  are typed out here, so a planner attribute rename would leave _build_row
  raising on the realtime thread — and update() self-disables permanently after
  MAX_CONSECUTIVE_CAPTURE_ERRORS — while all of those tests still pass. No
  acados solve is run; only construction.
  """
  CP = interfaces[CAR.KIA_EV6].get_non_essential_params(CAR.KIA_EV6)
  planner = LongitudinalPlanner(CP)
  yield planner
  # The recorder spawns its writer unconditionally now, so a module-scoped
  # fixture that never closed it would strand a live daemon thread for the run.
  planner.mark_recorder.close()


def test_build_row_binds_to_a_real_planner(real_planner):
  sm = _to_ticked_sm(10_000_000_000)
  row = mr.MarkRecorder._build_row(real_planner, sm)
  assert len(row) == len(mr.COLUMNS)
  assert row[mr.COLUMNS.index("modelLogMonoTime")] == 10_000_000_000


def test_build_row_planner_attribute_names_all_exist(real_planner):
  """Names _build_row reaches for with BARE attribute access have no default;
  a rename is an AttributeError on the SCHED_FIFO thread, not a null column."""
  src = textwrap.dedent(inspect.getsource(mr.MarkRecorder._build_row))
  tree = ast.parse(src)
  wanted = sorted({n.attr for n in ast.walk(tree)
                   if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "planner"})
  assert len(wanted) >= 12, wanted     # the AST walk itself must not silently find nothing
  missing = [a for a in wanted if not hasattr(real_planner, a)]
  assert missing == [], f"_build_row reads planner attributes that do not exist: {missing}"


def test_real_planner_exposes_the_mark_recorder(real_planner):
  """plannerd reaches mark_recorder through LongitudinalPlanner, which inherits
  it from LongitudinalPlannerSP."""
  assert isinstance(real_planner.mark_recorder, mr.MarkRecorder)
  assert callable(real_planner.mark_recorder.latch_press)
  assert callable(real_planner.mark_recorder.update)
