"""Driver-mark planner-internal state recorder (Phase 2).

Captures the planner/MPC debug state that never crosses a capnp boundary into a
bounded in-process ring buffer, and flushes a -20 s / +6 s window to a JSONL
sidecar when the driver taps the on-screen flag button (``bookmarkButton``).

Realtime contract (plannerd runs SCHED_FIFO prio 51, pinned to core 5, with
``gc.disable()``, via ``common/realtime.py``). On the realtime thread this module:
  * allocates exactly one tuple per tick on the capture path (plus, only on an
    unexpected failure, one short ``str`` naming the exception); each rare press
    also decodes one already-queued capnp envelope from a dedicated non-conflated
    socket,
  * opens no file, reads no ``Params``, and calls no logging function. Every
    ``Params`` round-trip, every ``os.listdir``, every ``json.dumps`` and every
    ``cloudlog`` call in this module runs on the writer thread. ``cloudlog`` is a
    stdlib ``Logger``: emitting takes the ``StreamHandler`` and
    ``UnixDomainSocketHandler`` ``RLock``s and writes to stderr, which is exactly
    the unbounded block a SCHED_FIFO thread must not take against a SCHED_OTHER
    lock holder. So the realtime thread only bumps plain integer counters and the
    writer reports them within ``DROP_LOG_INTERVAL_S`` — nothing is lost, it is
    just announced late,
  * takes exactly one lock, ``queue.put_nowait``'s own mutex. The writer holds
    that mutex only for a ``deque`` pop (``Queue.get`` releases it while waiting),
  * never calls ``Thread.start()``, whose tail is an unbounded
    ``self._started.wait()``. The writer is spawned once, in ``__init__``, which
    runs during plannerd startup rather than inside the 20 Hz loop, and there is
    deliberately no respawn path. A writer that cannot be started at all makes the
    recorder go inert and says so loudly, once, from ``__init__``,
  * is wrapped in try/except at every entry point and can never raise into the
    planner loop.

The writer thread:
  * demotes itself to SCHED_OTHER and widens its CPU affinity mask back to every
    core as its very first act (CPython's ``pthread_create`` uses
    ``PTHREAD_INHERIT_SCHED`` and ``clone()`` copies ``cpus_allowed``, so a thread
    spawned from plannerd would otherwise run at prio 51 pinned to core 5 and
    steal ticks from the 20 Hz planner loop while it serializes JSON),
  * holds only a ``weakref`` to the recorder, so a recorder that is dropped
    without ``close()`` still lets its thread and its ~1 MiB ring go away,
  * exits on the ``None`` sentinel, on ``close()``, or when that weakref dies.

Call ``close()`` (or use the recorder as a context manager) to stop the writer
deterministically. ``close()`` joins and therefore must never be called from the
planner loop; plannerd never does.

Join contract: ``COLUMNS[0] == "modelLogMonoTime"`` is ``sm.logMonoTime['modelV2']``,
byte-identical to ``longitudinalPlan.modelMonoTime`` and to
``EpisodeFrame.model_v2_log_mono_time_ns`` in the offline harness. The header
carries route, segment and the ``bookmarkButton`` envelope ``logMonoTime``, so a
sidecar is always joinable back to rlog by exact integer equality.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import queue
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any

from cereal import messaging
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import PC

ENABLE_PARAM = "LongitudinalMarkRecorderEnabled"
MARKS_DIR_DEFAULT = Path("/data/media/0/LongMarks")
REALDATA_DIR = Path("/data/media/0/realdata")

SCHEMA_VERSION = 1
EXPECTED_HZ = 20.0            # plannerd ticks on modelV2; DT_MDL = 0.05 (common/realtime.py)
PRE_WINDOW_S = 20.0           # binding user decision
POST_WINDOW_S = 6.0           # binding user decision
DEBOUNCE_S = 1.0              # binding user decision: presses > 1 s apart are separate incidents
REFRESH_INTERVAL_S = 2.0      # Params re-read cadence (NEVER per tick)

# 26 s of window (20 pre + 6 post) at 20 Hz = 520 rows; 1.5x headroom for modelV2
# cadence overshoot and for a second mark armed while the first is still pending.
RING_MAXLEN = max(600, int(math.ceil((PRE_WINDOW_S + POST_WINDOW_S) * EXPECTED_HZ * 1.5)))

MAX_PENDING_MARKS = 4         # presses armed but whose +6 s has not elapsed
MAX_LATCHED_PRESSES = 4       # presses seen by latch_press() but not yet armed by update()
MAX_PRESS_DRAIN_PER_ITERATION = 8  # bounds work while leaving excess queued for the next iteration
FLUSH_QUEUE_MAXSIZE = 4       # bounded handoff to the writer thread
MAX_BYTES_PER_MARK = 4 * 1024 * 1024
MAX_MARK_FILES = 50           # retention: newest N sidecars survive
MAX_MARK_DIR_BYTES = 256 * 1024 * 1024
DISK_BACKOFF_S = 60.0         # after an OSError, stop trying to write for this long
MAX_CONSECUTIVE_CAPTURE_ERRORS = 20
DROP_LOG_INTERVAL_S = 10.0
WRITER_THREAD_NAME = "long-mark-writer"
WRITER_POLL_INTERVAL_S = 0.5  # writer wakes this often to poll Params and drain deferred logs
WRITER_SPAWN_ATTEMPTS = 3     # all at construction; there is no realtime-thread retry
CLOSE_JOIN_TIMEOUT_S = 2.0

# Row schema. Order is load-bearing: _build_row() emits a tuple in exactly this
# order and the header declares the names once. test_columns_arity_matches_row
# guards the single silent-corruption mode of a positional schema.
COLUMNS: tuple[str, ...] = (
  # --- frame identity / join keys -------------------------------------------
  "modelLogMonoTime",               # 0  sm.logMonoTime['modelV2'] == longitudinalPlan.modelMonoTime
  "radarStateLogMonoTime",          # 1  frame identity write_episode_bundle keys on
  "carStateLogMonoTime",            # 2
  "frame",                          # 3  sm.frame
  "tMonotonicS",                    # 4  time.monotonic() at capture
  # --- ego / plan output ----------------------------------------------------
  "vEgoMps",                        # 5
  "aEgoMps2",                       # 6
  "effectiveVCruiseMps",            # 7
  "outputATargetMps2",              # 8
  "outputShouldStop",               # 9
  "aDesiredMps2",                   # 10
  "vDesiredMps",                    # 11
  "fcw",                            # 12
  "longActive",                     # 13
  "gasPressed",                     # 14
  "accelClipLoMps2",                # 15
  "accelClipHiMps2",                # 16
  "mpcSource",                      # 17
  "mpcMode",                        # 18
  # --- CD5 relatch blend (longitudinal_planner.py relatch_blend_debug) -------
  "relatchActive",                  # 19
  "relatchFramesLeft",              # 20
  "relatchBypassed",                # 21
  "relatchBypassReason",            # 22
  "relatchNegCapMps2",              # 23
  "relatchClipped",                 # 24
  # --- CD6 handoff limiter (handoff_limit_debug) ----------------------------
  "handoffActive",                  # 25
  "handoffFramesLeft",              # 26
  "handoffDownBypassed",            # 27
  "handoffBypassReason",            # 28
  "handoffOpeningCapAppeared",      # 29
  "handoffEdge1Capped",             # 30
  "handoffClipped",                 # 31
  # --- CD7 comfort jerk envelope (comfort_jerk_debug) -----------------------
  "comfortActive",                  # 32
  "comfortBypassed",                # 33
  "comfortBypassReason",            # 34
  "comfortGatedReason",             # 35
  "comfortMaxStepMps2",             # 36
  "comfortClipped",                 # 37
  "comfortUpwardFloorOwner",        # 38  engaged branch only -> .get() default None
  "comfortUpwardMaxDeltaMps2",      # 39  engaged branch only
  "comfortUpwardStepMps2",          # 40  engaged branch only
  "comfortReleaseSlewActive",       # 41  added by the trailing .update()
  "comfortReleaseSlewClipped",      # 42
  # --- lead brake release floor + nested closing-recovery bridge -------------
  "brakeReleaseActive",             # 43
  "brakeReleaseReason",             # 44
  "brakeReleaseOutputBound",        # 45  assigned in place after the clip
  "brakeReleaseFloorMps2",          # 46
  "bridgeReason",                   # 47  nested closing_recovery_bridge
  "bridgeApplied",                  # 48
  "bridgeDesiredOutputMps2",        # 49
  "bridgeOutputUpliftMps2",         # 50
  # --- steady-parity safety cap (steady_parity_threat_debug) ----------------
  "steadyParityActive",             # 51
  "steadyParityPreComfortMps2",     # 52
  "steadyParityFinalMps2",          # 53
  "steadyParitySafetyCapApplied",   # 54
  # --- CD5(b) cruise reacquire ramp (cruise_reacquire_debug) ----------------
  "cruiseReacquireActive",          # 55
  "cruiseReacquireFramesLeft",      # 56
  "cruiseReacquireAllowedJerkMps3",  # 57
  "cruiseReacquireSlewCeilingMps2",  # 58
  "cruiseReacquireClipped",         # 59
  "cruiseReacquireExitCause",       # 60
  # --- MPC lead-role classifier + acc source --------------------------------
  "leadRoleLead0",                  # 61  lead_role_debug["roles"]["lead0"]
  "leadRoleGateActive",             # 62
  "leadRoleDuplicatePair",          # 63
  "leadRoleDroppedSlot",            # 64
  "leadRoleCutinPromoted",          # 65  true when EITHER slot was cut-in promoted
  "accSourceActiveMode",            # 66  acc_source_debug["active_mode"]
  "accSourceReason",                # 67
  # --- stabilized control leads ---------------------------------------------
  "lead0Status",                    # 68
  "lead0DRelM",                     # 69
  "lead0VRelMps",                   # 70
  "lead0ModelProb",                 # 71
  "lead1Status",                    # 72
  "lead1DRelM",                     # 73
  "lead1VRelMps",                   # 74
  # --- post-MPC slowdown arbitration ---------------------------------------
  "slowdownArbitrationReason",       # 75
  "slowdownArbitrationUrgent",       # 76
  "slowdownRawCeilingMps2",          # 77
  "slowdownEffectiveCeilingMps2",    # 78
  "slowdownMpcAccelMps2",            # 79
  "slowdownModelAccelMps2",          # 80
)

_EMPTY: dict[str, Any] = {}


class MarkRecorder:
  """Ring buffer + off-thread JSONL flush, owned by ``LongitudinalPlannerSP``."""

  def __init__(
    self,
    *,
    params: Params | None = None,
    marks_dir: str | Path = MARKS_DIR_DEFAULT,
    realdata_dir: str | Path = REALDATA_DIR,
    time_fn=time.monotonic,
    thread_factory=threading.Thread,
  ):
    self._params = params if params is not None else Params()
    self._marks_dir = Path(marks_dir)
    self._realdata_dir = Path(realdata_dir)
    self._time_fn = time_fn
    self._thread_factory = thread_factory

    self._ring: collections.deque[tuple] = collections.deque(maxlen=RING_MAXLEN)
    self._pending: list[dict[str, Any]] = []
    self._latched_presses: collections.deque[int] = collections.deque(maxlen=MAX_LATCHED_PRESSES)
    self._queue: queue.Queue = queue.Queue(maxsize=FLUSH_QUEUE_MAXSIZE)
    self._writer: Any = None
    self._finalizer: Any = None
    self._stop_requested = False

    # Realtime-thread-owned. Every one is a plain int the realtime thread only
    # ever increments and the writer thread only ever reads, so reporting them
    # needs no lock in either direction (see _drain_deferred_logs).
    self._last_press_mono_ns = 0
    self._capture_errors = 0
    self._dead = False
    self._dropped_marks = 0            # total lost on the realtime side; goes in the header
    self._dropped_queue_full = 0
    self._dropped_pending_cap = 0
    self._dropped_latch_overflow = 0
    self._dropped_handoff_errors = 0
    self._press_socket_errors = 0
    self._last_handoff_error = ""
    self._last_press_socket_error = ""
    self._death_reason = ""

    # Enable state. The Params round-trip itself happens on the writer thread
    # (_poll_enable) and publishes _enabled_requested; the realtime thread applies
    # the transition because the ring it has to clear is realtime-thread-owned.
    self._enabled = False
    self._enabled_requested = False

    # Writer-thread-owned.
    self._enable_poll_at_s = 0.0
    self._param_read_errors = 0
    self._last_param_error = ""
    self._unknown_key_errors = 0
    self._reported_unknown_key = False
    self._reported_death = False
    self._reported_queue_full = 0
    self._reported_pending_cap = 0
    self._reported_latch_overflow = 0
    self._reported_handoff_errors = 0
    self._reported_press_socket_errors = 0
    self._reported_param_errors = 0
    self._last_deferred_log_s = -DROP_LOG_INTERVAL_S
    self._disk_dropped_marks = 0
    self._disk_backoff_until_s = 0.0
    self._last_backoff_log_s = -DROP_LOG_INTERVAL_S

    # Construction happens during plannerd startup, off the 20 Hz loop, so the
    # first Params round-trip, Thread.start()'s unbounded _started.wait() and the
    # cloudlog for anything either of them reported all land here.
    now_s = float(self._time_fn())
    self._enabled = bool(self._read_enabled())
    self._enabled_requested = self._enabled
    self._enable_poll_at_s = now_s + REFRESH_INTERVAL_S
    # Spawned UNCONDITIONALLY, including when the param is off and including when
    # the param read itself failed. bool(None) is False, so gating the spawn on the
    # read left a failed read with no writer at all and pushed the spawn onto the
    # next successful refresh — which runs inside update(), i.e. on the realtime
    # thread. An idle writer costs one blocked thread and one wakeup every
    # WRITER_POLL_INTERVAL_S; that is the price of the guarantee.
    self._start_writer()
    self._drain_deferred_logs(now_s)

  # ---------------------------------------------------------------- lifecycle

  def _read_enabled(self) -> bool | None:
    """``True``/``False`` for an observed value, ``None`` when the read itself
    failed. The caller must not confuse a failed read with a live disable: an
    EINTR'd Params read would otherwise destroy 20 s of pre-window.

    Never logs and never runs on the realtime thread: ``__init__`` and
    ``_poll_enable`` (writer thread) are the only callers. Diagnostics are counted
    here and emitted by ``_drain_deferred_logs``.
    """
    try:
      return bool(self._params.get_bool(ENABLE_PARAM))
    except UnknownKeyName:
      # The compiled common/params_pyx bakes in the params_keys.h table at scons
      # time. Deploying the Python half without rebuilding leaves the recorder
      # permanently inert; _drain_deferred_logs says so exactly once.
      self._unknown_key_errors += 1
      return False
    except Exception as exc:
      self._param_read_errors += 1
      self._last_param_error = _safe_reason(exc)
      return None

  def _start_writer(self) -> None:
    """Spawn the daemon writer. Called ONLY from ``__init__``; never raises.

    There is deliberately no retry from ``update()``/``_service_pending``:
    ``Thread.start()`` ends in an unbounded ``self._started.wait()`` and must never
    execute on the SCHED_FIFO planner thread. All ``WRITER_SPAWN_ATTEMPTS`` happen
    here instead. ``self._writer`` is assigned only after ``start()`` returns, so a
    half-spawn cannot leave the recorder pointing at a thread that never ran.

    If every attempt fails there is no non-realtime thread left to retry from, so
    the recorder goes inert and says so — rather than quietly filling a
    consumerless queue and losing every press for the drive.
    """
    ref = weakref.ref(self)
    last_exc: BaseException | None = None
    for _attempt in range(WRITER_SPAWN_ATTEMPTS):
      try:
        thread = self._thread_factory(target=_writer_main, args=(ref, self._queue),
                                      daemon=True, name=WRITER_THREAD_NAME)
        thread.start()
      except Exception as exc:
        last_exc = exc
        continue
      self._writer = thread
      # Belt and braces for a recorder that is dropped without close(): wake the
      # writer immediately instead of waiting out one WRITER_POLL_INTERVAL_S. Holds
      # the queue, never the recorder, so it cannot keep the ring alive.
      self._finalizer = weakref.finalize(self, _request_stop, self._queue)
      return
    self._writer = None
    self._dead = True
    self._reported_death = True
    cloudlog.error("longitudinal mark recorder inert: writer thread could not be started in %d attempts (%s); " +
                   "no driver marks will be recorded this drive", WRITER_SPAWN_ATTEMPTS, _safe_reason(last_exc))

  def close(self, timeout: float = CLOSE_JOIN_TIMEOUT_S) -> None:
    """Stop the writer and join it. Idempotent, never raises.

    Teardown API only: it joins, so it must not be called from the planner loop.
    plannerd never calls it; tests, process_replay and anything that builds more
    than one planner must, or every construction strands a live daemon thread and
    the ~1 MiB ring it references.
    """
    self._stop_requested = True
    if self._finalizer is not None:
      try:
        self._finalizer.detach()
      except Exception:
        pass
      self._finalizer = None
    _request_stop(self._queue)
    thread = self._writer
    self._writer = None
    if thread is None:
      return
    try:
      thread.join(timeout)
    except Exception:
      pass

  def __enter__(self) -> MarkRecorder:
    return self

  def __exit__(self, *_exc) -> None:
    self.close()

  # ------------------------------------------------------------ hot path (RT)

  def drain_press_socket(self, sock) -> None:
    """Drain a bounded number of raw, non-conflated bookmark envelopes.

    plannerd owns a dedicated ``conflate=False`` socket. This is load-bearing:
    SubMaster sockets are always conflated and can expose only the newest press
    after a long model stall. Excess messages remain queued for the next planner
    iteration; malformed/failed receives are counted for deferred diagnostics.
    """
    if self._dead:
      return
    for _ in range(MAX_PRESS_DRAIN_PER_ITERATION):
      try:
        msg = messaging.recv_one_or_none(sock)
      except Exception as exc:
        self._press_socket_errors += 1
        self._last_press_socket_error = _safe_reason(exc)
        return
      if msg is None:
        return
      if not self._enabled:
        continue  # drain stale envelopes while capture is disabled
      try:
        self._latch_press_mono(int(msg.logMonoTime))
      except Exception as exc:
        self._press_socket_errors += 1
        self._last_press_socket_error = _safe_reason(exc)

  def _latch_press_mono(self, press_mono_ns: int) -> None:
    if len(self._latched_presses) >= MAX_LATCHED_PRESSES:
      self._dropped_marks += 1
      self._dropped_latch_overflow += 1
      return
    self._latched_presses.append(press_mono_ns)

  def latch_press(self, sm) -> None:
    """Compatibility/test entry point for a SubMaster-delivered press.

    Production plannerd uses ``drain_press_socket`` because SubMaster always
    conflates and cannot preserve two presses across a long model stall. This
    fallback remains load-bearing for replay/unit callers that drive ``update``
    with a synthetic SubMaster. Cheap and never raises.

    A stall long enough to hold two DEBOUNCE_S-separated presses is real, so the
    latch is a bounded ``deque``, not a single slot: a single slot let the second
    press overwrite the first with ``dropped == 0`` and no log at all. On genuine
    overflow the NEWEST press is refused — matching ``_arm``'s MAX_PENDING_MARKS
    rule — and counted so the writer reports it.
    """
    if self._dead or not self._enabled:
      return
    try:
      if sm.updated.get("bookmarkButton", False):
        self._latch_press_mono(int(sm.logMonoTime["bookmarkButton"]))
    except Exception:
      pass

  def update(self, planner, sm) -> None:
    """Single realtime entry point.

    Never raises, never blocks, never opens a file, never reads ``Params``, never
    calls ``cloudlog``, and takes no lock but the flush queue's own.
    """
    if self._dead:
      return
    try:
      now_s = float(self._time_fn())
      if self._enabled_requested != self._enabled:
        self._apply_enable_request()
      if not self._enabled:
        return
      self._ring.append(self._build_row(planner, sm))
      while self._latched_presses:
        self._arm(planner, self._latched_presses.popleft(), now_s)
      if sm.updated.get("bookmarkButton", False):
        # Redundant with latch_press() in plannerd (same press, same logMonoTime,
        # collapsed by _arm's debounce); load-bearing for any caller that only
        # drives update().
        self._arm(planner, int(sm.logMonoTime["bookmarkButton"]), now_s)
      if self._pending:
        self._service_pending(now_s)
      self._capture_errors = 0
    except Exception as exc:
      self._capture_errors += 1
      if self._capture_errors >= MAX_CONSECUTIVE_CAPTURE_ERRORS and not self._dead:
        # Written BEFORE _dead so a writer that observes the flag also sees the
        # reason. cloudlog.exception() here would take the logging handler locks on
        # the SCHED_FIFO thread; _drain_deferred_logs emits it instead.
        self._death_reason = _safe_reason(exc)
        self._dead = True

  def _apply_enable_request(self) -> None:
    """Realtime-thread side of a live enable toggle. The writer publishes the new
    value; the ring and the armed marks are realtime-thread-owned, so only this
    thread may clear them."""
    self._enabled = self._enabled_requested
    if not self._enabled:
      # Free the ~1 MiB immediately on a live disable.
      self._ring.clear()
      self._pending.clear()
      self._latched_presses.clear()

  def _arm(self, planner, press_mono_ns: int, now_s: float) -> None:
    if press_mono_ns <= 0:
      return
    if self._last_press_mono_ns != 0 and (press_mono_ns - self._last_press_mono_ns) < int(DEBOUNCE_S * 1e9):
      return  # debounce: same incident
    self._last_press_mono_ns = press_mono_ns
    if len(self._pending) >= MAX_PENDING_MARKS:
      self._dropped_marks += 1
      self._dropped_pending_cap += 1
      return
    mpc = getattr(planner, "mpc", None)
    self._pending.append({
      "pressLogMonoTime": press_mono_ns,
      "deadlineS": now_s + POST_WINDOW_S,
      # Frozen dataclass reference. as_dict() (~150 fields) is deliberately
      # deferred to the writer thread.
      "liveTuneCfg": getattr(mpc, "_live_tune_cfg", None),
      "carFingerprint": str(getattr(getattr(planner, "CP", None), "carFingerprint", "")),
    })

  def _service_pending(self, now_s: float) -> None:
    """Slice and hand off any mark whose +POST_WINDOW_S has elapsed.

    The slice is a list of *shared references* to the already-built row tuples;
    the tuples themselves are never copied. Runs once per mark, not per tick.

    ``rows`` is a filtered copy of a deque whose maxlen is already ``RING_MAXLEN``,
    so it needs no row-count clamp of its own; the byte budget is enforced on the
    writer thread, where the serialized size is actually known.
    """
    still: list[dict[str, Any]] = []
    for mark in self._pending:
      if now_s < mark["deadlineS"]:
        still.append(mark)
        continue
      press_ns = int(mark["pressLogMonoTime"])
      lo = press_ns - int(PRE_WINDOW_S * 1e9)
      hi = press_ns + int(POST_WINDOW_S * 1e9)
      rows = [r for r in self._ring if lo <= r[0] <= hi]
      # No _start_writer() retry hook here on purpose: Thread.start() must never
      # run on this thread. See _start_writer's docstring.
      try:
        self._queue.put_nowait((mark, rows))
      except queue.Full:
        self._dropped_marks += 1
        self._dropped_queue_full += 1
      except Exception as exc:
        self._dropped_marks += 1
        self._dropped_handoff_errors += 1
        self._last_handoff_error = _safe_reason(exc)
    self._pending = still

  @staticmethod
  def _build_row(planner, sm) -> tuple:
    """Flatten the frame-final planner/MPC debug state into one immutable tuple.

    Every extraction is ``dict.get()`` so a branch-dependent missing key becomes
    ``null`` in the sidecar instead of a KeyError on the realtime thread. The
    tuple is independent of whatever the planner mutates next cycle, so no
    defensive copy of any debug dict is needed.
    """
    mpc = planner.mpc
    relatch = planner.relatch_blend_debug or _EMPTY
    handoff = planner.handoff_limit_debug or _EMPTY
    comfort = planner.comfort_jerk_debug or _EMPTY
    release = planner.lead_brake_release_debug or _EMPTY
    bridge = release.get("closing_recovery_bridge") or _EMPTY
    parity = planner.steady_parity_threat_debug or _EMPTY
    reacq = getattr(planner, "cruise_reacquire_debug", None) or _EMPTY
    role = getattr(mpc, "lead_role_debug", None) or _EMPTY
    role_names = role.get("roles") or _EMPTY
    cutin = role.get("cutin_promoted") or _EMPTY
    acc_src = getattr(mpc, "acc_source_debug", None) or _EMPTY
    slowdown_arbitration = getattr(planner, "lead_slowdown_arbitration_debug", None) or _EMPTY
    leads = getattr(mpc, "control_leads", None) or (None, None)
    lead0 = leads[0] if len(leads) > 0 else None
    lead1 = leads[1] if len(leads) > 1 else None
    clip = planner.prev_accel_clip
    cs = sm["carState"]
    cc = sm["carControl"]
    return (
      sm.logMonoTime["modelV2"],
      sm.logMonoTime["radarState"],
      sm.logMonoTime["carState"],
      sm.frame,
      time.monotonic(),
      cs.vEgo,
      cs.aEgo,
      planner.effective_v_cruise_mps,
      planner.output_a_target,
      planner.output_should_stop,
      planner.a_desired,
      planner.v_desired_filter.x,
      planner.fcw,
      cc.longActive,
      cs.gasPressed,
      clip[0],
      clip[1],
      getattr(mpc, "source", ""),
      getattr(mpc, "mode", ""),
      relatch.get("active"),
      relatch.get("frames_left"),
      relatch.get("bypassed"),
      relatch.get("bypass_reason"),
      relatch.get("neg_cap_mps2"),
      relatch.get("clipped"),
      handoff.get("active"),
      handoff.get("frames_left"),
      handoff.get("down_bypassed"),
      handoff.get("bypass_reason"),
      handoff.get("opening_cap_appeared"),
      handoff.get("edge1_capped"),
      handoff.get("clipped"),
      comfort.get("active"),
      comfort.get("bypassed"),
      comfort.get("bypass_reason"),
      comfort.get("gated_reason"),
      comfort.get("max_step_mps2"),
      comfort.get("clipped"),
      comfort.get("upward_floor_owner"),
      comfort.get("upward_max_delta_mps2"),
      comfort.get("upward_step_mps2"),
      comfort.get("release_slew_active"),
      comfort.get("release_slew_clipped"),
      release.get("active"),
      release.get("reason"),
      release.get("output_bound"),
      planner.lead_brake_release_accel_floor,
      bridge.get("reason"),
      bridge.get("applied"),
      bridge.get("desired_output_mps2"),
      bridge.get("output_uplift_applied_mps2"),
      parity.get("active"),
      parity.get("pre_comfort_limiter_output_mps2"),
      parity.get("final_output_mps2"),
      parity.get("safety_cap_applied"),
      reacq.get("active"),
      reacq.get("frames_left"),
      reacq.get("allowed_jerk_mps3"),
      reacq.get("slew_ceiling_mps2"),
      reacq.get("clipped"),
      reacq.get("exit_cause"),
      role_names.get("lead0"),
      role.get("gate_active"),
      role.get("duplicate_pair"),
      role.get("dropped_slot"),
      bool(cutin.get("lead0", False) or cutin.get("lead1", False)),
      acc_src.get("active_mode"),
      acc_src.get("reason"),
      getattr(lead0, "status", None),
      getattr(lead0, "dRel", None),
      getattr(lead0, "vRel", None),
      getattr(lead0, "modelProb", None),
      getattr(lead1, "status", None),
      getattr(lead1, "dRel", None),
      getattr(lead1, "vRel", None),
      slowdown_arbitration.get("reason"),
      slowdown_arbitration.get("urgent"),
      slowdown_arbitration.get("raw_ceiling_mps2"),
      slowdown_arbitration.get("effective_ceiling_mps2"),
      slowdown_arbitration.get("mpc_accel_mps2"),
      slowdown_arbitration.get("model_accel_mps2"),
    )

  # ------------------------------------------------------ writer thread only

  @staticmethod
  def _demote_this_thread() -> None:
    """Undo the SCHED_FIFO priority and the core pinning inherited from plannerd.

    CPython's ``pthread_create`` passes default attributes, i.e.
    ``PTHREAD_INHERIT_SCHED``, and ``clone()`` copies ``cpus_allowed`` — so a
    thread spawned from a process that ran ``config_realtime_process(5,
    Priority.CTRL_LOW)`` starts at SCHED_FIFO prio 51 pinned to core 5, the
    planner's own core. This thread holds the GIL for milliseconds at a time
    while serializing JSON, so leaving it there would take those ticks directly
    out of the 20 Hz planner loop.

    The scheduler change is the one that matters: SCHED_OTHER never preempts
    SCHED_FIFO, so once demoted this thread cannot take a tick from the planner on
    any core. The affinity call does NOT move it away from core 5 — it widens the
    mask back to every core, core 5 included — it just stops the writer from being
    confined to the planner's core when other cores are idle.

    Guard style mirrors ``common/realtime.py``: Linux only, device only. Each
    call is separately guarded so one failure degrades instead of killing the
    writer.
    """
    if sys.platform != 'linux' or PC:
      return
    try:
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except Exception:
      cloudlog.exception("longitudinal mark writer could not drop to SCHED_OTHER")
    try:
      os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
    except Exception:
      cloudlog.exception("longitudinal mark writer could not widen its CPU affinity mask")

  def _writer_loop(self) -> None:
    """Direct (strong-ref) entry point, for tests and for anyone driving the loop
    synchronously. The spawned thread runs ``_writer_main`` with a weakref instead
    so that a dropped recorder does not strand a live thread."""
    _writer_main(weakref.ref(self), self._queue)

  def _writer_housekeeping(self) -> None:
    """Everything the writer owes the realtime thread, once per wakeup."""
    now_s = float(self._time_fn())
    self._poll_enable(now_s)
    self._drain_deferred_logs(now_s)

  def _poll_enable(self, now_s: float) -> None:
    """Bounded-cadence ``Params`` round-trip, on the WRITER thread.

    Publishes a plain bool the realtime thread reads once per tick; the realtime
    thread applies the transition itself. A failed read is NOT a live disable —
    treating it as one would destroy 20 s of pre-window and every armed mark.
    """
    if now_s < self._enable_poll_at_s:
      return
    self._enable_poll_at_s = now_s + REFRESH_INTERVAL_S
    enabled = self._read_enabled()
    if enabled is None:
      return
    self._enabled_requested = enabled

  def _drain_deferred_logs(self, now_s: float) -> None:
    """Emit, off the realtime thread, everything the realtime thread could only
    count. Never raises.

    The counters are monotonic ints incremented only by the realtime thread and
    read only here, so neither side needs a lock: the worst case is reporting a
    total one increment stale, which the next drain corrects. Nothing is ever
    reset, so a drop can be late but never invisible.
    """
    try:
      # One-shots. Not rate limited — each can fire at most once in the process.
      if self._unknown_key_errors and not self._reported_unknown_key:
        self._reported_unknown_key = True
        cloudlog.error("longitudinal mark recorder inert: Params key '%s' is unknown to the compiled params_pyx; " +
                       "rebuild it with scons after editing common/params_keys.h", ENABLE_PARAM)
      if self._dead and not self._reported_death:
        self._reported_death = True
        cloudlog.error("longitudinal mark recorder disabled after %d consecutive capture errors (%s)",
                       MAX_CONSECUTIVE_CAPTURE_ERRORS, self._death_reason)

      # Running counters, snapshotted once so the message and the bookkeeping
      # cannot disagree if the realtime thread increments mid-drain.
      queue_full = self._dropped_queue_full
      pending_cap = self._dropped_pending_cap
      latch_overflow = self._dropped_latch_overflow
      handoff_errors = self._dropped_handoff_errors
      press_socket_errors = self._press_socket_errors
      param_errors = self._param_read_errors
      total_dropped = self._dropped_marks

      pending_logs: list[tuple] = []
      if queue_full > self._reported_queue_full:
        pending_logs.append(("longitudinal mark dropped: writer queue full (new=%d, dropped=%d)",
                             queue_full - self._reported_queue_full, total_dropped))
      if pending_cap > self._reported_pending_cap:
        pending_logs.append(("longitudinal mark dropped: more than %d marks armed at once (new=%d, dropped=%d)",
                             MAX_PENDING_MARKS, pending_cap - self._reported_pending_cap, total_dropped))
      if latch_overflow > self._reported_latch_overflow:
        pending_logs.append(("longitudinal mark dropped: more than %d presses latched before the next planner tick " +
                             "(new=%d, dropped=%d)", MAX_LATCHED_PRESSES,
                             latch_overflow - self._reported_latch_overflow, total_dropped))
      if handoff_errors > self._reported_handoff_errors:
        pending_logs.append(("longitudinal mark dropped: handoff to the writer failed (new=%d, dropped=%d, last=%s)",
                             handoff_errors - self._reported_handoff_errors, total_dropped, self._last_handoff_error))
      if press_socket_errors > self._reported_press_socket_errors:
        pending_logs.append(("longitudinal mark press socket failed (new=%d, last=%s)",
                             press_socket_errors - self._reported_press_socket_errors, self._last_press_socket_error))
      if param_errors > self._reported_param_errors:
        pending_logs.append(("longitudinal mark recorder param read failed %d time(s); keeping the previous enable " +
                             "state (last=%s)", param_errors - self._reported_param_errors, self._last_param_error))

      if not pending_logs:
        return
      if now_s - self._last_deferred_log_s < DROP_LOG_INTERVAL_S:
        return  # counters keep running; the next drain reports the true totals
      self._last_deferred_log_s = now_s
      self._reported_queue_full = queue_full
      self._reported_pending_cap = pending_cap
      self._reported_latch_overflow = latch_overflow
      self._reported_handoff_errors = handoff_errors
      self._reported_press_socket_errors = press_socket_errors
      self._reported_param_errors = param_errors
      for entry in pending_logs:
        cloudlog.error(entry[0], *entry[1:])
    except Exception:
      cloudlog.exception("longitudinal mark recorder could not report deferred diagnostics")

  def _write_mark(self, mark: dict[str, Any], rows: list[tuple]) -> None:
    now_s = float(self._time_fn())
    if now_s < self._disk_backoff_until_s:
      # _service_pending already removed this mark from _pending, so returning
      # here discards a flagged incident permanently. Never do that silently.
      self._disk_dropped_marks += 1
      if now_s - self._last_backoff_log_s >= DROP_LOG_INTERVAL_S:
        self._last_backoff_log_s = now_s
        cloudlog.error("longitudinal mark dropped: disk backoff active for another %.0fs (disk_dropped=%d)",
                       self._disk_backoff_until_s - now_s, self._disk_dropped_marks)
      return
    # Params read and realdata listdir happen HERE, on the writer thread, never
    # on the SCHED_FIFO planner thread.
    route = self._safe_param_str("CurrentRoute")
    seg = self._guess_current_segment(route)
    press_ns = int(mark["pressLogMonoTime"])
    name = f"longmark__{route or 'unknownroute'}--{seg if seg is not None else 'X'}__{press_ns}.jsonl"
    path = self._marks_dir / name

    cfg = mark.get("liveTuneCfg")
    # Serialize BEFORE the header is built. The header is line 1 and cannot be
    # revised once written, so the truncation decision has to be made first:
    # declaring rowCount=len(rows) / truncated=False and only then discovering the
    # byte budget mid-body made the header describe 195 rows that were never
    # emitted, and mark_cli reports header["truncated"] verbatim as "complete".
    lines, truncated_at_byte = _serialize_rows(rows)
    kept = len(lines)
    coverage_tolerance_ns = int(math.ceil(1.5e9 / EXPECTED_HZ))
    window_start_ns = press_ns - int(PRE_WINDOW_S * 1e9)
    window_end_ns = press_ns + int(POST_WINDOW_S * 1e9)
    coverage_edges: list[str] = []
    if kept == 0 or int(rows[0][0]) > window_start_ns + coverage_tolerance_ns:
      coverage_edges.append("start")
    if kept == 0 or int(rows[kept - 1][0]) < window_end_ns - coverage_tolerance_ns:
      coverage_edges.append("end")
    truncated_by_byte = truncated_at_byte is not None
    header = {
      "type": "header",
      "schemaVersion": SCHEMA_VERSION,
      "recorderModule": "longitudinal_mark_recorder",
      "enableParam": ENABLE_PARAM,
      "rowFormat": "array",
      "columns": list(COLUMNS),
      "route": route,
      "segment": seg,
      "pressLogMonoTime": press_ns,
      "writtenUtc": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
      "preWindowS": PRE_WINDOW_S,
      "postWindowS": POST_WINDOW_S,
      # Authoritative: these describe the rows this file actually contains.
      "rowCount": kept,
      "rowCountRequested": len(rows),
      "droppedRows": len(rows) - kept,
      "firstModelLogMonoTime": (int(rows[0][0]) if kept else None),
      "lastModelLogMonoTime": (int(rows[kept - 1][0]) if kept else None),
      "ringMaxlen": RING_MAXLEN,
      "droppedMarks": int(self._dropped_marks),
      "droppedMarksDisk": int(self._disk_dropped_marks),
      "carFingerprint": mark.get("carFingerprint", ""),
      "gitCommit": self._safe_param_str("GitCommit"),
      "gitBranch": self._safe_param_str("GitBranch"),
      "liveTune": self._safe_live_tune(cfg),
      "truncated": truncated_by_byte or bool(coverage_edges),
      "truncatedByByte": truncated_by_byte,
      "truncatedEdges": coverage_edges,
      "coverageToleranceS": coverage_tolerance_ns / 1e9,
      "truncatedAtByte": truncated_at_byte,
      "maxBytesPerMark": MAX_BYTES_PER_MARK,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
      self._marks_dir.mkdir(parents=True, exist_ok=True)
      self._prune()
      renamed = False
      try:
        with open(tmp, "w", encoding="utf-8") as f:
          f.write(json.dumps(header, separators=(",", ":")) + "\n")
          for line in lines:
            f.write(line)
          if truncated_at_byte is not None:
            f.write(json.dumps({"type": "truncated", "atByte": truncated_at_byte, "rowCount": kept,
                                "rowCountRequested": len(rows)}, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
        renamed = True
      finally:
        # The realistic ENOSPC shape is open() succeeding and write/flush/close
        # failing, which leaves a partial .jsonl.tmp behind. Without this the
        # orphans accumulate forever.
        if not renamed:
          try:
            os.unlink(tmp)
          except OSError:
            pass
    except OSError:
      self._disk_backoff_until_s = now_s + DISK_BACKOFF_S
      cloudlog.exception("longitudinal mark sidecar write failed; backing off %.0fs", DISK_BACKOFF_S)
      return
    cloudlog.event("longitudinal mark saved", path=str(path), route=route, segment=seg,
                   press_mono_ns=press_ns, rows=kept, rows_requested=len(rows),
                   truncated=truncated_at_byte is not None)

  def _prune(self) -> None:
    """Delete oldest-by-mtime until the post-write count is exactly MAX_MARK_FILES
    and the directory fits MAX_MARK_DIR_BYTES. Every stat/unlink is individually
    guarded so a file vanishing mid-prune cannot abort the write."""
    stranded_bytes = 0
    try:
      # Sweep orphaned tmp files from a failed write first. They are never
      # useful, ``longmark__*.jsonl`` does not match them, and this runs before
      # this mark's own tmp is created on the single writer thread, so nothing
      # in flight can be deleted. Bytes we fail to reclaim still count against
      # the directory budget.
      for p in self._marks_dir.glob("longmark__*.jsonl.tmp"):
        try:
          size = p.stat().st_size
        except OSError:
          continue
        try:
          p.unlink()
        except OSError:
          stranded_bytes += size
      entries = []
      for p in self._marks_dir.glob("longmark__*.jsonl"):
        try:
          st = p.stat()
        except OSError:
          continue
        entries.append((st.st_mtime, st.st_size, p))
      entries.sort(key=lambda e: e[0])
    except OSError:
      return
    total = sum(e[1] for e in entries) + stranded_bytes
    while entries and (len(entries) >= MAX_MARK_FILES or total > MAX_MARK_DIR_BYTES):
      _mtime, size, victim = entries.pop(0)
      try:
        victim.unlink()
      except OSError:
        continue
      total -= size

  def _safe_param_str(self, key: str) -> str:
    try:
      raw = self._params.get(key)
      if raw is None:
        return ""
      if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="ignore")
      return str(raw)
    except Exception:
      return ""

  @staticmethod
  def _safe_live_tune(cfg) -> dict[str, Any] | None:
    if cfg is None:
      return None
    try:
      return {k: _json_safe(v) for k, v in cfg.as_dict().items()}
    except Exception:
      return None

  def _guess_current_segment(self, route: str) -> int | None:
    if not route:
      return None
    prefix = f"{route}--"
    segs: list[int] = []
    try:
      for entry in os.listdir(self._realdata_dir):
        if not entry.startswith(prefix):
          continue
        try:
          segs.append(int(entry[len(prefix):]))
        except ValueError:
          continue
    except OSError:
      return None
    return max(segs) if segs else None


_IDLE = object()   # the writer woke on its own poll timeout, with no work item


def _request_stop(work_queue: queue.Queue) -> None:
  """Best-effort stop sentinel. Never raises: a full queue simply means the writer
  notices ``_stop_requested`` / its dead weakref on the next poll instead."""
  try:
    work_queue.put_nowait(None)
  except Exception:
    pass


def _writer_main(recorder_ref, work_queue: queue.Queue) -> None:
  """Writer thread body.

  Module-level, and holds only a ``weakref`` to the recorder, on purpose: a bound
  ``self._writer_loop`` target kept the recorder — and its 780-slot ring — alive
  for the life of the process, so every construction leaked an un-reapable daemon
  thread. Here the recorder is resolved per iteration and dropped again before the
  next ``get()``, so the loop exits once nothing else references it.

  The ``get`` timeout is what lets the writer poll ``Params`` and drain the
  realtime thread's deferred diagnostics even when no mark is ever flushed.
  """
  MarkRecorder._demote_this_thread()
  while True:
    try:
      item = work_queue.get(timeout=WRITER_POLL_INTERVAL_S)
    except queue.Empty:
      item = _IDLE
    if item is None:
      return                                # explicit stop sentinel
    rec = recorder_ref()
    if rec is None:
      return                                # recorder was collected; nothing to serve
    if item is not _IDLE:
      try:
        rec._write_mark(*item)
      except Exception:
        cloudlog.exception("longitudinal mark write failed")
    try:
      rec._writer_housekeeping()
    except Exception:
      # A writer that dies here stops reporting the realtime thread's drops, i.e.
      # silent loss — the one outcome this module must never produce.
      cloudlog.exception("longitudinal mark writer housekeeping failed")
    stop = bool(rec._stop_requested)
    rec = None                              # never hold the recorder across the next get()
    if stop and work_queue.empty():
      return


def _serialize_rows(rows: list[tuple]) -> tuple[list[str], int | None]:
  """Serialize rows to JSONL lines under the byte budget, on the writer thread.

  Returns ``(lines, truncated_at_byte)``; ``truncated_at_byte`` is ``None`` when
  every row fit, else the cumulative byte offset at which the budget was blown
  (the offset the trailing ``truncated`` record reports, unchanged from before).
  Rows are pure ASCII after ``json.dumps``, so ``len(line)`` is the byte count.
  """
  lines: list[str] = []
  written = 0
  for row in rows:
    line = json.dumps([_json_safe(v) for v in row], separators=(",", ":")) + "\n"
    written += len(line)
    if written > MAX_BYTES_PER_MARK:
      return lines, written
    lines.append(line)
  return lines, None


def _safe_reason(exc: BaseException | None) -> str:
  """One short line naming an exception, cheap enough for the realtime thread.

  Deliberately not ``traceback.format_exc()``: that reads source through
  ``linecache``, i.e. disk I/O, which is exactly what must not happen here.
  """
  if exc is None:
    return "unknown"
  try:
    return f"{type(exc).__name__}: {exc}"
  except Exception:
    try:
      return type(exc).__name__
    except Exception:
      return "unknown"


def _json_safe(value: Any) -> Any:
  """Writer-thread coercion to a JSON primitive. Never raises."""
  if value is None or isinstance(value, (bool, int, str)):
    return value
  if isinstance(value, float):
    return round(value, 4) if math.isfinite(value) else None
  # numpy scalars (np.bool_ is NOT a bool subclass) expose .item().
  item = getattr(value, "item", None)
  if callable(item):
    try:
      scalar = item()
    except Exception:
      return str(value)
    if scalar is None or isinstance(scalar, (bool, int, str)):
      return scalar
    if isinstance(scalar, float):
      return round(scalar, 4) if math.isfinite(scalar) else None
  return str(value)
