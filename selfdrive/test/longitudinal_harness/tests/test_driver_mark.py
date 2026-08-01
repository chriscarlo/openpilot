"""Driver-mark extraction, tracing and CLI.

The synthetic route below is deliberately built the "legacy" way -- no
``RadarState.replayInputs`` contract -- because a driver mark must survive on a
capture that predates every replay contract. What it does carry is the thing the
feature is actually about: a ``bookmarkButton`` press, its ``userBookmark`` echo,
a ``LONGFLAG`` swaglog payload, a hard braking event, a second press ten seconds
later, and one press that lands nowhere near a recorded frame.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest

from cereal import log, messaging
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.selfdrive.test.longitudinal_harness import mark_cli
from openpilot.selfdrive.test.longitudinal_harness.catalog import get_route_rows, open_catalog
from openpilot.selfdrive.test.longitudinal_harness.marks import (
  DEFAULT_MARK_SIDECAR_ROOTS,
  MARK_SIDECAR_DEVICE_DIR,
  MARK_SIDECAR_PULL_COMMAND,
  POST_WINDOW_S,
  PRE_WINDOW_S,
  compute_mark_metrics,
  find_mark_sidecar,
  format_ascii_trace,
  join_sidecar_rows,
  load_mark_sidecar,
  load_trace,
  load_trace_window,
  resolve_segment_rlog_chain,
  scan_root_marks,
  scan_segment_marks,
)
from openpilot.selfdrive.test.longitudinal_harness.route_extract import (
  COOLDOWN_BY_TYPE_S,
  DRIVER_MARK_ECHO_WINDOW_S,
  DRIVER_MARK_FRAME_TOLERANCE_S,
  EXTRACTOR_VERSION,
  EpisodeCandidate,
  EpisodeNotReplayableError,
  SUPPRESSION_BY_TYPE_S,
  WINDOWS_BY_TYPE,
  _parse_longflag,
  _should_suppress_candidate,
  collapse_driver_mark_echoes,
  detect_episode_candidates,
  extract_ev6_episodes,
  index_ev6_routes,
  is_driver_mark_press_group,
  load_route_scan,
  split_exact_replay_runs,
)
from openpilot.tools.lib.logreader import save_log


_DT_NS = 50_000_000
_BASE_NS = 10_000_000_000
_FRAME_COUNT = 700
_EGO_SPEED_MPS = 25.0
_LEAD_GAP_M = 40.0
_FIRST_PRESS_FRAME = 400
_SECOND_PRESS_FRAME = 600
_BRAKE_START_FRAME = 380
_BRAKE_END_FRAME = 402
_HARSH_ACCEL_MPS2 = -3.0
_ORPHAN_PRESS_NS = _BASE_NS - 5_000_000_000
# Two presses 0.8 s apart: past the 0.5 s echo/suppression window, so both are
# deliberate incidents that must survive to the catalog. Their -20/+6 s windows
# both clamp to the same recorded frames (see the blocker test below).
_COLLIDING_PRESS_FRAME_A = 600
_COLLIDING_PRESS_FRAME_B = 616
_COLLIDING_DROPPED_FRAMES = tuple(range(200, 216))
# A RadarD hole with one surviving frame in the middle of it, and one press on
# each side of that frame. Both presses are inside DRIVER_MARK_FRAME_TOLERANCE_S
# (1.0 s) of the survivor and further from every other surviving frame, so both
# anchor to it and get an IDENTICAL event_t_s -- while being 1.4 s apart on the
# press clock, well past the recorder's 1.0 s debounce.
_SAME_ANCHOR_SURVIVING_FRAME = 600
_SAME_ANCHOR_PRESS_FRAME_A = 586
_SAME_ANCHOR_PRESS_FRAME_B = 614
_SAME_ANCHOR_DROPPED_FRAMES = tuple(range(570, _SAME_ANCHOR_SURVIVING_FRAME)) + tuple(range(_SAME_ANCHOR_SURVIVING_FRAME + 1, 631))
# A bare userBookmark 0.3 s AHEAD of a real press: inside the 0.5 s echo window,
# so it lands in the same group and would become that group's anchor.
_EARLY_ECHO_FRAME = 394
# Real segments are 60 s (SEGMENT_LENGTH in system/loggerd/loggerd.h); 20 s here
# keeps the synthetic logs small while still forcing a boundary crossing for the
# 26 s driver-mark window.
_SEG_FRAME_COUNT = 400
_BOUNDARY_PRESS_FRAME = _SEG_FRAME_COUNT + 10
_BUTTONS_CC_RELPATH = "selfdrive/ui/sunnypilot/qt/onroad/buttons.cc"
_RECORDER_RELPATH = "sunnypilot/selfdrive/controls/lib/longitudinal_mark_recorder.py"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _make_car_params():
  cp = CarInterface.get_non_essential_params(CAR.KIA_EV6)
  cp.carFingerprint = "KIA_EV6"
  cp.openpilotLongitudinalControl = True
  cp.radarUnavailable = True
  if len(cp.safetyConfigs):
    cp.safetyConfigs[0].safetyParam = 21
  return cp


def _frame_base_ns(frame_idx: int) -> int:
  return _BASE_NS + frame_idx * _DT_NS


def _press_ns(frame_idx: int) -> int:
  return _frame_base_ns(frame_idx) + 30_000_000


def _longflag_record(payload: dict[str, Any]) -> str:
  return json.dumps({
    "msg": "LONGFLAG " + json.dumps(payload, sort_keys=True),
    "ctx": {"daemon": "feedbackd"},
    "levelnum": 20,
  }, sort_keys=True)


def _append_model(messages: list[Any], *, mono_time_ns: int) -> None:
  model = messaging.new_message("modelV2")
  model.logMonoTime = mono_time_ns
  model.valid = True
  position = log.XYZTData.new_message()
  position.x = [0.0, 50.0, 100.0]
  position.y = [0.0, 0.0, 0.0]
  position.z = [0.0, 0.0, 0.0]
  position.t = [0.0, 0.5, 1.0]
  model.modelV2.position = position
  leads = model.modelV2.init("leadsV3", 3)
  for slot, lead in enumerate(leads):
    lead.t = [0.0, 0.5]
    lead.x = [_LEAD_GAP_M + RADAR_TO_CAMERA, _LEAD_GAP_M + RADAR_TO_CAMERA]
    lead.y = [0.0, 0.0]
    lead.v = [_EGO_SPEED_MPS, _EGO_SPEED_MPS]
    lead.a = [0.0, 0.0]
    lead.prob = 0.97 if slot == 0 else 0.0
    lead.probTime = 0.05
  messages.append(model.as_reader())


def _write_marked_route(route_root: Path,
                        *,
                        drop_car_state_frames: tuple[int, ...] = (),
                        include_orphan_press: bool = True,
                        include_second_press: bool = True,
                        include_longflag: bool = True,
                        press_frames: tuple[int, ...] | None = None,
                        lone_user_bookmark_frames: tuple[int, ...] = ()) -> Path:
  segment_dir = route_root / "0"
  segment_dir.mkdir(parents=True, exist_ok=True)
  rlog_path = segment_dir / "rlog.zst"
  messages: list[Any] = []

  init_data = messaging.new_message("initData")
  init_data.logMonoTime = 900_000_000
  init_data.valid = True
  messages.append(init_data.as_reader())

  car_params = messaging.new_message("carParams")
  car_params.logMonoTime = 950_000_000
  car_params.valid = True
  car_params.carParams = _make_car_params()
  messages.append(car_params.as_reader())

  if include_orphan_press:
    # Five seconds before the first recorded RadarD publication: a real press
    # that can never be anchored to a frame.
    orphan = messaging.new_message("bookmarkButton")
    orphan.logMonoTime = _ORPHAN_PRESS_NS
    orphan.valid = True
    messages.append(orphan.as_reader())

  for frame_idx in range(_FRAME_COUNT):
    frame_base = _frame_base_ns(frame_idx)
    braking = _BRAKE_START_FRAME <= frame_idx < _BRAKE_END_FRAME
    planner_accel_mps2 = _HARSH_ACCEL_MPS2 if braking else 0.2

    if frame_idx not in drop_car_state_frames:
      car_state = messaging.new_message("carState")
      car_state.logMonoTime = frame_base
      car_state.valid = True
      car_state.carState.vEgo = _EGO_SPEED_MPS
      car_state.carState.aEgo = planner_accel_mps2
      car_state.carState.vCruise = 108.0
      messages.append(car_state.as_reader())

    controls_state = messaging.new_message("controlsState")
    controls_state.logMonoTime = frame_base + 2_000_000
    controls_state.valid = True
    controls_state.controlsState.longControlState = "pid"
    messages.append(controls_state.as_reader())

    selfdrive_state = messaging.new_message("selfdriveState")
    selfdrive_state.logMonoTime = frame_base + 4_000_000
    selfdrive_state.valid = True
    selfdrive_state.selfdriveState.enabled = True
    selfdrive_state.selfdriveState.personality = "standard"
    messages.append(selfdrive_state.as_reader())

    car_control = messaging.new_message("carControl")
    car_control.logMonoTime = frame_base + 6_000_000
    car_control.valid = True
    car_control.carControl.longActive = True
    car_control.carControl.actuators.accel = planner_accel_mps2
    car_control.carControl.orientationNED = [0.0, 0.01, 0.0]
    messages.append(car_control.as_reader())

    car_output = messaging.new_message("carOutput")
    car_output.logMonoTime = frame_base + 7_000_000
    car_output.valid = True
    car_output.carOutput.actuatorsOutput.accel = planner_accel_mps2
    messages.append(car_output.as_reader())

    model_time = frame_base + 10_000_000
    _append_model(messages, mono_time_ns=model_time)

    radar_state = messaging.new_message("radarState")
    radar_state.logMonoTime = frame_base + 15_000_000
    radar_state.valid = True
    radar_state.radarState.mdMonoTime = model_time
    radar_state.radarState.carStateMonoTime = frame_base
    lead = radar_state.radarState.leadOne
    lead.status = True
    lead.dRel = _LEAD_GAP_M
    lead.vRel = -4.0 if braking else 1.0
    lead.vLead = _EGO_SPEED_MPS + (-4.0 if braking else 1.0)
    lead.vLeadK = lead.vLead
    lead.aLeadK = -1.5 if braking else 0.0
    lead.modelProb = 0.97
    lead.yRel = 0.0
    radar_state.radarState.leadTwo.status = False
    messages.append(radar_state.as_reader())

    longitudinal_plan = messaging.new_message("longitudinalPlan")
    longitudinal_plan.logMonoTime = frame_base + 20_000_000
    longitudinal_plan.valid = True
    longitudinal_plan.longitudinalPlan.modelMonoTime = model_time
    longitudinal_plan.longitudinalPlan.aTarget = planner_accel_mps2
    longitudinal_plan.longitudinalPlan.longitudinalPlanSource = "lead0" if braking else "cruise"
    messages.append(longitudinal_plan.as_reader())

    if press_frames is None:
      resolved_press_frames = [_FIRST_PRESS_FRAME] + ([_SECOND_PRESS_FRAME] if include_second_press else [])
    else:
      resolved_press_frames = list(press_frames)

    if frame_idx in lone_user_bookmark_frames:
      # feedbackd's LKAS / audio-completion path: userBookmark with no
      # bookmarkButton anywhere near it.
      lone_echo = messaging.new_message("userBookmark")
      lone_echo.logMonoTime = _press_ns(frame_idx)
      lone_echo.valid = True
      messages.append(lone_echo.as_reader())

    if frame_idx in resolved_press_frames:
      button = messaging.new_message("bookmarkButton")
      button.logMonoTime = _press_ns(frame_idx)
      button.valid = True
      messages.append(button.as_reader())

      # feedbackd's echo, 180 ms behind the press: one incident, not two.
      echo = messaging.new_message("userBookmark")
      echo.logMonoTime = _press_ns(frame_idx) + 180_000_000
      echo.valid = True
      messages.append(echo.as_reader())

      if include_longflag and frame_idx == resolved_press_frames[0]:
        payload = {"source": "sidebar", "seq": 1}
        flag = messaging.new_message(None, valid=True, logMessage=_longflag_record(payload))
        flag.logMonoTime = _press_ns(frame_idx) + 5_000_000
        messages.append(flag.as_reader())

        noise = messaging.new_message(None, valid=True, logMessage=json.dumps({"msg": "unrelated chatter"}))
        noise.logMonoTime = _press_ns(frame_idx) + 6_000_000
        messages.append(noise.as_reader())

  save_log(str(rlog_path), messages)
  return rlog_path


@pytest.fixture
def marked_route(tmp_path: Path) -> Path:
  route_root = tmp_path / "route_marked"
  _write_marked_route(route_root)
  return route_root


def _load_scan(tmp_path: Path, route_root: Path):
  conn = open_catalog(tmp_path / "catalog.db")
  index_ev6_routes(conn, [route_root])
  route_row = get_route_rows(conn)[0]
  scan = load_route_scan(conn, route_row)
  return conn, scan


def test_driver_mark_type_tables() -> None:
  assert EXTRACTOR_VERSION == "ev6_v12_driver_mark"
  assert WINDOWS_BY_TYPE["driver_mark"] == (-20.0, 6.0)
  # COOLDOWN_BY_TYPE_S is only ever indexed by string literals inside the six
  # heuristic detectors, so a driver_mark entry there would be dead code.
  assert "driver_mark" not in COOLDOWN_BY_TYPE_S
  # SUPPRESSION_BY_TYPE_S is indexed on event_t_s, which for a driver mark is the
  # anchored FRAME's clock rather than the press clock; a radius there deletes
  # real presses (see test_driver_mark_suppression_ignores_the_anchor_frame_clock).
  # The only echo radius is DRIVER_MARK_ECHO_WINDOW_S, on the exact press clock.
  assert "driver_mark" not in SUPPRESSION_BY_TYPE_S
  assert DRIVER_MARK_ECHO_WINDOW_S == 0.5


@pytest.mark.parametrize("record", [
  "",
  "not json at all",
  "LONGFLAG but not json",
  json.dumps({"msg": "no marker here"}),
  json.dumps({"msg": "LONGFLAG {not json}"}),
  json.dumps({"msg": ["LONGFLAG ", "list"]}),
  json.dumps({"msg": "LONGFLAG [1, 2, 3]"}),
  json.dumps(["LONGFLAG {}"]),
  '{"msg": "LONGFLAG {}"',
])
def test_parse_longflag_never_raises_and_rejects_malformed(record: str) -> None:
  assert _parse_longflag(record) is None


def test_parse_longflag_extracts_payload() -> None:
  assert _parse_longflag(_longflag_record({"source": "sidebar", "seq": 7})) == {"source": "sidebar", "seq": 7}


def test_route_scan_collects_marks_and_payloads(tmp_path: Path, marked_route: Path) -> None:
  conn, scan = _load_scan(tmp_path, marked_route)
  try:
    # Two anchored presses plus the deliberately unanchorable one. The echo of
    # each press has already been folded in.
    assert len(scan.driver_marks) == 3
    orphan = [mark for mark in scan.driver_marks if mark.status == "orphan"]
    assert len(orphan) == 1
    assert orphan[0].press_log_mono_time_ns == _ORPHAN_PRESS_NS
    assert orphan[0].frame_index is None

    anchored = sorted(
      (mark for mark in scan.driver_marks if mark.status != "orphan"),
      key=lambda mark: mark.press_log_mono_time_ns,
    )
    assert [mark.press_log_mono_time_ns for mark in anchored] == [
      _press_ns(_FIRST_PRESS_FRAME), _press_ns(_SECOND_PRESS_FRAME),
    ]
    assert anchored[0].services == ("bookmarkButton", "userBookmark")
    assert anchored[0].status == "ok"
    assert anchored[0].payload == {"source": "sidebar", "seq": 1}
    assert anchored[1].status == "no_payload"
    assert anchored[1].payload is None

    assert len(scan.longflag_payloads) == 1
    assert scan.longflag_payloads[0]["payload"] == {"source": "sidebar", "seq": 1}
  finally:
    conn.close()


def test_driver_mark_candidates_carry_press_time_and_window(tmp_path: Path, marked_route: Path) -> None:
  conn, scan = _load_scan(tmp_path, marked_route)
  try:
    candidates = [c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"]
    assert len(candidates) == 2
    first = candidates[0]
    assert first.confidence == 1.0
    assert first.rank_score == 1000.0
    assert first.event_t_s == pytest.approx(20.0, abs=0.06)
    assert first.t_start_s == pytest.approx(0.0, abs=0.06)
    assert first.t_end_s == pytest.approx(26.0, abs=0.06)
    # Phase 2's sidecar join keys on the press envelope time, not the frame's.
    assert first.notes_json["driverMarkPressLogMonoTime"] == _press_ns(_FIRST_PRESS_FRAME)
    assert first.notes_json["eventLogMonoTime"] != _press_ns(_FIRST_PRESS_FRAME)
    assert first.notes_json["driverMarkServices"] == ["bookmarkButton", "userBookmark"]
    assert first.metrics["driverMarkPayload"] == {"source": "sidebar", "seq": 1}
  finally:
    conn.close()


def test_two_presses_ten_seconds_apart_are_two_episodes(tmp_path: Path, marked_route: Path) -> None:
  conn, scan = _load_scan(tmp_path, marked_route)
  try:
    candidates = [c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"]
    event_times = sorted(c.event_t_s for c in candidates)
    assert len(event_times) == 2
    assert event_times[1] - event_times[0] == pytest.approx(10.0, abs=0.06)
    # Their 26 s windows overlap heavily; the overlap heuristic must not merge them.
    assert candidates[0].t_end_s > candidates[1].t_start_s
    assert len({c.episode_key for c in candidates}) == 2
  finally:
    conn.close()


def _bare_candidate(episode_type: str,
                    *,
                    event_t_s: float,
                    t_start_s: float,
                    t_end_s: float,
                    key_discriminator: str | None = None) -> EpisodeCandidate:
  return EpisodeCandidate(
    route_id=1,
    route_key="route",
    episode_type=episode_type,
    event_t_s=event_t_s,
    seg_start=0,
    seg_end=0,
    t_start_s=t_start_s,
    t_end_s=t_end_s,
    confidence=1.0,
    rank_score=1000.0,
    metrics={},
    notes_json={},
    frames=[],
    key_discriminator=key_discriminator,
  )


def test_overlap_suppression_is_exempted_for_driver_marks() -> None:
  first = _bare_candidate("driver_mark", event_t_s=20.0, t_start_s=0.0, t_end_s=26.0, key_discriminator="press1")
  second = _bare_candidate("driver_mark", event_t_s=30.0, t_start_s=10.0, t_end_s=36.0, key_discriminator="press2")
  # 16 s of overlap: any other episode type would collapse this pair.
  assert not _should_suppress_candidate(second, first)
  approach_first = _bare_candidate("approach", event_t_s=20.0, t_start_s=0.0, t_end_s=26.0)
  approach_second = _bare_candidate("approach", event_t_s=40.0, t_start_s=10.0, t_end_s=36.0)
  assert _should_suppress_candidate(approach_second, approach_first)
  # The same press twice collapses; that is the only driver-mark collapse there is.
  same_press = _bare_candidate("driver_mark", event_t_s=20.0, t_start_s=0.0, t_end_s=26.0, key_discriminator="press1")
  assert _should_suppress_candidate(same_press, first)


def test_driver_mark_suppression_ignores_the_anchor_frame_clock() -> None:
  """Two presses that anchored to the SAME radarState frame are still two presses.

  event_t_s for a driver mark is the anchored FRAME's route clock, not the press
  clock, so a RadarD hole can give two presses up to 2 x
  DRIVER_MARK_FRAME_TOLERANCE_S apart an identical event_t_s. Suppressing on that
  number deletes the second human press outright.
  """
  first = _bare_candidate("driver_mark", event_t_s=30.0, t_start_s=10.0, t_end_s=36.0, key_discriminator="press1000")
  colliding = _bare_candidate("driver_mark", event_t_s=30.0, t_start_s=10.0, t_end_s=36.0, key_discriminator="press2400000000")
  assert not _should_suppress_candidate(colliding, first)
  # And an unidentifiable driver mark is never silently swallowed either.
  anonymous = _bare_candidate("driver_mark", event_t_s=30.0, t_start_s=10.0, t_end_s=36.0)
  assert not _should_suppress_candidate(anonymous, first)
  assert not _should_suppress_candidate(anonymous, anonymous)


def test_mark_survives_broken_exact_replay_join(tmp_path: Path) -> None:
  route_root = tmp_path / "route_gap"
  # Deleting the carState targets right before the press drops those RadarD
  # frames, splitting the exact-replay runs across the mark.
  dropped = tuple(range(390, 399))
  _write_marked_route(route_root, drop_car_state_frames=dropped, include_second_press=False)
  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [route_root])
    route_row = get_route_rows(conn)[0]
    with pytest.warns(RuntimeWarning, match="Dropped 9/"):
      scan = load_route_scan(conn, route_row)
    assert len(split_exact_replay_runs(scan.frames)) > 1
    candidates = [c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"]
    assert len(candidates) == 1
    # One candidate is not enough: a per-run detector would also produce exactly
    # one, from a single run, with a window truncated at the dropout. Assert the
    # surviving window actually SPANS the dropped range.
    dropout_start_s = dropped[0] * _DT_NS / 1e9
    dropout_end_s = (dropped[-1] + 1) * _DT_NS / 1e9
    mark = candidates[0]
    assert mark.t_start_s < dropout_start_s
    assert mark.t_end_s > dropout_end_s
    assert mark.t_start_s == pytest.approx(0.0, abs=0.06)
    assert mark.t_end_s == pytest.approx(26.0, abs=0.06)
    frame_times = [frame.t_s for frame in mark.frames]
    assert [t for t in frame_times if t < dropout_start_s], "no frames before the dropout"
    assert [t for t in frame_times if t >= dropout_end_s], "no frames after the dropout"
    # And the dropout really is a hole inside that window, not merely an edge.
    assert not [t for t in frame_times if dropout_start_s <= t < dropout_end_s]
  finally:
    conn.close()


def test_unreplayable_driver_mark_is_catalogued_without_a_bundle(tmp_path: Path, marked_route: Path, monkeypatch) -> None:
  from openpilot.selfdrive.test.longitudinal_harness import route_extract

  def _refuse(*_args, **_kwargs):
    raise EpisodeNotReplayableError("synthetic dependency gap")

  monkeypatch.setattr(route_extract, "write_episode_bundle", _refuse)
  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [marked_route])
    recorded = extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    marks = [entry for entry in recorded if entry["episodeType"] == "driver_mark"]
    assert marks, "the driver mark must never be dropped on the floor"
    anchored = [entry for entry in marks if entry["status"] == "recorded_without_bundle"]
    orphans = [entry for entry in marks if entry["status"] == "orphan_recorded_without_bundle"]
    assert len(anchored) == 2
    assert len(orphans) == 1
    assert all(entry["reason"] == "synthetic dependency gap" for entry in anchored)

    rows = conn.execute(
      "SELECT bundle_path, notes_json FROM episodes WHERE episode_type = 'driver_mark'"
    ).fetchall()
    assert len(rows) == len(marks)
    for row in rows:
      assert row["bundle_path"] is None
      notes = json.loads(row["notes_json"])
      assert notes["bundleUnavailableReason"]
      assert "driverMarkPressLogMonoTime" in notes

    # Heuristic episodes keep the old skip-and-report behavior.
    others = [entry for entry in recorded if entry["episodeType"] != "driver_mark"]
    assert all(entry["status"] == "not_evaluated" for entry in others)
  finally:
    conn.close()


def test_scan_segment_marks_collapses_echo(marked_route: Path) -> None:
  marks = scan_segment_marks(marked_route / "0" / "rlog.zst", route_key="route_marked", seg_idx=0)
  assert len(marks) == 3
  by_status = {mark.status for mark in marks}
  assert by_status == {"ok", "no_payload", "orphan"}
  ok_mark = next(mark for mark in marks if mark.status == "ok")
  assert ok_mark.services == ("bookmarkButton", "userBookmark")
  assert ok_mark.payload == {"source": "sidebar", "seq": 1}
  assert ok_mark.mark_id == f"route_marked--0--{_press_ns(_FIRST_PRESS_FRAME)}"
  # Segment-relative, not route-relative: this route is one segment, so the two
  # clocks happen to coincide here. See marks.DriverMark for why the field is
  # named t_seg_rel_s and must never be compared against DriverMarkPress.t_s.
  assert ok_mark.t_seg_rel_s == pytest.approx(20.015, abs=0.001)


def test_scan_root_marks_discovers_segment(tmp_path: Path, marked_route: Path) -> None:
  marks = scan_root_marks([marked_route.parent])
  assert [mark.route_key for mark in marks] == ["route_marked"] * 3
  assert all(mark.rlog_path is not None and mark.rlog_path.name == "rlog.zst" for mark in marks)
  assert {mark.seg_idx for mark in marks} == {0}


def test_load_trace_and_metrics_flag_the_harsh_brake(marked_route: Path) -> None:
  rlog_path = marked_route / "0" / "rlog.zst"
  rows = load_trace(rlog_path, _press_ns(_FIRST_PRESS_FRAME))
  assert rows, "the -20/+6 s window must contain planner frames"
  assert all(-20.1 <= row["t_rel_s"] <= 6.1 for row in rows)
  assert rows[0]["model_log_mono_time_ns"] > 0

  metrics = compute_mark_metrics(rows)
  assert metrics["minATargetMps2"] == pytest.approx(_HARSH_ACCEL_MPS2)
  assert metrics["derivedCategory"] == "harsh_brake"
  assert metrics["minGapM"] == pytest.approx(_LEAD_GAP_M)
  assert metrics["plannerSourceTransitionCount"] >= 1
  assert metrics["driverIntervened"] is False


def test_compute_mark_metrics_skips_rows_without_planner_accel() -> None:
  rows = [
    {"t_s": 0.0, "planner_accel_mps2": None},
    {"t_s": 0.5, "planner_accel_mps2": -0.5, "lead_d_rel_m": 30.0, "lead_v_rel_mps": -1.0},
    {"t_s": 1.0, "planner_accel_mps2": -0.6},
  ]
  metrics = compute_mark_metrics(rows)
  assert metrics["minATargetMps2"] == pytest.approx(-0.6)
  assert metrics["minATargetTRelS"] == pytest.approx(1.0)
  assert metrics["derivedCategory"] == "unspecified"


def test_format_ascii_trace_keeps_documented_columns(marked_route: Path) -> None:
  rows = load_trace(marked_route / "0" / "rlog.zst", _press_ns(_FIRST_PRESS_FRAME))
  braking_rows = [row for row in rows if row["planner_accel_mps2"] == pytest.approx(_HARSH_ACCEL_MPS2)]
  assert braking_rows
  rendered = format_ascii_trace(braking_rows[:5], title="unit test")
  assert "unit test" in rendered
  for token in ("v=", "aE=", "lng=", "ss=", "aTgt=", "cc=", "co=", "lc=", "src=", "stop=", "lead=("):
    assert token in rendered
  # Line 0 is the title, line 1 the legend; the frames start after them.
  first_frame_line = rendered.splitlines()[2]
  assert "lng=1" in first_frame_line and "ss=0" in first_frame_line
  assert "src=lead0" in first_frame_line and "stop=0" in first_frame_line
  # aLeadK and modelProb are restored, so the lead tuple has four members.
  lead_tuple = first_frame_line.split("lead=(")[1].split(")")[0]
  d_rel, v_rel, a_lead_k, model_prob = (part.strip() for part in lead_tuple.split(","))
  assert float(d_rel) == pytest.approx(_LEAD_GAP_M, abs=0.1)
  assert float(v_rel) == pytest.approx(-4.0, abs=0.01)
  assert float(a_lead_k) == pytest.approx(-1.5, abs=0.01)
  assert float(model_prob) == pytest.approx(0.97, abs=0.01)


def test_format_ascii_trace_handles_empty_window() -> None:
  assert "no frames in window" in format_ascii_trace([])


def _write_sidecar(path: Path, *, press_ns: int, route: str, model_times: list[int]) -> None:
  # Column names taken from the on-device schema
  # (sunnypilot/selfdrive/controls/lib/longitudinal_mark_recorder.COLUMNS).
  columns = ["modelLogMonoTime", "relatchActive", "accSourceActiveMode"]
  lines = [json.dumps({
    "type": "header",
    "schemaVersion": 1,
    "rowFormat": "array",
    "columns": columns,
    "route": route,
    "segment": 0,
    "pressLogMonoTime": press_ns,
    "rowCount": len(model_times),
    "truncated": False,
  })]
  lines.extend(json.dumps([model_ns, True, "lead0"]) for model_ns in model_times)
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_find_mark_sidecar_prefers_exact_press_time(tmp_path: Path) -> None:
  root = tmp_path / "long_marks"
  root.mkdir()
  exact = root / "longmark__routeA--0__1000.jsonl"
  _write_sidecar(exact, press_ns=1000, route="someOtherRoute", model_times=[1])
  near = root / "longmark__routeB--0__1080.jsonl"
  _write_sidecar(near, press_ns=1080, route="routeKey", model_times=[1])

  # Exact wins even when its route string disagrees with the local route key.
  assert find_mark_sidecar("routeKey", 1000, [root]) == exact
  # 80 ms off with a matching route still joins.
  assert find_mark_sidecar("routeKey", 1000 + 80_000_000, [root]) == near
  # 300 ms off does not.
  assert find_mark_sidecar("routeKey", 1080 + 300_000_000, [root]) is None
  assert find_mark_sidecar("routeKey", 1000, [tmp_path / "missing"]) is None


def test_find_mark_sidecar_ignores_unreadable_files(tmp_path: Path) -> None:
  root = tmp_path / "long_marks"
  root.mkdir()
  (root / "longmark__broken__0.jsonl").write_text("not json\n", encoding="utf-8")
  (root / "longmark__noheader__0.jsonl").write_text(json.dumps({"type": "row"}) + "\n", encoding="utf-8")
  assert find_mark_sidecar("routeKey", 1000, [root]) is None


def test_sidecar_rows_join_trace_by_model_mono_time(tmp_path: Path, marked_route: Path) -> None:
  rows = load_trace(marked_route / "0" / "rlog.zst", _press_ns(_FIRST_PRESS_FRAME))
  model_times = [row["model_log_mono_time_ns"] for row in rows[:10]]
  sidecar_path = tmp_path / "longmark__route_marked--0__1.jsonl"
  _write_sidecar(sidecar_path, press_ns=_press_ns(_FIRST_PRESS_FRAME), route="route_marked", model_times=model_times)

  loaded = load_mark_sidecar(sidecar_path)
  assert loaded is not None
  header, sidecar_rows = loaded
  assert header["rowCount"] == len(model_times)
  assert header["integrityStatus"] == "ok"
  assert header["truncated"] is False
  assert len(sidecar_rows) == len(model_times)
  matched = join_sidecar_rows(rows, sidecar_rows)
  assert matched == len(model_times)
  assert rows[0]["sidecar"]["accSourceActiveMode"] == "lead0"
  assert "sidecar" not in rows[-1]
  rendered = format_ascii_trace(rows[:1])
  assert "dbg gates=relatch" in rendered
  assert "acc=lead0" in rendered


def test_load_mark_sidecar_rejects_headerless_file(tmp_path: Path) -> None:
  path = tmp_path / "longmark__bad__0.jsonl"
  path.write_text("[1,2,3]\n", encoding="utf-8")
  assert load_mark_sidecar(path) is None


def test_load_mark_sidecar_marks_a_short_body_truncated(tmp_path: Path) -> None:
  path = tmp_path / "longmark__short__0.jsonl"
  path.write_text("\n".join([
    json.dumps({
      "type": "header",
      "schemaVersion": 1,
      "rowFormat": "array",
      "columns": ["modelLogMonoTime"],
      "pressLogMonoTime": 10,
      "rowCount": 2,
      "truncated": False,
    }),
    json.dumps([1]),
  ]) + "\n", encoding="utf-8")

  loaded = load_mark_sidecar(path)
  assert loaded is not None
  header, rows = loaded
  assert len(rows) == 1
  assert header["declaredTruncated"] is False
  assert header["truncated"] is True
  assert header["integrityStatus"] == "truncated"
  assert "rowCount declares 2 but file contains 1 valid rows" in header["integrityErrors"]


def test_mark_cli_list_json(marked_route: Path, capsys) -> None:
  assert mark_cli.main(["list", "--root", str(marked_route.parent), "--json"]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert len(payload) == 3
  assert {entry["status"] for entry in payload} == {"ok", "no_payload", "orphan"}
  for key in ("routeKey", "segIdx", "pressMonoTimeNs", "tSegRelS", "services", "payload", "status", "markId"):
    assert key in payload[0]
  # The route-relative name is gone on purpose: it was the same name as
  # route_extract.DriverMarkPress.t_s on a different clock.
  assert "tRelS" not in payload[0]


def test_mark_cli_list_table(marked_route: Path, capsys) -> None:
  assert mark_cli.main(["list", "--root", str(marked_route.parent)]) == 0
  out = capsys.readouterr().out
  assert "markId" in out
  assert str(_press_ns(_FIRST_PRESS_FRAME)) in out


def test_mark_cli_show_json_without_sidecar(marked_route: Path, capsys) -> None:
  mark_id = f"route_marked--0--{_press_ns(_FIRST_PRESS_FRAME)}"
  assert mark_cli.main([
    "show", mark_id, "--root", str(marked_route.parent), "--sidecar-root", str(marked_route / "nope"), "--json",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["mark"]["markId"] == mark_id
  assert payload["metrics"]["derivedCategory"] == "harsh_brake"
  assert payload["sidecar"] == {"present": False}
  assert payload["trace"]


def test_mark_cli_show_ascii_degrades_without_sidecar(marked_route: Path, capsys) -> None:
  mark_id = f"route_marked--0--{_press_ns(_FIRST_PRESS_FRAME)}"
  assert mark_cli.main(["show", mark_id, "--root", str(marked_route.parent), "--no-sidecar"]) == 0
  out = capsys.readouterr().out
  assert "sidecar skipped" in out
  assert "src=" in out


def test_mark_cli_show_unknown_mark(marked_route: Path) -> None:
  with pytest.raises(SystemExit, match="no driver mark matches"):
    mark_cli.main(["show", "nope", "--root", str(marked_route.parent)])


# ---------------------------------------------------------------------------
# episode_key uniqueness
# ---------------------------------------------------------------------------


def _colliding_route(tmp_path: Path) -> Path:
  route_root = tmp_path / "route_collide"
  # Two presses 0.8 s apart, plus a RadarD hole 20 s earlier. The hole makes
  # both -20 s window starts clamp to the SAME surviving frame; both +6 s window
  # ends run past the last recorded frame and clamp to it too. Result: identical
  # (t_start_s, t_end_s) for two genuinely different presses.
  _write_marked_route(
    route_root,
    press_frames=(_COLLIDING_PRESS_FRAME_A, _COLLIDING_PRESS_FRAME_B),
    drop_car_state_frames=_COLLIDING_DROPPED_FRAMES,
    include_orphan_press=False,
  )
  return route_root


def test_two_presses_sharing_a_window_still_get_distinct_episode_keys(tmp_path: Path) -> None:
  route_root = _colliding_route(tmp_path)
  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [route_root])
    route_row = get_route_rows(conn)[0]
    with pytest.warns(RuntimeWarning, match=f"Dropped {len(_COLLIDING_DROPPED_FRAMES)}/"):
      scan = load_route_scan(conn, route_row)
    candidates = sorted(
      (c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"),
      key=lambda c: c.event_t_s,
    )
    assert len(candidates) == 2, "both presses must survive suppression; they are 0.8 s apart"

    # The collision this guards against, spelled out: same window, different press.
    assert candidates[0].t_start_s == candidates[1].t_start_s
    assert candidates[0].t_end_s == candidates[1].t_end_s
    assert candidates[0].event_t_s != candidates[1].event_t_s
    press_times = [c.notes_json["driverMarkPressLogMonoTime"] for c in candidates]
    assert press_times == [_press_ns(_COLLIDING_PRESS_FRAME_A), _press_ns(_COLLIDING_PRESS_FRAME_B)]

    # Without the press-time discriminator these two keys are byte-identical and
    # the second upsert_episode silently overwrites the first press's row.
    assert len({c.episode_key for c in candidates}) == 2
    for candidate, press_ns in zip(candidates, press_times, strict=True):
      assert candidate.episode_key.endswith(f":press{press_ns}")
  finally:
    conn.close()


def test_two_presses_sharing_a_window_both_reach_the_catalog(tmp_path: Path) -> None:
  route_root = _colliding_route(tmp_path)
  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [route_root])
    with pytest.warns(RuntimeWarning, match=f"Dropped {len(_COLLIDING_DROPPED_FRAMES)}/"):
      recorded = extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    marks = [entry for entry in recorded if entry["episodeType"] == "driver_mark"]
    assert len(marks) == 2

    rows = conn.execute(
      "SELECT episode_key, notes_json FROM episodes WHERE episode_type = 'driver_mark'"
    ).fetchall()
    assert len(rows) == 2, "a lost row here means one human press was overwritten by the other"
    assert {json.loads(row["notes_json"])["driverMarkPressLogMonoTime"] for row in rows} == {
      _press_ns(_COLLIDING_PRESS_FRAME_A), _press_ns(_COLLIDING_PRESS_FRAME_B),
    }
  finally:
    conn.close()


def _same_anchor_route(tmp_path: Path) -> Path:
  route_root = tmp_path / "route_same_anchor"
  _write_marked_route(
    route_root,
    press_frames=(_SAME_ANCHOR_PRESS_FRAME_A, _SAME_ANCHOR_PRESS_FRAME_B),
    drop_car_state_frames=_SAME_ANCHOR_DROPPED_FRAMES,
    include_orphan_press=False,
  )
  return route_root


def test_two_presses_anchored_to_one_frame_both_survive_dedupe(tmp_path: Path) -> None:
  """A RadarD hole must not let _dedupe_candidates eat the second press.

  ``_should_suppress_candidate`` used to compare ``event_t_s``, which for a driver
  mark is the anchored radarState frame's route clock -- NOT the press clock. With
  frames missing on both sides of one survivor, two presses 1.4 s apart both
  anchor to that survivor, get the same ``event_t_s``, and the second is dropped
  before it can reach the catalog. Its distinct ``episode_key`` never helps: the
  candidate is gone by then.
  """
  route_root = _same_anchor_route(tmp_path)
  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [route_root])
    route_row = get_route_rows(conn)[0]
    with pytest.warns(RuntimeWarning, match=f"Dropped {len(_SAME_ANCHOR_DROPPED_FRAMES)}/"):
      scan = load_route_scan(conn, route_row)

    # Both presses were resolved and both anchored -- the loss is purely in dedupe.
    press_times = [_press_ns(_SAME_ANCHOR_PRESS_FRAME_A), _press_ns(_SAME_ANCHOR_PRESS_FRAME_B)]
    assert [mark.press_log_mono_time_ns for mark in scan.driver_marks] == press_times
    assert all(mark.frame_index is not None for mark in scan.driver_marks)
    assert len({mark.frame_index for mark in scan.driver_marks}) == 1, "the setup requires ONE shared anchor frame"
    assert all(mark.frame_gap_s <= DRIVER_MARK_FRAME_TOLERANCE_S for mark in scan.driver_marks)
    # 1.4 s apart on the press clock: past the recorder's 1.0 s debounce, so two
    # deliberate incidents by the device's own definition.
    assert (press_times[1] - press_times[0]) / 1e9 == pytest.approx(1.4, abs=1e-6)

    candidates = [c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"]
    assert len(candidates) == 2, "the second deliberate press was swallowed by candidate dedupe"
    assert candidates[0].event_t_s == candidates[1].event_t_s, "the setup requires a colliding event clock"
    assert [c.notes_json["driverMarkPressLogMonoTime"] for c in candidates] == press_times
    assert len({c.episode_key for c in candidates}) == 2

    with pytest.warns(RuntimeWarning, match=f"Dropped {len(_SAME_ANCHOR_DROPPED_FRAMES)}/"):
      recorded = extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    marks = [entry for entry in recorded if entry["episodeType"] == "driver_mark"]
    assert len(marks) == 2
    rows = conn.execute(
      "SELECT notes_json FROM episodes WHERE episode_type = 'driver_mark'"
    ).fetchall()
    assert {json.loads(row["notes_json"])["driverMarkPressLogMonoTime"] for row in rows} == set(press_times)
  finally:
    conn.close()


def test_heuristic_episode_keys_keep_their_shape() -> None:
  # The discriminator is opt-in: heuristic detectors must not have their keys
  # (and therefore their catalog identity across re-extractions) changed.
  plain = _bare_candidate("approach", event_t_s=20.0, t_start_s=18.0, t_end_s=25.0)
  assert plain.episode_key == f"route1:route:approach:18000:25000:{EXTRACTOR_VERSION}"


def test_reextracting_a_route_rebuilds_driver_mark_rows(tmp_path: Path, marked_route: Path, monkeypatch) -> None:
  """Extract twice; a degraded second pass must not leave a good pointer behind.

  ``extract_ev6_episodes`` calls ``clear_route_extractions`` before every route,
  so each pass rebuilds the route's rows from scratch rather than upserting over
  the previous pass's. This test pins that: a successful pass writes a bundle
  pointer, a degraded pass leaves a bundle-less row (never a row pointing at a
  bundle it did not write), and a third successful pass restores the pointer.
  """
  from openpilot.selfdrive.test.longitudinal_harness import route_extract

  def _bundle_paths(conn) -> dict[str, Any]:
    return {
      row["episode_key"]: row["bundle_path"]
      for row in conn.execute("SELECT episode_key, bundle_path FROM episodes WHERE episode_type = 'driver_mark'")
    }

  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [marked_route])
    extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    first = _bundle_paths(conn)
    assert first, "the first pass must catalog the driver marks"

    def _refuse(*_args, **_kwargs):
      raise EpisodeNotReplayableError("synthetic dependency gap")

    monkeypatch.setattr(route_extract, "write_episode_bundle", _refuse)
    extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    degraded = _bundle_paths(conn)
    assert set(degraded) == set(first), "no press may disappear on a degraded re-extract"
    assert all(path is None for path in degraded.values())

    monkeypatch.undo()
    extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    assert _bundle_paths(conn) == first
  finally:
    conn.close()


# ---------------------------------------------------------------------------
# phantom presses from a bare userBookmark
# ---------------------------------------------------------------------------


def test_is_driver_mark_press_group_requires_the_button() -> None:
  assert is_driver_mark_press_group({"services": ["bookmarkButton"]})
  assert is_driver_mark_press_group({"services": ["bookmarkButton", "userBookmark"]})
  # feedbackd's LKAS path and its ~10 s-later audio-completion send.
  assert not is_driver_mark_press_group({"services": ["userBookmark"]})
  assert not is_driver_mark_press_group({})


def test_lone_user_bookmark_is_not_a_driver_mark(tmp_path: Path) -> None:
  route_root = tmp_path / "route_lkas"
  # Frame 100 is the LKAS-button bookmark (feedbackd:26-39, taken when MADS is
  # unavailable); frame 300 is the RecordAudioFeedback completion send
  # (feedbackd:54-55), 10 s after its own LKAS press. Neither has a
  # bookmarkButton. Frame 400 is the real flag press.
  _write_marked_route(
    route_root,
    include_orphan_press=False,
    include_second_press=False,
    lone_user_bookmark_frames=(100, 300),
  )
  conn, scan = _load_scan(tmp_path, route_root)
  try:
    assert [mark.press_log_mono_time_ns for mark in scan.driver_marks] == [_press_ns(_FIRST_PRESS_FRAME)]
    candidates = [c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"]
    assert len(candidates) == 1
    assert candidates[0].notes_json["driverMarkPressLogMonoTime"] == _press_ns(_FIRST_PRESS_FRAME)
  finally:
    conn.close()

  # Same rule on the marks.py path, which scans a single log directly.
  segment_marks = scan_segment_marks(route_root / "0" / "rlog.zst", route_key="route_lkas", seg_idx=0)
  assert [mark.press_log_mono_time_ns for mark in segment_marks] == [_press_ns(_FIRST_PRESS_FRAME)]
  assert segment_marks[0].services == ("bookmarkButton", "userBookmark")


def test_echo_collapse_anchors_on_the_button_not_an_earlier_user_bookmark() -> None:
  """The group's press time must be the bookmarkButton's, not the first message's.

  ``feedbackd`` publishes bare ``userBookmark`` on its LKAS and audio-completion
  paths, so one can land shortly BEFORE a real tap and fall inside the same 0.5 s
  echo window. Phase 2's recorder stamps the sidecar's ``pressLogMonoTime`` from
  ``sm.logMonoTime['bookmarkButton']``; if the offline half anchors on the stray
  echo instead, the two clocks disagree and ``marks.find_mark_sidecar`` misses --
  the planner-internal detail for a real press silently disappears.
  """
  groups = collapse_driver_mark_echoes([
    {"segIdx": 3, "logMonoTimeNs": 1_000_000_000, "service": "userBookmark"},
    {"segIdx": 4, "logMonoTimeNs": 1_300_000_000, "service": "bookmarkButton"},
    {"segIdx": 4, "logMonoTimeNs": 1_350_000_000, "service": "userBookmark"},
  ])
  assert len(groups) == 1
  assert is_driver_mark_press_group(groups[0])
  assert groups[0]["logMonoTimeNs"] == 1_300_000_000
  assert groups[0]["segIdx"] == 4
  assert groups[0]["services"] == ["userBookmark", "bookmarkButton"]
  # The returned shape is unchanged; no bookkeeping key leaks to callers.
  assert set(groups[0]) == {"segIdx", "logMonoTimeNs", "services"}


def test_echo_collapse_membership_still_keys_off_the_first_message() -> None:
  # Re-anchoring must not slide the 0.5 s membership window forward and swallow a
  # later, separate press. Group starts at 1.000 s, so 1.550 s is outside it even
  # though it is only 0.250 s past the re-anchored bookmarkButton at 1.300 s.
  groups = collapse_driver_mark_echoes([
    {"segIdx": 0, "logMonoTimeNs": 1_000_000_000, "service": "userBookmark"},
    {"segIdx": 0, "logMonoTimeNs": 1_300_000_000, "service": "bookmarkButton"},
    {"segIdx": 0, "logMonoTimeNs": 1_550_000_000, "service": "bookmarkButton"},
  ])
  assert [group["logMonoTimeNs"] for group in groups] == [1_300_000_000, 1_550_000_000]
  # A group that never sees a bookmarkButton keeps its own first message and is
  # still rejected by is_driver_mark_press_group.
  lone = collapse_driver_mark_echoes([{"segIdx": 0, "logMonoTimeNs": 5, "service": "userBookmark"}])
  assert lone == [{"segIdx": 0, "logMonoTimeNs": 5, "services": ["userBookmark"]}]
  assert not is_driver_mark_press_group(lone[0])


def test_stray_user_bookmark_before_a_press_does_not_move_the_press_time(tmp_path: Path) -> None:
  route_root = tmp_path / "route_early_echo"
  _write_marked_route(
    route_root,
    include_orphan_press=False,
    include_second_press=False,
    lone_user_bookmark_frames=(_EARLY_ECHO_FRAME,),
  )
  conn, scan = _load_scan(tmp_path, route_root)
  try:
    assert [mark.press_log_mono_time_ns for mark in scan.driver_marks] == [_press_ns(_FIRST_PRESS_FRAME)]
    assert scan.driver_marks[0].press_log_mono_time_ns != _press_ns(_EARLY_ECHO_FRAME)
    candidates = [c for c in detect_episode_candidates(scan) if c.episode_type == "driver_mark"]
    assert [c.notes_json["driverMarkPressLogMonoTime"] for c in candidates] == [_press_ns(_FIRST_PRESS_FRAME)]
  finally:
    conn.close()

  # The marks.py path resolves the same press time, so mark_id -- and therefore
  # the sidecar join in mark_cli show -- agrees with the device.
  segment_marks = scan_segment_marks(route_root / "0" / "rlog.zst", route_key="route_early_echo", seg_idx=0)
  assert [mark.press_log_mono_time_ns for mark in segment_marks] == [_press_ns(_FIRST_PRESS_FRAME)]
  assert segment_marks[0].payload == {"source": "sidebar", "seq": 1}


# ---------------------------------------------------------------------------
# orphan presses
# ---------------------------------------------------------------------------


def test_orphan_press_is_reported_and_catalogued(tmp_path: Path, marked_route: Path) -> None:
  conn = open_catalog(tmp_path / "catalog.db")
  try:
    index_ev6_routes(conn, [marked_route])
    recorded = extract_ev6_episodes(conn, bundle_root=tmp_path / "bundles")
    orphans = [entry for entry in recorded if entry["status"] == "orphan_recorded_without_bundle"]
    assert len(orphans) == 1, "an unanchorable human press must still be reported"
    assert orphans[0]["driverMarkPressLogMonoTime"] == _ORPHAN_PRESS_NS
    assert "could not be anchored" in orphans[0]["reason"]

    row = conn.execute(
      "SELECT * FROM episodes WHERE episode_key = ?", (orphans[0]["episodeKey"],)
    ).fetchone()
    assert row is not None, "the orphan must be recoverable from the catalog, not just the report"
    assert row["bundle_path"] is None
    notes = json.loads(row["notes_json"])
    assert notes["driverMarkOrphan"] is True
    # The press logMonoTime is the only join key that survives a missing frame:
    # marks.load_trace and the Phase 2 sidecar both key on it.
    assert notes["driverMarkPressLogMonoTime"] == _ORPHAN_PRESS_NS
    assert notes["bundleUnavailableReason"]
  finally:
    conn.close()


# ---------------------------------------------------------------------------
# segment-boundary windows
# ---------------------------------------------------------------------------


def _write_trace_segment(seg_dir: Path, frame_indices: range, *, press_frames: tuple[int, ...] = ()) -> Path:
  """A minimal segment rlog: enough services for load_trace, nothing else."""
  seg_dir.mkdir(parents=True, exist_ok=True)
  rlog_path = seg_dir / "rlog.zst"
  messages: list[Any] = []
  for frame_idx in frame_indices:
    frame_base = _frame_base_ns(frame_idx)

    car_state = messaging.new_message("carState")
    car_state.logMonoTime = frame_base
    car_state.valid = True
    car_state.carState.vEgo = _EGO_SPEED_MPS
    messages.append(car_state.as_reader())

    car_control = messaging.new_message("carControl")
    car_control.logMonoTime = frame_base + 6_000_000
    car_control.valid = True
    car_control.carControl.longActive = True
    messages.append(car_control.as_reader())

    radar_state = messaging.new_message("radarState")
    radar_state.logMonoTime = frame_base + 15_000_000
    radar_state.valid = True
    lead = radar_state.radarState.leadOne
    lead.status = True
    lead.dRel = _LEAD_GAP_M
    lead.vRel = -1.0
    lead.modelProb = 0.9
    messages.append(radar_state.as_reader())

    plan = messaging.new_message("longitudinalPlan")
    plan.logMonoTime = frame_base + 20_000_000
    plan.valid = True
    plan.longitudinalPlan.modelMonoTime = frame_base + 10_000_000
    plan.longitudinalPlan.aTarget = -0.5
    plan.longitudinalPlan.longitudinalPlanSource = "cruise"
    messages.append(plan.as_reader())

    if frame_idx in press_frames:
      button = messaging.new_message("bookmarkButton")
      button.logMonoTime = _press_ns(frame_idx)
      button.valid = True
      messages.append(button.as_reader())

  save_log(str(rlog_path), messages)
  return rlog_path


@pytest.fixture
def two_segment_route(tmp_path: Path) -> Path:
  route_root = tmp_path / "route_segmented"
  _write_trace_segment(route_root / "0", range(_SEG_FRAME_COUNT), press_frames=(10,))
  _write_trace_segment(
    route_root / "1",
    range(_SEG_FRAME_COUNT, 2 * _SEG_FRAME_COUNT),
    press_frames=(_BOUNDARY_PRESS_FRAME,),
  )
  return route_root


def test_resolve_segment_rlog_chain_finds_neighbours(two_segment_route: Path) -> None:
  chain = resolve_segment_rlog_chain(two_segment_route / "1" / "rlog.zst")
  assert set(chain) == {0, 1}
  assert chain[0] == two_segment_route / "0" / "rlog.zst"
  assert resolve_segment_rlog_chain(two_segment_route / "0" / "rlog.zst").keys() == {0, 1}


def test_load_trace_reads_back_across_the_segment_boundary(two_segment_route: Path) -> None:
  press_ns = _press_ns(_BOUNDARY_PRESS_FRAME)
  window = load_trace_window(two_segment_route / "1" / "rlog.zst", press_ns)
  assert window.rows

  # The press is 0.5 s into its own segment. Reading only that segment yields a
  # 0.5 s pre-roll -- the entire brake build-up the -20 s window exists to
  # capture is in the previous segment's rlog.
  assert min(row["t_rel_s"] for row in window.rows) == pytest.approx(-20.0, abs=0.1)
  assert max(row["t_rel_s"] for row in window.rows) == pytest.approx(6.0, abs=0.1)
  assert window.segments_read == (0, 1)
  assert window.missing_segments == ()
  assert window.truncated is False
  assert window.covered_pre_s == pytest.approx(20.0, abs=0.05)

  metrics = compute_mark_metrics(window.rows, window=window)
  assert metrics["windowTruncated"] is False
  assert metrics["windowCoverage"]["segmentsRead"] == [0, 1]

  # And the plain-list entry point still behaves like the window's rows.
  assert load_trace(two_segment_route / "1" / "rlog.zst", press_ns) == window.rows


def test_load_trace_window_declares_a_missing_neighbour(two_segment_route: Path) -> None:
  # A press 0.5 s into segment 0: there is no segment -1 anywhere, so the -20 s
  # pre-roll genuinely does not exist and the window must say so.
  window = load_trace_window(two_segment_route / "0" / "rlog.zst", _press_ns(10))
  assert window.rows
  assert window.truncated is True
  assert window.truncated_start is True
  assert window.truncated_end is False
  assert window.missing_segments == (-1,)
  assert window.covered_pre_s == pytest.approx(0.5, abs=0.1)
  assert window.covered_post_s == pytest.approx(6.0, abs=0.05)

  metrics = compute_mark_metrics(window.rows, window=window)
  assert metrics["windowTruncated"] is True
  assert metrics["windowCoverage"]["truncatedEdges"] == ["start"]
  assert metrics["windowCoverage"]["requestedPreS"] == pytest.approx(20.0)

  # Coverage is unknowable from bare rows, so it is reported as unknown rather
  # than as "fine".
  assert compute_mark_metrics(window.rows)["windowTruncated"] is None


def test_mark_cli_show_flags_a_truncated_window(two_segment_route: Path, capsys) -> None:
  mark_id = f"route_segmented--0--{_press_ns(10)}"
  assert mark_cli.main(["show", mark_id, "--root", str(two_segment_route.parent), "--no-sidecar"]) == 0
  out = capsys.readouterr().out
  assert "window TRUNCATED at start" in out
  assert "missing neighbour segments: -1" in out
  assert "WINDOW TRUNCATED, actually covers" in out
  assert '"windowTruncated": true' in out


def test_mark_cli_show_reports_a_full_window(two_segment_route: Path, capsys) -> None:
  mark_id = f"route_segmented--1--{_press_ns(_BOUNDARY_PRESS_FRAME)}"
  assert mark_cli.main(["show", mark_id, "--root", str(two_segment_route.parent), "--no-sidecar"]) == 0
  out = capsys.readouterr().out
  assert "window full" in out
  assert "TRUNCATED" not in out


# ---------------------------------------------------------------------------
# sidecar roots
# ---------------------------------------------------------------------------


def test_sidecar_default_root_matches_the_device_directory() -> None:
  # longitudinal_mark_recorder.MARKS_DIR_DEFAULT is /data/media/0/LongMarks, so
  # `adb pull` lands a directory literally named LongMarks. On a case-sensitive
  # filesystem a snake_case default never finds it and `mark_cli show` degrades
  # to "sidecar none" with no explanation.
  assert MARK_SIDECAR_DEVICE_DIR == "/data/media/0/LongMarks"
  device_dir_name = MARK_SIDECAR_DEVICE_DIR.rsplit("/", 1)[-1]
  assert DEFAULT_MARK_SIDECAR_ROOTS[0] == Path(".cache") / device_dir_name
  assert MARK_SIDECAR_DEVICE_DIR in MARK_SIDECAR_PULL_COMMAND
  assert ".cache/" in MARK_SIDECAR_PULL_COMMAND


def test_mark_cli_show_names_the_pull_command_when_no_sidecar(marked_route: Path, capsys) -> None:
  mark_id = f"route_marked--0--{_press_ns(_FIRST_PRESS_FRAME)}"
  assert mark_cli.main([
    "show", mark_id, "--root", str(marked_route.parent), "--sidecar-root", str(marked_route / "nope"),
  ]) == 0
  out = capsys.readouterr().out
  assert "sidecar none" in out
  assert MARK_SIDECAR_PULL_COMMAND in out


# ---------------------------------------------------------------------------
# the real on-device LONGFLAG record shape
# ---------------------------------------------------------------------------


def _button_source() -> str:
  return (_REPO_ROOT / _BUTTONS_CC_RELPATH).read_text(encoding="utf-8")


def _longflag_snprintf_format() -> str:
  """The C++ format literal, unescaped.

  Mirrors LongitudinalFlagButtonSP::fire() in
  selfdrive/ui/sunnypilot/qt/onroad/buttons.cc:

    snprintf(buf, sizeof(buf), "{\\"src\\":\\"hud\\",\\"seq\\":%u,\\"monoNs\\":%llu,\\"preS\\":%.1f,\\"postS\\":%.1f}",
             seq, (unsigned long long)now_ns, kFlagRetroPreS, kFlagRetroPostS);
    LOG("LONGFLAG %s", buf);
  """
  source = _button_source()
  match = re.search(r'snprintf\(buf,\s*sizeof\(buf\),\s*"((?:[^"\\]|\\.)*)"', source)
  assert match is not None, "the LONGFLAG snprintf format literal moved; update this test with it"
  return match.group(1).replace('\\"', '"')


def _real_swaglog_record(msg: str) -> str:
  """A byte-faithful cloudlog record, as logmessaged publishes it.

  cloudlog_common() in common/swaglog.cc builds exactly these keys
  (ctx / levelnum / filename / lineno / funcname / created, plus msg), dumps them
  to JSON and prefixes ONE level byte. system/logmessaged.py strips that byte
  (``record = dat[1:]``) and publishes the remainder verbatim as
  ``logMessage``, which is what _parse_longflag is handed out of the rlog.
  """
  return json.dumps({
    "ctx": {
      "daemon": "ui",
      "device": "tici",
      "dirty": True,
      "version": "0.10.0",
    },
    "levelnum": 20,  # CLOUDLOG_INFO; LOG() in common/swaglog.h
    "filename": _BUTTONS_CC_RELPATH,
    "lineno": 143,
    "funcname": "fire",
    "created": 1753372800.1234567,
    "msg": msg,
  })


def test_longflag_parses_a_real_swaglog_record() -> None:
  fmt = _longflag_snprintf_format()
  # %u -> int, %llu -> int, %.1f -> one decimal place. Rendered exactly as the
  # device would render it, then wrapped the way LOG() + cloudlog_common() do.
  buf = fmt.replace("%u", "7").replace("%llu", "123456789012")
  # Two %.1f in order: kFlagRetroPreS then kFlagRetroPostS. str.format is not
  # usable here -- the literal is JSON and is full of braces.
  buf = buf.replace("%.1f", f"{20.0:.1f}", 1).replace("%.1f", f"{6.0:.1f}", 1)
  record = _real_swaglog_record(f"LONGFLAG {buf}")

  payload = _parse_longflag(record)
  assert payload == {"src": "hud", "seq": 7, "monoNs": 123456789012, "preS": 20.0, "postS": 6.0}


def test_longflag_payload_keys_match_the_button_source() -> None:
  """Tripwire: a C++-side payload rename must break a Python test.

  Every other LONGFLAG test in this file fabricates its own payload, so the
  offline side could drift from the button indefinitely. This one reads the
  actual format literal out of buttons.cc.
  """
  source = _button_source()
  assert 'LOG("LONGFLAG %s", buf)' in source, "the button no longer emits a LONGFLAG cloudlog line"
  fmt = _longflag_snprintf_format()
  assert set(re.findall(r'"(\w+)":', fmt)) == {"src", "seq", "monoNs", "preS", "postS"}
  # The advertised retro window must stay the one the extractor actually cuts.
  pre_s, post_s = WINDOWS_BY_TYPE["driver_mark"]
  assert "constexpr double kFlagRetroPreS = 20.0;" in _repo_text("selfdrive/ui/sunnypilot/qt/onroad/buttons.h")
  assert "constexpr double kFlagRetroPostS = 6.0;" in _repo_text("selfdrive/ui/sunnypilot/qt/onroad/buttons.h")
  assert (pre_s, post_s) == (-20.0, 6.0)


def _repo_text(relpath: str) -> str:
  return (_REPO_ROOT / relpath).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the Phase 2 recorder's capture window
# ---------------------------------------------------------------------------


def _recorder_source_float(name: str) -> float:
  """One module-level float constant, read literally out of the recorder source.

  Read from source rather than imported for the same reason the buttons.h pin is:
  it cannot be skipped. sunnypilot.selfdrive.controls.lib.longitudinal_mark_recorder
  imports openpilot.common.params and openpilot.system.hardware, so an import here
  would make this tripwire evaporate (collection error or skip) on exactly the
  machines that have no built native extensions -- which is where a drifting
  constant would go unnoticed longest. The import IS still checked below, as a
  bonus, wherever it happens to work.
  """
  source = _repo_text(_RECORDER_RELPATH)
  match = re.search(rf"^{name} = ([0-9]+\.[0-9]+)", source, re.MULTILINE)
  assert match is not None, f"{name} moved in {_RECORDER_RELPATH}; update this test with it"
  return float(match.group(1))


def test_recorder_capture_window_matches_the_extractor_window() -> None:
  """Tripwire: the device's capture window and the offline cut must stay equal.

  ``longitudinal_mark_recorder.PRE_WINDOW_S``/``POST_WINDOW_S`` decide how much
  planner-internal state the device writes into a sidecar;
  ``WINDOWS_BY_TYPE["driver_mark"]`` decides how much log the extractor cuts
  around the same press. Nothing links them, so changing the recorder to 30/8
  leaves every test green while the sidecar silently covers a different span than
  the trace it is joined onto -- rows outside the trace are dropped on the floor
  and the sidecar's own header advertises a window nobody else has.

  With this test, buttons.h (the advertised retro window),
  WINDOWS_BY_TYPE["driver_mark"] (the offline cut) and the recorder (the device
  capture) are all pinned to one another; see
  test_longflag_payload_keys_match_the_button_source for the C++ half.
  """
  pre_s, post_s = WINDOWS_BY_TYPE["driver_mark"]
  assert _recorder_source_float("PRE_WINDOW_S") == abs(pre_s)
  assert _recorder_source_float("POST_WINDOW_S") == post_s

  try:
    from openpilot.sunnypilot.selfdrive.controls.lib import longitudinal_mark_recorder
  except ImportError:  # pragma: no cover - only on a tree without built natives
    return
  assert longitudinal_mark_recorder.PRE_WINDOW_S == abs(pre_s)
  assert longitudinal_mark_recorder.POST_WINDOW_S == post_s
  # marks.PRE_WINDOW_S/POST_WINDOW_S are derived from WINDOWS_BY_TYPE, so they
  # come along for free -- assert it rather than assume it.
  assert (PRE_WINDOW_S, POST_WINDOW_S) == (pre_s, post_s)
