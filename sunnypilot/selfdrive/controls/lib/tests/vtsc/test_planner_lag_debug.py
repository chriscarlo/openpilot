import json

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import planner_lag_debug as dbg


class _FakeParams:
  def __init__(self, *, enabled: bool, route: str = ""):
    self.enabled = enabled
    self.route = route

  def get_bool(self, key: str) -> bool:
    assert key == dbg.ENABLE_PARAM
    return bool(self.enabled)

  def get(self, key: str):
    if key == "CurrentRoute":
      return self.route
    return None


class _FakeClock:
  def __init__(self, now_s: float):
    self.now_s = float(now_s)

  def now(self) -> float:
    return float(self.now_s)

  def advance(self, dt_s: float) -> None:
    self.now_s += float(dt_s)


class _InlineThread:
  def __init__(self, *, target, args=(), kwargs=None, **_ignored):
    self._target = target
    self._args = args
    self._kwargs = kwargs or {}

  def start(self) -> None:
    self._target(*self._args, **self._kwargs)


def test_summarize_window_ranks_suspect_split():
  cycle_a = {
    "frame": 101,
    "is_bad_cycle": True,
    "planner_publish_gap_ms": 63.0,
    "planner_loop_dt_ms": 58.0,
    dbg.SPAN_MAP_TAIL_CAP: 17.0,
    dbg.SPAN_MAP_CAP_STRATEGIC: 15.0,
    dbg.SPAN_HELPER_PREDICT: 6.0,
    dbg.SPAN_HELPER_CRUISE_CAP: 3.0,
    dbg.SPAN_PREVIEW_FROM_MAP: 4.0,
    dbg.SPAN_PREVIEW_BRANCH_STUBS: 1.0,
    dbg.SPAN_VTSC_UPDATE: 18.0,
    dbg.SPAN_UPDATE_V_CRUISE_TOTAL: 46.0,
    dbg.SPAN_PLANNER_UPDATE_TOTAL: 51.0,
    dbg.SPAN_MPC_UPDATE: 4.0,
    dbg.SPAN_PUBLISH_LONGITUDINAL_PLAN_SP: 3.0,
    dbg.SPAN_PREVIEW_ENCODE: 1.0,
    dbg.SPAN_DRIVER_ASSISTANCE_PUBLISH: 1.0,
    "curve_preview_points": 40,
  }
  cycle_b = {
    "frame": 102,
    "is_bad_cycle": True,
    "planner_publish_gap_ms": 60.0,
    "planner_loop_dt_ms": 56.0,
    dbg.SPAN_MAP_TAIL_CAP: 16.0,
    dbg.SPAN_MAP_CAP_STRATEGIC: 14.0,
    dbg.SPAN_HELPER_PREDICT: 5.0,
    dbg.SPAN_HELPER_CRUISE_CAP: 4.0,
    dbg.SPAN_PREVIEW_FROM_MAP: 3.0,
    dbg.SPAN_PREVIEW_BRANCH_STUBS: 2.0,
    dbg.SPAN_VTSC_UPDATE: 17.0,
    dbg.SPAN_UPDATE_V_CRUISE_TOTAL: 45.0,
    dbg.SPAN_PLANNER_UPDATE_TOTAL: 50.0,
    dbg.SPAN_MPC_UPDATE: 4.0,
    dbg.SPAN_PUBLISH_LONGITUDINAL_PLAN_SP: 3.0,
    dbg.SPAN_PREVIEW_ENCODE: 1.0,
    dbg.SPAN_DRIVER_ASSISTANCE_PUBLISH: 1.0,
    "curve_preview_points": 38,
  }

  summary = dbg.summarize_window([cycle_a, cycle_b], trigger_cycle=cycle_b)

  assert summary["dominant_component_bad_cycles"] == "helper_total_ms"
  assert summary["component_stats_bad_cycles_ms"]["helper_total_ms"]["mean_ms"] == pytest.approx(9.0)
  assert summary["component_stats_bad_cycles_ms"]["strategic_outside_helper_ms"]["mean_ms"] == pytest.approx(5.5)
  assert summary["component_stats_bad_cycles_ms"]["preview_total_ms"]["mean_ms"] == pytest.approx(5.0)
  assert summary["top_bad_cycles"][0]["curve_preview_points"] in (38, 40)


def test_recorder_dumps_event_bundle_on_low_cadence_streak(tmp_path, monkeypatch):
  monkeypatch.setattr(dbg.threading, "Thread", _InlineThread)
  monkeypatch.setattr(dbg.PlannerLagRecorder, "_guess_current_segment", staticmethod(lambda route: 90))
  monkeypatch.setattr(dbg, "STARTUP_GRACE_S", 0.0)

  clock = _FakeClock(dbg.STARTUP_GRACE_S + 1.0)
  params = _FakeParams(enabled=True, route="00000099--22d46c0f3d")
  recorder = dbg.PlannerLagRecorder(params=params, events_dir=tmp_path, time_fn=clock.now)

  for frame in range(dbg.LOW_CADENCE_STREAK_TRIGGER):
    assert recorder.begin_cycle(frame=frame, model_logmono_ns=frame * 50_000_000) is True
    recorder.record_span_ns(dbg.SPAN_MAP_TAIL_CAP, int(17e6))
    recorder.record_span_ns(dbg.SPAN_MAP_CAP_STRATEGIC, int(12e6))
    recorder.record_span_ns(dbg.SPAN_HELPER_PREDICT, int(6e6))
    recorder.record_span_ns(dbg.SPAN_HELPER_CRUISE_CAP, int(3e6))
    recorder.record_span_ns(dbg.SPAN_PREVIEW_FROM_MAP, int(4e6))
    recorder.record_span_ns(dbg.SPAN_PREVIEW_BRANCH_STUBS, int(1e6))
    recorder.record_span_ns(dbg.SPAN_VTSC_UPDATE, int(18e6))
    recorder.record_span_ns(dbg.SPAN_UPDATE_V_CRUISE_TOTAL, int(46e6))
    recorder.record_span_ns(dbg.SPAN_PLANNER_UPDATE_TOTAL, int(51e6))
    recorder.record_span_ns(dbg.SPAN_MPC_UPDATE, int(4e6))
    recorder.record_span_ns(dbg.SPAN_PUBLISH_LONGITUDINAL_PLAN_SP, int(3e6))
    recorder.record_span_ns(dbg.SPAN_PREVIEW_ENCODE, int(1e6))
    recorder.record_span_ns(dbg.SPAN_DRIVER_ASSISTANCE_PUBLISH, int(1e6))
    recorder.record_fields(
      strategy_mode="strategic",
      strategy_state="idle",
      curve_preview_points=36,
      curve_preview_branch_stubs=0,
      vtsc_state_name="disabled",
      map_geometry_valid=False,
      nearby_segment_count=0,
    )
    clock.advance(0.058)
    cycle = recorder.finish_cycle(planner_loop_dt_s=0.057, publish_end_s=clock.now())
    assert cycle is not None

  event_dirs = list(tmp_path.iterdir())
  assert len(event_dirs) == 1

  event_dir = event_dirs[0]
  summary = json.loads((event_dir / "summary.json").read_text(encoding="utf-8"))
  trigger = json.loads((event_dir / "trigger_cycle.json").read_text(encoding="utf-8"))
  trace_lines = (event_dir / "trace_20s.jsonl").read_text(encoding="utf-8").strip().splitlines()

  assert summary["bad_cycles"] >= dbg.LOW_CADENCE_STREAK_TRIGGER - 1
  assert summary["dominant_component_bad_cycles"] == "helper_total_ms"
  assert "low_cadence_streak" in trigger["lag_reasons"]
  assert trigger["curve_preview_points"] == 36
  assert len(trace_lines) == dbg.LOW_CADENCE_STREAK_TRIGGER
