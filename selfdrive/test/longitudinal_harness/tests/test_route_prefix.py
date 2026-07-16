from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from openpilot.selfdrive.test.longitudinal_harness import route_prefix
from openpilot.selfdrive.test.longitudinal_harness.fidelity import FAIL, NOT_EVALUATED, PASS
from openpilot.selfdrive.test.longitudinal_harness.inputs import SnapshotBundle, StepInput
from openpilot.selfdrive.test.longitudinal_harness.route_prefix import (
  MarkerWindow,
  evaluate_recurrent_debug_convergence,
  evaluate_route_prefix_trace,
  load_marker_windows,
  run_route_prefix_convergence,
  select_marker_trace_rows,
)


_BASE_NS = 1_000_000_000_000
_SHORT_THRESHOLDS = {
  "radar": {
    "min_kinematic_samples": 1,
    "min_scorable_samples": 1,
    "min_scorable_fraction": 1.0,
    "min_contiguous_duration_s": 0.0,
  },
  "planner": {
    "min_scorable_samples": 1,
    "min_scorable_fraction": 1.0,
    "min_contiguous_duration_s": 0.0,
  },
}
_SHORT_RECURRENT = {
  "min_scorable_samples": 1,
  "min_scorable_fraction": 1.0,
  "min_contiguous_duration_s": 0.0,
}


def _marker(offset_s: float = 20.0, marker_id: str = "tap") -> MarkerWindow:
  return MarkerWindow(marker_id, _BASE_NS + int(offset_s * 1e9))


def _row(offset_s: float, *, debug_reference: bool = True, debug_value: int = 3) -> dict:
  timestamp = _BASE_NS + int(offset_s * 1e9)
  reference = {
    "logMonoTimeNs": timestamp,
    "plannerRadarResolution": "exact",
    "plannerContextStatus": "exact",
    "plannerStateInitializationProvenance": {
      "status": "missing",
      "version": 0,
      "appliedAtReplayStart": False,
    },
    "radardGateEligible": True,
    "radardServiceAssociationProvenance": {
      "contract": {"status": "exact", "version": 1},
      "modelV2": {"status": "exact"},
      "carState": {"status": "exact"},
      "liveTracks": {"status": "exact", "emptyPayloadValid": True},
    },
    "radarState": {
      "leadOne": {"status": True, "dRelM": 40.0, "vRelMps": -0.5},
      "leadTwo": {"status": False, "dRelM": 0.0, "vRelMps": 0.0},
    },
    "longitudinalPlan": {"aTargetMps2": -0.1, "source": "lead0"},
  }
  if debug_reference:
    reference["plannerRecurrentDebug"] = {"mpc_acc_source_debug": {"holdFrames": 3}}
  return {
    "t_s": offset_s,
    "replay_reference": reference,
    "planner_accel_mps2": -0.1,
    "planner_source": "lead0",
    "lead_one_published_status": True,
    "lead_one_published_d_rel_m": 40.0,
    "lead_one_published_v_rel_mps": -0.5,
    "lead_two_published_status": False,
    "lead_two_published_d_rel_m": None,
    "lead_two_published_v_rel_mps": None,
    "mpc_acc_source_debug": {"holdFrames": debug_value, "unrecordedExtra": True},
  }


def _trace(*, debug_reference: bool = True, debug_value: int = 3) -> list[dict]:
  offsets = [round(8.0 + (index * 0.05), 2) for index in range(237)]
  return [_row(value, debug_reference=debug_reference, debug_value=debug_value) for value in offsets]


def test_load_marker_windows_uses_generic_keys_and_qualification(tmp_path: Path) -> None:
  path = tmp_path / "markers.json"
  path.write_text(json.dumps({
    "events": [
      {"clock": 100, "use": False, "name": "skip"},
      {"clock": 20_000_000_000, "use": True, "name": "keep"},
    ],
  }))

  markers = load_marker_windows(
    path,
    collection_key="events",
    timestamp_key="clock",
    qualification_key="use",
    label_key="name",
  )

  assert [marker.marker_id for marker in markers] == ["keep"]
  assert markers[0].start_log_mono_time_ns == 10_000_000_000
  assert markers[0].end_log_mono_time_ns == 19_800_000_000


def test_select_marker_trace_rows_uses_reference_clock_and_distinct_preroll() -> None:
  rows = _trace()
  marker = _marker()

  window = select_marker_trace_rows(rows, marker)
  pre_roll = select_marker_trace_rows(rows, marker, pre_roll_s=2.0)

  assert window[0]["t_s"] == 10.0
  assert window[-1]["t_s"] == 19.8
  assert len(window) == 197
  assert pre_roll[0]["t_s"] == 8.0
  assert pre_roll[-1]["t_s"] == 9.95
  assert len(pre_roll) == 40


def test_route_prefix_keeps_formal_planner_not_evaluated_but_reports_diagnostics() -> None:
  report = evaluate_route_prefix_trace(
    _trace(),
    [_marker()],
    captured_metadata={},
    replay_metadata={},
    thresholds=_SHORT_THRESHOLDS,
    recurrent_thresholds=_SHORT_RECURRENT,
  )
  result = report["windows"][0]

  assert result["formalFidelity"]["planner"]["status"] == NOT_EVALUATED
  assert result["diagnosticFidelity"]["planner"]["status"] == PASS
  assert result["convergence"]["source"]["status"] == PASS
  assert result["convergence"]["aTarget"]["status"] == PASS
  assert result["convergence"]["recurrentDebug"]["status"] == PASS
  assert result["status"] == PASS
  assert report["gateEligible"] is False


def test_missing_recorded_recurrent_debug_is_not_evaluated_not_inferred() -> None:
  report = evaluate_route_prefix_trace(
    _trace(debug_reference=False),
    [_marker()],
    captured_metadata={},
    replay_metadata={},
    thresholds=_SHORT_THRESHOLDS,
    recurrent_thresholds=_SHORT_RECURRENT,
  )
  result = report["windows"][0]

  assert result["convergence"]["source"]["status"] == PASS
  assert result["convergence"]["aTarget"]["status"] == PASS
  assert result["convergence"]["recurrentDebug"]["status"] == NOT_EVALUATED
  assert result["status"] == NOT_EVALUATED
  assert "writer-v1" in result["convergence"]["recurrentDebug"]["reasons"][0]


def test_recorded_recurrent_debug_mismatch_fails() -> None:
  result = evaluate_recurrent_debug_convergence(
    _trace(debug_value=4),
    thresholds=_SHORT_RECURRENT,
  )
  assert result["status"] == FAIL
  assert result["agreementFraction"] == 0.0


def test_run_route_prefix_warms_once_for_multiple_markers(monkeypatch, tmp_path: Path) -> None:
  calls = []
  timeline = [
    StepInput(
      t_s=0.0,
      cruise_speed_mps=30.0,
      replay_reference={"logMonoTimeNs": _BASE_NS + int(offset_s * 1e9)},
    )
    for offset_s in (7.0, 19.8, 29.8)
  ]
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={},
    params={},
    timeline=timeline,
    initial_speed_mps=20.0,
    name="prefix",
  )
  trace = _trace() + [
    _row(value) for value in (20.0, 20.05, 20.1, 28.0, 28.05, 28.1, 29.8)
  ]
  monkeypatch.setattr(route_prefix, "resolve_fidelity_vehicle_config", lambda _bundle: object())

  def run(**kwargs):
    calls.append(kwargs)
    return SimpleNamespace(trace=trace, summary={"ok": True})

  monkeypatch.setattr(route_prefix, "run_harness", run)
  report = run_route_prefix_convergence(
    bundle,
    [_marker(), _marker(30.0, "second")],
    replay_metadata={},
    thresholds=_SHORT_THRESHOLDS,
    recurrent_thresholds=_SHORT_RECURRENT,
  )

  assert len(calls) == 1
  assert report["runCount"] == 1
  assert report["markerCount"] == 2


def test_incomplete_prefix_returns_not_evaluated_without_running(monkeypatch, tmp_path: Path) -> None:
  bundle = SnapshotBundle(
    path=tmp_path,
    vehicle={},
    params={},
    timeline=[StepInput(
      t_s=0.0,
      cruise_speed_mps=30.0,
      replay_reference={"logMonoTimeNs": _BASE_NS + int(19.0 * 1e9)},
    )],
    initial_speed_mps=20.0,
    name="short",
  )
  monkeypatch.setattr(route_prefix, "run_harness", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run")))

  report = run_route_prefix_convergence(bundle, [_marker()], replay_metadata={})

  assert report["status"] == NOT_EVALUATED
  assert report["runCount"] == 0
  assert "required clean pre-roll" in report["reason"]
  assert len(report["windows"]) == 1
  assert report["windows"][0]["convergence"]["source"]["status"] == NOT_EVALUATED
  assert report["windows"][0]["convergence"]["aTarget"]["status"] == NOT_EVALUATED
  assert report["windows"][0]["convergence"]["recurrentDebug"]["status"] == NOT_EVALUATED
