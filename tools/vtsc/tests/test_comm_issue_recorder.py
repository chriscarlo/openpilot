#!/usr/bin/env python3
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.comm_issue_recorder import evaluate_row, format_summary_report, summarize_trace_window


def _base_row(**overrides):
  row = {
    "t": 100.0,
    "route": "abc|2026-03-08--0",
    "started": True,
    "allSystemsReady": True,
    "subsystems": {"PLN": 2, "RAD": 2, "CAM": 2},
    "subsystemStatusNames": {"PLN": "GREEN", "RAD": "GREEN", "CAM": "GREEN"},
    "onroadEvents": [],
    "managerNotRunning": [],
    "managerProcesses": [],
    "serviceHealth": {
      "selfdriveStateSP": {"alive": True, "freq_ok": True, "valid": True},
      "onroadEvents": {"alive": True, "freq_ok": True, "valid": True},
      "managerState": {"alive": True, "freq_ok": True, "valid": True},
      "longitudinalPlan": {"alive": True, "freq_ok": True, "valid": True},
      "driverAssistance": {"alive": True, "freq_ok": True, "valid": True},
      "radarState": {"alive": True, "freq_ok": True, "valid": True},
    },
    "vEgo": 22.5,
  }
  row.update(overrides)
  return row


def test_evaluate_row_triggers_on_non_green_subsystem():
  row = evaluate_row(_base_row(subsystems={"PLN": 1, "RAD": 2, "CAM": 2}))

  assert row["degraded"] is True
  assert row["badSubsystems"] == ["PLN"]
  assert row["yellowSubsystems"] == ["PLN"]
  assert row["triggerReasons"] == ["subsystem_not_green"]


def test_evaluate_row_triggers_on_comm_event_and_manager_failure():
  row = evaluate_row(_base_row(
    onroadEvents=["commIssueAvgFreq"],
    managerNotRunning=["plannerd"],
    serviceHealth={
      "selfdriveStateSP": {"alive": True, "freq_ok": True, "valid": True},
      "onroadEvents": {"alive": True, "freq_ok": True, "valid": True},
      "managerState": {"alive": True, "freq_ok": True, "valid": True},
      "longitudinalPlan": {"alive": True, "freq_ok": False, "valid": True},
      "driverAssistance": {"alive": True, "freq_ok": False, "valid": True},
    },
  ))

  assert row["degraded"] is True
  assert row["commEvents"] == ["commIssueAvgFreq"]
  assert row["triggerManagerNotRunning"] == ["plannerd"]
  assert row["serviceNotFreqOk"] == ["driverAssistance", "longitudinalPlan"]
  assert row["triggerReasons"] == ["comm_event", "manager_process_down"]


def test_summarize_trace_window_and_report():
  rows = [
    evaluate_row(_base_row(
      t=100.0,
      subsystems={"PLN": 1, "RAD": 2, "CAM": 2},
      serviceHealth={
        "selfdriveStateSP": {"alive": True, "freq_ok": True, "valid": True},
        "onroadEvents": {"alive": True, "freq_ok": True, "valid": True},
        "managerState": {"alive": True, "freq_ok": True, "valid": True},
        "longitudinalPlan": {"alive": True, "freq_ok": False, "valid": True},
        "driverAssistance": {"alive": True, "freq_ok": False, "valid": True},
      },
    )),
    evaluate_row(_base_row(
      t=100.2,
      onroadEvents=["commIssueAvgFreq"],
      subsystems={"PLN": 0, "RAD": 2, "CAM": 2},
      serviceHealth={
        "selfdriveStateSP": {"alive": True, "freq_ok": True, "valid": True},
        "onroadEvents": {"alive": True, "freq_ok": True, "valid": True},
        "managerState": {"alive": True, "freq_ok": True, "valid": True},
        "longitudinalPlan": {"alive": True, "freq_ok": False, "valid": True},
        "driverAssistance": {"alive": True, "freq_ok": False, "valid": True},
      },
    )),
  ]

  summary = summarize_trace_window(rows)
  report = format_summary_report(
    meta={
      "event_id": "evt",
      "created_utc": "2026-03-08T12:00:00Z",
      "route": "abc",
      "seg_guess": 7,
      "t0_monotonic_s": 100.0,
    },
    summary=summary,
    trigger_row=rows[0],
  )

  assert summary["degraded_rows"] == 2
  assert summary["bad_subsystems"][0] == {"name": "PLN", "samples": 2}
  assert summary["service_not_freq_ok"][0] == {"name": "driverAssistance", "samples": 2}
  assert summary["service_not_freq_ok"][1] == {"name": "longitudinalPlan", "samples": 2}
  assert summary["comm_events"] == [{"name": "commIssueAvgFreq", "samples": 1}]
  assert "bad_subsystems_at_trigger: PLN" in report
  assert "service_not_freq_ok:" in report
