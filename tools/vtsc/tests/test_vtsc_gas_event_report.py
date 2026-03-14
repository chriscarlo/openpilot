#!/usr/bin/env python3
"""Tests for the gas-event RCA summary CLI."""
from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import vtsc.vtsc_gas_event_report as gas_event_report


def test_gas_event_report_writes_summary_tsv(tmp_path, capsys):
  base = tmp_path / "bundle"
  event_dir = base / "events_offline" / "gas_seg5_demo"
  event_dir.mkdir(parents=True)
  (event_dir / "event.json").write_text(json.dumps({
    "event_id": "gas_seg5_demo",
    "action": "gas",
    "route": "000000a4--demo",
    "seg": 5,
    "t0": 1478.4,
  }), encoding="utf-8")
  with (event_dir / "trace_rlog_20s_plus.jsonl").open("w", encoding="utf-8") as f:
    for row in [
      {"dt": -0.4, "vEgo": 11.9, "vtscVelMps": 9.8, "lowSpeedCalActive": True, "lowSpeedCalScale": 0.97, "lowSpeedCalReason": "tighten_tracking", "activeCap": "visible"},
      {"dt": 0.0, "vEgo": 11.8, "vtscVelMps": 9.6, "lowSpeedCalActive": True, "lowSpeedCalScale": 0.965, "lowSpeedCalReason": "tighten_tracking", "activeCap": "visible"},
      {"dt": 0.4, "vEgo": 11.7, "vtscVelMps": 9.7, "lowSpeedCalActive": True, "lowSpeedCalScale": 0.968, "lowSpeedCalReason": "tighten_tracking", "activeCap": "visible"},
    ]:
      f.write(json.dumps(row) + "\n")

  out_path = base / "gas_summary.tsv"
  rc = gas_event_report.main([str(base), "--out", str(out_path)])

  assert rc == 0
  rows = list(csv.DictReader(out_path.open(), delimiter="\t"))
  assert len(rows) == 1
  assert rows[0]["event_id"] == "gas_seg5_demo"
  assert rows[0]["constraint_label"] == "pressing_through_cap"
  assert rows[0]["calibration_label"] == "tighten_bias"

  stdout = capsys.readouterr().out
  assert "gas_seg5_demo" in stdout


def test_gas_event_report_falls_back_to_replay_samples_tsv(tmp_path, monkeypatch, capsys):
  base = tmp_path / "bundle"
  event_dir = base / "events_offline" / "gas_seg10_demo"
  realdata_dir = base / "realdata" / "000000a4--demo--10"
  event_dir.mkdir(parents=True)
  realdata_dir.mkdir(parents=True)
  (realdata_dir / "rlog.zst").write_bytes(b"placeholder")
  (event_dir / "event.json").write_text(json.dumps({
    "event_id": "gas_seg10_demo",
    "action": "gas",
    "route": "000000a4--demo",
    "seg": 10,
    "t0": 1769.465,
  }), encoding="utf-8")
  with (base / "gas_calibration_samples.tsv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
      "route", "segment", "t", "v_ego",
      "after_vtsc_cmd", "after_active_cap",
      "after_low_speed_calibration_active",
      "after_low_speed_calibration_scale",
      "after_low_speed_calibration_reason",
    ], delimiter="\t")
    writer.writeheader()
    for row in [
      {"route": "000000a4--demo", "segment": 10, "t": 6.00, "v_ego": 7.4, "after_vtsc_cmd": 10.1, "after_active_cap": "visible", "after_low_speed_calibration_active": False, "after_low_speed_calibration_scale": 1.0002, "after_low_speed_calibration_reason": "tighten_tracking"},
      {"route": "000000a4--demo", "segment": 10, "t": 6.08, "v_ego": 7.3, "after_vtsc_cmd": 10.0, "after_active_cap": "visible", "after_low_speed_calibration_active": False, "after_low_speed_calibration_scale": 1.0004, "after_low_speed_calibration_reason": "tighten_tracking"},
      {"route": "000000a4--demo", "segment": 10, "t": 6.16, "v_ego": 7.2, "after_vtsc_cmd": 9.9, "after_active_cap": "visible", "after_low_speed_calibration_active": True, "after_low_speed_calibration_scale": 1.0008, "after_low_speed_calibration_reason": "tighten_tracking"},
    ]:
      writer.writerow(row)

  monkeypatch.setattr(gas_event_report, "_first_model_time_s", lambda _path: 1763.385)

  out_path = base / "gas_summary.tsv"
  rc = gas_event_report.main([str(base), "--out", str(out_path)])

  assert rc == 0
  rows = list(csv.DictReader(out_path.open(), delimiter="\t"))
  assert len(rows) == 1
  assert rows[0]["event_id"] == "gas_seg10_demo"
  assert rows[0]["constraint_label"] == "not_constraining"
  assert rows[0]["calibration_label"] == "not_constraining"

  captured = capsys.readouterr()
  assert "[WARN]" not in captured.err
