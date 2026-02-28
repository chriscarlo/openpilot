#!/usr/bin/env python3
"""Tests for analyze_snapshots.py: load_jsonl, TSV dump, flag_handoff_conflicts."""
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the analysis dir is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_snapshots import (
  flag_handoff_conflicts,
  load_jsonl,
  main,
  _get_float,
  _row_target_speed,
)


def _make_snapshot(**overrides):
  d = {
    "ts": 1000.0,
    "v": 15.0,
    "v_base": 18.0,
    "v_vis": 17.0,
    "v_occ": 14.0,
    "final": 15.5,
    "raw": 16.0,
    "active_cap": "visible",
    "cap_visible_vmin": 16.5,
    "cap_occl_vmin": 14.0,
    "cap_map_vmin": 17.0,
    "map_tail_active": False,
    "map_tail_coverage": 0.8,
    "vision_status": "FULL",
    "conf": 0.85,
    "occluded": False,
    "s_visible_m": 100.0,
    "comfort_decel": -1.47,
    "max_adaptive_decel": -6.0,
  }
  d.update(overrides)
  return d


def _write_jsonl(rows, path):
  with open(path, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")


class TestLoadJsonl:
  def test_load_valid(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
      for i in range(5):
        f.write(json.dumps(_make_snapshot(ts=1000.0 + i)) + "\n")
      f.flush()
      rows = load_jsonl(f.name)
    os.unlink(f.name)
    assert len(rows) == 5
    assert rows[0]["ts"] == 1000.0

  def test_load_skips_blank_lines(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
      f.write(json.dumps(_make_snapshot()) + "\n")
      f.write("\n")
      f.write(json.dumps(_make_snapshot(ts=1001.0)) + "\n")
      f.flush()
      rows = load_jsonl(f.name)
    os.unlink(f.name)
    assert len(rows) == 2

  def test_load_skips_malformed_json(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
      f.write(json.dumps(_make_snapshot()) + "\n")
      f.write("{bad json\n")
      f.write(json.dumps(_make_snapshot(ts=1001.0)) + "\n")
      f.flush()
      rows = load_jsonl(f.name)
    os.unlink(f.name)
    assert len(rows) == 2


class TestTsvDump:
  def test_tsv_header_and_row_count(self):
    rows = [_make_snapshot(ts=1000.0 + i * 0.05) for i in range(10)]
    with tempfile.TemporaryDirectory() as td:
      jsonl_path = os.path.join(td, "snaps.jsonl")
      tsv_path = os.path.join(td, "out.tsv")
      _write_jsonl(rows, jsonl_path)
      # Invoke main via sys.argv override
      orig_argv = sys.argv
      sys.argv = ["analyze_snapshots.py", jsonl_path, "--dump-tsv", tsv_path]
      try:
        main(jsonl_path)
      finally:
        sys.argv = orig_argv
      assert os.path.exists(tsv_path)
      with open(tsv_path) as f:
        lines = f.readlines()
      # First line is header (starts with #)
      assert lines[0].startswith("# ts")
      # Should have 17 tab-separated columns
      header_cols = lines[0].lstrip("# ").strip().split("\t")
      assert len(header_cols) == 17
      # Data rows == number of snapshots
      data_lines = [l for l in lines[1:] if l.strip()]
      assert len(data_lines) == 10
      # Each data line should have 17 tab-separated fields
      for dl in data_lines:
        fields = dl.strip().split("\t")
        assert len(fields) == 17

  def test_tsv_includes_map_columns(self):
    rows = [_make_snapshot(cap_map_vmin=12.5, map_tail_coverage=0.45)]
    with tempfile.TemporaryDirectory() as td:
      jsonl_path = os.path.join(td, "snaps.jsonl")
      tsv_path = os.path.join(td, "out.tsv")
      _write_jsonl(rows, jsonl_path)
      orig_argv = sys.argv
      sys.argv = ["analyze_snapshots.py", jsonl_path, "--dump-tsv", tsv_path]
      try:
        main(jsonl_path)
      finally:
        sys.argv = orig_argv
      with open(tsv_path) as f:
        lines = f.readlines()
      header = lines[0].lstrip("# ").strip()
      assert "cap_map" in header
      assert "map_cov" in header
      # Data row should contain the map values
      data = lines[1].strip().split("\t")
      # cap_map is column index 8 (0-indexed), map_cov is 9
      assert "12.500" in data[8]
      assert "0.450" in data[9]


class TestFlagHandoffConflicts:
  def test_no_conflict_when_vision_not_full(self):
    rows = [_make_snapshot(vision_status="PARTIAL", map_tail_active=True,
                           active_cap="map", cap_visible_vmin=15.0, cap_map_vmin=16.0)]
    bad, total = flag_handoff_conflicts(rows)
    assert total == 0
    assert bad == 0

  def test_no_conflict_when_map_not_active(self):
    rows = [_make_snapshot(vision_status="FULL", map_tail_active=False)]
    bad, total = flag_handoff_conflicts(rows)
    assert total == 0

  def test_conflict_detected(self):
    rows = [_make_snapshot(
      vision_status="FULL",
      map_tail_active=True,
      active_cap="map",
      cap_visible_vmin=15.0,
      cap_map_vmin=16.0,
    )]
    bad, total = flag_handoff_conflicts(rows)
    assert total == 1
    assert bad == 1

  def test_no_conflict_when_visible_is_looser(self):
    rows = [_make_snapshot(
      vision_status="FULL",
      map_tail_active=True,
      active_cap="map",
      cap_visible_vmin=20.0,
      cap_map_vmin=15.0,
    )]
    bad, total = flag_handoff_conflicts(rows)
    assert total == 1
    assert bad == 0


class TestRowTargetSpeed:
  def test_minimum_of_all_caps(self):
    r = _make_snapshot(cap_visible_vmin=16.0, cap_occl_vmin=14.0, cap_map_vmin=15.0)
    assert _row_target_speed(r) == 14.0

  def test_ignores_zero_caps(self):
    r = _make_snapshot(cap_visible_vmin=16.0, cap_occl_vmin=0.0, cap_map_vmin=0.0)
    assert _row_target_speed(r) == 16.0

  def test_falls_back_to_final(self):
    r = _make_snapshot(cap_visible_vmin=0.0, cap_occl_vmin=0.0, cap_map_vmin=0.0, final=13.0)
    assert _row_target_speed(r) == 13.0
