#!/usr/bin/env python3
"""Tests for vtsc_watch.py render_line and evaluate_flags."""
import sys
from pathlib import Path

# Ensure the tools dir is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.vtsc_watch import evaluate_flags, render_line


def _base_snapshot(**overrides):
  d = {
    "v": 15.0,
    "v_base": 18.0,
    "raw": 16.0,
    "final": 15.5,
    "v_vis": 17.0,
    "v_occ": 14.0,
    "active_cap": "visible",
    "cap_visible_vmin": 16.5,
    "cap_occl_vmin": 14.0,
    "cap_map_vmin": 17.0,
    "psi_vis": 0.3,
    "psi_thresh": 0.2,
    "conf": 0.85,
    "occlusion_reason": "",
    "tail_frac": 0.6,
    "s_tail": 80.0,
    "kappa_vis": 0.001,
    "s_visible_m": 130.0,
    "occluded": False,
    "map_tail_coverage": 0.8,
    "map_tail_active": True,
  }
  d.update(overrides)
  return d


class TestRenderLine:
  def test_output_contains_map_cap(self):
    d = _base_snapshot(cap_map_vmin=12.3)
    out = render_line(d, "snap")
    assert "map=12.3" in out

  def test_output_contains_all_caps(self):
    d = _base_snapshot()
    out = render_line(d, "swag")
    assert "vis=" in out
    assert "occ=" in out
    assert "map=" in out
    assert "cap=visible" in out

  def test_output_format_source_tag(self):
    d = _base_snapshot()
    out = render_line(d, "snap")
    assert out.startswith("[snap]")

  def test_map_cap_zero_renders_as_number(self):
    d = _base_snapshot(cap_map_vmin=0.0)
    out = render_line(d, "snap")
    assert "map=0.0" in out


class TestEvaluateFlags:
  def test_no_flags_baseline(self):
    d = _base_snapshot()
    flags = evaluate_flags(d)
    assert flags == []

  def test_freeway_failopen_missed(self):
    d = _base_snapshot(
      active_cap="occlusion",
      kappa_vis=0.0,
      s_visible_m=150.0,
      conf=0.75,
    )
    flags = evaluate_flags(d)
    assert "freeway_failopen_missed" in flags

  def test_double_occl_cap_suspect(self):
    d = _base_snapshot(
      active_cap="occlusion",
      raw=14.2,
      cap_occl_vmin=14.0,
    )
    flags = evaluate_flags(d)
    assert "double_occl_cap_suspect" in flags

  def test_pretrigger_with_high_conf(self):
    d = _base_snapshot(occlusion_reason="pretrigger", conf=0.80)
    flags = evaluate_flags(d)
    assert "pretrigger_with_high_conf" in flags

  def test_psi_below_thresh(self):
    d = _base_snapshot(
      active_cap="occlusion",
      psi_vis=0.10,
      psi_thresh=0.20,
    )
    flags = evaluate_flags(d)
    assert "psi_below_thresh" in flags

  def test_map_low_coverage_flagged(self):
    d = _base_snapshot(
      active_cap="map",
      map_tail_coverage=0.15,
      cap_map_vmin=14.0,
      v_base=18.0,
    )
    flags = evaluate_flags(d)
    assert "map_low_coverage" in flags

  def test_map_low_coverage_not_flagged_when_coverage_high(self):
    d = _base_snapshot(
      active_cap="map",
      map_tail_coverage=0.5,
      cap_map_vmin=14.0,
      v_base=18.0,
    )
    flags = evaluate_flags(d)
    assert "map_low_coverage" not in flags

  def test_map_low_coverage_not_flagged_when_cap_close_to_base(self):
    d = _base_snapshot(
      active_cap="map",
      map_tail_coverage=0.1,
      cap_map_vmin=17.5,
      v_base=18.0,
    )
    flags = evaluate_flags(d)
    assert "map_low_coverage" not in flags

  def test_map_low_coverage_not_flagged_when_active_cap_not_map(self):
    d = _base_snapshot(
      active_cap="visible",
      map_tail_coverage=0.1,
      cap_map_vmin=12.0,
      v_base=18.0,
    )
    flags = evaluate_flags(d)
    assert "map_low_coverage" not in flags
