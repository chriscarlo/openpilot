#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpilot.sunnypilot.selfdrive.controls.lib.planner_lag_debug import EVENTS_DIR_DEFAULT


def _load_json(path: Path) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def _resolve_event_dir(raw: str | None) -> Path:
  if raw:
    return Path(raw)
  candidates = sorted((p for p in EVENTS_DIR_DEFAULT.iterdir() if p.is_dir()), key=lambda p: p.name)
  if not candidates:
    raise FileNotFoundError(f"no planner lag events under {EVENTS_DIR_DEFAULT}")
  return candidates[-1]


def _fmt_ms(value: float | None) -> str:
  if value is None:
    return "-"
  return f"{float(value):.2f} ms"


def main() -> int:
  ap = argparse.ArgumentParser(description="Summarize a planner-local lag bundle captured by plannerd.")
  ap.add_argument("event_dir", nargs="?", help="Event directory. Defaults to the newest bundle on device.")
  ap.add_argument("--top", type=int, default=8, help="How many ranked components to print")
  args = ap.parse_args()

  event_dir = _resolve_event_dir(args.event_dir)
  event = _load_json(event_dir / "event.json")
  summary = _load_json(event_dir / "summary.json")
  trigger = _load_json(event_dir / "trigger_cycle.json")

  print(f"Event: {event.get('event_id', event_dir.name)}")
  print(f"Dir:   {event_dir}")
  print(f"Route: {event.get('route') or '-'}")
  print(f"Seg:   {event.get('seg_guess') if event.get('seg_guess') is not None else '-'}")
  print(f"Reasons: {', '.join(summary.get('trigger_reasons', []) or ['-'])}")
  print(f"Cycles:  {summary.get('window_cycles', 0)} total, {summary.get('bad_cycles', 0)} bad")
  print(f"Dominant suspect split: {summary.get('dominant_component_bad_cycles') or '-'}")
  print()

  print("Suspect Split Means On Bad Cycles")
  suspect_rank = summary.get("suspect_split_rank_bad_cycles_ms", [])[: max(1, int(args.top))]
  for item in suspect_rank:
    print(f"  {item['name']}: {_fmt_ms(item.get('mean_ms'))}")
  print()

  print("All Component Means On Bad Cycles")
  component_rank = summary.get("component_mean_rank_bad_cycles_ms", [])[: max(1, int(args.top))]
  for item in component_rank:
    print(f"  {item['name']}: {_fmt_ms(item.get('mean_ms'))}")
  print()

  print("Trigger Cycle")
  trigger_fields = (
    ("planner_publish_gap_ms", "publish_gap"),
    ("planner_loop_dt_ms", "planner_loop"),
    ("strategy_mode", "strategy_mode"),
    ("strategy_state", "strategy_state"),
    ("vtsc_state_name", "vtsc_state"),
    ("vtsc_velocity", "vtsc_velocity"),
    ("curve_preview_points", "preview_points"),
    ("curve_preview_branch_stubs", "branch_stubs"),
    ("curve_distance_m", "curve_distance_m"),
    ("curve_time_to_s", "curve_time_to_s"),
    ("map_geometry_valid", "map_geometry_valid"),
    ("nearby_segment_count", "nearby_segment_count"),
    ("active_cap", "active_cap"),
    ("map_tail_reason", "map_tail_reason"),
    ("map_tail_compute_reason", "map_tail_compute_reason"),
    ("strategic_outside_helper_ms", "strategic_outside_helper"),
    ("helper_total_ms", "helper_total"),
    ("preview_total_ms", "preview_total"),
    ("vtsc_non_map_tail_ms", "vtsc_non_map_tail"),
    ("planner_update_non_vtsc_ms", "planner_update_non_vtsc"),
    ("publish_non_preview_encode_ms", "publish_non_preview_encode"),
    ("planner_loop_other_ms", "planner_loop_other"),
  )
  for key, label in trigger_fields:
    value = trigger.get(key)
    if key.endswith("_ms"):
      print(f"  {label}: {_fmt_ms(value)}")
    else:
      print(f"  {label}: {value if value is not None else '-'}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
