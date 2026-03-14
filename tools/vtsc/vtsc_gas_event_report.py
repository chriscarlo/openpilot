#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

try:
  from vtsc.vtsc_gas_event_window import summarize_gas_event_window
except ImportError:
  from vtsc_gas_event_window import summarize_gas_event_window


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
  rows: List[Dict[str, Any]] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def _first_model_time_s(rlog_path: Path) -> float | None:
  from openpilot.tools.lib.logreader import LogReader

  for msg in LogReader(str(rlog_path)):
    if msg.which() != "modelV2":
      continue
    return float(msg.logMonoTime) * 1e-9
  return None


def _load_samples_tsv(path: Path) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
  rows_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
  with path.open(encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      route = str(row.get("route") or "")
      try:
        seg = int(row.get("segment") or row.get("seg") or -1)
      except Exception:
        continue
      rows_by_key.setdefault((route, seg), []).append(row)
  return rows_by_key


def _trace_rows_from_samples_tsv(
  *,
  event: Dict[str, Any],
  rows_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]],
  realdata_root: Path,
) -> List[Dict[str, Any]]:
  route = str(event.get("route") or "")
  try:
    seg = int(event.get("seg"))
    t0 = float(event.get("t0"))
  except Exception:
    return []
  sample_rows = rows_by_key.get((route, seg), [])
  if not sample_rows:
    return []

  first_model_s = _first_model_time_s(realdata_root / f"{route}--{seg}" / "rlog.zst")
  if first_model_s is None:
    return []
  rel_t0 = float(t0 - first_model_s)

  out: List[Dict[str, Any]] = []
  for row in sample_rows:
    try:
      sample_t = float(row.get("t") or 0.0)
    except Exception:
      continue
    out.append({
      "dt": float(sample_t - rel_t0),
      "vEgo": row.get("v_ego"),
      "vtscVelMps": row.get("after_vtsc_cmd") or row.get("vtscVelMps"),
      "lowSpeedCalActive": row.get("after_low_speed_calibration_active") or row.get("lowSpeedCalActive"),
      "lowSpeedCalScale": row.get("after_low_speed_calibration_scale") or row.get("lowSpeedCalScale"),
      "lowSpeedCalReason": row.get("after_low_speed_calibration_reason") or row.get("lowSpeedCalReason"),
      "activeCap": row.get("after_active_cap") or row.get("activeCap"),
    })
  return out


def main(argv: List[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description="Summarize offline gas events from RCA trace windows.")
  ap.add_argument("base", help="RCA bundle root containing events_offline/")
  ap.add_argument("--events-subdir", default="events_offline")
  ap.add_argument("--realdata-subdir", default="realdata")
  ap.add_argument("--trace-name", default="trace_rlog_20s_plus.jsonl")
  ap.add_argument("--samples-tsv", default="", help="Optional replay samples TSV used when traces are absent")
  ap.add_argument("--window-half-s", type=float, default=1.0)
  ap.add_argument("--out", default="", help="Optional TSV output path")
  args = ap.parse_args(argv)

  base = Path(args.base).resolve()
  events_root = base / args.events_subdir
  realdata_root = base / args.realdata_subdir
  if not events_root.is_dir():
    raise SystemExit(f"events dir missing: {events_root}")

  samples_tsv = Path(args.samples_tsv).resolve() if args.samples_tsv else None
  if samples_tsv is None:
    default_samples = base / "gas_calibration_samples.tsv"
    if default_samples.exists():
      samples_tsv = default_samples
  sample_rows_by_key = _load_samples_tsv(samples_tsv) if samples_tsv and samples_tsv.exists() else {}

  rows: List[Dict[str, Any]] = []
  for event_dir in sorted(events_root.iterdir()):
    if not event_dir.is_dir():
      continue
    event_path = event_dir / "event.json"
    if not event_path.exists():
      continue
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if str(event.get("action") or "") != "gas":
      continue

    trace_path = event_dir / args.trace_name
    if trace_path.exists():
      trace_rows = _load_jsonl_rows(trace_path)
    elif sample_rows_by_key:
      trace_rows = _trace_rows_from_samples_tsv(
        event=event,
        rows_by_key=sample_rows_by_key,
        realdata_root=realdata_root,
      )
    else:
      trace_rows = []

    if not trace_rows:
      print(f"[WARN] missing trace for {event_dir.name}: {trace_path}", file=sys.stderr)
      continue

    summary = summarize_gas_event_window(trace_rows, half_s=float(args.window_half_s))
    rows.append({
      "event_id": str(event.get("event_id") or event_dir.name),
      "route": str(event.get("route") or ""),
      "seg": event.get("seg"),
      "t0": event.get("t0"),
      "window_half_s": summary["window_half_s"],
      "samples": summary["samples"],
      "constraint_label": summary["constraint_label"],
      "calibration_label": summary["calibration_label"],
      "active_share": summary["active_share"],
      "gap_median": summary["gap_median"],
      "gap_max": summary["gap_max"],
      "scale_min": summary["scale_min"],
      "scale_max": summary["scale_max"],
      "reason_mode": summary["reason_mode"],
      "cap_mode": summary["cap_mode"],
    })

  fieldnames = [
    "event_id", "route", "seg", "t0",
    "window_half_s", "samples",
    "constraint_label", "calibration_label",
    "active_share", "gap_median", "gap_max",
    "scale_min", "scale_max",
    "reason_mode", "cap_mode",
  ]
  if args.out:
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)

  writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
  writer.writeheader()
  writer.writerows(rows)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
