from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import open_catalog, record_snapshot_bundle
from .inputs import SnapshotBundle, load_snapshot_bundle, write_snapshot_bundle


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Normalize a snapshot bundle into the EV6 longitudinal harness format")
  parser.add_argument("--vehicle-json", type=Path, required=True)
  parser.add_argument("--params-json", type=Path, required=True)
  parser.add_argument("--timeline-jsonl", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--record-to-db", action="store_true", help="Register the normalized bundle in the SQLite catalog")
  parser.add_argument("--db-path", type=Path, default=None)
  parser.add_argument("--route-id", type=int, default=None)
  parser.add_argument("--episode-id", type=int, default=None)
  args = parser.parse_args(argv)

  vehicle_payload = json.loads(args.vehicle_json.read_text())
  params_payload = json.loads(args.params_json.read_text())
  bundle = SnapshotBundle(
    path=args.out_dir,
    vehicle=vehicle_payload,
    params=params_payload,
    timeline=load_snapshot_bundle_from_files(args.timeline_jsonl),
    initial_speed_mps=float(vehicle_payload.get("initialSpeedMps", 0.0)),
    initial_accel_mps2=float(vehicle_payload.get("initialAccelMps2", 0.0)),
    name=str(vehicle_payload.get("name", args.out_dir.name)),
  )
  write_snapshot_bundle(bundle, args.out_dir)
  if args.record_to_db:
    conn = open_catalog(args.db_path)
    try:
      snapshot_id = record_snapshot_bundle(
        conn,
        args.out_dir,
        route_id=args.route_id,
        episode_id=args.episode_id,
        bundle=bundle,
      )
      conn.commit()
      print(json.dumps({"snapshotId": snapshot_id, "bundlePath": str(args.out_dir)}, indent=2, sort_keys=True))
    finally:
      conn.close()
  return 0


def load_snapshot_bundle_from_files(timeline_jsonl: Path):
  from .inputs import StepInput

  return [
    StepInput.from_json(json.loads(line))
    for line in timeline_jsonl.read_text().splitlines()
    if line.strip()
  ]


if __name__ == "__main__":
  raise SystemExit(main())
