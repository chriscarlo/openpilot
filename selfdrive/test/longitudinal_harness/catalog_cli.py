from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import (
  get_snapshot_id_by_bundle_path,
  list_episodes,
  list_routes,
  open_catalog,
  record_snapshot_bundle,
  record_sweep_result,
  show_episode,
)
from .route_extract import extract_ev6_episodes, index_ev6_routes


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="EV6 longitudinal harness SQLite catalog")
  parser.add_argument("--db-path", type=Path, default=None)
  subparsers = parser.add_subparsers(dest="command", required=True)

  index_routes = subparsers.add_parser("index-routes", help="Scan local EV6 route roots and index matching routes")
  index_routes.add_argument("--root", action="append", default=[], help="Route root to scan")

  extract = subparsers.add_parser("extract-episodes", help="Extract EV6 approach/pullaway/cutin/handoff/dropout episodes")
  extract.add_argument("--route-key", action="append", default=[], help="Restrict extraction to one or more route keys")
  extract.add_argument("--bundle-root", type=Path, default=Path(".cache/longitudinal_harness/snapshots"))

  list_routes_parser = subparsers.add_parser("list-routes", help="List indexed EV6 routes")
  list_routes_parser.add_argument("--route-key", default=None)
  list_routes_parser.add_argument("--limit", type=int, default=100)

  list_episodes_parser = subparsers.add_parser("list-episodes", help="List extracted episodes")
  list_episodes_parser.add_argument("--route-key", default=None)
  list_episodes_parser.add_argument("--episode-type", default=None)
  list_episodes_parser.add_argument("--has-bundle", action="store_true")
  list_episodes_parser.add_argument("--missing-bundle", action="store_true")
  list_episodes_parser.add_argument("--limit", type=int, default=100)

  show_episode_parser = subparsers.add_parser("show-episode", help="Show one extracted episode")
  show_episode_parser.add_argument("episode_id", type=int)

  record_snapshot = subparsers.add_parser("record-snapshot", help="Register an existing snapshot bundle in the catalog")
  record_snapshot.add_argument("--bundle-path", type=Path, required=True)
  record_snapshot.add_argument("--route-id", type=int, default=None)
  record_snapshot.add_argument("--episode-id", type=int, default=None)

  record_sweep = subparsers.add_parser("record-sweep", help="Register a sweep JSON payload in the catalog")
  record_sweep.add_argument("--payload-json", type=Path, required=True)
  record_sweep.add_argument("--snapshot-id", type=int, default=None)
  record_sweep.add_argument("--snapshot-path", type=Path, default=None)
  record_sweep.add_argument("--seed", type=int, default=0)

  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  conn = open_catalog(args.db_path)
  try:
    if args.command == "index-routes":
      payload = index_ev6_routes(conn, roots=args.root or None)
    elif args.command == "extract-episodes":
      payload = extract_ev6_episodes(
        conn,
        route_keys=args.route_key or None,
        bundle_root=args.bundle_root,
      )
    elif args.command == "list-routes":
      payload = list_routes(conn, route_key=args.route_key, limit=args.limit)
    elif args.command == "list-episodes":
      if args.has_bundle and args.missing_bundle:
        raise SystemExit("--has-bundle and --missing-bundle are mutually exclusive")
      has_bundle = True if args.has_bundle else False if args.missing_bundle else None
      payload = list_episodes(
        conn,
        route_key=args.route_key,
        episode_type=args.episode_type,
        has_bundle=has_bundle,
        limit=args.limit,
      )
    elif args.command == "show-episode":
      payload = show_episode(conn, args.episode_id)
      if payload is None:
        raise SystemExit(f"episode {args.episode_id} not found")
    elif args.command == "record-snapshot":
      snapshot_id = record_snapshot_bundle(
        conn,
        args.bundle_path,
        route_id=args.route_id,
        episode_id=args.episode_id,
      )
      conn.commit()
      payload = {"snapshotId": snapshot_id, "bundlePath": str(args.bundle_path)}
    elif args.command == "record-sweep":
      payload_json = json.loads(args.payload_json.read_text())
      snapshot_id = args.snapshot_id
      if snapshot_id is None and args.snapshot_path is not None:
        snapshot_id = get_snapshot_id_by_bundle_path(conn, args.snapshot_path)
        if snapshot_id is None:
          snapshot_id = record_snapshot_bundle(conn, args.snapshot_path)
      sweep_id = record_sweep_result(conn, payload_json, snapshot_id=snapshot_id, seed=args.seed)
      conn.commit()
      payload = {"sweepId": sweep_id, "snapshotId": snapshot_id}
    else:
      raise SystemExit(f"unsupported command '{args.command}'")
  finally:
    conn.close()

  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
