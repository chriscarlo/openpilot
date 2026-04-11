from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

from .inputs import SnapshotBundle, load_snapshot_bundle


DEFAULT_CATALOG_PATH = Path(".cache/longitudinal_harness/catalog.sqlite3")
CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RouteCatalogRecord:
  source_root: str
  route_key: str
  car_fingerprint: str
  brand: str
  topology: str | None
  openpilot_longitudinal: bool
  radar_unavailable: bool
  safety_param: int | None
  segment_count: int
  first_segment_path: str
  notes_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentCatalogRecord:
  seg_idx: int
  rlog_path: str
  qlog_path: str | None = None
  start_log_mono_time: int | None = None
  end_log_mono_time: int | None = None


@dataclass(frozen=True)
class EpisodeCatalogRecord:
  route_id: int
  episode_key: str
  episode_type: str
  seg_start: int
  seg_end: int
  t_start_s: float
  t_end_s: float
  confidence: float
  extractor_version: str
  source_event_count: int
  metrics_json: dict[str, Any]
  bundle_path: str | None = None
  notes_json: dict[str, Any] = field(default_factory=dict)


def resolve_catalog_path(db_path: str | Path | None = None) -> Path:
  path = Path(db_path) if db_path is not None else DEFAULT_CATALOG_PATH
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def open_catalog(db_path: str | Path | None = None) -> sqlite3.Connection:
  path = resolve_catalog_path(db_path)
  conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA journal_mode=WAL")
  conn.execute("PRAGMA foreign_keys=ON")
  migrate_catalog(conn)
  return conn


def migrate_catalog(conn: sqlite3.Connection) -> None:
  conn.execute("""
    CREATE TABLE IF NOT EXISTS schema_version (
      version INTEGER NOT NULL
    )
  """)
  row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
  if row is None:
    current_version = 0
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version(version) VALUES (0)")
  else:
    current_version = int(row["version"])

  if current_version < 1:
    _apply_migration_1(conn)
    conn.execute("UPDATE schema_version SET version = 1")
    conn.commit()


def _apply_migration_1(conn: sqlite3.Connection) -> None:
  conn.executescript("""
    CREATE TABLE IF NOT EXISTS routes (
      route_id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_root TEXT NOT NULL,
      route_key TEXT NOT NULL,
      car_fingerprint TEXT NOT NULL,
      brand TEXT NOT NULL,
      topology TEXT,
      openpilot_longitudinal INTEGER NOT NULL,
      radar_unavailable INTEGER NOT NULL,
      safety_param INTEGER,
      segment_count INTEGER NOT NULL,
      first_segment_path TEXT NOT NULL,
      notes_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(source_root, route_key)
    );

    CREATE TABLE IF NOT EXISTS segments (
      segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
      route_id INTEGER NOT NULL REFERENCES routes(route_id) ON DELETE CASCADE,
      seg_idx INTEGER NOT NULL,
      rlog_path TEXT NOT NULL,
      qlog_path TEXT,
      start_log_mono_time INTEGER,
      end_log_mono_time INTEGER,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(route_id, seg_idx)
    );

    CREATE TABLE IF NOT EXISTS episodes (
      episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
      route_id INTEGER NOT NULL REFERENCES routes(route_id) ON DELETE CASCADE,
      episode_key TEXT NOT NULL UNIQUE,
      episode_type TEXT NOT NULL,
      seg_start INTEGER NOT NULL,
      seg_end INTEGER NOT NULL,
      t_start_s REAL NOT NULL,
      t_end_s REAL NOT NULL,
      confidence REAL NOT NULL,
      extractor_version TEXT NOT NULL,
      source_event_count INTEGER NOT NULL,
      metrics_json TEXT NOT NULL,
      bundle_path TEXT,
      notes_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS snapshots (
      snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
      episode_id INTEGER REFERENCES episodes(episode_id) ON DELETE SET NULL,
      route_id INTEGER REFERENCES routes(route_id) ON DELETE SET NULL,
      bundle_path TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      topology TEXT,
      controller_mode TEXT,
      noise_profile_default TEXT,
      params_json TEXT NOT NULL,
      vehicle_json TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sweeps (
      sweep_id INTEGER PRIMARY KEY AUTOINCREMENT,
      snapshot_id INTEGER REFERENCES snapshots(snapshot_id) ON DELETE SET NULL,
      mode TEXT NOT NULL,
      scenario_names_json TEXT NOT NULL,
      noise_profile TEXT NOT NULL,
      controller_mode TEXT,
      topology TEXT,
      seed INTEGER NOT NULL,
      score_weights_json TEXT NOT NULL,
      git_sha TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sweep_candidates (
      candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
      sweep_id INTEGER NOT NULL REFERENCES sweeps(sweep_id) ON DELETE CASCADE,
      rank INTEGER NOT NULL,
      score REAL NOT NULL,
      overrides_json TEXT NOT NULL,
      weighted_penalties_json TEXT NOT NULL,
      scenario_summaries_json TEXT NOT NULL
    );
  """)


def upsert_route(conn: sqlite3.Connection, route: RouteCatalogRecord) -> int:
  conn.execute(
    """
    INSERT INTO routes (
      source_root, route_key, car_fingerprint, brand, topology,
      openpilot_longitudinal, radar_unavailable, safety_param,
      segment_count, first_segment_path, notes_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(source_root, route_key) DO UPDATE SET
      car_fingerprint=excluded.car_fingerprint,
      brand=excluded.brand,
      topology=excluded.topology,
      openpilot_longitudinal=excluded.openpilot_longitudinal,
      radar_unavailable=excluded.radar_unavailable,
      safety_param=excluded.safety_param,
      segment_count=excluded.segment_count,
      first_segment_path=excluded.first_segment_path,
      notes_json=excluded.notes_json
    """,
    (
      route.source_root,
      route.route_key,
      route.car_fingerprint,
      route.brand,
      route.topology,
      int(route.openpilot_longitudinal),
      int(route.radar_unavailable),
      route.safety_param,
      route.segment_count,
      route.first_segment_path,
      _json_dumps(route.notes_json),
    ),
  )
  row = conn.execute(
    "SELECT route_id FROM routes WHERE source_root = ? AND route_key = ?",
    (route.source_root, route.route_key),
  ).fetchone()
  assert row is not None
  return int(row["route_id"])


def upsert_segment(conn: sqlite3.Connection, route_id: int, segment: SegmentCatalogRecord) -> int:
  conn.execute(
    """
    INSERT INTO segments (
      route_id, seg_idx, rlog_path, qlog_path, start_log_mono_time, end_log_mono_time
    ) VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(route_id, seg_idx) DO UPDATE SET
      rlog_path=excluded.rlog_path,
      qlog_path=excluded.qlog_path,
      start_log_mono_time=excluded.start_log_mono_time,
      end_log_mono_time=excluded.end_log_mono_time
    """,
    (
      route_id,
      segment.seg_idx,
      segment.rlog_path,
      segment.qlog_path,
      segment.start_log_mono_time,
      segment.end_log_mono_time,
    ),
  )
  row = conn.execute(
    "SELECT segment_id FROM segments WHERE route_id = ? AND seg_idx = ?",
    (route_id, segment.seg_idx),
  ).fetchone()
  assert row is not None
  return int(row["segment_id"])


def upsert_episode(conn: sqlite3.Connection, episode: EpisodeCatalogRecord) -> int:
  conn.execute(
    """
    INSERT INTO episodes (
      route_id, episode_key, episode_type, seg_start, seg_end, t_start_s, t_end_s,
      confidence, extractor_version, source_event_count, metrics_json, bundle_path, notes_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(episode_key) DO UPDATE SET
      episode_type=excluded.episode_type,
      seg_start=excluded.seg_start,
      seg_end=excluded.seg_end,
      t_start_s=excluded.t_start_s,
      t_end_s=excluded.t_end_s,
      confidence=excluded.confidence,
      extractor_version=excluded.extractor_version,
      source_event_count=excluded.source_event_count,
      metrics_json=excluded.metrics_json,
      bundle_path=excluded.bundle_path,
      notes_json=excluded.notes_json
    """,
    (
      episode.route_id,
      episode.episode_key,
      episode.episode_type,
      episode.seg_start,
      episode.seg_end,
      episode.t_start_s,
      episode.t_end_s,
      episode.confidence,
      episode.extractor_version,
      episode.source_event_count,
      _json_dumps(episode.metrics_json),
      episode.bundle_path,
      _json_dumps(episode.notes_json),
    ),
  )
  row = conn.execute(
    "SELECT episode_id FROM episodes WHERE episode_key = ?",
    (episode.episode_key,),
  ).fetchone()
  assert row is not None
  return int(row["episode_id"])


def update_episode_bundle_path(conn: sqlite3.Connection, episode_id: int, bundle_path: str) -> None:
  conn.execute("UPDATE episodes SET bundle_path = ? WHERE episode_id = ?", (bundle_path, episode_id))


def clear_route_extractions(conn: sqlite3.Connection, route_id: int) -> None:
  conn.execute(
    "DELETE FROM snapshots WHERE route_id = ? AND episode_id IS NOT NULL",
    (route_id,),
  )
  conn.execute("DELETE FROM episodes WHERE route_id = ?", (route_id,))


def record_snapshot_bundle(conn: sqlite3.Connection,
                           bundle_path: str | Path,
                           *,
                           route_id: int | None = None,
                           episode_id: int | None = None,
                           bundle: SnapshotBundle | None = None) -> int:
  bundle_root = Path(bundle_path)
  loaded_bundle = bundle if bundle is not None else load_snapshot_bundle(bundle_root)
  vehicle = dict(loaded_bundle.vehicle)
  params = dict(loaded_bundle.params)
  controller_mode = str(vehicle.get("controllerMode") or vehicle.get("resolvedControllerMode") or "")
  row = conn.execute(
    """
    INSERT INTO snapshots (
      episode_id, route_id, bundle_path, name, topology, controller_mode,
      noise_profile_default, params_json, vehicle_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(bundle_path) DO UPDATE SET
      episode_id=excluded.episode_id,
      route_id=excluded.route_id,
      name=excluded.name,
      topology=excluded.topology,
      controller_mode=excluded.controller_mode,
      noise_profile_default=excluded.noise_profile_default,
      params_json=excluded.params_json,
      vehicle_json=excluded.vehicle_json
    """,
    (
      episode_id,
      route_id,
      str(bundle_root),
      loaded_bundle.name,
      vehicle.get("topology"),
      controller_mode or None,
      vehicle.get("noiseProfile"),
      _json_dumps(params),
      _json_dumps(vehicle),
    ),
  )
  found = conn.execute("SELECT snapshot_id FROM snapshots WHERE bundle_path = ?", (str(bundle_root),)).fetchone()
  assert found is not None
  return int(found["snapshot_id"])


def get_snapshot_id_by_bundle_path(conn: sqlite3.Connection, bundle_path: str | Path) -> int | None:
  row = conn.execute("SELECT snapshot_id FROM snapshots WHERE bundle_path = ?", (str(bundle_path),)).fetchone()
  return None if row is None else int(row["snapshot_id"])


def record_sweep_result(conn: sqlite3.Connection,
                        payload: dict[str, Any],
                        *,
                        snapshot_id: int | None = None,
                        seed: int = 0) -> int:
  first_vehicle = payload["rankedCandidates"][0]["vehicle"] if payload.get("rankedCandidates") else {}
  row = conn.execute(
    """
    INSERT INTO sweeps (
      snapshot_id, mode, scenario_names_json, noise_profile, controller_mode,
      topology, seed, score_weights_json, git_sha
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
      snapshot_id,
      payload.get("mode"),
      _json_dumps(payload.get("scenarioNames", [])),
      payload.get("noiseProfile"),
      first_vehicle.get("resolvedControllerMode") or first_vehicle.get("controllerMode"),
      first_vehicle.get("topology"),
      seed,
      _json_dumps(payload.get("scoreWeights", {})),
      _git_sha(),
    ),
  )
  sweep_id = int(row.lastrowid)
  for entry in payload.get("rankedCandidates", []):
    conn.execute(
      """
      INSERT INTO sweep_candidates (
        sweep_id, rank, score, overrides_json, weighted_penalties_json, scenario_summaries_json
      ) VALUES (?, ?, ?, ?, ?, ?)
      """,
      (
        sweep_id,
        entry.get("rank"),
        entry.get("score"),
        _json_dumps(entry.get("overrides", {})),
        _json_dumps(entry.get("weightedPenalties", {})),
        _json_dumps(entry.get("scenarioSummaries", {})),
      ),
    )
  return sweep_id


def list_routes(conn: sqlite3.Connection, *, route_key: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
  query = "SELECT * FROM routes"
  args: list[Any] = []
  if route_key is not None:
    query += " WHERE route_key = ?"
    args.append(route_key)
  query += " ORDER BY created_at DESC, route_id DESC LIMIT ?"
  args.append(limit)
  return [_row_to_dict(row) for row in conn.execute(query, args).fetchall()]


def list_episodes(conn: sqlite3.Connection,
                  *,
                  route_key: str | None = None,
                  episode_type: str | None = None,
                  has_bundle: bool | None = None,
                  limit: int = 100) -> list[dict[str, Any]]:
  query = """
    SELECT e.*, r.route_key, r.source_root, r.topology
    FROM episodes e
    JOIN routes r ON r.route_id = e.route_id
  """
  clauses = []
  args: list[Any] = []
  if route_key is not None:
    clauses.append("r.route_key = ?")
    args.append(route_key)
  if episode_type is not None:
    clauses.append("e.episode_type = ?")
    args.append(episode_type)
  if has_bundle is not None:
    clauses.append("e.bundle_path IS NOT NULL" if has_bundle else "e.bundle_path IS NULL")
  if clauses:
    query += " WHERE " + " AND ".join(clauses)
  query += " ORDER BY e.route_id, e.t_start_s LIMIT ?"
  args.append(limit)
  return [_row_to_dict(row) for row in conn.execute(query, args).fetchall()]


def show_episode(conn: sqlite3.Connection, episode_id: int) -> dict[str, Any] | None:
  row = conn.execute(
    """
    SELECT e.*, r.route_key, r.source_root, r.topology, s.snapshot_id
    FROM episodes e
    JOIN routes r ON r.route_id = e.route_id
    LEFT JOIN snapshots s ON s.episode_id = e.episode_id
    WHERE e.episode_id = ?
    """,
    (episode_id,),
  ).fetchone()
  return None if row is None else _row_to_dict(row)


def get_episode_record(conn: sqlite3.Connection, episode_id: int) -> dict[str, Any] | None:
  row = conn.execute(
    """
    SELECT e.*, r.route_key, r.source_root, r.topology, s.snapshot_id
    FROM episodes e
    JOIN routes r ON r.route_id = e.route_id
    LEFT JOIN snapshots s ON s.episode_id = e.episode_id
    WHERE e.episode_id = ?
    """,
    (episode_id,),
  ).fetchone()
  return None if row is None else _row_to_dict(row)


def get_snapshot_record_by_bundle_path(conn: sqlite3.Connection, bundle_path: str | Path) -> dict[str, Any] | None:
  row = conn.execute(
    """
    SELECT s.*, r.route_key, r.source_root, e.episode_type
    FROM snapshots s
    LEFT JOIN routes r ON r.route_id = s.route_id
    LEFT JOIN episodes e ON e.episode_id = s.episode_id
    WHERE s.bundle_path = ?
    """,
    (str(bundle_path),),
  ).fetchone()
  return None if row is None else _row_to_dict(row)


def get_route_rows(conn: sqlite3.Connection, *, route_keys: list[str] | None = None) -> list[sqlite3.Row]:
  if route_keys:
    placeholders = ", ".join("?" for _ in route_keys)
    return conn.execute(
      f"SELECT * FROM routes WHERE route_key IN ({placeholders}) ORDER BY route_id",
      route_keys,
    ).fetchall()
  return conn.execute("SELECT * FROM routes ORDER BY route_id").fetchall()


def get_route_segments(conn: sqlite3.Connection, route_id: int) -> list[sqlite3.Row]:
  return conn.execute(
    "SELECT * FROM segments WHERE route_id = ? ORDER BY seg_idx",
    (route_id,),
  ).fetchall()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
  record = dict(row)
  for key in ("notes_json", "metrics_json", "params_json", "vehicle_json", "scenario_names_json", "score_weights_json", "overrides_json", "weighted_penalties_json", "scenario_summaries_json"):
    if key in record and record[key] is not None and isinstance(record[key], str):
      try:
        record[key] = json.loads(record[key])
      except json.JSONDecodeError:
        pass
  return record


def _json_dumps(value: Any) -> str:
  return json.dumps(value, sort_keys=True)


def _git_sha() -> str | None:
  try:
    proc = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
    )
  except Exception:
    return None
  return proc.stdout.strip() or None
