from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfdrive.test.longitudinal_harness.benchmark import (
  load_benchmark_manifest,
  main as benchmark_main,
  run_manifest_benchmark,
)
from selfdrive.test.longitudinal_harness.catalog import list_episodes, list_routes, open_catalog
from selfdrive.test.longitudinal_harness.route_extract import extract_ev6_episodes, index_ev6_routes
from selfdrive.test.longitudinal_harness.sweep import enumerate_candidates
from selfdrive.test.longitudinal_harness.tests.test_catalog import FIXTURE_DIR, _write_route_segment


def test_run_manifest_benchmark_aggregates_case_scores(tmp_path: Path) -> None:
  manifest_path = tmp_path / "smoke_manifest.json"
  manifest_path.write_text(json.dumps({
    "schemaVersion": 1,
    "name": "fixture-smoke",
    "description": "two fixture-backed cases",
    "cases": [
      {
        "caseId": "fixture_a",
        "snapshotPath": str(FIXTURE_DIR),
      },
      {
        "caseId": "fixture_b",
        "snapshotPath": str(FIXTURE_DIR),
      },
    ],
  }, indent=2, sort_keys=True))
  manifest = load_benchmark_manifest(manifest_path)

  payload = run_manifest_benchmark(
    manifest=manifest,
    candidates=enumerate_candidates({}, {"Longitudinal.LiveTune.GapReclaimStrength": ["0.45", "0.55"]}),
    controller_mode="auto",
    hyundai_tuning_mode=None,
    noise="off",
    seed=17,
    noise_seeds=None,
    score_weights={
      "excessBrakeMps2": 1.0,
      "maxFollowOvershootMps": 1.0,
      "maxFollowUndershootMps": 1.0,
      "reclaimDelayS": 1.0,
      "leadAcquireLatencyS": 1.0,
      "handoffBrakeLatencyS": 1.0,
      "longControlRealizedDivergenceMps2": 1.0,
    },
  )

  assert payload["caseCount"] == 2
  assert payload["candidateCount"] == 2
  best = payload["rankedCandidates"][0]
  assert set(best["caseScores"]) == {"fixture_a", "fixture_b"}
  assert best["score"] == pytest.approx(sum(best["caseScores"].values()), abs=1e-6)


def test_write_manifest_keeps_duplicate_route_keys_distinct(tmp_path: Path, capsys) -> None:
  db_path = tmp_path / "catalog.sqlite3"
  bundle_root = tmp_path / "snapshots"
  gaps_m = [120.0 - 0.42 * idx for idx in range(260)]
  root_a = tmp_path / "root_a"
  root_b = tmp_path / "root_b"
  _write_route_segment(root_a / "dup_route", gaps_m=gaps_m)
  _write_route_segment(root_b / "dup_route", gaps_m=gaps_m)

  conn = open_catalog(db_path)
  try:
    index_ev6_routes(conn, roots=[root_a, root_b])
    recorded = extract_ev6_episodes(conn, route_keys=["dup_route"], bundle_root=bundle_root)
    routes = list_routes(conn, route_key="dup_route", limit=10)
    episodes = list_episodes(conn, route_key="dup_route", has_bundle=True, limit=50)
  finally:
    conn.close()

  assert len(routes) == 2
  assert len({entry["routeId"] for entry in recorded}) == 2
  assert len({Path(entry["bundlePath"]).parts[-2] for entry in recorded}) == 2

  chosen_episode_ids = []
  seen_route_ids: set[int] = set()
  for episode in episodes:
    route_id = int(episode["route_id"])
    if route_id in seen_route_ids:
      continue
    seen_route_ids.add(route_id)
    chosen_episode_ids.append(int(episode["episode_id"]))
  assert len(chosen_episode_ids) == 2

  manifest_path = tmp_path / "dup_manifest.json"
  assert benchmark_main([
    "write-manifest",
    "--db-path", str(db_path),
    "--output", str(manifest_path),
    "--name", "dup-routes",
    "--episode-id", str(chosen_episode_ids[0]),
    "--episode-id", str(chosen_episode_ids[1]),
  ]) == 0
  manifest_payload = json.loads(capsys.readouterr().out)

  assert len(manifest_payload["cases"]) == 2
  assert {case["routeKey"] for case in manifest_payload["cases"]} == {"dup_route"}
  assert len({case["routeId"] for case in manifest_payload["cases"]}) == 2
  assert len({case["sourceRoot"] for case in manifest_payload["cases"]}) == 2
