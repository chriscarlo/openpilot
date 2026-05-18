from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cereal import messaging

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.tools.lib.logreader import save_log
from selfdrive.test.longitudinal_harness.catalog import (
  get_snapshot_id_by_bundle_path,
  list_episodes,
  list_routes,
  open_catalog,
)
from selfdrive.test.longitudinal_harness.catalog_cli import main as catalog_main
from selfdrive.test.longitudinal_harness.ingest import main as ingest_main
from selfdrive.test.longitudinal_harness.route_extract import extract_ev6_episodes, index_ev6_routes
from selfdrive.test.longitudinal_harness.sweep import enumerate_candidates, run_sweep


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "testdata" / "ev6_lka_snapshot"


def _make_car_params(*, car_fingerprint: str = "KIA_EV6"):
  cp = CarInterface.get_non_essential_params(CAR.KIA_EV6)
  cp.carFingerprint = car_fingerprint
  cp.openpilotLongitudinalControl = True
  cp.radarUnavailable = True
  if len(cp.safetyConfigs):
    cp.safetyConfigs[0].safetyParam = 21
  return cp


def _lead_spec(
  *,
  gap_m: float,
  v_lead_mps: float,
  v_rel_mps: float | None = None,
  model_prob: float = 0.92,
  radar_track_id: int = 101,
  y_rel_m: float = 0.0,
  radar: bool = True,
) -> dict[str, Any]:
  return {
    "status": True,
    "gap_m": gap_m,
    "v_lead_mps": v_lead_mps,
    "v_rel_mps": v_rel_mps,
    "model_prob": model_prob,
    "radar_track_id": radar_track_id,
    "y_rel_m": y_rel_m,
    "radar": radar,
  }


def _apply_lead_spec(lead_msg, spec: dict[str, Any] | None, *, v_ego_mps: float) -> None:
  if not spec or not spec.get("status", True):
    lead_msg.status = False
    return

  v_lead_mps = float(spec["v_lead_mps"])
  v_rel_value = spec.get("v_rel_mps")
  v_rel_mps = float(v_lead_mps - v_ego_mps if v_rel_value is None else v_rel_value)
  lead_msg.status = True
  lead_msg.dRel = float(spec["gap_m"])
  lead_msg.vRel = v_rel_mps
  lead_msg.vLead = v_lead_mps
  lead_msg.vLeadK = float(spec.get("v_lead_k_mps", v_lead_mps))
  lead_msg.aLeadK = float(spec.get("a_lead_k_mps2", 0.0))
  lead_msg.modelProb = float(spec.get("model_prob", 0.92))
  lead_msg.yRel = float(spec.get("y_rel_m", 0.0))
  lead_msg.radar = bool(spec.get("radar", True))
  lead_msg.radarTrackId = int(spec.get("radar_track_id", 0))


def _write_route_segment(route_root: Path,
                         *,
                         car_fingerprint: str = "KIA_EV6",
                         gaps_m: list[float] | None = None,
                         frame_specs: list[dict[str, Any]] | None = None) -> Path:
  if frame_specs is None:
    assert gaps_m is not None
    frame_specs = [
      {
        "v_ego_mps": 28.0,
        "lead_one": _lead_spec(gap_m=gap_m, v_lead_mps=25.5, v_rel_mps=-2.5, radar_track_id=101),
      }
      for gap_m in gaps_m
    ]

  cp = _make_car_params(car_fingerprint=car_fingerprint)
  segment_dir = route_root / "0"
  segment_dir.mkdir(parents=True, exist_ok=True)
  rlog_path = segment_dir / "rlog.zst"

  msgs = []
  mono_time = 1_000_000_000

  car_params = messaging.new_message("carParams")
  car_params.logMonoTime = mono_time
  car_params.valid = True
  car_params.carParams = cp
  msgs.append(car_params.as_reader())
  mono_time += 5_000_000

  car_control_sp = messaging.new_message("carControlSP")
  car_control_sp.logMonoTime = mono_time
  car_control_sp.valid = True
  car_control_sp.carControlSP.init("params", 2)
  car_control_sp.carControlSP.params[0].key = "HyundaiLongitudinalTuning"
  car_control_sp.carControlSP.params[0].value = "0"
  car_control_sp.carControlSP.params[1].key = "LongTuningDecelRate"
  car_control_sp.carControlSP.params[1].value = "0.35"
  msgs.append(car_control_sp.as_reader())
  mono_time += 5_000_000

  for idx, frame_spec in enumerate(frame_specs):
    frame_base = mono_time + idx * 50_000_000
    v_ego_mps = float(frame_spec.get("v_ego_mps", 28.0))
    a_ego_mps2 = float(frame_spec.get("a_ego_mps2", 0.0))
    cruise_speed_kph = float(frame_spec.get("v_cruise_kph", 110.0))

    car_state = messaging.new_message("carState")
    car_state.logMonoTime = frame_base
    car_state.valid = True
    car_state.carState.vEgo = v_ego_mps
    car_state.carState.aEgo = a_ego_mps2
    car_state.carState.vCruise = cruise_speed_kph
    msgs.append(car_state.as_reader())

    controls_state = messaging.new_message("controlsState")
    controls_state.logMonoTime = frame_base + 5_000_000
    controls_state.valid = True
    controls_state.controlsState.longControlState = "pid"
    controls_state.controlsState.forceDecel = False
    msgs.append(controls_state.as_reader())

    selfdrive_state = messaging.new_message("selfdriveState")
    selfdrive_state.logMonoTime = frame_base + 10_000_000
    selfdrive_state.valid = True
    selfdrive_state.selfdriveState.experimentalMode = False
    msgs.append(selfdrive_state.as_reader())

    car_control = messaging.new_message("carControl")
    car_control.logMonoTime = frame_base + 15_000_000
    car_control.valid = True
    car_control.carControl.longActive = True
    car_control.carControl.init("orientationNED", 3)
    car_control.carControl.orientationNED[1] = 0.01
    msgs.append(car_control.as_reader())

    radar_state = messaging.new_message("radarState")
    radar_state.logMonoTime = frame_base + 20_000_000
    radar_state.valid = True
    _apply_lead_spec(radar_state.radarState.leadOne, frame_spec.get("lead_one"), v_ego_mps=v_ego_mps)
    _apply_lead_spec(radar_state.radarState.leadTwo, frame_spec.get("lead_two"), v_ego_mps=v_ego_mps)
    msgs.append(radar_state.as_reader())

  save_log(str(rlog_path), msgs)
  return rlog_path


def _build_local_logs(tmp_path: Path) -> Path:
  root = tmp_path / "logs"
  _write_route_segment(root / "ev6_route", gaps_m=[120.0 - 0.42 * idx for idx in range(260)])
  _write_route_segment(root / "other_route", car_fingerprint="KIA_SORENTO", gaps_m=[50.0 for _ in range(20)])
  return root


def test_index_ev6_routes_skips_non_ev6_and_is_idempotent(tmp_path: Path) -> None:
  db_path = tmp_path / "catalog.sqlite3"
  log_root = _build_local_logs(tmp_path)
  conn = open_catalog(db_path)
  try:
    indexed = index_ev6_routes(conn, roots=[log_root])
    reindexed = index_ev6_routes(conn, roots=[log_root])
    routes = list_routes(conn)
  finally:
    conn.close()

  assert len(indexed) == 1
  assert len(reindexed) == 1
  assert len(routes) == 1
  assert routes[0]["route_key"] == "ev6_route"
  assert routes[0]["topology"] == "lka"
  assert routes[0]["openpilot_longitudinal"] == 1
  assert routes[0]["radar_unavailable"] == 1


def test_extract_ev6_episodes_creates_snapshot_bundle(tmp_path: Path) -> None:
  db_path = tmp_path / "catalog.sqlite3"
  log_root = _build_local_logs(tmp_path)
  bundle_root = tmp_path / "snapshots"
  conn = open_catalog(db_path)
  try:
    index_ev6_routes(conn, roots=[log_root])
    recorded = extract_ev6_episodes(conn, bundle_root=bundle_root)
    episodes = list_episodes(conn, has_bundle=True)
  finally:
    conn.close()

  assert recorded
  approach_entries = [entry for entry in recorded if entry["episodeType"] == "approach"]
  assert len(approach_entries) == 1
  assert episodes
  bundle_path = Path(episodes[0]["bundle_path"])
  assert bundle_path.exists()
  assert bundle_path.joinpath("vehicle.json").exists()
  assert bundle_path.joinpath("timeline.jsonl").exists()


def test_extract_ev6_episodes_detects_stable_high_speed_handoff(tmp_path: Path) -> None:
  db_path = tmp_path / "catalog.sqlite3"
  bundle_root = tmp_path / "snapshots"
  log_root = tmp_path / "logs"
  handoff_frames = []
  for idx in range(80):
    lead_one_gap = 34.0 + 0.50 * idx
    lead_two_gap = 58.0 - 0.15 * idx
    handoff_frames.append({
      "v_ego_mps": 29.5,
      "lead_one": _lead_spec(gap_m=lead_one_gap, v_lead_mps=29.0, radar_track_id=101),
      "lead_two": _lead_spec(gap_m=lead_two_gap, v_lead_mps=24.5, radar_track_id=202),
    })
  _write_route_segment(log_root / "handoff_route", frame_specs=handoff_frames)

  conn = open_catalog(db_path)
  try:
    index_ev6_routes(conn, roots=[log_root])
    recorded = extract_ev6_episodes(conn, route_keys=["handoff_route"], bundle_root=bundle_root)
    episodes = list_episodes(conn, route_key="handoff_route", episode_type="handoff", has_bundle=True, limit=10)
  finally:
    conn.close()

  assert any(entry["episodeType"] == "handoff" for entry in recorded)
  assert len(episodes) == 1
  assert episodes[0]["metrics_json"]["leadSwitchCount"] >= 1
  assert episodes[0]["metrics_json"]["incomingLeadSpeedDeltaMps"] > 3.0


def test_extract_ev6_episodes_rejects_low_speed_duplicate_churn_as_handoff(tmp_path: Path) -> None:
  db_path = tmp_path / "catalog.sqlite3"
  bundle_root = tmp_path / "snapshots"
  log_root = tmp_path / "logs"
  churn_frames = []
  for idx in range(80):
    lead_one_gap = 3.0 + (0.2 if idx % 2 == 0 else -0.2)
    lead_two_gap = 3.0 + (-0.2 if idx % 2 == 0 else 0.2)
    churn_frames.append({
      "v_ego_mps": 1.4,
      "lead_one": _lead_spec(gap_m=lead_one_gap, v_lead_mps=1.3, radar_track_id=101),
      "lead_two": _lead_spec(gap_m=lead_two_gap, v_lead_mps=1.35, radar_track_id=202),
    })
  _write_route_segment(log_root / "duplicate_churn_route", frame_specs=churn_frames)

  conn = open_catalog(db_path)
  try:
    index_ev6_routes(conn, roots=[log_root])
    extract_ev6_episodes(conn, route_keys=["duplicate_churn_route"], bundle_root=bundle_root)
    episodes = list_episodes(conn, route_key="duplicate_churn_route", episode_type="handoff", has_bundle=True, limit=10)
  finally:
    conn.close()

  assert episodes == []


def test_catalog_cli_records_snapshot_and_sweep(tmp_path: Path, capsys) -> None:
  db_path = tmp_path / "catalog.sqlite3"

  assert catalog_main(["--db-path", str(db_path), "record-snapshot", "--bundle-path", str(FIXTURE_DIR)]) == 0
  snapshot_payload = json.loads(capsys.readouterr().out)
  assert snapshot_payload["snapshotId"] > 0

  payload = run_sweep(
    candidates=enumerate_candidates({}, {"Longitudinal.LiveTune.AccelChangeCost": ["350", "400"]}),
    scenarios=["approach"],
    mode="snapshot",
    snapshot=FIXTURE_DIR,
    topology="lka",
    controller_mode="auto",
    hyundai_tuning_mode=None,
    noise="off",
    duration_s=1.0,
    seed=9,
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
  payload_path = tmp_path / "sweep.json"
  payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

  assert catalog_main([
    "--db-path", str(db_path),
    "record-sweep",
    "--payload-json", str(payload_path),
    "--snapshot-path", str(FIXTURE_DIR),
    "--seed", "9",
  ]) == 0
  sweep_payload = json.loads(capsys.readouterr().out)
  assert sweep_payload["sweepId"] > 0
  assert sweep_payload["snapshotId"] == snapshot_payload["snapshotId"]


def test_ingest_can_register_normalized_bundle(tmp_path: Path) -> None:
  db_path = tmp_path / "catalog.sqlite3"
  out_dir = tmp_path / "normalized_bundle"

  assert ingest_main([
    "--vehicle-json", str(FIXTURE_DIR / "vehicle.json"),
    "--params-json", str(FIXTURE_DIR / "params.json"),
    "--timeline-jsonl", str(FIXTURE_DIR / "timeline.jsonl"),
    "--out-dir", str(out_dir),
    "--record-to-db",
    "--db-path", str(db_path),
  ]) == 0

  conn = open_catalog(db_path)
  try:
    snapshot_id = get_snapshot_id_by_bundle_path(conn, out_dir)
  finally:
    conn.close()

  assert snapshot_id is not None
  assert out_dir.joinpath("vehicle.json").exists()
