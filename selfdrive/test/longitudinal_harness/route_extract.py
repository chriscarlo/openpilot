from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from openpilot.tools.lib.logreader import LogReader

from opendbc.car.hyundai.values import HyundaiFlags, HyundaiSafetyFlags

from .catalog import (
  clear_route_extractions,
  EpisodeCatalogRecord,
  RouteCatalogRecord,
  SegmentCatalogRecord,
  get_route_rows,
  get_route_segments,
  record_snapshot_bundle,
  update_episode_bundle_path,
  upsert_episode,
  upsert_route,
  upsert_segment,
)
from .inputs import LeadDirective, SnapshotBundle, StepInput, write_snapshot_bundle


EXTRACTOR_VERSION = "ev6_v3"
DEFAULT_ROUTE_ROOTS = (
  Path(".cache/commaCar"),
  Path(".cache/commaAdb"),
  Path(".cache/tici_logs"),
  Path(".cache/route_00000094_853be5484c"),
  Path(".cache/route_00000094_853be5484c_recent"),
  Path(".cache/route_00000099_22d46c0f3d"),
)
WINDOWS_BY_TYPE = {
  "approach": (-2.0, 5.0),
  "pullaway": (-2.0, 6.0),
  "cutin": (-2.0, 4.0),
  "handoff": (-2.0, 4.0),
  "dropout": (-2.0, 4.0),
}
COOLDOWN_BY_TYPE_S = {
  "approach": 6.0,
  "pullaway": 6.0,
  "cutin": 5.0,
  "handoff": 5.0,
  "dropout": 4.0,
}
SUPPRESSION_BY_TYPE_S = {
  "approach": 10.0,
  "pullaway": 10.0,
  "cutin": 8.0,
  "handoff": 6.0,
  "dropout": 6.0,
}


@dataclass(frozen=True)
class DiscoveredSegment:
  source_root: Path
  route_key: str
  seg_idx: int
  rlog_path: Path
  qlog_path: Path | None


@dataclass(frozen=True)
class RouteMetadata:
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
  notes_json: dict[str, Any]


@dataclass(frozen=True)
class RouteLeadFrame:
  status: bool
  d_rel_m: float | None
  v_rel_mps: float | None
  v_lead_mps: float | None
  a_lead_k_mps2: float | None
  v_lead_k_mps: float | None
  model_prob: float
  y_rel_m: float
  d_path_m: float | None
  v_lat_mps: float
  radar: bool
  radar_track_id: int


@dataclass(frozen=True)
class EpisodeFrame:
  route_key: str
  seg_idx: int
  log_mono_time: int
  t_s: float
  v_ego_mps: float
  a_ego_mps2: float
  cruise_speed_mps: float
  long_active: bool
  long_control_state: str
  force_decel: bool
  experimental_mode: bool
  pitch_rad: float
  lead_one: RouteLeadFrame
  lead_two: RouteLeadFrame


@dataclass
class RouteScanResult:
  route_id: int
  metadata: RouteMetadata
  observed_params: dict[str, str]
  frames: list[EpisodeFrame]


@dataclass(frozen=True)
class EpisodeCandidate:
  route_id: int
  route_key: str
  episode_type: str
  event_t_s: float
  seg_start: int
  seg_end: int
  t_start_s: float
  t_end_s: float
  confidence: float
  rank_score: float
  metrics: dict[str, Any]
  notes_json: dict[str, Any]
  frames: list[EpisodeFrame]

  @property
  def episode_key(self) -> str:
    return f"route{self.route_id}:{self.route_key}:{self.episode_type}:{int(round(self.t_start_s * 1000.0))}:{int(round(self.t_end_s * 1000.0))}:{EXTRACTOR_VERSION}"


def index_ev6_routes(conn, roots: list[str | Path] | None = None) -> list[dict[str, Any]]:
  grouped_segments = discover_ev6_route_groups(roots)
  indexed_routes = []
  for (_, route_key), segments in sorted(grouped_segments.items(), key=lambda item: (item[0][0], item[0][1])):
    metadata = read_route_metadata(segments)
    if metadata is None or metadata.car_fingerprint != "KIA_EV6":
      continue

    route_id = upsert_route(conn, RouteCatalogRecord(**metadata.__dict__))
    for segment in segments:
      start_time, end_time = read_segment_time_bounds(segment.rlog_path)
      upsert_segment(conn, route_id, SegmentCatalogRecord(
        seg_idx=segment.seg_idx,
        rlog_path=str(segment.rlog_path),
        qlog_path=str(segment.qlog_path) if segment.qlog_path is not None else None,
        start_log_mono_time=start_time,
        end_log_mono_time=end_time,
      ))
    indexed_routes.append({
      "routeId": route_id,
      "routeKey": metadata.route_key,
      "topology": metadata.topology,
      "segmentCount": metadata.segment_count,
      "sourceRoot": metadata.source_root,
    })

  conn.commit()
  return indexed_routes


def discover_ev6_route_groups(roots: list[str | Path] | None = None) -> dict[tuple[str, str], list[DiscoveredSegment]]:
  route_roots = [Path(root) for root in roots] if roots is not None else list(DEFAULT_ROUTE_ROOTS)
  grouped: dict[tuple[str, str], list[DiscoveredSegment]] = {}
  for root in route_roots:
    if not root.exists():
      continue
    for rlog_path in sorted(root.rglob("*.zst")):
      segment_identity = parse_segment_identity(rlog_path, root)
      if segment_identity is None:
        continue
      source_key, route_key, seg_idx, qlog_path = segment_identity
      grouped.setdefault((source_key, route_key), []).append(DiscoveredSegment(
        source_root=root,
        route_key=route_key,
        seg_idx=seg_idx,
        rlog_path=rlog_path,
        qlog_path=qlog_path,
      ))

  for key in grouped:
    grouped[key] = sorted(grouped[key], key=lambda segment: segment.seg_idx)
  return grouped


def parse_segment_identity(rlog_path: Path, root: Path) -> tuple[str, str, int, Path | None] | None:
  if not rlog_path.name.endswith((".zst", ".bz2")):
    return None
  if "rlog" not in rlog_path.name:
    return None

  parent_name = rlog_path.parent.name
  route_key = None
  seg_idx = None

  if parent_name.isdigit():
    route_key = rlog_path.parent.parent.name
    seg_idx = int(parent_name)
  else:
    match = re.match(r"(.+)--(\d+)$", parent_name)
    if match and rlog_path.name in ("rlog.zst", "rlog.bz2"):
      route_key = match.group(1)
      seg_idx = int(match.group(2))
    else:
      match = re.match(r"seg(\d+)$", parent_name)
      if match:
        route_key = rlog_path.parent.parent.name
        seg_idx = int(match.group(1))

  if route_key is None or seg_idx is None:
    file_match = re.match(r"(?:(.+)--)?(\d+)\.rlog\.(?:zst|bz2)$", rlog_path.name)
    if file_match:
      route_key = file_match.group(1) or rlog_path.parent.name
      seg_idx = int(file_match.group(2))

  if route_key is None or seg_idx is None:
    file_match = re.match(r"rlog_(\d+)\.(?:zst|bz2)$", rlog_path.name)
    if file_match:
      route_key = rlog_path.parent.name
      seg_idx = int(file_match.group(1))

  if route_key is None or seg_idx is None:
    if rlog_path.name in ("rlog.zst", "rlog.bz2"):
      route_key = rlog_path.parent.name
      seg_idx = 0
    else:
      return None

  qlog_path = _guess_neighbor_log(rlog_path, "qlog")
  return (str(root), route_key, seg_idx, qlog_path)


def read_route_metadata(segments: list[DiscoveredSegment]) -> RouteMetadata | None:
  if not segments:
    return None

  first_segment = segments[0]
  car_params = None
  for segment in segments:
    for msg in LogReader(str(segment.rlog_path)):
      if msg.which() == "carParams":
        car_params = msg.carParams
        break
    if car_params is not None:
      break

  if car_params is None:
    return None

  safety_param = None
  if len(car_params.safetyConfigs):
    safety_param = int(car_params.safetyConfigs[0].safetyParam)
    if int(car_params.safetyConfigs[0].safetyParamDEPRECATED) != 0:
      safety_param = int(car_params.safetyConfigs[0].safetyParamDEPRECATED)
  elif int(car_params.safetyParamDEPRECATED) != 0:
    safety_param = int(car_params.safetyParamDEPRECATED)

  topology = infer_topology(car_params.flags, safety_param)
  return RouteMetadata(
    source_root=str(first_segment.source_root),
    route_key=first_segment.route_key,
    car_fingerprint=str(car_params.carFingerprint),
    brand=str(car_params.brand),
    topology=topology,
    openpilot_longitudinal=bool(car_params.openpilotLongitudinalControl),
    radar_unavailable=bool(car_params.radarUnavailable),
    safety_param=safety_param,
    segment_count=len(segments),
    first_segment_path=str(first_segment.rlog_path),
    notes_json={"flags": int(car_params.flags)},
  )


def read_segment_time_bounds(rlog_path: Path) -> tuple[int | None, int | None]:
  first_time = None
  last_time = None
  for msg in LogReader(str(rlog_path)):
    if first_time is None:
      first_time = int(msg.logMonoTime)
    last_time = int(msg.logMonoTime)
  return first_time, last_time


def infer_topology(flags: int, safety_param: int | None) -> str | None:
  if flags & int(HyundaiFlags.CANFD_LKA_STEERING):
    return "lka"
  if safety_param is not None and safety_param & int(HyundaiSafetyFlags.CANFD_LKA_STEERING):
    return "lka"
  return "lfa"


def load_route_scan(conn, route_row) -> RouteScanResult:
  segment_rows = get_route_segments(conn, int(route_row["route_id"]))
  observed_params: dict[str, str] = {}
  latest_car_state = None
  latest_controls_state = None
  latest_selfdrive_state = None
  latest_car_control = None
  first_radar_time = None
  frames: list[EpisodeFrame] = []

  for segment_row in segment_rows:
    for msg in LogReader(segment_row["rlog_path"]):
      which = msg.which()
      if which == "carControlSP":
        observed_params.update({param.key: param.value for param in msg.carControlSP.params})
      elif which == "carState":
        latest_car_state = msg.carState
      elif which == "controlsState":
        latest_controls_state = msg.controlsState
      elif which == "selfdriveState":
        latest_selfdrive_state = msg.selfdriveState
      elif which == "carControl":
        latest_car_control = msg.carControl
      elif which == "radarState":
        if latest_car_state is None or latest_controls_state is None or latest_selfdrive_state is None or latest_car_control is None:
          continue
        if first_radar_time is None:
          first_radar_time = int(msg.logMonoTime)
        t_s = (int(msg.logMonoTime) - first_radar_time) / 1e9
        frames.append(EpisodeFrame(
          route_key=str(route_row["route_key"]),
          seg_idx=int(segment_row["seg_idx"]),
          log_mono_time=int(msg.logMonoTime),
          t_s=t_s,
          v_ego_mps=float(latest_car_state.vEgo),
          a_ego_mps2=float(latest_car_state.aEgo),
          cruise_speed_mps=float(latest_car_state.vCruise) / 3.6 if float(latest_car_state.vCruise) > 0.0 else float(latest_car_state.vEgo),
          long_active=bool(latest_car_control.longActive),
          long_control_state=str(latest_controls_state.longControlState),
          force_decel=bool(latest_controls_state.forceDecel),
          experimental_mode=bool(latest_selfdrive_state.experimentalMode),
          pitch_rad=float(latest_car_control.orientationNED[1]) if len(latest_car_control.orientationNED) > 1 else 0.0,
          lead_one=_lead_from_message(msg.radarState.leadOne),
          lead_two=_lead_from_message(msg.radarState.leadTwo),
        ))

  metadata = RouteMetadata(
    source_root=str(route_row["source_root"]),
    route_key=str(route_row["route_key"]),
    car_fingerprint=str(route_row["car_fingerprint"]),
    brand=str(route_row["brand"]),
    topology=str(route_row["topology"]) if route_row["topology"] is not None else None,
    openpilot_longitudinal=bool(route_row["openpilot_longitudinal"]),
    radar_unavailable=bool(route_row["radar_unavailable"]),
    safety_param=int(route_row["safety_param"]) if route_row["safety_param"] is not None else None,
    segment_count=int(route_row["segment_count"]),
    first_segment_path=str(route_row["first_segment_path"]),
    notes_json=_decode_json_column(route_row["notes_json"]),
  )
  return RouteScanResult(route_id=int(route_row["route_id"]), metadata=metadata, observed_params=observed_params, frames=frames)


def extract_ev6_episodes(conn,
                         *,
                         route_keys: list[str] | None = None,
                         bundle_root: str | Path = ".cache/longitudinal_harness/snapshots") -> list[dict[str, Any]]:
  bundle_root_path = Path(bundle_root)
  bundle_root_path.mkdir(parents=True, exist_ok=True)
  recorded = []
  for route_row in get_route_rows(conn, route_keys=route_keys):
    scan = load_route_scan(conn, route_row)
    clear_route_extractions(conn, scan.route_id)
    candidates = detect_episode_candidates(scan)
    for candidate in candidates:
      bundle_path = write_episode_bundle(scan, candidate, bundle_root_path)
      episode_id = upsert_episode(conn, EpisodeCatalogRecord(
        route_id=scan.route_id,
        episode_key=candidate.episode_key,
        episode_type=candidate.episode_type,
        seg_start=candidate.seg_start,
        seg_end=candidate.seg_end,
        t_start_s=candidate.t_start_s,
        t_end_s=candidate.t_end_s,
        confidence=candidate.confidence,
        extractor_version=EXTRACTOR_VERSION,
        source_event_count=int(candidate.metrics.get("sourceEventCount", 1)),
        metrics_json=candidate.metrics,
        bundle_path=str(bundle_path),
        notes_json=candidate.notes_json,
      ))
      update_episode_bundle_path(conn, episode_id, str(bundle_path))
      snapshot_id = record_snapshot_bundle(conn, bundle_path, route_id=scan.route_id, episode_id=episode_id)
      recorded.append({
        "routeId": scan.route_id,
        "episodeId": episode_id,
        "snapshotId": snapshot_id,
        "episodeType": candidate.episode_type,
        "bundlePath": str(bundle_path),
        "confidence": candidate.confidence,
      })
  conn.commit()
  return recorded


def detect_episode_candidates(scan: RouteScanResult) -> list[EpisodeCandidate]:
  frames = scan.frames
  if not frames:
    return []

  candidates: list[EpisodeCandidate] = []
  candidates.extend(_detect_cutin(scan))
  candidates.extend(_detect_handoff(scan))
  candidates.extend(_detect_dropout(scan))
  candidates.extend(_detect_pullaway(scan))
  candidates.extend(_detect_approach(scan))
  candidates = _dedupe_candidates(candidates)
  candidates.sort(key=lambda candidate: (candidate.t_start_s, candidate.episode_type))
  return candidates


def write_episode_bundle(scan: RouteScanResult, candidate: EpisodeCandidate, bundle_root: Path) -> Path:
  episode_root = bundle_root / f"route_{scan.route_id}_{scan.metadata.route_key}" / f"{candidate.episode_type}_{int(round(candidate.event_t_s * 1000.0)):09d}"
  timeline = []
  prev_status = {"leadOne": False, "leadTwo": False}
  event_marked = False
  for frame in candidate.frames:
    lead_one = _directive_from_frame(frame.lead_one, not prev_status["leadOne"] and frame.lead_one.status)
    lead_two = _directive_from_frame(frame.lead_two, not prev_status["leadTwo"] and frame.lead_two.status)
    event_name = None
    if not event_marked and frame.t_s >= candidate.event_t_s:
      event_name = _event_name_for_episode(candidate.episode_type)
      event_marked = True
    timeline.append(StepInput(
      t_s=float(frame.t_s - candidate.frames[0].t_s),
      cruise_speed_mps=frame.cruise_speed_mps,
      lead_one=lead_one,
      lead_two=lead_two,
      event=event_name,
      note=f"{scan.metadata.route_key}:{candidate.episode_type}",
      pitch_rad=frame.pitch_rad,
      force_decel=frame.force_decel,
      experimental_mode=frame.experimental_mode,
    ))
    prev_status["leadOne"] = frame.lead_one.status
    prev_status["leadTwo"] = frame.lead_two.status

  controller_mode = "shaped" if (not scan.metadata.radar_unavailable and int(scan.observed_params.get("HyundaiLongitudinalTuning", "0")) != 0) else "passthrough"
  bundle = SnapshotBundle(
    path=episode_root,
    vehicle={
      "name": f"{scan.metadata.route_key}_{candidate.episode_type}",
      "routeId": scan.route_id,
      "routeKey": scan.metadata.route_key,
      "episodeType": candidate.episode_type,
      "topology": scan.metadata.topology,
      "controllerMode": controller_mode,
      "openpilotLongitudinalControl": scan.metadata.openpilot_longitudinal,
      "radarUnavailable": scan.metadata.radar_unavailable,
      "safetyParam": scan.metadata.safety_param,
      "sourceRoot": scan.metadata.source_root,
      "segStart": candidate.seg_start,
      "segEnd": candidate.seg_end,
      "tStartS": candidate.t_start_s,
      "tEndS": candidate.t_end_s,
      "confidence": candidate.confidence,
      "metrics": candidate.metrics,
    },
    params=dict(scan.observed_params),
    timeline=timeline,
    initial_speed_mps=candidate.frames[0].v_ego_mps,
    initial_accel_mps2=candidate.frames[0].a_ego_mps2,
    name=f"{scan.metadata.route_key}_{candidate.episode_type}",
  )
  write_snapshot_bundle(bundle, episode_root)
  return episode_root


def _detect_cutin(scan: RouteScanResult) -> list[EpisodeCandidate]:
  candidates = []
  last_event_t = -math.inf
  for idx in range(15, len(scan.frames) - 15):
    frame = scan.frames[idx]
    primary_slot, primary = _primary_lead_with_slot(frame)
    if primary_slot is None or primary is None or primary.d_rel_m is None or primary.v_rel_mps is None or primary.d_rel_m > 35.0:
      continue
    prev_window = scan.frames[idx - 15:idx]
    long_prev_window = scan.frames[max(0, idx - 40):idx]
    future_window = scan.frames[idx:min(len(scan.frames), idx + 15)]
    if _lead_presence_fraction(prev_window) > 0.05:
      continue
    if _lead_presence_fraction(long_prev_window) > 0.20:
      continue
    if _lead_presence_fraction(future_window) < 0.80:
      continue
    if _slot_presence_fraction(future_window, primary_slot) < 0.70:
      continue
    if frame.t_s - last_event_t < COOLDOWN_BY_TYPE_S["cutin"]:
      continue
    if max(0.0, -(primary.v_rel_mps or 0.0)) < 0.25:
      continue
    confidence = min(1.0, 0.60 + max(0.0, (35.0 - primary.d_rel_m) / 30.0) + max(0.0, -(primary.v_rel_mps)) / 10.0)
    rank_score = (35.0 - primary.d_rel_m) + (max(0.0, -(primary.v_rel_mps)) * 4.0) + (_lead_presence_fraction(future_window) * 5.0)
    candidates.append(_make_candidate(scan, idx, "cutin", confidence, {
      "leadGapM": primary.d_rel_m,
      "closingSpeedMps": max(0.0, -(primary.v_rel_mps or 0.0)),
    }, rank_score=rank_score))
    last_event_t = frame.t_s
  return candidates


def _detect_handoff(scan: RouteScanResult) -> list[EpisodeCandidate]:
  candidates = []
  last_event_t = -math.inf
  pre_window_frames = 10
  post_window_frames = 10
  min_overlap_fraction = 0.70
  min_primary_fraction = 0.80
  min_ego_speed_mps = 12.0
  min_gap_spread_m = 3.0
  min_incoming_slower_mps = 1.0
  for idx in range(pre_window_frames, len(scan.frames) - post_window_frames):
    frame = scan.frames[idx]
    if frame.t_s - last_event_t < COOLDOWN_BY_TYPE_S["handoff"]:
      continue
    if frame.v_ego_mps < min_ego_speed_mps:
      continue
    prev_slot, _ = _primary_lead_with_slot(scan.frames[idx - 1])
    curr_slot, _ = _primary_lead_with_slot(frame)
    if prev_slot is None or curr_slot is None or prev_slot == curr_slot:
      continue
    overlap_window = scan.frames[idx - pre_window_frames:idx + post_window_frames]
    if _both_leads_presence_fraction(overlap_window) < min_overlap_fraction:
      continue
    pre_window = scan.frames[idx - pre_window_frames:idx]
    post_window = scan.frames[idx:idx + post_window_frames]
    if _primary_slot_fraction(pre_window, prev_slot) < min_primary_fraction:
      continue
    if _primary_slot_fraction(post_window, curr_slot) < min_primary_fraction:
      continue
    if _slot_presence_fraction(pre_window, curr_slot) < min_overlap_fraction:
      continue

    outgoing_pre_gaps = _slot_metric_values(pre_window, prev_slot, "d_rel_m")
    incoming_pre_gaps = _slot_metric_values(pre_window, curr_slot, "d_rel_m")
    outgoing_pre_speeds = _slot_metric_values(pre_window, prev_slot, "v_lead_mps")
    incoming_pre_speeds = _slot_metric_values(pre_window, curr_slot, "v_lead_mps")
    if not outgoing_pre_gaps or not incoming_pre_gaps or not outgoing_pre_speeds or not incoming_pre_speeds:
      continue
    gap_spread_m = _mean(incoming_pre_gaps) - _mean(outgoing_pre_gaps)
    incoming_slower_mps = _mean(outgoing_pre_speeds) - _mean(incoming_pre_speeds)
    if gap_spread_m < min_gap_spread_m:
      continue
    if incoming_slower_mps < min_incoming_slower_mps:
      continue

    confidence = min(1.0, 0.65 + min(0.20, gap_spread_m / 20.0) + min(0.15, incoming_slower_mps / 6.0))
    rank_score = gap_spread_m + incoming_slower_mps * 4.0 + frame.v_ego_mps * 0.1
    candidates.append(_make_candidate(scan, idx, "handoff", confidence, {
      "outgoingSlot": prev_slot,
      "incomingSlot": curr_slot,
      "outgoingLeadGapM": _slot_metric(frame, prev_slot, "d_rel_m"),
      "incomingLeadGapM": _slot_metric(frame, curr_slot, "d_rel_m"),
      "prerevealGapSpreadM": gap_spread_m,
      "incomingLeadSpeedDeltaMps": incoming_slower_mps,
    }, rank_score=rank_score))
    last_event_t = frame.t_s
  return candidates


def _detect_dropout(scan: RouteScanResult) -> list[EpisodeCandidate]:
  candidates = []
  last_event_t = -math.inf
  for idx in range(10, len(scan.frames) - 10):
    frame = scan.frames[idx]
    if _primary_lead(frame) is not None:
      continue
    if frame.t_s - last_event_t < COOLDOWN_BY_TYPE_S["dropout"]:
      continue
    prev_window = scan.frames[idx - 10:idx]
    if _lead_presence_fraction(prev_window) < 0.90:
      continue
    reacquire = None
    for future_idx in range(idx + 1, min(len(scan.frames), idx + 20)):
      if _primary_lead(scan.frames[future_idx]) is not None:
        reacquire = future_idx
        break
    if reacquire is None:
      continue
    dropout_duration = scan.frames[reacquire].t_s - frame.t_s
    if dropout_duration > 1.0 or dropout_duration < 0.15:
      continue
    reacquire_window = scan.frames[reacquire:min(len(scan.frames), reacquire + 10)]
    if _lead_presence_fraction(reacquire_window) < 0.80:
      continue
    confidence = min(1.0, 0.65 + (dropout_duration / 1.5) + (_lead_presence_fraction(reacquire_window) * 0.15))
    rank_score = dropout_duration * 20.0 + (_lead_presence_fraction(reacquire_window) * 5.0)
    candidates.append(_make_candidate(scan, idx, "dropout", confidence, {
      "dropoutDurationS": dropout_duration,
    }, custom_end_idx=reacquire, rank_score=rank_score))
    last_event_t = frame.t_s
  return candidates


def _detect_pullaway(scan: RouteScanResult) -> list[EpisodeCandidate]:
  candidates = []
  last_event_t = -math.inf
  for idx in range(20, len(scan.frames) - 40):
    frame = scan.frames[idx]
    primary = _primary_lead(frame)
    if primary is None or primary.d_rel_m is None or primary.v_rel_mps is None:
      continue
    if frame.t_s - last_event_t < COOLDOWN_BY_TYPE_S["pullaway"]:
      continue
    if not (10.0 <= primary.d_rel_m <= 80.0 and primary.v_rel_mps > 1.0):
      continue
    prev_window = scan.frames[idx - 20:idx]
    future_window = scan.frames[idx: idx + 40]
    if _lead_presence_fraction(prev_window) < 0.80 or _lead_presence_fraction(future_window) < 0.80:
      continue
    prev_primary = [_primary_lead(sample) for sample in prev_window]
    future_primary = [_primary_lead(sample) for sample in future_window]
    prev_v_rel = [lead.v_rel_mps for lead in prev_primary if lead and lead.v_rel_mps is not None]
    future_gaps = [lead.d_rel_m for lead in future_primary if lead and lead.d_rel_m is not None]
    if not prev_v_rel or max(abs(value) for value in prev_v_rel) > 1.0:
      continue
    if not future_gaps or max(future_gaps) - primary.d_rel_m < 8.0:
      continue
    gap_growth_m = max(future_gaps) - primary.d_rel_m
    confidence = min(1.0, 0.60 + min(0.20, primary.v_rel_mps / 6.0) + min(0.20, gap_growth_m / 25.0))
    rank_score = gap_growth_m + primary.v_rel_mps * 4.0
    candidates.append(_make_candidate(scan, idx, "pullaway", confidence, {
      "openingSpeedMps": primary.v_rel_mps,
      "gapGrowthM": gap_growth_m,
    }, rank_score=rank_score))
    last_event_t = frame.t_s
  return candidates


def _detect_approach(scan: RouteScanResult) -> list[EpisodeCandidate]:
  candidates = []
  last_event_t = -math.inf
  for idx in range(10, len(scan.frames) - 30):
    frame = scan.frames[idx]
    primary = _primary_lead(frame)
    if primary is None or primary.d_rel_m is None or primary.v_rel_mps is None:
      continue
    if frame.t_s - last_event_t < COOLDOWN_BY_TYPE_S["approach"]:
      continue
    if not (20.0 <= primary.d_rel_m <= 120.0 and primary.v_rel_mps < -1.0):
      continue
    prev_window = scan.frames[max(0, idx - 10):idx]
    future_window = scan.frames[idx: idx + 30]
    if _lead_presence_fraction(prev_window) < 0.80 or _lead_presence_fraction(future_window) < 0.80:
      continue
    future_primary = [_primary_lead(sample) for sample in future_window]
    future_gaps = [lead.d_rel_m for lead in future_primary if lead and lead.d_rel_m is not None]
    if not future_gaps or primary.d_rel_m - min(future_gaps) < 12.0:
      continue
    gap_reduction_m = primary.d_rel_m - min(future_gaps)
    confidence = min(1.0, 0.55 + min(0.20, gap_reduction_m / 30.0) + min(0.20, max(0.0, -primary.v_rel_mps) / 8.0))
    rank_score = gap_reduction_m + max(0.0, -primary.v_rel_mps) * 4.0
    candidates.append(_make_candidate(scan, idx, "approach", confidence, {
      "initialGapM": primary.d_rel_m,
      "minGapM": min(future_gaps),
      "closingSpeedMps": max(0.0, -primary.v_rel_mps),
    }, rank_score=rank_score))
    last_event_t = frame.t_s
  return candidates


def _make_candidate(scan: RouteScanResult,
                    event_idx: int,
                    episode_type: str,
                    confidence: float,
                    metrics: dict[str, Any],
                    *,
                    custom_end_idx: int | None = None,
                    rank_score: float | None = None) -> EpisodeCandidate:
  event_frame = scan.frames[event_idx]
  pre_s, post_s = WINDOWS_BY_TYPE[episode_type]
  start_t = max(0.0, event_frame.t_s + pre_s)
  end_t = event_frame.t_s + post_s
  if custom_end_idx is not None:
    end_t = max(end_t, scan.frames[custom_end_idx].t_s + 1.0)

  start_idx = next(idx for idx, frame in enumerate(scan.frames) if frame.t_s >= start_t)
  end_idx = len(scan.frames) - 1
  for idx in range(event_idx, len(scan.frames)):
    if scan.frames[idx].t_s >= end_t:
      end_idx = idx
      break

  episode_frames = scan.frames[start_idx:end_idx + 1]
  summary = summarize_episode_frames(episode_frames)
  summary.update(metrics)
  candidate_rank_score = confidence if rank_score is None else rank_score
  summary["candidateScore"] = candidate_rank_score
  return EpisodeCandidate(
    route_id=scan.route_id,
    route_key=scan.metadata.route_key,
    episode_type=episode_type,
    event_t_s=event_frame.t_s,
    seg_start=episode_frames[0].seg_idx,
    seg_end=episode_frames[-1].seg_idx,
    t_start_s=episode_frames[0].t_s,
    t_end_s=episode_frames[-1].t_s,
    confidence=confidence,
    rank_score=candidate_rank_score,
    metrics=summary,
    notes_json={"eventLogMonoTime": event_frame.log_mono_time},
    frames=episode_frames,
  )


def summarize_episode_frames(frames: list[EpisodeFrame]) -> dict[str, Any]:
  primary_gaps = []
  lead_switch_count = 0
  prev_slot = None
  for frame in frames:
    slot, lead = _primary_lead_with_slot(frame)
    if lead is not None and lead.d_rel_m is not None:
      primary_gaps.append(lead.d_rel_m)
    if slot is not None and prev_slot is not None and slot != prev_slot:
      lead_switch_count += 1
    if slot is not None:
      prev_slot = slot
  return {
    "minGapM": min(primary_gaps) if primary_gaps else None,
    "maxGapM": max(primary_gaps) if primary_gaps else None,
    "leadSwitchCount": lead_switch_count,
    "sourceEventCount": lead_switch_count + 1,
  }


def _dedupe_candidates(candidates: list[EpisodeCandidate]) -> list[EpisodeCandidate]:
  accepted: list[EpisodeCandidate] = []
  for candidate in sorted(candidates, key=lambda item: (item.episode_type, -item.rank_score, -item.confidence, item.event_t_s)):
    if any(_should_suppress_candidate(candidate, existing) for existing in accepted):
      continue
    accepted.append(candidate)
  return accepted


def _should_suppress_candidate(candidate: EpisodeCandidate, existing: EpisodeCandidate) -> bool:
  if candidate.episode_type != existing.episode_type:
    return False
  if abs(candidate.event_t_s - existing.event_t_s) <= SUPPRESSION_BY_TYPE_S[candidate.episode_type]:
    return True
  overlap_s = min(candidate.t_end_s, existing.t_end_s) - max(candidate.t_start_s, existing.t_start_s)
  if overlap_s <= 0.0:
    return False
  shorter_window_s = min(candidate.t_end_s - candidate.t_start_s, existing.t_end_s - existing.t_start_s)
  return overlap_s >= max(1.5, 0.35 * shorter_window_s)


def _lead_presence_fraction(frames: list[EpisodeFrame]) -> float:
  if not frames:
    return 0.0
  return sum(1 for frame in frames if _primary_lead(frame) is not None) / len(frames)


def _both_leads_presence_fraction(frames: list[EpisodeFrame]) -> float:
  if not frames:
    return 0.0
  return sum(
    1
    for frame in frames
    if frame.lead_one.status and frame.lead_one.d_rel_m is not None and frame.lead_two.status and frame.lead_two.d_rel_m is not None
  ) / len(frames)


def _slot_presence_fraction(frames: list[EpisodeFrame], slot_name: str) -> float:
  if not frames:
    return 0.0
  if slot_name == "leadOne":
    present = sum(1 for frame in frames if frame.lead_one.status)
  else:
    present = sum(1 for frame in frames if frame.lead_two.status)
  return present / len(frames)


def _primary_slot_fraction(frames: list[EpisodeFrame], slot_name: str) -> float:
  if not frames:
    return 0.0
  return sum(1 for frame in frames if _primary_lead_with_slot(frame)[0] == slot_name) / len(frames)


def _slot_metric_values(frames: list[EpisodeFrame], slot_name: str, metric_name: str) -> list[float]:
  values = []
  for frame in frames:
    value = _slot_metric(frame, slot_name, metric_name)
    if value is not None:
      values.append(value)
  return values


def _slot_metric(frame: EpisodeFrame, slot_name: str, metric_name: str) -> float | None:
  lead = frame.lead_one if slot_name == "leadOne" else frame.lead_two
  return getattr(lead, metric_name) if lead.status else None


def _mean(values: list[float]) -> float:
  return sum(values) / len(values)


def _lead_from_message(lead_msg) -> RouteLeadFrame:
  return RouteLeadFrame(
    status=bool(lead_msg.status),
    d_rel_m=float(lead_msg.dRel) if bool(lead_msg.status) else None,
    v_rel_mps=float(lead_msg.vRel) if bool(lead_msg.status) else None,
    v_lead_mps=float(lead_msg.vLead) if bool(lead_msg.status) else None,
    a_lead_k_mps2=float(lead_msg.aLeadK) if bool(lead_msg.status) else None,
    v_lead_k_mps=float(lead_msg.vLeadK) if bool(lead_msg.status) else None,
    model_prob=float(lead_msg.modelProb),
    y_rel_m=float(lead_msg.yRel),
    d_path_m=float(lead_msg.dPath) if bool(lead_msg.status) else None,
    v_lat_mps=float(lead_msg.vLat),
    radar=bool(lead_msg.radar),
    radar_track_id=int(lead_msg.radarTrackId),
  )


def _directive_from_frame(lead: RouteLeadFrame, acquisition_reset: bool) -> LeadDirective:
  return LeadDirective(
    status=lead.status,
    v_lead_mps=lead.v_lead_mps or 0.0,
    model_prob_target=lead.model_prob,
    measured_d_rel_m=lead.d_rel_m,
    measured_v_rel_mps=lead.v_rel_mps,
    a_lead_k_mps2=lead.a_lead_k_mps2,
    v_lead_k_mps=lead.v_lead_k_mps,
    y_rel_m=lead.y_rel_m,
    d_path_m=lead.d_path_m,
    v_lat_mps=lead.v_lat_mps,
    fcw=False,
    radar=lead.radar,
    radar_track_id=lead.radar_track_id,
    acquisition_reset=acquisition_reset,
  )


def _event_name_for_episode(episode_type: str) -> str:
  return {
    "pullaway": "pullaway_start",
    "cutin": "lead_reveal",
    "handoff": "handoff_reveal",
    "dropout": "dropout_start",
    "approach": "approach_start",
  }.get(episode_type, episode_type)


def _primary_lead(frame: EpisodeFrame) -> RouteLeadFrame | None:
  _, lead = _primary_lead_with_slot(frame)
  return lead


def _primary_lead_with_slot(frame: EpisodeFrame) -> tuple[str | None, RouteLeadFrame | None]:
  leads = []
  if frame.lead_one.status and frame.lead_one.d_rel_m is not None:
    leads.append(("leadOne", frame.lead_one))
  if frame.lead_two.status and frame.lead_two.d_rel_m is not None:
    leads.append(("leadTwo", frame.lead_two))
  if not leads:
    return None, None
  return min(leads, key=lambda item: item[1].d_rel_m if item[1].d_rel_m is not None else float("inf"))


def _guess_neighbor_log(rlog_path: Path, prefix: str) -> Path | None:
  for candidate_name in (
    rlog_path.name.replace("rlog", prefix),
    rlog_path.name.replace(".rlog", f".{prefix}"),
  ):
    candidate = rlog_path.with_name(candidate_name)
    if candidate.exists():
      return candidate
  return None


def _decode_json_column(value: Any) -> dict[str, Any]:
  if isinstance(value, dict):
    return value
  if isinstance(value, str) and value:
    try:
      decoded = json.loads(value)
    except json.JSONDecodeError:
      return {}
    return decoded if isinstance(decoded, dict) else {}
  return {}
