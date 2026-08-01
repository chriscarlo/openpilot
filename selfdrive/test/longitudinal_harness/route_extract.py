from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
import warnings
from typing import Any

from openpilot.tools.lib.logreader import LogReader

from opendbc.car.hyundai.values import HyundaiFlags, HyundaiSafetyFlags
from openpilot.selfdrive.controls.radard import add_path_relative_lead_metrics, get_RadarState_from_vision

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
from .config import captured_param_manifest, DEFAULT_PARAM_VALUES
from .inputs import LeadDirective, SnapshotBundle, StepInput, serialize_model_frame, write_snapshot_bundle
from .planner_state import build_route_start_initialization_claim, ROUTE_START_DIAGNOSTIC_REASON
from .replay_contracts import (
  is_supported_radard_replay_version,
  PLANNER_REPLAY_INPUTS_VERSION,
)


EXTRACTOR_VERSION = "ev6_v12_driver_mark"
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
  "false_closing": (-15.0, 5.0),
  # A human press is a reaction, so the interesting build-up is already behind
  # the driver by the time the thumb lands. The long pre-roll is deliberate.
  "driver_mark": (-20.0, 6.0),
}
COOLDOWN_BY_TYPE_S = {
  "approach": 6.0,
  "pullaway": 6.0,
  "cutin": 5.0,
  "handoff": 5.0,
  "dropout": 4.0,
  "false_closing": 5.0,
}
SUPPRESSION_BY_TYPE_S = {
  "approach": 10.0,
  "pullaway": 10.0,
  "cutin": 8.0,
  "handoff": 6.0,
  "dropout": 6.0,
  "false_closing": 6.0,
  # "driver_mark" is deliberately absent, like it is from COOLDOWN_BY_TYPE_S.
  # A time radius here is measured on event_t_s, which for a driver mark is the
  # ANCHOR FRAME's route clock, not the press clock -- see
  # _should_suppress_candidate. Driver marks dedupe on press identity instead,
  # and the only echo radius that exists is DRIVER_MARK_ECHO_WINDOW_S, applied by
  # collapse_driver_mark_echoes on the exact press logMonoTime.
}
MAX_EXACT_REPLAY_FRAME_GAP_S = 0.075
DRIVER_MARK_SERVICES = ("bookmarkButton", "userBookmark")
# The only service a flag press is actually proven by; see is_driver_mark_press_group.
DRIVER_MARK_PRESS_SERVICE = "bookmarkButton"
DRIVER_MARK_ECHO_WINDOW_S = 0.5
DRIVER_MARK_FRAME_TOLERANCE_S = 1.0
DRIVER_MARK_PAYLOAD_WINDOW_S = 2.0
LONGFLAG_PREFIX = "LONGFLAG "
RADARD_DEPENDENCY_WARMUP_S = 15.0
LONGITUDINAL_PLAN_SP_PAIR_MAX_NS = 20_000_000
MPH_TO_MPS = 0.44704
class EpisodeNotReplayableError(ValueError):
  """A detected episode lacks required recorded dependencies for faithful replay."""
PLANNER_CRITICAL_SERVICES = (
  "carState",
  "controlsState",
  "carControl",
  "selfdriveState",
  "modelV2",
  "radarState",
)
PARAM_CHANGE_NEAR_MARGIN_NS = 75_000_000


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
class CachedCarState:
  v_ego_mps: float
  a_ego_mps2: float
  v_cruise_kph: float
  gas_pressed: bool
  standstill: bool


@dataclass(frozen=True)
class CachedLongitudinalPlanSP:
  log_mono_time_ns: int
  slc_active: bool
  slc_state: str
  slc_speed_limit_mps: float
  slc_speed_limit_offset_mps: float
  vtsc_state: str
  vtsc_velocity_mps: float
  object_hazard_active: bool
  replay_inputs_valid: bool = False
  replay_inputs_version: int = 0
  replay_effective_cruise_mps: float = 0.0
  replay_plan_log_mono_time_ns: int = 0
  replay_input_clocks_ns: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CachedLongitudinalPlan:
  log_mono_time_ns: int
  model_mono_time_ns: int
  solver_execution_time_s: float
  radar_state_mono_time_ns: int
  v_cruise_deprecated_mps: float
  a_target_mps2: float
  source: str
  plan_sp: CachedLongitudinalPlanSP | None = None


@dataclass(frozen=True)
class PlannerRadarAssociation:
  target_log_mono_time_ns: int | None
  candidate_log_mono_times_ns: tuple[int, ...]
  resolution: str
  reason: str


@dataclass(frozen=True)
class EffectiveCruiseContext:
  speed_mps: float | None
  limiter: str | None
  provenance: str | None
  status: str
  reason: str


@dataclass(frozen=True)
class PlannerContextResolution:
  status: str
  reason: str
  inputs: dict[str, Any]
  service_log_mono_time_ns: dict[str, int]
  service_provenance: dict[str, Any]


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
  raw_lead_one: RouteLeadFrame | None
  raw_lead_two: RouteLeadFrame | None
  raw_model: dict[str, Any] | None
  model_v2_log_mono_time_ns: int | None
  car_state_log_mono_time_ns: int | None
  live_tracks_log_mono_time_ns: int | None
  radard_service_association_status: str
  radard_service_association_provenance: dict[str, Any]
  radard_gate_eligible: bool
  radar_state_log_mono_time_ns: int
  longitudinal_plan_log_mono_time_ns: int | None
  longitudinal_plan_solver_execution_time_s: float | None
  planner_radar_state_log_mono_time_ns: int | None
  planner_radar_state_candidates_ns: tuple[int, ...]
  planner_radar_resolution: str
  planner_radar_reason: str
  planner_inputs: dict[str, Any]
  planner_service_log_mono_time_ns: dict[str, int]
  planner_service_association_provenance: dict[str, Any]
  recorded_effective_cruise_mps: float | None
  recorded_effective_cruise_limiter: str | None
  recorded_effective_cruise_provenance: str | None
  recorded_effective_cruise_status: str
  planner_context_status: str
  planner_context_reason: str
  gas_pressed: bool
  personality: int
  planner_accel_mps2: float | None
  planner_source: str | None
  params_snapshot: dict[str, str]
  param_updates: dict[str, str]


@dataclass(frozen=True)
class DriverMarkPress:
  """One deliberate driver flag press, already collapsed across its service echo."""
  route_key: str
  seg_idx: int
  press_log_mono_time_ns: int
  t_s: float | None
  services: tuple[str, ...]
  payload: dict[str, Any] | None
  frame_index: int | None
  frame_log_mono_time_ns: int | None
  frame_gap_s: float | None
  status: str

  @property
  def mark_id(self) -> str:
    return f"{self.route_key}--{self.seg_idx}--{self.press_log_mono_time_ns}"


@dataclass
class RouteScanResult:
  route_id: int
  metadata: RouteMetadata
  observed_params: dict[str, str]
  frames: list[EpisodeFrame]
  service_join_diagnostics: dict[str, Any] = field(default_factory=dict)
  planner_route_start_provenance: dict[str, Any] = field(default_factory=dict)
  driver_marks: list[DriverMarkPress] = field(default_factory=list)
  longflag_payloads: list[dict[str, Any]] = field(default_factory=list)


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
  # Appended verbatim to episode_key. Heuristic detectors leave this None: their
  # window bounds plus per-type overlap suppression already make the key unique.
  # A driver mark cannot rely on either -- overlap suppression is exempted for
  # it on purpose -- so it carries its press logMonoTime here. Without it, two
  # presses whose windows clamp or run off the end of the recorded frames
  # resolve to the same start/end frame and the second upsert_episode silently
  # overwrites the first press's row.
  key_discriminator: str | None = None

  @property
  def episode_key(self) -> str:
    start_ms = int(round(self.t_start_s * 1000.0))
    end_ms = int(round(self.t_end_s * 1000.0))
    key = f"route{self.route_id}:{self.route_key}:{self.episode_type}:{start_ms}:{end_ms}:{EXTRACTOR_VERSION}"
    return key if self.key_discriminator is None else f"{key}:{self.key_discriminator}"


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
  car_params_sp = None
  init_provenance: dict[str, Any] | None = None
  for segment in segments:
    for msg in LogReader(str(segment.rlog_path)):
      if msg.which() == "initData" and init_provenance is None:
        init_provenance = _extract_init_provenance(msg.initData)
      elif msg.which() == "carParams" and car_params is None:
        car_params = msg.carParams
      elif msg.which() == "carParamsSP" and car_params_sp is None:
        car_params_sp = msg.carParamsSP
      if car_params is not None and car_params_sp is not None and init_provenance is not None:
        break
    if car_params is not None and car_params_sp is not None and init_provenance is not None:
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
  notes_json: dict[str, Any] = {
    "flags": int(car_params.flags),
    "pcmCruise": bool(car_params.pcmCruise),
    "longitudinalActuatorDelay": float(car_params.longitudinalActuatorDelay),
    "vEgoStopping": float(car_params.vEgoStopping),
    "vEgoStarting": float(car_params.vEgoStarting),
    "stoppingDecelRate": float(car_params.stoppingDecelRate),
    "startAccel": float(car_params.startAccel),
    "startingState": bool(car_params.startingState),
    **(init_provenance or {}),
  }
  if car_params_sp is not None:
    notes_json.update({
      "spFlags": int(car_params_sp.flags),
      "spSafetyParam": int(car_params_sp.safetyParam),
    })
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
    notes_json=notes_json,
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


@dataclass(frozen=True)
class RouteReplayIndex:
  car_state_by_mono_time: dict[int, CachedCarState]
  planner_inputs_by_service: dict[str, dict[int, dict[str, Any]]]
  live_tracks_mono_times: tuple[int, ...]
  radar_state_mono_times: tuple[int, ...]
  plans_by_model_mono_time: dict[int, CachedLongitudinalPlan]
  param_change_mono_times_ns: tuple[int, ...]
  rti_zero_threats_proven: bool
  rti_state_count: int
  planner_input_mono_times_by_service: dict[str, tuple[int, ...]] = field(default_factory=dict)
  conflicting_input_mono_times_by_service: dict[str, tuple[int, ...]] = field(default_factory=dict)
  identical_duplicate_counts_by_service: dict[str, int] = field(default_factory=dict)
  planner_route_start_provenance: dict[str, Any] = field(default_factory=dict)


def _build_planner_route_start_provenance(
  *,
  segment_rows,
  capture_first_log_mono_time_ns: int | None,
  capture_last_log_mono_time_ns: int | None,
  start_of_route_mono_times_ns: list[int],
  process_identities: dict[str, set[tuple[int, int]]],
  manager_plannerd_observations: list[tuple[int, int, bool, bool]],
  model_v2_mono_times_ns: list[int],
  plans: list[CachedLongitudinalPlan],
  paired_plans: list[CachedLongitudinalPlan],
  plan_sp_messages: list[CachedLongitudinalPlanSP],
) -> dict[str, Any]:
  """Establish a diagnostic replay from the daemons' constructor boundary.

  A route slice cannot satisfy this contract. The capture must begin before
  both daemon PIDs, contain segment zero and every following segment, and show
  the first planner publication consuming the first logged model publication.
  This is intentionally stricter than numerical convergence, but it is not a
  formal checkpoint: rlogs lack writer-attested publication completeness.
  """
  reasons: list[str] = []
  segment_indices = sorted({int(row["seg_idx"]) for row in segment_rows})
  segments_contiguous = bool(
    segment_indices and
    segment_indices[0] == 0 and
    segment_indices == list(range(segment_indices[-1] + 1))
  )
  if not segments_contiguous:
    reasons.append("loaded route does not contain a contiguous segment-zero prefix")
  if capture_first_log_mono_time_ns is None or capture_last_log_mono_time_ns is None:
    reasons.append("capture has no logMonoTime bounds")
  if len(start_of_route_mono_times_ns) != 1:
    reasons.append("capture does not contain exactly one startOfRoute sentinel")

  planner_identities = process_identities.get("plannerd", set())
  radard_identities = process_identities.get("radard", set())
  if len(planner_identities) != 1:
    reasons.append("plannerd process lifetime is missing or non-unique")
  if len(radard_identities) != 1:
    reasons.append("radard process lifetime is missing or non-unique")
  planner_pid, planner_start_ns = next(iter(planner_identities), (0, 0))
  _, radard_start_ns = next(iter(radard_identities), (0, 0))

  first_plan = min(paired_plans, key=lambda plan: plan.log_mono_time_ns) if paired_plans else None
  first_model_ns = min(model_v2_mono_times_ns, default=0)
  first_radar_ns = 0
  if first_plan is None:
    reasons.append("capture has no longitudinalPlan publication")
  else:
    if first_plan.plan_sp is not None:
      first_radar_ns = int(first_plan.plan_sp.replay_input_clocks_ns.get("radarState", 0))
    if int(first_plan.model_mono_time_ns) != first_model_ns:
      reasons.append("first planner publication does not consume the first logged modelV2 publication")
    if planner_start_ns > 0 and int(first_plan.log_mono_time_ns) <= planner_start_ns:
      reasons.append("first planner publication does not follow the captured plannerd process start")

  paired_exactly = bool(
    plans and
    len(paired_plans) == len(plans) == len(plan_sp_messages) and
    len({int(plan.model_mono_time_ns) for plan in plans}) == len(plans) and
    all(
      plan.plan_sp is not None and
      plan.plan_sp.replay_inputs_valid and
      plan.plan_sp.replay_inputs_version == PLANNER_REPLAY_INPUTS_VERSION and
      plan.plan_sp.replay_plan_log_mono_time_ns == plan.log_mono_time_ns and
      plan.plan_sp.replay_input_clocks_ns.get("modelV2") == plan.model_mono_time_ns
      for plan in paired_plans
    )
  )
  if not paired_exactly:
    reasons.append("planner publications are not one-to-one paired with exact replayInputs")

  manager_identity_exact = bool(
    planner_pid > 0 and
    any(
      pid == planner_pid and running and should_be_running and
      (first_plan is None or mono_time_ns <= first_plan.log_mono_time_ns)
      for mono_time_ns, pid, running, should_be_running in manager_plannerd_observations
    ) and
    all(
      not (running and should_be_running) or pid == planner_pid
      for _, pid, running, should_be_running in manager_plannerd_observations
    )
  )
  if not manager_identity_exact:
    reasons.append("managerState does not bind the captured planner publications to one plannerd PID")

  if (
    capture_first_log_mono_time_ns is not None and
    (planner_start_ns <= capture_first_log_mono_time_ns or radard_start_ns <= capture_first_log_mono_time_ns)
  ):
    reasons.append("capture does not begin before both daemon process starts")
  if first_plan is not None and any(plan.log_mono_time_ns < planner_start_ns for plan in plans):
    reasons.append("capture contains planner output predating the proven plannerd process")
  if first_radar_ns <= 0:
    reasons.append("first planner publication has no exact radarState dependency")

  proof = {
    "status": "diagnostic" if not reasons else "missing",
    "version": 1 if not reasons else 0,
    "formalFidelityEligible": False,
    "sourcePublicationCompletenessAttested": False,
    "loadedSegmentStart": segment_indices[0] if segment_indices else None,
    "loadedSegmentEnd": segment_indices[-1] if segment_indices else None,
    "segmentsContiguous": segments_contiguous,
    "captureFirstLogMonoTimeNs": capture_first_log_mono_time_ns,
    "captureLastLogMonoTimeNs": capture_last_log_mono_time_ns,
    "startOfRouteMonoTimeNs": start_of_route_mono_times_ns[0] if len(start_of_route_mono_times_ns) == 1 else None,
    "plannerProcessPid": planner_pid or None,
    "plannerProcessStartMonoTimeNs": planner_start_ns or None,
    "radardProcessStartMonoTimeNs": radard_start_ns or None,
    "plannerProcessUnique": len(planner_identities) == 1,
    "radardProcessUnique": len(radard_identities) == 1,
    "plannerManagerIdentityExact": manager_identity_exact,
    "plannerPublicationsPairedExactly": paired_exactly,
    "plannerPublicationCount": len(plans),
    "firstPlannerPlanMonoTimeNs": None if first_plan is None else int(first_plan.log_mono_time_ns),
    "firstPlannerModelMonoTimeNs": None if first_plan is None else int(first_plan.model_mono_time_ns),
    "firstPlannerRadarStateMonoTimeNs": first_radar_ns or None,
    "firstPlannerModelIsFirstLoggedModel": bool(first_plan is not None and int(first_plan.model_mono_time_ns) == first_model_ns),
    "reasons": [*reasons, *([] if reasons else [ROUTE_START_DIAGNOSTIC_REASON])],
  }
  return proof


def _enum_name(value: Any) -> str:
  text = str(value)
  return text.rsplit(".", 1)[-1]


def _has_planner_replay_inputs_v1(plan: CachedLongitudinalPlan | None) -> bool:
  return bool(
    plan is not None and
    plan.plan_sp is not None and
    plan.plan_sp.replay_inputs_valid and
    plan.plan_sp.replay_inputs_version == PLANNER_REPLAY_INPUTS_VERSION and
    plan.plan_sp.replay_plan_log_mono_time_ns == plan.log_mono_time_ns and
    int(plan.plan_sp.replay_input_clocks_ns.get("modelV2", 0)) == plan.model_mono_time_ns
  )


def _pair_longitudinal_plans_with_sp(
  plans: list[CachedLongitudinalPlan],
  plan_sp_messages: list[CachedLongitudinalPlanSP],
) -> list[CachedLongitudinalPlan]:
  """Pair a plan only when exactly one SP publication fits its writer window."""
  paired: list[CachedLongitudinalPlan] = []
  sorted_plans = sorted(plans, key=lambda entry: entry.log_mono_time_ns)
  sorted_sp = sorted(plan_sp_messages, key=lambda entry: entry.log_mono_time_ns)
  sp_times = [entry.log_mono_time_ns for entry in sorted_sp]
  for plan_idx, plan in enumerate(sorted_plans):
    start_ns = int(plan.log_mono_time_ns)
    end_ns = start_ns + LONGITUDINAL_PLAN_SP_PAIR_MAX_NS
    next_plan_time_ns = (
      int(sorted_plans[plan_idx + 1].log_mono_time_ns)
      if plan_idx + 1 < len(sorted_plans) else None
    )
    left = bisect_left(sp_times, start_ns)
    right = bisect_right(sp_times, end_ns)
    if next_plan_time_ns is not None:
      # A same-clock SP belongs to the next plan, never the previous one.
      right = min(right, bisect_left(sp_times, next_plan_time_ns))
    candidates = sorted_sp[left:right]
    # Duplicate/competing SP publications are indistinguishable in legacy logger
    # order. Refuse to attach either, because a guessed v1 writer contract would
    # otherwise make the downstream exactness checks fail open.
    plan_sp = candidates[0] if len(candidates) == 1 else None
    paired.append(replace(plan, plan_sp=plan_sp))
  return paired


def resolve_planner_radar_association(
  plan: CachedLongitudinalPlan,
  radar_state_mono_times: list[int] | tuple[int, ...],
  conflicting_radar_state_mono_times: list[int] | tuple[int, ...] = (),
) -> PlannerRadarAssociation:
  """Resolve the RadarD publication available to plannerd without output oracles.

  New logs carry an exact pointer in LongitudinalPlanSP.replayInputs. Deprecated
  longitudinalPlan fields are diagnostics only. Legacy logs can only be called unique when
  publication timing leaves one possible input: the last publication before the
  model poll plus publications that could have completed before solver start.
  """
  # RouteReplayIndex already owns a sorted, unique tuple. Preserve that fast
  # path: this resolver runs once per plan across routes with >100k frames.
  radar_times = (
    radar_state_mono_times
    if isinstance(radar_state_mono_times, tuple)
    else tuple(sorted(set(int(value) for value in radar_state_mono_times)))
  )
  conflicting_radar_times = frozenset(int(value) for value in conflicting_radar_state_mono_times)
  if _has_planner_replay_inputs_v1(plan):
    assert plan.plan_sp is not None
    pointer = int(plan.plan_sp.replay_input_clocks_ns.get("radarState", 0))
    if pointer <= 0:
      return PlannerRadarAssociation(
        None,
        (),
        "missing",
        "LongitudinalPlanSP.replayInputs v1 records no received radarState publication",
      )
    if pointer in conflicting_radar_times:
      return PlannerRadarAssociation(
        None,
        (pointer,),
        "conflict",
        f"v1 replay radarState clock {pointer} has conflicting duplicate payloads",
      )
    pointer_idx = bisect_left(radar_times, pointer)
    if pointer_idx < len(radar_times) and radar_times[pointer_idx] == pointer:
      return PlannerRadarAssociation(
        pointer,
        (pointer,),
        "exact",
        "LongitudinalPlanSP.replayInputs v1 exact radarState clock",
      )
    return PlannerRadarAssociation(
      pointer,
      (),
      "missing",
      f"v1 replay radarState clock {pointer} is absent from the loaded route",
    )

  model_time_ns = int(plan.model_mono_time_ns)
  solver_time_s = float(plan.solver_execution_time_s)
  solver_ns = 0 if not math.isfinite(solver_time_s) else max(0, int(round(solver_time_s * 1e9)))
  solver_start_ns = int(plan.log_mono_time_ns) - solver_ns

  model_idx = bisect_left(radar_times, model_time_ns)
  baseline = radar_times[model_idx - 1] if model_idx > 0 else None
  candidates: list[int] = []
  if baseline is not None:
    candidates.append(baseline)
  solver_end_idx = bisect_right(radar_times, solver_start_ns, lo=model_idx)
  later_candidates = list(radar_times[model_idx:solver_end_idx])
  candidates.extend(later_candidates)
  candidates = sorted(set(candidates))
  conflicting_candidates = tuple(candidate for candidate in candidates if candidate in conflicting_radar_times)
  if conflicting_candidates:
    return PlannerRadarAssociation(
      None,
      tuple(candidates),
      "conflict",
      f"legacy timing candidates include conflicting duplicate radarState payloads at {list(conflicting_candidates)}",
    )
  if baseline is None:
    return PlannerRadarAssociation(
      None,
      tuple(candidates),
      "missing",
      "legacy/default replay contract has no predecessor radar publication before model poll; segment-boundary input is unknown",
    )
  if len(candidates) == 1:
    return PlannerRadarAssociation(
      candidates[0],
      tuple(candidates),
      "legacy_timing_unique",
      "legacy publication timing leaves one candidate; deprecated pointer is not proof",
    )
  if len(candidates) > 1:
    return PlannerRadarAssociation(
      None,
      tuple(candidates),
      "legacy_ambiguous",
      "legacy timing permits multiple publications; planner output was not used to choose",
    )
  return PlannerRadarAssociation(None, (), "missing", "no radar publication was available by solver start")


def _param_bool(params: dict[str, str], key: str) -> bool | None:
  if key not in params:
    return None
  value = str(params[key]).strip().lower()
  if value in ("1", "true", "yes", "on"):
    return True
  if value in ("0", "false", "no", "off", ""):
    return False
  return None


def derive_effective_cruise_context(
  *,
  plan: CachedLongitudinalPlan | None,
  raw_cruise_mps: float,
  force_decel: bool,
  long_active: bool,
  gas_pressed: bool,
  params: dict[str, str],
  rti_zero_threats_proven: bool,
) -> EffectiveCruiseContext:
  """Read a v1 writer cap or conservatively derive a non-gating legacy cap."""
  if _has_planner_replay_inputs_v1(plan):
    assert plan is not None and plan.plan_sp is not None
    speed_mps = float(plan.plan_sp.replay_effective_cruise_mps)
    if not math.isfinite(speed_mps) or speed_mps < 0.0:
      return EffectiveCruiseContext(
        None,
        None,
        "LongitudinalPlanSP.replayInputs.effectiveCruiseMps",
        "unscorable",
        "v1 writer contract contains an invalid effective cruise cap",
      )
    return EffectiveCruiseContext(
      speed_mps,
      "writerContractV1",
      "LongitudinalPlanSP.replayInputs.effectiveCruiseMps",
      "exact",
      "v1 planner producer recorded the final MPC-boundary cap in m/s",
    )
  if force_decel:
    return EffectiveCruiseContext(
      0.0,
      "forceDecel",
      "controlsState.forceDecel",
      "legacy_derived",
      "legacy forceDecel deterministically derives zero, but no v1 writer contract exists",
    )
  if plan is None:
    return EffectiveCruiseContext(None, None, None, "unscorable", "missing longitudinalPlan for model frame")
  if plan.plan_sp is None:
    return EffectiveCruiseContext(
      None,
      None,
      None,
      "unscorable",
      "no one-to-one longitudinalPlanSP publication within 20 ms",
    )

  plan_sp = plan.plan_sp
  if plan_sp.object_hazard_active:
    return EffectiveCruiseContext(
      None,
      None,
      None,
      "unscorable",
      "object hazard was active but its planner input state was not captured",
    )

  candidates: list[tuple[str, float, str]] = [
    ("rawCruise", float(raw_cruise_mps), "carState.vCruise"),
  ]
  if plan_sp.slc_active:
    slc_cap = float(plan_sp.slc_speed_limit_mps + plan_sp.slc_speed_limit_offset_mps)
    candidates.append((
      "speedLimitControl",
      slc_cap,
      "paired longitudinalPlanSP.slc.speedLimit+speedLimitOffset",
    ))

  vtsc_enabled = _param_bool(params, "VisionTurnSpeedControl")
  if vtsc_enabled is None:
    return EffectiveCruiseContext(None, None, None, "unscorable", "missing VisionTurnSpeedControl parameter")
  if vtsc_enabled and long_active:
    if gas_pressed:
      return EffectiveCruiseContext(
        None,
        None,
        None,
        "unscorable",
        "VTSC enabled while gasPressed; published velocity applicability is not provable",
      )
    if not math.isfinite(plan_sp.vtsc_velocity_mps) or plan_sp.vtsc_velocity_mps <= 0.0:
      return EffectiveCruiseContext(
        None,
        None,
        None,
        "unscorable",
        "VTSC enabled and longActive but paired publication has no usable velocity",
      )
    # The enum's disabled value is not used as proof of inactivity. Treat the
    # published velocity as a candidate whenever the controller can apply.
    candidates.append((
      "visionTurnSpeedControl",
      float(plan_sp.vtsc_velocity_mps),
      f"paired longitudinalPlanSP.visionTurnSpeedControl.velocity(state={plan_sp.vtsc_state})",
    ))

  rti_enabled = _param_bool(params, "RTIEnabled")
  if rti_enabled is None:
    return EffectiveCruiseContext(None, None, None, "unscorable", "missing RTIEnabled parameter")
  if rti_enabled and not rti_zero_threats_proven:
    return EffectiveCruiseContext(
      None,
      None,
      None,
      "unscorable",
      "RTI enabled without route-wide zero-threat proof",
    )

  weather_enabled = _param_bool(params, "WeatherAwareControlEnabled")
  if weather_enabled is None:
    return EffectiveCruiseContext(None, None, None, "unscorable", "missing WeatherAwareControlEnabled parameter")
  selected = min(candidates, key=lambda candidate: candidate[1])
  proof_parts = [candidate[2] for candidate in candidates]
  if rti_enabled:
    proof_parts.append("all logged rtiStateSP publications contained zero threats")
  if weather_enabled:
    reductions_mph: list[float] = []
    for key, default_mph in (
      ("WeatherSpeedReductionLight", 5.0),
      ("WeatherSpeedReductionModerate", 10.0),
      ("WeatherSpeedReductionHeavy", 15.0),
    ):
      try:
        reductions_mph.append(float(params.get(key, default_mph)))
      except (TypeError, ValueError):
        return EffectiveCruiseContext(None, None, None, "unscorable", f"invalid {key} parameter")
    weather_floor_mps = max(2.24, float(raw_cruise_mps) - max(reductions_mph) * MPH_TO_MPS)
    if weather_floor_mps < selected[1] - 1e-6:
      return EffectiveCruiseContext(
        None,
        None,
        None,
        "unscorable",
        f"weather could undercut selected cap (floor {weather_floor_mps:.6f} < {selected[1]:.6f} m/s)",
      )
    proof_parts.append(f"weather configured floor {weather_floor_mps:.6f} m/s cannot undercut selected cap")

  return EffectiveCruiseContext(
    selected[1],
    selected[0],
    "; ".join(proof_parts),
    "legacy_derived",
    f"legacy paired publications derive {selected[0]}; no v1 writer contract exists",
  )


def resolve_planner_context(
  *,
  plan: CachedLongitudinalPlan | None,
  replay_index: RouteReplayIndex,
  radar_association: PlannerRadarAssociation,
  effective_cruise: EffectiveCruiseContext,
  params_snapshot: dict[str, str],
) -> PlannerContextResolution:
  """Join the v1 writer clocks to exact planner inputs without output inference."""
  if plan is None:
    return PlannerContextResolution(
      "unscorable", "missing longitudinalPlan for model frame", {}, {}, {}
    )

  param_manifest = captured_param_manifest(params_snapshot)
  plan_sp = plan.plan_sp
  if not _has_planner_replay_inputs_v1(plan):
    if plan_sp is not None and plan_sp.replay_inputs_valid:
      if plan_sp.replay_inputs_version != PLANNER_REPLAY_INPUTS_VERSION:
        reason = f"unsupported LongitudinalPlanSP.replayInputs version {plan_sp.replay_inputs_version}"
      elif plan_sp.replay_plan_log_mono_time_ns != plan.log_mono_time_ns:
        reason = (
          "LongitudinalPlanSP.replayInputs longitudinalPlan pointer mismatch: "
          f"{plan_sp.replay_plan_log_mono_time_ns} != {plan.log_mono_time_ns}"
        )
      else:
        reason = (
          "LongitudinalPlanSP.replayInputs modelV2 pointer mismatch: "
          f"{int(plan_sp.replay_input_clocks_ns.get('modelV2', 0))} != {plan.model_mono_time_ns}"
        )
      return PlannerContextResolution("unscorable", reason, {}, {}, {
        "contract": {
          "status": "unsupported" if plan_sp.replay_inputs_version != PLANNER_REPLAY_INPUTS_VERSION else "mismatch",
          "reason": reason,
        },
      })

    # Legacy logs predate the writer-side clock contract. For non-gating
    # diagnostics, use only information available before the triggering model
    # publication: the model payload at modelMonoTime and the latest publication
    # of each other core service at or before that clock. This is deterministic
    # and output-independent, but it cannot prove what SubMaster consumed when
    # services raced the model poll, so every association remains inferred.
    model_clock = int(plan.model_mono_time_ns)
    inputs: dict[str, Any] = {}
    clocks: dict[str, int] = {}
    provenance: dict[str, Any] = {
      "contract": {
        "status": "inferred",
        "reason": "legacy/schema-default planner inputs; diagnostic latest-at-or-before-model join",
      },
    }
    failures: list[str] = []
    for service in PLANNER_CRITICAL_SERVICES:
      if service == "radarState":
        if radar_association.target_log_mono_time_ns is not None:
          radar_clock = int(radar_association.target_log_mono_time_ns)
          if radar_clock in replay_index.conflicting_input_mono_times_by_service.get(service, ()):
            provenance[service] = {
              "status": "conflict",
              "clockNs": radar_clock,
              "reason": "route contains conflicting duplicate radarState payloads at this clock",
            }
            failures.append("radarState diagnostic payload has conflicting duplicates")
          else:
            clocks[service] = radar_clock
            provenance[service] = {
              "status": "inferred",
              "clockNs": radar_clock,
              "reason": f"legacy {radar_association.resolution} timing association; not a writer pointer",
            }
        else:
          provenance[service] = {
            "status": (
              "conflict" if radar_association.resolution == "conflict"
              else "ambiguous" if radar_association.candidate_log_mono_times_ns
              else "missing"
            ),
            "candidateClocksNs": list(radar_association.candidate_log_mono_times_ns),
            "reason": radar_association.reason,
          }
          if radar_association.resolution == "conflict":
            failures.append("radarState diagnostic association has conflicting duplicate payloads")
        continue

      snapshots = replay_index.planner_inputs_by_service.get(service, {})
      service_times = replay_index.planner_input_mono_times_by_service.get(service)
      if service_times is None:
        # Synthetic/unit-test indexes may omit the precomputed route-wide keys.
        service_times = tuple(sorted(int(clock) for clock in snapshots))
      if service == "modelV2":
        clock = model_clock if model_clock in snapshots else None
        join_reason = "legacy longitudinalPlan.modelMonoTime exact modelV2 payload join"
      else:
        service_idx = bisect_right(service_times, model_clock) - 1
        clock = int(service_times[service_idx]) if service_idx >= 0 else None
        join_reason = "legacy latest publication at or before modelMonoTime; SubMaster race remains unknown"

      if clock is not None and clock in replay_index.conflicting_input_mono_times_by_service.get(service, ()):
        provenance[service] = {
          "status": "conflict",
          "clockNs": clock,
          "cutoffClockNs": model_clock,
          "reason": f"recorded {service} clock has conflicting duplicate payloads",
        }
        failures.append(f"{service} diagnostic payload has conflicting duplicates")
        continue
      if clock is None or clock <= 0 or clock not in snapshots:
        provenance[service] = {
          "status": "missing",
          "cutoffClockNs": model_clock,
          "reason": f"no recorded {service} payload satisfies the legacy diagnostic join",
        }
        failures.append(f"{service} diagnostic payload is missing")
        continue
      clocks[service] = clock
      inputs[service] = dict(snapshots[clock])
      provenance[service] = {
        "status": "inferred",
        "clockNs": clock,
        "cutoffClockNs": model_clock,
        "reason": join_reason,
      }

    plan_time_ns = int(plan.log_mono_time_ns)
    warmup_ns = int(round(RADARD_DEPENDENCY_WARMUP_S * 1e9))
    change_start_idx = bisect_right(replay_index.param_change_mono_times_ns, plan_time_ns - warmup_ns)
    change_end_idx = bisect_right(
      replay_index.param_change_mono_times_ns,
      plan_time_ns + PARAM_CHANGE_NEAR_MARGIN_NS,
    )
    nearby_param_changes = replay_index.param_change_mono_times_ns[change_start_idx:change_end_idx]
    if not param_manifest["complete"]:
      provenance["params"] = {
        **param_manifest,
        "status": "incomplete",
        "reason": f"captured parameter manifest is missing {len(param_manifest['missingKeys'])} replay-relevant values",
      }
      failures.append("captured parameter manifest is incomplete")
    elif nearby_param_changes:
      provenance["params"] = {
        **param_manifest,
        "status": "unstable",
        "changeClocksNs": list(nearby_param_changes),
        "reason": (
          f"legacy parameter change occurred within the {RADARD_DEPENDENCY_WARMUP_S:.1f} s dependency window"
        ),
      }
      failures.append("captured legacy parameters were not stable through dependency warmup")
    else:
      provenance["params"] = {
        **param_manifest,
        "status": "inferred",
        "reason": "captured legacy parameter snapshot; no writer-side planner clock contract",
      }
    if effective_cruise.speed_mps is None:
      failures.append(
        f"effective cruise context is {effective_cruise.status}: {effective_cruise.reason}"
      )

    reason = (
      "legacy diagnostic core inputs joined at or before modelMonoTime; asynchronous consumption remains unproven"
      if not failures else "; ".join(failures)
    )
    return PlannerContextResolution(
      "legacy_derived" if not failures else "unscorable",
      reason,
      inputs,
      clocks,
      provenance,
    )

  assert plan_sp is not None
  clocks = {
    service: int(clock)
    for service, clock in plan_sp.replay_input_clocks_ns.items()
  }
  inputs: dict[str, Any] = {}
  provenance: dict[str, Any] = {
    "contract": {
      "status": "exact",
      "version": plan_sp.replay_inputs_version,
      "reason": "LongitudinalPlanSP.replayInputs v1 producer contract",
    },
  }
  for service, clock in clocks.items():
    if service not in PLANNER_CRITICAL_SERVICES:
      provenance[service] = {
        "status": "recorded_clock" if clock > 0 else "absent",
        "clockNs": clock,
        "reason": "optional v1 planner service clock; payload not required for core exactness",
      }
  failures: list[str] = []

  for service in PLANNER_CRITICAL_SERVICES:
    clock = int(clocks.get(service, 0))
    if clock <= 0:
      provenance[service] = {"status": "missing", "clockNs": clock, "reason": "v1 clock is zero"}
      failures.append(f"{service} clock is zero")
      continue

    if service == "radarState":
      if clock in replay_index.conflicting_input_mono_times_by_service.get(service, ()):
        provenance[service] = {
          "status": "conflict",
          "clockNs": clock,
          "reason": "route contains conflicting duplicate radarState payloads at this clock",
        }
        failures.append(f"radarState target {clock} has conflicting duplicate payloads")
      elif radar_association.resolution != "exact" or radar_association.target_log_mono_time_ns != clock:
        provenance[service] = {
          "status": "missing",
          "clockNs": clock,
          "reason": radar_association.reason,
        }
        failures.append(f"radarState target {clock} is not exactly joined")
      else:
        provenance[service] = {
          "status": "exact",
          "clockNs": clock,
          "reason": "v1 clock joined to recorded radarState publication",
        }
      continue

    if service == "modelV2" and int(plan.model_mono_time_ns) != clock:
      provenance[service] = {
        "status": "mismatch",
        "clockNs": clock,
        "builtInClockNs": int(plan.model_mono_time_ns),
        "reason": "v1 modelV2 clock disagrees with longitudinalPlan.modelMonoTime",
      }
      failures.append("modelV2 v1/built-in clock mismatch")
      continue

    if clock in replay_index.conflicting_input_mono_times_by_service.get(service, ()):
      provenance[service] = {
        "status": "conflict",
        "clockNs": clock,
        "reason": f"recorded {service} target has conflicting duplicate payloads",
      }
      failures.append(f"{service} target {clock} has conflicting duplicate payloads")
      continue

    snapshot = replay_index.planner_inputs_by_service.get(service, {}).get(clock)
    if snapshot is None:
      provenance[service] = {
        "status": "missing",
        "clockNs": clock,
        "reason": f"recorded {service} target is absent from the loaded route",
      }
      failures.append(f"{service} target {clock} is missing")
      continue
    inputs[service] = dict(snapshot)
    provenance[service] = {
      "status": "exact",
      "clockNs": clock,
      "reason": f"v1 clock exact route-wide {service} join",
    }

  if effective_cruise.status != "exact" or effective_cruise.speed_mps is None:
    failures.append(f"effective cruise writer contract is {effective_cruise.status}")

  if plan_sp.object_hazard_active:
    object_clock = int(plan_sp.replay_input_clocks_ns.get("objectHazardStateSP", 0))
    provenance["objectHazardStateSP"] = {
      "status": "missing_payload",
      "clockNs": object_clock,
      "reason": "active object hazard payload is not serialized by the v1 replay harness",
    }
    failures.append("active object hazard state is not exactly replayed")

  plan_time_ns = int(plan.log_mono_time_ns)
  warmup_ns = int(round(RADARD_DEPENDENCY_WARMUP_S * 1e9))
  change_start_idx = bisect_right(replay_index.param_change_mono_times_ns, plan_time_ns - warmup_ns)
  change_end_idx = bisect_right(
    replay_index.param_change_mono_times_ns,
    plan_time_ns + PARAM_CHANGE_NEAR_MARGIN_NS,
  )
  nearby_param_changes = replay_index.param_change_mono_times_ns[change_start_idx:change_end_idx]
  if not param_manifest["complete"]:
    provenance["params"] = {
      **param_manifest,
      "status": "incomplete",
      "reason": f"captured parameter manifest is missing {len(param_manifest['missingKeys'])} replay-relevant values",
    }
    failures.append("captured parameter manifest is incomplete")
  elif nearby_param_changes:
    provenance["params"] = {
      **param_manifest,
      "status": "unstable",
      "changeClocksNs": list(nearby_param_changes),
      "reason": f"parameter change occurred within the {RADARD_DEPENDENCY_WARMUP_S:.1f} s dependency window",
    }
    failures.append("captured parameters were not stable through dependency warmup")
  else:
    provenance["params"] = {
      **param_manifest,
      "status": "exact",
      "reason": f"captured parameters stable for at least {RADARD_DEPENDENCY_WARMUP_S:.1f} s",
    }

  if failures:
    return PlannerContextResolution("unscorable", "; ".join(failures), inputs, clocks, provenance)
  return PlannerContextResolution(
    "exact",
    "v1 writer contract; all critical planner clocks joined; captured parameters stable",
    inputs,
    clocks,
    provenance,
  )


def _record_route_core_payload(
  *,
  service: str,
  mono_time_ns: int,
  payload: dict[str, Any],
  payloads_by_service: dict[str, dict[int, dict[str, Any]]],
  fingerprints_by_service: dict[str, dict[int, str]],
  conflicts_by_service: dict[str, set[int]],
  identical_duplicate_counts: dict[str, int],
  retain_payload: bool = True,
) -> str:
  """Keep one payload per service clock and classify duplicate publications.

  Logger segment overlap can repeat a byte-equivalent publication. That is a
  safe route-wide join. Two different payloads carrying the same service clock
  are not order-resolvable, so retain the first only for diagnostics and mark
  the clock conflicting for every exactness decision.
  """
  service_payloads = payloads_by_service.setdefault(service, {})
  service_fingerprints = fingerprints_by_service.setdefault(service, {})
  service_conflicts = conflicts_by_service.setdefault(service, set())
  fingerprint = json.dumps(payload, allow_nan=True, separators=(",", ":"), sort_keys=True)
  previous_fingerprint = service_fingerprints.get(mono_time_ns)
  if previous_fingerprint is None:
    if retain_payload:
      service_payloads[mono_time_ns] = payload
    service_fingerprints[mono_time_ns] = fingerprint
    return "new"
  if previous_fingerprint == fingerprint:
    identical_duplicate_counts[service] = identical_duplicate_counts.get(service, 0) + 1
    return "identical_duplicate"
  service_conflicts.add(mono_time_ns)
  return "conflicting_duplicate"


def _cache_route_replay_inputs(segment_rows) -> RouteReplayIndex:
  """Index logged replay inputs independent of logger event ordering."""
  car_state_by_mono_time: dict[int, CachedCarState] = {}
  planner_inputs_by_service: dict[str, dict[int, dict[str, Any]]] = {
    service: {} for service in (*PLANNER_CRITICAL_SERVICES, "liveTracks")
  }
  fingerprints_by_service: dict[str, dict[int, str]] = {
    service: {} for service in planner_inputs_by_service
  }
  conflicts_by_service: dict[str, set[int]] = {
    service: set() for service in planner_inputs_by_service
  }
  identical_duplicate_counts: dict[str, int] = {}
  plans: list[CachedLongitudinalPlan] = []
  plan_sp_messages: list[CachedLongitudinalPlanSP] = []
  cached_params: dict[str, str] = {}
  param_change_mono_times_ns: list[int] = []
  rti_state_count = 0
  rti_zero_threats_proven = True
  capture_first_log_mono_time_ns: int | None = None
  capture_last_log_mono_time_ns: int | None = None
  start_of_route_mono_times_ns: list[int] = []
  process_identities: dict[str, set[tuple[int, int]]] = {"plannerd": set(), "radard": set()}
  manager_plannerd_observations: list[tuple[int, int, bool, bool]] = []
  model_v2_mono_times_ns: list[int] = []
  for segment_row in segment_rows:
    for msg in LogReader(segment_row["rlog_path"]):
      which = msg.which()
      mono_time_ns = int(msg.logMonoTime)
      if capture_first_log_mono_time_ns is None or mono_time_ns < capture_first_log_mono_time_ns:
        capture_first_log_mono_time_ns = mono_time_ns
      if capture_last_log_mono_time_ns is None or mono_time_ns > capture_last_log_mono_time_ns:
        capture_last_log_mono_time_ns = mono_time_ns
      if which == "sentinel" and str(msg.sentinel.type) == "startOfRoute":
        start_of_route_mono_times_ns.append(mono_time_ns)
      elif which == "procLog":
        for process in msg.procLog.procs:
          command = tuple(str(value) for value in process.cmdline)
          process_key = (
            "plannerd" if "selfdrive.controls.plannerd" in command else
            "radard" if "selfdrive.controls.radard" in command else None
          )
          if process_key is not None and int(process.pid) > 0 and float(process.startTime) > 0.0:
            process_identities[process_key].add((
              int(process.pid),
              int(round(float(process.startTime) * 1e9)),
            ))
      elif which == "managerState":
        for process in msg.managerState.processes:
          if str(process.name) == "plannerd":
            manager_plannerd_observations.append((
              mono_time_ns,
              int(process.pid),
              bool(process.running),
              bool(process.shouldBeRunning),
            ))
      if which == "initData":
        cached_params.update(_extract_init_params(msg.initData))
      elif which == "carControlSP":
        updates = {str(param.key): str(param.value) for param in msg.carControlSP.params}
        if any(cached_params.get(key) != value for key, value in updates.items()):
          param_change_mono_times_ns.append(mono_time_ns)
        cached_params.update(updates)
      elif which == "carState":
        cached = CachedCarState(
          v_ego_mps=float(msg.carState.vEgo),
          a_ego_mps2=float(msg.carState.aEgo),
          v_cruise_kph=float(msg.carState.vCruise),
          gas_pressed=bool(msg.carState.gasPressed),
          standstill=bool(msg.carState.standstill),
        )
        payload = {
          "vEgoMps": cached.v_ego_mps,
          "aEgoMps2": cached.a_ego_mps2,
          "vCruiseKph": cached.v_cruise_kph,
          "gasPressed": cached.gas_pressed,
          "standstill": cached.standstill,
        }
        duplicate_status = _record_route_core_payload(
          service="carState",
          mono_time_ns=mono_time_ns,
          payload=payload,
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
        )
        if duplicate_status == "new":
          car_state_by_mono_time[mono_time_ns] = cached
      elif which == "controlsState":
        _record_route_core_payload(
          service="controlsState",
          mono_time_ns=mono_time_ns,
          payload={
            "longControlState": int(msg.controlsState.longControlState.raw),
            "forceDecel": bool(msg.controlsState.forceDecel),
          },
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
        )
      elif which == "carControl":
        _record_route_core_payload(
          service="carControl",
          mono_time_ns=mono_time_ns,
          payload={
            "longActive": bool(msg.carControl.longActive),
            "orientationNED": [float(value) for value in msg.carControl.orientationNED],
          },
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
        )
      elif which == "selfdriveState":
        _record_route_core_payload(
          service="selfdriveState",
          mono_time_ns=mono_time_ns,
          payload={
            "enabled": bool(msg.selfdriveState.enabled),
            "experimentalMode": bool(msg.selfdriveState.experimentalMode),
            "personality": int(msg.selfdriveState.personality.raw),
          },
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
        )
      elif which == "modelV2":
        model_v2_mono_times_ns.append(mono_time_ns)
        raw_model = serialize_model_frame(msg.modelV2)
        raw_model["logMonoTimeNs"] = mono_time_ns
        _record_route_core_payload(
          service="modelV2",
          mono_time_ns=mono_time_ns,
          payload={"rawModel": raw_model},
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
        )
      elif which == "liveTracks":
        _record_route_core_payload(
          service="liveTracks",
          mono_time_ns=mono_time_ns,
          payload=msg.liveTracks.to_dict(),
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
          retain_payload=False,
        )
      elif which == "radarState":
        _record_route_core_payload(
          service="radarState",
          mono_time_ns=mono_time_ns,
          payload=msg.radarState.to_dict(),
          payloads_by_service=planner_inputs_by_service,
          fingerprints_by_service=fingerprints_by_service,
          conflicts_by_service=conflicts_by_service,
          identical_duplicate_counts=identical_duplicate_counts,
          retain_payload=False,
        )
      elif which == "longitudinalPlan":
        plans.append(CachedLongitudinalPlan(
          log_mono_time_ns=mono_time_ns,
          model_mono_time_ns=int(msg.longitudinalPlan.modelMonoTime),
          solver_execution_time_s=float(msg.longitudinalPlan.solverExecutionTime),
          radar_state_mono_time_ns=int(msg.longitudinalPlan.radarStateMonoTimeDEPRECATED),
          v_cruise_deprecated_mps=float(msg.longitudinalPlan.vCruiseDEPRECATED),
          a_target_mps2=float(msg.longitudinalPlan.aTarget),
          source=str(msg.longitudinalPlan.longitudinalPlanSource),
        ))
      elif which == "longitudinalPlanSP":
        replay_inputs = msg.longitudinalPlanSP.replayInputs
        plan_sp_messages.append(CachedLongitudinalPlanSP(
          log_mono_time_ns=mono_time_ns,
          slc_active=bool(msg.longitudinalPlanSP.slc.active),
          slc_state=_enum_name(msg.longitudinalPlanSP.slc.state),
          slc_speed_limit_mps=float(msg.longitudinalPlanSP.slc.speedLimit),
          slc_speed_limit_offset_mps=float(msg.longitudinalPlanSP.slc.speedLimitOffset),
          vtsc_state=_enum_name(msg.longitudinalPlanSP.visionTurnSpeedControl.state),
          vtsc_velocity_mps=float(msg.longitudinalPlanSP.visionTurnSpeedControl.velocity),
          object_hazard_active=bool(msg.longitudinalPlanSP.objectHazardControl.active),
          replay_inputs_valid=bool(replay_inputs.valid),
          replay_inputs_version=int(replay_inputs.version),
          replay_effective_cruise_mps=float(replay_inputs.effectiveCruiseMps),
          replay_plan_log_mono_time_ns=int(replay_inputs.longitudinalPlanMonoTimeNs),
          replay_input_clocks_ns={
            "radarState": int(replay_inputs.radarStateMonoTimeNs),
            "carState": int(replay_inputs.carStateMonoTimeNs),
            "carControl": int(replay_inputs.carControlMonoTimeNs),
            "controlsState": int(replay_inputs.controlsStateMonoTimeNs),
            "selfdriveState": int(replay_inputs.selfdriveStateMonoTimeNs),
            "liveParameters": int(replay_inputs.liveParametersMonoTimeNs),
            "modelV2": int(replay_inputs.modelV2MonoTimeNs),
            "liveMapDataSP": int(replay_inputs.liveMapDataSPMonoTimeNs),
            "carStateSP": int(replay_inputs.carStateSPMonoTimeNs),
            "rtiStateSP": int(replay_inputs.rtiStateSPMonoTimeNs),
            "objectHazardStateSP": int(replay_inputs.objectHazardStateSPMonoTimeNs),
            "gpsLocation": int(replay_inputs.gpsLocationMonoTimeNs),
            "gpsLocationExternal": int(replay_inputs.gpsLocationExternalMonoTimeNs),
          },
        ))
      elif which == "rtiStateSP":
        rti_state_count += 1
        rti_zero_threats_proven = bool(
          rti_zero_threats_proven and
          len(msg.rtiStateSP.threats) == 0 and
          not bool(msg.rtiStateSP.threatAhead) and
          float(msg.rtiStateSP.recommendedSpeed) <= 0.0
        )

  paired_plans = _pair_longitudinal_plans_with_sp(plans, plan_sp_messages)
  planner_route_start_provenance = _build_planner_route_start_provenance(
    segment_rows=segment_rows,
    capture_first_log_mono_time_ns=capture_first_log_mono_time_ns,
    capture_last_log_mono_time_ns=capture_last_log_mono_time_ns,
    start_of_route_mono_times_ns=start_of_route_mono_times_ns,
    process_identities=process_identities,
    manager_plannerd_observations=manager_plannerd_observations,
    model_v2_mono_times_ns=model_v2_mono_times_ns,
    plans=plans,
    paired_plans=paired_plans,
    plan_sp_messages=plan_sp_messages,
  )
  plans_by_model_mono_time: dict[int, CachedLongitudinalPlan] = {}
  ambiguous_plan_model_clocks: set[int] = set()
  for plan in paired_plans:
    model_clock = int(plan.model_mono_time_ns)
    if model_clock <= 0 or model_clock in ambiguous_plan_model_clocks:
      continue
    if model_clock in plans_by_model_mono_time:
      plans_by_model_mono_time.pop(model_clock, None)
      ambiguous_plan_model_clocks.add(model_clock)
      continue
    plans_by_model_mono_time[model_clock] = plan
  return RouteReplayIndex(
    car_state_by_mono_time=car_state_by_mono_time,
    planner_inputs_by_service=planner_inputs_by_service,
    live_tracks_mono_times=tuple(sorted(fingerprints_by_service["liveTracks"])),
    radar_state_mono_times=tuple(sorted(fingerprints_by_service["radarState"])),
    plans_by_model_mono_time=plans_by_model_mono_time,
    param_change_mono_times_ns=tuple(sorted(set(param_change_mono_times_ns))),
    rti_zero_threats_proven=bool(rti_state_count > 0 and rti_zero_threats_proven),
    rti_state_count=rti_state_count,
    planner_input_mono_times_by_service={
      service: tuple(sorted(snapshots))
      for service, snapshots in planner_inputs_by_service.items()
    },
    conflicting_input_mono_times_by_service={
      service: tuple(sorted(clocks))
      for service, clocks in conflicts_by_service.items()
      if clocks
    },
    identical_duplicate_counts_by_service=dict(sorted(identical_duplicate_counts.items())),
    planner_route_start_provenance=planner_route_start_provenance,
  )


def load_route_scan(conn, route_row, *, strict_service_joins: bool = False) -> RouteScanResult:
  segment_rows = get_route_segments(conn, int(route_row["route_id"]))
  replay_index = _cache_route_replay_inputs(segment_rows)
  car_state_by_mono_time = replay_index.car_state_by_mono_time
  live_tracks_mono_times = replay_index.live_tracks_mono_times
  observed_params: dict[str, str] = {}
  pending_param_updates: dict[str, str] = {}
  latest_car_state = None
  latest_car_state_log_mono_time_ns: int | None = None
  orphan_car_state_references: list[dict[str, int]] = []
  radar_state_count = 0
  latest_controls_state = None
  latest_selfdrive_state = None
  latest_car_control = None
  latest_model = None
  model_by_mono_time: dict[int, tuple[Any, dict[str, Any]]] = {}
  processed_radar_state_mono_times: set[int] = set()
  first_radar_time = None
  frames: list[EpisodeFrame] = []
  raw_driver_mark_presses: list[dict[str, Any]] = []
  longflag_payloads: list[dict[str, Any]] = []

  for segment_row in segment_rows:
    for msg in LogReader(segment_row["rlog_path"]):
      which = msg.which()
      if which == "initData":
        _merge_param_updates(observed_params, pending_param_updates, _extract_init_params(msg.initData))
      elif which in DRIVER_MARK_SERVICES:
        # A driver flag is human intent, never a heuristic. Keep every raw echo
        # here; collapsing and frame anchoring happen once the scan is complete.
        raw_driver_mark_presses.append({
          "segIdx": int(segment_row["seg_idx"]),
          "logMonoTimeNs": int(msg.logMonoTime),
          "service": which,
        })
      elif which == "logMessage":
        longflag = _parse_longflag(msg.logMessage)
        if longflag is not None:
          longflag_payloads.append({
            "segIdx": int(segment_row["seg_idx"]),
            "logMonoTimeNs": int(msg.logMonoTime),
            "payload": longflag,
          })
      elif which == "carControlSP":
        _merge_param_updates(
          observed_params,
          pending_param_updates,
          {str(param.key): str(param.value) for param in msg.carControlSP.params},
        )
      elif which == "carState":
        latest_car_state_log_mono_time_ns = int(msg.logMonoTime)
        latest_car_state = car_state_by_mono_time[latest_car_state_log_mono_time_ns]
      elif which == "controlsState":
        latest_controls_state = msg.controlsState
      elif which == "selfdriveState":
        latest_selfdrive_state = msg.selfdriveState
      elif which == "carControl":
        latest_car_control = msg.carControl
      elif which == "modelV2":
        model_payload = serialize_model_frame(msg.modelV2)
        model_payload["logMonoTimeNs"] = int(msg.logMonoTime)
        model_clock_ns = int(msg.logMonoTime)
        if model_clock_ns not in model_by_mono_time:
          model_by_mono_time[model_clock_ns] = (msg.modelV2, model_payload)
        latest_model = model_by_mono_time[model_clock_ns]
        while len(model_by_mono_time) > 200:
          model_by_mono_time.pop(next(iter(model_by_mono_time)))
      elif which == "radarState":
        radar_state_count += 1
        radar_state_clock_ns = int(msg.logMonoTime)
        if radar_state_clock_ns in processed_radar_state_mono_times:
          continue
        processed_radar_state_mono_times.add(radar_state_clock_ns)
        radar_publication_conflict = bool(
          radar_state_clock_ns in replay_index.conflicting_input_mono_times_by_service.get("radarState", ())
        )
        # Route-relative time is anchored to the first recorded radar publish,
        # even if that boundary frame references a carState outside the pulled
        # segment and must be excluded from exact replay.
        if first_radar_time is None:
          first_radar_time = int(msg.logMonoTime)
        radar_replay_inputs = msg.radarState.replayInputs
        radar_replay_valid = bool(radar_replay_inputs.valid)
        radar_replay_version = int(radar_replay_inputs.version)
        # v2 appends recovery-floor and calm-position amplifier-veto telemetry;
        # the exact service clocks introduced by v1 are unchanged, so both
        # versions remain valid dependency contracts for route extraction.
        radar_replay_supported = bool(
          radar_replay_valid and is_supported_radard_replay_version(radar_replay_version)
        )
        built_in_car_state_clock = int(msg.radarState.carStateMonoTime)
        built_in_model_clock = int(msg.radarState.mdMonoTime)
        if radar_replay_supported:
          radar_car_state_mono_time = int(radar_replay_inputs.carStateMonoTimeNs)
          car_clock_matches = radar_car_state_mono_time == built_in_car_state_clock
          frame_car_state = car_state_by_mono_time.get(radar_car_state_mono_time)
          frame_car_state_mono_time_ns = radar_car_state_mono_time
          if radar_car_state_mono_time in replay_index.conflicting_input_mono_times_by_service.get("carState", ()):
            car_state_association = {
              "status": "conflict",
              "clockNs": radar_car_state_mono_time,
              "reason": f"RadarState.replayInputs v{radar_replay_version} carState target has conflicting duplicate payloads",
            }
          elif not car_clock_matches:
            car_state_association = {
              "status": "mismatch",
              "clockNs": radar_car_state_mono_time,
              "builtInClockNs": built_in_car_state_clock,
              "reason": "RadarState.replayInputs carState clock disagrees with carStateMonoTime",
            }
          elif radar_car_state_mono_time <= 0:
            car_state_association = {
              "status": "missing",
              "clockNs": radar_car_state_mono_time,
              "reason": f"RadarState.replayInputs v{radar_replay_version} carState clock is zero",
            }
          elif frame_car_state is None:
            car_state_association = {
              "status": "missing",
              "clockNs": radar_car_state_mono_time,
              "reason": f"RadarState.replayInputs v{radar_replay_version} carState target is absent from the loaded route",
            }
          else:
            car_state_association = {
              "status": "exact",
              "clockNs": radar_car_state_mono_time,
              "reason": f"RadarState.replayInputs v{radar_replay_version} exact route-wide carState join",
            }
          if frame_car_state is None:
            frame_car_state = latest_car_state
        else:
          radar_car_state_mono_time = built_in_car_state_clock
          if radar_car_state_mono_time > 0:
            frame_car_state = car_state_by_mono_time.get(radar_car_state_mono_time)
            if frame_car_state is None:
              diagnostic = {
                "radarStateLogMonoTimeNs": int(msg.logMonoTime),
                "carStateLogMonoTimeNs": radar_car_state_mono_time,
                "segment": int(segment_row["seg_idx"]),
              }
              orphan_car_state_references.append(diagnostic)
              if strict_service_joins:
                raise ValueError(
                  f"radarState at {int(msg.logMonoTime)} references carStateMonoTime={radar_car_state_mono_time}, "
                  "but that exact carState is missing from the loaded route"
                )
              continue
            frame_car_state_mono_time_ns = radar_car_state_mono_time
            if radar_car_state_mono_time in replay_index.conflicting_input_mono_times_by_service.get("carState", ()):
              car_state_association = {
                "status": "conflict",
                "clockNs": radar_car_state_mono_time,
                "reason": "legacy radarState.carStateMonoTime target has conflicting duplicate payloads",
              }
            else:
              car_state_association = {
                "status": "exact",
                "reason": "legacy radarState.carStateMonoTime exact route-wide join",
              }
          else:
            frame_car_state = latest_car_state
            frame_car_state_mono_time_ns = latest_car_state_log_mono_time_ns
            car_state_association = {
              "status": "inferred" if frame_car_state is not None else "missing",
              "reason": "legacy zero carStateMonoTime; latest logger-order carState fallback",
            }

        if frame_car_state is None or latest_controls_state is None or latest_selfdrive_state is None or latest_car_control is None:
          continue
        model_record = None
        model_mono_time = (
          int(radar_replay_inputs.modelV2MonoTimeNs) if radar_replay_supported else built_in_model_clock
        )
        if model_mono_time > 0:
          model_record = model_by_mono_time.get(model_mono_time)
        # A nonzero mdMonoTime is an exact association contract. Falling back to
        # a nearby frame when that model is missing silently pairs RadarD output
        # with the wrong perception input and defeats recorded replay fidelity.
        if model_record is None and not radar_replay_supported and model_mono_time <= 0 and latest_model is not None:
          latest_model_time = int(latest_model[1].get("logMonoTimeNs", 0))
          if abs(int(msg.logMonoTime) - latest_model_time) <= 150_000_000:
            model_record = latest_model

        raw_lead_one = None
        raw_lead_two = None
        raw_model = None
        frame_model_v2_mono_time_ns = None
        if model_record is not None:
          model_msg, raw_model = model_record
          frame_model_v2_mono_time_ns = (
            model_mono_time if radar_replay_supported else int(raw_model.get("logMonoTimeNs", 0)) or None
          )
          raw_lead_one = _lead_from_model(model_msg, 0, frame_car_state.v_ego_mps)
          raw_lead_two = _lead_from_model(model_msg, 1, frame_car_state.v_ego_mps)
        elif radar_replay_supported:
          frame_model_v2_mono_time_ns = model_mono_time
        if model_mono_time in replay_index.conflicting_input_mono_times_by_service.get("modelV2", ()):
          model_association = {
            "status": "conflict",
            "clockNs": model_mono_time,
            "reason": "associated modelV2 clock has conflicting duplicate payloads",
          }
        elif radar_replay_supported and model_mono_time != built_in_model_clock:
          model_association = {
            "status": "mismatch",
            "clockNs": model_mono_time,
            "builtInClockNs": built_in_model_clock,
            "reason": "RadarState.replayInputs modelV2 clock disagrees with mdMonoTime",
          }
        elif radar_replay_supported and model_mono_time > 0 and model_record is not None:
          model_association = {
            "status": "exact",
            "clockNs": model_mono_time,
            "reason": f"RadarState.replayInputs v{radar_replay_version} exact route-wide modelV2 join",
          }
        elif radar_replay_supported and model_mono_time <= 0:
          model_association = {
            "status": "missing",
            "clockNs": model_mono_time,
            "reason": f"RadarState.replayInputs v{radar_replay_version} modelV2 clock is zero",
          }
        elif radar_replay_supported:
          model_association = {
            "status": "missing",
            "clockNs": model_mono_time,
            "reason": f"RadarState.replayInputs v{radar_replay_version} modelV2 target is absent from the loaded route",
          }
        elif model_mono_time > 0 and model_record is not None:
          model_association = {
            "status": "exact",
            "reason": "legacy radarState.mdMonoTime exact modelV2 join",
          }
        elif model_record is not None:
          model_association = {
            "status": "inferred",
            "reason": "legacy zero mdMonoTime; bounded latest modelV2 fallback",
          }
        else:
          model_association = {
            "status": "missing",
            "reason": "no modelV2 payload associated with radarState",
          }

        radar_unavailable_proven = bool(route_row["radar_unavailable"])
        if radar_replay_supported:
          frame_live_tracks_mono_time_ns = int(radar_replay_inputs.liveTracksMonoTimeNs)
          if frame_live_tracks_mono_time_ns == 0:
            live_tracks_association = {
              "status": "exact",
              "clockNs": 0,
              "reason": f"RadarState.replayInputs v{radar_replay_version} explicitly records liveTracks not yet received",
              "emptyPayloadValid": True,
            }
          elif frame_live_tracks_mono_time_ns in replay_index.conflicting_input_mono_times_by_service.get("liveTracks", ()):
            live_tracks_association = {
              "status": "conflict",
              "clockNs": frame_live_tracks_mono_time_ns,
              "reason": f"RadarState.replayInputs v{radar_replay_version} liveTracks target has conflicting duplicate payloads",
              "emptyPayloadValid": False,
            }
          elif frame_live_tracks_mono_time_ns not in live_tracks_mono_times:
            live_tracks_association = {
              "status": "missing",
              "clockNs": frame_live_tracks_mono_time_ns,
              "reason": f"RadarState.replayInputs v{radar_replay_version} liveTracks target is absent from the loaded route",
              "emptyPayloadValid": False,
            }
          else:
            live_tracks_association = {
              "status": "exact",
              "clockNs": frame_live_tracks_mono_time_ns,
              "reason": f"RadarState.replayInputs v{radar_replay_version} exact route-wide liveTracks join",
              "emptyPayloadValid": radar_unavailable_proven,
              "payloadStatus": "empty_radarless" if radar_unavailable_proven else "not_serialized",
            }
        else:
          available_cutoff_clocks = [
            clock for clock in (frame_model_v2_mono_time_ns, frame_car_state_mono_time_ns)
            if clock is not None
          ]
          service_clock_cutoff_ns = max(available_cutoff_clocks, default=0)
          live_tracks_idx = bisect_right(live_tracks_mono_times, service_clock_cutoff_ns) - 1
          frame_live_tracks_mono_time_ns = live_tracks_mono_times[live_tracks_idx] if live_tracks_idx >= 0 else None
          if frame_live_tracks_mono_time_ns is None:
            live_tracks_association = {
              "status": "missing",
              "reason": "no liveTracks publication precedes the inferred RadarD input cutoff",
              "emptyPayloadValid": radar_unavailable_proven,
            }
          elif radar_unavailable_proven:
            live_tracks_association = {
              "status": "inferred",
              "reason": "schema-default log; consumed liveTracks clock is timing-inferred",
              "emptyPayloadValid": True,
            }
          else:
            live_tracks_association = {
              "status": "missing",
              "reason": "radar-capable capture requires serialized liveTracks payload; empty substitution is invalid",
              "emptyPayloadValid": False,
            }
        service_associations = {
          "modelV2": model_association,
          "carState": car_state_association,
          "liveTracks": live_tracks_association,
          "capture": {
            "radarUnavailable": radar_unavailable_proven,
            "reason": "CarParams.radarUnavailable from the indexed route",
          },
          "contract": {
            "status": (
              "exact" if radar_replay_supported else
              "unsupported" if radar_replay_valid else
              "inferred"
            ),
            "valid": radar_replay_valid,
            "version": radar_replay_version,
            "reason": (
              f"RadarState.replayInputs v{radar_replay_version}" if radar_replay_supported else
              f"unsupported RadarState.replayInputs version {radar_replay_version}" if radar_replay_valid else
              "schema-default/legacy RadarState; liveTracks must be inferred"
            ),
          },
          "radarStatePublication": {
            "status": "conflict" if radar_publication_conflict else "unique",
            "clockNs": radar_state_clock_ns,
            "reason": (
              "route contains conflicting duplicate radarState payloads at this clock"
              if radar_publication_conflict else
              "one unique payload after accepting identical route-overlap duplicates"
            ),
          },
        }
        service_statuses = [
          str(model_association["status"]),
          str(car_state_association["status"]),
          str(live_tracks_association["status"]),
          str(service_associations["radarStatePublication"]["status"]),
        ]
        radard_service_status = (
          "conflict" if "conflict" in service_statuses
          else "mismatch" if "mismatch" in service_statuses
          else "unsupported" if radar_replay_valid and not radar_replay_supported
          else "missing" if "missing" in service_statuses
          else "inferred" if "inferred" in service_statuses
          else "exact"
        )
        radard_gate_eligible = bool(
          radard_service_status == "exact" and live_tracks_association.get("emptyPayloadValid") is True
        )

        t_s = (int(msg.logMonoTime) - first_radar_time) / 1e9
        frame_param_updates = dict(pending_param_updates)
        pending_param_updates.clear()
        frame_params_snapshot = dict(observed_params)
        raw_cruise_mps = (
          frame_car_state.v_cruise_kph / 3.6
          if frame_car_state.v_cruise_kph > 0.0 else frame_car_state.v_ego_mps
        )
        plan = replay_index.plans_by_model_mono_time.get(model_mono_time)
        if plan is None:
          planner_radar = PlannerRadarAssociation(None, (), "missing", "missing longitudinalPlan for model frame")
        else:
          planner_radar = resolve_planner_radar_association(
            plan,
            replay_index.radar_state_mono_times,
            replay_index.conflicting_input_mono_times_by_service.get("radarState", ()),
          )
        effective_cruise = derive_effective_cruise_context(
          plan=plan,
          raw_cruise_mps=raw_cruise_mps,
          force_decel=bool(latest_controls_state.forceDecel),
          long_active=bool(latest_car_control.longActive),
          gas_pressed=frame_car_state.gas_pressed,
          params=frame_params_snapshot,
          rti_zero_threats_proven=replay_index.rti_zero_threats_proven,
        )
        planner_context = resolve_planner_context(
          plan=plan,
          replay_index=replay_index,
          radar_association=planner_radar,
          effective_cruise=effective_cruise,
          params_snapshot=frame_params_snapshot,
        )
        frames.append(EpisodeFrame(
          route_key=str(route_row["route_key"]),
          seg_idx=int(segment_row["seg_idx"]),
          log_mono_time=int(msg.logMonoTime),
          t_s=t_s,
          v_ego_mps=frame_car_state.v_ego_mps,
          a_ego_mps2=frame_car_state.a_ego_mps2,
          cruise_speed_mps=raw_cruise_mps,
          long_active=bool(latest_car_control.longActive),
          long_control_state=str(latest_controls_state.longControlState),
          force_decel=bool(latest_controls_state.forceDecel),
          experimental_mode=bool(latest_selfdrive_state.experimentalMode),
          pitch_rad=float(latest_car_control.orientationNED[1]) if len(latest_car_control.orientationNED) > 1 else 0.0,
          lead_one=_lead_from_message(msg.radarState.leadOne),
          lead_two=_lead_from_message(msg.radarState.leadTwo),
          raw_lead_one=raw_lead_one,
          raw_lead_two=raw_lead_two,
          raw_model=raw_model,
          model_v2_log_mono_time_ns=frame_model_v2_mono_time_ns,
          car_state_log_mono_time_ns=frame_car_state_mono_time_ns,
          live_tracks_log_mono_time_ns=frame_live_tracks_mono_time_ns,
          radard_service_association_status=radard_service_status,
          radard_service_association_provenance=service_associations,
          radard_gate_eligible=radard_gate_eligible,
          radar_state_log_mono_time_ns=int(msg.logMonoTime),
          longitudinal_plan_log_mono_time_ns=None if plan is None else plan.log_mono_time_ns,
          longitudinal_plan_solver_execution_time_s=None if plan is None else plan.solver_execution_time_s,
          planner_radar_state_log_mono_time_ns=planner_radar.target_log_mono_time_ns,
          planner_radar_state_candidates_ns=planner_radar.candidate_log_mono_times_ns,
          planner_radar_resolution=planner_radar.resolution,
          planner_radar_reason=planner_radar.reason,
          planner_inputs=planner_context.inputs,
          planner_service_log_mono_time_ns=planner_context.service_log_mono_time_ns,
          planner_service_association_provenance=planner_context.service_provenance,
          recorded_effective_cruise_mps=effective_cruise.speed_mps,
          recorded_effective_cruise_limiter=effective_cruise.limiter,
          recorded_effective_cruise_provenance=effective_cruise.provenance,
          recorded_effective_cruise_status=effective_cruise.status,
          planner_context_status=planner_context.status,
          planner_context_reason=planner_context.reason,
          gas_pressed=frame_car_state.gas_pressed,
          personality=int(latest_selfdrive_state.personality.raw),
          planner_accel_mps2=None if plan is None else plan.a_target_mps2,
          planner_source=None if plan is None else plan.source,
          params_snapshot=frame_params_snapshot,
          param_updates=frame_param_updates,
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
  exact_runs = split_exact_replay_runs(frames)
  radard_association_counts: dict[str, int] = {}
  for frame in frames:
    status = frame.radard_service_association_status
    radard_association_counts[status] = radard_association_counts.get(status, 0) + 1
  exact_car_state_join_count = sum(
    isinstance(frame.radard_service_association_provenance.get("carState"), dict) and
    frame.radard_service_association_provenance["carState"].get("status") == "exact"
    for frame in frames
  )
  service_join_diagnostics = {
    "radarStateFrameCount": radar_state_count,
    "uniqueRadarStatePublicationCount": len(processed_radar_state_mono_times),
    "exactCarStateJoinCount": exact_car_state_join_count,
    "droppedMissingCarStateJoinCount": len(orphan_car_state_references),
    "missingCarStateJoinExamples": orphan_car_state_references[:20],
    "exactContiguousRunCount": len(exact_runs),
    "longestExactContiguousRunFrameCount": max((len(run) for run in exact_runs), default=0),
    "longestExactContiguousRunDurationS": max(
      (run[-1].t_s - run[0].t_s for run in exact_runs),
      default=0.0,
    ),
    "radardServiceAssociationStatusCounts": radard_association_counts,
    "radardGateEligibleFrameCount": sum(1 for frame in frames if frame.radard_gate_eligible),
    "coreInputConflictingDuplicateClocks": {
      service: list(clocks)
      for service, clocks in replay_index.conflicting_input_mono_times_by_service.items()
    },
    "coreInputIdenticalDuplicateCounts": dict(replay_index.identical_duplicate_counts_by_service),
  }
  if orphan_car_state_references:
    warnings.warn(
      f"Dropped {len(orphan_car_state_references)}/{radar_state_count} radarState frames from route "
      f"{route_row['route_key']}: their nonzero carStateMonoTime has no exact logged carState",
      RuntimeWarning,
      stacklevel=2,
    )
  driver_marks = _resolve_driver_marks(
    raw_driver_mark_presses,
    longflag_payloads,
    frames,
    route_key=metadata.route_key,
    first_radar_time_ns=first_radar_time,
  )
  return RouteScanResult(
    route_id=int(route_row["route_id"]),
    metadata=metadata,
    observed_params=observed_params,
    frames=frames,
    service_join_diagnostics=service_join_diagnostics,
    planner_route_start_provenance=dict(replay_index.planner_route_start_provenance),
    driver_marks=driver_marks,
    longflag_payloads=longflag_payloads,
  )


def extract_ev6_episodes(conn,
                         *,
                         route_keys: list[str] | None = None,
                         bundle_root: str | Path = ".cache/longitudinal_harness/snapshots",
                         route_start_replay: bool = False) -> list[dict[str, Any]]:
  bundle_root_path = Path(bundle_root)
  bundle_root_path.mkdir(parents=True, exist_ok=True)
  recorded = []
  for route_row in get_route_rows(conn, route_keys=route_keys):
    scan = load_route_scan(conn, route_row)
    clear_route_extractions(conn, scan.route_id)
    candidates = detect_episode_candidates(scan)
    for candidate in candidates:
      try:
        bundle_path = write_episode_bundle(
          scan,
          candidate,
          bundle_root_path,
          **({"route_start_replay": True} if route_start_replay else {}),
        )
      except EpisodeNotReplayableError as exc:
        if candidate.episode_type != "driver_mark":
          recorded.append({
            "routeId": scan.route_id,
            "episodeKey": candidate.episode_key,
            "episodeType": candidate.episode_type,
            "status": "not_evaluated",
            "reason": str(exc),
            "confidence": candidate.confidence,
          })
          continue
        # A human mark must never vanish because replay dependencies are
        # incomplete. Catalog it without a bundle so the incident stays visible.
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
          bundle_path=None,
          notes_json={**candidate.notes_json, "bundleUnavailableReason": str(exc)},
        ))
        recorded.append({
          "routeId": scan.route_id,
          "episodeId": episode_id,
          "episodeKey": candidate.episode_key,
          "episodeType": candidate.episode_type,
          "status": "recorded_without_bundle",
          "reason": str(exc),
          "confidence": candidate.confidence,
        })
        continue
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
        "status": "recorded",
      })
    # A press that never anchored to a frame produces no candidate above, so it
    # would otherwise leave no trace anywhere: no bundle, no catalog row, no
    # report entry. The usual cause is a >1 s radarState gap at the press, i.e.
    # a RadarD dropout -- one of the top reasons a human reaches for the flag in
    # the first place. Catalog it bundle-less, carrying the press logMonoTime.
    # getattr, not attribute access: test_catalog.py drives this function with a
    # SimpleNamespace stand-in for RouteScanResult. A real scan always has the field.
    for mark in getattr(scan, "driver_marks", ()) or ():
      if mark.frame_index is not None:
        continue
      orphan = _orphan_driver_mark_candidate(scan, mark)
      reason = _orphan_driver_mark_reason(mark)
      episode_id = upsert_episode(conn, EpisodeCatalogRecord(
        route_id=scan.route_id,
        episode_key=orphan.episode_key,
        episode_type=orphan.episode_type,
        seg_start=orphan.seg_start,
        seg_end=orphan.seg_end,
        t_start_s=orphan.t_start_s,
        t_end_s=orphan.t_end_s,
        confidence=orphan.confidence,
        extractor_version=EXTRACTOR_VERSION,
        source_event_count=1,
        metrics_json=orphan.metrics,
        bundle_path=None,
        notes_json={**orphan.notes_json, "bundleUnavailableReason": reason},
      ))
      recorded.append({
        "routeId": scan.route_id,
        "episodeId": episode_id,
        "episodeKey": orphan.episode_key,
        "episodeType": orphan.episode_type,
        "status": "orphan_recorded_without_bundle",
        "reason": reason,
        "confidence": orphan.confidence,
        "driverMarkPressLogMonoTime": mark.press_log_mono_time_ns,
      })
  conn.commit()
  return recorded


def detect_episode_candidates(scan: RouteScanResult) -> list[EpisodeCandidate]:
  if not scan.frames:
    return []

  candidates: list[EpisodeCandidate] = []
  # Missing exact carState joins are dropped by load_route_scan. Run every
  # temporal detector on uninterrupted spans only, so a sparse set of surviving
  # frames cannot masquerade as a continuous recorded replay window.
  for frames in split_exact_replay_runs(scan.frames):
    run_scan = replace(scan, frames=frames)
    candidates.extend(_detect_false_closing(run_scan))
    candidates.extend(_detect_cutin(run_scan))
    candidates.extend(_detect_handoff(run_scan))
    candidates.extend(_detect_dropout(run_scan))
    candidates.extend(_detect_pullaway(run_scan))
    candidates.extend(_detect_approach(run_scan))
  # Deliberately outside the exact-replay run loop: a driver mark is human
  # intent, and a RadarD dropout that splits the runs is itself a plausible
  # cause of whatever was flagged. Detecting per-run would delete the evidence.
  candidates.extend(_detect_driver_mark(scan))
  candidates = _dedupe_candidates(candidates)
  candidates.sort(key=lambda candidate: (candidate.t_start_s, candidate.episode_type))
  return candidates


def split_exact_replay_runs(frames: list[EpisodeFrame]) -> list[list[EpisodeFrame]]:
  if not frames:
    return []
  runs: list[list[EpisodeFrame]] = [[frames[0]]]
  for frame in frames[1:]:
    previous = runs[-1][-1]
    if 0.0 < frame.t_s - previous.t_s <= MAX_EXACT_REPLAY_FRAME_GAP_S:
      runs[-1].append(frame)
    else:
      runs.append([frame])
  return runs


def _parse_longflag(record: Any) -> dict[str, Any] | None:
  """Decode one ``LONGFLAG {...}`` swaglog line into its payload dict.

  Returns None for anything that is not a well-formed LONGFLAG record. This is
  fed every ``logMessage`` in a route, so the substring prefilter deliberately
  runs before any JSON parsing, and no input may ever raise.
  """
  text = record if isinstance(record, str) else str(record)
  if LONGFLAG_PREFIX not in text:
    return None
  try:
    envelope = json.loads(text)
  except (TypeError, ValueError):
    return None
  if not isinstance(envelope, dict):
    return None
  message = envelope.get("msg", "")
  if not isinstance(message, str):
    return None
  marker = message.find(LONGFLAG_PREFIX)
  if marker < 0:
    return None
  try:
    payload = json.loads(message[marker + len(LONGFLAG_PREFIX):])
  except (TypeError, ValueError):
    return None
  return payload if isinstance(payload, dict) else None


def _nearest_sorted_index(sorted_values: list[float], target: float) -> int:
  idx = bisect_left(sorted_values, target)
  if idx <= 0:
    return 0
  if idx >= len(sorted_values):
    return len(sorted_values) - 1
  return idx if (sorted_values[idx] - target) < (target - sorted_values[idx - 1]) else idx - 1


def _nearest_longflag_payload(longflag_payloads: list[dict[str, Any]], press_log_mono_time_ns: int) -> dict[str, Any] | None:
  window_ns = int(DRIVER_MARK_PAYLOAD_WINDOW_S * 1e9)
  best: tuple[int, dict[str, Any]] | None = None
  for entry in longflag_payloads:
    delta = abs(int(entry["logMonoTimeNs"]) - press_log_mono_time_ns)
    if delta > window_ns:
      continue
    if best is None or delta < best[0]:
      best = (delta, entry["payload"])
  return None if best is None else best[1]


def collapse_driver_mark_echoes(raw_presses: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Fold the bookmarkButton/userBookmark echo of one press into a single group.

  MEMBERSHIP is anchored on the first message of a group rather than chained, so
  a long train of presses cannot be swallowed by repeated near-misses.

  The group's reported ``logMonoTimeNs`` is a different question, and it is NOT
  the first message's. ``feedbackd`` publishes bare ``userBookmark`` on paths that
  have nothing to do with the flag button (see is_driver_mark_press_group), so a
  stray echo landing shortly BEFORE a real tap would otherwise become the press
  time of that tap. That time is the cross-tool join key: Phase 2's recorder
  stamps ``pressLogMonoTime`` from ``sm.logMonoTime['bookmarkButton']``, so an
  anchor taken off a ``userBookmark`` desynchronises the device sidecar from the
  offline extraction and ``marks.find_mark_sidecar`` misses.

  So the group reports the FIRST ``bookmarkButton`` in it when one is present --
  that message is the authoritative press, published by both the sidebar button
  and the onroad HUD flag button. Membership still keys off the first message, so
  re-anchoring cannot widen the window and pull in a later, separate press.
  """
  groups: list[dict[str, Any]] = []
  echo_window_ns = int(DRIVER_MARK_ECHO_WINDOW_S * 1e9)
  for press in sorted(raw_presses, key=lambda item: int(item["logMonoTimeNs"])):
    press_ns = int(press["logMonoTimeNs"])
    service = str(press["service"])
    if groups and press_ns - int(groups[-1]["_groupStartNs"]) <= echo_window_ns:
      group = groups[-1]
      if service not in group["services"]:
        group["services"].append(service)
      if service == DRIVER_MARK_PRESS_SERVICE and not group["_anchored"]:
        group["_anchored"] = True
        group["logMonoTimeNs"] = press_ns
        group["segIdx"] = int(press["segIdx"])
      continue
    groups.append({
      "segIdx": int(press["segIdx"]),
      "logMonoTimeNs": press_ns,
      "services": [service],
      "_groupStartNs": press_ns,
      "_anchored": service == DRIVER_MARK_PRESS_SERVICE,
    })
  # Bookkeeping only; the returned shape is the same three keys it always was.
  for group in groups:
    del group["_groupStartNs"]
    del group["_anchored"]
  return groups


def is_driver_mark_press_group(group: dict[str, Any]) -> bool:
  """True only when a collapsed group actually contains a flag-button press.

  ``userBookmark`` alone is NOT a driver mark. ``selfdrive/ui/feedback/feedbackd.py``
  publishes ``userBookmark`` on two paths that never involve the flag button:

  * the LKAS steering-wheel button, when ``not sm['selfdriveStateSP'].mads.available``
    and ``RecordAudioFeedback`` is off (``should_send_bookmark = True`` directly);
  * roughly ``FEEDBACK_MAX_DURATION`` (10 s) AFTER an LKAS press, when
    ``RecordAudioFeedback`` is on and the audio block budget runs out.

  Both would otherwise become confidence-1.0 ``driver_mark`` episodes, the second
  anchored ~10 s away from anything the driver reacted to. Every real press --
  the onroad flag button and the sidebar bookmark button alike -- publishes
  ``bookmarkButton``, which feedbackd then echoes, so requiring it costs nothing.
  """
  return DRIVER_MARK_PRESS_SERVICE in tuple(group.get("services", ()))


def _resolve_driver_marks(raw_presses: list[dict[str, Any]],
                          longflag_payloads: list[dict[str, Any]],
                          frames: list[EpisodeFrame],
                          *,
                          route_key: str,
                          first_radar_time_ns: int | None) -> list[DriverMarkPress]:
  if not raw_presses:
    return []
  frame_times = [frame.t_s for frame in frames]
  marks: list[DriverMarkPress] = []
  for group in collapse_driver_mark_echoes(raw_presses):
    if not is_driver_mark_press_group(group):
      continue
    press_ns = int(group["logMonoTimeNs"])
    t_s = None if first_radar_time_ns is None else (press_ns - int(first_radar_time_ns)) / 1e9
    payload = _nearest_longflag_payload(longflag_payloads, press_ns)
    frame_index: int | None = None
    frame_gap_s: float | None = None
    frame_log_mono_time_ns: int | None = None
    if t_s is not None and frame_times:
      nearest = _nearest_sorted_index(frame_times, t_s)
      frame_gap_s = abs(frame_times[nearest] - t_s)
      if frame_gap_s <= DRIVER_MARK_FRAME_TOLERANCE_S:
        frame_index = nearest
        frame_log_mono_time_ns = frames[nearest].log_mono_time
    # An unanchored press is still evidence that a human flagged something; it
    # is reported as an orphan rather than dropped.
    status = "orphan" if frame_index is None else ("ok" if payload is not None else "no_payload")
    marks.append(DriverMarkPress(
      route_key=route_key,
      seg_idx=int(group["segIdx"]),
      press_log_mono_time_ns=press_ns,
      t_s=t_s,
      services=tuple(group["services"]),
      payload=payload,
      frame_index=frame_index,
      frame_log_mono_time_ns=frame_log_mono_time_ns,
      frame_gap_s=frame_gap_s,
      status=status,
    ))
  return marks


def _driver_mark_notes(mark: DriverMarkPress) -> dict[str, Any]:
  return {
    "driverMarkId": mark.mark_id,
    "driverMarkPressLogMonoTime": mark.press_log_mono_time_ns,
    "driverMarkServices": list(mark.services),
    "driverMarkStatus": mark.status,
    "driverMarkPayload": mark.payload,
  }


def _driver_mark_key_discriminator(mark: DriverMarkPress) -> str:
  """Per-press suffix for episode_key. See EpisodeCandidate.key_discriminator."""
  return f"press{mark.press_log_mono_time_ns}"


def _detect_driver_mark(scan: RouteScanResult) -> list[EpisodeCandidate]:
  """Promote every anchored driver flag press to a full-confidence episode.

  Presses that could not be anchored to a recorded frame produce no candidate --
  there are no frames to bundle -- but they are NOT lost: ``extract_ev6_episodes``
  catalogues them separately via ``_orphan_driver_mark_candidate``.
  """
  candidates: list[EpisodeCandidate] = []
  for mark in scan.driver_marks:
    if mark.frame_index is None:
      continue
    mark_notes = _driver_mark_notes(mark)
    candidates.append(_make_candidate(
      scan,
      mark.frame_index,
      "driver_mark",
      1.0,
      {
        **mark_notes,
        "driverMarkFrameGapS": mark.frame_gap_s,
        "driverMarkSegIdx": mark.seg_idx,
      },
      rank_score=1000.0,
      extra_notes=mark_notes,
      key_discriminator=_driver_mark_key_discriminator(mark),
    ))
  return candidates


def _orphan_driver_mark_candidate(scan: RouteScanResult, mark: DriverMarkPress) -> EpisodeCandidate:
  """A press with no anchorable frame, shaped so it can still be catalogued.

  ``frames`` is empty on purpose: this is never handed to ``write_episode_bundle``.
  The window is derived from the press clock alone so a human can still find the
  moment in the rlog, and the press logMonoTime is the join key for
  ``marks.load_trace`` / the Phase 2 sidecar, neither of which needs a frame.
  """
  pre_s, post_s = WINDOWS_BY_TYPE["driver_mark"]
  t_s = 0.0 if mark.t_s is None else float(mark.t_s)
  # A press before the first recorded radarState has a negative route clock, so
  # the end is floored against the start rather than left inverted in the DB.
  t_start_s = max(0.0, t_s + pre_s)
  t_end_s = max(t_start_s, t_s + post_s)
  mark_notes = _driver_mark_notes(mark)
  return EpisodeCandidate(
    route_id=scan.route_id,
    route_key=scan.metadata.route_key,
    episode_type="driver_mark",
    event_t_s=t_s,
    seg_start=mark.seg_idx,
    seg_end=mark.seg_idx,
    t_start_s=t_start_s,
    t_end_s=t_end_s,
    confidence=1.0,
    rank_score=1000.0,
    metrics={
      **mark_notes,
      "sourceEventCount": 1,
      "driverMarkOrphan": True,
      "driverMarkFrameGapS": mark.frame_gap_s,
      "driverMarkSegIdx": mark.seg_idx,
      # Route-relative, matching DriverMarkPress.t_s -- NOT marks.DriverMark.t_seg_rel_s.
      "driverMarkPressRouteTRelS": mark.t_s,
    },
    notes_json={"driverMarkOrphan": True, **mark_notes},
    frames=[],
    key_discriminator=_driver_mark_key_discriminator(mark),
  )


def _orphan_driver_mark_reason(mark: DriverMarkPress) -> str:
  if mark.t_s is None:
    return "driver mark press has no route clock: the route recorded no radarState at all"
  gap = "unknown" if mark.frame_gap_s is None else f"{mark.frame_gap_s:.2f} s"
  return (
    f"driver mark press could not be anchored to a recorded radarState frame (nearest frame is {gap} away, " +
    f"tolerance {DRIVER_MARK_FRAME_TOLERANCE_S:.1f} s); the press itself is preserved"
  )


def _detect_false_closing(scan: RouteScanResult) -> list[EpisodeCandidate]:
  """Find sustained raw-model recovery contradicted by pessimistic RadarD output."""
  candidates = []
  last_event_t = -math.inf
  evidence_frames = 10  # 0.5 s at model/radard rate
  for idx in range(20, len(scan.frames) - evidence_frames):
    frame = scan.frames[idx]
    if frame.t_s - last_event_t < COOLDOWN_BY_TYPE_S["false_closing"]:
      continue
    window = scan.frames[idx:idx + evidence_frames]
    qualifying = [sample for sample in window if _is_false_closing_sample(sample)]
    if len(qualifying) < 3:
      continue

    track_ids = [sample.lead_one.radar_track_id for sample in window if sample.lead_one.status]
    if not track_ids:
      continue
    dominant_track_id = max(set(track_ids), key=track_ids.count)
    if dominant_track_id == -1 or track_ids.count(dominant_track_id) < int(math.ceil(0.8 * len(track_ids))):
      continue

    worst = max(
      qualifying,
      key=lambda sample: float(sample.raw_lead_one.v_rel_mps) - float(sample.lead_one.v_rel_mps),
    )
    closing_excess = float(worst.raw_lead_one.v_rel_mps) - float(worst.lead_one.v_rel_mps)
    min_planner_accel = min(
      float(sample.planner_accel_mps2)
      for sample in window
      if sample.planner_accel_mps2 is not None
    )
    confidence = min(1.0, 0.55 + min(0.30, closing_excess / 8.0) + min(0.15, len(qualifying) / 20.0))
    candidates.append(_make_candidate(scan, idx, "false_closing", confidence, {
      "rawVRelMps": worst.raw_lead_one.v_rel_mps,
      "publishedVRelMps": worst.lead_one.v_rel_mps,
      "publishedClosingExcessMps": closing_excess,
      "minPlannerAccelMps2": min_planner_accel,
      "radarTrackId": dominant_track_id,
      "evidenceFrameCount": len(qualifying),
    }, rank_score=(closing_excess * 4.0) + max(0.0, -min_planner_accel) + len(qualifying)))
    last_event_t = frame.t_s
  return candidates


def _is_false_closing_sample(frame: EpisodeFrame) -> bool:
  raw = frame.raw_lead_one
  published = frame.lead_one
  if (
    raw is None or not raw.status or raw.v_rel_mps is None or raw.a_lead_k_mps2 is None or
    not published.status or published.v_rel_mps is None or
    frame.planner_accel_mps2 is None or not frame.long_active
  ):
    return False
  return bool(
    raw.model_prob >= 0.85 and
    raw.v_rel_mps >= -1.0 and
    raw.a_lead_k_mps2 >= -0.5 and
    published.v_rel_mps <= -1.25 and
    (raw.v_rel_mps - published.v_rel_mps) >= 1.0 and
    frame.planner_accel_mps2 <= -0.30
  )


def write_episode_bundle(
  scan: RouteScanResult,
  candidate: EpisodeCandidate,
  bundle_root: Path,
  *,
  route_start_replay: bool = False,
) -> Path:
  episode_root = bundle_root / f"route_{scan.route_id}_{scan.metadata.route_key}" / f"{candidate.episode_type}_{int(round(candidate.event_t_s * 1000.0)):09d}"
  candidate_first_publish_ns = candidate.frames[0].radar_state_log_mono_time_ns
  candidate_first_idx = next(
    idx for idx, frame in enumerate(scan.frames)
    if frame.radar_state_log_mono_time_ns == candidate_first_publish_ns
  )
  route_start_seed_times: set[int] = set()
  if route_start_replay:
    proof = scan.planner_route_start_provenance
    if proof.get("status") != "diagnostic":
      raise EpisodeNotReplayableError(
        f"route-start planner replay for {candidate.episode_key} is unavailable: " +
        "; ".join(str(reason) for reason in proof.get("reasons", ["missing process-start diagnostic proof"]))
      )
    first_plan_ns = int(proof["firstPlannerPlanMonoTimeNs"])
    first_planner_idx = next((
      idx for idx, frame in enumerate(scan.frames)
      if frame.longitudinal_plan_log_mono_time_ns == first_plan_ns
    ), None)
    if first_planner_idx is None:
      raise EpisodeNotReplayableError("route-start proof names a first planner publication absent from the loaded scan")
    first_radar_ns = int(proof["firstPlannerRadarStateMonoTimeNs"])
    seed_idx = next((
      idx for idx, frame in enumerate(scan.frames[:first_planner_idx + 1])
      if frame.radar_state_log_mono_time_ns == first_radar_ns
    ), None)
    if seed_idx is None or seed_idx > first_planner_idx:
      raise EpisodeNotReplayableError("route-start proof names a radarState outside the first planner frame")
    if candidate_first_idx <= first_planner_idx:
      raise EpisodeNotReplayableError("episode begins before the first proven planner publication")
    dependency_start_idx = seed_idx
    if seed_idx < first_planner_idx:
      route_start_seed_times.add(first_radar_ns)
  else:
    dependency_start_idx = candidate_first_idx
    while dependency_start_idx > 0:
      previous = scan.frames[dependency_start_idx - 1]
      current = scan.frames[dependency_start_idx]
      if not (0.0 < current.t_s - previous.t_s <= MAX_EXACT_REPLAY_FRAME_GAP_S):
        break
      dependency_start_idx -= 1
      # Include the first frame that reaches/passes the warmup boundary. At the
      # normal 20 Hz RadarD cadence, stopping before it leaves the evaluation
      # window about one frame short of the strict 15.0 s coverage requirement.
      if candidate.frames[0].t_s - previous.t_s >= RADARD_DEPENDENCY_WARMUP_S:
        break
  dependency_frames = scan.frames[dependency_start_idx:candidate_first_idx]
  replay_frames = [*dependency_frames, *candidate.frames]
  replay_publish_times = {frame.radar_state_log_mono_time_ns for frame in replay_frames}
  scan_frames_by_publish_time = {frame.radar_state_log_mono_time_ns: frame for frame in scan.frames}
  scheduler_seed_frames: list[EpisodeFrame] = []
  # Planner replay may consume the RadarD publication immediately preceding
  # the first dependency frame. Seed every external target needed by the
  # replay frames, then suppress planner evaluation for those seed-only rows.
  # Looking only at candidate frames misses this first warmup predecessor and
  # makes closed-loop replay abort before reaching the evaluation window.
  for frame in replay_frames:
    if frame.planner_radar_resolution not in ("exact", "legacy_timing_unique"):
      continue
    target_ns = frame.planner_radar_state_log_mono_time_ns
    if target_ns is None or target_ns in replay_publish_times:
      continue
    seed_frame = scan_frames_by_publish_time.get(target_ns)
    if seed_frame is None:
      raise EpisodeNotReplayableError(
        f"planner radar target {target_ns} for {candidate.episode_key} is outside the loaded route; " +
        "pull one more pre-roll segment before building this snapshot"
      )
    scheduler_seed_frames.append(seed_frame)
  bundle_frames = sorted(
    {
      frame.radar_state_log_mono_time_ns: frame
      for frame in [*scheduler_seed_frames, *replay_frames]
    }.values(),
    key=lambda frame: frame.log_mono_time,
  )
  if route_start_replay:
    post_start_param_changes = [
      frame for frame in bundle_frames[1:]
      if frame.param_updates
    ]
    if post_start_param_changes:
      examples = [
        {
          "logMonoTimeNs": frame.log_mono_time,
          "keys": sorted(frame.param_updates),
        }
        for frame in post_start_param_changes[:3]
      ]
      raise EpisodeNotReplayableError(
        "route-start planner replay has behavior-manifest parameter transitions without an exact planner read clock: " +
        json.dumps(examples, separators=(",", ":"), sort_keys=True)
      )
  dependency_frame_times = {frame.radar_state_log_mono_time_ns for frame in dependency_frames}
  scheduler_seed_times = route_start_seed_times | {
    frame.radar_state_log_mono_time_ns
    for frame in scheduler_seed_frames
    if frame.radar_state_log_mono_time_ns not in replay_publish_times
  }
  warmup_ready_t_s = bundle_frames[0].t_s + RADARD_DEPENDENCY_WARMUP_S
  raw_required_frames = [
    frame for frame in bundle_frames
    if frame.radar_state_log_mono_time_ns not in route_start_seed_times
  ]
  raw_frame_complete = [
    frame.raw_model is not None and frame.raw_lead_one is not None and frame.raw_lead_two is not None
    for frame in raw_required_frames
  ]
  if any(raw_frame_complete) and not all(raw_frame_complete):
    missing_count = len(raw_frame_complete) - sum(raw_frame_complete)
    message = f"partial raw-model coverage for {candidate.episode_key}: {missing_count}/{len(raw_frame_complete)} frames are incomplete"
    raise EpisodeNotReplayableError(message)
  raw_replay = bool(raw_frame_complete) and all(raw_frame_complete)
  exact_live_tracks_not_received = all(
    frame.live_tracks_log_mono_time_ns == 0 and
    isinstance(frame.radard_service_association_provenance.get("liveTracks"), dict) and
    frame.radard_service_association_provenance["liveTracks"].get("status") == "exact"
    for frame in raw_required_frames
  )
  if raw_replay and not scan.metadata.radar_unavailable and not exact_live_tracks_not_received:
    raise EpisodeNotReplayableError(
      f"raw RadarD replay for {candidate.episode_key} cannot substitute empty liveTracks on a radar-capable capture " +
      "without an exact supported replay contract with a zero liveTracks clock"
    )
  required_service_clocks_complete = [
    frame.model_v2_log_mono_time_ns is not None and frame.model_v2_log_mono_time_ns > 0 and
    frame.car_state_log_mono_time_ns is not None and frame.car_state_log_mono_time_ns > 0
    for frame in raw_required_frames
  ]
  if raw_replay and not all(required_service_clocks_complete):
    missing_count = len(required_service_clocks_complete) - sum(required_service_clocks_complete)
    raise EpisodeNotReplayableError(
      f"partial required RadarD service-clock coverage for {candidate.episode_key}: " +
      f"{missing_count}/{len(required_service_clocks_complete)} frames are incomplete"
    )
  timeline = []
  prev_status = {"leadOne": False, "leadTwo": False}
  event_marked = False
  for frame_idx, frame in enumerate(bundle_frames):
    published_seed = frame.radar_state_log_mono_time_ns in route_start_seed_times
    source_lead_one = frame.raw_lead_one if raw_replay and not published_seed else frame.lead_one
    source_lead_two = frame.raw_lead_two if raw_replay and not published_seed else frame.lead_two
    assert source_lead_one is not None and source_lead_two is not None
    lead_one = _directive_from_frame(
      source_lead_one,
      not prev_status["leadOne"] and source_lead_one.status,
      exact_model_prob=raw_replay,
    )
    lead_two = _directive_from_frame(
      source_lead_two,
      not prev_status["leadTwo"] and source_lead_two.status,
      exact_model_prob=raw_replay,
    )
    event_name = None
    if not event_marked and frame.t_s >= candidate.event_t_s:
      event_name = _event_name_for_episode(candidate.episode_type)
      event_marked = True
    scheduler_seed = frame.radar_state_log_mono_time_ns in scheduler_seed_times
    dependency_warmup = frame.radar_state_log_mono_time_ns in dependency_frame_times
    warmup_ready = frame.t_s + 1e-9 >= warmup_ready_t_s
    replay_warmup_status = "ready" if warmup_ready else "warmup"
    replay_warmup_reason = (
      f"RadarD/planner dependency history is at least {RADARD_DEPENDENCY_WARMUP_S:.1f} s"
      if warmup_ready else
      f"advancing dependency state; only {max(0.0, frame.t_s - bundle_frames[0].t_s):.3f}/"
      f"{RADARD_DEPENDENCY_WARMUP_S:.1f} s elapsed"
    )
    radard_gate_eligible = bool(frame.radard_gate_eligible and warmup_ready and not scheduler_seed)
    replay_reference = _replay_reference_from_frame(frame) if raw_replay else {}
    if scheduler_seed:
      replay_reference["plannerRadarResolution"] = "missing"
      replay_reference["plannerRadarResolutionReason"] = "scheduler seed frame; predecessor intentionally outside episode"
      if isinstance(replay_reference.get("longitudinalPlan"), dict):
        replay_reference["longitudinalPlan"]["radarStateLogMonoTimeNs"] = None
        replay_reference["longitudinalPlan"]["radarStateCandidatesNs"] = []
        replay_reference["longitudinalPlan"]["radarResolution"] = "missing"
        replay_reference["longitudinalPlan"]["radarResolutionReason"] = (
          "scheduler seed frame; predecessor intentionally outside episode"
        )
    replay_reference["replayWarmupStatus"] = replay_warmup_status
    replay_reference["replayWarmupReason"] = replay_warmup_reason
    replay_reference["radardGateEligible"] = radard_gate_eligible
    if not radard_gate_eligible:
      if "radarState" in replay_reference:
        replay_reference["radarStateDiagnostic"] = replay_reference.pop("radarState")
      blocked_reason = (
        replay_warmup_reason if not warmup_ready
        else f"RadarD service association is {frame.radard_service_association_status}, not explicit exact"
      )
      replay_reference["plannerRadarResolution"] = "unscorable"
      replay_reference["plannerRadarResolutionReason"] = blocked_reason
      if isinstance(replay_reference.get("longitudinalPlan"), dict):
        replay_reference["longitudinalPlan"]["radarResolution"] = "unscorable"
        replay_reference["longitudinalPlan"]["radarResolutionReason"] = blocked_reason
    timeline.append(StepInput(
      t_s=float(frame.t_s - bundle_frames[0].t_s),
      cruise_speed_mps=frame.cruise_speed_mps,
      lead_one=lead_one,
      lead_two=lead_two,
      event=event_name,
      note=(
        f"{scan.metadata.route_key}:{candidate.episode_type}" +
        (":dependency_warmup" if dependency_warmup else "") +
        (":scheduler_seed" if scheduler_seed else "")
      ),
      pitch_rad=frame.pitch_rad,
      force_decel=frame.force_decel,
      experimental_mode=frame.experimental_mode,
      recorded_v_ego_mps=frame.v_ego_mps if raw_replay else None,
      recorded_a_ego_mps2=frame.a_ego_mps2 if raw_replay else None,
      long_active=frame.long_active if raw_replay else None,
      personality=frame.personality if raw_replay else None,
      raw_model=frame.raw_model if raw_replay and not published_seed else None,
      recorded_perception_mode="published_seed" if published_seed else "radard" if raw_replay else None,
      recorded_model_v2_log_mono_time_ns=frame.model_v2_log_mono_time_ns if raw_replay and not published_seed else None,
      recorded_car_state_log_mono_time_ns=frame.car_state_log_mono_time_ns if raw_replay and not published_seed else None,
      recorded_live_tracks_log_mono_time_ns=frame.live_tracks_log_mono_time_ns if raw_replay and not published_seed else None,
      radard_service_association_status=frame.radard_service_association_status if raw_replay else None,
      radard_service_association_provenance=(
        dict(frame.radard_service_association_provenance) if raw_replay else {}
      ),
      radard_gate_eligible=radard_gate_eligible if raw_replay else False,
      replay_warmup_status=replay_warmup_status if raw_replay else None,
      replay_warmup_reason=replay_warmup_reason if raw_replay else None,
      recorded_radar_state_log_mono_time_ns=frame.radar_state_log_mono_time_ns if raw_replay else None,
      recorded_longitudinal_plan_log_mono_time_ns=frame.longitudinal_plan_log_mono_time_ns if raw_replay else None,
      recorded_longitudinal_plan_solver_execution_time_s=(
        frame.longitudinal_plan_solver_execution_time_s if raw_replay else None
      ),
      planner_radar_state_log_mono_time_ns=(
        None if scheduler_seed or not raw_replay else frame.planner_radar_state_log_mono_time_ns
      ),
      planner_radar_state_candidates_ns=(
        [] if scheduler_seed or not raw_replay else list(frame.planner_radar_state_candidates_ns)
      ),
      planner_radar_resolution=("missing" if scheduler_seed and raw_replay else frame.planner_radar_resolution if raw_replay else None),
      recorded_planner_inputs=dict(frame.planner_inputs) if raw_replay else {},
      recorded_planner_service_log_mono_time_ns=(
        dict(frame.planner_service_log_mono_time_ns) if raw_replay else {}
      ),
      planner_service_association_provenance=(
        dict(frame.planner_service_association_provenance) if raw_replay else {}
      ),
      recorded_effective_cruise_mps=frame.recorded_effective_cruise_mps if raw_replay else None,
      recorded_effective_cruise_limiter=frame.recorded_effective_cruise_limiter if raw_replay else None,
      recorded_effective_cruise_provenance=frame.recorded_effective_cruise_provenance if raw_replay else None,
      recorded_effective_cruise_status=frame.recorded_effective_cruise_status if raw_replay else None,
      planner_context_status=frame.planner_context_status if raw_replay else None,
      planner_context_reason=frame.planner_context_reason if raw_replay else None,
      recorded_gas_pressed=frame.gas_pressed if raw_replay else None,
      # The first frame's parameter snapshot is the bundle baseline; only later
      # deltas belong in the timeline.
      param_updates={} if frame_idx == 0 else dict(frame.param_updates),
      replay_reference=replay_reference,
    ))
    prev_status["leadOne"] = source_lead_one.status
    prev_status["leadTwo"] = source_lead_two.status

  episode_params = dict(bundle_frames[0].params_snapshot or scan.observed_params)
  planner_state_claim: dict[str, Any] | None = None
  if route_start_replay:
    non_seed_steps = [
      step for step in timeline
      if step.recorded_perception_mode != "published_seed"
    ]
    if not non_seed_steps or any(step.recorded_longitudinal_plan_log_mono_time_ns is None for step in non_seed_steps):
      raise EpisodeNotReplayableError("route-start prefix contains an unpaired planner update after its scheduler seed")
    planner_state_claim = build_route_start_initialization_claim(
      route_start_proof=scan.planner_route_start_provenance,
      steps=timeline,
      # Production plannerd constructs LongitudinalPlanner(CP) with these
      # defaults; the recorded ego state remains a separate replay input.
      initial_speed_mps=0.0,
      initial_accel_mps2=0.0,
      params=episode_params,
    )
    for step in timeline:
      step.replay_reference["plannerStateInitializationProvenance"] = dict(planner_state_claim)
  association_status_counts: dict[str, int] = {}
  for step in timeline:
    status = step.radard_service_association_status or "missing"
    association_status_counts[status] = association_status_counts.get(status, 0) + 1
  ready_steps = [step for step in timeline if step.replay_warmup_status == "ready"]
  gate_eligible_steps = [step for step in ready_steps if step.radard_gate_eligible]
  bundle_radard_gate_eligible = bool(ready_steps and len(gate_eligible_steps) == len(ready_steps))
  bundle_radard_service_status = (
    "conflict" if association_status_counts.get("conflict", 0) else
    "missing" if association_status_counts.get("missing", 0) else
    "inferred" if association_status_counts.get("inferred", 0) else
    "exact" if association_status_counts.get("exact", 0) else "missing"
  )
  if not scan.metadata.radar_unavailable and int(episode_params.get("HyundaiLongitudinalTuning", "0")) != 0:
    controller_mode = "shaped"
  else:
    # radar-unavailable routes always ran the CarController's no-radar EMA stage
    controller_mode = "device"
  bundle = SnapshotBundle(
    path=episode_root,
    vehicle={
      "name": f"{scan.metadata.route_key}_{candidate.episode_type}",
      "routeId": scan.route_id,
      "routeKey": scan.metadata.route_key,
      "episodeType": candidate.episode_type,
      "topology": scan.metadata.topology,
      "controllerMode": controller_mode,
      # Raw-aware bundles replay captured modelV2 through real RadarD. Legacy logs
      # without modelV2 retain the published-radarState direct fallback.
      "perceptionFilter": "radard" if raw_replay else "direct",
      "egoReplayMode": "recorded" if raw_replay else "plant",
      "rawModelReplay": raw_replay,
      "radardServiceAssociationStatusCounts": association_status_counts,
      "radardServiceAssociationStatus": bundle_radard_service_status,
      "radardServiceAssociationProvenance": {
        "representativeFrame": dict(candidate.frames[0].radard_service_association_provenance),
        "requirement": "modelV2, carState, and liveTracks associations must all be explicit exact",
      },
      "radardGateEligible": bundle_radard_gate_eligible,
      "radardGateEligibleFrameCount": len(gate_eligible_steps),
      "radardReadyFrameCount": len(ready_steps),
      "radardDependencyWarmupS": RADARD_DEPENDENCY_WARMUP_S,
      "dependencyHistoryS": max(0.0, candidate.frames[0].t_s - bundle_frames[0].t_s),
      "plannerStateInitializationMethod": "route_start_replay" if planner_state_claim is not None else "missing",
      "plannerRouteStartProvenance": (
        dict(scan.planner_route_start_provenance) if planner_state_claim is not None else {}
      ),
      "liveTracksPayloadMode": (
        "empty_not_received" if raw_replay and exact_live_tracks_not_received else
        "empty_radarless" if raw_replay else
        "published_radar_state_direct"
      ),
      "warmupS": max(0.0, candidate.event_t_s - bundle_frames[0].t_s),
      "paramSource": "initData+carControlSP" if any(key.startswith("Longitudinal.LiveTune.") for key in episode_params) else "carControlSP-partial",
      "paramManifest": captured_param_manifest(episode_params),
      "openpilotLongitudinalControl": scan.metadata.openpilot_longitudinal,
      "radarUnavailable": scan.metadata.radar_unavailable,
      "safetyParam": scan.metadata.safety_param,
      "sourceRoot": scan.metadata.source_root,
      "segStart": bundle_frames[0].seg_idx,
      "segEnd": candidate.seg_end,
      "tStartS": bundle_frames[0].t_s,
      "evaluationTStartS": candidate.t_start_s,
      "tEndS": candidate.t_end_s,
      "confidence": candidate.confidence,
      "metrics": candidate.metrics,
      "cpFlags": scan.metadata.notes_json.get("flags"),
      "pcmCruise": scan.metadata.notes_json.get("pcmCruise"),
      "spFlags": scan.metadata.notes_json.get("spFlags"),
      "spSafetyParam": scan.metadata.notes_json.get("spSafetyParam"),
      "longitudinalActuatorDelay": scan.metadata.notes_json.get("longitudinalActuatorDelay"),
      "vEgoStopping": scan.metadata.notes_json.get("vEgoStopping"),
      "vEgoStarting": scan.metadata.notes_json.get("vEgoStarting"),
      "stoppingDecelRate": scan.metadata.notes_json.get("stoppingDecelRate"),
      "startAccel": scan.metadata.notes_json.get("startAccel"),
      "startingState": scan.metadata.notes_json.get("startingState"),
      "gitCommit": scan.metadata.notes_json.get("gitCommit"),
      "gitBranch": scan.metadata.notes_json.get("gitBranch"),
      "gitRemote": scan.metadata.notes_json.get("gitRemote"),
      "gitDirty": scan.metadata.notes_json.get("gitDirty"),
      "gitDiffEmpty": scan.metadata.notes_json.get("gitDiffEmpty"),
      "gitDiffSha256": scan.metadata.notes_json.get("gitDiffSha256"),
      "runtimeDeviceType": scan.metadata.notes_json.get("runtimeDeviceType"),
      "runtimePlatform": scan.metadata.notes_json.get("runtimePlatform"),
      "runtimeMachine": scan.metadata.notes_json.get("runtimeMachine"),
      "runtimeKernelVersion": scan.metadata.notes_json.get("runtimeKernelVersion"),
      "runtimeOsVersion": scan.metadata.notes_json.get("runtimeOsVersion"),
    },
    params=episode_params,
    timeline=timeline,
    initial_speed_mps=bundle_frames[0].v_ego_mps,
    initial_accel_mps2=bundle_frames[0].a_ego_mps2,
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
                    rank_score: float | None = None,
                    extra_notes: dict[str, Any] | None = None,
                    key_discriminator: str | None = None) -> EpisodeCandidate:
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
    notes_json={"eventLogMonoTime": event_frame.log_mono_time, **(extra_notes or {})},
    frames=episode_frames,
    key_discriminator=key_discriminator,
  )


def summarize_episode_frames(frames: list[EpisodeFrame]) -> dict[str, Any]:
  primary_gaps = []
  lead_switch_count = 0
  prev_slot = None
  source_transition_count = 0
  prev_source = None
  for frame in frames:
    slot, lead = _primary_lead_with_slot(frame)
    if lead is not None and lead.d_rel_m is not None:
      primary_gaps.append(lead.d_rel_m)
    if slot is not None and prev_slot is not None and slot != prev_slot:
      lead_switch_count += 1
    if slot is not None:
      prev_slot = slot
    if frame.planner_source is not None and prev_source is not None and frame.planner_source != prev_source:
      source_transition_count += 1
    if frame.planner_source is not None:
      prev_source = frame.planner_source
  return {
    "minGapM": min(primary_gaps) if primary_gaps else None,
    "maxGapM": max(primary_gaps) if primary_gaps else None,
    "leadSwitchCount": lead_switch_count,
    "sourceEventCount": lead_switch_count + 1,
    "plannerSourceTransitionCount": source_transition_count,
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
  if candidate.episode_type == "driver_mark":
    # A driver mark is suppressed ONLY when it is literally the same press.
    #
    # This branch is deliberately ahead of the event_t_s proximity test below.
    # For a driver mark, event_t_s is not the press clock: it is the route time
    # of the recorded radarState frame the press was ANCHORED to, chosen by
    # nearest-neighbour within DRIVER_MARK_FRAME_TOLERANCE_S (1.0 s). Frames are
    # 20 Hz and go missing whenever RadarD drops out, so two presses as much as
    # 2.0 s apart can anchor to the same surviving frame and land on an identical
    # event_t_s -- and the second deliberate press would then be dropped here,
    # before it ever reached the catalog, no matter how distinct its episode_key.
    #
    # Echo collapse is the one and only authority on "same press", and it already
    # ran upstream in collapse_driver_mark_echoes on the exact, boot-unique press
    # logMonoTime (DRIVER_MARK_ECHO_WINDOW_S). Anything that arrives here with a
    # different press time is a second human press: keep it.
    discriminator = candidate.key_discriminator
    return discriminator is not None and discriminator == existing.key_discriminator
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


def _lead_from_model(model_msg, slot: int, v_ego_mps: float) -> RouteLeadFrame | None:
  if len(model_msg.leadsV3) <= slot:
    return None
  lead_msg = model_msg.leadsV3[slot]
  if not len(lead_msg.x) or not len(lead_msg.v) or not len(lead_msg.a):
    return None
  model_v_ego = float(model_msg.velocity.x[0]) if len(model_msg.velocity.x) else float(v_ego_mps)
  lead_dict = get_RadarState_from_vision(lead_msg, float(v_ego_mps), model_v_ego)
  add_path_relative_lead_metrics(lead_dict, model_msg, lead_msg)
  return RouteLeadFrame(
    status=bool(float(lead_msg.prob) > 0.0),
    d_rel_m=float(lead_dict["dRel"]),
    v_rel_mps=float(lead_dict["vRel"]),
    v_lead_mps=float(lead_dict["vLead"]),
    a_lead_k_mps2=float(lead_dict["aLeadK"]),
    v_lead_k_mps=float(lead_dict["vLeadK"]),
    model_prob=float(lead_msg.prob),
    y_rel_m=float(lead_dict["yRel"]),
    d_path_m=float(lead_dict["dPath"]),
    v_lat_mps=float(lead_dict["vLat"]),
    radar=False,
    radar_track_id=-1,
  )


def _directive_from_frame(lead: RouteLeadFrame, acquisition_reset: bool, *, exact_model_prob: bool = False) -> LeadDirective:
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
    exact_model_prob=exact_model_prob,
  )


def _lead_reference(lead: RouteLeadFrame) -> dict[str, Any]:
  return {
    "status": lead.status,
    "dRelM": lead.d_rel_m,
    "vRelMps": lead.v_rel_mps,
    "vLeadMps": lead.v_lead_mps,
    "aLeadKMps2": lead.a_lead_k_mps2,
    "modelProb": lead.model_prob,
    "radarTrackId": lead.radar_track_id,
  }


def _replay_reference_from_frame(frame: EpisodeFrame) -> dict[str, Any]:
  return {
    "logMonoTimeNs": frame.log_mono_time,
    "radardServiceAssociationStatus": frame.radard_service_association_status,
    "radardServiceAssociationProvenance": dict(frame.radard_service_association_provenance),
    "radardGateEligible": frame.radard_gate_eligible,
    "plannerRadarResolution": frame.planner_radar_resolution,
    "plannerRadarResolutionReason": frame.planner_radar_reason,
    "plannerContextStatus": frame.planner_context_status,
    "plannerContextReason": frame.planner_context_reason,
    "plannerStateInitializationProvenance": {
      "status": "missing",
      "version": 0,
      "appliedAtReplayStart": False,
      "reason": (
        "RadarState replay-inputs v1/v2 and longitudinalPlan replay-inputs v1 capture external service snapshots "
        "but not the recurrent LongitudinalPlanner/LongitudinalMpc state"
      ),
    },
    "serviceLogMonoTimeNs": {
      "modelV2": frame.model_v2_log_mono_time_ns,
      "carState": frame.car_state_log_mono_time_ns,
      "liveTracks": frame.live_tracks_log_mono_time_ns,
    },
    "plannerServiceLogMonoTimeNs": dict(frame.planner_service_log_mono_time_ns),
    "plannerServiceAssociationProvenance": dict(frame.planner_service_association_provenance),
    "radarState": {
      "logMonoTimeNs": frame.radar_state_log_mono_time_ns,
      "leadOne": _lead_reference(frame.lead_one),
      "leadTwo": _lead_reference(frame.lead_two),
    },
    "longitudinalPlan": {
      "logMonoTimeNs": frame.longitudinal_plan_log_mono_time_ns,
      "solverExecutionTimeS": frame.longitudinal_plan_solver_execution_time_s,
      "aTargetMps2": frame.planner_accel_mps2,
      "source": frame.planner_source,
      "radarStateLogMonoTimeNs": frame.planner_radar_state_log_mono_time_ns,
      "radarStateCandidatesNs": list(frame.planner_radar_state_candidates_ns),
      "radarResolution": frame.planner_radar_resolution,
      "radarResolutionReason": frame.planner_radar_reason,
      "effectiveCruiseMps": frame.recorded_effective_cruise_mps,
      "effectiveCruiseLimiter": frame.recorded_effective_cruise_limiter,
      "effectiveCruiseProvenance": frame.recorded_effective_cruise_provenance,
      "effectiveCruiseStatus": frame.recorded_effective_cruise_status,
      "contextStatus": frame.planner_context_status,
      "contextReason": frame.planner_context_reason,
    },
  }


def _event_name_for_episode(episode_type: str) -> str:
  return {
    "pullaway": "pullaway_start",
    "cutin": "lead_reveal",
    "handoff": "handoff_reveal",
    "dropout": "dropout_start",
    "approach": "approach_start",
    "false_closing": "false_closing_start",
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


def _extract_init_params(init_data) -> dict[str, str]:
  params: dict[str, str] = {}
  for entry in init_data.params.entries:
    key = str(entry.key)
    if not _is_replay_param(key):
      continue
    try:
      value = bytes(entry.value).decode("utf-8")
    except (UnicodeDecodeError, TypeError, ValueError):
      continue
    if "\x00" in value or len(value) > 10_000:
      continue
    params[key] = value
  return params


def _extract_init_provenance(init_data) -> dict[str, Any]:
  device_type = str(init_data.deviceType)
  embedded_arm64 = device_type in {"tici", "tizi", "mici"}
  provenance: dict[str, Any] = {
    "gitCommit": str(init_data.gitCommit),
    "gitBranch": str(init_data.gitBranch),
    "gitRemote": str(init_data.gitRemote),
    "gitDirty": bool(init_data.dirty),
    "runtimeDeviceType": device_type,
    "runtimePlatform": "linux" if embedded_arm64 else None,
    "runtimeMachine": "aarch64" if embedded_arm64 else None,
    "runtimeKernelVersion": str(init_data.kernelVersion),
    "runtimeOsVersion": str(init_data.osVersion),
  }
  git_diff: bytes | None = None
  for entry in init_data.params.entries:
    if str(entry.key) != "GitDiff":
      continue
    try:
      git_diff = bytes(entry.value)
    except (TypeError, ValueError):
      git_diff = None
    break
  if git_diff is not None:
    provenance["gitDiffEmpty"] = len(git_diff) == 0
    provenance["gitDiffSha256"] = hashlib.sha256(git_diff).hexdigest()
  return provenance


def _is_replay_param(key: str) -> bool:
  return bool(
    key in DEFAULT_PARAM_VALUES or
    key.startswith(("Longitudinal.LiveTune.", "LongTuning", "VibeTune.")) or
    key.startswith(("VisionTurnSpeedControl", "SpeedLimit", "RTI", "Weather")) or
    key in {
      "AccelPersonality",
      "LongitudinalPersonality",
      "HyundaiLongitudinalTuning",
      "VibePersonalityEnabled",
      "VibeFollowPersonalityEnabled",
      "VibeAccelPersonalityEnabled",
      "DynamicExperimentalControl",
      "ExperimentalMode",
      "IsMetric",
      "ObjectHazardEnabled",
    }
  )


def _merge_param_updates(current: dict[str, str], pending: dict[str, str], updates: dict[str, str]) -> None:
  for key, value in updates.items():
    text_value = str(value)
    if current.get(key) == text_value:
      continue
    current[key] = text_value
    pending[key] = text_value


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
