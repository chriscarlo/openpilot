from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import re
from typing import Any


ROUTE_START_REPLAY_METHOD = "route_start_replay"
PLANNER_STATE_INITIALIZATION_VERSION = 1
ROUTE_START_DIAGNOSTIC_REASON = (
  "source rlogs do not carry writer-attested publication completeness or a restorable planner checkpoint"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def replay_input_prefix_sha256(steps: Iterable[Any]) -> str:
  """Hash only inputs that can advance RadarD/planner recurrent state.

  Recorded output references, human notes, and event labels are deliberately
  excluded. The digest is recomputed by ``run_harness`` before it constructs
  either daemon, so an edited/truncated route prefix cannot retain even its
  diagnostic state-initialization claim.
  """
  payload = []
  for step in steps:
    row = dict(step.to_json())
    row.pop("replayReference", None)
    row.pop("event", None)
    row.pop("note", None)
    payload.append(row)
  return _sha256_json(payload)


def build_route_start_initialization_claim(
  *,
  route_start_proof: Mapping[str, Any],
  steps: Iterable[Any],
  initial_speed_mps: float,
  initial_accel_mps2: float,
  params: Mapping[str, Any],
) -> dict[str, Any]:
  step_list = list(steps)
  if not _well_formed_route_start_proof(route_start_proof):
    raise ValueError("route-start planner diagnostic proof is missing or invalid")
  if not step_list:
    raise ValueError("route-start planner replay requires a non-empty prefix")

  prefix_sha256 = replay_input_prefix_sha256(step_list)
  initial_params_sha256 = _sha256_json({str(key): str(value) for key, value in params.items()})
  state_sha256 = _sha256_json({
    "method": ROUTE_START_REPLAY_METHOD,
    "routeStartProof": dict(route_start_proof),
    "initialSpeedMps": _finite_number(initial_speed_mps, "initial_speed_mps"),
    "initialAccelMps2": _finite_number(initial_accel_mps2, "initial_accel_mps2"),
    "initialParamsSha256": initial_params_sha256,
  })
  first_clock = _step_reference_clock(step_list[0])
  last_clock = _step_reference_clock(step_list[-1])
  if first_clock is None or last_clock is None or last_clock < first_clock:
    raise ValueError("route-start planner replay requires ordered positive reference clocks")
  return {
    # A segment-zero replay is useful for diagnostics, but the logger does not
    # prove that every producer update made it into the rlog.  It therefore
    # cannot be a formal recurrent-state restoration claim, even when every
    # message that is present pairs cleanly.
    "status": "diagnostic",
    "version": PLANNER_STATE_INITIALIZATION_VERSION,
    "method": ROUTE_START_REPLAY_METHOD,
    "appliedAtReplayStart": True,
    "formalFidelityEligible": False,
    "diagnosticReason": ROUTE_START_DIAGNOSTIC_REASON,
    "stateSha256": state_sha256,
    "prefixSha256": prefix_sha256,
    "prefixFrameCount": len(step_list),
    "prefixFirstLogMonoTimeNs": first_clock,
    "prefixLastLogMonoTimeNs": last_clock,
    "initialParamsSha256": initial_params_sha256,
    "routeStartProof": dict(route_start_proof),
  }


def verify_route_start_initialization_claim(
  claim: Any,
  *,
  steps: Iterable[Any],
  initial_speed_mps: float,
  initial_accel_mps2: float,
  params: Mapping[str, Any],
) -> bool:
  if not isinstance(claim, Mapping) or not _route_start_claim_shape_exact(claim):
    return False
  step_list = list(steps)
  if not step_list or claim.get("prefixFrameCount") != len(step_list):
    return False
  first_clock = _step_reference_clock(step_list[0])
  last_clock = _step_reference_clock(step_list[-1])
  if claim.get("prefixFirstLogMonoTimeNs") != first_clock or claim.get("prefixLastLogMonoTimeNs") != last_clock:
    return False
  if claim.get("prefixSha256") != replay_input_prefix_sha256(step_list):
    return False

  params_sha256 = _sha256_json({str(key): str(value) for key, value in params.items()})
  if claim.get("initialParamsSha256") != params_sha256:
    return False
  expected_state_sha256 = _sha256_json({
    "method": ROUTE_START_REPLAY_METHOD,
    "routeStartProof": dict(claim["routeStartProof"]),
    "initialSpeedMps": _finite_number(initial_speed_mps, "initial_speed_mps"),
    "initialAccelMps2": _finite_number(initial_accel_mps2, "initial_accel_mps2"),
    "initialParamsSha256": params_sha256,
  })
  return claim.get("stateSha256") == expected_state_sha256


def well_formed_route_start_initialization_claim(claim: Any) -> bool:
  return isinstance(claim, Mapping) and _route_start_claim_shape_exact(claim)


def _route_start_claim_shape_exact(claim: Mapping[str, Any]) -> bool:
  return bool(
    claim.get("status") == "diagnostic" and
    claim.get("version") == PLANNER_STATE_INITIALIZATION_VERSION and
    claim.get("method") == ROUTE_START_REPLAY_METHOD and
    claim.get("appliedAtReplayStart") is True and
    claim.get("formalFidelityEligible") is False and
    claim.get("diagnosticReason") == ROUTE_START_DIAGNOSTIC_REASON and
    _valid_sha256(claim.get("stateSha256")) and
    _valid_sha256(claim.get("prefixSha256")) and
    _valid_sha256(claim.get("initialParamsSha256")) and
    isinstance(claim.get("prefixFrameCount"), int) and not isinstance(claim.get("prefixFrameCount"), bool) and
    claim["prefixFrameCount"] > 0 and
    _positive_int(claim.get("prefixFirstLogMonoTimeNs")) and
    _positive_int(claim.get("prefixLastLogMonoTimeNs")) and
    claim["prefixLastLogMonoTimeNs"] >= claim["prefixFirstLogMonoTimeNs"] and
    _well_formed_route_start_proof(claim.get("routeStartProof"))
  )


def _well_formed_route_start_proof(proof: Any) -> bool:
  if not isinstance(proof, Mapping):
    return False
  positive_fields = (
    "captureFirstLogMonoTimeNs",
    "captureLastLogMonoTimeNs",
    "startOfRouteMonoTimeNs",
    "plannerProcessStartMonoTimeNs",
    "radardProcessStartMonoTimeNs",
    "firstPlannerPlanMonoTimeNs",
    "firstPlannerModelMonoTimeNs",
    "firstPlannerRadarStateMonoTimeNs",
  )
  if not all(_positive_int(proof.get(field)) for field in positive_fields):
    return False
  if not (
    proof.get("status") == "diagnostic" and
    proof.get("version") == 1 and
    proof.get("formalFidelityEligible") is False and
    proof.get("sourcePublicationCompletenessAttested") is False and
    proof.get("loadedSegmentStart") == 0 and
    proof.get("segmentsContiguous") is True and
    proof.get("plannerProcessUnique") is True and
    proof.get("radardProcessUnique") is True and
    proof.get("plannerManagerIdentityExact") is True and
    proof.get("plannerPublicationsPairedExactly") is True and
    proof.get("firstPlannerModelIsFirstLoggedModel") is True
  ):
    return False
  capture_first = int(proof["captureFirstLogMonoTimeNs"])
  capture_last = int(proof["captureLastLogMonoTimeNs"])
  planner_start = int(proof["plannerProcessStartMonoTimeNs"])
  radard_start = int(proof["radardProcessStartMonoTimeNs"])
  route_start = int(proof["startOfRouteMonoTimeNs"])
  first_plan = int(proof["firstPlannerPlanMonoTimeNs"])
  first_model = int(proof["firstPlannerModelMonoTimeNs"])
  first_radar = int(proof["firstPlannerRadarStateMonoTimeNs"])
  return bool(
    capture_first < min(planner_start, radard_start) <= capture_last and
    capture_first <= route_start <= first_plan <= capture_last and
    max(planner_start, radard_start) < first_plan and
    capture_first <= min(first_radar, first_model) and
    max(first_radar, first_model) <= first_plan
  )


def _step_reference_clock(step: Any) -> int | None:
  reference = getattr(step, "replay_reference", None)
  value = reference.get("logMonoTimeNs") if isinstance(reference, Mapping) else None
  return int(value) if _positive_int(value) else None


def _finite_number(value: Any, name: str) -> float:
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"{name} must be finite")
  return number


def _positive_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_sha256(value: Any) -> bool:
  return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None and set(value) != {"0"}


def _sha256_json(payload: Any) -> str:
  encoded = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


__all__ = [
  "PLANNER_STATE_INITIALIZATION_VERSION",
  "ROUTE_START_DIAGNOSTIC_REASON",
  "ROUTE_START_REPLAY_METHOD",
  "build_route_start_initialization_claim",
  "replay_input_prefix_sha256",
  "verify_route_start_initialization_claim",
  "well_formed_route_start_initialization_claim",
]
