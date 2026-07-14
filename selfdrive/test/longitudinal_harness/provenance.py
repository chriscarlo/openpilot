from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Literal


ProvenanceStatus = Literal["exact", "instrumentation_only", "counterfactual", "unknown"]

_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@dataclass(frozen=True)
class ProvenanceResult:
  """Classification of the code provenance used for a recorded replay.

  ``accepted`` means the requested replay mode was explicit enough to run. It
  does not make the result eligible for a fidelity gate: ``gateEligible`` is
  deliberately true only for an exact capture/replay revision match.
  """

  status: ProvenanceStatus
  gateEligible: bool
  accepted: bool
  reasons: tuple[str, ...]
  capturedCommit: str | None
  replayCommit: str | None

  @property
  def gate_eligible(self) -> bool:
    return self.gateEligible

  def as_dict(self) -> dict[str, Any]:
    return {
      "status": self.status,
      "gateEligible": self.gateEligible,
      "accepted": self.accepted,
      "reasons": list(self.reasons),
      "capturedCommit": self.capturedCommit,
      "replayCommit": self.replayCommit,
    }


@dataclass(frozen=True)
class _GitSnapshot:
  commit: str | None
  dirty: bool | None
  diff_empty: bool | None
  diff_sha256: str | None


def _normalize_text(value: Any) -> str | None:
  if value is None:
    return None
  normalized = str(value).strip()
  return normalized or None


def _normalize_sha(value: Any) -> str | None:
  normalized = _normalize_text(value)
  return normalized.lower() if normalized is not None else None


def _normalize_commit(value: Any) -> str | None:
  normalized = _normalize_sha(value)
  if (
    normalized is None
    or set(normalized) == {"0"}
    or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", normalized) is None
  ):
    return None
  return normalized


def _normalize_diff_sha256(value: Any) -> str | None:
  normalized = _normalize_sha(value)
  if normalized is None or set(normalized) == {"0"} or re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
    return None
  return normalized


def _normalize_bool(value: Any) -> bool | None:
  if isinstance(value, bool):
    return value
  if isinstance(value, int) and value in (0, 1):
    return bool(value)
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes"):
      return True
    if normalized in ("0", "false", "no"):
      return False
  return None


def _snapshot(metadata: Mapping[str, Any] | None, *, infer_from_diff_digest: bool) -> _GitSnapshot:
  values = metadata or {}
  commit = _normalize_commit(values.get("gitCommit"))
  dirty = _normalize_bool(values.get("gitDirty"))
  diff_empty = _normalize_bool(values.get("gitDiffEmpty"))
  diff_sha256 = _normalize_diff_sha256(values.get("gitDiffSha256"))

  if infer_from_diff_digest and "gitDiffSha256" in values:
    raw_digest = values.get("gitDiffSha256")
    inferred_empty = raw_digest == "" or diff_sha256 == _EMPTY_SHA256
    if diff_empty is None:
      diff_empty = inferred_empty
    if dirty is None:
      dirty = not inferred_empty

  return _GitSnapshot(commit=commit, dirty=dirty, diff_empty=diff_empty, diff_sha256=diff_sha256)


def _unknown_reasons(label: str, snapshot: _GitSnapshot) -> list[str]:
  reasons: list[str] = []
  if snapshot.commit is None:
    reasons.append(f"{label} git commit is missing or invalid")
  if snapshot.dirty is None:
    reasons.append(f"{label} git dirty state is missing or invalid")
  if snapshot.diff_empty is None:
    reasons.append(f"{label} git diff-empty state is missing or invalid")
  if snapshot.diff_sha256 is None:
    reasons.append(f"{label} git diff digest is missing or invalid")

  if snapshot.dirty is False and snapshot.diff_empty is False:
    reasons.append(f"{label} metadata is inconsistent: clean checkout has a non-empty diff")
  elif snapshot.dirty is True and snapshot.diff_empty is not False:
    # A dirty checkout with an empty tracked diff can contain untracked or
    # otherwise unhashed content. Its source identity is therefore unknown.
    reasons.append(f"{label} dirty checkout is not represented by a non-empty diff")
  elif snapshot.diff_empty is True and snapshot.diff_sha256 != _EMPTY_SHA256:
    reasons.append(f"{label} metadata is inconsistent: empty diff has a non-empty digest")
  elif snapshot.diff_empty is False and snapshot.diff_sha256 == _EMPTY_SHA256:
    reasons.append(f"{label} metadata is inconsistent: non-empty diff has the empty digest")
  return reasons


def classify_replay_provenance(
  captured_metadata: Mapping[str, Any] | None,
  replay_metadata: Mapping[str, Any] | None = None,
  *,
  current_commit: str | None = None,
  current_diff_sha256: str | None = None,
  current_dirty: bool | None = None,
  current_diff_empty: bool | None = None,
  counterfactual: bool = False,
  acknowledge_instrumentation_only: bool = False,
) -> ProvenanceResult:
  """Classify capture/replay source identity without reading the live git tree.

  Callers may provide a complete replay metadata mapping, explicit current git
  values, or both (explicit values override the mapping). A mismatch is rejected
  by default. Diagnostic replay of known-mismatched code requires either
  ``counterfactual=True`` or an explicit
  ``acknowledge_instrumentation_only=True`` acknowledgement.

  Unknown capture/replay metadata cannot be acknowledged into a known class and
  never becomes fidelity-gate eligible.
  """
  if counterfactual and acknowledge_instrumentation_only:
    raise ValueError("counterfactual and instrumentation-only modes are mutually exclusive")

  replay_values = dict(replay_metadata or {})
  if current_commit is not None:
    replay_values["gitCommit"] = current_commit
  if current_diff_sha256 is not None:
    replay_values["gitDiffSha256"] = current_diff_sha256
  if current_dirty is not None:
    replay_values["gitDirty"] = current_dirty
  if current_diff_empty is not None:
    replay_values["gitDiffEmpty"] = current_diff_empty

  captured = _snapshot(captured_metadata, infer_from_diff_digest=False)
  replay = _snapshot(replay_values, infer_from_diff_digest=True)
  unknown_reasons = _unknown_reasons("capture", captured) + _unknown_reasons("replay", replay)
  if unknown_reasons:
    return ProvenanceResult(
      status="unknown",
      gateEligible=False,
      accepted=False,
      reasons=tuple(unknown_reasons),
      capturedCommit=captured.commit,
      replayCommit=replay.commit,
    )

  mismatch_reasons: list[str] = []
  if captured.commit != replay.commit:
    mismatch_reasons.append(f"git commit differs: capture={captured.commit}, replay={replay.commit}")

  if captured.dirty != replay.dirty:
    mismatch_reasons.append(
      f"git dirty state differs: capture={captured.dirty}, replay={replay.dirty}"
    )
  elif captured.dirty:
    if captured.diff_sha256 != replay.diff_sha256:
      mismatch_reasons.append(
        f"git diff digest differs: capture={captured.diff_sha256}, replay={replay.diff_sha256}"
      )

  if not mismatch_reasons:
    return ProvenanceResult(
      status="exact",
      gateEligible=True,
      accepted=True,
      reasons=("capture and replay commit and working-tree state match exactly",),
      capturedCommit=captured.commit,
      replayCommit=replay.commit,
    )

  if acknowledge_instrumentation_only:
    return ProvenanceResult(
      status="instrumentation_only",
      gateEligible=False,
      accepted=True,
      reasons=tuple(mismatch_reasons + ["caller explicitly acknowledged the mismatch as instrumentation-only"]),
      capturedCommit=captured.commit,
      replayCommit=replay.commit,
    )

  if counterfactual:
    return ProvenanceResult(
      status="counterfactual",
      gateEligible=False,
      accepted=True,
      reasons=tuple(mismatch_reasons + ["caller explicitly requested counterfactual replay"]),
      capturedCommit=captured.commit,
      replayCommit=replay.commit,
    )

  return ProvenanceResult(
    status="unknown",
    gateEligible=False,
    accepted=False,
    reasons=tuple(mismatch_reasons + [
      "revision mismatch requires explicit counterfactual or acknowledged instrumentation-only mode",
    ]),
    capturedCommit=captured.commit,
    replayCommit=replay.commit,
  )


__all__ = ["ProvenanceResult", "ProvenanceStatus", "classify_replay_provenance"]
