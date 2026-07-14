from __future__ import annotations

import pytest

from openpilot.selfdrive.test.longitudinal_harness.provenance import classify_replay_provenance


_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _clean_metadata(commit: str) -> dict[str, object]:
  return {
    "gitCommit": commit,
    "gitBranch": "chauffeur-exp01",
    "gitRemote": "git@github.com:example/chauffeur.git",
    "gitDirty": False,
    "gitDiffSha256": _EMPTY_SHA256,
    "gitDiffEmpty": True,
  }


def _dirty_metadata(commit: str, digest: str | None) -> dict[str, object]:
  return {
    "gitCommit": commit,
    "gitBranch": "chauffeur-exp01",
    "gitRemote": "git@github.com:example/chauffeur.git",
    "gitDirty": True,
    "gitDiffSha256": digest,
    "gitDiffEmpty": False,
  }


def test_exact_clean_revision_is_the_only_default_gate_eligible_class() -> None:
  result = classify_replay_provenance(
    _clean_metadata("a" * 40),
    current_commit="a" * 40,
    current_diff_sha256=_EMPTY_SHA256,
  )

  assert result.status == "exact"
  assert result.accepted
  assert result.gateEligible
  assert result.gate_eligible
  assert result.as_dict()["gateEligible"] is True


def test_commit_mismatch_requires_explicit_counterfactual_mode() -> None:
  capture = _clean_metadata("a" * 40)
  replay = _clean_metadata("b" * 40)

  rejected = classify_replay_provenance(capture, replay)
  explicit = classify_replay_provenance(capture, replay, counterfactual=True)

  assert rejected.status == "unknown"
  assert not rejected.accepted
  assert not rejected.gateEligible
  assert any("requires explicit counterfactual" in reason for reason in rejected.reasons)
  assert explicit.status == "counterfactual"
  assert explicit.accepted
  assert not explicit.gateEligible


def test_dirty_capture_with_unknown_diff_digest_is_unknown() -> None:
  result = classify_replay_provenance(
    _dirty_metadata("a" * 40, None),
    _dirty_metadata("a" * 40, "1" * 64),
    counterfactual=True,
  )

  assert result.status == "unknown"
  assert not result.accepted
  assert not result.gateEligible
  assert any("capture git diff digest is missing or invalid" in reason for reason in result.reasons)


def test_same_commit_with_different_dirty_diffs_is_not_exact() -> None:
  capture = _dirty_metadata("a" * 40, "1" * 64)
  replay = _dirty_metadata("a" * 40, "2" * 64)

  rejected = classify_replay_provenance(capture, replay)
  counterfactual = classify_replay_provenance(capture, replay, counterfactual=True)

  assert rejected.status == "unknown"
  assert not rejected.gateEligible
  assert any("git diff digest differs" in reason for reason in rejected.reasons)
  assert counterfactual.status == "counterfactual"
  assert counterfactual.accepted
  assert not counterfactual.gateEligible


def test_explicit_instrumentation_only_acknowledgement_is_never_exact() -> None:
  capture = _clean_metadata("a" * 40)
  replay = _dirty_metadata("a" * 40, "1" * 64)

  result = classify_replay_provenance(
    capture,
    replay,
    acknowledge_instrumentation_only=True,
  )

  assert result.status == "instrumentation_only"
  assert result.accepted
  assert not result.gateEligible
  assert any("explicitly acknowledged" in reason for reason in result.reasons)


def test_missing_capture_commit_remains_unknown_in_explicit_modes() -> None:
  capture = _clean_metadata("a" * 40)
  del capture["gitCommit"]

  result = classify_replay_provenance(capture, _clean_metadata("a" * 40), counterfactual=True)

  assert result.status == "unknown"
  assert not result.accepted
  assert not result.gateEligible
  assert any("capture git commit is missing" in reason for reason in result.reasons)


def test_matching_dirty_diff_can_be_exact() -> None:
  capture = _dirty_metadata("a" * 40, "1" * 64)
  result = classify_replay_provenance(capture, dict(capture))

  assert result.status == "exact"
  assert result.accepted
  assert result.gateEligible


@pytest.mark.parametrize("sentinel", ["unknown", "0", "same", "not-a-git-identity"])
def test_sentinel_commit_values_never_become_exact(sentinel: str) -> None:
  capture = _clean_metadata(sentinel)
  replay = _clean_metadata(sentinel)

  result = classify_replay_provenance(capture, replay)

  assert result.status == "unknown"
  assert not result.accepted
  assert not result.gateEligible
  assert any("git commit is missing or invalid" in reason for reason in result.reasons)


@pytest.mark.parametrize("sentinel", ["unknown", "same", "0", "not-a-digest"])
def test_sentinel_dirty_diff_digests_never_become_exact(sentinel: str) -> None:
  capture = _dirty_metadata("a" * 40, sentinel)
  replay = _dirty_metadata("a" * 40, sentinel)

  result = classify_replay_provenance(capture, replay)

  assert result.status == "unknown"
  assert not result.accepted
  assert not result.gateEligible
  assert any("git diff digest is missing or invalid" in reason for reason in result.reasons)


def test_dirty_metadata_cannot_claim_the_empty_diff_digest() -> None:
  capture = _dirty_metadata("a" * 40, _EMPTY_SHA256)

  result = classify_replay_provenance(capture, dict(capture))

  assert result.status == "unknown"
  assert any("non-empty diff has the empty digest" in reason for reason in result.reasons)


def test_full_length_zero_git_identities_are_invalid() -> None:
  zero_commit = _clean_metadata("0" * 40)
  commit_result = classify_replay_provenance(zero_commit, dict(zero_commit))
  assert commit_result.status == "unknown"
  assert any("git commit is missing or invalid" in reason for reason in commit_result.reasons)

  zero_diff = _dirty_metadata("a" * 40, "0" * 64)
  diff_result = classify_replay_provenance(zero_diff, dict(zero_diff))
  assert diff_result.status == "unknown"
  assert any("git diff digest is missing or invalid" in reason for reason in diff_result.reasons)


def test_explicit_non_fidelity_modes_are_mutually_exclusive() -> None:
  with pytest.raises(ValueError, match="mutually exclusive"):
    classify_replay_provenance(
      _clean_metadata("a" * 40),
      _clean_metadata("b" * 40),
      counterfactual=True,
      acknowledge_instrumentation_only=True,
    )
