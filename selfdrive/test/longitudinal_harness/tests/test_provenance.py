from __future__ import annotations

import pytest

from openpilot.selfdrive.test.longitudinal_harness.provenance import (
  classify_planner_runtime_provenance,
  classify_replay_provenance,
  exact_planner_state_initialization,
  well_formed_planner_state_initialization_claim,
)


_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _clean_metadata(commit: str) -> dict[str, object]:
  return {
    "gitCommit": commit,
    "gitBranch": "chauffeur-exp01",
    "gitRemote": "git@github.com:example/chauffeur.git",
    "gitDirty": False,
    "gitDiffSha256": _EMPTY_SHA256,
    "gitDiffEmpty": True,
    "runtimeDeviceType": "tici",
    "runtimePlatform": "linux",
    "runtimeMachine": "aarch64",
    "runtimeOsVersion": "12.6",
    "runtimeKernelVersion": "Linux version 4.9.103 test",
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


def test_planner_runtime_requires_matching_device_platform_and_machine() -> None:
  capture = _clean_metadata("a" * 40)
  exact = classify_planner_runtime_provenance(capture, dict(capture))
  mac = dict(capture, runtimeDeviceType="pc", runtimePlatform="darwin", runtimeMachine="arm64")
  mismatch = classify_planner_runtime_provenance(capture, mac)

  assert exact.status == "exact"
  assert exact.gateEligible is True
  assert mismatch.status == "mismatch"
  assert mismatch.gateEligible is False
  assert any("deviceType differs" in reason for reason in mismatch.reasons)


def test_planner_runtime_missing_identity_is_unknown() -> None:
  capture = _clean_metadata("a" * 40)
  del capture["runtimeMachine"]
  result = classify_planner_runtime_provenance(capture, _clean_metadata("a" * 40))

  assert result.status == "unknown"
  assert result.gateEligible is False
  assert any("capture planner runtime machine is missing" in reason for reason in result.reasons)


@pytest.mark.parametrize("field", ["runtimeOsVersion", "runtimeKernelVersion"])
def test_planner_runtime_missing_os_or_kernel_is_unknown(field: str) -> None:
  capture = _clean_metadata("a" * 40)
  del capture[field]
  result = classify_planner_runtime_provenance(capture, _clean_metadata("a" * 40))

  assert result.status == "unknown"
  assert result.gateEligible is False


@pytest.mark.parametrize(
  ("field", "value", "reason_fragment"),
  [
    ("runtimeOsVersion", "13.0", "osVersion differs"),
    ("runtimeKernelVersion", "Linux version 5.15 different", "kernelVersion differs"),
  ],
)
def test_planner_runtime_os_and_kernel_must_match(field: str, value: str, reason_fragment: str) -> None:
  capture = _clean_metadata("a" * 40)
  replay = dict(capture, **{field: value})
  result = classify_planner_runtime_provenance(capture, replay)

  assert result.status == "mismatch"
  assert result.gateEligible is False
  assert any(reason_fragment in reason for reason in result.reasons)


def test_planner_runtime_version_whitespace_is_normalized() -> None:
  capture = _clean_metadata("a" * 40)
  replay = dict(capture, runtimeOsVersion=" 12.6\n", runtimeKernelVersion="Linux   version 4.9.103 test\n")

  assert classify_planner_runtime_provenance(capture, replay).status == "exact"


def test_planner_state_initialization_requires_applied_hashed_v1_seed() -> None:
  exact = {
    "status": "exact",
    "version": 1,
    "appliedAtReplayStart": True,
    "stateSha256": "1" * 64,
  }
  assert well_formed_planner_state_initialization_claim(exact) is True
  assert exact_planner_state_initialization(exact) is False
  assert exact_planner_state_initialization(exact, restoration_verified=True) is True
  assert well_formed_planner_state_initialization_claim(dict(exact, appliedAtReplayStart=False)) is False
  assert well_formed_planner_state_initialization_claim(dict(exact, stateSha256="not-a-digest")) is False
  assert well_formed_planner_state_initialization_claim(dict(exact, stateSha256="0" * 64)) is False


def test_planner_state_initialization_rejects_empty_payload_digest() -> None:
  claim = {
    "status": "exact",
    "version": 1,
    "appliedAtReplayStart": True,
    "stateSha256": _EMPTY_SHA256,
  }

  assert well_formed_planner_state_initialization_claim(claim) is False
  assert exact_planner_state_initialization(claim, restoration_verified=True) is False
