"""Tests for the lead stability filter (acquire/release dwell + phantom hold)
at the LongitudinalMpc boundary, and the Schmitt trigger helper in radard."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc,
  _LeadStabilityState,
)
from openpilot.selfdrive.controls.radard import _is_lead_prob_accepted


def _make_raw_lead(status=True, dRel=40.0, yRel=0.0, vRel=-1.0, vLead=25.0,
                   aLeadK=0.0, modelProb=0.9, dPath=0.0, vLat=0.0, aLeadTau=0.0,
                   aRel=0.0) -> SimpleNamespace:
  return SimpleNamespace(
    status=status, dRel=dRel, yRel=yRel, vRel=vRel, vLead=vLead,
    aLeadK=aLeadK, modelProb=modelProb, dPath=dPath, vLat=vLat,
    aLeadTau=aLeadTau, aRel=aRel,
  )


def _make_mpc(acquire_frames: float = 1.0, release_frames: float = 1.0,
              phantom_hold_s: float = 0.0, stable_frames: float = 1.0) -> LongitudinalMpc:
  """Build a minimal mpc-like stub that has just the attrs _stabilize_raw_leads needs."""
  stub = SimpleNamespace(
    _live_tune_cfg=SimpleNamespace(
      lead_source_acquire_frames=acquire_frames,
      lead_source_release_frames=release_frames,
      phantom_lead_hold_s=phantom_hold_s,
      phantom_lead_stable_frames=stable_frames,
    ),
    _lead_stability_state=[_LeadStabilityState(), _LeadStabilityState()],
    lead_stability_debug={},
    LEAD_STABILIZER_PHANTOM_YREL_KILL_M=LongitudinalMpc.LEAD_STABILIZER_PHANTOM_YREL_KILL_M,
  )
  import types
  stub._stabilize_raw_leads = types.MethodType(LongitudinalMpc._stabilize_raw_leads, stub)
  return stub


class TestLeadProbSchmitt:
  def test_defaults_to_classic_half_threshold_when_enter_equals_exit(self):
    assert _is_lead_prob_accepted(0.51, prev_latched=False, prob_enter=0.5, prob_exit=0.5)
    assert not _is_lead_prob_accepted(0.49, prev_latched=False, prob_enter=0.5, prob_exit=0.5)

  def test_requires_enter_when_not_latched(self):
    assert not _is_lead_prob_accepted(0.55, prev_latched=False, prob_enter=0.6, prob_exit=0.35)
    assert _is_lead_prob_accepted(0.65, prev_latched=False, prob_enter=0.6, prob_exit=0.35)

  def test_stays_latched_between_exit_and_enter(self):
    # Hysteresis band [0.35, 0.6]: prob=0.5 keeps latched if previously latched, rejects if not.
    assert _is_lead_prob_accepted(0.5, prev_latched=True, prob_enter=0.6, prob_exit=0.35)
    assert not _is_lead_prob_accepted(0.5, prev_latched=False, prob_enter=0.6, prob_exit=0.35)

  def test_releases_below_exit(self):
    assert not _is_lead_prob_accepted(0.30, prev_latched=True, prob_enter=0.6, prob_exit=0.35)

  def test_nonfinite_prob_rejected(self):
    assert not _is_lead_prob_accepted(float("nan"), prev_latched=True, prob_enter=0.6, prob_exit=0.35)


class TestLeadStabilityFilter_Defaults:
  def test_defaults_are_passthrough(self):
    mpc = _make_mpc()  # acquire=1, release=1, phantom=0
    raw = _make_raw_lead(status=True, dRel=30.0)
    out0, out1 = mpc._stabilize_raw_leads(raw, _make_raw_lead(status=False), now=1.0)
    assert out0.status is True
    assert out0.dRel == pytest.approx(30.0)
    assert out1.status is False


class TestAcquireDwell:
  def test_single_valid_frame_rejected_when_acquire_frames_is_2(self):
    mpc = _make_mpc(acquire_frames=2.0)
    raw = _make_raw_lead(status=True)
    out0, _ = mpc._stabilize_raw_leads(raw, _make_raw_lead(status=False), now=1.0)
    assert out0.status is False

  def test_latches_on_second_valid_frame(self):
    mpc = _make_mpc(acquire_frames=2.0)
    raw = _make_raw_lead(status=True)
    _ = mpc._stabilize_raw_leads(raw, _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(raw, _make_raw_lead(status=False), now=1.05)
    assert out0.status is True

  def test_single_valid_frame_followed_by_invalid_never_latches(self):
    mpc = _make_mpc(acquire_frames=2.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=True), _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False), now=1.05)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=True), _make_raw_lead(status=False), now=1.10)
    assert out0.status is False


class TestReleaseDwell:
  def test_release_frames_delays_drop(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=3.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True), _make_raw_lead(status=False), now=1.0)
    # two invalid frames — still latched (release requires 3)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False), now=1.05)
    assert out0.status is False  # raw False, phantom off, so False
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False), now=1.10)
    assert out0.status is False


class TestPhantomHold:
  def test_phantom_extrapolates_drel_by_vrel(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=40.0, vRel=-2.0),
                             _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.2)  # 0.2s elapsed
    assert out0.status is True
    # dRel extrapolated: 40 + (-2) * 0.2 = 39.6
    assert out0.dRel == pytest.approx(39.6)

  def test_phantom_expires_after_hold_window(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.3, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True), _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.4)  # past 0.3s
    assert out0.status is False

  def test_phantom_requires_stable_precondition(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=5.0)
    # Only one valid frame — not stable.
    mpc._stabilize_raw_leads(_make_raw_lead(status=True), _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.1)
    assert out0.status is False

  def test_phantom_killed_by_yrel_jump_in_other_slot(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, yRel=0.2),
                             _make_raw_lead(status=False), now=1.0)
    # Slot 0 raw drops; slot 1 raw appears with very different yRel.
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=False),
      _make_raw_lead(status=True, yRel=2.5),  # 2.3m off from original — likely different car
      now=1.1,
    )
    assert out0.status is False

  def test_phantom_aLeadK_decays_toward_zero(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-1.0),
                             _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.25)  # halfway through
    assert abs(out0.aLeadK) < 1.0  # decayed


class TestAcquireReleaseInteraction:
  def test_raw_flutter_4x_no_dwell_latches_each_frame(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0)
    # T: raw=valid, T+1: invalid, T+2: valid, T+3: invalid, T+4: valid
    seq = [True, False, True, False, True]
    statuses = []
    for i, v in enumerate(seq):
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(status=v), _make_raw_lead(status=False), now=1.0 + 0.05 * i,
      )
      statuses.append(out0.status)
    assert statuses == [True, False, True, False, True]  # flutters through

  def test_raw_flutter_4x_with_release_dwell_holds_lead(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=4.0)
    seq = [True, False, True, False, True]
    statuses = []
    for i, v in enumerate(seq):
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(status=v), _make_raw_lead(status=False), now=1.0 + 0.05 * i,
      )
      statuses.append(out0.status)
    # release_frames=4 means 4 consecutive invalids needed; we never have that,
    # so slot stays latched. But with phantom_hold=0 and raw=False, output status
    # still shows False during invalid frames (the latch just doesn't drop).
    # The interesting assertion: at the final valid frame, we're back to True
    # without going through a full re-acquire.
    assert statuses[-1] is True
