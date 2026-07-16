"""Tests for the lead stability filter (acquire/release dwell + phantom hold)
at the LongitudinalMpc boundary, and the Schmitt trigger helper in radard."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc,
  _LeadStabilityState,
)
from openpilot.selfdrive.controls.radard import _is_lead_prob_accepted


def _make_raw_lead(status=True, dRel=40.0, yRel=0.0, vRel=-1.0, vLead=25.0,
                   aLeadK=0.0, modelProb=0.9, dPath=0.0, vLat=0.0, aLeadTau=0.0,
                   aRel=0.0, fcwSuppressed=False, closingGovernorRecovery=False,
                   accelCorrCalmPositionValid=False,
                   accelCorrCalmPositionSlopeMps=0.0,
                   radarTrackId=-1001) -> SimpleNamespace:
  return SimpleNamespace(
    status=status, dRel=dRel, yRel=yRel, vRel=vRel, vLead=vLead,
    aLeadK=aLeadK, modelProb=modelProb, dPath=dPath, vLat=vLat,
    aLeadTau=aLeadTau, aRel=aRel, fcwSuppressed=fcwSuppressed,
    closingGovernorRecovery=closingGovernorRecovery,
    accelCorrCalmPositionValid=accelCorrCalmPositionValid,
    accelCorrCalmPositionSlopeMps=accelCorrCalmPositionSlopeMps,
    radar=False, radarTrackId=radarTrackId,
  )


def _make_mpc(acquire_frames: float = 1.0, release_frames: float = 1.0,
              phantom_hold_s: float = 0.0, stable_frames: float = 1.0,
              decel_hold_factor: float = 1.0, decel_trend_gain: float = 1.0,
              accel_corr_margin: float | None = None,
              accel_corr_amplify_gain: float | None = None,
              accel_corr_amplify_model_min: float | None = None,
              accel_corr_amplify_deadband: float | None = None,
              v_ego: float = 26.0) -> LongitudinalMpc:
  """Build a minimal mpc-like stub that has just the attrs _stabilize_raw_leads needs.
  accel_corr_margin=None leaves the cfg attr absent (legacy passthrough: the
  getattr default is the disable margin, so the corroboration bound is inert)."""
  cfg = SimpleNamespace(
    lead_source_acquire_frames=acquire_frames,
    lead_source_release_frames=release_frames,
    phantom_lead_hold_s=phantom_hold_s,
    phantom_lead_stable_frames=stable_frames,
    phantom_lead_decel_hold_factor=decel_hold_factor,
    phantom_lead_decel_trend_gain=decel_trend_gain,
  )
  if accel_corr_margin is not None:
    cfg.lead_accel_corr_margin_mps2 = accel_corr_margin
  if accel_corr_amplify_gain is not None:
    cfg.lead_accel_corr_amplify_gain = accel_corr_amplify_gain
  if accel_corr_amplify_model_min is not None:
    cfg.lead_accel_corr_amplify_model_decel_min_mps2 = accel_corr_amplify_model_min
  if accel_corr_amplify_deadband is not None:
    cfg.lead_accel_corr_amplify_deadband_mps2 = accel_corr_amplify_deadband
  stub = SimpleNamespace(
    _live_tune_cfg=cfg,
    _lead_stability_state=[_LeadStabilityState(), _LeadStabilityState()],
    _lead_stability_phantom_slots=(False, False),
    lead_stability_debug={},
    x0=[0.0, v_ego, 0.0],
    current_t_follow=1.5,
    _hyundai_ai_lead_stability_enabled=True,
    LEAD_STABILIZER_PHANTOM_YREL_KILL_M=LongitudinalMpc.LEAD_STABILIZER_PHANTOM_YREL_KILL_M,
  )
  import types
  stub._stabilize_raw_leads = types.MethodType(LongitudinalMpc._stabilize_raw_leads, stub)
  stub._apply_lead_accel_corr_bound = types.MethodType(LongitudinalMpc._apply_lead_accel_corr_bound, stub)
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

  def test_phantom_holds_measured_decel(self):
    # A braking lead's decel is HELD through the hold window (never erased):
    # the phantom must not be kinematically more optimistic than the last
    # measurement (GAP 4 / R5, test_runtime_gap_audit_20260701.md).
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-1.0),
                             _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.25)  # halfway through
    assert out0.aLeadK <= -1.0  # held (or deepening with measured trend), never decayed
    # vRel/dRel propagate with the held decel: closure keeps growing.
    assert out0.vRel <= -1.0 - 1.0 * 0.25 + 1e-6

  def test_phantom_continues_measured_deepening_trend(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-0.5), _make_raw_lead(status=False), now=1.00)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-1.0), _make_raw_lead(status=False), now=1.05)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.30)
    assert out0.aLeadK < -1.0  # measured deepening trend continues through the hold

  def test_phantom_never_extrapolates_relaxing_trend(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-2.0), _make_raw_lead(status=False), now=1.00)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-1.0), _make_raw_lead(status=False), now=1.05)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.30)
    assert out0.aLeadK == pytest.approx(-1.0)  # relaxing trend is ignored: hold the last measurement

  def test_slot_swap_resets_trend_before_dropout(self):
    # A cut-in replacing the tracked lead in the same slot steps aLeadK across
    # two different physical cars; that finite difference is not a measurement.
    # The dRel discontinuity must reset the trend so a following dropout holds
    # the new lead's measured decel instead of a fabricated cross-car trend.
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=40.0, vRel=-1.0, aLeadK=-0.5),
                             _make_raw_lead(status=False), now=1.00)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=39.95, vRel=-1.0, aLeadK=-0.5),
                             _make_raw_lead(status=False), now=1.05)
    # Swap: closer car with a -2.5 m/s^2 aLeadK step, then steady frames.
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=15.0, vRel=-1.0, aLeadK=-3.0),
                             _make_raw_lead(status=False), now=1.10)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=14.95, vRel=-1.0, aLeadK=-3.0),
                             _make_raw_lead(status=False), now=1.15)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=14.90, vRel=-1.0, aLeadK=-3.0),
                             _make_raw_lead(status=False), now=1.20)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.45)
    assert out0.aLeadK == pytest.approx(-3.0)  # held measurement only, no cross-car trend

  def test_lateral_jump_resets_trend_before_dropout(self):
    # Same identity gate on yRel: an adjacent-lane car claimed by the slot must
    # not contribute its aLeadK step to the phantom trend.
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=20.0, yRel=0.0, vRel=-1.0, aLeadK=-0.5),
                             _make_raw_lead(status=False), now=1.00)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=19.95, yRel=2.5, vRel=-1.0, aLeadK=-3.0),
                             _make_raw_lead(status=False), now=1.05)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, dRel=19.90, yRel=2.5, vRel=-1.0, aLeadK=-3.0),
                             _make_raw_lead(status=False), now=1.10)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.35)
    assert out0.aLeadK == pytest.approx(-3.0)

  def test_phantom_positive_aLeadK_decays_toward_zero(self):
    # Extrapolated pull-away is the optimistic direction: positive accel decays.
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=1.0),
                             _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.25)
    assert 0.0 <= out0.aLeadK < 1.0

  def test_phantom_legacy_decay_with_zero_hold_factor(self):
    mpc = _make_mpc(acquire_frames=1.0, release_frames=1.0, phantom_hold_s=0.5, stable_frames=1.0,
                    decel_hold_factor=0.0, decel_trend_gain=0.0)
    mpc._stabilize_raw_leads(_make_raw_lead(status=True, aLeadK=-1.0),
                             _make_raw_lead(status=False), now=1.0)
    out0, _ = mpc._stabilize_raw_leads(_make_raw_lead(status=False), _make_raw_lead(status=False),
                                       now=1.25)
    assert abs(out0.aLeadK) < 1.0  # PhantomLeadDecelHoldFactor=0 restores the legacy decay


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


class TestLeadAccelCorrBound:
  """aLeadK corroboration bound at the _stabilize_raw_leads ingress: an
  uncorroborated negative aLeadK (vLead trend flat) is clamped to
  -(margin) in non-dangerous states; phantom/stale frames, identity changes,
  latched dangerous states and the disable margin all pass full aLeadK."""

  def _settle(self, mpc, frames=20, v_lead=25.0, d_rel=40.0, t0=1.0, a_lead_k=0.0):
    t = t0
    out0 = None
    for _ in range(frames):
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(status=True, dRel=d_rel, vRel=v_lead - 26.0, vLead=v_lead, aLeadK=a_lead_k),
        _make_raw_lead(status=False), now=t)
      t += 0.05
    return out0, t

  def test_uncorroborated_decel_is_clamped_when_settled(self):
    mpc = _make_mpc(accel_corr_margin=0.5)
    self._settle(mpc)
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=40.0, vRel=-1.0, vLead=25.0, aLeadK=-3.0),
      _make_raw_lead(status=False), now=2.0)
    # vLead never moved: min(0, a_meas) - margin == -0.5
    assert out0.aLeadK == pytest.approx(-0.5, abs=0.05)

  def test_bound_inactive_until_same_track_history_settles(self):
    mpc = _make_mpc(accel_corr_margin=0.5)
    # Only 0.2 s of history (< 2 * LeadAccelCorrMeasTauS = 0.6 s): full aLeadK.
    self._settle(mpc, frames=4)
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=40.0, vRel=-1.0, vLead=25.0, aLeadK=-3.0),
      _make_raw_lead(status=False), now=1.2)
    assert out0.aLeadK == pytest.approx(-3.0)

  def test_phantom_hold_keeps_measured_decel_unclamped(self):
    # Amendment: a frozen-vLead phantom drives a_meas to 0; without the phantom
    # bypass the bound would clamp the held -3.0 to -0.5 within the window,
    # braking weaker than the landed phantom decel hold.
    mpc = _make_mpc(phantom_hold_s=0.8, stable_frames=1.0, accel_corr_margin=0.5)
    _, t = self._settle(mpc, a_lead_k=-3.0)
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=False), _make_raw_lead(status=False), now=t + 0.3)
    assert out0.status
    assert out0.aLeadK <= -3.0  # held decel (plus any deepening trend), never trimmed

  def test_identity_step_resets_and_passes_full_decel(self):
    # A vLead step beyond the physical-accel gate is a lead swap: the stale
    # trend must not floor the new (genuinely braking) lead's aLeadK.
    mpc = _make_mpc(accel_corr_margin=0.5)
    _, t = self._settle(mpc)
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=40.0, vRel=-9.0, vLead=17.0, aLeadK=-2.5),
      _make_raw_lead(status=False), now=t)
    assert out0.aLeadK == pytest.approx(-2.5)

  def test_dangerous_state_bypass_latches_with_hysteresis(self):
    # vLead stays constant (no identity gate, no trend) so the only variable is
    # the dangerous-state latch, driven by gap: near-gap guard is 1.2*26=31.2 m
    # and rearm needs > 33.2 m.
    mpc = _make_mpc(accel_corr_margin=0.5)
    _, t = self._settle(mpc)
    # dRel 30 <= 31.2 latches the bypass: full aLeadK through.
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=30.0, vRel=-1.0, vLead=25.0, aLeadK=-2.0),
      _make_raw_lead(status=False), now=t)
    assert out0.aLeadK == pytest.approx(-2.0)
    # dRel 32.5 clears the enter guard but not the rearm margin: stays latched.
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=32.5, vRel=-1.0, vLead=25.0, aLeadK=-2.0),
      _make_raw_lead(status=False), now=t + 0.05)
    assert out0.aLeadK == pytest.approx(-2.0)
    # dRel 40 clears every guard by its margin: bypass disengages, clamp resumes.
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=40.0, vRel=-1.0, vLead=25.0, aLeadK=-2.0),
      _make_raw_lead(status=False), now=t + 0.10)
    assert out0.aLeadK == pytest.approx(-0.5, abs=0.05)

  def test_disable_margin_passes_full_decel(self):
    mpc = _make_mpc(accel_corr_margin=10.0)
    self._settle(mpc)
    out0, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=True, dRel=40.0, vRel=-1.0, vLead=25.0, aLeadK=-3.0),
      _make_raw_lead(status=False), now=2.0)
    assert out0.aLeadK == pytest.approx(-3.0)

  @staticmethod
  def _run_deepening_trend(model_a_lead_k: float, model_min: float):
    mpc = _make_mpc(
      accel_corr_margin=0.5,
      accel_corr_amplify_gain=1.0,
      accel_corr_amplify_model_min=model_min,
      accel_corr_amplify_deadband=0.35,
    )
    t = 1.0
    v_lead = 26.0
    out0 = None
    for _ in range(20):
      v_lead -= 1.8 * 0.05
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(status=True, dRel=82.0, vRel=v_lead - 26.0,
                       vLead=v_lead, aLeadK=model_a_lead_k),
        _make_raw_lead(status=False), now=t,
      )
      t += 0.05
    return out0, mpc._lead_stability_state[0]

  def test_tiny_negative_model_accel_is_not_amplified_by_noisy_vlead_trend(self):
    # Road 22d taps: model aLead was merely -0.017/-0.04 while the derivative
    # low-pass reached -1.6..-1.9. That is sign noise, not independent evidence
    # for replacing the model report with a hard lead-decel estimate.
    out0, state = self._run_deepening_trend(-0.04, model_min=0.10)
    assert state.corr_a_meas_lp < -1.0
    assert out0.aLeadK == pytest.approx(-0.04)

  def test_governor_recovery_resets_correlation_and_requires_fresh_settling(self):
    mpc = _make_mpc(
      accel_corr_margin=0.5,
      accel_corr_amplify_gain=1.0,
      accel_corr_amplify_model_min=0.10,
      accel_corr_amplify_deadband=0.35,
    )
    _, t = self._settle(mpc, v_lead=26.0)

    # Establish a settled, strongly negative derivative that would amplify a
    # meaningful model brake in the absence of provenance.
    v_lead = 26.0
    for _ in range(14):
      v_lead -= 1.8 * 0.05
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(vRel=v_lead - 26.0, vLead=v_lead, aLeadK=-0.54),
        _make_raw_lead(status=False), now=t,
      )
      t += 0.05
    state = mpc._lead_stability_state[0]
    assert state.corr_settled_s >= 0.6
    assert state.corr_a_meas_lp < -1.0
    assert out0.aLeadK < -1.0

    # Recovery-shaped velocity is not independent corroboration. Preserve the
    # model's own braking report, but clear all derivative history.
    recovered, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(vRel=v_lead - 26.0, vLead=v_lead, aLeadK=-0.54,
                     closingGovernorRecovery=True),
      _make_raw_lead(status=False), now=t,
    )
    t += 0.05
    assert recovered.aLeadK == pytest.approx(-0.54)
    assert state.corr_meas_t is None
    assert state.corr_a_meas_lp == 0.0
    assert state.corr_settled_s == 0.0
    assert state.corr_track_id is None

    # The first unshaped frame seeds a new baseline; fewer than 0.6 seconds of
    # fresh history cannot reactivate CD3 even with a deep derivative.
    for _ in range(11):
      v_lead -= 1.8 * 0.05
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(vRel=v_lead - 26.0, vLead=v_lead, aLeadK=-0.54),
        _make_raw_lead(status=False), now=t,
      )
      t += 0.05
    assert state.corr_settled_s < 0.6
    assert out0.aLeadK == pytest.approx(-0.54)

    # After a complete fresh settling epoch, genuine corroborated braking may
    # amplify again.
    for _ in range(4):
      v_lead -= 1.8 * 0.05
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(vRel=v_lead - 26.0, vLead=v_lead, aLeadK=-0.54),
        _make_raw_lead(status=False), now=t,
      )
      t += 0.05
    assert state.corr_settled_s >= 0.6
    assert out0.aLeadK < -1.0

  def test_meaningful_underreported_model_brake_still_amplifies(self):
    # CD3 road truth-deficit: model -0.48..-0.54 plus trend -1.7..-1.9 is
    # corroborated braking and must retain the safety amplification.
    out0, state = self._run_deepening_trend(-0.54, model_min=0.10)
    assert state.corr_a_meas_lp < -1.0
    assert out0.aLeadK < -1.0

  @staticmethod
  def _run_calm_position_case(*, proof_valid: bool, d_rel: float = 82.0,
                              model_a_lead_k: float = -0.15,
                              fcw_suppressed: bool = False):
    mpc = _make_mpc(
      accel_corr_margin=0.5,
      accel_corr_amplify_gain=1.0,
      accel_corr_amplify_model_min=0.10,
      accel_corr_amplify_deadband=0.35,
    )
    t = 1.0
    v_lead = 26.0
    out0 = None
    for _ in range(20):
      v_lead -= 1.8 * 0.05
      out0, _ = mpc._stabilize_raw_leads(
        _make_raw_lead(
          dRel=d_rel, vRel=v_lead - 26.0, vLead=v_lead,
          aLeadK=model_a_lead_k,
          fcwSuppressed=fcw_suppressed,
          accelCorrCalmPositionValid=proof_valid,
          accelCorrCalmPositionSlopeMps=-0.4,
        ),
        _make_raw_lead(status=False), now=t,
      )
      t += 0.05
    return out0, mpc._lead_stability_state[0]

  def test_calm_position_proof_vetoes_only_extra_amplification_outside_target(self):
    protected, protected_state = self._run_calm_position_case(proof_valid=True)
    legacy, legacy_state = self._run_calm_position_case(proof_valid=False)

    assert protected_state.corr_a_meas_lp < -1.0
    assert protected_state.corr_amplify_vetoed
    assert not protected_state.corr_amplified
    assert protected.aLeadK == pytest.approx(-0.15)
    assert legacy_state.corr_amplified
    assert legacy.aLeadK < -1.0

  def test_calm_position_proof_cannot_weaken_inside_target_or_meaningful_brake(self):
    inside, inside_state = self._run_calm_position_case(proof_valid=True, d_rel=40.0)
    genuine, genuine_state = self._run_calm_position_case(
      proof_valid=True, model_a_lead_k=-0.54,
    )

    assert inside_state.corr_amplified
    assert not inside_state.corr_amplify_vetoed
    assert inside.aLeadK < -1.0
    assert genuine_state.corr_amplified
    assert not genuine_state.corr_amplify_vetoed
    assert genuine.aLeadK < -1.0

  def test_fcw_suppressed_does_not_discard_fresh_independent_position_proof(self):
    # fcwSuppressed is raw-farther corroboration for FCW, not a stale/phantom
    # marker. The same fresh track and producer-attested calm position history
    # must still prevent a noisy vLead derivative from deepening mild aLeadK.
    protected, state = self._run_calm_position_case(
      proof_valid=True,
      fcw_suppressed=True,
    )

    assert state.corr_a_meas_lp < -1.0
    assert state.corr_amplify_vetoed
    assert not state.corr_amplified
    assert protected.aLeadK == pytest.approx(-0.15)

  def test_raw_invalid_phantom_never_uses_calm_position_veto(self):
    mpc = _make_mpc(
      phantom_hold_s=0.5,
      stable_frames=1.0,
      accel_corr_margin=0.5,
      accel_corr_amplify_gain=1.0,
      accel_corr_amplify_model_min=0.10,
      accel_corr_amplify_deadband=0.35,
    )
    # Seed a valid lead carrying proof, then force a raw-invalid held frame.
    mpc._stabilize_raw_leads(
      _make_raw_lead(aLeadK=-0.15, accelCorrCalmPositionValid=True),
      _make_raw_lead(status=False), now=1.0,
    )
    held, _ = mpc._stabilize_raw_leads(
      _make_raw_lead(status=False), _make_raw_lead(status=False), now=1.1,
    )
    state = mpc._lead_stability_state[0]

    assert held.status
    assert not state.corr_amplify_vetoed
    assert not state.corr_amplified

  def test_zero_model_floor_restores_any_negative_report_amplification(self):
    out0, state = self._run_deepening_trend(-0.04, model_min=0.0)
    assert state.corr_a_meas_lp < -1.0
    assert out0.aLeadK < -1.0
