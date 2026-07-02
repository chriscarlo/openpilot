"""FCW corroboration veto (stop-slam M3 defense in depth).

A phantom-collapsed model-lead track (filtered dRel far below what the model
keeps measuring) must not accrue mpc.crash_cnt toward longitudinalPlan.fcw and
the Hyundai carcontroller emergency_control() override. radard's ModelLeadTrack
votes per frame on raw-vs-filtered dRel agreement and publishes
radarState.leadX.fcwSuppressed; long_mpc treats it as a veto on crash_cnt
accrual only. Genuine threats must be unaffected: on a real collision course
the closing-side filter lags on the FAR side of raw, so agreement always holds,
and isolated outward measurement outliers are bridged by the majority vote.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, N
from openpilot.selfdrive.controls.radard import ModelLeadTrack

DT = 0.05


def _cfg(**overrides) -> LeadResponseTuningConfig:
  cfg = LeadResponseTuningConfig()
  return replace(cfg, **overrides) if overrides else cfg


def _lead_dict(*, d_rel, v_rel=0.0, v_ego=10.0, prob=0.98, y_rel=0.0, v_lat=0.0, a_lead=0.0):
  return {
    "dRel": float(d_rel),
    "yRel": float(y_rel),
    "vRel": float(v_rel),
    "vLead": float(v_ego + v_rel),
    "vLeadK": float(v_ego + v_rel),
    "aLeadK": float(a_lead),
    "aLeadTau": 0.3,
    "modelProb": float(prob),
    "dPath": float(y_rel),
    "vLat": float(v_lat),
  }


class TestModelLeadFcwCorroborationVote:
  def test_new_track_is_not_suppressed(self):
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d_rel=12.0), 0.0, 0)
    state = track.update(_lead_dict(d_rel=12.0), DT, 10.0, _cfg(), 0)
    assert state["fcwSuppressed"] is False

  def test_phantom_divergence_is_suppressed_within_two_frames(self):
    # Collapsed internal state (filtered ~2 m) while the model keeps measuring
    # ~13 m: the one-way open-slew ratchet cannot heal it, and it must not be
    # FCW-eligible. v_ego above the M2 open-recovery speed gate isolates the
    # M3 veto from the M2 healing path.
    cfg = _cfg()
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d_rel=2.0), 0.0, 0)
    states = [track.update(_lead_dict(d_rel=13.0), (i + 1) * DT, 12.0, cfg, 0) for i in range(4)]
    assert float(track.dRel) < 4.0, "test premise: filter must still be collapsed"
    assert states[-1]["fcwSuppressed"] is True
    assert states[1]["fcwSuppressed"] is True, "suppression must engage by the second disagreeing frame"

  def test_genuine_hard_closing_is_never_suppressed(self):
    # Real collision course: raw dRel falls 8 m/s; the closing-side EMA lags on
    # the FAR side of raw (raw <= filtered), which is agreement by sign.
    cfg = _cfg()
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d_rel=40.0, v_rel=-8.0), 0.0, 0)
    suppressed = []
    for i in range(1, 80):
      d_rel = max(1.0, 40.0 - 8.0 * i * DT)
      state = track.update(_lead_dict(d_rel=d_rel, v_rel=-8.0), i * DT, 15.0, cfg, 0)
      suppressed.append(state["fcwSuppressed"])
    assert not any(suppressed)

  def test_opening_then_hard_brake_reversal_never_suppressed(self):
    # Judge-mandated pin of the sign argument the entire veto's safety rests on
    # ("genuine threats always agree"): while a lead opens fast the vRel-aware
    # prediction must track raw closely enough that raw - filtered stays under
    # the tolerance (worst case for the opening side: filtered dRel lags BELOW
    # raw while vRel converges), and at the instant the lead hard-brakes into a
    # genuine closing threat the vote must already be agreeing — fcwSuppressed
    # False on EVERY frame, including the closing-onset frame. A future filter
    # or prediction change (e.g. removing vRel from predict_drel or reworking
    # the opening slew) must not silently break this.
    cfg = _cfg()
    v_ego = 12.0  # above the M2 open-recovery speed gate: recovery cannot mask an opening lag
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d_rel=10.0, v_ego=v_ego), 0.0, 0)
    suppressed = []
    d_rel = 10.0
    frame = 0
    # Phase 1: lead opens at +6 m/s for 2 s.
    for _ in range(40):
      frame += 1
      d_rel += 6.0 * DT
      state = track.update(_lead_dict(d_rel=d_rel, v_rel=6.0, v_ego=v_ego), frame * DT, v_ego, cfg, 0)
      suppressed.append(state["fcwSuppressed"])
    # Phase 2: immediate reversal — the lead hard-brakes and the gap closes at 8 m/s.
    closing_onset_idx = len(suppressed)
    for _ in range(60):
      frame += 1
      d_rel = max(1.0, d_rel - 8.0 * DT)
      state = track.update(_lead_dict(d_rel=d_rel, v_rel=-8.0, v_ego=v_ego, a_lead=-8.0),
                           frame * DT, v_ego, cfg, 0)
      suppressed.append(state["fcwSuppressed"])
    assert suppressed[closing_onset_idx] is False, "closing-onset frame must be FCW-eligible"
    assert not any(suppressed), (
      f"suppressed frames at indices {[i for i, s in enumerate(suppressed) if s]} "
      f"(closing onset at {closing_onset_idx})"
    )

  def test_isolated_outward_outlier_is_bridged(self):
    # A single heavy-tail outlier on the FAR side (raw +9 m for one frame on a
    # steadily followed lead) must not flicker suppression on: 2-of-3 majority.
    cfg = _cfg()
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d_rel=20.0), 0.0, 0)
    suppressed = []
    for i in range(1, 40):
      d_rel = 29.0 if i == 20 else 20.0
      state = track.update(_lead_dict(d_rel=d_rel), i * DT, 10.0, cfg, 0)
      suppressed.append(state["fcwSuppressed"])
    assert not any(suppressed)

  def test_min_agree_zero_disables_suppression(self):
    cfg = _cfg(model_lead_fcw_corrob_min_agree=0.0)
    track = ModelLeadTrack.from_lead_dict(-1001, _lead_dict(d_rel=2.0), 0.0, 0)
    states = [track.update(_lead_dict(d_rel=13.0), (i + 1) * DT, 12.0, cfg, 0) for i in range(6)]
    assert not any(s["fcwSuppressed"] for s in states)


def _configure_vibe_follow(headway=1.3):
  params = Params()
  params.put_bool('VibePersonalityEnabled', True)
  params.put_bool('VibeFollowPersonalityEnabled', True)
  params.put_bool('VibeAccelPersonalityEnabled', False)
  params.put('LongitudinalPersonality', int(log.LongitudinalPersonality.standard))
  for idx in range(4):
    params.put(f'VibeTune.Follow.Standard.Headway{idx}', float(headway))


def _make_lead(*, status=True, d_rel=8.0, v_rel=-6.0, v_lead=4.0, a_lead=-2.0,
               model_prob=0.99, fcw_suppressed=False, y_rel=0.05, v_lat=0.10):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    yRel=y_rel,
    vRel=v_rel,
    aRel=0.0,
    vLead=v_lead,
    dPath=y_rel,
    vLat=v_lat,
    vLeadK=v_lead,
    aLeadK=a_lead,
    fcw=False,
    fcwSuppressed=fcw_suppressed,
    aLeadTau=1.5,
    modelProb=model_prob,
    radar=False,
    radarTrackId=-1,
  )


class _MonotonicStub:
  def __init__(self, start=100.0, step=0.2):
    self.value = start
    self.step = step

  def __call__(self):
    current = self.value
    self.value += self.step
    return current


def _run_update(mpc: LongitudinalMpc, lead0, lead1, *, v_cruise=35.0):
  radarstate = SimpleNamespace(leadOne=lead0, leadTwo=lead1)
  zeros = np.zeros(N + 1)
  mpc.update(radarstate, v_cruise, zeros, zeros, zeros, zeros,
             personality=log.LongitudinalPersonality.standard)


@pytest.fixture(autouse=True)
def _planner_test_setup():
  _configure_vibe_follow()


class TestCrashCntFcwSuppressedVeto:
  """Mirror of test_hyundai_ai_lead_stability.py::test_fcw_counter_still_accumulates_for_active_closing_lead
  with the producer veto asserted in both directions."""

  def _mpc(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = LongitudinalMpc(CP=SimpleNamespace(brand='hyundai'))
    mpc.mode = 'acc'
    mpc.set_cur_state(12.0, 0.3)
    return mpc

  def test_crash_cnt_accumulates_without_suppression(self, monkeypatch):
    mpc = self._mpc(monkeypatch)
    for _ in range(10):
      _run_update(mpc,
                  _make_lead(fcw_suppressed=False),
                  _make_lead(d_rel=8.1, v_rel=-5.98, v_lead=4.02, y_rel=0.08, v_lat=0.12,
                             fcw_suppressed=False))
    assert mpc.source == "lead0"
    assert mpc.crash_cnt > 2

  def test_crash_cnt_vetoed_by_fcw_suppressed_lead(self, monkeypatch):
    mpc = self._mpc(monkeypatch)
    for _ in range(10):
      _run_update(mpc,
                  _make_lead(fcw_suppressed=True),
                  _make_lead(d_rel=8.1, v_rel=-5.98, v_lead=4.02, y_rel=0.08, v_lat=0.12,
                             fcw_suppressed=True))
    assert mpc.source == "lead0"
    assert mpc.crash_cnt == 0
