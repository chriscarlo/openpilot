from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, N


def _make_lead(*, status=True, d_rel=44.0, y_rel=0.0, d_path=None, v_lat=0.0, v_rel=0.0,
               v_lead=29.0, a_lead=0.0, model_prob=0.95):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    yRel=y_rel,
    vRel=v_rel,
    aRel=0.0,
    vLead=v_lead,
    dPath=y_rel if d_path is None else d_path,
    vLat=v_lat,
    vLeadK=v_lead,
    aLeadK=a_lead,
    fcw=False,
    aLeadTau=1.5,
    modelProb=model_prob,
    radar=False,
    radarTrackId=-1,
  )


def _configure_vibe_follow(headway=1.3):
  params = Params()
  params.put_bool('VibePersonalityEnabled', True)
  params.put_bool('VibeFollowPersonalityEnabled', True)
  params.put_bool('VibeAccelPersonalityEnabled', False)
  params.put('LongitudinalPersonality', int(log.LongitudinalPersonality.standard))
  for idx in range(4):
    params.put(f'VibeTune.Follow.Standard.Headway{idx}', float(headway))


def _make_hyundai_mpc():
  mpc = LongitudinalMpc(CP=SimpleNamespace(brand='hyundai'))
  mpc.mode = 'acc'
  mpc.set_cur_state(29.0, 0.0)
  return mpc


def _run_update(mpc: LongitudinalMpc, lead0, lead1, *, v_cruise=40.0):
  radarstate = SimpleNamespace(leadOne=lead0, leadTwo=lead1)
  x = np.zeros(N + 1)
  v = np.zeros(N + 1)
  a = np.zeros(N + 1)
  j = np.zeros(N + 1)
  mpc.update(radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard)


class _MonotonicStub:
  def __init__(self, start=100.0, step=0.2):
    self.value = start
    self.step = step

  def __call__(self):
    current = self.value
    self.value += self.step
    return current


@pytest.fixture(autouse=True)
def _planner_test_setup():
  _configure_vibe_follow()


class TestHyundaiAiLeadStability:
  def test_duplicate_pair_keeps_virtual_winner_stable_across_small_slot_jitter(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.1),
    )
    mpc = _make_hyundai_mpc()

    samples = [
      (
        _make_lead(d_rel=44.90, y_rel=0.05, d_path=0.05, v_lat=0.45, v_rel=-0.1, model_prob=0.96),
        _make_lead(d_rel=44.85, y_rel=0.08, d_path=0.08, v_lat=4.20, v_rel=-0.08, model_prob=0.92),
      ),
      (
        _make_lead(d_rel=44.92, y_rel=0.04, d_path=0.04, v_lat=0.55, v_rel=-0.1, model_prob=0.95),
        _make_lead(d_rel=44.80, y_rel=0.06, d_path=0.06, v_lat=4.05, v_rel=-0.09, model_prob=0.93),
      ),
      (
        _make_lead(d_rel=44.88, y_rel=0.05, d_path=0.05, v_lat=0.50, v_rel=-0.1, model_prob=0.95),
        _make_lead(d_rel=44.82, y_rel=0.07, d_path=0.07, v_lat=4.10, v_rel=-0.08, model_prob=0.93),
      ),
    ]

    selected_slots = []
    for lead0, lead1 in samples:
      _run_update(mpc, lead0, lead1)
      assert mpc.lead_role_debug["virtual_duplicate"]["active"] is True
      selected_slots.append(mpc.lead_role_debug["virtual_duplicate"]["selected_raw_slot"])

    assert selected_slots == [0, 0, 0]
    assert mpc.control_leads[0].status is True
    assert mpc.control_leads[1].status is False

  def test_source_hysteresis_holds_lead_through_brief_jitter_then_releases_after_sustained_gap_growth(self, monkeypatch):
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      _MonotonicStub(step=0.2),
    )
    mpc = _make_hyundai_mpc()

    warmup_lead0 = _make_lead(d_rel=41.6, y_rel=0.05, d_path=0.05, v_lat=0.45, v_rel=-0.1, model_prob=0.96)
    warmup_lead1 = _make_lead(d_rel=41.55, y_rel=0.08, d_path=0.08, v_lat=4.10, v_rel=-0.08, model_prob=0.92)
    for _ in range(4):
      _run_update(mpc, warmup_lead0, warmup_lead1)
    assert mpc.source == "lead0"

    jitter_sources = []
    for d_rel in (44.7, 45.0, 44.8, 45.1):
      _run_update(
        mpc,
        _make_lead(d_rel=d_rel, y_rel=0.05, d_path=0.05, v_lat=0.50, v_rel=0.0, model_prob=0.95),
        _make_lead(d_rel=d_rel - 0.04, y_rel=0.08, d_path=0.08, v_lat=4.00, v_rel=0.02, model_prob=0.93),
      )
      jitter_sources.append(mpc.source)

    assert jitter_sources == ["lead0", "lead0", "lead0", "lead0"]

    for _ in range(5):
      _run_update(
        mpc,
        _make_lead(d_rel=48.4, y_rel=0.05, d_path=0.05, v_lat=0.45, v_rel=0.0, model_prob=0.95),
        _make_lead(d_rel=48.35, y_rel=0.07, d_path=0.07, v_lat=4.05, v_rel=0.02, model_prob=0.93),
      )

    assert mpc.source == "cruise"
    assert mpc.acc_source_debug["used_hysteresis"] is True

  def test_cutin_promotion_reaches_virtual_duplicate_lead(self, monkeypatch):
    monotonic = _MonotonicStub(step=0.2)
    monkeypatch.setattr(
      "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.time.monotonic",
      monotonic,
    )
    mpc = _make_hyundai_mpc()

    _run_update(
      mpc,
      _make_lead(d_rel=43.0, y_rel=3.8, d_path=3.8, v_lat=0.0, v_rel=-0.8, model_prob=0.94),
      _make_lead(d_rel=43.1, y_rel=3.9, d_path=3.9, v_lat=0.0, v_rel=-0.8, model_prob=0.92),
    )

    _run_update(
      mpc,
      _make_lead(d_rel=42.6, y_rel=3.2, d_path=3.2, v_lat=-1.2, v_rel=-0.8, model_prob=0.94),
      _make_lead(d_rel=42.7, y_rel=3.3, d_path=3.3, v_lat=-1.0, v_rel=-0.8, model_prob=0.92),
    )

    assert mpc.lead_role_debug["virtual_duplicate"]["active"] is True
    assert mpc._cutin_event_t["lead0"] is not None
