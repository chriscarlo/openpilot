import dataclasses

import numpy as np
import pytest

from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  N,
  LongitudinalMpc,
  _SteadyParityState,
  _StabilizedLead,
  _fcw_candidate_predicts_crash,
  get_headway_follow_distance,
)
from openpilot.selfdrive.controls.radard import ModelLeadTrack


V_EGO = 29.0
TRACK_ID = -1001


def _cfg(**updates) -> LeadResponseTuningConfig:
  return dataclasses.replace(LeadResponseTuningConfig.defaults(), **updates)


def _raw_lead(*, d_rel=64.0, v_rel=-0.8, a_lead=0.0, prob=0.98,
              d_path=0.0, v_lat=0.0) -> dict:
  return {
    "dRel": float(d_rel),
    "yRel": float(d_path),
    "vRel": float(v_rel),
    "vLead": float(V_EGO + v_rel),
    "vLeadK": float(V_EGO + v_rel),
    "aLeadK": float(a_lead),
    "aLeadTau": 0.3,
    "modelProb": float(prob),
    "dPath": float(d_path),
    "vLat": float(v_lat),
  }


def _qualified_track(cfg: LeadResponseTuningConfig | None = None) -> tuple[ModelLeadTrack, dict]:
  cfg = _cfg() if cfg is None else cfg
  raw = _raw_lead()
  track = ModelLeadTrack.from_lead_dict(TRACK_ID, raw, 0.0, 0)
  out = track.get_RadarState(cfg)
  for frame in range(1, 46):
    out = track.update(raw, frame * 0.05, V_EGO, cfg, 0)
  return track, out


def _consumer() -> LongitudinalMpc:
  # The correction method has no solver dependency. A constructor-free MPC
  # keeps these tests focused on its gating/state semantics.
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.dt = 0.05
  mpc._hyundai_ai_lead_stability_enabled = True
  mpc._live_tune_cfg = _cfg()
  mpc._lead_stability_phantom_slots = (False, False)
  mpc._steady_parity_state = [_SteadyParityState(), _SteadyParityState()]
  mpc._steady_parity_fcw_unshaped_leads = []
  mpc.steady_parity_debug = {}
  return mpc


def _planner_lead(*, d_rel=64.0, v_rel=-0.8, vrel_floor=-0.2,
                  candidate_valid=True, a_lead=0.0, d_path=0.0,
                  v_lat=0.0, prob=0.98, track_id=TRACK_ID) -> _StabilizedLead:
  return _StabilizedLead(
    status=True,
    dRel=d_rel,
    yRel=d_path,
    vRel=v_rel,
    vLead=V_EGO + v_rel,
    vLeadK=V_EGO + v_rel,
    aLeadK=a_lead,
    aLeadTau=0.3,
    modelProb=prob,
    dPath=d_path,
    vLat=v_lat,
    radar=False,
    radarTrackId=track_id,
    steadyParityCandidateValid=candidate_valid,
    steadyParityPositionSlopeMps=0.0,
    steadyParityVRelFloorMps=vrel_floor,
  )


def _missing_lead() -> _StabilizedLead:
  return _StabilizedLead(status=False)


class TestSteadyParityProducer:
  def test_dense_position_proof_publishes_evidence_without_reshaping_radar_kinematics(self):
    enabled_cfg = _cfg(steady_parity_trust_deficit_mps=0.2)
    disabled_cfg = _cfg(steady_parity_trust_deficit_mps=99.0)
    enabled_track, enabled = _qualified_track(enabled_cfg)
    _, disabled = _qualified_track(disabled_cfg)

    assert enabled["steadyParityCandidateValid"] is True
    assert enabled["steadyParityPositionSlopeMps"] == pytest.approx(0.0, abs=1e-6)
    assert enabled["steadyParityVRelFloorMps"] == pytest.approx(-0.2)
    assert enabled["steadyParityHeld"] is False
    assert enabled_track.steady_parity_sample_count >= 40
    assert enabled_track.steady_parity_window_span_s >= 1.95
    assert enabled_track.steady_parity_max_sample_gap_s == pytest.approx(0.05)

    # The only difference is telemetry: RadarD's published control kinematics
    # stay bit-for-bit on the pre-existing filter/governor path.
    assert disabled["steadyParityCandidateValid"] is False
    for field in ("dRel", "vRel", "vLead", "vLeadK", "aLeadK"):
      assert enabled[field] == pytest.approx(disabled[field], abs=1e-12)

  @pytest.mark.parametrize("unsafe", [
    {"v_rel": -2.0},
    {"d_path": 1.8},
    {"d_path": 1.3, "v_lat": 0.8},
    {"prob": 0.4},
  ])
  def test_current_threat_clears_candidate_and_requires_fresh_epoch(self, unsafe):
    cfg = _cfg()
    track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True

    cleared = track.update(_raw_lead(**unsafe), 2.30, V_EGO, cfg, 0)
    assert cleared["steadyParityCandidateValid"] is False
    assert len(track.steady_parity_evidence) == 0

    fresh = track.update(_raw_lead(), 2.35, V_EGO, cfg, 0)
    assert fresh["steadyParityCandidateValid"] is False
    assert len(track.steady_parity_evidence) == 1

  def test_current_raw_braking_vetoes_same_frame_but_window_mean_rejects_only_sustained_braking(self):
    cfg = _cfg()
    track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True

    vetoed = track.update(_raw_lead(a_lead=-0.5), 2.30, V_EGO, cfg, 0)
    assert vetoed["steadyParityCandidateValid"] is False
    assert track.steady_parity_reason == "current_raw_braking"
    assert len(track.steady_parity_evidence) >= 40

    recovered = track.update(_raw_lead(), 2.35, V_EGO, cfg, 0)
    assert recovered["steadyParityCandidateValid"] is True

    for frame in range(48, 70):
      sustained = track.update(_raw_lead(a_lead=-0.5), frame * 0.05, V_EGO, cfg, 0)
    assert sustained["steadyParityCandidateValid"] is False
    assert track.steady_parity_reason in ("window_braking", "published_braking")

  def test_slot_change_clears_evidence_epoch(self):
    cfg = _cfg()
    track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True

    changed = track.update(_raw_lead(), 2.30, V_EGO, cfg, 1)
    assert changed["steadyParityCandidateValid"] is False
    assert len(track.steady_parity_evidence) == 1

  def test_capnp_telemetry_round_trip_has_safe_defaults(self):
    _, lead = _qualified_track()
    msg = log.RadarState.new_message()
    msg.leadOne = lead

    assert msg.leadOne.steadyParityCandidateValid is True
    assert msg.leadOne.steadyParityPositionSlopeMps == pytest.approx(0.0)
    assert msg.leadOne.steadyParityVRelFloorMps == pytest.approx(-0.2)
    assert msg.leadTwo.steadyParityCandidateValid is False
    assert msg.leadTwo.steadyParityVRelFloorMps == pytest.approx(0.0)


class TestSteadyParityConsumer:
  def test_arms_above_exact_target_and_slews_only_less_urgent_direction(self):
    mpc = _consumer()
    target = get_headway_follow_distance(V_EGO, 1.5)
    assert target == pytest.approx(49.5)

    first, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=64.0), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.0,
    )
    assert first.vRel == pytest.approx(-0.76)
    assert first.vLead == pytest.approx(V_EGO - 0.76)
    assert mpc.steady_parity_debug["slot0"]["active"] is True

    second, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=64.0), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.05,
    )
    assert second.vRel == pytest.approx(-0.72)

    urgent, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=64.0, v_rel=-1.2, vrel_floor=-1.0), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.10,
    )
    assert urgent.vRel == pytest.approx(-1.0)

  def test_target_threat_phantom_and_rollback_restore_unshaped_same_frame(self):
    mpc = _consumer()
    mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.0,
    )

    target_gap = get_headway_follow_distance(V_EGO, 1.5)
    at_target, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=target_gap, v_rel=-0.8), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.05,
    )
    assert at_target.vRel == pytest.approx(-0.8)
    assert mpc.steady_parity_debug["slot0"]["reason"] == "target_reached"

    # Re-arm, then prove stale-valid telemetry cannot mask fresh threat data.
    mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.10,
    )
    threat, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=64.0, v_rel=-3.0, vrel_floor=-0.2, a_lead=-1.0), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.15,
    )
    assert threat.vRel == pytest.approx(-3.0)
    assert mpc.steady_parity_debug["slot0"]["reason"] == "current_threat_restore"

    mpc._lead_stability_phantom_slots = (True, False)
    phantom, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.20,
    )
    assert phantom.vRel == pytest.approx(-0.8)

    mpc._lead_stability_phantom_slots = (False, False)
    mpc._live_tune_cfg = _cfg(steady_parity_vrel_slew_mps2=0.0)
    rollback, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.25,
    )
    assert rollback.vRel == pytest.approx(-0.8)
    assert mpc.steady_parity_debug["slot0"]["reason"] == "rollback_disabled"

  def test_does_not_arm_without_both_surplus_and_thw_gates_or_manufacture_positive_vrel(self):
    mpc = _consumer()
    target_gap = get_headway_follow_distance(V_EGO, 1.5)

    no_surplus, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=target_gap + 2.9), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.0,
    )
    assert no_surplus.vRel == pytest.approx(-0.8)
    assert mpc.steady_parity_debug["slot0"]["active"] is False

    positive, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=64.0, v_rel=0.2, vrel_floor=0.0), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.05,
    )
    assert positive.vRel == pytest.approx(0.2)

    unknown, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(d_rel=64.0, track_id=-1), _missing_lead(),
      v_ego=V_EGO, t_follow=1.5, now=1.10,
    )
    assert unknown.vRel == pytest.approx(-0.8)
    assert mpc.steady_parity_debug["slot0"]["candidate_valid"] is False

  def test_unshaped_active_lead_remains_an_independent_fcw_candidate(self):
    mpc = _consumer()
    corrected, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.0,
    )
    assert corrected.vRel > -0.8
    assert len(mpc._steady_parity_fcw_unshaped_leads) == 1
    assert mpc._steady_parity_fcw_unshaped_leads[0].vRel == pytest.approx(-0.8)

    x_sol = np.zeros((N + 1, 3))
    safe = np.column_stack((np.full(N + 1, 10.0), np.zeros(N + 1)))
    crash = np.column_stack((np.full(N + 1, 0.1), np.zeros(N + 1)))
    assert not _fcw_candidate_predicts_crash(safe, x_sol, 0.99, False)
    assert _fcw_candidate_predicts_crash(crash, x_sol, 0.99, False)
    assert not _fcw_candidate_predicts_crash(crash, x_sol, 0.99, True)
