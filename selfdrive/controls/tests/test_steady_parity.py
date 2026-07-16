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
from openpilot.selfdrive.controls.radard import ModelLeadTrack, ModelLeadTracker


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
                  v_lat=0.0, prob=0.98, track_id=TRACK_ID,
                  current_threat=False) -> _StabilizedLead:
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
    steadyParityCurrentThreat=current_threat,
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

  @pytest.mark.parametrize(("unsafe", "expected_reason", "expected_threat"), [
    ({"v_rel": -6.0}, "short_raw_ttc", True),
    ({"d_rel": 15.0}, "near_threat", True),
    ({"d_path": 1.8}, "lateral_ambiguity", False),
    ({"d_path": 1.3, "v_lat": 0.8}, "lateral_ambiguity", False),
    ({"prob": 0.4}, "low_probability", False),
  ])
  def test_current_threat_clears_candidate_and_requires_fresh_epoch(self, unsafe, expected_reason, expected_threat):
    cfg = _cfg()
    track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True

    cleared = track.update(_raw_lead(**unsafe), 2.30, V_EGO, cfg, 0)
    assert cleared["steadyParityCandidateValid"] is False
    assert cleared["steadyParityCurrentThreat"] is expected_threat
    assert track.steady_parity_reason == expected_reason
    assert len(track.steady_parity_evidence) == 0

    fresh = track.update(_raw_lead(), 2.35, V_EGO, cfg, 0)
    assert fresh["steadyParityCandidateValid"] is False
    assert fresh["steadyParityCurrentThreat"] is False
    assert len(track.steady_parity_evidence) == 1

  def test_velocity_only_spike_clears_proof_and_requires_fresh_epoch(self):
    enabled_cfg = _cfg(steady_parity_trust_deficit_mps=0.2)
    disabled_cfg = _cfg(steady_parity_trust_deficit_mps=99.0)
    track, qualified = _qualified_track(enabled_cfg)
    disabled_track, _ = _qualified_track(disabled_cfg)
    assert qualified["steadyParityCandidateValid"] is True

    sequence = [
      (2.30, _raw_lead(v_rel=-2.0)),
      (2.35, _raw_lead()),
      (2.40, _raw_lead()),
    ]
    enabled_outputs = []
    for now, raw in sequence:
      enabled = track.update(raw, now, V_EGO, enabled_cfg, 0)
      disabled = disabled_track.update(raw, now, V_EGO, disabled_cfg, 0)
      enabled_outputs.append((enabled, track.steady_parity_reason))
      # Steady-parity remains producer telemetry only: clearing the proof does
      # not alter any RadarD control kinematic, including on the spike/reset.
      for field in ("dRel", "vRel", "vLead", "vLeadK", "aLeadK"):
        assert enabled[field] == pytest.approx(disabled[field], abs=1e-12)

    (spike, spike_reason), (calm_one, calm_one_reason), (calm_two, calm_two_reason) = enabled_outputs
    assert spike["steadyParityCandidateValid"] is False
    assert spike["steadyParityVRelFloorMps"] == pytest.approx(0.0)
    assert spike["steadyParityCurrentThreat"] is False
    assert spike_reason == "fast_close"
    assert calm_one["steadyParityCandidateValid"] is False
    assert calm_one_reason == "sparse_window"
    assert calm_two["steadyParityCandidateValid"] is False
    assert calm_two_reason == "sparse_window"
    assert len(track.steady_parity_evidence) == 2

  def test_sustained_velocity_only_close_stays_fail_closed_without_threat_attestation(self):
    cfg = _cfg()
    track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True

    for offset in range(7):
      output = track.update(_raw_lead(v_rel=-2.0), 2.30 + offset * 0.05, V_EGO, cfg, 0)
      assert output["steadyParityCandidateValid"] is False
      assert track.steady_parity_reason == "fast_close"
      assert output["steadyParityCurrentThreat"] is False
      assert len(track.steady_parity_evidence) == 0

  def test_fast_close_attestation_requires_braking_or_position_confirmation(self):
    cfg = _cfg()
    braking_track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True
    braking = braking_track.update(_raw_lead(v_rel=-2.0, a_lead=-0.5), 2.30, V_EGO, cfg, 0)
    assert braking["steadyParityCandidateValid"] is False
    assert braking_track.steady_parity_reason == "current_raw_braking"
    assert braking["steadyParityCurrentThreat"] is True
    assert len(braking_track.steady_parity_evidence) == 0

    position_track, qualified = _qualified_track(cfg)
    assert qualified["steadyParityCandidateValid"] is True
    # A 2 m/s newest range closure exceeds the independent position-step gate,
    # so this fast-close revocation is a genuine current threat.
    position = position_track.update(_raw_lead(d_rel=63.9, v_rel=-2.0), 2.30, V_EGO, cfg, 0)
    assert position["steadyParityCandidateValid"] is False
    assert position_track.steady_parity_reason == "fast_close"
    assert position["steadyParityCurrentThreat"] is True
    assert len(position_track.steady_parity_evidence) == 0

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
    assert changed["steadyParityCurrentThreat"] is False
    assert len(track.steady_parity_evidence) == 1

  def test_capnp_telemetry_round_trip_has_safe_defaults(self):
    _, lead = _qualified_track()
    msg = log.RadarState.new_message()
    msg.leadOne = lead

    assert msg.leadOne.steadyParityCandidateValid is True
    assert msg.leadOne.steadyParityPositionSlopeMps == pytest.approx(0.0)
    assert msg.leadOne.steadyParityVRelFloorMps == pytest.approx(-0.2)
    assert msg.leadOne.steadyParityCurrentThreat is False
    assert msg.leadTwo.steadyParityCandidateValid is False
    assert msg.leadTwo.steadyParityVRelFloorMps == pytest.approx(0.0)
    assert msg.leadTwo.steadyParityCurrentThreat is False

  @pytest.mark.parametrize(("raw", "expected"), [
    ({"a_lead": -0.20}, False),
    ({"a_lead": -0.2001}, True),
    ({"d_rel": 0.55 * V_EGO}, True),
    ({"d_rel": 0.55 * V_EGO + 1e-3}, False),
    ({"d_rel": 55.0, "v_rel": -(55.0 / 12.0)}, True),
    ({"d_rel": 55.0, "v_rel": -(55.0 / 12.001)}, False),
  ])
  def test_current_threat_attestation_threshold_edges(self, raw, expected):
    cfg = _cfg()
    track, _ = _qualified_track(cfg)
    out = track.update(_raw_lead(**raw), 2.30, V_EGO, cfg, 0)
    assert out["steadyParityCurrentThreat"] is expected

  def test_exact_fast_close_boundary_clears_without_attesting_velocity_only_threat(self):
    cfg = _cfg()
    below_track, _ = _qualified_track(cfg)
    below = below_track.update(_raw_lead(v_rel=-1.4999), 2.30, V_EGO, cfg, 0)
    assert below["steadyParityCandidateValid"] is True

    edge_track, _ = _qualified_track(cfg)
    edge = edge_track.update(_raw_lead(v_rel=-1.5), 2.30, V_EGO, cfg, 0)
    assert edge["steadyParityCandidateValid"] is False
    assert edge_track.steady_parity_reason == "fast_close"
    assert edge["steadyParityCurrentThreat"] is False
    assert len(edge_track.steady_parity_evidence) == 0

  def test_missed_frame_clears_one_frame_threat_attestation(self):
    class _NoParams:
      def get(self, _key):
        return None

    cfg = _cfg()
    track, _ = _qualified_track(cfg)
    threat = track.update(_raw_lead(a_lead=-0.5), 2.30, V_EGO, cfg, 0)
    assert threat["steadyParityCurrentThreat"] is True

    tracker = ModelLeadTracker(params=_NoParams())
    tracker._tracks[TRACK_ID] = track
    tracker.begin_frame(2.35)
    tracker.end_frame()
    assert track.steady_parity_current_threat is False


class TestAccelCorrCalmPositionProducer:
  def test_dense_position_proof_survives_false_onset_raw_accel_but_not_hard_braking(self):
    cfg = _cfg()
    track, qualified = _qualified_track(cfg)
    assert qualified["accelCorrCalmPositionValid"] is True
    assert qualified["accelCorrRawHardBraking"] is False
    assert qualified["accelCorrCalmPositionSlopeMps"] == pytest.approx(0.0, abs=1e-6)
    assert track.accel_corr_calm_position_sample_count >= 40
    assert track.accel_corr_calm_position_window_span_s >= 1.95

    # Exact route separation: the false-onset raw report reached -0.3506 while
    # its dense position slope remained calm. It must remain eligible.
    mild = track.update(_raw_lead(a_lead=-0.35), 2.30, V_EGO, cfg, 0)
    assert mild["accelCorrCalmPositionValid"] is True
    assert mild["accelCorrRawHardBraking"] is False
    assert track.accel_corr_calm_position_reason == "position_proven"

    # Known genuine CD3 evidence is -0.48..-0.54. The fixed -0.40 same-frame
    # raw gate clears all proof state before the published accel EMA can lag it.
    hard = track.update(_raw_lead(a_lead=-0.41), 2.35, V_EGO, cfg, 0)
    assert hard["accelCorrCalmPositionValid"] is False
    assert hard["accelCorrRawHardBraking"] is True
    assert track.accel_corr_calm_position_reason == "raw_hard_braking"
    assert len(track.accel_corr_position_evidence) == 0

  def test_dense_closing_slope_and_lateral_ambiguity_fail_closed(self):
    cfg = _cfg()
    closing_raw = _raw_lead(d_rel=64.0, v_rel=-2.0)
    closing_track = ModelLeadTrack.from_lead_dict(TRACK_ID, closing_raw, 0.0, 0)
    closing = closing_track.get_RadarState(cfg)
    for frame in range(1, 46):
      closing = closing_track.update(
        _raw_lead(d_rel=64.0 - 2.0 * frame * 0.05, v_rel=-2.0),
        frame * 0.05, V_EGO, cfg, 0,
      )
    assert closing["accelCorrCalmPositionValid"] is False
    assert closing_track.accel_corr_calm_position_reason in ("position_slope_veto", "sparse_window")

    lateral_track, qualified = _qualified_track(cfg)
    assert qualified["accelCorrCalmPositionValid"] is True
    lateral = lateral_track.update(_raw_lead(d_path=1.4, v_lat=0.8), 2.30, V_EGO, cfg, 0)
    assert lateral["accelCorrCalmPositionValid"] is False
    assert lateral_track.accel_corr_calm_position_reason == "lateral_ambiguity"

  def test_capnp_round_trip_defaults_false_and_preserves_proof(self):
    _, lead = _qualified_track()
    msg = log.RadarState.new_message()
    msg.leadOne = lead
    hard = dict(lead)
    hard["accelCorrRawHardBraking"] = True
    msg.leadTwo = hard
    reader = msg.as_reader()

    assert reader.leadOne.accelCorrCalmPositionValid
    assert reader.leadOne.accelCorrCalmPositionSlopeMps == pytest.approx(0.0)
    assert not reader.leadOne.accelCorrRawHardBraking
    assert reader.leadTwo.accelCorrCalmPositionValid
    assert reader.leadTwo.accelCorrCalmPositionSlopeMps == pytest.approx(0.0)
    assert reader.leadTwo.accelCorrRawHardBraking


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

  def test_same_track_attested_threat_marks_one_frame_planner_bypass(self):
    mpc = _consumer()
    mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.0,
    )

    threat, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(candidate_valid=False, a_lead=-0.08, current_threat=True),
      _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.05,
    )
    debug = mpc.steady_parity_debug["slot0"]
    assert threat.steadyParityThreatRestore is True
    assert debug["was_active"] is True
    assert debug["same_track"] is True
    assert debug["producer_current_threat"] is True
    assert debug["current_kinematic_threat"] is True
    assert debug["current_threat_reason"] == "lead_braking"

    # The classification is one-shot: after state reset, persistent producer
    # threat telemetry no longer carries a stale correction anchor.
    following, _ = mpc._apply_steady_parity_to_leads(
      _planner_lead(candidate_valid=False, a_lead=-0.15, current_threat=True),
      _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.10,
    )
    assert following.steadyParityThreatRestore is False
    assert mpc.steady_parity_debug["slot0"]["current_kinematic_threat"] is False

  @pytest.mark.parametrize("lead", [
    _planner_lead(candidate_valid=False, current_threat=False, v_rel=-2.0),
    _planner_lead(candidate_valid=False, current_threat=True, d_path=1.8),
    _planner_lead(candidate_valid=False, current_threat=True, prob=0.4),
    _planner_lead(candidate_valid=False, current_threat=True, track_id=-1002),
  ])
  def test_velocity_only_lateral_probability_and_identity_exits_do_not_bypass(self, lead):
    mpc = _consumer()
    mpc._apply_steady_parity_to_leads(
      _planner_lead(), _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.0,
    )
    restored, _ = mpc._apply_steady_parity_to_leads(
      lead, _missing_lead(), v_ego=V_EGO, t_follow=1.5, now=1.05,
    )
    assert restored.steadyParityThreatRestore is False
    assert mpc.steady_parity_debug["slot0"]["current_kinematic_threat"] is False

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
    assert not _fcw_candidate_predicts_crash(crash, x_sol, 0.90, False)
    assert _fcw_candidate_predicts_crash(crash, x_sol, 0.900001, False)
    assert not _fcw_candidate_predicts_crash(crash, x_sol, 0.99, True)
