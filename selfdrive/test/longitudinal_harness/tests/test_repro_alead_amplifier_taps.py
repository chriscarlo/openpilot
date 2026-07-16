"""Regressions for two 2026-07-12 transient taps caused by aLead amplification.

Both road events were separate from the fixed radard position-collapse governor.
The model reported only a near-zero negative aLeadK (-0.017/-0.04), while a
noisy same-track vLead derivative reached roughly -1.6..-1.9 m/s^2. The MPC
treated that sign noise as independent lead-braking evidence and amplified it:

* a far, cruise-owned lead defeated the cap-collapse comfort limiter through
  its ``lead_decel`` bypass and produced a one-frame brake tap;
* a moderate approach raised the ownership threshold from ~4.5 to ~7.5 s and
  acquired the lead about 0.6 s before raw closure independently justified it.

These full EV6 device-controller + real-radard twins inject the vLead error as a
measurement bias while the true lead speed stays constant. The live-tunable
model-decel floor is the only fix/rollback difference.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 7.0
TREND_START_S = 3.0
TREND_END_S = 4.0
MEASURED_VLEAD_SLOPE_MPS2 = -1.8
MODEL_DECEL_FLOOR_FIX_MPS2 = 0.10
MODEL_DECEL_FLOOR_ROLLBACK_MPS2 = 0.0
ROAD_HANDOFF_WINDOW_S = 1.0
ROAD_HANDOFF_MAX_DELTA_MPS2 = 0.12


@dataclass(frozen=True)
class _Case:
  name: str
  ego_v0_mps: float
  cruise_mps: float
  lead_v_mps: float
  gap_m: float
  model_a_lead_k_mps2: float


FAR_CAP_TAP = _Case(
  name="far_cap_tap",
  ego_v0_mps=12.0,
  cruise_mps=18.0,
  lead_v_mps=14.0,
  gap_m=82.0,
  model_a_lead_k_mps2=-0.03,
)
MODERATE_APPROACH_TAP = _Case(
  name="moderate_approach_tap",
  ego_v0_mps=24.0,
  cruise_mps=26.2,
  lead_v_mps=24.0,
  gap_m=50.0,
  model_a_lead_k_mps2=-0.04,
)


def _vlead_bias(t_s: float) -> float:
  if t_s < TREND_START_S:
    return 0.0
  return MEASURED_VLEAD_SLOPE_MPS2 * min(t_s - TREND_START_S, TREND_END_S - TREND_START_S)


def _build_steps(case: _Case) -> list[StepInput]:
  steps: list[StepInput] = []
  for i in range(int(round(DURATION_S / DT_MDL))):
    t_s = i * DT_MDL
    trend_active = TREND_START_S <= t_s < TREND_END_S
    lead = LeadDirective(
      status=True,
      # Ground truth is constant. Only the raw model vLead measurement gets the
      # downward ramp, reproducing the road derivative without a real hard brake.
      v_lead_mps=case.lead_v_mps,
      v_lead_bias_mps=_vlead_bias(t_s),
      a_lead_k_mps2=case.model_a_lead_k_mps2 if trend_active else 0.0,
      model_prob_target=0.99,
      d_rel_override_m=case.gap_m if i == 0 else None,
      acquisition_reset=i == 0,
    )
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=case.cruise_mps,
      lead_one=lead,
      event="noisy_vlead_trend" if abs(t_s - TREND_START_S) < DT_MDL * 0.5 else None,
      note="constant-speed true lead; model vLead trend and near-zero aLead sign noise",
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _vehicle_config(model_decel_floor_mps2: float):
  return resolve_ev6_vehicle_config(param_overrides={
    "Longitudinal.LiveTune.LeadAccelCorrAmplifyModelDecelMinMps2": f"{model_decel_floor_mps2:g}",
    "Longitudinal.LiveTune.HandoffLimitWindowS": f"{ROAD_HANDOFF_WINDOW_S:g}",
    "Longitudinal.LiveTune.HandoffLimitMaxDeltaMps2": f"{ROAD_HANDOFF_MAX_DELTA_MPS2:g}",
    "Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2": "10.0",
    "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.4",
    # Isolate the downstream aLead-amplifier fix/rollback. The opening-governor
    # integration oracle separately verifies that the new default can remove
    # this false-closing input before it reaches CD3.
    "Longitudinal.LiveTune.OpeningGovernorHoldS": "0",
  })


@functools.lru_cache(maxsize=8)
def _run(case: _Case, model_decel_floor_mps2: float,
         disable_calm_position_proof: bool = False) -> SimulationResult:
  # The legacy model-floor A/B predates the independent dense-position proof.
  # Keep that oracle isolated by explicitly disabling only the new telemetry;
  # separate assertions below exercise the integrated default path.
  from openpilot.selfdrive.controls.radard import ModelLeadTrack

  original_get_radar_state = ModelLeadTrack.get_RadarState

  def get_radar_state_without_calm_position_proof(self, *args, **kwargs):
    lead = original_get_radar_state(self, *args, **kwargs)
    lead["accelCorrCalmPositionValid"] = False
    lead["accelCorrCalmPositionSlopeMps"] = 0.0
    return lead

  if disable_calm_position_proof:
    ModelLeadTrack.get_RadarState = get_radar_state_without_calm_position_proof
  try:
    return run_harness(
      vehicle_config=_vehicle_config(model_decel_floor_mps2),
      scenario_name=(
        f"alead_amplifier_{case.name}_{model_decel_floor_mps2:g}_"
        f"proof_{'off' if disable_calm_position_proof else 'on'}"
      ),
      steps=_build_steps(case),
      initial_speed_mps=case.ego_v0_mps,
      noise_profile="off",
      seed=42,
      perception_filter="auto",
    )
  finally:
    if disable_calm_position_proof:
      ModelLeadTrack.get_RadarState = original_get_radar_state


def _planner_rows(result: SimulationResult) -> list[dict]:
  return result.trace[::5]


def _window(result: SimulationResult, start_s=3.2, end_s=5.0) -> list[dict]:
  return [row for row in _planner_rows(result) if start_s <= row["t_s"] <= end_s]


def _worst_drop(result: SimulationResult, start_s=3.2, end_s=5.0) -> tuple[float, dict, dict]:
  rows = _planner_rows(result)
  candidates = [(cur["planner_accel_mps2"] - prev["planner_accel_mps2"], prev, cur)
                for prev, cur in zip(rows[:-1], rows[1:], strict=False)
                if start_s <= cur["t_s"] <= end_s]
  return min(candidates, key=lambda item: item[0])


def test_amplifier_tap_scenarios_use_real_radard_without_true_lead_braking() -> None:
  for case in (FAR_CAP_TAP, MODERATE_APPROACH_TAP):
    result = _run(case, MODEL_DECEL_FLOOR_FIX_MPS2)
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"
    assert result.vehicle["noiseProfile"] == "off"

    active_speeds = {row["active_lead_speed_mps"] for row in result.trace
                     if row["active_lead_speed_mps"] is not None}
    assert active_speeds == {case.lead_v_mps}
    injected = [row for row in _planner_rows(result) if 3.2 <= row["t_s"] < TREND_END_S]
    assert injected
    assert all(abs(row["lead_one_raw_a_lead_k_mps2"] - case.model_a_lead_k_mps2) < 1e-6
               for row in injected)


def test_far_cap_collapse_tap_is_bounded_and_rollback_reproduces_it() -> None:
  fix = _run(FAR_CAP_TAP, MODEL_DECEL_FLOOR_FIX_MPS2, True)
  rollback = _run(FAR_CAP_TAP, MODEL_DECEL_FLOOR_ROLLBACK_MPS2, True)
  fix_drop, _, fix_row = _worst_drop(fix, end_s=4.2)
  roll_drop, _, roll_row = _worst_drop(rollback, end_s=4.2)

  physics = (
    f"far cruise-cap tap at t={fix_row['t_s']:.2f}s, published dRel/vRel/aLead "
    f"{fix_row['lead_one_published_d_rel_m']:.1f} m / {fix_row['lead_one_published_v_rel_mps']:+.2f} m/s / "
    f"{fix_row['lead_one_published_a_lead_k_mps2']:+.3f} m/s^2: fix delta {fix_drop:+.3f}, "
    f"aTarget {fix_row['planner_accel_mps2']:+.3f}; rollback delta {roll_drop:+.3f}, "
    f"aTarget {roll_row['planner_accel_mps2']:+.3f}"
  )
  assert all(row["planner_source"] == "cruise" for row in _window(fix, end_s=4.2)), physics
  assert fix_row["mpc_acc_source_debug"]["approach_reacquire_lead_decel_mps2"] < 0.05, physics
  assert fix_row["planner_handoff_limit_debug"]["down_bypassed"] is False, physics
  assert fix_row["planner_handoff_limit_debug"]["clipped"] is True, physics
  assert fix_drop >= -ROAD_HANDOFF_MAX_DELTA_MPS2 - 1e-9, physics
  assert min(row["planner_accel_mps2"] for row in _window(fix, end_s=4.2)) >= -0.05, physics

  assert roll_row["mpc_acc_source_debug"]["approach_reacquire_lead_decel_mps2"] > 1.5, physics
  assert roll_row["planner_handoff_limit_debug"]["down_bypassed"] is True, physics
  assert roll_row["planner_handoff_limit_debug"]["bypass_reason"] == "lead_decel", physics
  assert roll_drop < -0.6, physics
  assert roll_row["planner_accel_mps2"] < -0.3, physics


def test_moderate_approach_waits_for_raw_ttc_instead_of_amplified_alead() -> None:
  fix = _run(MODERATE_APPROACH_TAP, MODEL_DECEL_FLOOR_FIX_MPS2, True)
  rollback = _run(MODERATE_APPROACH_TAP, MODEL_DECEL_FLOOR_ROLLBACK_MPS2, True)
  fix_rows = _window(fix, end_s=5.5)
  roll_rows = _window(rollback, end_s=5.5)
  fix_first = next(row for row in fix_rows if row["mpc_acc_source_debug"].get("approach_reacquire"))
  roll_first = next(row for row in roll_rows if row["mpc_acc_source_debug"].get("approach_reacquire"))
  fix_at_roll_t = next(row for row in fix_rows if row["t_s"] == roll_first["t_s"])

  roll_dbg = roll_first["mpc_acc_source_debug"]
  fix_same_dbg = fix_at_roll_t["mpc_acc_source_debug"]
  fix_dbg = fix_first["mpc_acc_source_debug"]
  physics = (
    f"moderate approach ownership: rollback t={roll_first['t_s']:.2f}s at raw TTC-to-headway "
    f"{roll_dbg['raw_ttc_to_headway_s']:.2f}s / amplified threshold "
    f"{roll_dbg['approach_reacquire_ttc_threshold_s']:.2f}s; fix waits until t={fix_first['t_s']:.2f}s at "
    f"raw TTC {fix_dbg['raw_ttc_to_headway_s']:.2f}s / steady threshold "
    f"{fix_dbg['approach_reacquire_ttc_threshold_s']:.2f}s"
  )
  assert roll_dbg["approach_reacquire_lead_decel_mps2"] > 1.0, physics
  assert roll_dbg["approach_reacquire_ttc_threshold_s"] > 7.0, physics
  # The integrated closing-recovery bridge may already own the urgency bypass;
  # either reason is fail-safe, while the assertions above prove amplification
  # still controls the early ownership threshold in this isolated A/B.
  assert roll_first["planner_handoff_limit_debug"]["bypass_reason"] in ("lead_decel", "closing"), physics

  assert fix_same_dbg["approach_reacquire"] is False, physics
  assert fix_same_dbg["approach_reacquire_lead_decel_mps2"] < 0.05, physics
  assert fix_same_dbg["approach_reacquire_ttc_threshold_s"] < 4.6, physics
  assert fix_same_dbg["raw_ttc_to_headway_s"] > fix_same_dbg["approach_reacquire_ttc_threshold_s"], physics
  assert fix_first["t_s"] >= roll_first["t_s"] + 0.5, physics
  assert fix_dbg["raw_ttc_to_headway_s"] <= fix_dbg["approach_reacquire_ttc_threshold_s"], physics


def test_dense_position_proof_prevents_far_false_tap_with_floor_rollback() -> None:
  protected = _run(FAR_CAP_TAP, MODEL_DECEL_FLOOR_ROLLBACK_MPS2)
  unprotected = _run(FAR_CAP_TAP, MODEL_DECEL_FLOOR_ROLLBACK_MPS2, True)
  protected_drop, _, protected_row = _worst_drop(protected, end_s=4.2)
  unprotected_drop, _, unprotected_row = _worst_drop(unprotected, end_s=4.2)

  physics = (
    f"dense-position proof at t={protected_row['t_s']:.2f}s: protected delta "
    f"{protected_drop:+.3f}, aTarget {protected_row['planner_accel_mps2']:+.3f}; "
    f"proof-disabled delta {unprotected_drop:+.3f}, "
    f"aTarget {unprotected_row['planner_accel_mps2']:+.3f}"
  )
  protected_debug = protected_row["mpc_lead_stability_debug"]["slot0"]
  assert protected_debug["accel_corr_calm_position_valid"], physics
  assert protected_debug["accel_corr_amplify_vetoed"], physics
  assert not protected_debug["accel_corr_amplified"], physics
  assert protected_drop >= -ROAD_HANDOFF_MAX_DELTA_MPS2 - 1e-9, physics
  assert unprotected_row["mpc_lead_stability_debug"]["slot0"]["accel_corr_amplified"], physics
  assert unprotected_drop < -0.6, physics


def test_cd3_meaningful_model_brake_still_amplifies() -> None:
  # Reuse the established genuine-braking safety scenario: true -1.8 m/s^2,
  # model reports 0.3x. Its published -0.367 signal must still amplify promptly.
  from selfdrive.test.longitudinal_harness.tests.test_repro_lead_decel_deficit import (
    DECEL_START_S,
    REPORTED_ACCEL_RATIO,
    _run as _run_cd3,
  )

  eligible = next(
    row for row in _planner_rows(_run_cd3(REPORTED_ACCEL_RATIO))
    if row["t_s"] >= DECEL_START_S
    and row["lead_one_published_a_lead_k_mps2"] is not None
    and row["lead_one_published_a_lead_k_mps2"] <= -0.35
    and row["mpc_acc_source_debug"].get("approach_reacquire_lead_decel_mps2", 0.0) >= 1.0
  )
  assert eligible["t_s"] <= DECEL_START_S + 0.5
  assert eligible["mpc_acc_source_debug"]["approach_reacquire_lead_decel_mps2"] >= 1.0
