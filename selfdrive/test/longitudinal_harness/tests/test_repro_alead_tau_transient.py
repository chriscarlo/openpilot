"""Repro R9 (docs/chauffeur/longitudinal/test_runtime_gap_audit_20260701.md):

Ego brakes/hangs back for seconds after a transient negative vision aLeadK even
though the lead never actually slowed, because the EV6 runtime publishes
aLeadTau=0.3 (radard.py model leads) and extrapolate_lead retains ~55% of the
blip 2 s into the MPC horizon versus ~5% at the legacy radar-track tau=1.5 every
green test used to feed the MPC.
"""
from __future__ import annotations

from functools import lru_cache

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
from selfdrive.test.longitudinal_harness.closed_loop import run_harness
from selfdrive.test.longitudinal_harness.config import (
  EV6_MODEL_LEAD_A_LEAD_TAU_S,
  resolve_ev6_vehicle_config,
)
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

CRUISE_SPEED_MPS = 30.0
EGO_V0_MPS = 28.0
LEAD_SPEED_MPS = 28.0
INITIAL_GAP_M = 36.0
DURATION_S = 10.0
# 0.6 s aLeadK burst mimicking a post-EMA vision accel transient (radard.py
# filters aLeadK with a 0.60 s EMA, so real transients survive at this width).
BLIP_START_S = 4.0
BLIP_END_S = 4.6
BLIP_A_LEAD_K_MPS2 = -1.0

# Audit R9 must-fail bounds: a lead whose truth never changes should not draw a
# sustained planner decel, a multi-second speed sag, or a ballooned gap.
PLANNER_BRAKE_FLOOR_MPS2 = -0.35
MAX_SPEED_LOSS_MPS = 0.6
MAX_GAP_GROWTH_RATIO = 1.15
# Post-blip persistence window: the directive aLeadK is back to 0.0 by 4.6 s,
# so any braking here is the MPC's retained lead-accel estimate shaped by tau.
POST_BLIP_LO_S = 5.4
POST_BLIP_HI_S = 7.0
BRAKE_HOLD_THRESHOLD_MPS2 = -0.15


def _blip_steps() -> list[StepInput]:
  steps: list[StepInput] = []
  for idx in range(int(round(DURATION_S / DT_MDL))):
    t_s = idx * DT_MDL
    in_blip = BLIP_START_S <= t_s < BLIP_END_S
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_SPEED_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=LEAD_SPEED_MPS,
        model_prob_target=0.9,
        d_rel_override_m=INITIAL_GAP_M if idx == 0 else None,
        a_lead_k_mps2=BLIP_A_LEAD_K_MPS2 if in_blip else 0.0,
      ),
      note="steady 28 m/s lead; 0.6 s aLeadK=-1.0 vision transient at 4.0 s",
    ))
  return steps


@lru_cache(maxsize=None)
def _run(a_lead_tau_s: float, corr_bound_disabled: bool = False):
  # Direct perception is required here: the radard stage always republishes
  # aLeadTau=0.3 for accepted leads, which would erase the tau A/B under test.
  vehicle = resolve_ev6_vehicle_config(
    a_lead_tau_s=a_lead_tau_s,
    perception_filter="direct",
    param_overrides={"lead_accel_corr_margin_mps2": "10.0"} if corr_bound_disabled else None,
  )
  return run_harness(
    vehicle_config=vehicle,
    scenario_name="alead_blip_steady_follow",
    steps=_blip_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="off",
    seed=42,
  )


def _measure(result) -> dict[str, float | None]:
  rows = result.trace

  def _rows(lo: float, hi: float) -> list[dict]:
    return [r for r in rows if lo <= r["t_s"] <= hi]

  blip_window = _rows(BLIP_START_S, POST_BLIP_HI_S)
  sag_window = _rows(BLIP_START_S, 9.0)
  v_at_blip = next(r["v_ego_true_mps"] for r in rows if r["t_s"] >= BLIP_START_S)
  min_v = min(r["v_ego_true_mps"] for r in sag_window)
  gap_at_blip = next(r["lead_one_true_d_rel_m"] for r in rows if r["t_s"] >= BLIP_START_S)
  gap_at_9s = next(r["lead_one_true_d_rel_m"] for r in rows if r["t_s"] >= 9.0)
  return {
    "aLeadTauS": result.vehicle["aLeadTauS"],
    "brake_onset_s": next(
      (r["t_s"] for r in rows if r["t_s"] >= BLIP_START_S and r["planner_accel_mps2"] <= PLANNER_BRAKE_FLOOR_MPS2),
      None),
    "brake_hold_until_s": max(
      (r["t_s"] for r in rows if r["t_s"] >= BLIP_START_S and r["planner_accel_mps2"] <= BRAKE_HOLD_THRESHOLD_MPS2),
      default=None),
    "min_planner_accel_mps2": min(r["planner_accel_mps2"] for r in blip_window),
    "min_post_blip_planner_accel_mps2": min(r["planner_accel_mps2"] for r in _rows(POST_BLIP_LO_S, POST_BLIP_HI_S)),
    "peak_controller_brake_mps2": result.summary["peakControllerBrakeMps2"],
    "v_at_blip_mps": v_at_blip,
    "min_v_mps": min_v,
    "speed_loss_mps": v_at_blip - min_v,
    "gap_at_blip_m": gap_at_blip,
    "gap_at_9s_m": gap_at_9s,
    "gap_growth_ratio": gap_at_9s / gap_at_blip,
    "min_true_gap_m": result.summary["minTrueGapM"],
  }


def _fmt(metrics: dict[str, float | None]) -> str:
  return " ".join(
    f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
    for key, value in metrics.items()
  )


# GAP R9 (test_runtime_gap_audit_20260701.md): was strict-xfail until the
# aLeadK corroboration bound (LeadAccelCorr* live-tune params) clamped
# uncorroborated transient lead decel at the _stabilize_raw_leads ingress;
# now a green regression test.
def test_runtime_tau_alead_blip_stays_calm() -> None:
  result = _run(EV6_MODEL_LEAD_A_LEAD_TAU_S)
  metrics = _measure(result)
  assert metrics["aLeadTauS"] == pytest.approx(EV6_MODEL_LEAD_A_LEAD_TAU_S)

  # The lead's true kinematics never change, so a faithful response keeps the
  # planner near coast, holds speed, and does not let the gap balloon.
  assert metrics["min_planner_accel_mps2"] >= PLANNER_BRAKE_FLOOR_MPS2, (
    f"sustained planner brake on a lead that never slowed: {_fmt(metrics)}")
  assert metrics["speed_loss_mps"] <= MAX_SPEED_LOSS_MPS, (
    f"multi-second speed sag from a 0.6 s aLeadK transient: {_fmt(metrics)}")
  assert metrics["gap_growth_ratio"] <= MAX_GAP_GROWTH_RATIO, (
    f"gap ballooned behind a constant-speed lead: {_fmt(metrics)}")


def test_alead_blip_persistence_tracks_a_lead_tau() -> None:
  # Mechanism attribution for the (former) xfail above, re-baselined when the
  # aLeadK corroboration bound landed. Two halves:
  #   (1) With the bound at the shipped LeadAccelCorrMarginMps2=0.5, the blip is
  #       clamped for BOTH taus: min planner accel stays above the audit's
  #       -0.35 floor at tau=0.3 and tau=1.5, and the tau-dependent brake-hold
  #       persistence collapses (the old >=0.5 s divergence is gone).
  #   (2) Disabling the bound (margin >= 10) restores the measured
  #       tau-dependent divergence — the original R9 assertions, kept verbatim
  #       as the rollback regression test.
  runtime = _measure(_run(EV6_MODEL_LEAD_A_LEAD_TAU_S))
  legacy = _measure(_run(float(_LEAD_ACCEL_TAU)))
  assert legacy["aLeadTauS"] == pytest.approx(1.5)

  detail = f"runtime[{_fmt(runtime)}] legacy[{_fmt(legacy)}]"
  # Measured at landing: min planner accel -0.184 at both taus (bound floor
  # min(0, a_meas) - 0.5 during the blip), brake hold until 4.74 s at both.
  assert runtime["min_planner_accel_mps2"] >= PLANNER_BRAKE_FLOOR_MPS2, detail
  assert legacy["min_planner_accel_mps2"] >= PLANNER_BRAKE_FLOOR_MPS2, detail
  assert runtime["min_post_blip_planner_accel_mps2"] >= PLANNER_BRAKE_FLOOR_MPS2, detail
  assert legacy["min_post_blip_planner_accel_mps2"] >= PLANNER_BRAKE_FLOOR_MPS2, detail
  assert runtime["brake_hold_until_s"] < legacy["brake_hold_until_s"] + 0.5, detail

  runtime_off = _measure(_run(EV6_MODEL_LEAD_A_LEAD_TAU_S, corr_bound_disabled=True))
  legacy_off = _measure(_run(float(_LEAD_ACCEL_TAU), corr_bound_disabled=True))
  detail_off = f"runtime[{_fmt(runtime_off)}] legacy[{_fmt(legacy_off)}]"
  # Measured with the bound disabled (same numbers as the pre-fix baseline):
  # post-blip min -0.239 vs -0.182 m/s^2, brake held until 6.69 s vs 5.64 s,
  # speed loss 0.799 vs 0.623 m/s.
  assert runtime_off["min_post_blip_planner_accel_mps2"] < legacy_off["min_post_blip_planner_accel_mps2"] - 0.03, detail_off
  assert runtime_off["brake_hold_until_s"] > legacy_off["brake_hold_until_s"] + 0.5, detail_off
  assert runtime_off["speed_loss_mps"] > legacy_off["speed_loss_mps"] + 0.1, detail_off
