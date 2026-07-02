"""Repro R5 (docs/chauffeur/longitudinal/test_runtime_gap_audit_20260701.md):
a hard-braking lead loses model probability for 0.7 s; radard's Schmitt latch
drops the published lead, and the MPC lead-stabilizer phantom
(selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py:2413-2429) serves a
stand-in with the STALE drop-time vRel and an aLeadK decaying to zero — erasing
the lead's deceleration from the MPC prediction, so planned decel stops
deepening while the true gap keeps collapsing.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

CRUISE_SPEED_MPS = 29.0
EGO_INITIAL_SPEED_MPS = 27.0
LEAD_INITIAL_GAP_M = 32.0
LEAD_INITIAL_SPEED_MPS = 27.0
LEAD_BRAKE_ONSET_S = 3.0
LEAD_DECEL_MPS2 = 3.0
LEAD_FLOOR_SPEED_MPS = 15.0
# 0.7 s dropout sits inside PhantomLeadHoldS=0.80 (device snapshot
# docs/chauffeur/longitudinal/device_livetune_snapshot_20260701.txt /
# common/params_keys.h default), so the phantom — not unlatch — owns the window.
DROPOUT_START_S = 3.6
DROPOUT_END_S = 4.3
DURATION_S = 12.0
# Spec margin from R5 must-fail assertion (1): while the true lead brakes at
# -3 m/s^2, the planned decel must deepen by at least this much over the window.
MIN_WINDOW_DEEPENING_MPS2 = 0.3
BRAKE_ONSET_THRESHOLD_MPS2 = -1.5
_EPS_S = 1e-6


def _build_steps(*, dropout: bool, dt_s: float = DT_MDL) -> list[StepInput]:
  steps: list[StepInput] = []
  for idx in range(int(round(DURATION_S / dt_s))):
    t_s = idx * dt_s
    if t_s < LEAD_BRAKE_ONSET_S:
      lead_speed_mps = LEAD_INITIAL_SPEED_MPS
    else:
      lead_speed_mps = max(LEAD_FLOOR_SPEED_MPS,
                           LEAD_INITIAL_SPEED_MPS - LEAD_DECEL_MPS2 * (t_s - LEAD_BRAKE_ONSET_S))
    in_dropout = dropout and DROPOUT_START_S - _EPS_S <= t_s < DROPOUT_END_S - _EPS_S
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_SPEED_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=lead_speed_mps,
        # A prob dropout, not a track loss: status stays True so the plant-side
        # track keeps integrating ground truth; radard turns prob 0 into a
        # dropped published lead (Schmitt exit below LeadProbEnter/Exit).
        model_prob_target=0.0 if in_dropout else 0.98,
        d_rel_override_m=LEAD_INITIAL_GAP_M if idx == 0 else None,
        acquisition_reset=idx == 0,
      ),
      note="lead brakes -3 m/s^2 with a 0.7 s model prob dropout" if dropout else "lead brakes -3 m/s^2",
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _run(dropout: bool) -> SimulationResult:
  # Tici-fidelity default config: device controller mode + the real radard
  # pipeline in the loop so the prob dropout reaches the MPC the way the car's
  # perception stack delivers it.
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="phantom_freeze_dropout" if dropout else "phantom_freeze_clean",
    steps=_build_steps(dropout=dropout),
    initial_speed_mps=EGO_INITIAL_SPEED_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="auto",
  )


def _window_rows(result: SimulationResult) -> list[dict]:
  return [row for row in result.trace
          if DROPOUT_START_S - _EPS_S <= row["t_s"] < DROPOUT_END_S - _EPS_S]


def _brake_onset_s(result: SimulationResult) -> float | None:
  for row in result.trace:
    if row["t_s"] >= LEAD_BRAKE_ONSET_S and row["planner_accel_mps2"] <= BRAKE_ONSET_THRESHOLD_MPS2:
      return float(row["t_s"])
  return None


def _physics_report(clean: SimulationResult, dropout: SimulationResult) -> str:
  clean_window = _window_rows(clean)
  drop_window = _window_rows(dropout)
  lines = [
    f"phantom window [{DROPOUT_START_S:.2f}, {DROPOUT_END_S:.2f}) s; true lead braking at -{LEAD_DECEL_MPS2:.1f} m/s^2 throughout:",
    (f"  planner accel over window: dropout {drop_window[0]['planner_accel_mps2']:+.3f} -> {drop_window[-1]['planner_accel_mps2']:+.3f} m/s^2"
     f" (deepened {drop_window[0]['planner_accel_mps2'] - drop_window[-1]['planner_accel_mps2']:+.3f}),"
     f" clean {clean_window[0]['planner_accel_mps2']:+.3f} -> {clean_window[-1]['planner_accel_mps2']:+.3f} m/s^2"
     f" (deepened {clean_window[0]['planner_accel_mps2'] - clean_window[-1]['planner_accel_mps2']:+.3f})"),
    (f"  true gap over window (dropout run): {drop_window[0]['true_min_gap_m']:.1f} -> {drop_window[-1]['true_min_gap_m']:.1f} m"),
    (f"  brake onset (first planner accel <= {BRAKE_ONSET_THRESHOLD_MPS2:.1f}): dropout {_brake_onset_s(dropout)} s, clean {_brake_onset_s(clean)} s"),
    (f"  minTrueGapM: dropout {dropout.summary['minTrueGapM']:.2f} m, clean {clean.summary['minTrueGapM']:.2f} m"),
    (f"  peakPlannerBrakeMps2: dropout {dropout.summary['peakPlannerBrakeMps2']:.2f}, clean {clean.summary['peakPlannerBrakeMps2']:.2f}"),
    (f"  peakControllerBrakeMps2: dropout {dropout.summary['peakControllerBrakeMps2']:.2f}, clean {clean.summary['peakControllerBrakeMps2']:.2f}"),
  ]
  return "\n".join(lines)


def test_phantom_window_mechanism_is_active():
  """Guard: the xfail below must fail through the phantom, not a wiring bug."""
  clean = _run(False)
  dropout = _run(True)

  drop_window = _window_rows(dropout)
  assert drop_window, "dropout window produced no trace rows"
  # radard dropped the published lead for the whole window...
  assert all(row["lead_one_published_d_rel_m"] is None for row in drop_window)
  # ...while the MPC stayed in lead-following mode, i.e. the stabilizer phantom
  # (not lead loss) owned the window.
  assert all(row["planner_source"] == "lead0" for row in drop_window)
  # Published lead is live immediately before the window and reacquires at its
  # true kinematic position right after (track continuity is real).
  before = [row for row in dropout.trace if row["t_s"] < DROPOUT_START_S - _EPS_S][-1]
  after = [row for row in dropout.trace if row["t_s"] >= DROPOUT_END_S - _EPS_S][0]
  assert before["lead_one_published_d_rel_m"] is not None
  assert after["lead_one_published_d_rel_m"] is not None
  # Track continuity at reacquire: never optimistic vs truth; the closing-only
  # publish-side lag compensation (ModelLeadFilterLagCompS x closing, ~2.1 m at
  # this reacquire's closing rate) may move the published gap closer, never wider.
  assert after["lead_one_published_d_rel_m"] <= after["true_min_gap_m"] + 1.0
  assert after["lead_one_published_d_rel_m"] >= after["true_min_gap_m"] - 3.5
  # Ground truth keeps flowing during the dropout and the gap keeps collapsing.
  assert drop_window[-1]["true_min_gap_m"] < drop_window[0]["true_min_gap_m"] - 1.0
  # The clean twin never loses its published lead.
  assert all(row["lead_one_published_d_rel_m"] is not None for row in _window_rows(clean))


# GAP 4 / R5 (test_runtime_gap_audit_20260701.md): was strict-xfail until the
# lead-stabilizer phantom held measured decel (PhantomLeadDecelHoldFactor /
# PhantomLeadDecelTrendGain); now a green regression test.
def test_planned_decel_keeps_deepening_through_prob_dropout():
  clean = _run(False)
  dropout = _run(True)
  report = _physics_report(clean, dropout)

  drop_window = _window_rows(dropout)
  a_start = float(drop_window[0]["planner_accel_mps2"])
  a_end = float(drop_window[-1]["planner_accel_mps2"])
  # R5 must-fail (1): with the true lead braking at -3 m/s^2 the whole window,
  # the planned decel must keep deepening through the dropout.
  assert a_end <= a_start - MIN_WINDOW_DEEPENING_MPS2, (
    f"planned decel stopped deepening during the phantom window: "
    f"{a_start:+.3f} -> {a_end:+.3f} m/s^2 (needed <= {a_start - MIN_WINDOW_DEEPENING_MPS2:+.3f})\n{report}"
  )
  # R5 must-fail (2): the dropout must not cost meaningful true gap.
  assert dropout.summary["minTrueGapM"] >= clean.summary["minTrueGapM"] - 2.0, (
    f"prob dropout collapsed the true gap: {dropout.summary['minTrueGapM']:.2f} m "
    f"vs clean {clean.summary['minTrueGapM']:.2f} m\n{report}"
  )
  # R5 must-fail (3): no panic overbrake on reacquire.
  assert dropout.summary["peakControllerBrakeMps2"] >= clean.summary["peakControllerBrakeMps2"] - 0.5, (
    f"reacquire panic brake: {dropout.summary['peakControllerBrakeMps2']:.2f} m/s^2 "
    f"vs clean {clean.summary['peakControllerBrakeMps2']:.2f} m/s^2\n{report}"
  )
