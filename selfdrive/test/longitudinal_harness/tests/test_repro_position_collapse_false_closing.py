"""Repro: same-track model-position collapse fabricates freeway closing speed.

Route anchor (2026-07-12, 2023 CAN-FD HDA2 Kia EV6):
``0000022a--4f781e2080--1``, plan monotonic 199.745-200.391 s. The
same ``radarTrackId=-1004`` raw model lead collapsed from 49.25 m to 37.37 m
in 0.646 s while its velocity and acceleration streams stayed benign. Around
the trigger, the 0.6 s raw-vRel mean was only about -0.42 m/s and raw aLead was
about +0.03 m/s^2, yet RadarD published -1.96/-1.99 m/s and the planner stepped
from +0.05 to -1.00 m/s^2.

The mechanism is the closing governor's binary position-trust grant: the raw
dRel slope arms the governor, position-derived TTC below 6 s grants the full
``ClosingGovernorPosTrustExcessMps=1.5`` even though neither independent raw
velocity nor aLead corroborates that much closing, and the publish clamp turns
roughly -0.4 m/s into -1.9 m/s. This deterministic replay preserves the route's
three essential inputs while removing unrelated model jitter:

* raw dRel collapses 12.8 m over 0.6 s and then recovers;
* raw vRel stays at a calm -0.4 m/s (below the 0.75 m/s trust threshold);
* raw aLead stays at 0.0 m/s^2.

The scenario runs through the real RadarD EV6 perception stage, planner,
LongControl, Hyundai controller, delay, and vehicle plant. The behavioral
oracle permits up to 0.5 m/s of position-derived publish headroom; that amount
measures -0.25 m/s^2 peak planner braking and 0.23 m gap growth in this replay.
The pre-fix binary +1.5 m/s grant measures -1.00 m/s^2 and 1.66 m respectively.
"""
from __future__ import annotations

import functools

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 11.0
EVENT_START_S = 5.0
COLLAPSE_DURATION_S = 0.6
COLLAPSE_HOLD_S = 0.6
RECOVERY_DURATION_S = 0.6

EGO_AND_LEAD_SPEED_MPS = 29.5
INITIAL_TRUE_GAP_M = 49.0
RAW_GAP_START_M = 49.0
RAW_GAP_COLLAPSED_M = 36.2
RAW_VREL_MPS = -0.4
RAW_ALEAD_MPS2 = 0.0

# Post-fix contract. A bounded/graded position contribution may remain, but a
# weak raw velocity stream must never receive the full 1.5 m/s binary grant.
MAX_PUBLISH_CLOSING_EXCESS_MPS = 0.5
MIN_PLANNER_ACCEL_MPS2 = -0.30
MAX_TRUE_GAP_EXPANSION_M = 0.50
_EPS = 1e-6


def _raw_gap_m(t_s: float) -> float:
  collapse_end = EVENT_START_S + COLLAPSE_DURATION_S
  hold_end = collapse_end + COLLAPSE_HOLD_S
  recovery_end = hold_end + RECOVERY_DURATION_S
  if t_s < EVENT_START_S:
    return RAW_GAP_START_M
  if t_s < collapse_end:
    progress = (t_s - EVENT_START_S) / COLLAPSE_DURATION_S
    return RAW_GAP_START_M + progress * (RAW_GAP_COLLAPSED_M - RAW_GAP_START_M)
  if t_s < hold_end:
    return RAW_GAP_COLLAPSED_M
  if t_s < recovery_end:
    progress = (t_s - hold_end) / RECOVERY_DURATION_S
    return RAW_GAP_COLLAPSED_M + progress * (RAW_GAP_START_M - RAW_GAP_COLLAPSED_M)
  return RAW_GAP_START_M


def _build_steps() -> list[StepInput]:
  steps: list[StepInput] = []
  for idx in range(int(round(DURATION_S / DT_MDL))):
    t_s = idx * DT_MDL
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=EGO_AND_LEAD_SPEED_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=EGO_AND_LEAD_SPEED_MPS,
        model_prob_target=0.995,
        d_rel_override_m=INITIAL_TRUE_GAP_M if idx == 0 else None,
        measured_d_rel_m=_raw_gap_m(t_s),
        measured_v_rel_mps=RAW_VREL_MPS,
        a_lead_k_mps2=RAW_ALEAD_MPS2,
        acquisition_reset=idx == 0,
      ),
      note="route 22a same-track raw-position collapse with benign vRel/aLead",
    ))
  return steps


@functools.lru_cache(maxsize=1)
def _run() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="route_22a_position_collapse_false_closing",
    steps=_build_steps(),
    initial_speed_mps=EGO_AND_LEAD_SPEED_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="radard",
  )


def _planner_rows() -> list[dict]:
  return _run().trace[::5]


def _event_rows() -> list[dict]:
  return [row for row in _planner_rows()
          if EVENT_START_S - _EPS <= row["t_s"] <= EVENT_START_S + 5.0 + _EPS]


def test_position_collapse_repro_wiring() -> None:
  """Keep this a real-RadarD, same-track, position-only contradiction."""
  result = _run()
  rows = _event_rows()
  assert rows
  assert result.vehicle["candidate"] == "KIA_EV6"
  assert result.vehicle["radarUnavailable"] is True
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"

  raw_gaps = [row["lead_one_raw_d_rel_m"] for row in rows]
  assert max(raw_gaps) - min(raw_gaps) >= 12.7
  assert all(abs(row["lead_one_raw_v_rel_mps"] - RAW_VREL_MPS) < 1e-5 for row in rows)
  assert all(abs(row["lead_one_raw_a_lead_k_mps2"] - RAW_ALEAD_MPS2) < 1e-5 for row in rows)

  track_ids = {
    row["lead_one_radard_debug"]["track_id"]
    for row in rows
    if row["lead_one_radard_debug"]["track_id"] is not None
  }
  assert len(track_ids) == 1, f"scenario changed track identity: {sorted(track_ids)}"


def test_position_only_collapse_does_not_get_full_publish_trust_or_expand_gap() -> None:
  rows = _event_rows()
  pre_event = next(row for row in reversed(_planner_rows()) if row["t_s"] < EVENT_START_S)

  max_publish_excess = max(
    max(0.0, -row["lead_one_published_v_rel_mps"])
    - max(0.0, -row["lead_one_raw_v_rel_mps"])
    for row in rows
  )
  min_planner_accel = min(row["planner_accel_mps2"] for row in rows)
  max_gap_expansion = max(row["lead_one_true_d_rel_m"] for row in rows) - pre_event["lead_one_true_d_rel_m"]

  publish_error = (
    f"position-only collapse received {max_publish_excess:.3f} m/s publish excess; raw vRel={RAW_VREL_MPS:+.2f}, " +
    f"raw aLead={RAW_ALEAD_MPS2:+.2f} do not corroborate the full binary grant"
  )
  planner_error = (
    f"position-only collapse drove planner accel to {min_planner_accel:+.3f} m/s^2; calm bound is " +
    f"{MIN_PLANNER_ACCEL_MPS2:+.2f}"
  )
  gap_error = (
    f"position-only collapse expanded the true gap by {max_gap_expansion:.3f} m; bound is " +
    f"{MAX_TRUE_GAP_EXPANSION_M:.2f} m"
  )
  assert max_publish_excess <= MAX_PUBLISH_CLOSING_EXCESS_MPS + 1e-5, publish_error
  assert min_planner_accel >= MIN_PLANNER_ACCEL_MPS2, planner_error
  assert max_gap_expansion <= MAX_TRUE_GAP_EXPANSION_M, gap_error
