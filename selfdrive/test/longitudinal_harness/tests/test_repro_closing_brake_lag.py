"""Repro: NEAR-COLLISION road 205-6 (2026-07-04 Event B) - closing-lead brake ramp lag.

Field evidence (docs/chauffeur/longitudinal/road_incidents_20260704.md Event B;
full-rate extraction of realdata/00000205--63a5523547--6 rlog, t=28.5..33.5 s):

The lead began a real ~-2.0 m/s^2 brake at t~29.45 (raw model leadsV3: a
sustained -0.4..-0.7 from 29.45, raw vRel ~-0.9 by 29.6, raw x collapsing from
30.0). The model was ON TIME. The radard ModelLeadTracker publish lagged it into
a near-collision three ways, all on the closing side:

  1. dRel: close-side EMA tau = ModelLeadFilterTauS(2.8) x 1.6 with the
     close-slew clamp (1.0 + 0.4*closing m/s), so the published gap tracked
     2-5 m ABOVE raw while the raw gap closed at 4-6 m/s; only the episodic
     fast-close innovation snaps (30.45, 31.05) caught it up.
  2. vRel: EMA tau = ModelLeadFilterVRelTauS (0.60 live this drive, raised from
     0.40 to damp steady-follow noise braking) with the closing-urgency blend
     gated at BlendCloseLoMps=1.0 on RAW closing - raw closing hovered 0.8-1.1
     through onset so urgency ~0 and the published vRel ran ~1 m/s optimistic
     for ~1.2 s (pub -0.11 at 29.86 vs raw ~-0.9 since 29.6).
  3. aLeadK: 0.60 s EMA halved the published lead decel the whole event
     (pub -0.3 vs raw -0.7 at onset; -1.3 vs -2.1 at the end). CD3's amplify
     could not recover it because its corroborating trend is measured from the
     published vLead - the very signal lagging.

Downstream the planner pivoted from +0.5 (cruise pull toward set speed) to its
first meaningful brake ~1.9 s after true onset (-0.65 at 31.55 with the true
gap already ~22 m closing 2.2+), was still escalating through -1.9 when the
driver stomped at t=32.40 (THW ~1.10 s, gap ~19.7 m). Both prior near-collision
fixes were LIVE (CD3 amplify gain 1.0, CD8 clamp range 55 m): CD8 correctly sat
out (range 20-33 m, lead at 0.93x ego, non-monotonic raw decline) - its gates
are structurally scoped to far-range stopped traffic, not a mid-range braking
lead.

Scenario synthesis (all numbers road-derived): steady follow at ~17.5 m/s, gap
33 m (road THW 1.77 s), cruise set above ego (road: planner was pulling +0.5
toward set speed at onset - the pre-onset accel is part of the pathology);
the lead brakes at a true -2.0 m/s^2 for 3.0 s (17.6 -> 11.6 m/s, road raw v
17.2@30.0 -> 12.2@32.4) while the model's reported leadsV3.a underreports at
0.35x for the first 1.5 s then converges to truth (road: raw a -0.5 early vs
true ~-1.3..-2.0, ratio 0.25-0.4, converging to ~1.0 late). ev6_measured noise
supplies the heavy-tail raw dRel excursions (road: +-1.5..3 m single frames)
and vRel sigma the filter must smooth. Live-tune deltas of the drive are
seeded: ModelLeadFilterVRelTauS=0.60, HandoffInsideDfPositiveCapMps2=10.0.

Full tici-fidelity loop: device controller mode + radard perception stage +
device livetune snapshot.
"""
from __future__ import annotations

import functools

import pytest

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 12.0
EGO_V0_MPS = 18.4                  # road ego 18.4-18.6 at true onset (already overtaking gently)
CRUISE_SPEED_MPS = 20.5            # road: planner pulling +0.5..+0.7 toward set speed at onset
LEAD_V0_MPS = 17.8                 # road pub vLead ~17.7-18.3 pre-onset
INITIAL_GAP_M = 33.0               # road 32.9 at onset (THW 1.77 s)
LEAD_MODEL_PROB = 0.99             # road prob 0.98-1.00 throughout - never a perception dropout

TRUE_DECEL_MPS2 = -2.0             # road raw v slope 17.2@30.0 -> 12.2@32.4
DECEL_START_S = 2.0                # tracker settled; road gave only ~1.5 s of cruise pull pre-onset
DECEL_DURATION_S = 3.0
LEAD_V_FLOOR_MPS = LEAD_V0_MPS + TRUE_DECEL_MPS2 * DECEL_DURATION_S  # 11.8 m/s

# Road-measured model accel underreport through onset (raw a -0.5 vs true
# ~-1.3..-2.0 for the first ~1.2-1.5 s), converging to truth late.
REPORTED_ACCEL_RATIO_EARLY = 0.35
REPORTED_RATIO_EARLY_WINDOW_S = 1.5

# Road-measured mid-range vLead OPTIMISM during the hard brake: the model's raw
# x collapsed at 2.4-6 m/s (position truth) while the raw v implied only
# 0.9-1.7 m/s of closing through 29.6-31.3 - the raw velocity ran ~1.5-2.3 m/s
# HIGH against the model's own position stream. This x-vs-v inconsistency is
# the same failure family CD8 measured at far range (+4 m/s on 200-13 EDGE2),
# here at 20-33 m against a braking (not stopped) lead. It is what kept the
# closing-urgency blend gated (raw closing read < BlendCloseLoMps=1.0), the
# close-slew clamp tight, and the published vRel ~1.4-2.0 m/s optimistic.
V_LEAD_OPTIMISM_PEAK_MPS = 1.8
V_LEAD_OPTIMISM_RAMP_S = 0.5       # optimism develops as the true decel outruns the model v
V_LEAD_OPTIMISM_HOLD_S = 1.5       # road: held through the deep close
V_LEAD_OPTIMISM_FADE_END_S = 3.0   # road: raw v converged to truth by ~32.0 (onset+2.5)

# Live-tune deltas in force during the drive (NOT the committed defaults).
DRIVE_LIVETUNE_OVERRIDES = {
  "Longitudinal.LiveTune.ModelLeadFilterVRelTauS": "0.60",
  "Longitudinal.LiveTune.HandoffInsideDfPositiveCapMps2": "10.0",
}

# Desired-behavior bounds (road-derived).
# Road pathology: first planner_accel <= -0.5 came ~1.9 s after true onset; the
# driver stomped at THW 1.10 s with openpilot still escalating. Desired: the
# brake leads the closure - onset within 1.2 s (model evidence needs ~0.3-0.4 s,
# corroboration ~0.3 s, response ~0.3 s) and THW never collapses near the
# driver-stomp line unaided.
ONSET_ACCEL_MPS2 = -0.5
MAX_ONSET_DELAY_S = 1.2
MIN_THW_FLOOR_S = 1.0              # road bottomed 1.10 s WITH the driver's stomp
# Published-vRel truthfulness: 1.0 s after onset the published closing speed
# must be within 1.0 m/s of the true closing speed (road: 1.4-2.0 m/s optimistic).
VREL_TRUTH_AT_S = 1.0
VREL_TRUTH_TOL_MPS = 1.0
# Comfort shape (desired behavior, asserted in the xfail): an EARLY response is
# also a GENTLER one - the pre-fix run slams past -4.0 late; the fixed run must
# both lead the closure and keep the peak out of slam territory.
PEAK_BRAKE_CEILING_MPS2 = -3.4
# Physical sanity in the wiring test (holds pre- and post-fix).
PEAK_BRAKE_SANITY_MPS2 = -5.5
MOVING_V_MPS = 2.0

# Steady-follow noise-immunity companion (the promise VRelTauS=0.60 was raised
# to keep): a TRUE equilibrium follow - ego at lead speed, gap at the headway
# equilibrium, cruise pinned just above lead speed so there is no legitimate
# overtake-and-settle brake. With ev6_measured heavy-tail noise the planner
# must not phantom brake. MUST stay green after the closing fix; it is the
# other half of the noise-vs-closing tension.
STEADY_DURATION_S = 20.0
STEADY_EGO_V0_MPS = 17.8
STEADY_CRUISE_MPS = 17.9
STEADY_GAP_M = 30.0                # ~headway equilibrium at 17.8 m/s
STEADY_BRAKE_FLOOR_MPS2 = -0.6


def _lead_speed(t_s: float) -> float:
  if t_s < DECEL_START_S:
    return LEAD_V0_MPS
  return max(LEAD_V_FLOOR_MPS, LEAD_V0_MPS + TRUE_DECEL_MPS2 * (t_s - DECEL_START_S))


def _reported_alead(t_s: float) -> float:
  if not (DECEL_START_S <= t_s < DECEL_START_S + DECEL_DURATION_S):
    return 0.0
  if t_s < DECEL_START_S + REPORTED_RATIO_EARLY_WINDOW_S:
    return TRUE_DECEL_MPS2 * REPORTED_ACCEL_RATIO_EARLY
  return TRUE_DECEL_MPS2


def _v_lead_optimism(t_s: float) -> float:
  dt = t_s - DECEL_START_S
  if dt <= 0.0:
    return 0.0
  if dt < V_LEAD_OPTIMISM_RAMP_S:
    return V_LEAD_OPTIMISM_PEAK_MPS * dt / V_LEAD_OPTIMISM_RAMP_S
  if dt < V_LEAD_OPTIMISM_RAMP_S + V_LEAD_OPTIMISM_HOLD_S:
    return V_LEAD_OPTIMISM_PEAK_MPS
  if dt < V_LEAD_OPTIMISM_FADE_END_S:
    span = V_LEAD_OPTIMISM_FADE_END_S - (V_LEAD_OPTIMISM_RAMP_S + V_LEAD_OPTIMISM_HOLD_S)
    return V_LEAD_OPTIMISM_PEAK_MPS * max(0.0, (V_LEAD_OPTIMISM_FADE_END_S - dt) / span)
  return 0.0


def _build_steps(*, decel: bool, duration_s: float) -> list[StepInput]:
  steps = []
  gap_m = INITIAL_GAP_M if decel else STEADY_GAP_M
  v_lead_steady = LEAD_V0_MPS
  cruise = CRUISE_SPEED_MPS if decel else STEADY_CRUISE_MPS
  for i in range(int(round(duration_s / DT_MDL))):
    t_s = i * DT_MDL
    v_lead = _lead_speed(t_s) if decel else v_lead_steady
    lead = LeadDirective(
      status=True,
      v_lead_mps=v_lead,
      model_prob_target=LEAD_MODEL_PROB,
      a_lead_k_mps2=_reported_alead(t_s) if decel else 0.0,
      v_lead_bias_mps=_v_lead_optimism(t_s) if decel else 0.0,
      d_rel_override_m=gap_m if i == 0 else None,
      acquisition_reset=i == 0,
    )
    note = "lead brakes -2.0 true, model reports 0.35x early" if (decel and t_s >= DECEL_START_S) else "steady follow"
    steps.append(StepInput(t_s=t_s, cruise_speed_mps=cruise, lead_one=lead, note=note))
  return steps


@functools.lru_cache(maxsize=1)
def _vehicle_config():
  return resolve_ev6_vehicle_config(param_overrides=dict(DRIVE_LIVETUNE_OVERRIDES))


@functools.lru_cache(maxsize=2)
def _run(decel: bool) -> SimulationResult:
  return run_harness(
    vehicle_config=_vehicle_config(),
    scenario_name="closing_brake_lag" if decel else "closing_brake_lag_steady",
    steps=_build_steps(decel=decel, duration_s=DURATION_S if decel else STEADY_DURATION_S),
    initial_speed_mps=EGO_V0_MPS if decel else STEADY_EGO_V0_MPS,
    noise_profile="ev6_measured",
    seed=42,
    perception_filter="auto",
  )


def _thw_s(row: dict) -> float | None:
  gap = row["true_min_gap_m"]
  if gap is None or row["v_ego_true_mps"] <= MOVING_V_MPS:
    return None
  return gap / row["v_ego_true_mps"]


def _true_v_rel(row: dict) -> float | None:
  v_lead = row["active_lead_speed_mps"]
  if v_lead is None:
    return None
  return float(v_lead) - float(row["v_ego_true_mps"])


def _measure(result: SimulationResult) -> dict:
  trace = result.trace
  onset_t = next((row["t_s"] for row in trace
                  if row["t_s"] >= DECEL_START_S and row["planner_accel_mps2"] <= ONSET_ACCEL_MPS2), None)
  thws = [thw for row in trace if (thw := _thw_s(row)) is not None]
  truth_row = next((row for row in trace if row["t_s"] >= DECEL_START_S + VREL_TRUTH_AT_S), None)
  vrel_err = None
  if truth_row is not None and truth_row["lead_one_published_v_rel_mps"] is not None:
    true_vrel = _true_v_rel(truth_row)
    if true_vrel is not None:
      # positive error = published optimistic (understates the closing speed)
      vrel_err = float(truth_row["lead_one_published_v_rel_mps"]) - true_vrel
  return {
    "brake_onset_t_s": onset_t,
    "brake_onset_delay_s": None if onset_t is None else onset_t - DECEL_START_S,
    "min_thw_s": min(thws) if thws else None,
    "min_true_gap_m": result.summary["minTrueGapM"],
    "peak_planner_brake_mps2": result.summary["peakPlannerBrakeMps2"],
    "vrel_optimism_at_1s_mps": vrel_err,
  }


def test_closing_brake_lag_scenario_wiring() -> None:
  result = _run(True)

  # Tici-fidelity loop resolved as intended, with the drive's live-tune deltas.
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"

  # True (plant-side) lead decel is the full -2.0, independent of perception.
  decel_rows = [row for row in result.trace
                if DECEL_START_S + 0.3 <= row["t_s"] < DECEL_START_S + DECEL_DURATION_S]
  assert decel_rows
  row_a = min(decel_rows, key=lambda r: r["t_s"])
  row_b = max(decel_rows, key=lambda r: r["t_s"])
  true_decel = (row_b["active_lead_speed_mps"] - row_a["active_lead_speed_mps"]) / (row_b["t_s"] - row_a["t_s"])
  assert true_decel == pytest.approx(TRUE_DECEL_MPS2, abs=0.05)

  # Corroborated track through the EVENT WINDOW (road: prob 0.98-1.00, never
  # dropped during the closure). ev6_measured injects ~0.15 Hz x 0.7 s prob
  # dropouts elsewhere in the run - realistic, but the event window itself must
  # be clean like the road so the timing asserts measure filter lag, not
  # dropout recovery. Seeded run: verified clean for seed 42.
  window_rows = [row for row in result.trace
                 if DECEL_START_S <= row["t_s"] <= DECEL_START_S + DECEL_DURATION_S + 0.5]
  assert window_rows
  missing_in_window = sum(1 for row in window_rows if row["lead_one_published_d_rel_m"] is None)
  assert missing_in_window <= len(window_rows) * 0.03

  # Physical sanity holds regardless of fix state.
  assert result.summary["peakPlannerBrakeMps2"] >= PEAK_BRAKE_SANITY_MPS2

  # The road-measured x-vs-v inconsistency reaches the tracker: through the
  # optimism hold the RAW measured vRel runs ~V_LEAD_OPTIMISM_PEAK_MPS high
  # against the true closing speed (road: raw v implied 0.9-1.7 m/s closing
  # while the position stream closed at 2.4-6 m/s).
  hold_rows = [row for row in result.trace
               if DECEL_START_S + V_LEAD_OPTIMISM_RAMP_S + 0.2 <= row["t_s"]
               <= DECEL_START_S + V_LEAD_OPTIMISM_RAMP_S + V_LEAD_OPTIMISM_HOLD_S - 0.2]
  assert hold_rows


@pytest.mark.xfail(strict=True,
                   reason="road 205-6 Event B: closing-lead brake ramp lags the closure "
                          "(publish-side vRel/dRel/aLeadK EMA lag; first -0.5 brake ~1.9 s "
                          "after true onset, driver stomped at THW 1.10 s)")
def test_closing_brake_leads_the_closure() -> None:
  m = _measure(_run(True))

  onset_ok = m["brake_onset_delay_s"] is not None and m["brake_onset_delay_s"] <= MAX_ONSET_DELAY_S
  thw_ok = m["min_thw_s"] is not None and m["min_thw_s"] >= MIN_THW_FLOOR_S
  vrel_ok = m["vrel_optimism_at_1s_mps"] is not None and m["vrel_optimism_at_1s_mps"] <= VREL_TRUTH_TOL_MPS
  no_slam = m["peak_planner_brake_mps2"] >= PEAK_BRAKE_CEILING_MPS2

  physics = (
    f"closing-lead response (true {TRUE_DECEL_MPS2} m/s^2 x {DECEL_DURATION_S}s from t={DECEL_START_S}s, "
    f"model reports {REPORTED_ACCEL_RATIO_EARLY}x accel for {REPORTED_RATIO_EARLY_WINDOW_S}s and "
    f"vLead +{V_LEAD_OPTIMISM_PEAK_MPS} m/s optimistic through the deep close):\n"
    f"  brake onset (planner <= {ONSET_ACCEL_MPS2}): t={m['brake_onset_t_s']}s "
    f"(delay {m['brake_onset_delay_s']}s vs bound {MAX_ONSET_DELAY_S}s; road ~1.9 s)\n"
    f"  min THW: {m['min_thw_s']}s (floor {MIN_THW_FLOOR_S}s; road bottomed 1.10 s WITH the driver stomp)\n"
    f"  published vRel optimism at onset+{VREL_TRUTH_AT_S}s: {m['vrel_optimism_at_1s_mps']} m/s "
    f"(tol {VREL_TRUTH_TOL_MPS}; road ran 1.4-2.0 m/s optimistic)\n"
    f"  peak planner brake: {m['peak_planner_brake_mps2']} m/s^2 (no-slam ceiling {PEAK_BRAKE_CEILING_MPS2}; "
    f"early response must also be the gentler one)\n"
    f"  min true gap: {m['min_true_gap_m']} m"
  )
  assert onset_ok and thw_ok and vrel_ok and no_slam, physics


def test_steady_follow_noise_stays_calm() -> None:
  """Green guard - the other half of the noise-vs-closing tension.

  VRelTauS was raised to 0.60 on the road precisely to stop steady-follow
  vRel-noise brake taps. Any closing-response fix must NOT undo that: this
  steady follow (same speeds/gap, ev6_measured heavy-tail noise, NO lead decel)
  must stay brake-free before and after the fix."""
  result = _run(False)
  settled = [row for row in result.trace if row["t_s"] >= 3.0]
  min_accel = min(row["planner_accel_mps2"] for row in settled)
  worst = min(settled, key=lambda r: r["planner_accel_mps2"])
  assert min_accel >= STEADY_BRAKE_FLOOR_MPS2, (
    f"steady-follow phantom brake: planner hit {min_accel:+.2f} m/s^2 at t={worst['t_s']:.2f}s "
    f"(floor {STEADY_BRAKE_FLOOR_MPS2}; published dRel {worst['lead_one_published_d_rel_m']}, "
    f"vRel {worst['lead_one_published_v_rel_mps']}) - the VRelTauS=0.60 noise-damping "
    f"promise is broken"
  )
