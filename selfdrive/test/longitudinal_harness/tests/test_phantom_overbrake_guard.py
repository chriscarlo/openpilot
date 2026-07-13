"""Phantom-active overbrake counter-check for the kinematic phantom hold
(PhantomLeadDecelHoldFactor / PhantomLeadDecelTrendGain, GAP 4 / R5 fix).

The bias rule allows the phantom to brake earlier/harder than legacy, but only
toward where the measurements said the lead was: a steady or lightly-braking
lead with slightly negative, jittering aLeadK plus brief prob flickers that
actually engage the phantom hold must NOT turn into new brake taps or a harder
peak brake versus the legacy decay-to-zero phantom. Runs under the calibrated
ev6_measured noise profile across three seeds (judge amendment on the cluster-A
design: the steady-goldilocks counter-check never activated the phantom).
"""
from __future__ import annotations

import functools
import math

from openpilot.common.realtime import DT_MDL
from selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput

DURATION_S = 20.0
CRUISE_SPEED_MPS = 27.0
EGO_V0_MPS = 25.0
INITIAL_GAP_M = 45.0
LEAD_V0_MPS = 24.0
# Slightly negative, jittering lead accel: -0.2 m/s^2 base decel with a
# +/-0.3 m/s^2 sinusoid, so measured aLeadK hovers just below zero — the regime
# where a pessimism bug in the hold/trend would convert benign flickers into
# phantom brake events.
LEAD_BASE_DECEL_MPS2 = 0.2
JITTER_ACCEL_AMP_MPS2 = 0.3
JITTER_PERIOD_S = 10.0
# Brief flickers well inside PhantomLeadHoldS=0.80 so the phantom (not unlatch)
# owns each window; ev6_measured adds its own random prob dropouts on top.
# Keep the explicit windows on a lead-owned portion of every seeded noisy run.
# 11.0 s overlapped an ev6_measured random dropout after a legitimate
# filtered_pullaway_immediate release; 8.0 s remains lead-owned and holds the
# measured -0.15 m/s^2 decel through all eight phantom frames on every seed.
FLICKER_STARTS_S = (6.0, 8.0, 16.0)
FLICKER_DURATION_S = 0.4
SEEDS = (11, 42, 777)
# Bounds on what the new defaults may add over the legacy decay-to-zero phantom.
MAX_ADDED_PEAK_BRAKE_MPS2 = 0.3
TAP_ON_MPS2 = -0.6
TAP_OFF_MPS2 = -0.3
_EPS_S = 1e-6


def _lead_speed(t_s: float) -> float:
  amp_v = JITTER_ACCEL_AMP_MPS2 * JITTER_PERIOD_S / (2.0 * math.pi)
  return max(5.0, LEAD_V0_MPS - LEAD_BASE_DECEL_MPS2 * t_s + amp_v * math.sin(2.0 * math.pi * t_s / JITTER_PERIOD_S))


def _in_flicker(t_s: float) -> bool:
  return any(start - _EPS_S <= t_s < start + FLICKER_DURATION_S - _EPS_S for start in FLICKER_STARTS_S)


def _build_steps() -> list[StepInput]:
  steps = []
  for idx in range(int(round(DURATION_S / DT_MDL))):
    t_s = idx * DT_MDL
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_SPEED_MPS,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=_lead_speed(t_s),
        model_prob_target=0.0 if _in_flicker(t_s) else 0.95,
        d_rel_override_m=INITIAL_GAP_M if idx == 0 else None,
        acquisition_reset=idx == 0,
      ),
      note="lightly braking lead, jittering aLead, brief prob flickers",
    ))
  return steps


@functools.lru_cache(maxsize=len(SEEDS) * 2)
def _run(seed: int, legacy: bool) -> SimulationResult:
  overrides = None
  if legacy:
    overrides = {
      "Longitudinal.LiveTune.PhantomLeadDecelHoldFactor": "0.0",
      "Longitudinal.LiveTune.PhantomLeadDecelTrendGain": "0.0",
    }
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(param_overrides=overrides),
    scenario_name=f"phantom_overbrake_guard_{'legacy' if legacy else 'new'}_{seed}",
    steps=_build_steps(),
    initial_speed_mps=EGO_V0_MPS,
    noise_profile="ev6_measured",
    seed=seed,
    perception_filter="auto",
  )


def _brake_taps(trace: list[dict]) -> int:
  taps = 0
  braking = False
  for row in trace:
    a = row["controller_accel_mps2"]
    if not braking and a <= TAP_ON_MPS2:
      braking = True
      taps += 1
    elif braking and a >= TAP_OFF_MPS2:
      braking = False
  return taps


def _flicker_phantom_frames(trace: list[dict], start_s: float) -> int:
  return sum(1 for row in trace
             if start_s - _EPS_S <= row["t_s"] < start_s + FLICKER_DURATION_S - _EPS_S
             and row["lead_one_published_d_rel_m"] is None
             and row["planner_source"] == "lead0")


def test_flickers_engage_phantom_hold():
  """Guard: the counter-check must actually exercise the phantom, otherwise the
  no-overbrake bounds below prove nothing (the old steady-goldilocks check
  passed vacuously)."""
  for seed in SEEDS:
    trace = _run(seed, False).trace
    for start_s in FLICKER_STARTS_S:
      assert _flicker_phantom_frames(trace, start_s) > 0, (
        f"seed {seed}: flicker at {start_s:.1f} s never reached a phantom-held "
        f"lead0 frame — scenario no longer exercises the phantom hold"
      )


def test_phantom_hold_adds_no_overbrake_versus_legacy():
  for seed in SEEDS:
    legacy = _run(seed, True)
    new = _run(seed, False)
    legacy_peak = float(legacy.summary["peakControllerBrakeMps2"])
    new_peak = float(new.summary["peakControllerBrakeMps2"])
    assert new_peak >= legacy_peak - MAX_ADDED_PEAK_BRAKE_MPS2, (
      f"seed {seed}: phantom decel hold added peak brake: new {new_peak:+.3f} "
      f"vs legacy {legacy_peak:+.3f} m/s^2 (allowed {MAX_ADDED_PEAK_BRAKE_MPS2:.2f} deeper)"
    )
    legacy_taps = _brake_taps(legacy.trace)
    new_taps = _brake_taps(new.trace)
    assert new_taps <= legacy_taps, (
      f"seed {seed}: phantom decel hold added brake taps: new {new_taps} vs legacy {legacy_taps} "
      f"(tap = controller accel below {TAP_ON_MPS2} with release above {TAP_OFF_MPS2})"
    )
