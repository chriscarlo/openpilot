"""Repro: sparse/jagged range proof lets a mild aLead report become a hard tap.

Route anchor: ``00000240--3ec16764ac`` at 2026-07-16 09:05:22.995 PDT,
snapshot ``false_closing_000093895``.  The exact 20 Hz scalar sequence below is
the 2.14 s immediately preceding and spanning the tap.  It is intentionally
fed through real RadarD, LongitudinalMpc, LongControl, and the EV6 device
controller; no planner/RadarState oracle is injected.

The road inputs disagree in the precise way that exposed the bug:

* published model aLeadK is only about -0.11/-0.12 m/s^2;
* same-slot raw range is jagged (including a 4.45 m single-frame step), so the
  independent position window repeatedly rebuilds and is only ~0.60 s old;
* the private published-vLead derivative nevertheless settles at
  -1.15..-1.45 m/s^2 and previously authorized CD3 extra amplification;
* that amplified value drove the lead-slowdown ceiling and lead-decel comfort
  bypass from a mild response to a -1.0 m/s^2 planner tap.

The fixed guard does not pretend the surrounding false-closing signal is calm;
it only refuses to turn a mild, outside-target model brake into a much deeper
one while the independent position proof is unavailable.  A separate CD9
regression owns the surrounding false-closing recovery.
"""
from __future__ import annotations

import functools

from cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls import radard as radard_module
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as long_mpc_module
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput


# (t, raw dRel, raw vRel, raw aLead, model probability, dPath, vLat, vEgo, aEgo)
# copied from the exact route bundle, normalized to t=0 at 09:05:20.998 PDT.
ROAD_SAMPLES = (
  (0.000000000, 62.919186096, -1.249370575, -0.125446916, 0.993539989, 0.025914491, -0.894861472, 31.337675095, -0.153666139),
  (0.050047916, 60.963684540, -1.267248154, -0.128198236, 0.994769990, -0.040763968, -0.775278448, 31.346029282, -0.036419120),
  (0.096042708, 61.230751495, -1.775331497, -0.151186585, 0.993648946, 0.006478479, -0.791888493, 31.349714279, 0.021029575),
  (0.146839062, 62.562267761, -0.805467606, -0.149411500, 0.992012322, -0.010164818, -1.009597456, 31.338045120, -0.065964937),
  (0.192546354, 63.700596313, -1.285610199, -0.119374886, 0.986694217, 0.067366927, -1.188140132, 31.331264496, -0.093278572),
  (0.245379479, 64.978962402, -1.591093063, -0.148062125, 0.986167610, 0.059261336, -1.428874246, 31.314403534, -0.202625707),
  (0.298622343, 62.761364441, -1.575206757, -0.125057325, 0.987774730, -0.030851388, -1.320712961, 31.313817978, -0.135537222),
  (0.341741823, 61.897621613, -0.947116852, -0.112619899, 0.988525569, 0.028161264, -1.284819741, 31.314283371, -0.071382746),
  (0.395926614, 60.983845215, -1.323953629, -0.111695468, 0.989233971, 0.027729803, -1.271125377, 31.309717178, -0.088815056),
  (0.460272916, 61.322845917, -0.858402252, -0.098449305, 0.990427971, 0.045352428, -1.664915787, 31.316003799, -0.008022296),
  (0.504952083, 59.700317841, -0.275642395, -0.100430034, 0.991815746, 0.095667120, -1.641467280, 31.315351486, -0.019796642),
  (0.546735000, 60.806934814, -0.020812988, -0.105319619, 0.991672814, -0.017418467, -1.813319402, 31.316951752, 0.004810654),
  (0.600620729, 60.812530975, -0.532833099, -0.125344843, 0.990232825, 0.024515259, -1.866213754, 31.312725067, -0.038257893),
  (0.645176458, 62.245205383, -0.475444794, -0.124418274, 0.986291587, 0.096251053, -1.966910178, 31.302309036, -0.108710863),
  (0.692953697, 61.849319916, -0.541477203, -0.105913199, 0.989046514, -0.019715308, -2.240917908, 31.306249619, -0.038365044),
  (0.744411249, 57.916756134, -0.353391647, -0.126353875, 0.993803144, 0.083461116, -1.976292519, 31.298379898, -0.060373165),
  (0.803140885, 59.007576447, -0.854490280, -0.115729637, 0.993092060, 0.110970752, -2.117391019, 31.286705017, -0.097882688),
  (0.853081249, 59.759205322, -1.014728546, -0.110130325, 0.990874767, 0.066445299, -2.183857040, 31.296138763, 0.017805040),
  (0.899364583, 59.211365204, -0.393344879, -0.080815032, 0.990885437, 0.037806772, -2.396827350, 31.288101196, -0.052707348),
  (0.945793958, 60.338604431, 0.027475357, -0.092321359, 0.989407003, -0.057248210, -2.687980110, 31.290506363, -0.023014145),
  (1.000384479, 60.655987244, -0.891061783, -0.128786579, 0.985416114, -0.110359029, -2.684529467, 31.275278091, -0.132247016),
  (1.044706770, 63.252552490, 0.182092667, -0.123904243, 0.979149282, -0.088534345, -3.225978338, 31.275934219, -0.076364636),
  (1.096630989, 61.240726929, -0.535684586, -0.145164356, 0.980383337, -0.144565613, -3.223455193, 31.286928177, 0.045216966),
  (1.144653645, 59.714672546, -0.643451691, -0.133834943, 0.983852088, -0.103802364, -3.239412867, 31.289037704, 0.035307791),
  (1.195995312, 62.010418396, 0.073305130, -0.131754458, 0.985545695, -0.256009489, -3.743369499, 31.289234161, 0.035423230),
  (1.251913489, 61.884739380, -0.837520599, -0.119639918, 0.981393039, -0.274132686, -3.703534885, 31.281320572, -0.027968857),
  (1.303885364, 61.757580261, -0.585296631, -0.131835327, 0.968295217, -0.248364490, -3.813693452, 31.280960083, -0.008604890),
  (1.345437499, 57.304722290, -1.140884399, -0.122695386, 0.972979605, -0.133837169, -3.652722040, 31.284837723, 0.035531912),
  (1.393730312, 59.859081726, -0.221090317, -0.098575786, 0.967628002, -0.293051459, -4.414814515, 31.285146713, 0.033850390),
  (1.445295155, 58.083870392, -0.956081390, -0.105024122, 0.963925838, -0.241744711, -4.141564790, 31.291648865, 0.058322359),
  (1.506568489, 55.358513336, -0.630496979, -0.107291281, 0.976510644, -0.231002503, -4.153080565, 31.301691055, 0.111359850),
  (1.553276145, 51.582714539, -1.127429962, -0.122255370, 0.984122157, -0.074466444, -3.878910243, 31.308076859, 0.128771722),
  (1.599880780, 52.670246582, -1.411924362, -0.142593533, 0.980398715, -0.061795374, -4.297258058, 31.317249298, 0.164891899),
  (1.641993541, 54.717606049, -0.853847504, -0.129169777, 0.982343316, -0.144701063, -4.834143351, 31.326263428, 0.187770724),
  (1.692187343, 58.522705536, -1.455595016, -0.119364604, 0.960462809, 0.127820840, -5.206947143, 31.336650848, 0.180183724),
  (1.744088541, 56.387970428, -0.985107422, -0.136629239, 0.977151453, -0.028263015, -5.071805813, 31.346815109, 0.203459799),
  (1.799432030, 56.149120789, -1.065513611, -0.120124720, 0.974195361, 0.049353237, -5.196390863, 31.361356735, 0.221547112),
  (1.845681197, 55.635597687, -1.859737396, -0.116023384, 0.965503156, 0.167159741, -5.294653240, 31.367509842, 0.200142652),
  (1.899655624, 54.029861908, -1.613557816, -0.105832309, 0.975334942, 0.118341124, -5.554557586, 31.381610870, 0.209344938),
  (1.945928697, 53.835625153, -1.618503571, -0.093966939, 0.983349800, -0.010903718, -5.714597942, 31.382265091, 0.113988586),
  (1.996557811, 54.281811218, -1.529125214, -0.091121882, 0.974867404, 0.050314623, -5.898404738, 31.396301270, 0.171444699),
  (2.043715416, 54.972618561, -1.424766541, -0.089723162, 0.968417883, -0.106273781, -6.404943394, 31.404081345, 0.172108173),
  (2.098317343, 51.714333038, -2.030788422, -0.091090091, 0.978468716, -0.004312172, -5.892876283, 31.403997421, 0.110150844),
  (2.142093801, 53.689499359, -1.707452774, -0.077084504, 0.976173103, -0.070861089, -6.384744604, 31.400756836, 0.031570788),
)

EVENT_START_S = 1.55
EVENT_END_S = 2.15
SHIPPED_CONFIRMATION_BOUND_MPS2 = 0.20
ROLLBACK_CONFIRMATION_BOUND_MPS2 = 0.0

BRAKING_TWIN_ONSET_S = 1.50
BRAKING_TWIN_DURATION_S = 4.50
BRAKING_TWIN_EGO_SPEED_MPS = 25.0
BRAKING_TWIN_INITIAL_GAP_M = 35.0
BRAKING_TWIN_RANGE_NOISE_M = (0.0, 1.8, -1.3, 2.5, -2.0, 1.2, -1.0)
RAW_HARD_BRAKE_BOUNDARY_MPS2 = radard_module.ACCEL_CORR_CALM_POSITION_RAW_ALEAD_HARD_VETO_MPS2
BRAKING_TWIN_BELOW_BOUNDARY_MPS2 = -(RAW_HARD_BRAKE_BOUNDARY_MPS2 - 0.001)
BRAKING_TWIN_ABOVE_BOUNDARY_MPS2 = -(RAW_HARD_BRAKE_BOUNDARY_MPS2 + 0.001)


def _build_steps() -> list[StepInput]:
  steps: list[StepInput] = []
  for idx, (t_s, d_rel, v_rel, a_lead, prob, d_path, v_lat, v_ego, a_ego) in enumerate(ROAD_SAMPLES):
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=29.444444656,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=v_ego + v_rel,
        model_prob_target=prob,
        measured_d_rel_m=d_rel,
        measured_v_rel_mps=v_rel,
        a_lead_k_mps2=a_lead,
        d_path_m=d_path,
        v_lat_mps=v_lat,
        exact_model_prob=True,
        acquisition_reset=idx == 0,
      ),
      recorded_v_ego_mps=v_ego,
      recorded_a_ego_mps2=a_ego,
      long_active=True,
      personality=1,
      note="exact 2026-07-16 09:05:20.998-09:05:23.140 PDT road vector",
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _run(unconfirmed_bound_mps2: float) -> SimulationResult:
  original = long_mpc_module.LEAD_ACCEL_CORR_UNCONFIRMED_MAX_ALEAD_ABS_MPS2
  long_mpc_module.LEAD_ACCEL_CORR_UNCONFIRMED_MAX_ALEAD_ABS_MPS2 = unconfirmed_bound_mps2
  try:
    return run_harness(
      vehicle_config=resolve_ev6_vehicle_config(param_overrides={
        "Longitudinal.LiveTune.LeadAccelCorrAmplifyGain": "1.0",
        "Longitudinal.LiveTune.LeadAccelCorrAmplifyModelDecelMinMps2": "0.10",
      }),
      scenario_name=f"20260716_sparse_alead_confirm_{unconfirmed_bound_mps2:g}",
      steps=_build_steps(),
      initial_speed_mps=ROAD_SAMPLES[0][7],
      initial_accel_mps2=ROAD_SAMPLES[0][8],
      noise_profile="off",
      seed=42,
      perception_filter="radard",
      ego_replay_mode="recorded",
    )
  finally:
    long_mpc_module.LEAD_ACCEL_CORR_UNCONFIRMED_MAX_ALEAD_ABS_MPS2 = original


def _event_rows(result: SimulationResult) -> list[dict]:
  return [row for row in result.trace[::5] if EVENT_START_S <= row["t_s"] <= EVENT_END_S]


def _build_genuine_braking_steps(raw_a_lead_mps2: float) -> list[StepInput]:
  steps: list[StepInput] = []
  for idx in range(round(BRAKING_TWIN_DURATION_S / DT_MDL)):
    t_s = idx * DT_MDL
    braking_age_s = max(0.0, t_s - BRAKING_TWIN_ONSET_S)
    v_rel_mps = raw_a_lead_mps2 * braking_age_s
    physical_d_rel_m = BRAKING_TWIN_INITIAL_GAP_M + 0.5 * raw_a_lead_mps2 * braking_age_s ** 2
    measured_d_rel_m = physical_d_rel_m + BRAKING_TWIN_RANGE_NOISE_M[idx % len(BRAKING_TWIN_RANGE_NOISE_M)]
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=30.0,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=BRAKING_TWIN_EGO_SPEED_MPS + v_rel_mps,
        d_rel_override_m=measured_d_rel_m if idx == 0 else None,
        measured_d_rel_m=measured_d_rel_m,
        measured_v_rel_mps=v_rel_mps,
        a_lead_k_mps2=0.0 if t_s < BRAKING_TWIN_ONSET_S else raw_a_lead_mps2,
        model_prob_target=0.99,
        exact_model_prob=True,
        acquisition_reset=idx == 0,
      ),
      recorded_v_ego_mps=BRAKING_TWIN_EGO_SPEED_MPS,
      recorded_a_ego_mps2=0.0,
      long_active=True,
      personality=1,
      note="jagged-range genuine lead brake straddling RadarD raw-aLead boundary",
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _run_genuine_braking_twin(raw_a_lead_mps2: float) -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(param_overrides={
      "Longitudinal.LiveTune.LeadAccelCorrAmplifyGain": "1.0",
      "Longitudinal.LiveTune.LeadAccelCorrAmplifyModelDecelMinMps2": "0.10",
    }),
    scenario_name=f"jagged_genuine_brake_{raw_a_lead_mps2:+.3f}",
    steps=_build_genuine_braking_steps(raw_a_lead_mps2),
    initial_speed_mps=BRAKING_TWIN_EGO_SPEED_MPS,
    noise_profile="off",
    seed=42,
    perception_filter="radard",
    ego_replay_mode="recorded",
  )


def test_exact_road_vector_uses_full_ev6_radard_path_and_is_sparse_jagged() -> None:
  result = _run(SHIPPED_CONFIRMATION_BOUND_MPS2)
  rows = _event_rows(result)
  raw_gaps = [row["lead_one_raw_d_rel_m"] for row in result.trace[::5]]

  assert result.vehicle["candidate"] == "KIA_EV6"
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["egoReplayMode"] == "recorded"
  assert result.vehicle["noiseProfile"] == "off"
  assert max(abs(b - a) for a, b in zip(raw_gaps, raw_gaps[1:], strict=False)) >= 4.4
  assert all(abs(row["lead_one_raw_a_lead_k_mps2"]) <= 0.16 for row in rows)
  assert all(row["lead_one_radard_debug"]["accel_corr_calm_position_reason"] == "sparse_window" for row in rows)
  assert max(row["lead_one_radard_debug"]["accel_corr_calm_position_window_span_s"] for row in rows) < 0.8
  assert not any(row["mpc_lead_stability_debug"]["slot0"]["accel_corr_raw_hard_braking"] for row in rows)


def test_sparse_position_guard_removes_only_the_alead_amplifier_tap() -> None:
  fixed = _event_rows(_run(SHIPPED_CONFIRMATION_BOUND_MPS2))
  rollback = _event_rows(_run(ROLLBACK_CONFIRMATION_BOUND_MPS2))
  fixed_veto = [row for row in fixed if row["mpc_lead_stability_debug"]["slot0"]["accel_corr_amplify_vetoed"]]
  rollback_amp = [row for row in rollback if row["mpc_lead_stability_debug"]["slot0"]["accel_corr_amplified"]]

  assert fixed_veto
  assert all(
    row["mpc_lead_stability_debug"]["slot0"]["accel_corr_amplify_veto_reason"] ==
    "unconfirmed_mild_outside_target"
    for row in fixed_veto
  )
  assert not any(row["mpc_lead_stability_debug"]["slot0"]["accel_corr_amplified"] for row in fixed)
  # Restored RadarD governor hardening changes the upstream aLead filter
  # trajectory for this exact road vector. The shared -1.20 m/s^2 signal is
  # still large enough to prove this test is isolating the planner amplifier.
  assert min(row["mpc_lead_stability_debug"]["slot0"]["accel_corr_a_meas_lp"] for row in fixed) < -1.1
  assert max(row["mpc_acc_source_debug"]["approach_reacquire_lead_decel_mps2"] for row in fixed) < 0.16
  fixed_min_accel = min(row["planner_accel_mps2"] for row in fixed)
  rollback_min_accel = min(row["planner_accel_mps2"] for row in rollback)
  # The surrounding CD9 false-closing hold is deliberately left visible in
  # this isolated fix.  What disappears here is the additional hard tap caused
  # by the correlation amplifier.
  assert fixed_min_accel > -0.45

  assert rollback_amp
  assert min(row["mpc_lead_stability_debug"]["slot0"]["accel_corr_a_meas_lp"] for row in rollback) < -1.1
  assert max(row["mpc_acc_source_debug"]["approach_reacquire_lead_decel_mps2"] for row in rollback) > 1.1
  assert rollback_min_accel <= -0.70
  assert fixed_min_accel - rollback_min_accel >= 0.30


def test_jagged_range_genuine_braking_straddles_raw_alead_boundary_without_delaying_response() -> None:
  below = _run_genuine_braking_twin(BRAKING_TWIN_BELOW_BOUNDARY_MPS2)
  above = _run_genuine_braking_twin(BRAKING_TWIN_ABOVE_BOUNDARY_MPS2)

  assert "accelCorrRawHardBraking" in log.RadarState.LeadData.schema.fields
  for result in (below, above):
    assert result.vehicle["candidate"] == "KIA_EV6"
    assert result.vehicle["resolvedControllerMode"] == "device"
    assert result.vehicle["perceptionFilter"] == "radard"
    assert result.vehicle["egoReplayMode"] == "recorded"

  before_below = [row for row in below.trace if BRAKING_TWIN_ONSET_S - 0.10 <= row["t_s"] < BRAKING_TWIN_ONSET_S]
  before_above = [row for row in above.trace if BRAKING_TWIN_ONSET_S - 0.10 <= row["t_s"] < BRAKING_TWIN_ONSET_S]
  event_below = [row for row in below.trace if BRAKING_TWIN_ONSET_S <= row["t_s"] <= BRAKING_TWIN_ONSET_S + 0.75]
  event_above = [row for row in above.trace if BRAKING_TWIN_ONSET_S <= row["t_s"] <= BRAKING_TWIN_ONSET_S + 0.75]
  assert before_below and before_above and event_below and event_above

  # The measured range is deliberately too jagged to manufacture a calm
  # position proof. The true lead signal is independently coherent: vRel ramps
  # at the asserted raw aLead on both sides of the fixed -0.40 m/s^2 boundary.
  raw_drel = [row["lead_one_raw_d_rel_m"] for row in below.trace]
  assert max(abs(b - a) for a, b in zip(raw_drel, raw_drel[1:], strict=False)) >= 4.4
  assert not any(
    row["mpc_lead_stability_debug"]["slot0"]["accel_corr_calm_position_valid"]
    for row in event_below + event_above
  )

  # This checks the producer -> Cap'n Proto RadarState -> MPC consumer chain,
  # not a directly fabricated planner lead. Every post-onset raw sample below
  # the threshold keeps the bit clear; every sample just beyond it asserts it.
  assert all(
    abs(row["lead_one_raw_a_lead_k_mps2"] - BRAKING_TWIN_BELOW_BOUNDARY_MPS2) < 1e-6
    for row in event_below
  )
  assert all(
    abs(row["lead_one_raw_a_lead_k_mps2"] - BRAKING_TWIN_ABOVE_BOUNDARY_MPS2) < 1e-6
    for row in event_above
  )
  assert all(
    not row["mpc_lead_stability_debug"]["slot0"]["accel_corr_raw_hard_braking"]
    for row in event_below
  )
  assert all(
    row["mpc_lead_stability_debug"]["slot0"]["accel_corr_raw_hard_braking"]
    for row in event_above
  )
  assert not any(
    row["mpc_lead_stability_debug"]["slot0"]["accel_corr_amplify_vetoed"]
    for row in event_below + event_above
  )

  baseline_below = min(row["planner_accel_mps2"] for row in before_below)
  baseline_above = min(row["planner_accel_mps2"] for row in before_above)
  onset_below = next(
    row for row in event_below if row["planner_accel_mps2"] <= baseline_below - 0.04
  )
  onset_above = next(
    row for row in event_above if row["planner_accel_mps2"] <= baseline_above - 0.04
  )
  min_planner_below = min(row["planner_accel_mps2"] for row in event_below)
  min_planner_above = min(row["planner_accel_mps2"] for row in event_above)
  min_longcontrol_below = min(row["longcontrol_accel_mps2"] for row in event_below)
  min_longcontrol_above = min(row["longcontrol_accel_mps2"] for row in event_above)
  min_controller_below = min(row["controller_accel_mps2"] for row in event_below)
  min_controller_above = min(row["controller_accel_mps2"] for row in event_above)
  physics = "".join((
    f"boundary={RAW_HARD_BRAKE_BOUNDARY_MPS2:.3f}; ",
    f"planner onset below/above={onset_below['t_s'] - BRAKING_TWIN_ONSET_S:.3f}/",
    f"{onset_above['t_s'] - BRAKING_TWIN_ONSET_S:.3f}s; ",
    f"planner minima={min_planner_below:.3f}/{min_planner_above:.3f}; ",
    f"LongControl minima={min_longcontrol_below:.3f}/{min_longcontrol_above:.3f}; ",
    f"EV6 controller minima={min_controller_below:.3f}/{min_controller_above:.3f}",
  ))
  assert onset_below["t_s"] - BRAKING_TWIN_ONSET_S <= 0.35, physics
  assert onset_above["t_s"] - BRAKING_TWIN_ONSET_S <= 0.35, physics
  assert min_planner_below <= baseline_below - 0.08, physics
  assert min_planner_above <= baseline_above - 0.08, physics
  assert min_longcontrol_below <= baseline_below - 0.08, physics
  assert min_longcontrol_above <= baseline_above - 0.08, physics
  assert min_controller_below <= -0.05, physics
  assert min_controller_above <= -0.05, physics

  # Crossing the evidence bit must not create a response cliff. Both sides are
  # already materially stronger than the mild-brake guard and must preserve the
  # same prompt planner/LongControl/EV6-controller behavior.
  assert abs(onset_below["t_s"] - onset_above["t_s"]) <= DT_MDL, physics
  assert abs(min_planner_below - min_planner_above) <= 0.02, physics
  assert abs(min_longcontrol_below - min_longcontrol_above) <= 0.02, physics
  assert abs(min_controller_below - min_controller_above) <= 0.02, physics
