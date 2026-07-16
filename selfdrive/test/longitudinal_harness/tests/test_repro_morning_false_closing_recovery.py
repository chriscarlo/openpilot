"""Road replay: noisy same-track freeway lead must shed stale CD9 closure.

This is the exact 20 Hz longitudinal stream from route
``00000241--9a8617cb82`` at 2026-07-16 10:08:54.833-10:08:59.281 PDT. The
synthetic track stayed fixed while raw range had clustered 3-6 m jumps (one
12.64 m reversal), raw aLead stayed within +/-0.13 m/s^2, and raw vRel often
claimed 2-4 m/s closure even though the robust two-second range trend was
opening. CD9 latched 4.18 m/s closure and held it through the contradiction;
the recorded planner/LongControl output stayed near -0.7 m/s^2.

Every row is ``t,dRel,vRel,aLead,vEgo,aEgo,modelProb`` from the rlog. Recorded
ego state is fed to RadarD/planner while the harness separately runs the full
EV6 counterfactual chain: real RadarD -> planner/MPC -> LongControl -> Hyundai
controller -> delay/plant. This is intentionally not a white-noise unit proxy.
"""
from __future__ import annotations

import functools

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls import radard as radard_module
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput


ROAD_TRACE = """
0.000000,56.142327,-1.452606,-0.127135,31.359877,-0.063237,0.987708
0.058068,54.184460,-1.756264,-0.101318,31.340965,-0.180562,0.987336
0.098270,55.292576,-1.962557,-0.111041,31.329460,-0.230933,0.987190
0.144826,49.047440,-1.110966,-0.086329,31.330751,-0.138431,0.987254
0.197736,51.883542,-1.142889,-0.064148,31.324783,-0.124313,0.987273
0.244119,57.277657,-1.036556,-0.047964,31.318808,-0.130950,0.986764
0.303498,56.261567,-0.691977,-0.040433,31.318247,-0.085637,0.986204
0.346279,51.376671,-2.361855,-0.065836,31.306406,-0.146553,0.986391
0.397519,54.582749,-1.033321,-0.024460,31.301550,-0.135477,0.986327
0.443623,59.231076,-0.754049,-0.036551,31.280586,-0.260796,0.986344
0.497223,58.519707,-0.897789,-0.038008,31.242815,-0.457199,0.986475
0.549621,55.248700,-0.867140,-0.042641,31.192125,-0.640054,0.986871
0.594305,56.678170,-1.719244,-0.038569,31.147989,-0.795650,0.987031
0.655255,55.591671,-2.251766,-0.046458,31.094995,-0.817747,0.987539
0.698290,53.859780,-2.688927,-0.083213,31.042768,-0.921632,0.987876
0.745139,54.062462,-2.405798,-0.048233,31.009388,-0.824292,0.988240
0.796505,53.395672,-3.967596,-0.127769,30.977716,-0.740382,0.988387
0.848650,55.325798,-3.239843,-0.079848,30.937624,-0.767333,0.988592
0.901262,54.616528,-2.620541,-0.075118,30.895531,-0.809897,0.988937
0.944414,56.076638,-1.925859,-0.022238,30.872328,-0.660209,0.989028
1.009293,53.752839,-2.000969,-0.020821,30.834597,-0.694430,0.989261
1.055676,56.714074,-1.494253,-0.006862,30.799358,-0.707721,0.989392
1.098749,55.432251,-1.722101,-0.002054,30.763140,-0.718208,0.989547
1.144916,54.106194,-1.331148,-0.001333,30.714037,-0.887078,0.989751
1.196567,53.402668,-1.360985,0.016547,30.658871,-0.908153,0.989919
1.245596,55.147835,-2.492821,-0.023359,30.620827,-0.845885,0.990186
1.299080,53.888337,-2.495991,-0.044520,30.579645,-0.903620,0.990528
1.348156,53.889683,-2.579958,-0.043156,30.536314,-0.824062,0.990798
1.396569,54.077908,-1.969734,-0.024366,30.482924,-0.920748,0.990968
1.448380,57.488152,-1.982048,-0.016969,30.437126,-0.910750,0.990962
1.514421,57.194771,-1.575504,0.015468,30.382856,-0.912983,0.990909
1.549494,57.928128,-2.194424,-0.008366,30.349577,-0.886049,0.990907
1.596925,57.182827,-1.890806,-0.003806,30.300865,-0.916468,0.990887
1.644781,59.562520,-2.696737,0.030492,30.261255,-0.877752,0.990534
1.699349,60.040677,-2.445309,0.027296,30.206467,-0.964930,0.990449
1.742132,58.611020,-3.047461,0.001396,30.160452,-1.024168,0.990428
1.799832,58.245717,-2.831526,0.007258,30.102116,-1.009930,0.990378
1.842041,57.110886,-2.743198,-0.006579,30.072054,-0.928329,0.990515
1.899058,58.254940,-3.087200,-0.003489,30.011095,-0.958797,0.990542
1.946393,60.293797,-2.975309,0.031139,29.976089,-0.934359,0.990596
1.997418,63.168324,-2.396235,0.050239,29.926266,-0.884028,0.990476
2.056978,61.160946,-2.868042,0.020151,29.881290,-0.874523,0.990349
2.104078,59.528798,-2.831474,-0.004273,29.834469,-0.912418,0.990043
2.147419,60.328160,-2.889139,-0.012513,29.796461,-0.841921,0.989845
2.196382,57.281369,-2.954702,-0.034619,29.746517,-0.891146,0.989673
2.245409,56.770977,-2.180820,-0.003338,29.700399,-0.925478,0.989688
2.295233,55.904026,-2.652328,-0.023770,29.668606,-0.872022,0.989908
2.345574,55.705647,-2.408871,0.002055,29.604271,-0.977272,0.990059
2.396481,54.981316,-3.323336,0.017728,29.571337,-0.941443,0.990173
2.447846,54.995686,-2.725761,0.017554,29.513777,-0.940116,0.990288
2.495608,56.984105,-1.489712,0.034882,29.455484,-1.032010,0.990449
2.544613,57.718960,-3.222670,0.066746,29.416914,-1.025551,0.990616
2.599051,55.614193,-2.514482,0.035170,29.348244,-1.068368,0.990819
2.644553,53.118626,-3.272343,0.039442,29.299520,-1.127669,0.990965
2.696948,54.820870,-3.009686,0.009411,29.245983,-1.005920,0.991199
2.744133,56.105282,-2.232248,0.051904,29.202328,-1.043573,0.991246
2.796506,56.330914,-2.508907,0.054419,29.151152,-0.972064,0.991431
2.854569,57.318112,-4.289227,0.067086,29.124928,-0.788225,0.991451
2.897982,59.935269,-2.212269,0.056836,29.121038,-0.530016,0.991470
2.948644,57.647522,-2.556252,0.060993,29.099663,-0.525031,0.991636
2.998822,55.795742,-1.800171,0.041090,29.062614,-0.533028,0.991610
3.044332,58.565987,-1.998198,0.054280,29.035833,-0.595020,0.990752
3.107469,58.935269,-1.815964,-0.009753,29.004089,-0.554623,0.990107
3.154150,61.018780,-2.247065,0.043513,28.931862,-0.879781,0.989571
3.199883,71.356259,-2.630373,0.024559,28.892500,-0.870116,0.988359
3.249326,58.715592,-2.128878,0.037928,28.864294,-0.709816,0.988498
3.296880,56.815648,-1.609325,0.021554,28.835871,-0.677617,0.988634
3.349311,61.837182,-1.796038,0.049815,28.804596,-0.609864,0.988864
3.391338,56.550240,-2.244184,0.034579,28.782755,-0.601680,0.989198
3.445551,56.739132,-1.441526,0.025721,28.756012,-0.524905,0.989230
3.491256,58.037648,-1.723598,0.047297,28.721640,-0.638878,0.989027
3.547045,63.834881,-1.538170,0.023706,28.682756,-0.649980,0.988276
3.594519,63.046017,-2.271255,0.055401,28.659777,-0.625900,0.987868
3.645906,61.172917,-2.346645,0.045902,28.627529,-0.601184,0.987744
3.699974,60.943593,-2.068344,0.036889,28.601555,-0.575812,0.987577
3.743960,59.034981,-2.558701,0.033483,28.555975,-0.713901,0.987790
3.800143,59.460602,-2.263548,0.040516,28.517597,-0.746276,0.987746
3.844084,58.080082,-2.235170,0.024139,28.483330,-0.769222,0.988006
3.898699,57.831830,-2.793734,0.038552,28.449978,-0.736970,0.988218
3.948993,57.685269,-2.029320,0.029242,28.401583,-0.806512,0.988366
3.995282,60.886837,-3.118570,0.038109,28.372393,-0.710865,0.988309
4.047924,59.436032,-2.962807,0.098270,28.338207,-0.695241,0.988290
4.093584,62.992329,-2.034929,0.104312,28.316603,-0.650176,0.988289
4.150572,61.870732,-2.061562,0.069328,28.272467,-0.684268,0.988326
4.203882,61.230591,-2.058367,0.080602,28.238541,-0.693674,0.988553
4.250081,64.455502,-0.563101,0.063615,28.211386,-0.626932,0.988721
4.293745,62.811856,-0.379072,0.059677,28.183329,-0.652173,0.989066
4.352966,61.013527,-0.253691,0.067072,28.153780,-0.582823,0.989562
4.405071,62.490719,0.280172,0.085119,28.118053,-0.619321,0.989878
4.448211,59.178689,-0.099981,0.077267,28.062864,-0.811436,0.990279
"""

ROAD_ROWS = tuple(tuple(float(value) for value in line.split(","))
                  for line in ROAD_TRACE.strip().splitlines())

# Only the seven live values that differed from source defaults in this route.
ROUTE_TUNE = {
  "Longitudinal.LiveTune.LeadSlowdownStrength": "0.18",
  "Longitudinal.LiveTune.HandoffLimitWindowS": "1.0",
  "Longitudinal.LiveTune.HandoffLimitMaxDeltaMps2": "0.12",
  "Longitudinal.LiveTune.ComfortJerkLimitMps3": "0.4",
  "Longitudinal.LiveTune.FlutterClampJerkMps3": "0.06",
  "Longitudinal.LiveTune.OpeningGovernorRawClosingVetoMps": "99.0",
  "Longitudinal.LiveTune.ClosingRecoveryBridgeMaxPositionClosingMps": "1.25",
}

EVENT_START_S = 2.45
RECOVERY_DISABLED_MAX_POSITION_CLOSING_MPS = -1.0
MAX_PUBLISH_EXCESS_MPS = 1.80
P95_PUBLISH_EXCESS_MPS = 1.50
MIN_PLANNER_ACCEL_MPS2 = -0.70
MIN_LONGCONTROL_ACCEL_MPS2 = -0.70
MAX_PLANNER_BRAKE_AREA_MPS = 0.85
MIN_MAX_EXCESS_REDUCTION_MPS = 1.50
MIN_P95_EXCESS_REDUCTION_MPS = 1.70
MIN_BRAKE_AREA_REDUCTION_MPS = 0.35
MIN_PEAK_BRAKE_REDUCTION_MPS2 = 0.10


def _steps() -> list[StepInput]:
  steps = []
  for idx, (t_s, d_rel, v_rel, a_lead, v_ego, a_ego, model_prob) in enumerate(ROAD_ROWS):
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=34.666667,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=v_ego + v_rel,
        model_prob_target=model_prob,
        d_rel_override_m=d_rel if idx == 0 else None,
        measured_d_rel_m=d_rel,
        measured_v_rel_mps=v_rel,
        a_lead_k_mps2=a_lead,
        exact_model_prob=True,
        acquisition_reset=idx == 0,
      ),
      recorded_v_ego_mps=v_ego,
      recorded_a_ego_mps2=a_ego,
      long_active=True,
      personality=1,
      note="route 241 fixed-track 10:08:54.833 PDT raw stream",
    ))
  return steps


@functools.lru_cache(maxsize=2)
def _run(recovery_enabled: bool = True) -> SimulationResult:
  original = radard_module.CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS
  if not recovery_enabled:
    # Behavioral rollback oracle: make the calm-position eligibility predicate
    # impossible without changing RadarD's ordinary filter, CD9 hold, planner,
    # controller, or road inputs.
    radard_module.CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS = (
      RECOVERY_DISABLED_MAX_POSITION_CLOSING_MPS
    )
  try:
    return run_harness(
      vehicle_config=resolve_ev6_vehicle_config(
        livetune_snapshot=None,
        param_overrides=ROUTE_TUNE,
      ),
      scenario_name=f"route_241_morning_false_closing_recovery_{recovery_enabled}",
      steps=_steps(),
      initial_speed_mps=ROAD_ROWS[0][4],
      noise_profile="off",
      seed=42,
      perception_filter="radard",
    )
  finally:
    radard_module.CLOSING_GOVERNOR_RECOVERY_MAX_POSITION_CLOSING_MPS = original


def _event_rows(result: SimulationResult) -> list[dict]:
  # The harness emits five control ticks per model/RadarD input. Do not select
  # one positional offset: LongControl and the EV6 controller keep evolving
  # between planner updates, and the incident oracle owns that complete path.
  return [row for row in result.trace if row["t_s"] >= EVENT_START_S]


def _percentile(values: list[float], quantile: float) -> float:
  ordered = sorted(values)
  return ordered[int(round(quantile * (len(ordered) - 1)))]


def _metrics(result: SimulationResult) -> dict[str, float]:
  rows = _event_rows(result)
  publish_excess = [
    max(0.0, -row["lead_one_published_v_rel_mps"])
    - max(0.0, -row["lead_one_raw_v_rel_mps"])
    for row in rows
  ]
  return {
    "max_publish_excess": max(publish_excess),
    "p95_publish_excess": _percentile(publish_excess, 0.95),
    "min_planner_accel": min(row["planner_accel_mps2"] for row in rows),
    "min_longcontrol_accel": min(row["longcontrol_accel_mps2"] for row in rows),
    "brake_area": sum(max(0.0, -row["planner_accel_mps2"]) * DT_CTRL for row in rows),
  }


def test_morning_false_closing_replay_uses_exact_road_stream_and_full_ev6_path() -> None:
  result = _run()
  rows = result.trace
  assert result.vehicle["candidate"] == "KIA_EV6"
  assert result.vehicle["radarUnavailable"] is True
  assert result.vehicle["resolvedControllerMode"] == "device"
  assert result.vehicle["perceptionFilter"] == "radard"
  assert result.vehicle["noiseProfile"] == "off"
  assert len(rows) == len(ROAD_ROWS) * round(DT_MDL / DT_CTRL)

  raw_drel = [row[1] for row in ROAD_ROWS]
  raw_alead = [row[3] for row in ROAD_ROWS]
  assert max(abs(b - a) for a, b in zip(raw_drel, raw_drel[1:], strict=False)) >= 12.6
  assert max(abs(value) for value in raw_alead) < 0.13

  track_ids = {
    row["lead_one_radard_debug"]["track_id"]
    for row in rows
    if row["lead_one_radard_debug"]["track_id"] is not None
  }
  assert len(track_ids) == 1, f"road stream changed synthetic track identity: {track_ids}"
  assert any(row["lead_one_radard_debug"]["closing_governor_calm_recovery_mode"] for row in rows)
  assert any(abs(row["longcontrol_accel_mps2"] - row["planner_accel_mps2"]) < 1e-6 for row in rows)


def test_morning_false_closing_recovery_sheds_only_stale_cd9_extra_closure() -> None:
  fixed_result = _run(True)
  rollback_result = _run(False)
  rows = _event_rows(fixed_result)
  fixed = _metrics(fixed_result)
  rollback = _metrics(rollback_result)

  physics = "".join((
    "all-frame stale-CD9 metrics: "
    + f"max excess rollback/fixed={rollback['max_publish_excess']:.3f}/{fixed['max_publish_excess']:.3f} m/s; ",
    f"p95 excess={rollback['p95_publish_excess']:.3f}/{fixed['p95_publish_excess']:.3f} m/s; ",
    f"min planner={rollback['min_planner_accel']:.3f}/{fixed['min_planner_accel']:.3f} m/s^2; ",
    f"min LongControl={rollback['min_longcontrol_accel']:.3f}/{fixed['min_longcontrol_accel']:.3f} m/s^2; ",
    f"brake area={rollback['brake_area']:.3f}/{fixed['brake_area']:.3f} m/s",
  ))
  # Broad absolute ceilings keep this a comfort/safety oracle; the paired
  # rollback deltas below prove that it is specifically the stale CD9 excess
  # being removed without pinning a floating-point value at its last decimal.
  assert fixed["max_publish_excess"] <= MAX_PUBLISH_EXCESS_MPS, physics
  assert fixed["p95_publish_excess"] <= P95_PUBLISH_EXCESS_MPS, physics
  assert fixed["min_planner_accel"] >= MIN_PLANNER_ACCEL_MPS2, physics
  assert fixed["min_longcontrol_accel"] >= MIN_LONGCONTROL_ACCEL_MPS2, physics
  assert fixed["brake_area"] <= MAX_PLANNER_BRAKE_AREA_MPS, physics
  assert rollback["max_publish_excess"] - fixed["max_publish_excess"] >= MIN_MAX_EXCESS_REDUCTION_MPS, physics
  assert rollback["p95_publish_excess"] - fixed["p95_publish_excess"] >= MIN_P95_EXCESS_REDUCTION_MPS, physics
  assert rollback["brake_area"] - fixed["brake_area"] >= MIN_BRAKE_AREA_REDUCTION_MPS, physics
  assert fixed["min_planner_accel"] - rollback["min_planner_accel"] >= MIN_PEAK_BRAKE_REDUCTION_MPS2, physics

  # While recovery is active, any >=2.5 m/s current raw close must remain fully
  # represented by CD9's working clamp; recovery may remove stale EXTRA closure,
  # never hide the current measurement behind the ordinary vRel EMA.
  fast_recovery_rows = [
    row for row in rows
    if row["lead_one_radard_debug"]["closing_governor_calm_recovery_mode"]
    and max(0.0, -row["lead_one_raw_v_rel_mps"]) >= 2.5
  ]
  assert fast_recovery_rows
  for row in fast_recovery_rows:
    assert row["lead_one_radard_debug"]["closing_governor_closing_mps"] >= (
      max(0.0, -row["lead_one_raw_v_rel_mps"]) - 1e-6
    )
