from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import open_catalog, record_snapshot_bundle, record_sweep_result
from .closed_loop import SimulationResult, run_harness
from .config import FRIENDLY_PARAM_NAMES, NOISE_PROFILES, NoiseSeeds, resolve_ev6_vehicle_config
from .inputs import SCENARIO_NAMES, build_synthetic_scenario, load_snapshot_bundle


DEFAULT_SCORE_WEIGHTS = {
  "excessBrakeMps2": 2.0,
  "maxFollowOvershootMps": 3.0,
  "leadEventOvershootGrowthMps": 3.0,
  "handoffPrerevealMeanOvershootMps": 0.0,
  "handoffPrerevealSpeedLossMps": 0.0,
  "handoffPrerevealCruiseFraction": 0.0,
  "maxFollowUndershootMps": 2.5,
  "reclaimDelayS": 1.5,
  "leadAcquireLatencyS": 1.5,
  "handoffBrakeLatencyS": 1.5,
  "longControlRealizedDivergenceMps2": 1.0,
}


@dataclass(frozen=True)
class SweepCandidate:
  overrides: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run EV6 longitudinal harness parameter sweeps")
  parser.add_argument("--mode", choices=("synthetic-ev6", "snapshot"), default="synthetic-ev6")
  parser.add_argument("--scenario", action="append", choices=SCENARIO_NAMES, default=None)
  parser.add_argument("--all-scenarios", action="store_true")
  parser.add_argument("--snapshot", type=Path, default=None, help="Snapshot bundle directory for --mode snapshot")
  parser.add_argument("--topology", choices=("lka", "lfa"), default="lfa")
  parser.add_argument("--controller-mode", choices=("auto", "passthrough", "shaped"), default="auto")
  parser.add_argument("--hyundai-tuning-mode", choices=("off", "dynamic", "predictive"), default=None)
  parser.add_argument("--noise", choices=tuple(NOISE_PROFILES.keys()), default="realistic")
  parser.add_argument("--duration", type=float, default=12.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--seed-drel", type=int, default=None)
  parser.add_argument("--seed-vrel", type=int, default=None)
  parser.add_argument("--seed-aego", type=int, default=None)
  parser.add_argument("--seed-vego", type=int, default=None)
  parser.add_argument("--override", action="append", default=[], help="Base override, either friendly_name=value or ParamKey=value")
  parser.add_argument("--grid", action="append", default=[], help="Sweep grid entry: friendly_name=v1,v2,v3")
  parser.add_argument("--score-weight", action="append", default=[], help="Override score weights: metric=weight")
  parser.add_argument("--top-k", type=int, default=5)
  parser.add_argument("--output-json", type=Path, default=None)
  parser.add_argument("--output-csv", type=Path, default=None)
  parser.add_argument("--record-to-db", action="store_true", help="Record this sweep in the SQLite catalog")
  parser.add_argument("--db-path", type=Path, default=None)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  scenarios = _resolve_scenarios(args.scenario, args.all_scenarios)
  base_overrides = _parse_overrides(args.override)
  grid_overrides = _parse_grid(args.grid)
  score_weights = _parse_score_weights(args.score_weight)
  hyundai_tuning_mode = _parse_hyundai_tuning_mode(args.hyundai_tuning_mode)
  noise_seeds = _parse_noise_seeds(args)

  candidates = enumerate_candidates(base_overrides, grid_overrides)
  payload = run_sweep(
    candidates=candidates,
    scenarios=scenarios,
    mode=args.mode,
    snapshot=args.snapshot,
    topology=args.topology,
    controller_mode=args.controller_mode,
    hyundai_tuning_mode=hyundai_tuning_mode,
    noise=args.noise,
    duration_s=args.duration,
    seed=args.seed,
    noise_seeds=noise_seeds,
    score_weights=score_weights,
  )
  if args.top_k > 0:
    payload["rankedCandidates"] = payload["rankedCandidates"][:args.top_k]

  if args.record_to_db:
    conn = open_catalog(args.db_path)
    try:
      snapshot_id = None
      if args.mode == "snapshot" and args.snapshot is not None:
        snapshot_id = record_snapshot_bundle(conn, args.snapshot)
      sweep_id = record_sweep_result(conn, payload, snapshot_id=snapshot_id, seed=args.seed)
      conn.commit()
      payload["catalogSweepId"] = sweep_id
      payload["catalogSnapshotId"] = snapshot_id
    finally:
      conn.close()
  print(json.dumps(payload, indent=2, sort_keys=True))
  if args.output_json:
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
  if args.output_csv:
    write_ranked_csv(payload["rankedCandidates"], args.output_csv)
  return 0


def enumerate_candidates(base_overrides: dict[str, str], grid_overrides: dict[str, list[str]]) -> list[SweepCandidate]:
  if not grid_overrides:
    return [SweepCandidate(dict(base_overrides))]

  keys = list(grid_overrides)
  values = [grid_overrides[key] for key in keys]
  candidates = []
  for combo in itertools.product(*values):
    overrides = dict(base_overrides)
    overrides.update(dict(zip(keys, combo, strict=True)))
    candidates.append(SweepCandidate(overrides))
  return candidates


def run_sweep(*,
              candidates: list[SweepCandidate],
              scenarios: list[str],
              mode: str,
              snapshot: Path | None,
              topology: str,
              controller_mode: str,
              hyundai_tuning_mode: int | None,
              noise: str,
              duration_s: float,
              seed: int,
              noise_seeds: NoiseSeeds | None = None,
              score_weights: dict[str, float]) -> dict[str, Any]:
  ranked_candidates = []
  bundle = load_snapshot_bundle(snapshot) if mode == "snapshot" and snapshot is not None else None

  if mode == "snapshot":
    if bundle is None:
      raise ValueError("--snapshot is required with --mode snapshot")
    scenario_inputs = [(bundle.name or snapshot.name, bundle.initial_speed_mps, bundle.initial_accel_mps2, bundle.timeline)]
    scenario_names = [bundle.name or snapshot.name]
  else:
    scenario_inputs = []
    for scenario_name in scenarios:
      initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario(scenario_name, duration_s=duration_s, dt_s=0.05)
      scenario_inputs.append((scenario_name, initial_speed_mps, initial_accel_mps2, steps))
    scenario_names = list(scenarios)

  scenario_noise_seeds = {
    scenario_name: (noise_seeds or NoiseSeeds.from_base(seed)).offset(scenario_offset * 1000)
    for scenario_offset, scenario_name in enumerate(scenario_names)
  }
  scenario_seeds = {
    scenario_name: seed + scenario_offset
    for scenario_offset, scenario_name in enumerate(scenario_names)
  }

  for candidate in candidates:
    scenario_summaries: dict[str, dict[str, Any]] = {}
    vehicle_description = None
    aggregate_penalties = {metric: 0.0 for metric in score_weights}

    for scenario_offset, (scenario_name, initial_speed_mps, initial_accel_mps2, steps) in enumerate(scenario_inputs):
      if mode == "snapshot":
        vehicle_config = resolve_ev6_vehicle_config(
          topology=str(bundle.vehicle.get("topology", topology)),
          controller_mode=controller_mode,
          tune_source="snapshot",
          param_overrides=candidate.overrides,
          hyundai_tuning_mode=hyundai_tuning_mode,
          snapshot_vehicle=bundle.vehicle,
          snapshot_params=bundle.params,
        )
      else:
        vehicle_config = resolve_ev6_vehicle_config(
          topology=topology,
          controller_mode=controller_mode,
          tune_source="cli" if candidate.overrides else "defaults",
          param_overrides=candidate.overrides,
          hyundai_tuning_mode=hyundai_tuning_mode,
        )

      result = run_harness(
        vehicle_config=vehicle_config,
        scenario_name=scenario_name,
        steps=steps,
        initial_speed_mps=initial_speed_mps,
        initial_accel_mps2=initial_accel_mps2,
        noise_profile=noise,
        seed=scenario_seeds[scenario_name],
        noise_seeds=scenario_noise_seeds[scenario_name],
      )
      vehicle_description = result.vehicle
      scenario_summaries[scenario_name] = result.summary
      penalties = score_summary(result.summary)
      for metric_name, metric_weight in score_weights.items():
        aggregate_penalties[metric_name] += metric_weight * penalties[metric_name]

    total_score = float(sum(aggregate_penalties.values()))
    ranked_candidates.append({
      "score": total_score,
      "weightedPenalties": {key: round(value, 6) for key, value in aggregate_penalties.items()},
      "overrides": candidate.overrides,
      "scenarioSummaries": scenario_summaries,
      "vehicle": vehicle_description,
    })

  ranked_candidates.sort(key=lambda entry: entry["score"])
  for rank, entry in enumerate(ranked_candidates, start=1):
    entry["rank"] = rank

  return {
    "mode": mode,
    "noiseProfile": noise,
    "scenarioNames": scenario_names,
    "scoreWeights": score_weights,
    "noiseSeeds": (noise_seeds or NoiseSeeds.from_base(seed)).as_dict(),
    "candidateCount": len(candidates),
    "rankedCandidates": ranked_candidates,
  }


def score_summary(summary: dict[str, Any]) -> dict[str, float]:
  peak_controller_brake = float(summary.get("peakControllerBrakeMps2") or 0.0)
  return {
    "excessBrakeMps2": max(0.0, abs(min(peak_controller_brake, 0.0)) - 2.5),
    "maxFollowOvershootMps": float(summary.get("maxFollowOvershootMps") or 0.0),
    "leadEventOvershootGrowthMps": float(summary.get("leadEventOvershootGrowthMps") or 0.0),
    "handoffPrerevealMeanOvershootMps": float(summary.get("handoffPrerevealMeanOvershootMps") or 0.0),
    "handoffPrerevealSpeedLossMps": float(summary.get("handoffPrerevealSpeedLossMps") or 0.0),
    "handoffPrerevealCruiseFraction": float(summary.get("handoffPrerevealCruiseFraction") or 0.0),
    "maxFollowUndershootMps": float(summary.get("maxFollowUndershootMps") or 0.0),
    "reclaimDelayS": float(summary.get("reclaimDelayS") or 0.0),
    "leadAcquireLatencyS": float(summary.get("leadAcquireLatencyS") or 0.0),
    "handoffBrakeLatencyS": float(summary.get("handoffBrakeLatencyS") or 0.0),
    "longControlRealizedDivergenceMps2": float(summary.get("longControlRealizedDivergenceMps2") or 0.0),
  }


def write_ranked_csv(ranked_candidates: list[dict[str, Any]], output_path: Path) -> None:
  fieldnames = ["rank", "score", "overrides", "weightedPenalties"]
  with output_path.open("w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for entry in ranked_candidates:
      writer.writerow({
        "rank": entry["rank"],
        "score": entry["score"],
        "overrides": json.dumps(entry["overrides"], sort_keys=True),
        "weightedPenalties": json.dumps(entry["weightedPenalties"], sort_keys=True),
      })


def _resolve_scenarios(raw_scenarios: list[str] | None, all_scenarios: bool) -> list[str]:
  if all_scenarios:
    return list(SCENARIO_NAMES)
  if raw_scenarios:
    return raw_scenarios
  return ["approach", "pullaway", "handoff", "cutin"]


def _parse_hyundai_tuning_mode(raw: str | None) -> int | None:
  if raw is None:
    return None
  mapping = {"off": 0, "dynamic": 1, "predictive": 2}
  return mapping[raw]


def _parse_overrides(items: list[str]) -> dict[str, str]:
  overrides: dict[str, str] = {}
  for item in items:
    if "=" not in item:
      raise SystemExit(f"invalid override '{item}', expected key=value")
    key, value = item.split("=", 1)
    overrides[FRIENDLY_PARAM_NAMES.get(key.strip(), key.strip())] = value.strip()
  return overrides


def _parse_grid(items: list[str]) -> dict[str, list[str]]:
  grid: dict[str, list[str]] = {}
  for item in items:
    if "=" not in item:
      raise SystemExit(f"invalid grid entry '{item}', expected key=v1,v2")
    key, values = item.split("=", 1)
    normalized_key = FRIENDLY_PARAM_NAMES.get(key.strip(), key.strip())
    parsed_values = [value.strip() for value in values.split(",") if value.strip()]
    if not parsed_values:
      raise SystemExit(f"invalid grid entry '{item}', expected at least one value")
    grid[normalized_key] = parsed_values
  return grid


def _parse_score_weights(items: list[str]) -> dict[str, float]:
  weights = dict(DEFAULT_SCORE_WEIGHTS)
  for item in items:
    if "=" not in item:
      raise SystemExit(f"invalid score weight '{item}', expected metric=weight")
    key, raw_value = item.split("=", 1)
    if key.strip() not in weights:
      raise SystemExit(f"unsupported score metric '{key.strip()}'")
    weights[key.strip()] = float(raw_value)
  return weights


def _parse_noise_seeds(args) -> NoiseSeeds:
  return NoiseSeeds.from_base(args.seed).with_overrides(
    drel=args.seed_drel,
    vrel=args.seed_vrel,
    aego=args.seed_aego,
    vego=args.seed_vego,
  )


if __name__ == "__main__":
  raise SystemExit(main())
