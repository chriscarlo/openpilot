from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closed_loop import run_harness
from .config import FRIENDLY_PARAM_NAMES, NOISE_PROFILES, NoiseSeeds, resolve_ev6_vehicle_config
from .inputs import SCENARIO_NAMES, build_synthetic_scenario, load_snapshot_bundle


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="EV6 CAN FD longitudinal dev harness")
  parser.add_argument("--mode", choices=("synthetic-ev6", "snapshot"), default="synthetic-ev6")
  parser.add_argument("--scenario", default="approach", choices=SCENARIO_NAMES)
  parser.add_argument("--snapshot", type=Path, default=None, help="Snapshot bundle directory for --mode snapshot")
  parser.add_argument("--topology", choices=("lka", "lfa"), default="lka")
  parser.add_argument("--controller-mode", choices=("auto", "device", "passthrough", "shaped"), default="device")
  parser.add_argument("--perception-filter", choices=("auto", "direct", "radard"), default="auto")
  parser.add_argument("--hyundai-tuning-mode", choices=("off", "dynamic", "predictive"), default=None)
  parser.add_argument("--noise", choices=tuple(NOISE_PROFILES.keys()), default="realistic")
  parser.add_argument("--duration", type=float, default=12.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--seed-drel", type=int, default=None)
  parser.add_argument("--seed-vrel", type=int, default=None)
  parser.add_argument("--seed-aego", type=int, default=None)
  parser.add_argument("--seed-vego", type=int, default=None)
  parser.add_argument("--override", action="append", default=[], help="Param override, either friendly_name=value or full ParamKey=value")
  parser.add_argument("--output-json", type=Path, default=None)
  parser.add_argument("--output-trace", type=Path, default=None)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  overrides = _parse_overrides(args.override)
  hyundai_tuning_mode = _parse_hyundai_tuning_mode(args.hyundai_tuning_mode)
  noise_seeds = _parse_noise_seeds(args)

  if args.mode == "snapshot":
    if args.snapshot is None:
      raise SystemExit("--snapshot is required with --mode snapshot")
    bundle = load_snapshot_bundle(args.snapshot)
    vehicle_config = resolve_ev6_vehicle_config(
      topology=str(bundle.vehicle.get("topology", args.topology)),
      controller_mode=args.controller_mode,
      tune_source="snapshot",
      param_overrides=overrides,
      hyundai_tuning_mode=hyundai_tuning_mode,
      snapshot_vehicle=bundle.vehicle,
      snapshot_params=bundle.params,
    )
    steps = bundle.timeline
    initial_speed_mps = bundle.initial_speed_mps
    initial_accel_mps2 = bundle.initial_accel_mps2
    scenario_name = bundle.name or args.scenario
  else:
    vehicle_config = resolve_ev6_vehicle_config(
      topology=args.topology,
      controller_mode=args.controller_mode,
      tune_source="defaults" if not overrides else "cli",
      param_overrides=overrides,
      hyundai_tuning_mode=hyundai_tuning_mode,
    )
    initial_speed_mps, initial_accel_mps2, steps = build_synthetic_scenario(args.scenario, duration_s=args.duration, dt_s=0.05)
    scenario_name = args.scenario

  result = run_harness(
    vehicle_config=vehicle_config,
    scenario_name=scenario_name,
    steps=steps,
    initial_speed_mps=initial_speed_mps,
    initial_accel_mps2=initial_accel_mps2,
    noise_profile=args.noise,
    seed=args.seed,
    noise_seeds=noise_seeds,
    perception_filter=args.perception_filter,
  )

  print(json.dumps(result.summary, indent=2, sort_keys=True))
  if args.output_json:
    args.output_json.write_text(json.dumps({"vehicle": result.vehicle, "summary": result.summary}, indent=2, sort_keys=True))
  if args.output_trace:
    args.output_trace.write_text(json.dumps(result.trace, indent=2))
  return 0


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
    key = FRIENDLY_PARAM_NAMES.get(key.strip(), key.strip())
    overrides[key] = value.strip()
  return overrides


def _parse_noise_seeds(args) -> NoiseSeeds:
  return NoiseSeeds.from_base(args.seed).with_overrides(
    drel=args.seed_drel,
    vrel=args.seed_vrel,
    aego=args.seed_aego,
    vego=args.seed_vego,
  )


if __name__ == "__main__":
  raise SystemExit(main())
