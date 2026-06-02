from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from .catalog import get_episode_record, get_snapshot_record_by_bundle_path, open_catalog
from .config import FRIENDLY_PARAM_NAMES, NoiseSeeds
from .inputs import load_snapshot_bundle
from .sweep import DEFAULT_SCORE_WEIGHTS, SweepCandidate, enumerate_candidates, run_sweep


CURRENT_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkCase:
  case_id: str
  snapshot_path: str
  weight: float = 1.0
  snapshot_name: str | None = None
  snapshot_id: int | None = None
  episode_id: int | None = None
  episode_type: str | None = None
  route_id: int | None = None
  route_key: str | None = None
  source_root: str | None = None
  confidence: float | None = None
  notes: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def from_dict(cls, payload: dict[str, Any]) -> BenchmarkCase:
    return cls(
      case_id=str(payload["caseId"]),
      snapshot_path=str(payload["snapshotPath"]),
      weight=float(payload.get("weight", 1.0)),
      snapshot_name=_optional_str(payload.get("snapshotName")),
      snapshot_id=_optional_int(payload.get("snapshotId")),
      episode_id=_optional_int(payload.get("episodeId")),
      episode_type=_optional_str(payload.get("episodeType")),
      route_id=_optional_int(payload.get("routeId")),
      route_key=_optional_str(payload.get("routeKey")),
      source_root=_optional_str(payload.get("sourceRoot")),
      confidence=_optional_float(payload.get("confidence")),
      notes=dict(payload.get("notes") or {}),
    )

  def to_dict(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "caseId": self.case_id,
      "snapshotPath": self.snapshot_path,
      "weight": self.weight,
    }
    _set_if_present(payload, "snapshotName", self.snapshot_name)
    _set_if_present(payload, "snapshotId", self.snapshot_id)
    _set_if_present(payload, "episodeId", self.episode_id)
    _set_if_present(payload, "episodeType", self.episode_type)
    _set_if_present(payload, "routeId", self.route_id)
    _set_if_present(payload, "routeKey", self.route_key)
    _set_if_present(payload, "sourceRoot", self.source_root)
    _set_if_present(payload, "confidence", self.confidence)
    if self.notes:
      payload["notes"] = dict(self.notes)
    return payload


@dataclass(frozen=True)
class BenchmarkManifest:
  path: Path
  schema_version: int
  name: str
  description: str
  cases: list[BenchmarkCase]

  def to_dict(self) -> dict[str, Any]:
    return {
      "schemaVersion": self.schema_version,
      "name": self.name,
      "description": self.description,
      "cases": [case.to_dict() for case in self.cases],
    }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="EV6 longitudinal harness benchmark manifests and batch runs")
  subparsers = parser.add_subparsers(dest="command", required=True)

  write_manifest = subparsers.add_parser("write-manifest", help="Create a benchmark manifest from catalog episodes and/or snapshot bundles")
  write_manifest.add_argument("--db-path", type=Path, default=None)
  write_manifest.add_argument("--output", type=Path, required=True)
  write_manifest.add_argument("--name", required=True)
  write_manifest.add_argument("--description", default="")
  write_manifest.add_argument("--episode-id", action="append", type=int, default=[])
  write_manifest.add_argument("--bundle-path", action="append", type=Path, default=[])

  run_manifest = subparsers.add_parser("run", help="Run a multi-bundle benchmark manifest")
  run_manifest.add_argument("--manifest", type=Path, required=True)
  run_manifest.add_argument("--controller-mode", choices=("auto", "passthrough", "shaped"), default="passthrough")
  run_manifest.add_argument("--hyundai-tuning-mode", choices=("off", "dynamic", "predictive"), default=None)
  run_manifest.add_argument("--noise", choices=("off", "realistic", "stress"), default="off")
  run_manifest.add_argument("--seed", type=int, default=42)
  run_manifest.add_argument("--seed-drel", type=int, default=None)
  run_manifest.add_argument("--seed-vrel", type=int, default=None)
  run_manifest.add_argument("--seed-aego", type=int, default=None)
  run_manifest.add_argument("--seed-vego", type=int, default=None)
  run_manifest.add_argument("--override", action="append", default=[], help="Base override, either friendly_name=value or ParamKey=value")
  run_manifest.add_argument("--grid", action="append", default=[], help="Sweep grid entry: friendly_name=v1,v2,v3")
  run_manifest.add_argument("--score-weight", action="append", default=[], help="Override score weights: metric=weight")
  run_manifest.add_argument("--top-k", type=int, default=5)
  run_manifest.add_argument("--output-json", type=Path, default=None)
  run_manifest.add_argument("--output-csv", type=Path, default=None)

  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.command == "write-manifest":
    payload = _handle_write_manifest(args)
  elif args.command == "run":
    payload = _handle_run_manifest(args)
  else:
    raise SystemExit(f"unsupported command '{args.command}'")

  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def load_benchmark_manifest(manifest_path: str | Path) -> BenchmarkManifest:
  path = Path(manifest_path)
  payload = json.loads(path.read_text())
  schema_version = int(payload.get("schemaVersion", CURRENT_MANIFEST_SCHEMA_VERSION))
  if schema_version != CURRENT_MANIFEST_SCHEMA_VERSION:
    raise ValueError(f"unsupported benchmark manifest schemaVersion={schema_version}")

  cases = [BenchmarkCase.from_dict(case_payload) for case_payload in payload.get("cases", [])]
  if not cases:
    raise ValueError("benchmark manifest must contain at least one case")

  seen_case_ids: set[str] = set()
  for case in cases:
    if case.case_id in seen_case_ids:
      raise ValueError(f"duplicate benchmark caseId '{case.case_id}'")
    seen_case_ids.add(case.case_id)

  return BenchmarkManifest(
    path=path,
    schema_version=schema_version,
    name=str(payload.get("name") or path.stem),
    description=str(payload.get("description") or ""),
    cases=cases,
  )


def run_manifest_benchmark(*,
                           manifest: BenchmarkManifest,
                           candidates: list[SweepCandidate],
                           controller_mode: str,
                           hyundai_tuning_mode: int | None,
                           noise: str,
                           seed: int,
                           noise_seeds: NoiseSeeds | None,
                           score_weights: dict[str, float]) -> dict[str, Any]:
  base_noise_seeds = noise_seeds or NoiseSeeds.from_base(seed)
  candidate_keys = [_candidate_key(candidate.overrides) for candidate in candidates]
  aggregate_entries: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {
    candidate_key: {
      "score": 0.0,
      "weightedPenalties": {metric_name: 0.0 for metric_name in score_weights},
      "overrides": dict(candidate.overrides),
      "caseScores": {},
      "caseWeightedPenalties": {},
      "caseSummaries": {},
      "vehicle": None,
    }
    for candidate, candidate_key in zip(candidates, candidate_keys, strict=True)
  }

  for case_index, case in enumerate(manifest.cases):
    snapshot_path = resolve_manifest_snapshot_path(case.snapshot_path, manifest.path)
    case_seed = seed + case_index * 1000
    case_payload = run_sweep(
      candidates=candidates,
      scenarios=[],
      mode="snapshot",
      snapshot=snapshot_path,
      topology="lka",
      controller_mode=controller_mode,
      hyundai_tuning_mode=hyundai_tuning_mode,
      noise=noise,
      duration_s=0.0,
      seed=case_seed,
      noise_seeds=base_noise_seeds.offset(case_index * 1000),
      score_weights=score_weights,
    )
    per_candidate = {
      _candidate_key(entry["overrides"]): entry
      for entry in case_payload["rankedCandidates"]
    }

    for candidate_key in candidate_keys:
      entry = per_candidate[candidate_key]
      aggregate = aggregate_entries[candidate_key]
      aggregate["score"] += case.weight * float(entry["score"])
      for metric_name, metric_value in entry["weightedPenalties"].items():
        aggregate["weightedPenalties"][metric_name] += case.weight * float(metric_value)
      aggregate["caseScores"][case.case_id] = round(float(entry["score"]), 6)
      aggregate["caseWeightedPenalties"][case.case_id] = {
        metric_name: round(float(metric_value), 6)
        for metric_name, metric_value in entry["weightedPenalties"].items()
      }
      aggregate["caseSummaries"][case.case_id] = next(iter(entry["scenarioSummaries"].values()))
      if aggregate["vehicle"] is None:
        aggregate["vehicle"] = dict(entry["vehicle"])

  ranked_candidates = []
  for aggregate in aggregate_entries.values():
    ranked_candidates.append({
      "score": round(aggregate["score"], 6),
      "weightedPenalties": {
        metric_name: round(metric_value, 6)
        for metric_name, metric_value in aggregate["weightedPenalties"].items()
      },
      "overrides": aggregate["overrides"],
      "caseScores": aggregate["caseScores"],
      "caseWeightedPenalties": aggregate["caseWeightedPenalties"],
      "caseSummaries": aggregate["caseSummaries"],
      "vehicle": aggregate["vehicle"],
    })

  ranked_candidates.sort(key=lambda entry: entry["score"])
  for rank, entry in enumerate(ranked_candidates, start=1):
    entry["rank"] = rank

  return {
    "mode": "benchmark-manifest",
    "manifestPath": str(manifest.path),
    "manifestName": manifest.name,
    "manifestDescription": manifest.description,
    "manifestCases": [case.to_dict() for case in manifest.cases],
    "noiseProfile": noise,
    "scoreWeights": dict(score_weights),
    "noiseSeeds": base_noise_seeds.as_dict(),
    "candidateCount": len(candidates),
    "caseCount": len(manifest.cases),
    "rankedCandidates": ranked_candidates,
  }


def write_benchmark_manifest(manifest: BenchmarkManifest, output_path: str | Path) -> None:
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


def write_benchmark_csv(payload: dict[str, Any], output_path: Path) -> None:
  manifest_cases = [case["caseId"] for case in payload.get("manifestCases", [])]
  fieldnames = ["rank", "score", "overrides", *manifest_cases]
  with output_path.open("w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for entry in payload["rankedCandidates"]:
      row = {
        "rank": entry["rank"],
        "score": entry["score"],
        "overrides": json.dumps(entry["overrides"], sort_keys=True),
      }
      row.update({case_id: entry["caseScores"].get(case_id) for case_id in manifest_cases})
      writer.writerow(row)


def resolve_manifest_snapshot_path(snapshot_path: str | Path, manifest_path: str | Path) -> Path:
  raw_path = Path(snapshot_path)
  if raw_path.is_absolute():
    return raw_path

  repo_candidate = Path.cwd() / raw_path
  if repo_candidate.exists():
    return repo_candidate
  return Path(manifest_path).parent / raw_path


def _handle_write_manifest(args) -> dict[str, Any]:
  if not args.episode_id and not args.bundle_path:
    raise SystemExit("write-manifest requires at least one --episode-id or --bundle-path")

  conn = open_catalog(args.db_path)
  try:
    cases = []
    for episode_id in args.episode_id:
      record = get_episode_record(conn, episode_id)
      if record is None:
        raise SystemExit(f"episode {episode_id} not found")
      if record.get("bundle_path") is None:
        raise SystemExit(f"episode {episode_id} does not have a snapshot bundle")
      cases.append(_case_from_episode_record(record))

    for bundle_path in args.bundle_path:
      bundle_record = get_snapshot_record_by_bundle_path(conn, bundle_path)
      if bundle_record is not None:
        cases.append(_case_from_snapshot_record(bundle_record))
      else:
        cases.append(_case_from_bundle_path(bundle_path))
  finally:
    conn.close()

  cases = _dedupe_case_ids(cases)
  manifest = BenchmarkManifest(
    path=Path(args.output),
    schema_version=CURRENT_MANIFEST_SCHEMA_VERSION,
    name=str(args.name),
    description=str(args.description),
    cases=cases,
  )
  write_benchmark_manifest(manifest, args.output)
  return manifest.to_dict()


def _handle_run_manifest(args) -> dict[str, Any]:
  manifest = load_benchmark_manifest(args.manifest)
  candidates = enumerate_candidates(_parse_overrides(args.override), _parse_grid(args.grid))
  payload = run_manifest_benchmark(
    manifest=manifest,
    candidates=candidates,
    controller_mode=args.controller_mode,
    hyundai_tuning_mode=_parse_hyundai_tuning_mode(args.hyundai_tuning_mode),
    noise=args.noise,
    seed=args.seed,
    noise_seeds=_parse_noise_seeds(args),
    score_weights=_parse_score_weights(args.score_weight),
  )
  if args.top_k > 0:
    payload["rankedCandidates"] = payload["rankedCandidates"][:args.top_k]
  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  if args.output_csv is not None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_benchmark_csv(payload, args.output_csv)
  return payload


def _candidate_key(overrides: dict[str, Any]) -> tuple[tuple[str, str], ...]:
  return tuple(sorted((str(key), str(value)) for key, value in overrides.items()))


def _case_from_episode_record(record: dict[str, Any]) -> BenchmarkCase:
  bundle_path = Path(str(record["bundle_path"]))
  return BenchmarkCase(
    case_id=f"{record['episode_type']}_r{record['route_id']}_e{record['episode_id']}",
    snapshot_path=_format_manifest_path(bundle_path),
    snapshot_name=bundle_path.name,
    snapshot_id=_optional_int(record.get("snapshot_id")),
    episode_id=int(record["episode_id"]),
    episode_type=str(record["episode_type"]),
    route_id=int(record["route_id"]),
    route_key=str(record["route_key"]),
    source_root=str(record["source_root"]),
    confidence=_optional_float(record.get("confidence")),
  )


def _case_from_snapshot_record(record: dict[str, Any]) -> BenchmarkCase:
  bundle_path = Path(str(record["bundle_path"]))
  notes = {}
  if record.get("episode_id") is None:
    notes["origin"] = "snapshot"
  return BenchmarkCase(
    case_id=_default_snapshot_case_id(record),
    snapshot_path=_format_manifest_path(bundle_path),
    snapshot_name=_optional_str(record.get("name")) or bundle_path.name,
    snapshot_id=int(record["snapshot_id"]),
    episode_id=_optional_int(record.get("episode_id")),
    episode_type=_optional_str(record.get("episode_type")),
    route_id=_optional_int(record.get("route_id")),
    route_key=_optional_str(record.get("route_key")),
    source_root=_optional_str(record.get("source_root")),
    notes=notes,
  )


def _case_from_bundle_path(bundle_path: Path) -> BenchmarkCase:
  bundle = load_snapshot_bundle(bundle_path)
  return BenchmarkCase(
    case_id=f"snapshot_{bundle_path.name}",
    snapshot_path=_format_manifest_path(bundle_path),
    snapshot_name=bundle.name or bundle_path.name,
    notes={"origin": "bundle-path"},
  )


def _dedupe_case_ids(cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
  seen: dict[str, int] = {}
  deduped = []
  for case in cases:
    next_index = seen.get(case.case_id, 0) + 1
    seen[case.case_id] = next_index
    if next_index == 1:
      deduped.append(case)
    else:
      deduped.append(replace(case, case_id=f"{case.case_id}_{next_index}"))
  return deduped


def _default_snapshot_case_id(record: dict[str, Any]) -> str:
  episode_type = _optional_str(record.get("episode_type"))
  route_id = _optional_int(record.get("route_id"))
  episode_id = _optional_int(record.get("episode_id"))
  if episode_type is not None and route_id is not None and episode_id is not None:
    return f"{episode_type}_r{route_id}_e{episode_id}"
  return f"snapshot_{record['snapshot_id']}"


def _format_manifest_path(path: Path) -> str:
  resolved = path.resolve()
  cwd = Path.cwd().resolve()
  try:
    return str(resolved.relative_to(cwd))
  except ValueError:
    return str(path)


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


def _parse_hyundai_tuning_mode(raw: str | None) -> int | None:
  if raw is None:
    return None
  mapping = {"off": 0, "dynamic": 1, "predictive": 2}
  return mapping[raw]


def _parse_noise_seeds(args) -> NoiseSeeds:
  return NoiseSeeds.from_base(args.seed).with_overrides(
    drel=args.seed_drel,
    vrel=args.seed_vrel,
    aego=args.seed_aego,
    vego=args.seed_vego,
  )


def _optional_str(value: Any) -> str | None:
  if value is None:
    return None
  return str(value)


def _optional_int(value: Any) -> int | None:
  if value is None:
    return None
  return int(value)


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  return float(value)


def _set_if_present(payload: dict[str, Any], key: str, value: Any) -> None:
  if value is not None:
    payload[key] = value


if __name__ == "__main__":
  raise SystemExit(main())
