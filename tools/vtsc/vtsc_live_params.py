#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
  from openpilot.common.params import ParamKeyType, Params, UnknownKeyName
except ModuleNotFoundError:
  # Allow direct execution from repo root without external PYTHONPATH setup.
  sys.path.append(str(Path(__file__).resolve().parents[2]))
  from openpilot.common.params import ParamKeyType, Params, UnknownKeyName

# Keys that matter for runtime VTSC/HUD tuning but do not share a common prefix.
CORE_KEYS = {
  "VisionTurnSpeedControl",
  "VTSCRallyCoPilotHUDEnabled",
  "VTSCExpertModeEnabled",
  "MTSCLookaheadEnabled",
  "VTSCFailOpen",
  "VTSCVerboseDebug",
  "VTSCWriteSnapshotFile",
  "VTSCInterventionRecorderEnabled",
}


@dataclass(frozen=True)
class Bound:
  lo: float
  hi: float


# Bound checks are intentionally strict for expert keys to avoid obvious footguns.
BOUNDS: dict[str, Bound] = {
  "VTSC.Expert.FreewayCurvEps": Bound(1e-7, 1e-2),
  "VTSC.Expert.FreewayMinVisibleM": Bound(20.0, 400.0),
  "VTSC.Expert.FreewayMinConf": Bound(0.05, 0.99),
  "VTSC.Expert.HighwayMinMps": Bound(8.0, 45.0),
  "VTSC.Expert.VTurnHoldMinVMps": Bound(8.0, 45.0),
  "VTSC.Expert.VTurnHoldDeltaMps": Bound(0.1, 8.0),
  "VTSC.Expert.VTurnHoldS": Bound(0.1, 5.0),
  "VTSC.Expert.VTurnHoldSOccluded": Bound(0.05, 3.0),
  "VTSC.Expert.EnteringPredLatAccTh": Bound(0.2, 5.0),
  "VTSC.Expert.TrajectoryPhaseAdvanceS": Bound(-1.0, 4.0),
  "VTSC.Expert.SteerFallbackModelKappaMax": Bound(1e-5, 0.02),
  "VTSC.Expert.SteerFallbackMinKappa": Bound(1e-5, 0.03),
  "VTSC.Expert.SteerFallbackMinVMps": Bound(1.0, 35.0),
  "VTSC.Expert.SevereOvershootSpeedScaleMin": Bound(0.5, 1.0),
  "VTSC.Expert.HiddenTurnVMaxMps": Bound(5.0, 45.0),
  "VTSC.Expert.HiddenTurnTHS": Bound(0.1, 5.0),
  "VTSC.Expert.HiddenTurnDeltaVMps": Bound(0.1, 10.0),
  "VTSC.Expert.HiddenTurnMinOccS": Bound(0.0, 5.0),
  "VTSC.Expert.HiddenTurnAvailScale": Bound(0.0, 2.0),
  "VTSC.Expert.HiddenTurnPhaseS": Bound(0.1, 10.0),
  "VTSC.Expert.HiddenTurnHeadingWinS": Bound(0.1, 5.0),
  "VTSC.Expert.HiddenTurnVisHeadingMaxRad": Bound(0.01, 0.8),
  "VTSC.Expert.LowSpeedMarginMaxVMps": Bound(1.0, 30.0),
  "VTSC.Expert.LowSpeedMarginCurvThresh": Bound(1e-6, 0.01),
  "VTSC.Expert.OcclBypassHeadwayVFloorMps": Bound(0.1, 20.0),
  "VTSC.Expert.OcclBypassLowSpeedVMps": Bound(0.1, 25.0),
  "VTSC.Expert.OcclBypassLeadDRelMaxM": Bound(5.0, 200.0),
  "VTSC.Expert.ConfidenceEnterSevere": Bound(0.10, 0.90),
  "VTSC.Expert.ConfidenceExitToPartial": Bound(0.11, 0.99),
  "VTSC.Expert.AdjLeadCenterYAbsMaxM": Bound(0.5, 8.0),
  "VTSC.Expert.AdjLeadCenterYAbsMinM": Bound(0.1, 6.0),
  "VTSC.Expert.AdjLeadCenterHystM": Bound(0.0, 3.0),
  "VTSC.Expert.AdjLeadCutInDRelMaxM": Bound(5.0, 200.0),
  "VTSC.Expert.AdjLeadCutInYRateMinMps": Bound(0.0, 10.0),
  "VTSC.Expert.AdjLeadLowSpeedBypassVMps": Bound(0.0, 30.0),
  "VTSC.Expert.AdjLeadDedupeDRelEpsM": Bound(0.1, 10.0),
  "VTSC.Expert.AdjLeadDedupeYRelEpsM": Bound(0.05, 5.0),
  "VTSC.Expert.AdjLeadDedupeVRelEpsMps": Bound(0.05, 10.0),
  "VTSCHUD.KappaShowMin": Bound(1e-6, 0.05),
  "VTSCHUD.KappaHoldMin": Bound(1e-6, 0.05),
  "VTSCHUD.CurveHoldNewDistMinM": Bound(0.0, 250.0),
  "VTSCHUD.GeometryEpsilonM": Bound(0.001, 1.0),
  "VTSCHUD.FadeInAlpha": Bound(0.01, 0.95),
  "VTSCHUD.FadeOutAlpha": Bound(0.01, 0.95),
  "VTSCHUD.Scale": Bound(0.5, 4.0),
  "VTSCHUD.BottomSafePxAtScale1": Bound(0.0, 120.0),
  "VTSCHUD.PadPxAtScale1": Bound(2.0, 80.0),
  "VTSCHUD.GapPxAtScale1": Bound(0.0, 80.0),
  "VTSCHUD.TopHeightPxAtScale1": Bound(8.0, 120.0),
  "VTSCHUD.BottomHeightPxAtScale1": Bound(8.0, 140.0),
  "VTSCHUD.MinCurveAreaPxAtScale1": Bound(16.0, 240.0),
  "VTSCHUD.RoadMainWidthPxAtScale1": Bound(2.0, 48.0),
  "VTSCHUD.GlowWidthPxAtScale1": Bound(2.0, 120.0),
  "VTSCHUD.OutlineWidthPxAtScale1": Bound(1.0, 100.0),
  "VTSCHUD.MainStrokeWidthPxAtScale1": Bound(1.0, 100.0),
  "VTSCHUD.DistanceLabelSepPxAtScale1": Bound(0.0, 120.0),
  "VTSCHUD.SpeedFontPxAtScale1": Bound(8.0, 80.0),
  "VTSCHUD.BottomFontPxAtScale1": Bound(8.0, 80.0),
  "VisionTurnSpeedControlLowSpeedLearnedState": Bound(-0.08, 0.04),
  "VisionTurnSpeedControlLowSpeedLearnedHighEndMph": Bound(20.0, 60.0),
}


def is_vtsc_key(key: str) -> bool:
  return (
    key in CORE_KEYS
    or key.startswith("VisionTurnSpeedControl")
    or key.startswith("VTSC.")
    or key.startswith("VTSCHUD.")
  )


def key_tier(key: str) -> str:
  if key.startswith("VTSC.Expert.") or key.startswith("VTSCHUD.") or key == "VTSCExpertModeEnabled":
    return "expert"
  return "safe"


def key_type_name(t: ParamKeyType) -> str:
  return {
    ParamKeyType.STRING: "string",
    ParamKeyType.BOOL: "bool",
    ParamKeyType.INT: "int",
    ParamKeyType.FLOAT: "float",
    ParamKeyType.TIME: "time",
    ParamKeyType.JSON: "json",
    ParamKeyType.BYTES: "bytes",
  }.get(t, "unknown")


def parse_bool(raw: str) -> bool:
  s = raw.strip().lower()
  if s in {"1", "true", "t", "yes", "y", "on"}:
    return True
  if s in {"0", "false", "f", "no", "n", "off"}:
    return False
  raise ValueError(f"invalid bool value: {raw}")


def apply_bounds(key: str, value: Any) -> Any:
  bound = BOUNDS.get(key)
  if bound is None:
    return value
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    v = float(value)
    if not math.isfinite(v):
      raise ValueError(f"{key}: non-finite numeric value")
    if v < bound.lo or v > bound.hi:
      raise ValueError(f"{key}: {v} out of range [{bound.lo}, {bound.hi}]")
    return int(v) if isinstance(value, int) else v
  return value


def parse_for_key(params: Params, key: str, raw: Any) -> Any:
  t = params.get_type(key)
  if t == ParamKeyType.BOOL:
    if isinstance(raw, bool):
      return raw
    return parse_bool(str(raw))
  if t == ParamKeyType.INT:
    return int(raw)
  if t == ParamKeyType.FLOAT:
    return float(raw)
  if t == ParamKeyType.JSON:
    if isinstance(raw, (dict, list)):
      return raw
    return json.loads(str(raw))
  if t == ParamKeyType.BYTES:
    if isinstance(raw, (bytes, bytearray)):
      return bytes(raw)
    return str(raw).encode("utf-8")
  # STRING/TIME fallback
  return str(raw)


def get_vtsc_keys(params: Params, tier: str) -> list[str]:
  keys = sorted(str(k) for k in params.all_keys() if is_vtsc_key(str(k)))
  if tier == "all":
    return keys
  return [k for k in keys if key_tier(k) == tier]


def ensure_expert_mode(params: Params, allow_enable: bool) -> None:
  if params.get_bool("VTSCExpertModeEnabled"):
    return
  if not allow_enable:
    raise RuntimeError("expert key write blocked: VTSCExpertModeEnabled is 0 (use --enable-expert)")
  params.put_bool("VTSCExpertModeEnabled", True)
  print("enabled VTSCExpertModeEnabled=1")


def cmd_list(args: argparse.Namespace) -> int:
  params = Params()
  keys = get_vtsc_keys(params, args.tier)
  if args.json:
    out = []
    for key in keys:
      t = params.get_type(key)
      out.append({
        "key": key,
        "tier": key_tier(key),
        "type": key_type_name(t),
        "current": params.get(key),
        "default": params.get_default_value(key),
        "bounds": vars(BOUNDS[key]) if key in BOUNDS else None,
      })
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0

  for key in keys:
    t = params.get_type(key)
    cur = params.get(key)
    default = params.get_default_value(key)
    bound = BOUNDS.get(key)
    btxt = f"[{bound.lo}, {bound.hi}]" if bound else "-"
    print(f"{key} tier={key_tier(key)} type={key_type_name(t)} current={cur} default={default} bounds={btxt}")
  return 0


def write_key(params: Params, key: str, value: Any, *, enable_expert: bool) -> None:
  if key_tier(key) == "expert" and key != "VTSCExpertModeEnabled":
    ensure_expert_mode(params, enable_expert)
  parsed = parse_for_key(params, key, value)
  parsed = apply_bounds(key, parsed)
  t = params.get_type(key)
  if t == ParamKeyType.BOOL:
    params.put_bool(key, bool(parsed))
  else:
    params.put(key, parsed)


def cmd_set(args: argparse.Namespace) -> int:
  params = Params()
  key = args.key
  try:
    params.check_key(key)
  except UnknownKeyName:
    print(f"unknown key: {key}", file=sys.stderr)
    return 2
  try:
    write_key(params, key, args.value, enable_expert=args.enable_expert)
  except Exception as e:
    print(f"failed to set {key}: {e}", file=sys.stderr)
    return 2
  print(f"set {key}={params.get(key)}")
  return 0


def cmd_apply(args: argparse.Namespace) -> int:
  params = Params()
  path = Path(args.path)
  if not path.exists():
    print(f"file not found: {path}", file=sys.stderr)
    return 2
  payload = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(payload, dict):
    items = payload.items()
  elif isinstance(payload, list):
    items = [(str(item["key"]), item["value"]) for item in payload]
  else:
    print("profile must be an object or list[{key,value}]", file=sys.stderr)
    return 2

  failures = 0
  for key, value in items:
    try:
      params.check_key(key)
      write_key(params, key, value, enable_expert=args.enable_expert)
      print(f"applied {key}={params.get(key)}")
    except Exception as e:
      failures += 1
      print(f"failed {key}: {e}", file=sys.stderr)
  return 1 if failures else 0


def cmd_watch(args: argparse.Namespace) -> int:
  params = Params()
  keys = get_vtsc_keys(params, args.tier)
  if args.keys:
    requested = [k.strip() for k in args.keys.split(",") if k.strip()]
    keys = [k for k in requested if is_vtsc_key(k)]
  prev = {k: params.get(k) for k in keys}
  print(f"watching {len(keys)} keys (interval={args.interval}s)")
  for k in keys:
    print(f"  {k}={prev[k]}")
  try:
    while True:
      time.sleep(args.interval)
      for k in keys:
        cur = params.get(k)
        if cur != prev[k]:
          print(f"{k}: {prev[k]} -> {cur}")
          prev[k] = cur
  except KeyboardInterrupt:
    return 0


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Live VTSC/HUD params utility (list/set/apply/watch).")
  sub = p.add_subparsers(dest="cmd", required=True)

  p_list = sub.add_parser("list", help="List VTSC/HUD keys and current values.")
  p_list.add_argument("--tier", choices=["safe", "expert", "all"], default="all")
  p_list.add_argument("--json", action="store_true")
  p_list.set_defaults(func=cmd_list)

  p_set = sub.add_parser("set", help="Set one key.")
  p_set.add_argument("key")
  p_set.add_argument("value")
  p_set.add_argument("--enable-expert", action="store_true", help="Auto-enable VTSCExpertModeEnabled when writing expert keys.")
  p_set.set_defaults(func=cmd_set)

  p_apply = sub.add_parser("apply", help="Apply a JSON profile {key: value}.")
  p_apply.add_argument("path")
  p_apply.add_argument("--enable-expert", action="store_true", help="Auto-enable VTSCExpertModeEnabled when writing expert keys.")
  p_apply.set_defaults(func=cmd_apply)

  p_watch = sub.add_parser("watch", help="Watch keys and print changes.")
  p_watch.add_argument("--tier", choices=["safe", "expert", "all"], default="all")
  p_watch.add_argument("--keys", help="Comma-separated explicit keys.")
  p_watch.add_argument("--interval", type=float, default=0.2)
  p_watch.set_defaults(func=cmd_watch)
  return p


def main() -> int:
  args = build_parser().parse_args()
  return int(args.func(args))


if __name__ == "__main__":
  raise SystemExit(main())
