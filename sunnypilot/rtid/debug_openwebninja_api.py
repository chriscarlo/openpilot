#!/usr/bin/env python3
"""
RTI OpenWeb Ninja API debug runner.

Purpose:
- Validate key loading in dev/TICI environments
- Exercise real API calls without requiring full onroad stack
- Print compact health + payload summaries for quick troubleshooting
"""

import argparse
import asyncio
import time
from collections import Counter

try:
    from .api_key_manager import API_KEY_PATHS, ENV_VAR_NAMES, get_api_key
    from .waze_api_client import WazeAPIClient
except ImportError:
    from sunnypilot.rtid.api_key_manager import API_KEY_PATHS, ENV_VAR_NAMES, get_api_key
    from sunnypilot.rtid.waze_api_client import WazeAPIClient


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Debug OpenWeb Ninja Waze API calls used by RTI")
    p.add_argument("--lat", type=float, default=37.7749, help="Latitude (default: San Francisco)")
    p.add_argument("--lon", type=float, default=-122.4194, help="Longitude (default: San Francisco)")
    p.add_argument("--radius-km", type=float, default=8.0, help="Search radius in kilometers")
    p.add_argument("--iterations", type=int, default=1, help="Number of calls to run (0 = forever)")
    p.add_argument("--interval-sec", type=float, default=5.0, help="Delay between calls for iterations > 1")
    return p


async def _run(args: argparse.Namespace) -> int:
    key = get_api_key()
    print("key_found=", bool(key))
    if key:
        print("key_len=", len(key))
        print("key_prefix=", f"{key[:4]}...")
    print("env_vars_checked=", ENV_VAR_NAMES)
    print("file_paths_checked=", API_KEY_PATHS)

    if not key:
        print("ERROR: API key not found")
        return 2

    client = WazeAPIClient(key)

    i = 0
    max_iters = args.iterations
    try:
        while max_iters == 0 or i < max_iters:
            i += 1
            t0 = time.time()
            alerts = await client.get_traffic_alerts(args.lat, args.lon, radius_km=args.radius_km)
            dt_ms = (time.time() - t0) * 1000.0

            type_counts = Counter(a.type for a in alerts)
            print(
                f"iter={i} elapsed_ms={dt_ms:.1f} alerts={len(alerts)} "
                f"health={client.get_health_status()} failures={client.consecutive_failures}"
            )
            print("type_counts=", dict(type_counts.most_common(10)))
            if alerts:
                a0 = alerts[0]
                print(
                    "sample_alert=",
                    {
                        "id": a0.id,
                        "type": a0.type,
                        "lat": round(a0.latitude, 5),
                        "lon": round(a0.longitude, 5),
                        "confidence": a0.confidence,
                    },
                )

            if max_iters != 0 and i >= max_iters:
                break
            await asyncio.sleep(max(args.interval_sec, 0.0))
    finally:
        await client.close()

    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
