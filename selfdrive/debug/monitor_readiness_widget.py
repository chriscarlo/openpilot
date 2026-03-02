#!/usr/bin/env python3
"""
Print the onroad "readiness widget" dot statuses (selfdriveStateSP.subsystemStatuses)
in a human-readable form.

This is intended for SSH use when the on-device HUD text is hard to read.

Example:
  ./selfdrive/debug/monitor_readiness_widget.py --once
  ./selfdrive/debug/monitor_readiness_widget.py --rate 2
"""

import argparse
import json
import time

import cereal.messaging as messaging


STATUS_TO_NAME = {
  0: "RED",
  1: "YELLOW",
  2: "GREEN",
}


def format_status(s: int) -> str:
  return STATUS_TO_NAME.get(s, f"UNKNOWN({s})")


def get_snapshot(timeout_ms: int):
  sm = messaging.SubMaster(["selfdriveStateSP"])
  sm.update(timeout_ms)
  if not sm.updated.get("selfdriveStateSP", False):
    return None
  return sm["selfdriveStateSP"]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
  parser.add_argument("--rate", type=float, default=1.0, help="Refresh rate in Hz (default: 1.0)")
  parser.add_argument("--timeout-ms", type=int, default=2000, help="SubMaster update timeout in ms (default: 2000)")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of formatted text")
  args = parser.parse_args()

  period_s = 1.0 / max(args.rate, 0.1)

  while True:
    ss = get_snapshot(args.timeout_ms)
    if ss is None:
      if args.json:
        print(json.dumps({"ok": False, "error": "timeout waiting for selfdriveStateSP"}))
      else:
        print("timeout waiting for selfdriveStateSP")
      if args.once:
        return 2
      time.sleep(period_s)
      continue

    statuses = []
    try:
      for st in ss.subsystemStatuses:
        statuses.append({
          "name": st.name,
          "status": int(st.status),
          "status_name": format_status(int(st.status)),
        })
    except Exception:
      # Tolerate schema/version mismatch; at least expose whether we received the msg.
      statuses = []

    if args.json:
      print(json.dumps({
        "ok": True,
        "allSystemsReady": bool(getattr(ss, "allSystemsReady", False)),
        "subsystemStatuses": statuses,
      }))
    else:
      ts = time.strftime("%Y-%m-%d %H:%M:%S")
      all_ready = bool(getattr(ss, "allSystemsReady", False))
      hdr = f"[{ts}] allSystemsReady={all_ready}"
      print(hdr)
      if not statuses:
        print("  (no subsystemStatuses field present)")
      else:
        for st in statuses:
          print(f"  {st['name']}: {st['status_name']}")

    if args.once:
      return 0
    time.sleep(period_s)


if __name__ == "__main__":
  raise SystemExit(main())

