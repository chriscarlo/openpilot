#!/usr/bin/env python3
"""
CAN Signal Hunter — interactive CAN reverse-engineering tool.

Run in --auto mode so an agent (or human) can:
  1. Wait for a stable baseline
  2. Perform a physical action (blinker, door, seatbelt …)
  3. Get a structured diff of which CAN bits changed
  4. Repeat without restarting

Usage (agent-driven, all buses):
  python3 selfdrive/debug/can_signal_hunter.py --auto --json

Usage (human, single bus, with DBC annotation):
  python3 selfdrive/debug/can_signal_hunter.py --bus 0 --dbc toyota_nodsu_pt_generated --table
"""
import argparse
import binascii
import json
import sys
import time
from collections import defaultdict

import cereal.messaging as messaging

try:
  from openpilot.selfdrive.debug.can_table import can_table
  _HAS_TABLE = True
except ImportError:
  _HAS_TABLE = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _emit(obj, use_json: bool):
  if use_json:
    print(json.dumps(obj), flush=True)
  else:
    t = obj.get("type", "")
    if t == "status":
      print(f"[{obj['phase']}] elapsed={obj['elapsed_s']:.1f}s addrs={obj.get('addrs_seen', '?')}", flush=True)
    elif t == "prompt":
      print(f"\n>>> {obj['msg']} (round {obj['round']}) <<<\n", flush=True)
    elif t == "diff":
      n = obj["summary"]["total_changed_addrs"]
      print(f"[diff round={obj['round']}] {n} addr(s) changed", flush=True)
      for c in obj["changes"]:
        known_tag = f" [{c['msg_name']}]" if c.get("msg_name") else (" [known]" if c["known"] else " [unknown]")
        print(f"  bus={c['bus']} addr={c['addr']}{known_tag}  changed_mask={c['changed_bytes_hex']}", flush=True)
    elif t == "summary":
      print(f"\n[summary] {len(obj['rounds'])} round(s) complete", flush=True)
    else:
      print(str(obj), flush=True)


def _load_dbc(names):
  """Return set of known addresses, dict addr->msg_name.  Empty on import failure."""
  known_addrs: set = set()
  addr_to_name: dict = {}
  if not names:
    return known_addrs, addr_to_name
  try:
    from opendbc.can.dbc import DBC  # type: ignore
    for name in names:
      try:
        dbc = DBC(name)
        for msg in dbc.msgs.values():
          known_addrs.add(msg.address)
          addr_to_name[msg.address] = msg.name
      except Exception as e:
        print(f"[warn] could not load DBC '{name}': {e}", file=sys.stderr)
  except ImportError:
    print("[warn] opendbc not available; DBC annotation disabled", file=sys.stderr)
  return known_addrs, addr_to_name


# ── core update ───────────────────────────────────────────────────────────────

def _update(msgs, bus_filter, dat, low_to_high, high_to_low):
  """Process a batch of CAN messages.  State dicts keyed by (bus, address).
  Returns True if any new bit transition was observed."""
  had_new = False
  for x in msgs:
    if x.which() != 'can':
      continue
    for y in x.can:
      if bus_filter is not None and y.src != bus_filter:
        continue
      key = (y.src, y.address)
      dat[key] = bytes(y.dat)
      i = int.from_bytes(y.dat, byteorder='big')
      old_lh = low_to_high[key]
      old_hl = high_to_low[key]
      new_lh = i | old_lh
      new_hl = (~i) | old_hl
      if new_lh != old_lh or new_hl != old_hl:
        had_new = True
      low_to_high[key] = new_lh
      high_to_low[key] = new_hl
  return had_new


def _build_diff(dat, lh_before, hl_before, lh_after, hl_after,
                known_addrs, addr_to_name, unknown_only):
  """Return list of change dicts for addresses where new bits appeared."""
  changes = []
  for key in sorted(dat.keys()):
    bus, addr = key
    init_seen = lh_before[key] & hl_before[key]
    now_seen  = lh_after[key]  & hl_after[key]
    delta = now_seen & ~init_seen
    if delta == 0:
      continue
    raw = dat[key]
    n_bytes = len(raw)
    mask_bytes = delta.to_bytes(n_bytes, byteorder='big')
    # before/after: reconstruct from transitions
    # "before" = last known dat value at baseline snapshot time is unavailable,
    # so we report current dat and the changed mask.
    data_hex = binascii.hexlify(raw).decode()
    mask_hex  = binascii.hexlify(mask_bytes).decode()
    changed_bits = []
    for byte_i, b in enumerate(mask_bytes):
      for bit_i in range(8):
        if b & (1 << bit_i):
          changed_bits.append({"byte": byte_i, "bit": bit_i})
    is_known = addr in known_addrs
    if unknown_only and is_known:
      continue
    changes.append({
      "bus": bus,
      "addr": hex(addr),
      "addr_dec": addr,
      "known": is_known,
      "msg_name": addr_to_name.get(addr),
      "changed_bytes_hex": mask_hex,
      "changed_bits": changed_bits,
      "data_hex": data_hex,
    })
  return changes


# ── main loop ─────────────────────────────────────────────────────────────────

def run(args):
  bus_filter = None if args.bus is None else args.bus
  use_json   = args.json
  known_addrs, addr_to_name = _load_dbc(args.dbc)

  if args.addr:
    messaging.context  # ensure messaging is importable
    sock = messaging.sub_sock('can', timeout=100, addr=args.addr)
  else:
    sock = messaging.sub_sock('can', timeout=100)

  dat:          dict = defaultdict(bytes)
  low_to_high:  dict = defaultdict(int)
  high_to_low:  dict = defaultdict(int)

  all_rounds = []
  round_num  = 0

  if not args.auto:
    # Non-auto: just print changes live until Ctrl-C
    _emit({"type": "status", "phase": "live_mode", "elapsed_s": 0.0, "addrs_seen": 0}, use_json)
    try:
      while True:
        msgs = messaging.drain_sock(sock)
        _update(msgs, bus_filter, dat, low_to_high, high_to_low)
        time.sleep(0.02)
    except KeyboardInterrupt:
      pass
    return

  # ── AUTO MODE ──────────────────────────────────────────────────────────────
  while True:
    round_num += 1

    # ── BASELINE PHASE ──
    t_start    = time.monotonic()
    t_last_new = t_start
    while True:
      msgs = messaging.drain_sock(sock)
      had_new = _update(msgs, bus_filter, dat, low_to_high, high_to_low)
      now = time.monotonic()
      if had_new:
        t_last_new = now
      elapsed = now - t_start
      settled_for = now - t_last_new
      if elapsed >= args.min_baseline and settled_for >= args.settle:
        break
      time.sleep(0.02)

    _emit({
      "type": "status",
      "phase": "baseline_complete",
      "round": round_num,
      "elapsed_s": round(time.monotonic() - t_start, 2),
      "addrs_seen": len(dat),
    }, use_json)

    # Snapshot baseline state
    lh_snap = low_to_high.copy()
    hl_snap = high_to_low.copy()

    # ── PROMPT ──
    _emit({"type": "prompt", "msg": "PERFORM ACTION NOW", "round": round_num}, use_json)

    # ── CAPTURE PHASE ──
    t_cap = time.monotonic()
    try:
      while time.monotonic() - t_cap < args.capture_timeout:
        msgs = messaging.drain_sock(sock)
        _update(msgs, bus_filter, dat, low_to_high, high_to_low)
        time.sleep(0.02)
    except KeyboardInterrupt:
      break

    _emit({
      "type": "status",
      "phase": "capture_complete",
      "round": round_num,
      "elapsed_s": round(time.monotonic() - t_cap, 2),
    }, use_json)

    # ── DIFF ──
    changes = _build_diff(dat, lh_snap, hl_snap, low_to_high, high_to_low,
                          known_addrs, addr_to_name, args.unknown_only)

    buses_changed = sorted({c["bus"] for c in changes})
    summary = {
      "total_changed_addrs":   len(changes),
      "unknown_changed_addrs": sum(1 for c in changes if not c["known"]),
      "known_changed_addrs":   sum(1 for c in changes if c["known"]),
      "buses_with_changes":    buses_changed,
    }
    diff_obj = {"type": "diff", "round": round_num, "changes": changes, "summary": summary}
    _emit(diff_obj, use_json)

    # Optional bit table (human mode)
    if args.table and _HAS_TABLE and not use_json:
      for c in changes:
        try:
          raw = bytes.fromhex(c["data_hex"])
          mask = bytes.fromhex(c["changed_bytes_hex"])
          masked = bytes(a & b for a, b in zip(raw, mask))
          name = c.get("msg_name") or c["addr"]
          print(f"\n{name} (bus {c['bus']}):")
          print(can_table(masked))
        except Exception:
          pass

    all_rounds.append(diff_obj)

  # ── SUMMARY (on Ctrl-C) ──
  _emit({"type": "summary", "rounds": all_rounds}, use_json)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
  parser = argparse.ArgumentParser(
    description="Interactive CAN reverse-engineering: baseline → action → diff.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--bus", type=int, default=None,
                      help="CAN bus to monitor (omit for all buses)")
  parser.add_argument("--json", action="store_true",
                      help="JSON-lines output for agent parsing")
  parser.add_argument("--settle", type=float, default=3.0,
                      help="Seconds with no new bit transitions to consider baseline stable")
  parser.add_argument("--min-baseline", type=float, default=5.0, dest="min_baseline",
                      help="Minimum baseline duration in seconds")
  parser.add_argument("--capture-timeout", type=float, default=10.0, dest="capture_timeout",
                      help="Max capture duration in seconds")
  parser.add_argument("--dbc", action="append", default=[], metavar="NAME",
                      help="DBC file name(s) to load for known-address annotation (repeatable)")
  parser.add_argument("--unknown-only", action="store_true", dest="unknown_only",
                      help="Only report addresses not found in loaded DBC files")
  parser.add_argument("--table", action="store_true",
                      help="Print cabana-style bit table for changed addresses (human mode)")
  parser.add_argument("--auto", action="store_true",
                      help="Auto mode: baseline → prompt → capture → diff → repeat")
  parser.add_argument("--addr", type=str, default=None,
                      help="Messaging address (default: localhost; use device IP for remote)")
  args = parser.parse_args()

  try:
    run(args)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
