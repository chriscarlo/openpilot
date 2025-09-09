#!/usr/bin/env python3

"""
Bluetooth helper for comma 3/3X (AGNOS)

CLI wrapper around BlueZ via bluetoothctl with bounded timeouts.
Implements status/power/discoverable/scan/pair/trust/connect/disconnect/remove.

Notes:
- Uses bluetoothctl interactively to handle scan/pair flows without extra deps.
- All operations are time-bounded; defaults chosen for UI responsiveness.
- Outputs human-readable by default and JSON with --json where applicable.
"""

import argparse
import json
import os
import re
import shlex
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import subprocess


DEFAULT_SHORT_TIMEOUT = 8.0
DEFAULT_SCAN_TIMEOUT = 10.0
DEFAULT_PAIR_TIMEOUT = 60.0


RE_CTRL_HEADER = re.compile(r"^Controller\s+([0-9A-F]{2}(?::[0-9A-F]{2}){5})\b", re.I)
RE_PROP_BOOL = re.compile(r"^(Powered|Discoverable|Pairable):\s*(yes|no)$", re.I)
RE_PROP_STR = re.compile(r"^(Name|Alias|Address):\s*(.+)$", re.I)

RE_EVT_NEW = re.compile(r"^\[NEW\]\s+Device\s+([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s+(.+)$", re.I)
RE_EVT_CHG_NAME = re.compile(r"^\[CHG\]\s+Device\s+([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s+Name:\s*(.+)$", re.I)
RE_EVT_CHG_RSSI = re.compile(r"^\[CHG\]\s+Device\s+([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s+RSSI:\s*(-?\d+)\b", re.I)

RE_DEVICES_LINE = re.compile(r"^Device\s+([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s+(.+)$", re.I)

RE_INFO_BOOL = re.compile(r"^(Paired|Trusted|Connected):\s*(yes|no)$", re.I)


class ExitCodes:
  OK = 0
  NO_CONTROLLER = 2
  TIMEOUT = 3
  NO_BLUEZ = 4
  CMD_FAILED = 5
  BAD_ARGS = 6
  NEED_CONFIRM = 10
  NEED_PIN = 11
  NEED_PASSKEY = 12


def which(cmd: str) -> Optional[str]:
  for path in os.environ.get("PATH", "").split(":"):
    p = os.path.join(path, cmd)
    if os.path.isfile(p) and os.access(p, os.X_OK):
      return p
  return None


def run(cmd: List[str], timeout: float = DEFAULT_SHORT_TIMEOUT) -> Tuple[int, str]:
  try:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    return 0, out
  except subprocess.CalledProcessError as e:
    return e.returncode if e.returncode != 0 else 1, e.output or ""
  except subprocess.TimeoutExpired as e:
    return ExitCodes.TIMEOUT, (e.output or "")


def ensure_bluetoothctl() -> None:
  if which("bluetoothctl") is None:
    print("bluetoothctl not found; install BlueZ first", file=sys.stderr)
    sys.exit(ExitCodes.NO_BLUEZ)


def btctl_show(timeout: float = DEFAULT_SHORT_TIMEOUT) -> Tuple[bool, Dict[str, object], str]:
  """Return (present, props, raw_output)."""
  ensure_bluetoothctl()
  code, out = run(["bluetoothctl", "show"], timeout=timeout)
  if code != 0:
    # bluetoothctl returns 0 even when no controller sometimes; detect by text
    return False, {}, out
  present = False
  props: Dict[str, object] = {}
  for line in out.splitlines():
    m = RE_CTRL_HEADER.search(line.strip())
    if m:
      present = True
      props["Address"] = m.group(1)
      continue
    m2 = RE_PROP_BOOL.search(line.strip())
    if m2:
      k = m2.group(1).title()
      props[k] = (m2.group(2).lower() == "yes")
      continue
    m3 = RE_PROP_STR.search(line.strip())
    if m3:
      props[m3.group(1).title()] = m3.group(2).strip()
  return present, props, out


def btctl_interactive(commands: List[str], timeout: float) -> Tuple[str, int]:
  """Run bluetoothctl interactively, feeding commands gradually.
  Returns (combined_output, exitcode).
  """
  ensure_bluetoothctl()
  p = subprocess.Popen(["bluetoothctl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
  start = time.time()
  combined: List[str] = []
  try:
    for cmd in commands:
      if time.time() - start > timeout:
        raise TimeoutError
      if p.stdin:
        p.stdin.write(cmd + "\n")
        p.stdin.flush()
      # read opportunistically for a short slice
      slice_end = time.time() + 0.2
      while time.time() < slice_end:
        if p.stdout and p.stdout.readable():
          line = p.stdout.readline()
          if not line:
            break
          combined.append(line.rstrip())
        else:
          break
    # always quit
    if p.stdin:
      p.stdin.write("quit\n")
      p.stdin.flush()
    # drain remaining output or until timeout
    while time.time() - start <= timeout:
      if p.poll() is not None:
        break
      if p.stdout and p.stdout.readable():
        line = p.stdout.readline()
        if not line:
          break
        combined.append(line.rstrip())
      else:
        time.sleep(0.05)
  except TimeoutError:
    try:
      p.kill()
    except Exception:
      pass
    return "\n".join(combined), ExitCodes.TIMEOUT
  finally:
    try:
      if p.stdin:
        p.stdin.close()
    except Exception:
      pass
  # collect final code
  try:
    rc = p.wait(timeout=1.0)
  except subprocess.TimeoutExpired:
    rc = 0
  return "\n".join(combined), rc


def cmd_status(args: argparse.Namespace) -> int:
  present, props, raw = btctl_show(timeout=args.timeout)
  if args.json:
    payload = {"present": present, **props}
    print(json.dumps(payload))
  else:
    if not present:
      print("No default controller available")
    else:
      name = props.get("Name") or props.get("Alias") or ""
      addr = props.get("Address") or ""
      powered = "yes" if props.get("Powered") else "no"
      discoverable = "yes" if props.get("Discoverable") else "no"
      print(f"Controller {addr} {name}\nPowered: {powered}\nDiscoverable: {discoverable}")
  return ExitCodes.OK if present else ExitCodes.NO_CONTROLLER


def cmd_set_powered(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  target = "on" if args.state.lower() == "on" else "off"
  out, rc = btctl_interactive(["power " + target], timeout=args.timeout)
  if args.verbose:
    print(out)
  # verify
  present, props, _ = btctl_show(timeout=DEFAULT_SHORT_TIMEOUT)
  ok = present and bool(props.get("Powered")) == (target == "on")
  if not ok:
    print("Failed to set power", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print(f"Powered set to {target}")
  return ExitCodes.OK


def cmd_set_discoverable(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  cmds = []
  if args.state.lower() == "on" and args.timeout_sec is not None:
    cmds.append(f"discoverable-timeout {int(args.timeout_sec)}")
  cmds.append("discoverable " + ("on" if args.state.lower() == "on" else "off"))
  out, rc = btctl_interactive(cmds, timeout=args.timeout)
  if args.verbose:
    print(out)
  present, props, _ = btctl_show(timeout=DEFAULT_SHORT_TIMEOUT)
  ok = present and bool(props.get("Discoverable")) == (args.state.lower() == "on")
  if not ok:
    print("Failed to set discoverable", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print(f"Discoverable set to {args.state.lower()}")
  return ExitCodes.OK


def parse_devices_from_output(lines: List[str]) -> Dict[str, Dict[str, object]]:
  devices: Dict[str, Dict[str, object]] = {}
  for line in lines:
    s = line.strip()
    m = RE_EVT_NEW.search(s)
    if m:
      addr, name = m.group(1), m.group(2).strip()
      devices.setdefault(addr, {"addr": addr, "name": name, "rssi": None})
      continue
    m = RE_EVT_CHG_NAME.search(s)
    if m:
      addr, name = m.group(1), m.group(2).strip()
      devices.setdefault(addr, {"addr": addr, "name": name, "rssi": None})
      devices[addr]["name"] = name
      continue
    m = RE_EVT_CHG_RSSI.search(s)
    if m:
      addr, rssi = m.group(1), int(m.group(2))
      devices.setdefault(addr, {"addr": addr, "name": None, "rssi": None})
      devices[addr]["rssi"] = rssi
      continue
    m = RE_DEVICES_LINE.search(s)
    if m:
      addr, name = m.group(1), m.group(2).strip()
      devices.setdefault(addr, {"addr": addr, "name": name, "rssi": None})
      devices[addr]["name"] = name
  return devices


def enrich_device_flags(addrs: List[str], timeout_each: float = 1.0) -> Dict[str, Dict[str, object]]:
  details: Dict[str, Dict[str, object]] = {}
  for addr in addrs:
    # bluetoothctl info <addr>
    code, out = run(["bluetoothctl", "info", addr], timeout=timeout_each)
    info = {"paired": None, "trusted": None, "connected": None}
    if code == 0 and out:
      for line in out.splitlines():
        m = RE_INFO_BOOL.search(line.strip())
        if m:
          key = m.group(1).lower()
          info[key] = (m.group(2).lower() == "yes")
    details[addr] = info
  return details


def cmd_scan(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  if args.mode == "start":
    out, rc = btctl_interactive(["scan on"], timeout=args.timeout)
    if args.verbose:
      print(out)
    print("scan started")
    return ExitCodes.OK if rc == 0 else ExitCodes.CMD_FAILED
  elif args.mode == "stop":
    out, rc = btctl_interactive(["scan off"], timeout=args.timeout)
    if args.verbose:
      print(out)
    print("scan stopped")
    return ExitCodes.OK if rc == 0 else ExitCodes.CMD_FAILED
  else:  # once
    # interactive: enable agent, scan on, wait, list devices, scan off
    ensure_bluetoothctl()
    p = subprocess.Popen(["bluetoothctl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    start = time.time()
    lines: List[str] = []
    def w(cmd: str) -> None:
      if p.stdin:
        p.stdin.write(cmd + "\n")
        p.stdin.flush()
    try:
      w("agent on")
      w("default-agent")
      w("scan on")
      # Read lines for args.timeout seconds
      while time.time() - start < args.timeout:
        if p.stdout and p.stdout.readable():
          line = p.stdout.readline()
          if not line:
            break
          lines.append(line.rstrip())
        else:
          time.sleep(0.02)
      # finalize: list devices and turn scan off
      w("devices")
      time.sleep(0.2)
      # drain a bit
      drain_end = time.time() + 0.6
      while time.time() < drain_end:
        if p.stdout and p.stdout.readable():
          line = p.stdout.readline()
          if not line:
            break
          lines.append(line.rstrip())
        else:
          time.sleep(0.02)
      w("scan off")
      w("quit")
    finally:
      try:
        if p.stdin:
          p.stdin.close()
      except Exception:
        pass
      try:
        p.wait(timeout=1.0)
      except subprocess.TimeoutExpired:
        try:
          p.kill()
        except Exception:
          pass
    devices = parse_devices_from_output(lines)
    if args.with_info:
      flags = enrich_device_flags(list(devices.keys()), timeout_each=1.0)
      for addr, info in flags.items():
        devices[addr].update(info)
    dev_list = list(devices.values())
    if args.json:
      print(json.dumps(dev_list))
    else:
      for d in dev_list:
        name = d.get("name") or "(unknown)"
        rssi = d.get("rssi")
        flags = []
        for k in ("paired", "trusted", "connected"):
          if d.get(k) is True:
            flags.append(k)
        flags_str = (" [" + ",".join(flags) + "]") if flags else ""
        print(f"{d['addr']}  {name}  RSSI={rssi}{flags_str}")
    return ExitCodes.OK


def cmd_pair(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  addr = args.addr
  # best-effort sequence; include 'yes' to auto-accept confirm prompts
  seq = [
    "agent on",
    "default-agent",
    f"pair {addr}",
    "yes",
    f"trust {addr}",
    f"connect {addr}",
  ]
  out, rc = btctl_interactive(seq, timeout=args.timeout)
  if args.verbose:
    print(out)
  ok = ("Pairing successful" in out) or ("Device {addr} Connected: yes" in out)
  if not ok:
    # check info for paired state
    code, info_out = run(["bluetoothctl", "info", addr], timeout=2.0)
    if code == 0 and re.search(r"Paired:\s*yes", info_out, re.I):
      ok = True
  if not ok:
    print("Pairing failed or timed out", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print("Paired successfully")
  return ExitCodes.OK


def detect_agent_prompt(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
  """Scan output for agent prompts.
  Returns (event, value):
    event in {"confirm", "pin", "passkey"}
    value: for confirm: passkey string; for pin/passkey: None
  """
  for s in lines:
    ls = s.strip()
    # Common bluetoothctl agent prompts
    if re.search(r"Confirm\s+passkey\s+(\d+)", ls, re.I):
      m = re.search(r"Confirm\s+passkey\s+(\d+)", ls, re.I)
      return "confirm", (m.group(1) if m else None)
    if re.search(r"Request\s+PIN\s*code", ls, re.I) or re.search(r"Enter\s+PIN\s*code", ls, re.I):
      return "pin", None
    if re.search(r"Request\s+passkey", ls, re.I):
      return "passkey", None
  return None, None


def cmd_pair_interactive(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER

  p = subprocess.Popen(["bluetoothctl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
  start = time.time()
  lines: List[str] = []

  def w(cmd: str) -> None:
    if p.stdin:
      p.stdin.write(cmd + "\n")
      p.stdin.flush()

  try:
    w("agent on")
    w("default-agent")
    w(f"pair {args.addr}")
    # read until we either succeed, see prompt, or timeout
    while time.time() - start < args.timeout:
      if p.stdout and p.stdout.readable():
        line = p.stdout.readline()
        if not line:
          break
        lines.append(line.rstrip())
        if args.verbose:
          print(lines[-1])
        if "Pairing successful" in lines[-1]:
          print("Paired successfully")
          return ExitCodes.OK
        ev, val = detect_agent_prompt([lines[-1]])
        if ev == "confirm":
          payload = {"event": "confirm", "addr": args.addr, "passkey": val}
          print(json.dumps(payload) if args.json else f"CONFIRM {val}")
          return ExitCodes.NEED_CONFIRM
        if ev == "pin":
          payload = {"event": "pin", "addr": args.addr}
          print(json.dumps(payload) if args.json else f"PIN_REQUIRED")
          return ExitCodes.NEED_PIN
        if ev == "passkey":
          payload = {"event": "passkey", "addr": args.addr}
          print(json.dumps(payload) if args.json else f"PASSKEY_REQUIRED")
          return ExitCodes.NEED_PASSKEY
      else:
        time.sleep(0.02)
    return ExitCodes.TIMEOUT
  finally:
    try:
      if p.stdin:
        p.stdin.write("quit\n")
        p.stdin.flush()
        p.stdin.close()
    except Exception:
      pass
    try:
      p.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
      try:
        p.kill()
      except Exception:
        pass


def cmd_pair_complete(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  intent = None
  if args.confirm is not None:
    intent = ("confirm", args.confirm)
  elif args.pin is not None:
    intent = ("pin", args.pin)
  elif args.passkey is not None:
    intent = ("passkey", args.passkey)
  else:
    print("pair-complete requires --confirm or --pin or --passkey", file=sys.stderr)
    return ExitCodes.BAD_ARGS

  p = subprocess.Popen(["bluetoothctl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
  start = time.time()

  def w(cmd: str) -> None:
    if p.stdin:
      p.stdin.write(cmd + "\n")
      p.stdin.flush()

  try:
    w("agent on")
    w("default-agent")
    w(f"pair {args.addr}")
    # Wait for the prompt then respond
    lines: List[str] = []
    responded = False
    while time.time() - start < args.timeout:
      if p.stdout and p.stdout.readable():
        line = p.stdout.readline()
        if not line:
          break
        s = line.rstrip()
        lines.append(s)
        if args.verbose:
          print(s)
        if not responded:
          ev, val = detect_agent_prompt([s])
          if ev == "confirm" and intent[0] == "confirm":
            w("yes" if intent[1].lower() in ("1", "y", "yes", "true") else "no")
            responded = True
          elif ev == "pin" and intent[0] == "pin":
            # Supply PIN code directly
            w(str(intent[1]))
            responded = True
          elif ev == "passkey" and intent[0] == "passkey":
            w(str(intent[1]))
            responded = True
        if "Pairing successful" in s:
          # trust + connect best effort
          w(f"trust {args.addr}")
          w(f"connect {args.addr}")
          print("Paired successfully")
          return ExitCodes.OK
      else:
        time.sleep(0.02)
    # Verify as fallback
    code, info_out = run(["bluetoothctl", "info", args.addr], timeout=2.0)
    if code == 0 and re.search(r"Paired:\s*yes", info_out, re.I):
      print("Paired successfully")
      return ExitCodes.OK
    print("Pairing completion failed or timed out", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  finally:
    try:
      if p.stdin:
        p.stdin.write("quit\n")
        p.stdin.flush()
        p.stdin.close()
    except Exception:
      pass
    try:
      p.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
      try:
        p.kill()
      except Exception:
        pass


def cmd_trust(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  addr = args.addr
  on = args.state.lower() == "on"
  out, rc = btctl_interactive([f"{'trust' if on else 'untrust'} {addr}"], timeout=args.timeout)
  if args.verbose:
    print(out)
  code, info_out = run(["bluetoothctl", "info", addr], timeout=2.0)
  ok = (code == 0 and re.search(rf"Trusted:\s*{'yes' if on else 'no'}", info_out, re.I) is not None)
  if not ok:
    print("Failed to set trust", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print(f"Trusted set to {on}")
  return ExitCodes.OK


def cmd_connect(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  addr = args.addr
  out, rc = btctl_interactive([f"connect {addr}"], timeout=args.timeout)
  if args.verbose:
    print(out)
  code, info_out = run(["bluetoothctl", "info", addr], timeout=2.0)
  ok = (code == 0 and re.search(r"Connected:\s*yes", info_out, re.I))
  if not ok:
    print("Connect failed", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print("Connected")
  return ExitCodes.OK


def cmd_disconnect(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  addr = args.addr
  out, rc = btctl_interactive([f"disconnect {addr}"], timeout=args.timeout)
  if args.verbose:
    print(out)
  code, info_out = run(["bluetoothctl", "info", addr], timeout=2.0)
  ok = (code == 0 and re.search(r"Connected:\s*no", info_out, re.I))
  if not ok:
    print("Disconnect failed", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print("Disconnected")
  return ExitCodes.OK


def cmd_remove(args: argparse.Namespace) -> int:
  present, _, _ = btctl_show(timeout=args.timeout)
  if not present:
    print("No default controller available", file=sys.stderr)
    return ExitCodes.NO_CONTROLLER
  addr = args.addr
  out, rc = btctl_interactive([f"remove {addr}"], timeout=args.timeout)
  if args.verbose:
    print(out)
  code, info_out = run(["bluetoothctl", "info", addr], timeout=2.0)
  ok = (code != 0) or ("not available" in info_out.lower())
  if not ok:
    print("Remove failed", file=sys.stderr)
    return ExitCodes.CMD_FAILED
  print("Removed")
  return ExitCodes.OK


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Bluetooth helper for comma 3/3X on AGNOS")
  sub = p.add_subparsers(dest="cmd", required=True)

  # status
  ps = sub.add_parser("status", help="Show adapter status")
  ps.add_argument("--json", action="store_true", help="Output JSON")
  ps.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  ps.set_defaults(func=cmd_status)

  # set-powered
  pp = sub.add_parser("set-powered", help="Set adapter power state")
  pp.add_argument("state", choices=["on", "off"]) 
  pp.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  pp.add_argument("--verbose", action="store_true")
  pp.set_defaults(func=cmd_set_powered)

  # set-discoverable
  pd = sub.add_parser("set-discoverable", help="Set adapter discoverable")
  pd.add_argument("state", choices=["on", "off"]) 
  pd.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  pd.add_argument("--timeout-sec", type=int, default=None, help="Discoverable timeout in seconds when turning on")
  pd.add_argument("--verbose", action="store_true")
  pd.set_defaults(func=cmd_set_discoverable)

  # scan
  pscan = sub.add_parser("scan", help="Control or perform a scan")
  pscan.add_argument("mode", choices=["start", "stop", "once"], help="start/stop continuous scan, or scan once and list devices")
  pscan.add_argument("--timeout", type=float, default=DEFAULT_SCAN_TIMEOUT)
  pscan.add_argument("--with-info", action="store_true", help="Enrich results with paired/trusted/connected flags")
  pscan.add_argument("--json", action="store_true", help="Output JSON list for once mode")
  pscan.add_argument("--verbose", action="store_true")
  pscan.set_defaults(func=cmd_scan)

  # pair
  ppair = sub.add_parser("pair", help="Pair with a device and trust/connect")
  ppair.add_argument("addr", help="Device address (AA:BB:CC:DD:EE:FF)")
  ppair.add_argument("--timeout", type=float, default=DEFAULT_PAIR_TIMEOUT)
  ppair.add_argument("--verbose", action="store_true")
  ppair.set_defaults(func=cmd_pair)

  # pair-interactive (detect prompts, exit with NEED_* and JSON)
  ppi = sub.add_parser("pair-interactive", help="Pair and detect if PIN/passkey confirmation is needed")
  ppi.add_argument("addr")
  ppi.add_argument("--timeout", type=float, default=DEFAULT_PAIR_TIMEOUT)
  ppi.add_argument("--verbose", action="store_true")
  ppi.add_argument("--json", action="store_true")
  ppi.set_defaults(func=cmd_pair_interactive)

  # pair-complete (respond to prompts)
  ppc = sub.add_parser("pair-complete", help="Complete pairing by responding to PIN/passkey prompts")
  ppc.add_argument("addr")
  g = ppc.add_mutually_exclusive_group(required=True)
  g.add_argument("--confirm", choices=["yes", "no"], default=None)
  g.add_argument("--pin", default=None)
  g.add_argument("--passkey", default=None)
  ppc.add_argument("--timeout", type=float, default=DEFAULT_PAIR_TIMEOUT)
  ppc.add_argument("--verbose", action="store_true")
  ppc.set_defaults(func=cmd_pair_complete)

  # trust
  ptrust = sub.add_parser("trust", help="Set trust flag")
  ptrust.add_argument("addr")
  ptrust.add_argument("state", choices=["on", "off"]) 
  ptrust.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  ptrust.add_argument("--verbose", action="store_true")
  ptrust.set_defaults(func=cmd_trust)

  # connect
  pconn = sub.add_parser("connect", help="Connect to a device")
  pconn.add_argument("addr")
  pconn.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  pconn.add_argument("--verbose", action="store_true")
  pconn.set_defaults(func=cmd_connect)

  # disconnect
  pdis = sub.add_parser("disconnect", help="Disconnect from a device")
  pdis.add_argument("addr")
  pdis.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  pdis.add_argument("--verbose", action="store_true")
  pdis.set_defaults(func=cmd_disconnect)

  # remove
  prem = sub.add_parser("remove", help="Remove (unpair) a device")
  prem.add_argument("addr")
  prem.add_argument("--timeout", type=float, default=DEFAULT_SHORT_TIMEOUT)
  prem.add_argument("--verbose", action="store_true")
  prem.set_defaults(func=cmd_remove)

  return p


def main(argv: Optional[List[str]] = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  try:
    return int(args.func(args))
  except KeyboardInterrupt:
    return ExitCodes.TIMEOUT


if __name__ == "__main__":
  sys.exit(main())
