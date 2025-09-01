#!/usr/bin/env python3
"""
Force onroad mode helper.

Usage:
  scripts/dev/force_onroad.py on        # set ForceOnroad=1 (and ensure MTSCEnabled=1)
  scripts/dev/force_onroad.py off       # set ForceOnroad=0
  scripts/dev/force_onroad.py toggle    # flip ForceOnroad
  scripts/dev/force_onroad.py status    # print ForceOnroad and started status

Optional flags:
  --no-mtsc      # do not touch MTSCEnabled on 'on'

Note: manager/hardwared read this param live; no restart required.
You can also export env var FORCE_ONROAD=1 before starting manager to force onroad.
"""
from __future__ import annotations

import argparse
import sys

try:
  # Use openpilot package imports as in codebase
  from openpilot.common.params import Params
  import cereal.messaging as messaging
except Exception as e:
  print("Failed to import openpilot modules: %s" % e)
  sys.exit(2)


def get_started() -> bool:
  try:
    sm = messaging.SubMaster(['deviceState'])
    sm.update(500)
    return bool(sm['deviceState'].started)
  except Exception:
    return False


def main() -> int:
  ap = argparse.ArgumentParser(description="Force onroad mode helper")
  ap.add_argument('cmd', choices=['on', 'off', 'toggle', 'status'], help='action to perform')
  ap.add_argument('--no-mtsc', action='store_true', help='(deprecated) no-op; MTSC publisher removed')
  args = ap.parse_args()

  p = Params()

  if args.cmd == 'on':
    p.put_bool('ForceOnroad', True)
    # MTSCEnabled no longer used
    print('ForceOnroad=1 set')
  elif args.cmd == 'off':
    p.put_bool('ForceOnroad', False)
    print('ForceOnroad=0 set')
  elif args.cmd == 'toggle':
    cur = p.get_bool('ForceOnroad')
    p.put_bool('ForceOnroad', not cur)
    print(f'ForceOnroad toggled to {int(not cur)}')
  elif args.cmd == 'status':
    pass

  cur = p.get_bool('ForceOnroad')
  started = get_started()
  print(f'ForceOnroad={int(cur)}, started={int(started)}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
