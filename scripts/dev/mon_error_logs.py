#!/usr/bin/env python3
from __future__ import annotations

import cereal.messaging as messaging


def main() -> int:
  sm = messaging.SubMaster(['errorLogMessage'])
  with open('/tmp/error_logs.log', 'w') as f:
    while True:
      sm.update(1000)
      if sm.updated['errorLogMessage']:
        msg = sm['errorLogMessage']
        f.write(f"{msg}\n\n")
        f.flush()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

