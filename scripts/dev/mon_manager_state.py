#!/usr/bin/env python3
from __future__ import annotations

import time
import cereal.messaging as messaging


def main() -> int:
  sm = messaging.SubMaster(['managerState', 'deviceState'], poll='managerState')
  start = time.time()
  with open('/tmp/manager_state.log', 'w') as f:
    while True:
      sm.update(100)
      procs = sm['managerState'].processes
      started = sm['deviceState'].started
      running = ', '.join([f"{p.name}:{int(p.running)}" for p in procs])
      f.write(f"t+{time.time()-start:6.1f}s started={int(started)} | {running}\n")
      f.flush()
      time.sleep(1.0)

  return 0


if __name__ == '__main__':
  raise SystemExit(main())

