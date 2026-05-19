from __future__ import annotations

import time


class ObjectdInferenceScheduler:
  def __init__(self, detector_hz: float, infer_budget_ms: float, phase_sec: float):
    self.interval_sec = 1.0 / max(0.1, detector_hz)
    self.infer_budget_sec = max(0.001, infer_budget_ms / 1000.0)
    self.next_run_time = time.monotonic() + max(0.0, phase_sec)
    self.skip_next_due_to_overrun = False

  def should_run(self, now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    if now < self.next_run_time:
      return False
    self.next_run_time = now + self.interval_sec
    if self.skip_next_due_to_overrun:
      self.skip_next_due_to_overrun = False
      return False
    return True

  def record_runtime(self, runtime_sec: float) -> None:
    if runtime_sec > self.infer_budget_sec:
      self.skip_next_due_to_overrun = True
