import pytest

from openpilot.common.realtime import Priority
from openpilot.selfdrive.controls import plannerd


def test_main_pins_plannerd_to_core6(monkeypatch):
  calls = []

  monkeypatch.setattr(plannerd, "config_realtime_process", lambda cores, priority: calls.append((cores, priority)))
  monkeypatch.setattr(plannerd.cloudlog, "info", lambda *args, **kwargs: None)

  class DummyParams:
    def get(self, *_args, **_kwargs):
      return b""

  monkeypatch.setattr(plannerd, "Params", DummyParams)

  def stop_after_affinity(*_args, **_kwargs):
    raise RuntimeError("stop_after_affinity")

  monkeypatch.setattr(plannerd.messaging, "log_from_bytes", stop_after_affinity)

  with pytest.raises(RuntimeError, match="stop_after_affinity"):
    plannerd.main()

  assert calls == [(6, Priority.CTRL_LOW)]
