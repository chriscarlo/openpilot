from types import SimpleNamespace

from openpilot.selfdrive.selfdrived.selfdrived import compute_subsystem_status


def make_sm(alive: bool, freq_ok: bool, valid: bool) -> SimpleNamespace:
  return SimpleNamespace(
    alive={"svc": alive},
    freq_ok={"svc": freq_ok},
    valid={"svc": valid},
  )


def test_subsystem_readiness_ignores_valid_for_ignored_services():
  sm = make_sm(alive=True, freq_ok=True, valid=False)
  status = compute_subsystem_status(sm, ["svc"], {"svc"}, set())
  assert status == 2


def test_subsystem_readiness_keeps_valid_for_non_ignored_services():
  sm = make_sm(alive=True, freq_ok=True, valid=False)
  status = compute_subsystem_status(sm, ["svc"], set(), set())
  assert status == 1


def test_subsystem_readiness_ignores_freq_for_configured_services():
  sm = make_sm(alive=True, freq_ok=False, valid=True)
  status = compute_subsystem_status(sm, ["svc"], set(), {"svc"})
  assert status == 2


def test_subsystem_readiness_prefers_alive_and_freq_checks():
  sm_not_alive = make_sm(alive=False, freq_ok=True, valid=True)
  sm_not_freq = make_sm(alive=True, freq_ok=False, valid=True)
  assert compute_subsystem_status(sm_not_alive, ["svc"], {"svc"}, {"svc"}) == 0
  assert compute_subsystem_status(sm_not_freq, ["svc"], {"svc"}, set()) == 1
