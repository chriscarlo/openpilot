import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-native compatibility coverage")


def test_windows_params_fallback_reads_defaults_and_round_trips_values():
  params = Params()

  assert params.get_default_value("Longitudinal.LiveTune.GapReclaimStrength") == pytest.approx(1.5)

  params.put("Longitudinal.LiveTune.GapReclaimStrength", 1.25)
  params.put_bool("VibePersonalityEnabled", True)

  assert params.get("Longitudinal.LiveTune.GapReclaimStrength") == pytest.approx(1.25)
  assert params.get_bool("VibePersonalityEnabled") is True


def test_windows_msgq_fallback_supports_in_process_pub_sub():
  import msgq

  pub = msgq.pub_sock("longitudinalWindowsCompat")
  sub = msgq.sub_sock("longitudinalWindowsCompat", conflate=False, timeout=0)

  pub.send(b"ok")

  assert sub.receive(non_blocking=True) == b"ok"


def test_windows_longitudinal_mpc_instantiates_without_native_acados():
  mpc = LongitudinalMpc()
  mpc.mode = "acc"
  mpc.set_cur_state(33.5, 0.0)
  mpc.control_leads = (SimpleNamespace(status=True, dRel=58.0, vLead=34.3, vRel=0.8, aLeadK=0.2), None)
  mpc.current_t_follow = 1.3
  mpc.last_v_cruise_clipped = np.array([33.5, 34.1])

  mpc._refresh_live_tune(100.0, force=True)

  assert mpc.get_gap_reclaim_floor() > 0.15
