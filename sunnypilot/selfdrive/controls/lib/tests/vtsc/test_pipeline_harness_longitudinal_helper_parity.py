from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_lead_helpers import (
  compute_relatch_required_decel,
  get_low_speed_launch_follow_factor,
)
from openpilot.sunnypilot.selfdrive.controls.lib.tests.vtsc.pipeline_harness import install_fake_long_mpc


MODULE = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc"


def test_fake_mpc_exports_the_shared_production_lead_helpers():
  import sys

  previous = sys.modules.get(MODULE)
  try:
    install_fake_long_mpc()
    fake = sys.modules[MODULE]
    assert fake.get_low_speed_launch_follow_factor is get_low_speed_launch_follow_factor
    assert fake.compute_relatch_required_decel is compute_relatch_required_decel
  finally:
    if previous is None:
      sys.modules.pop(MODULE, None)
    else:
      sys.modules[MODULE] = previous


def test_reviewer_golden_launch_and_relatch_vectors():
  launch = SimpleNamespace(status=True, dRel=20.0, vLead=4.0, vRel=2.0, aLeadK=0.0)
  assert get_low_speed_launch_follow_factor(2.0, launch, 1.5) == pytest.approx(0.8666666667)

  braking = SimpleNamespace(status=True, dRel=100.0, vLead=18.0, vRel=-2.0, aLeadK=-5.0)
  assert compute_relatch_required_decel(20.0, braking, 1.5) == pytest.approx(1.5822784810)


@pytest.mark.parametrize(
  "lead",
  [
    SimpleNamespace(status=True, dRel=100.0, vLead=0.0, vRel=0.0, aLeadK=0.0),
    SimpleNamespace(status=True, dRel=40.0, vLead=-3.0, vRel=-23.0, aLeadK=0.0),
    SimpleNamespace(status=True, dRel=100.0, vLead=18.0, vRel=-2.0, aLeadK=-5.0),
  ],
)
def test_shared_helpers_preserve_zero_oncoming_and_decelerating_values(lead):
  launch = get_low_speed_launch_follow_factor(20.0, lead, 1.5)
  relatch = compute_relatch_required_decel(20.0, lead, 1.5)
  assert 0.0 <= launch <= 1.0
  assert relatch >= 0.0
