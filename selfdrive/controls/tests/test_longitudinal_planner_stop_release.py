from types import SimpleNamespace

from cereal import car

from openpilot.selfdrive.controls.lib.longitudinal_planner import should_release_stop_for_lead_launch


def test_stop_release_for_pulling_away_lead_does_not_require_starting_state() -> None:
  CP = car.CarParams.new_message(vEgoStarting=0.1, startingState=False)
  lead = SimpleNamespace(status=True, vRel=0.25, vLead=0.25, aRel=0.0, aLeadK=0.0)

  assert should_release_stop_for_lead_launch(
    CP,
    standstill=True,
    v_ego=0.0,
    a_target=0.15,
    lead_source="lead0",
    control_leads=[lead, None],
  )


def test_stop_release_for_pulling_away_lead_requires_positive_pullaway_signal() -> None:
  CP = car.CarParams.new_message(vEgoStarting=0.1, startingState=False)
  stationary_lead = SimpleNamespace(status=True, vRel=0.0, vLead=0.0, aRel=0.0, aLeadK=0.0)

  assert not should_release_stop_for_lead_launch(
    CP,
    standstill=True,
    v_ego=0.0,
    a_target=0.15,
    lead_source="lead0",
    control_leads=[stationary_lead, None],
  )


def test_stop_release_for_pulling_away_lead_requires_standstill_and_active_lead() -> None:
  CP = car.CarParams.new_message(vEgoStarting=0.1, startingState=False)
  lead = SimpleNamespace(status=True, vRel=0.25, vLead=0.25, aRel=0.0, aLeadK=0.0)

  assert not should_release_stop_for_lead_launch(
    CP,
    standstill=False,
    v_ego=0.0,
    a_target=0.15,
    lead_source="lead0",
    control_leads=[lead, None],
  )
  assert not should_release_stop_for_lead_launch(
    CP,
    standstill=True,
    v_ego=0.0,
    a_target=0.15,
    lead_source="cruise",
    control_leads=[lead, None],
  )
