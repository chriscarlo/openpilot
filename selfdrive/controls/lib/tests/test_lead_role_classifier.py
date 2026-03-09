from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.lead_role_classifier import LeadRoleClassifier


def _make_lead(*, status=True, d_rel=80.0, y_rel=0.0, v_rel=-2.0, model_prob=0.95, radar=False, track_id=-1):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    yRel=y_rel,
    vRel=v_rel,
    aRel=0.0,
    vLead=30.0 + v_rel,
    dPath=0.0,
    vLat=0.0,
    vLeadK=30.0 + v_rel,
    aLeadK=0.0,
    fcw=False,
    aLeadTau=1.5,
    modelProb=model_prob,
    radar=radar,
    radarTrackId=track_id,
  )


class TestLeadRoleClassifier:
  def test_debug_logging_disabled_by_default(self):
    c = LeadRoleClassifier()
    lead0 = _make_lead(y_rel=0.25)
    lead1 = _make_lead(status=False)
    _ctrl0, _ctrl1, dbg = c.classify(v_ego=30.0, lead0=lead0, lead1=lead1, now=1.0)

    assert dbg["debug_log_enabled"] is False

  def test_adjacent_lead_is_awareness_not_control(self):
    c = LeadRoleClassifier()
    lead0 = _make_lead(y_rel=-11.0)
    lead1 = _make_lead(status=False)
    ctrl0, ctrl1, dbg = c.classify(v_ego=30.0, lead0=lead0, lead1=lead1, now=1.0)

    assert dbg["gate_active"] is True
    assert dbg["roles"]["lead0"] == LeadRoleClassifier.ADJ_RIGHT
    assert not ctrl0.status
    assert not ctrl1.status

  def test_center_lead_is_control(self):
    c = LeadRoleClassifier()
    lead0 = _make_lead(y_rel=0.25)
    lead1 = _make_lead(status=False)
    ctrl0, ctrl1, dbg = c.classify(v_ego=30.0, lead0=lead0, lead1=lead1, now=1.0)

    assert dbg["roles"]["lead0"] == LeadRoleClassifier.CENTER_CONTROL
    assert ctrl0.status
    assert not ctrl1.status

  def test_duplicate_pair_dedupes_second_slot(self):
    c = LeadRoleClassifier()
    lead0 = _make_lead(d_rel=70.0, y_rel=0.1, v_rel=-1.5)
    lead1 = _make_lead(d_rel=70.8, y_rel=0.15, v_rel=-1.4)
    ctrl0, ctrl1, dbg = c.classify(v_ego=28.0, lead0=lead0, lead1=lead1, now=1.0)

    assert dbg["duplicate_pair"] is True
    assert dbg["dropped_slot"] == 1
    assert ctrl0.status is True
    assert ctrl1.status is False

  def test_low_speed_bypass_keeps_control_lead(self):
    c = LeadRoleClassifier()
    lead0 = _make_lead(y_rel=9.0)
    lead1 = _make_lead(status=False)
    ctrl0, _ctrl1, dbg = c.classify(v_ego=4.0, lead0=lead0, lead1=lead1, now=1.0)

    assert dbg["low_speed_bypass"] is True
    assert dbg["gate_active"] is False
    assert ctrl0.status is True

  def test_cutin_promotion_promotes_to_control(self):
    c = LeadRoleClassifier()
    lead0_far = _make_lead(d_rel=45.0, y_rel=3.0, v_rel=-1.5)
    lead0_cutin = _make_lead(d_rel=42.0, y_rel=1.8, v_rel=-1.7)
    lead1 = _make_lead(status=False)

    # Prime state as adjacent first.
    c.classify(v_ego=30.0, lead0=lead0_far, lead1=lead1, now=1.0)
    ctrl0, _ctrl1, dbg = c.classify(v_ego=30.0, lead0=lead0_cutin, lead1=lead1, now=1.2)

    assert dbg["cutin_promoted"]["lead0"] is True
    assert dbg["roles"]["lead0"] == LeadRoleClassifier.CENTER_CONTROL
    assert ctrl0.status is True
