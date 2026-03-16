import unittest
from unittest.mock import patch

from openpilot.sunnypilot.mapd.mapd_manager import get_osm_offroad_alerts


class StubMapData:
  def __init__(self, issue: str | None):
    self.issue = issue

  def get_local_map_health_issue(self) -> str | None:
    return self.issue


class TestMapdManagerAlerts(unittest.TestCase):
  @patch("openpilot.sunnypilot.mapd.mapd_manager.get_files_for_cleanup", return_value=[])
  def test_map_health_issue_triggers_loud_alert(self, _mock_cleanup):
    alerts = get_osm_offroad_alerts(StubMapData("missing local map context"), True)

    self.assertEqual(
      alerts["Offroad_OSMDataUnavailable"],
      (True, "missing local map context"),
    )
    self.assertEqual(
      alerts["Offroad_OSMUpdateRequired"],
      (False, "This alert will be cleared when new maps are downloaded."),
    )

  @patch("openpilot.sunnypilot.mapd.mapd_manager.get_files_for_cleanup", return_value=["/tmp/db"])
  def test_update_required_alert_still_respects_osm_local_toggle(self, _mock_cleanup):
    alerts = get_osm_offroad_alerts(StubMapData(None), True)

    self.assertEqual(
      alerts["Offroad_OSMUpdateRequired"],
      (True, "This alert will be cleared when new maps are downloaded."),
    )
    self.assertEqual(
      alerts["Offroad_OSMDataUnavailable"],
      (False, ""),
    )

  @patch("openpilot.sunnypilot.mapd.mapd_manager.get_files_for_cleanup", return_value=["/tmp/db"])
  def test_alerts_stay_off_when_osm_local_is_disabled(self, _mock_cleanup):
    alerts = get_osm_offroad_alerts(StubMapData("missing local map context"), False)

    self.assertEqual(
      alerts["Offroad_OSMUpdateRequired"],
      (False, "This alert will be cleared when new maps are downloaded."),
    )
    self.assertEqual(
      alerts["Offroad_OSMDataUnavailable"],
      (False, ""),
    )


if __name__ == "__main__":
  unittest.main()
