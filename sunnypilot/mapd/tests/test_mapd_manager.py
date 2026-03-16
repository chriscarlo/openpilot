import unittest
import json
from pathlib import Path
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
      alerts["Offroad_OSMUpdateRequired"],
      (True, "missing local map context"),
    )

  @patch("openpilot.sunnypilot.mapd.mapd_manager.get_files_for_cleanup", return_value=["/tmp/db"])
  def test_update_required_alert_still_respects_osm_local_toggle(self, _mock_cleanup):
    alerts = get_osm_offroad_alerts(StubMapData(None), True)

    self.assertEqual(
      alerts["Offroad_OSMUpdateRequired"],
      (True, "This alert will be cleared when new maps are downloaded."),
    )

  @patch("openpilot.sunnypilot.mapd.mapd_manager.get_files_for_cleanup", return_value=["/tmp/db"])
  def test_alerts_stay_off_when_osm_local_is_disabled(self, _mock_cleanup):
    alerts = get_osm_offroad_alerts(StubMapData("missing local map context"), False)

    self.assertEqual(
      alerts["Offroad_OSMUpdateRequired"],
      (False, ""),
    )

  @patch("openpilot.sunnypilot.mapd.mapd_manager.get_files_for_cleanup", return_value=["/tmp/db"])
  def test_update_required_and_local_issue_are_combined(self, _mock_cleanup):
    alerts = get_osm_offroad_alerts(StubMapData("missing local map context"), True)

    self.assertEqual(
      alerts["Offroad_OSMUpdateRequired"],
      (True, "This alert will be cleared when new maps are downloaded.\nmissing local map context"),
    )

  def test_osm_offroad_alert_keys_are_registered(self):
    repo_root = Path(__file__).resolve().parents[3]
    alerts_path = repo_root / "selfdrive" / "selfdrived" / "alerts_offroad.json"
    params_keys_path = repo_root / "common" / "params_keys.h"

    alerts_json = json.loads(alerts_path.read_text())
    params_keys_text = params_keys_path.read_text()
    alert_names = set(get_osm_offroad_alerts(StubMapData("issue"), True).keys())

    for alert_name in alert_names:
      self.assertIn(alert_name, alerts_json)
      self.assertIn(f"\"{alert_name}\"", params_keys_text)


if __name__ == "__main__":
  unittest.main()
