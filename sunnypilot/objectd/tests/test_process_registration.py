from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import cereal.messaging as messaging

from openpilot.system.manager.process_config import managed_processes, object_hazard_enabled


def test_objectd_process_registered():
  assert "objectd" in managed_processes
  process = managed_processes["objectd"]
  assert process.name == "objectd"
  assert process.module == "sunnypilot.objectd.objectd"


def test_object_hazard_enabled_requires_onroad_param_and_real_car():
  params = Mock()
  cp = SimpleNamespace(notCar=False)

  params.get_bool.return_value = True
  assert object_hazard_enabled(True, params, cp) is True

  params.get_bool.return_value = False
  assert object_hazard_enabled(True, params, cp) is False

  params.get_bool.return_value = True
  assert object_hazard_enabled(False, params, cp) is False
  assert object_hazard_enabled(True, params, SimpleNamespace(notCar=True)) is False


def test_object_hazard_messages_expose_new_schema_fields():
  hazard_msg = messaging.new_message("objectHazardStateSP")
  hazard_msg.objectHazardStateSP.backend = "snpe_gpu"
  hazard_msg.objectHazardStateSP.hazardClass = "person"
  detections = hazard_msg.objectHazardStateSP.init("detections", 1)
  detections[0].className = "person"

  plan_msg = messaging.new_message("longitudinalPlanSP")
  plan_msg.longitudinalPlanSP.objectHazardControl.hazardClass = "person"
  plan_msg.longitudinalPlanSP.objectHazardControl.stopRequired = True

  assert str(hazard_msg.objectHazardStateSP.backend) == "snpe_gpu"
  assert str(hazard_msg.objectHazardStateSP.detections[0].className) == "person"
  assert str(plan_msg.longitudinalPlanSP.objectHazardControl.hazardClass) == "person"
  assert bool(plan_msg.longitudinalPlanSP.objectHazardControl.stopRequired) is True
