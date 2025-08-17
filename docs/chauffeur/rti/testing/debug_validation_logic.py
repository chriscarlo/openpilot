#!/usr/bin/env python3
"""
Debug the validation logic that prevents rti_threats from being populated
"""

import time
import cereal.messaging as messaging


def test_ui_submaster_behavior():
    """Test if the issue is with UI SubMaster validation."""
    print("UI SubMaster Validation Debug")
    print("=" * 40)

    # Create identical SubMaster to what UI uses
    # From ui.cc: sm = std::make_unique<SubMaster>(std::vector<const char*>{
    #   "modelV2", "controlsState", ... "rtiStateSP", ...
    ui_services = [
        "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
        "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
        "wideRoadCameraState", "managerState", "selfdriveState", "longitudinalPlan",
        "modelManagerSP", "selfdriveStateSP", "longitudinalPlanSP", "backupManagerSP",
        "carControl", "liveMapDataSP", "rtiStateSP"
    ]

    # Test simple SubMaster first
    simple_sm = messaging.SubMaster(['rtiStateSP'])

    # Test UI-style SubMaster
    ui_sm = messaging.SubMaster(ui_services)

    # Publisher
    pm = messaging.PubMaster(['rtiStateSP'])

    print("Testing SubMaster validation behavior...")
    print()

    for i in range(5):
        # Create message
        msg = messaging.new_message('rtiStateSP', valid=True)
        rti_state = msg.rtiStateSP

        rti_state.timeStamp = int(time.time() * 1e9)
        rti_state.threatAhead = True
        rti_state.threatDistanceM = 300.0
        rti_state.source = f'validation_test_{i}'
        rti_state.apiStatus = 'connected'

        # Add threat
        rti_state.init('threats', 1)
        threat_msg = rti_state.threats[0]
        threat_msg.id = f'val_police_{i:03d}'
        threat_msg.type = 'police'
        threat_msg.distance = 300.0
        threat_msg.direction = 'ahead'
        threat_msg.confidence = 0.85

        # Send message
        pm.send('rtiStateSP', msg)
        print(f"SENT {i+1}: {threat_msg.id}")

        time.sleep(0.1)

        # Test simple SubMaster
        simple_sm.update(0)
        simple_valid = simple_sm.valid['rtiStateSP']
        simple_updated = simple_sm.updated['rtiStateSP']

        # Test UI SubMaster
        ui_sm.update(0)
        ui_valid = ui_sm.valid['rtiStateSP']
        ui_updated = ui_sm.updated['rtiStateSP']

        print(f"  Simple SubMaster: valid={simple_valid}, updated={simple_updated}")
        print(f"  UI SubMaster:     valid={ui_valid}, updated={ui_updated}")

        if simple_valid and simple_updated:
            received = simple_sm['rtiStateSP']
            print(f"    Simple received: {len(received.threats)} threats")

        if ui_valid and ui_updated:
            received = ui_sm['rtiStateSP']
            print(f"    UI received: {len(received.threats)} threats")

        print()
        time.sleep(1)

    print("KEY FINDINGS:")
    print("- If Simple works but UI doesn't: SubMaster service list issue")
    print("- If both fail: Message format/validation issue")
    print("- If both work: HUD updateRTIThreats() logic issue")


if __name__ == "__main__":
    test_ui_submaster_behavior()
