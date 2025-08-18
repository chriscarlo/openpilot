#!/usr/bin/env python3
"""
Test script for EV6 dashboard speed limit functionality
Tests that the dashboard speed limit is correctly read from CAN messages
and made available to the speed limit controller
"""

import time
import sys
from cereal import messaging, car, custom
from openpilot.common.conversions import Conversions as CV
from openpilot.common.params import Params
from openpilot.selfdrive.test.helpers import with_processes


def test_speed_limit_reading():
    """Test that speed limit is correctly parsed from CAN messages"""
    print("Testing EV6 Dashboard Speed Limit Reading...")
    print("=" * 60)
    
    # Create sub sockets
    sm = messaging.SubMaster(['carStateSP', 'carState', 'liveMapDataSP'])
    
    # Create params
    params = Params()
    
    # Enable speed limit control for testing
    params.put_bool("SpeedLimitControl", True)
    params.put("SpeedLimitControlPolicy", "0")  # CAR_ONLY policy
    
    print("\nConfiguration:")
    print(f"  Speed Limit Control Enabled: {params.get_bool('SpeedLimitControl')}")
    print(f"  Speed Limit Control Policy: {params.get('SpeedLimitControlPolicy')}")
    print()
    
    # Monitor for speed limit data
    print("Monitoring for dashboard speed limit data...")
    print("(This will show speed limit values from the dashboard if available)")
    print()
    
    last_speed_limit = -1
    no_data_counter = 0
    max_no_data = 50  # 5 seconds at 10Hz
    
    try:
        while True:
            sm.update(100)  # 100ms timeout
            
            if sm.updated['carStateSP']:
                car_state_sp = sm['carStateSP']
                speed_limit_ms = car_state_sp.speedLimit
                
                # Convert to km/h for display
                if speed_limit_ms > 0:
                    speed_limit_kmh = speed_limit_ms * CV.MS_TO_KPH
                    
                    if speed_limit_ms != last_speed_limit:
                        print(f"[{time.strftime('%H:%M:%S')}] Dashboard Speed Limit Detected: "
                              f"{speed_limit_kmh:.0f} km/h ({speed_limit_ms:.2f} m/s)")
                        last_speed_limit = speed_limit_ms
                        no_data_counter = 0
                else:
                    no_data_counter += 1
                    if no_data_counter == max_no_data:
                        print(f"[{time.strftime('%H:%M:%S')}] No speed limit detected from dashboard")
                        no_data_counter = 0
            
            # Also check map data for comparison if available
            if sm.updated['liveMapDataSP']:
                map_data = sm['liveMapDataSP']
                if map_data.speedLimitValid:
                    map_speed_limit_kmh = map_data.speedLimit * CV.MS_TO_KPH
                    print(f"[{time.strftime('%H:%M:%S')}] Map Speed Limit: "
                          f"{map_speed_limit_kmh:.0f} km/h (for comparison)")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
        return


def test_speed_limit_controller_integration():
    """Test integration with speed limit controller"""
    print("\nTesting Speed Limit Controller Integration...")
    print("=" * 60)
    
    from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit_controller.speed_limit_controller import SpeedLimitController
    from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit_controller.common import Source
    from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
    
    # Create a minimal CP for testing
    class TestCP:
        def __init__(self):
            self.openpilotLongitudinalControl = True
            self.pcmCruise = False
    
    CP = TestCP()
    slc = SpeedLimitController(CP)
    
    # Create sub sockets
    sm = messaging.SubMaster(['carStateSP', 'carControl', 'carState'])
    
    print("\nWaiting for messages...")
    
    try:
        for i in range(100):  # Run for 10 seconds
            sm.update(100)
            
            if sm.updated['carStateSP'] and sm.updated['carState']:
                car_state = sm['carState']
                events_sp = EventsSP()
                
                # Update the speed limit controller
                slc.update(sm, car_state.vEgo, car_state.aEgo, car_state.cruiseState.speed, events_sp)
                
                if slc.speed_limit > 0:
                    speed_limit_kmh = slc.speed_limit * CV.MS_TO_KPH
                    print(f"[{time.strftime('%H:%M:%S')}] Speed Limit Controller:")
                    print(f"  Speed Limit: {speed_limit_kmh:.0f} km/h")
                    print(f"  Source: {slc.source}")
                    print(f"  State: {slc.state}")
                    print(f"  Offset: {slc.speed_limit_offset * CV.MS_TO_KPH:.1f} km/h")
                    print()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nIntegration test stopped by user")


def main():
    print("EV6 Dashboard Speed Limit Test Script")
    print("=" * 60)
    print("\nThis script tests the dashboard speed limit reading functionality")
    print("for the Hyundai EV6 and integration with the speed limit controller.\n")
    
    print("Options:")
    print("1. Test speed limit reading from CAN messages")
    print("2. Test speed limit controller integration")
    print("3. Run both tests")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        test_speed_limit_reading()
    elif choice == "2":
        test_speed_limit_controller_integration()
    elif choice == "3":
        test_speed_limit_reading()
        test_speed_limit_controller_integration()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we're running on the device or in simulation
    params = Params()
    
    if params.get("DongleId") is None:
        print("Warning: Not running on a comma device.")
        print("This test requires actual CAN data from an EV6.")
        print("For simulation, you'll need to replay a route with speed limit data.\n")
        
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    main()