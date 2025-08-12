#!/usr/bin/env python3
"""
ForceOnroad Development Mode Guide and Test
=============================================

This script demonstrates how to use the ForceOnroad feature for HUD/UI development
when the comma3x device is not connected to a vehicle.

WARNING: This feature bypasses critical safety checks and should ONLY be used
for development purposes when the device is NOT in a vehicle.

Usage:
    python force_onroad_guide.py enable   # Enable forced onroad mode
    python force_onroad_guide.py disable  # Disable forced onroad mode
    python force_onroad_guide.py status   # Check current status
"""

import sys
from openpilot.common.params import Params

def enable_force_onroad():
    """Enable ForceOnroad development mode"""
    params = Params()

    print("WARNING: Enabling ForceOnroad Development Mode")
    print("=" * 60)
    print("This mode bypasses critical safety systems!")
    print("ONLY use this for UI/HUD development when:")
    print("• Device is NOT in a vehicle")
    print("• Device is NOT connected to car")
    print("• Used for development/testing only")
    print("=" * 60)

    response = input("Type 'CONFIRM' to enable ForceOnroad mode: ")
    if response != "CONFIRM":
        print("ForceOnroad mode NOT enabled")
        return

    params.put_bool("ForceOnroad", True)
    print("ForceOnroad mode enabled")
    print("The device should now show onroad UI/HUD")
    print("hardwared will start onroad processes")
    print("\nTo disable: python force_onroad_guide.py disable")

def disable_force_onroad():
    """Disable ForceOnroad development mode"""
    params = Params()

    params.put_bool("ForceOnroad", False)
    print("ForceOnroad mode disabled")
    print("Device returning to normal offroad state")

def check_status():
    """Check current ForceOnroad status"""
    params = Params()

    force_onroad = params.get_bool("ForceOnroad")
    force_onroad_active = params.get_bool("ForceOnroadActive")
    is_onroad = params.get_bool("IsOnroad")
    is_offroad = params.get_bool("IsOffroad")

    print("ForceOnroad Status:")
    print(f"  ForceOnroad parameter: {'Enabled' if force_onroad else 'Disabled'}")
    print(f"  ForceOnroadActive: {'Active' if force_onroad_active else 'Inactive'}")
    print(f"  IsOnroad: {'True' if is_onroad else 'False'}")
    print(f"  IsOffroad: {'True' if is_offroad else 'False'}")

    if force_onroad and not force_onroad_active:
        print("\nForceOnroad is enabled but not active")
        print("This means hardwared is not running or hasn't processed the change yet")

    if force_onroad_active:
        print("\nDEVELOPMENT MODE ACTIVE - NO VEHICLE CONNECTED")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['enable', 'disable', 'status']:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'enable':
        enable_force_onroad()
    elif command == 'disable':
        disable_force_onroad()
    elif command == 'status':
        check_status()

if __name__ == "__main__":
    main()
