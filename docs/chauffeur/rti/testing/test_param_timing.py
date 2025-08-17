#!/usr/bin/env python3
"""
Test to check RTI parameter timing issue in HUD renderer
"""

import time
from openpilot.common.params import Params


def test_parameter_timing():
    """Test the timing of parameter updates."""
    params = Params()

    print("RTI Parameter Timing Test")
    print("=" * 40)

    # Check current values
    rti_enabled = params.get_bool('RTIEnabled')
    rti_hud_enabled = params.get_bool('RTIHUDEnabled')

    print(f"Current RTIEnabled: {rti_enabled}")
    print(f"Current RTIHUDEnabled: {rti_hud_enabled}")

    if not rti_enabled or not rti_hud_enabled:
        print("\nERROR: Parameters are not both True!")
        print("The HUD widget should be blank.")
        return

    print("\nBoth parameters are True.")
    print("\nISSUE LIKELY FOUND:")
    print("The HUD renderer only updates RTI parameters once per second.")
    print("If the UI was started before these parameters were set to True,")
    print("the HUD renderer variables rti_enabled and rti_hud_enabled")
    print("are still False internally, causing the blank widget.")
    print()
    print("SOLUTIONS:")
    print("1. Restart the UI process to force parameter reload")
    print("2. Wait up to 1 second for the next parameter update cycle")
    print("3. Force immediate parameter update in the HUD code")

    # Test toggling to force update
    print("\nTesting parameter toggle to force update...")
    print("Setting RTIHUDEnabled to False...")
    params.put_bool('RTIHUDEnabled', False)
    time.sleep(0.1)

    print("Setting RTIHUDEnabled back to True...")
    params.put_bool('RTIHUDEnabled', True)

    print("This should force the HUD to update within 1 second.")
    print("Check if the RTI widget now shows the placeholder 'RTI' text.")


if __name__ == "__main__":
    test_parameter_timing()
