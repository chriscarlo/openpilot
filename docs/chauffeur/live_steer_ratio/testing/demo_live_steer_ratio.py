#!/usr/bin/env python3
"""
Demo script showing how LiveSteerRatio works for KIA EV6 users
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.params import Params

def main():
    params = Params()

    print("=== LiveSteerRatio Demo for KIA EV6 ===\n")

    # Show default behavior
    print("1. Default behavior (no LiveSteerRatio set):")
    try:
        params.remove("LiveSteerRatio")
    except:
        pass

    print("   - Vehicle uses default steer ratio: 13.43")
    print("   - paramsd will use 13.43 as base for calculations")
    print("   - Bounds: 6.71 (min) to 26.86 (max)")

    # Show setting to 0
    print("\n2. Setting LiveSteerRatio to 0:")
    params.put("LiveSteerRatio", "0.0")
    value = params.get("LiveSteerRatio").decode('utf-8')
    print(f"   - LiveSteerRatio = {value}")
    print("   - System interprets 0 as 'use vehicle default'")
    print("   - Still uses 13.43 for KIA EV6")

    # Show custom value
    print("\n3. Setting LiveSteerRatio to custom value (15.0):")
    params.put("LiveSteerRatio", "15.0")
    value = params.get("LiveSteerRatio").decode('utf-8')
    print(f"   - LiveSteerRatio = {value}")
    print("   - paramsd now uses 15.0 as base steer ratio")
    print("   - New bounds: 7.50 (min) to 30.00 (max)")
    print("   - This allows real-time tuning without restart")

    # Show GUI interaction
    print("\n4. GUI Interaction:")
    print("   - Go to Settings -> Vehicle -> Hyundai")
    print("   - Click 'Live Steering Ratio' -> Edit")
    print("   - Enter value between 0.0 and 25.0")
    print("   - 0 = use vehicle default (13.43 for EV6)")
    print("   - Changes take effect immediately")

    # Show practical use case
    print("\n5. Practical Use Case:")
    print("   - Stock EV6 steer ratio: 16.0 (now changed to 13.43)")
    print("   - If steering feels too sensitive: increase value (e.g., 15.0)")
    print("   - If steering feels too heavy: decrease value (e.g., 12.0)")
    print("   - Experiment to find your preferred feel")

    # Clean up
    try:
        params.remove("LiveSteerRatio")
    except:
        pass

    print("\n✓ Demo complete. LiveSteerRatio has been reset.")


if __name__ == "__main__":
    main()
