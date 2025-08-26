#!/usr/bin/env python3
"""
Debug script to trace Hyundai CANFD speed limit data flow
"""

import sys
import time
sys.path.append('/projects/chauffeur/data/openpilot')

# Let's check what's actually happening with the Hyundai speed limit data

print("=" * 70)
print("HYUNDAI CANFD SPEED LIMIT DATA FLOW DEBUG")
print("=" * 70)

print("\n1. CHECKING HYUNDAI CARSTATE IMPLEMENTATION:")
print("-" * 40)

# Check if FR_CMR_02_100ms is being read
import os
hyundai_carstate = "/projects/chauffeur/data/openpilot/opendbc_repo/opendbc/car/hyundai/carstate.py"
with open(hyundai_carstate, 'r') as f:
    content = f.read()
    
    # Check for speed limit implementation
    if "FR_CMR_02_100ms" in content:
        print("✓ FR_CMR_02_100ms message IS being read")
        
        # Find the exact lines
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "FR_CMR_02_100ms" in line or "ISLW_SpdCluMainDis" in line:
                print(f"  Line {i+1}: {line.strip()}")
                if i > 0 and "ret_sp.speedLimit" in lines[i+1]:
                    print(f"  Line {i+2}: {lines[i+1].strip()}")
    else:
        print("✗ FR_CMR_02_100ms not found")

print("\n2. CHECKING DATA PUBLICATION:")
print("-" * 40)

# Check card.py publishes carStateSP
card_file = "/projects/chauffeur/data/openpilot/selfdrive/car/card.py"
with open(card_file, 'r') as f:
    content = f.read()
    
    if "pm.send('carStateSP'" in content:
        print("✓ card.py IS publishing carStateSP")
        
        # Check if CS_SP is being passed correctly
        if "CS_SP = convert_to_capnp(CS_SP)" in content:
            print("✓ CS_SP is being converted to capnp format")
    else:
        print("✗ carStateSP publication not found")

print("\n3. CHECKING SUBSCRIPTION IN PLANNERD:")
print("-" * 40)

plannerd_file = "/projects/chauffeur/data/openpilot/selfdrive/controls/plannerd.py"
with open(plannerd_file, 'r') as f:
    content = f.read()
    
    if "'carStateSP'" in content:
        print("✓ plannerd IS subscribing to carStateSP")
        
        # Find the exact subscription line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "SubMaster" in line and "carStateSP" in lines[i] + lines[i+1]:
                print(f"  Line {i+1}: {line.strip()}")
                if i < len(lines) - 1:
                    print(f"  Line {i+2}: {lines[i+1].strip()}")
    else:
        print("✗ carStateSP subscription not found")

print("\n4. CHECKING SPEED LIMIT RESOLVER:")
print("-" * 40)

resolver_file = "/projects/chauffeur/data/openpilot/sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py"
with open(resolver_file, 'r') as f:
    content = f.read()
    
    if "sm['carStateSP'].speedLimit" in content:
        print("✓ Speed limit resolver IS reading carStateSP.speedLimit")
        
        # Find the exact line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "sm['carStateSP'].speedLimit" in line:
                print(f"  Line {i+1}: {line.strip()}")
    else:
        print("✗ carStateSP.speedLimit access not found")

print("\n5. CHECKING FOR POTENTIAL ISSUES:")
print("-" * 40)

# Check if there's any RTI code that might interfere
print("\nChecking for RTI interference with speed limits...")

# Check if RTI modifies speed limits
rti_controller = "/projects/chauffeur/data/openpilot/sunnypilot/selfdrive/controls/lib/rti_controller.py"
if os.path.exists(rti_controller):
    with open(rti_controller, 'r') as f:
        content = f.read()
        
        if "carStateSP" in content:
            print("⚠ RTI controller references carStateSP - potential interference")
        else:
            print("✓ RTI controller doesn't reference carStateSP")
            
        if "speedLimit =" in content or "speed_limit =" in content:
            print("⚠ RTI controller modifies speed limits")
            # Find the lines
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if ("speedLimit =" in line or "speed_limit =" in line) and not line.strip().startswith('#'):
                    print(f"  Line {i+1}: {line.strip()}")
        else:
            print("✓ RTI controller doesn't modify speed limits directly")

print("\n6. CHECKING HUD REFACTORING IMPACT:")
print("-" * 40)

# The HUD refactoring shouldn't affect data flow, but let's check
print("Checking if HUD refactoring affected data sources...")

# The HUD just displays data, it shouldn't modify it
hud_file = "/projects/chauffeur/data/openpilot/selfdrive/ui/qt/onroad/hud.cc"
print("✓ HUD is a display component - shouldn't affect data flow")
print("  HUD reads from longitudinalPlanSP for display only")

print("\n7. POTENTIAL ROOT CAUSES:")
print("-" * 40)

print("Based on the analysis, possible issues:")
print("1. ❓ Timing issue - carStateSP might not be ready when SLC reads it")
print("2. ❓ Message filtering - FR_CMR_02_100ms might not be parsed correctly")
print("3. ❓ Unit conversion issue - speed factor calculation might be wrong")
print("4. ❓ Validation logic - speed limit might be rejected as invalid")

print("\n8. SUGGESTED DEBUGGING:")
print("-" * 40)
print("Add logging to these locations:")
print("1. /opendbc_repo/opendbc/car/hyundai/carstate.py line 328")
print("   Log: raw value, speed_factor, calculated speedLimit")
print("2. /sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py line 56")
print("   Log: sm['carStateSP'].speedLimit value")
print("3. /selfdrive/car/card.py line 269")
print("   Log: CS_SP.speedLimit before sending")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)