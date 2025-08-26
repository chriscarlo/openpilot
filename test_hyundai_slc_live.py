#!/usr/bin/env python3
"""
Test script to verify Hyundai speed limit data is actually being published
Run this while openpilot is running to see live data
"""

import sys
import time
sys.path.append('/projects/chauffeur/data/openpilot')

import cereal.messaging as messaging
from openpilot.common.params import Params

def main():
    print("=" * 70)
    print("LIVE HYUNDAI SPEED LIMIT DATA MONITOR")
    print("=" * 70)
    
    # Check current SLC policy setting
    params = Params()
    try:
        policy_raw = int(params.get("SpeedLimitControlPolicy"))
        policy_names = {0: "CAR_ONLY", 1: "MAP_ONLY", 2: "CAR_FIRST", 3: "MAP_FIRST", 4: "COMBINED"}
        print(f"\nCurrent SLC Policy: {policy_raw} ({policy_names.get(policy_raw, 'UNKNOWN')})")
    except:
        print("\nCould not read SLC policy")
    
    print("\nMonitoring live data (press Ctrl+C to stop)...")
    print("-" * 70)
    
    # Subscribe to relevant messages
    sm = messaging.SubMaster(['carStateSP', 'longitudinalPlanSP', 'liveMapDataSP'])
    
    last_car_speed_limit = None
    last_slc_speed_limit = None
    last_slc_source = None
    
    while True:
        sm.update(100)  # 100ms timeout
        
        # Check carStateSP speed limit (from Hyundai CAN)
        if sm.updated['carStateSP']:
            car_speed_limit = sm['carStateSP'].speedLimit
            if car_speed_limit != last_car_speed_limit:
                print(f"[carStateSP] speedLimit: {car_speed_limit:.3f} m/s ({car_speed_limit*3.6:.1f} km/h)")
                last_car_speed_limit = car_speed_limit
        
        # Check longitudinalPlanSP (what SLC is using)
        if sm.updated['longitudinalPlanSP']:
            slc = sm['longitudinalPlanSP'].slc
            slc_speed_limit = slc.speedLimit
            slc_source = slc.source
            
            if slc_speed_limit != last_slc_speed_limit or slc_source != last_slc_source:
                source_names = {0: "NONE", 1: "CAR", 2: "MAP"}
                source_name = source_names.get(slc_source, f"UNKNOWN({slc_source})")
                
                print(f"[longitudinalPlanSP] SLC speedLimit: {slc_speed_limit:.3f} m/s ({slc_speed_limit*3.6:.1f} km/h), source: {source_name}")
                
                if slc_speed_limit == 0 and last_car_speed_limit and last_car_speed_limit > 0:
                    print("  ⚠️  WARNING: carStateSP has speed limit but SLC shows 0!")
                
                last_slc_speed_limit = slc_speed_limit
                last_slc_source = slc_source
        
        # Check map data for comparison
        if sm.updated['liveMapDataSP']:
            if sm['liveMapDataSP'].speedLimitValid:
                map_speed_limit = sm['liveMapDataSP'].speedLimit
                print(f"[liveMapDataSP] speedLimit: {map_speed_limit:.3f} m/s ({map_speed_limit*3.6:.1f} km/h)")
        
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()