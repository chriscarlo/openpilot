#!/usr/bin/env python3
"""
Live test script to verify turn desires are working
Run this on the device while driving to see real-time desire status
"""

import time
from cereal import messaging
from openpilot.common.constants import CV

def main():
    sm = messaging.SubMaster(['modelV2', 'carState', 'controlsState'])
    
    print("Turn Desires Monitor")
    print("=" * 50)
    print("Threshold: 35 mph")
    print("Watching for turn desires...")
    print()
    
    last_desire = None
    
    while True:
        sm.update(0)
        
        if sm.updated['modelV2'] and sm.updated['carState']:
            v_ego_mph = sm['carState'].vEgo * CV.MS_TO_MPH
            left_blinker = sm['carState'].leftBlinker
            right_blinker = sm['carState'].rightBlinker
            
            # Get desire from model meta
            desire_state = sm['modelV2'].meta.desireState
            
            # Check which desire is highest probability
            max_prob = max(desire_state)
            desire_idx = desire_state.index(max_prob)
            
            desire_names = {
                0: "none",
                1: "turnLeft",
                2: "turnRight", 
                3: "laneChangeLeft",
                4: "laneChangeRight",
                5: "keepLeft",
                6: "keepRight",
                7: "unknown"
            }
            
            current_desire = desire_names.get(desire_idx, "unknown")
            
            # Print status line
            status = f"Speed: {v_ego_mph:5.1f} mph | "
            status += f"Blinker: {'L' if left_blinker else 'R' if right_blinker else '-'} | "
            status += f"Desire: {current_desire:15} | "
            status += f"Prob: {max_prob:.3f}"
            
            # Print update if desire changed
            if current_desire != last_desire:
                print(f"\n{'='*50}")
                print(f"DESIRE CHANGED: {last_desire} -> {current_desire}")
                print(f"{'='*50}")
            
            print(f"\r{status}", end='', flush=True)
            last_desire = current_desire
            
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")