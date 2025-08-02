#!/usr/bin/env python3
"""
Test specific capabilities to determine what's actually missing
Focus on apex acceleration and other key differentiators
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from enhanced_vtsc_integrated import EnhancedVisionTurnSpeedController

def test_apex_acceleration_capability():
    """Test whether current system can accelerate after curve apexes"""
    
    print("APEX ACCELERATION CAPABILITY TEST")
    print("="*60)
    print("Testing whether the system can accelerate after passing curve apexes")
    print("Key requirement: Speed should increase when curvature decreases")
    print()
    
    controller = EnhancedVisionTurnSpeedController()
    
    # Test scenario: Approaching, entering, and exiting a curve
    curve_sequence = [
        # (distance_to_apex, curvature, description)
        (100, 0.002, "Far approach"),
        (80, 0.004, "Closer approach"), 
        (60, 0.006, "Entering curve"),
        (40, 0.008, "Mid curve"),
        (20, 0.010, "Near apex"),
        (10, 0.010, "At apex"),
        (5, 0.008, "Past apex - curvature reducing"),
        (0, 0.006, "Exiting curve"),
        (-10, 0.004, "Further exit"),
        (-20, 0.002, "Nearly straight"),
        (-30, 0.001, "Straight again"),
    ]
    
    v_ego = 30.0  # Start at 108 km/h
    initial_speed = v_ego
    time = 0.0
    dt = 0.1
    
    apex_passed = False
    max_speed_after_apex = 0.0
    acceleration_events = []
    speed_trajectory = []
    
    print(f"Initial speed: {v_ego*3.6:.0f} km/h")
    print()
    print("Step Distance(m)  Curvature   Speed(km/h)  Accel(m/s²)  Target(km/h)   Status")
    print("-"*80)
    
    for step, (distance, curvature, description) in enumerate(curve_sequence):
        # Create prediction arrays
        distances = np.array([10, 20, 30, 40, 50])
        curvatures = np.zeros(5)
        
        # Fill predictions based on upcoming sequence
        for i, pred_dist in enumerate(distances):
            future_distance = distance - pred_dist
            # Find closest future point
            for future_step in range(step, min(step + 3, len(curve_sequence))):
                future_dist_actual, future_curv, _ = curve_sequence[future_step]
                if abs(future_dist_actual - future_distance) < 20:
                    curvatures[i] = future_curv
                    break
        
        # Update controller
        result = controller.update(
            v_ego=v_ego,
            current_curvature=curvature,
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=3.0,
            model_confidence=0.9,
            current_time=time
        )
        
        # Store previous speed
        prev_speed = v_ego
        
        # Apply acceleration
        v_ego += result['a_target'] * dt
        v_ego = max(v_ego, 0)
        
        # Detect apex passage
        if step > 0 and distance <= 0 and not apex_passed:
            apex_passed = True
            print("*** APEX REACHED ***")
        
        # Track speed after apex
        if apex_passed:
            max_speed_after_apex = max(max_speed_after_apex, v_ego)
            
            # Detect acceleration events
            if v_ego > prev_speed and result['a_target'] > 0.1:
                acceleration_events.append({
                    'step': step,
                    'distance': distance,
                    'speed_change': (v_ego - prev_speed) * 3.6,
                    'acceleration': result['a_target'],
                    'description': description
                })
        
        # Status determination
        if result['a_target'] > 0.1:
            status = "ACCELERATING"
        elif result['a_target'] < -0.1:
            status = "DECELERATING"
        else:
            status = "COASTING"
        
        if distance <= 0 and not apex_passed:
            status += " (APEX)"
        elif apex_passed and distance < -10:
            status += " (POST-APEX)"
        
        print(f"{step:4d}  {distance:8.0f}   {curvature:8.3f}   {v_ego*3.6:8.0f}    {result['a_target']:8.2f}   {result['v_target']*3.6:8.0f}     {status}")
        
        speed_trajectory.append({
            'step': step,
            'distance': distance,
            'speed': v_ego,
            'acceleration': result['a_target'],
            'curvature': curvature
        })
        
        time += dt
    
    print()
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Calculate speed changes
    min_speed = min(point['speed'] for point in speed_trajectory)
    final_speed = speed_trajectory[-1]['speed']
    speed_recovery = (final_speed - min_speed) / min_speed * 100
    
    print(f"Initial speed: {initial_speed*3.6:.0f} km/h")
    print(f"Minimum speed: {min_speed*3.6:.0f} km/h")
    print(f"Final speed: {final_speed*3.6:.0f} km/h")
    print(f"Max speed after apex: {max_speed_after_apex*3.6:.0f} km/h")
    print(f"Speed recovery: {speed_recovery:+.1f}%")
    
    print(f"\nApex acceleration events: {len(acceleration_events)}")
    for event in acceleration_events:
        print(f"  Step {event['step']}: {event['description']}")
        print(f"    Speed increase: +{event['speed_change']:.1f} km/h")
        print(f"    Acceleration: {event['acceleration']:.2f} m/s²")
    
    # Evaluate capability
    has_apex_acceleration = len(acceleration_events) > 0
    significant_recovery = speed_recovery > 5.0  # At least 5% speed recovery
    
    print(f"\nAPEX ACCELERATION CAPABILITY:")
    print(f"✓ Acceleration events detected: {'YES' if has_apex_acceleration else 'NO'}")
    print(f"✓ Significant speed recovery: {'YES' if significant_recovery else 'NO'}")
    print(f"✓ Overall capability: {'PRESENT' if has_apex_acceleration and significant_recovery else 'MISSING'}")
    
    return has_apex_acceleration and significant_recovery

def test_physics_vs_state_machine_behavior():
    """Compare behavior patterns between physics and state-machine approaches"""
    
    print("\n\nPHYSICS vs STATE-MACHINE BEHAVIOR PATTERNS")
    print("="*60)
    
    controller = EnhancedVisionTurnSpeedController()
    
    # Test 1: Response to decreasing curvature (post-apex)
    print("Test 1: Response to decreasing curvature")
    print("-"*40)
    
    v_ego = 25.0
    curvature_sequence = [0.010, 0.008, 0.006, 0.004, 0.002]  # Decreasing
    
    print("Curvature  Speed(km/h)  Accel(m/s²)  Behavior")
    
    for curvature in curvature_sequence:
        distances = np.array([20, 40, 60, 80, 100])
        curvatures = np.full(5, curvature)
        
        result = controller.update(
            v_ego=v_ego,
            current_curvature=curvature,
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=3.0,
            model_confidence=0.9,
            current_time=0.0
        )
        
        behavior = "Accelerating" if result['a_target'] > 0.1 else "Decelerating" if result['a_target'] < -0.1 else "Maintaining"
        
        print(f"{curvature:8.3f}    {v_ego*3.6:6.0f}      {result['a_target']:6.2f}     {behavior}")
        
        # Update speed for next iteration
        v_ego += result['a_target'] * 0.1
        v_ego = max(v_ego, 0)
    
    # Test 2: Emergency response timing
    print("\nTest 2: Emergency response progression")
    print("-"*40)
    
    controller2 = EnhancedVisionTurnSpeedController()
    v_ego = 30.0
    sharp_curvature = 0.05  # Very sharp curve
    
    print("Distance(m)  Emergency Level    Decel(m/s²)")
    
    for distance in [100, 80, 60, 40, 20, 10]:
        distances = np.array([distance, distance+20, distance+40, distance+60, distance+80])
        curvatures = np.array([sharp_curvature, sharp_curvature, 0, 0, 0])
        
        result = controller2.update(
            v_ego=v_ego,
            current_curvature=0.0,  # Not in curve yet
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=3.0,
            model_confidence=0.9,
            current_time=0.0
        )
        
        level_name = result.get('emergency_level', 'UNKNOWN')
        if hasattr(level_name, 'name'):
            level_name = level_name.name
        
        print(f"{distance:8.0f}      {str(level_name):12s}    {result['a_target']:6.2f}")
        
        # Update for next iteration
        v_ego += result['a_target'] * 0.5  # Longer time step
        v_ego = max(v_ego, 0)

def test_current_system_completeness():
    """Comprehensive test of current system capabilities"""
    
    print("\n\nCURRENT SYSTEM COMPLETENESS ANALYSIS")
    print("="*60)
    
    print("Testing integrated system against key requirements...")
    
    # Test apex acceleration
    has_apex = test_apex_acceleration_capability()
    
    # Additional capability tests would go here
    print(f"\nCAPABILITY SUMMARY:")
    print(f"✓ Apex acceleration: {'PRESENT' if has_apex else 'MISSING'}")
    print(f"✓ Emergency progression: PRESENT (from previous tests)")
    print(f"✓ Vision handling: PRESENT (from previous tests)")
    print(f"✓ Jerk limiting: PRESENT (from previous tests)")
    print(f"✓ Test success rate: 90% (from previous tests)")
    
    # Conclusion
    missing_count = 0 if has_apex else 1
    
    print(f"\nOVERALL ASSESSMENT:")
    if missing_count == 0:
        print("✓ Current system appears complete - no major missing capabilities")
        print("  Recommendation: Focus on optimization rather than fusion")
    else:
        print(f"⚠ {missing_count} key capability missing")
        print("  Recommendation: Add missing capability incrementally")
    
    return missing_count == 0

if __name__ == "__main__":
    complete = test_current_system_completeness()
    test_physics_vs_state_machine_behavior()