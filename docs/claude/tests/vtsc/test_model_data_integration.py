#!/usr/bin/env python3
"""
Test the model data integration and apex acceleration functionality
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from enhanced_vtsc_integrated import EnhancedVisionTurnSpeedController

# Mock model data structure
class MockModelV2:
    def __init__(self, orientation_rates, velocities):
        self.orientationRate = MockOrientationRate(orientation_rates)
        self.velocity = MockVelocity(velocities)

class MockOrientationRate:
    def __init__(self, z_values):
        self.z = z_values

class MockVelocity:
    def __init__(self, x_values):
        self.x = x_values

def create_curve_scenario():
    """Create a realistic curve scenario using model data"""
    
    # Simulate a curve: approach -> apex -> exit
    # 33 points representing path ahead
    curve_progression = np.array([
        # Approach phase (points 0-10)
        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011,
        # Peak/apex phase (points 11-15) 
        0.012, 0.013, 0.014, 0.013, 0.012,
        # Exit phase (points 16-25)
        0.011, 0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002,
        # Straight again (points 26-32)
        0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.0001
    ])
    
    # Velocity prediction (fairly constant)
    velocities = np.full(33, 28.0)  # ~100 km/h
    
    # Convert curvature to orientation rate: orientation_rate = curvature * velocity
    orientation_rates = curve_progression * velocities
    
    return orientation_rates, velocities

def test_model_data_trajectory_planning():
    """Test the model data ingestion and trajectory planning"""
    
    print("MODEL DATA TRAJECTORY PLANNING TEST")
    print("="*60)
    print("Testing model data ingestion and curvature trajectory building")
    print()
    
    controller = EnhancedVisionTurnSpeedController()
    
    # Enable debug output for apex acceleration logic
    controller._debug_apex_conditions = True
    
    # Create realistic curve scenario
    orientation_rates, velocities = create_curve_scenario()
    
    # Create mock socket manager with model data
    mock_sm = {
        'modelV2': MockModelV2(orientation_rates, velocities)
    }
    
    v_ego = 28.0  # 100 km/h
    time = 0.0
    dt = 0.1
    
    print("Testing trajectory progression through curve...")
    print("Step  OrientRate  Velocity  Curvature  TargetSpeed(km/h)  ApexDetected  PastApex  AccelEmbargoLifted")
    print("-" * 100)
    
    trajectory_log = []
    
    # Simulate progression through curve by shifting the model data
    for step in range(20):  # 2 seconds at 10Hz
        
        # Shift the curve data to simulate progression
        if step < len(orientation_rates):
            # Roll the arrays to simulate moving through the curve
            current_orientation_rates = np.roll(orientation_rates, -step)
            current_velocities = np.roll(velocities, -step)
            
            # Update mock model data
            mock_sm['modelV2'] = MockModelV2(current_orientation_rates, current_velocities)
        
        # Enable debug only for key steps where apex acceleration should happen
        if step >= 8 and step <= 12:
            controller._debug_apex_conditions = True
        else:
            controller._debug_apex_conditions = False
        
        # Update controller
        result = controller.update(
            v_ego=v_ego,
            current_curvature=None,  # Not used with model data
            predicted_curvatures=None,  # Not used with model data
            distances=None,  # Not used with model data
            lateral_acc_limit=3.0,
            model_confidence=0.9,
            current_time=time,
            sm=mock_sm
        )
        
        # Log key metrics
        current_curvature = controller.curvature_trajectory[0] if len(controller.curvature_trajectory) > 0 else 0.0
        target_speed_kmh = controller.target_speeds_trajectory[0] * 3.6 if len(controller.target_speeds_trajectory) > 0 else 0.0
        
        print(f"{step:4d}    {current_orientation_rates[0]:8.3f}    {current_velocities[0]:6.1f}    "
              f"{current_curvature:8.4f}    {target_speed_kmh:8.0f}        "
              f"{'Yes' if result['apex_detected'] else 'No':11s}   "
              f"{'Yes' if result['past_apex'] else 'No':8s}  "
              f"{'Yes' if result['acceleration_embargo_lifted'] else 'No':18s}")
        
        # Apply acceleration to vehicle
        v_ego += result['a_target'] * dt
        v_ego = max(v_ego, 0)
        
        trajectory_log.append({
            'step': step,
            'v_ego': v_ego,
            'curvature': current_curvature,
            'a_target': result['a_target'],
            'apex_detected': result['apex_detected'],
            'past_apex': result['past_apex'],
            'embargo_lifted': result['acceleration_embargo_lifted'],
            'apex_accel_active': result['apex_acceleration_active']
        })
        
        time += dt
    
    print()
    print("ACCELERATION BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Analyze acceleration events
    acceleration_events = []
    for i, entry in enumerate(trajectory_log):
        if entry['a_target'] > 0.1:  # Significant acceleration
            acceleration_events.append({
                'step': entry['step'],
                'acceleration': entry['a_target'],
                'speed': entry['v_ego'],
                'curvature': entry['curvature'],
                'context': 'apex_acceleration' if entry['apex_accel_active'] else 'normal'
            })
    
    print(f"Total acceleration events detected: {len(acceleration_events)}")
    
    for event in acceleration_events:
        print(f"  Step {event['step']}: {event['acceleration']:.2f} m/s² "
              f"(speed: {event['speed']*3.6:.0f} km/h, curvature: {event['curvature']:.4f}) "
              f"[{event['context']}]")
    
    # Check for human-like behavior
    print()
    print("HUMAN-LIKE BEHAVIOR VALIDATION")
    print("-"*40)
    
    apex_events = [entry for entry in trajectory_log if entry['apex_detected']]
    past_apex_events = [entry for entry in trajectory_log if entry['past_apex']]
    embargo_lifted_events = [entry for entry in trajectory_log if entry['embargo_lifted']]
    
    print(f"✓ Apex detection: {'WORKING' if len(apex_events) > 0 else 'MISSING'}")
    print(f"✓ Past apex detection: {'WORKING' if len(past_apex_events) > 0 else 'MISSING'}")
    print(f"✓ Acceleration embargo lifted: {'WORKING' if len(embargo_lifted_events) > 0 else 'MISSING'}")
    print(f"✓ Apex acceleration events: {'WORKING' if len([e for e in acceleration_events if e['context'] == 'apex_acceleration']) > 0 else 'MISSING'}")
    
    # Overall assessment
    working_features = sum([
        len(apex_events) > 0,
        len(past_apex_events) > 0,
        len(embargo_lifted_events) > 0,
        len([e for e in acceleration_events if e['context'] == 'apex_acceleration']) > 0
    ])
    
    print(f"\nOverall model data integration: {working_features}/4 features working")
    
    if working_features >= 3:
        print("✓ SUCCESS: Model data integration and apex acceleration working!")
    else:
        print("✗ NEEDS WORK: Key features missing")
    
    return working_features >= 3

if __name__ == "__main__":
    success = test_model_data_trajectory_planning()