#!/usr/bin/env python3
"""
Evidence-based analysis: Test each VTSC system individually
to determine actual performance differences
"""

import numpy as np
import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, '../reference/')

# Import test scenarios
from test_integrated_expanded import TEST_SCENARIOS, simulate_scenario
from emergency_scenarios_definition import EMERGENCY_SCENARIOS

# Mock dependencies for physics-based system
class MockParams:
    def get_bool(self, key):
        return True

class MockCP:
    def __init__(self):
        self.steerRatio = 15.0
        self.wheelbase = 2.7

class MockSM:
    def __init__(self, steer_angle_deg=0.0, gas_pressed=False):
        self.data = {
            'carState': MockCarState(steer_angle_deg, gas_pressed),
            'modelV2': MockModelV2(),
            'lateralPlan': MockLateralPlan()
        }
        self.valid = {'carState': True, 'modelV2': True, 'lateralPlan': True}
    
    def __getitem__(self, key):
        return self.data[key]

class MockCarState:
    def __init__(self, steer_angle_deg=0.0, gas_pressed=False):
        self.steeringAngleDeg = steer_angle_deg
        self.gasPressed = gas_pressed

class MockModelV2:
    def __init__(self):
        # Create mock model data with 33 points
        n_points = 33
        self.orientationRate = MockArray([0.01] * n_points)  # Small orientation rate
        self.velocity = MockArray([25.0] * n_points)  # Constant velocity prediction
        
        # Mock lane lines
        self.laneLines = [MockLaneLine() for _ in range(4)]
        self.laneLineProbs = [0.8, 0.9, 0.9, 0.8]  # Good confidence
        self.laneLineStds = [0.1, 0.1, 0.1, 0.1]   # Low std

class MockArray:
    def __init__(self, data):
        self.data = data
        self.z = data  # For orientation rate
        self.x = data  # For velocity
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class MockLaneLine:
    def __init__(self):
        self.t = list(range(33))
        self.x = list(range(0, 165, 5))  # 0 to 160m in 5m steps
        self.y = [0.0] * 33  # Straight line

class MockLateralPlan:
    def __init__(self):
        self.psis = list(range(50))
        self.dPathPoints = [0.0] * 50

# Test adapter for physics-based system
class PhysicsBasedAdapter:
    def __init__(self):
        # Import the physics-based controller
        from vision_turn_controller_physics_based import VisionTurnController
        
        self.controller = VisionTurnController(MockCP())
        self.name = "Physics-Based VTSC"
    
    def simulate_scenario(self, scenario, dt=0.1, debug=False):
        """Simulate using physics-based controller"""
        
        # Convert scenario to SI units
        v_ego = scenario.v_ego_kph / 3.6
        curve_curvature = 1.0 / scenario.curve_radius_m if scenario.curve_radius_m > 0 else 0.0
        distance_remaining = scenario.distance_to_curve_m
        
        # Physics target speed calculation
        lateral_limit = 3.0
        v_target_physics = np.sqrt(lateral_limit / curve_curvature) if curve_curvature > 0 else 100.0
        
        time = 0.0
        max_decel = 0.0
        reached_target = False
        started_anticipation = False
        apex_acceleration_seen = False
        max_speed = v_ego
        
        trajectory = []
        
        while distance_remaining > -10 and time < 10.0:
            # Create mock model data with curve ahead
            sm = MockSM()
            
            # Update model data based on distance to curve
            n_points = 33
            if distance_remaining > 0:
                # Scale orientation rate based on approaching curve
                curve_factor = max(0, 1 - distance_remaining / 100.0)
                orientation_rates = [curve_curvature * curve_factor * v_ego] * n_points
                velocities = [v_ego] * n_points
            else:
                # In the curve
                orientation_rates = [curve_curvature * v_ego] * n_points
                velocities = [v_ego] * n_points
            
            sm.data['modelV2'].orientationRate = MockArray(orientation_rates)
            sm.data['modelV2'].velocity = MockArray(velocities)
            
            # Update controller
            self.controller.update(
                sm=sm,
                enabled=True,
                v_ego=v_ego,
                a_ego=0.0,
                v_cruise_setpoint=scenario.v_ego_kph / 3.6,
                v_cruise_cluster_setpoint=scenario.v_ego_kph / 3.6
            )
            
            # Get results
            a_target = self.controller.a_target
            is_active = self.controller.is_active
            
            # Apply acceleration
            v_ego += a_target * dt
            v_ego = max(v_ego, 0)
            
            # Update distance
            distance_remaining -= v_ego * dt
            time += dt
            
            # Track metrics
            if a_target < max_decel:
                max_decel = a_target
            
            if v_ego > max_speed:
                max_speed = v_ego
                if distance_remaining < 0:  # After apex
                    apex_acceleration_seen = True
            
            if is_active and not started_anticipation:
                started_anticipation = True
            
            if v_ego <= v_target_physics * 1.05 and not reached_target:
                reached_target = True
            
            trajectory.append({
                'time': time,
                'speed': v_ego,
                'distance': distance_remaining,
                'decel': a_target,
                'active': is_active
            })
            
            if debug and int(time * 10) % 10 == 0:
                print(f"t={time:.1f}s: v={v_ego*3.6:5.0f}km/h, "
                      f"d={distance_remaining:4.0f}m, "
                      f"a={a_target:6.2f}m/s², "
                      f"active={is_active}")
        
        # Determine success
        success = False
        if scenario.category == "normal":
            success = (reached_target and abs(max_decel) <= 2.45 and started_anticipation)
        elif scenario.category == "challenging":
            success = (v_ego <= v_target_physics * 1.1 and abs(max_decel) <= 6.0)
        else:  # edge
            success = v_ego <= v_target_physics * 1.2 or abs(max_decel) >= 5.5
        
        return {
            'success': success,
            'category': scenario.category,
            'max_decel_g': abs(max_decel) / 9.81,
            'final_speed_error': (v_ego - v_target_physics) / v_target_physics if v_target_physics > 0 else 0,
            'apex_acceleration': apex_acceleration_seen,
            'anticipation': started_anticipation,
            'max_speed': max_speed
        }

# Test adapter for state-machine system
class StateMachineAdapter:
    def __init__(self):
        # Import the state-machine controller
        from vision_turn_controller import VisionTurnController
        
        self.controller = VisionTurnController(MockCP())
        self.name = "State-Machine VTSC"
    
    def simulate_scenario(self, scenario, dt=0.1, debug=False):
        """Simulate using state-machine controller"""
        
        # Convert scenario to SI units
        v_ego = scenario.v_ego_kph / 3.6
        curve_curvature = 1.0 / scenario.curve_radius_m if scenario.curve_radius_m > 0 else 0.0
        distance_remaining = scenario.distance_to_curve_m
        
        # Physics target speed calculation
        lateral_limit = 3.0
        v_target_physics = np.sqrt(lateral_limit / curve_curvature) if curve_curvature > 0 else 100.0
        
        time = 0.0
        max_decel = 0.0
        reached_target = False
        started_anticipation = False
        max_emergency_level = 0
        intervention_triggered = False
        
        trajectory = []
        
        while distance_remaining > -10 and time < 10.0:
            # Create mock model with polynomial path
            sm = MockSM()
            
            # Calculate lateral acceleration based on curve ahead
            if distance_remaining > 0:
                lat_acc = min(curve_curvature * v_ego**2 * (100 - distance_remaining) / 100, curve_curvature * v_ego**2)
            else:
                lat_acc = curve_curvature * v_ego**2
            
            # Update controller
            self.controller.update(
                sm=sm,
                enabled=True,
                v_ego=v_ego,
                a_ego=0.0,
                v_cruise_setpoint=scenario.v_ego_kph / 3.6
            )
            
            # Get results
            a_target = self.controller.a_target
            is_active = self.controller.is_active
            
            # Check for emergency features
            if hasattr(self.controller, 'emergency_level'):
                if self.controller.emergency_level.value > max_emergency_level:
                    max_emergency_level = self.controller.emergency_level.value
            
            if hasattr(self.controller, 'intervention_required'):
                if self.controller.intervention_required:
                    intervention_triggered = True
            
            # Apply acceleration
            v_ego += a_target * dt
            v_ego = max(v_ego, 0)
            
            # Update distance
            distance_remaining -= v_ego * dt
            time += dt
            
            # Track metrics
            if a_target < max_decel:
                max_decel = a_target
            
            if is_active and not started_anticipation:
                started_anticipation = True
            
            if v_ego <= v_target_physics * 1.05 and not reached_target:
                reached_target = True
            
            trajectory.append({
                'time': time,
                'speed': v_ego,
                'distance': distance_remaining,
                'decel': a_target,
                'active': is_active
            })
            
            if debug and int(time * 10) % 10 == 0:
                print(f"t={time:.1f}s: v={v_ego*3.6:5.0f}km/h, "
                      f"d={distance_remaining:4.0f}m, "
                      f"a={a_target:6.2f}m/s², "
                      f"active={is_active}")
        
        # Determine success
        success = False
        if scenario.category == "normal":
            success = (reached_target and abs(max_decel) <= 2.45 and started_anticipation)
        elif scenario.category == "challenging":
            success = (v_ego <= v_target_physics * 1.1 and abs(max_decel) <= 6.0)
        else:  # edge
            success = v_ego <= v_target_physics * 1.2 or intervention_triggered
        
        return {
            'success': success,
            'category': scenario.category,
            'max_decel_g': abs(max_decel) / 9.81,
            'final_speed_error': (v_ego - v_target_physics) / v_target_physics if v_target_physics > 0 else 0,
            'anticipation': started_anticipation,
            'max_emergency_level': max_emergency_level,
            'intervention': intervention_triggered
        }

def compare_systems():
    """Compare both systems on the same test scenarios"""
    
    print("EVIDENCE-BASED VTSC SYSTEM COMPARISON")
    print("="*80)
    print("Testing both systems individually on identical scenarios")
    print("Goal: Identify which specific elements are actually superior")
    print()
    
    # Initialize adapters
    physics_adapter = PhysicsBasedAdapter()
    state_adapter = StateMachineAdapter()
    
    # Test both systems
    physics_results = []
    state_results = []
    
    print(f"Testing {physics_adapter.name}...")
    for scenario in TEST_SCENARIOS[:5]:  # Test first 5 scenarios
        try:
            result = physics_adapter.simulate_scenario(scenario)
            result['scenario'] = scenario.name
            physics_results.append(result)
            print(f"  {scenario.name}: {'✓' if result['success'] else '✗'}")
        except Exception as e:
            print(f"  {scenario.name}: ERROR - {e}")
            physics_results.append({'success': False, 'scenario': scenario.name, 'error': str(e)})
    
    print(f"\nTesting {state_adapter.name}...")
    for scenario in TEST_SCENARIOS[:5]:  # Test first 5 scenarios
        try:
            result = state_adapter.simulate_scenario(scenario)
            result['scenario'] = scenario.name
            state_results.append(result)
            print(f"  {scenario.name}: {'✓' if result['success'] else '✗'}")
        except Exception as e:
            print(f"  {scenario.name}: ERROR - {e}")
            state_results.append({'success': False, 'scenario': scenario.name, 'error': str(e)})
    
    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    physics_success = sum(1 for r in physics_results if r.get('success', False))
    state_success = sum(1 for r in state_results if r.get('success', False))
    total = len(physics_results)
    
    print(f"\nSuccess Rates:")
    print(f"Physics-Based: {physics_success}/{total} ({physics_success/total*100:.0f}%)")
    print(f"State-Machine: {state_success}/{total} ({state_success/total*100:.0f}%)")
    
    # Detailed comparison
    print(f"\nDetailed Comparison:")
    print("Scenario".ljust(25) + "Physics".ljust(10) + "State".ljust(10) + "Winner")
    print("-" * 55)
    
    physics_wins = 0
    state_wins = 0
    
    for i in range(min(len(physics_results), len(state_results))):
        p_res = physics_results[i]
        s_res = state_results[i]
        scenario_name = p_res.get('scenario', f'Scenario_{i}')
        
        p_success = p_res.get('success', False)
        s_success = s_res.get('success', False)
        
        if p_success and not s_success:
            winner = "Physics"
            physics_wins += 1
        elif s_success and not p_success:
            winner = "State"
            state_wins += 1
        elif p_success and s_success:
            winner = "Both"
        else:
            winner = "Neither"
        
        print(f"{scenario_name[:24].ljust(25)}{'✓' if p_success else '✗'.ljust(9)}{'✓' if s_success else '✗'.ljust(9)}{winner}")
    
    print(f"\nWin Summary:")
    print(f"Physics-only wins: {physics_wins}")
    print(f"State-only wins: {state_wins}")
    print(f"Both successful: {min(physics_success, state_success) - max(0, physics_success + state_success - total)}")
    
    # Conclusions
    print(f"\n" + "="*80)
    print("EVIDENCE-BASED CONCLUSIONS")
    print("="*80)
    
    if physics_success > state_success:
        print(f"✓ Physics-Based system shows superior performance")
        print(f"  Recommendation: Use physics-based system as foundation")
    elif state_success > physics_success:
        print(f"✓ State-Machine system shows superior performance")
        print(f"  Recommendation: Use state-machine system as foundation")
    else:
        print(f"≈ Both systems show equivalent performance")
        print(f"  Recommendation: Identify specific superior elements from each")
    
    print(f"\nNext steps:")
    print(f"1. Analyze failure modes of each system")
    print(f"2. Identify specific technical advantages")
    print(f"3. Extract only proven superior elements")
    print(f"4. Build improved system incrementally")
    
    return physics_results, state_results

if __name__ == "__main__":
    physics_results, state_results = compare_systems()