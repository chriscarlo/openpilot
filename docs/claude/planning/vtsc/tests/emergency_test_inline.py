
import sys
sys.path.insert(0, '../implementation')
from enhanced_vtsc_integrated import EnhancedVisionTurnSpeedController, EmergencyLevel, VisionStatus
from emergency_scenarios_definition import EMERGENCY_SCENARIOS
import numpy as np

def test_scenario(scenario, controller):
    v_ego = scenario.v_ego_ms
    time = 0.0
    dt = 0.05
    distance_traveled = 0.0
    max_decel = 0.0
    max_level = EmergencyLevel.NORMAL
    reached_target = False
    intervention_triggered = False
    
    for i in range(200):  # 10 seconds max
        remaining_distance = scenario.distance_to_curve_m - distance_traveled
        if remaining_distance <= 0:
            break
            
        distances = np.array([10, 20, 30, 40, 50])
        curvatures = np.zeros(5)
        
        for j, d in enumerate(distances):
            if d >= remaining_distance:
                curvatures[j] = scenario.max_curvature
                
        result = controller.update(
            v_ego=v_ego,
            current_curvature=scenario.max_curvature if scenario.vision_status == VisionStatus.FULL_VISIBILITY else None,
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=3.0,
            model_confidence=scenario.confidence,
            current_time=time
        )
        
        v_ego += result['a_target'] * dt
        v_ego = max(v_ego, 0)
        distance_traveled += v_ego * dt
        time += dt
        
        max_decel = min(max_decel, result['a_target'])
        if result['emergency_level'].value > max_level.value:
            max_level = result['emergency_level']
            
        if result['intervention_required']:
            intervention_triggered = True
            
        if v_ego <= scenario.v_target_ms * 1.05:
            reached_target = True
            break
            
    success = reached_target and not intervention_triggered
    return success, max_decel, max_level, intervention_triggered

# Run all scenarios
controller = EnhancedVisionTurnSpeedController()
successes = 0
total = len(EMERGENCY_SCENARIOS)

for scenario in EMERGENCY_SCENARIOS:
    success, max_decel, max_level, intervention = test_scenario(scenario, controller)
    if success:
        successes += 1

print(f"Emergency Scenarios Success Rate: {successes}/{total} ({successes/total*100:.0f}%)")
print(f"✓ MEETS 80% CRITERIA" if successes/total >= 0.8 else "✗ NEEDS IMPROVEMENT")
