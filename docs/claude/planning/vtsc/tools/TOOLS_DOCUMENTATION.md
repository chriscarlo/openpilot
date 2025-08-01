# VTSC Debug and Analysis Tools

## Overview
This directory contains tools for debugging, analyzing, and tuning the Enhanced VTSC system.

## Tools

### 1. debug_intervention_trigger.py
**Purpose**: Analyzes when and why the system triggers driver intervention

**Key Features**:
- Traces emergency level calculations step-by-step
- Identifies intervention trigger conditions
- Helps tune intervention thresholds
- Validates intervention logic

**Usage**:
```bash
python3 debug_intervention_trigger.py
```

**Output Example**:
```
Debugging First Update - blind_hairpin_exceeds_fov
================================================================
Initial conditions:
  v_ego: 54.0 km/h
  Distance: 40.0m
  Max curvature: 0.050 (r=20m)
  Vision: CURVE_EXCEEDS_FOV

Manual calculations:
  v_safe for curve: 28 km/h
  Required decel to v_safe at 10m: -9.45 m/s² (0.96g)
  Expected emergency level: INTERVENTION
```

**Future Analysis Tools to Add**:

### 2. performance_analyzer.py (Planned)
**Purpose**: Analyze controller performance over time
- Track deceleration patterns
- Identify comfort violations
- Generate performance reports
- Compare different parameter settings

### 3. parameter_tuner.py (Planned)
**Purpose**: Interactive parameter tuning tool
- Adjust emergency thresholds
- Test anticipation timing
- Visualize deceleration profiles
- Export optimal parameters

### 4. scenario_replayer.py (Planned)
**Purpose**: Replay real-world scenarios
- Load recorded drives
- Test controller responses
- Compare with actual driver behavior
- Identify improvement areas

### 5. vision_simulator.py (Planned)
**Purpose**: Simulate various vision conditions
- Test occlusion handling
- Validate extrapolation logic
- Stress test edge cases
- Generate vision degradation scenarios

## Debug Workflow

### 1. Identifying Issues
```python
# Run comprehensive tests
python3 ../tests/test_integrated_comprehensive.py > test_output.log

# Look for failures
grep "FAILED\|INTERVENTION" test_output.log

# Debug specific scenario
python3 debug_intervention_trigger.py --scenario "blind_hairpin_exceeds_fov"
```

### 2. Analyzing Intervention Triggers
```python
# Check why intervention was triggered
result = controller.update(...)
if result['intervention_required']:
    print(f"Intervention at distance: {result['critical_distance']}")
    print(f"Required decel: {result['required_decel']}")
    print(f"Current level: {result['emergency_level']}")
```

### 3. Tuning Parameters
```python
# Test different deceleration limits
DECEL_LIMITS[EmergencyLevel.CRITICAL] = -6.5  # More aggressive
results = run_all_tests()
compare_performance(baseline_results, results)
```

## Common Debug Scenarios

### Premature Interventions
1. Check distance calculations
2. Verify vision status handling
3. Review emergency level thresholds
4. Analyze deceleration requirements

### Insufficient Deceleration
1. Check jerk limiting
2. Verify level transition timing
3. Review safety margins
4. Test emergency escalation

### Comfort Issues
1. Analyze deceleration profiles
2. Check anticipation timing
3. Review jerk limits
4. Test transition smoothness

## Adding New Debug Tools

Template for new debug tools:
```python
#!/usr/bin/env python3
"""
Debug tool for [specific purpose]
"""

import sys
sys.path.append('..')

from implementation.enhanced_vtsc_integrated import EnhancedVisionTurnSpeedController
from tests.emergency_scenarios_definition import EMERGENCY_SCENARIOS

def debug_[feature]():
    """Main debug function"""
    controller = EnhancedVisionTurnSpeedController()
    
    # Debug logic here
    
    print("Debug results...")

if __name__ == "__main__":
    debug_[feature]()
```

## Performance Metrics to Track

### Real-time Metrics
- Update computation time
- Memory usage
- CPU utilization

### Behavioral Metrics
- Deceleration smoothness
- Level transition frequency
- Intervention rate
- Anticipation accuracy

### Comfort Metrics
- Average jerk
- Maximum deceleration
- Comfort violations
- Passenger feedback scores