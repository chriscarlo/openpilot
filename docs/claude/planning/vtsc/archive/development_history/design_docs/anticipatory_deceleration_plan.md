# VTSC Anticipatory Deceleration Enhancement Plan

## Problem Statement

The current VTSC implements a "just in time" deceleration profile that reaches target speeds precisely when physics demands it. While mathematically optimal, this creates psychological discomfort for human passengers who expect to reach safe speeds with a comfort buffer before critical points in curves.

### Current Behavior
- Decelerates to reach target speed exactly at the point where lateral acceleration would exceed limits
- Maintains continuous deceleration through curve entry
- Perfect from a physics standpoint, uncomfortable from a human perspective

### Desired Behavior
- Reach target speed 1-4 seconds BEFORE it's physically necessary
- Maintain stable speed through the comfort buffer zone
- Seamlessly transition to existing acceleration behavior on curve exit
- Scale anticipation time based on multiple comfort factors

## Design Principles

1. **Elegance through Simplicity**: Start with simple, understandable calculations that can be enhanced
2. **Human-Centered Metrics**: Base timing on psychological comfort, not just physics
3. **Graceful Degradation**: Handle edge cases smoothly without abrupt transitions
4. **Measurable Comfort**: Define metrics we can test and validate

## Proposed Solution Architecture

### Core Concept: Anticipation Time Function

```python
anticipation_time = f(speed, curvature, speed_delta, visibility, comfort_preference)
```

### Key Factors Analysis

#### 1. Speed Factor (Primary)
- **Rationale**: Higher speeds require more mental processing time
- **Function**: Non-linear scaling, possibly logarithmic or sigmoid
- **Range**: 0.5x to 2.0x multiplier
- **Examples**:
  - 20 mph → 0.7x (less anticipation needed)
  - 45 mph → 1.0x (baseline)
  - 70 mph → 1.5x (more anticipation)
  - 90 mph → 1.8x (approaching maximum)

#### 2. Curvature Severity Factor
- **Rationale**: Sharper curves are more psychologically demanding
- **Function**: Based on predicted lateral acceleration
- **Range**: 0.8x to 1.5x multiplier
- **Examples**:
  - Gentle curve (0.5 m/s²) → 0.8x
  - Medium curve (1.5 m/s²) → 1.0x
  - Sharp curve (2.5 m/s²) → 1.3x
  - Hairpin (3.0 m/s²) → 1.5x

#### 3. Speed Delta Factor
- **Rationale**: Larger speed changes need more preparation time
- **Function**: Proportional to speed reduction percentage
- **Range**: 0.9x to 1.4x multiplier
- **Examples**:
  - 10% reduction → 0.9x
  - 25% reduction → 1.0x
  - 40% reduction → 1.2x
  - 50%+ reduction → 1.4x

#### 4. Visibility/Confidence Factor
- **Rationale**: Better prediction allows smoother deceleration
- **Function**: Based on model confidence and prediction horizon
- **Range**: 0.8x to 1.2x multiplier
- **Implementation**: Use existing model confidence metrics

#### 5. Deceleration Comfort Constraint
- **Rationale**: Ensure we don't create uncomfortable deceleration to achieve early arrival
- **Constraint**: Maximum comfortable deceleration ~0.15-0.2g
- **Implementation**: May extend anticipation time if needed for comfort

### Implementation Strategy

#### Phase 1: Basic Anticipation
```python
def calculate_anticipation_time_v1(v_ego, v_target, max_lat_acc):
    """Simple multiplicative model"""
    base_time = 2.0  # 2 second baseline
    
    # Speed factor: sigmoid curve
    speed_factor = 1 + 0.8 * sigmoid((v_ego - 20) / 15)  # 0.5 to 1.8x
    
    # Severity factor: based on lateral acceleration
    severity_factor = 0.8 + 0.5 * (max_lat_acc / 3.0)  # 0.8 to 1.5x
    
    # Delta factor: based on speed reduction
    if v_ego > 0.1:
        delta_ratio = (v_ego - v_target) / v_ego
        delta_factor = 0.9 + 0.5 * delta_ratio  # 0.9 to 1.4x
    else:
        delta_factor = 1.0
    
    return base_time * speed_factor * severity_factor * delta_factor
```

#### Phase 2: Comfort-Constrained Anticipation
```python
def calculate_anticipation_time_v2(v_ego, v_target, max_lat_acc, distance_to_apex):
    """Add deceleration comfort constraints"""
    # Get basic anticipation time
    base_anticipation = calculate_anticipation_time_v1(v_ego, v_target, max_lat_acc)
    
    # Check if we can achieve this comfortably
    required_decel = (v_ego**2 - v_target**2) / (2 * (distance_to_apex - base_anticipation * v_ego))
    
    if abs(required_decel) > COMFORT_DECEL_MAX:
        # Need more time for comfortable deceleration
        # Solve for time that gives us COMFORT_DECEL_MAX
        extra_time_needed = calculate_comfort_time(v_ego, v_target, distance_to_apex)
        return max(base_anticipation, extra_time_needed)
    
    return base_anticipation
```

#### Phase 3: Advanced Psychological Model
```python
def calculate_anticipation_time_v3(state_vector):
    """Neural network or advanced model based on human studies"""
    # Could incorporate:
    # - Driver style learning
    # - Road type classification
    # - Weather/visibility conditions
    # - Historical comfort feedback
    pass
```

### Integration Points

1. **Modify `_update_calculations()`**:
   - Calculate anticipation time when overshoot is detected
   - Adjust `_v_overshoot_distance` by subtracting anticipation distance

2. **State Machine Considerations**:
   - Ensure ENTERING state handles the anticipation buffer
   - Prevent oscillation during the "hold" period at target speed
   - Smooth transition to TURNING state

3. **Jerk Limiting Integration**:
   - Existing jerk limiting should smooth the earlier deceleration
   - May need to tune jerk limits for anticipatory deceleration

## Testing Strategy

### Unit Tests
1. **Anticipation Time Calculation**
   - Test various speed/curvature combinations
   - Verify bounds and edge cases
   - Check scaling factor interactions

2. **Distance Adjustment**
   - Verify correct distance calculations
   - Test minimum distance constraints
   - Check for negative distances

3. **State Transitions**
   - Ensure state machine handles early arrival
   - Test hold period at target speed
   - Verify smooth exit behavior

### Simulation Tests
1. **Comfort Metrics**
   - Measure deceleration rates
   - Calculate jerk profiles
   - Time at target speed before apex

2. **Scenario Testing**
   - Highway curves (high speed, gentle curves)
   - Mountain roads (medium speed, sharp curves)
   - City turns (low speed, tight curves)
   - S-curves (multiple apexes)

3. **Edge Cases**
   - Very short detection distances
   - Sudden curve detection
   - Speed changes during deceleration
   - Loss of vision confidence

### Real-World Validation Metrics
1. **Objective Metrics**
   - Time at target speed before apex
   - Maximum deceleration rate
   - Jerk measurements
   - Speed stability during hold

2. **Subjective Metrics**
   - Passenger comfort ratings
   - Driver intervention frequency
   - Perceived safety margins
   - Comparison to human driving patterns

## Implementation Phases

### Phase 1: Basic Implementation (Week 1)
- Implement `calculate_anticipation_time_v1`
- Integrate with existing VTSC
- Create basic unit tests
- Develop simulation framework

### Phase 2: Refinement (Week 2)
- Add comfort constraints
- Tune parameters based on simulation
- Implement comprehensive test suite
- Add debug visualization

### Phase 3: Advanced Features (Week 3+)
- Consider visibility/confidence factors
- Add user preference settings
- Implement learning/adaptation
- Create A/B testing framework

## Risk Mitigation

1. **Too Early Arrival**: Cap maximum anticipation at 4 seconds
2. **Comfort Violations**: Hard limit on deceleration rate
3. **Distance Constraints**: Minimum viable distance checks
4. **Oscillation**: Hysteresis in state transitions
5. **User Acceptance**: Make it tunable/disable-able

## Success Criteria

1. **Primary**: Reach target speed 1-3 seconds before physically necessary
2. **Comfort**: Maximum deceleration < 0.2g in normal conditions
3. **Stability**: Hold target speed steadily until acceleration phase
4. **Adoption**: >80% of users prefer anticipatory mode
5. **Safety**: No increase in interventions or safety events

## Next Steps

1. Review and refine this plan
2. Implement basic anticipation calculation
3. Create simulation test framework
4. Build visualization tools
5. Begin iterative testing and refinement