# Enhanced VTSC Pass Criteria

## Overall Requirements
- Emergency scenarios test: ≥80% pass rate
- Integrated system test: ≥80% pass rate  
- System deceleration constraint: -6.0 m/s² maximum

## Scenario Categories

### Normal Scenarios (Must Pass 100%)
- Highway curves with adequate warning distance (>100m)
- City turns with normal approach speeds (<60 km/h)
- Gentle curves with high visibility
- **Pass Definition**: Reaches target speed smoothly with ≤0.25g deceleration

### Challenging Scenarios (Must Pass ≥80%)
- Late detection curves (50-80m warning)
- Blind corners with partial visibility
- Decreasing radius curves
- Higher speed approaches (80-120 km/h)
- **Pass Definition**: Reaches safe speed with ≤0.6g deceleration, no driver intervention

### Edge Cases (Must Handle Gracefully ≥60%)
- Very late detection (<40m warning)
- Complete vision loss scenarios
- Physically impossible scenarios
- **Pass Definition**: Either reaches safe speed OR requests intervention appropriately (not prematurely)

## Specific Pass Criteria

### 1. Speed Achievement
- **PASS**: Final speed ≤ 110% of physics-based target speed
- **FAIL**: Final speed > 110% of target (unsafe for curve)

### 2. Deceleration Comfort
- **Normal**: Max decel ≤ 2.45 m/s² (0.25g)
- **Challenging**: Max decel ≤ 5.89 m/s² (0.6g)  
- **Emergency**: Max decel ≤ 6.0 m/s² (system limit)

### 3. Anticipatory Behavior
- **PASS**: Begins deceleration before physically necessary
- **Target**: Reach target speed 1-3 seconds before curve entry
- **Minimum**: Reach target speed before curve entry

### 4. Intervention Logic
- **Appropriate**: Only when physics demands > 6.0 m/s² deceleration
- **Premature**: Intervention when < 5.5 m/s² would suffice
- **FAIL**: Any premature intervention

### 5. Vision Handling
- **Full visibility**: Trust model predictions
- **Degraded vision**: Apply proportional safety margins (5-20% max)
- **Lost vision**: Graceful degradation, not panic response

## Test Requirements

### Emergency Scenarios (10 scenarios)
- 2 Normal (must pass 2/2)
- 6 Challenging (must pass 5/6)
- 2 Edge cases (must pass 1/2)
- **Overall**: 8/10 = 80%

### Integrated Tests (expand to 10 scenarios)
- 3 Normal (must pass 3/3)
- 5 Challenging (must pass 4/5)
- 2 Edge cases (must pass 1/2)
- **Overall**: 8/10 = 80%

## Failure Analysis Requirements
Any test failure must identify:
1. Required deceleration vs available
2. Distance available vs distance needed
3. Vision status impact
4. Specific failure mode