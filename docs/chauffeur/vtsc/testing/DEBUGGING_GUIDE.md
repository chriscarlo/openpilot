# Adaptive Deceleration System - Debugging Guide

## Issues Found During Testing

### 1. Adaptive Mode Not Triggering (Priority: HIGH)

**Symptom**: When deceleration requirement exceeds comfort limit (-1.47 m/s²), system doesn't escalate to adaptive mode.

**Location**: `_get_optimal_deceleration()` method around line 640-685

**Likely Cause**: The condition check for `comfort_sufficient` may be inverted or the comparison logic is incorrect.

**Debug Steps**:
```python
# Add debug logging in _get_optimal_deceleration():
print(f"Raw decel: {raw_decel}, Filtered: {self._filtered_decel_requirement}")
print(f"Comfort sufficient: {comfort_sufficient}, Hysteresis state: {self._decel_hysteresis_state}")
```

**Fix Suggestion**: Check the boolean logic in line ~650:
```python
# Current (potentially buggy):
elif not comfort_sufficient and comfort_sufficient != self._decel_hysteresis_state:

# Should possibly be:
elif not comfort_sufficient and not self._decel_hysteresis_state:
```

### 2. Hysteresis State Management (Priority: HIGH)

**Symptom**: Mode transitions not maintaining proper hysteresis, causing potential oscillation.

**Location**: Lines 649-673 in `_get_optimal_deceleration()`

**Likely Cause**: Complex conditional logic may have edge cases not properly handled.

**Debug Steps**:
```python
# Log all state transitions
if self._decel_hysteresis_state != previous_state:
    print(f"State transition: {previous_state} → {self._decel_hysteresis_state}")
    print(f"Triggered by: filtered_req={self._filtered_decel_requirement}, threshold={hysteresis_threshold}")
```

### 3. EMA Filter Variance Issue (Priority: MEDIUM)

**Symptom**: Filtered output has higher variance than expected, not smoothing effectively.

**Location**: Line 642-643

**Likely Cause**: Filter may be resetting unexpectedly or alpha coefficient applied incorrectly.

**Debug Steps**:
```python
# Track filter state
print(f"Filter before: {self._filtered_decel_requirement}")
print(f"Input: {raw_decel}, Alpha: {self._filter_alpha}")
print(f"Filter after: {self._filtered_decel_requirement}")
```

### 4. Jerk Limiting Sign Error (Priority: LOW)

**Symptom**: Test expects positive max_change but calculation may produce negative.

**Location**: Lines 676-685

**Fix**: Ensure absolute value is used for jerk limit:
```python
max_decel_change = abs(target_jerk_limit) * dt  # Use abs() to ensure positive
```

## Testing Improvements

### Add Debug Mode
Consider adding a debug flag to the controller:
```python
def __init__(self, CP):
    # ...
    self._debug_adaptive = os.environ.get('VTSC_DEBUG_ADAPTIVE', False)
    
def _log_adaptive(self, msg):
    if self._debug_adaptive:
        print(f"[ADAPTIVE] {msg}")
```

### Create Visualization Tools
```python
# Save state history for analysis
self._adaptive_history = []

def _record_adaptive_state(self):
    self._adaptive_history.append({
        'time': time.time(),
        'raw_decel': self._current_decel,
        'filtered': self._filtered_decel_requirement,
        'mode': 'adaptive' if self._decel_hysteresis_state else 'comfort',
        'v_ego': self._v_ego
    })
```

## Running Focused Tests

To debug specific issues:
```bash
# Test only adaptive system
PYTHONPATH=/projects/chauffeur/data/openpilot python3 \
    docs/chauffeur/vtsc/testing/adaptive_deceleration/test_adaptive_system.py \
    TestAdaptiveDeceleration.test_adaptive_escalation_needed

# Test with debug output
VTSC_DEBUG_ADAPTIVE=1 python3 <test_file>
```

## Verification After Fixes

1. **Unit Tests**: Re-run failed tests individually
2. **Integration Tests**: Verify full system behavior
3. **Replay Tests**: Use recorded drives to validate real-world performance
4. **Edge Cases**: Test rapid mode transitions, extreme deceleration scenarios

## Performance Considerations

- EMA filter alpha: Lower values (0.1-0.3) = smoother but slower response
- Hysteresis threshold: Higher values (0.3-0.5) = more stable but less responsive
- Safety bias: Balance between comfort (0.05-0.1) and safety (0.15-0.25)

## Next Steps

1. Fix the identified bugs in order of priority
2. Re-run test suite to verify fixes
3. Add more comprehensive state transition tests
4. Consider adding telemetry for production monitoring
5. Document the fixed behavior in code comments