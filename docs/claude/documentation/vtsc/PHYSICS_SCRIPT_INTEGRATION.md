# Vision Turn Speed Controller (VTSC) - Post-Apex Acceleration Fix

## Critical Issue Identified and Resolved

### The Problem
The Vision Turn Speed Controller contained a **fundamental bug** where post-apex acceleration was calculated but never used:

- Post-apex acceleration logic existed in `_update_solution()` 
- The acceleration value was stored in the `a_target` variable
- However, the `a_target` **property** always returned `self._current_decel` when active
- This meant the entire post-apex acceleration feature was **non-functional**

### Root Cause Analysis
**File**: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`

**Original Broken Code** (lines 176-180):
```python
@property
def a_target(self):
  # Always use enhanced deceleration when active
  if self.is_active:
    return self._current_decel  # <-- BUG: Always returns deceleration
  return self._a_ego
```

**Issue**: The property ignored the post-apex acceleration calculated in `_update_solution()`.

### The Fix Applied

**Fixed Code**:
```python
@property  
def a_target(self):
  if not self.is_active:
    return self._a_ego
  
  # Use post-apex acceleration when active
  if self._apex_acceleration_active:
    return self._apex_acceleration_value  # <-- FIX: Now returns acceleration
  
  # Otherwise use enhanced deceleration
  return self._current_decel
```

**Supporting Changes**:
1. Added `_apex_acceleration_active` flag to track when post-apex acceleration should be used
2. Added `_apex_acceleration_value` to store the calculated acceleration 
3. Modified `_update_solution()` to set these flags when post-apex logic is triggered
4. Reset flags when entering disabled state

## Verification Testing

**Test File**: `docs/claude/tests/vtsc/test_post_apex_acceleration.py`

**Test Results**: ✅ ALL TESTS PASSED
- When NOT in post-apex mode: `a_target` returns `_current_decel` 
- When IN post-apex mode: `a_target` returns `_apex_acceleration_value`
- When disabled: `a_target` returns `_a_ego`
- Post-apex logic triggers correctly and produces positive acceleration (2.0 m/s²)

## Current Status

**Production Code**: ✅ FIXED  
**Test Suite**: ✅ VERIFIED  
**Documentation**: ✅ ACCURATE (this document)  

### What Works Now
- Post-apex acceleration logic calculates appropriate acceleration values
- The `a_target` property correctly returns those values when post-apex is active
- The longitudinal planner receives positive acceleration commands after curve apex
- All existing VTSC functionality (emergency deceleration, anticipatory control) remains intact

### What Was Never Working Before
- Post-apex acceleration commands were never sent to the longitudinal planner
- The entire post-apex feature was a no-op due to the property bug
- Any previous claims of "performance metrics" were impossible since the feature was non-functional

## Technical Implementation Details

### Post-Apex Acceleration Logic
The existing logic in `_update_solution()` (lines 561-587) calculates:
1. Target speed using physics-based curvature-to-speed conversion
2. Speed error compared to current ego velocity  
3. Acceleration command with 2.0 m/s² limit
4. Sets `_apex_acceleration_active = True` and `_apex_acceleration_value` when triggered

### Integration Points
- **Longitudinal Planner**: Calls `controller.a_target` and uses returned value
- **Message Flow**: Post-apex acceleration now properly flows through the control pipeline
- **Safety**: Existing -6.0 m/s² system limits still apply through emergency deceleration system

## Lessons Learned

1. **Code Review Importance**: A simple property bug rendered an entire feature non-functional
2. **Testing Critical**: The original test suite tested local files, not production code
3. **Documentation Integrity**: Claims must be based on verified, working functionality
4. **Property vs Variable**: Be careful when properties and variables have the same name

## Files Modified

- `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` - Fixed `a_target` property
- `docs/claude/tests/vtsc/test_post_apex_acceleration.py` - New verification test

The Vision Turn Speed Controller post-apex acceleration feature is now **functional and verified**.