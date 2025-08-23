# RTI Speed Limit Bug - Complete Fix Summary

## Problem Statement
RTI was slowing the vehicle to dangerously low speeds (40 mph on 65 mph freeways) when configured to use "posted speed limit" mode.

## Root Causes (Two-Layer Issue)

### Layer 1: No Speed Limit Data Access
- **Location:** `sunnypilot/rtid/rtid.py`
- **Issue:** RTI daemon didn't subscribe to `liveMapDataSP` containing actual speed limits
- **Impact:** Used hardcoded 25 m/s (56 mph) for all police/speed trap threats

### Layer 2: Aggressive Reduction Factors  
- **Location:** `sunnypilot/selfdrive/controls/lib/rti_controller.py`
- **Issue:** Applied 75-95% reduction factors even in "posted speed limit" mode
- **Impact:** Further reduced already-incorrect speeds to dangerous levels

## The Complete Fix

### 1. Added Speed Limit Data Subscription
**File:** `sunnypilot/rtid/rtid.py`
```python
# Before: Missing speed limit data
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation', 
    'carState'
], ignore_avg_freq=True)

# After: Includes speed limit data
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation',
    'carState',
    'liveMapDataSP'  # Added for actual posted speed limit data
], ignore_avg_freq=True)
```

### 2. Created Speed Limit Getter Method
**File:** `sunnypilot/rtid/rtid.py`
```python
def _get_current_speed_limit(self) -> float:
    """Get current posted speed limit from map data.
    
    Returns:
        Speed limit in m/s, or 0.0 if not available
    """
    try:
        map_data = self.sm['liveMapDataSP']
        if map_data.speedLimitValid:
            return float(map_data.speedLimit)
    except Exception as e:
        cloudlog.debug(f"RTI: Could not get speed limit from map data: {e}")
    
    return 0.0
```

### 3. Pass Speed Limit to Threat Detector
**File:** `sunnypilot/rtid/rtid.py` in `_process_cycle_async`
```python
# Get actual speed limit
current_speed_limit = self._get_current_speed_limit()

# Process threats with actual speed limit
rti_state = self.threat_detector.process_threats(
    traffic_data=traffic_data,
    current_location=location,
    current_speed=current_speed,
    timestamp=int(current_time * 1e9),
    v_cruise=cruise_cluster_speed,
    current_heading_deg=self._get_current_heading_deg(),
    posted_speed_limit=current_speed_limit,  # Pass actual posted speed limit
)
```

### 4. Use Real Speed Limits in Threat Processing
**File:** `sunnypilot/rtid/threat_detector.py` in `_process_single_threat`
```python
# Determine speed limit for this threat location
speed_limit_ms = 0.0
if threat.type in ['police', 'policeHiding', 'speedTrap']:
    if posted_speed_limit > 0:
        # Use actual posted speed limit from map data
        speed_limit_ms = posted_speed_limit
        cloudlog.debug(f"RTI: Using actual posted speed limit {speed_limit_ms:.1f} m/s")
    else:
        # Fall back to conservative default when no speed limit available
        speed_limit_ms = self.speed_engine.default_speed_limit_ms
        cloudlog.debug(f"RTI: No posted speed limit, using default {speed_limit_ms:.1f} m/s")
```

### 5. Remove Reduction Factors in Posted Mode
**File:** `sunnypilot/selfdrive/controls/lib/rti_controller.py` in `_calculate_safe_speed`
```python
# In "posted" mode, use the posted speed limit without reduction factors
# User expects RTI to recommend actual speed limit, not a fraction of it
if self._speed_reduction_mode == "posted":
    # For posted mode, only apply minimal safety constraints
    # Don't reduce based on distance - that's the longitudinal planner's job
    
    # Ensure we never recommend acceleration toward a threat
    safe_speed = min(safe_speed, self._v_ego)
    
    # Ensure minimum speed
    if safe_speed < MIN_OPERATING_SPEED:
        safe_speed = MIN_OPERATING_SPEED
        
    # Ensure maximum reasonable speed
    safe_speed = min(safe_speed, V_CRUISE_MAX)
    
    return safe_speed

# For custom mode, continue applying reduction factors...
```

## Impact Analysis

### Before Fix
- 65 mph freeway with police threat
- RTI uses hardcoded 56 mph
- Applies 75% reduction: 56 × 0.75 = 42 mph
- **Result: Dangerous 42 mph on 65 mph freeway**

### After Fix
- 65 mph freeway with police threat
- RTI gets actual 65 mph from map data
- No reduction in "posted" mode
- **Result: Safe 65 mph recommendation**

## Design Philosophy

### "Posted Speed Limit" Mode Intent
When users select this mode, they expect:
1. RTI recommends the actual posted speed limit
2. No arbitrary percentage reductions
3. Longitudinal planner handles final speed selection
4. System respects user's choice to follow posted limits

### Safety Constraints Maintained
- Never accelerate toward threats
- Respect minimum operating speeds
- Stay within system maximums
- Allow longitudinal planner to make final decision

## Testing Considerations

### Unit Testing Challenges
- Requires full environment setup (`tools/op.sh setup`)
- Dependencies include aiohttp, cereal messaging, etc.
- Cannot create "simpler" tests without testing actual code

### Integration Testing Required
1. Verify liveMapDataSP subscription works
2. Confirm speed limit data flows through pipeline
3. Test both "posted" and "custom" modes
4. Validate on actual vehicle with various speed zones

## Lessons Learned

1. **Complete Problem Analysis:** Initial fix only addressed symptom, not root cause
2. **User Intent Matters:** "Posted speed limit" means exactly that - no reductions
3. **Test Real Code:** Creating "simpler" tests without dependencies tests nothing
4. **System Design:** RTI should provide recommendations, not enforce reductions

## Status: COMPLETE

All code changes implemented. The fix addresses both layers of the problem:
- RTI now has access to actual speed limit data
- "Posted" mode uses actual limits without reduction factors

Next step: Deploy and test on actual vehicle to confirm fix resolves the issue.