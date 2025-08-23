# RTI Speed Limit Bug - Root Cause Analysis

## Executive Summary
The RTI system is recommending dangerously low speeds (40 mph on 65 mph freeways) because it has no access to actual posted speed limit data and uses a hardcoded 56 mph fallback value.

## Root Cause
The RTI daemon (`rtid.py`) doesn't subscribe to speed limit data sources (`liveMapDataSP`). When configured to use "posted speed limit" mode, it defaults to a hardcoded 25 m/s (56 mph) for all police/speed trap threats, regardless of the actual speed limit.

## Bug Chain Analysis

### 1. Missing Speed Limit Data Subscription
**File:** `sunnypilot/rtid/rtid.py` line 32-36
```python
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation',
    'carState'
], ignore_avg_freq=True)
```
- Missing: `liveMapDataSP` which contains actual speed limit data

### 2. Hardcoded Fallback Speed
**File:** `sunnypilot/rtid/threat_detector.py` line 281
```python
self.default_speed_limit_ms = 25  # 55 mph default when unknown
```
- Comment says 55 mph but 25 m/s = 56 mph
- Used for ALL police/speed trap threats when no speed limit available

### 3. Threat Processing Logic
**File:** `sunnypilot/rtid/threat_detector.py` lines 519-530
```python
speed_limit_ms = 0.0
if threat.speed_limit:
    speed_limit_ms = threat.speed_limit / 3.6
else:
    if threat.type in ['police', 'policeHiding', 'speedTrap']:
        speed_limit_ms = self.speed_engine.default_speed_limit_ms  # 25 m/s
    else:
        speed_limit_ms = 0.0
```

### 4. RTI Controller Further Reductions
**File:** `sunnypilot/selfdrive/controls/lib/rti_controller.py` lines 223-227
```python
SPEED_REDUCTION_FACTORS = {
    'critical': 0.75,  # 75% of current speed for critical threats
    'near': 0.85,      # 85% of current speed for near threats  
    'normal': 0.95     # 95% of current speed for normal threats
}
```

## Impact Calculation
On a 65 mph (29 m/s) freeway:
1. RTI uses hardcoded 25 m/s (56 mph) instead of actual 29 m/s
2. Applies 75% reduction for close threats: 25 × 0.75 = 18.75 m/s
3. **Result: 42 mph in a 65 mph zone** - dangerously slow

## Fix Implementation Plan

### Phase 1: Add Speed Limit Data Access
1. Add `liveMapDataSP` subscription to rtid.py
2. Pass speed limit data to threat detector
3. Use actual speed limits instead of hardcoded value

### Phase 2: Improve Speed Recommendation Logic  
1. When in "posted" mode, use actual posted speed limit
2. Fall back to conservative reduction only when no speed limit available
3. Ensure recommendations are never dangerously low

### Phase 3: Testing & Validation
1. Test with various speed limit zones
2. Verify proper unit conversions
3. Ensure safe behavior in all scenarios

## SLC Integration Notes
- Speed Limit Controller applies offsets BEFORE publishing
- `speed_limit_offseted` property includes user-configured offsets
- RTI should use raw speed limit, not offseted value
- This allows longitudinal planner to select minimum of all recommendations

## Safety Considerations
- Never recommend speeds below safe minimums for road type
- Always validate against current vehicle speed
- Ensure gradual deceleration profiles
- Test thoroughly before deployment