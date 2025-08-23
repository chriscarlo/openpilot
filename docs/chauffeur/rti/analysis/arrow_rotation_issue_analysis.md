# RTI Arrow Rotation Issue Analysis

## Issue Summary
The RTI threat warning arrows are only rendering at 90-degree positions (straight up/down/left/right) instead of rotating smoothly with 1-degree resolution based on the actual relative azimuth between the ego vehicle and threats.

## Root Cause Analysis

### 1. Primary Issue: Fallback to Coarse Direction Enum
The system has two code paths for determining arrow rotation:

#### Path A: GPS-Based Precise Bearing (Intended)
- Uses actual lat/lon coordinates and ego heading
- Calculates precise relative bearing with 1-degree resolution
- Located in `selfdrive/ui/sunnypilot/qt/onroad/hud.cc:224-225`
- **SHOULD** provide smooth 360-degree rotation

#### Path B: Direction Enum Fallback (Currently Executing)
- Uses coarse `Direction` enum with only 4 values: ahead/behind/left/right
- Maps to fixed angles: 0°, 90°, 180°, -90°
- Located in `selfdrive/ui/sunnypilot/qt/onroad/hud.cc:227` and `hud.cc:734-744`
- **CAUSES** the 90-degree snap behavior

### 2. Why GPS Path Isn't Working

The code has a conditional check that determines which path to use:
```cpp
if (has_gps && std::isfinite(rti_threat_lat) && std::isfinite(rti_threat_lon) &&
    std::isfinite(ego_lat) && std::isfinite(ego_lon)) {
    // Use precise GPS bearing
} else {
    // Fall back to Direction enum
}
```

**Potential failures:**
1. **`has_gps` is false**: GPS data not being received/validated
2. **Threat coordinates are NaN/invalid**: Threat lat/lon not properly transmitted
3. **Ego coordinates are NaN/invalid**: Ego position not being updated

### 3. Data Flow Issues Identified

#### Threat Processing Pipeline:
1. **threat_detector.py:199-219**: Calculates coarse direction (45° sectors)
2. **rtid.py:265-268**: Passes both lat/lon AND direction to message
3. **hud.cc:498-501**: Receives threat with `has_location` flag
4. **hud.cc:606-612**: Decides whether to use GPS or direction fallback

#### Ego Position Pipeline:
1. **gpsLocationExternal**: Published at 10Hz with lat/lon/bearing
2. **hud.cc:137-142**: Updates ego position when GPS message received
3. **Issue**: GPS update tied to `s.sm->updated()` which may not align with draw calls

### 4. Specific Problems Found

1. **Direction Quantization in threat_detector.py:212-219**
   - Uses 45-degree sectors to classify into 4 directions
   - This coarse classification is used as fallback
   - Even when GPS coordinates are available, direction enum limits precision

2. **GPS State Management**
   - `has_gps` flag set in updateState but could be false during initial frames
   - GPS updates at 10Hz but RTI updates at 1Hz
   - Potential timing mismatch causing GPS data to be stale/invalid

3. **Threat Location Validation**
   - `has_location` flag depends on `std::isfinite(latitude) && std::isfinite(longitude)`
   - If backend sends 0,0 or invalid coords, falls back to direction enum

## Solution Plan

### Quick Fix (Minimal Changes)
1. **Ensure GPS data persistence**: Store last valid GPS state instead of resetting
2. **Debug logging**: Add logging to identify why GPS path isn't executing
3. **Validate threat coordinates**: Ensure backend sends valid lat/lon

### Proper Fix (Recommended)
1. **Remove direction quantization**: Calculate precise bearing at all levels
2. **Pass relative bearing directly**: Instead of Direction enum, pass float bearing
3. **Improve GPS state management**: Decouple GPS validity from update timing
4. **Add fallback bearing calculation**: Use last known good values if current invalid

### Implementation Steps

#### Step 1: Add Debug Logging
- Log when GPS path vs direction path is taken
- Log values of has_gps, threat coords, ego coords
- Identify exact failure point

#### Step 2: Fix GPS State Management
- Store last valid GPS position/heading
- Don't reset has_gps on exception
- Use cached values if current frame has no update

#### Step 3: Enhance Threat Message
- Add `relative_bearing_deg` field to threat message
- Calculate precise bearing in threat_detector
- Use this instead of Direction enum in UI

#### Step 4: Update HUD Logic
- Prefer relative_bearing_deg when available
- Fall back to calculated bearing from coords
- Only use Direction enum as last resort

## Testing Strategy
1. Monitor debug logs to verify GPS path execution
2. Test with simulated GPS data at various headings
3. Verify smooth rotation at 1-degree increments
4. Test fallback behavior when GPS unavailable

## Update Rates
- gpsLocationExternal: 10Hz
- rtiStateSP: 1Hz  
- UI refresh: ~20Hz
- Smoothing filter: α=0.2 (already implemented)

The smoothing should handle the rate mismatch, but the issue is we're not getting to the smoothing code because we're stuck in the direction enum path.