# RTI Arrow Rotation Fix - Implementation Plan

## Problem Statement
RTI threat arrows snap to 90-degree positions instead of showing precise bearing with 1-degree resolution.

## Root Cause
The system is falling back to the coarse Direction enum (ahead/behind/left/right) instead of using GPS-based precise bearing calculation, likely because:
1. GPS data validation is failing
2. Threat coordinates are invalid/missing
3. The condition check is too strict

## Immediate Diagnostic Steps

### 1. Add Debug Logging (NO CODE CHANGES YET)
First, we need to understand exactly why the GPS path isn't being taken.

**Location**: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc:220-230`

Add temporary debug output to identify the failure:
```cpp
// Line 221 - Add debug logging
LOGD("RTI Arrow Debug: has_gps=%d, threat_lat=%f, threat_lon=%f, ego_lat=%f, ego_lon=%f",
     has_gps, rti_threat_lat, rti_threat_lon, ego_lat, ego_lon);
```

**Location**: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc:606-612`

Add debug for multi-threat rendering:
```cpp
LOGD("RTI Multi Debug: has_gps=%d, has_location=%d, ego_lat=%f, ego_lon=%f, t_lat=%f, t_lon=%f",
     has_gps, threat.has_location, ego_lat, ego_lon, threat.latitude, threat.longitude);
```

### 2. Check GPS Message Reception
**Location**: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc:137-142`

Verify GPS updates are being received:
```cpp
// After line 142
LOGD("GPS Update: lat=%f, lon=%f, bearing=%f", ego_lat, ego_lon, ego_bearing);
```

## Proposed Fixes

### Fix Option 1: Improve GPS State Management (Minimal Risk)

**Problem**: `has_gps` flag gets reset on exceptions or when GPS message isn't "updated" in current frame.

**Solution**: Cache last valid GPS state and don't require "updated" flag.

**Changes needed**:

1. **File**: `selfdrive/ui/sunnypilot/qt/onroad/hud.h`
   - Add fields for GPS validity tracking:
   ```cpp
   // Line 90, after existing GPS fields
   bool gps_ever_valid = false;  // Track if we ever had valid GPS
   uint64_t last_gps_update_frame = 0;  // Track last GPS update
   ```

2. **File**: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`
   - Modify GPS update logic (lines 136-143):
   ```cpp
   // Only check valid, not updated - GPS at 10Hz, we at 20Hz
   if (s.sm->valid("gpsLocationExternal")) {
     uint64_t gps_frame = s.sm->rcv_frame("gpsLocationExternal");
     // Only update if newer than last processed
     if (gps_frame > last_gps_update_frame) {
       const auto gps = (*s.sm)["gpsLocationExternal"].getGpsLocationExternal();
       ego_lat = gps.getLatitude();
       ego_lon = gps.getLongitude(); 
       ego_bearing = gps.getBearingDeg();
       
       // Validate the data
       if (std::isfinite(ego_lat) && std::isfinite(ego_lon) && 
           std::isfinite(ego_bearing) && 
           std::abs(ego_lat) <= 90.0 && std::abs(ego_lon) <= 180.0) {
         has_gps = true;
         gps_ever_valid = true;
         last_gps_update_frame = gps_frame;
       }
     }
   }
   // Don't reset has_gps if no update - keep last valid state
   ```

3. **Modify bearing calculation condition** (line 223):
   ```cpp
   // Use gps_ever_valid instead of has_gps for more persistence
   if (gps_ever_valid && std::isfinite(rti_threat_lat) && std::isfinite(rti_threat_lon) &&
       std::isfinite(ego_lat) && std::isfinite(ego_lon)) {
   ```

### Fix Option 2: Add Precise Bearing to Threat Message (Proper Fix)

**Problem**: Direction enum only provides 4 coarse directions.

**Solution**: Calculate and transmit precise relative bearing from backend.

**Changes needed**:

1. **File**: `cereal/custom.capnp`
   - Add precise bearing field to Threat struct (line ~420):
   ```capnp
   relativeBearingDeg @9 :Float64;  # Precise relative bearing in degrees (-180 to 180)
   hasPreciseBearing @10 :Bool;     # Whether precise bearing is available
   ```

2. **File**: `sunnypilot/rtid/threat_detector.py`
   - Calculate precise bearing instead of coarse direction (lines 199-219):
   ```python
   def get_precise_relative_bearing(self, ego_lat: float, ego_lon: float,
                                   threat_lat: float, threat_lon: float,
                                   ego_heading: float = 0) -> float:
       """Calculate precise relative bearing to threat."""
       threat_bearing = GeoUtils.bearing(ego_lat, ego_lon, threat_lat, threat_lon)
       relative_bearing = (threat_bearing - ego_heading + 360) % 360
       # Normalize to -180 to 180
       if relative_bearing > 180:
           relative_bearing -= 360
       return relative_bearing
   ```

3. **File**: `sunnypilot/rtid/rtid.py`
   - Pass precise bearing in message (lines 268-270):
   ```python
   threat_msg.relativeBearingDeg = threat.relative_bearing_deg
   threat_msg.hasPreciseBearing = True
   ```

4. **File**: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`
   - Use precise bearing when available (lines 606-612):
   ```cpp
   double arrow_angle = 0.0;
   // First try precise bearing from message
   if (threat.hasPreciseBearing && std::isfinite(threat.relativeBearingDeg)) {
     arrow_angle = threat.relativeBearingDeg;
   }
   // Then try GPS calculation
   else if (gps_ever_valid && threat.has_location && ...) {
     arrow_angle = calculateRelativeBearing(...);
   }
   // Finally fall back to direction enum
   else {
     arrow_angle = angleForDirection(threat.direction);
   }
   ```

## Testing Plan

### 1. Diagnostic Testing
- Run with debug logging enabled
- Monitor logcat/console output
- Identify exact failure point

### 2. Fix Validation
- Test with real GPS data
- Verify smooth arrow rotation
- Test GPS loss/recovery scenarios
- Verify fallback behavior

### 3. Performance Testing
- Ensure no UI lag from calculations
- Verify smoothing filter works correctly
- Test update rate synchronization

## Risk Assessment

### Fix Option 1 (GPS State Management)
- **Risk**: Low
- **Impact**: Should immediately fix issue if GPS data is available
- **Rollback**: Easy, just revert hud.cc changes

### Fix Option 2 (Precise Bearing)
- **Risk**: Medium
- **Impact**: Permanent fix with best precision
- **Rollback**: Requires capnp schema change (backward compatible)

## Recommendation

1. **Immediate**: Implement diagnostic logging to understand failure
2. **Quick Fix**: Implement Fix Option 1 (GPS state management)  
3. **Proper Fix**: Implement Fix Option 2 (precise bearing in message)

The GPS state management fix should resolve the immediate issue while the precise bearing enhancement provides the long-term solution with better architecture.