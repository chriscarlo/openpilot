# RTI Arrow Rotation Fix - Implementation Summary

## Problem Fixed
RTI threat warning arrows were snapping to 90-degree positions (0°/±90°/180°) instead of rotating smoothly with 1-degree resolution.

## Root Cause
The UI was only subscribing to `gpsLocationExternal` (u-blox GPS module) and not `gpsLocation` (internal QCOM GPS). When external GPS was unavailable, the system fell back to a coarse Direction enum with only 4 values.

## Changes Implemented

### 1. Added GPS Fallback Subscription
**File**: `selfdrive/ui/sunnypilot/ui.cc:24`
- Added `"gpsLocation"` to SubMaster subscription list
- Now subscribes to both external (u-blox) and internal (QCOM) GPS sources

### 2. Added Frame Tracking for Staleness Check  
**File**: `selfdrive/ui/sunnypilot/qt/onroad/hud.h:90`
- Added `uint64_t last_gps_rcv_frame = 0;` field
- Tracks last GPS update frame for staleness validation

### 3. Updated GPS Ingestion Logic
**File**: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc:136-176`
- Implemented priority cascade: external GPS → internal GPS
- Added bearing validation (finite, 0-360° range)
- Frame-based staleness check: GPS valid if updated within 1 second (UI_FREQ frames)
- Removed GPS reset in exception handler (lines 178-183)

## Key Implementation Details

### GPS Priority Logic
```cpp
1. Check gpsLocationExternal (preferred - u-blox module)
2. If not available/updated, check gpsLocation (fallback - QCOM internal)
3. Validate bearing is finite and in [0, 360] range
4. Update last_gps_rcv_frame when valid data received
5. Set has_gps = true if last update within 1 second
```

### Staleness Formula
```cpp
has_gps = ((s.sm->frame - last_gps_rcv_frame) < UI_FREQ);
```
- UI_FREQ = 20Hz (20 frames/second)
- 1-second window handles occasional dropped frames
- Prevents flicker while ensuring timely updates

## Testing Status
- ✅ Compilation successful - no errors or warnings
- ✅ Both HUD and UI modules compile cleanly
- 🔲 Runtime testing pending (requires hardware/simulation)

## Expected Behavior After Fix
1. Arrows should rotate smoothly with 1-degree resolution
2. System uses external GPS when available (higher accuracy)
3. Falls back to internal GPS when external unavailable
4. Only uses coarse direction enum when NO GPS available
5. Handles frame drops gracefully without flickering

## Edge Cases Addressed
1. **Invalid bearings**: Validates bearing is finite and in valid range
2. **Frame drops**: 1-second staleness window prevents flicker
3. **GPS switching**: Priority system ensures best source is used
4. **Exception safety**: GPS state persists through exceptions

## Potential Future Enhancement
If 1Hz `gpsLocation` causes flicker with 1-second window, can extend to 1.2-1.5 seconds:
```cpp
has_gps = ((s.sm->frame - last_gps_rcv_frame) < (UI_FREQ * 1.2));
```

## Files Modified
1. `selfdrive/ui/sunnypilot/ui.cc` - Added GPS subscription
2. `selfdrive/ui/sunnypilot/qt/onroad/hud.h` - Added frame tracking
3. `selfdrive/ui/sunnypilot/qt/onroad/hud.cc` - Updated GPS logic

## No Changes Required To
- `calculateRelativeBearing()` - Math already correct
- `smoothAngleForThreat()` - Smoothing already works
- `drawRTIArrow()` - Rendering already correct
- `angleForDirection()` - Kept as last-resort fallback

The fix addresses the data ingestion problem, not the math or rendering which were already correct.