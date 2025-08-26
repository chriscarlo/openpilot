# RTI Bearing Calculation Refactoring Summary

## Date
2025-08-26

## Overview
Successfully refactored the RTI (Real-time Traffic Intelligence) system to move bearing calculations from the UI layer to the `rtid` daemon, improving architecture and eliminating redundant GPS subscriptions.

## Changes Made

### 1. Message Schema Updates (`cereal/custom.capnp`)
- Added `displayArrowAngle @9 :Float32` to `Threat` struct
- Added `hasLocation @10 :Bool` to `Threat` struct
- These fields allow rtid to pre-compute and pass display angles to UI

### 2. RTID Enhancements (`sunnypilot/rtid/rtid.py`)
Added new methods:
- `_normalize_180()` - Normalize angles to [-180, 180] range
- `_calculate_relative_bearing()` - Calculate bearing from ego to threat using spherical trigonometry
- `_angle_for_direction()` - Convert discrete directions to angles for fallback
- `_is_valid_gps_pair()` - Validate GPS coordinate pairs
- `_calculate_threat_display_data()` - Main method to compute display angles

Modified:
- `_publish_rti_state()` - Now computes and includes display angles in messages

### 3. UI Simplifications

#### Removed GPS Subscriptions (`selfdrive/ui/sunnypilot/ui.cc`)
- Removed `gpsLocationExternal` subscription
- Removed `gpsLocation` subscription
- UI no longer needs GPS data directly

#### Removed Functions (`selfdrive/ui/sunnypilot/qt/onroad/hud.cc`)
- Removed `isValidGPSPair()`
- Removed `isValidCoordinate()`
- Removed `calculateRelativeBearing()`
- Removed `angleForDirection()`

#### Removed State Variables (`selfdrive/ui/sunnypilot/qt/onroad/hud.h`)
- Removed ego GPS position variables (ego_lat, ego_lon, ego_bearing)
- Removed GPS tracking variables
- Removed arrow update caching (handled by rtid now)

#### Simplified Drawing Logic
- UI now directly uses `threat.getDisplayArrowAngle()` from message
- Removed complex GPS validation and bearing calculation logic
- Retained smoothing for visual stability

## Benefits Achieved

### 1. Improved Architecture
- **Single Responsibility**: rtid handles all GPS and bearing logic
- **Separation of Concerns**: UI focuses purely on rendering
- **Reduced Coupling**: UI no longer depends on GPS messages

### 2. Performance Improvements
- Eliminated redundant GPS subscriptions
- Calculations happen once in rtid vs per-threat in UI
- Reduced UI thread CPU load during rendering
- Smaller message passing overhead

### 3. Reliability Enhancements
- GPS validation happens upstream, once
- Fallback logic managed centrally
- No possibility of UI freeze from GPS issues
- Easier to unit test (logic isolated in Python)

### 4. Code Reduction
- Removed ~200 lines of C++ code from UI
- Simplified UI update logic
- Cleaner, more maintainable codebase

## Testing
- Created comprehensive test suite for bearing calculations
- All tests pass successfully
- UI builds without errors
- Logic verified equivalent to original implementation

## Implementation Notes
- Bearing calculations use forward azimuth formula (same as original)
- Maintains 1-degree precision for arrows
- Preserves fallback behavior for missing GPS
- Smoothing still applied in UI for visual stability

## Future Considerations
1. Could further optimize by caching bearing calculations in rtid
2. Consider adding bearing smoothing in rtid to reduce UI work
3. Potential to add more display pre-computations (e.g., colors, text)

## Conclusion
The refactoring successfully addresses the architectural issues identified:
- Eliminates bug-prone GPS handling in UI
- Improves system reliability and maintainability
- Follows proper separation of concerns
- Reduces overall system complexity

The implementation is complete, tested, and ready for production use.