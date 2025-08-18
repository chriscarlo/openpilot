# Speed Limit Controller Combined Mode Fix

## Issue
Commit 254e5adb fixed dashboard speed limit reading for CANFD Hyundai vehicles (like EV6) but inadvertently broke the "combined" mode logic for map vs dashboard speed limits.

## Root Cause
The original implementation in `SpeedLimitResolver._get_source_solution_according_to_policy()` used `np.argmax()` to select the highest speed limit value. When both map and dashboard values were equal, `np.argmax()` defaulted to the first index (car_state) instead of preferring map_data as specified.

## Requirements
The combined mode should work as follows:
1. **Both values same**: Use map value (prefer map_data source)
2. **Both values different**: Use the HIGHER value 
3. **Only one available**: Use what we have
4. **None available**: Return none

## Solution
Modified the `_get_source_solution_according_to_policy()` method in `sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py` to:

1. Explicitly handle the case where both values are equal (within epsilon tolerance of 0.01 m/s)
2. When equal, return `Source.map_data` 
3. When different, return the source with the higher value
4. Properly handle single source and no source scenarios

## Testing
Created comprehensive tests to verify:
- Equal values prefer map_data ✓
- Different values select the higher one ✓  
- Single source scenarios work correctly ✓
- No data returns Source.none ✓
- Edge cases with floating point precision ✓

## Files Changed
- `sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py` - Fixed the combined mode logic

## Verification Scripts
- `test_combined_mode_fix.py` - Initial test demonstrating the issue and fix
- `verify_combined_mode_fix.py` - Comprehensive verification with realistic scenarios