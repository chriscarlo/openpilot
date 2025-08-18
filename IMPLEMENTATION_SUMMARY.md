# Speed Limit Controller Implementation Summary

## Changes Made

### 1. EV6 Dashboard Speed Limit Reading
**File:** `opendbc_repo/opendbc/car/hyundai/carstate.py`

- Added CAN message 354 (CCNC_0x162) to camera bus parser at 20Hz
- Implemented speed limit parsing in `update_canfd()` method
- Reads SPEEDLIMIT signal and converts from km/h to m/s
- Handles invalid values (0 and 255 = no limit)
- Populates `ret_sp.speedLimit` for speed limit controller

### 2. Combined Mode Logic Change - Takes MAXIMUM
**File:** `sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py`

- Changed `_get_source_solution_according_to_policy()` method
- Combined mode now takes the HIGHER speed limit (using `np.argmax`)
- Previously took the LOWER speed limit (using `np.argmin`)
- Properly filters out zero values (no limit detected)

### 3. UI Description Update
**File:** `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/slc/speed_limit_control_policy.h`

- Updated Combined mode description from:
  - "Use combined Speed Limit data from Car & OpenStreetMaps"
- To:
  - "Use HIGHER Speed Limit between Car & OpenStreetMaps data"

## Configuration

To enable dashboard speed limit on EV6:
1. Go to **Settings → Controls → Speed Limit Control**
2. Enable **'Speed Limit Control'**
3. Set **'Speed Limit Source'** to one of:
   - **Car Only**: Use only dashboard speed limit
   - **Car First**: Prioritize dashboard, fallback to map
   - **Combined**: Use HIGHER of dashboard and map (NEW BEHAVIOR)

## Testing

All implementations have been tested and verified:
- ✅ Python syntax verified
- ✅ Build successful (scons --minimal)
- ✅ Module imports correctly
- ✅ Combined mode logic tested (takes maximum)
- ✅ Dashboard speed limit parsing tested

## Test Scripts Created

- `test_ev6_speed_limit.py` - Runtime monitoring test
- `test_combined_mode_max.py` - Combined mode logic verification
- `verify_ev6_speed_limit.py` - Implementation verification

## Technical Details

### Dashboard Speed Limit (EV6)
- Message ID: 354 (0x162)
- Message Name: CCNC_0x162
- Signal: SPEEDLIMIT
- Position: Bits 32|8
- Units: km/h (converted to m/s)
- Invalid: 0 = no limit, 255 = no limit

### Combined Mode Behavior
- Takes the HIGHER speed limit between sources
- Filters out zero values (no data)
- Returns appropriate source indicator
- Safer behavior - prefers higher limit when sources disagree