# LiveSteerRatio Implementation

## Overview
LiveSteerRatio is a feature that allows users to manually override the steering ratio in real-time through the GUI. This provides immediate adjustment of steering sensitivity without requiring a restart or waiting for the system to learn new parameters.

## Implementation Details

### 1. Parameter Storage
- Added `LiveSteerRatio` to `common/params_keys.h` with `PERSISTENT | BACKUP` flags
- Parameter value of 0 means "use vehicle default"
- Valid range: 5.0 to 25.0

### 2. GUI Control
- Located in **Settings → Steering** menu (`lateral_panel.cc`)
- Custom control with:
  - **+/-** buttons for fine adjustment (0.01 steps)
  - Current value display
  - Status indicator (Default/Modified)
  - **Reset** button to revert to default (sets to 0)

### 3. Control System Integration
- Modified `controlsd.py` to read `LiveSteerRatio` parameter
- Override logic in `state_control()` method:
  ```python
  # Check for live steering ratio override from GUI
  live_steer_ratio = float(self.params.get("LiveSteerRatio") or 0)
  if live_steer_ratio > 0.0:
      sr = live_steer_ratio
  else:
      sr = max(lp.steerRatio, 0.1)
  ```
- When LiveSteerRatio > 0, it overrides the learned value from paramsd
- Applied to VehicleModel immediately for instant effect

### 4. Vehicle-Specific Changes
- Updated KIA EV6 default steer ratio from 16 to 13.43 in `opendbc/car/hyundai/values.py`

## Files Modified

1. **common/params_keys.h**
   - Added LiveSteerRatio parameter definition

2. **selfdrive/controls/controlsd.py**
   - Added override logic to use LiveSteerRatio when set

3. **selfdrive/ui/sunnypilot/qt/offroad/settings/lateral_panel.cc/h**
   - Added LiveSteerRatioControl to Settings → Steering menu

4. **selfdrive/ui/sunnypilot/qt/offroad/settings/lateral/live_steer_ratio.cc/h**
   - New files implementing the GUI control

5. **selfdrive/ui/sunnypilot/SConscript**
   - Added live_steer_ratio.cc to build system

6. **opendbc/car/hyundai/values.py**
   - Changed KIA EV6 steer ratio from 16 to 13.43

## Usage

1. Navigate to **Settings → Steering** in the UI
2. Find the **Live Steering Ratio** control
3. Use **+/-** buttons to adjust the ratio
   - Lower values = more sensitive/quicker steering
   - Higher values = less sensitive/slower steering
4. The change takes effect immediately while driving
5. Use **Reset** button to return to vehicle default (sets to 0)

## Technical Notes

- The override happens at the control level, not at the parameter learning level
- This ensures immediate effect without waiting for convergence
- The learned value (lp.steerRatio) continues to update in the background
- When LiveSteerRatio is set to 0, the system reverts to using the learned value
- The parameter persists across reboots due to PERSISTENT flag

## Testing

A comprehensive test suite is provided in `docs/claude/tests/live_steer_ratio/test_live_steer_ratio.py` that verifies:
- Parameter storage and retrieval
- Persistence across restarts
- Override logic in controlsd
- Bounds validation
- Zero value behavior (use default)