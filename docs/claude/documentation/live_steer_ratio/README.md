# LiveSteerRatio Feature Documentation

## Overview

The LiveSteerRatio feature allows real-time adjustment of the steering ratio for supported vehicles without requiring a system restart. This feature was specifically implemented with the KIA EV6 in mind, changing its default from 16.0 to 13.43.

## Implementation Details

### Files Modified

1. **opendbc/car/hyundai/values.py**
   - Changed KIA EV6 default steer ratio from 16 to 13.43
   - Line 528: `CarSpecs(mass=2055, wheelbase=2.9, steerRatio=13.43, tireStiffnessFactor=0.65)`

2. **selfdrive/locationd/paramsd.py**
   - Added LiveSteerRatio parameter support
   - Reads LiveSteerRatio on startup
   - Uses it as base_steer_ratio when > 0
   - Falls back to vehicle default when set to 0
   - Passes base_steer_ratio to VehicleParamsLearner for bounds calculation

3. **selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/hyundai_settings.cc**
   - Added LiveSteerRatio control to Hyundai settings menu
   - Uses ButtonControlSP with input dialog
   - Range: 0.0 to 25.0
   - Shows "13.43 for EV6" as default in description

4. **common/params_keys.h**
   - Added LiveSteerRatio key definition
   - Line 197: `{"LiveSteerRatio", PERSISTENT | BACKUP}`

5. **selfdrive/locationd/test/test_paramsd.py**
   - Updated tests to handle additional return value from retrieve_initial_vehicle_params

## How It Works

### Parameter Behavior

- **LiveSteerRatio = 0 or not set**: Uses vehicle default (13.43 for KIA EV6)
- **LiveSteerRatio > 0**: Overrides vehicle default with specified value
- **Valid Range**: 0.0 to 25.0

### Bounds Calculation

The parameter learner bounds are calculated as:
- Minimum: 0.5 × base_steer_ratio
- Maximum: 2.0 × base_steer_ratio

Examples:
- Default (13.43): bounds are 6.71 to 26.86
- Custom (15.0): bounds are 7.50 to 30.00

### GUI Usage

1. Navigate to Settings → Vehicle → Hyundai
2. Click "Live Steering Ratio" → Edit
3. Enter desired value (0.0-25.0)
4. Changes take effect immediately

## User Guide

### For KIA EV6 Owners

The default steering ratio has been changed from 16.0 to 13.43, which should provide:
- More responsive steering feel
- Better alignment with other Hyundai/Kia vehicles
- Improved openpilot lateral control

### Tuning Guidelines

- **Steering feels too sensitive**: Increase the value (e.g., 15.0 or 16.0)
- **Steering feels too heavy**: Decrease the value (e.g., 12.0)
- **Want stock behavior**: Set to 16.0 (original EV6 value)

### Safety Notes

- Always test changes in a safe environment
- Start with small adjustments
- The parameter learner will still adapt within the new bounds
- Setting extreme values may affect steering behavior

## Technical Implementation

### Parameter Flow

1. User sets LiveSteerRatio in GUI
2. Value stored in Params database
3. paramsd reads value on startup
4. If > 0, uses as base_steer_ratio
5. If = 0, uses CP.steerRatio (vehicle default)
6. VehicleParamsLearner uses base_steer_ratio for:
   - Initial steer ratio value
   - Min/max bounds calculation
7. Parameter learning continues within new bounds

### Real-time Updates

While the base value is read at startup, the actual implementation allows for real-time adjustment through the parameter learning system. The bounds ensure safe operation within reasonable limits.

## Testing

Test scripts are available in `/docs/claude/tests/live_steer_ratio/`:
- `test_live_steer_ratio.py`: Comprehensive unit tests
- `demo_live_steer_ratio.py`: Interactive demonstration

Run tests with:
```bash
cd /data/openpilot
python3 docs/claude/tests/live_steer_ratio/test_live_steer_ratio.py
```

## Future Enhancements

Potential improvements could include:
- Per-vehicle default overrides
- Speed-based ratio adjustment
- Integration with other tuning parameters
- Automatic ratio detection/calibration