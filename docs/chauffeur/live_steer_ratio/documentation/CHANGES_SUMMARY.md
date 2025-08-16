# LiveSteerRatio Changes Summary

## Quick Reference

### What Changed
- KIA EV6 default steer ratio: 16.0 → 13.43
- Added LiveSteerRatio parameter for real-time adjustment
- Added GUI control in Hyundai settings menu

### Files Modified
1. `opendbc/car/hyundai/values.py` - Changed EV6 default
2. `selfdrive/locationd/paramsd.py` - Added parameter support
3. `selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/hyundai_settings.cc` - Added GUI
4. `common/params_keys.h` - Added parameter key
5. `selfdrive/locationd/test/test_paramsd.py` - Updated tests

### For Users
- Go to Settings → Vehicle → Hyundai → Live Steering Ratio
- Enter 0 to use default (13.43 for EV6)
- Enter 1-25 for custom ratio
- Changes apply immediately (no restart needed)

### For Developers
- Parameter stored as string in Params database
- Read at paramsd startup
- Used as base for parameter learning bounds
- Bounds: 0.5x to 2.0x of base value