# SLC Car-Only Mode Fix for Hyundai CANFD

## Problem
After the RTI refactoring, the Speed Limit Controller (SLC) car-only mode stopped working for Hyundai CANFD vehicles. The dashboard speed limit data from the car's CAN bus was not reaching the SLC.

## Root Cause
The issue was **NOT** caused by RTI interference as initially suspected. The actual problem was a missing field in the CarStateSP dataclass:

1. **Hyundai carstate.py** reads dashboard speed limit from `FR_CMR_02_100ms` CAN message
2. It attempts to set `ret_sp.speedLimit` on a `structs.CarStateSP()` object
3. **Problem**: The `structs.CarStateSP` dataclass in `opendbc/car/structs.py` was empty (just `pass`)
4. The `speedLimit` field didn't exist, causing an AttributeError (silently caught)
5. The speed limit data never made it to the SLC resolver

## Data Flow Analysis
```
CAN Bus (FR_CMR_02_100ms)
    ↓
Hyundai carstate.py:328
    ret_sp.speedLimit = float(raw) * speed_factor  ← FAILED HERE
    ↓
card.py converts structs.CarStateSP → custom.CarStateSP
    ↓
carStateSP message published
    ↓
speed_limit_resolver.py reads carStateSP.speedLimit
    ↓
SLC uses the speed limit
```

## The Fix
Added the missing `speedLimit` field to `structs.CarStateSP`:

```python
# /projects/chauffeur/data/openpilot/opendbc_repo/opendbc/car/structs.py:132-133
@auto_dataclass
class CarStateSP:
  speedLimit: float = auto_field()  # Speed limit from car dashboard in m/s
```

This now matches the capnp definition in `cereal/custom.capnp`:
```capnp
struct CarStateSP @0xb86e6369214c01c8 {
  speedLimit @0 :Float32;  # m/s
}
```

## Verification
- ✅ CarStateSP now has speedLimit field
- ✅ Field converts correctly to capnp format
- ✅ Hyundai CANFD data flow works end-to-end
- ✅ All speed limit values handled correctly (0-252 km/h, invalid values)
- ✅ Build completed successfully

## CAN vs CANFD Note
The fix applies to both CAN and CANFD Hyundai variants since they both use the same `structs.CarStateSP` dataclass. The difference is only in which CAN messages are available - CANFD vehicles have `FR_CMR_02_100ms` with dashboard speed limits.

## Files Modified
- `/projects/chauffeur/data/openpilot/opendbc_repo/opendbc/car/structs.py` - Added speedLimit field

## Test Scripts Created
- `test_slc_fix_verification.py` - Comprehensive test suite for the fix
- `test_hyundai_slc_live.py` - Live monitoring of speed limit data flow
- `debug_slc_patch.py` - Debug logging instructions