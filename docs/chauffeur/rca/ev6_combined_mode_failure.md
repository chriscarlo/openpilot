# Root Cause Analysis: EV6 Speed Limit Controller Combined Mode Failure

## Problem Statement
The speed limit controller's "combined" mode is not working properly on the EV6. It ignores car dashboard data both when the dashboard value is higher than map data AND when the dashboard is the only active source.

## Root Cause
**The EV6 fingerprint does not contain the dashboard speed limit CAN messages (0x1FA or 0x162) on the camera bus, preventing the system from ever parsing dashboard speed limit data.**

## Evidence Chain

### 1. Combined Mode Logic is Correct
Location: `sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py:193-211`

The `_get_source_solution_according_to_policy` method correctly implements taking the HIGHER speed limit:
```python
elif current_speed_limit_policy == Policy.combined:
    self._speed_limit = max(v_map, v_car)
```

### 2. Dashboard Speed Limit Parsing Requires Flags
Location: `opendbc/car/hyundai/carstate.py:163-167`

Messages are only added to the CAN parser if flags are set:
```python
if CP.flags & HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR:
    cam_messages.append(("FR_CMR_02_100ms", 10))
if CP.flags & HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_CCNC:
    cam_messages.append(("CCNC_0x162", 20))
```

### 3. Flags Are Set Based on Fingerprint
Location: `opendbc/car/hyundai/interface.py:115-119`

Flags are only set if specific messages exist in the fingerprint:
```python
if 0x1FA in fingerprint[CAN.CAM]:
    ret.flags |= HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR.value
if 0x162 in fingerprint[CAN.CAM]:
    ret.flags |= HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_CCNC.value
```

### 4. Data Flow Breakdown
1. **Fingerprint Missing Messages** → No 0x1FA or 0x162 in EV6 fingerprint
2. **Flags Not Set** → HAS_DASHBOARD_SPEED_LIMIT_* flags remain false
3. **Messages Not Parsed** → carstate.py doesn't add messages to CAN parser
4. **Speed Limit Always 0** → ret_sp.speedLimit = 0.0
5. **Combined Mode Fails** → max(map_speed, 0.0) always equals map_speed

## Test Results

### Unit Test Output
```
Test Case 1: Current EV6 fingerprint (missing 0x1FA and 0x162)
  Flags set: 0 (NO dashboard speed limit support)
  Result: Dashboard speed limit will ALWAYS be 0.0

Impact on Combined Mode:
  Map speed limit: 60.0 km/h
  Dashboard speed limit: 0.0 km/h (always 0!)
  Combined result: 60.0 km/h (always uses map only!)
```

## Solution
Add message ID 0x1FA (FR_CMR_02_100ms) and/or 0x162 (CCNC_0x162) to the EV6 fingerprint on the camera bus. This requires:

1. Capturing these messages from an actual EV6 (if they exist)
2. Adding them to the fingerprint database
3. Verifying the messages are parsed correctly

## Diagnostic Scripts Created
- `test_ev6_speed_limit_simple.py` - Monitor CAN bus for speed limit messages on real EV6
- `test_ev6_speed_limit_unit.py` - Unit test demonstrating the issue
- `verify_ev6_speed_limit.py` - Summary of root cause and solution

## Key Files Modified by Fix
No files modified - RCA only as requested. The fix would require updating the EV6 fingerprint data.