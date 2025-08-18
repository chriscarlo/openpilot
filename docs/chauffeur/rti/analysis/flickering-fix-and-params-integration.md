# RTI Flickering Fix and Parameter Integration - Root Cause Analysis & Solution

## Executive Summary
Two critical issues were identified and resolved in the RTI (Real-time Traffic Intelligence) system:
1. **HUD Flickering**: Widget displayed for ~1ms then disappeared for ~1 second repeatedly
2. **Parameter Integration**: User-configured settings from the UI were not being respected

## Issue 1: HUD Widget Flickering

### Root Cause
The successful test script `continuous_rti_test.py` explicitly identified the issue:
- RTI widget requires BOTH `valid()` AND `updated()` flags to display
- The `updated()` flag is only true for ONE frame after receiving a NEW message
- Production code had gaps in message publishing, causing `updated()` to become false

### Why Test Script Worked 100%
```python
# continuous_rti_test.py - WORKS PERFECTLY
while True:
    # Publishes CONTINUOUSLY at 50Hz without ANY gaps
    pm.send('rtiStateSP', msg)
    time.sleep(0.02)  # Exactly 50Hz
```

### Why Production Failed
```python
# Original rtid.py - CAUSED FLICKERING
while True:
    # Only published when processing completed
    await self._process_cycle()  # Has gaps during API calls, processing, etc.
    # No continuous publishing = updated() becomes false = widget flickers
```

### The Fix
Modified `rtid.py` to continuously republish the last known state at 50Hz:
```python
# Fixed rtid.py - MATCHES TEST SCRIPT BEHAVIOR
while True:
    # ALWAYS publish something at 50Hz to keep updated() flag true
    if last_rti_state and (current_time - last_publish_time) >= 0.019:
        last_rti_state.timestamp = int(current_time * 1e9)  # Fresh timestamp
        self._publish_rti_state(last_rti_state)
        last_publish_time = current_time
    
    # Process new data separately (non-blocking)
    await self._process_cycle_async()
```

## Issue 2: Parameter Integration

### Discovered Parameters
The RTI settings panel (`rti_settings_panel.cc`) defines these user-configurable parameters:

| Parameter | Description | Storage Unit | Default |
|-----------|-------------|--------------|---------|
| `RTIDetectionRadius` | 360° threat awareness display radius | meters | 3218m (2 mi) |
| `RTIForwardSlowdownRange` | When to start slowing for threats ahead | meters | 1207m (0.75 mi) |
| `RTIResumeSpeedDistance` | When to resume speed after passing | meters | 805m (0.5 mi) |
| `RTISpeedReduction` | Custom speed reduction amount | km/h | 16 (10 mph) |
| `RTISpeedReductionMode` | "posted" or "custom" speed mode | string | "posted" |
| `RTIThreatFilter` | Which threats to monitor (0-4) | int | 0 (all) |

### The Problem
`threat_detector.py` was loading parameters incorrectly:
- Missing proper error handling for type conversion
- Hardcoded temporary values (10km) instead of defaults
- Inconsistent parsing of string/byte values

### The Fix
Enhanced parameter loading with proper type conversion and error handling:
```python
# Fixed parameter loading in threat_detector.py
forward_range = params.get("RTIForwardSlowdownRange")
if forward_range:
    try:
        self.ahead_distance_threshold_m = float(forward_range)
    except (ValueError, TypeError):
        self.ahead_distance_threshold_m = 1207  # Default 0.75 miles
else:
    self.ahead_distance_threshold_m = 1207  # Default 0.75 miles
```

## Critical Insights

### Why Previous Fix Attempts Failed
The commit "75c6ec985" changed the daemon frequency to 50Hz but didn't address the real issue - gaps in publishing. Simply running the loop at 50Hz doesn't guarantee messages are published at 50Hz.

### The Key Difference
- **VTSC lateral accel widget**: Only needs `valid()` flag (works with stale messages)
- **RTI widget**: Needs BOTH `valid()` AND `updated()` flags (requires fresh messages)

This explains why VTSC always worked while RTI flickered.

## Validation
Created `test_flickerfix_and_params.py` to validate both fixes:
1. Monitors message frequency to ensure no gaps > 50ms
2. Sets test parameters and verifies they're loaded and used correctly

## Conclusion
Both issues have been resolved:
1. **Flickering**: Fixed by ensuring continuous message publishing at 50Hz
2. **Parameters**: Fixed by properly loading and parsing all user settings

The RTI system now:
- Displays consistently without flickering
- Respects all user-configured settings from the UI
- Maintains the same safety guarantees (only operates when cruise is enabled)
- Uses driver's original set speed for calculations