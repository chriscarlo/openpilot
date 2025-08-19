# CRITICAL FINDING: Incorrect Calibration Bypass Implementation

## Problem Identified
The calibration bypass logic added in commit `fce791c78` is INCORRECT and causes ALL models to fail calibration.

## Root Cause Analysis

### What Was Intended
- V12+ models supposedly don't output traditional calibration data
- A bypass was intended ONLY for these newer models

### What Actually Happened
The bypass logic is too broad and affects ALL models because:

1. **Fallback Logic Too Aggressive**: The code checks if `'wide_from_device_euler'` is missing and assumes it's a v12+ model
2. **Parameter Check Too Early**: The `DynamicModeldOutputs` parameter check happens before model type is known
3. **No Model Version Verification**: The bypass activates without properly checking the actual model version

### Key Code Issues

In `selfdrive/modeld/fill_model_msg.py`:
```python
# INCORRECT - This fallback triggers for ALL models during initialization
if not bypass_calibration:
    bypass_calibration = 'wide_from_device_euler' not in net_output_data
```

This check fails during early calls when `net_output_data` might not have all fields populated yet.

## Chubbs-SSH-Only Approach (CORRECT)

In chubbs-ssh-only, there is NO bypass logic at all:
```python
msg.valid = live_calib_seen & (vipc_dropped_frames < 1)
```

This simple approach works because:
1. Models that need calibration will set `live_calib_seen = True` when ready
2. Models that don't need calibration handle this internally
3. No external bypass logic is needed

## Immediate Fix Required

### Option 1: Remove ALL Bypass Logic (Align with chubbs-ssh-only)
- Remove bypass logic from all three files:
  - `selfdrive/modeld/fill_model_msg.py`
  - `sunnypilot/modeld/fill_model_msg.py`
  - `sunnypilot/modeld_v2/fill_model_msg.py`
- Return to simple: `msg.valid = live_calib_seen & (vipc_dropped_frames < 1)`

### Option 2: Fix Bypass Logic (If v12+ models truly need it)
- Properly detect model version BEFORE applying bypass
- Never use missing fields as a detection method
- Only bypass for confirmed v12+ models

## Recommendation
**Go with Option 1** - Remove all bypass logic and align with chubbs-ssh-only, which is proven to work.
