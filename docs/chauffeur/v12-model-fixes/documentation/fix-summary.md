# V12 Model Calibration Fix Summary

## Problem
The v12 "Falling Phoenix" model downloaded successfully but wouldn't calibrate (stuck at 0%). Investigation revealed multiple issues preventing the model from running properly.

## Root Causes Identified

### 1. Missing Calibration Outputs in v12 Models
- **Issue**: v12 models don't output `wide_from_device_euler` which is required for camera calibration
- **Impact**: Parser would crash when trying to access non-existent outputs
- **Fix**: Added conditional handling in both stock and sunnypilot modeld to check for presence before accessing

### 2. Process Configuration Issue
- **Issue**: `modeld_tinygrad` was defined as a NativeProcess instead of PythonProcess
- **Impact**: Python environment and dependencies (numpy) weren't available
- **Fix**: Changed to PythonProcess in `system/manager/process_config.py`

### 3. Model Runner Selection
- **Issue**: System was trying to use tinygrad runner for v12 models
- **Impact**: Tinygrad dependencies weren't properly configured
- **Fix**: Forced stock modeld runner by setting ModelRunnerTypeCache parameter

## Files Modified

### Stock modeld fixes:
- `/data/openpilot/selfdrive/modeld/parse_model_outputs.py`
  - Added conditional parsing for `wide_from_device_euler`
- `/data/openpilot/selfdrive/modeld/fill_model_msg.py`
  - Added fallback values for missing calibration outputs

### Sunnypilot modeld fixes:
- `/data/openpilot/sunnypilot/modeld_v2/parse_model_outputs_split.py`
  - Fixed v12 plan output parsing condition
  - Added conditional parsing for calibration outputs
- `/data/openpilot/sunnypilot/modeld_v2/fill_model_msg.py`
  - Added fallback handling for missing outputs

### Process configuration:
- `/data/openpilot/system/manager/process_config.py`
  - Changed modeld_tinygrad from NativeProcess to PythonProcess

## Actions Taken on TICI

1. Pulled latest fixes from chauffeur-dev2 branch
2. Set ModelRunnerTypeCache to force stock modeld (value: 2)
3. Rebooted device to apply all changes

## Verification Steps (When Car Powers On)

1. Check if modeld is running:
   ```bash
   ssh commaCar 'ps aux | grep modeld | grep -v grep'
   ```
   Should see `selfdrive.modeld.modeld` running (not modeld_tinygrad)

2. Check calibration status:
   ```bash
   ssh commaCar 'source ~/.bash_profile && python3 -c "from common.params import Params; p = Params(); print(f\"CalibrationParams: {p.get('CalibrationParams')}\")"'
   ```

3. Check model output debug log (if created):
   ```bash
   ssh commaCar 'cat /tmp/model_keys.txt'
   ```

4. Monitor tmux session for any errors:
   ```bash
   ssh commaCar 'tmux attach -t comma'
   ```
   (Detach with Ctrl+B, then D)

## Expected Outcome

After reboot with these fixes:
- Stock modeld should start successfully
- v12 model should run without crashes
- Calibration should progress from 0%
- No Python import errors

## Fallback Options

If issues persist:
1. Verify ModelRunnerTypeCache is set to 2 (stock)
2. Check for any remaining process errors in tmux
3. Consider switching to a different model temporarily
4. Review /data/openpilot/selfdrive.log for detailed errors

## Technical Details

The v12 models are "Generation 12" models that have simplified outputs:
- No Multi-Hypothesis Planning (MHP) 
- Simplified lead/plan outputs
- Missing some calibration-specific outputs

These models require special handling in the parser to avoid accessing non-existent fields and to provide reasonable defaults where needed.