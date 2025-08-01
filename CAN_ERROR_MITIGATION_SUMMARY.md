# CAN Error Mitigation for 2023 Kia EV6

## Problem
The 2023 Kia EV6 with CAN-FD experiences intermittent CAN communication issues that trigger immediate disengagement of OpenPilot with red warning messages. These are often spurious errors that don't represent actual communication failures.

## Solution Implemented
Modified the Hyundai CarInterface to implement a consecutive error threshold before triggering CAN error events.

### Changes Made
1. **File Modified**: `/data/openpilot/opendbc_repo/opendbc/car/hyundai/interface.py`

2. **Implementation Details**:
   - Added counters `can_invalid_cnt` and `can_timeout_cnt` to track consecutive CAN errors
   - Set threshold of 20 consecutive errors before triggering alerts (`CAN_ERROR_THRESHOLD = 20`)
   - Errors reset to 0 when valid CAN messages are received
   - Overrides `canValid` and `canTimeout` flags until threshold is reached
   - Added logging to track when mitigation is active

3. **Behavior**:
   - Spurious single or intermittent CAN errors will not cause disengagement
   - Only persistent CAN failures (20+ consecutive) will trigger the safety alerts
   - Logging will show:
     - Warning when mitigation starts (1st error)
     - Error when approaching threshold (19th error)
   - Once threshold is exceeded, normal error handling resumes

## Testing
To verify the implementation:
1. Monitor logs for "CAN error mitigation active" messages
2. Intermittent CAN issues should not cause disengagement
3. Persistent CAN failures (20+ consecutive) should still trigger safety alerts

## Rollback
To revert these changes, remove the `__init__` and `update` methods from the Hyundai CarInterface class.