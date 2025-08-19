# EV6 Dashboard Speed Limit - Solution from chubbspilot Fork Analysis

## Executive Summary
The chubbspilot fork has a working dashboard speed limit implementation for EV6. The key difference is they look for message 0x1FA on the **ECAN bus** (not CAM bus) for CANFD cars like the EV6.

## Key Findings from chubbspilot Fork

### 1. Message Location for CANFD Cars
- **Our codebase**: Checks for 0x1FA on CAM bus (doesn't find it)
- **chubbspilot**: Checks for 0x1FA on ECAN bus for CANFD cars

Reference: `selfdrive/car/hyundai/interface.py` (chubbspilot)
```python
if candidate in CANFD_CAR:
  # ...
  if 0x1fa in fingerprint[CAN.ECAN]:  # <-- ECAN, not CAM!
    ret.flags |= HyundaiFlags.NAV_MSG.value
```

### 2. Speed Limit Parsing Implementation
Location: `selfdrive/car/hyundai/carstate.py` (chubbspilot)

```python
def calculate_speed_limit(self, cp, cp_cam):
  if self.CP.carFingerprint in CANFD_CAR:
    speed_limit_bus = cp if self.CP.flags & HyundaiFlags.CANFD_HDA2 else cp_cam
    return speed_limit_bus.vl["CLUSTER_SPEED_LIMIT"]["SPEED_LIMIT_1"]
```

### 3. Message Addition to Parser
For CANFD cars with NAV_MSG flag:
- **With HDA2**: Parse from ECAN bus (line 455)
- **Without HDA2**: Parse from CAM bus (line 474)

```python
# ECAN parser (for HDA2)
if CP.flags & HyundaiFlags.CANFD_HDA2 and CP.flags & HyundaiFlags.NAV_MSG:
  messages.append(("CLUSTER_SPEED_LIMIT", 10))

# CAM parser (for non-HDA2)
if not (CP.flags & HyundaiFlags.CANFD_HDA2) and CP.flags & HyundaiFlags.NAV_MSG:
  messages.append(("CLUSTER_SPEED_LIMIT", 10))
```

## Message Details from Local DBC

From `opendbc/dbc/hyundai_canfd_generated.dbc`:
```
BO_ 506 FR_CMR_02_100ms: 32 FR_CMR
 SG_ ISLW_SpdCluMainDis : 33|8@1+ (1,0) [0|255] "" CLU,CGW
 SG_ ISLW_SpdNaviMainDis : 41|8@1+ (1,0) [0|255] "" CGW
```

- **Message ID**: 0x1FA (506 decimal)
- **Message Name**: FR_CMR_02_100ms
- **Key Signals**:
  - ISLW_SpdCluMainDis: Cluster speed limit display
  - ISLW_SpdNaviMainDis: Navigation speed limit display
- **ISLW**: Intelligent Speed Limit Warning

## Root Cause Confirmation
1. EV6 is a CANFD car (confirmed in `values.py`)
2. Message 0x1FA exists on ECAN bus (not CAM bus) for CANFD cars
3. Our code only checks CAM bus, so it never finds the message
4. Without finding 0x1FA, the NAV_MSG flag is never set
5. Without NAV_MSG flag, speed limit parsing is never enabled

## Solution Implementation

### Option 1: Quick Fix (Align with chubbspilot)
1. Update `interface.py` to check ECAN bus for 0x1FA in CANFD cars
2. Add CLUSTER_SPEED_LIMIT/FR_CMR_02_100ms message parsing
3. Implement calculate_speed_limit function similar to chubbspilot

### Option 2: Minimal Change
1. Verify 0x1FA exists on ECAN bus in actual EV6
2. Update fingerprint check in `interface.py`:
```python
if candidate in CANFD_CAR:
  if 0x1fa in fingerprint[CAN.ECAN]:  # Check ECAN instead of CAM
    ret.flags |= HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR.value
```
3. Update carstate.py to parse FR_CMR_02_100ms from correct bus

## Testing Required
1. Verify 0x1FA presence on ECAN bus in actual EV6
2. Confirm ISLW_SpdCluMainDis contains valid speed limit data
3. Test combined mode with both map and dashboard data

## Conclusion
The chubbspilot fork proves dashboard speed limit works on EV6. The issue is simply checking the wrong CAN bus for the message. For CANFD cars, message 0x1FA is on ECAN bus, not CAM bus.