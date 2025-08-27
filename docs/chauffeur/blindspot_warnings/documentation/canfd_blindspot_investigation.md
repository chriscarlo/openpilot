# Hyundai/Kia CANFD Blindspot Warning Investigation

## Problem
Blindspot warnings are not working on Hyundai/Kia CANFD cars (like EV6) even though:
1. The UI toggle exists in Settings → Visuals → "Show Blind Spot Warnings"
2. The UI rendering code exists to display blindspot indicators
3. The carstate code tries to read blindspot data
4. The DBC files contain the necessary message definitions

## Root Cause
The BLINDSPOTS_REAR_CORNERS message (0x1BA / 442) is not being added to the CAN parser for CANFD vehicles. The code in `update_canfd()` tries to read from this message, but it was never configured in the parser, causing the blindspot data to never be available.

## Current Implementation Status

### Working Components
1. **UI Toggle**: `selfdrive/ui/sunnypilot/qt/offroad/settings/visuals_panel.cc:29-34`
   - Parameter: "BlindSpot"
   
2. **UI Rendering**: `selfdrive/ui/sunnypilot/qt/onroad/model.cc:27-48`
   - Reads leftBlindspot/rightBlindspot from carState
   - Renders warning polygons when detected

3. **Interface BSM Detection**: `opendbc/car/hyundai/interface.py:48,91`
   - CANFD: Checks for 0x1E5 (BLINDSPOTS_FRONT_CORNER_1) in ECAN fingerprint
   - CAN: Checks for 0x58B (LCA11) in bus 0 fingerprint

4. **DBC Message Definition**: `opendbc/dbc/hyundai_canfd_generated.dbc`
   ```
   BO_ 442 BLINDSPOTS_REAR_CORNERS: 24 XXX
    SG_ FL_INDICATOR : 46|6@0+ (1,0) [0|1] "" XXX
    SG_ FR_INDICATOR : 54|6@0+ (1,0) [0|63] "" XXX
   ```

### Broken Component
**CAN Parser Configuration**: `opendbc/car/hyundai/carstate.py:337-351`
- The `get_can_parsers_canfd()` method does NOT add BLINDSPOTS_REAR_CORNERS to the message list
- This causes `cp.vl["BLINDSPOTS_REAR_CORNERS"]` to fail in `update_canfd()`

## Solution
Add BLINDSPOTS_REAR_CORNERS message to the CAN parser when BSM is enabled:

```python
def get_can_parsers_canfd(self, CP):
    msgs = []
    if not (CP.flags & HyundaiFlags.CANFD_ALT_BUTTONS):
      msgs += [
        ("CRUISE_BUTTONS", 50)
      ]
    
    # Dashboard speed limit for CAN-FD platforms
    msgs += [
      ("FR_CMR_02_100ms", 10),
    ]
    
    # Add blindspot monitoring message when BSM is enabled
    if CP.enableBsm:
      msgs += [
        ("BLINDSPOTS_REAR_CORNERS", 20),  # 20Hz update rate
      ]
    
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], msgs, CanBus(CP).ECAN),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], CanBus(CP).CAM),
    }
```

## Message Details
- **Message ID**: 0x1BA (442 decimal)
- **Message Name**: BLINDSPOTS_REAR_CORNERS
- **Bus**: ECAN (Bus 0 or 1 depending on LKA steering config)
- **Update Rate**: 20 Hz (based on chubbspilot implementation)
- **Key Signals**:
  - FL_INDICATOR: Front-left blindspot indicator (bit field)
  - FR_INDICATOR: Front-right blindspot indicator (bit field)
  
## Data Flow
1. **CAN Bus**: Message 0x1BA transmitted on ECAN bus
2. **Fingerprint Check**: interface.py checks for 0x1E5 in ECAN to enable BSM
3. **Parser Config**: BLINDSPOTS_REAR_CORNERS added to pt_messages (ECAN)
4. **Message Parse**: update_canfd reads FL_INDICATOR/FR_INDICATOR
5. **State Update**: ret.leftBlindspot/ret.rightBlindspot populated
6. **UI Rendering**: model.cc displays warning polygons when detected

## Testing Notes
- The chubbspilot fork (exp04 branch) has this working implementation
- Some models (like K5_2025) use ALT signal variants (FL_INDICATOR_ALT)
- Regular CAN cars use LCA11 message (0x58B) which works correctly