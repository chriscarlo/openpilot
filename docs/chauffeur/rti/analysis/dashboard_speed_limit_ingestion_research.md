# RTI Dashboard Speed Limit Ingestion - Research Findings

## Executive Summary
RTI currently only ingests speed limit data from map sources (liveMapDataSP) and completely misses car dashboard speed limit data (carStateSP). The Speed Limit Controller (SLC) successfully uses both sources, but RTI never subscribes to carStateSP, creating a dangerous blind spot.

**Critical Architecture Note**: Dashboard speed limit data (message 0x1FA) arrives on different CAN buses depending on vehicle type:
- **CANFD vehicles** (e.g., EV6): Message on ECAN bus
- **Regular CAN vehicles**: Message on CAM bus

This bus distinction has caused crashes when handled incorrectly. However, since RTI never subscribes to carStateSP at all, it avoids this complexity but also misses all dashboard data.

## The Two Speed Limit Data Sources

### 1. Map-Based Speed Limits (liveMapDataSP)
- **Source**: OSM map data, navigation systems
- **Service**: `liveMapDataSP`
- **Field**: `speedLimit` (m/s)
- **Validity Check**: `speedLimitValid` flag
- **RTI Status**: ✅ SUBSCRIBED AND USED

### 2. Dashboard Speed Limits (carStateSP)
- **Source**: Car's traffic sign recognition cameras
- **Service**: `carStateSP`
- **Field**: `speedLimit` (m/s)
- **Raw CAN Source**: Message 0x1FA (FR_CMR_02_100ms)
  - **CANFD vehicles**: ECAN bus (e.g., EV6, Genesis GV70)
  - **Regular CAN vehicles**: CAM bus (older Hyundai/Kia)
- **RTI Status**: ❌ NOT SUBSCRIBED, COMPLETELY IGNORED

## How Dashboard Speed Limits Flow Through the System

### Step 1: CAN Bus Message (CRITICAL: Bus Varies by Vehicle Type)

**For CANFD Vehicles (e.g., EV6, Genesis GV70):**
```
ECAN Bus → Message 0x1FA (FR_CMR_02_100ms)
           Signal: ISLW_SpdCluMainDis
           Units: km/h (raw value)
```

**For Regular CAN Vehicles (older Hyundai/Kia):**
```
CAM Bus → Message 0x1FA (FR_CMR_02_100ms)
          Signal: ISLW_SpdCluMainDis
          Units: km/h (raw value)
```

⚠️ **CRITICAL SAFETY NOTE**: Reading from the wrong bus (e.g., CAM instead of ECAN for CANFD vehicles) has caused crashes. The bus selection MUST match the vehicle type.

### Step 2: Car Interface Parsing
```python
# opendbc/car/hyundai/carstate.py
if "FR_CMR_02_100ms" in cp.vl:
    speed_limit_raw = cp.vl["FR_CMR_02_100ms"]["ISLW_SpdCluMainDis"]
    if speed_limit_raw != 0 and speed_limit_raw != 255:  # Valid values
        ret_sp.speedLimit = speed_limit_raw * 0.277778  # km/h to m/s
    else:
        ret_sp.speedLimit = 0.0  # Invalid/no limit
```

### Step 3: Publishing to System
```python
# selfdrive/car/card.py (line 269)
self.pm.send('carStateSP', cs_sp_send)
```

### Step 4: Speed Limit Controller Ingestion
```python
# sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py
def _get_from_car_state(self, sm: messaging.SubMaster) -> None:
    self._reset_limit_sources(Source.car_state)
    self._limit_solutions[Source.car_state] = sm['carStateSP'].speedLimit  # ✅ SLC reads it
    self._distance_solutions[Source.car_state] = 0.
```

### Step 5: RTI's Current Implementation
```python
# sunnypilot/rtid/rtid.py
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation',
    'carState',
    'liveMapDataSP'  # ✅ Map data subscribed
    # carStateSP missing! ❌ Dashboard data NOT subscribed
], ignore_avg_freq=True)

def _get_current_speed_limit(self) -> float:
    """Get current posted speed limit from map data."""
    try:
        map_data = self.sm['liveMapDataSP']
        if map_data.speedLimitValid:
            return float(map_data.speedLimit)  # Only uses map data!
    except Exception as e:
        cloudlog.debug(f"RTI: Could not get speed limit from map data: {e}")
    
    return 0.0  # Falls back to hardcoded default
```

## Critical Finding: RTI's Blind Spot

### What Speed Limit Controller Does (Correct Approach)
The SLC subscribes to BOTH sources and uses a resolver to combine them:
```python
# Combined mode: MAX(dashboard, map)
# Dashboard only: Use dashboard when map unavailable
# Map only: Use map when dashboard unavailable
```

### What RTI Does (Incomplete Approach)
RTI only subscribes to map data and ignores dashboard completely:
```python
# Only uses liveMapDataSP
# Never sees carStateSP
# Falls back to hardcoded 25 m/s (56 mph) when no map data
```

## The Two-Layer Problem Explained

### Layer 1: Missing Dashboard Data
- RTI doesn't subscribe to carStateSP
- Can't see traffic sign recognition data
- Misses speed limits that car's cameras detect

### Layer 2: Inadequate Fallback
- When map data unavailable, uses hardcoded 25 m/s
- Then applies 75% reduction factor in "posted" mode
- Results in dangerous 42 mph on 65 mph freeways

## Real-World Impact Scenarios

### Scenario 1: Construction Zone
- **Dashboard**: Detects temporary 45 mph sign
- **Map**: Shows outdated 65 mph limit
- **SLC**: Correctly uses 65 mph (MAX of both)
- **RTI**: Only sees 65 mph, misses temporary limit

### Scenario 2: Rural Highway
- **Dashboard**: Detects 55 mph sign
- **Map**: No data available
- **SLC**: Uses dashboard 55 mph
- **RTI**: Falls back to hardcoded 56 mph, then reduces to 42 mph

### Scenario 3: School Zone
- **Dashboard**: Detects electronic 25 mph when flashing
- **Map**: Shows static 35 mph
- **SLC**: Uses 35 mph (MAX of both)
- **RTI**: Only sees 35 mph, misses active school zone

## Data Flow Comparison

### Speed Limit Controller (Complete)
```
CAN Bus → carStateSP ↘
                      → SpeedLimitResolver → Combined Speed Limit
Map Data → liveMapDataSP ↗
```

### RTI (Incomplete)
```
CAN Bus → carStateSP → [NOT CONNECTED TO RTI]
Map Data → liveMapDataSP → RTI → Incomplete Speed Limit
```

## Key Code Locations

### Dashboard Speed Limit Parsing
- **DBC Definition**: `opendbc/dbc/hyundai_canfd_generated.dbc:720`
  - Message 0x1FA, Signal ISLW_SpdCluMainDis
- **Fingerprint Check**: `opendbc/car/hyundai/interface.py:72-77`
  - Checks ECAN bus for CANFD cars
- **Message Parsing**: `opendbc/car/hyundai/carstate.py:343-362`
  - Converts km/h to m/s
- **Publishing**: `selfdrive/car/card.py:269`
  - Sends carStateSP message

### Speed Limit Controller Usage
- **Subscription**: Part of controlsd's SubMaster
- **Reading**: `speed_limit_resolver.py:56`
  - `sm['carStateSP'].speedLimit`

### RTI's Missing Link
- **Subscription List**: `sunnypilot/rtid/rtid.py:33-37`
  - Missing carStateSP
- **Speed Limit Getter**: `sunnypilot/rtid/rtid.py:159-166`
  - Only reads liveMapDataSP

## Why This Matters for "Posted Speed Limit" Mode

When users select "posted speed limit" mode in RTI, they expect:
1. RTI to use actual posted speed limits
2. No arbitrary reductions
3. Complete awareness of all speed limit sources

Currently, RTI:
1. Misses dashboard-detected speed limits entirely
2. Falls back to hardcoded values when map data unavailable
3. Creates dangerous situations by recommending speeds well below actual limits

## Research Conclusion

The Speed Limit Controller demonstrates the correct approach: subscribing to and combining both dashboard (carStateSP) and map (liveMapDataSP) speed limit sources. RTI's failure to subscribe to carStateSP creates a critical blind spot, especially dangerous when map data is unavailable or outdated.

The car's dashboard speed limit data flows correctly through the system:
- CAN bus → Car interface → carStateSP → Available for consumption

But RTI never subscribes to this service, relying solely on map data and dangerous hardcoded fallbacks.

## Critical CAN vs CANFD Bus Architecture Difference

### Why Bus Selection Matters for Safety

The dashboard speed limit message (0x1FA) location varies by vehicle architecture:

| Vehicle Type | Bus Location | Bus Index | Parser Type |
|-------------|--------------|-----------|-------------|
| CANFD | ECAN | 0 or 1 | pt_messages |
| Regular CAN | CAM | 2 | cam_messages |

**Why This Is Safety-Critical:**
1. **Wrong Bus = No Data**: Attempting to read 0x1FA from CAM bus on a CANFD vehicle yields nothing
2. **Wrong Parser = Crash Risk**: Mixing up parsers has caused system crashes in production
3. **Silent Failure**: System doesn't error - it just never sees dashboard speed limits

### How This Affects RTI

Since RTI never subscribes to carStateSP, it avoids the CAN/CANFD complexity entirely. However, this also means:
- RTI misses ALL dashboard speed limit data regardless of vehicle type
- The complexity of bus selection is irrelevant because RTI never attempts to read it
- This creates a consistent blind spot across all vehicle types

### If RTI Were to Subscribe to carStateSP

The good news is that carStateSP abstracts away the bus complexity:
1. Car interface handles CAN vs CANFD bus selection internally
2. carStateSP.speedLimit provides unified access regardless of vehicle type
3. RTI wouldn't need to know about bus differences - just subscribe and read

The car interface already does the heavy lifting:
- CANFD vehicles: Reads from ECAN via pt_messages
- Regular CAN: Reads from CAM via cam_messages
- Result: carStateSP.speedLimit contains the value regardless

## Technical Recommendation (Research Only - No Implementation)

To achieve parity with the Speed Limit Controller, RTI would need to:
1. Subscribe to carStateSP in addition to liveMapDataSP
2. Implement logic to intelligently combine both sources
3. Remove hardcoded fallback values in favor of actual data

This would ensure RTI has complete visibility into all available speed limit information, matching the sophisticated approach already implemented in the Speed Limit Controller.