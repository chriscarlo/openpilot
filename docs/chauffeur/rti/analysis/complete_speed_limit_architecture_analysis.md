# Complete RTI Speed Limit Architecture Analysis

## The Full Picture: A Multi-Layer Architectural Gap

After extensive investigation, I've identified that RTI's speed limit handling has fundamental architectural gaps that create dangerous situations. Here's the complete analysis.

## The Three Data Sources RTI Should Consider

### 1. Map-Based Speed Limits (liveMapDataSP) ✅ PARTIALLY FIXED
- **Source**: OpenStreetMap, navigation data
- **Current Status**: Now subscribed after initial fix
- **Reliability**: Good for static limits, poor for temporary changes

### 2. Dashboard Speed Limits (carStateSP) ❌ STILL MISSING
- **Source**: Car's traffic sign recognition cameras
- **Current Status**: Never subscribed, completely ignored
- **Reliability**: Excellent for temporary signs, construction zones, school zones

### 3. Waze/Traffic API Speed Limits ⚠️ INCONSISTENT
- **Source**: Waze alerts sometimes include speed limit data
- **Current Status**: Read but often null/empty
- **Reliability**: Sporadic, user-reported, not authoritative

## The CAN Bus Architecture Complexity

### Critical Safety Distinction
Dashboard speed limit data arrives on different buses based on vehicle type:

```
CANFD Vehicles (e.g., EV6, Genesis GV70):
├── ECAN Bus (0 or 1) → Message 0x1FA → pt_messages parser
├── ACAN Bus (1 or 0) → Other ADAS messages
└── CAM Bus (2) → Camera messages

Regular CAN Vehicles (older Hyundai/Kia):
├── CAN Bus (0) → Most messages
├── HDA Bus (1) → ADAS messages  
└── CAM Bus (2) → Message 0x1FA → cam_messages parser
```

**Critical Finding**: Getting the bus wrong has caused production crashes. The EV6 fix moved 0x1FA parsing from CAM to ECAN bus for CANFD vehicles.

## How Speed Limit Controller Handles This Correctly

The SLC demonstrates best practices for speed limit handling:

### 1. Subscribes to Both Sources
```python
# In controlsd's SubMaster
'carStateSP',      # Dashboard speed limits
'liveMapDataSP',   # Map speed limits
```

### 2. Implements Sophisticated Combination Logic
```python
# Combined Mode Algorithm (speed_limit_resolver.py:117-145)
if both sources have data:
    if values are equal:
        prefer map_data  # More stable source
    else:
        use HIGHER value  # Safety: don't force below either limit
elif only one has data:
    use that source
else:
    no speed limit available
```

### 3. Policy Options for Different Scenarios
- `car_state_only`: Trust only dashboard
- `map_data_only`: Trust only maps
- `combined`: Use higher of both (default)
- `car_state_priority`: Prefer dashboard, fall back to map
- `map_data_priority`: Prefer map, fall back to dashboard

## How RTI Currently Fails

### Current Implementation Gaps
```python
# RTI's Limited Subscription (rtid.py)
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation',
    'carState',
    'liveMapDataSP'  # ✅ Map data only
    # ❌ Missing carStateSP!
])

# RTI's Single-Source Logic
def _get_current_speed_limit(self) -> float:
    try:
        map_data = self.sm['liveMapDataSP']
        if map_data.speedLimitValid:
            return float(map_data.speedLimit)
    except:
        pass
    return 0.0  # Falls back to nothing!
```

### Cascading Failure Modes

#### Scenario 1: Construction Zone (Critical Safety Issue)
```
Reality: 45 mph construction zone on 65 mph highway
Dashboard: Sees temporary 45 mph sign ✅
Map Data: Shows permanent 65 mph ✅
SLC Result: MAX(45, 65) = 65 mph (safe, follows higher limit)
RTI Result: Only sees 65 mph, misses construction zone entirely ❌
```

#### Scenario 2: Rural Road (Data Gap Issue)
```
Reality: 55 mph posted limit
Dashboard: Sees 55 mph sign ✅
Map Data: No coverage ❌
SLC Result: Uses dashboard 55 mph ✅
RTI Result: Falls back to hardcoded 56 mph, then reduces to 42 mph ❌
```

#### Scenario 3: School Zone (Dynamic Limit Issue)
```
Reality: 25 mph when lights flashing, 35 mph otherwise
Dashboard: Detects active 25 mph ✅
Map Data: Shows static 35 mph ✅
SLC Result: MAX(25, 35) = 35 mph (follows higher limit)
RTI Result: Only sees 35 mph, misses active school zone ❌
```

## The Abstraction Layer Advantage

The good news: carStateSP abstracts away all CAN/CANFD complexity:

```python
# Car Interface (Handles Bus Complexity)
if CANFD vehicle:
    read 0x1FA from ECAN via pt_messages
else:
    read 0x1FA from CAM via cam_messages

# Publish Unified Interface
publish('carStateSP', {speedLimit: value_in_ms})

# Consumers Just Subscribe and Read
speed_limit = sm['carStateSP'].speedLimit  # No bus knowledge needed!
```

## Why "Posted Speed Limit" Mode Is Broken

When users select "posted speed limit" mode, they expect RTI to:
1. Know the actual posted speed limit
2. Recommend that speed without arbitrary reductions
3. Have complete awareness of all speed limit sources

What actually happens:
1. RTI only sees map data (missing dashboard entirely)
2. Falls back to hardcoded 56 mph when map unavailable
3. Still applied 75% reduction (now fixed) = dangerous 42 mph

## The Complete Fix Would Require

### 1. Subscribe to Dashboard Data
```python
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation',
    'carState',
    'liveMapDataSP',
    'carStateSP'  # ADD THIS
])
```

### 2. Implement Intelligent Combination
```python
def _get_current_speed_limit(self) -> float:
    map_limit = self._get_map_speed_limit()
    dashboard_limit = self._get_dashboard_speed_limit()
    
    # Use same logic as SLC combined mode
    if map_limit > 0 and dashboard_limit > 0:
        return max(map_limit, dashboard_limit)  # Safety: use higher
    elif dashboard_limit > 0:
        return dashboard_limit
    elif map_limit > 0:
        return map_limit
    else:
        return 0.0  # No data available
```

### 3. Remove Hardcoded Fallbacks
Replace `default_speed_limit_ms = 25` with actual data or safe degradation.

## Architectural Lessons

### 1. Single Source of Truth Is Dangerous
- Maps can be outdated
- Dashboard can miss signs
- APIs can be unreliable
- Combination provides resilience

### 2. Abstraction Layers Prevent Disasters
- carStateSP hides CAN/CANFD complexity
- Direct bus access caused crashes
- Higher-level interfaces are safer

### 3. User Expectations vs Reality
- "Posted speed limit" implies all posted limits
- Not just map data
- Not just dashboard data
- But the actual speed limit from any authoritative source

## Conclusion

RTI's speed limit handling is architecturally incomplete. While the initial fix addressed map data subscription and reduction factors, the fundamental gap remains: **RTI has zero visibility into dashboard-detected speed limits**.

This creates dangerous blind spots in exactly the scenarios where accurate speed limit awareness is most critical:
- Temporary construction zones
- Active school zones
- Rural roads without map coverage
- Dynamic electronic speed signs

The Speed Limit Controller demonstrates the correct approach: subscribe to all available sources, combine them intelligently, and never rely on hardcoded fallbacks when real data exists.

The fix is conceptually simple (add carStateSP subscription) but requires careful implementation to maintain safety and avoid the CAN/CANFD pitfalls that have caused crashes before. Fortunately, the carStateSP abstraction layer handles this complexity, making RTI's integration straightforward if implemented correctly.