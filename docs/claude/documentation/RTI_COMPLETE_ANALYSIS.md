# RTI Speed Limit Decision Logic - Complete Line-by-Line Analysis

## The REAL Implementation (Not Simplified Bullshit)

The RTI system uses **EnhancedRoadMatcher** with actual OSM road geometry from mapd, NOT simple proximity circles.

## Data Flow - End to End

### 1. Main Loop (`rtid.py`)
```python
# Line 116: _process_cycle() runs at 1Hz
# Line 122: Gets GPS location from gpsLocationExternal/gpsLocation
# Line 139-142: API fetch only every 30 seconds (rate limiting)
# Line 147-154: Fetches from Waze API, caches for 5 minutes
# Line 174-179: Calls threat_detector.process_threats()
# Line 186-188: Publishes rtiStateSP message
```

### 2. MapD Integration (`enhanced_road_matcher.py`)
```python
# Line 45: Subscribes to 'liveMapDataSP' from mapd
# Line 154: Updates road geometry (non-blocking)
# Line 163: Sets road_geometry_valid flag
# Line 172-174: Parses current road segment (OSM way_id, lanes, barriers)
# Line 177-182: Parses nearby segments within ~500m
```

## OSM Road Matching - The Sophisticated Part

### Road Segment Data Structure (from mapd)
```python
# Lines 199-288: Parses from liveMapDataSP message:
- way_id: OSM road identifier
- centerline: List of coordinates with distance_from_start
- lanes: Width, type (driving/bus/bicycle/parking/shoulder/median)
- barriers: Type (median/guardrail/wall/fence/curb) with coordinates
- level_separation: -1=under, 0=ground, 1=bridge
- road_direction: Bearing at current position
```

### Threat-to-Road Projection (`_find_best_road_projection()`)
```python
# Line 375-376: Limits to 2-3 segments for performance
# Line 383-390: Pre-filters segments > 200-500m away
# Line 393: Projects position to road centerline
# Line 398-400: Tracks best match by minimum distance
# Line 404-406: Early exit if < 1.0m (perfect match)
# Line 418-420: Only returns if < 100m from road
```

### Same Road Detection (`_is_same_road_geometry_based()`)
```python
# Line 331-333: Projects ego position onto road segments
# Line 336-338: Projects threat position onto road segments
# Line 341-342: MUST have same way_id (same OSM road)
# Line 345-346: MUST have same level_separation (no bridge/tunnel mismatch)
# Line 349-350: Checks for median barriers between positions
# Line 353-356: Both must be within road width threshold
```

### Median Barrier Detection (`_has_barrier_between_positions()`)
```python
# Line 494-500: Checks each barrier in road segment
# Line 510-517: Different segments with medians = opposite directions
# Line 520-529: Same segment requires both > 1m from centerline + median exists
```

### Direction Determination (`_get_direction_road_aware()`)
```python
# Line 562-563: Projects both positions onto road
# Line 569-572: Different roads = use GPS bearing (fallback)
# Line 575-576: Gets distance_from_start along centerline
# Line 579-582: Compares centerline distances (NOT GPS distance):
  - threat > ego + 5m = "ahead"
  - threat < ego - 5m = "behind"
  - This accounts for road curves!
```

## Speed Recommendation Logic

### Threat Processing (`threat_detector.py`)

#### Phase 1: Filter and Match
```python
# Line 366: Apply user filter (police/cameras/hazards)
# Line 369: Deduplicate with DBSCAN clustering (100m radius)
# Line 373-377: Process each threat individually
```

#### Phase 2: Single Threat Processing (`_process_single_threat()`)
```python
# Line 435-437: Calculate haversine distance
# Line 440-441: Skip if > 2 miles (HUD detection radius)
# Line 444-446: Call EnhancedRoadMatcher.is_same_road() [OSM matching!]
# Line 449-451: Get direction using road geometry
# Line 456-467: Assign speed limit:
  - Has data: Use it
  - Police/speedTrap without data: 55 mph default
  - Other threats: 0.0 (no speed control)
```

#### Phase 3: Speed Calculation (`calculate_recommendation()`)
```python
# Line 258-270: CRITICAL FILTERING
  # Line 261: Skip if NOT on_same_road (OSM matched!)
  # Line 265-268: Distance thresholds:
    - ahead: 1207m (0.75 miles)
    - behind: 805m (0.5 miles)  
    - left/right: 0 (COMPLETELY IGNORED)
  # Line 269: Only adds if ahead/behind AND within threshold

# Line 272-273: Return (0.0, False) if no relevant threats
# Line 276: Filter for ONLY ahead threats
# Line 278-279: Return (0.0, False) if NO ahead threats

# Line 281: Get closest ahead threat
# Line 284-295: Calculate speed:
  - "posted": threat's speed or 55 mph default
  - "custom": current - 10 mph (min 22 mph)
  
# Line 299: CRITICAL SAFETY: Never exceed current speed
# Line 302-304: < 300m = reduce to 90% of current
# Line 307: Ensure non-negative
# Line 309: Return (target_speed, True)
```

## RTI Controller Integration

### Additional Safety Layers (`rti_controller.py`)
```python
# Line 73: Requires RTIEnabled parameter
# Line 80-82: Minimum 5 mph operating speed
# Line 85-87: Checks valid rtiStateSP message
# Line 106-108: Requires threatAhead=true and distance > 0
# Line 115-117: Max 1000m activation (more restrictive than backend!)
# Line 147-189: Distance-based reduction:
  - < 100m: 75% of current
  - < 300m: 85% of current
  - < 1000m: 95% of current
# Line 140-144: Only active if recommendation < cruise setpoint
```

## Key Truths About RTI

### What It REALLY Does:
1. **Uses actual OSM road data** via mapd's liveMapDataSP
2. **Projects GPS to road centerlines** - not proximity circles
3. **Checks OSM way_id** - same road means same OSM identifier
4. **Detects median barriers** - won't trigger for opposite traffic
5. **Considers level separation** - bridges/tunnels handled correctly
6. **Direction from centerline distance** - handles curves properly
7. **Only ahead threats matter** - left/right completely ignored for speed
8. **Never accelerates** - fundamental safety constraint

### When It Publishes Speed Limits:
- ✅ Threat is ahead (via centerline distance)
- ✅ On same OSM road (same way_id)
- ✅ No median barrier between
- ✅ Same level (no bridge/tunnel)
- ✅ Within 0.75 miles
- ✅ Speed less than current

### When It Publishes 0.0 (No Intervention):
- ❌ No threats detected
- ❌ Threats only to side/behind
- ❌ Different OSM road
- ❌ Median barrier present
- ❌ Beyond 0.75 miles
- ❌ Would require acceleration

## API Rate Limiting
- Fetches every 30 seconds (120 calls/hour max)
- Caches for up to 5 minutes
- Publishes at 1Hz using cached data

## Fallback Behavior
Only when `road_geometry_valid = false` (no mapd data):
- Uses simple 50m (roads) or 200m (highways) proximity
- Uses GPS bearing for direction
- This is the OLD logic, rarely used

## Summary

The RTI system is sophisticated, using actual road geometry from OSM via mapd. It properly handles:
- Divided highways (median detection)
- Bridges/tunnels (level separation)
- Curved roads (centerline distance)
- Multiple lanes (actual width calculation)

This is NOT the simple proximity-based system I initially described. The EnhancedRoadMatcher is the default, with complex OSM-based logic that accurately determines if threats are truly on the same road and in which direction relative to the actual road path, not just GPS bearing.