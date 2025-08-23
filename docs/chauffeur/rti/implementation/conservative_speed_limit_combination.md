# RTI Conservative Speed Limit Combination Implementation

## Overview
RTI now reads speed limits from BOTH dashboard (car's traffic sign recognition) and map sources, but uses a CONSERVATIVE combination approach that differs from the Speed Limit Controller.

## Key Difference: MIN vs MAX

### Speed Limit Controller (SLC) Approach
- **Combination Logic**: MAX(dashboard, map)
- **Philosophy**: Don't force driver below either posted limit
- **Example**: Construction zone 45 mph, map shows 65 mph → Uses 65 mph
- **Result**: Respects the higher limit for driver comfort

### RTI Conservative Approach
- **Combination Logic**: MIN(dashboard, map)
- **Philosophy**: Maximum safety for threat detection
- **Example**: Construction zone 45 mph, map shows 65 mph → Uses 45 mph
- **Result**: More conservative speed recommendations near threats

## Implementation Details

### 1. Dual Source Subscription
```python
self.sm = messaging.SubMaster([
    'gpsLocationExternal',
    'gpsLocation',
    'carState',
    'liveMapDataSP',  # Map-based speed limit
    'carStateSP'      # Dashboard-based speed limit (TSR camera)
], ignore_avg_freq=True)
```

### 2. Conservative Combination Logic
```python
def _get_current_speed_limit(self) -> float:
    """Get current posted speed limit from both sources.
    Uses CONSERVATIVE combination: MIN of both values."""
    
    map_limit = self._get_map_limit()
    dashboard_limit = self._get_dashboard_limit()
    
    if map_limit > 0 and dashboard_limit > 0:
        # CONSERVATIVE: Use LOWER value
        return min(map_limit, dashboard_limit)
    elif dashboard_limit > 0:
        return dashboard_limit
    elif map_limit > 0:
        return map_limit
    else:
        return 0.0
```

## Why Conservative for RTI?

### RTI's Mission: Threat Detection & Safety
1. **Primary Goal**: Warn about threats ahead
2. **Safety Priority**: Better to be too cautious than miss a hazard
3. **Speed Recommendation**: Should align with most restrictive limit

### Scenario Analysis

| Scenario | Dashboard | Map | SLC (MAX) | RTI (MIN) | Rationale |
|----------|-----------|-----|-----------|-----------|-----------|
| Construction | 45 mph | 65 mph | 65 mph | **45 mph** | RTI respects temporary restriction |
| School Zone | 25 mph | 35 mph | 35 mph | **25 mph** | RTI uses active limit |
| Rural Road | 55 mph | None | 55 mph | 55 mph | Same (single source) |
| Highway | 65 mph | 65 mph | 65 mph | 65 mph | Same (sources agree) |
| Exit Ramp | 35 mph | 65 mph | 65 mph | **35 mph** | RTI respects ramp limit |

## Advantages of Conservative Approach

### 1. Enhanced Safety
- Never recommends speed above the most restrictive limit
- Particularly important near police/speed trap threats
- Aligns with defensive driving principles

### 2. Construction Zone Awareness
- Dashboard cameras detect temporary signs
- Map data shows permanent limits
- RTI uses the temporary (lower) limit

### 3. School Zone Protection
- Active electronic signs detected by dashboard
- Static limits in map data
- RTI respects the active (lower) limit

## Data Flow

```
Traffic Sign → Car Camera → Dashboard Speed Limit ↘
                                                    → MIN → RTI Speed Recommendation
Map Data → Navigation System → Map Speed Limit ↗
```

## Logging for Debugging

The implementation includes detailed logging:
```
RTI: Map speed limit: 29.1 m/s (65 mph)
RTI: Dashboard speed limit: 20.1 m/s (45 mph)
RTI: Using dashboard speed limit (conservative): 20.1 m/s
```

## Impact on "Posted Speed Limit" Mode

When RTI is in "posted speed limit" mode:
- **Before**: Only had map data, fell back to hardcoded 56 mph
- **Now**: Has both sources, uses the MORE CONSERVATIVE one
- **Result**: Never recommends speeds above actual posted limits

## Testing Considerations

### Unit Testing
- Test MIN logic with various combinations
- Verify single-source fallback
- Confirm no-data handling

### Integration Testing
- Verify carStateSP subscription works
- Test on CANFD vehicles (ECAN bus)
- Test on regular CAN vehicles (CAM bus)
- Confirm conservative selection in real scenarios

## Comparison Summary

| Aspect | SLC | RTI |
|--------|-----|-----|
| Sources | Map + Dashboard | Map + Dashboard |
| Combination | MAX (higher) | MIN (lower) |
| Philosophy | Driver comfort | Maximum safety |
| Construction | Uses permanent | Uses temporary |
| Missing Data | Falls back | Falls back |
| Primary Goal | Cruise control | Threat warnings |

## Conclusion

RTI's conservative approach to speed limit combination provides maximum safety for threat detection scenarios. By using the MINIMUM of available speed limits, RTI ensures it never recommends speeds above the most restrictive posted limit, which is critical when approaching police, speed traps, or other threats where exceeding ANY posted limit could be problematic.

This implementation fixes RTI's previous blind spot (no dashboard data) while adding an extra layer of safety through conservative combination logic.