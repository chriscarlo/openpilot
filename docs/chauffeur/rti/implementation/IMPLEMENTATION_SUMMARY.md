# RTI Dashboard Speed Limit Integration - Implementation Summary

## What Was Implemented

### The Complete Fix: Three Components

1. **Added Dashboard Speed Limit Subscription**
   - RTI now subscribes to `carStateSP` for dashboard speed limits
   - This provides access to the car's traffic sign recognition data
   - Works for both CANFD (ECAN bus) and regular CAN (CAM bus) vehicles

2. **Implemented Conservative Combination Logic**
   - **KEY DIFFERENCE**: RTI uses MIN (lower value) vs SLC's MAX (higher value)
   - When both sources disagree, RTI chooses the MORE CONSERVATIVE limit
   - This aligns with RTI's safety-first threat detection mission

3. **Maintained Robust Fallback**
   - Single source fallback when only one has data
   - Returns 0.0 when no data available (no dangerous hardcoded values)

## Files Modified

### Core Implementation
- **`sunnypilot/rtid/rtid.py`**
  - Line 38: Added `'carStateSP'` to SubMaster subscription
  - Lines 155-205: Completely rewrote `_get_current_speed_limit()` method
  - Added conservative MIN logic with detailed logging

### Documentation Created
- `/docs/chauffeur/rti/implementation/conservative_speed_limit_combination.md`
- `/docs/chauffeur/rti/analysis/dashboard_speed_limit_ingestion_research.md`
- `/docs/chauffeur/rti/analysis/complete_speed_limit_architecture_analysis.md`
- `/docs/chauffeur/rti/analysis/speed_limit_data_flow_comparison.md`

### Test Files
- `test_rti_conservative_speed_limits.py` - Verifies MIN logic

## The Conservative Approach Explained

### Why MIN Instead of MAX?

**Speed Limit Controller (Driver Comfort)**
```python
# SLC: Don't force below either limit
combined = MAX(dashboard, map)  # Use higher value
```

**RTI (Threat Safety)**
```python
# RTI: Maximum caution near threats
combined = MIN(dashboard, map)  # Use lower value
```

### Real-World Impact

| Scenario | Dashboard | Map | SLC Uses | RTI Uses | Benefit |
|----------|-----------|-----|----------|----------|---------|
| Construction | 45 mph | 65 mph | 65 mph | **45 mph** | Respects temporary limit |
| School Zone | 25 mph | 35 mph | 35 mph | **25 mph** | Uses active restriction |
| Exit Ramp | 35 mph | 65 mph | 65 mph | **35 mph** | Follows ramp speed |

## How It Works Now

### Data Flow
```
1. Car's TSR camera detects speed sign
2. CAN message 0x1FA published
   - CANFD: On ECAN bus
   - Regular CAN: On CAM bus
3. Car interface parses and publishes carStateSP
4. RTI subscribes to BOTH:
   - carStateSP (dashboard)
   - liveMapDataSP (map)
5. RTI combines using MIN (conservative)
6. Threat detector uses actual speed limits
```

### The Complete Fix Addresses All Three Layers

1. ✅ **Dashboard Data**: Now subscribed to carStateSP
2. ✅ **Map Data**: Already subscribed to liveMapDataSP
3. ✅ **No Reductions**: "Posted" mode uses actual limits

## Testing Results

```bash
✓ Construction zone - uses lower dashboard: 20.0 m/s
✓ School zone - uses lower map: 16.0 m/s
✓ Same values - uses either: 29.0 m/s
✓ Dashboard unavailable - uses map: 29.0 m/s
✓ Map unavailable - uses dashboard: 25.0 m/s
✓ Both unavailable - returns 0: 0.0 m/s
✓ RTI is 20 mph more conservative than SLC
```

## Why This Matters

### Before This Fix
- RTI was blind to dashboard speed limits
- Fell back to hardcoded 56 mph when map unavailable
- Created dangerous situations (42 mph on 65 mph highways)

### After This Fix
- RTI sees BOTH dashboard and map speed limits
- Uses the MORE CONSERVATIVE value for safety
- No dangerous hardcoded fallbacks
- Complete awareness in "posted speed limit" mode

## Key Innovation: Conservative by Design

While the Speed Limit Controller optimizes for driver comfort (using MAX), RTI now optimizes for safety (using MIN). This makes sense because:

1. **RTI's Mission**: Warn about threats where ANY speed limit matters
2. **Construction Zones**: Temporary limits are enforced
3. **School Zones**: Active limits take precedence
4. **Exit Ramps**: Lower speeds are critical

## Summary

RTI now has complete speed limit awareness from both map and dashboard sources, combining them conservatively (MIN) for maximum safety in threat detection scenarios. This fixes the dangerous blind spot where RTI couldn't see dashboard-detected speed limits and inappropriately slowed vehicles when map data was unavailable.

The implementation is complete, tested, and ready for deployment.