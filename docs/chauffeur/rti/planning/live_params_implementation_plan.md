# RTI Live Parameters Implementation Plan

## Current State Analysis

### Identified Parameters (7 total)

1. **RTIEnabled** - Enable/disable RTI functionality
   - Location: rtid.py line 90
   - Current: ✅ LIVE (checked every loop iteration)

2. **RTIThreatFilter** - Filter threats by type (0=All, 1=Police, 2=Cameras, 3=Hazards)
   - Location: threat_detector.py line 457
   - Current: ❌ NOT LIVE (loaded in __init__)

3. **RTIDetectionRadius** - Max radius for threat detection (meters)
   - Location: threat_detector.py lines 299, 446 & rtid.py line 262
   - Current: ⚠️ PARTIALLY LIVE (API fetch only, not threat processing)

4. **RTIForwardSlowdownRange** - Distance to start slowing for threats ahead (meters)
   - Location: threat_detector.py line 310
   - Current: ❌ NOT LIVE (loaded in __init__)

5. **RTIResumeSpeedDistance** - Distance after threat to resume speed (meters)
   - Location: threat_detector.py line 321
   - Current: ❌ NOT LIVE (loaded in __init__)

6. **RTISpeedReductionMode** - Speed reduction strategy ("posted" or "custom")
   - Location: threat_detector.py line 331
   - Current: ❌ NOT LIVE (loaded in __init__)

7. **RTISpeedReduction** - Custom speed reduction amount (km/h)
   - Location: threat_detector.py line 342
   - Current: ❌ NOT LIVE (loaded in __init__)

### Current Loading Patterns

**Pattern 1: Init-time Loading (NOT LIVE)**
- threat_detector.py loads 6 params in __init__
- Values cached as instance variables
- Never re-read during operation

**Pattern 2: Per-cycle Loading (LIVE)**
- rtid.py checks RTIEnabled every loop
- Works but inefficient for multiple params

**Pattern 3: On-demand Loading (PARTIALLY LIVE)**
- RTIDetectionRadius checked during API fetch
- Only updates when new data fetched (30-60s intervals)

## Proposed Architecture

### Design Principles
1. **Minimal overhead** - Read params once per processing cycle, not per function
2. **Consistent timing** - All params refresh at the same rate
3. **Simple implementation** - Single pattern for all params
4. **Performance aware** - 50Hz loop must remain responsive

### Solution: Cycle-based Parameter Cache

```python
class LiveParams:
    """Manages live-reloadable parameters with per-cycle caching"""
    
    def __init__(self):
        self.params = Params()
        self._cache = {}
        self._cache_timestamp = 0
        self.CACHE_DURATION_NS = 100_000_000  # 100ms cache (5x per second refresh)
    
    def refresh_if_needed(self):
        """Refresh cache if expired"""
        now = time.monotonic_ns()
        if now - self._cache_timestamp > self.CACHE_DURATION_NS:
            self._load_all_params()
            self._cache_timestamp = now
    
    def _load_all_params(self):
        """Load all RTI params into cache"""
        # Single batch read of all params
        self._cache['enabled'] = self.params.get_bool("RTIEnabled")
        self._cache['threat_filter'] = self._parse_int("RTIThreatFilter", 0)
        self._cache['detection_radius'] = self._parse_float("RTIDetectionRadius", 4828)
        self._cache['forward_range'] = self._parse_float("RTIForwardSlowdownRange", 1207)
        self._cache['resume_distance'] = self._parse_float("RTIResumeSpeedDistance", 1207)
        self._cache['speed_mode'] = self._parse_string("RTISpeedReductionMode", "posted")
        self._cache['speed_reduction'] = self._parse_float("RTISpeedReduction", 16) / 3.6  # km/h to m/s
    
    @property
    def enabled(self): return self._cache.get('enabled', False)
    
    @property
    def threat_filter(self): return self._cache.get('threat_filter', 0)
    
    # ... other property accessors
```

## Implementation Plan

### Phase 1: Create LiveParams Class
1. Create new file: `openpilot/sunnypilot/rtid/live_params.py`
2. Implement LiveParams class with:
   - Centralized param loading
   - Time-based cache invalidation
   - Type-safe parsing methods
   - Property-based access

### Phase 2: Integrate into RTIDaemon
1. Replace `self.params = Params()` with `self.live_params = LiveParams()`
2. In main loop before `_process_cycle_async()`:
   ```python
   self.live_params.refresh_if_needed()
   ```
3. Pass live_params to ThreatDetector.process_threats()

### Phase 3: Refactor ThreatDetector
1. Remove all param loading from `__init__`
2. Accept live_params in `process_threats()` method
3. Access params via: `live_params.threat_filter` instead of `self.threat_filter`

### Phase 4: Refactor SpeedRecommendationEngine
1. Remove param loading from `__init__`
2. Accept live_params in `calculate_recommendation()`
3. Use live values for all thresholds

### Phase 5: Testing
1. Create test script to verify live updates:
   ```python
   # test_live_params.py
   # 1. Start rtid daemon
   # 2. Change each param via Params
   # 3. Verify behavior changes within 200ms
   # 4. No daemon restart needed
   ```

2. Integration tests:
   - Change threat filter while running → threats update immediately
   - Change detection radius → new radius applied next cycle
   - Change speed settings → recommendations update live

### Phase 6: Performance Validation
1. Measure impact on 50Hz loop timing
2. Expected overhead: <0.1ms per cycle (cache hit)
3. Cache refresh: ~1ms every 100ms (5Hz)

## Benefits

1. **User Experience**
   - Settings take effect immediately
   - No need to restart car/openpilot
   - Easier testing and tuning

2. **Development**
   - Consistent param handling pattern
   - Easier to add new params
   - Centralized validation logic

3. **Performance**
   - Minimal overhead (cached reads)
   - Batch param loading (single IPC)
   - Predictable timing

## Migration Notes

### Breaking Changes
- None for users (transparent upgrade)

### Code Changes Required
- ThreatDetector: ~50 lines modified
- SpeedRecommendationEngine: ~30 lines modified  
- RTIDaemon: ~10 lines modified
- New LiveParams class: ~100 lines

### Backwards Compatibility
- All existing param values preserved
- Default values unchanged
- No UI changes needed

## Timeline Estimate
- Phase 1-2: 1 hour (LiveParams + daemon integration)
- Phase 3-4: 2 hours (refactor threat processing)
- Phase 5: 1 hour (testing)
- Phase 6: 30 minutes (performance validation)

Total: ~4.5 hours of implementation work

## Risk Assessment

**Low Risk:**
- Well-defined scope
- No safety-critical changes (only param reading)
- Easy rollback if issues

**Mitigations:**
- Extensive testing before deployment
- Gradual rollout (start with non-critical params)
- Performance monitoring in production