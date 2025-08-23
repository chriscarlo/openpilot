# Speed Limit Data Flow: SLC vs RTI Comparison

## Current Architecture Comparison

### Speed Limit Controller (Complete Implementation)
```
┌─────────────────────────────────────────────────────────────┐
│                    SPEED LIMIT CONTROLLER                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CAN BUS ──┐                                               │
│            ├─→ Car Interface ─→ carStateSP ──┐             │
│            │                                  ↓             │
│  CANFD ────┘                            SpeedLimitResolver  │
│                                              ↑              │
│  Map API ────→ Map Service ──→ liveMapDataSP ┘             │
│                                                             │
│                          ↓                                  │
│                                                             │
│              Combined Speed Limit Output                    │
│              (MAX of both sources)                          │
└─────────────────────────────────────────────────────────────┘
```

### RTI (Incomplete Implementation)
```
┌─────────────────────────────────────────────────────────────┐
│                            RTI                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CAN BUS ──┐                                               │
│            ├─→ Car Interface ─→ carStateSP ─→ ❌ (ignored)  │
│            │                                                │
│  CANFD ────┘                                                │
│                                                             │
│  Map API ────→ Map Service ──→ liveMapDataSP ─→ ✅ RTI     │
│                                                             │
│                          ↓                                  │
│                                                             │
│              Partial Speed Limit Output                     │
│              (Map only, falls back to 56 mph)              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow by Source

### Dashboard Speed Limit Path
```
Traffic Sign
     ↓
Car's Camera
     ↓
Image Recognition
     ↓
CAN Message 0x1FA
     ↓
┌──────────────────┐
│ CANFD: ECAN Bus  │
│ CAN: CAM Bus     │
└──────────────────┘
     ↓
Car Interface Parser
     ↓
carStateSP.speedLimit
     ↓
┌──────────────────┐
│ SLC: ✅ Reads it │
│ RTI: ❌ Ignores  │
└──────────────────┘
```

### Map Speed Limit Path
```
OpenStreetMap Data
     ↓
Map Service/API
     ↓
liveMapDataSP.speedLimit
     ↓
┌──────────────────┐
│ SLC: ✅ Reads it │
│ RTI: ✅ Reads it │
└──────────────────┘
```

## Decision Logic Comparison

### SLC Combined Mode Logic
```python
if dashboard > 0 and map > 0:
    if dashboard == map:
        use map  # More stable
    else:
        use MAX(dashboard, map)  # Safety first
elif dashboard > 0:
    use dashboard
elif map > 0:
    use map
else:
    no limit available
```

### RTI Current Logic
```python
if map > 0:
    use map
else:
    use hardcoded 56 mph  # Dangerous fallback!
# Never even checks dashboard
```

## Real-World Scenario Matrix

| Scenario | Dashboard | Map | SLC Result | RTI Result | Safety Impact |
|----------|-----------|-----|------------|------------|---------------|
| Highway normal | 65 mph | 65 mph | 65 mph ✅ | 65 mph ✅ | Safe |
| Construction zone | 45 mph | 65 mph | 65 mph ✅ | 65 mph ⚠️ | Misses temporary limit |
| School zone active | 25 mph | 35 mph | 35 mph ✅ | 35 mph ⚠️ | Misses active zone |
| Rural road | 55 mph | None | 55 mph ✅ | 56→42 mph ❌ | Dangerous slowdown |
| Unmapped area | None | None | None ✅ | 56→42 mph ❌ | Inappropriate fallback |
| New road | 70 mph | None | 70 mph ✅ | 56→42 mph ❌ | Major hazard |

## Critical Findings

### 1. RTI's Blind Spots
- **100% miss rate** on dashboard-only speed limits
- **100% miss rate** on temporary/construction limits
- **Dangerous fallback** when no map data exists

### 2. SLC's Robustness
- **Dual source** resilience
- **Intelligent combination** for safety
- **No hardcoded fallbacks**

### 3. The Bus Complexity Is Hidden
- carStateSP **abstracts** the CAN/CANFD difference
- RTI wouldn't need to know about bus architecture
- Just subscribe and read - the interface handles it

## The Gap Summary

```
What RTI Has:
├── ✅ Map speed limits (liveMapDataSP)
├── ✅ Waze alerts (often no speed data)
└── ❌ Dashboard speed limits (carStateSP)

What RTI Needs:
├── ✅ Map speed limits (already has)
├── ✅ Dashboard speed limits (MISSING!)
├── ✅ Intelligent combination logic (MISSING!)
└── ✅ Remove hardcoded fallbacks (partially done)
```

## Implementation Complexity

### To Add Dashboard Support to RTI:
1. **Add subscription** (1 line):
   ```python
   'carStateSP',  # Add to SubMaster list
   ```

2. **Read the data** (4 lines):
   ```python
   def _get_dashboard_speed_limit(self) -> float:
       if self.sm['carStateSP'].speedLimit > 0:
           return self.sm['carStateSP'].speedLimit
       return 0.0
   ```

3. **Combine intelligently** (5 lines):
   ```python
   map_limit = self._get_map_speed_limit()
   dash_limit = self._get_dashboard_speed_limit()
   if map_limit > 0 and dash_limit > 0:
       return max(map_limit, dash_limit)
   return map_limit or dash_limit or 0.0
   ```

Total: ~10 lines of code to fix a critical safety gap.

## Why This Matters

RTI in "posted speed limit" mode promises to respect posted limits, but it:
- Can't see dashboard-detected signs
- Falls back to dangerous hardcoded values
- Misses exactly the limits that matter most (temporary, construction, school zones)

The Speed Limit Controller shows how it should be done. RTI needs to follow that pattern.