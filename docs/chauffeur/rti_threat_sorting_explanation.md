# RTI Threat Sorting Explanation

## Sorting Logic

The RTI widget sorts threats by **distance in ASCENDING order** (closest first).

### Implementation
```cpp
std::sort(rti_threats.begin(), rti_threats.end(), 
          [](const RTIThreatInfo& a, const RTIThreatInfo& b) {
            return a.distance < b.distance;  // Ascending order
          });
```

## Display Order

Threats are displayed from **TOP to BOTTOM** in order of urgency:

| Position | Description | Priority |
|----------|-------------|----------|
| **Line 1 (TOP)** | Closest threat | HIGHEST - Most urgent |
| **Line 2** | Second closest | HIGH |
| **Line 3** | Third closest | MEDIUM |
| **Line 4 (BOTTOM)** | Fourth closest | LOW - Least urgent |

## Example

Given these threats:
- CAMERA at 500m
- POLICE at 200m  
- ACCIDENT at 800m
- CONSTRUCTION at 1200m
- HAZARD at 350m

### After Sorting:
1. **POLICE** - 200m (TOP - Most urgent)
2. **HAZARD** - 350m
3. **CAMERA** - 500m
4. **ACCIDENT** - 800m (BOTTOM of displayed threats)
5. ~~CONSTRUCTION - 1200m~~ (Not shown - max 4 threats)

### Visual Display:
```
┌─────────────────────────────────────┐
│ RTI                                 │
│                                     │
│ → POLICE • 200m        [RED/ORANGE] │ ← TOP (Closest)
│ ← HAZARD • 350m        [ORANGE]     │
│ ↑ CAMERA • 500m        [YELLOW]     │
│ ↗ ACCIDENT • 800m      [YELLOW]     │ ← BOTTOM (Furthest shown)
│                                     │
└─────────────────────────────────────┘
```

## Rationale

This ordering ensures:
1. **Most urgent threats get immediate attention** - Closest threats are at eye level
2. **Natural reading order** - Top to bottom matches urgency hierarchy
3. **Quick scanning** - Driver can immediately see the most critical threat
4. **Logical progression** - Distance increases as you read down

## Color Coding Reinforcement

The color coding further emphasizes urgency:
- **RED** (< 200m): Critical - Immediate attention
- **ORANGE** (200-500m): Warning - Approaching soon
- **YELLOW** (500-1000m): Caution - Be aware
- **GRAY** (> 1000m): Information - Future concern

The combination of position (top = urgent) and color (red/orange = urgent) provides redundant visual cues for safety.