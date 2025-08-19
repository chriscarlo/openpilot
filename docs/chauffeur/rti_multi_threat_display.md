# RTI Multi-Threat Display with Directional Arrows

## Overview
The RTI widget has been redesigned to display up to 4 threats simultaneously, each with its own directional arrow showing where the threat is relative to the vehicle.

## Widget Layout

### Size and Position
- **Width:** 580px (expanded from 525px)
- **Height:** 280px (optimized for 4 lines)
- **Position:** Bottom-left corner, 15px margins

### Display Format
Each threat appears on a single line with:
```
[Arrow] THREAT_TYPE • Distance
```

Example display:
```
┌─────────────────────────────────────┐
│ RTI                                 │
│                                     │
│ ↗ POLICE • 0.3mi                    │
│ → CAMERA • 0.5mi                    │
│ ↑ ACCIDENT • 0.8mi                  │
│ ↘ CONSTRUCTION • 1.2mi              │
│                                     │
│ Target: 35 mph                      │
└─────────────────────────────────────┘
```

## Arrow Specifications

### Compact Arrow (Multi-threat)
- **Size:** 32x32 pixels (reduced from 48x48)
- **Position:** Left-aligned on each threat line
- **Rotation:** Real-time based on GPS bearing
- **Color:** Matches threat distance color

### Standard Arrow (Single threat - legacy)
- **Size:** 48x48 pixels
- **Used when:** Only one threat is present (backward compatibility)

## Color Coding
Each threat line is color-coded by distance:
- **Red:** < 200m - Critical proximity
- **Orange:** 200-500m - Near threat
- **Yellow:** 500-1000m - Normal range
- **Gray:** > 1000m - Far threat

## Features

### Multi-Threat Support
- Displays up to 4 threats simultaneously
- Automatically sorted by distance (closest first)
- Each threat has independent arrow direction
- Compact single-line format for efficiency

### GPS Integration
- Arrows only appear when GPS signal is available
- Updates at 1Hz (GPS update rate)
- Gracefully degrades to text-only without GPS

### Speed Recommendation
- Shows at bottom when RTI is actively controlling
- Format: "Target: XX mph/km/h"
- Only appears when speed adjustment is needed

## Technical Implementation

### Data Flow
1. `rtiStateSP` message provides array of threats (up to 5)
2. System processes first 4 threats
3. Sorts by distance for priority display
4. Calculates relative bearing for each threat
5. Renders compact view with individual arrows

### Performance Optimizations
- Arrow pixmaps cached at two sizes (32px and 48px)
- Bearing calculations only on GPS updates (1Hz)
- Single QTransform per arrow for rotation
- Efficient single-line layout reduces render overhead

### Key Methods
- `updateRTIThreats()` - Processes threat array from messages
- `drawRTIThreatIndicatorMulti()` - Renders multi-threat widget
- `drawRTIArrowCompact()` - Draws 32px arrows for each threat
- `formatDistance()` - Formats distance with proper units
- `getRTIThreatTextShort()` - Returns compact threat labels

## Backward Compatibility
The system maintains legacy single-threat variables for compatibility:
- First (closest) threat populates legacy variables
- Original single-threat view still available
- Seamless transition between single/multi display

## Visual Examples

### No Threats
```
┌─────────────────────────────────────┐
│                                     │
│                RTI                  │
│                                     │
└─────────────────────────────────────┘
```

### Single Threat (uses larger format)
```
┌─────────────────────────────────────┐
│         [ICON]                      │
│                                     │
│      ↗ POLICE                       │
│                                     │
│        0.5mi                        │
└─────────────────────────────────────┘
```

### Multiple Threats (compact format)
```
┌─────────────────────────────────────┐
│ RTI                                 │
│                                     │
│ ↑ POLICE • 0.2mi                    │
│ ↗ CAMERA • 0.4mi                    │
│ → HAZARD • 0.7mi                    │
│                                     │
│ Target: 45 mph                      │
└─────────────────────────────────────┘
```

## Benefits
1. **Enhanced Awareness** - See multiple threats at once
2. **Directional Context** - Know where each threat is located
3. **Priority Display** - Closest threats shown first
4. **Compact Design** - Efficient use of screen space
5. **Clear Information** - Single-line format is easy to scan

The multi-threat display significantly improves situational awareness by providing drivers with a comprehensive view of all nearby traffic threats and their relative positions.