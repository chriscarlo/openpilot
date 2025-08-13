# RTI Directional Arrow System - Implementation Plan

## Overview
Replace current emoji-based RTI threat icons with directional arrows that point toward threats in real-time, with simple text labels.

## Current State Analysis
- **Current Icons**: Emoji-style vector graphics (police car, camera, triangle, etc.)
- **Current Text**: Verbose labels ("POLICE", "CAMERA", "ACCIDENT", etc.)
- **Missing**: No directional indication relative to ego vehicle

## Requirements
1. **Arrow Icon**: Points toward threat relative to ego position
2. **Coordinate System**: "Upward" = forward direction for ego vehicle
3. **Real-time Tracking**: Updates as ego moves and orientation changes
4. **Persistent Display**: Not ephemeral, stays visible until threat passed
5. **Simple Text**: Concise labels ("police", "hazard", "construction")
6. **Performance**: Minimal computational overhead for real-time updates

## Data Sources Available

### Ego Vehicle Telemetry
```cpp
// GPS Position (lat/lon)
gpsLocationExternal.latitude, gpsLocationExternal.longitude
gpsLocation.latitude, gpsLocation.longitude

// Vehicle Heading/Orientation
carState.bearing  // or similar heading data
```

### Threat Data
```cpp
// From RTI messages
threat.latitude, threat.longitude    // Absolute coordinates
threat.type                         // Threat classification
threat.distance                     // Already calculated distance
threat.direction                    // Current enum (ahead/behind/left/right)
```

## Mathematical Approach

### 1. Bearing Calculation
```
bearing = atan2(
    sin(lon2 - lon1) * cos(lat2),
    cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
)
```

### 2. Relative Direction
```
relative_bearing = (threat_bearing - ego_heading + 360) % 360
```

### 3. Arrow Rotation
```
arrow_angle = relative_bearing  // 0° = up/forward, 90° = right, etc.
```

## Implementation Architecture

### Phase 1: Coordinate Calculation Module
```cpp
class RTIDirectionCalculator {
    float calculateBearing(double lat1, double lon1, double lat2, double lon2);
    float calculateRelativeDirection(float threat_bearing, float ego_heading);
    void updateThreatDirections(const EgoState& ego, const std::vector<Threat>& threats);
};
```

### Phase 2: Arrow Rendering System
```cpp
void drawRTIThreatArrow(QPainter &p, const QRect &icon_rect, float direction_angle) {
    // Draw arrow pointing at direction_angle
    // 0° = up (forward), 90° = right, 180° = down (behind), 270° = left
}
```

### Phase 3: Text Simplification
Replace current verbose text with concise labels:
- "POLICE" → "police"
- "SPEED_CAMERA" → "camera" 
- "ACCIDENT" → "accident"
- "CONSTRUCTION" → "construction"
- "JAM" → "traffic"
- "HAZARD" → "hazard"

### Phase 4: Real-time Updates
- Calculate directions in `HudRendererSP::updateState()`
- Store calculated angles in state variables
- Update every UI cycle (10Hz) for smooth tracking

## Technical Considerations

### Performance Optimization
- Cache expensive trigonometric calculations
- Only recalculate when ego position changes significantly (>10m)
- Use lookup tables for common angles if needed

### Error Handling
- Fallback to current direction enum if GPS unavailable
- Handle edge cases (threat at same location as ego)
- Graceful degradation if calculation fails

### Visual Design
- Arrow size: Match current icon dimensions (100x80px area)
- Arrow color: Use existing threat color system (red/orange/yellow by distance)
- Arrow style: Simple, bold triangle for visibility
- Text positioning: Below arrow, same as current layout

## Testing Strategy

### Unit Tests
- Bearing calculation accuracy with known coordinates
- Edge cases (crossing 0°/360°, poles, same location)
- Performance benchmarks for real-time calculation

### Integration Tests  
- End-to-end with live GPS and RTI data
- Visual verification of arrow directions
- Performance impact on UI rendering

## Rollout Plan

1. **Test Script**: Standalone coordinate calculation verification
2. **Prototype**: Isolated arrow rendering in test environment  
3. **Integration**: Add to existing RTI HUD widget
4. **Validation**: Real-world testing with known threat locations
5. **Deployment**: Replace current icon system

## Risk Mitigation

### GPS Accuracy
- Use both external and internal GPS sources
- Implement smoothing for noisy GPS data
- Fallback to static direction enum if GPS poor

### Performance Impact
- Profile calculation overhead
- Optimize for target 10Hz update rate
- Consider reducing update frequency if needed

### User Experience
- Ensure arrows are visually distinct and clear
- Test readability in various lighting conditions
- Maintain color coding for distance-based urgency

## Future Enhancements
- 3D arrow rendering for enhanced depth perception
- Animation/pulsing for high-priority threats
- Integration with turn-by-turn navigation
- Predictive direction based on ego trajectory