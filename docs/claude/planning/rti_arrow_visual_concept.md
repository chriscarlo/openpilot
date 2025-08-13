# RTI Directional Arrow System - Visual Concept

## Current vs. Proposed Design

### Current RTI Widget (200x260px)
```
┌──────────────────────┐
│       RTI            │
│    MONITORING        │
│                      │
│     [ICON] POLICE    │
│                      │
│      1.2km           │
│                      │  
│      ↓ 85            │
└──────────────────────┘
```

### Proposed RTI Widget (400x267px)
```
┌──────────────────────────────────────────┐
│                  RTI                     │
│                                          │
│            ↗  police                    │
│                                          │
│           1.2km                          │
│                                          │  
│           ↓ 85                           │
└──────────────────────────────────────────┘
```

## Arrow Direction Examples

### Threat Ahead (0°)
```
┌──────────────────────────────────────────┐
│                  RTI                     │
│                                          │
│            ↑  camera                    │
│                                          │
│           850m                           │
└──────────────────────────────────────────┘
```

### Threat to Right (90°)
```
┌──────────────────────────────────────────┐
│                  RTI                     │
│                                          │
│            →  construction              │
│                                          │
│           450m                           │
└──────────────────────────────────────────┘
```

### Threat Behind (180°)
```
┌──────────────────────────────────────────┐
│                  RTI                     │
│                                          │
│            ↓  accident                  │
│                                          │
│           2.1km                          │
└──────────────────────────────────────────┘
```

### Threat to Left (270°)
```
┌──────────────────────────────────────────┐
│                  RTI                     │
│                                          │
│            ←  hazard                    │
│                                          │
│           325m                           │
└──────────────────────────────────────────┘
```

### Multiple Threats (showing closest)
```
┌──────────────────────────────────────────┐
│                  RTI                     │
│                                          │
│            ↗  police                    │
│                                          │
│           650m                           │
│                                          │
│           ↓ 75                           │
└──────────────────────────────────────────┘
```

## Implementation Approach

### Arrow Rendering in Qt
```cpp
void drawDirectionalArrow(QPainter &p, const QRect &icon_rect, float angle_degrees) {
    p.save();
    
    // Move to center of icon area
    QPoint center = icon_rect.center();
    p.translate(center);
    
    // Rotate by calculated angle (0° = up/forward)
    p.rotate(angle_degrees);
    
    // Draw arrow pointing upward (will be rotated to correct direction)
    QPolygon arrow;
    arrow << QPoint(0, -20)     // tip
          << QPoint(-10, 10)    // left base
          << QPoint(-5, 10)     // left inner
          << QPoint(-5, 20)     // left tail
          << QPoint(5, 20)      // right tail  
          << QPoint(5, 10)      // right inner
          << QPoint(10, 10);    // right base
    
    p.drawPolygon(arrow);
    p.restore();
}
```

### Real-time Update Logic
```cpp
void HudRendererSP::updateState(const UIState &s) {
    // Get ego position and heading
    if (s.sm->valid("gpsLocationExternal")) {
        ego_lat = s.sm["gpsLocationExternal"].latitude;
        ego_lon = s.sm["gpsLocationExternal"].longitude;
    }
    
    if (s.sm->valid("carState")) {
        ego_heading = s.sm["carState"].bearing; // or similar heading field
    }
    
    // Calculate arrow direction for each threat
    if (s.sm->valid("rtiStateSP") && s.sm->updated("rtiStateSP")) {
        auto threats = rti_state.getThreats();
        if (threats.size() > 0) {
            auto threat = threats[0]; // closest threat
            
            // Calculate directional arrow angle
            rti_arrow_angle = calculateThreatDirection(
                ego_lat, ego_lon, ego_heading,
                threat.latitude, threat.longitude
            );
        }
    }
}
```

## Color Coding System

Arrows inherit the same color system as current threats:
- **Red** (< 100m): Immediate danger
- **Orange** (100-300m): Approaching
- **Yellow** (300-1000m): Advance warning
- **Gray** (> 1000m): Distant

## Text Simplification

Current verbose labels → Concise labels:
- "POLICE" → "police"
- "SPEED_CAMERA" → "camera"  
- "ACCIDENT" → "accident"
- "CONSTRUCTION" → "construction"
- "JAM" → "traffic"
- "HAZARD" → "hazard"
- "ROAD_CLOSED" → "closed"

## Real-world Behavior

1. **Stationary Threats**: Arrow points to fixed location, updates as ego moves
2. **Dynamic Updates**: Smooth rotation as ego changes heading/position  
3. **Multiple Threats**: Shows arrow for closest/highest priority threat
4. **GPS Loss**: Falls back to current direction enum (ahead/behind/left/right)
5. **Performance**: Updates at 10Hz with ego position changes

## Next Steps

1. DONE **Planning Complete**: Comprehensive design and mathematical validation
2. DONE **Test Script**: Coordinate calculation logic verified
3. TODO **Prototype**: Create isolated arrow rendering test
4. TODO **Integration**: Add to existing RTI HUD widget
5. TODO **Validation**: Real-world testing with known threat locations