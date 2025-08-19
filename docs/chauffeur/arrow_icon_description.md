# RTI Arrow Icon Visual Description

## Arrow Design
The arrow icon is a **48x48 pixel** programmatic shape (not an SVG file) created using QPainterPath. Here's what it looks like:

```
         ▲
        ╱│╲
       ╱ │ ╲
      ╱  │  ╲
     ╱   │   ╲
         │
         │
         │
```

## Size and Proportions
- **Arrow Size:** 48x48 pixels
- **Arrow Length:** 80% of icon size (38 pixels)
- **Arrow Width:** 50% of icon size (24 pixels)
- **Arrowhead:** Takes up top 1/3 of length
- **Shaft:** Bottom 2/3 of length, 1/3 the width of arrowhead

## Visual Characteristics
1. **Shape:** Classic arrow pointing upward by default
2. **Fill:** Solid color (tinted based on threat distance)
3. **Border:** 2px black border with 100/255 opacity for contrast
4. **Rotation:** Smooth rotation from 0-360° based on threat bearing

## Color Coding by Distance
- **Red** (255, 0, 0): Critical threats < 200m
- **Orange** (255, 165, 0): Near threats 200-500m  
- **Yellow** (255, 255, 0): Normal threats 500-1000m
- **Gray** (150, 150, 150): Far threats > 1000m

## Size Comparison with Text
In the RTI widget, the arrow appears alongside:
- **Threat Text:** 42px font (e.g., "POLICE")
- **Distance Text:** 45px bold font (e.g., "0.5mi")

The 48px arrow is appropriately sized - slightly larger than the text height to ensure visibility while not overwhelming the display.

## Positioning in HUD
The arrow is positioned:
- **X:** 180px from left edge of RTI widget
- **Y:** 125px from top edge of RTI widget
- Appears to the LEFT of the threat text
- Text is shifted 50px to the right when arrow is present

## Example Layout
```
┌─────────────────────────────┐
│                             │
│     [ICON]                  │
│                             │
│    ↗  POLICE                │  ← Arrow (48px) + Text (42px)
│                             │
│      0.5mi                  │  ← Distance (45px)
│                             │
└─────────────────────────────┘
```

## Rotation Examples
- **0°** (↑): Threat directly ahead
- **45°** (↗): Threat ahead-right  
- **90°** (→): Threat to the right
- **135°** (↘): Threat behind-right
- **180°** (↓): Threat directly behind
- **225°** (↙): Threat behind-left
- **270°** (←): Threat to the left
- **315°** (↖): Threat ahead-left

## Performance Notes
- Arrow pixmap is created once at startup and cached
- Only the rotation transform is applied per frame
- Tinting is done using QPainter composition modes
- Total memory footprint: ~9KB (48x48x4 bytes)

## Sizing Assessment
The 48px arrow size is **well-proportioned** for the RTI widget:
- Large enough to be clearly visible while driving
- Not so large that it dominates the threat text
- Provides good visual balance with the 42px threat text
- Clear directionality even at arm's length viewing distance
- Maintains crisp appearance when rotated to any angle

The arrow successfully enhances the RTI alert without cluttering the interface.