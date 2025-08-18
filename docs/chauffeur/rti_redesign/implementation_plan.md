# RTI Menu System Complete Redesign Plan

## Current State Analysis

### Existing UI-to-Logic Connections Inventory

#### Parameters Used
- `RTIEnabled` - Main toggle on/off (bool)
- `RTIDataSource` - API source selector (int: 0=Disabled, 1=Waze, etc)
- `RTIThreatFilter` - Threat type filter (int: 0=All, 1=Police, 2=Speed, 3=Hazards, 4=Custom)
- `RTIAggressiveness` - Response style (int: 0=Conservative, 1=Balanced, 2=Aggressive)
- `RTIMinDistance` - Minimum detection distance in meters (int)
- `RTIMaxDistance` - Maximum detection distance in meters (int)
- `RTISpeedReduction` - Speed reduction amount in km/h (int)
- `RTIHUDEnabled` - Show HUD display (bool)
- `RTIAudioAlerts` - Play audio alerts (bool)
- `RTIManualApiKey` - Stores the Waze API key from RapidAPI
- `RTIManualApiEndpoint` - API endpoint URL
- `RTIManualApiFormat` - API format identifier

#### API Key Storage
- Primary location: `/persist/rapidapi_key`
- Alternative paths checked:
  - `/data/persist/rapidapi_key`
  - `/data/openpilot/persist/rapidapi_key`
  - `/data/openpilot/rapidapi_key`
  - Environment variable: `RAPIDAPI_KEY`

#### File Structure
- Main control: `rti_control.cc/h` - ExpandableToggleRow widget
- Settings panel: `rti_settings_panel.cc/h` - Main settings interface
- Advanced panel: `rti_advanced_panel.cc/h` - Advanced configuration options
- Integration: `longitudinal_panel.cc` - Parent container

## New Design Requirements

### Core Principles
1. **Imperial Units Only** - All distances and speeds in miles/mph
2. **No Horizontal Scrolling** - Max width constraint at 1300px
3. **Reuse Live Steering Ratio Widget Pattern** - Clean, modern interface with +/- buttons
4. **Waze API Only** - Remove all other API source options
5. **Simplified Settings** - Remove "Response Style" completely

### Widget Specifications

#### 1. Main RTI Toggle
- Simple on/off toggle with title and description
- Title: "Real-time Traffic Intelligence"
- Description: "Monitor traffic ahead and adjust speed automatically"

#### 2. Threat Filter Dropdown
- Keep existing options:
  - All Threats
  - Police Only
  - Speed Cameras Only
  - Hazards Only
  - Custom Filter

#### 3. Detection & Response Settings (Live Steering Ratio Style)

**Detection Radius**
- **Purpose:** Display threats within this radius for situational awareness (360° coverage)
- **Range:** 0.25 - 3.0 miles
- **Increment:** 0.25 miles
- **Default:** 2.0 miles
- **Parameter:** RTIDetectionRadius (stored in meters)
- **Layout:** Minus button | Value Display | Plus button | Reset button

**Forward Slowdown Distance**  
- **Purpose:** Begin slowing when approaching a threat ahead on your route at this distance
- **Range:** 0.0 - 2.0 miles
- **Increment:** 0.25 miles
- **Default:** 0.75 miles
- **Parameter:** RTIForwardSlowdownRange (stored in meters)
- **Layout:** Minus button | Value Display | Plus button | Reset button

**Resume Speed Distance**
- **Purpose:** Resume normal cruise speed after passing a threat by this distance
- **Range:** 0.0 - 2.0 miles
- **Increment:** 0.25 miles
- **Default:** 0.5 miles
- **Parameter:** RTIResumeSpeedDistance (stored in meters)
- **Layout:** Minus button | Value Display | Plus button | Reset button

#### 4. Speed Reduction Selector
- **Options:** "Posted Speed Limit" | "Custom"
- **Custom Widget:** Hidden by default, revealed on "Custom" selection
- **Custom Range:** 0 - 30 mph reduction
- **Increment:** 1 mph
- **Layout:** Same as Detection Range (minus/plus/reset)

#### 5. Alerts & Display Section
- Keep existing toggles but update styling to match
- HUD Display toggle
- Audio Alerts toggle

## Implementation Steps

### Phase 1: File Cleanup
1. Backup existing files (for reference)
2. Wipe clean:
   - `rti_control.cc/h`
   - `rti_settings_panel.cc/h`
   - `rti_advanced_panel.cc/h`
3. Update `longitudinal_panel.cc` integration

### Phase 2: Core Widget Development
1. Create new `RTIToggle` class extending `AbstractControl`
2. Implement `RTIRangeControl` widget (based on LiveSteerRatioControl pattern)
3. Create `RTISpeedReductionControl` with conditional custom widget

### Phase 3: Main Settings Panel
1. Implement single-panel design (no advanced panel needed)
2. Enforce max width of 1300px
3. Apply consistent styling from Steering menu

### Phase 4: Unit Conversion Infrastructure
1. Create conversion utilities:
   - `milesToMeters(miles)` - returns meters
   - `metersToMiles(meters)` - returns miles
   - `mphToKmh(mph)` - returns km/h
   - `kmhToMph(kmh)` - returns mph
2. Apply conversions at parameter save/load points

### Phase 5: Integration & Testing
1. Connect all widgets to params
2. Verify Waze API key loading
3. Test all conversions
4. Validate no horizontal scrolling

## Style Guide

### Colors
- Background: `#292929` (frames)
- Secondary Background: `#393939` (buttons, inputs)
- Text Primary: `#E4E4E4`
- Text Secondary: `#999999`
- Accent: `#4a90e2`
- Warning: `#FFC107`

### Typography
- Title: 48-50px, weight 600
- Section Headers: 40px, weight 500
- Body Text: 36px
- Button Text: 35-40px
- Value Display: 70px, weight 500

### Spacing
- Frame padding: 25px
- Section spacing: 30px
- Control spacing: 20px
- Widget margins: 50px horizontal

## Parameter Mappings

### Display Conversions (Param → UI)
```
RTIDetectionRadius (meters) → Display as miles (meters * 0.000621371)
RTIForwardSlowdownRange (meters) → Display as miles (meters * 0.000621371)
RTIResumeSpeedDistance (meters) → Display as miles (meters * 0.000621371)
RTISpeedReduction (km/h) → Display as mph (km/h * 0.621371)
```

### Save Conversions (UI → Param)
```
UI Input (miles) → RTIDetectionRadius (miles * 1609.34)
UI Input (miles) → RTIForwardSlowdownRange (miles * 1609.34)
UI Input (miles) → RTIResumeSpeedDistance (miles * 1609.34)
UI Input (mph) → RTISpeedReduction (mph * 1.60934)
```

## Testing Checklist
- [ ] Main toggle enables/disables RTI
- [ ] Threat filter saves selection
- [ ] Detection Radius: 0.25-3.0 miles in 0.25 mile increments, default 2.0 miles
- [ ] Forward Slowdown Distance: 0.0-2.0 miles in 0.25 mile increments, default 0.75 miles
- [ ] Resume Speed Distance: 0.0-2.0 miles in 0.25 mile increments, default 0.5 miles
- [ ] Speed reduction: Posted/Custom selector works
- [ ] Custom speed: 0-30 mph in 1 mph increments
- [ ] No horizontal scrolling at any resolution
- [ ] All values display in imperial units
- [ ] All values save correctly in metric to params
- [ ] Reset buttons restore defaults
- [ ] Visual feedback for modified values
- [ ] Waze API key loads from `/persist/waze/rapidapi_key.json`

## Notes
- Advanced panel will be completely removed in this redesign
- All settings consolidated into single panel for simplicity
- Focus on clean, intuitive interface matching existing sunnypilot design language