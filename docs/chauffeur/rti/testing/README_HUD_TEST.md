# RTI HUD Widget Message Injection Test

## Overview

This test injects fake `rtiStateSP` cereal messages into the live message stream to test the RTI HUD widget display without requiring production code changes or builds.

## Setup

1. **Start the UI in onroad mode:**
   ```bash
   cd /data/openpilot
   FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &
   ```

2. **Run the test script:**
   ```bash
   cd /data/openpilot/docs/chauffeur/rti/testing
   python3 test_hud_message_injection.py
   ```

## Test Options

The script provides several test modes:

1. **Full test sequence** - Cycles through all threat scenarios
2. **Single POLICE threat** - Quick verification test (300m distance)
3. **Single CAMERA threat** - Test speed camera display (400m distance) 
4. **Multiple threats** - Test multi-threat widget display
5. **Custom single threat** - Specify threat type and distance

## Expected Behavior

### HUD Widget Location
- Bottom-left corner of the HUD
- Should appear as a rounded rectangle with RTI content

### Color Coding by Distance
- **RED (Critical)**: < 100m
- **ORANGE (Near)**: 100-300m  
- **YELLOW (Normal)**: 300-1000m
- **GRAY (Far)**: > 1000m

### Widget States
1. **No threats**: Shows "RTI" placeholder text
2. **Single threat**: Shows threat type, distance, directional arrow
3. **Multiple threats**: Shows list of threats sorted by distance
4. **Active speed control**: Shows recommended speed at bottom

## Troubleshooting

### If HUD widget doesn't appear:
1. Verify RTI is enabled in params: `RTIEnabled = true`
2. Verify HUD display is enabled: `RTIHUDEnabled = true`
3. Check UI is running in onroad mode (not offroad)
4. Ensure script is publishing messages (check console output)

### If messages don't update:
1. Check for cereal messaging errors in console
2. Verify script is running without exceptions
3. Try restarting the UI

## Test Scenarios

The script tests these scenarios:

1. **No threats** - Baseline state
2. **Close police (150m)** - Red/critical display  
3. **Medium camera (250m)** - Orange/near display
4. **Far accident (800m)** - Yellow/normal display
5. **Multiple threats** - Multi-threat widget with 4 different threats
6. **Construction with speed** - Speed recommendation display

## Message Structure

The script publishes `rtiStateSP` messages with this structure:
- `threatAhead`: boolean
- `threatDistanceM`: distance in meters
- `recommendedSpeed`: speed in m/s
- `threats[]`: array of threat objects with type, location, distance
- `apiStatus`: connection status
- `source`: data source identifier

## Usage Notes

- Each scenario runs for 8 seconds to allow observation
- Use Ctrl+C to stop the test cleanly
- Script sends a "no threats" cleanup message on exit
- Messages are published every 2 seconds in single threat tests