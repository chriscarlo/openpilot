# Driver Monitoring Disable Feature Plan

## Overview
This feature disables all driver monitoring functionality in the chauffeur-dev2 branch while maintaining interface compatibility.

## Commits Cherry-Picked
1. **4143fe403ca0ea356e5526613b7c2fe94ab9cd50** - Simplify driver monitoring to disable all monitoring features
2. **0fe7ba7a6db1c620c9cf07fcf32d23819bd80a13** - Remove driver monitoring icon from onroad HUD

## Changes Applied

### 1. Driver Monitoring Backend (`selfdrive/monitoring/helpers.py`)
- Replaced full monitoring implementation with minimal stub
- Always returns nominal values:
  - Face always detected
  - Never distracted
  - Awareness always at 1.0
- Maintains interface compatibility for dmonitoringd
- Effectively disables all driver attention monitoring

### 2. UI Changes
- **annotated_camera.cc**:
  - Removed `dmon.updateState(s)` call
  - Removed `dmon.draw(painter, rect())` call
- **annotated_camera.h**:
  - Removed `#include "selfdrive/ui/qt/onroad/driver_monitoring.h"`
  - Removed `DriverMonitorRenderer dmon` member variable

## Implementation Checklist

- [x] Research commits in chauffeur-dev branch
- [x] Apply monitoring backend changes (helpers.py)
- [x] Apply UI changes (remove monitoring icon)
- [x] Verify all changes are consistent
- [ ] Test build compilation
- [ ] Test runtime behavior
- [ ] Verify dmonitoringd still functions without errors
- [ ] Confirm no monitoring alerts appear during driving

## Testing Requirements

1. **Build Testing**
   - Ensure code compiles without errors
   - Check for any missing dependencies

2. **Runtime Testing**
   - Verify dmonitoringd process starts correctly
   - Confirm no driver monitoring alerts appear
   - Check that UI displays correctly without monitoring icon
   - Test with both LHD and RHD configurations

3. **Integration Testing**
   - Ensure no crashes or errors in log files
   - Verify car engagement/disengagement works normally
   - Confirm no side effects on other safety systems

## Risks and Considerations

1. **Safety Impact**: Driver monitoring is a safety feature. Disabling it removes attention tracking.
2. **Interface Compatibility**: Changes maintain interface to avoid breaking other components.
3. **Testing Coverage**: Thorough testing needed to ensure no unintended side effects.

## Next Steps

1. Attempt to build the project to verify compilation
2. Run unit tests for monitoring module
3. Test on device or in simulation
4. Document any issues encountered