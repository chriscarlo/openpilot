# LiveSteerRatio Implementation Notes

## Design Decisions

### 1. Why Change KIA EV6 Default to 13.43?

The original value of 16.0 was likely too high, resulting in:
- Less responsive steering
- Potential issues with lateral control tuning
- Misalignment with other Hyundai/Kia vehicles

The new value of 13.43 was chosen based on:
- Similar vehicles in the Hyundai/Kia lineup
- Community feedback and testing
- Better alignment with openpilot's control algorithms

### 2. Parameter Range (0.0 - 25.0)

The range was selected to:
- Allow full stock value (16.0) to be restored
- Provide headroom for experimentation
- Prevent unsafe extreme values
- Use 0 as a special "use default" value

### 3. GUI Implementation Choice

Used ButtonControlSP with InputDialog instead of a slider because:
- Precise value entry is important
- Range is too wide for comfortable slider use
- Consistent with other sunnypilot parameter inputs
- Shows current value in button description

## Implementation Challenges

### 1. Parameter Registration

The LiveSteerRatio parameter needed to be added to `params_keys.h` before it could be used. This required rebuilding the common module.

### 2. Enum vs Value Access

The CAR enum in opendbc returns platform configs, not raw values. Tests needed adjustment to access `CAR.KIA_EV6.value.specs` or just `CAR.KIA_EV6.specs`.

### 3. Backward Compatibility

The retrieve_initial_vehicle_params function signature changed to return an additional value (base_steer_ratio). All callers needed updates.

## Code Architecture

### Parameter Flow Diagram

```
GUI (hyundai_settings.cc)
    ↓
Params Database
    ↓
paramsd.py (on startup)
    ↓
retrieve_initial_vehicle_params()
    ↓
VehicleParamsLearner.__init__()
    ↓
Bounds calculation & initial value
```

### Key Functions

1. **retrieve_initial_vehicle_params()**
   - Reads LiveSteerRatio from params
   - Determines base_steer_ratio
   - Returns it along with other parameters

2. **VehicleParamsLearner.__init__()**
   - Accepts base_steer_ratio parameter
   - Uses it for bounds calculation
   - Falls back to CP.steerRatio if None

3. **GUI Input Handler**
   - Validates input range (0.0-25.0)
   - Stores as string in params
   - Shows current value in description

## Testing Strategy

### Unit Tests

1. **Default Value Test**
   - Verifies KIA EV6 uses 13.43
   - Direct access to CAR enum specs

2. **Parameter Handling Test**
   - Tests no parameter → uses default
   - Tests 0 value → uses default
   - Tests custom value → uses custom

3. **Bounds Calculation Test**
   - Verifies min = 0.5 × base
   - Verifies max = 2.0 × base
   - Tests with different base values

4. **Storage Test**
   - Verifies parameter persistence
   - Tests value encoding/decoding

### Integration Testing

The demo script provides a walkthrough of:
- Default behavior
- Setting to 0
- Custom values
- GUI interaction
- Practical use cases

## Security Considerations

1. **Input Validation**
   - GUI enforces 0.0-25.0 range
   - Backend uses bounds to limit learned values
   - No arbitrary code execution possible

2. **Safe Defaults**
   - 0 always means "use vehicle default"
   - Vehicle default is hardcoded, not user-modifiable
   - Bounds prevent extreme values

## Performance Impact

- Parameter read only at startup (negligible)
- No runtime overhead during driving
- Bounds check already part of parameter learning

## Maintenance Notes

### Adding LiveSteerRatio for Other Vehicles

1. No code changes needed in paramsd
2. Add GUI control to appropriate settings file
3. Update description with correct default value
4. Consider vehicle-specific range limits

### Debugging Issues

Common problems and solutions:
- **Parameter not recognized**: Rebuild common module
- **GUI not showing**: Check settings visibility logic
- **Value not persisting**: Check params permissions
- **Tests failing**: Check enum access pattern

## Related Systems

### Parameter Learning

The LiveSteerRatio affects:
- Initial steer ratio value
- Learning bounds (min/max)
- Does NOT affect learning rate or algorithm

### Lateral Control

Changes to steer ratio impact:
- Steering command scaling
- Path tracking accuracy
- User steering feel

Always test thoroughly after changes!