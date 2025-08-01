# Vision Turn Speed Controller (VTSC) - Technical Documentation

## Overview

The Vision Turn Speed Controller (VTSC) is an advanced physics-based turn speed control system that automatically adjusts vehicle speed for curves and turns while OpenPilot's longitudinal control is engaged. This enhanced version implements sophisticated algorithms for smoother, more accurate turn speed management.

## Key Features

### 1. Direct Model Data Access
- Utilizes `orientationRate.z` and `velocity.x` directly from modelV2
- More accurate than polynomial approximation methods
- Real-time curvature calculation: `curvature = |angular_velocity| / linear_velocity`

### 2. Sigmoid Lateral Acceleration Curves
- Maximum lateral acceleration: 3.05 m/s² 
- Sigmoid function parameters:
  - Steepness (k): 0.8
  - Midpoint (x0): 2.5 m/s
- Provides smooth transition between low and high-speed handling

### 3. Multi-Pass Trajectory Planning
- Apex detection algorithm identifies local curvature maxima
- Dynamic scaling based on apex distance:
  - 20m: 1.2x scaling
  - 50m: 1.0x scaling  
  - 100m: 0.8x scaling
- Predictive planning horizon: 10 seconds (33 points)

### 4. Jerk Limiting
- Maximum jerk: 2.0 m/s³
- Speed-dependent scaling:
  - 20 m/s: 0.5x jerk limit
  - 40 m/s: 1.0x jerk limit
- Ensures smooth acceleration transitions

### 5. State Machine

The VTSC operates through four distinct states:

#### DISABLED (0)
- Default state when system is off
- Transitions to ENTERING when:
  - Speed > 20 km/h
  - Predicted lateral acceleration ≥ 1.3 m/s²
  - Model confidence ≥ 0.5

#### ENTERING (1) 
- Prepares for upcoming turn
- Smooth deceleration: -0.2 to -1.0 m/s²
- Calculates required deceleration to reach target speed
- Transitions to TURNING when current lateral acceleration ≥ 1.6 m/s²

#### TURNING (2)
- Active turn management
- Dynamic acceleration based on current lateral acceleration
- Additional reduction near vehicle limits (>80% of max)
- Transitions to LEAVING when current lateral acceleration ≤ 1.3 m/s²

#### LEAVING (3)
- Exiting turn phase
- Comfortable acceleration: 0.5 m/s²
- Scaled based on proximity to cruise speed
- Returns to DISABLED when lateral acceleration < 1.1 m/s²

## Implementation Details

### File Structure
```
/data/openpilot/
├── common/
│   └── numpy_fast.py                    # Optimized numpy imports
├── sunnypilot/selfdrive/controls/lib/
│   └── vision_turn_controller.py        # Main VTSC implementation
├── selfdrive/controls/lib/
│   └── longitudinal_planner.py          # Modified to disable conflicting turn limits
├── selfdrive/ui/qt/onroad/
│   ├── hud.h                           # HUD declarations
│   └── hud.cc                          # HUD implementation with VTSC display
├── selfdrive/ui/sunnypilot/qt/offroad/settings/
│   ├── longitudinal_panel.h            # Settings panel declarations
│   └── longitudinal_panel.cc           # VTSC toggle implementation
└── tools/chauffeurVtsc/                # Test suite
```

### Key Classes and Functions

#### VisionTurnController Class
```python
class VisionTurnController:
    def __init__(self, CP)
    def update(self, sm, enabled, v_ego, a_ego, v_cruise_setpoint)
    
    # Properties
    @property def state() -> VisionTurnSpeedControlState
    @property def a_target() -> float  # Target acceleration with jerk limiting
    @property def v_turn() -> float    # Target turn speed
    @property def is_active() -> bool
    
    # Internal methods
    def _calculate_curvatures(sm) -> (curvatures, distances)
    def _update_calculations(sm) -> None
    def _state_transition() -> None
    def _update_solution() -> None
```

#### Helper Functions
- `sigmoid_lat_acc(v)`: Calculate max lateral acceleration using sigmoid
- `calculate_curvature_from_model(model_data)`: Direct curvature from model
- `find_apex_points(curvatures, distances)`: Detect turn apexes
- `apply_jerk_limiting(current, target, dt, v_ego)`: Smooth acceleration

### Configuration Parameters

#### State Transition Thresholds
- `_MIN_V`: 20 km/h - Minimum operating speed
- `_ENTERING_PRED_LAT_ACC_TH`: 1.3 m/s² - Enter turn threshold
- `_TURNING_LAT_ACC_TH`: 1.6 m/s² - Start turning threshold
- `_LEAVING_LAT_ACC_TH`: 1.3 m/s² - Start leaving threshold
- `_FINISH_LAT_ACC_TH`: 1.1 m/s² - Finish turn threshold

#### Physics Limits
- `_A_LAT_REG_MAX`: 3.05 m/s² - Maximum lateral acceleration
- `_MAX_JERK`: 2.0 m/s³ - Maximum jerk
- `_NO_OVERSHOOT_TIME_HORIZON`: 4.0 s - Planning horizon

#### Vision Confidence
- `_MIN_LANE_PROB`: 0.6 - Minimum lane probability
- `_MIN_PATH_CONFIDENCE`: 0.5 - Minimum path confidence

## Usage

### Enabling VTSC
1. Navigate to Settings → Driving → Longitudinal
2. Toggle "Vision Turn Speed Controller"
3. The setting is stored in params as "VisionTurnSpeedControl"

### HUD Display
When active, the HUD shows:
- Current state (TURN AHEAD, TURNING, LEAVING)
- Target speed (if significantly different from current)
- Lateral acceleration indicator bar
- State-specific background colors

### Integration Points

1. **Longitudinal Planner**: Receives `a_target` and `v_turn` from VTSC
2. **HUD System**: Displays VTSC state and metrics
3. **Settings Panel**: User control toggle
4. **Parameter Store**: Persistent enable/disable setting

## Testing

### Basic Validation
```bash
python3 docs/claude/tests/vtsc/test_vtsc_basic.py
```

### Test Coverage
- Initialization and state management
- State machine transitions
- Curvature calculations
- Jerk limiting functionality
- Model data processing

## Performance Considerations

1. **Computational Efficiency**
   - Direct model access avoids polynomial fitting overhead
   - numpy_fast module reduces import time
   - Optimized calculations run at planner frequency (20Hz)

2. **Memory Usage**
   - Minimal state storage
   - No historical data retention beyond current frame
   - Efficient numpy array operations

## Safety Features

1. **Graceful Degradation**
   - Falls back to polynomial method if model data unavailable
   - Disables on gas press or system disable
   - Respects minimum speed threshold

2. **Smooth Transitions**
   - Jerk limiting prevents abrupt changes
   - State hysteresis prevents oscillation
   - Gradual acceleration/deceleration profiles

3. **Conservative Limits**
   - 90% of theoretical maximum for comfort
   - Speed-dependent jerk scaling
   - Apex-aware deceleration timing

## Future Enhancements

1. **Adaptive Tuning**
   - Learn driver preferences over time
   - Adjust aggressiveness based on driving style
   - Road condition adaptation

2. **Enhanced Prediction**
   - Multi-apex handling for S-curves
   - Banking angle compensation
   - Weather/surface condition factors

3. **Integration Improvements**
   - Coordinate with ACC for smoother transitions
   - Map data integration for preview
   - V2V communication for curve warnings

## Troubleshooting

### Common Issues

1. **VTSC Not Activating**
   - Check minimum speed (>20 km/h)
   - Verify parameter enabled in settings
   - Ensure modelV2 data is valid

2. **Jerky Behavior**
   - Check jerk limiting is functioning
   - Verify model confidence thresholds
   - Ensure no conflicting longitudinal limits

3. **Conservative Speed**
   - Normal behavior for safety
   - Check lateral acceleration limits
   - Verify curvature calculations

### Debug Mode
Enable debug output by setting `_DEBUG = True` in vision_turn_controller.py

## Version History

### Enhanced Version (Current)
- Direct model data access
- Sigmoid lateral acceleration
- Apex detection
- Jerk limiting
- Complete fix suite from commits aa77e321e through 2cba09e3e

### Original Version
- Polynomial-based curvature estimation
- Fixed lateral acceleration limits
- Basic state machine
- No jerk limiting