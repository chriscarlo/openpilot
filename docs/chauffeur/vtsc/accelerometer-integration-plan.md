# VTSC Widget Accelerometer Integration Plan

## Overview
This document outlines a comprehensive plan to integrate the onboard LSM6DS3 accelerometer sensor into the Vision Turn Speed Control (VTSC) widget, replacing the current model-predicted lateral acceleration with direct sensor measurements for improved accuracy and responsiveness.

## Current State Analysis

### Existing Implementation
- **Data Source**: Model-predicted lateral acceleration from `vtsc.getCurrentLateralAccel()`
- **Update Rate**: ~20Hz (model inference rate)
- **Processing**: Direct display of model predictions
- **Accuracy**: Subject to model limitations and prediction delays

### Available Accelerometer Resources
- **Sensor**: LSM6DS3 3-axis accelerometer
- **Sampling Rate**: 104Hz (hardware interrupt-driven)
- **Output**: Device frame coordinates [y, -x, z] in m/s²
- **Service**: Published to "accelerometer" cereal service by sensord
- **Precision**: 16-bit resolution with ±2g range

## Implementation Plan

### Step 1: Data Access Architecture

#### 1.1 Create Accelerometer Subscriber
```cpp
// selfdrive/ui/qt/onroad/accelerometer_subscriber.h
class AccelerometerSubscriber : public QObject {
  Q_OBJECT
public:
  AccelerometerSubscriber();
  float getLateralAccel() const;
  bool isDataValid() const;
  
signals:
  void accelUpdated(float lateral_g);
  
private:
  std::unique_ptr<SubMaster> sm;
  mutable std::mutex data_mutex;
  float lateral_accel_mps2;
  uint64_t last_timestamp;
  bool data_valid;
};
```

#### 1.2 Integration Points
- Subscribe to "accelerometer" service in UI process
- Maintain circular buffer for filtering
- Handle coordinate transformations
- Provide fallback to model predictions

### Step 2: Coordinate Frame Transformation

#### 2.1 Frame Definitions
- **Device Frame**: LSM6DS3 output [y, -x, z]
- **Vehicle Frame**: Standard automotive [x_forward, y_left, z_up]
- **Display Frame**: Passenger-centric (left turn = pushed right)

#### 2.2 Transformation Pipeline
```cpp
// Device to Vehicle transformation
float device_to_vehicle_lateral(const cereal::SensorEventData::Acceleration& accel) {
  // Device Y-axis maps to vehicle lateral (with sign correction)
  // Account for device mounting orientation
  return accel.getV()[0]; // Device Y is index 0 after axis remapping
}

// Vehicle to Passenger perception
float vehicle_to_passenger_perception(float vehicle_lateral) {
  // Invert for passenger perspective (left turn = pushed right)
  return -vehicle_lateral;
}
```

#### 2.3 Gravity Compensation
- Implement high-pass filter to remove gravity component
- Use gyroscope data for device orientation estimation
- Apply complementary filter for robust gravity removal

### Step 3: Signal Processing

#### 3.1 Low-Pass Filter Design
```cpp
class ButterworthFilter {
  // 2nd order Butterworth, fc = 5Hz
  // Removes high-frequency noise while preserving turn dynamics
  float process(float input);
};
```

#### 3.2 Spike Rejection
- Median filter with 5-sample window
- Outlier detection: reject values > 3σ from moving average
- Smooth transitions during sensor recovery

#### 3.3 Latency Compensation
- Timestamp alignment between sensor and model data
- Predictive filtering using vehicle dynamics model
- Adjustable phase lead compensation

### Step 4: UI Integration

#### 4.1 HudRenderer Modifications
```cpp
// hud.h additions
class HudRenderer {
  // ...existing members...
  
  // New accelerometer data
  std::unique_ptr<AccelerometerSubscriber> accel_subscriber;
  float sensor_lateral_accel;
  bool use_sensor_accel;
  
  // Fusion parameters
  float sensor_confidence;
  float model_confidence;
  float fused_lateral_accel;
};
```

#### 4.2 Data Fusion Strategy
```cpp
float HudRenderer::computeFusedAccel() {
  if (!accel_subscriber->isDataValid()) {
    // Fallback to model only
    return vtsc_current_lateral_accel;
  }
  
  // Weighted fusion based on confidence
  float sensor_weight = sensor_confidence / (sensor_confidence + model_confidence);
  return sensor_lateral_accel * sensor_weight + 
         vtsc_current_lateral_accel * (1 - sensor_weight);
}
```

#### 4.3 Display Updates
- Update `drawLateralAccelMeter()` to use fused data
- Add sensor status indicator (optional)
- Smooth visual transitions during source switching

### Step 5: Testing Strategy

#### 5.1 Unit Tests
- Coordinate transformation validation
- Filter response characterization
- Spike rejection effectiveness
- Gravity compensation accuracy

#### 5.2 Integration Tests
- Sensor data flow validation
- Timing and synchronization
- Fallback mechanism testing
- Memory and CPU usage profiling

#### 5.3 On-Device Testing
```bash
# Test scenarios
1. Static device (gravity only) - verify zero lateral reading
2. Straight driving - minimal lateral acceleration
3. Gentle curves - smooth acceleration changes
4. Sharp turns - peak acceleration handling
5. Sensor disconnect - fallback to model
6. Rough road - noise rejection
```

#### 5.4 Validation Metrics
- Correlation with model predictions (should be high but sensor leads)
- Noise floor measurement
- Response latency
- Peak accuracy during turns

### Step 6: Implementation Rollout

#### 6.1 Phase 1: Data Collection (No UI Changes)
- Add accelerometer subscriber
- Log sensor vs model data
- Analyze correlation and timing
- Tune filter parameters

#### 6.2 Phase 2: Shadow Mode
- Process sensor data in parallel
- Display model data as before
- Log differences for analysis
- Validate transformation accuracy

#### 6.3 Phase 3: Hybrid Display
- Implement confidence-based fusion
- Default to 70% sensor, 30% model
- Add debug overlay (optional)
- Monitor user feedback

#### 6.4 Phase 4: Full Integration
- Sensor as primary source
- Model as fallback only
- Remove debug features
- Document configuration options

### Step 7: Configuration and Tuning

#### 7.1 User Settings
```python
# selfdrive/manager/manager.py
VTSC_ACCEL_SOURCE = Param("VTSCAccelSource")  # "sensor", "model", "hybrid"
VTSC_FILTER_CUTOFF = Param("VTSCFilterCutoff")  # Hz
VTSC_SENSOR_WEIGHT = Param("VTSCSensorWeight")  # 0.0 to 1.0
```

#### 7.2 Runtime Parameters
- Filter coefficients
- Spike rejection thresholds
- Confidence decay rates
- Transition smoothing factors

#### 7.3 Debug Interface
```cpp
// Optional debug overlay
void drawAccelDebugInfo(QPainter &p) {
  QString info = QString("Sensor: %1 m/s² | Model: %2 m/s² | Fused: %3 m/s²")
    .arg(sensor_lateral_accel, 0, 'f', 2)
    .arg(vtsc_current_lateral_accel, 0, 'f', 2)
    .arg(fused_lateral_accel, 0, 'f', 2);
  // Draw debug text
}
```

### Step 8: Future Enhancements

#### 8.1 Advanced Filtering
- Kalman filter with vehicle dynamics model
- Machine learning-based noise prediction
- Adaptive filter tuning based on road conditions

#### 8.2 Multi-Sensor Fusion
- Incorporate gyroscope for better orientation
- Use GPS for velocity-based validation
- Steering angle correlation

#### 8.3 Predictive Display
- Lead compensation for display latency
- Trajectory-based acceleration prediction
- Smooth preview of upcoming acceleration

#### 8.4 Calibration Improvements
- Automatic mounting angle detection
- Temperature compensation
- Aging compensation

## Risk Mitigation

### Potential Issues and Solutions

1. **Sensor Noise in Rough Roads**
   - Solution: Adaptive filtering based on road roughness detection
   - Fallback: Increase model weight in high-noise conditions

2. **Mounting Variations**
   - Solution: Auto-calibration routine on startup
   - Fallback: Manual calibration option in settings

3. **Sensor Failure**
   - Solution: Continuous health monitoring
   - Fallback: Seamless transition to model-only mode

4. **Processing Overhead**
   - Solution: Optimize filter implementations
   - Fallback: Reduce sampling rate if needed

5. **User Confusion**
   - Solution: Clear documentation and gradual rollout
   - Fallback: Option to disable sensor integration

## Success Metrics

1. **Accuracy**: <100ms latency improvement over model
2. **Smoothness**: <5% additional jitter vs model
3. **Reliability**: >99.9% uptime with fallback
4. **Performance**: <1% additional CPU usage
5. **User Satisfaction**: Positive feedback on responsiveness

## Timeline Estimate

- **Week 1-2**: Implement data access and transformation
- **Week 3**: Signal processing and filtering
- **Week 4**: UI integration and basic testing
- **Week 5-6**: On-device testing and tuning
- **Week 7**: Shadow mode deployment
- **Week 8**: Full rollout with monitoring

## Conclusion

This plan provides a systematic approach to integrating the LSM6DS3 accelerometer into the VTSC widget, offering improved accuracy and responsiveness while maintaining reliability through robust fallback mechanisms. The phased rollout ensures minimal risk while maximizing the benefits of direct sensor measurements.