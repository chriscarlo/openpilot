# RTI Metric/Imperial Unit Conversion Implementation Plan

## Overview

Implement metric/imperial unit conversion for RTI (Realtime Traffic Intelligence) distance sliders with proper increments and limits. The system currently displays all distances in meters; we need to add proper unit conversion and adjust slider increments based on the system's metric setting.

## Current System Analysis

### GUI Components
- **Main Settings Panel**: `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.cc/.h`
- **Advanced Panel**: `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_advanced_panel.cc/.h`
- **Control Widgets**: `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.cc/.h`

### Current Slider Configuration
```cpp
// Min distance slider: 50-2000m, step 50, default 100m
rti_min_slider->setRange(50, 2000);
rti_min_slider->setSingleStep(50);
rti_min_slider->setValue(safeStringToInt(params.get("RTIMinDistance"), 100));

// Max distance slider: 500-5000m, step 100, default 2000m  
rti_max_slider->setRange(500, 5000);
rti_max_slider->setSingleStep(100);
rti_max_slider->setValue(safeStringToInt(params.get("RTIMaxDistance"), 2000));
```

### Backend Parameters
- `RTIMinDistance`: stored in meters (default "100")
- `RTIMaxDistance`: stored in meters (default "2000")  
- System uses `params.get_bool("IsMetric")` for unit detection
- Backend expects all distance values in meters

### Requirements
- **Imperial Mode**: 0.25 mile increments, 0.25mi - 2mi range
- **Metric Mode**: 0.5 km increments, 0.5km - 5km range  
- **Internal Storage**: Continue storing in meters for backend compatibility

## Implementation Plan

### Phase 1: Unit Conversion Infrastructure

#### 1.1 Add conversion constants to RTI headers
Add to `rti_settings_panel.h` and `rti_advanced_panel.h`:
```cpp
// Conversion constants
static constexpr double METERS_TO_MILES = 0.000621371;
static constexpr double MILES_TO_METERS = 1609.344;
static constexpr double METERS_TO_KM = 0.001;
static constexpr double KM_TO_METERS = 1000.0;

// Slider increments 
static constexpr double IMPERIAL_INCREMENT_MI = 0.25;  // 0.25 miles
static constexpr double METRIC_INCREMENT_KM = 0.5;     // 0.5 km
```

#### 1.2 Add metric detection helper
```cpp
private:
    bool isMetricSystem() const {
        return params.get_bool("IsMetric");
    }
```

### Phase 2: Main Settings Panel Updates

#### 2.1 Update slider configuration in RTISettingsPanel constructor

Replace current slider setup with unit-aware configuration:

```cpp
// Distance Settings - Unit-aware configuration
private slots:
    void updateDistanceSliders();
    void onMinDistanceChanged(int value);
    void onMaxDistanceChanged(int value);

private:
    void configureDistanceSliders();
    QString formatDistanceLabel(int meters_value, bool is_minimum) const;
    int metersToSliderValue(int meters) const;
    int sliderValueToMeters(int slider_value) const;
```

#### 2.2 Implement dynamic slider configuration
```cpp
void RTISettingsPanel::configureDistanceSliders() {
    const bool is_metric = isMetricSystem();
    
    if (is_metric) {
        // Metric: 0.5km - 5km in 0.5km increments
        // Convert to meters: 500m - 5000m in 500m increments  
        rti_min_slider->setRange(500, 5000);
        rti_min_slider->setSingleStep(500);
        rti_max_slider->setRange(500, 5000); 
        rti_max_slider->setSingleStep(500);
    } else {
        // Imperial: 0.25mi - 2mi in 0.25mi increments
        // Convert to meters: 402m - 3219m in 402m increments
        const int min_meters = static_cast<int>(IMPERIAL_INCREMENT_MI * MILES_TO_METERS);
        const int max_meters = static_cast<int>(2.0 * MILES_TO_METERS);
        const int step_meters = min_meters;
        
        rti_min_slider->setRange(min_meters, max_meters);
        rti_min_slider->setSingleStep(step_meters);
        rti_max_slider->setRange(min_meters, max_meters);
        rti_max_slider->setSingleStep(step_meters);
    }
    
    // Update current values from params
    int current_min = safeStringToInt(params.get("RTIMinDistance"), 500);
    int current_max = safeStringToInt(params.get("RTIMaxDistance"), 2000);
    
    // Snap to nearest valid increment
    current_min = snapToValidIncrement(current_min);
    current_max = snapToValidIncrement(current_max);
    
    rti_min_slider->setValue(current_min);
    rti_max_slider->setValue(current_max);
}
```

#### 2.3 Add snap-to-increment logic
```cpp
int RTISettingsPanel::snapToValidIncrement(int meters) const {
    const bool is_metric = isMetricSystem();
    
    if (is_metric) {
        // Round to nearest 0.5km (500m)
        return ((meters + 250) / 500) * 500;
    } else {
        // Round to nearest 0.25mi (402m)  
        const int increment_m = static_cast<int>(IMPERIAL_INCREMENT_MI * MILES_TO_METERS);
        return ((meters + increment_m/2) / increment_m) * increment_m;
    }
}
```

#### 2.4 Update label formatting
```cpp
QString RTISettingsPanel::formatDistanceLabel(int meters_value, bool is_minimum) const {
    const bool is_metric = isMetricSystem();
    const QString prefix = is_minimum ? tr("Minimum: ") : tr("Maximum: ");
    
    if (is_metric) {
        const double km = meters_value * METERS_TO_KM;
        return QString("%1%2 km").arg(prefix).arg(km, 0, 'f', 1);
    } else {
        const double miles = meters_value * METERS_TO_MILES;  
        return QString("%1%2 mi").arg(prefix).arg(miles, 0, 'f', 2);
    }
}
```

### Phase 3: Advanced Panel Updates

#### 3.1 Update RTIVisualizationWidget
The visualization widget also needs to display proper units in its labels.

Update `RTIVisualizationWidget::drawThreatZones()` method:
```cpp
// Zone labels with proper units
painter.setFont(label_font);
painter.setPen(QColor(255, 255, 255, 180));

if (min_y > layout.road_end.y() + 50) {
    QString min_label = formatDistanceForVisualization(min_distance_m);
    painter.drawText(QRect(10, min_y - 15, width() - 20, 30),
                    Qt::AlignCenter, QString("Min Detection: %1").arg(min_label));
}
```

#### 3.2 Update RTIConfigDataPanel
Update distance display in the configuration data panel:
```cpp
void RTIConfigDataPanel::updateConfiguration(RTIAggressiveness aggressiveness, int min_dist, int max_dist, int speed_reduction) {
    // ... existing code ...
    
    // Format distance based on metric setting
    Params params;
    bool is_metric = params.get_bool("IsMetric");
    QString distance_text;
    
    if (is_metric) {
        double min_km = min_dist * 0.001;
        double max_km = max_dist * 0.001;
        distance_text = QString("• Detection Range: %1 - %2 km")
            .arg(min_km, 0, 'f', 1).arg(max_km, 0, 'f', 1);
    } else {
        double min_mi = min_dist * 0.000621371;
        double max_mi = max_dist * 0.000621371; 
        distance_text = QString("• Detection Range: %1 - %2 mi")
            .arg(min_mi, 0, 'f', 2).arg(max_mi, 0, 'f', 2);
    }
    
    distance_info->setText(distance_text);
    // ... rest of existing code ...
}
```

### Phase 4: Parameter Migration and Validation

#### 4.1 Add parameter validation on load
```cpp
void RTISettingsPanel::validateAndMigrateParameters() {
    // Get current values
    int current_min = safeStringToInt(params.get("RTIMinDistance"), 500);
    int current_max = safeStringToInt(params.get("RTIMaxDistance"), 2000);
    
    // Ensure they meet new constraints
    const bool is_metric = isMetricSystem();
    int valid_min = snapToValidIncrement(current_min);
    int valid_max = snapToValidIncrement(current_max);
    
    // Enforce range limits
    if (is_metric) {
        valid_min = std::max(500, std::min(5000, valid_min));   // 0.5km - 5km
        valid_max = std::max(500, std::min(5000, valid_max));
    } else {
        const int min_imperial_m = static_cast<int>(0.25 * MILES_TO_METERS);   // 0.25mi
        const int max_imperial_m = static_cast<int>(2.0 * MILES_TO_METERS);    // 2mi
        valid_min = std::max(min_imperial_m, std::min(max_imperial_m, valid_min));
        valid_max = std::max(min_imperial_m, std::min(max_imperial_m, valid_max));
    }
    
    // Ensure min < max
    if (valid_min >= valid_max) {
        if (is_metric) {
            valid_min = 500;  // 0.5km
            valid_max = 1000; // 1km
        } else {
            valid_min = static_cast<int>(0.25 * MILES_TO_METERS);  // 0.25mi
            valid_max = static_cast<int>(0.5 * MILES_TO_METERS);   // 0.5mi  
        }
    }
    
    // Update params if values changed
    if (valid_min != current_min) {
        params.put("RTIMinDistance", std::to_string(valid_min));
    }
    if (valid_max != current_max) {
        params.put("RTIMaxDistance", std::to_string(valid_max));
    }
}
```

### Phase 5: Responsive Updates

#### 5.1 Add metric setting change detection
```cpp
void RTISettingsPanel::showEvent(QShowEvent *event) {
    // Validate parameters and update sliders when panel is shown
    validateAndMigrateParameters();
    configureDistanceSliders();
    refresh();
}

// Also add to refresh() method:
void RTISettingsPanel::refresh() {
    // Check if metric setting changed
    static bool last_metric_state = isMetricSystem();
    bool current_metric_state = isMetricSystem();
    
    if (current_metric_state != last_metric_state) {
        // Metric setting changed - reconfigure sliders
        configureDistanceSliders();
        last_metric_state = current_metric_state;
    }
    
    // ... existing refresh logic ...
}
```

### Phase 6: Testing Updates

#### 6.1 Update integration tests
Add tests for metric/imperial conversion in `test_rti_integration.py`:

```python
def test_metric_imperial_conversion(self):
    """Test RTI parameter conversion between metric and imperial."""
    framework = RTITestFramework()
    framework.setup()
    
    # Test metric mode
    framework.params.put_bool("IsMetric", True)
    framework.params.put("RTIMinDistance", "1000")  # 1km
    # Validate GUI shows "1.0 km"
    
    # Test imperial mode  
    framework.params.put_bool("IsMetric", False)
    framework.params.put("RTIMinDistance", "1609")  # ~1 mile
    # Validate GUI shows "1.00 mi"
    
    framework.teardown()

def test_parameter_validation(self):
    """Test that RTI parameters are validated and snapped to valid increments."""
    framework = RTITestFramework()
    framework.setup()
    
    # Test invalid metric value gets snapped
    framework.params.put_bool("IsMetric", True)
    framework.params.put("RTIMinDistance", "750")  # Invalid - should snap to 500 or 1000
    # Validate parameter gets corrected
    
    # Test invalid imperial value gets snapped
    framework.params.put_bool("IsMetric", False)  
    framework.params.put("RTIMinDistance", "500")  # Invalid - should snap to nearest 0.25mi increment
    # Validate parameter gets corrected
    
    framework.teardown()
```

## Implementation Steps

1. **Phase 1**: Add conversion constants and helper functions
2. **Phase 2**: Update main settings panel slider configuration 
3. **Phase 3**: Update advanced panel visualizations
4. **Phase 4**: Add parameter validation and migration
5. **Phase 5**: Add responsive updates for metric setting changes
6. **Phase 6**: Update test suite
7. **Validation**: Run comprehensive tests and validate functionality
8. **Debugging**: Fix any issues found during testing

## Risk Assessment

- **Low Risk**: Unit conversion math and display formatting
- **Medium Risk**: Slider reconfiguration and value snapping logic
- **High Risk**: Parameter migration ensuring no existing settings are broken

## Success Criteria

1. ✅ RTI sliders show proper units (km/mi) based on IsMetric setting
2. ✅ Imperial mode: 0.25mi increments, 0.25mi-2mi range  
3. ✅ Metric mode: 0.5km increments, 0.5km-5km range
4. ✅ Internal storage remains in meters for backend compatibility
5. ✅ Existing RTI functionality unaffected
6. ✅ All tests pass
7. ✅ Proper parameter validation and migration
8. ✅ Responsive to IsMetric setting changes