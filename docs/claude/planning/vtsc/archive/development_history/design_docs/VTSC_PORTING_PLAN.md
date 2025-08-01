# Vision Turn Speed Controller (VTSC) Porting Plan

## Overview
This document outlines the comprehensive plan for porting the Vision Turn Speed Controller functionality from the `chauffeur-dev` branch to `chauffeur-dev2`.

## Phase 1: Core Implementation

### 1.1 Create numpy_fast Module
**File**: `/data/openpilot/common/numpy_fast.py`
**Source**: Commit 86b18cc46
**Action**: Create new file with optimized numpy imports

### 1.2 Port Enhanced VTSC Controller
**File**: `/data/openpilot/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
**Source**: Multiple commits (aa77e321e through 2cba09e3e)
**Key Changes**:
- Replace basic polynomial approach with advanced physics-based algorithms
- Implement direct model data access (orientationRate.z, velocity.x)
- Add sigmoid lateral acceleration curves (3.05 m/s² max)
- Implement multi-pass trajectory planning with apex detection
- Add jerk limiting with dynamic scaling
- Fix Cap'n Proto list slicing issues
- Fix division by zero protection
- Remove unnecessary speed reduction multipliers

### 1.3 Update Longitudinal Planner
**File**: `/data/openpilot/selfdrive/controls/lib/longitudinal_planner.py`
**Source**: Commit d900056ed
**Action**: Disable turn acceleration limiting that conflicts with VTSC

## Phase 2: UI Integration

### 2.1 HUD Display Components
**Files**: 
- `/data/openpilot/selfdrive/ui/qt/onroad/hud.cc`
- `/data/openpilot/selfdrive/ui/qt/onroad/hud.h`
**Source**: Commit f817eafd5
**Actions**:
- Add VTSC state display logic
- Add lateral acceleration visualization
- Fix lateral acceleration bar direction (commit 77c70c195)

### 2.2 Settings Panel
**Files**:
- `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal_panel.cc`
- `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal_panel.h`
**Source**: Multiple commits
**Action**: Add VisionTurnSpeedControl toggle

## Phase 3: System Configuration

### 3.1 Parameter Registration
**File**: `/data/openpilot/system/manager/manager.py`
**Action**: Register VisionTurnSpeedControl parameter

## Phase 4: Testing Infrastructure

### 4.1 Port Test Suite
**Directory**: `/data/openpilot/tools/chauffeurVtsc/`
**Files to port**:
- `README.md`
- `VTSC_TEST_REPORT.md`
- `analyze_log_events.py`
- `analyze_vtsc_from_logs.py`
- `inspect_vtsc_data.py`
- `test_vtsc_complete.py`
- `test_vtsc_comprehensive.py`
- `test_vtsc_methods_direct.py`
- `test_vtsc_real_data.py`
- `vtsc_comprehensive_analysis.png`

## Implementation Order

1. **Prerequisites** (Must be done first):
   - Create numpy_fast.py
   - Update system/manager/manager.py

2. **Core Implementation**:
   - Port vision_turn_controller.py with all fixes
   - Update longitudinal_planner.py

3. **UI Integration**:
   - Update HUD components
   - Add settings toggle

4. **Testing**:
   - Port test suite
   - Run comprehensive validation

## Critical Fixes to Include

1. **Import Error Fix** (6fb5fd17e):
   - Ensure correct import path for numpy_fast

2. **Cap'n Proto Fixes** (70c6488d5, d2f4aace3):
   - Convert Cap'n Proto lists to Python lists before slicing
   - Handle list access properly

3. **Division by Zero** (bd1ce70df):
   - Add protection in curvature calculations

4. **Acceleration Fixes** (208dc9701, a28afd432):
   - Remove neutering of acceleration
   - Fix apex detection logic

5. **Speed Limiting Fixes** (ff5bebea4):
   - Prevent limiting on straight roads

## Validation Checklist

- [ ] All files compile without errors
- [ ] No import errors (ModuleNotFoundError)
- [ ] No runtime errors (TypeError, ZeroDivisionError)
- [ ] VTSC toggle appears in settings
- [ ] HUD displays VTSC state correctly
- [ ] Lateral acceleration bar displays correctly
- [ ] Test suite passes all tests
- [ ] End-to-end testing shows proper turn speed control

## Dependencies

- Cap'n Proto (cereal)
- numpy
- Common openpilot modules (params, conversions, etc.)
- Qt framework for UI components

## Notes

- The enhanced VTSC represents a complete rewrite, not just patches
- Maintains API compatibility with existing state machine
- Fallback logic has been removed in favor of consistent physics-based approach
- All commits from aa77e321e through 2cba09e3e contain relevant changes