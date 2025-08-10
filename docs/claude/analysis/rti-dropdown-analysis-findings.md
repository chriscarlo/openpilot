# RTI Dropdown Analysis - Findings Document

## Problem Statement
- RTI dropdown menus render rotated 90 degrees counter-clockwise
- Click interactions are completely unresponsive
- Suspected code duplication causing conflicts

## Investigation Progress

### Phase 1: Initial Analysis
**Status**: Starting
**Target**: Main RTI settings panel and dropdown implementations

**Files Under Investigation**:
1. `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.cc` - Main dropdown file
2. `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.h` - Header
3. `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.cc` - Control impl
4. `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.h` - Control header
5. `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_advanced_panel.cc` - Advanced panel
6. `/data/openpilot/selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/longitudinal_panel.cc` - Integration

**Key Dropdown Controls Identified**:
- `rti_source_combo` (Data Source)
- `rti_filter_combo` (Threat Filter)  
- `rti_aggr_combo` (Response Style)

## Findings Log

### Finding #1: Main Dropdown Implementation Analysis ✅
**File**: `rti_settings_panel.cc`
**Lines**: 103-214

**Key Observations**:
- 3 QComboBox controls properly implemented: `rti_source_combo`, `rti_filter_combo`, `rti_aggr_combo`
- All 3 dropdowns share the SAME stylesheet via `rti_source_combo->styleSheet()` (lines 173, 199)
- CSS styling contains complex subcontrol positioning and arrow styling (lines 104-143)
- **POTENTIAL ISSUE**: Extensive custom QComboBox styling with manual arrow positioning and subcontrol positioning
- Event handlers look clean - using QOverload<int>::of pattern correctly
- No obvious duplication in this file yet

**Suspicious CSS Elements**:
```css
QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right; // ← Could cause rotation?
}
QComboBox::down-arrow {
  border-left: 12px solid transparent;
  border-right: 12px solid transparent; 
  border-top: 16px solid white;       // ← Triangle arrow
  margin: 22px 18px;                  // ← Manual positioning
}
```

**Next**: Check advanced panel for duplicate dropdown implementations

### Finding #2: Advanced Panel Dropdown Implementation ⚠️
**File**: `rti_advanced_panel.cc` 
**Lines**: 612-622

**CRITICAL DISCOVERY - CONFLICTING DROPDOWN STYLING**:
- Found ANOTHER QComboBox: `api_format` with **COMPLETELY DIFFERENT** styling
- Advanced panel dropdown uses SIMPLE styling vs. Main panel's COMPLEX subcontrol styling

**Conflicting Styles Comparison**:
**Main Panel Dropdowns** (complex):
```css
QComboBox::drop-down { subcontrol-position: top right; }
QComboBox::down-arrow { 
  border-left: 12px solid transparent;
  border-right: 12px solid transparent; 
  border-top: 16px solid white;
  margin: 22px 18px;
}
QComboBox QAbstractItemView { /* complex item view styling */ }
```

**Advanced Panel Dropdown** (simple):
```css
QComboBox {
  font-size: 28px; 
  padding: 12px; 
  background-color: #333; 
  color: white; 
  border: 2px solid #555; 
  border-radius: 8px;
}
/* NO subcontrol styling! */
```

**⚠️ POTENTIAL ROOT CAUSE**: Qt stylesheet inheritance conflict between complex and simple dropdown styles

### Finding #3: Function Duplication Detected ⚠️
**Files**: `rti_settings_panel.cc:43` vs `rti_advanced_panel.cc:18`

**DUPLICATION FOUND**: `safeStringToInt` function implemented twice with slight differences:
- **Settings Panel**: `static int safeStringToInt(const std::string& str, int defaultValue = 0)`  
- **Advanced Panel**: `static int safeStringToInt(const std::string& str, int defaultValue)`

**Impact**: Different default parameter handling could cause inconsistent behavior across panels

### Finding #4: Widget Hierarchy Analysis ✅
**Widget Structure**:
```
LongitudinalPanel (QStackedLayout main_layout)
├── RTISettingsPanel (QStackedWidget)
│   └── QScrollArea (horizontal disabled, vertical enabled) 
│       └── QWidget (maxWidth: 1300px)
│           ├── rti_source_combo (COMPLEX styling)
│           ├── rti_filter_combo (COMPLEX styling - shared)
│           └── rti_aggr_combo (COMPLEX styling - shared)
└── RTIAdvancedPanel 
    └── api_format (SIMPLE styling)
```

**Key Insight**: All problematic dropdowns are within the same scrollable widget hierarchy, suggesting layout or stylesheet cascading issues

## Suspected Root Causes

### PRIMARY: Qt Subcontrol Positioning Conflict
**Issue**: `subcontrol-position: top right` + manual arrow CSS conflicts with Qt's dropdown popup positioning algorithm
**Evidence**: Complex manual arrow styling with `margin: 22px 18px` forces Qt into incorrect geometry calculations  
**Impact**: Causes 90-degree rotation as Qt tries to fit dropdown within scroll area constraints

### SECONDARY: QScrollArea Layout Interference  
**Issue**: `setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff)` + `maxWidth(1300)` constraints interfere with popup positioning
**Evidence**: All affected dropdowns are within the constrained scroll widget
**Impact**: Qt popup geometry calculations fail within the restricted layout space

### CONTRIBUTING: Mixed Styling Paradigms
**Issue**: Complex subcontrol styling (main panel) conflicts with simple styling (advanced panel) 
**Evidence**: Stylesheet inheritance conflicts in QStackedLayout navigation
**Impact**: Cascading style conflicts compound positioning issues

## Remediation Plan

### Phase 1: IMMEDIATE FIX (Critical Priority)
1. **Simplify Main Panel Dropdown Styling**:
   - Remove complex `QComboBox::drop-down` and `QComboBox::down-arrow` styling
   - Use simple background/border styling like advanced panel
   - Test dropdown behavior after each change

2. **Eliminate Function Duplication**:
   - Extract `safeStringToInt` to shared utility header
   - Use consistent default parameter (recommend `= 0`)
   - Update both files to use shared function

### Phase 2: ARCHITECTURAL CLEANUP (High Priority)
3. **Standardize Dropdown Styling**:
   - Create reusable QComboBox style component
   - Apply consistent styling across all RTI panels
   - Remove manual subcontrol positioning

4. **Layout Optimization**:
   - Review QScrollArea constraints necessity
   - Test without `maxWidth(1300)` restriction
   - Consider alternative layout approaches

### Phase 3: LONG-TERM IMPROVEMENTS (Medium Priority)
5. **Code Quality Improvements**:
   - Move helper functions to shared utilities
   - Create RTI UI component library  
   - Implement consistent error handling patterns

6. **Testing & Validation**:
   - Add automated UI tests for dropdown interactions
   - Test across different screen sizes/orientations
   - Performance profiling of dropdown rendering