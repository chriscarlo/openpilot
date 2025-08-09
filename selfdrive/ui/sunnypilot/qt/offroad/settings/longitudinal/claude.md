# Claude Instructions for Longitudinal Settings UI

This directory contains the Qt-based offroad GUI panels for longitudinal control settings, including RTI (Realtime Traffic Intelligence), VTSC (Vision Turn Speed Control), DEC (Dynamic Engine Control), and other longitudinal features.

## NON-NEGOTIABLE UI CONSTRAINTS

**CRITICAL: Panel Width Constraints - MUST BE FOLLOWED**
- **Maximum Total Width: 1400px** (absolute maximum)
- **Horizontal scrolling is 100% FORBIDDEN**
- Vertical scrolling is acceptable when needed
- These constraints are derived from the VTSC anticipation distance panel dimensions:
  - Left margin: 15px
  - Graphic widget: 700px (max)
  - Spacing: 20px
  - Text widget: 650px (fixed)
  - Right margin: 15px
  - **Total: 1400px**

**ANY NEW MENU PANELS MUST RESPECT THESE CONSTRAINTS**

## Key Files:
- `rti_settings_panel.cc/.h` - Main RTI settings panel
- `rti_advanced_panel.cc/.h` - Advanced RTI configuration panel
- `rti_control.cc/.h` - RTI control widgets and enums
- `vtsc_settings_panel.cc/.h` - VTSC main settings panel
- `road_visualization_widget_v2.cc/.h` - Professional road visualization widget (reference implementation)
- `dec_controller.cc/.h` - Dynamic Engine Control settings
- `custom_acc_increment.cc/.h` - Custom ACC increment settings

## Panel Design Patterns:
- Use QStackedLayout for multi-screen panels
- Header with back button (200px width) and title
- Content area with proper margins (15px sides)
- Two-column layout: graphic widget (left) + text/controls (right)
- Professional styling with consistent color scheme
- Responsive sizing within width constraints

## RTI Panel Classes:
- `RTISettingsPanel` - Main settings interface
- `RTIAdvancedPanel` - Advanced configuration interface  
- `RTIControl` - Core control logic
- `RTIVisualizationWidget` - Visual components
- `RTIConfigDataPanel` - Configuration data management
- `RTIApiConfigPanel` - API configuration panel

## Dependencies:
- Qt widgets framework
- `selfdrive/ui/sunnypilot/qt/widgets/controls.h`
- `common/params.h` for parameter management
- Standard sunnypilot UI styling conventions