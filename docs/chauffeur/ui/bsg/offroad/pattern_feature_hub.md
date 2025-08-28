Title: Feature Hub Panel — Pattern Guide

Overview: Defines a generic multi‑feature hub menu (e.g., Cruise) that lists features as rows, each optionally linking to a dedicated settings sub‑panel. Use this as the baseline for any hub that aggregates related features.

Panel Composition
- Entry point: a Settings sidebar item (e.g., “Cruise”) that opens a hub view
- Container: `ScrollViewSP` wrapping a `ListWidget` of rows
- Row widget base: `AbstractControlSP` derivatives (e.g., `ParamControlSP`, or a custom gear+toggle row)
- Ordering: group related features top‑to‑bottom; place feature toggles without submenus near the top; rows with submenus (gear) grouped next; optional advanced features last

Row Layout Specs
- Title: 50px, weight ~450, left‑aligned; description hidden by default, 40px gray (#999999)
- Value label area: right‑aligned small secondary label when applicable
- Controls container on right: 40px gap between controls
- Toggle (`ToggleSP`): 150×100 (see offroad_settings_bsg.md)
- Settings gear button: circular 120×120 with 2px #696969 border, background #393939; pressed #4A4A4A; disabled dims
- Interaction: settings button enabled only when feature toggle is ON; emits `settingsClicked` to push the sub‑panel onto the Cruise stack

Sub‑Panel Navigation
- A hub screen hosts a `QStackedLayout` of:
  - Main hub list
  - Feature sub‑panels (one per feature)
- Tapping a feature’s settings button:
  - Saves current scroll position
  - Switches to the corresponding sub‑panel widget
- Sub‑panel top‑left `Back` button:
  - Restores Cruise list scroll position
  - Returns to main Cruise screen

Styling & Spacing
- Cruise list padding: inherited panel margins (typically 50px sides, 25px top/bottom around content)
- Row vertical height target: ~120px title line + controls; descriptions expand beneath when title tapped
- Inter‑row spacing: governed by the ListWidget item styles; maintain visual rhythm per Offroad BSG

Copy & Behavior Conventions
- Titles use Title Case, match feature names in navigation and sub‑panel titles
- Descriptions are actionable and safety‑oriented
- Do not expose settings when a feature is incompatible; show reason in description and disable the control

Code References
- Example hub (Cruise): `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal_panel.*`
- Example gear+toggle row: `.../longitudinal/rti_control.*` (use pattern generically)

Blueprint: New Feature Row (gear + toggle)
- Derive a row from `AbstractControlSP`
- Add:
  - `ToggleSP` (150×100) bound to a boolean Param (enable/disable)
  - Circular `QPushButton` (120×120) with gear glyph; enable state mirrors toggle
  - Signal `settingsClicked()` routed to the hub’s `QStackedLayout` to show the sub‑panel

Minimal Spec (example)
- id: feature_id
- title: Human title
- description: Short sentence
- param_enabled: Params key for toggle
- has_subpanel: true|false
- on_settings: route to sub‑panel widget if present

