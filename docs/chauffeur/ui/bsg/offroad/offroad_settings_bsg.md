Title: Offroad Settings UI — Brand Style Guide (Chauffeur)

Purpose: Provide a generic design system and patterns for Offroad settings so future menus can be created uniformly with only the “features desired”.

Scope: Applies to all Offroad settings screens, including hubs (e.g., Cruise) and feature sub‑panels. This document defines tokens and shared components; concrete patterns live in separate pattern guides.

Foundations
- Color Palette:
  - Page background: #000000 (black)
  - Section card: #292929 (dark card), radius 20px, padding 25px
  - Control surface: #393939 (buttons/chips), pressed: #4A4A4A
  - Text primary: #FFFFFF; secondary: #E4E4E4; tertiary/body: #999999; disabled: #5C5C5C–#666666
  - Accent (selection/attention): amber #FFC107
  - Toggle on (track): #1E79E8; toggle off (track): #292929; disabled track: #2D2D2D
  - Borders (subtle): #696969 (2px on circular settings button)
- Typography (Qt pixel sizes):
  - Panel title: 50px, weight 600
  - Section header: 42px, weight 500
  - Control label (row title): 50px, weight 450
  - Body/description: 34–40px, color #999999 (prefer 40px; use 34px where space is tight)
  - Big numeric value: 70px, weight 500
  - Button glyphs (±): 60px, reset/utility: 35px
  - Carousel base item: 48px (scales 0.5–1.0 with lensing)
- Layout & Spacing:
  - Screen margins: 50px left/right, 20px top/bottom
  - Vertical spacing between blocks: 30px; inner gaps: 15–20px
  - Panel padding for description text: 40px left/right, 20px top/bottom
  - List items are arranged as rows: [title | value] + right‑aligned control(s)
- Shape & Radius:
  - Cards: 20px radius; circular controls: use true circles (e.g., 100–120px diameter)
  - Toggle track height: 80px with fully rounded ends

Core Components
- PanelBackButton (`Back`):
  - Size: 400×100; style id `#back_btn`
  - Background: #393939; radius 30px; font 50px; pressed #4A4A4A
  - Placement: top‑left of feature sub‑panels
- Toggle (`ToggleSP`):
  - Size: 150×100; track height 80px; on color #1E79E8; off color #292929
  - Knob: 68×68 circle; respects enabled/disabled dimming
  - Interaction: snap; state persisted via Params (bool)
- Settings Gear Button (feature rows):
  - Circular 120×120 (radius 60px), 2px border #696969, background #393939
  - Glyph: “⚙” at ~63px; pressed #4A4A4A; disabled dims (track #2D2D2D, border #444444)
  - Placement: left of toggle on feature rows (RTI/VTSC/DEC)
- Section Card (`QFrame`):
  - Style: `background-color: #292929; border-radius: 20px; padding: 25px;`
  - Used to group related controls (ranges, carousels, alerts)
- Range Control (value with ± and Reset):
  - Label block: title 42px, desc 32px (#999999)
  - Value: 70px, centered; status under value: “(Default)” gray or “(Modified)” amber #FFC107
  - Buttons: ± 100×100; Reset 150×80; all #393939, pressed #4A4A4A
  - Display units explicit (e.g., “mi”, “mph”)
- Horizontal Carousel:
  - Height ~120px; dynamic item spacing based on text width
  - Lensing: items scale 0.5–1.0 with opacity 0.3–1.0 toward center
  - Edge fades: 100px left/right linear gradient overlay
  - Selection indicator: centered 60×3px line with amber gradient
  - Touch physics: inertial scroll with stronger snap magnetism, elastic bounds

Pattern Index
- Feature Hub Panel (multi‑feature menu with submenus): see `pattern_feature_hub.md`
- Feature Menu (specific feature sub‑panel): see `pattern_feature_menu.md`

Text & Content
- Titles: Title Case, concrete feature names (avoid slang)
- Descriptions: concise, one sentence where possible; present tense; clarify safety constraints
- Units: always show in label/value; keep integer steps for mph; show two decimals for miles where precision matters
- Status labels: use “(Default)” gray when at default; “(Modified)” amber when user‑changed

Interaction Rules
- Disable settings gear when feature toggle is OFF
- Show/hide dependent rows based on primary toggle (e.g., Vibe personalities)
- Persist changes immediately via Params on every increment/decrement/selection
- Touch targets: ≥100×100 for primary controls

Implementation References (code)
- Settings shell and panel plumbing: `selfdrive/ui/sunnypilot/qt/offroad/settings/settings.cc`
- Hubs and stacks (example: Cruise): `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal_panel.*`
- Base controls and gear+toggle rows: `selfdrive/ui/sunnypilot/qt/widgets/controls.*`, `toggle.*`
- Carousel pattern: `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/horizontal_carousel.*`

Authoring Checklist (for any new Offroad setting)
- Use `AbstractControlSP` (or derived) for rows; prefer toggle + optional settings gear
- Use section cards for grouped controls on sub‑panels; follow spacing and type scale
- Apply colors/radii/pressed/disabled states as defined
- Define Params keys, defaults, units; show “Default/Modified” status when applicable
- Provide a `Back` affordance with `PanelBackButton`

