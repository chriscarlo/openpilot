Title: Feature Menu — Pattern & Blueprint

Purpose: Define a generic pattern for a feature’s dedicated settings sub‑panel. Use this for any feature that needs a structured settings page with carousels, ranges, and toggles.

Top‑Level Pattern
- Sub‑panel header:
  - `Back` button (400×100) at top‑left
  - Title centered (feature title), 50px, weight 600, color #E4E4E4
  - One‑line description under title, 34px, #999999
- Panel margins: 50px left/right, 20px top/bottom; vertical spacing 30px

Sections (Cards)
Use section cards to group a feature’s settings. Common building blocks:
1) Choice Section (carousel)
   - Header 42px; `HorizontalCarousel` control with lensing + snap
   - Items: any string list; scale/opacity adapt automatically
   - Persist: int index or string; emit change on selection

2) Range Section(s)
   - One or more `Range Control` instances
   - Display units in value label; store canonical units in Params
   - Range control UI:
     - Title 42px; desc 32px #999999
     - Row: [− 100×100] [value 70px + status] [+ 100×100] [Reset 150×80]
     - Status: “(Default)” gray vs “(Modified)” amber #FFC107

3) Toggles Section
   - One or more boolean settings as label + right‑aligned `ToggleSP` (150×80–100)
   - Group under headers like “Alerts & Display”, “Safety”, “Advanced” as needed

Settings Card Appearance
- Each section wrapped in `QFrame` card with: `background-color: #292929; border-radius: 20px; padding: 25px;`
- Maintain 20px spacing between stacked Range controls

Behavior & Persistence
- Persist immediately to Params on interaction
- Define and document unit conversions (e.g., show mi, store m; show mph, store km/h) per feature needs
- Default vs Modified status reflects equality to default (use epsilon for floats)
- On `showEvent`, refresh from Params to reflect external changes

Optional Data Bootstrapping
- If a feature requires external credentials or endpoints, detect environment on show and seed Params accordingly. Keep this isolated and idempotent.

Microcopy Standards
- Titles: Title Case; unambiguous action/subject
- Descriptions: one short sentence; clarify directionality (“ahead on your route”) and timing (“after passing a threat”)
- Units in labels: always present (mi, mph)
- Avoid jargon; prefer concise, concrete phrasing

Code References
- Panel scaffolding examples: `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/*_settings_panel.*`
- Carousels: `horizontal_carousel.*`
- Gear+toggle row pattern: see `rti_control.*` as a neutral reference

Reusable Blueprint (summary)
- Main structure: Back, Title, Description → section cards
- Controls:
  - Carousel(s) for options with lensing & snap
  - Range controls with ±, units, status, reset
  - Toggles grouped by theme
- Persistence: define Params keys + units once; update on every interaction

Minimal Spec (example)
- id: feature_id
- title: Human title
- description: Short sentence
- sections:
  - type: choice | range | toggles
  - key(s): param keys (e.g., param_choice_index, param_speed_kmh)
  - units: display + canonical units as needed
  - defaults: default values

