# UI Style Guide

Use this document as the canonical source of UI/UX rules for driver-facing and operator-facing UI work in this repo.

## 0) Document metadata
- Product/repo: `openpilot` / Sunnypilot fork
- Version: `2026-03-14`
- Last updated: `2026-03-14`
- Owner(s): Codex working rules for this repo; update alongside substantive UI changes
- Linked specs:
  - `docs/chauffeur/vtsc/rally_copilot_tile_hud_spec.md`

## 1) Design contract
- Product adjectives (2-3): glanceable, stable, tactile
- Target users/personas: drivers glancing at the onroad HUD at speed; power users tuning chauffeur features offroad
- Platforms (web, mobile, desktop, tablet): embedded Qt in-car display first, offroad settings second
- Constraints (technical, legal/compliance, localization, brand):
  - Primary interaction budget is sub-second glance time while driving.
  - Motion must clarify state changes, never imply precise vehicle guidance.
  - Preserve the repo's existing Qt typography and general visual language unless the user explicitly asks for a redesign.

## 2) Principles
1. Clarity: every driver-facing overlay must read correctly in under one glance.
2. Hierarchy: the nearest actionable condition owns the strongest visual weight.
3. Stability: avoid continuous dancing motion; prefer discrete state transitions.
4. Performance: use paint primitives, cached geometry, and opacity/transform-style motion rather than layout-heavy churn.
5. Accessibility: maintain high contrast and keep legibility at varied cabin brightness.

## 3) Design tokens

### 3.1 Color tokens
- Surface/background tokens:
  - `hud.surface.scrim = rgba(7, 10, 14, 0.76)`
  - `hud.surface.tile = rgba(16, 20, 26, 0.90)`
  - `hud.surface.tileRaised = rgba(26, 32, 40, 0.94)`
  - `hud.surface.road = rgba(60, 67, 78, 0.88)`
- Text tokens:
  - `hud.text.primary = rgba(255, 255, 255, 0.96)`
  - `hud.text.secondary = rgba(212, 219, 229, 0.82)`
  - `hud.text.muted = rgba(170, 180, 192, 0.66)`
- Border/divider tokens:
  - `hud.border.subtle = rgba(255, 255, 255, 0.10)`
  - `hud.border.strong = rgba(255, 255, 255, 0.20)`
- Accent tokens:
  - `hud.accent.primary = rgba(245, 188, 92, 1.0)`
  - `hud.accent.primarySoft = rgba(245, 188, 92, 0.34)`
- Semantic tokens (success, warning, danger, info):
  - `hud.semantic.info = rgba(94, 188, 255, 1.0)`
  - `hud.semantic.warning = rgba(255, 183, 77, 1.0)`
  - `hud.semantic.danger = rgba(255, 103, 91, 1.0)`
  - `hud.semantic.success = rgba(95, 212, 150, 1.0)`

### 3.2 Typography tokens
- Font families:
  - Primary sans: `Inter`
  - Monospace/debug: `JetBrains Mono`
- Font size scale:
  - `caption = 18 px`
  - `label = 22 px`
  - `body = 26 px`
  - `title = 34 px`
  - `hero = 42 px`
- Font weight scale:
  - Regular, DemiBold, Bold
- Line-height scale:
  - Tight HUD numeric labels: `1.0`
  - General labels: `1.1`
- Letter-spacing:
  - Use default Inter metrics; do not add decorative tracking in the onroad HUD

### 3.3 Spacing tokens
- Base unit: `4 px`
- Spacing scale:
  - `4, 8, 12, 16, 20, 24, 32, 40`

### 3.4 Radius tokens
- Corner radius scale:
  - `8 px` micro
  - `16 px` standard
  - `24 px` large

### 3.5 Shadow tokens
- Elevation levels:
  - `shadow.soft = 0 14 32 rgba(0,0,0,0.24)`
  - `shadow.deep = 0 24 48 rgba(0,0,0,0.34)`

### 3.6 Border tokens
- Border widths:
  - `1 px`, `2 px`
- Border styles:
  - Solid only for HUD surfaces

### 3.7 Z-index tokens
- Layer stack (base/content/overlay/modal/toast/tooltip):
  - Onroad video
  - Lane/path rendering
  - HUD metrics
  - Advisory overlays
  - Alerts

### 3.8 Breakpoint tokens
- Breakpoint names and widths:
  - Embedded landscape only for the onroad HUD; no responsive web breakpoints apply
- Container widths:
  - Prefer region-relative sizing from the safe area instead of hard-coded display-wide constants

### 3.9 Motion tokens
- Duration scale:
  - `120 ms` micro feedback
  - `180 ms` stack shift
  - `260 ms` tile exit / drop-away
- Easing scale:
  - Standard enter/shift: ease-out cubic
  - Exit/drop: ease-in cubic
- Delay scale (if used):
  - `0 ms` default; avoid staged delays in driver-facing overlays

## 4) Typography rules
- Approved font families and fallbacks: use `InterFont(...)` in Qt HUD code; use repo-provided `JetBrainsMono` only for dense debugging views.
- Text styles (display, heading, title, body, label, caption):
  - Numeric speeds and primary state use `Bold`.
  - Secondary labels use `DemiBold` or `Regular`, never lighter.
- Usage rules by context (navigation, forms, data tables, long-form content):
  - Onroad overlays must stay terse: numbers, distances, and one- to two-word labels.
- Max line length and readability rules:
  - Avoid multi-line paragraphs in the onroad HUD.
- Do/Don't typography examples:
  - Do keep units and labels smaller than the primary number.
  - Don't mix more than two weights in one compact overlay.

## 5) Color system rules
- Surface hierarchy rules (page, panel, raised, overlay):
  - Driver-facing overlays use dark translucent surfaces over camera imagery.
  - Raised tiles may be slightly warmer or brighter than background scrims.
- Text-on-surface pairing rules:
  - Primary text stays near-white on dark surfaces.
  - Secondary metadata uses desaturated cool gray.
- Border and divider usage:
  - Use subtle borders to keep tile edges legible against the road feed.
- Accent usage rules (where accent is allowed/prohibited):
  - Accent amber is for the active or most immediate curve tile and selected route emphasis.
  - Do not use multiple competing accent hues in the same advisory widget.
- Curve preview styling:
  - VTSC rally co-pilot tiles may prioritize a beautiful, aggressively smoothed shape over centimeter-faithful geometry as long as turn direction, relative severity, and overall bend character remain truthful.
  - The rendered road shape should read as solid white with transparent fade-in/fade-out at the entry and exit.
  - A restrained outer glow or soft drop shadow is encouraged when it improves separation from the camera feed without making the widget feel neon or noisy.
- Semantic color usage for status and feedback:
  - Danger is reserved for warnings or critical system states, not routine turn previews.
- Contrast requirements (WCAG targets for normal/large text and UI components):
  - Large onroad labels should maintain at least 4.5:1 effective contrast against their immediate surface.

## 6) Spacing and layout rules
- Layout grid and column behavior:
  - Onroad overlays should align to safe-area regions, not arbitrary screen centers.
- Section and component spacing rhythm:
  - Use the 4 px scale; avoid one-off spacing values unless geometry demands it.
- Content width rules:
  - Keep advisory widgets within a bounded region so they never dominate the camera view.
- Negative space guidelines:
  - Let the camera feed breathe; empty space is preferable to filling every corner.
- Responsive layout behavior by breakpoint:
  - Scale from safe-area proportions; prioritize stable anchoring over aggressive resizing.

## 7) Component specifications

### Turn tile / stacked advisory card
- Anatomy (parts/slots):
  - Tile body
  - Curve glyph area
  - Distance/time metadata
  - Optional speed advisory
- Props and variants:
  - `active`, `queued`, `exiting`
  - `left`, `right`, `unknown`
  - `gentle`, `medium`, `tight`
- States (default, hover, focus, active, disabled, loading, error, success):
  - Driver-facing HUD uses `active`, `queued`, and `exiting`; hover/focus are not relevant.
- Keyboard behavior:
  - None onroad.
- Accessibility requirements (roles, labels, ARIA expectations):
  - N/A for Qt HUD; equivalents are legibility, contrast, and non-ambiguous iconography.
- Content rules (copy length, truncation, icon usage):
  - Metadata should fit in a single line.

### HUD labels and numeric readouts
- Primary numbers use bold, single-line layout.
- Secondary copy should stay within a 2-word budget when co-located with a graphic.

## 8) Interaction and motion rules
- When to animate vs not animate:
  - Animate discrete queue changes, fade-ins, and exits.
  - Do not continuously morph route geometry frame-to-frame when the change is merely a producer refresh.
- Curve geometry presentation:
  - Prefer stable, stylized ribbons with aggressive smoothing over twitchy fidelity.
  - Entry and exit fades should come from opacity gradients rather than abrupt clipping.
- Standard transition durations/easing by interaction type:
  - Stack reorder: `180 ms`, ease-out cubic.
  - Tile exit/drop: `260 ms`, ease-in cubic plus fade.
- Enter/exit rules for overlays and page transitions:
  - A new tile may fade/slide into its stacked slot.
  - The completed bottom tile may drop off-screen; remaining tiles shift downward without bounce.
- Loading/optimistic feedback patterns:
  - Prefer no widget over a placeholder if map data is absent.
- Reduced motion behavior:
  - Keep motion amplitude low enough that disabling continuous animation is sufficient; do not rely on motion alone to convey meaning.

## 9) Content and UX writing rules
- Voice and tone:
  - terse, directional, matter-of-fact
- Labels and action text conventions:
  - Prefer concrete labels like `35 mph` or `0.4 mi`, not promotional copy.
- Empty state writing rules:
  - Onroad overlays should omit empty copy entirely.
- Error message structure (what happened, why, how to recover):
  - Use existing alert systems for errors; advisory widgets should silently hide when data is invalid.
- Validation and helper text conventions:
  - Offroad settings docs may explain behavior, but onroad widgets should not.

## 10) Accessibility checklist
- Focus visibility and tab order:
  - Offroad surfaces must preserve Qt focus styling; onroad HUD is non-interactive.
- Keyboard-only completion for primary workflows:
  - Required for offroad settings, not onroad overlays.
- Semantic HTML and landmark usage:
  - N/A for Qt HUD.
- Screen reader expectations (names, roles, values, announcements):
  - N/A for Qt HUD.
- Color contrast verification process:
  - Verify major text and tile surfaces manually during UI changes.
- Touch target minimums and pointer alternatives:
  - Applies to offroad settings only.

## 11) Performance rules
- Scrolling and rendering constraints:
  - Onroad HUD rendering must avoid expensive per-frame allocations where practical.
- Virtualization thresholds for long lists/grids:
  - Not applicable to the onroad HUD.
- Asset loading strategy (critical CSS/fonts/images):
  - Reuse vector/path drawing and repo fonts; avoid bitmap assets for advisory glyphs.
- Skeleton/loading placeholders and fallback strategy:
  - Hide invalid advisory overlays rather than showing loading chrome.
- Interaction latency budget targets:
  - Paint-only HUD transitions should remain visually stable at display frame rate.

## 12) Do/Don't examples
- Do: prefer stable cards with one clear active tile.
- Don't: render a constantly writhing strip-map when the information should read as a queue.
- Do: use actual map-derived geometry, normalized into a stable tile frame.
- Don't: replace real curve shape with a generic left/right arrow.
- Do: let older/farther turns recede visually behind the active one.
- Don't: make distant turns visually compete with the bottom tile.

## 13) Change control
- This style guide is the source of truth for repo UI/UX decisions unless the user explicitly directs otherwise.
- Any intentional divergence should update this file first or in the same change.
