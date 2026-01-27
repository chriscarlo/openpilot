**OSM + VTSC Integration Primer**

- **Goal:** Enrich VTSC with map-based geometry so we can plan further ahead than the model’s ~33-slice horizon, especially under occlusion or short-horizon vision, and avoid late braking on hidden/abrupt turns.

**Where Data Lives**
- **Prod path (`Paths.mapd_root()`):** PC `~/.comma/media/0/osm`, device `/data/media/0/osm`.
- **Layout:**
  - `offline/` – mapd’s offline tiles (Cap’n Proto files grouped on a 2° grid; files are packed capnp binary with way centerlines + attributes).
  - `db/` – optional DBs or region files for other pipelines (e.g., `osm.db`, `*.poly`).
  - `metadata/` – helper JSON like `us_states_bounding_boxes.json`, `nation_bounding_boxes.json`.

**Getting OSM Offline Tiles (mapd)**
- **UI trigger:** Offroad > OSM panel writes params `OSMDownloadLocations`/`OSMDownloadBounds`; `mapd` downloads tiles into `offline/` and reports `OSMDownloadProgress`.
- **Manual (what we did for CA):** Fetched 2° tiles from `https://map-data.pfeifer.dev/offline/<lat>/<lon>.tar.gz` for California bounds and extracted into `~/.comma/media/0/osm/offline/`. Total ~468 MB.
- **Binary on device:** `third_party/mapd_pfeiferj/mapd` (pfeiferj/openpilot-mapd release build).
- **Source (vendored):** `mapd_repo/openpilot-mapd/` (key files: `download.go`, `generate_offline.go`, `mapd.go`, `offline.capnp`, `params.go`).

**Offline Tile Format (Cap’n Proto)**
- **Schema (subset):**
  - `Offline { minLat, minLon, maxLat, maxLon, ways: List(Way), overlap }`
  - `Way { name, ref, maxSpeed, minLat, minLon, maxLat, maxLon, nodes: List(Coordinates), lanes, advisorySpeed, hazard, oneWay, maxSpeedForward, maxSpeedBackward }`
  - `Coordinates { latitude, longitude }`
- **Reading tips:** Files are written with `MarshalPacked()`; in Python use `pycapnp.Offline.read_packed(f)`. The original schema includes Go import directives; for Python, parse a stripped copy of the schema (same structs, no Go import line) or use precompiled bindings.

**Geometry → Physics (VTSC Recap)**
- **Curvature (k):** 1/m. From 3-point segments, `k = 4A / (a*b*c)` (A = triangle area; a,b,c = chord lengths in meters).
- **Radius (R):** `R = 1 / k`.
- **Safe lateral accel (a_lat):** VTSC sigmoid clamped to `[1.8, 3.12] m/s²` (see `_physics_based_lateral_acceleration` in `vision_turn_controller.py`).
- **Speed from curvature:** `v_safe = sqrt(a_lat / k)` (m/s). VTSC’s `curvature_to_speed` implements this and applies small tunings (e.g., low-speed bias, global factor).
- **Highway bypass:** Above ~65 mph, VTSC blends to pure physics and bypasses occlusion gating.
- **Hidden-turn support:** Early, jerk-limited decel under occlusion when a short-horizon physics deficit is provably large (HIDDEN_TURN_* constants).

**Concrete Ramp Example (Validation)**
- **Location:** 38.729107, -120.803181 (US‑50 westbound off‑ramp, Placerville area).
- **Found in tile:** `offline/38/-122/38.500000_-121.000000_38.750000_-120.750000`.
- **Way:** Unnamed one-way connector (likely `motorway_link`), ~173 m long, 8 nodes.
- **Endpoints:** (38.729606, -120.801668) → (38.729230, -120.803468), bearing ~255° (WSW).
- **Nearest node to point:** (38.729109, -120.803160), ~1.8 m.
- **Local curvature samples (from nodes near the point):**
  - k ≈ 0.0222 1/m → R ≈ 45 m → v ≈ sqrt(1.97/0.0222) ≈ 9.4 m/s (≈21 mph)
  - k ≈ 0.0155 1/m → R ≈ 65 m → v ≈ 11.3 m/s (≈25 mph)
  - k ≈ 0.0055 1/m → R ≈ 182 m → v ≈ 20.0 m/s (≈45 mph)
- **Interpretation:** Tight section wants ~20–22 mph by VTSC physics; milder segments open to mid‑20s; approach/exit ~45 mph. This matches typical ramp advisory speeds and is ideal to demonstrate lookahead benefits.

**Integration Skeleton (15–30 s Lookahead)**
- **Inputs:** GPS position (livePose), mapd offline dataset (or mapd daemon outputs), vehicle speed `v_ego`.
- **Step 1 – Acquire map geometry:**
  - Load the relevant offline tile for current lat/lon.
  - Identify current/next way(s): either by following mapd’s `FindWaysAroundLocation` path logic, or initially by selecting the nearest one-way highway/motorway(_link) whose bearing aligns with ego heading.
  - Extract centerline nodes for the chosen way(s).
- **Step 2 – Build a curvature horizon:**
  - Compute 3-point curvature along nodes; resample to distance increments (e.g., every 2–5 m) and smooth lightly.
  - Generate a 15–30 s forward window based on current `v_ego` and a simple decel model (for spacing).
  - Optionally include mainline/merge way if ramp+merge context is needed.
- **Step 3 – Convert to a speed horizon:**
  - For each forward sample, compute `v_i = curvature_to_speed(k_i)` using VTSC’s physics function.
  - Derive a target speed profile with jerk-limited decel toward the minimum v_i in the horizon (reuse VTSC’s adaptive decel/jerk gating).
- **Step 4 – Blend with VTSC:**
  - Use map-derived curvature beyond the model horizon or when occluded/conf < threshold.
  - Gate by speed (≤ ~65 mph), road class (motorway, trunk, *_link), one-way alignment, and reasonable distance coverage.
  - Take the planner’s min() of physics speeds: `min(cruise_setpoint, model_vtsc_speed, map_vtsc_speed)`.
  - Preserve highway bypass and hidden-turn early decel semantics.

**Trust/Confidence Heuristics**
- **Geometry confidence:** Prefer ways with `ref`/name consistency (e.g., US‑50), matching one‑way and heading.
- **Distance coverage:** Require a minimum available fraction (e.g., >60% of the short horizon) before using map lookahead.
- **Class gating:** Focus on highway/motorway(_link)/trunk for initial rollout; avoid residential/service clutter.
- **Fallbacks:** If map coverage is sparse/conflicting, fall back to model-only curvature and original VTSC behavior.

**Quick Reference: Relevant Files**
- **Mapd source:** `mapd_repo/openpilot-mapd/{download.go, generate_offline.go, mapd.go, offline.capnp, params.go}`.
- **Mapd binary:** `third_party/mapd_pfeiferj/mapd`.
- **VTSC:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` (physics and blending), `longitudinal_planner.py` (min() integration).
- **Paths:** `openpilot/system/hardware/hw.py::Paths.mapd_root()`.
- **UI OSM:** `selfdrive/ui/sunnypilot/qt/offroad/settings/osm_panel.*`.

**Minimal Python Sketch (Reading Offline + Nearest Way)**
- Load schema (stripped of Go import) and packed file: `offline = Offline.read_packed(f)`.
- Find nearest way to (lat,lon), compute local curvature, and convert to target speed via VTSC formula.
- Use haversine for meters and 3-point curvature; resample and smooth for a horizon.

**Next Steps / TODOs**
- Build a `map_vtsc_provider` that publishes a curvature/speed horizon topic (or injects directly into VTSC) with above gating.
- Unit/integration tests: include the Placerville ramp and other hidden-turn cases.
- Add simple visualization/logs (curvature, v_target, jerk/decel usage) to validate comfort/safety.

**Key Takeaway**
- This is feasible. The Placerville ramp shows OSM curvature maps directly to VTSC physics speeds (~20–25 mph on the tight segment). Blending a map-derived horizon into VTSC provides earlier, safer braking under occlusion without regressing highway behavior.
