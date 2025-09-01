# Roadmap: VTSC Physics in mapd (Advisory Turn Speeds)

## Background
- Symptom (onroad with MTSC enabled):
  - HUD intermittently shows "Communication Issue Between Processes"; overlays flicker
  - Encoder backpressure/timeout in logs (frame drops, dequeue poll timeouts)
  - Earlier pub/sub churn from MTSC was patched, but map-derived debug payloads still heavy
  - Engagement is reliable when MTSC is off; with MTSC on, stability regresses
- Root-cause candidates:
  - Per-tick geometry/horizon building and large debug vectors on the bus
  - Excess pub/sub or Params churn when MTSC restarts or publishes large arrays
  - Any Python-based tile reads are too slow; heavy lookups must be compiled
- Current compiled solution on device: pfeiferj/openpilot-mapd ("mapd")
  - Offline tiles under `/data/media/0/osm/offline/<lat_deg>/<lon_deg>/` (Cap’n Proto-packed)
  - Finds ways around current position, selects current + next ways
  - Computes curvatures and simple physics target velocities (sqrt(a_lat/k))
  - Writes small JSON arrays into Params (not the message bus):
    - `MapCurvatures`, `MapTargetVelocities`, `MapAdvisoryLimit`, `NextMapAdvisoryLimit`, `MapSpeedLimit`, `NextMapSpeedLimit`, `RoadName`
  - Single-file lookup per location update (fast Cap’n Proto decode in Go)
- What we want:
  - Full control of mapd’s physics to match VTSC “pure physics” (no occlusion), 1:1 parity
  - Keep runtime lightweight (compiled lookups + minimal IPC)
  - MTSC publishes only a small scalar to the bus to avoid churn

## Objectives
1) Vendor and compile our own mapd that computes VTSC physics turn speeds end-to-end.
2) Maintain a light runtime path: compiled lookup → small Param outputs → MTSC publishes a scalar.
3) Achieve exact parity with VTSC’s physics envelope (pure physics mode; no occlusion/visibility adjustments).

## Scope & Architecture
- mapd (Go, compiled binary)
  - Replace mapd’s velocity calculation with VTSC physics: sigmoid-based lateral-acceleration model + `v = sqrt(a_lat / |k|)`.
  - Embed our physics constants at build-time (defaults in code) with optional Params overrides for tuning.
  - Keep outputs in Params for compatibility; optionally compute a per-way scalar (min velocity on current way).
- MTSC (Python)
  - Add an MTSC “lite” consumer path: read mapd Params and publish only a scalar via `mapTurnSpeedControlSP`.
  - Disable publishing of large debug vectors by default; leave gating/confidence logic intact.
  - Preserve existing schema; keep bus pressure minimal.

Dataflow (lite path):
GPS/pose → mapd (compiled) → Params: MapTargetVelocities/MapAdvisoryLimit → MTSC-lite → `mapTurnSpeedControlSP` (scalar)

## Vendor Strategy (No Submodules)
- Source of truth: https://github.com/pfeiferj/openpilot-mapd (pin a known-good commit).
- Vendor location in this repo: `mapd_repo/openpilot-mapd/` (follows `*_repo/` convention).
- Binary install location on device (already wired): `third_party/mapd_pfeiferj/mapd`.
- Keep upstream LICENSE and version note in `mapd_repo/openpilot-mapd/`.
- Do not modify `.gitmodules`; fix imports/includes locally if needed.

## Physics: VTSC Pure Physics (No Occlusion)
- Inputs/units:
  - Curvature `k` in 1/m (positive magnitude). Speed `v` in m/s. Lateral acceleration `a_lat` in m/s².
- Mapping:
  1. Compute safe lateral acceleration with our tuned sigmoid:
     `a_lat(k) = clamp(A / (1 + exp(B * (k - C))) + D, a_min, a_max)`
     - Typical defaults (from VTSC):
       - `A = -1.1751`, `B = -2000.0`, `C = 0.004778`, `D = 3.144734`
       - `a_min = 1.8`, `a_max = 3.12`
  2. Convert curvature to speed via physics:
     `v_target = sqrt(a_lat(k) / max(|k|, 1e-7))`
  3. Optional low-speed bias (mph taper) and global factor may remain 1.0 in pure physics mode.
- Pseudocode (Go):
  - `safeLat := clamp(A/(1+exp(B*(k-C))) + D, aMin, aMax)`
  - `v := sqrt(safeLat / max(k, 1e-7))`
  - `v = min(v, vMaxDefault)`
- Notes:
  - No occlusion/visibility logic here; that lives only in VTSC vision path, not in mapd.
  - All curvature inputs must be in meters (no degrees or scaled pixel spaces).

## Curvature Calculation in mapd
- Use OSM way nodes (lat/lon) → project to meters (ECEF or local tangent plane).
- Sample along arc length; compute curvature per segment:
  - For points P(i-1), P(i), P(i+1), fit circle or use discrete curvature: `k = |(Δθ/Δs)|`.
  - Ensure consistent meter-scale distances (WGS84→ENU).
- Decimate/limit samples per current way (e.g., ≤60 points) before writing Params.

## Outputs & JSON Schemas (Params)
- `MapTargetVelocities` (JSON array): `[ {"lat": <deg>, "lon": <deg>, "velocity": <mps>}, ... ]`
  - Decimate to keep array small (<~60 entries). Velocity in m/s.
- `MapCurvatures` (JSON array): `[ {"lat": <deg>, "lon": <deg>, "curvature": <1/m>}, ... ]` (optional if perf OK)
- `MapAdvisoryLimit` (string/number): scalar m/s — min `v_target` on current way beyond a start distance.
- `NextMapAdvisoryLimit` (optional scalar m/s), `RoadName`, `MapSpeedLimit`, `NextMapSpeedLimit` remain as-is.
- For minimal bus load, MTSC-lite should publish only a scalar into `mapTurnSpeedControlSP`.

## MTSC “Lite” Consumer
- Param gate: `MTSCLiteMode` (default off).
- Behavior when enabled:
  - Read `MapTargetVelocities` or `MapAdvisoryLimit` from Params at 5–10 Hz.
  - Choose `startDistanceM` ≈ visible horizon + margin (configurable, e.g., 30–60 m).
  - Compute single `targetSpeedMps` (min over samples at/after startDistance) or use scalar advisory if present.
  - Publish only scalar fields via `mapTurnSpeedControlSP`:
    - `available`, `confidence`, `targetSpeedMps`, `startDistanceM`, `matchedWayId` (optional), minimal diag.
  - Disable debug vectors by default; keep gating for speed band, road class, coverage, etc.

## Build & Toolchain
- Target: linux/arm64, static, `CGO_ENABLED=0`.
- Go: 1.22.x (tested), modules pinned via `go.mod` vendoring.
- Cap’n Proto: `capnp` compiler 0.10.x and `capnproto2` Go bindings.
  - If upstream repo includes generated `offline.capnp.go`, regeneration is optional; otherwise pin exact versions.
- Local build (host cross-compile):
  - `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -ldflags="-s -w" -o build/mapd ./cmd/mapd`
  - For full static link on some systems: `-ldflags "-extldflags=-static -s -w"` (cgo must remain disabled or musl toolchain used).
- Earthly/Container build (optional): reuse upstream Earthfile if present; pin base images for reproducibility.

Embedded Defaults vs. Runtime Overrides
- Embed VTSC physics defaults in Go code (constants).
- Optional runtime overrides via Params (Mapd config block) to avoid rebuilds:
  - `MapdPhysicsA/B/C/D`, `MapdLatAccMin/Max`, `MapdMaxSpeedDefault`, `MapdSpeedIncreaseFactor`.
  - Parse once at startup; clamp to safe ranges; log applied values.

## Packaging & Wiring
- Binary install path (already expected by manager): `third_party/mapd_pfeiferj/mapd`.
- Process config (already present): `system/manager/process_config.py` spawns `mapd` via `MAPD_PATH`.
- `Paths.mapd_root()` points to `/data/media/0/osm`; keep tiles and cache there.
- Update `MapdVersion` Param on startup to track our build/version.

## TICI (comma3x) Suitability
- Platform: aarch64 Linux on-device; static Go binary is appropriate (no SNPE/NN runtime needed).
- Resources:
  - CPU: mapd runs per GNSS update (e.g., 5–10 Hz); Go decode + small math → typically <5% one big core.
  - Memory: <50 MB RSS typical; binary size ~6–10 MB static.
  - I/O: reads a single tile file per update; use buffered I/O; avoid excessive logging.
- Filesystem: tiles live at `/data/media/0/osm/offline/...` (existing convention). Ensure directory exists.
- Logging: default to WARN/ERROR; gate DEBUG with a Param to avoid logcat pressure.

## Tile Generation & Updates
- Generation (offline): use upstream `generate_offline.go` to convert PBF → Cap’n Proto tiles.
  - Pin OSM extract versions; document region coverage; gzip tiles if supported.
- On-device refresh: rely on existing `mapd_manager` to manage downloads and cleanups.
  - Keys used today: `OsmDownloadedDate`, `OsmDbUpdatesCheck`, `OSMDownloadLocations`, bounds, etc.

## Validation Plan
- Unit tests (Go): curvature→speed parity
  - Port a dozen VTSC test vectors; assert `|v_go - v_py| < 0.2 m/s` across representative curvatures.
  - Clamp/tuning boundaries honored: [1.8, 3.12] m/s², no negative/superluminal speeds.
- MTSC-lite tests (Python):
  - Start-distance handling; scalar selection; message publishing cadence.
  - No large debug arrays by default.
- Onroad checks:
  - Params updated at expected cadence; no bus pressure increase; CPU usage within budget.
  - Target reference routes (e.g., Placerville ramp): speeds ~20–25 mph on tight ramp segment; aligns with VTSC.
  - No “Communication Issue” alerts; encoder timeouts disappear.

## Risks & Mitigations
- Physics parity drift:
  - Port exact sigmoid + clamps; unit tests against Python VTSC.
  - Allow safe Param overrides with tight clamps (for emergencies).
- Message pressure regressions:
  - Keep MTSC-lite scalar only; guard debug vectors behind Param.
- Tile coverage/bounds:
  - mapd clips to one tile per update; publish `available=false` and safe defaults when no data.
- Integration fragility:
  - Keep binary path and process name stable; avoid cereal schema changes.

## Deliverables
- mapd binary (VTSC physics) statically built for linux/arm64.
- MTSC-lite mode in `sunnypilot/selfdrive/controls/mtsc/mtscd.py` to consume Params and publish a scalar.
- Unit tests (Go+Python) for physics parity and MTSC-lite behavior.
- Operator notes for toggling and validation onroad.

## Step-by-Step Execution Checklist
1. Vendor upstream mapd to `mapd_repo/openpilot-mapd/`; keep LICENSE; pin commit.
2. Port VTSC physics to Go (`math.go`): sigmoid + `v = sqrt(a/k)`; add tests.
3. Replace mapd `GetTargetVelocities` path with VTSC physics; clamp and decimate outputs.
4. Keep existing Params keys; add `MapAdvisoryLimit` scalar (min v on current way beyond startDistance).
5. Build static linux/arm64 binary; install to `third_party/mapd_pfeiferj/mapd`; set `MapdVersion`.
6. Add `MTSCLiteMode` flag; implement MTSC-lite scalar publish; disable debug vectors by default.
7. Validate onroad; monitor CPU/logcat; confirm no alerts/backpressure; compare reference routes.
8. Document tunables (Params) and finalize operator instructions.

## Appendix: Example Code Snippets

Go (physics mapping):

```go
func safeLatAccel(k float64, A, B, C, D, aMin, aMax float64) float64 {
  if k < 1e-8 { k = 1e-8 }
  val := A/(1.0 + math.Exp(B*(k-C))) + D
  if val < aMin { val = aMin }
  if val > aMax { val = aMax }
  return val
}

func curvatureToSpeed(k float64, A, B, C, D, aMin, aMax, vmax float64) float64 {
  a := safeLatAccel(k, A, B, C, D, aMin, aMax)
  v := math.Sqrt(a / math.Max(k, 1e-7))
  if v > vmax { v = vmax }
  if v < 0 { v = 0 }
  return v
}
```

JSON (Params example):

```json
{"lat": 38.72912, "lon": -120.79734, "velocity": 11.18}
```

## References
- Upstream: https://github.com/pfeiferj/openpilot-mapd
- Device tiles: `/data/media/0/osm/offline/`
- Manager wiring: `system/manager/process_config.py` (`mapd` and `mapd_manager`)
- MTSC schema: `cereal/custom.capnp` (`mapTurnSpeedControlSP`)
- VTSC physics (sigmoid + mapping): `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`

— End of Expanded Roadmap —

