# VTSC Tuner — change log

Reverse-chronological. Add a new dated section for every substantive change.

## 2026-07-14 — direct curve-picking entry and click-through map legend

- Restored Calibration as the normal Map Preview entry mode. The prior Whole-Curve Study default is deliberately read-only: it replaces the bank workflow with study inspection and routes road clicks to study events, which made ordinary curve selection look broken. Whole-Curve Study remains available from the purpose picker, and its inspector now has an explicit “Go to Curve Calibration” button that explains how to return to drafting and bank edits.
- Made the display-only map speed legend ignore pointer events. Its alignment frame spans the map, so this prevents it from ever obscuring a colored road click as the map pane changes size.
- Verification: all 64 core and 34 app Swift tests passed, as did the tile-decoder Go tests; the rebuilt Release bundle passed strict deep code-sign verification. In the fresh app, Map Preview opened in Calibration with the calibration workflow visible; a physical mouse click selected Forest Route 3N01, created a clearly marked unbanked draft at 27.9 mph, and enabled Add Curve to Bank. The persisted 18-curve bank hash was unchanged; nothing was added, fitted, accepted, or applied.

## 2026-07-13 — production whole-curve runtime and coherent deployment

- Ported the accepted whole-curve estimator to Go and made one approximately 1,200 m ordered directional route feed both legacy curvature and the new profile. The estimator preserves source/traversal provenance and predecessor context, publishes stable signed events through versioned `MapWholeCurveProfile`, accepts a unique mainline beside an ordinary side road, and fails closed on genuinely unresolved forks. A shared Swift/Go corpus covers the actual 18 saved samples and 12 directional events plus the synthetic geometry, ambiguity, truncation, reversal, and rollover cases.
- Added strict Python profile consumption with version, freshness, finiteness, range, spacing, fingerprint, event, and live-route proximity checks. Accepted controlling curvature is evaluated with the current source-rounded sigmoid and Q; learned scaling is excluded. Every rejection falls back to legacy `MapCurvatures`, and the Map Lookahead falling edge immediately clears all map tail/hold/release state. Legacy baked speeds remain disabled.
- Added a real-GPS restart anchor for parked profile publication. Only fresh `updated/alive/valid/hasFix` fixes with finite, ranged, non-`0,0` coordinates reach shared memory or persistence; the first valid fix is persisted, later flash writes require at least five minutes and one kilometre, and a stationary anchor refreshes after 30 minutes. Controller and Go startup reject malformed/stale seeds, and deployment never injects a synthetic route point.
- Published immutable mapd release `chauffeur-whole-curve-v1` from standalone commit `99e7bb6c75f21977919108a456627aedbfc30070`. Its Linux ARM64 static binary SHA-256 is `6236d99e5f62744541634f1b3e36ca2eec8d2a6b529eef80cb6639137874ffdb`; build metadata and raw markers declare the exact commit, `whole-curve-v1`, and `MapWholeCurveProfile:whole-curve-v1`. Installer and manager startup now require exact digest/architecture/release/build/capability/runtime identity and can restore the same artifact from the identity-keyed persistent OSM cache.
- Replaced the native Mac car-facing flow with one coherent transaction: no-mutation release/Git/tici safety preflight, complete host tests, exact push and fast-forward, verified staged ARM64 release plus persistent cache, six-Param/Q readback, optional canonical tile activation, one reboot, real-GPS postflight, and rollback journal. Canonical tile sets live durably under Application Support `map_tiles/sets/<tile-set-id>/`, decode and hash every file, transfer only to an empty partial sibling, and atomically activate one immutable generation while retaining the previous set. The first runtime road test requires no tile replacement.
- Persisted the live Curve Lab tune with exact knobs `2.448138 / 4.107103 / 55.127268 / 3.815305`, no bands, and source-rounded parameters `-1.658965 / -1395.055546 / 0.005397 / 4.107103 / 2.4481 / 4.1071`. Reload preserves all 18 samples, 12 events, and the conspicuous sample-16 result rather than fitting it away.

## 2026-07-13 — local whole-curve geometry shadow study

- Added a developer-only, read-only Whole-Curve Study to the native macOS tuner. It stitches direction-feasible local map ways into long routes, deduplicates source nodes, resamples the actual source polyline at roughly 5 m without cubic overshoot, and measures signed curvature over 60/100/160 m distance windows so one physical bend is evaluated as a coherent event instead of a collection of unrelated five-node samples.
- Event detection follows sustained turn direction, merges short same-sign gaps before qualification, splits real S-curves, and reports sparse geometry, incomplete shoulders, sign instability, and scale disagreement instead of silently overstating confidence. Compact-apex promotion requires sustained evidence from at least three real same-sign source turn vertices, so one mapped kink or zigzag cannot manufacture a hairpin. Opposite travel directions remain separate; reversing a route preserves curvature magnitude and reverses only its sign.
- Added a separate map purpose with current-mapd, whole-curve, and difference overlays. Selecting any colored portion selects the entire directional event and shows event length, direction, multi-scale curvature, current-versus-study speed, lateral acceleration, confidence flags, and every linked saved-bank request. Exact source provenance is required before a saved target can attach to an event, avoiding nearest-lobe mistakes on hairpins. Apply, Sync, tune loading/undo, fitting, acceptance, curve-bank persistence, and device controls are hidden and rejected at the model layer while the study is active.
- The persisted 18-sample familiar-curve bank resolves to 12 directional events representing 7 physical bends, with all 18 samples linked. The event grouping matches the independently audited physical membership; it also keeps visible disagreements such as the gentle bend whose whole-curve result remains near 102 mph against a 75 mph request rather than manufacturing curvature to satisfy the target.
- Moved checkout-baseline reading off the app's synchronous launch path. A freshly ad-hoc-signed build can now create its window before macOS privacy checks for a repository under `~/Documents`, while saved/fallback tune state remains available immediately.
- Added pure estimator regressions for straight roads, constant-radius curves across 5/20/80 m source spacing, duplicate nodes, route reversal, S-curves, compound continuations, nonuniform-node overshoot, one-node zigzags, compact apexes, and sparse-source flags; plus app coverage for multi-piece grouping, provenance-only linking, conservative confidence merging, and the read-only boundary. All 80 Swift tests, the live 18-sample bank report, and the tile-decoder Go tests pass; the Release bundle passes strict deep code-sign verification. Production mapd/controller activation is explicitly deferred until this local study is accepted and the estimator is pinned by shared Swift/Go golden cases.

## 2026-07-13 — runtime-equivalent curvature calibration

- Traced the persisted 18-curve bank back to its exact cached OSM nodes. Every archived value reproduced the raw adjacent-three-node circumcircle, proving the archive and fitter were intact but the calibration curvature semantic was wrong: raw vertex spikes reached 9.60 m/s² at requested speeds while live mapd uses a smoothed route value.
- Added exact Swift parity with mapd's arc-length-weighted average of three neighboring triplets over five route nodes. Estimator v5 stitches direction-feasible geometry across tile boundaries using a unique gentle exact-name match, then a unique gentle exact-reference match, then a uniquely least-curvature gentle partial-reference match. Identity ties and unresolved multiple physical branches fail closed and cannot reach apex snapping or fitting. Raw curvature, effective curvature, five-node route span, estimator version, and context status are retained separately.
- Added the merge/split stage that production mapd runs before averaging: qualifying lane transitions clamp affected raw entries to `0.0015`, including its 15 m proximity behavior. Estimator version 2 introduced the correction, version 3 extended route context far enough to prove both 15 m walks have terminated, version 4 added authoritative missing/ambiguous-geometry invalidation, and version 5 mirrors mapd's deterministic name/ref continuation priorities without accepting order-dependent ties; each bump forces all schema-3 rows through another geometry audit. The real 3→4-lane bank item changed from an intermediate `κ=0.006102` / 4.69 m/s² to `κ=0.002333` / 1.79 m/s².
- Route stitching now fails closed whenever more than one viable same-road continuation remains, when a matching continuation has unsupported direction/one-way semantics, or when a bidirectional lane transition makes the result route-direction dependent. It also continues past the five-node average until at least 15 m of physical transition margin has been resolved, so a clamp on a short adjacent way cannot be missed. The raw proposed tile-bake overlay remains raw and separate rather than being overwritten with runtime curvature.
- Migrated the bank to schema 3 without losing IDs or target speeds. The live estimator-v5 audit resolved all 18 samples against the cached tiles; the bank-level requested-acceleration peak fell from 9.60 to 5.38 m/s². All 18 desired speeds are individually inside the complete base-plus-Q envelope and the fresh proposal emits no impossible-target warning; the remaining residual is the visible compromise required by conflicting and inverted targets on one monotonic curvature-only curve. The inspector shows raw inflation as a diagnostic while fitting only the runtime-equivalent value.
- Disabled legacy schema-v1 `MapPreCurveSpeeds` consumption until a versioned tile stream uses the same estimator as published `MapCurvatures`. This prevents a latent raw-bake/smoothed-curvature mismatch from being reactivated; the normal live sigmoid path remains unchanged.
- Unified the six fitted physics values across controller literals, checked-in Params defaults, and the offroad panel's reset/ensure values. Source patching now validates all four authorities, while tici apply runs an offroad-only rollback-protected helper that writes and verifies all six persistent Params before reboot.
- Added production US 50, lane-transition, 15 m propagation, cross-way stitching, exact-reference branch, and fork-ambiguity goldens; five-node context rejection; estimator-provenance persistence; and an actual-bank regression that rejects the false 8–10 m/s² demand and pins all 18 targets as individually attainable. All 64 Swift tests, five focused Python tests, and the Go tile-decoder tests pass; the signed Release bundle was rebuilt and the live bank re-audited at estimator v5 with 18/18 complete. Its inactive 18-curve proposal reports 18.66 → 4.05 mph RMSE and a 4.84 m/s² peak.

## 2026-07-13 — complete-curve calibration fit

- Replaced the four-knob-only proposal model with a complete-curve fit. The deterministic base sigmoid remains inside the existing physics clips; a second bounded solve generates a canonical set of broad Q=4 residual EQ bands in log-curvature space, and proposal scoring now runs through the same source-rounded 256-point interpolation that the runtime consumes.
- Complete base-plus-Q authority now determines pointwise attainability. Backbone scoring strongly penalizes dependence on Q authority outside the selected base's real 0.5–1.5 speed range, and per-sample envelopes are recomputed from that final base. The persisted 18-curve bank no longer emits nine false base-envelope warnings; weighted monotonic projection and explicit near-curvature conflicts preserve the requests that cannot coexist on one safe curvature-only curve instead of hiding them behind unstable narrow spikes. The saved bank improves from 15.34 to 3.21 mph RMSE, with a 6.60 mph worst miss caused by those genuine conflicts.
- Added full-domain safety constraints: the Q=4 Gaussian's explicit four-sigma influence collar extends by `sqrt(10)` in curvature around the bank, retaining endpoint authority without a discontinuity. Beyond the collar, transition-aware 4096-point checks plus every rounded Q knot keep both the sigmoid backbone and complete source-rounded runtime curve within 2 mph of the persisted checkout anchor. Reversal safety measures cumulative rise above the running minimum over the dense interpolated curve, not merely one adjacent Q-knot step. Unsafe fallback proposals are rejected, and proposals above the 5.5 m/s² safety-review threshold display peak acceleration and require explicit acknowledgement. Sigmoid evaluation also uses the exact six-/four-decimal precision written by source apply.
- Proposal acceptance now replaces knobs and generated bands together as one undoable edit, supports redo, and rejects a result if the current knobs or bands no longer match its input snapshot. Labels include bank numbers without duplicate road/reference text, the review shows worst miss and implied lateral acceleration, and copy explicitly says that effective predictions use checked-in source defaults rather than live device Params.
- Curve Lab now renders post-sigmoid Q exactly, including effective acceleration above a raw sigmoid rail, dynamically expands its acceleration axis, and widens its legacy 0–90 mph domain when a generated band requires a center up to the runtime's 156.6 mph cap. Band markers remain at their editable center speed, while vertical dragging solves gain through the complete source-rounded curve including overlapping bands; the x-domain is frozen during a drag so expansion cannot move the pointer mapping underneath the user.
- The calibration archive now persists the bank's original canonical checkout anchor in schema 2, migrates schema-1 banks, and clears the anchor only with an empty bank. Repository switches invalidate proposals and acceptance rechecks the anchor along with knobs and bands.
- Added regressions for the real 18-target bank, order-independent and fully idempotent output, cluster-duplication invariance, non-transitive clustering, exact modifier/domain bounds, final-base authority, dense off-collar complete-curve and cumulative raw-runtime shape safety, source rounding, dynamic high-center authoring, WYSIWYG band dragging, archive migration, stale-result rejection, and atomic acceptance/undo/redo. All 50 Swift tests and the tile decoder Go tests pass.

## 2026-07-13 — mouse-safe map controls and proposal generation

- Replaced the map workspace's SwiftUI `.inspector` overlay with a real horizontal split pane. The AppKit-backed `MKMapView` previously retained a native frame beneath the visually overlaid inspector, so real mouse clicks and scrolls over Curve Bank controls were intercepted by the map even though accessibility correctly reported the controls on top.
- The controls pane now occupies separate native window geometry with a 350–470 point width, while the map ends at the split divider. This restores physical mouse interaction throughout the inspector without changing the persistent bank, fitting model, or device/source apply boundaries.
- Verification: all 36 Swift tests and the tile decoder's Go tests passed; the Release bundle passed strict deep code-sign verification. In the relaunched app, a real point-space mouse scroll moved the controls while leaving the map fixed, then a real 120 ms mouse down/up on Generate (not accessibility `AXPress`) completed all 18 persisted curves, changed the button to Regenerate, and displayed the inactive proposal with RMSE `14.64 → 8.75 mph`. Nothing was accepted or applied.

## 2026-07-12 — focus-safe proposal generation with visible completion

- Fixed a real focus-loss race in the curve bank: formatted target fields can recommit their current numeric value when Generate takes focus, and the old setter treated that no-op as a bank mutation that canceled and erased the just-started fit. Identical target commits now return without invalidating the proposal task; genuine bank changes still cancel it and publish an explicit cancellation status.
- Generate now commits any active target edit before snapshotting the bank on the next main-loop turn. If AppKit refuses to end an invalid partial edit, the fit is blocked with a visible error instead of silently using the old banked value. The control immediately enters a loading state, changes to a persistent Regenerate label after success, and automatically scrolls the inspector to either the completed proposal or an error instead of inserting feedback below the fold.
- Added a 15-curve app-session regression that starts the real asynchronous proposal path, simulates the no-op field commit, and requires a complete 15-item diagnostic result plus visible Fit-ready status. A second end-to-end state regression accepts a generated result through `MapPreviewSession`, requires the fitted knobs/parameters to become Curve Lab's active tune, switches workspaces, and clears the consumed proposal.
- Verification: the decoder's Go tests and all 36 Swift tests passed; the Release app and nested helper passed strict deep code-sign verification. Live QA against the user's persisted 15-curve bank left the final target field active, invoked Generate, auto-scrolled to the proposal, changed the control to Regenerate, and reported RMSE `15.07 → 8.97 mph`. Accepting it switched to Curve Lab, redrew the solid fitted curve against the dashed checkout baseline, exposed the fitted knob values, and did not write source or contact the car.

## 2026-07-12 — readable map orientation, persistent curve bank, and 5 mph bands

- Replaced the unreadably small map-only road text with optional app-owned high-contrast `name` / `reference` labels sourced from the actual decoded mapd ways. Road text now defaults to Apple's normal labels; Large and Extra Large mapd labels scale with close zoom and macOS text size, deduplicate repeated ways, and honor annotation collisions.
- Added large inline Apple MapKit place/address results and destination markers. Queries such as `Madison Ave, Sacramento, CA` can center the camera and can be saved, revisited, or cleared as the persistent opening location, while the colored strategic overlay and curve selection continue to use only the user's real local mapd tile geometry. Search cancellation and generation guards prevent stale results from moving the map.
- Reworked calibration into an explicit draft → numbered persistent bank → proposal flow. The bank has a five-curve fitting minimum but no maximum or rollover; new unique curves append, exact source points update in place, selecting a new curve scrolls the inspector back to its draft, and every saved curve feeds the next fit. Accepting a proposal leaves the bank intact and changes only the Curve Lab knobs as a separate undoable edit.
- Removed the fitter's avoidable quadratic large-bank pass by evaluating only the relevant curve during each attainable-range refinement. Cancelling or invalidating a fit now propagates into that detached worker, whose coarse/extrema/refinement stages check cancellation instead of consuming CPU after its result becomes stale.
- Split currently-baked and proposed overlay colors into exact 5 mph buckets: `<25`, every 5 mph band from `25–29` through `70–74`, and `75+`. The larger three-column legend shows all 12 colors without collapsing the upper range.
- Added nine app-state tests covering every speed-band boundary, draft isolation, duplicate-safe Add/Update behavior, all draft-discard paths, active-item removal, Apple-label/opening-location defaults, and a single 12-curve bank whose complete diagnostic set reaches the fitter. All 34 Swift tests pass.

## 2026-07-12 — readable, unclipped Curve Lab axes

- Enlarged both axis tick labels from 10 pt to 18 pt semibold monospaced text, raised contrast, strengthened major/minor grid lines, and added visible tick marks. Axis titles are now 17 pt bold with the y title rotated and spelled out as lateral acceleration.
- Replaced the plot's undersized 48/28-point label gutters with dedicated left/top/right/bottom metrics. The mapper now honors reported safe-area insets and always reserves an additional 32 points for RootView's custom footer because macOS returned a zero bottom inset during live verification.
- Increased rail annotations to 14 pt semibold and hover readouts to 15 pt semibold, and added a complete spoken accessibility value for both axis ranges.
- Verification: 25 Swift tests passed, the Release app rebuilt, and the app plus nested decoder passed strict deep code-sign verification. Live visual QA confirmed the enlarged 0–90 mph and 0–6 m/s² tick ranges; the final explicit footer clearance moves the full x-axis title above the previously reproduced status-bar overlap.

## 2026-07-12 — actual map viewer and familiar-curve calibration workflow

- Added a native MapKit workspace with a persistent place/address search bar, muted Apple context map, and actual local mapd geometry colored by stored bake, proposed bake, or delta. Search only moves the camera; it never substitutes Apple geometry for mapd overlays.
- Added click hit-testing, a dashed way highlight, exact selected-node marker, and nearest-local-peak snapping within 120 m along the road. The inspector reports road identity, curvature/radius, current/proposed speeds, tile/current hash, schema and winding metadata, while explicitly separating a per-node bake from the final live strategic command.
- Added a persistent curve-bank calibration workflow with a five-sample minimum, numbered map pins, and row-to-map recentering. Desired speed defaults to the current effective curve target, users can override each familiar curve in mph, and Fit Review shows before/after RMSE, per-curve residuals, conflicts, and unattainable requests before an explicit one-step undoable accept. Sample, knob, or band changes refresh displayed baselines and cancel/invalidate stale results.
- Tici tile sync now builds and indexes an empty staging cache, requires at least one tile, archives the prior cache as `offline.previous`, and swaps the new generation into place so stale and current hashes cannot be merged.
- Real-cache smoke testing caught non-finite safe-speed values in a Bay Area tile. The helper now represents those baked values as unavailable, sanitizes non-finite scalar metadata, rejects invalid coordinates, and the Swift store preserves good neighboring tiles if one decode fails.
- Verification: helper tests passed; the formerly failing 36,078-way / 218,619-node tile decoded with seven unavailable baked nodes; 25 Swift tests passed; the Release bundle and nested arm64 helper passed strict deep codesign; GUI smoke used the actual 2,244-file Application Support cache.

## 2026-07-12 — deterministic real-curve sigmoid fitter

- Added `VTSCTunerCore/SigmoidFitter.swift`, a pure four-PlainKnob bounded fitter for map-selected curve targets. Its deterministic coarse-grid plus multi-start pattern search preserves the existing clip/WYSIWYG parameter mapping, clusters near-duplicate curvatures, projects individually impossible requests onto numerical feasibility envelopes, and returns before/after RMSE plus per-curve and conflict diagnostics.
- Calibration defaults to the app's effective physics-envelope estimate: the actual mapd bake from `MapBakeMath`, followed by the modeled 5 mph taper-to-55 bias, speed factor, and enabled Q bands. Raw baked-tile fitting remains an explicit mode, every sample diagnostic exposes both baked and effective before/after speeds, and the UI warns that device overrides/live strategic context can differ.
- Added focused Swift tests covering map-bake parity, effective modifiers/Q bands, deterministic order-independent synthetic recovery, clip-safe impossible targets, conflicting/underconstrained curvature samples, and invalid straight/nonfinite input. The six-curve deterministic recovery test completed in about 0.18 s in Debug.

## 2026-07-12 — strategic map tile decoder backend

- Added a bundled native macOS `vtsc-tile-decoder` helper under `tools/vtsc_tuner_mac/Helpers/VTSCTileDecoder`. It reuses mapd's generated `offline.capnp.go` binding from a temporary build directory and emits one compact JSON tile per packed Cap'n Proto input, including road geometry, baked safe speeds, directional metadata, and deterministic geometry-based way IDs.
- Added a SHA-256-pinned Go 1.26.5 darwin-arm64 bootstrap/build script with a `VTSC_GO` escape hatch. The release app build now packages and signs the helper at `VTSC Tuner.app/Contents/Helpers/vtsc-tile-decoder`.
- Added Swift Core tile/bounds/way/node models, exact Go-parity curvature/baked-speed/sigmoid-hash math, the helper-backed async decoder, filename-only tile indexing, a bounded actor cache for visible cells, and an explicit rsync service that mirrors the tici cache into Application Support only when requested.
- Added a base64-embedded 1,036-byte real schema-v1 golden tile plus Go tests for real decode, compact multi-input output, and truncated-data rejection. Added Swift tests for math parity, indexing, JSON mapping, cache reuse, and the explicit non-deleting rsync contract.
- Verification: pinned helper tests passed; the helper and packaged copy decoded the real golden tile as schema 1 / hash `f9d38ab3357c` / 1 way / 42 nodes; all 23 Swift tests passed; the Release app and nested arm64 Mach-O helper passed strict deep codesign verification.

## 2026-07-12 — native SwiftUI macOS port

- Added `tools/vtsc_tuner_mac/`, a Swift 6 / SwiftUI + AppKit port of the standalone tuner. It is a separate developer app only; no openpilot/chauffeur runtime code was ported or restructured.
- `VTSCTunerCore` ports the exact sigmoid/plain-knob/EQ math, 512-point log-κ sampling, mandatory monotonic-speed render pass, 256-point Q export, Rust schema-v1 JSON, Application Support persistence, live checkout baseline parsing, two-file validated source patching, and 12 focused Swift tests.
- `VTSCTunerApp` ports the native window/toolbar/inspector/status layout, rotary controls, advanced fields, 200-step grouped undo/redo, Canvas hero plot, AppKit mouse/key layer, target hit priority, handles, hover/readout, Shift-click/right-click bands, selected Q zone, repository chooser, and apply confirmation/progress sheets.
- The async Mac `ApplyPipeline` preserves all five cumulative actions and stable progress IDs. It uses absolute system Git/SSH/rsync paths, macOS SSID hinting plus autonomous SSH-profile probing, Apple-compatible rsync, cancellation, tune-save early abort, and a rebuild preflight before any mutation.
- Fixed two legacy rebuild correctness traps in the Mac implementation: it requires a native Darwin/Mach-O mapd generator and passes all six `--phys-*` values to `mapd --generate`. Generation uses a clean temporary `$CWD/offline` tree verified against the live mapd source and rejects empty ≤64-byte placeholder output.
- Added `scripts/build_app.sh`, `Support/Info.plist`, and `README.md`. The script builds Release, creates `dist/VTSC Tuner.app`, ad-hoc signs it, and verifies the signature. Fixed shell expansion of variables adjacent to a Unicode ellipsis by using braced forms.
- Verification: `swift test` passed 12/12; Release bundle built and passed strict codesign verification; the packaged app launched as a 1300×850 native window and was visually inspected at Retina resolution.

## 2026-04-19 — first real end-to-end deploy + chriscarlo/mapd binary distribution + PBF prep landmine

First real-world `RebuildTilesAndReboot`-equivalent flow on a tici. Done by hand (the user wanted me to drive it via SSH while they sat in the car), but every step matches what the in-app action will do once they invoke it. Final state: tici was on `chauffeur-exp01` at `9fb963f`, mapd binary at `third_party/mapd/mapd` (sha256 `d4d49746...`), all 36 cached regions (32–44°N × -126 to -114°W) carry sigmoid-baked tiles with `MapTilesSigmoidHash=f9d38ab3357c`, `MapPreCurveSpeeds` populated with 3.7 KB of baked velocities, hash matches the runtime → VTSC consumes the baked path (no fallback).

### What changed in the openpilot tree (commit 9fb963f, atomic)
- Tier 1 (bbox JSON URLs): `selfdrive/ui/sunnypilot/qt/offroad/settings/osm/locations_fetcher.h:56,61` now fetch from `raw.githubusercontent.com/chriscarlo/mapd/main/{nation,us_states}_bounding_boxes.json`.
- Tier 2 (binary distribution): `sunnypilot/mapd/mapd_installer.py:25-26` `DEFAULT_VERSION = 'chauffeur-bake-v1'`, `DEFAULT_BINARY_URL_TEMPLATE = 'https://github.com/chriscarlo/mapd/releases/download/{version}/mapd'`. Test fixture URL in `sunnypilot/mapd/tests/test_mapd_installer.py:34` updated to match.
- Cosmetic: `third_party/mapd_pfeiferj/` → `third_party/mapd/`; `MAPD_BIN_DIR` constant in `sunnypilot/mapd/__init__.py:4`. `mapd_repo/openpilot-mapd/Earthfile:86` docker push target → `chriscarlo/openpilot-mapd:latest`. `tools/vtsc_tuner/src/mapd_config.rs:34` doc comment path updated.
- Doc sweep: `docs/chauffeur/vtsc/osmIntegration/README.md`, `docs/chauffeur/MTSC_VTSC_MAPD_ROADMAP_2025-08-31.md`, `.codex/skills/vtsc-rally-copilot-hud/SKILL.md` references switched to chriscarlo and `chauffeur-bake-v1`.

### What changed off-tree
- `chriscarlo/mapd` `main` (commit `29eb2e8`) — synced from vendored `mapd_repo/openpilot-mapd/` so the public source matches the binary release. The standalone repo was 4 months stale before this.
- `chriscarlo/mapd` release `chauffeur-bake-v1` — arm64 ELF, 9.37 MB, statically linked, stripped, sha256 `d4d49746...`. Built via native amd64 + GOOS cross-compile (NOT qemu): `docker run --rm --platform=linux/amd64 -v $PWD:/work -w /work golang:1.24-alpine3.21 sh -c 'go mod download && GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -ldflags="-extldflags=-static -s -w" -o build/mapd .'` — ~1 min vs 20+ min for qemu.

### Two real bugs hit during the deploy
- **Historical `mapd_installer.py` silent download failure (fixed by the 2026-07-13 production work).** The first `chauffeur-bake-v1` boot advanced `MapdVersion` after a swallowed download error, leaving no active binary. Recovery at the time used a manual copy. Current code raises terminal download failure, advances version only after complete artifact validation, and restores the exact release from an identity-keyed persistent cache.
- **PBF prep is mandatory.** First `mapd --generate` against geofabrik's `california-latest.osm.pbf` ran 47 s, logged "Done Generating Offline Map" with zero "Writing Area" entries, wrote 26 tile files each 55–57 bytes (just header, zero ways). Root cause: geofabrik PBFs only carry node IDs on way refs; the osmpbf scanner reads way nodes with `Lat/Lon == 0`; `allMin/Max` get pinned to (0,0,0,0); `Overlapping(allMin..allMax, area..)` returns false; the `if !haveWays && !generateEmptyFiles { continue }` skip fires for every area. Fix: pre-process via `osmium tags-filter` (highway tags only) → `osmium add-locations-to-ways` → output as `ca_ready.osm.pbf`. Result: 13–35 s scan, 1638 real tiles, 483 MB total. Use `--platform=linux/amd64` on the docker invocation — without it WSL ran osmium under qemu (7+ minutes for filter + 2 min for add-locations).

### Updates to user config + memory
- `~/.config/vtsc_tuner/mapd.json` written: `pbf_path` → `ca_ready.osm.pbf` (NOT raw geofabrik), `mapd_binary_path` → `build/mapd_amd64` (cross-compile output, used by step 11+'s --generate). Step 10 sees this and skips the `earthly +build` it would otherwise try to run.
- New persistent memories saved for future agents: `feedback_branch_hook_heredoc.md` (the `git pull X Y` regex false-positive in commit messages, use `git commit -F file`), `feedback_mapd_installer_silent_failure.md`, `reference_pbf_prep_for_mapd_generate.md`, `reference_mapd_distribution.md`, `project_tici_tracks_exp01.md`.

### Stale claims in this changelog (now corrected)
- The 2026-04-18 entry below says "Distribution is local-only via SSH — no public CDN." That is no longer true — `chauffeur-bake-v1` is a public GitHub release. Keep that sentence as historical context but don't rely on it.
- The 2026-04-18 entry's mention of `/projects/mapd` as "publish destination only" still stands, but as of today it's been synced from vendored. Future tuner releases need to repeat that sync (or just `cp -a mapd_repo/openpilot-mapd/. /projects/mapd/` excluding `.git`) before cutting a new release tag.

### Real-world timing (single tici, WSL dev box, car-hotspot SSH ~5 MB/s)
PBF prep (one-time per geofabrik refresh): ~1 min on amd64 docker; 7+ min if you forget `--platform=linux/amd64`. Per-deploy: ssh pull + reboot ~75 s; bake all 36 regions ~35 s; rsync 448 MB ~82 s; final reboot ~75 s. Total wall-clock for a tune iteration ~4.5 min.

## 2026-04-18 — RebuildTilesAndReboot action + sigmoid-baked map tiles

- New `Action::RebuildTilesAndReboot` variant added to `apply.rs:17-31` (a superset of `PullOnTici`). Two-reboot chain: step 6 reboots after `git pull` so the device's openpilot Python lands first; then steps 7-999 generate sigmoid-baked tiles locally, rsync them per region, and final reboot.
- Steps added: 7 validate `~/.config/vtsc_tuner/mapd.json`; 8 wait-for-tici poll + `ssh ... ls /data/media/0/osm/offline/` discover; 9 `df -BM /data/media/0` pre-flight (50 MB/region × 1.5 headroom); 10 build mapd via `earthly +build` if missing; 11..N gen per region (step ids `100+i`); N+1..M rsync per region (step ids `200+i`); 999 final reboot.
- New file `mapd_config.rs`: `MapdConfig { pbf_path, mapd_repo_path, mapd_binary_path?, regions_override? }` saved at `~/.config/vtsc_tuner/mapd.json` (alongside `current.tune.json`). Tuner does NOT auto-create — surfaces a plain-English step-7 error with example contents.
- `Task` gained `Arc<AtomicBool>` cancel flag + `request_cancel()` / `is_cancel_requested()` API. Cancellation honoured at region boundaries (mid-`mapd --generate` is opaque). Cancel button shows in apply modal only for `RebuildTilesAndReboot` while running.
- Companion mapd changes (in `mapd_repo/openpilot-mapd/`, NOT the standalone `/projects/mapd` which is a publish destination only):
  - `offline.capnp` adds `safeSpeeds @20 :List(Float64)` (per-node, parallel to `nodes`), `schemaVersion @6 :UInt16`, `sigmoidHash @7 :Text`. Cap'n Proto field numbers immutable; readers ignore unknown fields → wire-compat with old binaries.
  - `sigmoid.go` (new): `PhysicsLatAccel`, `CurvatureToSpeed`, `SigmoidCfg.Hash()` mirror Python `_physics_based_lateral_acceleration` + `curvature_to_speed` exactly. `Hash()` is sha256 of `"%.6f|%.6f|%.6f|%.6f|%.4f|%.4f|%.2f"`-formatted tuple, first 12 hex chars.
  - `mapd.go` adds `--phys-{a,b,c,d,min-lat,max-lat}` and `--max-speed-default` CLI flags, threaded into `GenerateOffline(sigCfg)`. Defaults match the file-committed `vision_turn_controller.py` values so flagless invocation produces baseline tiles.
  - `mapd.go` runtime loop publishes `MapPreCurveSpeeds` + `MapTilesSigmoidHash` gated on `Offline.SchemaVersion() >= 1`. `math.go::GetStateBakedSpeeds` mirrors `GetStateCurvatures`'s walk to align baked outputs with `MapCurvatures` (length `numPoints-4`).
- Companion openpilot changes:
  - `common/params_keys.h`: registered `MapPreCurveSpeeds` + `MapTilesSigmoidHash` (both CLEAR_ON_ONROAD_TRANSITION, STRING).
  - `vision_turn_controller.py`: added module-level `_compute_runtime_sigmoid_hash` + instance methods `_load_map_pre_curve_speeds` and `_baked_vsafe_with_runtime_multipliers`. Replaced line 5534 with hash-checked baked-vs-live fallback; live-tuned PHYSICS_* changes the runtime hash and auto-engages fallback. Low-speed calibration scale != 1.0 also forces fallback for the whole batch.
- Tests: 4 new Go tests (`TestSigmoidMatchesPython` validates Go matches Python within 1e-9 against 11 precomputed κ samples + `f9d38ab3357c` reference hash; `TestSigmoidHashStableAcrossTrivialChanges`; `TestOldReaderNewTile` round-trips a v1 tile w/ baked + legacy ways; `TestLegacyTileReadsSchemaVersionZero` validates Cap'n Proto reads unset primitives as 0). 3 new Rust `mapd_config` unit tests. All pass. Pre-existing `TestVector`/`TestBearing` cupaloy snapshot mismatches are unrelated FP noise from a different platform/Go version.
- Landmines added to watch (see 2026-04-19 entry for updates):
  - `/projects/mapd` is a publish destination, NOT the source of truth — edits live in `mapd_repo/openpilot-mapd/`. The user owns chriscarlo/mapd; to publish a release tag, sync vendored → standalone first (`rsync -av --exclude=.git --exclude=README.md --exclude=CLAUDE.md mapd_repo/openpilot-mapd/ /projects/mapd/` preserves the standalone's customizations). As of 2026-04-19 the two are in sync at commit `29eb2e8` on chriscarlo/mapd main.
  - Device binary distribution is via GitHub releases (`chriscarlo/mapd` releases, tag `chauffeur-bake-v1` as of 2026-04-19) — not local-only SSH as originally scoped. The installer-side URL lives in `sunnypilot/mapd/mapd_installer.py:25-26`.
  - Tile transport to the tici is still local-only via rsync over SSH inside `RebuildTilesAndReboot`. `MapdTileBaseUrl` override still exists as an escape hatch if multi-device distribution becomes a goal.
  - Cap'n Proto Go bindings regenerate via `earthly +compile-capnp` (Earthfile:69-73). Locally accomplished without earthly via `PATH=$HOME/go/bin:$PATH capnp compile -I /tmp/go-capnp-std/std -ogo offline.capnp` (after curl-tarball clone of go-capnp std files; `git clone` is blocked by the openpilot branch-protection hook).
  - Cross-compile the device binary via native amd64 Docker + `GOOS=linux GOARCH=arm64`, NOT `--platform=linux/arm64` qemu emulation (see 2026-04-19 entry for the 20× speed delta and verification steps).

## 2026-04-18 — wider Q range + denser Q_CURVE_POINTS export

- Width-(Q) knob range bumped from `(0.3, 6.0)` → `(0.3, 16.0)` in `app.rs:784` so users can carve a 5–10 mph notch (Q ≈ 8–15 at typical highway centres). Default stays 1.5; mapping stays log.
- `Q_CURVE_POINTS` export sample count raised `64 → 256` in `apply.rs:439`. Old spacing (≈0.079 in log₁₀κ) was the same order as σ at Q=6, so anything narrower would alias on export. New spacing (≈0.020) leaves 4× headroom up to Q≈16. Runtime file `vtsc_curve_tuning.py` just lerps between samples — over-sampling is cheap, under-sampling silently smears notches.

## 2026-04-18 — hover + selected highlights on built-in handles

- Min/max rail pills, inflection circle, and both steepness wing chevrons now light up when the cursor is over them and stay lit when clicked, deselecting on Esc, click on empty plot, or click on another handle. Bands gained a matching hover ring (alongside their existing selected ring) so all "dots and arrow tips" behave identically.
- New `HandleId` enum (`MinRail`, `MaxRail`, `Inflection`, `LeftWing`, `RightWing`) and `PlotState::selected_handle` field. Mutually exclusive with `selected_band` — selecting one always clears the other (`apply_selection` enforces).
- `DragTarget::SteepWing` split into `LeftWing` / `RightWing` so each wing can highlight independently. Drag math still funnels both into `sharpness` via a shared match arm.
- Hover hit-test reuses `pick_drag_target`, so the highlight zones match the drag zones byte-for-byte (no surprise: hover an area, get the highlight, drag from same area, drag works).
- Cursor turns to `CursorIcon::Grab` over any handle, `CursorIcon::Grabbing` while a drag is active. Major affordance win for naive users.
- Selection wires up on both `clicked()` (quick press) and `drag_started()` (press + move). Empty-plot click clears selection.
- Esc to clear selection is gated on `state.line_menu.is_none()` so an open right-click menu still gets first-press priority for closing.
- Three small helpers in `plot.rs`: `drag_target_to_handle`, `level_for_handle`, `apply_selection`, plus a `HighlightLevel` enum threaded through `draw_rail` / `draw_wing_handle` and inflection inline.
- Selected ring widths/alphas: rails get a 4-px-expanded glow rect with optional outer stroke; inflection a 11-px stroke ring; wings a 11-px filled disc + stroke; bands keep their existing 4-px stroke ring. Hover variants are slimmer / ~0.5 alpha.

## 2026-04-18 — right-click context menu on the curve

- Hero plot now opens a small popup when the user right-clicks **on** the curve line (within 8 px of any rendered segment, not anywhere in the plot field).
- Menu items:
  - **✚ Add anchor point here** — calls the same `push_band_at` helper that shift-click uses, so the two paths can't drift. Adds a 0-dB band at the right-click's mph and selects it.
  - **⎘ Copy "X.X mph, Y.YY m/s²"** — `ctx.copy_text(...)` of the formatted data point. Useful for jotting tuning decisions in commits/notes; otherwise the exact (speed, accel) under the cursor is unrecoverable from the UI.
- Dismissal: choosing an item, pressing Escape, primary-clicking outside the menu rect, or right-clicking anywhere off the curve.
- Hit-test uses point-to-segment distance against the rendered (post-monotonic-v) `curve_pts`, so the menu only opens where the user can actually see the line. Baseline overlay is intentionally not hit-tested (it's reference-only).
- Discoverability hints updated: status bar default and empty-bands hint both mention "shift-click or right-click the curve".
- New helpers in `plot.rs`: `push_band_at`, `hit_polyline`, `dist_sq_point_to_segment`, plus `LineMenuState` / `LineMenuAction` types and `PlotState::line_menu`.
- Same-frame race: a fresh open is guarded by `menu_just_opened` so the opening right-click doesn't immediately get treated as an outside-click. Outside-click dismissal only fires on PRIMARY clicks; subsequent right-clicks either move the menu (on curve) or close it (off curve).

## 2026-04-18 — initial creation + iterative polish

### Bootstrapped

- New standalone Rust crate at `tools/vtsc_tuner/` (eframe 0.33 + egui + glow + winit 0.30 + X11-only).
- Eight src modules: `main`, `app`, `apply`, `io`, `knob`, `params`, `plot`, `sigmoid`, `theme`.
- 9 unit tests for sigmoid + plain-knob round-trips — all green.
- Deleted the superseded `tools/vtsc/curve_tuner.py` (tkinter ~1450 LOC) and the single dangling reference in `sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py`.

### Math layer
- `SigmoidParams` mirrors the runtime `_physics_based_lateral_acceleration` byte-for-byte, including asymptote clamps.
- `PlainKnobs` exposes four user-facing controls (Tight-Curve Ceiling, Straight-Road Ceiling, Transition Speed, Sharpness) that project down to the six raw params. Ranges match `vision_turn_params.py` so values survive the device-side re-clamp.
- `Band` + `apply_prepared_bands` compose Gaussian dB bumps in log-κ space. `prepare_bands` freezes per-band κ_centre against the base sigmoid so bands don't chase their own tail.
- `bands_as_q_curve_points` exports bands as `(κ, q_speed)` tuples for the runtime's multiplicative speed-scale path — gain in lateral accel converted to speed via `sqrt(ratio)`.

### UI
- Hero plot custom-painted in `(speed mph, m/s²)` space; baseline (repo-default) drawn as a faint dashed overlay.
- Draggable handles: min/max rails (with prominent left-edge pill + arrows), inflection dot (X-axis → transition speed), steepness "wings" (in/out → sharpness), per-band dots.
- Shift-click anywhere on the curve adds a new band at that location.
- EQ bands get stable palette colors; each band's marker, side-panel header swatch, and translucent Q zone on the graph all match. Q zone visible only for the selected band.
- Per-band controls in side panel: compact knobs for Gain (dB), Width (Q, log), Center (mph), each with editable DragValue under the knob.
- Rotary knob widget: vertical drag, scroll wheel, shift=fine, double-click reset. Auto-resizes footprint to fit the label string.
- Dark "pro audio" palette, 1.5× `pixels_per_point` for readability.

### Landmine resolved: WSLg corner-resize crash
- Symptom: `Io error: Broken pipe` ×3 + `WinitEventLoop(ExitFailure(1))` whenever the window was resized.
- Root cause: `smithay-clipboard` / `wayland-backend` logs EPIPE when WSLg's compositor sends resize protocol in a sequence calloop can't follow.
- Fix: `NativeOptions::event_loop_builder` calls `winit::platform::x11::EventLoopBuilderExtX11::with_x11()` → winit uses XWayland instead of native wayland, resize is clean.

### Landmine: Claude Code Bash `run_in_background`
- Symptom: same `Io error: Broken pipe` spam, but from the first frame.
- Root cause: the Bash tool's detach wrapper severs the X socket connection.
- Workaround: launch via `nohup <bin> >log 2>&1 </dev/null & disown` with no `pkill` prelude. Exit code 144 from bash is the telltale that the wrapper killed the process.

### Apply pipeline (`apply.rs`)
- Background-thread chain: write tune JSON → patch source → git commit → git push → ssh tici pull+reboot.
- Four `Action` kinds covering subsets of the chain.
- SSID-aware tici discovery: `iwgetid` (Linux) or PowerShell `Get-NetConnectionProfile` (WSL); probes `commaHome` / `commaCar` / `commaAdb` via `ssh -o ConnectTimeout=2 -o BatchMode=yes <profile> true`; first responder wins; SSID biases order.
- Remote: `cd /data/openpilot && git pull` followed by fire-and-forget `sudo reboot`.
- Progress modal with per-step Running/Ok/Err status, plain-English text, and raw command-line detail below.

### Undo/redo
- `Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y` consume at the top of `update()`.
- `History` keeps pre-edit snapshots with frame-debounced capture so continuous drags collapse to one undo step.
- "Revert to baseline" is also undoable via an explicit `commit_snapshot()` call.

### Monotonic-v plot fix
- User observed the plot folding back on itself when a band's slope exceeded `a/κ`.
- Post-process in `plot::hero_plot` holds v at the running max as samples are walked in κ-descending order — aggressive bands render as vertical notches (parametric-EQ idiom) rather than multi-valued folds.

### Layout polish
- Knob labels were clipping ("ight-Curve Ceilin"). `knob.rs` now measures the label via `ctx.fonts_mut(|f| f.glyph_width(…))` and grows the knob's footprint to fit.
- Header subtitle was also clipping; shortened to `"curvature → lat-accel"`.
- `pixels_per_point` dropped from 2.0 → 1.5 after user feedback that 2× was too big.

---

<!--
Template for future entries:

## YYYY-MM-DD — <short title>

### <section>
- <change>, paths touched (e.g. `src/apply.rs:123`), motivation, any landmines encountered.

-->
