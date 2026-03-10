# VTSC Post-Patch Drive Check — 2026-03-10

## Scope

- Repo/device branch: `chauffeur-dev4`
- Device commit verified on tici: `5c47ce2`
- Device params verified on tici:
  - `MTSCLookaheadEnabled=1`
  - `VTSCMapStrategy=strategic`
- Route analyzed: `0000009e--c5e182f3b8`
- Raw artifacts: `.cache/vtsc/commaCar_20260310_route9e/`

## Route timing / corridor

- Latest available route on device after the patch deployment
- Segment mtimes span approximately `2026-03-10T00:44:31Z` through `2026-03-10T00:59:31Z`
- Road sequence inferred from `liveMapDataSP.roadName`:
  - `0..3`: `Banbury Cross Road`
  - `4..6`: `French Creek Road`
  - `7..11`: `Old French Town Road`
  - `12`: `Mother Lode Drive`
  - `13..16`: `Pleasant Valley Road`

The mountain section relevant to the previous tuning complaint is the `Old French Town Road` block, especially segment `10`.

## Bottom line

- The patch is live on the device, but it did **not** solve the core complaint.
- The strongest post-patch `Old French Town Road` segment still shows a large release spike where VTSC accelerates far too hard out of a curve.
- I cannot claim a clean improvement from this route.

## Strongest post-patch evidence

### Segment 10 still has a large exit-release spike

Qlog `longitudinalPlanSP.visionTurnSpeedControl` on `Old French Town Road` segment `10` shows:

- around `696.8`, `v=10.00`, `vt=10.14`
- at `697.3`, `v=11.04`, `vt=23.82`
- then VTSC stays around `23.8 -> 23.4` for roughly two seconds while the road is still in the same winding section

This is the exact “pulls way too hard out of curves” behavior the driver reported.

Later in the same segment, VTSC does tighten for the next bends:

- around `713.3 -> 720.8`, future curve distance counts down from `162.5 m` to `28.4 m`
- around `742.3`, VTSC finally clamps to `7.55 m/s` for a `curveMaxCurvature` of `0.05642`

So the system still shows the same shape:

1. release aggressively after one bend
2. only later re-tighten for the next one

### Manual gas/brake interventions on the tuned road are still present

Filtered `Old French Town Road` gas/brake events (`v_ego >= 5 m/s`):

- seg `10` brake: `v=12.19`, `vt=8.07`, `delta=-4.11`, `pred_lat=3.76`
- seg `10` gas: `v=7.85`, `vt=8.18`, `delta=+0.33`
- seg `10` gas: `v=17.87`, `vt=17.27`, `delta=-0.60`, `curve_distance=8.3 m`
- seg `10` gas: `v=11.17`, `vt=7.55`, `delta=-3.62`
- seg `10` brake: `v=10.18`, `vt=7.55`, `delta=-2.63`

This is not a no-intervention result.

## Comparison to the pre-patch drive

Pre-patch route: `0000009d--378452f068`

Clean apples-to-apples comparison is limited because the new drive did not include `Greenstone Road`, which was a meaningful part of the earlier mountain run.

What can be said safely:

- The earlier pre-patch route had the known release spike on `Old French Town Road` segment `11`, where VTSC jumped back to `22.22 m/s` with the next curve still only `38.6 m` away.
- The new post-patch route still has a release spike on `Old French Town Road`, and in segment `10` it is more dramatic in raw target magnitude: `~10 m/s -> 23.8 m/s` immediately after curve exit.
- The broader intervention count may be somewhat lower than the earlier full mountain corridor, but this route is not close enough to a controlled repeat to claim improvement from counts alone.

## Occlusion / degraded vision finding

Controller-only replay on post-patch rlogs:

- `replay_seg10_cruise25.jsonl`
  - vision states: `FULL 593`, `PARTIAL 359`, `SEVERE 248`
  - `occluded=true` points: `0`
- `replay_seg11_cruise25.jsonl`
  - vision states: `FULL 676`, `PARTIAL 272`, `SEVERE 252`
  - `occluded=true` points: `0`

Interpretation:

- This drive again does **not** support “explicit FOV occlusion gate is the main culprit.”
- The remaining issue is still target shaping / map-vision behavior on chained bends, not a latched `occluded=true` state.

## Current conclusion

The deployed patch tightened one failure mode, but it did not materially fix the real mountain-road complaint. The new drive still shows VTSC releasing to a much too-permissive target out of a curve on `Old French Town Road`, then having to pull speed back down for the next bend.

Most likely implication:

- the real missing behavior is still a stronger chained-curve / two-curves-ahead map constraint
- or map-floor persistence tied to future anchors that are not yet being selected early enough on this road

This post-patch route is therefore a **not fixed** result.

## Follow-on iteration after this analysis

Subsequent local tuning work kept the occlusion subsystem disabled and shifted focus back to the strategic map path.

Current local hypothesis and implementation direction:

- the strategic chained-curve envelope was still too sparse because it only carried monotonic new-minimum anchors into the map chain logic
- that can miss the "re-accelerate, then retighten" pattern common on winding mountain roads
- the local fix now feeds the chain envelope from extracted future curve anchors that preserve meaningful local minima after a recovery rise, instead of only global minima

Why this matters for the road complaint:

- on roads like `Old French Town Road`, the next meaningful curve after the current bend can be looser than the current one but still close enough to matter for exit acceleration
- if the map chain only remembers new global minima, VTSC can briefly behave as if that intermediate future bend does not exist
- preserving those intermediate curve anchors gives the strategic path a real chance to hold back acceleration between bends instead of waiting to rediscover the next constraint later

Verification on the current local tree:

- `PYTHONPATH=$PWD .venv/bin/pytest sunnypilot/selfdrive/controls/lib/tests/vtsc`
- result: `94 passed`

This is still an on-desk synthetic/test-backed improvement, not a new onroad proof yet. The next decisive step is a fresh drive log on the no-occlusion + richer chain-anchor build.

## Windy-road detector iteration

To support future "delay slowing / slow harder later" tuning without a manual switch, the local tree now includes a behavior-neutral winding-road classifier driven from the same map lookahead inputs VTSC already uses.

What it currently does:

- classifies a `winding_road_active` context from the first ~`325 m` of upcoming map geometry
- uses repeated meaningful local minima in the map-derived `vsafe` profile plus short anchor-to-anchor gaps
- requires clustered curves, not just one tight ramp, and rejects sparse freeway bends and shallow rolling-road wiggles in synthetic tests
- only populates debug telemetry for now; it does **not** alter `vtsc_cmd`, arbitration, or planner behavior yet

Current telemetry exposed in `snapshot_debug_state()`:

- `winding_road_active`
- `winding_road_score`
- `winding_anchor_count`
- `winding_short_gap_count`
- `winding_curve_distance_m`
- `winding_road_horizon_m`
- `winding_reference_vsafe_mps`
- `winding_min_anchor_vsafe_mps`

Synthetic validation added so far:

- dense chained-bend cluster: detects as winding
- single off-ramp / single tight curve: rejected
- sparse freeway bends with long gaps: rejected
- shallow rolling-road wiggles: rejected
- small `vsafe` noise on a winding profile: remains active
- controller snapshot path: winding fields populate while strategic map behavior remains intact

Known weakness in this first cut:

- the classifier is intentionally based on the same unsigned map `vsafe` profile used by the cap path, so it knows "curve cluster ahead" but not true signed left/right alternation
- that means it should be thought of as a winding-context detector, not yet a full "switchbacks vs sweepers" classifier
- the next refinement, if needed, should pull in signed geometry from the existing curve-preview polyline rather than pretending the unsigned map path can answer that question

## Offline mapd winding metadata prototype

A standalone prototype benchmark now exists at:

- `.cache/mapd_winding_proto/main.go`

It does **not** touch the production mapd code path yet. The prototype:

- reads real mapd offline Cap'n Proto tiles from a tar archive
- scores each directed way with:
  - `windy 0..5`
  - continuous `score`
  - `confidence`
- uses explicit deterministic features over a rolling ~`325 m` window:
  - meaningful curve-anchor count
  - short-gap count between anchors
  - sign-change count between anchors
  - curved-distance fraction
  - cumulative heading change
  - tightest implied safe speed

Current reports:

- sample benchmark: `.cache/mapd_winding_proto/eldorado_sample_report_v2.json`
- full county-scope benchmark over all matching local sample tiles:
  `.cache/mapd_winding_proto/eldorado_county_full_report.json`

Approximate El Dorado County benchmark bbox used:

- lat `38.45 .. 39.35`
- lon `-121.60 .. -120.00`

Compute result on the local laptop CPU:

- sample archive: `.cache/mapd_samples/sample_38_-122.tar.gz`
- matching local offline tiles in county bbox: `35`
- random sampled tiles first pass: `12`
- mean serial processing cost per tile: about `49..52 ms`
- p95 tile cost on the sample: about `111..133 ms`
- slowest sampled tile: about `202 ms`
- full `35`-tile county-scope serial pass completed in about `1.7..1.8 s`
- linear estimate for a `64`-tile 2x2-degree archive at the same mean cost: about `3.1..3.3 s`

Interpretation:

- this does **not** currently look like a GPU problem
- the offline enrichment pass appears very feasible on the local laptop CPU
- if needed, parallel CPU workers across tiles should matter more than GPU offload for the first implementation

Representative strongest prototype outputs were sensible mountain / forest roads in the eastern sample tiles:

- `Camp Wolfeboro Road`
- `Forest Route 7N09`
- `Little Buck Road`
- `Spicer Reservoir Road`
- `Salt Springs Reservoir Road`

Current limitation before productionizing:

- the prototype scores each directed OSM way independently
- production metadata should likely become directional rolling-window summaries that can be composed with mapd's current-way plus next-ways chain, not just a blunt whole-way label
- the current `0..5` thresholds are a usable first pass but still need tuning against known-good / known-bad roads before being baked into offline generation

## Real-road fixture validation from offline prototype

To avoid overfitting the winding detector to hand-made synthetic `vsafe` patterns, I exported one representative profile per prototype severity level from the offline county pass:

- fixture: `sunnypilot/selfdrive/controls/lib/tests/vtsc/fixtures/winding_road_profiles/eldorado_representatives.json`

Each fixture row contains:

- prototype severity `level`
- human-readable road label when available
- downsampled `profile_s_m`
- downsampled `profile_vsafe_mps`

Those real-road windows are now part of VTSC regression coverage.

Observed behavior of the current runtime winding detector on these representatives:

- prototype `windy 0..2` stays `inactive`
- prototype `windy 3..5` flips `active`
- runtime winding score increases monotonically across the sampled `0..5` ladder
- the key boundary case is `windy 2`: it can already score moderately high from two hard bends, but it still stays out of winding-road mode because the bends are not tightly chained

That last point is useful. It means the current runtime gate is already distinguishing "two isolated hard corners" from "sustained winding-road context," which is exactly the failure mode we need to avoid before any later/harder-slowing logic is allowed to key off this signal.

## Recommended minimal offline schema shape

If this graduates into production map tiles, the smallest useful addition to `mapd_repo/openpilot-mapd/offline.capnp` is still directional summary metadata, not a single undirected `isCurvy` bit.

Recommended first-pass fields on `Way`:

- `windingForwardLevel` as `UInt8`
- `windingBackwardLevel` as `UInt8`
- `windingForwardScore` as `UInt8`
- `windingBackwardScore` as `UInt8`
- `windingForwardConfidence` as `UInt8`
- `windingBackwardConfidence` as `UInt8`

Reasoning:

- `level` carries the coarse user-facing severity bucket (`0..5`)
- `score` preserves a finer gradient for future tuning without needing a schema change
- `confidence` lets runtime ignore or down-weight uncertain classifications near merges / splits / sparse geometry
- `UInt8` keeps the payload small enough that we are not obviously bloating offline tiles just to stash heuristic metadata

I still would not bake a final runtime behavior off these fields yet. The first production use should be telemetry and gating only:

- expose the directional winding metadata at runtime
- compare it against the existing on-device winding detector and real drive logs
- only then decide whether it should unlock later/harder slowing or exit-accel restraint variants

This schema shape is now implemented locally in `mapd_repo/openpilot-mapd`:

- `offline.capnp` now carries the six compact directional winding fields on `Way`
- `generate_offline.go` now computes and stores forward/backward winding metadata per way during offline generation
- `winding.go` contains the deterministic scorer ported from the prototype, compacted to `UInt8 level/score/confidence`

Verification completed for the schema/generator path:

- Cap'n Proto bindings regenerated locally with `capnp` + `capnpc-go`
- compile-only package verification:
  `/projects/chauffeur/data/openpilot/.cache/toolchains/go/bin/go test -run '^$' ./...`
- targeted Go tests for the new schema/scorer path:
  `/projects/chauffeur/data/openpilot/.cache/toolchains/go/bin/go test -run 'TestConfiguredTileBaseURL|TestTileArchiveURL|TestComputeWayWindingMetadata|TestWindingLevelForMetrics' ./...`

That runtime bridge is now implemented locally as telemetry:

- `mapd.go` now reads the directional `Way.windingForward*` / `Way.windingBackward*` fields on the active current/next-way chain
- it writes a compact route-level JSON param `MapWindingSummary` containing:
  - `valid`
  - `level`
  - `score`
  - `confidence`
  - `currentLevel`
  - `currentScore`
  - `currentConfidence`
  - `wayCount`
- `OsmMapData` now republishes that summary into `liveMapDataSP`
- `VisionTurnController` now snapshots those fields as `mapd_winding_*` telemetry alongside the existing local winding detector output

What is still intentionally not done:

- VTSC does not yet change behavior based on the baked winding metadata
- no large offline planet or state rebuild has been kicked off yet
- there is still no onroad comparison dataset yet for:
  - local winding detector vs baked route summary
  - baked route summary vs real manual interventions on the mountain road

## Deployment / ownership constraint

If we take this beyond prototype work, we need to own both the `mapd` code path and the tile host path.

Current integration points in-tree:

- `sunnypilot/mapd/mapd_installer.py` downloads the `mapd` binary from:
  `https://github.com/pfeiferj/openpilot-mapd/releases/download/<VERSION>/mapd`
- `mapd_repo/openpilot-mapd/download.go` downloads offline archives from:
  `https://map-data.pfeifer.dev/offline/<lat>/<lon>.tar.gz`

That means a real forked deployment requires two things:

- a code fork or vendored copy of `openpilot-mapd`
- a hosted tile endpoint serving the expected `offline/<lat>/<lon>.tar.gz` layout

Important clarification:

- vendoring the `mapd` source does **not** imply shipping all maps on-device
- the device already downloads only requested 2x2-degree archive regions and unpacks them locally
- the storage concern only appears if we choose to bundle archives into the device image, which we do not need to do

Current recommendation:

- vendor the `openpilot-mapd` source into our tree so schema + generator changes live with the VTSC work
- keep map data external and host only the subset of generated archives we actually want to serve
- patch both binary-download and tile-download URLs to be configurable or to point at our owned endpoint before depending on any new metadata

That configurability is now implemented locally:

- `MapdReleaseVersion` lets the installer target a fork-owned release version instead of the baked-in upstream default
- `MapdBinaryUrl` lets the installer fetch the `mapd` binary from an owned host
- `MapdTileBaseUrl` lets the `mapd` downloader fetch `offline/<lat>/<lon>.tar.gz` from an owned tile endpoint

Current behavior:

- if none of those overrides are set, behavior remains on the upstream defaults
- the device still downloads only requested regional archives; this does **not** imply bundling all map data onto the device
- environment variable fallbacks also exist for local development:
  - `SP_MAPD_RELEASE_VERSION`
  - `SP_MAPD_BINARY_URL`
  - `MAPD_TILE_BASE_URL`

Verification completed for this override path:

- Python mapd tests:
  `PYTHONPATH=$PWD .venv/bin/pytest sunnypilot/mapd/tests/test_mapd_installer.py sunnypilot/mapd/tests/test_integration.py`
- Go targeted tests for tile URL resolution:
  `/projects/chauffeur/data/openpilot/.cache/toolchains/go/bin/go test -run 'TestConfiguredTileBaseURL|TestTileArchiveURL' ./...`

One unrelated caveat remains in the upstream `openpilot-mapd` repo copy:

- full `go test ./...` currently fails due pre-existing snapshot drift in `math_test.go`, not due the override-path changes

## Runtime bridge verification

Runtime validation now covers the full prototype path:

- mapd compile-only package verification:
  `/projects/chauffeur/data/openpilot/.cache/toolchains/go/bin/go test -run '^$' ./...`
- targeted Go tests for:
  - tile base-url overrides
  - directional winding metadata
  - route-level winding aggregation
  command:
  `/projects/chauffeur/data/openpilot/.cache/toolchains/go/bin/go test -run 'TestConfiguredTileBaseURL|TestTileArchiveURL|TestComputeWayWindingMetadata|TestWindingLevelForMetrics|TestDirectionalWindingSummary|TestAggregateRouteWindingSummary' ./...`
- Python mapd tests, including JSON param parsing and `liveMapDataSP` winding publication:
  `PYTHONPATH=$PWD .venv/bin/pytest sunnypilot/mapd/tests/test_integration.py sunnypilot/mapd/tests/test_mapd_installer.py`
- full VTSC regression suite, including snapshot exposure of `mapd_winding_*` telemetry:
  `PYTHONPATH=$PWD .venv/bin/pytest sunnypilot/selfdrive/controls/lib/tests/vtsc`

Latest results from this state:

- mapd targeted Go tests: `ok`
- Python mapd tests: `18 passed`
- VTSC suite: `97 passed`
