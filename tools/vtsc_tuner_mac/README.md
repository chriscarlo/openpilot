# VTSC Tuner for macOS

This directory contains the native macOS port of the standalone VTSC sigmoid
tuner. It is intentionally scoped to the tuner application; it is not a port of
the Chauffeur/openpilot repository.

The app is a Swift 6/SwiftUI package with no third-party Swift packages. SwiftPM
is the source of truth. Xcode can open `Package.swift` directly, so a generated
`.xcodeproj` is deliberately not checked in. A small bundled native Go helper
uses the pinned Cap'n Proto module to decode mapd's packed tile schema exactly.

## Requirements

- macOS 14 or newer.
- Xcode 16 or the matching Xcode Command Line Tools (`swift --version` should
  report Swift 6).
- A local Chauffeur checkout when using source-apply actions.
- The same command-line tools used by the relevant apply action (`git`, `ssh`,
  and, for map-tile workflows, the tools shown by the app).

The packaged app is intentionally **not App-Sandboxed**. It needs user-approved
access to a Chauffeur checkout and must be able to launch local `git`/`ssh`
processes. The build script applies an ad-hoc signature for reliable local
launching; it does not create a Developer ID-signed or notarized distribution.

## Test and run from SwiftPM

From this directory:

```bash
swift test
swift run VTSCTuner
```

The Map Preview decoder is packaged automatically by `build_app.sh`. For a
development build, create or test the helper independently with:

```bash
bash scripts/build_tile_decoder.sh --test
```

`swift run` then discovers `.build-tools/bin/vtsc-tile-decoder` from the package
directory. Set `VTSC_TILE_DECODER` to use another trusted development build.

To work in Xcode without generating project files:

```bash
open Package.swift
```

Select the `VTSCTuner` executable scheme in Xcode.

## Build a launchable app bundle

```bash
bash scripts/build_app.sh
```

The result is:

```text
dist/VTSC Tuner.app
```

Open that build in place with:

```bash
open "dist/VTSC Tuner.app"
```

Or package, install to `~/Applications`, and open it in one command:

```bash
bash scripts/build_app.sh --install --open
```

To install somewhere else, set `VTSC_APP_INSTALL_DIR`; for example, a user with
permission to write `/Applications` can run:

```bash
VTSC_APP_INSTALL_DIR=/Applications bash scripts/build_app.sh --install
```

Set `VTSC_APP_OUTPUT_DIR` to place the generated bundle somewhere other than
`dist`. The script always rebuilds the `VTSCTuner` release product, recreates
only its destination app bundle, validates `Info.plist`, applies an ad-hoc code
signature, and verifies the resulting signature.

## Chauffeur repository selection

The app validates a repository by requiring `.git` plus these two files:

```text
sunnypilot/selfdrive/controls/lib/vision_turn_controller.py
sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py
```

Use the toolbar folder button or **File → Choose Chauffeur Repository…**
(Cmd-Shift-O) if the app does not select the checkout you want. The chooser
accepts directories only and saves a validated path in the
`VTSCTuner.repositoryPath` user default. A packaged build left under this
repository can discover the checkout by walking up from its bundle. Installed
builds can also discover the usual `~/Documents/chauffeur` location. The app
additionally checks its saved selection, the current working directory, and
these environment overrides:

```text
VTSC_REPO_ROOT
VTSC_TUNER_REPO
```

For a one-off environment override, launch the bundle executable directly so
the process receives the variable:

```bash
VTSC_REPO_ROOT=/path/to/chauffeur \
  "dist/VTSC Tuner.app/Contents/MacOS/VTSCTuner"
```

Repository choice changes which checkout source-apply actions mutate. Confirm
the repository shown in the app before applying, committing, pushing, or
pulling on a tici.

## Per-user files

Native macOS state lives under Application Support:

```text
~/Library/Application Support/vtsc_tuner/current.tune.json
~/Library/Application Support/vtsc_tuner/mapd-release.json
~/Library/Application Support/vtsc_tuner/mapd.json
~/Library/Application Support/vtsc_tuner/curve_calibration_samples.json
~/Library/Application Support/vtsc_tuner/map_tiles/offline/
~/Library/Application Support/vtsc_tuner/map_tiles/sets/<tile-set-id>/
```

If the native tune file is absent, the app can read the Rust/WSL-era tune from:

```text
~/.config/vtsc_tuner/current.tune.json
```

Saving writes the native Application Support location. The mapd configuration
is not auto-created because its PBF and mapd checkout paths are
machine-specific.

Every car-facing action requires `mapd-release.json` to identify one immutable
Linux ARM64 mapd artifact by release/build/estimator/capability and SHA-256.
The app validates its ELF architecture and embedded markers before any tune,
Git, device, or reboot mutation.

For **Apply + deploy runtime + build canonical tiles**, `mapd.json`
must additionally point to a prepared PBF and a native macOS mapd generator. A
Linux/ELF generator cannot run on the Mac and is rejected during the same
no-mutation preflight. Example:

```json
{
  "pbf_path": "/absolute/path/ca_ready.osm.pbf",
  "mapd_repo_path": "/absolute/path/openpilot-mapd",
  "mapd_binary_path": "/absolute/path/mapd-darwin-arm64",
  "regions_override": null
}
```

The PBF must already contain locations on ways (for example, from
`osmium add-locations-to-ways`). The generator is invoked with all six tuned
`--phys-*` values and writes into a clean temporary directory. Every tile is
decoded and hashed before the finished tree and manifest move into the durable
`map_tiles/sets/<tile-set-id>/` store. Deployment transfers only to an empty
remote partial set, verifies the complete manifest, atomically activates one
immutable generation, and retains the previous generation for rollback. It
never rsyncs into the active tile tree.

**Apply + deploy runtime (keep current tiles)** does not require `mapd.json` or a tile set.
The runtime whole-curve estimator consumes existing raw tile geometry and the
first road test intentionally leaves the active tile identity unchanged.

## Map Preview and real-curve calibration

Choose **Map Preview** in the toolbar or press Cmd-2. The muted Apple map is
only geographic context and place search. Every colored road overlay comes
from the selected local mapd `offline` directory; Apple road geometry is never
substituted for missing mapd data.

Road text defaults to Apple's normal map labels. The **Road text** menu can add
larger high-contrast labels sourced from each decoded mapd way's own name and
reference. Search for a city, road, address, or place, choose the result, then
use **Set Opening** to make that result the map's persistent opening location;
the same menu can revisit or clear it.

The app automatically opens the Application Support tile cache when it exists.
You can also choose an existing `offline` folder, or explicitly copy the live
tici cache with **Sync from tici**. Sync reads
`/data/media/0/osm/offline/` through the selected SSH profile and does not run
while panning. It first indexes a new nonempty staging directory, then swaps it into
place and keeps the prior cache as `offline.previous`; current and stale tile
generations are never merged into one map view.

Map colors have three meanings:

- **Currently Baked** is the `Way.safeSpeeds` data actually stored in the tile.
- **Proposed Tile Bake** recomputes each node with the in-app sigmoid using the
  same three-point curvature and physics bake as mapd.
- **Delta** is proposed minus currently baked.

These overlays are schema-v1 per-node tile values, not a promise of the final
commanded speed. In particular, their stored/proposed bake uses the raw
three-node OSM vertex circle, while live mapd publishes an arc-length-weighted
average of three neighboring circles over five route nodes after applying its
lane merge/split correction. Route matching, runtime modifiers, deceleration
timing, calibration, and planner policy also participate when strategic VTSC
runs on the car. The selected-curve inspector keeps those distinctions visible.

To teach the tune from familiar roads:

1. Search or pan to a road, then click the colored overlay near the curve. The
   selection snaps to the nearest local peak in mapd-equivalent, route-smoothed
   curvature within 120 m along that road, without crossing to a more distant
   tighter bend.
2. Add the selected interior point to the persistent curve bank. Straight and way-end
   nodes are inspectable but cannot constrain the sigmoid.
3. Capture at least five curves with varied tightness and enter the speed you
   want at each one. Five is only the fitting minimum: the bank has no fixed
   maximum or rollover, and adding a later curve preserves every earlier item.
4. Run the fit across the entire bank and review the before/after RMSE, every
   curve's speed and implied-lateral-acceleration residual, and any warnings for
   conflicting requests, pointwise envelope limits, or high effective lateral
   acceleration.
5. Accept the fit to replace the four plain-English base controls and the EQ
   band set with the generated residual curve as one undoable edit.

The curve bank is saved in Application Support together with the original
checkout anchor captured for that bank. That anchor survives source apply and
relaunch so regenerating an accepted proposal remains idempotent; clearing the
bank clears the anchor. Selecting the exact same map point updates its existing
bank item rather than duplicating it. Numbered purple pins keep all selected
curves identifiable on the map, and the scope button on a row recenters that
saved sample after you pan elsewhere or relaunch the app.

Calibration never fits the raw vertex circle. The app stitches the unique
direction-feasible physical route across cached tile boundaries and reproduces mapd's current
five-node weighted curvature, including mapd's 0.0015 merge/split clamp before
averaging. Every saved row shows that effective curvature, its five-node route span,
and target lateral acceleration; a materially larger raw vertex value remains
visible only as a diagnostic. Schema-1/2 banks migrate to schema 3 by retaining
IDs and requested speeds but re-resolving geometry before they may fit, and the
stored estimator version forces re-audit whenever parity logic changes. Missing,
direction-dependent, or fork-ambiguous route context excludes the unresolved
row instead of silently choosing a branch or substituting raw curvature. The
persisted 18-curve bank's peak requested acceleration changed from 9.60 m/s² on
raw vertices to 5.38 m/s² on runtime-equivalent curvature; one 3→4-lane sample
fell from an intermediate 4.69 to the correct 1.79 m/s² after transition parity.

Schema-v1 `MapPreCurveSpeeds` were baked from raw vertex curvature and therefore
do not share the estimator used by live `MapCurvatures`. The Python controller
keeps that legacy stream disabled until a versioned tile/runtime pair guarantees
estimator alignment; the normal live sigmoid path continues to use mapd's
smoothed curvature.

The fitter targets the complete exported runtime curve: a bounded four-knob
sigmoid backbone followed by a canonical, bounded Q=4 residual curve. It refits
the backbone so projected targets fall inside the selected base's real
0.5–1.5 residual speed authority, then solves the residual against the whole
bank. The sigmoid and residual can reshape the bank through a four-sigma
log-curvature influence collar (a curvature ratio of `sqrt(10)` on each side).
Beyond that collar, dense transition-aware checks keep both the backbone and
the complete exported runtime curve within 2 mph of the persisted checkout
anchor. Requested speeds are projected to the nearest weighted monotonic curve
before the residual solve; nearby or inverted targets that one curvature-only
runtime curve cannot satisfy independently remain explicit conflicts rather
than becoming narrow speed reversals. Proposal scoring and Curve Lab both use
the source-rounded 256-point interpolation that will be written to
`Q_CURVE_POINTS`.

Safety evaluation uses the same six-decimal sigmoid constants, four-decimal
rails, rounded Q knots, and log interpolation that source apply writes. The
monotonicity guard measures cumulative speed rise above the running minimum on
a dense grid, so a sequence of individually tiny increases cannot hide a
larger tighter-curve speed pocket.

The proposal shows both RMSE and the worst individual miss. Generated curves
are checked for tighter-curve/faster-command reversals, and proposals above the
5.5 m/s² safety-review threshold require an explicit acknowledgement before they can
be moved into Curve Lab. Curve Lab keeps its usual 0–90 mph viewport, but
automatically expands to the 156.6 mph runtime center limit when a generated
high-center residual band must be reviewed or edited. Effective predictions
use the checked-in Params defaults (0 mph low-speed bias, 50 mph taper endpoint,
speed factor 1.0), not live or learned values from a connected device.
On-device overrides and route/live strategic context can therefore still
change the command. Accepting a result only changes the tune in memory; it
never applies source, commits, pushes, or contacts the car.

Source apply treats the fitted sigmoid as one checked-in authority. It updates
the six controller constants, their six `common/params_keys.h` defaults, both
the reset and ensure values in the offroad physics panel, and the Q curve only
after confirming that the checkout's existing copies agree. Tici actions
preflight a clean exact dev/origin/device commit, `IsOffroad=1`, `IsOnroad=0`,
and `MTSCLookaheadEnabled=0`; run the host verification gate; fast-forward the
exact pushed commit; install and persistently cache the verified ARM64 mapd
release; read back all six physics Params and exact Q source; optionally
activate a canonical tile generation; then reboot once. Complete postflight or
coherent rollback is required, so a Git push, file transfer, or reboot dispatch
alone is never reported as success.

The macOS app owns deployment policy, validation, transaction planning, and
result decoding in Swift. The tici side is deliberately limited to POSIX/Git
file and process primitives; it does not need a tici Python interpreter or
Python modules. One bundled static ARM64 helper performs the Linux-only atomic
tile-directory exchange, which Swift cannot execute on the device.

## Packaging verification

The build script performs bundle checks automatically. They can also be run
manually:

```bash
plutil -lint "dist/VTSC Tuner.app/Contents/Info.plist"
codesign --verify --deep --strict --verbose=2 "dist/VTSC Tuner.app"
```
