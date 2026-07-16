# Apply chain contract

The native macOS pipeline in `tools/vtsc_tuner_mac/` is authoritative for production deployment. The legacy Rust `apply.rs` implementation remains a local/WSL compatibility surface and still has its historical pull/rebuild behavior; do not use that older path as the production contract.

## Native actions

`ApplyPipeline` streams stable-ID `ApplyEvent`s through `AsyncStream`. Each action is cumulative:

| Action | Durable tune | Source authorities | Commit | Exact push | Production tici transaction | Canonical tiles |
|---|---|---|---|---|---|---|
| `local` | yes | yes | no | no | no | no |
| `commit` | yes | yes | yes | no | no | no |
| `push` | yes | yes | yes | yes | no | no |
| `pullOnTici` | yes | yes | yes | yes | yes | unchanged |
| `rebuildTilesAndReboot` | yes | yes | yes | yes | yes | build, verify, activate |

The runtime whole-curve-v3 profile uses existing raw map geometry, then route-bakes estimator-aligned physics speeds when mapd publishes the selected continuous route. `pullOnTici` is therefore the normal first-road-test action: it installs the v3 runtime/release while intentionally proving that the active tile identity did not change. `rebuildTilesAndReboot` remains reserved for a requested canonical tile generation.

## No-mutation preflight

Every car-facing action verifies all prerequisites before saving or patching the tune, committing, writing Params, replacing binaries, activating tiles, or rebooting:

- selected Chauffeur checkout and all four source authorities are parseable and mutually consistent;
- branch is exactly `chauffeur-exp01`, upstream is `origin/chauffeur-exp01`, the checkout is clean, and local/origin HEADs match;
- the immutable mapd manifest and host artifact pass SHA-256, Linux ARM64 ELF, release/build marker, estimator, and capability checks;
- a reachable SSH profile reports the same clean branch/HEAD, `IsOffroad=1`, `IsOnroad=0`, and `MTSCLookaheadEnabled=0`;
- for rebuild only, `mapd.json`, prepared PBF, native Darwin generator, selected regions, bundled decoder, and durable set root are valid.

Rebuild generation happens during this no-device-mutation phase. The generator writes a clean temporary `offline/` tree with all six source-rounded `--phys-*` values. `CanonicalTileSetBuilder` hashes the PBF and every tile, decodes every file, verifies schema/bounds/finiteness/sigmoid identity/regions, writes `tile-set-manifest.json`, and moves the finished artifact to:

```text
~/Library/Application Support/vtsc_tuner/map_tiles/sets/<tile-set-id>/
├── tile-set-manifest.json
└── offline/...
```

An existing identical set is revalidated rather than overwritten. A partial or mixed tree never becomes a canonical artifact.

## Host and Git transaction

After preflight:

1. Save `current.tune.json` atomically, including exact display knobs and bands.
2. Patch controller constants, `common/params_keys.h` defaults, offroad-panel reset/ensure literals, and Q source as one authority set.
3. Run the complete production verification suite before commit or device mutation.
4. Commit only the files changed by source apply when the app owns the commit.
5. Push `HEAD:refs/heads/chauffeur-exp01` and require `git ls-remote` to return the exact HEAD.

If the source already matches, the commit step is a successful no-op and the exact-head checks still run.

## Device transaction

The app persists a rollback journal before remote mutation. It then:

1. Rechecks offroad/onroad/kill-switch state immediately before mutation.
2. Fetches `chauffeur-exp01`, requires the exact pushed object, and fast-forwards with `git merge --ff-only <exact-head>`.
3. Transfers the immutable mapd artifact to a temporary path outside the active binary, verifies SHA-256/ELF/markers and `--build-info`, seeds the identity-keyed persistent cache under `/data/media/0/osm/binaries/`, and atomically replaces the disposable checkout binary.
4. Writes and reads back all six physics Params with `tools/vtsc/apply_physics_params.py`; verifies the exact checked-in Q source and enable state.
5. If requested, transfers the canonical tile artifact only to an empty remote partial set, checks every manifest entry and digest on-device, installs one immutable generation at `/data/media/0/osm/tile-generations/<tile-set-id>/`, and atomically switches the active `offline` pointer while retaining the prior generation.
6. Rechecks parked/offroad/kill-switch state and sends one reboot only after every artifact is ready.
7. Waits for the tici and requires exact commit, Params/Q, release/cache/active binary, native mapd process, estimator, real-GPS profile, and tile identity.

Postflight validates `MapWholeCurveProfile` through the native Swift production parser: v3 version, freshness, positive curvature coefficients (including sub-1.0 shoulders), ranged baked speeds, sigmoid-bound fingerprint, events, spacing, and proximity to actual `LastGPSPosition`. It reports pending or invalid GPS distinctly; it never imports the controller, depends on device NumPy, or injects a fake route point.

## Failure and rollback

Failure before remote mutation leaves the tici untouched. Failure after mutation starts coherent rollback from the journal. Before rollback mutation—and again before a rollback reboot—the app freshly requires `IsOffroad=1`, `IsOnroad=0`, and `MTSCLookaheadEnabled=0`. Rollback restores the previous exact Git head, six Params, Q source identity, mapd release/version and binary, and previous tile-generation pointer when one was changed. A reboot is sent only when a deployment reboot occurred and rollback mutation completed.

Never report success at Git push, file transfer, or reboot dispatch. Success means the complete postflight identity passed.

## Source precision and Q contract

- A/B/C/D use six fractional digits; MIN/MAX use four.
- `Q_CURVE_ENABLED` is true only when at least one enabled band exists.
- Empty bands produce an empty `Q_CURVE_POINTS` list.
- Nonempty bands export 256 log-uniform curvature points with source-rounded curvature/multiplier values.
- The tici uses the source-rounded values, not higher-precision in-memory controls.

## SSH discovery

The native app uses `/usr/sbin/networksetup` only as an ordering hint, then probes `commaHome`, `commaCar`, and `commaAdb` with absolute `/usr/bin/ssh`, short timeouts, batch mode, and `StrictHostKeyChecking=accept-new`. The first responder is used. Finder's sparse `PATH` therefore does not change Git/SSH behavior.

## Legacy Rust note

`tools/vtsc_tuner/src/apply.rs` still documents its original background-thread step IDs, direct region generation, and two-reboot flow. That implementation is not the safe whole-curve release pipeline. Keep tune math/schema compatibility, but do not copy its direct active-tree rsync, unpinned `git pull`, or multiple-reboot sequencing into native production work.
