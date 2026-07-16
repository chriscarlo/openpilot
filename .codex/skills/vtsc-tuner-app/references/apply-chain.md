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

- acquire the authoritative journal-directory global production-owner lock and durably publish this transaction's released-reader-unknown `preflightReserved` record as the first namespace action;
- prove no other exact VTSC Tuner process is alive before pruning or scanning, exclude only the exact live reservation from those operations, and reject every other unresolved journal;
- repeat peer-process and unresolved-journal checks immediately before the durable mutation claim, then freshly require the exact recorded tici boot/Git/Params/Q/mapd-build/cache/tile/parked baseline so a released old-first app cannot interleave or leave a stale rollback authority;
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
5. If requested, transfers the canonical tile artifact only to an empty remote partial set, checks every manifest entry and digest on-device, installs one immutable generation at `/data/media/0/osm/tile-generations/<tile-set-id>/`, and atomically switches the active `offline` pointer while retaining the prior generation. Before creating a generation or publishing a switch link, the helper durably records exact target/legacy/build/switch/retained ownership, whether each generation preexisted, and the content identity of preexisting generations and `offline.previous`. A direct legacy tree preserves a valid adjacent canonical identity or creates `legacy-*`; the app durably journals the exact prior ID and explicit `switched` / `notSwitched` result. Pre-exchange rollback revalidates and preserves preexisting generations, atomically renames only helper-owned trees to deterministic same-filesystem tombstones, and clears transaction authority only after tombstones are removed and parents synced; interrupted recursive deletion therefore replays safely, while older transactions lacking ownership authority fail closed. Same-target activation is a no-switch only when the host-recorded baseline matches and complete canonical or direct content equals the staged artifact. Snapshot/postflight accepts nil prior identity only as explicit clean `direct-unidentified` topology; canonical pointers must be safe relative `tile-generations/<id>/offline` links whose embedded manifest matches, and missing/broken/unsafe links or direct trees with pointer/transaction/generation/cleanup artifacts are invalid.
6. Rechecks parked/offroad/kill-switch state, captures a valid Linux `boot_id`, durably journals it with reboot intent, and only then sends one reboot after every artifact is ready.
7. Durably hands the transaction to `awaitingOutdoorPostflight`, releases mutation ownership, waits for the tici, and requires a different valid boot ID, stable parked-state brackets, and exact Git/tune/release/cache/build/manager/mapd/tile identity before reporting install success. A failed return/static proof reports failure, retains the pending journal, and never auto-rolls back. It does not claim controller acceptance from a raw parked profile. Every successful rollback restoration after `mutationInProgress`/`rollbackInProgress` always captures and durably journals a separate immediately-pre-rollback-reboot boot ID, sends a reboot, and requires a changed final ID through the static identity contract; liveMapData/profile/GPS remain exclusive to outdoor Resume. Only untouched `preflightReserved` cleanup is mutation- and reboot-free.

Final completion belongs only to **Resume Pending Outdoor Postflight**. Every tici read must remain at the journal's immutable deployed target HEAD or one exact descendant whose complete target-to-device diff is host-proven to touch only native tuner/skill paths. The clean local/origin tooling HEAD is separately recorded and merely bounds that ancestry; later production changes on the host do not authorize the tici at that later head. Completion records the exact device HEAD and compatibility paths. While ignition remains on after a normal GPS/profile-producing drive, Resume captures a fresh same-tune profile plus newly updated, valid `liveMapDataSP.roadGeometryValid=true` evidence. Only after the app explicitly reports that controller-ready capture does the user turn ignition off; the observer then requires stable offroad start/end brackets, unchanged static identity, and the same fully parsed profile before atomically completing the journal. New `awaitingOutdoorPostflight` journals must also prove their current boot differs from the durable deployment-preboot ID during onroad and final offroad validation. Only a true legacy serialized journal with raw missing `resolution`, missing boot field, incomplete+rebooted state, and a valid target retains the B174 compatibility path; an explicitly encoded modern lifecycle never does. It reports pending or invalid GPS distinctly and never injects a fake route point or mutates the car during proof.

## Failure and rollback

Failure before remote mutation leaves the tici untouched. Failure after mutation starts coherent rollback from the journal. Abort/Recover acquire global then per-journal ownership and, at every exclusion checkpoint, reject any peer tuner before pruning only proven-abandoned aged targetless legacy preflights; the selected journal is excluded by URL and identity, then every remaining foreign unresolved journal blocks. Those checks repeat immediately before claim and device mutation. The current checkout must equal the prior head, deployed target, or one exact completion-compatible descendant proven from the host repository; the remote Git shell and locked tile helper recheck that exact branch, HEAD, and empty worktree. Both Git-status reads propagate command failure explicitly; empty stdout from a failed `git status` cannot pass as clean. A target/tile precondition failure stops before all later Git/Params/mapd restoration. Before rollback mutation—and again before its mandatory reboot—the app freshly requires `IsOffroad=1`, `IsOnroad=0`, and `MTSCLookaheadEnabled=0`. Rollback restores the previous exact Git head, six Params, Q source identity, mapd release/version and binary, and previous tile-generation pointer only when the durable helper outcome says a switch occurred. A verified no-switch still permits the earlier source/Params/mapd legs to be restored, and postflight accepts active==previous==target only when that explicit lifecycle outcome proves no pointer exchange happened. Every claimed rollback that completes restoration sends and proves a fresh reboot, regardless of whether the deployment had recorded its own reboot.

Never report final deployment proof at Git push, file transfer, or reboot dispatch. The initial transaction reports **installed, outdoor proof pending** after its durable handoff; final success means the two-phase controller-ready outdoor postflight identity passed.

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
