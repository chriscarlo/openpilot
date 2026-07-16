# VTSC Tuner macOS changelog

## 2026-07-16 — Replay-safe tile no-switch rollback

- Split immutable activation ownership from mutable phase state. A separately
  synced pre-creation authority binds every path, preexistence observation, and
  digest; activate resumes that intent while rollback alone performs cleanup.
- Re-sync every artifact parent and the tile root on settlement replay, even
  when a crash already removed the source, tombstone, or transaction record.
- Journal and verify exact tile topology plus content. Migrated legacy state
  distinguishes immutable container ID from preserved logical tile ID and
  binds both to target provenance and the computed tree digest. Dangling or
  non-regular manifests and canonical/direct same-ID drift fail closed.
- Re-prove a durable same-target no-switch under the parked/Git/tile lock and
  skip the exchange helper; changed topology stops all later rollback mutation.
- Added durable ownership/preexistence and content authority for every
  generation/build/switch/retained artifact. Pre-exchange recovery preserves
  exact preexisting generations and rejects missing or tampered authority.
- Replaced direct recursive cleanup with same-filesystem transaction tombstones.
  Interrupted deletion of helper-owned target, legacy, build, or retained trees
  now replays safely without revalidating partially deleted content.
- Added explicit canonical, direct-identified, and direct-unidentified snapshot
  topology. Canonical links require exact safe relative grammar and matching
  embedded identity; nil identity is valid only for a clean manifestless direct
  tree, never for a missing/broken path or a tree with transaction artifacts.
- Made canonical already-active activation a host-baseline-bound, complete-content
  verified no-switch with zero transaction or pointer mutation.
- Isolated the vision-only pipeline integration fixture from the host's
  persistent `MTSCLookaheadEnabled` value, so the complete randomized 202-test
  production gate cannot inherit a stale strategic-map cap.
- Verification: both pinned helper Go suites, mapd Go, 63 mapd Python tests plus
  11 subtests, 202 VTSC Python tests, and a true-clean 208 Core + 40 App Swift
  run pass. The exact Release app and nested decoder pass strict signing checks;
  the packaged static Linux ARM64 transaction helper matches its built SHA-256.
- Made pre-exchange tile rollback validate and durably remove every helper-owned
  generation, build, switch, temporary-link, and retained-tree artifact before
  clearing its target-bound transaction. A lost successful reply now replays as
  the same exact no-switch result instead of leaving ambiguous filesystem state.
- Added an explicit durable `switched` / `notSwitched` activation outcome to the
  deployment journal. Rollback postflight now distinguishes a real pointer swap
  from a proven same-target or pre-activation no-switch, including full
  source/Params/mapd restoration and fresh-process retry.
- Allowed a no-transaction rollback to prove an unchanged direct legacy tree
  from its exact recorded adjacent manifest identity, while rejecting missing,
  mismatched, or transaction-artifact evidence. The helper reports both active
  and previous IDs and Swift requires both to match the durable journal.
- Restricted snapshot adjacent-manifest fallback to a real, non-symlink direct
  `offline/` directory. Missing paths, broken pointers, and canonical generation
  pointers without a readable embedded manifest now fail closed.

## 2026-07-15 — Swift-hosted tici deployment

- Replaced the active tici deployment and rollback path's Python dependency
  with typed Swift command builders and structured snapshot/postflight parsing.
- Kept the tici boundary deliberately narrow: shell, Git, files, and processes,
  plus a bundled static ARM64 helper for the kernel-only atomic tile swap.
- Added durable mapd and tile transaction recovery so an interrupted deployment
  can be recovered before the next deployment begins.
