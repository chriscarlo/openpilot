# VTSC Tuner macOS changelog

## 2026-07-16 — Replay-safe tile no-switch rollback

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
