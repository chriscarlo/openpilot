# VTSC Tuner macOS changelog

## 2026-07-15 — Swift-hosted tici deployment

- Replaced the active tici deployment and rollback path's Python dependency
  with typed Swift command builders and structured snapshot/postflight parsing.
- Kept the tici boundary deliberately narrow: shell, Git, files, and processes,
  plus a bundled static ARM64 helper for the kernel-only atomic tile swap.
- Added durable mapd and tile transaction recovery so an interrupted deployment
  can be recovered before the next deployment begins.
