# Model Manager Version History

## Version Bumps (chronological)

| Date | Commit | CURRENT | REQUIRED_MIN | URL | Notes |
|------|--------|---------|--------------|-----|-------|
| 2025-08-01 | `40f298f03` | 8 | 8 | v6 | Upstream raised floor |
| 2025-08-17 | `2808faa39` | 9 | 9 | v7 | Our bump for Falling Phoenix, DTR, Space Lab |
| 2025-08-19 | `8ab00b7ba` | — | — | — | Fixed params caching (JSON encode/decode, int→str) |
| 2025-08-19 | `6b758a09e` | — | — | — | Robust download queue + logging |
| 2026-01-24 | `5563618c7` | 13 | 12 | v10 | Full upstream sunnypilot sync |
| 2026-01-30 | `20fadff55` | — | — | — | Model fetching + temporal input improvements |
| 2026-03-01 | `a17a38d8c` | 15 | 14 | v15 | Upstream sync #1749 (introduced offPolicy in JSON) |
| 2026-03-03 | `5a85c423d` | 15 | 14 | v15 | Our explicit bump + 3 manager.py fixes |
| 2026-03-03 | `1239afd2e` | — | — | — | Added `offPolicy @4` to capnp + resilient parser |
| 2026-03-03 | `6ff7cf6ad` | — | — | — | Added offPolicy progress bar to models_panel UI |

## Model Type Evolution

| capnp ordinal | Type | First appeared | JSON key |
|---------------|------|----------------|----------|
| @0 | supercombo | Original | `supercombo` |
| @1 | navigation | Original | `navigation` |
| @2 | vision | v7+ era | `vision` |
| @3 | policy | v7+ era | `policy` |
| @4 | offPolicy | v15 (Feb 2026) | `offPolicy` |

## Bundle Composition Evolution

- **Pre-v7**: Single `supercombo` model per bundle
- **v7–v14**: `vision` + `policy` (split architecture)
- **v15+**: `vision` + `policy` + `offPolicy` (three-artifact bundles, Feb 2026+)

## Bugs Fixed During Updates

### Params caching (2025-08-19, `8ab00b7ba`)
- `ModelCache.get()/set()` stored raw dict instead of JSON string
- `ModelManager_LastSyncTime` stored int directly instead of str
- `ModelRunnerTypeCache` stored int instead of str

### Download index truthiness (2026-03-03, `5a85c423d`)
- `if index_to_download :=` was falsy for index 0
- Fixed to `is not None`

### Parse resilience (2026-03-03, `1239afd2e`)
- `parse_models()` list comprehension let one bad bundle crash the entire list
- Changed to per-bundle try/except with cloudlog warning

### Off-policy runtime contract gap (2026-03-08, working tree)
- v15 bundles such as `OMV4` split runtime outputs across three artifacts:
  `vision`, `policy`, and `offPolicy`
- `plan`, `lane_lines`, `road_edges`, `lead`, and `lead_prob` can live only in
  `offPolicy`, while `planplus` can live only in `policy`
- `TinygradSplitRunner` must run + merge `offPolicy`, and
  `parse_model_outputs_split.py` must parse standalone `planplus`
- If only capnp/UI support is added, onroad can stay unhealthy with calibration
  stuck at 0% because `modeld_tinygrad` is missing required outputs
