**VTSC Debug Docs Structure**
- **Root:** `docs/chauffeur/vtsc/debug/`
- **Daily Folders:** `debug_YYYY-MM-DD` (or `debug_YYYYMMDD_HHMM` for multiple sessions in a day)
- **Per-Day Contents:**
  - `NOTES.md`: summary, issues found, changelog (what files changed and why), next steps, and any results.
  - `monitor_vtsc_checklist.md`: step-by-step on-device monitoring checklist.
  - `artifacts/`: optional subfolder for copied snippets, small TSVs/plots, watcher output samples.
  - Links to routes/segments reviewed (paths under `/data/media/0/realdata/...`).

**Conventions**
- Keep entries concise and time-stamped; prefer links/paths over pasted logs.
- Place ad-hoc scripts under `tools/vtsc/` and reference them in the notes.
- When asked to “monitor VTSC debugging”, use the latest `monitor_vtsc_checklist.md` in the most recent daily folder.

**Key On‑Device Sources**
- Snapshots (if enabled): `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl`
- Swaglogs (rotating): `/data/log/swaglog.*`
- Watcher: `tools/vtsc/vtsc_watch.py` → prints compact VTSC lines in real time.

