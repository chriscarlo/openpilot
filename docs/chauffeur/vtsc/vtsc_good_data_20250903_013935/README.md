VTSC Good Data Drop

This folder contains a manifest and derived telemetry summaries from the most recent on-device rlogs to support MTSC/VTSC analysis.

Contents
- MANIFEST.json: files and metadata for the copied rlogs.
- rlog_rates.json: per-rlog message counts, median Hz, and P95 gaps for focus topics.
- logs/: local copies of 2–3 most recent rlogs (.zst). Note: .zst files are git-ignored to avoid pushing large binaries.

Device paths
- Original device logs: /data/media/0/realdata/*/*/rlog.zst
- Repo copies for convenience: /data/openpilot/$(echo "$TS_DIR" | sed 's|/data/openpilot/||')/logs/*.zst

Notes
- Use tools.lib.logreader LogReader to read .zst directly.
- If running analysis off-device, use rlog_rates.json, or pull the raw rlogs out-of-band.
