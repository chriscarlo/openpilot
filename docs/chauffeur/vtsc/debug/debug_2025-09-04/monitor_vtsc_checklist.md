**Monitor VTSC — On-Device Checklist**

- Ensure toggles: In Offroad → Cruise → VTSC → Settings
  - `Verbose VTSC Debug Logging` (param `VTSCVerboseDebug`) → ON
  - `Write Onroad VTSC Snapshots` (param `VTSCWriteSnapshotFile`) → ON

- Start watcher (one-time per session)
  - `nohup python3 tools/vtsc/vtsc_watch.py > .cache/vtsc_watch.out 2>&1 & echo $! > .cache/vtsc_watch.pid`
  - Follow output: `tail -f .cache/vtsc_watch.out`

- Expect live fields per line
  - Speeds: `v`, `v_base`, `v_vis`, `v_occ`, `final`
  - Caps: `cap=visible|occlusion`, `cap_visible_vmin`, `cap_occl_vmin`
  - Gating: `conf`, `psi/psi_thresh`, `reason`
  - Flags: `freeway_failopen_missed`, `pretrigger_with_high_conf`, `double_occl_cap_suspect`, `psi_below_thresh`

- If no output appears
  - Verify params: `python3 - <<'PY'\nfrom openpilot.common.params_pyx import Params; p=Params();\nprint('VTSCVerboseDebug=',p.get_bool('VTSCVerboseDebug'),' VTSCWriteSnapshotFile=',p.get_bool('VTSCWriteSnapshotFile'))\nPY`
  - Check snapshot file presence: `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl`
  - Confirm process: `ps | rg vtsc_watch.py` and PID file `.cache/vtsc_watch.pid`

- Post-drive quick analysis
  - `python docs/chauffeur/vtsc/analysis/analyze_snapshots.py /data/media/0/VTSCDebug/vtsc_snapshots.jsonl --dump-tsv OUT.tsv`
  - Inspect summary for LKG “floor saves” and gating anomalies.

- Optional runtime safety knobs (persisted)
  - TTL `VisionTurnSpeedControlVisionFloorTtlS`, mult `VisionTurnSpeedControlVisionFloorMult`
  - Dropout grace: `VisionTurnSpeedControlDropoutGraceS`

- Cleanup
  - Stop watcher: `kill "$(cat .cache/vtsc_watch.pid)"`
  - Optionally disable debug toggles in UI to reduce log noise.
