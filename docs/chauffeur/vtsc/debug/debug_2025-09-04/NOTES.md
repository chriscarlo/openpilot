**Summary**
- VTSC observed under-speed on straight, unoccluded city road (~16 mph in 25). User had to press accelerator; last section left to VTSC showed persistent crawl.
- Immediate live monitoring initially produced no data because VTSCDBG emission wasn’t active during the drive window we tried to watch.

**Root Causes Noted Today**
- Live watcher started before confirming VTSCDBG toggles were active in controller runtime → no VTSCDBG lines captured from that segment.
- Occlusion logic likely pretriggered or stayed sticky without freeway fail-open taking effect on straight/visible road.

**Changes Made Today**
- Controls (runtime behavior):
  - LKG anchoring and short ramp during occlusion.
  - Dropout grace to suppress pretrigger on brief model frame loss.
  - Vision-floor TTL uplift at occlusion onset (prevents crawl on straights for short TTL).
- Parameters (persistent; `common/params_keys.h`):
  - `VisionTurnSpeedControlVisionFloorTtlS`, `VisionTurnSpeedControlVisionFloorMult`, `VisionTurnSpeedControlDropoutGraceS`.
- Param wiring: loaded in `vision_turn_params.update_vtsc_params()` and consumed in `vision_turn_controller.py`.
- Tooling: added `tools/vtsc/vtsc_watch.py` (real-time watcher for VTSC).

**Verification/Wiring Checks**
- UI toggles for `VTSCVerboseDebug` and `VTSCWriteSnapshotFile` exist and map to Params.
- Controller polls `VTSCVerboseDebug` every ~2s and writes snapshots when enabled.
- New VTSC params are defined in `common/params_keys.h`, read in `vision_turn_params.py`, and used in controller logic.

**Next Steps**
- Capture a short clear-road drive with debug toggles ON and watcher running. Confirm `cap != occlusion` on straights and no `pretrigger_with_high_conf` flags.
- Post-drive: run `docs/chauffeur/vtsc/analysis/analyze_snapshots.py /data/media/0/VTSCDebug/vtsc_snapshots.jsonl --dump-tsv OUT.tsv`.
- If freeway crawl recurs, validate: `freeway_failopen_missed`, `psi_below_thresh`, and model dropout counts around the event.
