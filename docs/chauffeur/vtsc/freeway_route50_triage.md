# VTSC Freeway Crawl Triage — Route 50 (Sep 3, 6–6:45 pm PDT)

This report summarizes on‑device VTSC behavior during the reported freeway crawl window. It uses segment rlogs under `/data/media/0/realdata/0000007a--1f95ff2406--*/rlog.zst` and the VTSC debug snapshots logged as `VTSCDBG …`.

## Scope & Method
- Time window: last ~6 hours from now (covers 6–6:45 pm PDT) across 132 segments in `0000007a--1f95ff2406`.
- Parser: `tools/lib/logreader.py`; extracted outer `swaglog` JSON, then parsed the `msg` payload beginning with `VTSCDBG { … }`.
- Counted occlusion vs full visibility, positive‑margin occurrences, high‑speed samples (≥ ~50 mph; cruise ≥ 22 m/s), and minima for `v_occ`, `v_vis`, and `final`.
- “Route 50” hits were detected via UI/RTI log messages within the same directory.

## Key Findings
- Occlusion almost always active: 6,687/6,720 VTSC snapshot lines flagged vision status not FULL (~99.5%). At ≥ ~50 mph, 5,494/5,522 (~99.5%) were still not FULL.
- Positive margin almost never true: 9 true vs 6,711 false — the occlusion barrier was effectively “on” with no distance margin most of the time.
- Minimum occlusion speed cap observed: `v_occ ≈ 2.90 m/s` (~6.5 mph). This matches the reported “drops to ~10 mph” floor under occlusion.
- “Route 50” confirmed in this dataset (e.g., segment 26): RTI/UI messages include “Current road name: Route 50” and “drawing road name banner: Route 50”.
- Snapshot confidence during the freeway window was often near zero (e.g., `conf ~ 0.003–0.004`), with `vision_status: SEVERE` and large tail fractions (`tail_frac ≈ 0.9`), indicating occlusion logic dominated.

## Interpretation
- With confidence stuck low, the occlusion model remained engaged on straight freeway. Its tail growth (`gamma_per_m`) projected curvature, lowering `v_occ` (and therefore the effective cap) even when the path was likely straight and visible.
- The near‑zero positive margin indicates the barrier believed the available visible distance was insufficient to maintain current speed (i.e., “always in a deficit”), pushing VTSC toward a low cap.
- Together, these strongly suggest A) occlusion onset/gating is over‑triggered (vision/fov), and/or B) visibility horizon/geometry feeding the barrier is mis‑scaled in this environment.

## Minimal Checks (on device)
1) Confirm snapshots during freeway: grep VTSCDBG and note `vision_status`, `occl_positive_margin`, `v_occ`, `v_vis`. Expect FULL with positive margin on open freeway.
2) Enable param safety valve for quick A/B: set `VTSCFailOpen=true` (temporarily). Freeway crawl should disappear if occlusion was the cause.
3) With the latest build, also log `active_cap`, `vtsc_cmd`, `fail_open` (new snapshot fields) to see which cap won and if the freeway guard engaged.

## Representative VTSCDBG Lines (redacted)
```
{"msg": "VTSCDBG {\"v\":0.0,\"cruise\":33.06,\"lead\":true,\"hw\":33.75,\"conf\":0.0035,\"vision_status\":\"SEVERE\",\"k_model\":0.00058,\"k_occ\":0.00951,\"k_vis_last\":0.00259,\"v_base\":33.06,\"v_occ\":14.39,\"v_vis\":34.69,\"raw\":0.0,\"final\":0.0,\"occl_positive_margin\":false,\"tail_frac\":0.90,\"s_tail\":280.49,\"comfort_decel\":-1.47,\"max_adaptive_decel\":-6.00,\"decel_cmd\":0.0,\"a_cmd\":0.0}", "level": "DEBUG"}
{"msg": "VTSCDBG {\"v\":0.0,\"cruise\":33.06,\"lead\":true,\"hw\":33.98,\"conf\":0.0036,\"vision_status\":\"SEVERE\",\"k_model\":0.00068,\"k_occ\":0.00951,\"k_vis_last\":0.00259,\"v_base\":33.06,\"v_occ\":14.39,\"v_vis\":34.69,\"raw\":0.0,\"final\":0.0,\"occl_positive_margin\":false,\"tail_frac\":0.90,\"s_tail\":280.49}", "level": "DEBUG"}
```

## Commands Used (repeatable)
- Enumerate recent segments:
  - `ls -lt /data/media/0/realdata | head -n 30`
- Parse VTSCDBG summaries (last ~6 hours):
  - Python: `tools/lib/logreader.py` over `/data/media/0/realdata/0000007a--1f95ff2406--*/rlog.zst`, JSON‑parse `swaglog` → split `msg` on `VTSCDBG ` → parse inner JSON.

## Hotfix & Next Steps
- Hotfix in code: freeway fail‑open guard added (ignore occlusion when straight + long visibility + good confidence); snapshot now publishes `active_cap`, `vtsc_cmd`, `fail_open`, and `cap_*_vmin`.
- Immediate validation: quick freeway drive with `VTSCVerboseDebug=true` to confirm `active_cap != occlusion` and `fail_open=true` on straight freeway.
- Root cause drill‑down: fix occlusion onset/gating — verify confidence source (laneLineProbs), FOV geometry/s_visible_m, and unit scales feeding the barrier.
- Regression tests: focus on hidden‑turn integration tests; ensure freeway no‑crawl stays green and occlusion still decelerates on genuine hidden turns.

