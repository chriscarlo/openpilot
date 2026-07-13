# Trouble ticket: VTSC strategic-map false slowdown on a straight surface street

Status: open; intentionally deferred from the 2026-07-12 longitudinal-fix session.

## Summary

On the 2023 CAN-FD HDA2 Kia EV6, VTSC's strategic map path slowed the car from about 45 mph toward 28 mph on a fast surface street. A motorcycle turned into the adjacent number-2 lane during the intervention, which initially made the event look like false adjacent-lead control. Synchronized planner telemetry proves the motorcycle was coincidental: VTSC had already begun the slowdown about five seconds before the motorcycle appeared.

The driver pressed the accelerator at segment-23 offset about 56.7 seconds. No VTSC behavior is changed by this ticket.

## Environment and capture

- Branch/device commit: `chauffeur-exp01` at `f7c35c881` (`Fix EV6 position-collapse governor loop`).
- Route: `0000022f--c1a32d3b98`.
- Primary window: segment 23, qcamera offsets 44-57 seconds; the motorcycle enters the number-2 lane at offsets 49-51 seconds.
- Visual continuation: beginning of segment 24.
- Raw rlogs and qcamera videos are intentionally untracked under:
  `.cache/doc_artifacts/longitudinal/20260712T191742-f7c35c881-tici-adjacent-lane-control/`
- Human-viewable videos:
  `segment-23-qcamera-full-60s.mp4` and `segment-24-qcamera-full-60s.mp4`.

## Causal evidence

The onroad `VTSCDBG` records `active_cap="map"`, `strategy_mode="strategic"`, and `map_floor_active=true` throughout the slowdown. SLC remained non-binding at 26.15 m/s (58.5 mph).

| Segment-23 offset | Ego speed | VTSC command | Strategic anchor | Planner result |
| ---: | ---: | ---: | ---: | --- |
| 44.13 s | 20.11 m/s | 23.76 m/s | 289 m, 0.02137 1/m | No VTSC braking yet |
| 44.64 s | 20.16 m/s | 17.94 m/s | 247 m, 0.02049 1/m | Slowdown begins |
| 48.39 s | 18.08 m/s | 15.57 m/s | 181 m, 0.02049 1/m | Planner source becomes `cruise` |
| 49.50 s | 17.53 m/s | 11.07 m/s | 157 m, 0.02137 1/m | Motorcycle appears; VTSC was already active |
| 54.64 s | 13.49 m/s | 10.45 m/s | 67 m, 0.02137 1/m | Strategic cap remains binding |
| 56.74 s | 12.44 m/s | 11.09 m/s | 60 m, 0.02137 1/m | Driver accelerator override |

Lead telemetry does not support adjacent-lane control in this window: `radarState` publishes duplicated centered hypotheses for the visible in-lane lead, while the planner's negative target follows the VTSC map command after the source changes to `cruise`.

## State and tooling defects exposed

1. `longitudinalPlanSP.visionTurnSpeedControl.state` remains `disabled` while its published `velocity` is actively constraining the planner and `VTSCDBG.active_cap` is `map`. This makes HUD/telemetry attribution misleading.
2. The standard offline episode replay currently misses the onroad strategic intervention. It reports zero episodes and replays a 26.67 m/s visible cap through the event instead of the recorded 10.45-17.94 m/s strategic commands. Do not use that replay result to dismiss this capture.

Reproduction command:

```bash
.venv/bin/python tools/vtsc/vtsc_rlog_episode_report.py \
  --summary \
  --out .cache/doc_artifacts/longitudinal/20260712T191742-f7c35c881-tici-adjacent-lane-control/segment23-vtsc-episode-summary.tsv \
  --samples-out .cache/doc_artifacts/longitudinal/20260712T191742-f7c35c881-tici-adjacent-lane-control/segment23-vtsc-replay-samples.tsv \
  .cache/doc_artifacts/longitudinal/20260712T191742-f7c35c881-tici-adjacent-lane-control/0000022f--c1a32d3b98--23--rlog.zst
```

Expected current output: `n_ep_before=0`, `n_ep_after=0`; the sample file shows `before_vtsc_cmd=26.6667` in the road window. That output conflicts with the recorded onroad `VTSCDBG` commands above.

## Follow-up acceptance criteria

- Replay segment 23 with the recorded GPS and `MapCurvatures` inputs and reproduce the onroad strategic map command before tuning any constants.
- Identify why the tight strategic anchor (curvature about 0.02137 1/m) was treated as the driven path. Check branch/cross-street contamination and map-path selection before changing the curvature-to-speed mapping.
- Make the published VTSC state/HUD attribution reflect planner authority whenever VTSC velocity is the binding cruise cap.
- Add a route-backed regression for this exact window and a safety twin with a real driven-path curve.
- Prove the fix through `test_scenarios.py` and the planner-backed map-timing tests; preserve timely slowing for genuine tight curves.
