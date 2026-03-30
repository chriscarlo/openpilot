---
name: vtsc-tuner
description: >
  Capture and analyze Vision Turn Speed Control (VTSC) interventions, especially
  map/vision handoff misses and strategic-map timing problems. Use when tuning
  late or insufficient curve slowing, comparing advisory vs strategic map
  behavior, replaying pulled routes, or validating whether longitudinal planner
  actually reaches apex target speed before the anchor/apex.
---

# VTSC Tuner

## Overview

Use this skill for two related jobs:
- capture and package real VTSC interventions for offline RCA
- prove whether a proposed VTSC change actually fixes timing through the real planner path

Do not stop at controller-only outputs when the claim is about apex timing. `v_turn`
or `vtsc_cmd` can look better while the longitudinal planner still fails to get
`v_ego` down to target speed in time.
Do not start VTSC retuning until you have proved VTSC was actually the active
longitudinal cap. On this branch, Hyundai no-radar lead-follow bugs can mimic
"VTSC got weird" if you skip cap attribution first.

## Quick Start (Offline RCA Workbook)

Use this when you already have:
- `events_offline/<event_id>/event.json` with `route`, `seg`, and `t0` in monotonic seconds
- `realdata/<route>--<seg>/rlog.zst`

Run from repo root:
```bash
python tools/vtsc/vtsc_rca_workbook.py \
  --base /path/to/base_dir \
  --overwrite-traces
```

Outputs:
- `<base_dir>/vtsc_rca.xlsx`
- `<base_dir>/events_offline/<event_id>/trace_rlog_20s_plus.jsonl`

Column meanings: see `references/rca_columns.md`.
Live tuneable parameter semantics: see `references/live_tunable_param_glossary.md`.

If you only need gas-event triage, or the host cannot run the workbook path because
`pandas` is unavailable, use:
```bash
.venv/bin/python tools/vtsc/vtsc_gas_event_report.py \
  /path/to/base_dir \
  --out /path/to/base_dir/gas_event_report.tsv
```

This reads gas events from `events_offline/` and prefers
`trace_rlog_20s_plus.jsonl` when present. If traces are absent but
`gas_calibration_samples.tsv` exists under the same base dir, it falls back to
that replay output plus `realdata/<route>--<seg>/rlog.zst` to recover event-centered
windows without the workbook dependency.

Interpret the output labels as:
- `pressing_through_cap`: VTSC was still below ego at the gas press; this is a real
  "possibly too conservative" candidate worth deeper replay.
- `not_constraining`: VTSC was already above ego at the gas press; do not treat this
  as relax evidence.
- `tighten_bias`: low-speed calibration nudged the cap lower in the event window.
- `relax_candidate`: low-speed calibration nudged the cap upward while VTSC still
  constrained ego in the event window.

## Workflow Decision Tree

### A0) You are already in the car and need live capture immediately

Prefer the manager-owned recorder plus background watchers. This avoids duplicate
recorder processes and survives the normal onroad lifecycle better than manually
launching the recorder script yourself.

Use this exact bootstrap from the dev machine:
```bash
ssh commaCar 'bash -s' <<'EOF'
set -e
cd /data/openpilot
run_tag=$(date -u +%Y%m%dT%H%M%SZ)
live_root=/data/media/0/VTSCTuner/live
live_dir="$live_root/$run_tag"
mkdir -p "$live_dir"
ln -sfn "$live_dir" "$live_root/latest"
/usr/local/venv/bin/python3 - <<'PY'
from openpilot.common.params import Params
p = Params()
p.put_bool("VTSCInterventionRecorderEnabled", True)
print("VTSCInterventionRecorderEnabled=1")
PY
if ! pgrep -af 'tools.vtsc.vtsc_watch' >/dev/null; then
  nohup env PYTHONUNBUFFERED=1 /usr/local/venv/bin/python3 -m tools.vtsc.vtsc_watch --snapshots --swaglog > "$live_dir/vtsc_watch.log" 2>&1 < /dev/null &
  echo $! > "$live_dir/vtsc_watch.pid"
fi
if ! pgrep -af 'tools.vtsc.vtsc_stop_handoff_watch' >/dev/null; then
  nohup env PYTHONUNBUFFERED=1 /usr/local/venv/bin/python3 -m tools.vtsc.vtsc_stop_handoff_watch --hz 10 --only-alerts > "$live_dir/stop_handoff.log" 2>&1 < /dev/null &
  echo $! > "$live_dir/stop_handoff.pid"
fi
pgrep -af 'tools.vtsc.vtsc_watch|tools.vtsc.vtsc_stop_handoff_watch|vtsc_intervention_recorder'
echo "latest_dir=$live_dir"
EOF
```

Useful follow-up tails:
- `ssh commaCar 'tail -f /data/media/0/VTSCTuner/live/latest/vtsc_watch.log'`
- `ssh commaCar 'tail -f /data/media/0/VTSCTuner/live/latest/stop_handoff.log'`

Common startup pitfalls:
- In the car, prefer `commaCar`; `commaAdb` only works if the USB port-forward is active, and `commaHome` only works on the home Wi-Fi.
- `tools/vtsc/vtsc_live_params.py` requires a subcommand such as `list`, `set`, `apply`, or `watch`.
- If you background watchers with `nohup` and do not set `PYTHONUNBUFFERED=1`, the log files can look empty for too long and waste time.
- Repeated parked lines like `v=0.0 ... cap=map ... reason=short_vis` are expected while stopped and are not, by themselves, the anomaly you are looking for.
- Recorder run tags and most pulled route timestamps are UTC/Zulu. If the user
  reports a local event time, convert it before searching bundles or logs.

### A) You want always-on intervention capture for new drives

1. On tici, enable the manager-owned recorder:
- Set `VTSCInterventionRecorderEnabled=1`; manager starts `tools.vtsc.vtsc_intervention_recorder` onroad.
- Only launch `tools/vtsc/vtsc_intervention_recorder.py` manually if manager integration is unavailable.

2. Post-drive, pull bundles and logs to the dev machine:
- Event bundles: `/data/media/0/VTSCTuner/events/`
- Segments: `/data/media/0/realdata/<route>--<seg>/rlog.zst` and `qlog.zst`

3. Build the offline workbook:
- `python tools/vtsc/vtsc_rca_workbook.py --base ...`

### B) You already have route cache, but no event bundles

1. Create `events_offline/` bundles by scanning qlogs or by hand-picking timestamps.
2. Run `tools/vtsc/vtsc_rca_workbook.py`.

### C) You need replay or synthetic validation

Pick the narrowest layer that can falsify the claim:

- Controller-only replay:
  use `tools/vtsc/vtsc_rlog_episode_report.py` or `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_regression_rca_events.py`
  when the question is about cap flapping, vision confidence, or map-to-vision
  arbitration.
- VTSC helper/snapshot tests:
  use `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`
  when the question is advisory vs strategic map selection, release logic,
  synthetic map profiles, or parameter semantics.
- Planner-backed timing tests:
  use `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_map_timing.py`
  when the question is "did the car actually reach target speed before the anchor/apex?"

If the claim is about hitting apex speed in time, controller-only replay is not enough.
If the claim is really "why did the car surge, pulse, or follow a lead badly,"
and VTSC was not the active cap, stop here and switch to
`openpilot-longitudinal-tuner`.

## Current VTSC Map Strategy Facts (Code-Verified)

Use these as ground truth for the current branch, not the older advisory-only model.

- VTSC now has a runtime `VTSCMapStrategy` with `advisory` and `strategic`.
- Default map mode is `strategic`.
- Both modes live in the same controller. There is no forked `vision_turn_controller.py`.
- `mapd` still provides geometry-derived `MapCurvatures` and `MapTargetVelocities`, but VTSC speed control still uses `MapCurvatures` plus `LastGPSPosition`.
- `mapd` chained lookahead is about `500 m`, not "a mile."
- Longitudinal planning remains a speed-cap minimum. VTSC still does not command braking directly.

### Advisory Mode

- Map cap is an advisory tail beyond visible horizon.
- Severe or lost visibility starts map influence near `vis_margin_m` only.
- Partial visibility blends the advisory start distance between `vis_margin_m` and
  `v_ego * vis_horizon_s + vis_margin_m` using current-frame vision confidence.
- Map is no longer suppressed merely because vision has some turn evidence.
- Advisory map only yields when visibility is full, vision sees the turn, and
  vision is already at least as conservative as the map cap.

### Strategic Mode

- Strategic mode computes a persistent map floor from future `MapCurvatures`
  starting at ego, not just beyond visible horizon.
- Vision can tighten below the floor immediately.
- Vision can relax above the floor only after a real release condition:
  takeover dwell, counterevidence dwell, or post-apex release.
- Strategic mode tracks the limiting map anchor and exposes its distance and
  curvature in telemetry.
- Strategic mode uses the shared planner-response helper when available through
  `set_longitudinal_response_model(...)`.
- Strategic mode does not rely on VTSC smoothing `_max_decel` for reachability
  when that shared response model is present.

### Driver-Facing Timing Knobs

These semantics now apply to both advisory and strategic analysis:

- `VisionTurnSpeedControlFixedLeadTimeSeconds` is the time-to-anchor knob.
  It means "be at the anchor speed before the anchor by N seconds."
- `VisionTurnSpeedControlCurvePhaseOffsetS` shifts nominal curve timing earlier or later.
- `VisionTurnSpeedControlOvershootPhaseOffsetS` biases tighter-anchor timing earlier
  or later when the anchor is materially tighter than the current local target.
- `VisionTurnSpeedControlApexExitPhaseOffsetS` affects post-apex release timing.

If you are asking "be there 1 s, 2 s, or 3 s before the anchor," the correct knob
to sweep is `FixedLeadTimeSeconds`, not `CurvePhaseOffsetS`.

## Longitudinal Planner Coupling (What Changed)

Strategic map timing is now planner-informed instead of VTSC-internal fiction.

- Shared helper: `selfdrive/controls/lib/longitudinal_response_model.py`
- Onroad path: `longitudinal_planner.py` builds the response model and injects it into VTSC.
- Strategic map uses that shared helper to estimate the cruise cap needed for the
  planner to deliver the required average decel before the anchor.
- The helper is still a model of planner response, not the entire closed-loop vehicle.
- That is why planner-backed tests are still required for apex-timing claims.
- Real March 2026 planner dropouts on strategic VTSC were proven to be helper-dominated,
  not preview-generation dominated. In affected captures, `helper_total_ms` dominated
  bad planner cycles while `strategic_outside_helper_ms` and `preview_total_ms` stayed small.
- The helper hot path now uses a specialized scalar implementation of the sampled
  clipped-cruise math instead of rebuilding full profile arrays and gradients for every probe.
  The intent is CPU reduction with output parity, not behavior change.

### Planner-Parity Maintenance Footguns

- Treat `selfdrive/controls/lib/longitudinal_response_model.py` as a compatibility surface,
  not just an optimization sandbox. It now contains hand-optimized math that is meant to
  stay behaviorally aligned with planner cruise-profile sampling.
- If future planner logic changes in any of these areas, re-check VTSC helper parity:
  - cruise-profile clipping
  - `get_accel_from_plan(...)` semantics
  - action horizon / actuator delay handling
  - planner accel-limit clipping
  - response-model parameter construction in `longitudinal_planner.py` / MPC
- Potential future planner logic changes may require separate maintenance to the VTSC
  replay/report scripts to maintain compatibility and parity. Do not assume a controller-only
  VTSC script stays correct automatically when planner internals change.
- If you change planner response semantics, update both:
  - the fast helper path in `selfdrive/controls/lib/longitudinal_response_model.py`
  - the VTSC replay/report path that injects a default response model
- After any planner-response change, rerun parity tests before trusting RCA conclusions.

## Replay and RCA Caveats

- `snapshot_debug_state()['final']` is not always the true planner-facing VTSC output.
- `_dbg_target_final` is captured before some later arbitration and output-path logic.
- Prefer `vtsc_cmd` or `v_turn` when comparing what VTSC actually handed to the planner.
- For strategy analysis, also inspect:
  - `strategy_mode`
  - `strategy_state`
  - `map_advisory_cap`
  - `map_strategic_cap`
  - `vision_local_cap`
  - `selected_cap`
  - `map_floor_active`
  - `map_floor_reason`
  - `vision_relax_allowed`
  - `vision_relax_reason`
  - `map_floor_anchor_dist_m`
  - `map_floor_anchor_k`
  - `planner_response_decel_mps2`
  - `planner_response_delay_s`

Important offline limitation:
- A saved RCA JSON or controller fixture that lacks GPS and `MapCurvatures`
  cannot exercise the strategic map path, even if it now injects the shared
  planner-response helper.
- Use those fixtures for controller/handoff regressions only.
- Use pulled route segments with map inputs, or synthetic map profiles, to validate strategic behavior.
- `tools/vtsc/vtsc_rlog_episode_report.py` and similar controller-only replay paths inject a
  default shared response model, but they do not execute the full planner loop and do not
  reproduce planner-local CPU starvation by themselves.
- If planner-side math changes and the VTSC script is not updated to match, offline replay can
  preserve the old helper behavior and silently drift from onroad planner behavior.
- For planner-drop or `PLN` RCA, prefer planner-local lag bundles when available:
  - `/data/media/0/VTSCDebug/planner_lag_events/`
  - Compare `helper_total_ms`, `strategic_outside_helper_ms`, and `preview_total_ms`
  - Repeated high `helper.predict_average_decel_for_cruise_cap` counts in bad cycles are a strong
    sign of shared-helper overload, not preview or branch-stub overhead
- Do not over-index on branch-stub signals for planner performance RCA. Strategic/helper overhead
  can remain the real problem even when:
  - `curvePreviewBranchStubs == 0`
  - `roadGeometryValid == false`
  - `nearbyRoadSegments == 0`

## Recommended Verification Commands

Use exact commands rather than vague "replay it somehow."

### Offline RCA and before/after route comparison

```bash
python tools/vtsc/vtsc_rlog_episode_report.py .cache/commaCar/<route_dir>
```

This script now injects a default shared response model so controller-only replays
stay aligned with planner-side strategic assumptions as much as they can without
standing up the full planner.

### VTSC helper and synthetic map tests

```bash
.venv/bin/python -m pytest --noconftest -o addopts='' \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py -q
```

Use this file for:
- advisory vs strategic comparison
- hidden-apex synthetic profiles
- phase-offset and fixed-lead semantics
- map-floor release logic
- snapshot field regressions

### Shared helper parity tests

```bash
.venv/bin/python -m pytest --noconftest -o addopts='' \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_response_model.py -q
```

Use this file when you change:
- shared planner-response helper math
- strategic response probing behavior
- any optimization inside `longitudinal_response_model.py`

### Planner-backed apex timing proof

```bash
.venv/bin/python -m pytest --noconftest -o addopts='' \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_map_timing.py -q
```

This is the right test when the requirement is:
- hit target speed by the anchor/apex
- do so across a fixed-lead-time sweep
- prove larger lead times only move target attainment earlier

### Synthetic advisory-vs-strategic divergence sweep

```bash
.venv/bin/python tools/vtsc/sweep_vtsc_map_profiles.py
```

Use this to sweep blind hidden-apex profiles and see where `strategic` materially
tightens the cap relative to `advisory`.

## Diagnostic Heuristics

- If the event did not have VTSC as the active visible cap, do not tune VTSC on
  that evidence. Route the RCA to the actual longitudinal limiter first.
- If a real intervention shows map did see the curve, do not misclassify it as
  "map never saw it." The next question is whether map handed off too early.
- If the zero-lead planner-backed case still misses the target, the problem is
  bigger than offset tuning. Fix the baseline response model or arbitration first.
- If strategic map appears to work in `snapshot_debug_state()` but planner-backed
  `v_ego` still misses the anchor, trust the planner-backed result.
- Treat brake interventions as likely timing, visibility, or handoff problems
  first. Treat gas interventions as "possibly too conservative," but run
  `tools/vtsc/vtsc_gas_event_report.py` first and only escalate events labeled
  `pressing_through_cap` before retuning physics.
- A gas event labeled `not_constraining` is not relax evidence. In those cases,
  the driver pressed while VTSC was already above ego, so the event says more about
  driver intent than about an overly low VTSC cap.

More detailed heuristics: see `references/tuning_heuristics.md`.

## Live-Tuneable Parameter Glossary (Plain English)

When the user asks what a VTSC or HUD tuning knob actually does, use:
- `references/live_tunable_param_glossary.md`

This glossary remains the source of truth for plain-English translations of:
- `VTSCExpertModeEnabled`
- all `VTSC.Expert.*` live-tuneable keys
- all `VTSCHUD.*` live-tuneable keys

## Skill Maintenance

After any real VTSC RCA or tuning session:
- compare new evidence against existing guidance before finishing
- correct stale bullets instead of appending contradictory notes
- keep exact file paths and verification commands current
- prefer one clear proven failure mode and one reliable validation command over speculative lists
- add new gotchas only for real, repeatable footguns
- keep only durable workflow guidance in `SKILL.md`; move detailed examples or reference material to `references/` or `scripts/`
- update `agents/openai.yaml` only if the skill’s scope or trigger wording changed materially

Treat this skill as a recursive kaizen loop:
- every real invocation should leave it more accurate, more actionable, more compact, or all three
- reconcile new facts with old guidance in the same pass so the next invocation starts from one coherent workflow
- if new evidence proves an older bullet wrong, incomplete, or redundant, replace, tighten, or delete it instead of stacking another warning
- prefer editing and pruning over appending; a shorter, sharper skill beats a longer noisier one
