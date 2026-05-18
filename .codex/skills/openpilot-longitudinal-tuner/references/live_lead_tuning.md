# Live Lead Tuning

## Scope

- Covers two distinct live-tune surfaces in `LongitudinalMpc` / planner:
  - **ACC lead-response heuristics**:
    preview for a newly recognized slower lead, safe gap reclaim when a lead
    pulls away, and benign cut-in settle.
  - **Source-stability layer**:
    four cooperating mechanisms that kill or absorb flicker between
    `cruise` and `lead` at the edge of lead acquisition. Added after
    observing repeated rapid source flips on the Kia EV6 vision-only path.
- It is not the tuning surface for duplicate-lead collapse, low-speed queue
  hold, cruise-owned lead accel taper, or the reclaim-lead optimism release.
  Those remain fixed code in `LongitudinalMpc`.
- It does not change `forceDecel` behavior. That path is still the normal
  planner stop request used for DM / soft-disable handling.
- It is assistant-oriented in v1. There is no offroad UI for these knobs.

## Runtime Path

- Param keys live in `common/params_keys.h` under the `Longitudinal.LiveTune.*`
  namespace.
- The shared metadata and clamping logic live in
  `selfdrive/controls/lib/longitudinal_live_tune.py`.
- `LongitudinalMpc` refreshes those params every `0.5 s` inside
  `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`.
- The live config feeds four heuristics:
  `get_lead_approach_preview_buffer()`,
  `get_gap_reclaim_accel_floor()`, the cut-in settle decel cap applied
  after MPC, and the source-stability layer (Schmitt trigger in `radard`,
  dwell+phantom filter in `LongitudinalMpc._stabilize_raw_leads`, and jerk
  clamps in `LongitudinalPlanner._apply_cruise_reacquire_jerk_limit` and
  `_apply_flutter_mode_clamp`).
- Hyundai-only duplicate-lead stabilization and ACC source hysteresis also live
  in `LongitudinalMpc`, along with a filtered virtual lead used for stable
  lead-vs-cruise ownership. Those Hyundai source-stability paths are fixed code
  in v1, not live knobs.
- The current Hyundai fixed-code layer also includes:
  low-speed queue hold for slow close leads,
  a cruise-owned lead accel cap when a valid lead still exists,
  and a reclaim-lead state that drops optimistic pull-away dynamics quickly
  once the real lead stops pulling away.
- On Hyundai EVs, final accel and jerk still pass through the separate
  `opendbc/sunnypilot/car/hyundai/longitudinal/controller.py` overlay, which
  already has EV-specific shaping from
  `opendbc/sunnypilot/car/hyundai/longitudinal/config.py`.

## Knobs

- `LeadPreviewStrength`
  Scale how early the planner starts backing out of throttle for a newly
  recognized slower lead that is still outside nominal headway.
- `LeadPreviewGapMinM`
  Minimum extra slack above nominal headway before the preview path can engage.
- `LeadPreviewMaxBufferM`
  Hard cap on how much the preview logic can pull the lead obstacle closer.
- `GapReclaimStrength`
  Scale how eagerly ACC closes a safe extra gap when the lead is pulling away.
- `GapReclaimGapMinM`
  Minimum extra slack above nominal headway before reclaim is allowed.
- `GapReclaimMaxAccel`
  Hard cap on the positive accel floor used for safe gap reclaim.
- `CutInSettleDurationS`
  How long a benign cut-in gets a gradual headway-recovery grace window.
- `CutInSettleMaxDecel`
  Strongest braking the planner is allowed to ask for during that grace window.
- `CutInSettleMaxClosingSpeedMps`
  Highest closing speed that still counts as a benign cut-in rather than a
  situation that should brake normally.

## Source Stability Layer

Four cooperating mechanisms sit between the model's raw lead output and the
MPC's source decision. Each can be disabled individually. Defaults now match
the May 16, 2026 Kia EV6 freeway tune: keep lead ownership sticky, absorb
brief model dropouts, and heavily damp source-transition accel flaps.

### 1 — Prob Schmitt trigger (radard, always-on by default)

Asymmetric hysteresis on `leadV3[i].prob` inside
`selfdrive/controls/radard.py::get_lead()`. Replaces the legacy
`prob > .5` single threshold with per-slot latch state: must cross
`Enter` to latch on, falls below `Exit` to release. Directly kills
acquisition-edge flicker where prob hovers ±0.02 around 0.5.

- `LeadProbEnter` (default 0.6, range 0.0-1.0)
  Prob required to latch a slot on. Raise to reject flicker harder.
- `LeadProbExit` (default 0.25, range 0.0-1.0)
  Prob below which a latched slot releases. Must be <= Enter for
  hysteresis; setting both to 0.5 collapses to legacy behavior.

### 2 — Source acquire/release dwell (MPC, on by default for release)

Frame-count hysteresis at the MPC boundary. Implemented in
`LongitudinalMpc._stabilize_raw_leads()`. Requires N consecutive valid-lead
frames before the MPC can treat the lead as present, and M consecutive
invalid-lead frames before releasing a latched lead.

- `LeadSourceAcquireFrames` (default 1, range 1-20)
  Default 1 = no dwell. Raise to 2-3 to suppress one-frame false
  acquisitions that sneak past Schmitt. Each frame is `DT_MDL = 0.05 s`.
- `LeadSourceReleaseFrames` (default 20, range 1-40)
  Holds through brief dropouts before releasing a latched lead.
  While phantom hold (#3) is active and within its window, release dwell
  is effectively subsumed by phantom: the slot stays latched regardless
  of raw invalid streak until the phantom window expires or yRel kill
  fires.

### 3 — Phantom lead hold (MPC, on by default)

Velocity-extrapolated lead state fed to the MPC during brief raw dropouts.
Same implementation path as #2 (`_stabilize_raw_leads`). When raw goes
invalid and preconditions are met, a synthesized `_StabilizedLead` is
published in place with `dRel` extrapolated by last-known `vRel`,
`aLeadK`/`modelProb` decaying linearly to zero by end of window.

- `PhantomLeadHoldS` (default 0.80, range 0.0-1.5)
  Duration the phantom persists after raw `status` flips to False.
  Set 0 to disable phantom.
- `PhantomLeadStableFrames` (default 3, range 1-40)
  The lead must have been latched for at least this many frames before
  it is eligible for phantom hold. Prevents one-frame false acquisitions
  from generating a persistent ghost.
- `LEAD_STABILIZER_PHANTOM_YREL_KILL_M` (non-tunable, 1.75 m)
  Safety. If a new raw lead appears in the OTHER slot at a yRel more
  than this far from the phantom's last-known yRel, the phantom is
  dropped immediately. Prevents continuing to track an exited car when a
  different car has taken its place.

### 4 — Jerk-rate clamps at the planner (LongitudinalPlanner, mostly on)

Two cooperating slew-rate limits on `output_a_target`. Both live in
`selfdrive/controls/lib/longitudinal_planner.py` and read from the same
`LeadResponseTuningConfig`.

#### Cruise-reacquire jerk limit (always-on by default)

Triggered on a lead->cruise source transition. Clamps POSITIVE slew of
`output_a_target` only; braking and steady lead-follow are unaffected.
Window auto-closes early when `output_a_target` reaches the cruise accel
cap.

- `CruiseReacquirePosJerkLimit` (default 0.08, range 0.0-5.0 m/s^3)
  Max upward jerk on planner output. 0 disables the mechanism.
- `CruiseReacquireJerkWindowS` (default 3.0, range 0.0-3.0 s)
  Duration after transition during which the limit is enforced. 0 also
  disables.

#### Flutter-mode clamp (always-on by default)

Triggered when repeated source transitions happen inside a rolling
window. Once triggered, clamps jerk in BOTH directions until the flutter
settles. Bypasses when `modelAccel` strongly disagrees with the clamp,
so real braking is never delayed.

- `FlutterDetectTransitions` (default 2, range 1-10)
  Source-transition count within the window that triggers flutter mode.
- `FlutterDetectWindowS` (default 1.0, range 0.1-5.0 s)
  Rolling-window length for flutter detection.
- `FlutterClampJerkMps3` (default 0.12, range 0.0-5.0 m/s^3)
  Bidirectional jerk cap. 0 disables flutter-mode clamping entirely.
- `FlutterClampBypassDecelMps2` (default 1.5, range 0.0-5.0 m/s^2)
  If `modelAccel < -this`, the clamp is bypassed so hard braking is
  unaffected.

### Recommended enable order when tuning on-device

Attack from the output-facing layers inward. Each step is live-tunable
via `live_lead_tune.py`; no service restart required.

1. **Verify Schmitt is doing its job.** Default 0.6/0.25. If you see
   source flipping at legitimately-high prob (> 0.6), there is a
   different problem — inspect `LEADROLEDBG` raw prob before tuning.
2. **Tune flutter clamp only after checking source flips.** Current default
   `FlutterClampJerkMps3=0.12` is very soft for EV comfort. Raise it only if
   source-stability damping feels rubber-banded or delays real recovery.
3. **Adjust release dwell as a stability-vs-staleness trade.** Current
   default `LeadSourceReleaseFrames=20` (~1.0 s at DT_MDL) is deliberately
   sticky for no-radar EV6 freeway follow.
4. **Adjust phantom hold only after inspecting raw lead dropouts.** Current
   default `PhantomLeadHoldS=0.80` with `PhantomLeadStableFrames=3` bridges
   repeated brief classifier drops without requiring a service restart.
5. **Add acquire dwell only if needed.** Set `LeadSourceAcquireFrames=2`
   only if the Schmitt is admitting short false acquisitions that feel
   annoying. This is a feel choice; it costs ~50 ms of acquisition
   latency on every new lead.

### When the stability layer is not the right tool

- **`source=cruise` while a valid lead is still visibly present and
  accel spikes positive:** that is the Hyundai `source_hysteresis` and
  `lead_present_cruise_accel_cap` path, not the stability layer.
- **`source` stays on the lead but follow feels late / overshoots:**
  that is the reclaim path, not the stability layer.
- **Single abrupt handoff after a genuine cut-in:** that is cut-in
  settle knobs.

## Helper Script

- From the dev box, use the repo venv. On tici, swap in
  `/usr/local/venv/bin/python3`.

- Show current values:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show
```

- Show a one-line summary:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show --shell-summary
```

- Set one or more knobs without restarting services:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py set \
  --gap-reclaim-strength 1.20 \
  --gap-reclaim-max-accel 0.42
```

- Enable the recommended source-stability stack on-device (one shot):
```bash
/usr/local/venv/bin/python3 .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py set \
  --lead-prob-enter 0.6 \
  --lead-prob-exit 0.25 \
  --lead-source-release-frames 20 \
  --phantom-lead-hold-s 0.8 \
  --phantom-lead-stable-frames 3 \
  --flutter-clamp-jerk-mps3 0.12
```

- Remove overrides for this feature and fall back to defaults:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py reset
```

## Verification

- While `plannerd` is running, change a knob with the helper script and wait
  about one second. The running `LongitudinalMpc` instance should pick up the
  new effective value on its next refresh.
- For live observation, start the watcher with:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 10 --show-live-tune
```
- If behavior becomes springy or late, reset the overrides first and confirm
  the defaults are active before changing more than one knob at once.
- If `LEADROLEDBG` shows `virtual_duplicate.active=true`, debug lead stability
  through `virtual_duplicate`, `filtered_virtual_lead`, and
  `source_hysteresis` before touching live reclaim or cut-in knobs. Those
  knobs cannot fix duplicate same-car hypotheses or noisy lead-vs-cruise
  ownership in the fixed Hyundai source-stability path.
- If `source=cruise` while a valid lead is still present and accel spikes
  positive, debug `source_hysteresis.lead_present_cruise_accel_cap` first.
  Live reclaim knobs are not the root-cause surface for that behavior.
- If `source` stays on the lead but the car overshoots and then coasts or
  lightly slows for too long, debug the Hyundai reclaim path first. That is
  usually stale optimistic reclaim state, not a missing live tune knob.
- To verify the source-stability layer is actually doing work, run the
  monitor with `--show-lateral` alongside source transitions. With the
  default Schmitt + flutter clamp, the `src=` column should flip at most
  once per genuine lead change; multi-flip bursts in a single second
  indicate the layer is being bypassed or the thresholds are too loose.
  With `--enable-lead-role-log`, `LEADROLEDBG` cloudlog entries include
  both `lead_stability_debug` (the new filter state) and the prior
  `source_hysteresis` payload, so you can distinguish "stability filter
  dropped the lead" from "Hyundai hysteresis released ownership."
