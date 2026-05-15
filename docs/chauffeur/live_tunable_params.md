# Live Tunable Longitudinal Params

All params are read at runtime via `Longitudinal.LiveTune.*` keys. Changes take effect within ~1s without service restart (after initial build).

## MPC Weights

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ObstacleCost` | 4.0 | 1.0–20.0 | MPC obstacle proximity cost. Higher = reacts sooner to gap changes |
| `AccelChangeCost` | 200.0 | 10.0–500.0 | MPC accel direction-change penalty. Lower = more responsive, amplifies noise |
| `AccelCost` | 0.0 | 0.0–1.0 | MPC accel magnitude penalty ("prefer coast"). Asymmetric would be ideal |

## Gap Reclaim

| Param Key | Default | Range | Description |
|---|---|---|---|
| `GapReclaimStrength` | 1.5 | 0.0–2.0 | How eagerly ACC closes extra gap on pullaway |
| `GapReclaimGapMinM` | 0.0 | 0.0–10.0 | Minimum extra gap before reclaim activates |
| `GapReclaimMaxAccel` | 0.24 | 0.0–0.75 | Cap on positive accel floor for gap closing |

## Lead Keep-Up

Tiny immediate accel floor for a followed lead that starts pulling away. This is deliberately separate from Gap Reclaim: keep-up can start before a large gap exists, while reclaim remains the stronger response once the gap is clearly real.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadKeepUpStrength` | 1.0 | 0.0–2.0 | Scale for the immediate keep-up floor. The first hint stays tiny; confirmed pull-aways can climb to `LeadKeepUpMaxAccel` |
| `LeadKeepUpGapMinM` | 0.40 | 0.0–5.0 | Extra gap above nominal headway before distance-based keep-up starts |
| `LeadKeepUpMaxAccel` | 0.08 | 0.0–5.0 | Cap on keep-up floor before planner/personality accel limits. Leave the default tiny for EV comfort |

## Lead Slowdown

Accel ceiling for a followed slower/braking lead. The first hint only trims positive accel; confirmed closing, lead braking, or short TTC can request the full negative accel envelope.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadSlowdownStrength` | 0.5 | 0.0–2.0 | Scale for the normal slowdown ceiling; panic/short-TTC authority is not reduced by this |
| `LeadSlowdownMaxDecel` | 6.0 | 0.0–6.0 | Maximum braking magnitude the slowdown ceiling may request before vehicle/controller limits apply |

## Lead Preview

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadPreviewStrength` | 1.8 | 0.0–2.0 | How early a newly recognized slower lead shapes decel |
| `LeadPreviewGapMinM` | 1.5 | 0.0–10.0 | Min extra slack before preview activates |
| `LeadPreviewMaxBufferM` | 12.0 | 0.0–25.0 | Max closer-pull of previewed lead obstacle |
| `LeadAcquireWindowS` | 1.25 | 0.0–3.0 | Short stronger-preview window after a lead appears or jumps materially slower/closer |

## Cut-In Settle

| Param Key | Default | Range | Description |
|---|---|---|---|
| `CutInSettleDurationS` | 7.0 | 0.0–12.0 | Grace window length after cut-in detection |
| `CutInSettleMaxDecel` | 0.15 | 0.0–0.80 | Max braking magnitude during grace window |
| `CutInSettleMaxClosingSpeedMps` | 2.5 | 0.5–6.0 | Max ego-lead closing speed to qualify for grace |
| `CutInSettleAccelBiasMps2` | 0.20 | 0.0–0.30 | Positive accel offset to counteract EV regen during settle |

## Virtual Lead EMA Filter

| Param Key | Default | Range | Description |
|---|---|---|---|
| `VirtualLeadSlowTauS` | 1.00 | 0.10–3.0 | EMA tau for aLeadK in safe/noise-rejection direction. Sign transitions (decel→accel) use a fixed 0.30s tau regardless |

## dRel Noise Filter

| Param Key | Default | Range | Description |
|---|---|---|---|
| `DRelFilterTauCloseS` | 0.30 | 0.05–2.0 | Filter tau when lead appears closer (safety). Lower = faster |
| `DRelFilterTauOpenS` | 1.00 | 0.10–5.0 | Filter tau when lead appears farther (noise rejection). Higher = smoother |
| `DRelFilterOpenSlewMaxMps` | 1.25 | 0.25–5.0 | Max opening-side dRel motion admitted per second before correction |
| `DRelFilterInnovationGateM` | 30.0 | 5.0–60.0 | Snap to raw when prediction error exceeds this |
| `DRelFilterClosingGateM` | 20.0 | 5.0–40.0 | Snap to raw when lead appears this much closer than predicted |
| `ModelLeadFilterTauS` | 2.80 | 0.20–8.0 | Source-side no-radar model-lead dRel filter tau in radard. Higher rejects more model distance noise |
| `ModelLeadFilterOpenSlewMaxMps` | 1.20 | 0.10–6.0 | Max source-side opening dRel motion admitted per second without model velocity support |
| `ModelLeadFilterSafeTtcS` | 4.00 | 1.0–10.0 | Low-TTC threshold that fast-adopts closer model-lead measurements |
| `ModelLeadFilterAssocDRelM` | 12.0 | 3.0–35.0 | dRel gate for associating model-only lead hypotheses to a stable synthetic track id |
| `ModelLeadFilterVRelTauS` | 0.40 | 0.10–2.0 | Source-side model-lead relative-velocity filter tau. Lower = faster accel/decel recognition |
| `ModelLeadFilterFastVRelTauS` | 0.16 | 0.05–1.0 | Relative-velocity filter tau used after low-TTC or strongly closing model-lead admission |

## Cruise Reacquire Jerk Limit

Softens the upward accel slew after the MPC source transitions from lead-follow to cruise (lead lost / classifier dropout). Only clips positive excursion — braking and steady lead-follow are unaffected. Window auto-ends when output_a_target reaches the cruise accel cap.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `CruiseReacquirePosJerkLimit` | 0.6 | 0.0–5.0 | Max upward jerk (m/s^3) on planner output during cruise after a lead drops. 0 disables |
| `CruiseReacquireJerkWindowS` | 1.5 | 0.0–3.0 | Duration (s) the jerk limit is enforced after a lead → cruise transition. 0 disables |

## Lead Prob Schmitt Trigger (radard)

Asymmetric hysteresis on vision-model lead prob in radard. Per-slot latch: a slot must cross `Enter` to latch on, and fall below `Exit` to release. Defaults create a 0.25-wide hysteresis band around the old 0.5 threshold.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadProbEnter` | 0.6 | 0.0–1.0 | Prob required to latch a slot on. Raise to reject flicker |
| `LeadProbExit` | 0.35 | 0.0–1.0 | Prob below which a latched slot releases. Lower than enter = hysteresis |

## Lead Source Dwell + Phantom Hold (MPC)

Acquire/release dwell on lead `status` at the MPC boundary, plus velocity-extrapolated phantom hold for lead data continuity through brief dropouts. Set `PhantomLeadHoldS=0` to disable phantom. Set both dwell frame counts to 1 to disable dwell.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadSourceAcquireFrames` | 1 | 1–20 | Consecutive valid-lead frames required before MPC accepts the lead. Default 1 = no dwell; raise on-device to engage |
| `LeadSourceReleaseFrames` | 1 | 1–40 | Consecutive invalid-lead frames required before MPC releases a latched lead (ignored while phantom hold is active). Default 1 = no dwell; raise on-device to engage |
| `PhantomLeadHoldS` | 0.0 | 0.0–1.5 | Duration (s) the last-known lead is extrapolated after status goes False. 0 disables phantom (default off); raise on-device to engage |
| `PhantomLeadStableFrames` | 5 | 1–40 | Consecutive stable frames required before a dropped lead is eligible for phantom |

## Flutter Mode Clamp (bidirectional jerk)

When the MPC source flip-flops at the edge of lead acquisition (brake-tap sensation), enter flutter mode and clamp `output_a_target` slew in BOTH directions for comfort. Bypassed on strong modelAccel braking so real decel is not delayed.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `FlutterDetectTransitions` | 2 | 1–10 | Source transitions within the window that trigger flutter mode |
| `FlutterDetectWindowS` | 1.0 | 0.1–5.0 | Rolling-window length for flutter detection |
| `FlutterClampJerkMps3` | 0.8 | 0.0–5.0 | Bidirectional jerk cap (m/s^3) during flutter mode. 0 disables |
| `FlutterClampBypassDecelMps2` | 1.5 | 0.0–5.0 | If modelAccel < -this, clamp is bypassed |

## Setting Params from SSH

```bash
# Show current values
/usr/local/venv/bin/python3 /data/openpilot/.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show

# Set a value
/usr/local/venv/bin/python3 /data/openpilot/.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py set drel-filter-tau-open 1.5

# Reset all to defaults
/usr/local/venv/bin/python3 /data/openpilot/.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py reset
```
