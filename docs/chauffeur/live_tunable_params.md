# Live Tunable Longitudinal Params

All params are read at runtime via `Longitudinal.LiveTune.*` keys. Changes take effect within ~1s without service restart (after initial build). Defaults below match the May 16, 2026 EV6 freeway tune plus the follow-up synthetic `LeadSlowdownStrength=0.25` comfort adjustment captured in `docs/chauffeur/longitudinal/ev6_live_tune_20260516.md`.

## MPC Weights

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ObstacleCost` | 2.0 | 1.0–20.0 | MPC obstacle proximity cost. Higher = reacts sooner to gap changes |
| `AccelChangeCost` | 400.0 | 10.0–500.0 | MPC accel direction-change penalty. Higher = smoother accel/decel swings, lower = more responsive/noisier |
| `AccelCost` | 1.0 | 0.0–1.0 | MPC accel magnitude penalty ("prefer coast"). Asymmetric would be ideal |

## Gap Reclaim

| Param Key | Default | Range | Description |
|---|---|---|---|
| `GapReclaimStrength` | 0.55 | 0.0–2.0 | How eagerly ACC closes extra gap on pullaway |
| `GapReclaimGapMinM` | 3.0 | 0.0–10.0 | Minimum extra gap before reclaim activates |
| `GapReclaimMaxAccel` | 0.12 | 0.0–0.75 | Cap on positive accel floor for gap closing |

## Lead Keep-Up

Tiny immediate accel floor for a followed lead that starts pulling away. This is deliberately separate from Gap Reclaim: keep-up can start before a large gap exists, while reclaim remains the stronger response once the gap is clearly real.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadKeepUpStrength` | 1.15 | 0.0–2.0 | Scale for the immediate keep-up floor. The first hint stays tiny; confirmed pull-aways can climb to `LeadKeepUpMaxAccel` |
| `LeadKeepUpGapMinM` | 0.10 | 0.0–5.0 | Extra gap above nominal headway before distance-based keep-up starts |
| `LeadKeepUpMaxAccel` | 0.095 | 0.0–5.0 | Cap on keep-up floor before planner/personality accel limits. Leave the default tiny for EV comfort |

## Lead Slowdown

Accel ceiling for a followed slower/braking lead. The first hint only trims positive accel; confirmed closing, lead braking, or short TTC can request the full negative accel envelope.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadSlowdownStrength` | 0.25 | 0.0–2.0 | Scale for the normal slowdown ceiling; panic/short-TTC authority is not reduced by this |
| `LeadSlowdownMaxDecel` | 4.0 | 0.0–6.0 | Maximum braking magnitude the slowdown ceiling may request before vehicle/controller limits apply |

## Lead Brake Release

Vibe-follow-only accel floor that prevents continued heavy decel after the Vibe headway target has recovered or is about to recover. Safety gating uses relative closing distance against the planner's available negative accel (`-6 m/s²` on this Hyundai/EV6 GT path), while hard lead decel still blocks release.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadBrakeReleaseMinSpeedMps` | 5.0 | 0.0–20.0 | Minimum ego speed for brake release |
| `LeadBrakeReleaseBrakeDeficitMarginM` | 1.5 | 0.0–10.0 | Relative-braking-distance deficit allowed before release stays disabled |
| `LeadBrakeReleaseLookaheadS` | 2.0 | 0.1–6.0 | Projected time-to-target window for easing continued decel |
| `LeadBrakeReleaseMinPullawayMps` | 0.10 | 0.0–3.0 | Minimum opening speed for projected-recovery release |
| `LeadBrakeReleaseNearTargetMarginM` | 1.5 | 0.0–8.0 | Headway deficit treated as near target when closing is small |
| `LeadBrakeReleaseNearTargetMaxClosingMps` | 0.75 | 0.0–4.0 | Max closing speed eligible for near-target release |
| `LeadBrakeReleaseNearTargetFloorMps2` | -0.05 | -2.0–0.5 | Floor near target; raise toward/above zero to counter EV regen |
| `LeadBrakeReleaseLeadDecelMinMps2` | -0.75 | -6.0–0.0 | Disable release when the lead is braking harder than this |
| `LeadBrakeReleaseApproachFloorMps2` | -0.60 | -6.0–0.0 | Most decel allowed while projected recovery ramps toward near-coast |
| `LeadBrakeReleaseCoastBiasMps2` | 0.05 | -0.5–0.8 | Floor once target is recovered and ego is no longer closing |

## Lead Preview

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadPreviewStrength` | 1.5 | 0.0–2.0 | How early a newly recognized slower lead shapes decel |
| `LeadPreviewGapMinM` | 1.0 | 0.0–10.0 | Min extra slack before preview activates |
| `LeadPreviewMaxBufferM` | 10.0 | 0.0–25.0 | Max closer-pull of previewed lead obstacle |
| `LeadAcquireWindowS` | 1.5 | 0.0–3.0 | Short stronger-preview window after a lead appears or jumps materially slower/closer |

## Cut-In Settle

| Param Key | Default | Range | Description |
|---|---|---|---|
| `CutInSettleDurationS` | 6.0 | 0.0–12.0 | Grace window length after cut-in detection |
| `CutInSettleMaxDecel` | 0.15 | 0.0–0.80 | Max braking magnitude during grace window |
| `CutInSettleMaxClosingSpeedMps` | 2.2 | 0.5–6.0 | Max ego-lead closing speed to qualify for grace |
| `CutInSettleAccelBiasMps2` | 0.12 | 0.0–0.30 | Positive accel offset to counteract EV regen during settle |

## Virtual Lead EMA Filter

| Param Key | Default | Range | Description |
|---|---|---|---|
| `VirtualLeadSlowTauS` | 1.30 | 0.10–3.0 | EMA tau for aLeadK in safe/noise-rejection direction. Sign transitions (decel→accel) use a fixed 0.30s tau regardless |

## dRel Noise Filter

| Param Key | Default | Range | Description |
|---|---|---|---|
| `DRelFilterTauCloseS` | 0.30 | 0.05–2.0 | Filter tau when lead appears closer (safety). Lower = faster |
| `DRelFilterTauOpenS` | 0.80 | 0.10–5.0 | Filter tau when lead appears farther (noise rejection). Higher = smoother |
| `DRelFilterOpenSlewMaxMps` | 1.80 | 0.25–5.0 | Max opening-side dRel motion admitted per second before correction |
| `DRelFilterInnovationGateM` | 30.0 | 5.0–60.0 | Snap to raw when prediction error exceeds this |
| `DRelFilterClosingGateM` | 12.0 | 5.0–40.0 | Snap to raw when lead appears this much closer than predicted |
| `UseKalmanDRelFilter` | 1 | bool | Use the Kalman dRel filter path instead of the EMA dRel filter |
| `KalmanDRelQ` | 0.5 | float | Kalman process noise for dRel |
| `KalmanDRelR` | 6.0 | float | Kalman measurement noise for dRel |
| `KalmanDRelGainMax` | 0.25 | float | Max Kalman correction gain |
| `KalmanDRelDeadbandM` | 0.75 | float | Ignore small dRel innovations inside this deadband |
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
| `CruiseReacquirePosJerkLimit` | 0.08 | 0.0–5.0 | Max upward jerk (m/s^3) on planner output during cruise after a lead drops. 0 disables |
| `CruiseReacquireJerkWindowS` | 3.0 | 0.0–3.0 | Duration (s) the jerk limit is enforced after a lead → cruise transition. 0 disables |

## Lead Prob Schmitt Trigger (radard)

Asymmetric hysteresis on vision-model lead prob in radard. Per-slot latch: a slot must cross `Enter` to latch on, and fall below `Exit` to release. Defaults create a 0.35-wide hysteresis band around the old 0.5 threshold.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadProbEnter` | 0.6 | 0.0–1.0 | Prob required to latch a slot on. Raise to reject flicker |
| `LeadProbExit` | 0.25 | 0.0–1.0 | Prob below which a latched slot releases. Lower than enter = hysteresis |

## Lead Source Dwell + Phantom Hold (MPC)

Acquire/release dwell on lead `status` at the MPC boundary, plus velocity-extrapolated phantom hold for lead data continuity through brief dropouts. Set `PhantomLeadHoldS=0` to disable phantom. Set both dwell frame counts to 1 to disable dwell.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadSourceAcquireFrames` | 1 | 1–20 | Consecutive valid-lead frames required before MPC accepts the lead. Default 1 = no dwell; raise on-device to engage |
| `LeadSourceReleaseFrames` | 20 | 1–40 | Consecutive invalid-lead frames required before MPC releases a latched lead (ignored while phantom hold is active) |
| `PhantomLeadHoldS` | 0.80 | 0.0–1.5 | Duration (s) the last-known lead is extrapolated after status goes False. 0 disables phantom |
| `PhantomLeadStableFrames` | 3 | 1–40 | Consecutive stable frames required before a dropped lead is eligible for phantom |

## Flutter Mode Clamp (bidirectional jerk)

When the MPC source flip-flops at the edge of lead acquisition (brake-tap sensation), enter flutter mode and clamp `output_a_target` slew in BOTH directions for comfort. Bypassed on strong modelAccel braking so real decel is not delayed.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `FlutterDetectTransitions` | 2 | 1–10 | Source transitions within the window that trigger flutter mode |
| `FlutterDetectWindowS` | 1.0 | 0.1–5.0 | Rolling-window length for flutter detection |
| `FlutterClampJerkMps3` | 0.12 | 0.0–5.0 | Bidirectional jerk cap (m/s^3) during flutter mode. 0 disables |
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
