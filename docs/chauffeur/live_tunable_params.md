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

### Model-Lead Closing Urgency Blend + Lag Compensation (radard)

Fills the cliff between the slow model-lead dRel filter (tau 2.8, 1.6x when closing) and the hard fast-close gates: a continuous urgency u in [0,1] — computed ONLY when the measurement says the lead is closer than predicted — blends the dRel tau geometrically toward `BlendTauFloorS`, boosts the closing slew allowance by `BlendSlewBoostMps*u`, and blends the vRel tau toward `ModelLeadFilterFastVRelTauS`. The existing fast gates stay verbatim as the u=1 short-circuit, so nothing adopts a closer lead slower than before. Disable: `ModelLeadFilterBlendTauFloorS >= ModelLeadFilterTauS` forces u=0 everywhere (dRel tau, vRel tau and slew boost all revert), which is the verified byte-identical legacy rollback. Degenerate spans (close-lo at/above the fixed 2.5 m/s strong-closing gate, ttc-hi at/below `ModelLeadFilterSafeTtcS`) also collapse to u=0, never to u=1.

The lag compensation moves only the PUBLISHED dRel closer by `closing * LagCompS` (closing measured beyond `LagCompDeadzoneMps` on the filtered vRel); internal filter state is untouched and publication never moves farther. The compensation is capped at the active filter regime's effective tau, so the fast-close path (actual delay ~0.12 s) is not over-corrected at high closing speeds. `LagCompS=0` disables (verified byte-identical legacy publication). Raise toward 1.2 to bias earlier braking (measured +0.7 m min gap in the gray-zone sweep at negligible steady-noise cost).

The compensation additionally fades out in the stopping regime on ego speed: fully off at/below `LagCompFadeLoMps`, fully on at/above `LagCompFadeHiMps`, linear between. At low ego speed the filter-lag error is proportionally tiny (closing speeds are small), so the compensation adds little safety there while pushing the approach-to-stop point meters farther back (approach-to-stop behind a stopping lead from 15 m/s, clean final stopped gap: 8.85 m with full compensation, 6.19 m at the shipped 12/18 fade, 6.02 m with `LagCompS=0`; the 25-31 m/s gray-zone closed loop is bit-identical with and without the fade because ego never drops below `FadeHiMps` there). Ego speed is estimated publish-side as `vLead - vRel` from the track state. Disable the fade (full compensation at every speed, the pre-fade behavior) by setting `FadeHi <= FadeLo`, e.g. both 0 — the degenerate span resolves to the pessimistic direction.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ModelLeadFilterBlendTauFloorS` | 0.30 | 0.05–8.0 | dRel filter tau at full closing urgency. >= `ModelLeadFilterTauS` disables the whole blend (exact legacy) |
| `ModelLeadFilterBlendCloseLoMps` | 1.0 | 0.0–2.4 | Closing speed where urgency starts; hi endpoint is the fixed 2.5 m/s strong-closing gate |
| `ModelLeadFilterBlendTtcHiS` | 12.0 | 10.5–30.0 | TTC where urgency starts; low endpoint is `ModelLeadFilterSafeTtcS` (max 10.0) |
| `ModelLeadFilterBlendSlewBoostMps` | 6.0 | 0.0–20.0 | Extra closing dRel slew allowance (m/s) at full urgency. 0 disables |
| `ModelLeadFilterLagCompS` | 0.6 | 0.0–2.0 | Closing-only group-delay compensation (s) on published dRel, capped at the active regime's filter delay. 0 disables |
| `ModelLeadFilterLagCompDeadzoneMps` | 0.5 | 0.0–5.0 | Closing speed ignored by lag comp; keeps steady vRel jitter out of published dRel |
| `ModelLeadFilterLagCompFadeLoMps` | 12.0 | 0.0–30.0 | Ego speed at/below which lag comp is fully faded out (stopping regime) |
| `ModelLeadFilterLagCompFadeHiMps` | 18.0 | 0.0–40.0 | Ego speed at/above which lag comp is fully active. FadeHi <= FadeLo disables the fade (full comp everywhere) |
| `ModelLeadBlendMinSpan` | 0.01 | 0.001–1.0 | Degeneracy guard: a closing-urgency blend span (closing-speed, TTC, or lag-comp fade) narrower than this collapses to disabled (u=0) instead of a possible sign flip |
| `ModelLeadBlendTtcMinClosingMps` | 0.3 | 0.0–3.0 | Minimum closing speed before the TTC-based closing-urgency term is evaluated; guards TTC=dRel/closing near zero closing speed |

## Cruise Reacquire Jerk Limit

Softens the upward accel slew after the MPC source transitions from lead-follow to cruise (lead lost / classifier dropout). Only clips positive excursion — braking and steady lead-follow are unaffected. Window auto-ends when output_a_target reaches the cruise accel cap. The allowed jerk starts at `CruiseReacquirePosJerkLimit` and grows by `CruiseReacquireJerkRamp` every second, so the first frames after the handoff stay soft but recovery toward set speed is not pinned at the pre-departure follow accel for the whole window. When the ramped ceiling reaches the MPC request before the window expires (true at the 0.8 default in the audited departure scenario), the legacy accel step at window expiry is also avoided; small ramp values can still leave a residual step there. On a device that has not rebuilt `common` after this key landed, the planner falls back to the spec default (0.8) rather than legacy behavior; set `CruiseReacquireJerkRamp=0` for the verified exact-legacy fixed allowance.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `CruiseReacquirePosJerkLimit` | 0.08 | 0.0–5.0 | Max upward jerk (m/s^3) on planner output during cruise after a lead drops. 0 disables |
| `CruiseReacquireJerkWindowS` | 3.0 | 0.0–3.0 | Duration (s) the jerk limit is enforced after a lead → cruise transition. 0 disables |
| `CruiseReacquireJerkRamp` | 0.8 | 0.0–5.0 | Growth rate (m/s^3 per s) of the jerk allowance across the window. 0 = fixed allowance (legacy hang-back) |

## Lead Prob Schmitt Trigger (radard)

Asymmetric hysteresis on vision-model lead prob in radard. Per-slot latch: a slot must cross `Enter` to latch on, and fall below `Exit` to release. Defaults create a 0.35-wide hysteresis band around the old 0.5 threshold.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadProbEnter` | 0.6 | 0.0–1.0 | Prob required to latch a slot on. Raise to reject flicker |
| `LeadProbExit` | 0.25 | 0.0–1.0 | Prob below which a latched slot releases. Lower than enter = hysteresis |

## Lead Source Dwell + Phantom Hold (MPC)

Acquire/release dwell on lead `status` at the MPC boundary, plus kinematically-propagated phantom hold for lead data continuity through brief dropouts. A phantom of a decelerating lead holds the last measured decel (no decay toward zero), continues the measured deepening trend of the lagged aLeadK estimate, and propagates dRel/vRel/vLead with it — the held lead is never kinematically more optimistic than its last measurement. Positive (pull-away) accel still decays toward zero. The trend measurement resets on validity gaps >0.5 s and on track-identity discontinuities between consecutive valid frames (dRel >3 m off the propagated position, or a >1.5 m yRel jump), so a cut-in replacing the tracked lead never contributes a cross-car d(aLeadK)/dt to a later phantom. Set `PhantomLeadHoldS=0` to disable phantom. Set both dwell frame counts to 1 to disable dwell.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadSourceAcquireFrames` | 1 | 1–20 | Consecutive valid-lead frames required before MPC accepts the lead. Default 1 = no dwell; raise on-device to engage |
| `LeadSourceReleaseFrames` | 20 | 1–40 | Consecutive invalid-lead frames required before MPC releases a latched lead (ignored while phantom hold is active) |
| `PhantomLeadHoldS` | 0.80 | 0.0–1.5 | Duration (s) the last-known lead is extrapolated after status goes False. 0 disables phantom |
| `PhantomLeadStableFrames` | 3 | 1–40 | Consecutive stable frames required before a dropped lead is eligible for phantom |
| `PhantomLeadDecelHoldFactor` | 1.0 | 0.0–1.0 | Fraction of the last measured lead decel (aLeadK<0) held through the phantom window. 1 = full hold; 0 = legacy linear decay to zero |
| `PhantomLeadDecelTrendGain` | 1.0 | 0.0–1.0 | Fraction of the measured pre-drop d(aLeadK)/dt continued through the phantom window (deepening trends only). 0 = hold constant |
| `LeadStabilizerTrendTauS` | 0.20 | 0.05–1.0 | EMA time constant for the measured d(aLeadK)/dt used by the phantom trend hold. Lower = faster trend response, more noise passed through |
| `LeadStabilizerTrendDRelJumpM` | 3.0 | 1.0–10.0 | Identity gate: a dRel step this far off the propagated position between consecutive valid frames is a track swap, not a measurement |
| `LeadStabilizerTrendYRelJumpM` | 1.5 | 0.3–5.0 | Identity gate: a lateral (yRel) jump this large between consecutive valid frames is a track swap, not a measurement |

## Lead Accel Corroboration Bound (MPC)

Bounds uncorroborated transient negative aLeadK at the single MPC lead ingress (`_stabilize_raw_leads` output, feeding role classifier, previews, `process_lead` and the brake-release floor): when the low-passed finite-difference of stabilized vLead (a_meas) does not corroborate the model's decel claim, aLeadK is floored at `min(0, a_meas) - LeadAccelCorrMarginMps2`. The bound is bypassed entirely — full aLeadK passes — in any of: a dangerous state (TTC <= `TtcGuardS`, closing >= `ClosingGuardMps`, or gap <= `NearHeadwayS` x v_ego; latched with hysteresis so guard-boundary noise cannot chatter aLeadK), a phantom-held or stale (non-fresh-measurement) slot (the a_meas low-pass is frozen, not decayed, through the hold, and a held braking lead keeps its full measured decel), or before ~2x`MeasTauS` of same-track vLead history exists (the low-pass resets on track identity changes: radarTrackId change, reacquisition, or a vLead step beyond a 10 m/s^2 physical-accel gate). `LeadAccelCorrMarginMps2 >= 10` disables the bound entirely (verified rollback: restores the measured tau-dependent blip divergence in test_repro_alead_tau_transient.py). Do not raise `MeasTauS` casually: 0.6 measurably delayed hard-brake onset in the design sweep.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadAccelCorrMarginMps2` | 0.5 | 0.0–10.0 | Max uncorroborated lead decel below the measured vLead trend passed to the MPC. >= 10 disables |
| `LeadAccelCorrMeasTauS` | 0.3 | 0.1–2.0 | Low-pass tau for the measured vLead trend |
| `LeadAccelCorrTtcGuardS` | 8.0 | 2.0–20.0 | Bound bypassed at or below this TTC |
| `LeadAccelCorrClosingGuardMps` | 1.5 | 0.0–10.0 | Bound bypassed at or above this closing speed |
| `LeadAccelCorrNearHeadwayS` | 1.2 | 0.0–4.0 | Bound bypassed inside this headway of gap |
| `LeadAccelCorrClosingRearmMps` | 0.5 | 0.0–5.0 | Dangerous-state bypass hysteresis: closing speed must drop this far below `ClosingGuardMps` before the bypass can disengage |
| `LeadAccelCorrTtcRearmS` | 2.0 | 0.0–10.0 | Dangerous-state bypass hysteresis: TTC must rise this far above `TtcGuardS` before the bypass can disengage |
| `LeadAccelCorrHeadwayRearmM` | 2.0 | 0.0–10.0 | Dangerous-state bypass hysteresis: gap must exceed the near-headway gate by this many meters before the bypass can disengage |
| `LeadAccelCorrSettleTauMult` | 2.0 | 0.5–5.0 | Multiple of `MeasTauS` of same-track vLead history required before the bound can clamp aLeadK |
| `LeadAccelCorrMaxDtS` | 0.5 | 0.05–2.0 | Max frame-to-frame dt admitted as a same-track vLead measurement; a larger gap resets the corroboration low-pass |

## Flutter Mode Clamp (asymmetric jerk)

When the MPC source flip-flops at the edge of lead acquisition (brake-tap sensation), enter flutter mode and clamp `output_a_target` slew for comfort. Braking has its own, never-tighter allowance (`FlutterClampBrakeJerkMps3`): tap suppression comes from the slow positive release, while brake onset is not throttled below what the MPC requests. Bypassed entirely on strong modelAccel braking so hard decel is never delayed. On a device that has not rebuilt `common` after `FlutterClampBrakeJerkMps3` landed, the planner falls back to the spec default (1.5) rather than legacy behavior; set `FlutterClampBrakeJerkMps3=0` for the verified exact-legacy symmetric clamp (setting both it and `CruiseReacquireJerkRamp` to 0 is the verified full rollback of the post-flicker-hang fix).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `FlutterDetectTransitions` | 2 | 1–10 | Source transitions within the window that trigger flutter mode |
| `FlutterDetectWindowS` | 1.0 | 0.1–5.0 | Rolling-window length for flutter detection |
| `FlutterClampJerkMps3` | 0.12 | 0.0–5.0 | Upward jerk cap (m/s^3) during flutter mode. 0 disables the clamp |
| `FlutterClampBrakeJerkMps3` | 1.5 | 0.0–5.0 | Downward jerk cap (m/s^3) during flutter mode; effective cap is max(this, `FlutterClampJerkMps3`). 0 = symmetric legacy clamp |
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
