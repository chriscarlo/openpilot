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
| `GapReclaimMaxAccel` | 0.30 | 0.0–0.75 | Cap on positive accel floor for gap closing. Raised from 0.12 in the 2026-07-02 comfort retune: the post-fix road test showed reclaim intent saturating against the old cap (median THW-recovery 5.4 s / p90 10.7 s, 48% of steady-follow time above 2.0 s THW), so the seat preference is a faster, still-comfort-bounded reclaim |

## Lead Keep-Up

Tiny immediate accel floor for a followed lead that starts pulling away. This is deliberately separate from Gap Reclaim: keep-up can start before a large gap exists, while reclaim remains the stronger response once the gap is clearly real.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadKeepUpStrength` | 1.15 | 0.0–2.0 | Scale for the immediate keep-up floor. The first hint stays tiny; confirmed pull-aways can climb to `LeadKeepUpMaxAccel` |
| `LeadKeepUpGapMinM` | 0.10 | 0.0–5.0 | Extra gap above nominal headway before distance-based keep-up starts |
| `LeadKeepUpMaxAccel` | 0.22 | 0.0–5.0 | Cap on keep-up floor before planner/personality accel limits. Raised from 0.095 in the 2026-07-02 comfort retune (seat preference, same reclaim-saturation road data as `GapReclaimMaxAccel`); still small relative to the 0.75+ range so the "leave it tiny" EV-comfort intent holds, just less capped than before |

## Lead Slowdown

Accel ceiling for a followed slower/braking lead. The first hint only trims positive accel; confirmed closing, lead braking, or short TTC can request the full negative accel envelope.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadSlowdownStrength` | 0.25 | 0.0–2.0 | Scale for the normal slowdown ceiling; panic/short-TTC authority is not reduced by this |
| `LeadSlowdownMaxDecel` | 4.0 | 0.0–6.0 | Maximum braking magnitude the slowdown ceiling may request before vehicle/controller limits apply |
| `LeadSlowdownKinematicHeadroom` | 1.5 | 1.0–5.0 | Energy-consistency bound on the danger term: the gated demand may not exceed this multiple of the decel physics requires to stop `LeadSlowdownKinematicMarginM` short of the lead. Stops the danger-surplus collapse from slamming brakes on calm stops; genuine short-gap or hard-braking-lead threats are unaffected because their physical requirement is already large. Uses `max(0, vLead)` as the lead speed: exact for stopped/near-stopped leads (it discards radard's ~-1.2 m/s near-stop vRel-boost artifact), conservative (over-braking side) for moving leads since published vLead already carries the inward boost. Legacy rollback sentinel: write a very large value (e.g. `1e9`; the read path does not clamp to the spec range) to disable the bound live and restore pre-fix ceiling behavior exactly — no rebuild |
| `LeadSlowdownKinematicMarginM` | 4.0 | 1.0–5.0 | Gap reserve for the kinematic bound; once the lead is projected to stop inside this distance the danger term is uncapped (full authority). Code re-clamps to 1.0–5.0 (`STOP_DISTANCE - 1.0`) regardless of the stored param: 6 m readmits the calm-stop slam in full (verified), and below 1 m the bound brakes LESS and the inside-margin full-authority restoration is unreachable |
| `LeadSlowdownKinematicOncomingVLeadMps` | -2.5 | -6.0–-1.5 | Published vLead below which the kinematic bound is bypassed entirely: an oncoming/reversing lead's true closure (`v_ego + \|vLead\|`) exceeds what the `max(0, vLead)` clamp can express, so it keeps full legacy danger authority. Max -1.5 keeps the bypass clearly beyond the ~-1.2 m/s near-stop boost artifact (raising it toward 0 would re-enable the calm-stop slam); more negative than -6 denies genuinely oncoming leads their uncapped authority |

Known open case: a creeping lead (~0.8 m/s) approached at 8 m/s (noise-off) still saturates the ceiling to `LeadSlowdownMaxDecel` — this pre-dates the kinematic bound (legacy emulation is identical there) and is NOT covered by it.

Stop-gap shift at the K=1.5 / margin=4.0 defaults (harness, calm 5 m/s approach to a stopped lead): noise-off true stop gap moved 5.17 m (legacy) -> 4.67 m; with the ev6 measured-noise profile 5.81–6.25 m. On-device live-tune verification for these two knobs must include the felt stop gap against the ~6 m target (not just the 4.0 m harness safety floor): confirm a calm approach to a stopped car still settles near 6 m and shows no decel step beyond what the closing kinematics require before touching K or margin further.

## Lead Handoff (Stopping Need)

Cruise->lead obstacle handoff trigger for the Hyundai AI-lead-stability path, where the MPC solver sees ONLY the active obstacle: while cruise owns it, a stopped/slowing lead is invisible to the solver at any distance. This leg hands the solver the lead obstacle as soon as the kinematic decel required to stop `STOP_DISTANCE` (6 m) short of the lead reaches the threshold. It reuses the M1 kinematic-bound two-branch physics (`max(0, vLead)` clamp; closing-speed branch plus the lead-stop-extended `v_ego` branch for a decelerating lead) with INVERTED oncoming semantics: below `LeadSlowdownKinematicOncomingVLeadMps` the need is computed with the full closure rate `v_ego + |vLead|` so the leg still fires for oncoming leads (M1's bypass grants the ceiling authority; a trigger must engage instead). OR'd into `raw_requires_owner`, so it can only make the handoff EARLIER — it cannot delay or weaken any existing handoff, release, ceiling, FCW, or low-speed-queue path, and the `min(active, cruise)` clamp still prevents commanding above set speed. Fixes the midband (~18 mph) stop slam: 8 m/s / 80 m stopped-lead reveal moved from worst 0.3 s step 1.46 m/s² / peak -4.00 / stop gap 2.56 m to 0.46 / -2.64 / 6.51 m at composed-HEAD defaults (with the preview fade).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadHandoffStoppingNeedDecelMps2` | 0.80 | 0.05–1e9 | Required-decel threshold (m/s²) at/below the reference speed. Default rationale: 0.80 is the highest base at which 8 m/s and 7 m/s / 80 m approaches stay inside the calm-human bounds (worst 0.3 s step < 1.0 m/s², peak >= -3.0 m/s², stop gap 4.0–7.5 m) AND the 5 m/s / 45 m low-speed-queue case stays bit-identical (at 5 m/s the required decel maxes at ~0.78 before `low_speed_queue_hold` takes ownership at 22 m, so the leg never preempts the queue path). Lower values hand off earlier and brake more gently (0.45: peak -1.39 at 8/80) but change queue-case ownership; higher values re-approach the slam (0.9: peak -3.40 at 8/80). Legacy rollback sentinel: `1e9` (the spec maximum deliberately admits it) makes the leg unreachable and restores the pre-fix handoff exactly (verified bit-identical on the midband repro physics) |
| `LeadHandoffStoppingNeedRefSpeedMps` | 8.0 | 1.0–1e9 | Reference ego speed: below it the threshold is flat; above it the effective threshold is `base * v_ego / ref` (a constant-time-headway trigger for stopped leads, ~5 s to the 6 m margin at the defaults). This (a) keeps 9–10 m/s calm stops inside the 7.5 m human stop-gap bound (the tracker's inward dRel bias grows with the length of the gentle decel phase: flat 0.80 left 10 m/s / 80 m at 7.69 m; scaled it lands 7.32 m) and (b) bounds ghost/false-positive exposure at highway speed — flat 0.80 would fire for any latched stopped lead out to ~700 m equivalent at 25 m/s, while scaling engages only where physics genuinely demands proportional decel (~130–160 m at 25–30 m/s). Braking toward a REAL latched stopped lead at highway speed is the mandated safety direction; the scale exists to bound how much a vision ghost can command. Measured at the defaults (20/25/30 m/s, genuine 150–200 m reveals AND 1–1.5 s ghosts latched at 120–150 m): fix-on traces are BIT-IDENTICAL to the 1e9 rollback, because the scaled engagement range (~130 m at 25 m/s) sits inside the range the legacy `raw_obstacle_hold` leg already owns at highway speed (cruise-obstacle crossing ~167 m at 25 m/s) — the leg only adds earliness in the ~6–16 m/s band it was built for. The pre-existing highway ghost response (peak ~-4.7 m/s² for a 1 s ghost at 130 m / 25 m/s, releasing through `no_filtered_lead` ~3.5 s after the ghost drops) is unchanged; the original proposal's "~0.5 m/s² worst-case false positive" figure holds only in the 8 m/s midband where the leg actually adds engagement. Sentinel: `1e9` disables scaling (flat threshold at every speed) |

## Lead Brake Release

Vibe-follow-only accel floor that prevents continued heavy decel after the Vibe headway target has recovered or is about to recover. Safety gating uses relative closing distance against the planner's available negative accel (`-6 m/s²` on this Hyundai/EV6 GT path), while hard lead decel still blocks release.

Follow limit-cycle fix (2026-07-02, seat report: buck/slow/hold/late-re-accel/overshoot during steady follow): the gap error the release path measures recovery against is now vRel-aware — the MPC's own `desired_follow_distance(v_ego, v_lead, t_follow)` — implemented as a CREDIT of `max(0, vLead² - vEgo²)/(2·COMFORT_BRAKE)` on top of the legacy headway error. The credit basis takes the more pessimistic of `vLead` and `vEgo + vRel` (any closing state gets zero credit, so no closing state is looser than legacy), is projected forward by `max(0, aLeadK) · RecoveryProjS` to cover the tracker's vRel lag behind aLeadK when a lead finishes a transient slowdown, and saturates at `VrelCreditCapM`. Once recovered with the lead pulling away, the floor also rises above the coast bias toward `GapReclaimFollowMaxAccel` (phased in over 0.35–1.25 m/s of measured pullaway, tapered by the kinematic overshoot bound scaled by `GapReclaimTaperGain`) so ego matches lead speed BEFORE the equal-speed gap is regained instead of crawling at the MPC unwind jerk and then overshooting. Composition: whenever the release floor is applied AND the lead is a corroborated threat (published aLeadK < 0 or closing > 0), the M1 kinematic slowdown ceiling is re-applied AFTER the floor so the ceiling always wins — the ceiling's slew-limited release tail (a comfort mechanism) does not re-brake a recovered opening gap. Measured (noise-off oracle, `test_repro_follow_limit_cycle.py`): highway 1.5 m/s dip at 30 m/s — brake-hold after recovery 0.95 s → 0.0 s planner / 1.90 → 0.55 s realized, max re-accel deficit 1.57 → 0.98 m/s, rebound overshoot -3.61 → -1.68 m; city 0.4 m/s² ease at 13.5 m/s — ringing 2 → 1 crossing, settle 21.1 → 12.2 s. Mid-window threat probe: if the lead brakes -3 m/s² while the floor is active, the floor drops out 0.15–0.35 s after onset via the aLeadK gate; min true gap costs 1.5–2.7 m vs the pre-fix hold (8.8–9.4 m absolute at 30→12 m/s, peak decel -4.45 vs -4.30) — the bounded price of not holding brake on a recovered gap. Rollback: `VrelCreditCapM=0` + `RecoveryProjS=0` + `GapReclaimFollowMaxAccel=0` restores the legacy release behavior (the ceiling-wins composition is deliberately not knob-gated — it is strictly more-braking).

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
| `LeadBrakeReleaseLeadDecelProjectGain` | 1.0 | 0.0–2.0 | CD1 fix: scale on the lead's own decel magnitude added to the required ego decel in the release floor's closing and near-target branches, so a lead braking to a stop inside the `LeadBrakeReleaseLeadDecelMinMps2` veto can no longer clip the MPC's ramping brake above what the still-decelerating lead demands. Steady/accelerating lead adds zero (bit-identical to shipped). 0 = pre-fix instantaneous-closing floor (rollback) |
| `LeadBrakeReleaseVrelCreditCapM` | 10.0 | 0.0–20.0 | Cap (m) on the vRel-aware recovery credit; closing states always get zero credit. 0 = legacy headway-only gap error |
| `LeadBrakeReleaseRecoveryProjS` | 3.0 | 0.0–5.0 | Horizon (s) projecting the credit basis forward by the POSITIVE part of aLeadK only. 0 = no projection |
| `GapReclaimFollowMaxAccel` | 0.32 | 0.0–1.5 | Follow-regime cap on the recovered-gap re-accel floor (separate from the shared `GapReclaimMaxAccel`). <= coast bias (e.g. 0) disables the raise. Raised from 0.25 toward (but NOT to) the 0.45 comfort-retune target on 2026-07-02: `test_repro_follow_limit_cycle.py` (the steady-follow bucking oracle) starts failing at 0.40 (highway rebound overshoot -2.99 m breaches the -2.5 m floor) and fails harder at 0.45 (highway -3.52 m AND city ringing regresses to 2 cycles / 17.2 s settle); 0.32 was the swept value with real margin on both legs (highway overshoot -2.20 m vs -2.5 floor, city 1 cycle / 11.7 s settle vs the 16.0 s bound). `GapReclaimMaxAccel` and `LeadKeepUpMaxAccel` do not interact with this oracle (bit-identical highway/city metrics with those two alone raised to their new defaults) so only this knob was capped short of the requested value |
| `GapReclaimTaperGain` | 2.0 | 0.0–20.0 | Kinematic overshoot taper on the follow re-accel raise, 1/(m/s²): extra authority scales by clip(1 - gain·c_proj²/(2·gap_surplus), 0, 1). 0 = naive raise (diagnostic) |
| `LeadBrakeReleaseApproachFloorMps2` | -0.60 | -6.0–0.0 | Most decel allowed while projected recovery ramps toward near-coast |
| `LeadBrakeReleaseCoastBiasMps2` | 0.05 | -0.5–0.8 | Floor once target is recovered and ego is no longer closing |

## Lead Preview

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadPreviewStrength` | 1.5 | 0.0–2.0 | How early a newly recognized slower lead shapes decel |
| `LeadPreviewGapMinM` | 1.0 | 0.0–10.0 | Min extra slack before preview activates |
| `LeadPreviewMaxBufferM` | 10.0 | 0.0–25.0 | Max closer-pull of previewed lead obstacle |
| `LeadPreviewMinSpeedMps` | 6.0 | 0.0–8.0 | Ego speed at which the preview fades to zero, ramping linearly to full strength at 8.0 m/s. Replaces the historical hard cut at 8.0, which stepped the previewed obstacle by up to `LeadPreviewMaxBufferM` (10 m) INSTANTLY whenever the planner's drifting filtered speed wobbled across 8.0 — a latent jerk source whenever the lead already owns the obstacle (this knife-edge is what split the ~18 mph band: at v0 >= 8 the preview advanced the handoff, at v0 < 8 it never engaged). Activity superset at the default: at/above 8.0 m/s behavior is bit-identical to the hard gate; below it the fade only adds preview that used to be zero. The acquire-window path is untouched. Legacy rollback sentinel: 8.0 (the spec maximum) reproduces the hard cut exactly |
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

Considered and rejected (2026-07-02, follow limit-cycle target): narrowing the fade to `FadeLo=6 / FadeHi=10` to restore closing lag comp at 13.5 m/s was the proposed fix for the city-speed follow ringing, but after the lead-brake-release recovery fix landed it measures as a no-op on the limit-cycle oracles (highway dip bit-identical; city ease settle unchanged at 12.2 s, both bounds already green) while it would re-admit the long-stop feel this fade was tuned to remove (15 m/s approach stop gap 8.85 m at full comp vs 6.19 m at the shipped 12/18). The 12/18 defaults stay.

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

### Model-Lead Fast-Close Corroboration + Opening Recovery (radard)

Breaks the phantom close-lead ratchet: a single heavy-tailed inward dRel outlier (EV6-measured noise: ~2-3% of frames, 8 m extra sigma) used to satisfy every fast-close gate by itself — the TTC gate is computed from the raw outlier sample and `strong_closing` is true for ANY approach faster than 2.5 m/s — so it was adopted at alpha >= 0.65 in one 50 ms frame, and the 1.2 m/s opening slew could never out-run the still-closing prediction (one-way ratchet: published dRel collapsed 8-16 m below truth and stayed there, slamming the brakes mid-approach). Two changes, both corroboration-based so genuine threats keep their speed:

1. Fast-close adoption now requires `FastCloseConfirmFrames` consecutive qualifying frames (beyond-gate inward innovation plus a support condition). A real cut-in / close threat measures beyond the gate frame after frame; an isolated outlier cannot chain (2 consecutive qualifying outliers ~= 0.01-0.04% per frame pair). During the pending frame the closing-urgency blend still adopts pessimistically at the boosted close slew (~0.5 m/frame), and large genuine cut-ins whose vRel jump breaks the 8 m/s association gate bypass this entirely via instant new-track adoption (measured 0 ms added at a 42->18 m, 9 m/s-closing cut-in; +50 ms at a low-speed same-track cut-in; single-outlier collapse 15.6 m -> 0.47 m). **Confirmation-starvation band — do NOT assume a 2-frame worst-case adoption latency:** genuine same-track jumps of roughly 2.5-3.5 m have their frame-1 urgency-blend adoption pull the next frame's innovation back UNDER the 2.5 m close-innovation gate, so the confirm counter decrements and fast adoption never fires for that band; convergence there is via the urgency blend over ~0.3-0.6 s (measured: 2.4 m optimism -> 1.0 m in 0.3 s at 6 m/s ego). This is legacy-consistent risk: legacy accepted the same optimism magnitude for within-gate <= 2.5 m jumps, which never reached the fast path either.
2. Corroborated low-speed opening recovery: `OpenRecoveryConfirmFrames` consecutive frames of raw measuring more than `OpenRecoveryInnovGateM` FARTHER than the filter state (isolated opening outliers cannot chain that long) prove the state is wrong-too-close, and the filter heals toward the measurement at `OpenRecoveryTauS` (bypassing the opening slew cap) — but only at/below `OpenRecoveryMaxEgoMps` ego speed, where a wrongly optimistic dRel costs little stopping distance. A double-outlier adoption that slips past the confirmation now heals to <2.5 m error in ~1.2 s instead of never.

14-seed ev6_measured sweep of the calm 6 m/s approach to a stopped lead: phantom collapses 2 -> 0, mid-approach slam steps (>= 2 m/s^2 in 0.3 s at true gap > 10 m) 3 -> 0, worst published-vs-true deficit 11.7 m -> 2.2 m. Disable: `FastCloseConfirmFrames=1` restores legacy single-frame adoption; `OpenRecoveryMaxEgoMps=0` disables the recovery exactly, including at standstill (the arming gate requires `MaxEgoMps > 0`).

**Stopping-chain residual — OWNER: follow-up task "Lift composed-tree calm-stop true gap to ~6 m" (M1/M3 stopping chain, NOT this M2 filter).** Re-measured on the frozen composed tree (M1 kinematic ceiling + M2 corroboration + M3 FCW corroboration all active, 2026-07-02): phantom 0/14, slam 0/14, worst published-vs-true deficit 2.17 m, but 11/14 seeds stop at 2.9-4.0 m true gap (seed 99: 3.74 m), under both the repro tests' 4.0 m safety floor and the ~6 m calm-stop feel target. Ownership evidence (same-tree A/B with `FastCloseConfirmFrames=1` + `OpenRecoveryMaxEgoMps=0`): the M2-off sweep shows the identical 2.9-4.0 m short stops on every clean no-phantom seed (5/14: seeds 11, 42, 123, 512, 777) — the stopping chain stops short whenever perception is accurate, and M2's accuracy merely extends that regime to 11/14 seeds (phantom seeds with M2 off stop artificially early/far instead: seed 99 minTrueGap 7.44 m WITH the phantom slam vs 3.74 m fixed). So the residual is the M1/M3 kinematic-ceiling margin/fade composition near standstill, not lead-filter optimism. Conflicting attribution resolved: lowering `ModelLeadFilterOpenRecoveryInnovGateM` toward 1.0 was measured to make true stop gaps 0.1-0.4 m SHORTER (a still-more-accurate published gap lets the planner crawl closer), so it is NOT the fix for this residual despite the published-dRel bleed sitting under that gate. `test_repro_stop_slam_phantom_noise` and `test_repro_stop_slam_fcw_override` stay strict-xfail on this conjunct until the owner task lifts min true gap >= 4.0 m across the seed sweep without regressing the c55a1758d stop-gap fade. **Tracked sub-item (M2/radard-filter follow-up, explicitly open):** the residual low-speed published-dRel bleed itself — published dRel settles ~1.8 m below true at standstill (worst deficit 2.17 m across the 14-seed sweep), sitting under the 2.5 m `ModelLeadFilterOpenRecoveryInnovGateM` so the corroborated opening recovery never arms on it. It is pessimistic-direction (planner believes the lead is closer than truth), so it is NOT the gap-floor owner and must not be "fixed" by lowering the innovation gate (measured: that makes true stop gaps 0.1-0.4 m shorter); it stays tracked here so the two strict-xfails above are not mistaken for closed items — the calm-stop mission is done only when they flip.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ModelLeadFilterFastCloseConfirmFrames` | 2 | 1–6 | Consecutive qualifying 50 ms frames before fast-close adoption. 1 = legacy instant adoption |
| `ModelLeadFilterOpenRecoveryConfirmFrames` | 4 | 1–12 | Consecutive beyond-gate opening frames before the recovery engages |
| `ModelLeadFilterOpenRecoveryTauS` | 0.5 | 0.05–8.0 | Healing tau while the recovery is engaged; >= `ModelLeadFilterTauS` ~= legacy slew-only recovery |
| `ModelLeadFilterOpenRecoveryMaxEgoMps` | 8.0 | 0.0–40.0 | Ego speed at/below which the recovery may engage. 0 disables it entirely |
| `ModelLeadFilterOpenRecoveryInnovGateM` | 2.5 | 0.1–10.0 | Opening innovation a frame must exceed to count toward recovery confirmation. Lowering toward 1.0 also heals the ~1-2 m pessimistic noise-rectification bias near stops (measured 0.1-0.4 m shorter true stop gaps) |

### Model-Lead Association Gate (dPath-primary) + Opening Step Guard (CD4, radard)

CD4 (road 00000200--8cbf2c9481--3, t=51-59 s curve): `ModelLeadTracker`'s association and same-frame-duplicate gates keyed lateral continuity on RAW `yRel` at a shared 3.0 m gate, even though radard already computes the path-relative `dPath`. On a curve the raw `yRel` of a single physical lead drifts (+1.93 -> -8.3 m, 6.7 m excursion) while its `dPath` stays inside +/-0.9 m; the raw-`yRel` gate repeatedly rejected continuity on the SAME lead, churning the published `leadOne` track id (6 events / 4 ids on the fixture) and fabricating a 16.24 m single-frame published dRel step that downstream reads as a real gap change (wrong THW, forced cruise handoffs, seeding CD5/CD6).

The fix gates lateral continuity PRIMARILY on `dPath` (`ModelLeadAssocDPathGateM`, ~1.8 m — a genuinely different lane offset still spawns its own track) and gives the raw `yRel` a separate, LARGER tolerance (`ModelLeadAssocYRawTolM`, ~7 m) instead of the 3.0 m hard reject; `y_err` is also dropped from the same-frame-duplicate merge test (kept `path_err <= 0.8` and `vrel_err <= 2.0`). **Closing-side safety:** when the candidate is CLOSING (raw `vRel` < 0) the raw-`yRel` tolerance is held at the legacy 3.0 m, so the widening applies only to the opening/lane-relevant case and a slow-closing near lead momentarily reading a low `dPath` cannot be masked into a farther track; the `closer_safety_candidate` escape and the 8 m/s vRel gate are untouched, so a genuinely closer/closing threat masquerading as a duplicate is still kept separate. Rollback: set `ModelLeadAssocDPathGateM` and `ModelLeadAssocYRawTolM` both to 3.0 to restore the legacy shared 3.0 m gate exactly (the same-frame-duplicate `y_err` clause remains structurally dropped, matching the fixed tracker).

The published-dRel step guard is pure defense-in-depth: after lag-comp, a published dRel jump FARTHER than `max(ModelLeadStepGuardAbsM, ModelLeadStepGuardFrac*dRel)` in one frame is clamped to that bound while the same track persists with a prior published value. **It is one-sided by construction — it clamps ONLY the OPENING (farther) direction; a closing (nearer) reading is NEVER clamped, so it can never delay or attenuate emergency braking and is exempt from the rate-limit-must-not-delay-emergency-braking rule.** It never binds on a track's first publish (a fresh cut-in publishes its true close dRel unclamped) and never touches internal filter state (publish-only, like the lag comp). Once the association fix is in, the guard never binds on this drive (steps already 0). Disable: either `ModelLeadStepGuardAbsM=0` or `ModelLeadStepGuardFrac=0` restores exact legacy publication.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ModelLeadAssocDPathGateM` | 1.8 | 0.5–6.0 | PRIMARY path-relative lateral-continuity gate for model-lead association. Rollback: 3.0 alongside `ModelLeadAssocYRawTolM`=3.0 restores the legacy shared 3.0 m gate |
| `ModelLeadAssocYRawTolM` | 7.0 | 3.0–12.0 | Separate, larger RAW-yRel tolerance (replaces the 3.0 m hard reject); held at 3.0 m when the candidate is closing (vRel<0). Rollback: 3.0 |
| `ModelLeadStepGuardAbsM` | 3.0 | 0.0–20.0 | Absolute floor (m) of the OPENING-ONLY published-dRel single-frame step bound. Opening-only: never clamps a closing reading, exempt from the emergency-braking rate-limit rule. 0.0 disables the guard |
| `ModelLeadStepGuardFrac` | 0.10 | 0.0–1.0 | dRel-proportional term of the opening-only step bound `max(AbsM, Frac*dRel)`. 0.0 disables the guard |

### Model-Lead FCW Corroboration (radard -> planner crash_cnt veto)

A phantom-collapsed model-lead track (filtered dRel far below what the model keeps measuring, ratcheted by the opening slew) used to drive `mpc.crash_cnt` past its threshold on a calm approach, raising `longitudinalPlan.fcw` -> `VisualAlert.fcw` -> the Hyundai carcontroller `emergency_control()` floor (-5.5 m/s^2 in one 20 ms tick, bypassing all jerk shaping). The tracker now votes each frame on whether the raw measurement corroborates the filtered closeness — only "raw is more than `TolM` FARTHER than the filter" counts as disagreement, so genuine collision courses (where the closing-side EMA lags on the FAR side of raw) always stay corroborated and FCW timing is frame-identical. Fewer than `MinAgree` agreeing frames in the last `Window` frames publishes `radarState.leadX.fcwSuppressed=true`, which vetoes `crash_cnt` accrual in `long_mpc.py` (FCW, not braking: the planner/MPC response to the lead is unaffected). The vote history is seeded agreeing, so brand-new tracks (genuine sudden cut-ins) are never suppressed. Disable: `MinAgree=0` (legacy: any predicted crash with modelProb > 0.9 accrues crash_cnt).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ModelLeadFcwCorrobTolM` | 2.5 | 0.5–50.0 | Raw dRel this far ABOVE the filtered dRel is a disagreement vote (5 sigma of close-range base noise) |
| `ModelLeadFcwCorrobMinAgree` | 2 | 0–8 | Agreeing frames required in the window to stay FCW-eligible. 0 disables suppression (legacy) |
| `ModelLeadFcwCorrobWindow` | 3 | 1–8 | Vote window (50 ms frames); majority vote bridges isolated outward measurement outliers |
| `ModelLeadFcwCorrobRawClosingMinMps` | 1.0 | 0.0–20.0 | Raw-kinematic FCW-corroboration escape: min raw closing speed (m/s) for the raw measurement to hold FCW eligible even when filtered dRel runs more pessimistic than raw (deliberate closing-urgency blend, CD2). A phantom measures raw not closing, so it stays suppressed. 0 disables the escape (legacy raw-vs-filter veto; rollback) |
| `ModelLeadFcwCorrobRawTtcMaxS` | 3.5 | 0.0–15.0 | Raw-kinematic FCW-corroboration escape: max raw-side TTC (s, raw dRel / raw closing) at/under which the raw measurement holds FCW eligible regardless of the filtered-vs-raw delta (CD2). 0 disables the escape (rollback) |

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
| `LeadAccelCorrAmplifyGain` | 1.0 | 0.0–1.0 | CD3 lead-decel truth deficit: fraction of the way to pull an underreported model aLeadK toward the measured vLead trend per frame when both agree the lead brakes. 0 disables (bound-only rollback) |
| `LeadAccelCorrAmplifyDeadbandMps2` | 0.35 | 0.0–3.0 | The vLead trend must be this many m/s^2 more negative than the model aLeadK before amplify engages; rejects steady/lightly-braking finite-difference jitter |
| `LeadAccelCorrAmplifyCapMps2` | 2.0 | 0.0–10.0 | Max m/s^2 amplify may deepen aLeadK below the model report in one frame; bounds a single noisy trend sample |

### Lead Accel Amplify (CD3, model decel underreport)

The same low-passed vLead-trend finite-difference (`a_meas`) that the bound above uses to *cap* an uncorroborated decel is also used to *amplify* a corroborated one. The model's `leadsV3.a` chronically underreports real lead braking (road unit 200-13: published aLeadK peaked -0.48 m/s^2 while the position-derived truth was -1.3..-2.3, and radard's 0.6 s accel EMA halves it again); the vLead trend measures the true decel from vLead alone (in that event it tracked -1.7..-1.9). When the model already reports braking (`aLeadK < 0`) **and** the trend is more negative than the model by more than `AmplifyDeadbandMps2`, aLeadK is pulled a `AmplifyGain` fraction of the way toward the trend (never past it, never by more than `AmplifyCapMps2` in one frame). Unlike the downward bound, amplify runs even in a dangerous/closing state — that is exactly when the deficit is unsafe — because the model-already-negative + sign-agreement gates keep it corroborated; it can only *deepen* an already-reported brake, never fabricate one. On the deficit repro (true decel -1.8, model reports 0.3x) amplify lifts the MPC's effective aLeadK to the truthful value and holds min THW ~0.99 s / min TTC ~4.9 s vs the bound-only 0.49 s / 2.4 s. Noise-checked on `ev6_measured` steady-follow (seeds 11/42/777): zero added brake taps and bit-identical peak brake vs gain 0 — the deadband and model-negative gate fully reject vRel/prob-dropout jitter on a non-braking lead. `AmplifyGain=0` is the exact rollback to bound-only behavior.

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
