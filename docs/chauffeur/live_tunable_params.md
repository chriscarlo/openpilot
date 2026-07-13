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

## Lead-Present Cruise Cap (Far Catch-Up + Kinematic Chase)

While cruise owns the obstacle with a plausible on-path lead within 120 m, positive accel is capped by `get_lead_present_cruise_accel_cap`. bc6c853f9 pinned that cap to the gentle reclaim envelope (~0.32 m/s²) to kill the +2 m/s² lead-present cruise surge; the 2026-07-06 freeway trace showed the flat cap then strands ego once it has fallen well beyond target (steady headway parked at 2.2–3.4 s vs the ~1.55 s target in 5 of 7 captures, with lead loss near ~100–120 m and a full-personality accel discontinuity when the cap vanishes). Two allowances open the ceiling above the gentle cap, combined via `max()` (never summed): the FAR catch-up allowance (surplus-TIME scaled, speed-gated to ≥12 m/s) and the KINEMATIC CHASE allowance (2026-07-08): `gain × (max(0, aLeadK) + pullaway / tau)` — authority sized to match the lead's own acceleration and null the speed deficit over `tau`. The chase term has deliberately NO ego-speed gate: the Event A launch floor is lead-owned-only and fades by 10 m/s ego, and a hard-launching lead (2–3 m/s pullaway) triggers the ownership release in ~2 s, which previously stranded the entire 4–12 m/s (9–27 mph) chase at the flat 0.32 — the "aggravating, traffic-behind-me, driver-must-intervene" stoplight band. Surge protection at low speed comes from the guards that remain: the speed-shaped `speed_cap` (0.7–1.3 m/s² below ~6 m/s), the pullaway-proportional blend, closing-tighten, the closing≥1 m/s coast clamp, and 3 s closing-projection of the surplus.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadPresentCruiseFarCapMps2` | 0.85 | 0.0–2.0 | Ceiling the lead-present cruise cap may grow toward when far beyond target at speed. Trace anchors: the t=61.6 surge row (v=15.3, +2.17 commanded pre-fix) computes to ~0.40 under this default alone; a 25 m/s lead-visible catch-up from >1.6 s beyond target gets the full 0.85. Personality and speed caps still apply on top. Rollback sentinel: any value at or below the gentle reclaim cap (e.g. 0) removes this allowance |
| `LeadPresentCruiseChaseGain` | 1.0 | 0.0–3.0 | Gain K on the kinematic chase allowance: ceiling may rise by K × (aLeadK⁺ + pullaway/tau). Anchors: launch handoff at ego 6 m/s behind a lead at 9 accelerating 1.5 → speed_cap-bounded 1.3 (was 0.32); the 2026-07-06 t=61.6 pullaway-2.05 row → ~1.0 (was +2.17 pre-fix, 0.32 flat-capped). Sentinel: 0 removes the allowance |
| `LeadPresentCruiseChaseTauS` | 3.0 | 0.5–10.0 | Time constant for nulling the pullaway deficit (pullaway/tau term). Smaller = more eager chase. Does not affect the lead-accel matching term |

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

## Approach Ownership TTC Hysteresis

The Hyundai AI-lead-stability path acquires a closing lead when raw TTC-to-headway drops to the approach-reacquire threshold, and (since bc6c853f9's threshold split) can release a still-closing lead back to cruise (`far_closing_cruise`) when TTC rises above a threshold. With a SHARED threshold in both directions, raw TTC frame jitter (~2.5 s p90 on the no-radar EV6 path; 2026-07-06 freeway trace, including 0.1 s double-flips at the boundary) churns ownership during approaches, and each flip arms the handoff limiter / flutter clamp downstream. This knob adds a dead band: release requires TTC above `reacquire_threshold + hysteresis`; inside the band the current owner holds. Acquire-side thresholds are untouched, so handoffs can only get LATER, never earlier — no safety path is delayed. At the freeway steady threshold (4.5 s) the default restores a release boundary (~7.0 s) close to the pre-split 7.5 s release behavior while keeping the new tighter acquire boundary.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ApproachReleaseTtcHysteresisS` | 2.5 | 0.0–6.0 | Gap (s) added above the approach-reacquire TTC threshold before a lead-owned, still-closing follow may release to cruise. Sized to the observed ~2.5 s p90 raw TTC frame jitter. Rollback sentinel: 0 restores the shared-threshold (pre-hysteresis) behavior exactly |

## Lead Brake Release

Vibe-follow-only accel floor that prevents continued heavy decel after the Vibe headway target has recovered or is about to recover. Safety gating uses relative closing distance against the planner's available negative accel (`-6 m/s²` on this Hyundai/EV6 GT path), while hard lead decel still blocks release.

Follow limit-cycle fix (2026-07-02, seat report: buck/slow/hold/late-re-accel/overshoot during steady follow): the gap error the release path measures recovery against is now vRel-aware — the MPC's own `desired_follow_distance(v_ego, v_lead, t_follow)` — implemented as a CREDIT of `max(0, vLead² - vEgo²)/(2·COMFORT_BRAKE)` on top of the legacy headway error. The credit basis takes the more pessimistic of `vLead` and `vEgo + vRel` (any closing state gets zero credit, so no closing state is looser than legacy), is projected forward by `max(0, aLeadK) · RecoveryProjS` to cover the tracker's vRel lag behind aLeadK when a lead finishes a transient slowdown, and saturates at `VrelCreditCapM`. Once recovered with the lead pulling away, the floor also rises above the coast bias toward `GapReclaimFollowMaxAccel` (phased in over 0.35–1.25 m/s of measured pullaway, tapered by the kinematic overshoot bound scaled by `GapReclaimTaperGain`) so ego matches lead speed BEFORE the equal-speed gap is regained instead of crawling at the MPC unwind jerk and then overshooting. The coast-window branch is limited to closing speeds at/below `LeadBrakeReleaseNearTargetMaxClosingMps`; faster closure stays on the kinematic `closing_to_target` floor. When the floor is applied, the M1 slowdown ceiling is re-applied only for a braking lead or closure above that same threshold. This preserves real-threat authority without letting a lagged few-tenths-m/s closing estimate nullify recovery projection and keep braking an already-opening gap. Measured on the current noise-off oracle (`test_repro_follow_limit_cycle.py`): highway 1.5 m/s dip at 30 m/s — post-recovery brake hold 0.25 → 0.00 s planner / 1.20 → 0.30 s realized, max re-accel deficit 1.376 → 0.954 m/s, rebound overshoot −4.02 → −2.35 m; city 0.4 m/s² ease at 13.5 m/s — ringing 2 → 1 crossing, settle 19.8 → 11.8 s. Mid-window threat probes still route braking leads and fast closure through the more-braking path. Rollback: `VrelCreditCapM=0` + `RecoveryProjS=0` + `GapReclaimFollowMaxAccel=0` restores the legacy release behavior.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadBrakeReleaseMinSpeedMps` | 5.0 | 0.0–20.0 | Minimum ego speed for brake release |
| `LeadBrakeReleaseBrakeDeficitMarginM` | 1.5 | 0.0–10.0 | Relative-braking-distance deficit allowed before release stays disabled |
| `LeadBrakeReleaseLookaheadS` | 2.0 | 0.1–6.0 | Projected time-to-target window for easing continued decel |
| `LeadBrakeReleaseMinPullawayMps` | 0.10 | 0.0–3.0 | Minimum opening speed for projected-recovery release |
| `LeadBrakeReleaseNearTargetMarginM` | 1.5 | 0.0–8.0 | Headway deficit treated as near target when closing is small |
| `LeadBrakeReleaseNearTargetMaxClosingMps` | 0.75 | 0.0–4.0 | Max closing speed eligible for near-target release or the recovered-gap coast window. Faster closure keeps the kinematic slowdown ceiling authoritative |
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
2. Corroborated opening recovery: `OpenRecoveryConfirmFrames` consecutive frames of raw measuring more than `OpenRecoveryInnovGateM` FARTHER than the filter state (isolated opening outliers cannot chain that long) prove the state is wrong-too-close, and the filter heals toward the measurement at `OpenRecoveryTauS` (bypassing the opening slew cap). At/below `OpenRecoveryMaxEgoMps`, repeated position evidence is sufficient. Above it, recovery additionally requires a non-braking raw lead and raw TTC ≥ 6 s, so a freeway inward outlier can heal without relaxing any nearby or braking threat. A double-outlier adoption that slips past confirmation now heals instead of ratcheting indefinitely.

14-seed ev6_measured sweep of the calm 6 m/s approach to a stopped lead: phantom collapses 2 -> 0, mid-approach slam steps (>= 2 m/s^2 in 0.3 s at true gap > 10 m) 3 -> 0, worst published-vs-true deficit 11.7 m -> 2.2 m. Disable: `FastCloseConfirmFrames=1` restores legacy single-frame adoption; `OpenRecoveryMaxEgoMps=0` disables the recovery exactly, including at standstill (the arming gate requires `MaxEgoMps > 0`).

**Stopping-chain residual — OWNER: follow-up task "Lift composed-tree calm-stop true gap to ~6 m" (M1/M3 stopping chain, NOT this M2 filter).** Re-measured on the frozen composed tree (M1 kinematic ceiling + M2 corroboration + M3 FCW corroboration all active, 2026-07-02): phantom 0/14, slam 0/14, worst published-vs-true deficit 2.17 m, but 11/14 seeds stop at 2.9-4.0 m true gap (seed 99: 3.74 m), under both the repro tests' 4.0 m safety floor and the ~6 m calm-stop feel target. Ownership evidence (same-tree A/B with `FastCloseConfirmFrames=1` + `OpenRecoveryMaxEgoMps=0`): the M2-off sweep shows the identical 2.9-4.0 m short stops on every clean no-phantom seed (5/14: seeds 11, 42, 123, 512, 777) — the stopping chain stops short whenever perception is accurate, and M2's accuracy merely extends that regime to 11/14 seeds (phantom seeds with M2 off stop artificially early/far instead: seed 99 minTrueGap 7.44 m WITH the phantom slam vs 3.74 m fixed). So the residual is the M1/M3 kinematic-ceiling margin/fade composition near standstill, not lead-filter optimism. Conflicting attribution resolved: lowering `ModelLeadFilterOpenRecoveryInnovGateM` toward 1.0 was measured to make true stop gaps 0.1-0.4 m SHORTER (a still-more-accurate published gap lets the planner crawl closer), so it is NOT the fix for this residual despite the published-dRel bleed sitting under that gate. `test_repro_stop_slam_phantom_noise` and `test_repro_stop_slam_fcw_override` stay strict-xfail on this conjunct until the owner task lifts min true gap >= 4.0 m across the seed sweep without regressing the c55a1758d stop-gap fade. **Tracked sub-item (M2/radard-filter follow-up, explicitly open):** the residual low-speed published-dRel bleed itself — published dRel settles ~1.8 m below true at standstill (worst deficit 2.17 m across the 14-seed sweep), sitting under the 2.5 m `ModelLeadFilterOpenRecoveryInnovGateM` so the corroborated opening recovery never arms on it. It is pessimistic-direction (planner believes the lead is closer than truth), so it is NOT the gap-floor owner and must not be "fixed" by lowering the innovation gate (measured: that makes true stop gaps 0.1-0.4 m shorter); it stays tracked here so the two strict-xfails above are not mistaken for closed items — the calm-stop mission is done only when they flip.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ModelLeadFilterFastCloseConfirmFrames` | 2 | 1–6 | Consecutive qualifying 50 ms frames before fast-close adoption. 1 = legacy instant adoption |
| `ModelLeadFilterOpenRecoveryConfirmFrames` | 4 | 1–12 | Consecutive beyond-gate opening frames before the recovery engages |
| `ModelLeadFilterOpenRecoveryTauS` | 0.5 | 0.05–8.0 | Healing tau while the recovery is engaged; >= `ModelLeadFilterTauS` ~= legacy slew-only recovery |
| `ModelLeadFilterOpenRecoveryMaxEgoMps` | 8.0 | 0.0–40.0 | Ego speed at/below which repeated position evidence alone may engage recovery. Above it, non-braking-lead and safe-raw-TTC gates are also required. 0 disables recovery entirely |
| `ModelLeadFilterOpenRecoveryInnovGateM` | 2.5 | 0.1–10.0 | Opening innovation a frame must exceed to count toward recovery confirmation. Lowering toward 1.0 also heals the ~1-2 m pessimistic noise-rectification bias near stops (measured 0.1-0.4 m shorter true stop gaps) |

### Model-Lead Association Gate (dPath-primary) + Opening Step Guard (CD4, radard)

CD4 (road 00000200--8cbf2c9481--3, t=51-59 s curve): `ModelLeadTracker`'s association and same-frame-duplicate gates keyed lateral continuity on RAW `yRel` at a shared 3.0 m gate, even though radard already computes the path-relative `dPath`. On a curve the raw `yRel` of a single physical lead drifts (+1.93 -> -8.3 m, 6.7 m excursion) while its `dPath` stays inside +/-0.9 m; the raw-`yRel` gate repeatedly rejected continuity on the SAME lead, churning the published `leadOne` track id (6 events / 4 ids on the fixture) and fabricating a 16.24 m single-frame published dRel step that downstream reads as a real gap change (wrong THW, forced cruise handoffs, seeding CD5/CD6).

The fix gates lateral continuity PRIMARILY on `dPath` (`ModelLeadAssocDPathGateM`, ~1.8 m — a genuinely different lane offset still spawns its own track) and gives the raw `yRel` a separate, LARGER tolerance (`ModelLeadAssocYRawTolM`, ~7 m) instead of the 3.0 m hard reject; `y_err` is also dropped from the same-frame-duplicate merge test (kept `path_err <= 0.8` and `vrel_err <= 2.0`). **Closing-side safety:** when the candidate is CLOSING (raw `vRel` < 0) the raw-`yRel` tolerance is held at the legacy 3.0 m, so the widening applies only to the opening/lane-relevant case and a slow-closing near lead momentarily reading a low `dPath` cannot be masked into a farther track; the `closer_safety_candidate` escape and the 8 m/s vRel gate are untouched, so a genuinely closer/closing threat masquerading as a duplicate is still kept separate. Rollback: set `ModelLeadAssocDPathGateM` and `ModelLeadAssocYRawTolM` both to 3.0 to restore the legacy shared 3.0 m gate exactly. The same-frame-duplicate `y_err` clause (which the fix dropped) is re-gated on the SAME `ModelLeadAssocYRawTolM` knob: while it is at/below the legacy 3.0 m gate (the rollback sentinel) the pre-fix `y_err <= 0.8` duplicate-merge requirement is restored, so the sentinel FULLY reproduces the pre-fix churn (verified: 6 churn events / 4 ids / 6 fabricated dRel steps, matching the legacy reconstruction). At the default (7.0) the clause stays dropped, so default behavior is bit-identical to the shipped fix.

The published-dRel step guard is pure defense-in-depth: after lag-comp, a published dRel jump FARTHER than `max(ModelLeadStepGuardAbsM, ModelLeadStepGuardFrac*dRel)` in one frame is clamped to that bound while the same track persists with a prior published value. **It is one-sided by construction — it clamps ONLY the OPENING (farther) direction; a closing (nearer) reading is NEVER clamped, so it can never delay or attenuate emergency braking and is exempt from the rate-limit-must-not-delay-emergency-braking rule.** It never binds on a track's first publish (a fresh cut-in publishes its true close dRel unclamped) and never touches internal filter state (publish-only, like the lag comp). Once the association fix is in, the guard never binds on this drive (steps already 0). Disable: either `ModelLeadStepGuardAbsM=0` or `ModelLeadStepGuardFrac=0` restores exact legacy publication.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ModelLeadAssocDPathGateM` | 1.8 | 0.5–6.0 | PRIMARY path-relative lateral-continuity gate for model-lead association. Rollback: 3.0 alongside `ModelLeadAssocYRawTolM`=3.0 restores the legacy shared 3.0 m gate |
| `ModelLeadAssocYRawTolM` | 7.0 | 3.0–12.0 | Separate, larger RAW-yRel tolerance (replaces the 3.0 m hard reject); held at 3.0 m when the candidate is closing (vRel<0). Rollback: 3.0 |
| `ModelLeadStepGuardAbsM` | 3.0 | 0.0–20.0 | Absolute floor (m) of the OPENING-ONLY published-dRel single-frame step bound. Opening-only: never clamps a closing reading, exempt from the emergency-braking rate-limit rule. 0.0 disables the guard |
| `ModelLeadStepGuardFrac` | 0.10 | 0.0–1.0 | dRel-proportional term of the opening-only step bound `max(AbsM, Frac*dRel)`. 0.0 disables the guard |

### Far-Range Stopped-Traffic vLead Optimism Clamp (CD8, radard)

CD8 (road 200-13 EDGE2 — the successful-but-late FCW stop): while a far, newly-acquired stopped/slow lead is still stopping, the model's published `vLead` runs biased HIGH (road ~+4 m/s vs position-derived truth through the 43-48 s window). The kinematic stopping-need handoff term (`compute_lead_stopping_need_decel`, which uses `vLead` to size the closure) therefore computes a much smaller required decel than reality, so the cruise->lead handoff / braking starts late and the stop is forced into a concentrated hard brake (road: `aTarget` plateaued -1.5 for 2 s then dove to -4.76 with `carOutput` saturating -5.50 for 1.6 s; 42% of the kinetic energy shed in the last 3 s vs a -1.91 m/s² constant-decel ideal).

The fix (radard `ModelLeadTrack.update`, publish-time only) clamps the published `vLead`: when the RAW model lead velocity is declining monotonically across `LeadVLeadOptimismClampConfirmFrames`+1 recent frames (a stopping/decelerating lead) AND the range is beyond `LeadVLeadOptimismClampRangeM` AND the position-derived lead velocity is a genuinely slow/stopping lead (at/below `LeadVLeadOptimismClampSlowLeadFrac` * ego speed — which excludes a fast steady lead that suffered a transient vLead dip, the CD6 vLeadK-rollover non-regression), the published `vLead` is pulled toward the position-derived estimate `d(dRel)/dt + v_ego` (from the internal-filter dRel trend) by `LeadVLeadOptimismClampGain`, then `min()`'d against the model `vLead`. The car sees the truth (a slower/stopping lead) sooner, so the existing `LeadHandoffStoppingNeedDecelMps2` leg engages ~2 s earlier and the brake spreads. **SAFETY: strictly one-directional — it only ever makes the published `vLead` SLOWER / more urgent (bias toward earlier braking), NEVER faster / less urgent.** It is publish-time only (internal EMA state untouched, no feedback lag) and tightly guarded so it does NOT fire on a genuinely moving/steady lead (raw `vLead` not monotonically declining, or position-derived vLead above the slow-lead fraction of ego speed), on noise (a sustained multi-frame decline is required, not one frame), or on a near/normal follow (range below the far threshold). It does not synthesize a lead decel into `aLeadK` — the stopping-need trigger uses `vLead` directly, so correcting `vLead` is sufficient and avoids fabricating a decel the model never measured. Rollback sentinel: `LeadVLeadOptimismClampRangeM = 1e9` (range unreachable) OR `LeadVLeadOptimismClampGain = 0` disables the clamp exactly (legacy publish).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadVLeadOptimismClampRangeM` | 55.0 | 20.0–1e9 | Far range (m) beyond which the published-vLead optimism clamp arms. Rollback sentinel: `1e9` (the spec maximum deliberately admits it) makes the range unreachable and disables the clamp |
| `LeadVLeadOptimismClampGain` | 1.0 | 0.0–1.0 | Fraction of the way the published vLead is pulled toward the position-derived `d(dRel)/dt + v_ego` estimate when the clamp fires; still `min()`'d against the model vLead so it can only ever LOWER the published vLead. Rollback sentinel: `0.0` disables the clamp |
| `LeadVLeadOptimismClampSlowLeadFrac` | 0.5 | 0.0–1.0 | The clamp fires only when the position-derived lead velocity is at/below this fraction of ego speed (a genuinely slow/stopping lead), so a fast steady lead with a transient vLead dip (a vLeadK rollover) is excluded. Raise toward 1.0 to admit any lead slower than ego |
| `LeadVLeadOptimismClampConfirmFrames` | 3.0 | 1.0–6.0 | Consecutive RAW-model-vLead decline frames required before the clamp arms, so one noisy frame cannot trigger it and a flat/rising (steady/moving) raw vLead never qualifies |

### Corroborated-Closing Governor (CD9, radard)

CD9 (road 205-6 Event B — the 2026-07-04 near-collision, driver stomped at THW 1.10 s): the model's RAW streams showed a braking lead on time (sustained raw `aLead` −0.55 from onset, raw `vRel` closing, raw dRel collapsing at 4–6 m/s), but the published state ran a compounded EMA lag behind them — `vRel` tau (`ModelLeadFilterVRelTauS`, 0.60 live that drive) gated below the urgency blend's `BlendCloseLoMps`, dRel close-tau `ModelLeadFilterTauS`×1.6 under the ~1 m/s close-slew clamp, `aLeadK` 0.60 s tau halving the published decel — so the planner's brake ramp trailed the closure by ~1.9 s. CD3's amplify could not recover it (its corroborating trend reads the lagging published vLead) and CD8 correctly sat out (far-range stopped-traffic gates). This is the closing half of the noise-vs-closing tension: raising `ModelLeadFilterVRelTauS` to damp steady-follow noise braking directly worsened this closing latency until CD9 decoupled them.

The governor (radard `ModelLeadTrack._update_closing_governor`) keeps a windowed `(t, raw dRel, raw vRel, raw aLead)` evidence deque and latches when EITHER the endpoint-mean position slope of the raw dRel window closes faster than the currently PUBLISHED closing speed by `ClosingGovernorMarginMps` (position-excess path), OR the windowed mean raw lead accel is below −`ClosingGovernorAccelOnsetMps2` (sustained-decel path, the earliest reliable road signal) — BOTH paths requiring the windowed raw vRel to agree the lead is closing by more than `ClosingGovernorMinClosingMps`. While latched (held `ClosingGovernorHoldS` past the last qualifying frame), closing urgency is forced to 1, `aLeadK` runs at `ClosingGovernorALeadTauS`, and the published `vLead` is one-directionally clamped toward corroborated closure. Full `ClosingGovernorPosTrustExcessMps` authority requires independent evidence from sustained raw lead braking or a short TTC computed from current raw vRel; a TTC computed from the same position slope cannot corroborate itself. Otherwise position authority grows continuously only by windowed raw-vRel closure above `ClosingGovernorMarginMps`; one unconfirmed current raw-aLead sample may add only magnitude-based bridge authority capped by `ClosingGovernorUnconfirmedALeadTrustMps`. **SAFETY: nothing is fabricated — the fast taus still filter the model's own measurements and the clamp can only make the published lead SLOWER / more urgent.** Noise immunity is structural: single heavy-tail dRel outliers (road ±1.5–3 m frames) cannot dominate k-endpoint means over the window, and a raw-position collapse with calm vRel/aLead remains capped at velocity evidence. Rollback sentinels: `ClosingGovernorMarginMps >= 99` disables the governor entirely; `ClosingGovernorAccelOnsetMps2 >= 99` disables the decel arm path alone; `ClosingGovernorUnconfirmedALeadTrustMps = 0` disables only the one-frame bridge. Harnesses: the road-braking repro retains its 1.00 s minimum-THW floor, while the route-22a position-collapse repro removes the pre-fix −1.00 m/s² false brake and 1.68 m true-gap expansion.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ClosingGovernorMarginMps` | 0.75 | 0.05–100.0 | Position-excess arm margin (m/s): windowed position closure must exceed the published closing speed by this. Master rollback sentinel: `>= 99` disables the whole governor |
| `ClosingGovernorMinClosingMps` | 0.30 | 0.05–5.0 | Windowed raw-vRel closing agreement required by BOTH arm paths; keeps opening/steady follows and pure position-noise runs from latching |
| `ClosingGovernorWindowS` | 0.60 | 0.20–2.0 | Evidence window (s). Longer = more noise immunity but later latch and ~window/2 × closure-accel estimate lag. Sweep: 0.5 put the steady heavy-tail excursion at −0.53, uncomfortably near the −0.6 line |
| `ClosingGovernorAccelOnsetMps2` | 0.35 | 0.05–100.0 | Sustained-decel arm path: windowed mean raw lead accel below −this arms the governor (road separation: −0.55 braking vs −0.10 steady). Sentinel `>= 99` disables this path only |
| `ClosingGovernorPosTrustExcessMps` | 1.5 | 0.0–100.0 | Maximum position-derived closure authority beyond vRel evidence. Full authority requires sustained raw lead braking or a short current-raw-vRel TTC; otherwise velocity-earned authority grows continuously above `ClosingGovernorMarginMps`. 0.0 = strict min(position, vRel) |
| `ClosingGovernorUnconfirmedALeadTrustMps` | 0.41 | 0.0–1.5 | Cap on magnitude-based position authority from one unconfirmed raw-aLead braking sample while waiting one model frame for confirmation. Default is the smallest rounded value preserving the road-braking 1.00 s THW floor; 0.0 disables only this bridge |
| `ClosingGovernorHoldS` | 1.0 | 0.10–5.0 | Latch hold (s) past the last qualifying frame so the fast regime does not chatter mid-closure |
| `ClosingGovernorALeadTauS` | 0.18 | 0.05–2.0 | `aLeadK` EMA tau while latched (replaces the 0.60 s constant that halved the published decel through the road event) |

### Opening Governor (radard, CD9's mirror)

The publish pipeline's safety asymmetry (lag comp boosts vRel closing-ward, CD9 clamps vLead slower-ward, and opening dRel is normally slew-clamped) can make opening truth lag. Measured on the 2026-07-08 drive: **22.3% of lead-tracking frames published `vRel` ≤ −1.0 while the raw position stream showed the gap OPENING ≥ 0.2 m/s**, in 111 sustained runs up to 6.9 s, with the planner braking through 27% of those frames. Felt as "rides the brakes until a ~3 s gap builds"; also most of the ~2 s follow floor, because the MPC's desired distance carries a `(vEgo²−vLead²)/2·COMFORT_BRAKE` term — a phantom 2 m/s closing adds ~17 m (~0.7 s) of desired gap at freeway speed. The closing governor's 1 s dropout hold now releases early when its full raw-vRel window falls below half the minimum closing threshold and the lead is not braking; this removes stale held closure while leaving the ordinary filtered track and MPC safety paths active.

The opening governor watches the SAME raw evidence window CD9 trusts and, when the k-endpoint mean slope of raw dRel proves a sustained opening with no threat veto standing, floors the published `vRel` at `min(pos_opening − TrustDeficitMps, 0.0)` — one-directional (`max()` at publish: only ever less urgent), capped at parity, publish-time only (internal EMA/association state untouched). Exact mirror of CD9's `min()` clamp with the same position-primacy rationale. Vetoes (any → no relax): CD9 latched/holding, windowed raw vRel closing beyond `RawClosingVetoMps`, windowed raw aLead below −`ALeadVetoMps2`, published-closing TTC under 6 s (module constant `OPENING_GOVERNOR_MIN_PUBLISHED_TTC_S`), sparse window, missed frame (stale relax cleared on coasted publishes).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `OpeningGovernorTrustDeficitMps` | 0.3 | 0.0–100.0 | How far behind the position-proven opening rate the publish may stay: floor = min(pos_opening − this, 0). Master rollback sentinel: ≥ 99 disables the opening governor entirely |
| `OpeningGovernorMinOpeningMps` | 0.2 | 0.05–5.0 | Windowed raw-dRel slope must show opening at/above this before any relax arms (matches the 2026-07-08 phantom-run detector) |
| `OpeningGovernorRawClosingVetoMps` | 1.0 | 0.0–100.0 | Veto: windowed raw vRel mean closing beyond this blocks the relax (model's own velocity stream strongly disagrees → resolve toward braking) |
| `OpeningGovernorALeadVetoMps2` | 0.2 | 0.0–100.0 | Veto: windowed raw lead accel mean below −this (braking lead) blocks the relax |

### Stop-Launch Release + Launch-Follow Demand Floor (Event A, planner/longcontrol)

Event A (road 205-13 — the 2026-07-04 failed launch, driver pedaled): the stop latch held the full −2.0 stopAccel ~1.2 s after the lead visibly departed because the release's ABSOLUTE `dRel >= 5.0` arming gate made release latency depend on where the stop happened to settle (published 4.15 m that day → the lead had to open 0.85 m of slew-lagged published gap before arming); then the launch demand never exceeded +1.02 while the lead departed at +5 m/s — the MPC's jerk-shaped standstill ramp owned the launch window, the M1 slowdown ceiling's slew-limited release tail capped even that, and `get_low_speed_launch_follow_max_accel` only raises the permission CEILING, never the demand.

Three coordinated changes: (1) the release arming gate is now `min(LaunchReleaseMinDrelM, stop-settle-minimum published dRel + LaunchReleaseDepartGateM)` with a hard 2.0 m floor — departure EVIDENCE arms it (the planner tracks the per-stop minimum and resets when not at standstill); the pullaway-speed, `a_target > 0` and hold-frame conditions are unchanged, and the 0.20 m default deliberately sits above the road-observed ~0.1 m/s stopped-publish creep so sitting at a light cannot arm on creep. (2) The planner floors its output accel at `get_low_speed_launch_follow_factor × LaunchFollowAccelFloorMaxMps2` — the factor already scales by ego speed, lead speed, pullaway and gap surplus and is exactly 0 unless the lead is genuinely pulling away — applied AFTER the M1 slowdown ceiling and only while the lead is NOT threatening (`aLeadK >= 0` and `vRel >= 0`), so a real threat keeps the ceiling's authority. (3) longcontrol's `starting` state commands `max(startAccel, a_target)` so the floored demand reaches the CAN command through the EV6's ~0.85 s standstill dead zone; with the floor's 0.0 sentinel, `a_target` stays below `startAccel` there and this reduces to the legacy constant exactly. Rollback sentinels: `LaunchReleaseDepartGateM >= 99` restores the pure absolute arming; `LaunchFollowAccelFloorMaxMps2 = 0` disables the demand floor (and the passthrough's effect with it). Harness (road-205-13 repro, `test_repro_stop_launch_release.py`): release delay 0.95 → 0.65 s, peak launch demand +0.81 → +1.34, v@onset+3.5 s 1.21 → 1.96 m/s; fix-vs-rollback twins oracled.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LaunchReleaseMinDrelM` | 5.0 | 2.0–30.0 | Absolute published-dRel arming gate (m) for the stop-latch release on a lead launch (the legacy hardcoded 5.0); the arming requirement is the SMALLER of this and the departure-relative gate |
| `LaunchReleaseDepartGateM` | 0.20 | 0.0–100.0 | Departure-relative arming gate (m): published dRel rise above this stop's settle minimum that proves the lead is departing. Sentinel `>= 99` restores pure absolute arming |
| `LaunchFollowAccelFloorMaxMps2` | 1.8 | 0.0–2.5 | Launch-follow demand floor ceiling (m/s²): planner output accel is floored at launch-follow-factor × this while a lead departs at low ego speed. Sentinel `0.0` disables the floor (exact legacy demand) |

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

## CD5: Lost-vs-Departed Reacquire Memory + Relatch Obstacle Blend

Two composed mechanisms on top of the cruise-reacquire ramp, both live-tunable with rollback sentinels. Every added negative-leg limiter is bypassed under any urgency signal so genuine braking is never delayed; the collapse holdback only ever constrains the POSITIVE (cruise re-accel) leg.

**(b) Exit-cause classifier + collapse holdback.** At a lead0/lead1 → cruise handoff the departing lead's PUBLISHED `radarState` modelProb history classifies WHY it left: a *prob-collapse* (perception faded gradually — the last status-True published prob is low, near the Schmitt exit band) vs a *genuine departure* (an abrupt prob cliff with the prob still high). After a collapse-exit the reacquire ramp term is pinned at 0 (allowed jerk held at `CruiseReacquirePosJerkLimit`) for `CruiseCollapseHoldbackS`, so a phantom perception dropout does not license the full cruise re-acceleration escalation a real departure would (road ff4 +0.72 surge). A genuine departure keeps the full ramp. The holdback resets to 0 the instant a corroborated closing/threatening lead relatches (a collapse followed by a real re-approach is never held back). Ambiguity fails safe to "departure" (full ramp).

**(a) Relatch obstacle blend.** After a genuine lead→cruise handoff, a same-physical-lead relatch (matched by `radarTrackId` or a dRel-continuous re-presentation of the departed lead — a fresh cut-in track is never blended) has its fresh ObstacleCost slew-blended into `output_a_target` over `CruiseRelatchBlendS` instead of slamming in one frame (road 200-9 tap2: 2.7 m/s^2 swing in 270 ms). The DOWNWARD (brake-onset) leg is jerk-capped by `CruiseRelatchBlendJerkMps3` and is fully bypassed under any urgency signal (FCW, TTC ≤ `CruiseRelatchUrgentTtcS`, KINEMATICALLY REQUIRED decel ≥ |`CruiseRelatchBypassDecelMps2`|, raw closing ≥ `CruiseRelatchUrgentClosingMps` backstop, requested decel ≤ `CruiseRelatchBypassDecelMps2`, or lead aLeadK ≤ `CruiseRelatchUrgentLeadDecelMps2`). The UPWARD (brake-RELEASE) leg is jerk-capped by `CruiseRelatchReleaseJerkMps3` to smooth the release blip as the obstacle cost settles — this is always-safe (it only ever keeps MORE brake, never delays onset) so it is NOT bypassed. While blended, peak decel is bounded by a KINEMATIC cap: `min(CruiseRelatchMaxDecelMps2, -CruiseRelatchKinematicHeadroom × required_decel)` where `required_decel = compute_relatch_required_decel` (larger of closing shed within the gap surplus above the follow target, and the stopping-need decel). Continuous in the requirement — a routine far acquire glides at the flat floor, a genuine approach opens exactly proportional authority, and no bypass-threshold crossing produces a comfort cliff because the cap has already converged to the MPC demand on the way there. 2026-07-08 rationale: the previous closing-alone bypass (2.5 m/s) classified every routine freeway acquire (closing 2.6–5.3 m/s at TTC 12–21 s, 5/5 in the drive-home rlogs) as urgent, voiding the clamp exactly at acquisition and producing the measured "never follows closer than 2.5 s" floor. Only ONE binding negative-leg slew clamp exists per frame (this one runs last; the flutter clamp needs ≥2 transitions and cannot bind on a single relatch frame).

Rollback sentinels (restore exact pre-CD5 behavior): `CruiseCollapseHoldbackS=0` (ramp escalates regardless of exit cause), `CruiseRelatchBlendS=0` (hard obstacle swap), `CruiseRelatchBlendJerkMps3=0` and `CruiseRelatchReleaseJerkMps3=0` (both legs untouched). Setting all four to 0 is the verified full CD5 rollback (slam returns to ~1.6 m/s^2 one-frame, collapse ramp escalates to ~0.72 like a departure). Additionally `CruiseRelatchKinematicHeadroom=0` disables only the kinematic cap extension (flat `CruiseRelatchMaxDecelMps2` cap, post-CD5 pre-kinematic behavior).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `CruiseCollapseHoldbackS` | 2.0 | 0.0–5.0 | Seconds after a prob-COLLAPSE exit to pin the reacquire jerk at `CruiseReacquirePosJerkLimit` (no ramp escalation). 0 = pre-CD5 (ramp escalates regardless of exit cause) |
| `CruiseExitLookbackFrames` | 7 | 1–20 | Frames of published departing-lead history retained at a lead→cruise exit (confirms prob ended below the Schmitt exit band) |
| `CruiseExitAbruptProbDrop` | 0.3 | 0.05–1.0 | Last-status-True published prob at/above `1 − this` = abrupt track cliff → DEPARTURE; below = gradual fade → COLLAPSE. Raise to make collapse-classification (and the holdback) less eager |
| `CruiseRelatchBlendS` | 1.5 | 0.0–3.0 | Duration (s) the relatch blend stays armed after a same-lead cruise→lead relatch (spans the obstacle-cost settle). 0 = pre-CD5 hard obstacle swap |
| `CruiseRelatchBlendJerkMps3` | 2.0 | 0.0–10.0 | Downward (brake-onset) jerk cap during the relatch blend; urgency-bypassed. 0 = no downward slew |
| `CruiseRelatchReleaseJerkMps3` | 2.0 | 0.0–10.0 | Upward (brake-RELEASE) jerk cap during the relatch blend; always-safe (keeps more brake, never delays onset), NOT bypassed. 0 = release leg untouched (pre-CD5) |
| `CruiseRelatchUrgentTtcS` | 4.0 | 0.0–15.0 | Relatch TTC (s) at/below which the blend AND large-TTC decel cap are bypassed (full braking passes immediately) |
| `CruiseRelatchUrgentClosingMps` | 8.0 | 0.0–20.0 | Raw-closing BACKSTOP (m/s) at/above which the blend AND decel cap are bypassed regardless of geometry. Raised 2.5→8.0 (2026-07-08): closing-alone at 2.5 flagged every routine acquire as urgent; the kinematic bypass now owns the routine range, this catches sensor-odd fast closes only |
| `CruiseRelatchBypassDecelMps2` | -1.5 | -5.0–0.0 | Decel floor (m/s^2) at/below which the blend is bypassed — by a REQUESTED decel or by the KINEMATICALLY REQUIRED decel of the approach. One floor, two currencies: decels beyond it are never gated whether the MPC requests them or physics demands them. 0 disables both legs |
| `CruiseRelatchUrgentLeadDecelMps2` | -1.0 | -5.0–0.0 | Relatched-lead aLeadK (m/s^2) at/below which the blend AND decel cap are bypassed (anticipatory braking toward a decelerating lead; TTC/closing/FCW lag it) |
| `CruiseRelatchMaxDecelMps2` | -0.15 | -5.0–0.0 | FLAT comfort floor of the relatch decel cap on a non-urgent relatch; the kinematic extension opens beyond it proportionally. Lowered -0.8→-0.15 (2026-07-08): the kinematic term carries real approaches, the flat term only covers the glide-in. Removed by the urgency bypass. 0 = no cap |
| `CruiseRelatchKinematicHeadroom` | 1.5 | 0.0–5.0 | Multiplier K on the kinematically required decel the cap may open to: `min(flat, -K × required)`. Trace anchors: closing 2.72 / 14.7 m surplus → cap -0.38; closing 4.16 / 21.7 m surplus → cap -0.60 (vs the -1.10/-1.99 those events actually braked). Governs the comfort band above `CruiseRelatchBypassDecelMps2` only. 0 = flat cap only (sentinel) |

## CD6: Symmetric Post-Transition Handoff Limiter

Two mechanisms on `output_a_target` in `longitudinal_planner.py`, applied as the FINAL composed limiter (after the CD5 relatch blend). Both address the felt VACILLATION at a cruise↔lead source boundary.

**(1) Symmetric windowed delta clamp (road 200-6).** A vLeadK rollover on a FAR non-hazard lead (dRel > 75 m, TTC > 100 s) briefly makes the fresh lead0 obstacle undercut cruise; the MPC's cruise-owned accel cap COLLAPSES from a high value to ~0 in one frame and slams `aTarget` from ~+0.58 to ~−0.56 — one to a few frames BEFORE the source label flips to lead0. The window arms on (a) a real cruise↔lead source flip OR (b) a large single-frame drop in the MPC cruise-owned accel cap (the precise pre-flip-dive signal; a far slow lead the MPC steadily suppresses keeps the cap ~0 and never arms). While armed for `HandoffLimitWindowS`, `|output_a_target − prev_a|` is bounded to `HandoffLimitMaxDeltaMps2` per frame. **Asymmetric bypass:** the UPWARD (accel-increasing) leg ALWAYS applies (limiting acceleration is always safe); the DOWNWARD (braking) leg is bounded only when it enters braking territory (target < 0) and is bypassed under the shared relatch urgency signal (FCW / short-TTC / fast-close on the owned or approaching lead, or a requested hard decel) so emergency braking is NEVER delayed. Measured (noise-off oracle): the road's ~−1.13 m/s² one-frame flip and 1.15 m/s² cc-vs-co divergence drop to −0.30 / 0.32.

**(2) EDGE1 inside-df positive-accel cap (road 200-13 phase-1).** While the source is cruise and a slower lead is already CLEARLY inside the desired follow distance (`gap < 0.9·desired_follow_distance(v_ego, v_lead, t_follow)`) on a closing (ego-faster) trend, positive `aTarget` is capped to `HandoffInsideDfPositiveCapMps2` so the planner stops spending headway accelerating into the sub-target lead before it latches. It keys on the raw MODEL leads (`modelV2.leadsV3`, prob ≥ 0.15) because on the road the headway is spent on exactly the pre-latch frames where the lead's model prob is still ramping below the Schmitt enter band — not yet a control lead or a published radarState lead. Positive-only (never adds braking, so it cannot delay any decel); its RELEASE is rate-limited to `HandoffLimitMaxDeltaMps2` so a flickering model prob cannot inject a one-frame positive swing. Measured: the road's +0.49 (harness +0.85) inside-df cruise accel drops to +0.10.

Rollback sentinels: `HandoffLimitWindowS = 0` disables the windowed limiter entirely (pre-CD6 hard handoff); a large `HandoffInsideDfPositiveCapMps2` (e.g. 10, the spec maximum) disables the EDGE1 cap.

| Param Key | Default | Range | Description |
|---|---|---|---|
| `HandoffLimitWindowS` | 0.40 | 0.0–1.0 | Duration (s) the symmetric per-frame delta clamp is armed after a cruise↔lead source flip or a cruise-owned-cap collapse. 0 disables the windowed limiter (pre-CD6 hard handoff) |
| `HandoffLimitMaxDeltaMps2` | 0.30 | 0.0–2.0 | Max `|output_a_target − prev_a|` (m/s²) per frame while the window is active (comfortably under the oracle's 0.4 one-frame bound). Also the EDGE1 cap's release rate limit. Only meaningful when `HandoffLimitWindowS` > 0 |
| `HandoffInsideDfPositiveCapMps2` | 0.10 | 0.0–10.0 | Cap on positive `output_a_target` (m/s²) while cruise owns and a model lead is inside 0.9·df on a closing trend. Rollback: a large value (e.g. 10) disables the cap |

## CD7: Graded-Onset Comfort Anti-Jerk Envelope

The truly-FINAL limiter on `output_a_target` in `longitudinal_planner.py`, applied AFTER the CD6 handoff limiter (`_apply_comfort_jerk_envelope`).

**The defect (road 200-10 / 200-9 tap1 / 201-9).** Under a benign STEADY single-source lead follow — no source flips, benign kinematics, `aLeadK` ~0 — the MPC QP output can step-change `aTarget` hard on a single noisy vRel frame (road 200-10: `aTarget` −0.31 → −1.00 in 0.15 s ≈ 4.6 m/s³; 201-9: two 50 ms self-correcting −1.2 spikes; 200-9 tap1: a 0.86 m/s² sign reversal in 0.8 s). No earlier limiter bounds this because the handoff/relatch/flutter clamps only arm on a source flip or flutter, and there is none here — this is the audit's "no closed-loop test asserts any jerk/safety envelope". It is part of the felt THROTTLE_BLIP_JERK / VACILLATION.

**The envelope (asymmetric, down-leg only).** While NO hazard/urgency gate is active, the per-frame DOWNWARD (comfort-braking-onset) move of `output_a_target` is bounded to `ComfortJerkLimitMps3 · dt` (0.8 m/s³ · 0.05 s = 0.04 m/s²/frame) — the road's headline defect is a sudden unnecessary BRAKE (−0.31 → −1.00), and that felt brake blip is what the envelope kills. The UPWARD leg (brake-RELEASE and re-accel toward a followed lead) is always-safe and left FREE, so the legitimate managed release/re-accel moves are never blunted (this mirrors the CD5 relatch blend and flutter clamp, which also bound only the direction that can hurt). It only grades the ONSET of a downward step; no sustained comfort floor is lowered once `prev_a` catches up. Measured (repro oracle): the shipped un-enveloped peak downward comfort-brake jerk of ~4.3 m/s³ on a steady 30 m/s follow drops to ~0.8 m/s³ (bound 1.0), with the upward re-accel untouched.

**Scope gate (non-regressing).** The envelope engages ONLY in the steady-lead-follow regime the CD7 blip lives in: a `lead0`/`lead1` source with NO source flip this frame and NO other composed limiter or cap doing legitimate fast work — it defers whenever the source is cruise-owned (the cruise accel cap does legitimate one-frame accel suppression on a far slow lead), a source flip just occurred, or the CD6 handoff limiter / EDGE1 cap / CD5 relatch blend / flutter clamp is engaged. Those limiters make deliberate fast moves an unconditional per-frame bound would over-smooth, so the envelope only shaves the pure steady-follow brake blip they never touch.

**Hazard bypass (safety-critical).** Even inside its regime the envelope is FULLY BYPASSED whenever ANY hazard/urgency signal is active — the SAME `_relatch_urgency_bypass` predicate CD5/CD6 use (FCW / short-TTC / fast-close on the source-owned lead, or a requested hard decel at/below `ComfortJerkBypassDecelMps2`). So genuine braking passes the frame unmodified; real braking is NEVER rate-limited (verified: a real decelerating lead and a real cut-in brake with onset frame-identical to pre-CD7 HEAD, and stay frame-identical even at an absurdly tight limit).

Rollback sentinels: `ComfortJerkLimitMps3 = 0` disables the envelope; a large value (e.g. 50, the spec maximum → 2.5 m/s²/frame, above the whole real accel range) makes the per-frame bound unreachable and also disables it (pre-CD7 un-enveloped output). `ComfortJerkBypassDecelMps2 = 0` disables the CD7-specific requested-decel bypass floor (lead-object urgency signal only).

| Param Key | Default | Range | Description |
|---|---|---|---|
| `ComfortJerkLimitMps3` | 0.8 | 0.0–50.0 | Per-frame DOWNWARD (comfort-braking-onset) `output_a_target` step bound (m/s³ × dt) on the final planner output while a lead-owned steady follow has no hazard gate or other limiter active. The upward (release/re-accel) leg is free. 0 or a large value (e.g. 50) disables the envelope (pre-CD7 output) |
| `ComfortJerkBypassDecelMps2` | −1.5 | −5.0–0.0 | Requested-decel floor (m/s²) at/below which the envelope is bypassed regardless of the lead-object urgency tests (mirrors the flutter/relatch bypass). 0 disables this floor |

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

### Brake-onset floor (R11 phantom near-collision)

Measured floor for reaching -1.5 m/s^2 after a lead begins braking at -3 m/s^2 from 30 m/s (`test_repro_phantom_near_collision.py`, EV6 device-fidelity loop): **1.55 s**, decomposed as ~0.15 s radard EMA perception lag (shipped clean 1.55 s vs a perfect-perception oracle at 1.40 s using `perception_filter="direct"`) plus a 1.40 s floor owned by `longitudinalActuatorDelay`=0.5 s and comfort-shaped MPC convergence. The amplify path above already pulls the MPC-input aLeadK to the true -3.0 on the first braking frame, so there is no perception headroom left; the residual is irreducible without braking harder than comfort at a still-large real gap (~41 m). The phantom dropout adds **zero** delay vs a matched no-dropout run (1.55 s == 1.55 s), which is the R11 invariant the test now asserts (relative onset delta plus a 1.6 s absolute floor). The audit doc's aspirational 1.3 s bound was below this physical floor.

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
