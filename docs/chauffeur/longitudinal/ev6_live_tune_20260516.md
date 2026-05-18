# EV6 Freeway Live Tune - 2026-05-16

## Context

- Vehicle: 2023 Kia EV6, Hyundai CAN-FD HDA2 / LKA-steering path.
- Device: tici at `/data/openpilot`, branch `chauffeur-exp01`, commit `0986c0f`.
- Drive notes: freeway lead acquisition and steady follow felt much better after source-stability damping. Remaining complaint was slight eagerness to brake before feathering accel, then slight sluggishness on first pull-away.
- Raw artifacts are intentionally untracked under `.cache/doc_artifacts/longitudinal/20260516_ev6_freeway_live_tune/`, including the live capture archive and follow-up synthetic sweep JSON files.

## Repo Tune After Follow-Up Sims

The on-road final value for `LeadSlowdownStrength` was `0.35`; follow-up synthetic sweeps moved the repo default to `0.25` to soften normal comfort braking while leaving the short-TTC danger path intact.

```text
preview_strength=1.500 preview_gap_min_m=1.000 preview_max_buffer_m=10.000 lead_acquire_window_s=1.500
reclaim_strength=0.550 reclaim_gap_min_m=3.000 reclaim_max_accel=0.120
keepup_strength=1.150 keepup_gap_min_m=0.100 keepup_max_accel=0.095
slowdown_strength=0.250 slowdown_max_decel=4.000
cutin_settle_duration_s=6.000 cutin_settle_max_decel=0.150 cutin_settle_max_closing_speed_mps=2.200 cutin_settle_accel_bias=0.120
vl_slow_tau=1.300
drel_tau_close=0.300 drel_tau_open=0.800 drel_open_slew_max_mps=1.800 drel_ig=30.000 drel_cg=12.000
cruise_reacquire_pos_jerk_limit=0.080 cruise_reacquire_jerk_window_s=3.000
lead_prob_enter=0.600 lead_prob_exit=0.250
lead_source_acquire_frames=1.000 lead_source_release_frames=20.000
phantom_lead_hold_s=0.800 phantom_lead_stable_frames=3.000
flutter_detect_transitions=2.000 flutter_detect_window_s=1.000 flutter_clamp_jerk_mps3=0.120 flutter_clamp_bypass_decel_mps2=1.500
model_lead_tau=2.800 model_lead_open_slew=1.200 model_lead_safe_ttc=4.000 model_lead_assoc_drel=12.000 model_lead_vrel_tau=0.400 model_lead_fast_vrel_tau=0.160
```

Additional MPC/filter params:

```text
Longitudinal.LiveTune.ObstacleCost=2.0
Longitudinal.LiveTune.AccelChangeCost=400.0
Longitudinal.LiveTune.AccelCost=1.0
Longitudinal.LiveTune.UseKalmanDRelFilter=True
Longitudinal.LiveTune.KalmanDRelQ=0.5
Longitudinal.LiveTune.KalmanDRelR=6.0
Longitudinal.LiveTune.KalmanDRelGainMax=0.25
Longitudinal.LiveTune.KalmanDRelDeadbandM=0.75
```

## Interpretation

- The key improvement came from reducing close-gap lead/source flaps: `LeadSourceReleaseFrames=20`, `PhantomLeadHoldS=0.8`, `LeadProbExit=0.25`, and `FlutterClampJerkMps3=0.12`.
- The "too close, all decel; too far, all accel" feel improved when `AccelChangeCost` rose to `400.0`; follow-up sims for the overly eager normal brake feel moved `LeadSlowdownStrength` from `0.35` to `0.25`.
- To get immediate but gentle roll-on when the gap starts widening, keep-up now starts very early (`LeadKeepUpGapMinM=0.10`) but stays tiny (`LeadKeepUpMaxAccel=0.095`).
- Harder low-TTC authority is preserved through the separate slowdown max decel and danger path; the normal comfort slowdown gain is what was softened.

## Verification Targets

- Lead-follow source should avoid close-gap `cruise -> lead -> cruise` chatter once a lead has been stable.
- Pull-away should show a small positive accel floor quickly after `vRel` turns positive, not a delayed large reclaim.
- Slower-lead acquisition should avoid positive accel deep into closing, while normal high-TTC acquisition should avoid `-1.0 m/s^2` snap-braking at wide gaps.
