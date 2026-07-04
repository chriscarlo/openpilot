# Road incidents 2026-07-04 — launch failure + near-collision (pinpointed for follow-up analysis)

Two reproducible-from-log failures captured on a live drive. Both are **evidenced from the
full rlog** (not live-monitor guesses). rlogs are on this Mac at
`realdata/00000205--63a5523547--<seg>/rlog.zst` (segments 4–15 pulled). Analysis scripts:
`scratchpad/deep.py`, `scratchpad/traces.py` (this session's scratchpad).

Deployed code at time of drive: branch `chauffeur-exp01` HEAD `973d3c7`.
Live-tune state during the drive (relevant, non-default):
- `ModelLeadFilterVRelTauS = 0.60` (raised from 0.40 mid-drive to damp steady-follow vRel-noise
  braking — **this directly trades against closing-lead response**, see Event B).
- `HandoffInsideDfPositiveCapMps2 = 10.0` (EDGE1 cap DISABLED — it was misfiring on ghost model
  leads and chopping accel to 0.1; disabling fixed a "bucking" complaint).
- All other params at the committed device snapshot `device_livetune_snapshot_20260702.txt`.

Signals per frame below: `v`=vEgo, `aE`=aEgo, `en`=selfdriveState.enabled, `lng`=carControl.longActive,
`ss`=standstill, `aTgt`=longitudinalPlan.aTarget, `cc`=carControl.actuators.accel (planner→controller),
`co`=carOutput.actuatorsOutput.accel (post-Hyundai-controller, the real CAN command),
`lc`=controlsState.longControlState, `lead`=(dRel, vRel, aLeadK, modelProb).

---

## Event A — FAILED / TOO-SLOW LAUNCH FROM STOP  (seg 13, t≈1.5–4.1 s)

**Symptom (driver):** "sat through an entire green light waiting for it," had to add pedal.

**What actually happened (trace):**
```
 1.51 v=0.00 lng=1 ss=1 aTgt=-0.00 cc=-2.00 co=-2.00 lc=stopping stop=1 lead=(4.3, +0.45, 0.85, 1.0)
 2.03 v=0.00 lng=1 ss=1 aTgt=+0.12 cc=-2.00 co=-2.00 lc=stopping stop=1 lead=(4.8, +1.29, 1.35, 1.0)
 2.28 v=0.00 lng=1 ss=1 aTgt=+0.21 cc=-2.00 co=-1.64 lc=stopping stop=1 lead=(5.3, +1.84, 1.53, 1.0)   <- lead already pulling away, still full stop-brake
 2.54 v=0.00 lng=1 ss=1 aTgt=+0.37 cc=+1.00 co=+1.00 lc=starting stop=0 lead=(5.9, +2.38, 1.62, 1.0)   <- releases ~1s late
 3.05 v=0.00 lng=1 ss=1 aTgt=+0.66 cc=+1.00 co=+1.00 lc=starting stop=0 lead=(7.6, +3.43, 1.89, 1.0)   <- still not moving, lead 7.6m gone
 3.31 v=0.29 lng=1        aTgt=+0.79 cc=+0.78 co=+0.78 lc=pid      lead=(8.7, +4.16, 1.96, 1.0)
 3.83 v=0.89 lng=1        aTgt=+0.95 cc=+0.95 co=+0.93 lc=pid      lead=(11.7, +5.08, 1.95, 1.0)   <- gentle +0.95 while lead departs at +5 m/s
 4.08 v=1.15 lng=0 gas=1  aTgt=+1.53 cc=+0.00 co=+0.00 lc=off      lead=(13.1, +5.32, 1.96, 1.0)   <- DRIVER PEDALS (openpilot far behind)
```

**Root cause — two compounding defects, both in the stop→launch path:**
1. **`longControlState` holds `stopping` (cc=-2.0 full stop-brake) ~1 s too long** after the lead
   starts moving. Lead vRel is already +0.45→+1.84 m/s (clearly departing) from t=1.5, but the
   state stays `stopping` until t=2.54. The stop-latch / `shouldStop` release is late.
2. **Launch accel is far too weak once it does start:** `starting`/`pid` gives +0.86–1.0 m/s²
   while the lead is departing at +4.7–5.3 m/s and already 10–13 m ahead. The car falls so far
   behind the driver overrides. (Note also a ~0.5 s dead zone: `starting` commands +1.0 at t=2.54
   but vEgo stays ~0 until t=3.3 — EV/actuator launch lag not compensated.)

**Where to look (code):**
- `selfdrive/controls/lib/longcontrol.py` — `LongCtrlState` stopping→starting transition, the
  `vEgoStopping` / `vEgoStarting` thresholds, `stopping_target_accel` / stopAccel hold, and how
  `shouldStop` gates the release.
- `selfdrive/controls/lib/longitudinal_planner.py` — the stop latch and its release ("Release stop
  latch when lead launches" lineage; `test_longitudinal_planner_stop_release.py`), and
  `get_low_speed_launch_follow_max_accel` (the launch-follow accel cap — likely too low for a fast
  departing lead).
- Hyundai standstill/resume handshake in `opendbc/sunnypilot/car/hyundai/longitudinal/controller.py`
  / carcontroller — whether the "starting" command is being throttled at standstill.

**Fix direction to evaluate:** (a) release `stopping` as soon as the lead's vRel/dRel shows a real
departure (not a fixed dwell); (b) raise the launch-follow accel so a departing lead is matched
(scale to lead vRel, capped for comfort); (c) compensate the ~0.5 s standstill actuation dead zone.

---

## Event B — NEAR-COLLISION, UNDER-/LATE-BRAKING ON A CLOSING LEAD  (seg 6, t≈30.5–32.6 s, ~40 mph)

**Symptom (driver):** "almost had a collision with the car in front," had to brake.

**What actually happened (trace):** openpilot WAS engaged and braking (`lng=1`), but the brake
**ramped up too slowly** and stayed behind the closure the whole way down:
```
30.01 v=18.4 lng=1 aTgt=+0.51 cc=+0.51 lc=pid lead=(32.7, -0.11, -0.38, 0.99)   <- lead starts closing/decel
30.52 v=18.6 lng=1 aTgt=-0.11 cc=-0.11 lc=pid lead=(31.5, -0.83, -0.36, 0.99)   <- barely braking
31.03 v=18.6 lng=1 aTgt=-0.25 cc=-0.25 lc=pid lead=(27.9, -1.41, -0.33, 0.99)   <- closing 1.4 m/s, only -0.25
31.55 v=18.3 lng=1 aTgt=-0.65 cc=-0.65 lc=pid lead=(22.1, -2.25, -0.45, 0.99)   <- closing 2.25, only -0.65, gap 22m
32.06 v=17.9 lng=1 aTgt=-1.10 cc=-1.10 lc=pid lead=(20.5, -2.55, -0.79, 0.99)
32.32 v=17.6 lng=1 aTgt=-1.74 cc=-1.74 lc=pid lead=(19.5, -3.24, -1.19, 0.99)   <- finally -1.74, gap already 19.5m
32.57 v=17.2 en=0 gas=0 brk=1 aTgt=-2.82 cc=+0.00 lc=off lead=(18.6, -4.39, -1.47, 1.0)  <- DRIVER BRAKES (op wanted -2.82 but too late)
```
THW bottomed near **1.10 s** at ~19 m closing 3.2 m/s. openpilot's own `aTarget` was still
*escalating* (it reached -2.82) — it wasn't refusing to brake, it was **always ~1.5–2 s behind the
closure rate.**

**Root cause — closing-lead brake response is too laggy/gentle early:** from the first clear closing
signal (t≈30.5, vRel negative + aLeadK negative) the brake built far too gradually (-0.11 → -0.25 →
-0.65 over ~1 s while the gap fell 31→22 m). By the time it commanded meaningful brake the driver
had already committed.

**Contributing tune knob (IMPORTANT):** `ModelLeadFilterVRelTauS` was raised to **0.60** during this
drive to damp steady-follow vRel-noise braking. That smoothing **delays detection of a real closing
lead** and sluggs exactly this ramp. It is a direct tension: 0.60 helped Event-B-adjacent noise but
hurt Event B itself. (I was told to leave it at 0.60 for your analysis — it is currently 0.60.)

**Where to look (code):**
- `selfdrive/controls/radard.py` `ModelLeadTracker` — `model_lead_filter_vrel_tau_s` (0.60) and the
  fast-close urgency path (`model_lead_fast_vrel_tau_s`, the fast-close gates). The urgency/fast path
  either wasn't engaging early or the blended tau was too slow here.
- `long_mpc.py` — `ObstacleCost` (2.0) reaction to closing; the lead-decel amplification
  (`LeadAccelCorrAmplifyGain`, CD3) — aLeadK went -0.15→-1.19 but the brake still lagged, so the
  amplification/consumption may be under-driving the MPC early; `get_lead_slowdown_accel_ceiling`
  ramp shape.
- Cross-check the Hyundai controller: `cc` and `co` track closely here (co slightly less), so the
  lag is upstream in the planner/perception, not the Hyundai EMA.

**Fix direction to evaluate:** make the closing-lead brake ramp lead the closure — a
complementary/asymmetric vRel filter (fast toward closing, smooth on noise) so noise damping does
NOT cost closing latency; and/or earlier obstacle/decel response when vRel and aLeadK agree on a
sustained close. This is the correct resolution of the noise-vs-closing tension, not a single tau.

---

## Quick index for the analysis session
| Event | rlog | window | one-line |
|---|---|---|---|
| A: slow/failed launch | `00000205--63a5523547--13` | t≈1.5–4.1 s | `stopping` held ~1 s after lead departs + launch accel too weak (+0.9 vs lead +5 m/s) → driver pedals |
| B: near-collision | `00000205--63a5523547--6` | t≈30.5–32.6 s | closing-lead brake ramp lags the closure (−0.25 at 1.4 m/s closing, 22 m); vRelTau=0.60 worsens it → driver brakes at THW 1.1 s |

Both are longitudinal **response-timing** defects (release timing / ramp timing), not perception
dropouts — the lead was tracked at prob≈1.0 throughout both. EDGE1 cap and vRel tau are the two live
knobs touched this drive; note their values above before attributing anything to the committed tune.
