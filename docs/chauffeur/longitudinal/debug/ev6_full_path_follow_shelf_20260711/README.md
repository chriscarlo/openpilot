# EV6 full-path follow-distance shelf — 2026-07-11

## Scope

2023 Kia EV6 HDA2 CAN-FD, openpilot longitudinal, no usable vehicle radar. The reported failure was a requested 1.6–1.8 s follow distance repeatedly settling near 2.0–2.2 s, weak catch-up, and unnecessary slowing on lead acquisition.

No tici connection was available. Evidence came from the checked-in July road rlogs and the local closed-loop harness.

## Root cause

There is no fixed 2.0 s minimum wired into Hyundai car control. The Hyundai controller consumes acceleration and jerk; it does not recalculate follow distance.

The repeated shelf was produced earlier in the chain:

1. `ModelLeadTrack._update_closing_governor()` armed from a noisy decreasing `dRel` window.
2. Even when the raw velocity stream showed only weak closure and raw `aLead` showed no braking, the governor could spend the full `ClosingGovernorPosTrustExcessMps=1.5` allowance.
3. That published an artificially negative `radarState.leadOne.vRel`.
4. The MPC stopping-equivalence geometry reacted to the claimed closure, and the post-MPC lead-slowdown ceiling reinforced it. The car therefore held or increased an oversized gap even though the physical lead was not braking hard enough to justify it.

The MPC safety distance is dynamic (ego speed, lead speed, time gap, braking assumptions, and danger constraints). It can legitimately increase transient spacing for a slower lead, but it is not a literal 2.0–2.2 s hard floor and is not directly applied after the planner by the Hyundai controller.

The local road corpus corroborates the signature. In multiple engaged frames near 1.96–2.25 s headway, the non-braking raw stream showed roughly 0.4–0.7 m/s closure while the position window implied roughly 3–20 m/s closure; published closure rose to roughly 0.7–1.0 m/s and the planner was already requesting decel. These are short noisy frames, not proof that every road event had the same cause, but they exercise the exact unsafe trust asymmetry reproduced by the closed loop.

## Fix

The position-derived excess is now available only when a second threat signal corroborates it:

- current or windowed raw lead deceleration exceeds the existing veto;
- windowed raw closure reaches the existing closing-governor discrepancy margin;
- or position-derived collision TTC is at most 6 s.

Otherwise the governor may still arm, but its clamp is limited to the windowed raw-velocity closure. No new parameter was added. MPC obstacle geometry, danger constraints, FCW, and full position authority for corroborated threats are unchanged.

## Full-path simulation contract

The acceptance path is:

`modelV2.leadsV3 -> real RadarD/ModelLeadTracker -> lead classification + MPC -> post-MPC planner limits -> LongControl -> Hyundai no-radar device controller at device cadence -> command delay -> vehicle plant`

`test_ev6_tici_soup_to_nuts_matrix.py` runs all 34 built-in synthetic scenarios through that topology with EV6 measured-noise injection. `test_ev6_tici_soup_to_nuts_follow.py` adds a physical oracle for far slower-lead acquisition and convergence to a displayed 1.70 s target.

In the 50 s seeded follow case, the legacy trust rule held a 1.990 s tail mean and finished at 1.967 s. The fix produces a 1.834 s tail mean and finishes at 1.784 s, with no FCW/crash count and a 46.41 m observed minimum gap. The 43.13 m value from an earlier experimental cost-reference branch was only that branch's observed scenario minimum, not a hard-coded distance floor; that cost change was rejected because it worsened transient follow recovery.

## Verification

Focused acceptance:

```sh
.venv/bin/python -m pytest -q -n 0 \
  selfdrive/test/longitudinal_harness/tests/test_ev6_tici_soup_to_nuts_matrix.py \
  selfdrive/test/longitudinal_harness/tests/test_ev6_tici_soup_to_nuts_follow.py \
  selfdrive/test/longitudinal_harness/tests/test_repro_closing_brake_lag.py \
  selfdrive/test/longitudinal_harness/tests/test_repro_follow_limit_cycle.py
```

Unit coverage for the trust split is in `selfdrive/controls/tests/test_radard_model_lead_filter.py`.

The eight scenario-validity failures found during the first full run were corrected without loosening their behavioral limits:

- preview handoffs now start lead A at 60 m, keeping it genuinely `lead0`-owned through real radard before lead B reveals;
- the sign-flip fixture injects the raw 5.5 m/s dip required for radard to publish the road-sized ~3.5 m/s rollover, and disables the separate EDGE1 cap only in that CD6-isolation twin;
- the relatch lead now accounts for ego's dropout acceleration, producing an actual published -0.49 m/s relatch at ~42 m (>80 s TTC) instead of the accidental -1.74 m/s / ~24 s approach;
- release-floor tests now measure the projection-floor candidate before the later M1 threat ceiling, which correctly has final authority, and apply the rollback veto only after filtered `aLeadK` actually crosses its configured threshold.

Final full harness result: **152 passed, 0 failed**.
