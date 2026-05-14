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
| `GapReclaimStrength` | 1.0 | 0.0–2.0 | How eagerly ACC closes extra gap on pullaway |
| `GapReclaimGapMinM` | 1.5 | 0.0–10.0 | Minimum extra gap before reclaim activates |
| `GapReclaimMaxAccel` | 0.36 | 0.0–0.75 | Cap on positive accel floor for gap closing |

## Lead Preview

| Param Key | Default | Range | Description |
|---|---|---|---|
| `LeadPreviewStrength` | 1.0 | 0.0–2.0 | How early a newly recognized slower lead shapes decel |
| `LeadPreviewGapMinM` | 1.5 | 0.0–10.0 | Min extra slack before preview activates |
| `LeadPreviewMaxBufferM` | 12.0 | 0.0–25.0 | Max closer-pull of previewed lead obstacle |
| `LeadAcquireWindowS` | 1.25 | 0.0–3.0 | Short stronger-preview window after a lead appears or jumps materially slower/closer |

## Cut-In Settle

| Param Key | Default | Range | Description |
|---|---|---|---|
| `CutInSettleDurationS` | 7.0 | 0.0–12.0 | Grace window length after cut-in detection |
| `CutInSettleMaxDecel` | 0.30 | 0.0–0.80 | Max braking magnitude during grace window |
| `CutInSettleMaxClosingSpeedMps` | 2.5 | 0.5–6.0 | Max ego-lead closing speed to qualify for grace |
| `CutInSettleAccelBiasMps2` | 0.10 | 0.0–0.30 | Positive accel offset to counteract EV regen during settle |

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

## Setting Params from SSH

```bash
# Show current values
/usr/local/venv/bin/python3 /data/openpilot/.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show

# Set a value
/usr/local/venv/bin/python3 /data/openpilot/.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py set drel-filter-tau-open 1.5

# Reset all to defaults
/usr/local/venv/bin/python3 /data/openpilot/.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py reset
```
