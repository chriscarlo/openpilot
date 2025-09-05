# Sync Notes — Fix A constants vs prior tranche

This folder reflects local Fix A changes that may not yet be visible on the remote branch. Key deltas from prior docs/code cited by GPT‑5 Pro:

- Status hysteresis (compat): `CONFIDENCE_ENTER_PARTIAL = 0.65`, `CONFIDENCE_EXIT_TO_FULL = 0.70` (previously 0.75/0.85).
- LKG ramp: `_lkg_ramp_s = 0.9` (previously 1.5).
- Tail growth defaults: `gamma_per_m ≈ 2.5e‑4`, `envelope_horizon_s ≈ 1.0`.
- Enough‑vision gating and vision‑floor TTL were added in central arbitration.

Once pushed, GPT‑5 Pro should read the updated code/constants from `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` and `vision_turn_params.py` on `chubbs-merge`.
