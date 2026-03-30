# VTSC Tuning Heuristics (Chauffeur)

These are practical heuristics for adjusting VTSC so it feels correct without becoming a jagged, overfit mess.

## Classify the Intervention

- **Driver brake intervention** (user presses brake while VTSC was supposed to slow):
  - Treat this as “VTSC failed to slow early enough”.
  - The speed curve may not be the root cause.
  - Common suspects: late perception commitment, occlusion gating/failopen behavior, sudden drops in lane-line confidence, or a late spike in predicted lateral acceleration.

- **Driver gas intervention** (user presses accelerator while VTSC is limiting):
  - Treat this as “VTSC is too conservative” (cap too low for the actual turn).
  - Curve mapping is a more likely candidate, but confirm first (was the model predicting a turn that wasn’t real?).

## Keep Curve Changes Smooth (No Bumps)

If you adjust the curvature-to-speed function (sigmoid/piecewise):
- Don’t “fix” one isolated speed point; apply a broad change centered on the problematic region.
- Keep the mapping monotonic and differentiable enough to avoid jerky cap changes.
- If it’s too slow at ~35 mph, it’s probably also too slow at ~40–42 mph:
  - Apply the largest correction near the center and taper it smoothly as you move away.

## Prefer Offline Evidence Before Changing Constants

Use the RCA workbook and traces to validate:
- Was `vtscVelMps` tightening late (cap collapse), or was it low early and just not enforced?
- Did `llProbMean` and/or `modelConf` collapse shortly before the intervention?
- Did `longitudinalPlan.aTarget` start braking early enough relative to the cap?
- Was VTSC actually the active visible cap at the time?
  If not, do not retune VTSC on that event. Route the RCA to the real
  longitudinal limiter first.

## Do Not Misclassify General Longitudinal Bugs As VTSC

- A surge, follow-gap pulse, or lead-handling complaint with no active VTSC cap
  is not VTSC evidence just because it happened near a curve.
- If the driver report is about a lead vehicle rather than the road geometry,
  prove VTSC was constraining before you touch map timing or curve-speed
  constants.
- Use VTSC gas-event triage labels literally:
  `not_constraining` means the event is not relax evidence and usually belongs
  outside VTSC tuning.

## Hot-Reload (Optional Future Improvement)

For fast iteration, make the curve params reloadable without restarting:
- Read curve parameters from `Params` on a timer (e.g., 1 Hz) in the VTSC controller.
- Keep defaults hardcoded; only override when a valid param is present.
