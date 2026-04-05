---
name: object-hazard-live-monitor
description: >
  Monitor live bring-up of the experimental `objectd` / `objectHazardStateSP`
  pipeline on a tici, inspect planner-side `objectHazardControl` and
  `longitudinalPlan.shouldStop`, and classify CPU, GPU, memory, thermal, or
  process collisions so the next mitigation targets the smallest affected
  surface. Use when validating the auxiliary object-hazard path on device,
  checking whether the new daemon is publishing or planner-wired correctly, or
  deciding whether to lower cadence, lower resolution, change accelerator
  backend, trim logging, or move process placement.
---

# Object Hazard Live Monitor

## Overview

Use this skill when the experimental object-hazard stack is running on a tici and you need live evidence instead of static guesses.
Prefer the bundled harness over ad hoc `top`, `ps`, or one-off `ssh` commands because it correlates `deviceState`, `procLog`, `objectHazardStateSP`, planner outputs, and current process placement into one capture and ends with lower-impact mitigation guidance.

## Quick Loop

1. Confirm the device is actually running the branch you care about.
- The default remote repo is `/data/openpilot`.
- If the deployed repo does not contain `objectHazardStateSP`, the harness will tell you that the service is missing instead of pretending the pipeline is idle.

2. Run the harness from this checkout.

```bash
python3 .codex/skills/object-hazard-live-monitor/scripts/object_hazard_live_monitor.py \
  --ssh-profile commaHome \
  --duration 45
```

3. If you need raw data for a follow-up session, save the JSON payload.

```bash
python3 .codex/skills/object-hazard-live-monitor/scripts/object_hazard_live_monitor.py \
  --ssh-profile commaHome \
  --duration 45 \
  --save-json .cache/object-hazard-monitor.json
```

4. Route the next change by the narrowest failing surface.
- If `objectHazardStateSP` is absent, fix deployment or manager gating before tuning planner behavior.
- If `objectHazardStateSP` is active but `longitudinalPlanSP.objectHazardControl` never mirrors it, fix planner wiring before changing detector thresholds.
- If planner state is correct but device pressure is high, change cadence, resolution, backend choice, or debug verbosity before touching the control logic.

## What The Harness Checks

- `deviceState` for per-core CPU, GPU usage, memory usage, temperatures, thermal status, and whether the device was actually onroad during the capture.
- `procLog` for per-process CPU deltas and RSS so you can see whether `objectd`, `modeld`, `camerad`, `plannerd`, `ui`, or another process is consuming the budget.
- `ps` snapshots for current processor placement so you can spot repeated `objectd` collisions with `modeld`, `camerad`, or `plannerd` on the same core.
- `objectHazardStateSP` for backend, readiness, hazard activation, stop requests, and debug detections when the experimental branch is deployed.
- `longitudinalPlanSP.objectHazardControl` plus `longitudinalPlan.shouldStop` so you can tell whether hazard state is actually reaching the planner and affecting the longitudinal output.
- filtered `logMessage` entries so runtime failures such as missing assets, SNPE init errors, or repeated `objectd` exceptions appear in the same report.

## Decision Routing

- Treat missing services or disabled params as wiring problems, not performance problems.
- Treat planner mismatches as contract problems, not detector-threshold problems.
- If one CPU core is hot and `objectd` shares that core with `modeld`, `camerad`, or `plannerd`, lower `objectd` cadence or input size before changing planner code. Only move process placement after the cheaper reductions are exhausted.
- If GPU pressure is high while the backend is `snpe_gpu`, lower auxiliary cadence or resolution first. If pressure stays high, test a DSP backend. Do not silently add CPU inference as a fallback.
- If memory climbs, trim debug detections, verbose logging, and frame retention before changing model logic.
- If thermal status reaches `yellow` or above, shorten the run and lower auxiliary load before interpreting behavior changes as semantic bugs.
- If the capture is resource-stable but the stop behavior is wrong, work on hazard semantics and planner thresholds, not runtime budgets.

## References

- Read `references/signals-and-routing.md` when you need the exact fields, budget anchors, or the preferred mitigation order by failure mode.
- Read `scripts/remote_probe.py` if the harness needs a new signal from the device side; keep the host wrapper small and keep Cap'n Proto field handling on the device side.

## Skill Maintenance

After any real tici monitoring session:
- replace stale guidance instead of stacking another warning on top of it
- keep the mitigation order aligned with the current `objectd` implementation and the current planner contract
- update the reference thresholds only when you can point to a code-backed field or a repeated device observation
- keep `SKILL.md` focused on workflow and routing; move field inventories and threshold tables into `references/`
- rerun `python3 /home/chris/.codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/object-hazard-live-monitor` after edits

Treat this skill as a Kaizen loop:
- every live capture should either sharpen the signal list, remove an unhelpful step, or improve the mitigation ordering
- prefer rewriting or deleting vague advice over appending more prose
- add a new heuristic only when it changes the next engineering decision in a repeatable way
