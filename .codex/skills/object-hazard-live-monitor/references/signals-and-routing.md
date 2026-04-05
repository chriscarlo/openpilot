# Signals And Routing

## Live Signals The Harness Uses

### Base device health

- `deviceState.memoryUsagePercent` in [cereal/log.capnp](/projects/chauffeur/data/openpilot/cereal/log.capnp#L482)
- `deviceState.gpuUsagePercent` in [cereal/log.capnp](/projects/chauffeur/data/openpilot/cereal/log.capnp#L483)
- `deviceState.cpuUsagePercent` in [cereal/log.capnp](/projects/chauffeur/data/openpilot/cereal/log.capnp#L484)
- `deviceState.thermalStatus` in [cereal/log.capnp](/projects/chauffeur/data/openpilot/cereal/log.capnp#L504)
- `hardwared` publishes those in [hardwared.py](/projects/chauffeur/data/openpilot/system/hardware/hardwared.py#L266)

### Process-level load

- `procLog` is the durable source for per-process CPU deltas and RSS.
- The delta calculation pattern is already demonstrated in [live_cpu_and_temp.py](/projects/chauffeur/data/openpilot/selfdrive/debug/live_cpu_and_temp.py#L46).
- `ps -eo pid,psr,pcpu,rss,args` adds current processor placement so the harness can detect `objectd` collisions with `modeld`, `camerad`, or `plannerd`.

### Experimental object-hazard path

These fields exist only in the experimental worktree or a device that has deployed it:

- `ObjectHazardStateSP` in [/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/cereal/custom.capnp](/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/cereal/custom.capnp#L616)
- `LongitudinalPlanSP.objectHazardControl` in [/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/cereal/custom.capnp](/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/cereal/custom.capnp#L254)
- `objectd` publish loop in [/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/sunnypilot/objectd/objectd.py](/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/sunnypilot/objectd/objectd.py#L44)
- planner subscription in [/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/selfdrive/controls/plannerd.py](/projects/chauffeur/data/openpilot/.worktrees/exp/tici-aux-model-20260404/selfdrive/controls/plannerd.py#L33)

If the deployed device repo does not have those fields, the harness should report the service as unavailable instead of inferring that `objectd` is healthy but idle.

## Budget Anchors

Use these as directional anchors, not as hard pass-fail laws for every parked capture:

- The stock onroad CPU budget table lives in [test_onroad.py](/projects/chauffeur/data/openpilot/selfdrive/test/test_onroad.py#L31).
- `MAX_TOTAL_CPU = 280` across eight cores in [test_onroad.py](/projects/chauffeur/data/openpilot/selfdrive/test/test_onroad.py#L31).
- Baseline expectations for `camerad`, `modeld`, `plannerd`, `ui`, and other long-lived processes are listed in `PROCS` in [test_onroad.py](/projects/chauffeur/data/openpilot/selfdrive/test/test_onroad.py#L32).

The harness should use those values to answer "is this obviously out of family?" not "is this exactly equal to the lab budget table?"

## Preferred Mitigation Order

### Wiring and bring-up failures

1. If `ObjectHazardEnabled` is false or `objectHazardStateSP` is missing, fix the gate or deploy the correct branch.
2. If `objectd` exists but `modelReady` stays false, fix assets or accelerator runtime before planner work.
3. If `objectHazardStateSP` is active but `longitudinalPlanSP.objectHazardControl` never reflects it, fix planner wiring before tuning thresholds.
4. If `objectHazardControl.stopRequired` is true but `longitudinalPlan.shouldStop` never follows, fix the planner merge path before changing detection behavior.

### CPU pressure

1. Lower `objectd` cadence.
2. Lower detector input size or preprocessing cost.
3. Reduce debug output volume and per-frame book-keeping.
4. Move `objectd` off the hot core only after the cheaper reductions are exhausted.
5. Do not add CPU fallback inference just to keep the service alive.

### GPU pressure

1. Lower auxiliary cadence.
2. Lower detector input size.
3. Trim debug work that runs on the same cadence.
4. Test DSP backend if the GPU remains tight.
5. Keep `modeld` as the priority consumer of the GPU path.

### Memory pressure

1. Trim detection debug lists and verbose logging.
2. Remove unnecessary frame buffering or cached tensors.
3. Verify RSS growth per process before changing model architecture.

### Thermal pressure

1. Shorten the run and cool the device.
2. Lower auxiliary cadence and input size.
3. Re-run the same scenario before interpreting behavioral changes as semantic regressions.

## When To Leave The Runtime Alone

If CPU, GPU, memory, and thermal signals all stay within a comfortable range but the stop behavior is wrong:

- work on path association, hysteresis, or planner thresholds
- do not burn time moving processes or switching backends
- keep the monitoring harness focused on proving that the next change targets semantics, not load
