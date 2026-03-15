# Hyundai CAN FD and Kia EV6

## EV6 Is a Generic CAN FD Platform

- `opendbc/car/hyundai/values.py`
  declares `CAR.KIA_EV6` as a `HyundaiCanFDPlatformConfig` with the `EV` flag.
- EV6 does not get a bespoke longitudinal implementation. Most longitudinal
  behavior comes from generic Hyundai CAN FD logic plus sunnypilot overlays.
- Runtime flags such as `CANFD_LKA_STEERING`, `CANFD_CAMERA_SCC`, and
  `CANFD_ALT_BUTTONS` are inferred later from fingerprint and firmware data.

## Topology Matters More Than Badge Name

- CAN FD bus mapping changes when the car is classified as LKA-steering:
  default is `ACAN=1`, `ECAN=0`, `CAM=2`.
  LKA-steering swaps that to `ACAN=0`, `ECAN=1`, `CAM=2`.
- Hyundai CAN FD longitudinal bugs often reduce to “wrong topology”:
  wrong reads, wrong write bus, or the wrong ECU disable target.
- EV6 replay coverage already includes both LKA-steering and LFA-steering
  routes, so treat EV6 as at least two runtime topologies.

## Longitudinal Ownership Gate

- `opendbc/car/hyundai/interface.py`
  decides whether openpilot longitudinal is even allowed.
- For CAN FD Hyundai:
  - `openpilotLongitudinalControl = alpha_long and alphaLongitudinalAvailable`
  - `pcmCruise` is the inverse
  - safety adds `HyundaiSafetyFlags.LONG` when openpilot owns longitudinal
- LKA-steering cars have an extra gate:
  if firmware probing does not report `Ecu.adas`, alpha longitudinal is blocked.
- On EV6 HDA II-style setups, missing ADAS ECU detection is a first-class
  reason for “why is stock ACC still in charge?”

## Stock ACC vs Openpilot Longitudinal

- In `opendbc/car/hyundai/carstate.py`, CAN FD cruise state splits like this:
  - openpilot longitudinal:
    `cruiseState.enabled` comes from `TCS.ACC_REQ`
  - stock ACC:
    `SCC_CONTROL` drives enabled/speed/standstill, from PT or CAM depending on
    `CANFD_CAMERA_SCC`
- EV-only `MANUAL_SPEED_LIMIT_ASSIST` can mark cruise as non-adaptive and can
  look like a longitudinal bug if you do not check it.

## ECU Disable and Keepalive Ownership

- `opendbc/car/hyundai/interface.py`
  disables either:
  - radar ECU `0x7d0`, or
  - ADAS ECU `0x730` for CAN FD LKA-steering cars
- `opendbc/car/hyundai/carcontroller.py`
  keeps that ECU disabled with periodic tester-present traffic.
- If stock longitudinal still appears to be fighting openpilot, check this
  ownership handoff before touching tuning.

## Hyundai Command Path

- `opendbc/car/hyundai/carcontroller.py`
  calls the sunnypilot Hyundai `LongitudinalController` every other frame.
- `opendbc/car/hyundai/hyundaicanfd.py`
  sends `SCC_CONTROL` with:
  `aReqValue`, `aReqRaw`, `StopReq`, `JerkLowerLimit`, and `JerkUpperLimit`.
- On CAN FD, those values come from the sunnypilot tuning state, not directly
  from `LongControl`.

## Sunnypilot Hyundai Tuning Overlay

- Live param plumbing:
  `sunnypilot/selfdrive/controls/lib/param_store.py`
  publishes `HyundaiLongitudinalTuning` and `LongTuning*` params to controlsd.
- Interface initialization:
  `opendbc/sunnypilot/car/interfaces.py`
  converts `HyundaiLongitudinalTuning` into `HyundaiFlagsSP` and then calls
  `get_longitudinal_tuning_sp(...)`.
- Init-time tuning hook:
  `opendbc/sunnypilot/car/hyundai/longitudinal/helpers.py`
  updates `CarParams` values like `vEgoStopping`, `stoppingDecelRate`,
  `longitudinalActuatorDelay`, and sets `startingState = False`.
- Runtime shaping hook:
  `opendbc/sunnypilot/car/hyundai/longitudinal/controller.py`
  calculates jerk-limited `actual_accel` and comfort-band values.

## Important EV6 Subtlety: CAN FD Tune Wins Before EV Tune

- `get_car_config(...)` picks:
  car-specific config first, then `CANFD`, then `EV`, then `HYBRID`, then
  default.
- EV6 has no dedicated entry in `CAR_SPECIFIC_CONFIGS`.
- Result:
  EV6 uses the generic `TUNING_CONFIGS["CANFD"]` baseline, not the generic
  `EV` baseline, unless you add a car-specific override.

## Another Subtlety: `radarUnavailable` Can Disable the Fancy Hyundai Tuner

- The Hyundai `LongitudinalController.calculate_accel(...)` fast-exits when
  `CP.radarUnavailable` is true.
- In that case, `desired_accel` and `actual_accel` fall back to the raw accel
  command and comfort-band shaping is bypassed.
- That means some EV6 topologies will still honor the init-time `CarParams`
  tuning changes while bypassing most of the runtime jerk-shaping logic.

## EV6-Specific Anomaly Checklist

- Wrong LKA-vs-LFA classification.
- Missing `Ecu.adas` on an LKA-steering topology.
- Wrong ECU disable target or wrong CAN FD bus.
- `MANUAL_SPEED_LIMIT_ASSIST` active and being mistaken for ACC weirdness.
- `blockPcmEnable`, button history, or MADS main-cruise state blocking expected
  engagement behavior.
- `CANFD_ALT_BUTTONS` resume/cancel edge cases.
- Assuming EV tuning is active when the code is actually using the generic CAN
  FD tuning baseline.

## Known Documentation Drift

- Older EV6 dashboard-speed-limit docs and tests still refer to a stale
  dashboard flag that is intentionally not present in the current Hyundai flag
  enum in this fork.
- Current live behavior is simpler:
  CAN FD `FR_CMR_02_100ms` on ECAN is parsed into `CarStateSP.speedLimit`.
- Prefer current code over the older EV6 speed-limit docs if they disagree.
