Hyundai Live Tuning: chauffeur-dev2 → chubbs-merge Delta

This note captures which Hyundai live‑tuning params exist in chauffeur-dev2 but are missing in chubbs-merge, and for each, whether the hard‑coded value it adjusted/overrode still exists in chubbs-merge (with pointers). Use this as a quick reference for a future, fuller port.

Missing live tune params in chubbs-merge (present in chauffeur-dev2)

- Param: `LongTuningVEgoStarting`
  - Purpose: Overrides the starting velocity threshold used when leaving a stop.
  - chauffeur-dev2 target: `CarTuningConfig.v_ego_starting` (default 0.10)
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `get_longitudinal_tune` to set `CP.vEgoStarting`
  - chubbs-merge status: Param missing, hard‑coded default still exists
    - Field: `CarTuningConfig.v_ego_starting` (0.10)
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

- Param: `LongTuningLongitudinalActuatorDelay`
  - Purpose: Overrides longitudinal actuator delay used by controller and CarParams.
  - chauffeur-dev2 target: `CarTuningConfig.longitudinal_actuator_delay` (default 0.45)
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `get_longitudinal_tune` → `CP.longitudinalActuatorDelay`
  - chubbs-merge status: Param missing, hard‑coded default still exists
    - Field: `CarTuningConfig.longitudinal_actuator_delay` (0.45)
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

- Param: `LongTuningLookaheadJerkBp`
  - Purpose: Overrides speed breakpoints for predictive jerk lookahead window.
  - chauffeur-dev2 target: `CarTuningConfig.lookahead_jerk_bp` (default [5., 20.]; many platforms override to [2., 5., 20.])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `LongitudinalController._calculate_lookahead_jerk`
  - chubbs-merge status: Param missing, hard‑coded defaults still exist
    - Field: `CarTuningConfig.lookahead_jerk_bp` (base [5., 20.]) and overrides in `TUNING_CONFIGS` (e.g., [2., 5., 20.])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

- Param: `LongTuningLookaheadJerkUpperV`
  - Purpose: Overrides lookahead window (s) for upper jerk estimation vs speed.
  - chauffeur-dev2 target: `CarTuningConfig.lookahead_jerk_upper_v` (default [0.25, 0.5]; many platforms use [0.25, 0.5, 1.0])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `LongitudinalController._calculate_lookahead_jerk`
  - chubbs-merge status: Param missing, hard‑coded defaults still exist
    - Field: `CarTuningConfig.lookahead_jerk_upper_v` (+ per‑platform overrides in `TUNING_CONFIGS`)
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

- Param: `LongTuningLookaheadJerkLowerV`
  - Purpose: Overrides lookahead window (s) for lower jerk estimation vs speed.
  - chauffeur-dev2 target: `CarTuningConfig.lookahead_jerk_lower_v` (default [0.15, 0.3]; many platforms use [0.05, 0.10, 0.3])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `LongitudinalController._calculate_lookahead_jerk`
  - chubbs-merge status: Param missing, hard‑coded defaults still exist
    - Field: `CarTuningConfig.lookahead_jerk_lower_v` (+ per‑platform overrides in `TUNING_CONFIGS`)
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

- Param: `LongTuningUpperJerkV`
  - Purpose: Overrides upper jerk limit (m/s³) vs speed mapping used during acceleration.
  - chauffeur-dev2 target: `CarTuningConfig.upper_jerk_v` (default [2.0, 2.0, 1.2])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `LongitudinalController._calculate_speed_based_jerk_limits`
  - chubbs-merge status: Param missing, hard‑coded defaults still exist
    - Field: `CarTuningConfig.upper_jerk_v` (chubbs default [3.0, 3.0, 1.5])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

- Param: `LongTuningLowerJerkV`
  - Purpose: Overrides lower jerk limit (m/s³) vs speed mapping used during braking.
  - chauffeur-dev2 target: `CarTuningConfig.lower_jerk_v` (default [3.0, 3.0, 2.5])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py
    - Used by: `LongitudinalController._calculate_speed_based_jerk_limits`
  - chubbs-merge status: Param missing, hard‑coded defaults still exist
    - Field: `CarTuningConfig.lower_jerk_v` (chubbs default [5.0, 8.0, 5.0])
    - File: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py

What’s already present in chubbs-merge live tuning UI

- Params supported today: `LongTuningAccelMin`, `LongTuningAccelMax`, `LongTuningVEgoStopping`, `LongTuningStoppingDecelRate`, `LongTuningMinUpperJerk`, `LongTuningMinLowerJerk`, `LongTuningJerkLimits`.
  - UI: selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/hyundai_live_tuning.cc
  - Backend mapping: opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/helpers.py → `create_config_from_params`

Notes for future full port

- List‑type params (lookahead arrays, jerk‑vs‑speed arrays) were edited in chauffeur-dev2 via a list control UI (`createListControl`) that’s not present in chubbs-merge. Re‑enabling those would require:
  - Adding param keys to `common/params_keys.h`.
  - Restoring list controls and formatting/parsing logic in `hyundai_live_tuning.cc`.
  - Extending `create_config_from_params` to parse lists (re‑adding `get_float_list`‑style parsing).
- Controller usage to keep in mind:
  - `upper_jerk_v`/`lower_jerk_v` → `_calculate_speed_based_jerk_limits`
  - `lookahead_jerk_*` → `_calculate_lookahead_jerk`
  - `longitudinal_actuator_delay`/`v_ego_starting` → `get_longitudinal_tune` writes into `CP`
- Aggressiveness deltas (context only): chubbs-merge defaults for jerk maps are generally more aggressive than chauffeur-dev2 (e.g., higher upper/lower jerk values, wider lower jerk cap), while we just aligned `jerk_limits` default to 4.0 (from 3.65).

