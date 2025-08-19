# Rednose Filter Build Error Diagnosis

## Problem Summary
The build fails with: `scons: *** No tool module 'rednose_filter' found`

## Root Cause
The git submodules are not initialized. The rednose_repo submodule (and others) are defined in `.gitmodules` but have not been checked out.

## Evidence
1. **Git submodule status shows all submodules are uninitialized:**
   ```
   -fd7bd0df50a95dca3f180705721aa1fa300aef0f msgq_repo
   -5127b93eecf2a39d2c65f3331dc2160e7477f645 opendbc_repo
   -ff4733f95854e49f692ce8ee431af4c85b64ee33 panda
   -7fddc8e6d49def83c952a78673179bdc62789214 rednose_repo
   -389815b8ca5302ce7c1504b7841d4eb61a8cd51b sunnypilot/neural_network_data
   -03cac2d30e111e0689c0429cb8c1fe6cb5a905af teleoprtc_repo
   -7737cbb2a0635fce95a9085fb1b53d5bea1093f8 tinygrad_repo
   ```
   The `-` prefix indicates uninitialized submodules.

2. **The rednose_repo directory doesn't exist:**
   - `/data/openpilot/rednose_repo` is missing
   - But `/data/openpilot/rednose` is a symlink pointing to `rednose_repo/rednose`

3. **SCons configuration expects rednose_filter tool:**
   - In SConstruct: `tools=["default", "cython", "compilation_db", "rednose_filter"]`
   - Toolpath includes: `"#rednose_repo/site_scons/site_tools"`
   - The build system expects to find the tool at: `/data/openpilot/rednose_repo/site_scons/site_tools/rednose_filter.py`

4. **Rednose is a Kalman filter library:**
   - Used in `selfdrive/locationd/models/car_kf.py` and `pose_kf.py`
   - Imports: `from rednose.helpers.kalmanfilter import KalmanFilter`

## Solution
The submodules need to be initialized. Based on `tools/op.sh`, the correct command is:
```bash
git submodule update --init --recursive
```

Or use the openpilot setup script:
```bash
tools/op.sh setup
```

## Important Notes
- This is NOT related to the driver monitoring changes
- The error was pre-existing before the cherry-picks
- All builds will fail until submodules are initialized
- The project has multiple submodules that provide essential functionality