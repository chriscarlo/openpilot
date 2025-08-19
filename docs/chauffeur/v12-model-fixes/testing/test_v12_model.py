#!/usr/bin/env python3
"""
Test script to verify v12 model functionality and calibration.
Run this on the TICI device after fixes are applied.
"""

import time
import json
from openpilot.common.params import Params
from cereal import messaging

def check_model_status():
    """Check current model configuration and status."""
    params = Params()

    print("=" * 60)
    print("V12 Model Test Script")
    print("=" * 60)

    # Check model runner type
    runner_cache = params.get("ModelRunnerTypeCache")
    print(f"\n1. Model Runner Type: {runner_cache}")
    if runner_cache == b"2":
        print("   ✓ Using stock modeld (correct for v12)")
    elif runner_cache == b"1":
        print("   ✗ Using tinygrad modeld (may have issues)")
    elif runner_cache == b"0":
        print("   ? Using SNPE modeld")
    else:
        print("   ? Unknown runner type")

    # Check calibration status
    calib_params = params.get("CalibrationParams")
    if calib_params:
        try:
            calib_data = json.loads(calib_params)
            print("\n2. Calibration Status:")
            print(f"   Valid: {calib_data.get('valid', False)}")
            print(f"   Calibrated: {calib_data.get('calibrated', False)}")
            if 'calib_perc' in calib_data:
                print(f"   Progress: {calib_data['calib_perc']}%")
        except:
            print("\n2. Calibration Status: Unable to parse")
    else:
        print("\n2. Calibration Status: No calibration data")

    # Check if modeld is publishing
    print("\n3. Checking modeld output (5 seconds)...")
    sm = messaging.SubMaster(['modelV2', 'cameraOdometry'])

    model_msgs = 0
    camera_msgs = 0
    start_time = time.time()

    while time.time() - start_time < 5:
        sm.update(100)  # 100ms timeout

        if sm.updated['modelV2']:
            model_msgs += 1
            if model_msgs == 1:
                # Check first message for validity
                msg = sm['modelV2']
                print(f"   First modelV2 message valid: {msg.valid}")

        if sm.updated['cameraOdometry']:
            camera_msgs += 1
            if camera_msgs == 1:
                # Check calibration output
                msg = sm['cameraOdometry']
                print(f"   First cameraOdometry message valid: {msg.valid}")
                if msg.valid:
                    euler = msg.wideFromDeviceEuler
                    print(f"   wideFromDeviceEuler: {euler}")

    print(f"\n   Received {model_msgs} modelV2 messages")
    print(f"   Received {camera_msgs} cameraOdometry messages")

    if model_msgs > 0 and camera_msgs > 0:
        print("   ✓ modeld is running and publishing")
    elif model_msgs > 0:
        print("   ⚠ modeld running but no camera odometry")
    else:
        print("   ✗ modeld not publishing")

    # Check for model files
    print("\n4. Checking model files...")
    import os
    model_paths = [
        "/data/models/supercombo.onnx",
        "/data/models/supercombo.thneed",
        "/data/openpilot/selfdrive/modeld/models/supercombo.onnx",
        "/data/openpilot/selfdrive/modeld/models/supercombo.thneed",
    ]

    found_model = False
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   Found: {path} ({size_mb:.1f} MB)")
            found_model = True

    if not found_model:
        print("   ✗ No model files found")

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)

if __name__ == "__main__":
    check_model_status()
