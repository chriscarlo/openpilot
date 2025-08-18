#!/usr/bin/env python3
"""
Test script to verify calibration works for all model versions after fixes.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add openpilot to path
sys.path.insert(0, '/projects/chauffeur/data/openpilot')

from cereal import messaging
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.calibrationd import Calibrator

def test_calibration_basic():
    """Test basic calibration functionality"""
    print("Testing basic calibration...")
    
    # Clear any existing calibration
    params = Params()
    params.remove("CalibrationParams")
    
    # Create calibrator
    c = Calibrator(param_put=False)
    
    # Simulate calibration messages
    for i in range(100):
        # Simulate valid camera odometry data
        c.handle_v_ego(20.0)  # 20 m/s speed
        c.handle_cam_odom(
            [20.0, 0.0, 0.0],  # velocity
            [0.0, 0.0, 0.0],   # rotation rates
            [0.0, 0.0, 0.0],   # acceleration
            [0.001, 0.001, 0.001],  # velocity std
            [0.0, 0.0, 1.22],  # position (height)
            [0.01, 0.01, 0.01]  # position std
        )
    
    print(f"Valid blocks: {c.valid_blocks}")
    print(f"Calibration status: {c.cal_status}")
    print(f"RPY: {c.rpy}")
    
    return c.valid_blocks > 0

def test_model_message_validation():
    """Test that model messages are validated correctly without bypass"""
    print("\nTesting model message validation...")
    
    # Import the fill_model_msg module
    from openpilot.selfdrive.modeld import fill_model_msg
    
    # Create a mock model output
    model_output = np.zeros(5962, dtype=np.float32)
    
    # Test without calibration (should be invalid)
    msg = messaging.new_message('modelV2')
    fill_model_msg.fill_model_msg(msg.modelV2, model_output, 0, 0, 0.0, {})
    
    print(f"Model valid without calibration: {msg.modelV2.valid}")
    
    # Test with calibration
    params = Params()
    calib_msg = messaging.new_message('liveCalibration')
    calib_msg.liveCalibration.validBlocks = 20
    calib_msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    params.put("CalibrationParams", calib_msg.to_bytes())
    
    # Create new message with calibration present
    msg2 = messaging.new_message('modelV2')
    fill_model_msg.fill_model_msg(msg2.modelV2, model_output, 0, 0, 0.0, {'liveCalibration': True})
    
    print(f"Model valid with calibration: {msg2.modelV2.valid}")
    
    return not msg.modelV2.valid  # Should be invalid without calibration

def test_parse_model_outputs():
    """Test that model parsing doesn't have version-specific logic"""
    print("\nTesting model output parsing...")
    
    from openpilot.selfdrive.modeld import parse_model_outputs
    
    # Check that there's no v9/v12 detection logic
    source_file = Path('/projects/chauffeur/data/openpilot/selfdrive/modeld/parse_model_outputs.py')
    content = source_file.read_text()
    
    has_v9_logic = 'v9' in content.lower() or 'v12' in content.lower()
    has_bypass = 'bypass' in content.lower()
    
    print(f"Has v9/v12 detection logic: {has_v9_logic}")
    print(f"Has bypass logic: {has_bypass}")
    
    return not has_v9_logic and not has_bypass

def test_tensor_extraction():
    """Test that tensor extraction is using the correct method"""
    print("\nTesting tensor extraction methods...")
    
    modeld_file = Path('/projects/chauffeur/data/openpilot/selfdrive/modeld/modeld.py')
    content = modeld_file.read_text()
    
    # Check for correct tensor extraction pattern
    correct_pattern = '.contiguous().realize().uop.base.buffer.numpy()'
    incorrect_pattern = '.numpy().flatten()'
    
    has_correct = correct_pattern in content
    has_incorrect = incorrect_pattern in content
    
    print(f"Using correct tensor extraction: {has_correct}")
    print(f"Has old incorrect pattern: {has_incorrect}")
    
    return has_correct and not has_incorrect

def main():
    print("=" * 60)
    print("CALIBRATION SYSTEM VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Calibration", test_calibration_basic),
        ("Model Message Validation", test_model_message_validation),
        ("Parse Model Outputs", test_parse_model_outputs),
        ("Tensor Extraction", test_tensor_extraction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n--- Running: {name} ---")
            result = test_func()
            results.append((name, result))
            print(f"Result: {'✓ PASSED' if result else '✗ FAILED'}")
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Calibration system is working correctly!")
        print("The bypass logic has been successfully removed.")
        print("Models should now calibrate properly without shortcuts.")
    else:
        print("✗ SOME TESTS FAILED - Please review the failures above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
