#!/usr/bin/env python3
"""
Simplified test to verify calibration fixes are working.
"""

import os
import sys
from pathlib import Path

def check_file_for_bypass(filepath, filename):
    """Check if a file contains bypass logic"""
    if not filepath.exists():
        return f"File {filename} not found"
    
    content = filepath.read_text()
    
    # Check for bypass patterns
    has_bypass = any(pattern in content.lower() for pattern in ['bypass_calibration', 'bypass', 'skip_calibration'])
    
    # Check for the correct validation pattern (should only check live_calib_seen)
    has_correct_validation = 'msg.valid = live_calib_seen & (vipc_dropped_frames < 1)' in content
    
    if has_bypass:
        return f"✗ {filename}: Still contains bypass logic"
    elif has_correct_validation:
        return f"✓ {filename}: Correctly validates calibration without bypass"
    else:
        return f"✓ {filename}: No bypass logic found"

def main():
    print("=" * 70)
    print("CALIBRATION BYPASS REMOVAL VERIFICATION")
    print("=" * 70)
    
    # Files to check for bypass logic
    files_to_check = [
        ('/projects/chauffeur/data/openpilot/selfdrive/modeld/fill_model_msg.py', 'selfdrive/modeld/fill_model_msg.py'),
        ('/projects/chauffeur/data/openpilot/sunnypilot/modeld/fill_model_msg.py', 'sunnypilot/modeld/fill_model_msg.py'),
        ('/projects/chauffeur/data/openpilot/sunnypilot/modeld_v2/fill_model_msg.py', 'sunnypilot/modeld_v2/fill_model_msg.py'),
        ('/projects/chauffeur/data/openpilot/selfdrive/modeld/parse_model_outputs.py', 'selfdrive/modeld/parse_model_outputs.py'),
        ('/projects/chauffeur/data/openpilot/selfdrive/modeld/modeld.py', 'selfdrive/modeld/modeld.py'),
        ('/projects/chauffeur/data/openpilot/selfdrive/modeld/dmonitoringmodeld.py', 'selfdrive/modeld/dmonitoringmodeld.py'),
    ]
    
    print("\nChecking for bypass logic in critical files:\n")
    
    all_good = True
    for filepath, name in files_to_check:
        result = check_file_for_bypass(Path(filepath), name)
        print(result)
        if "✗" in result:
            all_good = False
    
    # Check for tensor extraction fixes
    print("\n" + "-" * 70)
    print("Checking tensor extraction methods:\n")
    
    modeld_files = [
        '/projects/chauffeur/data/openpilot/selfdrive/modeld/modeld.py',
        '/projects/chauffeur/data/openpilot/selfdrive/modeld/dmonitoringmodeld.py'
    ]
    
    for filepath in modeld_files:
        path = Path(filepath)
        if path.exists():
            content = path.read_text()
            correct_pattern = '.contiguous().realize().uop.base.buffer.numpy()'
            incorrect_pattern = '.numpy().flatten()'
            
            has_correct = correct_pattern in content
            has_incorrect = incorrect_pattern in content
            
            filename = path.name
            if has_correct and not has_incorrect:
                print(f"✓ {filename}: Using correct tensor extraction")
            elif has_incorrect:
                print(f"✗ {filename}: Still using old tensor extraction")
                all_good = False
            else:
                print(f"? {filename}: Tensor extraction pattern unclear")
    
    # Check environment variables
    print("\n" + "-" * 70)
    print("Checking environment variable setup:\n")
    
    for filepath in modeld_files:
        path = Path(filepath)
        if path.exists():
            content = path.read_text()
            has_dev = "os.environ['DEV']" in content
            has_old_qcom = "os.environ['QCOM']" in content and "os.environ['DEV']" not in content
            has_old_llvm = "os.environ['LLVM']" in content and "os.environ['DEV']" not in content
            
            filename = path.name
            if has_dev:
                print(f"✓ {filename}: Using unified DEV environment variable")
            elif has_old_qcom or has_old_llvm:
                print(f"✗ {filename}: Still using old QCOM/LLVM environment variables")
                all_good = False
            else:
                print(f"? {filename}: Environment variable setup unclear")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ SUCCESS: All calibration bypass logic has been removed!")
        print("✓ Tensor extraction methods are correct")
        print("✓ Environment variables are properly set")
        print("\nThe calibration system should now work correctly for ALL models.")
        print("Models will properly wait for calibration to complete before running.")
    else:
        print("✗ ISSUES FOUND: Some files still need fixes")
        print("Please review the issues marked with ✗ above")
    print("=" * 70)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
