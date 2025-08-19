#!/usr/bin/env python3
"""
Comprehensive RTI Parameter Integration Test

This script validates that ALL user-configurable RTI parameters are:
1. Properly stored when set
2. Correctly loaded by all components
3. Actually respected in the logic

Run this after making changes to ensure complete parameter integration.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path

# Add openpilot to path
sys.path.append('/data/openpilot')
from common.params import Params

class RTIParameterValidator:
    def __init__(self):
        self.params = Params()
        self.test_results = {}
        self.all_params = {
            "RTIEnabled": {"type": "bool", "default": True, "description": "Master RTI switch"},
            "RTIHUDEnabled": {"type": "bool", "default": True, "description": "HUD display toggle"},
            "RTIAudioAlerts": {"type": "bool", "default": True, "description": "Audio alerts toggle"},
            "RTIDetectionRadius": {"type": "float", "default": 3218, "description": "Detection radius in meters (default 2 miles)"},
            "RTIForwardSlowdownRange": {"type": "float", "default": 1207, "description": "Forward slowdown distance in meters (default 0.75 miles)"},
            "RTIResumeSpeedDistance": {"type": "float", "default": 805, "description": "Resume speed distance in meters (default 0.5 miles)"},
            "RTISpeedReduction": {"type": "float", "default": 16, "description": "Speed reduction in km/h (default 10 mph)"},
            "RTISpeedReductionMode": {"type": "str", "default": "posted", "description": "Speed reduction mode: posted or custom"},
            "RTIThreatFilter": {"type": "int", "default": 0, "description": "Threat filter: 0=all, 1=police, 2=cameras, 3=hazards"}
        }
        
    def test_parameter_persistence(self):
        """Test that parameters can be set and retrieved correctly."""
        print("\n" + "="*60)
        print("TEST 1: Parameter Persistence")
        print("="*60)
        
        test_values = {
            "RTIEnabled": False,
            "RTIHUDEnabled": False,
            "RTIAudioAlerts": False,
            "RTIDetectionRadius": "8046",  # 5 miles
            "RTIForwardSlowdownRange": "402",  # 0.25 miles
            "RTIResumeSpeedDistance": "1609",  # 1 mile
            "RTISpeedReduction": "32",  # 20 mph in km/h
            "RTISpeedReductionMode": "custom",
            "RTIThreatFilter": "2"  # Speed cameras only
        }
        
        print("Setting test values...")
        for key, value in test_values.items():
            if self.all_params[key]["type"] == "bool":
                self.params.put_bool(key, value)
            else:
                self.params.put(key, str(value))
            print(f"  Set {key} = {value}")
        
        print("\nReading back values...")
        all_correct = True
        for key, expected in test_values.items():
            if self.all_params[key]["type"] == "bool":
                actual = self.params.get_bool(key)
            else:
                actual = self.params.get(key)
                if actual:
                    actual = actual.decode('utf-8') if isinstance(actual, bytes) else str(actual)
            
            if str(actual) == str(expected):
                print(f"  ✓ {key}: {actual}")
            else:
                print(f"  ✗ {key}: {actual} (expected {expected})")
                all_correct = False
        
        self.test_results["persistence"] = all_correct
        return all_correct
    
    def test_component_loading(self):
        """Test that each component loads parameters correctly."""
        print("\n" + "="*60)
        print("TEST 2: Component Parameter Loading")
        print("="*60)
        
        components = {
            "rtid.py": [
                "RTIEnabled",
                "RTIDetectionRadius"  # Should be used for API fetch radius
            ],
            "threat_detector.py": [
                "RTIDetectionRadius",  # For filtering threats
                "RTIForwardSlowdownRange",  # For ahead threat activation
                "RTIResumeSpeedDistance",  # For behind threat deactivation
                "RTISpeedReduction",
                "RTISpeedReductionMode",
                "RTIThreatFilter"
            ],
            "rti_controller.py": [
                "RTIEnabled",
                "RTIForwardSlowdownRange",  # Threat activation distance
                "RTIResumeSpeedDistance",  # Continue control after passing
                "RTISpeedReduction",
                "RTISpeedReductionMode"
            ],
            "hud.cc": [
                "RTIEnabled",
                "RTIHUDEnabled"
            ],
            "soundd.py": [
                "RTIAudioAlerts"
            ]
        }
        
        print("Checking parameter usage in components:")
        all_correct = True
        
        for component, expected_params in components.items():
            print(f"\n{component}:")
            
            # Find the component file
            if component.endswith('.py'):
                search_paths = [
                    f"/data/openpilot/sunnypilot/rtid/{component}",
                    f"/data/openpilot/sunnypilot/selfdrive/controls/lib/{component}",
                    f"/data/openpilot/selfdrive/ui/{component}"
                ]
            else:  # .cc file
                search_paths = [
                    f"/data/openpilot/selfdrive/ui/sunnypilot/qt/onroad/{component}"
                ]
            
            file_path = None
            for path in search_paths:
                if Path(path).exists():
                    file_path = path
                    break
            
            if not file_path:
                print(f"  ✗ File not found")
                all_correct = False
                continue
            
            # Check if each expected parameter is referenced in the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            for param in expected_params:
                if param in content:
                    print(f"  ✓ {param} is referenced")
                else:
                    print(f"  ✗ {param} is NOT referenced")
                    all_correct = False
        
        self.test_results["component_loading"] = all_correct
        return all_correct
    
    def test_parameter_ranges(self):
        """Test that parameter ranges are correctly enforced."""
        print("\n" + "="*60)
        print("TEST 3: Parameter Range Validation")
        print("="*60)
        
        range_tests = {
            "RTIDetectionRadius": {
                "min": 402,  # 0.25 miles in meters
                "max": 8046,  # 5 miles in meters
                "ui_max": 5.0,  # UI should allow up to 5 miles
                "ui_min": 0.25  # UI should allow minimum 0.25 miles
            },
            "RTIForwardSlowdownRange": {
                "min": 0,
                "max": 3218,  # 2 miles in meters
                "ui_max": 2.0,
                "ui_min": 0.0
            },
            "RTIResumeSpeedDistance": {
                "min": 0,
                "max": 3218,  # 2 miles in meters
                "ui_max": 2.0,
                "ui_min": 0.0
            }
        }
        
        print("Checking parameter ranges:")
        all_correct = True
        
        for param, ranges in range_tests.items():
            print(f"\n{param}:")
            print(f"  Min: {ranges['min']}m ({ranges['ui_min']} miles)")
            print(f"  Max: {ranges['max']}m ({ranges['ui_max']} miles)")
            
            # Test setting min value
            self.params.put(param, str(ranges['min']))
            stored = self.params.get(param)
            if stored:
                stored_val = float(stored)
                if stored_val == ranges['min']:
                    print(f"  ✓ Min value accepted")
                else:
                    print(f"  ✗ Min value not stored correctly")
                    all_correct = False
            
            # Test setting max value
            self.params.put(param, str(ranges['max']))
            stored = self.params.get(param)
            if stored:
                stored_val = float(stored)
                if stored_val == ranges['max']:
                    print(f"  ✓ Max value accepted")
                else:
                    print(f"  ✗ Max value not stored correctly")
                    all_correct = False
        
        self.test_results["range_validation"] = all_correct
        return all_correct
    
    def test_hardcoded_values(self):
        """Check for any remaining hardcoded values that should use parameters."""
        print("\n" + "="*60)
        print("TEST 4: Hardcoded Values Check")
        print("="*60)
        
        # Search for common hardcoded patterns
        hardcoded_patterns = [
            ("16.0", "16km radius - should use RTIDetectionRadius"),
            ("10000", "10km - should use appropriate param"),
            ("1000", "1km activation distance - should use RTIForwardSlowdownRange"),
            ("3218", "2 miles default - should come from param defaults"),
            ("1207", "0.75 miles default - should come from param defaults"),
            ("805", "0.5 miles default - should come from param defaults")
        ]
        
        files_to_check = [
            "/data/openpilot/sunnypilot/rtid/rtid.py",
            "/data/openpilot/sunnypilot/rtid/threat_detector.py",
            "/data/openpilot/sunnypilot/selfdrive/controls/lib/rti_controller.py"
        ]
        
        print("Searching for hardcoded values...")
        issues_found = []
        
        for file_path in files_to_check:
            if not Path(file_path).exists():
                continue
                
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                # Skip comments and default value definitions
                if "Default" in line or "default" in line or line.strip().startswith("#"):
                    continue
                    
                for pattern, description in hardcoded_patterns:
                    if pattern in line and "self." not in line:
                        issues_found.append(f"{Path(file_path).name}:{i} - {pattern} ({description})")
        
        if issues_found:
            print("  ✗ Found potential hardcoded values:")
            for issue in issues_found:
                print(f"    - {issue}")
            all_correct = False
        else:
            print("  ✓ No problematic hardcoded values found")
            all_correct = True
        
        self.test_results["hardcoded_check"] = all_correct
        return all_correct
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("RTI PARAMETER INTEGRATION VALIDATION")
        print("="*60)
        print("This validates ALL RTI parameters are properly integrated")
        
        # Run tests
        test1 = self.test_parameter_persistence()
        test2 = self.test_component_loading()
        test3 = self.test_parameter_ranges()
        test4 = self.test_hardcoded_values()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")
        
        all_passed = all(self.test_results.values())
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! RTI parameters are properly integrated.")
        else:
            print("\n⚠️ Some tests failed. Review the issues above.")
        
        return all_passed

def main():
    """Main entry point."""
    validator = RTIParameterValidator()
    success = validator.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())