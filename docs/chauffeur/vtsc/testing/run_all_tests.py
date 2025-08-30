#!/usr/bin/env python3
"""
Master test runner for Adaptive Deceleration System
Executes all test suites and provides comprehensive results
"""

import sys
import os
import subprocess
import os
from pathlib import Path

# Test suite files
TEST_SUITES = [
    # Core suites
    "adaptive_deceleration/test_adaptive_system.py",
    "parameter_validation/test_parameter_loading.py",
    "physics_calculations/test_physics_decel.py",
    "filtering/test_ema_filtering.py",
    "integration/test_full_integration.py",
    "integration/test_high_value_scenarios.py",
    "integration/test_multi_occluded_curves.py",
    # Acceptance (business outcomes)
    "acceptance/test_vtsc_acceptance.py",
]

ROOT = Path(__file__).resolve().parents[3]


def run_test_suite(test_file):
    """Run a single test suite and return results"""
    test_path = Path(__file__).parent / test_file
    
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print(f"{'='*70}")
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH','')}"
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test suite timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run test suite: {e}")
        return False


def main():
    """Run all test suites and report overall results"""
    print("="*70)
    print("ADAPTIVE DECELERATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Running {len(TEST_SUITES)} test suites...")
    
    results = {}
    
    for suite in TEST_SUITES:
        success = run_test_suite(suite)
        results[suite] = success
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL TEST RESULTS")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for suite, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {suite}")
    
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY: {passed}/{total} test suites passed")
    print(f"Success rate: {(passed/total*100):.1f}%")
    print(f"{'='*70}")
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
