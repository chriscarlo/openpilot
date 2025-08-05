#!/usr/bin/env python3
"""
VTSC Unit Test Runner

Runs all VTSC unit tests and provides comprehensive results summary.
This is the main entry point for validating all integrated VTSC functionality.

Usage:
    python3 run_all_unit_tests.py              # Run all tests
    python3 run_all_unit_tests.py --verbose    # Run with detailed output
    python3 run_all_unit_tests.py --help       # Show help
"""

import sys
import argparse
import time
from pathlib import Path

# Add shared directory to path
current_dir = Path(__file__).parent
shared_dir = current_dir / 'shared'
unit_tests_dir = current_dir / 'unit_tests'
sys.path.append(str(shared_dir))

# Import test framework
from vtsc_test_framework import TestResult

# Import individual test suites
sys.path.append(str(unit_tests_dir))

def run_all_tests(verbose: bool = False) -> list[TestResult]:
    """Run all VTSC unit tests and return results"""

    all_results = []

    print("=" * 80)
    print("VTSC COMPREHENSIVE UNIT TEST SUITE")
    print("=" * 80)
    print(f"Running tests from: {unit_tests_dir}")
    print(f"Test framework: {shared_dir / 'vtsc_test_framework.py'}")
    print()

    # Test Suite 1: Vision Occlusion State Transitions
    print("VISION OCCLUSION STATE TESTS")
    print("-" * 40)

    try:
        from test_vision_occlusion_state_transitions import run_vision_occlusion_tests
        occlusion_results = run_vision_occlusion_tests()
        all_results.extend(occlusion_results)

        passed = sum(1 for r in occlusion_results if r.passed)
        total = len(occlusion_results)
        print(f"Vision Occlusion Tests: {passed}/{total} passed")

    except Exception as e:
        print(f"Vision Occlusion Tests: FAILED TO RUN - {e}")
        error_result = TestResult("VisionOcclusionTests")
        error_result.mark_failed(f"Failed to run: {e}", 0.0)
        all_results.append(error_result)

    print()

    # Test Suite 2: Emergency Escalation Logic
    print("EMERGENCY ESCALATION LOGIC TESTS")
    print("-" * 40)

    try:
        from test_emergency_escalation_logic import run_emergency_escalation_tests
        escalation_results = run_emergency_escalation_tests()
        all_results.extend(escalation_results)

        passed = sum(1 for r in escalation_results if r.passed)
        total = len(escalation_results)
        print(f"Emergency Escalation Tests: {passed}/{total} passed")

    except Exception as e:
        print(f"Emergency Escalation Tests: FAILED TO RUN - {e}")
        error_result = TestResult("EmergencyEscalationTests")
        error_result.mark_failed(f"Failed to run: {e}", 0.0)
        all_results.append(error_result)

    print()

    # TODO: Add future test suites here
    # Future test suites will be added as they are implemented:
    # - Curvature Processing Tests
    # - Activation Logic Tests
    # - Data Integration Tests
    # - Error Handling Tests

    return all_results

def print_detailed_results(results: list[TestResult], verbose: bool = False):
    """Print detailed test results summary"""

    print("=" * 80)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)

    # Overall statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests

    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")

    if total_tests > 0:
        pass_rate = (passed_tests / total_tests) * 100
        print(f"Pass Rate: {pass_rate:.1f}%")

    # Execution time summary
    total_time = sum(r.execution_time for r in results)
    avg_time = total_time / total_tests if total_tests > 0 else 0
    print(f"Total Execution Time: {total_time:.3f}s")
    print(f"Average Test Time: {avg_time:.3f}s")

    print()

    # Group results by test suite
    suites = {}
    for result in results:
        suite_name = result.test_name.split('.')[0] if '.' in result.test_name else "Unknown"
        if suite_name not in suites:
            suites[suite_name] = []
        suites[suite_name].append(result)

    # Print results by test suite
    for suite_name, suite_results in suites.items():
        suite_passed = sum(1 for r in suite_results if r.passed)
        suite_total = len(suite_results)
        suite_status = "PASS" if suite_passed == suite_total else "FAIL"

        print(f"{suite_status:4} | {suite_name:<40} | {suite_passed:2}/{suite_total:2} | {sum(r.execution_time for r in suite_results):.3f}s")

        if verbose or suite_passed < suite_total:
            for result in suite_results:
                status = "PASS" if result.passed else "FAIL"
                method_name = result.test_name.split('.')[-1] if '.' in result.test_name else result.test_name
                print(f"    {status:4} | {method_name:<35} | {result.execution_time:.3f}s")
                if not result.passed and verbose:
                    print(f"         | Error: {result.error_message}")
            print()

    print("=" * 80)

    # Critical findings summary
    if failed_tests > 0:
        print("CRITICAL FINDINGS:")
        print("-" * 40)
        for result in results:
            if not result.passed:
                print(f"• {result.test_name}")
                print(f"  Error: {result.error_message}")
        print()
        print("WARNING: VTSC integration has failing tests - review and fix before deployment!")
    else:
        print("ALL TESTS PASSED")
        print("-" * 40)
        print("SUCCESS: VTSC integration is validated and ready for production!")
        print("All critical safety and functionality tests are passing.")

    print("=" * 80)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run all VTSC unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 run_all_unit_tests.py              # Run all tests
    python3 run_all_unit_tests.py --verbose    # Detailed output
    python3 run_all_unit_tests.py -v           # Short form verbose
        """
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed test output and error messages'
    )

    args = parser.parse_args()

    # Record start time
    start_time = time.time()

    # Run all tests
    try:
        results = run_all_tests(verbose=args.verbose)
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)  # 128 + SIGINT
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        sys.exit(2)

    # Record end time
    end_time = time.time()
    total_execution_time = end_time - start_time

    print(f"\nTotal suite execution time: {total_execution_time:.3f}s")

    # Print results
    print_detailed_results(results, verbose=args.verbose)

    # Exit with appropriate code
    failed_tests = [r for r in results if not r.passed]
    if failed_tests:
        sys.exit(1)  # Tests failed
    else:
        sys.exit(0)  # All tests passed

if __name__ == "__main__":
    main()
