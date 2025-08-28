#!/bin/bash
# Run all test suites from project root with proper PYTHONPATH

echo "============================================================"
echo "RUNNING ALL ADAPTIVE DECELERATION SYSTEM TESTS"
echo "============================================================"

# Set PYTHONPATH to project root
export PYTHONPATH=/projects/chauffeur/data/openpilot:$PYTHONPATH

# Track results
PASSED=0
FAILED=0

# Test suites
TESTS=(
    "docs/chauffeur/vtsc/testing/adaptive_deceleration/test_adaptive_system.py"
    "docs/chauffeur/vtsc/testing/parameter_validation/test_parameter_loading.py"
    "docs/chauffeur/vtsc/testing/physics_calculations/test_physics_decel.py"
    "docs/chauffeur/vtsc/testing/filtering/test_ema_filtering.py"
    "docs/chauffeur/vtsc/testing/integration/test_full_integration.py"
)

# Run each test
for test in "${TESTS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Running: $test"
    echo "------------------------------------------------------------"
    
    if python3 "$test"; then
        echo "✓ PASSED: $test"
        ((PASSED++))
    else
        echo "✗ FAILED: $test"
        ((FAILED++))
    fi
done

# Summary
echo ""
echo "============================================================"
echo "FINAL SUMMARY"
echo "============================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total: $((PASSED + FAILED))"
echo "Success Rate: $(( PASSED * 100 / (PASSED + FAILED) ))%"
echo "============================================================"