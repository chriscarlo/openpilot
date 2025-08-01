# Enhanced VTSC Test Suite

## Production Integration Tests

### Core Tests
- `test_production_integration_simple.py` - Main integration test following KISS principles
- `test_production_integration.py` - Original integration test (deprecated)

### Performance Tests  
- `test_integrated_expanded.py` - 10-scenario comprehensive test suite
- `run_both_tests.py` - Runs both emergency and integrated test suites

### Basic Test
- `test_vtsc_basic.py` - Basic VTSC functionality test

## Running Tests

### Quick Integration Test
```bash
cd /data/openpilot/docs/claude/tests/vtsc
python3 test_production_integration_simple.py
```

### Comprehensive Performance Test
```bash
cd /data/openpilot/docs/claude/tests/vtsc  
python3 test_integrated_expanded.py
```

### All Tests
```bash
cd /data/openpilot/docs/claude/tests/vtsc
python3 run_both_tests.py
```

## Expected Results
- **Integration Test**: All tests should pass
- **Performance Test**: ≥90% success rate (9/10 scenarios)
- **System Constraint**: All deceleration ≤ 6.0 m/s²

## Test Environment
Tests use mock data and do not require actual vehicle hardware. They validate:
- Basic functionality
- Enhanced feature integration
- System constraint compliance
- Backward compatibility
- Parameter handling