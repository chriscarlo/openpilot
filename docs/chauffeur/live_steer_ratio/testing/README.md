# LiveSteerRatio Test Suite

This directory contains test scripts for the LiveSteerRatio feature.

## Test Scripts

### test_live_steer_ratio.py

Comprehensive unit tests covering:
- KIA EV6 default steer ratio verification
- LiveSteerRatio parameter handling
- Bounds calculation with different values
- GUI parameter storage and retrieval

Run with:
```bash
cd /data/openpilot
python3 docs/claude/tests/live_steer_ratio/test_live_steer_ratio.py
```

Expected output: All 4 tests should pass.

### demo_live_steer_ratio.py

Interactive demonstration showing:
- Default behavior without LiveSteerRatio
- Setting LiveSteerRatio to 0 (use default)
- Setting custom values
- GUI interaction instructions
- Practical tuning guidelines

Run with:
```bash
cd /data/openpilot
python3 docs/claude/tests/live_steer_ratio/demo_live_steer_ratio.py
```

## Test Coverage

The test suite covers:

1. **Configuration Tests**
   - Vehicle default values
   - Parameter storage/retrieval
   - Special value handling (0 = default)

2. **Calculation Tests**
   - Bounds calculation correctness
   - Parameter learning integration
   - Base vs learned value separation

3. **Integration Tests**
   - Parameter flow through system
   - GUI to backend communication
   - Persistence across restarts

## Running All Tests

To run the complete test suite:

```bash
cd /data/openpilot

# Run unit tests
python3 docs/claude/tests/live_steer_ratio/test_live_steer_ratio.py

# Run demo (interactive)
python3 docs/claude/tests/live_steer_ratio/demo_live_steer_ratio.py
```

## Adding New Tests

When adding tests:
1. Follow the existing test structure
2. Test both normal and edge cases
3. Verify cleanup (parameter removal)
4. Document expected behavior

## Known Issues

- Tests require rebuilt common module with LiveSteerRatio key
- Some tests may need mock objects for CI environments
- Parameter changes don't affect running instances of paramsd