#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from vtsc_test_framework import VTSCTestBase

# Create test instance
test = VTSCTestBase()
test.setUp()

# Test with specific curvature
test_curvature = 0.12
print(f'Testing with input curvature: {test_curvature}')

# Check the mock model data we create
mock_data = test.create_mock_model_data(curvature=test_curvature, vision_confidence=0.9)
print(f'Mock orientation_rate: {mock_data.orientationRate.z[0]}')
print(f'Mock velocity: {mock_data.velocity.x[0]}')
print(f'Expected max_pred_curvature: {mock_data.orientationRate.z[0] / mock_data.velocity.x[0]}')

# Update with high confidence (full visibility)
test.update_vtsc(vision_confidence=0.9, curvature=test_curvature)

# Check what we got
print(f'VTSC _filtered_curvature: {test.vtsc._filtered_curvature}')
print(f'Occlusion last_valid_curvature: {test.vtsc._occlusion_state.last_valid_curvature}')
print(f'Vision status: {test.vtsc._occlusion_state.vision_status}')

# Try multiple updates to see if EMA causes issues
print("\nTesting EMA convergence:")
for i in range(5):
    test.update_vtsc(vision_confidence=0.9, curvature=test_curvature)
    print(f'Update {i+1}: _filtered_curvature = {test.vtsc._filtered_curvature:.6f}, last_valid = {test.vtsc._occlusion_state.last_valid_curvature:.6f}')
