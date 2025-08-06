#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

from vtsc_test_framework import VTSCTestBase

# Create test instance
test = VTSCTestBase()
test.setUp()

# Start with full visibility and specific curvature
test_curvature = 0.12
print(f'=== Starting with full visibility curvature: {test_curvature} ===')

# Multiple updates for convergence
for i in range(10):
    test.update_vtsc(vision_confidence=0.9, curvature=test_curvature)
    print(f'Full visibility update {i+1}: last_valid = {test.vtsc._occlusion_state.last_valid_curvature:.6f}')

print('\nAfter convergence:')
print(f'Vision status: {test.vtsc._occlusion_state.vision_status}')
print(f'last_valid_curvature: {test.vtsc._occlusion_state.last_valid_curvature:.6f}')

# Store the converged value
converged_value = test.vtsc._occlusion_state.last_valid_curvature

# Transition to occlusion
print('\n=== Transitioning to occlusion ===')
test.update_vtsc(vision_confidence=0.25)  # SEVERE_OCCLUSION
print('After occlusion transition:')
print(f'Vision status: {test.vtsc._occlusion_state.vision_status}')
print(f'last_valid_curvature: {test.vtsc._occlusion_state.last_valid_curvature:.6f}')

# Test with different curvature during occlusion
print('\n=== Testing different curvature during occlusion ===')
different_curvature = 0.20
test.update_vtsc(vision_confidence=0.25, curvature=different_curvature)
print('After different curvature update during occlusion:')
print(f'Vision status: {test.vtsc._occlusion_state.vision_status}')
print(f'last_valid_curvature: {test.vtsc._occlusion_state.last_valid_curvature:.6f}')
print(f'Expected (should be same): {converged_value:.6f}')
print(f'Difference: {abs(test.vtsc._occlusion_state.last_valid_curvature - converged_value):.6f}')
