# Detailed Merge Strategy: Model Manager Version 9 Support

## Executive Summary

chauffeur-dev2 requires minimal changes to support version 9 models since it already has the necessary tinygrad backend configuration. The main updates needed are the selector version bump and model URL change.

## Critical Constraint

**⚠️ CHAUFFEUR-DEV2 MUST NOT USE SUBMODULES**
- All code must be directly integrated into the main repository
- Any tinygrad updates must be copied file-by-file, not via submodule updates
- The flattened repository structure must be preserved

## Phase 1: Pre-Merge Verification (REQUIRED)

### 1.1 Establish Baseline
```bash
# Document current working state
python3 -c "from sunnypilot.models.helpers import CURRENT_SELECTOR_VERSION; print(f'Current version: {CURRENT_SELECTOR_VERSION}')"
# Expected: Current version: 8

# Test current model loading capability
python3 sunnypilot/models/fetcher.py  # May need modification to test
```

### 1.2 Compare Tinygrad Versions
```bash
# Clone tinygrad at the target commit
git clone https://github.com/tinygrad/tinygrad.git /tmp/tinygrad_target
cd /tmp/tinygrad_target
git checkout d2bb1bcb976f106a41928f2d66d354ab7afd6f59

# Compare with current integration
diff -r /data/openpilot/tinygrad_repo /tmp/tinygrad_target > /tmp/tinygrad_diff.txt
# Review differences carefully
```

### 1.3 Backup Current State
```bash
# Create backup branch
git checkout -b pre-model-v9-backup
git add -A
git commit -m "Backup before model manager v9 integration"
git checkout chauffeur-dev2
```

## Phase 2: Core Updates

### 2.1 Update Selector Version
**File**: `sunnypilot/models/helpers.py`
```python
# Change from:
CURRENT_SELECTOR_VERSION = 8
REQUIRED_MIN_SELECTOR_VERSION = 6

# To:
CURRENT_SELECTOR_VERSION = 9
REQUIRED_MIN_SELECTOR_VERSION = 9
```

### 2.2 Update Model URL
**File**: `sunnypilot/models/fetcher.py`
```python
# Change from:
MODEL_URL = "https://docs.sunnypilot.ai/driving_models_v6.json"

# To:
MODEL_URL = "https://docs.sunnypilot.ai/driving_models_v7.json"
```

### 2.3 Clear Model Cache
```bash
# Clear any cached models to force fresh download
rm -rf /data/params/d/ModelManager_ModelsCache
rm -rf /data/params/d/ModelManager_ActiveBundle
```

## Phase 3: Tinygrad Updates (If Needed)

### 3.1 Identify Required Changes
Based on the diff from Phase 1.2, determine if any tinygrad updates are needed.

**DO NOT**:
- Use `git submodule update`
- Add `.gitmodules` file
- Create submodule references

**DO**:
- Copy individual files that have changed
- Maintain the flat directory structure
- Preserve any local modifications

### 3.2 Manual File Integration
If tinygrad updates are needed:
```bash
# For each changed file identified in the diff
cp /tmp/tinygrad_target/path/to/file.py /data/openpilot/tinygrad_repo/path/to/file.py

# Example for a critical file:
cp /tmp/tinygrad_target/tinygrad/device.py /data/openpilot/tinygrad_repo/tinygrad/device.py
```

## Phase 4: Testing

### 4.1 Basic Functionality Test
```python
# Test tinygrad import and backend
cd /data/openpilot
python3 -c "
import os
os.environ['LLVM'] = '1'
from tinygrad.tensor import Tensor
from tinygrad.device import Device
print(f'Device: {Device.DEFAULT}')
t = Tensor([1,2,3])
print(f'Tensor works: {t.numpy()}')
"
```

### 4.2 Model Selector Test
```python
# Test version compatibility
cd /data/openpilot
python3 -c "
from sunnypilot.models.helpers import (
    CURRENT_SELECTOR_VERSION, 
    REQUIRED_MIN_SELECTOR_VERSION,
    is_bundle_version_compatible
)
print(f'Selector Version: {CURRENT_SELECTOR_VERSION}')
print(f'Min Required: {REQUIRED_MIN_SELECTOR_VERSION}')

# Test bundle compatibility
test_bundle = {'minimumSelectorVersion': 9}
print(f'V9 bundle compatible: {is_bundle_version_compatible(test_bundle)}')

test_bundle_old = {'minimumSelectorVersion': 8}
print(f'V8 bundle compatible: {is_bundle_version_compatible(test_bundle_old)}')
"
```

### 4.3 Model Fetching Test
```python
# Test fetching new models
cd /data/openpilot
python3 -c "
from openpilot.common.params import Params
from sunnypilot.models.fetcher import ModelFetcher
params = Params()
fetcher = ModelFetcher(params)
# This should fetch from v7 endpoint
# Monitor for errors
"
```

## Phase 5: Integration Validation

### 5.1 Build Test
```bash
cd /data/openpilot
scons -u -j$(nproc) --minimal
# Ensure build succeeds with new configuration
```

### 5.2 Model Compilation Test
```bash
# Test that model compilation still works
cd /data/openpilot/selfdrive/modeld
# Check if compile scripts work with new tinygrad
```

### 5.3 RTI Feature Validation
```bash
# Ensure RTI features still work
python3 sunnypilot/rtid/tests/test_rtid.py
python3 sunnypilot/rtid/tests/test_integration_flow.py
```

## Phase 6: Commit and Document

### 6.1 Commit Changes
```bash
git add sunnypilot/models/helpers.py
git add sunnypilot/models/fetcher.py
# Add any tinygrad files if updated
git commit -m "feat: Update model manager to version 9

- Bump CURRENT_SELECTOR_VERSION to 9
- Bump REQUIRED_MIN_SELECTOR_VERSION to 9  
- Update model URL to driving_models_v7.json
- Enables access to new models: Falling Phoenix, Down To Ride, etc.

Note: Tinygrad backend configuration already compatible"
```

### 6.2 Document Changes
Create a changelog entry documenting:
- Version bump rationale
- New models now available
- Any breaking changes
- Testing performed

## Rollback Plan

If issues occur:
```bash
# Revert to backup
git checkout pre-model-v9-backup

# Or revert specific changes
git revert HEAD  # If already committed

# Clear potentially corrupted cache
rm -rf /data/params/d/ModelManager_*
```

## Risk Mitigation

### Known Risks
1. **Model Incompatibility**: V9 models may not load if tinygrad isn't fully updated
2. **Cache Corruption**: Old cached models may cause issues
3. **Build Failures**: Compilation flags may need adjustment

### Mitigation Steps
1. Test thoroughly in development environment first
2. Clear all model caches before and after update
3. Have rollback branch ready
4. Test both old and new model loading

## Success Criteria

✅ Selector version shows 9
✅ Model URL points to v7 endpoint  
✅ New models appear in model list
✅ Existing features (RTI, etc.) still work
✅ Build completes successfully
✅ No submodule dependencies introduced

## DO NOT DO

❌ Run `git submodule` commands
❌ Create `.gitmodules` file
❌ Reference external repositories in build
❌ Break the flattened repository structure
❌ Assume tinygrad versions are identical
❌ Skip testing phases