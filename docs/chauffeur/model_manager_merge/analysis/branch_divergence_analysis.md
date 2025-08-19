# Model Manager Merge Analysis: chubbs-ssh-only → chauffeur-dev2

## Executive Summary

The `chubbs-ssh-only` branch has critical model manager updates that enable new models, but merging requires careful handling due to architectural changes and the removal of submodules in `chauffeur-dev2`.

## Key Findings

### 1. Branch Divergence
- **Common Ancestor**: `ed4b6f8bfc87381fd02bc5ea792fc8d7688c24f8`
- **chauffeur-dev2**: 78 commits ahead
- **chubbs-ssh-only**: 331 commits ahead
- **Major Structural Difference**: chubbs-ssh-only uses submodules, chauffeur-dev2 is flattened

### 2. Model Selector Version Mismatch (CRITICAL)
**Current State:**
- **chubbs-ssh-only**: 
  - CURRENT_SELECTOR_VERSION = 9
  - REQUIRED_MIN_SELECTOR_VERSION = 9
- **chauffeur-dev2**: 
  - CURRENT_SELECTOR_VERSION = 8
  - REQUIRED_MIN_SELECTOR_VERSION = 6

**Impact**: New models on chubbs-ssh-only require selector version 9, making them invisible to chauffeur-dev2.

### 3. Architectural Changes

#### Tinygrad Backend Configuration (BREAKING CHANGE)
The way tinygrad is configured has fundamentally changed:

**Old Style (chauffeur-dev2):**
```python
os.environ['DEV'] = 'QCOM' if TICI else 'LLVM'
```

**New Style (chubbs-ssh-only):**
```python
if TICI:
  os.environ['QCOM'] = '1'
else:
  os.environ['LLVM'] = '1'
```

This affects:
- `selfdrive/modeld/modeld.py`
- `selfdrive/modeld/dmonitoringmodeld.py`
- `selfdrive/modeld/SConscript`

#### Tinygrad Submodule Version
- chubbs-ssh-only uses: `d2bb1bcb976f106a41928f2d66d354ab7afd6f59`
- This version includes the new backend configuration system

### 4. Submodules in chubbs-ssh-only
- panda (github.com/sunnyhaibin/panda.git)
- opendbc_repo (github.com/sunnypilot/opendbc.git)
- msgq_repo (github.com/sunnypilot/msgq.git)
- rednose_repo (github.com/commaai/rednose.git)
- teleoprtc_repo (github.com/commaai/teleoprtc)
- tinygrad_repo (github.com/tinygrad/tinygrad.git)
- sunnypilot/neural_network_data (github.com/sunnypilot/neural-network-data.git)

### 5. Custom Chauffeur Features to Preserve
- **RTI (Realtime Traffic Intelligence)**: Complete implementation in `sunnypilot/rtid/`
- **Enhanced road matching**: Custom implementation
- **Waze API integration**: Custom client
- **Speed control modifications**: Integrated with RTI
- **Custom HUD modifications**: For RTI display
- **SSH hardcoding**: Custom authentication for development
- **Athena bypass**: Disabled cloud registration

### 6. New Models Available (Version 9)
Recent models added in chubbs-ssh-only:
- Falling Phoenix
- Down To Ride
- Space Lab series
- Kumars vibe (weight average)
- Various other performance-tuned models

## Critical Files Requiring Manual Merge

1. **Model Configuration**:
   - `sunnypilot/models/helpers.py` (selector version)
   - `selfdrive/modeld/modeld.py` (tinygrad config)
   - `selfdrive/modeld/dmonitoringmodeld.py` (tinygrad config)
   - `selfdrive/modeld/SConscript` (build flags)

2. **Process Management**:
   - `system/manager/process_config.py` (RTI process)
   - `system/manager/manager.py` (custom params)

3. **UI Components**:
   - `selfdrive/ui/sunnypilot/qt/offroad/settings/models_panel.cc`
   - `selfdrive/ui/sunnypilot/qt/offroad/settings/osm/models_fetcher.cc`

## Risks and Challenges

1. **Binary Incompatibility**: Models compiled for selector v9 won't work with v8 infrastructure
2. **Tinygrad API Breaking Changes**: Backend configuration is completely different
3. **Missing Dependencies**: Submodule code needs to be properly integrated
4. **Feature Conflicts**: Custom Chauffeur features may conflict with upstream changes

## Recommended Merge Strategy

### Phase 1: Prepare Environment
1. Create backup branch of current chauffeur-dev2
2. Set up test environment with ability to rollback

### Phase 2: Core Infrastructure Updates
1. Update tinygrad to match chubbs-ssh-only version
2. Apply tinygrad backend configuration changes
3. Update model selector version to 9
4. Test basic model loading

### Phase 3: Feature-by-Feature Integration
1. Apply model manager UI updates
2. Update model fetching logic
3. Integrate new model definitions
4. Preserve all RTI customizations

### Phase 4: Testing and Validation
1. Verify existing models still work
2. Test new model downloading
3. Validate RTI features remain functional
4. Run full integration tests

## Next Steps

1. Create detailed file-by-file merge plan
2. Identify potential conflict resolution strategies
3. Set up testing infrastructure
4. Begin incremental merge process

## Command Reference

```bash
# View differences for critical files
git diff origin/chubbs-ssh-only chauffeur-dev2 -- FILE_PATH

# Cherry-pick specific commits
git cherry-pick COMMIT_HASH

# Test merge without committing
git merge --no-commit --no-ff origin/chubbs-ssh-only

# Compare specific functionality
git diff origin/chubbs-ssh-only:PATH chauffeur-dev2:PATH
```