# Root Cause Analysis: Selector Version 9 Requirements

## Critical Finding

The bump to selector version 9 was driven by a **fundamental tinygrad backend configuration change** that requires all models to be recompiled.

## Timeline of Changes

### 1. July 26, 2025 - Tinygrad Backend Architecture Change
**Commit**: `35ed6bc3a` - "Tinygrad DEV=DEVICE (#35814)"

This changed how tinygrad backends are configured:

**OLD Configuration (Version 8 and below):**
```python
os.environ['DEV'] = 'QCOM' if TICI else 'LLVM'
```

**NEW Configuration (Version 9):**
```python
if TICI:
  os.environ['QCOM'] = '1'
else:
  os.environ['LLVM'] = '1'
```

### 2. August 12, 2025 - Selector Version Bump
**Commit**: `68625222b` - "chore: sync tinygrad (#1151)"

This commit:
- Bumped `CURRENT_SELECTOR_VERSION` from 8 to 9
- Bumped `REQUIRED_MIN_SELECTOR_VERSION` from 8 to 9  
- Changed model URL from `driving_models_v6.json` to `driving_models_v7.json`
- Updated tinygrad submodule to `d2bb1bcb9`

## Why Version 9 is Required

According to `sunnypilot/models/README.md`, when there's a "deep change in selector behavior that requires all models to be recompiled", the team:

1. Creates a new JSON file (v6 → v7)
2. Assigns updated `minimumSelectorVersion` values
3. Bumps both `CURRENT_SELECTOR_VERSION` and `REQUIRED_MIN_SELECTOR_VERSION`

This ensures:
- Old selector versions can't load new models (they would crash due to backend mismatch)
- New selector versions won't load old cached models compiled with the old backend
- Clean separation between model generations

## Impact on Merge

### Binary Incompatibility
Models compiled with the old tinygrad backend (`DEV=QCOM/LLVM`) are **fundamentally incompatible** with the new backend configuration (`QCOM=1/LLVM=1`). Attempting to run old models with new code or vice versa will cause crashes.

### Required Changes for chauffeur-dev2

To support the new models, chauffeur-dev2 must:

1. **Update tinygrad** to at least commit `d2bb1bcb9`
2. **Apply backend configuration changes** in:
   - `selfdrive/modeld/modeld.py`
   - `selfdrive/modeld/dmonitoringmodeld.py`
   - `selfdrive/modeld/SConscript`
3. **Bump selector versions** to 9 in `sunnypilot/models/helpers.py`
4. **Update model URL** to `driving_models_v7.json`

### Models Requiring Version 9

All models compiled after July 26, 2025 require version 9:
- Down To Ride model
- Falling Phoenix
- Space Lab 3
- Kumars vibe (weight average)
- And all subsequent models

## Verification Steps

Before merging:
1. Verify current tinygrad version in chauffeur-dev2
2. Test that existing models still load after backend changes
3. Confirm new v9 models can be downloaded and loaded
4. Validate no runtime crashes with new backend configuration

## Conclusion

The version 9 requirement is **not optional** - it's a hard requirement due to fundamental changes in how tinygrad compiles and executes models. Without these changes, the new models cannot run on chauffeur-dev2.