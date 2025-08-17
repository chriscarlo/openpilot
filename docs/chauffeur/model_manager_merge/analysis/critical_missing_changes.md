# CRITICAL: Missing Changes for Model Version 9 Support

## Major Discovery

The version 9 update is NOT just about version numbers - it involves fundamental changes to model output schemas that require corresponding parser updates.

## Key Architectural Changes Required

### 1. Simplified Model Outputs (commit cd087a561 - "Simple plan")

The new v9 models use "simplified" outputs that don't use Multi-Hypothesis Prediction (MHP) for certain outputs. The chubbs-ssh-only branch added conditional logic to handle both formats:

**For 'lead' outputs:**
```python
# New logic in chubbs-ssh-only (MISSING in chauffeur-dev2)
if outs['lead'].shape[1] == 2 * ModelConstants.LEAD_MHP_SELECTION * ModelConstants.LEAD_TRAJ_LEN * ModelConstants.LEAD_WIDTH:
    # Simplified format (no MHP)
    self.parse_mdn('lead', outs, in_N=0, out_N=0,
                   out_shape=(ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH))
else:
    # Traditional format (with MHP)
    self.parse_mdn('lead', outs, in_N=ModelConstants.LEAD_MHP_N, out_N=ModelConstants.LEAD_MHP_SELECTION,
                   out_shape=(ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH))
```

**For 'plan' outputs:**
```python
# New logic in chubbs-ssh-only (MISSING in chauffeur-dev2)
if outs['plan'].shape[1] == 2 * ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH:
    # Simplified format (no MHP)
    self.parse_mdn('plan', outs, in_N=0, out_N=0,
                   out_shape=(ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
else:
    # Traditional format (with MHP)
    self.parse_mdn('plan', outs, in_N=ModelConstants.PLAN_MHP_N, out_N=ModelConstants.PLAN_MHP_SELECTION,
                   out_shape=(ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
```

### 2. Model Generation Concept

Sunnypilot has introduced a "generation" concept where:
- Generation >= 12 models use simplified outputs
- The generation comes from the model bundle metadata
- There's a `DynamicModeldOutputs` parameter that can enable dynamic parsing

**chauffeur-dev2 has NONE of this infrastructure**

### 3. Missing Parser Updates

**File**: `selfdrive/modeld/parse_model_outputs.py`

chauffeur-dev2 is missing the conditional parsing logic added in commit cd087a561. Without this, the new v9 models will likely crash or produce incorrect outputs when their simplified format isn't handled correctly.

## Impact Assessment

### Will v9 Models Work Without These Changes?

**NO** - The models will likely fail in one of these ways:

1. **Shape Mismatch Errors**: The parser expects MHP format but receives simplified format
2. **Incorrect Parsing**: The data will be interpreted wrong, leading to erratic driving behavior
3. **Runtime Crashes**: Shape mismatches will cause numpy reshape errors

### What Happens If We Only Update Version Numbers?

If we only update the selector version and URL (as done in the previous commit):
- Models will download successfully
- Models will load into memory
- **Models will FAIL when outputs are parsed**
- System will crash or behave unpredictably

## Required Additional Changes

### Minimum Required (High Risk)
1. Update `selfdrive/modeld/parse_model_outputs.py` with conditional parsing logic
2. Test thoroughly with both old and new model formats

### Recommended (Lower Risk)
1. Port the full sunnypilot parsing infrastructure:
   - Generation concept from model bundles
   - DynamicModeldOutputs parameter
   - Full conditional parsing logic
2. Update model bundle structure to include generation field
3. Extensive testing with various model versions

## Timeline Analysis

The changes happened in this sequence:
1. **July 19, 2025**: "Refactor Modeld to Allow Dynamic Plan and Lead" - Initial infrastructure
2. **July 26, 2025**: Tinygrad backend changes (DEV=DEVICE)
3. **August 11, 2025**: "Simple plan" - Added conditional parsing
4. **August 12, 2025**: Version bump to 9 and model URL change

## Recommendation

**DO NOT USE V9 MODELS WITHOUT THE PARSER UPDATES**

The current changes (just version numbers) are insufficient and dangerous. The system needs:
1. Parser updates to handle simplified outputs
2. Testing infrastructure to validate both formats
3. Gradual rollout with ability to rollback

## Test Case to Verify

```python
# This will fail with v9 models on current chauffeur-dev2:
import numpy as np
from selfdrive.modeld.parse_model_outputs import Parser

# Simulate simplified v9 model output shape
parser = Parser()
outs = {
    'plan': np.zeros((1, 2 * 33 * 15)),  # Simplified format
    'lead': np.zeros((1, 2 * 3 * 6 * 4))  # Simplified format  
}

# This will crash or produce wrong results
parser.parse_policy_outputs(outs)  # Will expect MHP format
parser.parse_vision_outputs(outs)  # Will expect MHP format
```

## Conclusion

The version 9 update involves **structural changes to model outputs** that require corresponding parser updates. Simply updating version numbers without these parser changes will result in a **non-functional or dangerous system**.

The changes go beyond just model manager versioning - they represent a fundamental shift in how model outputs are structured (simplified vs MHP format), requiring careful adaptation of the parsing logic.