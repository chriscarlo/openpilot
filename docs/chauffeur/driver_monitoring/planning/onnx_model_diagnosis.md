# ONNX Model Build Error Diagnosis

## Problem Summary
The build fails with two errors:
1. `google.protobuf.message.DecodeError: Error parsing message with type 'onnx.ModelProto'` for driving_policy.onnx
2. `ValueError: Unknown wire type: 6` for dmonitoring_model.onnx

## Root Cause
The ONNX model files are **Git LFS pointer files**, not actual model files. They contain only metadata pointers instead of the actual neural network data.

## Evidence

### 1. File Sizes Are Too Small
```
driving_vision.onnx   - 133 bytes (should be ~3MB)
driving_policy.onnx   - 133 bytes (should be ~15MB)  
dmonitoring_model.onnx - 132 bytes (should be ~7MB)
```

### 2. File Contents Are LFS Pointers
Example from driving_policy.onnx:
```
version https://git-lfs.github.com/spec/v1
oid sha256:18e3b8ed118a44ce23af4adc315d9ed699abc480f5392e83b2f9ca6520fbd992
size 15583374
```

### 3. Git LFS Configuration
- `.gitattributes` shows `*.onnx filter=lfs diff=lfs merge=lfs -text`
- Git LFS is installed: `git-lfs/3.4.1`
- Files are tracked by LFS: `git lfs ls-files` shows the three ONNX files

### 4. Build Scripts Expect Real Files
- `get_model_metadata.py` tries to load with `onnx.load()`
- `compile3.py` tries to parse the ONNX file structure
- Both fail because they're reading LFS pointer text instead of binary ONNX data

## Solution Required
The actual model files need to be downloaded with:
```bash
git lfs pull
```

## Additional Findings
- Sunnypilot has a custom model management system in `/sunnypilot/models/`
- This includes a fetcher that can download models from remote sources
- The modeld build is trying to use the standard openpilot models, not sunnypilot's custom system

## Important Notes
- This is NOT related to the driver monitoring changes
- This is NOT related to the rednose submodule issue
- The build would fail for anyone who cloned without LFS files
- This is a common issue when cloning repositories with large files