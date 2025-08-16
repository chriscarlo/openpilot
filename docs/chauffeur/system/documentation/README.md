# Claude Code Documentation & Tests

This directory contains organized, verified documentation and tests for features developed with Claude Code. All content reflects the **current state** of the codebase as of the latest organization effort.

## Organization Principles

✅ **Current & Verified**: All documentation and tests reflect actual working functionality  
✅ **No Templates**: Removed placeholder files with no content value  
✅ **Working Tests Only**: All test files execute successfully against production code  
✅ **Accurate Documentation**: All claims are verified against actual implementation  

## Directory Structure

### `/documentation/`
Production-ready feature documentation with implementation details.

#### `/documentation/dependencies/`
Comprehensive dependency analysis and setup guides:
- `README.md` - Overview of all OpenPilot dependencies (150+ packages)
- `build-dependencies.md` - Build system requirements
- `hardware-dependencies.md` - Hardware-specific drivers and requirements
- `python-dependencies.md` - Python package requirements
- `python312-venv-build-guide.md` - Python 3.12 virtual environment setup
- `system-dependencies.md` - System-level package requirements

#### `/documentation/live_steer_ratio/`
LiveSteerRatio feature allowing real-time steering ratio adjustment:
- `README.md` - Complete implementation overview and usage
- `CHANGES_SUMMARY.md` - Summary of changes made to implement feature
- `IMPLEMENTATION_NOTES.md` - Technical implementation details

**Status**: ✅ **IMPLEMENTED** - Feature exists in production codebase

#### `/documentation/system/`
System-level documentation:
- `claude_code_system_prompt.md` - Claude Code system integration notes

#### `/documentation/vtsc/`
Vision Turn Speed Controller (VTSC) documentation:
- `PHYSICS_SCRIPT_INTEGRATION.md` - **CRITICAL**: Documents the post-apex acceleration bug fix

**Status**: ✅ **CURRENT** - Reflects latest VTSC production implementation

### `/tests/`
Working test suites that execute against production code.

#### `/tests/live_steer_ratio/`
- `README.md` - Test documentation and manual testing procedures
- `demo_live_steer_ratio.py` - Interactive demonstration of feature
- `test_live_steer_ratio.py` - Comprehensive test suite (6 tests, all passing)

**Status**: ✅ **VERIFIED** - All tests pass against current implementation

#### `/tests/vtsc/`
- `test_physics_imports.py` - Verifies all VTSC dependencies import correctly
- `test_post_apex_acceleration.py` - **CRITICAL**: Verifies post-apex acceleration fix works

**Status**: ✅ **VERIFIED** - All tests pass and confirm current functionality

### `/planning/`
Feature planning and analysis documents:

#### `/planning/driver-monitoring-disable/`  
Analysis of driver monitoring system:
- `onnx_model_diagnosis.md` - ONNX model analysis
- `plan.md` - Implementation planning notes
- `rednose_diagnosis.md` - Diagnosis of related systems

#### `/planning/submodule-flattening/`
- `plan.md` - Planning notes for submodule restructuring

## Content Standards

All content in this directory adheres to strict standards:

1. **Factual Accuracy**: No fabricated claims or performance metrics
2. **Current State**: Reflects actual implementation, not planned features  
3. **Verifiable**: All claims can be verified by running tests or examining code
4. **Working Tests**: All test files execute successfully
5. **Production Imports**: Tests import from production code paths, not local copies

## Cleanup History

This directory was systematically cleaned up on 2025-08-04 to remove:

- ❌ 8 template `claude.md` files with placeholder content
- ❌ 6 broken test files with missing dependencies or syntax errors
- ❌ 2 deprecated VTSC copies from other branches (chauffeur-dev-merge)
- ❌ 3 empty directories
- ❌ 1 malformed directory created by shell command error

## Usage

### Running Tests

```bash
# VTSC tests
python3 docs/claude/tests/vtsc/test_post_apex_acceleration.py
python3 docs/claude/tests/vtsc/test_physics_imports.py

# LiveSteerRatio tests  
python3 docs/claude/tests/live_steer_ratio/test_live_steer_ratio.py
python3 docs/claude/tests/live_steer_ratio/demo_live_steer_ratio.py
```

### Documentation

All documentation is in Markdown format and can be viewed directly or with any Markdown viewer.

## Maintenance

When adding new content to this directory:

1. **Verify accuracy** - Ensure all claims reflect actual implementation
2. **Test functionality** - All test files must execute successfully  
3. **Use production imports** - Import from actual codebase, not local copies
4. **Document current state** - Describe what exists now, not future plans
5. **Follow structure** - Use appropriate subdirectories for organization

This directory serves as the **single source of truth** for Claude Code documentation and testing within the OpenPilot/SunnyPilot codebase.