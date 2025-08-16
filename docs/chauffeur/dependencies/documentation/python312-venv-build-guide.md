# Python 3.12 Virtual Environment Build Guide

## Overview

This document provides an exhaustive guide for building the OpenPilot codebase using Python 3.12 virtual environment. This became necessary after encountering Python module compatibility issues, specifically with Cython-compiled modules that showed errors like `ImportError: undefined symbol: PyType_FromMetaclass`.

## Critical Finding: Python Version Compatibility

**IMPORTANT**: The entire codebase MUST be built using Python 3.12 when using Cython modules. Mixing Python versions between build time and runtime causes binary incompatibility errors.

## Creating the Python 3.12 Virtual Environment

```bash
# Navigate to the project root
cd /data/openpilot

# Create Python 3.12 virtual environment
python3.12 -m venv .venv312

# Activate the virtual environment
source .venv312/bin/activate

# Verify Python version
python --version  # Should show Python 3.12.x
```

## Complete Dependency Installation List

The following is an EXHAUSTIVE list of all dependencies that must be installed in the Python 3.12 virtual environment for a successful build:

### Base Requirements Installation

```bash
# Activate the virtual environment first
source .venv312/bin/activate

# Install base requirements from the repository
pip install -r requirements.txt
```

### Additional Required Dependencies

During the build process, the following additional dependencies were required and must be installed:

```bash
# Core build dependencies
pip install cython              # Required for compiling .pyx files
pip install numpy               # Required for numerical computations
pip install scipy               # Required for scientific computing
pip install cffi                # Required for C Foreign Function Interface
pip install scons               # Required for the build system (if not already installed)
pip install pre-commit          # Required for git hooks
pip install pytest              # Required for testing
pip install pytest-xdist        # Required for parallel testing

# Cryptography
pip install pycryptodome        # Required for Crypto module (used by various components)

# Control systems
pip install casadi              # Required for Model Predictive Control (MPC) in lateral/longitudinal control

# Serialization
pip install pycapnp             # Required for Cap'n Proto serialization (cereal)

# Hardware interfaces
pip install smbus2              # Required for I2C communication (amplifier control)

# Machine learning
pip install onnx                # Required for neural network model format
pip install future-fstrings     # Required for f-string compatibility in generated code

# System utilities
pip install setproctitle        # Required for process title modification
```

### Complete Installation Command

For convenience, here's the complete installation command:

```bash
source .venv312/bin/activate
pip install cython numpy scipy cffi scons pre-commit pytest pytest-xdist \
            pycryptodome casadi pycapnp smbus2 onnx future-fstrings setproctitle
```

## Build Process with Python 3.12

### Environment Setup

**CRITICAL**: Always set PYTHONPATH when building:

```bash
source .venv312/bin/activate
export PYTHONPATH=/data/openpilot:$PYTHONPATH
```

### Which Components Require the Virtual Environment?

**Answer: The ENTIRE codebase must be built using the Python 3.12 virtual environment.**

This includes but is not limited to:
- **pandad** - CAN interface daemon
- **controlsd** - Main control loop (lateral/longitudinal MPC)
- **modeld** - Vision model inference
- **locationd** - Localization and calibration
- **ui** - User interface components
- **All Cython modules** (.pyx files)
- **All Python-based build scripts**

### Build Commands

```bash
# Full build
source .venv312/bin/activate
export PYTHONPATH=/data/openpilot:$PYTHONPATH
scons -u -j$(nproc)

# Minimal build (no tests, tools)
scons -u -j$(nproc) --minimal

# Build specific component
scons -u -j$(nproc) selfdrive/pandad/pandad

# Clean and rebuild
scons -c
scons -u -j$(nproc)
```

## Troubleshooting Common Issues

### 1. ImportError: undefined symbol: PyType_FromMetaclass

**Cause**: Python version mismatch between build time and runtime
**Solution**: Ensure you're using Python 3.12 venv for both building and running

### 2. ModuleNotFoundError during build

**Cause**: Missing Python dependency
**Solution**: Install the missing module using pip within the activated venv

### 3. Cython compilation errors

**Cause**: Outdated Cython or missing numpy headers
**Solution**: 
```bash
pip install --upgrade cython numpy
```

### 4. Build timeout

**Cause**: Large codebase with many components
**Solution**: Build specific components or use --minimal flag

## Verified Build Components

The following components have been successfully built using Python 3.12:

1. **pandad** - ✓ Built successfully
2. **Model files** - ✓ 3 neural network models compiled
   - dmonitoring_model_tinygrad.pkl
   - driving_policy_metadata.pkl
   - driving_vision_metadata.pkl
3. **MPC libraries** - ✓ 4 Model Predictive Control libraries built
   - lateral_mpc_lib (acados solver)
   - longitudinal_mpc_lib (acados solver)
4. **UI components** - ✓ Python helpers and assets built
5. **Cython modules** - ✓ All .pyx files compiled successfully
   - common/params_pyx.pyx
   - rednose/helpers/ekf_sym_pyx.pyx
   - selfdrive/pandad/pandad_api_impl.pyx
   - selfdrive/modeld/models/commonmodel_pyx.pyx
   - common/transformations/transformations.pyx

## Key Insights

1. **Python Version Consistency**: The most critical finding is that ALL Python-related build operations must use the same Python version (3.12) to avoid binary incompatibility.

2. **Dependency Chain**: Many dependencies pull in their own requirements, so the actual installed package count will be higher than the explicit list.

3. **Build Order**: Some components depend on others (e.g., MPC libraries must be built before controlsd), but SCons handles this automatically.

4. **Performance**: Using `-j$(nproc)` for parallel builds significantly reduces build time.

## Maintenance Notes

- Keep this virtual environment separate from the system Python
- Regularly update dependencies within the venv: `pip install --upgrade -r requirements.txt`
- If switching Python versions, create a new venv and rebuild everything
- The `.venv312/` directory should be added to .gitignore

## References

- Original error that led to this solution: Cython module import errors with system Python
- Python 3.12 was chosen as it's within the supported range (>=3.11, <3.13) per pyproject.toml
- This guide supplements the existing python-dependencies.md with build-specific information