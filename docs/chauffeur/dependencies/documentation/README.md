# OpenPilot Dependencies Documentation

This directory contains comprehensive documentation of all dependencies in the OpenPilot codebase, based on exhaustive analysis of dependency files.

## Overview

The OpenPilot project has a complex dependency structure with over 150+ individual dependencies across multiple categories:

- **100+ Python packages** across core runtime, testing, development, and tools
- **50+ system packages** for Ubuntu/macOS/Docker environments  
- **15+ third-party C++ libraries** integrated into the build system
- **Hardware-specific drivers** for CAN bus, cameras, sensors, and neural processing

## Documentation Files

### [Python Dependencies](python-dependencies.md)
Complete listing of Python packages from:
- Main project (`pyproject.toml`)
- All submodules (panda, opendbc, rednose, tinygrad, teleoprtc, msgq)
- Optional dependencies for testing, development, docs, and tools

### [System Dependencies](system-dependencies.md)  
System-level packages and tools for:
- Ubuntu packages (apt-get)
- macOS packages (Homebrew)
- Docker container dependencies
- Compiler toolchains and build tools

### [Build & Third-party Dependencies](build-dependencies.md)
Build system and external libraries:
- SCons build configuration
- Third-party C++ libraries
- Hardware acceleration libraries (OpenCL, SNPE)
- Qt5 GUI framework components

### [Hardware Dependencies](hardware-dependencies.md)
Hardware-specific drivers and libraries:
- USB and CAN bus interfaces
- Camera and sensor drivers  
- Neural processing engines
- OpenCL compute libraries

## Analysis Methodology

This documentation is based on **direct examination** of dependency files with specific file paths and line numbers as evidence. No assumptions or educated guesses were made - only dependencies explicitly declared in configuration files are included.

### Files Analyzed

**Python Dependencies:**
- `/data/openpilot/pyproject.toml` - Main project dependencies
- `/data/openpilot/*/pyproject.toml` - All submodule dependencies
- `/data/openpilot/*/setup.py` - Legacy setup files
- `/data/openpilot/*/requirements.txt` - Direct requirements

**System Dependencies:**
- `/data/openpilot/tools/install_ubuntu_dependencies.sh` 
- `/data/openpilot/tools/mac_setup.sh`
- `/data/openpilot/Dockerfile.*` files
- `/data/openpilot/.github/workflows/*.yaml` - CI/CD dependencies

**Build Dependencies:**
- `/data/openpilot/SConstruct` - Main build configuration
- `/data/openpilot/*/SConscript` - Component build files
- Third-party library directories

## Quick Reference

### Critical Runtime Dependencies
- Python 3.11-3.12
- NumPy >= 2.0
- Qt5 GUI framework
- OpenCL for GPU compute
- ONNX for neural networks
- ZeroMQ for messaging

### Development Requirements
- SCons build system
- Clang/GCC compilers
- Git with LFS support
- Docker (optional)
- Pytest for testing

### Hardware Support
- USB libraries for panda interface
- CAN bus drivers
- Camera/video libraries (FFmpeg)
- Neural processing (SNPE/OpenCL)

## Last Updated
Generated from comprehensive dependency analysis - January 2025