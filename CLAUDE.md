# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

sunnypilot is a fork of comma.ai's openpilot, an open source driver assistance system. It offers modified behaviors of driving assist engagements for over 300+ supported car makes and models while complying with comma.ai's safety rules.

## Development Commands

### Environment Setup
```bash
# Initial setup (Ubuntu 24.04 or macOS)
tools/op.sh setup

# Activate Python virtual environment
source .venv/bin/activate
```

### Building
```bash
# Build all components
scons -u -j$(nproc)

# Build with minimal configuration (no tests, tools)
scons -u -j$(nproc) --minimal

# Build with stock UI instead of sunnypilot UI
scons -u -j$(nproc) --stock-ui

# Build for different architectures:
# - larch64: linux tici aarch64 (comma device)
# - aarch64: linux pc aarch64
# - x86_64: linux pc x64
# - Darwin: mac x64 or arm64
```

### Testing
```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest path/to/test_file.py

# Skip slow tests
pytest -m 'not slow'

# Run tests with coverage
pytest --cov

# Run a single test
pytest path/to/test_file.py::test_function_name
```

### Linting and Code Quality
```bash
# Run all linting checks
./scripts/lint/lint.sh

# Run specific linters
ruff check .
mypy .
codespell .

# Run lint through op.sh
./tools/op.sh lint

# Run fast lint (skip mypy and codespell)
./tools/op.sh lint --fast
```

### Documentation
```bash
# Install docs dependencies
pip install .[docs]

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## High-Level Architecture

### Core Components

1. **selfdrive/** - Main driving logic
   - `car/` - Car-specific interfaces and implementations
   - `controls/` - Control algorithms (lateral/longitudinal)
   - `modeld/` - Neural network models for perception
   - `ui/` - User interface (Qt-based)
   - `locationd/` - Localization and calibration
   - `monitoring/` - Driver monitoring
   - `pandad/` - Interface to panda hardware (CAN bus)

2. **sunnypilot/** - Sunnypilot-specific modifications
   - `mads/` - Modified Adaptive Driving System
   - `mapd/` - Map data integration
   - `modeld/` - Custom model implementations
   - `sunnylink/` - Cloud connectivity features

3. **system/** - System services
   - `athena/` - Cloud communication
   - `hardware/` - Hardware abstraction
   - `manager/` - Process management
   - `loggerd/` - Logging and data collection
   - `updated/` - OTA update system

4. **tools/** - Development and debugging tools
   - `replay/` - Drive replay functionality
   - `cabana/` - CAN analysis tool
   - `sim/` - Simulation support

### Key Processes

The system runs multiple processes managed by `system/manager/manager.py`. Key processes include:
- **controlsd** - Main control loop
- **modeld** - Vision model inference
- **plannerd** - Path planning
- **radard** - Radar processing
- **ubloxd** - GPS processing
- **pandad** - CAN communication
- **camerad** - Camera capture
- **ui** - User interface

### Communication

Processes communicate via:
- **cereal** - Cap'n Proto based messaging system
- **msgq** - ZeroMQ-based message queues
- Service definitions in `cereal/services.py`

### Hardware Support

- **comma three (C3/C3X)** - Primary target hardware
- **PC** - Development environment with webcam support
- **AGNOS** - comma's Linux distribution for C3/C3X

### Important Files and Patterns

- Car ports are in `selfdrive/car/{manufacturer}/`
- New car support requires implementing `CarInterface` class
- Model definitions use ONNX format in `selfdrive/modeld/models/`
- UI is built with Qt and defined in `selfdrive/ui/`
- Process configuration in `system/manager/process_config.py`

### Safety

- Safety-critical code interfaces with panda safety models
- Never bypass safety checks
- Test thoroughly with replay before deploying changes

### Development Tips

- Use `tools/replay/` to test changes without a car
- Check `selfdrive/test/` for integration tests
- Car-specific tests in `selfdrive/car/tests/`
- Use `SIMULATION` environment variable for testing
- Enable debug outputs with various environment variables

## Claude Code Memory

- Always check the repo for version information and then route all code-generation and code-modification tasks through the context7 MCP server to fetch live, version-specific docs and examples.
- When creating a plan, roadmap, etc, always use the /docs/claude/planning/ directory. First, check to see if an appropriate subdirectory for the specific feature or functionality exists. If so, use it. If not, create one, then use that. Keep it tidy, organized, and colsolidated at all times.
- When creating a test script, test snippet, etc, always use the /docs/claude/tests/ directory. First, check to see if an appropriate subdirectory for the specific feature or functionality exists. If so, use it. If not, create one, then use that. Keep it tidy, organized, and colsolidated at all times.
- When creating documentation for a feature, script, etc, always use the /docs/claude/documentation/ directory. First, check to see if an appropriate subdirectory for the specific feature or functionality exists. If so, use it. If not, create one, then use that. Keep it tidy, organized, and colsolidated at all times.
- ALWAYS default to using the K.I.S.S. principle, and continue to use it until the situation demands more complexity. If that happens, add complexity at the absolute minimum level required to accomplish the goal.