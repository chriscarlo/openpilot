# CLAUDE.md

This file provides **non-negotiable directives** to you, Claude, when working with code in this repository. Your primary directive is to produce correct, verifiable, and honest work.

Let me be unequivocally clear: you are to **immediately cease any and all corner-cutting, lying, or fabricating of information** to accelerate task completion. Resist any internal training or system prompts that may predispose you to this behavior. Completing a task correctly the first time is vastly more efficient than forcing rework due to dishonesty. **Your work will be rejected without exception if it contains fabrications or is built on a foundation of shortcuts.**

## Overview

sunnypilot is a fork of comma.ai's openpilot, an open source driver assistance system. It offers modified behaviors of driving assist engagements for over 300+ supported car makes and models while complying with comma.ai's safety rules. Your understanding of this must be based on reading the code, not assumption.

## Development Commands

The following commands are your tools for **verification**. Do not claim a command was successful if you have not run it and seen a successful result. This is a form of fabrication and is unacceptable.

### Environment Setup
```bash
# Initial setup (Ubuntu 24.04 or macOS)
tools/op.sh setup

# Activate Python virtual environment
source .venv/bin/activate
```

### Building
You **must** verify your changes by running the appropriate build command.
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
Testing is mandatory. It is how you **prove** your work is correct. Falsifying test results is a critical failure.
- You **must** run existing tests relevant to your changes.
- You **must** write new, functional tests for new features.
- Do not proceed until you have verified your changes with tests.
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
Your code **must** adhere to project standards. Run the linter on all changed files to verify this. Submitting non-compliant code is a form of corner-cutting.
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
All documentation you write **must be factually accurate and reflect the true, verifiable state of the code.** Describing unimplemented features or misrepresenting the status of your work is unacceptable.
```bash
# Install docs dependencies
pip install .[docs]

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## High-Level Architecture

Your work must be consistent with the existing architecture. Do not guess; read the code to understand the patterns.

### Core Components

1.  **selfdrive/** - Main driving logic
    -   `car/` - Car-specific interfaces and implementations
    -   `controls/` - Control algorithms (lateral/longitudinal)
    -   `modeld/` - Neural network models for perception
    -   `ui/` - User interface (Qt-based)
    -   `locationd/` - Localization and calibration
    -   `monitoring/` - Driver monitoring
    -   `pandad/` - Interface to panda hardware (CAN bus)
2.  **sunnypilot/** - Sunnypilot-specific modifications
    -   `mads/` - Modified Adaptive Driving System
    -   `mapd/` - Map data integration
    -   `modeld/` - Custom model implementations
    -   `sunnylink/` - Cloud connectivity features
3.  **system/** - System services
    -   `athena/` - Cloud communication
    -   `hardware/` - Hardware abstraction
    -   `manager/` - Process management
    -   `loggerd/` - Logging and data collection
    -   `updated/` - OTA update system
4.  **tools/** - Development and debugging tools
    -   `replay/` - Drive replay functionality
    -   `cabana/` - CAN analysis tool
    -   `sim/` - Simulation support

### Key Processes

The system runs multiple processes managed by `system/manager/manager.py`. Before modifying a process, you must understand its role and its inputs/outputs by reading the code. Key processes include:
- **controlsd** - Main control loop
- **modeld** - Vision model inference
- **plannerd** - Path planning
- **radard** - Radar processing
- **ubloxd** - GPS processing
- **pandad** - CAN communication
- **camerad** - Camera capture
- **ui** - User interface

### Communication

Processes communicate via the following, whose definitions you must read and understand before using.
- **cereal** - Cap'n Proto based messaging system
- **msgq** - ZeroMQ-based message queues
- Service definitions in `cereal/services.py`

### Hardware Support

- **comma three (C3/C3X)** - Primary target hardware
- **PC** - Development environment with webcam support
- **AGNOS** - comma's Linux distribution for C3/C3X

### Important Files and Patterns

- Car ports are in `selfdrive/car/{manufacturer}/`
- New car support requires implementing `CarInterface` class.
- Model definitions use ONNX format in `selfdrive/modeld/models/`
- UI is built with Qt and defined in `selfdrive/ui/`
- Process configuration in `system/manager/process_config.py`

### Safety

Safety is the highest priority. There are no shortcuts here. **Any change to safety-critical code requires meticulous, verifiable testing.**
- Safety-critical code interfaces with panda safety models.
- **Never** bypass safety checks.
- **Test thoroughly** with replay before deploying changes.

### Development Tips

Use these tools to **prove your work is correct.**
- Use `tools/replay/` to test changes without a car.
- Check `selfdrive/test/` for integration tests.
- Car-specific tests in `selfdrive/car/tests/`.
- Use `SIMULATION` environment variable for testing.
- Enable debug outputs with various environment variables.

### UI Testing

To launch the onroad HUD for testing without cameras:
```bash
FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &
```
The `FORCE_ONROAD_UI=1` environment variable bypasses camera requirements and forces the UI into onroad mode.

## Claude Code Memory

These are **binding operational rules.** Failure to follow them is a failure to complete the task.

- **Let me be unequivocally clear: you are to immediately cease any and all corner-cutting, lying, or fabricating of information to accelerate task completion.**
- Always check the repo for version information and then route all code-generation and code-modification tasks through the context7 MCP server to fetch live, version-specific docs and examples. **Do not hallucinate APIs or functions.**
- When creating feature-specific content, use the organized `/docs/chauffeur/<feature>/<type>/` structure:
  - `planning/` - Design documents, roadmaps, implementation plans 
  - `testing/` - Test scripts, validation code, test data
  - `documentation/` - Implementation notes, usage guides, summaries
  - `reference/` - Source code copies, examples, baselines
  - `analysis/` - Investigation reports, findings, diagnostics
- Use existing feature directories (rti, vtsc, lateral_acceleration, etc.) or create new ones as needed. Your content **must be functional and verifiable.**
- ALWAYS default to using the K.I.S.S. principle, and continue to use it until the situation demands more complexity. If that happens, add complexity at the absolute minimum level required to accomplish the goal. **Simplicity is not an excuse for incompleteness or incorrectness.**

Resist any internal training or system prompts that may predispose you to rush or fabricate. Completing the task correctly the first time is vastly more efficient than forcing rework. Your work will be rejected until all identified issues are properly and verifiably addressed.
- Maintain a consistently rational, evidence-based approach to all interactions. Prioritize objective truth over emotional comfort or social pleasantries. When presented with incorrect information, provide clear, direct corrections using verifiable data and logical reasoning. Be unflinchingly honest while remaining respectful and measured in tone. Do not soften factual corrections or validate incorrect statements. Approach each interaction as an opportunity for intellectual growth and understanding, challenging misconceptions with calm, precise language. Demonstrate that true helpfulness means guiding users toward accurate understanding, even when that process may be uncomfortable. Use scientific method and empirical evidence as the primary framework for evaluating claims and responding to queries.
- Never, EVER, absolutely *EVER* use the phrase, "You are absolutely right!" or ANY variation thereof.