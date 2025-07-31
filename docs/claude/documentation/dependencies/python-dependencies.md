# Python Dependencies

Complete listing of all Python packages used in the OpenPilot codebase, organized by source and purpose.

## Main Project Dependencies

From `/data/openpilot/pyproject.toml`

### Core Runtime Dependencies

**Multi-user packages** (lines 12-18):
- `sounddevice` - Audio I/O for micd + soundd processes
- `pyserial` - Serial communication for pigeond + qcomgpsd
- `requests` - HTTP client library for various one-off uses  
- `sympy` - Symbolic mathematics for rednose + mathematical computations
- `crcmod` - CRC calculations for cars + qcomgpsd
- `tqdm` - Progress bars for cars (fw_versions.py) and other tools

**Hardware interfaces** (line 21):
- `smbus2` - I2C bus interface for configuring amplifier

**Core build/runtime** (lines 24-29):
- `cffi` - C Foreign Function Interface for Python-C integration
- `scons` - Build system used throughout the project
- `pycapnp` - Cap'n Proto serialization library  
- `Cython` - Python-to-C compiler for performance-critical code
- `setuptools` - Python packaging tools
- `numpy >=2.0, <2.2` - Numerical computing (linting issues with mypy in 2.2)

**WebRTC/Body communication** (lines 32-37):
- `aiohttp` - Async HTTP client/server for web interfaces
- `aiortc` - WebRTC implementation for remote control
- `pyopenssl < 24.3.0` - OpenSSL bindings (version pinned for aiortc compatibility)
- `pyaudio` - Audio I/O for voice communication

**Panda hardware interface** (lines 40-41):
- `libusb1` - USB communication with panda device
- `spidev; platform_system == 'Linux'` - SPI interface (Linux only)

**Machine learning/models** (lines 44-45):
- `onnx >= 1.14.0` - Open Neural Network Exchange format
- `llvmlite` - LLVM Python bindings for code generation

**Logging and monitoring** (lines 48-50):
- `pyzmq` - ZeroMQ messaging library
- `sentry-sdk` - Error tracking and monitoring
- `xattr` - Extended file attributes (macOS compatibility)

**Cloud connectivity (Athena)** (lines 53-55):
- `PyJWT` - JSON Web Token handling for authentication
- `json-rpc` - JSON-RPC protocol implementation
- `websocket_client` - WebSocket client for real-time communication

**Optimal control (Acados)** (lines 58-59):
- `casadi >=3.6.6` - Symbolic framework for nonlinear optimization
- `future-fstrings` - Python 2/3 compatibility for f-strings

**Input devices** (line 62):
- `inputs` - Cross-platform joystick/gamepad support

**System utilities** (lines 65-67):
- `psutil` - System and process utilities
- `pycryptodome` - Cryptographic library for updated/casync, panda, body
- `setproctitle` - Process title modification

**Data handling** (lines 70-73):
- `zstandard` - Zstandard compression for log files
- `qrcode` - QR code generation for pairing

## Optional Dependencies

### Documentation (lines 77-81)
- `Jinja2` - Template engine for documentation generation
- `natsort` - Natural sorting for documentation
- `mkdocs` - Static site generator for documentation

### Testing (lines 83-99)
- `hypothesis ==6.47.*` - Property-based testing framework
- `mypy` - Static type checker
- `pytest` - Testing framework
- `pytest-cpp` - C++ testing support
- `pytest-subtests` - Subtest support for pytest
- `pytest-xdist @ git+...` - Parallel test execution (custom fork)
- `pytest-timeout` - Test timeout functionality
- `pytest-randomly` - Randomized test execution
- `pytest-asyncio` - Async test support
- `pytest-mock` - Mocking utilities
- `pytest-repeat` - Test repetition functionality
- `ruff` - Fast Python linter and formatter
- `codespell` - Spell checker for code
- `pre-commit-hooks` - Git pre-commit hook utilities

### Development (lines 101-120)
- `av` - Video/audio processing library
- `azure-identity` - Azure authentication
- `azure-storage-blob` - Azure blob storage client
- `dbus-next` - D-Bus client library
- `dictdiffer` - Dictionary comparison utilities
- `matplotlib` - Plotting and visualization
- `opencv-python-headless` - Computer vision library (headless)
- `parameterized >=0.8, <0.9` - Parameterized testing
- `pyautogui` - GUI automation
- `pygame` - Game development library for simulations
- `pyopencl; platform_machine != 'aarch64'` - OpenCL Python bindings (not on ARM64)
- `pytools < 2024.1.11; platform_machine != 'aarch64'` - PyOpenCL utilities (ARM64 incompatible version)
- `pywinctl` - Window control utilities
- `pyprof2calltree` - Profiling output conversion
- `tabulate` - Pretty-print tabular data
- `types-requests` - Type stubs for requests
- `types-tabulate` - Type stubs for tabulate
- `raylib` - Graphics library for UI development

### Tools (lines 122-125)
- `metadrive-simulator @ https://...` - Driving simulation environment (not on ARM64)
- `rerun-sdk >= 0.18` - Data visualization and debugging

## Submodule Dependencies

### Rednose (Kalman Filter Library)

From `/data/openpilot/rednose_repo/requirements.txt`:
- `ruff` - Code formatting and linting
- `sympy` - Symbolic mathematics for filter equations
- `numpy` - Numerical computations
- `scipy` - Scientific computing algorithms
- `cffi` - C integration for performance
- `scons` - Build system
- `pre-commit` - Git hooks
- `Cython` - Python-to-C compilation
- `pytest` - Testing framework
- `pytest-xdist` - Parallel testing

From `/data/openpilot/rednose_repo/setup.py`:
- `numpy` - Core numerical library
- `cffi` - C Foreign Function Interface
- `sympy` - Symbolic computation

### OpenDBC (CAN Database Library)

From `/data/openpilot/opendbc_repo/pyproject.toml`:

**Core dependencies** (lines 14-23):
- `scons` - Build system
- `numpy` - Numerical operations
- `Cython` - Performance optimization
- `crcmod` - CRC calculations for CAN messages
- `tqdm` - Progress indicators
- `pycapnp` - Cap'n Proto serialization
- `setuptools` - Package management
- `pycryptodome` - Cryptographic operations

**Testing dependencies** (lines 26-45):
- `cffi` - C integration testing
- `gcovr` - Code coverage reporting
- `pytest` - Test framework
- `pytest-coverage` - Coverage integration
- `pytest-mock` - Mocking utilities  
- `pytest-randomly` - Randomized testing
- `pytest-xdist` - Parallel execution
- `pytest-subtests` - Subtest support
- `hypothesis==6.47.*` - Property-based testing
- `parameterized>=0.8,<0.9` - Parameterized tests
- `ruff` - Linting and formatting
- `ty` - Type checking utilities
- `lefthook` - Git hooks management
- `cpplint` - C++ code linting
- `cython-lint` - Cython code linting
- `codespell` - Spell checking

### Panda (Hardware Interface)

From `/data/openpilot/panda/pyproject.toml`:

**Core dependencies** (lines 14-17):
- `libusb1` - USB communication
- `opendbc @ git+...` - CAN database (specific commit)

**Development dependencies** (lines 20-34):
- `scons` - Build system
- `pycryptodome >= 3.9.8` - Cryptographic operations
- `cffi` - C integration
- `flaky` - Flaky test handling
- `pytest` - Testing framework
- `pytest-mock` - Mocking support
- `pytest-xdist` - Parallel testing
- `pytest-timeout` - Test timeouts
- `pytest-randomly` - Random test ordering
- `ruff` - Code linting
- `mypy` - Type checking
- `setuptools` - Package tools
- `spidev; platform_system == 'Linux'` - SPI interface (Linux only)

### TeleopRTC (WebRTC Library)

From `/data/openpilot/teleoprtc_repo/pyproject.toml`:

**Core dependencies** (lines 18-23):
- `aiortc>=1.6.0` - WebRTC implementation
- `aiohttp>=3.7.0` - Async HTTP framework
- `av>=11.0.0,<13.0.0` - Audio/video processing
- `numpy>=1.19.0` - Numerical operations

**Development dependencies** (lines 25-32):
- `parameterized>=0.8` - Parameterized testing
- `pre-commit` - Git hooks
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support
- `pytest-xdist` - Parallel test execution

### TinyGrad (Machine Learning Framework)

From `/data/openpilot/tinygrad_repo/setup.py`:

**No required dependencies** (line 36):
- `install_requires=[]` - Designed to be dependency-free

**Testing minimal** (lines 10-18):
- `numpy` - Numerical computations
- `torch` - PyTorch for compatibility testing
- `pytest` - Test framework
- `pytest-xdist` - Parallel testing
- `hypothesis` - Property-based testing
- `z3-solver` - SMT solver for symbolic execution
- `ml_dtypes` - Machine learning data types

**Full testing suite** (lines 56-79):
- `pillow` - Image processing
- `onnx==1.18.0` - Neural network format
- `onnx2torch` - ONNX to PyTorch conversion
- `onnxruntime` - ONNX runtime execution
- `opencv-python` - Computer vision
- `tabulate` - Table formatting
- `tqdm` - Progress bars
- `safetensors` - Safe tensor storage
- `transformers` - Transformer models
- `sentencepiece` - Text tokenization
- `tiktoken` - OpenAI tokenizer
- `blobfile` - File operations
- `librosa` - Audio processing
- `networkx` - Graph algorithms
- `nibabel` - Neuroimaging file formats
- `bottle` - Lightweight web framework
- `ggml-python` - GGML format support
- `capstone` - Disassembly framework
- `pycocotools` - COCO dataset utilities
- `boto3` - AWS SDK
- `pandas` - Data analysis
- `influxdb3-python` - Time series database

## Python Version Requirements

- **Main project**: `>= 3.11, < 3.13` (most restrictive)
- **OpenDBC**: `>=3.9,<3.13` (pycapnp doesn't work with 3.13)
- **Panda**: `>=3.11,<3.13`
- **TeleopRTC**: `>=3.8`
- **TinyGrad**: `>=3.10`

## Package Management

The project uses **UV** package manager for faster dependency resolution:
- UV installation and updates in `/data/openpilot/tools/install_python_dependencies.sh`
- Lockfile: `/data/openpilot/uv.lock`
- Configuration: `/data/openpilot/pyproject.toml`