# Build & Third-party Dependencies

Build system configuration and external libraries integrated into the OpenPilot build process.

## Build System

### SCons Configuration

From `/data/openpilot/SConstruct`

**Primary build tool:**
- **SCons** - Software construction tool (Python-based make replacement)
- Configuration file: `SConstruct` (main), `SConscript` (components)
- Parallel builds: `SetOption('num_jobs', int(os.cpu_count()/2))` (line 20)

**Build options:**
- `--kaitai` - Regenerate kaitai struct parsers
- `--asan` - Enable AddressSanitizer  
- `--ubsan` - Enable UndefinedBehaviorSanitizer
- `--coverage` - Enable test coverage options
- `--clazy` - Enable Qt code analysis
- `--ccflags` - Pass arbitrary compiler flags
- `--mutation` - Generate mutation-ready code
- `--minimal` - Minimal build (no tests, tools)
- `--stock-ui` - Build stock OpenPilot UI instead of sunnypilot UI

### Compilers and Toolchains

**Primary compilers** (lines 206-207):
- `clang` - C compiler (LLVM-based)
- `clang++` - C++ compiler (LLVM-based)

**Cross-compilation support:**
- `gcc-arm-none-eabi` - ARM bare-metal cross-compiler
- Architecture support: larch64, aarch64, x86_64, Darwin

**Language standards:**
- C: `std=gnu11` (line 212)
- C++: `std=c++1z` (line 213) 

### Compiler Flags

**Warning flags** (lines 178-190):
- `-g` - Debug information
- `-fPIC` - Position independent code
- `-O2` - Optimization level 2
- `-Wunused` - Warn on unused variables
- `-Werror` - Treat warnings as errors
- `-Wshadow` - Warn on variable shadowing
- `-Wno-unknown-warning-option` - Ignore unknown warnings
- `-Wno-inconsistent-missing-override` - Disable override warnings
- `-Wno-c99-designator` - Disable C99 designator warnings
- `-Wno-reorder-init-list` - Disable reorder warnings
- `-Wno-vla-cxx-extension` - Disable VLA warnings

**Platform-specific flags:**
- **AGNOS/QCOM2**: `-DQCOM2 -mcpu=cortex-a57` (lines 111-112)
- **macOS**: `-DGL_SILENCE_DEPRECATION` (lines 130-131)
- **sunnypilot**: `-DSUNNYPILOT` (lines 169-170) when not using `--stock-ui`

**Sanitizer support:**
- **AddressSanitizer**: `-fsanitize=address -fno-omit-frame-pointer`
- **UndefinedBehaviorSanitizer**: `-fsanitize=undefined`

### Linker Configuration

**Linker flags** (lines 165-166):
- `-Wl,--as-needed` - Only link needed libraries (not macOS)
- `-Wl,--no-undefined` - Reject undefined symbols (not macOS)

**RPATH configuration:**
- Runtime library paths for each architecture
- macOS uses `-Wl,-rpath,{path}` linker flags instead

## Third-party Libraries

### Core Libraries

From `/data/openpilot/third_party/SConscript`:

**JSON processing:**
- `json11` - Lightweight JSON library (line 3)
- Source: `json11/json11.cpp`
- Flags: `-Wno-unqualified-std-cast-call`

**Binary parsing:**
- `kaitai` - Binary data parsing library (line 4)  
- Source: `kaitaistream.cpp`
- Flags: `-DKS_STR_ENCODING_NONE`

### Include Paths

From `/data/openpilot/SConstruct` (lines 194-203):

**Third-party includes:**
- `#third_party/acados/include` - Optimal control library
- `#third_party/acados/include/blasfeo/include` - Linear algebra
- `#third_party/acados/include/hpipm/include` - Interior point methods  
- `#third_party/catch2/include` - C++ testing framework
- `#third_party/libyuv/include` - YUV image processing
- `#third_party/json11` - JSON library
- `#third_party/linux/include` - Linux kernel headers
- `#third_party/snpe/include` - Snapdragon Neural Processing Engine
- `#third_party` - General third-party root

### Library Paths

**Architecture-specific libraries:**

**AGNOS/larch64** (lines 100-113):
- `/usr/local/lib` - Local libraries
- `/system/vendor/lib64` - Vendor libraries
- `#third_party/acados/larch64/lib` - Acados ARM64
- `#third_party/snpe/larch64` - SNPE ARM64
- `#third_party/libyuv/larch64/lib` - libyuv ARM64
- `/usr/lib/aarch64-linux-gnu` - System ARM64 libraries

**macOS/Darwin** (lines 122-128):
- `#third_party/libyuv/Darwin/lib` - libyuv macOS
- `#third_party/acados/Darwin/lib` - Acados macOS
- `{brew_prefix}/lib` - Homebrew libraries
- `{brew_prefix}/opt/openssl@3.0/lib` - OpenSSL
- `/System/Library/Frameworks/OpenGL.framework/Libraries` - OpenGL

**Linux x86_64** (lines 139-152):
- `#third_party/acados/x86_64/lib` - Acados x86_64
- `#third_party/libyuv/x86_64/lib` - libyuv x86_64
- `#third_party/snpe/x86_64` - SNPE x86_64
- `/usr/lib` - System libraries
- `/usr/local/lib` - Local libraries

## Acados (Optimal Control)

### Configuration

**Environment variables** (lines 83-91):
```bash
ACADOS_SOURCE_DIR=#third_party/acados
ACADOS_PYTHON_INTERFACE_PATH=#third_party/acados/acados_template  
TERA_PATH=#/third_party/acados/{arch}/t_renderer
```

**Library paths:**
- `LD_LIBRARY_PATH=#third_party/acados/{arch}/lib`
- `PYTHONPATH=#third_party/acados`

**Architecture support:**
- ARM64 (larch64) for comma three device
- x86_64 for development machines  
- Darwin (macOS) for development

## Qt5 GUI Framework

### Qt Configuration

From `/data/openpilot/SConstruct` (lines 267-315):

**Qt modules used:**
- `Widgets` - UI widgets
- `Gui` - GUI foundation  
- `Core` - Core functionality
- `Network` - Network operations
- `Concurrent` - Threading utilities
- `DBus` - D-Bus integration (Linux)
- `Xml` - XML processing

**Platform-specific Qt setup:**

**macOS** (lines 272-280):
- Qt directory: `{brew_prefix}/opt/qt@5`
- Frameworks: `-F{qtdir}/lib`
- Libraries: `QtWidgets`, `QtGui`, `QtCore`, etc. as frameworks
- Additional: `OpenGL` framework

**Linux** (lines 282-300):
- Qt directory: From `qmake -query QT_INSTALL_PREFIX`
- Headers: From `qmake -query QT_INSTALL_HEADERS`  
- Libraries: `Qt5Widgets`, `Qt5Gui`, `Qt5Core`, etc.
- **AGNOS**: Additional `GLESv2`, `wayland-client`
- **PC Linux**: Additional `GL` (OpenGL)

**Qt compiler flags** (lines 305-312):
- `-D_REENTRANT` - Thread safety
- `-DQT_NO_DEBUG` - Release mode
- `-DQT_WIDGETS_LIB` - Widgets library
- `-DQT_GUI_LIB` - GUI library  
- `-DQT_CORE_LIB` - Core library
- `-DQT_MESSAGELOGCONTEXT` - Message logging

### Code Analysis

**Clazy support** (lines 317-326):
When `--clazy` option used:
- Compiler: `clazy` (Qt-aware static analyzer)
- Checks: `level0`, `level1`, `no-range-loop`, `no-non-pod-global-static`
- Ignore directories: Qt system headers

## Cython Build Environment

### Configuration

From `/data/openpilot/SConstruct` (lines 252-265):

**Python integration:**
- Include paths: Python headers + NumPy headers
- Compiler flags: `-Wno-#warnings -Wno-shadow -Wno-deprecated-declarations`
- Error handling: Removes `-Werror` flag for Cython code

**Platform-specific linking:**
- **macOS**: `-bundle -undefined dynamic_lookup` + rpath flags
- **Linux**: `-pthread -shared`

**NumPy version tracking:**
- `np_version = SCons.Script.Value(np.__version__)` - Build dependency on NumPy version

## Python Build Environment

### Critical: Python 3.12 Requirement

**IMPORTANT**: The entire codebase must be built using Python 3.12 to avoid binary compatibility issues with Cython modules.

See the comprehensive guide: [Python 3.12 Virtual Environment Build Guide](./python312-venv-build-guide.md)

Key points:
- All Cython modules (.pyx files) must be compiled with Python 3.12
- Runtime Python version must match build-time Python version
- A virtual environment approach is strongly recommended
- All Python dependencies must be installed within the Python 3.12 venv

## Cache Configuration

### SCons Cache

From `/data/openpilot/SConstruct` (lines 235-239):

**Cache directories:**
- **AGNOS**: `/data/scons_cache` (on-device storage)
- **Development**: `/tmp/scons_cache` (temporary storage)
- **Override**: `cache_dir` argument support
- **Cleanup**: Cache cleaned with build artifacts

**Cache usage:**
- Shared across builds for faster compilation
- Keyed by file content hashes
- Used in CI/CD for build acceleration

## Component Build Files

### SConscript Locations

**Core components:**
- `cereal/SConscript` - Serialization library
- `common/SConscript` - Common utilities  
- `selfdrive/SConscript` - Main driving logic
- `sunnypilot/SConscript` - sunnypilot modifications

**System services:**
- `system/ubloxd/SConscript` - GPS processing
- `system/loggerd/SConscript` - Logging service
- `system/logcatd/SConscript` - Android log capture (not Darwin)
- `system/proclogd/SConscript` - Process logging (not Darwin)
- `system/camerad/SConscript` - Camera service (larch64 only)

**Control libraries:**
- `selfdrive/controls/lib/lateral_mpc_lib/SConscript` - Lateral MPC
- `selfdrive/controls/lib/longitudinal_mpc_lib/SConscript` - Longitudinal MPC
- `selfdrive/locationd/SConscript` - Localization services
- `selfdrive/modeld/SConscript` - Neural network models

**Hardware interfaces:**
- `selfdrive/pandad/SConscript` - Panda hardware interface
- `selfdrive/ui/SConscript` - User interface

**Tools (when extras enabled):**
- `tools/replay/SConscript` - Drive replay functionality
- `tools/cabana/SConscript` - CAN analysis tool (not larch64)

**Submodules:**
- `msgq_repo/SConscript` - Message queue library
- `opendbc_repo/SConscript` - CAN database library  
- `panda/SConscript` - Panda hardware library
- `rednose/SConscript` - Kalman filter library

## External Libraries by Category

### Computer Vision & Media
- **libyuv** - YUV image format processing
- **OpenCV** - Computer vision (development dependency)
- **FFmpeg** - Video/audio encoding and decoding
- **JPEG** - Image compression

### Machine Learning
- **ONNX** - Neural network model format
- **SNPE** - Snapdragon Neural Processing Engine (Qualcomm)
- **OpenCL** - GPU compute acceleration
- **Intel OpenCL Runtime** - CPU-based OpenCL implementation

### Mathematics & Optimization  
- **Acados** - Optimal control and nonlinear programming
- **BLASFEO** - Basic Linear Algebra Subprograms
- **HPIPM** - High-performance interior point method
- **Eigen3** - Linear algebra library
- **CasADi** - Symbolic framework for optimization

### Communication & Serialization
- **Cap'n Proto** - Serialization protocol
- **ZeroMQ** - High-performance messaging
- **JSON11** - Lightweight JSON library
- **Protocol Buffers** - Google's serialization (via pycapnp)

### Graphics & UI
- **Qt5** - Cross-platform GUI framework
- **OpenGL/GLES** - Graphics rendering
- **GLFW** - OpenGL window management
- **GLEW** - OpenGL Extension Wrangler
- **Raylib** - Graphics library for tools

### System & Hardware
- **libusb** - USB device communication
- **D-Bus** - Inter-process communication
- **systemd** - Linux system manager integration
- **Wayland** - Display server protocol (ARM64)

### Compression & Archives
- **zlib** - General-purpose compression
- **bzip2** - Block-sorting compression
- **LZMA/XZ** - High-ratio compression  
- **zstandard** - Facebook's compression algorithm
- **libarchive** - Multi-format archive library

### Testing & Development
- **Catch2** - C++ testing framework
- **Kaitai Struct** - Binary data parsing
- **Clazy** - Qt-aware static analyzer