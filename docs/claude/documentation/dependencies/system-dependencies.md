# System Dependencies

System-level packages and tools required for building and running OpenPilot across different platforms.

## Ubuntu Dependencies

From `/data/openpilot/tools/install_ubuntu_dependencies.sh`

### Common Ubuntu Packages (lines 23-67)

**Compilers and build tools:**
- `clang` - LLVM C/C++ compiler (primary compiler)
- `build-essential` - Essential build tools (GCC, make, etc.)
- `gcc-arm-none-eabi` - ARM cross-compiler for embedded targets
- `g++-12` - GCC 12 C++ compiler (Ubuntu 24.04 specific)

**Core libraries:**
- `ca-certificates` - Certificate authority certificates
- `liblzma-dev` - LZMA compression library
- `libffi-dev` - Foreign Function Interface library
- `libssl-dev` - OpenSSL cryptographic library
- `libbz2-dev` - Bzip2 compression library
- `libncurses5-dev` - Terminal UI library
- `libsqlite3-dev` - SQLite database library
- `libsystemd-dev` - systemd library
- `libzmq3-dev` - ZeroMQ messaging library
- `libzstd-dev` - Zstandard compression library

**Cap'n Proto serialization:**
- `capnproto` - Cap'n Proto compiler
- `libcapnp-dev` - Cap'n Proto development headers

**Networking:**
- `curl` - HTTP client library
- `libcurl4-openssl-dev` - libcurl development headers

**Version control:**
- `git` - Git version control system
- `git-lfs` - Git Large File Storage

**Multimedia libraries (FFmpeg ecosystem):**
- `ffmpeg` - Multimedia framework
- `libavformat-dev` - Container format library
- `libavcodec-dev` - Codec library  
- `libavdevice-dev` - Device access library
- `libavutil-dev` - Utility library
- `libavfilter-dev` - Filter library

**Graphics and OpenGL:**
- `libeigen3-dev` - Linear algebra library
- `libglew-dev` - OpenGL Extension Wrangler
- `libgles2-mesa-dev` - OpenGL ES 2.0 library
- `libglfw3-dev` - OpenGL framework library
- `libglib2.0-0` - GLib utility library
- `libjpeg-dev` - JPEG image library

**Qt5 GUI framework:**
- `qtbase5-dev` - Qt5 base development files
- `qtchooser` - Qt version chooser
- `qt5-qmake` - Qt5 build system
- `qtbase5-dev-tools` - Qt5 development tools
- `qttools5-dev-tools` - Additional Qt5 tools
- `libqt5charts5-dev` - Qt5 charts library
- `libqt5svg5-dev` - Qt5 SVG library
- `libqt5serialbus5-dev` - Qt5 serial bus library
- `libqt5x11extras5-dev` - Qt5 X11 extras
- `libqt5opengl5-dev` - Qt5 OpenGL library

**OpenCL compute:**
- `opencl-headers` - OpenCL header files
- `ocl-icd-libopencl1` - OpenCL runtime
- `ocl-icd-opencl-dev` - OpenCL development files

**Audio:**
- `portaudio19-dev` - Cross-platform audio I/O library

**USB hardware interface:**
- `libusb-1.0-0-dev` - USB device access library

**System utilities:**
- `locales` - Locale support
- `xvfb` - X Virtual Framebuffer (headless GUI testing)

**Python environment:**
- `python3-dev` - Python development headers
- `python3-venv` - Python virtual environment support

### Supported Ubuntu Versions

From `/data/openpilot/tools/install_ubuntu_dependencies.sh` (lines 87-99):
- **Supported**: "jammy" (22.04), "kinetic" (22.10), "noble" (24.04), "focal" (20.04)
- **Primary target**: Ubuntu 24.04 LTS ("noble")
- **Legacy support**: Ubuntu 20.04 LTS ("focal")

## macOS Dependencies  

From `/data/openpilot/tools/mac_setup.sh`

### Homebrew Packages (lines 35-53)

**Version control:**
- `git-lfs` - Git Large File Storage

**Compression libraries:**
- `zlib` - Compression library

**Serialization:**
- `capnp` - Cap'n Proto compiler

**Core utilities:**
- `coreutils` - GNU core utilities

**Mathematics:**
- `eigen` - Linear algebra library

**Multimedia:**
- `ffmpeg` - Multimedia framework

**Graphics:**
- `glfw` - OpenGL framework

**Archive handling:**
- `libarchive` - Multi-format archive library

**USB interface:**
- `libusb` - USB device library

**Build tools:**
- `libtool` - Generic library support script

**Compilers:**
- `llvm` - LLVM compiler infrastructure  
- `gcc@13` - GCC 13 compiler
- `gcc-arm-embedded` - ARM cross-compiler (cask)

**Crypto:**
- `openssl@3.0` - OpenSSL 3.0 cryptographic library

**GUI framework:**
- `qt@5` - Qt5 GUI framework

**Messaging:**
- `zeromq` - ZeroMQ messaging library

**Audio:**
- `portaudio` - Cross-platform audio I/O

### macOS Environment Variables (lines 60-69)

**Library paths:**
```bash
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/zlib/lib"
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/bzip2/lib" 
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/openssl@3/lib"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/zlib/include"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/bzip2/include"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/openssl@3/include"
```

**PyCurl configuration:**
```bash
export PYCURL_CURL_CONFIG=/usr/bin/curl-config
export PYCURL_SSL_LIBRARY=openssl
```

## Docker Dependencies

From `/data/openpilot/Dockerfile.openpilot_base`

### Base Image (line 1)
- `ubuntu:24.04` - Ubuntu 24.04 LTS base image

### System Packages (lines 7-35)

**Desktop environment:**
- `sudo` - Privilege escalation
- `tzdata` - Timezone data
- `locales` - Locale support
- `ssh` - Secure Shell client
- `pulseaudio` - Audio server
- `xvfb` - X Virtual Framebuffer
- `x11-xserver-utils` - X11 utilities
- `gnome-screenshot` - Screenshot utility
- `python3-tk` - Python Tkinter GUI
- `python3-dev` - Python development headers

**Archive and compression:**
- `apt-utils` - APT utility programs
- `alien` - Package format converter
- `unzip` - Archive extraction
- `tar` - Archive utility
- `curl` - HTTP client
- `xz-utils` - XZ compression utilities

**System utilities:**
- `dbus` - D-Bus message bus
- `gcc-arm-none-eabi` - ARM cross-compiler
- `tmux` - Terminal multiplexer
- `vim` - Text editor
- `libx11-6` - X11 client library
- `wget` - Web downloader

**Fonts:**
- `fonts-noto-cjk` - Noto CJK fonts
- `fonts-noto-color-emoji` - Noto color emoji fonts

### Intel OpenCL Runtime (lines 37-56)

**OpenCL CPU runtime installation:**
- Intel OpenCL CPU Experimental Runtime 2024.17.3.0.09
- Intel Threading Building Blocks (TBB) 2021.12.0
- Runtime libraries linked in `/opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64/`
- ICD file: `/etc/OpenCL/vendors/intel_expcpu.icd`

### NVIDIA Support (lines 58-60)

**Environment variables:**
- `NVIDIA_VISIBLE_DEVICES=all` - All GPUs visible
- `NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute` - GPU capabilities
- `QTWEBENGINE_DISABLE_SANDBOX=1` - Qt WebEngine sandbox disabled

### User Configuration (lines 65-70)

**Default user setup:**
- User: `batman` (UID: 1001)
- Groups: `sudo` (passwordless sudo access)
- Home directory: `/home/batman`

## CI/CD Environment

From `/data/openpilot/.github/workflows/selfdrive_tests.yaml`

### GitHub Actions Runners

**Ubuntu runners:**
- `ubuntu-24.04` - Primary Linux runner
- `ubuntu-latest` - Latest Ubuntu (fallback)

**macOS runners:**
- `namespace-profile-macos-8x14` - Custom high-performance runner (comma.ai)
- `macos-latest` - Standard GitHub runner (fallback)

### GitHub Actions

**Core actions:**
- `actions/checkout@v4` - Repository checkout
- `actions/cache@v4` - Build cache management
- `actions/upload-artifact@v4` - Artifact storage

**Third-party actions:**
- `nick-fields/retry@7152eba30c6575329ac0576536151aca5a72780e` - Retry functionality
- `peter-evans/find-comment@3eae4d37986fb5a8592848f6a574fdf654e61f9e` - Comment management
- `peter-evans/create-or-update-comment@71345be0265236311c031f5c7866368bd1eff043` - Comment creation

## Environment Variables

### Common Environment Variables

**Build configuration:**
- `PYTHONWARNINGS=error` - Treat warnings as errors
- `CI=1` - CI environment indicator
- `FILEREADER_CACHE=1` - Enable file reader caching
- `NUM_JOBS` / `JOB_ID` - Parallel job configuration

**Cache directories:**
- `/tmp/scons_cache` - SCons build cache
- `/tmp/comma_download_cache` - Download cache
- `/tmp/openpilot_cache` - General OpenPilot cache

**macOS specific:**
- `HOMEBREW_NO_AUTO_UPDATE=1` - Disable Homebrew auto-update
- `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` - Disable fork safety (line 30 in install script)
- `ZMQ=1` - Use ZeroMQ instead of msgq (macOS compatibility)

## Hardware Requirements

### Minimum System Requirements

**Memory:**
- RAM: 8GB minimum (16GB recommended for compilation)
- Swap: Additional swap space recommended for compilation

**Storage:**  
- Disk space: 20GB+ for full development environment
- SSD recommended for build performance

**CPU:**
- Multi-core processor recommended
- ARM64 support for Apple Silicon and comma three device
- x86_64 support for development machines

### Supported Architectures

From `/data/openpilot/SConstruct` (lines 70-81):
- `larch64` - Linux TICI ARM64 (comma device)
- `aarch64` - Linux PC ARM64  
- `x86_64` - Linux PC x64
- `Darwin` - macOS (x64 or ARM64)

## Udev Rules (Linux)

From `/data/openpilot/tools/install_ubuntu_dependencies.sh` (lines 103-120):

**Panda device rules:**
```
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddee", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddee", MODE="0666"
```

**Jungle device rules:**
```
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddef", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"
```