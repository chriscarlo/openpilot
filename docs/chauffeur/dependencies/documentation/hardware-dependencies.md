# Hardware Dependencies

Hardware-specific drivers, libraries, and interfaces required for OpenPilot's interaction with physical devices.

## Hardware Interface Libraries

### USB Communication

**libusb1** (Python) / **libusb-1.0** (System)
- **Purpose**: Direct USB device communication with panda hardware
- **Usage**: Primary interface for CAN bus communication via panda device
- **Configuration**: Requires specific udev rules for device permissions
- **Python package**: `libusb1` in main project and panda submodule
- **System package**: `libusb-1.0-0-dev` (Ubuntu), `libusb` (macOS)

**Supported USB devices** (from udev rules):
- **Panda devices**: VID 3801/bbaa, PID ddcc/ddee
- **Panda DFU mode**: VID 0483, PID df11  
- **Jungle devices**: VID 3801/bbaa, PID ddcf/ddef

### SPI Communication

**spidev** (Linux only)
- **Purpose**: Serial Peripheral Interface communication
- **Platform**: Linux only (`platform_system == 'Linux'`)
- **Usage**: Alternative hardware communication method
- **Python package**: `spidev` (conditional dependency)
- **System package**: Available through kernel SPI subsystem

## Camera and Vision Hardware

### Camera Interface

**FFmpeg ecosystem** - Video capture and processing:
- `libavformat-dev` - Container format handling
- `libavcodec-dev` - Video/audio codec support
- `libavdevice-dev` - Device access for cameras
- `libavutil-dev` - Utility functions
- `libavfilter-dev` - Audio/video filtering

**System integration**:
- **camerad service** - Camera daemon (AGNOS/larch64 only)
- **Multiple camera support** - Front-facing and driver-facing cameras
- **Real-time processing** - Hardware-accelerated where available

### Image Processing

**libyuv** - YUV format processing:
- **Purpose**: Efficient YUV color space conversions
- **Architecture support**: larch64, x86_64, Darwin
- **Location**: `#third_party/libyuv/{arch}/lib`
- **Usage**: Camera frame processing and format conversion

**OpenCV** (development):
- **Python package**: `opencv-python-headless`
- **Purpose**: Computer vision algorithms and image processing
- **Usage**: Development and testing tools

## Neural Processing Hardware

### Snapdragon Neural Processing Engine (SNPE)

**Platform support**:
- **larch64**: ARM64 comma three device (`#third_party/snpe/larch64`)
- **x86_64**: x86 development machines (`#third_party/snpe/x86_64`)
- **Not supported**: macOS/Darwin

**Integration**:
- **Include path**: `#third_party/snpe/include`
- **Runtime libraries**: Platform-specific in third_party directory
- **Usage**: Hardware-accelerated neural network inference

### OpenCL Compute

**OpenCL support** - GPU/CPU acceleration:
- **Headers**: `opencl-headers` (Ubuntu)
- **Runtime**: `ocl-icd-libopencl1` (Ubuntu)
- **Development**: `ocl-icd-opencl-dev` (Ubuntu)
- **Include path**: `#third_party/opencl/include` (AGNOS)

**Intel OpenCL CPU Runtime** (Docker):
- **Version**: 2024.17.3.0.09 experimental
- **Location**: `/opt/intel/oclcpuexp_2024.17.3.0.09_rel/`
- **ICD file**: `/etc/OpenCL/vendors/intel_expcpu.icd`
- **Dependencies**: Intel TBB 2021.12.0

**Python integration**:
- **PyOpenCL**: `pyopencl` (not on ARM64 due to compatibility issues)  
- **PyTools**: `pytools < 2024.1.11` (ARM64 version has issues)

### NVIDIA GPU Support

**Docker environment variables**:
- `NVIDIA_VISIBLE_DEVICES=all` - All GPUs accessible
- `NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute` - GPU capabilities
- **Requirements**: NVIDIA Docker runtime for container GPU access

## Audio Hardware

### Audio Interface

**PortAudio** - Cross-platform audio I/O:
- **System package**: `portaudio19-dev` (Ubuntu), `portaudio` (macOS)
- **Python package**: `pyaudio` 
- **Usage**: Audio input/output for voice communication and alerts

**Sound system integration**:
- **PulseAudio**: System audio server (Linux)
- **soundd service**: OpenPilot sound daemon
- **micd service**: Microphone input daemon

**Audio hardware support**:
- **USB audio devices** - External audio interfaces
- **Built-in audio** - System integrated audio
- **Bluetooth audio** - Wireless audio devices (platform dependent)

## CAN Bus and Vehicle Interface

### Panda Hardware Interface

**Panda device** - CAN bus interface:
- **Communication**: USB 2.0 connection
- **Protocols**: CAN 2.0A/2.0B, OBD-II, K-line
- **Power**: 12V automotive power or USB power
- **Safety**: Hardware safety model enforcement

**CAN bus support**:
- **Multiple CAN buses** - Simultaneous multi-bus communication
- **CAN-FD support** - Extended CAN protocol
- **Message filtering** - Hardware-level message filtering
- **Timestamping** - Precise message timing

### Vehicle Communication Protocols

**OBD-II interface**:
- **Standard protocols**: ISO 9141-2, ISO 14230-4 (KWP2000), ISO 15765-4 (CAN)
- **Diagnostic services**: UDS, OBD-II parameter IDs
- **Security access**: Manufacturer-specific authentication

**Manufacturer protocols**:
- **Honda**: PGM-FI, SH-AWD integration
- **Toyota**: DSU, TSS integration  
- **GM**: GlobalConnect, CAN bus variants
- **Ford**: SYNC, FNV4 integration
- **Others**: Manufacturer-specific CAN implementations

## Sensor Hardware

### GPS and Positioning

**u-blox GPS receivers**:
- **ubloxd service** - GPS daemon
- **Protocols**: UBX binary protocol, RTCM corrections
- **Accuracy**: Centimeter-level with RTK corrections
- **Integration**: GNSS + IMU sensor fusion

**Positioning systems**:
- **GPS** - Global Positioning System
- **GLONASS** - Russian satellite navigation
- **Galileo** - European satellite navigation
- **BeiDou** - Chinese satellite navigation

### Inertial Measurement

**IMU sensors** - Accelerometer, gyroscope, magnetometer:
- **Data fusion** - Kalman filter integration
- **Calibration** - Automatic sensor calibration
- **Orientation** - Vehicle attitude estimation

**Sensor locations**:
- **comma three device** - Internal IMU
- **Vehicle integration** - CAN bus sensor data
- **Sensor fusion** - Multiple sensor combination

## Display and User Interface

### Display Hardware

**Touch displays**:
- **comma three** - Integrated touch display
- **External displays** - HDMI output support
- **Resolution support** - Multiple resolution handling

**Graphics acceleration**:
- **OpenGL ES** - Mobile graphics API (`libgles2-mesa-dev`)
- **OpenGL** - Desktop graphics API
- **GPU drivers** - Platform-specific GPU drivers

### Input Devices

**Touch input**:
- **Capacitive touch** - Multi-touch gesture support
- **Resistive touch** - Single-touch support
- **Touch protocols** - Platform-specific touch drivers

**Physical controls**:
- **Volume buttons** - Hardware volume control
- **Power button** - System power management
- **External buttons** - Optional external input devices

## Joystick and Game Controllers

### Input Device Support

**inputs library** - Cross-platform input device support:
- **USB joysticks** - Standard USB HID joysticks
- **Bluetooth controllers** - Wireless game controllers
- **Keyboard input** - Keyboard as input device
- **Mouse input** - Mouse as input device

**Supported controllers**:
- **Xbox controllers** - Microsoft Xbox One/Series controllers
- **PlayStation controllers** - Sony DualShock/DualSense controllers
- **Generic HID** - Standard USB HID game controllers
- **Custom controllers** - User-defined input mappings

## Network Hardware

### Ethernet Interface

**Wired networking**:
- **Gigabit Ethernet** - High-speed wired connection
- **Network configuration** - Static/DHCP configuration
- **Network services** - SSH, HTTP, WebRTC services

### Wi-Fi Hardware

**Wireless networking**:
- **Wi-Fi standards** - 802.11a/b/g/n/ac support
- **WPA security** - WPA2/WPA3 encryption
- **Access point mode** - Device as Wi-Fi hotspot
- **Client mode** - Connection to existing networks

**Wi-Fi management**:
- **Network manager** - NetworkManager integration
- **Connection profiles** - Saved network configurations
- **Signal strength** - Connection quality monitoring

### Cellular Connectivity

**LTE/5G modems** (comma three):
- **Data connection** - Cellular data for cloud connectivity
- **GPS assistance** - A-GPS for faster GPS lock
- **Network fallback** - Cellular when Wi-Fi unavailable

## Power Management

### Power Supply

**Automotive power**:
- **12V input** - Standard automotive 12V supply
- **Power management** - Voltage monitoring and protection
- **Low voltage detection** - Automatic shutdown protection
- **Ignition detection** - Vehicle ignition state monitoring

**USB power**:
- **USB-C power** - Development and testing power
- **Power delivery** - USB-PD for higher power requirements
- **Battery backup** - Optional battery power (development)

### Power Monitoring

**Voltage monitoring**:
- **Hardware monitoring** - Voltage sensor reading
- **Low voltage warnings** - User notifications
- **Automatic shutdown** - Protection against undervoltage
- **Power state management** - Sleep/wake functionality

**Thermal management**:
- **Temperature monitoring** - CPU/GPU temperature sensors
- **Fan control** - Active cooling management (comma three)
- **Thermal throttling** - Performance scaling for thermal protection

## Hardware Architecture Support

### Target Architectures

From `/data/openpilot/SConstruct` analysis:

**larch64** - Linux TICI ARM64 (comma three device):
- **CPU**: ARM Cortex-A57 (`-mcpu=cortex-a57`)
- **GPU**: Adreno GPU with OpenCL support
- **NPU**: Snapdragon Neural Processing Engine
- **Connectivity**: LTE modem, Wi-Fi, Bluetooth

**aarch64** - Linux PC ARM64:
- **CPU**: Various ARM64 processors
- **GPU**: Varies by hardware (Mali, Adreno, Apple GPU)
- **Development**: ARM64 development machines

**x86_64** - Linux PC x64:
- **CPU**: Intel/AMD x86_64 processors
- **GPU**: Intel/NVIDIA/AMD GPUs with OpenCL
- **Development**: Primary development platform

**Darwin** - macOS (x64 or ARM64):
- **CPU**: Intel x86_64 or Apple Silicon ARM64
- **GPU**: Intel/AMD/Apple GPUs with Metal/OpenCL
- **Development**: macOS development environment

## Device-Specific Hardware

### comma three Device (AGNOS)

**System-on-Chip**:
- **SoC**: Snapdragon-based ARM processor
- **RAM**: System memory for applications
- **Storage**: eMMC flash storage
- **GPU**: Adreno graphics with OpenCL compute

**Integrated sensors**:
- **IMU**: 9-axis inertial measurement unit
- **GPS**: u-blox GNSS receiver with corrections
- **Cameras**: Front-facing and driver-facing cameras
- **Microphone**: Audio input for voice commands

**Connectivity**:
- **CAN**: Integrated CAN bus interface
- **USB**: USB device and host support
- **Wi-Fi**: 802.11ac wireless networking
- **LTE**: Cellular data connectivity
- **Bluetooth**: Device pairing and audio

**Display and input**:
- **Touchscreen**: Capacitive touch display
- **Buttons**: Physical control buttons
- **LEDs**: Status indication LEDs
- **Audio**: Speaker and headphone output

### Development Hardware

**Development machines**:
- **Minimum RAM**: 8GB (16GB recommended)
- **Storage**: SSD recommended for build performance
- **USB ports**: For panda device connection
- **Network**: Ethernet or Wi-Fi for dependencies

**Optional hardware**:
- **CAN analyzer** - For CAN bus development and debugging
- **OBD-II adapter** - For vehicle protocol testing
- **Logic analyzer** - For hardware signal analysis
- **Oscilloscope** - For analog signal measurement