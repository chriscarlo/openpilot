# Comma Device Hardware Capabilities Primer

## Overview
The comma three (C3/C3X) devices provide extensive hardware control through systemd services, sysfs interfaces, GPIO pins, and D-Bus communication. This document provides a developer primer for understanding and interacting with the hardware capabilities.

## Device Types
- **tici** - comma three (C3)
- **mici** - comma three X (C3X) with additional thermal zones

## System Services

### Core Services
- **comma.service** - Main openpilot service orchestrator
- **ModemManager** - Cellular modem control and configuration
- **NetworkManager** - Network connectivity management
- **systemd-resolved** - DNS resolution service
- **lte.service** - LTE connectivity management

### Service Control
```bash
# Start/stop main service
sudo systemctl restart comma
sudo systemctl stop comma

# Modem management
sudo systemctl restart ModemManager
sudo systemctl restart NetworkManager
```

## Hardware Components

### 1. Cellular Modem (Quectel)

**Capabilities:**
- Full modem control via ModemManager D-Bus interface
- AT command access for diagnostics and configuration
- Real-time data usage monitoring
- Temperature monitoring via AT+QTEMP
- Signal strength/quality metrics
- eSIM configuration support
- Device paths: `/dev/cdc-wdm0` (QMI), `/dev/ttyUSB2` (AT)

**Key AT Commands:**
- `AT+QNWINFO` - Network information
- `AT+QENG="servingcell"` - Serving cell details
- `AT+QTEMP` - Temperature readings

### 2. Network Interfaces

**WiFi (wlan0):**
- Access point scanning
- Connection management via NetworkManager D-Bus
- Signal strength monitoring
- WPA/WPA2 support

**Cellular (wwan0):**
- Data usage statistics (TX/RX bytes)
- Metered connection detection
- Connection state monitoring

### 3. Display & Graphics

**Display Control:**
- Path: `/sys/class/backlight/panel0-backlight/`
- Brightness: 0-1023 levels
- Power on/off via `bl_power`
- Full blacklight control for power saving

**GPU (Adreno 630):**
- Path: `/sys/class/kgsl/kgsl-3d0/`
- Max frequency: 710 MHz
- Performance governors
- Power level control (min/max)
- Force bus/clock/rail states for performance
- GPU usage monitoring via `gpubusy`

### 4. CPU Management

**CPU Cores:**
- 8 cores total: 4 silver (efficiency) + 4 gold (performance)
- Individual core on/off control
- Path: `/sys/devices/system/cpu/cpu*/online`

**Frequency Governors:**
- Performance/powersave/schedutil modes
- Per-cluster frequency control
- Path: `/sys/devices/system/cpu/cpufreq/policy*/`

**IRQ Affinity:**
- Custom IRQ routing for optimized performance
- Dedicated cores for critical processes (pandad on core 3)

### 5. GPIO Pin Mapping

**Panda Interface (CAN/Vehicle):**
```python
STM_RST_N = 124      # STM32 reset (high to reset)
STM_BOOT0 = 134      # STM32 boot mode
STM_PWR_EN_N = 41    # STM32 power enable
```

**GPS/GNSS:**
```python
UBLOX_RST_N = 32     # u-blox reset
UBLOX_SAFEBOOT_N = 33  # u-blox safe boot
GNSS_PWR_EN = 34     # GNSS power enable
```

**LTE Modem:**
```python
LTE_RST_N = 50       # LTE reset (high to reset)
LTE_PWRKEY = 116     # LTE power key
LTE_BOOT = 52        # LTE boot mode
```

**Cameras:**
```python
CAM0_AVDD_EN = 8     # Road camera analog power
CAM0_RSTN = 9        # Road camera reset
CAM1_RSTN = 7        # Driver camera reset
CAM2_RSTN = 12       # Wide camera reset
```

**IMU Sensors:**
```python
BMX055_ACCEL_INT = 21  # Accelerometer interrupt
BMX055_GYRO_INT = 23   # Gyroscope interrupt
BMX055_MAGN_INT = 87   # Magnetometer interrupt
LSM_INT = 84           # LSM6DS3 IMU interrupt
```

**Other:**
```python
SIREN = 42           # Alert buzzer
HUB_RST_N = 30       # USB hub reset
SOM_ST_IO = 49       # Fan control GPIO
```

### 6. Audio System

**Amplifier (TAS2563):**
- I2C control (bus 2, address 0x4c)
- EQ parameter configuration
- Model-specific tuning profiles
- Global shutdown capability
- Initialize with device type for proper calibration

### 7. Power Monitoring

**Power Metrics:**
- Current draw: `/sys/class/hwmon/hwmon1/power1_input` (µW)
- Battery voltage: `/sys/class/power_supply/bms/voltage_now`
- Battery current: `/sys/class/power_supply/bms/current_now`
- Calculated SoM power = voltage × current

### 8. Thermal Management

**Thermal Zones:**
- CPU: 8 zones (4 silver + 4 gold cores)
- GPU: 2 zones (gpu0-usr, gpu1-usr)
- DSP: compute-hvx-usr
- Memory: ddr-usr
- PMIC: pm8998_tz, pm8005_tz
- C3X only: intake, exhaust, case zones

**Fan Control:**
- GPIO-controlled via SOM_ST_IO pin
- Automatic thermal management
- Configurable speed curves

### 9. Storage & Boot

**Partitions:**
- System partition management via AGNOS
- OTA update system with A/B slots
- Factory reset capability
- Boot mode control

### 10. Memory & Performance

**Memory Governors:**
```bash
/sys/class/devfreq/soc:qcom,cpubw/governor
/sys/class/devfreq/soc:qcom,memlat-cpu0/governor
/sys/class/devfreq/soc:qcom,memlat-cpu4/governor
```

**Video Encoder (VIDC):**
- Clock scaling control
- Thermal mitigation bypass for consistent performance

## Communication Interfaces

### D-Bus
Primary interface for system services:
- NetworkManager: `org.freedesktop.NetworkManager`
- ModemManager: `org.freedesktop.ModemManager1`
- Timeout for operations: typically 5 seconds

### I2C
- Amplifier control (bus 2)
- Sensor communication
- Power management ICs

### SPI
- Display interface
- High-priority with dedicated CPU core

### UART
- GPS module communication
- Modem AT commands
- Debug console

### CAN
- Vehicle communication via panda
- Safety-critical with real-time constraints

## Hardware Initialization Sequence

1. **Amplifier Configuration** - Device-specific audio tuning
2. **GPIO Setup** - Enable fan, configure pins
3. **IRQ Affinity** - Optimize interrupt routing
4. **GPU Configuration** - Set performance parameters
5. **CPU Governors** - Configure power management
6. **Video Encoder** - Optimize for consistent encoding

## Safety Considerations

- All vehicle-critical functions go through panda safety controller
- Hardware watchdog for system recovery
- Thermal protection with automatic throttling
- Power monitoring to prevent brownouts

## Development Tips

### Testing Without Hardware
```bash
# Force UI into onroad mode without cameras
FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &
```

### Performance Monitoring
```python
# GPU usage
with open('/sys/class/kgsl/kgsl-3d0/gpubusy') as f:
    busy, total = map(int, f.read().split())
    usage_percent = (busy / max(1, total)) * 100
```

### Network Diagnostics
```python
# Get modem signal strength
modem.Get(MM_MODEM, 'SignalQuality', dbus_interface=DBUS_PROPS)

# Check network type
modem.Get(MM_MODEM, 'AccessTechnologies', dbus_interface=DBUS_PROPS)
```

## Common Operations

### Reset Internal Panda
```python
gpio_set(GPIO.STM_RST_N, 1)  # Assert reset
time.sleep(0.1)
gpio_set(GPIO.STM_RST_N, 0)  # Release reset
```

### Control Display Brightness
```python
# Set to 50% brightness
with open("/sys/class/backlight/panel0-backlight/brightness", "w") as f:
    f.write(str(512))
```

### Monitor Power Draw
```python
# Read current power consumption in watts
with open("/sys/class/hwmon/hwmon1/power1_input") as f:
    power_uw = int(f.read())
    power_w = power_uw / 1e6
```

## Further Reading

- AGNOS kernel configuration: [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845)
- Hardware abstraction: `system/hardware/tici/hardware.py`
- GPIO definitions: `system/hardware/tici/pins.py`
- Thermal management: `system/hardware/hardwared.py`