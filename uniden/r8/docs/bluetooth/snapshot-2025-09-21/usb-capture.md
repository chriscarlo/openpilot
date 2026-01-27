## USB Handshake Capture (Windows)

Use the official Uniden Windows tool to elicit the handshake, and capture the serial USB traffic for offline analysis. We only need the initial connect/detect phase — do NOT start flashing.

### Option A: Wireshark + USBPcap (recommended)
1) Install Wireshark on Windows (include USBPcap).
2) Identify the CP210x COM port (Device Manager → Ports → Silicon Labs CP210x (COMx)).
3) Start Wireshark, choose the USBPcap interface (e.g., USBPcap1). Begin capture.
4) Launch the Uniden tool and perform a “detect” or “check version” operation. Let it run 3–5 seconds. Stop capture.
5) In Wireshark, filter for bulk transfers:
   - Display filter: `usb.transfer_type == 2 && usb.capdata`.
   - Optionally restrict to the device address (`usb.device_address == N`) by finding the CP210x attach event.
6) Export hex of payloads for IN and OUT separately:
   - OUT (host→device): `tshark -r capture.pcapng -Y "usb.transfer_type == 2 && usb.capdata && usb.endpoint_address.direction == 0" -T fields -e usb.capdata > out_hex.txt`
   - IN (device→host): `tshark -r capture.pcapng -Y "usb.transfer_type == 2 && usb.capdata && usb.endpoint_address.direction == 1" -T fields -e usb.capdata > in_hex.txt`
7) Copy `out_hex.txt` and `in_hex.txt` into: `uniden/r8/docs/bluetooth/snapshot-2025-09-21/logs/`.
8) Analyze in WSL/Linux (from `uniden/r8`): `python3 -m r8_tools.usb_hex_analyze --out docs/bluetooth/snapshot-2025-09-21/logs/out_hex.txt --in docs/bluetooth/snapshot-2025-09-21/logs/in_hex.txt`

### Option B: Serial Port Monitor (Windows)
- Use a serial sniffer (e.g., HHD Free Serial Port Monitor) to attach to COMx and log hex RX/TX. Save RX and TX as separate plain text files (one hex blob per line). Analyze via the same tool:
  - `python3 -m r8_tools.usb_hex_analyze --out tx_hex.txt --in rx_hex.txt`

### Safety
- Only run the detection/query step in the official tool. Do not start flashing. The handshake we want is present during the detect/version phases.

