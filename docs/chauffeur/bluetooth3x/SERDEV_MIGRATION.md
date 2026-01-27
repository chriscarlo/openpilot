Title: WCN3990 Serdev Migration (SDM845 / Comma 3/3X)

Objective
- Move from userspace btattach to kernel serdev `hci_qca` for WCN3990 bring-up, to get reliable power pulses, speed bump, TLV/NVM download, and hci0 creation without races.

Scope
- Kernel: 4.9.x (agnos-kernel-sdm845). Backport minimal `hci_qca` pieces for WCN3990 and enable `CONFIG_SERIAL_DEV_BUS`, `CONFIG_BT_HCIUART_SERDEV`.
- DT: Add a `qcom,wcn3990-bt` child under the SDM845 UART node wired to the BT controller.

Steps
- Kernel config
  - enable: `CONFIG_SERIAL_DEV_BUS=y`, `CONFIG_BT_HCIUART=y`, `CONFIG_BT_HCIUART_QCA=y`, `CONFIG_BT_HCIUART_SERDEV=y`
  - optional: `CONFIG_DYNAMIC_DEBUG=y` for early breadcrumbs

- hci_qca updates (patchset provided)
  - Apply patches in `docs/chauffeur/bluetooth3x/patches/`:
    - 0001: WCN3990 OFF/ON power pulses + reopen
    - 0002: Drop VSE 0x92 wait; add 3.2 Mbps mapping
    - 0003: btqca TLV/NVM hardening (adapt as needed)

- Device tree (sdm845)
  - Identify the UART line to the BT module (often `uart3` or `uart6` depending on board). Example snippet:

```
&uart6 {
  status = "okay";
  bluetooth {
    compatible = "qcom,wcn3990-bt";
    max-speed = <3200000>; /* or 3000000 */
    qcom,bt-vdd-io-supply = <&pm8998_lvs1>;
    qcom,bt-vdd-xtal-supply = <&pm8998_l11>;
    qcom,bt-vdd-pa-supply = <&pm8998_l22>;
    /* Optional reset GPIO if available */
    // reset-gpios = <&tlmm 85 GPIO_ACTIVE_HIGH>;
  };
};
```

  - Ensure regulators exist and are enabled for your board. Some designs use a single always-on supply; in that case, drop the properties.

- Build + flash
  - Build kernel/boot.img
  - Flash both slots; reboot; verify `serdev` creates `serial0-0` (or similar) device and that `hci0` appears.

Validation
- `dmesg | rg -i 'qca|wcn3990|serdev|bluetooth'`
- `ls -l /sys/bus/serdev/devices`
- `hciconfig -a` and `btmgmt info`

Fallback
- Keep `scripts/bluetooth/strict_attach_window.sh` for controlled userspace attach attempts during development. Once serdev is working, stop using btattach.

References
- Linux mainline `hci_qca.c` WCN3990 code path (power pulses, port reopen, VSE drop)
- Android common kernel implementations and downstream device trees for SDM845/SM8250 platforms using WCN399x

