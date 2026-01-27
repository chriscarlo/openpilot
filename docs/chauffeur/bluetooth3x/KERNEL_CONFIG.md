Title: Kernel Config — WCN3990 over UART (serdev) on SDM845

Essential
- CONFIG_BT=y
- CONFIG_BT_BREDR=y
- CONFIG_BT_LE=y
- CONFIG_BT_HCIUART=y
- CONFIG_BT_HCIUART_QCA=y
- CONFIG_BT_HCIUART_SERDEV=y
- CONFIG_SERIAL_DEV_BUS=y

Helpful
- CONFIG_DYNAMIC_DEBUG=y (runtime debug toggles)
- CONFIG_RFKILL=y

Notes
- For older 4.9 trees lacking `*_SERDEV`, backport minimal hci_qca serdev support or keep using userspace btattach temporarily while applying the patchset in `docs/chauffeur/bluetooth3x/patches/`.
- Device Tree must add a `qcom,wcn3990-bt` child under the chosen UART node; see SERDEV_MIGRATION.md.

