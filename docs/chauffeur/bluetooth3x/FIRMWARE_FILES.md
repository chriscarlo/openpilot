Title: WCN3990 Bluetooth Firmware Files

Expected filenames (WCN3990)
- Rampatch TLV: `crbtfw21.tlv`
- NVM blob: `crnv21.bin`

Locations
- Preferred: `/lib/firmware/qca/`
- Overrides (if kernel is configured): `/lib/firmware/updates/qca/` takes precedence.

How to verify on device
- `ls -l /lib/firmware/qca | rg -n 'crbtfw|crnv'`
- dmesg will log when btqca requests firmware; use `dmesg | rg -n 'btqca|qca|firmware'`.

References
- Linux kernel btqca.c firmware lookup logic (crbtfwXX.tlv, crnvXX.bin) for WCN399x
- Android common kernel btqca.c (same behavior)

