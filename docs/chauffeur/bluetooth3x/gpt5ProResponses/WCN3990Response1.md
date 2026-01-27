Here’s the exact mapping for **ROM 0x02140201 (WCN3990 on SDM845 over UART)** that `hci_qca/btqca` expects, plus what to do on a 4.9 tree that’s still trying the old “rampatch\_\*.bin” names.

---

## TL;DR (what to put on the device)

Place these two files from **linux‑firmware** under any firmware path your loader checks (e.g. `/lib/firmware/updates/qca/` or `/lib/firmware/qca/`):

* **`qca/crbtfw21.tlv`**  ← BT **rampatch** for WCN3990 (TLV format). ([kernel.googlesource.com][1])
* **`qca/crnv21.bin`**   ← BT **NVM/config** for WCN3990. ([kernel.googlesource.com][1])

No conversion needed. They’re already in the TLV (patch) and raw NVM formats the upstream `btqca` loader uses for WCN399x. ([lkml.iu.edu][2], [Android Git Repositories][3])

---

## Where these live upstream (paths & provenance)

* **Paths in linux‑firmware**: `qca/crbtfw21.tlv` and `qca/crnv21.bin`. They’re documented in **WHENCE** under “Driver: qca – Qualcomm Atheros Bluetooth support for WCN399x chips.” ([kernel.googlesource.com][1])
* **Added by**: “qca: Add firmware files for BT chip wcn3990” (2019‑02). Multiple downstream packagers/changelogs and mirrors reference that import. ([SUSE][4], [rpmfind.net][5])

  * Example mirrors showing the files today: Arch package file list includes `…/qca/crbtfw21.tlv` and `…/qca/crnv21.bin`. ([GitLab][6])

---

## How 0x02140201 maps to the filenames

When the controller reports **ROM version 0x02140201**, the upstream `hci_qca`/`btqca` path identifies it as **WCN3990** and downloads **`qca/crbtfw21.tlv`** (then **`qca/crnv21.bin`**). Kernel logs from that code path literally show:

```
Bluetooth: hci0: QCA controller version 0x02140201
Bluetooth: hci0: QCA Downloading qca/crbtfw21.tlv
```

—that’s WCN3990 using the **“21”** pair. ([lkml.iu.edu][2], [lkml.indiana.edu][7])

For contrast, **WCN3991/3998** use the “32” pair (**`crbtfw32.tlv` / `crnv32*.bin`**). Don’t use those for 0x02140201. The linux‑firmware WHENCE section lists the WCN399x families and their file pairs. ([kernel.googlesource.com][1])

---

## Why your 4.9 driver is asking for `rampatch_02140201.bin`

Older/downstream 4.9 trees (and some vendor forks) still print “**ROME controller version …**” and follow the *Rome* (QCA61x4) naming scheme (`rampatch_<rom>.bin` / `nvm_<rom>.bin`). Upstream support for WCN3990 changed the naming/format to **`crbtfw*.tlv` + `crnv*.bin`**. In upstream logs (5.x) you’ll see it correctly fetch **`qca/crbtfw21.tlv`**, not `rampatch_*.bin`. ([lkml.iu.edu][2], [Chromium Git Repositories][8])

### Two ways to make a 4.9 tree happy:

1. **Preferred (code‑correct) fix:** Backport the WCN3990 firmware‑name logic from upstream `btqca/hci_qca` so 0x02140201 maps to **`crbtfw21.tlv` + `crnv21.bin`**. That’s how the 2018–2019 WCN3990 series implemented it. (See the “Enable 3.2 Mbps … / Add WCN3990 firmware download support” patchset.) ([lkml.iu.edu][9])

2. **Pragmatic shim (no code changes):** If your 4.9 driver insists on `rampatch_02140201.bin` and `nvm_02140201.bin`, **symlink** them to the right files:

   ```
   /lib/firmware/qca/rampatch_02140201.bin -> crbtfw21.tlv
   /lib/firmware/qca/nvm_02140201.bin      -> crnv21.bin
   ```

   This works **only if** your loader doesn’t try to parse a Rome‑style (non‑TLV) blob. Some downstream 4.9 trees do; if so you’ll get parse errors (e.g., `-22/-84`) because TLV != old Rome binary. In that case, you must take option (1). The upstream log showing `crbtfw21.tlv` is the canonical behavior. ([lkml.iu.edu][2])

---

## Are there sub‑revisions for 0x02140201?

For **WCN3990** there is **one** public pair in linux‑firmware: **`crbtfw21.tlv`** and **`crnv21.bin`**. The “32” variants (and `crnv32u.bin`) are for WCN3991/3998 or USB variants—not for 3990. ([kernel.googlesource.com][1])

If your product needs board‑specific NVM tweaks (SCO/I2S pins, TX power, etc.), upstream supports overriding the NVM filename via the **`firmware-name`** DT property on the **`qcom,wcn3990-bt`** node (example in the binding doc), but the default for WCN3990 is `crnv21.bin`. ([Android Git Repositories][3])

---

## Baud rate quirk (115200 vs 3M/3.2M)

**WCN3990 runs the firmware download at high speed.** Upstream `hci_qca` for WCN3990 switches the link to \~**3.2 Mbps** *before*/around firmware download and had to add a fix to ignore a stray “baudrate change” vendor event that otherwise confused the TLV downloader. This is different from some Rome flows that stayed at 115200 for patching. So for WCN3990, don’t force “patch at 115200 only.” ([lkml.iu.edu][9], [lkml.indiana.edu][7])

---

## Action checklist for your AGNOS 4.9.103 bring‑up

1. **Install the files**:

   * `qca/crbtfw21.tlv` and `qca/crnv21.bin` under a firmware search path your system uses (you said `/lib/firmware/updates/...` works). See distro file lists confirming these exact names. ([GitLab][6])

2. **If driver still asks for `rampatch_02140201.bin`**:

   * Try **symlinks** first (quick test). If you still get `Failed to download patch (-11/-22/-84)` it’s probably parsing the wrong format → **backport the upstream WCN3990 logic** from `btqca/hci_qca` so it requests `crbtfw21.tlv`. Upstream logs and patches show the intended flow and filenames. ([lkml.iu.edu][2])

3. **DT quality‑of‑life** (optional but nice):

   * On your `qcom,wcn3990-bt` child of the UART, you may set `firmware-name = "crnv21.bin";` (matches the binding example) so you aren’t at the mercy of fallback naming. ([Android Git Repositories][3])

4. **Speed**: Let the driver manage the early bump to \~3.2 Mbps (don’t force 115200 throughout). The upstream fix was specifically for WCN3990. ([lkml.indiana.edu][7])

---

### References you can trust

* **linux‑firmware “WHENCE”**: lists WCN399x BT files and names (crbtfw21.tlv / crnv21.bin, plus 32‑series for 3991/8). ([kernel.googlesource.com][1])
* **Upstream kernel logs/patch** showing 0x02140201 → `qca/crbtfw21.tlv` for WCN3990. ([lkml.iu.edu][2], [lkml.indiana.edu][7])
* **DT binding doc** for `qcom,wcn3990-bt`, including `firmware-name` property. ([Android Git Repositories][3])
* **Distro file lists** confirming paths (`/lib/firmware/qca/crbtfw21.tlv`, `/lib/firmware/qca/crnv21.bin`). ([GitLab][6])

If you want, drop me the exact `dmesg` around the firmware load and I’ll tell you instantly whether your 4.9 is using the old Rome loader or the newer WCN3990 TLV flow.

[1]: https://kernel.googlesource.com/pub/scm/linux/kernel/git/firmware/linux-firmware/%2B/034e24b835d1f8b8e8266f766d610683aefe14c3/WHENCE "WHENCE - pub/scm/linux/kernel/git/firmware/linux-firmware - Git at Google"
[2]: https://lkml.iu.edu/1903.1/04409.html?utm_source=chatgpt.com "[PATCH v2 2/2] Bluetooth: hci_qca: wcn3990 ... - Linux-Kernel Archive"
[3]: https://android.googlesource.com/kernel/common/%2B/fb73974172ff/Documentation/devicetree/bindings/net/qualcomm-bluetooth.txt?utm_source=chatgpt.com "qualcomm-bluetooth.txt"
[4]: https://www.suse.com/support/update/announcement/2019/suse-ru-20191177-1/?utm_source=chatgpt.com "Recommended update for kernel-firmware"
[5]: https://www.rpmfind.net/linux/RPM/opensuse/15.5/noarch/kernel-firmware-platform-20230320-150500.1.1.noarch.html?utm_source=chatgpt.com "kernel-firmware-platform-20230320-150500.1.1.noarch RPM"
[6]: https://git.codelinaro.org/clo/linux-kernel/linux-firmware/-/tree/video-firmware/qca?ref_type=heads&utm_source=chatgpt.com "qca · video-firmware - linux-kernel - GitLab - CodeLinaro"
[7]: https://lkml.indiana.edu/1905.2/05260.html?utm_source=chatgpt.com "hci_qca: wcn3990: Drop baudrate change vendor event"
[8]: https://chromium.googlesource.com/chromiumos/third_party/kernel/%2B/refs/heads/firmware-trogdor-13577.B-chromeos-5.4/drivers/bluetooth/hci_qca.c?utm_source=chatgpt.com "drivers/bluetooth/hci_qca.c - chromiumos/third_party/kernel"
[9]: https://lkml.iu.edu/hypermail/linux/kernel/1806.3/00904.html?utm_source=chatgpt.com "[PATCH v8 0/7] Enable Bluetooth functionality for WCN3990"
