Short answer (what to drop in `/lib/firmware`):

* **Patch (RAM image):** `qca/crbtfw21.tlv`
* **NVM (configuration):** `qca/crnv21.bin`

Those two files are the **exact** pair the upstream Linux `hci_qca/btqca` stack uses for **WCN3990** when the controller reports **ROM/“controller version” 0x02140201** (the “21” is the ROM index used to select `…21.*`). They live in the top‑level `qca/` directory of the linux‑firmware tree and are consumed **as‑is** (TLV wrapped; no conversion). ([Code Browser][1], [Kernel Git Repositories][2], [Chromium Git Repositories][3])

---

## Why these filenames (and not `rampatch_02140201.bin`)?

Current kernels derive a one‑byte “ROM version” from the controller’s version and, for WCN399x, build firmware names like:

* **Patch:** `qca/crbtfw%02x.tlv`
* **NVM:**  `qca/crnv%02x[ u ].bin` (the `u` suffix is **only** for WCN3991)

So for a WCN3990 that logs `…version 0x02140201`, the ROM index is `0x21` → `crbtfw21.tlv` + `crnv21.bin`. This logic is in `drivers/bluetooth/btqca.c`. ([Code Browser][1])

> In contrast, **very old** kernels (or trees lacking the WCN399x special‑case) fall back to the legacy **Rome** naming, e.g. `qca/rampatch_%08x.bin` and `qca/nvm_%08x.bin`. If your 4.9 tree prints it is trying **`rampatch_02140201.bin`**, that’s a tell that WCN3990‑aware naming wasn’t backported.

Upstream linux‑firmware has shipped the WCN399x pair for years (look for `qca/crbtfw21.tlv` and `qca/crnv21.bin` in packages / repos). ([Kernel Git Repositories][2], [pkgs.alpinelinux.org][4])

---

## Exact files in linux‑firmware (paths you can vendor)

* `qca/crbtfw21.tlv`  — WCN3990 RAM patch (TLV)
* `qca/crnv21.bin`    — WCN3990 NVM config (TLV)

You can see them in the `WHENCE` manifest and in common distro packages mirroring upstream linux‑firmware:
• WHENCE lists WCN399x → `crbtfw21.tlv`, `crnv21.bin` (+ `32`/`32u` for 3991). ([Kernel Git Repositories][2])
• ChromiumOS firmware mirror shows the same files under `qca/`. ([Chromium Git Repositories][3])
• Distros note “qca: Add firmware files for BT chip wcn3990” (e.g., 2019 change logs). ([openSUSE Build Service][5])

Place them where the kernel looks (in order): `/lib/firmware/updates/qca/` or `/lib/firmware/qca/` (you said you bind‑mount `/data/firmware` → either layout works; `request_firmware()` will prefer `updates/` first).

---

## If your 4.9 kernel still asks for `rampatch_02140201.bin`

You have two options:

1. **Preferable — backport the WCN399x naming support.**
   The WCN3990 enablement landed upstream around mid‑2018 and evolves in 2019 with robustness fixes. Backporting the WCN399x bits in `hci_qca/btqca` lets the driver request `crbtfw21.tlv`/`crnv21.bin` directly, and pulls in stability fixes you’ll want anyway. (Examples: drop the stray baud‑change vendor event; add a short delay before NVM.) ([Linux-Kernel Archive][6], [lists-ec2.96boards.org][7])

2. **Pragmatic shim — provide legacy names as symlinks/copies** (works on old drivers that only ever try the “Rome” pattern, which still parses TLV payloads):

```bash
# Patch (RAM image) – point the legacy name at the TLV file
ln -sf crbtfw21.tlv /lib/firmware/qca/rampatch_02140201.bin

# NVM – same idea for NVM
ln -sf crnv21.bin    /lib/firmware/qca/nvm_02140201.bin
```

Newer btqca uses `crnv%02x.bin`, but older fallbacks use `nvm_%08x.bin`; this covers either code path. (Modern btqca’s default fallback shown here: `…/btqca.c` formats `nvm_%08x.bin` if it doesn’t hit a SoC‑specific case.) ([Code Browser][8])

---

## Do you need to unwrap/convert the files?

No. The driver expects **TLV** for WCN399x. You should place `crbtfw21.tlv` and `crnv21.bin` **as shipped**. In the driver you can see `config.type = TLV_TYPE_PATCH` for the patch stage, then the NVM stage uses the TLV reader as well. ([Code Browser][1])

---

## Sub‑revisions and picking the right pair

* **WCN3990 → ROM 0x21 →** `crbtfw21.tlv` + `crnv21.bin`
* **WCN3991 → ROM 0x32 →** `crbtfw32.tlv` + `crnv32u.bin` (note the **`u`** variant for 3991)
  The `btqca` switch‑case shows the mapping and the `u` suffix logic. Your controller is reporting 0x02140201, which selects the **21** pair. ([Code Browser][1])

---

## UART speed / “bonus” question

* **WCN3990 needs a “power‑on pulse” at 115200 bps.** That’s a one‑byte pulse used to wake the controller; you’ll see this in `hci_qca.c` comments. ([Android Git Repositories][9])
* **Patch/NVM don’t have to run at 115200.** The in‑tree driver will typically switch to the **operating speed (default 3,000,000 bps)** **before** downloading firmware, and it contains special handling for WCN3990’s baud‑change vendor event to avoid “frame reassembly / TLV size mismatch” errors. No requirement to keep patch at 115200 — the driver handles the transition. ([Android Git Repositories][9], [Linux-Kernel Archive][6])

If you’re staying on 4.9, I strongly recommend you also pick the upstream WCN3990 robustness fixes:

* **Drop stray baud‑change vendor event** (prevents TLV size mismatch during fw download). ([Linux-Kernel Archive][6])
* **Add 10 ms delay before NVM download** (prevents `TLV response size mismatch` after patch). ([lists-ec2.96boards.org][7])

---

## Quick checklist for your setup

1. Put **`crbtfw21.tlv`** and **`crnv21.bin`** in `/lib/firmware/qca/` (or the `updates/` override). ([Chromium Git Repositories][3])
2. If your dmesg shows it still wants `rampatch_02140201.bin`, either backport the WCN399x naming support or add the two **symlinks** shown above. ([Code Browser][8])
3. Confirm in dmesg that you now see lines like:
   `QCA Downloading qca/crbtfw21.tlv` → then `qca/crnv21.bin`; if you still see “TLV response size mismatch”, pull in the two fixes noted above. ([Linux-Kernel Archive][6], [lists-ec2.96boards.org][7])

If you want, paste the exact **`dmesg | grep -i qca -n`** bits you get after placing the files and I’ll sanity‑check the sequence.

[1]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html "btqca.c source code [linux/drivers/bluetooth/btqca.c] - Codebrowser "
[2]: https://kernel.googlesource.com/pub/scm/linux/kernel/git/firmware/linux-firmware/%2B/034e24b835d1f8b8e8266f766d610683aefe14c3/WHENCE?utm_source=chatgpt.com "WHENCE - pub/scm/linux/kernel/git/firmware ..."
[3]: https://chromium.googlesource.com/chromiumos/third_party/linux-firmware/%2B/refs/heads/master/qca/?utm_source=chatgpt.com "qca - chromiumos/third_party/linux-firmware"
[4]: https://pkgs.alpinelinux.org/contents?arch=x86&branch=v3.16&name=linux-firmware-qca&repo=main&utm_source=chatgpt.com "Package index - Alpine Linux packages"
[5]: https://build.opensuse.org/projects/SUSE%3ASLE-15-SP3%3AUpdate/packages/kernel-firmware.30186/files/kernel-firmware.changes?expand=0&utm_source=chatgpt.com "File kernel-firmware.changes of Package ..."
[6]: https://lkml.indiana.edu/1905.2/05260.html?utm_source=chatgpt.com "hci_qca: wcn3990: Drop baudrate change vendor event"
[7]: https://lists-ec2.96boards.org/archives/list/linux-stable-mirror%40lists.linaro.org/message/BPLNKTWYA7N3X3TERMPAYVPTL73SG6UZ/?utm_source=chatgpt.com "[PATCH 4.9 03/26] Bluetooth: btqca: Add a short delay ..."
[8]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html?utm_source=chatgpt.com "btqca.c source code [linux/drivers/bluetooth ..."
[9]: https://android.googlesource.com/kernel/common/%2B/34d3a78c681e8/drivers/bluetooth/hci_qca.c?autodive=0%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F "drivers/bluetooth/hci_qca.c - kernel/common - Git at Google"
