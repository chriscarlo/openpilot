Got it. You’re tripping over the *very* specific WCN3990 boot/handshake quirks that upstream solved in the serdev path, but that don’t exist (by default) in the old line‑discipline flow. Below is the exact sequence and the minimal kernel/userspace tweaks that consistently get past `0xfc00` and download `crbtfw21.tlv`/`crnv21.bin` on SDM845/WCN3990 with Linux 4.9 + `hci_qca` over UART.

---

## TL;DR (what to change)

1. **Use the WCN3990 “power pulses”, *not* an IBS wake**

   * **Power‑off pulse:** one byte **0xC0** at **2400 bps**, flow control **disabled**, then **sleep \~10 ms**.
   * **Power‑on pulse:** one byte **0xFC** at **115200 bps**, flow control **disabled**, then **sleep \~100 ms**.
     These are *not* HCI/IBS messages—they’re raw bytes on TX used by an external decode circuit. Upstream `hci_qca` does exactly this for WCN3990 and explicitly says to disable HW flow control for the pulses. ([Chromium Code Search][1], [Android Git Repositories][2])

2. **Order of operations at boot (UART + firmware)**
   Do the pulses, then read ROM at 115200, then bump the baud, then do the TLV download:

   * Set host to **2400 bps**, send **0xC0** (power‑off), **10 ms** delay.
   * Set host to **115200 bps**, send **0xFC** (power‑on), **100 ms** delay, **re-open the port** to re‑sync.
   * Now at 115200, issue **EDL\_PATCH\_VER\_REQ (0xFC00/0x19)** → read ROM **0x02140201**.
   * Bump both sides to **3.0–3.2 Mbps** (driver sends QCA vendor baud command, then host rate).
   * Download **`qca/crbtfw21.tlv`** patch, then **10 ms delay**, then **`qca/crnv21.bin`**.
     This is exactly how upstream `hci_qca` sequences WCN399x: version at init‑speed, *then* change to oper‑speed, *then* TLV download, with a 10 ms pause before NVM. ([Chromium Code Search][1], [Android Git Repositories][3], [96Boards Lists][4])

3. **Timeouts/retries that matter on 4.9**

   * **Baud change settle:** WCN399x only needs **\~10 ms** after the vendor baud command; upstream also *waits* up to **100 ms** for a “baudrate changed” vendor event and drops that stray event so it doesn’t corrupt the TLV flow. Port that behavior. ([Chromium Code Search][1])
   * **NVM stage delay:** add **10 ms** before NVM download (fixes “TLV response size mismatch” flakiness). You’ve already added this—good. It’s an upstream‑backported 4.9 stable fix. ([96Boards Lists][4])
   * **Init command timeout:** `qca_read_soc_version()` sends **EDL 0xFC00** with **`HCI_INIT_TIMEOUT`**; if you still get `tx timeout`, nudging that constant upward in 4.9 can help on slow boots. (The call path is `__hci_cmd_sync_ev(..., HCI_INIT_TIMEOUT)`.) ([Android Git Repositories][3])

4. **Segment size:** use the upstream TLV segment size **243** bytes, not 240. (Upstream `btqca.h`: `MAX_SIZE_PER_TLV_SEGMENT 243`.) ([Android Git Repositories][2])

5. **Regulators/DT (WCN3990 UART path, not GLINK):**
   Required supplies for `qcom,wcn3990-bt` are **vddio, vddxo, vddrf, vddch0**; an enable/reset GPIO is **not required** in the upstream binding for WCN3990 (different from the old vendor `bt_power` binding). The example shows **`&uart6`** with `max-speed = <3200000>` and no reset GPIO. ([Kernel.org][5])

6. **UART instance:** SDM845/SDM850 reference designs commonly put BT on **UART6 (QUPv3 SE6)**. Example: Lenovo **SDM850 Yoga C630** DT binds `qcom,wcn3990-bt` under `&uart6` with `max-speed=3200000`. That aligns with your current `ttyHS0 (SE6)`. ([Kernel Git Repositories][6])

---

## Minimal, robust bring‑up sequence (usable with `btattach`)

> If you can switch to **serdev**, do it—upstream `hci_qca`’s WCN3990 path (serdev) implements the pulses, reopens the port after power‑on, manages regulators/32 kHz susclk, drops the stray baud‑change vendor event, and sets the exact delays. With **line discipline**, you have to emulate that. ([Chromium Code Search][1])

**A. One‑time kernel adjustments (you already did most of this):**

* Map ROM 0x02140201 → **`qca/crbtfw21.tlv`** + **`qca/crnv21.bin`** (the “21” pair). TLV is used **as‑is**. (Upstream drivers use those filenames for ROM 0x21xx; you verified the mapping is right.)
* Use **MAX\_SIZE\_PER\_TLV\_SEGMENT = 243**.
* Keep the **10 ms** delay before NVM.
* Keep the “drop vendor 0x92 baud‑change event” quirk (make it conditional on WCN3990). All are upstream behaviors. ([Android Git Repositories][2], [Linux Kernel Archive][7], [96Boards Lists][4])

**B. Userspace pre‑flight (emulate serdev’s pulses):**

1. Ensure regulators/clocks are on (your `bt_power` driver or DT regulators already do this).
2. **Close** anything using the port.
3. **Set /dev/ttyHS0 to 2400 bps, no flow control**, and write **0xC0** (power‑off). Sleep **10 ms**.
4. **Set /dev/ttyHS0 to 115200 bps, no flow control**, and write **0xFC** (power‑on). Sleep **100 ms**.
5. **Close and reopen** `/dev/ttyHS0` (serdev does this to re‑sync RTS/CTS).

   * Rationale and exact timing are taken straight from `qca_wcn3990_init()`. ([Chromium Code Search][1])

(If you’d rather keep this in‑kernel, port the tiny `qca_wcn3990_init()` bits: set host to 2400, send **0xC0** (FC off), set **init\_speed** (115200), send **0xFC** (FC on), re‑open port, and don’t forget to **disable flow control while sending pulses**.)

**C. Attach at init speed and let the driver do the rest:**

```bash
rfkill unblock bluetooth
# Keep init at 115200 so we read ROM at the boot speed
btattach -B /dev/ttyHS0 -S 115200 -P qca
# hci_qca should:
#  - send EDL 0xFC00 (Read ROM) at 115200
#  - bump to 3.0–3.2 Mbps (vendor cmd + host change, wait up to ~100 ms for vendor event)
#  - download crbtfw21.tlv, wait 10 ms, then crnv21.bin
```

**Don’t** send an **IBS “wake” (0xFD)** here—that’s an HCI\_IBS control byte, not the WCN3990 power pulse. Power pulses are **0xC0/0xFC** at specific baud rates with flow control disabled. IBS still runs later for runtime power mgmt, but it’s irrelevant for the *boot* sequence. ([Chromium Code Search][1], [Android Git Repositories][2])

---

## Answers to your numbered questions

**1) Correct ordering (IBS/0xFD vs EDL 0xFC00 vs baud change):**

* For WCN3990, **don’t** rely on IBS for bring‑up. First do the **power pulses** (0xC0\@2400, then 0xFC\@115200, with the 10/100 ms sleeps and a port reopen). Then:

  1. **Read ROM** with **EDL\_PATCH\_VER\_REQ (0xFC00/0x19)** at **115200**.
  2. **Change baud** to **3.0–3.2 Mbps** (driver sends vendor baud, waits for baud‑change vendor event up to **100 ms**, then host speed change; include the **\~10 ms** settle for WCN399x).
  3. **Download TLV patch**, then **10 ms**, then **NVM**.
     That is the upstream `hci_qca` sequence for WCN399x. ([Chromium Code Search][1])

**2) Is a “0xFD power‑on pulse” sufficient?**
No. **0xFD** is **IBS WAKE\_IND**. The **WCN3990 power‑on pulse is 0xFC**, power‑off is **0xC0**; both must be sent as **raw bytes** with HW flow control disabled and at the specified baud rates. ([Android Git Repositories][2], [Chromium Code Search][1])

**3) Known 4.9‑era patches/timeouts:**

* **Drop the stray 0x92** vendor event after baud change (WCN3990‑only). Prevents TLV size mismatch during FW download. (You’ve already implemented this.) ([Linux Kernel Archive][7])
* **10 ms delay before NVM** to avoid TLV mismatch (upstream, backported to 4.9 stable). (You added this—good.) ([96Boards Lists][4])
* **Baud settle**: For WCN399x, driver uses **\~10 ms** delay after sending the baud command and **100 ms** wait for the vendor event. Use those values. ([Chromium Code Search][1])
* **Init timeout**: `qca_read_soc_version()` uses **`HCI_INIT_TIMEOUT`** for 0xFC00; if your platform is slow, bump that constant in 4.9. ([Android Git Repositories][3])

**4) Power/reset lines (“bt-reset-gpio” and “chip‑pwd”):**

* **Upstream serdev binding for WCN3990** requires only the regulators and 32 kHz clock; *no reset GPIO is required*. (Old vendor `bt_power` did define `qca,bt-reset-gpio` as “required”, but that’s not how upstream WCN3990 works.) ([Kernel.org][5], [Android Git Repositories][8])
* **`qca,bt-chip-pwd-supply`** is an optional vendor `bt_power` supply when Wi‑Fi/BT share a power‑down pin; upstream `hci_qca` for WCN3990 doesn’t use it. You can leave it out unless your `bt_power` driver truly needs it for your board. ([Android Git Repositories][8])

**5) UART instance and flow control:**

* On SDM845/SDM850 references, BT is commonly on **UART6 (SE6)**. The **Yoga C630** and various SDM845 boards wire `qcom,wcn3990-bt` under `&uart6` with `max-speed = <3200000>`. So **SE6 is the sane default** for BT on SDM845 boards (matching your `ttyHS0`). ([Kernel Git Repositories][6])
* **CTS/RTS** must be active for 3M/3.2M operation (H4+IBS). Upstream driver *temporarily* disables flow control only for the **power pulses** and while changing baud on WCN399x; the rest of the time, HW flow control is on. If you want a diagnostic, a quick `-N` (no‑flow‑control) attach may show life at 115200, but don’t expect stable high‑speed operation without CTS/RTS. ([Chromium Code Search][1])

**6) serdev vs `btattach`:**

* **Yes, enable serdev** (`CONFIG_SERIAL_DEV_BUS`, bind child `bluetooth` node under `&uart6` with `compatible = "qcom,wcn3990-bt"`). The **serdev** path picks **QCA\_WCN3990** (not “ROME”), performs the **0xC0/0xFC pulses**, **reopens the port**, manages regulators and the **32 kHz SUSCLK** vote, handles the **baud‑change vendor event**, and applies the **WCN399x‑specific delays**. Line discipline can work, but you’ll be re‑implementing all of that yourself. ([Chromium Code Search][1])

**7) Any other robustness patches for `0xfc00` timeouts?**

* Aside from the above, nothing special is needed. Most “can’t read ROM / `0xfc00` timeout” reports are from **missing power pulses** or **desync due to early host‑only baud bump**. Ensuring the **pulses + reopen** and using the **WCN399x path** behavior (drop vendor 0x92, proper delays) resolves it. ([Chromium Code Search][1], [Linux Kernel Archive][7])

**8) Debugging tips (to verify 115200 traffic):**

* Put the controller in the ready state (pulses done), keep port at 115200, and send a simple HCI command like **HCI\_Reset** (01 03 0C 00) *before* starting `btattach`; you should see an event back (0x04 …). If nothing comes back, your **power/enable stage** isn’t complete (typically: missed pulses or flow‑control stuck asserted). The upstream docs/code comments make clear that **BT is not enabled until the 115200 power‑on pulse is sent**. ([Chromium Code Search][1])

---

## Device‑tree checklist for UART WCN3990 (matches upstream)

Under **`&uart6`** (QUPv3 SE6):

```dts
&uart6 {
    status = "okay";
    bluetooth: bt@0 {
        compatible = "qcom,wcn3990-bt";
        vddio-supply = <&pm8998_s3>;   // 1.8V
        vddxo-supply = <&pm8998_l7>;   // 1.8V
        vddrf-supply = <&pm8998_l17>;  // 1.3V
        vddch0-supply = <&pm8998_l25>; // 3.3V
        max-speed = <3200000>;         // 3.2 Mbps
        /* SUSCLK (32k) via clocks property if present on your platform */
    };
};
```

This mirrors the **official binding** (supplies + max‑speed; no reset GPIO required). ([Kernel.org][5])

---

## Why your current logs look the way they do

* Seeing **“Set UART speed to 3200000”** *before* any successful ROM read is a red flag: the host is jumping to 3.2 M while the controller is still at 115200 and not yet “enabled” by the **0xFC** power‑on pulse. That’s how you end up with `0xfc00 tx timeout` and then `-110`. Upstream waits for ROM at 115200 first, *then* bumps the baud, with the **10 ms** settle and **vendor event** wait logic. ([Chromium Code Search][1])
* The fact that removing `bt-reset-gpio` suppressed a stuck reset is consistent with WCN3990 designs that **don’t use** a BT reset GPIO for UART bring‑up (power rails + XO + pulses only). That matches upstream DT for WCN3990. ([Kernel.org][5])

---

## If you can switch to serdev (recommended)

* Enable `CONFIG_SERIAL_DEV_BUS=y` and the serdev path in `hci_qca`.
* Move your BT node under `&uart6` as above (`qcom,wcn3990-bt`).
* Let `hci_qca` do the **pulses** + **reopen** + **delays** automatically; it also bulk‑enables the regulators and the **32 kHz** clock if you provide a `clocks` handle. This is the upstream‑supported path for WCN3990 and will remove a lot of the fragility you’re currently fighting with `btattach`. ([Chromium Code Search][1])

---

### Sources (key specifics)

* WCN3990 power‑pulse mechanism (0xC0\@2400 “off”, 0xFC\@115200 “on”), **disable flow control** during pulses; **100 ms** after “on”; **10 ms** after “off”; reopen port after power‑on: upstream `hci_qca.c`. ([Chromium Code Search][1])
* Vendor EDL flow (`0xFC00` + `0x19`), uses **`HCI_INIT_TIMEOUT`**: `btqca.c`. ([Android Git Repositories][3])
* **Baud‑change** handling for WCN399x: \~**10 ms** settle; **100 ms** wait for vendor event; **drop stray 0x92** so TLV download isn’t confused. ([Chromium Code Search][1], [Linux Kernel Archive][7])
* **10 ms** delay before NVM (fixes TLV mismatch), backported to 4.9: stable patch. ([96Boards Lists][4])
* Official **DT binding** for `qcom,wcn3990-bt` (supplies, `max-speed`, example under **`&uart6`**; no reset GPIO required): kernel.org YAML. ([Kernel.org][5])
* Real board example with `&uart6` + WCN3990 (Lenovo **SDM850 Yoga C630**): mainline DT. ([Kernel Git Repositories][6])

---

### Actionable next steps for your tree

1. **Replace** your “IBS wake pulse 0xFD” hack with the **two power pulses** (`0xC0` @ 2400; `0xFC` @ 115200), with HW flow **disabled** while sending and **100 ms** after the power‑on pulse, then **close/reopen** the port. (If you stick with line discipline, do this from userspace just before `btattach`.) ([Chromium Code Search][1])
2. Make sure your `hci_qca` **does not** bump to 3.2 M before a successful ROM read at 115200. Follow the upstream order. ([Chromium Code Search][1])
3. Keep **`crbtfw21.tlv`** / **`crnv21.bin`** and set **TLV segment size = 243**. ([Android Git Repositories][2])
4. If timeouts remain on the very first `0xfc00`, slightly **raise `HCI_INIT_TIMEOUT`** in 4.9. ([Android Git Repositories][3])
5. Consider **serdev** for a permanent, low‑maintenance solution. It exactly implements the above and removes most race conditions. ([Chromium Code Search][1])

If you want, I can sketch a tiny shell helper that performs the two pulses + reopen around `/dev/ttyHS0` so you don’t have to carry more kernel patches.

[1]: https://chromium.googlesource.com/chromiumos/third_party/kernel/%2B/refs/heads/firmware-trogdor-13577.B-chromeos-5.4/drivers/bluetooth/hci_qca.c "drivers/bluetooth/hci_qca.c - chromiumos/third_party/kernel - Git at Google"
[2]: https://android.googlesource.com/kernel/common/%2B/5e99456c20f7/drivers/bluetooth/btqca.h "drivers/bluetooth/btqca.h - kernel/common - Git at Google"
[3]: https://android.googlesource.com/kernel/common/%2B/30bac164aca7/drivers/bluetooth/btqca.c?utm_source=chatgpt.com "drivers/bluetooth/btqca.c - kernel/common - Git at Google"
[4]: https://lists-ec2.96boards.org/archives/list/linux-stable-mirror%40lists.linaro.org/message/BPLNKTWYA7N3X3TERMPAYVPTL73SG6UZ/?utm_source=chatgpt.com "[PATCH 4.9 03/26] Bluetooth: btqca: Add a short delay ..."
[5]: https://www.kernel.org/doc/Documentation/devicetree/bindings/net/bluetooth/qualcomm-bluetooth.yaml "www.kernel.org"
[6]: https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/%2B/f40ddce88593482919761f74910f42f4b84c004b/arch/arm64/boot/dts/qcom/sdm850-lenovo-yoga-c630.dts?utm_source=chatgpt.com "arch/arm64/boot/dts/qcom/sdm850-lenovo-yoga-c630.dts"
[7]: https://lkml.iu.edu/1903.1/04409.html?utm_source=chatgpt.com "hci_qca: wcn3990: Drop baudrate change vendor event"
[8]: https://android.googlesource.com/kernel/msm/%2B/android-7.1.0_r0.2/Documentation/devicetree/bindings/bluetooth/bluetooth_power.txt?utm_source=chatgpt.com "Documentation/devicetree/bindings/bluetooth ..."
