Below is a **battle‑tested bring‑up recipe** for WCN3990 (ROM **0x02140201**) on SDM845 with **Linux 4.9 + hci\_qca over UART (btattach)**. It’s tailored to the exact symptoms you reported (early 3.2 Mbaud bump + `0xfc00` TX timeout) and to the canonical Qualcomm driver behavior for WCN39xx.

---

## TL;DR (what’s actually biting you)

1. **WCN3990 requires a 1‑byte “power pulse” over the UART before it will answer 0xFC00.**
   That pulse is **0xFC at 115200 bps** (and in some sequences an optional **0xC0 at 2400 bps** as a “power‑off” pre‑pulse). While IBS `0xFD` is a *wake* byte, it is **not** the WCN3990 power‑on pulse. The upstream driver documents sending these pulses with **CTS/RTS temporarily disabled** and waiting **\~100 ms** after the **0xFC** ON pulse. ([Code Browser][1], [Android Git Repositories][2])

2. **ROM/version read must happen at the init speed (115200).** Only **after** reading the ROM (0xFC00) should you switch both the controller and host to **3.2 Mbps** (with a short settle wait **1–10 ms** on WCN399x, not 300 ms). Upstream moved to a short `usleep_range(1000, 10000)` delay for these chips. ([Code Browser][1])

3. **WCN399x emits a vendor event at the new baud (0x92) when changing speed.** Upstream’s hci\_qca sets a “drop vendor event” flag and waits **up to \~100 ms** for it so that it doesn’t confuse the setup state machine. If you’re on an older 4.9 snapshot, this handling may be missing—add it, or your “early bump” will de‑sync. ([Code Browser][1])

4. **NVM stage needs a tiny pause.** Upstream inserts a **10 ms** sleep between patch and NVM TLV stages for WCN39xx. Keep it. ([Code Browser][3])

---

## Exact, minimal sequence (4.9, line discipline, `btattach`)

This assumes your PM8998 rails and 32 kHz clock vote are already up (your `bt_power` node looks fine). We’ll emulate the **serdev-only power pulses** from userspace since the 4.9 line‑discipline path doesn’t send them.

1. **Power rails / clocks**
   Enable the regulators (`qca,bt-vdd-*`) and the 32 kHz SUSCLK/XO vote as you already do. No reset GPIO is needed on SDM845 WCN3990 reference designs (see DT section below). ([Android Git Repositories][4], [Kernel.org][5])

2. **(Optional but robust) “power‑off” pre‑pulse**
   Set the UART to **2400 8N1**, **disable CTS/RTS**, send **`0xC0`**, then wait **1–10 ms**:

> Reason: a short OFF pulse before ON substantially improved first‑try bring‑up on WCN3990; upstream added a delay after the OFF pulse to avoid the exact `0xfc00` timeout you see. ([Linux-Kernel Archive][6])

3. **“Power‑on” pulse (mandatory)**
   Set the UART to **115200 8N1**, **CTS/RTS disabled**, send **`0xFC`**, wait **\~100 ms**, **re‑enable CTS/RTS**. ([Code Browser][1])

   *One‑liners if you like:*

   ```bash
   stty -F /dev/ttyHS0 2400 -echo -crtscts; printf '\xC0' > /dev/ttyHS0; usleep 10000
   stty -F /dev/ttyHS0 115200 -echo -crtscts; printf '\xFC' > /dev/ttyHS0; sleep 0.1
   stty -F /dev/ttyHS0 115200 -echo crtscts   # re-enable HW flow control
   ```

4. **Attach at init speed**
   `btattach -B /dev/ttyHS0 -S 115200 -P qca`
   Let hci\_qca drive IBS; don’t send your own `0xFD` after this—IBS wake/ack (0xFD/0xFC) is handled by the driver during TX queueing. (Constants: WAKE\_IND=0xFD, WAKE\_ACK=0xFC, SLEEP\_IND=0xFE.) ([Chromium Code Search][7])

5. **ROM/version read (still 115200)**
   Driver sends **EDL\_PATCH\_VER\_REQ** (vendor **0xFC00**) and prints the ROM you already see (`0x02140201`). If this times out, **the power pulse step above did not reach the SoC** or HW flow control remained asserted. ([GitLab][8])

6. **Switch to 3.2 Mbps the *upstream* way**
   – Send the vendor **Set Baudrate** command (**OCF 0x48 / OGF 0x3F**) with argument matching 3.2 M.
   – Immediately change the host port to **3,200,000**.
   – **Wait \~1–10 ms** on WCN399x for settle; **ignore/drop** the vendor event (**0x92**) that arrives at the **new** speed. Upstream hci\_qca sets `QCA_DROP_VENDOR_EVENT` and waits up to \~**100 ms** for completion. (Your “drop 0x92” tweak is the right idea—just add the short delay instead of 300 ms.) ([Code Browser][1], [Mail Archive][9], [LKML][10])

7. **Download firmware (still 3.2 M)**
   – **Patch**: `qca/crbtfw21.tlv`
   – **NVM**: `qca/crnv21.bin`
   – **TLV segment size**: upstream uses **243**; 240 also works—both are common in vendor trees.
   – **Delay**: **msleep(10)** between patch and NVM to avoid “TLV response size mismatch” corner cases. ([Code Browser][3], [Android Git Repositories][11])

8. **HCI Reset + finish**
   Driver issues `HCI_Reset` and prints “QCA setup on UART is completed”. ([Code Browser][3])

> If you still see `command 0xfc00 tx timeout`, repeat steps **2–4** (issue OFF/ON pulses again), then re‑run `btattach`. This is exactly what upstream stabilized via the OFF‑then‑ON pulse sequence and added delays. ([Linux-Kernel Archive][6])

---

## Answers to your numbered questions

**1) Correct ordering (IBS / 0xFC00 / Baud change) & delays**

* **Order**: (a) Power pulse(s) → (b) attach at 115200 → (c) **0xFC00** ROM read @115200 → (d) baud change to **3.2 M** (controller first, then host, **1–10 ms** delay, drop vendor event) → (e) TLV patch → 10 ms → TLV NVM → HCI reset. ([Code Browser][1])
* **ROM read absolutely at 115200**; do **not** bump before reading ROM on WCN3990.
* **Delays**: \~**100 ms** after ON pulse; **1–10 ms** after baud change; **10 ms** before NVM. ([Code Browser][1], [LKML][10])

**2) Is `0xFD` enough?**
No. `0xFD` is **IBS WAKE\_IND**, not the WCN3990 power‑on pulse. The **power‑on pulse is `0xFC` at 115200 bps**; some sequences also send **`0xC0` at 2400 bps** just before to ensure a clean boot window. CTS/RTS must be **disabled** for the pulse, then re‑enabled immediately afterward. ([Code Browser][1])

**3) 4.9‑era patches that help**

* **Shorten baud settle**: replace 300 ms with **`usleep_range(1000,10000)`** for WCN399x. ([LKML][10])
* **Add delay after OFF pulse** (prevents initial `0xfc00` timeout). ([Linux-Kernel Archive][6])
* **Drop vendor 0x92 and wait \~100 ms completion** during baud change. (Upstream uses `QCA_DROP_VENDOR_EVENT`.) ([Code Browser][1])
* **(Optional)** Vendor optimization: inject a synthetic “command complete” for last TLV packet to avoid noisy timeouts on some SoC revs. ([Mail Archive][12])

**4) Power/reset lines**

* **Reset GPIO**: Typically **not used** on SDM845+WCN3990 (QRD/MTP/Cheza style). Bring‑up is by rails + power pulse; your seeing `bt-reset-gpio not provided` is normal. ([Android Git Repositories][4])
* **`qca,bt-chip-pwd-supply`**: Optional. It exists in older Qualcomm “bluetooth-power” bindings and is only used on some Wi‑Fi/BT module combos. Most SDM845 WCN3990 designs omit it; the main supplies are I/O/XO/RF/CH0. ([Android Git Repositories][13], [Kernel.org][5])

**5) UART instance (SE6 vs SE7) & flow control**

* **Reference mapping**: SDM845 boards overwhelmingly use **QUPv3 SE6 / `uart6`** for BT. Examples: **cheza** (`&uart6` + `wcn3990-bt`) and **SDM845 QRD** enabling `&qupv3_se6_4uart`. So your **SE6/ttyHS0** choice is aligned with reference designs. ([Linux Kernel Archive][14], [Android Git Repositories][4])
* **CTS/RTS**: **Required** for normal operation. Only disable it briefly for the **power pulses** and during the **baud‑switch window** (as upstream does). Don’t run `btattach -N` except as a quick sanity check for whether a single byte reaches the SoC. ([Code Browser][1])

**6) serdev vs btattach on 4.9**

* **Yes—serdev helps.** The upstream serdev path **sends the OFF/ON pulses, toggles HW flow for the baud switch, and waits for the 0x92 vendor event**. On plain 4.9 line‑discipline this logic isn’t there; that’s why you’re timing out at `0xfc00`. If you can’t backport full serdev, **mimic the power‑pulse & short‑delay behavior from userspace exactly as above**. ([Code Browser][1])

**7) Any other robustness patches for `0xfc00` timeouts?**

* **Frame reassembly** fixes for stray boot bytes on WCN3990 (older trees would occasionally misparse a few bytes at boot). Not strictly required, but reduces false errors at bring‑up. ([lkml.rescloud.iu.edu][15])

**8) Low‑level debug if you suspect UART is quiet at 115200**

* Before running `btattach`, set 115200, disable CTS/RTS, send **`0xFC`**, wait **100 ms**, enable CTS/RTS, attach. If `0xfc00` still times out, try **OFF** (`0xC0` @ 2400) → **ON** (`0xFC` @ 115200) again. You can also try sending a raw **HCI Reset** (`01 03 0C 00`) after the power pulse to see if you get any event bytes back. The need for OFF→ON pulses and a small delay before version query is documented in the upstream fixes that specifically target the `0xfc00` timeout. ([Linux-Kernel Archive][6])

---

## Firmware mapping (you already picked the right pair)

* **ROM 0x02140201 (WCN3990)** → **`qca/crbtfw21.tlv`** (patch) + **`qca/crnv21.bin`** (NVM). File names and ROM→name selection are hard‑coded in upstream `btqca.c`. No conversion needed; they are TLV/NVM as‑is. ([Code Browser][3])

* **Segment size**: upstream defines **`MAX_SIZE_PER_TLV_SEGMENT` = 243** in some Qualcomm kernels; 240 also works and is used by various 4.9 vendor trees. Keep `10 ms` before NVM. ([Android Git Repositories][11])

---

## Device‑tree reference & rails checklist (SDM845 + WCN3990)

* Modern binding (mainline) for UART‑attached WCN3990 uses `compatible = "qcom,wcn3990-bt"` under the UART node, supplies `vddio/vddxo/vddrf/vddch0`, and an optional `max-speed = <3200000>`. Example (Xiaomi SDM845):

  ```dts
  bluetooth: wcn3990-bt {
      compatible = "qcom,wcn3990-bt";
      vddio-supply = <&vreg_s4a_1p8>;
      vddxo-supply = <&vreg_l7a_1p8>;
      vddrf-supply = <&vreg_l17a_1p3>;
      vddch0-supply = <&vreg_l25a_3p3>;
      max-speed = <3200000>;
  };
  ```

  ([Kernel.org][5], [Sbexr][16])

* On **cheza (SDM845 Chromebook)** the Bluetooth UART is **`&uart6`** with a `wcn3990-bt` child; patch series literally “commonizes BT UART pinmux to uart6”. ([Linux Kernel Archive][14])

* Older **bluetooth‑power** binding (vendor trees) lists the PM8998 rails you’re already using and treats `qca,bt-chip-pwd-supply` as optional; most WCN3990 boards don’t use it. ([Android Git Repositories][13])

---

## Concrete changes to your current setup

* **Do not early‑bump to 3.2 M before 0xFC00.** Move the speed bump to **after** ROM read. Use a **short** settle (1–10 ms), **not 300 ms**. Drop/ignore the vendor `0x92` event at the new speed. ([Code Browser][1], [LKML][10])

* **Add the power‑pulse prelude** from userspace before `btattach`:

  ```bash
  # Optional OFF @ 2400
  stty -F /dev/ttyHS0 2400 -crtscts -echo; printf '\xC0' > /dev/ttyHS0; usleep 10000
  # Mandatory ON @ 115200
  stty -F /dev/ttyHS0 115200 -crtscts -echo; printf '\xFC' > /dev/ttyHS0; sleep 0.1
  stty -F /dev/ttyHS0 115200  crtscts -echo
  btattach -B /dev/ttyHS0 -S 115200 -P qca
  ```

  (If you later switch to **serdev**, the kernel will do this for you.) ([Code Browser][1])

* **Keep TLV seg size = 240** (fine) or bump to **243** to match upstream; keep the **10 ms** pre‑NVM delay (you already added it). ([Android Git Repositories][11], [Code Browser][3])

* **Stay on SE6 (`ttyHS0`)**—that matches SDM845 QRD/Cheza reference layouts. ([Linux Kernel Archive][14], [Android Git Repositories][4])

---

## Why this solves your `0xfc00` timeout

Your logs match the classic failure path addressed upstream in 2018–2019:

* WCN3990 **will not respond** to `0xfc00` until it has seen a **one‑byte ON pulse** at **115200** with flow control **disabled** and \~100 ms to boot. ([Code Browser][1])
* Bumping the host to **3.2 M** **before** ROM read and **without** the “drop 0x92 + short settle” handling will de‑sync the line discipline path and yield `tx timeout / -110`. The fixed behavior is in upstream `hci_qca`: **drop vendor** event `0x92`, wait \~100 ms, and only then proceed. ([Code Browser][1])

Apply the pulse + ordering + short delays above, and you should see:

```
Bluetooth: hci0: QCA controller version 0x02140201
Bluetooth: hci0: Set UART speed to 3200000
Bluetooth: hci0: ROME setup on UART is completed
```

…followed by a happy `hci0` and BlueZ scanning.

---

## Citations / references

* **IBS bytes & timeouts (driver)** – WAKE\_IND=0xFD, WAKE\_ACK=0xFC, SLEEP\_IND=0xFE; init/IBS timing and constants in `hci_qca.c`. ([Chromium Code Search][7])
* **WCN3990 power pulses & delays** – `qca_send_power_pulse()` docs/comments: single‑byte pulses, **`0xFC` ON @ 115200**, disable HW flow control for pulses, **100 ms** after ON. ([Code Browser][1])
* **Baud change handling for WCN399x** – short settle delay (1–10 ms) vs. older 300 ms; 3.2 M enable; vendor event 0x92 drop/complete wait. ([LKML][10], [Mail Archive][9], [Code Browser][1])
* **Patch→NVM 10 ms delay & ROM→file mapping** – upstream `btqca.c` selection of `crbtfw21.tlv`/`crnv21.bin` and the 10 ms sleep before NVM. ([Code Browser][3])
* **TLV segment size** – Qualcomm trees show **243** as `MAX_SIZE_PER_TLV_SEGMENT`; 240 is a common vendor choice too. ([Android Git Repositories][11])
* **UART instance on SDM845** – cheza: `&uart6` + `wcn3990-bt`; SDM845 QRD enabling `&qupv3_se6_4uart`. ([Linux Kernel Archive][14], [Android Git Repositories][4])
* **Bindings for WCN3990 supplies** – modern `qualcomm-bluetooth.yaml` (`vddio/vddxo/vddrf/vddch0`, `max-speed`) and older `bluetooth_power.txt` documenting optional `qca,bt-chip-pwd-supply`. ([Kernel.org][5], [Android Git Repositories][13])

---

If you want, I can sketch a tiny helper that sends the **OFF/ON pulses** and then execs `btattach` so you don’t forget the prelude.

[1]: https://codebrowser.dev/linux/linux/drivers/bluetooth/hci_qca.c.html "hci_qca.c source code [linux/drivers/bluetooth/hci_qca.c] - Codebrowser "
[2]: https://android.googlesource.com/kernel/gs/%2B/refs/heads/mirror-aosp-android-mainline/drivers/bluetooth/hci_qca.c?utm_source=chatgpt.com "drivers/bluetooth/hci_qca.c - kernel/gs - Git at Google"
[3]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html?utm_source=chatgpt.com "btqca.c source code [linux/drivers/bluetooth ..."
[4]: https://android.googlesource.com/kernel/msm.git/%2B/refs/heads/android-msm-tilefish-4.9-pie-wear-dr/arch/arm64/boot/dts/qcom/sdm845-qrd.dtsi?utm_source=chatgpt.com "arch/arm64/boot/dts/qcom/sdm845-qrd.dtsi - kernel/msm.git"
[5]: https://www.kernel.org/doc/Documentation/devicetree/bindings/net/bluetooth/qualcomm-bluetooth.yaml?utm_source=chatgpt.com "qualcomm-bluetooth.yaml"
[6]: https://lkml.indiana.edu/hypermail/linux/kernel/1902.3/01349.html?utm_source=chatgpt.com "[PATCH 0/3] Bluetooth: hci_qca: Add delay after power-off ..."
[7]: https://chromium.googlesource.com/chromiumos/third_party/kernel/%2B/refs/heads/firmware-trogdor-13577.B-chromeos-5.4/drivers/bluetooth/hci_qca.c "drivers/bluetooth/hci_qca.c - chromiumos/third_party/kernel - Git at Google"
[8]: https://git.rockylinux.org/staging/src-code/el9-upstream-kernel/-/blob/d9a41d6a8c258cb651f1ca44a3542688d7db80fe/drivers/bluetooth/btqca.h?utm_source=chatgpt.com "drivers/bluetooth/btqca.h - el9-upstream-kernel"
[9]: https://mail-archive.com/linux-kernel%40vger.kernel.org/msg1718826.html?utm_source=chatgpt.com "[PATCH v8 5/7] Bluetooth: hci_qca: Enable 3.2 Mbps ..."
[10]: https://lkml.org/lkml/2019/2/27/712?utm_source=chatgpt.com "[PATCH v2] hci_qca: Reduce delay after sending baudrate ..."
[11]: https://android.googlesource.com/kernel/msm/%2B/refs/tags/android-s-beta-5_r0.2/drivers/bluetooth/btqca.h?utm_source=chatgpt.com "drivers/bluetooth/btqca.h - kernel/msm - Git at Google"
[12]: https://mail-archive.com/linux-kernel%40vger.kernel.org/msg1884332.html?utm_source=chatgpt.com "btqca: inject command complete event during fw download"
[13]: https://android.googlesource.com/kernel/msm/%2B/android-7.1.0_r0.2/Documentation/devicetree/bindings/bluetooth/bluetooth_power.txt?utm_source=chatgpt.com "Documentation/devicetree/bindings/bluetooth ..."
[14]: https://lkml.iu.edu/2210.2/00285.html?utm_source=chatgpt.com "Linux-Kernel Archive: [PATCH v2 1/3] arm64: dts: qcom: sdm845"
[15]: https://lkml.rescloud.iu.edu/1811.1/08432.html?utm_source=chatgpt.com "Re: [PATCH v1 4/4] Bluetooth: hci_qca: Fix frame reassembly ..."
[16]: https://sbexr.rabexc.org/latest/sources/5f/a9e07af86859ff.html?utm_source=chatgpt.com "arch/arm64/boot/dts/qcom/sdm845-xiaomi-polaris.dts - Linux v6.6.1"
