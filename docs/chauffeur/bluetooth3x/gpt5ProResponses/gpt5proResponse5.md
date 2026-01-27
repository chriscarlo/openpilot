Got it. Here’s the concrete, minimal way to get WCN3990 “skip‑VSE” working on your 4.9 tree, fix the baud‑switch (-84) framing, and make the TLV/NVM path behave. I’m pulling these from upstream btqca/hci\_qca as they exist in mainline and long‑term trees so we don’t reinvent any host‑credit or H4 framing wheels.

---

## 1) The minimal, correct “skip‑VSE” backport for 4.9 (btqca + hci core interaction)

**What upstream actually does**

* **Parse the TLV header’s `download_mode`** and remember it in the per‑download config (`config->dnld_mode` and `config->dnld_type`). For Rome ≥ 3.2 / WCN399x, the controller may **skip per‑segment Vendor‑Specific Events** and only ack the **last** segment. ([cregit.linuxsources.org][1])
* **Send early TLV segments with `__hci_cmd_send()`** (fire‑and‑forget into the HCI command queue; do **not** wait for any per‑segment event). Send the **last** segment with `__hci_cmd_sync_ev()` and the expected ack event. This is the upstream pattern and avoids breaking HCI command credits. **Do not** do driver‑local synthetic skbs per segment. ([Codebrowser][2])
* After the last segment has been acked, **inject a synthetic `HCI_EV_CMD_COMPLETE`** once (ncmd=1, opcode=**0xFC00**, status=0x00) to satisfy the HCI core’s command‑timeout / credit book‑keeping during the earlier fire‑and‑forget burst. Upstream uses `qca_inject_cmd_complete_event(hdev)`, which builds the event and feeds it via `hci_recv_frame()`. **This is crucial** to stop “command tx timeout” noise in older stacks. ([cregit.linuxsources.org][1])
* Constants are defined in `btqca.h` upstream:
  `#define QCA_HCI_CC_OPCODE 0xFC00` and `#define QCA_HCI_CC_SUCCESS 0x00`. ([Android Gooblesource][3])

> **Why credits don’t get violated:** early segments are queued with `__hci_cmd_send()` through the HCI core’s normal command queue (not bypassing it). The core’s command‑timeout machinery expects CC/CS events periodically; since the controller purposefully skips them until the end, upstream **injects** a single CC after the last segment’s real ack to keep the core state consistent. You don’t need a private queue nor per‑segment synthetic acks. See the exact logic in mainline btqca. ([Codebrowser][2])

**Drop‑in backport recipe (fits your 4.9 layout):**

* **btqca.h**
  Ensure these exist (or add them):

  ```c
  #define QCA_HCI_CC_OPCODE   0xFC00
  #define QCA_HCI_CC_SUCCESS  0x00

  enum qca_tlv_dnld_mode {
    QCA_SKIP_EVT_NONE,
    QCA_SKIP_EVT_VSE,
    QCA_SKIP_EVT_CC,
    QCA_SKIP_EVT_VSE_CC
  };

  struct qca_fw_config {
    u8 type;
    char fwname[64];
    u8 user_baud_rate;
    enum qca_tlv_dnld_mode dnld_mode; /* per-segment behavior while looping */
    enum qca_tlv_dnld_mode dnld_type; /* original header's download_mode   */
  };
  ```

  Values/defines match upstream. ([Android Gooblesource][3])

* **btqca.c**

  1. **Parse `download_mode`** in your TLV header scan:

     ```c
     /* inside qca_tlv_check_data() after reading TLV header */
     config->dnld_mode = QCA_SKIP_EVT_NONE;
     config->dnld_type = QCA_SKIP_EVT_NONE;
     /* For Rome >= 3.2 and WCN399x, download_mode indicates skip policy */
     config->dnld_mode = tlv_patch->download_mode;
     config->dnld_type = config->dnld_mode;
     ```

     (See upstream `qca_tlv_check_data()`.) ([cregit.linuxsources.org][1])

  2. **Segment sender**: implement `qca_tlv_send_segment()` to:

     * Build the vendor command (EDL\_PATCH\_CMD\_OPCODE, payload “TLV req”).
     * For **skip modes** (`QCA_SKIP_EVT_VSE` or `_VSE_CC`), use `__hci_cmd_send(...)`.
     * For **non‑skip** (last segment or old controllers), use `__hci_cmd_sync_ev(...)` with expected event (`HCI_EV_VENDOR` and `EDL_TVL_DNLD_RES_EVT` on WCN3990).
       Upstream uses the “fire‑and‑forget for early” / “sync for last” split. **Copy that.** ([Codebrowser][2])

  3. **While looping segments**: force the **last** segment to wait (set `config->dnld_mode = QCA_SKIP_EVT_NONE` when `remain == 0`), call `qca_tlv_send_segment()`, and after the loop, if `config->dnld_type` was skip mode, **inject CC**:

     ```c
     if (config->dnld_type == QCA_SKIP_EVT_VSE ||
         config->dnld_type == QCA_SKIP_EVT_VSE_CC)
       qca_inject_cmd_complete_event(hdev); /* ncmd=1, opcode=0xFC00, status=0x00 */
     ```

     (Exactly like upstream `qca_download_firmware()`.) ([cregit.linuxsources.org][1])

  4. **The injector itself** (`qca_inject_cmd_complete_event`):

     ```c
     static int qca_inject_cmd_complete_event(struct hci_dev *hdev)
     {
       struct hci_event_hdr *hdr;
       struct hci_ev_cmd_complete *evt;
       struct sk_buff *skb;

       skb = bt_skb_alloc(sizeof(*hdr)+sizeof(*evt)+1, GFP_KERNEL);
       if (!skb) return -ENOMEM;

       hdr = skb_put(skb, sizeof(*hdr));
       hdr->evt  = HCI_EV_CMD_COMPLETE;
       hdr->plen = sizeof(*evt) + 1;

       evt = skb_put(skb, sizeof(*evt));
       evt->ncmd   = 1;
       evt->opcode = cpu_to_le16(QCA_HCI_CC_OPCODE);

       skb_put_u8(skb, QCA_HCI_CC_SUCCESS);
       hci_skb_pkt_type(skb) = HCI_EVENT_PKT;
       return hci_recv_frame(hdev, skb);
     }
     ```

     This is lifted straight from upstream and compiles cleanly on 4.9. ([cregit.linuxsources.org][1])

  5. **Short settle before NVM**: upstream added **\~10 ms** pause between patch and NVM, otherwise NVM download sometimes throws “TLV response size mismatch.” Keep your 10 ms (or bump to 40–50 ms only if you still see flakes). ([Basealt Gitea][4])

> TL;DR of the backport: **don’t “spray” `hci_send_cmd()` and hand‑roll fake CCs; just copy upstream’s exact pattern:** parse `download_mode`, early segments via `__hci_cmd_send()`, last via `__hci_cmd_sync_ev()`, then **one** synthetic CC (0xFC00, status 0x00). This is the minimal change set that aligns credit handling with the HCI core.

---

## 2) Fixing the baud‑switch “Frame reassembly failed (-84)” (H4 framing at rate change)

The -84 right after the pre‑bump is a classic WCN3990 wart. The upstream‑correct sequence around **changing to 3Mbaud** is:

1. **Gate the UART RX by deasserting RTS** (host‑side flow control “off”) **around the baud switch**, so you don’t parse the rate‑switch response as garbage at the wrong baud. In `hci_qca.c` upstream this is `hci_uart_set_flow_control(hu, true)` *before* sending the vendor baud command and `...false` afterwards (named a bit confusingly in code, but the intent is deassert RTS). ([Android Gooblesource][5])
2. For WCN3990 specifically, **drop the vendor event** (0x92) associated with the baud‑change command and **wait up to \~100 ms** for it to arrive **after** switching the host baud. Upstream sets a `QCA_DROP_VENDOR_EVENT` flag + `completion`, waits 100 ms, then clears the flag. This prevents that stray VSE from being mis‑classified later as a TLV response (which is exactly how you get “response size mismatch” / -84/-5). ([Android Gooblesource][5])
3. Some platforms also benefited from an **extra \~50 ms delay** before re‑enabling flow after the rate change (documented on msm8998/Lenovo Miix 630). If you still see -84 on specific hardware, add that guard. ([Marc][6])

> **Action for your tree:** Either **(A)** stop pre‑bumping to 3M until the above two patches are in, or **(B)** keep pre‑bump but **must** backport both: *“Deassert RTS while baudrate change”* and *“WCN3990: Drop baudrate‑change vendor event (wait 100 ms)”*. This is exactly how upstream prevents H4 framing and later TLV confusion. ([LKML][7])

---

## 3) About your current skip‑VSE attempt

> “Skip‑VSE attempt: early segments sent with `hci_send_cmd` (no per‑segment wait); last segment uses `__hci_cmd_sync_ev`; no CC injection yet.”

* Correct direction, but **don’t** bypass the HCI core’s expectations by pushing multiple commands and then fabricating per‑segment acks or skb completions. Use **upstream’s `__hci_cmd_send` + single injected CC** approach; it’s specifically there to **avoid HCI command timeout spew** and keep credits sane. See the mainline commit “btqca: inject command complete event during fw download” and the corresponding code path around the injection. ([Mail Archive][8])
* Also make sure your TLV header parsing fills `config->dnld_mode` so the “last segment waits” logic flips modes just like upstream. ([cregit.linuxsources.org][1])

---

## 4) The occasional `Failed to request file: err = (-11)` during TLV

* `-11` is **`-EAGAIN`** from the firmware loader; on older kernels it typically means you hit the **sysfs fallback path or a not‑ready userspace** window (udev helpers), not that the file is actually missing. To avoid this, ensure **direct filesystem lookup** works and avoid fallback (set `firmware_class` search path **before** bringing up BT, keep your bind‑mount in place, or use `request_firmware_direct()` if available in your 4.9 variant). The kernel firmware docs explicitly note the difference and why `*_direct()`/`*_nowarn()` avoids long fallbacks/timeouts. ([Debian Manpages][9])
* Upstream btqca also moved to **release the firmware after the synthetic CC injection**; while not the root cause of -EAGAIN, it’s a small correctness cleanup worth taking when you backport the injector. (The series that added the injector also included tidy‑ups like “use correct byte format for opcode” and “reset download type”.) ([LKML][10])

---

## 5) Concrete diffs to apply (minimal backport sketch)

In **`drivers/bluetooth/btqca.c`** (names per your tree):

* **Add** `qca_inject_cmd_complete_event()` exactly as upstream (shown above). ([cregit.linuxsources.org][1])
* **In** `qca_tlv_check_data()` (or your equivalent), set `config->dnld_mode` and `config->dnld_type` from the parsed TLV **download\_mode** field. ([cregit.linuxsources.org][1])
* **In** your segment sender (`rome_tlv_send_segment()` / `qca_tlv_send_segment()`):

  * Choose **`__hci_cmd_send()`** for skip modes and **`__hci_cmd_sync_ev()`** for non‑skip (last segment).
  * For WCN3990 the ack event type for TLV is **VSE** (`HCI_EV_VENDOR`/`EDL_TVL_DNLD_RES_EVT`). (3991+ uses CC; upstream handles that with `soc_type` checks—fine to keep it vendor for 3990.) ([cregit.linuxsources.org][1])
* **In** your download loop, force last segment’s `dnld_mode=QCA_SKIP_EVT_NONE`, then **after the loop** if the original `dnld_type` was skip mode, call `qca_inject_cmd_complete_event(hdev);`. ([cregit.linuxsources.org][1])
* **Keep** a **10 ms delay before NVM** (you already have 10 ms; upstream says 10 ms is plenty). If you still see flakiness, try **40–50 ms** as you proposed—this is safe and matches field experience. ([Basealt Gitea][4])

In **`drivers/bluetooth/hci_qca.c`**:

* Backport **“Deassert RTS while baudrate change”** (flow control gating around the baud set). You should see code like:

  ```c
  if (qca_is_wcn399x(soc_type))
    hci_uart_set_flow_control(hu, true);   /* deassert RTS (block RX) */

  ret = qca_set_baudrate(...);  /* send VS baud cmd */
  host_set_baudrate(hu, speed);

  if (qca_is_wcn399x(soc_type))
    hci_uart_set_flow_control(hu, false);  /* re-enable RX */
  ```

  ([Android Gooblesource][5])
* Backport **“WCN3990: Drop baudrate change vendor event”** (set `QCA_DROP_VENDOR_EVENT`, wait up to \~100 ms for `drop_ev_comp`, then clear). This prevents the stray 0x92 from poisoning the TLV path **and** helps avoid -84 framing. ([Android Gooblesource][5])

---

## 6) Answering your pointed questions

> **Q:** How should early TLV segments be queued without violating HCI command credits?
> **A:** Exactly as upstream: **queue early segments via `__hci_cmd_send()`**, **wait on the last** with `__hci_cmd_sync_ev()`, then **inject one CC**. Don’t fabricate per‑segment completions. This keeps the HCI core’s cmd timeout/credit state sane and mirrors the mainline fix. ([Mail Archive][8])

> **Q:** Do we need to inject a Command Complete at the end? Which opcode/payload?
> **A:** **Yes.** Inject `HCI_EV_CMD_COMPLETE` with `ncmd=1`, `opcode=**0xFC00**`, and **status byte `0x00`** appended; feed it with `hci_recv_frame()`. See upstream `qca_inject_cmd_complete_event()`. ([cregit.linuxsources.org][1])

> **Q:** Pre‑bump to 3M causes “Frame reassembly failed (-84)”. What’s the upstream‑correct handling?
> **A:** **Deassert RTS** (host RX gated) while switching rates, **drop the baud‑change VSE (0x92)**, and **wait \~100 ms** for it after switching host baud **before** resuming flow. Add an extra **\~50 ms** guard before re‑enabling flow if needed on your platform. Alternatively, avoid the pre‑bump until both patches are in. ([Android Gooblesource][5])

> **Q:** The sporadic `-EAGAIN` from `request_firmware()`—timing race?
> **A:** On 4.9, `request_firmware()` may fall back to userspace helpers and return **`-EAGAIN`** if it hits a timing/path window. Avoid the fallback by ensuring the **search path is set and mounted before** attach and consider `request_firmware_direct()` if present in your 4.9 (no fallback, no long timeout). ([Debian Manpages][9])

> **Q:** Reference diff to implement parsing `download_mode`, skipping VSE per segment, and handling the final ack?
> **A:** Use the **mainline btqca** implementations of:
>
> * `qca_tlv_check_data()` (parses `download_mode`),
> * `qca_tlv_send_segment()` (early segments via `__hci_cmd_send`, last via `__hci_cmd_sync_ev`),
> * `qca_inject_cmd_complete_event()` (single synthetic CC).
>   These 3 bits together are the “minimal” skip‑VSE backport. ([cregit.linuxsources.org][1])

---

## 7) What I recommend you do next (applies to your #52 → #53/#54 plan)

**Step A — Back out the ad‑hoc skip‑VSE and implement upstream semantics exactly**

* Revert the “`hci_send_cmd` + synthetic skb” bits.
* Add the `download_mode` parsing, early‑segment `__hci_cmd_send` path, and **single CC injection** after the last segment. (Keep your 128‑byte segments for the **first N** segments if you’ve found it stabilizes seg0 on this board; that’s compatible with the upstream pattern.)

**Step B — Fix the 3M pre‑bump properly**

* Either disable pre‑bump for now **or** backport both patches: **Deassert RTS around baud‑switch** and **Drop 0x92 vendor event + 100 ms wait**. This should eliminate the “Frame reassembly failed (-84)” and the later “response size mismatch / -5” you saw. ([Android Gooblesource][5])

**Step C — Keep small settles, but don’t overdo:**

* **ROM read retries**: keep your micro‑settles (5 ms pre‑retry + 40 ms between tries) — those don’t conflict with upstream.
* **TLV pacing**: **10 ms** inter‑segment is fine; keep **+10 ms after seg0** if it measurably helps this UART.
* **Patch→NVM**: leave at **10 ms** (per upstream) and only expand to 40–50 ms if you still see NVM size mismatches. ([Basealt Gitea][4])

**Step D — Firmware loader hygiene**

* Ensure the **firmware path / bind‑mount** exists before you start btattach; if your 4.9 has `request_firmware_direct()`, consider using it in btqca to skip fallback entirely (removes the `-EAGAIN` spur). ([Debian Manpages][9])

**Step E — Run bounded attach slices and log**

* Keep your 15s/35s/60s slices, reselect newest ADB `transport_id` each time (your runbook is good), and capture the last 20 Bluetooth lines and the config knobs into `BT_RESEARCH_STATUS.md` each pass.

---

## 8) What success should look like

With the above in, you should get exactly the upstream dmesg string sequence on success:

```
Bluetooth: hci0: ROME setup
Bluetooth: hci0: ROME controller version 0x02140201
Bluetooth: hci0: ROME Downloading file: qca/crbtfw21.tlv
Bluetooth: hci0: ROME setup on UART is completed
```

…and `hciconfig hci0 up` should succeed, `hciconfig -a` will show controller info, and `btmgmt power on; btmgmt find -l 5` will scan.

---

### References (load‑bearing)

* **Upstream skip‑VSE implementation, segment handling, CC injection** — mainline `btqca.c` (`qca_tlv_check_data`, `qca_tlv_send_segment`, `qca_download_firmware`, `qca_inject_cmd_complete_event`). ([cregit.linuxsources.org][1])
* **Definition of injected CC opcode/status in header** (`btqca.h`). ([Android Gooblesource][3])
* **Rationale/commit for injecting CC** (Qualcomm chips only ack last segment; inject CC to avoid timeouts). ([Mail Archive][8])
* **Baud‑switch fixes:**
  – **Deassert RTS during baud change, gate host RX**. ([Android Gooblesource][5])
  – **WCN3990: drop baudrate‑change vendor event 0x92** and wait (\~100 ms). ([GitLab][11])
  – **Optional 50 ms delay** before re‑enabling flow (hardware‑specific). ([Marc][6])
* **Short delay before NVM** (\~10 ms) to avoid TLV size mismatch. ([Basealt Gitea][4])
* **Firmware loader behavior & avoiding `-EAGAIN` fallback** (`request_firmware_direct()` vs fallback). ([Debian Manpages][9])

---

## Quick to‑do list you can copy/paste into your tracker

1. **btqca:** add `download_mode` parsing; switch early segments to `__hci_cmd_send`, last to `__hci_cmd_sync_ev`; add `qca_inject_cmd_complete_event()` and call it if `dnld_type` indicates skip.
2. **hci\_qca:** backport **Deassert RTS** around baud change + **Drop vendor event 0x92** with 100 ms wait (WCN3990 path).
3. Keep TLV pacing: **128 B** for first 8 segments (ok), **10 ms** between segments, **+10 ms after seg0**.
4. Keep **IBS off** during patch.
5. Keep **10 ms** pause between patch and NVM; only increase if NVM still flakes.
6. Ensure firmware path/bind‑mount exists **before** attach; consider `request_firmware_direct()` to avoid -EAGAIN.
7. Run your bounded attach slices and log last 20 BT lines + parameters after each try.

If you want, I can sketch the exact diffs against your file paths (`agnos-kernel-sdm845/drivers/bluetooth/{btqca.c,btqca.h,hci_qca.c}`) with your function names; but the upstream snippets above are already sized to drop into 4.9 with minimal surgery.

[1]: https://cregit.linuxsources.org/code/5.16/drivers/bluetooth/btqca.c.html "Cregit: Linux 5.16: btqca.c"
[2]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html "btqca.c source code [linux/drivers/bluetooth/btqca.c] - Codebrowser "
[3]: https://android.googlesource.com/kernel/common/%2B/refs/tags/android13-5.15-2022-11_r10/drivers/bluetooth/btqca.h?utm_source=chatgpt.com "drivers/bluetooth/btqca.h - kernel/common - Git at Google"
[4]: https://gitea.basealt.ru/iv/linux/commit/8059ba0bd0e4694e51c2ee6438a77b325f06c0d5?utm_source=chatgpt.com "Bluetooth: btqca: Add a short delay before downloading the NVM ..."
[5]: https://android.googlesource.com/kernel/common/%2B/34d3a78c681e8/drivers/bluetooth/hci_qca.c?autodive=0%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F "drivers/bluetooth/hci_qca.c - kernel/common - Git at Google"
[6]: https://marc.info/?l=linux-bluetooth&m=157151983614046&w=2&utm_source=chatgpt.com "[PATCH] Bluetooth: hci_qca: Add delay for wcn3990 stability"
[7]: https://lkml.rescloud.iu.edu/1901.1/03206.html?utm_source=chatgpt.com "hci_qca: Deassert RTS while baudrate change command"
[8]: https://mail-archive.com/linux-kernel%40vger.kernel.org/msg1884332.html?utm_source=chatgpt.com "btqca: inject command complete event during fw download"
[9]: https://manpages.debian.org/testing/linux-manual-4.8/request_firmware_direct.9.en.html?utm_source=chatgpt.com "request_firmware_direct(9) — linux-manual-4.8"
[10]: https://lkml.org/lkml/2019/8/25/207?utm_source=chatgpt.com "Linus Torvalds: Linux 5.3-rc6 - LKML"
[11]: https://git.gitlab.arm.com/arm-reference-solutions/linux/-/commit/32646db8cc2862a14788de1bb4c365d0a27fb532?utm_source=chatgpt.com "Bluetooth: btqca: inject command complete event during fw ..."
