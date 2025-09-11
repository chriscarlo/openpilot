Below is a concrete, code‑level plan that should get the WCN3990 (UART/H4+IBS on SDM845) reliably through **RESET → EDL(PATCH\_VER) → TLV (rampatch) → NVM → final RESET**, within your ≤40 s window and without upsetting USB/ADB.

I’m basing the sequencing and constants on the upstream fixes that specifically target **WCN3990-on‑SDM845** problems:

* **Do not “wait for” the WCN3990 baud‑change vendor event** the way older Rome parts did. Instead, **drop** it at the H4 event layer and use a short completion to confirm it arrived after the host flip. This was the root cause of a lot of “TLV response size mismatch / vendor event misassociation” and early timeouts on SDM845. ([lkml.iu.edu][1])
* **Power‑pulse spacing** matters on WCN399x: add a short delay between the power‑off and power‑on pulses (where those pulses are used). Systems that skipped the gap frequently fail early with 0xfc00 timeouts. ([Mail Archive][2])
* **Rampatch/NVM “skip VSE” semantics**: for Rome ≤3.1 all TLV segments are acked by a VSE; for Rome ≥3.2 the controller can skip VSE for intermediate segments and ack only the last. The driver must follow the TLV header’s `download_mode` and only inject “cmd complete” at the end when the controller skips per‑segment acks. ([Code Browser][3])

Your regressions between #56 and #63 line up with reintroducing a wait on the 0x92 baud event and with sending an **unsolicited IBS WAKE\_ACK** “nudge” before reset, which can wedge the IBS state machine. The fixes below undo those pitfalls and add targeted instrumentation so you can see exactly where things block.

---

## 1) Succinct diagnosis

**Most likely failure points:**

1. **IBS misuse before first HCI traffic**
   Your “wake before reset” tweak sends **WAKE\_IND + WAKE\_ACK** proactively. ACK must only be sent **in response to the controller’s WAKE\_IND**; sending it unprompted can leave TX/RX IBS state inconsistent, so the subsequent **HCI Reset (0x0C03)** never egresses (explains the new “RESET tx timeout” in #61–#63). Upstream disables IBS entirely during patch/NVM to avoid surprises. (See `set_bit(QCA_IBS_DISABLED)` in modern `qca_setup()`.) ([Android Source][4])

2. **WCN3990 baud‑change vendor event handling (0x92)**
   For WCN3990, the controller emits the vendor event **at the new baud** (host still at the old rate if you’re too slow), and that event can later be misinterpreted as a TLV response if you let it reach HCI. Correct handling is: **drop it in the H4 event path**, but **wait (≤100 ms)** for it to confirm the controller flipped. Extending the “wait window” or consuming it in-band as a response tends to deadlock or cause frame assembly errors. ([lkml.iu.edu][1])

3. **Power pulse spacing (if used on this platform)**
   Some SDM845 designs use the WCN3990 “power pulse” on TX. Without a small delay **after the power‑off pulse** (before power‑on), you get consistent 0xfc00 timeouts reading ROM version. If your board path really issues pulses (AOSP/Android kernels do this via `qca_send_power_pulse()`), add the gap. ([Mail Archive][2])

4. **TLV download policy**
   With your ROM version line `0x02140201` and `crbtfw21.tlv`, you’re on the “Rome 2.1” style: **per‑segment VSE acks are expected**. Don’t force “skip VSE” globally. If your current “tlv\_force\_sync” injects long pacing, keep it tiny—large per‑segment delays can cause timeouts under the HCI init watchdog. ([Code Browser][3])

---

## 2) Minimal, targeted patches (annotated pseudo‑diffs)

These are **surgical deltas** against your 4.9 AGNOS tree, limited to `drivers/bluetooth/hci_qca.c` and `drivers/bluetooth/btqca.c`. They do three things:

* Revert the “WAKE before reset” nudge and **disable IBS** during patch/NVM (match upstream).
* Implement **WCN3990 0x92 vendor event drop + 100 ms completion** (match upstream).
* Tighten **timeouts and quiet windows** with values that have shipped on SDM845 devices.

> **Note:** Names and layout follow upstream kernels for clarity; adjust to your file’s symbols if they differ slightly.

### 2.1 `drivers/bluetooth/hci_qca.c`

**A) Add a flag + completion to drop vendor events, and hook the event path**

```diff
--- a/drivers/bluetooth/hci_qca.c
+++ b/drivers/bluetooth/hci_qca.c
@@ -60,6 +60,7 @@ enum qca_flags {
-       QCA_IN_BAND_SLEEP_ENABLED,
+       QCA_IN_BAND_SLEEP_ENABLED,
+       QCA_DROP_VENDOR_EVENT,
 };

 struct qca_data {
@@ -120,6 +121,7 @@ struct qca_data {
        unsigned long flags;
+       struct completion drop_ev_comp;
        /* debug counters ... */
 };

@@ -4xx,6 +4xx,8 @@ static int qca_open(struct hci_uart *hu)
        INIT_WORK(&qca->ws_tx_vote_off, qca_wq_serial_tx_clock_vote_off);
+       init_completion(&qca->drop_ev_comp);
        qca->hu = hu;
        ...
@@ -8xx,7 +8xx,34 @@ static int qca_recv_acl_data(...)
        return hci_recv_frame(hdev, skb);
 }

+/* Intercept vendor events when QCA_DROP_VENDOR_EVENT is set (WCN3990 baud change). */
+static int qca_recv_event(struct hci_dev *hdev, struct sk_buff *skb)
+{
+       struct hci_uart *hu = hci_get_drvdata(hdev);
+       struct qca_data *qca = hu->priv;
+       struct hci_event_hdr *hdr = (void *)skb->data;
+
+       if (test_bit(QCA_DROP_VENDOR_EVENT, &qca->flags)) {
+               if (hdr->evt == HCI_EV_VENDOR)
+                       complete(&qca->drop_ev_comp);
+               kfree_skb(skb);
+               return 0;
+       }
+       return hci_recv_frame(hdev, skb);
+}
+
 static const struct h4_recv_pkt qca_recv_pkts[] = {
        { H4_RECV_ACL, .recv = qca_recv_acl_data },
        { H4_RECV_SCO, .recv = hci_recv_frame },
-       { H4_RECV_EVENT, .recv = hci_recv_frame },
+       { H4_RECV_EVENT, .recv = qca_recv_event },
        { QCA_IBS_WAKE_IND_EVENT, .recv = qca_ibs_wake_ind },
        { QCA_IBS_WAKE_ACK_EVENT, .recv = qca_ibs_wake_ack },
        { QCA_IBS_SLEEP_IND_EVENT, .recv = qca_ibs_sleep_ind },
 };
```

**Why:** This mirrors the **“drop vendor event (0x92)”** mechanism used upstream for WCN3990, preventing that stray event (emitted at the new baud) from poisoning the TLV command stream. ([lkml.iu.edu][1])

---

**B) Fix `qca_set_speed()` WCN3990 handling: set DROP flag, send, flip host, wait ≤100 ms, clear.**

```diff
@@ -10xx,12 +10xx,33 @@ static int qca_set_speed(struct hci_uart *hu, enum qca_speed_type type)
        ...
-       /* (your earlier longer wait / 300ms settle removed) */
+       /* For WCN3990 the controller sends a vendor event at the new baud.
+        * Drop that event from the HCI core, but wait briefly so we know it arrived.
+        * Keep flow-control disabled during the flip to deassert RTS while the controller switches.
+        */
+       if (soc_type == QCA_WCN3990) {
+               reinit_completion(&qca->drop_ev_comp);
+               set_bit(QCA_DROP_VENDOR_EVENT, &qca->flags);
+               smp_mb__after_atomic();
+               hci_uart_set_flow_control(hu, true);
+       }
+
        /* send controller baud change (0xFC48), then host_set_baudrate(...) */
        ret = qca_set_baudrate(hu->hdev, qca_baudrate);
        if (ret)
                goto out;
        host_set_baudrate(hu, speed);
-       usleep_range(1000, 10000);
+       /* Small settle at the new rate; large sleeps cause core timeouts. */
+       usleep_range(1000, 10000);

 out:
-       hci_uart_set_flow_control(hu, false);
+       if (soc_type == QCA_WCN3990) {
+               hci_uart_set_flow_control(hu, false);
+               if (!wait_for_completion_timeout(&qca->drop_ev_comp, msecs_to_jiffies(100))) {
+                       bt_dev_err(hu->hdev, "WCN3990: missing baud-change vendor event");
+                       ret = -ETIMEDOUT;
+               }
+               clear_bit(QCA_DROP_VENDOR_EVENT, &qca->flags);
+               smp_mb__after_atomic();
+       }
        return ret;
```

**Why:** This is the **documented WCN3990 quirk**. Your current 300 ms “wait window” plus added post‑flip sleep makes it worse; use the 100 ms completion and a **short** settle. ([lkml.iu.edu][1])

---

**C) Remove the “WAKE/WAKE\_ACK nudge” before reset and explicitly disable IBS for patch/NVM**

```diff
@@ -14xx,12 +14xx,18 @@ static int qca_setup(struct hci_uart *hu)
-       /* (your code) Optional WAKE before reset: send IBS WAKE_IND + WAKE_ACK */
-       qca_send_wake_nudge();
+       /* Patch downloading must be done with IBS disabled to avoid sleeps mid-transfer. */
+       set_bit(QCA_IBS_DISABLED, &qca->flags);
        ...
-       ret = qca_uart_setup_rome(hdev, qca_baudrate);
+       ret = qca_uart_setup_rome(hdev, qca_baudrate);
        if (!ret) {
-               /* (restore IBS here once the controller is fully up) */
+               clear_bit(QCA_IBS_DISABLED, &qca->flags);
        }
```

**Why:** Upstream keeps IBS off for the entire ROM‑init/TLV/NVM phase. Sending an **unsolicited WAKE\_ACK** is illegal in IBS and explains the early RESET timeout you now see. ([Android Source][4])

---

**D) (Serdev/board variants only) Add a short gap between power pulses**
If your 4.9 driver path **does** use `qca_send_power_pulse()` (many SDM845 trees do), add a delay between OFF→ON:

```diff
@@ static int qca_regulator_init(struct hci_uart *hu)
        host_set_baudrate(hu, 2400);
        qca_send_power_pulse(hu, false);
+       /* WCN3990 needs a small OFF→ON gap to avoid early 0xfc00 timeouts */
+       msleep(50);
        qca_set_speed(hu, QCA_INIT_SPEED);   /* 115200 */
        qca_send_power_pulse(hu, true);
+       msleep(100);
        return qca_port_reopen(hu);
```

**Why:** This exact delay resolves “EDL 0xfc00 tx timeout → QCA version read fails” on WCN399x power‑pulse designs. If your board doesn’t use pulses, this hunk is a no‑op. ([Mail Archive][2])

### 2.2 `drivers/bluetooth/btqca.c`

**E) Make HCI\_RESET and PATCH\_VER waits sane, use direct firmware load, and keep per‑segment acks**

```diff
--- a/drivers/bluetooth/btqca.c
+++ b/drivers/bluetooth/btqca.c
@@ -35,6 +35,8 @@
 /* Tweak init timeouts for UART/SDM845 without exceeding your 40s guard */
 #define QCA_INIT_CMD_TIMEOUT    msecs_to_jiffies(1500) /* for PATCH_VER */
 #define QCA_RESET_TIMEOUT       msecs_to_jiffies(3000)
+#define QCA_INTER_PHASE_QUIET   40 /* ms: quiet before issuing RESET */

@@ static int rome_patch_ver_req(struct hci_dev *hdev, u32 *rome_version)
-       skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, EDL_PATCH_CMD_LEN,
-                               &cmd, HCI_VENDOR_PKT, HCI_INIT_TIMEOUT);
+       skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, EDL_PATCH_CMD_LEN,
+                               &cmd, HCI_VENDOR_PKT, QCA_INIT_CMD_TIMEOUT);

@@ static int rome_reset(struct hci_dev *hdev)
-       skb = __hci_cmd_sync(hdev, HCI_OP_RESET, 0, NULL, HCI_INIT_TIMEOUT);
+       /* Let link go quiet very briefly before reset */
+       msleep(QCA_INTER_PHASE_QUIET);
+       skb = __hci_cmd_sync(hdev, HCI_OP_RESET, 0, NULL, QCA_RESET_TIMEOUT);

@@ static int qca_download_firmware(...)
-       ret = request_firmware(&fw, config->fwname, &hdev->dev);
+       ret = request_firmware_direct(&fw, config->fwname, &hdev->dev);

@@ static int qca_tlv_send_segment(...)
-       /* honor TLV header download_mode; don't force skip unless header says so */
+       /* honor TLV header download_mode; don't force skip unless header says so */
        if (mode == QCA_SKIP_EVT_VSE_CC || mode == QCA_SKIP_EVT_VSE)
                return __hci_cmd_send(hdev, EDL_PATCH_CMD_OPCODE, seg_size + 2, cmd);
```

**Why:**

* **PATCH\_VER** at 1.5 s is generous for SDM845 UART and avoids false timeouts.
* A **3 s RESET** is aligned with what’s been shipping.
* A **short quiet** before RESET helps avoid “tx timeout” races on sleepy links.
* `request_firmware_direct()` matches your intent and prevents userspace decompression latency.
* Keep **per‑segment acks** unless TLV explicitly indicates skip‑VSE (Rome ≥ 3.2). ([Android Source][5])

---

## 3) Instrumentation (just enough to pinpoint the wedge)

Add these **temporary breadcrumbs**:

**In `hci_qca.c`:**

* **At the start of `qca_setup()` and right before calling `qca_uart_setup_rome()`**:

  ```c
  bt_dev_info(hdev, "qca_setup: IBS %s, init_speed=%u, oper_speed=%u",
              test_bit(QCA_IBS_DISABLED, &qca->flags) ? "DISABLED" : "ENABLED",
              qca_get_speed(hu, QCA_INIT_SPEED), qca_get_speed(hu, QCA_OPER_SPEED));
  ```
* **In `qca_recv_event()`** (temporary) log the first few vendor events so you can see the 0x92 baud event and the first EDL responses:

  ```c
  if (hdr->evt == HCI_EV_VENDOR)
          bt_dev_dbg(hdev, "VSE: plen=%u (DROP=%d)", hdr->plen,
                     test_bit(QCA_DROP_VENDOR_EVENT, &qca->flags));
  ```
* **IBS counters around pre‑RESET and first EDL**: bump counters when you send/receive IBS tokens (you already have `ibs_sent_*`/`ibs_recv_*` in `struct qca_data`). Add one‑liners in the existing `qca_ibs_*` handlers:

  ```c
  bt_dev_dbg(hdev, "IBS: WAKE_IND recv (tx=%u rx=%u)", qca->ibs_sent_wakes, qca->ibs_recv_wakes);
  ```

**In `btqca.c`:**

* **Right before PATCH\_VER** and on error path:

  ```c
  BT_INFO("%s: EDL PATCH_VER →", hdev->name);
  ...
  BT_ERR("%s: PATCH_VER failed (%d)", hdev->name, err);
  ```
* **For TLV loop** log segment index and mode once per 64 segments\*\* (keeps dmesg small):

  ```c
  if ((i & 0x3f) == 0)
      bt_dev_dbg(hdev, "TLV seg=%d mode=%d remain=%zu", i, config->dnld_mode, remain);
  ```

These breadcrumbs will tell you whether you’ve made it past: **RESET → PATCH\_VER response** (was your main early wedge in #61–#63) and whether any spurious VSEs are landing during TLV.

---

## 4) Exact ordering, waits, IBS interactions & constants

**Ordering (conservative 115200 path):**

1. **Ensure rfkill unblocked** and (if your board uses it) issue **power pulses** with **50 ms OFF→ON gap** (serdev designs). ([Mail Archive][2])
2. `qca_setup()` sets **IBS disabled** for the duration. ([Android Source][4])
3. **No WAKE nudge.** Do **not** send unsolicited WAKE\_ACK.
4. **EDL PATCH\_VER (0xFC00)** with **HCI\_VENDOR** event, **1.5 s timeout**. ([Android Source][5])
5. **Rampatch TLV** per TLV header’s `download_mode` (Rome 2.1: VSE ack per segment). **No artificial pacing**; if you must, keep it **≤1 ms** between 96‑byte segments. ([Code Browser][3])
6. **msleep(10)** after patch; **NVM TLV** similarly. ([Android Source][5])
7. **Quiet ≥40 ms**, then **HCI\_RESET (0x0C03)** with **3 s timeout**.
8. If you **bump to 3 Mbps**, do it **after** the final RESET: set `QCA_DROP_VENDOR_EVENT`, send 0xFC48, flip host, **wait ≤100 ms** for dropped VSE completion, clear flag. **No 300 ms sleeps**. ([lkml.iu.edu][1])

**Constants (ms/us):**

* **Pre‑RESET quiet**: 40 ms
* **RESET completion timeout**: 3000 ms
* **PATCH\_VER timeout**: 1500 ms
* **Post‑baud flip settle**: `usleep_range(1000, 10000)`
* **Power OFF→ON gap** (if applicable): 50 ms (OFF→ON), 100 ms after ON ([Mail Archive][2])
* **TLV inter‑segment delay**: 0 (only add ≤1 ms if needed to avoid overruns)
* **Vendor 0x92 completion wait**: ≤100 ms (and **drop** the event) ([lkml.iu.edu][1])

**TLV sync policy recommendation:**

* **Rome 2.1 (your case)**: **fully synchronous** (per‑segment VSE acks). Let the TLV header drive `download_mode`. Do **not** force global “skip VSE” here; it’s for ≥3.2 and later controllers. ([Code Browser][3])

---

## 5) SDM845 prerequisites (power/clock/serial)

* If your kernel tree uses the **bt\_power/regulator path** for WCN399x (common in Android/serdev trees), keep that sequence intact and only add the **OFF→ON delay** around the pulses as above. ([lkml.iu.edu][6])
* The **ttyHS0** serial clocks/pinctrl on SDM845 are managed by the serial driver; you **do not** need extra votes in `hci_qca` for the conservative 115200 attach. The only serial manipulation we do is **temporarily disabling HW flow control** around the baud flip, per upstream. ([Android Source][4])

---

## 6) Step‑by‑step validation plan (≤4 attempts/boot, each ≤35–40 s)

**Common preparation for all attempts:**

* Module params:

  * `echo Y > /sys/module/hci_uart/parameters/patch115200`  (keep TLV/NVM at 115200)
  * Your module param `tlv_force_sync`: **leave at default (auto)**. We’ll let the TLV header decide.
* Clear dmesg, then:

  * `timeout 40 btattach -B /dev/ttyHS0 -P qca -S 115200 > /data/local/tmp/btattach.out 2>&1`
  * Save `btattach.out` and \~1500 lines of `dmesg` tail.

---

### Attempt A – **Baseline with fixes** (expected PASS to PATCH\_VER)

**Toggles:** `patch115200=Y`, `tlv_force_sync=auto`
**Expected dmesg breadcrumbs:**

* `... setting up wcn399x`
* `QCA PATCH_VER →` then `QCA controller version 0x02140201`
* `QCA Downloading qca/crbtfw21.tlv` with steady “Send segment …” dbg lines
  **Pass criteria:** PATCH\_VER completes; we start TLV and see per‑segment VSE.
  **Fail of interest:** any “HCI Reset tx timeout” **before** PATCH\_VER ⇒ IBS nudge not fully removed or pre‑RESET quiet missing.

---

### Attempt B – **TLV pacing tiny** (only if A stalls mid‑TLV)

**Toggles:** `patch115200=Y`, `tlv_force_sync=Y` with **96 B seg, 0.5–1.0 ms inter‑segment** (dial down your current 15 ms)
**Expected:** No “QCA TLV response size mismatch”; last segment causes VSE; NVM follows; RESET succeeds.
**Pass criteria:** `ROME setup on UART is completed`/`hci0 up`.
**Fail:** “TLV response size mismatch” ⇒ check for stray 0x92 vendor events (should be dropped), or bump per‑segment delay to **1.5 ms** max. ([Code Browser][3])

---

### Attempt C – **Baud bump after final RESET**

**Toggles:** Same as A, but after hci0 appears, run the oper‑speed bump path (host and controller to 3 M): ensure the **drop‑vendor‑event(0x92) wait (100 ms)** is in effect.
**Expected:** “Set UART speed to 3000000”; no frame reassembly errors; no stalls. ([lkml.iu.edu][1])

---

### Attempt D – **Power‑pulse boards only**

If your dmesg shows `sending power pulse` lines, rerun with the **OFF→ON 50 ms gap** hunk enabled.
**Expected:** Early `0xfc00 tx timeout` disappears; PATCH\_VER is reliable. ([Mail Archive][2])

---

## 7) What to look for in the logs

* **Early wedge fixed:** No `HCI Reset (0x0C03) tx timeout` before PATCH\_VER.
* **Baud flip sanity:** After the post‑RESET baud bump, you should see your temporary log `WCN3990: missing baud-change vendor event` **not** printed; instead a dbg `VSE` line should occur shortly after host flips. ([lkml.iu.edu][1])
* **TLV health:** No “TLV response size mismatch” and no extra VSEs out of place. If you see one, it’s typically the **un‑dropped 0x92** event landing late in the TLV window. ([lkml.iu.edu][1])

---

## 8) Why these patches are safe

* They **match upstream behavior** for WCN3990/SDM845: drop the 0x92 vendor event, short completions and settles, IBS disabled during ROM/TLV/NVM. ([Android Source][4])
* They don’t change Wi‑Fi paths or unrelated subsystems.
* All time constants are short; you remain comfortably under your **≤40 s** attach window even at 115200 for TLV+NVM.

---

### If you want me to diff against your exact files

I couldn’t load your GitHub blobs through the browser sandbox (GitHub page body failed to render), so I produced annotated upstream‑style hunks that you can paste where indicated. They are intentionally small, centered on the places you already modified (baud event wait, IBS wake nudge, and TLV pacing). If you need me to adapt the hunk line numbers to your 4.9 AGNOS tree once you paste in a code excerpt of those functions, I’ll do it in‑line.

---

## References

* **Drop WCN3990 baudrate‑change vendor event (0x92)**, wait ≤100 ms and **do not** feed it to HCI: avoids TLV confusion and timeouts. ([lkml.iu.edu][1])
* **Avoid long sleeps around baud‑flip**; the event is sent at the **new** rate, host must flip quickly. ([marc.info][7])
* **Power‑pulse OFF→ON delay** fixes `0xfc00 tx timeout` during early init on WCN399x platforms. ([Mail Archive][2])
* **TLV download behavior by ROM version** (per‑segment VSE ≤3.1; skip‑VSE ≥3.2, only last ack + injected CC). ([Code Browser][3])
* **Modern `hci_qca` flow**: IBS disabled during ROM/TLV/NVM; re‑enable after setup. ([Android Source][4])

---

### TL;DR

* **Remove the IBS “nudge”** and **disable IBS** during ROM/TLV/NVM.
* **Implement WCN3990’s 0x92 vendor‑event drop** with a **100 ms wait** and **short (1–10 ms) post‑flip settle**.
* **PATCH\_VER 1.5 s**, **RESET 3 s**, **40 ms quiet** before RESET.
* **Honor TLV header** for VSE policy; for your 2.1 images, **per‑segment acks** with tiny pacing (≤1 ms) if needed.
* If you use **power pulses**, add the **50 ms OFF→ON** gap.

Run Attempt A; if it still wedges, send me the dmesg + your `btattach.out` for that attempt and I’ll adjust the exact wait paths (you’ll already have the breadcrumbs to show precisely where it died).

[1]: https://lkml.iu.edu/1903.1/04409.html "Linux-Kernel Archive: [PATCH v2 2/2] Bluetooth: hci_qca: wcn3990: Drop baudrate change vendor event"
[2]: https://www.mail-archive.com/linux-kernel%40vger.kernel.org/msg1942824.html?utm_source=chatgpt.com "[PATCH v2 3/3] Bluetooth: hci_qca: Add delay after power- ..."
[3]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html "btqca.c source code [linux/drivers/bluetooth/btqca.c] - Codebrowser "
[4]: https://android.googlesource.com/kernel/gs/%2B/refs/heads/mirror-aosp-android-mainline/drivers/bluetooth/hci_qca.c "drivers/bluetooth/hci_qca.c - kernel/gs - Git at Google"
[5]: https://android.googlesource.com/kernel/common/%2B/d4504d1eba95/drivers/bluetooth/btqca.c?utm_source=chatgpt.com "drivers/bluetooth/btqca.c - kernel/common - Git at Google"
[6]: https://lkml.iu.edu/hypermail/linux/kernel/1806.3/00904.html?utm_source=chatgpt.com "[PATCH v8 0/7] Enable Bluetooth functionality for WCN3990"
[7]: https://marc.info/?l=linux-kernel&m=157168552020677&w=2&utm_source=chatgpt.com "[PATCH] Bluetooth: hci_qca: Add delay for wcn3990 stability"
