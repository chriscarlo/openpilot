Chris — here’s a concrete, code‑level plan to get the WCN3990 (SDM845, /dev/ttyHS0) reliably through RESET → PATCH\_VER → TLV+NVM → final RESET, within your 35–40 s window and without destabilizing USB/ADB.

Below I give: (1) diagnosis, (2) minimal patches (annotated diffs) to **hci\_qca.c** and **btqca.c**, (3) instrumentation points, and (4) a 4‑attempt runbook with exact toggles, time constants, and expected breadcrumbs.

---

## 1) Succinct diagnosis

**Most likely failure points, based on your chronology:**

1. **Early RESET timeout is self‑inflicted by the pre‑RESET “WAKE/ACK nudge” and IBS state churn.**
   On WCN399x the upstream sequence is *not* “RESET first” — you read ROM/PATCH version via EDL (0xFC00) *before* downloading TLV/NVM and only do an HCI Reset at the end. Mixing an extra RESET while IBS is still converging can starve TX or misparse RX (at 115200 baud, a few stray bytes at the wrong time are enough), causing your `0x0C03 tx timeout`. Upstream WCN399x support explicitly handles the stray RX during boot and defers RESET to the end of the sequence. ([Android Source][1])

2. **EDL (0xFC00) stalls during TLV are classic WCN3990 issues:**

   * **Stray vendor event 0x92 (baudrate‑change) arriving late** can be misinterpreted as a TLV response unless it’s explicitly dropped. Upstream fixes this by marking that event to be dropped and waiting for it immediately after the host port flips speed. If not handled, you get “TLV response size mismatch” / timeouts later. ([Linux-Kernel Archive][2])
   * **Skip‑VSE download mode vs. segment acks.** For Rome ≥3.2/WCN399x only the **last** segment may be acked if the TLV header’s “download mode” says to skip VSEs; if your host pacing/IBS sleeps don’t align, it’s safer to force per‑segment acks (synchronous mode) for at least the early segments or the entire transfer. Upstream code documents and handles this mode field. ([Code Browser][3])
   * **IBS allowing the link to fall asleep during EDL/TLV** produces `0xfc00 tx timeout`. Keeping the link “awake” (do **not** send SLEEP\_IND / ignore SLEEP\_IND temporarily) during **PATCH\_VER + first TLV window** stabilizes things.

3. **RX garbage during multi‑baud stage.** Upstream added a small but important quirk to **discard RX** until the controller is past the boot stage, precisely to avoid mis‑reassembly and bogus timeouts. If your 4.9 tree doesn’t have that, add it. ([Linux Kernel Archive][4])

Bottom line: remove the early RESET nudge, **guard IBS** (no sleeping) through PATCH\_VER + early TLV, **drop vendor 0x92** on pre‑bump, and (optionally) **force TLV synchronous** to eliminate the 0xFC00 stalls.

---

## 2) Minimal patch set (targeted diffs)

These are **surgical** changes on top of a stock 4.9 hci\_qca/btqca (they’ll apply with tiny context edits to AGNOS 4.9.103). I’m keeping your module params (`patch115200`, `tlv_force_sync`) and the `request_firmware_direct()` behavior you already added.

### 2.1 hci\_qca.c — IBS “awake guard” and drop the pre‑RESET nudge

**Goals:**
• Don’t do an early RESET.
• Keep the link awake (ignore SLEEP\_IND) for the fragile **PATCH\_VER + first TLV** window.
• If you already carry the upstream “drop vendor event 0x92” patch (you said logs show it), leave it as is; otherwise, add the btqca/hci\_qca bits in §2.2.

```diff
diff --git a/drivers/bluetooth/hci_qca.c b/drivers/bluetooth/hci_qca.c
index XXXXXXX..YYYYYYY 100644
--- a/drivers/bluetooth/hci_qca.c
+++ b/drivers/bluetooth/hci_qca.c
@@ -60,6 +60,7 @@ enum qca_flags {
 	QCA_IN_BAND_SLEEP_ENABLED,
+	QCA_AWAKE_GUARD,           /* Ignore IBS SLEEP_IND while set */
 	QCA_DROP_VENDOR_EVENT,
 };

@@ -120,6 +121,8 @@ struct qca_data {
 	/* existing fields ... */
+	unsigned long awake_guard_expires;   /* jiffies deadline */
 };

+static inline bool qca_awake_guard_active(struct qca_data *qca)
+{
+	if (!test_bit(QCA_AWAKE_GUARD, &qca->flags))
+		return false;
+	if (time_after(jiffies, qca->awake_guard_expires)) {
+		clear_bit(QCA_AWAKE_GUARD, &qca->flags);
+		smp_mb__after_atomic();
+		return false;
+	}
+	return true;
+}
+
 /* IBS sleep indication handler (name may be qca_ibs_sleep_ind() or device_want_to_sleep()) */
 static int qca_ibs_sleep_ind(struct hci_dev *hdev, struct sk_buff *skb)
 {
 	struct hci_uart *hu = hci_get_drvdata(hdev);
 	struct qca_data *qca = hu->priv;
+	if (qca_awake_guard_active(qca)) {
+		bt_dev_dbg(hdev, "IBS: ignore SLEEP_IND during awake guard");
+		kfree_skb(skb);
+		return 0;
+	}
 	/* existing handling ... */
 }

@@ -9xx,12 +1005,30 @@ static int qca_setup(struct hci_uart *hu)   /* or qca_uart_setup() in your tree */
 {
 	struct qca_data *qca = hu->priv;
 	struct hci_dev *hdev = hu->hdev;
 	int ret;

-	/* (REMOVE) any “wake before reset” nudge added previously */
-	/* (REMOVE) any early hci reset here */
+	/* Keep the link awake through PATCH_VER and early TLV.
+	 * On WCN399x the correct flow is:
+	 *   read version (EDL 0xFC00/0x19) -> patch+NVM -> final HCI Reset
+	 * Guard for ~600 ms; we will extend it over early TLV in btqca.
+	 */
+	set_bit(QCA_AWAKE_GUARD, &qca->flags);
+	qca->awake_guard_expires = jiffies + msecs_to_jiffies(600);
+	smp_mb__after_atomic();

 	/* proceed with normal qca setup (no early reset) */
 	/* ... call into btqca qca_uart_setup_* which does PATCH_VER then TLV/NVM ... */

 	return ret;
 }
```

**Why this helps**
• You stop sending a RESET while IBS can still be asleep; instead you keep the path awake and let `qca_read_soc_version()` complete, then TLV/NVM, **then** do the final RESET (as upstream does). ([Android Source][1])
• Ignoring SLEEP\_IND for this brief window removes the `0xfc00 tx timeout` stalls caused by the device dozing mid‑EDL.
• This is minimal risk: the guard auto‑expires, and you don’t touch Wi‑Fi or other subsystems.

> If your function names differ in 4.9 (e.g., `device_want_to_sleep()` instead of a discrete `qca_ibs_sleep_ind()`), put the guard at the top of whatever handler flips RX to ASLEEP upon SLEEP\_IND.

---

### 2.2 hci\_qca.c — ensure vendor 0x92 drop logic is present (if missing)

You likely already have this (your logs show it), but for completeness: upstream **drops** the stray vendor event (0x92) after the baud flip and prevents it from being consumed by the next EDL command. Patch adds `QCA_DROP_VENDOR_EVENT`, completion, and switches the H4 event recv path to a small wrapper. ([Linux-Kernel Archive][2])

If you **don’t** have it, cherry‑pick equivalent of Matthias Kaehlcke’s patch (adds `drop_ev_comp`, sets/clears `QCA_DROP_VENDOR_EVENT` around `qca_set_speed()`, and replaces `{ H4_RECV_EVENT, .recv = hci_recv_frame }` with a wrapper that discards the lone vendor event). ([Linux-Kernel Archive][2])

---

### 2.3 btqca.c — safer timeouts, quiet settle, and (optional) fully‑sync TLV

**Goals:**
• Don’t time out RESET at the end (give the controller more time to apply the patch).
• Give a short quiet time before RESET.
• Allow forcing **per‑segment acks** (synchronous) for TLV to avoid 0xFC00 stalls at 115200.

```diff
diff --git a/drivers/bluetooth/btqca.c b/drivers/bluetooth/btqca.c
index AAAAAAA..BBBBBBB 100644
--- a/drivers/bluetooth/btqca.c
+++ b/drivers/bluetooth/btqca.c
@@ -190,11 +190,17 @@ static int qca_send_reset(struct hci_dev *hdev)
 {
 	struct sk_buff *skb;
 	int err;

 	bt_dev_dbg(hdev, "QCA HCI_RESET");
-	skb = __hci_cmd_sync(hdev, HCI_OP_RESET, 0, NULL, HCI_INIT_TIMEOUT);
+	/* After rampatch/NVM, give controller a moment before RESET */
+	usleep_range(40000, 60000); /* 40–60 ms quiet settle */
+	/* WCN399x can take longer to ack RESET post-patch; use 3 s */
+	skb = __hci_cmd_sync(hdev, HCI_OP_RESET, 0, NULL,
+			     msecs_to_jiffies(3000));
 	if (IS_ERR(skb)) {
 		err = PTR_ERR(skb);
 		bt_dev_err(hdev, "QCA Reset failed (%d)", err);
 		return err;
 	}
 	kfree_skb(skb);
 	return 0;
 }
```

If your tree still uses the **Rome** names (4.4/4.9 era), this is the exact same change but inside `rome_reset()` instead of `qca_send_reset()`. ([Android Source][1])

**Force per‑segment acks, gated by your `tlv_force_sync` param**
Upstream honors the TLV header’s “download mode” (may skip VSEs and only ack the last segment). We’ll optionally **override** that to “no skip” for safety. ([Code Browser][3])

Add a module param in **btqca.c** (or reuse your existing one) and override `config->dnld_mode` prior to the TLV send loop:

```diff
@@ -40,6 +40,11 @@
 static bool tlv_force_sync = true;  /* keep your default Y; tunable at runtime */
 module_param(tlv_force_sync, bool, 0644);
 MODULE_PARM_DESC(tlv_force_sync, "Force per-segment acks for QCA TLV download");

@@ static int qca_download_firmware(struct hci_dev *hdev,
 	/* existing parsing populates config->dnld_mode from TLV header */
+	if (tlv_force_sync)
+		config->dnld_mode = QCA_SKIP_EVT_NONE; /* ack each segment */
+
 	/* ... then the loop that calls qca_tlv_send_segment() ... */
```

**(Optional) Conservative pacing at 115200**
If you *still* see stalls at 115200 on some units, pace the first \~8 KB:

```diff
@@ -6xx,6 +6xx,14 @@  while (remain) {
 	segsize = min_t(size_t, remain, MAX_SIZE_PER_TLV_SEGMENT);
 	ret = qca_tlv_send_segment(hdev, segsize, segment, config->dnld_mode, soc_type);
 	if (ret)
 		goto out;
+	/* Extra safety at 115200: short delay for early segments only */
+	if (hu->init_speed <= 115200) {
+		static const size_t guard_bytes = 8 * 1024; /* only early bytes */
+		if ((total_sent += segsize) <= guard_bytes)
+			usleep_range(7000, 9000);   /* 7–9 ms between segments */
+	}
 	segment += segsize;
 	remain  -= segsize;
 }
```

> The **module param** lets you A/B without reflashing; forcing full‑sync is slower but reliable. Once you’re green, you can relax it.

**Ensure the ordering is upstream‑correct**:
Inside `qca_uart_setup()` (Rome/WCN path) the sequence should be:
`qca_read_soc_version()` → `qca_send_patch_config_cmd()` → **download TLV** → short `msleep(10)` → **download NVM** → **RESET** → (optional) read build info, check BDADDR. Your baseline log that RESET fails at the *end* matches this ordering; keep it. ([Code Browser][5])

---

## 3) Targeted instrumentation (low‑noise, high‑signal)

**hci\_qca.c**

* In your IBS handlers, add the minimal breadcrumbs and counters:

```c
/* at qca_data: already have these in some trees, keep visible temporarily */
u64 ibs_sent_wakes, ibs_sent_wacks, ibs_recv_wakes, ibs_recv_wacks, ibs_recv_slps;

/* in send_hci_ibs_cmd(): */
bt_dev_dbg(hdev, "IBS TX cmd 0x%02x t=%u", cmd, jiffies_to_msecs(jiffies));

/* at top of qca_ibs_wake_ind / qca_ibs_wake_ack / qca_ibs_sleep_ind: */
bt_dev_dbg(hdev, "IBS RX <WAKE/ACK/SLEEP> t=%u", jiffies_to_msecs(jiffies));
```

* **Before** calling `qca_read_soc_version()`:

```c
bt_dev_info(hdev, "EDL PATCH_VER about to tx (guard=%dms left)",
            qca_awake_guard_active(qca) ?
              jiffies_to_msecs(qca->awake_guard_expires - jiffies) : 0);
```

* **After** getting the PATCH\_VER skb:

```c
bt_dev_info(hdev, "EDL PATCH_VER OK in ~%d ms", jiffies_to_msecs(jiffies - t_start));
```

**btqca.c**

* Around TLV send loop:

```c
bt_dev_dbg(hdev, "TLV seg %zu/%zu (%zu bytes) mode=%u",
           sent_bytes, total, segsize, config->dnld_mode);
```

* When forcing sync:

```c
if (tlv_force_sync)
  bt_dev_info(hdev, "TLV: forcing per-segment acks (sync)");
```

* For the final RESET:

```c
bt_dev_info(hdev, "QCA final RESET (pre-quiet 40ms, 3s timeout)");
```

These breadcrumbs let you pinpoint whether timeouts originate **before** any RX (IBS not awake) or after a few segments (IBS slept / event misparse), without spamming the log.

---

## 4) Validation plan — ≤4 attempts per boot (≤40 s each)

### Common preconditions (each attempt)

* `rfkill` unblocked (you already confirmed).
* Ensure firmware assets present (uncompressed):
  `/data/firmware/qca/crbtfw21.tlv`, `/data/firmware/qca/crnv21.bin`.
* **Keep TLV at 115200 first** to simplify: `echo Y > /sys/module/hci_uart/parameters/patch115200`
* Use:
  `timeout 35 btattach -B /dev/ttyHS0 -P qca -S 115200 > /data/local/tmp/btattach.out 2>&1`

---

### Attempt A (the “guarded baseline”) — **expected to pass PATCH\_VER and start TLV**

**Toggles:** `patch115200=Y`, `tlv_force_sync=Y`
**Kernel:** with patches §2.1 + §2.3 (RESET=3s, quiet settle, awake guard active)

**Expected dmesg/btattach breadcrumbs:**

* `hci0: QCA Version Request` → `QCA controller version 0x02140201` (or similar) **within \~200–400 ms**.
* `QCA Patch config` (short CC).
* `QCA Downloading qca/crbtfw21.tlv`

  * Repeated `TLV seg N` messages; **no “0xfc00 tx timeout”**.
* `QCA Downloading qca/crnv21.bin`
* `QCA final RESET (pre-quiet 40ms, 3s timeout)` → **no timeout**.
* `hci0: QCA setup on UART is completed` → `hci0 up` with `btmgmt --index 0 info` okay.

**Pass criteria:** PATCH\_VER succeeds; at least **first 4–8 TLV segments** ack cleanly; no RESET timeout at end.

---

### Attempt B (prove IBS was the culprit)

**Toggles:** `patch115200=Y`, `tlv_force_sync=N` (let TLV honor skip‑VSE)
**Kernel:** same as Attempt A.

**Expected:** PATCH\_VER still fine; some platforms will still pass TLV; if you regress to `0xfc00 tx timeout`, that **implicates the skip‑VSE mode** as fragile on your unit — stick with `tlv_force_sync=Y` for production. ([Code Browser][3])

---

### Attempt C (pre‑bump to 3M after PATCH\_VER; exercises 0x92 handling)

**Toggles:** `patch115200=N` (host will bump after controller’s baud command), `tlv_force_sync=Y`.
**Kernel:** ensure the **“drop vendor 0x92”** patch (§2.2) is present.
**Expected breadcrumbs:**

* After `qca_set_baudrate(QCA_OPER_SPEED)`: log shows `vendor event 0x92` is **dropped** and a **post‑flip settle \~300–400 ms** is respected. Patch download now runs much faster. **No TLV response size mismatch** or `0xfc00 tx timeout`. ([Linux-Kernel Archive][2])

**Pass criteria:** PATCH\_VER OK, TLV/NVM complete, final RESET OK.

---

### Attempt D (stress window within 35–40 s)

**Toggles:** `patch115200=Y`, `tlv_force_sync=Y`, but reduce artificial pacing to minimum (e.g., 0–3 ms early segment delays or remove pacing entirely).
**Goal:** Verify you still finish within the USB/ADB stability window with the conservative sync policy.

---

## Time constants (use these unless you see evidence to adjust)

* **IBS awake guard** (ignore SLEEP\_IND): **600 ms** starting just before `qca_read_soc_version()`.
  Optionally extend (by resetting the deadline) to cover only the **first \~8 KB** of TLV.
* **Pre‑RESET quiet**: **40–60 ms**.
* **Final RESET timeout**: **3000 ms** (3 s).
* **PATCH\_VER (EDL 0xFC00/0x19) timeout**: leave at **HCI\_INIT\_TIMEOUT (≈2000 ms)**; it should answer in <300 ms. ([Code Browser][5])
* **TLV per‑segment pacing at 115200** (only if needed): **7–9 ms** between segments for the **first 8 KB**; segment size **MAX\_SIZE\_PER\_TLV\_SEGMENT (243 bytes)**. ([Code Browser][6])
* **Baud change vendor event (0x92) wait**: **100–300 ms** after host flip is fine; keep your **\~300 ms + small settle** if you do pre‑bump. ([Linux-Kernel Archive][2])

---

## SDM845 prerequisites worth checking (once)

* **Device tree / power:** ensure the Bluetooth power/regulator path used by `hci_qca` is present and rfkill actually asserts BT\_EN/regulators. (Your baseline successfully read ROM → power path is likely OK.)
* **UART driver votes:** `hci_qca` uses serial clock vote helpers to gate clocks with IBS; our awake guard avoids an aggressive vote‑off in the first 600 ms. No additional TLMM/pinctrl tweaks needed at 115200 for ttyHS0. (This sequence is standard on 845 vendor kernels.)

---

## References that back the changes

* **Drop stray vendor event 0x92 (baud change)** — prevents the event being misinterpreted as a TLV response; also increases robustness of the speed flip. ([Linux-Kernel Archive][2])
* **Upstream WCN399x flow & RESET positioning** — read version, download TLV+NVM, then RESET; modern btqca.c shows RESET after downloads. ([Code Browser][5])
* **TLV download mode & per‑segment acks** — Rome ≥3.2 may skip acks except the last; forcing sync eliminates the EDL stalls on flaky links. ([Code Browser][3])
* **Discard RX during boot** — avoids mis‑reassembly/timeouts on WCN3990 during multi‑baud; if missing in 4.9, add equivalent. ([Linux Kernel Archive][4])
* **EDL opcodes** — PATCH CMD opcode 0xFC00, `PATCH_VER_REQ_CMD` 0x19, etc., for context when grepping traces. ([Android Source][7])

---

## What I expect to change in your logs with these patches

* The early `HCI Reset (0x0C03) tx timeout` **disappears entirely** (because we don’t send that RESET anymore before PATCH\_VER).
* `0xfc00 tx timeout` during PATCH\_VER/TLV is **replaced** by steady segment progress; if you see any `TLV response size mismatch`, confirm the 0x92 drop logic is active (Attempt C).
* Final RESET succeeds within \~150–600 ms (we gave it 3 s budget).

---

## If anything still hangs

* Flip `tlv_force_sync=N` (Attempt B) to test sensitivity to skip‑VSE mode. ([Code Browser][3])
* Extend the **awake guard** to 1200 ms (single line change) to cover all of PATCH\_VER + TLV start on slow units.
* If PATCH\_VER itself times out: log the **very first byte** seen on H4 RX after the command (dump `skb->data[0..7]`); if it’s an IBS frame instead of a VSE/CC, your guard wasn’t active or IBS was already ASLEEP — bump the guard start to **just before** queuing PATCH\_VER.

---

I kept this tight and minimally invasive so you can A/B quickly. If you want me to adapt the diffs to your exact 4.9 file layouts (e.g., `rome_*` symbol names) or fold the guard into a runtime module param for fast toggling, say the word and I’ll produce that variant.

[1]: https://android.googlesource.com/kernel/common/%2B/P-RELEASE_4.4/drivers/bluetooth/btqca.c?utm_source=chatgpt.com "drivers/bluetooth/btqca.c - kernel/common - Git at Google"
[2]: https://lkml.iu.edu/1903.1/04409.html "Linux-Kernel Archive: [PATCH v2 2/2] Bluetooth: hci_qca: wcn3990: Drop baudrate change vendor event"
[3]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html?utm_source=chatgpt.com "btqca.c source code [linux/drivers/bluetooth ..."
[4]: https://lkml.indiana.edu/1811.0/04170.html?utm_source=chatgpt.com "[PATCH v1 4/4] Bluetooth: hci_qca: Fix frame reassembly ..."
[5]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html "btqca.c source code [linux/drivers/bluetooth/btqca.c] - Codebrowser "
[6]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.h.html?utm_source=chatgpt.com "btqca.h source code [linux/drivers/bluetooth ..."
[7]: https://android.googlesource.com/kernel/common/%2B/5e99456c20f7/drivers/bluetooth/btqca.h?utm_source=chatgpt.com "drivers/bluetooth/btqca.h - kernel/common - Git at Google"
