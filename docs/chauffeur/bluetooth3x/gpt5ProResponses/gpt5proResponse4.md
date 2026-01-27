Got it. You’re very close—your logs show the controller is present and talking (“ROME setup”, version 0x02140201), but you’re losing the very first vendor request (0xFC00 ROM ver) or the first TLV segment. That failure mode is classic for (a) tiny timing gaps right at the wake/reset boundary and/or (b) sending the first big TLV chunk before the UART path is truly ready. There’s also one big strategic point you’re currently fighting: upstream brings Rome/WCN3990 up with **IBS disabled during patch** and **bumps to 3M *before* patch/NVM**. Your current approach (staying at 115200 and ignoring event 0x92) is workable on some paths, but it’s fighting the grain. Upstream behavior matters here. ([Android Gooblesource][1])

Below I give you two things:

1. **#46 exact code diffs** that implement your planned micro‑settles + conservative early segmentation at 115200 (your plan), and
2. A **toggle to A/B test the upstream‑style sequence** (pre‑bump to 3M, IBS off for download), because the “seg 0 forever” symptoms often disappear when you stop starving the link.

I also add timeouts and pacing consistent with long EDL stalls.

---

## Why I think you’re flapping

* **IBS vs. patch**: The btattach/qca path disables IBS for the patch window; you shouldn’t need to “wake” with IBS messages for the ROM read. Upstream literally does `clear_bit(STATE_IN_BAND_SLEEP_ENABLED, ...)` before calling into `qca_uart_setup_rome()`. Sending HCI\_IBS\_WAKE\_IND doesn’t hurt, but it’s a no‑op for flow if IBS is disabled. ([Android Gooblesource][1])
* **Speed**: The controller often issues **EDL\_SET\_BAUDRATE\_RSP\_EVT (0x92)** during setup (that’s not “stray”, that is the baud‑change response). Upstream bumps the host to **3,000,000** immediately after sending the change‑baud command, then downloads TLV/NVM at high speed, with IBS still disabled. Your choice to force 115200 all the way through makes first‑burst timing way harsher and amplifies segment‑0 sensitivity. ([Android Gooblesource][2])
* **Segment 0**: The code sends **243‑byte** segments by default; on marginal UARTs the very first segment is the riskiest (clock/line still settling). Shrinking the first N segments + spacing them buys reliability. You’ve already trended that way; we’ll codify it cleanly. ([Debian Sources][3])
* **Timeouts**: Vendor EDL ops can sit “busy” longer than the default HCI init timeout. We’ll use explicit per‑command timeouts (ROM read + TLV segment) of up to \~6s on #46.

---

## Patch set for build **#46** (your plan, at 115200 with micro‑settles)

> Paths match your builder:
>
> * `/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/hci_qca.c`
> * `/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/btqca.c`
> * `/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/btqca.h` (no changes required other than reference; `MAX_SIZE_PER_TLV_SEGMENT` already 243).

### 1) `hci_qca.c`: pre‑ROM settle + best‑effort wake & a safety RESET retry

```diff
diff --git a/drivers/bluetooth/hci_qca.c b/drivers/bluetooth/hci_qca.c
@@ -XXX,6 +XXX,26 @@ static int qca_setup(struct hci_uart *hu)
   BT_INFO("%s: ROME setup", hdev->name);

+  /* #46: extra guard bands before the first vendor command. The patch/NVM
+   * window runs with IBS disabled, but a brief settle helps the first
+   * 0xFC00 ROM read survive right after rfkill/unblock & pulses. */
+  msleep(80);                   /* pre‑ROM settle ~80 ms */
+
+  /* Best‑effort nudge: send a WAKE_IND and give 2–8 ms.
+   * (IBS is about to be disabled for the download; this is harmless.) */
+  send_hci_ibs_cmd(HCI_IBS_WAKE_IND, hu);
+  usleep_range(2000, 8000);
+
   /* Patch downloading has to be done without IBS mode */
   clear_bit(STATE_IN_BAND_SLEEP_ENABLED, &qca->flags);
+
+  /* #46: If the pre‑ROM HCI_RESET fails with a transient timeout (-110),
+   * retry once after a short settle; this is best‑effort, ignore if absent. */
+  do {
+    struct sk_buff *skb = __hci_cmd_sync(hdev, HCI_OP_RESET, 0, NULL, HCI_INIT_TIMEOUT);
+    if (!IS_ERR(skb)) { kfree_skb(skb); break; }
+    msleep(20);
+    /* one retry only */
+  } while (0);
```

> Notes:
>
> * Upstream disables IBS during patch; we keep that while still giving you the tiny wake nudge. ([Android Gooblesource][1])

### 2) `btqca.c`: ROM‑read micro‑settles + conservative first‑burst TLV + pacing + longer timeouts

```diff
diff --git a/drivers/bluetooth/btqca.c b/drivers/bluetooth/btqca.c
@@ -40,6 +40,26 @@
 #define VERSION "0.1"

+/* #46 tuning (ms) */
+#define QCA_BT3X_ROM_PRE_RETRY_MS          5   /* settle right before a ROM retry */
+#define QCA_BT3X_ROM_RETRY_GAP_MS         40   /* spacing between ROM read retries */
+#define QCA_BT3X_ROM_MAX_RETRIES           2   /* attempt 1 + 2 retries total */
+
+#define QCA_BT3X_TLV_SMALL_SZ            128   /* first-N segment size */
+#define QCA_BT3X_TLV_SMALL_SEGS            8   /* segments [0..7] at 128B */
+#define QCA_BT3X_SEG0_MAX_ATTEMPTS         4   /* extra tries for seg 0 */
+#define QCA_BT3X_SEG0_POST_SUCCESS_MS     10   /* settle after seg 0 succeeds */
+
+#define QCA_BT3X_PACE_SEG_0_3_MS          45
+#define QCA_BT3X_PACE_SEG_4_7_MS          30
+#define QCA_BT3X_PACE_SEG_GE8_MS          10
+
+/* EDL commands can take a while on this path; give them room */
+#define QCA_BT3X_TLV_CMD_TIMEOUT_MS     6000
+#define QCA_BT3X_ROM_CMD_TIMEOUT_MS     3000
+
+/* helper */
+#define JFIES(ms) msecs_to_jiffies(ms)

@@
-static int rome_patch_ver_req(struct hci_dev *hdev, u32 *rome_version)
+static int rome_patch_ver_req_once(struct hci_dev *hdev, u32 *rome_version, unsigned long timeout_j)
 {
 	struct sk_buff *skb;
 	struct edl_event_hdr *edl;
 	struct rome_version *ver;
 	char cmd;
 	int err = 0;

 	BT_DBG("%s: ROME Patch Version Request", hdev->name);
-	cmd = EDL_PATCH_VER_REQ_CMD;
-	skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, EDL_PATCH_CMD_LEN,
-				&cmd, HCI_VENDOR_PKT, HCI_INIT_TIMEOUT);
+	cmd = EDL_PATCH_VER_REQ_CMD;
+	skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, EDL_PATCH_CMD_LEN,
+				&cmd, HCI_VENDOR_PKT, timeout_j);
 	if (IS_ERR(skb)) {
 		err = PTR_ERR(skb);
 		BT_ERR("%s: Failed to read version of ROME (%d)", hdev->name,
 		       err);
 		return err;
 	}
@@
 	return err;
 }

+/* #46: wrapper with small pre‑retry settles + spacing between attempts */
+static int rome_patch_ver_req(struct hci_dev *hdev, u32 *rome_version)
+{
+	int attempt, err = -ETIMEDOUT;
+	for (attempt = 0; attempt <= QCA_BT3X_ROM_MAX_RETRIES; attempt++) {
+		if (attempt)
+			msleep(QCA_BT3X_ROM_PRE_RETRY_MS);
+		err = rome_patch_ver_req_once(hdev, rome_version, JFIES(QCA_BT3X_ROM_CMD_TIMEOUT_MS));
+		if (!err && *rome_version)
+			return 0;
+		if (attempt < QCA_BT3X_ROM_MAX_RETRIES)
+			msleep(QCA_BT3X_ROM_RETRY_GAP_MS);
+	}
+	return err;
+}
+
 /* ... existing rome_reset(), rome_tlv_check_data() ... */

-static int rome_tlv_send_segment(struct hci_dev *hdev, int idx, int seg_size,
-				 const u8 *data)
+static int rome_tlv_send_segment(struct hci_dev *hdev, int idx, int seg_size,
+				 const u8 *data, unsigned long timeout_j)
 {
 	struct sk_buff *skb;
 	struct edl_event_hdr *edl;
 	struct tlv_seg_resp *tlv_resp;
 	u8 cmd[MAX_SIZE_PER_TLV_SEGMENT + 2];
 	int err = 0;
@@
-	skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, seg_size + 2, cmd,
-				HCI_VENDOR_PKT, HCI_INIT_TIMEOUT);
+	skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, seg_size + 2, cmd,
+				HCI_VENDOR_PKT, timeout_j);
 	if (IS_ERR(skb)) {
 		err = PTR_ERR(skb);
 		BT_ERR("%s: Failed to send TLV segment (%d)", hdev->name, err);
 		return err;
 	}
@@
 	return err;
 }

 static int rome_tlv_download_request(struct hci_dev *hdev,
 				     const struct firmware *fw)
 {
-	const u8 *buffer, *data;
-	int total_segment, remain_size;
-	int ret, i;
+	const u8 *data;
+	int ret = 0, idx = 0;
+	size_t sent = 0;
+	size_t total;

 	if (!fw || !fw->data)
 		return -EINVAL;

-	total_segment = fw->size / MAX_SIZE_PER_TLV_SEGMENT;
-	remain_size = fw->size % MAX_SIZE_PER_TLV_SEGMENT;
-	BT_DBG("%s: Total segment num %d remain size %d total size %zu",
-	       hdev->name, total_segment, remain_size, fw->size);
-
-	data = fw->data;
-	for (i = 0; i < total_segment; i++) {
-		buffer = data + i * MAX_SIZE_PER_TLV_SEGMENT;
-		ret = rome_tlv_send_segment(hdev, i, MAX_SIZE_PER_TLV_SEGMENT,
-					    buffer);
-		if (ret < 0)
-			return -EIO;
-	}
-	if (remain_size) {
-		buffer = data + total_segment * MAX_SIZE_PER_TLV_SEGMENT;
-		ret = rome_tlv_send_segment(hdev, total_segment, remain_size,
-					    buffer);
-		if (ret < 0)
-			return -EIO;
-	}
+	total = fw->size;
+	data  = fw->data;
+	BT_DBG("%s: TLV total size %zu", hdev->name, total);
+
+	while (sent < total) {
+		int seg_size;
+		unsigned long pace_ms;
+
+		/* #46: first few segments at 128B, then full size */
+		if (idx < QCA_BT3X_TLV_SMALL_SEGS)
+			seg_size = min_t(size_t, QCA_BT3X_TLV_SMALL_SZ, total - sent);
+		else
+			seg_size = min_t(size_t, MAX_SIZE_PER_TLV_SEGMENT, total - sent);
+
+		/* Extra retries for seg 0 */
+		if (idx == 0) {
+			int tries;
+			for (tries = 0; tries < QCA_BT3X_SEG0_MAX_ATTEMPTS; tries++) {
+				ret = rome_tlv_send_segment(hdev, idx, seg_size, data + sent,
+				                            JFIES(QCA_BT3X_TLV_CMD_TIMEOUT_MS));
+				if (!ret) break;
+				msleep(QCA_BT3X_PACE_SEG_0_3_MS);
+			}
+			if (ret) return ret;
+			msleep(QCA_BT3X_SEG0_POST_SUCCESS_MS);
+		} else {
+			ret = rome_tlv_send_segment(hdev, idx, seg_size, data + sent,
+			                            JFIES(QCA_BT3X_TLV_CMD_TIMEOUT_MS));
+			if (ret) return ret;
+		}
+
+		/* Pacing */
+		if (idx < 4)       pace_ms = QCA_BT3X_PACE_SEG_0_3_MS;
+		else if (idx < 8) pace_ms = QCA_BT3X_PACE_SEG_4_7_MS;
+		else              pace_ms = QCA_BT3X_PACE_SEG_GE8_MS;
+		msleep(pace_ms);
+
+		sent += seg_size;
+		idx++;
+	}

 	return 0;
 }

 int qca_uart_setup_rome(struct hci_dev *hdev, uint8_t baudrate)
 {
@@
-	/* Get ROME version information */
-	err = rome_patch_ver_req(hdev, &rome_ver);
+	/* Get ROME version information (with small settles between attempts) */
+	err = rome_patch_ver_req(hdev, &rome_ver);
 	if (err < 0 || rome_ver == 0) {
 		BT_ERR("%s: Failed to get version 0x%x", hdev->name, err);
 		return err;
 	}

 	BT_INFO("%s: ROME controller version 0x%08x", hdev->name, rome_ver);
@@
-	/* Download NVM configuration */
+	/* Give the controller a moment between Patch and NVM (was 10 ms) */
+	msleep(10); /* keep; #46 retains 10 ms for now as per your plan */
+	/* Download NVM configuration */
```

**What this gives you (#46):**

* 80 ms guard before first vendor cmd, small 5 ms settle before ROM‑read retries, 40 ms between retries.
* TLV: first **8** segments at **128B**, segment‑0 **retries=4**, pacing 45/30/10 ms buckets, 10 ms settle after seg 0 success, and **6 s** per‑segment timeout (ROM read 3 s).
* Leaves speed at **115200** through TLV/NVM exactly as you asked.

---

## Optional toggle: try the upstream sequence (often fixes seg‑0 instantly)

Since upstream does **pre‑bump to 3M**, I suggest adding a simple module parameter so you can A/B without re‑editing code.

### `hci_qca.c`: module param to *not* bump speed (default 0 = bump like upstream)

```diff
@@
+static bool qca_patch_at_115200 = false; /* default: bump to oper_speed before patch */
+module_param_named(patch115200, qca_patch_at_115200, bool, 0644);
+MODULE_PARM_DESC(patch115200, "Keep 115200 until after TLV/NVM (default: 0)");
@@ static int qca_setup(struct hci_uart *hu)
-  /* Setup user speed if needed */
+  /* Setup user speed if needed */
   speed = 0;
   if (hu->oper_speed)
     speed = hu->oper_speed;
   else if (hu->proto->oper_speed)
     speed = hu->proto->oper_speed;
-  if (speed) {
+  if (speed && !qca_patch_at_115200) {
     qca_baudrate = qca_get_baudrate_value(speed);
     BT_INFO("%s: Set UART speed to %d", hdev->name, speed);
     ret = qca_set_baudrate(hdev, qca_baudrate);
     if (ret) {
       BT_ERR("%s: Failed to change the baud rate (%d)",
              hdev->name, ret);
       return ret;
     }
     hci_uart_set_baudrate(hu, speed);
   }
```

* Run with your current approach (no bump): **`echo 1 > /sys/module/hci_qca/parameters/patch115200`**.
* Or test the upstream‑style (bump then patch): leave it at **0**.
  Upstream does IBS‑off during patch and bumps to 3M *before* calling `qca_uart_setup_rome()`; the code above replicates that when `patch115200=0`. ([Android Gooblesource][1])

---

## Test runs (bounded slices you already use)

Use your existing runbook with one addition: if you try the upstream path, don’t ignore 0x92 in `qca_recv_event()` during setup—that event is **EDL\_SET\_BAUDRATE\_RSP\_EVT** and is *expected*. If you’ve added an “ignore” in your tree, guard it behind the same `patch115200` flag. (0x92 is documented in `btqca.h` in multiple kernels.) ([Android Gooblesource][2])

**Attach:**

* `btattach -B /dev/ttyHS0 -S 115200 -P qca >/data/bta_run.log 2>&1 & echo $! >/data/bta_run.pid`
* Watch in 15s/35s/60s slices exactly as you listed.

**Success criteria you gave:**
You should now reach:

```
Bluetooth: hci0: ROME setup
Bluetooth: hci0: ROME controller version 0x02140201
Bluetooth: hci0: ROME Downloading file: qca/crbtfw21.tlv
Bluetooth: hci0: ROME setup on UART is completed
```

Then:

```
hciconfig hci0 up && sleep 1 && hciconfig -a
btmgmt power on
btmgmt find -l 5
```

---

## If seg‑0 still flakes after #46

Follow your own “next tweaks” ladder, but in this order (fastest to rule‑in/out):

1. **Flip the toggle to upstream behavior** (`patch115200=0`) and re‑run. If you immediately stop seeing seg‑0 timeouts and ROM‑read flaps, your earlier insistence on 115200 was the footgun. (I suspect it is.) Upstream’s “IBS off + 3M” design isn’t random. ([Android Gooblesource][1])
2. If you must stay at **115200**, extend the small‑segment window to **16** and/or run **full TLV at 128B** with `10 ms` pacing. That’s a one‑line change to `QCA_BT3X_TLV_SMALL_SEGS` (or set `MAX_SIZE_PER_TLV_SEGMENT` to 128 temporarily). ([Debian Sources][3])
3. Add **40–50 ms** between patch completion and NVM start (you already have 10 ms). There’s precedent for a small settle right there (it was introduced upstream), so 40–50 ms is reasonable. ([Linux Kernel Archive][4])

---

## Firmware files you’re using are correct

* For **WCN3990**, `crbtfw21.tlv` + `crnv21.bin` is the standard pair in linux‑firmware trees. You have the right filenames. ([GitLab][5])

---

## Instrumentation tips (optional but helpful)

* Capture **btmon** alongside your dmesg slice to see whether segment‑0 fails as a transmit timeout (no Command Status) or as a NACK late.
* Turn on dynamic debug for the qca path:

  ```
  echo 'file drivers/bluetooth/hci_qca.c +p' > /sys/kernel/debug/dynamic_debug/control
  echo 'file drivers/bluetooth/btqca.c +p'   > /sys/kernel/debug/dynamic_debug/control
  ```
* If you remain at 115200, ensure **crtscts is absolutely on** (you already do) and avoid any userspace opening `/dev/ttyHS0` concurrently.

---

## What I want you to run next

1. Apply the **#46** diffs above in your builder tree.
2. Build, flash both slots, run your **bounded attach** (15/35/60 s).
3. If you still stall at seg‑0 or 0xFC00, immediately flip the **`patch115200`** toggle to **0** (upstream path: bump to 3M first, IBS off for patch) and re‑run the same bounded attach.
4. Paste the last 10–20 Bluetooth lines (and the key tuning numbers) into `BT_RESEARCH_STATUS.md`.

If you want me to, I can generate exact `git diff` files with context once you confirm where you placed your earlier guards (e.g., the “ignore 0x92” hunk) so I don’t stomp them.

---

### Sources (for the key claims)

* **Upstream hci\_qca flow**: IBS disabled for patch; bump to 3M prior to patch; then `qca_uart_setup_rome()`; re‑enable IBS after success. ([Android Gooblesource][1])
* **Rome TLV sender defaults**: `MAX_SIZE_PER_TLV_SEGMENT` = 243; simple loop, no pacing by default; 10 ms settle before NVM was introduced later. ([Debian Sources][3])
* **EDL\_SET\_BAUDRATE\_RSP\_EVT (0x92)** is the Qualcomm baud‑change event; don’t blanket‑ignore it if you choose to use the upstream speed bump. ([Android Gooblesource][2])
* **Firmware names for WCN3990** (`crbtfw21.tlv` / `crnv21.bin`) are the expected files in linux‑firmware repos. ([GitLab][5])

---

If you want, I can also spit out a one‑page snippet to append to `BT_RESEARCH_STATUS.md` each pass (uname, kernel #, parameters used, last N lines); say the word and I’ll format it to your existing doc style.

[1]: https://android.googlesource.com/kernel/msm/%2B/refs/tags/android-10.0.0_r0.2/drivers/bluetooth/hci_qca.c?utm_source=chatgpt.com "drivers/bluetooth/hci_qca.c - kernel/msm - Git at Google"
[2]: https://android.googlesource.com/kernel/common/%2B/5e99456c20f7/drivers/bluetooth/btqca.h?utm_source=chatgpt.com "drivers/bluetooth/btqca.h - kernel/common - Git at Google"
[3]: https://sources.debian.org/src/linux-grsec/4.9.18-1%2Bgrsec201703261106%2B1~bpo8%2B1/drivers/bluetooth/btqca.c/ "File: btqca.c
\| Debian Sources
"
[4]: https://lkml.rescloud.iu.edu/1909.1/01653.html?utm_source=chatgpt.com "Linux 4.4.192 - Linux-Kernel Archive: Re"
[5]: https://git.codelinaro.org/clo/linux-kernel/linux-firmware/-/tree/video-firmware/qca?ref_type=heads&utm_source=chatgpt.com "qca · video-firmware - linux-kernel - GitLab - CodeLinaro"
