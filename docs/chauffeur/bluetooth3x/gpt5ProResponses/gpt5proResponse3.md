Chris — this looks like two issues layered on top of each other:

1. **The first vendor command (ROM version read, 0xFC00 / 0x19)** is getting dropped right at the wake/reset boundary → your retries hit `-110`.
2. **When you do get past ROM read, TLV seg‑0 intermittently “times out”** → on older 4.9 trees this is *usually* a logic mismatch: the driver waits for a Vendor‐Specific Event (VSE) ack for every TLV segment, but **WCN3990 can skip VSE acks for all but the last segment** based on the *download\_mode* in the TLV header. If your 4.9 base lacks the “skip VSE” handling and CC injection, seg‑0 will look like a timeout no matter how much you pace it. Upstream `btqca` explicitly reads `download_mode` and, when it indicates “skip VSE”, it **doesn’t wait** for a response for early segments and **injects a Command Complete** after the last segment to keep the core state machine happy. ([codebrowser.dev][1])

There’s also a *known* WCN3990 quirk where the **stray vendor event from the *baud‑rate change*** (after the port bump) gets misinterpreted as a TLV response, causing “TLV response size mismatch” / early failures. Upstream fixes that by **dropping the orphan vendor event** for WCN3990. (You already implemented an ignore of vendor 0x92; below is the canonical version to backport.) ([lkml.iu.edu][2])

Below is a concrete #46 patch plan that combines: (A) robust pre‑ROM wake/reset settles, (B) the **skip‑VSE** semantics + CC injection (if you don’t already have them), (C) the official WCN3990 **drop‑vendor‑event** fix, and (D) your pacing knobs. Then the runbook to build/flash/test, and fallbacks if seg‑0 still flakes.

---

## A. Pre‑ROM wake/reset (hci\_qca.c)

**Goals:** ensure controller is awake *before* `EDL_PATCH_VER_REQ_CMD (0x19)` and allow one lightweight RESET retry if the very first command hits `-110`.

Apply to
`/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/hci_qca.c`

```diff
@@ -XXX,6 +XXX,56 @@ static int qca_setup(struct hci_uart *hu)
+       /* --- Pre-ROM wake/settle sequence for WCN3990 on UART --- */
+       if (qca_is_wcn399x(hu)) {
+               /* Nudge IBS awake before first vendor command. */
+               qca_send_ibs_cmd(hu->hdev, HCI_IBS_WAKE_IND);
+               /* tiny udelay range mitigates 115200bps jitter */
+               usleep_range(2000, 8000);
+       }
+
+       /* Give the ROM a bit more quiet before the first vendor opcode */
+       msleep(80);   /* was 50ms; empirical headroom for this board */
+
+       /* Best-effort HCI_RESET at init speed, to clear any stale state */
+       err = __qca_send_reset(hu->hdev); /* thin wrapper around __hci_cmd_sync */
+       if (err == -ETIMEDOUT) {
+               /* One retry with a small settle helps when IBS wakes late */
+               msleep(20);
+               err = __qca_send_reset(hu->hdev);
+       }
+       if (err && err != -ETIMEDOUT)
+               bt_dev_dbg(hu->hdev, "pre-ROM HCI_RESET non-fatal err=%d", err);
+
+       /* Tiny settle right before ROM read retry loop (see B) */
+       qca->pre_rom_retry_settle_ms = 5;
```

*(Rationale: the IBS wake/ack path is present in `hci_qca` and is the correct knob to avoid losing the first vendor command. Upstream `hci_qca` implements HCI\_IBS state, wake retransmit timers, etc.; we’re just biasing the very first command toward an awake link.)* ([Android Gooblesource][3])

---

## B. ROM read + TLV transfer robustness (btqca.c / btqca.h)

**Must‑have upstream semantics** (many 4.9 trees miss these for WCN3990):

* **Honor TLV `download_mode`:** for Rome ≥3.2/WCN399x the controller can skip VSE until the *last* segment. Driver must *not* block waiting for a VSE per segment in that mode. ([codebrowser.dev][1])
* **Inject a Command Complete** once TLV is done when using the skip‑VSE modes, to prevent HCI core timeouts. ([codebrowser.dev][1])

You likely already have some of this, but here’s a minimal, safe backport pattern. Apply to
`/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/btqca.h`:

```diff
+#define EDL_PATCH_CMD_OPCODE      0xFC00
+#define EDL_PATCH_TLV_REQ_CMD     0x1E
+#define EDL_PATCH_VER_REQ_CMD     0x19
+enum qca_tlv_dnld_mode {
+  QCA_SKIP_EVT_NONE = 0,
+  QCA_SKIP_EVT_VSE  = 1,  /* VSE skipped until last segment */
+  QCA_SKIP_EVT_VSE_CC = 2 /* ditto and needs CC injection */
+};
+#define MAX_SIZE_PER_TLV_SEGMENT  243
```

…and to
`/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/btqca.c` (key excerpts; align with your file):

```diff
@@ static int qca_tlv_check_data(struct hci_dev *hdev, struct qca_fw_config *config, ...)
-  config->dnld_mode = QCA_SKIP_EVT_NONE;
+  config->dnld_mode = QCA_SKIP_EVT_NONE;
   ...
   case TLV_TYPE_PATCH:
     ...
-    config->dnld_mode = tlv_patch->download_mode;
-    config->dnld_type = config->dnld_mode;
+    /* For Rome 3.2+/WCN399x, download_mode controls VSE skipping. */
+    config->dnld_mode = tlv_patch->download_mode;
+    config->dnld_type = config->dnld_mode;
@@ static int qca_tlv_send_segment(..., enum qca_tlv_dnld_mode mode, ...)
-  skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, seg_size+2, cmd, HCI_EV_VENDOR, HCI_INIT_TIMEOUT);
+  if (mode == QCA_SKIP_EVT_VSE || mode == QCA_SKIP_EVT_VSE_CC) {
+      /* Controller won't send VSE for this segment: fire-and-forget. */
+      return __hci_cmd_send(hdev, EDL_PATCH_CMD_OPCODE, seg_size + 2, cmd);
+  }
+  skb = __hci_cmd_sync_ev(hdev, EDL_PATCH_CMD_OPCODE, seg_size+2, cmd,
+                          HCI_EV_VENDOR, HCI_INIT_TIMEOUT);
@@ static int qca_download_firmware(...)
   while (remain > 0) {
-    int segsize = min(MAX_SIZE_PER_TLV_SEGMENT, remain);
+    int segsize;
+    /* Your pacing/size windowing: first 8 segments at 128B, then 243B */
+    if (i < 8) segsize = min(128, remain);
+    else       segsize = min(MAX_SIZE_PER_TLV_SEGMENT, remain);
     ...
     /* The last segment is always acked regardless of download_mode */
     if (!remain || segsize < MAX_SIZE_PER_TLV_SEGMENT)
         config->dnld_mode = QCA_SKIP_EVT_NONE;
@@
-  if (config->dnld_type == QCA_SKIP_EVT_VSE_CC || config->dnld_type == QCA_SKIP_EVT_VSE)
-      ret = qca_inject_cmd_complete_event(hdev);
+  /* When VSE was skipped during dnld, inject a CC to keep HCI happy. */
+  if (config->dnld_type == QCA_SKIP_EVT_VSE_CC || config->dnld_type == QCA_SKIP_EVT_VSE)
+      ret = qca_inject_cmd_complete_event(hdev);
+
+  /* Conservative settle before NVM (helps on noisy systems). */
+  msleep(10);
```

*(This is exactly how upstream handles WCN399x TLV ack semantics; the size windowing preserves your seg‑0–7 128‑byte plan.)* ([codebrowser.dev][4])

**ROM read retry settle:** in your retry path around `qca_read_soc_version()` add:

```diff
- bt_dev_err(hdev, "Retry ROM version read after 40ms (attempt %d/%d err=%d)", ...)
+ msleep(qca->pre_rom_retry_settle_ms ?: 5);
+ bt_dev_err(hdev, "Retry ROM version read after 40ms (+5ms settle) (attempt %d/%d err=%d)", ...)
```

Upstream ROM read uses `__hci_cmd_sync_ev(..., HCI_INIT_TIMEOUT)` (10s), not the 2s command timeout; make sure your call sites do that (value is in `include/net/bluetooth/hci.h`). ([codebrowser.dev][5])

---

## C. **WCN3990 stray vendor event** fix (canonical backport)

If you haven’t already mirrored the upstream behavior, backport this verbatim (pared to UART path) so the stray “baudrate change” vendor event is **consumed & dropped** instead of enqueued as a TLV response. Apply to
`/projects/agnos/agnos-builder/agnos-kernel-sdm845/drivers/bluetooth/hci_qca.c`

```diff
@@
+/* Track and drop the one-off vendor event after WCN3990 baud change. */
+enum qca_flags {
+  QCA_IN_BAND_SLEEP_ENABLED,
+  QCA_DROP_VENDOR_EVENT,
+};
+struct qca_data { ... unsigned long flags; struct completion drop_ev_comp; ... };
@@
+static int qca_recv_event(struct hci_dev *hdev, struct sk_buff *skb)
+{
+  struct hci_uart *hu = hci_get_drvdata(hdev);
+  struct qca_data *qca = hu->priv;
+  if (test_bit(QCA_DROP_VENDOR_EVENT, &qca->flags)) {
+    struct hci_event_hdr *hdr = (void *)skb->data;
+    if (hdr->evt == HCI_EV_VENDOR)
+      complete(&qca->drop_ev_comp);
+    kfree_skb(skb);
+    return 0;
+  }
+  return hci_recv_frame(hdev, skb);
+}
@@
- { H4_RECV_EVENT, .recv = hci_recv_frame },
+ { H4_RECV_EVENT, .recv = qca_recv_event },
@@
 static int qca_set_speed(struct hci_uart *hu, enum qca_speed_type t)
 {
   ...
-  if (is_wcn3990)
+  if (is_wcn3990) {
       hci_uart_set_flow_control(hu, true);
+      reinit_completion(&qca->drop_ev_comp);
+      set_bit(QCA_DROP_VENDOR_EVENT, &qca->flags);
+      smp_mb__after_atomic();
+  }
   ...
 error:
-  if (is_wcn3990)
+  if (is_wcn3990) {
       hci_uart_set_flow_control(hu, false);
+      if (!wait_for_completion_timeout(&qca->drop_ev_comp, msecs_to_jiffies(100))) {
+         bt_dev_err(hu->hdev, "WCN3990: baud-change vendor event timeout");
+         ret = -ETIMEDOUT;
+      }
+      clear_bit(QCA_DROP_VENDOR_EVENT, &qca->flags);
+      smp_mb__after_atomic();
+  }
```

*(This is the upstream fix that stopped TLV “response size mismatch” and related early failures on WCN3990.)* ([lkml.iu.edu][2])

---

## D. Your pacing knobs (keep, but simplify)

* Keep **first 8 segments at 128 B**, pacing seg0–3=45 ms, seg4–7=30 ms, seg≥8=10 ms; keep extra 10 ms settle after seg‑0 success.
* Add **5 ms settle** right before each ROM‑read retry (above).
* If seg‑0 still misbehaves, expand 128‑B window to **first 16 segments**; if still marginal, run entire TLV at **128 B with 10 ms inter‑segment** (this is slower but reliable on chatty systems).

> Why this order? In practice on WCN3990 the **VSE‑skip semantics** fix eliminates most seg‑0 “timeouts” because they were self‑inflicted waits; the pacing then just buys headroom against UART/IBS jitter. ([codebrowser.dev][4])

---

## Build, flash, and bounded test slices

1. **Build** kernel #46 in AGNOS builder:

```bash
cd /projects/agnos/agnos-builder
# (apply diffs above)
# build commands as you normally do for agnos-kernel-sdm845…
```

2. **Flash both slots**, reboot, reselect newest `transport_id` each slice (your script is good).

3. **Prep firmware visibility** (same as you have):

```bash
adb -t $transport shell 'echo "/lib/firmware:/usr/lib/firmware:/data/firmware:/firmware/image" > /sys/module/firmware_class/parameters/path'
adb -t $transport shell 'mount | grep -q "/data/firmware on /lib/firmware" || mount --bind /data/firmware /lib/firmware'
adb -t $transport shell 'ls -l /lib/firmware/qca | egrep "crbtfw21|crnv21"'
```

4. **Quiet logs & stop services** (as you listed), then **bounded attach**:

```bash
# Clean pulses + state
adb -t $transport shell 'pkill -f btattach || true; rfkill block bluetooth; sleep 0.2; rfkill unblock bluetooth'
adb -t $transport shell 'stty -F /dev/ttyHS0 2400 -echo -crtscts; printf "\xC0" > /dev/ttyHS0; sleep 0.03'
adb -t $transport shell 'stty -F /dev/ttyHS0 115200 -echo -crtscts; printf "\xFC" > /dev/ttyHS0; sleep 0.20'
adb -t $transport shell 'stty -F /dev/ttyHS0 115200 -echo crtscts'

# Logs
adb -t $transport shell 'dmesg -n 7; dmesg -C || true; (dmesg -w > /data/dmesg_bt.txt 2>&1 & echo $! > /data/dmesg_bt.pid)'

# Attach
adb -t $transport shell 'btattach -B /dev/ttyHS0 -S 115200 -P qca >/data/bta_run.log 2>&1 & echo $! > /data/bta_run.pid'
# Inspect at ~15s / 35s / 60s
adb -t $transport shell 'tail -n 300 /data/dmesg_bt.txt | egrep "Bluetooth:|hci0|ROME|QCA|ttyHS0|Downloading|Failed|timeout|segment" | tail -n 200'
adb -t $transport shell 'ls -l /sys/class/bluetooth'
```

5. **On success**:

```bash
adb -t $transport shell 'hciconfig hci0 up; sleep 1; hciconfig -a'
adb -t $transport shell 'btmgmt power on; btmgmt find -l 5'
```

Success criteria you stated should now show:

* `Bluetooth: hci0: QCA controller version 0x02140201`
* `Bluetooth: hci0: QCA Downloading qca/crbtfw21.tlv`
* `Bluetooth: hci0: QCA Downloading qca/crnv21.bin`
* `Bluetooth: hci0: QCA setup on UART is completed`
  …and `hciconfig -a` reports controller info. (These are the upstream‑known good lines for WCN3990 UART bring‑up.) ([lkml.indiana.edu][6])

---

## Quick validations (optional but strong signal)

* **Confirm TLV `download_mode`:** if you want to *prove* your TLV asks for VSE skipping:

  ```bash
  adb -t $transport shell 'hexdump -Cv -n 80 /data/firmware/qca/crbtfw21.tlv | sed -n "1,5p"'
  ```

  In the TLV PATCH header, `download_mode` is parsed by upstream and controls the skip‑VSE behavior just described. ([codebrowser.dev][4])

* **Dynamic debug during attach** (very noisy, but useful once):
  `echo 'module btqca +p' > /sys/kernel/debug/dynamic_debug/control` and capture.

---

## If it still stalls

1. **Seg‑0 still times out:** expand small‑segment window to **first 16 segs at 128B**; keep seg0–3=45 ms, seg4–15=30 ms, seg≥16=10 ms.
2. **ROM read still flaky:** bump the pre‑ROM settle to **100 ms** and set an **extra 5 ms** between *every* vendor ROM‑read retry.
3. **Patch→NVM gap:** extend the 10 ms settle to **40–50 ms** before NVM write (there’s a precedent upstream for a post‑patch settle). ([codebrowser.dev][4])

Document each pass in `BT_RESEARCH_STATUS.md` with:

* `uname -a` (should show `4.9.103 #46 …`)
* params (seg sizes, pacing, retries)
* last 10–20 BT lines.

---

## Why this should fix your two blockers

* **ROM read timeouts**: the early **IBS wake + real settle** prevents the first `0xFC00/0x19` from getting lost, and the 5 ms *before each retry* allows CTS/RTS to settle at 115200 so the retry isn’t back‑to‑back on a sleepy line. IBS handling/WAKE\_\*/ACK paths are in the driver and meant for exactly this. ([Android Gooblesource][3])
* **Seg‑0 failures**: with **skip‑VSE** and **CC injection** backported, the driver no longer waits for a per‑segment VSE that **will never arrive** in that mode, so seg‑0 stops “timing out.” Your pacing window then handles real UART/IBS jitter instead of fighting a logic bug. ([codebrowser.dev][1])
* **Stray vendor event**: the **drop‑vendor‑event** backport keeps the orphan baud‑change VSE from confusing the TLV state machine—this exact issue was documented for WCN3990 and fixed upstream. ([lkml.iu.edu][2])

---

## References (key upstream facts I matched your code to)

* **Skip‑VSE semantics + CC injection** for WCN399x TLV download in `btqca.c` (don’t wait for VSE on early segments; inject CC after last). ([codebrowser.dev][4])
* **Vendor command opcodes** + `MAX_SIZE_PER_TLV_SEGMENT` definitions, and the **10 s** `HCI_INIT_TIMEOUT` used for init‑phase commands. ([codebrowser.dev][7])
* **WCN3990 stray vendor event** fix (“Drop baudrate change vendor event”), with rationale and exact log lines you’re chasing (`controller version 0x02140201`, TLV download, etc.). ([lkml.iu.edu][2])
* Known‑good **bring‑up log** lines for WCN399x over UART (what you want to see verbatim). ([lkml.indiana.edu][6])

---

### What I’d have you run next (no guesswork):

1. Apply the diffs above exactly (#46).
2. Rebuild/flash both slots.
3. Run your bounded attach (15s/35s/60s).
4. If seg‑0 still flakes, expand the 128‑byte window to 16 segments, keep pacing; re‑run.
5. If needed, extend patch→NVM gap to 50 ms; re‑run.

Send me the **last 20 BT lines** plus the **params** you used, and I’ll hand you the next micro‑tweak without the fishing expedition. If the diffs land cleanly, I expect you’ll get:
`Bluetooth: hci0: QCA setup on UART is completed` and `hciconfig -a` showing the controller up. ([lkml.indiana.edu][6])

If you want, I can also spit out a ready‑to‑`git apply` patch bundle against your `agnos-kernel-sdm845` tree in the next pass.

[1]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html?utm_source=chatgpt.com "btqca.c source code [linux/drivers/bluetooth ..."
[2]: https://lkml.iu.edu/1903.1/04409.html "Linux-Kernel Archive: [PATCH v2 2/2] Bluetooth: hci_qca: wcn3990: Drop baudrate change vendor event"
[3]: https://android.googlesource.com/kernel/common/%2B/refs/tags/android12-5.4.173_r00/drivers/bluetooth/hci_qca.c?utm_source=chatgpt.com "drivers/bluetooth/hci_qca.c - kernel/common - Git at Google"
[4]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.c.html "btqca.c source code [linux/drivers/bluetooth/btqca.c] - Codebrowser "
[5]: https://codebrowser.dev/linux/linux/include/net/bluetooth/hci.h.html?utm_source=chatgpt.com "hci.h source code [linux/include/net/bluetooth/hci.h]"
[6]: https://lkml.indiana.edu/hypermail/linux/kernel/2307.0/06068.html?utm_source=chatgpt.com "Re: [PATCH RESEND 2/2] Bluetooth: fix use-bdaddr-property quirk"
[7]: https://codebrowser.dev/linux/linux/drivers/bluetooth/btqca.h.html?utm_source=chatgpt.com "btqca.h source code [linux/drivers/bluetooth ..."
