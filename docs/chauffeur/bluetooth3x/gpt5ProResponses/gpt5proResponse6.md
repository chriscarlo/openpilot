Chris — you’re running into two *separate* classes of problems:

1. **The pre‑bump to 3M without the upstream “drop the baud‑change event” plumbing** is what’s yielding the `Frame reassembly failed (-84)` and other junk immediately after the speed switch.

2. **Your “skip‑VSE” attempt violates HCI command‑credit semantics**, so the core gets starved/angry when segments don’t produce acks. Upstream solved this with a specific pattern (queue segments asynchronously *and* inject a synthetic Command Complete once), not by blasting `hci_send_cmd` in a loop.

Below is a minimal, battle‑tested backport plan to 4.9 that fixes both, plus answers to your specific questions.

---

## A. Correct “skip‑VSE” semantics for WCN399x on 4.9 (what to backport)

### 1) Parse TLV `download_mode`, queue early segments properly, and inject one final CC

Upstream `btqca` does three key things for WCN399x/Rome “skip‑VSE” devices:

* **Read `download_mode` from the TLV patch header** and store it in the per‑download config (both `dnld_mode` and `dnld_type`). ([Android Gooblesource][1])

* **Send each non‑final TLV segment with `__hci_cmd_send`** (asynchronous, no per‑segment wait) when `download_mode` is `QCA_SKIP_EVT_VSE[_CC]`. **Only the last segment is sent synchronously** (force `dnld_mode = NONE` for final/short segment). ([Android Gooblesource][1])

* **After all segments, inject a synthetic `HCI_EV_CMD_COMPLETE`** (opcode `QCA_HCI_CC_OPCODE` = `0xFC00`, payload status `0x00`, and `ncmd = 1`) so the HCI core’s command credit/accounting doesn’t time out. (The controller only emits a VSE for the *last* packet; this injected CC satisfies the core.) ([Android Gooblesource][1])

> **Minimal edits (4.9)**
> *btqca.h* — add the skip‑VSE enums and CC constants if your 4.9 tree doesn’t have them:
>
> ```c
> enum qca_tlv_dnld_mode { QCA_SKIP_EVT_NONE = 0, QCA_SKIP_EVT_VSE = 1, QCA_SKIP_EVT_VSE_CC = 2 };
> #define QCA_HCI_CC_OPCODE  0xFC00
> #define QCA_HCI_CC_SUCCESS 0x00
> ```
>
> *btqca.c* — in `qca_tlv_check_data()`, read `tlv_patch->download_mode` into `config->dnld_mode` and `config->dnld_type`;
> in `qca_tlv_send_segment()`, do:
>
> ```c
> if (mode == QCA_SKIP_EVT_VSE || mode == QCA_SKIP_EVT_VSE_CC)
>     return __hci_cmd_send(hdev, EDL_PATCH_CMD_OPCODE, seg_size+2, cmd);
> ```
>
> set `config->dnld_mode = QCA_SKIP_EVT_NONE` for the final/short segment;
> after the send loop, if `config->dnld_type` is a skip‑VSE type, call `qca_inject_cmd_complete_event(hdev)` which builds a tiny HCI event SKB and pushes it with `hci_recv_frame()`. This is exactly how upstream avoids command‑timeout spew. ([Android Gooblesource][1])

### 2) **Backport `__hci_cmd_send`** if 4.9 lacks it

Newer BlueZ kernels added `__hci_cmd_send(hdev, opcode, plen, param)` specifically so drivers can queue commands without a sync wait. It’s \~10–15 lines: allocate a command SKB, push header+payload, and call `hci_send_frame(hdev, skb)`. If your 4.9 doesn’t have it, copy it from mainline (or any LTS that has it) and export it; it only calls into existing core paths. ([Codebrowser][2])

> Why this matters: your current “call `hci_send_cmd` repeatedly without acks” goes through the synchronous/request machinery and credit gates; `__hci_cmd_send` is the upstream‑approved *driver‑internal queue* path used by `btqca` in skip‑VSE mode. ([Android Gooblesource][1])

### 3) **No “fake skb” injections per segment**

Upstream injects **one** synthetic `CMD_COMPLETE` **after** the whole TLV, not per segment. Don’t try to mint CCs segment‑by‑segment; that will break credits in the other direction. ([Android Gooblesource][1])

---

## B. Fixing the baud‑switch (“-84” framing) correctly

Your `-84` right after “Pre‑bump UART to 3000000…” is the classic symptom of not dropping (or mis‑classifying) the controller’s **baud‑change event** and/or switching host speed without gating flow control. Upstream WCN3990 handling does:

* **Before sending the vendor baud‑change command**, set a flag (`QCA_DROP_VENDOR_EVENT`) and **enable flow control** on the host UART. (Despite a confusing old comment, the point is to hold traffic steady while switching.) Then send the vendor command, switch the host UART to the new speed, **wait up to \~100 ms** for the controller’s vendor/CC event, **drop that event** from normal processing, and finally clear the flag and restore flow control. ([Linux-Kernel Archive][3])

Concretely (4.9 backport):

* In `struct qca_data`, add `struct completion drop_ev_comp;` and a `QCA_DROP_VENDOR_EVENT` flag bit; `init_completion()` it at open.
* Hook `H4_RECV_EVENT` to a **`qca_recv_event()` shim**: when the drop flag is set, if the header is a vendor event (WCN3990) or CC on newer cores, **`complete(&drop_ev_comp)` and `kfree_skb(skb)`** instead of passing it to `hci_recv_frame`. That’s the upstream fix which originally eliminated “TLV response size mismatch / -84” on WCN3990 after the speed change. ([Linux-Kernel Archive][3])
* In your baud‑bump path:

  ```c
  reinit_completion(&qca->drop_ev_comp);
  set_bit(QCA_DROP_VENDOR_EVENT, &qca->flags);
  hci_uart_set_flow_control(hu, true);
  qca_set_baudrate(...);          // send vendor cmd at old speed
  host_set_baudrate(hu, 3000000); // switch host
  if (!wait_for_completion_timeout(&qca->drop_ev_comp, msecs_to_jiffies(100)))
      return -ETIMEDOUT;
  clear_bit(QCA_DROP_VENDOR_EVENT, &qca->flags);
  hci_uart_set_flow_control(hu, false);
  ```

  This is lifted straight from modern `hci_qca`. It prevents the stray event from confusing the TLV path and eliminates the H4 reassembly error. ([Codebrowser][4])

> **Recommendation:** *Undo* your “pre‑bump to 3M before TLV/NVM” until you’ve backported the vendor‑event drop path above. Upstream normally downloads at the init speed and only bumps when appropriate. If you insist on pre‑bump, the **drop‑event + flow‑control + 100 ms wait** is mandatory, or you’ll keep seeing `-84`. ([Linux-Kernel Archive][3])

---

## C. About HCI credits and “do we need to inject CC?”

**Yes.** With skip‑VSE, the controller doesn’t emit per‑segment `CMD_COMPLETE`s — it only sends one VSE at the end. The Linux HCI core will otherwise log a command timeout / block the queue. Upstream injects **one** `HCI_EV_CMD_COMPLETE` with:

* **Opcode:** `0xFC00` (the Qualcomm EDL patch opcode the segments use)
* **ncmd:** `1` (so the core knows it can send more)
* **Payload:** one byte status `0x00`
  You push it into the stack with `hci_recv_frame(hdev, skb)` after the last segment completes. That’s the minimal compatible backport. ([Android Gooblesource][1])

---

## D. The sporadic “Failed to request file: err = (-11)” (`-EAGAIN`) during TLV

`request_firmware()` can return `-EAGAIN` when the **userspace fallback** path is engaged or races with availability; modern drivers either (a) treat it as “no firmware” and run with ROM, or (b) **use `request_firmware_direct()`** to bypass the userspace helper (and its timeouts) entirely. Upstream `hci_qca` even comments that `-EAGAIN` from the userspace loader means “not found; run with ROM.” In your case the blobs are present and your search path is widened, so this is likely the fallback path being touched unnecessarily. **Switch to `request_firmware_direct()`** in `btqca.c` to avoid `-EAGAIN` entirely when the file is there; add a small pre‑call settle only if you suspect your bind‑mount happens just-in-time. ([Kernel.org][5])

---

## E. Answers to your specific questions

> **What is the correct, minimal backport of WCN399x skip‑VSE semantics to Linux 4.9’s btqca/hci core?**

* In **`btqca`**: parse TLV `download_mode`; send non‑final segments with `__hci_cmd_send`; force sync on the last; inject one synthetic CC at the end. (See A.1.) ([Android Gooblesource][1])
* In **`hci_qca`**: add the **drop‑event** logic around the baud change (flag + completion + `qca_recv_event()`), with flow‑control gating and 100 ms wait. (See B.) ([Linux-Kernel Archive][3])
* In **HCI core**: backport **`__hci_cmd_send`** if missing. (See A.2.) ([Codebrowser][2])

> **How should early TLV segments be queued without violating HCI credits?**

Use **`__hci_cmd_send`** for segments in skip‑VSE mode; that path queues commands correctly in the core. **Do not** busy‑loop `hci_send_cmd` or hand‑roll your own CCs per segment. The single final injected CC is sufficient and matches upstream. ([Android Gooblesource][1])

> **Do we need to inject a Command Complete? Which opcode/payload?**

Yes — **one** `HCI_EV_CMD_COMPLETE` with **opcode `0xFC00`** and **status `0x00`**, **`ncmd = 1`**. Push via `hci_recv_frame`. ([Android Gooblesource][1])

> **How to prevent H4 “Frame reassembly failed (-84)” around the baud switch?**

Implement the **vendor‑event drop** window (flag + completion), **wrap the switch with flow‑control on/off**, and **wait \~100 ms** for the event before clearing the flag. This is the upstream‑correct fix; it was introduced specifically to prevent WCN3990 mis‑parsing of the vendor event as a TLV response after switching speed. ([Linux-Kernel Archive][3])

> **Is `-EAGAIN` from `request_firmware()` a real timing race?**

Often it’s the **userspace fallback** tripping. On 4.9, use **`request_firmware_direct()`** for your rampatch/NVM to bypass the helper and eliminate the spurious `-EAGAIN` when blobs exist and the path is already in place. ([Kernel.org][5])

> **Reference diff to implement skip‑VSE properly?**

The Android/common/linux implementation shows exactly this flow (download‑mode parse, async sends for early segments, synthetic CC injection). It’s a good template to follow when backporting. ([Android Gooblesource][1])

---

## F. Concrete patch sketch (targeted to your paths)

> **`drivers/bluetooth/btqca.h`** (ensure these are present)

```c
enum qca_tlv_dnld_mode { QCA_SKIP_EVT_NONE = 0, QCA_SKIP_EVT_VSE = 1, QCA_SKIP_EVT_VSE_CC = 2 };
#define QCA_HCI_CC_OPCODE  0xFC00
#define QCA_HCI_CC_SUCCESS 0x00
```

> **`drivers/bluetooth/btqca.c`** (essentials)

* In `qca_tlv_check_data()`: set `config->dnld_mode = tlv_patch->download_mode; config->dnld_type = config->dnld_mode;` ([Android Gooblesource][1])
* In `qca_tlv_send_segment()`:

```c
if (mode == QCA_SKIP_EVT_VSE || mode == QCA_SKIP_EVT_VSE_CC)
    return __hci_cmd_send(hdev, EDL_PATCH_CMD_OPCODE, seg_size+2, cmd);
```

* In the send loop:

```c
if (!remain || segsize < MAX_SIZE_PER_TLV_SEGMENT)
    config->dnld_mode = QCA_SKIP_EVT_NONE;
```

* After the loop:

```c
if (config->dnld_type == QCA_SKIP_EVT_VSE || config->dnld_type == QCA_SKIP_EVT_VSE_CC)
    ret = qca_inject_cmd_complete_event(hdev);
```

* Implement `qca_inject_cmd_complete_event()` to build an HCI Event packet with `ncmd=1`, `opcode=0xFC00`, status `0x00`, and call `hci_recv_frame()`. ([Android Gooblesource][1])
* Change firmware fetch to `request_firmware_direct()` to avoid `-EAGAIN`. ([Debian Manpages][6])

> **`drivers/bluetooth/hci_qca.c`** (baud switch & event drop)

* Add `struct completion drop_ev_comp;` and the `QCA_DROP_VENDOR_EVENT` flag into `struct qca_data`; init the completion in `qca_open()`.
* Replace `H4_RECV_EVENT` handler with `qca_recv_event()` that **completes+drops** the vendor/CC event while `QCA_DROP_VENDOR_EVENT` is set. ([Linux-Kernel Archive][3])
* In the baud‑switch path for **WCN3990**: gate flow control, set the drop flag, send the vendor baud‑change, switch host speed, wait up to \~100 ms for the dropped event, clear the flag, and restore flow control. (Matches mainline.) ([Codebrowser][4])

> **If missing in 4.9:** add `__hci_cmd_send()` to the HCI core and export it. It just allocates a command SKB and calls `hci_send_frame()`. ([Codebrowser][2])

---

## G. On your current knobs

* **Segment size & pacing:** with correct skip‑VSE + CC injection, you can keep **128‑byte segments** + **10 ms** pacing for the whole TLV for stability at 115200. Once stable, you can experiment with 243‑byte segments and/or reduce pacing. (Upstream doesn’t need pacing at high speed; your UART path is pickier.)

* **ROM read flakiness (0xFC00 timeouts):** your IBS wake/reset and small settles are fine. The bigger reliability win will come from **not** switching to 3M until the above baud‑change plumbing is correct.

* **Patch→NVM gap:** adding **40–50 ms** before NVM (up from 10 ms) is harmless and has helped on some noisy systems; consider it a follow‑up once TLV is reliable.

---

## H. Evaluation of your proposed next steps

* **Back out the current “hci\_send\_cmd + synthetic skb per segment” attempt and re‑implement skip‑VSE per upstream:** **Strong yes.** That’s the root of your HCI credit issues. ([Android Gooblesource][1])
* **Increase patch→NVM settle to 40–50 ms:** Reasonable as a second pass; do it after the two fundamental fixes above.
* **Bounded slices + doc updates:** Sensible; keep the attach bounded and log the last 10–20 lines + parameters to your status doc as you already planned.

---

## I. Quick checklist to get to “hci0 UP”

1. Implement the **baud‑switch drop‑event** handling + flow‑control window in `hci_qca`. **Do not** pre‑bump without it. ([Linux-Kernel Archive][3])
2. Backport `__hci_cmd_send` (if absent) and rework `btqca` to the **skip‑VSE** pattern (async early segments, sync last, one synthetic CC). ([Android Gooblesource][1])
3. Switch `request_firmware()` → **`request_firmware_direct()`** in `btqca.c`. ([Debian Manpages][6])
4. Keep TLV at 115200, **128‑byte segments**, 10 ms pacing; add **40–50 ms** before NVM if TLV succeeds but NVM is flaky.
5. Re‑run your bounded `btattach` slices and look for the success string:

   * `“Bluetooth: hci0: ROME setup … controller version 0x02140201 … ROME Downloading file: qca/crbtfw21.tlv … ROME setup on UART is completed”`
   * Then `hciconfig hci0 up` and `hciconfig -a` should succeed.

---

### References / upstream patterns cited

* **Skip‑VSE semantics, async send, & synthetic CC injection** in `btqca`: parsing `download_mode`, `__hci_cmd_send` for early segments, force sync on last, inject one `CMD_COMPLETE` (opcode `0xFC00`, status `0x00`). ([Android Gooblesource][1])
* **WCN3990 baud change drop‑event + flow control + completion wait (100 ms)** to prevent TLV mis‑parsing / `-84`. ([Linux-Kernel Archive][3])
* **`__hci_cmd_send`** addition in the Bluetooth core for drivers to queue commands asynchronously. ([Codebrowser][2])
* **Firmware loader behavior**: `-EAGAIN` from userspace fallback; use `request_firmware_direct()` to avoid it. Upstream `hci_qca` treats `-EAGAIN` from the helper as “run with ROM.” ([Kernel.org][5])

---

If you want, I can write the exact 4.9 patch hunks for your tree (`/projects/agnos/agnos-builder/agnos-kernel-sdm845/...`) following the structure above. But the shortest path to “`hci0 UP`” is: **(1)** drop‑event + flow‑control around baud switch, **(2)** real skip‑VSE backport using `__hci_cmd_send` + one injected CC, **(3)** `request_firmware_direct()`. After those three, the rest of your timing tweaks should become largely unnecessary.

[1]: https://android.googlesource.com/kernel/common/%2B/ecf61e4e1117/drivers/bluetooth/btqca.c "drivers/bluetooth/btqca.c - kernel/common - Git at Google"
[2]: https://codebrowser.dev/linux/linux/net/bluetooth/hci_core.c.html?utm_source=chatgpt.com "hci_core.c source code [linux/net/bluetooth ..."
[3]: https://lkml.iu.edu/1903.1/04409.html "Linux-Kernel Archive: [PATCH v2 2/2] Bluetooth: hci_qca: wcn3990: Drop baudrate change vendor event"
[4]: https://codebrowser.dev/linux/linux/drivers/bluetooth/hci_qca.c.html "hci_qca.c source code [linux/drivers/bluetooth/hci_qca.c] - Codebrowser "
[5]: https://www.kernel.org/doc/html/v4.17/driver-api/firmware/fallback-mechanisms.html?utm_source=chatgpt.com "Fallback mechanisms — The Linux Kernel documentation"
[6]: https://manpages.debian.org/testing/linux-manual-4.8/request_firmware_direct.9.en.html?utm_source=chatgpt.com "request_firmware_direct(9) — linux-manual-4.8"
