Changelog — Comma 3/3X Bluetooth Bring‑Up (brief)

2025-09-12
- Cycle 6 kernel (#71): explicit IBS disable during ROM/TLV/NVM; flashed.
- Cycle 7 kernel (#72): detach hardening in hci_qca close path (cancel timers/works with *_sync, flush/destroy workqueue, complete any drop‑baud waiters, re‑enable IBS flags). Built and flashed to both slots; no runtime watchers or log pulls used.

2025-09-11
- Rewrote BT_RESEARCH_STATUS.md into a live, minimal source of truth; added clear Plan of Record, access/ADB details, and maintenance policy.
- Documented current kernel (#56) driver changes (HCI_EV_VENDOR fix; skip‑VSE; baud‑switch drop window; request_firmware_direct). 
- Captured host ADB reliability guidance (USB vs TCP/IP) to avoid attach flaps.

2025-09-10
- Initial planning, multiple attach attempts logged; migrated firmware placement guidance; early notes on TLV pacing and retries.
