Bluetooth Bring‑Up — Lessons Learned (Do‑Not / Must‑Do List)

Purpose
- Single place to capture everything we must not do and the proven guardrails to avoid boot loops, wasted time, or silent failures during Bluetooth bring‑up on Comma 3/3X.

Do NOT (hard stops)
- Do not overwrite both boot slots with an unverified boot image. If the new image boot‑loops you’ll need EDL. Always flash the inactive slot only and keep the other as a parachute.
- Do not modify DTS beyond the single, known UART enablement for `/dev/ttyHS0`. Prior edits to APCS/GLINK/SMD/child nodes caused early‑boot crashes (pre‑userspace) and boot loops.
- Do not press “Install” on the on‑device AGNOS update prompt during testing. It can rebase system components mid‑iteration.
- Do not run raw `adb` without a timeout in this workflow. USB/ADB can flap during attach and leave the host stuck. Always use `scripts/dev/adb_safe.sh -t …`.
- Do not stage test binaries/logs under `/tmp` and expect them to persist. Use `/data/media/0` for binaries and logs you plan to pull later.

Must‑Do (safety rails)
- A/B safety:
  - Write to the inactive partition only; swap with `abctl --set_active <slotnum>`; verify; keep the other slot pristine for instant rollback.
- Visibility & UI:
  - Suppress the “Update Required (~1 GB)” overlay by making `AGNOS_VERSION` default to `/VERSION` in `launch_env.sh`; kill any running updater process. Mask `systemd-sysupdate*` and `packagekit-offline-update.service` at runtime while testing.
- Attach hygiene:
  - Keep attach windows short and bounded (≤40–60 s while iterating). Long windows have triggered USB gadget drops.
  - Don’t use `adb root/kill-server/tcpip` during an attach window. Start attach with `adb shell 'su -c …'`, wait quietly, then scrape logs after.
  - Prefer on‑device logging for attach attempts: run btattach via `nohup` and write rc/stdout/stderr to files in `/data/media/0/` before scraping.
- UART specifics:
  - Use `/dev/ttyHS0` (HS UART). Do not try `/dev/ttyMSM0` (console) for Bluetooth.
  - If you see “cannot execute binary file” for btattach/hciattach, you pushed the wrong architecture. Verify with `file` (must be aarch64/ELF).

Observed Failure Patterns (and how to avoid them)
- Boot loops after flashing both slots with an untested boot image → Always keep a parachute slot; never flash both.
- Pre‑userspace crash after bundling multiple early‑boot DTS changes → Leave DTS alone except for the single UART enablement we already know works.
- Lost logs or missing rc files after attach → Avoid `/tmp`; log to `/data/media/0/` and write the exit code explicitly to a `.rc` file.
- Silent ADB hangs while scraping → Use the ADB wrapper with explicit `-t` and retry after `wait-for-device`.

Notes
- These guardrails reflect actual incidents in this workspace; adhere to them for all Bluetooth bring‑up work until explicitly revised.

