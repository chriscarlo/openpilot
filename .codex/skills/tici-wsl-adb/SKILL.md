---
name: tici-wsl-adb
description: Attach a comma tici/comma3x USB ADB interface into WSL2 with `usbipd-win`, verify Linux-side `adb` sees the device, create or repair the `commaAdb` SSH path on `127.0.0.1:2222`, and recover common failures such as empty `adb devices`, missing `usbipd` attach, refused `ssh commaAdb`, or stale `[127.0.0.1]:2222` host keys. Use when the user wants to reach a tici from WSL, mentions `usbipd`, `adb forward`, Windows USB attachment, or repo helper scripts like `sshCommaAdb.sh`.
---

# Tici WSL ADB

## Overview

Use this skill when a tici is plugged into Windows USB but the work should happen from WSL.
Prefer WSL-native `adb` after `usbipd` attach; use the repo helpers to repair `commaAdb` rather than inventing a separate SSH flow.

## Workflow

1. Check the current state before changing anything:
- `"/mnt/c/Program Files/usbipd-win/usbipd.exe" list`
- `adb devices -l`
- `ssh -o BatchMode=yes -o ConnectTimeout=5 commaAdb 'echo ok'`

2. If WSL `adb` does not see the device, attach the Windows USB device into WSL with `usbipd`.
- Look for `04d8:1234` or a device description containing `ADB Interface`.
- Use the actual distro name from `wsl.exe -l -v`; this repo currently runs in `Ubuntu-24.04`.

3. Once the device is attached, use Linux-side `adb` for the rest.
- Verify `adb devices -l`.
- Start or confirm device SSH, then create `adb forward tcp:2222 tcp:22`.
- Reuse `./installCommaAdbProfile.sh` and `./sshCommaAdb.sh` instead of hand-building a new SSH command unless the user wants the raw command.

4. If `ssh commaAdb` fails with a host-key mismatch on `[127.0.0.1]:2222`, remove the stale local key and reconnect.

## Guardrails

- Do not assume Windows `adb.exe` is on `PATH`; verify it if you need it, but prefer WSL `adb` once `usbipd` attach works.
- Do not duplicate the durable workflow in `AGENTS.md`; keep detailed human-facing steps in `docs/chauffeur/adb/README.md`.
- If `adb` in WSL reports a permissions problem, retry the same commands with `sudo adb ...`.
- If multiple USB devices are present, inspect `usbipd list` instead of guessing the BUSID.

## References

- Read `references/wsl-device-access.md` for exact commands, repo doc pointers, failure signatures, and the verified recovery path.
