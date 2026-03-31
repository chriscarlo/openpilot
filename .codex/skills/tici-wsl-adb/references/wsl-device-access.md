# WSL Tici Device Access

## Relevant repo docs and helpers

- Durable workflow doc: `docs/chauffeur/adb/README.md`
- Repo device note: `AGENTS.md` under `## Device access`
- Install or repair the SSH profile: `./installCommaAdbProfile.sh`
- Connect through the forward: `./sshCommaAdb.sh`

## Working sequence

1. Inspect the Windows-side USB state from WSL:
- `"/mnt/c/Program Files/usbipd-win/usbipd.exe" list`
- Look for `04d8:1234` and `ADB Interface`.

2. Attach the device into WSL if needed:
- Check distro name: `wsl.exe -l -v`
- Attach: `"/mnt/c/Program Files/usbipd-win/usbipd.exe" attach --wsl <DISTRO> --busid <BUSID> --auto-attach`
- In this repo's current environment, `<DISTRO>` is `Ubuntu-24.04`.
- After attach, `usbipd list` should show the device as `Attached`.

3. Verify Linux-side ADB:
- `adb kill-server && adb devices -l`
- If that fails on permissions, retry with `sudo adb kill-server && sudo adb devices -l`

4. Start device SSH and create the local forward:
- `adb shell 'systemctl enable --now ssh || true'`
- `adb shell 'systemctl is-active ssh'`
- `adb forward tcp:2222 tcp:22`
- `adb forward --list`

5. Repair or use the repo SSH helpers:
- `./installCommaAdbProfile.sh -f`
- `ssh commaAdb`
- or `./sshCommaAdb.sh 'hostname; exit'`

## Failure signatures and fixes

- `adb devices` is empty in WSL:
  - The USB interface is not attached into WSL yet, or the wrong device was selected in `usbipd list`.

- `ssh: connect to host 127.0.0.1 port 2222: Connection refused`:
  - The ADB forward is missing. Re-run `adb forward tcp:2222 tcp:22`.

- `REMOTE HOST IDENTIFICATION HAS CHANGED` for `[127.0.0.1]:2222`:
  - Remove the stale cached key:
  - `ssh-keygen -f ~/.ssh/known_hosts -R '[127.0.0.1]:2222'`

- `./installCommaAdbProfile.sh` reports an unmanaged `Host commaAdb` block:
  - Re-run with `-f` so the managed block replaces it.

## Notes

- Prefer WSL `adb` once `usbipd` attach is working. Do not depend on Windows `adb.exe` being in `PATH`.
- If the process changes, update `docs/chauffeur/adb/README.md`; keep `AGENTS.md` short.
