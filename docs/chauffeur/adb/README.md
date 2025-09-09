# ADB → SSH/SFTP (WinSCP) Quick Reference

This guide shows how to browse and SSH into a comma 3/3X from a Windows 11 laptop using ADB port‑forwarding. It lists the exact PowerShell and WSL commands used here.

## Prereqs
- Windows PowerShell (run as Administrator)
- usbipd‑win 5.x: `winget install dorssel.usbipd-win`
- WSL packages: `sudo apt-get update && sudo apt-get install -y android-tools-adb usbutils libusb-1.0-0`

## 1) Attach the device USB to WSL (PowerShell, Admin)
1) Boot device normally (ADB available).
2) List USB and your WSL distro:
   - `usbipd list`
   - `wsl -l -v`  (note distro name, e.g. `Ubuntu`)
3) Attach the ADB device (BUSID from `usbipd list`, e.g. `2-1`):
   - `usbipd attach --wsl Ubuntu-24.04 --busid 2-1`
   - Recheck: `usbipd list` → device shows `Shared`.

## 2) Enable SSH on device + ADB port‑forward (WSL)
- Start ADB and verify device:
  - `sudo adb kill-server && sudo adb start-server`
  - `sudo adb devices`
- Enable/Start SSH on device (enable may warn about read‑only; active is enough):
  - `sudo adb shell 'systemctl enable --now ssh || true'`
  - Check: `sudo adb shell 'systemctl is-active ssh'`  → `active`
- Forward local port 2222 → device 22 and verify:
  - `sudo adb forward tcp:2222 tcp:22`
  - `sudo adb forward --list`

## 3) Connect (WinSCP or SSH)
- WinSCP (GUI): Protocol SFTP, Host `127.0.0.1`, Port `2222`, Username `comma`, Password `comma`.
- SSH (CLI): `ssh -p 2222 comma@127.0.0.1`
- Change password (optional): `sudo adb shell 'sudo passwd comma'`

Quick connect from this repo (one command)
- `./sshCommaAdb.sh` → sets up ADB forward and SSHes to `comma@127.0.0.1 -p 2222`.
  - Reuses the same key/cert as your `commaHome`/`commaCar` profiles automatically (falls back to `~/.ssh/id_ed25519`).
  - Extra ssh args are passed through, e.g.: `./sshCommaAdb.sh -v` or `./sshCommaAdb.sh 'hostname; exit'`.
  - Env knobs:
    - `COMMA_SSH_PROFILE=commaCar` (or `commaHome`) to force which profile to mirror.
    - `SKIP_ADB_FORWARD=1` to skip running `adb forward` inside the script.

Install a `ssh commaAdb` profile automatically
- From this repo, run: `./installCommaAdbProfile.sh`
  - Auto-detects identity from `commaHome`/`commaCar` and writes a managed Host block to `~/.ssh/config`.
  - Force a specific source profile: `./installCommaAdbProfile.sh -p commaCar`
  - Replace an existing stanza: `./installCommaAdbProfile.sh -f`
  - Different config path: `./installCommaAdbProfile.sh -t /some/ssh_config`
- Then use it like your other profiles: `ssh commaAdb`

## 3a) Reuse the exact commaHome/commaCar SSH certs
Goal: connect over ADB‑forwarded SSH using the same private key and OpenSSH certificate your `commaHome` or `commaCar` profile already uses.

What these files are called and where they live
- SSH config (Linux/macOS/WSL): `~/.ssh/config`
- SSH config (Windows OpenSSH): `C:\Users\<you>\.ssh\config`
- Private keys: usually `~/.ssh/id_ed25519` (or `id_rsa`)
- Matching OpenSSH certificate (if used): `~/.ssh/id_ed25519-cert.pub` (or `id_rsa-cert.pub`)

Find the exact files your commaHome/commaCar profiles use
- Linux/macOS/WSL:
  - Show the relevant config block(s):
    - `rg -n "^Host (commaHome|commaCar)($|\s)" ~/.ssh/config -n -C2 || sed -n '/^Host \(commaHome\|commaCar\)\($\| \)/,/^Host /p' ~/.ssh/config`
  - Print the fully‑resolved IdentityFile/CertificateFile OpenSSH will use:
    - `ssh -G commaHome | rg -n '^(identityfile|certificatefile) '`
    - `ssh -G commaCar  | rg -n '^(identityfile|certificatefile) '`
  - Verify the files exist:
    - `ls -l $(ssh -G commaHome | awk '/^identityfile /{print $2}')`
    - If a cert is listed: `ls -l $(ssh -G commaHome | awk '/^certificatefile /{print $2}')`

- Windows 11 PowerShell (OpenSSH):
  - Inspect config: `type $env:USERPROFILE\.ssh\config`
  - Show resolved files for `commaHome`/`commaCar`:
    - `ssh -G commaHome | findstr /R "^identityfile ^certificatefile"`
    - `ssh -G commaCar  | findstr /R "^identityfile ^certificatefile"`
  - Verify they exist:
    - `Get-Item (ssh -G commaHome | Select-String -Pattern '^identityfile ' | ForEach-Object { ($_ -split ' +')[1] })`
    - If present: `Get-Item (ssh -G commaHome | Select-String -Pattern '^certificatefile ' | ForEach-Object { ($_ -split ' +')[1] })`

Connect over ADB using those same files (one‑off)
- Linux/macOS/WSL (replace paths with the outputs you found above):
  - `ssh -p 2222 -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -o CertificateFile=~/.ssh/id_ed25519-cert.pub comma@127.0.0.1`
  - If you don’t use a cert, omit the `-o CertificateFile=...` option.
  - Note: OpenSSH auto‑loads `id_ed25519-cert.pub` when it sits next to `id_ed25519`, so you can usually omit `-o CertificateFile=...` if the names match.

- Windows 11 (OpenSSH):
  - `ssh -p 2222 -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\id_ed25519" comma@127.0.0.1`
  - If you use a cert and OpenSSH doesn’t auto‑load it, add: `-o CertificateFile="$env:USERPROFILE\.ssh\id_ed25519-cert.pub"`

Add a dedicated host alias that mirrors commaHome/commaCar
- Linux/macOS/WSL: edit `~/.ssh/config` and add (use the exact files you discovered):
  - `Host comma-adb`
  - `  HostName 127.0.0.1`
  - `  Port 2222`
  - `  User comma`
  - `  IdentitiesOnly yes`
  - `  IdentityFile ~/.ssh/id_ed25519`
  - `  CertificateFile ~/.ssh/id_ed25519-cert.pub`  (only if you use one)
- Windows (OpenSSH): edit `C:\Users\<you>\.ssh\config` with the same stanza (you can still write `~/.ssh/...` in OpenSSH on Windows).
- Then simply run: `ssh comma-adb`

WinSCP using the same key/cert
- If you use an agent (Pageant or OpenSSH agent) with your existing key, enable agent auth:
  - Session → Advanced → SSH → Authentication → check “Allow agent”.
- Otherwise, set “Private key file” to your existing private key:
  - Windows path: `C:\Users\<you>\.ssh\id_ed25519` (OpenSSH format). If WinSCP asks for `.ppk`, use PuTTYgen to convert: File → Load → Save private key.
- If you rely on an OpenSSH certificate (`-cert.pub`), keep it in the same folder; with an agent it is presented automatically. If not using an agent, connect once via `ssh` to verify cert pairing works.

Important notes
- ADB forwarding (`127.0.0.1:2222 → device:22`) does not change device SSH auth; it continues to honor the same authorized keys/certificates fetched from your GitHub username in device settings.
- First connection will prompt a host key for `127.0.0.1:2222` (it’s the device’s host key). Accept to cache it.

## 4) Common Paths
- `/data/openpilot` (source)
- `/data` (writable data)
- `/persist` (calibration/config)
- `/system` (read‑mostly)

## 5) Cleanup / Troubleshooting
- Remove forward: `sudo adb forward --remove tcp:2222` (or `--remove-all`)
- Stop SSH: `sudo adb shell 'systemctl stop ssh'`
- ADB permissions: use `sudo adb ...` in WSL.
- Not visible in WSL: `usbipd attach --wsl --busid <BUSID> --distribution "Ubuntu"`, ensure `usbipd list` shows `Shared`.
- Reboot: `sudo adb reboot`  |  EDL (flash): `sudo adb reboot edl`
