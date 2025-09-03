# Environment Report

- Timestamp (UTC): 2025-09-03T19:25:18Z
- Repo: /data/openpilot
- Branch: chubbs-merge
- Commit: 14e7cb31d
- Working changes:  1 file(s)

## System
- OS release: Linux comma-e521630c 4.9.103 #32 SMP PREEMPT Mon Aug 4 23:13:16 UTC 2025 aarch64 aarch64 aarch64 GNU/Linux
- /etc/os-release:

```
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
```

## Tooling
- Python path: /usr/local/venv/bin/python
- Python version: Python 3.12.3
- SCons: SCons by Steven Knight et al.:

## Device/Runtime
- TICI check: /TICI present (TICI device)
- Harness sandbox: danger-full-access
- Harness approval: never
- Harness network: enabled

## Notes
- Follow repo guidelines in AGENTS.md for build/test steps.
- Ensure correct environment (TICI vs dev) before running SCons.
