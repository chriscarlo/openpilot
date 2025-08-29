# Smoke (Sanity) Tests Primer

Smoke tests are fast, shallow checks that the system “basically works.” Run them after build/deploy to catch obviously broken releases quickly.

## When To Use
- Post-build and pre/post-deploy gates in CI/CD
- Quick validation that core commands, imports, and health endpoints work
- Early failure signal before running heavier suites

## How To Write (pytest)
- Keep total runtime short (seconds to a minute)
- One simple assertion per test; avoid complex setup
- Prefer public surfaces: `--version`, `--help`, import sanity, health checks

## Minimal Examples

```python
# file: tests/smoke/test_imports.py

def test_can_import_top_level_package():
  __import__("mypkg")
```

```python
# file: tests/smoke/test_cli_version.py
import sys
import subprocess

def test_cli_prints_version_quickly():
  proc = subprocess.run([sys.executable, "-m", "mypkg", "--version"],
                        capture_output=True, text=True, timeout=2)
  assert proc.returncode == 0
  assert proc.stdout.strip()  # non-empty version string
```

Optional service health check (if a local test instance is started in CI):

```python
# file: tests/smoke/test_health.py
import os
import socket

def test_health_port_open():
  host, port = os.environ.get("APP_HOST", "127.0.0.1"), int(os.environ.get("APP_PORT", 8080))
  with socket.create_connection((host, port), timeout=1) as _:
    assert True
```

## Running
- Only smoke: `pytest tests/smoke -q`
- Gate in CI before running the full suite

## Repo Notes
- Keep smoke tests hermetic and blazing fast
- Fail fast on obvious misconfigurations (missing env, invalid wiring)
- Broader correctness belongs in unit/integration/E2E suites

