# End-to-End (System) Tests Primer

End-to-End (E2E) tests exercise the system from the outside using public interfaces (CLI, API, UI), across realistic boundaries and configuration. They are the broadest and slowest tests.

## When To Use
- Validate that the whole stack is wired correctly
- Catch configuration/env issues that unit/integration tests miss
- Prove critical user journeys work (happy paths and a few failure paths)

## How To Write (pytest)
- Interact only via public interfaces (HTTP requests, CLI invocations)
- Use production-like settings and data; stub only third parties if necessary
- Add timeouts; make tests idempotent and environment-independent
- Prefer a few high-value scenarios over many brittle ones

## Minimal CLI Example

```python
# file: app/cli.py
import argparse

def main(argv=None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("add", nargs=2, type=int, help="add two integers")
  args = parser.parse_args(argv)
  a, b = args.add
  print(a + b)
  return 0

if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
```

```python
# file: tests/e2e/test_cli_add.py
import sys
import subprocess

def run_cli(args: list[str], timeout=3):
  return subprocess.run([sys.executable, "-m", "app.cli", *args],
                        capture_output=True, text=True, timeout=timeout)

def test_cli_add_e2e_happy_path():
  proc = run_cli(["2", "40"])  # public interface only
  assert proc.returncode == 0
  assert proc.stdout.strip() == "42"
  assert proc.stderr == ""
```

## Running
- E2E subset: `pytest tests/e2e -q`
- Mark slow or environment-heavy: `@pytest.mark.slow`

## Repo Notes
- Use stable, deterministic inputs; avoid sleeps—poll with timeouts instead
- Keep E2E count small; rely on unit/integration for depth and speed
- If targeting device-only flows, mark with `@pytest.mark.tici`

