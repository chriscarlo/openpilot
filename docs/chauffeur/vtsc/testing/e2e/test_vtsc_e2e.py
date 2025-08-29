#!/usr/bin/env python3
"""
VTSC End-to-End (E2E) Test

Runs the public-facing VTSC test runner script as a subprocess and
asserts successful completion. Mirrors the E2E primer in
docs/chauffeur/testing/end_to_end_tests.md by interacting via a CLI entry.
"""

import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def run_cli(args, timeout=60):
  env = os.environ.copy()
  # Ensure test runner can import project
  env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH','')}"
  return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout, env=env)


def test_vtsc_full_suite_e2e():
  runner = ROOT / "docs/chauffeur/vtsc/testing/run_all_tests.py"
  proc = run_cli([str(runner)])
  # Expect success across all included suites
  assert proc.returncode == 0, f"run_all_tests.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
  # Basic sanity in output
  assert "FINAL SUMMARY" in proc.stdout


def main():
  try:
    test_vtsc_full_suite_e2e()
    print("✓ VTSC E2E test passed")
    return True
  except AssertionError as e:
    print(f"✗ VTSC E2E test failed: {e}")
    return False


if __name__ == "__main__":
  ok = main()
  sys.exit(0 if ok else 1)
