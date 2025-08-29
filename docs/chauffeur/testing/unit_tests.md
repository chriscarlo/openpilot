# Unit Tests Primer

Unit tests verify a single function or class in isolation. They are fast, deterministic, and run on every change.

## When To Use
- Validate pure logic, small helpers, and edge cases
- Guard against regressions in core algorithms
- Design aid: write-first (TDD) or alongside implementation

## How To Write (pytest)
- Keep dependencies mocked or stubbed (no network/DB/files unless using `tmp_path`)
- Name files `test_*.py`; place next to the code or in a module `tests/` dir
- Prefer parametrization over loops; keep one assertion theme per test
- Use fixtures (`tmp_path`, `monkeypatch`) for environment isolation

## Minimal Example

Imagine a tiny utility we want to test thoroughly:

```python
# file: mypkg/util.py

def clamp(v: float, lo: float, hi: float) -> float:
  if lo > hi:
    raise ValueError("lo>hi")
  return min(max(v, lo), hi)
```

Unit tests focusing on behavior and edge cases:

```python
# file: tests/test_util.py
import pytest
from mypkg.util import clamp

@pytest.mark.parametrize("v,lo,hi,expected", [
  (5, 0, 10, 5),
  (-1, 0, 10, 0),
  (11, 0, 10, 10),
])
def test_clamp_in_range(v, lo, hi, expected):
  assert clamp(v, lo, hi) == expected

def test_clamp_raises_on_bad_bounds():
  with pytest.raises(ValueError):
    clamp(1, 10, 0)
```

Mocking or monkeypatching external calls:

```python
# file: mypkg/clock.py
import time

def now_seconds() -> int:
  return int(time.time())
```

```python
# file: tests/test_clock.py
def test_now_seconds_monkeypatched(monkeypatch):
  monkeypatch.setattr("time.time", lambda: 1234.56)
  from mypkg.clock import now_seconds
  assert now_seconds() == 1234
```

## Running
- Single file: `pytest tests/test_util.py -q`
- All fast tests: `pytest -m 'not slow'`

## Repo Notes
- Use `pytest` with markers: long tests `@pytest.mark.slow`, device-only `@pytest.mark.tici`
- Keep Python 2-space indentation; add type hints where reasonable
- Stick to focused, isolated checks; avoid hidden I/O/shared state

