# Integration Tests Primer

Integration tests verify that a few real components work together correctly (e.g., parser + calculator, service + DB). They catch interface, schema, and contract mismatches.

## When To Use
- Validate a “slice” of functionality across module boundaries
- Exercise real implementations for the tested slice (no mocks between them)
- Check serialization formats, DB migrations, file I/O, or message wiring

## How To Write (pytest)
- Choose a narrow, realistic path through 2–3 components
- Use real resources for that slice: files via `tmp_path`, in-memory or temp DBs, local queues
- Isolate side effects: clean up with fixtures; avoid hitting external networks
- Keep tests deterministic; control randomness/time via fixtures

## Minimal Example

Two components: one loads numbers from a file, the other computes stats.

```python
# file: mypkg/io_mod.py

def load_numbers(path: str) -> list[int]:
  nums: list[int] = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      nums.append(int(line))
  return nums
```

```python
# file: mypkg/stats_mod.py

def mean(nums: list[int]) -> float:
  if not nums:
    raise ValueError("empty")
  return sum(nums) / len(nums)
```

Integration test uses both components together and real file I/O:

```python
# file: tests/test_numbers_integration.py
from mypkg.io_mod import load_numbers
from mypkg.stats_mod import mean

def test_file_to_mean_happy_path(tmp_path):
  p = tmp_path / "nums.txt"
  p.write_text("1\n2\n3\n")
  nums = load_numbers(str(p))
  assert nums == [1, 2, 3]
  assert mean(nums) == 2.0

def test_file_with_blank_lines(tmp_path):
  p = tmp_path / "nums.txt"
  p.write_text("\n10\n\n20\n")
  assert mean(load_numbers(str(p))) == 15.0
```

## Running
- Single file: `pytest tests/test_numbers_integration.py -q`
- All integration (if marked): `pytest -m integration`

## Repo Notes
- Prefer temp resources (`tmp_path`, temp SQLite) over global/dev services
- If tests are slower, mark as `@pytest.mark.slow`; device-only as `@pytest.mark.tici`
- Keep scope small to reduce flakiness and speed up CI

