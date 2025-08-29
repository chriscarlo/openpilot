# Acceptance (Functional/UAT) Tests Primer

Acceptance tests validate business behavior against requirements. They read like executable specifications focused on outcomes, not implementation.

## When To Use
- Prove that a requirement is satisfied end-to-end
- Communicate expected behavior in business language
- Prevent regressions on critical user-visible rules

## How To Write (pytest)
- Start with a requirement and a short scenario (Given/When/Then in prose)
- Interact with the system at a high level; assert observable outcomes
- Name tests/scenarios after the requirement or user story ID
- Keep them stable and few; put logic/variations into lower-level tests

## Minimal Example

Business rule: orders >= 100 get 10% discount, else 0%.

```python
# file: mypkg/pricing.py

def discount(amount: float) -> float:
  return 0.10 if amount >= 100.0 else 0.0
```

```python
# file: tests/acceptance/test_discount_acceptance.py
import pytest
from mypkg.pricing import discount

def test_REQ_123_discount_threshold_behavior():
  """
  Given an order total
  When total is at or above 100
  Then a 10% discount applies
  And otherwise no discount applies
  """
  cases = [
    (99.99, 0.0),
    (100.00, 0.10),
    (250.00, 0.10),
  ]
  for amount, expected in cases:
    assert discount(amount) == expected

@pytest.mark.parametrize("amount", [100.0, 100.01, 500.0])
def test_REQ_123_applies_at_or_above_threshold(amount):
  assert discount(amount) == 0.10
```

## Running
- Acceptance folder: `pytest tests/acceptance -q`
- Optionally add a custom marker (register in `pytest.ini`): `@pytest.mark.acceptance`

## Repo Notes
- Keep acceptance tests readable and outcome-driven; avoid implementation details
- If they’re slow or device-coupled, mark as `slow`/`tici` accordingly
- Link scenarios back to requirement IDs for traceability

