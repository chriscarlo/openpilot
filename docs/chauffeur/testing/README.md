# Testing Primers

Quick, practical how‑tos for the most common test types. Each primer includes when to use it, how to write it with `pytest`, and a minimal example you can adapt.

- Unit tests: `docs/chauffeur/testing/unit_tests.md`
- Integration tests: `docs/chauffeur/testing/integration_tests.md`
- End‑to‑End (System) tests: `docs/chauffeur/testing/end_to_end_tests.md`
- Acceptance (Functional/UAT) tests: `docs/chauffeur/testing/acceptance_tests.md`
- Smoke (Sanity) tests: `docs/chauffeur/testing/smoke_tests.md`

Repo test basics:
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -e ".[testing,dev]"`
- Fast tests: `pytest -m 'not slow'`; full: `pytest`
- Marks: long tests `@pytest.mark.slow`; device-only `@pytest.mark.tici`
- Naming: files `test_*.py`; place next to code or under a module `tests/` folder

