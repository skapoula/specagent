# Testing — specagent
# Extends /workspace/.claude/rules/master-testing.md (loaded automatically).
# Only project-specific additions and overrides are listed here.

## Commands

```bash
pytest                              # full suite (excludes slow)
pytest -m unit                      # unit tests only
pytest -m integration               # real LanceDB (tmp)
pytest -m e2e                       # full pipeline tests
pytest --cov=src/specagent          # with coverage report
```

## Coverage

Maintain **>70%** coverage across `src/specagent/`.

## Test-First Development

Write tests FIRST in `tests/unit/test_<module>.py` before implementing new functionality.
Implement code to pass tests, not the other way around.

## Mocking External Services

Always mock LLM and embedding API calls (Groq, custom endpoints).
Use fixtures from `tests/conftest.py` — do not define mocks inline.

```python
@pytest.fixture
def mock_llm(mock_llm_response):
    # Use fixtures from conftest.py
    pass
```

Use `pytest-httpx` for mocking HTTP calls to the embedding/LLM endpoints.
