---
paths:
  - tests/**/*.py
  - src/**/*.py
---

# Testing Rules

## Test-First Development
When implementing new functionality:
1. Write tests FIRST in `tests/unit/test_<module>.py`
2. Use fixtures from `tests/conftest.py`
3. Implement code to pass tests
4. Run `pytest tests/unit/test_<module>.py -v` to verify

## Test Markers
- `@pytest.mark.unit` - Fast, no external dependencies
- `@pytest.mark.integration` - Multi-component, may use mocks
- `@pytest.mark.e2e` - Full pipeline tests
- `@pytest.mark.skip(reason="...")` - For unimplemented features

## Mocking External Services
Always mock LLM/embedding API calls (Groq, custom endpoint):
```python
@pytest.fixture
def mock_llm(mock_llm_response):
    # Use fixtures from conftest.py
    pass
```

## Coverage Target
Maintain >70% coverage. Check with: `pytest --cov=specagent`
