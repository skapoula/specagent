# Implement Node

Implement a LangGraph node following the project patterns.

## Process

1. **Review the placeholder** in `src/specagent/nodes/{node_name}.py`
2. **Read the PRD** for node requirements: `@docs/prd-3gpp-agentic-rag.md`
3. **Check existing patterns** in other implemented nodes
4. **Write tests first** in `tests/unit/test_{node_name}.py`:
   - Use fixtures from `tests/conftest.py`
   - Mock LLM responses with `mock_llm_response` fixture
   - Test both success and failure cases
5. **Implement the node** to pass tests
6. **Run verification**:
   ```bash
   pytest tests/unit/test_{node_name}.py -v
   mypy src/specagent/nodes/{node_name}.py
   ruff check src/specagent/nodes/{node_name}.py
   ```

## Node Requirements

- Follow signature: `def {node_name}(state: GraphState) -> GraphState`
- Use Pydantic for structured LLM output
- Include docstrings and type hints
- Handle errors gracefully (set `state["error"]` on failure)
- Log important decisions for debugging

## After Implementation

Update `src/specagent/nodes/__init__.py` exports if needed.
