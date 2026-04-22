---
paths:
  - src/specagent/nodes/**/*.py
  - src/specagent/graph/**/*.py
---

# LangGraph Node Rules

## Node Signature

Every node MUST follow this exact signature:

```python
def node_name(state: GraphState) -> GraphState:
    """Docstring explaining node purpose."""
    # implementation
    return state
```

## Structured Output with Pydantic

```python
class GradeResult(BaseModel):
    relevant: Literal["yes", "no"]
    confidence: float = Field(ge=0.0, le=1.0)

result = llm.with_structured_output(GradeResult).invoke(prompt)
```

## Error Handling

Nodes must never raise — store errors in state:

```python
try:
    result = llm.invoke(prompt)
except Exception as e:
    state["error"] = str(e)
    return state
```

## State Objects

Use dataclasses from `graph/state.py`: `RetrievedChunk`, `GradedChunk`, `Citation`.

Each node is wrapped by `create_timed_node()` in `workflow.py` — elapsed ms accumulates in `state["node_timings"]`.
