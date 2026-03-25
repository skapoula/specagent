# PRD Addendum: Evaluation & Testing Strategy

## Overview

Evaluating agentic RAG systems is fundamentally different from evaluating simple LLM applications. We must assess:

1. **Component quality** — Does each node (router, grader, rewriter, etc.) perform its function correctly?
2. **End-to-end quality** — Is the final answer accurate and grounded?
3. **Trajectory quality** — Did the agent take an optimal path through the graph?
4. **Online behavior** — How does the system perform in production with real queries?

This addendum specifies the evaluation framework, tools, metrics, and testing strategy for 3GPP SpecAgent.

---

## 1. Evaluation Tool Stack

### Recommended Stack

| Tool | Purpose | Cost | Rationale |
|------|---------|------|-----------|
| **Arize Phoenix** | Tracing, observability, LLM-as-judge | Free (self-hosted) | Open-source, OpenTelemetry-native, excellent LangGraph integration |
| **RAGAS** | RAG-specific metrics | Free | Industry standard for Faithfulness, Context Precision, Answer Relevancy |
| **agentevals** | Trajectory evaluation | Free | LangChain's official library for graph trajectory assessment |
| **pytest** | Unit/integration tests | Free | Standard Python testing |

### Why This Stack (Not Haystack)

**Haystack** is a full RAG framework—we'd be replacing LangGraph, not complementing it. Haystack's evaluation components are useful but tightly coupled to its pipeline abstraction.

**Arize Phoenix + RAGAS** gives us:
- Framework-agnostic evaluation (works with LangGraph natively)
- Separation of concerns (Phoenix for observability, RAGAS for metrics)
- Lower memory footprint (important for 4GB constraint)
- Better trajectory visualization for debugging agentic loops

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Evaluation Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        OFFLINE EVALUATION                               │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │
│  │  │   Component     │  │   End-to-End    │  │      Trajectory         │ │ │
│  │  │   Unit Tests    │  │   Evaluation    │  │      Evaluation         │ │ │
│  │  │                 │  │                 │  │                         │ │ │
│  │  │  pytest +       │  │  RAGAS:         │  │  agentevals:            │ │ │
│  │  │  custom evals   │  │  - Faithfulness │  │  - Path optimality      │ │ │
│  │  │                 │  │  - Relevancy    │  │  - Rewrite efficacy     │ │ │
│  │  │  Per-node:      │  │  - Correctness  │  │  - Node visit patterns  │ │ │
│  │  │  - Router       │  │                 │  │                         │ │ │
│  │  │  - Grader       │  │  TSpec-LLM      │  │  Graph trajectory       │ │ │
│  │  │  - Rewriter     │  │  Benchmark:     │  │  LLM-as-judge           │ │ │
│  │  │  - Generator    │  │  100 questions  │  │                         │ │ │
│  │  │  - Halluc.Check │  │                 │  │                         │ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        ONLINE EVALUATION                                │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                      Arize Phoenix                               │   │ │
│  │  │                                                                  │   │ │
│  │  │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │   │ │
│  │  │   │   Tracing    │  │  LLM-as-     │  │   Drift Detection  │    │   │ │
│  │  │   │              │  │  Judge       │  │                    │    │   │ │
│  │  │   │  - Latency   │  │              │  │  - Embedding drift │    │   │ │
│  │  │   │  - Token use │  │  - Relevance │  │  - Response length │    │   │ │
│  │  │   │  - Node path │  │  - Toxicity  │  │  - Rewrite rate    │    │   │ │
│  │  │   │  - Errors    │  │  - Quality   │  │                    │    │   │ │
│  │  │   └──────────────┘  └──────────────┘  └────────────────────┘    │   │ │
│  │  │                                                                  │   │ │
│  │  │   OpenTelemetry ──────────────────────────────────────▶ Phoenix │   │ │
│  │  │   Instrumentation                                       UI      │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Offline Evaluation: Component-Level Testing

### 2.1 Router Evaluation

**What we're testing**: Does the router correctly classify queries as in-scope (3GPP) vs. out-of-scope?

**Dataset**: 200 manually labeled queries
- 100 in-scope (3GPP technical questions)
- 100 out-of-scope (cooking, sports, generic coding, etc.)

**Metrics**:
| Metric | Target | Formula |
|--------|--------|---------|
| Accuracy | ≥95% | (TP + TN) / Total |
| False Positive Rate | <5% | FP / (FP + TN) — routing non-3GPP to retrieval |
| False Negative Rate | <2% | FN / (FN + TP) — rejecting valid 3GPP queries |

**Implementation**:
```python
# tests/test_router.py
import pytest
from specagent.nodes import router_node

@pytest.fixture
def router_test_cases():
    return [
        # In-scope
        {"query": "What is the maximum number of HARQ processes in NR?", "expected": "retrieve"},
        {"query": "Explain RRC connection re-establishment", "expected": "retrieve"},
        # Out-of-scope  
        {"query": "What's a good recipe for pasta?", "expected": "reject"},
        {"query": "Who won the World Cup in 2022?", "expected": "reject"},
    ]

def test_router_accuracy(router_test_cases):
    correct = 0
    for case in router_test_cases:
        result = router_node({"question": case["query"]})
        if result["route"] == case["expected"]:
            correct += 1
    accuracy = correct / len(router_test_cases)
    assert accuracy >= 0.95, f"Router accuracy {accuracy:.2%} below 95% threshold"
```

### 2.2 Retriever Evaluation

**What we're testing**: Does the retriever surface relevant chunks for a given query?

**Dataset**: 100 queries with manually annotated relevant document IDs

**Metrics**:
| Metric | Target | Description |
|--------|--------|-------------|
| Recall@5 | ≥0.75 | Relevant doc appears in top 5 |
| Recall@10 | ≥0.85 | Relevant doc appears in top 10 |
| MRR | ≥0.60 | Mean Reciprocal Rank |
| Context Precision (RAGAS) | ≥0.70 | Proportion of retrieved chunks that are relevant |

**Implementation with RAGAS**:
```python
from ragas.metrics import ContextPrecision, ContextRecall
from ragas import evaluate

def evaluate_retriever(test_dataset):
    """
    test_dataset format:
    {
        "question": str,
        "ground_truth": str,  # Expected answer
        "contexts": list[str],  # Retrieved chunks
    }
    """
    result = evaluate(
        dataset=test_dataset,
        metrics=[ContextPrecision(), ContextRecall()],
    )
    return result
```

### 2.3 Grader Evaluation

**What we're testing**: Does the grader correctly identify relevant vs. irrelevant chunks?

**Dataset**: 500 (query, chunk, relevance_label) triples
- Balanced: 250 relevant, 250 irrelevant

**Metrics**:
| Metric | Target | Rationale |
|--------|--------|-----------|
| Precision | ≥0.85 | Avoid passing irrelevant chunks to generator |
| Recall | ≥0.90 | Don't filter out relevant chunks |
| F1 | ≥0.87 | Balanced measure |

**Critical Failure Mode**: Low precision causes hallucinations; low recall triggers unnecessary rewrites.

### 2.4 Rewriter Evaluation

**What we're testing**: Does rewriting improve retrieval quality?

**Methodology**:
1. Take queries where initial retrieval had low grader scores
2. Apply rewriter
3. Re-run retrieval
4. Measure Δ in grader scores

**Metrics**:
| Metric | Target | Description |
|--------|--------|-------------|
| Improvement Rate | ≥60% | % of cases where rewrite improves retrieval |
| Avg Score Δ | ≥+0.15 | Average improvement in grader confidence |
| Degradation Rate | <10% | % of cases where rewrite makes retrieval worse |

### 2.5 Hallucination Checker Evaluation

**What we're testing**: Does the checker catch ungrounded claims?

**Dataset**: 200 (chunks, generated_answer, is_grounded) triples
- 100 grounded answers (checker should pass)
- 100 ungrounded answers (checker should flag)

**Metrics**:
| Metric | Target | Rationale |
|--------|--------|-----------|
| Precision | ≥0.80 | Don't reject good answers |
| Recall | ≥0.90 | Catch hallucinations — this is the priority |
| F1 | ≥0.85 | |

---

## 3. Offline Evaluation: End-to-End Testing

### 3.1 TSpec-LLM Benchmark

The TSpec-LLM paper provides a 100-question benchmark with difficulty labels. We will use this as our primary accuracy benchmark.

**Baseline Performance** (from paper):
| Model | Without RAG | With Naive RAG |
|-------|-------------|----------------|
| GPT-3.5 | 44% | 71% |
| GPT-4 | 51% | 72% |
| Gemini 1.0 Pro | 46% | 75% |

**Our Target**: ≥85% with Agentic RAG (10+ point improvement over naive RAG)

**Breakdown by Difficulty**:
| Difficulty | Count | Naive RAG Accuracy | Our Target |
|------------|-------|-------------------|------------|
| Easy | 33 | ~85% | ≥95% |
| Medium | 34 | ~75% | ≥85% |
| Hard | 33 | ~65% | ≥75% |

### 3.2 RAGAS End-to-End Metrics

```python
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy, 
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)

def run_e2e_evaluation(test_dataset):
    """
    test_dataset requires:
    - question: str
    - answer: str (generated)
    - contexts: list[str] (retrieved chunks)
    - ground_truth: str (reference answer)
    """
    results = evaluate(
        dataset=test_dataset,
        metrics=[
            Faithfulness(),      # Is answer grounded in context?
            AnswerRelevancy(),   # Does answer address the question?
            AnswerCorrectness(), # Is answer factually correct?
            ContextPrecision(),  # Are retrieved chunks relevant?
            ContextRecall(),     # Did we retrieve all needed info?
        ],
    )
    return results
```

**Targets**:
| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | ≥0.90 | Critical for telecom specs — no made-up information |
| Answer Relevancy | ≥0.85 | Answer addresses the actual question |
| Answer Correctness | ≥0.85 | Answer matches ground truth |
| Context Precision | ≥0.70 | Retrieved chunks are relevant |
| Context Recall | ≥0.80 | All needed info was retrieved |

---

## 4. Offline Evaluation: Trajectory Testing

### 4.1 Why Trajectory Evaluation Matters

Agentic systems can produce correct answers via suboptimal paths:
- **Unnecessary rewrites**: Query was clear, but agent rewrote anyway
- **Excessive retrieval loops**: Agent kept retrying instead of admitting uncertainty
- **Wrong node order**: Agent skipped grading and went straight to generation

Trajectory evaluation ensures the agent's *reasoning process* is sound, not just its output.

### 4.2 Using agentevals for Graph Trajectory

```python
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread

# Define expected trajectories for different query types
EXPECTED_TRAJECTORIES = {
    "simple_factual": ["router", "retriever", "grader", "generator", "hallucination_check"],
    "needs_rewrite": ["router", "retriever", "grader", "rewriter", "retriever", "grader", "generator", "hallucination_check"],
    "out_of_scope": ["router"],  # Should stop at router with rejection
}

def evaluate_trajectory(thread_id: str, query_type: str):
    """Evaluate if agent took optimal path."""
    trajectory = extract_langgraph_trajectory_from_thread(
        checkpointer=memory,
        thread_id=thread_id
    )
    
    expected = EXPECTED_TRAJECTORIES[query_type]
    actual = trajectory["steps"]
    
    # Strict match for simple cases
    if query_type == "out_of_scope":
        return actual == [["router"]]
    
    # For complex cases, use LLM-as-judge
    evaluator = create_graph_trajectory_llm_as_judge(
        model="gpt-4o-mini",
        criteria="""
        Evaluate if the agent took an efficient path:
        1. Did it avoid unnecessary rewrites?
        2. Did it stop when it had sufficient information?
        3. Did it correctly identify when to reject vs. proceed?
        """
    )
    
    return evaluator(outputs=trajectory)
```

### 4.3 Trajectory Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Path Optimality | ≥80% | Agent took shortest valid path |
| Rewrite Necessity | ≥70% | When agent rewrote, it was justified |
| Early Termination | ≥95% | Out-of-scope queries stop at router |
| Loop Avoidance | 100% | No infinite rewrite loops (max 2 enforced) |

---

## 5. Online Evaluation with Arize Phoenix

### 5.1 Instrumentation Setup

```python
# src/specagent/tracing.py
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

def setup_tracing():
    """Initialize Phoenix tracing for the application."""
    tracer_provider = register(
        project_name="3gpp-specagent",
        endpoint="http://localhost:6006/v1/traces",  # Local Phoenix instance
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider
```

### 5.2 Online Metrics Dashboard

Phoenix will track these metrics in real-time:

**Latency Metrics**:
| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| P50 Latency | >2s | Median response time |
| P95 Latency | >4s | 95th percentile |
| P99 Latency | >6s | 99th percentile |
| Node Latency Breakdown | — | Time spent in each node |

**Quality Metrics (LLM-as-Judge)**:
| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| Relevance Score | <0.7 avg | Does answer address query? |
| Groundedness Score | <0.8 avg | Is answer supported by sources? |
| Citation Accuracy | <0.9 avg | Do citations match claims? |

**Operational Metrics**:
| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| Rewrite Rate | >40% | Too many rewrites = poor initial retrieval |
| Rejection Rate | >20% | Too many rejections = router too aggressive |
| Hallucination Flag Rate | >10% | System catching too many hallucinations |
| Token Usage per Query | >8000 | Cost/efficiency monitoring |

### 5.3 Real-Time LLM-as-Judge with Phoenix

```python
from phoenix.evals import OpenAIModel, llm_classify
from phoenix.evals.templates import RAG_RELEVANCY_PROMPT_TEMPLATE

def online_relevance_eval(query: str, response: str, contexts: list[str]):
    """Run real-time relevance evaluation on responses."""
    eval_model = OpenAIModel(model="gpt-4o-mini")
    
    result = llm_classify(
        model=eval_model,
        template=RAG_RELEVANCY_PROMPT_TEMPLATE,
        data={
            "query": query,
            "response": response,
            "context": "\n".join(contexts),
        }
    )
    return result["score"]
```

### 5.4 Drift Detection

Monitor for distribution shifts that indicate degradation:

```python
# Track embedding drift over time
def check_embedding_drift(current_queries: list[str], baseline_embeddings: np.ndarray):
    """Detect if query distribution has shifted from training data."""
    current_embeddings = embed_queries(current_queries)
    
    # Compare centroid distances
    baseline_centroid = baseline_embeddings.mean(axis=0)
    current_centroid = current_embeddings.mean(axis=0)
    
    drift_score = cosine_distance(baseline_centroid, current_centroid)
    
    if drift_score > 0.3:
        alert("Query distribution drift detected", drift_score)
```

---

## 6. Testing Strategy

### 6.1 Test Pyramid

```
                    ┌─────────────────┐
                    │    E2E Tests    │  ← 20 tests, slow, run pre-deploy
                    │   (Full Graph)  │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │   Integration Tests     │  ← 50 tests, medium, run on PR
                │  (Multi-Node Flows)     │
                └────────────┬────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │           Unit Tests                     │  ← 200+ tests, fast, run always
        │  (Individual Nodes, Prompts, Utils)      │
        └──────────────────────────────────────────┘
```

### 6.2 Test Categories

**Unit Tests** (`tests/unit/`):
- Router prompt parsing
- Grader structured output
- Chunk metadata extraction
- Citation formatting

**Integration Tests** (`tests/integration/`):
- Router → Retriever flow
- Grader → Rewriter → Retriever loop
- Generator → Hallucination Checker flow

**E2E Tests** (`tests/e2e/`):
- TSpec-LLM benchmark subset (10 questions)
- Latency regression tests
- Error handling (API failures, empty retrieval)

### 6.3 CI/CD Pipeline

```yaml
# .github/workflows/evaluate.yml
name: Evaluation Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: pytest tests/unit/ -v --tb=short
        
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: pytest tests/integration/ -v --tb=short
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          
  e2e-evaluation:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Run RAGAS evaluation
        run: python scripts/run_ragas_eval.py --dataset tspec-benchmark
      - name: Check accuracy threshold
        run: |
          ACCURACY=$(cat eval_results.json | jq '.accuracy')
          if (( $(echo "$ACCURACY < 0.85" | bc -l) )); then
            echo "Accuracy $ACCURACY below 85% threshold"
            exit 1
          fi
```

---

## 7. Evaluation Dataset Management

### 7.1 Dataset Sources

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| TSpec-LLM Benchmark | Paper authors | 100 Q&A | Primary accuracy benchmark |
| Router Test Set | Manual curation | 200 queries | Router classification |
| Retrieval Gold Set | Manual annotation | 100 Q + relevant docs | Retrieval quality |
| Trajectory Examples | Manual annotation | 50 query trajectories | Path optimality |

### 7.2 Dataset Versioning

```
data/
├── evaluation/
│   ├── v1.0/
│   │   ├── tspec_benchmark.json
│   │   ├── router_test_set.json
│   │   ├── retrieval_gold.json
│   │   └── trajectory_examples.json
│   └── v1.1/
│       └── ...
└── README.md  # Dataset changelog
```

---

## 8. Evaluation Reporting

### 8.1 Automated Report Generation

After each evaluation run, generate a report:

```markdown
# Evaluation Report — 2025-01-15

## Summary
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| E2E Accuracy | 87.2% | ≥85% | ✅ PASS |
| Faithfulness | 0.92 | ≥0.90 | ✅ PASS |
| P95 Latency | 2.8s | <3s | ✅ PASS |
| Rewrite Rate | 22% | <30% | ✅ PASS |

## Component Breakdown
### Router
- Accuracy: 96.5%
- False Positive Rate: 3.2%

### Retriever
- Recall@10: 0.88
- MRR: 0.64

### Grader
- Precision: 0.87
- Recall: 0.91

## Failure Analysis
- 13 incorrect answers (see appendix)
- Common failure mode: Multi-hop questions spanning multiple specs
```

### 8.2 Dashboard Integration

Phoenix provides a built-in dashboard. For custom metrics, export to:
- **Prometheus** → Grafana for infrastructure metrics
- **CSV/JSON** → Jupyter notebooks for deep analysis

---

## 9. Interview Talking Points

This evaluation strategy demonstrates several senior-level competencies:

1. **Multi-layered evaluation thinking**: "I don't just check if the answer is right—I verify each component is working, the agent's reasoning path is optimal, and the system behaves well in production."

2. **Tool selection rationale**: "I chose Phoenix over LangSmith because it's open-source, self-hostable (fitting my k8s constraints), and uses OpenTelemetry for vendor-agnostic tracing."

3. **Metric prioritization**: "For telecom specs, Faithfulness is more important than fluency—a slightly awkward but accurate answer is far better than a confident hallucination."

4. **Trajectory evaluation awareness**: "Agentic systems can get the right answer the wrong way. I evaluate not just *what* the system outputs, but *how* it got there."

5. **Production mindset**: "Offline evals tell you if you *can* ship; online evals tell you if you *should have* shipped. I instrument both."

---

## 10. Implementation Priority

| Phase | Component | Week |
|-------|-----------|------|
| 1 | Unit tests for all nodes | Week 3 |
| 2 | TSpec-LLM benchmark integration | Week 4 |
| 3 | RAGAS E2E evaluation | Week 5 |
| 4 | Phoenix tracing setup | Week 5 |
| 5 | Trajectory evaluation | Week 6 |
| 6 | CI/CD pipeline | Week 6 |
| 7 | Online LLM-as-judge | Week 7 |

---

## References

1. Phoenix Documentation: https://docs.arize.com/phoenix
2. RAGAS Documentation: https://docs.ragas.io
3. agentevals: https://github.com/langchain-ai/agentevals
4. LangGraph Evaluation Guide: https://docs.langchain.com/langsmith/evaluate-graph
5. TSpec-LLM Paper: https://arxiv.org/abs/2406.01768
