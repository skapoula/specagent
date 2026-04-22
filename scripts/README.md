# Benchmark Runner

## Usage

```bash
# Full dataset
python scripts/run_benchmark.py --dataset data/qna/Sampled_3GPP_TR_Questions.json

# Limit questions (quick test)
python scripts/run_benchmark.py --limit 10

# Custom output directory
python scripts/run_benchmark.py --output-dir evaluation/results/2024-01-01
```

## Output

Two files per run:

1. `benchmark_<timestamp>.json` — machine-readable metrics
2. `benchmark_<timestamp>.md` — human-readable report (summary table, accuracy by difficulty, failed questions)

## Dataset Format

TSpec-LLM JSON:

```json
{
  "question_1": {
    "question": "What is the maximum number of HARQ processes for NR?",
    "option_1": "8",
    "option_2": "16",
    "option_3": "32",
    "option_4": "64",
    "answer": "option_2: 16",
    "explanation": "...",
    "category": "3GPP TR 38.321",
    "difficulty": "Easy"
  }
}
```

## Accuracy Checking

Multi-stage: exact match → fuzzy match → word-based match → LLM judge (semantic, enabled by default).

## Performance Targets

- **Accuracy**: 85%+ (baseline naive RAG: 71-75%)
- **Latency**: <3 seconds P95
- **Breakdown**: Easy / Intermediate / Hard difficulty levels
