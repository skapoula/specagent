# Confidence Analysis Feature

## Overview

Added comprehensive confidence analysis to the benchmark runner to track and analyze the LLM's confidence levels when answering questions based on retrieved passages.

## What Was Implemented

### 1. Confidence Distribution Analysis

**Function**: `compute_confidence_distribution(results)`

Groups confidence scores into 5 bins to create a histogram:
- `0.0-0.2`: Very low confidence
- `0.2-0.4`: Low confidence
- `0.4-0.6`: Medium confidence
- `0.6-0.8`: High confidence
- `0.8-1.0`: Very high confidence

**Purpose**: Understand the frequency distribution of confidence levels across all questions.

### 2. Confidence Statistics

**Function**: `compute_confidence_stats(results)`

Computes statistical metrics:
- **Mean**: Average confidence across all questions
- **Median**: Middle value of confidence distribution
- **Min**: Lowest confidence score
- **Max**: Highest confidence score
- **Std Dev**: Standard deviation (spread of confidence scores)

**Purpose**: Get a statistical summary of confidence levels.

### 3. Confidence by Correctness

**Function**: `analyze_confidence_by_correctness(results)`

Separates confidence analysis by answer correctness:
- **Correct answers**: Mean confidence and count
- **Incorrect answers**: Mean confidence and count

**Purpose**: Determine if the system is appropriately confident (high confidence for correct answers, low confidence for incorrect ones).

### 4. Enhanced Reporting

#### JSON Report
Now includes:
```json
{
  "confidence_distribution": {
    "0.0-0.2": 2,
    "0.2-0.4": 5,
    "0.4-0.6": 15,
    "0.6-0.8": 30,
    "0.8-1.0": 48
  },
  "confidence_stats": {
    "mean": 0.735,
    "median": 0.780,
    "min": 0.150,
    "max": 0.950,
    "std": 0.185
  }
}
```

#### Markdown Report
Includes new sections:
- **Confidence Statistics** table
- **Confidence Distribution** table with counts and percentages
- Confidence score shown for each failed question

#### Console Output
Displays two new tables:
1. **Confidence Statistics**: Mean, median, min, max, std dev
2. **Confidence Distribution**: Histogram with counts and percentages

## Understanding the Confidence Score

The confidence score represents the **average relevance confidence** from the grader node, which assesses how confident the system is that the retrieved passages are relevant to answering the question.

**High confidence (0.8-1.0)**:
- Retrieved passages are highly relevant
- System is confident in the answer

**Medium confidence (0.4-0.8)**:
- Retrieved passages have moderate relevance
- Answer may be partial or uncertain

**Low confidence (0.0-0.4)**:
- Retrieved passages have low relevance
- System is uncertain about the answer

## Example Output

### Console Display

```
┌────────────────────────────────────────┐
│      Confidence Statistics             │
├──────────────────┬─────────────────────┤
│ Metric           │               Value │
├──────────────────┼─────────────────────┤
│ Max              │               0.950 │
│ Mean             │               0.735 │
│ Median           │               0.780 │
│ Min              │               0.150 │
│ Std Dev          │               0.185 │
└──────────────────┴─────────────────────┘

┌────────────────────────────────────────┐
│      Confidence Distribution           │
├─────────────┬─────────┬────────────────┤
│ Range       │   Count │     Percentage │
├─────────────┼─────────┼────────────────┤
│ 0.0-0.2     │       2 │           2.0% │
│ 0.2-0.4     │       5 │           5.0% │
│ 0.4-0.6     │      15 │          15.0% │
│ 0.6-0.8     │      30 │          30.0% │
│ 0.8-1.0     │      48 │          48.0% │
└─────────────┴─────────┴────────────────┘
```

### Markdown Report Section

```markdown
## Confidence Analysis

### Confidence Statistics

| Metric | Value |
|--------|-------|
| Max | 0.950 |
| Mean | 0.735 |
| Median | 0.780 |
| Min | 0.150 |
| Standard Deviation | 0.185 |

### Confidence Distribution

Frequency of confidence scores assigned to generated answers:

| Confidence Range | Count | Percentage |
|------------------|-------|------------|
| 0.0-0.2 | 2 | 2.0% |
| 0.2-0.4 | 5 | 5.0% |
| 0.4-0.6 | 15 | 15.0% |
| 0.6-0.8 | 30 | 30.0% |
| 0.8-1.0 | 48 | 48.0% |
```

## Usage

The feature is automatically enabled when running benchmarks:

```bash
# Run full benchmark with confidence analysis
python scripts/run_benchmark.py --dataset data/qna/Sampled_3GPP_TR_Questions.json

# Test with limited questions
python scripts/run_benchmark.py --limit 10
```

The confidence analysis appears in:
- Console output (after accuracy tables)
- Markdown report (`evaluation/results/benchmark_*.md`)
- JSON report (`evaluation/results/benchmark_*.json`)

## Interpreting Results

### Ideal Distribution
- **High accuracy + High confidence**: System performs well and knows when it's right
- **Low accuracy + Low confidence**: System struggles but knows it's uncertain

### Warning Signs
- **Low accuracy + High confidence**: System is overconfident in wrong answers
- **High accuracy + Low confidence**: System is underconfident in correct answers

### Example Analysis

If you see:
```
Correct answers:   Mean confidence = 0.85
Incorrect answers: Mean confidence = 0.45
```

This indicates **good calibration** - the system is more confident when correct and less confident when incorrect.

If you see:
```
Correct answers:   Mean confidence = 0.60
Incorrect answers: Mean confidence = 0.75
```

This indicates **poor calibration** - the system is more confident when wrong, which is a problem.

## Testing

All features are fully tested:
- `test_compute_confidence_distribution()`: Tests histogram binning
- `test_compute_confidence_distribution_empty()`: Tests empty input
- `test_confidence_statistics()`: Tests statistical calculations
- `test_confidence_by_correctness()`: Tests correctness analysis
- `test_benchmark_report_includes_confidence_distribution()`: Tests report structure
- `test_benchmark_report_markdown_includes_confidence()`: Tests markdown generation

**Test Results**: 25/25 tests passing ✅

## Technical Details

### Data Flow

1. **Question Processing**: Each question is run through the RAG pipeline
2. **Grader Node**: Assigns confidence scores to retrieved chunks
3. **Average Confidence**: Computed as mean of chunk relevance scores
4. **Result Collection**: Confidence stored in `BenchmarkResult`
5. **Analysis**: Distribution and statistics computed from all results
6. **Reporting**: Metrics included in JSON, Markdown, and console output

### Code Structure

```
src/specagent/evaluation/benchmark.py
├── compute_confidence_distribution()  # Histogram binning
├── compute_confidence_stats()         # Statistical metrics
├── analyze_confidence_by_correctness() # Correctness analysis
├── BenchmarkReport                    # Enhanced with confidence fields
└── run_benchmark()                    # Computes confidence metrics

scripts/run_benchmark.py
└── display_results()                  # Shows confidence tables
```

## Future Enhancements

Potential additions:
1. **Confidence calibration curve**: Plot accuracy vs confidence
2. **Per-difficulty confidence**: Analyze confidence by Easy/Medium/Hard
3. **Confidence threshold tuning**: Find optimal confidence threshold for rejecting answers
4. **Time series**: Track confidence trends across multiple benchmark runs
5. **Correlation analysis**: Confidence vs latency, rewrites, etc.

## Conclusion

The confidence analysis feature provides deep insights into the LLM's certainty when answering questions. This helps:
- Identify when the system is overconfident or underconfident
- Understand the distribution of confidence levels
- Tune the system to reject low-confidence answers
- Validate that confidence correlates with correctness
