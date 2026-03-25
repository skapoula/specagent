# Review: run_benchmark.py vs benchmark.py Pattern

## Issues Identified

### 🔴 Critical: Redundant Limit Application

**Location:** `scripts/run_benchmark.py` lines 167-169 and 187

**Current Implementation:**
```python
# In main()
if args.limit:
    questions = questions[:args.limit]  # ❌ Manual slicing
    console.print(f"[yellow]Limited to {len(questions)} questions[/yellow]")

# Later...
report = run_benchmark(
    questions=questions,
    limit=None,  # ❌ Not using the limit parameter
    output_dir=args.output_dir,
)
```

**Issue:**
- The script manually limits the questions list before passing to `run_benchmark()`
- Then passes `limit=None`, bypassing the built-in limit functionality
- This violates the separation of concerns - the script shouldn't duplicate logic that already exists in `run_benchmark()`

**Pattern from benchmark.py:**
```python
def run_benchmark(
    questions: list[BenchmarkQuestion],
    limit: int | None = None,
    output_dir: str | Path = "evaluation/results",
) -> BenchmarkReport:
    # Apply limit if specified
    if limit is not None:
        questions = questions[:limit]
```

**Recommended Fix:**
```python
# In main() - Remove manual slicing
console.print(f"[green]Loaded {len(questions)} questions[/green]")
if args.limit:
    console.print(f"[yellow]Will process first {args.limit} questions[/yellow]")

# Pass limit directly to run_benchmark
report = run_benchmark(
    questions=questions,
    limit=args.limit,  # ✅ Let run_benchmark handle limiting
    output_dir=args.output_dir,
)
```

### 🟡 Minor: Unused Imports

**Location:** Line 22

**Current:**
```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
```

**Issue:** These imports are never used (as acknowledged in comment on line 183-184)

**Recommended Fix:**
Remove unused imports or implement progress tracking

### 🟢 Enhancement Opportunity: Progress Tracking

**Location:** Lines 183-184, 185-189

**Current:**
```python
# Note: run_benchmark doesn't have built-in progress tracking yet
# We could enhance it later with tqdm or rich progress
report = run_benchmark(
    questions=questions,
    limit=None,
    output_dir=args.output_dir,
)
```

**Issue:**
- No feedback during execution (could take minutes for 100 questions)
- User has no visibility into progress
- The imported Progress components aren't used

**Pattern Violation:**
The `run_benchmark()` function in `benchmark.py` is completely silent during execution. For a long-running operation, this is poor UX.

**Recommended Enhancement Options:**

#### Option 1: Add callback parameter to run_benchmark()
```python
# In benchmark.py
def run_benchmark(
    questions: list[BenchmarkQuestion],
    limit: int | None = None,
    output_dir: str | Path = "evaluation/results",
    progress_callback: callable | None = None,  # New parameter
) -> BenchmarkReport:
    # ...
    for i, question in enumerate(questions):
        if progress_callback:
            progress_callback(i, len(questions), question.id)
        # ... process question
```

```python
# In run_benchmark.py
with Progress() as progress:
    task = progress.add_task("[cyan]Processing questions...", total=len(questions))

    def update_progress(current, total, question_id):
        progress.update(task, completed=current + 1)

    report = run_benchmark(
        questions=questions,
        limit=args.limit,
        output_dir=args.output_dir,
        progress_callback=update_progress,
    )
```

#### Option 2: Use rich.status for simpler feedback
```python
# In run_benchmark.py
with console.status("[bold green]Processing questions..."):
    report = run_benchmark(
        questions=questions,
        limit=args.limit,
        output_dir=args.output_dir,
    )
```

#### Option 3: Keep simple, remove unused imports
```python
# Remove the unused Progress imports
# Accept that progress tracking is a future enhancement
```

### 🟢 Code Organization: Error Handling Pattern

**Observation:** Good consistency with benchmark.py

**Current (both files):**
```python
try:
    # Process question
    state = run_query(question.question)
    # ... handle result
except Exception as e:
    # Create error result
    results.append(BenchmarkResult(..., error=str(e)))
```

**Assessment:** ✅ Properly follows the pattern - errors are captured and included in results rather than crashing

## Alignment Analysis

### ✅ What's Correct

1. **Proper delegation**: Script delegates to `run_benchmark()` for core logic
2. **Error handling**: Matches the pattern from benchmark.py
3. **Output formatting**: Nice enhancement over bare benchmark.py output
4. **CLI design**: Clean argparse implementation with good defaults
5. **File validation**: Checks dataset exists before processing
6. **Result display**: Excellent Rich formatting that enhances the base output

### ❌ What Needs Fixing

1. **Limit application**: Remove manual slicing, pass to `run_benchmark()`
2. **Unused imports**: Remove Progress imports if not using them

### 🤔 Design Questions

1. **Progress tracking**: Should this be added to benchmark.py or kept in CLI script?
   - **Recommendation**: Add optional callback to benchmark.py for flexibility

2. **Output verbosity**: Should the script control what benchmark.py saves?
   - **Current**: benchmark.py always saves JSON + MD
   - **Consideration**: Add `--quiet` flag to suppress file output?

## Recommended Changes

### High Priority

```python
# Remove lines 167-169 (manual limit application)
# Change line 187 from:
limit=None,  # Already applied limit above
# To:
limit=args.limit,
```

### Medium Priority

```python
# Remove unused imports (line 22) OR implement progress tracking
# If not implementing progress, remove:
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
```

### Low Priority (Enhancement)

Consider adding to `benchmark.py`:
```python
from typing import Callable

ProgressCallback = Callable[[int, int, str], None]

def run_benchmark(
    questions: list[BenchmarkQuestion],
    limit: int | None = None,
    output_dir: str | Path = "evaluation/results",
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkReport:
```

## Conclusion

**Overall Assessment:** 7/10

The script is well-structured and provides excellent UX enhancements over the base benchmark.py. However, it violates the DRY principle by duplicating the limit logic and has unused imports.

**Critical Fix Required:**
- Remove manual question limiting and pass `args.limit` to `run_benchmark()`

**Recommended Enhancements:**
- Remove unused Progress imports or implement progress tracking
- Consider adding progress callback to benchmark.py for better UX

**Strengths:**
- Excellent error handling and user feedback
- Clean CLI interface
- Beautiful Rich formatting
- Good separation of display logic from core logic
