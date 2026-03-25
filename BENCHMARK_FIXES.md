# Benchmark Script Fixes

## Issues Fixed

Based on the review of `scripts/run_benchmark.py` against the pattern in `src/specagent/evaluation/benchmark.py`, the following issues were identified and fixed:

### 1. ✅ Removed Redundant Limit Application

**Problem:**
The script was manually slicing the questions list before passing to `run_benchmark()`, then passing `limit=None`, which duplicated logic and violated DRY principles.

**Before:**
```python
if args.limit:
    questions = questions[:args.limit]  # ❌ Manual slicing
    console.print(f"[yellow]Limited to {len(questions)} questions[/yellow]")

report = run_benchmark(
    questions=questions,
    limit=None,  # ❌ Not using the parameter
    output_dir=args.output_dir,
)
```

**After:**
```python
if args.limit:
    console.print(f"[yellow]Will process first {args.limit} questions[/yellow]")

report = run_benchmark(
    questions=questions,
    limit=args.limit,  # ✅ Proper delegation
    output_dir=args.output_dir,
)
```

**Benefit:**
- Follows single responsibility principle
- Delegates limiting logic to `run_benchmark()` where it belongs
- Easier to maintain - logic exists in one place only

### 2. ✅ Removed Unused Imports

**Problem:**
Progress tracking components were imported but never used.

**Before:**
```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
```

**After:**
```python
# Removed - not used in current implementation
```

**Benefit:**
- Cleaner code
- No misleading imports
- Faster import time

### 3. ✅ Added Simple Progress Indicator

**Problem:**
No feedback during long-running benchmark execution.

**Before:**
```python
report = run_benchmark(
    questions=questions,
    limit=None,
    output_dir=args.output_dir,
)
```

**After:**
```python
with console.status("[bold green]Processing questions..."):
    report = run_benchmark(
        questions=questions,
        limit=args.limit,
        output_dir=args.output_dir,
    )
```

**Benefit:**
- User sees spinner during execution
- Better UX for long-running operations
- Simple solution using `console.status()` instead of complex Progress bar

## Verification

All tests still pass:
```bash
$ python -m pytest tests/unit/test_benchmark.py -v
============================== 19 passed in 0.35s ==============================
```

Script help still works:
```bash
$ python scripts/run_benchmark.py --help
usage: run_benchmark.py [-h] [--dataset DATASET] [--output-dir OUTPUT_DIR]
                        [--limit LIMIT]
```

## Pattern Alignment

The script now properly follows the pattern from `benchmark.py`:

✅ **Separation of Concerns**: Script handles CLI/UX, `benchmark.py` handles logic
✅ **DRY Principle**: No duplicated limiting logic
✅ **Clean Imports**: Only imports what's actually used
✅ **User Feedback**: Simple spinner during execution
✅ **Error Handling**: Matches benchmark.py error patterns

## Code Quality

**Before Fixes:**
- Redundant logic in two places
- Unused imports
- Silent execution

**After Fixes:**
- Single source of truth for limiting
- Clean imports
- User feedback during execution
- Proper delegation to core module

## Remaining Enhancement Opportunities

For future consideration (not blocking):

1. **Detailed Progress Bar**: Could add callback to `benchmark.py` for per-question progress
   ```python
   # Potential future enhancement
   def run_benchmark(..., progress_callback: Callable | None = None):
       for i, question in enumerate(questions):
           if progress_callback:
               progress_callback(i + 1, total, question.id)
   ```

2. **Quiet Mode**: Add `--quiet` flag to suppress file output
   ```bash
   python scripts/run_benchmark.py --quiet  # Only console output
   ```

3. **JSON-only Mode**: For CI/CD pipelines
   ```bash
   python scripts/run_benchmark.py --json-only  # Machine-readable only
   ```

These are enhancements, not fixes - the current implementation is correct and follows best practices.
