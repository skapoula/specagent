"""
Command-line interface for SpecAgent.

Commands:
    serve     - Start the FastAPI server
    query     - Run a single query
    index     - Build or update the LanceDB index
    benchmark - Run evaluation benchmark
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="specagent",
    help="Agentic RAG for 3GPP specifications",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[green]Starting SpecAgent server on {host}:{port}[/green]")

    uvicorn.run(
        "specagent.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,  # Memory constraint
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run a single query through the pipeline."""
    from specagent.graph.workflow import run_query

    console.print(f"[blue]Question:[/blue] {question}\n")

    with console.status("[bold green]Processing..."):
        result = run_query(question)

    # Check for rejection
    if result.get("route_decision") == "reject":
        console.print("[yellow]This question is outside 3GPP specifications.[/yellow]")
        if verbose:
            console.print(f"[dim]Reasoning: {result.get('route_reasoning', 'N/A')}[/dim]")
        return

    # Display answer
    console.print("[green]Answer:[/green]")
    console.print(result.get("generation", "No answer generated."))
    console.print()

    # Display citations
    citations = result.get("citations", [])
    if citations:
        console.print("[blue]Citations:[/blue]")
        for c in citations:
            console.print(f"  • {c.raw_citation}")
        console.print()

    # Display metadata
    if verbose:
        table = Table(title="Metadata")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Latency", f"{result.get('processing_time_ms', 0):.0f}ms")
        table.add_row("Chunks Retrieved", str(len(result.get("retrieved_chunks", []))))
        table.add_row("Rewrites", str(result.get("rewrite_count", 0)))
        table.add_row("Hallucination Check", result.get("hallucination_check", "N/A"))
        table.add_row("Confidence", f"{result.get('average_confidence', 0):.2f}")

        console.print(table)


@app.command()
def index(
    docs_dir: str = typer.Option(
        "",
        "--docs-dir",
        "-d",
        help="Directory containing input files (default: DOCS_DIR from config)",
    ),
    library: str = typer.Option(
        "",
        "--library",
        "-l",
        help="Library name to index into (default: DEFAULT_LIBRARY from config)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Delete existing library and re-index all files",
    ),
    max_concurrency: int = typer.Option(
        4,
        "--max-concurrency",
        "-c",
        help="Maximum concurrent files during ingestion (default: 4)",
    ),
) -> None:
    """Build or rebuild the LanceDB index from documents in DOCS_DIR."""
    import asyncio
    from pathlib import Path

    from specagent.config import settings
    from specagent.retrieval.ingestor import ingest_folder
    from specagent.retrieval.resources import get_store

    target_dir = Path(docs_dir) if docs_dir else settings.docs_dir
    target_lib = library if library else settings.default_library

    if not target_dir.exists():
        console.print(f"[red]Docs directory not found: {target_dir}[/red]")
        console.print("Create it and add your .docx (or other) files inside.")
        raise typer.Exit(1)

    if force:
        console.print(f"[yellow]--force: clearing library '{target_lib}'...[/yellow]")
        try:
            store = get_store()
            docs = store.list_documents(library=target_lib, limit=10000, offset=0)
            for doc in docs:
                store.delete_document(doc["doc_id"])
            console.print(f"[green]Cleared {len(docs)} existing documents.[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: could not clear library: {e}[/yellow]")

    console.print(f"[blue]Indexing {target_dir} into library '{target_lib}'...[/blue]")

    result = asyncio.run(
        ingest_folder(
            folder=target_dir,
            library=target_lib,
            metadata=None,
            recursive=True,
            max_concurrency=max_concurrency,
        )
    )

    console.print("[green]Indexing complete![/green]")
    console.print(f"  Total files : {result.total_files}")
    console.print(f"  Indexed     : {result.indexed}")
    console.print(f"  Replaced    : {result.replaced}")
    console.print(f"  Skipped     : {result.skipped}")
    console.print(f"  Failed      : {result.failed}")

    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for err in result.errors:
            console.print(f"  {err['file']}: {err['error']}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    dataset: str = typer.Option(
        "data/evaluation/tspec_benchmark.json",
        help="Path to benchmark dataset",
    ),
    output_dir: str = typer.Option(
        "evaluation/results",
        help="Directory for results",
    ),
    limit: int = typer.Option(None, help="Limit number of questions"),
) -> None:
    """Run evaluation benchmark."""
    from pathlib import Path

    from specagent.evaluation.benchmark import (
        load_benchmark_questions,
    )

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading benchmark from {dataset_path}...[/blue]")
    questions = load_benchmark_questions(dataset_path)

    if limit:
        questions = questions[:limit]
        console.print(f"[yellow]Limited to {limit} questions[/yellow]")

    console.print(f"[blue]Running {len(questions)} questions...[/blue]")

    from specagent.evaluation.benchmark import run_benchmark  # noqa: PLC0415

    report = run_benchmark(
        questions=questions,
        limit=limit,
        output_dir=Path(output_dir),
        skip_health_check=True,
    )

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Questions", str(report.total_questions))
    table.add_row("Correct Answers", str(report.correct_answers))
    table.add_row("Accuracy", f"{report.accuracy:.1%}")
    for difficulty, acc in sorted(report.accuracy_by_difficulty.items()):
        table.add_row(f"  {difficulty}", f"{acc:.1%}")
    console.print(table)


@app.command()
def version() -> None:
    """Show version information."""
    from specagent import __version__

    console.print(f"SpecAgent v{__version__}")


if __name__ == "__main__":
    app()
