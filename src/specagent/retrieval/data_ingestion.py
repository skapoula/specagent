"""
Data ingestion from HuggingFace datasets.

Downloads 3GPP specification markdown files from the TSpec-LLM dataset
and stores them locally for indexing.
"""

import shutil
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def list_files_in_hf_dataset(
    repo_id: str,
    path: str,
    file_extension: str = ".md",
) -> list[str]:
    """
    List all files in a HuggingFace dataset directory.

    Uses the HuggingFace Hub API to list files in a dataset repository.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., "rasoul-nikbakht/TSpec-LLM")
        path: Path within the repository (e.g., "3GPP-clean/Rel-18/38_series")
        file_extension: File extension to filter (e.g., ".md")

    Returns:
        List of filenames with the specified extension

    Raises:
        httpx.HTTPError: If API request fails
    """
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{path}"

    with httpx.Client(timeout=30.0) as client:
        response = client.get(api_url)
        response.raise_for_status()
        items = response.json()

    # Filter for files (not directories) with the specified extension
    filenames = [
        item["path"].split("/")[-1]
        for item in items
        if item["type"] == "file" and item["path"].endswith(file_extension)
    ]

    return sorted(filenames)


def download_38_series_specs(
    output_dir: Path,
    release: str = "Rel-18",
    api_key: str | None = None,
) -> list[Path]:
    """
    Download 38_series specifications from TSpec-LLM dataset.

    Downloads all markdown files from the specified release's 38_series
    directory on HuggingFace.

    Args:
        output_dir: Directory to save downloaded files
        release: Release version (e.g., "Rel-18")
        api_key: HuggingFace API key (optional, loaded from settings if not provided)

    Returns:
        List of paths to downloaded files

    Raises:
        httpx.HTTPError: If download fails
    """
    from specagent.config import settings

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API key from settings if not provided
    if api_key is None:
        api_key = settings.hf_api_key_value

    # HuggingFace dataset repository
    repo_id = "rasoul-nikbakht/TSpec-LLM"
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    series_path = f"3GPP-clean/{release}/38_series"

    # Get list of all markdown files in the directory
    console.print(f"[cyan]Fetching file list from {repo_id}/{series_path}...[/cyan]")
    try:
        spec_files = list_files_in_hf_dataset(repo_id, series_path, file_extension=".md")
        console.print(f"[green]Found {len(spec_files)} markdown files[/green]\n")
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to list files: {e}[/red]")
        raise

    if not spec_files:
        console.print(f"[yellow]No markdown files found in {series_path}[/yellow]")
        return []

    downloaded_files: list[Path] = []

    # Prepare authentication headers
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Downloading {len(spec_files)} specifications...",
            total=len(spec_files),
        )

        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            for spec_file in spec_files:
                url = f"{base_url}/{series_path}/{spec_file}"
                output_path = output_dir / spec_file

                try:
                    progress.update(
                        task,
                        description=f"[cyan]Downloading {spec_file}...",
                    )

                    response = client.get(url)
                    response.raise_for_status()

                    # Save file
                    output_path.write_text(response.text, encoding="utf-8")
                    downloaded_files.append(output_path)

                    progress.console.print(
                        f"[green]  ✓ {spec_file} ({len(response.text):,} chars)[/green]"
                    )

                except httpx.HTTPError as e:
                    progress.console.print(
                        f"[red]  ✗ Failed to download {spec_file}: {e}[/red]"
                    )

                progress.advance(task)

    console.print(f"\n[green]Downloaded {len(downloaded_files)} files to {output_dir}[/green]")
    return downloaded_files


def discover_markdown_files(data_dir: Path) -> list[Path]:
    """
    Discover all markdown files in a directory and its subdirectories.

    Searches recursively for .md files, which allows it to find specs in:
    - data/raw/*.md (legacy flat structure)
    - data/raw_rel_*/*.md (new multi-release structure)
    - Any subdirectories under data_dir

    Args:
        data_dir: Directory to search

    Returns:
        List of markdown file paths, sorted alphabetically
    """
    md_files = list(data_dir.glob("**/*.md"))
    md_files.sort()
    return md_files


def discover_documents(data_dir: Path) -> list[Path]:
    """
    Discover all supported documents in a directory and its subdirectories.

    Searches recursively for all file formats supported by the converter
    (markdown, PDF, DOCX, PPTX, XLSX, etc.).

    Args:
        data_dir: Directory to search

    Returns:
        List of document file paths, sorted alphabetically
    """
    from specagent.retrieval.converter import ALL_SUPPORTED_EXTENSIONS

    found: list[Path] = []
    for ext in ALL_SUPPORTED_EXTENSIONS:
        found.extend(data_dir.glob(f"**/*{ext}"))
    found.sort()
    return found


def validate_data_directory(data_dir: Path) -> tuple[bool, str]:
    """
    Validate that a data directory contains supported documents.

    Args:
        data_dir: Directory to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not data_dir.exists():
        return False, f"Directory does not exist: {data_dir}"

    if not data_dir.is_dir():
        return False, f"Not a directory: {data_dir}"

    docs = discover_documents(data_dir)
    if not docs:
        return False, f"No supported documents found in: {data_dir}"

    return True, f"Found {len(docs)} documents"


def clear_data_directory(data_dir: Path) -> None:
    """
    Clear all files in a data directory.

    Args:
        data_dir: Directory to clear
    """
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)


def clone_dataset_with_git_lfs(
    output_dir: Path,
    release: str = "Rel-18",
) -> list[Path]:
    """
    Clone TSpec-LLM dataset using git and copy 38_series markdown files.

    Uses git clone with sparse checkout to fetch only the needed files.
    Requires git-lfs to be installed and HuggingFace CLI authentication.

    Args:
        output_dir: Directory to save downloaded files
        release: Release version (e.g., "Rel-18")

    Returns:
        List of paths to downloaded markdown files

    Raises:
        RuntimeError: If git clone or file copy fails
    """
    import subprocess
    import tempfile

    output_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM"
    series_path = f"3GPP-clean/{release}/38_series"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        repo_dir = tmpdir_path / "TSpec-LLM"

        console.print("[cyan]Cloning repository to temporary directory...[/cyan]")
        console.print("[dim]This may take a few minutes...[/dim]\n")

        try:
            # Clone with git-lfs (requires authentication via git credential helper)
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", repo_url, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Path to 38_series directory in cloned repo
            series_dir = repo_dir / series_path

            if not series_dir.exists():
                raise RuntimeError(f"Directory not found in cloned repo: {series_path}")

            # Copy all .md files
            md_files = list(series_dir.glob("*.md"))
            console.print(f"[green]Found {len(md_files)} markdown files[/green]\n")

            downloaded_files: list[Path] = []
            for md_file in md_files:
                dest_path = output_dir / md_file.name
                shutil.copy2(md_file, dest_path)
                downloaded_files.append(dest_path)
                console.print(f"[green]  ✓ {md_file.name} ({md_file.stat().st_size:,} bytes)[/green]")

            console.print(f"\n[green]Copied {len(downloaded_files)} files to {output_dir}[/green]")
            return downloaded_files

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Git clone failed: {e.stderr}[/red]")
            raise RuntimeError(f"Failed to clone repository: {e.stderr}") from e
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise


def download_series_specs(
    output_dir: Path,
    release: str,
    series: str,
    api_key: str | None = None,
) -> list[Path]:
    """
    Download all specifications from a specific series.

    Downloads all markdown files from the specified release and series
    directory on HuggingFace (e.g., Rel-17/36_series).

    Args:
        output_dir: Directory to save downloaded files
        release: Release version (e.g., "Rel-17")
        series: Series name (e.g., "36_series" or "38_series")
        api_key: HuggingFace API key (optional, loaded from settings if not provided)

    Returns:
        List of paths to downloaded files

    Raises:
        httpx.HTTPError: If download fails
    """
    from specagent.config import settings

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API key from settings if not provided
    if api_key is None:
        api_key = settings.hf_api_key_value

    # HuggingFace dataset repository
    repo_id = "rasoul-nikbakht/TSpec-LLM"
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    series_path = f"3GPP-clean/{release}/{series}"

    # Get list of all markdown files in the directory
    console.print(f"[cyan]Fetching file list from {repo_id}/{series_path}...[/cyan]")
    try:
        spec_files = list_files_in_hf_dataset(repo_id, series_path, file_extension=".md")
        console.print(f"[green]Found {len(spec_files)} markdown files[/green]\n")
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to list files: {e}[/red]")
        raise

    if not spec_files:
        console.print(f"[yellow]No markdown files found in {series_path}[/yellow]")
        return []

    downloaded_files: list[Path] = []

    # Prepare authentication headers
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Downloading {len(spec_files)} specifications...",
            total=len(spec_files),
        )

        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            for spec_file in spec_files:
                url = f"{base_url}/{series_path}/{spec_file}"
                output_path = output_dir / spec_file

                try:
                    progress.update(
                        task,
                        description=f"[cyan]Downloading {spec_file}...",
                    )

                    response = client.get(url)
                    response.raise_for_status()

                    # Save file
                    output_path.write_text(response.text, encoding="utf-8")
                    downloaded_files.append(output_path)

                    progress.console.print(
                        f"[green]  ✓ {spec_file} ({len(response.text):,} chars)[/green]"
                    )

                except httpx.HTTPError as e:
                    progress.console.print(
                        f"[red]  ✗ Failed to download {spec_file}: {e}[/red]"
                    )

                progress.advance(task)

    console.print(f"\n[green]Downloaded {len(downloaded_files)} files to {output_dir}[/green]")
    return downloaded_files


def download_specific_files(
    output_dir: Path,
    release: str,
    series: str,
    filenames: list[str],
    api_key: str | None = None,
) -> list[Path]:
    """
    Download specific specification files by name.

    Downloads only the specified markdown files from a release/series directory.

    Args:
        output_dir: Directory to save downloaded files
        release: Release version (e.g., "Rel-15")
        series: Series name (e.g., "36_series" or "38_series")
        filenames: List of specific filenames to download
        api_key: HuggingFace API key (optional, loaded from settings if not provided)

    Returns:
        List of paths to successfully downloaded files

    Raises:
        httpx.HTTPError: If download fails (but continues for other files)
    """
    from specagent.config import settings

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API key from settings if not provided
    if api_key is None:
        api_key = settings.hf_api_key_value

    # HuggingFace dataset repository
    repo_id = "rasoul-nikbakht/TSpec-LLM"
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    series_path = f"3GPP-clean/{release}/{series}"

    downloaded_files: list[Path] = []

    # Prepare authentication headers
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    console.print(f"[cyan]Downloading {len(filenames)} specific files from {series_path}...[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Downloading specific files...",
            total=len(filenames),
        )

        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            for filename in filenames:
                url = f"{base_url}/{series_path}/{filename}"
                output_path = output_dir / filename

                try:
                    progress.update(
                        task,
                        description=f"[cyan]Downloading {filename}...",
                    )

                    response = client.get(url)
                    response.raise_for_status()

                    # Save file
                    output_path.write_text(response.text, encoding="utf-8")
                    downloaded_files.append(output_path)

                    progress.console.print(
                        f"[green]  ✓ {filename} ({len(response.text):,} chars)[/green]"
                    )

                except httpx.HTTPError as e:
                    progress.console.print(
                        f"[red]  ✗ Failed to download {filename}: {e}[/red]"
                    )

                progress.advance(task)

    console.print(f"\n[green]Downloaded {len(downloaded_files)}/{len(filenames)} files to {output_dir}[/green]")
    return downloaded_files


def download_all_required_specs(
    base_dir: Path,
    api_key: str | None = None,
) -> dict[str, list[Path]]:
    """
    Download all required 3GPP specifications as per project requirements.

    Downloads:
    - ALL .md files from Rel-17/36_series → data/raw_rel_17_36_series
    - ALL .md files from Rel-17/38_series → data/raw_rel_17_38_series
    - Specific files from Rel-15/36_series → data/raw_rel_15_36_series
    - Specific files from Rel-15/38_series → data/raw_rel_15_38_series
    - Specific files from Rel-16/38_series → data/raw_rel_16_38_series

    Args:
        base_dir: Base directory for data storage (e.g., Path("data"))
        api_key: HuggingFace API key (optional, loaded from settings if not provided)

    Returns:
        Dictionary mapping category names to lists of downloaded file paths

    Example:
        >>> from pathlib import Path
        >>> results = download_all_required_specs(Path("data"))
        >>> print(f"Downloaded {len(results['rel_17_36_series'])} Rel-17 36-series specs")
    """
    from specagent.config import settings

    # Get API key from settings if not provided
    if api_key is None:
        api_key = settings.hf_api_key_value

    console.print("[bold cyan]Starting download of all required 3GPP specifications[/bold cyan]\n")

    results: dict[str, list[Path]] = {}

    # 1. Download ALL files from Rel-17/36_series
    console.print("[bold]1. Downloading Rel-17/36_series (all files)[/bold]")
    results["rel_17_36_series"] = download_series_specs(
        output_dir=base_dir / "raw_rel_17_36_series",
        release="Rel-17",
        series="36_series",
        api_key=api_key,
    )
    console.print()

    # 2. Download ALL files from Rel-17/38_series
    console.print("[bold]2. Downloading Rel-17/38_series (all files)[/bold]")
    results["rel_17_38_series"] = download_series_specs(
        output_dir=base_dir / "raw_rel_17_38_series",
        release="Rel-17",
        series="38_series",
        api_key=api_key,
    )
    console.print()

    # 3. Download specific files from Rel-15/36_series
    console.print("[bold]3. Downloading Rel-15/36_series (specific files)[/bold]")
    results["rel_15_36_series"] = download_specific_files(
        output_dir=base_dir / "raw_rel_15_36_series",
        release="Rel-15",
        series="36_series",
        filenames=["36777-f00_1.md", "36777-f00_2.md"],
        api_key=api_key,
    )
    console.print()

    # 4. Download specific file from Rel-15/38_series
    console.print("[bold]4. Downloading Rel-15/38_series (specific files)[/bold]")
    results["rel_15_38_series"] = download_specific_files(
        output_dir=base_dir / "raw_rel_15_38_series",
        release="Rel-15",
        series="38_series",
        filenames=["38811-f40.md"],
        api_key=api_key,
    )
    console.print()

    # 5. Download specific files from Rel-16/38_series
    console.print("[bold]5. Downloading Rel-16/38_series (specific files)[/bold]")
    results["rel_16_38_series"] = download_specific_files(
        output_dir=base_dir / "raw_rel_16_38_series",
        release="Rel-16",
        series="38_series",
        filenames=["38821-g20.md", "38901-g10.md"],
        api_key=api_key,
    )
    console.print()

    # Summary
    total_files = sum(len(files) for files in results.values())
    console.print(f"[bold green]✓ Download complete! Total files: {total_files}[/bold green]")
    for category, files in results.items():
        console.print(f"  {category}: {len(files)} files")

    return results
