# cli/main.py
# ============================================================
# VLM Classification — Command Line Interface
# ============================================================
# Typer-based CLI that runs inside the Docker container.
# Provides commands for OCR extraction, health checks, and
# performance benchmarking.
#
# Usage (from host machine):
#   docker compose run --rm ocr extract /app/input/document.png
#   docker compose run --rm ocr extract /app/input/report.pdf --output /app/output/
#   docker compose run --rm ocr health
#   docker compose run --rm ocr benchmark /app/input/ --pages 10
#
# Usage (inside Docker container):
#   python -m cli.main extract /app/input/document.png
#   python -m cli.main health
# ============================================================

from pathlib import Path
from typing import Optional

import typer
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import settings
from src.ocr.engine import OCREngine
from src.ocr.prompts import ParseMode, list_modes
from src.document.processor import DocumentProcessor
from src.pipeline.orchestrator import PipelineOrchestrator

# ============================================================
# CLI App Setup
# ============================================================

app = typer.Typer(
    name="vlm",
    help=(
        "🔍 VLM Classification — Fast Document OCR Pipeline\n\n"
        "Extract text from documents using PaddleOCR-VL-1.5 with vLLM backend.\n"
        "Supports PDFs, images, and batch directory processing."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# ============================================================
# Commands
# ============================================================

@app.command()
def extract(
    input_path: str = typer.Argument(
        ...,
        help="Path to input file (PDF, image) or directory of images.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (.md or .json). If directory, auto-names the output file.",
    ),
    mode: str = typer.Option(
        "document",
        "--mode", "-m",
        help=f"Extraction mode: {', '.join(list_modes())}.",
    ),
):
    """
    📄 Extract text from a document using PaddleOCR-VL-1.5.

    Processes PDFs (all pages), individual images, or entire directories.
    Output is saved as Markdown or JSON depending on the file extension.

    Examples:
        extract /app/input/invoice.png
        extract /app/input/report.pdf --output /app/output/report.md
        extract /app/input/ --output /app/output/ --mode table
    """
    # Validate the parse mode
    try:
        parse_mode = ParseMode(mode)
    except ValueError:
        console.print(
            f"[red]Error:[/red] Unknown mode '{mode}'. "
            f"Available modes: {', '.join(list_modes())}",
        )
        raise typer.Exit(code=1)

    # Validate input path
    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input path not found: {input_path}")
        raise typer.Exit(code=1)

    # Determine output path
    output_path = None
    if output:
        output_path = Path(output)
        # If output is a directory, auto-generate filename
        if output_path.is_dir() or str(output).endswith("/"):
            output_path.mkdir(parents=True, exist_ok=True)
            suffix = ".json" if settings.output_format == "json" else ".md"
            output_path = output_path / f"{input_file.stem}_ocr{suffix}"

    # Show a nice header
    console.print(Panel(
        f"[bold blue]PaddleOCR-VL-1.5[/bold blue] — Document Extraction\n"
        f"Input:  {input_path}\n"
        f"Mode:   {parse_mode.value}\n"
        f"Output: {output_path or 'stdout'}",
        title="🔍 VLM OCR",
        border_style="blue",
    ))

    # Initialize and run the pipeline
    pipeline = PipelineOrchestrator()
    
    async def run_pipeline():
        try:
            return await pipeline.process(
                file_path=input_path,
                mode=parse_mode,
                output_path=output_path,
            )
        finally:
            await pipeline.engine.aclose()

    result = asyncio.run(run_pipeline())

    # Display results summary
    _print_results_table(result)

    # If no output file specified, print the extracted text to stdout
    if not output_path:
        console.print("\n[bold]Extracted Text:[/bold]\n")
        console.print(result.full_text)

    # Exit with error code if any pages failed
    if result.successful_pages < result.total_pages:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to."),
    port: int = typer.Option(8100, help="Port to bind the server to."),
):
    """
    🚀 Start the OCR engine as a persistent background server.

    Keeps the model loaded in GPU memory for fast subsequent requests.
    """
    console.print(Panel(
        f"[bold green]Starting Persistent OCR Server[/bold green]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Model: {settings.ocr_model_name}",
        title="🚀 VLM Server",
        border_style="green",
    ))

    # Initialize engine (this starts the background vLLM process)
    engine = OCREngine(host=host, port=port)
    
    try:
        # Trigger model loading
        status = engine.health_check()
        if status["status"] == "healthy":
            console.print("\n[bold green]Server is ready![/bold green]")
            console.print("Press [bold]Ctrl+C[/bold] to stop.\n")
            
            # Keep the main thread alive while the background server runs
            while True:
                import time
                time.sleep(1)
        else:
            console.print(f"\n[red]Server failed to start healthy:[/red] {status['error']}")
            raise typer.Exit(code=1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
    finally:
        # engine.__del__ will be called or we can explicitly stop
        engine._stop_server()


@app.command()
def health():
    """
    🏥 Check the OCR engine health status.

    Verifies that the model is loaded, GPU is accessible,
    and inference is operational.
    """
    console.print("[bold]Running health check...[/bold]\n")

    engine = OCREngine()
    status = engine.health_check()

    # Build a summary table
    table = Table(title="OCR Engine Health", show_header=False)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    status_color = "green" if status["status"] == "healthy" else "red"
    table.add_row("Status", f"[{status_color}]{status['status']}[/{status_color}]")
    table.add_row("Model", status["model_name"])
    table.add_row("vLLM Backend", "✅ Enabled" if status["vllm_enabled"] else "❌ Disabled")
    table.add_row("GPU Available", "✅ Yes" if status["gpu_available"] else "❌ No")
    table.add_row("Model Loaded", "✅ Yes" if status["model_loaded"] else "❌ No")

    if status["error"]:
        table.add_row("Error", f"[red]{status['error']}[/red]")

    console.print(table)

    if status["status"] != "healthy":
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    input_path: str = typer.Argument(
        ...,
        help="Path to directory of images or a PDF to benchmark.",
    ),
    pages: int = typer.Option(
        10,
        "--pages", "-n",
        help="Number of pages to process for the benchmark.",
    ),
):
    """
    ⚡ Run a performance benchmark on the OCR pipeline.

    Processes N pages and reports latency statistics including
    average, min, max, and throughput (pages/minute).
    """
    console.print(Panel(
        f"[bold yellow]Performance Benchmark[/bold yellow]\n"
        f"Input:  {input_path}\n"
        f"Pages:  {pages}",
        title="⚡ Benchmark",
        border_style="yellow",
    ))

    # Run the pipeline
    pipeline = PipelineOrchestrator()
    
    async def run_pipeline():
        try:
            return await pipeline.process(file_path=input_path)
        finally:
            await pipeline.engine.aclose()

    result = asyncio.run(run_pipeline())

    # Limit to requested number of pages
    page_results = result.pages[:pages]

    if not page_results:
        console.print("[red]No pages processed. Check input path.[/red]")
        raise typer.Exit(code=1)

    # Calculate statistics
    latencies = [p.latency_ms for p in page_results if p.success]
    if not latencies:
        console.print("[red]All pages failed. No latency data.[/red]")
        raise typer.Exit(code=1)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    throughput_ppm = 60_000 / avg_latency  # pages per minute

    # Sort for percentile calculation
    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

    # Display results
    table = Table(title="Benchmark Results", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Pages Processed", f"{len(latencies)}/{len(page_results)}")
    table.add_row("Avg Latency", f"[cyan]{avg_latency:.0f}ms[/cyan]")
    table.add_row("Min Latency", f"[green]{min_latency:.0f}ms[/green]")
    table.add_row("Max Latency", f"[yellow]{max_latency:.0f}ms[/yellow]")
    table.add_row("P95 Latency", f"{p95_latency:.0f}ms")
    table.add_row("Throughput", f"[bold green]{throughput_ppm:.1f} pages/min[/bold green]")
    table.add_row("Total Time", f"{result.total_latency_ms:.0f}ms")

    console.print(table)


# ============================================================
# Helper Functions
# ============================================================

def _print_results_table(result) -> None:
    """Print a summary table of processing results."""
    table = Table(title="Processing Summary")
    table.add_column("Page", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Characters", justify="right")
    table.add_column("Latency (Net/Total)", justify="right")

    for page in result.pages:
        status = "[green]✅[/green]" if page.success else "[red]❌[/red]"
        chars = str(len(page.text)) if page.success else "-"
        latency = f"{page.network_latency_ms:.0f}/{page.latency_ms:.0f}ms"
        table.add_row(str(page.page_num), status, chars, latency)

    # Summary row
    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{result.successful_pages}/{result.total_pages}[/bold]",
        f"[bold]{sum(len(p.text) for p in result.pages)}[/bold]",
        f"[bold]PDF: {result.pdf_conversion_ms:.0f}ms | Total: {result.total_latency_ms:.0f}ms[/bold]",
    )

    console.print(table)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    app()
