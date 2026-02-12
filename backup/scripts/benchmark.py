# scripts/benchmark.py
# ============================================================
# Performance Benchmark Script
# ============================================================
# Measures OCR pipeline performance with detailed statistics.
# Designed to run inside the Docker container.
#
# Usage (inside Docker):
#   python scripts/benchmark.py /app/input/ --pages 10
#
# Usage (from host):
#   docker compose run --rm ocr python scripts/benchmark.py /app/input/
# ============================================================

import argparse
import time
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.ocr.engine import OCREngine
from src.document.processor import DocumentProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


def run_benchmark(input_path: str, max_pages: int = 10) -> None:
    """
    Run a detailed performance benchmark on the OCR pipeline.

    Processes images/PDFs and reports:
    - Per-page latency (avg, min, max, median, p95, p99)
    - Throughput (pages/minute)
    - Total processing time
    - Character extraction rate (chars/second)

    Args:
        input_path: Path to image file, PDF, or directory.
        max_pages: Maximum number of pages to benchmark.
    """
    console.print(Panel(
        f"[bold yellow]OCR Performance Benchmark[/bold yellow]\n"
        f"Input: {input_path}\n"
        f"Max pages: {max_pages}",
        title="⚡ Benchmark",
        border_style="yellow",
    ))

    # --- Load document ---
    processor = DocumentProcessor()
    pages = processor.load(input_path)
    pages = pages[:max_pages]

    if not pages:
        console.print("[red]No pages found. Exiting.[/red]")
        return

    console.print(f"\nBenchmarking with [bold]{len(pages)}[/bold] pages...\n")

    # --- Initialize engine (includes model loading time) ---
    load_start = time.perf_counter()
    engine = OCREngine()
    engine._load_model()  # Force model load before timing
    load_time = (time.perf_counter() - load_start) * 1000
    console.print(f"Model loaded in [cyan]{load_time:.0f}ms[/cyan]\n")

    # --- Process pages and collect latencies ---
    latencies = []
    char_counts = []
    errors = 0

    for page in pages:
        result = engine.extract(
            image_input=page.image,
            page_num=page.page_num,
        )
        if result.success:
            latencies.append(result.latency_ms)
            char_counts.append(len(result.text))
        else:
            errors += 1

    # --- Calculate statistics ---
    if not latencies:
        console.print("[red]All extractions failed. No stats to report.[/red]")
        return

    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    sorted_lat = sorted(latencies)
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0]

    throughput = 60_000 / avg_latency  # pages per minute
    total_chars = sum(char_counts)
    avg_chars_per_sec = total_chars / (sum(latencies) / 1000) if latencies else 0

    total_time = sum(latencies)

    # --- Display results ---
    table = Table(title="📊 Benchmark Results", show_header=False, border_style="bright_blue")
    table.add_column("Metric", style="bold", width=25)
    table.add_column("Value", justify="right", width=20)

    # Latency stats
    table.add_row("Pages Processed", f"{len(latencies)}/{len(pages)}")
    table.add_row("Failed Pages", f"[red]{errors}[/red]" if errors else "[green]0[/green]")
    table.add_section()
    table.add_row("Avg Latency", f"[cyan]{avg_latency:.1f}ms[/cyan]")
    table.add_row("Median Latency", f"{median_latency:.1f}ms")
    table.add_row("Min Latency", f"[green]{min_latency:.1f}ms[/green]")
    table.add_row("Max Latency", f"[yellow]{max_latency:.1f}ms[/yellow]")
    table.add_row("Std Dev", f"{stdev_latency:.1f}ms")
    table.add_row("P95 Latency", f"{p95:.1f}ms")
    table.add_row("P99 Latency", f"{p99:.1f}ms")
    table.add_section()
    table.add_row("Throughput", f"[bold green]{throughput:.1f} pages/min[/bold green]")
    table.add_row("Total Characters", f"{total_chars:,}")
    table.add_row("Chars/Second", f"{avg_chars_per_sec:,.0f}")
    table.add_row("Total Time", f"{total_time:.0f}ms")
    table.add_row("Model Load Time", f"{load_time:.0f}ms")

    console.print(table)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Pipeline Benchmark")
    parser.add_argument("input_path", help="Path to images or PDF to benchmark")
    parser.add_argument("--pages", type=int, default=10, help="Max pages to process")
    args = parser.parse_args()

    run_benchmark(args.input_path, args.pages)
