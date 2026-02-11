# src/pipeline/orchestrator.py
# ============================================================
# Pipeline Orchestrator — End-to-End Document Processing
# ============================================================
# Ties the OCR engine and document processor together into a
# complete workflow: file → pages → OCR → structured results.
#
# Design Decisions:
#   1. Post-processor hooks: Empty list now, but SmolLM2 summarizer
#      will plug in here later with zero code changes to this file.
#   2. Sequential processing: vLLM handles internal batching, so
#      we process pages sequentially through the engine.
#   3. Structured output: Returns DocumentResult with per-page text,
#      timing, and aggregate metadata.
#
# Usage:
#   from src.pipeline.orchestrator import PipelineOrchestrator
#   pipeline = PipelineOrchestrator()
#   result = pipeline.process("report.pdf")
#   result.save_markdown("output/report.md")
# ============================================================

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from config.settings import settings
from src.document.processor import DocumentProcessor
from src.ocr.engine import OCREngine, OCRResult
from src.ocr.prompts import ParseMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class DocumentResult:
    """
    Aggregated result of processing an entire document.

    Holds per-page OCR results plus document-level metadata like
    total processing time and success rate.

    Attributes:
        pages: List of OCRResult objects, one per page.
        source_path: Original document file path.
        total_pages: Number of pages processed.
        total_latency_ms: End-to-end processing time in milliseconds.
        pdf_conversion_ms: Time taken to convert PDF to images.
        successful_pages: Number of pages extracted successfully.
    """
    pages: list[OCRResult] = field(default_factory=list)
    source_path: str = ""
    total_pages: int = 0
    total_latency_ms: float = 0.0
    pdf_conversion_ms: float = 0.0
    successful_pages: int = 0

    @property
    def full_text(self) -> str:
        """
        Concatenate all pages' text into a single string.

        Pages are separated by Markdown horizontal rules for clarity.
        Useful for downstream processing or saving as a single file.

        Returns:
            Combined text from all successfully extracted pages.
        """
        texts = []
        for page in self.pages:
            if page.success and page.text:
                texts.append(
                    f"<!-- Page {page.page_num} -->\n{page.text}"
                )
        return "\n\n---\n\n".join(texts)

    def save_markdown(self, output_path: Union[str, Path]) -> str:
        """
        Save the full extracted text as a Markdown file.

        Args:
            output_path: File path for the output .md file.

        Returns:
            The absolute path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the Markdown document with metadata header
        header = (
            f"# OCR Extraction: {Path(self.source_path).name}\n\n"
            f"- **Pages:** {self.successful_pages}/{self.total_pages}\n"
            f"- **Total time:** {self.total_latency_ms:.0f}ms\n"
            f"- **Avg time/page:** {self.total_latency_ms / max(self.total_pages, 1):.0f}ms\n\n"
            f"---\n\n"
        )

        output_path.write_text(header + self.full_text, encoding="utf-8")
        logger.info(f"Saved Markdown to [bold]{output_path}[/bold]")
        return str(output_path.resolve())

    def save_json(self, output_path: Union[str, Path]) -> str:
        """
        Save the extraction results as a structured JSON file.

        Includes per-page metadata (text, latency, success status)
        for programmatic consumption.

        Args:
            output_path: File path for the output .json file.

        Returns:
            The absolute path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "source_path": self.source_path,
            "total_pages": self.total_pages,
            "successful_pages": self.successful_pages,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "pdf_conversion_ms": round(self.pdf_conversion_ms, 2),
            "pages": [
                {
                    "page_num": page.page_num,
                    "text": page.text,
                    "parse_mode": page.parse_mode,
                    "latency_ms": round(page.latency_ms, 2),
                    "network_latency_ms": round(page.network_latency_ms, 2),
                    "encoding_latency_ms": round(page.encoding_latency_ms, 2),
                    "success": page.success,
                    "error": page.error,
                }
                for page in self.pages
            ],
        }

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved JSON to [bold]{output_path}[/bold]")
        return str(output_path.resolve())


# ============================================================
# Pipeline Orchestrator
# ============================================================

class PipelineOrchestrator:
    """
    End-to-end document processing pipeline.

    Coordinates the document processor and OCR engine to convert
    input files into structured text output. Supports extensibility
    through post-processor hooks.

    Flow:
        1. DocumentProcessor.load() → list[PageImage]
        2. OCREngine.extract() for each page → list[OCRResult]
        3. Post-processors (optional) → modified results
        4. Package into DocumentResult

    Extensibility:
        Call add_post_processor() to register functions that run
        after OCR extraction on each page. The SmolLM2 summarizer
        will be added here in a future phase:

            def summarize(result: OCRResult) -> OCRResult:
                result.summary = smolm2.summarize(result.text)
                return result

            pipeline.add_post_processor(summarize)

    Example:
        >>> pipeline = PipelineOrchestrator()
        >>> result = pipeline.process("report.pdf")
        >>> result.save_markdown("output/report.md")
    """

    def __init__(
        self,
        engine: Optional[OCREngine] = None,
        processor: Optional[DocumentProcessor] = None,
    ):
        """
        Initialize the pipeline with an OCR engine and document processor.

        If not provided, default instances are created using settings.

        Args:
            engine: Pre-configured OCREngine instance.
            processor: Pre-configured DocumentProcessor instance.
        """
        self.engine = engine or OCREngine()
        self.processor = processor or DocumentProcessor()

        # Post-processor hook list
        # Functions here are called on each OCRResult after extraction.
        # Signature: (OCRResult) -> OCRResult
        # Future use: SmolLM2 summarizer, text cleaning, etc.
        self.post_processors: list[Callable[[OCRResult], OCRResult]] = []

        logger.info("PipelineOrchestrator initialized")

    def add_post_processor(self, fn: Callable[[OCRResult], OCRResult]) -> None:
        """
        Register a post-processing function.

        Post-processors run on each OCRResult after extraction.
        They receive and return an OCRResult, allowing modification
        of the text, addition of metadata (like summaries), etc.

        Args:
            fn: A callable that takes an OCRResult and returns an OCRResult.

        Example:
            >>> def uppercase(result: OCRResult) -> OCRResult:
            ...     result.text = result.text.upper()
            ...     return result
            >>> pipeline.add_post_processor(uppercase)
        """
        self.post_processors.append(fn)
        logger.info(
            f"Registered post-processor: [bold]{fn.__name__}[/bold] "
            f"(total: {len(self.post_processors)})"
        )

    async def process(
        self,
        file_path: Union[str, Path],
        mode: ParseMode = ParseMode.DOCUMENT,
        output_path: Optional[Union[str, Path]] = True,
    ) -> DocumentResult:
        """
        Process a document end-to-end: load → OCR → post-process → save.

        This is the primary entry point for the pipeline. It handles
        the complete flow from input file to structured output.

        Args:
            file_path: Path to a PDF, image file, or directory of images.
            mode: OCR extraction mode (document, table, formula, etc.).
            output_path: Optional path to save results. Format is determined
                         by settings.output_format (markdown or json).

        Returns:
            DocumentResult with per-page text and aggregate metadata.
        """
        file_path = Path(file_path)
        pipeline_start = time.perf_counter()

        logger.info(
            f"Pipeline starting — file: [bold]{file_path.name}[/bold], "
            f"mode: {mode.value}"
        )

        # --- Step 1: Load document into page images ---
        load_start = time.perf_counter()
        pages = self.processor.load(file_path)
        pdf_conversion_time = (time.perf_counter() - load_start) * 1000
        logger.info(f"Document loaded — {len(pages)} pages")

        # --- Step 2: Run OCR on each page in parallel ---
        import asyncio
        
        # Limit concurrency to avoid GPU OOM or timeouts
        semaphore = asyncio.Semaphore(settings.max_batch_size)
        
        async def _process_page(page):
            async with semaphore:
                result = await self.engine.aextract(
                    image_input=page.image,
                    mode=mode,
                    page_num=page.page_num,
                )
            # Update source path to the original document (not "<PIL.Image>")
            result.source_path = str(file_path)

            # --- Step 3: Run post-processors ---
            for post_fn in self.post_processors:
                try:
                    result = post_fn(result)
                except Exception as e:
                    logger.warning(
                        f"Post-processor '{post_fn.__name__}' failed on "
                        f"page {result.page_num}: {e}"
                    )
            return result

        # Use asyncio.gather to process all pages concurrently
        # vLLM will handle the batching internally
        ocr_results = await asyncio.gather(*[_process_page(p) for p in pages])

        # --- Step 4: Package results ---
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        successful = sum(1 for r in ocr_results if r.success)

        doc_result = DocumentResult(
            pages=ocr_results,
            source_path=str(file_path),
            total_pages=len(pages),
            total_latency_ms=total_latency,
            pdf_conversion_ms=pdf_conversion_time,
            successful_pages=successful,
        )

        logger.info(
            f"Pipeline complete — {successful}/{len(pages)} pages, "
            f"{total_latency:.0f}ms total, "
            f"avg {total_latency / max(len(pages), 1):.0f}ms/page"
        )

        # --- Step 5: Save output if requested ---
        if output_path:
            output_path = Path(output_path)
            if settings.output_format == "json" or output_path.suffix == ".json":
                doc_result.save_json(output_path)
            else:
                doc_result.save_markdown(output_path)

        return doc_result
