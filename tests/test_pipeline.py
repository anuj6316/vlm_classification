# tests/test_pipeline.py
# ============================================================
# Unit Tests — Pipeline Orchestrator
# ============================================================
# Tests the PipelineOrchestrator end-to-end flow using mocks
# for the OCR engine (no live model/GPU required).
#
# Integration tests (with live model) should be run inside Docker:
#   docker compose run --rm ocr pytest tests/test_pipeline.py -v -k "integration"
# ============================================================

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.ocr.engine import OCRResult
from src.pipeline.orchestrator import PipelineOrchestrator, DocumentResult


# ============================================================
# DocumentResult Tests
# ============================================================

class TestDocumentResult:
    """Test the DocumentResult dataclass and its methods."""

    def _make_result(self, num_pages=3) -> DocumentResult:
        """Helper: create a DocumentResult with mock page data."""
        pages = [
            OCRResult(
                text=f"Page {i} content with some text.",
                page_num=i,
                source_path="/test/doc.pdf",
                parse_mode="document",
                latency_ms=500.0 + i * 100,
                success=True,
            )
            for i in range(1, num_pages + 1)
        ]
        return DocumentResult(
            pages=pages,
            source_path="/test/doc.pdf",
            total_pages=num_pages,
            total_latency_ms=sum(p.latency_ms for p in pages),
            successful_pages=num_pages,
        )

    def test_full_text_concatenation(self):
        """full_text should join all pages with separators."""
        result = self._make_result(3)
        text = result.full_text
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        assert "Page 3 content" in text
        assert "---" in text  # Page separator

    def test_full_text_skips_failed_pages(self):
        """full_text should exclude pages that failed."""
        result = self._make_result(2)
        result.pages[1].success = False
        result.pages[1].text = ""
        text = result.full_text
        assert "Page 1 content" in text
        assert "Page 2" not in text

    def test_save_markdown(self, tmp_path):
        """save_markdown should write a valid .md file."""
        result = self._make_result(2)
        output_file = tmp_path / "output.md"
        saved_path = result.save_markdown(output_file)

        assert Path(saved_path).exists()
        content = Path(saved_path).read_text()
        assert "OCR Extraction" in content
        assert "Page 1 content" in content
        assert "Pages:" in content

    def test_save_json(self, tmp_path):
        """save_json should write valid, parseable JSON."""
        result = self._make_result(2)
        output_file = tmp_path / "output.json"
        saved_path = result.save_json(output_file)

        assert Path(saved_path).exists()
        data = json.loads(Path(saved_path).read_text())
        assert data["total_pages"] == 2
        assert data["successful_pages"] == 2
        assert len(data["pages"]) == 2
        assert data["pages"][0]["page_num"] == 1

    def test_save_creates_parent_dirs(self, tmp_path):
        """save methods should create parent directories."""
        result = self._make_result(1)
        deep_path = tmp_path / "a" / "b" / "c" / "output.md"
        result.save_markdown(deep_path)
        assert deep_path.exists()


# ============================================================
# PipelineOrchestrator Tests (with mocked engine)
# ============================================================

class TestPipelineOrchestrator:
    """Test the orchestrator's coordination logic."""

    def test_post_processor_registration(self):
        """Should register post-processor functions."""
        pipeline = PipelineOrchestrator()
        assert len(pipeline.post_processors) == 0

        def dummy_processor(result: OCRResult) -> OCRResult:
            return result

        pipeline.add_post_processor(dummy_processor)
        assert len(pipeline.post_processors) == 1

    def test_post_processor_is_called(self):
        """Post-processors should be called on each OCRResult."""
        pipeline = PipelineOrchestrator()

        # Track calls
        call_count = 0

        def counting_processor(result: OCRResult) -> OCRResult:
            nonlocal call_count
            call_count += 1
            result.text = result.text.upper()
            return result

        pipeline.add_post_processor(counting_processor)

        # Mock the engine and processor
        mock_engine = MagicMock()
        mock_engine.extract.return_value = OCRResult(
            text="test text",
            page_num=1,
            success=True,
            latency_ms=100.0,
        )

        mock_processor = MagicMock()
        mock_page = MagicMock()
        mock_page.image = Image.new("RGB", (100, 100))
        mock_page.page_num = 1
        mock_processor.load.return_value = [mock_page]

        pipeline.engine = mock_engine
        pipeline.processor = mock_processor

        result = pipeline.process("/fake/path.png")
        assert call_count == 1
        assert result.pages[0].text == "TEST TEXT"

    def test_post_processor_failure_doesnt_crash(self):
        """A failing post-processor should log warning, not crash."""
        pipeline = PipelineOrchestrator()

        def broken_processor(result: OCRResult) -> OCRResult:
            raise ValueError("Intentional failure")

        pipeline.add_post_processor(broken_processor)

        mock_engine = MagicMock()
        mock_engine.extract.return_value = OCRResult(
            text="original text",
            page_num=1,
            success=True,
            latency_ms=100.0,
        )

        mock_processor = MagicMock()
        mock_page = MagicMock()
        mock_page.image = Image.new("RGB", (100, 100))
        mock_page.page_num = 1
        mock_processor.load.return_value = [mock_page]

        pipeline.engine = mock_engine
        pipeline.processor = mock_processor

        # Should NOT raise — broken post-processor is caught
        result = pipeline.process("/fake/path.png")
        assert result.total_pages == 1
