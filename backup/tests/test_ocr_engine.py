# tests/test_ocr_engine.py
# ============================================================
# Unit Tests — OCR Engine
# ============================================================
# Tests the OCREngine class behavior including:
#   - OCRResult dataclass structure
#   - Engine initialization with custom config
#   - Graceful error handling when PaddleOCR is not installed
#     (these tests run on host machine without Docker)
#
# To run tests that require the live OCR model:
#   docker compose run --rm ocr pytest tests/ -v -k "integration"
# ============================================================

import pytest
from unittest.mock import patch, MagicMock

from src.ocr.engine import OCREngine, OCRResult
from src.ocr.prompts import ParseMode, get_prompt, list_modes


# ============================================================
# OCRResult Tests
# ============================================================

class TestOCRResult:
    """Test the OCRResult dataclass."""

    def test_default_values(self):
        """OCRResult should have sensible defaults."""
        result = OCRResult()
        assert result.text == ""
        assert result.page_num == 1
        assert result.success is True
        assert result.error is None
        assert result.latency_ms == 0.0

    def test_custom_values(self):
        """OCRResult should accept custom values."""
        result = OCRResult(
            text="Hello World",
            page_num=3,
            source_path="/test/image.png",
            parse_mode="table",
            latency_ms=850.5,
            success=True,
        )
        assert result.text == "Hello World"
        assert result.page_num == 3
        assert result.parse_mode == "table"
        assert result.latency_ms == 850.5

    def test_failed_result(self):
        """OCRResult should capture error information."""
        result = OCRResult(
            success=False,
            error="GPU out of memory",
        )
        assert result.success is False
        assert result.error == "GPU out of memory"
        assert result.text == ""


# ============================================================
# ParseMode & Prompts Tests
# ============================================================

class TestPrompts:
    """Test prompt templates and parse modes."""

    def test_all_modes_have_prompts(self):
        """Every ParseMode should have a corresponding prompt."""
        for mode in ParseMode:
            prompt = get_prompt(mode)
            assert isinstance(prompt, str)
            assert len(prompt) > 10  # Prompts should be meaningful

    def test_document_mode_mentions_markdown(self):
        """Document mode prompt should request Markdown output."""
        prompt = get_prompt(ParseMode.DOCUMENT)
        assert "markdown" in prompt.lower() or "Markdown" in prompt

    def test_table_mode_mentions_table(self):
        """Table mode prompt should mention table extraction."""
        prompt = get_prompt(ParseMode.TABLE)
        assert "table" in prompt.lower()

    def test_formula_mode_mentions_latex(self):
        """Formula mode prompt should mention LaTeX."""
        prompt = get_prompt(ParseMode.FORMULA)
        assert "latex" in prompt.lower() or "LaTeX" in prompt

    def test_list_modes_returns_all(self):
        """list_modes should return all mode name strings."""
        modes = list_modes()
        assert len(modes) == len(ParseMode)
        assert "document" in modes
        assert "table" in modes

    def test_invalid_mode_raises_error(self):
        """get_prompt should raise ValueError for unknown modes."""
        with pytest.raises(ValueError, match="Unknown parse mode"):
            get_prompt("nonexistent_mode")


# ============================================================
# OCREngine Tests (without live model)
# ============================================================

class TestOCREngine:
    """Test OCREngine behavior without requiring a live GPU/model."""

    def test_initialization_with_defaults(self):
        """Engine should initialize with settings defaults."""
        engine = OCREngine()
        assert engine.model_name == "PaddleOCR-VL-1.5-0.9B"
        assert engine.use_vllm is True
        assert engine._is_loaded is False

    def test_initialization_with_custom_values(self):
        """Engine should accept custom model name and vLLM flag."""
        engine = OCREngine(model_name="custom-model", use_vllm=False)
        assert engine.model_name == "custom-model"
        assert engine.use_vllm is False

    def test_model_not_loaded_initially(self):
        """Model should NOT be loaded at initialization (lazy loading)."""
        engine = OCREngine()
        assert engine._pipeline is None
        assert engine._is_loaded is False

    def test_extract_raises_without_paddle(self):
        """Extract should raise RuntimeError when PaddleOCR isn't installed.

        On the host machine (outside Docker), PaddleOCR is not available.
        The engine should raise a clear RuntimeError directing users to Docker.
        """
        engine = OCREngine()
        with pytest.raises(RuntimeError, match="PaddleOCR not found"):
            engine.extract("nonexistent_image.png")

    def test_health_check_structure(self):
        """Health check should return a properly structured dict."""
        engine = OCREngine()
        status = engine.health_check()
        assert "status" in status
        assert "model_name" in status
        assert "vllm_enabled" in status
        assert "gpu_available" in status
        assert "error" in status
