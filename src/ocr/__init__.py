# src/ocr/__init__.py
# ============================================================
# OCR Engine Package
# ============================================================
# Provides the core OCR capability using PaddleOCR-VL-1.5
# with vLLM backend for high-speed inference.
#
# Key classes:
#   - OCREngine: Main engine class that wraps PaddleOCRVL
#   - OCRResult: Dataclass holding extraction results
#   - ParseMode: Enum for selecting extraction type
# ============================================================

from src.ocr.engine import OCREngine, OCRResult
from src.ocr.prompts import ParseMode

__all__ = ["OCREngine", "OCRResult", "ParseMode"]
