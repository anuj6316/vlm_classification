# src/ocr/prompts.py
# ============================================================
# OCR Prompt Templates for PaddleOCR-VL-1.5
# ============================================================
# PaddleOCR-VL-1.5 uses different prompts to trigger different
# extraction modes. This module centralizes all prompt templates
# so they can be easily modified, extended, or A/B tested.
#
# Usage:
#   from src.ocr.prompts import ParseMode, get_prompt
#   prompt = get_prompt(ParseMode.DOCUMENT)
# ============================================================

from enum import Enum


class ParseMode(str, Enum):
    """
    Extraction modes supported by PaddleOCR-VL-1.5.

    Each mode triggers a different behavior in the model:
    - DOCUMENT: Full page parsing → structured Markdown output
    - TABLE: Focus on table recognition → HTML/Markdown tables
    - FORMULA: Mathematical formula extraction → LaTeX notation
    - SEAL: Stamp/seal text recognition → plain text
    - TEXT_SPOTTING: Text detection with bounding box coordinates
    """
    DOCUMENT = "document"
    TABLE = "table"
    FORMULA = "formula"
    SEAL = "seal"
    TEXT_SPOTTING = "text_spotting"


# ============================================================
# Prompt Templates
# ============================================================
# These prompts are sent alongside the image to PaddleOCR-VL-1.5.
# The model's behavior changes dramatically based on the prompt.
#
# NOTE: These are tuned for PaddleOCR-VL-1.5-0.9B. If the model
# version changes, prompts may need adjustment.
# ============================================================

_PROMPT_TEMPLATES: dict[ParseMode, str] = {
    # Full document parsing — preserves layout, headers, tables, reading order
    # Optimized trigger for PaddleOCR-VL-1.5
    ParseMode.DOCUMENT: "OCR:",

    # Table-focused extraction — converts table images to structured format
    ParseMode.TABLE: "Table OCR:",

    # Mathematical formula extraction — converts to LaTeX
    ParseMode.FORMULA: "Formula OCR:",

    # Seal/stamp recognition — extracts text from circular/rectangular stamps
    # Output format: Plain text content of the seal
    ParseMode.SEAL: (
        "Identify and extract all text content from stamps, seals, or "
        "official markings in this image. Output the text content of "
        "each seal found."
    ),

    # Text spotting — detection + recognition with coordinates
    # Output format: Text with bounding box coordinates
    ParseMode.TEXT_SPOTTING: (
        "Detect and recognize all text in this image. For each text region, "
        "provide the text content along with its bounding box coordinates."
    ),
}


def get_prompt(mode: ParseMode) -> str:
    """
    Get the prompt template for a specific parsing mode.

    Args:
        mode: The ParseMode enum value specifying the extraction type.

    Returns:
        The prompt string to send alongside the image.

    Raises:
        ValueError: If the mode is not recognized.

    Example:
        >>> prompt = get_prompt(ParseMode.DOCUMENT)
        >>> "Markdown" in prompt
        True
    """
    if mode not in _PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown parse mode: {mode}. "
            f"Available modes: {[m.value for m in ParseMode]}"
        )
    return _PROMPT_TEMPLATES[mode]


def list_modes() -> list[str]:
    """
    Get a list of all available parsing mode names.

    Useful for CLI --help text and validation.

    Returns:
        List of mode name strings (e.g., ["document", "table", ...]).
    """
    return [mode.value for mode in ParseMode]
