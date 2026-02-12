# src/utils/__init__.py
# ============================================================
# Shared Utilities Package
# ============================================================
# Provides reusable helpers used across the pipeline:
#   - logger: Structured logging with Rich formatting
#   - image: Image encoding, resizing, metadata extraction
# ============================================================

from src.utils.logger import get_logger
from src.utils.image import encode_image_base64, resize_if_needed, get_image_info

__all__ = [
    "get_logger",
    "encode_image_base64",
    "resize_if_needed",
    "get_image_info",
]
