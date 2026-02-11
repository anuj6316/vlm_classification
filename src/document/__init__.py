# src/document/__init__.py
# ============================================================
# Document Processing Package
# ============================================================
# Handles loading and preprocessing of input documents:
#   - PDFs → rendered page images (via pdf2image/poppler)
#   - Images → validated PIL Image objects
#
# Key classes:
#   - DocumentProcessor: Loads PDFs and images into page images
#   - PageImage: Dataclass holding a single page image + metadata
# ============================================================

from src.document.processor import DocumentProcessor, PageImage

__all__ = ["DocumentProcessor", "PageImage"]
