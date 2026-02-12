# config/__init__.py
# ============================================================
# Configuration package for VLM Classification pipeline.
# Provides centralized, validated settings loaded from .env file.
#
# Usage:
#   from config.settings import settings
#   print(settings.ocr_model_name)
# ============================================================

from config.settings import settings

__all__ = ["settings"]
