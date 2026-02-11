# config/settings.py
# ============================================================
# Centralized Configuration for VLM Classification Pipeline
# ============================================================
# All settings are loaded from environment variables (or .env file).
# Pydantic validates types and provides sensible defaults.
#
# Usage:
#   from config.settings import settings
#   engine = OCREngine(model_name=settings.ocr_model_name)
# ============================================================

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application-wide settings. Values are loaded from environment variables
    or a .env file. Every setting has a typed default so the app can run
    out-of-the-box with zero configuration.
    """

    # Pydantic V2 configuration (replaces deprecated class Config)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OCR Engine ---
    ocr_model_name: str = Field(
        default="PaddleOCR-VL-1.5-0.9B",
        description="Name of the PaddleOCR-VL model to load for inference.",
    )
    ocr_use_vllm: bool = Field(
        default=True,
        description="Enable vLLM backend for faster inference (PagedAttention).",
    )
    ocr_gpu_memory_utilization: float = Field(
        default=0.8,
        description="The fraction of GPU memory to be used for the model weights, activations, and KV cache.",
    )
    ocr_max_model_len: int = Field(
        default=4096,
        description="Maximum sequence length for the model. Reducing this can save GPU memory.",
    )
    ocr_max_num_batched_tokens: int = Field(
        default=32768,
        description="Maximum number of tokens that can be processed in a single batch.",
    )
    ocr_max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate per page.",
    )
    ocr_server_url: Optional[str] = Field(
        default=None,
        description="URL of an existing PaddleOCR-VL server. If set, the engine won't start its own.",
    )
    ocr_auto_start: bool = Field(
        default=True,
        description="Whether to automatically start a local server if one isn't running.",
    )
    ocr_default_mode: str = Field(
        default="document",
        description="Default parsing mode: document | table | formula | seal | text_spotting.",
    )

    # --- Output ---
    output_format: str = Field(
        default="markdown",
        description="Output format for extracted text: markdown | json | text.",
    )

    # --- Performance Tuning ---
    max_batch_size: int = Field(
        default=8,
        description="Maximum number of pages to process concurrently in a batch.",
    )
    max_image_dim: int = Field(
        default=4096,
        description="Maximum image dimension (px). Larger images are resized to prevent OOM.",
    )
    pdf_render_dpi: int = Field(
        default=50,
        description="DPI for rendering PDF pages to images. Higher = better quality, slower.",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR.",
    )





# ============================================================
# Singleton instance — import this everywhere:
#   from config.settings import settings
# ============================================================
settings = Settings()
