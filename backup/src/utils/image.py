# src/utils/image.py
# ============================================================
# Image Utility Functions
# ============================================================
# Shared image helpers used by both the OCR engine and the
# document processor. Handles encoding, resizing, and metadata
# extraction for PIL Images.
#
# Usage:
#   from src.utils.image import resize_if_needed, get_image_info
#   img = resize_if_needed(large_image, max_dim=4096)
#   info = get_image_info(img)
# ============================================================

import base64
import io
from typing import Union

from PIL import Image


import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

def encode_image_base64(image: Union[Image.Image, str], fmt: str = "PNG") -> str:
    """
    Encode a PIL Image or image file path to a base64 string.
    """
    start = time.perf_counter()
    if isinstance(image, str):
        # Load from file path
        image = Image.open(image)

    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    duration = (time.perf_counter() - start) * 1000
    logger.debug(f"Image encoding ({fmt}) took {duration:.2f}ms")
    return encoded


def resize_if_needed(image: Image.Image, max_dim: int = 4096) -> Image.Image:
    """
    Resize an image if its largest dimension exceeds max_dim.
    """
    start = time.perf_counter()
    width, height = image.size

    # No resize needed if both dimensions are within limits
    if width <= max_dim and height <= max_dim:
        return image

    # Calculate the scaling factor based on the larger dimension
    scale = max_dim / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    duration = (time.perf_counter() - start) * 1000
    logger.debug(f"Image resizing ({width}x{height} -> {new_width}x{new_height}) took {duration:.2f}ms")
    return resized


def get_image_info(image: Image.Image) -> dict:
    """
    Extract metadata from a PIL Image for logging and diagnostics.

    Args:
        image: PIL Image to inspect.

    Returns:
        Dictionary with width, height, mode (RGB/RGBA/L), and
        estimated size in MB.

    Example:
        >>> info = get_image_info(Image.new("RGB", (1920, 1080)))
        >>> info["width"]
        1920
    """
    width, height = image.size
    # Estimate uncompressed size: width * height * channels
    channels = len(image.getbands())
    estimated_bytes = width * height * channels
    estimated_mb = round(estimated_bytes / (1024 * 1024), 2)

    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "channels": channels,
        "estimated_size_mb": estimated_mb,
    }
