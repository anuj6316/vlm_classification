# src/document/processor.py
# ============================================================
# Document Processor — PDF & Image Loading
# ============================================================
# Converts input documents (PDFs, images) into a standardized
# list of PIL Image pages that the OCR engine can process.
#
# Supported formats:
#   - PDF: Rendered to images via pdf2image (poppler backend)
#   - Images: PNG, JPG/JPEG, TIFF, BMP, WEBP
#   - Directories: Processes all supported files in a directory
#
# Usage:
#   from src.document.processor import DocumentProcessor
#   processor = DocumentProcessor()
#   pages = processor.load("report.pdf")
#   for page in pages:
#       print(f"Page {page.page_num}: {page.image.size}")
# ============================================================

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from config.settings import settings
from src.utils.image import resize_if_needed, get_image_info
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported image file extensions (case-insensitive matching)
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class PageImage:
    """
    Represents a single page from a document as a PIL Image.

    Attributes:
        image: The PIL Image object for this page.
        page_num: 1-indexed page number within the source document.
        source_path: Original file path this page was extracted from.
        total_pages: Total number of pages in the source document.
    """
    image: Image.Image
    page_num: int
    source_path: str
    total_pages: int


# ============================================================
# Document Processor
# ============================================================

class DocumentProcessor:
    """
    Loads documents (PDFs, images, directories) into processed page images.

    This class handles the first stage of the pipeline: converting raw
    input files into a standardized format that the OCR engine expects.

    Features:
        - Auto-detects input type (PDF, image, or directory)
        - Renders PDF pages at configurable DPI
        - Resizes oversized images to prevent GPU OOM
        - Validates file formats before processing

    Example:
        >>> processor = DocumentProcessor(dpi=300, max_dim=4096)
        >>> pages = processor.load("annual_report.pdf")
        >>> print(f"Loaded {len(pages)} pages")
    """

    def __init__(
        self,
        dpi: Optional[int] = None,
        max_dim: Optional[int] = None,
    ):
        """
        Initialize the document processor.

        Args:
            dpi: DPI for rendering PDF pages. Higher DPI = better quality
                 but larger images and slower processing. Default: from settings.
            max_dim: Maximum image dimension (pixels). Images exceeding this
                     are resized to prevent GPU OOM. Default: from settings.
        """
        self.dpi = dpi or settings.pdf_render_dpi
        self.max_dim = max_dim or settings.max_image_dim

        logger.info(
            f"DocumentProcessor initialized — "
            f"DPI: {self.dpi}, max dimension: {self.max_dim}px"
        )

    def load(self, path: Union[str, Path]) -> list[PageImage]:
        """
        Load a document from a file path (PDF, image, or directory).

        Auto-detects the input type and delegates to the appropriate
        loading method. This is the primary entry point for document loading.

        Args:
            path: Path to a PDF file, image file, or directory of images.

        Returns:
            List of PageImage objects, sorted by page number.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file format is not supported.

        Example:
            >>> pages = processor.load("/app/input/invoice.pdf")
            >>> pages = processor.load("/app/input/scan.png")
            >>> pages = processor.load("/app/input/batch/")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        load_start = time.perf_counter()

        if path.is_dir():
            pages = self._load_directory(path)
        elif path.suffix.lower() == ".pdf":
            pages = self._load_pdf(path)
        elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            pages = self._load_image(path)
        else:
            raise ValueError(
                f"Unsupported file format: '{path.suffix}'. "
                f"Supported: PDF, {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

        load_time = (time.perf_counter() - load_start) * 1000
        logger.info(
            f"Loaded [green]{len(pages)}[/green] pages from "
            f"[bold]{path.name}[/bold] in {load_time:.0f}ms"
        )

        return pages

    def _load_pdf(self, path: Path) -> list[PageImage]:
        """
        Render a PDF file into a list of page images.

        Uses pdf2image (poppler backend) to convert each PDF page
        into a high-quality PIL Image at the configured DPI.

        Args:
            path: Path to the PDF file.

        Returns:
            List of PageImage objects, one per PDF page.
        """
        logger.info(f"Loading PDF: {path.name} (DPI: {self.dpi})")

        try:
            from pdf2image import convert_from_path

            # Convert all pages to PIL Images at the configured DPI
            raw_images = convert_from_path(
                str(path),
                dpi=self.dpi,
                fmt="png",
            )

            pages = []
            for idx, img in enumerate(raw_images, start=1):
                # Resize if the rendered image is too large for GPU
                processed_img = resize_if_needed(img, self.max_dim)

                info = get_image_info(processed_img)
                logger.debug(
                    f"  PDF page {idx}: {info['width']}x{info['height']} "
                    f"({info['estimated_size_mb']}MB)"
                )

                pages.append(PageImage(
                    image=processed_img,
                    page_num=idx,
                    source_path=str(path),
                    total_pages=len(raw_images),
                ))

            return pages

        except ImportError:
            raise RuntimeError(
                "pdf2image is not installed. Install it with: "
                "pip install pdf2image\n"
                "Also ensure poppler-utils is installed: "
                "apt-get install poppler-utils"
            )

    def _load_image(self, path: Path) -> list[PageImage]:
        """
        Load a single image file as a one-page document.

        Args:
            path: Path to the image file.

        Returns:
            List containing a single PageImage.
        """
        logger.info(f"Loading image: {path.name}")

        img = Image.open(str(path))

        # Convert to RGB if necessary (e.g., RGBA, palette mode)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize if too large
        img = resize_if_needed(img, self.max_dim)

        info = get_image_info(img)
        logger.debug(
            f"  Image: {info['width']}x{info['height']} "
            f"({info['estimated_size_mb']}MB)"
        )

        return [PageImage(
            image=img,
            page_num=1,
            source_path=str(path),
            total_pages=1,
        )]

    def _load_directory(self, dir_path: Path) -> list[PageImage]:
        """
        Load all supported images from a directory.

        Files are sorted alphabetically to maintain consistent page ordering.
        Subdirectories are not traversed (flat scan only).

        Args:
            dir_path: Path to the directory containing image files.

        Returns:
            List of PageImage objects from all valid image files.

        Raises:
            ValueError: If no supported image files are found.
        """
        logger.info(f"Scanning directory: {dir_path}")

        # Find all supported image files, sorted alphabetically
        image_files = sorted([
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])

        if not image_files:
            raise ValueError(
                f"No supported image files found in {dir_path}. "
                f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

        logger.info(f"Found {len(image_files)} image files")

        # Load each image as a "page"
        pages = []
        for idx, img_path in enumerate(image_files, start=1):
            img_pages = self._load_image(img_path)
            # Override page numbering for directory context
            for page in img_pages:
                page.page_num = idx
                page.total_pages = len(image_files)
            pages.extend(img_pages)

        return pages
