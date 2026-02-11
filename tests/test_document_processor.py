# tests/test_document_processor.py
# ============================================================
# Unit Tests — Document Processor
# ============================================================
# Tests the DocumentProcessor class for loading PDFs, images,
# and directories. Uses temporary test fixtures.
#
# Run:
#   pytest tests/test_document_processor.py -v
# ============================================================

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.document.processor import (
    DocumentProcessor,
    PageImage,
    SUPPORTED_IMAGE_EXTENSIONS,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def processor():
    """Create a DocumentProcessor with test-friendly settings."""
    return DocumentProcessor(dpi=50, max_dim=2048)


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test PNG image."""
    img = Image.new("RGB", (800, 600), color="white")
    img_path = tmp_path / "test_image.png"
    img.save(str(img_path))
    return img_path


@pytest.fixture
def temp_large_image(tmp_path):
    """Create a large image that should trigger resizing."""
    img = Image.new("RGB", (5000, 4000), color="blue")
    img_path = tmp_path / "large_image.png"
    img.save(str(img_path))
    return img_path


@pytest.fixture
def temp_image_directory(tmp_path):
    """Create a directory with multiple test images."""
    for i in range(3):
        img = Image.new("RGB", (400, 300), color=(i * 80, 100, 200))
        img.save(str(tmp_path / f"page_{i+1:03d}.png"))
    return tmp_path


# ============================================================
# PageImage Tests
# ============================================================

class TestPageImage:
    """Test the PageImage dataclass."""

    def test_creation(self):
        """PageImage should hold image data and metadata."""
        img = Image.new("RGB", (100, 100))
        page = PageImage(image=img, page_num=1, source_path="/test.png", total_pages=5)
        assert page.page_num == 1
        assert page.total_pages == 5
        assert page.image.size == (100, 100)


# ============================================================
# DocumentProcessor Tests — Image Loading
# ============================================================

class TestImageLoading:
    """Test loading individual image files."""

    def test_load_png(self, processor, temp_image):
        """Should load a PNG image as a single-page document."""
        pages = processor.load(temp_image)
        assert len(pages) == 1
        assert pages[0].page_num == 1
        assert pages[0].total_pages == 1
        assert isinstance(pages[0].image, Image.Image)

    def test_load_jpeg(self, processor, tmp_path):
        """Should load a JPEG image."""
        img = Image.new("RGB", (400, 300))
        jpg_path = tmp_path / "test.jpg"
        img.save(str(jpg_path), format="JPEG")

        pages = processor.load(jpg_path)
        assert len(pages) == 1

    def test_load_rgba_converts_to_rgb(self, processor, tmp_path):
        """Should convert RGBA images to RGB."""
        img = Image.new("RGBA", (400, 300), color=(255, 0, 0, 128))
        rgba_path = tmp_path / "test_rgba.png"
        img.save(str(rgba_path))

        pages = processor.load(rgba_path)
        assert pages[0].image.mode == "RGB"

    def test_large_image_gets_resized(self, processor, temp_large_image):
        """Images exceeding max_dim should be resized."""
        pages = processor.load(temp_large_image)
        assert max(pages[0].image.size) <= processor.max_dim


# ============================================================
# DocumentProcessor Tests — Directory Loading
# ============================================================

class TestDirectoryLoading:
    """Test loading directories of images."""

    def test_load_directory(self, processor, temp_image_directory):
        """Should load all images from a directory."""
        pages = processor.load(temp_image_directory)
        assert len(pages) == 3

    def test_pages_are_numbered_sequentially(self, processor, temp_image_directory):
        """Pages from a directory should have sequential numbering."""
        pages = processor.load(temp_image_directory)
        page_nums = [p.page_num for p in pages]
        assert page_nums == [1, 2, 3]

    def test_empty_directory_raises_error(self, processor, tmp_path):
        """Loading an empty directory should raise ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No supported image files"):
            processor.load(empty_dir)


# ============================================================
# DocumentProcessor Tests — Error Handling
# ============================================================

class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_nonexistent_path_raises_error(self, processor):
        """Should raise FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            processor.load("/nonexistent/path/file.png")

    def test_unsupported_format_raises_error(self, processor, tmp_path):
        """Should raise ValueError for unsupported file types."""
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("col1,col2\na,b")
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.load(bad_file)

    def test_supported_extensions_are_defined(self):
        """SUPPORTED_IMAGE_EXTENSIONS should include common formats."""
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".tiff" in SUPPORTED_IMAGE_EXTENSIONS
