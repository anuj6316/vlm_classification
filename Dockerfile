# ============================================================
# VLM Classification — Dockerfile
# ============================================================
# Builds the complete OCR pipeline environment:
#   - Base: Official PaddleOCR vLLM image (includes PaddlePaddle,
#     vLLM, CUDA — all complex deps pre-resolved)
#   - Adds: Our pipeline code + lightweight UI/tooling deps
#
# Usage:
#   docker compose build
#   docker compose run --rm ocr extract /app/input/document.png
# ============================================================

# --- Stage 1: Base image with PaddleOCR + vLLM pre-installed ---
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu AS base

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Ensure Baidu's local bins are in PATH
    PATH="/home/paddleocr/.local/bin:${PATH}" \
    # Optimize layout detection check
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# --- Stage 2: Install additional lightweight Python dependencies ---
# We avoid manual paddle/paddlex installs here to favor the base image versions
RUN pip install --no-cache-dir \
    pdf2image>=1.17.0 \
    Pillow>=10.0.0 \
    python-dotenv>=1.0.0 \
    pydantic-settings>=2.2.0 \
    typer>=0.12.0 \
    rich>=13.7.0

# Ensure root permissions for package installation
USER root

# Install poppler-utils for pdf2image (PDF → image conversion)
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 3: Copy our application code ---
COPY config/ /app/config/
COPY src/ /app/src/
COPY cli/ /app/cli/

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Permissions fix: ensure paddleocr user can write to /app
RUN chown -R paddleocr:paddleocr /app

# Switch to the base image user
USER paddleocr

# --- Entrypoint ---
ENTRYPOINT ["python", "-m", "cli.main"]
CMD ["--help"]
