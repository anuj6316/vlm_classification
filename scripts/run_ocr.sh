#!/bin/bash
# ============================================================
# run_ocr.sh — Host-Side Convenience Wrapper
# ============================================================
# Runs the OCR pipeline inside Docker from the host machine.
# Handles file path mapping to Docker volumes automatically.
#
# Usage:
#   ./scripts/run_ocr.sh path/to/document.pdf
#   ./scripts/run_ocr.sh path/to/image.png --mode table
#   ./scripts/run_ocr.sh path/to/directory/ --output output/
#
# Prerequisites:
#   1. Docker + NVIDIA Container Toolkit installed
#   2. Docker image built: docker compose build
#   3. .env file configured: cp .env.example .env
# ============================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================
# Validation
# ============================================================

# Check that Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check that nvidia-container-toolkit is available
if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA Container Toolkit may not be configured.${NC}"
    echo "GPU support might not work. See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check that at least one argument is provided
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage:${NC} $0 <input-file-or-dir> [--output <path>] [--mode <mode>]"
    echo ""
    echo "Examples:"
    echo "  $0 invoice.png"
    echo "  $0 report.pdf --output output/report.md"
    echo "  $0 scans/ --output output/ --mode document"
    echo ""
    echo "Modes: document, table, formula, seal, text_spotting"
    exit 1
fi

INPUT_PATH="$1"
shift  # Remove input path from args, pass the rest through

# ============================================================
# File Setup
# ============================================================

# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create input/output directories if they don't exist
mkdir -p "${PROJECT_ROOT}/input"
mkdir -p "${PROJECT_ROOT}/output"

# Copy input file to the input directory (Docker volume mount)
if [ -f "$INPUT_PATH" ]; then
    BASENAME=$(basename "$INPUT_PATH")
    cp "$INPUT_PATH" "${PROJECT_ROOT}/input/${BASENAME}"
    DOCKER_INPUT="/app/input/${BASENAME}"
    echo -e "${GREEN}Copied ${BASENAME} to input directory${NC}"
elif [ -d "$INPUT_PATH" ]; then
    # Copy all files from directory
    cp -r "$INPUT_PATH"/* "${PROJECT_ROOT}/input/" 2>/dev/null || true
    DOCKER_INPUT="/app/input/"
    echo -e "${GREEN}Copied directory contents to input/${NC}"
else
    echo -e "${RED}Error: Input path not found: ${INPUT_PATH}${NC}"
    exit 1
fi

# ============================================================
# Run OCR Pipeline
# ============================================================

echo -e "${GREEN}Starting OCR pipeline...${NC}"
echo ""

cd "$PROJECT_ROOT"
docker compose run --rm ocr extract "$DOCKER_INPUT" "$@"

echo ""
echo -e "${GREEN}Done! Check the output/ directory for results.${NC}"
