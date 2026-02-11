# src/pipeline/__init__.py
# ============================================================
# Pipeline Package
# ============================================================
# Contains the PipelineOrchestrator that ties the OCR engine
# and document processor together into an end-to-end workflow.
#
# Key classes:
#   - PipelineOrchestrator: End-to-end document → OCR → results
#   - DocumentResult: Dataclass holding all pages' OCR results
# ============================================================

from src.pipeline.orchestrator import PipelineOrchestrator, DocumentResult

__all__ = ["PipelineOrchestrator", "DocumentResult"]
