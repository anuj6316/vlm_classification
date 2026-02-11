# src/ocr/engine.py
# ============================================================
# OCR Engine — Managed vLLM Server Integration
# ============================================================
# This module manages a local PaddleOCR vLLM server process
# internally to bypass dependency conflicts in the Baidu image.
# It provides a clean library-like interface while using 
# the stable server-client architecture preferred by the model.
# ============================================================

import time
import subprocess
import socket
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from config.settings import settings
from src.ocr.prompts import ParseMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """Holds the result of a single OCR extraction."""
    text: str = ""
    page_num: int = 1
    source_path: str = ""
    parse_mode: str = "document"
    latency_ms: float = 0.0
    network_latency_ms: float = 0.0
    encoding_latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class OCREngine:
    """
    OCR engine that manages a local vLLM server for PaddleOCR-VL-1.5.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_vllm: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        server_url: Optional[str] = None,
    ):
        self.model_name = model_name or settings.ocr_model_name
        self.use_vllm = use_vllm if use_vllm is not None else settings.ocr_use_vllm
        self.host = host or "127.0.0.1"
        self.port = port or 8100
        
        # If server_url is provided (or in settings), we use it and don't manage the process
        self.server_url = server_url or settings.ocr_server_url
        self._managed_locally = False
        
        self._pipeline = None
        self._server_process = None
        self._is_loaded = False

        logger.info(
            f"OCREngine initialized — model: [bold]{self.model_name}[/bold], "
            f"vLLM: {'enabled' if self.use_vllm else 'disabled'}, "
            f"mode: {'[cyan]Remote[/cyan]' if self.server_url else '[yellow]Local[/yellow]'}"
        )

    def _is_port_open(self) -> bool:
        """Check if the local server port is open."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def _start_server(self) -> None:
        """Launch the PaddleOCR GenAI server as a background process."""
        if self._is_port_open():
            logger.info(f"Server already running on port {self.port}")
            return

        # 1. Generate vLLM backend config if needed
        backend_config_path = None
        if self.use_vllm:
            import json
            config_data = {
                "gpu_memory_utilization": settings.ocr_gpu_memory_utilization,
                "max_model_len": settings.ocr_max_model_len,
                "max_num_batched_tokens": settings.ocr_max_num_batched_tokens,
                "trust_remote_code": True,
                "enable_prefix_caching": False,
                "quantization": "fp8",
                "enforce_eager": True
            }
            config_file = Path("/tmp/vllm_config.json")
            config_file.write_text(json.dumps(config_data))
            backend_config_path = str(config_file)
            logger.info(f"Generated vLLM backend config: {config_data}")

        # 2. Build command
        cmd = [
            "paddleocr", "genai_server",
            "--model_name", self.model_name,
            "--backend", "vllm" if self.use_vllm else "native",
            "--port", str(self.port),
            "--host", self.host
        ]
        
        if backend_config_path:
            cmd.extend(["--backend_config", backend_config_path])
        
        logger.info(f"Starting managed vLLM server: {' '.join(cmd)}")
        
        # We run the server in the background. 
        # stdout/stderr are inherited so we can see Baidu's logs in the console.
        self._server_process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid  # Create a process group to kill children later
        )

    def _wait_for_server(self, timeout: int = 3600) -> bool:
        """Wait for the server to become healthy/responsive (up to 1 hour)."""
        start_time = time.perf_counter()
        logger.info(f"Waiting for server to start on {self.host}:{self.port}...")
        
        while time.perf_counter() - start_time < timeout:
            if self._is_port_open():
                # Short extra sleep to ensure the underlying vLLM engine is fully ready
                time.sleep(2)
                return True
            
            # Check if process died
            if self._server_process and self._server_process.poll() is not None:
                raise RuntimeError("OCR server process exited unexpectedly during startup.")
                
            time.sleep(2)
            
        return False

    def _load_model(self) -> None:
        """Ensure server is running and initialize the internal client."""
        if self._is_loaded:
            return

        try:
            # 1. Determine base URL and manage server if needed
            if self.server_url:
                base_url = self.server_url
                logger.info(f"Connecting to persistent server at {base_url}")
            else:
                base_url = f"http://{self.host}:{self.port}"
                if self.use_vllm and settings.ocr_auto_start:
                    if not self._is_port_open():
                        self._start_server()
                        self._managed_locally = True
                    
                    if not self._wait_for_server():
                        raise TimeoutError("Timed out waiting for vLLM server to start.")

            # 2. Initialize HTTP clients (both sync and async)
            import httpx
            self._client = httpx.Client(
                base_url=base_url,
                timeout=httpx.Timeout(300.0)  # Long timeout for OCR tasks
            )
            self._aclient = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(300.0)
            )
            
            self._is_loaded = True
            logger.info(f"OCREngine fully loaded — Connected to {base_url}")

        except Exception as e:
            logger.error(f"Failed to load OCREngine: {e}")
            if self._server_process:
                self._stop_server()
            raise

    async def aclose(self):
        """Async cleanup."""
        if hasattr(self, '_aclient') and self._aclient:
            await self._aclient.aclose()

    def _stop_server(self) -> None:
        """Terminate the background server process group."""
        if hasattr(self, '_client') and self._client:
            self._client.close()
            
        if self._server_process:
            logger.info("Stopping managed OCR server...")
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=10)
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            self._server_process = None

    async def aextract(
        self,
        image_input: Union[str, Path, Image.Image],
        mode: ParseMode = ParseMode.DOCUMENT,
        page_num: int = 1,
    ) -> OCRResult:
        """Extract text from an image asynchronously."""
        self._load_model()
        
        from src.ocr.prompts import get_prompt
        from src.utils.image import encode_image_base64
        
        source_path = str(image_input) if not isinstance(image_input, Image.Image) else "<PIL.Image>"
        start_time = time.perf_counter()
        encoding_time = 0.0
        network_time = 0.0

        try:
            # 1. Prepare base64 image
            enc_start = time.perf_counter()
            base64_image = encode_image_base64(image_input, fmt="JPEG")
            encoding_time = (time.perf_counter() - enc_start) * 1000
            
            prompt = get_prompt(mode)
            
            # 2. Prepare OpenAI-compatible payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": settings.ocr_max_tokens
            }

            # 3. Send request
            net_start = time.perf_counter()
            response = await self._aclient.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            network_time = (time.perf_counter() - net_start) * 1000
            
            # 4. Parse response
            result_json = response.json()
            extracted_text = result_json["choices"][0]["message"]["content"]

            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return OCRResult(
                text=extracted_text,
                page_num=page_num,
                source_path=source_path,
                parse_mode=mode.value,
                latency_ms=latency_ms,
                network_latency_ms=network_time,
                encoding_latency_ms=encoding_time,
                success=True,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Async extraction failed: {e}")
            return OCRResult(
                text="", page_num=page_num, source_path=source_path,
                latency_ms=latency_ms, success=False, error=str(e)
            )

    def extract(
        self,
        image_input: Union[str, Path, Image.Image],
        mode: ParseMode = ParseMode.DOCUMENT,
        page_num: int = 1,
    ) -> OCRResult:
        """Extract text from an image."""
        self._load_model()
        
        from src.ocr.prompts import get_prompt
        from src.utils.image import encode_image_base64
        
        source_path = str(image_input) if not isinstance(image_input, Image.Image) else "<PIL.Image>"
        start_time = time.perf_counter()

        try:
            # 1. Prepare base64 image
            base64_image = encode_image_base64(image_input, fmt="JPEG")
            prompt = get_prompt(mode)
            
            # 2. Prepare OpenAI-compatible payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": settings.ocr_max_tokens
            }

            # 3. Send request
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            # 4. Parse response
            result_json = response.json()
            extracted_text = result_json["choices"][0]["message"]["content"]

            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return OCRResult(
                text=extracted_text,
                page_num=page_num,
                source_path=source_path,
                parse_mode=mode.value,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Extraction failed: {e}")
            return OCRResult(
                text="", page_num=page_num, source_path=source_path,
                latency_ms=latency_ms, success=False, error=str(e)
            )

    def extract_batch(self, image_inputs: list, mode: ParseMode = ParseMode.DOCUMENT) -> list:
        """Sequential extraction for a batch of images."""
        return [self.extract(img, mode, i) for i, img in enumerate(image_inputs, 1)]

    def health_check(self) -> dict:
        """Diagnostic health check."""
        status = {
            "status": "unhealthy",
            "model_name": self.model_name,
            "vllm_enabled": self.use_vllm,
            "model_loaded": False,
            "gpu_available": False,
            "server_running": self._is_port_open(),
            "error": None,
        }

        try:
            import torch
            status["gpu_available"] = torch.cuda.is_available()
            
            # This triggers server start if not already there
            self._load_model()
            status["status"] = "healthy"
            status["server_running"] = self._is_port_open()
            status["model_loaded"] = self._is_loaded
        except Exception as e:
            status["error"] = str(e)
            
        return status

    def __del__(self):
        """Cleanup on object destruction."""
        self._stop_server()
