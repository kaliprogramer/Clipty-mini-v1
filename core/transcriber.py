"""
Production Whisper Transcription Engine.
Implements lazy model caching, dynamic device selection (CUDA/CPU),
compute precision resolution, and progress reporting.
"""

from threading import Lock
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import ctranslate2
from faster_whisper import WhisperModel

from core.config import TranscriptionConfig
from helper.logger import logger


class WhisperModelCache:
    """
    Thread-safe model cache singleton.
    Avoids loading models at import time and reuses loaded models across requests.
    """
    _instance: Optional["WhisperModelCache"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "WhisperModelCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._models: Dict[Tuple[str, str, str, int], WhisperModel] = {}
            return cls._instance

    @staticmethod
    def resolve_device_and_compute(
        device: str,
        compute_type: str
    ) -> Tuple[str, str]:
        """Automatically resolve 'auto' device and compute_type based on hardware capabilities."""
        resolved_device = device.lower()
        if resolved_device == "auto":
            try:
                has_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                has_cuda = False
            resolved_device = "cuda" if has_cuda else "cpu"

        resolved_compute = compute_type.lower()
        if resolved_compute == "auto":
            if resolved_device == "cuda":
                resolved_compute = "float16"
            else:
                resolved_compute = "int8"

        return resolved_device, resolved_compute

    def get_model(self, config: TranscriptionConfig) -> WhisperModel:
        """Retrieve a cached model instance or load a new one if not yet cached."""
        device, compute_type = self.resolve_device_and_compute(
            config.device,
            config.compute_type
        )
        cache_key = (config.model_size, device, compute_type, config.cpu_threads)

        with self._lock:
            if cache_key not in self._models:
                logger.info(
                    f"Loading Whisper model '{config.model_size}' "
                    f"(device={device}, compute_type={compute_type}, threads={config.cpu_threads})..."
                )
                start_time = perf_counter()
                
                try:
                    loaded_model = WhisperModel(
                        config.model_size,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=config.cpu_threads,
                        num_workers=1,
                    )
                except Exception as e:
                    # Fallback to CPU if CUDA fails (e.g. cuDNN mismatch)
                    if device == "cuda":
                        logger.warning(
                            f"CUDA load failed for Whisper '{config.model_size}' ({e}). Falling back to CPU/int8..."
                        )
                        device, compute_type = "cpu", "int8"
                        cache_key = (config.model_size, device, compute_type, config.cpu_threads)
                        loaded_model = WhisperModel(
                            config.model_size,
                            device=device,
                            compute_type=compute_type,
                            cpu_threads=config.cpu_threads,
                            num_workers=1,
                        )
                    else:
                        raise e

                elapsed = perf_counter() - start_time
                logger.info(f"Whisper model loaded in {elapsed:.2f}s")
                self._models[cache_key] = loaded_model

            return self._models[cache_key]


class WhisperModelHandler:
    """Handles audio transcription for a given media file."""

    def __init__(
        self,
        path: str,
        config: Optional[TranscriptionConfig] = None
    ):
        self.path = path
        self.config = config or TranscriptionConfig()
        self._cache = WhisperModelCache()

    def transcribe(self) -> List[dict]:
        """
        Transcribe the media file and return list of segments with start, end, and text.
        """
        started = perf_counter()
        logger.info(f"Starting transcription for: {self.path} with Whisper '{self.config.model_size}'")

        try:
            model = self._cache.get_model(self.config)

            segments, transcription_info = model.transcribe(
                self.path,
                beam_size=1,
                word_timestamps=False,
                vad_filter=self.config.vad_filter,
                condition_on_previous_text=False,
                language=self.config.language,
            )

            transcript: List[dict] = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue

                transcript.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                })

            elapsed = perf_counter() - started
            detected_lang = getattr(transcription_info, "language", "unknown")
            logger.info(
                f"Transcription completed in {elapsed:.2f}s. "
                f"Detected language: '{detected_lang}', Segments: {len(transcript)}"
            )

            return transcript

        except Exception as exc:
            logger.error(f"Transcription failed: {exc}")
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc
