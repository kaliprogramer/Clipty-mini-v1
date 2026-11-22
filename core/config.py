"""
Configuration models for Clipty-Mini production video clipping system.
Provides strongly-typed settings using Pydantic V2.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class LayoutMode(str, Enum):
    """Aspect ratio export mode for video clips."""
    ORIGINAL = "original"                  # Source aspect ratio (e.g. 16:9 Landscape)
    VERTICAL_BLUR = "vertical_blur"        # 9:16 Shorts/TikTok/Reels with blurred background
    VERTICAL_CROP = "vertical_crop"        # 9:16 Shorts/TikTok/Reels center crop


class TranscriptionConfig(BaseModel):
    """Configuration for Faster-Whisper speech-to-text engine."""
    model_size: str = Field(
        default="tiny",
        description="Whisper model size: tiny, base, small, medium, large-v3, turbo"
    )
    device: str = Field(
        default="auto",
        description="Device to run inference on: auto, cpu, cuda"
    )
    compute_type: str = Field(
        default="auto",
        description="Compute precision: auto, int8, float16, int8_float16"
    )
    cpu_threads: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of CPU threads for inference"
    )
    language: Optional[str] = Field(
        default=None,
        description="Audio language code (e.g. 'en') or None for auto-detect"
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable Silero VAD (Voice Activity Detection) filter"
    )


class ClipSelectionConfig(BaseModel):
    """Configuration for AI candidate generation and LLM moment selection."""
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Name of Ollama model to use for evaluation"
    )
    ollama_host: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama server host URL"
    )
    min_clip_duration: float = Field(
        default=15.0,
        ge=5.0,
        le=300.0,
        description="Minimum clip duration in seconds"
    )
    max_clip_duration: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Maximum clip duration in seconds"
    )
    max_clips: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of top clips to return"
    )
    min_score: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Minimum AI virality score threshold (1-10)"
    )
    max_candidates: int = Field(
        default=12,
        ge=3,
        le=30,
        description="Maximum candidate windows sent to the LLM for evaluation"
    )

    @model_validator(mode="after")
    def validate_durations(self):
        if self.min_clip_duration >= self.max_clip_duration:
            raise ValueError(
                f"min_clip_duration ({self.min_clip_duration}s) must be strictly less "
                f"than max_clip_duration ({self.max_clip_duration}s)."
            )
        return self


class ExportConfig(BaseModel):
    """Configuration for FFmpeg video clip rendering."""
    layout: LayoutMode = Field(
        default=LayoutMode.ORIGINAL,
        description="Video layout: original, vertical_blur (9:16), or vertical_crop (9:16)"
    )
    crf: int = Field(
        default=23,
        ge=16,
        le=32,
        description="Constant Rate Factor for x264 video quality (lower is higher quality)"
    )
    preset: str = Field(
        default="veryfast",
        description="FFmpeg x264 compression preset: ultrafast, veryfast, faster, fast, medium"
    )
    audio_bitrate: str = Field(
        default="128k",
        description="Audio output bitrate"
    )
    normalize_audio: bool = Field(
        default=False,
        description="Apply EBU R128 loudness normalization filter"
    )
