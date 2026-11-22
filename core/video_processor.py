"""
Production Video Processing Engine.
Handles ffprobe metadata inspection, multi-format validation,
smart aspect-ratio rendering (Original, 9:16 Blurred Background, 9:16 Center Crop),
batch exporting, and zip archiving.
"""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional
import uuid
import zipfile

from core.config import ExportConfig, LayoutMode
from helper.logger import logger


SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
    ".flv", ".m4v", ".ts", ".wmv"
}


class VideoHandler:
    """Handles video file validation, technical metadata extraction, and export rendering."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path).resolve()

    def validate(self) -> bool:
        """Validate that the file exists, is readable, and has a supported video extension."""
        if not (self.video_path.exists() and self.video_path.is_file()):
            logger.error(f"Video file does not exist: {self.video_path}")
            return False

        if self.video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error(
                f"Unsupported format '{self.video_path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return False

        return True

    def info(self) -> Dict[str, any]:
        """
        Extract detailed technical metadata using ffprobe.
        """
        if not self.validate():
            return {}

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name,display_aspect_ratio:format=duration,size,bit_rate",
            "-of", "json",
            str(self.video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as exc:
            logger.error(f"Failed to probe video metadata: {exc}")
            return {}

        streams = data.get("streams", [])
        fmt = data.get("format", {})

        if not streams:
            logger.error("No video stream found in file.")
            return {}

        stream = streams[0]
        
        # Calculate Frame Rate safely
        fps_val = 30.0
        fps_fraction_str = "30/1"
        try:
            fps_frac = Fraction(stream.get("r_frame_rate", "30/1"))
            fps_val = float(fps_frac)
            fps_fraction_str = f"{fps_frac.numerator}/{fps_frac.denominator}"
        except Exception:
            pass

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        duration = float(fmt.get("duration", 0.0))
        file_size_mb = float(fmt.get("size", 0)) / (1024 * 1024)
        bitrate_kbps = float(fmt.get("bit_rate", 0)) / 1000 if fmt.get("bit_rate") else 0

        aspect_ratio_str = stream.get("display_aspect_ratio")
        if not aspect_ratio_str and width and height:
            gcd = _calc_gcd(width, height)
            aspect_ratio_str = f"{width // gcd}:{height // gcd}"

        return {
            "duration": duration,
            "duration_formatted": f"{int(duration // 60):02d}:{int(duration % 60):02d}",
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "aspect_ratio": aspect_ratio_str or "N/A",
            "fps": round(fps_val, 2),
            "fps_fraction": fps_fraction_str,
            "codec": stream.get("codec_name", "unknown"),
            "file_size_mb": round(file_size_mb, 2),
            "bitrate_kbps": round(bitrate_kbps, 1)
        }

    def export_clip(
        self,
        output_path: Path,
        start_seconds: float,
        duration_seconds: float,
        config: Optional[ExportConfig] = None
    ) -> Path:
        """
        Export a specific video slice using FFmpeg with layout options:
        - original: clean cut preserving original aspect ratio
        - vertical_blur: 9:16 Shorts with blurred background
        - vertical_crop: 9:16 Shorts center crop
        """
        config = config or ExportConfig()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-ss", f"{start_seconds:.3f}",
            "-t", f"{duration_seconds:.3f}",
            "-i", str(self.video_path),
        ]

        # Apply aspect ratio filters
        if config.layout == LayoutMode.VERTICAL_BLUR:
            filter_str = (
                "[0:v]split=2[bg][fg];"
                "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                "boxblur=25:2[blurred];"
                "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[scaled];"
                "[blurred][scaled]overlay=(W-w)/2:(H-h)/2[v]"
            )
            cmd.extend(["-filter_complex", filter_str, "-map", "[v]"])
        elif config.layout == LayoutMode.VERTICAL_CROP:
            filter_str = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[v]"
            cmd.extend(["-filter_complex", filter_str, "-map", "[v]"])
        else:
            # ORIGINAL aspect ratio
            cmd.extend(["-map", "0:v:0"])

        # Audio mapping & normalization
        if config.normalize_audio:
            cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

        cmd.extend([
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", config.preset,
            "-crf", str(config.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", config.audio_bitrate,
            "-movflags", "+faststart",
            str(output_path)
        ])

        try:
            logger.info(f"Rendering clip: {output_path.name} ({duration_seconds:.1f}s, layout={config.layout.value})")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("FFmpeg generated an empty or missing output file.")

            return output_path

        except FileNotFoundError:
            raise RuntimeError("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
        except subprocess.CalledProcessError as exc:
            err = exc.stderr[-1000:] if exc.stderr else str(exc)
            raise RuntimeError(f"FFmpeg rendering failed:\n{err}")


def _calc_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def create_clips_zip(file_paths: List[Path], zip_output_path: Path) -> Path:
    """Package a list of exported clip files into a single zip archive."""
    zip_output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            if file_path.is_file():
                zip_file.write(file_path, arcname=file_path.name)
    return zip_output_path
